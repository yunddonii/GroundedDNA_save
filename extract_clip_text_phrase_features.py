"""v113 Method 4: phrase-level CLIP text feature extraction.

The default `extract_clip_text_features.py` pools each per-slot SENTENCE
through the CLIP text encoder (one pooled embedding per slot). When the
sentence contains multiple concepts ("red fabric, glossy metal, rough
wooden textures") the pooled embedding dilutes them. This script:
    1. Splits each per-slot sentence into PHRASES via comma/semicolon
       and the lightweight conjunction tokens (' and ', ' or ').
    2. Encodes EACH phrase through CLIP independently.
    3. Aggregates the phrase embeddings into a single per-slot vector
       via either mean-pool or self-attention pool over phrases.

Output schema matches `extract_clip_text_features.py`:
    text_part.f16.npy   [N, M=6, D_proj=512]
    has_text.bool.npy   [N]
The other cache files (image_ids, visual_*, meta) are SYMLINKED from
the donor directory so the cache is drop-in compatible with the
existing training loop.

Usage:
    python extract_clip_text_phrase_features.py \\
        --qwen_cache  cache/flickr25k_qwen_v4.jsonl \\
        --donor_dir   cache/flickr25k_clip_v4plus \\
        --out_dir     cache/flickr25k_clip_v4plus_phrase \\
        --aggregation mean
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from models.pretrained_backbone import coerce_pooled_to_tensor
from models.pretrained_backbone_clip import (
    CLIPBackbone,
    DEFAULT_CLIP_BACKBONE,
)
from dna_utils import (
    build_clip_text_tokenizer,
    DEFAULT_CLIP_TOKENIZER_NAME,
)


SLOT_KEYS_V3V4 = (
    "C_global",
    "C_primary_object",
    "C_secondary_object",
    "C_activity_or_relation",
    "C_color_texture",
    "C_scene_type",
)


# Lightweight phrase splitter. Order matters: commas/semicolons first,
# then conjunctions. We deliberately do NOT split on periods — captions
# are single-sentence by construction.
_PHRASE_SPLIT_RE = re.compile(r"[,;]| and | or ", flags=re.IGNORECASE)


def split_into_phrases(sentence: str, max_phrases: int = 8) -> List[str]:
    """Return a clipped list of cleaned phrases for one sentence."""
    s = (sentence or "").strip()
    if not s:
        return []
    parts = [p.strip(" .") for p in _PHRASE_SPLIT_RE.split(s)]
    parts = [p for p in parts if p and len(p) >= 2]
    if not parts:
        return [s]
    return parts[:max_phrases]


def load_qwen_jsonl(path: str) -> dict:
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            iid = row.get("image_id")
            cb  = row.get("codebook_texts", {}) or {}
            sents = [str(cb.get(k, "") or "") for k in SLOT_KEYS_V3V4]
            sents_clean = [
                s if s.strip().lower() not in ("", "none") else ""
                for s in sents
            ]
            if any(s for s in sents_clean):
                out[iid] = sents_clean
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen_cache", required=True)
    ap.add_argument("--donor_dir",  required=True,
                    help="Cache dir to symlink image_ids / visual_* / "
                         "has_text / meta from.")
    ap.add_argument("--out_dir",    required=True)
    ap.add_argument("--clip_backbone", default=DEFAULT_CLIP_BACKBONE)
    ap.add_argument("--tokenizer",     default=DEFAULT_CLIP_TOKENIZER_NAME)
    ap.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=256,
                    help="Phrase-unit batch size (not images).")
    ap.add_argument("--max_length", type=int, default=32,
                    help="Per-phrase max token length. Phrases are short.")
    ap.add_argument("--max_phrases_per_slot", type=int, default=8)
    ap.add_argument("--aggregation", type=str, default="mean",
                    choices=["mean", "attn"],
                    help="How to pool phrase embeddings per slot.")
    args = ap.parse_args()

    print(f"[clip-text-phrase] loading {args.clip_backbone} on {args.device} ...")
    backbone = CLIPBackbone(args.clip_backbone).to(args.device).eval()
    for p in backbone.parameters():
        p.requires_grad = False
    tok = build_clip_text_tokenizer(args.tokenizer)
    D = backbone.projection_dim
    print(f"[clip-text-phrase] text projection dim = {D}")

    donor_ids_p = os.path.join(args.donor_dir, "image_ids.json")
    if not os.path.exists(donor_ids_p):
        raise SystemExit(f"[clip-text-phrase] missing donor image_ids.json: {donor_ids_p}")
    image_ids: List[str] = json.load(open(donor_ids_p))
    N = len(image_ids)
    M = len(SLOT_KEYS_V3V4)
    print(f"[clip-text-phrase] donor image_ids: N={N}")

    print(f"[clip-text-phrase] loading qwen cache {args.qwen_cache} ...")
    qwen = load_qwen_jsonl(args.qwen_cache)
    coverage = sum(1 for iid in image_ids if iid in qwen)
    print(f"[clip-text-phrase] qwen coverage: {coverage}/{N}")

    # Flatten every (image, slot) -> list of phrases.
    # We assemble a global phrase buffer and remember the (i, m, start, end)
    # so we can pool back per slot after the GPU pass.
    all_phrases: List[str] = []
    spans: List[List[List[int]]] = [[[0, 0] for _ in range(M)] for _ in range(N)]
    n_phrase_total = 0
    n_phrase_empty = 0
    for i, iid in enumerate(image_ids):
        sents = qwen.get(iid) or ["" for _ in range(M)]
        for m in range(M):
            phrases = split_into_phrases(sents[m], max_phrases=args.max_phrases_per_slot)
            if not phrases:
                spans[i][m] = [n_phrase_total, n_phrase_total]
                n_phrase_empty += 1
                continue
            spans[i][m] = [n_phrase_total, n_phrase_total + len(phrases)]
            all_phrases.extend(phrases)
            n_phrase_total += len(phrases)
    print(f"[clip-text-phrase] total phrases = {n_phrase_total}  empty slots = {n_phrase_empty}")
    avg_phrases = (n_phrase_total / max(N * M - n_phrase_empty, 1))
    print(f"[clip-text-phrase] average phrases per non-empty slot = {avg_phrases:.2f}")

    # Encode all phrases.
    phrase_feats = np.zeros((n_phrase_total, D), dtype=np.float32)
    bs = args.batch_size
    t0 = time.time()
    with torch.no_grad():
        for i in tqdm(range(0, n_phrase_total, bs), desc="clip-text-phrase"):
            chunk = all_phrases[i:i+bs]
            enc = tok(
                chunk, padding="max_length", truncation=True,
                max_length=args.max_length, return_tensors="pt",
                return_attention_mask=True,
            ).to(args.device)
            input_ids = enc["input_ids"]
            attn = enc.get("attention_mask")
            if attn is None:
                pad_id = getattr(tok, "pad_token_id", None)
                attn = (input_ids != pad_id).long() if pad_id is not None else torch.ones_like(input_ids)
            t_feat = backbone.model.get_text_features(
                input_ids=input_ids, attention_mask=attn,
            )
            t_feat = coerce_pooled_to_tensor(t_feat)
            phrase_feats[i:i+t_feat.shape[0]] = t_feat.float().cpu().numpy()
    print(f"[clip-text-phrase] encode done in {time.time()-t0:.1f}s")

    # Aggregate per (i, m).
    feats = np.zeros((N, M, D), dtype=np.float32)
    if args.aggregation == "mean":
        for i in range(N):
            for m in range(M):
                s, e = spans[i][m]
                if e > s:
                    feats[i, m] = phrase_feats[s:e].mean(axis=0)
                # else leave zero (consistent with has_text=False rows)
    elif args.aggregation == "attn":
        # Lightweight attention pool: weight phrases by softmax over their
        # cosine similarity to the slot-mean. This emphasizes the dominant
        # concept while keeping the others present.
        for i in range(N):
            for m in range(M):
                s, e = spans[i][m]
                if e <= s:
                    continue
                block = phrase_feats[s:e]                         # [P, D]
                if block.shape[0] == 1:
                    feats[i, m] = block[0]
                    continue
                mu = block.mean(axis=0, keepdims=True)
                bn = block / (np.linalg.norm(block, axis=1, keepdims=True) + 1e-8)
                mn = mu    / (np.linalg.norm(mu,    axis=1, keepdims=True) + 1e-8)
                sim = (bn @ mn.T).squeeze(-1)                     # [P]
                w = np.exp(sim - sim.max())
                w = w / w.sum()
                feats[i, m] = (block * w[:, None]).sum(axis=0)
    else:
        raise SystemExit(f"[clip-text-phrase] unknown aggregation {args.aggregation!r}")

    print(f"[clip-text-phrase] feats final shape={feats.shape}")

    has_text = np.array([iid in qwen for iid in image_ids], dtype=bool)
    print(f"[clip-text-phrase] has_text=True: {int(has_text.sum())}/{N}")

    os.makedirs(args.out_dir, exist_ok=True)
    text_part_p = os.path.join(args.out_dir, "text_part.f16.npy")
    has_text_p  = os.path.join(args.out_dir, "has_text.bool.npy")
    np.save(text_part_p, feats.astype(np.float16))
    np.save(has_text_p,  has_text)
    print(f"[clip-text-phrase] wrote {text_part_p}")
    print(f"[clip-text-phrase] wrote {has_text_p}")

    # Symlink all other cache files from the donor directory.
    for fn in os.listdir(args.donor_dir):
        if fn in ("text_part.f16.npy", "has_text.bool.npy"):
            continue
        src = os.path.join(os.path.abspath(args.donor_dir), fn)
        dst = os.path.join(args.out_dir, fn)
        if os.path.exists(dst) or os.path.islink(dst):
            continue
        try:
            os.symlink(src, dst)
        except OSError as e:
            print(f"[clip-text-phrase] symlink failed {fn}: {e}")
    print(f"[clip-text-phrase] cache ready at {args.out_dir}")


if __name__ == "__main__":
    main()
