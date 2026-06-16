"""Extract CLIP TOKEN-LEVEL per-slot text features from a Qwen JSONL cache.

Token-level twin of ``extract_clip_text_features.py``. For each image and
each of the M=6 codebook captions, runs the FROZEN CLIP text encoder and
projects every token through ``text_projection`` to obtain CLIP-space
token embeddings.

Outputs (in addition to the standard pooled cache layout):
    text_tokens.f16.npy        [N, M=6, T, D_proj]   # projected token embeds
    text_token_mask.bool.npy   [N, M=6, T]           # True at real tokens

All other files (image_ids, visual_*, text_part, has_text, meta) are
symlinked from a donor directory so the cache is drop-in for training.

This cache enables the v162-family grounded-text-routing design:
visually-routed embeddings select top-k_t relevant text tokens per
codebook via a second Sinkhorn-OT pass, producing refined text
embeddings for the loss layer.

Usage:
    python extract_clip_text_token_features.py \\
        --qwen_cache  cache/flickr25k_qwen3_v4_trainset.jsonl \\
        --donor_dir   cache/flickr25k_clip_v4plus_qwen3 \\
        --out_dir     cache/flickr25k_clip_v4plus_qwen3_tokens
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import List

import numpy as np
import torch
from tqdm import tqdm

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen_cache", required=True)
    ap.add_argument("--donor_dir",  required=True,
                    help="Cache dir to symlink image_ids / visual_* / text_part / aug / meta from.")
    ap.add_argument("--out_dir",    required=True)
    ap.add_argument("--clip_backbone", default=DEFAULT_CLIP_BACKBONE)
    ap.add_argument("--tokenizer",     default=DEFAULT_CLIP_TOKENIZER_NAME)
    ap.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=256,
                    help="Caption-unit batch size (not images).")
    ap.add_argument("--max_length", type=int, default=32,
                    help="Per-caption token sequence length (padded).")
    args = ap.parse_args()

    print(f"[clip-text-tok] loading {args.clip_backbone} on {args.device} ...")
    backbone = CLIPBackbone(args.clip_backbone).to(args.device).eval()
    for p in backbone.parameters():
        p.requires_grad = False
    tok = build_clip_text_tokenizer(args.tokenizer)
    D = backbone.projection_dim
    T = args.max_length
    print(f"[clip-text-tok] text projection dim D={D}, max_length T={T}")

    donor_ids_p = os.path.join(args.donor_dir, "image_ids.json")
    if not os.path.exists(donor_ids_p):
        raise SystemExit(f"[clip-text-tok] donor image_ids.json missing: {donor_ids_p}")
    image_ids: List[str] = json.load(open(donor_ids_p))
    N = len(image_ids)
    M = len(SLOT_KEYS_V3V4)
    print(f"[clip-text-tok] donor image_ids: N={N}")

    print(f"[clip-text-tok] loading qwen cache {args.qwen_cache} ...")
    qwen = load_qwen_jsonl(args.qwen_cache)
    print(f"[clip-text-tok] qwen valid entries: {len(qwen)}")
    coverage = sum(1 for iid in image_ids if iid in qwen)
    print(f"[clip-text-tok] image_ids coverage: {coverage}/{N}")

    # ---- preallocate disk-backed outputs ---------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    text_tok_p = os.path.join(args.out_dir, "text_tokens.f16.npy")
    mask_p     = os.path.join(args.out_dir, "text_token_mask.bool.npy")
    tokens_mm = np.lib.format.open_memmap(
        text_tok_p, mode="w+", dtype=np.float16, shape=(N, M, T, D)
    )
    mask_mm = np.lib.format.open_memmap(
        mask_p, mode="w+", dtype=np.bool_, shape=(N, M, T)
    )
    print(f"[clip-text-tok] preallocated {text_tok_p} {tokens_mm.shape} "
          f"= {tokens_mm.nbytes/1e9:.2f} GB")

    # ---- assemble flat caption list in (N*M) order -----------------------
    captions: List[str] = []
    for iid in image_ids:
        sents = qwen.get(iid)
        if sents is None:
            sents = ["", "", "", "", "", ""]
        captions.extend(sents)
    assert len(captions) == N * M

    # ---- batched CLIP text forward (token-level) -------------------------
    bs = args.batch_size
    t0 = time.time()
    with torch.no_grad():
        for i in tqdm(range(0, len(captions), bs), desc="clip-text-tok"):
            chunk = captions[i:i+bs]
            enc = tok(
                chunk, padding="max_length", truncation=True,
                max_length=T, return_tensors="pt",
                return_attention_mask=True,
            ).to(args.device)
            input_ids = enc["input_ids"]           # [b, T]
            attn      = enc.get("attention_mask")
            if attn is None:
                pad_id = getattr(tok, "pad_token_id", None)
                attn = (input_ids != pad_id).long() if pad_id is not None else torch.ones_like(input_ids)
            out = backbone.model.text_model(
                input_ids=input_ids, attention_mask=attn,
            )
            # last_hidden_state: [b, T, H_t]
            tokens_h = out.last_hidden_state
            # project to shared image/text embedding space (CLIP applies the
            # SAME text_projection that get_text_features uses on the pooled
            # vector — here we apply it token-wise).
            tokens_d = backbone.model.text_projection(tokens_h)   # [b, T, D]
            tokens_d = tokens_d.float().cpu().numpy().astype(np.float16)
            mask_np  = attn.bool().cpu().numpy()
            # also clear zero-length captions: if the entire caption was
            # empty, the tokenizer still emits BOS/PAD only — keep as-is;
            # downstream has_text mask already filters those rows.

            # write into (n, m) view
            for k in range(tokens_d.shape[0]):
                row_global = (i + k) // M
                slot       = (i + k) % M
                tokens_mm[row_global, slot] = tokens_d[k]
                mask_mm   [row_global, slot] = mask_np[k]

    tokens_mm.flush(); del tokens_mm
    mask_mm.flush();   del mask_mm
    print(f"[clip-text-tok] encode done in {time.time()-t0:.1f}s")

    # ---- symlink everything else from donor -----------------------------
    keep_from_donor = (
        "image_ids.json", "meta.json",
        "text_part.f16.npy", "has_text.bool.npy",
        "text_whiten.npz", "text_whiten.npz.meta.json",
        "visual_global.f16.npy",
        "visual_global_aug0.f16.npy", "visual_global_aug1.f16.npy",
        "visual_tokens.f16.npy",
        "visual_tokens_aug0.f16.npy", "visual_tokens_aug1.f16.npy",
    )
    for fn in keep_from_donor:
        src = os.path.join(args.donor_dir, fn)
        if not os.path.exists(src):
            continue
        dst = os.path.join(args.out_dir, fn)
        if os.path.exists(dst) or os.path.islink(dst):
            os.remove(dst)
        os.symlink(os.path.abspath(src), dst)
        print(f"[clip-text-tok] symlink {fn}")

    # ---- update meta ---------------------------------------------------
    meta_p = os.path.join(args.out_dir, "meta.json")
    if os.path.islink(meta_p):
        os.remove(meta_p)
    donor_meta_p = os.path.join(args.donor_dir, "meta.json")
    donor_meta = json.load(open(donor_meta_p)) if os.path.exists(donor_meta_p) else {}
    new_meta = dict(donor_meta)
    new_meta["text_token_encoder"]    = args.clip_backbone
    new_meta["text_token_max_length"] = T
    new_meta["text_token_dim"]        = D
    new_meta["text_token_source"]     = "clip_text_projection_on_last_hidden_state"
    new_meta["qwen_cache"]            = os.path.basename(args.qwen_cache)
    with open(meta_p, "w") as f:
        json.dump(new_meta, f, indent=2)
    print(f"[clip-text-tok] wrote meta.json")

    print(f"[clip-text-tok] done. cache @ {args.out_dir}")


if __name__ == "__main__":
    main()
