"""Extract OpenAI-CLIP per-slot text features from a Qwen JSONL caption cache.

CLIP-backbone twin of ``extract_siglip2_text_features.py``. Produces a cache
directory by:
    - computing `text_part.f16.npy` of shape [N, M=6, D_proj=512] from the
      CLIP text encoder (pooled per-slot caption embedding via
      ``CLIPModel.get_text_features``).
    - SYMLINKing all OTHER files (image_ids.json, visual_*, has_text,
      meta.json) from a donor cache so the downstream training pipeline
      sees a complete cache.
    - writing has_text=True wherever the qwen cache has a valid entry.

Use case: a new Qwen prompt rev (e.g. V4) has been generated; build a
matching cache without re-running the visual encoder.

Usage:
    python extract_clip_text_features.py \\
        --qwen_cache cache/flickr25k_qwen_v4.jsonl \\
        --donor_dir  cache/flickr25k_clip \\
        --out_dir    cache/flickr25k_clip_v4plus
"""
from __future__ import annotations

import argparse
import json
import os
import sys
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


# Shared by extract_siglip2_text_features.py — must stay in sync.
SLOT_KEYS_V3V4 = (
    "C_global",
    "C_primary_object",
    "C_secondary_object",
    "C_activity_or_relation",
    "C_color_texture",
    "C_scene_type",
)


def load_qwen_jsonl(path: str) -> dict:
    """Parse a Qwen JSONL into {image_id: [6 sentences]}; mirrors
    ``extract_siglip2_text_features.load_qwen_jsonl``."""
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
                    help="Cache dir to symlink image_ids / visual_* / aug / meta from.")
    ap.add_argument("--out_dir",    required=True)
    ap.add_argument("--clip_backbone", default=DEFAULT_CLIP_BACKBONE)
    ap.add_argument("--tokenizer",     default=DEFAULT_CLIP_TOKENIZER_NAME)
    ap.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=256,
                    help="Caption-unit batch size (not images).")
    ap.add_argument("--max_length", type=int, default=64)
    args = ap.parse_args()

    print(f"[clip-text] loading {args.clip_backbone} on {args.device} ...")
    backbone = CLIPBackbone(args.clip_backbone).to(args.device).eval()
    for p in backbone.parameters():                                   # FROZEN
        p.requires_grad = False
    tok = build_clip_text_tokenizer(args.tokenizer)
    D   = backbone.projection_dim                                     # 512 for ViT-B/16
    print(f"[clip-text] text projection dim = {D}")

    # ---- donor image_ids canonical order ---------------------------------
    donor_ids_p = os.path.join(args.donor_dir, "image_ids.json")
    if not os.path.exists(donor_ids_p):
        raise SystemExit(f"[clip-text] donor image_ids.json missing: {donor_ids_p}")
    image_ids: List[str] = json.load(open(donor_ids_p))
    N = len(image_ids)
    M = len(SLOT_KEYS_V3V4)
    print(f"[clip-text] donor image_ids: N={N}")

    # ---- read qwen cache --------------------------------------------------
    print(f"[clip-text] loading qwen cache {args.qwen_cache} ...")
    qwen = load_qwen_jsonl(args.qwen_cache)
    print(f"[clip-text] qwen valid entries (any non-empty slot): {len(qwen)}")
    coverage = sum(1 for iid in image_ids if iid in qwen)
    print(f"[clip-text] image_ids coverage: {coverage}/{N}")

    # ---- assemble caption matrix in image_ids order ---------------------
    captions: List[str] = []
    for iid in image_ids:
        sents = qwen.get(iid)
        if sents is None:
            sents = ["", "", "", "", "", ""]
        captions.extend(sents)

    # ---- batched CLIP text forward --------------------------------------
    feats = np.zeros((N * M, D), dtype=np.float32)
    bs = args.batch_size
    t0 = time.time()
    with torch.no_grad():
        for i in tqdm(range(0, len(captions), bs), desc="clip-text"):
            chunk = captions[i:i+bs]
            enc = tok(
                chunk, padding="max_length", truncation=True,
                max_length=args.max_length, return_tensors="pt",
                return_attention_mask=True,
            ).to(args.device)
            input_ids = enc["input_ids"]                              # [b, L]
            attn      = enc.get("attention_mask")
            if attn is None:
                pad_id = getattr(tok, "pad_token_id", None)
                attn = (input_ids != pad_id).long() if pad_id is not None else torch.ones_like(input_ids)
            t_feat = backbone.model.get_text_features(
                input_ids=input_ids, attention_mask=attn,
            )
            t_feat = coerce_pooled_to_tensor(t_feat)                  # [b, D_proj]
            feats[i:i+t_feat.shape[0]] = t_feat.float().cpu().numpy()
    feats = feats.reshape(N, M, D)
    print(f"[clip-text] encode done in {time.time()-t0:.1f}s; shape={feats.shape}")

    # ---- build has_text mask --------------------------------------------
    has_text = np.array([iid in qwen for iid in image_ids], dtype=bool)
    print(f"[clip-text] has_text=True: {int(has_text.sum())}/{N}")

    # ---- write out dir --------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    text_part_p = os.path.join(args.out_dir, "text_part.f16.npy")
    np.save(text_part_p, feats.astype(np.float16))
    print(f"[clip-text] wrote {text_part_p}  ({feats.nbytes / 1e6:.1f} MB)")

    has_text_p = os.path.join(args.out_dir, "has_text.bool.npy")
    np.save(has_text_p, has_text)
    print(f"[clip-text] wrote {has_text_p}")

    # ---- symlink everything else from donor -----------------------------
    for fn in os.listdir(args.donor_dir):
        if fn in ("text_part.f16.npy", "has_text.bool.npy"):
            continue
        if fn.startswith("text_token"):
            continue
        src = os.path.join(args.donor_dir, fn)
        dst = os.path.join(args.out_dir, fn)
        if os.path.exists(dst) or os.path.islink(dst):
            os.remove(dst)
        os.symlink(os.path.abspath(src), dst)
        print(f"[clip-text] symlink {fn} -> {os.path.relpath(src, args.out_dir)}")

    # ---- updated meta ---------------------------------------------------
    meta_p = os.path.join(args.out_dir, "meta.json")
    if os.path.islink(meta_p):
        os.remove(meta_p)
    donor_meta_p = os.path.join(args.donor_dir, "meta.json")
    if os.path.exists(donor_meta_p):
        donor_meta = json.load(open(donor_meta_p))
    else:
        donor_meta = {}
    new_meta = dict(donor_meta)
    new_meta["text_encoder"]     = args.clip_backbone
    new_meta["text_max_length"]  = args.max_length
    new_meta["text_part_source"] = "clip"
    new_meta["qwen_cache"]       = os.path.basename(args.qwen_cache)
    new_meta["backbone_family"]  = "clip"
    with open(meta_p, "w") as f:
        json.dump(new_meta, f, indent=2)
    print(f"[clip-text] wrote meta.json (text_encoder={args.clip_backbone}, "
          f"qwen={os.path.basename(args.qwen_cache)})")

    # ---- diagnostic: cross-slot cos sim on has_text=True only ----------
    valid = feats[has_text]
    if valid.shape[0] > 0:
        n = valid / (np.linalg.norm(valid, axis=-1, keepdims=True) + 1e-8)
        cos = np.einsum("nmd,nkd->nmk", n, n).mean(axis=0)
        iu = np.triu_indices(M, k=1)
        off = cos[iu]
        print(f"[clip-text] cross-slot cos sim (valid {int(has_text.sum())} samples):")
        print(f"  off-diag mean={off.mean():.4f}  min={off.min():.4f}  max={off.max():.4f}")
        np.set_printoptions(precision=3, suppress=True, linewidth=120)
        print(cos)
        print(f"  Reference: SigLIP2 V4 cross-slot off-diag mean was 0.7335; "
              f"CLIP usually sits lower.")


if __name__ == "__main__":
    sys.exit(main())
