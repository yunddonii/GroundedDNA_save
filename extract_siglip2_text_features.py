"""Extract SigLIP2 per-slot text features from a Qwen JSONL caption cache.

Produces a SigLIP2-compatible cache directory by:
    - computing `text_part.f16.npy` of shape [N, M=6, D=768] from the
      SigLIP2 text encoder (pooled per-slot caption embedding).
    - SYMLINKing all OTHER files (image_ids.json, visual_*, has_text,
      meta.json) from a donor cache so the downstream training pipeline
      sees a complete v3plus-shape cache.
    - writing has_text=True wherever the qwen cache has a valid entry.

Use case: a new Qwen prompt rev (e.g. V4) has been generated; build a
matching v?plus cache without re-running the visual encoder.

Usage:
    python extract_siglip2_text_features.py \\
        --qwen_cache cache/flickr25k_qwen_v4.jsonl \\
        --donor_dir  cache/flickr25k_siglip2_v3plus \\
        --out_dir    cache/flickr25k_siglip2_v4plus
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
            cb = row.get("codebook_texts", {}) or {}
            sents = [str(cb.get(k, "") or "") for k in SLOT_KEYS_V3V4]
            # "none" -> empty (treat as no caption to mimic V3 behavior)
            sents_clean = [s if s.strip().lower() not in ("", "none") else "" for s in sents]
            # an entry counts as valid only if at least one slot is non-empty
            if any(s for s in sents_clean):
                out[iid] = sents_clean
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen_cache", required=True)
    ap.add_argument("--donor_dir",  required=True,
                    help="Cache dir to symlink image_ids/visual/aug/meta from.")
    ap.add_argument("--out_dir",    required=True)
    ap.add_argument("--siglip2_backbone", default="google/siglip2-base-patch16-224")
    ap.add_argument("--tokenizer",        default="google/siglip2-base-patch16-224")
    ap.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=256,
                    help="Caption-unit batch size (not images).")
    ap.add_argument("--max_length", type=int, default=64)
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoModel

    print(f"[siglip2-text] loading {args.siglip2_backbone} on {args.device} ...")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    full = AutoModel.from_pretrained(args.siglip2_backbone).to(args.device).eval()
    # SigLIP2 model wraps text_model + vision_model. We only need text_model.
    if hasattr(full, "text_model"):
        text_model = full.text_model
    else:
        text_model = full
    D = full.config.text_config.hidden_size if hasattr(full.config, "text_config") else full.config.hidden_size
    print(f"[siglip2-text] text hidden_size = {D}")

    # ---- donor image_ids canonical order ----------------------------------
    donor_ids_p = os.path.join(args.donor_dir, "image_ids.json")
    image_ids: List[str] = json.load(open(donor_ids_p))
    N = len(image_ids)
    M = len(SLOT_KEYS_V3V4)
    print(f"[siglip2-text] donor image_ids: N={N}")

    # ---- read qwen cache --------------------------------------------------
    print(f"[siglip2-text] loading qwen cache {args.qwen_cache} ...")
    qwen = load_qwen_jsonl(args.qwen_cache)
    print(f"[siglip2-text] qwen valid entries (any non-empty slot): {len(qwen)}")
    coverage = sum(1 for iid in image_ids if iid in qwen)
    print(f"[siglip2-text] image_ids coverage: {coverage}/{N}")

    # ---- assemble caption matrix in image_ids order ---------------------
    # Use empty string for missing entries -- the SigLIP2 text encoder will
    # produce a placeholder embedding for those rows (later masked out via
    # has_text=False).
    captions = []
    for iid in image_ids:
        sents = qwen.get(iid)
        if sents is None:
            sents = ["", "", "", "", "", ""]
        captions.extend(sents)

    # ---- batched SigLIP2 text forward -----------------------------------
    feats = np.zeros((N * M, D), dtype=np.float32)
    bs = args.batch_size
    t0 = time.time()
    with torch.no_grad():
        for i in tqdm(range(0, len(captions), bs), desc="siglip2-text"):
            chunk = captions[i:i+bs]
            enc = tok(chunk, padding="max_length", truncation=True,
                      max_length=args.max_length, return_tensors="pt").to(args.device)
            out = text_model(input_ids=enc["input_ids"],
                             attention_mask=enc.get("attention_mask"))
            # SigLIP2 text outputs use pooler_output (the "[EOS]" head)
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                pooled = out.pooler_output                       # [B, D]
            else:
                # fallback: last_hidden_state[:, -1] (SigLIP2 uses end-of-seq)
                pooled = out.last_hidden_state[:, -1, :]
            feats[i:i+pooled.shape[0]] = pooled.float().cpu().numpy()
    feats = feats.reshape(N, M, D)
    print(f"[siglip2-text] encode done in {time.time()-t0:.1f}s; shape={feats.shape}")

    # ---- build has_text mask --------------------------------------------
    has_text = np.array([iid in qwen for iid in image_ids], dtype=bool)
    print(f"[siglip2-text] has_text=True: {int(has_text.sum())}/{N}")

    # ---- write out dir --------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    text_part_p = os.path.join(args.out_dir, "text_part.f16.npy")
    np.save(text_part_p, feats.astype(np.float16))
    print(f"[siglip2-text] wrote {text_part_p}  ({feats.nbytes / 1e6:.1f} MB)")

    has_text_p = os.path.join(args.out_dir, "has_text.bool.npy")
    np.save(has_text_p, has_text)
    print(f"[siglip2-text] wrote {has_text_p}")

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
        print(f"[siglip2-text] symlink {fn} -> {os.path.relpath(src, args.out_dir)}")

    # ---- updated meta ---------------------------------------------------
    meta_p = os.path.join(args.out_dir, "meta.json")
    if os.path.islink(meta_p):
        os.remove(meta_p)
    donor_meta = json.load(open(os.path.join(args.donor_dir, "meta.json")))
    new_meta = dict(donor_meta)
    new_meta["text_encoder"]     = args.siglip2_backbone
    new_meta["text_max_length"]  = args.max_length
    new_meta["text_part_source"] = "siglip2"
    new_meta["qwen_cache"]       = os.path.basename(args.qwen_cache)
    with open(meta_p, "w") as f:
        json.dump(new_meta, f, indent=2)
    print(f"[siglip2-text] wrote meta.json (text_encoder={args.siglip2_backbone}, qwen={os.path.basename(args.qwen_cache)})")

    # ---- diagnostic: cross-slot cos sim on has_text=True only ----------
    valid = feats[has_text]
    n = valid / (np.linalg.norm(valid, axis=-1, keepdims=True) + 1e-8)
    cos = np.einsum("nmd,nkd->nmk", n, n).mean(axis=0)
    iu = np.triu_indices(M, k=1)
    off = cos[iu]
    print(f"[siglip2-text] cross-slot cos sim (valid {int(has_text.sum())} samples):")
    print(f"  off-diag mean={off.mean():.4f}  min={off.min():.4f}  max={off.max():.4f}")
    print(f"  per-pair matrix:")
    np.set_printoptions(precision=3, suppress=True, linewidth=120)
    print(cos)
    print(f"  Reference: SigLIP2 V3 (5K valid) off-diag mean=0.834")


if __name__ == "__main__":
    sys.exit(main())
