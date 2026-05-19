"""Extract BERT per-slot text features for an existing Qwen JSONL cache.

Produces a SigLIP2-compatible cache directory by:
    - computing `text_part.f16.npy` of shape [N, M=6, D=768] from BERT
      pooled outputs over each codebook-slot caption.
    - SYMLINK-ing all other files (image_ids.json, visual_*, has_text,
      meta.json) from a donor cache so the same downstream code paths
      (extract_siglip2_features.py / train_siglip2.py / ImgRtvDataset)
      keep working.

This is the v45 "swap SigLIP2 text encoder -> BERT" intervention to
counter SigLIP2 text-encoder cross-slot uniformity (cos ~0.97 in V1/V3
caches), per the §4-1 diagnosis. BERT is text-only MLM, never shared a
loss with images, so its per-slot embeddings should be more
linguistically discriminative.

Usage:
    python extract_bert_text_features.py \\
        --qwen_cache cache/flickr25k_qwen_v3.jsonl \\
        --donor_dir  cache/flickr25k_siglip2_v3plus \\
        --out_dir    cache/flickr25k_siglip2_v3plus_bert \\
        --bert       bert-base-uncased
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


SLOT_KEYS_V3 = (
    "C_global",
    "C_primary_object",
    "C_secondary_object",
    "C_activity_or_relation",
    "C_color_texture",
    "C_scene_type",
)


def load_qwen_jsonl(path: str) -> dict:
    """Read JSONL into {image_id: [6 sentences]} dict, V3 schema."""
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
            sents = [str(cb.get(k, "") or "") for k in SLOT_KEYS_V3]
            out[iid] = sents
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen_cache", required=True,
                    help="Path to flickr25k_qwen_v3.jsonl (V3 caption schema).")
    ap.add_argument("--donor_dir", required=True,
                    help="SigLIP2 cache directory to symlink other files from "
                         "(e.g. flickr25k_siglip2_v3plus).")
    ap.add_argument("--out_dir",   required=True,
                    help="Output cache directory to create.")
    ap.add_argument("--bert", default="bert-base-uncased",
                    help="HuggingFace BERT model name. Default bert-base "
                         "(D=768, matches existing d_model).")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=256,
                    help="BERT forward batch size (in caption units, not images).")
    ap.add_argument("--max_length", type=int, default=64,
                    help="BERT tokenizer max length per caption.")
    ap.add_argument("--pool", choices=("cls", "mean"), default="cls",
                    help="Pooling: CLS token (default, matches BERT's NSP head) "
                         "or mean of last hidden states (attention-mask weighted).")
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoModel

    print(f"[bert-text] loading {args.bert} on {args.device} ...")
    tok = AutoTokenizer.from_pretrained(args.bert)
    model = AutoModel.from_pretrained(args.bert).to(args.device).eval()
    D = model.config.hidden_size
    print(f"[bert-text] hidden_size = {D}")

    # ---- read image_ids.json from donor to get canonical order/N ------------
    donor_ids_p = os.path.join(args.donor_dir, "image_ids.json")
    with open(donor_ids_p) as f:
        image_ids: List[str] = json.load(f)
    N = len(image_ids)
    M = len(SLOT_KEYS_V3)
    print(f"[bert-text] donor image_ids: N={N}")

    # ---- read qwen cache ----------------------------------------------------
    print(f"[bert-text] loading qwen cache {args.qwen_cache} ...")
    qwen = load_qwen_jsonl(args.qwen_cache)
    print(f"[bert-text] qwen entries: {len(qwen)}")

    # ---- assemble caption matrix in image_ids order ------------------------
    captions = []  # length N*M flat list
    miss = 0
    for iid in image_ids:
        sents = qwen.get(iid)
        if sents is None:
            miss += 1
            sents = ["", "", "", "", "", ""]
        captions.extend(sents)
    if miss > 0:
        print(f"[bert-text] WARN: {miss} image_ids missing from qwen cache; "
              f"filled with empty strings (BERT will encode as [CLS][SEP]).")

    # ---- batched BERT forward ----------------------------------------------
    feats = np.zeros((N * M, D), dtype=np.float32)
    bs = args.batch_size
    t0 = time.time()
    with torch.no_grad():
        for i in tqdm(range(0, len(captions), bs), desc="bert-encode"):
            chunk = captions[i:i+bs]
            enc = tok(chunk, padding=True, truncation=True,
                      max_length=args.max_length, return_tensors="pt").to(args.device)
            out = model(**enc)
            last = out.last_hidden_state                # [B, L, D]
            if args.pool == "cls":
                pooled = last[:, 0, :]                  # CLS
            else:
                mask = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            feats[i:i+pooled.shape[0]] = pooled.float().cpu().numpy()
    feats = feats.reshape(N, M, D)
    print(f"[bert-text] BERT encode done in {time.time()-t0:.1f}s; shape={feats.shape}")

    # ---- write out dir ------------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    text_part_p = os.path.join(args.out_dir, "text_part.f16.npy")
    np.save(text_part_p, feats.astype(np.float16))
    print(f"[bert-text] wrote {text_part_p}  ({feats.nbytes / 1e6:.1f} MB)")

    # ---- symlink everything else from donor --------------------------------
    for fn in os.listdir(args.donor_dir):
        if fn == "text_part.f16.npy":
            continue  # we just wrote our own
        # also skip text_tokens (token-level cache from a different cache
        # variant; not present in v3plus, but be defensive).
        if fn.startswith("text_token"):
            continue
        src = os.path.join(args.donor_dir, fn)
        dst = os.path.join(args.out_dir, fn)
        if os.path.exists(dst) or os.path.islink(dst):
            os.remove(dst)
        os.symlink(os.path.abspath(src), dst)
        print(f"[bert-text] symlink {fn} -> {os.path.relpath(src, args.out_dir)}")

    # ---- update meta.json so downstream code knows ------------------------
    meta_p = os.path.join(args.out_dir, "meta.json")
    if os.path.islink(meta_p):
        os.remove(meta_p)
    donor_meta = json.load(open(os.path.join(args.donor_dir, "meta.json")))
    new_meta = dict(donor_meta)
    new_meta["text_encoder"]  = args.bert
    new_meta["text_pool"]     = args.pool
    new_meta["text_max_length"] = args.max_length
    new_meta["D_proj"]        = int(D)
    new_meta["text_part_source"] = "bert"  # tag for downstream sanity
    with open(meta_p, "w") as f:
        json.dump(new_meta, f, indent=2)
    print(f"[bert-text] wrote meta.json (text_encoder={args.bert})")

    # ---- quick diagnostic: cross-slot cos sim ------------------------------
    norm = feats / (np.linalg.norm(feats, axis=-1, keepdims=True) + 1e-8)
    cos = np.einsum("nmd,nkd->nmk", norm, norm).mean(axis=0)  # [M, M]
    iu = np.triu_indices(M, k=1)
    off = cos[iu]
    print(f"[bert-text] cross-slot cos sim (off-diag): "
          f"mean={off.mean():.4f}  min={off.min():.4f}  max={off.max():.4f}")
    print(f"            (SigLIP2 V3 reference: mean=0.9668, min=0.9523)")


if __name__ == "__main__":
    sys.exit(main())
