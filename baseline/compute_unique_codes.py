"""Compute unique-code stats for trained baseline checkpoints.

For each baseline checkpoint, loads the encoder, runs it over the QUERY
split of Flickr25k setting1, and reports:
    n_query     : number of query samples
    n_unique    : number of distinct 36-bit codes (sign of continuous head)
    unique_rate : n_unique / n_query
    n_dup       : n_query - n_unique
    dup_rate    : 1 - unique_rate
    p99         : 99th percentile collision count (max samples sharing one code)

Mirrors the unique_code_ratio metric reported for our DNA model so
v27b / v28a / v28b can be compared head-to-head against CIBHash, CIMON,
MLS3RDUH, and the supervised baselines on the same axis.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

# Use the same module path as base_model.main()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baseline.base_model import (
    BackboneWithEncoder,
    CachedFeatureDataset,
    _extract_codes,
)


def _bin_to_uint64_packed(bin_codes: np.ndarray) -> np.ndarray:
    """Pack [-1,+1] {N, bit} into uint64-tuples for fast hashing."""
    bits = (bin_codes > 0).astype(np.uint8)              # [N, bit]
    N, bit = bits.shape
    n_words = (bit + 63) // 64
    out = np.zeros((N, n_words), dtype=np.uint64)
    for i in range(bit):
        w = i // 64
        b = i %  64
        out[:, w] |= bits[:, i].astype(np.uint64) << np.uint64(b)
    return out


def compute_unique_stats(bin_codes: np.ndarray) -> dict:
    packed = _bin_to_uint64_packed(bin_codes)             # [N, n_words]
    keys   = [tuple(row.tolist()) for row in packed]
    from collections import Counter
    counts = Counter(keys)
    sizes  = np.array(sorted(counts.values()), dtype=np.int64)
    N      = len(keys)
    n_uni  = len(counts)
    return {
        "n_query":     int(N),
        "n_unique":    int(n_uni),
        "unique_rate": float(n_uni) / float(N),
        "n_dup":       int(N - n_uni),
        "dup_rate":    1.0 - float(n_uni) / float(N),
        "p99_collision": int(np.quantile(sizes, 0.99)),
        "max_collision": int(sizes.max()),
        "mean_collision": float(sizes.mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",        required=True, help="path to epoch_<n>.pth")
    ap.add_argument("--dataset",     default="Flickr25k")
    ap.add_argument("--setting",     default="setting1")
    ap.add_argument("--split",       default="test", choices=["train", "test", "database"])
    ap.add_argument("--dataset_dir", default="./dataset")
    ap.add_argument("--cache_dir",   default=None,
                    help="SigLIP2 feature cache dir (default per-dataset).")
    ap.add_argument("--bit",         type=int, default=36)
    ap.add_argument("--device",      default="cuda:0")
    ap.add_argument("--batch_size",  type=int, default=256)
    args = ap.parse_args()

    # --- detect whether the saved encoder has a BatchNorm layer ---
    sd = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    enc_sd = sd.get("encoder_layers", sd.get("state_dict", sd))
    has_bn = any(k.endswith("running_mean") for k in enc_sd.keys())

    # --- build encoder + load weights ---
    model = BackboneWithEncoder(d_in=768, bit=args.bit, hidden_nodes=None,
                                batch_norm=has_bn, finetune=False).to(args.device)
    if "encoder_layers" in sd:
        model.encoder_layers.load_state_dict(sd["encoder_layers"])
    elif "state_dict" in sd:
        model.load_state_dict(sd["state_dict"])
    else:
        model.load_state_dict(sd)
    model.eval()

    # --- dataset + loader ---
    ds = CachedFeatureDataset(
        args.dataset, args.setting, args.split,
        dataset_dir=args.dataset_dir, cache_dir=args.cache_dir,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # --- extract binary codes ---
    cont, bin_, _lbl = _extract_codes(model, loader, args.device)
    print(f"[unique] bin_codes shape: {bin_.shape}")

    stats = compute_unique_stats(bin_)
    out_p = os.path.join(os.path.dirname(args.ckpt),
                          f"unique_codes_{args.split}.json")
    with open(out_p, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[unique] saved -> {out_p}")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
