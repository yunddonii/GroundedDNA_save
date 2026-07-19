"""Extract a flat baseline's 36-bit codes for arbitrary splits (train/query/db).

`scripts/extract_flat_baseline.py` only ever wrote `extract_query.npz` +
`extract_db.npz`, and only knew about Flickr25k / MSCOCO / CIFAR10. The
held-out decoding control (`scripts/heldout_codon_decoding.py`,
REQUIRED_EXPERIMENTS §2.9) needs the baseline's codes for exactly the rows our
model's dictionary is built on -- i.e. the *train* split. For Flickr25k and
NUS-WIDE train is a subset of the DB extraction so it can be sliced out, but
MSCOCO's train split is disjoint from its DB, so an explicit train extraction
is mandatory there. This script produces any requested split, in the same npz
schema our own `extraction_siglip2.py` writes:

    base_indices     [N, 18] int64   2-bit groups  -> A/C/G/T index
    hash_2bit        [N, 36] uint8   binarised sign(continuous_code) in {0,1}
    codebook_indices [N,  6] int64   6-bit groups  -> pseudo-codeword id
    multi_hot_labels [N,  C] int64
    image_paths      [N,]    object  absolute image paths

The 6x6 / 18x2 grouping is IMPOSED, not learned -- that is the whole point of
the control (§2.9).

Usage:
    python scripts/baseline_extract_splits.py \
        --weights params_baseline/260714/cibhash_mscoco_clip_mapr_unsup60/epoch_019.pth \
        --dataset MSCOCO --cache_dir ./cache/mscoco_clip_v4plus \
        --out result_baseline/260719/cibhash_mscoco_clip_decodectl \
        --splits train query
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

from baseline.base_model import (  # noqa: E402
    BackboneWithEncoder, CachedFeatureDataset, NUM_CLASS, _extract_codes,
)
from scripts.extract_flat_baseline import pack_bits_to_indices  # noqa: E402

SPLIT_TO_MODE = {"train": "train", "query": "test", "db": "database"}
SPLIT_TO_FILE = {"train": "extract_train.npz", "query": "extract_query.npz",
                 "db": "extract_db.npz"}


def build_model(ckpt: dict, device: str) -> BackboneWithEncoder:
    """Rebuild the trained head from the checkpoint's encoder state dict."""
    enc = ckpt["encoder_layers"]
    bit = int(ckpt["config"]["bit"])
    if bit % 6 or bit % 2:
        raise ValueError(f"bit={bit} must be divisible by 6 and by 2")
    # Linear/GELU/Dropout triples => Linear layers live at net.0, net.3, ...
    n_lin = sum(1 for k in enc if k.startswith("net.") and k.endswith(".weight"))
    hidden: List[int] = []
    for i in range(n_lin - 1):
        w = enc.get(f"net.{i * 3}.weight")
        if w is not None:
            hidden.append(int(w.shape[0]))
    d_in = int(enc["net.0.weight"].shape[1])
    model = BackboneWithEncoder(
        d_in=d_in, bit=bit, hidden_nodes=hidden or None,
        batch_norm=bool(ckpt["config"].get("batch_norm", False)), finetune=False,
    ).to(device)
    missing, unexpected = model.encoder_layers.load_state_dict(enc, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"state dict mismatch: {missing} / {unexpected}")
    model.eval()
    return model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--dataset", required=True, choices=list(NUM_CLASS.keys()))
    ap.add_argument("--setting", default="setting1")
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--dataset_root", default=os.path.join(REPO, "dataset"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "query"],
                    choices=list(SPLIT_TO_MODE))
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    bit = int(ckpt["config"]["bit"])
    model = build_model(ckpt, args.device)
    print(f"[extract_splits] {args.dataset} bit={bit} "
          f"d_in={model.encoder_layers.net[0].in_features} weights={args.weights}")

    os.makedirs(args.out, exist_ok=True)
    ds_root = os.path.join(args.dataset_root, args.dataset)

    for split in args.splits:
        # Deterministic dataset: no paired-aug views, no index field. Extraction
        # order therefore follows the split manifest line order exactly.
        dset = CachedFeatureDataset(
            args.dataset, args.setting, SPLIT_TO_MODE[split],
            args.dataset_root, args.cache_dir,
            paired_aug=False, return_index=False,
        )
        loader = DataLoader(dset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
        _cont, bin_pm, lbls = _extract_codes(model, loader, args.device)
        bits01 = ((bin_pm + 1) // 2).astype(np.uint8)          # {-1,+1} -> {0,1}
        image_paths = np.array([os.path.join(ds_root, p) for p in dset.paths],
                               dtype=object)
        assert len(image_paths) == len(bits01) == len(lbls)

        out_path = os.path.join(args.out, SPLIT_TO_FILE[split])
        np.savez(
            out_path,
            base_indices=pack_bits_to_indices(bits01, 2).astype(np.int64),
            hash_2bit=bits01,
            codebook_indices=pack_bits_to_indices(bits01, 6).astype(np.int64),
            multi_hot_labels=lbls.astype(np.int64),
            image_paths=image_paths,
        )
        n_uniq = int(np.unique(bits01, axis=0).shape[0])
        print(f"[extract_splits] {split:5s} N={len(dset)} unique_codes={n_uniq} "
              f"({n_uniq / max(len(dset), 1):.4f}) -> {out_path}")

    with open(os.path.join(args.out, "args_extract.txt"), "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")
        f.write(f"bit: {bit}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
