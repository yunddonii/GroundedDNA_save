"""Extract 36-bit codes from a trained flat baseline (CIBHash / CIMON /
MLS3RDUH) and reshape them as if they were a 6-codebook x 3-codon x 2-bit
DNA hash. Saves `extract_db.npz` / `extract_query.npz` in the same schema as
our SigLIP2 result dirs so the existing compositional analysis tools
(pairwise_nmi.py, codebook_drop_ablation*.py, compositional_eval.py) can run
on them without modification.

Imposed partition: bits [6m : 6m+6] of the 36-bit code are treated as
"codebook m" (giving K=2^6=64 codewords, matching our K=64 setting), and
bits [2p : 2p+2] are treated as "base position p" (giving 4 bases A/C/G/T).
This is an ARTIFICIAL grouping with no learned structure; the whole point
is to compare the *imposed* compositional analysis on these flat codes
against the *learned* compositional structure in our DNA hash model.

Usage:
    python scripts/extract_flat_baseline.py \
        --weights params_baseline/260515/cibhash_flickr25k_unsup60/epoch_059.pth \
        --dataset Flickr25k --setting setting1 \
        --cache_dir cache/flickr25k_siglip2_v4plus \
        --out result_baseline/260515/cibhash_flickr25k_unsup60/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader

# Make repo root importable for `baseline.*` module path.
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

from baseline.base_model import BackboneWithEncoder, load_dataset, _extract_codes  # noqa: E402


def pack_bits_to_indices(bits01: np.ndarray, group_size: int) -> np.ndarray:
    """Pack consecutive groups of `group_size` bits into integer indices.

    bits01: [N, total_bits] in {0, 1}
    Returns [N, total_bits // group_size] of integers in [0, 2^group_size).
    Bit order within a group: MSB first.
    """
    assert bits01.ndim == 2
    N, total = bits01.shape
    assert total % group_size == 0, (total, group_size)
    G = total // group_size
    weights = 2 ** np.arange(group_size - 1, -1, -1, dtype=np.int64)  # MSB-first
    grouped = bits01.reshape(N, G, group_size).astype(np.int64)
    return (grouped * weights[None, None, :]).sum(axis=-1)             # [N, G]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True,
                    help="trained baseline .pth (must contain encoder_layers state_dict and config)")
    ap.add_argument("--dataset", required=True, choices=["Flickr25k", "MSCOCO", "CIFAR10"])
    ap.add_argument("--setting", default="setting1")
    ap.add_argument("--cache_dir", required=True,
                    help="SigLIP2 feature cache directory (e.g. cache/flickr25k_siglip2_v4plus)")
    ap.add_argument("--dataset_root", default=os.path.join(REPO, "dataset"))
    ap.add_argument("--out", required=True,
                    help="output directory (will write extract_db.npz + extract_query.npz)")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    # ---- Load checkpoint ------------------------------------------------
    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    bit = int(ckpt["config"]["bit"])
    if bit % 6 != 0 or bit % 2 != 0:
        raise ValueError(
            f"bit={bit} must be divisible by both 6 (codebooks) and 2 "
            f"(bits per base). Use a 36-bit checkpoint."
        )
    n_codebooks = bit // 6
    n_bases = bit // 2
    print(f"[extract_flat] bit={bit}  ->  {n_codebooks} codebooks x {n_bases // n_codebooks} codons x 2 bits")

    # ---- Build model + load weights -------------------------------------
    enc_layers = ckpt.get("encoder_layers", {})
    hidden_nodes: List[int] = []
    # Infer hidden dims from state-dict keys (net.0, net.3, ... are Linear layers)
    n_linears = sum(1 for k in enc_layers if k.endswith(".weight") and k.startswith("net."))
    if n_linears > 1:
        # Determine hidden sizes via .weight shapes
        for i in range(n_linears - 1):
            wkey = f"net.{i * 3}.weight"  # Linear, GELU, Dropout => 3-stride
            if wkey in enc_layers:
                hidden_nodes.append(int(enc_layers[wkey].shape[0]))
    # Auto-detect d_in from the FIRST Linear's weight matrix (which has
    # shape [out_dim, in_dim]). SigLIP2 baselines saved with in_dim=768,
    # CLIP baselines (D_proj=512) with in_dim=512.
    d_in = 768
    first_w = enc_layers.get("net.0.weight")
    if first_w is not None:
        d_in = int(first_w.shape[1])
    model = BackboneWithEncoder(
        d_in=d_in, bit=bit,
        hidden_nodes=hidden_nodes or None,
        batch_norm=bool(ckpt["config"].get("batch_norm", False)),
        finetune=False,
    ).to(args.device)
    missing, unexpected = model.encoder_layers.load_state_dict(enc_layers, strict=False)
    if missing or unexpected:
        print(f"[extract_flat] WARN missing={missing} unexpected={unexpected}")
    model.eval()
    print(f"[extract_flat] loaded encoder ({sum(p.numel() for p in model.encoder_layers.parameters())} params)")

    # ---- Build dataset splits (test + database) -------------------------
    _, test_set, db_set = load_dataset(
        dataset_root=args.dataset_root,
        dataset_name=args.dataset, setting=args.setting,
        load_train=False, load_test=True, load_database=True,
        cache_dir=args.cache_dir,
    )

    os.makedirs(args.out, exist_ok=True)

    for split, dset, out_name in [("query", test_set, "extract_query.npz"),
                                   ("db",    db_set,   "extract_db.npz")]:
        loader = DataLoader(dset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
        cont, bin_pm, lbls = _extract_codes(model, loader, args.device)
        bits01 = ((bin_pm + 1) // 2).astype(np.uint8)                       # {0,1}
        codebook_indices = pack_bits_to_indices(bits01, group_size=6)       # [N, M]
        base_indices     = pack_bits_to_indices(bits01, group_size=2)       # [N, R]
        # Image paths: dataset stores .paths attribute; CachedFeatureDataset
        # returns absolute-rooted relative paths like 'images/im00001.jpg'.
        # For compositional_eval path-matching we prepend the dataset folder.
        if hasattr(dset, "paths"):
            ds_root = os.path.join(args.dataset_root, args.dataset)
            image_paths = np.array(
                [os.path.join(ds_root, p) for p in dset.paths], dtype=object,
            )
        else:
            image_paths = np.array([f"{args.dataset}:{i}" for i in range(len(dset))],
                                   dtype=object)
        out_path = os.path.join(args.out, out_name)
        np.savez(
            out_path,
            base_indices=base_indices.astype(np.int64),
            hash_2bit=bits01.astype(np.uint8),
            codebook_indices=codebook_indices.astype(np.int64),
            multi_hot_labels=lbls.astype(np.int64),
            image_paths=image_paths,
        )
        unique_count = int(np.unique(codebook_indices, axis=0).shape[0])
        print(f"[extract_flat] {split}: N={len(dset)}  unique_codes={unique_count}  ->  {out_path}")

    # ---- Persist a minimal args.txt so analysis tools recognise the dir
    args_path = os.path.join(args.out, "args_extract.txt")
    with open(args_path, "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")
        f.write(f"bit: {bit}\nn_codebooks: {n_codebooks}\n")
    print(f"[extract_flat] wrote {args_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
