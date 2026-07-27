"""Extract 36/48-bit codes from a registered flat hashing baseline and reshape
them as fixed 6-bit codebooks and 2-bit DNA bases. The checkpoint
is reconstructed through its method class, so both legacy linear encoders and
modern custom heads are supported. Saves `extract_db.npz` /
`extract_query.npz` in the same schema as our result dirs so analysis tools
(pairwise_nmi.py, codebook_drop_ablation*.py, compositional_eval.py) can run
on them without modification.

Imposed partition: bits [6m : 6m+6] of the flat code are treated as
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
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

# Make repo root importable for `baseline.*` module path.
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

from baseline.base_model import (  # noqa: E402
    _extract_codes,
    build_model_from_checkpoint_payload,
    load_dataset,
    verify_checkpoint_data_context,
)


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
                    help="trained baseline .pth (full or legacy encoder-only checkpoint)")
    ap.add_argument("--dataset", required=True,
                    choices=["Flickr25k", "MSCOCO", "NUSWIDE", "CIFAR10"])
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
    if bit not in (36, 48):
        raise ValueError(
            f"bit={bit} is outside the registered comparison budgets; "
            "use a 36-bit or 48-bit checkpoint."
        )
    n_codebooks = bit // 6
    n_bases = bit // 2
    print(f"[extract_flat] bit={bit}  ->  {n_codebooks} codebooks x {n_bases // n_codebooks} codons x 2 bits")
    verify_checkpoint_data_context(
        ckpt, dataset=args.dataset, setting=args.setting,
        cache_dir=args.cache_dir, expected_bit=bit)

    # ---- Build dataset splits (test + database) -------------------------
    _, test_set, db_set = load_dataset(
        dataset_root=args.dataset_root,
        dataset_name=args.dataset, setting=args.setting,
        load_train=False, load_test=True, load_database=True,
        cache_dir=args.cache_dir,
    )

    # ---- Build the exact method model + load all checkpoint weights ------
    device = args.device if torch.cuda.is_available() else "cpu"
    d_in = int(test_set.visual_global.shape[1])
    model = build_model_from_checkpoint_payload(
        ckpt, d_in=d_in, device=device, strict=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[extract_flat] loaded {ckpt['config'].get('method', 'legacy')} model "
          f"from {getattr(model, '_checkpoint_load_format', 'unknown')} "
          f"({n_params} params)")

    os.makedirs(args.out, exist_ok=True)

    for split, dset, out_name in [("query", test_set, "extract_query.npz"),
                                   ("db",    db_set,   "extract_db.npz")]:
        loader = DataLoader(dset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
        cont, bin_pm, lbls = _extract_codes(model, loader, device)
        bits01 = ((bin_pm + 1) // 2).astype(np.uint8)                       # {0,1}
        codebook_indices = pack_bits_to_indices(bits01, group_size=6)       # [N, M]
        base_indices     = pack_bits_to_indices(bits01, group_size=2)       # [N, R]
        # Image paths: dataset stores .paths attribute; CachedFeatureDataset
        # returns absolute-rooted relative paths like 'images/im00001.jpg'.
        # For compositional_eval path-matching we prepend the dataset folder.
        if hasattr(dset, "paths"):
            ds_root = os.path.join(args.dataset_root, args.dataset)
            image_paths = np.array(
                [os.path.join(ds_root, p) for p in dset.paths], dtype=np.str_,
            )
        else:
            image_paths = np.array([f"{args.dataset}:{i}" for i in range(len(dset))],
                                   dtype=np.str_)
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
