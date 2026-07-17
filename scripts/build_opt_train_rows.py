"""Emit the cache row indices of the optimization-train split (P0 protocol).

The P0 validation protocol carves `--val_split_ratio` of the TRAIN split out as
the held-out validation query set. Any statistic fitted before training -- most
importantly the partial-whitening matrix (`text_whiten.npz`) -- must be
estimated on the optimization-train rows ONLY, otherwise it observes either the
official test split (a transductive leak into the reported number) or the
validation rows (which would contaminate epoch selection).

This script reproduces the exact split performed in `train_siglip2.py`
(same seed, same stratification rule) and writes the corresponding CACHE row
indices, so `build_text_whiten_matrix.py --row_index_npy` can restrict the fit.

Usage:
    python scripts/build_opt_train_rows.py \
        --dataset Flickr25k --dataset_dir dataset \
        --cache_dir cache/flickr25k_clip_v4plus_qwen3_tokens \
        --val_split_ratio 0.1 --val_split_seed 42 \
        --out cache/flickr25k_clip_v4plus_qwen3_tokens/opt_train_rows.npy
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloaders import DATASET
from val_split import cache_rows_for_dataset, carve_val_indices, get_train_labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--dataset_dir", default="dataset")
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--setting", default="setting1")
    ap.add_argument("--val_split_ratio", type=float, default=None,
                    help="Held-out fraction. Omit with --all_train.")
    ap.add_argument("--val_split_seed", type=int, default=42)
    ap.add_argument("--all_train", action="store_true",
                    help="Emit EVERY train row instead of carving a val split. "
                         "For the P0 stage-2 refit, which trains on 100%% of "
                         "train -- the whitening must then see all train rows "
                         "(but still never test/database).")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if not args.all_train and args.val_split_ratio is None:
        ap.error("--val_split_ratio is required unless --all_train is given")

    root = os.path.join(args.dataset_dir, args.dataset)
    ds = DATASET[args.dataset](root, None, None, "train", args.setting,
                               siglip2_feature_cache_dir=args.cache_dir)

    # ---- identical carve-out to train_siglip2.py (shared definition) -------
    if args.all_train:
        n_train = len(get_train_labels(ds))
        opt_idx = np.arange(n_train)
        val_idx = np.array([], dtype=np.int64)
        strat = "ALL train rows (stage-2 refit; no val carve-out)"
    else:
        opt_idx, val_idx, strat = carve_val_indices(
            get_train_labels(ds), ratio=args.val_split_ratio, seed=args.val_split_seed,
        )
        n_train = len(opt_idx) + len(val_idx)

    # ---- dataset index -> cache row ---------------------------------------
    row_map = cache_rows_for_dataset(ds, root)
    opt_rows = np.sort(np.unique(row_map[opt_idx]))
    val_rows = np.unique(row_map[val_idx])
    leaked = np.intersect1d(opt_rows, val_rows)
    if len(leaked):
        raise SystemExit(
            f"[opt-rows] ABORT: {len(leaked)} cache rows are shared between "
            f"opt-train and val (duplicate images across split rows). Fitting "
            f"the whitening on these would contaminate epoch selection."
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.save(args.out, opt_rows.astype(np.int64))
    print(f"[opt-rows] {args.dataset}: train {n_train} -> opt-train {len(opt_idx)} "
          f"+ val {len(val_idx)} ({strat}, seed={args.val_split_seed})")
    print(f"[opt-rows] wrote {len(opt_rows)} cache rows -> {args.out}")


if __name__ == "__main__":
    main()
