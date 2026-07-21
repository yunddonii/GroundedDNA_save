#!/usr/bin/env python
"""Post-hoc bio-projected mAP@R for one main-table cell.

The training scripts run their internal `-ev` final eval WITHOUT bio-projection
(evaluation() defaults bio_project=False when called from train_siglip2), so the
in-run JSON is the pre-projection number. The reported paper number is the
MANDATORY post-projection mAP@R (2026-07-21 invariant). This helper recomputes
it from the cell's extract_db.npz / extract_query.npz, applying the length-
dependent GC window:

  18-base (3-codon codebook): GC frac [0.40, 0.60]  -> count [8, 10]
  24-base (4-codon codebook): GC frac [0.416, 0.584] -> count [10, 14]

Emits a one-line [CELL-RESULT] marker (grep-able) and writes
evaluation_siglip2_base_bioproj.json into the result dir (evaluation() does the
write).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation_siglip2 import evaluation, resolve_map_at_r  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="result dir with extract_{db,query}.npz")
    ap.add_argument("--dataset", required=True,
                    help="canonical name: Flickr25k / MSCOCO / NUSWIDE / CIFAR10")
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--gc_min", type=float, default=0.40)
    ap.add_argument("--gc_max", type=float, default=0.60)
    args = ap.parse_args()

    R = resolve_map_at_r(args.dataset)
    res = evaluation(
        args.dir,
        distance_mode="base",
        codebook_size=args.K,
        bio_project=True,
        bio_gc_min_frac=args.gc_min,
        bio_gc_max_frac=args.gc_max,
        map_at_r=R,
    )
    map_r = res.get("mAP_at_R")
    pre = res.get("bio_stats", {}).get("mAP_pre_projection")
    print(
        f"[CELL-RESULT] dataset={args.dataset} K={args.K} "
        f"dir={os.path.basename(os.path.normpath(args.dir))} "
        f"mAP@R_bioproj={map_r if map_r is None else round(float(map_r), 4)} "
        f"full_mAP_bioproj={round(float(res['mAP']), 4)} "
        f"full_mAP_pre={None if pre is None else round(float(pre), 4)} "
        f"DNAunique_DB={round(float(res.get('unique_code_ratio', -1)), 4)} "
        f"gc=[{args.gc_min},{args.gc_max}]"
    )
    # also drop a tiny standalone marker file for easy aggregation later
    with open(os.path.join(args.dir, "cell_result.json"), "w") as f:
        json.dump({
            "dataset": args.dataset, "K": args.K,
            "mAP_at_R_bioproj": map_r,
            "full_mAP_bioproj": res["mAP"],
            "full_mAP_pre_projection": pre,
            "DNA_unique_DB": res.get("unique_code_ratio"),
            "gc_min_frac": args.gc_min, "gc_max_frac": args.gc_max,
            "map_r_cutoff": res.get("mAP_R_cutoff"),
        }, f, indent=2)


if __name__ == "__main__":
    main()
