#!/usr/bin/env python
"""Post-hoc bio-projected mAP@R for one main-table cell.

The training scripts run their internal `-ev` final eval WITHOUT bio-projection
(evaluation() defaults bio_project=False when called from train_siglip2), so the
in-run JSON is the pre-projection number. The reported paper number is the
MANDATORY post-projection mAP@R (2026-07-21 invariant). This helper recomputes
it from the cell's extract_db.npz / extract_query.npz, applying the length-
dependent GC window:

F07/Phase 2: the window is NOT a free parameter. It comes from
`dna_utils.gc_policy` (40-60% inclusive), keyed by the code length read from the
extraction itself:

  15 bases -> count [6, 9]     18 bases -> count [8, 10]
  20 bases -> count [8, 12]    24 bases -> count [10, 14]

`--gc_min/--gc_max` are now optional overrides that are CHECKED against that
policy rather than trusted. The historical `--gc_min 0.416 --gc_max 0.584`,
carried over from the 24-base panel, yields count [7, 8] at 15 bases -- a
different, narrower feasible set than the policy's [6, 9], so results projected
under it are not comparable with results projected under the policy.

Emits a one-line [CELL-RESULT] marker (grep-able) and writes
evaluation_siglip2_base_bioproj.json into the result dir (evaluation() does the
write).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

from dna_utils.bio_constraints import _resolve_gc_count_range  # noqa: E402
from dna_utils.gc_policy import resolve_gc_policy  # noqa: E402
from evaluation_siglip2 import evaluation, resolve_map_at_r  # noqa: E402


def _code_length(result_dir: str) -> int:
    """Bases per code, read from the extraction rather than assumed."""
    path = os.path.join(result_dir, "extract_db.npz")
    with np.load(path, allow_pickle=False) as payload:
        for key in ("base_indices", "codes", "dna_codes"):
            if key in payload:
                return int(np.asarray(payload[key]).shape[1])
    raise SystemExit(f"[eval_cell_bioproj] no base-code array in {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="result dir with extract_{db,query}.npz")
    ap.add_argument("--dataset", required=True,
                    help="canonical name: Flickr25k / MSCOCO / NUSWIDE / CIFAR10")
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--gc_min", type=float, default=None,
                    help="Optional; checked against the central GC policy.")
    ap.add_argument("--gc_max", type=float, default=None,
                    help="Optional; checked against the central GC policy.")
    ap.add_argument("--allow-backfilled", action="store_true",
                    help=("Admit retrospectively bound inputs. Off by default "
                          "so a paper path cannot pick them up silently."))
    ap.add_argument("--bases", type=int, default=None,
                    help="Code length; read from extract_db.npz when omitted.")
    args = ap.parse_args()

    total_bases = args.bases if args.bases else _code_length(args.dir)
    policy = resolve_gc_policy(total_bases)
    gc_min = 0.40 if args.gc_min is None else float(args.gc_min)
    gc_max = 0.60 if args.gc_max is None else float(args.gc_max)
    got = _resolve_gc_count_range(total_bases, gc_min, gc_max)
    if got != (policy.gc_min_count, policy.gc_max_count):
        raise SystemExit(
            f"[eval_cell_bioproj] gc=[{gc_min}, {gc_max}] gives count {got} at "
            f"{total_bases} bases, but the central policy "
            f"({policy.policy_version}) is "
            f"[{policy.gc_min_count}, {policy.gc_max_count}]. Projecting onto a "
            f"window nobody reviewed is how four conventions ended up in the "
            f"codebase; fix the caller rather than this check.")
    args.gc_min, args.gc_max = gc_min, gc_max

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
    # Validate the extraction BEFORE trusting anything computed from it, then
    # seal the numbers and their binding into one file the aggregator reads.
    from dna_utils.extraction_validation import (
        metric_input_binding, write_analysis_marker)
    from dna_utils.runtime_state import sha256_file as _sha
    binding = metric_input_binding(
        args.dir, allow_backfilled=args.allow_backfilled)

    # also drop a tiny standalone marker file for easy aggregation later
    with open(os.path.join(args.dir, "cell_result.json"), "w") as f:
        json.dump({
            "input_binding": binding,
            "dataset": args.dataset, "K": args.K,
            "mAP_at_R_bioproj": map_r,
            "full_mAP_bioproj": res["mAP"],
            "full_mAP_pre_projection": pre,
            "DNA_unique_DB": res.get("unique_code_ratio"),
            "gc_min_frac": args.gc_min, "gc_max_frac": args.gc_max,
            "total_bases": total_bases,
            "gc_count_min_inclusive": policy.gc_min_count,
            "gc_count_max_inclusive": policy.gc_max_count,
            "gc_policy_version": policy.policy_version,
            "map_r_cutoff": res.get("mAP_R_cutoff"),
        }, f, indent=2)

    # One canonical, sealed result. The aggregator reads THIS -- previously it
    # validated cell_result.json and then took its numbers from an unsigned
    # third file, so a planted mAP@R=999 was reported verbatim.
    write_analysis_marker(
        args.dir,
        metrics={
            "map_at_R_bioproj": map_r,
            "full_map_bioproj": res["mAP"],
            "full_map_pre_projection": pre,
            "dna_unique_db": res.get("unique_code_ratio"),
        },
        binding=binding,
        protocol={
            "dataset": args.dataset,
            "codebook_size": args.K,
            "map_r_cutoff": res.get("mAP_R_cutoff"),
            "total_bases": total_bases,
            "gc_policy_version": policy.policy_version,
            "gc_count_min_inclusive": policy.gc_min_count,
            "gc_count_max_inclusive": policy.gc_max_count,
            "gc_min_frac": args.gc_min,
            "gc_max_frac": args.gc_max,
        },
        sources={
            "eval_cell_bioproj_sha256": _sha(os.path.abspath(__file__)),
            "evaluation_siglip2_sha256": _sha(
                os.path.join(os.path.dirname(os.path.dirname(
                    os.path.abspath(__file__))), "evaluation_siglip2.py")),
        })


if __name__ == "__main__":
    main()
