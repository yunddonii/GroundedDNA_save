#!/usr/bin/env python
"""Re-measure DRAFT section 4.6 (bio-projection effect) on the new model.

The draft's 4.6 table is flagged "historical diagnostic ... 현재 main claim에
인용 금지" because it came from legacy `bio_projection_18base.json` artifacts
with no protocol/checkpoint provenance. This recomputes every column directly
from the current runs' own extractions:

    pre mAP@R        base-Hamming on the raw emitted codes
    post mAP@R       after the mandatory minimum-Hamming DP projection
    mean DB edits    average base substitutions the projection applies
    validity pre     fraction already satisfying GC window + homopolymer <= 3
    validity post    must be 1.0 by construction (asserted)
    DNA-unique       pre -> post on the DB

Usage:
  python scripts/bioproj_effect_newmodel.py --out docs/newmodel_analysis/bioproj_effect.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

import evaluation_siglip2 as ev                       # noqa: E402
from dna_utils import bio_constraints as bc           # noqa: E402

CUTOFF = {"Flickr25k": 5000, "MSCOCO": 5000, "NUSWIDE": 5000, "CIFAR10": 1000}


def _labels(z):
    lab = z["multi_hot_labels"] if "multi_hot_labels" in z.files else z["labels"]
    lab = np.asarray(lab)
    if lab.ndim == 1:
        lab = np.eye(int(lab.max()) + 1, dtype=np.int64)[lab]
    return lab.astype(np.int64)


def _bits(base):
    base = np.asarray(base).astype(np.int64)
    return np.stack([base >> 1, base & 1], -1).reshape(len(base), -1).astype(np.uint8)


def _valid_frac(base, gc_lo, gc_hi, max_run=3):
    gc = bc._IS_GC[base].sum(1)
    ok = (gc >= gc_lo) & (gc <= gc_hi)
    run = np.ones(len(base), dtype=np.int32); mx = np.ones(len(base), dtype=np.int32)
    for i in range(1, base.shape[1]):
        same = base[:, i] == base[:, i - 1]
        run = np.where(same, run + 1, 1); mx = np.maximum(mx, run)
    return float((ok & (mx <= max_run)).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", required=True,
                    help="DATASET=result_dir pairs")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = []
    print(f"{'dataset':10s}{'pre mAP':>9s}{'post mAP':>10s}{'Δ':>9s}"
          f"{'edits':>8s}{'valid pre':>10s}{'valid post':>11s}{'uniq pre→post':>18s}")
    for spec in a.cells:
        ds, d = spec.split("=", 1)
        q = np.load(os.path.join(d, "extract_query.npz"), allow_pickle=True)
        db = np.load(os.path.join(d, "extract_db.npz"), allow_pickle=True)
        qb = np.asarray(q["base_indices"]).astype(np.int8)
        bb = np.asarray(db["base_indices"]).astype(np.int8)
        L = bb.shape[1]
        gmin, gmax = (0.40, 0.60) if L == 18 else (0.416, 0.584)
        gc_lo, gc_hi = bc._resolve_gc_count_range(L, gmin, gmax)

        pr_q = bc.batch_project_to_valid(qb, gc_min_frac=gmin, gc_max_frac=gmax,
                                         max_run=3, progress=False)
        pr_b = bc.batch_project_to_valid(bb, gc_min_frac=gmin, gc_max_frac=gmax,
                                         max_run=3, progress=False)
        assert pr_q["compliance_rate"] == 1.0 and pr_b["compliance_rate"] == 1.0, \
            f"{ds}: projection did not reach full compliance"
        qp = np.asarray(pr_q["projected_codes"]).astype(np.int8)
        bp = np.asarray(pr_b["projected_codes"]).astype(np.int8)

        ql, bl = _labels(q), _labels(db)
        def _map(qa, ba):
            r = ev.evaluate_retrieval(
                {"base_indices": qa, "hash_2bit": _bits(qa), "multi_hot_labels": ql},
                {"base_indices": ba, "hash_2bit": _bits(ba), "multi_hot_labels": bl},
                distance_mode="base", map_at_r=CUTOFF[ds],
                multi_label_relevance_threshold=0.0)
            return float(r["mAP_at_R"])
        pre, post = _map(qb, bb), _map(qp, bp)
        u_pre = len({tuple(r) for r in bb.tolist()}) / len(bb)
        u_post = len({tuple(r) for r in bp.tolist()}) / len(bp)
        row = dict(dataset=ds, dir=d, L=int(L),
                   pre_mAP_at_R=pre, post_mAP_at_R=post, delta=post - pre,
                   mean_db_edits=float(pr_b["mean_edit_distance"]),
                   valid_frac_pre=_valid_frac(bb, gc_lo, gc_hi),
                   valid_frac_post=_valid_frac(bp, gc_lo, gc_hi),
                   dna_unique_pre=u_pre, dna_unique_post=u_post)
        rows.append(row)
        print(f"{ds:10s}{pre:9.4f}{post:10.4f}{post-pre:+9.4f}"
              f"{row['mean_db_edits']:8.3f}{row['valid_frac_pre']*100:9.1f}%"
              f"{row['valid_frac_post']*100:10.1f}%   {u_pre:.4f} → {u_post:.4f}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"note": "recomputed from each run's own extraction; "
                       "supersedes the legacy bio_projection_18base.json rows",
               "cells": rows}, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
