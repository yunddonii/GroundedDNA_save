#!/usr/bin/env python
"""Score every method in BOTH arenas: base-Hamming and bit-Hamming.

Fairness question (2026-07-29): GroundedDNA emits a 4-way softmax per base and
argmaxes it; the binary hashing baselines emit 36 independent sign() bits which
the DNA comparison then packs 2-bits-per-base. Base-Hamming is our native
distance and a *coarsening* of the distance the baselines actually optimised,
so a reviewer can object that the arena favours us.

The objection is answerable at zero training cost, because 36 bits <-> 18 bases
is a bijection and every extraction in this repo stores BOTH views
(`hash_2bit` and `base_indices`, verified consistent). So each method can be
scored in its own native metric as well as the other one:

    base : mAP@R under base Hamming   (ours native; the DNA arena)
    bit2 : mAP@R under bit  Hamming   (baselines native; the binary arena)

If GroundedDNA leads in BOTH columns, the "you picked your own distance"
objection has no measurement behind it.

Bio projection is applied exactly as the paper protocol requires (it rewrites
base_indices and rebuilds hash_2bit), so the two columns describe the same
deployed codes.

Usage:
  python scripts/dual_metric_arena.py --dataset Flickr25k \
      --out docs/dual_metric_arena_flickr.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

import evaluation_siglip2 as ev                                  # noqa: E402
from dna_utils import bio_constraints as bc                      # noqa: E402

CUTOFF = {"Flickr25k": 5000, "MSCOCO": 5000, "NUSWIDE": 5000, "CIFAR10": 1000}

OURS = {
    "Flickr25k": "result/260727+flickr25k_setting1_promptAblA_flickr_A_v4_P0refit_e4+bs+64+e+60+proj_lr+0.001",
    "MSCOCO":    "result/260724+mscoco_setting1_promptAblA_mscoco_A_v5b_P0refit_e39+bs+64+e+60+proj_lr+0.001",
    "NUSWIDE":   "result/260727+nuswide_setting1_promptAblA_nuswide_A_v4_P0refit_e4+bs+64+e+60+proj_lr+0.001",
    "CIFAR10":   "result/260724+cifar10_setting1_promptAblA_cifar_A_v4_P0refit_e19+bs+64+e+60+proj_lr+0.001",
}
GLOBS = {
    "CIBHash":    "result_baseline/p0_matrix_seeds42_legacy_cache/u0_cibhash_{ds}_36b_seed42/attempt_*/*/*_dnaeval",
    "CIMON":      "result_baseline/p0_matrix_seeds42_legacy_cache/u0_cimon_{ds}_36b_seed42/attempt_*/*/*_dnaeval",
    "OH":         "result_baseline/p0_matrix_seeds42_legacy_cache/u0_oh_{ds}_36b_seed42/attempt_*/*/*_dnaeval",
    "SDC":        "result_baseline/p0_matrix_seeds42_legacy_cache/u0_sdc-paper_{ds}_36b_seed42/attempt_*/*/*_dnaeval",
    "Bi-half":    "result_baseline/p0_matrix_seeds42_legacy_cache/u0_bihalf_{ds}_36b_seed42/attempt_*/*/*_dnaeval",
    "GreedyHash": "result_baseline/p0_matrix_seeds42_legacy_cache/u0_greedyhash_{ds}_36b_seed42/attempt_*/*/*_dnaeval",
    "HHCH":       "result_baseline/p0_matrix_seeds42_legacy_cache/u0_hhch_{ds}_36b_seed42/attempt_*/*/*_dnaeval",
    "MLS3RDUH":   "result_baseline/p0_matrix_seeds42_mls3rduh_paper_cache/u0_mls3rduh_{ds}_36b_seed42/attempt_*/*/*_dnaeval",
    "CroVCA":     "result_baseline/260722/crovca_{ds}_36b_P0refit_seed42_*_dnaeval",
}
DS_KEY = {"Flickr25k": "flickr25k", "MSCOCO": "mscoco",
          "NUSWIDE": "nuswide", "CIFAR10": "cifar10"}


def _bits_to_bases(h):
    b = (np.asarray(h) > 0).astype(np.int64)
    b = b.reshape(len(b), -1, 2)
    return (b[:, :, 0] * 2 + b[:, :, 1]).astype(np.int8)


def _bases_to_bits(base):
    base = np.asarray(base).astype(np.int64)
    hi, lo = base >> 1, base & 1
    return np.stack([hi, lo], -1).reshape(len(base), -1).astype(np.uint8)


def load_pair(d, project, gc_lo_f, gc_hi_f):
    """Return (query, db) dicts with consistent base_indices + hash_2bit."""
    out = []
    for fn in ("extract_query.npz", "extract_db.npz"):
        z = dict(np.load(os.path.join(d, fn), allow_pickle=True))
        if "base_indices" in z:
            base = np.asarray(z["base_indices"]).astype(np.int8)
        else:
            base = _bits_to_bases(z["hash_2bit"])
        if project:
            pr = bc.batch_project_to_valid(
                base, gc_min_frac=gc_lo_f, gc_max_frac=gc_hi_f, max_run=3,
                progress=False)
            if pr["compliance_rate"] < 1.0:
                raise SystemExit(f"{d}/{fn}: projection compliance "
                                 f"{pr['compliance_rate']:.4f} < 1.0")
            base = np.asarray(pr["projected_codes"]).astype(np.int8)
        lab = z["multi_hot_labels"] if "multi_hot_labels" in z else z["labels"]
        lab = np.asarray(lab)
        if lab.ndim == 1:
            lab = np.eye(int(lab.max()) + 1, dtype=np.int64)[lab]
        out.append({"base_indices": base, "hash_2bit": _bases_to_bits(base),
                    "multi_hot_labels": lab.astype(np.int64)})
    return out[0], out[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(CUTOFF))
    ap.add_argument("--out", required=True)
    ap.add_argument("--no_project", action="store_true")
    a = ap.parse_args()
    ds = a.dataset
    gc_lo_f, gc_hi_f = 0.40, 0.60
    project = not a.no_project

    entries = [("Ours (A-champion)", OURS[ds])]
    for name, g in GLOBS.items():
        hits = sorted(glob.glob(g.format(ds=DS_KEY[ds])))
        if hits:
            entries.append((name, hits[0]))

    rows = []
    print(f"=== {ds}  (mAP@{CUTOFF[ds]}, bio-projected={project}) ===")
    print(f"{'method':22s}{'base-Hamming':>14s}{'bit-Hamming':>13s}{'delta':>9s}")
    for name, d in entries:
        q, db = load_pair(d, project, gc_lo_f, gc_hi_f)
        got = {}
        for mode in ("base", "bit2"):
            r = ev.evaluate_retrieval(
                q, db, distance_mode=mode, map_at_r=CUTOFF[ds],
                multi_label_relevance_threshold=0.0)
            got[mode] = float(r["mAP_at_R"])
        rows.append({"dataset": ds, "method": name,
                     "base_mAP_at_R": got["base"], "bit_mAP_at_R": got["bit2"],
                     "dir": d})
        print(f"{name:22s}{got['base']:14.4f}{got['bit2']:13.4f}"
              f"{got['base']-got['bit2']:+9.4f}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"dataset": ds, "cutoff": CUTOFF[ds], "bio_projected": project,
               "note": "same deployed codes scored in both arenas; "
                       "36 bits <-> 18 bases is a bijection",
               "cells": rows}, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
