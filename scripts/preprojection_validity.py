#!/usr/bin/env python
"""Pre-projection bio-constraint validity: ours vs binary baselines vs chance.

Why this exists (2026-07-29). The paper's DNA framing was challenged as
decorative: GC / homopolymer are enforced only by the post-hoc DP projection
(`dna_utils/bio_constraints.py`), never by the training objective, and the same
projection is applied to baselines too. The naive capacity defence ("DNA has
4^L sub-codes vs binary 2^L") does NOT survive a matched-storage comparison:
36 bits <-> 18 bases is a bijection, so the alphabet buys nothing.

The defensible asymmetry is about REACHABILITY, not capacity. A model that
emits a per-position 4-way distribution over {A,C,G,T} has a natural handle on
GC and homopolymer runs; a model that emits 36 independent bits has to express
GC as a predicate over bit PAIRS and homopolymer runs as a predicate over
overlapping 8-bit windows. This script measures whether that asymmetry shows up
in the codes that actually get produced, BEFORE any projection:

    valid_frac = P(code satisfies GC window AND homopolymer <= max_run)

Reference points reported alongside:
    chance    = exact fraction of uniform 4^L sequences that are valid (DP)
    ours      = the model's emitted codes
    baselines = 36-bit hashes remapped 2 bits -> 1 base (the matched convention)

Usage:
  python scripts/preprojection_validity.py --out docs/preprojection_validity.json
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

from dna_utils import bio_constraints as bc   # noqa: E402


def valid_mask(base: np.ndarray, gc_lo: int, gc_hi: int, max_run: int):
    """Vectorised validity over [N, L] base ids. Returns (both, gc, hp)."""
    L = base.shape[1]
    gc = bc._IS_GC[base].sum(1)
    ok_gc = (gc >= gc_lo) & (gc <= gc_hi)
    run = np.ones(len(base), dtype=np.int32)
    mx = np.ones(len(base), dtype=np.int32)
    for i in range(1, L):
        same = base[:, i] == base[:, i - 1]
        run = np.where(same, run + 1, 1)
        mx = np.maximum(mx, run)
    ok_hp = mx <= max_run
    return ok_gc & ok_hp, ok_gc, ok_hp


def chance_valid_fraction(L: int, gc_lo: int, gc_hi: int, max_run: int) -> float:
    """EXACT fraction of the 4^L sequences that satisfy both constraints.

    DP over (position, gc_count, last_base, current_run_length). Counts are
    exact integers; the ratio is taken in float at the end.
    """
    from collections import defaultdict
    # state -> count.  state = (gc, last_base, run)
    cur = defaultdict(int)
    for b in range(4):
        cur[(int(bc._IS_GC[b]), b, 1)] += 1
    for _ in range(1, L):
        nxt = defaultdict(int)
        for (gc, lb, run), n in cur.items():
            for b in range(4):
                r = run + 1 if b == lb else 1
                if r > max_run:
                    continue
                g = gc + int(bc._IS_GC[b])
                if g > gc_hi:            # prune: GC can only grow
                    continue
                nxt[(g, b, r)] += n
        cur = nxt
    good = sum(n for (gc, _, _), n in cur.items() if gc_lo <= gc <= gc_hi)
    return good / float(4 ** L)


def hash_to_base(h2: np.ndarray) -> np.ndarray:
    """36-bit signed hash -> 18 bases, the repo's matched convention."""
    n = h2.shape[1] // 2
    b = (h2 > 0).astype(np.int64).reshape(len(h2), n, 2)
    return (b[:, :, 0] * 2 + b[:, :, 1]).astype(np.int8)


def load_codes(d: str):
    """Return [N, L] base ids from a result dir, or None."""
    for fn in ("extract_db.npz", "extract_train.npz"):
        p = os.path.join(d, fn)
        if not os.path.exists(p):
            continue
        z = np.load(p)
        if "base_indices" in z.files:
            return np.asarray(z["base_indices"]).astype(np.int8), fn
        if "hash_2bit" in z.files:
            return hash_to_base(np.asarray(z["hash_2bit"])), fn
    return None, None


OURS = {
    "Flickr25k": "result/260727+flickr25k_setting1_promptAblA_flickr_A_v4_P0refit_e4+bs+64+e+60+proj_lr+0.001",
    "MSCOCO":    "result/260724+mscoco_setting1_promptAblA_mscoco_A_v5b_P0refit_e39+bs+64+e+60+proj_lr+0.001",
    "NUSWIDE":   "result/260727+nuswide_setting1_promptAblA_nuswide_A_v4_P0refit_e4+bs+64+e+60+proj_lr+0.001",
    "CIFAR10":   "result/260724+cifar10_setting1_promptAblA_cifar_A_v4_P0refit_e19+bs+64+e+60+proj_lr+0.001",
}
# modern P0 matrix (raw-base E*), 36-bit, seed 42
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/preprojection_validity.json")
    ap.add_argument("--max_run", type=int, default=3)
    a = ap.parse_args()

    rows, chance_cache = [], {}
    for ds, odir in OURS.items():
        entries = [("Ours (A-champion)", odir)]
        for name, g in GLOBS.items():
            hits = sorted(glob.glob(g.format(ds=DS_KEY[ds])))
            if hits:
                entries.append((name, hits[0]))
        print(f"\n=== {ds} ===")
        for name, d in entries:
            base, src = load_codes(d)
            if base is None:
                print(f"  {name:20s} (no extraction)")
                continue
            L = base.shape[1]
            gc_lo, gc_hi = bc._resolve_gc_count_range(
                L, 0.40 if L == 18 else 0.416, 0.60 if L == 18 else 0.584)
            both, ok_gc, ok_hp = valid_mask(base, gc_lo, gc_hi, a.max_run)
            key = (L, gc_lo, gc_hi, a.max_run)
            if key not in chance_cache:
                chance_cache[key] = chance_valid_fraction(*key)
            rows.append({
                "dataset": ds, "method": name, "n": int(len(base)), "L": int(L),
                "gc_ok": float(ok_gc.mean()), "hp_ok": float(ok_hp.mean()),
                "valid_frac": float(both.mean()),
                "chance_valid_frac": chance_cache[key],
                "mean_gc": float(bc._IS_GC[base].sum(1).mean()),
                "source": os.path.join(d, src),
            })
            print(f"  {name:20s} valid={both.mean()*100:5.1f}%  "
                  f"(GC {ok_gc.mean()*100:5.1f}% / HP {ok_hp.mean()*100:5.1f}%)  "
                  f"meanGC={bc._IS_GC[base].sum(1).mean():.2f}  n={len(base)}")
        print(f"  {'[chance, uniform 4^L]':20s} "
              f"valid={chance_cache[(18, 8, 10, a.max_run)]*100:5.1f}%")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"note": "pre-projection validity; no DP projection applied",
               "max_run": a.max_run,
               "chance": {f"L{k[0]}_gc{k[1]}-{k[2]}_run{k[3]}": v
                          for k, v in chance_cache.items()},
               "cells": rows}, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
