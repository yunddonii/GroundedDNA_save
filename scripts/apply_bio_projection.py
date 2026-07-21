#!/usr/bin/env python
"""Apply the biological-constraint post-projection to every method and re-evaluate.

The paper is about DNA hashing, so the emitted codes must be valid DNA. This
runs the mandatory post-processing step (GC in [40,60]%, homopolymer run <= 3;
Hamming-minimum DP projection of violators; valid codes untouched) on BOTH our
model and the baselines, in the SAME 18-base DNA space, and re-computes base
mAP@R. Baselines are the E*-matched DNA-space extractions from 2026-07-21.

For each (method, dataset) it reports:
    pre_compliance    fraction of DB codes already valid before projection
    mean_edit         mean Hamming edits applied (over all DB codes)
    base mAP@R pre    base Hamming mAP@R on the raw codes  (sanity vs prior)
    base mAP@R post   base Hamming mAP@R after projecting BOTH query and DB

Ours numbers must reproduce the prior base-space table pre-projection; that is
the correctness gate.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dna_utils.bio_constraints import (          # noqa: E402
    is_valid_batch, project_to_valid,
    DEFAULT_GC_MIN_FRAC, DEFAULT_GC_MAX_FRAC, DEFAULT_MAX_HOMOPOLYMER_RUN,
)


def project_batch_memoized(codes: np.ndarray, gc_min, gc_max, max_run) -> dict:
    """Same as batch_project_to_valid but projects each UNIQUE code once.

    The DP projection is a pure function of the code, so identical codes map to
    identical projections. Our codes have low DB-unique (0.19-0.40), so most
    rows are duplicates -- deduping is exact and turns the O(N) DP loop into
    O(#unique). For high-unique baselines it is a no-op speedup.
    """
    arr = np.asarray(codes, dtype=np.int8)
    N, L = arr.shape
    uniq, inv = np.unique(arr, axis=0, return_inverse=True)      # [U, L], [N]
    valid_u = is_valid_batch(uniq, gc_min, gc_max, max_run)      # [U]
    proj_u = uniq.copy()
    edit_u = np.zeros(len(uniq), dtype=np.int32)
    for i in range(len(uniq)):
        if valid_u[i]:
            continue
        p, cost = project_to_valid(uniq[i], gc_min, gc_max, max_run)
        proj_u[i] = p
        edit_u[i] = cost
    out = proj_u[inv]
    edits = edit_u[inv]
    new_valid = is_valid_batch(out, gc_min, gc_max, max_run)
    succ = edits[edits >= 0]
    return {
        "projected_codes": out,
        "edit_distances": edits,
        "compliance_rate": float(new_valid.mean()),
        "mean_edit_distance": float(succ.mean()) if succ.size else 0.0,
        "n_unique": int(len(uniq)),
    }

CUTOFF = {"Flickr25k": 5000, "MSCOCO": 5000, "NUSWIDE": 5000, "CIFAR10": 1000}


def base_from_hash(h2: np.ndarray) -> np.ndarray:
    """[N,36] sign/binary hash -> [N,18] base ids (00=A,01=C,10=G,11=T)."""
    b = (h2 > 0).astype(np.int64).reshape(len(h2), 18, 2)
    return b[:, :, 0] * 2 + b[:, :, 1]


def get_base(npz) -> np.ndarray:
    if "base_indices" in npz.files:
        return np.asarray(npz["base_indices"], dtype=np.int64)
    return base_from_hash(npz["hash_2bit"])


def map_at_r(qb: np.ndarray, dbb: np.ndarray, ql: np.ndarray, dl: np.ndarray,
             R: int, qchunk: int = 256) -> float:
    """base-Hamming mAP@R, GPU, query-chunked & vectorized. Jaccard>0 relevance.

    CalcTopMap convention: AP over the top-R retrieved, each query normalised by
    the number of relevant items found within R (0 if none in R).
    """
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    D = torch.tensor(dbb, dtype=torch.int8, device=dev)          # [Nd, L]
    DL = torch.tensor(dl, dtype=torch.float32, device=dev)       # [Nd, C]
    ranks = torch.arange(1, R + 1, device=dev, dtype=torch.float32)
    aps = []
    for s in range(0, len(qb), qchunk):
        Q = torch.tensor(qb[s:s + qchunk], dtype=torch.int8, device=dev)   # [q, L]
        QL = torch.tensor(ql[s:s + qchunk], dtype=torch.float32, device=dev)
        dist = (Q[:, None, :] != D[None, :, :]).sum(-1)          # [q, Nd]
        topk = torch.topk(dist, R, dim=1, largest=False, sorted=True).indices  # [q, R]
        rel = ((QL @ DL.T).gather(1, topk) > 0).float()          # [q, R]
        hits = torch.cumsum(rel, dim=1)
        npos = rel.sum(dim=1)
        ap = torch.where(npos > 0,
                         (hits / ranks * rel).sum(1) / npos.clamp_min(1e-9),
                         torch.zeros_like(npos))
        aps.append(ap.cpu().numpy())
    return float(np.concatenate(aps).mean())


def run_cell(name: str, ddir: str, dataset: str, gc_min, gc_max, max_run) -> dict:
    db = np.load(os.path.join(ddir, "extract_db.npz"), allow_pickle=True)
    qy = np.load(os.path.join(ddir, "extract_query.npz"), allow_pickle=True)
    dbb, qb = get_base(db), get_base(qy)
    dl = np.asarray(db["multi_hot_labels"], dtype=np.int64)
    ql = np.asarray(qy["multi_hot_labels"], dtype=np.int64)
    R = CUTOFF[dataset]

    pre_valid_db = is_valid_batch(dbb, gc_min, gc_max, max_run)
    pre_valid_qy = is_valid_batch(qb, gc_min, gc_max, max_run)
    map_pre = map_at_r(qb, dbb, ql, dl, R)

    proj_db = project_batch_memoized(dbb, gc_min, gc_max, max_run)
    proj_qy = project_batch_memoized(qb, gc_min, gc_max, max_run)
    dbb_p, qb_p = proj_db["projected_codes"], proj_qy["projected_codes"]
    map_post = map_at_r(qb_p, dbb_p, ql, dl, R)

    return {
        "name": name, "dataset": dataset, "dir": ddir,
        "pre_compliance_db": float(pre_valid_db.mean()),
        "pre_compliance_qy": float(pre_valid_qy.mean()),
        "post_compliance_db": float(proj_db["compliance_rate"]),
        "mean_edit_db": float(proj_db["mean_edit_distance"]),
        "mean_edit_qy": float(proj_qy["mean_edit_distance"]),
        "map_at_R_pre": map_pre,
        "map_at_R_post": map_post,
        "delta_proj": map_post - map_pre,
        "R": R,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/bio_projection_comparison.json")
    ap.add_argument("--gc_min", type=float, default=DEFAULT_GC_MIN_FRAC)
    ap.add_argument("--gc_max", type=float, default=DEFAULT_GC_MAX_FRAC)
    ap.add_argument("--max_run", type=int, default=DEFAULT_MAX_HOMOPOLYMER_RUN)
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these datasets")
    args = ap.parse_args()

    ours = {
        "Flickr25k": "result/260717+flickr25k_setting1_flickr_P0refit_e4+bs+64+e+60+proj_lr+0.001",
        "MSCOCO": "result/260717+mscoco_setting1_mscoco_P0refit_e49+bs+64+e+60+proj_lr+0.001",
        "NUSWIDE": "result/260717+nuswide_setting1_nuswide_P0refit_e4+bs+64+e+60+proj_lr+0.001",
        "CIFAR10": "result/260717+cifar10_setting1_cifar10_P0refit_e14+bs+64+e+60+proj_lr+0.001",
    }
    cells = [("Ours", d, ds) for ds, d in ours.items()]
    for m in ["cibhash", "cimon", "mls3rduh"]:
        for ds in ["Flickr25k", "MSCOCO", "NUSWIDE", "CIFAR10"]:
            g = glob.glob(f"result_baseline/260721/{m}_{ds}_clip_E*_dnaeval")
            if g:
                cells.append((m, g[0], ds))

    if args.only:
        want = set(args.only)
        cells = [c for c in cells if c[2] in want]

    results = []
    cfg = {"gc_min_frac": args.gc_min, "gc_max_frac": args.gc_max,
           "max_homopolymer_run": args.max_run}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    print(f"{'method':10s}{'dataset':10s}{'preCompl':>9s}{'meanEdit':>9s}"
          f"{'mAP@R pre':>11s}{'mAP@R post':>12s}{'d_proj':>9s}", flush=True)
    for name, ddir, ds in cells:
        r = run_cell(name, ddir, ds, args.gc_min, args.gc_max, args.max_run)
        results.append(r)
        print(f"{name:10s}{ds:10s}{r['pre_compliance_db']:>9.3f}{r['mean_edit_db']:>9.3f}"
              f"{r['map_at_R_pre']:>11.4f}{r['map_at_R_post']:>12.4f}{r['delta_proj']:>+9.4f}",
              flush=True)
        with open(args.out, "w") as f:              # incremental checkpoint
            json.dump({"config": cfg, "cells": results}, f, indent=2)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
