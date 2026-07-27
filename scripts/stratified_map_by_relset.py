#!/usr/bin/env python
"""(c) Stratified mAP@R by |relevant set| — locate WHERE the MSCOCO gap lives.

Hypothesis under test: our deficit vs CroVCA/SDC on MSCOCO is concentrated in
the rare-concept tail (queries with few relevant items), because those queries
need fine rank resolution at shallow DB depth, which our low-diversity
compositional code (DNA-unique 0.19 = 5.3 DB items per distinct code) cannot
provide — while CroVCA/SDC are pure resolution maximisers.

Splits the query set into quartiles of |relevant set| and reports mAP@R inside
each quartile, per method, on the SAME queries. Also reports a cutoff sweep.
All post-hoc on saved codes; bio-projected base-Hamming throughout.

Usage:
  python scripts/stratified_map_by_relset.py --dataset MSCOCO
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from dna_utils.bio_constraints import batch_project_to_valid  # noqa: E402

MAP_R = {"CIFAR10": 1000, "NUSWIDE": 5000, "MSCOCO": 5000, "Flickr25k": 5000}


def _project(bases, gc_min, gc_max, max_run):
    """Memoised bio-projection (project unique rows only)."""
    uniq, inv = np.unique(np.asarray(bases), axis=0, return_inverse=True)
    out = batch_project_to_valid(uniq, gc_min, gc_max, max_run, progress=False)
    return out["projected_codes"].astype(np.int64)[inv]


def _load(path, project, gc):
    z = np.load(path, allow_pickle=True)
    b = z["base_indices"].astype(np.int64)
    l = z["multi_hot_labels"].astype(np.int64)
    if project:
        b = _project(b, gc[0], gc[1], 3)
    return b, l


def _ap_per_query(qb, db, ql, dl, cutoff, device, chunk=128):
    """Per-query AP@cutoff with base-Hamming ranking. Returns [Nq] float array."""
    q = torch.from_numpy(qb).to(device)
    d = torch.from_numpy(db).to(device)
    qlt = torch.from_numpy(ql).to(device).float()
    dlt = torch.from_numpy(dl).to(device).float()
    R = min(int(cutoff), d.shape[0])
    aps = []
    for s in range(0, q.shape[0], chunk):
        qs = q[s:s + chunk]
        # base Hamming = # differing positions
        dist = (qs.unsqueeze(1) != d.unsqueeze(0)).sum(dim=2)      # [b, Nd]
        idx = torch.argsort(dist, dim=1, stable=True)[:, :R]
        rel = (qlt[s:s + chunk] @ dlt.T > 0).float()
        topk = torch.gather(rel, 1, idx)
        tsum = topk.sum(dim=1)
        pos = torch.arange(1, R + 1, device=device).float().unsqueeze(0)
        ap = ((torch.cumsum(topk, 1) / pos) * topk).sum(1) / tsum.clamp_min(1e-12)
        ap[tsum == 0] = 0.0
        aps.append(ap.cpu())
    return torch.cat(aps).numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="MSCOCO")
    ap.add_argument("--ours_dir", required=True)
    ap.add_argument("--baseline_dirs", nargs="*", default=[],
                    help="dirs with extract_{db,query}_bioproj.npz or extract_*.npz")
    ap.add_argument("--baseline_names", nargs="*", default=[])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--gc", nargs=2, type=float, default=[0.40, 0.60])
    ap.add_argument("--out", default="docs/stratified_map_mscoco.json")
    a = ap.parse_args()

    dev = a.device if torch.cuda.is_available() else "cpu"
    R = MAP_R[a.dataset]

    methods = {}
    # ours: raw extractions -> project here
    qb, ql = _load(os.path.join(a.ours_dir, "extract_query.npz"), True, a.gc)
    db, dl = _load(os.path.join(a.ours_dir, "extract_db.npz"), True, a.gc)
    methods["Ours"] = (qb, db)
    ref_ql, ref_dl = ql, dl

    for i, d in enumerate(a.baseline_dirs):
        name = a.baseline_names[i] if i < len(a.baseline_names) else os.path.basename(d)[:12]
        qp = os.path.join(d, "extract_query_bioproj.npz")
        dp = os.path.join(d, "extract_db_bioproj.npz")
        proj = False
        if not os.path.exists(qp):
            qp, dp, proj = (os.path.join(d, "extract_query.npz"),
                            os.path.join(d, "extract_db.npz"), True)
        bq, bql = _load(qp, proj, a.gc)
        bd, bdl = _load(dp, proj, a.gc)
        assert bql.shape == ref_ql.shape and bdl.shape == ref_dl.shape, \
            f"{name}: label shape mismatch vs ours"
        methods[name] = (bq, bd)

    # |relevant set| per query (identical across methods -> use ours' labels)
    relcnt = ((ref_ql.astype(np.float32) @ ref_dl.astype(np.float32).T) > 0).sum(1)
    qs = np.quantile(relcnt, [0.25, 0.5, 0.75])
    strata = np.digitize(relcnt, qs)            # 0..3
    print(f"[{a.dataset}] |relevant| quartile edges: {qs.astype(int)}  "
          f"min={relcnt.min()} max={relcnt.max()}")

    res = {"dataset": a.dataset, "R": R, "quartile_edges": qs.tolist(),
           "n_query": int(len(relcnt)), "methods": {}}
    ap_cache = {}
    for name, (mq, md) in methods.items():
        aps = _ap_per_query(mq, md, ref_ql, ref_dl, R, dev)
        ap_cache[name] = aps
        row = {"overall_mAP_at_R": float(aps.mean())}
        for k in range(4):
            m = strata == k
            row[f"Q{k+1}"] = float(aps[m].mean())
            row[f"Q{k+1}_n"] = int(m.sum())
            row[f"Q{k+1}_relmedian"] = float(np.median(relcnt[m]))
        res["methods"][name] = row

    print(f"\n{'method':10s}{'overall':>9s}{'Q1(rare)':>10s}{'Q2':>9s}{'Q3':>9s}{'Q4(common)':>12s}")
    for name, row in res["methods"].items():
        print(f"{name:10s}{row['overall_mAP_at_R']:9.4f}{row['Q1']:10.4f}"
              f"{row['Q2']:9.4f}{row['Q3']:9.4f}{row['Q4']:12.4f}")
    if "Ours" in ap_cache:
        print("\ngap vs Ours (positive = baseline better):")
        for name in res["methods"]:
            if name == "Ours":
                continue
            d = {f"Q{k+1}": res["methods"][name][f"Q{k+1}"] - res["methods"]["Ours"][f"Q{k+1}"]
                 for k in range(4)}
            d["overall"] = (res["methods"][name]["overall_mAP_at_R"]
                            - res["methods"]["Ours"]["overall_mAP_at_R"])
            res["methods"][name]["gap_vs_ours"] = d
            print(f"  {name:10s} overall {d['overall']:+.4f} | "
                  + " ".join(f"Q{k+1} {d[f'Q{k+1}']:+.4f}" for k in range(4)))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
