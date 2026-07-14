"""Recompute paper-standard mAP@R for existing result dirs from saved extractions.

Paper cutoffs (deep-hashing CalcTopMap convention):
    CIFAR10 @1000 | NUS-WIDE @5000 | MS-COCO @5000 | Flickr25k @5000

For OUR model dirs: reads extract_db.npz + extract_query.npz (base_indices +
multi_hot_labels), recomputes mAP + mAP@R, and writes mAP_at_R / mAP_R_cutoff
back into evaluation_siglip2_base.json.

For BASELINE dirs (result_baseline/.../<method>_.../): same npz layout.

Usage:
    python scripts/recompute_map_at_r.py <result_dir> [<result_dir> ...]
    python scripts/recompute_map_at_r.py --glob 'result/260713*nuswide*'
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from evaluation_siglip2 import evaluate_retrieval, resolve_map_at_r, MAP_AT_R_BY_DATASET


def _infer_dataset(path: str) -> str | None:
    p = path.lower()
    if "cifar10" in p:   return "CIFAR10"
    if "nuswide" in p:   return "NUSWIDE"
    if "mscoco" in p:    return "MSCOCO"
    if "flickr25k" in p or "flickr" in p: return "Flickr25k"
    return None


def process(d: str, dataset: str | None = None, write: bool = True) -> dict | None:
    dbp = os.path.join(d, "extract_db.npz")
    qyp = os.path.join(d, "extract_query.npz")
    if not (os.path.exists(dbp) and os.path.exists(qyp)):
        print(f"  SKIP (no npz): {d}")
        return None
    ds = dataset or _infer_dataset(d)
    R = resolve_map_at_r(ds)
    if R is None:
        print(f"  SKIP (unknown dataset -> no R): {d}")
        return None
    db = dict(np.load(dbp, allow_pickle=True))
    qy = dict(np.load(qyp, allow_pickle=True))
    res = evaluate_retrieval(qy, db, distance_mode="base",
                             precision_at_k_list=(1, 10, 100, 1000),
                             remove_self_match=False, map_at_r=R)
    mAP, mAPr = res["mAP"], res["mAP_at_R"]
    print(f"  {ds:9s} R={R:5d}  mAP(full)={mAP:.4f}  mAP@{R}={mAPr:.4f}  | {os.path.basename(d)[:60]}")
    if write:
        jp = os.path.join(d, "evaluation_siglip2_base.json")
        j = json.load(open(jp)) if os.path.exists(jp) else {}
        j["mAP"] = mAP
        j["mAP_at_R"] = mAPr
        j["mAP_R_cutoff"] = R
        j["precision_at_k"] = {str(k): v for k, v in res["precision_at_k"].items()}
        j["recall_at_k"] = {str(k): v for k, v in res["recall_at_k"].items()}
        json.dump(j, open(jp, "w"), indent=2)
    return {"dir": d, "dataset": ds, "R": R, "mAP": mAP, "mAP_at_R": mAPr}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*")
    ap.add_argument("--glob", default=None)
    ap.add_argument("--dataset", default=None, help="force dataset (else inferred from path)")
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()
    dirs = list(a.dirs)
    if a.glob:
        dirs += sorted(glob.glob(a.glob))
    if not dirs:
        print("no dirs given"); return 1
    print(f"Paper mAP@R cutoffs: {MAP_AT_R_BY_DATASET}")
    out = []
    for d in dirs:
        r = process(d, dataset=a.dataset, write=not a.no_write)
        if r: out.append(r)
    print(f"\ndone: {len(out)} dirs updated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
