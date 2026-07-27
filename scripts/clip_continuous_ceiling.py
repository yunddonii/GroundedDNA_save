#!/usr/bin/env python
"""T1 — the frozen-CLIP continuous retrieval ceiling.

Computes mAP@R using the RAW frozen CLIP global embedding (cosine ranking, no
hashing, no quantization) on the official query/database splits, with the exact
same relevance rule and dataset cutoffs the paper uses (CIFAR10@1000, others
@5000; multi-hot relevance = share >= 1 label).

Why this matters: it separates "how much does our transform ADD to / SUBTRACT
from the frozen teacher" per dataset. Methods that are near-isometric
compressors of CLIP (CroVCA, SDC) should track this ceiling; a semantic
re-encoder (ours) can exceed it where the target is coarse categorical grouping
and fall below it where CLIP's raw geometry already ranks the ground truth.

Usage:
  python scripts/clip_continuous_ceiling.py --datasets Flickr25k MSCOCO NUSWIDE CIFAR10
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

from baseline.base_model import (  # noqa: E402
    CachedFeatureDataset, MAP_AT_R_BY_DATASET,
)

# Same cache per dataset as the baseline comparison table.
CACHE = {
    "CIFAR10":   "./cache/cifar10_clip",
    "Flickr25k": "./cache/flickr25k_clip_v4plus",
    "MSCOCO":    "./cache/mscoco_clip_v4plus",
    "NUSWIDE":   "./cache/nuswide_clip",
}


def _load(ds: str, setting: str, mode: str, root: str, cache: str):
    d = CachedFeatureDataset(ds, setting, mode, root, cache)
    # `visual_global` is the FULL cache array; `rows` maps split index -> cache
    # row. Indexing is mandatory or query/db would be the same 60K block.
    rows = np.asarray(d.rows, dtype=np.int64)
    feats = np.asarray(d.visual_global, dtype=np.float32)[rows]
    labels = np.asarray(d.labels, dtype=np.int64)
    assert feats.shape[0] == labels.shape[0] == len(d), (
        f"{ds}/{mode}: feats {feats.shape} labels {labels.shape} len {len(d)}")
    if labels.ndim == 1:                       # CIFAR ints -> one-hot
        n_cls = int(labels.max()) + 1
        oh = np.zeros((labels.shape[0], n_cls), dtype=np.int64)
        oh[np.arange(labels.shape[0]), labels] = 1
        labels = oh
    return feats, labels


def _map_at_r_cosine(qf, df, ql, dl, cutoff, device, q_chunk=256) -> float:
    """CalcTopMap-style mAP@R with cosine ranking on continuous features."""
    q = torch.from_numpy(qf).to(device)
    d = torch.from_numpy(df).to(device)
    q = torch.nn.functional.normalize(q, dim=1)
    d = torch.nn.functional.normalize(d, dim=1)
    qlt = torch.from_numpy(ql).to(device).float()
    dlt = torch.from_numpy(dl).to(device).float()
    R = int(cutoff) if cutoff else d.shape[0]
    R = min(R, d.shape[0])
    aps = []
    for s in range(0, q.shape[0], q_chunk):
        qs = q[s:s + q_chunk]
        sim = qs @ d.T                                   # [b, Nd] higher = closer
        idx = torch.argsort(-sim, dim=1, stable=True)[:, :R]
        rel = (qlt[s:s + q_chunk] @ dlt.T > 0).float()   # share >=1 label
        topk = torch.gather(rel, 1, idx)                 # [b, R]
        tsum = topk.sum(dim=1)
        pos = torch.arange(1, R + 1, device=device).float().unsqueeze(0)
        cum = torch.cumsum(topk, dim=1)
        prec = cum / pos
        ap = (prec * topk).sum(dim=1) / tsum.clamp_min(1e-12)
        ap[tsum == 0] = 0.0
        aps.append(ap.cpu())
    return float(torch.cat(aps).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["Flickr25k", "MSCOCO", "NUSWIDE", "CIFAR10"])
    ap.add_argument("--setting", default="setting1")
    ap.add_argument("--dataset_root", default=os.path.join(_REPO, "dataset"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="docs/clip_continuous_ceiling.json")
    a = ap.parse_args()

    dev = a.device if torch.cuda.is_available() else "cpu"
    rows = []
    print(f"{'dataset':11s}{'R':>7s}{'nQ':>8s}{'nDB':>9s}{'CLIP-cont mAP@R':>18s}")
    for ds in a.datasets:
        cache = CACHE[ds]
        qf, ql = _load(ds, a.setting, "test", a.dataset_root, cache)
        df, dl = _load(ds, a.setting, "database", a.dataset_root, cache)
        R = MAP_AT_R_BY_DATASET[ds]
        m = _map_at_r_cosine(qf, df, ql, dl, R, dev)
        rows.append({"dataset": ds, "map_at_R_clip_continuous": m,
                     "R": R, "n_query": int(qf.shape[0]),
                     "n_db": int(df.shape[0]), "cache": cache,
                     "feat_dim": int(qf.shape[1])})
        print(f"{ds:11s}{R:>7d}{qf.shape[0]:>8d}{df.shape[0]:>9d}{m:>18.4f}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"note": "raw frozen CLIP global cosine, no hashing; "
                       "relevance = share >=1 label; CalcTopMap mAP@R",
               "cells": rows}, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
