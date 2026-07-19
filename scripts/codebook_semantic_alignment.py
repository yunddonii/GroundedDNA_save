#!/usr/bin/env python
"""Codebook-level semantic alignment: is the codebook a METRIC space?

Motivation
    `scripts/slot_role_analysis.py` (B) found that on MSCOCO the correlation
    between codeword distance and semantic distance is *lower* than the same
    correlation at the codon level -- impossible if codeword-cosine were a
    faithful semantic metric.  The cone/hubness explanation was tested and
    refuted (mean pairwise cosine is ~0 on all datasets, and centering changes
    nothing).

    This script measures the codebook geometry directly, without going through
    image pairs:

        rho_m = Spearman( 1 - cos(e_a, e_b) ,  || P_a - P_b || )

    over all active codeword pairs (a, b) in slot m, where P_a is the empirical
    label distribution of the images assigned to codeword a.  High rho means
    "codewords that sit far apart geometrically also mean different things",
    i.e. the codebook carries graded semantic structure.  Low rho means the
    codebook has collapsed into a set of near-categorical symbols with no usable
    metric between them, so `similar meaning -> similar codeword` cannot hold
    however well the model is trained.

    Reported alongside:
        eff_rank  participation ratio of the centred codebook's singular values
                  -- how many dimensions the codewords actually span
        cos_sd    spread of pairwise cosines; near-zero sd with mean ~0 means
                  mutually near-orthogonal codewords (no metric structure)

Diagnostic only.  It uses labels, so it must never be used for model selection.

    python scripts/codebook_semantic_alignment.py --dirs <result_dir> [...] \
        [--split db|train] [--min_support 20]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from scipy.stats import spearmanr


def load_codes(run_dir: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    """-> codebook_indices [N, M], multi_hot_labels [N, C]."""
    path = os.path.join(run_dir, f"extract_{split}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    d = np.load(path, allow_pickle=True)
    lab_key = "multi_hot_labels" if "multi_hot_labels" in d.files else "labels"
    lab = np.asarray(d[lab_key])
    if lab.ndim == 1:
        lab = np.eye(int(lab.max()) + 1, dtype=np.int64)[lab]
    return d["codebook_indices"], lab


def analyse(run_dir: str, split: str, min_support: int) -> dict:
    sd = torch.load(os.path.join(run_dir, "model_state_dict.pth"), map_location="cpu")
    cbs = sd["quantizer.codebooks"].float().numpy()          # [M_cb, K, D]
    cw, lab = load_codes(run_dir, split)
    n_slots = cw.shape[1]

    per_slot = []
    for m in range(n_slots):
        # a shared codebook (--share_codebook) stores one bank for all slots
        bank = cbs[m] if cbs.shape[0] > m else cbs[0]
        u, c = np.unique(cw[:, m], return_counts=True)
        active = u[c >= min_support]
        if len(active) < 10:
            per_slot.append({"slot": m, "n_active": int(len(active)), "rho": None})
            continue

        P = np.stack([lab[cw[:, m] == a].mean(0) for a in active])
        E = bank[active]
        En = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
        iu = np.triu_indices(len(active), 1)
        d_cos = (1.0 - En @ En.T)[iu]
        d_lab = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)[iu]

        s = np.linalg.svd(E - E.mean(0), compute_uv=False)
        p = s ** 2 / max((s ** 2).sum(), 1e-12)
        per_slot.append({
            "slot": m,
            "n_active": int(len(active)),
            "rho": float(spearmanr(d_cos, d_lab).statistic),
            "cos_mean": float((En @ En.T)[iu].mean()),
            "cos_sd": float((En @ En.T)[iu].std()),
            "eff_rank": float(1.0 / (p ** 2).sum()),
            "dim": int(E.shape[1]),
        })

    rhos = [s["rho"] for s in per_slot if s["rho"] is not None]
    return {
        "dir": os.path.basename(run_dir.rstrip("/")),
        "split": split,
        "min_support": min_support,
        "mean_rho": float(np.mean(rhos)) if rhos else None,
        "mean_eff_rank": float(np.mean([s["eff_rank"] for s in per_slot if s["rho"] is not None]))
                         if rhos else None,
        "mean_cos_sd": float(np.mean([s["cos_sd"] for s in per_slot if s["rho"] is not None]))
                       if rhos else None,
        "per_slot": per_slot,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="*", default=None,
                    help="short display name per dir (defaults to basename)")
    ap.add_argument("--split", default="db", choices=["db", "train", "query"],
                    help="db is the default: every run has it, so runs stay comparable")
    ap.add_argument("--min_support", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    names = args.labels or [os.path.basename(d.rstrip("/"))[:60] for d in args.dirs]
    results = []
    print(f"{'run':44s}{'rho':>8s}{'eff_rank':>10s}{'cos_sd':>9s}   per-slot rho")
    for d, nm in zip(args.dirs, names):
        try:
            r = analyse(d, args.split, args.min_support)
        except Exception as e:
            print(f"{nm:44s}  ERROR {type(e).__name__}: {e}")
            continue
        r["label"] = nm
        results.append(r)
        ps = " ".join(f"{s['rho']:+.3f}" if s["rho"] is not None else "  n/a"
                      for s in r["per_slot"])
        print(f"{nm:44s}{r['mean_rho']:+8.3f}{r['mean_eff_rank']:10.1f}"
              f"{r['mean_cos_sd']:9.3f}   {ps}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
