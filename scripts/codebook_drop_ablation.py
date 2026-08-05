"""Codebook/slot distance-axis drop diagnostic.

Each codebook contributes 3 base positions (m*3, m*3+1, m*3+2) to the
final 18-position DNA code. "Dropping" codebook m replaces its 3 bases
with a fixed constant (0) across ALL images on both query and DB sides.

This neutralizes codebook m's discriminative contribution in Hamming distance.
The constant-masked rows are not treated as emitted DNA. Use ``--bio_project``
for paper-facing attribution in the deployed, valid-DNA metric space.

Usage:
    python codebook_drop_ablation.py --result_dir result/<tag>/
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dna_utils.dna_code_utils import base_hamming_distance
from scripts.codebook_drop_ablation_fast import codebook_slices, project_base_indices


def _ap_from_sorted_relevance(rel_sorted: np.ndarray) -> float:
    rel = rel_sorted.astype(np.float32)
    n_rel = rel.sum()
    if n_rel == 0:
        return 0.0
    cs = np.cumsum(rel)
    pos_at_k = cs / (np.arange(len(rel)) + 1.0)
    return float((pos_at_k * rel).sum() / n_rel)


def _relevance(q_lbl: np.ndarray, db_lbls: np.ndarray) -> np.ndarray:
    """Multi-label Jaccard > 0 -> 1, else 0. q_lbl [C], db_lbls [N, C]."""
    inter = (db_lbls * q_lbl[None, :]).sum(-1)
    return (inter > 0).astype(np.uint8)


def compute_mAP_pk(query_bi: np.ndarray, db_bi: np.ndarray,
                   q_labels: np.ndarray, db_labels: np.ndarray,
                   p_at_k_list=(1, 10, 100, 1000)) -> dict:
    """Return {mAP, P@k...} for given query/db base_indices + labels."""
    Nq = query_bi.shape[0]
    qt = torch.from_numpy(query_bi).long()
    dt = torch.from_numpy(db_bi).long()
    pk_acc = {k: 0.0 for k in p_at_k_list}
    aps = []
    for i in range(Nq):
        d = base_hamming_distance(qt[i:i+1], dt).numpy()[0]
        order = np.argsort(d, kind="stable")
        rel = _relevance(q_labels[i], db_labels)
        rel_sorted = rel[order]
        aps.append(_ap_from_sorted_relevance(rel_sorted))
        for k in p_at_k_list:
            pk_acc[k] += rel_sorted[:k].mean()
    out = {"mAP": float(np.mean(aps))}
    for k in p_at_k_list:
        out[f"P@{k}"] = float(pk_acc[k] / Nq)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", required=True)
    ap.add_argument("--num_codebooks", type=int, default=6)
    ap.add_argument("--bio_project", action="store_true")
    ap.add_argument("--gc_min_frac", type=float, default=None)
    ap.add_argument("--gc_max_frac", type=float, default=None)
    ap.add_argument("--max_run", type=int, default=3)
    args = ap.parse_args()
    rd = args.result_dir

    q = np.load(os.path.join(rd, "extract_query.npz"), allow_pickle=True)
    db = np.load(os.path.join(rd, "extract_db.npz"), allow_pickle=True)
    q_bi  = np.asarray(q["base_indices"]).copy()
    db_bi = np.asarray(db["base_indices"]).copy()
    gc_min_frac = args.gc_min_frac
    gc_max_frac = args.gc_max_frac
    if args.bio_project:
        if (gc_min_frac is None) != (gc_max_frac is None):
            raise ValueError(
                "--gc_min_frac and --gc_max_frac must be supplied together"
            )
        if gc_min_frac is None:
            if q_bi.shape[1] != 18:
                raise ValueError(
                    "non-18-base projection requires explicit --gc_min_frac "
                    "and --gc_max_frac"
                )
            gc_min_frac, gc_max_frac = 0.4444, 0.5556
        q_bi = project_base_indices(
            q_bi, gc_min_frac, gc_max_frac, args.max_run,
        )
        db_bi = project_base_indices(
            db_bi, gc_min_frac, gc_max_frac, args.max_run,
        )
    q_lbl = np.asarray(q["multi_hot_labels"])
    db_lbl = np.asarray(db["multi_hot_labels"])
    Nq, R = q_bi.shape
    M = int(args.num_codebooks)
    slices = codebook_slices(R, M)
    print(f"[drop_abl] R={R}, M={M}, Nq={Nq}, Ndb={db_bi.shape[0]}")

    # Baseline (no drop)
    print(f"[drop_abl] computing baseline (no drop) ...")
    base = compute_mAP_pk(q_bi, db_bi, q_lbl, db_lbl)
    print(f"  baseline: mAP={base['mAP']:.4f}  " +
          " ".join(f"P@{k}={base[f'P@{k}']:.4f}" for k in (1, 10, 100, 1000)))

    # Drop each codebook
    results = {
        "baseline": base,
        "drops": {},
        "input_code_space": (
            "bio_projected" if args.bio_project else "raw_unprojected"
        ),
        "drop_operator": "hamming_distance_axis_mask",
        "transformed_rows_are_emittable_dna": False,
        "bio_projection_applied": bool(args.bio_project),
        "paper_projection_compliant": False,
        "paper_result_eligible": False,
        "paper_eligibility_blockers": [
            "projection_protocol_manifest_not_bound"
        ],
        "bio_constraints": (
            {
                "gc_min_frac": gc_min_frac,
                "gc_max_frac": gc_max_frac,
                "max_run": args.max_run,
            }
            if args.bio_project else None
        ),
    }
    for m, codebook_slice in enumerate(slices):
        q_dr = q_bi.copy()
        db_dr = db_bi.copy()
        # Zero out positions m*3, m*3+1, m*3+2 on both query and db
        q_dr[:, codebook_slice] = 0
        db_dr[:, codebook_slice] = 0
        r = compute_mAP_pk(q_dr, db_dr, q_lbl, db_lbl)
        delta_mAP = r["mAP"] - base["mAP"]
        results["drops"][m] = {**r, "delta_mAP": delta_mAP}
        print(f"  drop cb{m}: mAP={r['mAP']:.4f} ΔmAP={delta_mAP:+.4f}  " +
              " ".join(f"P@{k}={r[f'P@{k}']:.4f}" for k in (1, 10, 100, 1000)))

    filename = (
        "codebook_drop_ablation_bioproj.json"
        if args.bio_project else "codebook_drop_ablation.json"
    )
    out_p = os.path.join(rd, filename)
    with open(out_p, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[drop_abl] saved -> {out_p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
