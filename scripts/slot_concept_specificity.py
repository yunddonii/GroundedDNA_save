#!/usr/bin/env python
"""Is a slot's effect CONCEPT-SPECIFIC, or does it just move retrieval?

The slot-intervention experiment we report measures one number: perturb slot m
and see what mAP@R loses. Every informative slot loses something, so the result
cannot separate "slot m carries the colour axis" from "slot m carries 3 bases of
whatever". That is the gap TCAV (Kim, Wattenberg, Gilmer, Cai, Wexler, Viegas,
Sayres, ICML 2018) was built to close: it does not ask whether a direction
matters, it asks whether it matters SELECTIVELY for the concept it is supposed
to encode, and it establishes that against random concept sets rather than
against zero.

ADAPTATION. TCAV trains a linear probe to get a concept direction and takes a
directional derivative of a class logit. We have no classifier and no logits --
the deployed system is retrieval over discrete codes. So the transfer is:

  concept c            -> the set of test queries whose ground-truth labels
                          contain c (dataset tags, NOT our own captions)
  directional change   -> neutralise slot m's bases in query AND database codes
  sensitivity          -> the drop in mAP@R restricted to those queries

  specificity(m, c) = drop(m | queries with c) - drop(m | queries without c)

A slot that owns an axis should hurt its own concept's queries more than other
queries. A slot that merely carries generic capacity hurts both equally and
scores ~0.

SIGNIFICANCE, as in the paper. A positive specificity means nothing on its own,
because concept subsets differ in size and difficulty. The null is built by
drawing RANDOM query subsets of the same size and recomputing the statistic
--n_random times; we report the fraction of random draws the real value exceeds
and a two-sided empirical p-value. Concepts with fewer than --min_queries
positives are dropped, since the metric is unstable there.

Codes are read from the existing extraction; no retraining and no forward pass.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts"))

SLOTS = ["global", "primary_object", "secondary_object",
         "activity_relation", "color_texture"]


def _map_at_r(qcodes, dcodes, qlab, dlab, R, chunk=256):
    """Base-Hamming mAP@R, per query, CalcTopMap convention."""
    out = np.zeros(len(qcodes), dtype=np.float64)
    for i0 in range(0, len(qcodes), chunk):
        q = qcodes[i0:i0 + chunk]
        # Hamming over bases: count of differing positions
        d = (q[:, None, :] != dcodes[None, :, :]).sum(-1)         # [c, Nd]
        idx = np.argsort(d, axis=1, kind="stable")[:, :R]
        rel = (qlab[i0:i0 + chunk][:, None, :] * dlab[idx]).sum(-1) > 0  # [c, R]
        csum = np.cumsum(rel, axis=1)
        prec = csum / np.arange(1, R + 1)[None, :]
        denom = np.maximum(rel.sum(axis=1), 1)
        out[i0:i0 + chunk] = (prec * rel).sum(axis=1) / denom
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--concept_names", default=None,
                    help="optional text file, one label name per line, matching "
                         "the multi-hot column order")
    ap.add_argument("--n_queries", type=int, default=1000)
    ap.add_argument("--n_db", type=int, default=8000)
    ap.add_argument("--R", type=int, default=None)
    ap.add_argument("--min_queries", type=int, default=50)
    ap.add_argument("--n_random", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    d = a.result_dir
    if os.path.exists(os.path.join(d, "withids", "extract_query.npz")):
        d = os.path.join(d, "withids")
    qy = np.load(os.path.join(d, "extract_query.npz"), allow_pickle=True)
    db = np.load(os.path.join(d, "extract_db.npz"), allow_pickle=True)

    rng = np.random.default_rng(a.seed)
    qi = rng.permutation(len(qy["base_indices"]))[:a.n_queries]
    di = rng.permutation(len(db["base_indices"]))[:a.n_db]
    Q = qy["base_indices"][qi].astype(np.int8)
    D = db["base_indices"][di].astype(np.int8)
    QL = qy["multi_hot_labels"][qi].astype(np.float32)
    DL = db["multi_hot_labels"][di].astype(np.float32)
    R = a.R or (1000 if a.dataset.upper().startswith("CIFAR") else 5000)
    R = min(R, len(D))

    n_slots = int(qy["codebook_indices"].shape[1])
    n_bases = Q.shape[1] // n_slots
    names = None
    if a.concept_names and os.path.exists(a.concept_names):
        names = [x.strip() for x in open(a.concept_names) if x.strip()]

    base = _map_at_r(Q, D, QL, DL, R)                      # per-query baseline
    print(f"{a.dataset}  queries={len(Q)} db={len(D)} R={R} slots={n_slots} "
          f"bases/slot={n_bases}")
    print(f"  baseline mAP@R = {base.mean():.4f}\n")

    # concepts with enough positive queries
    cols = [c for c in range(QL.shape[1]) if int(QL[:, c].sum()) >= a.min_queries]
    if not cols:
        raise SystemExit(f"no label has >= {a.min_queries} positive queries")
    print(f"  concepts kept: {len(cols)} / {QL.shape[1]} "
          f"(>= {a.min_queries} positive queries)\n")

    rows, sig_hits = {}, 0
    print(f"  {'slot':20s}{'best concept':>16s}{'specificity':>13s}"
          f"{'p (2-sided)':>13s}   drop(with) / drop(without)")
    for m in range(n_slots):
        # neutralise slot m: set its bases to a constant in BOTH sides, which
        # removes the information without changing code length or the metric.
        Qm, Dm = Q.copy(), D.copy()
        sl = slice(m * n_bases, (m + 1) * n_bases)
        Qm[:, sl] = 0
        Dm[:, sl] = 0
        pert = _map_at_r(Qm, Dm, QL, DL, R)
        drop = base - pert                                  # per query, >0 = hurt

        best = None
        for c in cols:
            pos = QL[:, c] > 0
            if pos.sum() < a.min_queries or (~pos).sum() < a.min_queries:
                continue
            spec = float(drop[pos].mean() - drop[~pos].mean())
            # null: random query subsets of the SAME size
            k = int(pos.sum())
            null = np.empty(a.n_random)
            for t in range(a.n_random):
                sel = rng.permutation(len(drop))[:k]
                mask = np.zeros(len(drop), bool); mask[sel] = True
                null[t] = drop[mask].mean() - drop[~mask].mean()
            p = float((np.abs(null) >= abs(spec)).mean())
            if best is None or spec > best[0]:
                best = (spec, c, p, float(drop[pos].mean()), float(drop[~pos].mean()))
        if best is None:
            continue
        spec, c, p, dw, dwo = best
        cname = names[c] if names and c < len(names) else f"label{c}"
        sig = p < 0.05
        sig_hits += int(sig)
        name = SLOTS[m] if m < len(SLOTS) else str(m)
        print(f"  {name:20s}{cname:>16s}{spec:13.5f}{p:13.4f}"
              f"   {dw:+.4f} / {dwo:+.4f}{'  *' if sig else ''}")
        rows[name] = {"best_concept": cname, "specificity": round(spec, 6),
                      "p_two_sided": round(p, 4), "significant_at_05": bool(sig),
                      "drop_with_concept": round(dw, 6),
                      "drop_without_concept": round(dwo, 6),
                      "mean_drop_all_queries": round(float(drop.mean()), 6)}

    print(f"\n  slots with a significantly concept-specific effect: "
          f"{sig_hits} / {len(rows)}")
    print("  A slot that merely carries generic capacity hurts both groups")
    print("  equally and lands near zero regardless of how large its total drop is.")

    out = {"dataset": a.dataset, "dir": d, "R": R, "n_queries": len(Q),
           "n_db": len(D), "n_random": a.n_random,
           "baseline_map_at_R": round(float(base.mean()), 5),
           "significant_slots": sig_hits, "slots": rows}
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=2)
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
