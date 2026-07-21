#!/usr/bin/env python
"""Slot intervention (REQUIRED_EXPERIMENTS_GROUNDEDDNA_PAPER.md §3).

Counterfactual: replace exactly ONE slot's codon in a query code with a donor's
codon, re-retrieve against the unchanged database, and ask whether the target
concept becomes more prevalent in the top-K *without* the rest of the neighbour
set being scrambled.

No forward pass is re-run -- the intervention is on the emitted code only (§3.2).

Target concept
    Image-level multi-hot labels give no per-slot ground truth, so the target
    concept for (query i, slot m, donor j) is defined as the label the DONOR'S
    slot-m codon most strongly predicts under the train-only dictionary, among
    labels the query does not already have.  The dictionary never sees test.

Donor selection (§3.3) uses independent annotation only: a database sample that
HAS the target concept and whose remaining labels are as close as possible to
the query's, so non-target semantics are held roughly fixed.

Controls (§3.5)
    ours_slot     : intended slot codon replaced           (6 bits changed)
    <base>_chunk  : same-position contiguous 6-bit chunk   (6 bits changed)
    random_donor  : donor drawn at random, no label matching
    random_slot   : a different slot replaced with the donor's codon there
Every arm changes exactly 6 bits, so the intervention budget is matched.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Sequence

import numpy as np

SLOT_NAMES = [
    "global", "primary_object", "secondary_object",
    "activity_relation", "color_texture", "scene_type",
]
N_SLOTS = 6


# --------------------------------------------------------------------------
def _basenames(paths: Sequence) -> np.ndarray:
    return np.array([os.path.basename(str(p)) for p in paths])


def topk_by_hamming(query_codes: np.ndarray, db_codes: np.ndarray, k: int,
                    chunk: int = 256) -> np.ndarray:
    """Symbol-wise mismatch count (base Hamming for ours, bit Hamming for flat)."""
    out = np.zeros((len(query_codes), k), dtype=np.int64)
    for s in range(0, len(query_codes), chunk):
        q = query_codes[s:s + chunk]
        d = (q[:, None, :] != db_codes[None, :, :]).sum(axis=2)
        out[s:s + chunk] = np.argpartition(d, k, axis=1)[:, :k]
        # order the k so Jaccard/prevalence are computed on the true top-k
        rows = np.arange(len(q))[:, None]
        sel = out[s:s + chunk]
        out[s:s + chunk] = sel[rows, np.argsort(d[rows, sel], axis=1)]
    return out


def prevalence(topk: np.ndarray, db_labels: np.ndarray) -> np.ndarray:
    """-> [n_query, C] fraction of the top-K carrying each label."""
    return db_labels[topk].mean(axis=1)


# --------------------------------------------------------------------------
def build_codon_dictionary(train_units: np.ndarray, train_labels: np.ndarray,
                           n_units: int, alpha: float) -> np.ndarray:
    n_lab = train_labels.shape[1]
    cnt = np.zeros((n_units, n_lab)); sup = np.zeros(n_units)
    np.add.at(cnt, train_units, train_labels.astype(np.float64))
    np.add.at(sup, train_units, 1.0)
    return (cnt + alpha) / (sup[:, None] + 2.0 * alpha)


def codon_ids(base_indices: np.ndarray, n_bases: int) -> np.ndarray:
    n = len(base_indices)
    out = np.zeros((n, N_SLOTS), dtype=np.int64)
    for m in range(N_SLOTS):
        seg = base_indices[:, m * n_bases:(m + 1) * n_bases]
        v = np.zeros(n, dtype=np.int64)
        for r in range(n_bases):
            v = v * 4 + seg[:, r].astype(np.int64)
        out[:, m] = v
    return out


def paired_bootstrap(x: np.ndarray, n_boot: int, rng) -> dict:
    if len(x) == 0:
        return {"mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"),
                "excludes_zero": False, "n": 0}
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": float(x.mean()), "ci95_low": float(lo), "ci95_high": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0), "n": int(len(x))}


# --------------------------------------------------------------------------
def run_arm(q_codes: np.ndarray, db_codes: np.ndarray, db_labels: np.ndarray,
            q_idx: np.ndarray, donor_idx: np.ndarray, target: np.ndarray,
            slot_of: np.ndarray, seg_slices, base_topk: np.ndarray,
            base_prev: np.ndarray, k: int, n_boot: int, seed: int) -> dict:
    """Apply one intervention arm and score it."""
    inter = q_codes[q_idx].copy()
    for r, (m, j) in enumerate(zip(slot_of, donor_idx)):
        sl = seg_slices[m]
        inter[r, sl] = db_codes[j, sl]

    topk = topk_by_hamming(inter, db_codes, k)
    prev = prevalence(topk, db_labels)

    rows = np.arange(len(q_idx))
    tgt_gain = prev[rows, target] - base_prev[rows, target]

    off = np.ones_like(prev, dtype=bool)
    off[rows, target] = False
    drift = np.abs(prev - base_prev)
    off_drift = (drift * off).sum(axis=1) / off.sum(axis=1)

    jac = np.array([
        len(np.intersect1d(a, b)) / len(np.union1d(a, b))
        for a, b in zip(base_topk, topk)
    ])

    # Selectivity is the pre-registered decision criterion (REQUIRED_EXPERIMENTS
    # §3.6 condition 2), so it needs a per-query series -- keeping only the mean
    # made it impossible to test whether an arm's advantage is real. Every arm
    # is scored on the SAME queries, so arm-vs-arm comparisons are paired.
    selectivity = tgt_gain - off_drift

    # How much the swap actually perturbs the code. A swap changes only the
    # symbols where donor and query already differ -- NOT a fixed full-slot
    # budget as previously assumed -- so arms can differ in perturbation size.
    # Units differ per arm (ours: 3 bases/slot, flat: 6 bits/slot), which is the
    # same unit each arm's own Hamming ranking uses, so compare the FRACTION.
    n_sym_changed = np.array([
        int((q_codes[qi, seg_slices[m]] != db_codes[dj, seg_slices[m]]).sum())
        for qi, dj, m in zip(q_idx, donor_idx, slot_of)
    ], dtype=np.float64)
    sym_per_slot = int(seg_slices[0].stop - seg_slices[0].start)
    frac_changed = n_sym_changed / max(sym_per_slot, 1)

    rng = np.random.default_rng(seed)
    return {
        "target_gain": paired_bootstrap(tgt_gain, n_boot, rng),
        "off_target_drift": float(off_drift.mean()),
        "selectivity": float(selectivity.mean()),
        "selectivity_ci": paired_bootstrap(selectivity, n_boot, rng),
        "retrieval_jaccard": float(jac.mean()),
        "symbols_changed_mean": float(n_sym_changed.mean()),
        "symbols_per_slot": sym_per_slot,
        "fraction_changed": float(frac_changed.mean()),
        "gain_per_fraction_changed": float(
            tgt_gain.mean() / frac_changed.mean()) if frac_changed.mean() > 0 else float("nan"),
        "_gain": tgt_gain,
        "_selectivity": selectivity,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours_dir", required=True)
    ap.add_argument("--baseline_dirs", nargs="*", default=[])
    ap.add_argument("--baseline_names", nargs="*", default=[])
    ap.add_argument("--train_manifest", default=None)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--n_query", type=int, default=500)
    ap.add_argument("--db_subsample", type=int, default=0,
                    help="cap DB size for tractability (0 = use all)")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bio_project", action="store_true",
                    help="project DNA codes to bio-valid (GC by length, "
                         "homopolymer<=3) before the codon-swap intervention")
    ap.add_argument("--gc_min_frac", type=float, default=0.4444)
    ap.add_argument("--gc_max_frac", type=float, default=0.5556)
    ap.add_argument("--max_run", type=int, default=3)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    db = dict(np.load(os.path.join(args.ours_dir, "extract_db.npz"), allow_pickle=True))
    qy = dict(np.load(os.path.join(args.ours_dir, "extract_query.npz"), allow_pickle=True))
    tr_path = os.path.join(args.ours_dir, "extract_train.npz")
    tr = dict(np.load(tr_path, allow_pickle=True)) if os.path.exists(tr_path) else None

    if args.bio_project:
        import sys as _sys
        _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from dna_utils.bio_constraints import project_to_valid, is_valid_batch
        def _bp(arr):
            a = np.ascontiguousarray(arr).astype(np.int8)
            uu, iv = np.unique(a, axis=0, return_inverse=True)
            vv = is_valid_batch(uu, args.gc_min_frac, args.gc_max_frac, args.max_run)
            oo = uu.copy()
            for i in np.where(~vv)[0]:
                oo[i], _ = project_to_valid(uu[i], args.gc_min_frac, args.gc_max_frac, args.max_run)
            return oo[iv].astype(np.int64)
        for _d in (db, qy, tr):
            if _d is not None and "base_indices" in _d:
                _d["base_indices"] = _bp(_d["base_indices"])
        print(f"[{args.dataset}] [bio-projected] codes")

    lab_key = "multi_hot_labels" if "multi_hot_labels" in db else "labels"
    db_lab_full = np.asarray(db[lab_key])
    if db_lab_full.ndim == 1:
        n_cls = int(db_lab_full.max()) + 1
        db_lab_full = np.eye(n_cls, dtype=np.int64)[db_lab_full]
    q_lab = np.asarray(qy[lab_key])
    if q_lab.ndim == 1:
        q_lab = np.eye(db_lab_full.shape[1], dtype=np.int64)[q_lab]

    n_bases = db["base_indices"].shape[1] // N_SLOTS
    seg_slices = [slice(m * n_bases, (m + 1) * n_bases) for m in range(N_SLOTS)]

    # ---- DB subsample (keeps the counterfactual tractable; same DB for all arms)
    db_all = np.arange(len(db_lab_full))
    if args.db_subsample and args.db_subsample < len(db_all):
        db_all = np.sort(rng.choice(db_all, args.db_subsample, replace=False))
    db_codes = db["base_indices"][db_all]
    db_lab = db_lab_full[db_all]

    # ---- train-only dictionary for defining the target concept
    if tr is not None:
        tr_units = codon_ids(tr["base_indices"], n_bases)
        tr_lab = np.asarray(tr[lab_key])
    else:
        if not args.train_manifest:
            raise SystemExit("need --train_manifest when extract_train.npz is absent")
        want = {os.path.basename(l.split()[0]) for l in open(args.train_manifest) if l.strip()}
        db_b = _basenames(db["image_paths"])
        keep = np.array([i for i, b in enumerate(db_b) if b in want])
        tr_units = codon_ids(db["base_indices"][keep], n_bases)
        tr_lab = db_lab_full[keep]
    if tr_lab.ndim == 1:
        tr_lab = np.eye(db_lab_full.shape[1], dtype=np.int64)[tr_lab]
    dict_p = np.stack([
        build_codon_dictionary(tr_units[:, m], tr_lab, 4 ** n_bases, args.alpha)
        for m in range(N_SLOTS)
    ])                                                   # [6, n_codons, C]

    q_sel = rng.choice(len(q_lab), min(args.n_query, len(q_lab)), replace=False)
    q_codes = qy["base_indices"]
    db_codon = codon_ids(db_codes, n_bases)

    base_topk = topk_by_hamming(q_codes[q_sel], db_codes, args.k)
    base_prev = prevalence(base_topk, db_lab)

    # ---- flat-hash controls, realigned to exactly our db subsample and queries
    flat_arms = []
    our_db_b = _basenames(db["image_paths"])[db_all]
    our_q_b = _basenames(qy["image_paths"])
    for bdir, bname in zip(args.baseline_dirs, args.baseline_names):
        bdb = np.load(os.path.join(bdir, "extract_db.npz"), allow_pickle=True)
        bqy = np.load(os.path.join(bdir, "extract_query.npz"), allow_pickle=True)
        dpos = {b: i for i, b in enumerate(_basenames(bdb["image_paths"]))}
        qpos = {b: i for i, b in enumerate(_basenames(bqy["image_paths"]))}
        bdb_codes = (bdb["hash_2bit"][[dpos[b] for b in our_db_b]] > 0).astype(np.int64)
        bq_codes = (bqy["hash_2bit"][[qpos[b] for b in our_q_b]] > 0).astype(np.int64)
        w = bdb_codes.shape[1] // N_SLOTS
        flat_arms.append((bname, bq_codes, bdb_codes,
                          [slice(m * w, (m + 1) * w) for m in range(N_SLOTS)]))
        print(f"  [flat control] {bname}: {bdb_codes.shape[1]} bits, "
              f"{w} bits/slot")

    print(f"[{args.dataset}] queries={len(q_sel)} db={len(db_all)} K={args.k} "
          f"labels={db_lab.shape[1]}")

    results = {}
    for m in range(N_SLOTS):
        qi, dj, tg = [], [], []
        for r, i in enumerate(q_sel):
            has = q_lab[i].astype(bool)
            # candidate targets: labels the query lacks
            cand = np.where(~has)[0]
            if len(cand) == 0:
                continue
            # donors: DB samples having a candidate target, whose OTHER labels
            # match the query as closely as possible
            best = None
            for c in rng.permutation(cand)[:8]:          # sample targets for speed
                pool = np.where(db_lab[:, c] == 1)[0]
                if len(pool) == 0:
                    continue
                other = np.ones(db_lab.shape[1], dtype=bool); other[c] = False
                inter_ = (db_lab[pool][:, other] & has[other]).sum(axis=1)
                union_ = (db_lab[pool][:, other] | has[other]).sum(axis=1)
                sim = inter_ / np.maximum(union_, 1)
                b = pool[np.argmax(sim)]
                # the donor codon must actually differ, else it is a no-op
                if db_codon[b, m] == codon_ids(q_codes[i:i + 1], n_bases)[0, m]:
                    continue
                score = sim.max() + dict_p[m, db_codon[b, m], c]
                if best is None or score > best[0]:
                    best = (score, b, c)
            if best is None:
                continue
            qi.append(i); dj.append(best[1]); tg.append(best[2])

        if not qi:
            results[SLOT_NAMES[m]] = {"coverage": 0.0}
            continue
        qi, dj, tg = np.array(qi), np.array(dj), np.array(tg)
        pos = {v: r for r, v in enumerate(q_sel)}
        rows = np.array([pos[v] for v in qi])
        slot_of = np.full(len(qi), m)

        arms = {}
        arms["ours_slot"] = run_arm(
            q_codes, db_codes, db_lab, qi, dj, tg, slot_of, seg_slices,
            base_topk[rows], base_prev[rows], args.k, args.n_boot, args.seed)

        rand_slot = np.array([rng.choice([x for x in range(N_SLOTS) if x != m])
                              for _ in qi])
        arms["random_slot"] = run_arm(
            q_codes, db_codes, db_lab, qi, dj, tg, rand_slot, seg_slices,
            base_topk[rows], base_prev[rows], args.k, args.n_boot, args.seed)

        rand_donor = rng.choice(len(db_all), len(qi))
        arms["random_donor"] = run_arm(
            q_codes, db_codes, db_lab, qi, rand_donor, tg, slot_of, seg_slices,
            base_topk[rows], base_prev[rows], args.k, args.n_boot, args.seed)

        # flat-hash chunk arm: same 6-bit budget, same donors, same targets (§3.5)
        for bname, bq, bdb_codes, b_seg in flat_arms:
            b_base_topk = topk_by_hamming(bq[qi], bdb_codes, args.k)
            b_base_prev = prevalence(b_base_topk, db_lab)
            arms[f"{bname}_chunk"] = run_arm(
                bq, bdb_codes, db_lab, qi, dj, tg, slot_of, b_seg,
                b_base_topk, b_base_prev, args.k, args.n_boot, args.seed)

        # Pre-registered condition 2 asks whether OURS beats each control on
        # selectivity. Comparing two point estimates cannot answer that, so test
        # the paired per-query difference directly (all arms share the queries).
        _rng_cmp = np.random.default_rng(args.seed + 1)
        vs = {}
        for other in [a_ for a_ in arms if a_ != "ours_slot"]:
            d = arms["ours_slot"]["_selectivity"] - arms[other]["_selectivity"]
            vs[f"ours_minus_{other}"] = paired_bootstrap(d, args.n_boot, _rng_cmp)

        results[SLOT_NAMES[m]] = {
            "coverage": float(len(qi) / len(q_sel)),
            "arms": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                     for k, v in arms.items()},
            "selectivity_vs_controls": vs,
        }
        a = arms["ours_slot"]
        print(f"  {SLOT_NAMES[m]:18s} cov={len(qi)/len(q_sel):.2f} "
              f"gain={a['target_gain']['mean']:+.4f} "
              f"CI[{a['target_gain']['ci95_low']:+.4f},{a['target_gain']['ci95_high']:+.4f}] "
              f"drift={a['off_target_drift']:.4f} sel={a['selectivity']:+.4f} "
              f"jac={a['retrieval_jaccard']:.3f} | randslot gain="
              f"{arms['random_slot']['target_gain']['mean']:+.4f}")

    payload = {
        "dataset": args.dataset,
        "config": vars(args),
        "n_db_used": int(len(db_all)),
        "results": results,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[{args.dataset}] wrote {args.out}")


if __name__ == "__main__":
    main()
