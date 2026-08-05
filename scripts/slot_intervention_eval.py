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
    ours_slot     : intended 3-base slot codon replaced
    <base>_chunk  : same-position 3-base / contiguous 6-bit chunk
    random_donor  : donor drawn at random, no label matching
    random_slot   : a different slot replaced with the donor's codon there

With ``--bio_project``, input codes are projected first and donor selection is
restricted to exact slot swaps that remain valid DNA.  Invalid splices are
excluded on one paired common-valid query subset; the donor slot is never
silently repaired and all other slots stay bit-for-bit unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Callable, Optional, Sequence

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
    """Symbol-wise mismatch count with a canonical database-order tie break."""
    if query_codes.ndim != 2 or db_codes.ndim != 2:
        raise ValueError("query_codes and db_codes must both be rank-2")
    if query_codes.shape[1] != db_codes.shape[1]:
        raise ValueError(
            f"code widths differ: {query_codes.shape[1]} vs {db_codes.shape[1]}"
        )
    if not 0 < int(k) <= len(db_codes):
        raise ValueError(f"k must be in [1, {len(db_codes)}], got {k}")
    out = np.zeros((len(query_codes), k), dtype=np.int64)
    for s in range(0, len(query_codes), chunk):
        q = query_codes[s:s + chunk]
        d = (q[:, None, :] != db_codes[None, :, :]).sum(axis=2)
        # Hamming codes have very large boundary ties.  argpartition chooses
        # an arbitrary subset of those ties, so identical artifacts could
        # produce materially different neighbours.  Stable sort implements
        # the repository-wide canonical policy: distance, then DB row index.
        out[s:s + chunk] = np.argsort(
            d, axis=1, kind="stable",
        )[:, :k]
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


def swap_segments(
    q_codes: np.ndarray,
    db_codes: np.ndarray,
    q_idx: np.ndarray,
    donor_idx: np.ndarray,
    slot_of: np.ndarray,
    seg_slices: Sequence[slice],
) -> np.ndarray:
    """Return exact one-segment swaps; all non-target positions are unchanged."""
    inter = q_codes[q_idx].copy()
    for row, (slot, donor) in enumerate(zip(slot_of, donor_idx)):
        segment = seg_slices[int(slot)]
        inter[row, segment] = db_codes[int(donor), segment]
    return inter


def valid_swap_candidates(
    query_code: np.ndarray,
    donor_codes: np.ndarray,
    slot: int,
    seg_slices: Sequence[slice],
    validity_fn: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Which donor segments preserve whole-strand validity after exact swap."""
    if len(donor_codes) == 0:
        return np.zeros(0, dtype=bool)
    candidates = np.repeat(query_code[None, :], len(donor_codes), axis=0)
    segment = seg_slices[int(slot)]
    candidates[:, segment] = donor_codes[:, segment]
    valid = np.asarray(validity_fn(candidates), dtype=bool)
    if valid.shape != (len(donor_codes),):
        raise ValueError(
            "validity_fn must return one boolean per candidate, got "
            f"{valid.shape}"
        )
    return valid


# --------------------------------------------------------------------------
def run_arm(q_codes: np.ndarray, db_codes: np.ndarray, db_labels: np.ndarray,
            q_idx: np.ndarray, donor_idx: np.ndarray, target: np.ndarray,
            slot_of: np.ndarray, seg_slices, base_topk: np.ndarray,
            base_prev: np.ndarray, k: int, n_boot: int, seed: int,
            validity_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> dict:
    """Apply one intervention arm and score it."""
    inter = swap_segments(
        q_codes, db_codes, q_idx, donor_idx, slot_of, seg_slices,
    )
    if validity_fn is not None:
        post_valid = np.asarray(validity_fn(inter), dtype=bool)
        if post_valid.shape != (len(inter),):
            raise ValueError("validity_fn returned an invalid shape")
        if not post_valid.all():
            raise RuntimeError(
                "exact slot intervention produced invalid DNA; donor/control "
                "selection must filter to the paired common-valid subset"
            )
    else:
        post_valid = None

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
        "n_requested": int(len(q_idx)),
        "n_evaluated": int(len(q_idx)),
        "invalid_after_swap": 0 if post_valid is not None else None,
        "post_swap_valid_fraction": (
            float(post_valid.mean()) if post_valid is not None else None
        ),
        "non_target_symbols_changed": 0,
        "_gain": tgt_gain,
        "_selectivity": selectivity,
        "_fraction_changed": frac_changed,
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

    if len(args.baseline_dirs) != len(args.baseline_names):
        raise ValueError(
            "--baseline_dirs and --baseline_names must have identical lengths"
        )
    normalized_baseline_names = [
        str(name).strip().casefold() for name in args.baseline_names
    ]
    if any(not name for name in normalized_baseline_names):
        raise ValueError("baseline names must be non-empty")
    if len(set(normalized_baseline_names)) != len(normalized_baseline_names):
        raise ValueError("baseline names must be unique")

    rng = np.random.default_rng(args.seed)
    db = dict(np.load(os.path.join(args.ours_dir, "extract_db.npz"), allow_pickle=True))
    qy = dict(np.load(os.path.join(args.ours_dir, "extract_query.npz"), allow_pickle=True))
    tr_path = os.path.join(args.ours_dir, "extract_train.npz")
    tr = dict(np.load(tr_path, allow_pickle=True)) if os.path.exists(tr_path) else None

    bio_validity_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None
    if args.bio_project:
        import sys as _sys
        _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from dna_utils.bio_constraints import project_to_valid, is_valid_batch

        def _is_bio_valid(arr):
            return is_valid_batch(
                arr, args.gc_min_frac, args.gc_max_frac, args.max_run,
            )

        bio_validity_fn = _is_bio_valid

        def _bp(arr):
            a = np.ascontiguousarray(arr).astype(np.int8)
            uu, iv = np.unique(a, axis=0, return_inverse=True)
            vv = _is_bio_valid(uu)
            oo = uu.copy()
            for i in np.where(~vv)[0]:
                oo[i], _ = project_to_valid(uu[i], args.gc_min_frac, args.gc_max_frac, args.max_run)
            projected = oo[iv].astype(np.int64)
            if not bool(_is_bio_valid(projected).all()):
                raise RuntimeError(
                    "bio projection failed to produce 100% valid input codes"
                )
            return projected
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
    q_codon = codon_ids(q_codes, n_bases)

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
        bdb_bits = (
            bdb["hash_2bit"][[dpos[b] for b in our_db_b]] > 0
        ).astype(np.int64)
        bq_bits = (
            bqy["hash_2bit"][[qpos[b] for b in our_q_b]] > 0
        ).astype(np.int64)
        if args.bio_project:
            if bdb_bits.shape[1] % 2 or bq_bits.shape[1] % 2:
                raise ValueError(
                    f"baseline {bname} has an odd bit width and cannot be "
                    "converted to DNA bases"
                )
            bdb_pairs = bdb_bits.reshape(len(bdb_bits), -1, 2)
            bq_pairs = bq_bits.reshape(len(bq_bits), -1, 2)
            bdb_codes = _bp(2 * bdb_pairs[..., 0] + bdb_pairs[..., 1])
            bq_codes = _bp(2 * bq_pairs[..., 0] + bq_pairs[..., 1])
        else:
            bdb_codes = bdb_bits
            bq_codes = bq_bits
        w = bdb_codes.shape[1] // N_SLOTS
        if w * N_SLOTS != bdb_codes.shape[1]:
            raise ValueError(
                f"baseline {bname} width {bdb_codes.shape[1]} is not "
                f"divisible into {N_SLOTS} slots"
            )
        flat_arms.append((bname, bq_codes, bdb_codes,
                          [slice(m * w, (m + 1) * w) for m in range(N_SLOTS)]))
        unit = "bases" if args.bio_project else "bits"
        print(f"  [flat control] {bname}: {bdb_codes.shape[1]} {unit}, "
              f"{w} {unit}/slot")

    def candidates_valid_across_methods(
        query_index: int,
        donor_candidates: np.ndarray,
        slot: int,
    ) -> np.ndarray:
        """Common non-noop/valid mask for ours and flat baseline controls."""
        donor_candidates = np.asarray(donor_candidates, dtype=np.int64)
        segment = seg_slices[int(slot)]
        ours_changed = np.sum(
            db_codes[donor_candidates, segment]
            != q_codes[int(query_index), segment],
            axis=1,
        )
        ours_width = int(segment.stop - segment.start)
        valid = ours_changed > 0
        if bio_validity_fn is not None:
            valid &= valid_swap_candidates(
                q_codes[query_index], db_codes[donor_candidates], slot,
                seg_slices, bio_validity_fn,
            )
        for _, baseline_q, baseline_db, baseline_slices in flat_arms:
            baseline_segment = baseline_slices[int(slot)]
            baseline_changed = np.sum(
                baseline_db[donor_candidates, baseline_segment]
                != baseline_q[int(query_index), baseline_segment],
                axis=1,
            )
            baseline_width = int(
                baseline_segment.stop - baseline_segment.start
            )
            # Exact per-query dose matching in each method's own Hamming
            # metric: changed/slot_width must be identical across arms.
            valid &= baseline_changed > 0
            valid &= (
                baseline_changed * ours_width
                == ours_changed * baseline_width
            )
            if bio_validity_fn is not None:
                valid &= valid_swap_candidates(
                    baseline_q[query_index], baseline_db[donor_candidates], slot,
                    baseline_slices, bio_validity_fn,
                )
        return valid

    def ours_candidates_valid(
        query_index: int,
        donor_candidates: np.ndarray,
        slot: int,
        required_changed_symbols: Optional[int] = None,
    ) -> np.ndarray:
        """Non-noop/valid mask for an ours-only random-control arm."""
        donor_candidates = np.asarray(donor_candidates, dtype=np.int64)
        segment = seg_slices[int(slot)]
        changed = np.sum(
            db_codes[donor_candidates, segment]
            != q_codes[int(query_index), segment],
            axis=1,
        )
        valid = changed > 0
        if required_changed_symbols is not None:
            valid &= changed == int(required_changed_symbols)
        if bio_validity_fn is not None:
            valid &= valid_swap_candidates(
                q_codes[query_index], db_codes[donor_candidates], slot,
                seg_slices, bio_validity_fn,
            )
        return valid

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
                # Pick the semantic donor from OUR representation only so a
                # baseline control cannot influence which counterfactual is
                # selected.  Cross-method validity/dose matching is applied
                # afterward as a query-level paired-subset filter.
                pool = pool[db_codon[pool, m] != q_codon[i, m]]
                if len(pool):
                    pool = pool[ours_candidates_valid(i, pool, m)]
                if len(pool) == 0:
                    continue
                other = np.ones(db_lab.shape[1], dtype=bool); other[c] = False
                inter_ = (db_lab[pool][:, other] & has[other]).sum(axis=1)
                union_ = (db_lab[pool][:, other] | has[other]).sum(axis=1)
                sim = inter_ / np.maximum(union_, 1)
                b = pool[np.argmax(sim)]
                score = sim.max() + dict_p[m, db_codon[b, m], c]
                if best is None or score > best[0]:
                    best = (score, b, c)
            if best is None:
                continue
            qi.append(i); dj.append(best[1]); tg.append(best[2])

        if not qi:
            results[SLOT_NAMES[m]] = {"coverage": 0.0}
            continue
        qi = np.asarray(qi, dtype=np.int64)
        dj = np.asarray(dj, dtype=np.int64)
        tg = np.asarray(tg, dtype=np.int64)
        n_target_eligible = len(qi)

        # Every arm must use the same query subset for paired bootstrap.  For
        # bio-projected evaluation, choose only controls whose exact one-slot
        # splice is valid; never repair globally after the swap because that
        # would alter non-target slots and invalidate the causal intervention.
        keep_rows = []
        random_slots = []
        random_donors = []
        for row, (query_index, donor_index) in enumerate(zip(qi, dj)):
            if not candidates_valid_across_methods(
                int(query_index), np.asarray([donor_index]), m,
            )[0]:
                continue
            target_segment = seg_slices[m]
            target_changed_symbols = int(np.sum(
                db_codes[int(donor_index), target_segment]
                != q_codes[int(query_index), target_segment]
            ))
            if target_changed_symbols <= 0:
                raise RuntimeError("target donor unexpectedly produced a no-op")
            valid_other_slots = []
            for candidate_slot in rng.permutation([
                slot for slot in range(N_SLOTS) if slot != m
            ]):
                if ours_candidates_valid(
                    int(query_index),
                    np.asarray([donor_index]),
                    int(candidate_slot),
                    required_changed_symbols=target_changed_symbols,
                )[0]:
                    valid_other_slots.append(int(candidate_slot))
            if not valid_other_slots:
                continue

            chosen_random_donor = None
            donor_order = rng.permutation(len(db_codes))
            # A random permutation followed by the first eligible candidate is
            # uniform over non-noop donors (and, in deployment mode, those
            # whose exact splice remains bio-valid).  Chunking avoids
            # materializing every candidate strand at once.
            for start in range(0, len(donor_order), 256):
                donor_chunk = donor_order[start:start + 256]
                valid = ours_candidates_valid(
                    int(query_index), donor_chunk, m,
                    required_changed_symbols=target_changed_symbols,
                )
                eligible = donor_chunk[valid]
                if len(eligible):
                    chosen_random_donor = int(eligible[0])
                    break
            if chosen_random_donor is None:
                continue
            keep_rows.append(row)
            random_slots.append(valid_other_slots[0])
            random_donors.append(chosen_random_donor)

        if not keep_rows:
            results[SLOT_NAMES[m]] = {
                "coverage": 0.0,
                "target_donor_coverage": float(
                    n_target_eligible / len(q_sel)
                ),
                "paired_common_subset_n": 0,
            }
            continue
        keep_rows = np.asarray(keep_rows, dtype=np.int64)
        qi, dj, tg = qi[keep_rows], dj[keep_rows], tg[keep_rows]
        rand_slot = np.asarray(random_slots, dtype=np.int64)
        rand_donor = np.asarray(random_donors, dtype=np.int64)

        pos = {v: r for r, v in enumerate(q_sel)}
        rows = np.array([pos[v] for v in qi])
        slot_of = np.full(len(qi), m)

        arms = {}
        arms["ours_slot"] = run_arm(
            q_codes, db_codes, db_lab, qi, dj, tg, slot_of, seg_slices,
            base_topk[rows], base_prev[rows], args.k, args.n_boot, args.seed,
            validity_fn=bio_validity_fn)

        arms["random_slot"] = run_arm(
            q_codes, db_codes, db_lab, qi, dj, tg, rand_slot, seg_slices,
            base_topk[rows], base_prev[rows], args.k, args.n_boot, args.seed,
            validity_fn=bio_validity_fn)

        arms["random_donor"] = run_arm(
            q_codes, db_codes, db_lab, qi, rand_donor, tg, slot_of, seg_slices,
            base_topk[rows], base_prev[rows], args.k, args.n_boot, args.seed,
            validity_fn=bio_validity_fn)

        # flat-hash chunk arm: same 6-bit budget, same donors, same targets (§3.5)
        for bname, bq, bdb_codes, b_seg in flat_arms:
            b_base_topk = topk_by_hamming(bq[qi], bdb_codes, args.k)
            b_base_prev = prevalence(b_base_topk, db_lab)
            arms[f"{bname}_chunk"] = run_arm(
                bq, bdb_codes, db_lab, qi, dj, tg, slot_of, b_seg,
                b_base_topk, b_base_prev, args.k, args.n_boot, args.seed,
                validity_fn=bio_validity_fn)

        reference_dose = np.asarray(
            arms["ours_slot"]["_fraction_changed"], dtype=np.float64,
        )
        for arm_name, arm in arms.items():
            arm_dose = np.asarray(arm["_fraction_changed"], dtype=np.float64)
            if not np.array_equal(reference_dose, arm_dose):
                raise RuntimeError(
                    f"intervention dose mismatch for {arm_name}; every arm "
                    "must change the same per-query fraction of one slot"
                )

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
            "target_donor_coverage": float(n_target_eligible / len(q_sel)),
            "paired_common_subset_n": int(len(qi)),
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

    required_cibhash_control = "cibhash" in normalized_baseline_names
    complete_slot_coverage = all(
        bool(slot_result.get("paired_common_subset_n", 0) > 0)
        for slot_result in results.values()
    )
    deployment_valid = bool(args.bio_project and complete_slot_coverage)
    payload = {
        "dataset": args.dataset,
        "config": vars(args),
        "intervention_protocol_version": 2,
        "validity_policy": (
            "exact_slot_common_valid_only"
            if args.bio_project else "unconstrained_diagnostic"
        ),
        "paired_subset_policy": "all_configured_arms_common_valid",
        "intervention_dose_policy": (
            "exact_per_query_changed_slot_fraction_matched"
        ),
        "ranking_tie_policy": "hamming_then_database_index_stable",
        "deployment_valid_intervention": deployment_valid,
        "preregistered_cibhash_control_present": required_cibhash_control,
        # This script binds extraction rows but not a canonical baseline run
        # manifest/checkpoint digest.  Never promote an output solely because
        # a directory was labelled "cibhash"; a paper table builder must bind
        # and validate that provenance separately.
        "paper_result_eligible": False,
        "paper_eligibility_blockers": [
            "baseline_control_manifest_provenance_not_bound"
        ] if deployment_valid and required_cibhash_control else [
            "incomplete_deployment_valid_or_preregistered_controls"
        ],
        "n_db_used": int(len(db_all)),
        "results": results,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[{args.dataset}] wrote {args.out}")


if __name__ == "__main__":
    main()
