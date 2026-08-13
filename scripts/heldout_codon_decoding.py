#!/usr/bin/env python
"""Held-out codeword/codon decoding (REQUIRED_EXPERIMENTS_GROUNDEDDNA_PAPER.md §2).

Question: does a `(slot, code) -> concept` dictionary built ONLY on train
predict the concepts of unseen test images from the code alone?

Protocol
    dictionary split : train rows  (Flickr/NUS: train is a subset of the DB
                       extraction, matched by image basename; MSCOCO ships its
                       own train extraction rows the same way)
    evaluation split : official test (extract_query.npz) -- disjoint from db
                       and from train, verified at load time
    target           : dataset multi-hot labels.  These are INDEPENDENT of the
                       Qwen captions used as the training teacher, so this is
                       not a circular evaluation (§2.2).
    decoder input    : the slot's integer code id and nothing else.  No test
                       caption, no test image feature.

Code units compared, all with the identical dictionary/decode procedure (§2.9):
    ours-codeword : codebook_indices[:, m]                      K per slot
    ours-codon    : 16*b0 + 4*b1 + b2 from base_indices[3m:3m+3] 64 per slot
    <base>-chunk  : contiguous 6-bit chunks of the 36-bit hash   64 per slot
    majority      : train label frequency, ignores the code
    shuffled      : train unit assignment permuted before the dictionary is
                    built, so the code carries no real information

Smoothing and min-support are fixed a priori (see DEFAULTS) and are NOT tuned
on test.  Codes with support below the threshold are `unknown` and fall back to
the prior; coverage is always reported alongside (§2.2, §2.7).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SLOT_NAMES = [
    "global", "primary_object", "secondary_object",
    "activity_relation", "color_texture", "scene_type",
]
# Slot count is INFERRED from the extraction at run time (see main), because a
# run may carry fewer than six: dropping `scene_type` -- the only slot whose
# removal IMPROVES decoding on 4/4 datasets -- leaves 5 slots / 15 bases.
# Left as a module global so the helpers below keep their current signatures;
# `_set_n_slots` is the single place that changes it.
N_SLOTS = 6


def _set_n_slots(n: int) -> None:
    """Point the module at an n-slot code. Must be called before decoding."""
    global N_SLOTS, SLOT_NAMES
    n = int(n)
    if not (2 <= n <= 6):
        raise ValueError(f"n_slots must be in [2, 6], got {n}")
    N_SLOTS = n
    SLOT_NAMES = SLOT_NAMES[:n]

# Pre-registered, never tuned on test.
DEFAULT_ALPHA = 1.0        # Beta(alpha, alpha) smoothing per label
DEFAULT_MIN_SUPPORT = 10   # train samples needed before a code is decodable
DEFAULT_N_SHUFFLE = 5
DEFAULT_N_BOOT = 1000


# --------------------------------------------------------------------------
# code extraction
# --------------------------------------------------------------------------
def codon_ids(base_indices: np.ndarray, n_bases: int = 3) -> np.ndarray:
    """[N, 6*n_bases] A/C/G/T indices -> [N, 6] codon ids in base-4."""
    n, total = base_indices.shape
    if total != N_SLOTS * n_bases:
        raise ValueError(f"expected {N_SLOTS * n_bases} bases, got {total}")
    out = np.zeros((n, N_SLOTS), dtype=np.int64)
    for m in range(N_SLOTS):
        seg = base_indices[:, m * n_bases:(m + 1) * n_bases]
        val = np.zeros(n, dtype=np.int64)
        for r in range(n_bases):
            val = val * 4 + seg[:, r].astype(np.int64)
        out[:, m] = val
    return out


def bit_chunk_ids(hash_2bit: np.ndarray, n_chunks: int | None = None) -> np.ndarray:
    # Default resolved at CALL time, not def time: `_set_n_slots` may have
    # lowered N_SLOTS after this function object was created.
    """36-bit flat hash -> `n_chunks` contiguous chunk ids (§2.9)."""
    if n_chunks is None:
        n_chunks = N_SLOTS
    n, nbits = hash_2bit.shape
    width = nbits // n_chunks
    bits = (hash_2bit > 0).astype(np.int64)
    out = np.zeros((n, n_chunks), dtype=np.int64)
    for j in range(n_chunks):
        seg = bits[:, j * width:(j + 1) * width]
        val = np.zeros(n, dtype=np.int64)
        for r in range(width):
            val = val * 2 + seg[:, r]
        out[:, j] = val
    return out


# --------------------------------------------------------------------------
# dictionary + decoding
# --------------------------------------------------------------------------
def build_dictionary(units: np.ndarray, labels: np.ndarray, n_units: int,
                     alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """-> p[n_units, C] smoothed P(label=1 | unit), support[n_units]."""
    n_lab = labels.shape[1]
    cnt = np.zeros((n_units, n_lab), dtype=np.float64)
    sup = np.zeros(n_units, dtype=np.float64)
    np.add.at(cnt, units, labels.astype(np.float64))
    np.add.at(sup, units, 1.0)
    p = (cnt + alpha) / (sup[:, None] + 2.0 * alpha)
    return p, sup


def label_ranking_ap(scores: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Per-sample AP over the label ranking. Samples with no positive -> nan."""
    n, c = scores.shape
    order = np.argsort(-scores, axis=1, kind="stable")
    hits = np.take_along_axis(truth, order, axis=1).astype(np.float64)
    csum = np.cumsum(hits, axis=1)
    ranks = np.arange(1, c + 1, dtype=np.float64)[None, :]
    prec = csum / ranks
    npos = hits.sum(axis=1)
    ap = np.where(npos > 0, (prec * hits).sum(axis=1) / np.maximum(npos, 1e-12), np.nan)
    return ap


def conditional_entropy(p: np.ndarray, sup: np.ndarray) -> float:
    """Support-weighted mean per-label Bernoulli entropy H(concept | code), nats."""
    w = sup / max(sup.sum(), 1e-12)
    pc = np.clip(p, 1e-12, 1 - 1e-12)
    h = -(pc * np.log(pc) + (1 - pc) * np.log(1 - pc))   # [n_units, C]
    return float((w[:, None] * h).sum(axis=0).mean())


def marginal_entropy(prior: np.ndarray) -> float:
    pc = np.clip(prior, 1e-12, 1 - 1e-12)
    return float((-(pc * np.log(pc) + (1 - pc) * np.log(1 - pc))).mean())


def decode_slot(train_units: np.ndarray, train_labels: np.ndarray,
                test_units: np.ndarray, test_labels: np.ndarray,
                n_units: int, alpha: float, min_support: int) -> dict:
    p, sup = build_dictionary(train_units, train_labels, n_units, alpha)
    prior = (train_labels.sum(axis=0) + alpha) / (len(train_labels) + 2.0 * alpha)

    covered = sup[test_units] >= min_support
    scores = np.where(covered[:, None], p[test_units], prior[None, :])

    ap = label_ranking_ap(scores, test_labels)
    top1_idx = np.argmax(scores, axis=1)
    top1 = test_labels[np.arange(len(test_labels)), top1_idx].astype(np.float64)

    valid = ~np.isnan(ap)
    both = valid & covered
    active = sup >= min_support
    return {
        "concept_mAP": float(np.nanmean(ap)),
        "concept_mAP_covered_only": float(np.nanmean(ap[both])) if both.any() else float("nan"),
        "top1_acc": float(top1[valid].mean()),
        "coverage": float(covered.mean()),
        "n_active_units": int(active.sum()),
        "n_units": int(n_units),
        "H_cond": conditional_entropy(p[active], sup[active]) if active.any() else float("nan"),
        "H_marg": marginal_entropy(prior),
        "_ap": ap,          # kept for bootstrap, stripped before serialising
    }


def decode_all_slots(train_units, train_labels, test_units, test_labels,
                     n_units, alpha, min_support) -> dict:
    per_slot = [
        decode_slot(train_units[:, m], train_labels, test_units[:, m],
                    test_labels, n_units, alpha, min_support)
        for m in range(N_SLOTS)
    ]
    ap_stack = np.stack([s["_ap"] for s in per_slot])        # [6, N]
    mean_ap = np.nanmean(ap_stack, axis=0)
    return {
        "per_slot": [{k: v for k, v in s.items() if k != "_ap"} for s in per_slot],
        "slot_mean": {
            "concept_mAP": float(np.nanmean(mean_ap)),
            "top1_acc": float(np.mean([s["top1_acc"] for s in per_slot])),
            "coverage": float(np.mean([s["coverage"] for s in per_slot])),
            "H_cond": float(np.nanmean([s["H_cond"] for s in per_slot])),
        },
        "_ap_per_sample": mean_ap,
    }


def majority_control(train_labels, test_labels, alpha) -> dict:
    prior = (train_labels.sum(axis=0) + alpha) / (len(train_labels) + 2.0 * alpha)
    scores = np.repeat(prior[None, :], len(test_labels), axis=0)
    ap = label_ranking_ap(scores, test_labels)
    top1 = test_labels[np.arange(len(test_labels)), np.argmax(scores, axis=1)]
    valid = ~np.isnan(ap)
    return {
        "concept_mAP": float(np.nanmean(ap)),
        "top1_acc": float(top1[valid].mean()),
        "H_marg": marginal_entropy(prior),
        "_ap_per_sample": ap,
    }


def shuffled_control(train_units, train_labels, test_units, test_labels,
                     n_units, alpha, min_support, n_rep, seed) -> dict:
    rng = np.random.default_rng(seed)
    maps, aps = [], []
    for _ in range(n_rep):
        perm = rng.permutation(len(train_units))
        res = decode_all_slots(train_units[perm], train_labels, test_units,
                               test_labels, n_units, alpha, min_support)
        maps.append(res["slot_mean"]["concept_mAP"])
        aps.append(res["_ap_per_sample"])
    return {
        "concept_mAP": float(np.mean(maps)),
        "concept_mAP_std": float(np.std(maps)),
        "n_rep": n_rep,
        "_ap_per_sample": np.nanmean(np.stack(aps), axis=0),
    }


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int, seed: int) -> dict:
    """95% CI of mean(a) - mean(b), paired over test samples (§2.11)."""
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "delta": float(a.mean() - b.mean()),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "excludes_zero": bool(lo > 0 or hi < 0),
    }


def codeword_codon_jsd(train_cw: np.ndarray, train_cd: np.ndarray,
                       train_labels: np.ndarray, k: int, alpha: float,
                       min_support: int) -> dict:
    """JSD between concept distributions of codewords merged into one codon (§2.8)."""
    p_cw, sup_cw = build_dictionary(train_cw, train_labels, k, alpha)
    jsds, n_pairs = [], 0
    for codon in np.unique(train_cd):
        members = np.unique(train_cw[train_cd == codon])
        members = [c for c in members if sup_cw[c] >= min_support]
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                p, q = p_cw[members[i]], p_cw[members[j]]
                m = 0.5 * (p + q)
                # per-label Bernoulli JSD, averaged over labels
                def kl(x, y):
                    x = np.clip(x, 1e-12, 1 - 1e-12); y = np.clip(y, 1e-12, 1 - 1e-12)
                    return x * np.log(x / y) + (1 - x) * np.log((1 - x) / (1 - y))
                jsds.append(float((0.5 * kl(p, m) + 0.5 * kl(q, m)).mean()))
                n_pairs += 1
    return {
        "mean_jsd": float(np.mean(jsds)) if jsds else float("nan"),
        "max_jsd": float(np.max(jsds)) if jsds else float("nan"),
        "n_merged_pairs": n_pairs,
    }


# --------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------
def _basenames(paths: Sequence) -> np.ndarray:
    return np.array([os.path.basename(str(p)) for p in paths])


def load_split(result_dir: str, train_manifest: str | None) -> dict:
    """Return the train-side and test-side code/label arrays.

    Two ways to get the train rows:
      * `extract_train.npz` present -> use it directly (MSCOCO, whose train
        split is disjoint from its DB);
      * otherwise slice them out of the DB extraction by image basename using
        `train_manifest` (Flickr25k, NUS-WIDE, where train is a DB subset).
    """
    db = np.load(os.path.join(result_dir, "extract_db.npz"), allow_pickle=True)
    qy = np.load(os.path.join(result_dir, "extract_query.npz"), allow_pickle=True)
    tr_path = os.path.join(result_dir, "extract_train.npz")
    tr_npz = np.load(tr_path, allow_pickle=True) if os.path.exists(tr_path) else None

    lab_key = "multi_hot_labels" if "multi_hot_labels" in db.files else "labels"
    qy_b = _basenames(qy["image_paths"])

    if tr_npz is not None:
        src, tr_idx = tr_npz, np.arange(len(tr_npz["image_paths"]))
        tr_b, missing = _basenames(tr_npz["image_paths"]), 0
        mode = "extract_train.npz"
    else:
        if not train_manifest:
            raise SystemExit("need --train_manifest when extract_train.npz is absent")
        db_b = _basenames(db["image_paths"])
        want = {os.path.basename(l.split()[0]) for l in open(train_manifest) if l.strip()}
        pos = {b: i for i, b in enumerate(db_b)}
        tr_idx = np.array([pos[b] for b in db_b if b in want and b in pos])
        src, tr_b = db, db_b[tr_idx]
        missing, mode = len(want) - len(tr_idx), "sliced from extract_db.npz"

    # leakage guards -- fail loudly rather than silently producing a nice number
    overlap_qt = len(set(qy_b) & set(tr_b))
    overlap_qd = len(set(qy_b) & set(_basenames(db["image_paths"])))
    if overlap_qt or overlap_qd:
        raise RuntimeError(f"split leak: query∩train={overlap_qt}, query∩db={overlap_qd}")

    def _as_multihot(arr, n_cls=None):
        arr = np.asarray(arr)
        if arr.ndim == 1:
            n_cls = n_cls or int(arr.max()) + 1
            return np.eye(n_cls, dtype=np.int64)[arr], n_cls
        return arr.astype(np.int64), arr.shape[1]

    tr_lab, n_cls = _as_multihot(src[lab_key][tr_idx])
    te_lab, _ = _as_multihot(qy[lab_key], n_cls)

    return {
        "train_labels": tr_lab, "test_labels": te_lab,
        "train_idx": tr_idx, "train_src": src, "train_basenames": tr_b,
        "db": db, "qy": qy,
        "n_train_missing": int(missing), "train_mode": mode,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours_dir", required=True)
    ap.add_argument("--baseline_dirs", nargs="*", default=[],
                    help="dirs with extract_{db,query}.npz for flat-hash chunk controls")
    ap.add_argument("--baseline_names", nargs="*", default=[])
    ap.add_argument("--train_manifest", default=None,
                    help="only needed when the run has no extract_train.npz")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--min_support", type=int, default=DEFAULT_MIN_SUPPORT)
    ap.add_argument("--n_shuffle", type=int, default=DEFAULT_N_SHUFFLE)
    ap.add_argument("--n_boot", type=int, default=DEFAULT_N_BOOT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bio_project", action="store_true",
                    help="project every 18-base DNA code to bio-valid (GC window "
                         "by code length, homopolymer <= 3) before codon "
                         "extraction, matching the deployed valid-DNA codes. "
                         "Baseline chunk control becomes the projected per-slot "
                         "codon of the baseline's own DNA code.")
    # F07: default None -> resolved from the actual code length by the
    # central policy. The old 0.4444/0.5556 gave L=15 the window [7,8]
    # while the main path projected onto [6,9].
    ap.add_argument("--gc_min_frac", type=float, default=None)
    ap.add_argument("--gc_max_frac", type=float, default=None)
    ap.add_argument("--max_run", type=int, default=3)
    args = ap.parse_args()

    # Infer the slot count BEFORE anything reads N_SLOTS. `codebook_indices` is
    # [N, M] with one entry per slot, so it is the authoritative source; the
    # base count alone is ambiguous (15 bases could be 5x3 or 3x5).
    import numpy as _np
    _probe = _np.load(os.path.join(args.ours_dir, "extract_query.npz"), allow_pickle=True)
    if "codebook_indices" in _probe:
        _set_n_slots(int(_probe["codebook_indices"].shape[1]))
        print(f"[{args.dataset}] slots inferred from extraction: {N_SLOTS}")

    d = load_split(args.ours_dir, args.train_manifest)
    tr_lab, te_lab = d["train_labels"], d["test_labels"]
    tr_idx, src, db, qy = d["train_idx"], d["train_src"], d["db"], d["qy"]
    n_bases = db["base_indices"].shape[1] // N_SLOTS

    print(f"[{args.dataset}] train={len(tr_lab)} ({d['train_mode']}) "
          f"test={len(te_lab)} labels={tr_lab.shape[1]} bases/slot={n_bases} "
          f"(train rows not found: {d['n_train_missing']})")

    # bio-projection (2026-07-21 invariant): project the 18-base DNA sequence to
    # bio-valid before codon extraction, so decoding uses the deployed valid-DNA
    # codes. codebook_indices are NOT touched, so ours_codeword is invariant.
    def _bioproj(base_arr):
        if not args.bio_project:
            return base_arr
        from dna_utils.bio_constraints import project_to_valid, is_valid_batch
        from dna_utils.gc_policy import resolve_gc_policy
        arr = np.ascontiguousarray(base_arr).astype(np.int8)
        # F07: resolve the window from the ACTUAL code length. The old default
        # 0.4444/0.5556 projected a 15-base code onto [7,8] while the main path
        # used [6,9], so decoding was scored on a different feasible set than
        # the retrieval numbers it was compared against.
        _pol = resolve_gc_policy(int(arr.shape[1]))
        _gmin = args.gc_min_frac if args.gc_min_frac is not None else _pol.gc_min_frac
        _gmax = args.gc_max_frac if args.gc_max_frac is not None else _pol.gc_max_frac
        _mrun = args.max_run if args.max_run is not None else _pol.max_run
        uniq, inv = np.unique(arr, axis=0, return_inverse=True)
        valid = is_valid_batch(uniq, _gmin, _gmax, _mrun)
        out = uniq.copy()
        for i in np.where(~valid)[0]:
            out[i], _ = project_to_valid(uniq[i], _gmin, _gmax, _mrun)
        return out[inv].astype(np.int64)

    tag_bp = " [bio-projected]" if args.bio_project else ""
    tr_base = _bioproj(src["base_indices"][tr_idx])
    qy_base = _bioproj(qy["base_indices"])

    K = int(max(src["codebook_indices"].max(), qy["codebook_indices"].max())) + 1
    tr_cw = src["codebook_indices"][tr_idx]
    units = {
        "ours_codeword": (tr_cw, qy["codebook_indices"], K),
        "ours_codon": (codon_ids(tr_base, n_bases),
                       codon_ids(qy_base, n_bases), 4 ** n_bases),
    }
    print(f"[{args.dataset}]{tag_bp} decoding units built")

    results: Dict[str, dict] = {}
    for name, (tu, qu, n_u) in units.items():
        results[name] = decode_all_slots(tu, tr_lab, qu, te_lab, n_u,
                                         args.alpha, args.min_support)
        print(f"  {name:16s} mAP={results[name]['slot_mean']['concept_mAP']:.4f} "
              f"cov={results[name]['slot_mean']['coverage']:.3f}")

    # flat-hash chunk controls
    for bdir, bname in zip(args.baseline_dirs, args.baseline_names):
        # Train-side pool: an explicit train extraction when the baseline ships
        # one (required for MSCOCO, whose train split is disjoint from its DB),
        # otherwise the DB extraction, which contains train as a subset
        # (Flickr25k, NUS-WIDE).
        btr_path = os.path.join(bdir, "extract_train.npz")
        bsrc_path = btr_path if os.path.exists(btr_path) \
            else os.path.join(bdir, "extract_db.npz")
        bdb = np.load(bsrc_path, allow_pickle=True)
        bqy = np.load(os.path.join(bdir, "extract_query.npz"), allow_pickle=True)
        # realign to OUR train/test rows by basename so the comparison is paired
        bpos = {b: i for i, b in enumerate(_basenames(bdb["image_paths"]))}
        qpos = {b: i for i, b in enumerate(_basenames(bqy["image_paths"]))}
        tr_b = np.array([bpos[b] for b in d["train_basenames"]])
        te_b = np.array([qpos[b] for b in _basenames(qy["image_paths"])])
        key = f"{bname}_chunk"
        if args.bio_project:
            # DNA-space-consistent control: the baseline's own sign code -> one
            # base per 2 bits -> project to bio-valid -> per-slot 3-base codon,
            # decoded exactly like ours. (Same 64 values/slot as the bit chunk.)
            #
            # The base count comes from the baseline's own width, NOT a constant:
            # it was 18 while the budget was 36 bits, and is 15 at 30 bits. A
            # baseline at a different budget from ours cannot be chunked into
            # N_SLOTS codons at all, so say which pair mismatched rather than
            # letting reshape fail with a bare size error.
            _n_base = bdb["hash_2bit"].shape[1] // 2
            if _n_base != N_SLOTS * n_bases:
                raise SystemExit(
                    f"baseline '{bname}' is {bdb['hash_2bit'].shape[1]}-bit "
                    f"({_n_base} bases) but ours is {N_SLOTS} slots x {n_bases} "
                    f"bases = {N_SLOTS * n_bases}; pass control dirs at the "
                    f"matching budget")

            def _hash_to_base(h2, _nb=_n_base):
                b = (h2 > 0).astype(np.int64).reshape(len(h2), _nb, 2)
                return b[:, :, 0] * 2 + b[:, :, 1]
            btr_codon = codon_ids(_bioproj(_hash_to_base(bdb["hash_2bit"][tr_b])), n_bases)
            bte_codon = codon_ids(_bioproj(_hash_to_base(bqy["hash_2bit"][te_b])), n_bases)
            results[key] = decode_all_slots(btr_codon, tr_lab, bte_codon, te_lab,
                                            4 ** n_bases, args.alpha, args.min_support)
        else:
            results[key] = decode_all_slots(
                bit_chunk_ids(bdb["hash_2bit"][tr_b]), tr_lab,
                bit_chunk_ids(bqy["hash_2bit"][te_b]), te_lab,
                2 ** (bdb["hash_2bit"].shape[1] // N_SLOTS), args.alpha, args.min_support)
        print(f"  {key:16s} mAP={results[key]['slot_mean']['concept_mAP']:.4f} "
              f"cov={results[key]['slot_mean']['coverage']:.3f}")

    results["majority"] = majority_control(tr_lab, te_lab, args.alpha)
    print(f"  {'majority':16s} mAP={results['majority']['concept_mAP']:.4f}")

    results["shuffled"] = shuffled_control(
        units["ours_codon"][0], tr_lab, units["ours_codon"][1], te_lab,
        units["ours_codon"][2], args.alpha, args.min_support,
        args.n_shuffle, args.seed)
    print(f"  {'shuffled':16s} mAP={results['shuffled']['concept_mAP']:.4f} "
          f"+-{results['shuffled']['concept_mAP_std']:.4f}")

    # §2.8 collision accounting
    results["collision"] = {
        "decoding_loss_codeword_minus_codon":
            results["ours_codeword"]["slot_mean"]["concept_mAP"]
            - results["ours_codon"]["slot_mean"]["concept_mAP"],
        "entropy_increase_codon_minus_codeword":
            results["ours_codon"]["slot_mean"]["H_cond"]
            - results["ours_codeword"]["slot_mean"]["H_cond"],
        "per_slot_jsd": [
            codeword_codon_jsd(tr_cw[:, m],
                               units["ours_codon"][0][:, m], tr_lab, K,
                               args.alpha, args.min_support)
            for m in range(N_SLOTS)
        ],
    }

    # §2.11 paired bootstrap of ours-codon against every control
    ref = results["ours_codon"]["_ap_per_sample"]
    results["bootstrap_vs_ours_codon"] = {
        name: paired_bootstrap(ref, results[name]["_ap_per_sample"],
                               args.n_boot, args.seed)
        for name in results
        if name != "ours_codon" and "_ap_per_sample" in results[name]
    }

    payload = {
        "dataset": args.dataset,
        "config": {
            "alpha": args.alpha, "min_support": args.min_support,
            "n_shuffle": args.n_shuffle, "n_boot": args.n_boot, "seed": args.seed,
            "n_train": int(len(tr_lab)), "n_test": int(len(te_lab)),
            "n_labels": int(tr_lab.shape[1]), "K": K, "n_bases_per_slot": n_bases,
            "ours_dir": args.ours_dir,
            "baselines": dict(zip(args.baseline_names, args.baseline_dirs)),
        },
        "slot_names": SLOT_NAMES,
        "results": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                    for k, v in results.items()},
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[{args.dataset}] wrote {args.out}")


if __name__ == "__main__":
    main()
