"""Retrieval evaluation for ``SigLIP2SemanticOTModel`` extraction outputs.

Reads the .npz files produced by ``extraction_siglip2.py`` and computes:
    - retrieval mAP, P@K, R@K, PR-curve (top-k sweep)
    - sub-codebook entropy / perplexity / dead-code ratio
    - DNA base-position entropy
    - unique-code ratio / duplicate rate
    - mean positive vs negative distance (sanity)

Two distance modes are supported:
    base : base-level Hamming on `base_indices` (every base mismatch = 1)
    bit2 : 2-bit bit-level Hamming on `hash_2bit` (matches DB hash table)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from typing import Any, Dict, Optional

import numpy as np
import torch
from tqdm import tqdm

from dna_utils import (
    base_hamming_distance,
    bit_hamming_distance_2bit,
)


# ----------------------------------------------------------------- helpers

def _compute_relevance(
    query_labels: Optional[np.ndarray],
    db_labels: Optional[np.ndarray],
    query_mh: Optional[np.ndarray],
    db_mh: Optional[np.ndarray],
    threshold: float = 0.0,
) -> np.ndarray:
    """Pairwise binary relevance matrix [Nq, Nd], values in {0, 1}."""
    if query_mh is not None and db_mh is not None:
        Lq = query_mh.astype(np.float32, copy=False)
        Ld = db_mh.astype(np.float32, copy=False)
        intersection = Lq @ Ld.T                         # [Nq, Nd]
        cq = Lq.sum(axis=1, keepdims=True)
        cd = Ld.sum(axis=1, keepdims=True).T
        union = cq + cd - intersection
        S = np.where(union > 0, intersection / np.maximum(union, 1e-12), 0.0)
        return (S > threshold).astype(np.uint8)
    if query_labels is not None and db_labels is not None:
        return (query_labels[:, None] == db_labels[None, :]).astype(np.uint8)
    raise ValueError(
        "[evaluation] need labels or multi_hot_labels for both query and db."
    )


def _compute_distance(
    query_base: np.ndarray,
    db_base: np.ndarray,
    query_2bit: Optional[np.ndarray],
    db_2bit: Optional[np.ndarray],
    mode: str,
) -> np.ndarray:
    """Return [Nq, Nd] distance matrix according to ``mode``."""
    if mode == "base":
        qb = torch.from_numpy(np.ascontiguousarray(query_base)).long()
        db = torch.from_numpy(np.ascontiguousarray(db_base)).long()
        return base_hamming_distance(qb, db).numpy()
    if mode == "bit2":
        if query_2bit is None or db_2bit is None:
            raise ValueError("[evaluation] mode='bit2' requires hash_2bit on both sides.")
        qb = torch.from_numpy(np.ascontiguousarray(query_2bit)).long()
        db = torch.from_numpy(np.ascontiguousarray(db_2bit)).long()
        return bit_hamming_distance_2bit(qb, db).numpy()
    raise ValueError(f"[evaluation] unknown distance_mode={mode!r}")


# Paper-standard mAP@R cutoffs (deep-hashing convention). Report mAP truncated
# at the top-R retrieved items per dataset — matches CIBHash / HashNet / CSQ /
# DPSH benchmark tables (their `CalcTopMap(..., topk=R)`).
MAP_AT_R_BY_DATASET: Dict[str, int] = {
    "CIFAR10":   1000,
    "NUSWIDE":   5000,
    "MSCOCO":    5000,
    "Flickr25k": 5000,
}

EVALUATION_SCHEMA_VERSION = 2
RANKING_POLICY = "stable_ascending_full_database_exact"
AP_NORMALIZATION_POLICY = "relevant_hits_within_top_R"
RELEVANCE_POLICY = "shared_label_jaccard_gt_threshold_or_single_label_equal"


def resolve_map_at_r(dataset_name: Optional[str]) -> Optional[int]:
    """Return the paper-standard mAP@R cutoff for a dataset (None if unknown)."""
    if dataset_name is None:
        return None
    return MAP_AT_R_BY_DATASET.get(str(dataset_name))


def _sha256_open_file(handle) -> str:
    digest = hashlib.sha256()
    handle.seek(0)
    while True:
        chunk = handle.read(1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    return digest.hexdigest()


def _load_npz_bound(path: str) -> tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Load an NPZ without pickle and bind arrays to the bytes consumed.

    Hashing a pathname and reopening it for ``np.load`` leaves a replacement
    race.  Hash and load through one read-only descriptor, re-hash after the
    load, then verify that the pathname still resolves to that descriptor's
    inode.  The metric record therefore names the bytes that produced it.
    """
    absolute = os.path.abspath(path)
    fd = os.open(absolute, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        with os.fdopen(fd, "rb", closefd=False) as handle:
            before_stat = os.fstat(handle.fileno())
            before_digest = _sha256_open_file(handle)
            handle.seek(0)
            with np.load(handle, allow_pickle=False) as stored:
                arrays = {key: np.asarray(stored[key]) for key in stored.files}
            after_stat = os.fstat(handle.fileno())
            after_digest = _sha256_open_file(handle)
    finally:
        os.close(fd)

    stat_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before_stat, field) != getattr(after_stat, field)
           for field in stat_fields) or before_digest != after_digest:
        raise RuntimeError(
            f"[evaluation] extraction changed while it was being read: {absolute}")
    current = os.stat(absolute, follow_symlinks=True)
    if any(getattr(after_stat, field) != getattr(current, field)
           for field in stat_fields):
        raise RuntimeError(
            f"[evaluation] extraction pathname was replaced while it was being "
            f"read: {absolute}")
    return arrays, {
        "path": absolute,
        "resolved_path": os.path.realpath(absolute),
        "sha256": after_digest,
        "size_bytes": int(after_stat.st_size),
        "st_dev": int(after_stat.st_dev),
        "st_ino": int(after_stat.st_ino),
        "st_mtime_ns": int(after_stat.st_mtime_ns),
        "st_ctime_ns": int(after_stat.st_ctime_ns),
    }


def _require_finite_json(value: Any, *, where: str = "result") -> None:
    """Reject NaN/Infinity before they become non-standard JSON tokens."""
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, np.integer)):
        return
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            raise ValueError(f"[evaluation] {where} is not finite: {value!r}")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            _require_finite_json(child, where=f"{where}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _require_finite_json(child, where=f"{where}[{index}]")
        return
    raise TypeError(
        f"[evaluation] {where} has non-JSON type {type(value).__name__}")


def _require_proportion(result: Dict[str, Any], key: str) -> None:
    value = result.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"[evaluation] {key} is not a numeric proportion")
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(
            f"[evaluation] {key}={value!r} lies outside the closed interval [0,1]")


def _atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    """Publish strict JSON in one rename, preserving any prior valid file."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        dir_fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _ap_at_r(rel_sorted: np.ndarray, R: Optional[int] = None) -> float:
    """Average Precision truncated at the top-R retrieved items.

    Canonical deep-hashing `CalcTopMap` convention (HashNet/DPSH/CSQ/CIBHash):
    normalize by the number of relevant items *found within the top-R*, i.e.
    ``mean(rank_count / rank_index)`` over the relevant hits in ``rel_sorted[:R]``.
    With ``R=None`` (or R>=len) this reduces exactly to the full AP.
    """
    tgnd = rel_sorted if R is None else rel_sorted[:R]
    tsum = int(tgnd.sum())
    if tsum == 0:
        return 0.0
    tindex = np.where(tgnd == 1)[0] + 1.0
    counts = np.arange(1, tsum + 1, dtype=np.float64)
    return float((counts / tindex).mean())


def _ap_from_sorted_relevance(rel_sorted: np.ndarray) -> float:
    """Average Precision from a 0/1 vector already sorted by ascending distance.

    Full-ranking AP (== ``_ap_at_r`` with R = len(rel_sorted)).
    """
    return _ap_at_r(rel_sorted, R=None)


# ----------------------------------------------------------------- retrieval

def evaluate_retrieval(
    query: Dict[str, np.ndarray],
    db:    Dict[str, np.ndarray],
    distance_mode: str = "base",
    precision_at_k_list=(1, 5, 10, 20, 50, 100, 500, 1000),
    multi_label_relevance_threshold: float = 0.0,
    remove_self_match: bool = False,
    map_at_r: Optional[int] = None,
    query_chunk_size: int = 64,
) -> Dict[str, Any]:
    """Evaluate exact retrieval while allocating only ``chunk x Nd`` pairs.

    Query chunking changes memory use only.  Every query is still compared
    with, and stable-ranked against, the complete database.  Consequently no
    database-side chunk merge or approximate top-k can alter global ranks.
    """
    if (isinstance(query_chunk_size, bool)
            or not isinstance(query_chunk_size, int)
            or query_chunk_size <= 0):
        raise ValueError("query_chunk_size must be a positive integer")

    query_base = np.asarray(query["base_indices"])
    db_base = np.asarray(db["base_indices"])
    if query_base.ndim != 2 or db_base.ndim != 2:
        raise ValueError("base_indices must be rank-2 [N, L] arrays")
    Nq = int(query_base.shape[0])
    Nd = int(db_base.shape[0])

    query_2bit = query.get("hash_2bit")
    db_2bit = db.get("hash_2bit")
    query_labels = query.get("labels")
    db_labels = db.get("labels")
    query_mh = query.get("multi_hot_labels")
    db_mh = db.get("multi_hot_labels")
    if (query_labels is None) != (db_labels is None) \
            or (query_mh is None) != (db_mh is None) \
            or (query_labels is None and query_mh is None):
        raise ValueError(
            "query/database must carry the same non-empty label representation")
    if query_labels is not None:
        query_labels = np.asarray(query_labels)
        db_labels = np.asarray(db_labels)
        if query_labels.shape != (Nq,) or db_labels.shape != (Nd,) \
                or not np.issubdtype(query_labels.dtype, np.integer) \
                or not np.issubdtype(db_labels.dtype, np.integer) \
                or (query_labels.size and query_labels.min() < 0) \
                or (db_labels.size and db_labels.min() < 0):
            raise ValueError(
                "query/database labels must be non-negative integer [N] arrays")
    if query_mh is not None:
        query_mh_raw = np.asarray(query_mh)
        db_mh_raw = np.asarray(db_mh)
        if query_mh_raw.ndim != 2 or db_mh_raw.ndim != 2 \
                or query_mh_raw.shape[0] != Nq or db_mh_raw.shape[0] != Nd \
                or query_mh_raw.shape[1] <= 0 \
                or query_mh_raw.shape[1] != db_mh_raw.shape[1] \
                or not (np.issubdtype(query_mh_raw.dtype, np.integer)
                        or np.issubdtype(query_mh_raw.dtype, np.bool_)) \
                or not (np.issubdtype(db_mh_raw.dtype, np.integer)
                        or np.issubdtype(db_mh_raw.dtype, np.bool_)) \
                or not np.isfinite(query_mh_raw).all() \
                or not np.isfinite(db_mh_raw).all() \
                or not np.isin(query_mh_raw, (0, 1)).all() \
                or not np.isin(db_mh_raw, (0, 1)).all():
            raise ValueError(
                "query/database multi_hot_labels must be finite binary [N,C] "
                "arrays with matching C")
        query_mh = query_mh_raw.astype(np.float32, copy=False)
        db_mh = db_mh_raw.astype(np.float32, copy=False)
    if query_labels is not None and query_mh is not None:
        if query_labels.size and query_labels.max() >= query_mh.shape[1] \
                or db_labels.size and db_labels.max() >= db_mh.shape[1] \
                or not np.array_equal(query_mh.sum(axis=1), np.ones(Nq)) \
                or not np.array_equal(db_mh.sum(axis=1), np.ones(Nd)) \
                or not np.array_equal(query_mh.argmax(axis=1), query_labels) \
                or not np.array_equal(db_mh.argmax(axis=1), db_labels):
            raise ValueError("labels and multi_hot_labels disagree")

    aps: list = []
    aps_at_r: list = []
    p_at_k: Dict[int, list] = {k: [] for k in precision_at_k_list}
    r_at_k: Dict[int, list] = {k: [] for k in precision_at_k_list}
    # Preserve the historical diagnostic contract: these fields first take a
    # mean within each query, then give every non-empty query equal weight.
    # Pair-weighted variants are accumulated separately and exposed under
    # explicit names below.
    positive_query_mean_distances: list = []
    negative_query_mean_distances: list = []
    positive_distance_sum = 0
    negative_distance_sum = 0
    positive_pair_count = 0
    negative_pair_count = 0

    starts = range(0, Nq, query_chunk_size)
    for start in tqdm(starts, desc=f"eval[{distance_mode}]"):
        stop = min(start + query_chunk_size, Nq)
        distances = _compute_distance(
            query_base[start:stop], db_base,
            None if query_2bit is None else query_2bit[start:stop], db_2bit,
            mode=distance_mode,
        )
        relevance = _compute_relevance(
            None if query_labels is None else query_labels[start:stop],
            db_labels,
            None if query_mh is None else query_mh[start:stop],
            db_mh,
            threshold=multi_label_relevance_threshold,
        )
        if distances.shape != (stop - start, Nd):
            raise ValueError(
                "distance chunk shape mismatch: expected "
                f"{(stop - start, Nd)}, found {distances.shape}")
        if relevance.shape != distances.shape:
            raise ValueError(
                "relevance chunk shape does not match distance chunk: "
                f"{relevance.shape} != {distances.shape}")

        for local_i, global_i in enumerate(range(start, stop)):
            d = distances[local_i]
            r = relevance[local_i]
            if remove_self_match and global_i < Nd:
                # This option is only meaningful for an aligned same-split
                # query/database evaluation.  Remove the row from the candidate
                # set itself; rel=0 would leave a zero-distance false candidate
                # at the head of the stable ranking.
                keep = np.ones(Nd, dtype=bool)
                keep[global_i] = False
                d = d[keep]
                r = r[keep]

            order = np.argsort(d, kind="stable")
            r_sorted = r[order]
            aps.append(_ap_from_sorted_relevance(r_sorted))
            if map_at_r is not None:
                aps_at_r.append(_ap_at_r(r_sorted, R=int(map_at_r)))
            nrel = int(r.sum())
            candidate_count = int(r.shape[0])
            for k in precision_at_k_list:
                kk = min(k, candidate_count)
                topk = r_sorted[:kk]
                p_at_k[k].append(float(topk.mean()) if kk > 0 else 0.0)
                r_at_k[k].append(float(topk.sum() / max(nrel, 1)))

            positive = r == 1
            negative = r == 0
            n_positive = int(positive.sum())
            n_negative = int(negative.sum())
            if n_positive:
                positive_query_mean_distances.append(
                    float(d[positive].mean()))
                positive_distance_sum += int(
                    d[positive].sum(dtype=np.int64))
                positive_pair_count += n_positive
            if n_negative:
                negative_query_mean_distances.append(
                    float(d[negative].mean()))
                negative_distance_sum += int(
                    d[negative].sum(dtype=np.int64))
                negative_pair_count += n_negative

    out: Dict[str, Any] = {
        "mAP":             float(np.mean(aps)) if aps else 0.0,
        "precision_at_k":  {int(k): float(np.mean(v)) for k, v in p_at_k.items()},
        **({
            "mAP_at_R":     float(np.mean(aps_at_r)) if aps_at_r else 0.0,
            "mAP_R_cutoff": int(map_at_r),
        } if map_at_r is not None else {}),
        "recall_at_k":     {int(k): float(np.mean(v)) for k, v in r_at_k.items()},
        "pr_curve": {
            "k":         [int(k) for k in precision_at_k_list],
            "precision": [float(np.mean(p_at_k[k])) for k in precision_at_k_list],
            "recall":    [float(np.mean(r_at_k[k])) for k in precision_at_k_list],
        },
        "mean_positive_distance": (
            float(np.mean(positive_query_mean_distances))
            if positive_query_mean_distances else 0.0),
        "mean_negative_distance": (
            float(np.mean(negative_query_mean_distances))
            if negative_query_mean_distances else 0.0),
        "pair_weighted_mean_positive_distance": (
            float(positive_distance_sum / positive_pair_count)
            if positive_pair_count else 0.0),
        "pair_weighted_mean_negative_distance": (
            float(negative_distance_sum / negative_pair_count)
            if negative_pair_count else 0.0),
    }
    return out


# ----------------------------------------------------------------- collapse

def evaluate_code_collapse(
    extraction: Dict[str, np.ndarray],
    codebook_size: int,
    num_bases: int = 4,
    num_dna_positions: int = 18,
) -> Dict[str, Any]:
    cb = extraction["codebook_indices"]   # [N, M=6]
    bi = extraction["base_indices"]        # [N, R=18]
    N, M = cb.shape
    # v78a: codebook can carry indices > codebook_size when adaptive K
    # split has been applied. Detect actual K from data.
    K = max(int(codebook_size), int(cb.max()) + 1) if N > 0 else int(codebook_size)
    R = bi.shape[1]
    if R != num_dna_positions:
        # not fatal; just adapt
        num_dna_positions = R
    eps = 1e-12

    cb_entropy        = np.zeros(M)
    cb_norm_entropy   = np.zeros(M)
    cb_perplexity     = np.zeros(M)
    dead_code_ratio   = np.zeros(M)
    counts_all        = np.zeros((M, K), dtype=np.int64)
    for m in range(M):
        counts = np.bincount(cb[:, m].astype(np.int64), minlength=K).astype(np.float64)
        counts_all[m] = counts.astype(np.int64)
        prob = counts / max(counts.sum(), 1)
        H = float(-np.sum(prob * np.log(prob + eps)))
        cb_entropy[m]      = H
        cb_norm_entropy[m] = H / max(np.log(K), eps)
        cb_perplexity[m]   = float(np.exp(H))
        dead_code_ratio[m] = float((counts == 0).sum() / K)

    base_counts = np.zeros((num_dna_positions, num_bases), dtype=np.int64)
    base_entropy = np.zeros(num_dna_positions)
    base_norm_entropy = np.zeros(num_dna_positions)
    for r in range(num_dna_positions):
        c = np.bincount(bi[:, r].astype(np.int64), minlength=num_bases).astype(np.float64)
        base_counts[r] = c.astype(np.int64)
        p = c / max(c.sum(), 1)
        H = float(-np.sum(p * np.log(p + eps)))
        base_entropy[r]      = H
        base_norm_entropy[r] = H / max(np.log(num_bases), eps)

    # unique sequences — treat each row as an int tuple, count distinct.
    # NB: round-tripping through np.array(..., dtype=object).tolist() converts
    # tuples back to lists (unhashable), so just build the set directly.
    unique_count = len({tuple(row) for row in bi.tolist()})
    unique_ratio = float(unique_count / N) if N > 0 else 0.0
    duplicate_rate = float(1.0 - unique_ratio)

    # ---- per-codebook unique-rate (compositional measure, v31) -----
    # Reshape [N, R=18] -> [N, M=6, R/M=3]. For each codebook m, count
    # how many distinct 3-base sequences appear in the split. Strong
    # per-codebook unique_rate => each codebook independently
    # discriminates samples (compositional independence). Weak =>
    # codebooks are mostly redundant or collapsed.
    per_cb_unique_count: list = [0] * M
    per_cb_unique_ratio: list = [0.0] * M
    if R % M == 0:
        bi_per_cb = bi.reshape(N, M, R // M)
        for m in range(M):
            uniq = {tuple(row) for row in bi_per_cb[:, m, :].tolist()}
            per_cb_unique_count[m] = int(len(uniq))
            per_cb_unique_ratio[m] = float(len(uniq) / N) if N > 0 else 0.0
    mean_per_cb_unique_ratio = (
        float(np.mean(per_cb_unique_ratio)) if per_cb_unique_ratio else 0.0
    )

    return {
        "codebook_entropy":             cb_entropy.tolist(),
        "codebook_normalized_entropy":  cb_norm_entropy.tolist(),
        "codebook_perplexity":          cb_perplexity.tolist(),
        "dead_code_ratio":              dead_code_ratio.tolist(),
        "codeword_usage_counts":        counts_all.tolist(),
        "base_entropy":                 base_entropy.tolist(),
        "base_normalized_entropy":      base_norm_entropy.tolist(),
        "mean_base_entropy":            float(np.mean(base_entropy)),
        "mean_base_normalized_entropy": float(np.mean(base_norm_entropy)),
        "unique_code_ratio":            unique_ratio,
        "duplicate_rate":               duplicate_rate,
        "per_codebook_unique_count":    per_cb_unique_count,
        "per_codebook_unique_ratio":    per_cb_unique_ratio,
        "mean_per_codebook_unique_ratio": mean_per_cb_unique_ratio,
    }


# ----------------------------------------------------------------- main

def _apply_bio_projection(
    extract: Dict[str, np.ndarray],
    gc_min_frac: float,
    gc_max_frac: float,
    max_run:     int,
    tag:         str,
) -> Dict[str, Any]:
    """Project all DNA codes in `extract` to bio-valid via DP minimum-edit.

    Mutates `extract['base_indices']` and (if present) `extract['hash_2bit']`
    so downstream retrieval uses the projected codes. Returns a stats dict.
    """
    from dna_utils import (
        batch_project_to_valid,
        is_valid_batch,
        base_indices_to_2bit_flat,
    )
    import torch as _t
    bi = np.ascontiguousarray(extract["base_indices"]).astype(np.int8)        # [N, L]
    pre_valid = is_valid_batch(bi, gc_min_frac, gc_max_frac, max_run)
    pre_compliance = float(pre_valid.mean())
    res = batch_project_to_valid(
        bi, gc_min_frac=gc_min_frac, gc_max_frac=gc_max_frac, max_run=max_run,
    )
    extract["base_indices"] = res["projected_codes"].astype(np.int64)
    # rebuild hash_2bit from the projected base indices so bit2-mode distance
    # stays consistent.
    if "hash_2bit" in extract:
        bi_t  = _t.from_numpy(extract["base_indices"]).long()
        extract["hash_2bit"] = base_indices_to_2bit_flat(bi_t).numpy().astype(np.uint8)
    edits = res["edit_distances"]
    succ_mask = edits >= 0
    stats = {
        f"{tag}_pre_compliance":   pre_compliance,
        f"{tag}_post_compliance":  res["compliance_rate"],
        f"{tag}_mean_edit_distance":   res["mean_edit_distance"],
        f"{tag}_max_edit_distance":    int(edits[succ_mask].max()) if succ_mask.any() else 0,
        f"{tag}_num_total":         int(bi.shape[0]),
        f"{tag}_num_pre_invalid":   int((~pre_valid).sum()),
        f"{tag}_num_proj_failed":   int((edits == -1).sum()),
    }
    print(
        f"[bio-project:{tag}] pre_compliance={pre_compliance:.4f} -> "
        f"post={res['compliance_rate']:.4f}; "
        f"mean_edit={res['mean_edit_distance']:.3f}, "
        f"max_edit={stats[f'{tag}_max_edit_distance']}, "
        f"invalid_in={stats[f'{tag}_num_pre_invalid']}/{stats[f'{tag}_num_total']}"
    )
    return stats


def evaluation(
    path: str,
    distance_mode: str = "base",
    codebook_size: int = 64,
    precision_at_k_list=(1, 5, 10, 20, 50, 100, 500, 1000),
    multi_label_relevance_threshold: float = 0.0,
    remove_self_match: bool = False,
    bio_project: bool = False,
    bio_gc_min_frac: float = 0.40,
    bio_gc_max_frac: float = 0.60,
    bio_max_homopolymer_run: int = 3,
    map_at_r: Optional[int] = None,
    query_chunk_size: int = 64,
    dataset_name: Optional[str] = None,
) -> Dict[str, Any]:
    if distance_mode not in {"base", "bit2"}:
        raise ValueError(f"[evaluation] unknown distance_mode={distance_mode!r}")
    if isinstance(codebook_size, bool) or not isinstance(codebook_size, int) \
            or codebook_size <= 0:
        raise ValueError("[evaluation] codebook_size must be a positive integer")
    if map_at_r is not None and (
            isinstance(map_at_r, bool) or not isinstance(map_at_r, int)
            or map_at_r <= 0):
        raise ValueError("[evaluation] map_at_r must be a positive integer")
    if not math.isfinite(float(multi_label_relevance_threshold)) or not (
            0.0 <= float(multi_label_relevance_threshold) <= 1.0):
        raise ValueError(
            "[evaluation] multi_label_relevance_threshold must be in [0,1]")
    if dataset_name is not None:
        expected_cutoff = resolve_map_at_r(dataset_name)
        if expected_cutoff is None:
            raise ValueError(
                f"[evaluation] no paper mAP@R policy for dataset {dataset_name!r}")
        if map_at_r != expected_cutoff:
            raise ValueError(
                f"[evaluation] dataset {dataset_name} requires mAP@R cutoff "
                f"{expected_cutoff}, found {map_at_r!r}")
    if bio_project:
        if (not math.isfinite(float(bio_gc_min_frac))
                or not math.isfinite(float(bio_gc_max_frac))
                or not 0.0 <= float(bio_gc_min_frac) <= float(bio_gc_max_frac) <= 1.0):
            raise ValueError(
                "[evaluation] bio GC fractions must satisfy 0 <= min <= max <= 1")
        if (isinstance(bio_max_homopolymer_run, bool)
                or not isinstance(bio_max_homopolymer_run, int)
                or bio_max_homopolymer_run <= 0):
            raise ValueError(
                "[evaluation] bio_max_homopolymer_run must be a positive integer")

    db_npz_path = os.path.join(path, "extract_db.npz")
    qy_npz_path = os.path.join(path, "extract_query.npz")
    if not os.path.exists(db_npz_path) or not os.path.exists(qy_npz_path):
        raise FileNotFoundError(
            f"[evaluation] expected extract_db.npz and extract_query.npz in {path}."
        )
    db, db_binding = _load_npz_bound(db_npz_path)
    qy, qy_binding = _load_npz_bound(qy_npz_path)
    for split, arrays in (("db", db), ("query", qy)):
        bases = np.asarray(arrays.get("base_indices"))
        if bases.ndim != 2 or bases.shape[0] <= 0 or bases.shape[1] <= 0:
            raise ValueError(
                f"[evaluation] {split} base_indices must be non-empty rank-2")
    if db["base_indices"].shape[1] != qy["base_indices"].shape[1]:
        raise ValueError(
            "[evaluation] query/database base-code widths do not match")

    bio_stats: Dict[str, Any] = {}
    if bio_project:
        # also evaluate the un-projected baseline so we can report Δ.
        retrieval_pre = evaluate_retrieval(
            qy, db,
            distance_mode=distance_mode,
            precision_at_k_list=precision_at_k_list,
            multi_label_relevance_threshold=multi_label_relevance_threshold,
            remove_self_match=remove_self_match,
            map_at_r=map_at_r,
            query_chunk_size=query_chunk_size,
        )
        bio_stats["mAP_pre_projection"] = retrieval_pre["mAP"]
        if "mAP_at_R" in retrieval_pre:
            bio_stats["mAP_at_R_pre_projection"] = retrieval_pre["mAP_at_R"]
            bio_stats["mAP_R_cutoff_pre_projection"] = retrieval_pre[
                "mAP_R_cutoff"]
        bio_stats.update(_apply_bio_projection(
            db, bio_gc_min_frac, bio_gc_max_frac, bio_max_homopolymer_run, tag="db",
        ))
        bio_stats.update(_apply_bio_projection(
            qy, bio_gc_min_frac, bio_gc_max_frac, bio_max_homopolymer_run, tag="qy",
        ))

    retrieval = evaluate_retrieval(
        qy, db,
        distance_mode=distance_mode,
        precision_at_k_list=precision_at_k_list,
        multi_label_relevance_threshold=multi_label_relevance_threshold,
        remove_self_match=remove_self_match,
        map_at_r=map_at_r,
        query_chunk_size=query_chunk_size,
    )
    collapse = evaluate_code_collapse(db, codebook_size=codebook_size)

    result: Dict[str, Any] = {
        **retrieval,
        **collapse,
        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
        "dataset": dataset_name,
        "distance_mode": distance_mode,
        "codebook_size": int(codebook_size),
        "n_query":       int(qy["base_indices"].shape[0]),
        "n_db":          int(db["base_indices"].shape[0]),
        "query_chunk_size": int(query_chunk_size),
        "bio_project": bool(bio_project),
        "input_artifacts": {
            "db": db_binding,
            "query": qy_binding,
        },
        "metric_policy": {
            "ranking": RANKING_POLICY,
            "ap_normalization": AP_NORMALIZATION_POLICY,
            "relevance": RELEVANCE_POLICY,
            "multi_label_relevance_threshold": float(
                multi_label_relevance_threshold),
            "remove_self_match": bool(remove_self_match),
            "map_at_r": None if map_at_r is None else int(map_at_r),
            "query_chunk_size": int(query_chunk_size),
        },
    }
    if bio_project:
        result["bio_gc_min_frac"] = float(bio_gc_min_frac)
        result["bio_gc_max_frac"] = float(bio_gc_max_frac)
        result["bio_max_homopolymer_run"] = int(bio_max_homopolymer_run)
        result["bio_stats"] = bio_stats
        # tag the output filename so pre/post artifacts don't overwrite each other
        suffix = "_bioproj"
    else:
        suffix = ""

    _require_finite_json(result)
    for key in ("mAP", "unique_code_ratio", "duplicate_rate"):
        _require_proportion(result, key)
    if map_at_r is not None:
        _require_proportion(result, "mAP_at_R")
        if result.get("mAP_R_cutoff") != map_at_r:
            raise ValueError(
                "[evaluation] reducer returned a different mAP@R cutoff than "
                "the requested protocol")
    out_json = os.path.join(path, f"evaluation_siglip2_{distance_mode}{suffix}.json")
    _atomic_write_json(out_json, result)
    print(f"[evaluation] mAP({distance_mode}{suffix}) = {result['mAP']:.4f}")
    if "mAP_at_R" in result:
        print(f"[evaluation] mAP@{result['mAP_R_cutoff']}({distance_mode}{suffix}) "
              f"= {result['mAP_at_R']:.4f}  (PAPER metric)")
    if bio_project:
        print(f"[evaluation] mAP delta from bio-projection: "
              f"{result['mAP'] - bio_stats['mAP_pre_projection']:+.4f}")
    print(f"[evaluation] saved -> {out_json}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--extraction_path", required=True)
    parser.add_argument("--distance_mode", default="base", choices=["base", "bit2"])
    parser.add_argument("--codebook_size", type=int, default=64)
    parser.add_argument(
        "--dataset", choices=sorted(MAP_AT_R_BY_DATASET), default=None,
        help=("Canonical dataset name. When set, the evaluator enforces and "
              "records that dataset's exact paper mAP@R cutoff."))
    parser.add_argument(
        "--map_at_r", type=int, default=None,
        help=("Explicit diagnostic cutoff. With --dataset it must equal the "
              "canonical cutoff; otherwise --dataset supplies the cutoff."))
    parser.add_argument("--remove_self_match", action="store_true")
    parser.add_argument("--multi_label_relevance_threshold", type=float, default=0.0)
    parser.add_argument(
        "--query_chunk_size", type=int, default=64,
        help=("Number of queries per exact retrieval chunk. Every chunk still "
              "compares against the full database."))
    # Biological-constraint projection is a MANDATORY, ALWAYS-ON step of the
    # method (2026-07-21): the paper reports DNA hashing, so every emitted code
    # must be a valid DNA strand (GC in [40,60]%, homopolymer run <= 3). The
    # Hamming-minimum DP projection enforces this on both query and DB codes
    # before retrieval, for our model AND the baselines symmetrically. Pass
    # --no-bio_project only for diagnostic pre-projection numbers.
    parser.add_argument("--bio_project", action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply bio-constraint minimum-edit projection to all DNA codes "
             "before retrieval evaluation. ON by default (mandatory invariant); "
             "use --no-bio_project to disable for diagnostics only.")
    parser.add_argument("--bio_gc_min_frac", type=float, default=0.40)
    parser.add_argument("--bio_gc_max_frac", type=float, default=0.60)
    parser.add_argument("--bio_max_homopolymer_run", type=int, default=3)
    cfg = parser.parse_args()
    cutoff = cfg.map_at_r
    if cfg.dataset is not None and cutoff is None:
        cutoff = resolve_map_at_r(cfg.dataset)
    evaluation(
        cfg.extraction_path,
        distance_mode=cfg.distance_mode,
        codebook_size=cfg.codebook_size,
        remove_self_match=cfg.remove_self_match,
        multi_label_relevance_threshold=cfg.multi_label_relevance_threshold,
        map_at_r=cutoff,
        query_chunk_size=cfg.query_chunk_size,
        dataset_name=cfg.dataset,
        bio_project=cfg.bio_project,
        bio_gc_min_frac=cfg.bio_gc_min_frac,
        bio_gc_max_frac=cfg.bio_gc_max_frac,
        bio_max_homopolymer_run=cfg.bio_max_homopolymer_run,
    )
