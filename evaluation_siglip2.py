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
import json
import os
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
        Lq = query_mh.astype(np.float32)
        Ld = db_mh.astype(np.float32)
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


def _ap_from_sorted_relevance(rel_sorted: np.ndarray) -> float:
    """Average Precision from a 0/1 vector already sorted by ascending distance."""
    nrel = int(rel_sorted.sum())
    if nrel == 0:
        return 0.0
    ranks = np.where(rel_sorted == 1)[0] + 1.0
    counts = np.arange(1, nrel + 1, dtype=np.float64)
    return float((counts / ranks).mean())


# ----------------------------------------------------------------- retrieval

def evaluate_retrieval(
    query: Dict[str, np.ndarray],
    db:    Dict[str, np.ndarray],
    distance_mode: str = "base",
    precision_at_k_list=(1, 5, 10, 20, 50, 100, 500, 1000),
    multi_label_relevance_threshold: float = 0.0,
    remove_self_match: bool = False,
) -> Dict[str, Any]:
    distances = _compute_distance(
        query["base_indices"], db["base_indices"],
        query.get("hash_2bit"), db.get("hash_2bit"),
        mode=distance_mode,
    )
    relevance = _compute_relevance(
        query.get("labels"), db.get("labels"),
        query.get("multi_hot_labels"), db.get("multi_hot_labels"),
        threshold=multi_label_relevance_threshold,
    )

    Nq, Nd = distances.shape
    aps: list = []
    p_at_k: Dict[int, list] = {k: [] for k in precision_at_k_list}
    r_at_k: Dict[int, list] = {k: [] for k in precision_at_k_list}
    pos_d:  list = []
    neg_d:  list = []

    for i in tqdm(range(Nq), desc=f"eval[{distance_mode}]"):
        d = distances[i]
        r = relevance[i].copy()
        if remove_self_match and i < Nd:
            # self-removal only meaningful when query and db are aligned
            r[i] = 0
        order = np.argsort(d, kind="stable")
        r_sorted = r[order]
        aps.append(_ap_from_sorted_relevance(r_sorted))
        nrel = int(r.sum())
        for k in precision_at_k_list:
            kk = min(k, Nd)
            topk = r_sorted[:kk]
            p_at_k[k].append(float(topk.mean()) if kk > 0 else 0.0)
            r_at_k[k].append(float(topk.sum() / max(nrel, 1)))
        if nrel > 0:
            pos_d.append(float(d[r == 1].mean()))
        if (r == 0).sum() > 0:
            neg_d.append(float(d[r == 0].mean()))

    out: Dict[str, Any] = {
        "mAP":             float(np.mean(aps)) if aps else 0.0,
        "precision_at_k":  {int(k): float(np.mean(v)) for k, v in p_at_k.items()},
        "recall_at_k":     {int(k): float(np.mean(v)) for k, v in r_at_k.items()},
        "pr_curve": {
            "k":         [int(k) for k in precision_at_k_list],
            "precision": [float(np.mean(p_at_k[k])) for k in precision_at_k_list],
            "recall":    [float(np.mean(r_at_k[k])) for k in precision_at_k_list],
        },
        "mean_positive_distance": float(np.mean(pos_d)) if pos_d else 0.0,
        "mean_negative_distance": float(np.mean(neg_d)) if neg_d else 0.0,
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
    K = int(codebook_size)
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
) -> Dict[str, Any]:
    db_npz_path = os.path.join(path, "extract_db.npz")
    qy_npz_path = os.path.join(path, "extract_query.npz")
    if not os.path.exists(db_npz_path) or not os.path.exists(qy_npz_path):
        raise FileNotFoundError(
            f"[evaluation] expected extract_db.npz and extract_query.npz in {path}."
        )
    db_npz = np.load(db_npz_path, allow_pickle=True)
    qy_npz = np.load(qy_npz_path, allow_pickle=True)
    db = {k: db_npz[k] for k in db_npz.files}
    qy = {k: qy_npz[k] for k in qy_npz.files}

    bio_stats: Dict[str, Any] = {}
    if bio_project:
        # also evaluate the un-projected baseline so we can report Δ.
        retrieval_pre = evaluate_retrieval(
            qy, db,
            distance_mode=distance_mode,
            precision_at_k_list=precision_at_k_list,
            multi_label_relevance_threshold=multi_label_relevance_threshold,
            remove_self_match=remove_self_match,
        )
        bio_stats["mAP_pre_projection"] = retrieval_pre["mAP"]
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
    )
    collapse = evaluate_code_collapse(db, codebook_size=codebook_size)

    result: Dict[str, Any] = {
        **retrieval,
        **collapse,
        "distance_mode": distance_mode,
        "codebook_size": int(codebook_size),
        "n_query":       int(qy["base_indices"].shape[0]),
        "n_db":          int(db["base_indices"].shape[0]),
    }
    if bio_project:
        result["bio_project"] = True
        result["bio_gc_min_frac"] = float(bio_gc_min_frac)
        result["bio_gc_max_frac"] = float(bio_gc_max_frac)
        result["bio_max_homopolymer_run"] = int(bio_max_homopolymer_run)
        result["bio_stats"] = bio_stats
        # tag the output filename so pre/post artifacts don't overwrite each other
        suffix = "_bioproj"
    else:
        suffix = ""
    out_json = os.path.join(path, f"evaluation_siglip2_{distance_mode}{suffix}.json")
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[evaluation] mAP({distance_mode}{suffix}) = {result['mAP']:.4f}")
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
    parser.add_argument("--remove_self_match", action="store_true")
    parser.add_argument("--multi_label_relevance_threshold", type=float, default=0.0)
    parser.add_argument("--bio_project", action="store_true",
        help="Apply bio-constraint minimum-edit projection to all DNA codes "
             "before retrieval evaluation.")
    parser.add_argument("--bio_gc_min_frac", type=float, default=0.40)
    parser.add_argument("--bio_gc_max_frac", type=float, default=0.60)
    parser.add_argument("--bio_max_homopolymer_run", type=int, default=3)
    cfg = parser.parse_args()
    evaluation(
        cfg.extraction_path,
        distance_mode=cfg.distance_mode,
        codebook_size=cfg.codebook_size,
        remove_self_match=cfg.remove_self_match,
        multi_label_relevance_threshold=cfg.multi_label_relevance_threshold,
        bio_project=cfg.bio_project,
        bio_gc_min_frac=cfg.bio_gc_min_frac,
        bio_gc_max_frac=cfg.bio_gc_max_frac,
        bio_max_homopolymer_run=cfg.bio_max_homopolymer_run,
    )
