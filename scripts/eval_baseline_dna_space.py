"""Fair DNA-space evaluation of a flat baseline extraction.

Takes an extraction produced by `scripts/baseline_extract_splits.py` (which
writes `extract_query.npz` + `extract_db.npz`, each carrying a 36-bit
`hash_2bit` in {0,1} plus `multi_hot_labels`), and evaluates it TWO ways on the
identical query/db splits:

  1. BIT space  -- 36-bit Hamming (signed-binary equivalent).  This is a pure
     sanity check: it must reproduce the P0 table's `test_mAP_at_R` for the same
     (method, dataset, E*) cell to within ~1e-3.  If it does not, the epoch or
     cache is wrong and the run is invalid.

  2. BASE (DNA) space -- reshape [N,36] -> [N,18,2], map each 2-bit pair to an
     A/C/G/T base id (BASE_TO_BITS: 00=A,01=C,10=G,11=T ; base = hi*2+lo), then
     18-position base Hamming.  This is the space our own model is evaluated in,
     so it is the fair common ground for the paper's DNA-hashing claim.

Metric is CalcTopMap mAP@R with the dataset cutoff, multi-hot Jaccard(>0)
relevance -- reusing baseline.base_model's `_ap_at_r` / `_multi_hot_relevance`
verbatim so the sanity number is bit-identical in construction to the P0 table.

Also reports `unique_code_ratio` on the 18-base DB sequences (compression axis)
and, for reference, on the 36-bit DB codes.

Usage:
    python scripts/eval_baseline_dna_space.py \
        --extract_dir result_baseline/260721/cibhash_Flickr25k_clip_E4_dnaeval \
        --dataset Flickr25k --expected_bit_map_at_r 0.8233066262741747 \
        --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from baseline.base_model import (  # noqa: E402
    MAP_AT_R_BY_DATASET, _ap_at_r, _multi_hot_relevance,
)


def _load_split(extract_dir: str, split: str):
    path = os.path.join(extract_dir, f"extract_{split}.npz")
    z = np.load(path, allow_pickle=True)
    bits = z["hash_2bit"].astype(np.uint8)          # [N, 36] in {0,1}
    lbls = z["multi_hot_labels"].astype(np.int64)   # [N, C]
    return bits, lbls


def _two_bit_to_bases(bits: np.ndarray) -> np.ndarray:
    """[N, 2L] {0,1} -> [N, L] base ids in {0,1,2,3}; base = hi*2 + lo."""
    n, w = bits.shape
    assert w % 2 == 0, f"width {w} not even"
    pairs = bits.reshape(n, w // 2, 2)
    return (pairs[:, :, 0].astype(np.int64) * 2 + pairs[:, :, 1].astype(np.int64))


def _map_at_r_from_codes(q_codes: np.ndarray, db_codes: np.ndarray,
                         q_lbl: np.ndarray, db_lbl: np.ndarray,
                         cutoff: int, device: str,
                         q_chunk: int = 256) -> float:
    """CalcTopMap mAP@R with Hamming distance = # differing positions.

    Works for BOTH bit codes ([N,36] {0,1}) and base codes ([N,18] {0..3}):
    Hamming = count of unequal positions.  Distances computed on GPU in query
    chunks; argsort is numpy stable (bit-identical to base_model's ranking).
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    dbc = torch.from_numpy(db_codes).to(dev)                    # [Nd, L]
    Nq = q_codes.shape[0]
    aps = []
    for s in range(0, Nq, q_chunk):
        qc = torch.from_numpy(q_codes[s:s + q_chunk]).to(dev)  # [c, L]
        # Hamming: (q[:,None,:] != db[None,:,:]).sum(-1)
        d = (qc.unsqueeze(1) != dbc.unsqueeze(0)).sum(dim=2)   # [c, Nd] int
        d = d.to(torch.int32).cpu().numpy()
        rel = _multi_hot_relevance(q_lbl[s:s + q_chunk], db_lbl)  # [c, Nd] uint8
        for i in range(d.shape[0]):
            order = np.argsort(d[i], kind="stable")
            aps.append(_ap_at_r(rel[i][order], R=int(cutoff)))
    return float(np.mean(aps)) if aps else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract_dir", required=True)
    ap.add_argument("--dataset", required=True, choices=list(MAP_AT_R_BY_DATASET))
    ap.add_argument("--expected_bit_map_at_r", type=float, default=None,
                    help="P0 table test_mAP_at_R for the sanity gate.")
    ap.add_argument("--sanity_tol", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--q_chunk", type=int, default=256)
    a = ap.parse_args()

    cutoff = MAP_AT_R_BY_DATASET[a.dataset]

    q_bits, q_lbl = _load_split(a.extract_dir, "query")
    db_bits, db_lbl = _load_split(a.extract_dir, "db")
    assert q_bits.shape[1] == 36 and db_bits.shape[1] == 36, \
        f"expected 36-bit codes, got q={q_bits.shape} db={db_bits.shape}"

    # ---- BIT space (sanity) ----
    bit_map = _map_at_r_from_codes(q_bits, db_bits, q_lbl, db_lbl,
                                   cutoff, a.device, a.q_chunk)

    # ---- BASE (DNA) space ----
    q_base = _two_bit_to_bases(q_bits)                          # [Nq, 18]
    db_base = _two_bit_to_bases(db_bits)                        # [Nd, 18]
    assert q_base.shape[1] == 18 and db_base.shape[1] == 18
    base_map = _map_at_r_from_codes(q_base, db_base, q_lbl, db_lbl,
                                    cutoff, a.device, a.q_chunk)

    # ---- unique-code ratios on the DB split ----
    n_db = db_bits.shape[0]
    base_uniq = int(np.unique(db_base, axis=0).shape[0])
    bit_uniq = int(np.unique(db_bits, axis=0).shape[0])
    base_uniq_ratio = base_uniq / max(n_db, 1)
    bit_uniq_ratio = bit_uniq / max(n_db, 1)

    sanity_ok = None
    sanity_delta = None
    if a.expected_bit_map_at_r is not None:
        sanity_delta = bit_map - a.expected_bit_map_at_r
        sanity_ok = bool(abs(sanity_delta) <= a.sanity_tol)

    out = {
        "dataset": a.dataset,
        "extract_dir": os.path.relpath(a.extract_dir, _REPO),
        "map_at_r_cutoff": cutoff,
        "n_query": int(q_bits.shape[0]),
        "n_db": int(n_db),
        "bit_mAP_at_R": bit_map,
        "base_mAP_at_R": base_map,
        "bit_to_base_delta": base_map - bit_map,
        "base_db_unique": base_uniq,
        "base_db_unique_ratio": base_uniq_ratio,
        "bit_db_unique": bit_uniq,
        "bit_db_unique_ratio": bit_uniq_ratio,
        "expected_bit_mAP_at_R": a.expected_bit_map_at_r,
        "sanity_delta": sanity_delta,
        "sanity_ok": sanity_ok,
        "sanity_tol": a.sanity_tol,
    }
    print(json.dumps(out, indent=2))
    with open(os.path.join(a.extract_dir, "dna_space_eval.json"), "w") as f:
        json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
