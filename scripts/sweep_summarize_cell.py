#!/usr/bin/env python
"""Emit one compact comparison row for a finished sweep cell.

Collects, in one place, the four axes the 2026-07-31 sweep optimises jointly:

    retrieval        mAP@R (bio-projected)          from cell_result.json
    diversity        DNA-unique on the DB           from cell_result.json
    interpretability held-out codon decoding        from the decoding JSON
    structure        per-slot distinct codons, slot0 gap and multi-information,
                     recomputed here by exact enumeration of the codeword->codon
                     map (valid while codon_input_source=quantized and
                     codon_residual_gamma=0; the script detects and reports when
                     that assumption does not hold)

Handles both codon-head layouts (`fc` shared across the L positions and
`fc_pos.{l}` per-position) and both chunk partitions (contiguous / interleaved),
reading the run's own `args.txt` so a row is never silently computed under the
wrong convention.

Usage:
  python scripts/sweep_summarize_cell.py --dir <result_dir> \
      [--decoding_json docs/....json] --out docs/sweep_rows/<tag>.json
"""
from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np
import torch


def _arg(txt: str, key: str, default=None):
    m = re.search(rf"^{re.escape(key)}-+(\S+)\s*$", txt, re.M)
    return m.group(1) if m else default


def _H(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--decoding_json", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    cell = json.load(open(os.path.join(a.dir, "cell_result.json")))
    txt = open(os.path.join(a.dir, "args.txt")).read()

    L = int(_arg(txt, "num_codons_per_codebook", "3"))
    interleave = _arg(txt, "codon_chunk_interleave", "False") == "True"
    src = _arg(txt, "codon_input_source", "quantized")
    gamma = float(_arg(txt, "codon_residual_gamma", "0.0"))
    enumerable = (src == "quantized") and (gamma == 0.0)

    row = {
        "tag": a.tag or os.path.basename(a.dir),
        "dir": a.dir,
        "dataset": cell.get("dataset"),
        "K": cell.get("K"),
        "mAP_at_R_bioproj": cell.get("mAP_at_R_bioproj"),
        "full_mAP_bioproj": cell.get("full_mAP_bioproj"),
        "full_mAP_pre_projection": cell.get("full_mAP_pre_projection"),
        "DNA_unique_DB": cell.get("DNA_unique_DB"),
        "lambda_codon_joint": _arg(txt, "lambda_codon_joint", "0.0"),
        "codon_joint_slots": _arg(txt, "codon_joint_slots", ""),
        "codon_input_source": src,
        "use_gumbel_softmax": _arg(txt, "use_gumbel_softmax", "True"),
        "codon_chunk_interleave": str(interleave),
        "codon_map_enumerable": enumerable,
    }

    if enumerable:
        sd = torch.load(os.path.join(a.dir, "model_state_dict.pth"),
                        map_location="cpu", weights_only=False)
        cb = sd["quantizer.codebooks"].float()
        M, K, D = cb.shape
        ch = D // L
        codons, gaps, mis = [], [], []
        for m in range(M):
            E = cb[m]
            C = (E.view(K, ch, L).transpose(1, 2) if interleave
                 else E.view(K, L, ch))
            kf = f"codon_heads.{m}.fc.weight"
            if kf in sd:
                W = sd[kf].float()
                b = sd[f"codon_heads.{m}.fc.bias"].float()
                lg = torch.einsum("klc,bc->klb", C, W) + b
            else:
                lg = torch.stack([
                    C[:, l, :] @ sd[f"codon_heads.{m}.fc_pos.{l}.weight"].float().T
                    + sd[f"codon_heads.{m}.fc_pos.{l}.bias"].float()
                    for l in range(L)], dim=1)
            arg = lg.argmax(-1).numpy()
            cid = np.zeros(K, dtype=np.int64)
            for l in range(L):
                cid = cid * 4 + arg[:, l]
            codons.append(int(len(np.unique(cid))))
            mg = [np.bincount(arg[:, l], minlength=4) / K for l in range(L)]
            pc = mg[0]
            for l in range(1, L):
                pc = np.einsum("i,j->ij", pc, mg[l]).ravel()
            gaps.append(round(codons[-1] - float((1 - (1 - pc) ** K).sum()), 2))
            mis.append(round(sum(_H(x) for x in mg)
                             - _H(np.bincount(cid, minlength=4 ** L) / K), 4))
        row.update(slot_codons=codons, slot_gaps=gaps, slot_multi_information=mis,
                   slot0_codons=codons[0], slot0_gap=gaps[0], slot0_MI=mis[0])

    if a.decoding_json and os.path.exists(a.decoding_json):
        dec = json.load(open(a.decoding_json))
        dec = dec.get("results", dec)          # decoding JSON nests under "results"
        def _sm(k):
            v = dec.get(k)
            if isinstance(v, dict):
                return (v.get("slot_mean") or {}).get("concept_mAP", v.get("concept_mAP"))
            return None
        base = {k: _sm(k) for k in dec if k.endswith("_chunk")}
        base = {k: v for k, v in base.items() if v is not None}
        row["decode_ours_codon"] = _sm("ours_codon")
        row["decode_ours_codeword"] = _sm("ours_codeword")
        row["decode_best_control"] = (max(base, key=base.get) if base else None)
        row["decode_best_control_mAP"] = (max(base.values()) if base else None)
        if row["decode_ours_codon"] is not None and base:
            row["decode_margin"] = round(row["decode_ours_codon"] - max(base.values()), 4)
        maj = dec.get("majority")
        row["decode_majority"] = (maj or {}).get("concept_mAP") if isinstance(maj, dict) else None

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(row, open(a.out, "w"), indent=2)
    print(json.dumps({k: row[k] for k in (
        "tag", "dataset", "mAP_at_R_bioproj", "DNA_unique_DB",
        "decode_ours_codon", "decode_margin", "slot0_codons", "slot0_gap")
        if k in row}, indent=None))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
