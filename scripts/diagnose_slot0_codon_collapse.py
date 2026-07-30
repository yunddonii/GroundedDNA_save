#!/usr/bin/env python
"""Exhaustive diagnosis of the slot-0 (global) codon collapse.

Standing facts (2026-07-28 / 07-29):
  * MSCOCO A-champion distinct codons per slot: [21, 61, 60, 60, 61, 58] / 64.
  * slot0 codeword usage is MAXIMAL (128/128 used) -- the collapse is NOT in
    the quantizer, it is in the deterministic codeword -> codon map.
  * slot0 chunk std 0.3946 (vs 0.858-0.924) and codon-head ||W|| 16.41 (vs
    1.58-2.17). Normalising the chunk (chunk-LayerNorm) fixed ||W|| and the
    ratio but made retrieval AND diversity worse.
  * Making slot0 symmetric (routing + all four global losses restored) drove it
    to 1/64 codons -- so "missing text supervision" is NOT the cause.

Why an exact enumeration is possible: with `codon_input_source=quantized` and
`codon_residual_gamma=0`, the codon is a PURE DETERMINISTIC FUNCTION of the
codeword index:

    e = quantizer.codebooks[m, k]            in R^768
    chunks = e.view(3, 256)                  3 codon positions
    logits = chunks @ fc.weight.T + fc.bias  shared Linear(256, 4) per slot
    codon  = 16*argmax(l0) + 4*argmax(l1) + argmax(l2)

So the entire map can be reconstructed for all K=128 codewords without running
the model. This script does that and then decomposes WHY the map is many-to-one
for slot 0, testing five hypotheses:

  H1 bias dominance   -- if spread(bias) >> std(W c) the argmax is bias-locked
                         and the codeword contributes almost nothing.
  H2 direction starve -- the chunk may vary a lot (high effective rank) but
                         almost none of that variance may survive projection
                         onto the 4 rows of W. Measures the surviving fraction.
  H3 shared-fc clash  -- one Linear serves all 3 positions of a slot; if the 3
                         chunks of a codeword occupy the same region, all three
                         positions emit the same base.
  H4 base_balance blind spot -- `loss_base_balance` is KL(uniform || mean_b u)
                         over SOFT probabilities. A head can be argmax-collapsed
                         while its batch-mean soft probability stays uniform, in
                         which case the regulariser sees nothing to fix.
  H5 saturation       -- logit scale so large that softmax is one-hot, so
                         `loss_quant` / `loss_entropy` have no gradient left.

Usage:
  python scripts/diagnose_slot0_codon_collapse.py \
      --dir result/...mscoco_A_v5b_P0refit_e39... --out docs/slot0_collapse_mscoco.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F


def eff_rank(x: np.ndarray) -> float:
    """exp(entropy of the normalised singular-value spectrum)."""
    if x.shape[0] < 2:
        return 1.0
    s = np.linalg.svd(x - x.mean(0, keepdims=True), compute_uv=False)
    p = s / max(s.sum(), 1e-12)
    p = p[p > 1e-12]
    return float(np.exp(-(p * np.log(p)).sum()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num_codons", type=int, default=3)
    a = ap.parse_args()

    sd = torch.load(os.path.join(a.dir, "model_state_dict.pth"),
                    map_location="cpu", weights_only=False)
    cb = sd["quantizer.codebooks"].float()                    # [M, K, D]
    M, K, D = cb.shape
    L = a.num_codons
    chunk = D // L

    z = np.load(os.path.join(a.dir, "extract_db.npz"))
    used = np.asarray(z["codebook_indices"])                  # [N, M]
    base = np.asarray(z["base_indices"])                      # [N, M*L]
    N = len(used)

    rows, per_slot = [], []
    print(f"codebooks {tuple(cb.shape)}  chunk={chunk}  DB rows={N}\n")

    for m in range(M):
        W = sd[f"codon_heads.{m}.fc.weight"].float()          # [4, chunk]
        b = sd[f"codon_heads.{m}.fc.bias"].float()            # [4]
        E = cb[m]                                             # [K, D]
        C = E.view(K, L, chunk)                               # [K, L, chunk]

        # ---- exact enumeration of the codeword -> codon map -----------------
        logits = torch.einsum("klc,bc->klb", C, W) + b        # [K, L, 4]
        arg = logits.argmax(-1)                               # [K, L]
        codon_of_codeword = (arg[:, 0] * 16 + arg[:, 1] * 4 + arg[:, 2]).numpy()

        # observed usage of each codeword on the DB
        cnt = np.bincount(used[:, m], minlength=K).astype(np.float64)
        freq = cnt / max(cnt.sum(), 1.0)
        used_k = int((cnt > 0).sum())
        # observed codons (from the extraction) must match the enumeration
        obs = (base[:, L * m] * 16 + base[:, L * m + 1] * 4 + base[:, L * m + 2])
        pred = codon_of_codeword[used[:, m]]
        agree = float((obs == pred).mean())

        codon_hist = np.bincount(codon_of_codeword[cnt > 0].astype(int),
                                 weights=cnt[cnt > 0], minlength=64)
        codon_hist = codon_hist / max(codon_hist.sum(), 1.0)
        distinct = int((codon_hist > 0).sum())
        top_share = float(codon_hist.max())

        # ---- H1 bias dominance ---------------------------------------------
        Wc = torch.einsum("klc,bc->klb", C, W)                # [K, L, 4] pre-bias
        wc_std = float(Wc.std())                              # spread from data
        bias_spread = float(b.max() - b.min())
        h1 = bias_spread / max(wc_std, 1e-12)

        # ---- H2 how much chunk variance survives W --------------------------
        ch = C.reshape(K * L, chunk).numpy()
        ch_std = float(ch.std())
        surviving = float(Wc.reshape(-1, 4).std(0).mean()) / max(
            float(np.linalg.norm(W.numpy(), axis=1).mean()) * ch_std, 1e-12)

        # ---- H3 do the 3 positions of a codeword emit the same base? -------
        same3 = float((arg[:, 0] == arg[:, 1]).float().mul(
            (arg[:, 1] == arg[:, 2]).float()).mean())
        # per-position argmax entropy (usage-weighted, HARD)
        hard_H, soft_H, soft_mean = [], [], []
        for l in range(L):
            p_hard = np.bincount(arg[:, l].numpy(), weights=cnt, minlength=4)
            p_hard = p_hard / max(p_hard.sum(), 1e-12)
            hard_H.append(float(-(p_hard[p_hard > 0]
                                  * np.log(p_hard[p_hard > 0])).sum() / np.log(4)))
            # ---- H4 what base_balance actually sees: SOFT batch mean --------
            sp = F.softmax(logits[:, l, :], dim=-1).numpy()   # [K, 4]
            pbar = (sp * freq[:, None]).sum(0)
            pbar = pbar / max(pbar.sum(), 1e-12)
            soft_H.append(float(-(pbar[pbar > 0]
                                  * np.log(pbar[pbar > 0])).sum() / np.log(4)))
            soft_mean.append([round(float(v), 4) for v in pbar])

        # ---- H5 saturation --------------------------------------------------
        sp_all = F.softmax(logits.reshape(-1, 4), dim=-1)
        pmax = float(sp_all.max(-1).values.mean())
        logit_range = float((logits.max(-1).values - logits.min(-1).values).mean())

        r = {
            "slot": m, "used_codewords": used_k, "K": K,
            "distinct_codons": distinct, "collapse_ratio": round(used_k / max(distinct, 1), 2),
            "top_codon_share": round(top_share, 4),
            "map_agreement_with_extraction": round(agree, 6),
            "W_norm": round(float(W.norm()), 3),
            "W_row_norm_mean": round(float(W.norm(dim=1).mean()), 3),
            "bias": [round(float(v), 4) for v in b],
            "bias_spread": round(bias_spread, 4),
            "Wc_std": round(wc_std, 4),
            "H1_bias_over_signal": round(h1, 3),
            "chunk_std": round(ch_std, 4),
            "chunk_eff_rank": round(eff_rank(ch), 1),
            "H2_surviving_var_frac": round(surviving, 5),
            "H3_all3_same_base": round(same3, 4),
            "hard_argmax_entropy_per_pos": [round(v, 4) for v in hard_H],
            "H4_soft_mean_entropy_per_pos": [round(v, 4) for v in soft_H],
            "H4_soft_mean_probs_pos0": soft_mean[0],
            "H5_softmax_pmax": round(pmax, 4),
            "H5_logit_range": round(logit_range, 3),
        }
        rows.append(r)
        per_slot.append(codon_hist)
        print(f"slot{m}: codons {distinct:2d}/64  ratio {r['collapse_ratio']:5.2f}:1  "
              f"top {top_share*100:5.1f}%  |W| {r['W_norm']:6.2f}  "
              f"chunkStd {ch_std:.3f}  bias/signal {h1:6.2f}  "
              f"survVar {surviving:.4f}  all3same {same3*100:4.1f}%  "
              f"hardH {np.mean(hard_H):.3f}  softH {np.mean(soft_H):.3f}  "
              f"pmax {pmax:.3f}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"dir": a.dir, "N_db": int(N), "slots": rows}, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")
    print("[read] H1 bias/signal >> 1 -> argmax is bias-locked, codeword ignored.")
    print("[read] H2 surviving_var_frac -> fraction of chunk variance reaching the logits.")
    print("[read] H4 hardH low but softH high -> base_balance cannot see the collapse.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
