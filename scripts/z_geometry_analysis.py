#!/usr/bin/env python
"""Where does the codebook's semantic metric structure come from -- or fail?

`codebook_semantic_alignment.py` found the CODEBOOK geometry degenerate on
MSCOCO (rho 0.13 vs 0.58/0.49) and refuted four codebook-side causes. This
script tests the one remaining, upstream mechanism: the geometry of the routed
pre-quantisation features z = quant_input (captured by extract_z_prequant.py).

The codebook is (EMA-)placed to minimise quantisation error of z, so it can only
be as graded as z is. Two outcomes, opposite prescriptions:

    rho_z LOW  (like rho_codeword) -> z itself is ungraded; the encoder/router
                                      is the problem. Codebook-side losses cannot
                                      manufacture structure that is not in z.
    rho_z HIGH (>> rho_codeword)   -> z is graded but quantisation destroys it;
                                      a codebook-side loss (Gram distillation,
                                      text-anchored codewords) is the right fix.

Measures per slot, on z:
    rho_z        Spearman(1 - cos(z_a, z_b), ||P_a - P_b||) where a,b are
                 individual images and P is their label vector -- the SAME
                 semantic-alignment statistic as the codebook version, but on the
                 continuous features. To match the codebook version (which is
                 over codeword prototypes), we also report:
    rho_z_proto  the codebook statistic recomputed with each codeword replaced
                 by the MEAN of its assigned z's (the empirical prototype),
                 instead of the learned codebook vector. This isolates
                 "quantisation geometry" (learned codebook) from "assignment
                 geometry" (where the z's actually sit).
    eff_rank_z   participation ratio of z's singular values (per slot)
    within/all   mean cosine within a codeword's z's vs between -- how tight the
                 clusters are relative to the whole slot.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from scipy.stats import spearmanr

N_SLOTS = 6


def _unit(x, axis=-1):
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), 1e-12)


def eff_rank(X):
    s = np.linalg.svd(X - X.mean(0), compute_uv=False)
    p = s ** 2 / max((s ** 2).sum(), 1e-12)
    return float(1.0 / (p ** 2).sum())


def pair_rho(dist_code, P, iu):
    d_lab = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)[iu]
    return float(spearmanr(dist_code, d_lab).statistic)


def analyse(run_dir, min_support, n_img_pairs, rng):
    zc = np.load(os.path.join(run_dir, "extract_z_db.npz"), allow_pickle=True)
    z = zc["z"].astype(np.float32)                     # [N, 6, D]
    cw = zc["codebook_indices"]
    lab_key = "multi_hot_labels" if "multi_hot_labels" in zc.files else "labels"
    lab = np.asarray(zc[lab_key])
    if lab.ndim == 1:
        lab = np.eye(int(lab.max()) + 1, dtype=np.int64)[lab]

    sd = torch.load(os.path.join(run_dir, "model_state_dict.pth"), map_location="cpu")
    cbs = sd["quantizer.codebooks"].float().numpy()

    per = []
    # image-pair sample shared across slots
    N = len(z)
    ii = rng.integers(0, N, n_img_pairs); jj = rng.integers(0, N, n_img_pairs)
    keep = ii != jj; ii, jj = ii[keep], jj[keep]
    d_lab_img = np.linalg.norm(lab[ii] - lab[jj], axis=1)

    for m in range(N_SLOTS):
        zm = z[:, m]
        u, c = np.unique(cw[:, m], return_counts=True)
        active = u[c >= min_support]

        # rho on individual-image z cosines (continuous)
        zn = _unit(zm)
        d_cos_img = 1.0 - (zn[ii] * zn[jj]).sum(1)
        rho_z = float(spearmanr(d_cos_img, d_lab_img).statistic)

        # prototype-level statistics over active codewords
        Pz = np.stack([lab[cw[:, m] == a].mean(0) for a in active])
        proto = _unit(np.stack([zm[cw[:, m] == a].mean(0) for a in active]))
        cbk = _unit(cbs[m][active])
        iu = np.triu_indices(len(active), 1)
        rho_z_proto = pair_rho((1.0 - proto @ proto.T)[iu], Pz, iu)
        rho_codebook = pair_rho((1.0 - cbk @ cbk.T)[iu], Pz, iu)

        # cluster tightness: proto-to-codebook agreement
        proto_cbk_cos = float((proto * cbk).sum(1).mean())

        per.append({
            "slot": m, "n_active": int(len(active)),
            "rho_z_image": rho_z,
            "rho_z_proto": rho_z_proto,
            "rho_codebook": rho_codebook,
            "eff_rank_z": eff_rank(zm),
            "proto_vs_codebook_cos": proto_cbk_cos,
        })

    def mean(k):
        return float(np.mean([p[k] for p in per]))
    return {
        "dir": os.path.basename(run_dir.rstrip("/")),
        "n_used": int(N),
        "mean_rho_z_image": mean("rho_z_image"),
        "mean_rho_z_proto": mean("rho_z_proto"),
        "mean_rho_codebook": mean("rho_codebook"),
        "mean_eff_rank_z": mean("eff_rank_z"),
        "mean_proto_vs_codebook_cos": mean("proto_vs_codebook_cos"),
        "per_slot": per,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--min_support", type=int, default=20)
    ap.add_argument("--n_img_pairs", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    names = args.labels or [os.path.basename(d.rstrip("/"))[:40] for d in args.dirs]
    res = []
    print(f"{'dataset':12s}{'rho_z(img)':>12s}{'rho_z(proto)':>14s}"
          f"{'rho_codebook':>14s}{'eff_rank_z':>12s}{'proto~cbk':>11s}")
    for d, nm in zip(args.dirs, names):
        try:
            r = analyse(d, args.min_support, args.n_img_pairs, rng)
        except FileNotFoundError as e:
            print(f"{nm:12s}  MISSING {os.path.basename(str(e))}")
            continue
        r["label"] = nm; res.append(r)
        print(f"{nm:12s}{r['mean_rho_z_image']:>12.3f}{r['mean_rho_z_proto']:>14.3f}"
              f"{r['mean_rho_codebook']:>14.3f}{r['mean_eff_rank_z']:>12.1f}"
              f"{r['mean_proto_vs_codebook_cos']:>11.3f}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
