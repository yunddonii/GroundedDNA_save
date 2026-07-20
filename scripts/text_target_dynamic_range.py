#!/usr/bin/env python
"""Does the TEXT teacher have usable WITHIN-SLOT dynamic range?

`z_geometry_analysis.py` located MSCOCO's degeneracy upstream of the codebook:
the routed features z are already ungraded (rho_z proto 0.245 vs 0.619/0.506).
The natural prescription was a within-slot relational loss -- pull z_i, z_j
together in proportion to how close their slot-m captions are.

That prescription is only sound if the TARGET has dynamic range. CLIP text
embeddings are famously anisotropic: if s_i^m and s_j^m sit at cosine ~0.9 for
every pair, the relational target is nearly constant and cannot shape anything.

Measured here, per slot, over image PAIRS within the same slot:
    cos mean/sd/p5/p95   the raw dynamic range of the target
    rho_text_label       Spearman(1 - cos(s_i, s_j), ||y_i - y_j||)
                         -- does text distance actually track semantic distance?
                         A wide range is useless if it is uncorrelated with meaning.
    eff_rank             participation ratio of the slot's text embeddings

Reported RAW and after the model's actual `partial_whiten` transform
(W = U diag((S+eps)^-gamma) U^T, gamma from the run's args), because that is the
representation the model really consumes.

Only train rows have captions on MSCOCO/NUS-WIDE, so the analysis is restricted
to rows with has_text.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy.stats import spearmanr

N_SLOTS = 6


def _unit(x, axis=-1):
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), 1e-12)


def whiten_matrix(npz_path: str, gamma: float, eps: float = 1e-5):
    b = np.load(npz_path)
    mu = np.asarray(b["mu"], dtype=np.float32)
    U = np.asarray(b["U"], dtype=np.float32)
    S = np.asarray(b["S"], dtype=np.float32)
    W = U @ np.diag((S + eps) ** (-gamma)) @ U.T
    return mu, W.astype(np.float32)


def eff_rank(X):
    s = np.linalg.svd(X - X.mean(0), compute_uv=False)
    p = s ** 2 / max((s ** 2).sum(), 1e-12)
    return float(1.0 / (p ** 2).sum())


def analyse(cache_dir, labels, row_ids, whiten_npz, gamma, n_pairs, rng):
    tp = np.load(os.path.join(cache_dir, "text_part.f16.npy"), mmap_mode="r")
    X = np.asarray(tp[row_ids], dtype=np.float32)              # [n, 6, D]
    y = labels

    n = len(X)
    ii = rng.integers(0, n, n_pairs); jj = rng.integers(0, n, n_pairs)
    keep = ii != jj; ii, jj = ii[keep], jj[keep]
    d_lab = np.linalg.norm(y[ii].astype(np.float32) - y[jj].astype(np.float32), axis=1)

    variants = {"raw": X}
    if whiten_npz and os.path.exists(whiten_npz):
        mu, W = whiten_matrix(whiten_npz, gamma)
        variants["whitened"] = ((X.reshape(-1, X.shape[-1]) - mu) @ W).reshape(X.shape)

    out = {}
    for vname, V in variants.items():
        per = []
        for m in range(N_SLOTS):
            Vn = _unit(V[:, m])
            cos = (Vn[ii] * Vn[jj]).sum(1)
            per.append({
                "slot": m,
                "cos_mean": float(cos.mean()), "cos_sd": float(cos.std()),
                "cos_p5": float(np.percentile(cos, 5)),
                "cos_p95": float(np.percentile(cos, 95)),
                "rho_text_label": float(spearmanr(1.0 - cos, d_lab).statistic),
                "eff_rank": eff_rank(V[:, m]),
            })
        out[vname] = {
            "per_slot": per,
            "cos_mean": float(np.mean([p["cos_mean"] for p in per])),
            "cos_sd": float(np.mean([p["cos_sd"] for p in per])),
            "rho_text_label": float(np.mean([p["rho_text_label"] for p in per])),
            "eff_rank": float(np.mean([p["eff_rank"] for p in per])),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", nargs="+", required=True,
                    help="NAME:CACHE_DIR:RESULT_DIR")
    ap.add_argument("--gamma", type=float, default=0.25)
    ap.add_argument("--n_pairs", type=int, default=200000)
    ap.add_argument("--max_rows", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    res = []
    hdr = (f"{'dataset':12s}{'variant':10s}{'cos mean':>10s}{'cos sd':>9s}"
           f"{'p5':>8s}{'p95':>8s}{'rho_txt~lab':>13s}{'eff_rank':>10s}")
    print(hdr)
    for spec in args.specs:
        name, cache, run = spec.split(":")
        # rows with captions, paired with labels from the run's train extraction
        tr_npz = os.path.join(run, "extract_train.npz")
        ids_json = os.path.join(cache, "image_ids.json")
        has_text_p = os.path.join(cache, "has_text.bool.npy")

        img_ids = json.load(open(ids_json))
        pos = {os.path.basename(p): i for i, p in enumerate(img_ids)}
        if os.path.exists(tr_npz):
            t = np.load(tr_npz, allow_pickle=True)
            names = [os.path.basename(str(p)) for p in t["image_paths"]]
            lab = np.asarray(t["multi_hot_labels"])
        else:
            db = np.load(os.path.join(run, "extract_db.npz"), allow_pickle=True)
            man = {"Flickr25k": "dataset/Flickr25k/setting1/train.txt",
                   "NUS-WIDE": "dataset/NUSWIDE/setting1/train.txt"}[name]
            want = {os.path.basename(l.split()[0]) for l in open(man) if l.strip()}
            allb = [os.path.basename(str(p)) for p in db["image_paths"]]
            k = [i for i, b in enumerate(allb) if b in want]
            names = [allb[i] for i in k]
            lab = np.asarray(db["multi_hot_labels"])[k]

        rows = np.array([pos[b] for b in names if b in pos])
        keepmask = np.array([b in pos for b in names])
        lab = lab[keepmask]
        if os.path.exists(has_text_p):
            ht = np.load(has_text_p)
            ok = ht[rows]
            rows, lab = rows[ok], lab[ok]
        if len(rows) > args.max_rows:
            sel = np.sort(rng.choice(len(rows), args.max_rows, replace=False))
            rows, lab = rows[sel], lab[sel]

        wn = os.path.join(cache, "text_whiten_trainOnly.npz")
        r = analyse(cache, lab, rows, wn, args.gamma, args.n_pairs, rng)
        r["dataset"] = name; r["n_rows"] = int(len(rows))
        res.append(r)
        for vname in ["raw", "whitened"]:
            if vname not in r:
                continue
            v = r[vname]
            ps = np.mean([p["cos_p5"] for p in v["per_slot"]])
            p95 = np.mean([p["cos_p95"] for p in v["per_slot"]])
            print(f"{name if vname=='raw' else '':12s}{vname:10s}"
                  f"{v['cos_mean']:>10.3f}{v['cos_sd']:>9.3f}{ps:>8.3f}{p95:>8.3f}"
                  f"{v['rho_text_label']:>13.3f}{v['eff_rank']:>10.1f}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
