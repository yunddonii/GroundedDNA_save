"""Is the column effect a property of the captions rather than of our model?

The slot role-alignment matrix is dominated by main effects: some caption slots
(global, scene) are organized well by EVERY code slot, others (secondary_object)
by none. Two readings:

  H_model  our model encodes global/scene information everywhere, so its slots
           all track those captions;
  H_data   global/scene caption embeddings are intrinsically easier to organize
           (low effective dimensionality / high baseline coherence), so ANY
           partition scores high on them.

H_data is testable without training anything: build the same lift matrix using
partitions that know nothing about our semantic slots -- the flat baselines'
arbitrary 6-bit chunks. If the column PROFILE (which captions score high) is the
same for CIBHash/CIMON as for us, the column effect cannot be attributed to our
model. We additionally measure each caption slot's intrinsic concentration and
check whether it predicts the column effect.

This matters for what the paper can claim: if H_data holds, image-level captions
cannot validate slot ROLE assignment at all -- the metric would reward any model
for organizing the easy caption dimensions -- and per-slot targets (e.g. CUB
attributes) become the only route.

Usage:
    python scripts/caption_column_effect.py \
        --ours_dir result/260717+flickr25k_..._P0refit_e4+... \
        --baseline_dirs result_baseline/260527/{cibhash,cimon,mls3rduh}_flickr25k_clip_unsup60 \
        --baseline_names cibhash cimon mls3rduh \
        --cache_dir cache/flickr25k_clip_v4plus_qwen3_tokens \
        --train_manifest dataset/Flickr25k/setting1/train.txt \
        --out docs/caption_column_effect_flickr25k.json
"""
import argparse
import json
import os

import numpy as np

SLOT_NAMES = ["global", "primary_object", "secondary_object",
              "activity_relation", "color_texture", "scene_type"]


def _basenames(p):
    return np.array([os.path.basename(str(x)) for x in p])


def mean_pairwise_cos(X):
    n = len(X)
    if n < 2:
        return np.nan
    G = X @ X.T
    return float((G.sum() - np.trace(G)) / (n * (n - 1)))


def column_profile(codes, text, min_support, n_shuffle, rng):
    """Mean lift of each caption slot k, averaged over all code slots m."""
    M, K = codes.shape[1], text.shape[1]
    lift = np.full((M, K), np.nan)
    for m in range(M):
        u, inv = np.unique(codes[:, m], return_inverse=True)
        groups = [np.where(inv == i)[0] for i in range(len(u))]
        groups = [g for g in groups if len(g) >= min_support]
        if not groups:
            continue
        sizes = np.array([len(g) for g in groups])
        for k in range(K):
            Xk = text[:, k, :]
            obs = np.mean([mean_pairwise_cos(Xk[g]) for g in groups])
            base = []
            for _ in range(n_shuffle):
                perm = rng.permutation(len(Xk))
                off, acc = 0, []
                for s in sizes:
                    acc.append(mean_pairwise_cos(Xk[perm[off:off + s]]))
                    off += s
                base.append(np.mean(acc))
            lift[m, k] = obs - np.mean(base)
    return lift


def intrinsic_stats(text):
    """Per caption slot: how concentrated is the embedding cloud itself?"""
    K = text.shape[1]
    out = []
    for k in range(K):
        X = text[:, k, :]
        # effective rank of the covariance (participation ratio of eigenvalues)
        C = np.cov(X.T)
        ev = np.linalg.eigvalsh(C)
        ev = np.clip(ev, 0, None)
        eff_rank = float((ev.sum() ** 2) / (np.square(ev).sum() + 1e-30))
        sub = X[np.random.default_rng(0).choice(len(X), min(3000, len(X)), replace=False)]
        out.append({
            "slot": SLOT_NAMES[k],
            "eff_rank": eff_rank,
            "mean_pairwise_cos": mean_pairwise_cos(sub),
            "top1_var_share": float(ev.max() / (ev.sum() + 1e-30)),
        })
    return out


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra @ rb) / (np.linalg.norm(ra) * np.linalg.norm(rb) + 1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours_dir", required=True)
    ap.add_argument("--baseline_dirs", nargs="*", default=[])
    ap.add_argument("--baseline_names", nargs="*", default=[])
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--train_manifest", default=None)
    ap.add_argument("--whiten_npz", default=None)
    ap.add_argument("--whiten_gamma", type=float, default=0.25)
    ap.add_argument("--min_support", type=int, default=10)
    ap.add_argument("--n_shuffle", type=int, default=5)
    ap.add_argument("--max_n", type=int, default=18000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    z = np.load(os.path.join(args.ours_dir, "extract_db.npz"), allow_pickle=True)
    bn = _basenames(z["image_paths"])
    keep = np.ones(len(bn), bool)
    if args.train_manifest:
        tr = {os.path.basename(l.split()[0]) for l in open(args.train_manifest) if l.strip()}
        keep &= ~np.isin(bn, list(tr))
    idx = np.where(keep)[0]
    if len(idx) > args.max_n:
        idx = np.sort(rng.choice(idx, args.max_n, replace=False))
    names_kept = bn[idx]
    print(f"[col] held-out rows: {len(idx)}")

    # captions for exactly these rows
    ids = json.load(open(os.path.join(args.cache_dir, "image_ids.json")))
    row_of = {os.path.basename(str(s)): i for i, s in enumerate(ids)}
    rows = np.array([row_of[b] for b in names_kept])
    T = np.asarray(np.load(os.path.join(args.cache_dir, "text_part.f16.npy"),
                           mmap_mode="r")[rows], dtype=np.float64)
    has = np.asarray(np.load(os.path.join(args.cache_dir, "has_text.bool.npy"),
                             mmap_mode="r")[rows]).astype(bool)
    if args.whiten_npz:
        w = np.load(args.whiten_npz)
        mu, U, S = w["mu"].astype(np.float64), w["U"].astype(np.float64), w["S"].astype(np.float64)
        W = U @ np.diag((S + 1e-5) ** (-args.whiten_gamma)) @ U.T
        T = ((T.reshape(-1, T.shape[-1]) - mu) @ W).reshape(-1, 6, W.shape[0])
    T = T - T.mean(0, keepdims=True)
    T /= (np.linalg.norm(T, axis=-1, keepdims=True) + 1e-12)

    def codon_of(b):
        return b[:, 0::3] * 16 + b[:, 1::3] * 4 + b[:, 2::3]

    partitions = {"ours": codon_of(z["base_indices"][idx])[has]}
    for bdir, bname in zip(args.baseline_dirs, args.baseline_names):
        zb = np.load(os.path.join(bdir, "extract_db.npz"), allow_pickle=True)
        pos = {b: i for i, b in enumerate(_basenames(zb["image_paths"]))}
        sel = np.array([pos[b] for b in names_kept])
        partitions[bname] = codon_of(zb["base_indices"][sel])[has]
    Tf = T[has]
    print(f"[col] partitions: {list(partitions)} | N={len(Tf)}\n")

    profiles = {}
    for name, codes in partitions.items():
        L = column_profile(codes, Tf, args.min_support, args.n_shuffle, rng)
        col = np.nanmean(L, axis=0)
        profiles[name] = col
        rank = np.argsort(np.argsort(-col)) + 1
        print(f"  {name:10s} " + "  ".join(f"{SLOT_NAMES[k][:8]}={col[k]:.3f}(#{rank[k]})"
                                           for k in range(6)))

    print("\n=== 열 프로파일 상관 (ours vs 각 flat baseline) ===")
    corrs = {}
    for name in profiles:
        if name == "ours":
            continue
        r = spearman(profiles["ours"], profiles[name])
        p = float(np.corrcoef(profiles["ours"], profiles[name])[0, 1])
        corrs[name] = {"spearman": r, "pearson": p}
        print(f"  ours vs {name:10s} spearman={r:+.3f}  pearson={p:+.3f}")

    print("\n=== caption slot 내재적 성질 ===")
    st = intrinsic_stats(Tf)
    for s in st:
        print(f"  {s['slot']:18s} eff_rank={s['eff_rank']:6.1f}  "
              f"mean_cos={s['mean_pairwise_cos']:+.4f}  top1_var={s['top1_var_share']:.3f}")
    er = np.array([s["eff_rank"] for s in st])
    print(f"\n  열효과(ours) vs eff_rank  spearman = "
          f"{spearman(profiles['ours'], -er):+.3f}   (음의 eff_rank 기준: 저차원일수록 조직 쉬움)")

    json.dump({
        "n": int(len(Tf)),
        "column_profiles": {k: v.tolist() for k, v in profiles.items()},
        "profile_correlation_vs_ours": corrs,
        "caption_intrinsic": st,
        "slot_names": SLOT_NAMES,
        "config": vars(args),
    }, open(args.out, "w"), indent=2)
    print(f"\n[col] wrote {args.out}")


if __name__ == "__main__":
    main()
