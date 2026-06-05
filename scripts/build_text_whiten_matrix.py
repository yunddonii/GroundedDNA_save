"""v113 Method 3: build partial-whitening matrix from cached CLIP/SigLIP2 text.

For a cache directory holding `text_part.f16.npy [N, M=6, D]`:
  1. Stack all rows into T_all [N*M, D]  (optionally drop has_text=False rows).
  2. mu     = T_all.mean(dim=0)                 [D]
  3. Sigma  = (T_all - mu)^T (T_all - mu) / (N-1)
  4. U, S, _ = svd(Sigma)                       # eigendecomp of Sigma
  5. Save  {mu, U, S} -> <out>.npz
Training-time builds W_gamma = U diag((S + eps)^-gamma) U^T as the actual
whitening matrix, so gamma can be swept without re-running this script.

Usage:
    python scripts/build_text_whiten_matrix.py \
        --cache_dir cache/flickr25k_clip_v4plus \
        --out       cache/flickr25k_clip_v4plus/text_whiten.npz \
        [--include_no_text]
"""
import argparse
import json
import os
import time

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True,
                    help="Directory containing text_part.f16.npy and "
                         "has_text.bool.npy.")
    ap.add_argument("--out", required=True,
                    help="Output .npz path. Will hold keys {mu, U, S}.")
    ap.add_argument("--include_no_text", action="store_true",
                    help="If set, do NOT filter by has_text mask. Default "
                         "is to include only rows with valid captions.")
    ap.add_argument("--residualize_first", action="store_true",
                    help="If set, subtract T_global (slot 0) from each local "
                         "slot (1..5) BEFORE computing the covariance. The "
                         "resulting whitening matrix is tailored to the "
                         "residualized local population (N * 5 vectors) and "
                         "should pair with --text_embed_transform "
                         "global_residual_whiten at training time.")
    args = ap.parse_args()

    tp_path = os.path.join(args.cache_dir, "text_part.f16.npy")
    ht_path = os.path.join(args.cache_dir, "has_text.bool.npy")
    if not os.path.exists(tp_path):
        raise SystemExit(f"[whiten] missing {tp_path}")

    print(f"[whiten] loading {tp_path} ...")
    t_arr = np.load(tp_path, mmap_mode="r")               # [N, M, D] float16
    if t_arr.ndim != 3:
        raise SystemExit(f"[whiten] expected [N, M, D], got {t_arr.shape}")
    N, M, D = t_arr.shape
    print(f"[whiten] shape N={N} M={M} D={D}")

    if not args.include_no_text and os.path.exists(ht_path):
        ht = np.load(ht_path)                              # [N] bool
        print(f"[whiten] has_text True: {int(ht.sum())}/{N}")
        keep_idx = np.where(ht)[0]
    else:
        keep_idx = np.arange(N)
    n_keep = len(keep_idx)
    if n_keep == 0:
        raise SystemExit("[whiten] no rows passed the has_text filter.")

    if args.residualize_first:
        print(f"[whiten] residualize_first: subtracting T_global (slot 0) "
              f"from local slots BEFORE covariance.")
        t0 = time.time()
        T_full = np.asarray(t_arr[keep_idx], dtype=np.float32)            # [N, 6, D]
        T_global = T_full[:, 0:1, :]                                       # [N, 1, D]
        T_local_res = T_full[:, 1:, :] - T_global                           # [N, 5, D]
        T_all = T_local_res.reshape(-1, D)                                  # [N*5, D]
        print(f"[whiten] T_all shape={T_all.shape}  load+residualize={time.time()-t0:.1f}s")
    else:
        print(f"[whiten] gathering {n_keep} rows * M={M} = {n_keep * M} vectors ...")
        t0 = time.time()
        # Materialize as float32 in one shot. N=25000, M=6, D=512 -> ~290 MB.
        T_all = np.asarray(t_arr[keep_idx], dtype=np.float32).reshape(-1, D)
        print(f"[whiten] T_all shape={T_all.shape}  load+cast={time.time()-t0:.1f}s")

    mu = T_all.mean(axis=0).astype(np.float32)             # [D]
    centered = T_all - mu
    # Sigma = X^T X / (n-1)
    print("[whiten] computing covariance ...")
    t0 = time.time()
    Sigma = (centered.T @ centered) / max(centered.shape[0] - 1, 1)
    print(f"[whiten] Sigma shape={Sigma.shape}  time={time.time()-t0:.1f}s")
    Sigma = (Sigma + Sigma.T) * 0.5                        # enforce symmetry

    print("[whiten] eigendecomp (symmetric) ...")
    t0 = time.time()
    # Eigh returns eigenvalues in ASCENDING order; sort descending for
    # interpretability ("top-PC first") but the matrix W gamma is
    # invariant to the column order of U.
    S_asc, U_asc = np.linalg.eigh(Sigma.astype(np.float64))
    order = np.argsort(-S_asc)
    S = S_asc[order].astype(np.float32)                    # [D]
    U = U_asc[:, order].astype(np.float32)                 # [D, D]
    print(f"[whiten] eigh done  time={time.time()-t0:.1f}s")
    print(f"[whiten] eigenvalue summary: max={S[0]:.4e}  median={np.median(S):.4e}  "
          f"min={S[-1]:.4e}  rank>0: {(S > 1e-8).sum()}/{len(S)}")
    pos_S = S[S > 1e-8]
    print(f"[whiten] effective participation: top1 = {pos_S[0] / pos_S.sum():.3f}, "
          f"top5 = {pos_S[:5].sum() / pos_S.sum():.3f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, mu=mu, U=U, S=S)
    meta = {
        "cache_dir": os.path.abspath(args.cache_dir),
        "N_kept": int(n_keep),
        "rows_used": int(centered.shape[0]),
        "D": int(D),
        "residualize_first": bool(args.residualize_first),
        "top1_ratio": float(pos_S[0] / pos_S.sum()),
        "top5_ratio": float(pos_S[:5].sum() / pos_S.sum()),
    }
    with open(args.out + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[whiten] wrote {args.out} + .meta.json")


if __name__ == "__main__":
    main()
