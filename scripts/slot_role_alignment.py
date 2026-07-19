"""Slot role validity: does code-slot m organize caption-slot m's semantics?

This tests a claim that does NOT require slots to be orthogonal:

  1. within-slot coherence -- samples sharing a codeword in slot m have more
     similar slot-m semantics than chance;
  2. role validity -- slot m's partition aligns with caption-slot m BETTER than
     with the other caption slots (diagonal dominance of the 6x6 lift matrix);
  3. codon preservation -- (1) and (2) survive the codeword -> codon merge.

Inter-slot redundancy is irrelevant here, and is expected on natural images
anyway (colour, objects and scene are correlated in the world, not just in the
model). What matters is whether each slot's partition tracks its own semantic
part.

Evaluated on images never trained on (database minus train, or the official
test split). Their captions were never used as supervision, so a lift here is
generalization, not memorization.

Lift for (code slot m, caption slot k):
    observed = mean over codewords u of mean pairwise cosine of caption-k
               embeddings among samples with slot-m code u
    baseline = same statistic under a size-matched random partition
    lift     = observed - baseline
Caption embeddings are centered per caption-slot first, so a slot's mean offset
cannot masquerade as alignment.

Usage:
    python scripts/slot_role_alignment.py \
        --ours_dir result/260717+flickr25k_..._P0refit_e4+... \
        --cache_dir cache/flickr25k_clip_v4plus_qwen3_tokens \
        --train_manifest dataset/Flickr25k/setting1/train.txt \
        --out docs/slot_role_alignment_flickr25k.json
"""
import argparse
import json
import os

import numpy as np

SLOT_NAMES = ["global", "primary_object", "secondary_object",
              "activity_relation", "color_texture", "scene_type"]


def _basenames(paths):
    return np.array([os.path.basename(str(p)) for p in paths])


def mean_pairwise_cos(X: np.ndarray) -> float:
    """Mean off-diagonal cosine among rows of X (already L2-normalized)."""
    n = len(X)
    if n < 2:
        return np.nan
    G = X @ X.T
    return float((G.sum() - np.trace(G)) / (n * (n - 1)))


def lift_matrix(codes: np.ndarray, text: np.ndarray, min_support: int,
                n_shuffle: int, rng) -> tuple:
    """codes [N, M] integer symbols; text [N, K, D] L2-normalized, centered.

    Returns (lift [M, K], observed [M, K], baseline [M, K], support [M]).
    """
    M = codes.shape[1]
    K = text.shape[1]
    obs = np.full((M, K), np.nan)
    base = np.full((M, K), np.nan)
    support = np.zeros(M, dtype=np.int64)

    for m in range(M):
        u, inv = np.unique(codes[:, m], return_inverse=True)
        groups = [np.where(inv == i)[0] for i in range(len(u))]
        groups = [g for g in groups if len(g) >= min_support]
        support[m] = len(groups)
        if not groups:
            continue
        sizes = np.array([len(g) for g in groups])

        for k in range(K):
            Xk = text[:, k, :]
            obs[m, k] = float(np.mean([mean_pairwise_cos(Xk[g]) for g in groups]))
            # size-matched random partition, averaged over shuffles
            vals = []
            for _ in range(n_shuffle):
                perm = rng.permutation(len(Xk))
                off = 0
                acc = []
                for s in sizes:
                    acc.append(mean_pairwise_cos(Xk[perm[off:off + s]]))
                    off += s
                vals.append(np.mean(acc))
            base[m, k] = float(np.mean(vals))
    return obs - base, obs, base, support


def summarize(lift: np.ndarray) -> dict:
    """Diagonal dominance: is slot m most aligned with caption slot m?"""
    M = lift.shape[0]
    diag = np.array([lift[m, m] for m in range(M)])
    offd = np.array([np.mean([lift[m, k] for k in range(M) if k != m])
                     for m in range(M)])
    rank = []
    for m in range(M):
        order = np.argsort(-lift[m])          # descending
        rank.append(int(np.where(order == m)[0][0]) + 1)   # 1 = best
    return {
        "diag_mean": float(diag.mean()),
        "offdiag_mean": float(offd.mean()),
        "diag_minus_offdiag": float((diag - offd).mean()),
        "per_slot_diag": diag.tolist(),
        "per_slot_offdiag": offd.tolist(),
        "per_slot_rank_of_own_caption": rank,
        "n_slots_rank1": int(sum(r == 1 for r in rank)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours_dir", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--train_manifest", default=None,
                    help="Exclude these basenames (they were trained on).")
    ap.add_argument("--split", default="db", choices=["db", "query"])
    ap.add_argument("--whiten_npz", default=None)
    ap.add_argument("--whiten_gamma", type=float, default=0.25)
    ap.add_argument("--min_support", type=int, default=10)
    ap.add_argument("--n_shuffle", type=int, default=5)
    ap.add_argument("--max_n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    z = np.load(os.path.join(args.ours_dir, f"extract_{args.split}.npz"),
                allow_pickle=True)
    bn = _basenames(z["image_paths"])
    keep = np.ones(len(bn), dtype=bool)
    if args.train_manifest:
        tr = {os.path.basename(l.split()[0])
              for l in open(args.train_manifest) if l.strip()}
        keep &= ~np.isin(bn, list(tr))
        print(f"[role] excluded {(~keep).sum()} trained-on rows -> {keep.sum()} held-out")
    idx = np.where(keep)[0]
    if len(idx) > args.max_n:
        idx = rng.choice(idx, args.max_n, replace=False)

    cb = z["codebook_indices"][idx]                       # [N, 6]
    b = z["base_indices"][idx]
    codon = b[:, 0::3] * 16 + b[:, 1::3] * 4 + b[:, 2::3]  # [N, 6] 0..63

    # caption embeddings for exactly these rows
    ids = json.load(open(os.path.join(args.cache_dir, "image_ids.json")))
    row_of = {os.path.basename(str(s)): i for i, s in enumerate(ids)}
    rows = np.array([row_of[b_] for b_ in bn[idx]])
    T = np.asarray(np.load(os.path.join(args.cache_dir, "text_part.f16.npy"),
                           mmap_mode="r")[rows], dtype=np.float64)   # [N, 6, D]
    has = np.asarray(np.load(os.path.join(args.cache_dir, "has_text.bool.npy"),
                             mmap_mode="r")[rows]).astype(bool)
    if not has.all():
        print(f"[role] dropping {(~has).sum()} rows without captions")
        cb, codon, T = cb[has], codon[has], T[has]

    if args.whiten_npz:
        w = np.load(args.whiten_npz)
        mu, U, S = w["mu"].astype(np.float64), w["U"].astype(np.float64), w["S"].astype(np.float64)
        W = U @ np.diag((S + 1e-5) ** (-args.whiten_gamma)) @ U.T
        T = (T.reshape(-1, T.shape[-1]) - mu) @ W
        T = T.reshape(-1, 6, W.shape[0])

    # center per caption slot, then L2 normalize -- a slot's mean offset must
    # not be readable as alignment
    T = T - T.mean(axis=0, keepdims=True)
    T /= (np.linalg.norm(T, axis=-1, keepdims=True) + 1e-12)
    print(f"[role] N={len(T)}  units: codeword K={cb.max()+1}, codon 64")

    out = {"ours_dir": args.ours_dir, "split": args.split, "n": int(len(T)),
           "config": vars(args)}
    for name, units in [("codeword", cb), ("codon", codon)]:
        lift, obs, base, sup = lift_matrix(units, T, args.min_support,
                                           args.n_shuffle, rng)
        s = summarize(lift)
        out[name] = {"lift": lift.tolist(), "observed": obs.tolist(),
                     "baseline": base.tolist(), "n_groups": sup.tolist(), **s}
        print(f"\n=== {name}: lift matrix (row = code slot, col = caption slot) ===")
        print("            " + "".join(f"{n[:8]:>9s}" for n in SLOT_NAMES))
        for m in range(6):
            cells = "".join(
                (f"{lift[m,k]:>8.4f}*" if k == m else f"{lift[m,k]:>9.4f}")
                for k in range(6))
            print(f"  {SLOT_NAMES[m][:10]:>10s}{cells}")
        print(f"  diag {s['diag_mean']:.4f} vs off-diag {s['offdiag_mean']:.4f} "
              f"(+{s['diag_minus_offdiag']:.4f});  own-caption rank-1 in "
              f"{s['n_slots_rank1']}/6 slots; ranks {s['per_slot_rank_of_own_caption']}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\n[role] wrote {args.out}")


if __name__ == "__main__":
    main()
