"""Pairwise normalized mutual information between codebook assignments.

Loads codebook_indices [N, M] from extract_db.npz, computes NMI between every
pair of codebooks, and writes a JSON + a tiny console table. Low pairwise NMI
means the codebooks carry independent information; high NMI means redundancy.
"""
import argparse, json, os
import numpy as np
from sklearn.metrics import normalized_mutual_info_score


def pairwise_nmi(codebook_indices):
    M = codebook_indices.shape[1]
    out = np.zeros((M, M), dtype=np.float64)
    for i in range(M):
        for j in range(M):
            if i == j:
                out[i, j] = 1.0
            else:
                out[i, j] = normalized_mutual_info_score(
                    codebook_indices[:, i], codebook_indices[:, j]
                )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True,
                    help="result directories each containing extract_db.npz")
    ap.add_argument("--out", default=None,
                    help="optional combined JSON output path")
    args = ap.parse_args()

    combined = {}
    for d in args.results:
        npz = os.path.join(d, "extract_db.npz")
        if not os.path.exists(npz):
            print(f"[skip] {npz} missing")
            continue
        z = np.load(npz, allow_pickle=True)
        if "codebook_indices" not in z.files:
            print(f"[skip] {npz} no codebook_indices")
            continue
        ci = z["codebook_indices"]
        nmi = pairwise_nmi(ci)
        off_diag = nmi[~np.eye(nmi.shape[0], dtype=bool)]
        tag = os.path.basename(d.rstrip("/"))
        combined[tag] = {
            "mean_off_diag_nmi": float(off_diag.mean()),
            "max_off_diag_nmi": float(off_diag.max()),
            "min_off_diag_nmi": float(off_diag.min()),
            "nmi_matrix": nmi.tolist(),
            "K_per_cb": [int(ci[:, m].max() + 1) for m in range(ci.shape[1])],
            "unique_codes_in_db": int(np.unique(ci, axis=0).shape[0]),
            "N": int(ci.shape[0]),
        }
        # Persist per-result file
        with open(os.path.join(d, "pairwise_nmi.json"), "w") as f:
            json.dump(combined[tag], f, indent=2)
        print(f"\n=== {tag} ===")
        print(f"  N={ci.shape[0]}  M={ci.shape[1]}  unique={combined[tag]['unique_codes_in_db']}")
        print(f"  mean off-diag NMI: {off_diag.mean():.4f}")
        print(f"  max  off-diag NMI: {off_diag.max():.4f}")
        print(f"  min  off-diag NMI: {off_diag.min():.4f}")
        # Pretty print matrix
        print("  matrix:")
        for r in range(nmi.shape[0]):
            print("   " + "  ".join(f"{v:.3f}" for v in nmi[r]))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(combined, f, indent=2)
        print(f"\nWrote combined JSON to {args.out}")


if __name__ == "__main__":
    main()
