"""Pairwise normalized mutual information between codebook assignments.

Loads codebook_indices [N, M] from extract_db.npz, computes NMI between every
pair of codebooks, and writes a JSON + a tiny console table. Low pairwise NMI
means the codebooks carry independent information; high NMI means redundancy.
"""
import argparse, json, os, sys

# `python scripts/pairwise_nmi.py` puts `scripts/` on sys.path, not the repo
# root, so the binding import below raised ModuleNotFoundError from every cwd --
# including the repo's own. It sat inside the per-directory loop, so an empty
# run exited 0 and hid it.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import sklearn
from sklearn.metrics import normalized_mutual_info_score

from dna_utils.extraction_validation import metric_input_binding


def pairwise_nmi(codebook_indices):
    M = codebook_indices.shape[1]
    out = np.zeros((M, M), dtype=np.float64)
    for i in range(M):
        for j in range(M):
            if i == j:
                out[i, j] = 1.0
            else:
                # `average_method` is stated rather than defaulted: it has
                # changed default across sklearn releases, and it changes every
                # number in this matrix.
                out[i, j] = normalized_mutual_info_score(
                    codebook_indices[:, i], codebook_indices[:, j],
                    average_method="arithmetic",
                )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True,
                    help="result directories each containing extract_db.npz")
    ap.add_argument("--allow-backfilled", action="store_true",
                    help=("Admit retrospectively bound inputs. Off by default "
                          "so a paper path cannot pick them up silently."))
    ap.add_argument("--out", default=None,
                    help="optional combined JSON output path")
    args = ap.parse_args()
    allow_backfilled = args.allow_backfilled

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
        # A broad `except` turned `No module named 'dna_utils'` into a data
        # field and still exited 0, so every failure was logged as `nmi ok`.
        # The import is now at module scope, where an unimportable dependency
        # fails before any directory is processed rather than per directory.
        _binding = metric_input_binding(d, allow_backfilled=allow_backfilled)
        off_diag = nmi[~np.eye(nmi.shape[0], dtype=bool)]
        tag = os.path.basename(d.rstrip("/"))
        combined[tag] = {
            "input_binding": _binding,
            # The settings that decide the number, recorded by the stage that
            # actually used them; the seal copies them into the protocol block
            # rather than asserting them from elsewhere.
            "nmi_average_method": "arithmetic",
            "sklearn_version": sklearn.__version__,
            "mean_off_diag_nmi": float(off_diag.mean()),
            "max_off_diag_nmi": float(off_diag.max()),
            "min_off_diag_nmi": float(off_diag.min()),
            "nmi_matrix": nmi.tolist(),
            "K_per_cb": [int(ci[:, m].max() + 1) for m in range(ci.shape[1])],
            "unique_codes_in_db": int(np.unique(ci, axis=0).shape[0]),
            "N": int(ci.shape[0]),
        }
        # Persist per-result file, atomically: the seal downstream refuses a
        # file it cannot parse, but a truncated one that still parses would be
        # read as a smaller, wrong NMI.
        direct = os.path.join(d, "pairwise_nmi.json")
        tmp = f"{direct}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump(combined[tag], f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, direct)
        print(f"\n=== {tag} ===")
        print(f"  N={ci.shape[0]}  M={ci.shape[1]}  unique={combined[tag]['unique_codes_in_db']}")
        print(f"  mean off-diag NMI: {off_diag.mean():.4f}")
        print(f"  max  off-diag NMI: {off_diag.max():.4f}")
        print(f"  min  off-diag NMI: {off_diag.min():.4f}")
        # Pretty print matrix
        print("  matrix:")
        for r in range(nmi.shape[0]):
            print("   " + "  ".join(f"{v:.3f}" for v in nmi[r]))

    # `--out` pointing at the per-result filename used to overwrite the direct
    # record with the combined layout, hiding the binding one level down.
    if args.out and os.path.basename(args.out) == "pairwise_nmi.json" and \
            os.path.dirname(os.path.abspath(args.out)) in {
                os.path.abspath(d) for d in args.results}:
        raise SystemExit(
            "--out must not be the per-result pairwise_nmi.json; that file is "
            "written per directory in the direct layout")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(combined, f, indent=2)
        print(f"\nWrote combined JSON to {args.out}")


if __name__ == "__main__":
    main()
