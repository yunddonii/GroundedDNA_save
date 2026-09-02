"""Pairwise normalized mutual information between codebook assignments.

Loads codebook_indices [N, M] from extract_db.npz, computes NMI between every
pair of codebooks, and writes a JSON + a tiny console table. Low pairwise NMI
means the codebooks carry independent information; high NMI means redundancy.
"""
import argparse, json, math, os, sys, tempfile

# `python scripts/pairwise_nmi.py` puts `scripts/` on sys.path, not the repo
# root, so the binding import below raised ModuleNotFoundError from every cwd --
# including the repo's own. It sat inside the per-directory loop, so an empty
# run exited 0 and hid it.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import sklearn
from sklearn.metrics import normalized_mutual_info_score

from dna_utils.extraction_validation import ANALYSIS_MARKER_NAME, metric_input_binding
from evaluation_siglip2 import _load_npz_bound


def pairwise_nmi(codebook_indices):
    M = codebook_indices.shape[1]
    out = np.zeros((M, M), dtype=np.float64)
    for i in range(M):
        out[i, i] = 1.0
        for j in range(i + 1, M):
            # `average_method` is stated rather than defaulted: it has
            # changed default across sklearn releases, and it changes every
            # number in this matrix.
            #
            # Each unordered pair is scored ONCE and mirrored. NMI is symmetric
            # by definition, but `score(a, b)` and `score(b, a)` are not
            # bit-identical -- the mutual-information and entropy sums
            # accumulate in a different order, which moved entries by ~1 ULP
            # (1.1e-16 observed). Scoring both halves therefore produced a
            # matrix that `seal_cell_analysis._check_nmi` rejects, since it
            # requires `array_equal(matrix, matrix.T)` exactly. On the
            # 2026-09-02 diagnostic that refused 29 of 30 cells, each with
            # "pairwise_nmi.json matrix is malformed", while the one cell whose
            # twenty off-diagonal entries happened to round identically passed.
            # Mirroring also halves the work.
            out[i, j] = out[j, i] = normalized_mutual_info_score(
                codebook_indices[:, i], codebook_indices[:, j],
                average_method="arithmetic",
            )
    return out


def _atomic_json(path, payload):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True,
                      allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _assert_artifact_current(artifact, *, what):
    """Require the pathname to still name the exact inode loaded by NMI."""
    path = str(artifact.get("path") or "")
    if os.path.abspath(path) != path or os.path.realpath(path) != path:
        raise RuntimeError(f"{what}: input path is not canonical: {path!r}")
    try:
        current = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise RuntimeError(f"{what}: input disappeared after load: {error}") \
            from None
    exact = {
        "st_dev": int(current.st_dev), "st_ino": int(current.st_ino),
        "size_bytes": int(current.st_size),
        "st_mtime_ns": int(current.st_mtime_ns),
        "st_ctime_ns": int(current.st_ctime_ns),
        "resolved_path": os.path.realpath(path),
    }
    wrong = {key: (artifact.get(key), value)
             for key, value in exact.items() if artifact.get(key) != value}
    if wrong:
        raise RuntimeError(
            f"{what}: input changed or pathname was replaced after the "
            f"same-descriptor load: {wrong}")


def _record_for_result(result_dir, *, allow_backfilled, required_splits):
    result_dir = os.path.abspath(result_dir)
    npz = os.path.join(result_dir, "extract_db.npz")
    # Validate and hash the complete extraction before opening an analysis
    # input. Missing/malformed inputs are fatal; silently skipping one result
    # made a multi-cell command exit 0 with an incomplete campaign.
    before = metric_input_binding(
        result_dir, required_splits=required_splits,
        allow_backfilled=allow_backfilled)
    arrays, input_artifact = _load_npz_bound(npz)
    if input_artifact.get("path") != npz \
            or input_artifact.get("resolved_path") != npz:
        raise RuntimeError(
            f"{npz}: NMI did not load the canonical adjacent DB extraction")
    if input_artifact.get("sha256") != (before.get("npz_sha256") or {}).get("db"):
        raise RuntimeError(
            f"{npz}: same-descriptor bytes differ from the validated DB "
            "extraction binding")
    if "codebook_indices" not in arrays:
        raise RuntimeError(f"{npz} has no codebook_indices")
    ci = np.asarray(arrays["codebook_indices"])
    if ci.ndim != 2 or ci.shape[0] <= 0 or ci.shape[1] < 2:
        raise RuntimeError(
            f"{npz} codebook_indices must have shape [N>0, M>=2], found "
            f"{ci.shape}")
    if not np.issubdtype(ci.dtype, np.integer):
        raise RuntimeError(f"{npz} codebook_indices dtype is {ci.dtype}, not integer")

    nmi = pairwise_nmi(ci)
    if not np.isfinite(nmi).all() or np.any(nmi < 0.0) or np.any(nmi > 1.0):
        raise RuntimeError(f"{npz} produced a non-finite/out-of-range NMI")
    after = metric_input_binding(
        result_dir, required_splits=required_splits,
        allow_backfilled=allow_backfilled)
    if after != before:
        raise RuntimeError(
            f"{result_dir}: extraction changed while pairwise NMI was computed")
    _assert_artifact_current(input_artifact, what=npz)

    off_diag = nmi[~np.eye(nmi.shape[0], dtype=bool)]
    mean = float(off_diag.mean())
    maximum = float(off_diag.max())
    minimum = float(off_diag.min())
    if not all(math.isfinite(value) for value in (mean, maximum, minimum)):
        raise RuntimeError(f"{npz} produced a non-finite NMI summary")
    return {
        "schema_version": 3,
        "input_binding": before,
        "codebook_input_artifact": input_artifact,
        "nmi_average_method": "arithmetic",
        "sklearn_version": sklearn.__version__,
        "mean_off_diag_nmi": mean,
        "max_off_diag_nmi": maximum,
        "min_off_diag_nmi": minimum,
        "nmi_matrix": nmi.tolist(),
        # Preserve the historical field semantics (largest observed index + 1)
        # and expose actual occupancy under a new, explicit name.
        "K_per_cb": [int(ci[:, m].max() + 1) for m in range(ci.shape[1])],
        "unique_codewords_per_cb": [int(np.unique(ci[:, m]).size)
                                    for m in range(ci.shape[1])],
        "unique_codes_in_db": int(np.unique(ci, axis=0).shape[0]),
        "N": int(ci.shape[0]),
        "num_codebooks": int(ci.shape[1]),
    }, nmi, ci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True,
                    help="result directories each containing extract_db.npz")
    ap.add_argument("--allow-backfilled", action="store_true",
                    help=("Admit retrospectively bound inputs. Off by default "
                          "so a paper path cannot pick them up silently."))
    ap.add_argument(
        "--require-train", action="store_true",
        help=("Require the MAIN db/query/train extraction transaction. Leave "
              "off only for two-split diagnostics."))
    ap.add_argument("--out", default=None,
                    help="optional combined JSON output path")
    args = ap.parse_args()
    allow_backfilled = args.allow_backfilled
    required_splits = ("db", "query", "train") if args.require_train else (
        "db", "query")

    canonical_results = [os.path.abspath(d) for d in args.results]
    if len(set(canonical_results)) != len(canonical_results):
        raise SystemExit("--results contains a duplicate result directory")
    tags = [os.path.basename(d.rstrip(os.sep)) for d in canonical_results]
    if len(set(tags)) != len(tags):
        raise SystemExit(
            "--results contains different directories with the same basename; "
            "the combined JSON key would collide")
    if args.out and os.path.basename(args.out) == "pairwise_nmi.json" and \
            os.path.dirname(os.path.abspath(args.out)) in set(canonical_results):
        raise SystemExit(
            "--out must not be the per-result pairwise_nmi.json; that file is "
            "written per directory in the direct layout")

    # Compute and validate every requested result first. No direct output is
    # published if a later member of the batch is missing or malformed.
    prepared = []
    combined = {}
    for d, tag in zip(canonical_results, tags):
        record, nmi, ci = _record_for_result(
            d, allow_backfilled=allow_backfilled,
            required_splits=required_splits)
        prepared.append((d, tag, record, nmi, ci))
        combined[tag] = record

    for d, tag, record, nmi, ci in prepared:
        direct = os.path.join(d, "pairwise_nmi.json")
        # Replacing either analysis half invalidates any earlier integrity
        # marker. Remove it before publishing the new NMI so an interrupted
        # re-analysis cannot leave an apparently complete stale result.
        stale = os.path.join(d, ANALYSIS_MARKER_NAME)
        if os.path.exists(stale):
            os.unlink(stale)
        _atomic_json(direct, record)
        print(f"\n=== {tag} ===")
        print(f"  N={ci.shape[0]}  M={ci.shape[1]}  unique={combined[tag]['unique_codes_in_db']}")
        print(f"  mean off-diag NMI: {record['mean_off_diag_nmi']:.4f}")
        print(f"  max  off-diag NMI: {record['max_off_diag_nmi']:.4f}")
        print(f"  min  off-diag NMI: {record['min_off_diag_nmi']:.4f}")
        # Pretty print matrix
        print("  matrix:")
        for r in range(nmi.shape[0]):
            print("   " + "  ".join(f"{v:.3f}" for v in nmi[r]))

    if args.out:
        _atomic_json(args.out, combined)
        print(f"\nWrote combined JSON to {args.out}")


if __name__ == "__main__":
    main()
