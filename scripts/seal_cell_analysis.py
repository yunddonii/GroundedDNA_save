"""Publish a cell's integrity-only `analysis_complete.json` after BOTH halves.

A cell's analysis is two computations with two different sources:

    scripts/eval_cell_bioproj.py  ->  cell_result.json   (retrieval, DNA unique)
    scripts/pairwise_nmi.py       ->  pairwise_nmi.json  (codebook redundancy)

The bio-evaluation used to seal `analysis_complete.json` itself, immediately
after its own half. The sealed metric set therefore had no `mean_off_diag_nmi`
at all, the NMI stage never re-sealed anything, and the aggregator read the
missing key with `.get()` and still counted the cell as paired -- so a report of
"15 paired" with a blank NMI column throughout was reachable. A file named
`analysis_complete` must be written by something that can see the whole
analysis, which is what this is.

It refuses unless, for the cell it is given:

  * both partial files exist, parse, and each declares an input binding;
  * both bindings are identical to the extraction's binding right now, so the
    two halves describe the same forward pass and not two different ones;
  * the metric set is exactly the five the contract requires, each a finite
    proportion;
  * the protocol block is complete, including the NMI averaging method and the
    sklearn version that produced it, copied from the stage that used them;
  * raw/post evaluation, cell result, NMI and extraction bytes/protocol all
    reopen exactly and every analysis source file is current.

This standalone seal is deliberately marked ``scientific_authority=false``.
It proves integrity and cross-file consistency, not that finite JSON metrics
were produced by the claimed arithmetic. Paper admission requires a downstream
receipt verifier that deterministically recomputes the metrics from bound NPZs.

Usage:
    python scripts/seal_cell_analysis.py --dir <cell> [--allow-backfilled]
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import sklearn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dna_utils.extraction_validation import (  # noqa: E402
    ANALYSIS_MARKER_NAME,
    ExtractionInvalid,
    analysis_source_digests,
    check_metric_input_binding,
    metric_input_binding,
    validate_extraction_run,
    write_analysis_marker,
)
from dna_utils.gc_policy import resolve_gc_policy  # noqa: E402
from dna_utils.runtime_state import sha256_file  # noqa: E402
from evaluation_siglip2 import (  # noqa: E402
    AP_NORMALIZATION_POLICY,
    RANKING_POLICY,
    RELEVANCE_POLICY,
)
from scripts.eval_cell_bioproj import (  # noqa: E402
    _check_evaluation_payload,
    _read_json_artifact,
    _validated_npz_artifacts,
)


class SealRefused(RuntimeError):
    """The two halves do not together make one admissible analysis."""


def _read(cell: Path, name: str) -> tuple[dict, dict]:
    path = Path(os.path.abspath(cell / name))
    try:
        payload, artifact = _read_json_artifact(str(path))
    except RuntimeError as error:
        raise SealRefused(str(error)) from None
    if artifact.get("path") != str(path) \
            or artifact.get("resolved_path") != str(path):
        raise SealRefused(
            f"{cell.name}/{name}: analysis input is not the canonical "
            "adjacent regular file")
    return payload, artifact


def _number(payload: dict, key: str, *, where: str) -> float:
    value = payload.get(key)
    if value is None:
        raise SealRefused(f"{where} has no {key}")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SealRefused(f"{where}.{key} is {value!r}, not a number")
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise SealRefused(
            f"{where}.{key}={value!r} is not a finite proportion")
    return numeric


def _atomic_json(path: Path, payload: dict) -> None:
    directory = str(path.parent)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True,
                      allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        dir_fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _check_metric_policy(raw: dict, post: dict, *, cutoff: int) -> None:
    expected = {
        "ranking": RANKING_POLICY,
        "ap_normalization": AP_NORMALIZATION_POLICY,
        "relevance": RELEVANCE_POLICY,
        "multi_label_relevance_threshold": 0.0,
        "remove_self_match": False,
        "map_at_r": cutoff,
        "query_chunk_size": 64,
    }
    for name, payload in (("raw", raw), ("bio_projected", post)):
        if payload.get("metric_policy") != expected \
                or payload.get("query_chunk_size") != 64:
            raise SealRefused(
                f"{name} evaluation does not carry the exact standalone "
                "retrieval protocol")


def _check_bio_stats(stats: object, *, raw: dict, total_bases: int) -> dict:
    if not isinstance(stats, dict):
        raise SealRefused("bio-projected evaluation has no bio_stats object")
    expected_keys = {
        "mAP_pre_projection", "mAP_at_R_pre_projection",
        "mAP_R_cutoff_pre_projection",
        *(f"{prefix}_{suffix}" for prefix in ("db", "qy") for suffix in (
            "pre_compliance", "post_compliance", "mean_edit_distance",
            "max_edit_distance", "num_total", "num_pre_invalid",
            "num_proj_failed")),
    }
    if set(stats) != expected_keys:
        raise SealRefused(
            "bio_stats has a noncanonical key set: "
            f"missing={sorted(expected_keys-set(stats))}, "
            f"extra={sorted(set(stats)-expected_keys)}")
    pre = {
        "mAP_pre_projection": raw.get("mAP"),
        "mAP_at_R_pre_projection": raw.get("mAP_at_R"),
        "mAP_R_cutoff_pre_projection": raw.get("mAP_R_cutoff"),
    }
    if any(stats.get(key) != value for key, value in pre.items()):
        raise SealRefused(
            "bio_stats pre-projection values do not equal the raw evaluation")
    for prefix in ("db", "qy"):
        for suffix in ("pre_compliance", "post_compliance"):
            _number(stats, f"{prefix}_{suffix}", where="bio_stats")
        if stats[f"{prefix}_post_compliance"] != 1.0 \
                or stats[f"{prefix}_num_proj_failed"] != 0:
            raise SealRefused(
                f"bio_stats records an incomplete {prefix} projection")
        for suffix in ("num_total", "num_pre_invalid", "num_proj_failed",
                       "max_edit_distance"):
            value = stats[f"{prefix}_{suffix}"]
            if isinstance(value, bool) or not isinstance(value, int) \
                    or value < 0:
                raise SealRefused(
                    f"bio_stats.{prefix}_{suffix} is not a nonnegative int")
        mean = stats[f"{prefix}_mean_edit_distance"]
        if isinstance(mean, bool) or not isinstance(mean, (int, float)) \
                or not math.isfinite(float(mean)) \
                or not 0.0 <= float(mean) <= total_bases:
            raise SealRefused(
                f"bio_stats.{prefix}_mean_edit_distance is invalid")
        if stats[f"{prefix}_max_edit_distance"] > total_bases \
                or stats[f"{prefix}_num_pre_invalid"] \
                > stats[f"{prefix}_num_total"]:
            raise SealRefused(f"bio_stats.{prefix} counts are inconsistent")
    return stats


def _check_nmi(nmi: dict, *, db_artifact: dict, expected_rows: int,
               expected_codebooks: int) -> None:
    if nmi.get("schema_version") != 3:
        raise SealRefused("pairwise_nmi.json is not current schema_version 3")
    if nmi.get("codebook_input_artifact") != db_artifact:
        raise SealRefused(
            "pairwise_nmi.json was not computed from the exact canonical DB "
            "inode bound by the extraction")
    if nmi.get("nmi_average_method") != "arithmetic" \
            or nmi.get("sklearn_version") != sklearn.__version__:
        raise SealRefused(
            "pairwise_nmi.json arithmetic/sklearn protocol differs from this "
            "environment")
    if isinstance(nmi.get("N"), bool) or isinstance(
            nmi.get("num_codebooks"), bool) \
            or nmi.get("N") != expected_rows \
            or nmi.get("num_codebooks") != expected_codebooks:
        raise SealRefused("pairwise_nmi.json shape identity is wrong")
    try:
        matrix = np.asarray(nmi.get("nmi_matrix"))
    except (TypeError, ValueError):
        raise SealRefused("pairwise_nmi.json matrix is malformed") from None
    if matrix.shape != (expected_codebooks, expected_codebooks) \
            or not np.issubdtype(matrix.dtype, np.number) \
            or not np.isfinite(matrix).all() \
            or np.any(matrix < 0.0) or np.any(matrix > 1.0) \
            or not np.array_equal(np.diag(matrix), np.ones(expected_codebooks)) \
            or not np.array_equal(matrix, matrix.T):
        raise SealRefused("pairwise_nmi.json matrix is malformed")
    off_diag = matrix[~np.eye(expected_codebooks, dtype=bool)]
    exact = {
        "mean_off_diag_nmi": float(off_diag.mean()),
        "max_off_diag_nmi": float(off_diag.max()),
        "min_off_diag_nmi": float(off_diag.min()),
    }
    if any(nmi.get(key) != value for key, value in exact.items()):
        raise SealRefused(
            "pairwise_nmi.json summaries do not equal its recorded matrix")


def seal(cell: Path, *, allow_backfilled: bool,
         required_splits: tuple[str, ...] = ("db", "query")) -> str:
    cell = Path(os.path.abspath(cell))
    evaluation, cell_artifact = _read(cell, "cell_result.json")
    nmi, nmi_artifact = _read(cell, "pairwise_nmi.json")

    # Both halves must name the SAME inputs, and those inputs must be this
    # cell's, checked against the extraction as it stands now rather than
    # against each other alone.
    for name, payload in (("cell_result.json", evaluation),
                          ("pairwise_nmi.json", nmi)):
        try:
            check_metric_input_binding(
                str(cell), payload, what=f"{cell.name}/{name}",
                allow_backfilled=allow_backfilled,
                required_splits=required_splits)
        except ExtractionInvalid as error:
            raise SealRefused(str(error)) from None
    if evaluation["input_binding"] != nmi["input_binding"]:
        raise SealRefused(
            f"{cell.name}: the two analysis halves declare different inputs; "
            f"they describe different forward passes")

    binding = metric_input_binding(
        str(cell), required_splits=required_splits,
        allow_backfilled=allow_backfilled)
    run = validate_extraction_run(
        str(cell), required_splits=required_splits,
        allow_backfilled=allow_backfilled)
    actual_bases, npz_artifacts = _validated_npz_artifacts(
        str(cell), required_splits=required_splits, binding=binding)
    evaluator_inputs = {
        split: npz_artifacts[split] for split in ("db", "query")}

    raw, raw_artifact = _read(cell, "evaluation_siglip2_base.json")
    post, post_artifact = _read(
        cell, "evaluation_siglip2_base_bioproj.json")
    declared_evaluations = evaluation.get("evaluation_artifacts")
    exact_evaluations = {
        "raw": raw_artifact, "bio_projected": post_artifact}
    if declared_evaluations != exact_evaluations:
        raise SealRefused(
            f"{cell.name}/cell_result.json does not bind the exact raw/post "
            "evaluation JSON files reopened by the seal")
    if evaluation.get("evaluator_input_artifacts") != evaluator_inputs:
        raise SealRefused(
            f"{cell.name}/cell_result.json evaluator inputs do not equal the "
            "canonical extraction NPZs")

    dataset = evaluation.get("dataset")
    codebook_size = evaluation.get("K")
    cutoff = evaluation.get("map_r_cutoff")
    try:
        _check_evaluation_payload(
            raw, where="evaluation_siglip2_base.json", dataset=dataset,
            codebook_size=codebook_size, cutoff=cutoff,
            expected_inputs=evaluator_inputs, bio_project=False)
        _check_evaluation_payload(
            post, where="evaluation_siglip2_base_bioproj.json",
            dataset=dataset, codebook_size=codebook_size, cutoff=cutoff,
            expected_inputs=evaluator_inputs, bio_project=True)
    except RuntimeError as error:
        raise SealRefused(str(error)) from None
    if dataset != run.common.get("dataset") \
            or codebook_size != run.common.get("codebook_size") \
            or evaluation.get("total_bases") != actual_bases:
        raise SealRefused(
            f"{cell.name}: cell/evaluation protocol does not equal the "
            "validated extraction identity")
    expected_rows = {
        "db": run.splits["db"]["n_rows"],
        "query": run.splits["query"]["n_rows"],
    }
    for name, payload in (("raw", raw), ("bio_projected", post)):
        if payload.get("n_db") != expected_rows["db"] \
                or payload.get("n_query") != expected_rows["query"]:
            raise SealRefused(
                f"{name} evaluation row counts differ from extraction manifests")
    _check_metric_policy(raw, post, cutoff=cutoff)

    policy = resolve_gc_policy(actual_bases)
    expected_gc = {
        "bio_gc_min_frac": policy.gc_min_frac,
        "bio_gc_max_frac": policy.gc_max_frac,
        "bio_max_homopolymer_run": policy.max_run,
    }
    if any(post.get(key) != value for key, value in expected_gc.items()):
        raise SealRefused(
            f"{cell.name}: post-BIO evaluator did not use the central GC "
            "protocol")
    bio_stats = _check_bio_stats(
        post.get("bio_stats"), raw=raw, total_bases=actual_bases)
    if evaluation.get("bio_stats") != bio_stats:
        raise SealRefused(
            f"{cell.name}/cell_result.json does not bind the exact BIO stats")

    _check_nmi(
        nmi, db_artifact=npz_artifacts["db"],
        expected_rows=expected_rows["db"],
        expected_codebooks=run.splits["db"]["num_slots"])

    metrics = {
        "map_at_R_bioproj": _number(evaluation, "mAP_at_R_bioproj",
                                    where=f"{cell.name}/cell_result.json"),
        "full_map_bioproj": _number(evaluation, "full_mAP_bioproj",
                                    where=f"{cell.name}/cell_result.json"),
        "full_map_pre_projection": _number(
            evaluation, "full_mAP_pre_projection",
            where=f"{cell.name}/cell_result.json"),
        "dna_unique_db": _number(evaluation, "DNA_unique_DB",
                                 where=f"{cell.name}/cell_result.json"),
        "mean_off_diag_nmi": _number(nmi, "mean_off_diag_nmi",
                                     where=f"{cell.name}/pairwise_nmi.json"),
    }
    exact_metrics = {
        "map_at_R_bioproj": post.get("mAP_at_R"),
        "full_map_bioproj": post.get("mAP"),
        "full_map_pre_projection": raw.get("mAP"),
        "dna_unique_db": post.get("unique_code_ratio"),
        "mean_off_diag_nmi": nmi.get("mean_off_diag_nmi"),
    }
    if metrics != exact_metrics:
        raise SealRefused(
            f"{cell.name}: cell/NMI summary metrics do not exactly equal the "
            "strictly reopened raw/post/NMI producer artifacts")

    method = nmi.get("nmi_average_method")
    version = nmi.get("sklearn_version")
    if not method or not version:
        raise SealRefused(
            f"{cell.name}/pairwise_nmi.json predates the recorded averaging "
            f"method and sklearn version; re-run the NMI so the number can be "
            f"interpreted")

    protocol = {
        "dataset": evaluation.get("dataset"),
        "codebook_size": evaluation.get("K"),
        "total_bases": evaluation.get("total_bases"),
        "map_r_cutoff": evaluation.get("map_r_cutoff"),
        "gc_policy_version": evaluation.get("gc_policy_version"),
        "gc_count_min_inclusive": evaluation.get("gc_count_min_inclusive"),
        "gc_count_max_inclusive": evaluation.get("gc_count_max_inclusive"),
        "gc_min_frac": evaluation.get("gc_min_frac"),
        "gc_max_frac": evaluation.get("gc_max_frac"),
        "bio_max_homopolymer_run": evaluation.get(
            "bio_max_homopolymer_run"),
        "nmi_average_method": method,
        "sklearn_version": version,
    }
    unset = sorted(k for k, v in protocol.items() if v is None)
    if unset:
        raise SealRefused(
            f"{cell.name}: the analysis protocol is incomplete, no "
            f"{', '.join(unset)}")

    if protocol != {
            "dataset": dataset,
            "codebook_size": codebook_size,
            "total_bases": actual_bases,
            "map_r_cutoff": cutoff,
            "gc_policy_version": policy.policy_version,
            "gc_count_min_inclusive": policy.gc_min_count,
            "gc_count_max_inclusive": policy.gc_max_count,
            "gc_min_frac": policy.gc_min_frac,
            "gc_max_frac": policy.gc_max_frac,
            "bio_max_homopolymer_run": policy.max_run,
            "nmi_average_method": "arithmetic",
            "sklearn_version": sklearn.__version__,
            }:
        raise SealRefused(
            f"{cell.name}: producer protocol is not the exact current "
            "extraction/evaluation/NMI protocol")

    # Reopen every producer and extraction once more immediately before the
    # atomic publish. This prevents a replacement after the first check from
    # being blessed by the final marker.
    frozen = {
        "raw": (raw, raw_artifact, "evaluation_siglip2_base.json"),
        "bio_projected": (
            post, post_artifact, "evaluation_siglip2_base_bioproj.json"),
        "cell_result": (evaluation, cell_artifact, "cell_result.json"),
        "pairwise_nmi": (nmi, nmi_artifact, "pairwise_nmi.json"),
    }
    for name, (old_payload, old_artifact, filename) in frozen.items():
        new_payload, new_artifact = _read(cell, filename)
        if new_payload != old_payload or new_artifact != old_artifact:
            raise SealRefused(
                f"{cell.name}: {name} changed while the seal was assembled")
    binding_final = metric_input_binding(
        str(cell), required_splits=required_splits,
        allow_backfilled=allow_backfilled)
    final_bases, final_npz_artifacts = _validated_npz_artifacts(
        str(cell), required_splits=required_splits, binding=binding_final)
    if binding_final != binding or final_bases != actual_bases \
            or final_npz_artifacts != npz_artifacts:
        raise SealRefused(
            f"{cell.name}: extraction changed while the seal was assembled")

    sources = analysis_source_digests()
    try:
        # Use the shared writer in an isolated staging directory to apply its
        # exact metric/protocol/source schema checks without ever exposing an
        # intermediate marker lacking the standalone authority declaration.
        with tempfile.TemporaryDirectory(
                prefix=".analysis-seal-stage.", dir=str(cell.parent)) as stage:
            staged = write_analysis_marker(
                stage, metrics=metrics, binding=binding, protocol=protocol,
                sources=sources)
            marker, _ = _read_json_artifact(staged)
    except (ExtractionInvalid, RuntimeError) as error:
        raise SealRefused(str(error)) from None

    # This stage proves byte integrity and mutual consistency only. It does not
    # independently recompute retrieval from arrays, so re-signing hand-written
    # finite JSON cannot become scientific authority. The Phase-3 exact-receipt
    # verifier supplies that authority by deterministic recomputation.
    marker.update({
        "artifact_kind": "standalone_analysis_integrity_marker",
        "scientific_authority": False,
        "paper_result_eligible": False,
        "standalone_authority": {
            "schema_version": 1,
            "scope": "integrity_only_no_independent_metric_recomputation",
            "scientific_authority": False,
            "paper_result_eligible": False,
            "requires_downstream_deterministic_recomputation": True,
        },
        "evidence_artifacts": {
            name: artifact for name, (_, artifact, _) in frozen.items()
        },
        "extraction_evidence": {
            "input_binding": binding,
            "npz_artifacts": npz_artifacts,
            "manifest_sha256": run.manifest_sha256,
            "completion_marker_sha256": run.completion_marker_sha256,
        },
        "sealer_source": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(str(Path(__file__).resolve())),
        },
    })
    output = cell / ANALYSIS_MARKER_NAME
    _atomic_json(output, marker)
    return str(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True, nargs="+",
                        help="cell directories to seal")
    parser.add_argument("--allow-backfilled", action="store_true",
                        help=("Admit retrospectively bound inputs. Off by "
                              "default so a paper path cannot pick them up."))
    parser.add_argument(
        "--require-train", action="store_true",
        help=("Require one completion marker over db/query/train. Mandatory "
              "for MAIN refits; off for two-split diagnostics."))
    args = parser.parse_args()
    required_splits = ("db", "query", "train") if args.require_train else (
        "db", "query")

    refused = []
    for raw in args.dir:
        cell = Path(raw)
        # A stale seal must not survive a refusal: it would keep describing
        # numbers that this run has just shown to be unpublishable.
        marker = cell / ANALYSIS_MARKER_NAME
        try:
            path = seal(
                cell, allow_backfilled=args.allow_backfilled,
                required_splits=required_splits)
        except SealRefused as error:
            if marker.exists():
                marker.unlink()
            refused.append(str(error))
            print(f"[seal] REFUSED {cell.name}: {error}")
            continue
        print(f"[seal] {cell.name} -> {path}")

    if refused:
        print(f"\n{len(refused)} of {len(args.dir)} cell(s) refused")
    return 1 if refused else 0


if __name__ == "__main__":
    raise SystemExit(main())
