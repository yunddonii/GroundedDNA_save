"""Emit the main-table baseline rows from an aggregate JSON, or refuse (F15).

The main table was assembled by hand from whatever aggregation happened to be
open, so a number could reach the paper without the matrix that produced it ever
having been admissible. This generator closes that path:

  * the ONLY input is an aggregate JSON emitted by
    `aggregate_baseline_p0_matrix.py`, so every cell carries its manifest path,
    its implementation fingerprint and its own eligibility flag;
  * `--require-paper-eligible` (the default) independently reconstructs the
    D6 record set, per-cell eligibility, aggregate mean/sample-standard-
    deviation, summary counts and admission state. The mutable top-level
    `paper_table_admission.eligible` flag is only cross-checked, never trusted;
  * the producer script path and current source SHA-256 must match the seal in
    the aggregate. A diagnostic-only or stale-producer matrix emits no TeX;
  * the mean comes from `mean_map_at_R_post`, which the aggregator leaves null
    for a diagnostic-only cell. `diagnostic_mean_map_at_R_post` is never read
    here, so a diagnostic number cannot be typeset by accident.

Refusal is the point. Passing `--allow-diagnostic` stamps every emitted row and
the file header as diagnostic-only, which is meant to be visible in the diff.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import pickle
import re
import statistics
import sys
from typing import Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dna_utils.bit_slice import resolve_bit_slice  # noqa: E402
from dna_utils.flat_geometry import resolve_flat_geometry  # noqa: E402
from baseline.cache_provenance import (  # noqa: E402
    DECODE_FAILURE_AUDIT_SCHEMA,
    DECODE_FAILURE_AUDIT_SCHEMA_VERSION,
    MSCOCO_EXPECTED_FAILED_IMAGE_IDS,
    MSCOCO_EXPECTED_FAILED_IMAGE_IDS_SHA256,
    MSCOCO_EXPECTED_FAILED_IMAGE_INDICES,
    MSCOCO_EXPECTED_FAILED_INDICES_SHA256,
)
from baseline.execution_environment import (  # noqa: E402
    ExecutionEnvironmentError,
    require_exact_environment,
    verify_execution_environment,
)

#: Row order in the paper. Anything absent from the aggregate prints as `-`.
ROW_ORDER = (
    "cibhash", "cimon", "mls3rduh", "greedyhash", "bihalf",
    "sdc-paper", "oh", "hhch", "crovca",
)
DISPLAY = {
    "cibhash": "CIBHash", "cimon": "CIMON", "mls3rduh": "MLS3RDUH",
    "greedyhash": "GreedyHash", "bihalf": "Bi-Half", "sdc-paper": "SDC",
    "oh": "OH", "hhch": "HHCH", "crovca": "CroVCA",
    "crh-supervised": "CRH (supervised)",
}
COLUMN_ORDER = ("Flickr25k", "MSCOCO", "NUSWIDE", "CIFAR10")
COLUMN_LABEL = {
    "Flickr25k": "Flickr25K", "MSCOCO": "MS-COCO",
    "NUSWIDE": "NUS-WIDE", "CIFAR10": "CIFAR-10",
}
CURRENT_AGGREGATE_SCHEMA_VERSION = 8
D6_PROTOCOL_MODE = "author_fixed_final"
D6_CHECKPOINT_POLICY = "author_horizon_last"
D6_PROTOCOL_STAGE = "author_fixed_single_stage"
D6_BIT = 30
MIN_UNIQUE_TRAIN_SEEDS = 3
AGGREGATOR_SOURCE_RELATIVE = "scripts/aggregate_baseline_p0_matrix.py"
AGGREGATOR_SOURCE = REPO / AGGREGATOR_SOURCE_RELATIVE
AGGREGATE_PRODUCER_SOURCE_RELATIVE_PATHS = (
    AGGREGATOR_SOURCE_RELATIVE,
    "scripts/run_modern_baseline_p0.py",
    "scripts/run_baseline_p0_matrix.py",
    "baseline/cache_provenance.py",
    "baseline/execution_environment.py",
    "dna_utils/runtime_environment.py",
    "dna_utils/gpu_lease.py",
    "dna_utils/bit_slice.py",
    "dna_utils/flat_geometry.py",
)
AGGREGATE_PRODUCER_SOURCE_FILES = {
    relative: REPO / relative
    for relative in AGGREGATE_PRODUCER_SOURCE_RELATIVE_PATHS
}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
# Aggregate values are JSON round-trips of statistics.mean/stdev over the same
# IEEE-754 inputs.  1e-12 catches edited table values while allowing harmless
# last-bit differences across supported Python builds.
FLOAT_REL_TOL = 1e-12
FLOAT_ABS_TOL = 1e-12
BIO_PROJECTION_TIE_POLICY = (
    "dp_base_hamming_v1:equal_cost_keeps_first_predecessor_under_"
    "prev_base_ACGT_then_run_asc_then_gc_asc;terminal_base_ACGT_"
    "then_run_asc_then_gc_asc"
)
BIO_RANKING_TIE_POLICY = "stable_database_order_numpy_stable_v1"
EMPTY_FAILED_INDICES_SHA256 = (
    "549f2a8cf4d1265d8dc1b2c41ca2b1cc51066ea94ffa209cdb95e6daab449592")
EMPTY_FAILED_IMAGE_IDS_SHA256 = (
    "8f8d5359ce539f79cbb23ab871b15fc57328454cd0ea196d27c81997cc31ae61")

PANEL_VARIANTS = {
    "u0": ROW_ORDER,
    "u2": ("umrch",),
    "supervised": ("crh-supervised",),
}
PANEL_DATASETS = {
    "u0": COLUMN_ORDER,
    "u2": ("Flickr25k", "MSCOCO", "NUSWIDE"),
    "supervised": COLUMN_ORDER,
}


class NotPaperEligible(RuntimeError):
    """The aggregate says its own numbers may not enter the paper table."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_closure_digest(source_sha256: Mapping[str, str]) -> str:
    encoded = json.dumps(
        dict(source_sha256), sort_keys=True,
        separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _strict_equal(actual: object, expected: object) -> bool:
    """JSON equality without Python's bool/int and int/float coercions."""
    if expected is None:
        return actual is None
    if isinstance(expected, bool):
        return actual is expected
    if isinstance(expected, int):
        return type(actual) is int and actual == expected
    if isinstance(expected, float):
        return type(actual) is float and actual == expected
    if isinstance(expected, str):
        return isinstance(actual, str) and actual == expected
    if isinstance(expected, (list, tuple)):
        return (
            isinstance(actual, type(expected))
            and len(actual) == len(expected)
            and all(_strict_equal(left, right)
                    for left, right in zip(actual, expected))
        )
    if isinstance(expected, Mapping):
        return (
            isinstance(actual, Mapping)
            and set(actual) == set(expected)
            and all(_strict_equal(actual[key], expected[key])
                    for key in expected)
        )
    return type(actual) is type(expected) and actual == expected


def _record_key(panel: str, variant: str, dataset: str,
                seed: int) -> str:
    return f"{panel}/{variant}/{dataset}/{D6_BIT}b/seed{seed}"


def _expected_record_identities(
        panels: Sequence[str], seeds: Sequence[int],
        ) -> dict[str, tuple[str, str, str, int, int]]:
    expected: dict[str, tuple[str, str, str, int, int]] = {}
    for panel in panels:
        for variant in PANEL_VARIANTS[panel]:
            for dataset in PANEL_DATASETS[panel]:
                for seed in seeds:
                    key = _record_key(panel, variant, dataset, seed)
                    expected[key] = (panel, variant, dataset, D6_BIT, seed)
    return expected


def _expected_aggregate_identities(
        panels: Sequence[str],
        ) -> dict[tuple[str, str, str, int], tuple[str, str, str, int]]:
    return {
        (panel, variant, dataset, D6_BIT):
        (panel, variant, dataset, D6_BIT)
        for panel in panels
        for variant in PANEL_VARIANTS[panel]
        for dataset in PANEL_DATASETS[panel]
    }


def _same_float(actual: object, expected: float) -> bool:
    return bool(
        _is_finite_number(actual)
        and math.isclose(
            float(actual), expected,
            rel_tol=FLOAT_REL_TOL, abs_tol=FLOAT_ABS_TOL)
    )


def _decode_audit_contract_failures(
        audit: object, *, dataset: str) -> list[str]:
    """Validate the immutable exact decode-failure policy in each D6 record."""
    if not isinstance(audit, Mapping):
        return ["cache_decode_failure_audit is missing"]
    failures: list[str] = []
    exact_common = {
        "schema": DECODE_FAILURE_AUDIT_SCHEMA,
        "schema_version": DECODE_FAILURE_AUDIT_SCHEMA_VERSION,
        "status": "verified",
        "dataset": dataset,
        "setting": "setting1",
        "official_train_query_overlap_count": 0,
    }
    for field, expected in exact_common.items():
        if not _strict_equal(audit.get(field), expected):
            failures.append(
                f"cache_decode_failure_audit.{field} is not {expected!r}")
    cache_dir = audit.get("cache_dir")
    if (not isinstance(cache_dir, str) or not cache_dir
            or not Path(cache_dir).expanduser().is_absolute()):
        failures.append("cache_decode_failure_audit.cache_dir is not absolute")
    for field in ("cache_meta_sha256", "cache_image_ids_sha256"):
        digest = audit.get(field)
        if not isinstance(digest, str) \
                or SHA256_PATTERN.fullmatch(digest) is None:
            failures.append(f"cache_decode_failure_audit.{field} is invalid")

    if dataset == "MSCOCO":
        expected = {
            "source_encoding": "explicit_meta_fields",
            "policy": "kept_and_disclosed_db_only",
            "sensitivity_analysis": (
                "later exclusion sensitivity; main inputs keep these rows"),
            "count": 16,
            "failed_image_indices": list(
                MSCOCO_EXPECTED_FAILED_IMAGE_INDICES),
            "failed_indices_sha256": MSCOCO_EXPECTED_FAILED_INDICES_SHA256,
            "failed_image_ids": list(MSCOCO_EXPECTED_FAILED_IMAGE_IDS),
            "failed_image_ids_sha256": (
                MSCOCO_EXPECTED_FAILED_IMAGE_IDS_SHA256),
            "official_membership": {
                "train_count": 0,
                "query_count": 0,
                "database_count": 16,
                "database_only_count": 16,
                "unassigned_count": 0,
            },
        }
    else:
        expected = {
            "policy": "no_decode_failures",
            "sensitivity_analysis": "not_applicable",
            "count": 0,
            "failed_image_indices": [],
            "failed_indices_sha256": EMPTY_FAILED_INDICES_SHA256,
            "failed_image_ids": [],
            "failed_image_ids_sha256": EMPTY_FAILED_IMAGE_IDS_SHA256,
            "official_membership": {
                "train_count": 0,
                "query_count": 0,
                "database_count": 0,
                "database_only_count": 0,
                "unassigned_count": 0,
            },
        }
        source_encoding = audit.get("source_encoding")
        allowed_encoding = (
            {"absent_means_zero_legacy_cifar_extractor",
             "explicit_meta_fields"}
            if dataset == "CIFAR10" else {"explicit_meta_fields"})
        if source_encoding not in allowed_encoding:
            failures.append(
                "cache_decode_failure_audit.source_encoding is invalid for "
                f"{dataset}")
    for field, expected_value in expected.items():
        if not _strict_equal(audit.get(field), expected_value):
            failures.append(
                f"cache_decode_failure_audit.{field} does not match the exact "
                f"{dataset} policy")

    membership_sources = audit.get("membership_source_sha256")
    expected_source_paths = (
        set() if dataset == "CIFAR10" else {
            f"{dataset}/setting1/{name}.txt"
            for name in ("train", "test", "database")
        })
    if (not isinstance(membership_sources, Mapping)
            or set(membership_sources) != expected_source_paths
            or any(not isinstance(digest, str)
                   or SHA256_PATTERN.fullmatch(digest) is None
                   for digest in membership_sources.values())):
        failures.append(
            "cache_decode_failure_audit.membership_source_sha256 path/digest "
            "set is invalid")
    return failures


def _resolve_manifest_path(raw: str, *, source: Path | None) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()
    parent = source.parent if source is not None else REPO
    return (parent / path).resolve()


def _paper_payload_invariant_failures(
        payload: Mapping[str, object], *, source: Path | None = None,
        ) -> list[str]:
    """Rebuild D6 paper admission from immutable record-level evidence.

    No top-level boolean or precomputed mean participates in the decision.  A
    diagnostic caller may still render these payloads, but strict paper output
    is admitted only when this independent reconstruction has no failures.
    """
    failures: list[str] = []

    if not _strict_equal(
            payload.get("schema_version"), CURRENT_AGGREGATE_SCHEMA_VERSION):
        failures.append(
            "schema_version is not current D6 aggregate schema "
            f"{CURRENT_AGGREGATE_SCHEMA_VERSION}")

    producer = payload.get("producer")
    if not isinstance(producer, Mapping):
        failures.append("producer source seal is missing")
    else:
        if producer.get("source_path") != AGGREGATOR_SOURCE_RELATIVE:
            failures.append("producer.source_path is not the audited aggregator")
        declared_source_sha = producer.get("source_sha256")
        current_source_sha = _sha256(AGGREGATOR_SOURCE)
        if declared_source_sha != current_source_sha:
            failures.append(
                "producer.source_sha256 does not match the current audited "
                "aggregator source")
        declared_closure = producer.get("source_closure_sha256")
        expected_paths = set(AGGREGATE_PRODUCER_SOURCE_RELATIVE_PATHS)
        if (not isinstance(declared_closure, Mapping)
                or set(declared_closure) != expected_paths
                or any(not isinstance(digest, str)
                       or SHA256_PATTERN.fullmatch(digest) is None
                       for digest in declared_closure.values())):
            failures.append(
                "producer.source_closure_sha256 has a missing, unexpected, "
                "or invalid source entry")
        else:
            current_closure = {
                relative: _sha256(AGGREGATE_PRODUCER_SOURCE_FILES[relative])
                for relative in AGGREGATE_PRODUCER_SOURCE_RELATIVE_PATHS
            }
            if not _strict_equal(declared_closure, current_closure):
                failures.append(
                    "producer.source_closure_sha256 does not match all current "
                    "aggregate validator sources")
            closure_digest = producer.get("source_closure_digest_sha256")
            declared_digest = _source_closure_digest(dict(declared_closure))
            current_digest = _source_closure_digest(current_closure)
            if (closure_digest != declared_digest
                    or closure_digest != current_digest):
                failures.append(
                    "producer.source_closure_digest_sha256 does not bind the "
                    "declared and current source closure")
            if declared_closure.get(
                    AGGREGATOR_SOURCE_RELATIVE) != declared_source_sha:
                failures.append(
                    "producer source_sha256 disagrees with its source closure")

    protocol = payload.get("protocol")
    if not isinstance(protocol, Mapping):
        failures.append("protocol block is missing")
        return failures
    if protocol.get("protocol_mode") != D6_PROTOCOL_MODE:
        failures.append("protocol_mode is not author_fixed_final")
    if protocol.get("checkpoint_policy") != D6_CHECKPOINT_POLICY:
        failures.append("checkpoint_policy is not author_horizon_last")
    if protocol.get("final_checkpoint_protocol_stage") != D6_PROTOCOL_STAGE:
        failures.append(
            "final_checkpoint_protocol_stage is not author_fixed_single_stage")
    if not _strict_equal(protocol.get("requested_bit_slice"), [D6_BIT]):
        failures.append("requested_bit_slice is not exactly [30]")
    if protocol.get("mandatory_bio_projection") is not True:
        failures.append("mandatory_bio_projection is not true")
    if protocol.get("validation_seed") is not None:
        failures.append("author_fixed_final validation_seed is not null")
    if not _strict_equal(protocol.get("validation_ratio"), 0.0):
        failures.append("author_fixed_final validation_ratio is not 0.0")
    if protocol.get("selection_metric_by_bit") is not None:
        failures.append("author_fixed_final selection_metric_by_bit is not null")

    raw_seeds = protocol.get("train_seeds")
    seeds: list[int] = []
    if (not isinstance(raw_seeds, list)
            or any(isinstance(seed, bool) or not isinstance(seed, int)
                   for seed in raw_seeds)):
        failures.append("train_seeds is not an integer list")
    else:
        seeds = list(raw_seeds)
        if len(seeds) != len(set(seeds)):
            failures.append("train_seeds contains duplicates")
        if len(seeds) < MIN_UNIQUE_TRAIN_SEEDS:
            failures.append("fewer_than_three_train_seeds")
    if protocol.get("three_seed_mean_std_status") != "candidate_aggregate":
        failures.append("three_seed_mean_std_status is not candidate_aggregate")

    raw_panels = protocol.get("comparison_panels")
    panels: list[str] = []
    if (not isinstance(raw_panels, list)
            or any(not isinstance(panel, str) for panel in raw_panels)):
        failures.append("comparison_panels is not a string list")
    else:
        panels = list(raw_panels)
        if not panels or len(panels) != len(set(panels)):
            failures.append("comparison_panels is empty or contains duplicates")
        unknown = sorted(set(panels) - set(PANEL_VARIANTS))
        if unknown:
            failures.append(f"comparison_panels contains unknown panels {unknown}")
        if "u0" not in panels:
            failures.append("comparison_panels does not contain required U0")
    if failures and (not seeds or not panels
                     or any(panel not in PANEL_VARIANTS for panel in panels)):
        return failures

    expected_records = _expected_record_identities(panels, seeds)
    expected_u0 = {
        key for key, identity in expected_records.items()
        if identity[0] == "u0"
    }
    raw_records = payload.get("records")
    records: dict[str, Mapping[str, object]] = {}
    duplicate_record_keys: set[str] = set()
    if not isinstance(raw_records, list):
        failures.append("records is not a list")
        raw_records = []
    for index, record in enumerate(raw_records):
        if not isinstance(record, Mapping):
            failures.append(f"records[{index}] is not an object")
            continue
        key = record.get("key")
        if not isinstance(key, str):
            failures.append(f"records[{index}].key is missing")
            continue
        if key in records:
            duplicate_record_keys.add(key)
            continue
        records[key] = record
    if duplicate_record_keys:
        failures.append(
            f"records contains duplicate keys {sorted(duplicate_record_keys)}")
    actual_record_keys = set(records)
    actual_u0 = {
        key for key, record in records.items()
        if record.get("panel") == "u0"
    }
    if actual_u0 != expected_u0:
        failures.append(
            "records U0 key set mismatch; "
            f"missing={len(expected_u0 - actual_u0)}, "
            f"unexpected={len(actual_u0 - expected_u0)}")
    if actual_record_keys != set(expected_records):
        failures.append(
            "records full key set mismatch; "
            f"missing={len(set(expected_records) - actual_record_keys)}, "
            f"unexpected={len(actual_record_keys - set(expected_records))}")

    manifest_paths: set[str] = set()
    bio_manifest_paths: set[str] = set()
    bio_artifact_paths: set[str] = set()
    reopened_run_manifests: dict[str, Mapping[str, object]] = {}
    reopened_run_paths: dict[str, Path] = {}
    for key, expected_identity in expected_records.items():
        record = records.get(key)
        if record is None:
            continue
        panel, variant, dataset, bit, seed = expected_identity
        actual_identity = (
            record.get("panel"), record.get("variant"),
            record.get("dataset"), record.get("bit"), record.get("seed"))
        if (not _strict_equal(actual_identity, expected_identity)
                or record.get("key") != key):
            failures.append(f"record {key} identity fields do not match its key")
        if record.get("protocol_mode") != D6_PROTOCOL_MODE:
            failures.append(f"record {key} protocol_mode is not author_fixed_final")
        if record.get("protocol_stage") != D6_PROTOCOL_STAGE:
            failures.append(
                f"record {key} protocol_stage is not author_fixed_single_stage")
        protocol_digest = record.get("protocol_digest_sha256")
        if (not isinstance(protocol_digest, str)
                or SHA256_PATTERN.fullmatch(protocol_digest) is None):
            failures.append(f"record {key} protocol_digest_sha256 is invalid")
        checkpoint_protocol = record.get("checkpoint_protocol")
        expected_checkpoint_protocol = {
            "protocol_mode": D6_PROTOCOL_MODE,
            "protocol_stage": D6_PROTOCOL_STAGE,
            "protocol_identity_sha256": protocol_digest,
        }
        if not _strict_equal(
                checkpoint_protocol, expected_checkpoint_protocol):
            failures.append(
                f"record {key} checkpoint_protocol does not match "
                "mode/stage/digest")
        final_checkpoint_sha = record.get("final_checkpoint_sha256")
        if (not isinstance(final_checkpoint_sha, str)
                or SHA256_PATTERN.fullmatch(final_checkpoint_sha) is None):
            failures.append(
                f"record {key} final_checkpoint_sha256 is invalid")
        if record.get("status") != "complete_main_eligible":
            failures.append(f"record {key} status is not complete_main_eligible")
        if record.get("declared_main_protocol_eligible") is not True:
            failures.append(
                f"record {key} declared_main_protocol_eligible is not true")
        if record.get("main_protocol_eligible") is not True:
            failures.append(f"record {key} main_protocol_eligible is not true")
        if record.get("implementation_comparison_eligible") is not True:
            failures.append(
                f"record {key} implementation_comparison_eligible is not true")
        if record.get("paper_table_eligible") is not True:
            failures.append(f"record {key} paper_table_eligible is not true")
        if record.get("legacy_cache_diagnostic_only") is not False:
            failures.append(
                f"record {key} legacy_cache_diagnostic_only is not false")
        if record.get("validation_errors") != []:
            failures.append(f"record {key} validation_errors is not empty")
        if not _is_finite_number(record.get("map_at_R_post")):
            failures.append(f"record {key} map_at_R_post is not finite")
        record_decode_audit = record.get("cache_decode_failure_audit")
        failures.extend(
            f"record {key} {failure}"
            for failure in _decode_audit_contract_failures(
                record_decode_audit, dataset=dataset))
        record_environment: dict[str, object] | None = None
        record_environment_sha256 = record.get(
            "execution_environment_sha256")
        try:
            record_environment = verify_execution_environment(
                record.get("execution_environment"),
                record_environment_sha256)
        except ExecutionEnvironmentError as error:
            failures.append(
                f"record {key} execution environment invalid: {error}")

        manifest = record.get("manifest")
        manifest_sha = record.get("manifest_sha256")
        if not isinstance(manifest, str) or not manifest:
            failures.append(f"record {key} manifest path is missing")
        elif manifest in manifest_paths:
            failures.append(f"record {key} reuses another record manifest path")
        else:
            manifest_paths.add(manifest)
            if (not isinstance(manifest_sha, str)
                    or SHA256_PATTERN.fullmatch(manifest_sha) is None):
                failures.append(f"record {key} manifest_sha256 is invalid")
            else:
                try:
                    manifest_path = _resolve_manifest_path(
                        manifest, source=source)
                    if source is not None and not manifest_path.is_file():
                        failures.append(
                            f"record {key} manifest cannot be reopened")
                    elif manifest_path.is_file():
                        if _sha256(manifest_path) != manifest_sha:
                            failures.append(
                                f"record {key} manifest content SHA-256 mismatch")
                        else:
                            reopened = json.loads(
                                manifest_path.read_text(encoding="utf-8"))
                            if not isinstance(reopened, Mapping):
                                failures.append(
                                    f"record {key} manifest root is not an object")
                            else:
                                reopened_run_manifests[key] = reopened
                                reopened_run_paths[key] = manifest_path
                                expected_manifest_fields = {
                                    "protocol_mode": D6_PROTOCOL_MODE,
                                    "training_stage": D6_PROTOCOL_STAGE,
                                    "protocol_digest_sha256": protocol_digest,
                                    "checkpoint_protocol": (
                                        expected_checkpoint_protocol),
                                    "final_checkpoint_sha256": (
                                        final_checkpoint_sha),
                                    "cache_decode_failure_audit": (
                                        record_decode_audit),
                                    "execution_environment": (
                                        record_environment),
                                    "execution_environment_sha256": (
                                        record_environment_sha256),
                                }
                                for field, expected in expected_manifest_fields.items():
                                    if not _strict_equal(
                                            reopened.get(field), expected):
                                        failures.append(
                                            f"record {key} reopened manifest "
                                            f"{field} disagrees with aggregate record")
                                identity_payload = reopened.get(
                                    "protocol_identity")
                                if not isinstance(identity_payload, Mapping):
                                    failures.append(
                                        f"record {key} reopened manifest lacks "
                                        "protocol_identity")
                                else:
                                    expected_protocol_identity = {
                                        "variant": expected_identity[1],
                                        "dataset": expected_identity[2],
                                        "bit": expected_identity[3],
                                        "seed": expected_identity[4],
                                    }
                                    for field, expected in (
                                            expected_protocol_identity.items()):
                                        if not _strict_equal(
                                                identity_payload.get(field), expected):
                                            failures.append(
                                                f"record {key} reopened manifest "
                                                f"protocol_identity.{field} mismatch")
                                    if not _strict_equal(
                                            identity_payload.get(
                                                "cache_decode_failure_audit"),
                                            record_decode_audit):
                                        failures.append(
                                            f"record {key} reopened manifest "
                                            "protocol_identity cache decode audit "
                                            "mismatch")
                                    if not _strict_equal(
                                            identity_payload.get(
                                                "execution_environment"),
                                            record_environment) \
                                            or not _strict_equal(
                                                identity_payload.get(
                                                    "execution_environment_sha256"),
                                                record_environment_sha256):
                                        failures.append(
                                            f"record {key} reopened manifest "
                                            "protocol_identity execution "
                                            "environment mismatch")
                                    identity_digest = hashlib.sha256(json.dumps(
                                        identity_payload, sort_keys=True,
                                        separators=(",", ":"),
                                    ).encode("utf-8")).hexdigest()
                                    if identity_digest != protocol_digest:
                                        failures.append(
                                            f"record {key} reopened manifest "
                                            "protocol_identity digest mismatch")
                                raw_checkpoint = reopened.get("final_checkpoint")
                                if isinstance(raw_checkpoint, str):
                                    checkpoint_path = Path(
                                        raw_checkpoint).expanduser()
                                    if not checkpoint_path.is_absolute():
                                        checkpoint_path = (
                                            manifest_path.parent /
                                            checkpoint_path)
                                    checkpoint_path = checkpoint_path.resolve()
                                    try:
                                        import torch
                                        if not checkpoint_path.is_file() \
                                                or _sha256(checkpoint_path) \
                                                != final_checkpoint_sha:
                                            raise ValueError(
                                                "checkpoint content SHA-256 "
                                                "does not match record")
                                        checkpoint_payload = torch.load(
                                            checkpoint_path, map_location="cpu",
                                            weights_only=True)
                                        checkpoint_config = (
                                            checkpoint_payload.get("config")
                                            if isinstance(
                                                checkpoint_payload, Mapping)
                                            else None)
                                        if not isinstance(
                                                checkpoint_config, Mapping):
                                            raise ValueError(
                                                "checkpoint config missing")
                                        actual_checkpoint_protocol = {
                                            "protocol_mode": (
                                                checkpoint_config.get(
                                                    "protocol_mode")),
                                            "protocol_stage": (
                                                checkpoint_config.get(
                                                    "protocol_stage")),
                                            "protocol_identity_sha256": (
                                                checkpoint_config.get(
                                                    "protocol_identity_sha256")),
                                        }
                                        if not _strict_equal(
                                                actual_checkpoint_protocol,
                                                expected_checkpoint_protocol):
                                            raise ValueError(
                                                "checkpoint protocol contract "
                                                "does not match record")
                                        require_exact_environment(
                                            record_environment,
                                            record_environment_sha256,
                                            actual=checkpoint_config.get(
                                                "execution_environment"),
                                            actual_digest=checkpoint_config.get(
                                                "execution_environment_sha256"))
                                    except (OSError, RuntimeError, ValueError,
                                            EOFError, pickle.UnpicklingError,
                                            ExecutionEnvironmentError) as error:
                                        failures.append(
                                            f"record {key} checkpoint execution "
                                            f"environment mismatch: {error}")
                                else:
                                    failures.append(
                                        f"record {key} reopened manifest lacks "
                                        "final_checkpoint path")
                except (OSError, ValueError, json.JSONDecodeError):
                    failures.append(f"record {key} manifest path is invalid")

        bio_manifest = record.get("bio_projection_manifest")
        bio_manifest_sha = record.get("bio_projection_manifest_sha256")
        if not isinstance(bio_manifest, str) or not bio_manifest:
            failures.append(f"record {key} bio projection manifest path is missing")
        elif bio_manifest in bio_manifest_paths:
            failures.append(
                f"record {key} reuses another record bio projection manifest")
        else:
            bio_manifest_paths.add(bio_manifest)
            if (not isinstance(bio_manifest_sha, str)
                    or SHA256_PATTERN.fullmatch(bio_manifest_sha) is None):
                failures.append(
                    f"record {key} bio_projection_manifest_sha256 is invalid")
            else:
                try:
                    bio_path = _resolve_manifest_path(
                        bio_manifest, source=source)
                    if source is not None and not bio_path.is_file():
                        failures.append(
                            f"record {key} bio projection manifest cannot be reopened")
                    elif bio_path.is_file():
                        if _sha256(bio_path) != bio_manifest_sha:
                            failures.append(
                                f"record {key} bio projection manifest SHA-256 mismatch")
                        else:
                            bio_payload = json.loads(
                                bio_path.read_text(encoding="utf-8"))
                            config = (
                                bio_payload.get("config")
                                if isinstance(bio_payload, Mapping) else None)
                            cells = (
                                bio_payload.get("cells")
                                if isinstance(bio_payload, Mapping) else None)
                            if not isinstance(config, Mapping):
                                failures.append(
                                    f"record {key} bio config is missing")
                            else:
                                bio_config_expected = {
                                    "expected_length": expected_identity[3] // 2,
                                    "gc_min_frac": 0.4,
                                    "gc_max_frac": 0.6,
                                    "max_homopolymer_run": 3,
                                    "projection_tie_policy": (
                                        BIO_PROJECTION_TIE_POLICY),
                                    "ranking_tie_policy": BIO_RANKING_TIE_POLICY,
                                }
                                for field, expected in bio_config_expected.items():
                                    if not _strict_equal(config.get(field), expected):
                                        failures.append(
                                            f"record {key} bio config {field} mismatch")
                            if not isinstance(cells, list) or len(cells) != 1 \
                                    or not isinstance(cells[0], Mapping):
                                failures.append(
                                    f"record {key} bio cells is not one object")
                            else:
                                cell = cells[0]
                                bio_cell_expected = {
                                    "name": expected_identity[1],
                                    "dataset": expected_identity[2],
                                    "L": expected_identity[3] // 2,
                                    "paper_result_eligible": True,
                                    "projection_invariant_satisfied": True,
                                    "post_compliance_db": 1.0,
                                    "post_compliance_qy": 1.0,
                                    "projection_failures_db": 0,
                                    "projection_failures_qy": 0,
                                    "projection_tie_policy": (
                                        BIO_PROJECTION_TIE_POLICY),
                                    "ranking_tie_policy": BIO_RANKING_TIE_POLICY,
                                    "map_at_R_post": record.get("map_at_R_post"),
                                }
                                for field, expected in bio_cell_expected.items():
                                    if not _strict_equal(cell.get(field), expected):
                                        failures.append(
                                            f"record {key} bio cell {field} mismatch")
                                provenance = cell.get("training_provenance")
                                if not isinstance(provenance, Mapping):
                                    failures.append(
                                        f"record {key} bio training provenance missing")
                                else:
                                    for field, expected in {
                                            "protocol_mode": D6_PROTOCOL_MODE,
                                            "protocol_stage": D6_PROTOCOL_STAGE,
                                            "protocol_identity_sha256": (
                                                protocol_digest),
                                            "cache_decode_failure_audit": (
                                                record_decode_audit),
                                            "execution_environment": (
                                                record_environment),
                                            "execution_environment_sha256": (
                                                record_environment_sha256),
                                    }.items():
                                        if not _strict_equal(
                                                provenance.get(field), expected):
                                            failures.append(
                                                f"record {key} bio provenance "
                                                f"{field} mismatch")
                                    checkpoint = provenance.get("checkpoint")
                                    checkpoint_expected = {
                                        "sha256": final_checkpoint_sha,
                                        "checkpoint_protocol_mode": (
                                            D6_PROTOCOL_MODE),
                                        "checkpoint_protocol_stage": (
                                            D6_PROTOCOL_STAGE),
                                        "protocol_identity_sha256": (
                                            protocol_digest),
                                        "execution_environment": (
                                            record_environment),
                                        "execution_environment_sha256": (
                                            record_environment_sha256),
                                    }
                                    if not isinstance(checkpoint, Mapping):
                                        failures.append(
                                            f"record {key} bio checkpoint missing")
                                    else:
                                        for field, expected in (
                                                checkpoint_expected.items()):
                                            if not _strict_equal(
                                                    checkpoint.get(field), expected):
                                                failures.append(
                                                    f"record {key} bio checkpoint "
                                                    f"{field} mismatch")

                                run_manifest = reopened_run_manifests.get(key)
                                run_manifest_path = reopened_run_paths.get(key)
                                if run_manifest is not None:
                                    if (run_manifest.get(
                                            "bio_projection_manifest_sha256")
                                            != bio_manifest_sha):
                                        failures.append(
                                            f"record {key} run/bio manifest SHA mismatch")
                                    raw_run_bio = run_manifest.get(
                                        "bio_projection_manifest")
                                    if isinstance(raw_run_bio, str):
                                        run_bio_path = Path(raw_run_bio).expanduser()
                                        if not run_bio_path.is_absolute() \
                                                and run_manifest_path is not None:
                                            run_bio_path = (
                                                run_manifest_path.parent / run_bio_path)
                                        if run_bio_path.resolve() != bio_path.resolve():
                                            failures.append(
                                                f"record {key} run/bio manifest path mismatch")
                                    else:
                                        failures.append(
                                            f"record {key} run manifest lacks bio path")

                                input_records = cell.get("input_artifacts")
                                output_records = cell.get("output_artifacts")
                                run_input_hashes = (
                                    run_manifest.get("extraction_artifact_sha256")
                                    if run_manifest is not None else None)
                                run_outputs = (
                                    run_manifest.get("bio_projected_artifacts")
                                    if run_manifest is not None else None)
                                for group_name, records_value, run_value in (
                                        ("input", input_records, run_input_hashes),
                                        ("output", output_records, run_outputs)):
                                    if not isinstance(records_value, Mapping) \
                                            or set(records_value) != {"db", "query"}:
                                        failures.append(
                                            f"record {key} bio {group_name} artifacts invalid")
                                        continue
                                    for split in ("db", "query"):
                                        artifact = records_value.get(split)
                                        if not isinstance(artifact, Mapping):
                                            failures.append(
                                                f"record {key} bio {group_name} "
                                                f"{split} record invalid")
                                            continue
                                        raw_path = artifact.get("path")
                                        digest = artifact.get("sha256")
                                        if not isinstance(raw_path, str) \
                                                or not isinstance(digest, str) \
                                                or SHA256_PATTERN.fullmatch(digest) is None:
                                            failures.append(
                                                f"record {key} bio {group_name} "
                                                f"{split} binding invalid")
                                            continue
                                        artifact_path = Path(raw_path).expanduser()
                                        if not artifact_path.is_absolute():
                                            artifact_path = bio_path.parent / artifact_path
                                        artifact_path = artifact_path.resolve()
                                        artifact_key = str(artifact_path)
                                        if artifact_key in bio_artifact_paths:
                                            failures.append(
                                                f"record {key} reuses bio artifact "
                                                f"{artifact_path}")
                                        bio_artifact_paths.add(artifact_key)
                                        if not artifact_path.is_file() \
                                                or _sha256(artifact_path) != digest:
                                            failures.append(
                                                f"record {key} bio {group_name} "
                                                f"{split} artifact SHA mismatch")
                                        if group_name == "input":
                                            expected_run_sha = (
                                                run_value.get(
                                                    f"extract_{split}.npz")
                                                if isinstance(run_value, Mapping)
                                                else None)
                                        else:
                                            run_record = (
                                                run_value.get(split)
                                                if isinstance(run_value, Mapping)
                                                else None)
                                            expected_run_sha = (
                                                run_record.get("sha256")
                                                if isinstance(run_record, Mapping)
                                                else None)
                                        if expected_run_sha != digest:
                                            failures.append(
                                                f"record {key} run/bio {group_name} "
                                                f"{split} SHA mismatch")

                                if run_manifest is not None:
                                    raw_sidecar = run_manifest.get(
                                        "bio_projection_sidecar")
                                    sidecar_sha = run_manifest.get(
                                        "bio_projection_sidecar_sha256")
                                    if not isinstance(raw_sidecar, str) \
                                            or not isinstance(sidecar_sha, str) \
                                            or SHA256_PATTERN.fullmatch(
                                                sidecar_sha) is None:
                                        failures.append(
                                            f"record {key} bio sidecar binding invalid")
                                    else:
                                        sidecar = Path(raw_sidecar).expanduser()
                                        if not sidecar.is_absolute() \
                                                and run_manifest_path is not None:
                                            sidecar = run_manifest_path.parent / sidecar
                                        sidecar = sidecar.resolve()
                                        try:
                                            sidecar_token = sidecar.read_text(
                                                encoding="utf-8").strip().split()[0]
                                        except (OSError, IndexError):
                                            sidecar_token = None
                                        if not sidecar.is_file() \
                                                or _sha256(sidecar) != sidecar_sha \
                                                or sidecar_token != bio_manifest_sha:
                                            failures.append(
                                                f"record {key} bio sidecar SHA mismatch")
                except (OSError, ValueError, json.JSONDecodeError):
                    failures.append(
                        f"record {key} bio projection manifest path is invalid")

    expected_aggregates = _expected_aggregate_identities(panels)
    raw_aggregates = payload.get("aggregates")
    aggregates: dict[tuple[str, str, str, int], Mapping[str, object]] = {}
    duplicate_aggregates: set[tuple[str, str, str, int]] = set()
    if not isinstance(raw_aggregates, list):
        failures.append("aggregates is not a list")
        raw_aggregates = []
    for index, aggregate in enumerate(raw_aggregates):
        if not isinstance(aggregate, Mapping):
            failures.append(f"aggregates[{index}] is not an object")
            continue
        identity = (
            aggregate.get("panel"), aggregate.get("variant"),
            aggregate.get("dataset"), aggregate.get("bit"))
        if not all(isinstance(value, str) for value in identity[:3]) \
                or isinstance(identity[3], bool) \
                or not isinstance(identity[3], int):
            failures.append(f"aggregates[{index}] identity is invalid")
            continue
        typed_identity = identity  # narrowed by the checks above
        if typed_identity in aggregates:
            duplicate_aggregates.add(typed_identity)  # type: ignore[arg-type]
            continue
        aggregates[typed_identity] = aggregate  # type: ignore[index]
    if duplicate_aggregates:
        failures.append(
            f"aggregates contains duplicate identities {sorted(duplicate_aggregates)}")
    if set(aggregates) != set(expected_aggregates):
        failures.append(
            "aggregate key set mismatch; "
            f"missing={len(set(expected_aggregates) - set(aggregates))}, "
            f"unexpected={len(set(aggregates) - set(expected_aggregates))}")

    for aggregate_identity in expected_aggregates:
        aggregate = aggregates.get(aggregate_identity)
        if aggregate is None:
            continue
        panel, variant, dataset, bit = aggregate_identity
        record_keys = [
            _record_key(panel, variant, dataset, seed) for seed in seeds]
        selected = [records.get(key) for key in record_keys]
        if any(record is None for record in selected):
            continue
        values = [float(record["map_at_R_post"]) for record in selected
                  if record is not None
                  and _is_finite_number(record.get("map_at_R_post"))]
        if len(values) != len(seeds) or len(values) < 2:
            continue
        expected_mean = statistics.mean(values)
        expected_std = statistics.stdev(values)
        exact_fields = {
            "seeds_expected": seeds,
            "seeds_complete": seeds,
            "all_seeds_complete": True,
            "aggregate_status": "complete_paper_table_eligible",
            "all_records_main_protocol_eligible": True,
            "all_records_paper_table_eligible": True,
            "paper_table_eligible": True,
            "diagnostic_only": False,
            "records": record_keys,
            "diagnostic_mean_map_at_R_post": None,
            "diagnostic_sample_std_map_at_R_post": None,
        }
        for field, expected in exact_fields.items():
            if not _strict_equal(aggregate.get(field), expected):
                failures.append(
                    f"aggregate {aggregate_identity} {field} does not match records")
        for field, expected in (
                ("mean_map_at_R_post", expected_mean),
                ("sample_std_map_at_R_post", expected_std),
                ("admitted_comparison_mean_map_at_R_post", expected_mean),
                ("admitted_comparison_sample_std_map_at_R_post", expected_std)):
            if not _same_float(aggregate.get(field), expected):
                failures.append(
                    f"aggregate {aggregate_identity} {field} does not match "
                    f"record recomputation within rel={FLOAT_REL_TOL}, "
                    f"abs={FLOAT_ABS_TOL}")

    expected_count = len(expected_records)
    summary_expected = {
        "expected": expected_count,
        "complete": expected_count,
        "complete_main_eligible": expected_count,
        "complete_diagnostic_only": 0,
        "implementation_comparison_blocked": 0,
        "paper_table_eligible": expected_count,
        "missing": 0,
        "invalid": 0,
        "duplicate": 0,
        "malformed": 0,
        "extraneous": 0,
        "source_profile_excluded": 0,
    }
    summary = payload.get("summary")
    if not isinstance(summary, Mapping):
        failures.append("summary block is missing")
    else:
        for field, expected in summary_expected.items():
            if not _strict_equal(summary.get(field), expected):
                failures.append(
                    f"summary.{field} does not match record reconstruction")

    for field in (
            "missing_cells", "invalid_cells", "duplicate_cells",
            "malformed_manifests", "extraneous_manifests",
            "source_profile_excluded_manifests"):
        if payload.get(field) != []:
            failures.append(f"{field} is not empty")
    if payload.get("duplicate_manifests") != {}:
        failures.append("duplicate_manifests is not empty")

    implementation_audit = payload.get("implementation_audit")
    if not isinstance(implementation_audit, Mapping):
        failures.append("implementation_audit block is missing")
    else:
        if implementation_audit.get("comparison_safe") is not True:
            failures.append("implementation_audit.comparison_safe is not true")
        if implementation_audit.get("blocked_cells") != {}:
            failures.append("implementation_audit.blocked_cells is not empty")

    admission = payload.get("paper_table_admission")
    if not isinstance(admission, Mapping):
        failures.append("aggregate has no paper_table_admission block")
    else:
        admission_expected = {
            "eligible": True,
            "reason_codes": [],
            "required_unique_train_seeds": MIN_UNIQUE_TRAIN_SEEDS,
            "observed_unique_train_seeds": len(seeds),
            "eligible_records": expected_count,
            "expected_records": expected_count,
        }
        for field, expected in admission_expected.items():
            if not _strict_equal(admission.get(field), expected):
                failures.append(
                    f"paper_table_admission.{field} does not match "
                    "record reconstruction")
    return failures


def admission_failure_reasons(
        payload: Mapping[str, object], *, source: Path | None = None,
        ) -> list[str]:
    """Return independently reconstructed reasons; never trust `eligible`."""
    reasons: list[str] = []
    admission = payload.get("paper_table_admission")
    if isinstance(admission, Mapping) and admission.get("eligible") is not True:
        raw = admission.get("reason_codes")
        if isinstance(raw, (list, tuple)):
            reasons.extend(str(code) for code in raw)
    reasons.extend(_paper_payload_invariant_failures(payload, source=source))
    return list(dict.fromkeys(reasons))


def _cell(entry: Mapping[str, object], *, allow_diagnostic: bool) -> str:
    mean = entry.get("mean_map_at_R_post")
    std = entry.get("sample_std_map_at_R_post")
    if mean is None and allow_diagnostic:
        mean = entry.get("diagnostic_mean_map_at_R_post")
        std = entry.get("diagnostic_sample_std_map_at_R_post")
    if mean is None:
        return "-"
    if std is None:
        return f"{float(mean):.4f}"
    return f"{float(mean):.4f} \\pm {float(std):.4f}"


def render_tex(payload: Mapping[str, object], *, source: Path,
               allow_diagnostic: bool = False) -> str:
    """The baseline block of the main table, plus its provenance header."""
    reasons = admission_failure_reasons(payload, source=source)
    if reasons and not allow_diagnostic:
        raise NotPaperEligible("; ".join(reasons))

    protocol = payload.get("protocol")
    protocol = protocol if isinstance(protocol, Mapping) else {}
    raw_slice = protocol.get("requested_bit_slice")
    bits = resolve_bit_slice(
        raw_slice if isinstance(raw_slice, (list, tuple)) else None)
    seeds = protocol.get("train_seeds")
    seeds = list(seeds) if isinstance(seeds, (list, tuple)) else []

    aggregates = payload.get("aggregates")
    aggregates = aggregates if isinstance(aggregates, list) else []
    by_cell: dict[tuple[int, str, str], Mapping[str, object]] = {}
    for entry in aggregates:
        if not isinstance(entry, Mapping):
            continue
        try:
            key = (int(entry["bit"]), str(entry["variant"]),
                   str(entry["dataset"]))
        except (KeyError, TypeError, ValueError):
            continue
        by_cell[key] = entry

    lines: list[str] = [
        "% GENERATED by scripts/build_maintable_from_aggregate.py -- do not edit.",
        f"% source: {source}",
        f"% source_sha256: {_sha256(source)}",
        f"% train_seeds: {seeds}",
        f"% generated_from_field: mean_map_at_R_post"
        f"{' (DIAGNOSTIC fallback allowed)' if allow_diagnostic else ''}",
    ]
    if reasons:
        lines.append("% DIAGNOSTIC-ONLY, NOT PAPER-ELIGIBLE: " + "; ".join(reasons))

    for bit in bits:
        geom = resolve_flat_geometry(bit)
        lines += [
            "",
            f"% ---- {bit} bit / {geom.total_bases} bases "
            f"({geom.num_codebooks}x{geom.bases_per_codebook}) ----",
        ]
        for variant in ROW_ORDER:
            cells = [
                _cell(by_cell[(bit, variant, dataset)],
                      allow_diagnostic=allow_diagnostic)
                if (bit, variant, dataset) in by_cell else "-"
                for dataset in COLUMN_ORDER
            ]
            label = DISPLAY.get(variant, variant)
            if reasons:
                label += r"$^{\dagger}$"
            lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Emit main-table baseline rows from an aggregate JSON.")
    parser.add_argument("aggregate", help="JSON from aggregate_baseline_p0_matrix.py")
    parser.add_argument("--out", default=None, help="TeX path; stdout if omitted.")
    parser.add_argument(
        "--require-paper-eligible", action=argparse.BooleanOptionalAction,
        default=True,
        help=("Refuse to emit anything unless the aggregate admits itself to "
              "the paper table (default true)."))
    parser.add_argument(
        "--allow-diagnostic", action="store_true",
        help=("Emit diagnostic-only numbers, stamping the header and every row "
              "with a dagger. Implies --no-require-paper-eligible."))
    args = parser.parse_args(argv)

    source = Path(args.aggregate).resolve()
    raw_payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw_payload, Mapping):
        print("REFUSED: aggregate JSON root is not an object.", file=sys.stderr)
        print("No TeX was written.", file=sys.stderr)
        return 3
    payload = raw_payload
    allow_diagnostic = bool(args.allow_diagnostic)
    if args.require_paper_eligible and not allow_diagnostic:
        reasons = admission_failure_reasons(payload, source=source)
        if reasons:
            print("REFUSED: the aggregate is not paper-table eligible.",
                  file=sys.stderr)
            for reason in reasons:
                print(f"  - {reason}", file=sys.stderr)
            print("No TeX was written. Re-run the matrix, or pass "
                  "--allow-diagnostic to emit dagger-stamped rows.",
                  file=sys.stderr)
            return 3

    tex = render_tex(payload, source=source, allow_diagnostic=allow_diagnostic)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(tex, encoding="utf-8")
        print(f"wrote {out}")
    else:
        print(tex, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
