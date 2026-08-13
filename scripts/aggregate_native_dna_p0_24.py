#!/usr/bin/env python3
"""Fail-closed aggregation for the sealed 24-base native-DNA P0 matrix.

The implementation reuses the mature 18-base aggregation machinery through a
versioned validation adapter.  It accepts only manifests and evaluations that
explicitly bind the 24-base pipeline, exact inclusive GC count range [10, 14],
and (for PRIMO) the frozen 80-to-24-nt predictor transfer.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import threading
from typing import Iterator, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import aggregate_native_dna_p0 as canonical  # noqa: E402
from scripts.run_native_dna_p0_24 import (  # noqa: E402
    GC_COUNT_MAX,
    GC_COUNT_MIN,
    MATCHED_LENGTH,
    PIPELINE_VARIANT,
    PRIMO_TRANSFER_BLOCKER,
)


from dna_utils.native_protocol import (  # noqa: E402
    NativeProtocol, coerce_protocol, resolve_native_protocol)

# F18: the sealed 24-base contract, as a value. The canonical validators now
# take it as an argument, so this module no longer has to relabel a 24-base
# evaluation as 18 to get past a length check that was pinned at import time.
PROTOCOL = resolve_native_protocol(MATCHED_LENGTH)
LENGTH = MATCHED_LENGTH
DEFAULT_SEEDS = canonical.DEFAULT_SEEDS
DATASETS = canonical.DATASETS
METHODS = canonical.METHODS
PANELS = canonical.PANELS
Key = canonical.Key
TEST_ACCESS_CONTRACT = canonical.TEST_ACCESS_CONTRACT
MANIFEST_NAME = canonical.MANIFEST_NAME
PROTOCOL_LABEL = "matched_24nt_adaptation"
COMMON_BIO_PROJECTION = {
    "algorithm": "minimum_hamming_dynamic_programming",
    "applied_to": ["query", "database"],
    "gc_min": 0.4,
    "gc_max": 0.6,
    "max_run": 3,
    "gc_count_min_inclusive": GC_COUNT_MIN,
    "gc_count_max_inclusive": GC_COUNT_MAX,
    "sequence_length_bases": LENGTH,
}
DISPLAY = dict(canonical.DISPLAY)
DISPLAY["bee2021"] = "PRIMO-24 frozen-predictor length-transfer"
# F18: one table, keyed by length, so this module and the canonical one cannot
# drift. The digests are unchanged; they now live beside 15 and 18.
METHOD_PROTOCOL_LOCK_SHA256 = canonical.method_protocol_lock_for(PROTOCOL)

_ORIGINAL_VALIDATE_EVALUATION = canonical._validate_evaluation
_ORIGINAL_VALIDATE_PROTOCOL = canonical._validate_protocol_identity
_ORIGINAL_VALIDATE_MANIFEST = canonical._validate_manifest
_LOCK = threading.RLock()


def _mapping(value: object) -> Mapping[str, object] | None:
    return value if isinstance(value, Mapping) else None


def _validate_evaluation_24(
    evaluation: Mapping[str, object],
    key: Key,
    errors: list[str],
    protocol: "NativeProtocol | int | None" = None,
):
    protocol = PROTOCOL if protocol is None else coerce_protocol(protocol)
    if evaluation.get("protocol") != PROTOCOL_LABEL:
        errors.append(
            f"evaluation.protocol: expected {PROTOCOL_LABEL!r}, "
            f"found {evaluation.get('protocol')!r}"
        )
    if evaluation.get("matched_capacity_bases") != LENGTH:
        errors.append(
            "evaluation.matched_capacity_bases: expected "
            f"{LENGTH}, found {evaluation.get('matched_capacity_bases')!r}"
        )
    if evaluation.get("common_dp_gc_count_range") != [
        GC_COUNT_MIN,
        GC_COUNT_MAX,
    ]:
        errors.append(
            "evaluation.common_dp_gc_count_range: expected "
            f"[{GC_COUNT_MIN}, {GC_COUNT_MAX}]"
        )
    # Previously this rewrote `protocol` to `matched_18nt_adaptation` so the
    # canonical validator -- whose label was fixed at import -- would accept a
    # 24-base evaluation. It now judges the artefact as what it is.
    return _ORIGINAL_VALIDATE_EVALUATION(evaluation, key, errors, protocol)


def _validate_protocol_identity_24(
    manifest: Mapping[str, object],
    key: Key,
    errors: list[str],
    protocol: "NativeProtocol | int | None" = None,
):
    protocol = PROTOCOL if protocol is None else coerce_protocol(protocol)
    declared, family = _ORIGINAL_VALIDATE_PROTOCOL(
        manifest, key, errors, protocol)
    identity = _mapping(manifest.get("protocol_identity"))
    if identity is None:
        return declared, family
    if identity.get("pipeline_variant") != PIPELINE_VARIANT:
        errors.append(
            "protocol_identity.pipeline_variant: expected "
            f"{PIPELINE_VARIANT!r}, found {identity.get('pipeline_variant')!r}"
        )
    if key.method == "bee2021":
        expected = {
            "source_predictor_length_bases": 80,
            "target_length_bases": LENGTH,
            "predictor_frozen": True,
            "alternating_predictor_refit": False,
            "eligibility_blocker": PRIMO_TRANSFER_BLOCKER,
        }
        if identity.get("primo_frozen_predictor_length_transfer") != expected:
            errors.append(
                "protocol_identity.primo_frozen_predictor_length_transfer: "
                "does not bind the frozen 80-to-24-nt transfer"
            )
    elif identity.get("primo_frozen_predictor_length_transfer") is not None:
        errors.append(
            "protocol_identity.primo_frozen_predictor_length_transfer: "
            "must be absent outside PRIMO"
        )
    return declared, family


def _validate_manifest_24_impl(
    path: Path,
    key: Key,
    *,
    verify_hashes: bool = True,
    protocol: "NativeProtocol | int | None" = None,
) -> dict[str, object]:
    protocol = PROTOCOL if protocol is None else coerce_protocol(protocol)
    record = _ORIGINAL_VALIDATE_MANIFEST(
        path, key, verify_hashes=verify_hashes, protocol=protocol
    )
    errors = record.get("validation_errors")
    if not isinstance(errors, list):
        errors = ["validation_errors: canonical validator returned a non-list"]
        record["validation_errors"] = errors
    try:
        loaded = canonical._load_json(path)
    except (OSError, json.JSONDecodeError):
        return record
    manifest = _mapping(loaded)
    if manifest is None:
        return record
    if manifest.get("pipeline_variant") != PIPELINE_VARIANT:
        errors.append(
            f"pipeline_variant: expected {PIPELINE_VARIANT!r}, "
            f"found {manifest.get('pipeline_variant')!r}"
        )
    if manifest.get("common_dp_gc_count_range") != [
        GC_COUNT_MIN,
        GC_COUNT_MAX,
    ]:
        errors.append(
            "common_dp_gc_count_range: expected "
            f"[{GC_COUNT_MIN}, {GC_COUNT_MAX}]"
        )
    if key.method == "bee2021":
        transfer = _mapping(
            manifest.get("primo_frozen_predictor_length_transfer")
        )
        expected = {
            "source_predictor_length_bases": 80,
            "target_length_bases": LENGTH,
            "predictor_frozen": True,
            "eligibility_blocker": PRIMO_TRANSFER_BLOCKER,
        }
        if transfer != expected:
            errors.append(
                "primo_frozen_predictor_length_transfer: does not bind the "
                "frozen 80-to-24-nt transfer"
            )
        blockers = manifest.get("main_eligibility_blockers")
        if (
            not isinstance(blockers, list)
            or PRIMO_TRANSFER_BLOCKER not in blockers
        ):
            errors.append(
                "main_eligibility_blockers: PRIMO-24 requires "
                f"{PRIMO_TRANSFER_BLOCKER!r}"
            )
    elif manifest.get("primo_frozen_predictor_length_transfer") is not None:
        errors.append(
            "primo_frozen_predictor_length_transfer: must be absent outside PRIMO"
        )
    if errors:
        record["status"] = "invalid"
        record["main_protocol_eligible"] = False
    return record


_PATCHED_FIELDS = {
    "LENGTH": LENGTH,
    "COMMON_BIO_PROJECTION": COMMON_BIO_PROJECTION,
    "DISPLAY": DISPLAY,
    "METHOD_PROTOCOL_LOCK_SHA256": METHOD_PROTOCOL_LOCK_SHA256,
    "_validate_evaluation": _validate_evaluation_24,
    "_validate_protocol_identity": _validate_protocol_identity_24,
    "_validate_manifest": _validate_manifest_24_impl,
}


@contextmanager
def configured_canonical_aggregator() -> Iterator[None]:
    with _LOCK:
        previous = {
            name: getattr(canonical, name) for name in _PATCHED_FIELDS
        }
        try:
            for name, value in _PATCHED_FIELDS.items():
                setattr(canonical, name, value)
            yield
        finally:
            for name, value in previous.items():
                setattr(canonical, name, value)


def _validate_manifest(
    path: Path,
    key: Key,
    *,
    verify_hashes: bool = True,
) -> dict[str, object]:
    with configured_canonical_aggregator():
        return canonical._validate_manifest(
            path, key, verify_hashes=verify_hashes
        )


def aggregate(
    roots: Sequence[Path],
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    blocked_methods: Mapping[str, str] | None = None,
    verify_hashes: bool = True,
):
    with configured_canonical_aggregator():
        return canonical.aggregate(
            roots,
            seeds=seeds,
            blocked_methods=blocked_methods,
            verify_hashes=verify_hashes,
        )


def _markdown(payload: Mapping[str, object]) -> str:
    with configured_canonical_aggregator():
        value = canonical._markdown(payload)
    return value.replace(
        "# Native-DNA common-P0 aggregation",
        "# Native-DNA 24-base common-P0 aggregation",
        1,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        action="append",
        default=[],
        help=(
            "Root to scan recursively for native_p0_run_manifest.json; repeat "
            "for multiple roots"
        ),
    )
    parser.add_argument(
        "--seeds",
        type=canonical._parse_seeds,
        default=DEFAULT_SEEDS,
    )
    parser.add_argument(
        "--blocked-method",
        action="append",
        default=[],
        metavar="METHOD=REASON",
    )
    parser.add_argument(
        "--out-json",
        default=str(REPO / "docs" / "native_dna_p0_24base_aggregate.json"),
    )
    parser.add_argument(
        "--out-md",
        default=str(REPO / "docs" / "native_dna_p0_24base_aggregate.md"),
    )
    args = parser.parse_args(argv)
    roots = (
        [Path(value) for value in args.root]
        if args.root
        else [REPO / "result_native_dna_24base"]
    )
    try:
        blocked = canonical._parse_blocked(args.blocked_method)
        payload, _ = aggregate(
            roots,
            seeds=args.seeds,
            blocked_methods=blocked,
            verify_hashes=True,
        )
    except ValueError as error:
        parser.error(str(error))
    canonical._atomic_json(
        Path(args.out_json).expanduser().resolve(), payload
    )
    canonical._atomic_text(
        Path(args.out_md).expanduser().resolve(), _markdown(payload)
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
