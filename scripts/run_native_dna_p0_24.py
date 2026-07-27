#!/usr/bin/env python3
"""Run one native-DNA predecessor at a sealed, matched 24-base capacity.

This is a versioned adapter over :mod:`scripts.run_native_dna_p0`.  It keeps
the already-sealed 18-base implementation byte-for-byte unchanged while
binding a distinct trainer entry point, protocol digest, cell name, exact
GC-count range, and PRIMO length-transfer declaration for 24-base runs.
"""

from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import run_native_dna_p0 as canonical  # noqa: E402


MATCHED_LENGTH = 24
GC_MIN = 0.4
GC_MAX = 0.6
GC_COUNT_MIN = 10
GC_COUNT_MAX = 14
MAX_RUN = 3
PIPELINE_VARIANT = "native_dna_matched_24base_p0_v1"
PRIMO_TRANSFER_BLOCKER = "primo_frozen_predictor_length_transfer_80_to_24nt"
TRAINER = REPO / "scripts" / "train_native_dna_baseline_24.py"

DEFAULT_CACHE = canonical.DEFAULT_CACHE
DATASETS = canonical.DATASETS
METHODS = canonical.METHODS
PAPER_SEEDS = canonical.PAPER_SEEDS
SOURCE_PROFILES = canonical.SOURCE_PROFILES
_audit_cache = canonical._audit_cache

_ORIGINAL_PROTOCOL_IDENTITY = canonical._protocol_identity
_ORIGINAL_TRAINING_COMMAND = canonical._training_command
_ORIGINAL_MANIFEST = canonical._manifest

_IMPLEMENTATION_PATHS = tuple(dict.fromkeys((
    "scripts/run_native_dna_p0_24.py",
    "scripts/train_native_dna_baseline_24.py",
    *canonical.IMPLEMENTATION_PATHS,
)))

_PRIMO_AUDIT = copy.deepcopy(canonical.SOURCE_REPRODUCTION_AUDIT)
_PRIMO_AUDIT["bee2021"]["declared_adaptation"] = (
    "The official 80-nt yield predictor is frozen and transferred to 24 nt; "
    "PRIMO's per-epoch NUPACK relabeling/predictor refit is not performed."
)
_PRIMO_AUDIT["bee2021"]["length_transfer"] = {
    "source_predictor_length_bases": 80,
    "target_length_bases": MATCHED_LENGTH,
    "predictor_frozen": True,
    "alternating_predictor_refit": False,
}


def _protocol_identity_24(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], str]:
    payload, _ = _ORIGINAL_PROTOCOL_IDENTITY(*args, **kwargs)
    payload["pipeline_variant"] = PIPELINE_VARIANT
    projection = dict(payload["common_bio_projection"])
    projection.update({
        "gc_count_min_inclusive": GC_COUNT_MIN,
        "gc_count_max_inclusive": GC_COUNT_MAX,
        "sequence_length_bases": MATCHED_LENGTH,
    })
    payload["common_bio_projection"] = projection
    if payload.get("method") == "bee2021":
        payload["primo_frozen_predictor_length_transfer"] = {
            "source_predictor_length_bases": 80,
            "target_length_bases": MATCHED_LENGTH,
            "predictor_frozen": True,
            "alternating_predictor_refit": False,
            "eligibility_blocker": PRIMO_TRANSFER_BLOCKER,
        }
    canonical_json = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return payload, hashlib.sha256(canonical_json).hexdigest()


def _training_command_24(*args: Any, **kwargs: Any) -> list[str]:
    command = _ORIGINAL_TRAINING_COMMAND(*args, **kwargs)
    if len(command) < 2:
        raise ValueError("canonical trainer command is unexpectedly short")
    command[1] = str(TRAINER)
    length_index = command.index("--length") + 1
    if command[length_index] != str(MATCHED_LENGTH):
        raise ValueError("canonical command did not bind the 24-base capacity")
    return command


def _manifest_24(*args: Any, **kwargs: Any) -> dict[str, Any]:
    payload = _ORIGINAL_MANIFEST(*args, **kwargs)
    payload["pipeline_variant"] = PIPELINE_VARIANT
    payload["common_dp_gc_count_range"] = [GC_COUNT_MIN, GC_COUNT_MAX]
    if payload.get("method") == "bee2021":
        payload["primo_frozen_predictor_length_transfer"] = {
            "source_predictor_length_bases": 80,
            "target_length_bases": MATCHED_LENGTH,
            "predictor_frozen": True,
            "eligibility_blocker": PRIMO_TRANSFER_BLOCKER,
        }
    return payload


_PATCHED_FIELDS = {
    "MATCHED_LENGTH": MATCHED_LENGTH,
    "GC_MIN": GC_MIN,
    "GC_MAX": GC_MAX,
    "MAX_RUN": MAX_RUN,
    "IMPLEMENTATION_PATHS": _IMPLEMENTATION_PATHS,
    "SOURCE_REPRODUCTION_AUDIT": _PRIMO_AUDIT,
    "PRIMO_TRANSFER_BLOCKER": PRIMO_TRANSFER_BLOCKER,
    "_protocol_identity": _protocol_identity_24,
    "_training_command": _training_command_24,
    "_manifest": _manifest_24,
}


@contextmanager
def configured_canonical_driver() -> Iterator[None]:
    """Temporarily configure the canonical driver for the 24-base variant."""
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


def _protocol_identity(
    *,
    method: str,
    dataset: str,
    setting: str,
    seed: int,
    cache_audit: Mapping[str, Any],
    dataset_root: Path,
    predictor: Path | None,
    device: str,
    num_workers: int,
    extract_batch_size: int,
    query_chunk: int,
) -> tuple[dict[str, Any], str]:
    with configured_canonical_driver():
        return canonical._protocol_identity(
            method=method,
            dataset=dataset,
            setting=setting,
            seed=seed,
            cache_audit=cache_audit,
            dataset_root=dataset_root,
            predictor=predictor,
            device=device,
            num_workers=num_workers,
            extract_batch_size=extract_batch_size,
            query_chunk=query_chunk,
        )


def _training_command(**kwargs: Any) -> list[str]:
    with configured_canonical_driver():
        return canonical._training_command(**kwargs)


def build_parser():
    with configured_canonical_driver():
        return canonical.build_parser()


def main(argv: Sequence[str] | None = None) -> int:
    with configured_canonical_driver():
        return canonical.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
