"""Leakage-safe codon decoding measurement for Phase 3 analysis.

This module deliberately operates on in-memory arrays.  It has no extraction
loader and accepts no query/test artefact path: the caller must provide the
optimization-train and held-out *training-validation* arrays explicitly.  A
dictionary is fitted on the former and evaluated on the latter.

This is diagnostic/paper-analysis evidence, not a Phase-3 selection gate.  The
public measurement contract fixes the two preregistered decoder constants
(``alpha=1.0`` and ``min_support=10``) and returns only concept mAP and
coverage.  In particular, it contains no reducer that combines these values
with retrieval, and no candidate/winner rule.
"""
from __future__ import annotations

from hashlib import sha256
import json
import math
from typing import Any, Mapping

import numpy as np


DEFAULT_ALPHA = 1.0
DEFAULT_MIN_SUPPORT = 10

METRIC_SCHEMA = "groundeddna.phase3-train-only-codon-metric"
METRIC_SCHEMA_VERSION = 1
SPLIT_SCHEMA = "groundeddna.train-validation-split-identity"
SPLIT_SCHEMA_VERSION = 1

_ROW_ID_ENCODING = "sha256-int64-le-c-order-v1"
_ARRAY_DIGEST_ENCODING = "sha256-canonical-array-v1"
_SPLIT_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "dataset_identity",
        "val_split_seed",
        "val_split_ratio",
        "source_row_count",
        "opt_row_count",
        "val_row_count",
        "row_id_encoding",
        "source_row_ids_sha256",
        "opt_row_ids_sha256",
        "val_row_ids_sha256",
        "identity_sha256",
    }
)


class TrainOnlyCodonMetricError(ValueError):
    """The requested measurement is not admissible under the contract."""


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _domain_digest(domain: str, payload: bytes) -> str:
    digest = sha256()
    digest.update(domain.encode("ascii"))
    digest.update(b"\0")
    digest.update(payload)
    return digest.hexdigest()


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TrainOnlyCodonMetricError(f"{name} must be a positive integer")
    value = int(value)
    if value <= 0:
        raise TrainOnlyCodonMetricError(f"{name} must be positive, got {value}")
    return value


def _row_ids(value: Any, *, name: str, allow_empty: bool = False) -> np.ndarray:
    rows = np.asarray(value)
    if rows.ndim != 1:
        raise TrainOnlyCodonMetricError(
            f"{name} must have shape [N], got {rows.shape}"
        )
    if not allow_empty and rows.size == 0:
        raise TrainOnlyCodonMetricError(f"{name} must not be empty")
    if np.issubdtype(rows.dtype, np.bool_) or not np.issubdtype(
        rows.dtype, np.integer
    ):
        raise TrainOnlyCodonMetricError(
            f"{name} must contain integer dataset-row identifiers, got {rows.dtype}"
        )
    if rows.size:
        # Check before casting: uint64 values above int64 would otherwise wrap.
        if np.any(rows < 0) or int(np.max(rows)) > np.iinfo(np.int64).max:
            raise TrainOnlyCodonMetricError(
                f"{name} row identifiers must be in [0, {np.iinfo(np.int64).max}]"
            )
    rows = np.ascontiguousarray(rows, dtype="<i8")
    if np.unique(rows).size != rows.size:
        raise TrainOnlyCodonMetricError(f"{name} contains duplicate row identifiers")
    return rows


def _validate_exact_partition(
    source_row_ids: Any,
    opt_row_ids: Any,
    val_row_ids: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = _row_ids(source_row_ids, name="source_row_ids")
    opt = _row_ids(opt_row_ids, name="opt_row_ids")
    val = _row_ids(val_row_ids, name="val_row_ids")

    if source.size != opt.size + val.size:
        raise TrainOnlyCodonMetricError(
            "opt/val row counts do not exactly partition source_row_ids: "
            f"{opt.size}+{val.size}!={source.size}"
        )
    if np.intersect1d(opt, val, assume_unique=True).size:
        raise TrainOnlyCodonMetricError(
            "optimization-train and held-out train-validation row ids overlap"
        )
    combined = np.sort(np.concatenate((opt, val)))
    if not np.array_equal(combined, np.sort(source)):
        raise TrainOnlyCodonMetricError(
            "opt/val row ids contain a gap or orphan relative to source_row_ids"
        )
    return source, opt, val


def row_ids_sha256(row_ids: Any) -> str:
    """Digest an ordered integer row-id vector with an explicit encoding."""
    rows = _row_ids(row_ids, name="row_ids", allow_empty=True)
    header = len(rows).to_bytes(8, "little", signed=False)
    return _domain_digest(_ROW_ID_ENCODING, header + rows.tobytes(order="C"))


def _normalise_split_metadata(
    *, dataset_identity: Any, val_split_seed: Any, val_split_ratio: Any
) -> tuple[str, int, float]:
    if not isinstance(dataset_identity, str) or not dataset_identity.strip():
        raise TrainOnlyCodonMetricError(
            "dataset_identity must be a non-empty string"
        )
    if isinstance(val_split_seed, (bool, np.bool_)) or not isinstance(
        val_split_seed, (int, np.integer)
    ):
        raise TrainOnlyCodonMetricError("val_split_seed must be an integer")
    seed = int(val_split_seed)
    if isinstance(val_split_ratio, (bool, np.bool_)) or not isinstance(
        val_split_ratio, (int, float, np.integer, np.floating)
    ):
        raise TrainOnlyCodonMetricError("val_split_ratio must be a finite number")
    ratio = float(val_split_ratio)
    if not math.isfinite(ratio) or not 0.0 < ratio < 0.5:
        raise TrainOnlyCodonMetricError(
            f"val_split_ratio must be finite and in (0, 0.5), got {ratio!r}"
        )
    return dataset_identity.strip(), seed, ratio


def build_split_identity(
    *,
    dataset_identity: str,
    val_split_seed: int,
    val_split_ratio: float,
    source_row_ids: Any,
    opt_row_ids: Any,
    val_row_ids: Any,
) -> dict[str, Any]:
    """Build evidence for one exact, ordered train -> opt/val partition.

    The function refuses overlap, gaps, or rows outside the source universe.
    The three ordered-vector digests also bind extraction-array row order, not
    merely set membership.
    """
    dataset, seed, ratio = _normalise_split_metadata(
        dataset_identity=dataset_identity,
        val_split_seed=val_split_seed,
        val_split_ratio=val_split_ratio,
    )
    source, opt, val = _validate_exact_partition(
        source_row_ids, opt_row_ids, val_row_ids
    )
    record: dict[str, Any] = {
        "schema": SPLIT_SCHEMA,
        "schema_version": SPLIT_SCHEMA_VERSION,
        "dataset_identity": dataset,
        "val_split_seed": seed,
        "val_split_ratio": ratio,
        "source_row_count": int(source.size),
        "opt_row_count": int(opt.size),
        "val_row_count": int(val.size),
        "row_id_encoding": _ROW_ID_ENCODING,
        "source_row_ids_sha256": row_ids_sha256(source),
        "opt_row_ids_sha256": row_ids_sha256(opt),
        "val_row_ids_sha256": row_ids_sha256(val),
    }
    record["identity_sha256"] = _domain_digest(
        SPLIT_SCHEMA, _canonical_json(record)
    )
    return record


def _verify_split_identity(
    split_identity: Mapping[str, Any],
    *,
    source: np.ndarray,
    opt: np.ndarray,
    val: np.ndarray,
) -> dict[str, Any]:
    if not isinstance(split_identity, Mapping):
        raise TrainOnlyCodonMetricError("split_identity must be a mapping")
    given_fields = frozenset(split_identity)
    if given_fields != _SPLIT_FIELDS:
        missing = sorted(_SPLIT_FIELDS - given_fields)
        extra = sorted(given_fields - _SPLIT_FIELDS)
        raise TrainOnlyCodonMetricError(
            f"split_identity schema mismatch: missing={missing}, extra={extra}"
        )
    expected = build_split_identity(
        dataset_identity=split_identity["dataset_identity"],
        val_split_seed=split_identity["val_split_seed"],
        val_split_ratio=split_identity["val_split_ratio"],
        source_row_ids=source,
        opt_row_ids=opt,
        val_row_ids=val,
    )
    for field in sorted(_SPLIT_FIELDS):
        # Strict type equality prevents True == 1 and similar JSON ambiguity.
        actual_value = split_identity[field]
        expected_value = expected[field]
        if type(actual_value) is not type(expected_value) or actual_value != expected_value:
            raise TrainOnlyCodonMetricError(
                f"split_identity {field} does not match the supplied row partition"
            )
    return expected


def _base_indices(
    value: Any,
    *,
    name: str,
    expected_rows: int,
    num_slots: int,
    bases_per_slot: int,
) -> np.ndarray:
    bases = np.asarray(value)
    expected_width = num_slots * bases_per_slot
    if bases.ndim != 2 or bases.shape != (expected_rows, expected_width):
        raise TrainOnlyCodonMetricError(
            f"{name} must have shape [{expected_rows}, {expected_width}], "
            f"got {bases.shape}"
        )
    if np.issubdtype(bases.dtype, np.bool_) or not np.issubdtype(
        bases.dtype, np.integer
    ):
        raise TrainOnlyCodonMetricError(
            f"{name} must contain integer A/C/G/T indices, got {bases.dtype}"
        )
    if bases.size and (np.any(bases < 0) or np.any(bases > 3)):
        raise TrainOnlyCodonMetricError(f"{name} contains a base outside [0, 3]")
    return np.ascontiguousarray(bases, dtype=np.uint8)


def _labels(
    value: Any,
    *,
    name: str,
    expected_rows: int,
    expected_labels: int | None = None,
) -> np.ndarray:
    labels = np.asarray(value)
    if labels.ndim != 2 or labels.shape[0] != expected_rows or labels.shape[1] == 0:
        suffix = "C" if expected_labels is None else str(expected_labels)
        raise TrainOnlyCodonMetricError(
            f"{name} must have shape [{expected_rows}, {suffix}] with C > 0, "
            f"got {labels.shape}"
        )
    if expected_labels is not None and labels.shape[1] != expected_labels:
        raise TrainOnlyCodonMetricError(
            f"{name} has {labels.shape[1]} labels; expected {expected_labels}"
        )
    if not (
        np.issubdtype(labels.dtype, np.number)
        or np.issubdtype(labels.dtype, np.bool_)
    ) or np.issubdtype(labels.dtype, np.complexfloating):
        raise TrainOnlyCodonMetricError(
            f"{name} must contain numeric binary labels, got {labels.dtype}"
        )
    try:
        finite = np.isfinite(labels)
    except TypeError:
        raise TrainOnlyCodonMetricError(
            f"{name} must contain finite binary labels"
        ) from None
    if not bool(finite.all()) or not bool(((labels == 0) | (labels == 1)).all()):
        raise TrainOnlyCodonMetricError(
            f"{name} must contain only finite binary values 0 or 1"
        )
    return np.ascontiguousarray(labels, dtype=np.uint8)


def _codon_ids(
    base_indices: np.ndarray, *, num_slots: int, bases_per_slot: int
) -> np.ndarray:
    units = np.zeros((len(base_indices), num_slots), dtype=np.int64)
    for slot in range(num_slots):
        segment = base_indices[
            :, slot * bases_per_slot : (slot + 1) * bases_per_slot
        ]
        value = np.zeros(len(base_indices), dtype=np.int64)
        for position in range(bases_per_slot):
            value = value * 4 + segment[:, position].astype(np.int64)
        units[:, slot] = value
    return units


def _label_ranking_ap(scores: np.ndarray, truth: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, axis=1, kind="stable")
    hits = np.take_along_axis(truth, order, axis=1).astype(np.float64)
    ranks = np.arange(1, scores.shape[1] + 1, dtype=np.float64)[None, :]
    precision = np.cumsum(hits, axis=1) / ranks
    positives = hits.sum(axis=1)
    return np.where(
        positives > 0,
        (precision * hits).sum(axis=1) / np.maximum(positives, 1e-12),
        np.nan,
    )


def _decode_slot(
    opt_units: np.ndarray,
    opt_labels: np.ndarray,
    val_units: np.ndarray,
    val_labels: np.ndarray,
) -> tuple[np.ndarray, float]:
    # Sparse over units observed in optimization-train.  This is algebraically
    # identical to the legacy dense 4**L table, while remaining safe for a
    # caller-selected L (the canonical experiment uses L=3).
    observed, inverse = np.unique(opt_units, return_inverse=True)
    support = np.bincount(inverse, minlength=len(observed)).astype(np.float64)
    counts = np.zeros((len(observed), opt_labels.shape[1]), dtype=np.float64)
    np.add.at(counts, inverse, opt_labels.astype(np.float64))
    probabilities = (counts + DEFAULT_ALPHA) / (
        support[:, None] + 2.0 * DEFAULT_ALPHA
    )
    prior = (opt_labels.sum(axis=0, dtype=np.float64) + DEFAULT_ALPHA) / (
        len(opt_labels) + 2.0 * DEFAULT_ALPHA
    )

    positions = np.searchsorted(observed, val_units)
    found = positions < len(observed)
    if bool(found.any()):
        found_rows = np.flatnonzero(found)
        found[found_rows] = observed[positions[found_rows]] == val_units[found_rows]
    matched_support = np.zeros(len(val_units), dtype=np.float64)
    matched_support[found] = support[positions[found]]
    covered = found & (matched_support >= DEFAULT_MIN_SUPPORT)

    scores = np.broadcast_to(prior, (len(val_units), len(prior))).copy()
    scores[covered] = probabilities[positions[covered]]
    if not bool(np.isfinite(scores).all()):
        raise TrainOnlyCodonMetricError("decoder produced a non-finite score")
    ap = _label_ranking_ap(scores, val_labels)
    valid_label_rows = val_labels.sum(axis=1) > 0
    if not bool(np.isfinite(ap[valid_label_rows]).all()):
        raise TrainOnlyCodonMetricError(
            "concept AP is non-finite on a validation row with a positive label"
        )
    if bool(np.isfinite(ap[~valid_label_rows]).any()):
        raise TrainOnlyCodonMetricError(
            "concept AP must be undefined on zero-positive validation rows"
        )
    return ap, float(covered.mean())


def _canonical_array_sha256(array: np.ndarray, *, semantic: str) -> str:
    canonical = np.ascontiguousarray(array)
    header = _canonical_json(
        {
            "encoding": _ARRAY_DIGEST_ENCODING,
            "semantic": semantic,
            "dtype": canonical.dtype.str,
            "shape": list(canonical.shape),
        }
    )
    return _domain_digest(
        _ARRAY_DIGEST_ENCODING, header + b"\0" + canonical.tobytes(order="C")
    )


def _input_evidence(
    *,
    opt_bases: np.ndarray,
    opt_labels: np.ndarray,
    opt_rows: np.ndarray,
    val_bases: np.ndarray,
    val_labels: np.ndarray,
    val_rows: np.ndarray,
) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "encoding": _ARRAY_DIGEST_ENCODING,
        "opt_base_indices_sha256": _canonical_array_sha256(
            opt_bases, semantic="opt_base_indices"
        ),
        "opt_labels_sha256": _canonical_array_sha256(
            opt_labels, semantic="opt_labels"
        ),
        "opt_row_ids_sha256": row_ids_sha256(opt_rows),
        "val_base_indices_sha256": _canonical_array_sha256(
            val_bases, semantic="val_base_indices"
        ),
        "val_labels_sha256": _canonical_array_sha256(
            val_labels, semantic="val_labels"
        ),
        "val_row_ids_sha256": row_ids_sha256(val_rows),
    }
    evidence["aggregate_sha256"] = _domain_digest(
        METRIC_SCHEMA + ".inputs", _canonical_json(evidence)
    )
    return evidence


def measure_train_only_codon_metric(
    *,
    opt_base_indices: Any,
    opt_labels: Any,
    opt_row_ids: Any,
    val_base_indices: Any,
    val_labels: Any,
    val_row_ids: Any,
    source_row_ids: Any,
    split_identity: Mapping[str, Any],
    num_slots: int,
    bases_per_slot: int,
) -> dict[str, Any]:
    """Fit on opt-train and measure codon decoding on held-out train-val.

    ``split_identity`` must have been established for the exact source/opt/val
    row vectors (for example with :func:`build_split_identity`).  It is
    recomputed and compared before any metric is evaluated.  ``num_slots`` (M)
    and ``bases_per_slot`` (L) are explicit and the base width must equal M*L.

    Returns a JSON-serialisable record whose ``metrics`` object has exactly two
    entries: ``concept_mAP`` and ``coverage``.
    """
    num_slots = _positive_int(num_slots, name="num_slots")
    bases_per_slot = _positive_int(bases_per_slot, name="bases_per_slot")
    # Base-4 identifiers must fit the signed int64 representation used here.
    if bases_per_slot > 31:
        raise TrainOnlyCodonMetricError(
            "bases_per_slot must be <= 31 for collision-free int64 base-4 ids"
        )

    source_rows, opt_rows, val_rows = _validate_exact_partition(
        source_row_ids, opt_row_ids, val_row_ids
    )
    verified_split = _verify_split_identity(
        split_identity, source=source_rows, opt=opt_rows, val=val_rows
    )

    opt_bases = _base_indices(
        opt_base_indices,
        name="opt_base_indices",
        expected_rows=len(opt_rows),
        num_slots=num_slots,
        bases_per_slot=bases_per_slot,
    )
    val_bases = _base_indices(
        val_base_indices,
        name="val_base_indices",
        expected_rows=len(val_rows),
        num_slots=num_slots,
        bases_per_slot=bases_per_slot,
    )
    opt_lab = _labels(
        opt_labels, name="opt_labels", expected_rows=len(opt_rows)
    )
    val_lab = _labels(
        val_labels,
        name="val_labels",
        expected_rows=len(val_rows),
        expected_labels=opt_lab.shape[1],
    )
    valid_label_rows = val_lab.sum(axis=1) > 0
    n_valid_label_rows = int(valid_label_rows.sum())
    n_zero_positive_rows = int(len(val_lab) - n_valid_label_rows)
    if n_valid_label_rows == 0:
        raise TrainOnlyCodonMetricError(
            "held-out train-validation contains no row with a positive label"
        )

    opt_units = _codon_ids(
        opt_bases, num_slots=num_slots, bases_per_slot=bases_per_slot
    )
    val_units = _codon_ids(
        val_bases, num_slots=num_slots, bases_per_slot=bases_per_slot
    )
    aps: list[np.ndarray] = []
    coverage: list[float] = []
    for slot in range(num_slots):
        slot_ap, slot_coverage = _decode_slot(
            opt_units[:, slot], opt_lab, val_units[:, slot], val_lab
        )
        aps.append(slot_ap)
        coverage.append(slot_coverage)

    # This is the legacy decoder's nanmean policy stated without a warning:
    # zero-positive rows have undefined AP and are excluded from concept mAP,
    # while coverage below intentionally retains *all* validation rows.
    ap_stack = np.stack(aps, axis=0)
    concept_map = float(ap_stack[:, valid_label_rows].mean(axis=0).mean())
    mean_coverage = float(np.mean(coverage))
    if not math.isfinite(concept_map) or not math.isfinite(mean_coverage):
        raise TrainOnlyCodonMetricError("decoder metric is non-finite")
    if not 0.0 <= concept_map <= 1.0 or not 0.0 <= mean_coverage <= 1.0:
        raise TrainOnlyCodonMetricError(
            "decoder metric lies outside its required [0, 1] range"
        )

    evidence = _input_evidence(
        opt_bases=opt_bases,
        opt_labels=opt_lab,
        opt_rows=opt_rows,
        val_bases=val_bases,
        val_labels=val_lab,
        val_rows=val_rows,
    )
    return {
        "schema": METRIC_SCHEMA,
        "schema_version": METRIC_SCHEMA_VERSION,
        "protocol": {
            "dictionary_fit_split": "optimization_train",
            "evaluation_split": "heldout_train_validation",
            "alpha": DEFAULT_ALPHA,
            "min_support": DEFAULT_MIN_SUPPORT,
            "zero_positive_label_policy": "exclude_from_concept_mAP",
            "coverage_denominator": "all_heldout_train_validation_rows",
        },
        "geometry": {
            "num_slots": num_slots,
            "bases_per_slot": bases_per_slot,
            "total_bases": num_slots * bases_per_slot,
            "codon_vocabulary_size": 4**bases_per_slot,
            "num_labels": int(opt_lab.shape[1]),
            "opt_rows": int(len(opt_rows)),
            "val_rows": int(len(val_rows)),
            "n_valid_label_rows": n_valid_label_rows,
            "n_zero_positive_rows": n_zero_positive_rows,
        },
        "split_identity": verified_split,
        "input_evidence": evidence,
        "metrics": {
            "concept_mAP": concept_map,
            "coverage": mean_coverage,
        },
    }


__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_MIN_SUPPORT",
    "METRIC_SCHEMA",
    "METRIC_SCHEMA_VERSION",
    "SPLIT_SCHEMA",
    "SPLIT_SCHEMA_VERSION",
    "TrainOnlyCodonMetricError",
    "build_split_identity",
    "measure_train_only_codon_metric",
    "row_ids_sha256",
]
