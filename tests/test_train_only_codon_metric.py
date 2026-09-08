from __future__ import annotations

import inspect
import warnings

import numpy as np
import pytest

from dna_utils import train_only_codon_metric as metric
from scripts import heldout_codon_decoding as legacy


def _bases_from_units(units: np.ndarray, bases_per_slot: int) -> np.ndarray:
    units = np.asarray(units, dtype=np.int64)
    bases = np.zeros(
        (units.shape[0], units.shape[1] * bases_per_slot), dtype=np.uint8
    )
    for slot in range(units.shape[1]):
        value = units[:, slot].copy()
        for position in range(bases_per_slot - 1, -1, -1):
            bases[:, slot * bases_per_slot + position] = value % 4
            value //= 4
        assert not value.any()
    return bases


def _labels(row_ids: np.ndarray, n_labels: int = 7) -> np.ndarray:
    row_ids = np.asarray(row_ids)
    labels = np.zeros((len(row_ids), n_labels), dtype=np.uint8)
    labels[:, 0] = row_ids % 2 == 0
    labels[:, 1] = row_ids % 3 == 0
    labels[:, 2] = row_ids % 5 <= 1
    labels[:, 3] = row_ids % 7 == 0
    labels[:, 4] = row_ids % 4 == 1
    labels[:, 5] = row_ids % 6 >= 4
    labels[:, 6] = labels.sum(axis=1) == 0
    assert (labels.sum(axis=1) > 0).all()
    return labels


def _fixture(num_slots: int = 6, bases_per_slot: int = 3) -> dict:
    n_opt, n_val = 64, 20
    opt_rows = np.arange(n_opt, dtype=np.int64)
    val_rows = np.arange(n_opt, n_opt + n_val, dtype=np.int64)
    source_rows = np.arange(n_opt + n_val, dtype=np.int64)

    row = np.arange(n_opt)[:, None]
    slot = np.arange(num_slots)[None, :]
    # Four train units per slot, each with support 16 (> canonical threshold).
    opt_units = (row + slot) % 4
    val_units = np.empty((n_val, num_slots), dtype=np.int64)
    val_units[: n_val // 2] = (
        np.arange(n_val // 2)[:, None] + slot
    ) % 4
    # The other half are valid but unseen units, exercising prior fallback.
    vocabulary = 4**bases_per_slot
    if vocabulary > 4:
        val_units[n_val // 2 :] = 4 + (slot % (vocabulary - 4))
    else:
        val_units[n_val // 2 :] = slot % 4

    split = metric.build_split_identity(
        dataset_identity="fixture/train",
        val_split_seed=42,
        val_split_ratio=n_val / (n_opt + n_val),
        source_row_ids=source_rows,
        opt_row_ids=opt_rows,
        val_row_ids=val_rows,
    )
    opt_labels = _labels(opt_rows)
    val_labels = _labels(val_rows)
    # Flickr's designated train split can contain zero-positive rows.  Keep
    # one in the golden fixture so equivalence covers the legacy nanmean rule.
    val_labels[0] = 0
    return {
        "opt_base_indices": _bases_from_units(opt_units, bases_per_slot),
        "opt_labels": opt_labels,
        "opt_row_ids": opt_rows,
        "val_base_indices": _bases_from_units(val_units, bases_per_slot),
        "val_labels": val_labels,
        "val_row_ids": val_rows,
        "source_row_ids": source_rows,
        "split_identity": split,
        "num_slots": num_slots,
        "bases_per_slot": bases_per_slot,
    }


@pytest.mark.parametrize("num_slots", [5, 6])
def test_golden_equivalence_to_legacy_decoder(num_slots: int) -> None:
    arrays = _fixture(num_slots=num_slots, bases_per_slot=3)
    result = metric.measure_train_only_codon_metric(**arrays)

    old_slots = legacy.N_SLOTS
    old_names = legacy.SLOT_NAMES
    try:
        legacy.N_SLOTS = num_slots
        legacy.SLOT_NAMES = old_names[:num_slots]
        opt_units = legacy.codon_ids(arrays["opt_base_indices"], n_bases=3)
        val_units = legacy.codon_ids(arrays["val_base_indices"], n_bases=3)
        with warnings.catch_warnings():
            # The golden implementation warns when nanmean encounters the
            # deliberately zero-positive row; excluding it is its policy.
            warnings.simplefilter("ignore", RuntimeWarning)
            expected = legacy.decode_all_slots(
                opt_units,
                arrays["opt_labels"],
                val_units,
                arrays["val_labels"],
                4**3,
                legacy.DEFAULT_ALPHA,
                legacy.DEFAULT_MIN_SUPPORT,
            )["slot_mean"]
    finally:
        legacy.N_SLOTS = old_slots
        legacy.SLOT_NAMES = old_names

    assert result["metrics"] == pytest.approx(
        {
            "concept_mAP": expected["concept_mAP"],
            "coverage": expected["coverage"],
        },
        abs=1e-15,
    )
    assert result["metrics"]["coverage"] == pytest.approx(0.5)
    assert set(result["metrics"]) == {"concept_mAP", "coverage"}
    assert result["protocol"] == {
        "dictionary_fit_split": "optimization_train",
        "evaluation_split": "heldout_train_validation",
        "alpha": 1.0,
        "min_support": 10,
        "zero_positive_label_policy": "exclude_from_concept_mAP",
        "coverage_denominator": "all_heldout_train_validation_rows",
    }


def test_dynamic_m_and_l_and_repeatable_evidence() -> None:
    arrays = _fixture(num_slots=3, bases_per_slot=2)
    first = metric.measure_train_only_codon_metric(**arrays)
    second = metric.measure_train_only_codon_metric(**arrays)
    assert first == second
    assert first["geometry"] == {
        "num_slots": 3,
        "bases_per_slot": 2,
        "total_bases": 6,
        "codon_vocabulary_size": 16,
        "num_labels": 7,
        "opt_rows": 64,
        "val_rows": 20,
        "n_valid_label_rows": 19,
        "n_zero_positive_rows": 1,
    }
    assert len(first["input_evidence"]["aggregate_sha256"]) == 64
    assert first["split_identity"] == arrays["split_identity"]


def test_measurement_does_not_mutate_caller_arrays() -> None:
    arrays = _fixture()
    before = {
        name: value.copy()
        for name, value in arrays.items()
        if isinstance(value, np.ndarray)
    }
    metric.measure_train_only_codon_metric(**arrays)
    for name, value in before.items():
        np.testing.assert_array_equal(arrays[name], value)


@pytest.mark.parametrize(
    ("field", "mutate", "match"),
    [
        ("opt_base_indices", lambda a: a.astype(np.float64), "integer A/C/G/T"),
        (
            "opt_base_indices",
            lambda a: np.pad(a, ((0, 0), (0, 1))),
            r"shape \[64, 18\]",
        ),
        (
            "val_base_indices",
            lambda a: np.where(np.arange(a.size).reshape(a.shape) == 0, 4, a),
            r"outside \[0, 3\]",
        ),
        ("opt_labels", lambda a: a[:, :-1], "labels; expected 6|shape"),
        (
            "val_labels",
            lambda a: np.where(np.arange(a.size).reshape(a.shape) == 0, 2, a),
            "binary values",
        ),
        (
            "val_labels",
            lambda a: np.where(np.arange(a.size).reshape(a.shape) == 0, np.nan, a),
            "finite binary",
        ),
    ],
)
def test_array_validation_fails_closed(field, mutate, match) -> None:
    arrays = _fixture()
    arrays[field] = mutate(arrays[field])
    with pytest.raises(metric.TrainOnlyCodonMetricError, match=match):
        metric.measure_train_only_codon_metric(**arrays)


def test_row_count_is_bound_to_each_array() -> None:
    arrays = _fixture()
    arrays["val_base_indices"] = arrays["val_base_indices"][:-1]
    with pytest.raises(metric.TrainOnlyCodonMetricError, match=r"shape \[20, 18\]"):
        metric.measure_train_only_codon_metric(**arrays)


@pytest.mark.parametrize("failure", ["overlap", "gap_or_orphan", "duplicate"])
def test_partition_refuses_overlap_gap_or_orphan_and_duplicates(failure: str) -> None:
    arrays = _fixture()
    if failure == "overlap":
        arrays["val_row_ids"] = arrays["val_row_ids"].copy()
        arrays["val_row_ids"][0] = arrays["opt_row_ids"][0]
    elif failure == "gap_or_orphan":
        arrays["val_row_ids"] = arrays["val_row_ids"].copy()
        arrays["val_row_ids"][0] = 999
    else:
        arrays["opt_row_ids"] = arrays["opt_row_ids"].copy()
        arrays["opt_row_ids"][1] = arrays["opt_row_ids"][0]
    with pytest.raises(metric.TrainOnlyCodonMetricError):
        metric.measure_train_only_codon_metric(**arrays)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("opt_row_ids_sha256", "0" * 64),
        ("source_row_count", 999),
        ("val_split_seed", 43),
        ("identity_sha256", "f" * 64),
    ],
)
def test_split_identity_tampering_is_refused(field: str, value) -> None:
    arrays = _fixture()
    arrays["split_identity"] = dict(arrays["split_identity"])
    arrays["split_identity"][field] = value
    with pytest.raises(metric.TrainOnlyCodonMetricError, match="split_identity"):
        metric.measure_train_only_codon_metric(**arrays)


def test_split_identity_rejects_unknown_schema_field() -> None:
    arrays = _fixture()
    arrays["split_identity"] = {**arrays["split_identity"], "note": "unbound"}
    with pytest.raises(metric.TrainOnlyCodonMetricError, match="schema mismatch"):
        metric.measure_train_only_codon_metric(**arrays)


def test_all_zero_positive_rows_are_refused() -> None:
    arrays = _fixture()
    arrays["val_labels"][:] = 0
    with pytest.raises(metric.TrainOnlyCodonMetricError, match="no row with a positive"):
        metric.measure_train_only_codon_metric(**arrays)


def test_public_measurement_has_no_artifact_or_policy_inputs() -> None:
    parameters = set(inspect.signature(metric.measure_train_only_codon_metric).parameters)
    assert not any("path" in name or "artifact" in name for name in parameters)
    assert not any("query" in name or "test" in name for name in parameters)
    assert "alpha" not in parameters
    assert "min_support" not in parameters
    assert not any("winner" in name or "retrieval" in name for name in parameters)


def test_row_digest_binds_order_and_is_dtype_canonical() -> None:
    assert metric.row_ids_sha256(np.array([1, 2, 3], dtype=np.int32)) == (
        metric.row_ids_sha256(np.array([1, 2, 3], dtype=np.int64))
    )
    assert metric.row_ids_sha256([1, 2, 3]) != metric.row_ids_sha256([3, 2, 1])
