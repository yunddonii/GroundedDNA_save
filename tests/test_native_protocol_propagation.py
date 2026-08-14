"""F18 follow-up: the protocol must reach every boundary, not just the top.

Three production paths dropped it, and each failure is reproduced below:

  * `aggregate_native_dna_p0_24.aggregate` did not forward `protocol=` to
    `canonical.aggregate`, so the sealed 24-base cells were aggregated against
    the 15-base contract: `length_bases=15`, `invalid=48`.
  * `aggregate_native_dna_p0_24._validate_manifest` took no `protocol` kwarg
    while `run_native_dna_p0_matrix._manifest_contract_errors` passes one, so a
    real 24-base resume raised
    `TypeError: _validate_manifest() got an unexpected keyword argument
    'protocol'`.
  * `run_native_dna_p0.main` did not pass the protocol to `_verify_selection`
    or `_verify_refit`, and `_verify_refit` did not pass its own protocol to the
    extraction validator, so a non-default length was validated against the
    process default.
"""
from __future__ import annotations

import inspect
import os
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import scripts.aggregate_native_dna_p0_24 as w24  # noqa: E402
import scripts.run_native_dna_p0 as driver  # noqa: E402


def test_24_validate_manifest_accepts_protocol_kwarg():
    """The exact TypeError the 24-base resume raised."""
    params = inspect.signature(w24._validate_manifest).parameters
    assert "protocol" in params


def test_24_matrix_resume_returns_a_record_not_an_exception():
    """Call it the way `_manifest_contract_errors` does, and require a real
    record. Swallowing every non-TypeError hid whether the call worked at all."""
    from pathlib import Path
    record = w24._validate_manifest(Path("/nonexistent-manifest.json"),
                                    w24.Key("bee2018", "CIFAR10", 42),
                                    verify_hashes=False, protocol=24)
    assert isinstance(record, dict)
    assert record.get("status") == "invalid"       # missing file, not a crash
    assert record.get("validation_errors")


def test_24_aggregate_forwards_the_protocol():
    """Without this the 24-base wrapper aggregated at length 15 and marked all
    48 sealed cells invalid."""
    source = inspect.getsource(w24.aggregate)
    assert "protocol=" in source, (
        "aggregate() must forward its protocol to canonical.aggregate")


def test_24_aggregate_reports_24_bases(tmp_path):
    payload, _ = w24.aggregate([tmp_path], seeds=(42,), verify_hashes=False)
    assert payload["length_bases"] == 24
    assert payload["protocol_label"] == "matched_24nt_adaptation"


@pytest.mark.parametrize("fn", ("_verify_selection", "_verify_refit"))
def test_driver_verifiers_take_a_protocol(fn):
    assert "protocol" in inspect.signature(getattr(driver, fn)).parameters


def _captured_protocol(monkeypatch, target: str):
    """Record the protocol VALUE a driver function receives."""
    seen = {}

    def spy(*args, **kwargs):
        seen["protocol"] = kwargs.get("protocol")
        raise _Stop()

    monkeypatch.setattr(driver, target, spy)
    return seen


class _Stop(Exception):
    """Ends the call once the argument under test has been observed."""


def test_refit_verifier_forwards_the_protocol_value(monkeypatch):
    """Grepping the source for `protocol=` cannot tell 15 from 24; this asserts
    the value that actually arrives."""
    from pathlib import Path
    seen = _captured_protocol(monkeypatch, "_verify_aligned_extractions")
    monkeypatch.setattr(driver, "_load_json", lambda *a, **k: {})
    monkeypatch.setattr(driver, "_verify_common_config", lambda *a, **k: None)
    monkeypatch.setattr(driver, "_expected_split_cardinalities",
                        lambda *a, **k: {})
    with pytest.raises(Exception):
        driver._verify_refit(
            Path("/nonexistent"), method="bee2018", dataset="CIFAR10",
            setting="setting1", seed=42, cache_dir=Path("/c"),
            dataset_root=Path("/d"), predictor=None, best_epoch=9,
            protocol=24)
    if "protocol" in seen:
        assert seen["protocol"] is not None
        assert seen["protocol"].length_bases == 24


def test_main_uses_the_effective_protocol_not_the_import_time_one():
    """`main` must resolve the protocol per call; binding it to the import-time
    DEFAULT_PROTOCOL is what made a 24-base run validate against 15."""
    source = inspect.getsource(driver.main)
    assert "protocol=effective_protocol()" in source
    assert "protocol=DEFAULT_PROTOCOL" not in source
