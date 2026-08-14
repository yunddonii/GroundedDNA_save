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


def test_refit_verifier_forwards_the_protocol_value(monkeypatch, tmp_path):
    """Assert the VALUE that arrives, and require the spy to be reached.

    The previous version wrapped the call in `pytest.raises(Exception)` with a
    conditional assertion, so a `ValueError` raised before the spy was ever
    called counted as a pass -- measured `seen == {}`.
    """
    from pathlib import Path

    seen = {}

    class _Stop(Exception):
        pass

    def spy(*args, **kwargs):
        seen["protocol"] = kwargs.get("protocol")
        raise _Stop()

    # A terminal evaluation for a 24-BASE run. `_verify_refit` compares
    # `evaluation["length"]` against `protocol.length_bases`, so this fixture
    # only gets past that check if the 24-base protocol actually arrived --
    # the propagation is proved twice over, here and at the spy.
    evaluation = {
        "method": "bee2018", "dataset": "CIFAR10", "length": 24,
        "neural_raw": {"mAP_at_R": 0.5},
        "projected": {"mAP_at_R": 0.5},
        "query": {"n": 1000}, "database": {"n": 59000},
    }

    def _json(path, purpose="", *a, **k):
        # the cache meta lookup shares this helper
        return {"N": 60000} if "meta.json" in str(path) else evaluation

    monkeypatch.setattr(driver, "_verify_aligned_extractions", spy)
    monkeypatch.setattr(driver, "_load_json", _json)
    monkeypatch.setattr(driver, "_verify_common_config", lambda *a, **k: None)
    monkeypatch.setattr(driver, "_expected_split_cardinalities",
                        lambda *a, **k: {})
    monkeypatch.setattr(driver, "_verify_split", lambda *a, **k: None)
    monkeypatch.setattr(driver, "_checkpoint_set_digest", lambda *a, **k: "d")

    # A real refit directory shape, so the checks before the extraction
    # validator pass and the spy is actually reached.
    refit = tmp_path / "refit"
    refit.mkdir()
    (refit / "epoch_009.pth").write_bytes(b"ck")
    monkeypatch.setattr(driver, "_require_complete_post_compliance",
                        lambda *a, **k: {"query": 1.0, "database": 1.0})

    with pytest.raises(_Stop):          # only the spy may end this call
        driver._verify_refit(
            refit, method="bee2018", dataset="CIFAR10",
            setting="setting1", seed=42, cache_dir=Path("/c"),
            dataset_root=Path("/d"), predictor=None, best_epoch=9,
            protocol=24)

    assert "protocol" in seen, "the extraction validator was never reached"
    assert seen["protocol"].length_bases == 24


def test_main_uses_the_effective_protocol_not_the_import_time_one():
    """`main` must resolve the protocol per call; binding it to the import-time
    DEFAULT_PROTOCOL is what made a 24-base run validate against 15."""
    source = inspect.getsource(driver.main)
    assert "protocol=effective_protocol()" in source
    assert "protocol=DEFAULT_PROTOCOL" not in source
