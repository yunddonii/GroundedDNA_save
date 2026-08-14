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


def test_24_matrix_resume_call_does_not_raise_typeerror():
    """Call it the way `_manifest_contract_errors` does."""
    from pathlib import Path
    try:
        w24._validate_manifest(Path("/nonexistent-manifest.json"),
                               w24.Key("bee2018", "CIFAR10", 42),
                               verify_hashes=False, protocol=24)
    except TypeError as error:      # the regression
        pytest.fail(f"protocol kwarg still rejected: {error}")
    except Exception:
        pass                        # a missing file is fine; a TypeError is not


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


def test_driver_main_passes_the_protocol_to_both_verifiers():
    source = inspect.getsource(driver.main)
    selection = source[source.index("_verify_selection("):]
    assert "protocol=" in selection[:600], (
        "main() must pass the protocol to _verify_selection")
    refit = source[source.index("_verify_refit("):]
    assert "protocol=" in refit[:600], (
        "main() must pass the protocol to _verify_refit")


def test_refit_verifier_forwards_protocol_to_extraction_validator():
    source = inspect.getsource(driver._verify_refit)
    call = source[source.index("_verify_aligned_extractions("):]
    assert "protocol=" in call[:800], (
        "_verify_refit must forward its protocol to the extraction validator")
