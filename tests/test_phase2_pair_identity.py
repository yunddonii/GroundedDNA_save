"""The F01 delta must prove the two sides are one run, not assume it.

The table's whole claim is "the SAME checkpoint, with the epoch restored". The
aggregator validated the legacy manifests and then threw them away, comparing
only the analysis sources and the GC window, so two different checkpoints would
have been differenced without a word (§27.6 finding 1).

Two other omissions in the same report: `full_map_pre_projection` was validated
and sealed by the contract and then dropped from every delta and mean, and the
provenance block recorded only the fixed side -- leaving the reader unable to
see the axis the pair actually differs along.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.aggregate_phase2_f01 import (  # noqa: E402
    _PAIR_INVARIANT,
    _REPORTED_METRICS,
    _pair_identity,
    expected_protocol,
)
from dna_utils.gc_policy import resolve_gc_policy  # noqa: E402

REPORT = REPO / "docs" / "phase2_f01_impact.json"


def _manifests(**overrides) -> dict:
    base = {
        "checkpoint_sha256": "a" * 64, "config_sha256": "b" * 64,
        "dataset": "cifar10", "random_seed": 42, "codebook_size": 64,
        "num_slots": 5, "bases_per_slot": 3, "total_bases": 15,
        "total_bits": 30, "lr_schedule_horizon": 5,
        "training_epoch_budget": 5, "training_stop_epoch": 4,
        "sinkhorn_schedule_horizon": 5, "sinkhorn_annealing_enabled": True,
        "n_rows": 1000,
    }
    base.update(overrides)
    return {"db": dict(base), "query": dict(base)}


def test_one_run_differing_only_along_f01_is_a_pair():
    """Epoch, its source and the epsilon that follows are allowed to differ."""
    fixed = _manifests()
    legacy = _manifests()
    same, differing = _pair_identity(fixed, legacy)
    assert same, differing


@pytest.mark.parametrize("field,value", [
    ("checkpoint_sha256", "c" * 64),
    ("config_sha256", "d" * 64),
    ("random_seed", 43),
    ("codebook_size", 128),
    ("total_bases", 24),
    ("training_stop_epoch", 9),
    ("sinkhorn_annealing_enabled", False),
    ("n_rows", 999),
])
def test_a_difference_outside_the_f01_axis_breaks_the_pair(field, value):
    same, differing = _pair_identity(_manifests(), _manifests(**{field: value}))
    assert not same
    assert any(field in entry for entry in differing)


def test_the_epoch_axis_itself_is_not_part_of_the_invariant():
    """F01 IS the epoch, so requiring it to match would refuse every pair."""
    assert "inference_epoch" not in _PAIR_INVARIANT
    assert "inference_epoch_source" not in _PAIR_INVARIANT
    assert "effective_sinkhorn_epsilon" not in _PAIR_INVARIANT


def test_the_expected_protocol_pins_the_gc_fractions_and_nmi_convention():
    """Side-vs-side equality shows the two agree, not that either is correct."""
    protocol = expected_protocol("cifar10", resolve_gc_policy(15))
    assert protocol["gc_min_frac"] == pytest.approx(0.4)
    assert protocol["gc_max_frac"] == pytest.approx(0.6)
    assert protocol["nmi_average_method"] == "arithmetic"


# ------------------------------------------------ the published report

def _report() -> dict:
    if not REPORT.is_file():
        pytest.skip("the Phase 2 report has not been generated")
    return json.loads(REPORT.read_text())


def test_the_published_report_verified_every_pair():
    """Proof the gate is wired to production, not merely defined."""
    payload = _report()
    assert payload["cells"], "no cells in the report"
    for cell in payload["cells"]:
        assert cell["provenance"]["pair_identity_verified"] is True


def test_the_published_report_carries_the_fifth_metric():
    payload = _report()
    assert "full_map_pre_projection" in _REPORTED_METRICS
    assert "full_map_pre_projection" in payload["mean_delta"]
    for cell in payload["cells"]:
        assert cell["delta"]["full_map_pre_projection"] is not None


def test_the_published_report_shows_the_f01_axis_on_both_sides():
    for cell in _report()["cells"]:
        fixed = cell["provenance"]["phase2"]
        legacy = cell["provenance"]["legacy"]
        assert legacy["inference_epoch"] == 0
        assert legacy["inference_epoch_source"] == "f01_unrestored"
        assert fixed["inference_epoch"] == cell["N"]
        assert fixed["inference_epoch_source"] == "explicit_flag"
        # The defect: the legacy side sat at the schedule's initial epsilon.
        assert legacy["effective_sinkhorn_epsilon"] > \
            fixed["effective_sinkhorn_epsilon"]
