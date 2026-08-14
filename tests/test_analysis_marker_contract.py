"""What a sealed analysis marker has to prove before its numbers are read.

Three counterexamples from the re-audit, all of which the previous contract
admitted:

* **§20.2** — the reader checked that `metrics`, `protocol` and
  `analysis_sources` were mappings and stopped. A probe carrying
  `map_at_R_bioproj: 999`, `protocol.dataset: "WRONG"` and
  `analysis_sources: {"sha": "garbage"}` was admitted, and the aggregator
  reported all four fabricated numbers with `fixed_side_sealed: true`.

* **§20.3** — `check_metric_input_binding` derived its `allow_backfilled` from
  the artefact's own `backfilled_inputs` field, so a marker that declared
  itself retrospectively bound authorised its own admission, defeating a reader
  that had explicitly asked for `allow_backfilled=False`.

* **§20.4** — the bio-evaluation sealed the marker before the NMI ran, so
  `mean_off_diag_nmi` was absent from a file named `analysis_complete`, and the
  aggregator counted the cell as paired anyway.

The fixtures are real extraction runs built by the shared helper, so these
exercise the production writer and the production reader, not a mock of either.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dna_utils.extraction_validation import (  # noqa: E402
    ExtractionInvalid,
    analysis_source_digests,
    check_metric_input_binding,
    metric_input_binding,
    read_analysis_marker,
    write_analysis_marker,
)
from tests.test_extraction_run_validation import _cell  # noqa: E402

_METRICS = {
    "map_at_R_bioproj": 0.88,
    "full_map_bioproj": 0.82,
    "full_map_pre_projection": 0.83,
    "dna_unique_db": 0.087,
    "mean_off_diag_nmi": 0.60,
}

_PROTOCOL = {
    "dataset": "CIFAR10",
    "codebook_size": 64,
    "total_bases": 15,
    "map_r_cutoff": 1000,
    "gc_policy_version": "gc-40-60-inclusive-v1",
    "gc_count_min_inclusive": 6,
    "gc_count_max_inclusive": 9,
    "gc_min_frac": 0.4,
    "gc_max_frac": 0.6,
    "nmi_average_method": "arithmetic",
    "sklearn_version": "1.3.0",
}


def _seal(cell: Path, *, metrics=None, protocol=None, sources=None) -> str:
    return write_analysis_marker(
        str(cell),
        metrics=dict(_METRICS if metrics is None else metrics),
        binding=metric_input_binding(str(cell)),
        protocol=dict(_PROTOCOL if protocol is None else protocol),
        sources=dict(analysis_source_digests() if sources is None else sources))


def _rewrite(cell: Path, **changes) -> None:
    """Plant a marker on disk, bypassing the writer's own checks."""
    path = cell / "analysis_complete.json"
    payload = json.loads(path.read_text())
    payload.update(changes)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def test_a_well_formed_marker_round_trips(tmp_path):
    cell = _cell(tmp_path)
    _seal(cell)
    payload = read_analysis_marker(str(cell), allow_backfilled=False)
    assert payload["metrics"]["mean_off_diag_nmi"] == pytest.approx(0.60)


# ------------------------------------------------------------------ §20.4

def test_sealing_without_the_nmi_is_refused(tmp_path):
    """The exact shape the bio-evaluation used to write on its own."""
    cell = _cell(tmp_path)
    partial = {k: v for k, v in _METRICS.items() if k != "mean_off_diag_nmi"}
    with pytest.raises(ExtractionInvalid) as excinfo:
        _seal(cell, metrics=partial)
    assert "mean_off_diag_nmi" in str(excinfo.value)
    assert not (cell / "analysis_complete.json").exists()


def test_a_planted_marker_missing_the_nmi_is_refused_on_read(tmp_path):
    cell = _cell(tmp_path)
    _seal(cell)
    _rewrite(cell, metrics={k: v for k, v in _METRICS.items()
                            if k != "mean_off_diag_nmi"})
    with pytest.raises(ExtractionInvalid):
        read_analysis_marker(str(cell), allow_backfilled=False)


# ------------------------------------------------------------------ §20.2

def test_a_fabricated_metric_is_refused(tmp_path):
    """`map_at_R_bioproj: 999` was previously read straight into the report."""
    cell = _cell(tmp_path)
    _seal(cell)
    _rewrite(cell, metrics={**_METRICS, "map_at_R_bioproj": 999})
    with pytest.raises(ExtractionInvalid) as excinfo:
        read_analysis_marker(str(cell), allow_backfilled=False)
    assert "[0, 1]" in str(excinfo.value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "0.8", None,
                                   True])
def test_a_metric_that_is_not_a_finite_number_is_refused(tmp_path, value):
    cell = _cell(tmp_path)
    with pytest.raises(ExtractionInvalid):
        _seal(cell, metrics={**_METRICS, "dna_unique_db": value})


def test_an_extra_metric_is_refused(tmp_path):
    """The set is fixed, so two cells are always comparable."""
    cell = _cell(tmp_path)
    with pytest.raises(ExtractionInvalid):
        _seal(cell, metrics={**_METRICS, "map_at_R_something_else": 0.5})


def test_a_protocol_the_reader_does_not_expect_is_refused(tmp_path):
    """`protocol.dataset: "WRONG"` was previously accepted verbatim."""
    cell = _cell(tmp_path)
    _seal(cell, protocol={**_PROTOCOL, "dataset": "WRONG"})
    with pytest.raises(ExtractionInvalid) as excinfo:
        read_analysis_marker(str(cell), allow_backfilled=False,
                             expected_protocol={"dataset": "CIFAR10"})
    assert "dataset" in str(excinfo.value)


def test_a_changed_gc_window_is_refused_against_the_expected_protocol(tmp_path):
    """Two cells scored under different GC windows do not form a delta."""
    cell = _cell(tmp_path)
    _seal(cell, protocol={**_PROTOCOL, "gc_count_max_inclusive": 12})
    with pytest.raises(ExtractionInvalid):
        read_analysis_marker(
            str(cell), allow_backfilled=False,
            expected_protocol={"gc_count_max_inclusive": 9})


def test_an_incomplete_protocol_is_refused_at_write_time(tmp_path):
    cell = _cell(tmp_path)
    partial = {k: v for k, v in _PROTOCOL.items()
               if k != "nmi_average_method"}
    with pytest.raises(ExtractionInvalid) as excinfo:
        _seal(cell, protocol=partial)
    assert "nmi_average_method" in str(excinfo.value)


def test_garbage_source_digests_are_refused(tmp_path):
    """`analysis_sources: {"sha": "garbage"}` was previously admitted."""
    cell = _cell(tmp_path)
    with pytest.raises(ExtractionInvalid):
        _seal(cell, sources={"sha": "garbage"})


def test_a_marker_that_outlived_its_evaluator_is_refused(tmp_path):
    """Numbers cannot survive a change to the code that produced them."""
    cell = _cell(tmp_path)
    _seal(cell)
    stale = dict(analysis_source_digests())
    stale["evaluation_siglip2_sha256"] = "0" * 64
    _rewrite(cell, analysis_sources=stale)
    with pytest.raises(ExtractionInvalid) as excinfo:
        read_analysis_marker(str(cell), allow_backfilled=False)
    assert "evaluation_siglip2_sha256" in str(excinfo.value)


def test_the_recorded_sources_cover_every_file_that_decides_a_number(tmp_path):
    digests = analysis_source_digests()
    assert {"eval_cell_bioproj_sha256", "evaluation_siglip2_sha256",
            "pairwise_nmi_sha256", "bio_constraints_sha256",
            "gc_policy_sha256", "dna_code_utils_sha256"} == set(digests)
    assert all(len(v) == 64 for v in digests.values())


# ------------------------------------------------------------------ §20.3

def test_an_artefact_cannot_authorise_its_own_backfill(tmp_path):
    """The reader's policy wins, not the field the artefact wrote itself."""
    cell = _cell(tmp_path)
    _seal(cell)
    payload = json.loads((cell / "analysis_complete.json").read_text())
    binding = dict(payload["input_binding"])
    binding["backfilled_inputs"] = True
    with pytest.raises(ExtractionInvalid) as excinfo:
        check_metric_input_binding(str(cell), {"input_binding": binding},
                                   what="probe", allow_backfilled=False)
    assert "backfilled" in str(excinfo.value)


def test_the_reader_must_state_its_trust_policy(tmp_path):
    """No default: a caller that forgets gets a TypeError, not a silent yes."""
    cell = _cell(tmp_path)
    _seal(cell)
    with pytest.raises(TypeError):
        read_analysis_marker(str(cell))
    with pytest.raises(TypeError):
        check_metric_input_binding(str(cell), {}, what="probe")


# --------------------------------- the checker is not one of the inputs

def test_a_marker_survives_a_change_to_the_validator(tmp_path):
    """Improving the checker must not invalidate every measurement taken.

    `validator_sha256` was compared for equality, so each fix to this module
    made every sealed marker unreadable and forced a full GPU recompute. It is
    recorded -- a reader can see which checker admitted the numbers -- but it
    does not describe what the numbers were computed from, so it is not a
    reason to reject them.
    """
    cell = _cell(tmp_path)
    _seal(cell)
    payload = json.loads((cell / "analysis_complete.json").read_text())
    binding = dict(payload["input_binding"])
    binding["validator_sha256"] = "0" * 64          # a different checker
    check_metric_input_binding(str(cell), {"input_binding": binding},
                               what="probe", allow_backfilled=False)


def test_a_binding_that_names_no_validator_is_refused(tmp_path):
    cell = _cell(tmp_path)
    _seal(cell)
    payload = json.loads((cell / "analysis_complete.json").read_text())
    binding = {k: v for k, v in payload["input_binding"].items()
               if k != "validator_sha256"}
    with pytest.raises(ExtractionInvalid):
        check_metric_input_binding(str(cell), {"input_binding": binding},
                                   what="probe", allow_backfilled=False)


def test_the_distance_implementation_is_a_recorded_source(tmp_path):
    """The retrieval numbers are Hamming distances; that code decides them."""
    assert "dna_code_utils_sha256" in analysis_source_digests()
