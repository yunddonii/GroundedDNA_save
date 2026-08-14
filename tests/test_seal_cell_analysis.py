"""The seal is published only when both analysis halves exist and agree.

`scripts/eval_cell_bioproj.py` produces the retrieval and DNA numbers;
`scripts/pairwise_nmi.py` produces the codebook redundancy. The evaluation used
to seal `analysis_complete.json` by itself, so the sealed metric set had no
`mean_off_diag_nmi` and a cell with half an analysis read as a finished one
(§20.4). `scripts/seal_cell_analysis.py` is the stage that can see both.

These run the real script as a subprocess against real extraction fixtures.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dna_utils.extraction_validation import metric_input_binding  # noqa: E402
from tests.test_extraction_run_validation import _cell  # noqa: E402

SCRIPT = REPO / "scripts" / "seal_cell_analysis.py"

_EVAL = {
    "dataset": "CIFAR10", "K": 64,
    "mAP_at_R_bioproj": 0.88, "full_mAP_bioproj": 0.82,
    "full_mAP_pre_projection": 0.83, "DNA_unique_DB": 0.087,
    "gc_min_frac": 0.4, "gc_max_frac": 0.6, "total_bases": 15,
    "gc_count_min_inclusive": 6, "gc_count_max_inclusive": 9,
    "gc_policy_version": "gc-40-60-inclusive-v1", "map_r_cutoff": 1000,
}

_NMI = {
    "nmi_average_method": "arithmetic", "sklearn_version": "1.3.0",
    "mean_off_diag_nmi": 0.60, "max_off_diag_nmi": 0.67,
    "min_off_diag_nmi": 0.55, "N": 100,
}


def _halves(cell: Path, *, evaluation=True, nmi=True, binding=None) -> None:
    bound = binding if binding is not None else metric_input_binding(str(cell))
    if evaluation:
        (cell / "cell_result.json").write_text(
            json.dumps({"input_binding": bound, **_EVAL}, indent=2))
    if nmi:
        (cell / "pairwise_nmi.json").write_text(
            json.dumps({"input_binding": bound, **_NMI}, indent=2))


def _seal(cell: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--dir", str(cell)],
        capture_output=True, text=True, cwd=os.environ.get("HOME", "/"),
        timeout=300)


def test_both_halves_present_seals_the_full_metric_set(tmp_path):
    cell = _cell(tmp_path)
    _halves(cell)
    proc = _seal(cell)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads((cell / "analysis_complete.json").read_text())
    assert set(payload["metrics"]) == {
        "map_at_R_bioproj", "full_map_bioproj", "full_map_pre_projection",
        "dna_unique_db", "mean_off_diag_nmi"}
    assert payload["metrics"]["mean_off_diag_nmi"] == pytest.approx(0.60)
    # The settings that decide the NMI travel with it.
    assert payload["protocol"]["nmi_average_method"] == "arithmetic"
    assert payload["protocol"]["sklearn_version"] == "1.3.0"


def test_the_evaluation_alone_cannot_seal(tmp_path):
    """Exactly what the bio-evaluation used to do on its own."""
    cell = _cell(tmp_path)
    _halves(cell, nmi=False)
    proc = _seal(cell)
    assert proc.returncode == 1
    assert "pairwise_nmi.json is missing" in proc.stdout
    assert not (cell / "analysis_complete.json").exists()


def test_the_nmi_alone_cannot_seal(tmp_path):
    cell = _cell(tmp_path)
    _halves(cell, evaluation=False)
    assert _seal(cell).returncode == 1
    assert not (cell / "analysis_complete.json").exists()


def test_halves_describing_different_inputs_are_refused(tmp_path):
    """Two forward passes cannot be sealed into one result."""
    cell = _cell(tmp_path)
    _halves(cell)
    other = metric_input_binding(str(cell))
    other["npz_sha256"]["db"] = "0" * 64
    (cell / "pairwise_nmi.json").write_text(
        json.dumps({"input_binding": other, **_NMI}, indent=2))
    proc = _seal(cell)
    assert proc.returncode == 1
    assert not (cell / "analysis_complete.json").exists()


def test_a_refusal_removes_a_stale_seal(tmp_path):
    """A seal describing superseded numbers must not survive a failed re-seal."""
    cell = _cell(tmp_path)
    _halves(cell)
    assert _seal(cell).returncode == 0
    marker = cell / "analysis_complete.json"
    assert marker.exists()

    (cell / "pairwise_nmi.json").unlink()
    assert _seal(cell).returncode == 1
    assert not marker.exists(), "the old seal outlived the numbers it described"


def test_an_nmi_without_its_averaging_method_is_refused(tmp_path):
    """A bare number cannot be interpreted, so it cannot be published."""
    cell = _cell(tmp_path)
    _halves(cell)
    payload = json.loads((cell / "pairwise_nmi.json").read_text())
    del payload["nmi_average_method"]
    (cell / "pairwise_nmi.json").write_text(json.dumps(payload, indent=2))
    proc = _seal(cell)
    assert proc.returncode == 1
    assert "averaging method" in proc.stdout


def test_the_seal_runs_from_any_cwd(tmp_path):
    """It is invoked by launchers whose cwd is not the repo."""
    cell = _cell(tmp_path)
    _halves(cell)
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--dir", str(cell)],
        capture_output=True, text=True, cwd=str(tmp_path), timeout=300)
    assert proc.returncode == 0, proc.stdout + proc.stderr
