"""Fail-closed production contract for pairwise codebook NMI."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import scripts.pairwise_nmi as nmi_script  # noqa: E402
from dna_utils.extraction_validation import ExtractionInvalid  # noqa: E402
from tests.test_extraction_run_validation import _cell  # noqa: E402

SCRIPT = REPO / "scripts" / "pairwise_nmi.py"


def _run(*arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments], capture_output=True,
        text=True, cwd="/", timeout=300)


def _new_cell(path: Path) -> Path:
    path.mkdir(parents=True)
    return _cell(path)


def test_honest_result_is_strict_bound_json(tmp_path):
    cell = _new_cell(tmp_path / "cell")
    process = _run("--results", str(cell))
    assert process.returncode == 0, process.stdout + process.stderr
    output = cell / "pairwise_nmi.json"
    payload = json.loads(
        output.read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )
    assert payload["schema_version"] == 3
    assert payload["input_binding"]["dataset"] == "CIFAR10"
    artifact = payload["codebook_input_artifact"]
    assert artifact["path"] == str((cell / "extract_db.npz").resolve())
    assert artifact["resolved_path"] == artifact["path"]
    assert artifact["sha256"] == payload["input_binding"]["npz_sha256"]["db"]
    assert payload["N"] == 4
    assert payload["num_codebooks"] == 5
    assert len(payload["nmi_matrix"]) == 5
    assert len(payload["unique_codewords_per_cb"]) == 5
    assert 0.0 <= payload["mean_off_diag_nmi"] <= 1.0


def test_main_three_split_transaction_requires_explicit_flag(tmp_path):
    path = tmp_path / "main"
    path.mkdir()
    cell = _cell(path, splits=("db", "query", "train"))
    assert _run("--results", str(cell)).returncode != 0
    process = _run("--results", str(cell), "--require-train")
    assert process.returncode == 0, process.stdout + process.stderr
    payload = json.loads((cell / "pairwise_nmi.json").read_text())
    assert set(payload["input_binding"]["npz_sha256"]) == {
        "db", "query", "train"}


def test_missing_later_result_publishes_no_partial_batch(tmp_path):
    first = _new_cell(tmp_path / "first")
    missing = tmp_path / "missing"
    process = _run("--results", str(first), str(missing))
    assert process.returncode != 0
    assert not (first / "pairwise_nmi.json").exists()


def test_duplicate_combined_key_is_refused_before_publication(tmp_path):
    first = _new_cell(tmp_path / "a" / "same")
    second = _new_cell(tmp_path / "b" / "same")
    process = _run("--results", str(first), str(second))
    assert process.returncode != 0
    assert "same basename" in process.stderr
    assert not (first / "pairwise_nmi.json").exists()
    assert not (second / "pairwise_nmi.json").exists()


def test_combined_output_is_strict_and_does_not_replace_direct_layout(tmp_path):
    first = _new_cell(tmp_path / "first")
    second = _new_cell(tmp_path / "second")
    combined = tmp_path / "combined.json"
    process = _run(
        "--results", str(first), str(second), "--out", str(combined))
    assert process.returncode == 0, process.stdout + process.stderr
    payload = json.loads(combined.read_text(encoding="utf-8"))
    assert set(payload) == {"first", "second"}
    assert json.loads((first / "pairwise_nmi.json").read_text())["N"] == 4
    assert not list(tmp_path.rglob(".pairwise_nmi.json.*.tmp"))


def test_per_result_filename_cannot_be_used_for_combined_output(tmp_path):
    cell = _new_cell(tmp_path / "cell")
    process = _run(
        "--results", str(cell), "--out", str(cell / "pairwise_nmi.json"))
    assert process.returncode != 0
    assert not (cell / "pairwise_nmi.json").exists()


def test_publishing_new_nmi_removes_stale_analysis_marker(tmp_path):
    cell = _new_cell(tmp_path / "cell")
    stale = cell / "analysis_complete.json"
    stale.write_text('{"stale": true}\n', encoding="utf-8")
    process = _run("--results", str(cell))
    assert process.returncode == 0, process.stdout + process.stderr
    assert not stale.exists()


def test_in_place_drift_during_nmi_is_refused(tmp_path, monkeypatch):
    cell = _new_cell(tmp_path / "cell")
    npz = cell / "extract_db.npz"
    real = nmi_script.pairwise_nmi

    def mutate_after_load(indices):
        result = real(indices)
        npz.write_bytes(npz.read_bytes() + b"drift")
        return result

    monkeypatch.setattr(nmi_script, "pairwise_nmi", mutate_after_load)
    with pytest.raises((ExtractionInvalid, RuntimeError), match="hashes to|changed"):
        nmi_script._record_for_result(
            str(cell), allow_backfilled=False,
            required_splits=("db", "query"))


def test_same_bytes_path_replacement_during_nmi_is_refused(
        tmp_path, monkeypatch):
    cell = _new_cell(tmp_path / "cell")
    npz = cell / "extract_db.npz"
    real = nmi_script.pairwise_nmi

    def replace_after_load(indices):
        result = real(indices)
        replacement = cell / "replacement.npz"
        replacement.write_bytes(npz.read_bytes())
        os.replace(replacement, npz)
        return result

    monkeypatch.setattr(nmi_script, "pairwise_nmi", replace_after_load)
    with pytest.raises(RuntimeError, match="replaced|changed"):
        nmi_script._record_for_result(
            str(cell), allow_backfilled=False,
            required_splits=("db", "query"))
