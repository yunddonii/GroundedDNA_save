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
import shutil
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dna_utils.extraction_validation import (  # noqa: E402
    ExtractionInvalid,
    metric_input_binding,
    read_analysis_marker,
)
import scripts.eval_cell_bioproj as bio_eval_script  # noqa: E402
from evaluation_siglip2 import evaluation as run_evaluation  # noqa: E402
from scripts.pairwise_nmi import _record_for_result  # noqa: E402
from tests.test_extraction_run_validation import _cell  # noqa: E402

SCRIPT = REPO / "scripts" / "seal_cell_analysis.py"

_EVAL = {
    "dataset": "CIFAR10", "K": 64,
    "mAP_at_R_bioproj": 0.88, "full_mAP_bioproj": 0.82,
    "full_mAP_pre_projection": 0.83, "DNA_unique_DB": 0.087,
    "gc_min_frac": 0.4, "gc_max_frac": 0.6, "total_bases": 15,
    "gc_count_min_inclusive": 6, "gc_count_max_inclusive": 9,
    "bio_max_homopolymer_run": 3,
    "gc_policy_version": "gc-40-60-inclusive-v1", "map_r_cutoff": 1000,
}

_NMI = {
    "nmi_average_method": "arithmetic", "sklearn_version": "1.3.0",
    "mean_off_diag_nmi": 0.60, "max_off_diag_nmi": 0.67,
    "min_off_diag_nmi": 0.55, "N": 100,
}


@pytest.mark.parametrize("stored,canonical", (
    ("cifar10", "CIFAR10"), ("flickr25k", "Flickr25k"),
    ("mscoco", "MSCOCO"), ("nuswide", "NUSWIDE"),
))
def test_bio_evaluator_accepts_manifest_dataset_slugs(stored, canonical):
    assert bio_eval_script._canonical_dataset_name(stored) == canonical


def _halves(cell: Path, *, evaluation=True, nmi=True, binding=None,
            required_splits=("db", "query")) -> None:
    bound = binding if binding is not None else metric_input_binding(
        str(cell), required_splits=required_splits)
    raw = run_evaluation(
        str(cell), distance_mode="base", codebook_size=64, map_at_r=1000,
        dataset_name="CIFAR10")
    post = run_evaluation(
        str(cell), distance_mode="base", codebook_size=64, map_at_r=1000,
        dataset_name="CIFAR10", bio_project=True,
        bio_gc_min_frac=0.4, bio_gc_max_frac=0.6,
        bio_max_homopolymer_run=3)
    _, raw_artifact = bio_eval_script._read_json_artifact(
        str(cell / "evaluation_siglip2_base.json"))
    _, post_artifact = bio_eval_script._read_json_artifact(
        str(cell / "evaluation_siglip2_base_bioproj.json"))
    if evaluation:
        (cell / "cell_result.json").write_text(
            json.dumps({
                "input_binding": bound,
                "evaluation_artifacts": {
                    "raw": raw_artifact, "bio_projected": post_artifact},
                "evaluator_input_artifacts": post["input_artifacts"],
                "bio_stats": post["bio_stats"],
                "dataset": "CIFAR10", "K": 64,
                "mAP_at_R_bioproj": post["mAP_at_R"],
                "full_mAP_bioproj": post["mAP"],
                "full_mAP_pre_projection": raw["mAP"],
                "DNA_unique_DB": post["unique_code_ratio"],
                "gc_min_frac": 0.4, "gc_max_frac": 0.6,
                "total_bases": 15,
                "gc_count_min_inclusive": 6,
                "gc_count_max_inclusive": 9,
                "bio_max_homopolymer_run": 3,
                "gc_policy_version": "gc-40-60-inclusive-v1",
                "map_r_cutoff": 1000,
            }, indent=2))
    if nmi:
        record, _, _ = _record_for_result(
            str(cell), allow_backfilled=False,
            required_splits=required_splits)
        (cell / "pairwise_nmi.json").write_text(
            json.dumps(record, indent=2))


def _seal(cell: Path, *, require_train=False) -> subprocess.CompletedProcess:
    command = [sys.executable, str(SCRIPT), "--dir", str(cell)]
    if require_train:
        command.append("--require-train")
    return subprocess.run(
        command,
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
    nmi = json.loads((cell / "pairwise_nmi.json").read_text())
    assert payload["metrics"]["mean_off_diag_nmi"] == pytest.approx(
        nmi["mean_off_diag_nmi"])
    # The settings that decide the NMI travel with it.
    assert payload["protocol"]["nmi_average_method"] == "arithmetic"
    assert payload["protocol"]["sklearn_version"] == nmi["sklearn_version"]
    assert payload["scientific_authority"] is False
    assert payload["paper_result_eligible"] is False
    assert payload["standalone_authority"][
        "requires_downstream_deterministic_recomputation"] is True
    assert set(payload["evidence_artifacts"]) == {
        "raw", "bio_projected", "cell_result", "pairwise_nmi"}
    assert set(payload["extraction_evidence"]["npz_artifacts"]) == {
        "db", "query"}


def test_main_requires_one_three_split_transaction(tmp_path):
    cell = _cell(tmp_path, splits=("db", "query", "train"))
    _halves(cell, required_splits=("db", "query", "train"))
    # A caller must explicitly select the MAIN contract; the diagnostic
    # two-split default cannot reinterpret a three-split marker.
    assert _seal(cell).returncode == 1
    process = _seal(cell, require_train=True)
    assert process.returncode == 0, process.stdout + process.stderr
    payload = json.loads((cell / "analysis_complete.json").read_text())
    assert set(payload["input_binding"]["npz_sha256"]) == {
        "db", "query", "train"}
    with pytest.raises(ExtractionInvalid):
        read_analysis_marker(str(cell), allow_backfilled=False)
    reopened = read_analysis_marker(
        str(cell), allow_backfilled=False,
        required_splits=("db", "query", "train"))
    assert set(reopened["input_binding"]["npz_sha256"]) == {
        "db", "query", "train"}


def test_bio_stage_binds_train_split_before_publishing_main_half(
        tmp_path, monkeypatch):
    cell = _cell(tmp_path, splits=("db", "query", "train"))
    run_evaluation(
        str(cell), distance_mode="base", codebook_size=64, map_at_r=1000,
        dataset_name="CIFAR10")
    monkeypatch.setattr(sys, "argv", [
        str(bio_eval_script.__file__), "--dir", str(cell),
        "--dataset", "CIFAR10", "--K", "64", "--require-train",
    ])
    bio_eval_script.main()
    payload = json.loads((cell / "cell_result.json").read_text())
    assert set(payload["input_binding"]["npz_sha256"]) == {
        "db", "query", "train"}
    assert payload["bio_stats"]
    assert set(payload["evaluation_artifacts"]) == {"raw", "bio_projected"}


def test_bio_stage_refuses_caller_bases_that_disagree_with_extraction(
        tmp_path, monkeypatch):
    cell = _cell(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        str(bio_eval_script.__file__), "--dir", str(cell),
        "--dataset", "CIFAR10", "--K", "64", "--bases", "24",
    ])
    with pytest.raises(SystemExit, match="actual extraction width 15"):
        bio_eval_script.main()


@pytest.mark.parametrize("attack", ["missing", "tampered", "external"])
def test_bio_stage_refuses_evaluator_input_binding_attack(
        tmp_path, monkeypatch, attack):
    cell_root = tmp_path / "cell"
    cell_root.mkdir()
    cell = _cell(cell_root)
    run_evaluation(
        str(cell), distance_mode="base", codebook_size=64, map_at_r=1000,
        dataset_name="CIFAR10")

    def attacked_evaluation(*args, **kwargs):
        result = run_evaluation(*args, **kwargs)
        if attack == "missing":
            result.pop("input_artifacts")
        elif attack == "tampered":
            result["input_artifacts"]["db"]["sha256"] = "0" * 64
        else:
            outside = tmp_path / "outside.npz"
            shutil.copyfile(cell / "extract_db.npz", outside)
            _, external = bio_eval_script._load_npz_bound(str(outside))
            result["input_artifacts"]["db"] = external
        return result

    monkeypatch.setattr(bio_eval_script, "evaluation", attacked_evaluation)
    monkeypatch.setattr(sys, "argv", [
        str(bio_eval_script.__file__), "--dir", str(cell),
        "--dataset", "CIFAR10", "--K", "64",
    ])
    with pytest.raises(RuntimeError, match="input_artifacts"):
        bio_eval_script.main()
    assert not (cell / "cell_result.json").exists()


def test_the_evaluation_alone_cannot_seal(tmp_path):
    """Exactly what the bio-evaluation used to do on its own."""
    cell = _cell(tmp_path)
    _halves(cell, nmi=False)
    proc = _seal(cell)
    assert proc.returncode == 1
    assert "pairwise_nmi.json" in proc.stdout
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


def test_swapped_nmi_from_another_cell_is_refused(tmp_path):
    first_root, second_root = tmp_path / "first", tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first = _cell(first_root)
    second = _cell(second_root)
    _halves(first)
    _halves(second)
    shutil.copyfile(second / "pairwise_nmi.json", first / "pairwise_nmi.json")
    process = _seal(first)
    assert process.returncode == 1
    assert "names run_dir" in process.stdout or "different inputs" in process.stdout
    assert not (first / "analysis_complete.json").exists()


def test_changed_raw_json_cannot_hide_behind_old_cell_binding(tmp_path):
    cell = _cell(tmp_path)
    _halves(cell)
    raw_path = cell / "evaluation_siglip2_base.json"
    raw = json.loads(raw_path.read_text())
    raw["mAP"] = 0.123456
    raw_path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n")
    process = _seal(cell)
    assert process.returncode == 1
    assert "exact raw/post" in process.stdout
    assert not (cell / "analysis_complete.json").exists()


def test_consistently_resigned_finite_metrics_remain_non_authoritative(tmp_path):
    """Integrity signing is not deterministic scientific recomputation."""
    cell = _cell(tmp_path)
    _halves(cell)
    raw_path = cell / "evaluation_siglip2_base.json"
    post_path = cell / "evaluation_siglip2_base_bioproj.json"
    result_path = cell / "cell_result.json"
    raw = json.loads(raw_path.read_text())
    post = json.loads(post_path.read_text())
    result = json.loads(result_path.read_text())

    raw["mAP"], raw["mAP_at_R"] = 0.21, 0.22
    post["mAP"], post["mAP_at_R"], post["unique_code_ratio"] = (
        0.31, 0.32, 0.33)
    post["bio_stats"]["mAP_pre_projection"] = raw["mAP"]
    post["bio_stats"]["mAP_at_R_pre_projection"] = raw["mAP_at_R"]
    raw_path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n")
    post_path.write_text(json.dumps(post, indent=2, sort_keys=True) + "\n")
    _, raw_artifact = bio_eval_script._read_json_artifact(str(raw_path))
    _, post_artifact = bio_eval_script._read_json_artifact(str(post_path))
    result.update({
        "evaluation_artifacts": {
            "raw": raw_artifact, "bio_projected": post_artifact},
        "bio_stats": post["bio_stats"],
        "mAP_at_R_bioproj": post["mAP_at_R"],
        "full_mAP_bioproj": post["mAP"],
        "full_mAP_pre_projection": raw["mAP"],
        "DNA_unique_DB": post["unique_code_ratio"],
    })
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")

    process = _seal(cell)
    assert process.returncode == 0, process.stdout + process.stderr
    marker = json.loads((cell / "analysis_complete.json").read_text())
    assert marker["metrics"]["map_at_R_bioproj"] == 0.32
    assert marker["scientific_authority"] is False
    assert marker["paper_result_eligible"] is False
    assert marker["standalone_authority"]["scope"] == (
        "integrity_only_no_independent_metric_recomputation")


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
    assert "arithmetic/sklearn" in proc.stdout


def test_the_seal_runs_from_any_cwd(tmp_path):
    """It is invoked by launchers whose cwd is not the repo."""
    cell = _cell(tmp_path)
    _halves(cell)
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--dir", str(cell)],
        capture_output=True, text=True, cwd=str(tmp_path), timeout=300)
    assert proc.returncode == 0, proc.stdout + proc.stderr
