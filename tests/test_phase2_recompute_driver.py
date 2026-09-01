"""The Phase-2 driver runs the four metric stages in one fail-closed order."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


REPO = Path(__file__).resolve().parents[1]
DRIVER = REPO / "scripts" / "phase2_recompute_metrics.sh"
CELLS = (
    "cifar10_N4", "cifar10_N9", "cifar10_N19", "cifar10_N39",
    "flickr25k_N4", "flickr25k_N9", "flickr25k_N19",
    "nuswide_N4", "nuswide_N9", "nuswide_N19", "nuswide_N39",
    "mscoco_N4", "mscoco_N9", "mscoco_N19", "mscoco_N39",
)

_STUB = r'''#!/usr/bin/env python3
import json, os, pathlib, sys
name = os.path.basename(sys.argv[1])
with open(os.environ["STUB_CALLS"], "a") as handle:
    handle.write(json.dumps([name, *sys.argv[2:]]) + "\n")
if os.environ.get("STUB_FAIL") == name:
    raise SystemExit(7)
def value(flag):
    return sys.argv[sys.argv.index(flag) + 1]
if name == "seal_cell_analysis.py":
    pathlib.Path(value("--dir"), "analysis_complete.json").write_text("{}\n")
elif name == "_phase2_cell_state.py":
    print("complete")
elif name == "aggregate_phase2_f01.py":
    if "--seal-input-pair" in sys.argv:
        pathlib.Path(value("--pair-receipt")).write_text("{}\n")
    elif "--verify-report-receipt" in sys.argv:
        pass
    else:
        for flag in ("--out-json", "--out-md", "--out-receipt"):
            pathlib.Path(value(flag)).write_text("{}\n")
'''


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    sandbox = tmp_path / "repo"
    (sandbox / "scripts").mkdir(parents=True)
    copied = sandbox / "scripts" / DRIVER.name
    shutil.copy2(DRIVER, copied)
    copied.chmod(0o755)
    subprocess.run(["git", "init", "-q"], cwd=sandbox, check=True)
    subprocess.run(["git", "add", "scripts"], cwd=sandbox, check=True)
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.com",
         "commit", "-qm", "fixture"], cwd=sandbox, check=True)
    fixed, legacy = tmp_path / "fixed", tmp_path / "legacy"
    for root in (fixed, legacy):
        for cell in CELLS:
            (root / cell).mkdir(parents=True)
    stub = tmp_path / "python-stub"
    stub.write_text(_STUB)
    stub.chmod(0o755)
    return sandbox, fixed, legacy, stub


def _run(tmp_path: Path, *, fail: str | None = None):
    sandbox, fixed, legacy, stub = _fixture(tmp_path)
    calls, logs, report = tmp_path / "calls.jsonl", tmp_path / "logs", tmp_path / "report"
    env = dict(os.environ, STUB_CALLS=str(calls))
    if fail:
        env["STUB_FAIL"] = fail
    process = subprocess.run(
        [str(sandbox / "scripts" / DRIVER.name),
         "--fixed-root", str(fixed), "--legacy-root", str(legacy),
         "--log-root", str(logs), "--report-root", str(report),
         "--python", str(stub), "--jobs", "4"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
    recorded = [json.loads(line) for line in calls.read_text().splitlines()]
    return process, recorded, report


def test_driver_runs_raw_bio_nmi_seal_then_receipts(tmp_path):
    process, calls, report = _run(tmp_path)
    assert process.returncode == 0, process.stdout + process.stderr
    stages = [entry[0] for entry in calls]
    assert stages.count("evaluation_siglip2.py") == 30
    assert stages.count("eval_cell_bioproj.py") == 30
    assert stages.count("pairwise_nmi.py") == 30
    assert stages.count("seal_cell_analysis.py") == 30
    assert stages.count("aggregate_phase2_f01.py") == 3
    verify_call = next(
        entry for entry in calls
        if entry[0] == "aggregate_phase2_f01.py"
        and "--verify-report-receipt" in entry)
    expected_paths = {
        "--expected-fixed-root": str((tmp_path / "fixed").resolve()),
        "--expected-legacy-root": str((tmp_path / "legacy").resolve()),
        "--expected-pair-receipt": str(
            (report / "phase2_input_pair_receipt.json").resolve()),
        "--expected-report-json": str(
            (report / "phase2_f01_impact.json").resolve()),
        "--expected-report-md": str(
            (report / "phase2_f01_impact.md").resolve()),
    }
    for flag, expected in expected_paths.items():
        assert verify_call[verify_call.index(flag) + 1] == expected
    for side in ("fixed", "legacy"):
        for cell in CELLS:
            per_cell = [entry[0] for entry in calls if any(
                str(value).endswith(f"/{side}/{cell}") for value in entry)]
            assert per_cell == [
                "evaluation_siglip2.py", "eval_cell_bioproj.py",
                "pairwise_nmi.py", "seal_cell_analysis.py",
                "_phase2_cell_state.py"]
    final_aggregate = max(
        index for index, entry in enumerate(calls)
        if entry[0] == "aggregate_phase2_f01.py")
    assert final_aggregate > max(
        index for index, entry in enumerate(calls)
        if entry[0] == "seal_cell_analysis.py")
    assert (report / "phase2_input_pair_receipt.json").is_file()
    assert (report / "phase2_f01_receipt.json").is_file()


def test_driver_failure_never_reaches_final_aggregate(tmp_path):
    process, calls, report = _run(tmp_path, fail="eval_cell_bioproj.py")
    assert process.returncode != 0
    stages = [entry[0] for entry in calls]
    assert "evaluation_siglip2.py" in stages
    assert "seal_cell_analysis.py" not in stages
    assert stages.count("aggregate_phase2_f01.py") == 1
    assert not (report / "phase2_f01_receipt.json").exists()


def test_driver_refuses_overlapping_output_roots(tmp_path):
    sandbox, fixed, legacy, stub = _fixture(tmp_path)
    process = subprocess.run(
        [str(sandbox / "scripts" / DRIVER.name),
         "--fixed-root", str(fixed), "--legacy-root", str(legacy),
         "--log-root", str(fixed / "logs"),
         "--report-root", str(tmp_path / "report"),
         "--python", str(stub)],
        cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert process.returncode == 2
    assert "must be disjoint" in process.stderr
    assert not (fixed / "logs").exists()


def test_cell_runner_refuses_dangling_log_symlink(tmp_path):
    sandbox, fixed, _, stub = _fixture(tmp_path)
    target = tmp_path / "must-not-be-created"
    log = tmp_path / "cell.log"
    log.symlink_to(target)
    process = subprocess.run(
        [str(sandbox / "scripts" / DRIVER.name), "--run-cell", str(sandbox),
         str(stub), "fixed", str(fixed / CELLS[0]), "CIFAR10", "64",
         str(log)],
        cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert process.returncode == 65
    assert log.is_symlink() and not target.exists()
