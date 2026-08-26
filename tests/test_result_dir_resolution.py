"""A runner must evaluate its OWN result, or refuse.

Eleven runners located their child's output with
`ls -dt result/*TAG* | head -1` -- "the newest directory whose name contains
this substring". Two cells running concurrently share `result/`, so the newest
match can belong to the other run, and the evaluation then scores someone else's
checkpoint and reports it under this cell's name. `head -1` also turns "no
match" into the empty string and "several matches" into an arbitrary pick, so
neither failure announces itself.

These tests drive the real shell helper through bash, and then check that the
runners actually call it -- the pattern that mattered when three earlier fixes
passed their unit tests while nothing was wired to production.
"""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
LIB = REPO / "scripts" / "lib" / "result_dir.sh"

#: Every runner that resolves a child run by tag.
RUNNERS = [
    "maintable_cell.sh",
    "prompt_ablation_A_cell.sh",
    "prompt_ablation_A_cell_fixedE.sh",
    "prompt_ablation_A_cell_fixedN.sh",
    "prompt_ablation_cell.sh",
    "mscoco_autopilot.sh",
    "sweep_joint_cell.sh",
    "sweep_joint_cell_s5.sh",
    "sweep_joint_cell_fixedN.sh",
    "sweep_joint_cell_fixedE.sh",
    "smoke_check_unique_dirs.sh",
]


def _call(root: Path, tag: str, *, required: str | None = None):
    if required is None:
        call = f'resolve_one_result_dir {tag!r} {str(root)!r}'
    else:
        call = (f'resolve_one_result_dir_with {tag!r} {required!r} '
                f'{str(root)!r}')
    return subprocess.run(
        ["bash", "-c", f'source {LIB}; {call}'],
        capture_output=True, text=True, timeout=60)


def _run_dir(root: Path, name: str, *, complete: bool = True) -> Path:
    path = root / name
    path.mkdir(parents=True)
    if complete:
        (path / "extract_db.npz").write_bytes(b"npz")
    return path


def test_a_unique_match_resolves(tmp_path):
    _run_dir(tmp_path, "260815+cifar10_setting1_myTag+bs+64")
    proc = _call(tmp_path, "myTag")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().endswith("myTag+bs+64")


def test_a_concurrent_second_run_is_refused_not_picked(tmp_path):
    """The defect: `ls -dt | head -1` returned whichever finished last."""
    _run_dir(tmp_path, "260815+cifar10_setting1_myTag+bs+64")
    _run_dir(tmp_path, "260815+cifar10_setting1_myTag+bs+128")
    proc = _call(tmp_path, "myTag")
    assert proc.returncode != 0
    assert "refusing to guess" in proc.stderr
    # Both candidates are named, so a human can see what collided.
    assert "bs+64" in proc.stderr and "bs+128" in proc.stderr
    assert proc.stdout.strip() == ""


def test_no_match_is_an_error_not_an_empty_string(tmp_path):
    proc = _call(tmp_path, "nothingHere")
    assert proc.returncode != 0
    assert "no run matches" in proc.stderr
    assert proc.stdout.strip() == ""


def test_an_empty_tag_is_refused(tmp_path):
    """An unset shell variable would otherwise match every run."""
    _run_dir(tmp_path, "260815+cifar10_setting1_myTag+bs+64")
    proc = _call(tmp_path, "")
    assert proc.returncode != 0
    assert proc.stdout.strip() == ""


def test_an_unfinished_run_is_refused_by_the_required_file(tmp_path):
    _run_dir(tmp_path, "260815+cifar10_setting1_myTag+bs+64", complete=False)
    proc = _call(tmp_path, "myTag", required="extract_db.npz")
    assert proc.returncode != 0
    assert "did not finish" in proc.stderr


def test_a_finished_run_passes_the_required_file(tmp_path):
    _run_dir(tmp_path, "260815+cifar10_setting1_myTag+bs+64")
    proc = _call(tmp_path, "myTag", required="extract_db.npz")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().endswith("bs+64")


@pytest.mark.parametrize("runner", RUNNERS)
def test_the_runner_uses_the_shared_resolver(runner):
    """Wired to production, not merely available."""
    source = (REPO / "scripts" / runner).read_text()
    assert "lib/result_dir.sh" in source, f"{runner} does not source the helper"
    # `resolve_one_claimed_result_dir` is the stronger variant -- exactly one
    # match AND a run manifest naming this run -- so it counts.
    assert ("resolve_one_result_dir" in source
            or "resolve_one_claimed_result_dir" in source), \
        f"{runner} does not call the shared resolver"


@pytest.mark.parametrize("runner", RUNNERS)
def test_the_runner_no_longer_picks_the_newest_run(runner):
    """`ls -dt result/... | head -1` must be gone from the selection path."""
    for line in (REPO / "scripts" / runner).read_text().splitlines():
        if "head -1" in line and "result/" in line and not line.lstrip().startswith("#"):
            pytest.fail(f"{runner} still selects a run with: {line.strip()}")


@pytest.mark.parametrize("runner", RUNNERS)
def test_the_runner_still_parses(runner):
    proc = subprocess.run(["bash", "-n", str(REPO / "scripts" / runner)],
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
