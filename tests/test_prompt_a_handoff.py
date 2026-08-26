"""prompt-A must not read August's artefacts as this run's (§31.2, §32.4).

The runner's log and result names are deterministic, and the August runs still
occupy exactly those names -- their P0 logs hold E*=4/39/4/19 and their refit
directories are the Phase 2 legacy sources, so they cannot simply be moved out
of the way. Three separate paths let those be picked up:

* the runner checked no child exit status (`set -u` only), so a stage that died
  before `tee` truncated the log left the OLD log to supply E*;
* the resolved refit directory was accepted on a substring match alone, and
  every one of the 198 old refit directories carries no run manifest;
* two of the four trainers pinned `--eval_cache_dir` to the old unprovenanced
  cache, which the loader now refuses -- so stage 1 died and the above kicked in.

These drive the real script with stub trainers.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "scripts" / "prompt_ablation_A_cell.sh"
TRAINERS = [
    "train_flickr25k_v185_bidirTokenPrune05_clip.sh",
    "train_mscoco_F2_sweep_clip.sh",
    "train_nuswide_v185_sweep_clip.sh",
    "train_cifar10_v185_bidirTokenPrune05_ccs01_clip.sh",
]


def _run(cwd: Path, *, suffix: str, env_extra: dict) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.update(TAG_SUFFIX=suffix, **env_extra)
    return subprocess.run(
        ["bash", str(cwd / "scripts" / RUNNER.name), "0", "cifar_A_v4"],
        capture_output=True, text=True, env=env, cwd=str(cwd), timeout=180)


@pytest.fixture
def sandbox(tmp_path):
    """A repo-shaped tree where the trainer is a stub we control."""
    (tmp_path / "scripts" / "lib").mkdir(parents=True)
    (tmp_path / "logs").mkdir()
    (tmp_path / "result").mkdir()
    shutil.copy(RUNNER, tmp_path / "scripts" / RUNNER.name)
    shutil.copy(REPO / "scripts" / "lib" / "result_dir.sh",
                tmp_path / "scripts" / "lib" / "result_dir.sh")
    shutil.copy(REPO / "scripts" / "_run_manifest_check.py",
                tmp_path / "scripts" / "_run_manifest_check.py")
    # The runner insists both whitening matrices exist before it starts, and
    # derives their directory from GDNA_CACHE_ROOT, which the tests point here.
    whiten = tmp_path / "cifar10_clip_tokens_foils"
    whiten.mkdir()
    for name in ("text_whiten_optTrain_localOnly.npz",
                 "text_whiten_trainOnly_localOnly.npz"):
        (whiten / name).write_bytes(b"x")
    (tmp_path / "cifar10_clip_tokens").mkdir()
    return tmp_path


def _install_trainer(sandbox: Path, body: str) -> None:
    path = sandbox / "scripts" / "train_cifar10_v185_bidirTokenPrune05_ccs01_clip.sh"
    path.write_text("#!/usr/bin/env bash\n" + body)
    path.chmod(0o755)


def _cache_env(sandbox: Path) -> dict:
    return {"GDNA_CACHE_ROOT": str(sandbox)}


# ------------------------------------------------------------- blocker 1

def test_the_preflight_is_the_first_line_and_the_log_check_the_second(sandbox):
    """Both defences exist; neither alone is relied on."""
    (sandbox / "logs" / "promptAblA_cifar_A_v4_probe_P0val.log").write_text("x")
    _install_trainer(sandbox, "exit 0\n")
    refused = _run(sandbox, suffix="_probe", env_extra=_cache_env(sandbox))
    assert refused.returncode == 8
    assert "REFUSING" in refused.stderr


def test_reusing_a_tag_that_already_has_artifacts_is_refused():
    """August owns the un-suffixed namespace; a new run must not join it."""
    proc = subprocess.run(
        ["bash", str(RUNNER), "0", "cifar_A_v4"],
        capture_output=True, text=True, cwd=str(REPO), timeout=180)
    assert proc.returncode == 8, proc.stdout + proc.stderr
    assert "REFUSING" in proc.stderr
    # It names what it found, and says how to proceed.
    assert "promptAblA_cifar_A_v4_P0val.log" in proc.stderr
    assert "TAG_SUFFIX" in proc.stderr


def test_every_stale_promptA_refit_dir_lacks_a_run_manifest():
    """The defence the resolver rests on, checked against the real tree."""
    stale = sorted((REPO / "result").glob("*promptAblA_*_P0refit_*"))
    if not stale:
        pytest.skip("no historical promptAblA results on this machine")
    assert stale, "expected the August runs to still be present"
    manifested = [d for d in stale if (d / "run_identity.json").exists()]
    assert not manifested, (
        "these would be accepted by a substring match: "
        f"{[d.name for d in manifested][:3]}")


def test_the_phase2_legacy_sources_are_still_in_place():
    """They must NOT be quarantined: Phase 2 binds them by absolute path."""
    from scripts.aggregate_phase2_f01 import LEGACY

    missing = [rel for rel in LEGACY.values() if not (REPO / rel).is_dir()]
    assert not missing, f"Phase 2 legacy sources moved: {missing[:3]}"


# ------------------------------------------------------------- blocker 2

def test_a_failing_stage1_stops_before_reading_any_log(sandbox):
    """The stale log must never get the chance to supply E*.

    Reached only with ALLOW_TAG_REUSE=1: the collision preflight would
    otherwise refuse this namespace first. This is the second line of defence,
    for when someone deliberately reuses a tag.
    """
    (sandbox / "logs" / "promptAblA_cifar_A_v4_probe_P0val.log").write_text(
        "new best mid-eval mAP=0.9 at epoch 19\n")
    _install_trainer(sandbox, 'echo "stage failed" >&2\nexit 3\n')

    proc = _run(sandbox, suffix="_probe",
                env_extra={**_cache_env(sandbox), "ALLOW_TAG_REUSE": "1"})
    assert proc.returncode == 6, proc.stdout + proc.stderr
    assert "stage 1 exited nonzero" in proc.stderr
    assert "E*=19" not in proc.stdout


def test_a_log_older_than_the_launch_is_refused(sandbox):
    """A child that exits 0 without writing leaves the previous log in place."""
    log = sandbox / "logs" / "promptAblA_cifar_A_v4_probe_P0val.log"
    log.write_text("new best mid-eval mAP=0.9 at epoch 19\n")
    os.utime(log, (1, 1))                      # 1970
    _install_trainer(sandbox, "exit 0\n")

    proc = _run(sandbox, suffix="_probe",
                env_extra={**_cache_env(sandbox), "ALLOW_TAG_REUSE": "1"})
    assert proc.returncode == 6, proc.stdout + proc.stderr
    assert "older than this launch" in proc.stderr


def test_a_stage1_that_selects_nothing_is_refused(sandbox):
    """Rather than falling back to the last epoch, as it used to."""
    _install_trainer(
        sandbox,
        'mkdir -p logs\n'
        'echo "trained, selected nothing" > "logs/${TAG}.log"\n')
    proc = _run(sandbox, suffix="_probe", env_extra=_cache_env(sandbox))
    assert proc.returncode == 5, proc.stdout + proc.stderr
    assert "did not select an epoch" in proc.stderr


# ------------------------------------------------------------- blocker 3

def test_every_trainer_lets_the_caller_choose_the_eval_cache():
    """Two of the four pinned the old cache, which the loader now refuses."""
    for name in TRAINERS:
        source = (REPO / "scripts" / name).read_text()
        assert '--eval_cache_dir "${EVAL_CACHE:-$CACHE}"' in source, name
        assert "--eval_cache_dir ./cache/" not in source, name


def test_the_runner_passes_the_eval_cache_through():
    source = RUNNER.read_text()
    assert source.count('EVAL_CACHE="${EVAL_CACHE:-$CACHE}"') == 2, \
        "both stages must pass the eval cache"


def test_the_runner_requires_a_manifested_result_dir():
    source = RUNNER.read_text()
    assert "resolve_one_claimed_result_dir" in source
    assert "resolve_one_result_dir_with " not in source
