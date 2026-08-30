"""The campaign launcher, executed end to end: plan -> reserve -> cells -> seal.

Re-audit §48.4 ended on this: the loop lived below `auto_chain_after_3seed.sh`'s
`exit 91`, so it was unreachable, and every test around it read the file's text
or the executor's `--dry-run` output. "Narrow safety PASS, not production E2E."

These tests run `scripts/run_ablation_campaign.sh` against a plan of stub cells
in a temporary tree -- the real launcher, the real executor, the real ledger --
and check the property that matters at each boundary: a campaign that loses a
cell must not produce a completion receipt, and one that finishes must.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO / "scripts" / "run_ablation_campaign.sh"
PY = sys.executable

CELLS = [(exp, cell) for exp in ("cifar_A_v4", "flickr_A_v4")
         for cell in ("A2_no_text", "A4_shared_codebook", "A5_none",
                      "A5_joint", "A5_nogumbel", "A5_both")]


def _stub_runner(tmp_path: Path, *, failing: set) -> str:
    """Stands in for the fixedN wrapper: same argv, same rc semantics."""
    path = tmp_path / "stub_runner.sh"
    path.write_text(
        'set -Eeuo pipefail\n'
        f'FAILING="{" ".join(sorted(failing))}"\n'
        'GPU="$1"; EXP="$2"\n'
        'echo "[stub] gpu=$GPU exp=$EXP suffix=$TAG_SUFFIX k=$K n=$FIXED_N"\n'
        'for f in $FAILING; do\n'
        '  if [ "${EXP}${TAG_SUFFIX}" = "$f" ]; then\n'
        '    echo "[stub] dying"; exit 23\n'
        '  fi\n'
        'done\n'
        'echo "[stub] DONE"\n')
    return str(path.relative_to(tmp_path))


def _plan(tmp_path: Path, runner: str) -> Path:
    sys.path.insert(0, str(REPO))
    from scripts.ablation_campaign_plan import ENV_PASSTHROUGH, _PINNED

    cells = []
    for exp, cell in CELLS:
        env = {k: "" for k in _PINNED}
        env.update({"K": "64", "FIXED_N": "4", "NUM_CODONS": "3",
                    "TAG_SUFFIX": f"_{cell}", "VIZ": "0"})
        cells.append({"exp": exp, "cell": cell, "runner": runner, "env": env,
                      "tag": f"promptAblA_{exp}_{cell}"})
    plan = {
        "schema_version": 2,
        "expected_cells": len(cells),
        "plan_digest": None,
        "env_passthrough": sorted(ENV_PASSTHROUGH),
        "env_passthrough_values": {"PATH": os.environ["PATH"]},
        "env_pinned": list(_PINNED),
        "runners": [runner],
        "cells": cells,
    }
    del plan["plan_digest"]
    plan["plan_digest"] = hashlib.sha256(
        json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    return path


def _run(tmp_path: Path, plan: Path, ledger: Path, *, extra_env=None):
    env = dict(os.environ)
    env.update({"PY": PY, "GPUS": "0 1", "STAGGER": "0",
                "LOGDIR": str(tmp_path / "logs")})
    env.update(extra_env or {})
    return subprocess.run(
        ["bash", str(LAUNCHER), "--plan", str(plan), "--ledger", str(ledger),
         "--repo", str(tmp_path)],
        cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=300)


def _tree(tmp_path: Path, *, failing=frozenset()):
    """A working directory with the repo's scripts/ reachable as `scripts/`."""
    (tmp_path / "scripts").symlink_to(REPO / "scripts")
    return _stub_runner(tmp_path, failing=failing)


def test_a_complete_campaign_seals(tmp_path):
    runner = _tree(tmp_path)
    ledger = tmp_path / "ledger"
    proc = _run(tmp_path, _plan(tmp_path, runner), ledger)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    receipt = json.loads((ledger / "campaign_complete.json").read_text())
    assert receipt["cell_count"] == len(CELLS)
    assert set(receipt["cells"]) == {f"promptAblA_{e}_{c}" for e, c in CELLS}


def test_one_dead_cell_leaves_no_receipt(tmp_path):
    """The whole point: 11 of 12 must not be readable as a table."""
    runner = _tree(tmp_path, failing={"flickr_A_v4_A5_both"})
    ledger = tmp_path / "ledger"
    proc = _run(tmp_path, _plan(tmp_path, runner), ledger)
    assert proc.returncode != 0
    assert not (ledger / "campaign_complete.json").exists()
    assert "FAILED rc=23" in proc.stdout
    recorded = json.loads(
        (ledger / "cell_promptAblA_flickr_A_v4_A5_both.json").read_text())
    assert recorded["status"] == "failed"


def test_a_second_campaign_is_refused_before_any_cell_starts(tmp_path):
    runner = _tree(tmp_path)
    ledger = tmp_path / "ledger"
    plan = _plan(tmp_path, runner)
    assert _run(tmp_path, plan, ledger).returncode == 0
    second = _run(tmp_path, plan, ledger)
    assert second.returncode != 0
    assert "already held" in second.stdout + second.stderr


def test_a_plan_edited_mid_campaign_stops_the_cells(tmp_path):
    """TOCTOU: the launcher hashes once and hands that digest to every cell."""
    runner = _tree(tmp_path, failing=set())
    ledger = tmp_path / "ledger"
    plan = _plan(tmp_path, runner)
    # A slow first cell gives the edit somewhere to land.
    slow = tmp_path / "slow_runner.sh"
    slow.write_text('set -Eeuo pipefail\nsleep 2\necho "[stub] $2 $TAG_SUFFIX"\n')
    payload = json.loads(plan.read_text())
    payload["cells"][0]["runner"] = str(slow.relative_to(tmp_path))
    payload["runners"].append(str(slow.relative_to(tmp_path)))
    payload.pop("plan_digest")
    payload["plan_digest"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    plan.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    proc = subprocess.Popen(
        ["bash", str(LAUNCHER), "--plan", str(plan), "--ledger", str(ledger),
         "--repo", str(tmp_path)],
        cwd=str(tmp_path), env={**os.environ, "PY": PY, "GPUS": "0",
                                "STAGGER": "1", "LOGDIR": str(tmp_path / "logs")},
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    # Rewrite the plan while the campaign is running.
    import time
    time.sleep(2)
    plan.write_text(plan.read_text().replace('"K": "64"', '"K": "999"'))
    out = proc.communicate(timeout=300)[0]
    assert proc.returncode != 0, out
    assert not (ledger / "campaign_complete.json").exists()
    # The refusal is the child's, so it lands in that cell's log rather than on
    # the launcher's stdout.
    refusals = [p for p in (tmp_path / "logs").glob("ABL_*.out")
                if "the plan changed after the run began" in p.read_text()]
    assert refusals, (
        "cells launched after the edit ran the new plan instead of refusing:\n"
        + out)


def test_the_launcher_passes_the_plans_environment_to_the_child(tmp_path):
    runner = _tree(tmp_path)
    ledger = tmp_path / "ledger"
    proc = _run(tmp_path, _plan(tmp_path, runner), ledger,
                extra_env={"K": "999", "NUM_CODONS": "4", "FIXED_N": "77"})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    logs = sorted((tmp_path / "logs").glob("ABL_*.out"))
    assert len(logs) == len(CELLS)
    for log in logs:
        text = log.read_text()
        assert "k=64" in text and "n=4" in text, text


def test_the_chain_delegates_to_this_launcher():
    """Wired to production, not merely available."""
    chain = (REPO / "scripts" / "auto_chain_after_3seed.sh").read_text()
    assert "run_ablation_campaign.sh" in chain
    assert "_ablation_exec.py" not in chain, (
        "the chain still launches cells itself; there are two loops to keep "
        "correct instead of one")


def test_a_dry_run_cannot_produce_a_completion_receipt(tmp_path):
    """The false seal I actually created, with the real 24-cell plan.

    `--dry-run` was passed through to the executor, which prints the command and
    returns 0; the launcher recorded each of those as `status=ok` with a null
    run directory and sealed. `/tmp/dryledger/campaign_complete.json` came out
    naming 24 cells after zero trainers ran and zero GPUs were claimed, and I
    wrote "planned, reserved, executed and sealed" in the project log on the
    strength of it. A receipt has to mean trainers ran.
    """
    runner = _tree(tmp_path)
    ledger_dry = tmp_path / "ledger_dry"
    dry = subprocess.run(
        ["bash", str(LAUNCHER), "--plan", str(_plan(tmp_path, runner)),
         "--ledger", str(ledger_dry), "--repo", str(tmp_path), "--dry-run"],
        cwd=str(tmp_path),
        env={**os.environ, "PY": PY, "GPUS": "0", "STAGGER": "0",
             "LOGDIR": str(tmp_path / "drylogs")},
        capture_output=True, text=True, timeout=300)
    assert dry.returncode == 0, dry.stdout + dry.stderr
    assert not (ledger_dry / "campaign_complete.json").exists(), (
        "a dry run sealed a completion receipt")
    assert not (ledger_dry / "campaign_reservation.json").exists(), (
        "a dry run reserved the campaign, so the real one cannot open it")
    assert not list(ledger_dry.glob("cell_*.json")) if ledger_dry.exists() else True
    printed = sorted((tmp_path / "drylogs").glob("DRY_*.json"))
    assert len(printed) == len(CELLS)
    assert json.loads(printed[0].read_text())["cmd"][:3] == ["bash", "-o",
                                                             "pipefail"]


def test_concurrent_cells_do_not_share_a_gpu(tmp_path):
    """`GPUS="0 1 2"` put all three smoke cells on GPU 0 while 1-5 sat idle.

    `next_gpu` took `${g%% *}` -- the first token of the candidate list -- on
    every call, so the explicit override was read as "use GPU 0". Re-audit
    §50.1 predicted it from the source; the three-cell top-p smoke then did it.
    The stub sleeps, so the cells are genuinely concurrent and a launcher that
    hands out one GPU repeatedly is caught.
    """
    runner = _tree(tmp_path)
    slow = tmp_path / "slow.sh"
    slow.write_text('set -Eeuo pipefail\n'
                    'echo "[stub] gpu=$CUDA_VISIBLE_DEVICES"\nsleep 3\n')
    plan_path = _plan(tmp_path, runner)
    payload = json.loads(plan_path.read_text())
    for cell in payload["cells"]:
        cell["runner"] = str(slow.relative_to(tmp_path))
    payload["runners"] = [str(slow.relative_to(tmp_path))]
    payload.pop("plan_digest")
    payload["plan_digest"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    plan_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    proc = _run(tmp_path, plan_path, tmp_path / "ledger",
                extra_env={"GPUS": "0 1 2", "STAGGER": "0", "GPU_WAIT": "1"})
    assert proc.returncode == 0, proc.stdout + proc.stderr

    assigned = {}
    for log in (tmp_path / "logs").glob("ABL_*.out"):
        line = log.read_text().strip().splitlines()[0]
        assigned[log.name] = line.split("gpu=")[1].strip()
    assert len(assigned) == len(CELLS)
    # Three GPUs, twelve cells, each cell sleeping: at any instant at most three
    # run, and no two of those may hold the same device.
    assert set(assigned.values()) == {"0", "1", "2"}, (
        f"cells did not spread across the offered GPUs: {assigned}")
