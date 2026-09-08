"""GPU-free adversarial lifecycle tests for the Phase-5 launcher/executor."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO / "scripts" / "run_ablation_campaign.sh"
EXECUTOR = REPO / "scripts" / "_ablation_exec.py"
PY = sys.executable

sys.path.insert(0, str(REPO))
from scripts.campaign_ledger import campaign_lock_key  # noqa: E402
from scripts._ablation_exec import (  # noqa: E402
    PlanRejected,
    _arm_parent_death_signal,
    _validate_campaign_lock_fd,
    child_env,
)
from scripts._ablation_campaign_launch import ProcessSupervisor  # noqa: E402
from scripts.ablation_campaign_plan import (  # noqa: E402
    ENV_PASSTHROUGH, RUNNERS, _PINNED)
from tests.test_campaign_ledger import _plan as canonical_plan  # noqa: E402


def _fake_nvidia(root: Path, uuid: str) -> Path:
    bindir = root / "bin"
    bindir.mkdir(parents=True, exist_ok=True)
    executable = bindir / "nvidia-smi"
    executable.write_text(
        "#!/usr/bin/env bash\n"
        f"printf '0, {uuid}, Synthetic GPU, 0000:01:00.0, test-driver\\n'\n")
    executable.chmod(0o755)
    return bindir


def _one_cell_plan(root: Path, *, uuid: str) -> tuple[Path, dict]:
    bindir = _fake_nvidia(root, uuid)
    runner = root / "runner.sh"
    runner.write_text(
        "#!/usr/bin/env bash\n"
        "set -Eeuo pipefail\n"
        "echo $$ > runner.pid\n"
        f'"{PY}" -c \'import signal; '
        'print(int(signal.SIGTERM in signal.pthread_sigmask(signal.SIG_BLOCK, [])))\' '
        "> signal-mask.txt\n"
        "sleep 300 &\n"
        "echo $! > nested.pid\n"
        "wait\n")
    runner.chmod(0o755)
    relative = runner.name
    path_value = f"{bindir}:/usr/bin:/bin"
    cell = {
        "exp": "synthetic", "dataset": "CIFAR10", "cell": "lifecycle",
        "N": 4, "tag": "synthetic_lifecycle", "runner": relative,
        "env": {}, "flags": [],
    }
    plan = {
        "schema_version": 2, "expected_cells": 1,
        "env_passthrough": ["PATH", "PY"],
        "env_passthrough_values": {"PATH": path_value, "PY": PY},
        "env_pinned": [], "runners": [relative], "cells": [cell],
    }
    plan["plan_digest"] = hashlib.sha256(
        json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    plan_path = root / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    return plan_path, {**os.environ, "PATH": path_value}


def _start_executor(root: Path, *, uuid: str) -> subprocess.Popen:
    plan, env = _one_cell_plan(root, uuid=uuid)
    digest = hashlib.sha256(plan.read_bytes()).hexdigest()
    return subprocess.Popen(
        [PY, str(EXECUTOR), "--plan", str(plan), "--index", "0",
         "--gpu", "0", "--repo", str(root),
         "--expect-plan-sha256", digest, "--test-only-unsealed"],
        cwd=str(root), env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True)


def _wait_for(path: Path, process: subprocess.Popen | None = None,
              timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        if process is not None and process.poll() is not None:
            output = process.stdout.read() if process.stdout else ""
            raise AssertionError(f"process exited before {path}: {output}")
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for {path}")


def _not_running(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().split()[2]
    except (FileNotFoundError, ProcessLookupError):
        return True
    return state == "Z"


def _wait_gone(pid: int, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _not_running(pid):
            return
        time.sleep(0.02)
    raise AssertionError(f"pid {pid} survived supervised termination")


def test_dry_run_is_cwd_independent_and_cannot_touch_the_ledger(tmp_path):
    plan = canonical_plan(tmp_path / "plan")
    ledger = tmp_path / "ledger"
    logs = tmp_path / "logs"
    process = subprocess.run(
        [str(LAUNCHER), "--plan", str(plan), "--ledger", str(ledger),
         "--dry-run"], cwd=str(tmp_path),
        env={**os.environ, "PY": PY, "LOGDIR": str(logs)},
        capture_output=True, text=True, timeout=120)
    assert process.returncode == 0, process.stdout + process.stderr
    assert not ledger.exists()
    dry_root = logs / f"phase5-dry-{hashlib.sha256(plan.read_bytes()).hexdigest()}"
    assert len(list(dry_root.glob("cell_*.json"))) == 24
    first = json.loads(next(dry_root.glob("cell_*.json")).read_text())
    assert first["cmd"][:3] == ["bash", "-o", "pipefail"]


def test_foreign_repo_is_refused_before_plan_or_child_access(tmp_path):
    marker = tmp_path / "must_not_run"
    plan = tmp_path / "missing-plan.json"
    process = subprocess.run(
        [str(LAUNCHER), "--plan", str(plan),
         "--ledger", str(tmp_path / "ledger"), "--repo", str(tmp_path)],
        cwd=str(tmp_path), capture_output=True, text=True)
    assert process.returncode != 0
    assert "must be the launcher source root" in process.stderr
    assert not marker.exists()


def test_noncanonical_plan_cannot_reserve_or_start_a_child(tmp_path):
    plan, env = _one_cell_plan(
        tmp_path / "cell", uuid="GPU-PHASE5-NONCANONICAL")
    process = subprocess.run(
        [str(LAUNCHER), "--plan", str(plan),
         "--ledger", str(tmp_path / "ledger")], cwd=str(tmp_path),
        env={**env, "PY": PY, "LOGDIR": str(tmp_path / "logs")},
        capture_output=True, text=True, timeout=60)
    assert process.returncode != 0
    assert "exactly 24" in process.stdout + process.stderr
    assert not (tmp_path / "cell" / "runner.pid").exists()
    assert not (tmp_path / "ledger" / "campaign_complete.json").exists()


def test_global_tag_set_lock_blocks_a_different_ledger_before_gpu(tmp_path):
    plan = canonical_plan(tmp_path / "plan")
    key = campaign_lock_key(plan)
    root = Path(f"/tmp/groundeddna-ablation-campaign-leases-{os.getuid()}")
    root.mkdir(mode=0o700, exist_ok=True)
    root.chmod(0o700)
    handle = (root / f"{key}.lock").open("a+")
    os.chmod(root / f"{key}.lock", 0o600)
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        process = subprocess.run(
            [str(LAUNCHER), "--plan", str(plan),
             "--ledger", str(tmp_path / "different-ledger")],
            cwd=str(tmp_path), env={**os.environ, "PY": PY},
            capture_output=True, text=True, timeout=60)
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
    assert process.returncode != 0
    assert "live campaign owner" in process.stdout + process.stderr
    assert not (tmp_path / "different-ledger" / "campaign_reservation.json").exists()


@pytest.mark.parametrize(
    ("requested_signal", "expected_returncode"),
    ((signal.SIGTERM, 143), (signal.SIGINT, 130)),
    ids=("term", "int"))
def test_signal_kills_the_entire_nested_process_group(
        tmp_path, requested_signal, expected_returncode):
    root = tmp_path / requested_signal.name.lower()
    root.mkdir()
    process = _start_executor(
        root, uuid=f"GPU-PHASE5-{requested_signal.name}-TEST")
    try:
        _wait_for(root / "nested.pid", process)
        assert (root / "signal-mask.txt").read_text().strip() == "0"
        runner_pid = int((root / "runner.pid").read_text())
        nested_pid = int((root / "nested.pid").read_text())
        process.send_signal(requested_signal)
        output = process.communicate(timeout=20)[0]
        assert process.returncode == expected_returncode, output
        _wait_gone(runner_pid)
        _wait_gone(nested_pid)
    finally:
        if process.poll() is None:
            process.kill()


def test_launcher_sigkill_triggers_executor_parent_death_cleanup(tmp_path):
    root = tmp_path / "pdeath"
    root.mkdir()
    plan, env = _one_cell_plan(root, uuid="GPU-PHASE5-PDEATH-TEST")
    digest = hashlib.sha256(plan.read_bytes()).hexdigest()
    command = (
        f'"{PY}" "{EXECUTOR}" --plan "{plan}" --index 0 --gpu 0 '
        f'--repo "{root}" --expect-plan-sha256 "{digest}" '
        '--test-only-unsealed & echo $! > executor.pid; wait')
    parent = subprocess.Popen(
        ["bash", "-c", command], cwd=str(root), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        _wait_for(root / "nested.pid", parent)
        executor_pid = int((root / "executor.pid").read_text())
        runner_pid = int((root / "runner.pid").read_text())
        nested_pid = int((root / "nested.pid").read_text())
        parent.kill()
        parent.wait(timeout=10)
        _wait_gone(executor_pid)
        _wait_gone(runner_pid)
        _wait_gone(nested_pid)
    finally:
        if parent.poll() is None:
            parent.kill()


def test_executor_sigkill_guardian_reaps_group_and_releases_gpu_lease(tmp_path):
    uuid = "GPU-PHASE5-INHERITED-LEASE"
    first_root = tmp_path / "first"
    first_root.mkdir()
    first = _start_executor(first_root, uuid=uuid)
    runner_pid = nested_pid = first_group = None
    blocked = second = None
    second_runner = second_nested = second_group = None
    try:
        _wait_for(first_root / "nested.pid", first)
        runner_pid = int((first_root / "runner.pid").read_text())
        nested_pid = int((first_root / "nested.pid").read_text())
        first_group = os.getpgid(runner_pid)
        # Freeze the guardian so the executor-death TERM is pending.  Its
        # inherited lease must still block a colliding executor during this
        # forced cleanup window, not merely after eventual reaping.
        os.kill(first_group, signal.SIGSTOP)
        first.kill()
        first.wait(timeout=10)
        blocked_root = tmp_path / "blocked"
        blocked_root.mkdir()
        blocked = _start_executor(blocked_root, uuid=uuid)
        blocked_output = blocked.communicate(timeout=10)[0]
        assert blocked.returncode != 0
        assert "already has a live GroundedDNA lease" in blocked_output
        assert not _not_running(runner_pid)
        assert not _not_running(nested_pid)
        os.kill(first_group, signal.SIGCONT)
        _wait_gone(runner_pid)
        _wait_gone(nested_pid)

        second_root = tmp_path / "second"
        second_root.mkdir()
        second = _start_executor(second_root, uuid=uuid)
        _wait_for(second_root / "nested.pid", second)
        second_runner = int((second_root / "runner.pid").read_text())
        second_nested = int((second_root / "nested.pid").read_text())
        second_group = os.getpgid(second_runner)
        second.send_signal(signal.SIGTERM)
        second.communicate(timeout=20)
        _wait_gone(second_runner)
        _wait_gone(second_nested)
    finally:
        if first_group is not None:
            try:
                os.kill(first_group, signal.SIGCONT)
            except ProcessLookupError:
                pass
            try:
                os.killpg(first_group, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if runner_pid is not None:
            _wait_gone(runner_pid)
        if nested_pid is not None:
            _wait_gone(nested_pid)
        if first.poll() is None:
            first.kill()
        if blocked is not None and blocked.poll() is None:
            blocked.kill()
        if second is not None and second.poll() is None:
            second.kill()
        if second_group is not None:
            try:
                os.killpg(second_group, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if second_runner is not None:
            _wait_gone(second_runner)
        if second_nested is not None:
            _wait_gone(second_nested)


def test_lock_leaf_symlink_is_refused_without_corrupting_target(tmp_path):
    plan = canonical_plan(tmp_path / "plan")
    key = campaign_lock_key(plan)
    root = Path(f"/tmp/groundeddna-ablation-campaign-leases-{os.getuid()}")
    root.mkdir(mode=0o700, exist_ok=True)
    root.chmod(0o700)
    leaf = root / f"{key}.lock"
    victim = tmp_path / "victim.txt"
    victim.write_text("DO-NOT-OVERWRITE\n")
    if leaf.exists() or leaf.is_symlink():
        leaf.unlink()
    leaf.symlink_to(victim)
    try:
        process = subprocess.run(
            [str(LAUNCHER), "--plan", str(plan),
             "--ledger", str(tmp_path / "ledger")], cwd=str(tmp_path),
            env={**os.environ, "PY": PY}, capture_output=True, text=True,
            timeout=60)
    finally:
        if leaf.is_symlink():
            leaf.unlink()
    assert process.returncode != 0
    assert "unsafe campaign lock leaf" in process.stdout + process.stderr
    assert victim.read_text() == "DO-NOT-OVERWRITE\n"


def test_executor_rejects_character_device_as_campaign_lock_fd(tmp_path):
    tag_set = hashlib.sha256(str(tmp_path).encode()).hexdigest()
    root = Path(f"/tmp/groundeddna-ablation-campaign-leases-{os.getuid()}")
    root.mkdir(mode=0o700, exist_ok=True)
    root.chmod(0o700)
    leaf = root / f"{tag_set}.lock"
    leaf.touch(mode=0o600, exist_ok=False)
    descriptor = os.open("/dev/null", os.O_RDWR)
    try:
        with pytest.raises(PlanRejected, match="canonical safe tag lock"):
            _validate_campaign_lock_fd(
                {"tag_set_sha256": tag_set}, descriptor)
    finally:
        os.close(descriptor)
        leaf.unlink()


def test_production_env_schema_rejects_bash_startup_injection():
    # These values are normalized by the launcher and therefore must not also
    # be compared with a plan built in the caller's pre-launch environment.
    assert {"PATH", "PYTHONPATH", "LD_LIBRARY_PATH"}.isdisjoint(
        ENV_PASSTHROUGH)
    values = {name: os.environ[name] for name in ENV_PASSTHROUGH
              if name in os.environ}
    values["PY"] = str(Path(PY).resolve())
    cell_env = {name: "" for name in _PINNED}
    plan = {
        "env_passthrough": sorted(ENV_PASSTHROUGH),
        "env_passthrough_values": values,
        "env_pinned": list(_PINNED), "runners": sorted(RUNNERS),
    }
    cell = {"tag": "canonical", "env": cell_env}
    child_env(plan, cell, production=True)
    plan["env_passthrough"].append("BASH_ENV")
    plan["env_passthrough_values"]["BASH_ENV"] = "/tmp/startup.sh"
    with pytest.raises(PlanRejected, match="schema is not canonical"):
        child_env(plan, cell, production=True)


def test_parent_death_arm_requires_the_declared_launcher_pid():
    with pytest.raises(PlanRejected, match="expected live launcher"):
        _arm_parent_death_signal(os.getppid() + 1_000_000)


def test_direct_entrypoint_drops_bash_env_before_parsing(tmp_path):
    marker = tmp_path / "bash-env-executed"
    startup = tmp_path / "startup.sh"
    startup.write_text(f"touch {marker}\n")
    process = subprocess.run(
        [str(LAUNCHER), "--definitely-invalid"],
        env={**os.environ, "BASH_ENV": str(startup), "PY": PY},
        capture_output=True, text=True)
    assert process.returncode == 2
    assert not marker.exists()


def test_direct_entrypoint_never_imports_bash_xtrace_or_functions(tmp_path):
    xtrace_marker = tmp_path / "xtrace-startup"
    function_marker = tmp_path / "function-startup"
    env = {
        **os.environ,
        "PY": PY,
        "SHELLOPTS": "xtrace",
        "PS4": f'$(/usr/bin/touch "{xtrace_marker}")',
        "FUNC_MARKER": str(function_marker),
        "BASH_FUNC_pwd%%": (
            "() { /usr/bin/touch \"$FUNC_MARKER\"; builtin pwd \"$@\"; }")
    }
    process = subprocess.run(
        [str(LAUNCHER), "--definitely-invalid"], env=env,
        capture_output=True, text=True)
    assert process.returncode == 2
    assert not xtrace_marker.exists()
    assert not function_marker.exists()


def test_term_then_hup_interrupts_hung_helper_and_finishes_cleanup(tmp_path):
    child_file = tmp_path / "stubborn.pid"
    helper_file = tmp_path / "helper.pid"
    command = f'''
import sys
sys.path.insert(0, {str(REPO)!r})
from scripts._ablation_campaign_launch import ProcessSupervisor
s = ProcessSupervisor(grace_seconds=1.0)
s.install_handlers()
s.spawn(["/usr/bin/bash", "-c",
         "trap '' INT TERM HUP; echo $$ > {child_file}; exec /usr/bin/sleep 300"])
helper = s.spawn(["/usr/bin/bash", "-c",
                  "trap '' INT TERM HUP; echo $$ > {helper_file}; exec /usr/bin/sleep 300"])
s.wait(helper)
'''
    process = subprocess.Popen(
        [PY, "-c", command], start_new_session=True,
        env={**os.environ, "PY": PY},
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    child_pid = helper_pid = None
    try:
        _wait_for(child_file, process)
        _wait_for(helper_file, process)
        child_pid = int(child_file.read_text())
        helper_pid = int(helper_file.read_text())
        process.send_signal(signal.SIGTERM)
        time.sleep(0.1)
        process.send_signal(signal.SIGHUP)
        output = process.communicate(timeout=10)[0]
        assert process.returncode == 143, output
        _wait_gone(child_pid)
    finally:
        if process.poll() is None:
            process.kill()
        if child_pid is not None and not _not_running(child_pid):
            os.kill(child_pid, signal.SIGKILL)
            _wait_gone(child_pid)
        if helper_pid is not None and not _not_running(helper_pid):
            os.kill(helper_pid, signal.SIGKILL)
            _wait_gone(helper_pid)


def test_malformed_cleanup_knob_is_refused_before_any_child(tmp_path):
    process = subprocess.run(
        [str(LAUNCHER), "--plan", str(tmp_path / "missing-plan"),
         "--ledger", str(tmp_path / "ledger")],
        env={**os.environ, "PY": PY,
             "CAMPAIGN_TERM_GRACE": "not-an-integer"},
        capture_output=True, text=True)
    assert process.returncode == 2
    assert "CAMPAIGN_TERM_GRACE must be" in process.stderr
    assert not (tmp_path / "ledger").exists()


def test_only_executors_receive_the_campaign_lock_fd(tmp_path):
    lock = tmp_path / "campaign.lock"
    descriptor = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
    supervisor = ProcessSupervisor(grace_seconds=0.2)
    code = (
        "import json,os; print(json.dumps([os.readlink('/proc/self/fd/'+f) "
        "for f in os.listdir('/proc/self/fd') if os.path.exists('/proc/self/fd/'+f)]))")
    try:
        rc, stdout, _ = supervisor.communicate([PY, "-c", code])
        assert rc == 0
        assert str(lock) not in json.loads(stdout)
        rc, stdout, _ = supervisor.communicate(
            [PY, "-c", code], pass_fds=(descriptor,))
        assert rc == 0
        assert str(lock) in json.loads(stdout)
    finally:
        supervisor.terminate_all()
        os.close(descriptor)


def test_sigkill_supervisor_cannot_leave_a_publishing_helper(tmp_path):
    helper_pid_file = tmp_path / "helper.pid"
    marker = tmp_path / "late-publication"
    command = f'''
import sys, time
sys.path.insert(0, {str(REPO)!r})
from scripts._ablation_campaign_launch import ProcessSupervisor
s = ProcessSupervisor(grace_seconds=1.0)
p = s.spawn(["/usr/bin/bash", "-c",
             "echo $$ > {helper_pid_file}; sleep 1; touch {marker}"])
time.sleep(300)
'''
    parent = subprocess.Popen([PY, "-c", command])
    helper_pid = None
    try:
        _wait_for(helper_pid_file, parent)
        helper_pid = int(helper_pid_file.read_text())
        parent.kill()
        parent.wait(timeout=10)
        _wait_gone(helper_pid)
        time.sleep(1.1)
        assert not marker.exists()
    finally:
        if parent.poll() is None:
            parent.kill()
        if helper_pid is not None and not _not_running(helper_pid):
            os.kill(helper_pid, signal.SIGKILL)
            _wait_gone(helper_pid)


def test_chain_delegates_to_the_single_supervised_launcher():
    chain = (REPO / "scripts" / "auto_chain_after_3seed.sh").read_text()
    assert "run_ablation_campaign.sh" in chain
    assert "_ablation_exec.py" not in chain


def test_launcher_and_wrapper_parse_before_any_gpu_execution():
    process = subprocess.run(
        [PY, "-m", "py_compile", str(LAUNCHER)],
        capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    process = subprocess.run(
        ["bash", "-n",
         str(REPO / "scripts" / "prompt_ablation_A_cell_fixedN.sh")],
        capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
