"""Adversarial tests for the D6 child/GPU execution-environment seal."""
from __future__ import annotations

import copy
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace

import pytest
import torch

from baseline.execution_environment import (
    ExecutionEnvironmentError,
    PACKAGE_DISTRIBUTIONS,
    capture_child_execution_environment,
    require_exact_environment,
    resolve_gpu_assignments,
    verify_execution_environment,
)
from dna_utils.gpu_lease import acquire_gpu_leases
from scripts.run_baseline_p0_matrix import _ManagedChildSupervisor
from scripts.run_modern_baseline_p0 import _read_verified_checkpoint_protocol


def _assignment(*, index: int = 2, uuid: str = "GPU-TEST-2222") -> dict:
    return {
        "physical_index": index,
        "uuid": uuid,
        "name": "Synthetic GPU",
        "pci_bus_id": "0000:22:00.0",
        "driver_version": "555.test",
    }


class _Cuda:
    def __init__(self, *, count: int = 1, name: str = "Synthetic GPU"):
        self._count = count
        self._name = name

    @staticmethod
    def is_available() -> bool:
        return True

    def device_count(self) -> int:
        return self._count

    def get_device_name(self, index: int) -> str:
        assert index == 0
        return self._name

    @staticmethod
    def get_device_properties(index: int) -> object:
        assert index == 0
        return SimpleNamespace(
            total_memory=24_000_000_000,
            major=8,
            minor=9,
            multi_processor_count=128,
            uuid="GPU-TEST-2222",
        )


def _fake_torch(*, count: int = 1, name: str = "Synthetic GPU") -> object:
    return SimpleNamespace(
        __version__="test-torch-1",
        cuda=_Cuda(count=count, name=name),
        version=SimpleNamespace(cuda="12.4"),
        backends=SimpleNamespace(
            cudnn=SimpleNamespace(version=lambda: 90100)),
    )


def _version(distribution: str) -> str:
    assert distribution in PACKAGE_DISTRIBUTIONS.values()
    return f"test-{distribution}-1"


def _capture(*, assignment: dict | None = None,
             environ: dict[str, str] | None = None,
             version=_version, torch_module=None,
             inventory=None):
    assignment = _assignment() if assignment is None else assignment
    environ = ({"CUDA_VISIBLE_DEVICES": assignment["uuid"]}
               if environ is None else environ)
    return capture_child_execution_environment(
        assignment,
        torch_module=_fake_torch() if torch_module is None else torch_module,
        inventory=[assignment] if inventory is None else inventory,
        version=version,
        executable=sys.executable,
        environ=environ,
    )


def test_actual_child_capture_has_exact_required_package_and_gpu_contract():
    payload, digest = _capture()
    assert set(payload["packages"]) == set(PACKAGE_DISTRIBUTIONS)
    assert payload["torch_visible_device_count"] == 1
    assert payload["cuda_visible_devices"] == "GPU-TEST-2222"
    assert payload["assigned_physical_gpu"] == _assignment()
    assert payload["logical_device_0"]["major"] == 8
    assert verify_execution_environment(payload, digest) == payload


def test_parent_assignment_index_uuid_mismatch_is_rejected():
    observed = _assignment(uuid="GPU-DIFFERENT-9999")
    with pytest.raises(ExecutionEnvironmentError, match="physical GPU"):
        _capture(inventory=[observed])


@pytest.mark.parametrize("visible", ["2", "GPU-DIFFERENT-9999", "GPU-TEST-2222,3"])
def test_cuda_visible_devices_spoof_is_rejected(visible):
    with pytest.raises(ExecutionEnvironmentError, match="CUDA_VISIBLE_DEVICES"):
        _capture(environ={"CUDA_VISIBLE_DEVICES": visible})


def test_more_than_one_logical_cuda_device_is_rejected():
    with pytest.raises(ExecutionEnvironmentError, match="exactly 1"):
        _capture(torch_module=_fake_torch(count=2))


def test_package_drift_between_runner_and_trainer_is_rejected():
    expected, expected_digest = _capture()

    def drifted(distribution: str) -> str:
        value = _version(distribution)
        return value + "-drift" if distribution == "numpy" else value

    actual, actual_digest = _capture(version=drifted)
    with pytest.raises(ExecutionEnvironmentError, match="drifted"):
        require_exact_environment(
            expected, expected_digest,
            actual=actual, actual_digest=actual_digest)


def test_missing_or_extra_schema_fields_are_rejected():
    payload, digest = _capture()
    del payload["logical_device_0"]
    with pytest.raises(ExecutionEnvironmentError, match="top-level"):
        verify_execution_environment(payload, digest)


def test_checkpoint_weights_only_preserves_and_reopens_exact_attestation():
    environment, environment_digest = _capture()
    protocol_digest = "a" * 64
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "epoch_059.pth"
        torch.save({
            "config": {
                "protocol_mode": "author_fixed_final",
                "protocol_stage": "author_fixed_single_stage",
                "protocol_identity_sha256": protocol_digest,
                "execution_environment": environment,
                "execution_environment_sha256": environment_digest,
            },
            "model_state_dict": {},
        }, checkpoint)
        reopened = _read_verified_checkpoint_protocol(
            checkpoint,
            protocol_mode="author_fixed_final",
            protocol_stage="author_fixed_single_stage",
            protocol_digest=protocol_digest,
            execution_environment=environment,
            execution_environment_sha256=environment_digest)
        assert reopened["protocol_identity_sha256"] == protocol_digest

        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        payload["config"]["execution_environment"] = copy.deepcopy(environment)
        payload["config"]["execution_environment"]["packages"]["numpy"][
            "version"] = "forged"
        torch.save(payload, checkpoint)
        with pytest.raises(ExecutionEnvironmentError):
            _read_verified_checkpoint_protocol(
                checkpoint,
                protocol_mode="author_fixed_final",
                protocol_stage="author_fixed_single_stage",
                protocol_digest=protocol_digest,
                execution_environment=environment,
                execution_environment_sha256=environment_digest)


def test_matrix_resolves_canonical_physical_indices_to_uuid_records():
    assignments = resolve_gpu_assignments(
        ["2"], inventory=[_assignment()])
    assert assignments == {"2": _assignment()}
    with pytest.raises(ExecutionEnvironmentError, match="canonical decimal"):
        resolve_gpu_assignments(["GPU-TEST-2222"], inventory=[_assignment()])


def test_host_global_uuid_lease_rejects_a_second_live_launcher():
    unique = f"GPU-TEST-LEASE-{os.getpid()}"
    assignment = _assignment(index=999_999, uuid=unique)
    first = acquire_gpu_leases(
        [assignment], owner_metadata={"launcher": "first"})
    try:
        with pytest.raises(RuntimeError, match="live GroundedDNA lease"):
            acquire_gpu_leases(
                [assignment], owner_metadata={"launcher": "second"})
        metadata = first.paths[0].read_text(encoding="utf-8")
        assert f'"pid": {os.getpid()}' in metadata
        assert '"boot_id":' in metadata
        assert '"command":' in metadata
    finally:
        first.release()


def _wait_for_pids(path: Path, *, timeout: float = 5.0) -> list[int]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            values = [int(value) for value in path.read_text().split()]
        except (FileNotFoundError, ValueError):
            values = []
        if len(values) == 2:
            return values
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for two process IDs in {path}")


def _process_is_live(pid: int) -> bool:
    try:
        stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except FileNotFoundError:
        return False
    right_paren = stat_text.rfind(")")
    return right_paren >= 0 and stat_text[right_paren + 2:].split()[0] != "Z"


def _wait_processes_dead(pids: list[int], *, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while any(_process_is_live(pid) for pid in pids) \
            and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not [pid for pid in pids if _process_is_live(pid)]


def _synthetic_campaign_code(
        *, lease_root: Path, uuid: str, pids_path: Path,
        handle_signals: bool,
        ) -> str:
    """No trainer/CUDA import: only a leased sleeper process tree."""
    child_code = (
        "import os,signal,subprocess,sys,time;"
        "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
        "grand=subprocess.Popen([sys.executable,'-c',"
        "'import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);'"
        "'time.sleep(60)']);"
        f"open({str(pids_path)!r},'w').write("
        "str(os.getpid())+' '+str(grand.pid));"
        "time.sleep(60)")
    signal_setup = (
        "lifecycle.install_signal_handlers();"
        if handle_signals else "")
    signal_wait = (
        "\nwhile not lifecycle.stop.wait(0.02): pass\n"
        "signum=lifecycle.interrupted_signum\n"
        "try:\n lifecycle.close()\nfinally:\n leases.release()\n"
        "raise SystemExit(128+signum)\n"
        if handle_signals else
        "\nwhile True: time.sleep(60)\n")
    return (
        "import sys,time\n"
        "from pathlib import Path\n"
        "from dna_utils.gpu_lease import acquire_gpu_leases\n"
        "from scripts.run_baseline_p0_matrix import _CampaignLifecycle\n"
        f"assignment={{'uuid':{uuid!r},'physical_index':999999}}\n"
        f"leases=acquire_gpu_leases([assignment],owner_metadata={{'launcher':'synthetic'}},lease_root=Path({str(lease_root)!r}))\n"
        "lifecycle=_CampaignLifecycle([h.fileno() for h in leases.handles],term_grace_seconds=0.15,kill_wait_seconds=1.0)\n"
        f"{signal_setup}\n"
        f"process=lifecycle.children.spawn([sys.executable,'-c',{child_code!r}])\n"
        f"{signal_wait}")


@pytest.mark.parametrize(
    ("signum", "expected_returncode"),
    [(signal.SIGINT, 130), (signal.SIGTERM, 143)],
)
def test_campaign_signal_kills_sleeper_grandchild_before_lease_release(
        tmp_path, signum, expected_returncode):
    lease_root = tmp_path / "leases"
    pids_path = tmp_path / "pids.txt"
    uuid = f"GPU-TEST-SIGNAL-{signum}-{os.getpid()}"
    launcher = subprocess.Popen(
        [sys.executable, "-c", _synthetic_campaign_code(
            lease_root=lease_root, uuid=uuid, pids_path=pids_path,
            handle_signals=True)],
        cwd=Path(__file__).resolve().parents[1],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    pids: list[int] = []
    try:
        pids = _wait_for_pids(pids_path)
        assert len(pids) == 2 and all(_process_is_live(pid) for pid in pids)
        launcher.send_signal(signum)
        stdout, stderr = launcher.communicate(timeout=5)
        assert launcher.returncode == expected_returncode, (stdout, stderr)
        _wait_processes_dead(pids)

        # close() finishes the whole process group before the lease release.
        reacquired = acquire_gpu_leases(
            [_assignment(index=999_999, uuid=uuid)],
            owner_metadata={"launcher": "post-signal"},
            lease_root=lease_root)
        reacquired.release()
    finally:
        if launcher.poll() is None:
            launcher.kill()
            launcher.wait(timeout=2)
        for pid in pids:
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def _lease_probe_code(*, lease_root: Path, uuid: str) -> str:
    return (
        "from pathlib import Path\n"
        "from dna_utils.gpu_lease import acquire_gpu_leases\n"
        f"assignment={{'uuid':{uuid!r},'physical_index':999999}}\n"
        "try:\n"
        f" leases=acquire_gpu_leases([assignment],owner_metadata={{'launcher':'second'}},lease_root=Path({str(lease_root)!r}))\n"
        "except RuntimeError:\n raise SystemExit(73)\n"
        "leases.release()\n")


def test_sigkill_launcher_cannot_release_lease_while_inherited_child_lives(
        tmp_path):
    lease_root = tmp_path / "leases"
    pids_path = tmp_path / "pids.txt"
    uuid = f"GPU-TEST-SIGKILL-{os.getpid()}"
    launcher = subprocess.Popen(
        [sys.executable, "-c", _synthetic_campaign_code(
            lease_root=lease_root, uuid=uuid, pids_path=pids_path,
            handle_signals=False)],
        cwd=Path(__file__).resolve().parents[1])
    pids: list[int] = []
    probe = [sys.executable, "-c", _lease_probe_code(
        lease_root=lease_root, uuid=uuid)]
    try:
        pids = _wait_for_pids(pids_path)
        os.kill(launcher.pid, signal.SIGKILL)
        assert launcher.wait(timeout=2) == -signal.SIGKILL
        assert _process_is_live(pids[0])

        # This is a distinct launcher process.  Parent SIGKILL closed its own
        # fd, but pass_fds left the same kernel flock held by the live child.
        blocked = subprocess.run(
            probe, cwd=Path(__file__).resolve().parents[1], check=False)
        assert blocked.returncode == 73

        os.killpg(pids[0], signal.SIGKILL)
        _wait_processes_dead(pids)
        deadline = time.monotonic() + 3.0
        while True:
            reopened = subprocess.run(
                probe, cwd=Path(__file__).resolve().parents[1], check=False)
            if reopened.returncode == 0 or time.monotonic() >= deadline:
                break
            time.sleep(0.02)
        assert reopened.returncode == 0
    finally:
        if launcher.poll() is None:
            launcher.kill()
            launcher.wait(timeout=2)
        for pid in pids:
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_spawn_registration_window_is_closed_before_shutdown_can_snapshot(
        tmp_path, monkeypatch):
    """A signal-equivalent shutdown cannot miss a Popen still registering."""
    lease_handle = (tmp_path / "lease.lock").open("w+")
    supervisor = _ManagedChildSupervisor(
        [lease_handle.fileno()], term_grace_seconds=0.05,
        kill_wait_seconds=1.0)
    real_popen = subprocess.Popen
    entered_popen = threading.Event()
    allow_popen = threading.Event()
    spawned: list[subprocess.Popen] = []
    outcomes: list[int] = []

    def delayed_popen(*args, **kwargs):
        entered_popen.set()
        assert allow_popen.wait(timeout=2)
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", delayed_popen)

    def launch() -> None:
        process = supervisor.spawn(
            [sys.executable, "-c", "import time;time.sleep(60)"])
        spawned.append(process)
        outcomes.append(supervisor.wait(process))

    worker = threading.Thread(target=launch)
    worker.start()
    assert entered_popen.wait(timeout=2)
    shutdown = threading.Thread(target=supervisor.terminate_all)
    shutdown.start()
    time.sleep(0.05)
    assert shutdown.is_alive()
    allow_popen.set()
    worker.join(timeout=3)
    shutdown.join(timeout=3)
    assert not worker.is_alive() and not shutdown.is_alive()
    assert spawned and outcomes and outcomes[0] != 0
    assert supervisor.active_count == 0
    lease_handle.close()
