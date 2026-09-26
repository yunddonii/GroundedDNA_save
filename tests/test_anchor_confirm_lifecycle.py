"""Campaign child lifecycle (audit 705/706): cleanup completes on live descendants,
never on a leader's exit or an empty registry, and no GPU lease is released while
an owned process lives.

Every process here is a private Python sleeper started by this file; the lease is
the real GpuLeaseSet on a private flock directory under tmp_path (never the
host-global GPU lease root). No trainer, model, dataset, GPU or real artifact is
touched. An unrelated sleeper in its own session must survive every cleanup.
"""
from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import scripts.phase3_selection_matrix as M                      # noqa: E402

TREE = r'''
import os, signal, subprocess, sys, time
mode, out = sys.argv[1], sys.argv[2]
if mode.startswith("child-"):
    if mode == "child-resist":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    with open(os.path.join(out, "child.ready"), "w") as handle:
        handle.write(str(os.getpid()))
    time.sleep(120)
else:
    _, child, after = mode.split(":")
    # close_fds=False: like bash and forked DataLoader workers, the descendant
    # inherits the launcher's lease descriptor.
    subprocess.Popen([sys.executable, __file__, child, out], close_fds=False)
    while not os.path.exists(os.path.join(out, "child.ready")):
        time.sleep(0.01)
    with open(os.path.join(out, "leader.ready"), "w") as handle:
        handle.write(str(os.getpid()))
    if after == "exit":
        sys.exit(0)
    time.sleep(120)
'''


def start_ticks(pid: int) -> int | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_bytes()
    except (FileNotFoundError, ProcessLookupError):
        return None
    return int(raw[raw.rindex(b")") + 2:].split()[19])


def live(identity) -> bool:
    """The very process (PID and start time) exists and is not a zombie -- read
    here from /proc, independently of the launcher's own scanner."""
    pid, start = identity
    try:
        raw = Path(f"/proc/{pid}/stat").read_bytes()
    except (FileNotFoundError, ProcessLookupError):
        return False
    fields = raw[raw.rindex(b")") + 2:].split()
    return int(fields[19]) == start and fields[0] not in (b"Z", b"X", b"x")


def state(pid: int) -> str | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_bytes()
    except (FileNotFoundError, ProcessLookupError):
        return None
    return raw[raw.rindex(b")") + 2:].split()[0].decode()


def pidfd_open(pid: int) -> int:
    """This file's own pidfd (the teardown must not depend on the code under test)."""
    import ctypes
    descriptor = ctypes.CDLL(None, use_errno=True).syscall(434, ctypes.c_int(pid), ctypes.c_uint(0))
    if descriptor < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return descriptor


def wait_for(predicate, seconds=10.0):
    deadline = time.monotonic() + seconds
    while not predicate():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.01)
    return True


@pytest.fixture
def lifecycle(tmp_path, monkeypatch):
    """Isolated launcher lifecycle state, short bounds, an unrelated control and
    a teardown that kills only this test's own identified processes."""
    monkeypatch.setattr(M, "_ACTIVE_CHILDREN", set())
    monkeypatch.setattr(M, "_CHILD_LAUNCH_BLOCKED", False)
    monkeypatch.setattr(M, "_CAMPAIGN_LEASE_FDS", ())
    # raising=False: the same file also runs against the pre-repair launcher (audit 705).
    monkeypatch.setattr(M, "_DRAIN_SECONDS", 30.0, raising=False)
    monkeypatch.setattr(M, "_TERM_GRACE_SECONDS", 0.5, raising=False)
    monkeypatch.setattr(M, "_KILL_CONFIRM_SECONDS", 5.0, raising=False)
    script = tmp_path / "tree.py"
    script.write_text(TREE)
    control = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"],
                               start_new_session=True)
    owned = []
    box = SimpleNamespace(script=script, out=tmp_path, control=control, owned=owned)

    def ready(name):
        path = tmp_path / f"{name}.ready"
        assert wait_for(path.exists), f"{name} never became ready"
        pid = int(path.read_text())
        identity = (pid, start_ticks(pid))
        owned.append(identity)
        return identity

    box.ready = ready
    yield box
    for pid, start in owned:
        if live((pid, start)):
            try:
                descriptor = pidfd_open(pid)
            except ProcessLookupError:
                continue
            try:
                if start_ticks(pid) == start:
                    signal.pidfd_send_signal(descriptor, signal.SIGKILL)
            finally:
                os.close(descriptor)
    control.kill()
    control.wait()


def worker(tmp_path, command):
    """_run_managed_process from a stream-like worker thread."""
    outcome = {}

    def run():
        try:
            outcome["result"] = M._run_managed_process(command, cwd=str(tmp_path),
                                                       env=dict(os.environ))
        except BaseException as error:              # noqa: BLE001 -- recorded
            outcome["error"] = error

    thread = threading.Thread(target=run)
    thread.start()
    return thread, outcome


def tree(box, mode):
    return [sys.executable, str(box.script), mode, str(box.out)]


# -- the three cases of audit 705.1 ------------------------------------------------------------

def test_cooperative_sleeper_tree_is_drained_by_cleanup(lifecycle):
    """Positive control: a tree that obeys SIGTERM."""
    thread, outcome = worker(lifecycle.out, tree(lifecycle, "leader:child-cooperative:stay"))
    leader, child = lifecycle.ready("leader"), lifecycle.ready("child")
    M._terminate_active_children()
    assert not live(leader) and not live(child)
    thread.join(10)
    assert not thread.is_alive() and outcome["result"].returncode == -signal.SIGTERM
    assert not M._ACTIVE_CHILDREN
    assert lifecycle.control.poll() is None             # unrelated process untouched


def test_term_resistant_descendant_is_killed_although_its_leader_dies_on_term(lifecycle):
    thread, outcome = worker(lifecycle.out, tree(lifecycle, "leader:child-resist:stay"))
    leader, child = lifecycle.ready("leader"), lifecycle.ready("child")
    began = time.monotonic()
    M._terminate_active_children()
    elapsed = time.monotonic() - began
    assert not live(leader)
    assert not live(child)                              # was LIVE at 1badc49 (audit 705.1)
    assert elapsed < 0.5 + 5.0 + 5.0                    # bounded: grace + kill bound + scans
    thread.join(10)
    assert not thread.is_alive() and outcome["result"].returncode == -signal.SIGTERM
    assert not M._ACTIVE_CHILDREN
    assert lifecycle.control.poll() is None


def test_descendant_of_a_leader_that_exited_first_is_found_by_cleanup(lifecycle):
    thread, outcome = worker(lifecycle.out, tree(lifecycle, "leader:child-resist:exit"))
    leader, child = lifecycle.ready("leader"), lifecycle.ready("child")
    assert wait_for(lambda: not live(leader))
    # The leader exited, but its record stays registered and the leader is kept
    # an unreaped zombie: its PID pins the session while a descendant lives.
    pinned = live(child) and state(leader[0]) == "Z" and bool(M._ACTIVE_CHILDREN)
    M._terminate_active_children()
    assert not live(child)                              # was LIVE at 1badc49 (audit 705.1)
    assert pinned
    thread.join(10)
    assert not thread.is_alive() and outcome["result"].returncode == 0
    assert not M._ACTIVE_CHILDREN and state(leader[0]) is None     # reaped only now
    assert lifecycle.control.poll() is None


def test_a_leftover_descendant_is_stopped_by_its_own_stream_and_the_cell_refused(
        lifecycle, monkeypatch):
    monkeypatch.setattr(M, "_DRAIN_SECONDS", 0.3)
    thread, outcome = worker(lifecycle.out, tree(lifecycle, "leader:child-resist:exit"))
    lifecycle.ready("leader")
    child = lifecycle.ready("child")
    thread.join(0.3 + 0.5 + 5.0 + 10)
    assert not thread.is_alive()
    assert not live(child)                              # 1badc49 returned with it LIVE
    assert isinstance(outcome.get("error"), M.CellRefused)
    assert "outlived its leader" in str(outcome["error"])
    assert not M._ACTIVE_CHILDREN
    assert lifecycle.control.poll() is None


# -- bounded termination; a timeout is reported, never taken as death ------------------------------

def test_a_session_that_outlives_sigkill_is_reported_within_the_bound_and_kept(
        lifecycle, monkeypatch):
    """Stand-in: an unkillable (uninterruptible) process cannot be made on purpose, so the
    scanner reports one phantom live member for this session only."""
    thread, outcome = worker(lifecycle.out, tree(lifecycle, "leader:child-cooperative:stay"))
    leader, child = lifecycle.ready("leader"), lifecycle.ready("child")
    real = M._session_live_members
    monkeypatch.setattr(M, "_session_live_members",
                        lambda sid: real(sid) + ([(2 ** 22 + 7, sid)] if sid == leader[0] else []))
    began = time.monotonic()
    with pytest.raises(RuntimeError, match="still have live processes after SIGKILL"):
        M._terminate_active_children(grace_seconds=0.2, kill_seconds=0.5)
    assert time.monotonic() - began < 0.2 + 0.5 + 5.0
    assert not live(leader) and not live(child)         # the real ones are gone
    assert [proc.pid for proc in M._ACTIVE_CHILDREN] == [leader[0]]   # still registered
    assert state(leader[0]) == "Z"                      # never reaped on a timeout
    monkeypatch.setattr(M, "_session_live_members", real)
    M._terminate_active_children()
    thread.join(10)
    assert not thread.is_alive() and not M._ACTIVE_CHILDREN


def test_a_reaped_record_is_never_signalled(monkeypatch):
    """After the reap its PID may name someone else's group: no signal at all."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    assert wait_for(lambda: os.waitid(os.P_PID, proc.pid,
                                      os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None)
    sent = []
    monkeypatch.setattr(M.os, "killpg", lambda pgid, signum: sent.append((pgid, signum)))
    M._signal_owned_session(proc, signal.SIGKILL)       # unreaped zombie: still ours
    assert sent == [(proc.pid, signal.SIGKILL)]
    proc.wait()
    M._signal_owned_session(proc, signal.SIGKILL)       # reaped: never again
    assert sent == [(proc.pid, signal.SIGKILL)]


def test_cleanup_blocks_later_launches(lifecycle):
    M._terminate_active_children()
    thread, outcome = worker(lifecycle.out, tree(lifecycle, "leader:child-cooperative:stay"))
    thread.join(10)
    assert isinstance(outcome.get("error"), M.CellRefused)
    assert "shutdown has begun" in str(outcome["error"])
    assert not (lifecycle.out / "leader.ready").exists()


# -- composed with the exact lease wrapper and the real GpuLeaseSet (audit 706) -------------------

UUID = "GPU-aaaaaaaa-bbbb-cccc-dddd-000000000706"


@pytest.fixture
def lease(lifecycle, tmp_path, monkeypatch):
    """The real acquire/release on a private lease root; release is observed, not replaced."""
    import dna_utils.gpu_lease as lease_module
    root = tmp_path / "leases"
    root.mkdir(mode=0o700)
    monkeypatch.setattr(M, "environment_fingerprint", lambda gpus: {
        "errors": [], "selected_gpus": [{"index": gpus[0], "uuid": UUID}]})
    real_acquire = lease_module.acquire_gpu_leases
    held = []
    monkeypatch.setattr(lease_module, "acquire_gpu_leases",
                        lambda assignments, *, owner_metadata: held.append(real_acquire(
                            assignments, owner_metadata=owner_metadata, lease_root=root))
                        or held[-1])
    at_release = []
    real_release = lease_module.GpuLeaseSet.release

    def release(self):
        at_release.append({"owned_live": [identity for identity in lifecycle.owned
                                          if live(identity)],
                           "lock_free": lock_free(), "time": time.monotonic()})
        return real_release(self)

    monkeypatch.setattr(lease_module.GpuLeaseSet, "release", release)
    # The interpreter-exit handler that acquire registers is recorded here instead of being
    # registered in pytest's own process; the launcher must unregister it (its release is gated).
    import atexit
    registered, unregistered = [], []
    monkeypatch.setattr(atexit, "register", lambda function, *a, **k: registered.append(function))
    monkeypatch.setattr(atexit, "unregister", lambda function: unregistered.append(function))

    def exit_release_pending():
        return [f for f in registered if not any(f == g for g in unregistered)]

    def lock_free() -> bool:
        """Another open file description tries the same private flock."""
        import fcntl
        with open(root / f"{UUID}.lock", "a") as other:
            try:
                fcntl.flock(other.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            fcntl.flock(other.fileno(), fcntl.LOCK_UN)
            return True

    box = SimpleNamespace(held=held, at_release=at_release, lock_free=lock_free,
                          exit_release_pending=exit_release_pending,
                          args=SimpleNamespace(gpus=None, gpu=0, namespace="lifecycle-fixture"))
    yield box
    for leases in held:           # a failed test must not leave the private lock held
        real_release(leases)


@pytest.mark.parametrize("mode", ["leader:child-cooperative:stay", "leader:child-resist:stay"])
def test_a_real_sigterm_releases_the_lease_only_after_every_owned_process_died(
        lifecycle, lease, mode):
    threads = []
    before = signal.getsignal(signal.SIGTERM)

    def callback():
        threads.append(worker(lifecycle.out, tree(lifecycle, mode)))
        lifecycle.ready("leader")
        lifecycle.ready("child")
        assert not lease.lock_free()                    # held inside the callback
        signal.raise_signal(signal.SIGTERM)             # the real handler path
        raise AssertionError("the campaign handler did not stop the callback")

    with pytest.raises(SystemExit) as stopped:
        M._with_campaign_gpu_leases(lease.args, callback)
    assert stopped.value.code == 128 + signal.SIGTERM
    assert len(lease.at_release) == 1
    assert lease.at_release[0]["owned_live"] == []      # no owned process at release
    assert lease.at_release[0]["lock_free"] is False    # released by this call, not before
    assert lease.lock_free()
    assert signal.getsignal(signal.SIGTERM) is before
    thread, _outcome = threads[0]
    thread.join(10)
    assert not thread.is_alive()
    assert lifecycle.control.poll() is None


def test_a_leader_that_exited_first_does_not_let_the_lease_go_early(lifecycle, lease):
    threads = []

    def callback():
        threads.append(worker(lifecycle.out, tree(lifecycle, "leader:child-resist:exit")))
        leader = lifecycle.ready("leader")
        lifecycle.ready("child")
        assert wait_for(lambda: not live(leader))
        return 17

    assert M._with_campaign_gpu_leases(lease.args, callback) == 17
    assert len(lease.at_release) == 1
    assert lease.at_release[0]["owned_live"] == []      # the descendant died first
    assert lease.lock_free()
    thread, outcome = threads[0]
    thread.join(10)
    assert not thread.is_alive() and outcome["result"].returncode == 0
    assert lifecycle.control.poll() is None


def test_a_survivor_keeps_the_lease_held_and_the_error_propagates(lifecycle, lease, monkeypatch):
    """Stand-in survivor as above; the kernel lock must still be held afterwards."""
    monkeypatch.setattr(M, "_KILL_CONFIRM_SECONDS", 0.5)
    real = M._session_live_members
    threads, leaders = [], []

    def callback():
        threads.append(worker(lifecycle.out, tree(lifecycle, "leader:child-cooperative:stay")))
        leaders.append(lifecycle.ready("leader"))
        lifecycle.ready("child")
        monkeypatch.setattr(M, "_session_live_members",
                            lambda sid: real(sid) + ([(2 ** 22 + 7, sid)]
                                                     if sid == leaders[0][0] else []))
        return 0

    with pytest.raises(RuntimeError, match="still have live processes after SIGKILL"):
        M._with_campaign_gpu_leases(lease.args, callback)
    assert lease.at_release == []                       # never released
    assert not lease.lock_free()                        # the kernel lock is still held
    assert lease.held and lease.exit_release_pending() == []   # nor at interpreter exit
    assert M._CHILD_LAUNCH_BLOCKED and M._CAMPAIGN_LEASE_FDS
    monkeypatch.setattr(M, "_session_live_members", real)
    M._terminate_active_children()
    threads[0][0].join(10)
    assert lifecycle.control.poll() is None
