"""The operational supervisor (audit 703/704): storage floor, device budget across stages,
wall-time limit, a stop through the launcher's own termination path, and no inference of
death from a timeout.

Every command here is a private synthetic process started by this file. The "launcher" in
the composed cases runs the REAL campaign lifecycle of scripts/phase3_selection_matrix.py
(the lease wrapper, the managed child and the signal handler) with a stand-in GPU inventory
and the real GpuLeaseSet on a private lock directory -- never the host-global lease root, a
GPU, a trainer, a model or a dataset. An unrelated process must survive every stop.
"""
from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import scripts.anchor_confirm_supervisor as S                     # noqa: E402

PY = sys.executable
PLENTY = 1 << 62
#: the same file also runs against the v3 supervisor, which had no watchdog (audit 707.1)
WATCHDOG = "watchdog_seconds" in inspect.signature(S.supervise).parameters

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
    subprocess.Popen([sys.executable, __file__, child, out], close_fds=False)
    while not os.path.exists(os.path.join(out, "child.ready")):
        time.sleep(0.01)
    with open(os.path.join(out, "leader.ready"), "w") as handle:
        handle.write(str(os.getpid()))
    if after == "exit":
        sys.exit(0)
    time.sleep(120)
'''

#: A synthetic launcher: the real lease wrapper and managed child of the campaign launcher,
#: with a stand-in inventory and the private lease root given on the command line.
LAUNCHER = r'''
import os, sys, threading
from pathlib import Path
from types import SimpleNamespace
repo, tree, out, lease_root, mode = sys.argv[1:6]
sys.path.insert(0, repo)
import scripts.phase3_selection_matrix as M
import dna_utils.gpu_lease as L
M.environment_fingerprint = lambda gpus: {"errors": [], "selected_gpus": [
    {"index": g, "uuid": "GPU-aaaaaaaa-bbbb-cccc-dddd-00000000070%d" % g} for g in gpus]}
real = L.acquire_gpu_leases
L.acquire_gpu_leases = lambda a, *, owner_metadata: real(
    a, owner_metadata=owner_metadata, lease_root=Path(lease_root))
outcome = []

def callback():
    stream = threading.Thread(target=lambda: outcome.append(M._run_managed_process(
        [sys.executable, tree, mode, out], cwd=out, env=dict(os.environ))))
    stream.start()
    stream.join()
    return 0 if outcome and outcome[0].returncode == 0 else 1

sys.exit(M._with_campaign_gpu_leases(SimpleNamespace(gpus="0", gpu=0, namespace="sup"), callback))
'''


def live(identity) -> bool:
    pid, start = identity
    try:
        raw = Path(f"/proc/{pid}/stat").read_bytes()
    except (FileNotFoundError, ProcessLookupError):
        return False
    fields = raw[raw.rindex(b")") + 2:].split()
    return int(fields[19]) == start and fields[0] not in (b"Z", b"X", b"x")


def identity_of(path: Path):
    pid = int(path.read_text())
    raw = Path(f"/proc/{pid}/stat").read_bytes()
    return pid, int(raw[raw.rindex(b")") + 2:].split()[19])


def wait_for(predicate, seconds=30.0):
    deadline = time.monotonic() + seconds
    while not predicate():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.02)
    return True


def records(ops: Path) -> list:
    path = ops / S.LEDGER_NAME
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def final(ops: Path) -> dict:
    return [r for r in records(ops) if r["event"] == "final"][-1]


@pytest.fixture
def world(tmp_path):
    """Private dirs, the tree script, an unrelated control, and a teardown that kills only
    this test's own identified processes."""
    ops, out, leases = tmp_path / "ops", tmp_path / "out", tmp_path / "leases"
    out.mkdir()
    leases.mkdir(mode=0o700)
    tree, launcher = tmp_path / "tree.py", tmp_path / "launcher.py"
    tree.write_text(TREE)
    launcher.write_text(LAUNCHER)
    control = subprocess.Popen([PY, "-c", "import time; time.sleep(120)"], start_new_session=True)
    owned = []

    class World:
        pass

    w = World()
    w.ops, w.out, w.leases, w.tree, w.launcher, w.control, w.owned = \
        ops, out, leases, tree, launcher, control, owned

    def ready(name):
        path = out / f"{name}.ready"
        assert wait_for(path.exists), f"{name} never became ready"
        identity = identity_of(path)
        owned.append(identity)
        return identity

    def launch(mode):
        return [PY, str(launcher), str(REPO), str(tree), str(out), str(leases), mode]

    def run(command, **overrides):
        options = dict(stage="stage-S-run", label="test", gpus=1, attempts="sessions",
                       planned_cells=1, watch_path=str(out), ops_root=str(ops),
                       manifest_sha256="f" * 64, poll_seconds=0.2, stop_bound_seconds=2.0,
                       ledger_every=1.0, lease_root=leases, free_bytes=lambda _p: PLENTY)
        if WATCHDOG:
            options["watchdog_seconds"] = 2.0
        else:                            # the v3 supervisor had no watchdog to set
            overrides.pop("watchdog_seconds", None)
        options.update(overrides)
        return S.supervise(command, **options)

    w.ready, w.launch, w.run = ready, launch, run
    yield w
    for pid, start in owned:
        if live((pid, start)):
            os.kill(pid, signal.SIGKILL)
    control.kill()
    control.wait()


def test_the_storage_rule_and_lease_root_are_the_launchers():
    import scripts.phase3_selection_matrix as M
    import dna_utils.gpu_lease as L
    assert (S.FREE_FLOOR_BYTES, S.CELL_OUTPUT_BYTES) == (M.ANCHOR_FREE_FLOOR_BYTES,
                                                         M.ANCHOR_CELL_OUTPUT_BYTES)
    assert S.FREE_FLOOR_BYTES == 10 * S.GiB and S.GPU_LEASE_ROOT == L.GPU_LEASE_ROOT


def test_space_refusal_before_the_command_starts(world):
    marker = world.out / "started"
    need = S.FREE_FLOOR_BYTES + 24 * S.CELL_OUTPUT_BYTES
    rc = world.run([PY, "-c", f"open({str(marker)!r}, 'w')"], planned_cells=24,
                   free_bytes=lambda _p: need - 1)
    assert rc == S.EXIT_REFUSED and not marker.exists()
    assert [r["event"] for r in records(world.ops)] == ["start", "final"]
    assert final(world.ops)["status"] == "refused-before-start"
    assert final(world.ops)["reason"].startswith("space:")
    assert world.run([PY, "-c", "pass"], planned_cells=24, free_bytes=lambda _p: need) == 0


def test_space_breach_while_a_child_lives_stops_through_the_launchers_own_path(world):
    free = {"bytes": PLENTY}
    stopper = threading.Thread(target=lambda: (world.ready("leader"), world.ready("child"),
                                               free.update(bytes=S.FREE_FLOOR_BYTES)))
    stopper.start()
    rc = world.run(world.launch("leader:child-resist:stay"), free_bytes=lambda _p: free["bytes"])
    stopper.join(5)
    assert rc == S.EXIT_STOPPED
    done = final(world.ops)
    assert done["status"] == "stopped" and done["reason"].startswith("space:")
    assert done["returncode"] == 128 + signal.SIGTERM        # the launcher's own handler
    assert all(not live(identity) for identity in world.owned)  # TERM-resistant one too
    assert done["orphaned_live_attempts"] == [] and done["leases_held_after_exit"] == []
    assert done["lease_gpu_seconds"] > 0 and len(done["attempts"]) == 1
    assert Path(done["command_log"]).exists()
    assert world.control.poll() is None                      # unrelated process untouched


def test_budget_breach_while_a_child_lives_stops_it(world):
    # headroom 1 x (2.0 watchdog + 2.0 stop bound) leaves 1.8 s for the attempt and its allowance
    budget = 4.0 + 1.8
    rc = world.run(world.launch("leader:child-cooperative:stay"), budget_seconds=budget)
    assert rc == S.EXIT_STOPPED
    done = final(world.ops)
    assert done["reason"].startswith("budget:")
    stop = [r for r in records(world.ops) if r["event"] == "stop"][-1]
    assert stop["projected_seconds"] >= budget and done["device_seconds"] > 0.5
    assert all(not live(identity) for identity in world.owned)
    assert world.control.poll() is None


def test_failed_attempts_are_charged_and_carried_into_the_next_stage(world):
    command = [PY, str(world.launcher), str(REPO), str(world.tree), str(world.out),
               str(world.leases), "leader:child-cooperative:exit"]
    # The leader exits at once but its descendant lives on: the launcher's stream stops it
    # (bounded) and refuses the cell, so the stage fails with a partial attempt.
    first = world.run(command, stage="stage-S-run")
    assert first == 1
    done = final(world.ops)
    assert done["status"] == "exited" and done["returncode"] == 1
    assert len(done["attempts"]) == 1 and done["device_seconds"] > 0
    charged = done["charged_seconds"]
    assert world.run([PY, "-c", "pass"], stage="stage-D-run", planned_cells=0) == 0
    start = [r for r in records(world.ops) if r["event"] == "start"][-1]
    assert start["prior_charged_seconds"] == pytest.approx(charged)
    assert final(world.ops)["cumulative_charged_seconds"] >= charged
    # ... and the next stage is refused once the carried charge meets the budget
    assert world.run([PY, "-c", "pass"], stage="probe", attempts="self", planned_cells=0,
                     budget_seconds=charged + 1.0) == S.EXIT_REFUSED
    assert final(world.ops)["reason"].startswith("budget:")


def test_a_failed_observation_stops_dispatch(world):
    calls = []

    def flaky(_path):
        calls.append(1)
        if len(calls) > 3:
            raise OSError("statvfs failed")
        return PLENTY

    rc = world.run([PY, "-c", "import time; time.sleep(60)"], attempts="self", stage="probe",
                   planned_cells=0, free_bytes=flaky)
    assert rc == S.EXIT_UNRESOLVED                # what was missed while blind is unknown
    done = final(world.ops)
    assert done["reason"].startswith("monitor failure:") and done["status"] == "unresolved"
    with pytest.raises(S.Refused, match="lost observation continuity"):
        world.run([PY, "-c", "pass"], attempts="self", stage="probe", planned_cells=0)


def test_a_command_slower_than_the_stop_bound_is_waited_for_not_declared_dead(world):
    slow = ("import signal, sys, time\n"
            "signal.signal(signal.SIGTERM, lambda *a: (time.sleep(2.0), sys.exit(0)))\n"
            "time.sleep(60)\n")
    began = time.monotonic()
    rc = world.run([PY, "-c", slow], attempts="self", stage="probe", planned_cells=0,
                   wall_limit_seconds=2.5, stop_bound_seconds=0.5, watchdog_seconds=1.0)
    assert rc == S.EXIT_STOPPED and time.monotonic() - began >= 2.0
    events = [r["event"] for r in records(world.ops)]
    assert "stop-overdue" in events and events.index("stop-overdue") < events.index("final")
    done = final(world.ops)
    assert done["reason"].startswith("wall:") and done["returncode"] == 0
    assert done["wall_seconds"] >= 2.0


def test_an_operator_signal_becomes_one_sigterm_to_the_command(world):
    before = signal.getsignal(signal.SIGTERM)
    def started():
        try:                           # the ledger may be mid-write
            return any(r["event"] == "attempt-start" for r in records(world.ops))
        except ValueError:
            return False

    sender = threading.Thread(target=lambda: (wait_for(started),
                                              os.kill(os.getpid(), signal.SIGTERM)))
    sender.start()
    rc = world.run([PY, "-c", "import time; time.sleep(60)"], attempts="self", stage="probe",
                   planned_cells=0)
    sender.join(5)
    assert rc == S.EXIT_STOPPED
    assert final(world.ops)["reason"] == "operator signal SIGTERM"
    assert final(world.ops)["returncode"] == -signal.SIGTERM
    assert signal.getsignal(signal.SIGTERM) is before


def test_a_launcher_that_leaves_a_live_attempt_is_unclean_and_blocks_the_ledger(world):
    crash = (f"import subprocess, sys, time\n"
             f"subprocess.Popen([sys.executable, {str(world.tree)!r}, 'child-cooperative', "
             f"{str(world.out)!r}], start_new_session=True)\n"
             f"import os\n"
             f"while not os.path.exists({str(world.out / 'child.ready')!r}): time.sleep(0.01)\n"
             f"time.sleep(0.6)\n")
    rc = world.run([PY, "-c", crash])
    orphan = world.ready("child")
    assert rc == S.EXIT_UNCLEAN
    done = final(world.ops)
    assert done["status"] == "unclean" and done["orphaned_live_attempts"] == [list(orphan)]
    assert live(orphan)                    # never signalled by the supervisor
    with pytest.raises(S.Refused, match="ended with live attempts"):
        world.run([PY, "-c", "pass"])


def test_an_unfinished_prior_run_and_a_second_supervisor_are_refused(world):
    world.ops.mkdir()
    (world.ops / S.LEDGER_NAME).write_text(json.dumps(
        {"schema": S.LEDGER_SCHEMA, "event": "start", "run_id": "r0"}) + "\n")
    with pytest.raises(S.Refused, match="no final record"):
        world.run([PY, "-c", "pass"])
    holder = S.Ledger(world.ops)
    try:
        with pytest.raises(S.Refused, match="another supervisor"):
            S.Ledger(world.ops)
    finally:
        holder.close()


def test_the_generation_check(tmp_path):
    good = {"new_generation": {"files_sha256": {S.SELF_PATH: S._IMPORTED_SOURCE_SHA256}}}
    path = tmp_path / "m.json"
    path.write_bytes(json.dumps(good).encode())
    import hashlib
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert S.verify_generation(path, sha) == good["new_generation"]["files_sha256"]
    with pytest.raises(S.Refused, match="not the pinned manifest"):
        S.verify_generation(path, "0" * 64)
    for files, match in (({}, "not the manifest's generation"),
                         ({S.SELF_PATH: S._IMPORTED_SOURCE_SHA256, "config.py": "0" * 64},
                          "the tree is not")):
        path.write_bytes(json.dumps({"new_generation": {"files_sha256": files}}).encode())
        with pytest.raises(S.Refused, match=match):
            S.verify_generation(path, hashlib.sha256(path.read_bytes()).hexdigest())


def test_the_cli_takes_the_gpu_count_from_the_launcher_and_refuses_before_starting(tmp_path):
    assert S.command_gpus("stage-S-run", ["x", "--gpus", "0,1,2", "--run"]) == 3
    assert S.command_gpus("probe", ["x"]) == 1
    for command, match in ((["x", "--run"], "names its GPUs once"),
                           (["x", "--gpus", "0,1,2,3"], "1 to 3 GPUs")):
        with pytest.raises(S.Refused, match=match):
            S.command_gpus("stage-D-run", command)
    marker = tmp_path / "started"
    rc = S.main(["--manifest", str(tmp_path / "missing.json"), "--manifest-sha256", "0" * 64,
                 "--stage", "probe", "--planned-cells", "0", "--watch-path", str(tmp_path),
                 "--ops-root", str(tmp_path / "ops"), "--",
                 PY, "-c", f"open({str(marker)!r}, 'w')"])
    assert rc == S.EXIT_REFUSED and not marker.exists()


# -- audit 707.1: an attempt between two observations, and a stalled observation ------------------

#: A launcher that waits for a release, then runs managed-like children one at a time (each its
#: own session leader) that stamp their own boot-clock start and end. Like the campaign launcher,
#: SIGTERM stops its current child before it dies.
GAP_LAUNCHER = r"""
import os, signal, subprocess, sys, time
go, stamp, runs = sys.argv[1], sys.argv[2], int(sys.argv[3])
current = []

def term(*_):
    # Popen.wait() here would re-enter the lock held by the interrupted wait
    # below; kill and reap by PID instead.
    for child in current:
        os.kill(child.pid, signal.SIGKILL)
        try:
            os.waitpid(child.pid, 0)
        except ChildProcessError:
            pass
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    os.kill(os.getpid(), signal.SIGTERM)

signal.signal(signal.SIGTERM, term)
signal.alarm(60)                     # a fixture never outlives a minute, whatever else fails
while not os.path.exists(go):
    time.sleep(0.01)
for n in range(runs):
    current[:] = [subprocess.Popen([sys.executable, "-c",
        "import sys, time\n"
        "path = sys.argv[1]\n"
        "open(path, 'w').write(repr(time.clock_gettime(time.CLOCK_BOOTTIME)))\n"
        "time.sleep(float(sys.argv[2]))\n"
        "open(path, 'a').write(' ' + repr(time.clock_gettime(time.CLOCK_BOOTTIME)))\n",
        f"{stamp}.{n}", "1.5" if runs == 1 else "0.8"], start_new_session=True)]
    current[0].wait()
    current.clear()
time.sleep(0.3 if runs == 1 else 30)
"""


def stamped(path: Path) -> float:
    """The child's own boot-clock lifetime (a lower bound of its process lifetime)."""
    assert wait_for(lambda: path.exists() and len(path.read_text().split()) == 2, 10)
    start, end = map(float, path.read_text().split())
    return end - start


def gap(world, block_seconds):
    """free_bytes that releases the launcher right after the first live scan, then blocks."""
    go = world.out / "go"
    calls = []

    def free(_path):
        calls.append(S.boot_seconds())
        if len(calls) == 2:              # call 1 is the pre-start check
            go.touch()
            time.sleep(block_seconds)
            calls.append(S.boot_seconds())
        return PLENTY

    script = world.out / "gap_launcher.py"
    script.write_text(GAP_LAUNCHER)
    return free, [PY, str(script), str(go), str(world.out / "stamp")]


def test_an_attempt_that_lives_between_two_observations_is_still_charged(world):
    free, command = gap(world, 2.5)
    rc = world.run(command + ["1"], planned_cells=1, poll_seconds=0.1, watchdog_seconds=10.0,
                   free_bytes=free)
    lived = stamped(world.out / "stamp.0")
    done = final(world.ops)
    assert done["attempts"] == []                       # no scan ever saw it
    assert done["charged_seconds"] >= lived >= 1.5      # f634fc55 charged 0.1 (audit 707.1)
    assert done["max_observation_window_seconds"] >= 2.5
    assert rc == 0 and done["status"] == "exited"       # within the watchdog bound: a valid bound


def test_a_normally_observed_attempt_is_charged_from_its_own_lifetime(world):
    """Control: the same child, observed."""
    go = world.out / "go"
    calls = []

    def free(_path):
        calls.append(1)
        if len(calls) == 2:
            go.touch()
        return PLENTY

    script = world.out / "gap_launcher.py"
    script.write_text(GAP_LAUNCHER)
    rc = world.run([PY, str(script), str(go), str(world.out / "stamp"), "1"], planned_cells=1,
                   poll_seconds=0.1, free_bytes=free)
    lived = stamped(world.out / "stamp.0")
    done = final(world.ops)
    assert rc == 0 and done["status"] == "exited" and len(done["attempts"]) == 1
    assert done["device_seconds"] >= lived >= 1.5
    assert done["max_observation_window_seconds"] < 1.0


def test_an_observation_stalled_past_the_watchdog_stops_the_command_and_is_unresolved(world):
    free, command = gap(world, 2.5)
    rc = world.run(command + ["1"], planned_cells=1, poll_seconds=0.1, watchdog_seconds=1.0,
                   free_bytes=free)
    assert (world.out / "stamp.0").exists()             # an attempt ran during the stall
    done = final(world.ops)
    assert rc == S.EXIT_UNRESOLVED and done["status"] == "unresolved"
    assert done["reason"].startswith("watchdog:")
    assert done["returncode"] == -signal.SIGTERM        # stopped while the observation hung
    assert done["pessimistic_bound_seconds"] >= done["wall_seconds"]
    with pytest.raises(S.Refused, match="lost observation continuity"):
        world.run([PY, "-c", "pass"], attempts="self", stage="probe", planned_cells=0)


def test_the_watchdog_signals_before_the_stalled_observation_returns(world):
    """Independent timing: the command stamps the moment SIGTERM reaches it."""
    stamp = world.out / "term"
    command = [PY, "-c",
               "import signal, sys, time\n"
               "def term(*_):\n"
               f"    open({str(stamp)!r}, 'w').write(repr(time.clock_gettime(time.CLOCK_BOOTTIME)))\n"
               "    sys.exit(0)\n"
               "signal.signal(signal.SIGTERM, term)\n"
               "time.sleep(60)\n"]
    blocked = []

    def free(_path):
        blocked.append(S.boot_seconds())
        if len(blocked) == 3:
            time.sleep(3.0)
            blocked.append(S.boot_seconds())
        return PLENTY

    rc = world.run(command, attempts="self", stage="probe", planned_cells=0, poll_seconds=0.1,
                   watchdog_seconds=1.0, free_bytes=free)
    assert rc == S.EXIT_UNRESOLVED
    signalled = float(stamp.read_text())
    assert blocked[2] + 0.9 <= signalled < blocked[3]   # during the stall, after the bound
    assert final(world.ops)["reason"].startswith("watchdog:")


def test_more_attempts_than_planned_cells_void_the_allowance(world):
    go = world.out / "go"
    go.touch()
    script = world.out / "gap_launcher.py"
    script.write_text(GAP_LAUNCHER)
    rc = world.run([PY, str(script), str(go), str(world.out / "stamp"), "2"], planned_cells=1,
                   poll_seconds=0.1)
    done = final(world.ops)
    assert rc == S.EXIT_UNRESOLVED and done["status"] == "unresolved"
    assert done["reason"].startswith("continuity lost: attempts: 2 observed for 1")


def test_the_headroom_covers_the_watchdog_and_the_stop_bound(world):
    common = dict(attempts="self", stage="probe", planned_cells=0, watchdog_seconds=2.0,
                  stop_bound_seconds=2.0)
    assert world.run([PY, "-c", "pass"], budget_seconds=4.0 - 0.01, **common) == S.EXIT_REFUSED
    assert final(world.ops)["reason"].startswith("budget:")
    assert world.run([PY, "-c", "pass"], budget_seconds=4.5, **common) == 0


def test_a_long_window_counts_against_the_budget_while_the_command_runs(world):
    """The live projection charges the longest window seen so far, not the nominal poll."""
    blocked = []

    def free(_path):
        blocked.append(1)
        if len(blocked) == 3:
            time.sleep(2.5)
        return PLENTY

    # headroom 1 x (10 + 2) = 12; ten planned cells charge 10 x the window: about 1-2 s while
    # polls are regular, about 26 s after one 2.5-s window
    rc = world.run([PY, "-c", "import time; time.sleep(60)"], planned_cells=10, poll_seconds=0.1,
                   watchdog_seconds=10.0, budget_seconds=12.0 + 10.0, free_bytes=free)
    assert rc == S.EXIT_STOPPED
    done = final(world.ops)
    assert done["reason"].startswith("budget:")
    assert done["unobserved_allowance_seconds"] >= 25.0
