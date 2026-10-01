"""Operational supervisor for ONE anchor-confirmation command (audit 703-707).

It supplies what the launcher and tmux_run.sh do not: a free-space floor watched
while cells run, the S + D + probes device budget carried across stages in one
append-only ledger, a wall-time limit per stage, and a stop that goes through the
launcher's own supervised termination path (SIGTERM to the launcher, whose
handler drains every owned session before it releases the GPU leases).

What it never does: signal anything except the one command it started, acquire
or test-lock a GPU lease, delete or move an artifact, or infer that a process
died from an observation timeout. A failed observation stops dispatch.

Accounting (audit 704.3, 707.1). ``device_seconds`` is the sum, over attempts,
of the attempt's process lifetime: the start is the kernel's start time of the
process, the end is the first scan that no longer sees it (an upper bound). An
attempt is a managed campaign child (a child session leader of the launcher --
one trainer on one GPU) or, for a probe, the probe process itself, which cannot
vanish before this process reaps it. It is an upper bound on per-attempt device
occupancy, NOT measured GPU utilisation.

A launcher attempt can start and end between two scans. Such an attempt lies
inside one observation window -- from the START of one scan to the END of the
next, measured on the boot clock, so every stat, scan, ledger write or scheduling
delay is inside it -- and the launcher runs one managed child per stage-1 cell
without retries, so ``unobserved_allowance_seconds`` = planned cells x the
longest window actually observed. More attempts observed than cells planned
voids that premise. A watchdog thread stops the command when no observation has
completed for WATCHDOG_SECONDS, so work can outrun a stalled observation by at
most that; the budget reserves GPUs x (WATCHDOG_SECONDS + STOP_BOUND_SECONDS).

If observation continuity is lost -- a window longer than the watchdog bound, a
watchdog stop, a failed observation or ledger write, or more attempts than cells
-- the run is stopped and its final record is ``unresolved``: its charge is
recorded with ``pessimistic_bound_seconds`` (GPUs x wall time), and no later
stage starts until the audit reconciles it; the same holds for a run without a
final record and for an ``unclean`` one (live attempts or held leases at exit).

``lease_gpu_seconds`` is the separate, larger idle-inclusive quantity: GPUs
leased by the launcher (from /proc/locks) times time. CPU input verification
before the lease is ``pre_lease_seconds``: separate time, not device work. The
budget binds charged work across every stage in the ledger.

Thresholds are constants of this pinned file, so the generation manifest that
names these bytes names them too.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import threading
import time

#: the bytes this module was imported from (audit 697)
with open(__file__, "rb") as _source:
    _IMPORTED_SOURCE_SHA256 = hashlib.sha256(_source.read()).hexdigest()

REPO = Path(__file__).resolve().parents[1]
SELF_PATH = "scripts/anchor_confirm_supervisor.py"
LEDGER_SCHEMA = "anchor-confirm-ops-ledger/1"
LEDGER_NAME = "device_budget_ledger.jsonl"
LOCK_NAME = "supervisor.lock"
#: on the root filesystem with the result root (user decision 2026-09-27: /data is 99 % used)
DEFAULT_OPS_ROOT = Path("/home/yschoi/gdna_anchor4_ops")
GPU_LEASE_ROOT = Path(f"/tmp/groundeddna-d6-gpu-leases-{os.getuid()}")

GiB = 1 << 30
#: the launcher's storage rule (ANCHOR_FREE_FLOOR_BYTES / ANCHOR_CELL_OUTPUT_BYTES); a test holds
#: the two files to one pair of values
FREE_FLOOR_BYTES = 10 << 30
CELL_OUTPUT_BYTES = 3 << 28                    # 0.75 GiB; a historical N39 cell holds 0.61 GiB
BUDGET_DEVICE_SECONDS = 12.5 * 3600.0          # S + D + probes (contract v3 section 12)
#: Stage L (contract L v1 section 8, Flickr25K first): its own budget in its own ledger root, so the
#: closed S/D/probe ledger is neither reused nor reset. Smoke + continuity control + five candidates,
#: a failed attempt, the observation allowance and the stop headroom.
L_STAGES = ("stage-L-smoke", "stage-L-run")
L_BUDGET_DEVICE_SECONDS = 1.0 * 3600.0
L_OPS_ROOT = Path("/home/yschoi/gdna_anchorL_ops")
#: Stages R and T (generation v9, contract R/T v1 section 9): their own ledger root and one budget,
#: so neither settled ledger (S/D/probe, L) is reused or reset. Planning reference: the twelve
#: incumbent p3rfB refits with their fused official-test chain took 66,415 s of wall time in total;
#: the proposed ceiling covers both smokes, R, T and that margin. A proposal, set by the audit.
RT_STAGES = ("stage-R-smoke", "stage-R-run", "stage-T-smoke", "stage-T-run")
RT_BUDGET_DEVICE_SECONDS = 80000.0
RT_OPS_ROOT = Path("/home/yschoi/gdna_anchorRT_ops")
POLL_SECONDS = 1.0
#: no completed observation for this long stops the command (and voids the run's settlement)
WATCHDOG_SECONDS = 10.0
LEDGER_EVERY_SECONDS = 30.0
#: the launcher's own cleanup is bounded by TERM 5 s + KILL 30 s plus /proc scans; three times that
STOP_BOUND_SECONDS = 120.0
#: stage -> (wall-time limit, most GPUs). Contract v3 runs one stream per dataset, four GPUs. S and
#: D: full input verification of the four stage-1 seals (about 532 GB; the historical four-dataset
#: figure is 82 min, an estimate) plus the slowest anchor stream (NUS-WIDE: 75 epochs x 1.08 min =
#: 81 min for S; at most 2 x 40 epochs = 86 min for D), about 2.8 h, with about 1.4x margin.
STAGES = {
    "stage-S-run": (4 * 3600.0, 4), "stage-S-smoke": (2 * 3600.0, 4),
    "stage-D-run": (4 * 3600.0, 4), "stage-D-smoke": (2 * 3600.0, 4),
    "probe": (15 * 60.0, 1),
    # stage L: one Flickr25K stream (control + five N=4 cells, about 2 to 5 minutes each)
    "stage-L-smoke": (2 * 3600.0, 1), "stage-L-run": (2 * 3600.0, 1),
    # stages R and T: one stream per dataset. The slowest incumbent stream (NUS-WIDE, R and T fused)
    # took 24,962 s for three seeds; a smoke is one cell, and the R smoke may include the full
    # historical verification of the refit seals.
    "stage-R-smoke": (3 * 3600.0, 1), "stage-R-run": (8 * 3600.0, 4),
    "stage-T-smoke": (3 * 3600.0, 1), "stage-T-run": (8 * 3600.0, 4),
}
EXIT_REFUSED, EXIT_STOPPED, EXIT_UNCLEAN, EXIT_UNRESOLVED = 2, 3, 4, 5


class Refused(RuntimeError):
    pass


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_generation(manifest_path, manifest_sha256) -> dict:
    """This file, and every other member, at the bytes the pinned manifest lists."""
    try:
        raw = Path(manifest_path).read_bytes()
    except OSError as error:
        raise Refused(f"{manifest_path}: unreadable manifest: {error}") from None
    if hashlib.sha256(raw).hexdigest() != manifest_sha256:
        raise Refused(f"{manifest_path} is not the pinned manifest {manifest_sha256}")
    try:
        files = (json.loads(raw).get("new_generation") or {}).get("files_sha256") or {}
    except (ValueError, AttributeError) as error:
        raise Refused(f"{manifest_path}: malformed manifest: {error}") from None
    if files.get(SELF_PATH) != _IMPORTED_SOURCE_SHA256:
        raise Refused("this supervisor is not the manifest's generation")
    drifted = sorted(rel for rel, want in files.items()
                     if not (REPO / rel).is_file() or _sha256(REPO / rel) != want)
    if drifted:
        raise Refused(f"the tree is not the manifest's generation: {drifted[:6]}")
    return files


def boot_seconds() -> float:
    return time.clock_gettime(time.CLOCK_BOOTTIME)


_TICKS = os.sysconf("SC_CLK_TCK")


def _stat(pid) -> tuple[str, int, int, int] | None:
    """(state, ppid, session, start ticks) of one process, or None once it is gone."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as handle:
            raw = handle.read()
    except (FileNotFoundError, ProcessLookupError):
        return None
    if b")" not in raw:         # read while the task was being released
        return None
    fields = raw[raw.rindex(b")") + 2:].split()
    return fields[0].decode("ascii"), int(fields[1]), int(fields[3]), int(fields[19])


def managed_sessions(parent: int) -> dict:
    """{(pid, start ticks): start seconds} of the parent's child session leaders."""
    found = {}
    with os.scandir("/proc") as entries:
        names = [entry.name for entry in entries if entry.name.isdigit()]
    for name in names:
        stat = _stat(name)
        if stat is not None and stat[1] == parent and stat[2] == int(name):
            found[(int(name), stat[3])] = stat[3] / _TICKS
    return found


def one_process(pid: int) -> dict:
    stat = _stat(pid)
    return {} if stat is None else {(pid, stat[3]): stat[3] / _TICKS}


def still_live(identity) -> bool:
    """The very process (PID and start time) is present and not a zombie."""
    stat = _stat(identity[0])
    return stat is not None and stat[3] == identity[1] and stat[0] not in "ZXx"


def held_leases(owner: int, lease_root: Path) -> set:
    """Names of the GPU lease files whose kernel flock was taken by ``owner``."""
    files = {}
    try:
        with os.scandir(lease_root) as entries:
            for entry in entries:
                if entry.name.endswith(".lock"):
                    st = entry.stat(follow_symlinks=False)
                    files[(os.major(st.st_dev), os.minor(st.st_dev), st.st_ino)] = entry.name
    except FileNotFoundError:
        return set()
    held = set()
    with open("/proc/locks", encoding="ascii") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 6 or parts[1] != "FLOCK" or "->" in parts:
                continue
            major, minor, inode = parts[5].split(":")
            key = (int(major, 16), int(minor, 16), int(inode))
            if key in files and int(parts[4]) == owner:
                held.add(files[key])
    return held


def free_bytes(path) -> int:
    probe = Path(path)
    while not probe.exists():
        probe = probe.parent
    usage = os.statvfs(probe)
    return usage.f_bavail * usage.f_frsize


class Ledger:
    """Append-only JSON lines, fsynced; one supervised run at a time."""

    def __init__(self, root: Path):
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / LEDGER_NAME
        self._lock = open(root / LOCK_NAME, "a")
        try:
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock.close()
            raise Refused(f"another supervisor holds {root / LOCK_NAME}") from None

    def prior_device_seconds(self) -> tuple[float, list]:
        runs, finals = [], {}
        if self.path.exists():
            for number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
                record = json.loads(line)
                if record.get("schema") != LEDGER_SCHEMA:
                    raise Refused(f"{self.path}:{number} is not a {LEDGER_SCHEMA} record")
                if record.get("event") == "start":
                    runs.append(record["run_id"])
                elif record.get("event") == "final":
                    finals[record["run_id"]] = record
        unfinished = [run for run in runs if run not in finals]
        if unfinished:
            raise Refused(f"run(s) {unfinished} in {self.path} have no final record: their device "
                          "time is unknown; the audit must reconcile the ledger first")
        unclean = [run for run, record in finals.items() if record.get("status") == "unclean"]
        if unclean:
            raise Refused(f"run(s) {unclean} in {self.path} ended with live attempts or held "
                          "leases: their device time is not final; the audit must reconcile it")
        unresolved = [run for run, record in finals.items()
                      if record.get("status") == "unresolved"]
        if unresolved:
            raise Refused(f"run(s) {unresolved} in {self.path} lost observation continuity: "
                          "their charge is unresolved; the audit must reconcile it")
        return sum(float(record["charged_seconds"]) for record in finals.values()), runs

    def write(self, record: dict) -> None:
        record = {"schema": LEDGER_SCHEMA,
                  "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), **record}
        line = json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
        descriptor = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(descriptor, line.encode("utf-8"))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def close(self) -> None:
        self._lock.close()


def supervise(command, *, stage, label, gpus, attempts, planned_cells, watch_path, ops_root,
              manifest_sha256, budget_seconds=BUDGET_DEVICE_SECONDS, poll_seconds=POLL_SECONDS,
              watchdog_seconds=WATCHDOG_SECONDS, stop_bound_seconds=STOP_BOUND_SECONDS,
              wall_limit_seconds=None, ledger_every=LEDGER_EVERY_SECONDS,
              lease_root=GPU_LEASE_ROOT, free_bytes=free_bytes) -> int:
    """Run ``command`` under the storage, budget and wall-time rules; returns an exit code."""
    wall_limit = STAGES[stage][0] if wall_limit_seconds is None else float(wall_limit_seconds)
    if not watchdog_seconds >= 2 * poll_seconds > 0:
        raise Refused("the watchdog bound must be at least two poll intervals")
    ledger = Ledger(Path(ops_root))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + secrets.token_hex(4)
    identity = {"run_id": run_id, "stage": stage, "label": label}
    try:
        prior, _runs = ledger.prior_device_seconds()
        # Work may outrun the last completed observation by the watchdog bound,
        # and the launcher's own cleanup by the stop bound, on every GPU.
        headroom = gpus * (watchdog_seconds + stop_bound_seconds)
        least_allowance = (planned_cells * poll_seconds) if attempts == "sessions" else 0.0
        free = free_bytes(watch_path)
        need = FREE_FLOOR_BYTES + planned_cells * CELL_OUTPUT_BYTES
        refusal = None
        if free < need:
            refusal = (f"space: {free} bytes free under {watch_path}, below {need} "
                       f"(floor + {planned_cells} planned cells)")
        elif prior + least_allowance + headroom >= budget_seconds:
            refusal = (f"budget: {prior:.0f} s already charged + {least_allowance + headroom:.0f} "
                       f"s allowance and headroom reach the {budget_seconds:.0f} s budget")
        ledger.write({"event": "start", **identity, "command": list(command),
                      "command_sha256": hashlib.sha256(json.dumps(list(command)).encode()).hexdigest(),
                      "manifest_sha256": manifest_sha256, "supervisor_sha256": _IMPORTED_SOURCE_SHA256,
                      "prior_charged_seconds": prior, "free_bytes": free, "refused": refusal,
                      "rules": {"free_floor_bytes": FREE_FLOOR_BYTES,
                                "cell_output_bytes": CELL_OUTPUT_BYTES,
                                "budget_seconds": budget_seconds, "poll_seconds": poll_seconds,
                                "watchdog_seconds": watchdog_seconds,
                                "stop_bound_seconds": stop_bound_seconds,
                                "headroom_seconds": headroom,
                                "wall_limit_seconds": wall_limit, "gpus": gpus,
                                "planned_cells": planned_cells, "attempts": attempts}})
        if refusal:
            ledger.write({"event": "final", **identity, "status": "refused-before-start",
                          "reason": refusal, "charged_seconds": 0.0, "device_seconds": 0.0})
            print(f"[supervisor] REFUSED before start: {refusal}", file=sys.stderr)
            return EXIT_REFUSED
        return _run(command, ledger, identity, prior=prior, headroom=headroom, gpus=gpus,
                    attempts=attempts, planned_cells=planned_cells, watch_path=watch_path,
                    budget_seconds=budget_seconds, poll_seconds=poll_seconds,
                    watchdog_seconds=watchdog_seconds, stop_bound_seconds=stop_bound_seconds,
                    wall_limit=wall_limit, ledger_every=ledger_every,
                    lease_root=Path(lease_root), free_bytes=free_bytes, ops_root=Path(ops_root))
    finally:
        ledger.close()


def _run(command, ledger, identity, *, prior, headroom, gpus, attempts, planned_cells,
         watch_path, budget_seconds, poll_seconds, watchdog_seconds, stop_bound_seconds,
         wall_limit, ledger_every, lease_root, free_bytes, ops_root) -> int:
    operator, failures, continuity = [], [], []

    def _operator_stop(signum, _frame):
        operator.append(signal.Signals(signum).name)

    def _write(record) -> None:
        # A ledger that cannot be written is a failed observation: it stops
        # dispatch, voids the settlement, and never ends the supervision of a live command.
        try:
            ledger.write(record)
        except Exception as error:                  # noqa: BLE001
            failures.append(f"ledger write failed: {error!r}")
            print(f"[supervisor] {failures[-1]} ({record.get('event')})", file=sys.stderr)

    previous = {signum: signal.signal(signum, _operator_stop)
                for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    log_path = ops_root / f"{identity['run_id']}.command.log"
    try:
        with open(log_path, "ab") as log:
            started = boot_seconds()
            # Its own session: terminal signals reach only this supervisor, which
            # turns them into ONE SIGTERM to the launcher's supervised path.
            proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
    except OSError as error:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        _write({"event": "final", **identity, "status": "failed-to-start", "reason": repr(error),
                "charged_seconds": 0.0, "device_seconds": 0.0})
        print(f"[supervisor] REFUSED: cannot start {command[:2]}: {error}", file=sys.stderr)
        return EXIT_REFUSED
    # The watchdog does not depend on the observation work: a stat, scan or ledger
    # write that does not return cannot keep the command running past the bound.
    # Reaping and signalling the command share one lock, so a signal never meets
    # a reused PID.
    proc_lock, halt = threading.Lock(), threading.Event()
    watch = {"last": started, "fired": None, "stop_sent": None}

    def _watchdog() -> None:
        while not halt.wait(min(0.25, watchdog_seconds / 4)):
            quiet = boot_seconds() - watch["last"]
            if quiet > watchdog_seconds and watch["fired"] is None:
                watch["fired"] = (f"watchdog: no completed observation for {quiet:.1f} s "
                                  f"(bound {watchdog_seconds:.1f} s)")
                with proc_lock:
                    if proc.returncode is None and watch["stop_sent"] is None:
                        proc.send_signal(signal.SIGTERM)
                        watch["stop_sent"] = boot_seconds()

    watcher = threading.Thread(target=_watchdog, name="supervisor-watchdog", daemon=True)
    watcher.start()
    live, done = {}, []
    lease_gpu_seconds, first_lease, leased, last = 0.0, None, 0, started
    stop = overdue_noted = None
    window_start, max_window, allowance, exit_seen = started, 0.0, 0.0, None
    last_written = started
    try:
        while True:
            scan_start = boot_seconds()
            reason = device = free = projected = None
            try:
                seen = (managed_sessions(proc.pid) if attempts == "sessions"
                        else one_process(proc.pid))
                now = boot_seconds()
                # An attempt missed by this scan and the previous one lived inside
                # (previous scan start, this scan end).
                window, window_start = now - window_start, scan_start
                max_window = max(max_window, window)
                if window > watchdog_seconds:
                    continuity.append(f"observation window {window:.1f} s exceeds the "
                                      f"{watchdog_seconds:.1f} s bound")
                ended_ids = {tuple(a["id"]) for a in done}
                for key, start in seen.items():
                    if key not in live and key not in ended_ids:
                        live[key] = start
                        _write({"event": "attempt-start", **identity, "id": list(key),
                                "start_boot": start})
                for key in [k for k in live if k not in seen]:
                    start = live.pop(key)
                    done.append({"id": list(key), "start_boot": start, "end_boot": now,
                                 "seconds": now - start})
                    _write({"event": "attempt-end", **identity, **done[-1]})
                if attempts == "sessions":
                    allowance = planned_cells * max_window
                    if len(done) + len(live) > planned_cells and not any(
                            c.startswith("attempts:") for c in continuity):
                        continuity.append(f"attempts: {len(done) + len(live)} observed for "
                                          f"{planned_cells} planned cells")
                device = sum(a["seconds"] for a in done) + sum(now - s for s in live.values())
                count = len(held_leases(proc.pid, lease_root))
                lease_gpu_seconds += max(count, leased) * (now - last)
                if count and first_lease is None:
                    first_lease = last              # conservative: the previous poll
                leased, last = count, now
                free = free_bytes(watch_path)
                projected = prior + device + allowance + headroom
                if operator:
                    reason = f"operator signal {operator[0]}"
                elif watch["fired"]:
                    reason = watch["fired"]
                elif continuity:
                    reason = f"continuity lost: {continuity[0]}"
                elif failures:
                    reason = f"monitor failure: {failures[0]}"
                elif free < FREE_FLOOR_BYTES + gpus * CELL_OUTPUT_BYTES:
                    reason = (f"space: {free} bytes free, below the floor plus {gpus} in-flight "
                              "cell outputs")
                elif projected >= budget_seconds:
                    reason = f"budget: projected {projected:.0f} s of {budget_seconds:.0f} s"
                elif now - started >= wall_limit - stop_bound_seconds - watchdog_seconds:
                    reason = f"wall: {now - started:.0f} s of the {wall_limit:.0f} s limit"
            except Exception as error:              # noqa: BLE001 -- never a silent pass
                now = boot_seconds()
                continuity.append(f"observation failed: {error!r}")
                reason = f"monitor failure: {error!r}"
            watch["last"] = boot_seconds()
            if watch["fired"] and not any(c.startswith("watchdog") for c in continuity):
                continuity.append(watch["fired"])
            if reason and stop is None:
                stop = reason
                with proc_lock:
                    if proc.returncode is None and watch["stop_sent"] is None:
                        # Exact: the launcher is this process's own unreaped child.
                        proc.send_signal(signal.SIGTERM)
                        watch["stop_sent"] = now
                _write({"event": "stop", **identity, "reason": reason, "device_seconds": device,
                        "free_bytes": free, "projected_seconds": projected,
                        "elapsed": now - started})
                print(f"[supervisor] STOP: {reason}; SIGTERM sent to {proc.pid}", file=sys.stderr)
            with proc_lock:
                exited = proc.poll() is not None
            if exited:
                exit_seen = boot_seconds()
                break
            if watch["stop_sent"] is not None and overdue_noted is None \
                    and now - watch["stop_sent"] > stop_bound_seconds:
                overdue_noted = now
                _write({"event": "stop-overdue", **identity, "pid": proc.pid,
                        "note": "still live after the stop bound; still waiting -- a timeout is "
                                "not evidence that it exited"})
            if now - last_written >= ledger_every:
                last_written = now
                _write({"event": "poll", **identity, "device_seconds": device,
                        "in_flight": len(live), "leased_gpus": leased, "free_bytes": free,
                        "projected_seconds": projected, "max_window_seconds": max_window,
                        "elapsed": now - started})
            time.sleep(poll_seconds)
    finally:
        halt.set()
        watcher.join()
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    ended = boot_seconds()
    # The last window: an attempt begun after the last scan ended before the exit was seen.
    window = (ended if exit_seen is None else exit_seen) - window_start
    max_window = max(max_window, window)
    if window > watchdog_seconds:
        continuity.append(f"observation window {window:.1f} s exceeds the "
                          f"{watchdog_seconds:.1f} s bound")
    if attempts == "sessions":
        allowance = planned_cells * max_window
    if watch["fired"] and not any(c.startswith("watchdog") for c in continuity):
        continuity.append(watch["fired"])
    for key, start in list(live.items()):
        live.pop(key)
        done.append({"id": list(key), "start_boot": start, "end_boot": ended,
                     "seconds": ended - start})
    lease_gpu_seconds += leased * (ended - last)
    orphans = [a["id"] for a in done if still_live(tuple(a["id"]))]
    leases_after = sorted(held_leases(proc.pid, lease_root))
    device = sum(a["seconds"] for a in done)
    status = ("unclean" if orphans or leases_after
              else "unresolved" if continuity or failures
              else "stopped" if stop else "exited")
    failures_before_final = len(failures)
    _write({
        "event": "final", **identity, "status": status, "reason": stop,
        "returncode": proc.returncode, "attempts": done, "device_seconds": device,
        "max_observation_window_seconds": max_window,
        "unobserved_allowance_seconds": allowance, "charged_seconds": device + allowance,
        "cumulative_charged_seconds": prior + device + allowance,
        "pessimistic_bound_seconds": gpus * (ended - started),
        "continuity_lost": continuity, "lease_gpu_seconds": lease_gpu_seconds,
        "pre_lease_seconds": None if first_lease is None else first_lease - started,
        "wall_seconds": ended - started, "orphaned_live_attempts": orphans,
        "leases_held_after_exit": leases_after, "monitor_failures": failures,
        "command_log": str(log_path),
        "accounting": "attempt process lifetimes (end at the first scan without them) plus "
                      "planned cells x the longest observation window: an upper bound on device "
                      "occupancy while continuity holds, not measured GPU utilisation"})
    print(f"[supervisor] {status}: rc={proc.returncode} device={device / 3600:.3f} h "
          f"allowance={allowance:.1f} s lease={lease_gpu_seconds / 3600:.3f} GPU-h log={log_path}",
          file=sys.stderr)
    if orphans or leases_after or len(failures) > failures_before_final:
        return EXIT_UNCLEAN
    if continuity or failures:
        return EXIT_UNRESOLVED
    if stop:
        return EXIT_STOPPED
    return proc.returncode


def stage_ledger(stage: str, ops_root) -> tuple:
    """(ledger root, device budget) of a stage. Stage L keeps its own ledger root and budget
    (contract L v1 section 8), stages R and T theirs (contract R/T v1 section 9); no stage writes
    another group's root. A crossed root refuses."""
    groups = ((L_STAGES, L_OPS_ROOT, L_BUDGET_DEVICE_SECONDS),
              (RT_STAGES, RT_OPS_ROOT, RT_BUDGET_DEVICE_SECONDS))
    own_root, budget = DEFAULT_OPS_ROOT, BUDGET_DEVICE_SECONDS
    for stages, root_of, budget_of in groups:
        if stage in stages:
            own_root, budget = root_of, budget_of
    root = Path(os.path.abspath(ops_root if ops_root is not None else own_root))
    if root != own_root and (own_root != DEFAULT_OPS_ROOT
                             or root in {r for _, r, _ in groups}):
        raise Refused(f"{stage} keeps its own ledger under {own_root}, not {root}")
    return root, budget


def command_gpus(stage: str, command) -> int:
    """The GPU count the supervised command itself names (a probe uses one)."""
    most = STAGES[stage][1]
    if stage == "probe":
        return most
    given = [command[i + 1] for i, part in enumerate(command[:-1]) if part == "--gpus"]
    if len(given) != 1:
        raise Refused(f"{stage} supervises a launcher that names its GPUs once with --gpus")
    count = len([gpu for gpu in given[0].split(",") if gpu])
    if not 1 <= count <= most:
        raise Refused(f"{stage} runs on 1 to {most} GPUs, not {count}")
    return count


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--stage", required=True, choices=sorted(STAGES))
    parser.add_argument("--label", default="")
    parser.add_argument("--planned-cells", type=int, required=True)
    parser.add_argument("--watch-path", required=True)
    parser.add_argument("--ops-root", default=None,
                        help=f"default {DEFAULT_OPS_ROOT}; stage L: {L_OPS_ROOT}; stages R/T: "
                             f"{RT_OPS_ROOT} (their own ledgers)")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    try:
        if not command:
            raise Refused("no command to supervise")
        if args.planned_cells < 0:
            raise Refused("--planned-cells must be >= 0")
        verify_generation(args.manifest, args.manifest_sha256)
        ops_root, budget = stage_ledger(args.stage, args.ops_root)
        return supervise(command, stage=args.stage, label=args.label,
                         gpus=command_gpus(args.stage, command),
                         attempts="self" if args.stage == "probe" else "sessions",
                         planned_cells=args.planned_cells, watch_path=args.watch_path,
                         ops_root=ops_root, manifest_sha256=args.manifest_sha256,
                         budget_seconds=budget)
    except Refused as error:
        print(f"[supervisor] REFUSED: {error}", file=sys.stderr)
        return EXIT_REFUSED


if __name__ == "__main__":
    raise SystemExit(main())
