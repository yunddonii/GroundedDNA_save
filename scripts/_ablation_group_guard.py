"""Own and reap one complete ablation wrapper process group.

The executor may itself receive SIGKILL, in which case Python ``finally``
cannot run.  Linux parent-death signaling therefore targets this small direct
child.  Unlike a Bash wrapper waiting on a foreground trainer, this guardian
handles TERM immediately and signals every other member of its process group,
then kills any survivor before releasing inherited campaign/GPU lease FDs.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


class _Stop(RuntimeError):
    def __init__(self, signum: int):
        self.signum = int(signum)


_STOP_SIGNAL: int | None = None


def _handle(signum, _frame) -> None:
    global _STOP_SIGNAL
    if _STOP_SIGNAL is None:
        _STOP_SIGNAL = int(signum)
        for watched in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(watched, signal.SIG_IGN)
        raise _Stop(signum)


def _group_members(group: int) -> set[int]:
    members: set[int] = set()
    for item in Path("/proc").iterdir():
        if not item.name.isdigit():
            continue
        try:
            raw = (item / "stat").read_text(encoding="utf-8")
            tail = raw[raw.rfind(")") + 2:].split()
            # tail: state, ppid, pgrp, ...
            if len(tail) >= 3 and int(tail[2]) == group \
                    and tail[0] != "Z":
                members.add(int(item.name))
        except (OSError, ValueError):
            continue
    return members


def _terminate_group(child: subprocess.Popen | None, *, grace: float = 2.0) -> None:
    own_pid, group = os.getpid(), os.getpgrp()
    if group != own_pid:
        raise RuntimeError(
            f"ablation guardian is not its process-group leader: {group}")
    # Group-directed TERM includes the guardian itself.  It must stay alive to
    # enforce the grace deadline and reap the direct wrapper.
    for watched in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(watched, signal.SIG_IGN)
    others = _group_members(group) - {own_pid}
    if others:
        try:
            os.killpg(group, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        others = _group_members(group) - {own_pid}
        if not others:
            break
        time.sleep(0.02)
    # Kill concrete PIDs rather than the group: the guardian must remain alive
    # long enough to reap its direct child and return a trustworthy status.
    for pid in _group_members(group) - {own_pid}:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if child is not None:
        try:
            child.wait(timeout=2)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                f"guarded wrapper {child.pid} survived SIGKILL") from error
    deadline = time.monotonic() + 2.0
    while _group_members(group) - {own_pid} and time.monotonic() < deadline:
        time.sleep(0.02)
    survivors = _group_members(group) - {own_pid}
    if survivors:
        raise RuntimeError(f"guarded process group still has members {survivors}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pass-fd", action="append", type=int, default=[])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        parser.error("a guarded command is required")
    for descriptor in args.pass_fd:
        os.fstat(descriptor)
    previous = {sig: signal.getsignal(sig)
                for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    for watched in previous:
        signal.signal(watched, _handle)
    child: subprocess.Popen | None = None
    returncode = 1
    try:
        child = subprocess.Popen(
            command, pass_fds=tuple(sorted(set(args.pass_fd))))
        returncode = child.wait()
    except _Stop as stopped:
        returncode = 128 + stopped.signum
    finally:
        _terminate_group(child)
        for watched, handler in previous.items():
            signal.signal(watched, handler)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
