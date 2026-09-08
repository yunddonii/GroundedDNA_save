"""Signal-safe launcher for the exact Phase-5 ablation campaign.

The executable shell entrypoint only sanitizes process startup and immediately
execs this supervisor.  Python owns every child PID, the tag-set flock, and the
bounded TERM/KILL lifecycle.  Only cell executors inherit the flock descriptor;
ledger helpers, GPU probes, and dry-run helpers are spawned with ``close_fds``.
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import stat
import subprocess
import sys
import time
from typing import IO, Sequence


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts._ablation_exec import load_plan  # noqa: E402
from scripts.campaign_ledger import campaign_lock_key  # noqa: E402


PYTHON = Path("/home/yschoi/.conda/envs/dna_hashing/bin/python").resolve()
EXECUTOR = REPO / "scripts" / "_ablation_exec.py"
LEDGER_TOOL = REPO / "scripts" / "campaign_ledger.py"
LOCK_ROOT = Path(f"/tmp/groundeddna-ablation-campaign-leases-{os.getuid()}")
TRUSTED_PATH = (
    "/usr/local/cuda-12.4/bin:/usr/local/sbin:/usr/local/bin:"
    "/usr/sbin:/usr/bin:/sbin:/bin")
TRUSTED_LD_LIBRARY_PATH = (
    "/usr/local/cuda-12.4/lib64:/usr/local/cuda/extras/CUPTI")
MONITORED_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)


class LaunchRefused(RuntimeError):
    """The campaign cannot safely start or finish."""


def _number(name: str, default: str, *, integer: bool,
            minimum: float, maximum: float) -> int | float:
    raw = os.environ.get(name, default)
    try:
        value = int(raw) if integer else float(raw)
    except (TypeError, ValueError):
        raise LaunchRefused(
            f"{name} must be a finite number in [{minimum}, {maximum}]") \
            from None
    if isinstance(value, float) and (value != value or value in (float("inf"),
                                                                 float("-inf"))):
        raise LaunchRefused(f"{name} must be finite")
    if value < minimum or value > maximum:
        raise LaunchRefused(
            f"{name}={value} is outside [{minimum}, {maximum}]")
    return value


def _child_environment() -> dict[str, str]:
    env = dict(os.environ)
    for name in (
            "BASH_ENV", "ENV", "PYTHONPATH", "PYTHONHOME", "LD_PRELOAD",
            "CDPATH", "GLOBIGNORE"):
        env.pop(name, None)
    env.update({
        "PATH": TRUSTED_PATH,
        "LD_LIBRARY_PATH": TRUSTED_LD_LIBRARY_PATH,
        "PYTHONNOUSERSITE": "1",
        "PY": str(PYTHON),
    })
    return env


class ProcessSupervisor:
    """Own all subprocesses and finish cleanup within fixed deadlines."""

    def __init__(self, *, grace_seconds: float) -> None:
        self.grace_seconds = grace_seconds
        self.active: dict[int, subprocess.Popen] = {}
        self._cleaning = False
        self._old_handlers: dict[signal.Signals, object] = {}

    def install_handlers(self) -> None:
        for signum in MONITORED_SIGNALS:
            self._old_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, self._on_signal)

    def restore_handlers(self) -> None:
        for signum, handler in self._old_handlers.items():
            signal.signal(signum, handler)
        self._old_handlers.clear()

    def _on_signal(self, signum: int, _frame: object) -> None:
        # Ignore a burst while the first cleanup owns the state transition.
        for monitored in MONITORED_SIGNALS:
            signal.signal(monitored, signal.SIG_IGN)
        print(
            f"[campaign] received {signal.Signals(signum).name}; "
            "terminating all managed processes", file=sys.stderr,
            flush=True)
        self.terminate_all()
        raise SystemExit(128 + signum)

    def spawn(self, argv: Sequence[str], *,
              stdout: int | IO[str] | None = None,
              stderr: int | IO[str] | None = None,
              pass_fds: Sequence[int] = ()) -> subprocess.Popen:
        """Spawn without the signal race and register before unmasking."""
        old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, MONITORED_SIGNALS)
        expected_parent = os.getpid()

        def restore_child_mask() -> None:
            signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
            # Helpers must not outlive a SIGKILLed supervisor and publish a
            # late record/receipt.  Executors replace this with their own TERM
            # parent-death contract after exec; their guardian then owns group
            # cleanup.  The before/after parent check closes the pre-arm race.
            if os.getppid() != expected_parent:
                os._exit(125)
            libc = ctypes.CDLL(None, use_errno=True)
            if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
                os._exit(126)
            if os.getppid() != expected_parent:
                os._exit(125)

        process: subprocess.Popen | None = None
        try:
            process = subprocess.Popen(
                list(argv), cwd=str(REPO), env=_child_environment(),
                stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                close_fds=True, pass_fds=tuple(pass_fds),
                preexec_fn=restore_child_mask, text=True)
            self.active[process.pid] = process
        finally:
            # A pending signal may raise SystemExit here.  The process is
            # already registered, so the handler still owns and terminates it.
            signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
        if process is None:  # pragma: no cover - Popen raised instead
            raise LaunchRefused(f"failed to spawn {argv[0]}")
        return process

    def communicate(self, argv: Sequence[str], *,
                    pass_fds: Sequence[int] = ()) -> tuple[int, str, str]:
        process = self.spawn(
            argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            pass_fds=pass_fds)
        try:
            stdout, stderr = process.communicate()
            return int(process.returncode), stdout, stderr
        finally:
            self.active.pop(process.pid, None)

    def wait(self, process: subprocess.Popen) -> int:
        try:
            return int(process.wait())
        finally:
            self.active.pop(process.pid, None)

    @staticmethod
    def _live(process: subprocess.Popen) -> bool:
        return process.poll() is None

    def terminate_all(self) -> None:
        if self._cleaning:
            return
        self._cleaning = True
        snapshot = list(self.active.values())
        for process in snapshot:
            if self._live(process):
                try:
                    process.send_signal(signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + self.grace_seconds
        while time.monotonic() < deadline \
                and any(self._live(process) for process in snapshot):
            time.sleep(0.02)
        for process in snapshot:
            if self._live(process):
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
        # Reap only processes known ready.  Waiting unconditionally after KILL
        # can hang forever on an uninterruptible D-state task.  Cell guardians
        # also carry parent-death cleanup, so supervisor exit remains bounded.
        reap_deadline = time.monotonic() + 2.0
        while time.monotonic() < reap_deadline:
            pending = [process for process in snapshot if self._live(process)]
            if not pending:
                break
            time.sleep(0.02)
        for process in snapshot:
            if process.poll() is not None:
                try:
                    process.wait(timeout=0)
                except (ChildProcessError, subprocess.TimeoutExpired):
                    pass
                self.active.pop(process.pid, None)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_lock(plan: Path, plan_sha: str) -> tuple[int, str]:
    key = campaign_lock_key(plan, expected_sha256=plan_sha)
    try:
        LOCK_ROOT.mkdir(mode=0o700, exist_ok=True)
        root_info = LOCK_ROOT.lstat()
    except OSError as error:
        raise LaunchRefused(f"unsafe campaign lease root {LOCK_ROOT}: {error}") \
            from None
    if LOCK_ROOT.is_symlink() or not stat.S_ISDIR(root_info.st_mode) \
            or root_info.st_uid != os.getuid() or root_info.st_mode & 0o077:
        raise LaunchRefused(f"unsafe campaign lease root {LOCK_ROOT}")
    path = LOCK_ROOT / f"{key}.lock"
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
        fd_info = os.fstat(descriptor)
        path_info = path.lstat()
    except OSError as error:
        raise LaunchRefused(f"unsafe campaign lock leaf {path}: {error}") \
            from None
    if path.is_symlink() or not stat.S_ISREG(fd_info.st_mode) \
            or fd_info.st_uid != os.getuid() \
            or fd_info.st_nlink != 1 or fd_info.st_mode & 0o077 \
            or (fd_info.st_dev, fd_info.st_ino) != (
                path_info.st_dev, path_info.st_ino):
        os.close(descriptor)
        raise LaunchRefused(f"unsafe campaign lock leaf {path}")
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(descriptor)
        raise LaunchRefused(
            "same 24-cell result set already has a live campaign owner") \
            from None
    return descriptor, key


def _write_lock_metadata(descriptor: int, *, plan_sha: str,
                         token: str, ledger: Path) -> None:
    payload = {
        "schema": "groundeddna.phase5-tag-set-lock", "schema_version": 1,
        "pid": os.getpid(), "plan_file_sha256": plan_sha,
        "campaign_token": token, "ledger": str(ledger),
        "started_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
    }
    raw = (json.dumps(payload, sort_keys=True) + "\n").encode()
    os.lseek(descriptor, 0, os.SEEK_SET)
    os.ftruncate(descriptor, 0)
    offset = 0
    while offset < len(raw):
        offset += os.write(descriptor, raw[offset:])
    os.fsync(descriptor)


def _helper(supervisor: ProcessSupervisor, argv: Sequence[str]) \
        -> tuple[int, str, str]:
    # Deliberately no pass_fds: ledger/hash/probe helpers must never keep the
    # tag-set flock alive if the launcher is killed.
    return supervisor.communicate(argv)


def _candidate_gpus(supervisor: ProcessSupervisor, *, idle_mib: int,
                    explicit: tuple[int, ...] | None) -> list[int]:
    if explicit is not None:
        return list(explicit)
    rc, stdout, stderr = _helper(supervisor, [
        "/usr/bin/nvidia-smi", "--query-gpu=index,memory.used",
        "--format=csv,noheader,nounits"])
    if rc != 0:
        raise LaunchRefused(
            "cannot query idle GPUs: " + (stderr.strip() or f"exit {rc}"))
    candidates = []
    for line in stdout.splitlines():
        fields = [part.strip() for part in line.split(",", 1)]
        if len(fields) != 2:
            raise LaunchRefused(f"malformed nvidia-smi row {line!r}")
        try:
            index, memory = int(fields[0]), int(fields[1])
        except ValueError:
            raise LaunchRefused(f"malformed nvidia-smi row {line!r}") \
                from None
        if index < 0:
            raise LaunchRefused(f"negative GPU index in {line!r}")
        if memory < idle_mib:
            candidates.append(index)
    return candidates


def _parse_explicit_gpus() -> tuple[int, ...] | None:
    raw = os.environ.get("GPUS")
    if raw is None:
        return None
    items = raw.split()
    if not items:
        raise LaunchRefused("GPUS is set but empty")
    try:
        values = tuple(int(item) for item in items)
    except ValueError:
        raise LaunchRefused("GPUS must be a whitespace-separated index list") \
            from None
    if any(value < 0 or str(value) not in items for value in values) \
            or len(values) != len(set(values)):
        raise LaunchRefused("GPUS must contain unique canonical nonnegative indices")
    return values


def _dry_run(supervisor: ProcessSupervisor, *, plan: Path, plan_sha: str,
             log_root: Path) -> int:
    data = load_plan(plan, expect_sha256=plan_sha)
    dry_root = log_root / f"phase5-dry-{plan_sha}"
    dry_root.mkdir(mode=0o700, parents=False, exist_ok=False)
    for index in range(len(data["cells"])):
        path = dry_root / f"cell_{index}.json"
        with path.open("x", encoding="utf-8") as output:
            process = supervisor.spawn([
                str(PYTHON), str(EXECUTOR), "--plan", str(plan),
                "--index", str(index), "--gpu", "0",
                "--expect-plan-sha256", plan_sha, "--dry-run"],
                stdout=output, stderr=subprocess.STDOUT)
            rc = supervisor.wait(process)
        if rc != 0:
            raise LaunchRefused(f"dry-run cell {index} exited {rc}")
    print("[campaign] DRY RUN only: no reservation, GPU lease, cell record, or receipt")
    return 0


def _record(supervisor: ProcessSupervisor, *, ledger: Path, token: str,
            tag: str, status: str, run_dir: str | None = None,
            outcome: Path | None = None, detail: str | None = None) -> bool:
    argv = [
        str(PYTHON), str(LEDGER_TOOL), "record", "--ledger", str(ledger),
        "--campaign-token", token, "--tag", tag, "--status", status]
    if run_dir is not None:
        argv += ["--run-dir", run_dir]
    if outcome is not None:
        argv += ["--outcome", str(outcome)]
    if detail is not None:
        argv += ["--detail", detail]
    rc, _stdout, stderr = _helper(supervisor, argv)
    if rc != 0 and stderr:
        print(stderr.rstrip(), file=sys.stderr)
    return rc == 0


def launch(args: argparse.Namespace, supervisor: ProcessSupervisor) -> int:
    plan = Path(args.plan).resolve()
    ledger_input = Path(args.ledger)
    if args.repo is not None and Path(args.repo).resolve() != REPO:
        raise LaunchRefused(f"--repo must be the launcher source root {REPO}")
    log_root = Path(os.environ.get("LOGDIR", str(REPO / "logs"))).resolve()
    log_root.mkdir(parents=True, exist_ok=True)
    plan_sha = _sha_file(plan)
    if args.dry_run:
        return _dry_run(
            supervisor, plan=plan, plan_sha=plan_sha, log_root=log_root)

    lock_fd, _lock_key = _safe_lock(plan, plan_sha)
    try:
        token = secrets.token_hex(32)
        if ledger_input.is_symlink():
            raise LaunchRefused("ledger path may not be a symlink")
        ledger = ledger_input.resolve()
        _write_lock_metadata(
            lock_fd, plan_sha=plan_sha, token=token, ledger=ledger)
        rc, _stdout, stderr = _helper(supervisor, [
            str(PYTHON), str(LEDGER_TOOL), "open", "--ledger", str(ledger),
            "--plan", str(plan), "--expect-plan-sha256", plan_sha,
            "--owner-pid", str(os.getpid()), "--campaign-token", token])
        if rc != 0:
            if stderr:
                print(stderr.rstrip(), file=sys.stderr)
            raise LaunchRefused("campaign not reserved; starting nothing")

        plan = (ledger / "campaign_plan.json").resolve(strict=True)
        outcome_dir = ledger / "outcomes"
        if outcome_dir.is_symlink():
            raise LaunchRefused("outcomes path may not be a symlink")
        outcome_dir.mkdir(mode=0o700, exist_ok=True)
        if outcome_dir.is_symlink() or outcome_dir.resolve() != outcome_dir:
            raise LaunchRefused("outcomes directory is not canonical")
        campaign_log = log_root / f"phase5-{token}"
        campaign_log.mkdir(mode=0o700, exist_ok=False)
        data = load_plan(plan, expect_sha256=plan_sha)
        cells = data["cells"]
        if len(cells) != 24:
            raise LaunchRefused("reserved plan is not exact-24")
        print(
            f"[campaign] 24 cells reserved, generation {token[:12]}",
            flush=True)

        stagger = float(args.stagger)
        gpu_wait = float(args.gpu_wait)
        explicit = args.explicit_gpus
        leased: dict[int, subprocess.Popen] = {}
        processes: list[subprocess.Popen] = []
        outcomes: list[Path] = []
        log_handles: list[IO[str]] = []
        try:
            for index, cell in enumerate(cells):
                while True:
                    leased = {gpu: process for gpu, process in leased.items()
                              if process.poll() is None}
                    candidates = _candidate_gpus(
                        supervisor, idle_mib=int(args.idle_mib),
                        explicit=explicit)
                    available = [gpu for gpu in candidates if gpu not in leased]
                    if available:
                        gpu = available[0]
                        break
                    time.sleep(gpu_wait)
                tag = str(cell["tag"])
                outcome = outcome_dir / f"{tag}.json"
                output = (campaign_log / f"cell_{index}.out").open(
                    "x", encoding="utf-8")
                log_handles.append(output)
                print(
                    f"[campaign]   {cell['cell']} {cell['exp']} -> "
                    f"physical GPU {gpu}", flush=True)
                process = supervisor.spawn([
                    str(PYTHON), str(EXECUTOR), "--plan", str(plan),
                    "--index", str(index), "--gpu", str(gpu),
                    "--expect-plan-sha256", plan_sha,
                    "--reservation", str(ledger / "campaign_reservation.json"),
                    "--campaign-token", token, "--outcome", str(outcome),
                    "--campaign-lock-fd", str(lock_fd),
                    "--launcher-pid", str(os.getpid())],
                    stdout=output, stderr=subprocess.STDOUT,
                    pass_fds=(lock_fd,))
                output.close()
                log_handles.pop()
                leased[gpu] = process
                processes.append(process)
                outcomes.append(outcome)
                if stagger:
                    time.sleep(stagger)
        finally:
            for handle in log_handles:
                handle.close()

        failed = False
        for index, (cell, process, outcome) in enumerate(
                zip(cells, processes, outcomes, strict=True)):
            rc = supervisor.wait(process)
            tag = str(cell["tag"])
            name = f"{cell['cell']}_{cell['exp']}"
            if rc == 0 and outcome.is_file() and not outcome.is_symlink():
                try:
                    payload = json.loads(outcome.read_text(encoding="utf-8"))
                    run_dir = str(payload["run_dir"])
                except (OSError, KeyError, TypeError, ValueError) as error:
                    run_dir = ""
                    print(f"[campaign] {tag}: invalid outcome JSON: {error}",
                          file=sys.stderr)
                if run_dir and _record(
                        supervisor, ledger=ledger, token=token, tag=tag,
                        status="ok", run_dir=run_dir, outcome=outcome):
                    print(f"[campaign]   {name} output-integrity ok -> {run_dir}")
                    continue
                print(
                    f"[campaign]   {name} exited 0 but strict output "
                    "validation refused")
                _record(
                    supervisor, ledger=ledger, token=token, tag=tag,
                    status="failed",
                    detail="rc=0 but output validation refused")
            else:
                print(f"[campaign]   {name} FAILED rc={rc}")
                _record(
                    supervisor, ledger=ledger, token=token, tag=tag,
                    status="failed",
                    detail=f"executor rc={rc} or missing exclusive outcome")
            failed = True

        if failed:
            raise LaunchRefused(
                "INCOMPLETE -- failed generation is immutable; use a fresh ledger")
        rc, _stdout, stderr = _helper(supervisor, [
            str(PYTHON), str(LEDGER_TOOL), "seal", "--ledger", str(ledger),
            "--campaign-token", token])
        if rc != 0:
            if stderr:
                print(stderr.rstrip(), file=sys.stderr)
            raise LaunchRefused("INCOMPLETE -- no completion receipt")
        print(
            "[campaign] done: 24 execution/analysis-integrity cells sealed; "
            "scientific authority remains false")
        return 0
    finally:
        os.close(lock_fd)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--repo")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    caller_python = os.environ.get("PY")
    if caller_python and Path(caller_python).resolve() != PYTHON:
        print(f"[campaign] PY must resolve to {PYTHON}", file=sys.stderr)
        return 2
    try:
        stagger = _number(
            "STAGGER", "150", integer=False, minimum=0, maximum=3600)
        gpu_wait = _number(
            "GPU_WAIT", "30", integer=False, minimum=0.02, maximum=3600)
        idle_mib = _number(
            "IDLE_MIB", "200", integer=True, minimum=0, maximum=10**9)
        grace = _number(
            "CAMPAIGN_TERM_GRACE", "10", integer=False,
            minimum=0.1, maximum=60)
        explicit = _parse_explicit_gpus()
        args = _parser().parse_args()
        args.stagger = stagger
        args.gpu_wait = gpu_wait
        args.idle_mib = idle_mib
        args.explicit_gpus = explicit
    except LaunchRefused as error:
        print(f"[campaign] {error}", file=sys.stderr)
        return 2

    supervisor = ProcessSupervisor(grace_seconds=float(grace))
    supervisor.install_handlers()
    try:
        return launch(args, supervisor)
    except SystemExit:
        raise
    except (LaunchRefused, OSError, RuntimeError, KeyError, TypeError) as error:
        supervisor.terminate_all()
        print(f"[campaign] FAILED: {error}", file=sys.stderr)
        return 1
    finally:
        supervisor.restore_handlers()


if __name__ == "__main__":
    raise SystemExit(main())
