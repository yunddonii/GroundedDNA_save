"""Execute one frozen Phase-5 ablation cell under a supervised GPU lease.

Production execution is accepted only from this file's repository, against the
ledger's immutable plan snapshot and campaign token. The wrapper runs in its
own process group; INT/TERM and Linux parent death terminate that whole group.
The host-global physical-GPU UUID lease is inherited by the wrapper, so even a
SIGKILL at the supervisor boundary cannot make an orphan look like a free GPU.
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
import signal
import stat
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SCHEMA_VERSION = 2
CANONICAL_RUNNER = "scripts/prompt_ablation_A_cell_fixedN.sh"
CANONICAL_BASH = "/usr/bin/bash"
TRUSTED_CHILD_PATH = (
    "/usr/local/cuda-12.4/bin:/usr/local/sbin:/usr/local/bin:"
    "/usr/sbin:/usr/bin:/sbin:/bin")
TRUSTED_LD_LIBRARY_PATH = (
    "/usr/local/cuda-12.4/lib64:/usr/local/cuda/extras/CUPTI")
_ACTIVE_CHILD: subprocess.Popen | None = None


class PlanRejected(RuntimeError):
    """The plan or its runtime binding cannot be trusted."""


def _digest_of(plan: dict) -> str:
    body = {key: value for key, value in plan.items() if key != "plan_digest"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_plan(path: Path, *, expect_sha256: str | None) -> dict:
    raw = path.read_bytes()
    if expect_sha256 is not None:
        actual = hashlib.sha256(raw).hexdigest()
        if actual != expect_sha256:
            raise PlanRejected(
                f"{path} hashes to {actual[:16]}..., but this campaign is "
                f"bound to {expect_sha256[:16]}...")
    try:
        plan = json.loads(
            raw.decode("utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value}")))
    except (UnicodeDecodeError, ValueError) as error:
        raise PlanRejected(f"{path}: {error}") from None
    if not isinstance(plan, dict):
        raise PlanRejected(f"{path} is not an object")
    if plan.get("schema_version") != SCHEMA_VERSION:
        raise PlanRejected(
            f"{path}: schema_version {plan.get('schema_version')!r}, not "
            f"{SCHEMA_VERSION}")
    stored = plan.get("plan_digest")
    if not isinstance(stored, str) or stored != _digest_of(plan):
        raise PlanRejected(f"{path}: plan_digest does not match its contents")

    cells = plan.get("cells")
    expected = plan.get("expected_cells")
    if not isinstance(cells, list) or not isinstance(expected, int) \
            or isinstance(expected, bool) or len(cells) != expected:
        raise PlanRejected(f"{path}: cell list/count is invalid")
    for key in ("env_passthrough", "env_pinned", "runners"):
        if not isinstance(plan.get(key), list):
            raise PlanRejected(f"{path}: {key} is not a list")
    if any(not isinstance(cell, dict) for cell in cells):
        raise PlanRejected(f"{path}: a cell is not an object")
    tags = [cell.get("tag") for cell in cells]
    if any(not isinstance(tag, str) or not tag for tag in tags):
        raise PlanRejected(f"{path}: a cell has no non-empty tag")
    if len(set(tags)) != len(tags):
        duplicated = sorted({tag for tag in tags if tags.count(tag) > 1})
        raise PlanRejected(f"{path}: duplicate result tags {duplicated}")
    return plan


def cell_at(plan: dict, index: int) -> dict:
    cells = plan["cells"]
    if index < 0 or index >= len(cells):
        raise PlanRejected(f"index {index} is not in 0..{len(cells) - 1}")
    return cells[index]


def child_env(plan: dict, cell: dict, *, production: bool = False) -> dict:
    allowed = set(plan["env_passthrough"])
    pinned = set(plan["env_pinned"])
    if production:
        from scripts.ablation_campaign_plan import (
            ENV_PASSTHROUGH, _PINNED, RUNNERS)
        if allowed != set(ENV_PASSTHROUGH) \
                or pinned != set(_PINNED) \
                or plan.get("runners") != sorted(RUNNERS):
            raise PlanRejected(
                "production plan environment/runner schema is not canonical")
    values = plan.get("env_passthrough_values")
    if not isinstance(values, dict):
        raise PlanRejected("plan records no env_passthrough_values")
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise PlanRejected(
            f"env_passthrough_values contains unapproved names {unknown}")
    cell_values = cell.get("env")
    if not isinstance(cell_values, dict):
        raise PlanRejected(f"{cell.get('tag')!r}: no pinned environment")
    stray = sorted(set(cell_values) - pinned)
    if stray:
        raise PlanRejected(
            f"{cell.get('tag')!r}: unapproved pinned variables {stray}")
    if production and set(cell_values) != pinned:
        raise PlanRejected(
            f"{cell.get('tag')!r}: pinned environment is incomplete")
    env = {str(key): str(value) for key, value in values.items()}
    env.update({str(key): str(value) for key, value in cell_values.items()})
    # Python module shadowing is source drift, not a supported campaign input.
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    if production:
        python = env.get("PY")
        if not python or Path(python).resolve() != Path(sys.executable).resolve():
            raise PlanRejected(
                "the wrapper's PY must be the exact interpreter executing "
                "the sealed campaign cell")
        # The plan records the caller's environment for provenance, but PATH
        # is not an executable authority.  A self-digested plan previously
        # placed a fake ``bash`` first and ran it while retaining the canonical
        # runner string.  Every production shell starts by absolute path and
        # child utility lookup is restricted to system locations.
        env["PATH"] = TRUSTED_CHILD_PATH
        env["LD_LIBRARY_PATH"] = TRUSTED_LD_LIBRARY_PATH
    return env


def child_argv(plan: dict, cell: dict, gpu_selector: str, *,
               production: bool = False) -> list[str]:
    runner = cell.get("runner")
    if runner not in set(plan.get("runners") or ()):
        raise PlanRejected(
            f"{cell.get('tag')!r}: runner {runner!r} is not allow-listed")
    if production and runner != CANONICAL_RUNNER:
        raise PlanRejected(
            f"production runner must be {CANONICAL_RUNNER!r}, got {runner!r}")
    if not isinstance(runner, str) or not runner:
        raise PlanRejected(f"{cell.get('tag')!r}: invalid runner")
    path = Path(runner)
    if production and (path.is_absolute() or ".." in path.parts
                       or (REPO / path).resolve() != REPO / path):
        raise PlanRejected(f"runner escapes the sealed repository: {runner!r}")
    exp = cell.get("exp")
    if not isinstance(exp, str) or not exp:
        raise PlanRejected(f"{cell.get('tag')!r}: no experiment name")
    shell = CANONICAL_BASH if production else "bash"
    return [shell, "-o", "pipefail", runner, str(gpu_selector), exp]


def _arm_parent_death_signal(expected_parent: int | None = None) -> None:
    """Ask Linux to send TERM if one *specific* parent dies.

    Capturing ``getppid()`` here is insufficient: if the launcher died before
    this process entered the function, PID 1 was captured and the orphan ran.
    Production therefore passes the reservation owner's PID across exec and
    checks it both before and after ``prctl``.
    """
    if not sys.platform.startswith("linux"):
        raise PlanRejected("production process supervision requires Linux")
    parent = os.getppid()
    wanted = parent if expected_parent is None else int(expected_parent)
    if wanted <= 1 or parent != wanted:
        raise PlanRejected(
            f"executor parent is {parent}, expected live launcher {wanted}")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        error = ctypes.get_errno()
        raise PlanRejected(f"cannot arm parent-death signal: errno {error}")
    if os.getppid() != wanted:
        os.kill(os.getpid(), signal.SIGTERM)


def _gpu_assignment(index_text: str, *,
                    executable: str = "/usr/bin/nvidia-smi") -> dict:
    try:
        index = int(index_text)
    except ValueError:
        raise PlanRejected(f"GPU index is not an integer: {index_text!r}") \
            from None
    if index < 0 or str(index) != str(index_text):
        raise PlanRejected(f"GPU index is not canonical: {index_text!r}")
    process = subprocess.run(
        [executable,
         "--query-gpu=index,uuid,name,pci.bus_id,driver_version",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False)
    if process.returncode != 0:
        raise PlanRejected(
            "cannot query physical GPU identity: "
            + (process.stderr.strip() or f"exit {process.returncode}"))
    matches = []
    for line in process.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", 4)]
        if len(fields) != 5:
            raise PlanRejected(f"malformed nvidia-smi row {line!r}")
        try:
            row_index = int(fields[0])
        except ValueError:
            raise PlanRejected(f"malformed nvidia-smi index {fields[0]!r}") \
                from None
        if row_index == index:
            matches.append({
                "index": row_index, "uuid": fields[1], "name": fields[2],
                "pci_bus_id": fields[3], "driver": fields[4]})
    if len(matches) != 1:
        raise PlanRejected(
            f"nvidia-smi contains GPU index {index} {len(matches)} times")
    return matches[0]


def _terminate_active_child(*, grace_seconds: float = 5.0) -> None:
    global _ACTIVE_CHILD
    child = _ACTIVE_CHILD
    if child is None:
        return
    process_group = child.pid

    def group_alive() -> bool:
        # Reap a finished group leader, then look for any non-zombie member.
        # killpg(0) alone treats an unreaped zombie as alive for the full grace
        # interval and says nothing about a surviving nested process.
        child.poll()
        for item in Path("/proc").iterdir():
            if not item.name.isdigit():
                continue
            try:
                raw = (item / "stat").read_text(encoding="utf-8")
                tail = raw[raw.rfind(")") + 2:].split()
                if len(tail) >= 3 and int(tail[2]) == process_group \
                        and tail[0] != "Z":
                    return True
            except (OSError, ValueError):
                continue
        return False

    if group_alive():
        try:
            os.killpg(process_group, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + grace_seconds
    while group_alive() and time.monotonic() < deadline:
        time.sleep(0.02)
    # The process-group leader can exit before a nested/background trainer.
    # Checking only Popen.poll() released the lease with that orphan alive.
    if group_alive():
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        child.wait(timeout=2)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"ablation process group {child.pid} survived SIGKILL") from error
    finally:
        _ACTIVE_CHILD = None


def _signal_handler(signum, _frame) -> None:
    # Do not call Popen.wait() re-entrantly from a Python signal handler while
    # the main frame is already inside wait(). Request group termination, then
    # let _run_managed's finally block perform the bounded reap after unwinding.
    child = _ACTIVE_CHILD
    if child is not None:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    raise SystemExit(128 + int(signum))


def _run_managed(command: list[str], *, cwd: Path, env: dict,
                 inherited_fds: tuple[int, ...]) -> int:
    global _ACTIVE_CHILD
    previous = {sig: signal.getsignal(sig)
                for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    for sig in previous:
        signal.signal(sig, _signal_handler)
    blocked = set(previous)
    old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, blocked)
    executor_pid = os.getpid()
    guardian = [sys.executable, str(REPO / "scripts" /
                                    "_ablation_group_guard.py")]
    for descriptor in inherited_fds:
        guardian.extend(("--pass-fd", str(descriptor)))
    guardian.extend(("--", *command))

    def child_setup() -> None:
        # Popen's restore_signals resets dispositions, not the pthread signal
        # mask inherited across fork.  Restore it before exec and arm the
        # canonical wrapper to die when this supervisor is SIGKILLed.
        signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
        _arm_parent_death_signal(executor_pid)

    try:
        try:
            _ACTIVE_CHILD = subprocess.Popen(
                guardian, cwd=str(cwd), env=env, start_new_session=True,
                pass_fds=inherited_fds, preexec_fn=child_setup)
        finally:
            # A pending TERM may be delivered from inside this call.  Keep it
            # under the outer cleanup finally so that delivery cannot strand a
            # newly spawned process group.
            signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
        return _ACTIVE_CHILD.wait()
    finally:
        try:
            _terminate_active_child()
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def _resolve_fresh_result(repo: Path, cell: dict, before: set[Path], *,
                          require_fresh: bool) -> Path:
    fixed_n = cell.get("N", (cell.get("env") or {}).get("FIXED_N"))
    fragment = f"_{cell['tag']}_P0refit_e{fixed_n}+"
    root = repo / "result"
    matches = {path.resolve() for path in root.iterdir()
               if path.is_dir() and fragment in path.name} if root.is_dir() else set()
    new = matches - before
    acceptable = (len(new) == 1 if require_fresh else len(matches) == 1)
    if not acceptable or len(matches) != 1:
        raise PlanRejected(
            f"{cell['tag']}: expected one fresh exact result directory, "
            f"found total={sorted(map(str, matches))} new={sorted(map(str, new))}")
    return next(iter(new if require_fresh else matches))


def _validate_campaign_lock_fd(held: dict, descriptor: int) -> Path:
    """Require the inherited FD to own this tag-set's canonical flock."""
    lock_root = Path(
        f"/tmp/groundeddna-ablation-campaign-leases-{os.getuid()}")
    lock_path = lock_root / f"{held['tag_set_sha256']}.lock"
    try:
        root_stat = lock_root.lstat()
        path_stat = lock_path.lstat()
        fd_stat = os.fstat(descriptor)
    except OSError as error:
        raise PlanRejected(f"campaign lock FD is not live: {error}") from None
    if not stat.S_ISDIR(root_stat.st_mode) or lock_root.is_symlink() \
            or root_stat.st_uid != os.getuid() or root_stat.st_mode & 0o077 \
            or not stat.S_ISREG(path_stat.st_mode) or lock_path.is_symlink() \
            or path_stat.st_uid != os.getuid() \
            or (fd_stat.st_dev, fd_stat.st_ino) != (
                path_stat.st_dev, path_stat.st_ino):
        raise PlanRejected("campaign lock FD is not the canonical safe tag lock")
    try:
        # On the inherited open-file description this is idempotent.  A direct
        # caller's separately opened FD fails while another campaign owns the
        # lock, and acquires it itself if no owner exists.
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise PlanRejected("campaign lock FD does not own the tag-set flock") \
            from None
    return lock_path


def _load_production_binding(args, plan_path: Path, plan: dict,
                             cell: dict) -> dict:
    if not args.reservation or not args.campaign_token or not args.outcome \
            or args.campaign_lock_fd is None or args.launcher_pid is None:
        raise PlanRejected(
            "production execution requires reservation, campaign token, "
            "exclusive outcome, launcher PID, and inherited campaign lock FD")
    from scripts.campaign_ledger import _reservation

    reservation_path = Path(args.reservation).resolve()
    held = _reservation(
        reservation_path.parent, campaign_token=args.campaign_token)
    if reservation_path != reservation_path.parent / "campaign_reservation.json":
        raise PlanRejected("reservation path is not canonical")
    if plan_path.resolve() != Path(held["plan_snapshot_path"]):
        raise PlanRejected("executor is not reading the reserved plan snapshot")
    if held["plan_file_sha256"] != args.expect_plan_sha256:
        raise PlanRejected("executor plan digest differs from reservation")
    if Path(args.repo).resolve() != REPO or held["repo_root"] != str(REPO):
        raise PlanRejected("production child repo is not this source root")
    if held.get("launcher_pid") != args.launcher_pid \
            or os.getppid() != args.launcher_pid:
        raise PlanRejected(
            "executor is not a live direct child of the reserving launcher")
    if plan["cells"][args.index].get("tag") != cell.get("tag"):
        raise PlanRejected("cell index/tag changed after reservation")
    outcome = Path(args.outcome)
    expected_parent = reservation_path.parent / "outcomes"
    expected_outcome = expected_parent / f"{cell['tag']}.json"
    if not outcome.is_absolute() or outcome != expected_outcome \
            or not expected_parent.is_dir() or expected_parent.is_symlink() \
            or expected_parent.resolve() != expected_parent \
            or outcome.resolve() != expected_outcome:
        raise PlanRejected(
            "executor outcome must be the canonical ledger/outcomes/tag.json")

    _validate_campaign_lock_fd(held, args.campaign_lock_fd)
    return held


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--expect-plan-sha256", default=None)
    parser.add_argument("--repo", default=str(REPO))
    parser.add_argument("--reservation", default=None)
    parser.add_argument("--campaign-token", default=None)
    parser.add_argument("--outcome", default=None)
    parser.add_argument("--campaign-lock-fd", type=int, default=None)
    parser.add_argument("--launcher-pid", type=int, default=None)
    parser.add_argument("--test-only-unsealed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    production = not args.dry_run and not args.test_only_unsealed
    try:
        if not args.dry_run:
            _arm_parent_death_signal(
                args.launcher_pid if production else None)
        plan_path = Path(args.plan)
        plan = load_plan(plan_path, expect_sha256=args.expect_plan_sha256)
        cell = cell_at(plan, args.index)
        env = child_env(plan, cell, production=production)
        held = (_load_production_binding(args, plan_path, plan, cell)
                if production else None)
        repo = REPO if production else Path(args.repo).resolve()
        if args.dry_run:
            command = child_argv(plan, cell, "GPU-DRY-RUN", production=False)
            json.dump({"cmd": command, "env": env, "repo": str(repo)},
                      sys.stdout, indent=2, sort_keys=True)
            print()
            return 0

        gpu = _gpu_assignment(
            args.gpu,
            executable=("/usr/bin/nvidia-smi" if production
                        else "nvidia-smi"))
        command = child_argv(
            plan, cell, str(gpu["uuid"]), production=production)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu["uuid"])
        root = repo / "result"
        fixed_n = cell.get("N", (cell.get("env") or {}).get("FIXED_N"))
        fragment = f"_{cell['tag']}_P0refit_e{fixed_n}+"
        before = {path.resolve() for path in root.iterdir()
                  if path.is_dir() and fragment in path.name} \
            if root.is_dir() else set()
        if production and before:
            raise PlanRejected(
                f"{cell['tag']}: matching output already exists; a campaign "
                "generation never resumes or adopts old output")

        from dna_utils.gpu_lease import acquire_gpu_leases
        leases = acquire_gpu_leases(
            [gpu], owner_metadata={
                "campaign": "phase5_ablation", "tag": cell["tag"],
                "campaign_token": args.campaign_token,
                "plan_file_sha256": args.expect_plan_sha256,
                "repo": str(repo),
            })
        inherited = [int(handle.fileno()) for handle in leases.handles]
        if production:
            inherited.append(int(args.campaign_lock_fd))
        try:
            returncode = _run_managed(
                command, cwd=repo, env=env,
                inherited_fds=tuple(sorted(set(inherited))))
            if returncode != 0:
                return returncode
            run_dir = _resolve_fresh_result(
                repo, cell, before, require_fresh=production)
            if production:
                from scripts.campaign_ledger import (
                    _publish_json_exclusive, source_digests)
                current_sources = source_digests(REPO)
                if current_sources != held["source_sha256"]:
                    raise PlanRejected(
                        "campaign source changed while the trainer ran")
                outcome = {
                    "schema": "groundeddna.ablation-cell-outcome",
                    "schema_version": 1,
                    "campaign_token": args.campaign_token,
                    "plan_file_sha256": args.expect_plan_sha256,
                    "cell_index": args.index, "tag": cell["tag"],
                    "child_returncode": 0, "repo_root": str(REPO),
                    "run_dir": str(run_dir), "gpu": gpu,
                    "gpu_lease_paths": [str(path) for path in leases.paths],
                    "source_sha256": current_sources,
                    "executor_pid": os.getpid(),
                    "completed_at_utc": datetime.now(
                        timezone.utc).isoformat(timespec="seconds"),
                }
                # Recheck the path immediately before exclusive publication;
                # the long-running trainer must not turn a replaced outcomes
                # symlink into an off-ledger write.
                reservation = Path(args.reservation).resolve()
                outcome_path = Path(args.outcome)
                expected_parent = reservation.parent / "outcomes"
                expected_outcome = expected_parent / f"{cell['tag']}.json"
                if not expected_parent.is_dir() \
                        or expected_parent.is_symlink() \
                        or expected_parent.resolve() != expected_parent \
                        or outcome_path != expected_outcome \
                        or outcome_path.resolve() != expected_outcome:
                    raise PlanRejected(
                        "canonical outcome path changed while trainer ran")
                _publish_json_exclusive(outcome_path, outcome)
            return 0
        finally:
            leases.release()
    except (PlanRejected, OSError, RuntimeError, KeyError, TypeError) as error:
        print(f"[ablation-exec] REFUSED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
