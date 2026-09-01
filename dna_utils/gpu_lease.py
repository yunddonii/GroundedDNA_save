"""Host-global physical-GPU UUID leases shared by paper campaigns.

Result/log-root locks cannot prevent independent launchers from scheduling the
same GPU.  This primitive uses kernel ``flock`` state as the authority and a
single per-user host namespace shared by every GroundedDNA campaign.  The lock
file survives a crash only as diagnostic metadata; the kernel lease itself is
released automatically on process death or reboot, so stale-file deletion is
neither necessary nor trusted.
"""
from __future__ import annotations

import atexit
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import sys
from typing import Mapping, Sequence


GPU_LEASE_ROOT = Path(f"/tmp/groundeddna-d6-gpu-leases-{os.getuid()}")
_GPU_UUID_RE = re.compile(r"^GPU-[A-Za-z0-9-]+$")


@dataclass
class GpuLeaseSet:
    """A set of live kernel-held physical-GPU UUID leases."""

    handles: list[object]
    paths: tuple[Path, ...]
    _released: bool = False

    def release(self) -> None:
        if self._released:
            return
        for handle in reversed(self.handles):
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)  # type: ignore[attr-defined]
            finally:
                handle.close()  # type: ignore[attr-defined]
        self._released = True

    def __enter__(self) -> "GpuLeaseSet":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()


def _boot_id() -> str:
    path = Path("/proc/sys/kernel/random/boot_id")
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise RuntimeError(f"cannot read host boot ID for GPU lease: {error}") \
            from error
    if not value:
        raise RuntimeError("host boot ID for GPU lease is empty")
    return value


def _safe_lease_root(root: Path) -> Path:
    root.mkdir(mode=0o700, parents=False, exist_ok=True)
    stat = root.lstat()
    if not root.is_dir() or stat.st_uid != os.getuid() \
            or stat.st_mode & 0o077:
        raise RuntimeError(f"unsafe host-global GPU lease directory: {root}")
    return root


def acquire_gpu_leases(
        assignments: Sequence[Mapping[str, object]], *,
        owner_metadata: Mapping[str, object] | None = None,
        lease_root: Path = GPU_LEASE_ROOT,
        ) -> GpuLeaseSet:
    """Acquire one nonblocking host-global lease per physical GPU UUID.

    ``assignments`` may use campaign-specific fields, but every record must
    contain a canonical NVIDIA ``uuid``.  Sorting prevents cross-launcher
    deadlock when a campaign requests several GPUs in a different order.
    """
    root = _safe_lease_root(Path(lease_root))
    normalized: list[dict[str, object]] = []
    seen: set[str] = set()
    for raw in assignments:
        assignment = dict(raw)
        uuid = assignment.get("uuid")
        if not isinstance(uuid, str) or _GPU_UUID_RE.fullmatch(uuid) is None:
            raise RuntimeError(f"unsafe GPU UUID for lease: {uuid!r}")
        if uuid in seen:
            raise RuntimeError(f"duplicate GPU UUID lease request: {uuid}")
        seen.add(uuid)
        normalized.append(assignment)
    if not normalized:
        raise RuntimeError("at least one GPU assignment is required for a lease")
    normalized.sort(key=lambda value: str(value["uuid"]))

    handles: list[object] = []
    paths: list[Path] = []
    try:
        for assignment in normalized:
            uuid = str(assignment["uuid"])
            path = root / f"{uuid}.lock"
            flags = os.O_RDWR | os.O_CREAT
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(path, flags, 0o600)
            handle = os.fdopen(descriptor, "r+", encoding="utf-8")
            try:
                fcntl.flock(
                    handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                handle.seek(0)
                holder = handle.read().strip()
                handle.close()
                raise RuntimeError(
                    "physical GPU already has a live GroundedDNA lease: "
                    f"{uuid}; holder={holder or 'metadata unavailable'}") from error
            handles.append(handle)
            paths.append(path)
            metadata = {
                "schema": "groundeddna.host-global-gpu-lease",
                "schema_version": 1,
                "pid": os.getpid(),
                "boot_id": _boot_id(),
                "acquired_at_utc": datetime.now(timezone.utc).isoformat(
                    timespec="seconds"),
                "command": list(sys.argv),
                "gpu_assignment": assignment,
                "owner": dict(owner_metadata or {}),
            }
            handle.seek(0)
            handle.truncate()
            json.dump(
                metadata, handle, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        for handle in reversed(handles):
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)  # type: ignore[attr-defined]
            handle.close()  # type: ignore[attr-defined]
        raise
    lease_set = GpuLeaseSet(handles=handles, paths=tuple(paths))
    atexit.register(lease_set.release)
    return lease_set
