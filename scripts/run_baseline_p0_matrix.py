#!/usr/bin/env python3
"""Launch the fixed baseline P0 matrix on a GPU pool.

This is an orchestration layer around ``run_modern_baseline_p0.py``.  It does
not override a method's training horizon, batch size, optimizer, scheduler, or
loss defaults.  Its production default is D6 ``author_fixed_final``: explicit
30-bit budget, explicit per-dataset eligible caches, one full-train run to the
author horizon, and terminal LAST checkpoint.  The former selector/refit path
is available only as ``validation_sensitivity`` and is always diagnostic.

The default strict visual-only (U0) panel contains 9 methods x 4 datasets x 2
bit budgets x train seed 42 = 72 cells.  UMRCH is launched as a separate
taxonomy-assisted U2 panel (3 datasets x 2 budgets = 6 cells).  The supervised
CRH baseline is opt-in via ``--panel supervised`` and is never included in U0,
U2, or the historical default ``all`` panel.  Exact DUH-EG is deliberately
absent: the ordered selected-noun artifact required to reproduce it is
unavailable.

Typical launch (six physical GPUs):

    /home/yschoi/.conda/envs/dna_hashing/bin/python \
      scripts/run_baseline_p0_matrix.py --gpus 0 1 2 3 4 5

Inspect the complete command plan without creating files or starting jobs:

    python scripts/run_baseline_p0_matrix.py --dry-run --gpus 0 1 2 3 4 5
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from typing import Iterable, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "scripts/run_modern_baseline_p0.py"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from baseline.execution_environment import (  # noqa: E402
    ExecutionEnvironmentError,
    resolve_gpu_assignments,
    validate_gpu_assignment,
)
from dna_utils.gpu_lease import (  # noqa: E402
    GpuLeaseSet,
    acquire_gpu_leases,
)


def _process_group_has_live_members(process_group_id: int) -> bool:
    """Return whether a process group still contains a non-zombie process.

    ``Popen.poll()`` covers only the direct child.  A runner can exit while a
    dataloader/grandchild in the same session remains alive, so shutdown must
    reason about the whole process group.  Linux ``/proc`` also lets us treat
    zombies as dead: they cannot execute CUDA work or retain the inherited GPU
    lease file description.  The kill(2) probe is a conservative fallback.
    """
    proc_root = Path("/proc")
    if proc_root.is_dir():
        try:
            for entry in proc_root.iterdir():
                if not entry.name.isdecimal():
                    continue
                try:
                    stat_text = (entry / "stat").read_text(encoding="utf-8")
                except (FileNotFoundError, PermissionError, ProcessLookupError):
                    continue
                right_paren = stat_text.rfind(")")
                if right_paren < 0:
                    continue
                fields = stat_text[right_paren + 2:].split()
                # After ``pid (comm)``, fields are state, ppid, pgrp, ...
                if len(fields) >= 3 \
                        and int(fields[2]) == int(process_group_id) \
                        and fields[0] != "Z":
                    return True
            return False
        except OSError:
            pass
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class _ManagedChildSupervisor:
    """Race-free child registry and bounded whole-group termination.

    The lease descriptors are deliberately inherited by every direct runner.
    If this launcher is SIGKILLed, the kernel therefore keeps the physical GPU
    lease held for as long as the surviving runner can still use that GPU.
    """

    def __init__(
            self, lease_fds: Sequence[int], *,
            term_grace_seconds: float = 5.0,
            kill_wait_seconds: float = 1.0,
            ) -> None:
        normalized = tuple(int(fd) for fd in lease_fds)
        if len(set(normalized)) != len(normalized):
            raise RuntimeError("duplicate GPU lease descriptor")
        for descriptor in normalized:
            os.fstat(descriptor)
        self._lease_fds = normalized
        self._term_grace_seconds = max(0.0, float(term_grace_seconds))
        self._kill_wait_seconds = max(0.0, float(kill_wait_seconds))
        self._lock = threading.RLock()
        self._active: set[subprocess.Popen[bytes]] = set()
        self._launch_blocked = False

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._active)

    def spawn(self, command: Sequence[str], **kwargs: object) \
            -> subprocess.Popen[bytes]:
        """Spawn and register without a signal-visible unregistered window."""
        if "start_new_session" in kwargs or "pass_fds" in kwargs:
            raise RuntimeError(
                "managed spawn owns start_new_session and pass_fds")
        # Python signal handlers execute in the main thread.  Popen is called
        # by matrix worker threads; a handler that arrives during Popen blocks
        # on this same lock until the PID/PGID has been registered.
        with self._lock:
            if self._launch_blocked:
                raise RuntimeError(
                    "matrix shutdown has begun; refusing a new child process")
            process = subprocess.Popen(
                list(command), start_new_session=True,
                pass_fds=self._lease_fds, **kwargs)  # type: ignore[arg-type]
            self._active.add(process)
        return process

    def wait(self, process: subprocess.Popen[bytes]) -> int:
        try:
            return process.wait()
        finally:
            # Keep the PGID registered if the direct child exited but a
            # grandchild remains.  ``terminate_all`` will close that group.
            with self._lock:
                if not _process_group_has_live_members(process.pid):
                    self._active.discard(process)

    @staticmethod
    def _signal_group(process_group_id: int, signum: int) -> None:
        try:
            os.killpg(process_group_id, signum)
        except ProcessLookupError:
            pass

    def terminate_all(self) -> None:
        """Block new launches, then TERM/KILL/wait every registered group."""
        with self._lock:
            self._launch_blocked = True
            children = tuple(self._active)

        for process in children:
            self._signal_group(process.pid, signal.SIGTERM)

        deadline = time.monotonic() + self._term_grace_seconds
        pending = [
            process for process in children
            if _process_group_has_live_members(process.pid)
        ]
        while pending and time.monotonic() < deadline:
            for process in children:
                process.poll()
            time.sleep(min(0.02, max(0.0, deadline - time.monotonic())))
            pending = [
                process for process in pending
                if _process_group_has_live_members(process.pid)
            ]

        for process in pending:
            self._signal_group(process.pid, signal.SIGKILL)

        kill_deadline = time.monotonic() + self._kill_wait_seconds
        still_live = [
            process for process in pending
            if _process_group_has_live_members(process.pid)
        ]
        while still_live and time.monotonic() < kill_deadline:
            for process in children:
                process.poll()
            time.sleep(min(
                0.02, max(0.0, kill_deadline - time.monotonic())))
            still_live = [
                process for process in still_live
                if _process_group_has_live_members(process.pid)
            ]

        unreaped: list[int] = []
        for process in children:
            try:
                process.wait(timeout=self._kill_wait_seconds)
            except subprocess.TimeoutExpired:
                unreaped.append(process.pid)
        still_live = [
            process for process in children
            if _process_group_has_live_members(process.pid)
        ]
        with self._lock:
            for process in children:
                if process.pid not in unreaped \
                        and process not in still_live:
                    self._active.discard(process)
        if unreaped or still_live:
            survivors = sorted(set(
                unreaped + [process.pid for process in still_live]))
            raise RuntimeError(
                "matrix child process groups survived SIGKILL/wait: "
                f"{survivors}")


class _CampaignLifecycle:
    """Main-thread signal supervision for one leased matrix campaign."""

    def __init__(
            self, lease_fds: Sequence[int], *,
            term_grace_seconds: float = 5.0,
            kill_wait_seconds: float = 1.0,
            ) -> None:
        self.children = _ManagedChildSupervisor(
            lease_fds,
            term_grace_seconds=term_grace_seconds,
            kill_wait_seconds=kill_wait_seconds)
        self.stop = threading.Event()
        self.interrupted_signum: int | None = None
        self._cleanup_errors: list[Exception] = []
        self._previous_handlers: dict[int, object] = {}
        self._installed = False

    def install_signal_handlers(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError(
                "matrix signal supervision must start in the main thread")
        if self._installed:
            raise RuntimeError("matrix signal handlers are already installed")
        # SIGHUP belongs here with the other two. `tmux kill-session` -- the
        # actual operational way a campaign gets stopped -- delivers SIGHUP,
        # which was unhandled and so took the default action: the matrix died
        # without running _handle_signal, and every direct child, living in its
        # own session/PGID from start_new_session=True, survived with PPID 1.
        # That is exactly the 2026-09-08 orphans: a trainer that ran on to
        # epoch 999 after its parent was gone, and a projector still holding
        # 12.8 GB of GPU. getattr keeps this total on platforms without SIGHUP.
        supervised_signals = [signal.SIGINT, signal.SIGTERM]
        sighup = getattr(signal, "SIGHUP", None)
        if sighup is not None:
            supervised_signals.append(sighup)
        self._previous_handlers = {
            signum: signal.getsignal(signum)
            for signum in supervised_signals
        }
        installed: list[int] = []
        try:
            for signum in self._previous_handlers:
                signal.signal(signum, self._handle_signal)
                installed.append(signum)
        except BaseException:
            for signum in installed:
                signal.signal(signum, self._previous_handlers[signum])
            self._previous_handlers = {}
            raise
        self._installed = True

    def _handle_signal(self, signum: int, _frame: object) -> None:
        if self.interrupted_signum is None:
            self.interrupted_signum = int(signum)
        self.stop.set()
        print(
            f"[matrix] received signal {signum}; stopping process groups",
            flush=True)
        try:
            self.children.terminate_all()
        except Exception as error:
            self._cleanup_errors.append(error)

    def close(self) -> None:
        """Finish child cleanup before restoring handlers/releasing leases."""
        self.stop.set()
        try:
            self.children.terminate_all()
        except Exception as error:
            self._cleanup_errors.append(error)
        finally:
            if self._installed:
                for signum, handler in self._previous_handlers.items():
                    signal.signal(signum, handler)
                self._installed = False
        if self._cleanup_errors:
            messages = "; ".join(
                f"{type(error).__name__}: {error}"
                for error in self._cleanup_errors)
            raise RuntimeError(f"matrix child cleanup failed: {messages}")

U0_VARIANTS = (
    "cibhash",
    "cimon",
    "mls3rduh",
    "greedyhash",
    "bihalf",
    "sdc-paper",
    "oh",
    "hhch",
    "crovca",
)
DATASETS = ("Flickr25k", "MSCOCO", "NUSWIDE", "CIFAR10")
U2_DATASETS = ("Flickr25k", "MSCOCO", "NUSWIDE")
SUPERVISED_VARIANTS = ("crh-supervised",)
SUPERVISED_DATASETS = DATASETS
# 30 added 2026-08-10: the 5-slot GroundedDNA variant emits 15 bases = 30
# bits, so the unsupervised baselines need a matched budget. `base_length`
# is already derived as bit // 2, so nothing else assumes 36/48.
BITS = (30, 36, 40, 48)
AUTHOR_FIXED_FINAL = "author_fixed_final"
VALIDATION_SENSITIVITY = "validation_sensitivity"
PROTOCOL_MODES = (AUTHOR_FIXED_FINAL, VALIDATION_SENSITIVITY)
DEFAULT_SEEDS = (42,)
METHOD_GROUPS: Mapping[str, tuple[str, tuple[str, ...]]] = {
    "all": ("all", U0_VARIANTS),
    "all_plus_supervised": ("all-plus-supervised", U0_VARIANTS),
    "all_no_crovca": (
        "all", tuple(variant for variant in U0_VARIANTS if variant != "crovca")),
    "u0": ("u0", U0_VARIANTS),
    "legacy": ("u0", ("cibhash", "cimon", "mls3rduh")),
    "modern": (
        "u0", ("greedyhash", "bihalf", "sdc-paper", "oh", "hhch", "crovca")),
    "modern_no_crovca": (
        "u0", ("greedyhash", "bihalf", "sdc-paper", "oh", "hhch")),
    "u2": ("u2", U0_VARIANTS),
    "umrch": ("u2", U0_VARIANTS),
    # --variants controls only the U0 branch.  The supervised branch has one
    # deliberately fixed variant, so an empty default is both clearer and safe.
    "supervised": ("supervised", ()),
    "crh": ("supervised", ()),
}

UMRCH_ASSETS: Mapping[str, tuple[str, str, str]] = {
    "Flickr25k": (
        "artifacts/umrch_flickr25k_embeddings.npy",
        "artifacts/umrch_flickr25k_vision_adapter.npz",
        "artifacts/umrch_flickr25k_manifest.json",
    ),
    "MSCOCO": (
        "artifacts/umrch_mscoco_embeddings.npy",
        "artifacts/umrch_mscoco_vision_adapter.npz",
        "artifacts/umrch_mscoco_manifest.json",
    ),
    "NUSWIDE": (
        "artifacts/umrch_nuswide_embeddings.npy",
        "artifacts/umrch_nuswide_vision_adapter.npz",
        "artifacts/umrch_nuswide_manifest.json",
    ),
}

# Scheduling weights affect only launch order.  They never become runner
# arguments, so the runner remains the sole source of method defaults.
ROUGH_HORIZON = {
    "cibhash": 60,
    "cimon": 150,
    "mls3rduh": 150,
    "greedyhash": 60,
    "sdc-paper": 100,
    "oh": 200,
    "hhch": 80,
    "crovca": 5,
    "umrch": 100,
}
BIHALF_HORIZON = {
    "Flickr25k": 100,
    "MSCOCO": 150,
    "NUSWIDE": 100,
    "CIFAR10": 300,
}
CRH_HORIZON = {
    "Flickr25k": 30,
    "MSCOCO": 30,
    "NUSWIDE": 30,
    "CIFAR10": 300,
}

LEGACY_LABEL = "legacy_cache_diagnostic_only_not_main_table_eligible"
AUTHOR_LABEL = "d6_author_fixed_final_paper_main_candidate"
# Completion/resume is source-profile aware.  Otherwise an internally valid
# historical run could suppress a required canonical rerun and only be
# rejected later by the aggregator, silently leaving a missing matrix cell.
# Keep these exact method-file profiles synchronized with
# aggregate_baseline_p0_matrix.CANONICAL_VARIANT_SOURCE_PROFILES.
CANONICAL_VARIANT_SOURCE_PROFILES: Mapping[str, tuple[str, str]] = {
    "cibhash": (
        "baseline/CIBHash.py",
        "ce1a1e3fde2c87eb9fe34644e11f63ca4c84fb757ac0eb17c126ccf0cabc3c8e",
    ),
    "cimon": (
        "baseline/CIMON.py",
        # F05: bumped from afd4696e2b36... after matching the objective to the
        # official implementation. Cells trained under the old digest are
        # excluded by the profile check rather than exempted.
        "6f7a4864d3c245afe095a2d2c6dc3c155197a17e10ebd743f1466b3c1ff1136a",
    ),
    "mls3rduh": (
        "baseline/MLS3RDUH.py",
        "98c8e29b488b71cf60e01835681d26e8231136b5ed30d0ccc1ab33de6998fc8f",
    ),
    "greedyhash": (
        "baseline/GreedyHash.py",
        "5d80dca3034f1d830b32d80c3c27628d53b131333ead5b1f8ac7fa3e35f3bc89",
    ),
    "bihalf": (
        "baseline/BiHalf.py",
        "b41d9dc79ae55501dabca0a10790fb699702eff5bb0b0e89198543190516a2c2",
    ),
    "sdc-paper": (
        "baseline/SDC.py",
        "bf2e13936866ffc80c163510321d69a5ba67bdc7bef6b5885eaf75edc166b1e6",
    ),
    "oh": (
        "baseline/OH.py",
        "53edb7e0081bfff56cd17b8f1baf9b559e5e91966ac03bb0732acdeae19f0ef1",
    ),
    "hhch": (
        "baseline/HHCH.py",
        "72c1f745dc9b795b99012571204fae74e59cf23a02f9c729de9d39c758856e96",
    ),
    "crovca": (
        "baseline/CroVCA.py",
        "58f71fb4411afae49c5117a2f476818b1736cfb8f78ea5e0585e53f66076cc4d",
    ),
    "umrch": (
        "baseline/UMRCH.py",
        "c747aca825d14a2e1eb31f5416208ee43446e57856c28753ada7ee15d7e01bf5",
    ),
    "crh-supervised": (
        "baseline/CRH.py",
        "ef09c8d29989dad4c98c466880133d52649975fc397a4b2a3a0c82619b37013d",
    ),
}
COMMON_REQUIRED_IMPLEMENTATION_PATHS = frozenset({
    "scripts/run_modern_baseline_p0.py",
    "scripts/run_baseline_p0_matrix.py",
    "baseline/base_model.py",
    "baseline/modern_unsupervised.py",
    "baseline/asset_provenance.py",
    "baseline/cache_provenance.py",
    "baseline/execution_environment.py",
    "dna_utils/runtime_environment.py",
    "dna_utils/gpu_lease.py",
    "scripts/baseline_val_select_p0.py",
    "scripts/extract_flat_baseline.py",
    "scripts/apply_bio_projection.py",
    "dna_utils/bio_constraints.py",
    "val_split.py",
})
KNOWN_NON_SCIENTIFIC_SOURCE_SHA_ALIASES: Mapping[str, frozenset[str]] = {
    "baseline/cache_provenance.py": frozenset({
        "ec761115371f09e6e3a00ab816888188b0df234805920b153ef06e08a598405c",
        "4fe40c862a23e265ecf0559232d8b6ac5b84b282df720670c3aed58e3daf588d",
        "3e137fa7a5180370f810a9b2555292aa0ac9a6794189d89550454a9967805ba8",
    }),
    # Exact reviewed transition: CRH was added only as a new lazy dispatch
    # branch/help entry.  Existing U0/U2 model construction is unchanged.
    "baseline/base_model.py": frozenset({
        "c9f39c05a27cca24ab0084bdd00462b634953499021a67c1cdc18dfc33837a47",
        "b9d7e1a0cfab613fd72ce3dc38fafeca82ca99f6121d20138bf22239ef34f3be",
        # D6 mode/stage provenance checks only; model/loss/split unchanged.
        "22ea888f88cd52cfd476b3ed76514345e5d45fc7ba159bf2904c35ee67fd767d",
        # Tightens only explicit validation-sensitivity ratio/stage metadata.
        "f5a1a1b25a89ac5e9ca8bdf17ab2605921cb81dfc90a38ccef6d1b12dcd57add",
    }),
    # Exact reviewed transition: the new CRH variant/semantic branch and
    # comparison-panel metadata do not alter any existing U0/U2 training,
    # selection, extraction, metric, or DNA-projection path.
    "scripts/run_modern_baseline_p0.py": frozenset({
        "9809a70fde66d473540fa11d10752a83453d60ac6bfc9b80d0490a1e7ce7eca5",
        "2d7234a2a8f959f32a1f0fa5199cf00556289084f351359f7cb3cbcbdd4f5b16",
        "1dec886eaed08b4f01cc04c8ebaec81b01952d1bc6913696cdcc7a75830461e0",
        # The two SUPPORTED_BITS widenings, 30 (52d315b8) and then 40
        # (21fbb21a). Without the current digest in this set, every manifest
        # recorded before them fails the profile check outright.
        "4e9959b28fd7118ecdbd794555e22ede53064df6fe447c2ab85d30c1c8541a6c",
        "c333bf1aafcf809b57f5e3e46eb977b1666936c4ec962e927972a2804798ad4a",
        # D6 adds a protocol-mode-discriminated author-fixed orchestration
        # branch while retaining the old selection/refit branch verbatim as
        # validation_sensitivity. Old manifests can match only that diagnostic
        # mode; author_fixed_final requires the new explicit schema.
        "afd8fd0d7d1fb894b7948a321c6cbbe6bf2f7786baf8e21e0c210f75529bb429",
        # Final checkpoint mode/stage/digest reopen; training is unchanged.
        "175654df2bcedb43800bd00c77d19b53572e53aff2327dc6d0a2532b812c6aec",
        # Same reopen moved immediately before pre-BIO manifest publication.
        "e412ea8e48b7075be6b26351e48a9050234716129706f871fbece900dc277091",
        # Binds the actual seven extracted CIFAR source files, not only tarball.
        "cccd1a307a04f3ca63b9cdf2eac735e1fcfc2ed6096c90a3fa2b361f520cfbe6",
        # D6 decode-failure provenance gate only. It changes admission and
        # evidence, never model/loss/schedule/split behavior; old D6 manifests
        # still fail the mandatory audit-field validator below.
        "ad7b73f3124ffbeedfa115650c7d804d964a191bf125c6b2b3cc7abfbe15fd54",
        # The digest every completed 30-bit author_fixed_final cell recorded.
        # Until 52faa66 it was also the current source, so no alias was needed
        # and its absence here went unnoticed.
        "3a50232b85fe167d9e22f8ef1b7f9826531fb12639e4f4355114d86c7dfe99dd",
        # 52faa66: the Bi-half/NUS-WIDE pair moves from an eligibility blocker
        # to a recorded source_boundary_adaptations entry. That branch is
        # guarded by `variant == 'bihalf' and dataset == 'NUSWIDE'` and had
        # produced no manifest at all -- it refused before training -- so no
        # completed cell of any variant ever executed it. Training, loss,
        # schedule, split, metric and projection are untouched.
        "dacee2e311061c2c877cc374418600a629cc22db5b7f5b862626bdf18d1decd7",
    }),
}
# Some shared-source edits are scientific only for one method, and an older
# source may predate a newly added variant entirely.  Scope those reviewed
# aliases by both the *recorded* digest and variant; a path-wide exemption
# would incorrectly admit old-horizon CIBHash or pre-dispatch CRH manifests.
KNOWN_NON_SCIENTIFIC_SOURCE_SHA_ALIAS_VARIANTS: Mapping[
    str, Mapping[str, frozenset[str]]
] = {
    "baseline/base_model.py": {
        # This snapshot predates the CRH lazy-dispatch branch.  The edit is
        # non-scientific for every pre-existing U0/U2 method, but a CRH result
        # cannot have been produced through this version of the dispatcher.
        "c9f39c05a27cca24ab0084bdd00462b634953499021a67c1cdc18dfc33837a47": (
            frozenset(CANONICAL_VARIANT_SOURCE_PROFILES)
            - frozenset(SUPERVISED_VARIANTS)
        ),
        "b9d7e1a0cfab613fd72ce3dc38fafeca82ca99f6121d20138bf22239ef34f3be": (
            frozenset(CANONICAL_VARIANT_SOURCE_PROFILES)
        ),
        "22ea888f88cd52cfd476b3ed76514345e5d45fc7ba159bf2904c35ee67fd767d": (
            frozenset(CANONICAL_VARIANT_SOURCE_PROFILES)
        ),
    },
    "scripts/run_modern_baseline_p0.py": {
        # The digest every completed 30-bit author_fixed_final cell recorded.
        # 52faa66 only moved the Bi-half/NUS-WIDE adaptation out of the
        # eligibility blockers and into source_boundary_adaptations. That
        # branch is guarded by `variant == 'bihalf' and dataset == 'NUSWIDE'`
        # and refused before training, so it produced no manifest and no
        # completed cell of any variant ever entered it. Non-scientific for
        # every variant, including bihalf on its other three datasets.
        "3a50232b85fe167d9e22f8ef1b7f9826531fb12639e4f4355114d86c7dfe99dd": (
            frozenset(CANONICAL_VARIANT_SOURCE_PROFILES)
        ),
        # Before CRH dispatch existed.  The CIBHash horizon was also stale.
        "9809a70fde66d473540fa11d10752a83453d60ac6bfc9b80d0490a1e7ce7eca5": (
            frozenset(CANONICAL_VARIANT_SOURCE_PROFILES)
            - {"cibhash"}
            - frozenset(SUPERVISED_VARIANTS)
        ),
        # CRH dispatch exists; only the later CIBHash horizon correction is
        # scientific, and only for CIBHash itself.
        "2d7234a2a8f959f32a1f0fa5199cf00556289084f351359f7cb3cbcbdd4f5b16": (
            frozenset(CANONICAL_VARIANT_SOURCE_PROFILES) - {"cibhash"}
        ),
        # Neither of these had a scope entry, and a recorded digest with no
        # entry fails closed -- so a manifest from either revision was rejected
        # for every variant. Both carry the corrected CIBHash horizon and differ
        # from current only by a SUPPORTED_BITS line, so both are comparable
        # everywhere, CIBHash included.
        "1dec886eaed08b4f01cc04c8ebaec81b01952d1bc6913696cdcc7a75830461e0": (
            frozenset(CANONICAL_VARIANT_SOURCE_PROFILES)
        ),
        "4e9959b28fd7118ecdbd794555e22ede53064df6fe447c2ab85d30c1c8541a6c": (
            frozenset(CANONICAL_VARIANT_SOURCE_PROFILES)
        ),
        "c333bf1aafcf809b57f5e3e46eb977b1666936c4ec962e927972a2804798ad4a": (
            frozenset(CANONICAL_VARIANT_SOURCE_PROFILES)
        ),
        "afd8fd0d7d1fb894b7948a321c6cbbe6bf2f7786baf8e21e0c210f75529bb429": (
            frozenset(CANONICAL_VARIANT_SOURCE_PROFILES)
        ),
        "175654df2bcedb43800bd00c77d19b53572e53aff2327dc6d0a2532b812c6aec": (
            frozenset(CANONICAL_VARIANT_SOURCE_PROFILES)
        ),
        "e412ea8e48b7075be6b26351e48a9050234716129706f871fbece900dc277091": (
            frozenset(CANONICAL_VARIANT_SOURCE_PROFILES)
        ),
        "cccd1a307a04f3ca63b9cdf2eac735e1fcfc2ed6096c90a3fa2b361f520cfbe6": (
            frozenset(CANONICAL_VARIANT_SOURCE_PROFILES)
        ),
    },
}
DUHEG_BLOCK = {
    "variant": "duheg",
    "status": "blocked_not_scheduled",
    "reason": (
        "Exact DUH-EG requires the authors' ordered selected-WordNet noun bank "
        "or an unambiguous selection specification; neither is available."
    ),
}


@dataclass(frozen=True)
class Job:
    panel: str
    variant: str
    dataset: str
    bit: int
    seed: int

    @property
    def key(self) -> tuple[str, str, int, int]:
        return self.variant, self.dataset, self.bit, self.seed

    @property
    def job_id(self) -> str:
        dataset = self.dataset.lower()
        return f"{self.panel}_{self.variant}_{dataset}_{self.bit}b_seed{self.seed}"

    @property
    def scheduling_weight(self) -> int:
        horizon = _expected_horizon(self.variant, self.dataset)
        # MSCOCO has the largest designated training split in this matrix.
        dataset_factor = {"MSCOCO": 4, "NUSWIDE": 2,
                          "Flickr25k": 1, "CIFAR10": 1}[self.dataset]
        return horizon * dataset_factor


@dataclass(frozen=True)
class Attempt:
    job: Job
    number: int
    model_root: Path
    result_root: Path
    compress_root: Path
    log_path: Path
    status_path: Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _absolute(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (REPO / value).resolve()


def _comparison_label(protocol_mode: str, *, allow_smoke: bool) -> str:
    if protocol_mode == AUTHOR_FIXED_FINAL:
        return (
            "d6_author_fixed_final_diagnostic_smoke_not_main_eligible"
            if allow_smoke else AUTHOR_LABEL)
    return "validation_sensitivity_diagnostic_only_not_paper_main"


def _parse_cache_dirs(values: Sequence[str]) -> dict[str, Path]:
    """Parse repeatable DATASET=PATH declarations without silent overwrite."""
    result: dict[str, Path] = {}
    for value in values:
        dataset, separator, raw_path = value.partition("=")
        if not separator or dataset not in DATASETS or not raw_path:
            raise ValueError(
                "--cache-dir must be repeated as DATASET=PATH, where DATASET "
                f"is one of {DATASETS}; found {value!r}")
        if dataset in result:
            raise ValueError(f"duplicate --cache-dir declaration for {dataset}")
        result[dataset] = _absolute(raw_path)
    return result


def _environment_tokens(name: str, fallback: str) -> list[str]:
    values = os.environ.get(name, fallback).replace(",", " ").split()
    return values or fallback.replace(",", " ").split()


def _expected_horizon(variant: str, dataset: str) -> int | None:
    if variant == "bihalf":
        return BIHALF_HORIZON.get(dataset)
    if variant == "crh-supervised":
        return CRH_HORIZON.get(dataset)
    return ROUGH_HORIZON.get(variant)


def _method_group_from_environment() -> tuple[str, list[str]]:
    raw = os.environ.get("METHOD_GROUP", "all").strip().lower()
    if raw in METHOD_GROUPS:
        panel, variants = METHOD_GROUPS[raw]
        return panel, list(variants)
    variants = raw.replace(",", " ").split()
    invalid = sorted(set(variants) - set(U0_VARIANTS))
    if not variants or invalid:
        raise ValueError(
            "METHOD_GROUP must be one of "
            f"{sorted(METHOD_GROUPS)} or an explicit U0 variant list; "
            f"invalid={invalid}")
    return "u0", variants


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _protocol_identity_digest(identity: Mapping[str, object]) -> str:
    canonical = json.dumps(
        identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _required_implementation_paths(variant: str) -> frozenset[str]:
    profile = CANONICAL_VARIANT_SOURCE_PROFILES.get(variant)
    if profile is None:
        return frozenset()
    return COMMON_REQUIRED_IMPLEMENTATION_PATHS | {profile[0]}


@lru_cache(maxsize=None)
def _current_source_sha256(source_path: str) -> str | None:
    path = (REPO / source_path).resolve()
    try:
        path.relative_to(REPO.resolve())
    except ValueError:
        return None
    return _sha256(path) if path.is_file() else None


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp.{os.getpid()}.{threading.get_ident()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _manifest_bit(payload: Mapping[str, object]) -> int | None:
    raw = payload.get("bit_length")
    if raw is None:
        identity = payload.get("protocol_identity")
        if isinstance(identity, Mapping):
            raw = identity.get("bit")
    try:
        return int(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


#: This module's own path, as recorded in every manifest's implementation map.
SELF_SOURCE_RELATIVE = "scripts/run_baseline_p0_matrix.py"


def _self_transition_is_reviewed(recorded: str, current: str) -> bool:
    """Is recorded->current an approved transition of THIS file?

    Fail closed on every uncertainty: a missing registry entry, an unreviewed
    digest on either side, or a current file that is not the registry's
    declared `after_sha256`. Editing this launcher arbitrarily still refuses;
    only the exact reviewed pair passes.
    """
    try:
        from scripts.aggregate_baseline_p0_matrix import (
            KNOWN_NON_SCIENTIFIC_IMPLEMENTATION_TRANSITIONS as registry,
            NON_SCIENTIFIC_TRANSITION_CLASSIFICATIONS)
    except Exception:                                      # noqa: BLE001
        return False
    entry = registry.get(SELF_SOURCE_RELATIVE)
    if not isinstance(entry, Mapping):
        return False

    # Review metadata is not decoration: `classification` is what the path
    # audit hands out for a reviewed transition, and `evidence` is the written
    # justification. An entry missing either was never reviewed, whatever its
    # digests say.
    classification = entry.get("classification")
    if classification not in NON_SCIENTIFIC_TRANSITION_CLASSIFICATIONS:
        return False
    evidence = entry.get("evidence")
    if not isinstance(evidence, str) or not evidence.strip():
        return False

    # This launcher trains nothing and scores nothing, so a transition of it is
    # either non-scientific for EVERY variant and bit budget or it does not
    # belong here at all. A scope key would claim otherwise, and the caller has
    # no variant/bit to check it against -- so refuse rather than ignore it.
    # `in`, not truthiness: an EMPTY scope map still makes the aggregator's
    # audit demand a per-digest entry and block every cell, so a launcher that
    # shrugged at `{}` would accept what the aggregator refuses -- the two
    # gates must not disagree (audit §255).
    for scope_key in ("non_scientific_variants_by_sha256",
                      "non_scientific_bits_by_sha256"):
        if scope_key in entry:
            return False

    reviewed = entry.get("reviewed_sha256") or ()
    return (current == entry.get("after_sha256")
            and recorded in reviewed and current in reviewed)


def _matches_canonical_source_profile(
        payload: Mapping[str, object], variant: str) -> bool:
    """Fail closed unless the manifest used the pinned method source."""
    profile = CANONICAL_VARIANT_SOURCE_PROFILES.get(variant)
    if profile is None:
        return False
    identity = payload.get("protocol_identity")
    if not isinstance(identity, Mapping):
        return False
    snapshot = identity.get("implementation_sha256")
    if not isinstance(snapshot, Mapping):
        return False
    source_path, expected_sha256 = profile
    if snapshot.get(source_path) != expected_sha256:
        return False
    required_paths = _required_implementation_paths(variant)
    if frozenset(snapshot) != required_paths:
        return False
    for path, raw_digest in snapshot.items():
        if not isinstance(path, str) or not isinstance(raw_digest, str):
            return False
        current = _current_source_sha256(path)
        if current == raw_digest:
            continue
        if path == SELF_SOURCE_RELATIVE:
            # This file cannot hold its own approved digest: writing the new
            # SHA into the map changes the SHA again, so no fixed point
            # exists. The aggregator is a separate file and is not part of the
            # hashed implementation set, so its transition registry is a
            # non-circular anchor. Import is function-local to keep the module
            # import graph acyclic.
            if not _self_transition_is_reviewed(raw_digest, current):
                return False
            continue
        aliases = KNOWN_NON_SCIENTIFIC_SOURCE_SHA_ALIASES.get(path)
        if aliases is None or current not in aliases or raw_digest not in aliases:
            return False
        scoped_aliases = KNOWN_NON_SCIENTIFIC_SOURCE_SHA_ALIAS_VARIANTS.get(path)
        if scoped_aliases is not None:
            allowed_variants = scoped_aliases.get(raw_digest)
            if allowed_variants is None or variant not in allowed_variants:
                return False
    return True


def _full_manifest_validation_passes(
        path: Path, variant: str, dataset: str, bit: int, seed: int,
        *, expected_protocol_mode: str | None = None) -> bool:
    """Reuse the aggregator's protocol/artifact validator for resume admission."""
    try:
        from scripts.aggregate_baseline_p0_matrix import (
            Key as AggregateKey,
            _validate_manifest,
        )
    except ModuleNotFoundError:
        from aggregate_baseline_p0_matrix import (  # type: ignore[no-redef]
            Key as AggregateKey,
            _validate_manifest,
        )
    if variant == "umrch":
        panel = "u2"
    elif variant in SUPERVISED_VARIANTS:
        panel = "supervised"
    else:
        panel = "u0"
    record = _validate_manifest(
        path, AggregateKey(panel, variant, dataset, bit, seed),
        verify_hashes=False,
        expected_protocol_mode=expected_protocol_mode)
    return str(record.get("status", "")).startswith("complete_")


def _matches_semantic_information_condition(
        payload: Mapping[str, object], variant: str, dataset: str) -> bool:
    """Require the declared U0/U2/S condition at both identity levels."""
    expected: dict[str, object]
    if variant == "umrch":
        expected = {
            "information_tier": "U2",
            "term_source_condition": "target-benchmark-taxonomy-byte-exact",
            "matched_official_taxonomy_dataset": dataset,
            "selection_verified": None,
        }
    elif variant in SUPERVISED_VARIANTS:
        expected = {
            "information_tier": "S",
            "term_source_condition": (
                "benchmark training labels and target class taxonomy"
            ),
            "matched_official_taxonomy_dataset": dataset,
            "selection_verified": None,
            "uses_training_labels_in_objective": True,
            "uses_training_labels_in_center_reassignment": True,
            "comparison_panel": "supervised",
        }
    elif variant in U0_VARIANTS:
        expected = {
            "information_tier": "U0",
            "term_source_condition": "visual-only",
            "matched_official_taxonomy_dataset": "",
            "selection_verified": None,
        }
    else:
        return False
    identity = payload.get("protocol_identity")
    if not isinstance(identity, Mapping):
        return False
    return bool(
        payload.get("information_tier") == expected["information_tier"]
        and (
            variant not in SUPERVISED_VARIANTS
            or payload.get("comparison_panel") == "supervised"
        )
        and payload.get("semantic_information_condition") == expected
        and identity.get("semantic_information_condition") == expected
    )


def _completed_keys(
        result_root: Path, *, expected_protocol_mode: str | None = None,
        cache_bindings: Mapping[tuple[str, str], Mapping[str, object]] | None = None,
        ) -> dict[tuple[str, str, int, int], list[Path]]:
    """Return only manifests that reached the mandatory bio-projection phase."""
    completed: dict[tuple[str, str, int, int], list[Path]] = {}
    if not result_root.exists():
        return completed
    for path in sorted(result_root.rglob("p0_run_manifest.json")):
        try:
            with path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
            bit = _manifest_bit(payload)
            variant = payload.get("variant")
            dataset = payload.get("dataset")
            seed = int(payload.get("seed"))
            bio_path = Path(str(payload.get("bio_projection_manifest", "")))
            if not bio_path.is_absolute():
                bio_path = (path.parent / bio_path).resolve()
            identity = payload.get("protocol_identity")
            expected_horizon = _expected_horizon(
                str(variant), str(dataset))
            actual_mode = payload.get("protocol_mode")
            if actual_mode is None:
                actual_mode = VALIDATION_SENSITIVITY
            author_fixed = actual_mode == AUTHOR_FIXED_FINAL
            requested_binding = (
                cache_bindings.get((str(variant), str(dataset)))
                if cache_bindings is not None else None)
            cache_matches = (
                True if requested_binding is None else bool(
                    isinstance(identity, Mapping)
                    and identity.get("cache_dir")
                    == requested_binding.get("cache_dir")
                    and identity.get("cache_meta_sha256")
                    == requested_binding.get("cache_meta_sha256")
                    and identity.get("cache_image_ids_sha256")
                    == requested_binding.get("cache_image_ids_sha256")
                    and identity.get("cache_artifact_sha256")
                    == requested_binding.get("cache_artifact_sha256")
                    and identity.get("cache_decode_failure_audit")
                    == requested_binding.get("cache_decode_failure_audit")
                    and payload.get("cache_decode_failure_audit")
                    == requested_binding.get("cache_decode_failure_audit")
                    and (
                        "dataset_root" not in requested_binding
                        or identity.get("dataset_root")
                        == requested_binding.get("dataset_root"))
                    and (
                        "split_sha256" not in requested_binding
                        or identity.get("split_sha256")
                        == requested_binding.get("split_sha256"))))
            complete = (
                isinstance(variant, str)
                and isinstance(dataset, str)
                and bit in BITS
                and isinstance(identity, Mapping)
                and isinstance(payload.get("protocol_digest_sha256"), str)
                and re.fullmatch(
                    r"[0-9a-f]{64}",
                    str(payload.get("protocol_digest_sha256"))) is not None
                and _protocol_identity_digest(identity)
                == payload.get("protocol_digest_sha256")
                and identity.get("variant") == variant
                and identity.get("dataset") == dataset
                and identity.get("bit") == bit
                and identity.get("seed") == seed
                and (expected_protocol_mode is None
                     or actual_mode == expected_protocol_mode)
                and identity.get("protocol_mode", VALIDATION_SENSITIVITY)
                == actual_mode
                and identity.get("val_seed") == (
                    None if author_fixed else 42)
                and identity.get("val_ratio") == (
                    0.0 if author_fixed else 0.1)
                and identity.get("eval_period") == (
                    expected_horizon if author_fixed else 5)
                and identity.get("horizon") == expected_horizon
                and identity.get("batch_size_override") is None
                and _matches_canonical_source_profile(payload, str(variant))
                and _matches_semantic_information_condition(
                    payload, str(variant), str(dataset))
                and cache_matches
                and payload.get("protocol_deviations") == []
                and payload.get("test_used_for_selection") is False
                and payload.get("final_checkpoint_count") == 1
                and payload.get("run_manifest_phase") == "bio_projection_completed"
                and payload.get("bio_projection_status")
                in {"paper_result_eligible", "completed_not_paper_eligible"}
                and bio_path.is_file()
                and payload.get("bio_projection_manifest_sha256") == _sha256(bio_path)
                and _full_manifest_validation_passes(
                    path, str(variant), str(dataset), int(bit), seed,
                    expected_protocol_mode=expected_protocol_mode)
            )
            if complete:
                key = (variant, dataset, int(bit), seed)
                completed.setdefault(key, []).append(path.resolve())
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            # The aggregator reports malformed artifacts in detail.  A launcher
            # must never treat them as completed or delete/overwrite them.
            continue
    return completed


def _jobs(panel: str, variants: Sequence[str], datasets: Sequence[str],
          bits: Sequence[int], seeds: Sequence[int]) -> list[Job]:
    jobs: list[Job] = []
    if panel in {"all", "all-plus-supervised", "u0"}:
        jobs.extend(
            Job("u0", variant, dataset, bit, seed)
            for variant in variants
            for dataset in datasets
            for bit in bits
            for seed in seeds
        )
    if panel in {"all", "all-plus-supervised", "u2"}:
        jobs.extend(
            Job("u2", "umrch", dataset, bit, seed)
            for dataset in datasets
            if dataset in U2_DATASETS
            for bit in bits
            for seed in seeds
        )
    if panel in {"all-plus-supervised", "supervised"}:
        jobs.extend(
            Job("supervised", "crh-supervised", dataset, bit, seed)
            for dataset in datasets
            if dataset in SUPERVISED_DATASETS
            for bit in bits
            for seed in seeds
        )
    return sorted(jobs, key=lambda job: (-job.scheduling_weight, job.job_id))


_ATTEMPT_RE = re.compile(r"^attempt_(\d{3,})$")


def _next_attempt(job: Job, roots: Iterable[Path], log_root: Path) -> int:
    numbers: list[int] = []
    for base in (*tuple(roots), log_root / "status"):
        parent = base / job.job_id
        if not parent.exists():
            continue
        for child in parent.iterdir():
            match = _ATTEMPT_RE.match(child.name)
            if match:
                numbers.append(int(match.group(1)))
        if base.name == "status":
            for path in parent.glob("attempt_*.json"):
                match = re.match(r"^attempt_(\d{3,})\.json$", path.name)
                if match:
                    numbers.append(int(match.group(1)))
    log_parent = log_root / job.job_id
    if log_parent.exists():
        for path in log_parent.glob("attempt_*.log"):
            match = re.match(r"^attempt_(\d{3,})\.log$", path.name)
            if match:
                numbers.append(int(match.group(1)))
    return max(numbers, default=0) + 1


def _make_attempt(job: Job, number: int, *, model_root: Path,
                  result_root: Path, compress_root: Path,
                  log_root: Path) -> Attempt:
    name = f"attempt_{number:03d}"
    return Attempt(
        job=job,
        number=number,
        model_root=model_root / job.job_id / name,
        result_root=result_root / job.job_id / name,
        compress_root=compress_root / job.job_id / name,
        log_path=log_root / job.job_id / f"{name}.log",
        status_path=log_root / "status" / job.job_id / f"{name}.json",
    )


def _command(attempt: Attempt, *, python: Path,
             gpu_assignment: Mapping[str, object],
             num_workers: int,
             protocol_mode: str = AUTHOR_FIXED_FINAL,
             cache_dirs: Mapping[str, Path] | None = None,
             allow_main_ineligible_smoke: bool = False) -> list[str]:
    job = attempt.job
    command = [
        str(python), str(RUNNER),
        "--protocol-mode", protocol_mode,
        "--variant", job.variant,
        "--dataset", job.dataset,
        "--bit", str(job.bit),
        "--seed", str(job.seed),
        "--device", "cuda:0",
        "--num-workers", str(num_workers),
        "--model-root", str(attempt.model_root),
        "--result-root", str(attempt.result_root),
        "--compress-root", str(attempt.compress_root),
        "--matrix-assigned-gpu-json", json.dumps(
            validate_gpu_assignment(gpu_assignment), sort_keys=True,
            separators=(",", ":"), allow_nan=False),
    ]
    if cache_dirs is not None and job.dataset in cache_dirs:
        command += ["--cache-dir", str(cache_dirs[job.dataset])]
    if allow_main_ineligible_smoke:
        command.append("--allow-main-ineligible-smoke")
    if job.variant == "umrch":
        embeddings, adapter, manifest = UMRCH_ASSETS[job.dataset]
        command.extend([
            "--umrch-concept-embeddings", str((REPO / embeddings).resolve()),
            "--umrch-vision-adapter", str((REPO / adapter).resolve()),
            "--umrch-asset-manifest", str((REPO / manifest).resolve()),
        ])
    return command


def _validate_inputs(
        python: Path, jobs: Sequence[Job], *, protocol_mode: str,
        cache_dirs: Mapping[str, Path], allow_main_ineligible_smoke: bool,
        ) -> dict[tuple[str, str], dict[str, object]]:
    if not python.is_file() or not os.access(python, os.X_OK):
        raise FileNotFoundError(f"Python interpreter is not executable: {python}")
    if not RUNNER.is_file():
        raise FileNotFoundError(f"P0 runner is missing: {RUNNER}")
    if protocol_mode == AUTHOR_FIXED_FINAL:
        scheduled_datasets = {job.dataset for job in jobs}
        missing_cache_declarations = sorted(scheduled_datasets - set(cache_dirs))
        if missing_cache_declarations:
            raise ValueError(
                "D6 author_fixed_final requires explicit eligible --cache-dir "
                "for every scheduled dataset; missing="
                f"{missing_cache_declarations}")
        unexpected_cache_declarations = sorted(set(cache_dirs) - scheduled_datasets)
        if unexpected_cache_declarations:
            raise ValueError(
                "D6 cache map contains datasets outside the scheduled matrix: "
                f"{unexpected_cache_declarations}")
    if any(job.variant == "umrch" for job in jobs):
        selected = {job.dataset for job in jobs if job.variant == "umrch"}
        missing = [
            str((REPO / path).resolve())
            for dataset, values in UMRCH_ASSETS.items()
            if dataset in selected
            for path in values
            if not (REPO / path).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "UMRCH U2 assets are missing:\n  " + "\n  ".join(missing))

    bindings: dict[tuple[str, str], dict[str, object]] = {}
    if cache_dirs:
        try:
            from scripts.run_modern_baseline_p0 import (
                VARIANTS, _check_cache, _dataset_source_binding)
        except ModuleNotFoundError:
            from run_modern_baseline_p0 import (  # type: ignore[no-redef]
                VARIANTS, _check_cache, _dataset_source_binding)
        for job in jobs:
            binding_key = (job.variant, job.dataset)
            if binding_key in bindings:
                continue
            cache_dir = cache_dirs.get(job.dataset)
            if cache_dir is None:
                continue
            specification = VARIANTS[job.variant]
            dataset_root, split_sha256 = _dataset_source_binding(
                job.dataset, "setting1", REPO / "dataset")
            audit = _check_cache(
                cache_dir, dataset=job.dataset,
                needs_aug=bool(specification["needs_aug"]),
                needs_tokens=bool(specification["needs_tokens"]),
                dataset_root=REPO / "dataset", setting="setting1",
            )
            blockers = list(audit["eligibility_blockers"])
            if blockers and not allow_main_ineligible_smoke:
                raise ValueError(
                    f"cache {cache_dir} is not paper eligible for "
                    f"{job.variant}/{job.dataset}: {blockers}")
            bindings[binding_key] = {
                "dataset_root": dataset_root,
                "split_sha256": split_sha256,
                "cache_dir": str(cache_dir),
                "cache_meta_sha256": _sha256(cache_dir / "meta.json"),
                "cache_image_ids_sha256": _sha256(
                    cache_dir / "image_ids.json"),
                "cache_artifact_sha256": audit["artifact_hashes"],
                "cache_decode_failure_audit": audit["decode_failure_audit"],
            }
    return bindings


def _plan_record(attempt: Attempt, command: Sequence[str], *,
                 gpu_assignment: Mapping[str, object],
                 protocol_mode: str, comparison_label: str) -> dict[str, object]:
    job = attempt.job
    return {
        "job_id": job.job_id,
        "panel": job.panel,
        "variant": job.variant,
        "dataset": job.dataset,
        "bit": job.bit,
        "base_length": job.bit // 2,
        "seed": job.seed,
        "attempt": attempt.number,
        "model_root": str(attempt.model_root),
        "result_root": str(attempt.result_root),
        "compress_root": str(attempt.compress_root),
        "log": str(attempt.log_path),
        "command_without_gpu_environment": list(command),
        "gpu_assignment": validate_gpu_assignment(gpu_assignment),
        "cuda_visible_devices": validate_gpu_assignment(
            gpu_assignment)["uuid"],
        "protocol_mode": protocol_mode,
        "comparison_label": comparison_label,
    }


def _status_payload(attempt: Attempt, command: Sequence[str],
                    gpu_assignment: Mapping[str, object],
                    *, state: str, started_at: str,
                    protocol_mode: str = AUTHOR_FIXED_FINAL,
                    comparison_label: str = AUTHOR_LABEL,
                    returncode: int | None = None,
                    message: str | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "state": state,
        "protocol_mode": protocol_mode,
        "comparison_label": comparison_label,
        "job_id": attempt.job.job_id,
        "panel": attempt.job.panel,
        "variant": attempt.job.variant,
        "dataset": attempt.job.dataset,
        "bit": attempt.job.bit,
        "seed": attempt.job.seed,
        "attempt": attempt.number,
        "physical_gpu": validate_gpu_assignment(gpu_assignment),
        "cuda_visible_devices": validate_gpu_assignment(
            gpu_assignment)["uuid"],
        "logical_device": "cuda:0",
        "started_at_utc": started_at,
        "updated_at_utc": _utc_now(),
        "command": list(command),
        "log": str(attempt.log_path),
        "returncode": returncode,
    }
    if message is not None:
        payload["message"] = message
    return payload


def _print_dry_run(attempts: Sequence[Attempt], *, python: Path,
                   gpus: Sequence[str],
                   gpu_assignments: Mapping[str, Mapping[str, object]],
                   num_workers: int,
                   skipped: Mapping[tuple[str, str, int, int], list[Path]],
                   protocol_mode: str, comparison_label: str,
                   cache_dirs: Mapping[str, Path],
                   allow_main_ineligible_smoke: bool) -> None:
    print(f"comparison_label={comparison_label}")
    print(f"protocol_mode={protocol_mode}")
    print(f"scheduled={len(attempts)} completed_skipped={len(skipped)}")
    print(f"gpu_pool={','.join(gpus)}")
    print(f"exact_duheg={DUHEG_BLOCK['status']}: {DUHEG_BLOCK['reason']}")
    for index, attempt in enumerate(attempts):
        gpu = gpus[index % len(gpus)]
        assignment = gpu_assignments[gpu]
        command = _command(
            attempt, python=python, gpu_assignment=assignment,
            num_workers=num_workers,
            protocol_mode=protocol_mode, cache_dirs=cache_dirs,
            allow_main_ineligible_smoke=allow_main_ineligible_smoke)
        print(
            f"CUDA_VISIBLE_DEVICES={shlex.quote(str(assignment['uuid']))} "
            + shlex.join(command))


def _publish_matrix_plan(
        *, attempts: Sequence[Attempt], args: argparse.Namespace,
        python: Path,
        gpu_assignments: Mapping[str, Mapping[str, object]],
        cache_dirs: Mapping[str, Path], comparison_label: str,
        log_root: Path, gpu_leases: GpuLeaseSet,
        cache_bindings: Mapping[tuple[str, str], Mapping[str, object]],
        completed: Mapping[tuple[str, str, int, int], list[Path]],
        ) -> tuple[str, Path]:
    """Durably publish the exact launch plan after leases are acquired."""
    created = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    plan_path = log_root / f"matrix_plan_{created}_{os.getpid()}.json"
    plan_jobs = []
    for index, attempt in enumerate(attempts):
        gpu = args.gpus[index % len(args.gpus)]
        assignment = gpu_assignments[gpu]
        plan_jobs.append(_plan_record(
            attempt,
            _command(
                attempt, python=python, gpu_assignment=assignment,
                num_workers=args.num_workers,
                protocol_mode=args.protocol_mode,
                cache_dirs=cache_dirs,
                allow_main_ineligible_smoke=bool(
                    args.allow_main_ineligible_smoke)),
            gpu_assignment=assignment,
            protocol_mode=args.protocol_mode,
            comparison_label=comparison_label))
    _atomic_json(plan_path, {
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "protocol_mode": args.protocol_mode,
        "comparison_label": comparison_label,
        "warning": (
            "Diagnostic smoke/validation-sensitivity cells are never paper-main "
            "eligible. D6 candidates still require every downstream manifest gate."
        ),
        "runner": str(RUNNER),
        "runner_sha256": _sha256(RUNNER),
        "train_seeds": list(args.seeds),
        "three_seed_mean_std_status": (
            "pending" if len(args.seeds) < 3 else "candidate_runs_scheduled"
        ),
        "val_seed": (
            None if args.protocol_mode == AUTHOR_FIXED_FINAL else 42),
        "validation_selection": args.protocol_mode != AUTHOR_FIXED_FINAL,
        "checkpoint_policy": (
            "author_horizon_last"
            if args.protocol_mode == AUTHOR_FIXED_FINAL
            else "validation_selected_scratch_refit"),
        "cache_dirs": {
            dataset: str(path) for dataset, path in sorted(cache_dirs.items())
        },
        "cache_bindings": {
            "|".join(key): value
            for key, value in sorted(cache_bindings.items())
        },
        "bits": list(args.bits),
        "datasets": list(args.datasets),
        "gpu_pool": list(args.gpus),
        "gpu_assignments": {
            gpu: gpu_assignments[gpu] for gpu in args.gpus
        },
        "host_global_gpu_lease_paths": [
            str(path) for path in gpu_leases.paths
        ],
        "exact_duheg": DUHEG_BLOCK,
        "completed_cells_skipped": {
            "|".join(map(str, key)): [str(path) for path in paths]
            for key, paths in sorted(completed.items())
        },
        "jobs": plan_jobs,
    })
    return created, plan_path


def main() -> int:
    default_python = Path("/home/yschoi/.conda/envs/dna_hashing/bin/python")
    if not default_python.is_file():
        default_python = Path(sys.executable)

    seeds_environment = _environment_tokens("SEEDS", "42")
    try:
        default_seeds = [int(value) for value in seeds_environment]
    except ValueError as error:
        raise ValueError(
            "SEEDS must be a comma- or space-separated integer list") from error
    if not default_seeds:
        default_seeds = list(DEFAULT_SEEDS)
    default_panel, default_variants = _method_group_from_environment()
    default_datasets = _environment_tokens(
        "DATASETS", "Flickr25k MSCOCO NUSWIDE CIFAR10")
    invalid_datasets = sorted(set(default_datasets) - set(DATASETS))
    if invalid_datasets:
        raise ValueError(f"DATASETS contains invalid values: {invalid_datasets}")
    try:
        default_bits = [int(value) for value in _environment_tokens("BITS", "36 48")]
    except ValueError as error:
        raise ValueError("BITS must contain only 36 and/or 48") from error
    invalid_bits = sorted(set(default_bits) - set(BITS))
    if invalid_bits:
        raise ValueError(f"BITS contains invalid values: {invalid_bits}")

    parser = argparse.ArgumentParser(
        description="Launch the D6 author-fixed baseline matrix.")
    parser.add_argument(
        "--protocol-mode", choices=PROTOCOL_MODES,
        default=AUTHOR_FIXED_FINAL,
        help=(
            "author_fixed_final is the D6 paper-main path; "
            "validation_sensitivity is diagnostic selector/refit only."),
    )
    parser.add_argument("--gpus", nargs="+", default=["0", "1", "2", "3", "4", "5"],
                        help="Physical GPU IDs; each worker exposes one as cuda:0.")
    parser.add_argument(
        "--panel",
        choices=("all", "all-plus-supervised", "u0", "u2", "supervised"),
        default=default_panel,
        help=(
            "Defaults from METHOD_GROUP. Historical `all` remains U0+U2 only; "
            "use `supervised` for CRH or `all-plus-supervised` explicitly."
        ))
    parser.add_argument("--variants", nargs="+", choices=U0_VARIANTS,
                        default=default_variants,
                        help=("U0 variants only; ignored by U2/supervised panels. "
                              "Defaults from METHOD_GROUP and leaves source method "
                              "defaults untouched."))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS,
                        default=default_datasets,
                        help="Defaults from DATASETS environment.")
    parser.add_argument("--bits", nargs="+", type=int, choices=BITS,
                        default=None,
                        help=("D6 requires an explicit `--bits 30`. The BITS "
                              "environment fallback is retained only for "
                              "validation_sensitivity."))
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=default_seeds,
        help="Training seeds; defaults to SEEDS environment or champion-matched 42.")
    parser.add_argument("--python", default=str(default_python))
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--model-root", default=None)
    parser.add_argument("--result-root", default=None)
    parser.add_argument("--compress-root", default=None)
    parser.add_argument("--log-root", default=None)
    parser.add_argument(
        "--cache-dir", action="append", default=[], metavar="DATASET=PATH",
        help=(
            "Explicit per-dataset feature cache. Repeat once for every "
            "scheduled dataset; mandatory and preflight-audited for D6."),
    )
    parser.add_argument(
        "--allow-main-ineligible-smoke", action="store_true",
        help=(
            "Explicit diagnostic opt-in. Never added automatically; changes "
            "the matrix label and prevents paper-main admission."),
    )
    parser.add_argument(
        "--resume-root", action="append", default=[],
        help=("Additional result root to scan for protocol-valid completed "
              "cells before scheduling; repeat for multiple external queues."))
    parser.add_argument("--no-resume", action="store_true",
                        help="Do not skip cells with a verified completed manifest.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands only; create no files and launch nothing.")
    args = parser.parse_args()

    if args.bits is None:
        if args.protocol_mode == AUTHOR_FIXED_FINAL:
            parser.error("D6 author_fixed_final requires explicit --bits 30")
        args.bits = default_bits
    if (args.protocol_mode == AUTHOR_FIXED_FINAL
            and tuple(args.bits) != (30,)):
        parser.error(
            "D6 author_fixed_final accepts exactly --bits 30; non-main budgets "
            "belong to validation_sensitivity")

    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    if not args.gpus or len(set(args.gpus)) != len(args.gpus):
        parser.error("--gpus must contain at least one unique GPU ID")
    try:
        gpu_assignments = resolve_gpu_assignments(args.gpus)
    except ExecutionEnvironmentError as error:
        parser.error(str(error))
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must contain at least one unique integer")

    python = _absolute(args.python)
    try:
        cache_dirs = _parse_cache_dirs(args.cache_dir)
    except ValueError as error:
        parser.error(str(error))
    comparison_label = _comparison_label(
        args.protocol_mode,
        allow_smoke=bool(args.allow_main_ineligible_smoke))
    seed_slug = "seeds" + "-".join(map(str, args.seeds))
    mode_slug = (
        "author_fixed_30b"
        if args.protocol_mode == AUTHOR_FIXED_FINAL
        else "validation_sensitivity_legacy")
    matrix_name = (
        f"p0_matrix_{seed_slug}_{args.panel}_{mode_slug}"
        if args.panel in {"supervised", "all-plus-supervised"}
        else f"p0_matrix_{seed_slug}_{mode_slug}"
    )
    model_root = _absolute(
        args.model_root or f"params_baseline/{matrix_name}")
    result_root = _absolute(
        args.result_root or f"result_baseline/{matrix_name}")
    compress_root = _absolute(
        args.compress_root or f"compress_baseline/{matrix_name}")
    log_root = _absolute(
        args.log_root or f"logs/{matrix_name}")
    jobs = _jobs(
        args.panel, args.variants, args.datasets, args.bits, args.seeds)
    cache_bindings = _validate_inputs(
        python, jobs, protocol_mode=args.protocol_mode,
        cache_dirs=cache_dirs,
        allow_main_ineligible_smoke=bool(args.allow_main_ineligible_smoke))

    completed: dict[tuple[str, str, int, int], list[Path]] = {}
    if not args.no_resume:
        for resume_root in (result_root, *map(_absolute, args.resume_root)):
            for key, paths in _completed_keys(
                    resume_root,
                    expected_protocol_mode=args.protocol_mode,
                    cache_bindings=cache_bindings).items():
                bucket = completed.setdefault(key, [])
                for path in paths:
                    if path not in bucket:
                        bucket.append(path)
    pending = [job for job in jobs if job.key not in completed]
    attempts = [
        _make_attempt(
            job,
            _next_attempt(
                job, (model_root, result_root, compress_root), log_root),
            model_root=model_root, result_root=result_root,
            compress_root=compress_root, log_root=log_root,
        )
        for job in pending
    ]

    if args.dry_run:
        _print_dry_run(
            attempts, python=python, gpus=args.gpus,
            gpu_assignments=gpu_assignments,
            num_workers=args.num_workers, skipped=completed,
            protocol_mode=args.protocol_mode,
            comparison_label=comparison_label, cache_dirs=cache_dirs,
            allow_main_ineligible_smoke=bool(
                args.allow_main_ineligible_smoke))
        return 0

    log_root.mkdir(parents=True, exist_ok=True)
    lock_path = log_root / "launcher.lock"
    lock_handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock_handle.close()
        raise RuntimeError(
            f"another matrix launcher holds {lock_path}; refusing overlap") from error
    try:
        gpu_leases = (
            acquire_gpu_leases(
                [gpu_assignments[gpu] for gpu in args.gpus],
                owner_metadata={
                    "campaign": "baseline_d6",
                    "protocol_mode": args.protocol_mode,
                    "repo": str(REPO),
                    "log_root": str(log_root),
                })
            if attempts else GpuLeaseSet(handles=[], paths=()))
    except BaseException:
        lock_handle.close()
        raise

    try:
        created, plan_path = _publish_matrix_plan(
            attempts=attempts, args=args, python=python,
            gpu_assignments=gpu_assignments, cache_dirs=cache_dirs,
            comparison_label=comparison_label, log_root=log_root,
            gpu_leases=gpu_leases, cache_bindings=cache_bindings,
            completed=completed)
        print(
            f"[matrix] mode={args.protocol_mode} label={comparison_label}",
            flush=True)
        print(
            f"[matrix] plan={plan_path} pending={len(attempts)} "
            f"completed_skipped={len(completed)} gpus={','.join(args.gpus)}",
            flush=True)
        print(
            f"[matrix] exact DUH-EG: {DUHEG_BLOCK['status']} "
            f"({DUHEG_BLOCK['reason']})", flush=True)
    except BaseException:
        try:
            gpu_leases.release()
        finally:
            lock_handle.close()
        raise

    if not attempts:
        print("[matrix] no pending cells", flush=True)
        gpu_leases.release()
        lock_handle.close()
        return 0

    work: queue.Queue[Attempt] = queue.Queue()
    for attempt in attempts:
        work.put(attempt)
    output_lock = threading.Lock()
    failures: list[tuple[str, int]] = []
    try:
        lifecycle = _CampaignLifecycle([
            int(handle.fileno()) for handle in gpu_leases.handles
        ])
        lifecycle.install_signal_handlers()
    except BaseException:
        try:
            gpu_leases.release()
        finally:
            lock_handle.close()
        raise

    def worker(gpu_id: str) -> None:
        gpu_assignment = gpu_assignments[gpu_id]
        while not lifecycle.stop.is_set():
            try:
                attempt = work.get_nowait()
            except queue.Empty:
                return
            command = _command(
                attempt, python=python, gpu_assignment=gpu_assignment,
                num_workers=args.num_workers,
                protocol_mode=args.protocol_mode, cache_dirs=cache_dirs,
                allow_main_ineligible_smoke=bool(
                    args.allow_main_ineligible_smoke))
            started = _utc_now()
            attempt.log_path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_json(
                attempt.status_path,
                _status_payload(
                    attempt, command, gpu_assignment, state="running",
                    started_at=started, protocol_mode=args.protocol_mode,
                    comparison_label=comparison_label))
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(
                gpu_assignment["uuid"])
            environment["PYTHONUNBUFFERED"] = "1"
            with output_lock:
                print(
                    f"[matrix][gpu {gpu_id}] START {attempt.job.job_id} "
                    f"attempt={attempt.number:03d} log={attempt.log_path}",
                    flush=True)
            returncode = 1
            message: str | None = None
            try:
                with attempt.log_path.open("x", encoding="utf-8") as log_handle:
                    log_handle.write(
                        f"# protocol_mode={args.protocol_mode}\n"
                        f"# comparison_label={comparison_label}\n"
                        f"# physical_gpu={json.dumps(gpu_assignment, sort_keys=True)} "
                        f"logical_device=cuda:0\n"
                        f"# command={shlex.join(command)}\n")
                    log_handle.flush()
                    process = lifecycle.children.spawn(
                        command, cwd=REPO, env=environment,
                        stdout=log_handle, stderr=subprocess.STDOUT)
                    returncode = lifecycle.children.wait(process)
                if returncode == 0:
                    completed_after_run = _completed_keys(
                        attempt.result_root,
                        expected_protocol_mode=args.protocol_mode,
                        cache_bindings=cache_bindings)
                    if attempt.job.key not in completed_after_run:
                        returncode = 3
                        message = (
                            "runner exited zero but no protocol-valid completed "
                            "p0_run_manifest + bio_projection pair was found")
            except Exception as error:  # preserve a durable failure record
                message = f"{type(error).__name__}: {error}"
                returncode = 1
            state = "completed" if returncode == 0 else "failed"
            _atomic_json(
                attempt.status_path,
                _status_payload(
                    attempt, command, gpu_assignment, state=state,
                    started_at=started, returncode=returncode,
                    message=message, protocol_mode=args.protocol_mode,
                    comparison_label=comparison_label))
            if returncode != 0:
                failures.append((attempt.job.job_id, returncode))
            with output_lock:
                print(
                    f"[matrix][gpu {gpu_id}] {state.upper()} "
                    f"{attempt.job.job_id} rc={returncode}", flush=True)
            work.task_done()

    try:
        threads = [
            threading.Thread(target=worker, args=(gpu,), name=f"gpu-{gpu}")
            for gpu in args.gpus
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        summary_path = log_root / f"matrix_summary_{created}_{os.getpid()}.json"
        _atomic_json(summary_path, {
            "schema_version": 1,
            "finished_at_utc": _utc_now(),
            "protocol_mode": args.protocol_mode,
            "comparison_label": comparison_label,
            "plan": str(plan_path),
            "scheduled": len(attempts),
            "failures": [
                {"job_id": job_id, "returncode": returncode}
                for job_id, returncode in failures
            ],
            "interrupted": lifecycle.interrupted_signum is not None,
            "interrupt_signal": lifecycle.interrupted_signum,
            "exact_duheg": DUHEG_BLOCK,
        })
        print(
            f"[matrix] finished failures={len(failures)} summary={summary_path}",
            flush=True)
        if lifecycle.interrupted_signum is not None:
            return 128 + lifecycle.interrupted_signum
        return 1 if failures else 0
    finally:
        # No lease may be released while a managed runner or one of its
        # process-group descendants can still execute GPU work.
        cleanup_succeeded = False
        try:
            lifecycle.close()
            cleanup_succeeded = True
        finally:
            try:
                if cleanup_succeeded:
                    gpu_leases.release()
            finally:
                lock_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
