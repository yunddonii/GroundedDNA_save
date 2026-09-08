"""Immutable lifecycle ledger for the exact 24-cell Phase-5 A campaign.

This ledger is deliberately *not* a paper-metric authority. It proves that
every planned trainer finished and that every output currently passes the
repository's RunIdentity, extraction-transaction, BIO-evaluation, NMI and
analysis-integrity checks. ``seal_cell_analysis.py`` labels its marker
non-authoritative because it does not independently recompute retrieval from
the arrays; the campaign receipt preserves that boundary.

The reservation owns one immutable plan snapshot and one random campaign
token. Every cell record and executor outcome is bound to both. Successful
records are create-once, carry exact artifact digests, and are reopened before
the create-once completion receipt is published. A failed generation is never
"repaired" by overwriting records: use a new ledger and campaign token.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import stat
import sys
import tempfile
from typing import Mapping

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from dna_utils.extraction_validation import (  # noqa: E402
    ExpectedIdentity,
    ExtractionInvalid,
    read_analysis_marker,
)
from dna_utils.run_identity import RunIdentity, load_run_manifest  # noqa: E402
from dna_utils.runtime_state import CheckpointMetadata  # noqa: E402
from scripts._ablation_exec import (  # noqa: E402
    CANONICAL_RUNNER,
    PlanRejected,
    _gpu_assignment,
    load_plan,
)


SCHEMA_VERSION = 2
RECEIPT_NAME = "campaign_complete.json"
RESERVATION_NAME = "campaign_reservation.json"
PLAN_SNAPSHOT_NAME = "campaign_plan.json"
STATUSES = ("ok", "failed")
EXPECTED_CELLS = 24

_TOKEN_RE = re.compile(r"^[0-9a-f]{64}$")
_TAG_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_EXPECTED_EXPERIMENTS = (
    "cifar_A_v4", "flickr_A_v4", "nuswide_A_v4", "mscoco_A_v5b")
_EXPECTED_ARMS = (
    "A2_no_text", "A4_shared_codebook", "A5_none", "A5_joint",
    "A5_nogumbel", "A5_both")


class CampaignRefused(RuntimeError):
    """The campaign cannot be opened, recorded, sealed, or verified."""


# Roots that can execute, decide the model/recipe, or admit completed analysis.
# Their first-party Python imports are closed transitively below.  A hand-kept
# flat list missed dataloaders, model components and bio policy modules; those
# bytes could change mid-campaign without changing the reservation digest.
PYTHON_SOURCE_ROOTS = (
    "scripts/_ablation_campaign_launch.py", "scripts/_ablation_exec.py",
    "scripts/_ablation_group_guard.py",
    "scripts/campaign_ledger.py", "scripts/ablation_campaign_plan.py",
    "scripts/eval_cell_bioproj.py", "scripts/pairwise_nmi.py",
    "scripts/seal_cell_analysis.py", "train_siglip2.py",
    "extraction_siglip2.py", "evaluation_siglip2.py", "model_siglip2.py",
    "loss_siglip2.py", "config.py",
)


def _resolve_local_module(name: str) -> Path | None:
    stem = REPO.joinpath(*name.split("."))
    module = stem.with_suffix(".py")
    if module.is_file():
        return module
    package = stem / "__init__.py"
    return package if package.is_file() else None


def _package_initializers(path: Path) -> tuple[Path, ...]:
    """Return package code Python executes before importing ``path``."""
    initializers = []
    parent = path.parent
    while parent != REPO:
        try:
            parent.relative_to(REPO)
        except ValueError:
            break
        initializer = parent / "__init__.py"
        if initializer.is_file():
            initializers.append(initializer)
        parent = parent.parent
    return tuple(initializers)


def _python_source_closure(roots=PYTHON_SOURCE_ROOTS) -> tuple[str, ...]:
    pending = [REPO / relative for relative in roots]
    seen: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        if not path.is_file() or path.is_symlink():
            raise CampaignRefused(
                f"Python source root/import is missing or symlinked: {path}")
        seen.add(path)
        # ``import pkg.module`` executes every ancestor package initializer
        # before the leaf.  Those bytes are execution authority too.
        for initializer in _package_initializers(path):
            if initializer not in seen:
                pending.append(initializer)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError, UnicodeError) as error:
            raise CampaignRefused(f"cannot parse campaign source {path}: {error}") \
                from None
        relative = path.relative_to(REPO)
        package = list(relative.parts[:-1])
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                base = package[:]
                if node.level:
                    trim = node.level - 1
                    base = base[:len(base) - trim] if trim else base
                else:
                    base = []
                module_parts = (node.module or "").split(".") \
                    if node.module else []
                prefix = ".".join(base + module_parts)
                if prefix:
                    names.append(prefix)
                    names.extend(
                        f"{prefix}.{alias.name}" for alias in node.names
                        if alias.name != "*")
            for name in names:
                imported = _resolve_local_module(name)
                if imported is not None and imported not in seen:
                    pending.append(imported)
    return tuple(sorted(str(path.relative_to(REPO)) for path in seen))


NON_PYTHON_SOURCE_PATHS = (
    "scripts/run_ablation_campaign.sh",
    "scripts/auto_chain_after_3seed.sh",
    "scripts/prompt_ablation_A_cell_fixedN.sh",
    "scripts/lib/result_dir.sh",
    "scripts/train_cifar10_v185_bidirTokenPrune05_ccs01_clip.sh",
    "scripts/train_flickr25k_v185_bidirTokenPrune05_clip.sh",
    "scripts/train_nuswide_v185_sweep_clip.sh",
    "scripts/train_mscoco_F2_sweep_clip.sh",
    "docs/newmodel_analysis/fixed_N.json",
    "artifacts/phase3_selection/selected_n.json",
)
SOURCE_PATHS = tuple(sorted(set(
    _python_source_closure()) | set(NON_PYTHON_SOURCE_PATHS)))


def _physical_gpu_assignment(index: int) -> dict:
    """Read the physical index/UUID tuple through the sealed system tool."""
    return _gpu_assignment(str(index), executable="/usr/bin/nvidia-smi")


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_bytes_exclusive(path: Path, raw: bytes, *, mode: int = 0o444) -> None:
    """Publish complete bytes atomically and refuse an existing final name."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise CampaignRefused(
                f"{path} already exists; campaign authority is create-once") \
                from error
        _fsync_dir(path.parent)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _publish_json_exclusive(path: Path, payload: Mapping) -> None:
    try:
        raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
               + "\n").encode("utf-8")
    except (TypeError, ValueError) as error:
        raise CampaignRefused(f"cannot encode {path.name}: {error}") from None
    _publish_bytes_exclusive(path, raw)


def _read_regular_json(path: Path, *, what: str) -> tuple[dict, str]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise CampaignRefused(f"cannot open {what} {path}: {error}") from None
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise CampaignRefused(f"{what} is not a regular file: {path}")
        chunks = []
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current = os.stat(path, follow_symlinks=False)
    fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns",
              "st_ctime_ns")
    if stat.S_ISLNK(current.st_mode) or any(
            getattr(before, key) != getattr(after, key)
            or getattr(after, key) != getattr(current, key)
            for key in fields):
        raise CampaignRefused(f"{what} changed or was replaced while read")
    raw = b"".join(chunks)
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value}")))
    except (UnicodeDecodeError, ValueError) as error:
        raise CampaignRefused(f"invalid {what} {path}: {error}") from None
    if not isinstance(payload, dict):
        raise CampaignRefused(f"{what} is not a JSON object: {path}")
    return payload, _sha_bytes(raw)


def source_digests(repo_root: Path = REPO) -> dict[str, str]:
    root = Path(repo_root).resolve()
    if root != REPO:
        raise CampaignRefused(
            f"campaign repo {root} is not the launcher source root {REPO}")
    out = {}
    for relative in SOURCE_PATHS:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise CampaignRefused(
                f"required campaign source is missing or symlinked: {path}")
        out[relative] = _sha_file(path)
    return out


def _assert_sources_current(held: Mapping) -> None:
    current = source_digests(Path(str(held.get("repo_root") or "")))
    if held.get("source_sha256") != current:
        changed = sorted(
            key for key in set(current) | set(held.get("source_sha256") or {})
            if current.get(key) != (held.get("source_sha256") or {}).get(key))
        raise CampaignRefused(
            f"campaign source changed after reservation: {changed}")


def _safe_ledger_dir(path: Path, *, create: bool = False) -> Path:
    """Return one owner-only, canonical ledger directory.

    Read-only receipt files inside a writable directory are not immutable: an
    unlink/recreate replaces them.  Requiring the directory to be owned by this
    uid and non-writable by group/other closes the cross-user version of that
    overwrite and lets the reservation bind its inode across CLI processes.
    """
    raw = Path(path)
    if create:
        try:
            raw.mkdir(mode=0o700, parents=True, exist_ok=False)
        except FileExistsError:
            pass
    if raw.is_symlink():
        raise CampaignRefused(f"ledger directory may not be a symlink: {raw}")
    try:
        resolved = raw.resolve(strict=True)
        info = resolved.lstat()
    except OSError as error:
        raise CampaignRefused(f"cannot resolve ledger directory {raw}: {error}") \
            from None
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() \
            or info.st_mode & 0o022:
        raise CampaignRefused(
            f"ledger directory must be owner-controlled and non-writable by "
            f"group/other: {resolved}")
    return resolved


def _process_identity(pid: int) -> dict:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1:
        raise CampaignRefused(f"invalid launcher owner pid {pid!r}")
    path = Path(f"/proc/{pid}/stat")
    try:
        fields = path.read_text(encoding="utf-8").split()
        uid = path.stat().st_uid
    except OSError as error:
        raise CampaignRefused(f"launcher pid {pid} is not live: {error}") \
            from None
    if len(fields) < 22 or uid != os.getuid() or fields[2] == "Z":
        raise CampaignRefused(f"launcher pid {pid} has no safe live identity")
    return {"pid": pid, "uid": uid, "start_ticks": int(fields[21])}


def _fresh_canonical_plan() -> dict:
    from scripts.ablation_campaign_plan import PlanRefused, build_plan
    try:
        return build_plan()
    except PlanRefused as error:
        raise CampaignRefused(
            f"cannot regenerate canonical Phase-5 plan authority: {error}") \
            from None


def _assert_canonical_recipe(plan: Mapping) -> None:
    """Compare every science-bearing plan field with the source-owned planner."""
    expected = _fresh_canonical_plan()
    authority_keys = (
        "schema_version", "expected_cells", "selected_n", "selection_sha256",
        "env_passthrough", "env_passthrough_values", "env_pinned", "runners",
        "cells")
    differing = [key for key in authority_keys
                 if plan.get(key) != expected.get(key)]
    values = plan.get("env_passthrough_values")
    allowed = set(plan.get("env_passthrough") or ())
    if not isinstance(values, dict) or set(values) - allowed:
        differing.append("env_passthrough_values.keys")
    elif Path(str(values.get("PY") or "")).resolve() \
            != Path(sys.executable).resolve():
        differing.append("env_passthrough_values.PY")
    if differing:
        raise CampaignRefused(
            "plan differs from the freshly regenerated canonical Phase-5 "
            f"recipe authority: {sorted(set(differing))}")


def _canonical_cells(plan: Mapping, *, require_recipe: bool = False) -> list[dict]:
    cells = plan.get("cells")
    if not isinstance(cells, list) or len(cells) != EXPECTED_CELLS \
            or plan.get("expected_cells") != EXPECTED_CELLS:
        raise CampaignRefused(
            f"Phase-5 campaign requires exactly {EXPECTED_CELLS} cells")
    expected = {
        (experiment, arm, f"promptAblA_{experiment}_{arm}")
        for experiment in _EXPECTED_EXPERIMENTS for arm in _EXPECTED_ARMS
    }
    actual = set()
    for cell in cells:
        if not isinstance(cell, dict):
            raise CampaignRefused("campaign cell is not an object")
        exp, arm, tag = cell.get("exp"), cell.get("cell"), cell.get("tag")
        actual.add((exp, arm, tag))
        if cell.get("runner") != CANONICAL_RUNNER:
            raise CampaignRefused(
                f"{tag!r}: runner must be {CANONICAL_RUNNER!r}")
        if not isinstance(tag, str) or _TAG_RE.fullmatch(tag) is None:
            raise CampaignRefused(f"unsafe campaign tag {tag!r}")
        n = cell.get("N")
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise CampaignRefused(f"{tag!r}: invalid fixed epoch {n!r}")
    if actual != expected:
        raise CampaignRefused(
            "campaign coordinate set is not the canonical four datasets x "
            f"six A arms; missing={sorted(expected-actual)} "
            f"extra={sorted(actual-expected)}")
    if require_recipe:
        _assert_canonical_recipe(plan)
    return cells


def _load_valid_plan(path: Path, *, expected_sha256: str | None,
                     canonical: bool,
                     require_recipe: bool = False) -> tuple[dict, bytes, str]:
    raw = path.read_bytes()
    actual = _sha_bytes(raw)
    if expected_sha256 is not None and actual != expected_sha256:
        raise CampaignRefused(
            f"plan hashes to {actual}, expected {expected_sha256}")
    try:
        plan = load_plan(path, expect_sha256=actual)
    except (PlanRejected, OSError) as error:
        raise CampaignRefused(str(error)) from None
    if canonical:
        _canonical_cells(plan, require_recipe=require_recipe)
    return plan, raw, actual


def campaign_lock_key(plan_path: Path, *,
                      expected_sha256: str | None = None) -> str:
    """Global lock key for this exact result-tag set, independent of ledger."""
    plan, _, _ = _load_valid_plan(
        plan_path, expected_sha256=expected_sha256, canonical=True)
    tags = sorted(str(cell["tag"]) for cell in plan["cells"])
    return _sha_bytes(json.dumps(tags, separators=(",", ":")).encode())


def _boot_id() -> str:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return ""


def open_campaign(ledger_dir: Path, plan_path: Path, *, owner_pid: int,
                  campaign_token: str, expected_plan_sha256: str | None = None,
                  repo_root: Path = REPO) -> dict:
    """Reserve a fresh immutable generation and copy its exact plan bytes."""
    owner = _process_identity(owner_pid)
    if owner_pid not in (os.getpid(), os.getppid()):
        raise CampaignRefused(
            "ledger writer is not the launcher or its direct child")
    if _TOKEN_RE.fullmatch(campaign_token) is None:
        raise CampaignRefused("campaign token must be 256 bits of lowercase hex")
    root = Path(repo_root).resolve()
    if root != REPO:
        raise CampaignRefused(f"campaign repo must be canonical {REPO}, got {root}")
    plan, raw, plan_sha = _load_valid_plan(
        plan_path, expected_sha256=expected_plan_sha256, canonical=True,
        require_recipe=True)
    cells = _canonical_cells(plan, require_recipe=False)
    tags = sorted(str(cell["tag"]) for cell in cells)
    tag_set_sha = _sha_bytes(
        json.dumps(tags, separators=(",", ":")).encode())
    ledger_dir = _safe_ledger_dir(ledger_dir, create=True)
    ledger_stat = ledger_dir.stat()
    sources = source_digests(root)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "campaign_token": campaign_token,
        "plan_source_path": str(plan_path.resolve()),
        "plan_snapshot_path": str((ledger_dir / PLAN_SNAPSHOT_NAME).resolve()),
        "plan_file_sha256": plan_sha,
        "plan_digest": plan.get("plan_digest"),
        "expected_cells": EXPECTED_CELLS,
        "tags": tags,
        "tag_set_sha256": tag_set_sha,
        "selected_n": plan.get("selected_n"),
        "selection_sha256": plan.get("selection_sha256"),
        "repo_root": str(root),
        "result_root": str((root / "result").resolve()),
        "source_sha256": sources,
        "host": socket.gethostname(),
        "launcher_pid": owner_pid,
        "launcher_uid": owner["uid"],
        "launcher_start_ticks": owner["start_ticks"],
        "ledger_writer_pid": os.getpid(),
        "ledger_root": str(ledger_dir),
        "ledger_dev": ledger_stat.st_dev,
        "ledger_ino": ledger_stat.st_ino,
        "boot_id": _boot_id(),
    }
    # Reservation first: a crash cannot leave a reusable directory containing
    # an orphan plan snapshot that a later generation silently adopts.
    _publish_json_exclusive(ledger_dir / RESERVATION_NAME, payload)
    _publish_bytes_exclusive(ledger_dir / PLAN_SNAPSHOT_NAME, raw)
    return payload


def _reservation(ledger_dir: Path, *, campaign_token: str | None = None,
                 require_sources: bool = True,
                 require_owner: bool = True) -> dict:
    ledger_dir = _safe_ledger_dir(ledger_dir)
    held, _ = _read_regular_json(
        ledger_dir / RESERVATION_NAME, what="campaign reservation")
    if held.get("schema_version") != SCHEMA_VERSION:
        raise CampaignRefused("unknown campaign reservation schema")
    if campaign_token is not None and held.get("campaign_token") != campaign_token:
        raise CampaignRefused("campaign token does not own this ledger generation")
    info = ledger_dir.stat()
    if held.get("ledger_root") != str(ledger_dir) \
            or held.get("ledger_dev") != info.st_dev \
            or held.get("ledger_ino") != info.st_ino:
        raise CampaignRefused("ledger directory changed after reservation")
    if require_owner:
        owner = _process_identity(held.get("launcher_pid"))
        if owner.get("uid") != held.get("launcher_uid") \
                or owner.get("start_ticks") != held.get(
                    "launcher_start_ticks") \
                or held.get("boot_id") != _boot_id():
            raise CampaignRefused(
                "reserving launcher identity is no longer live")
    snapshot = Path(str(held.get("plan_snapshot_path") or ""))
    if snapshot != (ledger_dir / PLAN_SNAPSHOT_NAME).resolve() \
            or not snapshot.is_file() or _sha_file(snapshot) != held.get(
                "plan_file_sha256"):
        raise CampaignRefused("immutable campaign plan snapshot is missing or changed")
    plan, _, plan_sha = _load_valid_plan(
        snapshot, expected_sha256=str(held["plan_file_sha256"]), canonical=True,
        require_recipe=require_owner)
    cells = _canonical_cells(plan)
    tags = sorted(str(cell["tag"]) for cell in cells)
    tag_set_sha = _sha_bytes(
        json.dumps(tags, separators=(",", ":")).encode())
    # O_EXCL/mode 0444 protects the API path but is not an immutable trust
    # anchor against the directory owner. Reconstruct every field that has an
    # independent plan, repository, process, or filesystem authority.
    reconstructed = {
        "schema_version": SCHEMA_VERSION,
        "plan_snapshot_path": str(snapshot),
        "plan_file_sha256": plan_sha,
        "plan_digest": plan.get("plan_digest"),
        "expected_cells": EXPECTED_CELLS,
        "tags": tags,
        "tag_set_sha256": tag_set_sha,
        "selected_n": plan.get("selected_n"),
        "selection_sha256": plan.get("selection_sha256"),
        "repo_root": str(REPO),
        "result_root": str((REPO / "result").resolve()),
        "ledger_root": str(ledger_dir),
        "ledger_dev": info.st_dev,
        "ledger_ino": info.st_ino,
        "launcher_uid": os.getuid(),
    }
    if require_owner:
        reconstructed.update({
            "host": socket.gethostname(), "boot_id": _boot_id()})
    wrong = {key: (held.get(key), want)
             for key, want in reconstructed.items()
             if held.get(key) != want}
    expected_keys = set(reconstructed) | {
        "campaign_token", "plan_source_path", "source_sha256",
        "launcher_pid", "launcher_start_ticks", "ledger_writer_pid",
    }
    if not require_owner:
        expected_keys |= {"host", "boot_id"}
    token = held.get("campaign_token")
    source_map = held.get("source_sha256")
    plan_source = held.get("plan_source_path")
    if set(held) != expected_keys:
        wrong["reservation_keys"] = (sorted(held), sorted(expected_keys))
    if not isinstance(token, str) or _TOKEN_RE.fullmatch(token) is None:
        wrong["campaign_token"] = (token, "64 lowercase hexadecimal bytes")
    if not isinstance(plan_source, str) or not Path(plan_source).is_absolute():
        wrong["plan_source_path"] = (plan_source, "absolute path")
    if not require_owner:
        for key in ("host", "boot_id"):
            if not isinstance(held.get(key), str) or not held.get(key):
                wrong[key] = (held.get(key), "non-empty historical identity")
    if not isinstance(source_map, dict) \
            or set(source_map) != set(SOURCE_PATHS) \
            or any(not isinstance(value, str)
                   or re.fullmatch(r"[0-9a-f]{64}", value) is None
                   for value in source_map.values()):
        wrong["source_sha256"] = ("invalid", "exact typed source closure")
    for key in ("launcher_pid", "launcher_start_ticks", "ledger_writer_pid"):
        value = held.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 1:
            wrong[key] = (value, "positive process identity integer")
    if wrong:
        raise CampaignRefused(
            f"campaign reservation differs from its immutable authorities: {wrong}")
    if require_sources:
        _assert_sources_current(held)
    return held


def _cell_for_tag(held: Mapping, tag: str) -> tuple[int, dict]:
    plan_path = Path(str(held["plan_snapshot_path"]))
    plan, _, _ = _load_valid_plan(
        plan_path, expected_sha256=str(held["plan_file_sha256"]), canonical=True,
        require_recipe=False)
    matches = [(index, cell) for index, cell in enumerate(plan["cells"])
               if cell.get("tag") == tag]
    if len(matches) != 1:
        raise CampaignRefused(f"{tag!r} is not exactly one reserved plan cell")
    return matches[0]


def _flag_value(flags: list, option: str, default=None):
    value = default
    for index, token in enumerate(flags):
        if token == option:
            if index + 1 >= len(flags):
                return True
            following = flags[index + 1]
            value = True if str(following).startswith("-") else following
    return value


def _expected_identity_fields(cell: Mapping) -> dict:
    env = cell.get("env") or {}
    flags = list(cell.get("flags") or ())
    n = int(cell["N"])
    cache = str(env["CACHE_OVERRIDE"])
    qwen = str(env["QWEN_OVERRIDE"])
    variant = str(env.get("WHITEN_VARIANT") or "")
    whiten = str(Path(str(env["WDIR_OVERRIDE"])) /
                  f"text_whiten_trainOnly{variant}.npz")
    epoch_budget = int(_flag_value(flags, "-e", n + 1))
    dataset = str(cell["dataset"])
    sinkhorn_init = 0.5 if dataset == "NUSWIDE" else 1.0
    slots, bases = (int(env["GDNA_NUM_SEMANTIC_PARTS"]),
                    int(env["NUM_CODONS"]))
    # MS-COCO's canonical trainer has 0.10 for these three losses; the other
    # three trainers use 0.05.  This is validator compatibility with the
    # existing recipe, not a recipe change.
    text_default = 0.10 if dataset == "MSCOCO" else 0.05
    return {
        "schema_version": 5,
        "dataset": dataset, "setting": "setting1", "seed": 42,
        "num_slots": slots, "bases_per_slot": bases,
        "total_bases": slots * bases, "total_bits": 2 * slots * bases,
        "stop_after_epoch": n, "epoch_budget": epoch_budget,
        "lr_schedule_horizon": epoch_budget,
        "sinkhorn_schedule_horizon": epoch_budget,
        "selection_mode": str(cell["cell"]),
        "codebook_size": int(env["K"]),
        "feature_cache": RunIdentity._artifact(cache),
        "eval_cache": RunIdentity._artifact(cache),
        "qwen_cache": RunIdentity._artifact(qwen),
        "text_whiten": RunIdentity._artifact(whiten),
        "sinkhorn_epsilon_init": sinkhorn_init,
        "sinkhorn_epsilon_final": 0.1,
        "val_split_ratio": 0.0, "val_split_seed": 42,
        "batch_size": 64, "proj_lr": 0.001,
        "routing_adaptive_topp": True,
        "no_routing_adaptive_topp": False,
        "routing_adaptive_topp_min": float(
            _flag_value(flags, "--routing_adaptive_topp_min")),
        "routing_adaptive_topp_max": float(
            _flag_value(flags, "--routing_adaptive_topp_max")),
        "routing_adaptive_topp_entropy": False,
        "routing_perplexity_topk": False,
        "codon_joint_slots": "", "codon_joint_floor": 1e-6,
        "share_codebook": "--share_codebook" in flags,
        "disable_text_supervision": "--disable_text_supervision" in flags,
        "use_gumbel_softmax": "--no_gumbel_softmax" not in flags,
        "lambda_codon_joint": float(
            _flag_value(flags, "--lambda_codon_joint", 0.0)),
        "lambda_text_code_kl": float(
            _flag_value(flags, "--lambda_text_code_kl", text_default)),
        "lambda_text_hash_ntxent": float(
            _flag_value(flags, "--lambda_text_hash_ntxent", text_default)),
        "lambda_xmodal_commit": float(
            _flag_value(flags, "--lambda_xmodal_commit", text_default)),
        "lambda_codeword_codon_sinkhorn": float(
            _flag_value(flags, "--lambda_codeword_codon_sinkhorn", 0.0)),
        "phase3_input_seal": "", "phase3_input_aggregate_sha256": "",
        "phase3_split_identity_sha256": "", "phase3_hf_identity_sha256": "",
        "clip_snapshot_dir": "", "clip_snapshot_revision": "",
        "clip_snapshot_weight_file": "", "clip_snapshot_weight_sha256": "",
        "clip_snapshot_config_sha256": "",
        "clip_snapshot_tokenizers_sha256_json": "",
    }


def _validate_cell_output(held: Mapping, cell: Mapping,
                          run_dir: str) -> dict:
    root = Path(str(held["result_root"]))
    raw_path = Path(run_dir)
    if not raw_path.is_absolute() or raw_path != raw_path.resolve() \
            or not raw_path.is_dir():
        raise CampaignRefused(
            f"{cell['tag']}: run_dir must be one existing canonical absolute directory")
    try:
        raw_path.relative_to(root)
    except ValueError:
        raise CampaignRefused(
            f"{cell['tag']}: run_dir is outside reserved result root {root}") \
            from None
    expected_fragment = f"_{cell['tag']}_P0refit_e{cell['N']}+"
    if expected_fragment not in raw_path.name:
        raise CampaignRefused(
            f"{cell['tag']}: result basename does not carry {expected_fragment!r}")

    identity = load_run_manifest(str(raw_path))
    if identity is None:
        raise CampaignRefused(
            f"{cell['tag']}: missing or invalid current RunIdentity manifest")
    expected = _expected_identity_fields(cell)
    wrong = {key: (getattr(identity, key), want)
             for key, want in expected.items() if getattr(identity, key) != want}
    if wrong:
        raise CampaignRefused(
            f"{cell['tag']}: RunIdentity differs from the planned cell: {wrong}")
    checkpoint = raw_path / "model_state_dict.pth"
    if not checkpoint.is_file() or checkpoint.is_symlink():
        raise CampaignRefused(
            f"{cell['tag']}: canonical checkpoint is missing or symlinked")
    metadata = CheckpointMetadata.load(str(checkpoint))
    expected_metadata = {
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": _sha_file(checkpoint),
        "checkpoint_epoch_zero_based": identity.stop_after_epoch,
        "training_epoch_budget": identity.epoch_budget,
        "stop_after_epoch": identity.stop_after_epoch,
        "lr_schedule_horizon": identity.lr_schedule_horizon,
        "sinkhorn_schedule_horizon": identity.sinkhorn_schedule_horizon,
        "sinkhorn_epsilon_init": identity.sinkhorn_epsilon_init,
        "sinkhorn_epsilon_final": identity.sinkhorn_epsilon_final,
    }
    if metadata is None:
        raise CampaignRefused(
            f"{cell['tag']}: checkpoint runtime metadata is missing or malformed")
    wrong_metadata = {
        key: (getattr(metadata, key), want)
        for key, want in expected_metadata.items()
        if getattr(metadata, key) != want
    }
    if wrong_metadata:
        raise CampaignRefused(
            f"{cell['tag']}: checkpoint runtime differs from plan/weights: "
            f"{wrong_metadata}")
    try:
        marker = read_analysis_marker(
            str(raw_path), allow_backfilled=False,
            expected=ExpectedIdentity(
                dataset=identity.dataset, random_seed=identity.seed,
                inference_epoch=identity.stop_after_epoch,
                num_slots=identity.num_slots,
                bases_per_slot=identity.bases_per_slot,
                codebook_size=identity.codebook_size),
            expected_protocol={
                "dataset": str(cell["dataset"]),
                "codebook_size": int((cell.get("env") or {})["K"]),
                "total_bases": expected["total_bases"],
            }, required_splits=("db", "query"))
    except ExtractionInvalid as error:
        raise CampaignRefused(
            f"{cell['tag']}: analysis completion is invalid: {error}") from None
    if marker.get("artifact_kind") != "standalone_analysis_integrity_marker" \
            or marker.get("scientific_authority") is not False \
            or marker.get("paper_result_eligible") is not False \
            or (marker.get("standalone_authority") or {}).get(
                "requires_downstream_deterministic_recomputation") is not True:
        raise CampaignRefused(
            f"{cell['tag']}: analysis marker does not preserve its declared "
            "integrity-only/non-paper authority boundary")

    mandatory = (
        "run_identity.json", "args.txt", "model_state_dict.pth",
        "model_state_dict.pth.runtime.json", "extraction_complete.json",
        "extraction_manifest_db.json", "extraction_manifest_query.json",
        "extract_db.npz", "extract_query.npz",
        "evaluation_siglip2_base.json",
        "evaluation_siglip2_base_bioproj.json", "cell_result.json",
        "pairwise_nmi.json", "analysis_complete.json",
    )
    artifacts = {}
    for name in mandatory:
        path = raw_path / name
        if not path.is_file() or path.is_symlink():
            raise CampaignRefused(
                f"{cell['tag']}: required output is absent or symlinked: {name}")
        artifacts[name] = {
            "sha256": _sha_file(path), "size_bytes": path.stat().st_size}
    return {
        "run_dir": str(raw_path), "run_identity_digest": identity.digest,
        "artifacts": artifacts, "analysis_metrics": dict(marker["metrics"]),
        "analysis_authority": {
            "scientific_authority": False, "paper_result_eligible": False,
            "requires_downstream_deterministic_recomputation": True,
        },
    }


def _validate_outcome(held: Mapping, *, path: Path, index: int, tag: str,
                      run_dir: str,
                      require_physical: bool = True) -> tuple[dict, str]:
    outcome, digest = _read_regular_json(path, what="executor outcome")
    expected = {
        "schema": "groundeddna.ablation-cell-outcome", "schema_version": 1,
        "campaign_token": held["campaign_token"],
        "plan_file_sha256": held["plan_file_sha256"],
        "cell_index": index, "tag": tag, "child_returncode": 0,
        "repo_root": held["repo_root"], "run_dir": run_dir,
        "source_sha256": held["source_sha256"],
    }
    wrong = {key: (outcome.get(key), want) for key, want in expected.items()
             if outcome.get(key) != want}
    gpu = outcome.get("gpu")
    lease_paths = outcome.get("gpu_lease_paths")
    expected_lease = None
    physical_gpu = None
    gpu_index = gpu.get("index") if isinstance(gpu, dict) else None
    gpu_uuid = gpu.get("uuid") if isinstance(gpu, dict) else None
    executor_pid = outcome.get("executor_pid")
    completed_text = outcome.get("completed_at_utc")
    completed = None
    if isinstance(completed_text, str):
        try:
            completed = datetime.fromisoformat(completed_text)
        except ValueError:
            completed = None
    if isinstance(gpu_uuid, str):
        from dna_utils.gpu_lease import GPU_LEASE_ROOT
        expected_lease = [str(
            GPU_LEASE_ROOT / f"{gpu_uuid}.lock")]
    expected_keys = set(expected) | {
        "gpu", "gpu_lease_paths", "executor_pid", "completed_at_utc"}
    if wrong or set(outcome) != expected_keys or not isinstance(gpu, dict) \
            or isinstance(gpu_index, bool) or not isinstance(gpu_index, int) \
            or gpu_index < 0 or not isinstance(gpu_uuid, str) \
            or re.fullmatch(r"GPU-[A-Za-z0-9-]+", gpu_uuid) is None \
            or lease_paths != expected_lease \
            or isinstance(executor_pid, bool) \
            or not isinstance(executor_pid, int) or executor_pid <= 1 \
            or completed is None or completed.tzinfo is None \
            or completed > datetime.now(timezone.utc) + timedelta(minutes=5):
        raise CampaignRefused(
            f"{tag}: executor outcome is not bound to this campaign: {wrong}")
    # A ``GPU-`` prefix plus a matching filename is only self-assertion.  The
    # record boundary independently binds the complete tuple to a device that
    # the sealed host tool actually reports at this index.
    if require_physical:
        try:
            physical_gpu = _physical_gpu_assignment(gpu_index)
        except (PlanRejected, OSError) as error:
            raise CampaignRefused(
                f"{tag}: cannot revalidate physical GPU assignment: {error}") \
                from None
        if gpu != physical_gpu:
            raise CampaignRefused(
                f"{tag}: recorded GPU does not match the physical host assignment")
    return outcome, digest


def _canonical_outcome_path(ledger_dir: Path, tag: str,
                            supplied: str | Path) -> Path:
    ledger = _safe_ledger_dir(ledger_dir)
    parent = ledger / "outcomes"
    expected = parent / f"{tag}.json"
    raw = Path(supplied)
    if not raw.is_absolute() or raw != expected or not parent.is_dir() \
            or parent.is_symlink() or parent.resolve() != parent \
            or raw.resolve() != expected:
        raise CampaignRefused(
            f"{tag}: outcome must be canonical ledger/outcomes/tag.json")
    return expected


def record_cell(ledger_dir: Path, tag: str, status: str, *,
                campaign_token: str, run_dir: str | None = None,
                outcome_path: str | None = None,
                detail: str | None = None) -> dict:
    if status not in STATUSES:
        raise CampaignRefused(f"status {status!r} is not one of {STATUSES}")
    ledger_dir = _safe_ledger_dir(ledger_dir)
    held = _reservation(ledger_dir, campaign_token=campaign_token)
    if (ledger_dir / RECEIPT_NAME).exists():
        raise CampaignRefused("sealed campaign cannot accept another cell record")
    if tag not in held["tags"]:
        raise CampaignRefused(f"{tag!r} is not a reserved campaign cell")
    index, cell = _cell_for_tag(held, tag)
    output_evidence = None
    outcome_sha = None
    if status == "ok":
        if not run_dir or not outcome_path:
            raise CampaignRefused(
                f"{tag}: success requires both run_dir and executor outcome")
        canonical_outcome = _canonical_outcome_path(
            ledger_dir, tag, outcome_path)
        _, outcome_sha = _validate_outcome(
            held, path=canonical_outcome, index=index, tag=tag,
            run_dir=run_dir)
        output_evidence = _validate_cell_output(held, cell, run_dir)
    elif run_dir is not None or outcome_path is not None:
        raise CampaignRefused(
            f"{tag}: failed record must not attach success output evidence")
    entry = {
        "schema_version": SCHEMA_VERSION, "campaign_token": campaign_token,
        "tag": tag, "cell_index": index, "status": status, "detail": detail,
        "plan_digest": held["plan_digest"],
        "plan_file_sha256": held["plan_file_sha256"],
        "outcome_path": (str(canonical_outcome) if status == "ok" else None),
        "outcome_sha256": outcome_sha,
        "output_evidence": output_evidence,
    }
    _publish_json_exclusive(ledger_dir / f"cell_{tag}.json", entry)
    return entry


def _reopen_success_entry(ledger_dir: Path, held: Mapping,
                          tag: str) -> tuple[dict, str]:
    path = ledger_dir / f"cell_{tag}.json"
    entry, entry_sha = _read_regular_json(path, what=f"cell record {tag}")
    index, cell = _cell_for_tag(held, tag)
    exact = {
        "schema_version": SCHEMA_VERSION,
        "campaign_token": held["campaign_token"], "tag": tag,
        "cell_index": index, "status": "ok",
        "detail": None,
        "plan_digest": held["plan_digest"],
        "plan_file_sha256": held["plan_file_sha256"],
    }
    wrong = {key: (entry.get(key), want) for key, want in exact.items()
             if entry.get(key) != want}
    expected_keys = set(exact) | {
        "outcome_path", "outcome_sha256", "output_evidence"}
    if wrong or set(entry) != expected_keys:
        raise CampaignRefused(f"{tag}: invalid cell record binding {wrong}")
    evidence = entry.get("output_evidence")
    if not isinstance(evidence, dict):
        raise CampaignRefused(f"{tag}: cell record has no output evidence")
    run_dir = str(evidence.get("run_dir") or "")
    outcome_path = entry.get("outcome_path")
    if not isinstance(outcome_path, str):
        raise CampaignRefused(f"{tag}: cell record has no executor outcome")
    canonical_outcome = _canonical_outcome_path(
        ledger_dir, tag, outcome_path)
    _, outcome_sha = _validate_outcome(
        held, path=canonical_outcome, index=index, tag=tag, run_dir=run_dir,
        require_physical=False)
    if outcome_sha != entry.get("outcome_sha256"):
        raise CampaignRefused(f"{tag}: executor outcome changed after record")
    current = _validate_cell_output(held, cell, run_dir)
    if current != evidence:
        raise CampaignRefused(f"{tag}: output bytes changed after cell record")
    return entry, entry_sha


def seal_campaign(ledger_dir: Path, *, campaign_token: str) -> dict:
    ledger_dir = _safe_ledger_dir(ledger_dir)
    held = _reservation(ledger_dir, campaign_token=campaign_token)
    entries, entry_sha, failures = {}, {}, []
    for tag in held["tags"]:
        path = ledger_dir / f"cell_{tag}.json"
        if not path.is_file():
            failures.append(f"{tag}:missing")
            continue
        try:
            entry, digest = _reopen_success_entry(ledger_dir, held, tag)
        except CampaignRefused as error:
            failures.append(f"{tag}:{error}")
            continue
        entries[tag], entry_sha[tag] = entry, digest
    if failures:
        raise CampaignRefused(
            f"cannot seal: {len(entries)} of {EXPECTED_CELLS} cells pass "
            f"completion validation; failures={failures}")
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "receipt_kind": "execution_and_analysis_integrity_only",
        "scientific_authority": False, "paper_result_eligible": False,
        "requires_downstream_deterministic_recomputation": True,
        "campaign_token": held["campaign_token"],
        "plan_digest": held["plan_digest"],
        "plan_file_sha256": held["plan_file_sha256"],
        "tag_set_sha256": held["tag_set_sha256"],
        "source_sha256": held["source_sha256"],
        "selected_n": held.get("selected_n"),
        "selection_sha256": held.get("selection_sha256"),
        "cell_record_sha256": entry_sha,
        "cells": {
            tag: entries[tag]["output_evidence"] for tag in sorted(entries)},
        "cell_count": len(entries),
    }
    _publish_json_exclusive(ledger_dir / RECEIPT_NAME, receipt)
    return receipt


def require_sealed(ledger_dir: Path, *, expected_cells: int = EXPECTED_CELLS,
                   plan_file_sha256: str | None = None) -> dict:
    ledger_dir = _safe_ledger_dir(ledger_dir)
    held = _reservation(ledger_dir, require_owner=False)
    receipt, receipt_sha = _read_regular_json(
        ledger_dir / RECEIPT_NAME, what="campaign completion receipt")
    exact = {
        "schema_version": SCHEMA_VERSION,
        "receipt_kind": "execution_and_analysis_integrity_only",
        "scientific_authority": False, "paper_result_eligible": False,
        "requires_downstream_deterministic_recomputation": True,
        "campaign_token": held["campaign_token"],
        "plan_digest": held["plan_digest"],
        "plan_file_sha256": held["plan_file_sha256"],
        "tag_set_sha256": held["tag_set_sha256"],
        "source_sha256": held["source_sha256"],
        "selected_n": held.get("selected_n"),
        "selection_sha256": held.get("selection_sha256"),
        "cell_count": expected_cells,
    }
    wrong = {key: (receipt.get(key), want) for key, want in exact.items()
             if receipt.get(key) != want}
    if wrong or expected_cells != EXPECTED_CELLS:
        raise CampaignRefused(f"campaign receipt contract differs: {wrong}")
    expected_receipt_keys = set(exact) | {"cell_record_sha256", "cells"}
    if set(receipt) != expected_receipt_keys:
        raise CampaignRefused(
            "campaign receipt contains missing or unrecognized fields")
    cells = receipt.get("cells")
    digests = receipt.get("cell_record_sha256")
    if not isinstance(cells, dict) or set(cells) != set(held["tags"]) \
            or not isinstance(digests, dict) or set(digests) != set(held["tags"]):
        raise CampaignRefused("campaign receipt does not cover the exact tag set")
    if plan_file_sha256 is not None \
            and receipt.get("plan_file_sha256") != plan_file_sha256:
        raise CampaignRefused("campaign receipt seals a different plan")
    for tag in held["tags"]:
        entry, digest = _reopen_success_entry(ledger_dir, held, tag)
        if digest != digests[tag] or entry["output_evidence"] != cells[tag]:
            raise CampaignRefused(f"{tag}: receipt does not match cell record")
    receipt["receipt_sha256"] = receipt_sha
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("key", "open", "record", "seal", "verify"):
        command = sub.add_parser(name)
        if name != "key":
            command.add_argument("--ledger", required=True)
        if name in ("key", "open", "verify"):
            command.add_argument("--plan", required=True)
        if name in ("key", "open"):
            command.add_argument("--expect-plan-sha256", default=None)
        if name == "open":
            command.add_argument("--owner-pid", required=True, type=int)
            command.add_argument("--campaign-token", required=True)
        if name == "record":
            command.add_argument("--campaign-token", required=True)
            command.add_argument("--tag", required=True)
            command.add_argument("--status", required=True, choices=STATUSES)
            command.add_argument("--run-dir", default=None)
            command.add_argument("--outcome", default=None)
            command.add_argument("--detail", default=None)
        if name == "seal":
            command.add_argument("--campaign-token", required=True)
    args = parser.parse_args()
    try:
        if args.action == "key":
            print(campaign_lock_key(
                Path(args.plan), expected_sha256=args.expect_plan_sha256))
        elif args.action == "open":
            held = open_campaign(
                Path(args.ledger), Path(args.plan), owner_pid=args.owner_pid,
                campaign_token=args.campaign_token,
                expected_plan_sha256=args.expect_plan_sha256)
            print(f"[ledger] reserved {len(held['tags'])} cells for "
                  f"generation {held['campaign_token'][:12]}")
        elif args.action == "record":
            entry = record_cell(
                Path(args.ledger), args.tag, args.status,
                campaign_token=args.campaign_token, run_dir=args.run_dir,
                outcome_path=args.outcome, detail=args.detail)
            print(f"[ledger] {entry['tag']} {entry['status']}")
        elif args.action == "seal":
            receipt = seal_campaign(
                Path(args.ledger), campaign_token=args.campaign_token)
            print(f"[ledger] sealed {receipt['cell_count']} integrity-only cells")
        else:
            raw = Path(args.plan).read_bytes()
            receipt = require_sealed(
                Path(args.ledger), expected_cells=EXPECTED_CELLS,
                plan_file_sha256=_sha_bytes(raw))
            print(f"[ledger] integrity-complete: {receipt['cell_count']} cells; "
                  "scientific_authority=false")
    except (CampaignRefused, OSError, KeyError, TypeError, ValueError) as error:
        print(f"[ledger] REFUSED: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
