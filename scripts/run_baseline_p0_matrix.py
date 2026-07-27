#!/usr/bin/env python3
"""Launch the fixed baseline P0 matrix on a GPU pool.

This is an orchestration layer around ``run_modern_baseline_p0.py``.  It does
not override a method's training horizon, batch size, optimizer, scheduler, or
loss defaults.  The only protocol opt-in is
``--allow-main-ineligible-smoke`` because the current caches predate the strict
provenance contract.  Consequently these runs are *diagnostic legacy-cache
results*, not main-table-eligible paper results.

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
from typing import Iterable, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "scripts/run_modern_baseline_p0.py"

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
BITS = (36, 48)
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
    "cibhash": 100,
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
# Completion/resume is source-profile aware.  Otherwise an internally valid
# historical run could suppress a required canonical rerun and only be
# rejected later by the aggregator, silently leaving a missing matrix cell.
# Keep these exact method-file profiles synchronized with
# aggregate_baseline_p0_matrix.CANONICAL_VARIANT_SOURCE_PROFILES.
CANONICAL_VARIANT_SOURCE_PROFILES: Mapping[str, tuple[str, str]] = {
    "cibhash": (
        "baseline/CIBHash.py",
        "1277bb94376e513f99aeb6f9d0c912c59502d035a4ba4626de9d8ca02ca696fc",
    ),
    "cimon": (
        "baseline/CIMON.py",
        "afd4696e2b3633d6117c6eb955ec08e32ab6be65374df046103671f504e0aaf5",
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
        "4593e5c08e81aeb10c23a1883700e0a79ef05634f14ae6cd2c1575aab09cd7f7",
    ),
    "sdc-paper": (
        "baseline/SDC.py",
        "7f875f5ff83cf52b1cbec8863672187135266f52580dd6a37246f70d0c0df706",
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
    "baseline/base_model.py",
    "baseline/modern_unsupervised.py",
    "baseline/asset_provenance.py",
    "baseline/cache_provenance.py",
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
    }),
    # Exact reviewed transition: CRH was added only as a new lazy dispatch
    # branch/help entry.  Existing U0/U2 model construction is unchanged.
    "baseline/base_model.py": frozenset({
        "c9f39c05a27cca24ab0084bdd00462b634953499021a67c1cdc18dfc33837a47",
        "b9d7e1a0cfab613fd72ce3dc38fafeca82ca99f6121d20138bf22239ef34f3be",
    }),
    # Exact reviewed transition: the new CRH variant/semantic branch and
    # comparison-panel metadata do not alter any existing U0/U2 training,
    # selection, extraction, metric, or DNA-projection path.
    "scripts/run_modern_baseline_p0.py": frozenset({
        "9809a70fde66d473540fa11d10752a83453d60ac6bfc9b80d0490a1e7ce7eca5",
        "2d7234a2a8f959f32a1f0fa5199cf00556289084f351359f7cb3cbcbdd4f5b16",
    }),
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
        aliases = KNOWN_NON_SCIENTIFIC_SOURCE_SHA_ALIASES.get(path)
        if aliases is None or current not in aliases or raw_digest not in aliases:
            return False
    return True


def _full_manifest_validation_passes(
        path: Path, variant: str, dataset: str, bit: int, seed: int) -> bool:
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
        verify_hashes=False)
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


def _completed_keys(result_root: Path) -> dict[tuple[str, str, int, int], list[Path]]:
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
                and identity.get("val_seed") == 42
                and identity.get("val_ratio") == 0.1
                and identity.get("eval_period") == 5
                and identity.get("horizon") == expected_horizon
                and identity.get("batch_size_override") is None
                and _matches_canonical_source_profile(payload, str(variant))
                and _matches_semantic_information_condition(
                    payload, str(variant), str(dataset))
                and payload.get("protocol_deviations") == []
                and payload.get("test_used_for_selection") is False
                and payload.get("final_checkpoint_count") == 1
                and payload.get("run_manifest_phase") == "bio_projection_completed"
                and payload.get("bio_projection_status")
                in {"paper_result_eligible", "completed_not_paper_eligible"}
                and bio_path.is_file()
                and payload.get("bio_projection_manifest_sha256") == _sha256(bio_path)
                and _full_manifest_validation_passes(
                    path, str(variant), str(dataset), int(bit), seed)
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


def _command(attempt: Attempt, *, python: Path, gpu_id: str,
             num_workers: int) -> list[str]:
    job = attempt.job
    command = [
        str(python), str(RUNNER),
        "--variant", job.variant,
        "--dataset", job.dataset,
        "--bit", str(job.bit),
        "--seed", str(job.seed),
        "--device", "cuda:0",
        "--num-workers", str(num_workers),
        "--model-root", str(attempt.model_root),
        "--result-root", str(attempt.result_root),
        "--compress-root", str(attempt.compress_root),
        "--allow-main-ineligible-smoke",
    ]
    if job.variant == "umrch":
        embeddings, adapter, manifest = UMRCH_ASSETS[job.dataset]
        command.extend([
            "--umrch-concept-embeddings", str((REPO / embeddings).resolve()),
            "--umrch-vision-adapter", str((REPO / adapter).resolve()),
            "--umrch-asset-manifest", str((REPO / manifest).resolve()),
        ])
    # gpu_id is deliberately represented in the environment, not --device.
    # Each process sees exactly one physical GPU as logical cuda:0.
    del gpu_id
    return command


def _validate_inputs(python: Path, jobs: Sequence[Job]) -> None:
    if not python.is_file() or not os.access(python, os.X_OK):
        raise FileNotFoundError(f"Python interpreter is not executable: {python}")
    if not RUNNER.is_file():
        raise FileNotFoundError(f"P0 runner is missing: {RUNNER}")
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


def _plan_record(attempt: Attempt, command: Sequence[str]) -> dict[str, object]:
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
        "comparison_label": LEGACY_LABEL,
    }


def _status_payload(attempt: Attempt, command: Sequence[str], gpu_id: str,
                    *, state: str, started_at: str,
                    returncode: int | None = None,
                    message: str | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "state": state,
        "comparison_label": LEGACY_LABEL,
        "job_id": attempt.job.job_id,
        "panel": attempt.job.panel,
        "variant": attempt.job.variant,
        "dataset": attempt.job.dataset,
        "bit": attempt.job.bit,
        "seed": attempt.job.seed,
        "attempt": attempt.number,
        "physical_gpu": gpu_id,
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
                   gpus: Sequence[str], num_workers: int,
                   skipped: Mapping[tuple[str, str, int, int], list[Path]]) -> None:
    print(f"comparison_label={LEGACY_LABEL}")
    print(f"scheduled={len(attempts)} completed_skipped={len(skipped)}")
    print(f"gpu_pool={','.join(gpus)}")
    print(f"exact_duheg={DUHEG_BLOCK['status']}: {DUHEG_BLOCK['reason']}")
    for index, attempt in enumerate(attempts):
        gpu = gpus[index % len(gpus)]
        command = _command(
            attempt, python=python, gpu_id=gpu, num_workers=num_workers)
        print(
            f"CUDA_VISIBLE_DEVICES={shlex.quote(gpu)} "
            + shlex.join(command))


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
        description="Launch the labelled legacy-cache P0 baseline matrix.")
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
                        default=default_bits,
                        help="Defaults from BITS environment.")
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
        "--resume-root", action="append", default=[],
        help=("Additional result root to scan for protocol-valid completed "
              "cells before scheduling; repeat for multiple external queues."))
    parser.add_argument("--no-resume", action="store_true",
                        help="Do not skip cells with a verified completed manifest.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands only; create no files and launch nothing.")
    args = parser.parse_args()

    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    if not args.gpus or len(set(args.gpus)) != len(args.gpus):
        parser.error("--gpus must contain at least one unique GPU ID")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must contain at least one unique integer")

    python = _absolute(args.python)
    seed_slug = "seeds" + "-".join(map(str, args.seeds))
    matrix_name = (
        f"p0_matrix_{seed_slug}_{args.panel}_legacy_cache"
        if args.panel in {"supervised", "all-plus-supervised"}
        else f"p0_matrix_{seed_slug}_legacy_cache"
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
    _validate_inputs(python, jobs)

    completed: dict[tuple[str, str, int, int], list[Path]] = {}
    if not args.no_resume:
        for resume_root in (result_root, *map(_absolute, args.resume_root)):
            for key, paths in _completed_keys(resume_root).items():
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
            num_workers=args.num_workers, skipped=completed)
        return 0

    log_root.mkdir(parents=True, exist_ok=True)
    lock_path = log_root / "launcher.lock"
    lock_handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(
            f"another matrix launcher holds {lock_path}; refusing overlap") from error

    created = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    plan_path = log_root / f"matrix_plan_{created}_{os.getpid()}.json"
    plan_jobs = []
    for index, attempt in enumerate(attempts):
        gpu = args.gpus[index % len(args.gpus)]
        plan_jobs.append(_plan_record(
            attempt,
            _command(attempt, python=python, gpu_id=gpu,
                     num_workers=args.num_workers)))
    _atomic_json(plan_path, {
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "comparison_label": LEGACY_LABEL,
        "warning": (
            "The explicit legacy-cache opt-in keeps every affected metric "
            "out of the paper main table until strict caches are regenerated."
        ),
        "runner": str(RUNNER),
        "runner_sha256": _sha256(RUNNER),
        "train_seeds": list(args.seeds),
        "three_seed_mean_std_status": (
            "pending" if len(args.seeds) < 3 else "candidate_runs_scheduled"
        ),
        "val_seed": 42,
        "bits": list(args.bits),
        "datasets": list(args.datasets),
        "gpu_pool": list(args.gpus),
        "exact_duheg": DUHEG_BLOCK,
        "completed_cells_skipped": {
            "|".join(map(str, key)): [str(path) for path in paths]
            for key, paths in sorted(completed.items())
        },
        "jobs": plan_jobs,
    })

    print(f"[matrix] label={LEGACY_LABEL}", flush=True)
    print(
        f"[matrix] plan={plan_path} pending={len(attempts)} "
        f"completed_skipped={len(completed)} gpus={','.join(args.gpus)}",
        flush=True)
    print(
        f"[matrix] exact DUH-EG: {DUHEG_BLOCK['status']} "
        f"({DUHEG_BLOCK['reason']})", flush=True)

    if not attempts:
        print("[matrix] no pending cells", flush=True)
        return 0

    work: queue.Queue[Attempt] = queue.Queue()
    for attempt in attempts:
        work.put(attempt)
    stop = threading.Event()
    output_lock = threading.Lock()
    process_lock = threading.Lock()
    active: dict[str, subprocess.Popen[bytes]] = {}
    failures: list[tuple[str, int]] = []

    previous_handlers: dict[int, object] = {}

    def request_stop(signum: int, _frame: object) -> None:
        stop.set()
        with output_lock:
            print(f"[matrix] received signal {signum}; stopping workers", flush=True)
        with process_lock:
            processes = list(active.values())
        for process in processes:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, request_stop)

    def worker(gpu_id: str) -> None:
        while not stop.is_set():
            try:
                attempt = work.get_nowait()
            except queue.Empty:
                return
            command = _command(
                attempt, python=python, gpu_id=gpu_id,
                num_workers=args.num_workers)
            started = _utc_now()
            attempt.log_path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_json(
                attempt.status_path,
                _status_payload(
                    attempt, command, gpu_id, state="running",
                    started_at=started))
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu_id
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
                        f"# comparison_label={LEGACY_LABEL}\n"
                        f"# physical_gpu={gpu_id} logical_device=cuda:0\n"
                        f"# command={shlex.join(command)}\n")
                    log_handle.flush()
                    process = subprocess.Popen(
                        command, cwd=REPO, env=environment,
                        stdout=log_handle, stderr=subprocess.STDOUT,
                        start_new_session=True)
                    with process_lock:
                        active[attempt.job.job_id] = process
                    returncode = process.wait()
                if returncode == 0:
                    completed_after_run = _completed_keys(attempt.result_root)
                    if attempt.job.key not in completed_after_run:
                        returncode = 3
                        message = (
                            "runner exited zero but no protocol-valid completed "
                            "p0_run_manifest + bio_projection pair was found")
            except Exception as error:  # preserve a durable failure record
                message = f"{type(error).__name__}: {error}"
                returncode = 1
            finally:
                with process_lock:
                    active.pop(attempt.job.job_id, None)
            state = "completed" if returncode == 0 else "failed"
            _atomic_json(
                attempt.status_path,
                _status_payload(
                    attempt, command, gpu_id, state=state,
                    started_at=started, returncode=returncode,
                    message=message))
            if returncode != 0:
                failures.append((attempt.job.job_id, returncode))
            with output_lock:
                print(
                    f"[matrix][gpu {gpu_id}] {state.upper()} "
                    f"{attempt.job.job_id} rc={returncode}", flush=True)
            work.task_done()

    threads = [
        threading.Thread(target=worker, args=(gpu,), name=f"gpu-{gpu}")
        for gpu in args.gpus
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    for signum, handler in previous_handlers.items():
        signal.signal(signum, handler)

    summary_path = log_root / f"matrix_summary_{created}_{os.getpid()}.json"
    _atomic_json(summary_path, {
        "schema_version": 1,
        "finished_at_utc": _utc_now(),
        "comparison_label": LEGACY_LABEL,
        "plan": str(plan_path),
        "scheduled": len(attempts),
        "failures": [
            {"job_id": job_id, "returncode": returncode}
            for job_id, returncode in failures
        ],
        "interrupted": stop.is_set(),
        "exact_duheg": DUHEG_BLOCK,
    })
    print(
        f"[matrix] finished failures={len(failures)} summary={summary_path}",
        flush=True)
    if stop.is_set():
        return 130
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
