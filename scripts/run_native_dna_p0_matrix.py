#!/usr/bin/env python3
"""Launch the sealed native-DNA P0 method x dataset x seed matrix.

Every GPU ID owns one worker thread and exposes only that physical GPU to its
child as logical ``cuda:0``.  A failed child is recorded but does not cancel
the remaining queue.  The launcher is intentionally fail-closed:

* ``--data-root`` is mandatory and must resolve below ``/data``;
* a protocol-valid completed manifest is skipped;
* any existing but incomplete, invalid, duplicate, or eligibility-mismatched
  cell aborts the matrix before a new child is launched; and
* a zero child return code is accepted only after the completed manifest has
  been validated again.

The single-cell driver owns model outputs.  Consequently there is no separate
model root here: runs are written to ``<data-root>/runs`` and launcher records
to ``<data-root>/logs``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import re
import shlex
import subprocess
import sys
import threading
from typing import Callable, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.aggregate_native_dna_p0 import (  # noqa: E402
    Key as AggregateKey,
    _validate_manifest,
)


DEFAULT_METHODS = ("bee2018", "bee2021", "koike2024", "koike2026")
DEFAULT_DATASETS = ("Flickr25k", "MSCOCO", "NUSWIDE", "CIFAR10")
DEFAULT_SEEDS = (42, 43, 44)
# Matched code length in bases. 18 was the 6-slot / 36-bit budget; the paper
# is 5 slots x 3 bases = 15 (30 bit) since 2026-08-11. Overridden by the env
# var rather than a flag because the constant is read at module scope by both
# this file and the matrix launcher, before argparse runs, and the two MUST
# agree or the matrix gate rejects every cell for base_length mismatch.
# F18: the module constant is the launcher's own configured length, but the
# manifest checker takes the protocol as an argument so a cell written at a
# different length can be judged as such rather than reported as a mismatch.
from dna_utils.native_protocol import (  # noqa: E402
    NativeProtocol, coerce_protocol, resolve_native_protocol)

DEFAULT_PROTOCOL = resolve_native_protocol()
MATCHED_LENGTH = DEFAULT_PROTOCOL.length_bases
SETTING = "setting1"
MANIFEST_NAME = "native_p0_run_manifest.json"
RUNNER = REPO / "scripts/run_native_dna_p0.py"

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_HORIZON = {
    "bee2018": 65,
    "bee2021": 100,
    "koike2024": 150,
    "koike2026": 1000,
}
_DATASET_WEIGHT = {
    "Flickr25k": 1,
    "MSCOCO": 4,
    "NUSWIDE": 2,
    "CIFAR10": 1,
}
_TEST_ACCESS_CONTRACT = {
    "stage1_test_evaluated": False,
    "refit_from_scratch": True,
    "refit_weights_argument": None,
    "terminal_trainer_invocations": 1,
    "terminal_official_test_extraction_passes": 1,
    "post_run_validation_reopens_official_test": False,
    "raw_and_post_dp_share_terminal_extraction": True,
    "full_train_extraction_saved_after_terminal_test": True,
}


class MatrixStateError(RuntimeError):
    """Raised when an existing matrix root cannot be resumed safely."""


@dataclass(frozen=True)
class Job:
    method: str
    dataset: str
    seed: int

    @property
    def key(self) -> tuple[str, str, int]:
        return self.method, self.dataset, self.seed

    @property
    def aggregate_key(self) -> AggregateKey:
        return AggregateKey(self.method, self.dataset, self.seed)

    @property
    def job_id(self) -> str:
        dataset = self.dataset.lower().replace("_", "")
        return f"{self.method}_{dataset}_{MATCHED_LENGTH}nt_P0_seed{self.seed}"

    @property
    def cell_prefix(self) -> str:
        return f"{self.job_id}_p"

    @property
    def scheduling_weight(self) -> int:
        return _HORIZON[self.method] * _DATASET_WEIGHT[self.dataset]


@dataclass(frozen=True)
class LaunchOptions:
    python: Path
    driver: Path
    dataset_root: Path
    result_root: Path
    log_root: Path
    predictor: Path | None
    num_workers: int
    extract_batch_size: int
    query_chunk: int
    allow_diagnostic: bool


@dataclass(frozen=True)
class RunResult:
    job: Job
    gpu_id: str
    returncode: int
    log_path: Path
    message: str | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _absolute(path: str | Path) -> Path:
    value = Path(path).expanduser()
    if value.is_absolute():
        return value.resolve()
    return (REPO / value).resolve()


def _validate_data_root(path: str | Path) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute():
        raise ValueError("--data-root must be an absolute path below /data")
    resolved = value.resolve()
    data = Path("/data")
    if resolved == data or not resolved.is_relative_to(data):
        raise ValueError(
            "--data-root must resolve to a dedicated directory below /data"
        )
    return resolved


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _jobs(
    methods: Sequence[str],
    datasets: Sequence[str],
    seeds: Sequence[int],
) -> list[Job]:
    jobs = [
        Job(method, dataset, seed)
        for method in methods
        for dataset in datasets
        for seed in seeds
    ]
    return sorted(jobs, key=lambda item: (-item.scheduling_weight, item.job_id))


def _canonical_protocol_digest(identity: Mapping[str, object]) -> str:
    encoded = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _expected_protocol(
    job: Job,
    options: LaunchOptions,
) -> tuple[dict[str, object], str]:
    """Recompute the exact identity the current canonical cell driver will use."""
    from scripts.run_native_dna_p0 import (
        DEFAULT_CACHE,
        _audit_cache,
        _protocol_identity,
    )

    cache_dir = _absolute(DEFAULT_CACHE[job.dataset])
    cache_audit = _audit_cache(cache_dir)
    predictor = options.predictor if job.method == "bee2021" else None
    identity, digest = _protocol_identity(
        method=job.method,
        dataset=job.dataset,
        setting=SETTING,
        seed=job.seed,
        cache_audit=cache_audit,
        dataset_root=options.dataset_root,
        predictor=predictor,
        device="cuda:0",
        num_workers=options.num_workers,
        extract_batch_size=options.extract_batch_size,
        query_chunk=options.query_chunk,
    )
    return identity, digest


def _manifest_contract_errors(
    path: Path,
    job: Job,
    *,
    expected_identity: Mapping[str, object],
    expected_digest: str,
    protocol: "NativeProtocol | int | None" = None,
) -> tuple[str, list[str]]:
    """Return aggregator status and launcher-specific contract errors."""
    protocol = coerce_protocol(protocol)
    record = _validate_manifest(
        path, job.aggregate_key, verify_hashes=True, protocol=protocol)
    status = str(record.get("status", "invalid"))
    errors = [
        str(item) for item in record.get("validation_errors", [])
    ]
    if status == "invalid":
        return status, errors

    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        return "invalid", [f"cannot reload manifest: {error}"]
    if not isinstance(payload, dict):
        return "invalid", ["manifest root must be an object"]

    if payload.get("schema_version") != 1:
        errors.append("schema_version must equal 1")
    if payload.get("setting") != SETTING:
        errors.append(f"setting must equal {SETTING!r}")
    if payload.get("base_length") != protocol.length_bases:
        errors.append(f"base_length must equal {protocol.length_bases}")

    identity = payload.get("protocol_identity")
    digest = payload.get("protocol_digest_sha256")
    if not isinstance(identity, dict):
        errors.append("protocol_identity must be an object")
    else:
        expected_core = {
            "method": job.method,
            "dataset": job.dataset,
            "setting": SETTING,
            "seed": job.seed,
            "matched_length_bases": protocol.length_bases,
        }
        for key, expected in expected_core.items():
            if identity.get(key) != expected:
                errors.append(
                    f"protocol_identity.{key} must equal {expected!r}"
                )
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            errors.append(
                "protocol_digest_sha256 must be 64 lowercase hex digits"
            )
        elif _canonical_protocol_digest(identity) != digest:
            errors.append(
                "protocol_digest_sha256 does not bind protocol_identity"
            )
        if identity != expected_identity:
            errors.append(
                "protocol_identity does not match the current canonical "
                "driver/cache/code/source/projection/execution contract"
            )
        if digest != expected_digest:
            errors.append(
                "protocol_digest_sha256 does not match the current canonical "
                f"protocol (expected {expected_digest}, found {digest!r})"
            )

    expected_cell = path.parent.resolve()
    declared_cell = payload.get("cell_root")
    if (
        not isinstance(declared_cell, str)
        or Path(declared_cell).expanduser().resolve() != expected_cell
    ):
        errors.append("cell_root does not resolve to the manifest parent")
    for label, directory in (
        ("stage1_dir", "stage1"),
        ("refit_dir", "refit"),
    ):
        value = payload.get(label)
        expected = expected_cell / directory
        if (
            not isinstance(value, str)
            or Path(value).expanduser().resolve() != expected
            or not expected.is_dir()
        ):
            errors.append(
                f"{label} must resolve to the existing {expected.name}/"
            )

    access = payload.get("test_access_contract")
    if access != _TEST_ACCESS_CONTRACT:
        errors.append("test_access_contract does not match the sealed P0 contract")

    compliance = payload.get("post_compliance")
    if not isinstance(compliance, dict):
        errors.append("post_compliance must be an object")
    else:
        for split in ("query", "database"):
            value = compliance.get(split)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isclose(
                    float(value), 1.0, rel_tol=0.0, abs_tol=1e-12
                )
            ):
                errors.append(f"post_compliance.{split} must equal 1.0")

    return ("invalid" if errors else status), errors


def _matching_cell_paths(result_root: Path, job: Job) -> list[Path]:
    if not result_root.exists():
        return []
    if not result_root.is_dir():
        raise MatrixStateError(f"result root is not a directory: {result_root}")
    return sorted(
        path
        for path in result_root.iterdir()
        if path.name.startswith(job.cell_prefix)
    )


def _completed_cells(
    result_root: Path,
    jobs: Sequence[Job],
    *,
    allow_diagnostic: bool,
    options: LaunchOptions | None = None,
    expected_protocols: Mapping[
        tuple[str, str, int], tuple[Mapping[str, object], str]
    ] | None = None,
    protocol: "NativeProtocol | int | None" = None,
) -> dict[tuple[str, str, int], Path]:
    """Validate resumable cells and fail on every ambiguous partial state."""
    protocol = coerce_protocol(protocol)
    completed: dict[tuple[str, str, int], Path] = {}
    problems: list[str] = []
    suffix_re = re.compile(r"[0-9a-f]{12}")
    for job in jobs:
        valid: list[Path] = []
        for cell in _matching_cell_paths(result_root, job):
            suffix = cell.name[len(job.cell_prefix):]
            if not cell.is_dir() or suffix_re.fullmatch(suffix) is None:
                problems.append(
                    f"{job.job_id}: malformed existing cell path {cell}"
                )
                continue
            manifest = cell / MANIFEST_NAME
            if not manifest.is_file():
                problems.append(
                    f"{job.job_id}: incomplete existing cell lacks "
                    f"{MANIFEST_NAME}: {cell}"
                )
                continue
            expected = (
                None
                if expected_protocols is None
                else expected_protocols.get(job.key)
            )
            if expected is None and options is not None:
                expected = _expected_protocol(job, options)
            if expected is None:
                problems.append(
                    f"{job.job_id}: cannot validate an existing cell without "
                    "the expected current protocol identity"
                )
                continue
            expected_identity, expected_digest = expected
            if suffix != expected_digest[:12]:
                problems.append(
                    f"{job.job_id}: existing cell protocol suffix {suffix!r} "
                    f"does not match current {expected_digest[:12]!r}: {cell}"
                )
                continue
            status, errors = _manifest_contract_errors(
                manifest,
                job,
                expected_identity=expected_identity,
                expected_digest=expected_digest,
                protocol=protocol,
            )
            accepted = status == "complete_main_eligible" or (
                allow_diagnostic and status == "complete_diagnostic_only"
            )
            if not accepted:
                detail = "; ".join(errors) if errors else (
                    "diagnostic-only completion requires "
                    "--allow-main-ineligible-diagnostic"
                )
                problems.append(
                    f"{job.job_id}: existing manifest is not resumable "
                    f"({status}): {manifest}; {detail}"
                )
                continue
            valid.append(manifest.resolve())
        if len(valid) > 1:
            problems.append(
                f"{job.job_id}: duplicate completed cells are ambiguous: "
                + ", ".join(map(str, valid))
            )
        elif len(valid) == 1:
            completed[job.key] = valid[0]
    if problems:
        raise MatrixStateError(
            "native-DNA matrix preflight found existing incomplete/invalid "
            "state; use a fresh --data-root or explicitly archive the old "
            "cell(s):\n  " + "\n  ".join(problems)
        )
    return completed


def _command(job: Job, options: LaunchOptions) -> list[str]:
    command = [
        str(options.python),
        str(options.driver),
        "--method",
        job.method,
        "--dataset",
        job.dataset,
        "--setting",
        SETTING,
        "--dataset-root",
        str(options.dataset_root),
        "--result-root",
        str(options.result_root),
        "--device",
        "cuda:0",
        "--seed",
        str(job.seed),
        "--num-workers",
        str(options.num_workers),
        "--extract-batch-size",
        str(options.extract_batch_size),
        "--query-chunk",
        str(options.query_chunk),
    ]
    if job.method == "bee2021":
        if options.predictor is None:
            raise ValueError("bee2021 requires a frozen PRIMO predictor")
        command.extend(
            ["--primo-predictor-npz", str(options.predictor)]
        )
    if options.allow_diagnostic:
        command.append("--allow-main-ineligible-diagnostic")
    return command


def _validate_inputs(options: LaunchOptions, jobs: Sequence[Job]) -> None:
    if not options.python.is_file() or not os.access(options.python, os.X_OK):
        raise FileNotFoundError(
            f"Python interpreter is not executable: {options.python}"
        )
    if not options.driver.is_file():
        raise FileNotFoundError(
            f"native-DNA single-cell driver is missing: {options.driver}"
        )
    if options.driver.resolve() != RUNNER.resolve():
        raise ValueError(
            "sealed matrix execution requires the canonical current driver at "
            f"{RUNNER.resolve()}, found {options.driver.resolve()}"
        )
    if not options.dataset_root.is_dir():
        raise FileNotFoundError(
            f"dataset root is missing: {options.dataset_root}"
        )
    if any(job.method == "bee2021" for job in jobs):
        if options.predictor is None or not options.predictor.is_file():
            expected = options.result_root.parent / (
                "artifacts/primo_yield_predictor.npz"
            )
            raise FileNotFoundError(
                "bee2021 requires a frozen PRIMO predictor. Supply "
                "--primo-predictor-npz or place it at "
                f"{expected}"
            )


def _log_path(options: LaunchOptions, job: Job) -> Path:
    return options.log_root / f"{job.job_id}.log"


def _status_path(options: LaunchOptions, job: Job) -> Path:
    return options.log_root / "status" / f"{job.job_id}.json"


def _status_payload(
    job: Job,
    gpu_id: str,
    command: Sequence[str],
    log_path: Path,
    *,
    state: str,
    started_at: str,
    returncode: int | None = None,
    message: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "state": state,
        "job_id": job.job_id,
        "method": job.method,
        "dataset": job.dataset,
        "seed": job.seed,
        "physical_gpu": gpu_id,
        "logical_device": "cuda:0",
        "started_at_utc": started_at,
        "updated_at_utc": _utc_now(),
        "command": list(command),
        "log": str(log_path),
        "returncode": returncode,
    }
    if message is not None:
        payload["message"] = message
    return payload


def _run_cell(
    job: Job,
    gpu_id: str,
    options: LaunchOptions,
) -> RunResult:
    command = _command(job, options)
    log_path = _log_path(options, job)
    status_path = _status_path(options, job)
    started_at = _utc_now()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        status_path,
        _status_payload(
            job,
            gpu_id,
            command,
            log_path,
            state="running",
            started_at=started_at,
        ),
    )
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = gpu_id
    environment["PYTHONUNBUFFERED"] = "1"
    returncode = 1
    message: str | None = None
    try:
        with log_path.open("x", encoding="utf-8") as handle:
            handle.write(
                f"# physical_gpu={gpu_id} logical_device=cuda:0\n"
                f"# command={shlex.join(command)}\n"
            )
            handle.flush()
            process = subprocess.run(
                command,
                cwd=REPO,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
                start_new_session=True,
            )
            returncode = int(process.returncode)
        if returncode == 0:
            try:
                completed = _completed_cells(
                    options.result_root,
                    [job],
                    allow_diagnostic=options.allow_diagnostic,
                    options=options,
                )
                if job.key not in completed:
                    returncode = 3
                    message = (
                        "child exited zero but produced no protocol-valid "
                        "completed manifest"
                    )
            except MatrixStateError as error:
                returncode = 3
                message = str(error)
    except Exception as error:  # preserve failure while the pool continues
        returncode = 1
        message = f"{type(error).__name__}: {error}"

    state = "completed" if returncode == 0 else "failed"
    _atomic_json(
        status_path,
        _status_payload(
            job,
            gpu_id,
            command,
            log_path,
            state=state,
            started_at=started_at,
            returncode=returncode,
            message=message,
        ),
    )
    return RunResult(job, gpu_id, returncode, log_path, message)


def _run_pool(
    jobs: Sequence[Job],
    gpus: Sequence[str],
    run_one: Callable[[Job, str], RunResult],
) -> list[RunResult]:
    """Run all jobs; an individual failure never stops queue consumption."""
    work: queue.Queue[Job] = queue.Queue()
    for job in jobs:
        work.put(job)
    results: list[RunResult] = []
    result_lock = threading.Lock()

    def worker(gpu_id: str) -> None:
        while True:
            try:
                job = work.get_nowait()
            except queue.Empty:
                return
            try:
                result = run_one(job, gpu_id)
            except Exception as error:
                result = RunResult(
                    job,
                    gpu_id,
                    1,
                    Path(),
                    f"{type(error).__name__}: {error}",
                )
            with result_lock:
                results.append(result)
            work.task_done()

    threads = [
        threading.Thread(target=worker, args=(gpu,), name=f"native-p0-gpu-{gpu}")
        for gpu in gpus
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return sorted(results, key=lambda item: item.job.job_id)


def build_parser() -> argparse.ArgumentParser:
    default_python = Path(
        "/home/yschoi/.conda/envs/dna_hashing/bin/python"
    )
    if not default_python.is_file():
        default_python = Path(sys.executable)
    parser = argparse.ArgumentParser(
        description="Launch the sealed native-DNA P0 GPU matrix."
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help=(
            "Dedicated absolute directory below /data. Runs use ROOT/runs, "
            "launcher records use ROOT/logs, and the default PRIMO artifact "
            "is ROOT/artifacts/primo_yield_predictor.npz."
        ),
    )
    parser.add_argument(
        "--gpus",
        nargs="+",
        required=True,
        help="Unique physical GPU IDs; each child sees its GPU as cuda:0.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=DEFAULT_METHODS,
        default=list(DEFAULT_METHODS),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DEFAULT_DATASETS,
        default=list(DEFAULT_DATASETS),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        choices=DEFAULT_SEEDS,
        default=list(DEFAULT_SEEDS),
    )
    parser.add_argument("--python", default=str(default_python))
    parser.add_argument("--driver", default=str(RUNNER))
    parser.add_argument("--dataset-root", default=str(REPO / "dataset"))
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--extract-batch-size", type=int, default=512)
    parser.add_argument("--query-chunk", type=int, default=64)
    parser.add_argument("--primo-predictor-npz", default=None)
    parser.add_argument(
        "--allow-main-ineligible-diagnostic",
        action="store_true",
        help=(
            "Pass the explicit legacy-cache diagnostic opt-in to every cell "
            "and allow completed diagnostic manifests to be resumed."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Audit and print commands without creating files or launching children.",
    )
    return parser


def _unique_or_error(
    parser: argparse.ArgumentParser,
    name: str,
    values: Sequence[object],
) -> None:
    if not values or len(values) != len(set(values)):
        parser.error(f"--{name} must contain at least one unique value")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        data_root = _validate_data_root(args.data_root)
    except ValueError as error:
        parser.error(str(error))
    _unique_or_error(parser, "gpus", args.gpus)
    _unique_or_error(parser, "methods", args.methods)
    _unique_or_error(parser, "datasets", args.datasets)
    _unique_or_error(parser, "seeds", args.seeds)
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    if args.extract_batch_size <= 0 or args.query_chunk <= 0:
        parser.error("extraction batch/chunk sizes must be positive")

    result_root = data_root / "runs"
    log_root = data_root / "logs"
    default_predictor = data_root / "artifacts/primo_yield_predictor.npz"
    predictor = (
        _absolute(args.primo_predictor_npz)
        if args.primo_predictor_npz is not None
        else default_predictor if default_predictor.is_file() else None
    )
    options = LaunchOptions(
        python=_absolute(args.python),
        driver=_absolute(args.driver),
        dataset_root=_absolute(args.dataset_root),
        result_root=result_root,
        log_root=log_root,
        predictor=predictor,
        num_workers=args.num_workers,
        extract_batch_size=args.extract_batch_size,
        query_chunk=args.query_chunk,
        allow_diagnostic=args.allow_main_ineligible_diagnostic,
    )
    jobs = _jobs(args.methods, args.datasets, args.seeds)
    _validate_inputs(options, jobs)
    completed = _completed_cells(
        result_root,
        jobs,
        allow_diagnostic=options.allow_diagnostic,
        options=options,
    )
    pending = [job for job in jobs if job.key not in completed]

    if args.dry_run:
        print(
            f"[native-p0] scheduled={len(pending)} "
            f"completed_skipped={len(completed)} "
            f"gpu_pool={','.join(args.gpus)}"
        )
        for index, job in enumerate(pending):
            gpu = args.gpus[index % len(args.gpus)]
            print(
                f"CUDA_VISIBLE_DEVICES={shlex.quote(gpu)} "
                + shlex.join(_command(job, options))
            )
        return 0

    collisions = [
        path
        for job in pending
        for path in (_log_path(options, job), _status_path(options, job))
        if path.exists()
    ]
    if collisions:
        raise MatrixStateError(
            "pending cells have existing launcher logs/status and will not be "
            "overwritten:\n  " + "\n  ".join(map(str, collisions))
        )
    if not pending:
        print(
            f"[native-p0] no pending cells; skipped {len(completed)} completed",
            flush=True,
        )
        return 0

    log_root.mkdir(parents=True, exist_ok=True)
    lock_path = log_root / "launcher.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(
                lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
            )
        except BlockingIOError as error:
            raise RuntimeError(
                f"another native-DNA launcher holds {lock_path}"
            ) from error

        created = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        plan_path = log_root / f"matrix_plan_{created}_{os.getpid()}.json"
        _atomic_json(
            plan_path,
            {
                "schema_version": 1,
                "created_at_utc": _utc_now(),
                "data_root": str(data_root),
                "result_root": str(result_root),
                "log_root": str(log_root),
                "gpu_pool": list(args.gpus),
                "completed_cells_skipped": {
                    "|".join(map(str, key)): str(path)
                    for key, path in sorted(completed.items())
                },
                "jobs": [
                    {
                        "job_id": job.job_id,
                        "method": job.method,
                        "dataset": job.dataset,
                        "seed": job.seed,
                        "command_without_gpu_environment": _command(job, options),
                        "log": str(_log_path(options, job)),
                    }
                    for job in pending
                ],
            },
        )
        print(
            f"[native-p0] plan={plan_path} pending={len(pending)} "
            f"completed_skipped={len(completed)} "
            f"gpus={','.join(args.gpus)}",
            flush=True,
        )
        output_lock = threading.Lock()

        def run_one(job: Job, gpu_id: str) -> RunResult:
            with output_lock:
                print(
                    f"[native-p0][gpu {gpu_id}] START {job.job_id} "
                    f"log={_log_path(options, job)}",
                    flush=True,
                )
            result = _run_cell(job, gpu_id, options)
            state = "COMPLETED" if result.returncode == 0 else "FAILED"
            with output_lock:
                print(
                    f"[native-p0][gpu {gpu_id}] {state} {job.job_id} "
                    f"rc={result.returncode}",
                    flush=True,
                )
            return result

        results = _run_pool(pending, args.gpus, run_one)
        failures = [item for item in results if item.returncode != 0]
        summary_path = log_root / (
            f"matrix_summary_{created}_{os.getpid()}.json"
        )
        _atomic_json(
            summary_path,
            {
                "schema_version": 1,
                "finished_at_utc": _utc_now(),
                "plan": str(plan_path),
                "scheduled": len(pending),
                "completed": len(results) - len(failures),
                "failures": [
                    {
                        "job_id": item.job.job_id,
                        "gpu": item.gpu_id,
                        "returncode": item.returncode,
                        "log": str(item.log_path),
                        "message": item.message,
                    }
                    for item in failures
                ],
            },
        )
        print(
            f"[native-p0] finished failures={len(failures)} "
            f"summary={summary_path}",
            flush=True,
        )
        return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
