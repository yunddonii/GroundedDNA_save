"""Fail-closed execution-environment provenance for baseline D6 children.

The matrix launcher knows which *physical* GPU it assigned, but only the
runner/trainer child can attest to the Python packages and logical CUDA device
that it actually sees.  This module deliberately contains provenance code
only: it does not configure CUDA, import model code, or alter training.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence

from dna_utils.runtime_environment import semantic_digest


EXECUTION_ENVIRONMENT_SCHEMA = (
    "groundeddna.baseline-child-execution-environment")
EXECUTION_ENVIRONMENT_SCHEMA_VERSION = 1
PACKAGE_DISTRIBUTIONS = {
    "torch": "torch",
    "transformers": "transformers",
    "numpy": "numpy",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "pandas": "pandas",
    "timm": "timm",
    "tensorboard": "tensorboard",
    "tqdm": "tqdm",
    "Pillow": "Pillow",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GPU_UUID_RE = re.compile(r"^GPU-[A-Za-z0-9-]+$")
_ASSIGNMENT_KEYS = frozenset({
    "physical_index", "uuid", "name", "pci_bus_id", "driver_version",
})
_TOP_LEVEL_KEYS = frozenset({
    "schema", "schema_version", "python", "packages", "cuda_runtime",
    "cuda_visible_devices", "torch_visible_device_count",
    "logical_device_0", "assigned_physical_gpu",
})


class ExecutionEnvironmentError(ValueError):
    """The D6 child environment is incomplete or differs from its assignment."""


def canonical_digest(payload: Mapping[str, Any]) -> str:
    """Use the repository-wide canonical environment digest primitive."""
    return semantic_digest(payload)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def query_gpu_inventory(*, run: Callable[..., Any] = subprocess.run) \
        -> list[dict[str, Any]]:
    """Return the stable physical GPU identity reported by ``nvidia-smi``."""
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,pci.bus_id,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        process = run(
            command, text=True, capture_output=True, check=False,
        )
    except Exception as error:  # provenance failures must remain explicit
        raise ExecutionEnvironmentError(
            f"cannot execute nvidia-smi GPU identity query: {error}") from error
    if int(process.returncode) != 0:
        detail = str(getattr(process, "stderr", "")).strip()
        raise ExecutionEnvironmentError(
            "nvidia-smi GPU identity query failed: "
            f"{detail or f'exit {process.returncode}'}")
    rows: list[dict[str, Any]] = []
    try:
        reader = csv.reader(io.StringIO(str(process.stdout)))
        for raw in reader:
            fields = [field.strip() for field in raw]
            if not fields:
                continue
            if len(fields) != 5:
                raise ExecutionEnvironmentError(
                    f"malformed nvidia-smi row: {raw!r}")
            index_text, uuid, name, pci_bus_id, driver_version = fields
            index = int(index_text)
            if index < 0 or _GPU_UUID_RE.fullmatch(uuid) is None:
                raise ExecutionEnvironmentError(
                    f"invalid nvidia-smi GPU identity row: {raw!r}")
            rows.append({
                "physical_index": index,
                "uuid": uuid,
                "name": name,
                "pci_bus_id": pci_bus_id,
                "driver_version": driver_version,
            })
    except (TypeError, ValueError) as error:
        raise ExecutionEnvironmentError(
            f"cannot parse nvidia-smi GPU identity output: {error}") from error
    if not rows:
        raise ExecutionEnvironmentError("nvidia-smi returned no GPUs")
    indices = [row["physical_index"] for row in rows]
    uuids = [row["uuid"] for row in rows]
    if len(indices) != len(set(indices)) or len(uuids) != len(set(uuids)):
        raise ExecutionEnvironmentError(
            "nvidia-smi returned duplicate physical indices or UUIDs")
    return rows


def validate_gpu_assignment(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _ASSIGNMENT_KEYS:
        raise ExecutionEnvironmentError(
            "GPU assignment must contain exactly physical_index, uuid, name, "
            "pci_bus_id, and driver_version")
    index = value.get("physical_index")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ExecutionEnvironmentError("GPU physical_index must be non-negative int")
    normalized = {key: value.get(key) for key in _ASSIGNMENT_KEYS}
    for key in ("uuid", "name", "pci_bus_id", "driver_version"):
        if not isinstance(normalized[key], str) or not normalized[key]:
            raise ExecutionEnvironmentError(f"GPU assignment {key} is missing")
    if _GPU_UUID_RE.fullmatch(str(normalized["uuid"])) is None:
        raise ExecutionEnvironmentError("GPU assignment uuid is invalid")
    return {
        "physical_index": index,
        "uuid": normalized["uuid"],
        "name": normalized["name"],
        "pci_bus_id": normalized["pci_bus_id"],
        "driver_version": normalized["driver_version"],
    }


def resolve_gpu_assignments(
        gpu_ids: Sequence[str | int], *,
        inventory: Sequence[Mapping[str, Any]] | None = None,
        ) -> dict[str, dict[str, Any]]:
    """Resolve user physical indices to exact index+UUID assignment records."""
    available = (
        query_gpu_inventory() if inventory is None
        else [validate_gpu_assignment(row) for row in inventory])
    by_index = {row["physical_index"]: row for row in available}
    resolved: dict[str, dict[str, Any]] = {}
    seen_indices: set[int] = set()
    for raw in gpu_ids:
        token = str(raw)
        if re.fullmatch(r"0|[1-9][0-9]*", token) is None:
            raise ExecutionEnvironmentError(
                f"physical GPU IDs must be canonical decimal indices, got {token!r}")
        index = int(token)
        if index in seen_indices:
            raise ExecutionEnvironmentError(f"duplicate physical GPU index {index}")
        if index not in by_index:
            raise ExecutionEnvironmentError(
                f"nvidia-smi inventory has no physical GPU index {index}")
        resolved[token] = dict(by_index[index])
        seen_indices.add(index)
    if not resolved:
        raise ExecutionEnvironmentError("at least one physical GPU is required")
    return resolved


def parse_gpu_assignment_json(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ExecutionEnvironmentError(
            f"invalid matrix GPU assignment JSON: {error}") from error
    return validate_gpu_assignment(value)


def _package_records(
        version: Callable[[str], str] = importlib.metadata.version,
        ) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for logical, distribution in PACKAGE_DISTRIBUTIONS.items():
        try:
            value = version(distribution)
        except importlib.metadata.PackageNotFoundError:
            records[logical] = {
                "distribution": distribution,
                "installed": False,
                "version": None,
            }
        else:
            records[logical] = {
                "distribution": distribution,
                "installed": True,
                "version": str(value),
            }
    return records


def validate_execution_environment(value: object) -> dict[str, Any]:
    """Validate the exact JSON schema and return a detached normalized copy."""
    if not isinstance(value, Mapping) or set(value) != _TOP_LEVEL_KEYS:
        raise ExecutionEnvironmentError(
            "execution environment has a missing or unexpected top-level field")
    if value.get("schema") != EXECUTION_ENVIRONMENT_SCHEMA \
            or value.get("schema_version") != EXECUTION_ENVIRONMENT_SCHEMA_VERSION:
        raise ExecutionEnvironmentError("execution environment schema mismatch")
    python = value.get("python")
    expected_python_keys = {
        "executable", "executable_sha256", "implementation", "version",
        "version_info",
    }
    if not isinstance(python, Mapping) or set(python) != expected_python_keys:
        raise ExecutionEnvironmentError("execution environment python block invalid")
    if any(not isinstance(python.get(key), str) or not python.get(key)
           for key in ("executable", "implementation", "version")):
        raise ExecutionEnvironmentError("execution environment python strings invalid")
    if not isinstance(python.get("executable_sha256"), str) \
            or _SHA256_RE.fullmatch(str(python["executable_sha256"])) is None:
        raise ExecutionEnvironmentError("python executable SHA-256 invalid")
    version_info = python.get("version_info")
    if not isinstance(version_info, list) or len(version_info) != 5 \
            or any(isinstance(item, bool) or not isinstance(item, int)
                   for item in (*version_info[:3], version_info[4])) \
            or not isinstance(version_info[3], str):
        raise ExecutionEnvironmentError("python version_info invalid")

    packages = value.get("packages")
    if not isinstance(packages, Mapping) \
            or set(packages) != set(PACKAGE_DISTRIBUTIONS):
        raise ExecutionEnvironmentError("execution environment package set invalid")
    for logical, distribution in PACKAGE_DISTRIBUTIONS.items():
        record = packages.get(logical)
        if not isinstance(record, Mapping) \
                or set(record) != {"distribution", "installed", "version"} \
                or record.get("distribution") != distribution \
                or not isinstance(record.get("installed"), bool):
            raise ExecutionEnvironmentError(
                f"execution environment package record invalid: {logical}")
        installed = record.get("installed")
        package_version = record.get("version")
        if installed is True and (not isinstance(package_version, str)
                                  or not package_version):
            raise ExecutionEnvironmentError(
                f"installed package version missing: {logical}")
        if installed is False and package_version is not None:
            raise ExecutionEnvironmentError(
                f"missing package must have null version: {logical}")

    cuda_runtime = value.get("cuda_runtime")
    if not isinstance(cuda_runtime, Mapping) \
            or set(cuda_runtime) != {
                "torch_version", "torch_cuda", "cudnn_version"}:
        raise ExecutionEnvironmentError("execution environment CUDA runtime invalid")
    if not isinstance(cuda_runtime.get("torch_version"), str) \
            or not cuda_runtime.get("torch_version"):
        raise ExecutionEnvironmentError("imported torch version missing")
    if packages["torch"].get("installed") is not True \
            or packages["torch"].get("version") \
            != cuda_runtime.get("torch_version"):
        raise ExecutionEnvironmentError(
            "imported torch version disagrees with package metadata")
    if not isinstance(cuda_runtime.get("torch_cuda"), str) \
            or not cuda_runtime.get("torch_cuda"):
        raise ExecutionEnvironmentError("torch CUDA runtime version missing")
    cudnn = cuda_runtime.get("cudnn_version")
    if isinstance(cudnn, bool) or not isinstance(cudnn, int) or cudnn <= 0:
        raise ExecutionEnvironmentError("cuDNN runtime version missing")
    if not isinstance(value.get("cuda_visible_devices"), str) \
            or not value.get("cuda_visible_devices"):
        raise ExecutionEnvironmentError("CUDA_VISIBLE_DEVICES is missing")
    count = value.get("torch_visible_device_count")
    if isinstance(count, bool) or not isinstance(count, int) or count != 1:
        raise ExecutionEnvironmentError(
            "D6 child must see exactly one torch CUDA device")
    logical = value.get("logical_device_0")
    logical_keys = {
        "logical_index", "name", "total_memory", "major", "minor",
        "multi_processor_count", "uuid",
    }
    if not isinstance(logical, Mapping) or set(logical) != logical_keys:
        raise ExecutionEnvironmentError("logical CUDA device 0 block invalid")
    if logical.get("logical_index") != 0 \
            or not isinstance(logical.get("name"), str) \
            or not logical.get("name"):
        raise ExecutionEnvironmentError("logical CUDA device 0 identity invalid")
    for key in ("total_memory", "major", "minor", "multi_processor_count"):
        item = logical.get(key)
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ExecutionEnvironmentError(
                f"logical CUDA device property invalid: {key}")
    if logical.get("uuid") is not None \
            and not isinstance(logical.get("uuid"), str):
        raise ExecutionEnvironmentError("logical CUDA device UUID invalid")
    assignment = validate_gpu_assignment(value.get("assigned_physical_gpu"))
    if value.get("cuda_visible_devices") != assignment["uuid"]:
        raise ExecutionEnvironmentError(
            "CUDA_VISIBLE_DEVICES is not the assigned physical GPU UUID")
    if logical.get("name") != assignment["name"]:
        raise ExecutionEnvironmentError(
            "logical CUDA device name does not match assigned physical GPU")
    normalized = json.loads(json.dumps(value, allow_nan=False))
    normalized["assigned_physical_gpu"] = assignment
    return normalized


def verify_execution_environment(
        value: object, digest: object) -> dict[str, Any]:
    normalized = validate_execution_environment(value)
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
        raise ExecutionEnvironmentError("execution environment SHA-256 invalid")
    actual = canonical_digest(normalized)
    if actual != digest:
        raise ExecutionEnvironmentError(
            "execution environment canonical SHA-256 mismatch")
    return normalized


def capture_child_execution_environment(
        assignment: Mapping[str, Any], *,
        torch_module: Any | None = None,
        inventory: Sequence[Mapping[str, Any]] | None = None,
        version: Callable[[str], str] = importlib.metadata.version,
        executable: str | None = None,
        environ: Mapping[str, str] | None = None,
        ) -> tuple[dict[str, Any], str]:
    """Recompute the child environment and verify its physical assignment."""
    expected_assignment = validate_gpu_assignment(assignment)
    current_inventory = (
        query_gpu_inventory() if inventory is None
        else [validate_gpu_assignment(row) for row in inventory])
    physical_matches = [
        row for row in current_inventory
        if row["physical_index"] == expected_assignment["physical_index"]
    ]
    if physical_matches != [expected_assignment]:
        raise ExecutionEnvironmentError(
            "assigned physical GPU index/UUID/properties differ from current "
            "nvidia-smi inventory")
    environment = os.environ if environ is None else environ
    visible = environment.get("CUDA_VISIBLE_DEVICES")
    if visible != expected_assignment["uuid"]:
        raise ExecutionEnvironmentError(
            "CUDA_VISIBLE_DEVICES differs from the matrix-assigned GPU UUID")

    if torch_module is None:
        import torch as torch_module  # type: ignore[no-redef]
    if not bool(torch_module.cuda.is_available()):
        raise ExecutionEnvironmentError("CUDA is unavailable in D6 child")
    count = int(torch_module.cuda.device_count())
    if count != 1:
        raise ExecutionEnvironmentError(
            f"D6 child sees {count} torch CUDA devices; expected exactly 1")
    properties = torch_module.cuda.get_device_properties(0)
    logical_name = str(torch_module.cuda.get_device_name(0))
    if logical_name != expected_assignment["name"]:
        raise ExecutionEnvironmentError(
            "torch logical device 0 name differs from assigned physical GPU")
    torch_cuda = getattr(getattr(torch_module, "version", None), "cuda", None)
    cudnn_version = torch_module.backends.cudnn.version()
    if torch_cuda is None or cudnn_version is None:
        raise ExecutionEnvironmentError(
            "torch CUDA/cuDNN runtime metadata is unavailable")

    executable_path = Path(
        os.path.realpath(sys.executable if executable is None else executable))
    if not executable_path.is_file():
        raise ExecutionEnvironmentError(
            f"Python executable is not a file: {executable_path}")
    property_uuid = getattr(properties, "uuid", None)
    payload = {
        "schema": EXECUTION_ENVIRONMENT_SCHEMA,
        "schema_version": EXECUTION_ENVIRONMENT_SCHEMA_VERSION,
        "python": {
            "executable": str(executable_path),
            "executable_sha256": _sha256_file(executable_path),
            "implementation": platform.python_implementation(),
            "version": sys.version,
            "version_info": [
                int(sys.version_info.major), int(sys.version_info.minor),
                int(sys.version_info.micro), str(sys.version_info.releaselevel),
                int(sys.version_info.serial),
            ],
        },
        "packages": _package_records(version),
        "cuda_runtime": {
            "torch_version": str(torch_module.__version__),
            "torch_cuda": str(torch_cuda),
            "cudnn_version": int(cudnn_version),
        },
        "cuda_visible_devices": visible,
        "torch_visible_device_count": count,
        "logical_device_0": {
            "logical_index": 0,
            "name": logical_name,
            "total_memory": int(properties.total_memory),
            "major": int(properties.major),
            "minor": int(properties.minor),
            "multi_processor_count": int(properties.multi_processor_count),
            "uuid": None if property_uuid is None else str(property_uuid),
        },
        "assigned_physical_gpu": expected_assignment,
    }
    normalized = validate_execution_environment(payload)
    return normalized, canonical_digest(normalized)


def require_exact_environment(
        expected: object, expected_digest: object, *,
        actual: object, actual_digest: object) -> dict[str, Any]:
    """Validate two attestations and require exact, type-sensitive equality."""
    expected_normalized = verify_execution_environment(
        expected, expected_digest)
    actual_normalized = verify_execution_environment(actual, actual_digest)
    if actual_normalized != expected_normalized or actual_digest != expected_digest:
        changed = sorted(
            key for key in _TOP_LEVEL_KEYS
            if actual_normalized.get(key) != expected_normalized.get(key))
        raise ExecutionEnvironmentError(
            f"child execution environment drifted after runner seal: {changed}")
    return actual_normalized
