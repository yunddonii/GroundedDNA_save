"""Exact execution-environment attestations for Phase-3 trainer children.

The campaign launcher sees physical GPU indices, while a sanitized child sees
one logical CUDA device (index 0).  A parent-side digest copied into the child
does not prove that the child actually imported the same Python packages or
ran on the GPU the plan assigned.  This module is intentionally independent of
the model/training modules so it can be used at the trainer admission boundary
before a dataset or model is constructed.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
PACKAGE_DISTRIBUTIONS = {
    "torch": ("torch",),
    "torchvision": ("torchvision",),
    "numpy": ("numpy",),
    "scipy": ("scipy",),
    "sklearn": ("scikit-learn",),
    "transformers": ("transformers",),
    "tokenizers": ("tokenizers",),
    "Pillow": ("Pillow",),
    "opencv": ("opencv-python", "opencv-python-headless"),
    "pandas": ("pandas",),
    "timm": ("timm",),
    "tensorboard": ("tensorboard",),
    "tqdm": ("tqdm",),
    "PyYAML": ("PyYAML",),
    "matplotlib": ("matplotlib",),
}
RUNTIME_ENV_KEYS = (
    "LD_LIBRARY_PATH", "HF_HOME", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
)


def caller_environment() -> dict:
    """The runtime variables as the CALLER supplied them, not as imports left them.

    Read from `/proc/self/environ`, which holds the block this process was
    exec'd with and is immutable for its lifetime, rather than from
    `os.environ`, which any imported package may rewrite.

    `cv2` rewrites `LD_LIBRARY_PATH` on import, prepending its bundled lib
    directory, and it does so unconditionally -- importing it twice yields the
    directory twice. The trainer's import chain pulls cv2 in; the launcher's
    does not. Reading `os.environ` therefore compared a parent that had never
    imported cv2 against a child that had, so the two could never agree and
    every cell was refused with `library_environment`. Presetting the variable
    does not help either, because the child prepends on top of whatever it
    inherits: the parent is always exactly one prepend behind.

    What the attestation is actually for is the environment the launcher hands
    the child, and that is what the exec block records. A caller supplying a
    different `LD_LIBRARY_PATH` is still caught; a library rewriting its own
    copy in memory is not mistaken for one.
    """
    try:
        with open("/proc/self/environ", "rb") as handle:
            block = handle.read()
    except OSError:
        # Non-Linux or a hidden /proc. Fall back to os.environ and accept that
        # an import-time rewrite would be visible; nothing else here is
        # portable either, and the callers all run on this host.
        return {key: os.environ.get(key) for key in RUNTIME_ENV_KEYS}
    # The block can repeat a name. `getenv(3)` returns the FIRST match, so
    # first-wins is the value the process actually runs under; last-wins would
    # attest something no library ever reads.
    supplied: dict = {}
    for entry in block.split(b"\0"):
        if not entry or b"=" not in entry:
            continue
        name, _, value = entry.partition(b"=")
        try:
            supplied.setdefault(name.decode(), value.decode())
        except UnicodeDecodeError:
            continue
    return {key: supplied.get(key) for key in RUNTIME_ENV_KEYS}


class EnvironmentAttestationError(RuntimeError):
    """The actual child environment is not the one sealed by its plan."""


def semantic_digest(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_versions(errors: list[str]) -> dict:
    packages = {}
    for logical, candidates in PACKAGE_DISTRIBUTIONS.items():
        found = None
        for distribution in candidates:
            try:
                found = {
                    "distribution": distribution,
                    "version": importlib.metadata.version(distribution),
                }
                break
            except importlib.metadata.PackageNotFoundError:
                continue
        if found is None:
            errors.append(f"missing package metadata for {logical}")
        packages[logical] = found
    return packages


def _torch_runtime(errors: list[str], *, visible: bool) -> tuple[dict, dict | None]:
    runtime = {"torch_cuda": None, "cudnn_version": None}
    device = None
    try:
        import torch

        runtime = {
            "torch_cuda": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
        }
        if runtime["torch_cuda"] is None or runtime["cudnn_version"] is None:
            errors.append("torch CUDA/cuDNN runtime metadata is unavailable")
        if visible:
            count = int(torch.cuda.device_count())
            device = {
                "device_count": count,
                "logical_index": 0,
                "name": torch.cuda.get_device_name(0) if count else None,
            }
            if count != 1:
                errors.append(
                    f"Phase-3 child sees {count} CUDA devices, expected exactly 1"
                )
    except Exception as error:  # noqa: BLE001 - provenance must name failures
        errors.append(f"cannot query torch runtime: {error}")
    return runtime, device


def _gpu_inventory(errors: list[str]) -> list[dict]:
    try:
        proc = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,uuid,name,pci.bus_id,driver_version",
             "--format=csv,noheader,nounits"],
            text=True, capture_output=True, check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or f"exit {proc.returncode}")
        inventory = []
        for line in proc.stdout.splitlines():
            fields = [field.strip() for field in line.split(",", 4)]
            if len(fields) != 5:
                raise RuntimeError(f"malformed nvidia-smi row {line!r}")
            inventory.append({
                "index": int(fields[0]), "uuid": fields[1],
                "name": fields[2], "pci_bus_id": fields[3],
                "driver": fields[4],
            })
        return inventory
    except Exception as error:  # noqa: BLE001 - fail closed through errors
        errors.append(f"cannot query nvidia-smi GPU identity: {error}")
        return []


def capture_parent_environment(gpu_ids: Sequence[int] = ()) -> dict:
    """Capture the launcher environment and the requested physical GPUs."""
    errors: list[str] = []
    packages = _package_versions(errors)
    runtime, _ = _torch_runtime(errors, visible=False)
    inventory = _gpu_inventory(errors)
    requested = [int(gpu) for gpu in gpu_ids]
    by_index = {row["index"]: row for row in inventory}
    missing = sorted(set(requested) - set(by_index))
    if missing:
        errors.append(f"nvidia-smi has no requested GPU indices {missing}")
    if not requested:
        errors.append("campaign declares no selected GPU indices")
    executable = os.path.realpath(sys.executable)
    return {
        "schema_version": SCHEMA_VERSION,
        "python_executable": executable,
        "python_executable_sha256": _sha256_file(executable),
        "python_version": sys.version,
        "packages": packages,
        "torch_runtime": runtime,
        "requested_gpu_indices": requested,
        "selected_gpus": [by_index[index] for index in requested
                          if index in by_index],
        "caller_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "library_environment": caller_environment(),
        "pythonpath": os.environ.get("PYTHONPATH"),
        "errors": errors,
    }


def expected_child_environment(parent: Mapping[str, Any],
                               physical_gpu_index: int) -> dict:
    """Project a parent fingerprint into one exact sanitized-child contract."""
    index = int(physical_gpu_index)
    matches = [row for row in parent.get("selected_gpus", ())
               if row.get("index") == index]
    if len(matches) != 1:
        raise EnvironmentAttestationError(
            f"parent fingerprint does not contain physical GPU {index} exactly once"
        )
    physical = dict(matches[0])
    return {
        "schema_version": SCHEMA_VERSION,
        "parent_environment_sha256": semantic_digest(parent),
        "python_executable": parent.get("python_executable"),
        "python_executable_sha256": parent.get("python_executable_sha256"),
        "python_version": parent.get("python_version"),
        "packages": parent.get("packages"),
        "torch_runtime": parent.get("torch_runtime"),
        "library_environment": parent.get("library_environment"),
        "pythonpath": None,
        # UUID spelling removes CUDA_DEVICE_ORDER/numeric-enumeration
        # ambiguity and directly names the physical device the plan sealed.
        "cuda_visible_devices": physical.get("uuid"),
        "physical_gpu": physical,
        "torch_visible_device": {
            "device_count": 1, "logical_index": 0,
            "name": physical.get("name"),
        },
        "errors": [],
    }


def capture_child_environment(expected: Mapping[str, Any]) -> dict:
    """Recompute the normalized child payload from the running process."""
    errors: list[str] = []
    packages = _package_versions(errors)
    runtime, visible = _torch_runtime(errors, visible=True)
    inventory = _gpu_inventory(errors)
    physical_expected = expected.get("physical_gpu") or {}
    index = physical_expected.get("index")
    matches = [row for row in inventory if row.get("index") == index]
    if len(matches) != 1:
        errors.append(
            f"nvidia-smi does not contain assigned physical GPU {index} exactly once"
        )
        physical = None
    else:
        physical = matches[0]
    executable = os.path.realpath(sys.executable)
    return {
        "schema_version": SCHEMA_VERSION,
        "parent_environment_sha256": expected.get(
            "parent_environment_sha256"),
        "python_executable": executable,
        "python_executable_sha256": _sha256_file(executable),
        "python_version": sys.version,
        "packages": packages,
        "torch_runtime": runtime,
        "library_environment": caller_environment(),
        "pythonpath": os.environ.get("PYTHONPATH"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_gpu": physical,
        "torch_visible_device": visible,
        "errors": errors,
    }


def verify_child_environment(expected: Mapping[str, Any], *,
                             actual: Mapping[str, Any] | None = None) -> dict:
    """Return actual bytes only when every child field matches the plan."""
    expected_copy = json.loads(json.dumps(expected, allow_nan=False))
    observed = (capture_child_environment(expected_copy) if actual is None
                else json.loads(json.dumps(actual, allow_nan=False)))
    if observed != expected_copy:
        changed = sorted(
            key for key in set(expected_copy) | set(observed)
            if expected_copy.get(key) != observed.get(key)
        )
        raise EnvironmentAttestationError(
            f"Phase-3 child environment differs from its plan: {changed}"
        )
    if observed.get("errors"):
        raise EnvironmentAttestationError(
            f"Phase-3 child environment is incomplete: {observed['errors']}"
        )
    return observed
