#!/usr/bin/env python
"""Apply the biological-constraint post-projection to every method and re-evaluate.

The paper is about DNA hashing, so the emitted codes must be valid DNA. This
runs the mandatory post-processing step (GC in [40,60]%, homopolymer run <= 3;
Hamming-minimum DP projection of violators; valid codes untouched) on BOTH our
model and the baselines, in the SAME requested 18/24-base DNA space, and
re-computes base
mAP@R. Baselines are the E*-matched DNA-space extractions from 2026-07-21.

For each (method, dataset) it reports:
    pre_compliance    fraction of DB codes already valid before projection
    mean_edit         mean Hamming edits applied (over all DB codes)
    base mAP@R pre    base Hamming mAP@R on the raw codes  (sanity vs prior)
    base mAP@R post   base Hamming mAP@R after projecting BOTH query and DB

Paper-grade cells are fail-closed: extraction schema, projection feasibility,
post-projection compliance, and checkpoint/protocol provenance are validated;
all consumed and emitted artifacts are content-hashed.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dna_utils.bio_constraints import (          # noqa: E402
    is_valid_batch, project_to_valid,
    DEFAULT_GC_MIN_FRAC, DEFAULT_GC_MAX_FRAC, DEFAULT_MAX_HOMOPOLYMER_RUN,
    PROJECTION_TIE_POLICY,
)
from baseline.base_model import _ap_at_r, _multi_hot_relevance  # noqa: E402


REPO = Path(__file__).resolve().parents[1]
RETRIEVAL_TIE_POLICY = "stable_database_order_numpy_stable_v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: os.PathLike[str] | str, payload: dict) -> str:
    """Atomically replace a JSON manifest and return its SHA-256."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent,
            prefix=f".{destination.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return sha256_file(destination)


def _atomic_write_text(path: os.PathLike[str] | str, value: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent,
            prefix=f".{destination.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_savez(path: os.PathLike[str] | str, **arrays: np.ndarray) -> str:
    """Atomically replace an NPZ and return the hash of the final bytes."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", dir=destination.parent,
            prefix=f".{destination.name}.", suffix=".npz", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return sha256_file(destination)


def _validate_constraint_args(gc_min: float, gc_max: float, max_run: int) -> None:
    if not np.isfinite(gc_min) or not np.isfinite(gc_max):
        raise ValueError("GC bounds must be finite")
    if not (0.0 <= gc_min <= gc_max <= 1.0):
        raise ValueError(
            "expected 0 <= gc_min <= gc_max <= 1; got "
            f"{gc_min}, {gc_max}")
    if (isinstance(max_run, (bool, np.bool_))
            or not isinstance(max_run, (int, np.integer))
            or int(max_run) < 1):
        raise ValueError(f"max_run must be a positive integer; got {max_run!r}")


def _validate_integer_array(array: np.ndarray, name: str) -> None:
    if (not np.issubdtype(array.dtype, np.integer)
            or np.issubdtype(array.dtype, np.bool_)):
        raise TypeError(f"{name} must have integer dtype; got {array.dtype}")


def _base_from_hash_checked(hash_2bit: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(hash_2bit)
    _validate_integer_array(array, name)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty rank-2 array; got {array.shape}")
    if array.shape[1] % 2:
        raise ValueError(f"{name} bit length must be even; got {array.shape[1]}")
    values = set(int(value) for value in np.unique(array))
    if not (values <= {0, 1} or values <= {-1, 1}):
        raise ValueError(
            f"{name} must use either {{0,1}} or {{-1,+1}} encoding; got {sorted(values)}")
    bits = (array > 0).astype(np.int8, copy=False).reshape(
        array.shape[0], array.shape[1] // 2, 2)
    return (bits[:, :, 0] * 2 + bits[:, :, 1]).astype(np.int8, copy=False)


def _load_extraction(
    path: os.PathLike[str] | str,
    tag: str,
    *,
    strict_image_paths: bool = False,
) -> dict[str, Any]:
    """Load and strictly validate one extraction artifact."""
    artifact = Path(path)
    if not artifact.is_file():
        raise FileNotFoundError(f"missing {tag} extraction: {artifact}")
    stat_before = artifact.stat()
    with np.load(artifact, allow_pickle=False) as npz:
        names = set(npz.files)
        if "multi_hot_labels" not in names:
            raise ValueError(f"{artifact}: missing required multi_hot_labels")
        if "base_indices" not in names and "hash_2bit" not in names:
            raise ValueError(f"{artifact}: need base_indices or hash_2bit")
        labels = np.asarray(npz["multi_hot_labels"])
        base = np.asarray(npz["base_indices"]) if "base_indices" in names else None
        hash_2bit = np.asarray(npz["hash_2bit"]) if "hash_2bit" in names else None
        image_paths = None
        image_paths_status = "absent"
        if "image_paths" in names:
            try:
                image_paths = np.asarray(npz["image_paths"])
            except ValueError as error:
                if "Object arrays cannot be loaded" not in str(error):
                    raise
                if strict_image_paths:
                    raise ValueError(
                        f"{artifact}: object-dtype image_paths requires pickle and "
                        "is forbidden in paper-grade mode; regenerate the extraction "
                        "with the current Unicode-string extractor") from error
                image_paths_status = "legacy_object_omitted_without_unpickling"
            else:
                if image_paths.dtype.kind not in ("U", "S"):
                    raise TypeError(
                        f"{tag}.image_paths must have Unicode/bytes string dtype; "
                        f"got {image_paths.dtype}")
                image_paths_status = "safe_string"

    _validate_integer_array(labels, f"{tag}.multi_hot_labels")
    if labels.ndim != 2 or labels.shape[0] == 0 or labels.shape[1] == 0:
        raise ValueError(
            f"{tag}.multi_hot_labels must be non-empty rank-2; got {labels.shape}")
    if int(labels.min()) < 0 or int(labels.max()) > 1:
        raise ValueError(f"{tag}.multi_hot_labels must contain only 0/1")
    derived = None
    if hash_2bit is not None:
        derived = _base_from_hash_checked(hash_2bit, f"{tag}.hash_2bit")
    if base is not None:
        _validate_integer_array(base, f"{tag}.base_indices")
        if base.ndim != 2 or base.shape[0] == 0 or base.shape[1] == 0:
            raise ValueError(
                f"{tag}.base_indices must be non-empty rank-2; got {base.shape}")
        if int(base.min()) < 0 or int(base.max()) > 3:
            raise ValueError(f"{tag}.base_indices must contain only 0,1,2,3")
        base = base.astype(np.int8, copy=False)
        if derived is not None and (
                derived.shape != base.shape or not np.array_equal(derived, base)):
            raise ValueError(
                f"{tag}: base_indices and hash_2bit encode different DNA codes")
    else:
        assert derived is not None
        base = derived

    if labels.shape[0] != base.shape[0]:
        raise ValueError(
            f"{tag}: code/label row mismatch {base.shape[0]} != {labels.shape[0]}")

    if image_paths is not None:
        if image_paths.ndim != 1 or image_paths.shape[0] != base.shape[0]:
            raise ValueError(
                f"{tag}.image_paths must be [N]; got {image_paths.shape} for N={len(base)}")

    stat_after = artifact.stat()
    stable_fields = ("st_ino", "st_size", "st_mtime_ns")
    if any(getattr(stat_before, field) != getattr(stat_after, field)
           for field in stable_fields):
        raise RuntimeError(f"{artifact} changed while it was being validated")
    return {
        "base_indices": base,
        "multi_hot_labels": labels.astype(np.int64, copy=False),
        "image_paths": image_paths,
        "image_paths_status": image_paths_status,
        "sha256": sha256_file(artifact),
        "path": str(artifact.resolve()),
    }


def project_batch_memoized(
    codes: np.ndarray,
    gc_min: float,
    gc_max: float,
    max_run: int,
    *,
    allow_projection_failures: bool = False,
) -> dict:
    """Same as batch_project_to_valid but projects each UNIQUE code once.

    The DP projection is a pure function of the code, so identical codes map to
    identical projections. Our codes have low DB-unique (0.19-0.40), so most
    rows are duplicates -- deduping is exact and turns the O(N) DP loop into
    O(#unique). For high-unique baselines it is a no-op speedup.
    """
    _validate_constraint_args(gc_min, gc_max, max_run)
    arr = np.asarray(codes)
    _validate_integer_array(arr, "codes")
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"codes must be a non-empty rank-2 array; got {arr.shape}")
    if int(arr.min()) < 0 or int(arr.max()) > 3:
        raise ValueError("codes must contain only canonical base IDs 0,1,2,3")
    arr = arr.astype(np.int8, copy=False)
    N, L = arr.shape
    gc_min_count = int(np.ceil(gc_min * L))
    gc_max_count = int(np.floor(gc_max * L))
    if gc_min_count > gc_max_count and not allow_projection_failures:
        raise ValueError(
            f"no integer GC count satisfies [{gc_min}, {gc_max}] for L={L}: "
            f"[{gc_min_count}, {gc_max_count}]")
    uniq, inv = np.unique(arr, axis=0, return_inverse=True)      # [U, L], [N]
    valid_u = is_valid_batch(uniq, gc_min, gc_max, max_run)      # [U]
    proj_u = uniq.copy()
    edit_u = np.zeros(len(uniq), dtype=np.int32)
    for i in range(len(uniq)):
        if valid_u[i]:
            continue
        p, cost = project_to_valid(uniq[i], gc_min, gc_max, max_run)
        proj_u[i] = p
        edit_u[i] = cost
    out = proj_u[inv]
    edits = edit_u[inv]
    new_valid = is_valid_batch(out, gc_min, gc_max, max_run)
    succ = edits[edits >= 0]
    n_failed = int((edits < 0).sum())
    if n_failed and not allow_projection_failures:
        raise RuntimeError(
            f"bio-projection failed for {n_failed}/{N} rows; refusing to "
            "evaluate invalid DNA (use --allow_projection_failures only for diagnostics)")
    return {
        "projected_codes": out,
        "edit_distances": edits,
        "was_valid": is_valid_batch(arr, gc_min, gc_max, max_run),
        "compliance_rate": float(new_valid.mean()),
        "mean_edit_distance": float(succ.mean()) if succ.size else 0.0,
        "max_edit_distance": int(succ.max()) if succ.size else 0,
        "n_failed": n_failed,
        "n_unique": int(len(uniq)),
    }

CUTOFF = {"Flickr25k": 5000, "MSCOCO": 5000, "NUSWIDE": 5000, "CIFAR10": 1000}
SAVE_PROJECTED = False


def base_from_hash(h2: np.ndarray) -> np.ndarray:
    """[N,2L] signed/binary hash -> [N,L] canonical base IDs."""
    return _base_from_hash_checked(h2, "hash_2bit").astype(np.int64)


def get_base(npz) -> np.ndarray:
    if "base_indices" in npz.files:
        base = np.asarray(npz["base_indices"])
        _validate_integer_array(base, "base_indices")
        if base.ndim != 2 or base.size == 0:
            raise ValueError(f"base_indices must be non-empty rank-2; got {base.shape}")
        if int(base.min()) < 0 or int(base.max()) > 3:
            raise ValueError("base_indices must contain only 0,1,2,3")
        if "hash_2bit" in npz.files:
            derived = base_from_hash(npz["hash_2bit"])
            if derived.shape != base.shape or not np.array_equal(derived, base):
                raise ValueError("base_indices and hash_2bit are inconsistent")
        return base.astype(np.int64, copy=False)
    return base_from_hash(npz["hash_2bit"])


def _parse_args_extract(path: Path) -> dict[str, str]:
    parsed: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            if ":" not in raw_line:
                continue
            key, value = raw_line.rstrip("\n").split(":", 1)
            parsed[key.strip()] = value.strip()
    return parsed


def _resolve_recorded_path(raw_path: str, *, relative_to: Path) -> Optional[Path]:
    candidate = Path(raw_path).expanduser()
    possibilities = [candidate] if candidate.is_absolute() else [
        (REPO / candidate), (relative_to / candidate), candidate.resolve(),
    ]
    for possibility in possibilities:
        if possibility.is_file():
            return possibility.resolve()
    return None


def _read_checkpoint_protocol(checkpoint: Path) -> dict[str, Any]:
    """Read only primitive checkpoint metadata used to bind the P0 protocol."""
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
        raise ValueError(f"checkpoint lacks a config dictionary: {checkpoint}")
    config = payload["config"]
    protocol = config.get("protocol_identity_sha256")
    if protocol is not None:
        protocol = str(protocol).lower()
        if _SHA256_RE.fullmatch(protocol) is None:
            raise ValueError(
                f"invalid checkpoint protocol_identity_sha256: {protocol!r}")
    return {
        "protocol_identity_sha256": protocol,
        "checkpoint_method": config.get("method"),
        "checkpoint_dataset": config.get("dataset"),
        "checkpoint_setting": config.get("setting"),
        "checkpoint_bit": config.get("bit"),
    }


def _collect_provenance(
    directory: Path,
    *,
    requested_name: str,
    dataset: str,
    expected_bit: int,
    extraction_input_sha256: dict[str, str],
    require_provenance: bool,
) -> dict[str, Any]:
    args_path = directory / "args_extract.txt"
    checkpoint: Optional[Path] = None
    parsed: dict[str, str] = {}
    if args_path.is_file():
        parsed = _parse_args_extract(args_path)
        if parsed.get("dataset") not in (None, dataset):
            raise ValueError(
                f"args_extract dataset={parsed.get('dataset')!r} does not match {dataset!r}")
        if parsed.get("bit") is not None and int(parsed["bit"]) != expected_bit:
            raise ValueError(
                f"args_extract bit={parsed['bit']} does not match extraction "
                f"bit={expected_bit}")
        if parsed.get("weights"):
            checkpoint = _resolve_recorded_path(
                parsed["weights"], relative_to=args_path.parent)

    metadata: dict[str, Any] = {}
    metadata_error = None
    if checkpoint is not None:
        try:
            metadata = _read_checkpoint_protocol(checkpoint)
            if metadata.get("checkpoint_dataset") not in (None, dataset):
                raise ValueError(
                    f"checkpoint dataset={metadata.get('checkpoint_dataset')!r} "
                    f"does not match {dataset!r}")
            if (metadata.get("checkpoint_bit") is not None
                    and int(metadata["checkpoint_bit"]) != expected_bit):
                raise ValueError(
                    f"checkpoint bit={metadata.get('checkpoint_bit')} does not "
                    f"match extraction bit={expected_bit}")
            if (parsed.get("setting") is not None
                    and metadata.get("checkpoint_setting") is not None
                    and parsed["setting"] != str(metadata["checkpoint_setting"])):
                raise ValueError("args_extract/checkpoint setting mismatch")
        except Exception as error:  # surface below in strict paper mode
            metadata_error = f"{type(error).__name__}: {error}"

    protocol_artifacts = []
    protocol_manifest = None
    protocol_manifest_path: Optional[Path] = None
    protocol_manifest_record = None
    # Never hash the post-bio completion manifest here: it binds this script's
    # outputs, so doing so would create a circular dependency.  Only immutable
    # pre-bio protocol artifacts are valid inputs.
    candidates = [directory / "p0_protocol_manifest.json"]
    if directory.name.endswith("_dnaeval"):
        refit_result = directory.parent / directory.name[:-len("_dnaeval")]
        candidates.append(refit_result / "p0_protocol_summary.json")
    protocol = metadata.get("protocol_identity_sha256")
    for artifact in candidates:
        if not artifact.is_file():
            continue
        with artifact.open(encoding="utf-8") as handle:
            artifact_payload = json.load(handle)
        if artifact.name == "p0_protocol_manifest.json":
            protocol_manifest = artifact_payload
            protocol_manifest_path = artifact
        artifact_protocol = (
            artifact_payload.get("protocol_identity_sha256")
            or artifact_payload.get("protocol_digest_sha256")
        )
        if artifact_protocol is not None:
            artifact_protocol = str(artifact_protocol).lower()
            if _SHA256_RE.fullmatch(artifact_protocol) is None:
                raise ValueError(f"invalid protocol digest in {artifact}")
            if protocol is not None and artifact_protocol != protocol:
                raise ValueError(
                    f"protocol digest mismatch: checkpoint={protocol}, "
                    f"{artifact}={artifact_protocol}")
            protocol = protocol or artifact_protocol
        artifact_record = {
            "path": str(artifact.resolve()),
            "sha256": sha256_file(artifact),
            "declared_protocol_identity_sha256": artifact_protocol,
        }
        protocol_artifacts.append(artifact_record)
        if artifact.name == "p0_protocol_manifest.json":
            protocol_manifest_record = artifact_record

    eligibility_reasons = []
    selection_record = None
    if protocol_manifest is None:
        eligibility_reasons.append("missing immutable p0_protocol_manifest.json")
    else:
        if protocol_manifest.get("dataset") != dataset:
            raise ValueError(
                f"protocol manifest dataset={protocol_manifest.get('dataset')!r} does not "
                f"match {dataset!r}")
        manifest_method = protocol_manifest.get("method")
        manifest_variant = protocol_manifest.get("variant")
        if manifest_variant != requested_name:
            raise ValueError(
                f"protocol manifest variant={manifest_variant!r} does not match "
                f"cell name={requested_name!r}")
        checkpoint_method = metadata.get("checkpoint_method")
        if (checkpoint_method is not None and manifest_method is not None
                and checkpoint_method != manifest_method):
            raise ValueError(
                f"checkpoint method={checkpoint_method!r} does not match protocol "
                f"manifest method={manifest_method!r}")
        declared_extraction = protocol_manifest.get("extraction_dir")
        if declared_extraction is None:
            eligibility_reasons.append("protocol manifest does not bind extraction_dir")
        else:
            declared_path = Path(str(declared_extraction)).expanduser()
            if not declared_path.is_absolute():
                declared_path = REPO / declared_path
            if declared_path.resolve() != directory.resolve():
                raise ValueError(
                    f"protocol manifest extraction_dir={declared_path.resolve()} does "
                    f"not match evaluated directory={directory.resolve()}")
        if protocol_manifest.get("main_protocol_eligible") is not True:
            eligibility_reasons.append("protocol manifest main_protocol_eligible is not true")
        if protocol_manifest.get("protocol_deviations") not in ([], ()):
            eligibility_reasons.append("protocol manifest declares protocol deviations")
        if protocol_manifest.get("main_eligibility_blockers") not in (None, [], ()):
            eligibility_reasons.append("protocol manifest declares eligibility blockers")
        if protocol_manifest.get("test_used_for_selection") is not False:
            eligibility_reasons.append("test_used_for_selection is not false")
        if protocol_manifest.get("final_checkpoint_count") != 1:
            eligibility_reasons.append("final_checkpoint_count is not one")
        declared_bit_length = protocol_manifest.get("bit_length")
        declared_base_length = protocol_manifest.get("base_length")
        if declared_bit_length is None or declared_base_length is None:
            eligibility_reasons.append("protocol manifest lacks bit/base length labels")
        elif (int(declared_bit_length) != expected_bit
              or int(declared_base_length) != expected_bit // 2):
            raise ValueError(
                "protocol manifest bit/base length labels do not match extraction")
        expected_selection_metric = (
            f"raw_{expected_bit // 2}base_base_hamming_mAP_at_R")
        if protocol_manifest.get("selection_metric") != expected_selection_metric:
            eligibility_reasons.append("unexpected or missing selection metric")
        if protocol_manifest.get("run_manifest_phase") != "pre_bio_projection":
            eligibility_reasons.append(
                "protocol manifest phase is not pre_bio_projection")
        protocol_payload = protocol_manifest.get("protocol_identity")
        if not isinstance(protocol_payload, dict):
            eligibility_reasons.append("protocol manifest lacks protocol identity payload")
        else:
            if int(protocol_payload.get("bit", -1)) != expected_bit:
                raise ValueError(
                    "protocol manifest bit budget does not match extraction bit length")
            payload_digest = hashlib.sha256(json.dumps(
                protocol_payload, sort_keys=True,
                separators=(",", ":")).encode("utf-8")).hexdigest()
            if payload_digest != protocol:
                raise ValueError(
                    "protocol manifest protocol_identity payload does not match its SHA-256")
        declared_inputs = protocol_manifest.get("extraction_artifact_sha256")
        expected_inputs = dict(extraction_input_sha256)
        if args_path.is_file():
            expected_inputs["args_extract.txt"] = sha256_file(args_path)
        if not isinstance(declared_inputs, dict):
            eligibility_reasons.append(
                "protocol manifest does not content-bind extraction artifacts")
        else:
            normalized_declared = {
                str(key): str(value).lower()
                for key, value in declared_inputs.items()
            }
            if normalized_declared != expected_inputs:
                raise ValueError(
                    "protocol manifest extraction artifact SHA-256 map mismatch")

        if checkpoint is not None:
            declared_checkpoint = protocol_manifest.get("final_checkpoint")
            declared_checkpoint_sha = protocol_manifest.get("final_checkpoint_sha256")
            if declared_checkpoint is None or declared_checkpoint_sha is None:
                eligibility_reasons.append("protocol manifest does not content-bind final checkpoint")
            else:
                declared_checkpoint_path = Path(str(declared_checkpoint)).expanduser()
                if not declared_checkpoint_path.is_absolute():
                    declared_checkpoint_path = REPO / declared_checkpoint_path
                if declared_checkpoint_path.resolve() != checkpoint.resolve():
                    raise ValueError(
                        "protocol manifest final_checkpoint disagrees with args_extract weights")
                checkpoint_sha = sha256_file(checkpoint)
                if str(declared_checkpoint_sha).lower() != checkpoint_sha:
                    raise ValueError("protocol manifest final checkpoint SHA-256 mismatch")

        selection_path_raw = protocol_manifest.get("selection_artifact")
        selection_sha_declared = protocol_manifest.get("selection_artifact_sha256")
        if selection_path_raw is None or selection_sha_declared is None:
            eligibility_reasons.append("protocol manifest lacks selection artifact binding")
        else:
            assert protocol_manifest_path is not None
            selection_path = _resolve_recorded_path(
                str(selection_path_raw), relative_to=protocol_manifest_path.parent)
            if selection_path is None:
                eligibility_reasons.append("bound selection artifact is missing")
            else:
                selection_sha = sha256_file(selection_path)
                if str(selection_sha_declared).lower() != selection_sha:
                    raise ValueError("selection artifact SHA-256 mismatch")
                with selection_path.open(encoding="utf-8") as handle:
                    selection = json.load(handle)
                if selection.get("dataset") != dataset:
                    raise ValueError("selection artifact dataset mismatch")
                if selection.get("method") != manifest_method:
                    raise ValueError("selection artifact method mismatch")
                if selection.get("protocol_identity_sha256") != protocol:
                    raise ValueError("selection artifact protocol identity mismatch")
                if selection.get("selection_only") is not True:
                    eligibility_reasons.append("selection artifact is not selection-only")
                if selection.get("test_touched_during_selection") is not False:
                    eligibility_reasons.append("selection artifact touched test data")
                if (selection.get("bit_length") != expected_bit
                        or selection.get("base_length") != expected_bit // 2):
                    eligibility_reasons.append(
                        "selection artifact bit/base length binding mismatch")
                if selection.get("selection_metric") != protocol_manifest.get("selection_metric"):
                    eligibility_reasons.append("selection metric binding mismatch")
                if selection.get("best_epoch") != protocol_manifest.get("best_epoch_zero_based"):
                    eligibility_reasons.append("selected epoch binding mismatch")
                selection_record = {
                    "path": str(selection_path), "sha256": selection_sha,
                    "best_epoch": selection.get("best_epoch"),
                }
                protocol_artifacts.append(selection_record)

    missing = []
    if not args_path.is_file():
        missing.append("args_extract.txt")
    if checkpoint is None:
        missing.append("checkpoint referenced by args_extract.txt")
    if metadata_error is not None:
        missing.append(f"readable checkpoint config ({metadata_error})")
    if protocol is None:
        missing.append("protocol_identity_sha256")
    if require_provenance and missing:
        raise ValueError(
            f"{directory}: incomplete paper-result provenance: {', '.join(missing)}; "
            "use --allow_missing_provenance only for explicitly nonstandard diagnostics")

    identity_complete = not missing
    paper_protocol_eligible = identity_complete and not eligibility_reasons
    status = (
        "paper_protocol_eligible" if paper_protocol_eligible else
        "identity_complete_not_paper_eligible" if identity_complete else
        "incomplete_nonstandard"
    )
    return {
        "status": status,
        "missing": missing,
        "args_extract": ({
            "path": str(args_path.resolve()), "sha256": sha256_file(args_path),
        } if args_path.is_file() else None),
        "checkpoint": ({
            "path": str(checkpoint), "sha256": sha256_file(checkpoint), **metadata,
        } if checkpoint is not None else None),
        "protocol_identity_sha256": protocol,
        "protocol_manifest": protocol_manifest_record,
        "protocol_artifacts": protocol_artifacts,
        "selection_artifact": selection_record,
        "paper_protocol_eligible": paper_protocol_eligible,
        "eligibility_reasons": eligibility_reasons,
    }


def _resolve_retrieval_device(device: Optional[str]) -> str:
    import torch

    if device is None:
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def map_at_r(qb: np.ndarray, dbb: np.ndarray, ql: np.ndarray, dl: np.ndarray,
             R: int, qchunk: int = 256,
             device: Optional[str] = None) -> float:
    """Canonical base-Hamming mAP@R with stable database-order tie breaking.

    Hamming codes have many ties. ``torch.topk`` does not promise stable order
    across devices/versions, so distances are computed in GPU chunks but each
    row is ranked by NumPy's stable argsort, matching ``evaluation_siglip2``.
    """
    import torch
    dev = _resolve_retrieval_device(device)
    D = torch.tensor(dbb, dtype=torch.int8, device=dev)          # [Nd, L]
    aps = []
    for s in range(0, len(qb), qchunk):
        Q = torch.tensor(qb[s:s + qchunk], dtype=torch.int8, device=dev)   # [q, L]
        dist = (Q[:, None, :] != D[None, :, :]).sum(-1)
        dist_np = dist.to(torch.int16).cpu().numpy()
        relevance = _multi_hot_relevance(ql[s:s + qchunk], dl)
        for row in range(dist_np.shape[0]):
            order = np.argsort(dist_np[row], kind='stable')
            aps.append(_ap_at_r(relevance[row][order], R=int(R)))
    return float(np.mean(aps)) if aps else 0.0


def run_cell(
    name: str,
    ddir: str,
    dataset: str,
    gc_min: float,
    gc_max: float,
    max_run: int,
    *,
    expected_length: Optional[int] = 18,
    allow_projection_failures: bool = False,
    require_provenance: bool = False,
    device: Optional[str] = None,
) -> dict:
    directory = Path(ddir)
    db = _load_extraction(
        directory / "extract_db.npz", "db",
        strict_image_paths=require_provenance)
    qy = _load_extraction(
        directory / "extract_query.npz", "query",
        strict_image_paths=require_provenance)
    dbb, qb = db["base_indices"], qy["base_indices"]
    dl, ql = db["multi_hot_labels"], qy["multi_hot_labels"]
    if dbb.shape[1] != qb.shape[1]:
        raise ValueError(
            f"query/database DNA lengths differ: {qb.shape[1]} != {dbb.shape[1]}")
    if expected_length is not None and dbb.shape[1] != expected_length:
        raise ValueError(
            f"expected {expected_length}-base comparison code, got L={dbb.shape[1]}")
    if dl.shape[1] != ql.shape[1]:
        raise ValueError(
            f"query/database label dimensions differ: {ql.shape[1]} != {dl.shape[1]}")
    R = CUTOFF[dataset]
    provenance = _collect_provenance(
        directory, requested_name=name, dataset=dataset,
        expected_bit=2 * dbb.shape[1],
        extraction_input_sha256={
            "extract_db.npz": db["sha256"],
            "extract_query.npz": qy["sha256"],
        },
        require_provenance=require_provenance)

    L = dbb.shape[1]
    gc_min_c = int(np.ceil(gc_min * L)); gc_max_c = int(np.floor(gc_max * L))
    if gc_min_c > gc_max_c and not allow_projection_failures:
        raise ValueError(
            f"no feasible integer GC count for L={L}: [{gc_min_c}, {gc_max_c}]")

    def dna_unique(codes):
        return len(np.unique(codes, axis=0)) / len(codes)

    pre_valid_db = is_valid_batch(dbb, gc_min, gc_max, max_run)
    pre_valid_qy = is_valid_batch(qb, gc_min, gc_max, max_run)
    retrieval_device = _resolve_retrieval_device(device)
    map_pre = map_at_r(qb, dbb, ql, dl, R, device=retrieval_device)
    dna_unique_pre = dna_unique(dbb)

    proj_db = project_batch_memoized(
        dbb, gc_min, gc_max, max_run,
        allow_projection_failures=allow_projection_failures)
    proj_qy = project_batch_memoized(
        qb, gc_min, gc_max, max_run,
        allow_projection_failures=allow_projection_failures)
    dbb_p, qb_p = proj_db["projected_codes"], proj_qy["projected_codes"]
    map_post = map_at_r(qb_p, dbb_p, ql, dl, R, device=retrieval_device)
    dna_unique_post = dna_unique(dbb_p)
    projection_invariant_satisfied = (
        proj_db["n_failed"] == 0 and proj_qy["n_failed"] == 0
        and proj_db["compliance_rate"] == 1.0
        and proj_qy["compliance_rate"] == 1.0
    )
    if not projection_invariant_satisfied and not allow_projection_failures:
        raise RuntimeError("post-projection DNA validity invariant was not satisfied")

    output_artifacts: dict[str, dict[str, str]] = {}
    if SAVE_PROJECTED:
        def projected_payload(source: dict, projection: dict) -> dict[str, np.ndarray]:
            bases = projection["projected_codes"].astype(np.int64)
            bits = np.stack((bases // 2, bases % 2), axis=-1).reshape(len(bases), -1)
            payload = {
                "base_indices": bases,
                "hash_2bit": bits.astype(np.uint8),
                "multi_hot_labels": source["multi_hot_labels"],
                "projection_edit_distances": projection["edit_distances"],
                "projection_was_valid": projection["was_valid"],
                "bio_gc_min_frac": np.asarray(gc_min, dtype=np.float64),
                "bio_gc_max_frac": np.asarray(gc_max, dtype=np.float64),
                "bio_max_homopolymer_run": np.asarray(max_run, dtype=np.int64),
                "projection_tie_policy": np.asarray(PROJECTION_TIE_POLICY),
                "source_npz_sha256": np.asarray(source["sha256"]),
                "protocol_identity_sha256": np.asarray(
                    provenance.get("protocol_identity_sha256") or ""),
                "protocol_manifest_sha256": np.asarray(
                    (provenance.get("protocol_manifest") or {}).get("sha256", "")),
                "checkpoint_sha256": np.asarray(
                    (provenance.get("checkpoint") or {}).get("sha256", "")),
            }
            if source["image_paths"] is not None:
                payload["image_paths"] = source["image_paths"]
            return payload

        for tag, source, projection in (
                ("db", db, proj_db), ("query", qy, proj_qy)):
            destination = directory / f"extract_{tag}_bioproj.npz"
            digest = _atomic_savez(
                destination, **projected_payload(source, projection))
            output_artifacts[tag] = {
                "path": str(destination.resolve()), "sha256": digest,
            }

    cell_paper_result_eligible = (
        projection_invariant_satisfied
        and provenance["paper_protocol_eligible"]
        and set(output_artifacts) == {"db", "query"}
    )

    return {
        "name": name, "dataset": dataset, "dir": ddir,
        "L": int(L), "gc_count_range": [gc_min_c, gc_max_c], "max_run": int(max_run),
        "pre_compliance_db": float(pre_valid_db.mean()),
        "pre_compliance_qy": float(pre_valid_qy.mean()),
        "post_compliance_db": float(proj_db["compliance_rate"]),
        "post_compliance_qy": float(proj_qy["compliance_rate"]),
        "mean_edit_db": float(proj_db["mean_edit_distance"]),
        "mean_edit_qy": float(proj_qy["mean_edit_distance"]),
        "max_edit_db": int(proj_db["max_edit_distance"]),
        "max_edit_qy": int(proj_qy["max_edit_distance"]),
        "projection_failures_db": int(proj_db["n_failed"]),
        "projection_failures_qy": int(proj_qy["n_failed"]),
        "projection_invariant_satisfied": projection_invariant_satisfied,
        "paper_result_eligible": cell_paper_result_eligible,
        "map_at_R_pre": map_pre,
        "map_at_R_post": map_post,
        "delta_proj": map_post - map_pre,
        "dna_unique_pre": dna_unique_pre,
        "dna_unique_post": dna_unique_post,
        "dna_unique_delta": dna_unique_post - dna_unique_pre,
        "R": R,
        "ranking_tie_policy": RETRIEVAL_TIE_POLICY,
        "retrieval_device": retrieval_device,
        "projection_tie_policy": PROJECTION_TIE_POLICY,
        "input_artifacts": {
            "db": {"path": db["path"], "sha256": db["sha256"],
                   "image_paths_status": db["image_paths_status"]},
            "query": {"path": qy["path"], "sha256": qy["sha256"],
                      "image_paths_status": qy["image_paths_status"]},
        },
        "training_provenance": provenance,
        "output_artifacts": output_artifacts,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/bio_projection_comparison.json")
    ap.add_argument("--gc_min", type=float, default=DEFAULT_GC_MIN_FRAC)
    ap.add_argument("--gc_max", type=float, default=DEFAULT_GC_MAX_FRAC)
    ap.add_argument("--max_run", type=int, default=DEFAULT_MAX_HOMOPOLYMER_RUN)
    ap.add_argument(
        "--device", default=None,
        help="Device for retrieval-distance evaluation (e.g. cpu or cuda:1).",
    )
    ap.add_argument(
        "--expected_length", type=int, default=18,
        help="required DNA length for every cell (use 24 explicitly for a 24-base study)",
    )
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these datasets")
    ap.add_argument(
        "--cell", action="append", default=[], metavar="NAME,DATASET,DIR",
        help=("Evaluate an explicit extraction directory. Repeat for multiple "
              "cells. When supplied, the legacy auto-discovered cells are not used."),
    )
    ap.add_argument("--save_projected", action="store_true",
                    help="also save bio-projected extractions for downstream compositional eval")
    ap.add_argument(
        "--allow_projection_failures", action="store_true",
        help=("DIAGNOSTIC ONLY: keep and evaluate rows for which no valid projection "
              "exists; marks the manifest nonstandard"),
    )
    ap.add_argument(
        "--allow_missing_provenance", action="store_true",
        help=("DIAGNOSTIC/LEGACY ONLY: permit an explicit --cell without a hashed "
              "checkpoint and protocol identity"),
    )
    args = ap.parse_args()
    _validate_constraint_args(args.gc_min, args.gc_max, args.max_run)
    if args.expected_length < 1:
        raise ValueError("--expected_length must be positive")
    global SAVE_PROJECTED
    SAVE_PROJECTED = args.save_projected

    ours = {
        "Flickr25k": "result/260717+flickr25k_setting1_flickr_P0refit_e4+bs+64+e+60+proj_lr+0.001",
        "MSCOCO": "result/260717+mscoco_setting1_mscoco_P0refit_e49+bs+64+e+60+proj_lr+0.001",
        "NUSWIDE": "result/260717+nuswide_setting1_nuswide_P0refit_e4+bs+64+e+60+proj_lr+0.001",
        "CIFAR10": "result/260717+cifar10_setting1_cifar10_P0refit_e14+bs+64+e+60+proj_lr+0.001",
    }
    if args.cell:
        cells = []
        for specification in args.cell:
            try:
                name, dataset, directory = specification.split(',', 2)
            except ValueError as error:
                raise ValueError(
                    f"--cell must be NAME,DATASET,DIR; got {specification!r}"
                ) from error
            if dataset not in CUTOFF:
                raise ValueError(
                    f"unknown dataset {dataset!r} in --cell; choose {sorted(CUTOFF)}")
            cells.append((name, directory, dataset))
    else:
        cells = [("Ours", d, ds) for ds, d in ours.items()]
        for m in ["cibhash", "cimon", "mls3rduh"]:
            for ds in ["Flickr25k", "MSCOCO", "NUSWIDE", "CIFAR10"]:
                g = glob.glob(f"result_baseline/260721/{m}_{ds}_clip_E*_dnaeval")
                if g:
                    cells.append((m, g[0], ds))

    if args.only:
        want = set(args.only)
        cells = [c for c in cells if c[2] in want]
    if not cells:
        raise ValueError("no bio-projection cells were selected")
    identities = [(name, dataset) for name, _directory, dataset in cells]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate (method name, dataset) bio-projection cell")
    directories = [str(Path(directory).resolve()) for _name, directory, _dataset in cells]
    if len(set(directories)) != len(directories):
        raise ValueError("one extraction directory was assigned to multiple cells")

    results = []
    explicit_cells = bool(args.cell)
    cfg = {
        "schema_version": 2,
        "gc_min_frac": args.gc_min,
        "gc_max_frac": args.gc_max,
        "max_homopolymer_run": args.max_run,
        "expected_length": args.expected_length,
        "device_request": args.device,
        "projection_tie_policy": PROJECTION_TIE_POLICY,
        "ranking_tie_policy": RETRIEVAL_TIE_POLICY,
        "allow_projection_failures": args.allow_projection_failures,
        "allow_missing_provenance": args.allow_missing_provenance,
        "paper_result_eligible": False,
    }
    print(f"{'method':10s}{'dataset':10s}{'preCompl':>9s}{'meanEdit':>9s}"
          f"{'mAP@R pre':>11s}{'mAP@R post':>12s}{'d_proj':>9s}", flush=True)
    for name, ddir, ds in cells:
        r = run_cell(
            name, ddir, ds, args.gc_min, args.gc_max, args.max_run,
            expected_length=args.expected_length,
            allow_projection_failures=args.allow_projection_failures,
            require_provenance=(explicit_cells and not args.allow_missing_provenance),
            device=args.device,
        )
        results.append(r)
        print(f"{name:10s}{ds:10s}{r['pre_compliance_db']:>9.3f}{r['mean_edit_db']:>9.3f}"
              f"{r['map_at_R_pre']:>11.4f}{r['map_at_R_post']:>12.4f}{r['delta_proj']:>+9.4f}",
              flush=True)
    cfg["paper_result_eligible"] = (
        explicit_cells
        and not args.allow_projection_failures
        and not args.allow_missing_provenance
        and all(cell["paper_result_eligible"] for cell in results)
    )
    cfg["paper_ineligibility_reasons"] = [
        {
            "name": cell["name"], "dataset": cell["dataset"],
            "reasons": cell["training_provenance"]["eligibility_reasons"],
            "projection_invariant_satisfied": cell["projection_invariant_satisfied"],
            "projected_outputs_saved": set(cell["output_artifacts"]) == {"db", "query"},
        }
        for cell in results if not cell["paper_result_eligible"]
    ]
    manifest_sha256 = _atomic_write_json(
        args.out, {"config": cfg, "cells": results})
    sidecar = f"{args.out}.sha256"
    _atomic_write_text(
        sidecar, f"{manifest_sha256}  {Path(args.out).name}\n")
    print(
        f"\nwrote {args.out} (sha256={manifest_sha256}; sidecar={sidecar})",
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
