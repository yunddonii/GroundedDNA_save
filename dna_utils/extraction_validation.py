"""One validator for an extraction run, shared by every reader (re-audit §17.9).

Each consumer previously carried its own partial idea of "complete", and every
one of them was satisfiable by artefacts that do not exist:

  * `aggregate_phase2_f01._require_manifests` checked that two JSON files were
    present and that four fields agreed. A probe with no NPZ, no checkpoint, no
    config, `schema_version=999` and `dataset=WRONG` was accepted.
  * `phase2_f01_reinference.sh` skipped a cell whenever the two manifest
    *filenames* existed; two files containing `not-json` counted as complete.
  * `_write_completion_marker` accepted whatever split set it was handed, so a
    marker naming only `db` was written and looked authoritative.
  * the manifest writer checked shapes but not dtypes, not that `hash_2bit` is
    the 2-bit encoding of `base_indices`, and not that codebook indices fall
    inside the codebook.

The contract here is deliberately strict and fail-closed: it opens every file it
names, recomputes every digest it compares, and refuses on the first
disagreement. `allow_backfilled` has to be passed explicitly, so a
retrospectively bound diagnostic can never be mistaken for an extraction that
recorded itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from typing import Iterable, Mapping, Sequence

import numpy as np

from dna_utils.runtime_state import sha256_file

#: `base_indices` -> `hash_2bit`, the encoding every consumer assumes.
_BASE_TO_BITS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8)

MARKER_NAME = "extraction_complete.json"
DEFAULT_SPLITS = ("db", "query")

#: Fields that must be identical across the splits of one run: they describe the
#: run, not the split. A disagreement means the splits came from different
#: forward passes, which is the failure F01 exists to make visible.
_COMMON_IDENTITY = (
    "checkpoint_sha256",
    "config_sha256",
    "inference_epoch",
    "inference_epoch_source",
    "effective_sinkhorn_epsilon",
    "sinkhorn_schedule_horizon",
    "lr_schedule_horizon",
    "training_epoch_budget",
    "training_stop_epoch",
    "num_slots",
    "bases_per_slot",
    "total_bases",
    "total_bits",
    "dataset",
    "random_seed",
)


class ExtractionInvalid(RuntimeError):
    """An extraction run cannot be admitted."""


@dataclass
class ValidatedRun:
    run_dir: str
    splits: dict
    backfilled: bool
    common: dict = field(default_factory=dict)


def base_indices_to_2bit(codes: np.ndarray) -> np.ndarray:
    """The canonical encoding, so writer and reader cannot drift apart."""
    return _BASE_TO_BITS[np.asarray(codes)].reshape(len(codes), -1)


def _load_json(path: str, what: str) -> dict:
    if not os.path.isfile(path):
        raise ExtractionInvalid(f"{what} is missing: {path}")
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ExtractionInvalid(f"{what} is not readable JSON: {error}") from None
    if not isinstance(payload, dict):
        raise ExtractionInvalid(f"{what} is not an object: {path}")
    return payload


def _validate_npz(manifest: Mapping, *, split: str) -> None:
    """Open the NPZ the manifest names and hold it to every declared field."""
    npz_path = str(manifest.get("npz_path") or "")
    if not os.path.isfile(npz_path):
        raise ExtractionInvalid(
            f"{split}: manifest names {npz_path!r}, which does not exist")
    digest = sha256_file(npz_path)
    if digest != manifest.get("npz_sha256"):
        raise ExtractionInvalid(
            f"{split}: {npz_path} hashes to {digest[:12]}... but the manifest "
            f"declares {str(manifest.get('npz_sha256'))[:12]}...")

    slots = int(manifest["num_slots"])
    per_slot = int(manifest["bases_per_slot"])
    bases = slots * per_slot
    rows = int(manifest["n_rows"])

    with np.load(npz_path, allow_pickle=False) as stored:
        for name in ("base_indices", "hash_2bit", "codebook_indices"):
            if name not in stored:
                raise ExtractionInvalid(f"{split}: {npz_path} has no {name}")
        base = np.asarray(stored["base_indices"])
        hashed = np.asarray(stored["hash_2bit"])
        codebook = np.asarray(stored["codebook_indices"])

    for name, arr in (("base_indices", base), ("hash_2bit", hashed),
                      ("codebook_indices", codebook)):
        if not np.issubdtype(arr.dtype, np.integer):
            raise ExtractionInvalid(
                f"{split}: {name} has dtype {arr.dtype}, expected an integer "
                f"type -- a float code is not a code")
    if base.shape != (rows, bases):
        raise ExtractionInvalid(
            f"{split}: base_indices is {base.shape}, expected {(rows, bases)}")
    if hashed.shape != (rows, 2 * bases):
        raise ExtractionInvalid(
            f"{split}: hash_2bit is {hashed.shape}, expected "
            f"{(rows, 2 * bases)}")
    if codebook.shape != (rows, slots):
        raise ExtractionInvalid(
            f"{split}: codebook_indices is {codebook.shape}, expected "
            f"{(rows, slots)}")
    if base.size and (base.min() < 0 or base.max() > 3):
        raise ExtractionInvalid(
            f"{split}: base_indices outside 0..3 [{base.min()}, {base.max()}]")
    if hashed.size and (hashed.min() < 0 or hashed.max() > 1):
        raise ExtractionInvalid(
            f"{split}: hash_2bit outside 0..1 [{hashed.min()}, {hashed.max()}]")
    if base.size and not np.array_equal(
            base_indices_to_2bit(base), hashed.astype(np.uint8)):
        raise ExtractionInvalid(
            f"{split}: hash_2bit is not the 2-bit encoding of base_indices; "
            f"the two arrays describe different codes")
    codebook_size = manifest.get("codebook_size")
    if isinstance(codebook_size, int) and codebook.size:
        if codebook.min() < 0 or codebook.max() >= codebook_size:
            raise ExtractionInvalid(
                f"{split}: codebook_indices outside 0..{codebook_size - 1} "
                f"[{codebook.min()}, {codebook.max()}]")


def validate_extraction_run(
    run_dir: str,
    *,
    required_splits: Sequence[str] = DEFAULT_SPLITS,
    allow_backfilled: bool = False,
    verify_npz: bool = True,
) -> ValidatedRun:
    """Admit a run only if every artefact it names exists and agrees.

    `allow_backfilled` is explicit on purpose: a manifest written after the fact
    binds a cell to its inputs but is not evidence that the extraction recorded
    itself, so a paper path must opt in to accepting one.
    """
    required = tuple(required_splits)
    marker_path = os.path.join(run_dir, MARKER_NAME)
    marker = _load_json(marker_path, "completion marker")
    declared = tuple(marker.get("splits") or ())
    if set(declared) != set(required):
        raise ExtractionInvalid(
            f"{run_dir}: completion marker declares splits {list(declared)}, "
            f"expected exactly {list(required)}")

    manifests, backfilled = {}, False
    for split in required:
        path = os.path.join(run_dir, f"extraction_manifest_{split}.json")
        manifest = _load_json(path, f"{split} extraction manifest")
        recorded = (marker.get("manifest_sha256") or {}).get(split)
        actual = sha256_file(path)
        if recorded != actual:
            raise ExtractionInvalid(
                f"{split}: the marker records manifest digest "
                f"{str(recorded)[:12]}... but the file hashes to "
                f"{actual[:12]}...")
        if manifest.get("split") != split:
            raise ExtractionInvalid(
                f"{split}: manifest declares split {manifest.get('split')!r}")
        if manifest.get("schema_version") != 1:
            raise ExtractionInvalid(
                f"{split}: unknown manifest schema_version "
                f"{manifest.get('schema_version')!r}")
        for name, key in (("checkpoint", "checkpoint_path"),
                          ("config", "config_path")):
            file_path = str(manifest.get(key) or "")
            if not os.path.isfile(file_path):
                raise ExtractionInvalid(
                    f"{split}: manifest names a {name} that does not exist: "
                    f"{file_path!r}")
            digest = sha256_file(file_path)
            if digest != manifest.get(f"{name}_sha256"):
                raise ExtractionInvalid(
                    f"{split}: {name} {file_path} hashes to {digest[:12]}... "
                    f"but the manifest declares "
                    f"{str(manifest.get(f'{name}_sha256'))[:12]}...")
        if verify_npz:
            _validate_npz(manifest, split=split)
        backfilled = backfilled or bool(manifest.get("backfilled", False))
        manifests[split] = manifest

    first = manifests[required[0]]
    common = {}
    for key in _COMMON_IDENTITY:
        values = {split: manifests[split].get(key) for split in required}
        if len(set(map(repr, values.values()))) != 1:
            raise ExtractionInvalid(
                f"{run_dir}: splits disagree on {key}: {values}. They were not "
                f"produced by the same runtime state.")
        common[key] = first.get(key)

    if backfilled and not allow_backfilled:
        raise ExtractionInvalid(
            f"{run_dir}: the manifests were backfilled after the fact. They "
            f"bind the cell to its inputs but are not evidence the extraction "
            f"recorded itself; pass allow_backfilled=True to admit it as a "
            f"diagnostic.")

    return ValidatedRun(run_dir=run_dir, splits=manifests,
                        backfilled=backfilled, common=common)


def describe_failure(run_dir: str, **kwargs) -> str | None:
    """`None` when the run is admissible, else the reason. For shell callers."""
    try:
        validate_extraction_run(run_dir, **kwargs)
    except ExtractionInvalid as error:
        return str(error)
    return None
