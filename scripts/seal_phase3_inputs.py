#!/usr/bin/env python3
"""Create and verify byte-level seals for canonical Phase-3 inputs.

This utility is deliberately separate from the Phase-3 launcher.  It does not
load a model, construct a dataloader, or mutate a cache.  It records the exact
files that the canonical stage-1/refit paths consume, plus the upstream Qwen,
foil, whitening, and row-index provenance needed to explain those bytes.

Two details are important for the rebuilt caches:

* cache entries are often symlinks.  A seal records the link text and link
  lstat, the final resolved target and target stat, and the target bytes;
* the foil overlay points many logical entries at the same very large targets
  as the feature cache.  Logical entries remain distinct in the manifest, but
  target hashes are memoized by resolved path and inode/stat identity so the
  same tensor is not read twice during one seal/verify operation.

The output is published without replacement: a complete temporary inode is
hard-linked to the requested path, so an existing seal is never overwritten
and a reader never observes a partial JSON document.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import pickle
import re
import secrets
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, MutableMapping, Sequence

import numpy as np


SCHEMA = "groundeddna.phase3-input-seal"
SCHEMA_VERSION = 2
HASH_ALGORITHM = "sha256"
HASH_CHUNK_BYTES = 8 * 1024 * 1024
PREFLIGHT_JSON_MAX_BYTES = 64 * 1024 * 1024
REPO = Path(__file__).resolve().parents[1]

SPLIT_IDENTITY_SCHEMA = "groundeddna.phase3-canonical-split-identity"
SPLIT_IDENTITY_SCHEMA_VERSION = 1
_ROW_VECTOR_ENCODING = "sha256-int64-le-c-order-v1"

_HF_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
)
_HF_CONFIG_FILE = "config.json"
_HF_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")

_WHITENING_KEYS = frozenset(("mu", "U", "S"))
_WHITENING_ORTHONORMAL_ATOL = 2.0e-5
# Float32 eigensolvers can leave tiny negative roundoff in a PSD covariance.
# Production v6prov reaches about -2.2e-10, so this is deliberately tight.
_WHITENING_EIGENVALUE_NEGATIVE_ATOL = 1.0e-8
_WHITENING_SOURCE_SLOTS = (1, 2, 3, 4, 5)
_WHITENING_RUNTIME_LOCAL_SLOTS = (1, 2, 3, 4)

_DATASETS = {
    "cifar10": ("CIFAR10", "cifar10"),
    "flickr25k": ("Flickr25k", "flickr"),
    "mscoco": ("MSCOCO", "mscoco"),
    "nuswide": ("NUSWIDE", "nuswide"),
}
_DATASET_ALIASES = {
    "cifar10": "cifar10",
    "flickr25k": "flickr25k",
    "flickr": "flickr25k",
    "mscoco": "mscoco",
    "coco": "mscoco",
    "nuswide": "nuswide",
    "nus-wide": "nuswide",
}
_STAGE_ALIASES = {
    "stage1": "stage1",
    "select": "stage1",
    "refit": "refit",
}

# dataloaders._SigLIP2FeatureCache opens all six unconditionally.
_CACHE_CORE = (
    "meta.json",
    "image_ids.json",
    "visual_tokens.f16.npy",
    "visual_global.f16.npy",
    "text_part.f16.npy",
    "has_text.bool.npy",
)

# Canonical Phase 3 enables bidirectional token pruning, so this otherwise
# optional pair is required rather than silently treated as absent.
_FACTUAL_TOKEN_PAIR = (
    "text_tokens.f16.npy",
    "text_token_mask.bool.npy",
)

# The loader consumes these when a foil overlay is used.  The row-order files
# are part of the loader's alignment checks, not merely annotations.
_FOIL_RUNTIME = (
    "text_foil_part.f16.npy",
    "text_foil_valid.bool.npy",
    "text_foil_image_ids.json",
    "text_foil_tokens.f16.npy",
    "text_foil_token_mask.bool.npy",
    "text_foil_token_image_ids.json",
)

# These close the derivation chain for the runtime foil tensors.
_FOIL_PROVENANCE = (
    "text_foil_meta.json",
    "text_foil_token_meta.json",
    "text_foil_edits.jsonl",
    "semantic_detail_cache_manifest.json",
)

_AUGMENTED_RE = re.compile(
    r"^visual_(tokens|global)_aug([0-9]+)\.f16\.npy$"
)


class SealError(RuntimeError):
    """An input set cannot be sealed or no longer matches its seal."""


@dataclass(frozen=True)
class SealRequest:
    dataset: str
    stage: str
    dataset_root: str
    feature_cache: str
    foil_cache: str
    whitening: str
    qwen: str
    split_rows: str

    @classmethod
    def normalized(
        cls,
        *,
        dataset: str,
        stage: str,
        dataset_root: os.PathLike[str] | str,
        feature_cache: os.PathLike[str] | str,
        foil_cache: os.PathLike[str] | str,
        whitening: os.PathLike[str] | str,
        qwen: os.PathLike[str] | str,
        split_rows: os.PathLike[str] | str,
    ) -> "SealRequest":
        dataset_key = _DATASET_ALIASES.get(str(dataset).strip().lower())
        if dataset_key is None:
            raise SealError(
                f"unsupported Phase-3 dataset {dataset!r}; expected one of "
                f"{sorted(_DATASETS)}"
            )
        stage_key = _STAGE_ALIASES.get(str(stage).strip().lower())
        if stage_key is None:
            raise SealError(
                f"unsupported Phase-3 stage {stage!r}; expected stage1/refit"
            )

        def absolute(value: os.PathLike[str] | str) -> str:
            # abspath preserves the final path component instead of resolving a
            # symlink; the logical spelling is evidence and belongs in the seal.
            return os.path.abspath(os.fspath(value))

        return cls(
            dataset=dataset_key,
            stage=stage_key,
            dataset_root=absolute(dataset_root),
            feature_cache=absolute(feature_cache),
            foil_cache=absolute(foil_cache),
            whitening=absolute(whitening),
            qwen=absolute(qwen),
            split_rows=absolute(split_rows),
        )

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "SealRequest":
        expected = {
            "dataset", "stage", "dataset_root", "feature_cache",
            "foil_cache", "whitening", "qwen", "split_rows",
        }
        if set(value) != expected:
            raise SealError(
                "seal request fields are not the schema-v1 field set: "
                f"expected={sorted(expected)}, actual={sorted(value)}"
            )
        return cls.normalized(**dict(value))

    def as_json(self) -> dict[str, str]:
        return {
            "dataset": self.dataset,
            "stage": self.stage,
            "dataset_root": self.dataset_root,
            "feature_cache": self.feature_cache,
            "foil_cache": self.foil_cache,
            "whitening": self.whitening,
            "qwen": self.qwen,
            "split_rows": self.split_rows,
        }


@dataclass(frozen=True)
class _CachePlan:
    """Small-file/inventory admission completed before tensor hashing."""

    cache_dir: str
    role: str
    require_foils: bool
    root: dict[str, Any]
    chosen: tuple[str, ...]
    meta: dict[str, Any]


@dataclass(frozen=True)
class _HFPlan:
    """Validated immutable Hugging Face paths, before byte hashing."""

    checkpoint: str
    revision: str
    declared_weight_path: str
    declared_weight_sha256: str
    snapshot_dir: str
    snapshot_weight_path: str
    snapshot_config_path: str
    tokenizer_declared_sha256: tuple[tuple[str, str], ...]
    tokenizer_snapshot_paths: tuple[tuple[str, str], ...]


def _stat_json(value: os.stat_result) -> dict[str, int]:
    """Stable stat fields whose drift must invalidate an existing seal."""
    return {
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "mode": int(value.st_mode),
        "nlink": int(value.st_nlink),
        "uid": int(value.st_uid),
        "gid": int(value.st_gid),
        "size": int(value.st_size),
        "mtime_ns": int(value.st_mtime_ns),
        "ctime_ns": int(value.st_ctime_ns),
    }


def _content_record(size: int, digest: str) -> dict[str, Any]:
    return {
        "algorithm": HASH_ALGORITHM,
        "size": int(size),
        "sha256": digest,
    }


class _HashMemo:
    """Hash each resolved content object at most once per build.

    Resolved path is intentionally part of the key as well as inode/stat
    identity.  It keeps a hard link at an unrelated provenance path from being
    collapsed accidentally, while the feature-cache -> donor and overlay ->
    feature-cache -> donor symlink chains still converge on one key.
    """

    def __init__(self) -> None:
        self._digests: dict[tuple[Any, ...], str] = {}
        self.hashed_bytes = 0

    @staticmethod
    def _key(path: str, value: os.stat_result) -> tuple[Any, ...]:
        return (
            os.path.realpath(path),
            int(value.st_dev),
            int(value.st_ino),
            int(value.st_size),
            int(value.st_mtime_ns),
            int(value.st_ctime_ns),
        )

    def digest(self, path: str, expected: os.stat_result) -> str:
        if not stat.S_ISREG(expected.st_mode):
            raise SealError(f"input target is not a regular file: {path}")
        key = self._key(path, expected)
        cached = self._digests.get(key)
        if cached is not None:
            return cached

        flags = os.O_RDONLY
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(path, flags)
        except OSError as error:
            raise SealError(f"cannot open input without following a final link: {path}: {error}") from error
        try:
            opened = os.fstat(fd)
            if _stat_json(opened) != _stat_json(expected):
                raise SealError(f"input changed while it was being opened: {path}")
            digest = hashlib.sha256()
            while True:
                block = os.read(fd, HASH_CHUNK_BYTES)
                if not block:
                    break
                digest.update(block)
            after = os.fstat(fd)
            if _stat_json(after) != _stat_json(opened):
                raise SealError(f"input changed while it was being hashed: {path}")
        finally:
            os.close(fd)

        answer = digest.hexdigest()
        self._digests[key] = answer
        self.hashed_bytes += int(expected.st_size)
        return answer

    @property
    def unique_objects(self) -> int:
        return len(self._digests)


def _seal_file(path: os.PathLike[str] | str, memo: _HashMemo) -> dict[str, Any]:
    logical = os.path.abspath(os.fspath(path))
    try:
        before = os.lstat(logical)
    except FileNotFoundError as error:
        raise SealError(f"required input is missing: {logical}") from error

    if stat.S_ISLNK(before.st_mode):
        logical_bytes = os.fsencode(logical)
        try:
            link_bytes = os.readlink(logical_bytes)
            resolved = str(Path(logical).resolve(strict=True))
            target_before = os.stat(resolved, follow_symlinks=False)
        except (OSError, RuntimeError) as error:
            raise SealError(f"broken or cyclic input symlink: {logical}: {error}") from error
        if not stat.S_ISREG(target_before.st_mode):
            raise SealError(
                f"input symlink does not resolve to a regular file: "
                f"{logical} -> {resolved}"
            )
        digest = memo.digest(resolved, target_before)

        try:
            after = os.lstat(logical)
            link_after = os.readlink(logical_bytes)
            resolved_after = str(Path(logical).resolve(strict=True))
            target_after = os.stat(resolved_after, follow_symlinks=False)
        except (OSError, RuntimeError) as error:
            raise SealError(f"input symlink changed while sealing: {logical}: {error}") from error
        if (
            _stat_json(after) != _stat_json(before)
            or link_after != link_bytes
            or resolved_after != resolved
            or _stat_json(target_after) != _stat_json(target_before)
        ):
            raise SealError(f"input symlink changed while sealing: {logical}")

        return {
            "path": logical,
            "file_type": "symlink",
            "lstat": _stat_json(before),
            "link_text": os.fsdecode(link_bytes),
            "link_text_sha256": hashlib.sha256(link_bytes).hexdigest(),
            "resolved_target": {
                "path": resolved,
                "stat": _stat_json(target_before),
                "content": _content_record(target_before.st_size, digest),
            },
        }

    if not stat.S_ISREG(before.st_mode):
        raise SealError(f"required input is not a regular file or symlink: {logical}")
    digest = memo.digest(logical, before)
    after = os.lstat(logical)
    if _stat_json(after) != _stat_json(before):
        raise SealError(f"input changed while sealing: {logical}")
    return {
        "path": logical,
        "file_type": "regular",
        "stat": _stat_json(before),
        "content": _content_record(before.st_size, digest),
    }


def _effective_sha(record: Mapping[str, Any]) -> str:
    if record.get("file_type") == "symlink":
        return str(record["resolved_target"]["content"]["sha256"])
    return str(record["content"]["sha256"])


def _read_json_bound(path: str, record: Mapping[str, Any], memo: _HashMemo) -> Any:
    """Parse a small JSON file and prove those were the sealed bytes."""
    try:
        with open(path, "rb") as handle:
            raw = handle.read()
    except OSError as error:
        raise SealError(f"cannot read JSON input {path}: {error}") from error
    if hashlib.sha256(raw).hexdigest() != _effective_sha(record):
        raise SealError(f"JSON input changed between hashing and parsing: {path}")
    expected_record = dict(record)
    # Cache-group entries add a logical_name label around the underlying file
    # record.  It is not returned by _seal_file itself.
    expected_record.pop("logical_name", None)
    if _seal_file(path, memo) != expected_record:
        raise SealError(f"JSON input stat changed between hashing and parsing: {path}")
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SealError(f"invalid UTF-8 JSON input {path}: {error}") from error


def _read_small_json_preflight(path: str, *, description: str) -> tuple[Any, str]:
    """Read one bounded JSON input with stat/link stability, without ``memo``.

    This intentionally runs before `_HashMemo.digest`: malformed row identity
    must not be discovered only after hundreds of gigabytes of tensors have
    been read.  The normal seal pass still records the independent raw-byte
    SHA/stat for this file and proves that the same bytes were parsed.
    """
    logical = os.path.abspath(path)
    try:
        logical_before = os.lstat(logical)
        resolved_before = str(Path(logical).resolve(strict=True))
        target_before = os.stat(resolved_before, follow_symlinks=False)
        link_before = (
            os.readlink(os.fsencode(logical))
            if stat.S_ISLNK(logical_before.st_mode)
            else None
        )
    except (OSError, RuntimeError) as error:
        raise SealError(f"cannot inspect {description} {logical}: {error}") from error
    if not stat.S_ISREG(target_before.st_mode):
        raise SealError(f"{description} is not a regular file: {logical}")
    if target_before.st_size > PREFLIGHT_JSON_MAX_BYTES:
        raise SealError(
            f"{description} exceeds the {PREFLIGHT_JSON_MAX_BYTES}-byte "
            f"preflight bound: {logical}"
        )

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(resolved_before, flags | getattr(os, "O_NOFOLLOW", 0))
    except OSError as error:
        raise SealError(f"cannot open {description} {logical}: {error}") from error
    try:
        opened = os.fstat(fd)
        if _stat_json(opened) != _stat_json(target_before):
            raise SealError(f"{description} changed while opening: {logical}")
        chunks: list[bytes] = []
        total = 0
        while True:
            block = os.read(fd, min(1024 * 1024, PREFLIGHT_JSON_MAX_BYTES + 1 - total))
            if not block:
                break
            chunks.append(block)
            total += len(block)
            if total > PREFLIGHT_JSON_MAX_BYTES:
                raise SealError(
                    f"{description} exceeds the preflight byte bound: {logical}"
                )
        after_read = os.fstat(fd)
        if _stat_json(after_read) != _stat_json(opened):
            raise SealError(f"{description} changed while reading: {logical}")
    finally:
        os.close(fd)
    raw = b"".join(chunks)

    try:
        logical_after = os.lstat(logical)
        resolved_after = str(Path(logical).resolve(strict=True))
        target_after = os.stat(resolved_after, follow_symlinks=False)
        link_after = (
            os.readlink(os.fsencode(logical))
            if stat.S_ISLNK(logical_after.st_mode)
            else None
        )
    except (OSError, RuntimeError) as error:
        raise SealError(f"{description} changed after reading: {logical}: {error}") from error
    if (
        _stat_json(logical_after) != _stat_json(logical_before)
        or resolved_after != resolved_before
        or _stat_json(target_after) != _stat_json(target_before)
        or link_after != link_before
    ):
        raise SealError(f"{description} changed during preflight: {logical}")
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SealError(f"invalid UTF-8 JSON {description} {logical}: {error}") from error
    return parsed, hashlib.sha256(raw).hexdigest()


def _directory_identity(path: str) -> dict[str, Any]:
    """Bind an explicit cache/root spelling without hashing directory entries."""
    logical = os.path.abspath(path)
    try:
        value = os.lstat(logical)
    except FileNotFoundError as error:
        raise SealError(f"required input directory is missing: {logical}") from error
    if stat.S_ISLNK(value.st_mode):
        link_bytes = os.readlink(os.fsencode(logical))
        try:
            resolved = str(Path(logical).resolve(strict=True))
            target = os.stat(resolved, follow_symlinks=False)
        except (OSError, RuntimeError) as error:
            raise SealError(f"broken input-directory symlink {logical}: {error}") from error
        if not stat.S_ISDIR(target.st_mode):
            raise SealError(f"input-directory symlink target is not a directory: {logical}")
        return {
            "path": logical,
            "path_type": "symlink",
            "lstat_identity": {
                key: _stat_json(value)[key]
                for key in ("device", "inode", "mode", "size", "mtime_ns", "ctime_ns")
            },
            "link_text": os.fsdecode(link_bytes),
            "link_text_sha256": hashlib.sha256(link_bytes).hexdigest(),
            "resolved_path": resolved,
            "target_identity": {
                key: _stat_json(target)[key]
                for key in ("device", "inode", "mode")
            },
        }
    if not stat.S_ISDIR(value.st_mode):
        raise SealError(f"required input directory is not a directory: {logical}")
    return {
        "path": logical,
        "path_type": "directory",
        "resolved_path": os.path.realpath(logical),
        "identity": {
            key: _stat_json(value)[key]
            for key in ("device", "inode", "mode")
        },
    }


def _entries(cache_dir: str) -> set[str]:
    try:
        with os.scandir(cache_dir) as iterator:
            return {
                entry.name
                for entry in iterator
                if entry.is_file(follow_symlinks=True) or entry.is_symlink()
            }
    except OSError as error:
        raise SealError(f"cannot inventory cache directory {cache_dir}: {error}") from error


def _cache_inventory_preflight(
    cache_dir: str,
    *,
    role: str,
    require_foils: bool,
) -> _CachePlan:
    root = _directory_identity(cache_dir)
    names = _entries(cache_dir)

    required = set(_CACHE_CORE) | set(_FACTUAL_TOKEN_PAIR)
    missing = sorted(required - names)
    if missing:
        raise SealError(f"{role} cache is missing required loader files: {missing}")

    # The loader discovers numbered views.  A half-pair, a gap, or a file beyond
    # meta.save_aug_views would change/fall out of the consumed input set.
    actual_aug: dict[str, set[int]] = {"tokens": set(), "global": set()}
    for name in names:
        match = _AUGMENTED_RE.fullmatch(name)
        if match:
            actual_aug[match.group(1)].add(int(match.group(2)))

    meta_path = os.path.join(cache_dir, "meta.json")
    meta, _ = _read_small_json_preflight(
        meta_path, description=f"{role} cache metadata"
    )
    if not isinstance(meta, Mapping):
        raise SealError(f"{role} cache meta.json is not a JSON object")
    declared = meta.get("save_aug_views")
    if isinstance(declared, bool) or not isinstance(declared, int) or declared < 2:
        raise SealError(
            f"{role} cache meta.save_aug_views={declared!r}; canonical Phase 3 "
            "requires at least two cached paired-augmentation views"
        )
    expected_aug = set(range(declared))
    if actual_aug["tokens"] != actual_aug["global"]:
        raise SealError(
            f"{role} cache has a gap/orphan augmented-view half-pair: "
            f"{actual_aug}"
        )
    if actual_aug["tokens"] != expected_aug:
        raise SealError(
            f"{role} cache augmented views have a gap/orphan or disagree with "
            f"meta.save_aug_views: expected={sorted(expected_aug)}, "
            f"actual={sorted(actual_aug['tokens'])}"
        )

    foil_present = names.intersection(_FOIL_RUNTIME)
    if require_foils:
        missing_foil = sorted(set(_FOIL_RUNTIME) - names)
        if missing_foil:
            raise SealError(f"foil overlay is missing runtime foil files: {missing_foil}")
    elif foil_present and foil_present != set(_FOIL_RUNTIME):
        raise SealError(
            f"{role} cache has an orphan/incomplete foil sidecar group: "
            f"present={sorted(foil_present)}, missing="
            f"{sorted(set(_FOIL_RUNTIME) - foil_present)}"
        )

    chosen = set(required)
    for index in sorted(expected_aug):
        chosen.add(f"visual_tokens_aug{index}.f16.npy")
        chosen.add(f"visual_global_aug{index}.f16.npy")
    if require_foils or foil_present:
        chosen.update(_FOIL_RUNTIME)
    if require_foils:
        missing_provenance = sorted(set(_FOIL_PROVENANCE) - names)
        if missing_provenance:
            raise SealError(
                f"foil overlay is missing derivation provenance files: "
                f"{missing_provenance}"
            )
        chosen.update(_FOIL_PROVENANCE)

    return _CachePlan(
        cache_dir=os.path.abspath(cache_dir),
        role=role,
        require_foils=require_foils,
        root=root,
        chosen=tuple(sorted(chosen)),
        meta=dict(meta),
    )


def _hf_provenance_preflight(
    request: SealRequest, meta: Mapping[str, Any]
) -> _HFPlan:
    """Validate declared CLIP identity and the exact tokenizer snapshot set."""
    checkpoint = meta.get("backbone")
    tokenizer = meta.get("tokenizer")
    if not isinstance(checkpoint, str) or not checkpoint:
        raise SealError("feature-cache metadata has no backbone checkpoint")
    if tokenizer != checkpoint:
        raise SealError(
            f"feature-cache tokenizer/backbone mismatch: {tokenizer!r} != "
            f"{checkpoint!r}"
        )

    hf = meta.get("hf_provenance")
    if not isinstance(hf, Mapping):
        raise SealError("feature-cache metadata has no hf_provenance object")
    if hf.get("checkpoint") != checkpoint:
        raise SealError("hf_provenance checkpoint differs from top-level backbone")
    revision = hf.get("model_revision")
    if not isinstance(revision, str) or _REVISION_RE.fullmatch(revision) is None:
        raise SealError("hf_provenance model_revision is not immutable 40-hex")

    expected_transform = (
        {
            "resize": [224, 224],
            "interpolation": "bicubic",
            "normalization": "openai_clip",
        }
        if request.dataset == "cifar10"
        else {
            "resize": [224, 224],
            "interpolation": "bilinear",
            "crop": False,
            "normalization": "openai_clip",
        }
    )
    if meta.get("canonical_transform") != expected_transform:
        raise SealError(
            "feature-cache canonical_transform is not the audited dataset-specific "
            f"CLIP preprocessing spec: actual={meta.get('canonical_transform')!r}, "
            f"expected={expected_transform!r}"
        )

    declared_weight = hf.get("model_weight_file")
    declared_weight_sha = hf.get("model_weight_sha256")
    if not isinstance(declared_weight, str) or not os.path.isabs(declared_weight):
        raise SealError("hf_provenance model_weight_file must be an absolute path")
    if (
        not isinstance(declared_weight_sha, str)
        or _SHA256_RE.fullmatch(declared_weight_sha) is None
    ):
        raise SealError("hf_provenance model_weight_sha256 is not 64-hex")
    try:
        real_weight = Path(declared_weight).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise SealError(
            f"declared Hugging Face model weight is unavailable: {declared_weight}: "
            f"{error}"
        ) from error
    if not real_weight.is_file():
        raise SealError("declared Hugging Face model weight is not a regular file")

    # `_resolve_hf_provenance` records the immutable blob.  Reconstruct the
    # snapshot aliases to prove which revision and logical artifact names led
    # to that blob, rather than trusting a self-reported digest/path pair.
    if real_weight.parent.name != "blobs":
        raise SealError(
            "model_weight_file does not resolve inside a Hugging Face blobs directory"
        )
    repo_root = real_weight.parent.parent
    expected_repo_name = "models--" + checkpoint.replace("/", "--")
    if repo_root.name != expected_repo_name:
        raise SealError(
            "model_weight_file Hugging Face repository does not match checkpoint: "
            f"{repo_root.name!r} != {expected_repo_name!r}"
        )
    snapshot = repo_root / "snapshots" / revision
    if not snapshot.is_dir():
        raise SealError(
            f"immutable Hugging Face snapshot directory is missing: {snapshot}"
        )
    weight_aliases = [
        snapshot / name for name in _HF_WEIGHT_FILES if (snapshot / name).exists()
    ]
    if len(weight_aliases) != 1:
        raise SealError(
            "immutable Hugging Face snapshot must expose exactly one supported "
            f"model weight file, found {[path.name for path in weight_aliases]}"
        )
    snapshot_weight = weight_aliases[0]
    if snapshot_weight.resolve(strict=True) != real_weight:
        raise SealError(
            "snapshot model checkpoint does not resolve to model_weight_file"
        )

    declared_tokenizers = hf.get("tokenizer_files_sha256")
    if not isinstance(declared_tokenizers, Mapping):
        raise SealError("hf_provenance tokenizer_files_sha256 is not an object")
    normalized_declared: dict[str, str] = {}
    resolved_declared: dict[str, str] = {}
    for raw_path, raw_sha in declared_tokenizers.items():
        if (
            not isinstance(raw_path, str)
            or not os.path.isabs(raw_path)
            or not isinstance(raw_sha, str)
            or _SHA256_RE.fullmatch(raw_sha) is None
        ):
            raise SealError("invalid tokenizer_files_sha256 path/digest entry")
        path = os.path.abspath(raw_path)
        try:
            resolved = str(Path(path).resolve(strict=True))
        except (OSError, RuntimeError) as error:
            raise SealError(f"declared tokenizer file is unavailable: {path}: {error}") from error
        if not Path(resolved).is_file():
            raise SealError(f"declared tokenizer file is not regular: {path}")
        if path in normalized_declared or resolved in resolved_declared:
            raise SealError("tokenizer provenance contains a duplicate file/target")
        normalized_declared[path] = raw_sha
        resolved_declared[resolved] = raw_sha

    snapshot_tokenizers: list[tuple[str, str]] = []
    for name in _HF_TOKENIZER_FILES:
        logical = snapshot / name
        try:
            resolved = str(logical.resolve(strict=True))
        except (OSError, RuntimeError) as error:
            raise SealError(
                f"immutable snapshot tokenizer set is missing {name}: {error}"
            ) from error
        snapshot_tokenizers.append((name, str(logical)))
        if resolved not in resolved_declared:
            raise SealError(
                f"tokenizer_files_sha256 omits immutable snapshot file {name}"
            )
    snapshot_targets = {
        str(Path(path).resolve(strict=True)) for _, path in snapshot_tokenizers
    }
    if snapshot_targets != set(resolved_declared):
        raise SealError(
            "tokenizer_files_sha256 is not the exact five-file tokenizer snapshot set"
        )

    config_path = snapshot / _HF_CONFIG_FILE
    try:
        resolved_config = config_path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise SealError(
            f"immutable Hugging Face snapshot config is missing: {error}"
        ) from error
    if not resolved_config.is_file():
        raise SealError(
            "immutable Hugging Face snapshot config is not a regular file"
        )

    return _HFPlan(
        checkpoint=checkpoint,
        revision=revision,
        declared_weight_path=os.path.abspath(declared_weight),
        declared_weight_sha256=declared_weight_sha,
        snapshot_dir=str(snapshot.resolve(strict=True)),
        snapshot_weight_path=str(snapshot_weight),
        snapshot_config_path=str(config_path),
        tokenizer_declared_sha256=tuple(sorted(normalized_declared.items())),
        tokenizer_snapshot_paths=tuple(snapshot_tokenizers),
    )


def _seal_hf_provenance(plan: _HFPlan, memo: _HashMemo) -> dict[str, Any]:
    """Hash declared blobs and their immutable snapshot aliases once each."""
    candidates: dict[str, list[str]] = {}

    def add(path: str, role: str) -> None:
        candidates.setdefault(os.path.abspath(path), []).append(role)

    add(plan.declared_weight_path, "declared_model_weight")
    add(plan.snapshot_weight_path, "snapshot_model_weight")
    add(plan.snapshot_config_path, "snapshot_config:config.json")
    declared_token_sha = dict(plan.tokenizer_declared_sha256)
    resolved_token_sha = {
        str(Path(path).resolve(strict=True)): digest
        for path, digest in plan.tokenizer_declared_sha256
    }
    for path, _ in plan.tokenizer_declared_sha256:
        add(path, "declared_tokenizer_blob")
    for name, path in plan.tokenizer_snapshot_paths:
        add(path, f"snapshot_tokenizer:{name}")

    records: list[dict[str, Any]] = []
    for path in sorted(candidates):
        record = _seal_file(path, memo)
        roles = sorted(candidates[path])
        expected: str | None = None
        if "declared_model_weight" in roles or "snapshot_model_weight" in roles:
            expected = plan.declared_weight_sha256
        if any(role.startswith("declared_tokenizer") for role in roles):
            expected = declared_token_sha[path]
        if any(role.startswith("snapshot_tokenizer:") for role in roles):
            token_expected = resolved_token_sha[os.path.realpath(path)]
            if expected is not None and expected != token_expected:
                raise SealError("conflicting declared digest for tokenizer alias")
            expected = token_expected
        # config.json has no self-reported digest in the legacy extractor
        # metadata.  Its immutable revision path plus the seal's own content
        # hash is the authority; every other HF input retains the extractor's
        # independently declared digest check.
        if expected is None and roles == ["snapshot_config:config.json"]:
            expected = _effective_sha(record)
        if expected is None or _effective_sha(record) != expected:
            raise SealError(
                f"Hugging Face artifact bytes do not match declared SHA-256: {path}"
            )
        records.append({"logical_roles": roles, **record})

    return {
        "checkpoint": plan.checkpoint,
        "model_revision": plan.revision,
        "snapshot_dir": plan.snapshot_dir,
        "snapshot_contract": {
            "config_file": _HF_CONFIG_FILE,
            "model_weight_names": list(_HF_WEIGHT_FILES),
            "selected_model_weight": os.path.basename(plan.snapshot_weight_path),
            "tokenizer_file_set": list(_HF_TOKENIZER_FILES),
        },
        "files": records,
    }


def _seal_cache_inventory(
    plan: _CachePlan, memo: _HashMemo
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Hash the files admitted by the small-file/inventory preflight."""
    meta_path = os.path.join(plan.cache_dir, "meta.json")
    meta_record = _seal_file(meta_path, memo)
    sealed_meta = _read_json_bound(meta_path, meta_record, memo)
    if not isinstance(sealed_meta, Mapping) or dict(sealed_meta) != plan.meta:
        raise SealError(f"{plan.role} cache metadata changed after preflight")

    files = []
    for name in plan.chosen:
        record = meta_record if name == "meta.json" else _seal_file(
            os.path.join(plan.cache_dir, name), memo
        )
        files.append({"logical_name": name, **record})
    return {"root": plan.root, "files": files}, dict(sealed_meta)


def _strict_image_ids(path: str, *, description: str) -> tuple[list[str], str]:
    value, raw_sha = _read_small_json_preflight(path, description=description)
    if not isinstance(value, list) or not value:
        raise SealError(f"{description} must be a non-empty JSON list[str]")
    if any(not isinstance(item, str) or not item for item in value):
        raise SealError(f"{description} must contain only non-empty strings")
    if len(set(value)) != len(value):
        raise SealError(f"{description} contains duplicate image identifiers")
    return value, raw_sha


def _image_id_alignment_preflight(
    request: SealRequest,
    *,
    feature_plan: _CachePlan,
    overlay_plan: _CachePlan,
) -> dict[str, Any]:
    """Validate loader-visible row identity before any tensor is hashed."""
    logical = (
        (
            "feature_image_ids",
            os.path.join(request.feature_cache, "image_ids.json"),
        ),
        (
            "overlay_image_ids",
            os.path.join(request.foil_cache, "image_ids.json"),
        ),
        (
            "pooled_foil_image_ids",
            os.path.join(request.foil_cache, "text_foil_image_ids.json"),
        ),
        (
            "token_foil_image_ids",
            os.path.join(request.foil_cache, "text_foil_token_image_ids.json"),
        ),
    )
    parsed: dict[str, list[str]] = {}
    raw_sha: dict[str, str] = {}
    for role, path in logical:
        parsed[role], raw_sha[role] = _strict_image_ids(
            path, description=role
        )
    expected = parsed["feature_image_ids"]
    for role, _ in logical[1:]:
        if parsed[role] != expected:
            raise SealError(
                f"{role} does not exactly match feature image ids by element/order"
            )

    # Cache metadata is semantic JSON too.  Compare parsed objects here so a
    # mismatch refuses before large-array I/O; raw-byte equality remains a
    # separate, stronger provenance assertion in the final manifest contract.
    if feature_plan.meta != overlay_plan.meta:
        raise SealError(
            "foil overlay parsed metadata differs from feature-cache metadata"
        )
    qwen_name = feature_plan.meta.get("qwen_cache")
    if (
        not isinstance(qwen_name, str)
        or os.path.basename(qwen_name) != os.path.basename(request.qwen)
    ):
        raise SealError(
            f"feature-cache meta names qwen_cache={qwen_name!r}, not the "
            f"explicit Qwen file {os.path.basename(request.qwen)!r}"
        )

    # The derivation metadata intentionally stores the raw SHA of the base
    # image_ids.json.  Check those claims now, but do not require foil JSON
    # formatting to reproduce that raw byte stream.
    semantic, _ = _read_small_json_preflight(
        os.path.join(request.foil_cache, "semantic_detail_cache_manifest.json"),
        description="semantic-detail manifest",
    )
    pooled_meta, _ = _read_small_json_preflight(
        os.path.join(request.foil_cache, "text_foil_meta.json"),
        description="pooled foil metadata",
    )
    token_meta, _ = _read_small_json_preflight(
        os.path.join(request.foil_cache, "text_foil_token_meta.json"),
        description="token foil metadata",
    )
    for description, value in (
        ("semantic-detail manifest", semantic),
        ("pooled foil metadata", pooled_meta),
        ("token foil metadata", token_meta),
    ):
        if not isinstance(value, Mapping):
            raise SealError(f"{description} is not a JSON object")
    feature_raw_sha = raw_sha["feature_image_ids"]
    if semantic.get("image_ids_sha256") != feature_raw_sha:
        raise SealError(
            "foil manifest image_ids_sha256 does not match feature image_ids raw bytes"
        )
    for description, value in (
        ("pooled foil metadata", pooled_meta),
        ("token foil metadata", token_meta),
    ):
        if value.get("cache_image_ids_sha256") != feature_raw_sha:
            raise SealError(
                f"{description} image-id digest is not bound to feature rows"
            )

    canonical = json.dumps(
        expected,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    semantic_sha = hashlib.sha256(canonical).hexdigest()
    return {
        "algorithm": HASH_ALGORITHM,
        "canonicalization": (
            "UTF-8 JSON list[str], exact strings/order, ensure_ascii=false, "
            "separators=(',', ':')"
        ),
        "row_count": len(expected),
        "sha256": semantic_sha,
        "logical_files": [
            {
                "role": role,
                "path": os.path.abspath(path),
                "raw_sha256": raw_sha[role],
                "semantic_sha256": semantic_sha,
            }
            for role, path in logical
        ],
    }


def _read_regular_bytes_stable(
    path: os.PathLike[str] | str, *, description: str
) -> tuple[bytes, str]:
    """Read one regular file while binding logical link and target identity."""
    logical = os.path.abspath(os.fspath(path))
    try:
        logical_before = os.lstat(logical)
        resolved_before = str(Path(logical).resolve(strict=True))
        target_before = os.stat(resolved_before, follow_symlinks=False)
        link_before = (
            os.readlink(os.fsencode(logical))
            if stat.S_ISLNK(logical_before.st_mode)
            else None
        )
    except (OSError, RuntimeError) as error:
        raise SealError(f"cannot inspect {description} {logical}: {error}") from error
    if not stat.S_ISREG(target_before.st_mode):
        raise SealError(f"{description} is not a regular file: {logical}")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(resolved_before, flags)
    except OSError as error:
        raise SealError(f"cannot open {description} {logical}: {error}") from error
    try:
        opened = os.fstat(fd)
        if _stat_json(opened) != _stat_json(target_before):
            raise SealError(f"{description} changed while opening: {logical}")
        chunks: list[bytes] = []
        while True:
            block = os.read(fd, HASH_CHUNK_BYTES)
            if not block:
                break
            chunks.append(block)
        after_read = os.fstat(fd)
        if _stat_json(after_read) != _stat_json(opened):
            raise SealError(f"{description} changed while reading: {logical}")
    finally:
        os.close(fd)
    raw = b"".join(chunks)

    try:
        logical_after = os.lstat(logical)
        resolved_after = str(Path(logical).resolve(strict=True))
        target_after = os.stat(resolved_after, follow_symlinks=False)
        link_after = (
            os.readlink(os.fsencode(logical))
            if stat.S_ISLNK(logical_after.st_mode)
            else None
        )
    except (OSError, RuntimeError) as error:
        raise SealError(f"{description} changed after reading: {logical}: {error}") from error
    if (
        _stat_json(logical_after) != _stat_json(logical_before)
        or resolved_after != resolved_before
        or _stat_json(target_after) != _stat_json(target_before)
        or link_after != link_before
    ):
        raise SealError(f"{description} changed while it was read: {logical}")
    return raw, hashlib.sha256(raw).hexdigest()


def _row_vector_sha256(rows: np.ndarray) -> str:
    value = np.ascontiguousarray(rows, dtype="<i8").reshape(-1)
    digest = hashlib.sha256()
    digest.update(_ROW_VECTOR_ENCODING.encode("ascii"))
    digest.update(b"\0")
    digest.update(len(value).to_bytes(8, "little", signed=False))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _typed_array_sha256(value: np.ndarray, *, semantic: str) -> str:
    array = np.ascontiguousarray(value)
    header = json.dumps(
        {
            "semantic": semantic,
            "dtype": array.dtype.str,
            "shape": list(array.shape),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(b"groundeddna.phase3-array-v1\0")
    digest.update(header)
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _non_cifar_split_contract(
    request: SealRequest, split_name: str, *, require_labels: bool
) -> tuple[list[str], np.ndarray | None, dict[str, Any]]:
    canonical, _ = _DATASETS[request.dataset]
    split_path = os.path.join(
        request.dataset_root, canonical, "setting1", f"{split_name}.txt"
    )
    raw, raw_sha = _read_regular_bytes_stable(
        split_path, description=f"official {split_name} rows"
    )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SealError(
            f"official {split_name} rows are not UTF-8: {split_path}: {error}"
        ) from error
    image_ids: list[str] = []
    labels: list[list[int]] = []
    dataset_dir = os.path.join(request.dataset_root, canonical)
    for line_number, line in enumerate(text.splitlines(), 1):
        fields = line.split()
        if not fields or (require_labels and len(fields) < 2):
            raise SealError(
                f"official {split_name} row {line_number} has no path"
                + (" plus labels" if require_labels else "")
            )
        image_ids.append(
            os.path.relpath(os.path.join(dataset_dir, fields[0]), dataset_dir)
        )
        if require_labels:
            try:
                labels.append([int(item) for item in fields[1:]])
            except ValueError as error:
                raise SealError(
                    f"official {split_name} row {line_number} has a non-integer label"
                ) from error
    if not image_ids or len(set(image_ids)) != len(image_ids):
        raise SealError(f"official {split_name} rows are empty or contain duplicates")
    label_array: np.ndarray | None = None
    if require_labels:
        if not labels or len({len(row) for row in labels}) != 1:
            raise SealError("designated train labels are empty or non-rectangular")
        label_array = np.asarray(labels, dtype=np.int64)
        if not bool(((label_array == 0) | (label_array == 1)).all()):
            raise SealError("designated train labels are not binary")
    return image_ids, label_array, {
        "path": os.path.abspath(split_path), "sha256": raw_sha
    }


def _non_cifar_train_contract(
    request: SealRequest,
) -> tuple[np.ndarray, list[str], list[dict[str, Any]]]:
    image_ids, labels, source = _non_cifar_split_contract(
        request, "train", require_labels=True
    )
    assert labels is not None
    return labels, image_ids, [source]


def _cifar_train_contract(
    request: SealRequest,
) -> tuple[np.ndarray, list[str], list[dict[str, Any]]]:
    canonical, _ = _DATASETS[request.dataset]
    base = os.path.join(
        request.dataset_root, canonical, "cifar-10-batches-py"
    )
    images: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    sources: list[dict[str, Any]] = []
    for index in range(1, 6):
        path = os.path.join(base, f"data_batch_{index}")
        raw, raw_sha = _read_regular_bytes_stable(
            path, description=f"CIFAR train batch {index}"
        )
        try:
            payload = pickle.loads(raw, encoding="latin1")
            data = np.asarray(payload["data"], dtype=np.uint8)
            labels = np.asarray(payload["labels"], dtype=np.int64)
        except (KeyError, TypeError, ValueError, pickle.UnpicklingError) as error:
            raise SealError(f"invalid CIFAR train batch {path}: {error}") from error
        if data.ndim != 2 or data.shape[1] != 3 * 32 * 32 or len(data) != len(labels):
            raise SealError(f"invalid CIFAR train batch geometry: {path}")
        images.append(data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1))
        targets.append(labels)
        sources.append({"path": os.path.abspath(path), "sha256": raw_sha})
    all_images = np.concatenate(images, axis=0)
    all_targets = np.concatenate(targets, axis=0)

    # Exact dataloaders.get_idx_for_uniform_sampling(setting1 train) rule.
    selected: list[np.ndarray] = []
    for label in range(10):
        rows = np.where(all_targets == label)[0]
        permutation = np.random.RandomState(0).permutation(len(rows))
        selected.append(rows[permutation[:500]])
    chosen = np.concatenate(selected)
    selected_images = all_images[chosen]
    selected_targets = all_targets[chosen]
    image_ids = [
        hashlib.md5(np.ascontiguousarray(image).tobytes()).hexdigest()[:16]
        for image in selected_images
    ]
    return selected_targets, image_ids, sources


def _split_identity_preflight(
    request: SealRequest,
    *,
    feature_image_ids: Sequence[str],
    feature_image_ids_semantic_sha256: str,
) -> dict[str, Any]:
    """Recompute the exact canonical train carve and bind its row vectors."""
    expected_name = (
        "opt_train_rows.npy" if request.stage == "stage1"
        else "train_all_rows.npy"
    )
    if os.path.basename(request.split_rows) != expected_name:
        raise SealError(
            f"{request.stage} requires {expected_name}, got "
            f"{os.path.basename(request.split_rows)!r}"
        )

    if request.dataset == "cifar10":
        labels, train_image_ids, label_sources = _cifar_train_contract(request)
    else:
        labels, train_image_ids, label_sources = _non_cifar_train_contract(request)

    cache_row = {image_id: index for index, image_id in enumerate(feature_image_ids)}
    if len(cache_row) != len(feature_image_ids):
        raise SealError("feature image ids contain duplicates during split verification")
    missing = sorted(set(train_image_ids) - set(cache_row))
    if missing:
        raise SealError(
            f"{len(missing)} designated train rows are absent from feature image ids"
        )
    dataset_to_cache = np.asarray(
        [cache_row[image_id] for image_id in train_image_ids], dtype=np.int64
    )
    if np.unique(dataset_to_cache).size != len(dataset_to_cache):
        raise SealError("designated train rows map to duplicate feature-cache rows")

    val_split_path = REPO / "val_split.py"
    val_split_raw, val_split_sha = _read_regular_bytes_stable(
        val_split_path, description="canonical val split source"
    )
    val_split_namespace: dict[str, Any] = {
        "__file__": str(val_split_path),
        "__name__": "_groundeddna_sealed_val_split",
    }
    try:
        exec(
            compile(val_split_raw, str(val_split_path), "exec"),
            val_split_namespace,
        )
        carve_val_indices = val_split_namespace["carve_val_indices"]
    except (KeyError, SyntaxError, TypeError) as error:
        raise SealError(f"cannot execute canonical val_split.py bytes: {error}") from error

    if request.stage == "stage1":
        opt_idx, val_idx, strategy = carve_val_indices(
            labels, ratio=0.1, seed=42
        )
        ratio = 0.1
    else:
        # Refit consumes every designated training dataset index exactly once.
        opt_idx = np.arange(len(labels), dtype=np.int64)
        val_idx = np.empty(0, dtype=np.int64)
        strategy = "all designated train rows (refit; no validation carve)"
        ratio = 0.0

    expected_opt = np.sort(np.unique(dataset_to_cache[opt_idx])).astype(np.int64)
    expected_val = np.sort(np.unique(dataset_to_cache[val_idx])).astype(np.int64)
    source_rows = np.sort(dataset_to_cache).astype(np.int64)
    if expected_opt.size != len(opt_idx) or expected_val.size != len(val_idx):
        raise SealError("canonical split lost rows while mapping into feature cache")
    if np.intersect1d(expected_opt, expected_val, assume_unique=True).size:
        raise SealError("canonical optimization/validation cache rows overlap")
    if not np.array_equal(
        np.sort(np.concatenate((expected_opt, expected_val))), source_rows
    ):
        raise SealError("canonical optimization/validation rows do not partition train")

    split_raw, split_raw_sha = _read_regular_bytes_stable(
        request.split_rows, description="explicit split-row array"
    )
    try:
        actual = np.load(io.BytesIO(split_raw), allow_pickle=False)
    except (OSError, ValueError) as error:
        raise SealError(f"split-row input is not a readable NPY array: {error}") from error
    if actual.ndim != 1 or actual.dtype != np.int64:
        raise SealError(
            f"split-row array must be int64 [N], got {actual.shape}/{actual.dtype}"
        )
    if not np.array_equal(actual, expected_opt):
        raise SealError(
            f"{expected_name} is not the exact ordered canonical {request.stage} "
            f"row vector (actual={len(actual)}, expected={len(expected_opt)})"
        )

    evidence: dict[str, Any] = {
        "schema": SPLIT_IDENTITY_SCHEMA,
        "schema_version": SPLIT_IDENTITY_SCHEMA_VERSION,
        "dataset": request.dataset,
        "stage": request.stage,
        "val_split_ratio": ratio,
        "val_split_seed": 42,
        "strategy": strategy,
        "row_vector_encoding": _ROW_VECTOR_ENCODING,
        "counts": {
            "designated_train": int(len(labels)),
            "optimization_train": int(len(expected_opt)),
            "heldout_train_validation": int(len(expected_val)),
        },
        "digests": {
            "dataset_to_cache_rows_sha256": _row_vector_sha256(dataset_to_cache),
            "source_cache_rows_sha256": _row_vector_sha256(source_rows),
            "optimization_cache_rows_sha256": _row_vector_sha256(expected_opt),
            "validation_cache_rows_sha256": _row_vector_sha256(expected_val),
            "train_labels_sha256": _typed_array_sha256(
                np.asarray(labels), semantic="designated_train_labels"
            ),
            "feature_image_ids_semantic_sha256": feature_image_ids_semantic_sha256,
        },
        "split_rows": {
            "path": os.path.abspath(request.split_rows),
            "raw_sha256": split_raw_sha,
        },
        "label_sources": label_sources,
        "protocol_source": {
            "path": str(val_split_path),
            "sha256": val_split_sha,
        },
    }
    evidence["identity_sha256"] = hashlib.sha256(
        json.dumps(
            evidence,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    return evidence


def _decode_failure_audit_preflight(
    request: SealRequest,
    *,
    feature_meta: Mapping[str, Any],
    feature_image_ids: Sequence[str],
) -> dict[str, Any]:
    """Bind extractor decode failures to official split membership."""
    has_count = "n_failed_image_decode" in feature_meta
    has_indices = "failed_image_indices" in feature_meta
    if has_count != has_indices:
        raise SealError(
            "feature metadata must declare decode-failure count and indices together"
        )
    if not has_count:
        if request.dataset != "cifar10":
            raise SealError(
                "non-CIFAR feature metadata omits decode-failure provenance"
            )
        count: Any = 0
        indices_value: Any = []
        encoding = "absent_means_zero_legacy_cifar_extractor"
    else:
        count = feature_meta.get("n_failed_image_decode")
        indices_value = feature_meta.get("failed_image_indices")
        encoding = "explicit_meta_fields"
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise SealError(f"invalid n_failed_image_decode={count!r}")
    if not isinstance(indices_value, list) or any(
        isinstance(item, bool) or not isinstance(item, int)
        for item in indices_value
    ):
        raise SealError("failed_image_indices must be a JSON list[int]")
    failed = np.asarray(indices_value, dtype=np.int64)
    if len(failed) != count:
        raise SealError(
            "decode-failure count does not equal failed_image_indices length"
        )
    if len(failed) and (
        not np.array_equal(failed, np.unique(failed))
        or int(failed[0]) < 0
        or int(failed[-1]) >= len(feature_image_ids)
    ):
        raise SealError(
            "failed_image_indices must be strictly increasing, unique cache rows "
            "within image_ids bounds"
        )

    source_records: list[dict[str, Any]] = []
    split_rows: dict[str, set[int]] = {
        "train": set(), "query": set(), "database": set()
    }
    if request.dataset != "cifar10":
        cache_row = {
            image_id: index for index, image_id in enumerate(feature_image_ids)
        }
        for split_name, evidence_name in (
            ("train", "train"),
            ("test", "query"),
            ("database", "database"),
        ):
            ids, _, source = _non_cifar_split_contract(
                request, split_name, require_labels=False
            )
            missing = [image_id for image_id in ids if image_id not in cache_row]
            if missing:
                raise SealError(
                    f"official {split_name} membership has {len(missing)} rows "
                    "absent from feature image ids"
                )
            split_rows[evidence_name] = {cache_row[image_id] for image_id in ids}
            source_records.append(source)

    failed_set = set(int(item) for item in failed.tolist())
    train_failed = failed_set & split_rows["train"]
    query_failed = failed_set & split_rows["query"]
    database_failed = failed_set & split_rows["database"]
    union = set().union(*split_rows.values())
    unassigned = failed_set - union
    database_only = database_failed - train_failed - query_failed

    if request.dataset != "mscoco" and failed_set:
        raise SealError(
            f"{request.dataset} is expected to have zero image decode failures"
        )
    if request.dataset == "mscoco" and failed_set:
        if train_failed or query_failed or unassigned or database_only != failed_set:
            raise SealError(
                "MSCOCO decode failures are not exclusively official database-only rows"
            )
        policy = "kept_and_disclosed_db_only"
    else:
        policy = "no_decode_failures"

    return {
        "source_encoding": encoding,
        "policy": policy,
        "sensitivity_analysis": (
            "later exclusion sensitivity; main inputs keep these rows"
            if policy == "kept_and_disclosed_db_only"
            else "not_applicable"
        ),
        "count": int(count),
        "failed_image_indices": [int(item) for item in failed.tolist()],
        "failed_indices_sha256": _row_vector_sha256(failed),
        "bounds": {
            "cache_row_count": len(feature_image_ids),
            "minimum": (int(failed[0]) if len(failed) else None),
            "maximum": (int(failed[-1]) if len(failed) else None),
        },
        "official_membership": {
            "train_count": len(train_failed),
            "query_count": len(query_failed),
            "database_count": len(database_failed),
            "database_only_count": len(database_only),
            "unassigned_count": len(unassigned),
        },
        "membership_sources": source_records,
    }


def _npy_geometry_stable(
    path: os.PathLike[str] | str, *, description: str
) -> tuple[tuple[int, ...], str]:
    """Read only an NPY header/memmap while detecting path/stat drift."""
    logical = os.path.abspath(os.fspath(path))
    try:
        logical_before = os.lstat(logical)
        resolved = str(Path(logical).resolve(strict=True))
        target_before = os.stat(resolved, follow_symlinks=False)
        link_before = (
            os.readlink(os.fsencode(logical))
            if stat.S_ISLNK(logical_before.st_mode) else None
        )
    except (OSError, RuntimeError) as error:
        raise SealError(f"cannot inspect {description}: {logical}: {error}") from error
    try:
        array = np.load(resolved, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as error:
        raise SealError(f"invalid NPY {description}: {logical}: {error}") from error
    try:
        shape = tuple(int(item) for item in array.shape)
        dtype = array.dtype.str
    finally:
        mmap = getattr(array, "_mmap", None)
        if mmap is not None:
            mmap.close()
        del array
    try:
        logical_after = os.lstat(logical)
        resolved_after = str(Path(logical).resolve(strict=True))
        target_after = os.stat(resolved_after, follow_symlinks=False)
        link_after = (
            os.readlink(os.fsencode(logical))
            if stat.S_ISLNK(logical_after.st_mode) else None
        )
    except (OSError, RuntimeError) as error:
        raise SealError(f"{description} changed during geometry read: {error}") from error
    if (
        _stat_json(logical_before) != _stat_json(logical_after)
        or resolved_after != resolved
        or _stat_json(target_before) != _stat_json(target_after)
        or link_before != link_after
    ):
        raise SealError(f"{description} changed during geometry read: {logical}")
    return shape, dtype


def _whitening_preflight(
    request: SealRequest,
    *,
    feature_meta: Mapping[str, Any],
    feature_image_ids: Sequence[str],
    split_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the selected whitening bundle against actual cache rows."""
    whitening_raw, whitening_sha = _read_regular_bytes_stable(
        request.whitening, description="selected whitening NPZ"
    )
    try:
        with np.load(io.BytesIO(whitening_raw), allow_pickle=False) as bundle:
            if set(bundle.files) != _WHITENING_KEYS:
                raise SealError(
                    f"whitening NPZ must have exact keys mu/U/S, got {bundle.files}"
                )
            mu = np.asarray(bundle["mu"])
            U = np.asarray(bundle["U"])
            S = np.asarray(bundle["S"])
    except SealError:
        raise
    except (OSError, ValueError, TypeError) as error:
        raise SealError(f"selected whitening is not a safe NPZ: {error}") from error
    for name, value in (("mu", mu), ("U", U), ("S", S)):
        if value.dtype.kind != "f":
            raise SealError(f"whitening {name} must have floating dtype")
        if not bool(np.isfinite(value).all()):
            raise SealError(f"whitening {name} contains non-finite values")
    if (
        mu.ndim != 1
        or mu.size == 0
        or U.shape != (mu.size, mu.size)
        or S.shape != mu.shape
    ):
        raise SealError(
            f"invalid whitening geometry mu={mu.shape}, U={U.shape}, S={S.shape}"
        )
    dimension = int(mu.size)
    minimum_eigenvalue = float(np.min(S))
    if minimum_eigenvalue < -_WHITENING_EIGENVALUE_NEGATIVE_ATOL:
        raise SealError(
            "whitening S has a materially negative eigenvalue: "
            f"{minimum_eigenvalue} < -{_WHITENING_EIGENVALUE_NEGATIVE_ATOL}"
        )
    gram = np.asarray(U, dtype=np.float64).T @ np.asarray(U, dtype=np.float64)
    orthonormal_max_abs_error = float(
        np.max(np.abs(gram - np.eye(dimension, dtype=np.float64)), initial=0.0)
    )
    if orthonormal_max_abs_error > _WHITENING_ORTHONORMAL_ATOL:
        raise SealError(
            "whitening U is not orthonormal within tolerance: "
            f"{orthonormal_max_abs_error} > {_WHITENING_ORTHONORMAL_ATOL}"
        )

    metadata_path = request.whitening + ".meta.json"
    metadata, metadata_sha = _read_small_json_preflight(
        metadata_path, description="selected whitening metadata"
    )
    if not isinstance(metadata, Mapping):
        raise SealError("whitening metadata is not a JSON object")
    for key in ("D", "N_kept", "rows_used"):
        value = metadata.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise SealError(f"whitening metadata {key} is invalid: {value!r}")
    if metadata.get("D") != dimension:
        raise SealError("whitening metadata D differs from NPZ geometry")
    if metadata.get("local_slots_only") is not True:
        raise SealError("whitening metadata must record local_slots_only=true")
    if metadata.get("residualize_first") is not False:
        raise SealError("local-only whitening must record residualize_first=false")
    if metadata.get("leakage_free_fit") is not True:
        raise SealError("whitening metadata must record leakage_free_fit=true")

    text_shape, text_dtype = _npy_geometry_stable(
        os.path.join(request.feature_cache, "text_part.f16.npy"),
        description="feature-cache text_part",
    )
    if len(text_shape) != 3 or text_shape[0] != len(feature_image_ids):
        raise SealError(
            f"text_part geometry does not match feature rows: {text_shape}"
        )
    if text_shape[1] != 6 or text_shape[2] != dimension:
        raise SealError(
            "whitening source must be canonical [N,M=6,D] text_part: "
            f"actual={text_shape}, whitening D={dimension}"
        )
    if text_dtype != np.dtype(np.float16).str:
        raise SealError(
            f"canonical text_part must be float16, got dtype {text_dtype}"
        )
    if feature_meta.get("N") != len(feature_image_ids):
        raise SealError("feature metadata N differs from image_ids row count")
    if feature_meta.get("D_proj") != dimension:
        raise SealError("feature metadata D_proj differs from whitening/cache D")

    has_text_path = os.path.join(request.feature_cache, "has_text.bool.npy")
    has_text_raw, has_text_sha = _read_regular_bytes_stable(
        has_text_path, description="feature-cache has_text"
    )
    try:
        has_text = np.load(io.BytesIO(has_text_raw), allow_pickle=False)
    except (OSError, ValueError) as error:
        raise SealError(f"has_text is not a readable NPY array: {error}") from error
    if has_text.dtype != np.bool_ or has_text.shape != (len(feature_image_ids),):
        raise SealError(
            "has_text must be bool [feature rows], got "
            f"{has_text.shape}/{has_text.dtype}"
        )

    split_raw, split_sha = _read_regular_bytes_stable(
        request.split_rows, description="whitening split-row array"
    )
    try:
        split_rows = np.load(io.BytesIO(split_raw), allow_pickle=False)
    except (OSError, ValueError) as error:
        raise SealError(f"whitening split rows are not readable NPY: {error}") from error
    if split_rows.dtype != np.int64 or split_rows.ndim != 1:
        raise SealError("whitening split rows must be int64 [N]")
    if (
        len(split_rows) == 0
        or int(split_rows[0]) < 0
        or int(split_rows[-1]) >= len(feature_image_ids)
        or not np.array_equal(split_rows, np.unique(split_rows))
    ):
        raise SealError("whitening split rows are not sorted unique in bounds")
    expected_split_sha = split_identity["split_rows"]["raw_sha256"]
    if split_sha != expected_split_sha:
        raise SealError("whitening split rows changed after split-identity preflight")
    expected_kept = int(has_text[split_rows].sum())
    expected_vectors = expected_kept * len(_WHITENING_SOURCE_SLOTS)
    if metadata.get("N_kept") != expected_kept:
        raise SealError(
            f"whitening metadata N_kept={metadata.get('N_kept')} does not match "
            f"has_text-selected rows={expected_kept}"
        )
    if metadata.get("rows_used") != expected_vectors:
        raise SealError(
            f"whitening metadata rows_used={metadata.get('rows_used')} does not "
            f"match {expected_kept} rows * 5 source local slots"
        )

    return {
        "artifact_raw_sha256": whitening_sha,
        "metadata_raw_sha256": metadata_sha,
        "has_text_raw_sha256": has_text_sha,
        "split_rows_raw_sha256": split_sha,
        "npz": {
            "keys": sorted(_WHITENING_KEYS),
            "dtypes": {"mu": mu.dtype.str, "U": U.dtype.str, "S": S.dtype.str},
            "D": dimension,
            "finite": True,
            "minimum_eigenvalue": minimum_eigenvalue,
            "negative_eigenvalue_atol": _WHITENING_EIGENVALUE_NEGATIVE_ATOL,
            "orthonormal_max_abs_error": orthonormal_max_abs_error,
            "orthonormal_atol": _WHITENING_ORTHONORMAL_ATOL,
        },
        "fit_geometry": {
            "feature_rows": len(feature_image_ids),
            "text_part_shape": list(text_shape),
            "text_part_dtype": text_dtype,
            "split_rows": len(split_rows),
            "has_text_true_in_split": expected_kept,
            "source_M": 6,
            "source_local_slots": list(_WHITENING_SOURCE_SLOTS),
            "source_vectors_per_kept_row": 5,
            "source_vectors_total": expected_vectors,
        },
        "effective_runtime_preprocessing": {
            "runtime_M": 5,
            "runtime_slots": [0, 1, 2, 3, 4],
            "runtime_local_slots_whitened": list(_WHITENING_RUNTIME_LOCAL_SLOTS),
            "note": (
                "sealed source whitening remains the existing M=6 slots 1..5 "
                "fit; runtime M=5 consumes local slots 1..4 without refitting"
            ),
        },
    }


def _record_by_name(group: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    matches = [entry for entry in group["files"] if entry["logical_name"] == name]
    if len(matches) != 1:
        raise SealError(f"internal inventory error for logical file {name!r}")
    return matches[0]


def _dataset_inventory(request: SealRequest, memo: _HashMemo) -> dict[str, Any]:
    canonical, _ = _DATASETS[request.dataset]
    dataset_dir = os.path.join(request.dataset_root, canonical)
    root = _directory_identity(dataset_dir)
    roles: list[tuple[str, str, str]] = []
    if request.dataset == "cifar10":
        # torchvision.CIFAR10._check_integrity reads all train and test files
        # even when stage 1 asks for train=True.  batches.meta is then opened by
        # _load_meta.  Therefore the official-test batch is an actual byte input
        # to both stages (integrity only in stage 1, data+labels in refit eval).
        base = os.path.join(dataset_dir, "cifar-10-batches-py")
        for index in range(1, 6):
            roles.append((
                f"cifar_train_batch_{index}",
                os.path.join(base, f"data_batch_{index}"),
                "runtime_consumed",
            ))
        roles.append((
            "cifar_test_batch", os.path.join(base, "test_batch"),
            "runtime_consumed",
        ))
        roles.append((
            "cifar_class_metadata", os.path.join(base, "batches.meta"),
            "runtime_consumed",
        ))
    else:
        split = os.path.join(dataset_dir, "setting1")
        roles.extend([
            (
                "train_rows", os.path.join(split, "train.txt"),
                "runtime_consumed",
            ),
            (
                "query_rows", os.path.join(split, "test.txt"),
                "runtime_consumed" if request.stage == "refit"
                else "derivation_provenance",
            ),
            (
                "database_rows", os.path.join(split, "database.txt"),
                "runtime_consumed" if request.stage == "refit"
                else "derivation_provenance",
            ),
        ])
    return {
        "root": root,
        "files": [
            {
                "logical_name": name,
                "consumption_role": consumption_role,
                **_seal_file(path, memo),
            }
            for name, path, consumption_role in roles
        ],
    }


def _assert_same_real_path(left: str, right: str, description: str) -> None:
    if os.path.realpath(left) != os.path.realpath(right):
        raise SealError(
            f"{description} mismatch: {left!r} does not resolve to {right!r}"
        )


def _assert_manifest_contract(
    request: SealRequest,
    *,
    feature: Mapping[str, Any],
    feature_meta: Mapping[str, Any],
    overlay: Mapping[str, Any],
    overlay_meta: Mapping[str, Any],
    whitening_record: Mapping[str, Any],
    whitening_meta: Mapping[str, Any],
    split_record: Mapping[str, Any],
    qwen_record: Mapping[str, Any],
    semantic_manifest: Mapping[str, Any],
    foil_meta: Mapping[str, Any],
    foil_token_meta: Mapping[str, Any],
) -> None:
    expected_split_name = (
        "opt_train_rows.npy" if request.stage == "stage1"
        else "train_all_rows.npy"
    )
    if os.path.basename(request.split_rows) != expected_split_name:
        raise SealError(
            f"{request.stage} requires {expected_split_name}, got "
            f"{os.path.basename(request.split_rows)!r}"
        )
    if os.path.dirname(os.path.realpath(request.split_rows)) != os.path.realpath(request.feature_cache):
        raise SealError("split-row index must resolve inside the explicit feature cache")

    if os.path.dirname(os.path.realpath(request.whitening)) != os.path.realpath(request.foil_cache):
        raise SealError("selected whitening artifact must resolve inside the explicit foil overlay")
    expected_whitening = (
        "text_whiten_optTrain_localOnly.npz"
        if request.stage == "stage1"
        else "text_whiten_trainOnly_localOnly.npz"
    )
    if os.path.basename(request.whitening) != expected_whitening:
        raise SealError(
            f"{request.stage} requires {expected_whitening}, got "
            f"{os.path.basename(request.whitening)!r}"
        )

    if whitening_meta.get("leakage_free_fit") is not True:
        raise SealError("whitening metadata must record literal leakage_free_fit=true")
    rows_used = whitening_meta.get("rows_used")
    if isinstance(rows_used, bool) or not isinstance(rows_used, int) or rows_used <= 0:
        raise SealError(f"whitening metadata rows_used is invalid: {rows_used!r}")
    meta_rows = whitening_meta.get("row_index_npy")
    if not isinstance(meta_rows, str):
        raise SealError("whitening metadata has no string row_index_npy")
    _assert_same_real_path(meta_rows, request.split_rows, "whitening row-index")
    meta_cache = whitening_meta.get("cache_dir")
    if not isinstance(meta_cache, str):
        raise SealError("whitening metadata has no string cache_dir")
    _assert_same_real_path(meta_cache, request.feature_cache, "whitening cache")

    qwen_name = feature_meta.get("qwen_cache")
    if not isinstance(qwen_name, str) or os.path.basename(qwen_name) != os.path.basename(request.qwen):
        raise SealError(
            f"feature-cache meta names qwen_cache={qwen_name!r}, not the "
            f"explicit Qwen file {os.path.basename(request.qwen)!r}"
        )
    if _effective_sha(_record_by_name(feature, "meta.json")) != _effective_sha(
        _record_by_name(overlay, "meta.json")
    ):
        raise SealError("foil overlay meta.json bytes differ from feature-cache meta.json")
    if dict(feature_meta) != dict(overlay_meta):
        raise SealError("foil overlay parsed metadata differs from feature-cache metadata")

    # Raw formatting of a JSON array is not loader semantics.  Exact string
    # elements/order (plus unique/non-empty list validation) was checked in the
    # preflight.  Each of these files still retains its own raw-byte seal.
    feature_ids_sha = _effective_sha(_record_by_name(feature, "image_ids.json"))

    expected_dataset = _DATASETS[request.dataset][1]
    if semantic_manifest.get("dataset") != expected_dataset:
        raise SealError(
            f"foil manifest dataset={semantic_manifest.get('dataset')!r}, "
            f"expected {expected_dataset!r}"
        )
    base_cache = semantic_manifest.get("base_cache")
    overlay_path = semantic_manifest.get("overlay")
    if not isinstance(base_cache, str) or not isinstance(overlay_path, str):
        raise SealError("foil manifest does not name base_cache and overlay")
    _assert_same_real_path(base_cache, request.feature_cache, "foil-manifest base cache")
    _assert_same_real_path(overlay_path, request.foil_cache, "foil-manifest overlay")
    if semantic_manifest.get("image_ids_sha256") != feature_ids_sha:
        raise SealError("foil manifest image_ids_sha256 does not match sealed image_ids")
    qwen_sha = _effective_sha(qwen_record)
    if semantic_manifest.get("qwen_jsonl_sha256") != qwen_sha:
        raise SealError("foil manifest qwen_jsonl_sha256 does not match explicit Qwen bytes")

    whitening_table = semantic_manifest.get("whitening")
    if not isinstance(whitening_table, Mapping):
        raise SealError("foil manifest has no whitening mapping")
    whitening_entry = whitening_table.get(os.path.basename(request.whitening))
    if not isinstance(whitening_entry, Mapping):
        raise SealError("foil manifest has no entry for the selected whitening artifact")
    if whitening_entry.get("sha256") != _effective_sha(whitening_record):
        raise SealError("foil manifest whitening SHA does not match selected whitening bytes")
    if whitening_entry.get("row_index_sha256") != _effective_sha(split_record):
        raise SealError("foil manifest row-index SHA does not match split-row bytes")
    if whitening_entry.get("rows_used") != whitening_meta.get("rows_used"):
        raise SealError(
            "foil manifest whitening rows_used differs from selected metadata"
        )
    manifest_rows = whitening_entry.get("row_index")
    if not isinstance(manifest_rows, str):
        raise SealError("foil manifest whitening entry has no row_index")
    _assert_same_real_path(manifest_rows, request.split_rows, "foil-manifest row index")

    for label, meta in (("pooled foil", foil_meta), ("token foil", foil_token_meta)):
        if meta.get("cache_image_ids_sha256") != feature_ids_sha:
            raise SealError(f"{label} metadata image-id digest is not bound to feature rows")
        cache_dir = meta.get("cache_dir")
        if not isinstance(cache_dir, str):
            raise SealError(f"{label} metadata has no cache_dir")
        _assert_same_real_path(cache_dir, request.feature_cache, f"{label} cache")
        if meta.get("foil_jsonl_sha256") != semantic_manifest.get("foil_jsonl_sha256"):
            raise SealError(f"{label} metadata disagrees on the foil JSONL digest")


def _flatten_file_records(value: Any) -> Iterator[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        if value.get("file_type") in {"regular", "symlink"} and "path" in value:
            yield value
            return
        for child in value.values():
            yield from _flatten_file_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from _flatten_file_records(child)


def _assert_record_stat_current(record: Mapping[str, Any]) -> None:
    path = str(record["path"])
    try:
        current = os.lstat(path)
    except FileNotFoundError as error:
        raise SealError(f"input disappeared before seal completion: {path}") from error
    if record["file_type"] == "regular":
        if _stat_json(current) != record["stat"]:
            raise SealError(f"input stat drifted before seal completion: {path}")
        return
    link_bytes = os.readlink(os.fsencode(path))
    try:
        resolved = str(Path(path).resolve(strict=True))
        target = os.stat(resolved, follow_symlinks=False)
    except (OSError, RuntimeError) as error:
        raise SealError(f"input symlink broke before seal completion: {path}: {error}") from error
    if (
        _stat_json(current) != record["lstat"]
        or hashlib.sha256(link_bytes).hexdigest() != record["link_text_sha256"]
        or os.fsdecode(link_bytes) != record["link_text"]
        or resolved != record["resolved_target"]["path"]
        or _stat_json(target) != record["resolved_target"]["stat"]
    ):
        raise SealError(f"input symlink/target stat drifted before seal completion: {path}")


def build_seal(request: SealRequest) -> dict[str, Any]:
    """Hash and validate one explicit Phase-3 input set in memory."""
    # Fail-fast pass: directory inventories and all loader-visible row-id/meta
    # JSON are validated before `_HashMemo` is even constructed.  In the real
    # cache this prevents a semantic JSON mismatch being discovered only after
    # the 400 GB tensor set has been read.
    feature_plan = _cache_inventory_preflight(
        request.feature_cache, role="feature", require_foils=False
    )
    overlay_plan = _cache_inventory_preflight(
        request.foil_cache, role="foil overlay", require_foils=True
    )
    image_id_alignment = _image_id_alignment_preflight(
        request, feature_plan=feature_plan, overlay_plan=overlay_plan
    )
    hf_plan = _hf_provenance_preflight(request, feature_plan.meta)
    feature_image_ids, _ = _strict_image_ids(
        os.path.join(request.feature_cache, "image_ids.json"),
        description="feature_image_ids_for_split",
    )
    split_identity = _split_identity_preflight(
        request,
        feature_image_ids=feature_image_ids,
        feature_image_ids_semantic_sha256=image_id_alignment["sha256"],
    )
    decode_failure_audit = _decode_failure_audit_preflight(
        request,
        feature_meta=feature_plan.meta,
        feature_image_ids=feature_image_ids,
    )
    whitening_validation = _whitening_preflight(
        request,
        feature_meta=feature_plan.meta,
        feature_image_ids=feature_image_ids,
        split_identity=split_identity,
    )

    memo = _HashMemo()
    feature, feature_meta = _seal_cache_inventory(feature_plan, memo)
    overlay, overlay_meta = _seal_cache_inventory(overlay_plan, memo)
    hf_provenance = _seal_hf_provenance(hf_plan, memo)

    sealed_id_records = {
        "feature_image_ids": _record_by_name(feature, "image_ids.json"),
        "overlay_image_ids": _record_by_name(overlay, "image_ids.json"),
        "pooled_foil_image_ids": _record_by_name(
            overlay, "text_foil_image_ids.json"
        ),
        "token_foil_image_ids": _record_by_name(
            overlay, "text_foil_token_image_ids.json"
        ),
    }
    for evidence in image_id_alignment["logical_files"]:
        role = str(evidence["role"])
        if _effective_sha(sealed_id_records[role]) != evidence["raw_sha256"]:
            raise SealError(f"{role} bytes changed after semantic preflight")
    dataset = _dataset_inventory(request, memo)

    val_split_source_record = _seal_file(REPO / "val_split.py", memo)
    if (
        _effective_sha(val_split_source_record)
        != split_identity["protocol_source"]["sha256"]
    ):
        raise SealError("val_split.py changed after canonical split preflight")
    dataset_records_by_path = {
        str(record["path"]): record for record in dataset["files"]
    }
    for source in split_identity["label_sources"]:
        record = dataset_records_by_path.get(str(source["path"]))
        if record is None or _effective_sha(record) != source["sha256"]:
            raise SealError(
                f"designated train label source changed after split preflight: "
                f"{source['path']}"
            )
    for source in decode_failure_audit["membership_sources"]:
        record = dataset_records_by_path.get(str(source["path"]))
        if record is None or _effective_sha(record) != source["sha256"]:
            raise SealError(
                "official decode-failure membership source changed after "
                f"preflight: {source['path']}"
            )

    whitening_record = _seal_file(request.whitening, memo)
    if (
        _effective_sha(whitening_record)
        != whitening_validation["artifact_raw_sha256"]
    ):
        raise SealError("whitening NPZ changed after validation preflight")
    whitening_meta_path = request.whitening + ".meta.json"
    whitening_meta_record = _seal_file(whitening_meta_path, memo)
    if (
        _effective_sha(whitening_meta_record)
        != whitening_validation["metadata_raw_sha256"]
    ):
        raise SealError("whitening metadata changed after validation preflight")
    whitening_meta = _read_json_bound(
        whitening_meta_path, whitening_meta_record, memo
    )
    if not isinstance(whitening_meta, Mapping):
        raise SealError("whitening metadata is not a JSON object")

    split_record = _seal_file(request.split_rows, memo)
    if _effective_sha(split_record) != split_identity["split_rows"]["raw_sha256"]:
        raise SealError("split-row bytes changed after canonical split preflight")
    has_text_record = _record_by_name(feature, "has_text.bool.npy")
    if (
        _effective_sha(has_text_record)
        != whitening_validation["has_text_raw_sha256"]
    ):
        raise SealError("has_text bytes changed after whitening validation")
    # The overlay deliberately exposes the same row index as a logical symlink.
    # Preserve that link evidence too, while memoizing its target hash.
    overlay_split_path = os.path.join(
        request.foil_cache, os.path.basename(request.split_rows)
    )
    overlay_split_record = _seal_file(overlay_split_path, memo)
    if _effective_sha(overlay_split_record) != _effective_sha(split_record):
        raise SealError("foil-overlay split-row link does not resolve to the explicit split rows")

    qwen_record = _seal_file(request.qwen, memo)

    semantic_path = os.path.join(
        request.foil_cache, "semantic_detail_cache_manifest.json"
    )
    foil_meta_path = os.path.join(request.foil_cache, "text_foil_meta.json")
    foil_token_meta_path = os.path.join(
        request.foil_cache, "text_foil_token_meta.json"
    )
    semantic_manifest = _read_json_bound(
        semantic_path,
        _record_by_name(overlay, "semantic_detail_cache_manifest.json"),
        memo,
    )
    foil_meta = _read_json_bound(
        foil_meta_path, _record_by_name(overlay, "text_foil_meta.json"), memo
    )
    foil_token_meta = _read_json_bound(
        foil_token_meta_path,
        _record_by_name(overlay, "text_foil_token_meta.json"),
        memo,
    )
    for label, value in (
        ("semantic-detail manifest", semantic_manifest),
        ("pooled foil metadata", foil_meta),
        ("token foil metadata", foil_token_meta),
    ):
        if not isinstance(value, Mapping):
            raise SealError(f"{label} is not a JSON object")

    _assert_manifest_contract(
        request,
        feature=feature,
        feature_meta=feature_meta,
        overlay=overlay,
        overlay_meta=overlay_meta,
        whitening_record=whitening_record,
        whitening_meta=whitening_meta,
        split_record=split_record,
        qwen_record=qwen_record,
        semantic_manifest=semantic_manifest,
        foil_meta=foil_meta,
        foil_token_meta=foil_token_meta,
    )

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "request": request.as_json(),
        "contract": {
            "runtime_loader": (
                "dataloaders._SigLIP2FeatureCache receives request.feature_cache; "
                "request.foil_cache is not passed as the runtime cache"
            ),
            "protocol": "canonical Phase-3 stage1/refit",
            "symlink_policy": "link text + final resolved target + target bytes",
            "verification_policy": "content and recorded stat identity must match",
            "consumption_roles": {
                "runtime_consumed": [
                    "inputs.dataset_rows.files[consumption_role=runtime_consumed]",
                    "inputs.feature_cache",
                    "inputs.whitening",
                    "inputs.split_rows.feature_cache_entry",
                ],
                "derivation_provenance": [
                    "inputs.dataset_rows.files[consumption_role=derivation_provenance]",
                    "inputs.foil_overlay",
                    "inputs.image_id_alignment",
                    "inputs.qwen_jsonl",
                    "inputs.split_rows.foil_overlay_entry",
                    "inputs.split_identity",
                    "inputs.decode_failure_audit",
                    "inputs.hf_provenance",
                    "inputs.protocol_sources",
                ],
            },
        },
        "inputs": {
            "dataset_rows": dataset,
            "feature_cache": feature,
            "foil_overlay": overlay,
            "image_id_alignment": image_id_alignment,
            "split_identity": split_identity,
            "decode_failure_audit": decode_failure_audit,
            "hf_provenance": hf_provenance,
            "protocol_sources": {
                "val_split": val_split_source_record,
            },
            "whitening": {
                "artifact": whitening_record,
                "metadata": whitening_meta_record,
                "validation": whitening_validation,
            },
            "split_rows": {
                "feature_cache_entry": split_record,
                "foil_overlay_entry": overlay_split_record,
            },
            "qwen_jsonl": qwen_record,
        },
        "hash_summary": {
            "algorithm": HASH_ALGORITHM,
            "logical_file_records": 0,  # filled after the structure is final
            "unique_resolved_content_objects": memo.unique_objects,
            "unique_resolved_content_bytes": memo.hashed_bytes,
        },
    }
    records = list(_flatten_file_records(payload["inputs"]))
    paths = [str(record["path"]) for record in records]
    if len(paths) != len(set(paths)):
        duplicates = sorted({path for path in paths if paths.count(path) > 1})
        raise SealError(f"internal seal inventory contains duplicate logical paths: {duplicates}")
    payload["hash_summary"]["logical_file_records"] = len(records)

    # A 400 GB seal can take long enough for an early file to be replaced while
    # a late one is hashing.  Re-stat every logical entry before publishing.
    for record in records:
        _assert_record_stat_current(record)
    return payload


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def with_aggregate(payload: Mapping[str, Any]) -> dict[str, Any]:
    if "aggregate_digest" in payload:
        raise SealError("aggregate_digest must not be present before aggregation")
    result = dict(payload)
    result["aggregate_digest"] = {
        "algorithm": HASH_ALGORITHM,
        "canonicalization": "UTF-8 JSON, sort_keys=true, separators=(',', ':'), excluding aggregate_digest",
        "sha256": hashlib.sha256(_canonical_bytes(payload)).hexdigest(),
    }
    return result


def _publish_exclusive(path: os.PathLike[str] | str, payload: Mapping[str, Any]) -> None:
    output = Path(os.path.abspath(os.fspath(path)))
    if not output.parent.is_dir():
        raise SealError(f"seal output parent does not exist: {output.parent}")
    data = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    temporary = output.parent / (
        f".{output.name}.tmp.{os.getpid()}.{secrets.token_hex(8)}"
    )
    fd = None
    try:
        fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise SealError(f"short write while creating seal {temporary}")
            view = view[written:]
        os.fsync(fd)
        os.fchmod(fd, 0o444)
        os.close(fd)
        fd = None
        try:
            os.link(temporary, output)
        except FileExistsError as error:
            raise SealError(f"refusing to overwrite existing seal: {output}") from error
        directory_fd = os.open(output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if fd is not None:
            os.close(fd)
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def create_seal(request: SealRequest, output: os.PathLike[str] | str) -> dict[str, Any]:
    sealed = with_aggregate(build_seal(request))
    _publish_exclusive(output, sealed)
    return sealed


def _first_difference(left: Any, right: Any, prefix: str = "$seal") -> str:
    if type(left) is not type(right):
        return f"{prefix}: type {type(left).__name__} -> {type(right).__name__}"
    if isinstance(left, Mapping):
        left_keys, right_keys = set(left), set(right)
        if left_keys != right_keys:
            return (
                f"{prefix}: keys removed={sorted(left_keys-right_keys)}, "
                f"added={sorted(right_keys-left_keys)}"
            )
        for key in sorted(left):
            difference = _first_difference(left[key], right[key], f"{prefix}.{key}")
            if difference:
                return difference
        return ""
    if isinstance(left, list):
        if len(left) != len(right):
            return f"{prefix}: length {len(left)} -> {len(right)}"
        for index, (old, new) in enumerate(zip(left, right)):
            difference = _first_difference(old, new, f"{prefix}[{index}]")
            if difference:
                return difference
        return ""
    if left != right:
        return f"{prefix}: {left!r} -> {right!r}"
    return ""


def _load_seal_payload(
    path: os.PathLike[str] | str,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    seal_path = os.path.abspath(os.fspath(path))
    try:
        with open(seal_path, encoding="utf-8") as handle:
            sealed = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise SealError(f"cannot read existing seal {seal_path}: {error}") from error
    if not isinstance(sealed, MutableMapping):
        raise SealError("existing seal is not a JSON object")
    if sealed.get("schema") != SCHEMA or sealed.get("schema_version") != SCHEMA_VERSION:
        raise SealError(
            f"unsupported seal schema/version: "
            f"{sealed.get('schema')!r}/{sealed.get('schema_version')!r}"
        )
    aggregate = sealed.get("aggregate_digest")
    if not isinstance(aggregate, Mapping) or aggregate.get("algorithm") != HASH_ALGORITHM:
        raise SealError("existing seal has no schema-v1 aggregate digest")
    payload = dict(sealed)
    payload.pop("aggregate_digest", None)
    expected_digest = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    if aggregate.get("sha256") != expected_digest:
        raise SealError("existing seal aggregate digest does not match its JSON payload")
    return seal_path, dict(sealed), payload


def _assert_directory_identity_current(record: Mapping[str, Any]) -> None:
    current = _directory_identity(str(record["path"]))
    if current != record:
        raise SealError(f"input directory identity drifted: {record['path']}")


def _assert_cache_names_current(
    group: Mapping[str, Any], *, include_foil_provenance: bool
) -> None:
    expected = {str(entry["logical_name"]) for entry in group["files"]}
    present = _entries(str(group["root"]["path"]))
    relevant = set(_CACHE_CORE) | set(_FACTUAL_TOKEN_PAIR) | set(_FOIL_RUNTIME)
    if include_foil_provenance:
        relevant.update(_FOIL_PROVENANCE)
    actual = {
        name for name in present
        if name in relevant or _AUGMENTED_RE.fullmatch(name)
    }
    if actual != expected:
        raise SealError(
            f"cache consumed-file inventory drifted at {group['root']['path']}: "
            f"removed={sorted(expected-actual)}, added={sorted(actual-expected)}"
        )


def verify_seal_stats(
    path: os.PathLike[str] | str,
    *,
    expected_aggregate_sha256: str | None = None,
) -> dict[str, Any]:
    """Fast fail-closed check after one full verify in the same campaign.

    This checks the seal JSON aggregate, every logical file's lstat/link text,
    final resolved target and target stat, explicit directory identity, and the
    current loader-consumed filename inventory.  It intentionally does *not*
    reread content bytes.  A campaign should call :func:`verify_seal` once at
    admission, retain its aggregate digest, and pass that value here before and
    after cells.  A new process must perform another full verify first.
    """
    _, sealed, payload = _load_seal_payload(path)
    aggregate_sha = str(sealed["aggregate_digest"]["sha256"])
    if (
        expected_aggregate_sha256 is not None
        and aggregate_sha != expected_aggregate_sha256
    ):
        raise SealError(
            "existing seal aggregate differs from the campaign-admission digest"
        )

    inputs = payload.get("inputs")
    if not isinstance(inputs, Mapping):
        raise SealError("existing seal has no inputs object")
    try:
        dataset = inputs["dataset_rows"]
        feature = inputs["feature_cache"]
        overlay = inputs["foil_overlay"]
    except KeyError as error:
        raise SealError(f"existing seal is missing input group {error.args[0]!r}") from error
    for group in (dataset, feature, overlay):
        if not isinstance(group, Mapping) or not isinstance(group.get("root"), Mapping):
            raise SealError("existing seal contains a malformed rooted input group")
        _assert_directory_identity_current(group["root"])
    _assert_cache_names_current(feature, include_foil_provenance=False)
    _assert_cache_names_current(overlay, include_foil_provenance=True)

    for record in _flatten_file_records(inputs):
        _assert_record_stat_current(record)
    return sealed


def verify_seal(path: os.PathLike[str] | str) -> dict[str, Any]:
    """Fully rehash and verify every sealed byte (campaign admission only)."""
    _, sealed, payload = _load_seal_payload(path)

    request_value = payload.get("request")
    if not isinstance(request_value, Mapping):
        raise SealError("existing seal has no request object")
    request = SealRequest.from_json(request_value)
    current = build_seal(request)
    if current != payload:
        difference = _first_difference(payload, current)
        raise SealError(f"Phase-3 input bytes/stat/inventory drifted: {difference}")
    return sealed


AUTHORITY_SCHEMA = "groundeddna.phase3-input-authority"
AUTHORITY_SCHEMA_VERSION = 1


def _record_for_role(files: Sequence[Mapping[str, Any]], role: str) -> Mapping[str, Any]:
    matches = [record for record in files if role in record.get("logical_roles", ())]
    if len(matches) != 1:
        raise SealError(
            f"sealed Hugging Face inventory has {len(matches)} records for {role!r}"
        )
    return matches[0]


def _plain_file_sha256(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def authority_from_verified_seal(
    path: os.PathLike[str] | str,
    sealed: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the compact immutable authority handed to a Phase-3 trainer.

    ``sealed`` must be the value returned by :func:`verify_seal` (or by a
    same-process stats verification after that full admission check).  The
    authority contains no mutable reconstruction: every value below is copied
    from content committed by the seal JSON and the seal file itself is
    independently hashed.
    """
    seal_path = os.path.abspath(os.fspath(path))
    if sealed.get("schema") != SCHEMA or sealed.get("schema_version") != SCHEMA_VERSION:
        raise SealError("cannot derive runtime authority from a non-current seal")
    aggregate = (sealed.get("aggregate_digest") or {}).get("sha256")
    if not isinstance(aggregate, str) or _SHA256_RE.fullmatch(aggregate) is None:
        raise SealError("verified seal has no valid aggregate SHA-256")
    request = sealed.get("request")
    inputs = sealed.get("inputs")
    if not isinstance(request, Mapping) or not isinstance(inputs, Mapping):
        raise SealError("verified seal lacks request/inputs")
    split = inputs.get("split_identity")
    hf = inputs.get("hf_provenance")
    if not isinstance(split, Mapping) or not isinstance(hf, Mapping):
        raise SealError("verified seal lacks split/Hugging Face authority")
    split_digest = split.get("identity_sha256")
    if not isinstance(split_digest, str) or _SHA256_RE.fullmatch(split_digest) is None:
        raise SealError("verified seal split identity has no valid digest")
    files = hf.get("files")
    if not isinstance(files, list):
        raise SealError("verified seal Hugging Face inventory has no file list")
    weight_name = (hf.get("snapshot_contract") or {}).get("selected_model_weight")
    if weight_name not in _HF_WEIGHT_FILES:
        raise SealError("verified seal selected an unsupported model weight")
    weight_record = _record_for_role(files, "snapshot_model_weight")
    config_record = _record_for_role(files, "snapshot_config:config.json")
    tokenizers = {}
    for name in _HF_TOKENIZER_FILES:
        record = _record_for_role(files, f"snapshot_tokenizer:{name}")
        tokenizers[name] = _effective_sha(record)
    tokenizer_digest = hashlib.sha256(
        json.dumps(tokenizers, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    hf_runtime: dict[str, Any] = {
        "checkpoint": hf.get("checkpoint"),
        "revision": hf.get("model_revision"),
        "snapshot_dir": os.path.abspath(str(hf.get("snapshot_dir", ""))),
        "weight_file": str(weight_name),
        "weight_sha256": _effective_sha(weight_record),
        "config_file": _HF_CONFIG_FILE,
        "config_sha256": _effective_sha(config_record),
        "tokenizer_files_sha256": tokenizers,
        "tokenizer_set_sha256": tokenizer_digest,
        "local_files_only": True,
    }
    hf_runtime["identity_sha256"] = hashlib.sha256(
        _canonical_bytes(hf_runtime)
    ).hexdigest()
    # Keep the logical pathname and its lstat/link target evidence in the
    # compact trainer authority.  Comparing only realpaths here would allow a
    # caller to substitute an unsealed alias for a sealed pathname.  The full
    # seal already commits these records; this projection lets the trainer
    # re-check the exact objects it is about to consume without reopening and
    # parsing the complete (potentially very large) inventory.
    try:
        runtime_path_identities = {
            "dataset_rows_root": dict(inputs["dataset_rows"]["root"]),
            "feature_cache_root": dict(inputs["feature_cache"]["root"]),
            "qwen": dict(inputs["qwen_jsonl"]),
            "whitening": dict(inputs["whitening"]["artifact"]),
        }
    except (KeyError, TypeError) as error:
        raise SealError(
            "verified seal lacks runtime path identity evidence"
        ) from error
    authority: dict[str, Any] = {
        "schema": AUTHORITY_SCHEMA,
        "schema_version": AUTHORITY_SCHEMA_VERSION,
        "seal_path": seal_path,
        "seal_file_sha256": _plain_file_sha256(seal_path),
        "aggregate_sha256": aggregate,
        "dataset": request.get("dataset"),
        "stage": request.get("stage"),
        "request": dict(request),
        "split_identity_sha256": split_digest,
        "split_identity": {
            "counts": dict(split.get("counts") or {}),
            "digests": dict(split.get("digests") or {}),
            "split_rows_raw_sha256": (split.get("split_rows") or {}).get("raw_sha256"),
        },
        "runtime_path_identities": runtime_path_identities,
        "hf_runtime": hf_runtime,
    }
    authority["authority_sha256"] = hashlib.sha256(
        _canonical_bytes(authority)
    ).hexdigest()
    return authority


def verify_seal_authority(
    path: os.PathLike[str] | str,
    *,
    expected: Mapping[str, Any] | None = None,
    full: bool = True,
) -> dict[str, Any]:
    """Verify a seal and return/recheck its compact runtime authority."""
    if full:
        sealed = verify_seal(path)
    else:
        if expected is None:
            raise SealError("stats-only authority verification requires admission authority")
        sealed = verify_seal_stats(
            path, expected_aggregate_sha256=str(expected.get("aggregate_sha256", ""))
        )
    authority = authority_from_verified_seal(path, sealed)
    if expected is not None and dict(authority) != dict(expected):
        difference = _first_difference(dict(expected), authority, "$authority")
        raise SealError(f"Phase-3 input authority changed: {difference}")
    return authority


def assert_runtime_paths(authority: Mapping[str, Any], *, dataset: str,
                         stage: str, dataset_root: str, feature_cache: str,
                         qwen: str, whitening: str) -> None:
    """Fail unless the trainer's actual path arguments are the sealed request."""
    if authority.get("schema") != AUTHORITY_SCHEMA \
            or authority.get("schema_version") != AUTHORITY_SCHEMA_VERSION:
        raise SealError("trainer received a non-current input authority")
    request = authority.get("request")
    if not isinstance(request, Mapping):
        raise SealError("trainer input authority has no request")
    scalar = {"dataset": dataset, "stage": stage}
    wrong = {name: (request.get(name), value) for name, value in scalar.items()
             if request.get(name) != value}
    for name, value in (
        ("dataset_root", dataset_root), ("feature_cache", feature_cache),
        ("qwen", qwen), ("whitening", whitening),
    ):
        expected_path = request.get(name)
        # ``SealRequest.normalized`` intentionally uses abspath, not realpath:
        # the logical spelling (including a symlink component) is authority.
        # An alias that happens to resolve to the same inode was not sealed.
        if not isinstance(expected_path, str) or \
                os.path.abspath(os.fspath(value)) != expected_path:
            wrong[name] = (expected_path, value)
    if wrong:
        raise SealError(f"trainer runtime inputs differ from sealed request: {wrong}")

    identities = authority.get("runtime_path_identities")
    if not isinstance(identities, Mapping) or set(identities) != {
        "dataset_rows_root", "feature_cache_root", "qwen", "whitening"
    }:
        raise SealError("trainer input authority has no exact runtime path identities")
    canonical_dataset_dir = os.path.join(
        str(request["dataset_root"]), _DATASETS[str(request["dataset"])][0]
    )
    expected_record_paths = {
        "dataset_rows_root": canonical_dataset_dir,
        "feature_cache_root": str(request["feature_cache"]),
        "qwen": str(request["qwen"]),
        "whitening": str(request["whitening"]),
    }
    for name, expected_path in expected_record_paths.items():
        record = identities.get(name)
        if not isinstance(record, Mapping) or record.get("path") != expected_path:
            raise SealError(
                f"trainer runtime path identity {name!r} is not bound to "
                f"the sealed logical path {expected_path!r}"
            )
        if name.endswith("_root"):
            _assert_directory_identity_current(record)
        else:
            _assert_record_stat_current(record)


def assert_runtime_split_rows(
    authority: Mapping[str, Any],
    *,
    dataset_to_cache_rows: Sequence[int] | np.ndarray,
    optimization_indices: Sequence[int] | np.ndarray,
    validation_indices: Sequence[int] | np.ndarray,
) -> None:
    """Compare the loader's actual cache-row mapping to the sealed split."""
    split = authority.get("split_identity")
    if not isinstance(split, Mapping):
        raise SealError("trainer input authority has no split identity")
    digests = split.get("digests")
    counts = split.get("counts")
    if not isinstance(digests, Mapping) or not isinstance(counts, Mapping):
        raise SealError("trainer input authority has malformed split identity")
    rows = np.asarray(dataset_to_cache_rows, dtype=np.int64).reshape(-1)
    opt_indices = np.asarray(optimization_indices, dtype=np.int64).reshape(-1)
    val_indices = np.asarray(validation_indices, dtype=np.int64).reshape(-1)
    if len(rows) != int(counts.get("designated_train", -1)):
        raise SealError("trainer designated-train row count differs from seal")
    if np.any(opt_indices < 0) or np.any(opt_indices >= len(rows)) \
            or np.any(val_indices < 0) or np.any(val_indices >= len(rows)):
        raise SealError("trainer split indices are outside the dataset row mapping")
    actual_source = np.sort(rows)
    actual_opt = np.sort(rows[opt_indices])
    actual_val = np.sort(rows[val_indices])
    if len(np.unique(actual_source)) != len(actual_source) \
            or len(np.unique(actual_opt)) != len(actual_opt) \
            or len(np.unique(actual_val)) != len(actual_val):
        raise SealError("trainer cache-row mapping contains duplicates")
    actual = {
        # This digest is deliberately order-sensitive: dataset example i must
        # still address the cache row sealed for example i.  Sorting it, as the
        # membership digests below require, admits a feature/label permutation.
        "dataset_to_cache_rows_sha256": _row_vector_sha256(rows),
        "source_cache_rows_sha256": _row_vector_sha256(actual_source),
        "optimization_cache_rows_sha256": _row_vector_sha256(actual_opt),
        "validation_cache_rows_sha256": _row_vector_sha256(actual_val),
    }
    wrong = {name: (digests.get(name), value) for name, value in actual.items()
             if digests.get(name) != value}
    expected_counts = {
        "optimization_train": len(actual_opt),
        "heldout_train_validation": len(actual_val),
    }
    wrong.update({name: (counts.get(name), value)
                  for name, value in expected_counts.items()
                  if counts.get(name) != value})
    if np.intersect1d(actual_opt, actual_val, assume_unique=True).size:
        wrong["optimization_validation_overlap"] = (0, "nonzero")
    if wrong:
        raise SealError(f"trainer split/cache rows differ from sealed identity: {wrong}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    seal = subparsers.add_parser("seal", help="create a new exclusive input seal")
    seal.add_argument("--dataset", required=True, choices=sorted(_DATASETS))
    seal.add_argument("--stage", required=True, choices=("stage1", "refit"))
    seal.add_argument("--dataset-root", required=True)
    seal.add_argument("--feature-cache", required=True)
    seal.add_argument("--foil-cache", required=True)
    seal.add_argument("--whitening", required=True)
    seal.add_argument("--qwen", required=True)
    seal.add_argument("--split-rows", required=True)
    seal.add_argument("--output", required=True)

    verify = subparsers.add_parser("verify", help="fail unless an existing seal still matches")
    verify.add_argument("--seal", required=True, dest="seal_path")
    verify_stats = subparsers.add_parser(
        "verify-stats",
        help="after a full in-process verify, recheck inventory and stats only",
    )
    verify_stats.add_argument("--seal", required=True, dest="seal_path")
    verify_stats.add_argument("--expected-aggregate-sha256")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "seal":
            request = SealRequest.normalized(
                dataset=args.dataset,
                stage=args.stage,
                dataset_root=args.dataset_root,
                feature_cache=args.feature_cache,
                foil_cache=args.foil_cache,
                whitening=args.whitening,
                qwen=args.qwen,
                split_rows=args.split_rows,
            )
            sealed = create_seal(request, args.output)
            print(f"sealed {args.output} {sealed['aggregate_digest']['sha256']}")
        elif args.command == "verify":
            sealed = verify_seal(args.seal_path)
            print(f"verified {args.seal_path} {sealed['aggregate_digest']['sha256']}")
        else:
            sealed = verify_seal_stats(
                args.seal_path,
                expected_aggregate_sha256=args.expected_aggregate_sha256,
            )
            print(
                f"verified-stats {args.seal_path} "
                f"{sealed['aggregate_digest']['sha256']}"
            )
    except SealError as error:
        print(f"[phase3-input-seal] REFUSED: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
