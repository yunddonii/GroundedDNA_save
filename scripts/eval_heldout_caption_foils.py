#!/usr/bin/env python3
"""Frozen stage-1 held-out minimal-pair evaluation for caption foils.

This evaluator is deliberately separate from training.  It loads one or more
stage-1 ``model_state_dict_best.pth`` checkpoints, evaluates the exact
``train_all_rows - opt_train_rows`` cache rows, and never enables train mode,
gradients, an optimizer, EMA updates, or stochastic codon sampling.

The evaluated rows are *optimization-held-out* but were used to select E*.
Consequently, this is a selection-used diagnostic, not an independent test.
The generated lexical foils also share a generator and may share exact
surface contexts and generator edit rules with optimization-train foils.  Both
overlap notions are measured separately; an unseen surface context is never
silently called a template-disjoint test.

For each valid local slot, the text comparison is a true one-slot replacement:
the factual C_global and every non-target local caption remain factual, while
only the target local token pool is replaced by its foil.  Both factual and
foil token caches use the model's existing
``_pool_local_foil_tokens_like_factual`` path, followed by
``_adapt_pooled_text_for_loss`` and
``_encode_text_tokens_to_dna(allow_mm_ema=False,
deterministic_codon=True)``.  The image is encoded once through the deployed
image-only eval path.  There is therefore no "visual codeword flip" metric;
the meaningful visual diagnostics are preference/distance and fixed visual
codeword match advantage against factual versus foil text DNA.

Example (preflight first):

    python scripts/eval_heldout_caption_foils.py \
      --run A=result/<stage1-A> \
      --run AB=result/<stage1-AB> \
      --run ABC=result/<stage1-ABC> \
      --cache-dir cache/<foil-overlay> \
      --foil-jsonl artifacts/<dataset>.foils.jsonl \
      --out artifacts/<dataset>.heldout_foil.json \
      --preflight-only

Remove ``--preflight-only`` and add ``--device cuda:0`` to evaluate.  The
summary is written to ``--out`` and pair-level rows to
``<out-stem>.pairs.jsonl``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).absolute().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCHEMA_VERSION = "groundeddna-heldout-caption-foil-eval-v1"
EXPECTED_FOIL_SCHEMA = "groundeddna-local-caption-foil-v1"
N_SLOTS = 6
LOCAL_SLOTS = tuple(range(1, N_SLOTS))
FIXED_MARGIN = 0.02
DEFAULT_N_BOOTSTRAP = 2000
DEFAULT_BOOTSTRAP_SEED = 20260724

FACTUAL_CACHE_FILES = (
    "image_ids.json",
    "visual_tokens.f16.npy",
    "visual_global.f16.npy",
    "text_part.f16.npy",
    "has_text.bool.npy",
    "text_tokens.f16.npy",
    "text_token_mask.bool.npy",
    "opt_train_rows.npy",
    "train_all_rows.npy",
)

PAIR_METRICS = (
    "visual_factual_cosine",
    "visual_foil_cosine",
    "visual_cosine_gap",
    "factual_preference",
    "margin_violation",
    "margin_satisfied",
    "text_codeword_flip",
    "text_dna_base_hamming_count",
    "text_dna_base_hamming_fraction",
    "visual_text_base_distance_factual",
    "visual_text_base_distance_foil",
    "visual_text_base_distance_advantage",
    "visual_dna_prefers_factual_by_base_distance",
    "visual_codeword_matches_factual",
    "visual_codeword_matches_foil",
    "visual_codeword_match_advantage",
    "visual_codeword_exclusive_factual_match",
    "visual_codeword_exclusive_foil_match",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_strings(items: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for item in items:
        payload = str(item).encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _sha256_int_rows(rows: np.ndarray) -> str:
    canonical = np.asarray(rows, dtype="<i8").reshape(-1)
    digest = hashlib.sha256()
    digest.update(str(tuple(canonical.shape)).encode("ascii"))
    digest.update(memoryview(np.ascontiguousarray(canonical)).cast("B"))
    return digest.hexdigest()


def _stable_seed(seed: int, scope: str) -> int:
    payload = f"{int(seed)}\0{scope}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _as_builtin(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return [_as_builtin(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _as_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_builtin(item) for item in value]
    return value


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=".heldout-foil-", suffix=".json", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                _as_builtin(payload),
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def _atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=".heldout-foil-pairs-", suffix=".jsonl", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        _as_builtin(row),
                        ensure_ascii=False,
                        sort_keys=True,
                        allow_nan=False,
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return Path(os.path.abspath(path))


@dataclass(frozen=True)
class RunSpec:
    label: str
    run_dir: Path
    config_path: Path
    checkpoint_path: Path
    config: Dict[str, Any]


@dataclass(frozen=True)
class FoilEntry:
    image_id: str
    slot_keys: Tuple[str, ...]
    texts: Tuple[str, ...]
    valid: Tuple[bool, ...]
    edits: Tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class PairSpec:
    pair_id: str
    cache_row: int
    image_id: str
    slot: int
    slot_name: str
    family: str
    contrast_set: str
    source_atom: str
    target_atom: str
    exact_surface_context_sha256: str
    exact_surface_context_overlap_opt_train: bool
    generator_rule_edit_template_sha256: str
    generator_rule_edit_template_overlap_opt_train: bool


@dataclass
class CacheBundle:
    root: Path
    image_ids: List[str]
    visual_tokens: np.ndarray
    visual_global: np.ndarray
    text_part: np.ndarray
    has_text: np.ndarray
    text_tokens: np.ndarray
    text_token_mask: np.ndarray
    text_foil_part: np.ndarray
    text_foil_valid: np.ndarray
    text_foil_tokens: np.ndarray
    text_foil_token_mask: np.ndarray
    opt_train_rows: np.ndarray
    train_all_rows: np.ndarray
    heldout_rows: np.ndarray

    @classmethod
    def load(cls, cache_dir: Path) -> "CacheBundle":
        root = Path(os.path.abspath(cache_dir))
        required = FACTUAL_CACHE_FILES + (
            "semantic_detail_cache_manifest.json",
            "text_foil_part.f16.npy",
            "text_foil_valid.bool.npy",
            "text_foil_image_ids.json",
            "text_foil_tokens.f16.npy",
            "text_foil_token_mask.bool.npy",
            "text_foil_token_image_ids.json",
        )
        missing = [str(root / name) for name in required if not (root / name).is_file()]
        if missing:
            raise ValueError("held-out foil cache is incomplete: " + ", ".join(missing))

        with (root / "image_ids.json").open("r", encoding="utf-8") as handle:
            image_ids = json.load(handle)
        if (
            not isinstance(image_ids, list)
            or any(not isinstance(item, str) or not item for item in image_ids)
            or len(set(image_ids)) != len(image_ids)
        ):
            raise ValueError("image_ids.json must be a unique non-empty string list")
        n_rows = len(image_ids)

        def load_array(name: str) -> np.ndarray:
            return np.load(root / name, mmap_mode="r")

        visual_tokens = load_array("visual_tokens.f16.npy")
        visual_global = load_array("visual_global.f16.npy")
        text_part = load_array("text_part.f16.npy")
        has_text = load_array("has_text.bool.npy")
        text_tokens = load_array("text_tokens.f16.npy")
        text_token_mask = load_array("text_token_mask.bool.npy")
        text_foil_part = load_array("text_foil_part.f16.npy")
        text_foil_valid = load_array("text_foil_valid.bool.npy")
        text_foil_tokens = load_array("text_foil_tokens.f16.npy")
        text_foil_token_mask = load_array("text_foil_token_mask.bool.npy")

        if visual_tokens.ndim != 3 or visual_tokens.shape[0] != n_rows:
            raise ValueError(f"invalid visual token shape {visual_tokens.shape}")
        if visual_global.ndim != 2 or visual_global.shape[0] != n_rows:
            raise ValueError(f"invalid visual global shape {visual_global.shape}")
        if text_part.ndim != 3 or text_part.shape[:2] != (n_rows, N_SLOTS):
            raise ValueError(f"invalid factual pooled-text shape {text_part.shape}")
        if has_text.shape != (n_rows,) or has_text.dtype != np.bool_:
            raise ValueError("has_text.bool.npy must be bool [N]")
        if (
            text_tokens.ndim != 4
            or text_tokens.shape[:2] != (n_rows, N_SLOTS)
            or text_tokens.shape[-1] != text_part.shape[-1]
        ):
            raise ValueError(f"invalid factual text-token shape {text_tokens.shape}")
        if (
            text_token_mask.shape != text_tokens.shape[:3]
            or text_token_mask.dtype != np.bool_
        ):
            raise ValueError("factual token mask must be bool [N,6,T]")
        if text_foil_part.shape != text_part.shape:
            raise ValueError("factual and foil pooled-text shapes differ")
        if (
            text_foil_valid.shape != (n_rows, N_SLOTS)
            or text_foil_valid.dtype != np.bool_
        ):
            raise ValueError("foil validity must be bool [N,6]")
        if bool(np.asarray(text_foil_valid[:, 0]).any()):
            raise ValueError("C_global foil validity must be false for every row")
        if text_foil_tokens.shape != text_tokens.shape:
            raise ValueError("factual and foil token shapes differ")
        if (
            text_foil_token_mask.shape != text_token_mask.shape
            or text_foil_token_mask.dtype != np.bool_
        ):
            raise ValueError("foil token mask must be bool and match factual shape")
        if bool(np.asarray(text_foil_token_mask[:, 0, :]).any()):
            raise ValueError("C_global foil token mask must be empty")
        foil_token_valid = np.asarray(text_foil_token_mask).any(axis=-1)
        if not np.array_equal(foil_token_valid, np.asarray(text_foil_valid)):
            raise ValueError("foil token coverage does not equal pooled validity")

        for filename in (
            "text_foil_image_ids.json",
            "text_foil_token_image_ids.json",
        ):
            with (root / filename).open("r", encoding="utf-8") as handle:
                aligned_ids = json.load(handle)
            if aligned_ids != image_ids:
                raise ValueError(f"{filename} does not match image_ids.json")

        opt_train_rows = _load_unique_rows(root / "opt_train_rows.npy", n_rows)
        train_all_rows = _load_unique_rows(root / "train_all_rows.npy", n_rows)
        if not set(opt_train_rows.tolist()).issubset(set(train_all_rows.tolist())):
            raise ValueError("opt_train_rows is not a subset of train_all_rows")
        heldout_rows = np.setdiff1d(
            train_all_rows, opt_train_rows, assume_unique=True
        ).astype(np.int64)
        if heldout_rows.size == 0:
            raise ValueError("train_all - opt_train produced no held-out rows")
        if np.intersect1d(opt_train_rows, heldout_rows).size:
            raise AssertionError("optimization-train and held-out rows overlap")

        return cls(
            root=root,
            image_ids=image_ids,
            visual_tokens=visual_tokens,
            visual_global=visual_global,
            text_part=text_part,
            has_text=has_text,
            text_tokens=text_tokens,
            text_token_mask=text_token_mask,
            text_foil_part=text_foil_part,
            text_foil_valid=text_foil_valid,
            text_foil_tokens=text_foil_tokens,
            text_foil_token_mask=text_foil_token_mask,
            opt_train_rows=opt_train_rows,
            train_all_rows=train_all_rows,
            heldout_rows=heldout_rows,
        )


def validate_cache_manifest(
    cache: CacheBundle, foil_jsonl: Path, configured_whiten_npz: Path
) -> Dict[str, Any]:
    """Bind the supplied foil JSONL to the exact pooled/token sidecar overlay."""

    path = cache.root / "semantic_detail_cache_manifest.json"
    try:
        with path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid semantic-detail cache manifest {path}: {exc}") from exc
    if not isinstance(manifest, Mapping):
        raise ValueError("semantic-detail cache manifest must be a JSON object")
    if manifest.get("schema_version") != (
        "groundeddna-semantic-detail-cache-overlay-v1"
    ):
        raise ValueError(
            "unsupported semantic-detail cache manifest schema: "
            f"{manifest.get('schema_version')!r}"
        )
    foil_sha = _sha256_file(foil_jsonl)
    image_ids_sha = _sha256_file(cache.root / "image_ids.json")
    if manifest.get("foil_jsonl_sha256") != foil_sha:
        raise ValueError(
            "cache manifest foil_jsonl_sha256 does not match supplied JSONL"
        )
    if manifest.get("image_ids_sha256") != image_ids_sha:
        raise ValueError(
            "cache manifest image_ids_sha256 does not match cache row order"
        )
    pooled_shape = [int(item) for item in cache.text_foil_part.shape]
    token_shape = [int(item) for item in cache.text_foil_tokens.shape]
    valid_per_slot = [
        int(np.asarray(cache.text_foil_valid[:, slot]).sum())
        for slot in range(N_SLOTS)
    ]
    if manifest.get("pooled_shape") != pooled_shape:
        raise ValueError(
            "cache manifest pooled_shape does not match foil pooled sidecar"
        )
    if manifest.get("token_shape") != token_shape:
        raise ValueError(
            "cache manifest token_shape does not match foil token sidecar"
        )
    if manifest.get("valid_per_slot") != valid_per_slot:
        raise ValueError(
            "cache manifest valid_per_slot does not match foil validity sidecar"
        )
    whiten_path = Path(os.path.abspath(configured_whiten_npz))
    if not whiten_path.is_file():
        raise ValueError(f"configured whitening NPZ is missing: {whiten_path}")
    whitening = manifest.get("whitening")
    if not isinstance(whitening, Mapping):
        raise ValueError("cache manifest has no whitening provenance object")
    whiten_entry = whitening.get(whiten_path.name)
    if not isinstance(whiten_entry, Mapping):
        raise ValueError(
            "cache manifest has no entry for configured whitening file "
            f"{whiten_path.name}"
        )
    whiten_sha = _sha256_file(whiten_path)
    opt_rows_sha = _sha256_file(cache.root / "opt_train_rows.npy")
    if whiten_entry.get("sha256") != whiten_sha:
        raise ValueError(
            "cache manifest whitening SHA-256 does not match configured NPZ"
        )
    if whiten_entry.get("row_index_sha256") != opt_rows_sha:
        raise ValueError(
            "cache manifest whitening row_index_sha256 does not match "
            "opt_train_rows.npy"
        )
    # The localOnly whitening builder flattens five local slots per image;
    # its historical ``rows_used`` field therefore counts vectors, not image
    # rows (e.g. Flickr: 4,500 opt-train images -> 22,500 vectors).
    expected_whiten_vectors = len(cache.opt_train_rows) * len(LOCAL_SLOTS)
    if whiten_entry.get("rows_used") != expected_whiten_vectors:
        raise ValueError(
            "cache manifest whitening rows_used does not equal "
            "opt-train images times five local slots"
        )
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "schema_version": manifest["schema_version"],
        "foil_jsonl_sha256": foil_sha,
        "image_ids_sha256": image_ids_sha,
        "pooled_shape": pooled_shape,
        "token_shape": token_shape,
        "valid_per_slot": valid_per_slot,
        "whitening": {
            "path": str(whiten_path),
            "basename": whiten_path.name,
            "sha256": whiten_sha,
            "row_index_sha256": opt_rows_sha,
            "opt_train_image_rows": len(cache.opt_train_rows),
            "local_vectors_used": expected_whiten_vectors,
            "validation_rows_excluded_from_fit": True,
        },
        "verified": True,
    }


def _load_unique_rows(path: Path, n_rows: int) -> np.ndarray:
    rows = np.asarray(np.load(path), dtype=np.int64).reshape(-1)
    if rows.size == 0:
        raise ValueError(f"{path} contains no rows")
    if np.unique(rows).size != rows.size:
        raise ValueError(f"{path} contains duplicate rows")
    if int(rows.min()) < 0 or int(rows.max()) >= n_rows:
        raise ValueError(f"{path} contains out-of-range rows for N={n_rows}")
    return np.sort(rows)


def _load_foil_entries(path: Path) -> Dict[str, FoilEntry]:
    entries: Dict[str, FoilEntry] = {}
    schemas: set[Tuple[str, ...]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            image_id = row.get("image_id")
            if not isinstance(image_id, str) or not image_id:
                raise ValueError(f"{path}:{line_number}: invalid image_id")
            if image_id in entries:
                raise ValueError(f"{path}:{line_number}: duplicate image_id {image_id}")
            if row.get("foil_schema_version") != EXPECTED_FOIL_SCHEMA:
                raise ValueError(
                    f"{path}:{line_number}: unsupported foil schema "
                    f"{row.get('foil_schema_version')!r}"
                )
            slot_keys_raw = row.get("slot_keys")
            texts_raw = row.get("foil_codebook_texts")
            valid_raw = row.get("foil_valid")
            edits_raw = row.get("foil_edits")
            if (
                not isinstance(slot_keys_raw, list)
                or len(slot_keys_raw) != N_SLOTS
                or slot_keys_raw[0] != "C_global"
                or len(set(slot_keys_raw)) != N_SLOTS
                or not isinstance(texts_raw, dict)
                or not isinstance(valid_raw, dict)
                or not isinstance(edits_raw, dict)
            ):
                raise ValueError(f"{path}:{line_number}: malformed foil row")
            slot_keys = tuple(str(item) for item in slot_keys_raw)
            schemas.add(slot_keys)
            texts: List[str] = []
            valid: List[bool] = []
            edits: List[Mapping[str, Any]] = []
            for slot, key in enumerate(slot_keys):
                text = str(texts_raw.get(key, "") or "")
                is_valid = bool(valid_raw.get(key, False))
                edit = edits_raw.get(key)
                if not isinstance(edit, dict):
                    raise ValueError(f"{path}:{line_number}: missing edit for {key}")
                if bool(edit.get("valid", False)) != is_valid:
                    raise ValueError(
                        f"{path}:{line_number}: edit/valid mismatch for {key}"
                    )
                if slot == 0 and is_valid:
                    raise ValueError(f"{path}:{line_number}: C_global is valid")
                if is_valid and not text:
                    raise ValueError(
                        f"{path}:{line_number}: valid {key} has empty foil text"
                    )
                if not is_valid and text:
                    raise ValueError(
                        f"{path}:{line_number}: invalid {key} has foil text"
                    )
                if is_valid:
                    _validate_edit_span(path, line_number, key, text, edit)
                texts.append(text)
                valid.append(is_valid)
                edits.append(edit)
            entries[image_id] = FoilEntry(
                image_id=image_id,
                slot_keys=slot_keys,
                texts=tuple(texts),
                valid=tuple(valid),
                edits=tuple(edits),
            )
    if not entries:
        raise ValueError(f"{path}: no foil rows")
    if len(schemas) != 1:
        raise ValueError(f"{path}: mixed slot schemas are unsupported")
    return entries


def _validate_edit_span(
    path: Path,
    line_number: int,
    slot_key: str,
    foil_text: str,
    edit: Mapping[str, Any],
) -> None:
    target_span = edit.get("target_span")
    source_span = edit.get("source_span")
    source_atom = edit.get("source_atom")
    target_atom = edit.get("target_atom")
    role = edit.get("role")
    family = edit.get("family")
    contrast_set = edit.get("contrast_set")
    if (
        not isinstance(target_span, list)
        or len(target_span) != 2
        or not all(isinstance(item, int) for item in target_span)
        or not isinstance(source_span, list)
        or len(source_span) != 2
        or not all(isinstance(item, int) for item in source_span)
        or not isinstance(source_atom, str)
        or not source_atom
        or not isinstance(target_atom, str)
        or not target_atom
        or not isinstance(role, str)
        or not role.strip()
        or not isinstance(family, str)
        or not family.strip()
        or not isinstance(contrast_set, str)
        or not contrast_set.strip()
    ):
        raise ValueError(
            f"{path}:{line_number}: malformed atomic edit for {slot_key}"
        )
    start, end = target_span
    if start < 0 or end <= start or end > len(foil_text):
        raise ValueError(
            f"{path}:{line_number}: out-of-range target span for {slot_key}"
        )
    if foil_text[start:end] != target_atom:
        raise ValueError(
            f"{path}:{line_number}: target span/atom mismatch for {slot_key}"
        )
    if edit.get("operation") != "single_span_contrast_substitution":
        raise ValueError(
            f"{path}:{line_number}: non-atomic operation for {slot_key}"
        )
    if source_atom.casefold() == target_atom.casefold():
        raise ValueError(
            f"{path}:{line_number}: source_atom equals target_atom for {slot_key}"
        )
    if edit.get("target_absent_from_all_factual_captions") is not True:
        raise ValueError(
            f"{path}:{line_number}: target-absence contract is not true for "
            f"{slot_key}"
        )


def exact_surface_context_signature(entry: FoilEntry, slot: int) -> str:
    """Hash the exact target-caption context with only the edited span masked.

    No case folding or whitespace normalization is used.  The slot key is part
    of the signature, so identical English scaffolds in different semantic
    roles are not conflated.
    """

    if slot not in LOCAL_SLOTS or not entry.valid[slot]:
        raise ValueError("exact surface context requires a valid local foil")
    text = entry.texts[slot]
    edit = entry.edits[slot]
    start, end = edit["target_span"]
    surface = (
        entry.slot_keys[slot]
        + "\0"
        + text[:start]
        + "<EDIT>"
        + text[end:]
    )
    return hashlib.sha256(surface.encode("utf-8")).hexdigest()


def generator_rule_edit_template_signature(entry: FoilEntry, slot: int) -> str:
    """Hash the exact generator rule/edit identity for leakage auditing.

    Unlike the full-caption surface-context signature, this deliberately
    collapses different natural sentences that reused the same lexical edit
    rule.  Canonical JSON serialization makes the definition stable and
    inspectable.
    """

    if slot not in LOCAL_SLOTS or not entry.valid[slot]:
        raise ValueError("generator-rule template requires a valid local foil")
    edit = entry.edits[slot]
    identity = {
        "slot_key": entry.slot_keys[slot],
        "role": str(edit["role"]),
        "family": str(edit["family"]),
        "contrast_set": str(edit["contrast_set"]),
        "source_atom": str(edit["source_atom"]),
        "target_atom": str(edit["target_atom"]),
    }
    canonical = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_foil_alignment(
    cache: CacheBundle, entries: Mapping[str, FoilEntry]
) -> None:
    id_to_row = {image_id: row for row, image_id in enumerate(cache.image_ids)}
    expected = np.zeros_like(np.asarray(cache.text_foil_valid), dtype=bool)
    for image_id, entry in entries.items():
        row = id_to_row.get(image_id)
        if row is None:
            raise ValueError(f"foil image_id is absent from cache: {image_id}")
        expected[row] = np.asarray(entry.valid, dtype=bool)
    observed = np.asarray(cache.text_foil_valid, dtype=bool)
    if not np.array_equal(expected, observed):
        mismatch = np.argwhere(expected != observed)
        row, slot = mismatch[0].tolist()
        raise ValueError(
            "foil JSONL and sidecar validity differ at "
            f"row={row}, image_id={cache.image_ids[row]!r}, slot={slot}"
        )
    factual_token_valid = np.asarray(cache.text_token_mask).any(axis=-1)
    invalid_factual = np.argwhere(observed & ~factual_token_valid)
    if invalid_factual.size:
        row, slot = invalid_factual[0].tolist()
        raise ValueError(
            "valid foil has an all-false factual target token mask at "
            f"row={row}, image_id={cache.image_ids[row]!r}, slot={slot}"
        )


def build_pair_specs(
    cache: CacheBundle, entries: Mapping[str, FoilEntry]
) -> List[PairSpec]:
    """Build the fixed pair universe shared by every checkpoint."""

    _validate_foil_alignment(cache, entries)
    train_surface_contexts: set[str] = set()
    train_generator_rule_templates: set[str] = set()
    for row in cache.opt_train_rows.tolist():
        entry = entries.get(cache.image_ids[row])
        if entry is None:
            continue
        for slot in LOCAL_SLOTS:
            if entry.valid[slot]:
                train_surface_contexts.add(
                    exact_surface_context_signature(entry, slot)
                )
                train_generator_rule_templates.add(
                    generator_rule_edit_template_signature(entry, slot)
                )

    pairs: List[PairSpec] = []
    for row in cache.heldout_rows.tolist():
        image_id = cache.image_ids[row]
        entry = entries.get(image_id)
        observed_valid = np.asarray(cache.text_foil_valid[row], dtype=bool)
        if observed_valid[0]:
            raise ValueError("C_global entered held-out pair universe")
        if entry is None:
            if observed_valid.any():
                raise ValueError(f"valid sidecar has no metadata for {image_id}")
            continue
        for slot in LOCAL_SLOTS:
            if not observed_valid[slot]:
                continue
            if not bool(cache.has_text[row]):
                raise ValueError(
                    f"valid held-out foil belongs to has_text=False row {image_id}"
                )
            if not bool(np.asarray(cache.text_token_mask[row, slot]).any()):
                raise ValueError(
                    "valid held-out foil has an empty factual target token "
                    f"mask: image_id={image_id!r}, slot={slot}"
                )
            edit = entry.edits[slot]
            surface_signature = exact_surface_context_signature(entry, slot)
            rule_signature = generator_rule_edit_template_signature(entry, slot)
            pair_id = hashlib.sha256(
                (
                    f"{image_id}\0{slot}\0{surface_signature}\0"
                    f"{rule_signature}"
                ).encode("utf-8")
            ).hexdigest()
            pairs.append(
                PairSpec(
                    pair_id=pair_id,
                    cache_row=int(row),
                    image_id=image_id,
                    slot=int(slot),
                    slot_name=entry.slot_keys[slot],
                    family=str(edit.get("family", "unknown")),
                    contrast_set=str(edit.get("contrast_set", "unknown")),
                    source_atom=str(edit.get("source_atom", "")),
                    target_atom=str(edit.get("target_atom", "")),
                    exact_surface_context_sha256=surface_signature,
                    exact_surface_context_overlap_opt_train=(
                        surface_signature in train_surface_contexts
                    ),
                    generator_rule_edit_template_sha256=rule_signature,
                    generator_rule_edit_template_overlap_opt_train=(
                        rule_signature in train_generator_rule_templates
                    ),
                )
            )
    pairs.sort(key=lambda item: (item.cache_row, item.slot, item.pair_id))
    if not pairs:
        raise ValueError("held-out rows contain no valid local factual/foil pairs")
    if any(pair.slot == 0 for pair in pairs):
        raise AssertionError("C_global pair escaped exclusion")
    return pairs


def pair_set_sha256(pairs: Sequence[PairSpec]) -> str:
    return _sha256_strings(
        (
            f"{pair.cache_row}\0{pair.image_id}\0{pair.slot}\0"
            f"{pair.exact_surface_context_sha256}\0"
            f"{pair.generator_rule_edit_template_sha256}"
        )
        for pair in pairs
    )


def _parse_run(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise ValueError("--run must be LABEL=STAGE1_RESULT_DIR")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", label):
        raise ValueError(f"invalid run label {label!r}")
    path = _resolve_repo_path(raw_path.strip())
    return label, path


def load_run_specs(values: Sequence[str]) -> List[RunSpec]:
    specs: List[RunSpec] = []
    seen: set[str] = set()
    for value in values:
        label, run_dir = _parse_run(value)
        if label in seen:
            raise ValueError(f"duplicate run label {label!r}")
        seen.add(label)
        if not run_dir.is_dir():
            raise ValueError(f"run directory not found: {run_dir}")
        config_path = run_dir / "config.pt"
        checkpoint_path = run_dir / "model_state_dict_best.pth"
        if not config_path.is_file():
            raise ValueError(f"stage-1 config not found: {config_path}")
        if not checkpoint_path.is_file():
            raise ValueError(
                f"selected stage-1 checkpoint not found: {checkpoint_path}"
            )
        config_obj = torch.load(
            config_path, map_location="cpu", weights_only=False
        )
        if not isinstance(config_obj, Mapping):
            raise ValueError(f"{config_path} did not contain a config mapping")
        specs.append(
            RunSpec(
                label=label,
                run_dir=run_dir,
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                config=dict(config_obj),
            )
        )
    if not specs:
        raise ValueError("at least one --run is required")
    return specs


def _same_value(left: Any, right: Any) -> bool:
    if isinstance(left, float) or isinstance(right, float):
        try:
            return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    return left == right


def validate_run_protocol(specs: Sequence[RunSpec], cache: CacheBundle) -> Dict[str, Any]:
    """Fail closed unless all checkpoints share the same stage-1 protocol."""

    common_defaults = {
        "setting": "setting1",
        "val_split_ratio": 0.0,
        "val_split_seed": 42,
        "num_codebooks": N_SLOTS,
        "num_codons_per_codebook": 3,
        "backbone_type": "siglip2",
        "text_hash_counterfactual_margin": FIXED_MARGIN,
    }
    common_fields = (
        "dataset",
        "setting",
        "val_split_ratio",
        "val_split_seed",
        "num_codebooks",
        "num_codons_per_codebook",
        "codebook_size",
        "backbone_type",
        "text_hash_counterfactual_margin",
        "bidirectional_token_prune",
        "bidirectional_token_prune_mode",
        "bidirectional_token_prune_visual_ratio",
        "bidirectional_token_prune_text_ratio",
        "text_embed_transform",
        "text_whiten_npz",
    )

    def value(spec: RunSpec, field: str) -> Any:
        return spec.config.get(field, common_defaults.get(field))

    reference = specs[0]
    arm_contract_checks: Dict[str, Any] = {}
    selection_evidence: Dict[str, Any] = {}
    for spec in specs:
        if float(value(spec, "val_split_ratio")) <= 0.0:
            raise ValueError(
                f"{spec.label}: checkpoint is not a stage-1 val-split run"
            )
        if (
            "final_epoch_eval" not in spec.config
            or spec.config.get("final_epoch_eval") is not False
        ):
            raise ValueError(
                f"{spec.label}: stage-1 checkpoint has final_epoch_eval=True"
            )
        if (
            "stop_after_epoch" not in spec.config
            or spec.config.get("stop_after_epoch") is not None
        ):
            raise ValueError(
                f"{spec.label}: stage-1 checkpoint has non-null stop_after_epoch"
            )
        required_selection_fields = {
            "_val_protocol": True,
            "best_save": True,
            "val_select_metric": "mAP_at_R",
            "_best_mid_metric": "eval_mAP_at_R",
            "random_seed": 42,
        }
        for field, expected in required_selection_fields.items():
            observed = spec.config.get(field)
            if observed != expected:
                raise ValueError(
                    f"{spec.label}: stage-1 selection contract {field} "
                    f"expected {expected!r}, found {observed!r}"
                )
        best_epoch = spec.config.get("_best_mid_epoch")
        best_map = spec.config.get("_best_mid_mAP")
        if (
            isinstance(best_epoch, bool)
            or not isinstance(best_epoch, int)
            or best_epoch < 0
        ):
            raise ValueError(
                f"{spec.label}: invalid saved _best_mid_epoch {best_epoch!r}"
            )
        try:
            best_map_float = float(best_map)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{spec.label}: invalid saved _best_mid_mAP {best_map!r}"
            ) from exc
        if not math.isfinite(best_map_float):
            raise ValueError(
                f"{spec.label}: non-finite saved _best_mid_mAP {best_map!r}"
            )
        selection_evidence[spec.label] = {
            **required_selection_fields,
            "_best_mid_epoch": best_epoch,
            "_best_mid_mAP": best_map_float,
            "verified": True,
        }
        if int(value(spec, "num_codebooks")) != N_SLOTS:
            raise ValueError(f"{spec.label}: evaluator requires six codebooks")
        if not bool(value(spec, "bidirectional_token_prune")):
            raise ValueError(
                f"{spec.label}: evaluator requires the token-matched "
                "bidirectional pruning path used by A/AB/ABC"
            )
        for ratio_field in (
            "bidirectional_token_prune_visual_ratio",
            "bidirectional_token_prune_text_ratio",
        ):
            try:
                ratio_value = float(value(spec, ratio_field))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{spec.label}: invalid {ratio_field} "
                    f"{value(spec, ratio_field)!r}"
                ) from exc
            if not 0.0 < ratio_value <= 1.0:
                raise ValueError(
                    f"{spec.label}: {ratio_field} must be in (0, 1], "
                    f"found {ratio_value}"
                )
        if str(value(spec, "backbone_type")) != "clip":
            raise ValueError(
                f"{spec.label}: token-matched foil pooling currently requires CLIP"
            )
        if not math.isclose(
            float(value(spec, "text_hash_counterfactual_margin")),
            FIXED_MARGIN,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"{spec.label}: counterfactual margin is not fixed at {FIXED_MARGIN}"
            )
        unsupported = {
            "use_text_token_attention": bool(
                spec.config.get("use_text_token_attention", False)
            ),
            "soft_visual_grounded_text_pool": bool(
                spec.config.get("soft_visual_grounded_text_pool", False)
            ),
            "cosine_visual_grounded_text_pool": bool(
                spec.config.get("cosine_visual_grounded_text_pool", False)
            ),
            "grounded_text_routing": bool(
                spec.config.get("grounded_text_routing", False)
            ),
        }
        active_unsupported = [key for key, enabled in unsupported.items() if enabled]
        if active_unsupported:
            raise ValueError(
                f"{spec.label}: unsupported factual/foil loss-side text path: "
                + ", ".join(active_unsupported)
            )

        arm = spec.label if spec.label in {"A", "AB", "ABC"} else None
        if arm is not None:
            expected_arm: Dict[str, Any] = {
                "evaluation": True,
                "post_eval_compositional": False,
                "xmodal_commit_skip_global": True,
                "cibhash_dynamic_tau_skip_global": True,
                "cibhash_ntxent_source": "visual_token",
                "lambda_cibhash_kl": 0.001,
                "codebook_size": 128,
                "num_codons_per_codebook": 4,
                "text_hash_counterfactual_weight": (
                    0.0 if arm == "A" else 0.10
                ),
                "text_hash_counterfactual_margin": FIXED_MARGIN,
                "text_hash_counterfactual_warmup_epochs": (
                    0 if arm == "A" else 1
                ),
                "cibhash_visual_token_bit_kl": arm == "ABC",
            }
            observed_arm: Dict[str, Any] = {}
            for field, expected in expected_arm.items():
                observed = spec.config.get(field)
                observed_arm[field] = observed
                if isinstance(expected, float):
                    try:
                        valid = math.isclose(
                            float(observed),
                            expected,
                            rel_tol=0.0,
                            abs_tol=1e-12,
                        )
                    except (TypeError, ValueError):
                        valid = False
                else:
                    valid = observed == expected
                if not valid:
                    raise ValueError(
                        f"{spec.label}: arm contract {field} expected "
                        f"{expected!r}, found {observed!r}"
                    )
            arm_contract_checks[spec.label] = {
                "arm": arm,
                "verified": True,
                "expected": expected_arm,
                "observed": observed_arm,
            }
        else:
            arm_contract_checks[spec.label] = {
                "arm": None,
                "verified": False,
                "reason": (
                    "label is not one of A/AB/ABC; shared protocol was "
                    "validated but semantic-detail arm identity was not inferred"
                ),
            }

        expected_cache_raw = spec.config.get("siglip2_feature_cache_dir")
        if not isinstance(expected_cache_raw, str) or not expected_cache_raw:
            raise ValueError(f"{spec.label}: config has no feature cache path")
        expected_cache = _resolve_repo_path(expected_cache_raw)
        _validate_factual_cache_identity(
            expected_cache=expected_cache,
            evaluation_cache=cache.root,
            label=spec.label,
        )

    for spec in specs[1:]:
        for field in common_fields:
            left = value(reference, field)
            right = value(spec, field)
            if field == "text_whiten_npz":
                left = str(_resolve_repo_path(str(left)))
                right = str(_resolve_repo_path(str(right)))
            if not _same_value(left, right):
                raise ValueError(
                    f"run protocol mismatch for {field}: "
                    f"{reference.label}={left!r}, {spec.label}={right!r}"
                )

    ratio = float(value(reference, "val_split_ratio"))
    observed_ratio = float(len(cache.heldout_rows)) / float(len(cache.train_all_rows))
    # Class-stratified rounding can move the count slightly; a one-per-class
    # tolerance is intentionally looser than exact equality but still catches
    # the wrong row contract.
    if abs(observed_ratio - ratio) > 0.02:
        raise ValueError(
            "cache held-out fraction is inconsistent with saved val protocol: "
            f"observed={observed_ratio:.6f}, configured={ratio:.6f}"
        )

    return {
        "dataset": value(reference, "dataset"),
        "setting": value(reference, "setting"),
        "val_split_ratio": ratio,
        "val_split_seed": int(value(reference, "val_split_seed")),
        "val_select_metric": reference.config.get("val_select_metric"),
        "num_codons_per_codebook": int(
            value(reference, "num_codons_per_codebook")
        ),
        "codebook_size": int(value(reference, "codebook_size")),
        "backbone_type": str(value(reference, "backbone_type")),
        "text_embed_transform": str(value(reference, "text_embed_transform")),
        "bidirectional_token_prune_visual_ratio": float(
            value(reference, "bidirectional_token_prune_visual_ratio")
        ),
        "bidirectional_token_prune_text_ratio": float(
            value(reference, "bidirectional_token_prune_text_ratio")
        ),
        "text_whiten_npz": str(
            _resolve_repo_path(str(value(reference, "text_whiten_npz")))
        ),
        "counterfactual_margin": FIXED_MARGIN,
        "observed_heldout_fraction": observed_ratio,
        "stage1_identity": {
            "checkpoint_name": "model_state_dict_best.pth",
            "final_epoch_eval": False,
            "stop_after_epoch": None,
            "verified": True,
        },
        "selection_evidence_by_run": selection_evidence,
        "arm_contract_checks": arm_contract_checks,
    }


def _validate_factual_cache_identity(
    expected_cache: Path, evaluation_cache: Path, label: str
) -> None:
    """Require every factual file to be the same inode/linked donor.

    The A arm points at the factual donor while AB/ABC point at the foil
    overlay.  The cache preparer symlinks donor files into the overlay, so
    ``samefile`` proves that all arms are evaluated on identical visual and
    factual text features without hashing multi-gigabyte arrays.
    """

    if not expected_cache.is_dir():
        raise ValueError(f"{label}: saved feature cache is missing: {expected_cache}")
    for filename in FACTUAL_CACHE_FILES:
        left = expected_cache / filename
        right = evaluation_cache / filename
        if not left.is_file() or not right.is_file():
            raise ValueError(f"{label}: missing factual cache file {filename}")
        try:
            identical = os.path.samefile(left, right)
        except OSError:
            identical = False
        if not identical:
            raise ValueError(
                f"{label}: evaluation overlay does not share factual file "
                f"{filename} with the checkpoint's saved cache"
            )


def _hash_model_state(model: torch.nn.Module) -> str:
    """Canonical SHA-256 over all parameters and buffers, including nonpersistent."""

    items: List[Tuple[str, torch.Tensor]] = []
    items.extend((f"parameter:{name}", tensor) for name, tensor in model.named_parameters())
    items.extend((f"buffer:{name}", tensor) for name, tensor in model.named_buffers())
    digest = hashlib.sha256()
    for name, tensor in sorted(items, key=lambda item: item[0]):
        detached = tensor.detach().cpu().contiguous()
        # Hash raw bytes through a uint8 view so uncommon checkpoint dtypes
        # such as bfloat16 do not depend on NumPy dtype support.
        array = detached.reshape(-1).view(torch.uint8).numpy()
        header = (
            f"{name}\0{str(detached.dtype)}\0{tuple(detached.shape)}\0"
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(memoryview(array))
    return digest.hexdigest()


class FrozenEvaluationGuard:
    """Detect state, RNG, mode, or gradient side effects during evaluation."""

    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.parameter_versions = {
            name: int(parameter._version)
            for name, parameter in model.named_parameters()
        }
        self.buffers = {
            name: buffer.detach().clone()
            for name, buffer in model.named_buffers()
        }
        self.cpu_rng = torch.random.get_rng_state().clone()
        self.cuda_rng = (
            torch.cuda.get_rng_state(device).clone()
            if device.type == "cuda"
            else None
        )

    def verify(self) -> Dict[str, Any]:
        if self.model.training or any(module.training for module in self.model.modules()):
            raise RuntimeError("evaluation changed a module back to train mode")
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise RuntimeError("evaluation model contains requires_grad=True parameters")
        if any(parameter.grad is not None for parameter in self.model.parameters()):
            raise RuntimeError("evaluation materialized parameter gradients")
        changed_versions = [
            name
            for name, parameter in self.model.named_parameters()
            if int(parameter._version) != self.parameter_versions[name]
        ]
        if changed_versions:
            raise RuntimeError(
                "evaluation mutated parameter tensors: "
                + ", ".join(changed_versions[:10])
            )
        changed_buffers = [
            name
            for name, buffer in self.model.named_buffers()
            if name not in self.buffers
            or not torch.equal(buffer, self.buffers[name])
        ]
        if changed_buffers:
            raise RuntimeError(
                "evaluation mutated model buffers/EMA state: "
                + ", ".join(changed_buffers[:10])
            )
        if not torch.equal(torch.random.get_rng_state(), self.cpu_rng):
            raise RuntimeError("evaluation advanced the torch CPU RNG")
        if self.cuda_rng is not None and not torch.equal(
            torch.cuda.get_rng_state(self.device), self.cuda_rng
        ):
            raise RuntimeError("evaluation advanced the torch CUDA RNG")
        return {
            "model_eval_only": True,
            "all_parameters_frozen": True,
            "no_parameter_gradients": True,
            "parameter_versions_unchanged": True,
            "buffers_and_ema_unchanged": True,
            "torch_rng_unchanged": True,
        }


def _batch_tensor(array: np.ndarray, rows: np.ndarray, device: torch.device) -> torch.Tensor:
    materialized = np.asarray(array[rows])
    tensor = torch.from_numpy(materialized.copy())
    return tensor.to(device=device, non_blocking=False)


def encode_text_minimal_pairs(
    model: Any,
    *,
    factual_raw: torch.Tensor,
    factual_tokens: torch.Tensor,
    factual_token_mask: torch.Tensor,
    foil_tokens: torch.Tensor,
    foil_token_mask: torch.Tensor,
    visual_tokens_raw: torch.Tensor,
    visual_attention_mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Encode factual DNA and five independent one-slot foil worlds.

    This function is intentionally testable with a small fake model.  The
    production model methods are reused directly and are called only while the
    caller holds ``torch.inference_mode()`` and the model is in eval mode.
    """

    if factual_raw.dim() != 3 or factual_raw.shape[1] != N_SLOTS:
        raise ValueError("factual_raw must be [B,6,D]")
    batch_size = factual_raw.shape[0]
    factual_local = model._pool_local_foil_tokens_like_factual(
        factual_tokens,
        factual_token_mask,
        visual_tokens_raw,
        visual_attention_mask,
    )
    foil_local = model._pool_local_foil_tokens_like_factual(
        foil_tokens,
        foil_token_mask,
        visual_tokens_raw,
        visual_attention_mask,
    )
    if factual_local.shape[:2] != (batch_size, len(LOCAL_SLOTS)):
        raise ValueError("factual local token pool must be [B,5,D]")
    if foil_local.shape != factual_local.shape:
        raise ValueError("factual/foil local token pools differ in shape")

    factual_pooled = torch.cat(
        [factual_raw[:, :1, :], factual_local.to(factual_raw.dtype)], dim=1
    )
    # [B, 5 target-worlds, 6 slots, D].  Every world begins factual.
    foil_worlds = factual_pooled.unsqueeze(1).repeat(
        1, len(LOCAL_SLOTS), 1, 1
    )
    target_world = torch.arange(
        len(LOCAL_SLOTS), device=factual_raw.device
    )
    target_slot = torch.arange(1, N_SLOTS, device=factual_raw.device)
    foil_worlds[:, target_world, target_slot, :] = foil_local

    factual_adapted = model._adapt_pooled_text_for_loss(factual_pooled)
    flat_worlds = foil_worlds.reshape(
        batch_size * len(LOCAL_SLOTS), N_SLOTS, factual_pooled.shape[-1]
    )
    foil_adapted = model._adapt_pooled_text_for_loss(flat_worlds)
    factual_dna = model._encode_text_tokens_to_dna(
        factual_adapted,
        allow_mm_ema=False,
        deterministic_codon=True,
    )
    foil_world_dna = model._encode_text_tokens_to_dna(
        foil_adapted,
        allow_mm_ema=False,
        deterministic_codon=True,
    )

    factual_continuous_flat = factual_dna["continuous_code"]
    if factual_continuous_flat.shape[0] != batch_size:
        raise ValueError("factual DNA batch size changed")
    total_bases = factual_continuous_flat.shape[1]
    if total_bases % N_SLOTS:
        raise ValueError("text DNA length is not divisible by six slots")
    bases_per_slot = total_bases // N_SLOTS
    factual_continuous = factual_continuous_flat.reshape(
        batch_size, N_SLOTS, bases_per_slot, 4
    )
    foil_all = foil_world_dna["continuous_code"].reshape(
        batch_size,
        len(LOCAL_SLOTS),
        N_SLOTS,
        bases_per_slot,
        4,
    )
    foil_target_continuous = factual_continuous.clone()
    foil_target_codeword = factual_dna["codebook_indices"].clone()
    foil_all_codeword = foil_world_dna["codebook_indices"].reshape(
        batch_size, len(LOCAL_SLOTS), N_SLOTS
    )
    for world, slot in enumerate(LOCAL_SLOTS):
        foil_target_continuous[:, slot] = foil_all[:, world, slot]
        foil_target_codeword[:, slot] = foil_all_codeword[:, world, slot]

    return {
        "factual_continuous": factual_continuous,
        "factual_codeword": factual_dna["codebook_indices"],
        "foil_target_continuous": foil_target_continuous,
        "foil_target_codeword": foil_target_codeword,
        # Exposed for focused invariance tests/audits, not serialized.
        "factual_pooled": factual_pooled,
        "foil_worlds": foil_worlds,
    }


def _cosine_numpy(left: np.ndarray, right: np.ndarray) -> float:
    left_flat = np.asarray(left, dtype=np.float64).reshape(-1)
    right_flat = np.asarray(right, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(left_flat) * np.linalg.norm(right_flat))
    if denom <= 0.0:
        raise ValueError("zero-norm DNA vector in cosine diagnostic")
    return float(np.dot(left_flat, right_flat) / denom)


def compute_pair_records(
    *,
    run_label: str,
    row_order: Sequence[int],
    pair_specs: Sequence[PairSpec],
    visual_continuous: np.ndarray,
    visual_codeword: np.ndarray,
    factual_continuous: np.ndarray,
    factual_codeword: np.ndarray,
    foil_continuous: np.ndarray,
    foil_codeword: np.ndarray,
    margin: float = FIXED_MARGIN,
) -> List[Dict[str, Any]]:
    """Compute one record for every fixed (image, local-slot) pair."""

    row_to_batch = {int(row): index for index, row in enumerate(row_order)}
    batch_size, n_slots, bases_per_slot, alphabet = visual_continuous.shape
    expected_cont_shape = (batch_size, N_SLOTS, bases_per_slot, 4)
    for name, array in (
        ("visual", visual_continuous),
        ("factual", factual_continuous),
        ("foil", foil_continuous),
    ):
        if array.shape != expected_cont_shape:
            raise ValueError(
                f"{name} continuous DNA has shape {array.shape}, "
                f"expected {expected_cont_shape}"
            )
    if n_slots != N_SLOTS or alphabet != 4:
        raise ValueError("DNA tensor must be [B,6,L,4]")
    expected_cw_shape = (batch_size, N_SLOTS)
    for name, array in (
        ("visual", visual_codeword),
        ("factual", factual_codeword),
        ("foil", foil_codeword),
    ):
        if array.shape != expected_cw_shape:
            raise ValueError(f"{name} codewords must be [B,6]")

    records: List[Dict[str, Any]] = []
    for pair in pair_specs:
        batch_index = row_to_batch.get(pair.cache_row)
        if batch_index is None:
            continue
        slot = pair.slot
        if slot == 0:
            raise ValueError("C_global cannot be evaluated as a foil pair")
        visual = visual_continuous[batch_index, slot]
        factual = factual_continuous[batch_index, slot]
        foil = foil_continuous[batch_index, slot]
        factual_cosine = _cosine_numpy(visual, factual)
        foil_cosine = _cosine_numpy(visual, foil)
        cosine_gap = factual_cosine - foil_cosine

        visual_bases = np.argmax(visual, axis=-1)
        factual_bases = np.argmax(factual, axis=-1)
        foil_bases = np.argmax(foil, axis=-1)
        text_hamming_count = int(np.count_nonzero(factual_bases != foil_bases))
        text_hamming_fraction = float(text_hamming_count / bases_per_slot)
        factual_distance = float(
            np.count_nonzero(visual_bases != factual_bases) / bases_per_slot
        )
        foil_distance = float(
            np.count_nonzero(visual_bases != foil_bases) / bases_per_slot
        )
        distance_advantage = foil_distance - factual_distance

        visual_cw = int(visual_codeword[batch_index, slot])
        factual_cw = int(factual_codeword[batch_index, slot])
        foil_cw = int(foil_codeword[batch_index, slot])
        visual_matches_factual = visual_cw == factual_cw
        visual_matches_foil = visual_cw == foil_cw

        records.append(
            {
                "run": run_label,
                "pair_id": pair.pair_id,
                "cache_row": pair.cache_row,
                "image_id": pair.image_id,
                "slot": slot,
                "slot_name": pair.slot_name,
                "family": pair.family,
                "contrast_set": pair.contrast_set,
                "source_atom": pair.source_atom,
                "target_atom": pair.target_atom,
                "exact_surface_context_sha256": (
                    pair.exact_surface_context_sha256
                ),
                "exact_surface_context_overlap_opt_train": (
                    pair.exact_surface_context_overlap_opt_train
                ),
                "exact_surface_context_unseen_in_opt_train": (
                    not pair.exact_surface_context_overlap_opt_train
                ),
                "generator_rule_edit_template_sha256": (
                    pair.generator_rule_edit_template_sha256
                ),
                "generator_rule_edit_template_overlap_opt_train": (
                    pair.generator_rule_edit_template_overlap_opt_train
                ),
                "generator_rule_edit_template_unseen_in_opt_train": (
                    not pair.generator_rule_edit_template_overlap_opt_train
                ),
                "bases_per_local_slot": bases_per_slot,
                "visual_factual_cosine": factual_cosine,
                "visual_foil_cosine": foil_cosine,
                "visual_cosine_gap": cosine_gap,
                "factual_preference": cosine_gap > 0.0,
                "margin": float(margin),
                "margin_violation": cosine_gap <= float(margin),
                "margin_satisfied": cosine_gap > float(margin),
                "factual_text_codeword": factual_cw,
                "foil_text_codeword": foil_cw,
                "text_codeword_flip": factual_cw != foil_cw,
                "text_dna_base_hamming_count": text_hamming_count,
                "text_dna_base_hamming_fraction": text_hamming_fraction,
                "visual_text_base_distance_factual": factual_distance,
                "visual_text_base_distance_foil": foil_distance,
                "visual_text_base_distance_advantage": distance_advantage,
                "visual_dna_prefers_factual_by_base_distance": (
                    distance_advantage > 0.0
                ),
                "visual_codeword": visual_cw,
                "visual_codeword_matches_factual": visual_matches_factual,
                "visual_codeword_matches_foil": visual_matches_foil,
                "visual_codeword_match_advantage": (
                    int(visual_matches_factual) - int(visual_matches_foil)
                ),
                "visual_codeword_exclusive_factual_match": (
                    visual_matches_factual and not visual_matches_foil
                ),
                "visual_codeword_exclusive_foil_match": (
                    visual_matches_foil and not visual_matches_factual
                ),
            }
        )
    return records


def _metric_bootstrap(
    records: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    n_bootstrap: int,
    seed: int,
) -> Dict[str, Any]:
    values = np.asarray([float(record[metric]) for record in records], dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return {
            "mean": None,
            "ci95_image_bootstrap": [None, None],
            "n_finite_pairs": 0,
        }
    values = values[finite]
    image_ids = np.asarray(
        [str(record["image_id"]) for record, keep in zip(records, finite) if keep],
        dtype=object,
    )
    unique_images = sorted(set(image_ids.tolist()))
    image_index = {image_id: index for index, image_id in enumerate(unique_images)}
    sums = np.zeros(len(unique_images), dtype=np.float64)
    counts = np.zeros(len(unique_images), dtype=np.float64)
    for image_id, value in zip(image_ids.tolist(), values.tolist()):
        index = image_index[image_id]
        sums[index] += value
        counts[index] += 1.0
    point = float(values.mean())
    if n_bootstrap <= 0:
        interval: List[Optional[float]] = [None, None]
    else:
        rng = np.random.default_rng(int(seed))
        samples = rng.integers(
            0,
            len(unique_images),
            size=(int(n_bootstrap), len(unique_images)),
            endpoint=False,
        )
        boot_sum = sums[samples].sum(axis=1)
        boot_count = counts[samples].sum(axis=1)
        boot = boot_sum / np.maximum(boot_count, 1.0)
        interval = [
            float(np.quantile(boot, 0.025)),
            float(np.quantile(boot, 0.975)),
        ]
    return {
        "mean": point,
        "ci95_image_bootstrap": interval,
        "n_finite_pairs": int(values.size),
    }


def aggregate_records(
    records: Sequence[Mapping[str, Any]],
    *,
    n_bootstrap: int,
    seed: int,
    scope: str,
) -> Dict[str, Any]:
    if not records:
        return {
            "n_pairs": 0,
            "n_images": 0,
            "bootstrap_unit": "image_cluster",
            "n_bootstrap": int(n_bootstrap),
            "metrics": {
                metric: {
                    "mean": None,
                    "ci95_image_bootstrap": [None, None],
                    "n_finite_pairs": 0,
                }
                for metric in PAIR_METRICS
            },
        }
    scope_seed = _stable_seed(seed, scope)
    return {
        "n_pairs": len(records),
        "n_images": len({str(record["image_id"]) for record in records}),
        "bootstrap_unit": "image_cluster",
        "n_bootstrap": int(n_bootstrap),
        "bootstrap_seed": int(scope_seed),
        "metrics": {
            metric: _metric_bootstrap(
                records,
                metric=metric,
                n_bootstrap=n_bootstrap,
                seed=_stable_seed(scope_seed, metric),
            )
            for metric in PAIR_METRICS
        },
    }


def summarize_records(
    records: Sequence[Mapping[str, Any]],
    *,
    n_bootstrap: int,
    seed: int,
    scope_prefix: str,
) -> Dict[str, Any]:
    def grouped(field: str) -> Dict[str, List[Mapping[str, Any]]]:
        groups: Dict[str, List[Mapping[str, Any]]] = {}
        for record in records:
            groups.setdefault(str(record[field]), []).append(record)
        return groups

    by_slot = grouped("slot_name")
    by_family = grouped("family")
    surface_overlap = [
        record
        for record in records
        if bool(record["exact_surface_context_overlap_opt_train"])
    ]
    surface_unseen = [
        record
        for record in records
        if not bool(record["exact_surface_context_overlap_opt_train"])
    ]
    rule_overlap = [
        record
        for record in records
        if bool(record["generator_rule_edit_template_overlap_opt_train"])
    ]
    rule_unseen = [
        record
        for record in records
        if not bool(record["generator_rule_edit_template_overlap_opt_train"])
    ]
    return {
        "overall": aggregate_records(
            records,
            n_bootstrap=n_bootstrap,
            seed=seed,
            scope=f"{scope_prefix}/overall",
        ),
        "by_slot": {
            key: aggregate_records(
                value,
                n_bootstrap=n_bootstrap,
                seed=seed,
                scope=f"{scope_prefix}/slot/{key}",
            )
            for key, value in sorted(by_slot.items())
        },
        "by_edit_family": {
            key: aggregate_records(
                value,
                n_bootstrap=n_bootstrap,
                seed=seed,
                scope=f"{scope_prefix}/family/{key}",
            )
            for key, value in sorted(by_family.items())
        },
        "by_exact_surface_context_status": {
            "overlap_with_opt_train": aggregate_records(
                surface_overlap,
                n_bootstrap=n_bootstrap,
                seed=seed,
                scope=f"{scope_prefix}/surface-context/overlap",
            ),
            "unseen_in_opt_train": aggregate_records(
                surface_unseen,
                n_bootstrap=n_bootstrap,
                seed=seed,
                scope=f"{scope_prefix}/surface-context/unseen",
            ),
        },
        "by_generator_rule_edit_template_status": {
            "overlap_with_opt_train": aggregate_records(
                rule_overlap,
                n_bootstrap=n_bootstrap,
                seed=seed,
                scope=f"{scope_prefix}/generator-rule/overlap",
            ),
            "unseen_in_opt_train": aggregate_records(
                rule_unseen,
                n_bootstrap=n_bootstrap,
                seed=seed,
                scope=f"{scope_prefix}/generator-rule/unseen",
            ),
        },
    }


def compare_run_records(
    left_label: str,
    left_records: Sequence[Mapping[str, Any]],
    right_label: str,
    right_records: Sequence[Mapping[str, Any]],
    *,
    n_bootstrap: int,
    seed: int,
) -> Dict[str, Any]:
    """Paired right-minus-left deltas on the identical pair universe."""

    left_by_id = {str(record["pair_id"]): record for record in left_records}
    right_by_id = {str(record["pair_id"]): record for record in right_records}
    if set(left_by_id) != set(right_by_id):
        raise ValueError(
            f"pair universe differs between {left_label} and {right_label}"
        )
    deltas: List[Dict[str, Any]] = []
    for pair_id in sorted(left_by_id):
        left = left_by_id[pair_id]
        right = right_by_id[pair_id]
        identity_fields = (
            "image_id",
            "cache_row",
            "slot",
            "slot_name",
            "family",
            "exact_surface_context_sha256",
            "exact_surface_context_overlap_opt_train",
            "generator_rule_edit_template_sha256",
            "generator_rule_edit_template_overlap_opt_train",
        )
        if any(left[field] != right[field] for field in identity_fields):
            raise ValueError(f"pair metadata changed across runs for {pair_id}")
        row: Dict[str, Any] = {
            "pair_id": pair_id,
            **{field: left[field] for field in identity_fields},
        }
        for metric in PAIR_METRICS:
            row[metric] = float(right[metric]) - float(left[metric])
        deltas.append(row)
    return {
        "left": left_label,
        "right": right_label,
        "direction": f"{right_label}_minus_{left_label}",
        "paired_on": "identical_image_and_local_slot_pair_id",
        "summary": summarize_records(
            deltas,
            n_bootstrap=n_bootstrap,
            seed=seed,
            scope_prefix=f"comparison/{right_label}-minus-{left_label}",
        ),
    }


def _load_model_for_run(
    spec: RunSpec, device: torch.device
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    from model_siglip2 import SigLIP2SemanticOTModel

    config = dict(spec.config)
    config["device"] = torch.device("cpu")
    whiten = config.get("text_whiten_npz")
    if isinstance(whiten, str) and whiten:
        config["text_whiten_npz"] = str(_resolve_repo_path(whiten))
    model = SigLIP2SemanticOTModel(SimpleNamespace(**config))
    checkpoint = torch.load(
        spec.checkpoint_path, map_location="cpu", weights_only=False
    )
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"{spec.checkpoint_path} is not a state dictionary")
    incompat = model.load_state_dict(checkpoint, strict=True)
    if incompat.missing_keys or incompat.unexpected_keys:
        raise RuntimeError(
            f"{spec.label}: strict checkpoint load produced incompatibilities"
        )
    del checkpoint
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    state_before = _hash_model_state(model)
    model.to(device)
    model.device = device
    model.eval()
    return model, {
        "checkpoint_sha256": _sha256_file(spec.checkpoint_path),
        "model_state_sha256_before": state_before,
        "checkpoint_load_strict": True,
    }


def evaluate_one_run(
    spec: RunSpec,
    cache: CacheBundle,
    pairs: Sequence[PairSpec],
    *,
    device: torch.device,
    batch_size: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    model, provenance = _load_model_for_run(spec, device)
    guard = FrozenEvaluationGuard(model, device)
    pair_by_row: Dict[int, List[PairSpec]] = {}
    for pair in pairs:
        pair_by_row.setdefault(pair.cache_row, []).append(pair)

    records: List[Dict[str, Any]] = []
    with torch.inference_mode():
        for offset in range(0, len(cache.heldout_rows), batch_size):
            rows = cache.heldout_rows[offset : offset + batch_size]
            visual_tokens_raw = _batch_tensor(cache.visual_tokens, rows, device).float()
            visual_global = _batch_tensor(cache.visual_global, rows, device).float()
            factual_raw = _batch_tensor(cache.text_part, rows, device).float()
            factual_tokens = _batch_tensor(cache.text_tokens, rows, device).float()
            factual_token_mask = _batch_tensor(
                cache.text_token_mask, rows, device
            ).bool()
            foil_tokens = _batch_tensor(cache.text_foil_tokens, rows, device).float()
            foil_token_mask = _batch_tensor(
                cache.text_foil_token_mask, rows, device
            ).bool()

            # Deployed image-only inference.  No text fallback is supplied:
            # routing is codebook-mean exactly as extraction_siglip2.py.
            visual_out = model(
                pixel_values=None,
                part_input_ids=None,
                part_attention_mask=None,
                return_routing=True,
                cached_visual_tokens_raw=visual_tokens_raw,
                cached_visual_global=visual_global,
                cached_text_part_raw=None,
                cached_has_text=None,
                compute_text_foil=False,
            )
            if visual_out.get("routing_mode") != "codebook_mean":
                raise RuntimeError(
                    f"{spec.label}: deployed visual path did not use codebook_mean"
                )
            text_out = encode_text_minimal_pairs(
                model,
                factual_raw=factual_raw,
                factual_tokens=factual_tokens,
                factual_token_mask=factual_token_mask,
                foil_tokens=foil_tokens,
                foil_token_mask=foil_token_mask,
                visual_tokens_raw=visual_tokens_raw,
                visual_attention_mask=None,
            )
            visual_continuous = (
                visual_out["continuous_codes_per_codebook"].detach().cpu().numpy()
            )
            visual_codeword = (
                visual_out["codebook_indices"].detach().cpu().numpy()
            )
            factual_continuous = (
                text_out["factual_continuous"].detach().cpu().numpy()
            )
            factual_codeword = (
                text_out["factual_codeword"].detach().cpu().numpy()
            )
            foil_continuous = (
                text_out["foil_target_continuous"].detach().cpu().numpy()
            )
            foil_codeword = (
                text_out["foil_target_codeword"].detach().cpu().numpy()
            )
            batch_pairs = [
                pair
                for row in rows.tolist()
                for pair in pair_by_row.get(int(row), ())
            ]
            records.extend(
                compute_pair_records(
                    run_label=spec.label,
                    row_order=rows.tolist(),
                    pair_specs=batch_pairs,
                    visual_continuous=visual_continuous,
                    visual_codeword=visual_codeword,
                    factual_continuous=factual_continuous,
                    factual_codeword=factual_codeword,
                    foil_continuous=foil_continuous,
                    foil_codeword=foil_codeword,
                    margin=FIXED_MARGIN,
                )
            )

    guard_result = guard.verify()
    if len(records) != len(pairs):
        raise RuntimeError(
            f"{spec.label}: emitted {len(records)} records for {len(pairs)} pairs"
        )
    del guard
    model.to(torch.device("cpu"))
    state_after = _hash_model_state(model)
    if state_after != provenance["model_state_sha256_before"]:
        raise RuntimeError(
            f"{spec.label}: full model state hash changed during evaluation"
        )
    provenance.update(
        {
            "model_state_sha256_after": state_after,
            "full_model_state_unchanged": True,
            "state_guard": guard_result,
        }
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return records, provenance


def _template_diagnostic(pairs: Sequence[PairSpec]) -> Dict[str, Any]:
    surface_overlap = [
        pair for pair in pairs if pair.exact_surface_context_overlap_opt_train
    ]
    surface_unseen = [
        pair for pair in pairs if not pair.exact_surface_context_overlap_opt_train
    ]
    rule_overlap = [
        pair
        for pair in pairs
        if pair.generator_rule_edit_template_overlap_opt_train
    ]
    rule_unseen = [
        pair
        for pair in pairs
        if not pair.generator_rule_edit_template_overlap_opt_train
    ]
    rule_unseen_images = {pair.image_id for pair in rule_unseen}
    # This threshold is a reporting warning, not an inferential cutoff.
    tiny_threshold_pairs = 100
    tiny_threshold_images = 30
    return {
        "reference_split": "opt_train_rows",
        "n_pairs": len(pairs),
        "exact_surface_context": {
            "definition": (
                "SHA256(slot_key + NUL + foil target caption with exactly the "
                "edited target span replaced by <EDIT>); no case or whitespace "
                "normalization"
            ),
            "n_overlap_pairs": len(surface_overlap),
            "n_unseen_pairs": len(surface_unseen),
            "overlap_fraction": len(surface_overlap) / len(pairs),
            "all_surface_contexts_unseen": len(surface_overlap) == 0,
            "interpretation": (
                "This is a strict full-caption scaffold identity check. "
                "An unseen surface context does not imply an unseen generator "
                "rule and must not be called a template-disjoint test."
            ),
        },
        "generator_rule_edit_template": {
            "definition": (
                "SHA256 of canonical JSON over slot_key, role, family, "
                "contrast_set, source_atom, and target_atom"
            ),
            "n_overlap_pairs": len(rule_overlap),
            "n_unseen_pairs": len(rule_unseen),
            "overlap_fraction": len(rule_overlap) / len(pairs),
            "n_unseen_images": len(rule_unseen_images),
            "all_generator_rule_templates_unseen": len(rule_overlap) == 0,
            "unseen_subset_is_tiny": (
                len(rule_unseen) < tiny_threshold_pairs
                or len(rule_unseen_images) < tiny_threshold_images
            ),
            "tiny_warning_definition": {
                "pair_threshold": tiny_threshold_pairs,
                "image_threshold": tiny_threshold_images,
                "rule": "n_pairs < pair_threshold OR n_images < image_threshold",
            },
            "interpretation": (
                "This is the primary generator-leakage caveat. Reusing a rule "
                "means the lexical edit mechanism was observed in opt-train "
                "even when the natural sentence scaffold is new."
            ),
        },
        "interpretation": (
            "All overlap/unseen strata remain diagnostics within the same "
            "E*-selection-used validation rows; none is an independent test."
        ),
    }


def _pair_output_path(summary_path: Path) -> Path:
    if summary_path.suffix.lower() == ".json":
        return summary_path.with_name(summary_path.stem + ".pairs.jsonl")
    return Path(str(summary_path) + ".pairs.jsonl")


def _preflight_payload(
    *,
    specs: Sequence[RunSpec],
    cache: CacheBundle,
    pairs: Sequence[PairSpec],
    protocol: Mapping[str, Any],
    foil_jsonl: Path,
    cache_manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "preflight_only",
        "protocol": {
            **protocol,
            "row_role": "optimization-held-out validation",
            "selection_used_for_E_star": True,
            "independent_test": False,
            "claim_status": (
                "selection-used/generator-rule-overlap diagnostic only"
            ),
            "C_global_excluded": True,
            "visual_inference": "deployed image-only codebook_mean routing",
            "foil_world": "one target local slot replaced; factual global/others",
        },
        "cache": {
            "path": str(cache.root),
            "n_rows": len(cache.image_ids),
            "n_train_all_rows": len(cache.train_all_rows),
            "n_opt_train_rows": len(cache.opt_train_rows),
            "n_heldout_rows": len(cache.heldout_rows),
            "heldout_rows_sha256": _sha256_int_rows(cache.heldout_rows),
            "image_ids_sha256": _sha256_file(cache.root / "image_ids.json"),
        },
        "foil_jsonl": {
            "path": str(foil_jsonl),
            "sha256": _sha256_file(foil_jsonl),
        },
        "semantic_detail_cache_manifest": dict(cache_manifest),
        "pair_set": {
            "n_pairs": len(pairs),
            "n_images_with_pairs": len({pair.image_id for pair in pairs}),
            "sha256": pair_set_sha256(pairs),
        },
        "template_diagnostic": _template_diagnostic(pairs),
        "runs": [
            {
                "label": spec.label,
                "run_dir": str(spec.run_dir),
                "config": str(spec.config_path),
                "checkpoint": str(spec.checkpoint_path),
            }
            for spec in specs
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=STAGE1_RESULT_DIR",
        help=(
            "Repeat for A/AB/ABC. The evaluator always loads "
            "model_state_dict_best.pth from the supplied stage-1 directory."
        ),
    )
    parser.add_argument("--cache-dir", required=True, help="Foil overlay cache.")
    parser.add_argument("--foil-jsonl", required=True)
    parser.add_argument("--out", required=True, help="Aggregate JSON output.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate rows, arms, sidecars, and templates; load no model and write nothing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing summary/pair output.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.n_bootstrap < 0:
        parser.error("--n-bootstrap must be non-negative")

    cache_dir = _resolve_repo_path(args.cache_dir)
    foil_jsonl = _resolve_repo_path(args.foil_jsonl)
    summary_path = _resolve_repo_path(args.out)
    pairs_path = _pair_output_path(summary_path)
    if not foil_jsonl.is_file():
        raise SystemExit(f"[heldout-foil] foil JSONL not found: {foil_jsonl}")

    specs = load_run_specs(args.run)
    cache = CacheBundle.load(cache_dir)
    entries = _load_foil_entries(foil_jsonl)
    pairs = build_pair_specs(cache, entries)
    protocol = validate_run_protocol(specs, cache)
    cache_manifest = validate_cache_manifest(
        cache,
        foil_jsonl,
        Path(str(protocol["text_whiten_npz"])),
    )
    preflight = _preflight_payload(
        specs=specs,
        cache=cache,
        pairs=pairs,
        protocol=protocol,
        foil_jsonl=foil_jsonl,
        cache_manifest=cache_manifest,
    )
    print(
        "[heldout-foil] "
        f"heldout_images={len(cache.heldout_rows)} "
        f"valid_pairs={len(pairs)} "
        f"pair_sha256={preflight['pair_set']['sha256']}"
    )
    print(
        "[heldout-foil] DIAGNOSTIC ONLY: validation rows were used for E* "
        "selection; independent_test=false"
    )
    if args.preflight_only:
        print(json.dumps(_as_builtin(preflight), indent=2, sort_keys=True))
        print("[heldout-foil] preflight-only: no model loaded and no files written")
        return 0

    existing = [path for path in (summary_path, pairs_path) if path.exists()]
    if existing and not args.overwrite:
        raise SystemExit(
            "[heldout-foil] refusing to overwrite: "
            + ", ".join(str(path) for path in existing)
            + " (pass --overwrite explicitly)"
        )
    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("[heldout-foil] CUDA requested but unavailable")

    all_records: List[Dict[str, Any]] = []
    records_by_run: Dict[str, List[Dict[str, Any]]] = {}
    run_results: Dict[str, Any] = {}
    for spec in specs:
        print(f"[heldout-foil] evaluating frozen run {spec.label}: {spec.run_dir}")
        records, state_provenance = evaluate_one_run(
            spec,
            cache,
            pairs,
            device=device,
            batch_size=args.batch_size,
        )
        records_by_run[spec.label] = records
        all_records.extend(records)
        run_results[spec.label] = {
            "run_dir": str(spec.run_dir),
            "config_path": str(spec.config_path),
            "config_sha256": _sha256_file(spec.config_path),
            "checkpoint_path": str(spec.checkpoint_path),
            **state_provenance,
            "metrics": summarize_records(
                records,
                n_bootstrap=args.n_bootstrap,
                seed=args.bootstrap_seed,
                scope_prefix=f"run/{spec.label}",
            ),
        }

    comparisons: Dict[str, Any] = {}
    for left_index, left in enumerate(specs):
        for right in specs[left_index + 1 :]:
            key = f"{right.label}_minus_{left.label}"
            comparisons[key] = compare_run_records(
                left.label,
                records_by_run[left.label],
                right.label,
                records_by_run[right.label],
                n_bootstrap=args.n_bootstrap,
                seed=args.bootstrap_seed,
            )

    _atomic_write_jsonl(pairs_path, all_records)
    summary = {
        **preflight,
        "mode": "frozen_evaluation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "batch_size": int(args.batch_size),
        "n_bootstrap": int(args.n_bootstrap),
        "bootstrap_seed": int(args.bootstrap_seed),
        "metric_definitions": {
            "visual_cosine_gap": (
                "cos(fixed image-only visual continuous DNA, factual text DNA) "
                "- cos(fixed image-only visual continuous DNA, one-slot foil text DNA)"
            ),
            "factual_preference": "visual_cosine_gap > 0",
            "margin_violation": f"visual_cosine_gap <= {FIXED_MARGIN}",
            "text_codeword_flip": "factual text codeword != foil text codeword",
            "text_dna_base_hamming_fraction": (
                "argmax-base Hamming(factual text DNA, foil text DNA) / L"
            ),
            "visual_text_base_distance_advantage": (
                "Hamming(fixed visual DNA, foil text DNA)/L - "
                "Hamming(fixed visual DNA, factual text DNA)/L; positive favors factual"
            ),
            "visual_codeword_match_advantage": (
                "I(fixed visual codeword == factual text codeword) - "
                "I(fixed visual codeword == foil text codeword)"
            ),
            "visual_codeword_flip": (
                "undefined and intentionally not reported: the image and its "
                "deployed visual code are fixed; captions do not re-encode the image"
            ),
        },
        "pair_records": {
            "path": str(pairs_path),
            "sha256": _sha256_file(pairs_path),
            "n_records": len(all_records),
            "n_pairs_per_run": len(pairs),
        },
        "runs": run_results,
        "paired_run_comparisons": comparisons,
    }
    _atomic_write_json(summary_path, summary)
    print(f"[heldout-foil] wrote {summary_path}")
    print(f"[heldout-foil] wrote {pairs_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
