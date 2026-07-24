#!/usr/bin/env python3
"""Encode local counterfactual caption foils into cache-aligned sidecars.

Input must be produced by ``build_counterfactual_caption_foils.py``.  Rows are
aligned to a GroundedDNA feature cache through its canonical
``image_ids.json``.  Only valid local captions are encoded; invalid entries,
including all C_global entries, remain zero and are masked by
``text_foil_valid.bool.npy``.

No existing visual or factual-text cache file is modified.  The output
directory contains:

* ``text_foil_part.f16.npy``: [N, 6, D], zero in invalid slots;
* ``text_foil_valid.bool.npy``: [N, 6], C0 always false;
* ``text_foil_image_ids.json``: the exact cache row order;
* ``text_foil_edits.jsonl``: sparse row/atom audit metadata;
* ``text_foil_meta.json``: hashes, encoder settings, and coverage.

Use ``--dry-run`` first; it validates alignment without loading a text model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from collections import Counter
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


SCHEMA_VERSION = "groundeddna-local-caption-foil-features-v1"
EXPECTED_FOIL_SCHEMA = "groundeddna-local-caption-foil-v1"
OUTPUT_FILES = (
    "text_foil_part.f16.npy",
    "text_foil_valid.bool.npy",
    "text_foil_image_ids.json",
    "text_foil_edits.jsonl",
    "text_foil_meta.json",
)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _load_foil_jsonl(path: str) -> Dict[str, dict]:
    result: Dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
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
            if image_id in result:
                raise ValueError(f"{path}:{line_number}: duplicate image_id {image_id!r}")
            if row.get("foil_schema_version") != EXPECTED_FOIL_SCHEMA:
                raise ValueError(
                    f"{path}:{line_number}: unsupported foil_schema_version "
                    f"{row.get('foil_schema_version')!r}"
                )
            keys = row.get("slot_keys")
            texts = row.get("foil_codebook_texts")
            valid = row.get("foil_valid")
            edits = row.get("foil_edits")
            if (
                not isinstance(keys, list)
                or len(keys) != 6
                or keys[0] != "C_global"
                or len(set(keys)) != 6
                or not isinstance(texts, dict)
                or not isinstance(valid, dict)
                or not isinstance(edits, dict)
            ):
                raise ValueError(f"{path}:{line_number}: malformed foil row")
            ordered_texts: List[str] = []
            ordered_valid: List[bool] = []
            for slot, key in enumerate(keys):
                text = str(texts.get(key, "") or "").strip()
                is_valid = bool(valid.get(key, False))
                edit = edits.get(key)
                if not isinstance(edit, dict) or bool(edit.get("valid", False)) != is_valid:
                    raise ValueError(
                        f"{path}:{line_number}: slot {key} edit/valid mismatch"
                    )
                if slot == 0 and is_valid:
                    raise ValueError(
                        f"{path}:{line_number}: C_global must always be invalid"
                    )
                if is_valid and not text:
                    raise ValueError(
                        f"{path}:{line_number}: valid slot {key} has empty foil"
                    )
                if not is_valid and text:
                    raise ValueError(
                        f"{path}:{line_number}: invalid slot {key} has non-empty foil"
                    )
                if not is_valid:
                    text = ""
                ordered_texts.append(text)
                ordered_valid.append(is_valid)
            result[image_id] = {
                "slot_keys": tuple(keys),
                "texts": tuple(ordered_texts),
                "valid": tuple(ordered_valid),
                "edits": edits,
            }
    if not result:
        raise ValueError(f"{path}: no foil rows")
    schemas = {entry["slot_keys"] for entry in result.values()}
    if len(schemas) != 1:
        raise ValueError(f"{path}: mixed slot schemas are not supported: {schemas!r}")
    return result


def _build_alignment(
    image_ids: Sequence[str], foil_rows: Mapping[str, dict]
) -> Tuple[List[Tuple[int, int, str]], np.ndarray, Counter]:
    valid_mask = np.zeros((len(image_ids), 6), dtype=bool)
    encode_items: List[Tuple[int, int, str]] = []
    stats: Counter = Counter()
    for row_index, image_id in enumerate(image_ids):
        entry = foil_rows.get(image_id)
        if entry is None:
            stats["cache_rows_without_foil_row"] += 1
            continue
        stats["cache_rows_with_foil_row"] += 1
        for slot in range(6):
            if not entry["valid"][slot]:
                continue
            if slot == 0:
                raise AssertionError("C_global passed validation as a foil")
            valid_mask[row_index, slot] = True
            encode_items.append((row_index, slot, entry["texts"][slot]))
            stats[f"valid_slot_{slot}"] += 1
    extra = set(foil_rows).difference(image_ids)
    stats["foil_rows_not_in_cache"] = len(extra)
    return encode_items, valid_mask, stats


def _print_alignment(
    image_ids: Sequence[str],
    foil_rows: Mapping[str, dict],
    encode_items: Sequence[Tuple[int, int, str]],
    stats: Counter,
) -> None:
    print(f"[foil-features] cache rows: {len(image_ids)}")
    print(f"[foil-features] foil JSONL rows: {len(foil_rows)}")
    print(
        "[foil-features] aligned rows: "
        f"{stats['cache_rows_with_foil_row']}/{len(image_ids)}"
    )
    print(
        f"[foil-features] foil rows absent from cache: "
        f"{stats['foil_rows_not_in_cache']}"
    )
    print(f"[foil-features] valid local captions to encode: {len(encode_items)}")
    print("[foil-features] slot 0: valid=0 (excluded by contract)")
    for slot in range(1, 6):
        print(
            f"[foil-features] slot {slot}: valid={stats[f'valid_slot_{slot}']}"
        )


def _load_clip_encoder(
    model_name: Optional[str], tokenizer_name: Optional[str], device: str
):
    import torch

    from dna_utils import DEFAULT_CLIP_TOKENIZER_NAME, build_clip_text_tokenizer
    from models.pretrained_backbone_clip import (
        CLIPBackbone,
        DEFAULT_CLIP_BACKBONE,
    )

    resolved_model = model_name or DEFAULT_CLIP_BACKBONE
    resolved_tokenizer = tokenizer_name or DEFAULT_CLIP_TOKENIZER_NAME
    backbone = CLIPBackbone(resolved_model).to(device).eval()
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    tokenizer = build_clip_text_tokenizer(resolved_tokenizer)

    def encode(texts: Sequence[str], max_length: int):
        from models.pretrained_backbone import coerce_pooled_to_tensor

        encoded = tokenizer(
            list(texts),
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(device)
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            pad_id = getattr(tokenizer, "pad_token_id", None)
            attention_mask = (
                (input_ids != pad_id).long()
                if pad_id is not None
                else torch.ones_like(input_ids)
            )
        feature = backbone.model.get_text_features(
            input_ids=input_ids, attention_mask=attention_mask
        )
        return coerce_pooled_to_tensor(feature)

    return encode, int(backbone.projection_dim), resolved_model, resolved_tokenizer


def _load_siglip2_encoder(
    model_name: Optional[str], tokenizer_name: Optional[str], device: str
):
    from transformers import AutoModel, AutoTokenizer

    resolved_model = model_name or "google/siglip2-base-patch16-224"
    resolved_tokenizer = tokenizer_name or resolved_model
    tokenizer = AutoTokenizer.from_pretrained(resolved_tokenizer)
    full_model = AutoModel.from_pretrained(resolved_model).to(device).eval()
    for parameter in full_model.parameters():
        parameter.requires_grad = False
    text_model = (
        full_model.text_model if hasattr(full_model, "text_model") else full_model
    )
    if hasattr(full_model.config, "text_config"):
        dimension = int(full_model.config.text_config.hidden_size)
    else:
        dimension = int(full_model.config.hidden_size)

    def encode(texts: Sequence[str], max_length: int):
        encoded = tokenizer(
            list(texts),
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(device)
        output = text_model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded.get("attention_mask"),
        )
        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            return output.pooler_output
        return output.last_hidden_state[:, -1, :]

    return encode, dimension, resolved_model, resolved_tokenizer


def _atomic_save_npy(path: str, array: np.ndarray) -> None:
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".foil-array-", suffix=".npy", dir=directory)
    try:
        with os.fdopen(fd, "wb") as f:
            np.save(f, array)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def _atomic_write_json(path: str, payload: object) -> None:
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".foil-json-", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def _write_sparse_edit_audit(
    path: str,
    image_ids: Sequence[str],
    foil_rows: Mapping[str, dict],
    valid_mask: np.ndarray,
) -> None:
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=".foil-edits-", suffix=".jsonl", dir=directory
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for row_index, image_id in enumerate(image_ids):
                if not valid_mask[row_index].any():
                    continue
                entry = foil_rows[image_id]
                valid_edits = {
                    key: entry["edits"][key]
                    for slot, key in enumerate(entry["slot_keys"])
                    if valid_mask[row_index, slot]
                }
                payload = {
                    "row": row_index,
                    "image_id": image_id,
                    "edits": valid_edits,
                }
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foil-jsonl", required=True)
    parser.add_argument(
        "--cache-dir",
        required=True,
        help="Existing factual feature cache; used only for row order and shape.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Sidecar output directory. Existing factual cache files are untouched.",
    )
    parser.add_argument("--encoder", choices=("clip", "siglip2"), default="clip")
    parser.add_argument("--model", default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate JSONL/cache alignment without loading a text model or writing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing foil sidecar files in --out-dir.",
    )
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise SystemExit("[foil-features] --batch-size must be positive")
    if not os.path.isfile(args.foil_jsonl):
        raise SystemExit(f"[foil-features] missing foil JSONL: {args.foil_jsonl}")
    image_ids_path = os.path.join(args.cache_dir, "image_ids.json")
    factual_features_path = os.path.join(args.cache_dir, "text_part.f16.npy")
    if not os.path.isfile(image_ids_path):
        raise SystemExit(f"[foil-features] missing {image_ids_path}")
    if not os.path.isfile(factual_features_path):
        raise SystemExit(f"[foil-features] missing {factual_features_path}")

    with open(image_ids_path, "r", encoding="utf-8") as f:
        image_ids = json.load(f)
    if not isinstance(image_ids, list) or len(set(image_ids)) != len(image_ids):
        raise SystemExit("[foil-features] image_ids.json must be a unique JSON list")
    factual_shape = np.load(factual_features_path, mmap_mode="r").shape
    if len(factual_shape) != 3 or factual_shape[:2] != (len(image_ids), 6):
        raise SystemExit(
            f"[foil-features] unexpected factual text shape {factual_shape}; "
            f"expected ({len(image_ids)}, 6, D)"
        )

    foil_rows = _load_foil_jsonl(args.foil_jsonl)
    encode_items, valid_mask, stats = _build_alignment(image_ids, foil_rows)
    _print_alignment(image_ids, foil_rows, encode_items, stats)
    if valid_mask[:, 0].any():
        raise AssertionError("global foil mask invariant failed")
    if not encode_items:
        raise SystemExit("[foil-features] no aligned valid local foil captions")
    if args.dry_run:
        print("[foil-features] dry-run: no model loaded and no files written")
        return 0

    output_paths = {
        filename: os.path.join(args.out_dir, filename) for filename in OUTPUT_FILES
    }
    if not args.overwrite:
        existing = [path for path in output_paths.values() if os.path.exists(path)]
        if existing:
            raise SystemExit(
                "[foil-features] refusing to overwrite existing sidecars: "
                + ", ".join(existing)
                + " (pass --overwrite explicitly)"
            )

    import torch
    from tqdm import tqdm

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[foil-features] loading {args.encoder} text encoder on {device}")
    if args.encoder == "clip":
        encode, dimension, model_name, tokenizer_name = _load_clip_encoder(
            args.model, args.tokenizer, device
        )
    else:
        encode, dimension, model_name, tokenizer_name = _load_siglip2_encoder(
            args.model, args.tokenizer, device
        )
    if dimension != factual_shape[2]:
        raise SystemExit(
            f"[foil-features] encoder dimension {dimension} does not match "
            f"factual cache dimension {factual_shape[2]}"
        )

    os.makedirs(args.out_dir, exist_ok=True)
    feature_shape = (len(image_ids), 6, dimension)
    fd, temporary_feature_path = tempfile.mkstemp(
        prefix=".foil-features-", suffix=".npy", dir=args.out_dir
    )
    os.close(fd)
    features = None
    started = time.time()
    try:
        # Write directly into an on-disk .npy memmap.  A large MSCOCO cache
        # otherwise needs both a float32 working array and a float16 copy.
        features = np.lib.format.open_memmap(
            temporary_feature_path,
            mode="w+",
            dtype=np.float16,
            shape=feature_shape,
        )
        features[:] = 0
        with torch.no_grad():
            for offset in tqdm(
                range(0, len(encode_items), args.batch_size), desc="foil-text"
            ):
                batch = encode_items[offset : offset + args.batch_size]
                texts = [item[2] for item in batch]
                encoded = encode(texts, args.max_length).float().cpu().numpy()
                if encoded.shape != (len(batch), dimension):
                    raise RuntimeError(
                        f"encoder returned {encoded.shape}, expected "
                        f"({len(batch)}, {dimension})"
                    )
                for batch_index, (row_index, slot, _) in enumerate(batch):
                    features[row_index, slot] = encoded[batch_index]
        if np.any(features[:, 0] != 0):
            raise AssertionError("global foil feature entries must remain zero")
        features.flush()
        del features
        features = None
        os.replace(
            temporary_feature_path, output_paths["text_foil_part.f16.npy"]
        )
        temporary_feature_path = ""
    finally:
        if features is not None:
            features.flush()
            del features
        if temporary_feature_path and os.path.exists(temporary_feature_path):
            os.remove(temporary_feature_path)
    elapsed = time.time() - started

    _atomic_save_npy(output_paths["text_foil_valid.bool.npy"], valid_mask)
    _atomic_write_json(output_paths["text_foil_image_ids.json"], image_ids)
    _write_sparse_edit_audit(
        output_paths["text_foil_edits.jsonl"], image_ids, foil_rows, valid_mask
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "foil_jsonl": os.path.abspath(args.foil_jsonl),
        "foil_jsonl_sha256": _sha256_file(args.foil_jsonl),
        "cache_dir": os.path.abspath(args.cache_dir),
        "cache_image_ids_sha256": _sha256_file(image_ids_path),
        "encoder_family": args.encoder,
        "encoder_model": model_name,
        "tokenizer": tokenizer_name,
        "max_length": args.max_length,
        "shape": list(feature_shape),
        "feature_dtype": "float16",
        "valid_mask_dtype": "bool",
        "invalid_feature_fill": 0.0,
        "global_slot_valid": False,
        "aligned_rows": stats["cache_rows_with_foil_row"],
        "valid_per_slot": [int(valid_mask[:, slot].sum()) for slot in range(6)],
        "encoded_captions": len(encode_items),
        "encode_seconds": elapsed,
    }
    _atomic_write_json(output_paths["text_foil_meta.json"], metadata)
    print(
        f"[foil-features] wrote {args.out_dir}: shape={feature_shape}, "
        f"valid={int(valid_mask.sum())}, elapsed={elapsed:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
