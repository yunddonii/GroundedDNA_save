#!/usr/bin/env python3
"""Encode counterfactual foils in the factual CLIP token-feature domain.

This tokens-only extractor mirrors ``extract_clip_text_token_features.py``:

    CLIP text_model.last_hidden_state -> CLIP text_projection

It aligns foil rows to ``cache_dir/image_ids.json`` and writes only token
sidecars.  Pooled ``text_foil_part.f16.npy`` is never created or replaced.
Invalid slots, including all C_global slots, contain zero vectors and an
all-false token mask.

Outputs:

* ``text_foil_tokens.f16.npy``: [N, 6, T, D_proj];
* ``text_foil_token_mask.bool.npy``: [N, 6, T];
* ``text_foil_token_image_ids.json``: exact cache row order;
* ``text_foil_token_meta.json``: provenance and alignment statistics.

The shared ``text_foil_valid.bool.npy`` is written only when absent.  If it
already exists (for example beside pooled foil features), it must exactly
match the foil JSONL and is left untouched.

The factual ``text_tokens.f16.npy`` and ``text_token_mask.bool.npy`` in
``--cache-dir`` are required: their T/D shape and encoder metadata define the
domain the foil tokens must match.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from typing import Optional, Sequence

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
for import_path in (SCRIPT_DIR, REPO_ROOT):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from extract_counterfactual_foil_features import (  # noqa: E402
    _atomic_save_npy,
    _atomic_write_json,
    _build_alignment,
    _load_foil_jsonl,
    _print_alignment,
    _sha256_file,
)


SCHEMA_VERSION = "groundeddna-local-caption-foil-token-features-v1"
TOKEN_OUTPUT_FILES = (
    "text_foil_tokens.f16.npy",
    "text_foil_token_mask.bool.npy",
    "text_foil_token_image_ids.json",
    "text_foil_token_meta.json",
)


def _load_reference_contract(
    cache_dir: str, requested_max_length: Optional[int]
) -> dict:
    image_ids_path = os.path.join(cache_dir, "image_ids.json")
    tokens_path = os.path.join(cache_dir, "text_tokens.f16.npy")
    mask_path = os.path.join(cache_dir, "text_token_mask.bool.npy")
    meta_path = os.path.join(cache_dir, "meta.json")
    for path in (image_ids_path, tokens_path, mask_path):
        if not os.path.isfile(path):
            raise SystemExit(f"[foil-token] missing factual cache artifact: {path}")

    with open(image_ids_path, "r", encoding="utf-8") as f:
        image_ids = json.load(f)
    if (
        not isinstance(image_ids, list)
        or any(not isinstance(item, str) for item in image_ids)
        or len(set(image_ids)) != len(image_ids)
    ):
        raise SystemExit("[foil-token] image_ids.json must be a unique string list")

    reference_tokens = np.load(tokens_path, mmap_mode="r")
    reference_mask = np.load(mask_path, mmap_mode="r")
    token_shape = tuple(reference_tokens.shape)
    mask_shape = tuple(reference_mask.shape)
    token_dtype = reference_tokens.dtype
    mask_dtype = reference_mask.dtype
    del reference_tokens, reference_mask

    if len(token_shape) != 4 or token_shape[:2] != (len(image_ids), 6):
        raise SystemExit(
            f"[foil-token] factual token shape {token_shape} is not "
            f"({len(image_ids)}, 6, T, D)"
        )
    expected_mask_shape = token_shape[:3]
    if mask_shape != expected_mask_shape:
        raise SystemExit(
            f"[foil-token] factual mask shape {mask_shape} does not match "
            f"{expected_mask_shape}"
        )
    if token_dtype != np.float16:
        raise SystemExit(
            f"[foil-token] factual token dtype must be float16, got {token_dtype}"
        )
    if mask_dtype != np.bool_:
        raise SystemExit(
            f"[foil-token] factual token mask dtype must be bool, got {mask_dtype}"
        )

    max_length = token_shape[2]
    dimension = token_shape[3]
    if requested_max_length is not None and requested_max_length != max_length:
        raise SystemExit(
            f"[foil-token] requested max length {requested_max_length} differs "
            f"from factual token cache T={max_length}"
        )

    metadata = {}
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    meta_length = metadata.get("text_token_max_length")
    meta_dimension = metadata.get("text_token_dim")
    if meta_length is not None and int(meta_length) != max_length:
        raise SystemExit(
            f"[foil-token] meta text_token_max_length={meta_length} but "
            f"factual array has T={max_length}"
        )
    if meta_dimension is not None and int(meta_dimension) != dimension:
        raise SystemExit(
            f"[foil-token] meta text_token_dim={meta_dimension} but "
            f"factual array has D={dimension}"
        )

    return {
        "image_ids": image_ids,
        "image_ids_path": image_ids_path,
        "tokens_path": tokens_path,
        "mask_path": mask_path,
        "meta_path": meta_path,
        "metadata": metadata,
        "max_length": max_length,
        "dimension": dimension,
        "shape": token_shape,
    }


def _resolve_encoder_identity(
    reference_meta: dict,
    requested_model: Optional[str],
    requested_tokenizer: Optional[str],
) -> tuple:
    from dna_utils import DEFAULT_CLIP_TOKENIZER_NAME
    from models.pretrained_backbone_clip import DEFAULT_CLIP_BACKBONE

    reference_model = (
        reference_meta.get("text_token_encoder")
        or reference_meta.get("backbone")
        or reference_meta.get("text_encoder")
    )
    reference_tokenizer = reference_meta.get("tokenizer")
    model_name = requested_model or reference_model or DEFAULT_CLIP_BACKBONE
    tokenizer_name = (
        requested_tokenizer
        or reference_tokenizer
        or model_name
        or DEFAULT_CLIP_TOKENIZER_NAME
    )
    if requested_model and reference_model and requested_model != reference_model:
        raise SystemExit(
            f"[foil-token] requested encoder {requested_model!r} differs from "
            f"factual token encoder {reference_model!r}"
        )
    if (
        requested_tokenizer
        and reference_tokenizer
        and requested_tokenizer != reference_tokenizer
    ):
        raise SystemExit(
            f"[foil-token] requested tokenizer {requested_tokenizer!r} differs "
            f"from factual tokenizer {reference_tokenizer!r}"
        )
    return model_name, tokenizer_name


def _open_zero_memmap(path: str, dtype, shape: Sequence[int]):
    array = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    array[:] = 0
    return array


def _check_existing_shared_valid_mask(
    path: str, valid_mask: np.ndarray
) -> bool:
    if not os.path.exists(path):
        return False
    existing = np.load(path, mmap_mode="r")
    if existing.shape != valid_mask.shape or existing.dtype != np.bool_:
        raise SystemExit(
            f"[foil-token] existing shared valid mask {path} has "
            f"shape/dtype {existing.shape}/{existing.dtype}; expected "
            f"{valid_mask.shape}/bool"
        )
    matches = np.array_equal(existing, valid_mask)
    del existing
    if not matches:
        raise SystemExit(
            f"[foil-token] existing shared valid mask disagrees with "
            f"--foil-jsonl: {path}"
        )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foil-jsonl", required=True)
    parser.add_argument(
        "--cache-dir",
        required=True,
        help="Factual token cache defining image order and CLIP token domain.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Token-sidecar destination; may be the factual cache directory.",
    )
    parser.add_argument(
        "--clip-backbone",
        default=None,
        help="Defaults to factual cache metadata; an explicit mismatch is rejected.",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Defaults to factual cache metadata; an explicit mismatch is rejected.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Optional assertion; must equal factual text token cache T.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate alignment/domain without loading CLIP or writing files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing token-specific sidecars; pooled files are untouched.",
    )
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise SystemExit("[foil-token] --batch-size must be positive")
    if not os.path.isfile(args.foil_jsonl):
        raise SystemExit(f"[foil-token] missing foil JSONL: {args.foil_jsonl}")

    reference = _load_reference_contract(args.cache_dir, args.max_length)
    image_ids = reference["image_ids"]
    foil_rows = _load_foil_jsonl(args.foil_jsonl)
    encode_items, valid_mask, stats = _build_alignment(image_ids, foil_rows)
    _print_alignment(image_ids, foil_rows, encode_items, stats)
    if valid_mask[:, 0].any():
        raise AssertionError("global foil validity invariant failed")
    if not encode_items:
        raise SystemExit("[foil-token] no aligned valid local foil captions")

    model_name, tokenizer_name = _resolve_encoder_identity(
        reference["metadata"], args.clip_backbone, args.tokenizer
    )
    n_rows = len(image_ids)
    max_length = reference["max_length"]
    dimension = reference["dimension"]
    token_shape = (n_rows, 6, max_length, dimension)
    mask_shape = token_shape[:3]
    token_bytes = int(np.prod(token_shape)) * np.dtype(np.float16).itemsize
    mask_bytes = int(np.prod(mask_shape)) * np.dtype(np.bool_).itemsize
    print(
        f"[foil-token] factual domain: model={model_name}, "
        f"tokenizer={tokenizer_name}, T={max_length}, D={dimension}"
    )
    print(
        f"[foil-token] planned token sidecars: {token_shape}, "
        f"{(token_bytes + mask_bytes) / 1e9:.2f} GB"
    )
    if args.dry_run:
        print("[foil-token] dry-run: no model loaded and no files written")
        return 0

    os.makedirs(args.out_dir, exist_ok=True)
    output_paths = {
        filename: os.path.join(args.out_dir, filename)
        for filename in TOKEN_OUTPUT_FILES
    }
    if not args.overwrite:
        existing_outputs = [
            path for path in output_paths.values() if os.path.exists(path)
        ]
        if existing_outputs:
            raise SystemExit(
                "[foil-token] refusing to overwrite token sidecars: "
                + ", ".join(existing_outputs)
                + " (pass --overwrite explicitly)"
            )
    shared_valid_path = os.path.join(args.out_dir, "text_foil_valid.bool.npy")
    reused_shared_valid = _check_existing_shared_valid_mask(
        shared_valid_path, valid_mask
    )
    free_bytes = shutil.disk_usage(args.out_dir).free
    if free_bytes < token_bytes + mask_bytes:
        raise SystemExit(
            f"[foil-token] insufficient free disk: need at least "
            f"{(token_bytes + mask_bytes) / 1e9:.2f} GB, "
            f"have {free_bytes / 1e9:.2f} GB"
        )

    import torch
    from tqdm import tqdm

    from dna_utils import build_clip_text_tokenizer
    from models.pretrained_backbone_clip import CLIPBackbone

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[foil-token] loading {model_name} on {device}")
    backbone = CLIPBackbone(model_name).to(device).eval()
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    if int(backbone.projection_dim) != dimension:
        raise SystemExit(
            f"[foil-token] encoder projection D={backbone.projection_dim} "
            f"does not match factual token D={dimension}"
        )
    tokenizer = build_clip_text_tokenizer(tokenizer_name)

    token_fd, temporary_tokens = tempfile.mkstemp(
        prefix=".foil-tokens-", suffix=".npy", dir=args.out_dir
    )
    mask_fd, temporary_mask = tempfile.mkstemp(
        prefix=".foil-token-mask-", suffix=".npy", dir=args.out_dir
    )
    os.close(token_fd)
    os.close(mask_fd)
    token_memmap = None
    mask_memmap = None
    true_token_count = 0
    started = time.time()
    try:
        token_memmap = _open_zero_memmap(
            temporary_tokens, np.float16, token_shape
        )
        mask_memmap = _open_zero_memmap(
            temporary_mask, np.bool_, mask_shape
        )
        with torch.no_grad():
            for offset in tqdm(
                range(0, len(encode_items), args.batch_size),
                desc="foil-text-token",
            ):
                batch = encode_items[offset : offset + args.batch_size]
                captions = [item[2] for item in batch]
                encoded = tokenizer(
                    captions,
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
                text_output = backbone.model.text_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                token_hidden = text_output.last_hidden_state
                token_projected = backbone.model.text_projection(token_hidden)
                token_numpy = (
                    token_projected.float().cpu().numpy().astype(np.float16)
                )
                mask_numpy = attention_mask.bool().cpu().numpy()
                expected_token_shape = (len(batch), max_length, dimension)
                if token_numpy.shape != expected_token_shape:
                    raise RuntimeError(
                        f"CLIP returned token shape {token_numpy.shape}, "
                        f"expected {expected_token_shape}"
                    )
                if mask_numpy.shape != (len(batch), max_length):
                    raise RuntimeError(
                        f"tokenizer returned mask shape {mask_numpy.shape}, "
                        f"expected {(len(batch), max_length)}"
                    )
                true_token_count += int(mask_numpy.sum())
                for batch_index, (row_index, slot, _) in enumerate(batch):
                    if slot == 0:
                        raise AssertionError("C_global entered foil token batch")
                    token_memmap[row_index, slot] = token_numpy[batch_index]
                    mask_memmap[row_index, slot] = mask_numpy[batch_index]

        if bool(mask_memmap[:, 0].any()):
            raise AssertionError("C_global foil token mask must remain false")
        token_memmap.flush()
        mask_memmap.flush()
        del token_memmap, mask_memmap
        token_memmap = None
        mask_memmap = None
        os.replace(
            temporary_tokens, output_paths["text_foil_tokens.f16.npy"]
        )
        temporary_tokens = ""
        os.replace(
            temporary_mask, output_paths["text_foil_token_mask.bool.npy"]
        )
        temporary_mask = ""
    finally:
        if token_memmap is not None:
            token_memmap.flush()
            del token_memmap
        if mask_memmap is not None:
            mask_memmap.flush()
            del mask_memmap
        for temporary_path in (temporary_tokens, temporary_mask):
            if temporary_path and os.path.exists(temporary_path):
                os.remove(temporary_path)
    elapsed = time.time() - started

    if reused_shared_valid:
        valid_mask_action = "reused_existing_identical"
    else:
        _atomic_save_npy(shared_valid_path, valid_mask)
        valid_mask_action = "written"
    _atomic_write_json(
        output_paths["text_foil_token_image_ids.json"], image_ids
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "foil_jsonl": os.path.abspath(args.foil_jsonl),
        "foil_jsonl_sha256": _sha256_file(args.foil_jsonl),
        "cache_dir": os.path.abspath(args.cache_dir),
        "cache_image_ids_sha256": _sha256_file(reference["image_ids_path"]),
        "reference_text_tokens": os.path.realpath(reference["tokens_path"]),
        "reference_text_token_mask": os.path.realpath(reference["mask_path"]),
        "token_feature_source": (
            "clip_text_model_last_hidden_state_then_text_projection"
        ),
        "encoder_model": model_name,
        "tokenizer": tokenizer_name,
        "shape": list(token_shape),
        "mask_shape": list(mask_shape),
        "feature_dtype": "float16",
        "mask_dtype": "bool",
        "max_length": max_length,
        "projection_dim": dimension,
        "invalid_feature_fill": 0.0,
        "invalid_token_mask_fill": False,
        "global_slot_valid": False,
        "aligned_rows": stats["cache_rows_with_foil_row"],
        "valid_per_slot": [int(valid_mask[:, slot].sum()) for slot in range(6)],
        "encoded_captions": len(encode_items),
        "true_token_count": true_token_count,
        "shared_valid_mask": os.path.abspath(shared_valid_path),
        "shared_valid_mask_action": valid_mask_action,
        "encode_seconds": elapsed,
    }
    _atomic_write_json(output_paths["text_foil_token_meta.json"], metadata)
    print(
        f"[foil-token] wrote token sidecars in {args.out_dir}; "
        f"captions={len(encode_items)}, true_tokens={true_token_count}, "
        f"elapsed={elapsed:.1f}s"
    )
    print(f"[foil-token] shared valid mask: {valid_mask_action}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
