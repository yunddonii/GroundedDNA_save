"""SigLIP2 tokenization utilities for codebook texts.

Bridges the offline Qwen2.5-VL JSON cache (see `vlm_qwen25_descriptions.py`
and `preprocess_qwen_codebook_texts.py`) and the runtime dataloader.

IMPORTANT:
    The tokenizer here is the SigLIP2 tokenizer — NOT Qwen's.
    Qwen2.5-VL is offline-only; the SigLIP2 text encoder is what consumes
    `part_input_ids` during training/inference.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union

import torch


# Order MUST match `model_siglip2.PART_ORDER` and
# `vlm_qwen25_descriptions.CODEBOOK_KEYS`.
CODEBOOK_TEXT_KEYS: Tuple[str, ...] = (
    "C_global",
    "C_head_or_main_part",
    "C_body_or_secondary_part",
    "C_limb_or_detail_part",
    "C_color_texture",
    "C_background_null",
)
# V2 (Option C, scene-aware) keyset. Same positional order so downstream
# routing/PART_ORDER stays compatible; the prompt and per-slot semantics change.
CODEBOOK_TEXT_KEYS_V2: Tuple[str, ...] = (
    "C_global",
    "C_primary_object",
    "C_secondary_object",
    "C_activity_or_relation",
    "C_color_texture",
    "C_scene_type",
)

DEFAULT_SIGLIP2_TOKENIZER_NAME = "google/siglip2-base-patch16-224"
# Companion default for the CLIP backbone (openai/clip-vit-base-patch16).
# The OpenAI CLIP tokenizer ships with the same checkpoint name as the model.
DEFAULT_CLIP_TOKENIZER_NAME    = "openai/clip-vit-base-patch16"
DEFAULT_FALLBACK_TEXT = "none"


def _detect_codebook_schema(cb: Dict[str, Any]) -> Tuple[str, ...]:
    """Return whichever of V1 / V2 codebook-key schemas a parsed entry uses.

    Auto-detect lets downstream callers consume V1 and V2 JSONL caches
    interchangeably without a CLI flag. We do not mix schemas within a single
    entry -- whichever distinctive key (V2's ``C_primary_object`` /
    V1's ``C_head_or_main_part``) is present wins.
    """
    if "C_primary_object" in cb or "C_scene_type" in cb:
        return CODEBOOK_TEXT_KEYS_V2
    return CODEBOOK_TEXT_KEYS


# ----------------------------------------------------------------- extractors

def extract_codebook_texts(description_json: Dict[str, Any]) -> List[str]:
    """Return the six codebook texts in fixed order from a parsed Qwen JSON.

    Accepts either:
      - {"codebook_texts": {C_global: ..., ...}}    (full Qwen JSON)
      - {C_global: ..., ...}                         (already-extracted dict)

    Missing / non-string / blank entries are coerced to ``DEFAULT_FALLBACK_TEXT``.

    Output shape concept: ``[6]`` (list of 6 strings).
    """
    if "codebook_texts" in description_json and isinstance(
        description_json["codebook_texts"], dict
    ):
        cb = description_json["codebook_texts"]
    else:
        cb = description_json
    keys = _detect_codebook_schema(cb)
    out: List[str] = []
    for k in keys:
        v = cb.get(k, "")
        if v is None:
            v = ""
        if not isinstance(v, str):
            v = str(v)
        v = v.strip()
        out.append(v if v else DEFAULT_FALLBACK_TEXT)
    if len(out) != 6:
        raise ValueError(
            f"[extract_codebook_texts] expected 6 keys ({keys}), "
            f"got {len(out)} entries"
        )
    return out


def batch_extract_codebook_texts(
    description_jsons: List[Dict[str, Any]],
) -> List[List[str]]:
    """Output shape concept: ``[B, 6]``."""
    return [extract_codebook_texts(j) for j in description_jsons]


# ----------------------------------------------------------------- tokenizer

def build_siglip2_text_tokenizer(name: str = DEFAULT_SIGLIP2_TOKENIZER_NAME):
    """Lazy-import SigLIP2 tokenizer."""
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "transformers is required to load the SigLIP2 tokenizer. "
            "Install via `pip install 'transformers>=4.45'`."
        ) from e
    return AutoTokenizer.from_pretrained(name)


def build_clip_text_tokenizer(name: str = DEFAULT_CLIP_TOKENIZER_NAME):
    """Lazy-import the OpenAI-CLIP tokenizer.

    Companion to ``build_siglip2_text_tokenizer`` for the CLIP backbone path.
    The downstream `tokenize_codebook_texts` helper is tokenizer-agnostic
    -- it relies only on the standard ``__call__`` signature and on
    ``pad_token_id``.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "transformers is required to load the CLIP tokenizer. "
            "Install via `pip install 'transformers>=4.45'`."
        ) from e
    return AutoTokenizer.from_pretrained(name)


def tokenize_codebook_texts(
    batch_texts: List[List[str]],
    tokenizer,
    max_length: int = 64,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """Tokenize ``[B, 6]`` texts into ``[B, 6, L]`` tensors.

    Returns:
        {
          "part_input_ids":      LongTensor [B, 6, L],
          "part_attention_mask": Tensor     [B, 6, L],
          "flat_texts":          List[str]  (length B*6, row-major),
        }
    """
    if not batch_texts:
        raise ValueError("[tokenize_codebook_texts] empty batch.")
    M = len(batch_texts[0])
    if any(len(row) != M for row in batch_texts):
        raise ValueError(
            f"[tokenize_codebook_texts] inconsistent row sizes; expected {M} per row."
        )
    flat_texts: List[str] = [t for row in batch_texts for t in row]   # [B*M]

    enc = tokenizer(
        flat_texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_attention_mask=True,        # SigLIP2 tokenizer omits this by default
    )
    B = len(batch_texts)
    L = enc["input_ids"].shape[-1]

    # SigLIP2's tokenizer has ``model_input_names=['input_ids']`` so it does
    # NOT include ``attention_mask`` by default; reconstruct from pad_token_id
    # when missing so callers don't crash with KeyError.
    if "attention_mask" in enc:
        attn = enc["attention_mask"]
    else:
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is None:
            attn = torch.ones_like(enc["input_ids"])
        else:
            attn = (enc["input_ids"] != pad_id).long()

    part_input_ids      = enc["input_ids"].view(B, M, L)
    part_attention_mask = attn.view(B, M, L)
    if device is not None:
        part_input_ids      = part_input_ids.to(device)
        part_attention_mask = part_attention_mask.to(device)

    return {
        "part_input_ids":      part_input_ids,
        "part_attention_mask": part_attention_mask,
        "flat_texts":          flat_texts,
    }
