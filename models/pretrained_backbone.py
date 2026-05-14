"""SigLIP2 dual-encoder pretrained backbone (shared between TextEncoder and VisualEncoder).

Why a shared backbone:
    SigLIP2's HF checkpoint ships the vision tower, the text tower, AND both
    projection heads inside a single ``Siglip2Model`` object. If TextEncoder and
    VisualEncoder each call ``AutoModel.from_pretrained(...)``, we end up with
    two copies of the same weights on GPU. This module loads it ONCE; the two
    encoders just hold a reference to the same backbone instance.

What this module does NOT do:
    - tokenize text (callers must supply ``input_ids`` / ``attention_mask``)
    - preprocess images (callers must supply already-normalized ``pixel_values``)
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn


DEFAULT_BACKBONE = "google/siglip2-base-patch16-224"


class SigLIP2Backbone(nn.Module):
    """Thin holder around ``transformers.AutoModel`` for SigLIP2.

    Members:
        self.model              the full Siglip2Model (vision + text + projections)
        self.vision_hidden_dim  H_v -- last_hidden_state dim of vision tower
        self.text_hidden_dim    H_t -- last_hidden_state dim of text tower
        self.projection_dim     D   -- aligned embedding dim returned by
                                       get_image_features / get_text_features
    """

    def __init__(self, pretrained_name: str = DEFAULT_BACKBONE) -> None:
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as e:
            raise ImportError(
                "transformers is required to load SigLIP2. "
                "Install via `pip install 'transformers>=4.45'`."
            ) from e

        self.pretrained_name = pretrained_name
        self.model = AutoModel.from_pretrained(pretrained_name)

        cfg = self.model.config
        self.vision_hidden_dim: int = int(cfg.vision_config.hidden_size)
        self.text_hidden_dim: int = int(cfg.text_config.hidden_size)
        self.projection_dim: int = int(getattr(cfg, "projection_dim", self.vision_hidden_dim))

    # convenience accessors so encoders can stay readable
    @property
    def vision_model(self) -> nn.Module:
        return self.model.vision_model

    @property
    def text_model(self) -> nn.Module:
        return self.model.text_model

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device


def build_pretrained_backbone(pretrained_name: Optional[str] = None) -> SigLIP2Backbone:
    """Build a shared SigLIP2 backbone. Pass the same instance into both encoders."""
    if pretrained_name is None:
        pretrained_name = DEFAULT_BACKBONE
    return SigLIP2Backbone(pretrained_name=pretrained_name)


def coerce_pooled_to_tensor(x) -> torch.Tensor:
    """Normalize the return value of ``get_text_features`` / ``get_image_features``.

    Older transformers versions returned a raw pooled tensor; newer ones can
    return a ``BaseModelOutputWithPooling`` (or similar ModelOutput) instead.
    Both encoders use this to always end up with a [B, D] ``torch.Tensor``.

    Resolution order:
        1. tensor             → return as-is
        2. ``.pooler_output`` → preferred (post-projection pooled embedding)
        3. ``.text_embeds`` / ``.image_embeds`` (some VLM heads use these)
        4. ``.last_hidden_state`` → last resort; usually NOT pooled
        5. tuple-like         → index 1 if available else index 0
        6. otherwise raise TypeError with a clear message
    """
    if isinstance(x, torch.Tensor):
        return x
    for attr in ("pooler_output", "text_embeds", "image_embeds", "embeds"):
        v = getattr(x, attr, None)
        if isinstance(v, torch.Tensor):
            return v
    lhs = getattr(x, "last_hidden_state", None)
    if isinstance(lhs, torch.Tensor):
        return lhs
    if isinstance(x, (tuple, list)) and len(x) > 0:
        return x[1] if len(x) > 1 and isinstance(x[1], torch.Tensor) else x[0]
    raise TypeError(
        f"[coerce_pooled_to_tensor] cannot extract a pooled tensor from "
        f"{type(x).__name__}; available attrs={list(getattr(x, '__dict__', {}).keys())}"
    )
