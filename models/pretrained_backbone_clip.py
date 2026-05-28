"""OpenAI-CLIP dual-encoder pretrained backbone.

Mirrors ``models/pretrained_backbone.SigLIP2Backbone`` so the downstream model
code (``VisualEncoder`` / ``TextEncoder`` / ``SigLIP2SemanticOTModel``) can
treat the two backbones interchangeably. The two encoders held by the parent
model both keep a reference to the SAME ``CLIPBackbone`` instance.

Why a parallel module (not a branch inside `pretrained_backbone.py`):
    The two HuggingFace checkpoints come from different model families
    (Siglip2Model vs CLIPModel). Their `vision_model` / `text_model`
    sub-modules expose different attributes (e.g. SigLIP2 uses MAP-pool, CLIP
    uses [CLS] pooling; their `last_hidden_state` shapes differ by the CLS
    token). Isolating them in separate files prevents one model's quirks from
    leaking into the other's path, and matches the project's "new file
    centric" change policy.

What this module does NOT do:
    - tokenize text (caller supplies ``input_ids`` / ``attention_mask``
      produced by the CLIP tokenizer)
    - preprocess images (caller supplies CLIP-normalized ``pixel_values``;
      the extraction script applies the CLIP-specific mean/std).
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn


DEFAULT_CLIP_BACKBONE = "openai/clip-vit-base-patch16"


class CLIPBackbone(nn.Module):
    """Thin holder around ``transformers.CLIPModel``.

    API parity with ``SigLIP2Backbone``:
        self.model              the full CLIPModel (vision + text + 2 projections)
        self.vision_hidden_dim  H_v -- vision_model last_hidden_state dim
        self.text_hidden_dim    H_t -- text_model last_hidden_state dim
        self.projection_dim     D   -- shared image/text embedding dim returned
                                       by get_image_features / get_text_features

    For ``openai/clip-vit-base-patch16``:
        H_v = 768, H_t = 512, D = 512
    """

    def __init__(self, pretrained_name: str = DEFAULT_CLIP_BACKBONE) -> None:
        super().__init__()
        try:
            from transformers import CLIPModel
        except ImportError as e:
            raise ImportError(
                "transformers is required to load CLIP. "
                "Install via `pip install 'transformers>=4.45'`."
            ) from e

        self.pretrained_name = pretrained_name
        self.model = CLIPModel.from_pretrained(pretrained_name)

        cfg = self.model.config
        self.vision_hidden_dim: int = int(cfg.vision_config.hidden_size)
        self.text_hidden_dim:   int = int(cfg.text_config.hidden_size)
        self.projection_dim:    int = int(cfg.projection_dim)

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


def build_clip_backbone(pretrained_name: Optional[str] = None) -> CLIPBackbone:
    """Build a shared CLIP backbone. Pass the same instance into both encoders."""
    if pretrained_name is None:
        pretrained_name = DEFAULT_CLIP_BACKBONE
    return CLIPBackbone(pretrained_name=pretrained_name)
