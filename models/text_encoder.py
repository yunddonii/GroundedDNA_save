"""SigLIP2-based TextEncoder.

Role:
    Run the text tower of a (shared) SigLIP2 backbone on already-tokenized
    inputs and return BOTH:
      - the *aligned* global embedding from the projection head
        (``model.get_text_features(...)``), for retrieval / hashing
      - the raw token-level sequence (``text_model(...).last_hidden_state``),
        for future codon / OT / clustering modules

IMPORTANT:
    This module does NOT tokenize. Upstream code must produce model-compatible
    ``input_ids`` (and optionally ``attention_mask``) using the SigLIP2 tokenizer.
"""

from __future__ import annotations
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pretrained_backbone import (
    SigLIP2Backbone,
    build_pretrained_backbone,
    coerce_pooled_to_tensor,
    DEFAULT_BACKBONE,
)


class TextEncoder(nn.Module):
    """SigLIP2 text tower wrapper.

    Output of forward() is a dict with the following keys
    (None when not requested):

        global_feat       [B, D]       aligned embedding (projection head output).
                                       L2-normalized when ``normalize=True``.
        token_feat        [B, L, H_t]  token-level last_hidden_state of the text
                                       tower. NOT projected. None when
                                       ``return_tokens=False``.
        pooled_feat       [B, H_t]     text tower's pooler_output (raw, NOT the
                                       aligned global embedding). None if the
                                       tower does not expose one or tokens were
                                       not requested.
        hidden_states     tuple([B, L, H_t]) per-layer hidden states. None
                                              unless ``output_hidden_states=True``.
        attention_mask    [B, L] or None     forwarded as-is.
    """

    def __init__(
        self,
        pretrained_name: str = DEFAULT_BACKBONE,
        backbone: Optional[SigLIP2Backbone] = None,
    ) -> None:
        super().__init__()
        # share the dual-encoder if one was provided; otherwise build our own
        if backbone is None:
            backbone = build_pretrained_backbone(pretrained_name)
        self.backbone = backbone
        self.hidden_dim: int = self.backbone.text_hidden_dim
        self.projection_dim: int = self.backbone.projection_dim

    # ------------------------------------------------------------- utilities

    def get_output_dim(self) -> int:
        """Dim of ``global_feat`` (the aligned embedding D)."""
        return self.projection_dim

    def freeze(self) -> None:
        for p in self.backbone.model.parameters():
            p.requires_grad = False

    def unfreeze(self) -> None:
        for p in self.backbone.model.parameters():
            p.requires_grad = True

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # --------------------------------------------------------------- forward

    def forward(
        self,
        input_ids: torch.LongTensor,                     # [B, L]
        attention_mask: Optional[torch.Tensor] = None,   # [B, L] or None
        output_hidden_states: bool = False,
        normalize: bool = True,
        return_tokens: bool = True,
    ) -> Dict[str, Any]:
        # 1) aligned global embedding via the projection head.
        #    NOTE: must come from get_text_features (NOT from raw text_model
        #    last_hidden_state), per the project's design rule.
        # Newer transformers versions return a ModelOutput here instead of a
        # raw tensor — coerce defensively.
        global_feat = self.backbone.model.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        global_feat = coerce_pooled_to_tensor(global_feat)        # [B, D]
        if normalize:
            global_feat = F.normalize(global_feat, dim=-1)

        # 2) optionally run the text tower again to expose token-level outputs.
        #    Only paid when the caller asks for them.
        token_feat: Optional[torch.Tensor] = None
        pooled_feat: Optional[torch.Tensor] = None
        hidden_states = None
        if return_tokens or output_hidden_states:
            text_outputs = self.backbone.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=output_hidden_states,
            )
            if return_tokens:
                token_feat = text_outputs.last_hidden_state              # [B, L, H_t]
            pooled_feat = getattr(text_outputs, "pooler_output", None)   # [B, H_t] or None
            if output_hidden_states:
                hidden_states = getattr(text_outputs, "hidden_states", None)

        return {
            "global_feat": global_feat,
            "token_feat": token_feat,
            "pooled_feat": pooled_feat,
            "hidden_states": hidden_states,
            "attention_mask": attention_mask,
        }


def build_text_encoder(
    cfg_or_none: Any = None,
    backbone: Optional[SigLIP2Backbone] = None,
    **kwargs: Any,
) -> TextEncoder:
    """Build a TextEncoder using config fallbacks.

    Resolution order for the pretrained name:
        kwargs['pretrained_name'] > cfg.text_backbone_name >
        cfg.siglip2_backbone     > DEFAULT_BACKBONE
    """
    name = kwargs.pop("pretrained_name", None)
    if name is None and cfg_or_none is not None:
        name = (
            getattr(cfg_or_none, "text_backbone_name", None)
            or getattr(cfg_or_none, "siglip2_backbone", None)
        )
    if name is None:
        name = DEFAULT_BACKBONE
    return TextEncoder(pretrained_name=name, backbone=backbone, **kwargs)
