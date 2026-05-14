"""SigLIP2-based VisualEncoder.

Role:
    Run the vision tower of a (shared) SigLIP2 backbone on already-preprocessed
    pixel_values and return BOTH:
      - the *aligned* global embedding from the projection head
        (``model.get_image_features(...)``), for retrieval / hashing
      - the raw patch-level sequence (``vision_model(...).last_hidden_state``),
        for future codon / OT / clustering modules

IMPORTANT:
    This module does NOT preprocess images. Upstream code must produce
    model-compatible ``pixel_values`` (correct resize + SigLIP2-specific
    normalization), e.g. via ``Siglip2ImageProcessor``.

Note on tokens:
    SigLIP / SigLIP2 do NOT prepend a [CLS] token. Every position in
    ``token_feat`` is a patch token, so no special-token slicing is needed.
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


class VisualEncoder(nn.Module):
    """SigLIP2 vision tower wrapper.

    Output of forward() is a dict with the following keys
    (None when not requested):

        global_feat       [B, D]       aligned embedding (projection head output).
                                       L2-normalized when ``normalize=True``.
        token_feat        [B, N, H_v]  patch-level last_hidden_state of the
                                       vision tower. NOT projected. NO [CLS].
                                       None when ``return_tokens=False``.
        pooled_feat       [B, H_v]     vision tower's pooler_output (raw, NOT
                                       the aligned global embedding). None if
                                       the tower does not expose one or tokens
                                       were not requested.
        hidden_states     tuple([B, N, H_v]) per-layer hidden states. None
                                              unless ``output_hidden_states=True``.
        patch_mask        [B, N] or None     forwarded as-is.
    """

    def __init__(
        self,
        pretrained_name: str = DEFAULT_BACKBONE,
        backbone: Optional[SigLIP2Backbone] = None,
    ) -> None:
        super().__init__()
        if backbone is None:
            backbone = build_pretrained_backbone(pretrained_name)
        self.backbone = backbone
        self.hidden_dim: int = self.backbone.vision_hidden_dim
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
        pixel_values: torch.FloatTensor,                       # [B, C, H, W]
        pixel_attention_mask: Optional[torch.Tensor] = None,   # [B, N] or None (Naflex variant)
        spatial_shapes: Optional[torch.Tensor] = None,         # [B, 2] or None (Naflex variant)
        output_hidden_states: bool = False,
        normalize: bool = True,
        return_tokens: bool = True,
    ) -> Dict[str, Any]:
        # The base SigLIP2 model takes only ``pixel_values``; the Naflex variant
        # additionally accepts pixel_attention_mask / spatial_shapes. Forward them
        # only when supplied so this works for both checkpoints.
        kw: Dict[str, torch.Tensor] = {"pixel_values": pixel_values}
        if pixel_attention_mask is not None:
            kw["pixel_attention_mask"] = pixel_attention_mask
        if spatial_shapes is not None:
            kw["spatial_shapes"] = spatial_shapes

        # 1) aligned global embedding via the projection head.
        #    NOTE: must come from get_image_features (NOT from raw vision_model
        #    last_hidden_state), per the project's design rule.
        # Newer transformers versions return a ModelOutput here instead of a
        # raw tensor — coerce defensively.
        global_feat = self.backbone.model.get_image_features(**kw)
        global_feat = coerce_pooled_to_tensor(global_feat)        # [B, D]
        if normalize:
            global_feat = F.normalize(global_feat, dim=-1)

        # 2) optionally run the vision tower to expose patch-level outputs.
        token_feat: Optional[torch.Tensor] = None
        pooled_feat: Optional[torch.Tensor] = None
        hidden_states = None
        if return_tokens or output_hidden_states:
            vm_kwargs = dict(kw)
            vm_kwargs["output_hidden_states"] = output_hidden_states
            vision_outputs = self.backbone.vision_model(**vm_kwargs)
            if return_tokens:
                token_feat = vision_outputs.last_hidden_state                # [B, N, H_v]
            pooled_feat = getattr(vision_outputs, "pooler_output", None)     # [B, H_v] or None
            if output_hidden_states:
                hidden_states = getattr(vision_outputs, "hidden_states", None)

        return {
            "global_feat": global_feat,
            "token_feat": token_feat,
            "pooled_feat": pooled_feat,
            "hidden_states": hidden_states,
            "patch_mask": pixel_attention_mask,
        }


def build_visual_encoder(
    cfg_or_none: Any = None,
    backbone: Optional[SigLIP2Backbone] = None,
    **kwargs: Any,
) -> VisualEncoder:
    """Build a VisualEncoder using config fallbacks.

    Resolution order for the pretrained name:
        kwargs['pretrained_name'] > cfg.visual_backbone_name >
        cfg.vision_backbone_name > cfg.siglip2_backbone > DEFAULT_BACKBONE
    """
    name = kwargs.pop("pretrained_name", None)
    if name is None and cfg_or_none is not None:
        name = (
            getattr(cfg_or_none, "visual_backbone_name", None)
            or getattr(cfg_or_none, "vision_backbone_name", None)
            or getattr(cfg_or_none, "siglip2_backbone", None)
        )
    if name is None:
        name = DEFAULT_BACKBONE
    return VisualEncoder(pretrained_name=name, backbone=backbone, **kwargs)
