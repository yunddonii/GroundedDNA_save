"""SigLIP2 + text-guided semantic OT routing model + per-part codebooks.

Skeleton mirroring `model.py`'s style:
    - constructor takes ``args`` (Config-like; getattr fallbacks for new keys)
    - holds ``self.backbone`` (the shared SigLIP2 dual-encoder)
    - exposes ``feature_extraction`` and ``forward`` returning explicit dicts

C_global vs the 5 local parts (KEY DESIGN POINT):
    The 6 semantic parts are split into one *global* slot and 5 *local* slots.
        index 0 = C_global    -- handled by global average pooling over all
                                 visual tokens (NOT through Sinkhorn OT).
        index 1..5 = local    -- Sinkhorn OT routing, with text centroids
                                 (training, when text is supplied) OR codebook
                                 mean anchors (inference / when text absent).
    The final ``semantic_visual_tokens`` ([B, 6, D]) is therefore
        [global_visual_token | local_semantic_visual_tokens]
    where ``local_semantic_visual_tokens`` is [B, 5, D].

Per-part codebooks (this stage):
    There are 6 INDEPENDENT codebooks, one per semantic part. The shared
    parameter tensor has shape [M=6, K, D]. ``semantic_visual_tokens[:, m, :]``
    is quantized in ``codebooks[m]`` by squared-L2 nearest-neighbour. We expose
    quantization-loss-friendly tensors (raw + STE quantized + indices +
    distances) but do NOT add any quantization loss to the training objective
    in this stage.

Routing-mode rule (forward()):
    if self.training and part_input_ids is not None:
        routing_mode = "text"            -> text-guided OT routing
    else:
        routing_mode = "codebook_mean"   -> codebook mean anchor OT routing

What's OUT of scope this stage (do NOT add here):
    codon head, tanh/sign hash head, DNA encoding, Hamming loss,
    quantization loss in the training objective, balance / uncorrelated loss,
    positive consistency loss, retrieval evaluation rewrite.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.pretrained_backbone import build_pretrained_backbone, DEFAULT_BACKBONE
from models.pretrained_backbone_clip import (
    build_clip_backbone,
    DEFAULT_CLIP_BACKBONE,
)
from models.visual_encoder import VisualEncoder
from models.text_encoder import TextEncoder
from models.adapters import (
    VisualAdapter, TextAdapter,
    VisualLinearAdapter, TextLinearAdapter,
)
from models.semantic_router import (
    SemanticSinkhornRouter,
    SemanticAttentionRouter,
    SemanticSlotAttentionRouter,
)


# ----------------------------------------------------------------- constants

# Order is fixed by the project spec — DO NOT reorder.
# Must match `vlm_qwen25_descriptions.CODEBOOK_KEYS`.
# NB: these names are LEGACY labels from the part-based schema. The pipeline is
# positional -- the v4 caption cache stores [N, 6, D] in the order
#   C_global, C_primary_object, C_secondary_object, C_activity_or_relation,
#   C_color_texture, C_scene_type
# and only the INDEX matters. (This mismatch is also why the routing heatmaps
# are labelled C_head/C_body/C_limb.)
_FULL_PART_ORDER: Tuple[str, ...] = (
    "C_global",                  # index 0  -- global average pooling
    "C_head_or_main_part",       # index 1  ┐
    "C_body_or_secondary_part",  # index 2  │
    "C_limb_or_detail_part",     # index 3  ├─ Sinkhorn OT routing (local)
    "C_color_texture",           # index 4  │
    "C_background_null",         # index 5  ┘  = scene_type in the v4 cache
)

# Slot count is a STRUCTURAL constant read at import time, because 99 call sites
# in this file reference it at module scope and argparse has not run yet when
# this module is imported (train_siglip2.py imports config at line 40 and this
# module at line 42, but Config.get_config() runs later inside main()).
# An env var is therefore the only mechanism that can reach them all.
#
# Truncation keeps the FIRST n slots, and `scene_type` is index 5, so
# GDNA_NUM_SEMANTIC_PARTS=5 drops exactly the slot the drop-ablation identified
# as free: deleting it IMPROVES held-out codon decoding on 4/4 datasets
# (Flickr +.0059, MS-COCO +.0144, NUS +.0117, CIFAR +.0111) and it is the
# smallest retrieval contributor on CIFAR (-.0089).
#
# `--num_semantic_parts` mirrors this into args.txt and is cross-checked at
# model construction, so a mismatch between the env var and the recorded config
# fails loudly instead of silently producing an unreproducible run.
_N_PARTS = int(os.environ.get("GDNA_NUM_SEMANTIC_PARTS", str(len(_FULL_PART_ORDER))))
if not (2 <= _N_PARTS <= len(_FULL_PART_ORDER)):
    raise ValueError(
        f"GDNA_NUM_SEMANTIC_PARTS must be in [2, {len(_FULL_PART_ORDER)}], got {_N_PARTS}")
PART_ORDER: Tuple[str, ...] = _FULL_PART_ORDER[:_N_PARTS]
NUM_SEMANTIC_PARTS = len(PART_ORDER)        # 6 by default
LOCAL_PART_ORDER: Tuple[str, ...] = PART_ORDER[1:]
NUM_LOCAL_PARTS = len(LOCAL_PART_ORDER)     # 5 by default


def _cls_verified_consensus_mask(
    visual_tokens_shared: torch.Tensor,
    local_text_shared: torch.Tensor,
    visual_global_shared: torch.Tensor,
    visual_mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Mask text-common patches only when they are globally weak.

    Every comparison is centered within the current image, so this rule has
    no absolute cosine threshold. The returned masks are detached because
    they select OT support rather than define a differentiable objective.

    Shapes:
        visual_tokens_shared: [B, N, D_shared]
        local_text_shared:    [B, M_local, D_shared]
        visual_global_shared: [B, D_shared]
        visual_mask:          [B, N] bool, or None
    """
    assert visual_tokens_shared.dim() == 3
    assert local_text_shared.dim() == 3
    assert visual_global_shared.dim() == 2
    B, N, D_shared = visual_tokens_shared.shape
    Bt, M_local, Dt = local_text_shared.shape
    assert (Bt, Dt) == (B, D_shared), (
        "local_text_shared must align with visual tokens, got "
        f"{tuple(local_text_shared.shape)} vs {tuple(visual_tokens_shared.shape)}"
    )
    assert M_local == NUM_LOCAL_PARTS
    assert visual_global_shared.shape == (B, D_shared)
    if visual_mask is None:
        visual_valid = torch.ones(
            B, N, dtype=torch.bool, device=visual_tokens_shared.device,
        )
    else:
        assert visual_mask.shape == (B, N)
        visual_valid = visual_mask.to(
            device=visual_tokens_shared.device, dtype=torch.bool,
        )

    visual_n = F.normalize(visual_tokens_shared, dim=-1)             # [B, N, D_shared]
    text_n = F.normalize(local_text_shared, dim=-1)                  # [B, M_local, D_shared]
    global_n = F.normalize(visual_global_shared, dim=-1)             # [B, D_shared]
    slot_similarity = torch.einsum(
        "bnd,bmd->bnm", visual_n, text_n,
    )                                                                # [B, N, M_local]
    valid_f = visual_valid.to(slot_similarity.dtype)                 # [B, N]
    valid_count = valid_f.sum(dim=-1, keepdim=True).clamp_min(1.0)   # [B, 1]
    slot_mean = (
        (slot_similarity * valid_f.unsqueeze(-1)).sum(dim=1)
        / valid_count
    )                                                                # [B, M_local]
    common_candidate = (
        (slot_similarity > slot_mean.unsqueeze(1)).all(dim=-1)
        & visual_valid
    )                                                                # [B, N]

    global_similarity = torch.einsum(
        "bnd,bd->bn", visual_n, global_n,
    )                                                                # [B, N]
    global_mean = (
        (global_similarity * valid_f).sum(dim=-1, keepdim=True)
        / valid_count
    )                                                                # [B, 1]
    global_low = (global_similarity < global_mean) & visual_valid    # [B, N]
    prune = (common_candidate & global_low).detach()                 # [B, N]
    remaining = visual_valid & ~prune                               # [B, N]
    fallback = remaining.sum(dim=-1, keepdim=True) == 0              # [B, 1]
    keep = torch.where(fallback, visual_valid, remaining).detach()   # [B, N]

    valid_count_safe = visual_valid.sum(dim=-1).clamp_min(1).to(slot_similarity.dtype)
    return {
        "keep": keep,
        "common_candidate": common_candidate.detach(),
        "global_low": global_low.detach(),
        "prune": prune,
        "common_candidate_ratio": (
            common_candidate.sum(dim=-1).to(slot_similarity.dtype) / valid_count_safe
        ),                                                           # [B]
        "global_low_ratio": (
            global_low.sum(dim=-1).to(slot_similarity.dtype) / valid_count_safe
        ),                                                           # [B]
        "mask_ratio": (
            prune.sum(dim=-1).to(slot_similarity.dtype) / valid_count_safe
        ),                                                           # [B]
        "remaining_ratio": (
            keep.sum(dim=-1).to(slot_similarity.dtype) / valid_count_safe
        ),                                                           # [B]
        "fallback": fallback.squeeze(-1).to(slot_similarity.dtype), # [B]
    }


def _slot_consensus_residual_importance(
    importance: torch.Tensor,
    visual_mask: Optional[torch.Tensor] = None,
    slot_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Remove evidence shared uniformly by all valid local slots.

    ``importance`` is first normalized over visual tokens per slot. The
    geometric-mean distribution across slots is then treated as consensus,
    and only the positive pointwise information above that consensus remains.

    Shapes:
        importance: [B, M_local, N], non-negative
        visual_mask: [B, N] bool, or None
        slot_mask:   [B, M_local] bool, or None
    """
    assert importance.dim() == 3, (
        f"importance must be [B, M_local, N], got {tuple(importance.shape)}"
    )
    B, M_local, N = importance.shape
    if visual_mask is None:
        visual_valid = torch.ones(
            B, N, dtype=torch.bool, device=importance.device,
        )
    else:
        assert visual_mask.shape == (B, N), (
            f"visual_mask must be {(B, N)}, got {tuple(visual_mask.shape)}"
        )
        visual_valid = visual_mask.to(device=importance.device, dtype=torch.bool)
    if slot_mask is None:
        slot_valid = torch.ones(
            B, M_local, dtype=torch.bool, device=importance.device,
        )
    else:
        assert slot_mask.shape == (B, M_local), (
            f"slot_mask must be {(B, M_local)}, got {tuple(slot_mask.shape)}"
        )
        slot_valid = slot_mask.to(device=importance.device, dtype=torch.bool)

    valid = visual_valid[:, None, :] & slot_valid[:, :, None]       # [B, M_local, N]
    positive = importance.clamp_min(0.0) * valid.to(importance.dtype)
    distribution = positive / positive.sum(dim=-1, keepdim=True).clamp_min(eps)

    # log G_n = mean_m log P_mn over valid slots. A visual token receives a
    # positive residual for slot m only when P_mn exceeds this consensus.
    log_distribution = distribution.clamp_min(eps).log()            # [B, M_local, N]
    slot_weight = slot_valid.to(importance.dtype).unsqueeze(-1)      # [B, M_local, 1]
    log_consensus = (
        (log_distribution * slot_weight).sum(dim=1, keepdim=True)
        / slot_weight.sum(dim=1, keepdim=True).clamp_min(1.0)
    )                                                               # [B, 1, N]
    residual = distribution * F.relu(log_distribution - log_consensus)
    residual = residual * valid.to(residual.dtype)                  # [B, M_local, N]

    # Exact equality across slots makes every residual zero. In that
    # degenerate case, preserve the original normalized ranking so top-k
    # selection remains well-defined rather than depending on argsort ties.
    degenerate = residual.sum(dim=-1, keepdim=True) <= eps           # [B, M_local, 1]
    residual = torch.where(degenerate & slot_valid.unsqueeze(-1), distribution, residual)
    assert residual.shape == (B, M_local, N)
    return residual


def _topk_visual_keep(
    importance: torch.Tensor,
    visual_mask: Optional[torch.Tensor],
    keep_ratio: float,
) -> torch.Tensor:
    """Return an exact-count slot-specific visual keep mask."""
    assert importance.dim() == 3
    B, M_local, N = importance.shape
    if not (0.0 < float(keep_ratio) <= 1.0):
        raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")
    if visual_mask is None:
        visual_valid = torch.ones(B, N, dtype=torch.bool, device=importance.device)
    else:
        assert visual_mask.shape == (B, N)
        visual_valid = visual_mask.to(device=importance.device, dtype=torch.bool)
    ranked = importance.masked_fill(~visual_valid[:, None, :], -float("inf"))
    order = ranked.argsort(dim=-1, descending=True)
    rank = order.argsort(dim=-1)
    valid_count = visual_valid.sum(dim=-1).clamp_min(1)              # [B]
    k = torch.ceil(
        valid_count.to(torch.float32) * float(keep_ratio)
    ).to(torch.long).clamp(min=1, max=N)                             # [B]
    keep = rank < k[:, None, None]                                   # [B, M_local, N]
    keep = keep & visual_valid[:, None, :]
    assert keep.shape == (B, M_local, N)
    return keep


def _mutual_dual_softmax_pruning(
    similarity: torch.Tensor,
    text_mask: Optional[torch.Tensor],
    visual_mask: Optional[torch.Tensor],
    visual_keep_ratio: float,
    text_keep_ratio: float,
    suppress_visual_consensus: bool = False,
) -> Dict[str, torch.Tensor]:
    """Build slot-specific visual/text masks from mutual token matching.

    Shapes:
        similarity:  [B, M_local, N, T]
        text_mask:   [B, M_local, T] bool, or None
        visual_mask: [B, N] bool, or None

    Unlike ``softmax(...).sum(softmax_axis)``, the geometric mean of the
    visual-to-text and text-to-visual attentions is not constant. A pair gets
    a high score only when the patch and text token select each other.
    """
    assert similarity.dim() == 4, (
        f"similarity must be [B, M_local, N, T], got {tuple(similarity.shape)}"
    )
    B, M_local, N, T = similarity.shape
    if not (0.0 < float(visual_keep_ratio) <= 1.0):
        raise ValueError(f"visual_keep_ratio must be in (0, 1], got {visual_keep_ratio}")
    if not (0.0 < float(text_keep_ratio) <= 1.0):
        raise ValueError(f"text_keep_ratio must be in (0, 1], got {text_keep_ratio}")

    if text_mask is None:
        text_valid = torch.ones(
            B, M_local, T, dtype=torch.bool, device=similarity.device,
        )
    else:
        assert text_mask.shape == (B, M_local, T), (
            f"text_mask must be {(B, M_local, T)}, got {tuple(text_mask.shape)}"
        )
        text_valid = text_mask.to(device=similarity.device, dtype=torch.bool)
    if visual_mask is None:
        visual_valid = torch.ones(B, N, dtype=torch.bool, device=similarity.device)
    else:
        assert visual_mask.shape == (B, N), (
            f"visual_mask must be {(B, N)}, got {tuple(visual_mask.shape)}"
        )
        visual_valid = visual_mask.to(device=similarity.device, dtype=torch.bool)

    pair_valid = (
        visual_valid[:, None, :, None]
        & text_valid[:, :, None, :]
    )                                                               # [B, M_local, N, T]
    masked_similarity = similarity.masked_fill(~pair_valid, -1e4)
    visual_to_text = masked_similarity.softmax(dim=-1)              # [B, M_local, N, T]
    text_to_visual = masked_similarity.softmax(dim=-2)              # [B, M_local, N, T]
    mutual = torch.sqrt((visual_to_text * text_to_visual).clamp_min(0.0))
    mutual = mutual * pair_valid.to(mutual.dtype)                    # [B, M_local, N, T]

    visual_importance_raw = mutual.sum(dim=-1)                       # [B, M_local, N]
    text_importance = mutual.sum(dim=-2)                             # [B, M_local, T]

    slot_has_text = text_valid.any(dim=-1)                           # [B, M_local]
    visual_importance = (
        _slot_consensus_residual_importance(
            visual_importance_raw,
            visual_mask=visual_valid,
            slot_mask=slot_has_text,
        )
        if suppress_visual_consensus else visual_importance_raw
    )                                                               # [B, M_local, N]

    # Rank/scatter gives an exact keep count even when scores are tied.
    visual_keep = _topk_visual_keep(
        visual_importance, visual_valid, visual_keep_ratio,
    )                                                               # [B, M_local, N]

    # Empty text slots carry no evidence. Keep every valid patch for those
    # slots so pruning cannot create an infeasible Sinkhorn support.
    visual_keep = torch.where(
        slot_has_text.unsqueeze(-1),
        visual_keep,
        visual_valid[:, None, :].expand(-1, M_local, -1),
    )

    text_order = text_importance.argsort(dim=-1, descending=True)
    text_rank = text_order.argsort(dim=-1)
    text_count = text_valid.sum(dim=-1)                              # [B, M_local]
    text_k = torch.ceil(
        text_count.to(torch.float32) * float(text_keep_ratio)
    ).to(torch.long).clamp(min=1, max=T)                             # [B, M_local]
    text_keep = (text_rank < text_k.unsqueeze(-1)) & text_valid      # [B, M_local, T]

    assert visual_keep.shape == (B, M_local, N)
    assert text_keep.shape == (B, M_local, T)
    return {
        "mutual_scores": mutual,
        "visual_importance_raw": visual_importance_raw,
        "visual_importance": visual_importance,
        "text_importance": text_importance,
        "visual_keep": visual_keep,
        "text_keep": text_keep,
    }


# =====================================================================
# Semantic codebook quantizer
# =====================================================================

class SemanticCodebookQuantizer(nn.Module):
    """Per-part vector quantizer with M independent codebooks.

    Parameter:
        self.codebooks  nn.Parameter [M, K, D]
                        codebooks[m, k, :] = the k-th codeword of part m.
                        Initialized as ``Normal(0, 1/sqrt(D))`` (typical VQ
                        initialization scale).

    Forward:
        Input  semantic_visual_tokens : [B, M, D]
        For each (b, m), pick the codeword in codebooks[m] with the smallest
        squared-L2 distance to semantic_visual_tokens[b, m, :].

        Output dict:
            quantized_tokens      [B, M, D]   STE-quantized; gradient passes
                                              through to the encoder. USE THIS
                                              for downstream training paths.
            quantized_tokens_raw  [B, M, D]   Pure codeword embeddings (NO
                                              gradient to encoder; gradient
                                              flows only into the codebook).
            codebook_indices      [B, M]      argmin index per (b, m).
            codebook_distances    [B, M, K]   squared-L2 distances per (b, m).
    """

    def __init__(
        self,
        num_codebooks: int,
        codebook_size: int,
        d_model: int,
        update_mode: str = "gradient",
        ema_decay: float = 0.99,
        ema_eps: float = 1e-5,
        revive_dead: bool = True,
        revive_threshold: float = 0.01,
        revive_every: int = 1,
        repel_strength: float = 0.0,
        repel_sigma_factor: float = 0.5,
        repel_every: int = 1,
        distance_mode: str = "euclidean",
        K_max: int = 0,
        share_codebook: bool = False,
        freeze_after_epoch: int = -1,
    ) -> None:
        super().__init__()
        self.num_codebooks = int(num_codebooks)
        self.codebook_size = int(codebook_size)
        # A4 ablation: tie all M slots to a single shared codebook (slot 0).
        # Tests whether SEPARATE per-slot codebooks are needed. Use with a
        # matched-capacity codebook_size (e.g. 6*128=768) so total codeword
        # count equals the full model.
        self.share_codebook = bool(share_codebook)
        self.d_model       = int(d_model)
        self.update_mode   = str(update_mode)
        if self.update_mode not in ("gradient", "ema"):
            raise ValueError(
                f"[SemanticCodebookQuantizer] update_mode must be "
                f"'gradient' or 'ema', got {self.update_mode!r}"
            )
        self.ema_decay     = float(ema_decay)
        self.ema_eps       = float(ema_eps)
        # Dead-code rejuvenation: codewords with EMA cluster_size below
        # `revive_threshold * max(cluster_size_in_codebook)` are replaced with
        # a random z sampled from the current batch every `revive_every`
        # training forward calls. Only active when EMA mode is on.
        # (A): epoch from which the codebook stops moving. -1 disables the
        # feature entirely. `_frozen_epoch_now` is refreshed by the model's
        # set_current_epoch, because the quantizer has no epoch of its own.
        self.freeze_after_epoch = int(freeze_after_epoch)
        self._frozen_epoch_now  = 0
        # (c-1, 2026-09-20): True during the VQ-free first stage of the
        # two-stage schedule; set by the model from the epoch. The quantizer
        # is then the identity and the codebook is neither read nor updated.
        self.bypass = False
        self.revive_dead       = bool(revive_dead)
        self.revive_threshold  = float(revive_threshold)
        self.revive_every      = max(1, int(revive_every))
        # Option α: Codeword Repulsion. After each EMA update step, push each
        # codeword away from its closest neighbours within the same codebook.
        # `strength` is the step-size as a fraction of the unit repulsion
        # direction; `sigma_factor` controls the Gaussian width relative to
        # the median pairwise distance per codebook. `repel_every` rate-limits
        # the operation. Default strength=0 ⇒ disabled (baseline behaviour).
        self.repel_strength     = float(repel_strength)
        self.repel_sigma_factor = float(repel_sigma_factor)
        self.repel_every        = max(1, int(repel_every))
        # v76: distance metric for codebook nearest-neighbour lookup
        if distance_mode not in ("euclidean", "cosine"):
            raise ValueError(
                f"[SemanticCodebookQuantizer] distance_mode must be "
                f"'euclidean' or 'cosine', got {distance_mode!r}"
            )
        self.distance_mode: str = str(distance_mode)
        # internal step counter (training-only)
        self.register_buffer(
            "_revive_step", torch.zeros((), dtype=torch.long), persistent=False,
        )
        self.register_buffer(
            "_ema_step", torch.zeros((), dtype=torch.long), persistent=False,
        )

        # v78a (adaptive K): K_max ≥ codebook_size. Tensor sized K_max so we
        # can grow the active codeword set via split during training. When
        # K_max == codebook_size (default), the active mask covers all
        # codewords and behaviour is bit-exact identical to the legacy path.
        self.K_max: int = int(K_max) if int(K_max) > 0 else int(self.codebook_size)
        if self.K_max < self.codebook_size:
            raise ValueError(
                f"[SemanticCodebookQuantizer] K_max ({self.K_max}) must be "
                f">= codebook_size ({self.codebook_size})"
            )
        if self.share_codebook and self.K_max != self.codebook_size:
            raise ValueError(
                "[SemanticCodebookQuantizer] share_codebook cannot be combined "
                "with adaptive-K capacity (K_max != codebook_size). The shared "
                "bank has one active-mask/split state, while the adaptive-K "
                "implementation is per-slot."
            )
        # active_mask [M, K_max]: True = codeword participates in lookup.
        # Initial state: first `codebook_size` entries active.
        am = torch.zeros(self.num_codebooks, self.K_max, dtype=torch.bool)
        am[:, :self.codebook_size] = True
        self.register_buffer("active_mask", am)

        # codebooks: [M, K_max, D]
        # init: scaled Normal -- standard VQ init at 1/sqrt(D)
        cb_init = torch.empty(self.num_codebooks, self.K_max, self.d_model)
        nn.init.normal_(cb_init, mean=0.0, std=1.0 / math.sqrt(self.d_model))
        if self.update_mode == "gradient":
            # legacy path: codebook learns through the VQ MSE term in the loss
            self.codebooks = nn.Parameter(cb_init.clone())
        else:
            # EMA path (VQ-VAE-2 / DALL-E style):
            # codebook is a buffer updated in-place every training forward.
            #   cluster_size : [M, K_max]    EMA-smoothed assignment counts
            #   embed_avg    : [M, K_max, D] EMA-smoothed weighted sum of inputs
            #   embed_sqavg  : [M, K_max, D] EMA second-moment (v78a, for variance)
            self.register_buffer("codebooks",    cb_init.clone())
            self.register_buffer("cluster_size", torch.ones(self.num_codebooks, self.K_max))
            self.register_buffer("embed_avg",    cb_init.clone())
            self.register_buffer("embed_sqavg",  cb_init.clone() ** 2)

        # Keep serialized/diagnostic rows coherent too.  Bank 0 is the sole
        # canonical state in the A4 shared-codebook ablation; rows 1..M-1 are
        # compatibility mirrors because historical checkpoints use [M,K,D].
        if self.share_codebook:
            self._sync_shared_state_from_bank0()

    # -------------------------------------------------------- helpers

    def get_effective_codebooks(self) -> torch.Tensor:
        """Codebooks actually used by lookup, losses, and diagnostics.

        A4 keeps the historical ``[M,K,D]`` state-dict shape, but only bank 0
        is a real parameter/buffer.  Returning an expanded view makes every
        consumer observe that same bank and, in gradient mode, accumulates all
        slot gradients into bank 0.
        """
        if self.share_codebook:
            return self.codebooks[0:1].expand(self.num_codebooks, -1, -1)
        return self.codebooks

    def get_effective_active_mask(self) -> torch.Tensor:
        """Active-code mask corresponding to :meth:`get_effective_codebooks`."""
        if self.share_codebook:
            return self.active_mask[0:1].expand(self.num_codebooks, -1)
        return self.active_mask

    @torch.no_grad()
    def _sync_shared_state_from_bank0(self) -> None:
        """Mirror the canonical shared bank into compatibility state rows."""
        if not self.share_codebook or self.num_codebooks <= 1:
            return
        self.codebooks[1:].copy_(
            self.codebooks[0:1].expand(self.num_codebooks - 1, -1, -1)
        )
        self.active_mask[1:].copy_(
            self.active_mask[0:1].expand(self.num_codebooks - 1, -1)
        )
        if self.update_mode == "ema":
            self.cluster_size[1:].copy_(
                self.cluster_size[0:1].expand(self.num_codebooks - 1, -1)
            )
            self.embed_avg[1:].copy_(
                self.embed_avg[0:1].expand(self.num_codebooks - 1, -1, -1)
            )
            self.embed_sqavg[1:].copy_(
                self.embed_sqavg[0:1].expand(self.num_codebooks - 1, -1, -1)
            )

    def _save_to_state_dict(self, destination, prefix, keep_vars) -> None:
        # Gradient-mode optimizers update only canonical bank 0.  Mirror it
        # immediately before serialization so offline checkpoint diagnostics
        # cannot mistake unused compatibility rows for independent banks.
        if self.share_codebook:
            self._sync_shared_state_from_bank0()
        super()._save_to_state_dict(destination, prefix, keep_vars)

    @torch.no_grad()
    def num_trainable_codewords(self) -> int:
        if self.share_codebook:
            return self.codebooks[0].numel()
        return self.codebooks.numel()

    @torch.no_grad()
    def initialize_from_text_anchors(
        self,
        text_anchors: torch.Tensor,
        mode: str = "kmeans",
        seed: int = 42,
    ) -> Dict[str, float]:
        """5-G: replace random Gaussian codebooks with text-derived vectors.

        Args:
            text_anchors: [N, M, D] -- collected `cached_text_part_raw`.
            mode: 'mean' (random K samples) or 'kmeans' (sklearn K-means).
            seed: random / K-means seed.
        Returns:
            diagnostic dict with per-mode timing + per-codebook inertia (kmeans).
        """
        import time
        import numpy as np
        M_cb, K, D = self.codebooks.shape
        N, M_in, D_in = text_anchors.shape
        if M_in != M_cb or D_in != D:
            raise ValueError(
                f"[initialize_from_text_anchors] text_anchors shape "
                f"[N={N}, M={M_in}, D={D_in}] incompatible with codebooks "
                f"[M={M_cb}, K={K}, D={D}]"
            )
        target_std = 1.0 / math.sqrt(self.d_model)
        diag = {
            "mode": mode,
            "N": int(N),
            "K": int(K),
            "M": 1 if self.share_codebook else int(M_cb),
        }
        t0 = time.time()
        text_np = text_anchors.detach().cpu().float().numpy()  # [N, M, D]

        # A shared bank must be initialized from all semantic slots, not just
        # the global slot.  Otherwise initialization reproduces the same
        # global-only contamination as the former EMA update bug.
        init_banks = range(1) if self.share_codebook else range(M_cb)
        for m in init_banks:
            feats = (
                text_np.reshape(N * M_in, D)
                if self.share_codebook else text_np[:, m, :]
            )
            if mode == "mean":
                rng = np.random.default_rng(seed + m)
                replace = feats.shape[0] < K
                idx = rng.choice(feats.shape[0], size=K, replace=replace)
                centers = feats[idx]  # [K, D]
            elif mode == "kmeans":
                from sklearn.cluster import KMeans
                km = KMeans(
                    n_clusters=K,
                    random_state=int(seed + m),
                    n_init=4,
                    max_iter=100,
                )
                km.fit(feats)
                centers = km.cluster_centers_  # [K, D]
                diag[f"inertia_m{m}"] = float(km.inertia_)
            else:
                raise ValueError(f"[initialize_from_text_anchors] unknown mode={mode!r}")

            centers_t = torch.from_numpy(centers).to(
                dtype=self.codebooks.dtype, device=self.codebooks.device,
            )
            # Rescale to match VQ-VAE init scale (1/sqrt(D)). Per-codeword
            # norm comparable to legacy Normal(0, 1/sqrt(D)) init keeps the
            # downstream MSE / squared-L2 distance ranges familiar to the
            # rest of the training loop.
            cur_std = centers_t.std()
            if cur_std > 0:
                centers_t = centers_t * (target_std / cur_std)

            if self.update_mode == "ema":
                # tensor write into buffers: shape [K, D]
                self.codebooks[m].copy_(centers_t)
                # cluster_size set to N/K so revival doesn't immediately
                # mark these as dead (revival threshold is relative to
                # max(cluster_size_in_codebook)).
                init_cs = max(1.0, float(feats.shape[0]) / float(K))
                self.cluster_size[m].fill_(init_cs)
                self.embed_avg[m].copy_(centers_t * init_cs)
                if self.share_codebook:
                    self.embed_sqavg[m].copy_((centers_t ** 2) * init_cs)
            else:
                # gradient mode: codebook is nn.Parameter
                self.codebooks.data[m].copy_(centers_t)

        if self.share_codebook:
            self._sync_shared_state_from_bank0()

        diag["elapsed_sec"] = float(time.time() - t0)
        return diag

    def get_codebook_mean_anchors(self, exclude_global: bool = True) -> torch.Tensor:
        """Return per-part anchor vectors used at inference for OT routing.

        For each codebook c (shape [K, D]):
            anchor = normalize( normalize(c, dim=-1).mean(dim=0), dim=-1 )

        Per-codeword normalization first stops a small number of large-norm
        codewords from dominating the mean. With v78a active_mask, only
        currently-active codewords contribute to the mean.
        """
        cb_all = self.get_effective_codebooks()
        mask_all = self.get_effective_active_mask()
        if exclude_global:
            cb       = cb_all[1:]                                       # [M', K_max, D]
            mask     = mask_all[1:]                                     # [M', K_max]
        else:
            cb       = cb_all
            mask     = mask_all
        cb_n   = F.normalize(cb, dim=-1)                                # [M', K_max, D]
        # Mask inactive codewords with zero before averaging; divide by active count.
        mask_f = mask.unsqueeze(-1).to(cb_n.dtype)                      # [M', K_max, 1]
        cb_n_masked = cb_n * mask_f
        n_active = mask.sum(dim=-1, keepdim=True).clamp_min(1).to(cb_n.dtype)  # [M', 1]
        anchors = cb_n_masked.sum(dim=1) / n_active                     # [M', D]
        anchors = F.normalize(anchors, dim=-1)                          # [M', D]
        return anchors

    # -------------------------------------------------------- forward

    @torch.no_grad()
    def _revive_dead_codes(self, z: torch.Tensor) -> int:
        """Replace EMA-dead codewords with random samples from the batch.

        A codeword in codebook m is "dead" when its smoothed cluster_size
        is below ``revive_threshold * max(cluster_size_in_codebook_m)``.
        For each dead slot we:
            - draw a random z from this batch (per-codebook)
            - copy it into ``self.codebooks[m, dead_idx]``
            - reset ``embed_avg[m, dead_idx]``  to z * median_active_cs
            - reset ``cluster_size[m, dead_idx]`` to the median of active codes
              in this codebook (so revived codes start with a non-trivial mass).

        Shapes:
            z : [B, M, D]
        Returns total number of revived codewords across all codebooks.
        """
        B, M, D = z.shape
        assert M == self.num_codebooks, (M, self.num_codebooks)
        if self.share_codebook:
            cs = self.cluster_size[0]
            active = self.active_mask[0]
            cs_active = cs[active]
            if cs_active.numel() == 0:
                return 0
            dead_mask = active & (
                cs < cs_active.max() * self.revive_threshold
            )
            n_dead = int(dead_mask.sum().item())
            if n_dead == 0:
                return 0
            z_pool = z.reshape(B * M, D)
            idx_pool = torch.randint(
                0, z_pool.shape[0], (n_dead,), device=z.device,
            )
            new_codes = z_pool[idx_pool]
            active_alive_cs = cs[active & ~dead_mask]
            if active_alive_cs.numel() > 0:
                init_cs = active_alive_cs.median().clamp_min(1.0)
            else:
                init_cs = torch.tensor(
                    1.0, device=cs.device, dtype=cs.dtype,
                )
            self.codebooks[0, dead_mask] = new_codes
            self.cluster_size[0, dead_mask] = init_cs
            self.embed_avg[0, dead_mask] = new_codes * init_cs
            self.embed_sqavg[0, dead_mask] = (new_codes ** 2) * init_cs
            self._sync_shared_state_from_bank0()
            return n_dead

        n_revived = 0
        n_banks = 1 if self.share_codebook else M
        for m in range(n_banks):
            cs   = self.cluster_size[m]                                       # [K_max]
            active_m = self.active_mask[m]                                    # [K_max] bool
            # only consider active codewords for "dead" detection; inactive
            # are not eligible (their slot is reserved for future split).
            cs_active = cs[active_m]
            if cs_active.numel() == 0:
                continue
            cs_max = cs_active.max()
            thr  = cs_max * self.revive_threshold
            dead_mask = active_m & (cs < thr)                                 # [K_max] bool
            n_dead = int(dead_mask.sum().item())
            if n_dead == 0:
                continue
            idx_pool   = torch.randint(0, B, (n_dead,), device=z.device)
            new_codes  = z[idx_pool, m, :]                                    # [n_dead, D]
            active_alive_cs = cs[active_m & ~dead_mask]
            if active_alive_cs.numel() > 0:
                init_cs = active_alive_cs.median().clamp_min(1.0)
            else:
                init_cs = torch.tensor(1.0, device=cs.device, dtype=cs.dtype)
            self.codebooks[m, dead_mask]    = new_codes
            self.cluster_size[m, dead_mask] = init_cs
            self.embed_avg[m, dead_mask]    = new_codes * init_cs
            self.embed_sqavg[m, dead_mask]  = (new_codes ** 2) * init_cs
            n_revived += n_dead
        return n_revived

    @torch.no_grad()
    def _codeword_repulsion(self) -> None:
        """Option α: Gaussian-weighted codeword repulsion (in-place).

        For each codebook, push each codeword away from close neighbours.
        Force on codeword e_k:
            F_k = Σ_{j≠k} w_kj · (e_k − e_j) / ||e_k − e_j||
            w_kj = exp(−||e_k − e_j||² / σ²)
            σ²   = sigma_factor² · median_{j≠k} ||e_k − e_j||²   (auto, per cb)
        After applying, mirror the codebook change into ``embed_avg`` so the
        next EMA step does not immediately undo the displacement.
        """
        M, K, D = self.codebooks.shape
        eye = torch.eye(K, device=self.codebooks.device, dtype=torch.bool)  # [K, K]
        for m in range(1 if self.share_codebook else M):
            e = self.codebooks[m]                                          # [K, D]
            diff = e.unsqueeze(0) - e.unsqueeze(1)                         # [K, K, D]: diff[i,j] = e[i]-e[j]... wait
            # NOTE: convention -- diff[i, j, :] should be the vector pointing
            # FROM j TO i, so adding it to e[i] pushes e[i] further from e[j].
            # e.unsqueeze(1) is [K, 1, D] (row index = i), broadcast over j.
            # e.unsqueeze(0) is [1, K, D] (col index = j), broadcast over i.
            # So `e.unsqueeze(1) - e.unsqueeze(0)` = e[i] - e[j].
            diff = e.unsqueeze(1) - e.unsqueeze(0)                         # [K, K, D]
            dist_sq = (diff * diff).sum(-1)                                # [K, K]
            dist_sq = dist_sq.masked_fill(eye, float("inf"))
            # auto-sigma per codebook: median over off-diagonal entries
            finite = dist_sq[~eye].view(K, K - 1)
            median_dsq = finite.median().clamp_min(1e-6)
            sigma_sq = median_dsq * (self.repel_sigma_factor ** 2)
            weights = torch.exp(-dist_sq / sigma_sq.clamp_min(1e-6))       # [K, K]
            dist = dist_sq.clamp_min(1e-8).sqrt()                          # [K, K]
            unit = diff / dist.unsqueeze(-1)                               # [K, K, D]
            force = (weights.unsqueeze(-1) * unit).sum(dim=1)              # [K, D]
            # apply repulsion
            self.codebooks[m].add_(force, alpha=self.repel_strength)
        # sync embed_avg so EMA doesn't snap back next step
        if self.share_codebook:
            n_total = self.cluster_size[0].sum(dim=-1, keepdim=True)
            K_ = self.codebook_size
            cs = (
                (self.cluster_size[0] + self.ema_eps)
                / (n_total + K_ * self.ema_eps)
                * n_total
            )
            self.embed_avg[0].copy_(self.codebooks[0] * cs.unsqueeze(-1))
            self._sync_shared_state_from_bank0()
        else:
            n_total = self.cluster_size.sum(dim=-1, keepdim=True)
            K_ = self.codebook_size
            cs = (self.cluster_size + self.ema_eps) / (n_total + K_ * self.ema_eps) * n_total
            self.embed_avg.copy_(self.codebooks * cs.unsqueeze(-1))

    @torch.no_grad()
    def _ema_update(self, z: torch.Tensor, indices: torch.Tensor) -> None:
        """In-place EMA update of self.codebooks using new (z, indices) pairs.

        Shapes:
            z       : [B, M, D]
            indices : [B, M]                 # values in [0, K_max)
            cluster_size : [M, K_max]
            embed_avg    : [M, K_max, D]
            embed_sqavg  : [M, K_max, D]     # second moment for v78a variance
        """
        B, M, D = z.shape
        K_eff = self.K_max     # tensor dim = K_max; inactive entries just decay
        assert M == self.num_codebooks, (M, self.num_codebooks)
        if self.share_codebook:
            # All B*M assignments update the one bank used by every slot.
            # The former per-slot update wrote local-slot observations into
            # banks 1..M-1 even though lookup only read bank 0, so the shared
            # bank learned from the global slot alone.
            indices_flat = indices.reshape(B * M)
            z_flat = z.reshape(B * M, D)
            onehot = F.one_hot(
                indices_flat, num_classes=K_eff,
            ).to(z.dtype)                                                   # [B*M, K]
            cluster_size_b = onehot.sum(dim=0)                              # [K]
            embed_sum = onehot.transpose(0, 1) @ z_flat                     # [K, D]
            embed_sq_sum = onehot.transpose(0, 1) @ (z_flat * z_flat)       # [K, D]
            alpha = 1.0 - self.ema_decay
            self.cluster_size[0].mul_(self.ema_decay).add_(
                cluster_size_b, alpha=alpha,
            )
            self.embed_avg[0].mul_(self.ema_decay).add_(
                embed_sum, alpha=alpha,
            )
            self.embed_sqavg[0].mul_(self.ema_decay).add_(
                embed_sq_sum, alpha=alpha,
            )
            n_total = self.cluster_size[0].sum(dim=-1, keepdim=True)
            cs = (
                (self.cluster_size[0] + self.ema_eps)
                / (n_total + K_eff * self.ema_eps)
                * n_total
            )
            self.codebooks[0].copy_(self.embed_avg[0] / cs.unsqueeze(-1))
            self._sync_shared_state_from_bank0()
            if self.repel_strength > 0.0:
                self._ema_step.add_(1)
                if int(self._ema_step.item()) % self.repel_every == 0:
                    self._codeword_repulsion()
            return

        # one-hot per codebook -- [M, B, K_max]
        onehot = F.one_hot(indices.transpose(0, 1), num_classes=K_eff).to(z.dtype)
        cluster_size_b = onehot.sum(dim=1)                                       # [M, K_max]
        z_perm = z.transpose(0, 1)                                               # [M, B, D]
        # weighted sum + sum of squares per (m, k)
        embed_sum    = torch.bmm(onehot.transpose(1, 2), z_perm)                 # [M, K_max, D]
        embed_sq_sum = torch.bmm(onehot.transpose(1, 2), z_perm * z_perm)        # [M, K_max, D]
        # decay running stats
        self.cluster_size.mul_(self.ema_decay).add_(cluster_size_b, alpha=(1.0 - self.ema_decay))
        self.embed_avg   .mul_(self.ema_decay).add_(embed_sum,      alpha=(1.0 - self.ema_decay))
        self.embed_sqavg .mul_(self.ema_decay).add_(embed_sq_sum,   alpha=(1.0 - self.ema_decay))
        # Laplace-smoothed normalization (active codeword count, not K_max)
        n_total = self.cluster_size.sum(dim=-1, keepdim=True)                    # [M, 1]
        cs = (self.cluster_size + self.ema_eps) / (n_total + K_eff * self.ema_eps) * n_total
        self.codebooks.copy_(self.embed_avg / cs.unsqueeze(-1))
        # Option α: post-EMA codeword repulsion (rate-limited)
        if self.repel_strength > 0.0:
            self._ema_step.add_(1)
            if int(self._ema_step.item()) % self.repel_every == 0:
                self._codeword_repulsion()

    # =================== v78a: adaptive K / codeword split ================

    @torch.no_grad()
    def warm_start_from_state(
        self,
        cb: torch.Tensor,
        cs: torch.Tensor,
        ea: torch.Tensor,
        K_init: int,
    ) -> None:
        """Copy an existing K=K_init codebook into the K_max tensor.

        cb, cs, ea : tensors of shapes [M, K_init, D], [M, K_init], [M, K_init, D]
        Marks the first K_init codewords as active; the rest stay inactive.
        embed_sqavg is initialised to embed_avg**2 / cluster_size (rough
        second moment) so variance computation starts from a reasonable
        prior.
        """
        if self.share_codebook:
            raise ValueError(
                "warm_start_from_state is not defined for share_codebook; "
                "initialize the canonical shared bank directly instead"
            )
        assert cb.shape[0] == self.num_codebooks
        assert K_init <= self.K_max
        device, dtype = self.codebooks.device, self.codebooks.dtype
        self.codebooks[:, :K_init, :].copy_(cb.to(device=device, dtype=dtype))
        self.cluster_size[:, :K_init].copy_(cs.to(device=device, dtype=dtype))
        self.embed_avg[:, :K_init, :].copy_(ea.to(device=device, dtype=dtype))
        # rough second-moment initialisation
        cs_safe = cs.to(device=device, dtype=dtype).clamp_min(1e-6).unsqueeze(-1)
        z_mean  = ea.to(device=device, dtype=dtype) / cs_safe
        self.embed_sqavg[:, :K_init, :].copy_((z_mean ** 2) * cs_safe)
        # Active mask: first K_init True, rest False
        self.active_mask.zero_()
        self.active_mask[:, :K_init] = True
        print(f"[Quantizer.warm_start] copied K={K_init} into K_max={self.K_max}, "
              f"active count per codebook = {K_init}")

    @torch.no_grad()
    def compute_codeword_variance(self) -> torch.Tensor:
        """Per-codeword feature variance from EMA second moment.

        var(m, k) = mean over D of (E[z^2] − E[z]^2),
        where E[·] = embed_*_avg / cluster_size.

        Returns [M, K_max] tensor; inactive entries are 0.
        """
        cs = self.cluster_size.clamp_min(1e-6).unsqueeze(-1)
        z_mean = self.embed_avg   / cs                       # [M, K_max, D]
        z_sqm  = self.embed_sqavg / cs
        var    = (z_sqm - z_mean ** 2).clamp_min(0)          # [M, K_max, D]
        var_scalar = var.mean(dim=-1)                        # [M, K_max]
        var_scalar = var_scalar * self.active_mask.to(var_scalar.dtype)
        return var_scalar

    @torch.no_grad()
    def do_split(
        self,
        max_splits: int,
        z_per_cw: "list[list[Optional[torch.Tensor]]]",
        collision_pressure: Optional[torch.Tensor] = None,
        min_cluster_for_split: int = 16,
    ) -> Dict[str, int]:
        """v78a codeword split.

        Args:
            max_splits: hard cap on the number of (m, k) splits this call.
            z_per_cw: nested list, z_per_cw[m][k] = [N_mk, D] tensor of z's
                that were assigned to codeword k in codebook m during the
                most recent inference pass (None for inactive / unsampled).
            collision_pressure: [M, K_max] tensor or None. If None, falls
                back to using cluster_size as a proxy.

        Returns dict with {"n_split_total", "split_count_per_codebook"}.
        """
        if self.share_codebook:
            raise ValueError(
                "adaptive codeword splitting is not supported with "
                "share_codebook"
            )
        device = self.codebooks.device
        M, K_max = self.active_mask.shape

        usage = self.cluster_size.clone()                                # [M, K_max]
        variance = self.compute_codeword_variance()                      # [M, K_max]
        if collision_pressure is None:
            collision_pressure = usage.clone()
        cp = collision_pressure.to(device=device, dtype=usage.dtype)

        # split_score(m, k) = usage * variance * collision_pressure
        score = usage * variance * cp                                    # [M, K_max]
        # Mask out inactive codewords (can't split inactive) and those with
        # too few samples
        score = score * self.active_mask.to(score.dtype)
        # find inactive slots per codebook
        n_split_total = 0
        split_count_per_codebook = [0] * M

        # Greedy: pick the (m, k) pair with highest score globally, split,
        # repeat until max_splits or no eligible candidates remain.
        for _ in range(max_splits):
            inactive_slots_per_m = [
                (~self.active_mask[m]).nonzero(as_tuple=False).flatten().tolist()
                for m in range(M)
            ]
            # mask out (m, k) for which we have no inactive slot
            mask_no_slot = torch.tensor(
                [len(s) == 0 for s in inactive_slots_per_m],
                device=device,
            )                                                            # [M]
            if mask_no_slot.all():
                break
            score_local = score.clone()
            score_local[mask_no_slot] = -1.0
            # also drop entries whose collected z sample size is below threshold
            for m in range(M):
                for k in range(K_max):
                    if (z_per_cw[m][k] is None or
                        (z_per_cw[m][k] is not None and z_per_cw[m][k].shape[0] < min_cluster_for_split)):
                        score_local[m, k] = -1.0
            if score_local.max() <= 0:
                break
            flat_idx = score_local.view(-1).argmax()
            m = int(flat_idx // K_max)
            k = int(flat_idx %  K_max)
            inactive_slots = inactive_slots_per_m[m]
            if not inactive_slots:
                score[m, k] = 0
                continue
            new_k = inactive_slots[0]
            # collect z's mapped to codeword k in codebook m
            z_samples = z_per_cw[m][k]                                   # [N, D]
            if z_samples.shape[0] < min_cluster_for_split:
                score[m, k] = 0
                continue
            # 2-means via random init from 2 distinct samples + a couple of iterations
            N, D = z_samples.shape
            rng = torch.randperm(N, device=z_samples.device)[:2]
            cA = z_samples[rng[0]].clone()
            cB = z_samples[rng[1]].clone()
            for _it in range(5):
                dA = ((z_samples - cA.unsqueeze(0)) ** 2).sum(-1)
                dB = ((z_samples - cB.unsqueeze(0)) ** 2).sum(-1)
                assign = (dB < dA)                                       # True -> B
                nA = (~assign).sum().clamp_min(1)
                nB = assign.sum().clamp_min(1)
                cA = z_samples[~assign].mean(0)
                cB = z_samples[ assign].mean(0)
            # apply: codeword k <- cA, new_k <- cB
            ema_cs_per_side = self.cluster_size[m, k].item() * 0.5
            self.codebooks[m, k].copy_(cA)
            self.codebooks[m, new_k].copy_(cB)
            self.cluster_size[m, k]     = ema_cs_per_side
            self.cluster_size[m, new_k] = ema_cs_per_side
            self.embed_avg  [m, k].copy_   (cA * ema_cs_per_side)
            self.embed_avg  [m, new_k].copy_(cB * ema_cs_per_side)
            self.embed_sqavg[m, k].copy_   ((cA ** 2) * ema_cs_per_side)
            self.embed_sqavg[m, new_k].copy_((cB ** 2) * ema_cs_per_side)
            self.active_mask[m, new_k] = True
            # zero score of the split-source to prevent re-selection this call
            score[m, k] = 0
            split_count_per_codebook[m] += 1
            n_split_total += 1
        return {
            "n_split_total": n_split_total,
            "split_count_per_codebook": split_count_per_codebook,
            "active_K_per_codebook": [int(self.active_mask[m].sum().item()) for m in range(M)],
        }

    def forward(self, semantic_visual_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        # semantic_visual_tokens: [B, M, D]
        B, M, D = semantic_visual_tokens.shape
        if M != self.num_codebooks:
            raise ValueError(
                f"[SemanticCodebookQuantizer] semantic_visual_tokens has M={M} "
                f"but quantizer was built with num_codebooks={self.num_codebooks}"
            )
        if D != self.d_model:
            raise ValueError(
                f"[SemanticCodebookQuantizer] semantic_visual_tokens has D={D} "
                f"but quantizer was built with d_model={self.d_model}"
            )

        # A4 ablation: when share_codebook, all M slots quantize against the
        # SAME codebook (slot 0). Expand is a view -> gradient accumulates into
        # codebooks[0]; slots 1..M-1 params are unused (report effective count).
        codebooks = self.get_effective_codebooks()
        active_mask = self.get_effective_active_mask()

        # nearest-neighbour distance to codewords. Distance tensor is sized
        # K_max; inactive codewords are masked with +inf so argmin can never
        # select them (v78a adaptive K).
        if self.distance_mode == "cosine":
            z_n  = F.normalize(semantic_visual_tokens, dim=-1)                 # [B, M, D]
            cb_n = F.normalize(codebooks, dim=-1)                             # [M, K_max, D]
            cos_sim   = torch.einsum("bmd,mkd->bmk", z_n, cb_n)                # [B, M, K_max]
            distances = 1.0 - cos_sim                                          # [B, M, K_max] in [0, 2]
        else:
            diff = semantic_visual_tokens.unsqueeze(2) - codebooks.unsqueeze(0)
            distances = (diff ** 2).sum(dim=-1)                                # [B, M, K_max]
        # v78a: mask inactive codewords from lookup
        if self.K_max != self.codebook_size or (~active_mask).any():
            inactive = (~active_mask).unsqueeze(0)                             # [1, M, K_max]
            distances = distances.masked_fill(inactive, float("inf"))
        indices   = distances.argmin(dim=-1)                                   # [B, M]

        # gather quantized embeddings via advanced indexing:
        #   quantized[b, m, :] = codebooks[m, indices[b, m], :]
        device = semantic_visual_tokens.device
        m_idx  = torch.arange(self.num_codebooks, device=device).view(1, -1).expand(B, -1)  # [B, M]
        quantized = codebooks[m_idx, indices]                              # [B, M, D]

        # straight-through estimator (gradient passes through to the encoder
        # while the actual values are the quantized codewords)
        if self.bypass:
            # (c-1) stage 1: identity quantisation.
            quantized = semantic_visual_tokens
            quantized_st = semantic_visual_tokens
        else:
            quantized_st = semantic_visual_tokens + (quantized - semantic_visual_tokens).detach()

        # EMA codebook update (only when training; no-op in 'gradient' mode).
        # Done AFTER computing indices/quantized so this batch's outputs are
        # consistent with the codebook state seen during the loss computation.
        # (A): `freeze_after_epoch` >= 0 stops the codebook moving from that
        # epoch on. Both the EMA write and dead-code revival are suppressed,
        # because reviving a codeword also rewrites it. -1 (default) keeps the
        # historical behaviour exactly.
        _frozen = (
            self.freeze_after_epoch >= 0
            and int(self._frozen_epoch_now) >= self.freeze_after_epoch
        )
        if self.update_mode == "ema" and self.training and not _frozen and not self.bypass:
            self._ema_update(semantic_visual_tokens.detach(), indices)
            # dead-code rejuvenation -- run every `revive_every` training
            # forwards. We skip it on the very first step so the EMA has at
            # least one batch of real assignments to compute "max" against.
            if self.revive_dead:
                self._revive_step.add_(1)
                if int(self._revive_step.item()) > 1 and (int(self._revive_step.item()) % self.revive_every == 0):
                    self._revive_dead_codes(semantic_visual_tokens.detach())

        return {
            "quantized_tokens":     quantized_st,    # [B, M, D]
            "quantized_tokens_raw": quantized,       # [B, M, D]
            "codebook_indices":     indices,         # [B, M]
            "codebook_distances":   distances,       # [B, M, K]
        }


# =====================================================================
# Per-codebook codon head
# =====================================================================

class CodonHead(nn.Module):
    """Per-codebook codon head: ``[B, D]`` → ``[B, 3, 4]``.

    Splits a codeword embedding into 3 codon-position chunks of ``D/3`` each,
    then maps each chunk to a 4-class (A/C/G/T) logit vector via a single
    Linear(D/3, 4). Within one head the same Linear is shared across the 3
    positions, but ACROSS the 6 codebooks each ``CodonHead`` instance is
    SEPARATE (no weight sharing).

    Differentiable hard path:
        - In training, ``dna_hash_code`` is the *straight-through* tensor
          (forward = hard one-hot, backward = soft / continuous gradient).
            * if ``use_gumbel_softmax=True`` (default), it is
              ``F.gumbel_softmax(logits, tau, hard=True)``.
            * else, it is the deterministic STE: ``hard - cont.detach() + cont``.
        - In eval, ``dna_hash_code`` is the deterministic argmax one-hot.

    Forward returns:
        logits             [B, 3, 4]   pre-softmax
        continuous_code    [B, 3, 4]   softmax(logits, dim=-1)
        base_indices       [B, 3]      argmax(continuous_code, dim=-1)
        dna_hash_code      [B, 3, 4]   ST in training, deterministic hard in eval
        dna_hash_code_hard [B, 3, 4]   always deterministic argmax one-hot
        dna_hash_code_st   [B, 3, 4]   always the ST (gradient-bearing) path
    """

    def __init__(
        self,
        d_model: int,
        use_gumbel_softmax: bool = True,
        gumbel_tau: float = 1.0,
        use_residual: bool = False,
        head_hidden_dim: int = 0,
        use_text_anchor: bool = False,
        anchor_temperature: float = 0.1,
        position_specific_head: bool = False,
        position_residual_adapter: bool = False,
        residual_split: bool = False,
        residual_gate: bool = False,
        use_full_linear: bool = False,
        num_codons: int = 3,
        chunk_layernorm: bool = False,                 # v122: L = codon-positions per codebook
        chunk_interleave: bool = False,
    ) -> None:
        super().__init__()
        self.num_codons: int = int(num_codons)
        if self.num_codons < 1:
            raise ValueError(f"[CodonHead] num_codons must be >= 1; got {self.num_codons}")
        # (Exp1') Normalise each codon-position chunk before the 4-way Linear.
        # Diagnosis 2026-07-27: slot0 receives siglip2_global (chunk std 0.39 vs
        # 0.86-0.92 for routed slots); its head compensated with |W|=16.4 (8-10x
        # the others), saturating the logits and collapsing 128 codewords onto
        # 21 codons (6.1:1 vs the 2:1 pigeonhole floor). Normalising the chunk
        # removes the scale imbalance that drives that compensation.
        self.chunk_norm = None
        if bool(chunk_layernorm):
            self.chunk_norm = nn.LayerNorm(int(d_model) // self.num_codons)
        if d_model % self.num_codons != 0:
            raise ValueError(
                f"[CodonHead] d_model must be divisible by num_codons={self.num_codons} "
                f"for codon-position heads; got d_model={d_model}."
            )
        if self.num_codons != 3 and residual_split:
            raise ValueError(
                "[CodonHead] residual_split is hard-wired to 3 positions "
                "(semantic for pos 0,1; residual for pos 2); incompatible "
                f"with num_codons={self.num_codons}."
            )
        self.d_model: int = int(d_model)
        self.chunk:   int = self.d_model // self.num_codons
        # 2026-07-30: how the d_model vector is split into `num_codons` chunks.
        # Contiguous slicing (legacy) gives position l dims [l*chunk,(l+1)*chunk);
        # interleaved gives position l dims [l, l+L, l+2L, ...]. Motivation: the
        # slot-0 collapse is a JOINT (3-way) dependence that survives removing
        # the shared head, so the remaining head-side freedom is the partition
        # itself. Zero parameters either way.
        self.chunk_interleave: bool = bool(chunk_interleave)
        # v62 (Option A): residual-conditioned codon head.
        self.use_residual: bool = bool(use_residual)
        # v69b (Exp 2): residual-split — codon position 0,1 from codeword,
        # position 2 from gamma*residual. Bypasses the concat -> input_proj
        # path entirely; uses 3 separate Linears.
        self.residual_split: bool = bool(residual_split)
        if self.use_residual and not self.residual_split:
            self.input_proj = nn.Linear(2 * self.d_model, self.d_model)
        else:
            self.input_proj = None
        # v71a (Exp 5): per-codebook learnable residual gate. a, b are scalars.
        # effective_residual = sigmoid(a * ||z-q|| + b) * residual.
        self.residual_gate: bool = bool(residual_gate)
        if self.residual_gate and self.use_residual:
            self.residual_gate_a = nn.Parameter(torch.zeros(()))
            self.residual_gate_b = nn.Parameter(torch.zeros(()))
        else:
            self.residual_gate_a = None
            self.residual_gate_b = None
        # v66: per-codon text-anchored prototype classifier.
        # Replaces Linear(chunk, 4) with cosine similarity to a learnable
        # [3, 4, chunk] prototype tensor. Aux text supervision happens via
        # `forward(text_chunks=...)` which derives a per-position target class
        # = argmax(cos(text_chunk, proto)) and returns a CE loss.
        self.use_text_anchor: bool = bool(use_text_anchor)
        self.anchor_temperature: float = float(anchor_temperature)
        # v65: light MLP fc head (mutually exclusive with prototype head).
        self.head_hidden_dim: int = int(head_hidden_dim)
        # v69a (Exp 1): position-specific Linear(chunk, 4) per codon position
        # (3 independent layers instead of shared self.fc). Mutually exclusive
        # with use_text_anchor / head_hidden_dim / residual_split (the last
        # one already uses 3 separate Linears by its own design).
        self.position_specific_head: bool = bool(position_specific_head)
        # v87a: keep the shared classifier as the semantic backbone, then add
        # zero-init per-position residual adapters to logits.
        self.position_residual_adapter: bool = bool(position_residual_adapter)
        # v105: full-input codon decoder (Linear(d_model, 12) -> view [B, 3, 4])
        # instead of chunk-partition shared Linear(chunk, 4) (3 positions).
        self.use_full_linear: bool = bool(use_full_linear)
        if self.use_full_linear and (
            use_text_anchor or residual_split or position_specific_head
            or head_hidden_dim > 0
        ):
            raise ValueError(
                "[CodonHead] --codon_full_linear is mutually exclusive with "
                "--codon_text_anchor / --codon_residual_split / "
                "--codon_position_specific_head / --codon_head_hidden_dim>0."
            )
        if self.use_text_anchor:
            self.proto = nn.Parameter(
                torch.randn(self.num_codons, 4, self.chunk) / math.sqrt(self.chunk)
            )
            self.fc = None
        elif self.residual_split:
            # v69b: 3 separate Linears — positions 0,1 = semantic, position 2 = residual
            self.semantic_fc_pos0 = nn.Linear(self.chunk, 4)
            self.semantic_fc_pos1 = nn.Linear(self.chunk, 4)
            self.residual_fc      = nn.Linear(self.chunk, 4)
            self.fc = None
        elif self.position_specific_head:
            # v69a: L independent Linear(chunk, 4) layers
            self.fc_pos = nn.ModuleList(
                [nn.Linear(self.chunk, 4) for _ in range(self.num_codons)]
            )
            self.fc = None
        elif self.head_hidden_dim > 0:
            self.fc = nn.Sequential(
                nn.Linear(self.chunk, self.head_hidden_dim),
                nn.GELU(),
                nn.Linear(self.head_hidden_dim, 4),
            )
        elif self.use_full_linear:
            # v105: Linear(d_model, L*4) -> view [B, L, 4]. Each (position, base)
            # output uses ALL d_model dims instead of just its chunk.
            self.fc = nn.Linear(self.d_model, self.num_codons * 4)
        else:
            self.fc = nn.Linear(self.chunk, 4)
        if (
            self.position_residual_adapter
            and not self.use_text_anchor
            and not self.residual_split
            and not self.position_specific_head
        ):
            self.fc_pos_adapter = nn.ModuleList(
                [nn.Linear(self.chunk, 4) for _ in range(self.num_codons)]
            )
            for adapter in self.fc_pos_adapter:
                nn.init.zeros_(adapter.weight)
                nn.init.zeros_(adapter.bias)
        else:
            self.fc_pos_adapter = None
        self.use_gumbel_softmax: bool = bool(use_gumbel_softmax)
        self.gumbel_tau: float = float(gumbel_tau)

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        gamma: float = 0.0,
        text_chunks: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # x: [B, D]; optional residual: [B, D]; gamma: scaling factor
        # text_chunks: [B, D] -- per-codebook text caption embedding (only used
        # by the v66 text-anchored prototype path).
        B, D = x.shape
        assert D == self.d_model, (
            f"[CodonHead] expected last-dim {self.d_model}, got {D}"
        )
        # Track mean gate value across batch for logging (v71a)
        gate_mean = torch.zeros((), device=x.device, dtype=x.dtype)
        residual_active = (
            self.use_residual and residual is not None and gamma > 0
        )
        # v71a (Exp 5): residual gate. effective_residual = sigmoid(a*||r|| + b) * residual
        if residual_active and self.residual_gate and self.residual_gate_a is not None:
            r_norm = residual.norm(dim=-1, keepdim=True).detach()         # [B, 1]
            gate   = torch.sigmoid(
                self.residual_gate_a * r_norm.squeeze(-1) + self.residual_gate_b
            )                                                              # [B]
            gate_mean = gate.mean().detach()
            residual = gate.unsqueeze(-1) * residual                       # [B, D]
        L = self.num_codons

        def _split(t: torch.Tensor) -> torch.Tensor:
            """[B, d_model] -> [B, L, chunk]. Contiguous unless interleaving."""
            if self.chunk_interleave:
                # position l gets dims l, l+L, l+2L, ...
                return t.view(t.shape[0], self.chunk, L).transpose(1, 2)
            return t.view(t.shape[0], L, self.chunk)

        # v69b (Exp 2): residual-split path (3-position only; guarded at __init__).
        if residual_active and self.residual_split:
            q_chunks = _split(x)                                           # [B, L, chunk]
            if self.chunk_norm is not None:
                q_chunks = self.chunk_norm(q_chunks)
            r_chunks = _split(gamma * residual)
            l0 = self.semantic_fc_pos0(q_chunks[:, 0, :])                  # [B, 4]
            l1 = self.semantic_fc_pos1(q_chunks[:, 1, :])                  # [B, 4]
            l2 = self.residual_fc(r_chunks[:, 2, :])                       # [B, 4]
            logits = torch.stack([l0, l1, l2], dim=1)                      # [B, 3, 4]
            loss_text_anchor = torch.zeros((), device=x.device, dtype=x.dtype)
        else:
            # v62 residual injection (legacy concat path; only when split is OFF)
            if residual_active and not self.residual_split:
                assert residual.shape == x.shape, (
                    f"[CodonHead] residual shape {tuple(residual.shape)} != "
                    f"x shape {tuple(x.shape)}"
                )
                combined = torch.cat([x, gamma * residual], dim=-1)      # [B, 2D]
                x = self.input_proj(combined)                             # [B, D]
            h = _split(x)                                                 # [B, L, D/L]
            if self.chunk_norm is not None:
                h = self.chunk_norm(h)
            # v66 text-anchored prototype head
            if self.use_text_anchor:
                h_n     = F.normalize(h, dim=-1)
                proto_n = F.normalize(self.proto, dim=-1)
                logits  = torch.einsum('bpc,pkc->bpk', h_n, proto_n) / self.anchor_temperature
                loss_text_anchor = torch.zeros((), device=x.device, dtype=x.dtype)
                if text_chunks is not None and self.training:
                    t        = _split(text_chunks)
                    t_n      = F.normalize(t, dim=-1)
                    text_logits = torch.einsum('bpc,pkc->bpk', t_n, proto_n) / self.anchor_temperature
                    target   = text_logits.argmax(dim=-1).detach()
                    loss_text_anchor = F.cross_entropy(
                        logits.reshape(-1, 4), target.reshape(-1),
                    )
            elif self.position_specific_head:
                # v69a: per-position independent Linear(chunk, 4)
                logits = torch.stack(
                    [self.fc_pos[p](h[:, p, :]) for p in range(L)], dim=1
                )                                                          # [B, L, 4]
                loss_text_anchor = torch.zeros((), device=x.device, dtype=x.dtype)
            elif self.use_full_linear:
                # v105: full-input decoder. Linear(d_model, L*4) on the WHOLE
                # codeword (no chunk partition) -> reshape to [B, L, 4].
                logits = self.fc(x).view(B, L, 4)                          # [B, L, 4]
                loss_text_anchor = torch.zeros((), device=x.device, dtype=x.dtype)
            else:
                logits = self.fc(h)                                       # [B, L, 4]
                if self.fc_pos_adapter is not None:
                    # h: [B, L, chunk] -> delta: [B, L, 4]
                    delta = torch.stack(
                        [self.fc_pos_adapter[p](h[:, p, :]) for p in range(L)],
                        dim=1,
                    )
                    assert delta.shape == logits.shape, (
                        f"[CodonHead] adapter delta shape {tuple(delta.shape)} "
                        f"!= logits shape {tuple(logits.shape)}"
                    )
                    logits = logits + delta
                loss_text_anchor = torch.zeros((), device=x.device, dtype=x.dtype)
        cont   = F.softmax(logits, dim=-1)                               # [B, L, 4]
        idx    = cont.argmax(dim=-1)                                     # [B, L]
        hard   = F.one_hot(idx, num_classes=4).to(cont.dtype)            # [B, L, 4]

        if self.training:
            if self.use_gumbel_softmax:
                # F.gumbel_softmax with hard=True does ST internally:
                # forward = one-hot, backward = soft.
                st = F.gumbel_softmax(
                    logits, tau=self.gumbel_tau, hard=True, dim=-1,
                )                                                        # [B, L, 4]
            else:
                # deterministic straight-through estimator
                st = hard - cont.detach() + cont                         # [B, L, 4]
        else:
            # at eval, both hard and st collapse to the deterministic argmax
            st = hard

        return {
            "logits":             logits,   # [B, L, 4]
            "continuous_code":    cont,     # [B, L, 4]
            "base_indices":       idx,      # [B, L]
            "dna_hash_code":      st,       # train: ST, eval: hard
            "dna_hash_code_hard": hard,     # always deterministic one-hot
            "dna_hash_code_st":   st,       # always the gradient-bearing path
            "loss_text_anchor":   loss_text_anchor,  # scalar (v66 aux CE)
            "residual_gate_mean": gate_mean,         # scalar (v71a logging)
        }

    def decode_codeword(self, codewords: torch.Tensor) -> torch.Tensor:
        """v106: codeword-level codon decoder for bijection loss.

        Applies the codon decoder to raw codeword embeddings WITHOUT
        residual injection (i.e., what the decoder would output if a
        sample's quantized representation was exactly the codeword).
        Mirrors the active decoding path in forward(), handling all
        decoder variants (text_anchor / residual_split / position_specific
        / full_linear / shared fc).

        codewords: [K, d_model]
        returns  : codon logits [K, 3, 4]
        """
        K, D = codewords.shape
        assert D == self.d_model, (
            f"[CodonHead.decode_codeword] expected last-dim {self.d_model}, got {D}"
        )
        L = self.num_codons
        h = codewords.view(K, L, self.chunk)
        if self.use_text_anchor:
            h_n = F.normalize(h, dim=-1)
            proto_n = F.normalize(self.proto, dim=-1)
            return torch.einsum('kpc,pqc->kpq', h_n, proto_n) / self.anchor_temperature
        if self.residual_split:
            # No residual at codeword level; map all 3 positions via the
            # semantic branches and use the residual_fc as if residual=0.
            l0 = self.semantic_fc_pos0(h[:, 0, :])                   # [K, 4]
            l1 = self.semantic_fc_pos1(h[:, 1, :])                   # [K, 4]
            l2 = self.residual_fc(torch.zeros_like(h[:, 2, :]))      # [K, 4]
            return torch.stack([l0, l1, l2], dim=1)
        if self.position_specific_head:
            return torch.stack(
                [self.fc_pos[p](h[:, p, :]) for p in range(L)], dim=1
            )
        if self.use_full_linear:
            return self.fc(codewords).view(K, L, 4)
        # Default shared Linear(chunk, 4) path (with optional v87a adapter)
        logits = self.fc(h)                                          # [K, L, 4]
        if self.fc_pos_adapter is not None:
            delta = torch.stack(
                [self.fc_pos_adapter[p](h[:, p, :]) for p in range(L)],
                dim=1,
            )
            logits = logits + delta
        return logits

# =====================================================================
# Reconstruction decoders (v28 ablation, 2026-05-15)
# =====================================================================
# Both take the 6 selected codewords (quantized_tokens, [B, 6, D])
# concatenated to [B, 6*D] and predict a reconstruction target.
# v28a uses PixelDecoder (target = raw image),
# v28b uses FeatureDecoder (target = frozen SigLIP2 visual_global).
#
# Goal: replace v27b's noisy batch-cosine pseudo-positive with a strong,
# dense per-sample unsupervised signal that forces each codebook to encode
# information actually useful for image recovery -- the natural
# architectural justification for the "compositional code" claim.

class FeatureDecoder(nn.Module):
    """Reconstruct frozen SigLIP2 visual_global from 6 concatenated codewords.

    Input  : [B, 6 * D]   (concat of quantized_tokens flattened on dim 1)
    Output : [B, D_proj]  (MSE / cosine target = cached visual_global)

    Lightweight 2-layer MLP. Default hidden = 4*D_proj. Output normalized
    to unit length so cosine similarity with the cached SigLIP2 feature is
    well-behaved (matches the cos-similarity geometry we already use for
    the v27 unsupervised positive signal).
    """
    def __init__(self, d_model: int, d_proj: int, num_codebooks: int = 6,
                 hidden_mult: int = 4):
        super().__init__()
        in_dim = num_codebooks * d_model
        hid    = hidden_mult * d_proj
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.GELU(),
            nn.Linear(hid, d_proj),
        )

    def forward(self, codewords: torch.Tensor) -> torch.Tensor:
        # codewords [B, 6, D] -> [B, 6*D] -> [B, D_proj]
        B = codewords.shape[0]
        flat = codewords.reshape(B, -1)
        return self.net(flat)


class PixelDecoder(nn.Module):
    """Reconstruct a 224x224 RGB image from 6 concatenated codewords.

    Input  : [B, 6 * D]
    Output : [B, 3, 224, 224]    (in the same ImageNet-normalized space the
                                  dataloader feeds to SigLIP2 as input)

    Architecture (bottlenecked: ~15M params, not 100M+):
        Linear (6D     -> 1024)             # bottleneck projection
        GELU
        Linear (1024   -> 256 * 7 * 7)
        Reshape to [B, 256, 7, 7]
        ConvTranspose stack: 7 -> 14 -> 28 -> 56 -> 112 -> 224
        (no activation on the last layer; the loss compares against
        ImageNet-normalized pixel values which can take any sign).

    Avoiding a single huge Linear(4608, 25088) is critical -- that one
    projection alone is ~115M params and slows training without buying
    additional capacity.
    """
    def __init__(self, d_model: int, num_codebooks: int = 6,
                 bottleneck: int = 1024):
        super().__init__()
        in_dim = num_codebooks * d_model
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, 256 * 7 * 7),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 7  -> 14
            nn.GELU(),
            nn.ConvTranspose2d(128,  64, kernel_size=4, stride=2, padding=1),  # 14 -> 28
            nn.GELU(),
            nn.ConvTranspose2d( 64,  32, kernel_size=4, stride=2, padding=1),  # 28 -> 56
            nn.GELU(),
            nn.ConvTranspose2d( 32,  16, kernel_size=4, stride=2, padding=1),  # 56 -> 112
            nn.GELU(),
            nn.ConvTranspose2d( 16,   3, kernel_size=4, stride=2, padding=1),  # 112-> 224
        )

    def forward(self, codewords: torch.Tensor) -> torch.Tensor:
        B = codewords.shape[0]
        x = self.in_proj(codewords.reshape(B, -1))
        x = x.view(B, 256, 7, 7)
        return self.deconv(x)


# =====================================================================
# v70a / v72a: small MLP heads from the flattened 18*4=72-dim DNA hash
# =====================================================================

class HashReconDecoder(nn.Module):
    """v70a: maps DNA hash (flattened 72-d) back to a SigLIP2 embedding.

    Two-layer MLP. Output is cosine-aligned with a frozen target
    (visual_global / text_global). Decoder is unused at inference.
    """

    def __init__(self, in_dim: int = 72, hidden_dim: int = 256, out_dim: int = 768) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, hash_flat: torch.Tensor) -> torch.Tensor:
        # hash_flat: [B, 72]
        return self.fc(hash_flat)                                          # [B, out_dim]


class DualHashProj(nn.Module):
    """v72a: dual projection heads from flattened DNA hash.

    `semantic_proj` -> cosine alignment with text / visual_global.
    `instance_proj` -> NtXent across paired-aug views.
    Both are MLPs; inference unchanged (heads not used at retrieval).
    """

    def __init__(self, in_dim: int = 72, hidden_dim: int = 256, out_dim: int = 768) -> None:
        super().__init__()
        self.semantic_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.instance_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, hash_flat: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "semantic": self.semantic_proj(hash_flat),
            "instance": self.instance_proj(hash_flat),
        }


# =====================================================================
# Top-level model
# =====================================================================

class SigLIP2SemanticOTModel(nn.Module):
    """SigLIP2 dual-encoder + visual/text adapters + Sinkhorn router + codebooks.

    Members (mirroring `model.py` style):
        self.device, self.batch_size
        self.backbone           shared SigLIP2 dual-encoder, loaded ONCE
        self.visual_encoder     reads backbone.vision_model
        self.text_encoder       reads backbone.text_model    (separate weights)
        self.visual_adapter     trainable; SEPARATE from text_adapter
        self.text_adapter       trainable; SEPARATE from visual_adapter
        self.router             Sinkhorn semantic router (5 local parts only)
        self.quantizer          SemanticCodebookQuantizer with 6 codebooks
        self.d_model            adapter output dim
        self.num_semantic_parts == 6
        self.num_codebooks      == 6
        self.codebook_size      == K (default 64)
    """

    def __init__(self, args: Any) -> None:
        super().__init__()

        self.device     = getattr(args, "device", torch.device("cpu"))
        self.batch_size = int(getattr(args, "batch_size", 1))

        # ---------- backbone (loaded once, shared between two encoders) ---
        # Backbone family is selected by --backbone_type ('siglip2' default,
        # or 'clip'). VisualEncoder / TextEncoder only touch the duck-typed
        # interface (`.vision_hidden_dim`, `.text_hidden_dim`,
        # `.projection_dim`, `.vision_model`, `.text_model`, `.model`) which
        # both `SigLIP2Backbone` and `CLIPBackbone` implement.
        self.backbone_type: str = str(getattr(args, "backbone_type", "siglip2"))
        if self.backbone_type not in ("siglip2", "clip"):
            raise ValueError(
                f"[model_siglip2] backbone_type must be 'siglip2' or 'clip', "
                f"got {self.backbone_type!r}"
            )
        if self.backbone_type == "clip":
            clip_name = (
                getattr(args, "clip_backbone", None)
                or DEFAULT_CLIP_BACKBONE
            )
            phase3_snapshot = None
            snapshot_dir = getattr(args, "clip_snapshot_dir", None)
            if snapshot_dir:
                import json
                try:
                    tokenizers = json.loads(
                        getattr(args, "clip_snapshot_tokenizers_sha256_json")
                    )
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        "Phase-3 CLIP tokenizer authority is not valid JSON"
                    ) from error
                phase3_snapshot = {
                    "checkpoint": clip_name,
                    "snapshot_dir": snapshot_dir,
                    "revision": getattr(args, "clip_snapshot_revision", None),
                    "weight_file": getattr(args, "clip_snapshot_weight_file", None),
                    "weight_sha256": getattr(args, "clip_snapshot_weight_sha256", None),
                    "config_sha256": getattr(args, "clip_snapshot_config_sha256", None),
                    "tokenizer_files_sha256": tokenizers,
                }
            self.backbone = build_clip_backbone(
                clip_name, phase3_snapshot=phase3_snapshot
            )
        else:
            backbone_name = (
                getattr(args, "siglip2_backbone", None)
                or getattr(args, "backbone_name", None)
                or DEFAULT_BACKBONE
            )
            self.backbone = build_pretrained_backbone(backbone_name)

        # The two encoders share the same dual-encoder OBJECT, but read
        # disjoint sub-modules: vision_model vs text_model. The two towers
        # have no tied weights, by SigLIP2 / CLIP design.
        self.visual_encoder = VisualEncoder(backbone=self.backbone)
        self.text_encoder   = TextEncoder  (backbone=self.backbone)

        # ---------- d_model & adapter dims --------------------------------
        # proj_dim is the backbone's shared image/text embedding dim:
        #   SigLIP2-base-patch16-224: 768   (== vision_hidden, divisible by 3)
        #   CLIP-ViT-B/16          : 512   (NOT divisible by 3)
        # d_model defaults to proj_dim, BUT the codon head requires
        # d_model % 3 == 0 (chunk = d_model // 3). When backbone=clip we
        # therefore promote the default to 768 (== vision_hidden_dim for both
        # SigLIP2 and CLIP-B/16) which keeps chunk=256 identical to the
        # SigLIP2 path. Users can still override via --d_model.
        proj_dim = self.visual_encoder.get_output_dim()                 # backbone projection_dim
        self.proj_dim: int = int(proj_dim)
        d_model_arg = getattr(args, "d_model", None)
        if d_model_arg is not None:
            self.d_model: int = int(d_model_arg)
        elif self.backbone_type == "clip":
            # vision_hidden_dim is 768 for clip-vit-base-patch16; pick that so
            # visual_adapter does 768 -> 768 (residual MLP) and text_adapter
            # does 512 -> 768 (up-projection). chunk = 256, identical to
            # SigLIP2 defaults.
            self.d_model: int = int(self.visual_encoder.hidden_dim)
            print(f"[model_siglip2] backbone_type=clip with d_model unset -> "
                  f"defaulting d_model={self.d_model} (= visual hidden_dim) so "
                  f"d_model % 3 == 0 holds for CodonHead.")
        else:
            self.d_model: int = int(proj_dim)
        adapter_hidden  = getattr(args, "adapter_hidden_dim", None)
        adapter_hidden  = int(adapter_hidden) if adapter_hidden is not None else int(self.d_model * 2)
        adapter_dropout = float(getattr(args, "adapter_dropout", 0.0))
        # v59: text-only bottleneck override
        _text_hidden_override = getattr(args, "text_adapter_hidden_dim", None)
        text_adapter_hidden = (
            int(_text_hidden_override) if _text_hidden_override is not None else adapter_hidden
        )

        # ---------- C_global slot source ---------------------------------
        # See docstring on `c_global_source` for behavior.
        # `siglip2_global` plumbs `feats["visual_global"]` (the SigLIP2
        # MAP-pooled + projection-head output, [B, proj_dim]) through a small
        # dedicated linear into d_model space. `mean_pool` keeps the legacy
        # behavior (mean of post-adapter visual_tokens).
        # (Exp2) PQ-style slot subspace partition. Default OFF.
        # Diagnosis (2026-07-27): the 6 slots share the full d_model and differ
        # only by routing weights, so their codes are 65% mutually redundant
        # (sum H(codon)=31.2 -> H(joint)=11.1 bits). Product-Quantization-style
        # disjoint subspaces make that redundancy structurally impossible:
        # slot m only sees dims [m*D/M, (m+1)*D/M).
        self.pq_slot_subspace: bool = bool(getattr(args, "pq_slot_subspace", False))
        self.codon_chunk_layernorm: bool = bool(getattr(args, "codon_chunk_layernorm", False))
        self.c_global_source: str = str(getattr(args, "c_global_source", "siglip2_global"))
        if self.c_global_source not in ("mean_pool", "siglip2_global"):
            raise ValueError(
                f"[model_siglip2] c_global_source must be one of "
                f"{{mean_pool, siglip2_global}}, got {self.c_global_source!r}"
            )
        if self.c_global_source == "siglip2_global":
            self.global_adapter = nn.Linear(self.proj_dim, self.d_model)
        else:
            self.global_adapter = None

        # SigLIP2 vision tower last_hidden_state dim != projection dim in
        # general, so the visual adapter input is the vision tower hidden_dim.
        self.adapter_type = str(getattr(args, "adapter_type", "mlp"))
        if self.adapter_type == "linear":
            self.visual_adapter = VisualLinearAdapter(
                in_dim=self.visual_encoder.hidden_dim,
                out_dim=self.d_model,
                residual=True,
            )
        else:
            self.visual_adapter = VisualAdapter(
                in_dim=self.visual_encoder.hidden_dim,   # H_v
                out_dim=self.d_model,                    # D
                hidden_dim=adapter_hidden,
                dropout=adapter_dropout,
                residual=True,                           # auto-disabled if H_v != D
            )
        # The text branch consumes the per-part global feature (already in
        # SigLIP2 projection space, dim == proj_dim) and adapts to d_model.
        #
        # SigLIP2's text encoder maps the six per-slot captions to nearly-
        # identical pooled vectors (measured cos sim ~0.88 across slots; per-
        # slot effective rank ~2.7), because it was contrastively trained on
        # image-caption pairs, not on fine-grained text discrimination. With
        # a single shared `text_adapter`, those near-identical inputs come
        # out near-identical too -> Sinkhorn router has nearly-identical
        # centroids -> codebook supervision is weak -> high cross-codebook
        # correlation.
        #
        # `--per_slot_text_adapter` swaps the single MLP for an
        # `nn.ModuleList` of six independent MLPs (one per slot). The slot-
        # specific weights have the structural freedom to push the six near-
        # identical inputs apart into distinct subspaces.
        self.per_slot_text_adapter = bool(getattr(args, "per_slot_text_adapter", False))
        # 1 global + 5 local = 6 codebook slots total
        _n_text_slots = 1 + NUM_LOCAL_PARTS
        if self.per_slot_text_adapter:
            if self.adapter_type == "linear":
                self.text_adapter = nn.ModuleList([
                    TextLinearAdapter(
                        in_dim=int(proj_dim),
                        out_dim=self.d_model,
                        residual=True,
                    )
                    for _ in range(_n_text_slots)
                ])
            else:
                self.text_adapter = nn.ModuleList([
                    TextAdapter(
                        in_dim=int(proj_dim),
                        out_dim=self.d_model,
                        hidden_dim=text_adapter_hidden,
                        dropout=adapter_dropout,
                        residual=True,
                    )
                    for _ in range(_n_text_slots)
                ])
        else:
            if self.adapter_type == "linear":
                self.text_adapter = TextLinearAdapter(
                    in_dim=int(proj_dim),
                    out_dim=self.d_model,
                    residual=True,
                )
            else:
                self.text_adapter = TextAdapter(
                    in_dim=int(proj_dim),                # D_proj
                    out_dim=self.d_model,                # D
                    hidden_dim=text_adapter_hidden,
                    dropout=adapter_dropout,
                    residual=True,                       # auto-disabled if D_proj != D
                )

        # v79b (#2.3): learnable per-codebook text prompts.
        # Adds a [M, proj_dim] parameter (one prompt vector per slot)
        # that is summed onto the cached text_part_raw before the
        # text_adapter. Gives each slot a learnable bias in text space.
        self.use_codebook_text_prompts: bool = bool(getattr(args, "use_codebook_text_prompts", False))
        if self.use_codebook_text_prompts:
            scale = float(getattr(args, "codebook_text_prompt_init_scale", 0.02))
            self.codebook_text_prompts = nn.Parameter(
                torch.randn(_n_text_slots, int(proj_dim)) * scale
            )
        else:
            self.codebook_text_prompts = None

        # ---------- v113: text-embed pre-transform (anisotropy mitigation) -----
        # Applied to `raw = feats["text_part_raw"]` BEFORE the text_adapter.
        # See config.py for the mode choices. partial_whiten loads a
        # precomputed (mu, U, S) bundle and builds W = U diag((S+eps)^-gamma) U^T.
        self.text_embed_transform: str = str(
            getattr(args, "text_embed_transform", "none")
        )
        self.residualize_visual_for_routing: bool = bool(
            getattr(args, "residualize_visual_for_routing", False)
        )
        # Foreground-text mask: keep only top-K% patches by cosine similarity to
        # C_global text embedding before Sinkhorn routing. Background patches
        # get visual_mask=0 so they are excluded from all 6 codebook updates.
        # Default None = disabled. CUB / fine-grained single-object datasets:
        # try 0.4-0.6. Multi-object scenes: leave disabled.
        self.foreground_text_mask_topk_ratio = getattr(
            args, "foreground_text_mask_topk_ratio", None
        )
        if self.foreground_text_mask_topk_ratio is not None:
            self.foreground_text_mask_topk_ratio = float(
                self.foreground_text_mask_topk_ratio
            )
        # v175: text source for foreground mask (global vs local-pooled).
        self.foreground_text_mask_source: str = str(
            getattr(args, "foreground_text_mask_source", "global")
        )
        # A2 ablation: disable text supervision entirely (visual-only anchors).
        self.disable_text_supervision: bool = bool(
            getattr(args, "disable_text_supervision", False)
        )
        # v185: bidirectional token pruning (visual + text).
        self.bidirectional_token_prune: bool = bool(
            getattr(args, "bidirectional_token_prune", False)
        )
        self.bidirectional_token_prune_visual_ratio: float = float(
            getattr(args, "bidirectional_token_prune_visual_ratio", 0.5)
        )
        self.bidirectional_token_prune_text_ratio: float = float(
            getattr(args, "bidirectional_token_prune_text_ratio", 0.5)
        )
        self.bidirectional_token_prune_mode: str = str(
            getattr(args, "bidirectional_token_prune_mode", "legacy")
        )
        if self.bidirectional_token_prune_mode not in (
            "legacy", "mutual_dual_softmax", "mutual_consensus_residual",
        ):
            raise ValueError(
                "bidirectional_token_prune_mode must be 'legacy', "
                "'mutual_dual_softmax', or 'mutual_consensus_residual', "
                f"got {self.bidirectional_token_prune_mode!r}"
            )
        self.bidirectional_prune_only: bool = bool(
            getattr(args, "bidirectional_prune_only", False)
        )
        if self.bidirectional_prune_only and (
            not self.bidirectional_token_prune
            or self.bidirectional_token_prune_mode != "mutual_consensus_residual"
        ):
            raise ValueError(
                "--bidirectional_prune_only requires --bidirectional_token_prune "
                "and --bidirectional_token_prune_mode mutual_consensus_residual"
            )
        # v184: inference routing centroid source (codebook_mean vs text_prototype).
        self.eval_routing_mode: str = str(
            getattr(args, "eval_routing_mode", "codebook_mean")
        )
        self.text_prototype_ema_decay: float = float(
            getattr(args, "text_prototype_ema_decay", 0.999)
        )
        self.routing_cls_verified_consensus_mask: bool = bool(
            getattr(args, "routing_cls_verified_consensus_mask", False)
        )
        # Register EMA text prototype buffer (5 local slots x d_model).
        # Initialized as zeros. Updated during training when text_part_tokens
        # available. Used at inference when eval_routing_mode == text_prototype.
        _NUM_LOCAL_PARTS = NUM_LOCAL_PARTS
        _d_model_arg = getattr(args, "d_model", None)
        _dm = int(_d_model_arg) if _d_model_arg is not None else 768
        self.register_buffer(
            "text_prototype_ema",
            torch.zeros(_NUM_LOCAL_PARTS, _dm),
        )
        self.register_buffer(
            "_text_prototype_initialized",
            torch.zeros(1, dtype=torch.bool),
        )
        if self.routing_cls_verified_consensus_mask:
            self.register_buffer(
                "cls_mask_text_prototype_ema",
                torch.zeros(_NUM_LOCAL_PARTS, int(self.proj_dim)),
            )
            self.register_buffer(
                "_cls_mask_text_prototype_initialized",
                torch.zeros(1, dtype=torch.bool),
            )
        # v176a: soft text-evidence routing prior. Instead of hard-pruning
        # background tokens, lower the Sinkhorn cost for visual patches that
        # already match the routed text/codebook centroid. This is a routing
        # bias only; the Wasserstein loss still reports the original cost.
        self.routing_text_evidence_beta: float = float(
            getattr(args, "routing_text_evidence_beta", 0.0)
        )
        self.routing_text_evidence_warmup_epochs: int = int(
            getattr(args, "routing_text_evidence_warmup_epochs", 0)
        )
        self.routing_text_evidence_keep_ratio: float = float(
            getattr(args, "routing_text_evidence_keep_ratio", 1.0)
        )
        self.routing_text_evidence_penalty: float = float(
            getattr(args, "routing_text_evidence_penalty", 0.0)
        )
        self.routing_token_ot_evidence: bool = bool(
            getattr(args, "routing_token_ot_evidence", False)
        )
        self.routing_token_ot_beta: float = float(
            getattr(args, "routing_token_ot_beta", 0.0)
        )
        self.routing_token_ot_eps: float = float(
            getattr(args, "routing_token_ot_eps", 0.05)
        )
        self.routing_token_ot_topk_text: int = int(
            getattr(args, "routing_token_ot_topk_text", 8)
        )
        self.routing_token_ot_warmup_epochs: int = int(
            getattr(args, "routing_token_ot_warmup_epochs", 0)
        )
        if self.routing_text_evidence_beta < 0.0:
            raise ValueError(
                f"routing_text_evidence_beta must be >= 0, got {self.routing_text_evidence_beta}"
            )
        if self.routing_text_evidence_warmup_epochs < 0:
            raise ValueError(
                "routing_text_evidence_warmup_epochs must be >= 0, got "
                f"{self.routing_text_evidence_warmup_epochs}"
            )
        if not (0.0 < self.routing_text_evidence_keep_ratio <= 1.0):
            raise ValueError(
                "routing_text_evidence_keep_ratio must be in (0, 1], got "
                f"{self.routing_text_evidence_keep_ratio}"
            )
        if self.routing_text_evidence_penalty < 0.0:
            raise ValueError(
                "routing_text_evidence_penalty must be >= 0, got "
                f"{self.routing_text_evidence_penalty}"
            )
        if self.routing_token_ot_beta < 0.0:
            raise ValueError(
                "routing_token_ot_beta must be >= 0, got "
                f"{self.routing_token_ot_beta}"
            )
        if self.routing_token_ot_eps <= 0.0:
            raise ValueError(
                "routing_token_ot_eps must be > 0, got "
                f"{self.routing_token_ot_eps}"
            )
        if self.routing_token_ot_topk_text <= 0:
            raise ValueError(
                "routing_token_ot_topk_text must be >= 1, got "
                f"{self.routing_token_ot_topk_text}"
            )
        if self.routing_token_ot_warmup_epochs < 0:
            raise ValueError(
                "routing_token_ot_warmup_epochs must be >= 0, got "
                f"{self.routing_token_ot_warmup_epochs}"
            )
        self.route_global_text: bool = bool(getattr(args, "route_global_text", False))
        self.routed_cls_add_gamma: float = float(getattr(args, "routed_cls_add_gamma", 0.0))
        self.routed_cls_add_scope: str = str(getattr(args, "routed_cls_add_scope", "local"))
        if self.routed_cls_add_scope not in ("local", "all"):
            raise ValueError(
                f"routed_cls_add_scope must be 'local' or 'all', got {self.routed_cls_add_scope!r}"
            )
        # v122: C0-orthogonal local quantization. C_global keeps the coarse
        # semantic axis; local codebooks quantize the residual concept.
        # Sequential cross-slot residual pooling (2026-07-21). See the pooling
        # site for the diagnosis; default off so every existing run is unchanged.
        self.slot_sequential_residual: bool = bool(
            getattr(args, "slot_sequential_residual", False))
        self.slot_seq_residual_gamma: float = float(
            getattr(args, "slot_seq_residual_gamma", 1.0))
        self.local_residual_quant: bool = bool(getattr(args, "local_residual_quant", False))
        self.local_residual_gamma: float = float(getattr(args, "local_residual_gamma", 1.0))
        self.local_residual_detach_global: bool = bool(
            getattr(args, "local_residual_detach_global", True)
        )
        self.local_residual_text: bool = bool(getattr(args, "local_residual_text", False))
        if self.local_residual_gamma < 0.0:
            raise ValueError(
                f"local_residual_gamma must be >= 0, got {self.local_residual_gamma}"
            )
        self._text_whiten_ready: bool = False
        if self.text_embed_transform in ("partial_whiten", "global_residual_whiten"):
            _npz_path = getattr(args, "text_whiten_npz", None)
            if _npz_path is None or not os.path.exists(str(_npz_path)):
                raise SystemExit(
                    f"[model_siglip2] --text_embed_transform partial_whiten "
                    f"requires --text_whiten_npz (got {_npz_path!r}). "
                    f"Generate via scripts/build_text_whiten_matrix.py."
                )
                # NOTE: `os` not imported at module top in current file;
                # we import lazily below if needed.
            # This matrix is fitted before training, so it must have seen the
            # optimization-train rows only. All four rebuilt v6prov caches were
            # first written with `leakage_free_fit: false` -- fitted over every
            # cache row, which includes the validation rows that select the
            # epoch and the query/DB rows that ARE the reported number.
            from dna_utils.cache_provenance import require_leakage_free_whitening
            require_leakage_free_whitening(str(_npz_path))
            import numpy as _np
            _b = _np.load(str(_npz_path))
            _mu_np = _np.asarray(_b["mu"], dtype=_np.float32)               # [D]
            _U_np  = _np.asarray(_b["U"],  dtype=_np.float32)               # [D, D]
            _S_np  = _np.asarray(_b["S"],  dtype=_np.float32)               # [D]
            _gamma = float(getattr(args, "text_whiten_gamma", 0.25))
            _eps   = float(getattr(args, "text_whiten_eps",   1e-5))
            _inv_pow = (_S_np + _eps) ** (-_gamma)                          # [D]
            _W_np = _U_np @ _np.diag(_inv_pow) @ _U_np.T                    # [D, D]
            self.register_buffer(
                "text_whiten_mu",
                torch.from_numpy(_mu_np).to(torch.float32),
                persistent=False,
            )
            self.register_buffer(
                "text_whiten_W",
                torch.from_numpy(_W_np.astype(_np.float32)),
                persistent=False,
            )
            self._text_whiten_ready = True
            print(f"[model_siglip2] v113 partial whitening loaded: "
                  f"D={_mu_np.shape[0]}, gamma={_gamma:.3f}, eps={_eps:.1e}")

        # ---------- Optional: visual-cross-attention text pooling (Option B / v22b)
        # When `--use_text_token_attention` is set, the model expects per-image
        # cached TOKEN-level text features (`cached_text_tokens [B, 6, T, D_proj]`
        # + `cached_text_token_mask [B, 6, T]`) and replaces the static pooled
        # text embedding with one produced by cross-attending visual patches
        # over the slot-specific text tokens. This gives a visually-grounded
        # text representation that varies per image while preserving slot
        # identity through slot-specific K/V.
        self.use_text_token_attention = bool(getattr(args, "use_text_token_attention", False))
        if self.use_text_token_attention:
            n_heads = int(getattr(args, "text_attn_num_heads", 4))
            self.text_token_attn = nn.MultiheadAttention(
                embed_dim=self.d_model, num_heads=n_heads,
                batch_first=True, dropout=0.0,
            )
            self.text_token_ln = nn.LayerNorm(self.d_model)

        # v162: grounded text routing (Stage-2 top-k token pruning)
        self.grounded_text_routing     = bool(getattr(args, "grounded_text_routing",     False))
        self.grounded_text_k_t         = int (getattr(args, "grounded_text_k_t",         5))
        self.grounded_text_eps         = float(getattr(args, "grounded_text_eps",        0.05))
        self.grounded_text_stage1_sg   = bool(getattr(args, "grounded_text_stage1_sg",   True))
        self.grounded_text_skip_global = bool(getattr(args, "grounded_text_skip_global", True))
        if self.grounded_text_routing:
            # Project text-token cache (already in CLIP D=512 space) through
            # the SHARED text_adapter so it lands in d_model space — the
            # same projection the global text_part_tokens receive — keeping
            # Stage-2 outputs comparable to baseline embeds in distribution.
            self.grounded_text_ln = nn.LayerNorm(self.d_model)

        # Soft visual-grounded text pooling. Unlike the v162 path, this keeps
        # every valid text token and uses detached local OT mass to aggregate
        # visual-query cross-attention. It refines the text supervision after
        # routing without feeding the result back into the same OT pass.
        self.soft_visual_grounded_text_pool = bool(
            getattr(args, "soft_visual_grounded_text_pool", False)
        )
        if self.soft_visual_grounded_text_pool and self.grounded_text_routing:
            raise ValueError(
                "--soft_visual_grounded_text_pool and "
                "--grounded_text_routing are mutually exclusive"
            )
        if self.soft_visual_grounded_text_pool:
            n_heads = int(getattr(args, "text_attn_num_heads", 4))
            self.soft_grounded_text_attn = nn.MultiheadAttention(
                embed_dim=self.d_model,
                num_heads=n_heads,
                batch_first=True,
                dropout=0.0,
            )
            self.soft_grounded_text_ln = nn.LayerNorm(self.d_model)

        # v192: parameter-free alternative to the failed learnable MHA pool.
        # Each zero-initialized ReZero gate interpolates from the unchanged
        # baseline text slot toward an OT-grounded all-token cosine pool.
        self.cosine_visual_grounded_text_pool = bool(
            getattr(args, "cosine_visual_grounded_text_pool", False)
        )
        if self.cosine_visual_grounded_text_pool and (
            self.grounded_text_routing or self.soft_visual_grounded_text_pool
        ):
            raise ValueError(
                "--cosine_visual_grounded_text_pool is mutually exclusive "
                "with --grounded_text_routing and "
                "--soft_visual_grounded_text_pool"
            )
        if self.cosine_visual_grounded_text_pool:
            self.cosine_grounded_text_gate = nn.Parameter(
                torch.zeros(NUM_LOCAL_PARTS)
            )                                                       # [5]
        else:
            self.register_parameter("cosine_grounded_text_gate", None)

        # Semantic-instance separation for the dominant visual-token CIBHash
        # objective. The six heads are discarded at inference; VQ and DNA use
        # the unprojected semantic_visual_tokens exactly as before.
        self.cibhash_visual_projection_head = bool(
            getattr(args, "cibhash_visual_projection_head", False)
        )
        if self.cibhash_visual_projection_head:
            if str(getattr(args, "cibhash_ntxent_source", "continuous_code")) != "visual_token":
                raise ValueError(
                    "--cibhash_visual_projection_head requires "
                    "--cibhash_ntxent_source visual_token"
                )
            self.cibhash_visual_projectors = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.d_model, self.d_model),
                    nn.GELU(),
                    nn.Linear(self.d_model, self.d_model),
                )
                for _ in range(NUM_SEMANTIC_PARTS)
            ])
        else:
            self.cibhash_visual_projectors = None

        # ---------- Sinkhorn router (5 local parts) ----------------------
        self.num_semantic_parts: int = int(
            getattr(args, "num_semantic_parts", NUM_SEMANTIC_PARTS)
        )
        if self.num_semantic_parts != NUM_SEMANTIC_PARTS:
            print(
                f"[model_siglip2] WARNING: num_semantic_parts={self.num_semantic_parts}"
                f" differs from spec value {NUM_SEMANTIC_PARTS}. Make sure your"
                f" part_input_ids second axis matches."
            )
        # router_type: 'sinkhorn' (legacy, balanced OT) or 'attention'
        # (per-part softmax over patches; no marginal balance — lets text-irrelevant
        # parts attend weakly when image lacks that content).
        self.router_type: str = str(getattr(args, "router_type", "sinkhorn"))
        if self.router_type == "sinkhorn":
            self.router = SemanticSinkhornRouter(
                epsilon=float(getattr(args, "sinkhorn_temperature", 0.05)),
                num_iters=int(getattr(args, "sinkhorn_iters", 20)),
                cost_mode=str(getattr(args, "sinkhorn_cost_mode", "one_minus_cos")),
            )
        elif self.router_type == "attention":
            self.router = SemanticAttentionRouter(
                temperature=float(getattr(args, "attention_router_temperature", 0.1)),
            )
        elif self.router_type == "slot":
            # v107a-slot: Slot Attention (Locatello et al. NeurIPS 2020).
            # M textual semantic embeddings are used as slot init;
            # competitive softmax over slots + GRU/MLP slot updates.
            self.router = SemanticSlotAttentionRouter(
                d_model=int(self.d_model),
                n_iters=int(getattr(args, "slot_attention_iters", 3)),
            )
        elif self.router_type == "cluster_attn":
            # v141: DiVT-inspired soft cluster + masked cross-attention.
            # Operates on RAW visual patches (BEFORE visual_adapter), and
            # returns M=6 semantic tokens directly in the encoder's hidden
            # dim. Pair with --visual_adapter_after_router so the adapter
            # gets applied to the M output tokens, projecting them to
            # d_model for the codebook lookup.
            from models.cluster_attention_router import ClusterAttentionRouter
            self.router = ClusterAttentionRouter(
                num_parts=NUM_SEMANTIC_PARTS,
                d_model=int(self.d_model),
                num_heads=int(getattr(args, "cluster_attn_heads", 4)),
                mlp_ratio=float(getattr(args, "cluster_attn_mlp_ratio", 4.0)),
                sinkhorn_eps=float(getattr(args, "cluster_attn_sinkhorn_eps", 0.1)),
                sinkhorn_iters=int(getattr(args, "cluster_attn_sinkhorn_iters", 3)),
                pool_temperature=float(getattr(args, "cluster_attn_pool_temperature", 0.3)),
            )
        elif self.router_type == "cross_attn":
            # v146: text-as-query cross-attention router. Uses the Sinkhorn
            # router in parallel as a baseline for residual blending during
            # warmup (alpha annealed 0 -> 1 over warmup_epochs).
            from models.text_cross_attention_router import TextCrossAttentionRouter
            self.cross_attn_router = TextCrossAttentionRouter(
                d_model=int(self.d_model),
                num_parts=NUM_SEMANTIC_PARTS,
                num_heads=int(getattr(args, "cross_attn_heads", 4)),
                dropout=float(getattr(args, "cross_attn_dropout", 0.1)),
                temp_init=float(getattr(args, "cross_attn_temp_init", 0.2)),
                temp_final=float(getattr(args, "cross_attn_temp_final", 0.07)),
                near_identity_scale=float(getattr(args, "cross_attn_near_identity_scale", 0.1)),
            )
            # Build the Sinkhorn router as a parallel baseline (used during
            # warmup blending). self.router remains the Sinkhorn router so
            # the existing routing code path executes unmodified.
            self.router = SemanticSinkhornRouter(
                epsilon=float(getattr(args, "sinkhorn_temperature", 0.05)),
                num_iters=int(getattr(args, "sinkhorn_iters", 20)),
                cost_mode=str(getattr(args, "sinkhorn_cost_mode", "one_minus_cos")),
            )
            self.cross_attn_warmup_epochs = int(getattr(args, "cross_attn_warmup_epochs", 20))
            self.cross_attn_alpha_final = float(getattr(args, "cross_attn_alpha_final", 1.0))
        else:
            raise ValueError(
                f"[model_siglip2] router_type must be 'sinkhorn' / 'attention' / "
                f"'slot' / 'cluster_attn' / 'cross_attn', got {self.router_type!r}"
            )
        # v141: visual_adapter relocation flag.
        self.visual_adapter_after_router: bool = bool(
            getattr(args, "visual_adapter_after_router", False)
        )
        # v33a: epsilon annealing for Sinkhorn router. When both init+final
        # are set, the effective epsilon is cosine-interpolated from init
        # (epoch 0) to final (last epoch). Smaller epsilon -> sharper
        # routing. v33b: top-k mask over per-patch part assignment (k>=M
        # disables; k=1 = hard argmax).
        self.sinkhorn_epsilon_init  = getattr(args, "sinkhorn_epsilon_init",  None)
        self.sinkhorn_epsilon_final = getattr(args, "sinkhorn_epsilon_final", None)
        self.routing_topk           = getattr(args, "routing_topk",           None)
        self.routing_topp           = getattr(args, "routing_topp",           None)
        # v80/v81: confidence-adaptive sparse routing. Default-off, passed
        # through to SemanticSinkhornRouter only when enabled.
        self.routing_ambiguity_topk      = bool(getattr(args, "routing_ambiguity_topk", False))
        self.routing_ambiguity_threshold = float(getattr(args, "routing_ambiguity_threshold", 0.6))
        self.routing_ambiguity_k         = int(getattr(args, "routing_ambiguity_k", 2))
        self.routing_adaptive_topp       = bool(getattr(args, "routing_adaptive_topp", False)) \
            and not bool(getattr(args, "no_routing_adaptive_topp", False))
        self.routing_adaptive_topp_min   = float(getattr(args, "routing_adaptive_topp_min", 0.5))
        self.routing_adaptive_topp_max   = float(getattr(args, "routing_adaptive_topp_max", 0.9))
        self.routing_adaptive_topp_entropy = bool(getattr(args, "routing_adaptive_topp_entropy", False))
        self.routing_specificity_marginal = bool(
            getattr(args, "routing_specificity_marginal", False)
        )
        self.routing_centered_consensus_mask = bool(
            getattr(args, "routing_centered_consensus_mask", False)
        )
        _consensus_modes = sum(bool(v) for v in (
            self.routing_specificity_marginal,
            self.routing_centered_consensus_mask,
            self.routing_cls_verified_consensus_mask,
        ))
        if _consensus_modes > 1:
            raise ValueError(
                "specificity marginal, centered consensus mask, and "
                "CLS-verified consensus mask are mutually exclusive"
            )
        if (
            self.routing_specificity_marginal
            or self.routing_centered_consensus_mask
            or self.routing_cls_verified_consensus_mask
        ) and self.router_type != "sinkhorn":
            raise ValueError(
                "specificity marginal / centered consensus mask requires "
                "--router_type sinkhorn"
            )
        # v84a: perplexity-rounded top-k routing (zero hand-tuned thresholds).
        # k_n = ceil(M^H_norm(P_n)). Mutually exclusive with adaptive_topp.
        self.routing_perplexity_topk     = bool(getattr(args, "routing_perplexity_topk", False))
        # v91: text-to-DNA-hash matching. When lambda_text_hash > 0, the
        # forward pass *also* runs the text_part_tokens through the shared
        # quantizer + codon_heads to produce a text-derived continuous_code,
        # which the loss matches to the image-derived continuous_code via MSE.
        self.lambda_text_hash            = float(getattr(args, "lambda_text_hash", 0.0))
        # v109: surfaced on the model so the text-codon path can be activated
        # by the InfoNCE-form visual-textual contrastive loss alone (without
        # requiring the MSE-form lambda_text_hash > 0).
        self.lambda_text_hash_ntxent     = float(getattr(args, "lambda_text_hash_ntxent", 0.0))
        # Counterfactual foil path is train-only and loss-only: foils never
        # become routing centroids or quantizer EMA observations.
        self.text_hash_counterfactual_weight = float(
            getattr(args, "text_hash_counterfactual_weight", 0.0)
        )
        # v93: per-codebook cross-modal codeword InfoNCE. When > 0, the same
        # text path used by lambda_text_hash is activated (text_part_tokens
        # through EMA-disabled quantizer) and the resulting text codeword is
        # exposed as `text_quantized_tokens`. The loss module then runs an
        # InfoNCE between visual `quantized_tokens` and `text_quantized_tokens`
        # per codebook. Single text-path forward serves both v91 and v93.
        self.lambda_cw_xmodal            = float(getattr(args, "lambda_cw_xmodal", 0.0))
        # v160 (Uni-Code Eq.8): cross-modal commitment weight.
        self.lambda_xmodal_commit        = float(getattr(args, "lambda_xmodal_commit", 0.0))
        # v161 (Uni-Code MM-EMA, simplified): bi-modal EMA codebook update flag.
        self.mm_ema                      = bool(getattr(args, "mm_ema", False))
        self.lambda_codeword_text_proto  = float(getattr(args, "lambda_codeword_text_proto", 0.0))
        # v85: expert-choice-inspired codebook-side token filtering.
        self.routing_codebook_choice = bool(getattr(args, "routing_codebook_choice", False))
        self.routing_codebook_choice_capacity = float(getattr(args, "routing_codebook_choice_capacity", 1.5))
        self.routing_codebook_choice_beta = float(getattr(args, "routing_codebook_choice_beta", 1.0))
        self.routing_codebook_choice_warmup_epochs = int(
            getattr(args, "routing_codebook_choice_warmup_epochs", 0)
        )
        # v79c (#4.1): hard routing via Gumbel-Softmax (one-hot per patch)
        self.routing_hard           = bool(getattr(args, "routing_hard",      False))
        self.routing_hard_tau       = float(getattr(args, "routing_hard_tau", 1.0))
        # v79d (#1.3-lite): per-codebook learnable attention pool over
        # cached visual_tokens. Replaces local_semantic_visual_tokens (the
        # Sinkhorn-pooled output) for cb1..5 with per-cb attention pool.
        # cb0 (global) still uses visual_global. ot_cost from Sinkhorn is
        # zeroed (wasserstein loss becomes inactive for this experiment).
        self.use_per_cb_attn_pool   = bool(getattr(args, "use_per_cb_attn_pool", False))
        self.per_cb_attn_pool_temp  = float(getattr(args, "per_cb_attn_pool_temp", 1.0))
        if self.use_per_cb_attn_pool:
            self.per_cb_attn_queries = nn.Parameter(
                torch.randn(NUM_LOCAL_PARTS, self.d_model) / math.sqrt(self.d_model)
            )
        else:
            self.per_cb_attn_queries = None
        # v55: UOT KL marginal penalties (None = balanced Sinkhorn)
        self.sinkhorn_lambda_a      = getattr(args, "sinkhorn_lambda_a",      None)
        self.sinkhorn_lambda_b      = getattr(args, "sinkhorn_lambda_b",      None)
        # v56: optional learnable null/background centroid
        self.use_null_centroid      = bool(getattr(args, "use_null_centroid", False))
        # 2026-09-20, off-protocol branch arch-exp-2026-09; all default off.
        self.routing_mass_alpha     = float(getattr(args, "routing_mass_alpha", 0.0) or 0.0)   # (a-1)
        self.null_free_marginal     = bool(getattr(args, "null_free_marginal", False))          # (a-2)
        self.lambda_mec             = float(getattr(args, "lambda_mec", 0.0) or 0.0)           # (b-1)
        self.vq_bypass_epochs       = int(getattr(args, "vq_bypass_epochs", 0) or 0)           # (c-1)
        # arch-exp-3 (P2): per-image axis-deviation representation.
        self.axis_center            = str(getattr(args, "axis_center", "none") or "none")
        self.total_epochs           = int(getattr(args, "epoch", 60))
        # D2: the epsilon anneal has its own horizon. Using total_epochs
        # meant `-e N+1` compressed the LR cosine as well, so choosing N
        # retuned three things at once. Unset falls back to total_epochs.
        _sk_h = getattr(args, "sinkhorn_schedule_horizon", None)
        self.sinkhorn_schedule_horizon = int(_sk_h) if _sk_h else self.total_epochs
        # mutable per-step state set by trainer via set_current_epoch()
        self._current_epoch: int = 0

        # ---------- codebook quantizer -----------------------------------
        _cfg_parts = int(getattr(args, "num_semantic_parts", NUM_SEMANTIC_PARTS))
        if _cfg_parts != NUM_SEMANTIC_PARTS:
            raise ValueError(
                f"--num_semantic_parts={_cfg_parts} but the module was imported with "
                f"GDNA_NUM_SEMANTIC_PARTS -> {NUM_SEMANTIC_PARTS}. Set the env var "
                f"BEFORE launching python; args.txt would otherwise not describe "
                f"the model that actually ran.")
        self.num_codebooks: int = int(getattr(args, "num_codebooks", NUM_SEMANTIC_PARTS))
        self.codebook_size: int = int(getattr(args, "codebook_size", 64))
        if self.num_codebooks != NUM_SEMANTIC_PARTS:
            print(
                f"[model_siglip2] WARNING: num_codebooks={self.num_codebooks}"
                f" differs from spec value {NUM_SEMANTIC_PARTS}."
            )
        self.quantizer = SemanticCodebookQuantizer(
            num_codebooks=self.num_codebooks,
            codebook_size=self.codebook_size,
            d_model=self.d_model,
            update_mode=str(getattr(args, "codebook_update", "gradient")),
            ema_decay=float(getattr(args, "codebook_ema_decay", 0.99)),
            ema_eps=float(getattr(args, "codebook_ema_eps", 1e-5)),
            revive_dead=bool(getattr(args, "codebook_revive", True)),
            revive_threshold=float(getattr(args, "codebook_revive_threshold", 0.01)),
            revive_every=int(getattr(args, "codebook_revive_every", 50)),
            repel_strength=float(getattr(args, "codebook_repel_strength", 0.0)),
            repel_sigma_factor=float(getattr(args, "codebook_repel_sigma_factor", 0.5)),
            repel_every=int(getattr(args, "codebook_repel_every", 1)),
            distance_mode=str(getattr(args, "vq_distance_mode", "euclidean")),
            K_max=int(getattr(args, "codebook_K_max", 0)),
            share_codebook=bool(getattr(args, "share_codebook", False)),
            freeze_after_epoch=int(getattr(args, "codebook_freeze_after_epoch", -1)),
        )

        # ---------- gated global addition --------------------------------
        # Per-local-codebook trainable gate logits. We initialize to -3.0 so
        # sigmoid(-3.0) ≈ 0.0474 and the global codeword is added very softly
        # at the start of training. There are 5 gates -- one per local codebook.
        self.use_stop_grad_global: bool = bool(getattr(args, "use_stop_grad_global", True))
        # v23b: optionally skip the entire `q_local + sigmoid(alpha) * q_global`
        # blending step before the codon heads. When True, each local codon
        # head sees the local codeword in isolation (no C_global injection).
        self.disable_global_gate: bool = bool(getattr(args, "disable_global_gate", False))
        # v138: source of codon_head input.
        #   "quantized" (default): legacy VQ output (post-gate quantized_tokens)
        #   "routed": raw router weighted-sum vectors (semantic_visual_tokens
        #             passes through gate too but bypasses VQ codebook lookup).
        self.codon_input_source: str = str(getattr(args, "codon_input_source", "quantized"))
        if self.codon_input_source not in ("quantized", "routed"):
            raise ValueError(
                f"[model_siglip2] codon_input_source must be 'quantized' or "
                f"'routed', got {self.codon_input_source!r}"
            )
        gate_init = float(getattr(args, "global_gate_init_logit", -3.0))
        self.global_gate_logits = nn.Parameter(
            torch.full((NUM_LOCAL_PARTS,), gate_init, dtype=torch.float32)
        )

        # v56: learnable null/background centroid. Acts as an extra "reject"
        # part the Sinkhorn router can send uninformative patches to. Init
        # to small random vector so it doesn't dominate any specific direction.
        if self.use_null_centroid:
            d_for_null = int(getattr(args, "d_model", 768) or 768)
            self.null_centroid = nn.Parameter(
                torch.randn(d_for_null, dtype=torch.float32) * 0.02
            )
        else:
            self.register_parameter("null_centroid", None)

        # (a-1) statistics for the content-dependent slot mass. Persistent,
        # so a checkpoint carries the exact prototypes it was trained with and
        # deployment computes the same targets. Filled before training by
        # dna_utils.arch_exp.init_routing_mass_stats.
        if self.routing_mass_alpha > 0.0:
            if not (0.0 < self.routing_mass_alpha <= 1.0):
                raise ValueError("--routing_mass_alpha must be in (0, 1]")
            self.register_buffer("routing_mass_proto",
                                 torch.zeros(int(NUM_LOCAL_PARTS), int(self.proj_dim)))
            self.register_buffer("routing_mass_center", torch.zeros(int(NUM_LOCAL_PARTS)))
            self.register_buffer("routing_mass_tau", torch.ones(()))
            self.register_buffer("routing_mass_ready", torch.zeros((), dtype=torch.bool))
        if self.null_free_marginal and not self.use_null_centroid:
            raise ValueError("--null_free_marginal requires --use_null_centroid")
        # (b-1) masked entity completion (OVSegmentor port), training only.
        if self.lambda_mec > 0.0:
            import json as _json
            from models.mec_head import MaskedEntityCompletion
            _mec_dir = getattr(args, "mec_cache_dir", None)
            if not _mec_dir:
                raise ValueError("--lambda_mec > 0 requires --mec_cache_dir")
            _mec_meta = _json.load(open(os.path.join(_mec_dir, "meta.json")))
            self.mec = MaskedEntityCompletion(
                slot_dim=int(self.d_model), text_dim=int(_mec_meta["text_dim"]),
                temperature=float(getattr(args, "mec_temperature", 0.07)),
                seed=int(getattr(args, "random_seed", 0) or 0),
            )
            self.mec.load_cache(_mec_dir)
        else:
            self.mec = None

        # ---------- per-codebook codon heads -----------------------------
        # 6 SEPARATE CodonHead instances (no weight sharing across codebooks).
        # Each maps [B, D] -> [B, L, 4] (L codon positions × 4 base classes).
        # v122: L = num_codons_per_codebook (default 3, optionally 4 for the
        # 48-bit DNA spec that frees K up to 4^4 = 256 codons).
        self.num_codons_per_codebook: int = int(
            getattr(args, "num_codons_per_codebook", 3)
        )
        if self.num_codons_per_codebook not in (3, 4):
            raise ValueError(
                f"[model_siglip2] num_codons_per_codebook must be 3 or 4; "
                f"got {self.num_codons_per_codebook}."
            )
        if self.d_model % self.num_codons_per_codebook != 0:
            raise ValueError(
                f"[model_siglip2] d_model must be divisible by "
                f"num_codons_per_codebook={self.num_codons_per_codebook}; "
                f"got d_model={self.d_model}."
            )
        # Differentiable hard-code path config (Gumbel-Softmax STE by default).
        self.use_gumbel_softmax: bool = bool(getattr(args, "use_gumbel_softmax", True))
        self.gumbel_tau: float       = float(getattr(args, "gumbel_tau", 1.0))
        # v62 (Option A): residual-conditioned codon head
        self.codon_residual_gamma: float = float(getattr(args, "codon_residual_gamma", 0.0))
        _use_residual_codon = self.codon_residual_gamma > 0.0
        self.codon_head_hidden_dim: int = int(getattr(args, "codon_head_hidden_dim", 0))
        # v66 text-anchored prototype head (mutually exclusive with codon_head_hidden_dim)
        self.codon_text_anchor: bool = bool(getattr(args, "codon_text_anchor", False))
        self.codon_anchor_temperature: float = float(getattr(args, "codon_anchor_temperature", 0.1))
        # v69a / v69b / v71a (Exp 1, 2, 5)
        self.codon_position_specific_head: bool = bool(getattr(args, "codon_position_specific_head", False))
        self.codon_chunk_interleave: bool = bool(getattr(args, "codon_chunk_interleave", False))
        self.codon_position_residual_adapter: bool = bool(getattr(args, "codon_position_residual_adapter", False))
        self.codon_residual_split: bool        = bool(getattr(args, "codon_residual_split", False))
        self.codon_residual_gate: bool         = bool(getattr(args, "codon_residual_gate", False))
        # v105: full-input codon decoder (Linear(d_model, 12) instead of Linear(chunk, 4))
        self.codon_full_linear: bool           = bool(getattr(args, "codon_full_linear", False))
        # v106: enable codeword-level codon decoder output for bijection loss
        # (Sinkhorn-OT / aggregated entropy / pairwise). True iff any of the
        # lambda_codeword_codon_* flags > 0.
        _lam_sinkhorn = float(getattr(args, "lambda_codeword_codon_sinkhorn", 0.0))
        _lam_agg_ent  = float(getattr(args, "lambda_codeword_codon_agg_ent", 0.0))
        _lam_pairw    = float(getattr(args, "lambda_codeword_codon_pairwise", 0.0))
        _lam_txt_clu  = float(getattr(args, "lambda_text_cluster_codon_ot", 0.0))
        self.lambda_text_codon_rel = float(getattr(args, "lambda_text_codon_rel", 0.0))
        _lam_hcc      = float(getattr(args, "lambda_hierarchical_cluster_codon", 0.0))  # v112
        self._compute_codeword_codon_logits: bool = (
            (_lam_sinkhorn + _lam_agg_ent + _lam_pairw
             + _lam_txt_clu + self.lambda_text_codon_rel + _lam_hcc) > 0.0
        )
        # 2026-07-30: `position_specific_head` may be restricted to a subset of
        # slots. Empty spec -> apply `codon_position_specific_head` to every
        # slot (legacy behaviour, bit-identical).
        _psh_spec = str(getattr(args, "codon_position_specific_head_slots", "") or "").strip()
        _psh_slots = ({int(s) for s in _psh_spec.split(",") if s.strip() != ""}
                      if _psh_spec else None)
        if _psh_slots is not None and not _psh_slots <= set(range(self.num_codebooks)):
            raise ValueError(
                f"--codon_position_specific_head_slots={_psh_spec!r} out of range "
                f"for num_codebooks={self.num_codebooks}")
        self.codon_heads = nn.ModuleList(
            [
                CodonHead(
                    self.d_model,
                    use_gumbel_softmax=self.use_gumbel_softmax,
                    gumbel_tau=self.gumbel_tau,
                    use_residual=_use_residual_codon,
                    head_hidden_dim=self.codon_head_hidden_dim,
                    use_text_anchor=self.codon_text_anchor,
                    anchor_temperature=self.codon_anchor_temperature,
                    position_specific_head=(
                        self.codon_position_specific_head if _psh_slots is None
                        else (_m in _psh_slots)),
                    position_residual_adapter=self.codon_position_residual_adapter,
                    residual_split=self.codon_residual_split,
                    residual_gate=self.codon_residual_gate,
                    use_full_linear=self.codon_full_linear,
                    num_codons=self.num_codons_per_codebook,
                    chunk_layernorm=self.codon_chunk_layernorm,
                    chunk_interleave=self.codon_chunk_interleave,
                )
                for _m in range(self.num_codebooks)
            ]
        )

        # ---------- v70a (Exp 3): hash reconstruction decoder ------------
        self.use_hash_recon: bool = bool(getattr(args, "use_hash_recon", False))
        self.hash_recon_target: str = str(getattr(args, "hash_recon_target", "siglip_visual"))
        if self.use_hash_recon:
            self.hash_recon_decoder = HashReconDecoder(
                in_dim=NUM_SEMANTIC_PARTS * self.num_codons_per_codebook * 4,
                hidden_dim=int(getattr(args, "hash_recon_hidden", 256)),
                out_dim=self.proj_dim,
            )
        else:
            self.hash_recon_decoder = None

        # ---------- v72a (Exp 6): dual hash projection heads --------------
        self.use_dual_hash_proj: bool   = bool(getattr(args, "use_dual_hash_proj", False))
        self.dual_hash_proj_target: str = str(getattr(args, "dual_hash_proj_target", "siglip_visual"))
        if self.use_dual_hash_proj:
            self.dual_hash_proj = DualHashProj(
                in_dim=NUM_SEMANTIC_PARTS * self.num_codons_per_codebook * 4,
                hidden_dim=int(getattr(args, "dual_hash_proj_hidden", 256)),
                out_dim=self.proj_dim,
            )
        else:
            self.dual_hash_proj = None

        # ---------- v32: train-only text injection into quantizer ------
        # When `text_inject_train_only=add`, the routed visual tokens are
        # combined with the per-codebook text token *during training only*
        # before codebook lookup. The codeword embeddings absorb text-
        # semantic structure but at inference / eval we use the visual
        # token unchanged (no captions required at deploy). Default off.
        self.text_inject_train_only = str(getattr(args, "text_inject_train_only", "none"))
        self.text_inject_alpha      = float(getattr(args, "text_inject_alpha", 0.0))
        self.text_inject_detach     = bool(getattr(args, "text_inject_detach", True))

        # ---------- optional reconstruction decoder (v28 ablation) -----
        # `decoder_target='siglip_feat'` -> FeatureDecoder (cheap MLP, target
        # is the cached visual_global). 'pixel' -> PixelDecoder (~21M params,
        # ConvTranspose stack to 224x224). Off by default; enabled via
        # --use_decoder + --decoder_target.
        self.use_decoder    = bool(getattr(args, "use_decoder", False))
        self.decoder_target = str (getattr(args, "decoder_target", "siglip_feat"))
        if self.use_decoder:
            if self.decoder_target == "pixel":
                self.decoder = PixelDecoder(
                    d_model=self.d_model, num_codebooks=self.num_codebooks,
                )
            elif self.decoder_target == "siglip_feat":
                self.decoder = FeatureDecoder(
                    d_model=self.d_model, d_proj=self.proj_dim,
                    num_codebooks=self.num_codebooks,
                )
            else:
                raise ValueError(
                    f"[model_siglip2] unknown decoder_target {self.decoder_target!r}; "
                    f"choose 'pixel' or 'siglip_feat'."
                )
        else:
            self.decoder = None

        # ---------- freezing --------------------------------------------
        if bool(getattr(args, "freeze_backbone", True)):
            for p in self.backbone.parameters():
                p.requires_grad = False

        if isinstance(self.device, (str, torch.device)):
            self.to(self.device)

    # ====================================================== sanity helpers


    def _axis_center_local(self, x, n_global: int = 1, mask=None):
        """Subtract, per image, the mean across the LOCAL axis columns.

        arch-exp-3 (P2). ``x`` is [B, n_global + M_local, D]. Column(s) before
        ``n_global`` are returned untouched. With ``mask`` ([B, n_global +
        M_local]) the mean is taken over active columns only. Pure
        representation change: no parameters, no loss term.
        """
        loc = x[:, n_global:, :]
        if mask is not None:
            w = mask[:, n_global:].to(loc.dtype).unsqueeze(-1)
            mu = (loc * w).sum(dim=1, keepdim=True) / w.sum(dim=1, keepdim=True).clamp_min(1e-6)
        else:
            mu = loc.mean(dim=1, keepdim=True)
        return torch.cat([x[:, :n_global, :], loc - mu], dim=1)

    def _routing_mass_kwargs(self, feats, route_centroids_aug, route_global_text_active) -> Dict[str, Any]:
        """(a-1 / a-2, 2026-09-20) column targets for the Sinkhorn router.

        Local columns get the content-dependent mass
            (1 - alpha) / M_loc + alpha * softmax_m((cos(g, p_m) - mu_m) / tau)
        scaled to their usual share of the budget. g is the frozen CLIP image
        embedding (available with or without captions, so train and deployment
        compute the same targets); p_m, mu_m, tau are fixed training-set
        statistics stored as buffers. With --null_free_marginal the null column
        is excluded from the budget and left unconstrained in the solver.
        """
        B, M_route = int(route_centroids_aug.shape[0]), int(route_centroids_aug.shape[1])
        has_null = bool(self.use_null_centroid and self.null_centroid is not None)
        n_glob = 1 if route_global_text_active else 0
        pm = torch.ones(B, M_route, device=route_centroids_aug.device, dtype=torch.float32)
        kw: Dict[str, Any] = {}
        if self.null_free_marginal and has_null:
            pm[:, -1] = 0.0
            free = torch.zeros(M_route, dtype=torch.bool, device=pm.device)
            free[-1] = True
            kw["free_part_mask"] = free
        if self.routing_mass_alpha > 0.0:
            if not bool(self.routing_mass_ready.item()):
                raise RuntimeError("routing mass statistics were never initialised "
                                   "(dna_utils.arch_exp.init_routing_mass_stats)")
            g = feats.get("visual_global") if isinstance(feats, dict) else None
            if g is None:
                raise RuntimeError("--routing_mass_alpha needs feats['visual_global']")
            g = F.normalize(g.float(), dim=-1)                                   # [B, D_shared]
            p = F.normalize(self.routing_mass_proto.float(), dim=-1)             # [M_loc, D_shared]
            s = g @ p.t() - self.routing_mass_center.float()                     # [B, M_loc]
            w = torch.softmax(s / self.routing_mass_tau.float().clamp_min(1e-6), dim=-1)
            M_loc = int(p.shape[0])
            if n_glob + M_loc > M_route - (1 if has_null else 0):
                raise ValueError(f"route columns {M_route} cannot hold {M_loc} local slots")
            mass = (1.0 - self.routing_mass_alpha) / M_loc + self.routing_mass_alpha * w
            loc = slice(n_glob, n_glob + M_loc)
            pm[:, loc] = mass * pm[:, loc].sum(dim=-1, keepdim=True)
        kw["part_marginal"] = pm.detach()
        return kw

    def set_current_epoch(self, epoch: int) -> None:
        """Trainer calls this once per epoch so the router can compute the
        annealed epsilon. v33a only; no-op when annealing is off."""
        self._current_epoch = int(epoch)
        # (A): the quantizer has no epoch of its own; hand it the current one
        # so `freeze_after_epoch` can take effect at the right time.
        _q = getattr(self, "quantizer", None)
        if _q is not None and hasattr(_q, "_frozen_epoch_now"):
            _q._frozen_epoch_now = int(epoch)
        # (c-1): the quantizer is the identity for epochs < vq_bypass_epochs.
        if _q is not None and hasattr(_q, "bypass"):
            _q.bypass = bool(self.vq_bypass_epochs > 0
                             and int(epoch) < self.vq_bypass_epochs)

    def _grounded_text_routing(
        self,
        semantic_visual_tokens: torch.Tensor,        # [B, M, D] — Stage-1 routed visual repr per codebook
        cached_text_tokens:     torch.Tensor,        # [B, M, T, D_proj] — CLIP-projected text tokens per slot
        cached_text_token_mask: Optional[torch.Tensor],  # [B, M, T] bool (True = real)
    ) -> torch.Tensor:
        """v162 Stage-2: visually-grounded top-k_t text token pruning.

        For each codebook m, the routed visual repr v_m queries the M-th
        slot's T cached text tokens. Tokens score by cosine-similarity-like
        softmax (eps controls sharpness; eps -> 0 ≈ hard top-1). The top
        k_t tokens are kept, renormalized, and pooled into a refined per-
        codebook text embedding that REPLACES the static pooled text embed
        flowing into the loss layer.

        Returns
        -------
        refined : [B, M, D] in d_model space.
        """
        B_, M_, T_, _ = cached_text_tokens.shape
        eps  = float(max(self.grounded_text_eps, 1e-4))
        k_t  = int(min(max(self.grounded_text_k_t, 1), T_))

        # ---- 1) project text tokens into d_model space via shared adapter ----
        # Identical projection as the global text_part_tokens path so the
        # refined embed is distributionally comparable to baseline embeds.
        tk_flat = cached_text_tokens.reshape(B_ * M_ * T_, -1)
        if isinstance(self.text_adapter, nn.ModuleList):
            # per-slot: apply slot-m adapter to slot-m tokens
            tk_per_slot: List[torch.Tensor] = []
            tk_reshape = cached_text_tokens.reshape(B_, M_, T_, -1)
            for m in range(M_):
                tk_m = self.text_adapter[m](tk_reshape[:, m].reshape(B_ * T_, -1))
                tk_per_slot.append(tk_m.reshape(B_, T_, self.d_model))
            tk = torch.stack(tk_per_slot, dim=1)              # [B, M, T, D]
        else:
            tk = self.text_adapter(tk_flat).reshape(B_, M_, T_, self.d_model)

        # ---- 2) Stage-1 query (with optional stop-gradient) -----------------
        q = semantic_visual_tokens
        if self.grounded_text_stage1_sg:
            q = q.detach()                                    # [B, M, D]

        # ---- 3) cosine-scaled scores per (b, m, t) --------------------------
        q_n  = F.normalize(q,  dim=-1)                        # [B, M, D]
        tk_n = F.normalize(tk, dim=-1)                        # [B, M, T, D]
        scores = (q_n.unsqueeze(2) * tk_n).sum(dim=-1)        # [B, M, T]
        scores = scores / eps
        if cached_text_token_mask is not None:
            scores = scores.masked_fill(~cached_text_token_mask.bool(), -1e4)

        # ---- 4) top-k_t selection + renormalized weighted pool --------------
        topk_v, topk_idx = scores.topk(k_t, dim=-1)           # [B, M, k_t]
        topk_w = torch.softmax(topk_v, dim=-1)                # renormalize over k_t
        idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, -1, self.d_model)
        selected = torch.gather(tk, dim=2, index=idx_exp)     # [B, M, k_t, D]
        refined  = (selected * topk_w.unsqueeze(-1)).sum(dim=2)   # [B, M, D]
        refined  = self.grounded_text_ln(refined)
        return refined

    def _soft_visual_grounded_text_pooling(
        self,
        visual_tokens: torch.Tensor,                  # [B, N, D]
        routing_matrix: torch.Tensor,                 # [B, N, M=6]
        base_text_tokens: torch.Tensor,               # [B, M=6, D]
        cached_text_tokens: torch.Tensor,             # [B, M=6, T, D_proj]
        cached_text_token_mask: Optional[torch.Tensor],  # [B, M=6, T] bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Refine local text slots with OT-weighted soft cross-attention.

        Visual patches query all valid tokens in the corresponding text slot.
        The per-query outputs are pooled with detached local OT mass, so only
        patches surviving the visual mask and assigned to slot ``m`` affect
        that slot's text representation. C_0 remains unchanged.

        Returns:
            refined_text: [B, 6, D]
            normalized_attention_entropy: [B, 5]
        """
        B, N, D = visual_tokens.shape
        assert D == self.d_model
        assert routing_matrix.shape == (B, N, NUM_SEMANTIC_PARTS), (
            f"routing_matrix must be {(B, N, NUM_SEMANTIC_PARTS)}, got "
            f"{tuple(routing_matrix.shape)}"
        )
        assert base_text_tokens.shape == (B, NUM_SEMANTIC_PARTS, D)
        assert cached_text_tokens.dim() == 4
        assert cached_text_tokens.shape[:2] == (B, NUM_SEMANTIC_PARTS)

        adapted_text = self._adapt_cached_text_tokens_for_routing(
            cached_text_tokens,
        )                                                           # [B, 6, T, D]
        _, _, T, _ = adapted_text.shape
        if cached_text_token_mask is None:
            text_valid = torch.ones(
                B, NUM_SEMANTIC_PARTS, T,
                dtype=torch.bool,
                device=visual_tokens.device,
            )
        else:
            assert cached_text_token_mask.shape == (B, NUM_SEMANTIC_PARTS, T)
            text_valid = cached_text_token_mask.to(
                device=visual_tokens.device, dtype=torch.bool,
            )

        # The visual query is evidence, not an optimization shortcut: text
        # losses train the attention/text branch but cannot move the router by
        # choosing easier words through this query path.
        visual_query = visual_tokens.detach()                         # [B, N, D]
        local_route = routing_matrix[:, :, 1:].detach().clamp_min(0.0) # [B, N, 5]
        refined_local: List[torch.Tensor] = []
        entropy_local: List[torch.Tensor] = []
        for m in range(NUM_LOCAL_PARTS):
            token_m = adapted_text[:, m + 1, :, :]                    # [B, T, D]
            valid_m = text_valid[:, m + 1, :]                         # [B, T]
            # A cache row should always contain at least one real token. Keep
            # the operation finite if malformed input reaches this path.
            empty = ~valid_m.any(dim=-1)
            if bool(empty.any()):
                valid_m = valid_m.clone()
                valid_m[empty, 0] = True

            attn_out, attn_weights = self.soft_grounded_text_attn(
                query=visual_query,
                key=token_m,
                value=token_m,
                key_padding_mask=~valid_m,
                need_weights=True,
                average_attn_weights=True,
            )                                                        # [B,N,D], [B,N,T]
            assert attn_out.shape == (B, N, D)
            assert attn_weights.shape == (B, N, T)

            route_m = local_route[:, :, m]                           # [B, N]
            route_m = route_m / route_m.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            pooled_m = (attn_out * route_m.unsqueeze(-1)).sum(dim=1) # [B, D]
            refined_m = self.soft_grounded_text_ln(
                base_text_tokens[:, m + 1, :] + pooled_m,
            )                                                        # [B, D]
            refined_local.append(refined_m)

            token_weight = (
                attn_weights * route_m.unsqueeze(-1)
            ).sum(dim=1)                                             # [B, T]
            token_weight = token_weight * valid_m.to(token_weight.dtype)
            token_weight = token_weight / token_weight.sum(
                dim=-1, keepdim=True,
            ).clamp_min(1e-8)
            entropy = -(
                token_weight * token_weight.clamp_min(1e-8).log()
            ).sum(dim=-1)                                            # [B]
            valid_count = valid_m.sum(dim=-1).to(entropy.dtype)
            norm = valid_count.log()
            entropy = torch.where(
                valid_count > 1.0,
                entropy / norm.clamp_min(1e-8),
                torch.zeros_like(entropy),
            )
            entropy_local.append(entropy)

        refined = torch.cat([
            base_text_tokens[:, :1, :],
            torch.stack(refined_local, dim=1),
        ], dim=1)                                                      # [B, 6, D]
        attention_entropy = torch.stack(entropy_local, dim=1)         # [B, 5]
        assert refined.shape == (B, NUM_SEMANTIC_PARTS, D)
        assert attention_entropy.shape == (B, NUM_LOCAL_PARTS)
        return refined, attention_entropy

    def _project_cibhash_visual_tokens(
        self,
        semantic_visual_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Project each semantic slot for the train-only CIBHash objective."""
        B, M, D = semantic_visual_tokens.shape
        assert (M, D) == (NUM_SEMANTIC_PARTS, self.d_model)
        assert self.cibhash_visual_projectors is not None
        projected = torch.stack([
            self.cibhash_visual_projectors[m](semantic_visual_tokens[:, m, :])
            for m in range(NUM_SEMANTIC_PARTS)
        ], dim=1)                                                       # [B, 6, D]
        assert projected.shape == (B, NUM_SEMANTIC_PARTS, D)
        return projected

    def _cosine_visual_grounded_text_pooling(
        self,
        visual_tokens: torch.Tensor,                  # [B, N, D]
        routing_matrix: torch.Tensor,                 # [B, N, M=6]
        base_text_tokens: torch.Tensor,               # [B, M=6, D]
        cached_text_tokens: torch.Tensor,             # [B, M=6, T, D_proj]
        cached_text_token_mask: Optional[torch.Tensor],  # [B, M=6, T] bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Interpolate local text slots toward an OT-grounded cosine pool.

        The attention has no projection matrix or temperature parameter.
        Detached visual tokens and OT mass only define token weights; the
        learned text branch and the five zero-initialized residual gates
        receive gradients. C_0 is unchanged.

        Returns:
            refined_text: [B, 6, D]
            normalized_attention_entropy: [B, 5]
            effective_token_support_ratio: [B, 5]
        """
        B, N, D = visual_tokens.shape
        assert D == self.d_model
        assert routing_matrix.shape == (B, N, NUM_SEMANTIC_PARTS), (
            f"routing_matrix must be {(B, N, NUM_SEMANTIC_PARTS)}, got "
            f"{tuple(routing_matrix.shape)}"
        )
        assert base_text_tokens.shape == (B, NUM_SEMANTIC_PARTS, D)
        assert cached_text_tokens.dim() == 4
        assert cached_text_tokens.shape[:2] == (B, NUM_SEMANTIC_PARTS)
        assert self.cosine_grounded_text_gate is not None
        assert self.cosine_grounded_text_gate.shape == (NUM_LOCAL_PARTS,)

        adapted_text = self._adapt_cached_text_tokens_for_routing(
            cached_text_tokens,
        )                                                           # [B, 6, T, D]
        _, _, T, _ = adapted_text.shape
        if cached_text_token_mask is None:
            text_valid = torch.ones(
                B, NUM_SEMANTIC_PARTS, T,
                dtype=torch.bool,
                device=visual_tokens.device,
            )
        else:
            assert cached_text_token_mask.shape == (B, NUM_SEMANTIC_PARTS, T)
            text_valid = cached_text_token_mask.to(
                device=visual_tokens.device, dtype=torch.bool,
            )

        local_tokens = adapted_text[:, 1:, :, :]                    # [B, 5, T, D]
        local_valid = text_valid[:, 1:, :]                           # [B, 5, T]
        empty = ~local_valid.any(dim=-1)                              # [B, 5]
        if bool(empty.any()):
            local_valid = local_valid.clone()
            empty_b, empty_m = empty.nonzero(as_tuple=True)
            local_valid[empty_b, empty_m, 0] = True

        visual_n = F.normalize(
            visual_tokens.detach(), dim=-1,
        )                                                            # [B, N, D]
        text_n = F.normalize(local_tokens, dim=-1)                    # [B, 5, T, D]
        cosine = torch.einsum(
            "bnd,bmtd->bmnt", visual_n, text_n,
        )                                                            # [B, 5, N, T]
        cosine = cosine.masked_fill(
            ~local_valid[:, :, None, :], -1e4,
        )
        patch_to_text = cosine.softmax(dim=-1)                       # [B, 5, N, T]

        local_route = (
            routing_matrix[:, :, 1:].detach().clamp_min(0.0)
            .transpose(1, 2)
        )                                                            # [B, 5, N]
        local_route = local_route / local_route.sum(
            dim=-1, keepdim=True,
        ).clamp_min(1e-6)
        token_weight = (
            patch_to_text * local_route.unsqueeze(-1)
        ).sum(dim=2)                                                 # [B, 5, T]
        token_weight = token_weight * local_valid.to(token_weight.dtype)
        token_weight = token_weight / token_weight.sum(
            dim=-1, keepdim=True,
        ).clamp_min(1e-8)

        pooled_local = (
            token_weight.unsqueeze(-1) * local_tokens
        ).sum(dim=2)                                                 # [B, 5, D]
        base_local = base_text_tokens[:, 1:, :]                      # [B, 5, D]
        gate = self.cosine_grounded_text_gate.view(1, NUM_LOCAL_PARTS, 1)
        refined_local = base_local + gate * (pooled_local - base_local)
        refined = torch.cat([
            base_text_tokens[:, :1, :], refined_local,
        ], dim=1)                                                     # [B, 6, D]

        entropy = -(
            token_weight * token_weight.clamp_min(1e-8).log()
        ).sum(dim=-1)                                                # [B, 5]
        valid_count = local_valid.sum(dim=-1).to(entropy.dtype)      # [B, 5]
        normalized_entropy = torch.where(
            valid_count > 1.0,
            entropy / valid_count.log().clamp_min(1e-8),
            torch.zeros_like(entropy),
        )
        effective_support = torch.exp(entropy) / valid_count.clamp_min(1.0)

        assert refined.shape == (B, NUM_SEMANTIC_PARTS, D)
        assert normalized_entropy.shape == (B, NUM_LOCAL_PARTS)
        assert effective_support.shape == (B, NUM_LOCAL_PARTS)
        return refined, normalized_entropy, effective_support

    def _current_sinkhorn_epsilon(self) -> Optional[float]:
        """Return the annealed Sinkhorn epsilon for the current epoch, or
        None if annealing is not configured (router falls back to its
        stored static epsilon)."""
        eps_i = self.sinkhorn_epsilon_init
        eps_f = self.sinkhorn_epsilon_final
        if eps_i is None or eps_f is None:
            return None
        t_max = max(self.sinkhorn_schedule_horizon - 1, 1)
        t = min(max(self._current_epoch, 0), t_max) / t_max
        # cosine schedule (smooth, no plateau)
        cos_t = 0.5 * (1.0 + math.cos(math.pi * t))      # 1.0 -> 0.0
        return float(eps_f) + (float(eps_i) - float(eps_f)) * cos_t

    def _current_codebook_choice_beta(self) -> float:
        """Return beta for v124 soft expert-choice routing.

        beta=0 keeps the pre-choice routing matrix; beta=1 recovers the
        hard v85 expert-choice filter. Optional warm-up lets early OT/text
        alignment form before the codebook-side capacity filter sharpens.
        """
        beta = max(0.0, min(float(self.routing_codebook_choice_beta), 1.0))
        warmup = max(int(self.routing_codebook_choice_warmup_epochs), 0)
        if warmup <= 0:
            return beta
        t = min(max(self._current_epoch, 0) + 1, warmup) / float(warmup)
        return beta * t

    def _current_routing_text_evidence_scale(self) -> float:
        """Return the v176/v177 warm-up multiplier in [0, 1]."""
        warmup = max(int(getattr(self, "routing_text_evidence_warmup_epochs", 0)), 0)
        if warmup <= 0:
            return 1.0
        t = min(max(self._current_epoch, 0) + 1, warmup) / float(warmup)
        return float(t)

    def _current_routing_text_evidence_beta(self) -> float:
        """Return the current v176a soft text-evidence bias strength."""
        beta = max(0.0, float(getattr(self, "routing_text_evidence_beta", 0.0)))
        if beta <= 0.0:
            return beta
        return beta * self._current_routing_text_evidence_scale()

    def _current_routing_text_evidence_penalty(self) -> float:
        """Return the current v177 evidence candidate penalty strength."""
        penalty = max(0.0, float(getattr(self, "routing_text_evidence_penalty", 0.0)))
        if penalty <= 0.0:
            return penalty
        return penalty * self._current_routing_text_evidence_scale()

    def _current_routing_token_ot_beta(self) -> float:
        """Return the current v179 token-evidence routing bias strength."""
        if not bool(getattr(self, "routing_token_ot_evidence", False)):
            return 0.0
        beta = max(0.0, float(getattr(self, "routing_token_ot_beta", 0.0)))
        if beta <= 0.0:
            return beta
        warmup = max(int(getattr(self, "routing_token_ot_warmup_epochs", 0)), 0)
        if warmup <= 0:
            return beta
        t = min(max(self._current_epoch, 0) + 1, warmup) / float(warmup)
        return beta * float(t)

    def _adapt_cached_text_tokens_for_routing(
        self,
        cached_text_tokens: torch.Tensor,          # [B, 6, T, D_proj]
    ) -> torch.Tensor:
        """Project cached text-token embeddings into the routing d_model.

        The transform mirrors the pooled text path as closely as possible.
        For v179a we primarily use ``partial_whiten``; other text transforms
        are left unchanged because token-level analogues of image-mean or
        global-residual pooling are not well-defined.
        """
        assert cached_text_tokens.dim() == 4, (
            "cached_text_tokens must be [B, 6, T, D_proj], got "
            f"{tuple(cached_text_tokens.shape)}"
        )
        B_, M_, T_, _ = cached_text_tokens.shape
        tokens = cached_text_tokens
        tx_mode = str(getattr(self, "text_embed_transform", "none"))
        if tx_mode == "partial_whiten":
            if not getattr(self, "_text_whiten_ready", False):
                raise RuntimeError(
                    "[model_siglip2] routing_token_ot_evidence with "
                    "partial_whiten requires loaded whitening buffers "
                    "(see --text_whiten_npz)."
                )
            orig_dtype = tokens.dtype
            flat = tokens.reshape(-1, tokens.shape[-1]).to(self.text_whiten_mu.dtype)
            flat = (flat - self.text_whiten_mu) @ self.text_whiten_W
            tokens = flat.reshape(B_, M_, T_, -1).to(orig_dtype)

        if self.codebook_text_prompts is not None:
            assert self.codebook_text_prompts.shape[0] == M_, (
                "codebook_text_prompts slot count must match cached text tokens, "
                f"got {self.codebook_text_prompts.shape[0]} vs {M_}"
            )
            tokens = tokens + self.codebook_text_prompts.unsqueeze(0).unsqueeze(2)

        if isinstance(self.text_adapter, nn.ModuleList):
            assert len(self.text_adapter) == M_, (
                "per-slot text_adapter count must match cached text tokens, "
                f"got {len(self.text_adapter)} vs {M_}"
            )
            per_slot_tokens = []
            for m in range(M_):
                # [B, T, D_proj] -> [B*T, D_proj] -> [B, T, D]
                tk_m = self.text_adapter[m](tokens[:, m].reshape(B_ * T_, -1))
                per_slot_tokens.append(tk_m.reshape(B_, T_, self.d_model))
            adapted = torch.stack(per_slot_tokens, dim=1)       # [B, 6, T, D]
        else:
            flat = tokens.reshape(B_ * M_ * T_, -1)
            adapted = self.text_adapter(flat).reshape(B_, M_, T_, self.d_model)

        assert adapted.shape == (B_, M_, T_, self.d_model), (
            "adapted text tokens must be [B, 6, T, D], got "
            f"{tuple(adapted.shape)}"
        )
        return adapted

    def _routing_token_ot_cost_bias(
        self,
        visual_tokens_for_routing: torch.Tensor,        # [B, N, D]
        route_centroids_aug: torch.Tensor,              # [B, M_route, D]
        cached_text_tokens: torch.Tensor,               # [B, 6, T, D_proj]
        cached_text_token_mask: Optional[torch.Tensor], # [B, 6, T] bool
        visual_attention_mask: Optional[torch.Tensor],  # [B, N] or None
        route_global_text_active: bool,
    ) -> torch.Tensor:
        """v179a local token-level text evidence for Sinkhorn cost bias.

        Each local part keeps the top-K most visually grounded text tokens.
        A semi-balanced token-to-visual transport then produces an evidence
        map over patches. Only the text-token marginal is fixed; the visual
        marginal is intentionally relaxed so background patches are not forced
        to receive uniform mass.
        """
        B, N, D = visual_tokens_for_routing.shape
        M_route = route_centroids_aug.shape[1]
        assert route_centroids_aug.shape == (B, M_route, D)
        assert cached_text_tokens.shape[0] == B and cached_text_tokens.shape[1] >= NUM_SEMANTIC_PARTS, (
            "cached_text_tokens must include 6 semantic slots, got "
            f"{tuple(cached_text_tokens.shape)}"
        )

        text_tokens = self._adapt_cached_text_tokens_for_routing(cached_text_tokens)
        local_text_tokens = text_tokens[:, 1:1 + NUM_LOCAL_PARTS]          # [B, 5, T, D]
        B_t, M_local, T, D_t = local_text_tokens.shape
        assert (B_t, M_local, D_t) == (B, NUM_LOCAL_PARTS, D), (
            "local adapted text tokens must be [B, 5, T, D], got "
            f"{tuple(local_text_tokens.shape)}"
        )

        if cached_text_token_mask is None:
            token_mask = torch.ones(B, NUM_LOCAL_PARTS, T, dtype=torch.bool,
                                    device=visual_tokens_for_routing.device)
        else:
            token_mask = cached_text_token_mask[:, 1:1 + NUM_LOCAL_PARTS].to(
                device=visual_tokens_for_routing.device,
                dtype=torch.bool,
            )                                                           # [B, 5, T]
        assert token_mask.shape == (B, NUM_LOCAL_PARTS, T), (
            "local token mask must be [B, 5, T], got "
            f"{tuple(token_mask.shape)}"
        )

        v_n = F.normalize(visual_tokens_for_routing, dim=-1)              # [B, N, D]
        t_n = F.normalize(local_text_tokens, dim=-1)                      # [B, 5, T, D]
        sim = torch.einsum("bnd,bmtd->bmnt", v_n, t_n)                   # [B, 5, N, T]
        assert sim.shape == (B, NUM_LOCAL_PARTS, N, T)

        token_scores = sim.max(dim=2).values                              # [B, 5, T]
        token_scores = token_scores.masked_fill(~token_mask, -float("inf"))
        k_text = max(1, min(int(getattr(self, "routing_token_ot_topk_text", 8)), T))
        topk_idx = token_scores.topk(k=k_text, dim=-1).indices            # [B, 5, K]
        token_keep = torch.zeros_like(token_mask)
        token_keep.scatter_(dim=-1, index=topk_idx, value=True)
        token_keep = token_keep & token_mask                              # [B, 5, T]

        eps = max(float(getattr(self, "routing_token_ot_eps", 0.05)), 1e-6)
        logits = sim / eps                                                # [B, 5, N, T]
        logits = logits.masked_fill(~token_keep.unsqueeze(2), -1e4)
        if visual_attention_mask is not None:
            patch_mask = visual_attention_mask.to(torch.bool).unsqueeze(1).unsqueeze(-1)  # [B,1,N,1]
            logits = logits.masked_fill(~patch_mask, -1e4)

        # Semi-balanced transport: each kept text token distributes its mass
        # over visual patches; visual patches are not forced to a uniform
        # marginal. This keeps evidence discriminative rather than flat.
        patch_mass = torch.softmax(logits, dim=2)                         # [B, 5, N, T]
        patch_mass = patch_mass * token_keep.unsqueeze(2).to(patch_mass.dtype)
        denom = token_keep.sum(dim=-1).clamp_min(1).to(patch_mass.dtype)   # [B, 5]
        evidence = patch_mass.sum(dim=-1) / denom.unsqueeze(-1)            # [B, 5, N]
        evidence = evidence / evidence.amax(dim=-1, keepdim=True).clamp_min(1e-6)
        evidence = evidence.transpose(1, 2).contiguous()                  # [B, N, 5]
        assert evidence.shape == (B, N, NUM_LOCAL_PARTS), (
            "token OT evidence must be [B, N, 5], got "
            f"{tuple(evidence.shape)}"
        )

        bias = torch.zeros(B, N, M_route, device=visual_tokens_for_routing.device,
                           dtype=visual_tokens_for_routing.dtype)
        local_start = 1 if route_global_text_active else 0
        local_end = min(local_start + NUM_LOCAL_PARTS, M_route)
        if local_end > local_start:
            local_count = local_end - local_start
            bias[..., local_start:local_end] = evidence[..., :local_count].to(bias.dtype)
        return bias

    @staticmethod
    def _remove_global_projection(
        slots: torch.Tensor,
        gamma: float,
        detach_global: bool,
        name: str,
    ) -> torch.Tensor:
        """Remove C0/global projection from local slots.

        slots: [B, 6, D]. Output shape is unchanged; slots[:, 0, :] is kept
        as-is, while slots[:, 1:, :] become local residuals.
        """
        assert slots.dim() == 3 and slots.shape[1] == NUM_SEMANTIC_PARTS, (
            f"{name} must be [B, 6, D], got {tuple(slots.shape)}"
        )
        B, M, D = slots.shape
        assert M == NUM_SEMANTIC_PARTS, f"{name} M={M} must be {NUM_SEMANTIC_PARTS}"
        g = slots[:, 0, :]                                      # [B, D]
        g_ref = g.detach() if detach_global else g              # [B, D]
        g_n = F.normalize(g_ref, dim=-1).unsqueeze(1)           # [B, 1, D]
        local = slots[:, 1:, :]                                 # [B, 5, D]
        proj = (local * g_n).sum(dim=-1, keepdim=True) * g_n    # [B, 5, D]
        local_res = local - float(gamma) * proj                 # [B, 5, D]
        out = torch.cat([g.unsqueeze(1), local_res], dim=1)     # [B, 6, D]
        assert out.shape == (B, NUM_SEMANTIC_PARTS, D), (
            f"{name} residual output shape {tuple(out.shape)} is invalid"
        )
        return out

    def assert_no_shared_trainable_params(self, verbose: bool = False) -> None:
        v_ids = {id(p) for p in self.visual_adapter.parameters() if p.requires_grad}
        t_ids = {id(p) for p in self.text_adapter.parameters()   if p.requires_grad}
        shared = v_ids & t_ids
        if verbose:
            print(f"[model_siglip2] visual_adapter trainable tensors = {len(v_ids)}")
            print(f"[model_siglip2] text_adapter   trainable tensors = {len(t_ids)}")
            print(f"[model_siglip2] shared between branches          = {len(shared)}")
        assert not shared, (
            "visual_adapter and text_adapter share trainable parameters — "
            "they must be SEPARATE nn.Module instances."
        )

    @staticmethod
    def _count_params(module: nn.Module) -> Tuple[int, int]:
        total = sum(p.numel() for p in module.parameters())
        train = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return total, train

    def parameter_summary(self) -> Dict[str, Tuple[int, int]]:
        gate_total = self.global_gate_logits.numel()
        gate_train = gate_total if self.global_gate_logits.requires_grad else 0
        return {
            "all":             self._count_params(self),
            "backbone":        self._count_params(self.backbone),
            "visual_encoder":  self._count_params(self.visual_encoder),
            "text_encoder":    self._count_params(self.text_encoder),
            "visual_adapter":  self._count_params(self.visual_adapter),
            "text_adapter":    self._count_params(self.text_adapter),
            "router":          self._count_params(self.router),
            "quantizer":       self._count_params(self.quantizer),
            "global_gate":     (gate_total, gate_train),
            "codon_heads":     self._count_params(self.codon_heads),
        }

    def print_parameter_summary(self) -> None:
        s = self.parameter_summary()
        print(f"{'module':<20s}{'total':>15s}{'trainable':>15s}")
        for name, (total, train) in s.items():
            print(f"{name:<20s}{total:>15,d}{train:>15,d}")

    def set_gumbel_tau(self, tau: float) -> None:
        """Update the Gumbel-Softmax temperature on every per-codebook codon
        head. Called from the training loop to anneal `tau` across epochs.

        Each ``CodonHead`` stores its own ``gumbel_tau`` so the change is just
        a scalar overwrite.
        """
        tau = float(tau)
        for head in self.codon_heads:
            head.gumbel_tau = tau
        self.gumbel_tau = tau

    def _adapt_pooled_text_for_loss(self, raw: torch.Tensor) -> torch.Tensor:
        """Apply the factual loss-side pooled-text transform to foil features.

        Counterfactual captions are intentionally not routed and do not use
        visual-conditioned token attention.  They do, however, share the
        same fixed whitening/centering transform, prompt bias, and learnable
        per-slot text adapter as the factual text-DNA path.
        """
        if raw.dim() != 3 or raw.shape[1] != NUM_SEMANTIC_PARTS:
            raise ValueError(
                "pooled text for DNA loss must be [B, 6, D_proj], got "
                f"{tuple(raw.shape)}"
            )
        original_dtype = raw.dtype
        tx_mode = self.text_embed_transform
        routing_only = bool(getattr(self, "text_transform_routing_only", False))
        if routing_only and tx_mode in (
            "per_image_mean", "global_residual", "partial_whiten",
        ):
            # The factual branch uses untransformed embeddings for losses in
            # this mode; only its routing centroids are transformed.
            tx_mode = "none"

        if tx_mode == "per_image_mean":
            global_slot = raw[:, :1, :]
            local = raw[:, 1:, :]
            local = F.normalize(local - local.mean(dim=1, keepdim=True), dim=-1)
            raw = torch.cat([global_slot, local], dim=1)
        elif tx_mode == "global_residual":
            global_slot = raw[:, :1, :]
            local = F.normalize(raw[:, 1:, :] - global_slot, dim=-1)
            raw = torch.cat([global_slot, local], dim=1)
        elif tx_mode == "partial_whiten":
            if not getattr(self, "_text_whiten_ready", False):
                raise RuntimeError(
                    "counterfactual text path requires loaded partial-whiten buffers"
                )
            shape = raw.shape
            flat = raw.reshape(-1, shape[-1]).to(self.text_whiten_mu.dtype)
            flat = (flat - self.text_whiten_mu) @ self.text_whiten_W
            raw = flat.reshape(shape).to(original_dtype)
        elif tx_mode == "global_residual_whiten":
            if not getattr(self, "_text_whiten_ready", False):
                raise RuntimeError(
                    "counterfactual text path requires loaded residual-whiten buffers"
                )
            global_slot = raw[:, :1, :]
            local = raw[:, 1:, :] - global_slot
            shape = local.shape
            flat = local.reshape(-1, shape[-1]).to(self.text_whiten_mu.dtype)
            flat = (flat - self.text_whiten_mu) @ self.text_whiten_W
            local = flat.reshape(shape).to(original_dtype)
            raw = torch.cat([global_slot, local], dim=1)

        if self.codebook_text_prompts is not None:
            raw = raw + self.codebook_text_prompts.unsqueeze(0)
        if self.per_slot_text_adapter:
            return torch.stack(
                [self.text_adapter[m](raw[:, m, :])
                 for m in range(NUM_SEMANTIC_PARTS)],
                dim=1,
            )
        return self.text_adapter(raw)

    def _pool_local_foil_tokens_like_factual(
        self,
        cached_text_foil_tokens: torch.Tensor,
        cached_text_foil_token_mask: torch.Tensor,
        visual_tokens_raw: torch.Tensor,
        visual_attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Reproduce v185's factual token-prune/mean pool for local foils.

        Using the ordinary CLIP pooled/EOS feature here would give the
        classifier a trivial representation-domain cue instead of forcing it
        to learn the edited word. The implementation intentionally mirrors
        the factual block in ``forward`` without letting foil tokens affect
        the visual keep mask or routing.
        """
        if self.backbone_type != "clip":
            raise RuntimeError(
                "token-matched counterfactual pooling currently requires CLIP"
            )
        if cached_text_foil_tokens.dim() != 4:
            raise ValueError(
                "cached_text_foil_tokens must be [B, 6, T, D_proj]"
            )
        B, M, T, D_proj = cached_text_foil_tokens.shape
        if M != NUM_SEMANTIC_PARTS or D_proj != self.proj_dim:
            raise ValueError(
                "cached_text_foil_tokens must be [B, 6, T, D_proj], got "
                f"{tuple(cached_text_foil_tokens.shape)}"
            )
        if cached_text_foil_token_mask.dtype != torch.bool:
            raise TypeError(
                "cached_text_foil_token_mask must have dtype torch.bool"
            )
        if cached_text_foil_token_mask.shape != (B, M, T):
            raise ValueError(
                "cached_text_foil_token_mask shape does not match token cache"
            )

        visual_projection = self.backbone.model.visual_projection
        visual_shared = visual_projection(visual_tokens_raw)
        visual_shared_n = F.normalize(visual_shared, dim=-1)
        token_local = cached_text_foil_tokens[:, 1:, :, :]
        mask_local = cached_text_foil_token_mask[:, 1:, :]
        token_local_n = F.normalize(token_local, dim=-1)
        similarity = torch.einsum(
            "bnd,bmtd->bmnt", visual_shared_n, token_local_n,
        )
        _, M_local, N, _ = similarity.shape
        if M_local != NUM_LOCAL_PARTS:
            raise ValueError("foil token cache must contain five local slots")

        if self.bidirectional_token_prune_mode in (
            "mutual_dual_softmax", "mutual_consensus_residual",
        ):
            logit_scale = getattr(self.backbone.model, "logit_scale", None)
            if logit_scale is not None:
                scale = logit_scale.exp().detach().clamp(max=100.0)
                similarity = similarity * scale.to(similarity.dtype)
            mutual = _mutual_dual_softmax_pruning(
                similarity=similarity,
                text_mask=mask_local,
                visual_mask=visual_attention_mask,
                visual_keep_ratio=self.bidirectional_token_prune_visual_ratio,
                text_keep_ratio=self.bidirectional_token_prune_text_ratio,
                suppress_visual_consensus=(
                    self.bidirectional_token_prune_mode
                    == "mutual_consensus_residual"
                ),
            )
            text_keep = mutual["text_keep"]
        else:
            similarity = similarity + (
                ~mask_local.unsqueeze(2)
            ).float() * (-1e4)
            attention_text = similarity.softmax(dim=-2)
            text_importance = attention_text.sum(dim=-2)
            keep_count = max(
                1,
                int(T * self.bidirectional_token_prune_text_ratio),
            )
            threshold = text_importance.topk(
                keep_count, dim=-1,
            )[0][:, :, -1:]
            text_keep = (text_importance >= threshold) & mask_local

        keep = text_keep.unsqueeze(-1).to(token_local.dtype)
        pooled = (token_local * keep).sum(dim=2)
        pooled = pooled / keep.sum(dim=2).clamp(min=1.0)
        if pooled.shape != (B, NUM_LOCAL_PARTS, self.proj_dim):
            raise AssertionError(
                f"unexpected pooled foil shape {tuple(pooled.shape)}"
            )
        return pooled

    def _encode_text_tokens_to_dna(
        self,
        text_tokens: torch.Tensor,
        *,
        allow_mm_ema: bool,
        deterministic_codon: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Shared text -> VQ -> codon path for factual captions and foils."""
        if text_tokens.dim() != 3 or text_tokens.shape[1:] != (
            NUM_SEMANTIC_PARTS, self.d_model,
        ):
            raise ValueError(
                "text DNA encoder expects [B, 6, d_model], got "
                f"{tuple(text_tokens.shape)}"
            )
        B = text_tokens.shape[0]
        quantizer_tokens = text_tokens
        if (
            self.local_residual_quant
            and self.local_residual_text
            and self.local_residual_gamma > 0.0
        ):
            quantizer_tokens = self._remove_global_projection(
                text_tokens,
                gamma=self.local_residual_gamma,
                detach_global=self.local_residual_detach_global,
                name="text_tokens",
            )

        prev_train = self.quantizer.training
        if not (allow_mm_ema and self.mm_ema):
            self.quantizer.eval()
        try:
            tq_out = self.quantizer(quantizer_tokens)
        finally:
            if prev_train and not (allow_mm_ema and self.mm_ema):
                self.quantizer.train()

        text_q_st = tq_out["quantized_tokens"]
        text_q_raw = tq_out["quantized_tokens_raw"]
        text_cb_indices = tq_out["codebook_indices"]
        text_codon_residual = (
            quantizer_tokens - text_q_raw
            if self.codon_residual_gamma > 0.0 else None
        )
        text_head_inputs = (
            quantizer_tokens if self.codon_input_source == "routed"
            else text_q_st
        )
        continuous = []
        previous_head_modes = [head.training for head in self.codon_heads]
        if deterministic_codon:
            # CodonHead computes the same softmax probabilities in eval mode
            # but skips its unused train-time Gumbel sample. This keeps the
            # detached foil branch from perturbing the factual RNG stream.
            for head in self.codon_heads:
                head.eval()
        try:
            for m, head in enumerate(self.codon_heads):
                residual_m = (
                    text_codon_residual[:, m, :]
                    if text_codon_residual is not None else None
                )
                text_chunk_m = quantizer_tokens[:, m, :] \
                    if self.codon_text_anchor else None
                head_out = head(
                    text_head_inputs[:, m, :],
                    residual=residual_m,
                    gamma=self.codon_residual_gamma,
                    text_chunks=text_chunk_m,
                )
                continuous.append(head_out["continuous_code"])
        finally:
            if deterministic_codon:
                for head, was_training in zip(
                    self.codon_heads, previous_head_modes,
                ):
                    head.train(was_training)
        L = self.num_codons_per_codebook
        return {
            "continuous_code": torch.stack(continuous, dim=1).reshape(
                B, NUM_SEMANTIC_PARTS * L, 4,
            ),
            "quantized_tokens": text_q_st,
            "codebook_indices": text_cb_indices,
            "quantizer_input": quantizer_tokens,
        }

    # ====================================================== encoders

    def feature_extraction(
        self,
        pixel_values:           torch.Tensor,                # [B, 3, H, W]
        part_input_ids:         Optional[torch.LongTensor] = None,   # [B, M, L] or None
        part_attention_mask:    Optional[torch.Tensor]     = None,   # [B, M, L] or None
        visual_attention_mask:  Optional[torch.Tensor]     = None,   # [B, N] (Naflex) or None
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Vision tower (always) + text tower (only when text input provided)."""
        # ---- visual: keep the full token axis ---------------------------
        v_out = self.visual_encoder(
            pixel_values=pixel_values,
            pixel_attention_mask=visual_attention_mask,
            return_tokens=True,
            normalize=False,
        )
        visual_tokens_raw = v_out["token_feat"]               # [B, N, H_v]
        visual_global     = v_out["global_feat"]              # [B, D_proj]

        # ---- text: per-part global features (skip if no text supplied) --
        text_part_raw: Optional[torch.Tensor] = None
        if part_input_ids is not None:
            B, M_in, L = part_input_ids.shape
            flat_ids  = part_input_ids.reshape(B * M_in, L)
            flat_mask = (part_attention_mask.reshape(B * M_in, L)
                         if part_attention_mask is not None else None)
            t_out = self.text_encoder(
                input_ids=flat_ids,
                attention_mask=flat_mask,
                return_tokens=False,
                normalize=False,
            )
            text_global_flat = t_out["global_feat"]           # [B*M, D_proj]
            D_proj = text_global_flat.shape[-1]
            text_part_raw = text_global_flat.reshape(B, M_in, D_proj)  # [B, M, D_proj]

        return {
            "visual_tokens_raw": visual_tokens_raw,           # [B, N, H_v]
            "visual_global":     visual_global,               # [B, D_proj]
            "text_part_raw":     text_part_raw,               # [B, M, D_proj] or None
        }

    # ====================================================== forward

    def forward(
        self,
        pixel_values:           Optional[torch.Tensor] = None,         # [B, 3, H, W]
        part_input_ids:         Optional[torch.LongTensor] = None,     # [B, M=6, L] or None
        part_attention_mask:    Optional[torch.Tensor]     = None,     # [B, M=6, L] or None
        visual_attention_mask:  Optional[torch.Tensor]     = None,     # [B, N] or None
        part_mask:              Optional[torch.Tensor]     = None,     # [B, M=6] or None
        return_routing:         bool = True,
        cached_visual_tokens_raw: Optional[torch.Tensor] = None,       # [B, N, H_v]
        cached_visual_global:     Optional[torch.Tensor] = None,       # [B, D_proj]
        cached_text_part_raw:     Optional[torch.Tensor] = None,       # [B, M, D_proj]
        cached_has_text:          Optional[torch.Tensor] = None,       # [B] bool
        cached_text_tokens:       Optional[torch.Tensor] = None,       # [B, M, T, D_proj]
        cached_text_token_mask:   Optional[torch.Tensor] = None,       # [B, M, T] bool
        cached_text_foil_raw:     Optional[torch.Tensor] = None,       # [B, M, D_proj]
        cached_text_foil_valid:   Optional[torch.Tensor] = None,       # [B, M] bool
        cached_text_foil_tokens:  Optional[torch.Tensor] = None,       # [B, M, T, D_proj]
        cached_text_foil_token_mask: Optional[torch.Tensor] = None,    # [B, M, T] bool
        compute_text_foil:        bool = True,
        mec_image_ids:            Optional[Sequence[str]] = None,      # (b-1) training batch ids
    ) -> Dict[str, Any]:
        """One full pass: encoders -> adapters -> routing -> quantization.

        Routing-mode rule:
            if self.training and part_input_ids is not None:
                routing_mode = "text"
                local centroids = text_part_tokens[:, 1:, :]
            else:
                routing_mode = "codebook_mean"
                local centroids = quantizer.get_codebook_mean_anchors(exclude_global=True)
                                  expanded to [B, 5, D]

        ``visual_tokens`` is the SigLIP2 vision tower output AFTER VisualAdapter.
        SigLIP2 has no [CLS] token, so every position is a patch token; we do
        not strip the first position.

        Returned dict (None when not applicable in the current mode):
            visual_tokens                  [B, N, D]
            text_part_tokens               [B, 6, D] | None
            global_visual_token            [B, D]
            global_text_token              [B, D]    | None
            local_text_tokens              [B, 5, D] | None
            local_anchor_tokens            [B, 5, D] | None
            local_routing_matrix           [B, N, 5] | None
            routing_matrix                 [B, N, 6] | None
                                              col 0 = global avg-pool weights
                                              cols 1..5 = local Sinkhorn plan
            local_semantic_visual_tokens   [B, 5, D] | None
            semantic_visual_tokens         [B, 6, D] | None
            quantized_tokens               [B, 6, D] | None  (STE; downstream-friendly)
            quantized_tokens_raw           [B, 6, D] | None  (pure codewords; no STE)
            codebook_indices               [B, 6]    | None
            codebook_distances             [B, 6, K] | None

            # gated global addition (computed only when return_routing=True):
            head_inputs                    [B, 6, D] | None
                                              [:, 0]   = q_global codeword
                                              [:, 1:]  = q_local + sigmoid(alpha) * (sg)q_global
            global_gate_values             [5] | None        sigmoid(alpha_m), m=1..5
            use_stop_grad_global           bool

            # per-codebook codon-head outputs (computed only when return_routing=True):
            codon_logits_per_codebook      [B, 6, 3, 4] | None    pre-softmax
            continuous_codes_per_codebook  [B, 6, 3, 4] | None    softmax -> A/C/G/T probs
            dna_hash_codes_per_codebook    [B, 6, 3, 4] | None    one-hot of argmax (HARD)
            base_indices_per_codebook      [B, 6, 3]    | None

            # final concatenated 18-position code (= 6 codebooks x 3 codon positions):
            continuous_code                [B, 18, 4] | None
            dna_hash_code                  [B, 18, 4] | None
            base_indices                   [B, 18]    | None

            visual_global_feat             [B, D_proj]
            text_global_feat               [B, 6, D_proj] | None
            routing_mode                   "text" or "codebook_mean"
        """
        # A2 is a strict causal ablation: no tensor derived from captions may
        # influence routing, auxiliary losses, or gradients.  Clear every
        # live/cached factual and foil text input at the boundary so later
        # fallback paths cannot accidentally reconstruct text supervision.
        if bool(getattr(self, "disable_text_supervision", False)):
            part_input_ids = None
            part_attention_mask = None
            cached_text_part_raw = None
            cached_text_tokens = None
            cached_text_token_mask = None
            cached_text_foil_raw = None
            cached_text_foil_valid = None
            cached_text_foil_tokens = None
            cached_text_foil_token_mask = None
            compute_text_foil = False

        # Cached-features path: when the dataloader supplies precomputed
        # SigLIP2 outputs we skip the encoder pass entirely. The text-routing
        # rule then becomes "use text if (training AND any-row-has-real-text)";
        # rows with `cached_has_text==False` got 'none' fallback at extract time
        # but still produce valid (if degenerate) text features, so we still
        # use text routing if ANY row in the batch has real text. (The loss
        # module already masks out 'none' rows via lambda weighting.)
        use_cached = cached_visual_tokens_raw is not None
        if use_cached:
            if pixel_values is not None:
                # Defensive: cached_* takes priority. We do not run the encoder.
                pass
            has_real_text = cached_text_part_raw is not None and cached_has_text is not None and bool(cached_has_text.any().item())
            # A2 ablation: force visual-only (no text supervision, codebook_mean
            # routing during training too). Text-derived losses go to 0 because
            # the text path is never taken.
            if getattr(self, "disable_text_supervision", False):
                has_real_text = False
            use_text_routing = bool(self.training and has_real_text)
            routing_mode = "text" if use_text_routing else "codebook_mean"
            feats = {
                "visual_tokens_raw": cached_visual_tokens_raw,
                "visual_global":     cached_visual_global,
                "text_part_raw":     cached_text_part_raw if use_text_routing else None,
            }
        else:
            use_text_routing = bool(self.training and (part_input_ids is not None))
            routing_mode = "text" if use_text_routing else "codebook_mean"

            # 0) encode (text path only when actually used)
            feats = self.feature_extraction(
                pixel_values=pixel_values,
                part_input_ids=part_input_ids if use_text_routing else None,
                part_attention_mask=part_attention_mask if use_text_routing else None,
                visual_attention_mask=visual_attention_mask,
            )

        # 1) adapter projection (SEPARATE parameters per branch).
        # v141: when --visual_adapter_after_router is set, SKIP the
        # per-patch adapter and apply it AFTER the router (on M=6 tokens).
        if self.visual_adapter_after_router:
            visual_tokens = feats["visual_tokens_raw"]                   # [B, N, D_raw]
        else:
            visual_tokens = self.visual_adapter(feats["visual_tokens_raw"])  # [B, N, D]
        B, N, D = visual_tokens.shape

        text_part_tokens:  Optional[torch.Tensor] = None
        global_text_token: Optional[torch.Tensor] = None
        local_text_tokens: Optional[torch.Tensor] = None
        global_text_token_for_routing: Optional[torch.Tensor] = None
        local_text_tokens_for_routing: Optional[torch.Tensor] = None  # v116
        _cls_mask_text_raw: Optional[torch.Tensor] = None             # [B, M_local, D_shared]
        _bi_visual_keep_union: Optional[torch.Tensor] = None          # [B, N]
        _bi_visual_keep_per_slot: Optional[torch.Tensor] = None       # [B, M_local, N]
        _bi_visual_importance_per_slot: Optional[torch.Tensor] = None # [B, M_local, N]
        _bi_text_keep_per_slot: Optional[torch.Tensor] = None         # [B, M_local, T]
        _bi_visual_keep_ratio_per_slot: Optional[torch.Tensor] = None # [B, M_local]
        _bi_visual_union_keep_ratio: Optional[torch.Tensor] = None    # [B]
        _bi_text_keep_ratio_per_slot: Optional[torch.Tensor] = None   # [B, M_local]
        if use_text_routing and feats["text_part_raw"] is not None:
            # v185: bidirectional token pruning. If enabled, OVERRIDE
            # feats["text_part_raw"] with a mean-pool over kept text tokens
            # BEFORE the whiten/adapter path so ALL downstream text embeddings
            # (whiten, adapter, text_token_attention, losses) observe only
            # the pruned text tokens. Simultaneously compute a visual keep
            # mask to be applied in the fg_mask block below.
            if (
                self.bidirectional_token_prune
                and self.backbone_type == "clip"
                and cached_text_tokens is not None
                and cached_text_tokens.dim() == 4
                and cached_text_tokens.shape[1] >= 2
            ):
                _vp = self.backbone.model.visual_projection                 # Linear(768, 512) frozen
                _v_hidden = feats["visual_tokens_raw"]                      # [B, N, 768]
                _v_shared = _vp(_v_hidden)                                  # [B, N, 512]
                _v_shared_n = F.normalize(_v_shared, dim=-1)                # [B, N, 512]
                # Local slots only (skip cb0 global).
                _tt_loc = cached_text_tokens[:, 1:, :, :]                   # [B, M_loc, T, 512]
                _tm_loc = (
                    cached_text_token_mask[:, 1:, :]
                    if cached_text_token_mask is not None else None
                )                                                            # [B, M_loc, T]
                _tt_loc_n = F.normalize(_tt_loc, dim=-1)                    # [B, M_loc, T, 512]
                _sim_tok = torch.einsum(
                    'bnd,bmtd->bmnt', _v_shared_n, _tt_loc_n
                )                                                            # [B, M_loc, N, T]
                _B, _M_loc, _N, _T = _sim_tok.shape
                assert _B == B and _M_loc == NUM_LOCAL_PARTS and _N == N, (
                    "bidirectional similarity must be [B, M_local, N, T], got "
                    f"{tuple(_sim_tok.shape)}"
                )

                if self.bidirectional_token_prune_mode in (
                    "mutual_dual_softmax", "mutual_consensus_residual",
                ):
                    # Reuse CLIP's pretrained contrast scale; no extra pruning
                    # temperature is introduced. The score stays in the frozen
                    # CLIP shared space used to produce both token streams.
                    _logit_scale = getattr(self.backbone.model, "logit_scale", None)
                    if _logit_scale is not None:
                        _scale = _logit_scale.exp().detach().clamp(max=100.0)
                        _sim_tok = _sim_tok * _scale.to(_sim_tok.dtype)
                    _mutual = _mutual_dual_softmax_pruning(
                        similarity=_sim_tok,
                        text_mask=_tm_loc,
                        visual_mask=visual_attention_mask,
                        visual_keep_ratio=self.bidirectional_token_prune_visual_ratio,
                        text_keep_ratio=self.bidirectional_token_prune_text_ratio,
                        suppress_visual_consensus=(
                            self.bidirectional_token_prune_mode
                            == "mutual_consensus_residual"
                        ),
                    )
                    _v_keep_per_slot = _mutual["visual_keep"]                # [B, M_loc, N]
                    _bi_visual_importance_per_slot = _mutual["visual_importance"]  # [B, M_loc, N]
                    _t_keep_per_slot = _mutual["text_keep"]                 # [B, M_loc, T]
                else:
                    # Legacy v185 path retained for exact experiment
                    # reproducibility. Its softmax-sum importance is constant.
                    _sim_legacy = _sim_tok
                    if _tm_loc is not None:
                        _sim_legacy = _sim_legacy + (
                            ~_tm_loc.unsqueeze(2)
                        ).float() * (-1e4)
                    _attn_v = _sim_legacy.softmax(dim=-1)                    # [B, M_loc, N, T]
                    _v_imp = _attn_v.sum(dim=-1)                             # [B, M_loc, N]
                    _kv = max(1, int(_N * self.bidirectional_token_prune_visual_ratio))
                    _thr_v = _v_imp.topk(_kv, dim=-1)[0][:, :, -1:]          # [B, M_loc, 1]
                    _v_keep_per_slot = (_v_imp >= _thr_v)                    # [B, M_loc, N]

                    _attn_t = _sim_legacy.softmax(dim=-2)                    # [B, M_loc, N, T]
                    _t_imp = _attn_t.sum(dim=-2)                             # [B, M_loc, T]
                    _kt = max(1, int(_T * self.bidirectional_token_prune_text_ratio))
                    _thr_t = _t_imp.topk(_kt, dim=-1)[0][:, :, -1:]          # [B, M_loc, 1]
                    _t_keep_per_slot = (_t_imp >= _thr_t)                    # [B, M_loc, T]
                    if _tm_loc is not None:
                        _t_keep_per_slot = _t_keep_per_slot & _tm_loc

                _v_keep_union = _v_keep_per_slot.any(dim=1)                  # [B, N]
                _bi_visual_keep_union = _v_keep_union
                _bi_visual_keep_per_slot = _v_keep_per_slot
                _bi_text_keep_per_slot = _t_keep_per_slot
                _visual_valid_count = (
                    visual_attention_mask.to(torch.bool).sum(dim=-1)
                    if visual_attention_mask is not None
                    else torch.full(
                        (B,), N, dtype=torch.long, device=visual_tokens.device,
                    )
                ).clamp_min(1)                                                # [B]
                _text_valid_count = (
                    _tm_loc.to(torch.bool).sum(dim=-1)
                    if _tm_loc is not None
                    else torch.full(
                        (B, _M_loc), _T, dtype=torch.long, device=visual_tokens.device,
                    )
                ).clamp_min(1)                                                # [B, M_local]
                _bi_visual_keep_ratio_per_slot = (
                    _v_keep_per_slot.sum(dim=-1) / _visual_valid_count.unsqueeze(-1)
                )                                                             # [B, M_local]
                _bi_visual_union_keep_ratio = (
                    _v_keep_union.sum(dim=-1) / _visual_valid_count
                )                                                             # [B]
                _bi_text_keep_ratio_per_slot = (
                    _t_keep_per_slot.sum(dim=-1) / _text_valid_count
                )                                                             # [B, M_local]

                # Rebuild local text_part_raw as mean over kept text tokens.
                _mask = _t_keep_per_slot.unsqueeze(-1).to(_tt_loc.dtype)     # [B, M_loc, T, 1]
                _num  = (_tt_loc * _mask).sum(dim=2)                         # [B, M_loc, 512]
                _den  = _mask.sum(dim=2).clamp(min=1.0)                      # [B, M_loc, 1]
                _pooled_local_pruned = _num / _den                           # [B, M_loc, 512]

                # C_global slot (cb0) is left unchanged; only local slots
                # (cb1..cb5) are replaced with pruned-pool.
                _cg = feats["text_part_raw"][:, 0:1, :]                      # [B, 1, 512]
                _new_text_part_raw = torch.cat(
                    [_cg, _pooled_local_pruned.to(_cg.dtype)], dim=1
                )                                                             # [B, 6, 512]
                feats = dict(feats)                                          # avoid mutating caller
                feats["text_part_raw"] = _new_text_part_raw

            raw = feats["text_part_raw"]                                    # [B, 6, D_proj]
            if self.routing_cls_verified_consensus_mask:
                _cls_mask_text_raw = raw[:, 1:, :]                           # [B, M_local, D_proj]
                assert _cls_mask_text_raw.shape == (
                    B, NUM_LOCAL_PARTS, self.proj_dim,
                )

            # ----------------------------------------------------------------
            # v113: pre-adapter text-embedding transform (anisotropy fix).
            # Operates on the CACHED CLIP-projection space (D_proj=512 or 768),
            # i.e. BEFORE codebook_text_prompts and text_adapter so the adapter
            # observes the corrected distribution.
            # ----------------------------------------------------------------
            _tx_mode = getattr(self, "text_embed_transform", "none")
            if _tx_mode == "per_image_mean":
                # T_centered = T_local - T_local.mean(dim=1); slot 0 untouched
                _g = raw[:, 0:1, :]
                _loc = raw[:, 1:, :]
                _loc = _loc - _loc.mean(dim=1, keepdim=True)
                _loc = F.normalize(_loc, dim=-1)
                raw = torch.cat([_g, _loc], dim=1)
            elif _tx_mode == "global_residual":
                # T_local <- T_local - T_global (slot 0); keep global as-is
                _g = raw[:, 0:1, :]
                _loc = raw[:, 1:, :] - _g
                _loc = F.normalize(_loc, dim=-1)
                raw = torch.cat([_g, _loc], dim=1)
            elif _tx_mode == "partial_whiten":
                if not getattr(self, "_text_whiten_ready", False):
                    raise RuntimeError(
                        "[model_siglip2] partial_whiten selected but whitening "
                        "buffers were not loaded (see --text_whiten_npz)."
                    )
                _shape = raw.shape
                _flat = raw.reshape(-1, _shape[-1]).to(self.text_whiten_mu.dtype)
                _flat = (_flat - self.text_whiten_mu) @ self.text_whiten_W
                raw = _flat.reshape(_shape).to(feats["text_part_raw"].dtype)
            elif _tx_mode == "global_residual_whiten":
                # v117: keep slot 0 raw; subtract slot 0 from local slots,
                # then apply partial whitening to the residualized local
                # population only. Whitening matrix must have been built
                # via --residualize_first so its (mu, W) match the
                # residualized population's statistics.
                if not getattr(self, "_text_whiten_ready", False):
                    raise RuntimeError(
                        "[model_siglip2] global_residual_whiten selected but "
                        "whitening buffers were not loaded "
                        "(see --text_whiten_npz)."
                    )
                _g = raw[:, 0:1, :]                                              # [B, 1, D]
                _loc = raw[:, 1:, :] - _g                                        # [B, 5, D]
                _orig_dtype = feats["text_part_raw"].dtype
                _flat = _loc.reshape(-1, _loc.shape[-1]).to(self.text_whiten_mu.dtype)
                _flat = (_flat - self.text_whiten_mu) @ self.text_whiten_W
                _loc = _flat.reshape(_loc.shape).to(_orig_dtype)
                raw = torch.cat([_g, _loc], dim=1)                              # [B, 6, D]
            # "phrase_concept" and "none" require no runtime op here.

            # v79b (#2.3): inject learnable per-codebook text prompt bias
            if self.codebook_text_prompts is not None:
                raw = raw + self.codebook_text_prompts.unsqueeze(0)         # [B, 6, D_proj]

            if self.use_text_token_attention and cached_text_tokens is not None:
                # Visual-cross-attention text pooling (Option B / v22b).
                # - cached_text_tokens     : [B, 6, T, D_proj]
                # - cached_text_token_mask : [B, 6, T] bool (True = valid)
                # Adapt each token through the (shared) text_adapter, then
                # for each slot m cross-attend visual_tokens to slot-m text
                # tokens. The visually-conditioned pooled output replaces the
                # static pooled embedding.
                B_, M_, T_, _ = cached_text_tokens.shape
                tk_flat = self.text_adapter(cached_text_tokens.reshape(B_ * M_ * T_, -1))  # [B*6*T, D]
                tk = tk_flat.reshape(B_, M_, T_, self.d_model)                            # [B, 6, T, D]
                attended_slots: List[torch.Tensor] = []
                for m in range(M_):
                    K = tk[:, m, :, :]                                                   # [B, T, D]
                    pad_mask = cached_text_token_mask is None
                    key_padding = None
                    if cached_text_token_mask is not None:
                        # MultiheadAttention's key_padding_mask: True positions are ignored
                        km = cached_text_token_mask[:, m, :].to(torch.bool)
                        key_padding = ~km                                                 # [B, T]
                    attn_out, _ = self.text_token_attn(
                        query=visual_tokens, key=K, value=K,
                        key_padding_mask=key_padding,
                        need_weights=False,
                    )                                                                    # [B, N, D]
                    pooled = attn_out.mean(dim=1)                                        # [B, D]
                    attended_slots.append(pooled)
                text_part_tokens = torch.stack(attended_slots, dim=1)                    # [B, 6, D]
                text_part_tokens = self.text_token_ln(text_part_tokens)
            elif self.per_slot_text_adapter:
                # Each slot gets its own MLP -- forces the six near-identical
                # SigLIP2 pooled inputs into per-slot subspaces.
                per_slot = [self.text_adapter[m](raw[:, m, :]) for m in range(raw.shape[1])]
                text_part_tokens = torch.stack(per_slot, dim=1)             # [B, 6, D]
            else:
                text_part_tokens  = self.text_adapter(raw)                  # [B, 6, D]
            global_text_token = text_part_tokens[:, 0, :]                   # [B, D]
            local_text_tokens = text_part_tokens[:, 1:, :]                  # [B, 5, D]
            global_text_token_for_routing = global_text_token               # [B, D]

            # ----------------------------------------------------------------
            # v116: routing-only mode. If `--text_transform_routing_only` is
            # set AND a transform was applied, run the adapter SECOND TIME on
            # the ORIGINAL (untransformed) cached text_part_raw, and use
            # that untransformed output for ALL downstream losses. The
            # transformed adapter output remains the routing centroid source.
            # ----------------------------------------------------------------
            _routing_only = bool(getattr(self, "text_transform_routing_only", False))
            _tx_active_for_split = _tx_mode in (
                "per_image_mean", "global_residual", "partial_whiten"
            )
            if _routing_only and _tx_active_for_split:
                # keep the transformed text_part_tokens slice for routing
                global_text_token_for_routing = global_text_token
                local_text_tokens_for_routing = local_text_tokens
                # re-run adapter on UNTRANSFORMED raw for losses
                _raw_orig = feats["text_part_raw"]
                if self.codebook_text_prompts is not None:
                    _raw_orig_a = _raw_orig + self.codebook_text_prompts.unsqueeze(0)
                else:
                    _raw_orig_a = _raw_orig
                if self.use_text_token_attention and cached_text_tokens is not None:
                    # complex attention path — fall back: do not split here
                    # (the transformed text_part_tokens already factored in
                    # the cross-attention; re-running would require a full
                    # second cross-attention pass, which is out of scope).
                    pass  # text_part_tokens stays transformed
                elif self.per_slot_text_adapter:
                    _per_slot_orig = [
                        self.text_adapter[m](_raw_orig_a[:, m, :])
                        for m in range(_raw_orig_a.shape[1])
                    ]
                    text_part_tokens = torch.stack(_per_slot_orig, dim=1)
                else:
                    text_part_tokens = self.text_adapter(_raw_orig_a)
                global_text_token = text_part_tokens[:, 0, :]
                local_text_tokens = text_part_tokens[:, 1:, :]
            else:
                global_text_token_for_routing = global_text_token
                local_text_tokens_for_routing = local_text_tokens

        # 2) C_global slot input
        # ----------------------------------------------------------------
        # Two paths (controlled by `--c_global_source`):
        #   `siglip2_global` : route SigLIP2's MAP-pooled + projection-head
        #                       output (`feats["visual_global"]`) through the
        #                       dedicated `global_adapter` linear into d_model.
        #                       This taps the class-aware embedding the
        #                       backbone was trained for.
        #   `mean_pool`       : legacy -- weighted mean of post-adapter
        #                       visual_tokens (no extra learnable params).
        # `global_weights` is exposed in routing_matrix col 0 only for
        # downstream visualization / reporting; it doesn't drive the loss.
        if self.c_global_source == "siglip2_global":
            assert feats.get("visual_global") is not None, (
                "[model_siglip2] c_global_source='siglip2_global' but "
                "feats['visual_global'] was not provided. Either pass "
                "pixel_values (so feature_extraction runs the projection head) "
                "or supply cached_visual_global."
            )
            visual_global_raw = feats["visual_global"].to(visual_tokens.dtype)       # [B, D_proj]
            assert visual_global_raw.dim() == 2 and visual_global_raw.shape[0] == B, (
                f"visual_global shape {tuple(visual_global_raw.shape)} != ({B}, D_proj)"
            )
            global_visual_token = self.global_adapter(visual_global_raw)             # [B, D]
            assert global_visual_token.shape == (B, D)
            # global_weights kept as uniform purely for plotting; col 0 of
            # routing_matrix has no functional role under siglip2_global.
            global_weights = torch.full(
                (B, N, 1), 1.0 / N,
                device=visual_tokens.device, dtype=visual_tokens.dtype,
            )                                                                        # [B, N, 1]
        else:
            # legacy: weighted / uniform mean of post-adapter patches
            if visual_attention_mask is None:
                global_visual_token = visual_tokens.mean(dim=1)                      # [B, D]
                global_weights = torch.full(
                    (B, N, 1), 1.0 / N,
                    device=visual_tokens.device, dtype=visual_tokens.dtype,
                )                                                                    # [B, N, 1]
            else:
                w = visual_attention_mask.to(visual_tokens.dtype)                    # [B, N]
                w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-6)
                global_visual_token = (visual_tokens * w.unsqueeze(-1)).sum(dim=1)   # [B, D]
                global_weights = w.unsqueeze(-1)                                     # [B, N, 1]

        # Always compute the local codebook mean anchors. Used by the
        # codebook_mean routing branch AND exposed in the output dict so the
        # loss module can build the anchor-alignment objective.
        local_codebook_mean_anchors_raw = self.quantizer.get_codebook_mean_anchors(
            exclude_global=True,
        )                                                                # [5, D]

        # 3) pick the centroids that drive the Sinkhorn router
        # v116: under --text_transform_routing_only, the variable
        # local_text_tokens_for_routing == TRANSFORMED slice, while
        # local_text_tokens (used by losses) has been re-bound to the
        # UNTRANSFORMED adapter output. In the default mode both point at
        # the same tensor, so this swap is a no-op.
        # v119: optionally include the slot-0 global caption as a routed
        # C_0 centroid. Legacy mode still routes only C_1..C_5.
        local_anchor_tokens: Optional[torch.Tensor] = None
        route_global_text_active = bool(self.route_global_text and use_text_routing)
        # v141: cluster_attn router always emits M=NUM_SEMANTIC_PARTS=6
        # cluster tokens (cluster 0 plays the role of the global / "C_0"
        # part). Force the global+local routing path so downstream code
        # consumes all 6 tokens from the cluster_attn output.
        if self.router_type == "cluster_attn":
            route_global_text_active = True
        if use_text_routing:
            local_centroids   = (
                local_text_tokens_for_routing
                if local_text_tokens_for_routing is not None
                else local_text_tokens
            )                                                            # [B, 5, D]
            assert local_centroids is not None and local_centroids.shape == (B, NUM_LOCAL_PARTS, D), (
                f"local_centroids shape {None if local_centroids is None else tuple(local_centroids.shape)} "
                f"!= expected ({B}, {NUM_LOCAL_PARTS}, {D})"
            )
            if route_global_text_active:
                global_route_centroid = (
                    global_text_token_for_routing
                    if global_text_token_for_routing is not None
                    else global_text_token
                )                                                        # [B, D]
                assert global_route_centroid is not None and global_route_centroid.shape == (B, D), (
                    f"global_route_centroid shape "
                    f"{None if global_route_centroid is None else tuple(global_route_centroid.shape)} "
                    f"!= expected ({B}, {D})"
                )
                route_centroids = torch.cat(
                    [global_route_centroid.unsqueeze(1), local_centroids], dim=1,
                )                                                        # [B, 6, D]
                route_part_mask_used = part_mask if part_mask is not None else None
            else:
                route_centroids = local_centroids                         # [B, 5, D]
                route_part_mask_used = part_mask[:, 1:] if part_mask is not None else None
        else:
            # v184: at inference, optionally use text_prototype (EMA of trainset
            # text_part_tokens local slots) instead of codebook_mean anchors.
            # Rationale (diagnostic 2026-07-10): codebook_mean at inference has
            # near-uniform cost matrix (routing max 0.01) while text-anchored
            # alignment shows sharp peaks (cos sim max 0.6). Text prototype
            # substitution keeps 0-cost inference while injecting text signal
            # geometry into the routing.
            if (
                (not self.training)
                and getattr(self, "eval_routing_mode", "codebook_mean") == "text_prototype"
                and bool(self._text_prototype_initialized.item())
            ):
                _tp = self.text_prototype_ema.to(local_codebook_mean_anchors_raw.dtype)  # [5, D]
                local_anchor_tokens = _tp.unsqueeze(0).expand(B, -1, -1)   # [B, 5, D]
            else:
                local_anchor_tokens = local_codebook_mean_anchors_raw.unsqueeze(0).expand(B, -1, -1)  # [B, 5, D]
            local_centroids = local_anchor_tokens
            route_centroids = local_centroids                             # [B, 5, D]
            route_part_mask_used = None

        # ---- arch-exp-3 (P2) anchors: axis-deviation transport cost -------
        # Sharpening the routing mask (P1) sent 67 % of patches to exactly one
        # slot and still produced no axis roles, which says the cost itself does
        # not separate the axes. Removing the component that every axis anchor
        # of this image shares leaves the cost with only the differences.
        if self.axis_center in ("anchors", "both") and route_centroids is not None:
            _ng = 1 if route_global_text_active else 0
            route_centroids = self._axis_center_local(
                route_centroids, _ng, route_part_mask_used,
            )

        # v184: EMA update text_prototype during training when text_part_tokens available.
        if (
            self.training
            and text_part_tokens is not None
            and text_part_tokens.shape[1] > 1
        ):
            with torch.no_grad():
                _tp_batch = text_part_tokens[:, 1:, :].mean(dim=0).detach()  # [5, D]
                if not bool(self._text_prototype_initialized.item()):
                    self.text_prototype_ema.copy_(_tp_batch.to(self.text_prototype_ema.dtype))
                    self._text_prototype_initialized.fill_(True)
                else:
                    _decay = float(self.text_prototype_ema_decay)
                    self.text_prototype_ema.mul_(_decay).add_(
                        _tp_batch.to(self.text_prototype_ema.dtype), alpha=(1.0 - _decay)
                    )
        if (
            self.routing_cls_verified_consensus_mask
            and self.training
            and _cls_mask_text_raw is not None
        ):
            with torch.no_grad():
                _raw_tp_batch = _cls_mask_text_raw.mean(dim=0).detach()      # [M_local, D_shared]
                if not bool(self._cls_mask_text_prototype_initialized.item()):
                    self.cls_mask_text_prototype_ema.copy_(
                        _raw_tp_batch.to(self.cls_mask_text_prototype_ema.dtype)
                    )
                    self._cls_mask_text_prototype_initialized.fill_(True)
                else:
                    _decay = float(self.text_prototype_ema_decay)
                    self.cls_mask_text_prototype_ema.mul_(_decay).add_(
                        _raw_tp_batch.to(self.cls_mask_text_prototype_ema.dtype),
                        alpha=(1.0 - _decay),
                    )

        out: Dict[str, Any] = {
            "visual_tokens":                visual_tokens,
            "text_part_tokens":             text_part_tokens,
            "global_visual_token":          global_visual_token,
            "global_text_token":            global_text_token,
            "local_text_tokens":            local_text_tokens,
            "local_anchor_tokens":          local_anchor_tokens,
            "local_codebook_mean_anchors":  local_codebook_mean_anchors_raw,   # [5, D]
            "visual_global_feat":           feats["visual_global"],
            "text_global_feat":             feats["text_part_raw"],     # [B, 6, D_proj] or None
            "local_routing_matrix":         None,
            "routing_matrix":               None,
            "routing_mean_effective_k":      None,
            "routing_fraction_top1":         None,
            "routing_visual_specificity_mean": None,
            "routing_visual_marginal_effective_ratio": None,
            "routing_visual_consensus_candidate_ratio": None,
            "routing_visual_cls_low_ratio": None,
            "routing_visual_consensus_mask_ratio": None,
            "routing_visual_consensus_remaining_ratio": None,
            "routing_visual_consensus_fallback": None,
            "bidirectional_visual_keep_ratio_per_slot": _bi_visual_keep_ratio_per_slot,
            "bidirectional_visual_union_keep_ratio": _bi_visual_union_keep_ratio,
            "bidirectional_text_keep_ratio_per_slot": _bi_text_keep_ratio_per_slot,
            "bidirectional_visual_importance_per_slot": _bi_visual_importance_per_slot,
            "soft_grounded_text_attention_entropy": None,
            "cosine_grounded_text_attention_entropy": None,
            "cosine_grounded_text_effective_support": None,
            "cosine_grounded_text_gate_mean": None,
            "cosine_grounded_text_gate_abs_mean": None,
            "local_semantic_visual_tokens": None,
            "semantic_visual_tokens":       None,
            "cibhash_visual_tokens":         None,
            "quantizer_input":              None,
            "quantized_tokens":             None,
            "quantized_tokens_raw":         None,
            "codebook_indices":             None,
            "codebook_distances":           None,
            # gated global addition + codon heads (filled only when return_routing=True)
            "head_inputs":                   None,
            "global_gate_values":            None,
            "use_stop_grad_global":          self.use_stop_grad_global,
            "codon_logits_per_codebook":          None,
            "continuous_codes_per_codebook":      None,
            "dna_hash_codes_per_codebook":        None,
            "dna_hash_codes_per_codebook_hard":   None,
            "dna_hash_codes_per_codebook_st":     None,
            "base_indices_per_codebook":          None,
            "continuous_code":                    None,
            "dna_hash_code":                      None,
            "dna_hash_code_hard":                 None,
            "dna_hash_code_st":                   None,
            "base_indices":                       None,
            "text_continuous_code":               None,
            "text_quantized_tokens":              None,
            "codeword_text_tokens":               None,
            "loss_text_anchor":                   None,
            "hash_recon_pred":                    None,
            "dual_hash_semantic":                 None,
            "dual_hash_instance":                 None,
            "routing_mode":                       routing_mode,
        }

        if not return_routing:
            return out

        # 4) Sinkhorn OT routing on the 5 LOCAL parts.
        # v33a: pass an annealed epsilon if --sinkhorn_epsilon_init / _final
        # are set; otherwise router uses its stored static epsilon.
        # v33b: top-k mask per patch (only for Sinkhorn router; attention
        # router ignores).
        # v55: --sinkhorn_lambda_a / _b switches to unbalanced OT (Chizat
        # et al. NeurIPS 2018) — uninformative patches can have row sum
        # < 1/N (partial rejection).
        # v56: --use_null_centroid prepends a learnable null/background
        # centroid to the 5 text centroids before Sinkhorn, then slices
        # the null column out post-routing.
        if self.use_null_centroid and self.null_centroid is not None:
            # Expand learnable null to per-sample dim and append as the last
            # route-only/background centroid.
            B_loc = route_centroids.shape[0]
            null_exp = self.null_centroid.unsqueeze(0).unsqueeze(0).expand(B_loc, 1, -1)  # [B, 1, D]
            route_centroids_aug = torch.cat([route_centroids, null_exp], dim=1)         # [B, 6/7, D]
            if route_part_mask_used is not None:
                # Always-keep null part in mask
                ones = torch.ones(B_loc, 1, dtype=route_part_mask_used.dtype,
                                  device=route_part_mask_used.device)
                route_part_mask_used_aug = torch.cat([route_part_mask_used, ones], dim=1)
            else:
                route_part_mask_used_aug = None
        else:
            route_centroids_aug = route_centroids
            route_part_mask_used_aug = route_part_mask_used

        # v113 Method 2 companion: residualize patches against patch-mean and
        # L2-normalize BEFORE feeding the router. visual_tokens used elsewhere
        # (codeword extraction, losses, gated global add) stays unmodified.
        if getattr(self, "residualize_visual_for_routing", False):
            _v_global = visual_tokens.mean(dim=1, keepdim=True)          # [B, 1, D]
            visual_tokens_for_routing = F.normalize(
                visual_tokens - _v_global, dim=-1
            )
        else:
            visual_tokens_for_routing = visual_tokens

        _cls_verified_diag: Optional[Dict[str, torch.Tensor]] = None

        # Foreground-text mask: cosine(visual_tokens, C_global text) → keep
        # top-K% patches; rest get visual_attention_mask=0 so they are excluded
        # from all 6 codebook updates. Mass conservation still holds on the
        # remaining K patches. Designed for single-object fine-grained datasets
        # (CUB-200) where background tokens dominate the router by patch count.
        # v185/v186: the union controls only the visual row marginal. In
        # mutual-dual-softmax mode, the distinct [B, M_local, N] masks are also
        # applied to their matching Sinkhorn columns below.
        if self.bidirectional_token_prune and _bi_visual_keep_union is not None:
            _fg_mask_bi = _bi_visual_keep_union.to(visual_tokens_for_routing.dtype)  # [B, N]
            if visual_attention_mask is not None:
                visual_attention_mask = visual_attention_mask.to(
                    visual_tokens_for_routing.dtype
                ) * _fg_mask_bi
            else:
                visual_attention_mask = _fg_mask_bi
        fg_ratio = self.foreground_text_mask_topk_ratio
        if (
            not self.bidirectional_token_prune
            and fg_ratio is not None
            and 0.0 < fg_ratio < 1.0
            and text_part_tokens is not None
        ):
            # v175: choose text anchor for foreground mask.
            #   global         -> cb0 C_global text (default, legacy).
            #   local_pooled   -> GAP over local slots cb1..cb5 (anatomy-focused).
            #   per_slot_union -> independent top-K per local slot, take UNION.
            #   per_slot_token_attention -> v182: pre-adapter cross-attention on
            #     raw CLIP text tokens (per-token) and CLIP-projected visual
            #     patches; per-slot importance from full attention scores;
            #     UNION across slots for visual keep-mask.
            _v_n = F.normalize(visual_tokens_for_routing, dim=-1)           # [B, N, D]
            _N = _v_n.shape[1]
            _k = max(1, int(_N * fg_ratio))
            _use_low_attn = (
                self.foreground_text_mask_source == "per_slot_token_attention_low"
            )
            if (
                self.foreground_text_mask_source in (
                    "per_slot_token_attention",
                    "per_slot_token_attention_low",
                )
                and cached_text_tokens is not None
                and cached_text_tokens.shape[1] > 1
                and self.backbone_type == "clip"
            ):
                # v182: token-level per-slot cross-attention.
                #   text_tokens_raw : [B, M, T, D_proj=512]  (CLIP-projected)
                #   visual_tokens_raw: [B, N, H_v=768] (pre-adapter hidden)
                # Project visual patches to shared 512 via CLIP's visual_projection
                # (frozen; no new params). Then compute per-slot cross-attention.
                _vp = self.backbone.model.visual_projection                 # Linear(768, 512)
                _v_hidden = feats["visual_tokens_raw"]                      # [B, N, 768]
                _v_shared = _vp(_v_hidden)                                  # [B, N, 512]
                _v_shared_n = F.normalize(_v_shared, dim=-1)
                # Local slots only (skip cb0)
                _tt = cached_text_tokens[:, 1:, :, :]                       # [B, M_loc, T, 512]
                _tm = cached_text_token_mask[:, 1:, :] if cached_text_token_mask is not None else None
                _tt_n = F.normalize(_tt, dim=-1)                            # [B, M_loc, T, 512]
                # sim[b, m, n, t] = cos(v_shared[b, n], text[b, m, t])
                _sim_tok = torch.einsum('bnd,bmtd->bmnt', _v_shared_n, _tt_n)  # [B, M_loc, N, T]
                if _tm is not None:
                    _sim_tok = _sim_tok + (~_tm.unsqueeze(2)).float() * (-1e4)
                # per-patch importance per slot = sum over valid text tokens of softmax scores
                _attn = _sim_tok.softmax(dim=-1)                            # [B, M_loc, N, T]
                _v_imp_slot = _attn.sum(dim=-1)                             # [B, M_loc, N]
                # per-slot top-K (or bottom-K if 'low' variant), then union across slots.
                # 'low' variant tests ICML26 fine-grained hypothesis: species-distinctive
                # patches have LOW attention to generic caption words; keeping bottom-K
                # attention scores may retain more discriminative subtle features.
                if _use_low_attn:
                    _thr_per_slot = _v_imp_slot.topk(_k, dim=-1, largest=False)[0][:, :, -1:]  # [B, M_loc, 1]
                    _per_slot_mask = (_v_imp_slot <= _thr_per_slot)         # [B, M_loc, N]
                else:
                    _thr_per_slot = _v_imp_slot.topk(_k, dim=-1)[0][:, :, -1:]  # [B, M_loc, 1]
                    _per_slot_mask = (_v_imp_slot >= _thr_per_slot)         # [B, M_loc, N]
                _fg_mask = _per_slot_mask.any(dim=1).to(_v_n.dtype)         # [B, N]
            elif (
                self.foreground_text_mask_source == "per_slot_union"
                and text_part_tokens.shape[1] > 1
            ):
                # Per-slot top-K, union over local slots (cb1..cb5).
                _local_text = text_part_tokens[:, 1:, :]                    # [B, M_loc, D]
                _local_text_n = F.normalize(_local_text, dim=-1)            # [B, M_loc, D]
                _sims = torch.einsum('bnd,bmd->bnm', _v_n, _local_text_n)   # [B, N, M_loc]
                # Per-slot top-K threshold; mask[m, p]=1 if patch p in top-K for slot m.
                _thr_per_slot = _sims.topk(_k, dim=1)[0][:, -1:, :]         # [B, 1, M_loc]
                _per_slot_mask = (_sims >= _thr_per_slot)                   # [B, N, M_loc]
                _fg_mask = _per_slot_mask.any(dim=-1).to(_v_n.dtype)        # [B, N]
            else:
                if (
                    self.foreground_text_mask_source == "local_pooled"
                    and text_part_tokens.shape[1] > 1
                ):
                    _g_text = text_part_tokens[:, 1:, :].mean(dim=1)        # [B, D]
                else:
                    _g_text = text_part_tokens[:, 0, :]                     # [B, D]
                _g_n = F.normalize(_g_text, dim=-1)                         # [B, D]
                _sim = (_v_n * _g_n.unsqueeze(1)).sum(dim=-1)               # [B, N]
                _thr = _sim.topk(_k, dim=-1)[0][:, -1:]                     # [B, 1]
                _fg_mask = (_sim >= _thr).to(visual_tokens_for_routing.dtype)   # [B, N]
            if visual_attention_mask is not None:
                visual_attention_mask = visual_attention_mask.to(
                    visual_tokens_for_routing.dtype
                ) * _fg_mask
            else:
                visual_attention_mask = _fg_mask

        if self.routing_cls_verified_consensus_mask:
            assert self.backbone_type == "clip", (
                "--routing_cls_verified_consensus_mask requires CLIP shared-space features"
            )
            assert feats.get("visual_global") is not None
            _visual_projection = self.backbone.model.visual_projection
            _visual_shared = _visual_projection(
                feats["visual_tokens_raw"]
            )                                                               # [B, N, D_shared]
            assert _visual_shared.shape == (B, N, self.proj_dim)
            _global_shared = feats["visual_global"].to(_visual_shared.dtype) # [B, D_shared]
            assert _global_shared.shape == (B, self.proj_dim)
            if _cls_mask_text_raw is not None:
                _local_text_shared = _cls_mask_text_raw.to(_visual_shared.dtype)
            else:
                if not bool(self._cls_mask_text_prototype_initialized.item()):
                    raise RuntimeError(
                        "CLS-verified mask inference requires initialized raw text prototypes"
                    )
                _local_text_shared = self.cls_mask_text_prototype_ema.to(
                    device=_visual_shared.device, dtype=_visual_shared.dtype,
                ).unsqueeze(0).expand(B, -1, -1)                            # [B, M_local, D_shared]
            assert _local_text_shared.shape == (B, NUM_LOCAL_PARTS, self.proj_dim)
            _cls_verified_diag = _cls_verified_consensus_mask(
                visual_tokens_shared=_visual_shared,
                local_text_shared=_local_text_shared,
                visual_global_shared=_global_shared,
                visual_mask=visual_attention_mask,
            )
            visual_attention_mask = _cls_verified_diag["keep"]             # [B, N]

        router_kwargs = {
            "visual_tokens":    visual_tokens_for_routing,
            "text_part_tokens": route_centroids_aug,           # [B, 5/6 (+ null), D]
            "visual_mask":      visual_attention_mask,
            "part_mask":        route_part_mask_used_aug,
        }
        if self.router_type == "sinkhorn" and (
                self.routing_mass_alpha > 0.0 or self.null_free_marginal):
            router_kwargs.update(self._routing_mass_kwargs(
                feats, route_centroids_aug, route_global_text_active))
        if self.router_type == "sinkhorn":
            # v186: slot-specific visual pruning support. The row mask above
            # removes patches selected by no local slot. This [B, N, M_route]
            # bias additionally prevents a patch selected for slot m from being
            # reused by a different slot. Global/null route-only columns remain
            # available on every surviving row.
            cost_bias: Optional[torch.Tensor] = None
            if (
                self.bidirectional_token_prune
                and self.bidirectional_token_prune_mode in (
                    "mutual_dual_softmax", "mutual_consensus_residual",
                )
                and not self.bidirectional_prune_only
                and _bi_visual_keep_per_slot is not None
            ):
                assert _bi_visual_keep_per_slot.shape == (B, NUM_LOCAL_PARTS, N), (
                    "slot-specific visual keep mask must be [B, M_local, N], got "
                    f"{tuple(_bi_visual_keep_per_slot.shape)}"
                )
                _local_route_keep = _bi_visual_keep_per_slot.transpose(1, 2)  # [B, N, M_local]
                _surviving_rows = (
                    visual_attention_mask.to(torch.bool)
                    if visual_attention_mask is not None
                    else torch.ones(B, N, dtype=torch.bool, device=visual_tokens.device)
                )                                                             # [B, N]
                _route_keep_parts = []
                if route_global_text_active:
                    _route_keep_parts.append(_surviving_rows.unsqueeze(-1))   # [B, N, 1]
                _route_keep_parts.append(_local_route_keep)                   # [B, N, 5]
                if self.use_null_centroid and self.null_centroid is not None:
                    _route_keep_parts.append(_surviving_rows.unsqueeze(-1))   # [B, N, 1]
                _route_keep = torch.cat(_route_keep_parts, dim=-1)            # [B, N, M_route]
                assert _route_keep.shape == (
                    B, N, route_centroids_aug.shape[1],
                ), (
                    "slot-specific route mask must match router columns, got "
                    f"{tuple(_route_keep.shape)} vs "
                    f"{(B, N, route_centroids_aug.shape[1])}"
                )
                cost_bias = torch.zeros(
                    _route_keep.shape,
                    dtype=visual_tokens_for_routing.dtype,
                    device=visual_tokens_for_routing.device,
                ).masked_fill(~_route_keep, -1e4)                             # [B, N, M_route]

            cur_text_evidence_beta = self._current_routing_text_evidence_beta()
            cur_text_evidence_penalty = self._current_routing_text_evidence_penalty()
            cur_token_ot_beta = self._current_routing_token_ot_beta()
            if (
                cur_text_evidence_beta > 0.0
                or cur_text_evidence_penalty > 0.0
                or (cur_token_ot_beta > 0.0 and use_text_routing)
            ):
                # v176/v177: soft evidence bias for Sinkhorn routing.
                # visual_tokens_for_routing: [B, N, D]
                # route_centroids_aug:       [B, M_route, D]
                # sim/cost_bias:             [B, N, M_route]
                _v_ev = F.normalize(visual_tokens_for_routing, dim=-1)
                _t_ev = F.normalize(route_centroids_aug, dim=-1)
                sim_ev = torch.einsum("bnd,bmd->bnm", _v_ev, _t_ev)
                evidence_cost_bias = cur_text_evidence_beta * sim_ev
                if cur_token_ot_beta > 0.0 and use_text_routing:
                    if cached_text_tokens is None:
                        raise RuntimeError(
                            "[model_siglip2] --routing_token_ot_evidence requires "
                            "cached_text_tokens in text-routing training mode. "
                            "Use a *_tokens feature cache built by "
                            "extract_clip_text_token_features.py."
                        )
                    token_ot_bias = self._routing_token_ot_cost_bias(
                        visual_tokens_for_routing=visual_tokens_for_routing,
                        route_centroids_aug=route_centroids_aug,
                        cached_text_tokens=cached_text_tokens,
                        cached_text_token_mask=cached_text_token_mask,
                        visual_attention_mask=visual_attention_mask,
                        route_global_text_active=route_global_text_active,
                    )
                    assert token_ot_bias.shape == evidence_cost_bias.shape, (
                        "token OT routing bias must match cost_bias, got "
                        f"{tuple(token_ot_bias.shape)} vs {tuple(evidence_cost_bias.shape)}"
                    )
                    evidence_cost_bias = evidence_cost_bias + cur_token_ot_beta * token_ot_bias
                assert evidence_cost_bias.shape == (
                    visual_tokens_for_routing.shape[0],
                    visual_tokens_for_routing.shape[1],
                    route_centroids_aug.shape[1],
                ), (
                    "routing text-evidence bias must be [B, N, M_route], got "
                    f"{tuple(evidence_cost_bias.shape)}"
                )
                keep_ratio = float(getattr(self, "routing_text_evidence_keep_ratio", 1.0))
                if cur_text_evidence_penalty > 0.0 and 0.0 < keep_ratio < 1.0:
                    # v177: evidence-aware local candidate pressure. For each
                    # local route column, keep top-r visual tokens by text
                    # evidence; outside candidates are not deleted, but receive
                    # a negative bias so Sinkhorn can still use them if needed.
                    B_ev, N_ev, M_route = sim_ev.shape
                    local_start = 1 if route_global_text_active else 0
                    local_end = min(local_start + NUM_LOCAL_PARTS, M_route)
                    if local_end > local_start:
                        local_sim = sim_ev[..., local_start:local_end]       # [B, N, M_local]
                        assert local_sim.shape == (
                            B_ev, N_ev, local_end - local_start,
                        ), (
                            "local evidence sim must be [B, N, M_local], got "
                            f"{tuple(local_sim.shape)}"
                        )
                        k_keep = max(1, min(N_ev, int(math.ceil(N_ev * keep_ratio))))
                        local_sim_for_topk = local_sim
                        if visual_attention_mask is not None:
                            valid_patch = visual_attention_mask.to(torch.bool).unsqueeze(-1)  # [B, N, 1]
                            local_sim_for_topk = local_sim.masked_fill(~valid_patch, -float("inf"))
                        # Threshold per local route column: [B, 1, M_local].
                        kth = local_sim_for_topk.topk(k=k_keep, dim=1).values[:, -1:, :]
                        keep = local_sim_for_topk >= kth                       # [B, N, M_local]
                        assert keep.shape == local_sim.shape, (
                            "local evidence keep mask must match local sim, got "
                            f"{tuple(keep.shape)} vs {tuple(local_sim.shape)}"
                        )
                        # Map cosine evidence [-1, 1] to confidence [0, 1] so
                        # the penalty is bounded and strongest on clear non-evidence.
                        evidence01 = (0.5 * (local_sim + 1.0)).clamp(0.0, 1.0)
                        penalty_term = cur_text_evidence_penalty * (1.0 - evidence01)
                        local_bias = evidence_cost_bias[..., local_start:local_end]
                        local_bias = torch.where(keep, local_bias, local_bias - penalty_term)
                        evidence_cost_bias = evidence_cost_bias.clone()
                        evidence_cost_bias[..., local_start:local_end] = local_bias
                if self.use_null_centroid and self.null_centroid is not None:
                    evidence_cost_bias[..., -1] = 0.0
                cost_bias = (
                    evidence_cost_bias
                    if cost_bias is None
                    else cost_bias + evidence_cost_bias
                )
            if cost_bias is not None:
                router_kwargs["cost_bias"] = cost_bias.detach()
            cur_eps = self._current_sinkhorn_epsilon()
            if cur_eps is not None:
                router_kwargs["epsilon_override"] = cur_eps
            if self.routing_topk is not None:
                router_kwargs["topk_per_patch"]   = int(self.routing_topk)
            if self.routing_topp is not None:
                router_kwargs["topp_per_patch"]   = float(self.routing_topp)
            if self.routing_ambiguity_topk:
                router_kwargs["ambiguity_topk_threshold"] = float(self.routing_ambiguity_threshold)
                router_kwargs["ambiguity_topk_ambiguous_k"] = int(self.routing_ambiguity_k)
            if self.routing_adaptive_topp:
                router_kwargs["adaptive_topp_min"] = float(self.routing_adaptive_topp_min)
                router_kwargs["adaptive_topp_max"] = float(self.routing_adaptive_topp_max)
                router_kwargs["adaptive_topp_use_entropy"] = bool(self.routing_adaptive_topp_entropy)
            if self.routing_specificity_marginal:
                if route_global_text_active:
                    raise ValueError(
                        "--routing_specificity_marginal currently supports the "
                        "five local routing slots only; disable --route_global_text."
                    )
                if self.use_null_centroid and self.null_centroid is not None:
                    raise ValueError(
                        "--routing_specificity_marginal does not include a null "
                        "centroid in its slot entropy; disable --use_null_centroid."
                    )
                router_kwargs["specificity_weighted_marginal"] = True
            if self.routing_centered_consensus_mask:
                if route_global_text_active:
                    raise ValueError(
                        "--routing_centered_consensus_mask supports the five "
                        "local routing slots only; disable --route_global_text."
                    )
                if self.use_null_centroid and self.null_centroid is not None:
                    raise ValueError(
                        "--routing_centered_consensus_mask does not include a "
                        "null centroid; disable --use_null_centroid."
                    )
                router_kwargs["centered_consensus_mask"] = True
            if self.routing_cls_verified_consensus_mask:
                router_kwargs["hard_visual_mask"] = True
            if self.routing_perplexity_topk:
                if self.routing_adaptive_topp:
                    raise ValueError(
                        "--routing_perplexity_topk and --routing_adaptive_topp "
                        "are mutually exclusive; pick one routing mask policy."
                    )
                router_kwargs["perplexity_topk"] = True
            if self.routing_codebook_choice:
                router_kwargs["codebook_choice_capacity"] = float(self.routing_codebook_choice_capacity)
                router_kwargs["codebook_choice_beta"] = self._current_codebook_choice_beta()
            if self.sinkhorn_lambda_a is not None:
                router_kwargs["uot_lambda_a"]     = float(self.sinkhorn_lambda_a)
            if self.sinkhorn_lambda_b is not None:
                router_kwargs["uot_lambda_b"]     = float(self.sinkhorn_lambda_b)
        elif self.router_type == "attention":
            # v107a-attn: forward the confidence-adaptive top-p kwargs to the
            # attention router (analogous to sinkhorn router's adaptive_topp).
            if self.routing_adaptive_topp:
                router_kwargs["adaptive_topp_min"] = float(self.routing_adaptive_topp_min)
                router_kwargs["adaptive_topp_max"] = float(self.routing_adaptive_topp_max)
                router_kwargs["adaptive_topp_use_entropy"] = bool(self.routing_adaptive_topp_entropy)
        if self.bidirectional_prune_only:
            if route_global_text_active:
                raise ValueError(
                    "--bidirectional_prune_only does not support "
                    "--route_global_text; C0 uses its existing global path."
                )
            if self.use_null_centroid and self.null_centroid is not None:
                raise ValueError(
                    "--bidirectional_prune_only does not support a null centroid."
                )

            if _bi_visual_keep_per_slot is None:
                # Image-only evaluation has no captions. Reconstruct the same
                # consensus-residual selector from learned slot anchors rather
                # than falling back to OT routing.
                assert local_anchor_tokens is not None and local_anchor_tokens.shape == (
                    B, NUM_LOCAL_PARTS, D,
                ), (
                    "prune-only image evaluation requires local slot anchors "
                    f"[B, M_local, D], got "
                    f"{None if local_anchor_tokens is None else tuple(local_anchor_tokens.shape)}"
                )
                _v_anchor_n = F.normalize(visual_tokens_for_routing, dim=-1)  # [B, N, D]
                _anchor_n = F.normalize(local_anchor_tokens, dim=-1)          # [B, M_local, D]
                _anchor_sim = torch.einsum(
                    "bnd,bmd->bmn", _v_anchor_n, _anchor_n,
                )                                                             # [B, M_local, N]
                _eval_visual_valid = (
                    visual_attention_mask.to(torch.bool)
                    if visual_attention_mask is not None
                    else torch.ones(B, N, dtype=torch.bool, device=visual_tokens.device)
                )                                                             # [B, N]
                _anchor_sim = _anchor_sim.masked_fill(
                    ~_eval_visual_valid[:, None, :], -float("inf"),
                )
                _anchor_importance = _anchor_sim.softmax(dim=-1)              # [B, M_local, N]
                _bi_visual_importance_per_slot = _slot_consensus_residual_importance(
                    _anchor_importance,
                    visual_mask=_eval_visual_valid,
                )                                                             # [B, M_local, N]
                _bi_visual_keep_per_slot = _topk_visual_keep(
                    _bi_visual_importance_per_slot,
                    visual_mask=_eval_visual_valid,
                    keep_ratio=self.bidirectional_token_prune_visual_ratio,
                )                                                             # [B, M_local, N]
                _bi_visual_keep_union = _bi_visual_keep_per_slot.any(dim=1)   # [B, N]
                _valid_count = _eval_visual_valid.sum(dim=-1).clamp_min(1)    # [B]
                _bi_visual_keep_ratio_per_slot = (
                    _bi_visual_keep_per_slot.sum(dim=-1)
                    / _valid_count.unsqueeze(-1)
                )                                                             # [B, M_local]
                _bi_visual_union_keep_ratio = (
                    _bi_visual_keep_union.sum(dim=-1) / _valid_count
                )                                                             # [B]

            assert _bi_visual_keep_per_slot.shape == (B, NUM_LOCAL_PARTS, N), (
                "prune-only selection mask must be [B, M_local, N], got "
                f"{tuple(_bi_visual_keep_per_slot.shape)}"
            )
            prune_selection = _bi_visual_keep_per_slot.transpose(1, 2).to(
                visual_tokens.dtype
            )                                                                 # [B, N, M_local]
            r_out = {
                "routing_matrix": prune_selection,
                "ot_cost": None,
            }
            # Diagnostics may have been constructed after the initial output
            # dictionary in the image-only evaluation branch.
            out["bidirectional_visual_keep_ratio_per_slot"] = (
                _bi_visual_keep_ratio_per_slot
            )
            out["bidirectional_visual_union_keep_ratio"] = (
                _bi_visual_union_keep_ratio
            )
            out["bidirectional_visual_importance_per_slot"] = (
                _bi_visual_importance_per_slot
            )
        elif self.router_type == "cluster_attn":
            # v141: ClusterAttentionRouter operates on raw patches only;
            # no text_part_tokens / part_mask / topp kwargs apply. It
            # returns {routing_matrix, semantic_tokens, ot_cost}.
            r_out = self.router(visual_tokens_for_routing)
        else:
            r_out = self.router(**router_kwargs)
        full_routing_matrix = r_out["routing_matrix"]              # [B, N, 5/6 (+ null)]
        # Pre-sparsification plan for the slot-diversity loss; only the
        # sinkhorn router publishes it, so keep this None-safe.
        _premask = r_out.get("routing_matrix_premask", None)
        routing_matrix_premask = (
            _premask[..., :-1]
            if (_premask is not None and self.use_null_centroid
                and self.null_centroid is not None)
            else _premask
        )
        if self.use_null_centroid and self.null_centroid is not None:
            routed_matrix = full_routing_matrix[..., :-1]           # [B, N, 5/6]
        else:
            routed_matrix = full_routing_matrix                     # [B, N, 5/6]

        # v79c (#4.1): hard routing via Gumbel-Softmax. After Sinkhorn
        # soft routing, re-cast each patch row to one-hot via
        # gumbel_softmax(hard=True). Sum over patches per part still
        # makes sense; each patch then contributes to exactly one part.
        if bool(getattr(self, "routing_hard", False)) and self.training:
            # gumbel_softmax expects logits; treat log(soft+eps) as logits
            tau = float(getattr(self, "routing_hard_tau", 1.0))
            log_p = torch.log(routed_matrix.clamp_min(1e-9))
            routed_matrix = F.gumbel_softmax(
                log_p, tau=tau, hard=True, dim=-1
            )                                                       # [B, N, 5/6] one-hot per patch
        elif bool(getattr(self, "routing_hard", False)):
            # eval: deterministic argmax one-hot
            idx = routed_matrix.argmax(dim=-1)                      # [B, N]
            one_hot = F.one_hot(idx, num_classes=routed_matrix.shape[-1]).to(routed_matrix.dtype)
            routed_matrix = one_hot

        assert routed_matrix.dim() == 3, (
            f"routed_matrix must be [B, N, 5/6], got {tuple(routed_matrix.shape)}"
        )
        expected_route_parts = NUM_SEMANTIC_PARTS if route_global_text_active else NUM_LOCAL_PARTS
        assert routed_matrix.shape[0] == B and routed_matrix.shape[-1] == expected_route_parts, (
            f"routed_matrix shape {tuple(routed_matrix.shape)} "
            f"does not match B={B}, M={expected_route_parts}"
        )
        local_routing_matrix = (
            routed_matrix[:, :, 1:] if route_global_text_active else routed_matrix
        )                                                       # [B, N, 5]
        route_nonzero = local_routing_matrix > 0.0                   # [B, N, 5]
        routing_effective_k = route_nonzero.to(local_routing_matrix.dtype).sum(dim=-1)  # [B, N]
        if (
            self.routing_centered_consensus_mask
            or self.routing_cls_verified_consensus_mask
        ):
            route_has_mass = local_routing_matrix.sum(dim=-1) > 0.0  # [B, N]
            active_count = route_has_mass.sum().clamp_min(1).to(local_routing_matrix.dtype)
            routing_fraction_top1 = (
                ((routing_effective_k <= 1.0) & route_has_mass)
                .to(local_routing_matrix.dtype).sum() / active_count
            )
            routing_mean_effective_k = (
                routing_effective_k.sum() / active_count
            )
        else:
            routing_fraction_top1 = (
                (routing_effective_k <= 1.0)
                .to(local_routing_matrix.dtype).mean()
            )
            routing_mean_effective_k = routing_effective_k.mean()

        if route_global_text_active:
            # v119: all six codebooks, including C_0, are pooled from visual
            # tokens using their own text-driven routing columns.
            global_weights = routed_matrix[:, :, :1]                                  # [B, N, 1]
            routing_matrix = routed_matrix                                            # [B, N, 6]
            denom_all = routing_matrix.sum(dim=1).clamp_min(1e-6)                     # [B, 6]
            semantic_visual_tokens = (
                torch.bmm(routing_matrix.transpose(1, 2), visual_tokens)              # [B, 6, D]
                / denom_all.unsqueeze(-1)
            )
            local_semantic_visual_tokens = semantic_visual_tokens[:, 1:, :]           # [B, 5, D]
        else:
            # explicit re-normalize for numerical stability
            denom = local_routing_matrix.sum(dim=1).clamp_min(1e-6)                   # [B, 5]
            local_semantic_visual_tokens = (
                torch.bmm(local_routing_matrix.transpose(1, 2), visual_tokens)        # [B, 5, D]
                / denom.unsqueeze(-1)
            )

            # ---- sequential cross-slot residual pooling (2026-07-21) --------
            # Diagnosis it targets: every local slot pools ~61-64% of the SAME
            # 196 patches (measured eff_k 120-125/196 on all datasets), so the
            # six z^m are re-weightings of one global content and carry highly
            # redundant information (local-slot pairwise NMI 0.74-0.82;
            # cross-slot decoding puts only 3/6 slots' own code at the column
            # argmax). Per-slot text supervision cannot induce specialisation
            # when the visual evidence is not separable.
            #
            # Mechanism: pool slot m from the token residual left by slots
            # 1..m-1 -- each token has the component already explained by an
            # earlier slot removed before the next slot reads it. This is
            # CONSTRUCTIVE (a reparameterisation of what each slot sees), not a
            # redundancy penalty: penalising inter-slot MI was tried and
            # rejected earlier because it traded away retrieval, and sharpening
            # / pruning the routing was refuted separately.
            #
            # Distinct from `--local_residual_quant` (v132a, no-op/harmful),
            # which removes only the single global C0 projection from the slot
            # VECTORS. Here the removal is token-level and chained across the
            # five local slots. Routing weights are untouched, so this is a
            # single delta on the pooling step alone.
            if self.slot_sequential_residual and self.slot_seq_residual_gamma > 0.0:
                g = float(self.slot_seq_residual_gamma)
                v_res = visual_tokens                                             # [B, N, D]
                pooled: List[torch.Tensor] = []
                n_local = local_routing_matrix.shape[-1]
                for m in range(n_local):
                    w = local_routing_matrix[:, :, m]                             # [B, N]
                    z_m = torch.bmm(w.unsqueeze(1), v_res).squeeze(1) / \
                        w.sum(dim=1).clamp_min(1e-6).unsqueeze(-1)                # [B, D]
                    pooled.append(z_m)
                    if m < n_local - 1:
                        z_dir = F.normalize(z_m, dim=-1)                          # [B, D]
                        coef = torch.einsum("bnd,bd->bn", v_res, z_dir)           # [B, N]
                        v_res = v_res - g * coef.unsqueeze(-1) * z_dir.unsqueeze(1)
                local_semantic_visual_tokens = torch.stack(pooled, dim=1)         # [B, 5, D]

            # v79d (#1.3-lite): override local_semantic_visual_tokens with
            # per-codebook learnable attention pool over visual_tokens.
            if self.use_per_cb_attn_pool and self.per_cb_attn_queries is not None:
                # visual_tokens: [B, N, D]; queries: [M_local=5, D]
                attn_logits = torch.einsum("bnd,md->bnm", visual_tokens, self.per_cb_attn_queries)
                attn = F.softmax(attn_logits / max(self.per_cb_attn_pool_temp, 1e-4), dim=1)  # softmax over N
                local_semantic_visual_tokens = torch.einsum(
                    "bnm,bnd->bmd", attn, visual_tokens
                )                                                                      # [B, 5, D]
                # also overwrite local_routing_matrix so downstream stitching
                # (routing_matrix concat) sees the new per-cb attention weights
                local_routing_matrix = attn                                            # [B, N, 5]

            # 5) stitch [global | local] into the [B, N, 6] / [B, 6, D] interfaces
            routing_matrix = torch.cat(
                [global_weights, local_routing_matrix], dim=-1,
            )                                                                          # [B, N, 6]
            semantic_visual_tokens = torch.cat(
                [global_visual_token.unsqueeze(1), local_semantic_visual_tokens], dim=1,
            )                                                                          # [B, 6, D]

        # v141: ClusterAttentionRouter override.
        # The cluster_attn router has already produced M=6 disentangled
        # tokens via masked cross-attention; replace the weighted-sum
        # output of the global+local branch above with those tokens.
        # Apply visual_adapter HERE (--visual_adapter_after_router) so its
        # gradient stays connected through the cross-attention path back
        # to the encoder, instead of being cut by raw-patch projection.
        if self.router_type == "cluster_attn":
            semantic_visual_tokens = r_out["semantic_tokens"]                          # [B, 6, D_raw]
            if self.visual_adapter_after_router:
                semantic_visual_tokens = self.visual_adapter(semantic_visual_tokens)   # [B, 6, D_model]
            local_semantic_visual_tokens = semantic_visual_tokens[:, 1:, :]            # [B, 5, D]
            # routing_matrix already set above; keep for loss_wasserstein /
            # loss_bu compatibility.

        # v146: TextCrossAttentionRouter override.
        # The Sinkhorn router has already run and produced semantic_visual_tokens
        # as the baseline. Now we run the cross-attention router with text
        # as the query and blend with alpha annealing.
        if self.router_type == "cross_attn":
            # Determine queries:
            # - Training: text_part_tokens (text-driven query)
            # - Inference: codebook anchors (text-aligned by training)
            if (
                self.training
                and text_part_tokens is not None
                and text_part_tokens.shape == (B, NUM_SEMANTIC_PARTS, D)
            ):
                cross_queries = text_part_tokens
            else:
                # Inference (or text unavailable): use codebook anchors as
                # query. They are tracked to match text mean via the
                # text_code_kl / text_hash_ntxent losses.
                anchors = self.quantizer.get_codebook_mean_anchors(
                    exclude_global=False,
                )                                                          # [M=6, D]
                cross_queries = anchors.unsqueeze(0).expand(B, -1, -1).contiguous()

            # Alpha blend: 0 at epoch 0 -> alpha_final after warmup_epochs.
            _epoch = int(getattr(self, "_current_epoch", 0))
            _warmup = max(int(self.cross_attn_warmup_epochs), 1)
            _alpha_final = float(self.cross_attn_alpha_final)
            alpha_blend = _alpha_final * min(1.0, float(_epoch) / float(_warmup))

            ca_out = self.cross_attn_router(
                queries=cross_queries,
                patches=visual_tokens,
                sinkhorn_baseline=routing_matrix,
                sinkhorn_tokens=semantic_visual_tokens,
                alpha_blend=alpha_blend,
                epoch=_epoch,
                warmup_epochs=_warmup,
                return_attn=False,
            )
            # Replace semantic_visual_tokens with the blended output.
            semantic_visual_tokens = ca_out["semantic_visual_tokens"]
            routing_matrix = ca_out["routing_matrix"]
            local_semantic_visual_tokens = semantic_visual_tokens[:, 1:, :]

        if self.soft_visual_grounded_text_pool and use_text_routing:
            if cached_text_tokens is None:
                raise ValueError(
                    "--soft_visual_grounded_text_pool requires cached text tokens"
                )
            if text_part_tokens is None:
                raise ValueError(
                    "--soft_visual_grounded_text_pool requires preliminary text_part_tokens"
                )
            refined_text, attention_entropy = self._soft_visual_grounded_text_pooling(
                visual_tokens=visual_tokens,
                routing_matrix=routing_matrix,
                base_text_tokens=text_part_tokens,
                cached_text_tokens=cached_text_tokens,
                cached_text_token_mask=cached_text_token_mask,
            )
            text_part_tokens = refined_text                              # [B, 6, D]
            global_text_token = text_part_tokens[:, 0, :]
            local_text_tokens = text_part_tokens[:, 1:, :]
            out["text_part_tokens"] = text_part_tokens
            out["global_text_token"] = global_text_token
            out["local_text_tokens"] = local_text_tokens
            out["soft_grounded_text_attention_entropy"] = attention_entropy

        if self.cosine_visual_grounded_text_pool and use_text_routing:
            if cached_text_tokens is None:
                raise ValueError(
                    "--cosine_visual_grounded_text_pool requires cached text tokens"
                )
            if text_part_tokens is None:
                raise ValueError(
                    "--cosine_visual_grounded_text_pool requires preliminary "
                    "text_part_tokens"
                )
            refined_text, attention_entropy, effective_support = (
                self._cosine_visual_grounded_text_pooling(
                    visual_tokens=visual_tokens,
                    routing_matrix=routing_matrix,
                    base_text_tokens=text_part_tokens,
                    cached_text_tokens=cached_text_tokens,
                    cached_text_token_mask=cached_text_token_mask,
                )
            )
            text_part_tokens = refined_text                          # [B, 6, D]
            global_text_token = text_part_tokens[:, 0, :]
            local_text_tokens = text_part_tokens[:, 1:, :]
            out["text_part_tokens"] = text_part_tokens
            out["global_text_token"] = global_text_token
            out["local_text_tokens"] = local_text_tokens
            out["cosine_grounded_text_attention_entropy"] = attention_entropy
            out["cosine_grounded_text_effective_support"] = effective_support
            out["cosine_grounded_text_gate_mean"] = (
                self.cosine_grounded_text_gate.mean().detach()
            )
            out["cosine_grounded_text_gate_abs_mean"] = (
                self.cosine_grounded_text_gate.abs().mean().detach()
            )

        # v162: grounded text routing — visually-routed semantic_visual_tokens
        # query per-codebook local text tokens (Stage-2 top-k_t pool). The
        # refined per-codebook text embed REPLACES the static pooled
        # text_part_tokens that goes into the loss layer.
        if (
            self.grounded_text_routing
            and cached_text_tokens is not None
            and text_part_tokens is not None
        ):
            refined_text = self._grounded_text_routing(
                semantic_visual_tokens=semantic_visual_tokens,
                cached_text_tokens=cached_text_tokens,
                cached_text_token_mask=cached_text_token_mask,
            )
            if self.grounded_text_skip_global:
                # keep C_0 (global slot) unchanged; refine only local 5
                refined_text = torch.cat(
                    [text_part_tokens[:, :1, :], refined_text[:, 1:, :]], dim=1,
                )
            text_part_tokens  = refined_text                                # [B, 6, D]
            global_text_token = text_part_tokens[:, 0, :]
            local_text_tokens = text_part_tokens[:, 1:, :]
            # overwrite the earlier-frozen out-dict entries so the loss
            # layer (which reads from outputs) sees the refined embeds.
            out["text_part_tokens"]   = text_part_tokens
            out["global_text_token"]  = global_text_token
            out["local_text_tokens"]  = local_text_tokens

        # ---- arch-exp-3 (P2) readout: axis-deviation slot / text tokens ---
        # Applied to BOTH sides: a slot token centred across slots has to be
        # compared with a text token centred the same way, or every alignment
        # term is left comparing a deviation with an absolute vector.
        if self.axis_center in ("readout", "both"):
            semantic_visual_tokens = self._axis_center_local(semantic_visual_tokens, 1)
            local_semantic_visual_tokens = semantic_visual_tokens[:, 1:, :]
            if text_part_tokens is not None:
                text_part_tokens = self._axis_center_local(text_part_tokens, 1)
                global_text_token = text_part_tokens[:, 0, :]
                local_text_tokens = text_part_tokens[:, 1:, :]
                for _k, _v in (("text_part_tokens", text_part_tokens),
                               ("global_text_token", global_text_token),
                               ("local_text_tokens", local_text_tokens)):
                    if _k in out:
                        out[_k] = _v

        cibhash_visual_tokens = None
        if self.cibhash_visual_projection_head:
            cibhash_visual_tokens = self._project_cibhash_visual_tokens(
                semantic_visual_tokens,
            )                                                            # [B, 6, D]

        # 6) v32: train-only text injection into the quantizer input.
        # When `--text_inject_train_only=add` and the text path is active,
        # we add `alpha * t_m` to each routed visual token *before*
        # codebook lookup, so the codeword embeddings absorb text-semantic
        # structure during training. At eval/inference we skip this so the
        # forward pass stays text-free (no Qwen captions needed). When
        # `text_inject_detach=True` (variant b) the text gradient is cut
        # so only the codebook moves, not the text adapter.
        quant_input = semantic_visual_tokens
        text_quantizer_tokens = text_part_tokens
        if self.local_residual_quant and self.local_residual_gamma > 0.0:
            quant_input = self._remove_global_projection(
                semantic_visual_tokens,
                gamma=self.local_residual_gamma,
                detach_global=self.local_residual_detach_global,
                name="semantic_visual_tokens",
            )
            if (
                self.local_residual_text
                and text_part_tokens is not None
                and text_part_tokens.shape == semantic_visual_tokens.shape
            ):
                text_quantizer_tokens = self._remove_global_projection(
                    text_part_tokens,
                    gamma=self.local_residual_gamma,
                    detach_global=self.local_residual_detach_global,
                    name="text_part_tokens",
                )
        if self.routed_cls_add_gamma > 0.0:
            assert global_visual_token.shape == (B, D), (
                f"global_visual_token shape {tuple(global_visual_token.shape)} "
                f"!= expected ({B}, {D})"
            )
            # v119b/c: weakly condition routed token(s) with the adapted
            # visual CLS/global token before VQ. Shape stays [B, 6, D].
            cls_delta = self.routed_cls_add_gamma * global_visual_token.detach().unsqueeze(1)  # [B, 1, D]
            if self.routed_cls_add_scope == "local":
                quant_input = quant_input.clone()
                quant_input[:, 1:, :] = quant_input[:, 1:, :] + cls_delta
            else:
                quant_input = quant_input + cls_delta
            assert quant_input.shape == semantic_visual_tokens.shape, (
                f"quant_input shape {tuple(quant_input.shape)} "
                f"!= semantic_visual_tokens {tuple(semantic_visual_tokens.shape)}"
            )
        if (
            self.training
            and self.text_inject_train_only == "add"
            and self.text_inject_alpha > 0.0
            and text_part_tokens is not None
            and text_part_tokens.shape == semantic_visual_tokens.shape
        ):
            t_used = text_part_tokens.detach() if self.text_inject_detach else text_part_tokens
            quant_input = quant_input + self.text_inject_alpha * t_used
        assert quant_input.shape == (B, NUM_SEMANTIC_PARTS, D), (
            f"quant_input shape {tuple(quant_input.shape)} "
            f"!= expected ({B}, {NUM_SEMANTIC_PARTS}, {D})"
        )

        if self.pq_slot_subspace:
            # Zero every dimension outside slot m's own block. VQ distances then
            # depend only on that block, so slots cannot encode each other's
            # information. Shapes are preserved (codebook learns 0 elsewhere).
            _B, _M, _D = quant_input.shape
            if not hasattr(self, "_pq_mask") or self._pq_mask.shape != (1, _M, _D) \
               or self._pq_mask.device != quant_input.device:
                _blk = _D // _M
                _mk = torch.zeros(1, _M, _D, device=quant_input.device,
                                  dtype=quant_input.dtype)
                for _m in range(_M):
                    _hi = (_m + 1) * _blk if _m < _M - 1 else _D
                    _mk[0, _m, _m * _blk:_hi] = 1.0
                self._pq_mask = _mk
            quant_input = quant_input * self._pq_mask.to(quant_input.dtype)
        q_out = self.quantizer(quant_input)
        quantized_tokens     = q_out["quantized_tokens"]      # [B, 6, D]   STE
        quantized_tokens_raw = q_out["quantized_tokens_raw"]  # [B, 6, D]   pure codewords
        codebook_indices     = q_out["codebook_indices"]      # [B, 6]
        codebook_distances   = q_out["codebook_distances"]    # [B, 6, K]

        # 7) gated global addition: blend C_global codeword into local codewords
        #    q_conditioned_m = q_local_m + sigmoid(alpha_m) * (sg)q_global
        # `--disable_global_gate` skips the addition entirely (v23b ablation).
        # v138: when codon_input_source=="routed", use raw routed tokens
        # (quant_input) instead of quantized_tokens. Quantizer still runs to
        # provide codebook_distances for the prototype-cluster InfoNCE loss
        # but does NOT bottleneck codon output.
        head_src = quant_input if self.codon_input_source == "routed" else quantized_tokens
        q_global_cw = head_src[:, 0, :]                        # [B, D]
        q_local_cw  = head_src[:, 1:, :]                       # [B, 5, D]

        if self.disable_global_gate:
            # No C_global -> C_1..5 conditioning. Each codon head sees its own
            # codeword only. `gate_values` is reported as all-zero for logging.
            q_conditioned_local = q_local_cw
            gate_values = torch.zeros(NUM_LOCAL_PARTS, device=q_local_cw.device,
                                       dtype=q_local_cw.dtype)
        else:
            q_global_for_local = q_global_cw.detach() if self.use_stop_grad_global else q_global_cw
            gate_values = torch.sigmoid(self.global_gate_logits)   # [5]
            # broadcast: gate [5] -> [1, 5, 1], q_global_for_local [B, D] -> [B, 1, D]
            global_addition  = gate_values.view(1, -1, 1) * q_global_for_local.unsqueeze(1)  # [B, 5, D]
            q_conditioned_local = q_local_cw + global_addition                                # [B, 5, D]
        head_inputs = torch.cat(
            [q_global_cw.unsqueeze(1), q_conditioned_local], dim=1,
        )                                                       # [B, 6, D]

        # v62 (Option A): residual for codon-head conditioning.
        # residual = quant_input - quantized_tokens_raw (per codebook)
        # When `codon_residual_gamma > 0`, each codon head sees
        # (codeword, gamma * residual) so that two images with same codeword
        # index but different residual content get different codon outputs.
        # Compositional structure preserved (codebook index unchanged); only
        # codon-level fine variation is enabled.
        if self.codon_residual_gamma > 0.0:
            codon_residual = (quant_input - quantized_tokens_raw)  # [B, 6, D]
        else:
            codon_residual = None

        # 8) per-codebook codon heads -> per-codebook outputs
        codon_logits_list:  list[torch.Tensor] = []
        continuous_list:    list[torch.Tensor] = []
        hash_list:          list[torch.Tensor] = []
        hard_list:          list[torch.Tensor] = []
        st_list:            list[torch.Tensor] = []
        indices_list:       list[torch.Tensor] = []
        # v66 aux: accumulate per-codebook text-anchored CE loss
        loss_text_anchor_sum: torch.Tensor = torch.zeros(
            (), device=head_inputs.device, dtype=head_inputs.dtype,
        )
        for m, head in enumerate(self.codon_heads):
            r_m = codon_residual[:, m, :] if codon_residual is not None else None
            # v66: per-codebook text caption embedding (post-text_adapter)
            t_m = (
                text_part_tokens[:, m, :]
                if (text_part_tokens is not None and self.codon_text_anchor)
                else None
            )
            h_out = head(head_inputs[:, m, :], residual=r_m,
                         gamma=self.codon_residual_gamma,
                         text_chunks=t_m)                            # dict
            codon_logits_list.append(h_out["logits"])                # [B, 3, 4]
            continuous_list.append (h_out["continuous_code"])        # [B, 3, 4]
            hash_list.append       (h_out["dna_hash_code"])          # [B, 3, 4]
            hard_list.append       (h_out["dna_hash_code_hard"])     # [B, 3, 4]
            st_list.append         (h_out["dna_hash_code_st"])       # [B, 3, 4]
            indices_list.append    (h_out["base_indices"])           # [B, 3]
            loss_text_anchor_sum = loss_text_anchor_sum + h_out["loss_text_anchor"]
        codon_logits_per_codebook        = torch.stack(codon_logits_list, dim=1)  # [B, 6, 3, 4]
        continuous_codes_per_codebook    = torch.stack(continuous_list,   dim=1)  # [B, 6, 3, 4]
        dna_hash_codes_per_codebook      = torch.stack(hash_list,         dim=1)  # [B, 6, 3, 4]
        dna_hash_codes_per_codebook_hard = torch.stack(hard_list,         dim=1)  # [B, 6, 3, 4]
        dna_hash_codes_per_codebook_st   = torch.stack(st_list,           dim=1)  # [B, 6, 3, 4]
        # v106: codeword-level codon decoder outputs for the bijection loss.
        # Apply each codon head's decoder to the codebook codewords directly
        # (no residual, no sample dependence). Returns [M, K, 3, 4] logits.
        # Only compute when at least one bijection-loss lambda > 0 (cheap O(M*K)
        # per forward); always compute during training to keep autograd graph
        # consistent across batches if downstream Workspaces care.
        if self.training and getattr(self, "_compute_codeword_codon_logits", False):
            cb_logits_list = []
            cb_full = self.quantizer.get_effective_codebooks()        # [M, K_max, D]
            cb_mask = self.quantizer.get_effective_active_mask()      # [M, K_max] bool
            for m, head in enumerate(self.codon_heads):
                # Only consider currently-active codewords (handles K_max > K_active)
                idx_active = cb_mask[m].nonzero(as_tuple=True)[0]
                cb_m = cb_full[m, idx_active]                         # [K_active, D]
                cb_logits_list.append(head.decode_codeword(cb_m))     # [K_active, L, 4]
            # Pad to K_max so all codebooks share a tensor (K_active may differ
            # per codebook with adaptive K). We'll record both the logits and
            # a per-codebook K count so the loss can mask correctly.
            K_max = max(t.shape[0] for t in cb_logits_list)
            cb_logits_padded = torch.zeros(
                (len(cb_logits_list), K_max,
                 self.num_codons_per_codebook, 4),
                device=cb_full.device, dtype=cb_logits_list[0].dtype,
            )
            cb_K_active = torch.zeros(
                len(cb_logits_list), device=cb_full.device, dtype=torch.long,
            )
            for m, t in enumerate(cb_logits_list):
                cb_logits_padded[m, :t.shape[0]] = t
                cb_K_active[m] = t.shape[0]
            codeword_codon_logits = cb_logits_padded                  # [M, K_max, 3, 4]
            codeword_K_active     = cb_K_active                       # [M] long
        else:
            codeword_codon_logits = None
            codeword_K_active     = None
        base_indices_per_codebook        = torch.stack(indices_list,      dim=1)  # [B, 6, 3]
        # 9) flatten to a single concatenated code: 6 codebooks * 3 codon positions = 18
        L = self.num_codons_per_codebook
        Mp3 = NUM_SEMANTIC_PARTS * L                                         # M*L (was always 6*3=18; now 6*L)
        continuous_code    = continuous_codes_per_codebook   .reshape(B, Mp3, 4)   # [B, M*L, 4]
        dna_hash_code      = dna_hash_codes_per_codebook     .reshape(B, Mp3, 4)   # [B, M*L, 4]
        dna_hash_code_hard = dna_hash_codes_per_codebook_hard.reshape(B, Mp3, 4)   # [B, M*L, 4]
        dna_hash_code_st   = dna_hash_codes_per_codebook_st  .reshape(B, Mp3, 4)   # [B, M*L, 4]
        base_indices       = base_indices_per_codebook       .reshape(B, Mp3)      # [B, M*L]

        # 9b) v91 — TEXT-DNA path (Option F: text-to-image hash matching)
        # When lambda_text_hash > 0, push text_part_tokens [B, 6, D] through
        # the *same* quantizer + codon_heads to produce a text-derived
        # continuous_code [B, 18, 4]. Loss will MSE-match this to the
        # image-derived continuous_code. We:
        #   (a) call quantizer with EMA temporarily disabled so the text
        #       pass doesn't move the codebook (image is the canonical
        #       update signal),
        #   (b) reuse the same codon-residual mechanism (text-residual is
        #       text_part_tokens - text_quantized_raw),
        #   (c) intentionally SKIP the global gate mixing — text already
        #       has its own per-slot semantic, no need to inject cb0.
        # `text_part_tokens` is only set inside the training-time text
        # routing branch. If it's None and we still have cached
        # text_part_raw, run text_adapter on it directly so the text-DNA
        # path also works at inference.
        if (
            (float(self.lambda_text_hash) > 0.0
             or float(self.lambda_text_hash_ntxent) > 0.0    # v109: include InfoNCE form
             or float(self.lambda_codeword_text_proto) > 0.0  # v123: text prototype teacher
             or float(self.lambda_text_codon_rel) > 0.0)     # v113: text teacher, no text-DNA path
            and text_part_tokens is None
            and cached_text_part_raw is not None
        ):
            _raw_for_text_dna = cached_text_part_raw
            if (
                self.codebook_text_prompts is not None
                and _raw_for_text_dna.shape[-1] == self.codebook_text_prompts.shape[-1]
            ):
                _raw_for_text_dna = _raw_for_text_dna + self.codebook_text_prompts.unsqueeze(0)
            if isinstance(self.text_adapter, nn.ModuleList):
                _per_slot = [self.text_adapter[m](_raw_for_text_dna[:, m, :])
                             for m in range(_raw_for_text_dna.shape[1])]
                text_part_tokens = torch.stack(_per_slot, dim=1)
            else:
                text_part_tokens = self.text_adapter(_raw_for_text_dna)
            text_quantizer_tokens = text_part_tokens
            if self.local_residual_quant and self.local_residual_text and self.local_residual_gamma > 0.0:
                assert text_part_tokens.shape == (B, NUM_SEMANTIC_PARTS, D), (
                    f"text_part_tokens shape {tuple(text_part_tokens.shape)} "
                    f"!= expected ({B}, {NUM_SEMANTIC_PARTS}, {D})"
                )
                text_quantizer_tokens = self._remove_global_projection(
                    text_part_tokens,
                    gamma=self.local_residual_gamma,
                    detach_global=self.local_residual_detach_global,
                    name="text_part_tokens",
                )
        text_continuous_code = None
        text_quantized_tokens = None
        text_cb_indices = None
        # v109: include lambda_text_hash_ntxent in the activation check so that
        # the text codon path is computed even when only the InfoNCE-form
        # visual-textual contrastive loss is enabled (lambda_text_hash=0 +
        # lambda_text_hash_ntxent>0). Previously gated only by MSE-form +
        # cw_xmodal.
        _lam_text_hash_ntxent_local = float(getattr(self, "lambda_text_hash_ntxent", 0.0))
        _lam_xmodal_commit_local = float(getattr(self, "lambda_xmodal_commit", 0.0))
        _text_path_active = (
            (float(self.lambda_text_hash)        > 0.0
             or _lam_text_hash_ntxent_local      > 0.0
             or float(self.lambda_cw_xmodal)     > 0.0
             or _lam_xmodal_commit_local         > 0.0)
            and text_part_tokens is not None
            and text_part_tokens.shape == (B, NUM_SEMANTIC_PARTS, D)
        )
        if _text_path_active:
            factual_text_dna = self._encode_text_tokens_to_dna(
                text_part_tokens,
                allow_mm_ema=True,
            )
            text_continuous_code = factual_text_dna["continuous_code"]
            text_quantized_tokens = factual_text_dna["quantized_tokens"]
            text_cb_indices = factual_text_dna["codebook_indices"]
            text_quantizer_tokens = factual_text_dna["quantizer_input"]

        # B variant: encode one independently edited minimal-pair caption for
        # every valid local slot through the same text adapter, quantizer, and
        # codon heads. Foils never participate in routing or EMA updates and
        # are consumed only as detached own-sample negatives by the loss.
        text_foil_continuous_code = None
        text_foil_codebook_indices = None
        text_foil_valid_mask = None
        if (
            self.training
            and
            self.text_hash_counterfactual_weight > 0.0
            and compute_text_foil
        ):
            if cached_text_foil_raw is None or cached_text_foil_valid is None:
                raise RuntimeError(
                    "counterfactual text-DNA training is enabled, but the "
                    "batch has no text foil sidecars. Build/load "
                    "text_foil_part.f16.npy and text_foil_valid.bool.npy."
                )
            if cached_text_foil_raw.dim() != 3 or cached_text_foil_raw.shape != (
                B, NUM_SEMANTIC_PARTS, self.proj_dim,
            ):
                raise ValueError(
                    "cached_text_foil_raw must be [B, 6, D_proj], got "
                    f"{tuple(cached_text_foil_raw.shape)}"
                )
            if cached_text_foil_valid.dtype != torch.bool:
                raise TypeError(
                    "cached_text_foil_valid must have dtype torch.bool, got "
                    f"{cached_text_foil_valid.dtype}"
                )
            if cached_text_foil_valid.shape != (B, NUM_SEMANTIC_PARTS):
                raise ValueError(
                    "cached_text_foil_valid must be [B, 6], got "
                    f"{tuple(cached_text_foil_valid.shape)}"
                )
            if bool(cached_text_foil_valid[:, 0].any().item()):
                raise ValueError("counterfactual C_global (slot 0) must be invalid")
            if (
                bool(cached_text_foil_valid.any().item())
                and not bool(torch.isfinite(
                    cached_text_foil_raw[cached_text_foil_valid],
                ).all().item())
            ):
                raise ValueError("valid counterfactual text features must be finite")

            # No gradient may optimize the foil encoder toward an easy
            # language-only shortcut. Its value is nevertheless refreshed
            # every step from the current factual text adapter/codon heads.
            with torch.no_grad():
                foil_raw_for_loss = cached_text_foil_raw
                if self.bidirectional_token_prune:
                    if (
                        cached_text_foil_tokens is None
                        or cached_text_foil_token_mask is None
                    ):
                        raise RuntimeError(
                            "v185 counterfactual training requires token-level "
                            "foil sidecars so factual and foil captions use "
                            "the same prune/mean pooling path"
                        )
                    pooled_foil_local = self._pool_local_foil_tokens_like_factual(
                        cached_text_foil_tokens,
                        cached_text_foil_token_mask,
                        feats["visual_tokens_raw"],
                        visual_attention_mask,
                    )
                    foil_raw_for_loss = torch.cat(
                        [
                            torch.zeros_like(cached_text_foil_raw[:, :1, :]),
                            pooled_foil_local.to(cached_text_foil_raw.dtype),
                        ],
                        dim=1,
                    )
                foil_text_tokens = self._adapt_pooled_text_for_loss(
                    foil_raw_for_loss,
                )
                foil_text_dna = self._encode_text_tokens_to_dna(
                    foil_text_tokens,
                    allow_mm_ema=False,
                    deterministic_codon=True,
                )
            text_foil_continuous_code = foil_text_dna["continuous_code"]
            text_foil_codebook_indices = foil_text_dna["codebook_indices"]
            text_foil_valid_mask = cached_text_foil_valid

        # 10) shape sanity --------------------------------------------------
        assert local_routing_matrix.shape == (B, N, NUM_LOCAL_PARTS), (
            f"local_routing_matrix shape {tuple(local_routing_matrix.shape)} "
            f"!= expected ({B}, {N}, {NUM_LOCAL_PARTS})"
        )
        assert routing_matrix.shape == (B, N, NUM_SEMANTIC_PARTS), (
            f"routing_matrix shape {tuple(routing_matrix.shape)} "
            f"!= expected ({B}, {N}, {NUM_SEMANTIC_PARTS})"
        )
        assert semantic_visual_tokens.shape == (B, NUM_SEMANTIC_PARTS, D), (
            f"semantic_visual_tokens shape {tuple(semantic_visual_tokens.shape)} "
            f"!= expected ({B}, {NUM_SEMANTIC_PARTS}, {D})"
        )
        assert quantized_tokens.shape == (B, NUM_SEMANTIC_PARTS, D)
        assert codebook_indices.shape  == (B, NUM_SEMANTIC_PARTS)
        assert codebook_distances.shape == (B, NUM_SEMANTIC_PARTS, self.quantizer.K_max)
        assert head_inputs.shape == (B, NUM_SEMANTIC_PARTS, D)
        assert gate_values.shape == (NUM_LOCAL_PARTS,)
        assert continuous_codes_per_codebook.shape == (B, NUM_SEMANTIC_PARTS, L, 4)
        assert dna_hash_codes_per_codebook  .shape == (B, NUM_SEMANTIC_PARTS, L, 4)
        assert base_indices_per_codebook    .shape == (B, NUM_SEMANTIC_PARTS, L)
        assert continuous_code.shape == (B, Mp3, 4)
        assert dna_hash_code  .shape == (B, Mp3, 4)
        assert base_indices   .shape == (B, Mp3)

        out.update({
            "local_routing_matrix":         local_routing_matrix,
            "routing_matrix":               routing_matrix,
            "routing_matrix_premask":       routing_matrix_premask,
            "routing_mean_effective_k":      routing_mean_effective_k.detach(),
            "routing_fraction_top1":         routing_fraction_top1.detach(),
            "routing_visual_specificity_mean": r_out.get(
                "visual_specificity_mean", None,
            ),
            "routing_visual_marginal_effective_ratio": r_out.get(
                "visual_marginal_effective_ratio", None,
            ),
            "routing_visual_consensus_candidate_ratio": (
                _cls_verified_diag["common_candidate_ratio"]
                if _cls_verified_diag is not None else None
            ),
            "routing_visual_cls_low_ratio": (
                _cls_verified_diag["global_low_ratio"]
                if _cls_verified_diag is not None else None
            ),
            "routing_visual_consensus_mask_ratio": r_out.get(
                "visual_consensus_mask_ratio", None,
            ) if _cls_verified_diag is None else _cls_verified_diag["mask_ratio"],
            "routing_visual_consensus_remaining_ratio": r_out.get(
                "visual_consensus_remaining_ratio", None,
            ) if _cls_verified_diag is None else _cls_verified_diag["remaining_ratio"],
            "routing_visual_consensus_fallback": r_out.get(
                "visual_consensus_fallback", None,
            ) if _cls_verified_diag is None else _cls_verified_diag["fallback"],
            "local_semantic_visual_tokens": local_semantic_visual_tokens,
            "semantic_visual_tokens":       semantic_visual_tokens,
            "cibhash_visual_tokens":         cibhash_visual_tokens,
            # entropic-OT cost (W_e) per sample, [B]. Used by the Wasserstein
            # alignment loss to push visual_adapter and text_adapter into a
            # shared space where the patch <-> text-part coupling is "cheap".
            "ot_cost":                      r_out.get("ot_cost", None),
            "quantizer_input":              quant_input,                       # [B, 6, D]
            "quantized_tokens":             quantized_tokens,
            "quantized_tokens_raw":         quantized_tokens_raw,
            "codebook_indices":             codebook_indices,
            "codebook_distances":           codebook_distances,

            # gated global addition
            "head_inputs":                  head_inputs,                       # [B, 6, D]
            "global_gate_values":           gate_values,                       # [5]
            "use_stop_grad_global":         self.use_stop_grad_global,         # bool

            # per-codebook codon outputs
            "codon_logits_per_codebook":         codon_logits_per_codebook,          # [B, 6, 3, 4]
            "continuous_codes_per_codebook":     continuous_codes_per_codebook,      # [B, 6, 3, 4]
            "dna_hash_codes_per_codebook":       dna_hash_codes_per_codebook,        # [B, 6, 3, 4]
            "dna_hash_codes_per_codebook_hard":  dna_hash_codes_per_codebook_hard,   # [B, 6, 3, 4]
            "dna_hash_codes_per_codebook_st":    dna_hash_codes_per_codebook_st,     # [B, 6, 3, 4]
            "base_indices_per_codebook":         base_indices_per_codebook,          # [B, 6, 3]

            # final concatenated code
            "continuous_code":                   continuous_code,                    # [B, 18, 4]
            "dna_hash_code":                     dna_hash_code,                      # [B, 18, 4]
            "dna_hash_code_hard":                dna_hash_code_hard,                 # [B, 18, 4]
            "dna_hash_code_st":                  dna_hash_code_st,                   # [B, 18, 4]
            "base_indices":                      base_indices,                       # [B, 18]

            # v91 text-to-image DNA-hash matching path. None when
            # lambda_text_hash == 0 AND lambda_cw_xmodal == 0; otherwise
            # [B, 18, 4] continuous code derived from text_part_tokens by
            # reusing the shared quantizer (EMA-disabled) and codon_heads.
            "text_continuous_code":              text_continuous_code,               # [B, 18, 4] or None
            "text_codebook_indices":              text_cb_indices,                    # [B, 6] or None
            "text_foil_continuous_code":          text_foil_continuous_code,          # [B, 18, 4] or None
            "text_foil_codebook_indices":         text_foil_codebook_indices,         # [B, 6] or None
            "text_foil_valid_mask":               text_foil_valid_mask,               # [B, 6] bool or None
            # v93 text codeword path. None when text path inactive; otherwise
            # the per-codebook quantized text codeword (codebook entry the
            # text_part_tokens were nearest-neighbour to). Used by
            # lambda_cw_xmodal cross-modal InfoNCE in loss_siglip2.
            "text_quantized_tokens":             text_quantized_tokens,              # [B, 6, D] or None
            # v123 text prototypes use the same residualized text space that
            # the text-only quantizer path sees when --local_residual_text.
            "codeword_text_tokens":              text_quantizer_tokens,              # [B, 6, D] or None

            # v66 text-anchored prototype head: per-batch CE between visual
            # codon logits and text-derived target classes, summed across
            # 6 codebooks. Zero when codon_text_anchor flag is off.
            "loss_text_anchor":                  loss_text_anchor_sum,
            # Codebook tensor + active mask (for v79a #1.2 ortho loss).
            # The buffer itself has no gradient (EMA mode); the loss
            # consumer treats it as a measurement signal that *also*
            # affects routing indirectly via z (the anchor loss family
            # routes via local_codebook_mean_anchors -> z via Sinkhorn
            # cost). For gradient codebook mode this term is fully active.
            "codebooks_buffer":                  self.quantizer.get_effective_codebooks(),
            "codebook_active_mask":              self.quantizer.get_effective_active_mask(),
            # v144: full codebook tensor exposed for text_code_kl loss.
            # In EMA mode this is a buffer (no autograd); in gradient mode
            # it is a Parameter. Either way the loss can compute logits =
            # z @ C.T without per-codeword indexing.
            "codebooks":                         self.quantizer.get_effective_codebooks(),
            # v106: codeword-level codon decoder outputs for bijection loss.
            # Shape [M, K_max, 3, 4] logits; None when bijection loss inactive.
            # Companion `codeword_K_active` [M] gives per-codebook active K
            # (handles adaptive K where K_active may vary per codebook).
            "codeword_codon_logits":             codeword_codon_logits,
            "codeword_K_active":                 codeword_K_active,
        })

        # 11) optional reconstruction head (v28a / v28b)
        # Concatenate the 6 STE-quantized codewords and decode either to a
        # 224x224 RGB image (v28a) or to the cached SigLIP2 visual_global
        # (v28b). `quantized_tokens` carries the STE so gradient flows back
        # to the codebooks AND the encoder.
        if self.decoder is not None:
            recon = self.decoder(quantized_tokens)               # [B, 3, 224, 224] or [B, D_proj]
            out["reconstruction"]        = recon
            out["reconstruction_target"] = self.decoder_target   # 'pixel' or 'siglip_feat'
        else:
            out["reconstruction"]        = None
            out["reconstruction_target"] = None
        # 12) v70a (Exp 3): hash reconstruction decoder (uses ST hash so grad
        #     flows back to the codon heads / quantizer)
        if self.hash_recon_decoder is not None:
            hash_flat = dna_hash_code_st.reshape(B, -1)          # [B, 72]
            out["hash_recon_pred"] = self.hash_recon_decoder(hash_flat)  # [B, D_proj]
        else:
            out["hash_recon_pred"] = None
        # 13) v72a (Exp 6): dual projection heads from flattened hash
        if self.dual_hash_proj is not None:
            hash_flat = dna_hash_code_st.reshape(B, -1)
            dh = self.dual_hash_proj(hash_flat)
            out["dual_hash_semantic"] = dh["semantic"]            # [B, D_proj]
            out["dual_hash_instance"] = dh["instance"]            # [B, D_proj]
        else:
            out["dual_hash_semantic"] = None
            out["dual_hash_instance"] = None
        # 14) (b-1, 2026-09-20) masked entity completion on the 4 local slots'
        #     straight-through codes; training forward of view 1 only.
        if (self.mec is not None and self.training and mec_image_ids is not None
                and quantized_tokens is not None):
            out["loss_mec"], out["mec_acc"] = self.mec(
                quantized_tokens[:, 1:1 + int(NUM_LOCAL_PARTS), :], mec_image_ids)
        return out
