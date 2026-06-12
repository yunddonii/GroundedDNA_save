"""v146: Text-as-Query Cross-Attention Router with stability safeguards.

Replaces the Sinkhorn-OT router with a multi-head cross-attention where:
  - Q = text_part_tokens [B, M=6, D]   (at training)
  - Q = codebook_anchors [6, D]         (at inference; text-aligned by training)
  - K, V = visual_patches [B, N, D]

Stability safeguards (learned from v141 failure):
  1. Near-identity init on W_Q/K/V (avoids random-init shock to encoder)
  2. Zero-init on W_O (initial cross-attn contribution = 0)
  3. Residual blend with Sinkhorn baseline via alpha-annealing
     alpha = 0 at epoch 0 -> Sinkhorn dominant
     alpha = 1 after warmup -> cross-attn dominant
  4. Pre-norm transformer block (LayerNorm before Q/K/V projection)
  5. Multi-head attention with sqrt(d_head) scaling
  6. Temperature annealing (init 0.2 -> final 0.07)
  7. Returns attention weights for visualization + grounding loss

Train/Inference symmetry resolution:
  - Train: text_part_tokens [B, M, D] is the Q
  - Inference: codebook_anchors [M, D] -> [B, M, D] (broadcast)
  - Both have the same shape and lie in the same D-dim space
  - Codebook learns to match text mean during training via
    text_code_kl + text_hash_ntxent losses

Author: GroundedDNA (2026-06-12, v146a)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_near_identity(linear: nn.Linear, scale: float = 0.1) -> None:
    """Initialize Linear to near-identity (eye + small noise).

    For W_Q/W_K/W_V at the start, the cross-attention acts approximately
    as an identity transform on the query/key/value tokens. This avoids
    a sudden shock to the encoder when cross-attn is freshly inserted.

    Requires square weight (in_features == out_features).
    """
    d_in, d_out = linear.in_features, linear.out_features
    if d_in != d_out:
        # Can't apply eye; fall back to small-noise init.
        nn.init.normal_(linear.weight, std=scale)
    else:
        nn.init.eye_(linear.weight)
        with torch.no_grad():
            linear.weight.data += torch.randn_like(linear.weight) * scale
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


def _init_zero_output(linear: nn.Linear) -> None:
    """Zero-init weight and bias so the layer initially outputs zeros.

    Used on W_O so the cross-attention's initial contribution is zero,
    and only the residual / Sinkhorn baseline flows. The layer learns
    its useful weight during training.
    """
    nn.init.zeros_(linear.weight)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


class TextCrossAttentionRouter(nn.Module):
    """Text-conditioned cross-attention router for grounded routing.

    Forward signature differs from SemanticSinkhornRouter -- it takes
    queries explicitly (text at train, codebook anchors at inference)
    rather than implicitly computing them.

    Parameters
    ----------
    d_model : int
        Token dimensionality (must match encoder hidden dim).
    num_parts : int
        Number of semantic parts M (typically 6).
    num_heads : int
        Multi-head attention head count. d_model must be divisible.
    dropout : float
        Attention dropout probability.
    temp_init : float
        Initial softmax temperature (larger -> softer attention).
    temp_final : float
        Final softmax temperature after annealing (CLIP standard 0.07).
    near_identity_scale : float
        Std for the near-identity noise on W_Q/W_K/W_V init.
    """

    def __init__(
        self,
        d_model: int = 768,
        num_parts: int = 6,
        num_heads: int = 4,
        dropout: float = 0.1,
        temp_init: float = 0.2,
        temp_final: float = 0.07,
        near_identity_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )
        self.D = int(d_model)
        self.M = int(num_parts)
        self.num_heads = int(num_heads)
        self.head_dim = self.D // self.num_heads
        self.dropout_p = float(dropout)
        self.temp_init = float(temp_init)
        self.temp_final = float(temp_final)

        # Pre-norm (transformer-block convention).
        self.text_norm = nn.LayerNorm(self.D)
        self.visual_norm = nn.LayerNorm(self.D)

        # Q/K/V projections (near-identity init for stability).
        self.W_Q = nn.Linear(self.D, self.D, bias=True)
        self.W_K = nn.Linear(self.D, self.D, bias=True)
        self.W_V = nn.Linear(self.D, self.D, bias=True)
        _init_near_identity(self.W_Q, scale=float(near_identity_scale))
        _init_near_identity(self.W_K, scale=float(near_identity_scale))
        _init_near_identity(self.W_V, scale=float(near_identity_scale))

        # Output projection (zero-init so initial cross-attn contribution = 0).
        self.W_O = nn.Linear(self.D, self.D, bias=True)
        _init_zero_output(self.W_O)

        # Attention dropout.
        self.attn_dropout = nn.Dropout(self.dropout_p)

    # ------------------------------------------------------------------ #
    # Helper                                                              #
    # ------------------------------------------------------------------ #

    def _current_temperature(self, epoch: int, warmup_epochs: int) -> float:
        """Linear anneal from temp_init -> temp_final across warmup epochs."""
        if warmup_epochs <= 0:
            return self.temp_final
        progress = min(1.0, max(0.0, float(epoch) / float(warmup_epochs)))
        return self.temp_init + progress * (self.temp_final - self.temp_init)

    # ------------------------------------------------------------------ #
    # Core forward                                                        #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        queries: torch.Tensor,       # [B, M, D] (text at train, codebook at infer)
        patches: torch.Tensor,       # [B, N, D]
        sinkhorn_baseline: torch.Tensor = None,   # [B, N, M] routing matrix (optional)
        sinkhorn_tokens: torch.Tensor = None,     # [B, M, D] sinkhorn-aggregated tokens
        alpha_blend: float = 1.0,
        epoch: int = 0,
        warmup_epochs: int = 20,
        return_attn: bool = True,
    ) -> dict:
        """
        Parameters
        ----------
        queries : Tensor [B, M, D]
            Text part tokens at train (text_part_tokens) or codebook anchors
            at inference (broadcast to batch dimension).
        patches : Tensor [B, N, D]
            Visual patch tokens (post visual_adapter).
        sinkhorn_baseline : Tensor [B, N, M], optional
            Routing matrix from a parallel Sinkhorn router, used for the
            residual blend. Required when alpha_blend < 1.0.
        sinkhorn_tokens : Tensor [B, M, D], optional
            Semantic visual tokens from Sinkhorn (weighted sum). Used for
            output blend when alpha_blend < 1.0.
        alpha_blend : float in [0, 1]
            Cross-attn contribution weight. 0 = full Sinkhorn baseline,
            1 = pure cross-attn. Typically annealed during training.
        epoch : int
            Current training epoch (for temperature annealing).
        warmup_epochs : int
            Number of warmup epochs for temperature anneal.
        return_attn : bool
            If True, returns the attention matrix [B, h, M, N] (averaged
            over heads to [B, M, N] in the output for compatibility).

        Returns
        -------
        dict with keys
            "semantic_visual_tokens" : [B, M, D] -- the cross-attention output
                (blended with Sinkhorn baseline if alpha_blend < 1)
            "routing_matrix" : [B, N, M] -- attention transposed to match
                Sinkhorn router output shape (averaged over heads, for
                downstream loss compatibility)
            "ot_cost" : [B] -- placeholder (zero); cross-attn has no OT cost
            "attn_per_head" : [B, h, M, N] -- raw attention (if return_attn)
        """
        B, M, D = queries.shape
        _, N, _ = patches.shape
        assert M == self.M, f"queries M={M} != self.M={self.M}"
        assert D == self.D, f"queries D={D} != self.D={self.D}"

        # Temperature anneal.
        tau = self._current_temperature(epoch, warmup_epochs)
        tau = max(tau, 1e-3)

        # Pre-norm.
        q_in = self.text_norm(queries)            # [B, M, D]
        kv_in = self.visual_norm(patches)         # [B, N, D]

        # Projections.
        Q = self.W_Q(q_in)                         # [B, M, D]
        K = self.W_K(kv_in)                        # [B, N, D]
        V = self.W_V(kv_in)                        # [B, N, D]

        # Multi-head reshape.
        Q = Q.view(B, M, self.num_heads, self.head_dim).transpose(1, 2)  # [B, h, M, d]
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, h, N, d]
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, h, N, d]

        # Attention logits [B, h, M, N]
        attn_logit = torch.einsum("bhmd,bhnd->bhmn", Q, K) / math.sqrt(self.head_dim)
        attn_logit = attn_logit / tau
        attn = F.softmax(attn_logit, dim=-1)
        attn = self.attn_dropout(attn)

        # Output: [B, h, M, d]
        out = torch.einsum("bhmn,bhnd->bhmd", attn, V)
        # Merge heads -> [B, M, D]
        out = out.transpose(1, 2).contiguous().view(B, M, self.D)
        out = self.W_O(out)                        # [B, M, D]

        # Residual blend with Sinkhorn baseline.
        if alpha_blend < 1.0 and sinkhorn_tokens is not None:
            blended_tokens = alpha_blend * out + (1.0 - alpha_blend) * sinkhorn_tokens
        else:
            blended_tokens = out

        # Routing matrix for downstream compatibility (average over heads).
        # Attention is [B, h, M, N]; we want [B, N, M].
        attn_avg = attn.mean(dim=1)                # [B, M, N]
        routing_matrix = attn_avg.transpose(1, 2)  # [B, N, M]

        # Blend routing too (for loss_wasserstein / loss_bu compatibility).
        if alpha_blend < 1.0 and sinkhorn_baseline is not None:
            routing_matrix = (
                alpha_blend * routing_matrix
                + (1.0 - alpha_blend) * sinkhorn_baseline
            )

        out_dict = {
            "semantic_visual_tokens": blended_tokens,        # [B, M, D]
            "routing_matrix": routing_matrix,                # [B, N, M]
            "ot_cost": torch.zeros(B, device=queries.device, dtype=queries.dtype),
        }
        if return_attn:
            out_dict["attn_per_head"] = attn                 # [B, h, M, N]
        return out_dict
