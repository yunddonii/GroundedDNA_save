"""Text-guided Sinkhorn semantic routing (LOCAL parts only).

This router is intended for the 5 LOCAL semantic parts only:
    C_head_or_main_part, C_body_or_secondary_part, C_limb_or_detail_part,
    C_color_texture, C_background_null
The C_global slot is handled OUTSIDE this module by global average pooling of
the full visual-token sequence — it does not compete with the local parts for
token mass and must NOT be passed in here.

Given:
    visual_tokens     [B, N, D]
    text_part_tokens  [B, M_local, D]   (M_local = 5 in this stage)

the router computes a soft balanced routing matrix

    P                 [B, N, M_local]   row/col marginals are balanced

via log-space Sinkhorn-Knopp on the kernel  log K = -cost / epsilon, where
cost = 1 - cosine_similarity by default. It then weighted-pools visual tokens
by P to obtain

    semantic_visual_tokens  [B, M_local, D]

which is the per-local-part representation downstream modules will consume.

We use log-space Sinkhorn so the iteration is numerically stable AND
differentiable end-to-end (gradients flow through both `visual_tokens` and
`text_part_tokens`).

The class is dimension-agnostic (works for any M), so passing the 5 local
text tokens directly is enough — no code change needed beyond the call site.
"""

from __future__ import annotations
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _log_sinkhorn(
    log_K: torch.Tensor,    # [B, N, M]
    log_a: torch.Tensor,    # [B, N]
    log_b: torch.Tensor,    # [B, M]
    num_iters: int = 20,
) -> torch.Tensor:
    """Stable log-space Sinkhorn-Knopp.

    Returns log_P: [B, N, M] such that  exp(log_P)  has row marginal exp(log_a)
    and column marginal exp(log_b) (approximately, after num_iters iterations).
    """
    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)
    for _ in range(num_iters):
        # row update : u <- a / (K v)
        log_u = log_a - torch.logsumexp(log_K + log_v.unsqueeze(1), dim=-1)
        # col update : v <- b / (K^T u)
        log_v = log_b - torch.logsumexp(log_K + log_u.unsqueeze(-1), dim=1)
    return log_u.unsqueeze(-1) + log_K + log_v.unsqueeze(1)


class SemanticSinkhornRouter(nn.Module):
    """Soft balanced router using Sinkhorn OT (text-centroid similarity).

    Args:
        epsilon:     entropic regularization (similarity is divided by this).
                     Smaller -> sharper routing.
        num_iters:   Sinkhorn iterations. 20 is usually enough.
        cost_mode:   'one_minus_cos'  (cost = 1 - cosine_sim; >= 0)
                     'neg_cos'        (cost = -cosine_sim;     can be < 0)
                     Both are equivalent up to a constant in log_K.
    """

    def __init__(
        self,
        epsilon: float = 0.05,
        num_iters: int = 20,
        cost_mode: str = "one_minus_cos",
    ) -> None:
        super().__init__()
        if cost_mode not in ("one_minus_cos", "neg_cos"):
            raise ValueError(
                f"cost_mode must be 'one_minus_cos' or 'neg_cos', got {cost_mode!r}"
            )
        self.epsilon: float = float(epsilon)
        self.num_iters: int = int(num_iters)
        self.cost_mode: str = cost_mode

    def forward(
        self,
        visual_tokens:    torch.Tensor,                 # [B, N, D]
        text_part_tokens: torch.Tensor,                 # [B, M, D]
        visual_mask:      Optional[torch.Tensor] = None,  # [B, N]
        part_mask:        Optional[torch.Tensor] = None,  # [B, M]
    ) -> Dict[str, torch.Tensor]:
        # ---- shape sanity ------------------------------------------------
        B, N, D = visual_tokens.shape
        Bt, M, Dt = text_part_tokens.shape
        if not (B == Bt and D == Dt):
            raise ValueError(
                f"shape mismatch in router: visual_tokens={tuple(visual_tokens.shape)} "
                f"vs text_part_tokens={tuple(text_part_tokens.shape)}"
            )

        # ---- 1) cosine similarity ---------------------------------------
        v_n = F.normalize(visual_tokens,    dim=-1)      # [B, N, D]
        t_n = F.normalize(text_part_tokens, dim=-1)      # [B, M, D]
        sim = torch.matmul(v_n, t_n.transpose(-1, -2))   # [B, N, M]

        # ---- 2) cost & log kernel ---------------------------------------
        if self.cost_mode == "one_minus_cos":
            cost = 1.0 - sim                             # [B, N, M], >= 0
        else:
            cost = -sim                                  # [B, N, M], in [-1, 1]
        log_K = -cost / max(self.epsilon, 1e-6)          # [B, N, M]

        # ---- 3) marginals (uniform unless masks supplied) ---------------
        device = visual_tokens.device
        dtype  = visual_tokens.dtype
        if visual_mask is None:
            a = torch.full((B, N), 1.0 / N, device=device, dtype=dtype)
        else:
            m = visual_mask.to(dtype)
            a = m / m.sum(dim=1, keepdim=True).clamp_min(1e-12)
        if part_mask is None:
            b = torch.full((B, M), 1.0 / M, device=device, dtype=dtype)
        else:
            m = part_mask.to(dtype)
            b = m / m.sum(dim=1, keepdim=True).clamp_min(1e-12)

        log_a = torch.log(a.clamp_min(1e-12))
        log_b = torch.log(b.clamp_min(1e-12))

        # ---- 4) Sinkhorn iterations in log-space ------------------------
        log_P = _log_sinkhorn(log_K, log_a, log_b, num_iters=self.num_iters)
        P = torch.exp(log_P)                             # [B, N, M]

        # ---- 5) weighted pooling: P^T @ V then col-normalize ------------
        # semantic_v[b, m, :] = sum_n P[b, n, m] * v[b, n, :] / sum_n P[b, n, m]
        denom = P.sum(dim=1).clamp_min(1e-12)            # [B, M]
        semantic_v = torch.einsum("bnm,bnd->bmd", P, visual_tokens) / denom.unsqueeze(-1)
        # semantic_v: [B, M, D]

        # ---- 5b) entropic-OT cost W_e = <pi, cost>, per sample ----------
        # Reduced to a per-batch tensor [B] so the caller can mean over the
        # batch (or weight it) before adding to the total loss. Captures the
        # current alignment between visual patches and text parts; gradient
        # flows back through `cost` -> v_n, t_n -> visual/text adapter weights.
        ot_cost = (P * cost).sum(dim=(1, 2))             # [B]

        # ---- 6) shape assertions ----------------------------------------
        assert P.shape == (B, N, M), (
            f"routing_matrix shape unexpected: got {tuple(P.shape)}, expected {(B, N, M)}"
        )
        assert semantic_v.shape == (B, M, D), (
            f"semantic_visual_tokens shape unexpected: "
            f"got {tuple(semantic_v.shape)}, expected {(B, M, D)}"
        )

        return {
            "routing_matrix":         P,            # [B, N, M]
            "semantic_visual_tokens": semantic_v,   # [B, M, D]
            "ot_cost":                ot_cost,      # [B]   per-sample W_e value
        }


class SemanticAttentionRouter(nn.Module):
    """Cross-attention router: text part = query, visual tokens = key/value.

    Drop-in replacement for ``SemanticSinkhornRouter`` (same forward signature
    + same output dict). The two differ in the routing_matrix's marginal
    structure:

        Sinkhorn:   sum over N = 1/M   AND   sum over M = 1/N    (balanced both ways)
        Attention:  sum over N = 1     AND   sum over M is free  (per-part softmax over patches)

    Sinkhorn forces every part to receive ~equal patch mass even when an image
    has no content for that part (e.g. C_limb on a landscape). Attention lets
    parts pull whichever patches they like, with no marginal-balance penalty
    on either side. This typically makes the per-part heatmap focus on
    semantically relevant regions.

    Args:
        temperature: softmax temperature. Smaller -> sharper attention.
                     (Roughly analogous to Sinkhorn's `epsilon`.)
    """

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature: float = float(temperature)

    def forward(
        self,
        visual_tokens:    torch.Tensor,                   # [B, N, D]
        text_part_tokens: torch.Tensor,                   # [B, M, D]
        visual_mask:      Optional[torch.Tensor] = None,  # [B, N]
        part_mask:        Optional[torch.Tensor] = None,  # [B, M]
    ) -> Dict[str, torch.Tensor]:
        B, N, D = visual_tokens.shape
        Bt, M, Dt = text_part_tokens.shape
        if not (B == Bt and D == Dt):
            raise ValueError(
                f"shape mismatch: visual={tuple(visual_tokens.shape)} "
                f"vs text={tuple(text_part_tokens.shape)}"
            )

        # cosine similarity   sim[b, n, m]
        v_n = F.normalize(visual_tokens,    dim=-1)        # [B, N, D]
        t_n = F.normalize(text_part_tokens, dim=-1)        # [B, M, D]
        sim = torch.matmul(v_n, t_n.transpose(-1, -2))     # [B, N, M]
        logits = sim / max(self.temperature, 1e-6)         # [B, N, M]

        # mask out invalid patches BEFORE softmax (per-part softmax is over N)
        if visual_mask is not None:
            mfill = (visual_mask == 0).unsqueeze(-1)       # [B, N, 1]
            logits = logits.masked_fill(mfill, float("-inf"))

        # softmax over the PATCH axis (N), separately per part m.
        # routing_matrix[b, n, m] = (part m allocates this fraction of its
        # attention to patch n).  sum_n routing_matrix[b, :, m] = 1 for each
        # (b, m).  This is the "text-as-query, visual-as-key/value" formulation.
        routing = F.softmax(logits, dim=1)                 # [B, N, M]

        # optional part-mask: zero out parts that should be ignored entirely
        if part_mask is not None:
            routing = routing * part_mask.unsqueeze(1).to(routing.dtype)

        # pooled per-part token. Since sum_n routing == 1 (per part), this is
        # just a weighted average. We still divide by sum to be safe under
        # part_mask zeroing.
        denom = routing.sum(dim=1).clamp_min(1e-12)         # [B, M]
        semantic_v = torch.einsum("bnm,bnd->bmd", routing, visual_tokens) / denom.unsqueeze(-1)

        # Match the Sinkhorn router's output schema so downstream code can be
        # router-agnostic. The "cost" here is 1 - cos, identical convention.
        cost = 1.0 - sim                                    # [B, N, M]
        ot_cost = (routing * cost).sum(dim=(1, 2))          # [B]
        assert routing.shape == (B, N, M), (B, N, M, routing.shape)
        assert semantic_v.shape == (B, M, D), (B, M, D, semantic_v.shape)
        return {
            "routing_matrix":         routing,    # [B, N, M] (softmax over N per m)
            "semantic_visual_tokens": semantic_v, # [B, M, D]
            "ot_cost":                ot_cost,    # [B]
        }


def sinkhorn_semantic_routing(
    visual_tokens:    torch.Tensor,
    text_part_tokens: torch.Tensor,
    visual_mask:      Optional[torch.Tensor] = None,
    part_mask:        Optional[torch.Tensor] = None,
    epsilon:          float = 0.05,
    num_iters:        int = 20,
    cost_mode:        str = "one_minus_cos",
) -> Dict[str, torch.Tensor]:
    """Functional wrapper around SemanticSinkhornRouter for ad-hoc use."""
    return SemanticSinkhornRouter(
        epsilon=epsilon, num_iters=num_iters, cost_mode=cost_mode,
    )(visual_tokens, text_part_tokens, visual_mask, part_mask)
