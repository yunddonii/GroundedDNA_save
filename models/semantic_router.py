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
import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _log_sinkhorn(
    log_K: torch.Tensor,    # [B, N, M]
    log_a: torch.Tensor,    # [B, N]
    log_b: torch.Tensor,    # [B, M]
    num_iters: int = 20,
    epsilon: float = 0.05,
    lambda_a: Optional[float] = None,    # KL marginal penalty on visual side
    lambda_b: Optional[float] = None,    # KL marginal penalty on part side
) -> torch.Tensor:
    """Stable log-space Sinkhorn-Knopp (balanced OR unbalanced).

    Balanced mode (default — lambda_a=lambda_b=None):
        Returns log_P: [B, N, M] such that exp(log_P) has row marginal
        exp(log_a) and column marginal exp(log_b) exactly (after num_iters).

    Unbalanced OT mode (when lambda_a or lambda_b is finite, Chizat et al.
    NeurIPS 2018):
        min <pi, cost> + eps H(pi)
             + lambda_a · KL(pi · 1 || a)         (visual marginal)
             + lambda_b · KL(1^T · pi || b)        (part marginal)
        Update rule: each scaling step multiplied by tau = lambda / (lambda + eps)
        so as lambda → ∞ we recover balanced Sinkhorn; as lambda → 0 the
        marginal becomes free. lambda_a finite + lambda_b finite gives
        full unbalanced OT; only one finite gives semi-unbalanced.

    Effect (v55 motivation): a finite lambda_a allows patches whose total
    cost to all M parts is high (e.g. background / blur / uninformative
    regions) to have row sum LESS than 1/N — i.e. they are partially
    "rejected" from routing rather than forced into some part.
    """
    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)
    eps_safe = max(float(epsilon), 1e-12)
    # tau_a = lambda_a / (lambda_a + eps)  ;  tau_a → 1 as lambda_a → ∞ (balanced)
    tau_a = 1.0 if lambda_a is None else float(lambda_a) / (float(lambda_a) + eps_safe)
    tau_b = 1.0 if lambda_b is None else float(lambda_b) / (float(lambda_b) + eps_safe)
    for _ in range(num_iters):
        # row update : u <- (a / (K v))^tau_a
        log_u = tau_a * (log_a - torch.logsumexp(log_K + log_v.unsqueeze(1), dim=-1))
        # col update : v <- (b / (K^T u))^tau_b
        log_v = tau_b * (log_b - torch.logsumexp(log_K + log_u.unsqueeze(-1), dim=1))
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
        epsilon_override: Optional[float] = None,
        topk_per_patch:   Optional[int]   = None,
        topp_per_patch:   Optional[float] = None,
        ambiguity_topk_threshold: Optional[float] = None,
        ambiguity_topk_ambiguous_k: int = 2,
        adaptive_topp_min: Optional[float] = None,
        adaptive_topp_max: Optional[float] = None,
        adaptive_topp_use_entropy: bool = False,
        perplexity_topk: bool = False,
        codebook_choice_capacity: Optional[float] = None,
        codebook_choice_beta: float = 1.0,
        cost_bias: Optional[torch.Tensor] = None,
        specificity_weighted_marginal: bool = False,
        centered_consensus_mask: bool = False,
        hard_visual_mask: bool = False,
        uot_lambda_a:     Optional[float] = None,
        uot_lambda_b:     Optional[float] = None,
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
        if cost_bias is not None:
            if tuple(cost_bias.shape) != (B, N, M):
                raise ValueError(
                    f"cost_bias must be [B, N, M]={(B, N, M)}, got {tuple(cost_bias.shape)}"
                )
            # v176a: positive bias lowers Sinkhorn transport cost but leaves
            # the reported Wasserstein cost below on the original cosine cost.
            bias = cost_bias.to(device=cost.device, dtype=cost.dtype)       # [B, N, M]
            kernel_cost = cost - bias                                      # [B, N, M]
        else:
            kernel_cost = cost                                             # [B, N, M]
        eps_eff = float(epsilon_override) if epsilon_override is not None else self.epsilon
        log_K = -kernel_cost / max(eps_eff, 1e-6)        # [B, N, M]

        # ---- 3) marginals (uniform unless masks / specificity supplied) --
        device = visual_tokens.device
        dtype  = visual_tokens.dtype
        visual_valid = (
            torch.ones(B, N, dtype=torch.bool, device=device)
            if visual_mask is None else visual_mask.to(device=device, dtype=torch.bool)
        )                                                               # [B, N]
        visual_specificity_mean = None
        visual_marginal_effective_ratio = None
        visual_consensus_mask_ratio = None
        visual_consensus_remaining_ratio = None
        visual_consensus_fallback = None
        if specificity_weighted_marginal and centered_consensus_mask:
            raise ValueError(
                "specificity_weighted_marginal and centered_consensus_mask "
                "are mutually exclusive"
            )
        if centered_consensus_mask:
            part_valid = (
                torch.ones(B, M, dtype=torch.bool, device=device)
                if part_mask is None else part_mask.to(device=device, dtype=torch.bool)
            )                                                           # [B, M]
            visual_count = visual_valid.sum(dim=1, keepdim=True).clamp_min(1)
            slot_mean = (
                (sim * visual_valid.unsqueeze(-1).to(dtype)).sum(dim=1)
                / visual_count.to(dtype)
            )                                                           # [B, M]
            above_slot_mean = sim > slot_mean.unsqueeze(1)              # [B, N, M]
            above_or_invalid = above_slot_mean | ~part_valid.unsqueeze(1)
            common_visual = (
                above_or_invalid.all(dim=-1)
                & part_valid.any(dim=-1, keepdim=True)
                & visual_valid
            ).detach()                                                  # [B, N]
            remaining_visual = visual_valid & ~common_visual            # [B, N]

            # This fallback is only a feasibility guard for malformed or
            # empty masks. Under the strict above-mean criterion, at least
            # one valid token necessarily remains for every non-empty slot.
            fallback = remaining_visual.sum(dim=-1, keepdim=True) == 0   # [B, 1]
            effective_visual_valid = torch.where(
                fallback, visual_valid, remaining_visual,
            )                                                           # [B, N]
            valid_count_f = visual_valid.sum(dim=-1).clamp_min(1).to(dtype)
            visual_consensus_mask_ratio = (
                common_visual.sum(dim=-1).to(dtype) / valid_count_f
            )                                                           # [B]
            visual_consensus_remaining_ratio = (
                effective_visual_valid.sum(dim=-1).to(dtype) / valid_count_f
            )                                                           # [B]
            visual_consensus_fallback = fallback.squeeze(-1).to(dtype)  # [B]
            visual_valid = effective_visual_valid

            valid_weight = visual_valid.to(dtype)
            a = valid_weight / valid_weight.sum(
                dim=1, keepdim=True,
            ).clamp_min(1e-12)                                         # [B, N]
        elif specificity_weighted_marginal:
            part_valid = (
                torch.ones(B, M, dtype=torch.bool, device=device)
                if part_mask is None else part_mask.to(device=device, dtype=torch.bool)
            )                                                           # [B, M]
            sim_for_specificity = sim.masked_fill(
                ~part_valid.unsqueeze(1), -1e4,
            )                                                           # [B, N, M]
            q_slot = sim_for_specificity.softmax(dim=-1)
            q_slot = q_slot * part_valid.unsqueeze(1).to(q_slot.dtype) # [B, N, M]
            entropy = -(
                q_slot * q_slot.clamp_min(1e-12).log()
            ).sum(dim=-1)                                               # [B, N]
            valid_part_count = part_valid.sum(dim=-1).clamp_min(1)      # [B]
            log_part_count = valid_part_count.to(dtype).log()           # [B]
            entropy_norm = torch.where(
                valid_part_count[:, None] > 1,
                entropy / log_part_count[:, None].clamp_min(1e-12),
                torch.zeros_like(entropy),
            ).clamp(0.0, 1.0)                                          # [B, N]
            specificity = (1.0 - entropy_norm) * visual_valid.to(dtype) # [B, N]
            specificity = specificity.detach()

            # If every patch is exactly slot-uniform, retain the legacy
            # uniform valid-patch marginal instead of creating a zero mass.
            valid_weight = visual_valid.to(dtype)                       # [B, N]
            has_specific_mass = specificity.sum(dim=-1, keepdim=True) > 1e-12
            visual_weight = torch.where(
                has_specific_mass, specificity, valid_weight,
            )                                                           # [B, N]
            a = visual_weight / visual_weight.sum(
                dim=1, keepdim=True,
            ).clamp_min(1e-12)                                         # [B, N]

            valid_count = visual_valid.sum(dim=-1).clamp_min(1).to(dtype)
            visual_specificity_mean = (
                specificity.sum(dim=-1) / valid_count
            )                                                           # [B]
            marginal_entropy = -(
                a * a.clamp_min(1e-12).log()
            ).sum(dim=-1)                                               # [B]
            visual_marginal_effective_ratio = (
                marginal_entropy.exp() / valid_count
            ).clamp(0.0, 1.0)                                          # [B]
        else:
            valid_weight = visual_valid.to(dtype)
            a = valid_weight / valid_weight.sum(
                dim=1, keepdim=True,
            ).clamp_min(1e-12)                                         # [B, N]
        if part_mask is None:
            b = torch.full((B, M), 1.0 / M, device=device, dtype=dtype)
        else:
            m = part_mask.to(dtype)
            b = m / m.sum(dim=1, keepdim=True).clamp_min(1e-12)

        log_a = torch.log(a.clamp_min(1e-12))
        log_b = torch.log(b.clamp_min(1e-12))

        # ---- 4) Sinkhorn iterations in log-space ------------------------
        # When uot_lambda_a / uot_lambda_b are provided, switches to
        # unbalanced OT (v55): KL-relaxed marginals let high-cost patches
        # have row sum < 1/N (effective rejection of "uninformative" tokens).
        log_P = _log_sinkhorn(
            log_K, log_a, log_b,
            num_iters=self.num_iters,
            epsilon=eps_eff,
            lambda_a=uot_lambda_a,
            lambda_b=uot_lambda_b,
        )
        P = torch.exp(log_P)                             # [B, N, M]
        if centered_consensus_mask or hard_visual_mask:
            # Make the candidate exclusion exact. The legacy visual-mask
            # path remains bit-for-bit unchanged when both flags are disabled.
            P = P * visual_valid.unsqueeze(-1).to(P.dtype)             # [B, N, M]

        # ---- 4b) optional top-k mask per patch (v33b hardening) ---------
        # Keep only the top-k largest part-assignments per patch (along the
        # M axis) and zero out the rest, then renormalize each row to keep
        # patch marginal = a_n. k=1 -> hard argmax (Sinkhorn balance is
        # broken). k>=M -> no-op. Renormalization is row-wise (per patch),
        # so each patch's total mass stays the same; the *column* sums
        # (per-part mass) drift away from the Sinkhorn target.
        # Preserve UOT mass relaxation when topk/topp masks renormalize:
        # use the actual Sinkhorn row sum (which may be < a[n] under UOT)
        # instead of forcing back to a[n].
        target_row_sum = P.sum(dim=-1, keepdim=True).clamp_min(1e-12)  # [B, N, 1]

        if topk_per_patch is not None and 1 <= int(topk_per_patch) < M:
            k = int(topk_per_patch)
            # threshold per patch n: k-th largest value in P[n, :]
            kth = P.topk(k=k, dim=-1).values[..., -1:]      # [B, N, 1]
            mask = (P >= kth).to(P.dtype)                    # [B, N, M] in {0,1}
            P_masked = P * mask
            row_sum = P_masked.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            # rescale so each patch row keeps its actual Sinkhorn row sum
            # (= a[n] in balanced mode; < a[n] for rejected patches in UOT)
            P = P_masked / row_sum * target_row_sum

        # ---- 4c) optional top-p (cumulative-mass) mask per patch (v46) -----
        # Adaptive-k routing: for each patch n, keep the smallest set of parts
        # whose probabilities (sorted descending) cumulatively reach `topp`.
        # Top-1 always kept so every patch routes to at least one part.
        # Use case: patches that are clearly about one part get k_eff=1
        # (forcing codebook specialization), ambiguous patches get larger
        # k_eff (preserving distributed routing). Mutually exclusive with
        # `topk_per_patch` -- if both are set, top-k runs first then top-p.
        # Renormalization keeps each patch's row marginal at a[n] (same as
        # the top-k branch above).
        if topp_per_patch is not None and 0.0 < float(topp_per_patch) < 1.0:
            tau = float(topp_per_patch)
            P_prob = P / target_row_sum                                       # [B, N, M], row sums to 1
            assert P_prob.shape == P.shape, (
                f"P_prob must match P shape, got {tuple(P_prob.shape)} vs {tuple(P.shape)}"
            )
            sorted_P, sorted_idx = P_prob.sort(dim=-1, descending=True)       # [B, N, M]
            cum = sorted_P.cumsum(dim=-1)                                     # [B, N, M]
            # keep position i iff cum[..., i-1] < tau (i.e. we haven't yet
            # passed threshold when entering this position). always keep i=0.
            prev_cum = torch.cat(
                [torch.zeros_like(cum[..., :1]), cum[..., :-1]], dim=-1
            )                                                                 # [B, N, M]
            keep_sorted = (prev_cum < tau)                                    # [B, N, M] bool
            keep_sorted[..., 0] = True
            # scatter back to original part order
            keep = torch.zeros_like(P, dtype=torch.bool).scatter_(
                -1, sorted_idx, keep_sorted,
            )                                                                 # [B, N, M] bool
            P_masked = P * keep.to(P.dtype)
            row_sum = P_masked.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            P = P_masked / row_sum * target_row_sum

        # ---- 4d) optional ambiguity-aware top-k mask per patch ------------
        # Confident patches keep only top-1, while ambiguous patches keep
        # top-k. This is a middle ground between v46 top-p and v79c one-hot
        # hard routing: clear patches specialize, ambiguous patches preserve
        # multi-part evidence. Confidence is max_m P[b, n, m].
        if ambiguity_topk_threshold is not None:
            th = float(ambiguity_topk_threshold)
            if 0.0 < th < 1.0:
                k_amb = max(1, min(int(ambiguity_topk_ambiguous_k), M))
                P_prob = P / target_row_sum                                   # [B, N, M], row sums to 1
                assert P_prob.shape == P.shape, (
                    f"P_prob must match P shape, got {tuple(P_prob.shape)} vs {tuple(P.shape)}"
                )
                p_max = P_prob.max(dim=-1).values                             # [B, N]
                sorted_P, sorted_idx = P_prob.sort(dim=-1, descending=True)    # [B, N, M]
                rank = torch.arange(M, device=P.device).view(1, 1, M)          # [1, 1, M]
                k_eff = torch.where(
                    p_max >= th,
                    torch.ones_like(p_max, dtype=torch.long),
                    torch.full_like(p_max, k_amb, dtype=torch.long),
                )                                                              # [B, N]
                keep_sorted = rank < k_eff.unsqueeze(-1)                       # [B, N, M]
                keep = torch.zeros_like(P, dtype=torch.bool).scatter_(
                    -1, sorted_idx, keep_sorted,
                )                                                              # [B, N, M]
                P_masked = P * keep.to(P.dtype)
                row_sum = P_masked.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                P = P_masked / row_sum * target_row_sum

        # Snapshot the plan BEFORE any sparsification. The slot-diversity loss
        # (ConceptHash's L_csd, Eq. 8) has to see it: once a slot is masked out
        # its column is exactly zero, so a cosine between slot columns would
        # read a starved slot as perfectly "diverse" and give it no gradient —
        # the opposite of what the term is for.
        P_premask = P

        # ---- 4e) optional confidence-adaptive top-p mask per patch --------
        # Patch-specific threshold is driven either by max-probability
        # confidence or normalized entropy over the local parts. Confident /
        # low-entropy patches get a lower threshold (sparser routing);
        # ambiguous / high-entropy patches keep more parts.
        if adaptive_topp_min is not None and adaptive_topp_max is not None:
            tau_min = float(adaptive_topp_min)
            tau_max = float(adaptive_topp_max)
            if 0.0 < tau_min <= tau_max < 1.0:
                P_prob = P / target_row_sum                                    # [B, N, M], row sums to 1
                assert P_prob.shape == P.shape, (
                    f"P_prob must match P shape, got {tuple(P_prob.shape)} vs {tuple(P.shape)}"
                )
                if adaptive_topp_use_entropy:
                    entropy = -(P_prob.clamp_min(1e-12) * P_prob.clamp_min(1e-12).log()).sum(
                        dim=-1, keepdim=True,
                    )                                                          # [B, N, 1]
                    entropy_norm = entropy / torch.log(
                        torch.tensor(float(M), device=P.device, dtype=P.dtype)
                    ).clamp_min(1e-12)                                         # [B, N, 1]
                    tau_signal = entropy_norm.clamp(0.0, 1.0)                  # [B, N, 1]
                else:
                    p_max = P_prob.max(dim=-1, keepdim=True).values            # [B, N, 1]
                    tau_signal = (1.0 - p_max).clamp(0.0, 1.0)                 # [B, N, 1]
                tau = tau_min + tau_signal * (tau_max - tau_min)               # [B, N, 1]
                sorted_P, sorted_idx = P_prob.sort(dim=-1, descending=True)    # [B, N, M]
                cum = sorted_P.cumsum(dim=-1)                                  # [B, N, M]
                prev_cum = torch.cat(
                    [torch.zeros_like(cum[..., :1]), cum[..., :-1]], dim=-1
                )                                                              # [B, N, M]
                keep_sorted = prev_cum < tau                                   # [B, N, M]
                keep_sorted[..., 0] = True
                keep = torch.zeros_like(P, dtype=torch.bool).scatter_(
                    -1, sorted_idx, keep_sorted,
                )                                                              # [B, N, M]
                P_masked = P * keep.to(P.dtype)
                row_sum = P_masked.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                P = P_masked / row_sum * target_row_sum

        # ---- 4f) optional perplexity top-k mask per patch (v84a) ---------
        # Per-patch effective-k is determined by the routing distribution's
        # own perplexity: k_n = ceil(M^{H_norm(P_n)}), where
        # H_norm = H(P_n) / log(M) in [0, 1]. Confident patches (H≈0) get
        # k=1 (hard routing); uniform patches (H=log M) get k=M (dense).
        # Zero hand-tuned hyperparameters — the per-patch sparsity is set
        # entirely by Shannon perplexity. Mutually exclusive with the
        # adaptive_topp branch (caller enforces).
        if perplexity_topk:
            P_prob = P / target_row_sum                                    # [B, N, M], row sums to 1
            assert P_prob.shape == P.shape, (
                f"P_prob must match P shape, got {tuple(P_prob.shape)} vs {tuple(P.shape)}"
            )
            entropy = -(P_prob.clamp_min(1e-12) * P_prob.clamp_min(1e-12).log()).sum(
                dim=-1, keepdim=False,
            )                                                              # [B, N]
            log_M = math.log(float(M))
            H_norm = (entropy / max(log_M, 1e-12)).clamp(0.0, 1.0)         # [B, N]
            # k_eff = ceil(M^H_norm), clamped to [1, M]
            k_eff_f = float(M) ** H_norm                                    # [B, N], in [1, M]
            k_eff = k_eff_f.ceil().clamp(1, M).long()                       # [B, N]
            sorted_P, sorted_idx = P_prob.sort(dim=-1, descending=True)    # [B, N, M]
            rank = torch.arange(M, device=P.device).view(1, 1, M)          # [1, 1, M]
            keep_sorted = rank < k_eff.unsqueeze(-1)                       # [B, N, M] bool
            keep = torch.zeros_like(P, dtype=torch.bool).scatter_(
                -1, sorted_idx, keep_sorted,
            )                                                              # [B, N, M] bool
            P_masked = P * keep.to(P.dtype)
            row_sum = P_masked.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            P = P_masked / row_sum * target_row_sum

        # ---- 4g) optional codebook-choice token filtering (v85) ----------
        # Expert-choice-inspired second pass: after patch-wise masks propose
        # candidate codebooks, each codebook keeps only its strongest visual
        # tokens under a per-sample capacity budget. This sharpens codebook
        # specialization while a row-wise top-1 fallback guarantees every
        # visual token still routes somewhere. Shapes:
        #   P_before_choice: [B, N, M], keep_col/fallback/keep: [B, N, M]
        if codebook_choice_capacity is not None:
            cap = float(codebook_choice_capacity)
            if cap > 0.0:
                keep_k = int(math.ceil((float(N) / float(M)) * cap))
                keep_k = max(1, min(keep_k, N))
                P_before_choice = P
                top_idx = P_before_choice.topk(k=keep_k, dim=1).indices     # [B, keep_k, M]
                keep_col = torch.zeros_like(P_before_choice, dtype=torch.bool).scatter_(
                    1, top_idx, True,
                )                                                           # [B, N, M]
                P_masked = P_before_choice * keep_col.to(P_before_choice.dtype)
                row_has_mass = P_masked.sum(dim=-1, keepdim=True) > 0        # [B, N, 1]
                fallback_idx = P_before_choice.argmax(dim=-1, keepdim=True)  # [B, N, 1]
                fallback = torch.zeros_like(P_before_choice, dtype=torch.bool).scatter_(
                    -1, fallback_idx, True,
                )                                                           # [B, N, M]
                keep = torch.where(row_has_mass, keep_col, fallback)         # [B, N, M]
                assert keep.shape == P_before_choice.shape, (
                    f"codebook-choice keep shape mismatch: "
                    f"{tuple(keep.shape)} vs {tuple(P_before_choice.shape)}"
                )
                P_masked = P_before_choice * keep.to(P_before_choice.dtype)
                row_sum = P_masked.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                P_choice = P_masked / row_sum * target_row_sum              # [B, N, M]
                assert P_choice.shape == P_before_choice.shape, (
                    f"codebook-choice output shape mismatch: "
                    f"{tuple(P_choice.shape)} vs {tuple(P_before_choice.shape)}"
                )
                beta = max(0.0, min(float(codebook_choice_beta), 1.0))
                P = (1.0 - beta) * P_before_choice + beta * P_choice

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
            "routing_matrix_premask": P_premask,    # [B, N, M] before 4e/4f
            "semantic_visual_tokens": semantic_v,   # [B, M, D]
            "ot_cost":                ot_cost,      # [B]   per-sample W_e value
            "visual_specificity_mean": visual_specificity_mean,
            "visual_marginal_effective_ratio": visual_marginal_effective_ratio,
            "visual_consensus_mask_ratio": visual_consensus_mask_ratio,
            "visual_consensus_remaining_ratio": visual_consensus_remaining_ratio,
            "visual_consensus_fallback": visual_consensus_fallback,
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
        adaptive_topp_min: Optional[float] = None,        # v107a-attn
        adaptive_topp_max: Optional[float] = None,        # v107a-attn
        adaptive_topp_use_entropy: bool = False,          # v107a-attn
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

        # v107a-attn: confidence-adaptive top-p mask per patch. We re-normalize
        # routing[b, n, :] into a per-patch distribution over parts (this row
        # does NOT sum to 1 in the attention router by construction; here we
        # just normalize to apply the same top-p selection logic as Sinkhorn).
        # The masked routing is then used for pooling, with denom recomputed
        # to keep semantic_v a proper convex combination.
        if adaptive_topp_min is not None and adaptive_topp_max is not None:
            tau_min = float(adaptive_topp_min)
            tau_max = float(adaptive_topp_max)
            if 0.0 < tau_min <= tau_max < 1.0:
                eps = 1e-12
                # Per-patch distribution over parts (sums to 1 along M).
                P_patch = routing / routing.sum(dim=-1, keepdim=True).clamp_min(eps)  # [B, N, M]
                if adaptive_topp_use_entropy:
                    entropy = -(P_patch.clamp_min(eps) * P_patch.clamp_min(eps).log()).sum(
                        dim=-1, keepdim=True,
                    )                                                  # [B, N, 1]
                    entropy_norm = entropy / torch.log(
                        torch.tensor(float(M), device=P_patch.device, dtype=P_patch.dtype)
                    ).clamp_min(eps)                                   # [B, N, 1]
                    tau_signal = entropy_norm.clamp(0.0, 1.0)
                else:
                    p_max = P_patch.max(dim=-1, keepdim=True).values   # [B, N, 1]
                    tau_signal = (1.0 - p_max).clamp(0.0, 1.0)
                tau = tau_min + tau_signal * (tau_max - tau_min)       # [B, N, 1]
                sorted_P, sorted_idx = P_patch.sort(dim=-1, descending=True)  # [B, N, M]
                cum = sorted_P.cumsum(dim=-1)                          # [B, N, M]
                prev_cum = torch.cat(
                    [torch.zeros_like(cum[..., :1]), cum[..., :-1]], dim=-1
                )                                                       # [B, N, M]
                keep_sorted = prev_cum < tau                            # [B, N, M]
                keep_sorted[..., 0] = True                              # always keep top-1
                keep = torch.zeros_like(P_patch, dtype=torch.bool).scatter_(
                    -1, sorted_idx, keep_sorted,
                )                                                       # [B, N, M]
                routing = routing * keep.to(routing.dtype)

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


class SemanticSlotAttentionRouter(nn.Module):
    """v107a-slot: Slot-Attention router (Locatello et al. NeurIPS 2020).

    The M textual semantic embeddings are used to *initialize* M slots, which
    then iteratively bind to visual patches via competitive attention.
    At each iteration:
      q       = Linear_Q( LN(slots) )                            # [B, M, D]
      k, v    = Linear_K( LN(visual_tokens) ), Linear_V(...)     # [B, N, D]
      dots    = q @ k.T / sqrt(D)                                # [B, M, N]
      attn    = softmax(dots, dim=SLOT_axis)                     # softmax over M (NOT N)
      weights = attn / sum_n(attn)                               # normalize per slot
      updates = weights @ v                                      # [B, M, D]
      slots   = GRU(updates, slots)                              # shared per slot
      slots   = slots + MLP(LN(slots))                           # residual MLP

    The competitive softmax (over slots) is the key contrast with the
    cross-attention router (softmax over patches): in slot attention, each
    *patch* distributes its attention across slots, so slots COMPETE for
    explaining each patch. This is the property that gives slot attention
    its object-binding behaviour.

    Drop-in replacement for SemanticSinkhornRouter / SemanticAttentionRouter
    with the same forward signature + output dict.

    Args:
        d_model:  feature dimension (= slot dimension).
        n_iters:  number of slot-attention iterations (paper default 3).
        eps:      numerical stability for normalization.
    """

    def __init__(
        self,
        d_model: int,
        n_iters: int = 3,
        eps:     float = 1e-8,
    ) -> None:
        super().__init__()
        self.d_model: int = int(d_model)
        self.n_iters: int = max(1, int(n_iters))
        self.eps:     float = float(eps)
        # Per-iteration shared transformations (Locatello et al. Table 1)
        self.norm_slots = nn.LayerNorm(self.d_model)
        self.norm_input = nn.LayerNorm(self.d_model)
        self.norm_mlp   = nn.LayerNorm(self.d_model)
        self.to_q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.to_k = nn.Linear(self.d_model, self.d_model, bias=False)
        self.to_v = nn.Linear(self.d_model, self.d_model, bias=False)
        # GRU cell: input = updates [B*M, D], state = slots [B*M, D]
        # Shared across all slots (paper).
        self.gru = nn.GRUCell(self.d_model, self.d_model)
        # Residual MLP (paper uses 2 layers w/ ReLU + hidden = d_model*2 or 4*d_model;
        # we follow the original 2*d_model hidden as in slot_attention reference
        # implementations).
        self.mlp = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.ReLU(),
            nn.Linear(self.d_model * 2, self.d_model),
        )
        # Attention scale 1/sqrt(D) (paper default).
        self.scale: float = float(self.d_model ** -0.5)

    def forward(
        self,
        visual_tokens:    torch.Tensor,                   # [B, N, D]
        text_part_tokens: torch.Tensor,                   # [B, M, D]
        visual_mask:      Optional[torch.Tensor] = None,  # [B, N]
        part_mask:        Optional[torch.Tensor] = None,  # [B, M]
        # Accept and ignore extra kwargs for router-agnostic call sites.
        **_unused,
    ) -> Dict[str, torch.Tensor]:
        B, N, D = visual_tokens.shape
        Bt, M, Dt = text_part_tokens.shape
        if not (B == Bt and D == Dt):
            raise ValueError(
                f"shape mismatch: visual={tuple(visual_tokens.shape)} "
                f"vs text={tuple(text_part_tokens.shape)}"
            )
        if D != self.d_model:
            raise ValueError(
                f"[SemanticSlotAttentionRouter] expected d_model={self.d_model}, got {D}"
            )

        # Slot initialization: M text-derived embeddings.
        slots = text_part_tokens                                  # [B, M, D]

        # Precompute k, v once (do not depend on slot state).
        x_norm = self.norm_input(visual_tokens)                   # [B, N, D]
        k = self.to_k(x_norm)                                     # [B, N, D]
        v = self.to_v(x_norm)                                     # [B, N, D]

        # Patch / part masks pre-computed for use inside the loop. The patch
        # mask is applied AFTER the slot-axis softmax (to zero out contributions
        # from invalid patches without producing NaN: pre-softmax fill would
        # give 0/0 since the softmax is over the SLOT axis, not patch axis).
        if visual_mask is not None:
            patch_keep = visual_mask.to(slots.dtype).unsqueeze(1)  # [B, 1, N]
        else:
            patch_keep = None
        if part_mask is not None:
            part_keep = part_mask.to(slots.dtype).unsqueeze(-1)   # [B, M, 1]
        else:
            part_keep = None

        attn = None
        for _ in range(self.n_iters):
            slots_prev = slots
            q = self.to_q(self.norm_slots(slots)) * self.scale    # [B, M, D]
            # Dot products: [B, M, N]
            dots = torch.einsum('bmd,bnd->bmn', q, k)
            # Competitive softmax: over SLOT axis (M, dim=1).
            # Each patch's attention is distributed across the M slots, so
            # slots compete for explaining each patch.
            attn = F.softmax(dots, dim=1)                         # [B, M, N]
            # Zero out invalid patches (post-softmax to avoid NaN from
            # all--inf rows under slot-axis softmax).
            if patch_keep is not None:
                attn = attn * patch_keep
            # Optional part mask (zero out specific slots).
            if part_keep is not None:
                attn = attn * part_keep
            # Per-slot normalization over patches: weights[b, m, n] = attn[b, m, n]
            # / sum_n attn[b, m, n] -> each slot's weighted contributions sum to 1.
            weights = attn / attn.sum(dim=-1, keepdim=True).clamp_min(self.eps)
            # Aggregated updates from patches into slots.
            updates = torch.einsum('bmn,bnd->bmd', weights, v)    # [B, M, D]
            # GRU update -- shared across slots.
            slots = self.gru(
                updates.reshape(B * M, D),
                slots_prev.reshape(B * M, D),
            ).reshape(B, M, D)
            # Residual MLP -- shared across slots.
            slots = slots + self.mlp(self.norm_mlp(slots))

        # Output: routing matrix in [B, N, M] convention for downstream compatibility.
        routing = attn.transpose(1, 2)                            # [B, N, M]
        # semantic_visual_tokens = final slot states (after T iterations).
        semantic_v = slots                                        # [B, M, D]
        # OT-like cost using the SAME 1-cos convention as the other routers
        # (for logging compatibility; not used by the slot attention forward).
        v_n = F.normalize(visual_tokens,    dim=-1)
        t_n = F.normalize(text_part_tokens, dim=-1)
        sim = torch.matmul(v_n, t_n.transpose(-1, -2))            # [B, N, M]
        cost = 1.0 - sim
        ot_cost = (routing * cost).sum(dim=(1, 2))                # [B]
        assert routing.shape == (B, N, M), (B, N, M, routing.shape)
        assert semantic_v.shape == (B, M, D), (B, M, D, semantic_v.shape)
        return {
            "routing_matrix":         routing,    # [B, N, M] (softmax over M, normalized over N per m)
            "semantic_visual_tokens": semantic_v, # [B, M, D] (final slots)
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
