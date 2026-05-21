"""DNACodonHashLoss for ``SigLIP2SemanticOTModel``.

This module aggregates all stage-1 losses on the model's forward output
dict. It is intentionally additive — `train_siglip2.py` only needs to
instantiate ``DNACodonHashLoss`` once and call it after each forward pass.

Loss components (all returned as scalar tensors in the output dict):

    loss_hash       — soft DNA code preserves label similarity
                      (MSE between continuous-code pairwise sim and label sim S).
    loss_hash_hard  — STE-quantized hard DNA code preserves label similarity.
                      Closes the train↔test gap by directly supervising the
                      hard one-hot path with a gradient that flows via STE.
    loss_vq         — VQ codebook-side commitment between
                      semantic_visual_tokens (z) and quantized_tokens_raw (q).
    loss_quant      — codon-side commitment between continuous_code and
                      dna_hash_code_hard. Mirrors ``loss_vq`` but in
                      base-probability space rather than codeword space.
    loss_anchor     — codebook mean anchors follow EMA-accumulated training
                      text anchors (local 5 codebooks; 0 when text absent).
    loss_dna        — entropy + base-balance regularizer on continuous_code.
    loss_bu         — codebook usage balance + assignment uncorrelated.

The R1–R4 alignment family (`loss_align`, `loss_siglip_align`,
`loss_wasserstein`), v15 `loss_codebook_ortho`, and deprecated
`loss_global` were removed on 2026-05-13 after exhaustive ablation showed
they were either no-ops (`loss_global` under EMA codebook mode) or net
mAP regressions. See `docs/PROJECT_LOG.md` and
`backup/results_R_series/README.md` for the archived experiment record.

Label format:
    ``train_siglip2.py``'s existing dataloader returns ``batch['label']`` as a
    ``[B, num_classes]`` multi-hot / one-hot tensor (see ``dataloaders.py``).
    Pass it as ``multi_hot_labels``. For single-label datasets, passing class
    indices via ``labels`` ([B] LongTensor) also works.

Persisting the EMA text anchor:
    ``ema_text_anchor`` is a registered buffer; ``criterion.state_dict()``
    captures it. ``train_siglip2.py`` saves the criterion alongside the model
    so the EMA survives resume / checkpoint reload.
"""

from __future__ import annotations
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------- helpers

def get_off_diagonal_mask(batch_size: int, device: torch.device) -> torch.Tensor:
    """[B, B] BoolTensor with True off-diagonal."""
    m = torch.ones((batch_size, batch_size), dtype=torch.bool, device=device)
    m.fill_diagonal_(False)
    return m


def build_label_similarity(
    labels: Optional[torch.Tensor] = None,
    multi_hot_labels: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return pairwise label similarity matrix S with shape [B, B] in [0, 1].

    - multi_hot_labels [B, C] → Jaccard similarity
                                S = |L_i ∩ L_j| / |L_i ∪ L_j|
    - labels           [B]    → 1 if labels[i] == labels[j] else 0

    For one-hot inputs Jaccard reduces exactly to the equality form, so a
    one-hot tensor passed as ``multi_hot_labels`` is handled correctly.
    """
    if multi_hot_labels is not None:
        L = multi_hot_labels.float()
        intersection = L @ L.T                                  # [B, B]
        counts = L.sum(dim=-1)                                  # [B]
        union = counts.unsqueeze(1) + counts.unsqueeze(0) - intersection  # [B, B]
        return intersection / union.clamp_min(eps)
    if labels is not None:
        L = labels.view(-1)
        return (L.unsqueeze(0) == L.unsqueeze(1)).float()
    raise ValueError(
        "[build_label_similarity] one of `labels` or `multi_hot_labels` must be provided."
    )


def build_siglip_cos_similarity(
    visual_global_feat: torch.Tensor,
) -> torch.Tensor:
    """Return self-supervised pairwise similarity from frozen SigLIP2 features.

    visual_global_feat : [B, D_proj]   SigLIP2 image embedding (frozen).
    Returns S ∈ [0, 1] via S = (cos_sim + 1) / 2, fully detached.

    Used when `hash_target_mode == 'siglip_cos'` (v27a) to remove the label
    signal from loss_hash / loss_hash_hard and run a truly unsupervised
    setting comparable to CIBHash / CIMON / SPQ / MLS3RDUH.

    NOTE: v27a empirically collapses (mAP 0.54, unique 5e-4) because the
    SigLIP2 cos distribution is narrow around [0.7, 0.9], so the rescaled
    S target is ~0.85 for all pairs — the HashNet logistic then pushes
    every pair toward the same positive code. Use
    `build_siglip_cos_topk_similarity` (v27b) for a binarized variant
    that mixes positive and negative pseudo-labels.
    """
    v = F.normalize(visual_global_feat.detach().float(), dim=-1)   # [B, D]
    cos = v @ v.T                                                  # [B, B] ∈ [-1, 1]
    return ((cos + 1.0) * 0.5).clamp(0.0, 1.0)                     # [B, B] ∈ [ 0, 1]


def build_siglip_cos_topk_similarity(
    visual_global_feat: torch.Tensor,
    pos_rate: float = 0.2,
) -> torch.Tensor:
    """Binary pseudo-label pairwise similarity from SigLIP2 cos top-k.

    visual_global_feat : [B, D_proj]   SigLIP2 image embedding (frozen).
    pos_rate           : fraction of off-diagonal pairs marked positive.

    Returns S ∈ {0, 1} where the top `pos_rate` fraction of off-diagonal
    cosine similarities (within the batch) are 1 and the rest are 0.
    Diagonal is forced to 1 (self-similarity).

    Self-supervised replacement for the supervised Jaccard target. The
    binarization fixes v27a's collapse: half the pairs now provide a
    "stay apart" signal, restoring the contrastive pressure that the
    HashNet logistic needs to spread codes across the codebook. Matches
    the operational pattern of CIBHash (NtXent positives), CIMON (spectral
    pseudo-labels), and MLS3RDUH (kNN graph).
    """
    v = F.normalize(visual_global_feat.detach().float(), dim=-1)   # [B, D]
    cos = v @ v.T                                                  # [B, B] ∈ [-1, 1]
    B = cos.shape[0]
    mask = ~torch.eye(B, dtype=torch.bool, device=cos.device)      # [B, B]
    offdiag = cos[mask]                                            # [B*(B-1)]
    rate = float(pos_rate)
    rate = max(min(rate, 1.0 - 1e-6), 1e-6)
    tau = torch.quantile(offdiag, 1.0 - rate)                       # scalar
    S = (cos > tau).to(cos.dtype)                                   # [B, B]
    # Diagonal: always self-similar (mirrors Jaccard / one-hot conventions).
    S = S.masked_fill(~mask, 1.0)
    return S


# --------------------------------------------------------------- main loss

class DNACodonHashLoss(nn.Module):
    """Aggregate stage-1 loss for ``SigLIP2SemanticOTModel`` outputs.

    Constructor reads weights / hyperparameters from a ``Config``-like object
    via ``getattr`` fallbacks, so plugging this into the existing ``train_siglip2.py``
    requires no edits to ``config.py``.
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cfg = cfg

        # ---------- loss weights ------------------------------------------
        # Active set after the 2026-05-13 cleanup. The R1–R4 alignment family
        # (`loss_align`, `loss_siglip_align`, `loss_wasserstein`), v15
        # `loss_codebook_ortho`, and the deprecated `loss_global` were all
        # removed -- see `docs/PROJECT_LOG.md` for the full rationale and
        # `backup/results_R_series/README.md` for archived results.
        self.lambda_hash       = float(getattr(cfg, "lambda_hash",      1.0))
        self.lambda_hash_hard  = float(getattr(cfg, "lambda_hash_hard", 0.5))
        # ----- Form selector for `loss_hash` ------------------------------
        # 'mse'     : MSE between continuous-code pairwise sim and Jaccard
        #              label-sim (legacy default).
        # 'hashnet' : HashNet-style class-weighted logistic likelihood on a
        #              centred-and-scaled score (Option A from PROJECT_LOG).
        #              Uses BINARY S (any shared label) and a continuation
        #              parameter `hashnet_alpha`. Less aggressive about
        #              identical-target collapse than MSE, so it should
        #              reduce the 60% Flickr collision rate.
        self.lambda_hash_type  = str  (getattr(cfg, "lambda_hash_type", "mse"))
        self.hashnet_alpha     = float(getattr(cfg, "hashnet_alpha",    1.0))
        # Wasserstein alignment loss (R4 in v11, restored for v24a).
        # Uses the per-sample <pi, cost> already computed by the Sinkhorn
        # router; adding it as a loss term pulls the visual_adapter +
        # text_adapter into a shared space where the OT transport cost is
        # small. Default 0.0 keeps it off; v11 sweet spot was 0.05.
        self.lambda_wasserstein = float(getattr(cfg, "lambda_wasserstein", 0.0))
        # v66: per-codon text-anchored aux CE loss weight. The model computes
        # `out["loss_text_anchor"]` per forward (sum across 6 codebooks). 0
        # default keeps the loss off.
        self.lambda_codon_text_anchor = float(getattr(cfg, "lambda_codon_text_anchor", 0.0))
        # If True, the hashnet logistic uses fractional Jaccard S instead of
        # binary any-shared S. Preserves per-label-combo granularity in the
        # target probability so different powerset combinations get different
        # codes -- intended to reduce v18's 90% collision rate.
        self.hashnet_use_jaccard = bool(getattr(cfg, "hashnet_use_jaccard", False))
        # Cap on S_target for the hashnet logistic (v20a). When set below 1.0,
        # full-label-overlap positive pairs no longer aim for sigmoid(score)=1
        # exactly; the target similarity is bounded so the cluster of same-
        # powerset samples keeps a residual degree of freedom -> distinct
        # codes within the cluster -> higher unique_code_ratio. Default 1.0
        # disables the cap and reproduces v18.
        self.hashnet_S_cap     = float(getattr(cfg, "hashnet_S_cap",   1.0))
        # v27a: source of the pairwise similarity target S for
        # loss_hash / loss_hash_hard. 'jaccard' (default) uses multi_hot
        # labels (supervised). 'siglip_cos' substitutes the frozen SigLIP2
        # visual_global cosine similarity rescaled to [0,1] (unsupervised).
        self.hash_target_mode  = str  (getattr(cfg, "hash_target_mode", "jaccard"))
        self.siglip_cos_pos_rate = float(getattr(cfg, "siglip_cos_pos_rate", 0.2))
        # v28 reconstruction loss (pixel or siglip_feat decoder).
        self.lambda_recon       = float(getattr(cfg, "lambda_recon",       0.0))
        self.decoder_target     = str  (getattr(cfg, "decoder_target",     "siglip_feat"))
        # v29 NtXent contrastive loss on DNA codes (paired-aug).
        self.lambda_ntxent      = float(getattr(cfg, "lambda_ntxent",      0.0))
        self.ntxent_temperature = float(getattr(cfg, "ntxent_temperature", 0.3))
        # v31 ablation: NtXent target granularity.
        # 'global' contrasts full DNA code (v29 default).
        # 'per_codebook' computes 6 NtXents (one per codebook's 3-codon
        # group) and averages -- forces each codebook to discriminate
        # independently (compositional-independence test).
        self.ntxent_mode        = str  (getattr(cfg, "ntxent_mode",        "global"))
        # v42 (Q2): dynamic per-pair temperature modulated by text-caption
        # cosine sim. tau_ij = base_tau * (1 + alpha * cos(text_i, text_j)).
        # Only used when ntxent_mode='per_codebook' AND text_part_raw is
        # available (text path on) AND alpha > 0.
        self.ntxent_dynamic_tau       = bool (getattr(cfg, "ntxent_dynamic_tau",       False))
        self.ntxent_dynamic_tau_alpha = float(getattr(cfg, "ntxent_dynamic_tau_alpha", 0.0))
        # v60: skip dynamic-tau for m=0 (use static base_tau on C_global)
        self.ntxent_dynamic_tau_skip_global = bool(getattr(cfg, "ntxent_dynamic_tau_skip_global", False))
        # v61: for m=0 dynamic-tau, use mean of 5 local text features
        self.ntxent_global_use_local_mean   = bool(getattr(cfg, "ntxent_global_use_local_mean",   False))
        # v44 (B1): cross-slot text orthogonality reg on text_part_tokens.
        self.lambda_ortho_text        = float(getattr(cfg, "lambda_ortho_text",        0.0))
        self.lambda_vq         = float(getattr(cfg, "lambda_vq",        0.25))
        self.lambda_quant      = float(getattr(cfg, "lambda_quant",     0.05))
        self.lambda_anchor     = float(getattr(cfg, "lambda_anchor",    0.05))
        self.lambda_dna        = float(getattr(cfg, "lambda_dna",       0.01))
        self.lambda_bu         = float(getattr(cfg, "lambda_bu",        0.01))

        # ---------- internal hyperparameters ------------------------------
        self.beta_vq                  = float(getattr(cfg, "beta_vq",                  0.25))
        self.eta_base_balance         = float(getattr(cfg, "eta_base_balance",         1.0))
        self.tau_codebook_assignment  = float(getattr(cfg, "tau_codebook_assignment",  0.1))
        self.rho_cb_uncorr            = float(getattr(cfg, "rho_cb_uncorr",            0.1))
        self.bu_warmup_epochs         = int  (getattr(cfg, "bu_warmup_epochs",         0))
        self.anchor_ema_momentum      = float(getattr(cfg, "anchor_ema_momentum",      0.99))

        self.eps = 1e-8

        # ---------- EMA buffer for stable text anchor ---------------------
        # Registered as a persistent buffer so it travels with state_dict.
        # Lazily resized on first forward (we don't know D until then).
        # See `_update_ema_text_anchor`.
        self.register_buffer(
            "ema_text_anchor",
            torch.zeros(0),
            persistent=True,
        )

    # ============================================== EMA anchor helper

    def _update_ema_text_anchor(self, batch_anchor: torch.Tensor) -> torch.Tensor:
        """Maintain an EMA over the L2-normalized batch-mean text anchor.

        Args:
            batch_anchor : [5, D]   already L2-normalized per row.

        First call (or shape change) → buffer is replaced with a clone of
        ``batch_anchor``. Subsequent calls do an in-place EMA update with
        momentum ``self.anchor_ema_momentum``:

            ema ← m · ema + (1 − m) · batch_anchor

        Returns the EMA tensor (caller is expected to .detach() before use).
        """
        # buffer 미초기화 또는 shape 변경 시 새로 등록 (resume 안전: shape이
        # 같으면 진짜 EMA를 누적하니까 첫-호출-overwrite는 일어나지 않는다).
        if (
            self.ema_text_anchor.numel() == 0
            or self.ema_text_anchor.shape != batch_anchor.shape
        ):
            self.ema_text_anchor = batch_anchor.detach().clone()
            return self.ema_text_anchor
        m = self.anchor_ema_momentum
        with torch.no_grad():
            self.ema_text_anchor.mul_(m).add_(batch_anchor.detach(), alpha=(1.0 - m))
        return self.ema_text_anchor

    # ============================================== individual components

    def _loss_hash(
        self, u: torch.Tensor, S: torch.Tensor, mask: torch.Tensor,
    ) -> torch.Tensor:
        """Soft DNA retrieval loss.

        u : continuous_code  [B, 18, 4]   (softmax over A/C/G/T at each position)
        sim_dna[i, j] = mean over 18 positions of <u[i, r, :], u[j, r, :]>
                      = expected fraction of positions where two samples agree.
        """
        # einsum: brc, src -> bsr   (sum over base axis c, keep position r)
        sim_dna = torch.einsum("brc,src->bsr", u, u).mean(dim=-1)   # [B, B]
        return F.mse_loss(sim_dna[mask], S[mask])

    def _loss_hash_hashnet(
        self, u: torch.Tensor, S: torch.Tensor, mask: torch.Tensor,
    ) -> torch.Tensor:
        """HashNet-style logistic-likelihood retrieval loss on continuous DNA code.

        Adaptation of the HashNet (Cao et al., ICCV 2017) pairwise loss to
        our continuous-code representation `u : continuous_code [B, 18, 4]`
        (softmax probs over A/C/G/T at each of 18 positions).

        Pipeline:
            sim_dna  ∈ [0, 1]   = mean_r <u[i,r,:], u[j,r,:]>
            score    ∈ [-α, α]  = α · (2 · sim_dna − 1)        # centre + scale
            S_bin    ∈ {0, 1}   = (label-sim > 0)              # any-shared = positive
            nll_ij              = softplus(-score) + (1 − S_bin) · score
                                  ≡  log(1 + exp(score)) − S_bin · score   # HashNet form
            loss                = class-balanced mean of nll over off-diagonal pairs.

        Compared to the MSE form (`_loss_hash`), positive pairs are not
        forced to be *identical* in code space -- only to score positive in
        the logistic sense (sigmoid(score) > 0.5). This removes the
        "same labels -> identical code" collapse that drives the 60% hash
        duplicate rate of the v6 baseline on Flickr25k. The continuation
        parameter α controls how sharply the score saturates; HashNet starts
        from α≈1 and increases over epochs. We default α=1.0 (static) for
        the first ablation; can be scheduled in `train_siglip2.py` later.
        """
        sim_dna = torch.einsum("brc,src->bsr", u, u).mean(dim=-1)   # [B, B] ∈ [0,1]
        alpha = max(self.hashnet_alpha, 1e-6)
        score = alpha * (2.0 * sim_dna - 1.0)                       # [B, B] ∈ [-α, α]

        # `S_target` is the value plugged into the HashNet logistic likelihood
        #     nll_ij = log(1+exp(score)) − S_target · score   (≡ softplus(-score) + (1-S_target)·score)
        # which makes `sigmoid(score)` learn toward `S_target`. Binary form
        # (v18 default) collapses all positive pairs onto a shared region;
        # Jaccard form (v19a) preserves per-label-combo granularity.
        if self.hashnet_use_jaccard:
            S_target = S.to(score.dtype)                            # fractional in [0,1]
            S_pos    = (S > 0).to(score.dtype)                      # for class-balance weighting only
        else:
            S_target = (S > 0).to(score.dtype)                      # binary
            S_pos    = S_target

        # v20a: cap the target similarity so same-powerset positive pairs do
        # not converge to identical codes. Leaves a residual degree of freedom
        # within each label-combo cluster -> distinct codes per cluster.
        if self.hashnet_S_cap < 1.0 - 1e-6:
            S_target = S_target.clamp(max=float(self.hashnet_S_cap))

        # Numerically-stable HashNet logistic likelihood (binary form derivation
        # generalises to fractional S_target via the same identity).
        nll = F.softplus(-score) + (1.0 - S_target) * score          # [B, B]

        # Off-diagonal-only with class-imbalance reweighting (HashNet's S/S0, S/S1).
        # We weight by the binary positive-mask even under Jaccard so that the
        # rebalancing semantics stay unchanged (you can have a fractional
        # target but still want to weight rare-positive pairs upward).
        mf = mask.to(score.dtype)
        n_pos = (mf * S_pos).sum().clamp_min(1.0)
        n_neg = (mf * (1.0 - S_pos)).sum().clamp_min(1.0)
        total = n_pos + n_neg
        w = torch.where(S_pos > 0, total / n_pos, total / n_neg)     # [B, B]
        return (nll * w * mf).sum() / total

    def _loss_hash_hard(
        self, u_st: torch.Tensor, S: torch.Tensor, mask: torch.Tensor,
    ) -> torch.Tensor:
        """Hard DNA retrieval loss via STE.

        u_st : dna_hash_code_st [B, 18, 4]
            Forward = hard one-hot, backward = continuous gradient (Gumbel-Softmax
            STE or deterministic STE — see ``model_siglip2.CodonHead``).
            This makes the *retrieval-time* hard distribution directly trainable
            and closes the train-test gap that ``loss_hash`` (continuous-only)
            would leave open.
        """
        sim_hard = torch.einsum("brc,src->bsr", u_st, u_st).mean(dim=-1)   # [B, B]
        return F.mse_loss(sim_hard[mask], S[mask])

    def _loss_ntxent_dna(
        self,
        u_st_view1: torch.Tensor,
        u_st_view2: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        """SimCLR / CIBHash-style NtXent contrastive loss on DNA codes.

        u_st_view1, u_st_view2 : [B, 18, 4]   (STE one-hot per position)
            Two augmented views of the same B images. The forward path is
            hard one-hot; gradient flows through STE back to the codebooks
            and encoder.

        Similarity:
            sim_dna(u_i, u_j) = mean over 18 positions of <u_i[r,:], u_j[r,:]>
                              = expected fraction of agreeing positions
                              ∈ [0, 1]
            (Same definition `_loss_hash_*` already use, so the geometry
             is consistent with the rest of the loss.)

        NtXent:
            z = [u_view1; u_view2]                                  # [2B, 18, 4]
            sim_matrix[i, j] = sim_dna(z[i], z[j])                  # [2B, 2B]
            logits = sim_matrix / temperature
            For row i, positive = (i + B) mod 2B; mask out self;
            softmax CE → encourages positive to dominate the row.

        Per-image positive guarantee replaces v27b's batch top-k cosine
        pseudo-positive (noisy, drifts batch-to-batch).
        """
        B = u_st_view1.shape[0]
        z = torch.cat([u_st_view1, u_st_view2], dim=0)                  # [2B, 18, 4]
        sim = torch.einsum("brc,src->bsr", z, z).mean(dim=-1)            # [2B, 2B]
        sim = sim / max(float(temperature), 1e-6)

        # Mask out the diagonal (self-similarity).
        N = 2 * B
        eye = torch.eye(N, device=sim.device, dtype=torch.bool)
        sim = sim.masked_fill(eye, -1e9)

        # Positive index: row i in [0, B) pairs with i+B; row i in [B, 2B)
        # pairs with i-B. Cross-entropy target = positive index.
        pos_idx = torch.cat([
            torch.arange(B, 2 * B, device=sim.device),
            torch.arange(0, B,     device=sim.device),
        ])                                                              # [2B]
        return F.cross_entropy(sim, pos_idx)

    def _loss_ntxent_dna_per_codebook(
        self,
        u_st_view1: torch.Tensor,
        u_st_view2: torch.Tensor,
        temperature: float,
        num_codebooks: int = 6,
        text_part_raw: Optional[torch.Tensor] = None,
        dynamic_tau_alpha: float = 0.0,
        skip_global_dyn: bool = False,
        global_use_local_mean: bool = False,
    ) -> torch.Tensor:
        """Per-codebook NtXent (v31b) with optional v42 dynamic tau.

        Splits the [B, 18, 4] DNA code into 6 codebook groups
        [B, 6, 3, 4], runs an independent NtXent on each codebook's
        3-codon block, and averages the 6 losses.

        Forces each codebook to *independently* discriminate samples ->
        compositional-independence test. Cross-codebook redundancy is
        penalized (each codebook gets its own gradient signal).

        v42 (Q2): when ``text_part_raw`` is provided and
        ``dynamic_tau_alpha > 0``, each codebook m uses a per-pair
        temperature  tau_ij = T * (1 + alpha * cos(t_i^(m), t_j^(m)))
        so semantically-similar samples get soft push (large tau) and
        semantically-distant ones get hard push (small tau). Targets
        the uniformity-tolerance dilemma of Wang et al. CVPR 2021.

        Args:
            text_part_raw: [B, num_codebooks, D] raw SigLIP2 text-encoder
                per-slot pooled embeddings (i.e. `cached_text_part_raw`
                / `outputs['text_global_feat']`). None disables dynamic
                tau even when alpha > 0.
            dynamic_tau_alpha: alpha in [0, 1). 0 disables. 0.5 maps to
                tau in [0.5*base, 1.5*base].
        """
        B, R, C = u_st_view1.shape
        assert R % num_codebooks == 0, (
            f"[ntxent_per_codebook] DNA length {R} not divisible by "
            f"num_codebooks={num_codebooks}"
        )
        chunk = R // num_codebooks                              # 3
        u1 = u_st_view1.view(B, num_codebooks, chunk, C)         # [B, 6, 3, 4]
        u2 = u_st_view2.view(B, num_codebooks, chunk, C)
        N = 2 * B
        T = max(float(temperature), 1e-6)
        eye = torch.eye(N, device=u_st_view1.device, dtype=torch.bool)
        pos_idx = torch.cat([
            torch.arange(B, 2 * B, device=u_st_view1.device),
            torch.arange(0, B,     device=u_st_view1.device),
        ])

        # v42 dynamic-tau setup
        use_dyn = (
            text_part_raw is not None
            and float(dynamic_tau_alpha) > 0.0
            and text_part_raw.shape[0] == B
            and text_part_raw.shape[1] == num_codebooks
        )
        alpha = float(dynamic_tau_alpha)
        # Clamp alpha < 1 so tau_ij stays strictly positive for cos in [-1, 1].
        alpha = min(alpha, 0.999)
        # Floor for tau just in case of numerical noise.
        tau_floor = 1e-4

        loss_sum = u_st_view1.new_zeros(())
        for m in range(num_codebooks):
            z = torch.cat([u1[:, m], u2[:, m]], dim=0)            # [2B, 3, 4]
            sim = torch.einsum("brc,src->bsr", z, z).mean(dim=-1)  # [2B, 2B]
            # v60: skip dynamic-tau on m=0 if requested
            use_dyn_this_m = use_dyn and not (m == 0 and skip_global_dyn)
            if use_dyn_this_m:
                # v61: for m=0, optionally use the mean of the 5 local text
                # features as the similarity source instead of the C_global
                # caption embedding.
                if m == 0 and global_use_local_mean and text_part_raw.shape[1] >= 2:
                    t_m = text_part_raw[:, 1:, :].mean(dim=1)     # [B, D] = mean over 5 local slots
                else:
                    t_m = text_part_raw[:, m, :]                  # [B, D]
                t_all = torch.cat([t_m, t_m], dim=0)              # [2B, D]
                t_n   = F.normalize(t_all, dim=-1)
                cos_t = t_n @ t_n.t()                             # [2B, 2B]
                T_ij  = T * (1.0 + alpha * cos_t)                 # [2B, 2B]
                T_ij  = T_ij.clamp(min=tau_floor)
                sim   = (sim / T_ij).masked_fill(eye, -1e9)
            else:
                sim   = (sim / T).masked_fill(eye, -1e9)
            loss_sum = loss_sum + F.cross_entropy(sim, pos_idx)
        return loss_sum / num_codebooks

    def _loss_ortho_text(self, text_part_tokens: torch.Tensor) -> torch.Tensor:
        """v44 (B1): cross-slot text orthogonality regularizer.

        text_part_tokens : [B, M, D]  post-adapter per-slot text features.

        For each sample b:
            T_n     = normalize(text_part_tokens[b], dim=-1)   # [M, D]
            G_b     = T_n @ T_n.T                              # [M, M]
            L_b     = ((G_b - I)**2).sum() / (M * (M - 1))
        and average over batch. Note ((G_b - I)**2) has 0 on diag for
        unit-norm vectors so the (M*(M-1)) normalizer counts only the
        M(M-1) off-diagonal pair contributions.

        Targets the diagnosed SigLIP2 cross-slot uniformity (V1 cos~0.977,
        V3 cos~0.967) by pushing the *adapted* text features (which we
        control via the trainable text_adapter) much further apart.
        Pairs especially well with --per_slot_text_adapter (6 independent
        MLPs) since each slot's adapter has the freedom to rotate its
        output independently.
        """
        if text_part_tokens is None:
            return torch.zeros((), device=self.eps if isinstance(self.eps, torch.Tensor) else None)
        B, M, _ = text_part_tokens.shape
        if M < 2:
            return text_part_tokens.new_zeros(())
        T_n = F.normalize(text_part_tokens, dim=-1)              # [B, M, D]
        G   = torch.einsum("bmd,bnd->bmn", T_n, T_n)             # [B, M, M]
        I   = torch.eye(M, device=G.device, dtype=G.dtype)       # [M, M]
        diff = G - I                                              # [B, M, M]
        # ((diff)^2) -- diag should be ~0 for unit vectors; small numerical
        # noise; off-diag is the cross-slot cosine which we want to push to 0.
        per_sample = (diff * diff).sum(dim=(1, 2)) / float(M * (M - 1))
        return per_sample.mean()

    def _loss_recon(
        self,
        recon: torch.Tensor,
        target: torch.Tensor,
        target_kind: str,
    ) -> torch.Tensor:
        """Reconstruction loss for the v28 decoder ablation.

        target_kind = 'pixel'        : MSE on ImageNet-normalized pixels.
        target_kind = 'siglip_feat'  : 1 - cosine(recon, target). Bounded
                                       in [0, 2]. Matches the cos geometry
                                       used elsewhere for SigLIP2 features.
        """
        if target_kind == "pixel":
            return F.mse_loss(recon, target)
        if target_kind == "siglip_feat":
            r = F.normalize(recon,  dim=-1)
            t = F.normalize(target.detach(), dim=-1)
            return (1.0 - (r * t).sum(dim=-1)).mean()
        raise ValueError(
            f"[loss_recon] unknown decoder_target {target_kind!r}; "
            f"expected 'pixel' or 'siglip_feat'."
        )

    def _loss_quant(
        self, continuous_code: torch.Tensor, dna_hash_code_hard: torch.Tensor,
    ) -> torch.Tensor:
        """Codon-side commitment loss (codon-level VQ).

        Pulls the continuous codon distribution toward the hard one-hot it
        currently selects. Mirror of ``loss_vq`` but in the base-probability
        space ``[B, 18, 4]`` rather than the codeword space ``[B, 6, D]``.

        continuous_code     : [B, 18, 4]   softmax probs (gradient-bearing)
        dna_hash_code_hard  : [B, 18, 4]   one-hot of argmax (no gradient via .detach)
        """
        return F.mse_loss(continuous_code, dna_hash_code_hard.detach())

    def _loss_vq(self, z: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """VQ-VAE style codebook + commitment.

        z : semantic_visual_tokens   [B, 6, D]   pre-quantization
        q : quantized_tokens_raw     [B, 6, D]   selected codeword
        """
        if z.shape != q.shape:
            raise ValueError(
                f"[loss_vq] shape mismatch: semantic_visual_tokens {tuple(z.shape)} "
                f"vs quantized_tokens_raw {tuple(q.shape)}"
            )
        codebook_loss   = F.mse_loss(q, z.detach())
        commitment_loss = F.mse_loss(z, q.detach())
        return codebook_loss + self.beta_vq * commitment_loss

    def _loss_anchor(
        self,
        text_part_tokens: Optional[torch.Tensor],
        local_codebook_mean_anchors: torch.Tensor,
        ref: torch.Tensor,
    ) -> torch.Tensor:
        """Anchor alignment between training text anchors and codebook anchors.

        text_part_tokens             : [B, 6, D] or None
        local_codebook_mean_anchors  : [5, D]
        ref                          : a tensor used solely to take dtype/device
                                       (so the zero-fallback matches u/q).

        Stable target via EMA:
            batch_anchor[b]  = mean over batch of text_part_tokens[:, 1+m, :]
            ema_anchor[m]    = m · ema_anchor + (1−m) · batch_anchor
        codebook anchors are pulled toward ``ema_anchor`` (detached) so they
        follow a smooth dataset-wide target rather than batch-noisy means.
        """
        if text_part_tokens is None:
            return ref.new_zeros(())
        local_text = text_part_tokens[:, 1:, :]                                  # [B, 5, D]
        batch_anchor = F.normalize(local_text.mean(dim=0), dim=-1)               # [5, D]
        ema_anchor   = self._update_ema_text_anchor(batch_anchor).detach()       # [5, D]
        ema_anchor   = F.normalize(ema_anchor, dim=-1)                           # [5, D]
        cb_anchor    = F.normalize(local_codebook_mean_anchors, dim=-1)          # [5, D]
        return (1.0 - (ema_anchor * cb_anchor).sum(dim=-1)).mean()

    def _loss_dna(
        self, u: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Entropy + base-balance regularizer.

        u : continuous_code [B, 18, 4]
        """
        # 10-1 entropy: encourage each position to commit to one base
        entropy = -(u * (u + self.eps).log()).sum(dim=-1)        # [B, 18]
        loss_entropy = entropy.mean()
        # 10-2 base balance: prevent collapse to a single base across the dataset
        base_usage = u.mean(dim=0)                                # [18, 4]
        uniform = torch.full_like(base_usage, 0.25)
        loss_base_balance = F.mse_loss(base_usage, uniform)
        loss = loss_entropy + self.eta_base_balance * loss_base_balance
        return {
            "loss_dna":          loss,
            "loss_entropy":      loss_entropy,
            "loss_base_balance": loss_base_balance,
        }

    def _loss_bu(
        self, distances: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Codebook balance + assignment uncorrelated.

        distances : codebook_distances [B, M=6, K]
        """
        B, M, K = distances.shape
        tau = max(self.tau_codebook_assignment, 1e-6)
        P = F.softmax(-distances / tau, dim=-1)                   # [B, M, K]

        # 11-1 usage balance per codebook: each codeword used ~ uniformly
        usage = P.mean(dim=0)                                     # [M, K]
        uniform = torch.full_like(usage, 1.0 / K)
        loss_cb_balance = F.mse_loss(usage, uniform)

        # 11-2 assignment uncorrelated: off-diagonal Gram penalty per codebook
        # G[m, i, j] = (K / B) * Σ_b P[b, m, i] * P[b, m, j]
        G = (K / float(B)) * torch.einsum("bmk,bml->mkl", P, P)   # [M, K, K]
        eye = torch.eye(K, device=distances.device, dtype=distances.dtype).unsqueeze(0)  # [1, K, K]
        offdiag = G * (1.0 - eye)
        loss_cb_uncorr = (offdiag ** 2).mean()

        loss = loss_cb_balance + self.rho_cb_uncorr * loss_cb_uncorr
        return {
            "loss_bu":         loss,
            "loss_cb_balance": loss_cb_balance,
            "loss_cb_uncorr":  loss_cb_uncorr,
        }

    # ============================================== forward

    def forward(
        self,
        outputs: Dict[str, Any],
        labels: Optional[torch.Tensor] = None,
        multi_hot_labels: Optional[torch.Tensor] = None,
        epoch: Optional[int] = None,
        pixel_target: Optional[torch.Tensor] = None,
        outputs_view2: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        # ---- pull required tensors from the model output dict ------------
        u                            = outputs.get("continuous_code")          # [B, 18, 4]
        u_st                         = outputs.get("dna_hash_code_st")          # [B, 18, 4] | None
        u_hard                       = outputs.get("dna_hash_code_hard")        # [B, 18, 4] | None
        z                            = outputs.get("semantic_visual_tokens")    # [B, 6, D]
        q                            = outputs.get("quantized_tokens_raw")      # [B, 6, D]
        distances                    = outputs.get("codebook_distances")        # [B, 6, K]
        text_part_tokens             = outputs.get("text_part_tokens")          # [B, 6, D] | None
        local_codebook_mean_anchors  = outputs.get("local_codebook_mean_anchors")  # [5, D]

        if any(t is None for t in (u, z, q, distances)):
            raise ValueError(
                "[DNACodonHashLoss] outputs must contain `continuous_code`, "
                "`semantic_visual_tokens`, `quantized_tokens_raw`, `codebook_distances`. "
                "Did you call model(..., return_routing=True)?"
            )
        if local_codebook_mean_anchors is None:
            raise ValueError(
                "[DNACodonHashLoss] outputs must contain `local_codebook_mean_anchors` [5, D]."
            )

        device = u.device
        B = u.shape[0]
        if self.hash_target_mode in ("siglip_cos", "siglip_cos_topk"):
            vg = outputs.get("visual_global_feat")
            if vg is None:
                raise ValueError(
                    "[DNACodonHashLoss] hash_target_mode='siglip_cos*' requires "
                    "outputs['visual_global_feat']. Either run with a SigLIP2 "
                    "feature cache that includes visual_global or feed live "
                    "pixel_values so the model populates it."
                )
            if self.hash_target_mode == "siglip_cos_topk":
                S = build_siglip_cos_topk_similarity(vg, pos_rate=self.siglip_cos_pos_rate).to(device)
            else:
                S = build_siglip_cos_similarity(vg).to(device)
        else:
            S = build_label_similarity(labels=labels, multi_hot_labels=multi_hot_labels).to(device)
        mask = get_off_diagonal_mask(B, device)

        # ---- individual losses -------------------------------------------
        # Dispatch hash loss form: MSE (legacy) or HashNet logistic.
        if self.lambda_hash_type == "hashnet":
            loss_hash = self._loss_hash_hashnet(u, S, mask)
        else:
            loss_hash = self._loss_hash(u, S, mask)
        # hard-path retrieval loss via STE (None-safe).
        # NOTE: We deliberately keep `loss_hash_hard` on the MSE-Jaccard form
        # even when `lambda_hash_type=hashnet`. The v25 ablation showed that
        # the mixed combo (logistic soft / MSE-Jaccard hard) is what actually
        # produces v23b's unique-code-ratio 0.111: the Jaccard target on the
        # hard path injects a graduated per-image target that prevents same-
        # label clusters from collapsing onto a single hash. Switching the
        # hard path to hashnet form (v25a: −0.027 mAP, unique 0.044) or
        # disabling it (v25b: −0.025 mAP, unique 0.051) both regress.
        if u_st is not None:
            loss_hash_hard = self._loss_hash_hard(u_st, S, mask)
        else:
            loss_hash_hard = u.new_zeros(())
        # codon-side quantization commitment (None-safe)
        if u_hard is not None:
            loss_quant = self._loss_quant(u, u_hard)
        else:
            loss_quant = u.new_zeros(())
        loss_vq     = self._loss_vq(z, q)
        loss_anchor = self._loss_anchor(
            text_part_tokens, local_codebook_mean_anchors, ref=u,
        )
        dna_components = self._loss_dna(u)
        bu_components  = self._loss_bu(distances)
        # Wasserstein alignment: per-sample <pi, cost> from the router (None-safe).
        ot = outputs.get("ot_cost")
        if ot is not None:
            loss_wasserstein = ot.mean()
        else:
            loss_wasserstein = u.new_zeros(())

        # v29 paired-aug NtXent on DNA codes (None-safe). Requires the
        # trainer to forward the model on a second augmented view per image
        # and pass that output dict via `outputs_view2`. Both view dicts
        # must contain `dna_hash_code_st` of shape [B, 18, 4].
        if (outputs_view2 is not None) and (self.lambda_ntxent > 0.0):
            u_st_v1 = outputs.get("dna_hash_code_st")
            u_st_v2 = outputs_view2.get("dna_hash_code_st")
            if u_st_v1 is None or u_st_v2 is None:
                raise ValueError(
                    "[DNACodonHashLoss] paired-aug NtXent requires both "
                    "outputs and outputs_view2 to contain dna_hash_code_st."
                )
            if self.ntxent_mode == "per_codebook":
                # v42 (Q2): pull raw SigLIP2 text per-slot embedding from
                # the model outputs (stored under 'text_global_feat' --
                # confusingly named but it is the raw cached_text_part_raw
                # tensor of shape [B, M, D_proj], see model_siglip2.py
                # build_outputs return dict).
                _text_anchors = outputs.get("text_global_feat") \
                    if self.ntxent_dynamic_tau else None
                loss_ntxent = self._loss_ntxent_dna_per_codebook(
                    u_st_v1, u_st_v2, temperature=self.ntxent_temperature,
                    text_part_raw=_text_anchors,
                    dynamic_tau_alpha=(
                        self.ntxent_dynamic_tau_alpha
                        if self.ntxent_dynamic_tau else 0.0
                    ),
                    skip_global_dyn=self.ntxent_dynamic_tau_skip_global,
                    global_use_local_mean=self.ntxent_global_use_local_mean,
                )
            else:
                loss_ntxent = self._loss_ntxent_dna(
                    u_st_v1, u_st_v2, temperature=self.ntxent_temperature,
                )
        else:
            loss_ntxent = u.new_zeros(())

        # v28 reconstruction loss (None-safe). Active only when the model
        # was built with --use_decoder; otherwise outputs['reconstruction']
        # is None and we skip the term entirely.
        recon = outputs.get("reconstruction")
        if recon is not None and self.lambda_recon > 0.0:
            tk = outputs.get("reconstruction_target", self.decoder_target)
            if tk == "siglip_feat":
                target = outputs.get("visual_global_feat")
                if target is None:
                    raise ValueError(
                        "[DNACodonHashLoss] decoder_target='siglip_feat' but "
                        "outputs['visual_global_feat'] is None."
                    )
            elif tk == "pixel":
                if pixel_target is None:
                    raise ValueError(
                        "[DNACodonHashLoss] decoder_target='pixel' but "
                        "pixel_target was not passed to forward()."
                    )
                target = pixel_target
            else:
                raise ValueError(f"[loss] unknown decoder_target {tk!r}")
            loss_recon = self._loss_recon(recon, target, tk)
        else:
            loss_recon = u.new_zeros(())

        loss_dna = dna_components["loss_dna"]
        loss_bu  = bu_components ["loss_bu"]

        # ---- BU warm-up --------------------------------------------------
        if (epoch is not None) and (epoch < self.bu_warmup_epochs):
            eff_lambda_bu = 0.0
        else:
            eff_lambda_bu = self.lambda_bu

        # v44 (B1): cross-slot text orthogonality regularizer. Active only
        # when text_part_tokens is supplied (text path on) AND lambda > 0.
        if self.lambda_ortho_text > 0.0 and text_part_tokens is not None:
            loss_ortho_text = self._loss_ortho_text(text_part_tokens)
        else:
            loss_ortho_text = u.new_zeros(())

        # v66: per-codon text-anchored aux CE loss (model computes per forward)
        loss_text_anchor = outputs.get("loss_text_anchor")
        if loss_text_anchor is None:
            loss_text_anchor = u.new_zeros(())

        # ---- total -------------------------------------------------------
        total = (
            self.lambda_hash       * loss_hash
            + self.lambda_hash_hard  * loss_hash_hard
            + self.lambda_vq         * loss_vq
            + self.lambda_quant      * loss_quant
            + self.lambda_anchor     * loss_anchor
            + self.lambda_dna        * loss_dna
            + eff_lambda_bu          * loss_bu
            + self.lambda_wasserstein * loss_wasserstein
            + self.lambda_recon       * loss_recon
            + self.lambda_ntxent      * loss_ntxent
            + self.lambda_ortho_text  * loss_ortho_text
            + self.lambda_codon_text_anchor * loss_text_anchor
        )

        return {
            "loss":              total,
            "loss_hash":         loss_hash,
            "loss_hash_hard":    loss_hash_hard,
            "loss_vq":           loss_vq,
            "loss_quant":        loss_quant,
            "loss_anchor":       loss_anchor,
            "loss_wasserstein":  loss_wasserstein,
            "loss_recon":        loss_recon,
            "loss_ntxent":       loss_ntxent,
            "loss_ortho_text":   loss_ortho_text,
            "loss_dna":          loss_dna,
            "loss_bu":           loss_bu,
            "loss_entropy":      dna_components["loss_entropy"],
            "loss_base_balance": dna_components["loss_base_balance"],
            "loss_cb_balance":   bu_components ["loss_cb_balance"],
            "loss_cb_uncorr":    bu_components ["loss_cb_uncorr"],
            "loss_codon_text_anchor": loss_text_anchor,
        }
