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

import math
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
        # v119: CIBHash-style per-codebook NtXent + symmetric Bernoulli KL.
        self.lambda_cibhash_ntxent = float(getattr(cfg, "lambda_cibhash_ntxent", 0.0))
        self.lambda_cibhash_kl     = float(getattr(cfg, "lambda_cibhash_kl",     0.0))
        self.cibhash_temperature   = float(getattr(cfg, "cibhash_temperature",   0.3))
        # v120: CIBHash extension knobs (mode + text-cos dynamic tau).
        self.cibhash_mode               = str(getattr(cfg, "cibhash_mode", "per_codebook"))
        self.cibhash_dynamic_tau        = bool(getattr(cfg, "cibhash_dynamic_tau", False))
        # v121: SwAV-style swapped balanced codeword-assignment loss.
        self.lambda_swav_assign         = float(getattr(cfg, "lambda_swav_assign", 0.0))
        self.swav_assign_tau            = float(getattr(cfg, "swav_assign_tau",    0.1))
        self.swav_sinkhorn_eps          = float(getattr(cfg, "swav_sinkhorn_eps",  0.05))
        self.swav_sinkhorn_iters        = int(getattr(cfg, "swav_sinkhorn_iters",  3))
        self.swav_assign_include_global = bool(getattr(cfg, "swav_assign_include_global", False))
        self.cibhash_dynamic_tau_alpha  = float(getattr(cfg, "cibhash_dynamic_tau_alpha", 0.0))
        # v138: prototype-cluster paired-view InfoNCE on codebook_distances.
        self.lambda_proto_cluster        = float(getattr(cfg, "lambda_proto_cluster",        0.0))
        self.proto_cluster_temperature   = float(getattr(cfg, "proto_cluster_temperature",   0.3))
        # v91 (text-to-DNA-hash matching): MSE between image-derived
        # continuous_code and text-derived continuous_code (latter produced
        # by the model's parallel text path through the shared quantizer
        # and codon_heads). 0 = disabled (legacy bit-exact).
        self.lambda_text_hash   = float(getattr(cfg, "lambda_text_hash", 0.0))
        # v97: swap text_hash MSE → symmetric NtXent (text DNA as augmented view)
        self.text_hash_use_ntxent     = bool(getattr(cfg, "text_hash_use_ntxent", False))
        self.text_hash_ntxent_temperature = float(getattr(cfg, "text_hash_ntxent_temperature", 0.07))
        # v100: ADDITIVE text-DNA NtXent (extra term on top of MSE / swap form)
        self.lambda_text_hash_ntxent = float(getattr(cfg, "lambda_text_hash_ntxent", 0.0))
        # v128: granularity mode for the ADDITIVE text-DNA NtXent.
        # See `--text_hash_ntxent_mode` config; "global" preserves legacy
        # v100-v125d behaviour (full flattened DNA), "per_codebook" runs
        # M independent symmetric InfoNCEs on per-codebook (L*4)-dim DNA
        # segments.
        self.text_hash_ntxent_mode = str(getattr(cfg, "text_hash_ntxent_mode", "global"))
        # v93: per-codebook cross-modal codeword InfoNCE. Symmetric InfoNCE
        # between visual `quantized_tokens[:, m, :]` and text
        # `text_quantized_tokens[:, m, :]` for each codebook m.
        # Positive: same sample; negatives: other samples in the batch.
        # 0 = disabled (legacy bit-exact).
        self.lambda_cw_xmodal      = float(getattr(cfg, "lambda_cw_xmodal", 0.0))
        self.cw_xmodal_temperature = float(getattr(cfg, "cw_xmodal_temperature", 0.07))
        # v123: per-(codebook, codeword) text prototypes. The EMA prototype
        # is updated from text tokens assigned to each visual codeword, then
        # visual quantizer inputs are classified against those prototypes.
        self.lambda_codeword_text_proto = float(getattr(cfg, "lambda_codeword_text_proto", 0.0))
        self.codeword_text_proto_tau = float(getattr(cfg, "codeword_text_proto_tau", 0.1))
        self.codeword_text_proto_momentum = float(getattr(cfg, "codeword_text_proto_momentum", 0.95))
        self.codeword_text_proto_min_count = int(getattr(cfg, "codeword_text_proto_min_count", 4))
        self.codeword_text_proto_include_global = bool(
            getattr(cfg, "codeword_text_proto_include_global", False)
        )
        self.register_buffer("_codeword_text_proto", torch.empty(0), persistent=True)
        self.register_buffer("_codeword_text_seen", torch.empty(0), persistent=True)
        # v106: codeword <-> DNA codon bijection losses. Operate on the
        # codon decoder applied to the codebook codewords directly
        # (no residual, no sample dependence). Three variants implemented:
        #   - Sinkhorn-OT bijection (recommended): hard marginal constraints
        #     enforce K codewords -> K distinct codons mapping when K = 64.
        #   - Aggregated entropy: uniform aggregate codon usage + per-codeword
        #     sharpness. Cheaper but weaker (local-min vulnerable).
        #   - Pairwise distinctness: sum of off-diagonal inner products of
        #     codeword codon distributions. Even cheaper, even weaker.
        # All zero by default. v103a recipe + lambda_codeword_codon_sinkhorn
        # > 0 launches the v106a experiment.
        self.lambda_codeword_codon_sinkhorn  = float(getattr(cfg, "lambda_codeword_codon_sinkhorn",  0.0))
        self.lambda_codeword_codon_agg_ent   = float(getattr(cfg, "lambda_codeword_codon_agg_ent",   0.0))
        self.lambda_codeword_codon_pairwise  = float(getattr(cfg, "lambda_codeword_codon_pairwise",  0.0))
        self.codeword_codon_sinkhorn_warmup_epochs = int(getattr(cfg, "codeword_codon_sinkhorn_warmup_epochs", 0))
        self.codeword_codon_sinkhorn_eps     = float(getattr(cfg, "codeword_codon_sinkhorn_eps",     0.1))
        self.codeword_codon_sinkhorn_iters   = int  (getattr(cfg, "codeword_codon_sinkhorn_iters",   30))
        self.codeword_codon_agg_ent_alpha    = float(getattr(cfg, "codeword_codon_agg_ent_alpha",    0.5))
        # v112b: text-semantic cluster -> DNA-codon OT allocation. The
        # prototypes/statistics are EMA buffers, not optimizer parameters,
        # matching the existing EMA-codebook style.
        self.lambda_text_cluster_codon_ot = float(getattr(cfg, "lambda_text_cluster_codon_ot", 0.0))
        self.text_cluster_count           = int  (getattr(cfg, "text_cluster_count", 8))
        self.text_cluster_temperature     = float(getattr(cfg, "text_cluster_temperature", 0.1))
        self.text_cluster_ema_momentum    = float(getattr(cfg, "text_cluster_ema_momentum", 0.95))
        self.text_cluster_conf_gamma      = float(getattr(cfg, "text_cluster_conf_gamma", 1.0))
        self.text_cluster_codon_ot_eps    = float(getattr(cfg, "text_cluster_codon_ot_eps", 0.1))
        self.text_cluster_codon_ot_iters  = int  (getattr(cfg, "text_cluster_codon_ot_iters", 30))
        self.register_buffer("_text_cluster_prototypes", torch.empty(0), persistent=True)
        self.register_buffer("_text_cluster_codeword", torch.empty(0), persistent=True)
        # v113: weak, rank-based text-codon relational tendency. Unlike
        # v112b, this does not allocate clusters to codons with OT. It only
        # asks text-neighbor pairs to have higher codon-distribution
        # similarity than text-distant pairs within each codebook.
        self.lambda_text_codon_rel       = float(getattr(cfg, "lambda_text_codon_rel", 0.0))
        self.text_codon_rel_top_frac     = float(getattr(cfg, "text_codon_rel_top_frac", 0.10))
        self.text_codon_rel_bottom_frac  = float(getattr(cfg, "text_codon_rel_bottom_frac", 0.30))
        self.text_codon_rel_margin       = float(getattr(cfg, "text_codon_rel_margin", 0.05))
        self.text_codon_rel_min_pairs    = int  (getattr(cfg, "text_codon_rel_min_pairs", 8))
        # v107: prototype cosine clustering (InfoNCE between z and codewords).
        # Pulls z[b, m] toward its assigned codeword and pushes away from
        # other codewords in cosine space. Operates on the EMA codebook
        # buffer + the (B, M) routing assignment.
        self.lambda_proto_cluster_cos = float(getattr(cfg, "lambda_proto_cluster_cos", 0.0))
        self.proto_cluster_cos_tau    = float(getattr(cfg, "proto_cluster_cos_tau",    0.1))
        # v112 (Hierarchical Codon Decomposition): force first base of each
        # codon to encode text-similarity cluster identity of the assigned
        # codeword. Remaining 2 bases encode within-cluster variation.
        self.lambda_hierarchical_cluster_codon = float(getattr(cfg, "lambda_hierarchical_cluster_codon", 0.0))
        self.hierarchical_cluster_n_clusters   = int  (getattr(cfg, "hierarchical_cluster_n_clusters",   4))
        self.hierarchical_cluster_refresh_every = int (getattr(cfg, "hierarchical_cluster_refresh_every", 5))
        self.hierarchical_cluster_warmup_epochs = int (getattr(cfg, "hierarchical_cluster_warmup_epochs", 5))
        # Cluster labels buffer [M=6, K_max=64] long. Refreshed via
        # `refresh_clusters()` called from the training loop.
        _M = int(getattr(cfg, "num_codebooks", 6))
        _K = int(getattr(cfg, "codebook_size", 64))
        self.register_buffer(
            "hierarchical_cluster_labels",
            torch.zeros(_M, _K, dtype=torch.long),
            persistent=True,
        )
        # Flag tracking whether clusters have been initialized at least once.
        self.register_buffer(
            "hierarchical_cluster_initialized",
            torch.zeros((), dtype=torch.bool),
            persistent=True,
        )
        # v66: per-codon text-anchored aux CE loss weight. The model computes
        # `out["loss_text_anchor"]` per forward (sum across 6 codebooks). 0
        # default keeps the loss off.
        self.lambda_codon_text_anchor = float(getattr(cfg, "lambda_codon_text_anchor", 0.0))
        # v70a / v72a (Exp 3, 6)
        self.lambda_hash_recon      = float(getattr(cfg, "lambda_hash_recon",     0.0))
        self.hash_recon_target_kind = str  (getattr(cfg, "hash_recon_target",     "siglip_visual"))
        self.lambda_dual_semantic   = float(getattr(cfg, "lambda_dual_semantic",  0.0))
        self.lambda_dual_instance   = float(getattr(cfg, "lambda_dual_instance",  0.0))
        self.dual_hash_proj_target_kind = str(getattr(cfg, "dual_hash_proj_target", "siglip_visual"))
        self.dual_hash_proj_ntxent_tau  = float(getattr(cfg, "dual_hash_proj_ntxent_tau", 0.5))
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
        # v67: dynamic-tau variant selector. "text_cos" (default) = legacy v42
        # per-pair tau = base * (1 + alpha * cos(text_i, text_j)).
        # "neg_only_norm_model" = (i) positive pair uses static base_tau,
        # (ii) negative pair uses base_tau * model_scale_m * semantic_scale_ij,
        # where semantic_scale uses batch-normalized + tanh-clipped text
        # affinity, and model_scale_m is a MACL-style per-codebook factor
        # adaptive to current positive alignment A_m. See `_loss_ntxent_dna_per_codebook`.
        self.ntxent_dynamic_tau_variant = str(getattr(cfg, "ntxent_dynamic_tau_variant", "text_cos"))
        self.ntxent_dynamic_tau_model_beta      = float(getattr(cfg, "ntxent_dynamic_tau_model_beta",      0.5))
        self.ntxent_dynamic_tau_model_a0        = float(getattr(cfg, "ntxent_dynamic_tau_model_a0",        0.6))
        self.ntxent_dynamic_tau_model_scale_min = float(getattr(cfg, "ntxent_dynamic_tau_model_scale_min", 0.75))
        self.ntxent_dynamic_tau_model_scale_max = float(getattr(cfg, "ntxent_dynamic_tau_model_scale_max", 1.25))
        self.ntxent_dynamic_tau_semantic_scale_min = float(getattr(cfg, "ntxent_dynamic_tau_semantic_scale_min", 0.7))
        self.ntxent_dynamic_tau_semantic_scale_max = float(getattr(cfg, "ntxent_dynamic_tau_semantic_scale_max", 1.3))
        # v87 (Wang & Liu, CVPR 2021): hard-negative sampling for per-codebook
        # NtXent. alpha = fraction of negatives kept in softmax denominator;
        # alpha=1.0 disables (legacy, bit-exact).
        self.ntxent_hard_neg_alpha = float(getattr(cfg, "ntxent_hard_neg_alpha", 1.0))
        # v88 (Huang et al., ICML 2023; MACL Algorithm 1 adapted per-codebook):
        # model-aware tau scaling driven by paired-augmentation positive
        # alignment of semantic_visual_tokens. Composes multiplicatively
        # with v42 text_cos dyn-tau. alpha=0 disables (legacy bit-exact).
        self.ntxent_macl_alpha = float(getattr(cfg, "ntxent_macl_alpha", 0.0))
        self.ntxent_macl_a0    = float(getattr(cfg, "ntxent_macl_a0",    0.0))
        # v73 (Exp 7): global DNA NtXent auxiliary loss alongside per-codebook
        self.lambda_global_dna_ntxent = float(getattr(cfg, "lambda_global_dna_ntxent", 0.0))
        # v76b: cosine VQ loss (replaces MSE in _loss_vq with (1 - cos)).
        # Only meaningful when --vq_distance_mode=cosine.
        self.vq_loss_cosine = bool(getattr(cfg, "vq_loss_cosine", False))
        # v79a: cross-codebook orthogonality loss
        self.lambda_codebook_ortho = float(getattr(cfg, "lambda_codebook_ortho", 0.0))
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
        variant: str = "text_cos",
        model_beta: float = 0.5,
        model_a0: float = 0.6,
        model_scale_min: float = 0.75,
        model_scale_max: float = 1.25,
        semantic_scale_min: float = 0.7,
        semantic_scale_max: float = 1.3,
        hard_neg_alpha: float = 1.0,
        paired_align_signal: Optional[torch.Tensor] = None,   # v88: [B, M] paired-aug cosine of semantic_v
        macl_alpha: float = 0.0,                              # v88: alpha (0 disables MACL scaling)
        macl_a0: float = 0.0,                                 # v88: A_0 baseline
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

        # v67: pre-compute positive-pair location mask for the new variant.
        # pos_idx[i] gives the positive partner of row i; build [N, N] bool.
        if variant == "neg_only_norm_model":
            arange_N = torch.arange(N, device=u_st_view1.device)
            pos_mask = torch.zeros(N, N, device=u_st_view1.device, dtype=torch.bool)
            pos_mask[arange_N, pos_idx] = True                    # [2B, 2B]

        # v88 (MACL-paired): pre-compute per-codebook MACL scaling factor.
        # macl_factor_m = 1 + alpha * (A_m - A_0), where A_m is the batch-mean
        # paired-aug positive cosine in codebook m (detached scalar). This
        # multiplies the base temperature T -> T_eff_m for codebook m. When
        # combined with v42 text_cos (variant="text_cos"), the final tau is
        # tau_eff = T_eff_m * (1 + alpha_text * cos(text_i^m, text_j^m)).
        use_macl = (
            float(macl_alpha) > 0.0
            and paired_align_signal is not None
            and paired_align_signal.dim() == 2
            and paired_align_signal.shape[1] == num_codebooks
        )

        loss_sum = u_st_view1.new_zeros(())
        for m in range(num_codebooks):
            z = torch.cat([u1[:, m], u2[:, m]], dim=0)            # [2B, 3, 4]
            sim = torch.einsum("brc,src->bsr", z, z).mean(dim=-1)  # [2B, 2B]
            # v60: skip dynamic-tau on m=0 if requested
            use_dyn_this_m = use_dyn and not (m == 0 and skip_global_dyn)
            # v88: compute per-codebook MACL temperature factor (scalar).
            if use_macl:
                A_m = paired_align_signal[:, m].mean().detach()    # scalar in [-1, 1]
                macl_factor = 1.0 + float(macl_alpha) * (A_m - float(macl_a0))
                T_eff = max(float(T), 1e-6) * macl_factor.clamp(min=tau_floor / max(float(T), 1e-6))
            else:
                T_eff = T                                          # python float
            if use_dyn_this_m and variant == "neg_only_norm_model":
                # v68 (model_scale removed) -----------------------------------
                # (1) positive pair uses static base_tau (no weakening)
                # (2) negative pair uses base_tau * semantic_scale_ij ONLY
                #   - semantic_scale_ij = clamp(1 + alpha * tanh((cos_t - mu) / std), s_min, s_max)
                # The earlier MACL-style model_scale term (depending on A_m =
                # positive codon agreement) is intentionally removed: the codon
                # output `u_st` is the same tensor whose loss we are computing,
                # so coupling tau to A_m closes a loop between similarity and
                # temperature that is hard to diagnose. v68 isolates the pure
                # text-affinity effect on the negative-pair temperature.
                # All tau-modulating quantities are detached.
                # Config flags model_beta / model_a0 / model_scale_(min,max)
                # are accepted for backward-compat but unused in this branch.
                if m == 0 and global_use_local_mean and text_part_raw.shape[1] >= 2:
                    t_m = text_part_raw[:, 1:, :].mean(dim=1)
                else:
                    t_m = text_part_raw[:, m, :]
                t_all = torch.cat([t_m, t_m], dim=0)              # [2B, D]
                t_n   = F.normalize(t_all, dim=-1).detach()
                cos_t = (t_n @ t_n.t()).detach()                  # [2B, 2B]
                # offdiag normalization
                offdiag = cos_t[~eye]                             # [N*(N-1)]
                mu_m  = offdiag.mean()
                std_m = offdiag.std().clamp_min(1e-6)
                g_ij  = torch.tanh((cos_t - mu_m) / std_m)        # [2B, 2B] approx in (-1, 1)
                semantic_scale = (1.0 + alpha * g_ij).clamp(
                    min=float(semantic_scale_min),
                    max=float(semantic_scale_max),
                )                                                  # [2B, 2B]
                # negative-pair tau matrix (semantic-only)
                # v88: T_eff already incorporates the per-cb MACL factor
                T_neg = (T_eff * semantic_scale).clamp(min=tau_floor) # [2B, 2B]
                # positive pair uses base T_eff; negative pair uses T_neg
                if isinstance(T_eff, torch.Tensor):
                    T_pos_fill = T_eff
                else:
                    T_pos_fill = T_neg.new_full((), float(T_eff))
                T_matrix = torch.where(pos_mask, T_pos_fill, T_neg)
                T_matrix = T_matrix.clamp(min=tau_floor)
                sim = (sim / T_matrix).masked_fill(eye, -1e9)
            elif use_dyn_this_m:
                # legacy v42 "text_cos" variant
                # v88: T_eff already incorporates the per-cb MACL factor;
                # the (1 + alpha*cos_t) factor multiplies on top.
                if m == 0 and global_use_local_mean and text_part_raw.shape[1] >= 2:
                    t_m = text_part_raw[:, 1:, :].mean(dim=1)     # [B, D] = mean over 5 local slots
                else:
                    t_m = text_part_raw[:, m, :]                  # [B, D]
                t_all = torch.cat([t_m, t_m], dim=0)              # [2B, D]
                t_n   = F.normalize(t_all, dim=-1)
                cos_t = t_n @ t_n.t()                             # [2B, 2B]
                T_ij  = T_eff * (1.0 + alpha * cos_t)             # [2B, 2B]
                T_ij  = T_ij.clamp(min=tau_floor)
                sim   = (sim / T_ij).masked_fill(eye, -1e9)
            else:
                # v88: T_eff = T * macl_factor (or just T if macl disabled).
                if isinstance(T_eff, torch.Tensor):
                    T_eff_clamped = T_eff.clamp(min=tau_floor)
                    sim = (sim / T_eff_clamped).masked_fill(eye, -1e9)
                else:
                    sim = (sim / T_eff).masked_fill(eye, -1e9)

            # v87 (Wang & Liu, CVPR 2021 Eq 9): per-anchor explicit hard-
            # negative sampling. After the (tau-scaled) sim matrix is built,
            # keep only the top-alpha fraction of NEGATIVE entries (highest
            # similarity = informative hard negs); the positive entry and
            # diagonal self-entry are preserved unconditionally. This makes
            # the InfoNCE denominator look only at informative negatives,
            # decoupling uniformity from tolerance to semantically similar
            # samples.  alpha=1.0 = legacy (no truncation, bit-exact path).
            if 0.0 < float(hard_neg_alpha) < 1.0:
                # Build positive-pair mask [2B, 2B] (positions of positive partners).
                pos_mask_alpha = torch.zeros(
                    N, N, device=sim.device, dtype=torch.bool,
                )
                pos_mask_alpha[torch.arange(N, device=sim.device), pos_idx] = True
                # Number of negatives kept (exclude self + positive partner).
                n_neg_avail = N - 2
                k_keep = max(1, int(round(float(hard_neg_alpha) * n_neg_avail)))
                k_keep = min(k_keep, n_neg_avail)
                # Mask out self + positive before quantile so they don't enter
                # the top-k of negatives.
                sim_for_q = sim.masked_fill(eye | pos_mask_alpha, float("-inf"))
                # Top-k per anchor; threshold = k-th largest negative value.
                topk_vals, _ = sim_for_q.topk(k_keep, dim=-1)         # [2B, k_keep]
                thresh = topk_vals[:, -1:].expand_as(sim)              # [2B, 2B]
                # Keep iff: positive pair, OR (sim >= threshold AND not diag).
                keep_mask = pos_mask_alpha | ((sim >= thresh) & ~eye)
                sim = sim.masked_fill(~keep_mask, -1e9)

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

    def _loss_codebook_ortho(self, z: torch.Tensor) -> torch.Tensor:
        """v79a (#1.2): cross-codebook orthogonality on per-batch z means.

        z : [B, M, D] semantic_visual_tokens (pre-quant, has grad via
            visual_adapter)

        Penalises cos(z_m_batch_mean, z_n_batch_mean)^2 for m ≠ n. The
        batch-mean per codebook is the *empirical direction* into which
        each codebook gets EMA-updated. Pushing these directions apart
        in z-space gradient-trains visual_adapter to produce slot-
        differentiated features. Codebook buffer itself has no
        gradient (EMA mode) but follows z by construction.
        """
        z_mean = z.mean(dim=0)                                            # [M, D]
        z_n = F.normalize(z_mean, dim=-1)                                 # [M, D]
        sim = z_n @ z_n.t()                                                # [M, M]
        eye = torch.eye(sim.shape[0], device=sim.device, dtype=torch.bool)
        return (sim[~eye] ** 2).mean()

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

    # ------------------------------------------------------------------
    # v119: CIBHash-style per-codebook NtXent + symmetric KL on binary hash
    # ------------------------------------------------------------------
    @staticmethod
    def _continuous_code_to_bit_probs(u: torch.Tensor, num_codebooks: int = 6) -> torch.Tensor:
        """Map per-codon softmax probs [B, M*L, 4] over (A, C, G, T) to a
        (2L)-bit-per-codebook probability vector [B, M, 2*L] using the DNA
        encoding A=00, C=01, G=10, T=11.
            bit_0_prob = P(G) + P(T)   (= probability that base's first bit is 1)
            bit_1_prob = P(C) + P(T)   (= probability that base's second bit is 1)
        Returns probabilities in [0, 1] (no sigmoid needed). L is inferred
        from the input shape (L = R / num_codebooks).
        """
        B, R, C = u.shape
        assert C == 4, (
            f"[cibhash] expected continuous_code last-dim 4; got [{B}, {R}, {C}]"
        )
        assert R % num_codebooks == 0, (
            f"[cibhash] R={R} not divisible by num_codebooks={num_codebooks}; "
            f"cannot infer codon length L."
        )
        L = R // num_codebooks                                       # codon positions per codebook
        # P(G) is u[..., 2], P(T) is u[..., 3], P(C) is u[..., 1]
        bit_0 = u[..., 2] + u[..., 3]                                # [B, M*L]
        bit_1 = u[..., 1] + u[..., 3]                                # [B, M*L]
        bits  = torch.stack([bit_0, bit_1], dim=-1)                  # [B, M*L, 2]
        return bits.reshape(B, num_codebooks, 2 * L)                 # [B, M, 2*L]

    @staticmethod
    def _ste_sign(prob: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Straight-through-estimator sign(prob - threshold).

        Forward: ±1 (ties broken to +1 to avoid 0 outputs).
        Backward: identity gradient pass-through.
        Note: CIBHash's ``torch.sign`` returns 0 at delta=0. At our
        codon-head initialization, ``continuous_code`` is near-uniform
        4-way softmax so bit_probs = P(G)+P(T) and P(C)+P(T) sit at
        exactly 0.5 → delta = 0 → z = 0 (zero vector) → cosine sim is
        NaN. Use the indicator instead so z is always in {-1, +1} and
        the cosine normalization is well-defined.
        """
        delta = prob - threshold
        z_hard = 2.0 * (delta >= 0).to(prob.dtype) - 1.0
        return delta + (z_hard - delta).detach()

    def _loss_proto_cluster_per_codebook(
        self,
        codebook_distances_v1: torch.Tensor,
        codebook_distances_v2: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        """v138: paired-view prototype-cluster InfoNCE on `codebook_distances`.

        Inputs are the [B, M, K] cost tensors from the quantizer for two
        augmented views. We treat softmax(-distances / tau) as a soft
        prototype assignment vector per (sample, codebook). For each
        codebook independently we form a [B, B] cosine similarity matrix
        between view1 assignments and view2 assignments and apply a
        symmetric InfoNCE (diagonal = positive, off-diag = negative).

        The per-codebook losses are averaged. Provides a per-codebook
        paired-view consistency signal WITHOUT bottlenecking codon_head
        input (which uses raw routed tokens when
        `--codon_input_source routed`).
        """
        assert codebook_distances_v1.shape == codebook_distances_v2.shape, (
            f"shape mismatch: v1 {tuple(codebook_distances_v1.shape)} vs "
            f"v2 {tuple(codebook_distances_v2.shape)}"
        )
        B, M, K = codebook_distances_v1.shape
        T = max(float(temperature), 1e-6)
        p_v1 = F.softmax(-codebook_distances_v1 / T, dim=-1)  # [B, M, K]
        p_v2 = F.softmax(-codebook_distances_v2 / T, dim=-1)
        losses: list = []
        labels = torch.arange(B, device=p_v1.device)
        for m in range(M):
            a = F.normalize(p_v1[:, m, :], dim=-1, eps=1e-8)  # [B, K]
            b = F.normalize(p_v2[:, m, :], dim=-1, eps=1e-8)
            sim = (a @ b.T) / T                                # [B, B]
            loss_a2b = F.cross_entropy(sim,   labels)
            loss_b2a = F.cross_entropy(sim.T, labels)
            losses.append(0.5 * (loss_a2b + loss_b2a))
        return torch.stack(losses).mean()

    @staticmethod
    def _cibhash_kl(prob_view1: torch.Tensor, prob_view2: torch.Tensor,
                    eps: float = 1e-6) -> torch.Tensor:
        """Symmetric Bernoulli KL between two probability tensors with the
        same shape, averaged over batch and summed over bit-dim.

        NaN-safety:
        - eps must be > float32 epsilon (≈ 1.19e-7). With eps=1e-8 the
          expression `1 - (1 - eps)` underflows to 0 in float32, yielding
          log(0) = -inf and (-inf) - (-inf) = NaN. eps=1e-6 stays safely
          above the float32 representable gap near 1.
        - Both `p` and `1 - p` are explicitly clamp_min'd to eps so any
          remaining numerical pinches still produce finite logs.

        CIBHash's compute_kl is asymmetric (KL(prob || prob_v.detach())).
        We make it symmetric (both directions, both detached against the
        other view inside their own term — gradient flows from each view
        only through that view's contribution).
        """
        p = prob_view1.clamp(eps, 1.0 - eps)
        q = prob_view2.clamp(eps, 1.0 - eps)
        one_mp = (1.0 - p).clamp_min(eps)
        one_mq = (1.0 - q).clamp_min(eps)
        log_p   = p.log();     log_1mp = one_mp.log()
        log_q_d = q.detach().log();         log_1mq_d = one_mq.detach().log()
        kl_pq = p * (log_p - log_q_d) + one_mp * (log_1mp - log_1mq_d)
        log_q   = q.log();     log_1mq = one_mq.log()
        log_p_d = p.detach().log();         log_1mp_d = one_mp.detach().log()
        kl_qp = q * (log_q - log_p_d) + one_mq * (log_1mq - log_1mp_d)
        kl = 0.5 * (kl_pq.sum(dim=-1).mean() + kl_qp.sum(dim=-1).mean())
        return kl

    def _loss_cibhash_per_codebook(
        self,
        continuous_code_view1: torch.Tensor,
        continuous_code_view2: torch.Tensor,
        temperature: float,
        mode: str = "per_codebook",
        text_part_raw: Optional[torch.Tensor] = None,
        dynamic_tau_alpha: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """v119/v120: CIBHash NtXent + symmetric Bernoulli KL on binary
        hash, with two switches added in v120.

        Args:
            continuous_code_view{1,2}: [B, 18, 4] paired-aug softmax probs.
            temperature: NtXent temperature (CIBHash paper default 0.3).
            mode:
                "per_codebook" (v119a default) — compute NtXent and KL
                    independently per codebook m=0..5 on the 6-bit
                    sub-vectors and average. Forces per-codebook
                    discriminativity.
                "global"        (v120f) — flatten bit_probs to [B, 36]
                    and compute one NtXent + KL on the full DNA hash.
                    Mirrors the original CIBHash design (one binary code
                    per image).
            text_part_raw: [B, M=6, D] per-slot CLIP text embeddings.
                Required (with `dynamic_tau_alpha > 0`) for the v120e
                text-cos dynamic-tau variant. Reuses the same pairwise
                text similarity geometry as the per-codebook DNA-NtXent
                (v42): for each positive/negative pair (i, j), the per-
                codebook temperature is
                    tau_ij^m = T * (1 + alpha * cos(t_i^m, t_j^m))
                so semantically similar samples get a softer push (larger
                tau) and semantically distant ones get a harder push
                (smaller tau). Targets the uniformity-tolerance dilemma.
                Only takes effect in `mode == "per_codebook"`.
            dynamic_tau_alpha: alpha in [0, 1). 0 disables.
        Returns:
            (ntxent_loss, kl_loss).
        """
        # Per-codebook bit probabilities
        bits_v1 = self._continuous_code_to_bit_probs(continuous_code_view1)   # [B, M=6, 6]
        bits_v2 = self._continuous_code_to_bit_probs(continuous_code_view2)   # [B, M=6, 6]
        # STE sign to binary hash
        z_v1 = self._ste_sign(bits_v1)                                         # [B, 6, 6] in {-1, +1}
        z_v2 = self._ste_sign(bits_v2)
        B, M, K_bits = bits_v1.shape                                           # M=6, K_bits=6
        T = max(float(temperature), 1e-6)

        # ----- Global mode: flatten to one 36-bit hash, one NtXent + one KL
        if mode == "global":
            z1_flat = z_v1.reshape(B, M * K_bits)                              # [B, 36]
            z2_flat = z_v2.reshape(B, M * K_bits)
            z = torch.cat([z1_flat, z2_flat], dim=0)                           # [2B, 36]
            z_n = F.normalize(z, dim=-1, eps=1e-8)
            sim = (z_n @ z_n.T) / T
            N = 2 * B
            eye = torch.eye(N, device=sim.device, dtype=torch.bool)
            sim = sim.masked_fill(eye, -1e9)
            pos_idx = torch.cat([
                torch.arange(B, 2 * B, device=sim.device),
                torch.arange(0, B,     device=sim.device),
            ])
            ntxent = F.cross_entropy(sim, pos_idx)

            pm1 = bits_v1.reshape(B, M * K_bits)
            pm2 = bits_v2.reshape(B, M * K_bits)
            kl = self._cibhash_kl(pm1, pm2)
            return ntxent, kl

        # ----- Per-codebook mode (v119a, possibly with text-cos dynamic tau)
        use_dyn = (
            text_part_raw is not None
            and float(dynamic_tau_alpha) > 0.0
            and text_part_raw.shape[0] == B
            and text_part_raw.shape[1] == M
        )
        alpha = min(float(dynamic_tau_alpha), 0.999)

        ntxent_per_cb: list = []
        kl_per_cb: list     = []
        for m in range(M):
            zm1 = z_v1[:, m, :]                                                # [B, 6]
            zm2 = z_v2[:, m, :]                                                # [B, 6]
            z = torch.cat([zm1, zm2], dim=0)                                   # [2B, 6]
            z_n = F.normalize(z, dim=-1, eps=1e-8)
            sim_raw = z_n @ z_n.T                                              # [2B, 2B], cosine

            if use_dyn:
                # Per-pair temperature from text-cos for codebook m.
                t_m = text_part_raw[:, m, :]                                    # [B, D]
                t_m_n = F.normalize(t_m.float(), dim=-1)
                cos_tt = (t_m_n @ t_m_n.T).clamp(-1.0, 1.0)                     # [B, B]
                # Tile to [2B, 2B] (two views share the same text).
                cos_tt_2 = cos_tt.repeat(2, 2)
                tau_ij = T * (1.0 + alpha * cos_tt_2).clamp_min(1e-4)           # [2B, 2B]
                sim = sim_raw / tau_ij
            else:
                sim = sim_raw / T

            N = 2 * B
            eye = torch.eye(N, device=sim.device, dtype=torch.bool)
            sim = sim.masked_fill(eye, -1e9)
            pos_idx = torch.cat([
                torch.arange(B, 2 * B, device=sim.device),
                torch.arange(0, B,     device=sim.device),
            ])
            ntxent_m = F.cross_entropy(sim, pos_idx)
            ntxent_per_cb.append(ntxent_m)

            pm1 = bits_v1[:, m, :]                                              # [B, 6]
            pm2 = bits_v2[:, m, :]                                              # [B, 6]
            kl_m = self._cibhash_kl(pm1, pm2)
            kl_per_cb.append(kl_m)

        ntxent = torch.stack(ntxent_per_cb).mean()
        kl     = torch.stack(kl_per_cb).mean()
        return ntxent, kl

    # ------------------------------------------------------------------
    # v121: SwAV-style swapped balanced codeword-assignment loss
    # ------------------------------------------------------------------
    @staticmethod
    def _swav_sinkhorn_target(
        logits: torch.Tensor,
        eps: float,
        n_iters: int,
        inactive_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Log-domain Sinkhorn-Knopp balanced soft-assignment target.

        Mirrors SwAV's `distributed_sinkhorn` (Caron et al., NeurIPS 2020,
        Algorithm 1). Given assignment logits L [B, K], returns a doubly-
        normalized soft target Q [B, K] where each row sums to 1 (per-
        sample distribution over codewords) and each column sums to ~B/K
        (uniform codeword usage across the batch).

        Args:
            logits: [B, K] (assignment logits; will be detached).
            eps: Sinkhorn entropy regularizer; smaller => sharper target.
            n_iters: number of row/col normalization passes.
            inactive_mask: [K] bool, True = inactive codeword. Inactive
                columns are forced to zero mass.
        Returns:
            Q [B, K], detached (teacher target).
        """
        assert logits.dim() == 2, (
            f"[swav_sinkhorn] expected logits [B, K]; got {tuple(logits.shape)}"
        )
        B, K = logits.shape
        # Stop-gradient on the target; rescale by eps.
        L = logits.detach().float() / max(float(eps), 1e-12)
        if inactive_mask is not None:
            mask_b = inactive_mask.to(L.device).bool().unsqueeze(0).expand(B, -1)
            L = L.masked_fill(mask_b, -1e9)
        # Work in log-domain on Q^T [K, B] for SwAV's column-uniform marginal.
        log_Q = L.t()                                                 # [K, B]
        # Normalize total mass to 1 (joint distribution).
        log_Q = log_Q - log_Q.flatten().logsumexp(dim=0)
        log_K = math.log(K)
        log_B = math.log(B)
        for _ in range(int(max(n_iters, 0))):
            # row-normalize: each row sums to 1/K (codeword marginal uniform)
            log_Q = log_Q - log_Q.logsumexp(dim=1, keepdim=True) - log_K
            # col-normalize: each column sums to 1/B (sample marginal uniform)
            log_Q = log_Q - log_Q.logsumexp(dim=0, keepdim=True) - log_B
        # Final column-sum should be 1 (per-sample distribution): multiply by B.
        log_Q = log_Q + log_B
        # Convert to probabilities; clamp to avoid -inf -> 0 numerical issues.
        Q = log_Q.exp().t().clamp_min(0.0)                            # [B, K]
        # Hard-zero inactive columns for safety, then renormalize rows.
        if inactive_mask is not None:
            Q = Q.masked_fill(inactive_mask.to(Q.device).bool().unsqueeze(0), 0.0)
            row_sum = Q.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            Q = Q / row_sum
        return Q.detach()

    def _loss_swav_assign(
        self,
        distances_view1: torch.Tensor,
        distances_view2: torch.Tensor,
        tau_pred: float,
        sinkhorn_eps: float,
        sinkhorn_iters: int,
        active_mask: Optional[torch.Tensor] = None,
        include_global: bool = False,
    ) -> torch.Tensor:
        """v121: SwAV-style swapped balanced codeword-assignment loss
        (paper-internal: 'codeword-level' supervision applied BEFORE the
        codon decoder, complementing v119a's bit-level CIBHash NtXent).

        For each codebook m in the local set (slot 1..5 by default):
            logits_m^v = -distances_view{v}[:, m, :] / tau_pred           # [B, K]
            Q_m^v       = Sinkhorn-balanced soft target from logits_m^v   # [B, K]
            log_p_m^v   = log_softmax(logits_m^v, dim=-1)
            loss_m      = -mean_b[ sum_k Q_m^1 * log_p_m^2 + Q_m^2 * log_p_m^1 ]

        The swapped target encourages (a) paired-aug assignment consistency
        and (b) batch-level codeword usage balance. Inactive codewords are
        masked out of the softmax / Sinkhorn / loss.

        Args:
            distances_view{1,2}: [B, M, K_max] squared-L2 distances from
                the quantizer (= outputs['codebook_distances'] for each view).
            tau_pred: prediction softmax temperature.
            sinkhorn_eps: Sinkhorn entropy regularizer.
            sinkhorn_iters: Sinkhorn-Knopp iteration count.
            active_mask: [M, K_max] bool, True = active codeword.
            include_global: if True, include slot 0; default False (skip C_0).
        Returns:
            scalar loss = mean over considered slots of (CE(q1,p2) + CE(q2,p1)).
        """
        assert distances_view1.dim() == 3 and distances_view2.dim() == 3, (
            f"[swav_assign] expected distances [B, M, K]; got "
            f"{tuple(distances_view1.shape)} / {tuple(distances_view2.shape)}"
        )
        assert distances_view1.shape == distances_view2.shape, (
            f"[swav_assign] view1/view2 shape mismatch: "
            f"{tuple(distances_view1.shape)} vs {tuple(distances_view2.shape)}"
        )
        B, M, K = distances_view1.shape
        T = max(float(tau_pred), 1e-6)
        if active_mask is not None:
            assert active_mask.dim() == 2 and active_mask.shape == (M, K), (
                f"[swav_assign] active_mask shape {tuple(active_mask.shape)} "
                f"!= expected ({M}, {K})"
            )
        # Local slots are 1..M-1; C_global (slot 0) is the default exclusion.
        slot_indices = list(range(0, M)) if include_global else list(range(1, M))

        losses: list = []
        for m in slot_indices:
            d1 = distances_view1[:, m, :]                                    # [B, K]
            d2 = distances_view2[:, m, :]
            logits1 = (-d1 / T)
            logits2 = (-d2 / T)
            inactive_m: Optional[torch.Tensor] = None
            if active_mask is not None:
                inactive_m = (~active_mask[m].bool()).to(logits1.device)     # [K]
                mask_b = inactive_m.unsqueeze(0).expand(B, -1)
                # Mask inactive logits to large negative so they vanish in
                # both softmax (prediction) and Sinkhorn (target).
                logits1 = logits1.masked_fill(mask_b, -1e9)
                logits2 = logits2.masked_fill(mask_b, -1e9)
            # Balanced teacher targets (detached).
            q1 = self._swav_sinkhorn_target(
                logits1, eps=sinkhorn_eps, n_iters=sinkhorn_iters,
                inactive_mask=inactive_m,
            )
            q2 = self._swav_sinkhorn_target(
                logits2, eps=sinkhorn_eps, n_iters=sinkhorn_iters,
                inactive_mask=inactive_m,
            )
            # Soft-label cross-entropy (swapped).
            log_p1 = F.log_softmax(logits1, dim=-1).clamp_min(-1e9)
            log_p2 = F.log_softmax(logits2, dim=-1).clamp_min(-1e9)
            ce_12 = -(q1 * log_p2).sum(dim=-1).mean()
            ce_21 = -(q2 * log_p1).sum(dim=-1).mean()
            losses.append(ce_12 + ce_21)
        if not losses:
            return distances_view1.new_zeros(())
        return torch.stack(losses).mean()

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

        v76b: when `--vq_loss_cosine` is set, both terms become
        (1 - cos(., .)) instead of MSE. This matches the cosine VQ lookup
        in `SemanticCodebookQuantizer` (--vq_distance_mode=cosine) so the
        commitment term pulls z toward q in the same geometry used at
        lookup time.
        """
        if z.shape != q.shape:
            raise ValueError(
                f"[loss_vq] shape mismatch: semantic_visual_tokens {tuple(z.shape)} "
                f"vs quantized_tokens_raw {tuple(q.shape)}"
            )
        if self.vq_loss_cosine:
            # cos sim per (b, m) -> average. (1 - cos) ∈ [0, 2].
            codebook_loss   = (
                1.0 - F.cosine_similarity(q, z.detach(), dim=-1)
            ).mean()
            commitment_loss = (
                1.0 - F.cosine_similarity(z, q.detach(), dim=-1)
            ).mean()
        else:
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
        # KL(uniform || p_bar): forward KL, mode-covering -> strong penalty on
        # unused bases (p_bar_c -> 0 makes -log p_bar diverge).
        base_usage = u.mean(dim=0)                                # [18, 4]
        log_base_usage = (base_usage + self.eps).log()            # [18, 4]
        uniform = torch.full_like(base_usage, 0.25)
        loss_base_balance = F.kl_div(
            log_base_usage, uniform, reduction="batchmean"
        )
        loss = loss_entropy + self.eta_base_balance * loss_base_balance
        return {
            "loss_dna":          loss,
            "loss_entropy":      loss_entropy,
            "loss_base_balance": loss_base_balance,
        }

    # ------------------------------------------------------------------
    # v106: codeword <-> DNA codon bijection losses (operate on the codon
    # decoder applied to codewords; no sample dependence). All three take
    # codeword_codon_logits [M, K, 3, 4] from the model forward.
    # ------------------------------------------------------------------

    @staticmethod
    def _joint_codon_distribution(logits: torch.Tensor, eps: float) -> torch.Tensor:
        """logits [M, K, L, V=4] -> joint codon distribution [M, K, V^L].

        Generalized to arbitrary L (=num_codons_per_codebook). For L=3 this
        is the original 3-position outer product giving |C| = 4^3 = 64
        codons; for L=4 it gives |C| = 4^4 = 256 codons (used by v122a to
        re-enable the Sinkhorn codeword-codon bijection at K up to 256
        without the K=128 pigeonhole forced-collision diagnosed in v120c).
        """
        assert logits.dim() == 4, (
            f"[joint_codon] expected logits [M, K, L, V]; got {tuple(logits.shape)}"
        )
        p = F.softmax(logits, dim=-1)                                # [M, K, L, V]
        M, K, L, V = p.shape
        # Iterative outer product across the L codon positions: at step l
        # P has shape [M, K, V^l]; multiply by p[:, :, l, :] (V) -> [M, K, V^(l+1)].
        P = p[:, :, 0, :]                                             # [M, K, V]
        for l in range(1, L):
            P = (P.unsqueeze(-1) * p[:, :, l, :].unsqueeze(-2)).reshape(M, K, -1)
        return P                                                      # [M, K, V^L]

    def _loss_codeword_codon_sinkhorn(
        self,
        codeword_codon_logits: torch.Tensor,            # [M, K_max, 3, 4]
        codeword_K_active: torch.Tensor,                # [M] long
    ) -> torch.Tensor:
        """Sinkhorn-OT bijection loss.

        For each codebook m, solves the entropy-regularized OT problem
        between K_active codewords and |C|=64 codons with uniform marginals
        (1/K, 1/64). The optimal transport plan T* enforces hard marginal
        constraints (no codon can receive >1/64 mass), which prevents the
        K' < K collapse failure mode of softer regularizers.

        Loss = sum_{m, k, j} T*_{kj} . (-log P_k(j))
             = expected NLL of codewords under the optimal soft permutation.
        """
        eps_reg = max(float(self.codeword_codon_sinkhorn_eps), 1e-3)
        n_iter  = max(int(self.codeword_codon_sinkhorn_iters), 1)
        eps     = self.eps
        # codeword_codon_logits: [M, K_max, L, V=4]; |C| = V**L (=64 at L=3, =256 at L=4).
        M, K_max, L, V = codeword_codon_logits.shape
        num_codons = V ** L
        device = codeword_codon_logits.device
        log_p_codon_full = (
            self._joint_codon_distribution(codeword_codon_logits, eps).clamp(min=eps).log()
        )                                                              # [M, K_max, |C|]
        per_codebook = []
        for m in range(M):
            K = int(codeword_K_active[m].item()) if codeword_K_active is not None else K_max
            if K <= 0:
                continue
            cost = -log_p_codon_full[m, :K]                            # [K, |C|]
            log_a = torch.full((K,),          -math.log(K),          device=device, dtype=cost.dtype)
            log_b = torch.full((num_codons,), -math.log(num_codons), device=device, dtype=cost.dtype)
            log_K_mat = -cost / eps_reg                                # [K, |C|]
            u = torch.zeros(K,          device=device, dtype=cost.dtype)
            v = torch.zeros(num_codons, device=device, dtype=cost.dtype)
            for _ in range(n_iter):
                u = log_a - torch.logsumexp(log_K_mat + v[None, :], dim=1)
                v = log_b - torch.logsumexp(log_K_mat + u[:, None], dim=0)
            log_T = log_K_mat + u[:, None] + v[None, :]                # [K, |C|]
            T = log_T.exp()                                            # [K, |C|]
            per_codebook.append((T * cost).sum())
        if not per_codebook:
            return codeword_codon_logits.new_zeros(())
        return torch.stack(per_codebook).mean()

    def _effective_codeword_codon_sinkhorn_lambda(
        self,
        epoch: Optional[int],
    ) -> float:
        """Return the scalar Sinkhorn-bijection weight for this epoch.

        The loss inputs stay [M, K_max, 3, 4]; v111b only changes this
        multiplier so the codeword->codon shape contract is unchanged.
        """
        lam = float(self.lambda_codeword_codon_sinkhorn)
        warmup = max(int(self.codeword_codon_sinkhorn_warmup_epochs), 0)
        if lam <= 0.0 or warmup <= 0:
            return lam
        if epoch is None:
            return lam
        progress = float(min(max(int(epoch) + 1, 0), warmup)) / float(warmup)
        return lam * progress

    def _loss_codeword_codon_agg_ent(
        self,
        codeword_codon_logits: torch.Tensor,            # [M, K_max, 3, 4]
        codeword_K_active: torch.Tensor,                # [M] long
    ) -> torch.Tensor:
        """Aggregated entropy bijection loss (weaker, simpler).

        L = (log K - H(P_bar)) + alpha . mean_k H(P_k)
        where P_bar = mean_k P_k is the aggregate codon distribution.
        """
        alpha = float(self.codeword_codon_agg_ent_alpha)
        eps = self.eps
        P_full = self._joint_codon_distribution(codeword_codon_logits, eps)  # [M, K_max, 64]
        M, K_max = P_full.shape[0], P_full.shape[1]
        per_codebook = []
        for m in range(M):
            K = int(codeword_K_active[m].item()) if codeword_K_active is not None else K_max
            if K <= 0:
                continue
            P = P_full[m, :K]                                          # [K, 64]
            P_bar = P.mean(dim=0)                                      # [64]
            term_uniform = math.log(K) + (P_bar * (P_bar + eps).log()).sum()
            H_per = -(P * (P + eps).log()).sum(dim=-1)                 # [K]
            term_sharp = H_per.mean()
            per_codebook.append(term_uniform + alpha * term_sharp)
        if not per_codebook:
            return codeword_codon_logits.new_zeros(())
        return torch.stack(per_codebook).mean()

    def _loss_proto_cluster_cos(
        self,
        z: torch.Tensor,                # [B, M, D] semantic_visual_tokens
        codebooks: torch.Tensor,        # [M, K_max, D] EMA codebook buffer
        codebook_indices: torch.Tensor, # [B, M] long, assigned codeword per (b, m)
        codebook_active_mask: Optional[torch.Tensor] = None,  # [M, K_max] bool
    ) -> torch.Tensor:
        """v107: Prototype cosine clustering (InfoNCE).

        For each (b, m), compute cosine similarity between z[b, m] and ALL
        K_active codewords in codebook m, then apply cross-entropy with the
        assigned codebook_index as the target class. This pulls z toward
        its assigned prototype AND pushes it away from the other prototypes
        in cosine space. Conceptually equivalent to SwAV / ProtoCL clustering
        with codewords as cluster centers.
        """
        tau = max(float(self.proto_cluster_cos_tau), 1e-6)
        eps = self.eps
        B, M, D = z.shape
        K_max = codebooks.shape[1]
        # Normalize
        z_n  = F.normalize(z, dim=-1)                                 # [B, M, D]
        cb_n = F.normalize(codebooks, dim=-1)                         # [M, K_max, D]
        # Cosine similarity (logits): [B, M, K_max]
        sim = torch.einsum('bmd,mkd->bmk', z_n, cb_n) / tau
        # Mask out inactive codewords (adaptive-K case) by sending them to -inf
        if codebook_active_mask is not None:
            inactive = ~codebook_active_mask                          # [M, K_max]
            sim = sim.masked_fill(inactive[None, :, :], float('-inf'))
        # Cross-entropy with the routing assignment as target
        sim_flat   = sim.reshape(B * M, K_max)                        # [B*M, K_max]
        target_flat = codebook_indices.reshape(B * M).long()          # [B*M]
        return F.cross_entropy(sim_flat, target_flat)

    @torch.no_grad()
    def refresh_clusters(
        self,
        codebooks: torch.Tensor,                          # [M, K_max, D]
        codebook_active_mask: Optional[torch.Tensor] = None,  # [M, K_max] bool
        method: str = "kmeans",
        seed: int = 42,
    ) -> Dict[str, float]:
        """v112: re-cluster codewords within each codebook via k-means on
        cosine-normalized codeword embeddings. Updates
        self.hierarchical_cluster_labels [M, K_max].
        """
        import numpy as np
        import time
        from sklearn.cluster import KMeans
        M, K_max, D = codebooks.shape
        C = max(int(self.hierarchical_cluster_n_clusters), 2)
        new_labels = torch.zeros(M, K_max, dtype=torch.long, device=codebooks.device)
        cb_np = codebooks.detach().cpu().float().numpy()
        t0 = time.time()
        n_active_per_cb = []
        for m in range(M):
            if codebook_active_mask is not None:
                active = codebook_active_mask[m].cpu().numpy()
                valid_idx = np.nonzero(active)[0]
            else:
                valid_idx = np.arange(K_max)
            if len(valid_idx) < C:
                # not enough codewords to cluster; assign all to cluster 0
                n_active_per_cb.append(int(len(valid_idx)))
                continue
            feats = cb_np[m, valid_idx]
            # L2 normalize for cosine-based k-means
            feats = feats / (np.linalg.norm(feats, axis=-1, keepdims=True) + 1e-12)
            km = KMeans(n_clusters=C, random_state=int(seed + m), n_init=4, max_iter=100)
            km.fit(feats)
            for i, k in enumerate(valid_idx):
                new_labels[m, k] = int(km.labels_[i])
            n_active_per_cb.append(int(len(valid_idx)))
        self.hierarchical_cluster_labels.copy_(new_labels)
        self.hierarchical_cluster_initialized.fill_(True)
        return {
            "C": int(C),
            "M": int(M),
            "K_max": int(K_max),
            "elapsed_sec": float(time.time() - t0),
            "active_per_cb": n_active_per_cb,
        }

    def _loss_hierarchical_cluster_codon(
        self,
        codeword_codon_logits: torch.Tensor,              # [M, K_max, 3, 4]
        codeword_K_active: torch.Tensor,                   # [M] long
        codebook_active_mask: Optional[torch.Tensor] = None,  # [M, K_max] bool
    ) -> torch.Tensor:
        """v112: Hierarchical codon decomposition CE loss.

        For each codeword k in codebook m, the FIRST base of its predicted
        codon should classify cluster_labels[m, k] (a value in [0, C)).
        Since each base has 4 options (A/C/G/T) and we use C=4 clusters
        (default), this is exactly a 4-way classification on the first base.

        L = CE( logits[m, k, 0, :], cluster_labels[m, k] )  averaged over
            all active codewords across all codebooks.
        """
        if not bool(self.hierarchical_cluster_initialized.item()):
            return codeword_codon_logits.new_zeros(())
        M, K_max = codeword_codon_logits.shape[0], codeword_codon_logits.shape[1]
        C_classes = int(self.hierarchical_cluster_n_clusters)
        # First-base logits: [M, K_max, 4]
        first_base_logits = codeword_codon_logits[:, :, 0, :]
        # Targets: [M, K_max]
        targets = self.hierarchical_cluster_labels.to(device=first_base_logits.device).long()
        # Build active mask
        if codebook_active_mask is not None:
            mask = codebook_active_mask.to(device=first_base_logits.device).bool()
        else:
            mask = torch.ones(M, K_max, dtype=torch.bool, device=first_base_logits.device)
        # Only target classes < 4 (== number of bases) are valid CE targets.
        # If user sets n_clusters > 4, we still treat as 4-way classifier
        # by clamping targets (interpreted as: clusters >= 4 share base 3).
        targets_clamped = targets.clamp_max(3)
        # Flatten + mask
        flat_logits = first_base_logits[mask]                 # [N_active, 4]
        flat_targets = targets_clamped[mask]                  # [N_active]
        if flat_targets.numel() == 0:
            return codeword_codon_logits.new_zeros(())
        return F.cross_entropy(flat_logits, flat_targets)

    def _loss_codeword_codon_pairwise(
        self,
        codeword_codon_logits: torch.Tensor,
        codeword_K_active: torch.Tensor,
    ) -> torch.Tensor:
        """Pairwise distinctness loss (cheapest, weakest)."""
        eps = self.eps
        P_full = self._joint_codon_distribution(codeword_codon_logits, eps)  # [M, K_max, 64]
        M, K_max = P_full.shape[0], P_full.shape[1]
        per_codebook = []
        for m in range(M):
            K = int(codeword_K_active[m].item()) if codeword_K_active is not None else K_max
            if K <= 1:
                continue
            P = P_full[m, :K]                                          # [K, 64]
            S = P @ P.T                                                # [K, K]
            offdiag = S - torch.diag(torch.diag(S))
            per_codebook.append(offdiag.sum() / (K * (K - 1)))
        if not per_codebook:
            return codeword_codon_logits.new_zeros(())
        return torch.stack(per_codebook).mean()

    @torch.no_grad()
    def _ensure_codeword_text_proto(
        self,
        text_part_tokens: torch.Tensor,       # [B, M, D]
        K_max: int,
    ) -> None:
        B, M, D = text_part_tokens.shape
        shape_proto = (M, int(K_max), D)
        shape_seen = (M, int(K_max))
        if (
            self._codeword_text_proto.numel() == 0
            or tuple(self._codeword_text_proto.shape) != shape_proto
            or self._codeword_text_proto.device != text_part_tokens.device
            or self._codeword_text_proto.dtype != text_part_tokens.dtype
        ):
            self._codeword_text_proto = torch.zeros(
                shape_proto,
                device=text_part_tokens.device,
                dtype=text_part_tokens.dtype,
            )
            self._codeword_text_seen = torch.zeros(
                shape_seen,
                device=text_part_tokens.device,
                dtype=text_part_tokens.dtype,
            )

    def _loss_codeword_text_proto(
        self,
        z: torch.Tensor,                      # [B, M, D]
        text_part_tokens: torch.Tensor,       # [B, M, D]
        codebook_indices: torch.Tensor,       # [B, M]
        active_mask: Optional[torch.Tensor] = None,  # [M, K_max] bool
    ) -> torch.Tensor:
        """v123: align visual codeword assignments to text prototypes.

        The text prototype for (m, k) is an EMA mean of text_part_tokens[:, m]
        over samples whose visual slot m selected codeword k. Local slots are
        used by default; slot 0 is included only when explicitly requested.
        """
        assert z.dim() == 3, f"z must be [B, M, D], got {tuple(z.shape)}"
        assert text_part_tokens.dim() == 3, (
            f"text_part_tokens must be [B, M, D], got {tuple(text_part_tokens.shape)}"
        )
        assert codebook_indices.dim() == 2, (
            f"codebook_indices must be [B, M], got {tuple(codebook_indices.shape)}"
        )
        B, M, D = z.shape
        assert text_part_tokens.shape == (B, M, D), (
            f"text_part_tokens shape {tuple(text_part_tokens.shape)} "
            f"!= z shape {(B, M, D)}"
        )
        assert codebook_indices.shape == (B, M), (
            f"codebook_indices shape {tuple(codebook_indices.shape)} != {(B, M)}"
        )
        if active_mask is not None:
            assert active_mask.dim() == 2 and active_mask.shape[0] == M, (
                f"active_mask must be [M, K], got {tuple(active_mask.shape)}"
            )
            K_max = int(active_mask.shape[1])
        else:
            cfg_K = int(getattr(self.cfg, "codebook_size", 0))
            idx_K = int(codebook_indices.max().item()) + 1
            K_max = max(cfg_K, idx_K)

        self._ensure_codeword_text_proto(text_part_tokens, K_max)
        momentum = min(max(float(self.codeword_text_proto_momentum), 0.0), 0.9999)
        start_m = 0 if self.codeword_text_proto_include_global else 1

        with torch.no_grad():
            t_n = F.normalize(text_part_tokens.detach(), dim=-1)        # [B, M, D]
            for m in range(start_m, M):
                idx = codebook_indices[:, m].long().clamp(min=0, max=K_max - 1)  # [B]
                counts = torch.bincount(idx, minlength=K_max).to(t_n.dtype)      # [K]
                sums = torch.zeros(K_max, D, device=t_n.device, dtype=t_n.dtype)
                sums.index_add_(0, idx, t_n[:, m, :])                            # [K, D]
                valid = counts > 0
                if not bool(valid.any()):
                    continue
                mean = sums[valid] / counts[valid].unsqueeze(-1).clamp_min(self.eps)
                mean = F.normalize(mean, dim=-1)                                  # [K_valid, D]
                old = self._codeword_text_proto[m, valid]
                seen_old = self._codeword_text_seen[m, valid] > 0
                updated = torch.where(
                    seen_old.unsqueeze(-1),
                    momentum * old + (1.0 - momentum) * mean,
                    mean,
                )
                self._codeword_text_proto[m, valid] = F.normalize(updated, dim=-1)
                self._codeword_text_seen[m].add_(counts)

        proto = F.normalize(self._codeword_text_proto.detach().to(z), dim=-1)    # [M, K, D]
        seen = self._codeword_text_seen.detach().to(device=z.device)
        z_n = F.normalize(z, dim=-1)                                             # [B, M, D]
        tau = max(float(self.codeword_text_proto_tau), 1e-6)
        min_count = max(int(self.codeword_text_proto_min_count), 1)
        losses = []
        for m in range(start_m, M):
            valid_proto = seen[m] >= float(min_count)                            # [K]
            if active_mask is not None:
                valid_proto = valid_proto & active_mask[m].to(device=z.device).bool()
            if not bool(valid_proto.any()):
                continue
            idx = codebook_indices[:, m].long().clamp(min=0, max=K_max - 1)      # [B]
            sample_valid = valid_proto.gather(0, idx)                            # [B]
            if int(sample_valid.sum().item()) == 0:
                continue
            logits = (z_n[sample_valid, m, :] @ proto[m].T) / tau                # [Bv, K]
            logits = logits.masked_fill(~valid_proto.unsqueeze(0), -1e9)
            losses.append(F.cross_entropy(logits, idx[sample_valid]))
        if not losses:
            return z.new_zeros(())
        return torch.stack(losses).mean()

    @torch.no_grad()
    def _init_text_cluster_buffers(
        self,
        text_part_tokens: torch.Tensor,         # [B, M, D]
        K_max: int,
    ) -> None:
        B, M, D = text_part_tokens.shape
        C = max(int(self.text_cluster_count), 1)
        t = F.normalize(text_part_tokens.detach(), dim=-1)
        proto = torch.empty(M, C, D, device=t.device, dtype=t.dtype)
        for m in range(M):
            if B >= C:
                idx = torch.linspace(0, B - 1, C, device=t.device).round().long()
            else:
                idx = torch.arange(C, device=t.device).remainder(B)
            proto[m] = t[idx, m]
        self._text_cluster_prototypes = F.normalize(proto, dim=-1)      # [M, C, D]
        self._text_cluster_codeword = torch.full(
            (M, C, K_max), 1.0 / float(K_max),
            device=t.device, dtype=t.dtype,
        )                                                              # [M, C, K]

    def _loss_text_cluster_codon_ot(
        self,
        text_part_tokens: torch.Tensor,          # [B, M, D]
        codebook_indices: torch.Tensor,          # [B, M]
        codeword_codon_logits: torch.Tensor,     # [M, K_max, 3, 4]
        codeword_K_active: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Confidence-weighted text-cluster -> codon OT allocation.

        Text clusters are discovered per codebook from text_part_tokens. EMA
        estimates P(codeword | text-cluster), then each cluster's codon
        distribution is matched to a balanced codon allocation by Sinkhorn.
        """
        assert text_part_tokens.dim() == 3, "text_part_tokens must be [B, M, D]"
        assert codebook_indices.dim() == 2, "codebook_indices must be [B, M]"
        assert codeword_codon_logits.dim() == 4, (
            "codeword_codon_logits must be [M, K_max, 3, 4]"
        )
        B, M, D = text_part_tokens.shape
        M2, K_max = codeword_codon_logits.shape[:2]
        assert M == M2, "text clusters and codeword logits must share M codebooks"
        C = max(int(self.text_cluster_count), 1)
        if self._text_cluster_prototypes.numel() == 0:
            self._init_text_cluster_buffers(text_part_tokens, K_max)
        if (
            self._text_cluster_prototypes.shape != (M, C, D)
            or self._text_cluster_codeword.shape != (M, C, K_max)
        ):
            self._init_text_cluster_buffers(text_part_tokens, K_max)

        eps = self.eps
        tau = max(float(self.text_cluster_temperature), 1e-6)
        t_n = F.normalize(text_part_tokens, dim=-1)                    # [B, M, D]
        proto = F.normalize(self._text_cluster_prototypes.to(t_n), dim=-1)
        sim = torch.einsum("bmd,mcd->bmc", t_n, proto)                 # [B, M, C]
        q = F.softmax(sim / tau, dim=-1)                               # [B, M, C]
        conf = q.max(dim=-1).values.clamp(min=eps)                     # [B, M]
        conf_w = conf.pow(max(float(self.text_cluster_conf_gamma), 0.0))

        with torch.no_grad():
            momentum = min(max(float(self.text_cluster_ema_momentum), 0.0), 0.9999)
            # Update text prototypes with confidence-weighted soft clusters.
            mass_mc = (conf_w.unsqueeze(-1) * q).sum(dim=0).clamp(min=eps)  # [M, C]
            proto_new = torch.einsum("bmc,bmd->mcd", conf_w.unsqueeze(-1) * q, t_n)
            proto_new = proto_new / mass_mc.unsqueeze(-1)
            proto_new = F.normalize(proto_new, dim=-1)
            self._text_cluster_prototypes.mul_(momentum).add_(proto_new * (1.0 - momentum))
            self._text_cluster_prototypes.copy_(
                F.normalize(self._text_cluster_prototypes, dim=-1)
            )

            # Batch estimate of P(codeword | text-cluster), then EMA it.
            usage = torch.zeros(M, C, K_max, device=t_n.device, dtype=t_n.dtype)
            for m in range(M):
                idx_m = codebook_indices[:, m].long().clamp(min=0, max=K_max - 1)  # [B]
                src = conf_w[:, m].unsqueeze(-1) * q[:, m, :]              # [B, C]
                # usage[m, c, k] accumulates confidence-weighted membership
                # for text cluster c assigned to visual codeword k.
                assert src.shape == (B, C), "src must be [B, C]"
                for c in range(C):
                    usage[m, c].scatter_add_(dim=0, index=idx_m, src=src[:, c])
            usage = usage / usage.sum(dim=-1, keepdim=True).clamp(min=eps) # [M, C, K]
            self._text_cluster_codeword.mul_(momentum).add_(usage * (1.0 - momentum))
            self._text_cluster_codeword.div_(
                self._text_cluster_codeword.sum(dim=-1, keepdim=True).clamp(min=eps)
            )

        # codeword_codon_logits: [M, K_max, L, V=4]; |C| = V**L (=64 / 256 etc.)
        _M, _Kmax, _L, _V = codeword_codon_logits.shape
        num_codons = _V ** _L
        P_codeword = self._joint_codon_distribution(
            codeword_codon_logits, eps
        ).clamp(min=eps)                                                # [M, K_max, |C|]
        W = self._text_cluster_codeword.to(P_codeword).detach()         # [M, C, K]

        eps_reg = max(float(self.text_cluster_codon_ot_eps), 1e-3)
        n_iter = max(int(self.text_cluster_codon_ot_iters), 1)
        per_codebook = []
        device = P_codeword.device
        for m in range(M):
            K = int(codeword_K_active[m].item()) if codeword_K_active is not None else K_max
            if K <= 0:
                continue
            W_m = W[m, :, :K]
            W_m = W_m / W_m.sum(dim=-1, keepdim=True).clamp(min=eps)    # [C, K]
            P_cluster = torch.einsum("ck,kj->cj", W_m, P_codeword[m, :K])
            P_cluster = P_cluster.clamp(min=eps)
            P_cluster = P_cluster / P_cluster.sum(dim=-1, keepdim=True).clamp(min=eps)
            cost = -P_cluster.log()                                    # [C, |C|]

            log_a = torch.full((C,),          -math.log(C),          device=device, dtype=cost.dtype)
            log_b = torch.full((num_codons,), -math.log(num_codons), device=device, dtype=cost.dtype)
            log_K_mat = -cost / eps_reg                                # [C, |C|]
            u = torch.zeros(C,          device=device, dtype=cost.dtype)
            v = torch.zeros(num_codons, device=device, dtype=cost.dtype)
            for _ in range(n_iter):
                u = log_a - torch.logsumexp(log_K_mat + v[None, :], dim=1)
                v = log_b - torch.logsumexp(log_K_mat + u[:, None], dim=0)
            T = (log_K_mat + u[:, None] + v[None, :]).exp()            # [C, |C|]
            per_codebook.append((T * cost).sum() * conf[:, m].mean())

        if not per_codebook:
            zero = codeword_codon_logits.new_zeros(())
            return zero, zero, zero
        loss = torch.stack(per_codebook).mean()
        usage = q.mean(dim=0).clamp(min=eps)                            # [M, C]
        usage_entropy = (-(usage * usage.log()).sum(dim=-1) / math.log(C)).mean()
        return loss, conf.mean().detach(), usage_entropy.detach()

    def _loss_text_codon_rel(
        self,
        text_part_tokens: torch.Tensor,          # [B, M, D]
        codebook_indices: torch.Tensor,          # [B, M]
        codeword_codon_logits: torch.Tensor,     # [M, K_max, 3, 4]
        codeword_K_active: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Weak text-neighborhood -> codon-neighborhood tendency.

        For each codebook, rank batch pairs by centered text similarity.
        The top text-neighbor pairs should have assigned-codeword codon
        distributions more similar than the bottom text-distant pairs. This
        avoids hard cluster<->code assignment while still grounding codon
        usage in text semantics.
        """
        assert text_part_tokens.dim() == 3, "text_part_tokens must be [B, M, D]"
        assert codebook_indices.dim() == 2, "codebook_indices must be [B, M]"
        assert codeword_codon_logits.dim() == 4, (
            "codeword_codon_logits must be [M, K_max, 3, 4]"
        )
        B, M, _ = text_part_tokens.shape
        M2, K_max = codeword_codon_logits.shape[:2]
        assert codebook_indices.shape == (B, M), "codebook_indices must be [B, M]"
        assert M == M2, "text tokens and codeword logits must share M codebooks"
        if B <= 2:
            zero = codeword_codon_logits.new_zeros(())
            return zero, zero, zero

        eps = self.eps
        # Stop-gradient text teacher; batch-centering reduces CLIP/SigLIP
        # anisotropy before pair ranking.
        t = text_part_tokens.detach()
        t = t - t.mean(dim=0, keepdim=True)                            # [B, M, D]
        t = F.normalize(t, dim=-1)                                      # [B, M, D]
        P_full = self._joint_codon_distribution(
            codeword_codon_logits, eps
        ).clamp(min=eps)                                                # [M, K_max, 64]

        eye = torch.eye(B, device=text_part_tokens.device, dtype=torch.bool)
        top_frac = min(max(float(self.text_codon_rel_top_frac), 0.0), 1.0)
        bot_frac = min(max(float(self.text_codon_rel_bottom_frac), 0.0), 1.0)
        min_pairs = max(int(self.text_codon_rel_min_pairs), 1)
        margin = float(self.text_codon_rel_margin)

        losses = []
        pos_sims = []
        neg_sims = []
        for m in range(M):
            K = int(codeword_K_active[m].item()) if codeword_K_active is not None else K_max
            if K <= 0:
                continue
            idx = codebook_indices[:, m].long().clamp(min=0, max=K - 1)  # [B]
            P_assigned = P_full[m, idx]                                  # [B, |C|]
            _num_codons = P_full.shape[-1]
            assert P_assigned.shape == (B, _num_codons), (
                f"assigned codon dist must be [B, {_num_codons}]; got {tuple(P_assigned.shape)}"
            )

            text_sim = t[:, m, :] @ t[:, m, :].T                         # [B, B]
            codon_sim = P_assigned @ P_assigned.T                        # [B, B]
            text_flat = text_sim[~eye]                                   # [B*(B-1)]
            codon_flat = codon_sim[~eye]                                 # [B*(B-1)]
            n_pair = int(text_flat.numel())
            if n_pair <= 0:
                continue
            n_pos = min(n_pair, max(min_pairs, int(round(top_frac * n_pair))))
            n_neg = min(n_pair, max(min_pairs, int(round(bot_frac * n_pair))))
            pos_idx = torch.topk(text_flat, k=n_pos, largest=True).indices
            neg_idx = torch.topk(text_flat, k=n_neg, largest=False).indices
            pos_mean = codon_flat[pos_idx].mean()
            neg_mean = codon_flat[neg_idx].mean()
            losses.append(F.relu(codon_flat.new_tensor(margin) + neg_mean - pos_mean))
            pos_sims.append(pos_mean.detach())
            neg_sims.append(neg_mean.detach())

        if not losses:
            zero = codeword_codon_logits.new_zeros(())
            return zero, zero, zero
        loss = torch.stack(losses).mean()
        pos_mean = torch.stack(pos_sims).mean()
        neg_mean = torch.stack(neg_sims).mean()
        return loss, pos_mean, neg_mean

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

        # v91 text-to-image DNA-hash matching (None-safe). When enabled, the
        # model's forward also produced a text-derived continuous_code
        # ([B, 18, 4]) via the shared quantizer + codon_heads on
        # text_part_tokens. Two formulations:
        #   - MSE (default): F.mse_loss(text_cc, image_cc)  -- v91 original.
        #   - NtXent (v97): symmetric InfoNCE treating text_cc as an
        #     augmented view of image_cc. Each sample's (image_dna, text_dna)
        #     is a positive pair; other samples in the batch are negatives.
        #     Operates on the FLATTENED 72-dim DNA code [B, 72].
        #     Activated by --text_hash_use_ntxent.
        text_cc = outputs.get("text_continuous_code")
        # v97: text_hash MSE or NtXent (swap form via text_hash_use_ntxent)
        if text_cc is not None and self.lambda_text_hash > 0.0:
            if getattr(self, "text_hash_use_ntxent", False):
                B = text_cc.shape[0]
                t_flat = text_cc.reshape(B, -1)                          # [B, 72]
                i_flat = u.reshape(B, -1)                                # [B, 72]
                t_n = F.normalize(t_flat, dim=-1)
                i_n = F.normalize(i_flat, dim=-1)
                tau = max(float(getattr(self, "text_hash_ntxent_temperature", 0.07)), 1e-6)
                logits_it = (i_n @ t_n.T) / tau                          # [B, B]
                labels = torch.arange(B, device=logits_it.device)
                loss_text_hash = 0.5 * (
                    F.cross_entropy(logits_it,     labels) +
                    F.cross_entropy(logits_it.T,   labels)
                )
            else:
                loss_text_hash = F.mse_loss(text_cc, u)
        else:
            loss_text_hash = u.new_zeros(())

        # v100 + v128: ADDITIVE text-DNA NtXent. When lambda_text_hash_ntxent
        # > 0, compute symmetric InfoNCE between visual continuous DNA code u
        # [B, R, 4] and textual continuous DNA code text_cc [B, R, 4].
        # Two modes (controlled by --text_hash_ntxent_mode):
        #
        #   "global"       (default; v100-v125d): flatten u, text_cc to
        #       [B, R*4] and run ONE symmetric InfoNCE on the full hash.
        #
        #   "per_codebook" (v128 / contribution #2 granularity): reshape
        #       both to [B, M, L*4] (M codebooks, each a L-codon flattened
        #       DNA segment of dimension L*4), and run M INDEPENDENT
        #       symmetric InfoNCEs (one per codebook), averaged over M.
        #       Each codebook's DNA segment is supervised by text DIRECTLY
        #       to be discriminative across the batch, matching the
        #       compositional claim that each codebook encodes a distinct
        #       semantic part. R must be divisible by M (= num_codebooks).
        lam_th_nt = float(getattr(self, "lambda_text_hash_ntxent", 0.0))
        if text_cc is not None and lam_th_nt > 0.0:
            tau = max(float(getattr(self, "text_hash_ntxent_temperature", 0.07)), 1e-6)
            th_mode = str(getattr(self, "text_hash_ntxent_mode", "global"))
            B = text_cc.shape[0]
            if th_mode == "global":
                t_flat = text_cc.reshape(B, -1)
                i_flat = u.reshape(B, -1)
                t_n = F.normalize(t_flat, dim=-1)
                i_n = F.normalize(i_flat, dim=-1)
                logits_it = (i_n @ t_n.T) / tau                        # [B, B]
                labels = torch.arange(B, device=logits_it.device)
                loss_text_hash_ntxent_add = 0.5 * (
                    F.cross_entropy(logits_it,   labels) +
                    F.cross_entropy(logits_it.T, labels)
                )
            elif th_mode == "per_codebook":
                assert text_cc.shape == u.shape, (
                    f"text_hash_ntxent per_codebook expects matching shapes; "
                    f"got text={tuple(text_cc.shape)} visual={tuple(u.shape)}"
                )
                assert text_cc.dim() == 3 and text_cc.shape[-1] == 4, (
                    f"text_hash_ntxent per_codebook expects [B, R, 4]; "
                    f"got {tuple(text_cc.shape)}"
                )
                R_th = text_cc.shape[1]
                M_th = int(getattr(self, "num_codebooks", 6))
                assert R_th % M_th == 0, (
                    f"text_hash_ntxent per_codebook needs R={R_th} divisible "
                    f"by num_codebooks={M_th} (codon length L = R/M integer)."
                )
                L_th = R_th // M_th
                # [B, R, 4] -> [B, M, L, 4] -> flatten last two -> [B, M, L*4]
                t = text_cc.view(B, M_th, L_th, 4).reshape(B, M_th, L_th * 4)
                i = u      .view(B, M_th, L_th, 4).reshape(B, M_th, L_th * 4)
                t_n = F.normalize(t, dim=-1)                          # [B, M, L*4]
                i_n = F.normalize(i, dim=-1)
                # logits_it[m, i, j]: visual sample i vs text sample j for
                # codebook m. Positives = diagonal per codebook. M
                # independent CE losses, averaged.
                logits_it = torch.einsum("bmd,cmd->mbc", i_n, t_n) / tau  # [M, B, B]
                labels = torch.arange(B, device=logits_it.device)
                labels_m = labels.unsqueeze(0).expand(M_th, B).reshape(-1)
                loss_i2t = F.cross_entropy(
                    logits_it              .reshape(M_th * B, B), labels_m,
                )
                loss_t2i = F.cross_entropy(
                    logits_it.transpose(1, 2).reshape(M_th * B, B), labels_m,
                )
                loss_text_hash_ntxent_add = 0.5 * (loss_i2t + loss_t2i)
            else:
                raise ValueError(
                    f"Unsupported text_hash_ntxent_mode={th_mode!r}; "
                    "expected 'global' or 'per_codebook'."
                )
        else:
            loss_text_hash_ntxent_add = u.new_zeros(())

        # v93: per-codebook cross-modal codeword InfoNCE.
        # Symmetric InfoNCE per codebook m between
        #   visual_cw_m = outputs["quantized_tokens"][:, m, :]      [B, D]
        #   text_cw_m   = outputs["text_quantized_tokens"][:, m, :] [B, D]
        # Positive: same-sample (i, i). Negatives: other samples in batch.
        # Averaged over the 6 codebooks. STE form is used on both sides so
        # gradient flows through visual_adapter / text_adapter via the
        # quantizer's straight-through estimator.
        v_cw = outputs.get("quantized_tokens")
        t_cw = outputs.get("text_quantized_tokens")
        if (
            self.lambda_cw_xmodal > 0.0
            and v_cw is not None
            and t_cw is not None
            and v_cw.shape == t_cw.shape
            and v_cw.dim() == 3
        ):
            B_cw, M_cw, _D_cw = v_cw.shape
            tau_cw = max(float(self.cw_xmodal_temperature), 1e-6)
            v_n = F.normalize(v_cw, dim=-1)                      # [B, M, D]
            t_n = F.normalize(t_cw, dim=-1)                      # [B, M, D]
            labels_cw = torch.arange(B_cw, device=v_cw.device)
            cw_losses = []
            for m in range(M_cw):
                logits_vt = (v_n[:, m, :] @ t_n[:, m, :].t()) / tau_cw  # [B, B]
                cw_losses.append(
                    0.5 * (F.cross_entropy(logits_vt, labels_cw)
                         + F.cross_entropy(logits_vt.t(), labels_cw))
                )
            loss_cw_xmodal = torch.stack(cw_losses).mean()
        else:
            loss_cw_xmodal = u.new_zeros(())

        # v123: codeword-level text prototype alignment. Use quantizer_input
        # instead of semantic_visual_tokens when v122 residual quantization is
        # active, so the visual/text prototype spaces match.
        loss_codeword_text_proto = u.new_zeros(())
        if self.lambda_codeword_text_proto > 0.0:
            z_proto = outputs.get("quantizer_input")
            if z_proto is None:
                z_proto = outputs.get("semantic_visual_tokens")
            t_proto = outputs.get("codeword_text_tokens")
            if t_proto is None:
                t_proto = text_part_tokens
            codebook_indices_t = outputs.get("codebook_indices")
            if z_proto is None or t_proto is None or codebook_indices_t is None:
                raise ValueError(
                    "[DNACodonHashLoss] lambda_codeword_text_proto > 0 requires "
                    "outputs to contain quantizer_input/semantic_visual_tokens, "
                    "codeword_text_tokens/text_part_tokens, and codebook_indices."
                )
            loss_codeword_text_proto = self._loss_codeword_text_proto(
                z_proto, t_proto, codebook_indices_t,
                active_mask=outputs.get("codebook_active_mask"),
            )

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
                # v88: compute paired-augmentation per-codebook positive
                # alignment magnitude from both views' semantic_visual_tokens.
                # Detached scalar per (sample, codebook) cosine; the loss fn
                # averages over batch to produce A_m per codebook.
                _paired_align = None
                if float(self.ntxent_macl_alpha) > 0.0:
                    sv1 = outputs.get("semantic_visual_tokens")
                    sv2 = outputs_view2.get("semantic_visual_tokens")
                    if sv1 is not None and sv2 is not None:
                        sv1_n = F.normalize(sv1, dim=-1)
                        sv2_n = F.normalize(sv2, dim=-1)
                        _paired_align = (sv1_n * sv2_n).sum(dim=-1).detach()    # [B, M]
                loss_ntxent = self._loss_ntxent_dna_per_codebook(
                    u_st_v1, u_st_v2, temperature=self.ntxent_temperature,
                    text_part_raw=_text_anchors,
                    dynamic_tau_alpha=(
                        self.ntxent_dynamic_tau_alpha
                        if self.ntxent_dynamic_tau else 0.0
                    ),
                    skip_global_dyn=self.ntxent_dynamic_tau_skip_global,
                    global_use_local_mean=self.ntxent_global_use_local_mean,
                    variant=self.ntxent_dynamic_tau_variant,
                    model_beta=self.ntxent_dynamic_tau_model_beta,
                    model_a0=self.ntxent_dynamic_tau_model_a0,
                    model_scale_min=self.ntxent_dynamic_tau_model_scale_min,
                    model_scale_max=self.ntxent_dynamic_tau_model_scale_max,
                    semantic_scale_min=self.ntxent_dynamic_tau_semantic_scale_min,
                    semantic_scale_max=self.ntxent_dynamic_tau_semantic_scale_max,
                    hard_neg_alpha=self.ntxent_hard_neg_alpha,
                    paired_align_signal=_paired_align,
                    macl_alpha=self.ntxent_macl_alpha,
                    macl_a0=self.ntxent_macl_a0,
                )
                # v73 (Exp 7): add weak global NtXent on the full 18-codon
                # code with STATIC ntxent_temperature (no dynamic tau).
                # Re-uses _loss_ntxent_dna so the inner geometry is identical.
                if self.lambda_global_dna_ntxent > 0.0:
                    loss_global = self._loss_ntxent_dna(
                        u_st_v1, u_st_v2, temperature=self.ntxent_temperature,
                    )
                    loss_ntxent = loss_ntxent + self.lambda_global_dna_ntxent * loss_global
            else:
                loss_ntxent = self._loss_ntxent_dna(
                    u_st_v1, u_st_v2, temperature=self.ntxent_temperature,
                )
        else:
            loss_ntxent = u.new_zeros(())

        # v121: SwAV-style swapped balanced codeword-assignment loss on
        # local codebooks (paired-augmented; None-safe). Computed BEFORE the
        # codon decoder using outputs['codebook_distances'] from both views.
        # Operates on slots 1..5 by default (--swav_assign_include_global
        # flips slot 0 in).
        loss_swav_assign_v = u.new_zeros(())
        if outputs_view2 is not None and self.lambda_swav_assign > 0.0:
            _d_v1 = outputs.get("codebook_distances")
            _d_v2 = outputs_view2.get("codebook_distances")
            if _d_v1 is not None and _d_v2 is not None:
                _am = outputs.get("codebook_active_mask")
                # Prefer view1's mask if both expose it; equality not asserted.
                if _am is None and outputs_view2 is not None:
                    _am = outputs_view2.get("codebook_active_mask")
                loss_swav_assign_v = self._loss_swav_assign(
                    _d_v1, _d_v2,
                    tau_pred=self.swav_assign_tau,
                    sinkhorn_eps=self.swav_sinkhorn_eps,
                    sinkhorn_iters=self.swav_sinkhorn_iters,
                    active_mask=_am,
                    include_global=self.swav_assign_include_global,
                )

        # v119: CIBHash-style per-codebook NtXent + symmetric KL on binary
        # hash (paired-augmented; None-safe). Activates only when the
        # corresponding lambdas are >0 and the trainer forwarded a second
        # augmented view via `outputs_view2`.
        loss_cibhash_ntxent_v = u.new_zeros(())
        loss_cibhash_kl_v     = u.new_zeros(())
        if outputs_view2 is not None and (
            self.lambda_cibhash_ntxent > 0.0 or self.lambda_cibhash_kl > 0.0
        ):
            u_v1 = outputs.get("continuous_code")
            u_v2 = outputs_view2.get("continuous_code")
            if u_v1 is not None and u_v2 is not None:
                _ttp_raw = (
                    outputs.get("text_global_feat")
                    if getattr(self, "cibhash_dynamic_tau", False)
                    else None
                )
                loss_cibhash_ntxent_v, loss_cibhash_kl_v = self._loss_cibhash_per_codebook(
                    u_v1, u_v2, temperature=self.cibhash_temperature,
                    mode=getattr(self, "cibhash_mode", "per_codebook"),
                    text_part_raw=_ttp_raw,
                    dynamic_tau_alpha=(
                        float(getattr(self, "cibhash_dynamic_tau_alpha", 0.0))
                        if getattr(self, "cibhash_dynamic_tau", False) else 0.0
                    ),
                )

        # v138: prototype-cluster paired-view InfoNCE on codebook_distances.
        # Activates when --lambda_proto_cluster > 0 AND paired view is present
        # AND both views expose `codebook_distances`. Operates per codebook:
        # p_m = softmax(-codebook_distances[m] / tau)        # [B, K]
        # sim = cos(p_v1_m, p_v2_m) / tau                    # [B, B]
        # InfoNCE: diagonal positives, off-diag negatives, both directions.
        loss_proto_cluster_v = u.new_zeros(())
        if outputs_view2 is not None and self.lambda_proto_cluster > 0.0:
            d_v1 = outputs.get("codebook_distances")
            d_v2 = outputs_view2.get("codebook_distances")
            if d_v1 is not None and d_v2 is not None:
                loss_proto_cluster_v = self._loss_proto_cluster_per_codebook(
                    d_v1, d_v2, temperature=self.proto_cluster_temperature,
                )

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

        # v79a (#1.2): cross-codebook orthogonality on z (has grad via
        # visual_adapter). Codebook follows z via EMA, so pushing z's
        # per-codebook batch means apart indirectly drives codeword
        # orthogonality without trying to gradient-train an EMA buffer.
        if self.lambda_codebook_ortho > 0.0:
            loss_codebook_ortho = self._loss_codebook_ortho(z)
        else:
            loss_codebook_ortho = u.new_zeros(())

        # v66: per-codon text-anchored aux CE loss (model computes per forward)
        loss_text_anchor = outputs.get("loss_text_anchor")
        if loss_text_anchor is None:
            loss_text_anchor = u.new_zeros(())

        # v70a (Exp 3): hash reconstruction loss
        # 1 - cos(decoder(hash_st), target.detach())
        loss_hash_recon = u.new_zeros(())
        if self.lambda_hash_recon > 0.0:
            pred = outputs.get("hash_recon_pred")
            if pred is not None:
                # build target
                tgt_v = outputs.get("visual_global_feat")
                tgt_t = outputs.get("text_global_feat")        # [B, 6, D] or None
                terms = []
                if self.hash_recon_target_kind in ("siglip_visual", "both") and tgt_v is not None:
                    terms.append(
                        1.0 - F.cosine_similarity(pred, tgt_v.detach(), dim=-1).mean()
                    )
                if self.hash_recon_target_kind in ("text_global", "both") and tgt_t is not None:
                    # use mean of per-slot text features (or just slot 0)
                    t_pool = tgt_t.mean(dim=1) if tgt_t.dim() == 3 else tgt_t
                    terms.append(
                        1.0 - F.cosine_similarity(pred, t_pool.detach(), dim=-1).mean()
                    )
                if terms:
                    loss_hash_recon = sum(terms) / len(terms)

        # v72a (Exp 6): dual projection auxiliary losses
        # semantic = 1 - cos(semantic_proj, text_global or visual_global.detach())
        # instance = NtXent on instance_proj between two paired-aug views
        loss_dual_semantic = u.new_zeros(())
        loss_dual_instance = u.new_zeros(())
        if self.lambda_dual_semantic > 0.0:
            sem_pred = outputs.get("dual_hash_semantic")
            if sem_pred is not None:
                if self.dual_hash_proj_target_kind == "siglip_visual":
                    tgt = outputs.get("visual_global_feat")
                else:  # text_global
                    tgt_t = outputs.get("text_global_feat")
                    tgt = tgt_t.mean(dim=1) if (tgt_t is not None and tgt_t.dim() == 3) else tgt_t
                if tgt is not None:
                    loss_dual_semantic = (
                        1.0 - F.cosine_similarity(sem_pred, tgt.detach(), dim=-1).mean()
                    )
        if self.lambda_dual_instance > 0.0 and outputs_view2 is not None:
            ins1 = outputs.get("dual_hash_instance")
            ins2 = outputs_view2.get("dual_hash_instance")
            if ins1 is not None and ins2 is not None:
                # standard NtXent over batch
                z1 = F.normalize(ins1, dim=-1)
                z2 = F.normalize(ins2, dim=-1)
                z = torch.cat([z1, z2], dim=0)                  # [2B, D]
                sim = z @ z.t()                                  # [2B, 2B]
                Nz = z.shape[0]
                eye_d = torch.eye(Nz, device=z.device, dtype=torch.bool)
                sim = (sim / float(self.dual_hash_proj_ntxent_tau)).masked_fill(eye_d, -1e9)
                Bd = z1.shape[0]
                pos_idx = torch.cat([
                    torch.arange(Bd, 2 * Bd, device=z.device),
                    torch.arange(0, Bd,     device=z.device),
                ])
                loss_dual_instance = F.cross_entropy(sim, pos_idx)

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
            + self.lambda_hash_recon      * loss_hash_recon
            + self.lambda_dual_semantic   * loss_dual_semantic
            + self.lambda_dual_instance   * loss_dual_instance
            + self.lambda_codebook_ortho  * loss_codebook_ortho
            + self.lambda_text_hash       * loss_text_hash
            + self.lambda_text_hash_ntxent * loss_text_hash_ntxent_add
            + self.lambda_cw_xmodal       * loss_cw_xmodal
            + self.lambda_codeword_text_proto * loss_codeword_text_proto
            + self.lambda_cibhash_ntxent  * loss_cibhash_ntxent_v
            + self.lambda_cibhash_kl      * loss_cibhash_kl_v
            + self.lambda_swav_assign     * loss_swav_assign_v
            + self.lambda_proto_cluster   * loss_proto_cluster_v
        )

        # v106: codeword <-> DNA codon bijection losses (None-safe).
        # Operate on the codeword-level codon decoder outputs supplied by
        # the model when at least one bijection lambda > 0.
        codeword_codon_logits = outputs.get("codeword_codon_logits")
        codeword_K_active     = outputs.get("codeword_K_active")
        eff_lambda_codeword_codon_sinkhorn = self._effective_codeword_codon_sinkhorn_lambda(epoch)
        if codeword_codon_logits is not None:
            assert codeword_codon_logits.dim() == 4, (
                "codeword_codon_logits must be [M, K_max, 3, 4]"
            )
            if eff_lambda_codeword_codon_sinkhorn > 0.0:
                loss_codeword_codon_sinkhorn = self._loss_codeword_codon_sinkhorn(
                    codeword_codon_logits, codeword_K_active,
                )
                total = total + eff_lambda_codeword_codon_sinkhorn * loss_codeword_codon_sinkhorn
            else:
                loss_codeword_codon_sinkhorn = u.new_zeros(())
            if self.lambda_codeword_codon_agg_ent > 0.0:
                loss_codeword_codon_agg_ent = self._loss_codeword_codon_agg_ent(
                    codeword_codon_logits, codeword_K_active,
                )
                total = total + self.lambda_codeword_codon_agg_ent * loss_codeword_codon_agg_ent
            else:
                loss_codeword_codon_agg_ent = u.new_zeros(())
            if self.lambda_codeword_codon_pairwise > 0.0:
                loss_codeword_codon_pairwise = self._loss_codeword_codon_pairwise(
                    codeword_codon_logits, codeword_K_active,
                )
                total = total + self.lambda_codeword_codon_pairwise * loss_codeword_codon_pairwise
            else:
                loss_codeword_codon_pairwise = u.new_zeros(())
            if self.lambda_text_cluster_codon_ot > 0.0:
                codebook_indices_t = outputs.get("codebook_indices")       # [B, M]
                if text_part_tokens is None or codebook_indices_t is None:
                    raise ValueError(
                        "[DNACodonHashLoss] lambda_text_cluster_codon_ot > 0 requires "
                        "outputs to contain 'text_part_tokens' and 'codebook_indices'."
                    )
                loss_text_cluster_codon_ot, text_cluster_conf_mean, text_cluster_usage_entropy = (
                    self._loss_text_cluster_codon_ot(
                        text_part_tokens, codebook_indices_t,
                        codeword_codon_logits, codeword_K_active,
                    )
                )
                total = total + self.lambda_text_cluster_codon_ot * loss_text_cluster_codon_ot
            else:
                loss_text_cluster_codon_ot = u.new_zeros(())
                text_cluster_conf_mean = u.new_zeros(())
                text_cluster_usage_entropy = u.new_zeros(())
            if self.lambda_text_codon_rel > 0.0:
                codebook_indices_t = outputs.get("codebook_indices")       # [B, M]
                if text_part_tokens is None or codebook_indices_t is None:
                    raise ValueError(
                        "[DNACodonHashLoss] lambda_text_codon_rel > 0 requires "
                        "outputs to contain 'text_part_tokens' and 'codebook_indices'."
                    )
                loss_text_codon_rel, text_codon_rel_pos_sim, text_codon_rel_neg_sim = (
                    self._loss_text_codon_rel(
                        text_part_tokens, codebook_indices_t,
                        codeword_codon_logits, codeword_K_active,
                    )
                )
                total = total + self.lambda_text_codon_rel * loss_text_codon_rel
            else:
                loss_text_codon_rel = u.new_zeros(())
                text_codon_rel_pos_sim = u.new_zeros(())
                text_codon_rel_neg_sim = u.new_zeros(())
        else:
            loss_codeword_codon_sinkhorn = u.new_zeros(())
            loss_codeword_codon_agg_ent  = u.new_zeros(())
            loss_codeword_codon_pairwise = u.new_zeros(())
            loss_text_cluster_codon_ot = u.new_zeros(())
            text_cluster_conf_mean = u.new_zeros(())
            text_cluster_usage_entropy = u.new_zeros(())
            loss_text_codon_rel = u.new_zeros(())
            text_codon_rel_pos_sim = u.new_zeros(())
            text_codon_rel_neg_sim = u.new_zeros(())
            eff_lambda_codeword_codon_sinkhorn = 0.0

        # v112: hierarchical codon decomposition loss (uses codeword_codon_logits)
        if (
            self.lambda_hierarchical_cluster_codon > 0.0
            and outputs.get("codeword_codon_logits") is not None
        ):
            codeword_codon_logits = outputs["codeword_codon_logits"]
            codeword_K_active = outputs.get("codeword_K_active")
            codebook_active_mask = outputs.get("codebook_active_mask")
            loss_hierarchical_cluster_codon = self._loss_hierarchical_cluster_codon(
                codeword_codon_logits, codeword_K_active, codebook_active_mask,
            )
            total = total + self.lambda_hierarchical_cluster_codon * loss_hierarchical_cluster_codon
        else:
            loss_hierarchical_cluster_codon = u.new_zeros(())

        # v107: prototype cosine clustering (InfoNCE between z and codewords)
        if self.lambda_proto_cluster_cos > 0.0:
            codebook_indices_t   = outputs.get("codebook_indices")           # [B, M]
            codebooks_buffer     = outputs.get("codebooks_buffer")           # [M, K_max, D]
            codebook_active_mask = outputs.get("codebook_active_mask")       # [M, K_max] bool
            if codebook_indices_t is None or codebooks_buffer is None:
                raise ValueError(
                    "[DNACodonHashLoss] lambda_proto_cluster_cos > 0 requires "
                    "outputs to contain 'codebook_indices' and 'codebooks_buffer'."
                )
            loss_proto_cluster_cos = self._loss_proto_cluster_cos(
                z, codebooks_buffer, codebook_indices_t, codebook_active_mask,
            )
            total = total + self.lambda_proto_cluster_cos * loss_proto_cluster_cos
        else:
            loss_proto_cluster_cos = u.new_zeros(())

        return {
            "loss":              total,
            "loss_hash":         loss_hash,
            "loss_hash_hard":    loss_hash_hard,
            "loss_vq":           loss_vq,
            "loss_quant":        loss_quant,
            "loss_anchor":       loss_anchor,
            "loss_wasserstein":  loss_wasserstein,
            "loss_text_hash":    loss_text_hash,
            "loss_text_hash_ntxent_add": loss_text_hash_ntxent_add,
            "loss_cw_xmodal":    loss_cw_xmodal,
            "loss_codeword_text_proto": loss_codeword_text_proto,
            "loss_codeword_codon_sinkhorn": loss_codeword_codon_sinkhorn,
            "loss_codeword_codon_agg_ent":  loss_codeword_codon_agg_ent,
            "loss_codeword_codon_pairwise": loss_codeword_codon_pairwise,
            "loss_text_cluster_codon_ot":   loss_text_cluster_codon_ot,
            "text_cluster_conf_mean":       text_cluster_conf_mean,
            "text_cluster_usage_entropy":   text_cluster_usage_entropy,
            "loss_text_codon_rel":          loss_text_codon_rel,
            "text_codon_rel_pos_sim":       text_codon_rel_pos_sim,
            "text_codon_rel_neg_sim":       text_codon_rel_neg_sim,
            "eff_lambda_codeword_codon_sinkhorn": u.new_tensor(eff_lambda_codeword_codon_sinkhorn),
            "loss_proto_cluster_cos":       loss_proto_cluster_cos,
            "loss_hierarchical_cluster_codon": loss_hierarchical_cluster_codon,
            "loss_recon":        loss_recon,
            "loss_ntxent":       loss_ntxent,
            "loss_cibhash_ntxent": loss_cibhash_ntxent_v,
            "loss_cibhash_kl":     loss_cibhash_kl_v,
            "loss_proto_cluster":  loss_proto_cluster_v,
            "loss_swav_assign":    loss_swav_assign_v,
            "loss_ortho_text":   loss_ortho_text,
            "loss_dna":          loss_dna,
            "loss_bu":           loss_bu,
            "loss_entropy":      dna_components["loss_entropy"],
            "loss_base_balance": dna_components["loss_base_balance"],
            "loss_cb_balance":   bu_components ["loss_cb_balance"],
            "loss_cb_uncorr":    bu_components ["loss_cb_uncorr"],
            "loss_codon_text_anchor": loss_text_anchor,
            "loss_hash_recon":        loss_hash_recon,
            "loss_dual_semantic":     loss_dual_semantic,
            "loss_dual_instance":     loss_dual_instance,
            "loss_codebook_ortho":    loss_codebook_ortho,
        }
