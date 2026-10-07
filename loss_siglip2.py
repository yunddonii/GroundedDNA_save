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

import numpy as np
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
        self.num_codebooks = int(getattr(cfg, "num_codebooks", 6))

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
        self.cibhash_dynamic_tau_skip_global = bool(
            getattr(cfg, "cibhash_dynamic_tau_skip_global", False)
        )
        # v121: SwAV-style swapped balanced codeword-assignment loss.
        self.lambda_swav_assign         = float(getattr(cfg, "lambda_swav_assign", 0.0))
        self.swav_assign_tau            = float(getattr(cfg, "swav_assign_tau",    0.1))
        self.swav_sinkhorn_eps          = float(getattr(cfg, "swav_sinkhorn_eps",  0.05))
        self.swav_sinkhorn_iters        = int(getattr(cfg, "swav_sinkhorn_iters",  3))
        self.swav_assign_include_global = bool(getattr(cfg, "swav_assign_include_global", False))
        self.cibhash_dynamic_tau_alpha  = float(getattr(cfg, "cibhash_dynamic_tau_alpha", 0.0))
        # v149: continuous NtXent flag (skip STE-sign, use shifted bit_probs).
        self.cibhash_ntxent_continuous  = bool(getattr(cfg, "cibhash_ntxent_continuous", False))
        # v150: cibhash NtXent input source -- "continuous_code" (legacy) or
        # "visual_token" (pre-VQ semantic_visual_tokens [B, M, D]).
        self.cibhash_ntxent_source = str(getattr(cfg, "cibhash_ntxent_source", "continuous_code"))
        self.cibhash_visual_token_bit_kl = bool(
            getattr(cfg, "cibhash_visual_token_bit_kl", False)
        )
        # v138: prototype-cluster paired-view InfoNCE on codebook_distances.
        self.lambda_proto_cluster        = float(getattr(cfg, "lambda_proto_cluster",        0.0))
        self.proto_cluster_temperature   = float(getattr(cfg, "proto_cluster_temperature",   0.3))
        # v144: text -> code KL distillation hyperparameters.
        self.lambda_text_code_kl           = float(getattr(cfg, "lambda_text_code_kl",           0.0))
        self.text_code_kl_tau_v            = float(getattr(cfg, "text_code_kl_tau_v",            0.1))
        self.text_code_kl_tau_t            = float(getattr(cfg, "text_code_kl_tau_t",            0.07))
        self.text_code_kl_conf_threshold   = float(getattr(cfg, "text_code_kl_conf_threshold",   0.0))
        self.text_code_kl_skip_global      = bool(getattr(cfg, "text_code_kl_skip_global",       False))
        # v172: routing-text per-patch supervision
        self.lambda_routing_text           = float(getattr(cfg, "lambda_routing_text",           0.0))
        self.routing_text_tau              = float(getattr(cfg, "routing_text_tau",              0.1))
        self.routing_text_skip_global      = bool(getattr(cfg, "routing_text_skip_global",       False))
        # v173: text-codeword contrastive (RAW text vs quantized visual codeword)
        self.lambda_text_codeword_contrastive       = float(getattr(cfg, "lambda_text_codeword_contrastive",       0.0))
        self.text_codeword_contrastive_tau          = float(getattr(cfg, "text_codeword_contrastive_tau",          0.07))
        self.text_codeword_contrastive_skip_global  = bool(getattr(cfg, "text_codeword_contrastive_skip_global",   False))
        # v174 Option alpha: text vs PRE-QUANT semantic_visual_tokens contrastive
        self.lambda_text_preq_contrastive           = float(getattr(cfg, "lambda_text_preq_contrastive",           0.0))
        self.text_preq_contrastive_tau              = float(getattr(cfg, "text_preq_contrastive_tau",              0.07))
        self.text_preq_contrastive_skip_global      = bool(getattr(cfg, "text_preq_contrastive_skip_global",       False))
        # v174 Option gamma: hash-code text<->visual InfoNCE
        self.lambda_text_visual_hash_contrastive    = float(getattr(cfg, "lambda_text_visual_hash_contrastive",    0.0))
        self.text_visual_hash_contrastive_tau       = float(getattr(cfg, "text_visual_hash_contrastive_tau",       0.07))
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
        # Counterfactual minimal-pair extension. This is folded into the
        # visual->text InfoNCE denominator rather than added as a separate
        # TripletLoss, so rho=0 exactly recovers the legacy objective.
        self.text_hash_counterfactual_weight = float(
            getattr(cfg, "text_hash_counterfactual_weight", 0.0)
        )
        self.text_hash_counterfactual_margin = float(
            getattr(cfg, "text_hash_counterfactual_margin", 0.02)
        )
        self.text_hash_counterfactual_warmup_epochs = int(
            getattr(cfg, "text_hash_counterfactual_warmup_epochs", 0)
        )
        if self.text_hash_counterfactual_weight < 0.0:
            raise ValueError("text_hash_counterfactual_weight must be >= 0")
        if self.text_hash_counterfactual_margin < 0.0:
            raise ValueError("text_hash_counterfactual_margin must be >= 0")
        if self.text_hash_counterfactual_warmup_epochs < 0:
            raise ValueError("text_hash_counterfactual_warmup_epochs must be >= 0")
        if (
            self.text_hash_counterfactual_weight > 0.0
            and self.lambda_text_hash_ntxent <= 0.0
        ):
            raise ValueError(
                "counterfactual text-DNA negatives require "
                "--lambda_text_hash_ntxent > 0"
            )
        if (
            self.text_hash_counterfactual_weight > 0.0
            and self.text_hash_ntxent_mode != "per_codebook"
        ):
            raise ValueError(
                "counterfactual text-DNA negatives require "
                "--text_hash_ntxent_mode per_codebook"
            )
        if (
            self.text_hash_counterfactual_weight > 0.0
            and not bool(getattr(cfg, "text_hash_ntxent_skip_global", False))
        ):
            raise ValueError(
                "counterfactual text-DNA training is local-only; enable "
                "--text_hash_ntxent_skip_global"
            )
        # v93: per-codebook cross-modal codeword InfoNCE. Symmetric InfoNCE
        # between visual `quantized_tokens[:, m, :]` and text
        # `text_quantized_tokens[:, m, :]` for each codebook m.
        # Positive: same sample; negatives: other samples in the batch.
        # 0 = disabled (legacy bit-exact).
        self.lambda_cw_xmodal      = float(getattr(cfg, "lambda_cw_xmodal", 0.0))
        self.cw_xmodal_temperature = float(getattr(cfg, "cw_xmodal_temperature", 0.07))
        # v160 (Uni-Code Eq.8): cross-modal commitment weight (= beta/2 in paper).
        self.lambda_xmodal_commit  = float(getattr(cfg, "lambda_xmodal_commit", 0.0))
        # v176: skip cb0 (C_global) options for xmodal_commit + text_hash_ntxent
        self.xmodal_commit_skip_global    = bool(getattr(cfg, "xmodal_commit_skip_global",    False))
        self.text_hash_ntxent_skip_global = bool(getattr(cfg, "text_hash_ntxent_skip_global", False))
        # TD (2026-10-07): text-dropout path consistency. KL between the
        # caption-routed codeword assignment distribution (teacher, detached)
        # and the deployment-routed (no-text) one, both from codebook_distances.
        self.lambda_path_consistency         = float(getattr(cfg, "lambda_path_consistency", 0.0))
        self.path_consistency_tau            = float(getattr(cfg, "path_consistency_tau", 0.1))
        self.path_consistency_include_global = bool(getattr(cfg, "path_consistency_include_global", False))
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
        # (b) SDC-style similarity-spread calibration (default OFF).
        # Motivation (2026-07-27 root-cause): on MSCOCO the two baselines that
        # beat us (CroVCA, SDC) are pure code-RESOLUTION maximisers, and our
        # code is resolution-poor (DB DNA-unique 0.19 = 5.3 items/code). SDC's
        # published contribution is anti-similarity-collapse: fit the code
        # cosine distribution onto a wide symmetric Beta target while
        # preserving the frozen-teacher similarity RANK ORDER.
        self.lambda_sim_spread = float(getattr(cfg, "lambda_sim_spread", 0.0))
        self.sim_spread_beta   = float(getattr(cfg, "sim_spread_beta", 5.0))
        self.sim_spread_pairs  = int(getattr(cfg, "sim_spread_pairs", 4096))
        # metric: "cosine" = literal SDC port (valid for +-1 binary codes);
        # "base_match" = the 4-ary DNA adaptation. Our retrieval distance is
        # base Hamming over R positions of a 4-way softmax codon, so cosine on
        # flattened logits does NOT map linearly to it (SDC Eq.2 assumes +-1
        # bits). base_match uses the differentiable expected base-agreement
        # s_ij = mean_r <p_i[r], p_j[r]> in [0,1], which IS 1 - E[baseHamming]/R.
        self.sim_spread_metric = str(getattr(cfg, "sim_spread_metric", "cosine"))
        # ---- constraint-AWARE training (2026-07-29), default OFF -------------
        # Until now GC / homopolymer were enforced ONLY by the post-hoc DP
        # projection (dna_utils/bio_constraints.py) -- the training objective
        # never saw them, so "DNA is load-bearing" was unsupported. These two
        # differentiable surrogates put the same two constraints in the loss.
        self.lambda_bio_constraint    = float(getattr(cfg, "lambda_bio_constraint", 0.0))
        self.bio_constraint_gc_min    = getattr(cfg, "bio_constraint_gc_min", None)
        self.bio_constraint_gc_max    = getattr(cfg, "bio_constraint_gc_max", None)
        self.bio_constraint_max_run   = int(getattr(cfg, "bio_constraint_max_run", 3))
        self.bio_constraint_gc_weight = float(getattr(cfg, "bio_constraint_gc_weight", 1.0))
        self.bio_constraint_hp_weight = float(getattr(cfg, "bio_constraint_hp_weight", 1.0))
        # ---- joint codon-diversity regulariser (2026-07-30), default OFF ----
        # `loss_base_balance` is KL(uniform || mean_b u) computed PER POSITION,
        # so it is structurally blind to a slot whose three positions have fine
        # marginals but a collapsed JOINT. Measured on the MSCOCO champion:
        # slot0 reaches 21/64 codons where its own marginals allow 48.6, while
        # slots 1-5 sit at their marginal budget. This term regularises the
        # per-slot batch-mean distribution over the 4**L codons directly.
        self.lambda_codon_joint = float(getattr(cfg, "lambda_codon_joint", 0.0))
        # 2026-09-20, off-protocol branch arch-exp-2026-09 (default off):
        # (c-1) no codebook-KL while the quantizer is bypassed (the codebook is
        # still random then); (b-1) weight of the model's masked-entity loss.
        self.vq_bypass_epochs = int(getattr(cfg, "vq_bypass_epochs", 0) or 0)
        self.lambda_mec = float(getattr(cfg, "lambda_mec", 0.0) or 0.0)
        # arch-exp-3 (P4)
        self.lambda_role = float(getattr(cfg, "lambda_role", 0.0) or 0.0)
        self.role_tau    = float(getattr(cfg, "role_tau", 0.07) or 0.07)
        self.role_source = str(getattr(cfg, "role_source", "quantized") or "quantized")
        # (stage 7) local-slot target of the visual-token NT-Xent, and the caption-
        # concept cross-entropy of the named concept codebook.
        self.cibhash_local_target = str(getattr(cfg, "cibhash_local_target", "instance") or "instance")
        self.cibhash_local_target_tau = float(getattr(cfg, "cibhash_local_target_tau", 0.2))
        self.text_hash_ntxent_target = str(getattr(cfg, "text_hash_ntxent_target", "instance") or "instance")
        self.text_hash_ntxent_target_tau = float(getattr(cfg, "text_hash_ntxent_target_tau", 0.2))
        self._axis_mean_ema = None                  # per-axis caption mean, EMA over batches
        self.cibhash_local_queue = int(getattr(cfg, "cibhash_local_queue", 0) or 0)
        self._q_v = self._q_t = None                # (stage 8, B2) FIFO of past slot tokens / captions
        self._q_n = self._q_ptr = 0
        self.lambda_concept = float(getattr(cfg, "lambda_concept", 0.0) or 0.0)
        self.concept_tau = float(getattr(cfg, "concept_tau", 0.5))
        self._concept_centers = self._concept_axis_mean = None
        if self.lambda_concept > 0.0:
            _cc = str(getattr(cfg, "concept_codebook_npz", "") or "")
            if not _cc:
                raise ValueError("--lambda_concept needs --concept_codebook_npz")
            _z = np.load(_cc)
            self._concept_centers = torch.as_tensor(_z["centers"], dtype=torch.float32)       # [4, 64, D]
            self._concept_axis_mean = torch.as_tensor(_z["axis_mean"], dtype=torch.float32)  # [5, D]
        # ConceptHash Eq. 8 (`L_csd`) ported onto the routing matrix.
        self.lambda_slot_diversity      = float(getattr(cfg, "lambda_slot_diversity", 0.0))
        self.slot_diversity_skip_global = bool(getattr(cfg, "slot_diversity_skip_global", True))
        self.codon_joint_floor  = float(getattr(cfg, "codon_joint_floor", 1e-6))
        _cjs = str(getattr(cfg, "codon_joint_slots", "") or "").strip()
        self.codon_joint_slots = (
            sorted({int(x) for x in _cjs.split(",") if x.strip() != ""}) if _cjs else None)
        self.num_codons_per_codebook = int(
            getattr(cfg, "num_codons_per_codebook", 3) or 3)
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
        # Validation is observational.  It may consume a differently-sized
        # last batch (or be the first loss call after resume), but it must not
        # initialise, resize, or advance the persistent training EMA.  Reuse a
        # compatible training anchor when one exists; otherwise the current
        # batch is an ephemeral detached target and never enters state_dict().
        if not self.training:
            if self.ema_text_anchor.numel() > 0 \
                    and self.ema_text_anchor.shape == batch_anchor.shape:
                return self.ema_text_anchor
            return batch_anchor.detach()

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


    def _centred_axis_text(self, t_raw):
        """(stage 7, B) per-axis caption embeddings minus an EMA of their mean, unit norm."""
        if t_raw is None:
            raise ValueError("--cibhash_local_target axis_soft needs the cached caption embeddings")
        t = t_raw.detach().float()
        with torch.no_grad():
            if self._axis_mean_ema is None:
                self._axis_mean_ema = t.mean(dim=0)
            else:
                self._axis_mean_ema = 0.99 * self._axis_mean_ema.to(t.device) + 0.01 * t.mean(dim=0)
        return F.normalize(t - self._axis_mean_ema.to(t.device), dim=-1)

    @torch.no_grad()
    def _enqueue(self, v, t):
        """(stage 8, B2) FIFO of normalised slot tokens [B, M, D] and centred captions [B, M, Dt]."""
        Q = self.cibhash_local_queue
        if self._q_v is None:
            self._q_v = v.new_zeros((Q,) + tuple(v.shape[1:]))
            self._q_t = t.new_zeros((Q,) + tuple(t.shape[1:]))
        n = min(v.shape[0], Q)
        idx = (self._q_ptr + torch.arange(n, device=v.device)) % Q
        self._q_v[idx] = v[:n].to(self._q_v.dtype)
        self._q_t[idx] = t[:n].to(self._q_t.dtype)
        self._q_ptr = int((self._q_ptr + n) % Q)
        self._q_n = min(self._q_n + n, Q)

    def _loss_concept(self, t_raw, distances):
        """(stage 7, A) local slot m must pick the codeword of its own caption concept.

        The concept of a caption is its nearest k-means centre (per-axis-centred cache
        embedding, the space the centres were fitted in). Logits are standardised
        negative codeword distances, the same quantity the concept codon uses."""
        if t_raw is None or distances is None:
            return None
        t = t_raw.detach().float()[:, 1:, :]                                   # [B, 4, D]
        mu = self._concept_axis_mean.to(t.device)[1:]                          # [4, D]
        c = self._concept_centers.to(t.device)                                 # [4, 64, D]
        y = torch.cdist((t - mu).transpose(0, 1), c).argmin(dim=-1).transpose(0, 1)  # [B, 4]
        valid = t.norm(dim=-1) > 0                                              # captionless rows
        d = distances[:, 1:, :64].float()
        z = (d - d.mean(-1, keepdim=True)) / d.std(-1, keepdim=True).clamp_min(1e-6)
        logits = (-z / self.concept_tau)
        if not bool(valid.any()):
            return None
        return F.cross_entropy(logits[valid], y[valid])

    def _loss_role_axis(self, q_local, t_local):
        """(arch-exp-3 P4) In-image cross-axis alignment.

        ``q_local`` [B, M, D] are the local slot codes (straight-through, so the
        gradient reaches the encoder) and ``t_local`` [B, M, D] the caption
        embedding of each axis OF THE SAME IMAGE. For every image the M x M
        cosine matrix is read as a two-way classification whose negatives are
        the other axes of that image -- the comparison no other active term
        makes. The text side is detached, exactly as ``text_code_kl`` detaches
        its text view, so the pressure falls on the code.

        Both sides are centred per column over the batch first. Without that, a
        slot could win every softmax by emitting one constant direction per axis
        (the axis means are far apart after the per-slot adapter); after it, only
        this image's deviation from its axis mean carries any signal.
        """
        if q_local is None or t_local is None:
            return None
        if q_local.dim() != 3 or t_local.shape != q_local.shape:
            return None
        B, M, _ = q_local.shape
        if B < 4 or M < 2:
            return None
        q = q_local.float()
        t = t_local.float().detach()
        q = q - q.mean(dim=0, keepdim=True)
        t = t - t.mean(dim=0, keepdim=True)
        q = F.normalize(q, dim=-1)
        t = F.normalize(t, dim=-1)
        tau = max(float(self.role_tau), 1e-6)
        S = torch.einsum("bmd,bad->bma", q, t) / tau          # [B, code m, axis a]
        tgt = torch.arange(M, device=S.device).expand(B, M).reshape(-1)
        l_code = F.cross_entropy(S.reshape(-1, M), tgt)
        l_axis = F.cross_entropy(S.transpose(1, 2).reshape(-1, M), tgt)
        return 0.5 * (l_code + l_axis)

    def _loss_text_code_kl_per_codebook(
        self,
        z_visual: torch.Tensor,           # [B, M, D]  semantic_visual_tokens (pre-VQ)
        z_text:   torch.Tensor,           # [B, M, D]  text_part_tokens
        codebooks: torch.Tensor,          # [M, K, D]  full codebook tensor
        codebook_active_mask: Optional[torch.Tensor] = None,  # [M, K] bool
    ) -> torch.Tensor:
        """v144: per-codebook text -> code KL distillation with confidence
        weighting.

        For each codebook m (local codebooks only when
        ``text_code_kl_skip_global`` is set), build K-way categorical
        distributions over the codewords for both visual and text views:

            logits_v[b, k] = cos(z_visual[b, m], C_m[k]) / tau_v
            logits_t[b, k] = cos(z_text  [b, m], C_m[k]) / tau_t

        Minimize KL(p_t.detach() || p_v) per sample, weighted by the per-
        sample text confidence
            conf[b] = 1 - H(p_t[b]) / log(K)
        with optional threshold filtering (``conf_threshold``). Samples
        whose text distribution is too uncertain (e.g. ambiguous caption)
        are dropped, preventing them from injecting noise into the
        codebook supervision signal.

        Gradient flows through z_visual (encoder + visual_adapter) only;
        z_text is detached via p_t. The codebook ``C`` is a buffer in EMA
        mode so receives no gradient (codebook updates via its own EMA).
        """
        assert z_visual.dim() == 3 and z_text.dim() == 3, (
            f"z_visual / z_text must be [B, M, D]"
        )
        assert codebooks.dim() == 3, (
            f"codebooks must be [M, K, D], got {tuple(codebooks.shape)}"
        )
        B, M, D = z_visual.shape
        M_cb, K_cb, D_cb = codebooks.shape
        assert M == M_cb and D == D_cb, (
            f"shape mismatch: z_visual {tuple(z_visual.shape)} vs "
            f"codebooks {tuple(codebooks.shape)}"
        )
        tau_v = max(float(self.text_code_kl_tau_v), 1e-6)
        tau_t = max(float(self.text_code_kl_tau_t), 1e-6)
        conf_thresh = float(self.text_code_kl_conf_threshold)
        skip_global = bool(self.text_code_kl_skip_global)

        # L2-normalize for cosine geometry.
        z_v_n = F.normalize(z_visual, dim=-1)
        z_t_n = F.normalize(z_text,   dim=-1)
        cb_n  = F.normalize(codebooks, dim=-1)

        log_K = math.log(float(K_cb))
        start_m = 1 if skip_global else 0
        losses: list = []
        for m in range(start_m, M):
            C_m = cb_n[m]                                            # [K, D]
            # Cosine similarity logits.
            logits_v = z_v_n[:, m, :] @ C_m.T / tau_v                # [B, K]
            logits_t = z_t_n[:, m, :] @ C_m.T / tau_t                # [B, K]

            if codebook_active_mask is not None:
                # Mask inactive codewords by setting logits to -inf so
                # they get ~0 probability after softmax.
                active = codebook_active_mask[m].to(dtype=torch.bool)  # [K]
                inactive_logit = torch.full_like(logits_v, -1e9)
                logits_v = torch.where(active.unsqueeze(0), logits_v, inactive_logit)
                logits_t = torch.where(active.unsqueeze(0), logits_t, inactive_logit)

            p_v_log = F.log_softmax(logits_v, dim=-1)                # [B, K]
            p_t     = F.softmax    (logits_t, dim=-1)                # [B, K]

            # Per-sample text entropy and confidence.
            with torch.no_grad():
                entropy_t = -(p_t * (p_t.clamp_min(1e-8)).log()).sum(dim=-1)  # [B]
                conf      = (1.0 - entropy_t / log_K).clamp(0.0, 1.0)         # [B]
                if conf_thresh > 0.0:
                    mask = conf > conf_thresh                                  # [B]
                    conf = torch.where(mask, conf, torch.zeros_like(conf))

            # KL(p_t.detach() || p_v) per sample.
            kl_per_sample = F.kl_div(
                p_v_log, p_t.detach(), reduction="none",
            ).sum(dim=-1)                                                       # [B]
            weighted = kl_per_sample * conf.detach()                            # [B]

            # Mean over samples with non-zero confidence (so dropped samples
            # don't shrink the gradient).
            if conf_thresh > 0.0:
                denom = (conf > 0.0).to(weighted.dtype).sum().clamp_min(1.0)
                losses.append(weighted.sum() / denom)
            else:
                losses.append(weighted.mean())

        if not losses:
            return z_visual.new_zeros(())
        return torch.stack(losses).mean()

    def _loss_routing_text_per_codebook(
        self,
        visual_tokens:    torch.Tensor,    # [B, P, D] post-adapter pre-routing
        text_part_tokens: torch.Tensor,    # [B, M, D] per-slot text (post-adapter)
        routing_matrix:   torch.Tensor,    # [B, M, P] Sinkhorn assignment weights
        tau: float = 0.1,
        skip_global: bool = True,
    ) -> torch.Tensor:
        """v172: per-patch routing supervision via text similarity.

        For each (image i, codebook m, patch p):
            sim[i, m, p] = cos(text_part[i, m], visual_token[i, p])
            target[i, m, p] = softmax_p(sim / tau)
        Loss: KL(target.detach() || routing_matrix) averaged over (B, M_local).

        This closes the gap that text_code_kl supervises codeword-INDEX
        distribution but not which patches feed each codebook. Gradient
        flows through routing_matrix back to Sinkhorn cost, encoder
        + visual_adapter. text_part_tokens are detached (text path
        supervised by other text-side losses).
        """
        assert visual_tokens.dim() == 3, (
            f"visual_tokens must be [B, P, D], got {tuple(visual_tokens.shape)}"
        )
        assert text_part_tokens.dim() == 3, (
            f"text_part_tokens must be [B, M, D], got {tuple(text_part_tokens.shape)}"
        )
        assert routing_matrix.dim() == 3, (
            f"routing_matrix must be 3D, got {tuple(routing_matrix.shape)}"
        )
        B, P, D = visual_tokens.shape
        Bt, M, Dt = text_part_tokens.shape
        # Model outputs routing_matrix as [B, P, M]; transpose to [B, M, P].
        if routing_matrix.shape[1] == P and routing_matrix.shape[2] == M:
            routing_matrix = routing_matrix.transpose(1, 2).contiguous()  # [B, M, P]
        Br, Mr, Pr = routing_matrix.shape
        assert B == Bt == Br and D == Dt and M == Mr and P == Pr, (
            f"shape mismatch visual={tuple(visual_tokens.shape)} "
            f"text={tuple(text_part_tokens.shape)} routing={tuple(routing_matrix.shape)}"
        )
        # Normalize for cosine geometry.
        v_n = F.normalize(visual_tokens, dim=-1)                  # [B, P, D]
        t_n = F.normalize(text_part_tokens, dim=-1)               # [B, M, D]
        # Per-(b, m, p) cosine similarity.
        sim = torch.einsum('bmd,bpd->bmp', t_n, v_n)              # [B, M, P]
        # Target routing distribution over patches per (b, m).
        target = F.softmax(sim / max(float(tau), 1e-6), dim=-1)   # [B, M, P]
        # Actual routing (Sinkhorn assignment) normalized over P.
        actual = routing_matrix.clamp_min(1e-9)
        actual = actual / actual.sum(dim=-1, keepdim=True)
        # Forward KL: KL(target || actual) = sum target * (log target - log actual)
        log_actual = actual.log()
        log_target = target.detach().clamp_min(1e-9).log()
        kl_per_cb = (target.detach() * (log_target - log_actual)).sum(dim=-1)   # [B, M]
        start_m = 1 if skip_global else 0
        return kl_per_cb[:, start_m:].mean()

    def _loss_text_codeword_contrastive(
        self,
        text_part_tokens: torch.Tensor,    # [B, M, D] RAW post-adapter (NOT quantized)
        quantized_tokens: torch.Tensor,    # [B, M, D] post-VQ visual (= codebook[m, k_visual*])
        tau: float = 0.07,
        skip_global: bool = True,
    ) -> torch.Tensor:
        """v173: per-codebook InfoNCE between RAW text and visual codeword.

        For each slot m, positive pair = (text[i, m], quantized_visual[i, m])
        from the same image i. Negative pairs = (text[i, m], quantized_visual[j, m])
        for j != i (other images, same slot).

        Bypasses text-quantization noise present in xmodal_commit (uses RAW text)
        and works at codeword level (NOT routing level), avoiding routingText's
        weak per-patch signal failure mode. Symmetric (text->visual + visual->text)
        like CLIP/CIBHash NtXent.

        Gradient flows: visual via STE through quantization back to routing +
        codebooks + encoder; text via text_adapter (codebook for text not used).
        """
        assert text_part_tokens.dim() == 3, (
            f"text_part_tokens must be [B, M, D], got {tuple(text_part_tokens.shape)}"
        )
        assert quantized_tokens.dim() == 3, (
            f"quantized_tokens must be [B, M, D], got {tuple(quantized_tokens.shape)}"
        )
        B, M, D = text_part_tokens.shape
        Bv, Mv, Dv = quantized_tokens.shape
        assert B == Bv and M == Mv and D == Dv, (
            f"shape mismatch text={tuple(text_part_tokens.shape)} "
            f"visual={tuple(quantized_tokens.shape)}"
        )
        if B < 2:
            return text_part_tokens.new_zeros(())
        t_n = F.normalize(text_part_tokens, dim=-1)
        v_n = F.normalize(quantized_tokens, dim=-1)
        tau_eff = max(float(tau), 1e-6)
        start_m = 1 if skip_global else 0
        losses: list = []
        labels = torch.arange(B, device=text_part_tokens.device)
        for m in range(start_m, M):
            sim = t_n[:, m, :] @ v_n[:, m, :].T / tau_eff            # [B, B]
            L_tv = F.cross_entropy(sim,   labels)                    # text  -> visual
            L_vt = F.cross_entropy(sim.T, labels)                    # visual -> text
            losses.append(0.5 * (L_tv + L_vt))
        if not losses:
            return text_part_tokens.new_zeros(())
        return torch.stack(losses).mean()

    def _loss_text_preq_contrastive(
        self,
        text_part_tokens:       torch.Tensor,    # [B, M, D] RAW post-adapter
        semantic_visual_tokens: torch.Tensor,    # [B, M, D] PRE-VQ continuous
        tau: float = 0.07,
        skip_global: bool = True,
    ) -> torch.Tensor:
        """v174 Option alpha: text <-> PRE-QUANT visual semantic tokens InfoNCE.

        Unlike v173 (which contrasts text with quantized visual codeword and
        creates a collapse attractor on K=128 codebook), this operates on the
        CONTINUOUS pre-VQ semantic_visual_tokens — no quantization bottleneck.
        Codewords are learned by other losses (vq, quant, EMA); this loss
        only shapes the encoder + routing so that semantic_visual_tokens align
        with text per-slot.
        """
        assert text_part_tokens.dim() == 3 and semantic_visual_tokens.dim() == 3
        B, M, D = text_part_tokens.shape
        Bv, Mv, Dv = semantic_visual_tokens.shape
        assert B == Bv and M == Mv and D == Dv, (
            f"shape mismatch text={tuple(text_part_tokens.shape)} "
            f"visual_preq={tuple(semantic_visual_tokens.shape)}"
        )
        if B < 2:
            return text_part_tokens.new_zeros(())
        t_n = F.normalize(text_part_tokens, dim=-1)
        v_n = F.normalize(semantic_visual_tokens, dim=-1)
        tau_eff = max(float(tau), 1e-6)
        start_m = 1 if skip_global else 0
        losses: list = []
        labels = torch.arange(B, device=text_part_tokens.device)
        for m in range(start_m, M):
            sim = t_n[:, m, :] @ v_n[:, m, :].T / tau_eff
            L_tv = F.cross_entropy(sim,   labels)
            L_vt = F.cross_entropy(sim.T, labels)
            losses.append(0.5 * (L_tv + L_vt))
        if not losses:
            return text_part_tokens.new_zeros(())
        return torch.stack(losses).mean()

    def _loss_text_visual_hash_contrastive(
        self,
        text_continuous_code:   torch.Tensor,    # [B, 18, 4] text-derived continuous
        visual_continuous_code: torch.Tensor,    # [B, 18, 4] visual-derived continuous
        tau: float = 0.07,
    ) -> torch.Tensor:
        """v174 Option gamma: hash-level text <-> visual InfoNCE.

        Operates on the 18x4 codon-base continuous codes. Flattens to [B, 72]
        and applies per-image InfoNCE (positive = same image's text+visual
        hash, negative = other images). The 2^36 binary hash space has much
        higher capacity than K=128 codewords, so no collapse attractor.
        """
        assert text_continuous_code.dim() == 3 and visual_continuous_code.dim() == 3
        B, M_codon, B_per = text_continuous_code.shape
        Bv, Mv, Bvp = visual_continuous_code.shape
        assert B == Bv and M_codon == Mv and B_per == Bvp, (
            f"shape mismatch text_code={tuple(text_continuous_code.shape)} "
            f"visual_code={tuple(visual_continuous_code.shape)}"
        )
        if B < 2:
            return text_continuous_code.new_zeros(())
        t_flat = text_continuous_code.reshape(B, -1)        # [B, M_codon * B_per]
        v_flat = visual_continuous_code.reshape(B, -1)
        t_n = F.normalize(t_flat, dim=-1)
        v_n = F.normalize(v_flat, dim=-1)
        tau_eff = max(float(tau), 1e-6)
        sim = t_n @ v_n.T / tau_eff                          # [B, B]
        labels = torch.arange(B, device=t_n.device)
        L_tv = F.cross_entropy(sim,   labels)
        L_vt = F.cross_entropy(sim.T, labels)
        return 0.5 * (L_tv + L_vt)

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

    def _loss_cibhash_visual_per_codebook(
        self,
        visual_tokens_view1: torch.Tensor,    # [B, M, D] pre-VQ
        visual_tokens_view2: torch.Tensor,    # [B, M, D] pre-VQ
        temperature: float,
        text_part_raw: Optional[torch.Tensor] = None,
        dynamic_tau_alpha: float = 0.0,
        dynamic_tau_skip_global: bool = False,
        local_target: str = "instance",
        local_target_tau: float = 0.2,
        axis_text: Optional[torch.Tensor] = None,     # [B, M, D_text] centred, normalised
        queue: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,   # ([Q, M, D], [Q, M, D_text])
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """v150: per-codebook NtXent on pre-VQ routed visual tokens.

        Stage 7: ``local_target`` changes what the LOCAL slots (m >= 1) are asked
        for; the global slot always keeps the instance target. "axis_soft" uses the
        soft target softmax(cos(axis_text_i^m, axis_text_j^m) / local_target_tau)
        over the other 2B-1 rows (the augmentation partner shares the caption, so
        it keeps the largest share); "none" drops the local terms.

        Same per_codebook geometry as `_loss_cibhash_per_codebook` with
        mode="per_codebook", but operates on the D-dim semantic_visual_tokens
        from the router output (BEFORE VQ quantization and codon decoding).
        Restores full continuous cosine granularity (vs the 6-bit slice
        ceiling of bit_probs).

        KL term is not defined on continuous embeddings -> returns zeros.

        Args:
            visual_tokens_view{1,2}: [B, M, D] paired-aug semantic_visual_tokens.
            temperature: NtXent temperature.
            text_part_raw, dynamic_tau_alpha: per-pair temperature mod from
                text-cos geometry (identical mechanism to per_codebook bit-mode).
        Returns:
            (ntxent_loss, kl_loss=0).
        """
        B, M, D = visual_tokens_view1.shape
        T = max(float(temperature), 1e-6)
        use_dyn = (
            text_part_raw is not None
            and float(dynamic_tau_alpha) > 0.0
            and text_part_raw.shape[0] == B
            and text_part_raw.shape[1] == M
        )
        alpha = min(float(dynamic_tau_alpha), 0.999)

        ntxent_per_cb: list = []
        for m in range(M):
            if m > 0 and local_target == "none":
                continue
            vm1 = visual_tokens_view1[:, m, :]                            # [B, D]
            vm2 = visual_tokens_view2[:, m, :]                            # [B, D]
            v = torch.cat([vm1, vm2], dim=0)                               # [2B, D]
            v_n = F.normalize(v, dim=-1, eps=1e-8)
            sim_raw = v_n @ v_n.T                                          # [2B, 2B], cosine

            if m > 0 and local_target == "axis_soft":
                eye_ax = torch.eye(2 * B, device=sim_raw.device, dtype=torch.bool)
                t2 = torch.cat([axis_text[:, m, :], axis_text[:, m, :]], dim=0).float()
                t_logit = (t2 @ t2.T).masked_fill(eye_ax, -1e9)
                v_logit = (sim_raw.float() / T).masked_fill(eye_ax, -1e9)
                if queue is not None:                     # (stage 8, B2) past pairs join the candidates
                    qv, qt = queue
                    v_logit = torch.cat([v_logit, (v_n.float() @ qv[:, m, :].T) / T], dim=1)
                    t_logit = torch.cat([t_logit, t2 @ qt[:, m, :].T], dim=1)
                tgt = torch.softmax(t_logit / local_target_tau, dim=1)
                logp = F.log_softmax(v_logit, dim=1)
                ntxent_per_cb.append(-(tgt.detach() * logp).sum(dim=1).mean())
                continue

            if use_dyn and not (dynamic_tau_skip_global and m == 0):
                t_m = text_part_raw[:, m, :]                                # [B, D_text]
                t_m_n = F.normalize(t_m.float(), dim=-1)
                cos_tt = (t_m_n @ t_m_n.T).clamp(-1.0, 1.0)                 # [B, B]
                cos_tt_2 = cos_tt.repeat(2, 2)                              # [2B, 2B]
                tau_ij = T * (1.0 + alpha * cos_tt_2).clamp_min(1e-4)
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

        ntxent = torch.stack(ntxent_per_cb).mean()
        kl = visual_tokens_view1.new_zeros(())
        return ntxent, kl

    def _loss_cibhash_per_codebook(
        self,
        continuous_code_view1: torch.Tensor,
        continuous_code_view2: torch.Tensor,
        temperature: float,
        mode: str = "per_codebook",
        text_part_raw: Optional[torch.Tensor] = None,
        dynamic_tau_alpha: float = 0.0,
        dynamic_tau_skip_global: bool = False,
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
        # v149: continuous NtXent (shifted bit_probs) vs legacy STE-sign.
        if getattr(self, "cibhash_ntxent_continuous", False):
            # Linearly shift (0,1) -> (-1,+1); continuous, fully differentiable,
            # full cosine granularity (no 7-level quantization ceiling).
            z_v1 = 2.0 * bits_v1 - 1.0                                         # [B, 6, 6] in (-1, +1)
            z_v2 = 2.0 * bits_v2 - 1.0
        else:
            # Legacy: STE sign to binary hash (v119-v148 behavior).
            z_v1 = self._ste_sign(bits_v1)                                     # [B, 6, 6] in {-1, +1}
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

            if use_dyn and not (dynamic_tau_skip_global and m == 0):
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

    def _loss_cibhash_bit_kl_only(
        self,
        continuous_code_view1: torch.Tensor,
        continuous_code_view2: torch.Tensor,
        mode: str = "per_codebook",
    ) -> torch.Tensor:
        """Original-domain CIBHash KL for the visual-token hybrid variant.

        Visual-token NT-Xent lives on unconstrained D-dimensional embeddings,
        where Bernoulli KL is undefined.  This helper deliberately computes
        *only* the KL on the two views' post-VQ DNA bit probabilities, keeping
        the domains of the two CIBHash terms explicit.
        """
        bits_v1 = self._continuous_code_to_bit_probs(
            continuous_code_view1, num_codebooks=self.num_codebooks,
        )
        bits_v2 = self._continuous_code_to_bit_probs(
            continuous_code_view2, num_codebooks=self.num_codebooks,
        )
        if bits_v1.shape != bits_v2.shape:
            raise ValueError(
                "CIBHash hybrid bit KL requires matching view shapes; "
                f"got {tuple(bits_v1.shape)} and {tuple(bits_v2.shape)}"
            )
        if mode == "global":
            return self._cibhash_kl(
                bits_v1.reshape(bits_v1.shape[0], -1),
                bits_v2.reshape(bits_v2.shape[0], -1),
            )
        if mode != "per_codebook":
            raise ValueError(
                f"Unsupported cibhash_mode={mode!r}; expected per_codebook/global"
            )
        per_codebook = [
            self._cibhash_kl(bits_v1[:, m, :], bits_v2[:, m, :])
            for m in range(bits_v1.shape[1])
        ]
        return torch.stack(per_codebook).mean()

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
        enable_counterfactual: bool = True,
        outputs_notext: Optional[Dict[str, Any]] = None,
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

        # Diagnostics for the integrated counterfactual hard-negative branch.
        # They stay zero when the branch is disabled or a batch has no valid
        # foils; no separate weighted loss is added to `total`.
        loss_text_hash_counterfactual = u.new_zeros(())
        counterfactual_pos_sim = u.new_zeros(())
        counterfactual_foil_sim = u.new_zeros(())
        counterfactual_margin_violation = u.new_zeros(())
        counterfactual_codeword_flip = u.new_zeros(())
        counterfactual_valid_ratio = u.new_zeros(())

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
                # v176: optionally skip cb0 (C_global).
                if self.text_hash_ntxent_skip_global and M_th > 1:
                    t = t[:, 1:, :]
                    i = i[:, 1:, :]
                    M_th_eff = M_th - 1
                else:
                    M_th_eff = M_th
                t_n = F.normalize(t, dim=-1)                          # [B, M', L*4]
                i_n = F.normalize(i, dim=-1)
                # logits_it[m, i, j]: visual sample i vs text sample j for
                # codebook m. Positives = diagonal per codebook. M'
                # independent CE losses, averaged.
                logits_it = torch.einsum("bmd,cmd->mbc", i_n, t_n) / tau  # [M', B, B]
                labels = torch.arange(B, device=logits_it.device)
                labels_m = labels.unsqueeze(0).expand(M_th_eff, B).reshape(-1)
                # Counterfactual minimal-pair integration. Each visual anchor
                # gets exactly one own foil column; other samples' foils are
                # deliberately excluded to avoid uncontrolled false negatives.
                rho_cfg = float(self.text_hash_counterfactual_weight)
                if not enable_counterfactual:
                    rho_eff = 0.0
                elif self.text_hash_counterfactual_warmup_epochs > 0:
                    _ep = 0 if epoch is None else max(int(epoch), 0)
                    rho_eff = rho_cfg * min(
                        1.0,
                        float(_ep) / float(self.text_hash_counterfactual_warmup_epochs),
                    )
                else:
                    rho_eff = rho_cfg
                foil_cc = outputs.get("text_foil_continuous_code")
                foil_valid = outputs.get("text_foil_valid_mask")
                if rho_eff > 0.0 and (
                    foil_cc is None or foil_valid is None
                ):
                    raise RuntimeError(
                        "counterfactual text-DNA loss is active, but model "
                        "outputs contain no foil code/mask"
                    )
                use_foil = rho_eff > 0.0
                if use_foil:
                    if foil_cc.shape != text_cc.shape:
                        raise ValueError(
                            "text foil continuous code must match factual text "
                            f"shape; got foil={tuple(foil_cc.shape)} "
                            f"factual={tuple(text_cc.shape)}"
                        )
                    if foil_valid.shape != (B, int(getattr(self, "num_codebooks", 6))):
                        raise ValueError(
                            "text_foil_valid_mask must be [B, M]; got "
                            f"{tuple(foil_valid.shape)}"
                        )
                    if foil_valid.dtype != torch.bool:
                        raise TypeError(
                            "text_foil_valid_mask must have dtype torch.bool, "
                            f"got {foil_valid.dtype}"
                        )
                    if bool(foil_valid[:, 0].any().item()):
                        raise ValueError(
                            "counterfactual C_global (slot 0) must be invalid"
                        )
                    f4 = foil_cc.view(B, M_th, L_th, 4)
                    if (
                        bool(foil_valid.any().item())
                        and not bool(torch.isfinite(
                            f4[foil_valid],
                        ).all().item())
                    ):
                        raise ValueError(
                            "valid counterfactual text-DNA codes must be finite"
                        )
                    f = f4.reshape(
                        B, M_th, L_th * 4,
                    )
                    fv = foil_valid
                    if self.text_hash_ntxent_skip_global and M_th > 1:
                        f = f[:, 1:, :]
                        fv = fv[:, 1:]
                    # Stop-gradient on the foil prevents the negative text
                    # branch from satisfying the loss by learning a language
                    # shortcut. Positive text and visual branches retain their
                    # existing gradients.
                    f_n = F.normalize(f.detach(), dim=-1)
                    foil_cos = torch.einsum("bmd,bmd->mb", i_n, f_n)
                    own_foil_logits = (
                        (foil_cos + float(self.text_hash_counterfactual_margin))
                        / tau
                        + math.log(max(rho_eff, 1e-12))
                    )                                                        # [M', B]
                    own_foil_logits = own_foil_logits.masked_fill(
                        ~fv.transpose(0, 1), -float("inf"),
                    )
                    if bool(fv.any().item()):
                        logits_i2t = torch.cat(
                            [logits_it, own_foil_logits.unsqueeze(-1)], dim=-1,
                        )                                                    # [M', B, B+1]
                        base_i2t_per = F.cross_entropy(
                            logits_it.reshape(M_th_eff * B, B),
                            labels_m,
                            reduction="none",
                        ).reshape(M_th_eff, B)
                        foil_i2t_per = F.cross_entropy(
                            logits_i2t.reshape(M_th_eff * B, B + 1),
                            labels_m,
                            reduction="none",
                        ).reshape(M_th_eff, B)
                        valid_mb = fv.transpose(0, 1)
                        # Preserve the full factual InfoNCE mean, then average
                        # only the integrated own-foil increment across valid
                        # anchors. This prevents sparse slot coverage from
                        # implicitly shrinking rho.
                        loss_i2t = (
                            base_i2t_per.mean()
                            + (foil_i2t_per - base_i2t_per)[valid_mb].mean()
                        )
                        pos_cos = torch.einsum("bmd,bmd->mb", i_n, t_n)
                        cf_soft_triplet = F.softplus(
                            (
                                foil_cos
                                - pos_cos
                                + float(self.text_hash_counterfactual_margin)
                            ) / tau
                        )
                        loss_text_hash_counterfactual = cf_soft_triplet[valid_mb].mean()
                        counterfactual_pos_sim = pos_cos[valid_mb].mean()
                        counterfactual_foil_sim = foil_cos[valid_mb].mean()
                        counterfactual_margin_violation = (
                            pos_cos[valid_mb]
                            <= foil_cos[valid_mb]
                               + float(self.text_hash_counterfactual_margin)
                        ).to(i_n.dtype).mean()
                        counterfactual_valid_ratio = fv.to(i_n.dtype).mean()
                        pos_idx_text = outputs.get("text_codebook_indices")
                        foil_idx_text = outputs.get("text_foil_codebook_indices")
                        if (
                            pos_idx_text is not None
                            and foil_idx_text is not None
                            and pos_idx_text.shape == foil_idx_text.shape == (B, M_th)
                        ):
                            pos_idx_local = pos_idx_text[:, 1:] \
                                if self.text_hash_ntxent_skip_global else pos_idx_text
                            foil_idx_local = foil_idx_text[:, 1:] \
                                if self.text_hash_ntxent_skip_global else foil_idx_text
                            counterfactual_codeword_flip = (
                                (pos_idx_local != foil_idx_local)[fv]
                                .to(i_n.dtype).mean()
                            )
                    else:
                        loss_i2t = F.cross_entropy(
                            logits_it.reshape(M_th_eff * B, B), labels_m,
                        )
                else:
                    loss_i2t = F.cross_entropy(
                        logits_it.reshape(M_th_eff * B, B), labels_m,
                    )
                loss_t2i = F.cross_entropy(
                    logits_it.transpose(1, 2).reshape(M_th_eff * B, B), labels_m,
                )
                if self.text_hash_ntxent_target == "axis_soft":
                    # (stage 9) codon-level axis-neighbour target: slot m's image codon should
                    # match the text codons of images whose axis-m caption is similar, in the
                    # proportion softmax(cos / tau_t); the own caption (cos 1) keeps the largest
                    # share. Replaces the one-hot target on the local slots only.
                    assert rho_cfg <= 0.0, "axis_soft target and the counterfactual branch are exclusive"
                    ax = self._centred_axis_text(outputs.get("text_part_raw_cached"))     # [B, M, Dt]
                    ax = ax[:, 1:, :] if (self.text_hash_ntxent_skip_global and M_th > 1) else ax
                    tgt = torch.softmax(torch.einsum("bmd,cmd->mbc", ax, ax)
                                        / self.text_hash_ntxent_target_tau, dim=2).detach()  # [M', B, B]
                    lp_i2t = F.log_softmax(logits_it.float(), dim=2)
                    lp_t2i = F.log_softmax(logits_it.transpose(1, 2).float(), dim=2)
                    loss_i2t = -(tgt * lp_i2t).sum(dim=2).mean()
                    loss_t2i = -(tgt * lp_t2i).sum(dim=2).mean()        # symmetric cosines: same rows
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

        # v160 (Uni-Code Eq.8): cross-modal commitment loss.
        # Symmetric MSE between each modality's encoder output and the
        # OPPOSITE modality's quantized codeword (with stop-gradient on the
        # quantized target). Standard self-modality commitment (lambda_quant)
        # is preserved upstream; this term ADDS the cross-modal extension at
        # weight beta/2 (per paper, lambda_xmodal_commit = 0.025 recommended
        # when lambda_quant = 0.05).
        loss_xmodal_commit = u.new_zeros(())
        if self.lambda_xmodal_commit > 0.0:
            z_v = outputs.get("semantic_visual_tokens")      # [B, M, D]  visual encoder output
            q_v = outputs.get("quantized_tokens_raw")         # [B, M, D]  visual quantized (sg target for text)
            t_v = outputs.get("text_part_tokens")             # [B, M, D]  text encoder output (post text_adapter)
            q_t = outputs.get("text_quantized_tokens")        # [B, M, D]  text quantized (sg target for visual)
            # v176: optionally skip cb0 (C_global) — architectural mismatch
            # because C_global uses pooled visual_global (not text-anchored).
            start_m = 1 if self.xmodal_commit_skip_global else 0
            if (
                z_v is not None and q_t is not None
                and z_v.shape == q_t.shape
            ):
                loss_xmodal_visual = F.mse_loss(z_v[:, start_m:, :], q_t[:, start_m:, :].detach())
            else:
                loss_xmodal_visual = u.new_zeros(())
            if (
                t_v is not None and q_v is not None
                and t_v.shape == q_v.shape
            ):
                loss_xmodal_text   = F.mse_loss(t_v[:, start_m:, :], q_v[:, start_m:, :].detach())
            else:
                loss_xmodal_text   = u.new_zeros(())
            # Average the two symmetric directions (so the lambda represents
            # the per-direction weight, matching the paper's beta/2 per side).
            loss_xmodal_commit = 0.5 * (loss_xmodal_visual + loss_xmodal_text)

        # TD (2026-10-07): text-dropout path consistency. `outputs` is the
        # caption-routed forward (teacher, no gradient); `outputs_notext` is
        # the extra deployment-routed forward on the SAME view-1 visual inputs
        # (student; gradient reaches the visual adapter through it). Both
        # distributions come from the squared-Euclidean codebook_distances the
        # argmin assignment uses (NOT the cosine logits of text_code_kl).
        # Every other term stays on `outputs` exactly as before.
        #
        # The term is computed ONLY when the teacher forward was actually
        # caption-routed (outputs["routing_mode"] == "text"). A batch with no
        # real text routes `outputs` by codebook_mean too, and the KL would
        # then self-distil two no-text forwards (identical inputs, identical
        # routing) -- a meaningless, near-zero signal that would still spend
        # a backward pass. Such batches return the zero tensor instead.
        loss_path_consistency = u.new_zeros(())
        if (
            self.lambda_path_consistency > 0.0
            and outputs_notext is not None
            and outputs.get("routing_mode") == "text"
        ):
            d_notext = outputs_notext.get("codebook_distances")        # [B, M, K]
            if d_notext is None or tuple(d_notext.shape) != tuple(distances.shape):
                raise ValueError(
                    "[DNACodonHashLoss] lambda_path_consistency > 0 requires "
                    "outputs_notext['codebook_distances'] with the same shape "
                    f"as outputs['codebook_distances'] ({tuple(distances.shape)})."
                )
            start_m = 0 if self.path_consistency_include_global else 1
            tau_pc = max(float(self.path_consistency_tau), 1e-6)
            d_text_pc = distances[:, start_m:, :]                      # [B, M', K]
            logits_t = (-d_text_pc / tau_pc).detach()                  # teacher
            logits_n = -d_notext[:, start_m:, :] / tau_pc              # student
            # Inactive codewords: the quantizer fills their distances with
            # +inf (model_siglip2.py `distances.masked_fill(inactive, inf)`),
            # so `-inf / tau` would make log_softmax produce NaN. Build the
            # mask from the teacher's non-finite entries, OR'ed with the
            # explicit active mask when the model provides one, and apply it
            # to BOTH logits so the two distributions share a support.
            inactive_pc = ~torch.isfinite(d_text_pc)                   # [B, M', K]
            active_pc = outputs.get("codebook_active_mask")           # [M, K] bool | None
            if active_pc is not None:
                inactive_from_mask = ~active_pc.to(device=logits_t.device, dtype=torch.bool)
                inactive_pc = inactive_pc | inactive_from_mask[start_m:, :].unsqueeze(0)
            logits_t = logits_t.masked_fill(inactive_pc, -1e9)
            logits_n = logits_n.masked_fill(inactive_pc, -1e9)
            log_p_t = F.log_softmax(logits_t, dim=-1)
            log_p_n = F.log_softmax(logits_n, dim=-1)
            kl_pc = (log_p_t.exp() * (log_p_t - log_p_n)).sum(dim=-1)   # [B, M']
            loss_path_consistency = kl_pc.mean()

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
            _ttp_raw = (
                outputs.get("text_global_feat")
                if getattr(self, "cibhash_dynamic_tau", False)
                else None
            )
            _dyn_alpha = (
                float(getattr(self, "cibhash_dynamic_tau_alpha", 0.0))
                if getattr(self, "cibhash_dynamic_tau", False) else 0.0
            )
            _src = getattr(self, "cibhash_ntxent_source", "continuous_code")
            if _src == "visual_token":
                # v150: NtXent on pre-VQ semantic_visual_tokens [B, M, D].
                # Optional projection heads isolate this dominant instance
                # objective from the semantic VQ/DNA representation.
                sv_v1 = outputs.get("cibhash_visual_tokens")
                sv_v2 = outputs_view2.get("cibhash_visual_tokens")
                if sv_v1 is None:
                    sv_v1 = outputs.get("semantic_visual_tokens")
                if sv_v2 is None:
                    sv_v2 = outputs_view2.get("semantic_visual_tokens")
                if sv_v1 is not None and sv_v2 is not None:
                    _axis_text = _queue = None
                    if self.cibhash_local_target == "axis_soft":
                        _axis_text = self._centred_axis_text(outputs.get("text_part_raw_cached"))
                        if self.cibhash_local_queue > 0 and self._q_n > 0:
                            # a snapshot: the enqueue below rewrites the FIFO in place before
                            # backward, and autograd keeps the queue tensor for d(logit)/d(token)
                            _queue = (self._q_v[: self._q_n].clone(), self._q_t[: self._q_n].clone())
                    loss_cibhash_ntxent_v, loss_cibhash_kl_v = self._loss_cibhash_visual_per_codebook(
                        sv_v1, sv_v2, temperature=self.cibhash_temperature,
                        text_part_raw=_ttp_raw,
                        dynamic_tau_alpha=_dyn_alpha,
                        dynamic_tau_skip_global=self.cibhash_dynamic_tau_skip_global,
                        local_target=self.cibhash_local_target,
                        local_target_tau=self.cibhash_local_target_tau,
                        axis_text=_axis_text,
                        queue=_queue,
                    )
                    if self.cibhash_local_queue > 0 and _axis_text is not None and self.training:
                        self._enqueue(F.normalize(sv_v1.detach().float(), dim=-1), _axis_text)
                    if (
                        self.cibhash_visual_token_bit_kl
                        and self.lambda_cibhash_kl > 0.0
                    ):
                        u_v1 = outputs.get("continuous_code")
                        u_v2 = outputs_view2.get("continuous_code")
                        if u_v1 is None or u_v2 is None:
                            raise ValueError(
                                "--cibhash_visual_token_bit_kl requires "
                                "continuous_code from both augmented views"
                            )
                        loss_cibhash_kl_v = self._loss_cibhash_bit_kl_only(
                            u_v1, u_v2, mode=self.cibhash_mode,
                        )
            else:
                u_v1 = outputs.get("continuous_code")
                u_v2 = outputs_view2.get("continuous_code")
                if u_v1 is not None and u_v2 is not None:
                    loss_cibhash_ntxent_v, loss_cibhash_kl_v = self._loss_cibhash_per_codebook(
                        u_v1, u_v2, temperature=self.cibhash_temperature,
                        mode=getattr(self, "cibhash_mode", "per_codebook"),
                        text_part_raw=_ttp_raw,
                        dynamic_tau_alpha=_dyn_alpha,
                        dynamic_tau_skip_global=self.cibhash_dynamic_tau_skip_global,
                    )

        # v144: text -> code KL distillation. Per-codebook distribution
        # matching between visual and text views over the K codewords.
        # Only fires when --lambda_text_code_kl > 0 and the model exposed
        # the full codebook tensor + text_part_tokens.
        loss_text_code_kl = u.new_zeros(())
        _vq_bypassed = (self.vq_bypass_epochs > 0 and epoch is not None
                        and int(epoch) < self.vq_bypass_epochs)
        if self.lambda_text_code_kl > 0.0 and not _vq_bypassed:
            z_tck = outputs.get("semantic_visual_tokens")
            if z_tck is None:
                z_tck = outputs.get("quantizer_input")
            t_tck = outputs.get("text_part_tokens")
            cb_tck = outputs.get("codebooks")
            if z_tck is not None and t_tck is not None and cb_tck is not None:
                loss_text_code_kl = self._loss_text_code_kl_per_codebook(
                    z_tck, t_tck, cb_tck,
                    codebook_active_mask=outputs.get("codebook_active_mask"),
                )

        # v172: per-patch routing supervision via text similarity.
        loss_routing_text = u.new_zeros(())
        if self.lambda_routing_text > 0.0:
            v_rt = outputs.get("visual_tokens")        # [B, P, D]
            t_rt = outputs.get("text_part_tokens")     # [B, M, D]
            r_rt = outputs.get("routing_matrix")       # [B, M, P]
            if v_rt is not None and t_rt is not None and r_rt is not None:
                loss_routing_text = self._loss_routing_text_per_codebook(
                    v_rt, t_rt, r_rt,
                    tau=self.routing_text_tau,
                    skip_global=self.routing_text_skip_global,
                )

        # v173: text-codeword contrastive — RAW text_part vs quantized_tokens.
        loss_text_codeword_contrastive = u.new_zeros(())
        if self.lambda_text_codeword_contrastive > 0.0:
            t_tcc = outputs.get("text_part_tokens")    # [B, M, D] raw post-adapter
            q_tcc = outputs.get("quantized_tokens")    # [B, M, D] post-VQ visual
            if t_tcc is not None and q_tcc is not None:
                loss_text_codeword_contrastive = self._loss_text_codeword_contrastive(
                    t_tcc, q_tcc,
                    tau=self.text_codeword_contrastive_tau,
                    skip_global=self.text_codeword_contrastive_skip_global,
                )

        # v174 Option alpha: text vs PRE-QUANT semantic_visual_tokens contrastive.
        loss_text_preq_contrastive = u.new_zeros(())
        if self.lambda_text_preq_contrastive > 0.0:
            t_tpq = outputs.get("text_part_tokens")
            z_tpq = outputs.get("semantic_visual_tokens")   # PRE-quantization
            if t_tpq is not None and z_tpq is not None:
                loss_text_preq_contrastive = self._loss_text_preq_contrastive(
                    t_tpq, z_tpq,
                    tau=self.text_preq_contrastive_tau,
                    skip_global=self.text_preq_contrastive_skip_global,
                )

        # v174 Option gamma: hash-code level text<->visual contrastive.
        loss_text_visual_hash_contrastive = u.new_zeros(())
        if self.lambda_text_visual_hash_contrastive > 0.0:
            t_thash = outputs.get("text_continuous_code")    # [B, 18, 4]
            v_thash = outputs.get("continuous_code")          # [B, 18, 4]
            if t_thash is not None and v_thash is not None:
                loss_text_visual_hash_contrastive = self._loss_text_visual_hash_contrastive(
                    t_thash, v_thash,
                    tau=self.text_visual_hash_contrastive_tau,
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
            + self.lambda_xmodal_commit   * loss_xmodal_commit
            + self.lambda_codeword_text_proto * loss_codeword_text_proto
            + self.lambda_cibhash_ntxent  * loss_cibhash_ntxent_v
            + self.lambda_cibhash_kl      * loss_cibhash_kl_v
            + self.lambda_swav_assign     * loss_swav_assign_v
            + self.lambda_proto_cluster   * loss_proto_cluster_v
            + self.lambda_text_code_kl    * loss_text_code_kl
            + self.lambda_routing_text    * loss_routing_text
            + self.lambda_text_codeword_contrastive * loss_text_codeword_contrastive
            + self.lambda_text_preq_contrastive     * loss_text_preq_contrastive
            + self.lambda_text_visual_hash_contrastive * loss_text_visual_hash_contrastive
        )
        # TD (2026-10-07): added only when active so the default total is
        # bit-identical to the legacy sum.
        if self.lambda_path_consistency > 0.0 and outputs_notext is not None:
            total = total + self.lambda_path_consistency * loss_path_consistency

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

        # ---- (b) similarity-spread calibration (SDC-style), default OFF ----
        if self.lambda_sim_spread > 0.0 and vg is not None and B >= 8:
            _t = F.normalize(vg.detach().float(), dim=-1)            # frozen teacher
            _base_match = (self.sim_spread_metric == "base_match")
            if _base_match:
                _p = u.float()                                       # [B,R,4] softmax probs
            else:
                _c = F.normalize(u.reshape(B, -1).float(), dim=-1)   # continuous code
            _n = min(int(self.sim_spread_pairs), B * (B - 1) // 2)
            _ia = torch.randint(0, B, (_n,), device=device)
            _ib = torch.randint(0, B, (_n,), device=device)
            _keep = _ia != _ib
            _ia, _ib = _ia[_keep], _ib[_keep]
            if _ia.numel() >= 8:
                _ts = (_t[_ia] * _t[_ib]).sum(-1).detach()           # teacher cos
                if _base_match:
                    # E[# matching bases]/R = mean_r <p_i[r], p_j[r]>  in [0,1]
                    _cs = (_p[_ia] * _p[_ib]).sum(-1).mean(-1)
                else:
                    _cs = (_c[_ia] * _c[_ib]).sum(-1)                # code cos
                _order = torch.argsort(_ts)                          # rank by teacher
                _m = _ia.numel()
                # symmetric Beta(b,b) quantile targets, mapped to [-1, 1]
                # torch.distributions.Beta has no icdf -> use scipy ppf, cached
                # per (m, beta) since the target vector is deterministic.
                if _base_match:
                    # Calibration prior for a 4-ary code (SDC Eq.7 analogue):
                    # two uniform DNA codes agree per position w.p. 1/4, so the
                    # match count ~ Binomial(R, 1/4). Moment-matched Beta:
                    #   alpha = (R-1)/4,  beta = 3(R-1)/4   (mean 1/4)
                    _R = int(u.shape[1])
                    _al = max((_R - 1) / 4.0, 1e-3)
                    _be = max(3.0 * (_R - 1) / 4.0, 1e-3)
                    _lo, _hi = 0.0, 1.0                              # s already in [0,1]
                else:
                    _al = _be = float(self.sim_spread_beta)
                    _lo, _hi = -1.0, 1.0                             # cosine range
                _key = (int(_m), _al, _be, _lo)
                _cache = getattr(self, "_sim_spread_tgt_cache", None)
                if _cache is None:
                    _cache = {}
                    self._sim_spread_tgt_cache = _cache
                if _key not in _cache:
                    from scipy.stats import beta as _sp_beta
                    import numpy as _np
                    _qn = (_np.arange(_m, dtype=_np.float64) + 0.5) / _m
                    _tn = _sp_beta.ppf(_qn, _al, _be) * (_hi - _lo) + _lo
                    _cache[_key] = torch.tensor(_tn, dtype=torch.float32)
                _tgt = _cache[_key].to(device)
                loss_sim_spread = F.l1_loss(_cs[_order], _tgt.detach())
            else:
                loss_sim_spread = u.new_zeros(())
            total = total + self.lambda_sim_spread * loss_sim_spread
        else:
            loss_sim_spread = u.new_zeros(())

        # ---- (c) constraint-AWARE training: GC + homopolymer, default OFF ----
        # u : [B, R, 4] softmax over (A, C, G, T) at each of the R = 18/24
        # positions of the concatenated code -- the SAME sequence the post-hoc
        # DP projection operates on, so the penalty spans slot boundaries
        # exactly as the constraint does.  Base order A=0 C=1 G=2 T=3 matches
        # dna_utils/bio_constraints._IS_GC = [0, 1, 1, 0].
        if self.lambda_bio_constraint > 0.0 and u is not None:
            _R = int(u.shape[1])
            _uf = u.float()
            # (i) GC hinge on the EXPECTED GC count.  Bounds follow the
            # evaluation convention: ceil(min_frac*R) .. floor(max_frac*R)
            # (18 -> [8, 10]; 24 -> [10, 14]).
            _gmin_f = (0.40 if self.bio_constraint_gc_min is None
                       else float(self.bio_constraint_gc_min))
            _gmax_f = (0.60 if self.bio_constraint_gc_max is None
                       else float(self.bio_constraint_gc_max))
            _gc_lo = float(math.ceil (_gmin_f * _R))
            _gc_hi = float(math.floor(_gmax_f * _R))
            _gc = _uf[:, :, 1].sum(dim=1) + _uf[:, :, 2].sum(dim=1)      # [B]
            _l_gc = (F.relu(_gc_lo - _gc) + F.relu(_gc - _gc_hi)).mean() / _R
            # (ii) Homopolymer: expected number of (max_run + 1)-windows whose
            # bases are all identical.  Under a per-position independence
            # surrogate, P(window w constant at base b) = prod_j p[w+j, b].
            _w = int(self.bio_constraint_max_run) + 1
            if _R > _w:
                _win = _uf.unfold(1, _w, 1)                # [B, R-w+1, 4, w]
                _l_hp = _win.prod(dim=-1).sum(dim=-1).sum(dim=-1).mean() / _R
            else:
                _l_hp = _uf.new_zeros(())
            loss_bio_constraint = (self.bio_constraint_gc_weight * _l_gc
                                   + self.bio_constraint_hp_weight * _l_hp)
            total = total + self.lambda_bio_constraint * loss_bio_constraint
        else:
            loss_bio_constraint = u.new_zeros(())

        # ---- (d) joint codon diversity per slot, default OFF -----------------
        # u : [B, R, 4] with R = M*L. Each sample's per-slot joint over the L
        # positions is the outer product of its L position posteriors (the
        # positions are independent GIVEN the sample -- each is its own softmax).
        # Averaging over the batch gives Q_m in the 4**L simplex; we push Q_m to
        # uniform with the SAME forward-KL form `loss_base_balance` uses, so the
        # mode-covering penalty falls on codons the slot never emits.
        # ---- slot spatial diversity (ConceptHash Eq. 8, `L_csd`) ----------
        #   L = 1/(N*M*(M-1)) * sum_{i != j} cos(A_i, A_j)
        # A_m is the slot's distribution over patches -- for us a column of the
        # routing matrix rather than a ViT attention map -- taken BEFORE the
        # adaptive top-p mask. A masked-out slot has an exactly-zero column,
        # whose cosine to everything is zero, so on the post-mask plan a starved
        # slot would look maximally diverse and get no gradient at all.
        # Neither squared nor absolute-valued, matching the paper: negative
        # correlation is rewarded, so slots are pushed apart rather than merely
        # kept from overlapping.
        loss_slot_diversity = None
        if self.lambda_slot_diversity > 0.0:
            _Pp = outputs.get("routing_matrix_premask")
            if _Pp is not None and _Pp.dim() == 3:
                _P = _Pp.float()
                if _P.shape[1] < _P.shape[2]:      # tolerate a [B, M, N] layout
                    _P = _P.transpose(1, 2)
                _A = _P.transpose(1, 2)            # [B, M, N_patch]
                # The router publishes its OWN columns. With
                # --c_global_source siglip2_global slot 0 bypasses the router
                # and is concatenated afterwards, so the pre-mask snapshot has
                # 5 columns (all local) while the post-mask matrix has 6.
                # Verified on Flickr: premask [8,196,5] vs postmask [8,196,6].
                # Dropping column 0 unconditionally would delete a real local
                # slot, so only drop it when global is actually present.
                _full = outputs.get("routing_matrix")
                _full_M = (min(_full.shape[1], _full.shape[2])
                           if (_full is not None and _full.dim() == 3) else None)
                _has_global = (_full_M is not None and _A.shape[1] == _full_M)
                if self.slot_diversity_skip_global and _has_global and _A.shape[1] > 1:
                    _A = _A[:, 1:, :]
                _Msd = _A.shape[1]
                if _Msd > 1:
                    _An = F.normalize(_A, dim=-1, eps=1e-8)
                    _G = torch.bmm(_An, _An.transpose(1, 2))          # [B, M, M]
                    _off = ~torch.eye(_Msd, dtype=torch.bool, device=_G.device)
                    loss_slot_diversity = _G[:, _off].mean()
                    total = total + self.lambda_slot_diversity * loss_slot_diversity

        if self.lambda_codon_joint > 0.0 and u is not None:
            _R = int(u.shape[1])
            _L = int(getattr(self, "num_codons_per_codebook", 0)) or (
                _R // 6 if _R % 6 == 0 else 3)
            _M = _R // _L
            _p = u.float().view(B, _M, _L, 4)
            _j = _p[:, :, 0, :]                                   # [B, M, 4]
            for _l in range(1, _L):
                _j = (_j.unsqueeze(-1) * _p[:, :, _l, :].unsqueeze(-2)).flatten(-2)
            _Q = _j.mean(dim=0)                                   # [M, 4**L]
            if self.codon_joint_slots is not None:
                _sel = [m for m in self.codon_joint_slots if 0 <= m < _M]
                if not _sel:
                    raise ValueError(
                        f"--codon_joint_slots selects no valid slot for M={_M}")
                _Q = _Q[_sel]                                     # [|sel|, 4**L]
            _Q = _Q / _Q.sum(-1, keepdim=True).clamp_min(1e-12)
            _logQ = _Q.clamp_min(self.codon_joint_floor).log()
            _unif = torch.full_like(_Q, 1.0 / _Q.shape[-1])
            loss_codon_joint = F.kl_div(_logQ, _unif, reduction="batchmean")
            total = total + self.lambda_codon_joint * loss_codon_joint
        else:
            loss_codon_joint = u.new_zeros(())

        # (P4) in-image cross-axis alignment; replaces text_code_kl.
        loss_role = u.new_zeros(())
        if self.lambda_role > 0.0 and not _vq_bypassed:
            if self.role_source == "pre_quant":
                _q_role = outputs.get("semantic_visual_tokens")
                if _q_role is None:
                    _q_role = outputs.get("quantizer_input")
            else:
                _q_role = outputs.get("quantized_tokens")
                if _q_role is None:
                    _q_role = outputs.get("semantic_visual_tokens")
            _t_role = outputs.get("text_part_tokens")
            if _q_role is not None and _t_role is not None:
                _r = self._loss_role_axis(_q_role[:, 1:, :], _t_role[:, 1:, :])
                if _r is not None:
                    loss_role = _r
                    total = total + self.lambda_role * loss_role

        # (stage 7, A) caption-concept cross-entropy for the named concept codebook.
        loss_concept = u.new_zeros(())
        if self.lambda_concept > 0.0 and not _vq_bypassed:
            _c = self._loss_concept(outputs.get("text_part_raw_cached"),
                                    outputs.get("codebook_distances"))
            if _c is not None:
                loss_concept = _c
                total = total + self.lambda_concept * loss_concept

        # (b-1) masked entity completion, computed inside the model forward.
        loss_mec = outputs.get("loss_mec")
        mec_acc = outputs.get("mec_acc")
        if self.lambda_mec > 0.0 and loss_mec is not None:
            total = total + self.lambda_mec * loss_mec
        else:
            loss_mec = u.new_zeros(())
        if mec_acc is None:
            mec_acc = u.new_zeros(())

        _ret = {
            "loss":              total,
            "loss_codon_joint":  loss_codon_joint,
            "loss_mec":          loss_mec,
            "loss_role":         loss_role,
            "mec_acc":           mec_acc,
            "loss_slot_diversity": (loss_slot_diversity
                                    if loss_slot_diversity is not None
                                    else u.new_zeros(())),
            "loss_sim_spread":   loss_sim_spread,
            "loss_bio_constraint": loss_bio_constraint,
            "loss_hash":         loss_hash,
            "loss_hash_hard":    loss_hash_hard,
            "loss_vq":           loss_vq,
            "loss_quant":        loss_quant,
            "loss_anchor":       loss_anchor,
            "loss_wasserstein":  loss_wasserstein,
            "loss_text_hash":    loss_text_hash,
            "loss_text_hash_ntxent_add": loss_text_hash_ntxent_add,
            "loss_text_hash_counterfactual": loss_text_hash_counterfactual,
            "counterfactual_pos_sim": counterfactual_pos_sim,
            "counterfactual_foil_sim": counterfactual_foil_sim,
            "counterfactual_margin_violation": counterfactual_margin_violation,
            "counterfactual_codeword_flip": counterfactual_codeword_flip,
            "counterfactual_valid_ratio": counterfactual_valid_ratio,
            "loss_cw_xmodal":    loss_cw_xmodal,
            "loss_xmodal_commit": loss_xmodal_commit,
            "path_consistency":  loss_path_consistency,   # TD (2026-10-07)
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
            "loss_text_code_kl":   loss_text_code_kl,
            "loss_routing_text":   loss_routing_text,
            "loss_text_codeword_contrastive": loss_text_codeword_contrastive,
            "loss_text_preq_contrastive":     loss_text_preq_contrastive,
            "loss_text_visual_hash_contrastive": loss_text_visual_hash_contrastive,
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
        if self.lambda_concept > 0.0:                       # (stage 7) only when it is on
            _ret["loss_concept"] = loss_concept
        return _ret
