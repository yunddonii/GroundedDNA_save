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

        loss_dna = dna_components["loss_dna"]
        loss_bu  = bu_components ["loss_bu"]

        # ---- BU warm-up --------------------------------------------------
        if (epoch is not None) and (epoch < self.bu_warmup_epochs):
            eff_lambda_bu = 0.0
        else:
            eff_lambda_bu = self.lambda_bu

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
        )

        return {
            "loss":              total,
            "loss_hash":         loss_hash,
            "loss_hash_hard":    loss_hash_hard,
            "loss_vq":           loss_vq,
            "loss_quant":        loss_quant,
            "loss_anchor":       loss_anchor,
            "loss_wasserstein":  loss_wasserstein,
            "loss_dna":          loss_dna,
            "loss_bu":           loss_bu,
            "loss_entropy":      dna_components["loss_entropy"],
            "loss_base_balance": dna_components["loss_base_balance"],
            "loss_cb_balance":   bu_components ["loss_cb_balance"],
            "loss_cb_uncorr":    bu_components ["loss_cb_uncorr"],
        }
