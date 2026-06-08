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
from typing import Any, Dict, Optional, Tuple

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
PART_ORDER: Tuple[str, ...] = (
    "C_global",                  # index 0  -- global average pooling
    "C_head_or_main_part",       # index 1  ┐
    "C_body_or_secondary_part",  # index 2  │
    "C_limb_or_detail_part",     # index 3  ├─ Sinkhorn OT routing (local)
    "C_color_texture",           # index 4  │
    "C_background_null",         # index 5  ┘
)
NUM_SEMANTIC_PARTS = len(PART_ORDER)        # 6
LOCAL_PART_ORDER: Tuple[str, ...] = PART_ORDER[1:]
NUM_LOCAL_PARTS = len(LOCAL_PART_ORDER)     # 5


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
    ) -> None:
        super().__init__()
        self.num_codebooks = int(num_codebooks)
        self.codebook_size = int(codebook_size)
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

    # -------------------------------------------------------- helpers

    @torch.no_grad()
    def num_trainable_codewords(self) -> int:
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
        diag = {"mode": mode, "N": int(N), "K": int(K), "M": int(M_cb)}
        t0 = time.time()
        text_np = text_anchors.detach().cpu().float().numpy()  # [N, M, D]

        for m in range(M_cb):
            feats = text_np[:, m, :]  # [N, D]
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
                init_cs = max(1.0, float(N) / float(K))
                self.cluster_size[m].fill_(init_cs)
                self.embed_avg[m].copy_(centers_t * init_cs)
            else:
                # gradient mode: codebook is nn.Parameter
                self.codebooks.data[m].copy_(centers_t)

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
        if exclude_global:
            cb       = self.codebooks[1:]                               # [M', K_max, D]
            mask     = self.active_mask[1:]                             # [M', K_max]
        else:
            cb       = self.codebooks
            mask     = self.active_mask
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
        n_revived = 0
        for m in range(M):
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
        for m in range(M):
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

        # nearest-neighbour distance to codewords. Distance tensor is sized
        # K_max; inactive codewords are masked with +inf so argmin can never
        # select them (v78a adaptive K).
        if self.distance_mode == "cosine":
            z_n  = F.normalize(semantic_visual_tokens, dim=-1)                 # [B, M, D]
            cb_n = F.normalize(self.codebooks, dim=-1)                         # [M, K_max, D]
            cos_sim   = torch.einsum("bmd,mkd->bmk", z_n, cb_n)                # [B, M, K_max]
            distances = 1.0 - cos_sim                                          # [B, M, K_max] in [0, 2]
        else:
            diff = semantic_visual_tokens.unsqueeze(2) - self.codebooks.unsqueeze(0)
            distances = (diff ** 2).sum(dim=-1)                                # [B, M, K_max]
        # v78a: mask inactive codewords from lookup
        if self.K_max != self.codebook_size or (~self.active_mask).any():
            inactive = (~self.active_mask).unsqueeze(0)                        # [1, M, K_max]
            distances = distances.masked_fill(inactive, float("inf"))
        indices   = distances.argmin(dim=-1)                                   # [B, M]

        # gather quantized embeddings via advanced indexing:
        #   quantized[b, m, :] = codebooks[m, indices[b, m], :]
        device = semantic_visual_tokens.device
        m_idx  = torch.arange(self.num_codebooks, device=device).view(1, -1).expand(B, -1)  # [B, M]
        quantized = self.codebooks[m_idx, indices]                         # [B, M, D]

        # straight-through estimator (gradient passes through to the encoder
        # while the actual values are the quantized codewords)
        quantized_st = semantic_visual_tokens + (quantized - semantic_visual_tokens).detach()

        # EMA codebook update (only when training; no-op in 'gradient' mode).
        # Done AFTER computing indices/quantized so this batch's outputs are
        # consistent with the codebook state seen during the loss computation.
        if self.update_mode == "ema" and self.training:
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
        num_codons: int = 3,                 # v122: L = codon-positions per codebook
    ) -> None:
        super().__init__()
        self.num_codons: int = int(num_codons)
        if self.num_codons < 1:
            raise ValueError(f"[CodonHead] num_codons must be >= 1; got {self.num_codons}")
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
        # v69b (Exp 2): residual-split path (3-position only; guarded at __init__).
        if residual_active and self.residual_split:
            q_chunks = x.view(B, L, self.chunk)                            # [B, L, chunk]
            r_chunks = (gamma * residual).view(B, L, self.chunk)
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
            h = x.view(B, L, self.chunk)                                  # [B, L, D/L]
            # v66 text-anchored prototype head
            if self.use_text_anchor:
                h_n     = F.normalize(h, dim=-1)
                proto_n = F.normalize(self.proto, dim=-1)
                logits  = torch.einsum('bpc,pkc->bpk', h_n, proto_n) / self.anchor_temperature
                loss_text_anchor = torch.zeros((), device=x.device, dtype=x.dtype)
                if text_chunks is not None and self.training:
                    t        = text_chunks.view(B, L, self.chunk)
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
            self.backbone = build_clip_backbone(clip_name)
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
        self.route_global_text: bool = bool(getattr(args, "route_global_text", False))
        self.routed_cls_add_gamma: float = float(getattr(args, "routed_cls_add_gamma", 0.0))
        self.routed_cls_add_scope: str = str(getattr(args, "routed_cls_add_scope", "local"))
        if self.routed_cls_add_scope not in ("local", "all"):
            raise ValueError(
                f"routed_cls_add_scope must be 'local' or 'all', got {self.routed_cls_add_scope!r}"
            )
        # v122: C0-orthogonal local quantization. C_global keeps the coarse
        # semantic axis; local codebooks quantize the residual concept.
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
        else:
            raise ValueError(
                f"[model_siglip2] router_type must be 'sinkhorn' / 'attention' / 'slot', "
                f"got {self.router_type!r}"
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
        self.routing_adaptive_topp       = bool(getattr(args, "routing_adaptive_topp", False))
        self.routing_adaptive_topp_min   = float(getattr(args, "routing_adaptive_topp_min", 0.5))
        self.routing_adaptive_topp_max   = float(getattr(args, "routing_adaptive_topp_max", 0.9))
        self.routing_adaptive_topp_entropy = bool(getattr(args, "routing_adaptive_topp_entropy", False))
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
        # v93: per-codebook cross-modal codeword InfoNCE. When > 0, the same
        # text path used by lambda_text_hash is activated (text_part_tokens
        # through EMA-disabled quantizer) and the resulting text codeword is
        # exposed as `text_quantized_tokens`. The loss module then runs an
        # InfoNCE between visual `quantized_tokens` and `text_quantized_tokens`
        # per codebook. Single text-path forward serves both v91 and v93.
        self.lambda_cw_xmodal            = float(getattr(args, "lambda_cw_xmodal", 0.0))
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
        self.total_epochs           = int(getattr(args, "epoch", 60))
        # mutable per-step state set by trainer via set_current_epoch()
        self._current_epoch: int = 0

        # ---------- codebook quantizer -----------------------------------
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
                    position_specific_head=self.codon_position_specific_head,
                    position_residual_adapter=self.codon_position_residual_adapter,
                    residual_split=self.codon_residual_split,
                    residual_gate=self.codon_residual_gate,
                    use_full_linear=self.codon_full_linear,
                    num_codons=self.num_codons_per_codebook,
                )
                for _ in range(self.num_codebooks)
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

    def set_current_epoch(self, epoch: int) -> None:
        """Trainer calls this once per epoch so the router can compute the
        annealed epsilon. v33a only; no-op when annealing is off."""
        self._current_epoch = int(epoch)

    def _current_sinkhorn_epsilon(self) -> Optional[float]:
        """Return the annealed Sinkhorn epsilon for the current epoch, or
        None if annealing is not configured (router falls back to its
        stored static epsilon)."""
        eps_i = self.sinkhorn_epsilon_init
        eps_f = self.sinkhorn_epsilon_final
        if eps_i is None or eps_f is None:
            return None
        t_max = max(self.total_epochs - 1, 1)
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

        # 1) adapter projection (SEPARATE parameters per branch)
        visual_tokens = self.visual_adapter(feats["visual_tokens_raw"])  # [B, N, D]
        B, N, D = visual_tokens.shape

        text_part_tokens:  Optional[torch.Tensor] = None
        global_text_token: Optional[torch.Tensor] = None
        local_text_tokens: Optional[torch.Tensor] = None
        global_text_token_for_routing: Optional[torch.Tensor] = None
        local_text_tokens_for_routing: Optional[torch.Tensor] = None  # v116
        if use_text_routing and feats["text_part_raw"] is not None:
            raw = feats["text_part_raw"]                                    # [B, 6, D_proj]

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
            local_anchor_tokens = local_codebook_mean_anchors_raw.unsqueeze(0).expand(B, -1, -1)  # [B, 5, D]
            local_centroids = local_anchor_tokens
            route_centroids = local_centroids                             # [B, 5, D]
            route_part_mask_used = None

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
            "local_semantic_visual_tokens": None,
            "semantic_visual_tokens":       None,
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
        router_kwargs = {
            "visual_tokens":    visual_tokens_for_routing,
            "text_part_tokens": route_centroids_aug,           # [B, 5/6 (+ null), D]
            "visual_mask":      visual_attention_mask,
            "part_mask":        route_part_mask_used_aug,
        }
        if self.router_type == "sinkhorn":
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
        r_out = self.router(**router_kwargs)
        full_routing_matrix = r_out["routing_matrix"]              # [B, N, 5/6 (+ null)]
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
        routing_fraction_top1 = (routing_effective_k <= 1.0).to(local_routing_matrix.dtype).mean()
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

        q_out = self.quantizer(quant_input)
        quantized_tokens     = q_out["quantized_tokens"]      # [B, 6, D]   STE
        quantized_tokens_raw = q_out["quantized_tokens_raw"]  # [B, 6, D]   pure codewords
        codebook_indices     = q_out["codebook_indices"]      # [B, 6]
        codebook_distances   = q_out["codebook_distances"]    # [B, 6, K]

        # 7) gated global addition: blend C_global codeword into local codewords
        #    q_conditioned_m = q_local_m + sigmoid(alpha_m) * (sg)q_global
        # `--disable_global_gate` skips the addition entirely (v23b ablation).
        q_global_cw = quantized_tokens[:, 0, :]                # [B, D]
        q_local_cw  = quantized_tokens[:, 1:, :]               # [B, 5, D]

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
            cb_full = self.quantizer.codebooks                        # [M, K_max, D]
            cb_mask = self.quantizer.active_mask                      # [M, K_max] bool
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
        # v109: include lambda_text_hash_ntxent in the activation check so that
        # the text codon path is computed even when only the InfoNCE-form
        # visual-textual contrastive loss is enabled (lambda_text_hash=0 +
        # lambda_text_hash_ntxent>0). Previously gated only by MSE-form +
        # cw_xmodal.
        _lam_text_hash_ntxent_local = float(getattr(self, "lambda_text_hash_ntxent", 0.0))
        _text_path_active = (
            (float(self.lambda_text_hash)        > 0.0
             or _lam_text_hash_ntxent_local      > 0.0
             or float(self.lambda_cw_xmodal)     > 0.0)
            and text_quantizer_tokens is not None
            and text_quantizer_tokens.shape == (B, NUM_SEMANTIC_PARTS, D)
        )
        if _text_path_active:
            # (a) text quantization, EMA-disabled
            prev_train = self.quantizer.training
            self.quantizer.eval()
            try:
                tq_out = self.quantizer(text_quantizer_tokens)
            finally:
                if prev_train:
                    self.quantizer.train()
            text_q_st       = tq_out["quantized_tokens"]       # [B, 6, D]  STE
            text_q_raw      = tq_out["quantized_tokens_raw"]   # [B, 6, D]  codeword
            text_cb_indices = tq_out["codebook_indices"]       # [B, 6]
            # v93 exposure: text codeword (STE form so the InfoNCE gradient
            # flows back through text_adapter via straight-through).
            text_quantized_tokens = text_q_st
            # (b) text codon-residual
            if self.codon_residual_gamma > 0.0:
                text_codon_residual = text_quantizer_tokens - text_q_raw
            else:
                text_codon_residual = None
            # (c) skip global gate -> head input == text codeword directly
            text_head_inputs = text_q_st
            text_continuous_list = []
            for m, head in enumerate(self.codon_heads):
                r_m = (text_codon_residual[:, m, :]
                       if text_codon_residual is not None else None)
                t_m = (text_quantizer_tokens[:, m, :] if self.codon_text_anchor else None)
                h_out = head(text_head_inputs[:, m, :],
                             residual=r_m,
                             gamma=self.codon_residual_gamma,
                             text_chunks=t_m)
                text_continuous_list.append(h_out["continuous_code"])  # [B, 3, 4]
            text_continuous_code = torch.stack(text_continuous_list, dim=1).reshape(B, Mp3, 4)

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
            "routing_mean_effective_k":      routing_mean_effective_k.detach(),
            "routing_fraction_top1":         routing_fraction_top1.detach(),
            "local_semantic_visual_tokens": local_semantic_visual_tokens,
            "semantic_visual_tokens":       semantic_visual_tokens,
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
            "codebooks_buffer":                  self.quantizer.codebooks,
            "codebook_active_mask":              self.quantizer.active_mask,
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
        return out
