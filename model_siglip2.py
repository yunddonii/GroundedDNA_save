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
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.pretrained_backbone import build_pretrained_backbone, DEFAULT_BACKBONE
from models.visual_encoder import VisualEncoder
from models.text_encoder import TextEncoder
from models.adapters import (
    VisualAdapter, TextAdapter,
    VisualLinearAdapter, TextLinearAdapter,
)
from models.semantic_router import SemanticSinkhornRouter, SemanticAttentionRouter


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
        # internal step counter (training-only)
        self.register_buffer(
            "_revive_step", torch.zeros((), dtype=torch.long), persistent=False,
        )

        # codebooks: [M, K, D]
        # init: scaled Normal -- standard VQ init at 1/sqrt(D)
        cb_init = torch.empty(self.num_codebooks, self.codebook_size, self.d_model)
        nn.init.normal_(cb_init, mean=0.0, std=1.0 / math.sqrt(self.d_model))
        if self.update_mode == "gradient":
            # legacy path: codebook learns through the VQ MSE term in the loss
            self.codebooks = nn.Parameter(cb_init.clone())
        else:
            # EMA path (VQ-VAE-2 / DALL-E style):
            # codebook is a buffer updated in-place every training forward.
            #   cluster_size : [M, K]    EMA-smoothed assignment counts
            #   embed_avg    : [M, K, D] EMA-smoothed weighted sum of inputs
            self.register_buffer("codebooks",    cb_init.clone())
            self.register_buffer("cluster_size", torch.ones(self.num_codebooks, self.codebook_size))
            self.register_buffer("embed_avg",    cb_init.clone())

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
        codewords from dominating the mean.

        Args:
            exclude_global: skip codebook 0 (the C_global slot) — local routing
                            consumes only the 5 local anchors.

        Returns:
            anchors: [M_local, D] if exclude_global else [M, D]
        """
        cb = self.codebooks[1:] if exclude_global else self.codebooks   # [M', K, D]
        cb_n   = F.normalize(cb, dim=-1)                                # [M', K, D]
        anchors = cb_n.mean(dim=1)                                      # [M', D]
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
        K = self.codebook_size
        assert M == self.num_codebooks, (M, self.num_codebooks)
        n_revived = 0
        for m in range(M):
            cs   = self.cluster_size[m]                                       # [K]
            cs_max = cs.max()
            thr  = cs_max * self.revive_threshold
            dead_mask = cs < thr                                              # [K] bool
            n_dead = int(dead_mask.sum().item())
            if n_dead == 0:
                continue
            # sample fresh z's from the current batch with replacement
            idx_pool   = torch.randint(0, B, (n_dead,), device=z.device)
            new_codes  = z[idx_pool, m, :]                                    # [n_dead, D]
            # cluster_size to assign: median of currently-active codes (or 1.0
            # if every code in the codebook was dead).
            active_cs  = cs[~dead_mask]
            if active_cs.numel() > 0:
                init_cs = active_cs.median().clamp_min(1.0)
            else:
                init_cs = torch.tensor(1.0, device=cs.device, dtype=cs.dtype)
            self.codebooks[m, dead_mask]    = new_codes
            self.cluster_size[m, dead_mask] = init_cs
            self.embed_avg[m, dead_mask]    = new_codes * init_cs
            n_revived += n_dead
        return n_revived

    @torch.no_grad()
    def _ema_update(self, z: torch.Tensor, indices: torch.Tensor) -> None:
        """In-place EMA update of self.codebooks using new (z, indices) pairs.

        Shapes:
            z       : [B, M, D]
            indices : [B, M]
            cluster_size : [M, K]
            embed_avg    : [M, K, D]
        """
        B, M, D = z.shape
        K = self.codebook_size
        assert M == self.num_codebooks, (M, self.num_codebooks)
        # one-hot per codebook -- [M, B, K]
        onehot = F.one_hot(indices.transpose(0, 1), num_classes=K).to(z.dtype)  # [M, B, K]
        cluster_size_b = onehot.sum(dim=1)                                       # [M, K]
        z_perm = z.transpose(0, 1)                                               # [M, B, D]
        # weighted sum of z's mapped to each codeword (per codebook)
        embed_sum = torch.bmm(onehot.transpose(1, 2), z_perm)                    # [M, K, D]
        # decay both running stats
        self.cluster_size.mul_(self.ema_decay).add_(cluster_size_b, alpha=(1.0 - self.ema_decay))
        self.embed_avg   .mul_(self.ema_decay).add_(embed_sum,      alpha=(1.0 - self.ema_decay))
        # Laplace-smoothed normalization to avoid div-by-zero on dead codes
        n_total = self.cluster_size.sum(dim=-1, keepdim=True)                    # [M, 1]
        cs = (self.cluster_size + self.ema_eps) / (n_total + K * self.ema_eps) * n_total  # [M, K]
        self.codebooks.copy_(self.embed_avg / cs.unsqueeze(-1))

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

        # squared-L2 distance via expansion (M and K are small here -- 6 and {64,32})
        # diff: [B, M, K, D] -> distances: [B, M, K]
        diff = semantic_visual_tokens.unsqueeze(2) - self.codebooks.unsqueeze(0)
        distances = (diff ** 2).sum(dim=-1)                                # [B, M, K]
        indices   = distances.argmin(dim=-1)                               # [B, M]

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
    ) -> None:
        super().__init__()
        if d_model % 3 != 0:
            raise ValueError(
                f"[CodonHead] d_model must be divisible by 3 for codon-position "
                f"heads; got d_model={d_model}."
            )
        self.d_model: int = int(d_model)
        self.chunk:   int = self.d_model // 3
        # v62 (Option A): residual-conditioned codon head. When enabled, the
        # forward expects an optional `residual` input (= z - q, pre-quant
        # minus post-quant) and concats `(x, gamma * residual)` -> [2D] then
        # projects back to D before the per-chunk 4-class head. Allows the
        # codon head to depend on image-specific residual info in addition
        # to the quantized codeword index. Disabled by default; checkpoint-
        # compatible because input_proj is only constructed when used.
        self.use_residual: bool = bool(use_residual)
        if self.use_residual:
            self.input_proj = nn.Linear(2 * self.d_model, self.d_model)
        else:
            self.input_proj = None
        self.fc = nn.Linear(self.chunk, 4)
        self.use_gumbel_softmax: bool = bool(use_gumbel_softmax)
        self.gumbel_tau: float = float(gumbel_tau)

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        gamma: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        # x: [B, D]; optional residual: [B, D]; gamma: scaling factor
        B, D = x.shape
        assert D == self.d_model, (
            f"[CodonHead] expected last-dim {self.d_model}, got {D}"
        )
        # v62 residual injection
        if self.use_residual and residual is not None and gamma > 0:
            assert residual.shape == x.shape, (
                f"[CodonHead] residual shape {tuple(residual.shape)} != "
                f"x shape {tuple(x.shape)}"
            )
            combined = torch.cat([x, gamma * residual], dim=-1)          # [B, 2D]
            x = self.input_proj(combined)                                 # [B, D]
        h      = x.view(B, 3, self.chunk)                                # [B, 3, D/3]
        logits = self.fc(h)                                              # [B, 3, 4]
        cont   = F.softmax(logits, dim=-1)                               # [B, 3, 4]
        idx    = cont.argmax(dim=-1)                                     # [B, 3]
        hard   = F.one_hot(idx, num_classes=4).to(cont.dtype)            # [B, 3, 4]

        if self.training:
            if self.use_gumbel_softmax:
                # F.gumbel_softmax with hard=True does ST internally:
                # forward = one-hot, backward = soft.
                st = F.gumbel_softmax(
                    logits, tau=self.gumbel_tau, hard=True, dim=-1,
                )                                                        # [B, 3, 4]
            else:
                # deterministic straight-through estimator
                st = hard - cont.detach() + cont                         # [B, 3, 4]
        else:
            # at eval, both hard and st collapse to the deterministic argmax
            st = hard

        return {
            "logits":             logits,   # [B, 3, 4]
            "continuous_code":    cont,     # [B, 3, 4]
            "base_indices":       idx,      # [B, 3]
            "dna_hash_code":      st,       # train: ST, eval: hard
            "dna_hash_code_hard": hard,     # always deterministic one-hot
            "dna_hash_code_st":   st,       # always the gradient-bearing path
        }


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
        backbone_name = (
            getattr(args, "siglip2_backbone", None)
            or getattr(args, "backbone_name", None)
            or DEFAULT_BACKBONE
        )
        self.backbone = build_pretrained_backbone(backbone_name)

        # The two encoders share the same dual-encoder OBJECT, but read
        # disjoint sub-modules: vision_model vs text_model. The two towers
        # have no tied weights, by SigLIP2 design.
        self.visual_encoder = VisualEncoder(backbone=self.backbone)
        self.text_encoder   = TextEncoder  (backbone=self.backbone)

        # ---------- d_model & adapter dims --------------------------------
        proj_dim = self.visual_encoder.get_output_dim()  # SigLIP2 projection_dim
        self.proj_dim: int = int(proj_dim)
        d_model_arg = getattr(args, "d_model", None)
        self.d_model: int = int(d_model_arg) if d_model_arg is not None else int(proj_dim)
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
        else:
            raise ValueError(
                f"[model_siglip2] router_type must be 'sinkhorn' or 'attention', "
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
        # Each maps [B, D] -> [B, 3, 4] (3 codon positions × 4 base classes).
        if self.d_model % 3 != 0:
            raise ValueError(
                f"[model_siglip2] d_model must be divisible by 3 for codon-position "
                f"heads; got d_model={self.d_model}."
            )
        # Differentiable hard-code path config (Gumbel-Softmax STE by default).
        self.use_gumbel_softmax: bool = bool(getattr(args, "use_gumbel_softmax", True))
        self.gumbel_tau: float       = float(getattr(args, "gumbel_tau", 1.0))
        # v62 (Option A): residual-conditioned codon head
        self.codon_residual_gamma: float = float(getattr(args, "codon_residual_gamma", 0.0))
        _use_residual_codon = self.codon_residual_gamma > 0.0
        self.codon_heads = nn.ModuleList(
            [
                CodonHead(
                    self.d_model,
                    use_gumbel_softmax=self.use_gumbel_softmax,
                    gumbel_tau=self.gumbel_tau,
                    use_residual=_use_residual_codon,
                )
                for _ in range(self.num_codebooks)
            ]
        )

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
        if use_text_routing and feats["text_part_raw"] is not None:
            raw = feats["text_part_raw"]                                    # [B, 6, D_proj]

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

        # 3) pick the centroids that drive the 5-part Sinkhorn router
        local_anchor_tokens: Optional[torch.Tensor] = None
        if use_text_routing:
            local_centroids   = local_text_tokens                       # [B, 5, D]
            local_part_mask_used = part_mask[:, 1:] if part_mask is not None else None
        else:
            local_anchor_tokens = local_codebook_mean_anchors_raw.unsqueeze(0).expand(B, -1, -1)  # [B, 5, D]
            local_centroids = local_anchor_tokens
            local_part_mask_used = None

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
            "local_semantic_visual_tokens": None,
            "semantic_visual_tokens":       None,
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
            # Expand learnable null to per-sample dim and prepend (= part index 5)
            B_loc = local_centroids.shape[0]
            null_exp = self.null_centroid.unsqueeze(0).unsqueeze(0).expand(B_loc, 1, -1)  # [B, 1, D]
            local_centroids_aug = torch.cat([local_centroids, null_exp], dim=1)         # [B, 6, D]
            if local_part_mask_used is not None:
                # Always-keep null part in mask
                ones = torch.ones(B_loc, 1, dtype=local_part_mask_used.dtype,
                                  device=local_part_mask_used.device)
                local_part_mask_used_aug = torch.cat([local_part_mask_used, ones], dim=1)
            else:
                local_part_mask_used_aug = None
        else:
            local_centroids_aug = local_centroids
            local_part_mask_used_aug = local_part_mask_used

        router_kwargs = {
            "visual_tokens":    visual_tokens,
            "text_part_tokens": local_centroids_aug,           # [B, 5 or 6, D]
            "visual_mask":      visual_attention_mask,
            "part_mask":        local_part_mask_used_aug,
        }
        if self.router_type == "sinkhorn":
            cur_eps = self._current_sinkhorn_epsilon()
            if cur_eps is not None:
                router_kwargs["epsilon_override"] = cur_eps
            if self.routing_topk is not None:
                router_kwargs["topk_per_patch"]   = int(self.routing_topk)
            if self.routing_topp is not None:
                router_kwargs["topp_per_patch"]   = float(self.routing_topp)
            if self.sinkhorn_lambda_a is not None:
                router_kwargs["uot_lambda_a"]     = float(self.sinkhorn_lambda_a)
            if self.sinkhorn_lambda_b is not None:
                router_kwargs["uot_lambda_b"]     = float(self.sinkhorn_lambda_b)
        r_out = self.router(**router_kwargs)
        full_routing_matrix = r_out["routing_matrix"]              # [B, N, 5 or 6]
        if self.use_null_centroid and self.null_centroid is not None:
            # Slice null (last) column out — null mass naturally rejected
            # from codebook updates. Remaining 5 columns retain whatever
            # mass the Sinkhorn (balanced or UOT) assigned to real parts.
            local_routing_matrix = full_routing_matrix[..., :-1]    # [B, N, 5]
        else:
            local_routing_matrix = full_routing_matrix              # [B, N, 5]

        # explicit re-normalize for numerical stability
        denom = local_routing_matrix.sum(dim=1).clamp_min(1e-6)                       # [B, 5]
        local_semantic_visual_tokens = (
            torch.bmm(local_routing_matrix.transpose(1, 2), visual_tokens)            # [B, 5, D]
            / denom.unsqueeze(-1)
        )

        # 5) stitch [global | local] into the [B, N, 6] / [B, 6, D] interfaces
        routing_matrix = torch.cat(
            [global_weights, local_routing_matrix], dim=-1,
        )                                                                              # [B, N, 6]
        semantic_visual_tokens = torch.cat(
            [global_visual_token.unsqueeze(1), local_semantic_visual_tokens], dim=1,
        )                                                                              # [B, 6, D]

        # 6) v32: train-only text injection into the quantizer input.
        # When `--text_inject_train_only=add` and the text path is active,
        # we add `alpha * t_m` to each routed visual token *before*
        # codebook lookup, so the codeword embeddings absorb text-semantic
        # structure during training. At eval/inference we skip this so the
        # forward pass stays text-free (no Qwen captions needed). When
        # `text_inject_detach=True` (variant b) the text gradient is cut
        # so only the codebook moves, not the text adapter.
        quant_input = semantic_visual_tokens
        if (
            self.training
            and self.text_inject_train_only == "add"
            and self.text_inject_alpha > 0.0
            and text_part_tokens is not None
            and text_part_tokens.shape == semantic_visual_tokens.shape
        ):
            t_used = text_part_tokens.detach() if self.text_inject_detach else text_part_tokens
            quant_input = semantic_visual_tokens + self.text_inject_alpha * t_used

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
        for m, head in enumerate(self.codon_heads):
            r_m = codon_residual[:, m, :] if codon_residual is not None else None
            h_out = head(head_inputs[:, m, :], residual=r_m,
                         gamma=self.codon_residual_gamma)            # dict
            codon_logits_list.append(h_out["logits"])                # [B, 3, 4]
            continuous_list.append (h_out["continuous_code"])        # [B, 3, 4]
            hash_list.append       (h_out["dna_hash_code"])          # [B, 3, 4]
            hard_list.append       (h_out["dna_hash_code_hard"])     # [B, 3, 4]
            st_list.append         (h_out["dna_hash_code_st"])       # [B, 3, 4]
            indices_list.append    (h_out["base_indices"])           # [B, 3]
        codon_logits_per_codebook        = torch.stack(codon_logits_list, dim=1)  # [B, 6, 3, 4]
        continuous_codes_per_codebook    = torch.stack(continuous_list,   dim=1)  # [B, 6, 3, 4]
        dna_hash_codes_per_codebook      = torch.stack(hash_list,         dim=1)  # [B, 6, 3, 4]
        dna_hash_codes_per_codebook_hard = torch.stack(hard_list,         dim=1)  # [B, 6, 3, 4]
        dna_hash_codes_per_codebook_st   = torch.stack(st_list,           dim=1)  # [B, 6, 3, 4]
        base_indices_per_codebook        = torch.stack(indices_list,      dim=1)  # [B, 6, 3]

        # 9) flatten to a single concatenated code: 6 codebooks * 3 codon positions = 18
        Mp3 = NUM_SEMANTIC_PARTS * 3
        continuous_code    = continuous_codes_per_codebook   .reshape(B, Mp3, 4)   # [B, 18, 4]
        dna_hash_code      = dna_hash_codes_per_codebook     .reshape(B, Mp3, 4)   # [B, 18, 4]
        dna_hash_code_hard = dna_hash_codes_per_codebook_hard.reshape(B, Mp3, 4)   # [B, 18, 4]
        dna_hash_code_st   = dna_hash_codes_per_codebook_st  .reshape(B, Mp3, 4)   # [B, 18, 4]
        base_indices       = base_indices_per_codebook       .reshape(B, Mp3)      # [B, 18]

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
        assert codebook_distances.shape == (B, NUM_SEMANTIC_PARTS, self.codebook_size)
        assert head_inputs.shape == (B, NUM_SEMANTIC_PARTS, D)
        assert gate_values.shape == (NUM_LOCAL_PARTS,)
        assert continuous_codes_per_codebook.shape == (B, NUM_SEMANTIC_PARTS, 3, 4)
        assert dna_hash_codes_per_codebook  .shape == (B, NUM_SEMANTIC_PARTS, 3, 4)
        assert base_indices_per_codebook    .shape == (B, NUM_SEMANTIC_PARTS, 3)
        assert continuous_code.shape == (B, Mp3, 4)
        assert dna_hash_code  .shape == (B, Mp3, 4)
        assert base_indices   .shape == (B, Mp3)

        out.update({
            "local_routing_matrix":         local_routing_matrix,
            "routing_matrix":               routing_matrix,
            "local_semantic_visual_tokens": local_semantic_visual_tokens,
            "semantic_visual_tokens":       semantic_visual_tokens,
            # entropic-OT cost (W_e) per sample, [B]. Used by the Wasserstein
            # alignment loss to push visual_adapter and text_adapter into a
            # shared space where the patch <-> text-part coupling is "cheap".
            "ot_cost":                      r_out.get("ot_cost", None),
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
        return out
