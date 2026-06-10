"""v141: Cluster-Attention Router (DiVT-inspired) for M=6 semantic parts.

Replaces the Sinkhorn-OT router that operates on (patches, text_centroids) with
a *visual-side clustering* + *DiVT-style soft-masked cross-attention* design:

  patches [B, N, D]
     |
     |  Soft Sinkhorn-balanced cluster assignment (DINO/SwAV-style)
     |  using learnable prototypes [M, D]
     v
  P_cluster [B, N, M]    (rows sum to 1, columns balanced ~ N/M)
     |
     |  Attention-pooled centroid (differentiable)
     |    centroid_m = sum_n softmax_n(score(x_n, mean_m)) * x_n
     v
  centroids [B, M, D]
     |
     |  Soft-masked cross-attention (DiVT Eq. 2-4)
     |    Q from centroid, K from patches, V from patches + pos_emb
     |    Soft mask = log(P_cluster)  (replaces DiVT's hard -inf mask)
     v
  visual tokens [B, M, D]

The downstream visual_adapter (parent model) projects [B, M, D] to D_model
*after* this router, so its gradient flows through the cross-attention path
all the way back to the encoder (no hard-clustering gradient cut-off).

At inference time we set ``training=False``: the attention-pooled centroid
and masked cross-attention are skipped and the M tokens are obtained as a
simple cluster-weighted sum (equivalent to the v133a inference path, but
using cluster prototypes instead of codebook anchors as the cost source).
This trades a small train/test distribution mismatch for ~3x faster
inference.

Author: GroundedDNA (2026-06-10, v141a)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClusterAttentionRouter(nn.Module):
    """DiVT-inspired soft cluster + masked cross-attention router.

    Parameters
    ----------
    num_parts : int
        Number of semantic parts M (default 6, matches GroundedDNA codebooks).
    d_model : int
        Dimensionality of the input patch tokens (must match the encoder's
        visual hidden dim, since the router operates on RAW patches before
        the parent model's visual_adapter).
    num_heads : int
        Multi-head attention head count (default 4). d_model must be
        divisible by num_heads.
    mlp_ratio : float
        Hidden dim ratio for the post-attention MLP. Default 4.0
        (transformer convention).
    sinkhorn_eps : float
        Temperature for log-domain Sinkhorn clustering.
    sinkhorn_iters : int
        Number of Sinkhorn iterations (3-5 sufficient for ~uniform balance).
    pool_temperature : float
        Temperature for the attention-pooled centroid softmax.
    max_patches : int
        Maximum patch count supported by the learnable positional embedding.
    pos_emb_init_std : float
        Std for trunc_normal initialization of the positional embedding.
    """

    def __init__(
        self,
        num_parts: int = 6,
        d_model: int = 768,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        sinkhorn_eps: float = 0.1,
        sinkhorn_iters: int = 3,
        pool_temperature: float = 0.3,
        max_patches: int = 1024,
        pos_emb_init_std: float = 0.02,
        proto_init_std: float = 0.01,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )
        self.M = int(num_parts)
        self.D = int(d_model)
        self.num_heads = int(num_heads)
        self.head_dim = self.D // self.num_heads
        self.sinkhorn_eps = float(sinkhorn_eps)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.pool_temperature = float(pool_temperature)
        self.max_patches = int(max_patches)

        # Learnable cluster prototypes (used for both Sinkhorn cost matrix
        # at train AND inference time -- they are the *routing anchors*).
        self.prototypes = nn.Parameter(
            torch.randn(self.M, self.D) * float(proto_init_std)
        )

        # Cross-attention projections (DiVT Eq. 2).
        self.W_Q = nn.Linear(self.D, self.D, bias=True)
        self.W_K = nn.Linear(self.D, self.D, bias=True)
        self.W_V = nn.Linear(self.D, self.D, bias=True)
        # Output projection after attention.
        self.W_O = nn.Linear(self.D, self.D, bias=True)

        # Layer norms (pre-norm transformer style).
        self.norm_q = nn.LayerNorm(self.D)
        self.norm_k = nn.LayerNorm(self.D)
        self.norm_mlp = nn.LayerNorm(self.D)

        # MLP after attention (residual).
        mlp_hidden = int(self.D * float(mlp_ratio))
        self.mlp = nn.Sequential(
            nn.Linear(self.D, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, self.D),
        )

        # Positional embedding added to V only (DiVT Eq. 2 design choice).
        self.pos_emb = nn.Parameter(
            torch.empty(self.max_patches, self.D)
        )
        nn.init.trunc_normal_(self.pos_emb, std=float(pos_emb_init_std))

    # ------------------------------------------------------------------ #
    # Core operations                                                    #
    # ------------------------------------------------------------------ #

    def sinkhorn_soft_cluster(
        self,
        patches: torch.Tensor,           # [B, N, D]
        eps: float = None,
        iters: int = None,
    ) -> torch.Tensor:
        """Sinkhorn-balanced soft cluster assignment.

        Computes a transport plan P in R^{B,N,M} such that:
          - rows sum to 1 (each patch's assignment is a probability dist.)
          - columns are approximately balanced (each cluster ~ N/M mass)

        Uses log-domain iteration for numerical stability.
        """
        if eps is None:
            eps = self.sinkhorn_eps
        if iters is None:
            iters = self.sinkhorn_iters
        eps_f = max(float(eps), 1e-3)

        B, N, D = patches.shape
        # Cosine similarity between patches and prototypes (anchors).
        proto_n = F.normalize(self.prototypes, dim=-1)          # [M, D]
        patches_n = F.normalize(patches, dim=-1)                 # [B, N, D]
        sim = torch.einsum("bnd,md->bnm", patches_n, proto_n)    # [B, N, M]

        log_K = sim / eps_f                                       # [B, N, M]
        # Log-uniform col mass target: each cluster receives N/M units of
        # mass total (so the joint P sums to N per batch). Implemented as
        # log(N/M) = log_b.
        log_b = math.log(float(N) / float(self.M))

        # Log-domain Sinkhorn-Knopp iterations.
        log_u = torch.zeros(B, N, device=patches.device, dtype=patches.dtype)
        log_v = torch.zeros(B, self.M, device=patches.device, dtype=patches.dtype)
        for _ in range(int(iters)):
            log_u = -torch.logsumexp(log_K + log_v.unsqueeze(1), dim=-1)
            log_v = log_b - torch.logsumexp(log_K + log_u.unsqueeze(-1), dim=1)

        # Transport plan in log-space, then row-normalize to get a per-patch
        # probability distribution over clusters (required for downstream
        # cross-attention soft mask).
        log_T = log_K + log_u.unsqueeze(-1) + log_v.unsqueeze(1)
        log_T = log_T - torch.logsumexp(log_T, dim=-1, keepdim=True)
        P = log_T.exp()
        return P                                                  # [B, N, M]

    def cluster_pool(
        self,
        patches: torch.Tensor,           # [B, N, D]
        P_cluster: torch.Tensor,          # [B, N, M]
    ) -> torch.Tensor:
        """Soft cluster-weighted mean.

        Used both (a) as an intermediate for attention-pooled centroid
        computation and (b) as the inference-time output (cheaper than
        full cross-attention).
        """
        # Per-cluster mass (column sums).
        col_sum = P_cluster.sum(dim=1, keepdim=True).clamp_min(1e-6)  # [B, 1, M]
        # Normalize columns so each cluster's weights sum to 1.
        P_norm = P_cluster / col_sum                                   # [B, N, M]
        return torch.einsum("bnm,bnd->bmd", P_norm, patches)           # [B, M, D]

    def attention_pooled_centroid(
        self,
        patches: torch.Tensor,           # [B, N, D]
        P_cluster: torch.Tensor,          # [B, N, M]
    ) -> torch.Tensor:
        """Differentiable centroid: softmax-weighted within-cluster pool.

        Replaces the medoid (DiVT-style argmax) with a soft attention pool
        scored by cosine similarity to the cluster mean.
        """
        cluster_mean = self.cluster_pool(patches, P_cluster)           # [B, M, D]

        p_n = F.normalize(patches, dim=-1)
        cm_n = F.normalize(cluster_mean, dim=-1)
        score = torch.einsum("bnd,bmd->bnm", p_n, cm_n) / max(self.pool_temperature, 1e-3)  # [B, N, M]

        # Soft mask via log P_cluster: patches outside cluster m have
        # P~0 -> -inf log -> attention weight ~ 0.
        log_w = torch.log(P_cluster.clamp_min(1e-8)) + score           # [B, N, M]
        # Softmax over N per cluster.
        w = F.softmax(log_w, dim=1)                                    # [B, N, M]
        centroid = torch.einsum("bnm,bnd->bmd", w, patches)            # [B, M, D]
        return centroid

    def soft_masked_cross_attention(
        self,
        centroid: torch.Tensor,          # [B, M, D]
        patches: torch.Tensor,           # [B, N, D]
        P_cluster: torch.Tensor,          # [B, N, M]
    ) -> torch.Tensor:
        """DiVT-style cluster-restricted cross-attention with soft mask.

        Q comes from centroid (cluster identity), K from patches (content
        match), V from patches + positional embedding (spatial context).
        The DiVT hard mask M[m,i] = (0 if i in C_m else -inf) is replaced
        by a *soft* mask log(P_cluster) so gradient flows everywhere.
        """
        B, N, D = patches.shape
        M = centroid.shape[1]

        # Pre-norm.
        q_in = self.norm_q(centroid)                                   # [B, M, D]
        k_in = self.norm_k(patches)                                    # [B, N, D]

        Q = self.W_Q(q_in)                                              # [B, M, D]
        K = self.W_K(k_in)                                              # [B, N, D]
        pos = self.pos_emb[:N].unsqueeze(0).expand(B, -1, -1)            # [B, N, D]
        V = self.W_V(k_in + pos)                                        # [B, N, D]

        # Multi-head attention.
        Q = Q.view(B, M, self.num_heads, self.head_dim).transpose(1, 2)  # [B, h, M, d]
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, h, N, d]
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, h, N, d]

        # Attention logits per head: [B, h, M, N]
        attn_logit = torch.einsum("bhmd,bhnd->bhmn", Q, K) / math.sqrt(self.head_dim)

        # Soft mask: log(P_cluster) shared across heads.
        log_P = torch.log(P_cluster.clamp_min(1e-8))                    # [B, N, M]
        log_P = log_P.permute(0, 2, 1).unsqueeze(1)                      # [B, 1, M, N]
        attn_logit = attn_logit + log_P

        attn = F.softmax(attn_logit, dim=-1)                            # [B, h, M, N]
        attn_out = torch.einsum("bhmn,bhnd->bhmd", attn, V)             # [B, h, M, d]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, M, D)  # [B, M, D]
        attn_out = self.W_O(attn_out)                                   # [B, M, D]

        # Residual + MLP (transformer block style).
        out = centroid + attn_out
        out = out + self.mlp(self.norm_mlp(out))
        return out                                                       # [B, M, D]

    # ------------------------------------------------------------------ #
    # Forward                                                             #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        patches: torch.Tensor,           # [B, N, D]
        use_attention: bool = None,
    ) -> dict:
        """
        Parameters
        ----------
        patches : Tensor [B, N, D]
            RAW visual patch tokens (e.g. from ViT encoder, BEFORE the
            parent model's visual_adapter).
        use_attention : Optional[bool]
            If True, runs full cluster + attention path (train).
            If False, runs cheap weighted-sum path (inference).
            If None (default), follows self.training.

        Returns
        -------
        dict with keys
            "routing_matrix" : [B, N, M] (= P_cluster)
            "semantic_tokens" : [B, M, D]
            "ot_cost"         : per-sample mean Sinkhorn cost (for
                                downstream `loss_wasserstein`)
        """
        if use_attention is None:
            use_attention = bool(self.training)

        P_cluster = self.sinkhorn_soft_cluster(patches)                 # [B, N, M]

        if use_attention:
            centroid = self.attention_pooled_centroid(patches, P_cluster)
            tokens = self.soft_masked_cross_attention(
                centroid, patches, P_cluster,
            )
        else:
            tokens = self.cluster_pool(patches, P_cluster)

        # OT cost = <P, C> with C = 1 - cos(patch, prototype) in [0, 2].
        # Per-patch averaged so the magnitude matches v133a's
        # loss_wasserstein (~0.4 typical). Gradient flows through BOTH P
        # (Sinkhorn assignment) and self.prototypes (via sim), pulling
        # prototypes toward cluster means and tightening clusters.
        proto_n = F.normalize(self.prototypes, dim=-1)
        patches_n = F.normalize(patches, dim=-1)
        sim = torch.einsum("bnd,md->bnm", patches_n, proto_n)            # [B, N, M]
        cost = 1.0 - sim                                                 # [B, N, M] in [0, 2]
        # Per-batch transport cost normalized by patch count so the loss
        # magnitude is dataset-invariant.
        N_ = patches.shape[1]
        ot_cost = (P_cluster * cost).sum(dim=(1, 2)) / float(N_)         # [B]

        return {
            "routing_matrix": P_cluster,                                 # [B, N, M]
            "semantic_tokens": tokens,                                   # [B, M, D]
            "ot_cost": ot_cost,                                           # [B]
        }
