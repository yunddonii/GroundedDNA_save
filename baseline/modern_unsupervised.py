"""Clean-room mathematical cores for recent uni-modal hashing baselines.

The functions in this module were written from the published equations and
audited against the authors' public training programs.  No upstream source is
vendored.  They deliberately separate the paper objective from the repository
protocol adaptation (frozen cached CLIP features, 36 bits, held-out P0 split).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SDCFeatureHash(nn.Module):
    """SDC hash head with an explicit paper/release hidden width.

    The released implementation uses ``hidden_dim=d_in`` while the paper
    describes a 4096-wide hidden layer.  Keeping this an argument prevents a
    silent hybrid of the two specifications.
    """

    def __init__(self, d_in: int, bit: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = int(d_in if hidden_dim is None else hidden_dim)
        self.backbone = nn.Identity()
        self.encoder_layers = nn.Sequential(
            nn.Linear(d_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bit),
            nn.BatchNorm1d(bit),
        )

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = features.float()
        codes = self.encoder_layers(features)
        return {'continuous_code': codes, 'cnn_feat': features,
                'backbone_last_output': features}


@lru_cache(maxsize=64)
def _sdc_beta_quantiles(n_pairs: int, beta: float,
                        orthogonal_only: bool) -> np.ndarray:
    if n_pairs <= 0:
        raise ValueError('SDC needs at least one feature pair')
    if beta <= 0:
        raise ValueError('beta shape must be positive')
    from scipy.stats import beta as beta_distribution

    # Match the public implementation's float32 midpoint grid before SciPy
    # evaluates the inverse CDF; this removes a small avoidable audit delta.
    grid = np.arange(1.0, 2.0 * n_pairs, 2.0, dtype=np.float32) / (2.0 * n_pairs)
    target = 2.0 * beta_distribution.ppf(grid, beta, beta) - 1.0
    if orthogonal_only:
        target = np.maximum(target, 0.0)
    return target.astype(np.float32)


def sdc_loss(features: torch.Tensor, codes: torch.Tensor, *, beta: float = 5.0,
             reconstruction: str = 'l1', quantization: str = 'cosine',
             reconstruction_weight: float = 1.0,
             quantization_weight: float = 1.0,
             orthogonal_only: bool = False,
             contrastive_pair_layout: bool = False
             ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Similarity Distribution Calibration objective (BMVC 2023, Eq. 6)."""
    if features.shape[0] != codes.shape[0]:
        raise ValueError('features/codes batch mismatch')
    if features.shape[0] % 2:
        features, codes = features[:-1], codes[:-1]
    n_rows = features.shape[0]
    if n_rows < 2:
        raise ValueError('SDC requires a batch of at least two')

    if contrastive_pair_layout:
        # The two-view trainer concatenates [view1_B; view2_B].  The released
        # SDC pairing routine deliberately forms random first-half/second-half
        # pairs *within each view*, rather than treating two augmentations of
        # the same image as SDC calibration pairs.
        if n_rows % 4:
            raise ValueError(
                'two-view SDC layout needs [view1_B;view2_B] with even B')
        block = n_rows // 4
        a_idx = torch.cat((
            torch.arange(0, block, device=features.device),
            torch.arange(2 * block, 3 * block, device=features.device),
        ))
        b_idx = torch.cat((
            torch.arange(block, 2 * block, device=features.device),
            torch.arange(3 * block, 4 * block, device=features.device),
        ))
    else:
        half = n_rows // 2
        a_idx = torch.arange(0, half, device=features.device)
        b_idx = torch.arange(half, 2 * half, device=features.device)

    feature_similarity = F.cosine_similarity(
        features[a_idx], features[b_idx]).detach()
    code_similarity = F.cosine_similarity(codes[a_idx], codes[b_idx])
    order = torch.argsort(feature_similarity)
    ordered_code_similarity = code_similarity[order]
    target = torch.from_numpy(
        _sdc_beta_quantiles(len(a_idx), float(beta), bool(orthogonal_only))
    ).to(device=codes.device, dtype=codes.dtype)

    if reconstruction == 'l1':
        rec = F.l1_loss(ordered_code_similarity, target)
    elif reconstruction == 'l2':
        rec = F.mse_loss(ordered_code_similarity, target)
    elif reconstruction == 'hinge':
        rec = F.relu(ordered_code_similarity - target).mean()
    else:
        raise ValueError(f'unknown SDC reconstruction={reconstruction!r}')

    signed = torch.sign(codes)
    if quantization == 'cosine':
        quan = (1.0 - F.cosine_similarity(codes, signed)).mean()
    elif quantization == 'l1':
        quan = (codes - signed).abs().mean()
    elif quantization == 'l2':
        quan = (codes - signed).square().mean()
    else:
        raise ValueError(f'unknown SDC quantization={quantization!r}')
    total = reconstruction_weight * rec + quantization_weight * quan
    return total, {'reconstruction': rec, 'quantization': quan}


def simclr_nt_xent(first: torch.Tensor, second: torch.Tensor,
                   temperature: float = 0.3) -> torch.Tensor:
    """Symmetric SimCLR NT-Xent used by SDC's two-view configuration."""
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError('NT-Xent expects two [B,D] tensors of identical shape')
    if first.shape[0] < 2:
        raise ValueError('NT-Xent needs at least two paired samples')
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    n = first.shape[0]
    z = F.normalize(torch.cat((first, second), dim=0), dim=1)
    logits = z @ z.T / float(temperature)
    logits.fill_diagonal_(-torch.inf)
    positives = torch.cat((
        torch.arange(n, 2 * n, device=z.device),
        torch.arange(0, n, device=z.device),
    ))
    return F.cross_entropy(logits, positives)


class DUHEGFeatureHash(nn.Module):
    """DUH-EG's symmetric image/external-guidance hashing heads."""

    def __init__(self, image_dim: int, text_dim: int, bit: int,
                 hidden_dim: int = 512):
        super().__init__()
        self.backbone = nn.Identity()
        self.encoder_layers = self._head(image_dim, hidden_dim, bit)
        self.text_encoder = self._head(text_dim, hidden_dim, bit)

    @staticmethod
    def _head(d_in: int, hidden: int, bit: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, bit),
            nn.Tanh(),
        )

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        # The released DUH-EG artifacts normalize every canonical, augmented,
        # query, and database CLIP vector before either hashing head.
        features = F.normalize(features.float(), dim=1)
        codes = self.encoder_layers(features)
        return {'continuous_code': codes, 'cnn_feat': features,
                'backbone_last_output': features}

    def encode_guidance(self, guidance: torch.Tensor) -> torch.Tensor:
        return self.text_encoder(F.normalize(guidance.float(), dim=1))


class UMRCHFeatureHash(nn.Module):
    """UMRCH linear hash head with its released CLIP normalization path."""

    def __init__(self, d_in: int, bit: int):
        super().__init__()
        self.backbone = nn.Identity()
        self.encoder_layers = nn.Linear(d_in, bit)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = F.normalize(features.float(), dim=1)
        codes = self.encoder_layers(features)
        return {'continuous_code': codes, 'cnn_feat': features,
                'backbone_last_output': features}


@torch.no_grad()
def duheg_external_guidance(image_features: torch.Tensor,
                            noun_features: torch.Tensor,
                            temperature: float = 0.004,
                            chunk_size: int = 8192,
                            compute_dtype: torch.dtype = torch.float32,
                            ) -> torch.Tensor:
    """Create the per-image WordNet guidance vector used by DUH-EG."""
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    if compute_dtype not in (torch.float16, torch.float32, torch.float64):
        raise ValueError(f'unsupported DUH-EG guidance dtype: {compute_dtype}')
    # The public DUH-EG preparation first normalizes the NumPy feature arrays
    # in their saved FP32 precision and only then casts both operands to FP16
    # inside ``combine_nouns``.  Normalizing after the cast is measurably
    # different at the release temperature (0.004).
    image_features = F.normalize(image_features.float(), dim=1).to(compute_dtype)
    noun_features = F.normalize(noun_features.float(), dim=1).to(compute_dtype)
    if image_features.shape[1] != noun_features.shape[1]:
        raise ValueError('image/noun embedding dimensions differ')
    chunks = []
    for start in range(0, image_features.shape[0], chunk_size):
        image = image_features[start:start + chunk_size]
        weights = torch.softmax(image @ noun_features.T / temperature, dim=1)
        # ``combine_nouns`` normalizes the aggregate in FP16. ``sim_word``
        # immediately normalizes that saved FP16 array once more before
        # TensorDataset converts it to FP32, so preserve both roundings.
        aggregate = F.normalize(weights @ noun_features, dim=1)
        aggregate = F.normalize(aggregate, dim=1)
        chunks.append(aggregate.float())
    return torch.cat(chunks, dim=0)


def duheg_multi_positive_nce(first: torch.Tensor, second: torch.Tensor,
                             positive_mask: torch.Tensor,
                             temperature: float) -> torch.Tensor:
    """DUH-EG's row-wise multi-positive cross-space contrastive loss."""
    first = F.normalize(first, dim=1)
    second = F.normalize(second, dim=1)
    probabilities = torch.softmax(first @ second.T / temperature, dim=1)
    probabilities = probabilities.clamp(1e-12, 1.0 - 1e-12)
    positive_mass = (positive_mask.to(probabilities.dtype) * probabilities).sum(dim=1)
    return -torch.log(positive_mass.clamp_min(1e-12)).mean()


@torch.no_grad()
def uhscm_concept_probabilities(image_features: torch.Tensor,
                                concept_features: torch.Tensor,
                                temperature_scale: float = 3.0,
                                chunk_size: int = 4096) -> Tuple[torch.Tensor, torch.Tensor]:
    """UHSCM concept denoising and normalized per-image distributions."""
    images = F.normalize(image_features.float(), dim=1)
    concepts = F.normalize(concept_features.float(), dim=1)
    if images.shape[1] != concepts.shape[1]:
        raise ValueError('image/concept embedding dimensions differ')
    n_concepts = concepts.shape[0]
    counts = torch.zeros(n_concepts, device=images.device, dtype=torch.long)
    scale = n_concepts * float(temperature_scale)
    for start in range(0, images.shape[0], chunk_size):
        logits = images[start:start + chunk_size] @ concepts.T * scale
        counts += torch.bincount(logits.argmax(dim=1), minlength=n_concepts)
    selected = counts.float() > (images.shape[0] / float(n_concepts)) * 0.5
    if not bool(selected.any()):
        raise RuntimeError('UHSCM concept denoising selected no concepts')

    scale = int(selected.sum()) * float(temperature_scale)
    distributions = []
    for start in range(0, images.shape[0], chunk_size):
        logits = images[start:start + chunk_size] @ concepts.T * scale
        logits[:, ~selected] = -torch.inf
        distributions.append(F.normalize(torch.softmax(logits, dim=1), dim=1))
    return torch.cat(distributions, dim=0), selected


def uhscm_loss(codes: torch.Tensor, concept_probabilities: torch.Tensor, *,
               similarity_threshold: float = 0.8, temperature: float = 0.2,
               quantization_weight: float = 0.001,
               contrastive_weight: float = 0.2) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """UHSCM similarity reconstruction, quantization, and contrastive terms."""
    codes = torch.tanh(codes)
    semantic = 2.0 * (F.normalize(concept_probabilities, dim=1)
                      @ F.normalize(concept_probabilities, dim=1).T) - 1.0
    code_norm = F.normalize(codes, dim=1)
    code_similarity = code_norm @ code_norm.T
    reconstruction = (code_similarity - semantic).square().mean()
    quantization = (torch.sign(codes) - codes).square().sum() / codes.shape[0]

    positive = (semantic >= similarity_threshold).to(codes.dtype)
    exponentiated = torch.exp(code_similarity / temperature)
    negative_mass = ((1.0 - positive) * exponentiated).sum(dim=1, keepdim=True) + 1e-5
    contrastive = -(torch.log(exponentiated / negative_mass) * positive).sum()
    contrastive = contrastive / positive.sum().clamp_min(1.0)
    total = reconstruction + quantization_weight * quantization + contrastive_weight * contrastive
    return total, {'reconstruction': reconstruction, 'quantization': quantization,
                   'contrastive': contrastive}


def umrch_semantic_similarity(global_features: torch.Tensor,
                              local_features: torch.Tensor,
                              concept_features: torch.Tensor, *,
                              vl_temperature: float = 0.01,
                              concept_threshold: float = 0.3,
                              negative_threshold: float = 0.0) -> torch.Tensor:
    """UMRCH global/local concept aggregation and soft-IoU similarity."""
    global_features = F.normalize(global_features.float(), dim=-1)
    local_features = F.normalize(local_features.float(), dim=-1)
    concepts = F.normalize(concept_features.float(), dim=-1)
    if global_features.shape[-1] != concepts.shape[-1]:
        raise ValueError('global/concept dimensions differ')
    if local_features.shape[-1] != concepts.shape[-1]:
        raise ValueError('local/concept dimensions differ')

    global_prob = torch.softmax(global_features @ concepts.T / vl_temperature, dim=-1)
    local_prob = torch.softmax(local_features @ concepts.T / vl_temperature, dim=-1)
    local_max = local_prob.max(dim=1).values
    local_min = local_prob.min(dim=1).values
    selected_local = torch.where(local_max > concept_threshold, local_max, local_min)
    fused = 0.5 * (selected_local + global_prob)
    activated = torch.where(fused > concept_threshold, fused, torch.zeros_like(fused))
    intersection = activated @ activated.T
    mass = activated.sum(dim=1)
    union = mass[:, None] + mass[None, :] - intersection
    soft_iou = torch.where(union > 0, intersection / union, torch.zeros_like(union))
    return torch.where(soft_iou > negative_threshold, soft_iou,
                       torch.zeros_like(soft_iou))


def umrch_loss(codes: torch.Tensor, semantic_similarity: torch.Tensor, *,
               contrastive_temperature: float = 1.0,
               distribution_weight: float = 1.0,
               contrastive_weight: float = 150.0) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """UMRCH multi-semantic reconstruction and weighted contrastive loss."""
    codes = torch.tanh(codes)
    code_similarity = F.normalize(codes, dim=1) @ F.normalize(codes, dim=1).T
    distribution = F.mse_loss(semantic_similarity / 0.1, code_similarity / 0.1)

    n = codes.shape[0] // 2
    if codes.shape[0] != 2 * n or n == 0:
        raise ValueError('UMRCH expects two equally sized augmented-view blocks')
    soft_labels = semantic_similarity.clone()
    idx = torch.arange(n, device=codes.device)
    soft_labels[idx, idx + n] = 1.0
    soft_labels[idx + n, idx] = 1.0
    mask = (soft_labels > 0).to(codes.dtype)
    logits = code_similarity / contrastive_temperature
    logits = logits - logits.max(dim=1, keepdim=True).values
    exp_logits = torch.exp(logits)
    positive_prob = exp_logits * mask
    negative_mass = (exp_logits * (1.0 - mask)).sum(dim=1, keepdim=True)
    eps = 1e-8
    log_fraction = torch.log(
        (positive_prob + eps) / (positive_prob + negative_mass + eps))
    weighted = torch.where(mask > 0, soft_labels * log_fraction,
                           torch.zeros_like(log_fraction))
    contrastive = -(weighted.sum(dim=1) / soft_labels.sum(dim=1).clamp_min(eps)).mean()
    total = distribution_weight * distribution + contrastive_weight * contrastive
    return total, {'distribution': distribution, 'contrastive': contrastive}


# ---------------------------------------------------------------------------
# HHCH (IEEE TIP 2024): clean-room paper equations


def poincare_project(points: torch.Tensor, curvature: float,
                     eps: float = 1e-3) -> torch.Tensor:
    """Project points inside the Poincare ball of curvature ``-curvature``.

    HHCH writes the ball as ``c ||x||^2 < 1`` and uses ``sqrt(c)`` in its
    distance and exponential-map equations.  Thus ``curvature`` here is the
    paper's positive parameter ``c`` (the sectional curvature is ``-c`` under
    this convention).
    """
    if curvature <= 0:
        raise ValueError('HHCH curvature must be positive')
    norm = points.norm(dim=-1, keepdim=True).clamp_min(1e-15)
    max_norm = (1.0 - float(eps)) / float(curvature) ** 0.5
    scale = torch.clamp(torch.as_tensor(
        max_norm, dtype=points.dtype, device=points.device) / norm, max=1.0)
    return points * scale


def poincare_expmap0(tangent: torch.Tensor, curvature: float, *,
                     clip_radius: float | None = None) -> torch.Tensor:
    """HHCH Eq. (3) at the origin, followed by numerical ball projection."""
    if curvature <= 0:
        raise ValueError('HHCH curvature must be positive')
    tangent = tangent.float() if not tangent.is_floating_point() else tangent
    norm = tangent.norm(dim=-1, keepdim=True)
    if clip_radius is not None:
        if clip_radius <= 0:
            raise ValueError('clip_radius must be positive when supplied')
        tangent = tangent * torch.clamp(
            torch.as_tensor(float(clip_radius), dtype=tangent.dtype,
                            device=tangent.device) / norm.clamp_min(1e-15),
            max=1.0,
        )
        norm = tangent.norm(dim=-1, keepdim=True)
    sqrt_c = float(curvature) ** 0.5
    safe_norm = norm.clamp_min(1e-15)
    mapped = (torch.tanh(sqrt_c * safe_norm)
              * tangent / (sqrt_c * safe_norm))
    return poincare_project(mapped, curvature)


def poincare_pairwise_distance(first: torch.Tensor, second: torch.Tensor,
                               curvature: float) -> torch.Tensor:
    """Pairwise Poincare distance equivalent to HHCH Eq. (2).

    The arcosh form is algebraically equivalent to the paper's Mobius-addition
    form but avoids constructing an ``[N, M, D]`` tensor.  This matters for
    epoch-wise clustering of all training examples.
    """
    if first.ndim != 2 or second.ndim != 2:
        raise ValueError('Poincare inputs must be rank-2 tensors')
    if first.shape[1] != second.shape[1]:
        raise ValueError('Poincare embedding dimensions differ')
    if curvature < 0:
        raise ValueError('curvature cannot be negative')
    if curvature == 0:
        # The limit stated immediately below HHCH Eq. (2).
        return 2.0 * torch.cdist(first, second, p=2)

    c = torch.as_tensor(curvature, dtype=first.dtype, device=first.device)
    first_sq = first.square().sum(dim=1, keepdim=True)
    second_sq = second.square().sum(dim=1, keepdim=True).T
    euclidean_sq = (first_sq + second_sq
                    - 2.0 * (first @ second.T)).clamp_min(0.0)
    denominator = ((1.0 - c * first_sq)
                   * (1.0 - c * second_sq)).clamp_min(1e-15)
    argument = 1.0 + 2.0 * c * euclidean_sq / denominator
    # ``acosh(1)`` is exact but has an infinite derivative.  A dtype-sized
    # floor keeps training finite; it changes only coincident-point distance.
    tiny = max(float(torch.finfo(first.dtype).eps), 1e-15)
    distance = torch.acosh(argument.clamp_min(1.0 + tiny)) / torch.sqrt(c)
    # Calls used for within-view contrast pass the very same tensor twice.
    # Restore the mathematically exact zero diagonal after the finite-gradient
    # safeguard above; those entries are excluded from the denominator.
    if (first.shape == second.shape
            and first.device == second.device
            and first.data_ptr() == second.data_ptr()):
        identity = torch.eye(first.shape[0], dtype=torch.bool,
                             device=first.device)
        distance = distance.masked_fill(identity, 0.0)
    return distance


def _poincare_aligned_distance(first: torch.Tensor, second: torch.Tensor,
                               curvature: float) -> torch.Tensor:
    """Eq. (2) for aligned rows, used by K-Means convergence checks."""
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError('aligned Poincare inputs must have the same [N,D] shape')
    if curvature <= 0:
        raise ValueError('curvature must be positive')
    c = torch.as_tensor(curvature, dtype=first.dtype, device=first.device)
    first_sq = first.square().sum(dim=1, keepdim=True)
    second_sq = second.square().sum(dim=1, keepdim=True)
    inner = (first * second).sum(dim=1, keepdim=True)
    numerator = ((1.0 - 2.0 * c * inner + c * second_sq) * (-first)
                 + (1.0 - c * first_sq) * second)
    denominator = (1.0 - 2.0 * c * inner
                   + c.square() * first_sq * second_sq).clamp_min(1e-15)
    mobius_norm = (numerator / denominator).norm(dim=1)
    argument = (torch.sqrt(c) * mobius_norm).clamp(
        min=0.0, max=1.0 - max(float(torch.finfo(first.dtype).eps), 1e-15))
    return 2.0 * torch.atanh(argument) / torch.sqrt(c)


def poincare_einstein_midpoint(points: torch.Tensor,
                               curvature: float) -> torch.Tensor:
    """Einstein midpoint from HHCH Eqs. (4)--(6)."""
    if points.ndim != 2 or points.shape[0] == 0:
        raise ValueError('Einstein midpoint needs a non-empty [N,D] tensor')
    if curvature <= 0:
        raise ValueError('HHCH curvature must be positive')
    c = torch.as_tensor(curvature, dtype=points.dtype, device=points.device)
    poincare_sq = points.square().sum(dim=1, keepdim=True)
    klein = 2.0 * points / (1.0 + c * poincare_sq)
    gamma = torch.rsqrt((1.0 - c * klein.square().sum(
        dim=1, keepdim=True)).clamp_min(1e-15))
    klein_midpoint = (gamma * klein).sum(dim=0) / gamma.sum()
    klein_sq = klein_midpoint.square().sum()
    midpoint = klein_midpoint / (
        1.0 + torch.sqrt((1.0 - c * klein_sq).clamp_min(1e-15)))
    return poincare_project(midpoint.unsqueeze(0), curvature).squeeze(0)


def _hhch_pairwise_distance_chunked(points: torch.Tensor,
                                    centers: torch.Tensor,
                                    curvature: float,
                                    chunk_size: int) -> torch.Tensor:
    chunks = []
    for start in range(0, points.shape[0], int(chunk_size)):
        chunks.append(poincare_pairwise_distance(
            points[start:start + int(chunk_size)], centers, curvature))
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def hyperbolic_kmeans_plusplus(points: torch.Tensor, n_clusters: int, *,
                              curvature: float, max_iter: int = 30,
                              tolerance: float = 1e-3, seed: int = 1,
                              chunk_size: int = 4096
                              ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Paper-faithful hyperbolic K-Means with K-Means++ initialization.

    Assignment uses HHCH Eq. (2), and each update uses the Einstein midpoint
    from Eqs. (4)--(6).  The paper does not prescribe empty-cluster handling;
    if one occurs, its center is deterministically reseeded with the currently
    farthest point so that the requested number of prototypes is preserved.
    """
    if points.ndim != 2 or points.shape[0] == 0:
        raise ValueError('points must be a non-empty [N,D] tensor')
    if not 1 <= int(n_clusters) <= points.shape[0]:
        raise ValueError('n_clusters must be in [1, number of points]')
    if max_iter <= 0 or chunk_size <= 0:
        raise ValueError('max_iter and chunk_size must be positive')
    if tolerance < 0:
        raise ValueError('tolerance cannot be negative')
    points = poincare_project(points.detach(), curvature)
    n_points = points.shape[0]
    generator = torch.Generator(device=points.device)
    generator.manual_seed(int(seed))

    # Standard K-Means++: probability proportional to squared distance to the
    # closest existing prototype, with HHCH's hyperbolic distance substituted.
    first_idx = int(torch.randint(
        n_points, (1,), generator=generator, device=points.device).item())
    selected = torch.zeros(n_points, dtype=torch.bool, device=points.device)
    selected[first_idx] = True
    center_list = [points[first_idx]]
    nearest_sq = poincare_pairwise_distance(
        points, points[first_idx:first_idx + 1], curvature
    ).squeeze(1).square()
    for _ in range(1, int(n_clusters)):
        probabilities = nearest_sq.clone()
        probabilities[selected] = 0.0
        mass = probabilities.sum()
        if not bool(torch.isfinite(mass)) or float(mass) <= 0.0:
            candidates = torch.nonzero(~selected, as_tuple=False).flatten()
            next_idx = int(candidates[0].item())
        else:
            next_idx = int(torch.multinomial(
                probabilities / mass, 1, generator=generator).item())
        selected[next_idx] = True
        center_list.append(points[next_idx])
        distance_sq = poincare_pairwise_distance(
            points, points[next_idx:next_idx + 1], curvature
        ).squeeze(1).square()
        nearest_sq = torch.minimum(nearest_sq, distance_sq)
    centers = torch.stack(center_list, dim=0)

    for _ in range(int(max_iter)):
        distances = _hhch_pairwise_distance_chunked(
            points, centers, curvature, chunk_size)
        assignments = distances.argmin(dim=1)

        # Vectorized Einstein means: Poincare -> Klein, gamma-weighted means,
        # then Klein -> Poincare.  This is exactly Eqs. (4)--(6).
        c = torch.as_tensor(curvature, dtype=points.dtype,
                            device=points.device)
        poincare_sq = points.square().sum(dim=1, keepdim=True)
        klein = 2.0 * points / (1.0 + c * poincare_sq)
        gamma = torch.rsqrt((1.0 - c * klein.square().sum(
            dim=1, keepdim=True)).clamp_min(1e-15))
        weighted_sum = torch.zeros_like(centers)
        weight_sum = torch.zeros(
            int(n_clusters), 1, dtype=points.dtype, device=points.device)
        weighted_sum.index_add_(0, assignments, gamma * klein)
        weight_sum.index_add_(0, assignments, gamma)
        nonempty = weight_sum.squeeze(1) > 0

        new_centers = centers.clone()
        klein_means = weighted_sum[nonempty] / weight_sum[nonempty]
        klein_norm_sq = klein_means.square().sum(dim=1, keepdim=True)
        new_centers[nonempty] = klein_means / (
            1.0 + torch.sqrt((1.0 - c * klein_norm_sq).clamp_min(1e-15)))

        empty = torch.nonzero(~nonempty, as_tuple=False).flatten()
        if empty.numel():
            farthest = torch.topk(
                distances.min(dim=1).values,
                k=int(empty.numel()), largest=True).indices
            new_centers[empty] = points[farthest]
        new_centers = poincare_project(new_centers, curvature)

        movement = _poincare_aligned_distance(
            centers, new_centers, curvature).max()
        centers = new_centers
        if float(movement) <= float(tolerance):
            break

    final_distance = _hhch_pairwise_distance_chunked(
        points, centers, curvature, chunk_size)
    return centers, final_distance.argmin(dim=1)


@torch.no_grad()
def hhch_hierarchical_kmeans(points: torch.Tensor,
                             cluster_counts: Tuple[int, ...] | list[int], *,
                             curvature: float, max_iter: int = 30,
                             tolerance: float = 1e-3, seed: int = 1,
                             chunk_size: int = 4096) -> Dict[str, object]:
    """Bottom-up HHCH hierarchy (Algorithm 2).

    Returns per-level prototypes and each original sample's ancestor index at
    every level.  Higher levels cluster the prototypes immediately below.
    """
    counts = tuple(int(value) for value in cluster_counts)
    if not counts or any(value <= 0 for value in counts):
        raise ValueError('cluster_counts must contain positive integers')
    data = points
    centers_by_level = []
    sample_ancestors = None
    for level, count in enumerate(counts):
        if count > data.shape[0]:
            raise ValueError(
                f'level {level} requests {count} clusters for {data.shape[0]} points')
        centers, assignment = hyperbolic_kmeans_plusplus(
            data, count, curvature=curvature, max_iter=max_iter,
            tolerance=tolerance, seed=int(seed) + level,
            chunk_size=chunk_size,
        )
        centers_by_level.append(centers)
        if sample_ancestors is None:
            sample_ancestors = assignment[:, None]
        else:
            next_ancestor = assignment[sample_ancestors[:, -1]]
            sample_ancestors = torch.cat(
                (sample_ancestors, next_ancestor[:, None]), dim=1)
        data = centers
    return {'centers': centers_by_level, 'assignments': sample_ancestors}


class HHCHFeatureHash(nn.Module):
    """HHCH paper head adapted to a frozen cached visual backbone.

    The hash path is ``D -> 512 -> K -> tanh`` and the training-only
    projection is ``K -> 128 -> Exp_0^c``.  Inference returns only the
    continuous hash code, as prescribed by HHCH.
    """

    def __init__(self, d_in: int, bit: int, *, hyper_dim: int = 128,
                 curvature: float = 0.01, dropout: float = 0.0,
                 clip_radius: float | None = 2.3):
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError('dropout must be in [0,1)')
        self.backbone = nn.Identity()
        layers = [nn.Linear(d_in, 512)]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.extend((nn.ReLU(), nn.Linear(512, bit), nn.Tanh()))
        # Name retained for DeepHashBase checkpoint compatibility.
        self.encoder_layers = nn.Sequential(*layers)
        self.projection_head = nn.Linear(bit, hyper_dim)
        self.curvature = float(curvature)
        self.clip_radius = clip_radius

    def encode_hash(self, features: torch.Tensor) -> torch.Tensor:
        return self.encoder_layers(features.float())

    def project_hash(self, codes: torch.Tensor) -> torch.Tensor:
        return poincare_expmap0(
            self.projection_head(codes), self.curvature,
            clip_radius=self.clip_radius)

    def project_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.project_hash(self.encode_hash(features))

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = features.float()
        codes = self.encode_hash(features)
        return {'continuous_code': codes, 'cnn_feat': features,
                'backbone_last_output': features}


def hhch_hierarchical_instance_loss(view_one: torch.Tensor,
                                    view_two: torch.Tensor,
                                    ancestors: torch.Tensor, *,
                                    temperature: float = 0.2,
                                    curvature: float = 0.01
                                    ) -> torch.Tensor:
    """HHCH Eqs. (7), (9)--(11), including both anchor directions."""
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    if view_one.shape != view_two.shape or view_one.ndim != 2:
        raise ValueError('HHCH views must have the same [B,D] shape')
    if ancestors.ndim != 2 or ancestors.shape[0] != view_one.shape[0]:
        raise ValueError('ancestors must have shape [B,L]')
    n_levels = ancestors.shape[1]
    if n_levels == 0:
        raise ValueError('HHCH needs at least one hierarchy level')

    score_12 = -poincare_pairwise_distance(
        view_one, view_two, curvature) / temperature
    score_11 = -poincare_pairwise_distance(
        view_one, view_one, curvature) / temperature
    score_21 = score_12.T
    score_22 = -poincare_pairwise_distance(
        view_two, view_two, curvature) / temperature
    batch_size = view_one.shape[0]
    diagonal = torch.eye(batch_size, dtype=torch.bool, device=view_one.device)
    total = view_one.new_zeros(())
    for level in range(n_levels):
        same_ancestor = ancestors[:, level, None].eq(
            ancestors[:, level][None, :])
        allow_cross = (~same_ancestor) | diagonal
        allow_same = (~same_ancestor) & (~diagonal)

        logits_one = torch.cat((score_12, score_11), dim=1)
        mask_one = torch.cat((allow_cross, allow_same), dim=1)
        denominator_one = torch.logsumexp(
            logits_one.masked_fill(~mask_one, -torch.inf), dim=1)
        loss_one = -(score_12.diagonal() - denominator_one).mean()

        logits_two = torch.cat((score_21, score_22), dim=1)
        mask_two = torch.cat((allow_cross.T, allow_same), dim=1)
        denominator_two = torch.logsumexp(
            logits_two.masked_fill(~mask_two, -torch.inf), dim=1)
        loss_two = -(score_21.diagonal() - denominator_two).mean()
        total = total + (loss_one + loss_two) / float(level + 1)
    return total / float(n_levels)


def hhch_hierarchical_prototype_loss(view_one: torch.Tensor,
                                     view_two: torch.Tensor,
                                     ancestors: torch.Tensor,
                                     centers_by_level: list[torch.Tensor], *,
                                     temperature: float = 0.2,
                                     curvature: float = 0.01
                                     ) -> torch.Tensor:
    """HHCH Eqs. (12)--(14), with all non-ancestor prototypes negative."""
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    if view_one.shape != view_two.shape or view_one.ndim != 2:
        raise ValueError('HHCH views must have the same [B,D] shape')
    if ancestors.ndim != 2 or ancestors.shape[0] != view_one.shape[0]:
        raise ValueError('ancestors must have shape [B,L]')
    if len(centers_by_level) != ancestors.shape[1] or not centers_by_level:
        raise ValueError('prototype levels and ancestor levels differ')
    total = view_one.new_zeros(())
    for level, centers in enumerate(centers_by_level):
        target = ancestors[:, level].long()
        logits_one = -poincare_pairwise_distance(
            view_one, centers, curvature) / temperature
        logits_two = -poincare_pairwise_distance(
            view_two, centers, curvature) / temperature
        level_loss = (F.cross_entropy(logits_one, target)
                      + F.cross_entropy(logits_two, target))
        total = total + level_loss / float(level + 1)
    return total / float(len(centers_by_level))


def hhch_logcosh_quantization_loss(view_one_codes: torch.Tensor,
                                   view_two_codes: torch.Tensor) -> torch.Tensor:
    """HHCH Eq. (15): ``1/2 sum_{i,k,v} log cosh(|h|-1)``."""
    if view_one_codes.shape != view_two_codes.shape:
        raise ValueError('HHCH code views must have identical shapes')

    def stable_log_cosh(value: torch.Tensor) -> torch.Tensor:
        absolute = value.abs()
        return absolute + F.softplus(-2.0 * absolute) - np.log(2.0)

    first = stable_log_cosh(view_one_codes.abs() - 1.0).sum()
    second = stable_log_cosh(view_two_codes.abs() - 1.0).sum()
    return 0.5 * (first + second)


def hhch_loss(view_one_codes: torch.Tensor, view_two_codes: torch.Tensor,
              view_one_hyperbolic: torch.Tensor,
              view_two_hyperbolic: torch.Tensor,
              ancestors: torch.Tensor,
              centers_by_level: list[torch.Tensor], *,
              temperature: float = 0.2, curvature: float = 0.01,
              quantization_weight: float = 0.01
              ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """HHCH Eq. (16): HIC + HPC + lambda times paper quantization."""
    instance = hhch_hierarchical_instance_loss(
        view_one_hyperbolic, view_two_hyperbolic, ancestors,
        temperature=temperature, curvature=curvature)
    prototype = hhch_hierarchical_prototype_loss(
        view_one_hyperbolic, view_two_hyperbolic, ancestors,
        centers_by_level, temperature=temperature, curvature=curvature)
    quantization = hhch_logcosh_quantization_loss(
        view_one_codes, view_two_codes)
    total = instance + prototype + float(quantization_weight) * quantization
    return total, {'hierarchical_instance': instance,
                   'hierarchical_prototype': prototype,
                   'quantization': quantization}
