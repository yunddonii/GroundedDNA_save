"""Native DNA image-retrieval baselines used by GroundedDNA.

This module is an independent PyTorch reimplementation of the computational
cores described by the direct predecessors below.  It intentionally does not
copy their TensorFlow source code or claim to reproduce wet-lab measurements.

* Stewart et al., DNA24 2018: continuous cosine/Hamming approximation followed by
  the fitted logistic approximation of hybridization yield.
* Bee et al., Nature Communications 2021 (PRIMO): 4096->2048->Lx4 encoder,
  entropy regularization, and a frozen differentiable local-match yield model.
* Koike et al., DATE/DAC 2024: masked-softmax DNA output and semi-hard triplet
  loss under normalized nucleotide Hamming distance.
* Koike et al., IEEE TCBB 2026: the triplet model plus the published source
  implementation's probability, homopolymer, and GC-balance penalties.

The paper protocols use VGG features, 30 or 80 nt, CIFAR/OpenImages, and in
some cases NUPACK or physical hybridization.  The GroundedDNA comparison uses
the *adapted* protocol: the same frozen feature cache, split, 18-nt capacity,
base-Hamming metric, and post-hoc biological projection as our method.  Callers
must keep those two result families labelled separately.

Base orders are easy to get subtly wrong.  The predecessor repositories use
``A,T,C,G`` internally.  GroundedDNA extraction files use canonical
``A,C,G,T`` indices so that ``00=A, 01=C, 10=G, 11=T`` and the shared
bio-constraint utilities remain valid.  Conversion happens only at extraction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


PAPER_BASES: Tuple[str, ...] = ("A", "T", "C", "G")
CANONICAL_BASES: Tuple[str, ...] = ("A", "C", "G", "T")
PAPER_TO_CANONICAL = np.asarray([0, 3, 1, 2], dtype=np.int64)

METHODS = ("bee2018", "bee2021", "koike2024", "koike2026")


def entropy_regularizer(probs: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Mean per-position Shannon entropy used by PRIMO and Koike encoders."""

    return -(probs * torch.log(probs.clamp_min(eps))).sum(dim=-1).mean()


def keras_activity_entropy_loss(*prob_batches: torch.Tensor) -> torch.Tensor:
    """Reproduce Keras ``activity_regularizer`` scaling in the upstream code.

    The predecessor regularizer itself returns a batch mean.  Keras then divides
    every activity-regularizer value by the input batch size.  PRIMO calls the
    shared encoder twice, so its two divided values are added; Koike calls it
    once.  Keeping this otherwise-surprising factor is necessary for a
    source-equivalent baseline.
    """

    if not prob_batches:
        raise ValueError("at least one probability batch is required")
    batch_size = int(prob_batches[0].shape[0])
    if batch_size < 1:
        raise ValueError("empty probability batch")
    if any(int(probs.shape[0]) != batch_size for probs in prob_batches):
        raise ValueError("all probability batches must have equal batch size")
    return sum(entropy_regularizer(probs) for probs in prob_batches) / float(batch_size)


def masked_softmax_code(probs: torch.Tensor) -> torch.Tensor:
    """Koike's differentiable maximum extraction.

    The official implementation does not emit a unit one-hot tensor during
    training.  It retains the winning softmax probability and masks the other
    three channels.  Gradients therefore reach the selected probability while
    the argmax selection itself remains discrete.
    """

    winner = probs.argmax(dim=-1)
    return probs * F.one_hot(winner, num_classes=4).to(probs.dtype)


def pairwise_normalized_dna_distance(probs: torch.Tensor) -> torch.Tensor:
    """Pairwise Koike distance matrix, ``L1(masked codes)/(2L)``.

    For exact one-hot sequences this is nucleotide Hamming distance divided by
    sequence length.  For soft outputs this follows DATE Eq. (2) and the
    symmetric TCBB successor implementation rather than silently replacing it
    by an expected mismatch probability.  The released DATE/DAC helper fills
    only its lower triangle and then accidentally adds its empty upper triangle;
    that apparent symmetrization bug is intentionally not reproduced here.
    """

    if probs.ndim != 3 or probs.shape[-1] != 4:
        raise ValueError(f"expected [B,L,4], got {tuple(probs.shape)}")
    masked = masked_softmax_code(probs)
    length = int(masked.shape[1])
    return (masked[:, None] - masked[None, :]).abs().sum(dim=(2, 3)) / (2.0 * length)


def _label_adjacency(labels: torch.Tensor) -> torch.Tensor:
    """Same-class/overlapping-label adjacency for single- or multi-label data."""

    if labels.ndim == 1:
        return labels[:, None].eq(labels[None, :])
    if labels.ndim != 2:
        raise ValueError(f"expected labels [B] or [B,C], got {tuple(labels.shape)}")
    return labels.float().matmul(labels.float().T).gt(0)


def semi_hard_triplet_loss(
    distance_matrix: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.8,
) -> torch.Tensor:
    """TensorFlow-Addons-style semi-hard triplet loss from a distance matrix.

    For every anchor-positive pair, choose the closest negative farther than
    the positive.  If none exists, choose the farthest negative.  Multi-label
    adaptation defines positives by any label overlap and negatives by none.
    Anchors without both a positive and a negative are skipped.
    """

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be square")
    n = int(distance_matrix.shape[0])
    adjacent = _label_adjacency(labels)
    eye = torch.eye(n, device=distance_matrix.device, dtype=torch.bool)
    positive = adjacent & ~eye
    negative = ~adjacent
    # [anchor, positive, negative-candidate].  This is algebraically identical
    # to the TensorFlow-Addons-style nested loop, but avoids one GPU
    # synchronization per anchor-positive pair.
    positive_distance = distance_matrix.unsqueeze(-1)
    negative_distance = distance_matrix.unsqueeze(1)
    outside_mask = (
        negative.unsqueeze(1)
        & (negative_distance > positive_distance)
    )
    closest_outside = negative_distance.masked_fill(
        ~outside_mask, torch.inf
    ).min(dim=-1).values
    has_outside = outside_mask.any(dim=-1)
    farthest_negative = distance_matrix.masked_fill(
        ~negative, -torch.inf
    ).max(dim=-1).values
    selected_negative = torch.where(
        has_outside,
        closest_outside,
        farthest_negative.unsqueeze(1),
    )
    valid = positive & negative.any(dim=-1, keepdim=True)
    if not valid.any():
        # Preserve a valid autograd path for pathological small/imbalanced batches.
        return distance_matrix.sum() * 0.0
    losses = F.relu(
        distance_matrix - selected_negative + float(margin)
    )
    return losses[valid].mean()


def koike_homopolymer_loss(probs: torch.Tensor, max_run: int = 4) -> torch.Tensor:
    """TCBB source-equivalent soft penalty for runs longer than ``max_run``.

    A frame contains ``max_run`` adjacent-position similarities, so it detects
    a potential run of ``max_run + 1`` bases.  ``sqrt(2)`` and the threshold of
    one follow the authors' public implementation.
    """

    if max_run < 1 or probs.shape[1] <= max_run:
        return probs.sum() * 0.0
    normalized = F.normalize(probs, p=2, dim=-1)
    adjacent = math.sqrt(2.0) * (normalized[:, 1:] * normalized[:, :-1]).sum(dim=-1)
    frames = adjacent.unfold(dimension=1, size=int(max_run), step=1)
    frame_min = frames.min(dim=-1).values
    frame_sum = frames.sum(dim=-1)
    masked = torch.where(frame_min >= 1.0, frame_sum, torch.zeros_like(frame_sum))
    return masked.mean(dim=-1).mean()


def koike_probability_penalty(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """TCBB source penalty ``mean[-log(max(p)-min(p))]``."""

    spread = probs.max(dim=-1).values - probs.min(dim=-1).values
    return -torch.log(spread + eps).mean()


def koike_gc_balance_loss(probs: torch.Tensor, target: float = 0.5) -> torch.Tensor:
    """TCBB source GC penalty under the predecessor ``A,T,C,G`` base order."""

    gc_fraction = (probs[..., 2] + probs[..., 3]).mean(dim=-1)
    return (gc_fraction - float(target)).abs().square().mean()


def bee2018_cosine_distance(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Mean per-position cosine distance used as soft Hamming in DNA24."""

    if first.shape != second.shape or first.ndim != 3 or first.shape[-1] != 4:
        raise ValueError(f"expected matching [B,L,4], got {first.shape} and {second.shape}")
    return (1.0 - F.cosine_similarity(first, second, dim=-1)).mean(dim=-1)


def bee2018_approximate_yield(
    distance: torch.Tensor,
    slope: float = 12.1,
    midpoint: float = 0.5,
) -> torch.Tensor:
    """DNA24's sigmoid fit from normalized Hamming distance to NUPACK yield."""

    return torch.sigmoid(-float(slope) * (distance - float(midpoint)))


class DenseDNAEncoder(nn.Module):
    """PRIMO/Koike dense encoder: ``D -> H -> 4L``, ReLU, reshape, softmax."""

    def __init__(self, input_dim: int, length: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden = int(hidden_dim if hidden_dim is not None else input_dim // 2)
        self.input_dim = int(input_dim)
        self.length = int(length)
        self.hidden_dim = hidden
        self.fc1 = nn.Linear(self.input_dim, hidden)
        self.fc2 = nn.Linear(hidden, self.length * 4)
        # Keras Dense defaults used by the released PRIMO/Koike code.
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.fc1(features.float()))
        pre_softmax = F.relu(self.fc2(hidden)).reshape(-1, self.length, 4)
        return F.softmax(pre_softmax, dim=-1)


class Bee2018Encoder(nn.Module):
    """DNA24 element-wise shared convolutions plus a dense DNA output.

    The original first reduces VGG-FC2 to ten PCA components.  ``pca_mean`` and
    ``pca_components`` make that preprocessing part of the checkpoint and avoid
    fitting it on query/database rows.
    """

    def __init__(
        self,
        input_dim: int,
        length: int = 30,
        pca_dim: int = 10,
        channels: int = 128,
        pca_mean: Optional[torch.Tensor] = None,
        pca_components: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.length = int(length)
        self.pca_dim = int(pca_dim)
        self.channels = int(channels)
        if pca_mean is None:
            pca_mean = torch.zeros(self.input_dim)
        if pca_components is None:
            pca_components = torch.eye(self.input_dim, self.pca_dim)
        if tuple(pca_mean.shape) != (self.input_dim,):
            raise ValueError(f"pca_mean shape {tuple(pca_mean.shape)}")
        if tuple(pca_components.shape) != (self.input_dim, self.pca_dim):
            raise ValueError(f"pca_components shape {tuple(pca_components.shape)}")
        self.register_buffer("pca_mean", pca_mean.float())
        self.register_buffer("pca_components", pca_components.float())
        # Conv1D(kernel=1) over independently treated feature dimensions is
        # exactly a shared scalar-to-channel affine transform.
        self.elementwise1 = nn.Linear(1, self.channels)
        self.elementwise2 = nn.Linear(self.channels, self.channels)
        self.output = nn.Linear(self.pca_dim * self.channels, self.length * 4)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        reduced = (features.float() - self.pca_mean).matmul(self.pca_components)
        hidden = torch.sin(self.elementwise1(reduced.unsqueeze(-1)))
        hidden = F.relu(self.elementwise2(hidden)).flatten(start_dim=1)
        pre_softmax = F.relu(self.output(hidden)).reshape(-1, self.length, 4)
        return F.softmax(pre_softmax, dim=-1)


@torch.no_grad()
def fit_pca(
    features: np.ndarray,
    n_components: int = 10,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fit train-only PCA and return ``(mean[D], components[D,K])``."""

    data = torch.as_tensor(np.asarray(features, dtype=np.float32))
    if data.ndim != 2:
        raise ValueError(f"expected [N,D], got {tuple(data.shape)}")
    k = min(int(n_components), int(data.shape[0]) - 1, int(data.shape[1]))
    if k < 1:
        raise ValueError("PCA needs at least two samples")
    mean = data.mean(dim=0)
    centered = data - mean
    # pca_lowrank uses randomized initialization; isolate and record the seed.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        _, _, right = torch.pca_lowrank(centered, q=k, center=False)
    return mean.cpu(), right[:, :k].cpu()


class PRIMOYieldPredictor(nn.Module):
    """PyTorch form of the official PRIMO local-match CNN.

    It accepts two *uncomplemented* soft sequence encodings ``[B,L,4]`` and
    returns logits.  The released predictor was calibrated with NUPACK for
    80-nt sequences at 21 C and 1 nM.  Reusing it at 18 nt is a length-transfer
    adaptation, not a physical reproduction.
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(36, 36, kernel_size=3)
        self.logit = nn.Linear(36, 1)

    @staticmethod
    def local_interactions(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        if first.shape != second.shape or first.ndim != 3 or first.shape[-1] != 4:
            raise ValueError("PRIMO predictor expects matching [B,L,4] tensors")
        if first.shape[1] < 11:
            raise ValueError("PRIMO local-match/pool/conv stack requires L >= 11")
        first_windows = first.unfold(dimension=1, size=3, step=1)   # [B,L-2,4,3]
        second_windows = second.unfold(dimension=1, size=3, step=1)
        outer = first_windows.unsqueeze(-1) * second_windows.unsqueeze(-2)
        return outer.flatten(start_dim=2)                           # [B,L-2,36]

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        interactions = self.local_interactions(first, second).transpose(1, 2)
        pooled = F.avg_pool1d(interactions, kernel_size=3, stride=3)
        convolved = torch.tanh(self.conv(pooled))
        return self.logit(convolved.mean(dim=-1)).squeeze(-1)

    def load_official_npz(self, path: str) -> None:
        """Load arrays written by ``scripts/convert_primo_predictor.py``."""

        arrays = np.load(path)
        required = {"conv_weight", "conv_bias", "dense_weight", "dense_bias"}
        missing = required.difference(arrays.files)
        if missing:
            raise KeyError(f"{path} misses {sorted(missing)}")
        state = {
            "conv.weight": torch.from_numpy(arrays["conv_weight"]).float(),
            "conv.bias": torch.from_numpy(arrays["conv_bias"]).float(),
            "logit.weight": torch.from_numpy(arrays["dense_weight"]).float(),
            "logit.bias": torch.from_numpy(arrays["dense_bias"]).float(),
        }
        self.load_state_dict(state, strict=True)


@dataclass
class LossBreakdown:
    total: torch.Tensor
    triplet: torch.Tensor
    entropy: torch.Tensor
    probability: torch.Tensor
    homopolymer: torch.Tensor
    gc: torch.Tensor

    def scalars(self) -> Dict[str, float]:
        return {
            key: float(getattr(self, key).detach().cpu())
            for key in ("total", "triplet", "entropy", "probability", "homopolymer", "gc")
        }


def koike_objective(
    probs: torch.Tensor,
    labels: torch.Tensor,
    method: str,
    margin: float = 0.8,
    entropy_strength: float = 1e-2,
    max_run: int = 4,
    hp_weight: float = 1.0,
    gc_weight: float = 1.0,
) -> LossBreakdown:
    """DATE/DAC or TCBB objective, including source-level regularizers."""

    if method not in ("koike2024", "koike2026"):
        raise ValueError(method)
    triplet = semi_hard_triplet_loss(
        pairwise_normalized_dna_distance(probs), labels, margin=margin
    )
    entropy = entropy_regularizer(probs)
    zero = probs.sum() * 0.0
    probability = zero
    homopolymer = zero
    gc = zero
    # The encoder's Keras activity_regularizer divides its already batch-mean
    # entropy by B.  See ``keras_activity_entropy_loss`` above.
    total = triplet + float(entropy_strength) * entropy / float(probs.shape[0])
    if method == "koike2026":
        probability = koike_probability_penalty(probs)
        homopolymer = koike_homopolymer_loss(probs, max_run=max_run)
        gc = koike_gc_balance_loss(probs)
        # Exact structure in the released implementation: probability
        # sharpening is grouped with the homopolymer branch.
        total = total + float(hp_weight) * (
            homopolymer + float(entropy_strength) * probability
        ) + float(gc_weight) * gc
    return LossBreakdown(total, triplet, entropy, probability, homopolymer, gc)


def paper_indices_to_canonical(indices: np.ndarray) -> np.ndarray:
    """Convert hard indices from ``A,T,C,G`` to GroundedDNA ``A,C,G,T``."""

    arr = np.asarray(indices)
    if arr.size and (arr.min() < 0 or arr.max() > 3):
        raise ValueError("base indices must be in {0,1,2,3}")
    return PAPER_TO_CANONICAL[arr].astype(np.int64, copy=False)


def canonical_indices_to_strings(indices: np.ndarray) -> np.ndarray:
    arr = np.asarray(indices, dtype=np.int64)
    alphabet = np.asarray(CANONICAL_BASES)
    return np.asarray(["".join(row) for row in alphabet[arr]], dtype=np.str_)


def canonical_indices_to_bits(indices: np.ndarray) -> np.ndarray:
    """Pack canonical base ids as two explicit bits per base."""

    arr = np.asarray(indices, dtype=np.uint8)
    return np.stack(((arr >> 1) & 1, arr & 1), axis=-1).reshape(len(arr), -1)


def canonical_indices_to_codons(indices: np.ndarray) -> np.ndarray:
    """Convert each consecutive 3-base group to a number in ``[0,63]``."""

    arr = np.asarray(indices, dtype=np.int64)
    if arr.shape[1] % 3:
        return np.empty((len(arr), 0), dtype=np.int64)
    triples = arr.reshape(len(arr), -1, 3)
    return triples[..., 0] * 16 + triples[..., 1] * 4 + triples[..., 2]


def koike_break_homopolymers(probs: np.ndarray, max_run: int = 4) -> np.ndarray:
    """Reimplement the TCBB repository's inference-time HP correction.

    The heuristic repeatedly finds hard runs longer than ``max_run`` and
    suppresses the current winning channel at low-confidence positions.  It
    guarantees neither minimum edit distance nor GC validity; the common DP
    projection must still follow it in the matched GroundedDNA protocol.

    Returns hard indices in the predecessor ``A,T,C,G`` order.
    """

    adjusted = np.asarray(probs, dtype=np.float32).copy()
    original = adjusted.copy()
    if adjusted.ndim != 3 or adjusted.shape[-1] != 4:
        raise ValueError(f"expected [N,L,4], got {adjusted.shape}")
    if int(max_run) < 1:
        raise ValueError("max_run must be positive")

    def bad_runs(sequence: np.ndarray):
        runs = []
        begin = 0
        while begin < len(sequence):
            end = begin + 1
            while end < len(sequence) and sequence[end] == sequence[begin]:
                end += 1
            if end - begin > int(max_run):
                runs.append((begin, end, int(sequence[begin])))
            begin = end
        return runs

    for sample in range(len(adjusted)):
        sequence = adjusted[sample].argmax(axis=-1)
        runs = bad_runs(sequence)
        guard = 0
        while runs:
            guard += 1
            if guard > adjusted.shape[1] * 4:
                raise RuntimeError("TCBB homopolymer heuristic did not converge")
            for begin, end, base in runs:
                run_probs = adjusted[sample, begin:end].max(axis=-1)
                # This stopping rule mirrors the authors' select_base routine.
                for local_index in np.argsort(run_probs):
                    position = begin + int(local_index)
                    adjusted[sample, position, base] = -original[sample, position, base]
                    if (end - begin) - int(local_index) - 1 <= int(max_run):
                        break
            sequence = adjusted[sample].argmax(axis=-1)
            runs = bad_runs(sequence)
    return adjusted.argmax(axis=-1).astype(np.int64)


@torch.no_grad()
def extract_native_codes(
    model: nn.Module,
    loader: Iterable[Dict[str, torch.Tensor]],
    device: str,
    koike_hp_postprocess: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return neural-raw/deployment bases, labels, and probabilities.

    Both base arrays use canonical ``A,C,G,T`` ids.  ``deployment_bases`` is
    identical to ``neural_raw_bases`` except when the TCBB homopolymer heuristic
    is requested.  Keeping both prevents a post-heuristic sequence from being
    mistaken for the direct argmax of ``soft_probabilities``.
    """

    model.eval()
    neural_raw_bases, deployment_bases, labels, probabilities = [], [], [], []
    for batch in loader:
        probs = model(batch["img"].to(device))
        probs_numpy = probs.cpu().numpy()
        raw_paper_ids = probs_numpy.argmax(axis=-1)
        if koike_hp_postprocess is None:
            deployment_paper_ids = raw_paper_ids
        else:
            deployment_paper_ids = koike_break_homopolymers(
                probs_numpy, max_run=int(koike_hp_postprocess)
            )
        neural_raw_bases.append(paper_indices_to_canonical(raw_paper_ids))
        deployment_bases.append(paper_indices_to_canonical(deployment_paper_ids))
        labels.append(batch["label"].cpu().numpy().astype(np.int64))
        probabilities.append(probs_numpy.astype(np.float16))
    return (
        np.concatenate(neural_raw_bases, axis=0),
        np.concatenate(deployment_bases, axis=0),
        np.concatenate(labels, axis=0),
        np.concatenate(probabilities, axis=0),
    )


def calibrate_feature_threshold(
    features: np.ndarray,
    quantile: float = 0.1,
    n_pairs: int = 100_000,
    seed: int = 42,
) -> float:
    """Train-only Euclidean threshold for a portable Bee-style adaptation."""

    if not 0.0 < float(quantile) < 1.0:
        raise ValueError("quantile must be in (0,1)")
    feats = np.asarray(features)
    n = int(len(feats))
    if n < 2:
        raise ValueError("at least two features are required")
    rng = np.random.RandomState(int(seed))
    first = rng.randint(0, n, size=int(n_pairs))
    second = rng.randint(0, n - 1, size=int(n_pairs))
    second += second >= first
    delta = feats[first].astype(np.float32) - feats[second].astype(np.float32)
    distances = np.sqrt(np.square(delta).sum(axis=1))
    return float(np.quantile(distances, float(quantile)))


def sample_balanced_feature_pairs(
    features: np.ndarray,
    threshold: float,
    batch_size: int,
    rng: np.random.RandomState,
    max_rounds: int = 100,
    distance_features: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce PRIMO's source-level balanced-pair generator.

    Each source chunk contributes all similar pairs and at most the same number
    of dissimilar pairs.  Chunks are pooled until their balanced *count* reaches
    ``batch_size``; a random truncation of the pooled rows then forms the
    returned batch.  Consequently, overshoot means an individual final batch is
    only statistically balanced, not forced to contain exactly 50/50 labels.
    """

    feats = np.asarray(features)
    metric_feats = feats if distance_features is None else np.asarray(distance_features)
    if len(metric_feats) != len(feats):
        raise ValueError("features and distance_features must have equal length")
    n = int(len(feats))
    requested = int(batch_size)
    if requested < 1:
        raise ValueError("batch_size must be positive")
    pooled_pairs = []
    balanced_count = 0
    for _ in range(int(max_rounds)):
        pairs = _sample_disjoint_pairs(n, requested, rng)
        first, second = pairs[:, 0], pairs[:, 1]
        delta = (metric_feats[first].astype(np.float32)
                 - metric_feats[second].astype(np.float32))
        distance = np.sqrt(np.square(delta).sum(axis=1))
        similar = distance <= float(threshold)
        n_similar = int(similar.sum())
        pooled_pairs.extend((pairs[similar], pairs[~similar][:n_similar]))
        # This intentionally mirrors PRIMO's source counter.  Under the
        # intended low positive-rate threshold there are at least n_similar
        # negatives in each chunk.
        balanced_count += 2 * n_similar
        pooled_size = sum(len(chunk) for chunk in pooled_pairs)
        if balanced_count >= requested and pooled_size >= requested:
            break
    if not pooled_pairs or sum(len(chunk) for chunk in pooled_pairs) < requested:
        raise RuntimeError(
            "could not sample a balanced Bee batch; adjust the train-only "
            "feature threshold/quantile"
        )
    pool = np.concatenate(pooled_pairs, axis=0)
    pairs = pool[rng.permutation(len(pool))[:requested]]
    first, second = pairs[:, 0], pairs[:, 1]
    metric_delta = (
        metric_feats[first].astype(np.float32)
        - metric_feats[second].astype(np.float32)
    )
    targets = (
        np.sqrt(np.square(metric_delta).sum(axis=1)) <= float(threshold)
    ).astype(np.float32)
    return pairs, targets, np.asarray(feats[pairs], dtype=np.float32)


def sample_random_feature_pairs(
    features: np.ndarray,
    threshold: float,
    batch_size: int,
    rng: np.random.RandomState,
    distance_features: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """DNA24-style unconditioned random pairs and threshold-derived labels."""

    feats = np.asarray(features)
    metric_feats = feats if distance_features is None else np.asarray(distance_features)
    if len(metric_feats) != len(feats):
        raise ValueError("features and distance_features must have equal length")
    n = int(len(feats))
    pairs = _sample_disjoint_pairs(n, int(batch_size), rng)
    pair_features = np.asarray(feats[pairs], dtype=np.float32)
    metric_pairs = np.asarray(metric_feats[pairs], dtype=np.float32)
    delta = metric_pairs[:, 0] - metric_pairs[:, 1]
    distance = np.sqrt(np.square(delta).sum(axis=1))
    targets = (distance <= float(threshold)).astype(np.float32)
    return pairs, targets, pair_features


def _sample_disjoint_pairs(
    n_items: int,
    n_pairs: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """PRIMO-style pairs without self-pairs or within-chunk row reuse.

    The official static/OpenImages generators permute ``n`` rows and reshape
    the first ``2 * batch`` rows.  If a caller requests more pairs than that
    permits, independent permutation chunks are concatenated; reuse can then
    occur across chunks but never inside one chunk.
    """

    n = int(n_items)
    remaining = int(n_pairs)
    if n < 2:
        raise ValueError("at least two items are required")
    chunks = []
    per_chunk = n // 2
    while remaining > 0:
        take = min(remaining, per_chunk)
        chunks.append(rng.permutation(n)[: 2 * take].reshape(-1, 2))
        remaining -= take
    return np.concatenate(chunks, axis=0).astype(np.int64, copy=False)


def project_codes_memoized(
    codes: np.ndarray,
    gc_min: float = 0.4,
    gc_max: float = 0.6,
    max_run: int = 3,
) -> Dict[str, object]:
    """Hamming-minimum DP projection, evaluating every unique sequence once."""

    from dna_utils.bio_constraints import is_valid_batch, project_to_valid

    arr = np.asarray(codes, dtype=np.int8)
    unique, inverse = np.unique(arr, axis=0, return_inverse=True)
    before_unique = is_valid_batch(unique, gc_min, gc_max, max_run)
    projected_unique = unique.copy()
    edits_unique = np.zeros(len(unique), dtype=np.int32)
    for index in range(len(unique)):
        if before_unique[index]:
            continue
        projected, edits = project_to_valid(unique[index], gc_min, gc_max, max_run)
        projected_unique[index] = projected
        edits_unique[index] = edits
    projected = projected_unique[inverse]
    edits = edits_unique[inverse]
    valid_before = is_valid_batch(arr, gc_min, gc_max, max_run)
    valid_after = is_valid_batch(projected, gc_min, gc_max, max_run)
    successful = edits[edits >= 0]
    return {
        "projected_codes": projected.astype(np.int64),
        "edit_distances": edits,
        "pre_compliance": float(valid_before.mean()) if len(arr) else 0.0,
        "post_compliance": float(valid_after.mean()) if len(arr) else 0.0,
        "mean_edit_distance": float(successful.mean()) if len(successful) else 0.0,
        "n_unique": int(len(unique)),
    }


def evaluate_base_retrieval(
    query_codes: np.ndarray,
    database_codes: np.ndarray,
    query_labels: np.ndarray,
    database_labels: np.ndarray,
    map_at_r: int,
    device: str = "cpu",
    query_chunk: int = 64,
    precision_at: Sequence[int] = (1, 10, 100),
) -> Dict[str, object]:
    """CalcTopMap-style image retrieval under nucleotide Hamming distance."""

    from baseline.base_model import _ap_at_r, _multi_hot_relevance

    q = np.asarray(query_codes, dtype=np.int8)
    d = np.asarray(database_codes, dtype=np.int8)
    if q.ndim != 2 or d.ndim != 2 or q.shape[1] != d.shape[1]:
        raise ValueError(f"query/database code shapes do not match: {q.shape}, {d.shape}")
    use_device = torch.device(device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu")
    db_tensor = torch.from_numpy(d).to(use_device)
    aps = []
    precision = {int(k): [] for k in precision_at}
    for start in range(0, len(q), int(query_chunk)):
        q_tensor = torch.from_numpy(q[start:start + int(query_chunk)]).to(use_device)
        distances = (q_tensor[:, None, :] != db_tensor[None, :, :]).sum(dim=-1).cpu().numpy()
        relevance = _multi_hot_relevance(
            np.asarray(query_labels)[start:start + int(query_chunk)], np.asarray(database_labels)
        )
        for row in range(len(distances)):
            order = np.argsort(distances[row], kind="stable")
            sorted_relevance = relevance[row, order]
            aps.append(_ap_at_r(sorted_relevance, R=int(map_at_r)))
            for k in precision:
                top = sorted_relevance[:min(k, len(sorted_relevance))]
                precision[k].append(float(top.mean()) if len(top) else 0.0)
    return {
        "mAP_at_R": float(np.mean(aps)) if aps else 0.0,
        "mAP_R_cutoff": int(map_at_r),
        "precision_at_k": {k: float(np.mean(values)) if values else 0.0
                           for k, values in precision.items()},
    }


def extraction_payload(
    base_indices: np.ndarray,
    labels: np.ndarray,
    image_paths: Sequence[str],
    soft_probs: Optional[np.ndarray] = None,
    neural_raw_base_indices: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Build the repository's common extraction ``npz`` schema."""

    bases = np.asarray(base_indices, dtype=np.int64)
    payload = {
        "base_indices": bases,
        "hash_2bit": canonical_indices_to_bits(bases).astype(np.uint8),
        "codebook_indices": canonical_indices_to_codons(bases),
        "multi_hot_labels": np.asarray(labels, dtype=np.int64),
        "image_paths": np.asarray(image_paths, dtype=np.str_),
        "dna_strings": canonical_indices_to_strings(bases),
    }
    if soft_probs is not None:
        payload["soft_probs_atcg"] = np.asarray(soft_probs, dtype=np.float16)
    if neural_raw_base_indices is not None:
        neural_raw = np.asarray(neural_raw_base_indices, dtype=np.int64)
        if neural_raw.shape != bases.shape:
            raise ValueError("neural_raw_base_indices must match base_indices shape")
        payload["neural_raw_base_indices"] = neural_raw
    return payload
