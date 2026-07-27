"""Supervised Center-Reassigned Hashing (CRH; AAAI 2026).

Clean-room implementation of:

    Yin et al., "Codebook-Centric Deep Hashing: End-to-End Joint
    Learning of Semantic Hash Centers and Neural Hash Function", AAAI 2026.

CRH is a *supervised* pointwise hashing method.  Training labels enter both
the margin cross-entropy objective (paper Eqs. 1--2) and the dynamic
class-to-codebook reassignment (Eqs. 5--8).  It must therefore be reported in
the supervised ``S`` panel, never as a U0/U1/U2 baseline.

The official implementation has no license file at the audited commit, so no
upstream source is vendored or copied.  This module derives the mathematical
core from the published equations and records the adaptation boundary:

* Original input/hash function: augmented pixels, ResNet-34, linear hash head.
* Matched comparison: the repository's immutable frozen visual features and
  one linear hash head.
* Preserved CRH core: ``M=2C`` codebook, adaptive multi-head construction,
  margin cosine CE, tanh L2 quantization, weighted multi-label reassignment,
  random-class-order greedy one-to-one assignment, update cadence, optimizer,
  scheduler, and source training horizons.

Official paper: https://ojs.aaai.org/index.php/AAAI/article/view/38190
Official code:  https://github.com/iFamilyi/CRH
Audited commit: bc3efd3757501f11d4f308be81167be7e2bd5342
"""

from __future__ import annotations

import json
import math
import os
from argparse import ArgumentParser
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, Dataset, Subset

from .base_model import DeepHashBase, NUM_CLASS


OFFICIAL_COMMIT = "bc3efd3757501f11d4f308be81167be7e2bd5342"
OFFICIAL_PAPER_URL = (
    "https://ojs.aaai.org/index.php/AAAI/article/view/38190"
)
OFFICIAL_CODE_URL = "https://github.com/iFamilyi/CRH"


def adaptive_head_dim(code_length: int, codebook_size: int) -> int:
    """Smallest divisor ``d`` of ``K`` for which ``2**d >= M``.

    Each of the ``H=K/d`` sub-codebooks must contain ``M`` mutually distinct
    ``d``-bit vectors.  This is the executable form of the paper's
    ``H <= K / log2(M)`` collision-avoidance constraint.
    """

    code_length = int(code_length)
    codebook_size = int(codebook_size)
    if code_length <= 0:
        raise ValueError("CRH code length must be positive")
    if codebook_size <= 0:
        raise ValueError("CRH codebook size must be positive")
    for dimension in range(1, code_length + 1):
        if code_length % dimension == 0 and (1 << dimension) >= codebook_size:
            return dimension
    # ``d=K`` always works for the benchmark settings.  Keep the validation
    # explicit for callers that request more than the full K-bit space.
    raise ValueError(
        f"cannot place {codebook_size} unique vectors in any head of a "
        f"{code_length}-bit code"
    )


def _integers_to_signed_bits(values: torch.Tensor, dimension: int) -> torch.Tensor:
    """Convert non-negative integers to MSB-first vectors in ``{-1,+1}``."""

    shifts = torch.arange(dimension - 1, -1, -1, dtype=torch.long)
    bits = ((values.to(torch.long).unsqueeze(1) >> shifts) & 1).float()
    return bits.mul(2.0).sub(1.0)


def generate_multi_head_codebook(
    code_length: int,
    codebook_size: int,
    *,
    seed: int,
) -> torch.Tensor:
    """Uniformly sample a collision-free codebook for every CRH head.

    Returns:
        Tensor of shape ``[H, M, d]`` in ``{-1,+1}``.

    The paper prescribes uniform sampling of ``M`` unique subcodes when
    ``d <= 16``.  All registered 36/48-bit benchmark configurations fall in
    that tractable regime after adaptive head selection.
    """

    dimension = adaptive_head_dim(code_length, codebook_size)
    if dimension > 16:
        raise ValueError(
            "the audited exact unique-space sampler supports head dimensions "
            f"up to 16, got {dimension}"
        )
    num_heads = int(code_length) // dimension
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    space_size = 1 << dimension
    heads = []
    for _ in range(num_heads):
        selected = torch.randperm(space_size, generator=generator)[
            :int(codebook_size)
        ]
        heads.append(_integers_to_signed_bits(selected, dimension))
    return torch.stack(heads, dim=0)


def compose_class_centers(
    codebook: torch.Tensor, assignments: torch.Tensor
) -> torch.Tensor:
    """Concatenate independently selected sub-centers (paper Eq. 8).

    Args:
        codebook: ``[H, M, d]``.
        assignments: codebook indices ``[H, C]``; indices must be distinct
            across classes within each head.
    Returns:
        Full class-center matrix ``[C, H*d]``.
    """

    if codebook.ndim != 3 or assignments.ndim != 2:
        raise ValueError("CRH codebook/assignments must have ranks 3 and 2")
    heads, codebook_size, dimension = codebook.shape
    if assignments.shape[0] != heads:
        raise ValueError("CRH assignment head count does not match codebook")
    if assignments.numel() and (
        int(assignments.min()) < 0 or int(assignments.max()) >= codebook_size
    ):
        raise ValueError("CRH assignment index is outside the codebook")
    gathered = torch.gather(
        codebook,
        dim=1,
        index=assignments.to(torch.long).unsqueeze(-1).expand(
            -1, -1, dimension
        ),
    )
    return gathered.transpose(0, 1).reshape(assignments.shape[1], -1)


def crh_margin_cross_entropy(
    logits: torch.Tensor,
    class_centers: torch.Tensor,
    labels: torch.Tensor,
    *,
    scale: float,
    margin: float,
) -> torch.Tensor:
    """CRH margin cosine cross entropy (paper Eqs. 1--2).

    Multi-hot targets are normalized by their per-sample L1 norm exactly as in
    Eq. (1).  A one-hot target therefore reduces to ordinary cross entropy.
    """

    if logits.ndim != 2 or class_centers.ndim != 2 or labels.ndim != 2:
        raise ValueError("CRH loss inputs must all be rank-2 tensors")
    if logits.shape[0] != labels.shape[0]:
        raise ValueError("CRH logit/label batch sizes differ")
    if logits.shape[1] != class_centers.shape[1]:
        raise ValueError("CRH code and class-center widths differ")
    if labels.shape[1] != class_centers.shape[0]:
        raise ValueError("CRH label and class-center counts differ")
    if not math.isfinite(float(scale)) or float(scale) <= 0.0:
        raise ValueError("CRH scale must be finite and positive")
    if not math.isfinite(float(margin)) or float(margin) < 0.0:
        raise ValueError("CRH margin must be finite and non-negative")

    targets = labels.float()
    cardinality = targets.sum(dim=1, keepdim=True)
    if torch.any(cardinality <= 0):
        raise ValueError("every CRH training sample must have a positive label")
    targets = targets / cardinality
    cosine = F.normalize(logits, dim=1) @ F.normalize(
        class_centers.float(), dim=1
    ).transpose(0, 1)
    margin_logits = float(scale) * (
        cosine - float(margin) * labels.float()
    )
    return -(targets * F.log_softmax(margin_logits, dim=1)).sum(dim=1).mean()


def crh_tanh_quantization(logits: torch.Tensor) -> torch.Tensor:
    """Tanh-relaxation L2 quantization term (paper Eq. 3)."""

    if logits.ndim != 2:
        raise ValueError("CRH quantization expects [batch, bit] logits")
    relaxed = torch.tanh(logits)
    return (relaxed.abs() - 1.0).square().mean()


def crh_objective(
    logits: torch.Tensor,
    class_centers: torch.Tensor,
    labels: torch.Tensor,
    *,
    scale: float,
    margin: float = 0.2,
    quantization_weight: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Complete CRH objective (paper Eq. 4)."""

    quantization_weight = float(quantization_weight)
    if not math.isfinite(quantization_weight) or quantization_weight < 0.0:
        raise ValueError(
            "CRH quantization weight must be finite and non-negative"
        )
    classification = crh_margin_cross_entropy(
        logits,
        class_centers,
        labels,
        scale=float(scale),
        margin=float(margin),
    )
    quantization = crh_tanh_quantization(logits)
    total = classification + quantization_weight * quantization
    return total, {
        "margin_cross_entropy": classification,
        "quantization": quantization,
    }


def crh_weighted_reassignment_cost(
    signed_codes: torch.Tensor,
    labels: torch.Tensor,
    codebook: torch.Tensor,
) -> torch.Tensor:
    """Compute every per-head class/code candidate cost from paper Eq. (7).

    A sample with ``q=||y||_1`` positive labels contributes weight ``1/q`` to
    each of those classes.  The result has shape ``[H,C,M]`` and contains the
    weighted mean squared Euclidean distance between binary subcodes and
    candidate sub-centers.  For one-hot labels this reduces exactly to Eq. (5).
    """

    if signed_codes.ndim != 2 or labels.ndim != 2 or codebook.ndim != 3:
        raise ValueError("CRH reassignment inputs have invalid ranks")
    heads, codebook_size, dimension = codebook.shape
    if signed_codes.shape[0] != labels.shape[0]:
        raise ValueError("CRH code/label sample counts differ")
    if signed_codes.shape[1] != heads * dimension:
        raise ValueError("CRH signed-code width does not match the codebook")
    if signed_codes.shape[0] == 0:
        raise ValueError("CRH reassignment requires at least one sample")
    label_count = labels.float().sum(dim=1, keepdim=True)
    if torch.any(label_count <= 0):
        raise ValueError("every CRH reassignment sample needs a label")

    # The paper binarizes h before measuring squared Euclidean distance.
    # Tie exact zeros to +1 so every component lies in {-1,+1}.
    binary = torch.where(
        signed_codes >= 0,
        torch.ones_like(signed_codes),
        -torch.ones_like(signed_codes),
    ).float()
    weighted_labels = labels.float() / label_count
    class_weights = weighted_labels.sum(dim=0)
    if torch.any(class_weights <= 0):
        missing = torch.where(class_weights <= 0)[0].tolist()
        raise ValueError(
            f"CRH cannot reassign classes absent from optimization-train: "
            f"{missing}"
        )

    # Eq. (7) can be evaluated without materializing [N,H,C,M,d]:
    # E_w ||b-z||^2 = E_w ||b||^2 + ||z||^2 - 2 E_w[b]^T z.
    subcodes = binary.reshape(binary.shape[0], heads, dimension)
    weighted_sum = torch.einsum(
        "nc,nhd->hcd", weighted_labels, subcodes
    )
    mean_subcode = weighted_sum / class_weights.view(1, -1, 1)
    mean_code_norm = torch.einsum(
        "nc,nhd->hc", weighted_labels, subcodes.square()
    ) / class_weights.view(1, -1)
    candidate_norm = codebook.float().square().sum(dim=2)
    dot = torch.einsum("hcd,hmd->hcm", mean_subcode, codebook.float())
    return (
        mean_code_norm.unsqueeze(-1)
        + candidate_norm.unsqueeze(1)
        - 2.0 * dot
    )


def greedy_distinct_assignments(
    costs: torch.Tensor,
    *,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Random-class-order greedy assignment from paper Eq. (6).

    Each head is optimized independently.  Within a head, each class chooses
    its lowest-cost candidate not already used by an earlier class in that
    head's seeded random order.
    """

    if costs.ndim != 3:
        raise ValueError("CRH costs must have shape [heads, classes, centers]")
    heads, num_classes, codebook_size = costs.shape
    if num_classes > codebook_size:
        raise ValueError("CRH one-to-one assignment requires M >= C")
    if not torch.isfinite(costs).all():
        raise ValueError("CRH assignment costs must be finite")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    assignments = torch.empty(
        (heads, num_classes), dtype=torch.long, device=costs.device
    )
    totals = torch.zeros(heads, dtype=costs.dtype, device=costs.device)
    for head in range(heads):
        class_order = torch.randperm(
            num_classes, generator=generator
        ).tolist()
        available = torch.ones(
            codebook_size, dtype=torch.bool, device=costs.device
        )
        for class_index in class_order:
            masked = costs[head, class_index].masked_fill(
                ~available, torch.inf
            )
            center_index = int(torch.argmin(masked).item())
            assignments[head, class_index] = center_index
            totals[head] += costs[head, class_index, center_index]
            available[center_index] = False
    return assignments, totals


def crh_should_reassign(epoch_zero_based: int) -> bool:
    """Source cadence: epochs 1--20, then every fifth epoch."""

    epoch_zero_based = int(epoch_zero_based)
    if epoch_zero_based < 0:
        raise ValueError("CRH epoch must be non-negative")
    return epoch_zero_based < 20 or (epoch_zero_based + 1) % 5 == 0


def prepare_crh_supervised_training_dataset(
    trainset: Dataset,
    *,
    protocol_stage: str,
) -> Tuple[Dataset, dict]:
    """Restrict only CRH's optimization loader to positive-label rows.

    Paper Eqs. (1), (2), and (7) divide by ``||y||_1`` and are undefined for
    an all-zero target.  Some matched repository splits contain such rows.
    They are outside CRH's supervised optimization domain and are therefore
    omitted from the current stage's *training loader only*.

    The input dataset is never mutated.  In particular, held-out validation,
    test, and database datasets remain untouched, and stage-1 validation still
    ranks against the complete, already-fixed optimization-train database.
    """

    if not hasattr(trainset, "labels"):
        raise TypeError(
            "CRH training dataset must expose its stage-specific label matrix"
        )
    labels = np.asarray(getattr(trainset, "labels"))
    if labels.ndim != 2 or labels.shape[0] != len(trainset):
        raise ValueError(
            "CRH training labels must have shape [len(trainset), classes]"
        )
    if labels.shape[1] == 0:
        raise ValueError("CRH training labels contain no classes")
    if not np.isfinite(labels).all() or np.any(labels < 0):
        raise ValueError("CRH training labels must be finite and non-negative")

    cardinality = labels.sum(axis=1)
    retained_indices = np.flatnonzero(cardinality > 0)
    dropped = int(labels.shape[0] - retained_indices.size)
    if retained_indices.size == 0:
        raise ValueError(
            "CRH has no positive-label rows in the current training partition"
        )
    positive_counts = labels[retained_indices].sum(axis=0)
    missing_classes = np.flatnonzero(positive_counts <= 0)
    if missing_classes.size:
        raise ValueError(
            "CRH zero-label filtering leaves classes absent from the current "
            f"training partition: {missing_classes.tolist()}"
        )

    # Always use a Subset, even when no row is dropped.  This makes the
    # training-only boundary explicit and prevents future code from mutating
    # the protocol-owned dataset through a method-specific helper.
    filtered: Dataset = Subset(
        trainset, retained_indices.astype(np.int64).tolist()
    )
    audit = {
        "crh_training_partition_stage": str(protocol_stage),
        "crh_training_rows_total": int(labels.shape[0]),
        "crh_training_rows_retained": int(retained_indices.size),
        "crh_training_zero_label_rows_dropped": dropped,
        "crh_training_all_classes_present": True,
        "crh_training_class_positive_counts": [
            float(value) for value in positive_counts.tolist()
        ],
        "crh_zero_label_policy": (
            "drop_from_current_stage_training_loader_only"
        ),
        "crh_training_filter_scope": (
            "training-loader-only; validation/test/database datasets unchanged"
        ),
    }
    return filtered, audit


class CRHFeatureHash(nn.Module):
    """One-linear-layer CRH hash function over frozen cached features."""

    def __init__(
        self,
        d_in: int,
        bit: int,
        num_classes: int,
        *,
        seed: int,
        codebook_scale: int = 2,
    ) -> None:
        super().__init__()
        if int(d_in) <= 0 or int(bit) <= 0 or int(num_classes) <= 1:
            raise ValueError("CRH model dimensions are invalid")
        if int(codebook_scale) != 2:
            raise ValueError("the registered CRH model requires M=2C")
        self.bit = int(bit)
        self.num_classes = int(num_classes)
        self.codebook_size = int(codebook_scale) * self.num_classes
        if self.codebook_size < self.num_classes:
            raise ValueError("CRH requires M >= C")

        self.encoder_layers = nn.Linear(int(d_in), self.bit)
        codebook = generate_multi_head_codebook(
            self.bit, self.codebook_size, seed=int(seed)
        )
        initial = torch.arange(self.num_classes, dtype=torch.long).expand(
            codebook.shape[0], -1
        ).clone()
        centers = compose_class_centers(codebook, initial)
        # These are non-parametric but method-defining state.  Registering all
        # three makes extraction from a selected checkpoint auditable.
        self.register_buffer("codebook", codebook)
        self.register_buffer("assignments", initial)
        self.register_buffer("current_class_centers", centers)

    @property
    def head_dim(self) -> int:
        return int(self.codebook.shape[2])

    @property
    def num_heads(self) -> int:
        return int(self.codebook.shape[0])

    @torch.no_grad()
    def install_assignments(self, assignments: torch.Tensor) -> None:
        if assignments.shape != self.assignments.shape:
            raise ValueError("new CRH assignments have the wrong shape")
        assignments = assignments.to(
            device=self.assignments.device, dtype=torch.long
        )
        for head in range(self.num_heads):
            if torch.unique(assignments[head]).numel() != self.num_classes:
                raise ValueError(
                    "CRH assignments must be distinct within every head"
                )
        centers = compose_class_centers(self.codebook, assignments)
        self.assignments.copy_(assignments)
        self.current_class_centers.copy_(centers)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = features.float()
        logits = self.encoder_layers(features)
        return {
            # Retrieval sign(tanh(v)) == sign(v), so the common extractor can
            # consume the pre-tanh logits without changing a single bit.
            "continuous_code": logits,
            "crh_logits": logits,
            "crh_relaxed_code": torch.tanh(logits),
            "cnn_feat": features,
            "backbone_last_output": features,
        }


class CRH(DeepHashBase):
    """Supervised CRH under the repository's leakage-safe P0 protocol."""

    def _get_default_config_dict(self) -> dict:
        return {
            "batch_size": 128,
            # Resolved to 30 (multi-label) or 300 (single-label) once the
            # dataset is known.  The common driver can still pass it
            # explicitly and binds that value into its protocol digest.
            "max_epoch": None,
            "eval_period": 5,
            "learning_rate": 1e-4,
            "optimizer_name": "adam",
            "adam_weight_decay": 1e-5,
            "lr_scheduler": "coslr",
            "crh_margin": 0.2,
            "crh_codebook_scale": 2,
            "crh_update_rounds": 20,
            "crh_update_interval": 5,
            "crh_gradient_clip_norm": 1.0,
        }

    def _get_config_dict_for_dataset(
        self, default_config: dict, dataset: str
    ) -> dict:
        config = dict(default_config)
        config["max_epoch"] = 30 if dataset in {
            "Flickr25k", "MSCOCO", "NUSWIDE"
        } else 300
        return config

    def _get_fixed_config_dict(self) -> dict:
        return {
            "gen_code_method": "sign",
            "finetune": False,
            "dataset_return_index": False,
            "dataset_return_paired_aug_img": False,
            "dataset_return_visual_tokens": False,
            "transform": "none",
            "encoder_layers": "crh_single_linear",
            "batch_norm": False,
            "supervision_regime": "supervised",
            "uses_train_labels_in_objective": True,
            "uses_train_labels_in_center_reassignment": True,
            "information_condition": (
                "S: target training labels and benchmark class taxonomy"
            ),
            "information_tier": "S",
            "implementation_variant": (
                "clean-room-paper-equations-matched-cache-adapter"
            ),
            "baseline_display_name": "CRH (supervised)",
            "source_paper": (
                "Yin et al., Codebook-Centric Deep Hashing: End-to-End "
                "Joint Learning of Semantic Hash Centers and Neural Hash "
                "Function, AAAI 2026"
            ),
            "source_paper_url": OFFICIAL_PAPER_URL,
            "source_code_url": OFFICIAL_CODE_URL,
            "source_code_commit": OFFICIAL_COMMIT,
            "upstream_license_at_audit": "none-detected",
            "backbone_training": "frozen-precomputed-matched-cache",
            "official_backbone": "ResNet-34 pixel input",
            "codebook_generation": (
                "uniform unique binary subcodes per adaptive head"
            ),
            "class_center_optimization": (
                "dynamic random-class-order greedy distinct reassignment"
            ),
            "reassignment_cost": (
                "paper Eq.7 weighted mean squared Euclidean distance"
            ),
            "official_result_difference": (
                "official experiments use augmented pixels, ResNet-34 and "
                "reported 16/32/64-bit protocols; matched runner uses the "
                "selected immutable cache, repository splits and 36/48 bits"
            ),
            "numeric_precision": "float32",
        }

    def _add_model_specific_args_into_parser(
        self, parser: ArgumentParser
    ) -> ArgumentParser:
        parser.add_argument("--crh_margin", type=float, default=0.2)
        parser.add_argument("--crh_codebook_scale", type=int, default=2)
        parser.add_argument("--crh_update_rounds", type=int, default=20)
        parser.add_argument("--crh_update_interval", type=int, default=5)
        parser.add_argument(
            "--crh_gradient_clip_norm", type=float, default=1.0
        )
        return parser

    def init_experiment(self, config: dict, rename: bool = True) -> None:
        dataset = str(config["dataset"])
        setting = str(config["setting"])
        num_classes = int(NUM_CLASS[dataset][setting])
        codebook_scale = int(config.get("crh_codebook_scale", 2))
        if codebook_scale != 2:
            raise ValueError("the registered CRH variant requires M=2C")
        if int(config.get("crh_update_rounds", 20)) != 20:
            raise ValueError(
                "the registered CRH variant updates every epoch for 20 rounds"
            )
        if int(config.get("crh_update_interval", 5)) != 5:
            raise ValueError(
                "the registered CRH variant uses a five-epoch update interval"
            )
        codebook_size = codebook_scale * num_classes
        dimension = adaptive_head_dim(int(config["bit"]), codebook_size)
        is_multi_label = dataset in {"Flickr25k", "MSCOCO", "NUSWIDE"}
        source_horizon = 30 if is_multi_label else 300
        if config.get("max_epoch") is None:
            config["max_epoch"] = source_horizon
        config.update({
            "crh_num_classes": num_classes,
            "crh_codebook_size": codebook_size,
            "crh_head_dim": dimension,
            "crh_num_heads": int(config["bit"]) // dimension,
            "crh_scale": math.sqrt(2.0) * math.log(num_classes - 1),
            "crh_quantization_weight": 0.0 if is_multi_label else 0.1,
            "crh_source_horizon": source_horizon,
            "crh_source_optimizer": (
                "Adam(lr=1e-4, betas=(0.5,0.999), weight_decay=1e-5)"
            ),
            "crh_source_scheduler": (
                "CosineAnnealingLR(T_max=source_horizon, eta_min=1e-7)"
            ),
        })
        super().init_experiment(config, rename=rename)

    def _build_model_from_config(self, d_in: int, config: dict) -> Module:
        dataset = str(config["dataset"])
        setting = str(config.get("setting", "setting1"))
        num_classes = int(
            config.get("crh_num_classes", NUM_CLASS[dataset][setting])
        )
        return CRHFeatureHash(
            d_in=int(d_in),
            bit=int(config["bit"]),
            num_classes=num_classes,
            seed=int(config.get("seed", 1)),
            codebook_scale=int(config.get("crh_codebook_scale", 2)),
        )

    def _init_optimizer(
        self,
        model: Module,
        name: str,
        learning_rate: float,
        **opts,
    ) -> Optimizer:
        if str(name).lower() != "adam":
            raise ValueError("CRH's audited optimizer is Adam")
        if not isinstance(model, CRHFeatureHash):
            raise TypeError("CRH optimizer received an incompatible model")
        weight_decay = float(opts.get("adam_weight_decay", 1e-5))
        if weight_decay != 1e-5:
            raise ValueError("CRH requires Adam weight_decay=1e-5")
        return torch.optim.Adam(
            model.encoder_layers.parameters(),
            lr=float(learning_rate),
            betas=(0.5, 0.999),
            weight_decay=1e-5,
        )

    def _init_scheduler(
        self, optimizer: Optimizer, name: str, **opts
    ) -> _LRScheduler:
        if str(name).lower() != "coslr":
            raise ValueError("CRH's audited scheduler is cosine annealing")
        horizon = int(
            opts.get("schedule_horizon")
            or opts.get("crh_source_horizon")
            or opts.get("max_epoch", 300)
        )
        if horizon <= 0:
            raise ValueError("CRH cosine horizon must be positive")
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=horizon, eta_min=1e-7
        )

    def _train_model(
        self,
        backbone_with_encoder: Module,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        trainset: Dataset,
        testset: Dataset,
        dbset: Dataset,
        model_dir: str,
        result_dir: str,
        device: str,
        batch_size: int,
        eval_period: int,
        config: dict,
    ) -> None:
        del testset, dbset, model_dir, result_dir, batch_size
        model = backbone_with_encoder.to(device)
        if not isinstance(model, CRHFeatureHash):
            raise TypeError("CRH training received an incompatible model")

        supervised_trainset, partition_audit = (
            prepare_crh_supervised_training_dataset(
                trainset,
                protocol_stage=str(config.get("protocol_stage", "unknown")),
            )
        )
        config.update(partition_audit)
        # Logger writes its config snapshot before method-specific training
        # starts.  Refresh that CRH snapshot now that the stage-specific
        # supervised domain has been audited.  Checkpoints receive the same
        # live config mapping below.
        if self.logger is not None:
            self.logger.config.update(partition_audit)
            snapshot = {
                key: value
                for key, value in config.items()
                if isinstance(
                    value, (str, int, float, bool, list, type(None))
                )
            }
            with open(
                os.path.join(self.result_dir, "config.json"),
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(snapshot, handle, indent=2)

        loader = DataLoader(
            supervised_trainset,
            batch_size=int(config["batch_size"]),
            shuffle=True,
            num_workers=int(config.get("num_workers", 4)),
            drop_last=False,
        )
        if len(loader) == 0:
            raise ValueError("CRH training split is empty")

        margin = float(config["crh_margin"])
        scale = float(config["crh_scale"])
        quantization_weight = float(config["crh_quantization_weight"])
        gradient_clip = float(config["crh_gradient_clip_norm"])
        if gradient_clip <= 0.0:
            raise ValueError("CRH gradient clipping norm must be positive")
        max_epoch = int(config["max_epoch"])

        for epoch in range(max_epoch):
            model.train()
            collect_reassignment = crh_should_reassign(epoch)
            signed_chunks = []
            label_chunks = []
            for batch in loader:
                features = batch["img"].to(device)
                # Labels are method-defining supervision, not evaluator-only
                # metadata.  Stage 1's dataset is already restricted to the
                # optimization-train rows by DeepHashBase.
                labels = batch["label"].to(device)
                output = model(features)
                logits = output["crh_logits"]
                loss, parts = crh_objective(
                    logits,
                    model.current_class_centers,
                    labels,
                    scale=scale,
                    margin=margin,
                    quantization_weight=quantization_weight,
                )
                if collect_reassignment:
                    signed_chunks.append(logits.detach())
                    label_chunks.append(labels.detach())

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = nn.utils.clip_grad_norm_(
                    model.encoder_layers.parameters(), gradient_clip
                )
                optimizer.step()
                self._insert_iter_loss_sum(
                    {
                        "loss": loss.item(),
                        "margin_cross_entropy": parts[
                            "margin_cross_entropy"
                        ].item(),
                        "quantization": parts["quantization"].item(),
                        "gradient_norm": float(gradient_norm),
                        "crh_training_rows_total": float(
                            partition_audit["crh_training_rows_total"]
                        ),
                        "crh_training_rows_retained": float(
                            partition_audit["crh_training_rows_retained"]
                        ),
                        "crh_training_zero_label_rows_dropped": float(
                            partition_audit[
                                "crh_training_zero_label_rows_dropped"
                            ]
                        ),
                    },
                    current_batch_size=int(features.shape[0]),
                )

            scheduler.step()
            assignment_changes = 0
            assignment_cost = 0.0
            if collect_reassignment:
                all_logits = torch.cat(signed_chunks, dim=0)
                all_labels = torch.cat(label_chunks, dim=0)
                costs = crh_weighted_reassignment_cost(
                    all_logits, all_labels, model.codebook
                )
                assignments, per_head_cost = greedy_distinct_assignments(
                    costs,
                    # An epoch-derived seed makes the random class order
                    # deterministic without hidden global RNG state.
                    seed=int(config["seed"]) + 1_000_003 * (epoch + 1),
                )
                assignment_changes = int(
                    (assignments != model.assignments).sum().item()
                )
                assignment_cost = float(per_head_cost.sum().item())
                model.install_assignments(assignments)

            self._compute_loss_per_epoch()
            model_id = f"{epoch:03d}"
            self._save_train_model_log(model_id, "epoch", True)
            if collect_reassignment:
                print(
                    f"[CRH] epoch={epoch + 1} "
                    f"assignment_changes={assignment_changes} "
                    f"assignment_cost={assignment_cost:.6f}"
                )

            is_validation_stage = self.valset is not None
            should_eval = (
                epoch + 1 == max_epoch
                or (
                    is_validation_stage
                    and (epoch + 1) % int(eval_period) == 0
                )
            )
            if should_eval:
                self.start_eval_process(model_id, "epoch")
                self._save_train_model_params(
                    model_id,
                    "epoch",
                    {
                        "crh_assignments": model.assignments.detach().cpu(),
                        "crh_class_centers": (
                            model.current_class_centers.detach().cpu()
                        ),
                        "crh_training_partition_audit": dict(
                            partition_audit
                        ),
                    },
                )
