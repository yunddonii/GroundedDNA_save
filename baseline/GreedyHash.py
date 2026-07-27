"""Unsupervised GreedyHash (NeurIPS 2018) on the matched feature cache.

This module implements the *unsupervised* experiment from:

    Su et al., "Greedy Hash: Towards Fast Optimization for Accurate Hash
    Coding in CNN", NeurIPS 2018.

The paper and its official ``unsupervised_vgg.py`` runner use a frozen VGG16
fc7 representation, a single trainable linear hash layer, and random
first-half/second-half pairs within each shuffled mini-batch.  GroundedDNA's
comparison protocol replaces only that frozen fc7 representation with the
selected frozen visual-feature cache.  The method-defining objective remains
unchanged:

``MSE(cos(sign_STE(h_a), sign_STE(h_b)), cos(x_a, x_b))``
``+ alpha * mean(abs((abs(h) - 1) ** 3))``.

No training label is read.  Labels remain available only to the inherited
held-out retrieval evaluator.  This is a clean-room implementation checked
against the paper and the authors' public code at commit
``e5089b2354b8f283d662bb0584f1b006faf2c2db``; no upstream source is vendored.

Reproduction boundary
---------------------
The official unsupervised result uses online VGG16/fc7 features and CIFAR-10,
whereas this matched-cache adapter uses the repository's immutable backbone,
splits, 36-bit budget, P0 validation protocol, and common binary/base-Hamming
evaluation.  Every sample participating in an update has exactly one partner.
Like the official loader, an even final short batch is retained.  A matched
repository split may have an odd tail, in which case only its final unpaired
row is skipped explicitly.
"""

from __future__ import annotations

from argparse import ArgumentParser
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, Dataset

from .base_model import DeepHashBase


OFFICIAL_COMMIT = "e5089b2354b8f283d662bb0584f1b006faf2c2db"
OFFICIAL_PAPER = (
    "Su et al., Greedy Hash: Towards Fast Optimization for Accurate Hash "
    "Coding in CNN, NeurIPS 2018"
)
OFFICIAL_PAPER_URL = (
    "https://papers.neurips.cc/paper_files/paper/2018/hash/"
    "13f3cf8c531952d72e5847c4183e6910-Abstract.html"
)


class _GreedySignSTE(torch.autograd.Function):
    """The authors' hash layer: ``torch.sign`` forward, identity backward.

    The release uses PyTorch's exact sign semantics, including ``sign(0)=0``.
    This differs at the measure-zero boundary from the paper prose, which
    describes non-positive values as ``-1``.  Preserving the executable
    behavior here makes the implementation auditable rather than silently
    selecting a third convention.
    """

    @staticmethod
    def forward(ctx, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        del ctx
        return torch.sign(inputs)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        del ctx
        return grad_output


def greedy_sign_ste(inputs: torch.Tensor) -> torch.Tensor:
    """Apply the exact released hard-sign straight-through estimator."""

    return _GreedySignSTE.apply(inputs)


class GreedyHashFeatureHash(nn.Module):
    """Frozen-feature GreedyHash with its released one-linear-layer head."""

    def __init__(self, d_in: int, bit: int) -> None:
        super().__init__()
        if int(d_in) <= 0 or int(bit) <= 0:
            raise ValueError(
                f"GreedyHash dimensions must be positive, got {d_in}, {bit}"
            )
        # ``encoder_layers`` is the common checkpoint-facing name.  Current
        # PyTorch Linear reset_parameters is algebraically the same
        # U(-1/sqrt(fan_in), 1/sqrt(fan_in)) initialization used by the
        # official PyTorch-0.3 nn.Linear layer.
        self.encoder_layers = nn.Linear(int(d_in), int(bit))
        self.bit = int(bit)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = features.float()
        logits = self.encoder_layers(features)
        binary = greedy_sign_ste(logits)
        # GreedyHash trains and retrieves with the genuinely discrete output,
        # not a relaxed logit.  The logits remain explicit for the cubic
        # quantization term and for checkpoint/objective audits.
        return {
            "continuous_code": binary,
            "greedyhash_binary": binary,
            "greedyhash_logits": logits,
            "cnn_feat": features,
            "backbone_last_output": features,
        }


def greedyhash_unsupervised_loss(
    features: torch.Tensor,
    logits: torch.Tensor,
    binary: torch.Tensor | None = None,
    *,
    quantization_weight: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """The official unsupervised GreedyHash objective.

    A shuffled mini-batch ``[x_0, ..., x_(B-1)]`` is divided into equal
    contiguous halves.  Pair ``i`` is ``(i, i+B/2)``.  This is deliberately
    not an all-pairs objective and not an adjacent-pair objective.

    Args:
        features: Frozen source-space features ``[B, d]``.
        logits: Pre-sign hash activations ``h`` with shape ``[B, bit]``.
        binary: Optional already-computed hard STE codes.  Supplying the model
            output avoids evaluating the custom sign layer twice.
        quantization_weight: The release's ``alpha`` (default ``0.1``).
    """

    if features.ndim != 2 or logits.ndim != 2:
        raise ValueError(
            "GreedyHash expects rank-2 feature and logit tensors, got "
            f"{tuple(features.shape)} and {tuple(logits.shape)}"
        )
    if features.shape[0] != logits.shape[0]:
        raise ValueError("GreedyHash feature/logit batch sizes differ")
    batch_size = int(features.shape[0])
    if batch_size < 2 or batch_size % 2:
        raise ValueError(
            "GreedyHash needs an even mini-batch of at least two samples"
        )
    if not torch.isfinite(torch.as_tensor(float(quantization_weight))):
        raise ValueError("GreedyHash quantization weight must be finite")
    if float(quantization_weight) < 0.0:
        raise ValueError("GreedyHash quantization weight must be non-negative")

    if binary is None:
        binary = greedy_sign_ste(logits)
    if binary.shape != logits.shape:
        raise ValueError("GreedyHash binary/logit shapes differ")

    half = batch_size // 2
    code_similarity = F.cosine_similarity(
        binary[:half], binary[half:], dim=1
    )
    # In the release, x is produced by a frozen VGG16.  Explicitly detaching
    # preserves that target contract if this mathematical core is reused with
    # a tensor that happens to require gradients.
    frozen_features = features.detach()
    feature_similarity = F.cosine_similarity(
        frozen_features[:half], frozen_features[half:], dim=1
    )
    similarity = F.mse_loss(code_similarity, feature_similarity)

    # Literal release expression:
    # mean(abs(pow(abs(h) - ones, 3))).
    quantization = (logits.abs() - 1.0).pow(3).abs().mean()
    total = similarity + float(quantization_weight) * quantization
    return total, {
        "similarity_preservation": similarity,
        "quantization": quantization,
    }


class GreedyHash(DeepHashBase):
    """U0 unsupervised GreedyHash under the shared frozen-cache protocol."""

    def _get_default_config_dict(self) -> dict:
        # Exact defaults in official ``unsupervised_vgg.py`` for 64 bits.  The
        # source changes only the 16-bit horizon to 300 epochs; GroundedDNA's
        # storage-matched default is 36 bits and therefore uses 60 epochs.
        return {
            "batch_size": 32,
            "max_epoch": 60,
            "eval_period": 5,
            "learning_rate": 1e-4,
            "optimizer_name": "sgd",
            "sgd_momentum": 0.9,
            "sgd_weight_decay": 5e-4,
            "lr_scheduler": "none",
            "greedyhash_quantization_weight": 0.1,
        }

    def _get_config_dict_for_dataset(self, default_config: dict,
                                     dataset: str) -> dict:
        # The official unsupervised experiment reports CIFAR-10 only.  Other
        # repository datasets retain the executable hyperparameters and are
        # marked as matched-cache adaptations in the fixed metadata below.
        del dataset
        return default_config

    def _get_fixed_config_dict(self) -> dict:
        return {
            "gen_code_method": "sign",
            "finetune": False,
            "dataset_return_index": False,
            "dataset_return_paired_aug_img": False,
            "dataset_return_visual_tokens": False,
            "transform": "none",
            "encoder_layers": "greedyhash_single_linear_hard_sign_STE",
            "batch_norm": False,
            "implementation_variant": (
                "official-unsupervised-core-matched-cache-adapter"
            ),
            "baseline_display_name": "UGH (GreedyHash unsupervised setting)",
            "official_task_variant": (
                "unsupervised_vgg.py; excludes supervised cifar/imagenet losses"
            ),
            "information_condition": (
                "U0: target-label-free visual-only optimization"
            ),
            "information_tier": "U0",
            "source_paper": OFFICIAL_PAPER,
            "source_paper_url": OFFICIAL_PAPER_URL,
            "source_code_commit": OFFICIAL_COMMIT,
            "upstream_license_at_audit": "none-detected",
            "backbone_training": "frozen-precomputed-matched-cache",
            "official_result_difference": (
                "official unsupervised result uses frozen online VGG16-fc7 "
                "with train-mode classifier dropout, CIFAR-10 splits, and "
                "16/32/64 bits; matched runner uses the selected immutable "
                "deterministic cache, repository splits, and 36 bits"
            ),
            "paper_release_boundary": (
                "paper defines sign(0)=-1; official torch.sign gives 0; "
                "this core follows the release and the shared extractor "
                "ties an exact zero to +1 for retrieval"
            ),
            "batch_pairing": (
                "shuffle then contiguous first-half/second-half one-to-one pairs"
            ),
            "tail_batch_policy": (
                "retain even short tail; skip at most one final row of odd tail"
            ),
            "official_feature_stochasticity": (
                "VGG weights frozen but cnn.train() leaves fc6/fc7 dropout active"
            ),
            "matched_feature_stochasticity": "immutable deterministic cache",
            "official_16bit_training_horizon": 300,
            "internal_hash_alphabet": "hard_minus1_zero_plus1_torch_sign_STE",
            "extractor_hash_alphabet": "signed_minus1_plus1_zero_tied_plus1",
            "quantization_penalty": "mean(abs((abs(h)-1)^3))",
            "numeric_precision": "float32",
        }

    def _add_model_specific_args_into_parser(
        self, parser: ArgumentParser
    ) -> ArgumentParser:
        parser.add_argument(
            "--greedyhash_quantization_weight", type=float, default=0.1
        )
        return parser

    def _build_model_from_config(self, d_in: int, config: dict) -> Module:
        return GreedyHashFeatureHash(
            d_in=int(d_in), bit=int(config["bit"])
        )

    def _init_optimizer(self, model: Module, name: str,
                        learning_rate: float, **opts) -> Optimizer:
        if str(name).lower() != "sgd":
            raise ValueError("GreedyHash's audited optimizer is SGD")
        momentum = float(opts.get("sgd_momentum", 0.9))
        weight_decay = float(opts.get("sgd_weight_decay", 5e-4))
        if momentum != 0.9 or weight_decay != 5e-4:
            raise ValueError(
                "official unsupervised GreedyHash requires SGD momentum=0.9 "
                "and weight_decay=5e-4"
            )
        if not isinstance(model, GreedyHashFeatureHash):
            raise TypeError("GreedyHash optimizer received an incompatible model")
        # Official code optimizes only ``cnn.fc_encode.parameters()``.
        return torch.optim.SGD(
            model.encoder_layers.parameters(),
            lr=float(learning_rate),
            momentum=0.9,
            weight_decay=5e-4,
        )

    def _train_model(self, backbone_with_encoder: Module,
                     optimizer: Optimizer, scheduler: _LRScheduler,
                     trainset: Dataset, testset: Dataset, dbset: Dataset,
                     model_dir: str, result_dir: str, device: str,
                     batch_size: int, eval_period: int,
                     config: dict) -> None:
        del testset, dbset, model_dir, result_dir, batch_size
        configured_batch = int(config["batch_size"])
        if configured_batch < 2 or configured_batch % 2:
            raise ValueError(
                "GreedyHash batch_size must be an even integer >= 2"
            )
        loader = DataLoader(
            trainset,
            batch_size=configured_batch,
            shuffle=True,
            num_workers=int(config.get("num_workers", 4)),
            # The official loader's default is drop_last=False.  An even
            # short tail is therefore a valid GreedyHash update.
            drop_last=False,
        )
        if len(trainset) < 2 or len(loader) == 0:
            raise ValueError(
                "GreedyHash needs at least two training samples"
            )

        model = backbone_with_encoder.to(device)
        if not isinstance(model, GreedyHashFeatureHash):
            raise TypeError("GreedyHash training received an incompatible model")

        for epoch in range(int(config["max_epoch"])):
            model.train()
            for batch in loader:
                # Deliberately never read ``batch['label']``.
                features = batch["img"].to(device)
                # Official CIFAR-10 has an even 16-row tail.  Other matched
                # splits can be odd; omit only the unpairable final row while
                # retaining every pair in that tail.
                if features.shape[0] % 2:
                    features = features[:-1]
                if features.shape[0] == 0:
                    continue
                output = model(features)
                loss, parts = greedyhash_unsupervised_loss(
                    output["cnn_feat"],
                    output["greedyhash_logits"],
                    output["greedyhash_binary"],
                    quantization_weight=float(
                        config["greedyhash_quantization_weight"]
                    ),
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                self._insert_iter_loss_sum(
                    {
                        "loss": loss.item(),
                        "similarity_preservation": parts[
                            "similarity_preservation"
                        ].item(),
                        "quantization": parts["quantization"].item(),
                    },
                    current_batch_size=int(output["greedyhash_logits"].shape[0]),
                )

            scheduler.step()
            self._compute_loss_per_epoch()
            model_id = f"{epoch:03d}"
            self._save_train_model_log(model_id, "epoch", True)
            is_validation_stage = self.valset is not None
            should_eval = (
                epoch + 1 == int(config["max_epoch"])
                or (
                    is_validation_stage
                    and (epoch + 1) % int(eval_period) == 0
                )
            )
            if should_eval:
                self.start_eval_process(model_id, "epoch")
                self._save_train_model_params(model_id, "epoch")
