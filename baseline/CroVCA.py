"""Clean-room CroVCA/HashCoder baseline for the matched feature-cache protocol.

This module implements the *target-label-free visual* objective from:

    Moummad et al., "Image Hashing via Cross-View Code Alignment in the Age
    of Foundation Models", CVPR Workshops 2026 (arXiv:2510.27584v3).

Reproduction boundary
---------------------
The paper supports both (a) LoRA adaptation of a foundation-model image
encoder and (b) HashCoder probing on frozen embeddings.  GroundedDNA stores
frozen backbone outputs, so this runner is the latter: the same cached visual
backbone, train/validation/test split, 36-bit budget, and DNA-base retrieval
metric used by the other repository baselines.  Its two positive views are
``visual_global_aug0`` and ``visual_global_aug1`` from the selected cache.
No target label is read during optimization; labels are used only by the
inherited held-out evaluator.

The objective, small HashCoder, initialization, optimizer, and iteration-wise
cosine learning-rate schedule follow the public implementation at commit
``c59aa7cb981777a27e0bc311fc511761bd6e783c``.  This is deliberately named a
``matched-cache probing adaptation`` rather than a reproduction of the paper's
task-specific LoRA rows.  The upstream repository had no LICENSE file at the
audited commit, so the implementation below is written from the published
equations and independently checked behavior; no upstream source is copied.

Audited upstream caveats are persisted in the experiment config:

* Eq. (11) in the paper and Algorithm 1/public code do not use identical
  notation for coding rate.  We use the executable Algorithm-1 form, including
  ``eps`` and the ``(b+B)/(bB)`` factor.
* the public SimDINOv2 loader constructs/remaps a checkpoint state dictionary
  but never loads it, and its DFN loader treats the returned model/preprocess
  tuple as a model.  Neither path is used here because the shared cache fixes
  backbone provenance.
* the paper's principal dataset-specific rows use LoRA and asymmetric Hamming;
  this matched runner freezes the cache and inherits the common symmetric
  36-bit -> 18-base evaluator, as required for a controlled comparison.
"""

from __future__ import annotations

from argparse import ArgumentParser
import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, Dataset

from .base_model import DeepHashBase


OFFICIAL_COMMIT = "c59aa7cb981777a27e0bc311fc511761bd6e783c"
OFFICIAL_PAPER = "Moummad et al., CVPRW 2026, arXiv:2510.27584v3"


class CroVCAHashCoder(nn.Module):
    """The paper's small two-linear-layer HashCoder.

    For an input width ``d`` and code width ``b`` the architecture is
    ``Linear(d,d) -> ReLU -> Linear(d,b) -> BatchNorm1d(b)``.  Linear weights
    use truncated-normal initialization with standard deviation 0.02 and zero
    bias, matching the released module.
    """

    def __init__(self, d_in: int, bit: int) -> None:
        super().__init__()
        if d_in <= 0 or bit <= 0:
            raise ValueError(f"d_in and bit must be positive, got {d_in}, {bit}")
        self.net = nn.Sequential(
            nn.Linear(d_in, d_in),
            nn.ReLU(),
            nn.Linear(d_in, bit),
            nn.BatchNorm1d(bit),
        )
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class CroVCAFeatureHash(nn.Module):
    """HashCoder wrapper satisfying ``DeepHashBase``'s output convention."""

    def __init__(self, d_in: int, bit: int) -> None:
        super().__init__()
        # The checkpoint helper expects method heads under ``encoder_layers``.
        self.encoder_layers = CroVCAHashCoder(d_in=d_in, bit=bit)
        self.bit = int(bit)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = features.float()
        logits = self.encoder_layers(features)
        return {
            "continuous_code": logits,
            "cnn_feat": features,
            "backbone_last_output": features,
        }


def crovca_alignment_loss(logits_one: torch.Tensor,
                           logits_two: torch.Tensor) -> torch.Tensor:
    """Symmetric stop-gradient cross-view BCE (paper Eq. 8).

    Each branch is a student exactly once.  The opposite branch is thresholded
    at zero into a hard ``{0,1}`` teacher, with an explicit detach documenting
    that no straight-through gradient is intended.
    """

    if logits_one.shape != logits_two.shape or logits_one.ndim != 2:
        raise ValueError(
            "CroVCA expects two same-shaped [batch, bit] logit tensors, got "
            f"{tuple(logits_one.shape)} and {tuple(logits_two.shape)}"
        )
    teacher_one = (logits_one.detach() > 0).to(logits_two.dtype)
    teacher_two = (logits_two.detach() > 0).to(logits_one.dtype)
    return 0.5 * (
        F.binary_cross_entropy_with_logits(logits_one, teacher_two)
        + F.binary_cross_entropy_with_logits(logits_two, teacher_one)
    )


def crovca_coding_rate_loss(logits: torch.Tensor,
                             eps: float = 0.05) -> torch.Tensor:
    """Negative coding rate used by Algorithm 1 and the released ``MCR``.

    With row-normalized logits ``V in R^{B x b}``, the executable public form is

    ``- 1/2 logdet(I_b + b/(B eps) V^T V) * (b+B)/(bB)``.

    Cholesky's summed log diagonal evaluates ``1/2 logdet`` exactly.  Float16
    inputs are promoted only for this linear-algebra operation; this is a
    numerical stabilization and does not change the objective.
    """

    if logits.ndim != 2:
        raise ValueError(f"expected [batch, bit] logits, got {tuple(logits.shape)}")
    batch_size, bit = logits.shape
    if batch_size <= 0 or bit <= 0:
        raise ValueError("batch and bit dimensions must be nonzero")
    if not math.isfinite(float(eps)) or eps <= 0:
        raise ValueError(f"eps must be finite and positive, got {eps}")

    work = logits if logits.dtype in (torch.float32, torch.float64) else logits.float()
    normalized = F.normalize(work, dim=1, p=2)
    covariance_sum = normalized.transpose(0, 1) @ normalized
    identity = torch.eye(bit, dtype=work.dtype, device=work.device)
    matrix = identity + (bit / (batch_size * float(eps))) * covariance_sum
    cholesky, info = torch.linalg.cholesky_ex(matrix)
    if bool((info != 0).any()):
        raise RuntimeError("coding-rate matrix was unexpectedly non-positive-definite")
    half_logdet = cholesky.diagonal(dim1=-2, dim2=-1).log().sum()
    return -half_logdet * ((bit + batch_size) / (bit * batch_size))


def crovca_loss(logits_one: torch.Tensor, logits_two: torch.Tensor,
                 eps: float = 0.05, alignment_weight: float = 1.0,
                 diversity_weight: float = 0.1
                 ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Complete target-free CroVCA loss for exactly two augmented views."""

    alignment = crovca_alignment_loss(logits_one, logits_two)
    diversity = 0.5 * (
        crovca_coding_rate_loss(logits_one, eps=eps)
        + crovca_coding_rate_loss(logits_two, eps=eps)
    )
    total = float(alignment_weight) * alignment + float(diversity_weight) * diversity
    return total, {"alignment": alignment, "diversity": diversity}


def crovca_cosine_learning_rate(step: int, total_steps: int,
                                 base_value: float,
                                 final_value: float) -> float:
    """Iteration schedule used by the public release (zero warm-up default)."""

    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    if step < 0 or step >= total_steps:
        raise ValueError(f"step must be in [0, {total_steps}), got {step}")
    return float(final_value + 0.5 * (base_value - final_value)
                 * (1.0 + math.cos(math.pi * step / total_steps)))


class CroVCA(DeepHashBase):
    """Strict visual target-free CroVCA, matched to GroundedDNA's cache."""

    def _get_default_config_dict(self) -> dict:
        # CVPRW paper Table 3, "Small Datasets", plus public eps/min-lr defaults.
        return {
            "batch_size": 256,
            "max_epoch": 5,
            "learning_rate": 1e-3,
            "optimizer_name": "adamw",
            "adam_weight_decay": 1e-2,
            "lr_scheduler": "crovca_iteration_cosine",
            "crovca_min_lr": 1e-6,
            "crovca_eps": 0.05,
            "crovca_alignment_weight": 1.0,
            "crovca_diversity_weight": 0.1,
        }

    def _get_config_dict_for_dataset(self, default_config: dict,
                                     dataset: str) -> dict:
        return default_config

    def _get_fixed_config_dict(self) -> dict:
        return {
            "gen_code_method": "sign",
            "finetune": False,
            "dataset_return_index": False,
            "dataset_return_paired_aug_img": True,
            "dataset_return_visual_tokens": False,
            "transform": "cached_two_view",
            "encoder_layers": "crovca_small_d_to_d_to_bit_bn",
            "batch_norm": True,
            "implementation_variant": "official-objective-matched-cache-probing",
            "information_condition": "U0: target-label-free visual-only optimization",
            "information_tier": "U0",
            "source_paper": OFFICIAL_PAPER,
            "source_code_commit": OFFICIAL_COMMIT,
            "upstream_license_at_audit": "none-detected",
            "backbone_training": "frozen-precomputed-matched-cache",
            "official_result_difference": (
                "paper task-specific rows use LoRA; this runner is frozen probing"
            ),
            "official_evaluation_difference": (
                "paper uses asymmetric Hamming; matched protocol uses symmetric "
                "36-bit and packed 18-base Hamming"
            ),
            "coding_rate_resolution": (
                "Algorithm-1/public-code eps-scaled form; paper display notation differs"
            ),
            "numeric_precision": "float32",
            "audited_source_caveats": [
                "SimDINOv2 checkpoint state is remapped but not loaded upstream",
                "DFN loader returns a tuple that upstream does not unpack",
            ],
        }

    def _add_model_specific_args_into_parser(
            self, parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument("--crovca_min_lr", type=float, default=1e-6)
        parser.add_argument("--crovca_eps", type=float, default=0.05)
        parser.add_argument("--crovca_alignment_weight", type=float, default=1.0)
        parser.add_argument("--crovca_diversity_weight", type=float, default=0.1)
        return parser

    def _build_model_from_config(self, d_in: int, config: dict) -> Module:
        return CroVCAFeatureHash(d_in=d_in, bit=int(config["bit"]))

    def _init_optimizer(self, model: Module, name: str,
                        learning_rate: float, **opts) -> Optimizer:
        if str(name).lower() != "adamw":
            raise ValueError("CroVCA's audited optimizer is AdamW")
        return torch.optim.AdamW(
            model.parameters(), lr=float(learning_rate),
            weight_decay=float(opts.get("adam_weight_decay", 1e-2)),
        )

    def _init_scheduler(self, optimizer: Optimizer, name: str,
                        **opts) -> _LRScheduler:
        if str(name) != "crovca_iteration_cosine":
            raise ValueError("CroVCA requires its iteration-wise cosine schedule")
        # Learning rates are assigned before every optimizer step below.  This
        # inert object preserves DeepHashBase's scheduler interface/checkpoint.
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    def _train_model(self, backbone_with_encoder: Module, optimizer: Optimizer,
                     scheduler: _LRScheduler, trainset: Dataset,
                     testset: Dataset, dbset: Dataset, model_dir: str,
                     result_dir: str, device: str, batch_size: int,
                     eval_period: int, config: dict) -> None:
        del scheduler, testset, dbset, model_dir, result_dir, batch_size
        loader = DataLoader(
            trainset,
            batch_size=int(config["batch_size"]),
            shuffle=True,
            num_workers=int(config.get("num_workers", 4)),
            drop_last=True,
        )
        if len(loader) == 0:
            raise ValueError(
                "CroVCA uses drop_last=True like the release, but the training "
                "split is smaller than batch_size"
            )

        model = backbone_with_encoder.to(device)
        max_epoch = int(config["max_epoch"])
        schedule_horizon = int(config.get("schedule_horizon") or max_epoch)
        if schedule_horizon < max_epoch:
            raise ValueError(
                "CroVCA schedule_horizon cannot be shorter than max_epoch")
        # P0 refit stops at E* but retains the nominal source schedule.  Using
        # max_epoch here would compress the entire cosine decay into E*+1 and
        # change the trajectory selected during stage 1.
        total_steps = schedule_horizon * len(loader)
        global_step = 0
        for epoch in range(max_epoch):
            model.train()
            for batch in loader:
                # Deliberately do not read batch['label'].
                view_one = batch["img_tr1"].to(device)
                view_two = batch["img_tr2"].to(device)
                lr = crovca_cosine_learning_rate(
                    global_step, total_steps,
                    base_value=float(config["learning_rate"]),
                    final_value=float(config["crovca_min_lr"]),
                )
                for group in optimizer.param_groups:
                    group["lr"] = lr

                logits_one = model(view_one)["continuous_code"]
                logits_two = model(view_two)["continuous_code"]
                loss, parts = crovca_loss(
                    logits_one,
                    logits_two,
                    eps=float(config["crovca_eps"]),
                    alignment_weight=float(config["crovca_alignment_weight"]),
                    diversity_weight=float(config["crovca_diversity_weight"]),
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                self._insert_iter_loss_sum(
                    {
                        "loss": loss.item(),
                        "alignment": parts["alignment"].item(),
                        "diversity": parts["diversity"].item(),
                        "learning_rate": lr,
                    },
                    current_batch_size=view_one.shape[0],
                )
                global_step += 1

            self._compute_loss_per_epoch()
            model_id = f"{epoch:03d}"
            self._save_train_model_log(model_id, "epoch", True)
            # Periodic metrics are held-out validation only.  A stage-2 run
            # has an already fixed horizon and evaluates only its terminal
            # checkpoint here, without test-dependent selection.
            is_validation_stage = self.valset is not None
            should_eval = (epoch + 1 == max_epoch
                           or (is_validation_stage
                               and (epoch + 1) % int(eval_period) == 0))
            if should_eval:
                self.start_eval_process(model_id, "epoch")
                self._save_train_model_params(model_id, "epoch")


__all__ = [
    "CroVCA",
    "CroVCAFeatureHash",
    "CroVCAHashCoder",
    "crovca_alignment_loss",
    "crovca_coding_rate_loss",
    "crovca_cosine_learning_rate",
    "crovca_loss",
]
