"""Bi-half Net (AAAI 2021) matched-cache baseline.

Clean-room implementation of the image-hashing method from:

    Yunqiang Li and Jan van Gemert,
    "Deep Unsupervised Image Hashing by Maximizing Bit Entropy."

The method-defining layer is kept exact.  For every hash bit, the training
mini-batch is sorted independently; the largest ``floor(N/2)`` values are
assigned ``+1`` and the rest ``-1``.  Its proxy derivative is

    dL/dU = dL/dB + gamma * (U - B) / (N * K),

which is the operation in the authors' image-hashing release.  At retrieval
time the release does *not* use batch ranks: it applies ``sign`` to the
continuous hash output.  Consequently ``forward`` exposes the continuous
output expected by :mod:`baseline.base_model`, while ``training_codes`` is
the only path that invokes the batch-dependent Bi-half layer.

Protocol boundary
-----------------
The released model is frozen ImageNet VGG-16 features followed by one linear
hash layer.  GroundedDNA replaces those frozen features with the experiment's
audited frozen visual cache and preserves the one-linear-layer hash head and
the complete loss/quantizer.  This is therefore a matched-backbone adapter,
not a reproduction of the paper's VGG table entries.

Audited upstream snapshot:
``liyunqianggyn/Deep-Unsupervised-Image-Hashing@1cd671a33fa010e24fdb36b625837dea9858301f``.
"""

from __future__ import annotations

from argparse import ArgumentParser
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn import Module
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset

from .base_model import DeepHashBase


OFFICIAL_COMMIT = "1cd671a33fa010e24fdb36b625837dea9858301f"
OFFICIAL_DOI = "10.1609/aaai.v35i3.16296"


class _BiHalfFunction(Function):
    """Exact released mini-batch assignment and proxy derivative."""

    @staticmethod
    def forward(ctx, continuous: torch.Tensor,
                gamma: float) -> torch.Tensor:
        if continuous.ndim != 2:
            raise ValueError("Bi-half expects a rank-2 [batch, bit] tensor")
        if continuous.shape[0] < 2 or continuous.shape[1] < 1:
            raise ValueError("Bi-half needs at least two rows and one bit")
        if not torch.is_floating_point(continuous):
            raise TypeError("Bi-half expects floating-point continuous codes")
        gamma = float(gamma)
        if gamma < 0:
            raise ValueError("Bi-half gamma must be non-negative")

        # This deliberately mirrors the public program's descending sort and
        # scatter.  In particular, an odd N has floor(N/2) positive values and
        # ceil(N/2) negative values; no median/sign shortcut is equivalent.
        indices = torch.sort(continuous, dim=0, descending=True).indices
        n_rows, n_bits = continuous.shape
        n_positive = n_rows // 2
        ordered_targets = torch.cat((
            torch.ones(
                (n_positive, n_bits), device=continuous.device,
                dtype=continuous.dtype),
            -torch.ones(
                (n_rows - n_positive, n_bits), device=continuous.device,
                dtype=continuous.dtype),
        ), dim=0)
        binary = torch.zeros_like(continuous).scatter(
            0, indices, ordered_targets)
        ctx.gamma = gamma
        ctx.save_for_backward(continuous, binary)
        return binary

    @staticmethod
    def backward(ctx, grad_binary: torch.Tensor
                 ) -> Tuple[torch.Tensor, None]:
        continuous, binary = ctx.saved_tensors
        proxy = grad_binary + ctx.gamma * (
            continuous - binary) / binary.numel()
        return proxy, None


def bihalf_assign(continuous: torch.Tensor, gamma: float = 6.0
                  ) -> torch.Tensor:
    """Quantize one training mini-batch with the released Bi-half layer."""

    return _BiHalfFunction.apply(continuous, float(gamma))


def bihalf_similarity_loss(features: torch.Tensor,
                           binary_codes: torch.Tensor) -> torch.Tensor:
    """Released unsupervised cosine-relation reconstruction objective.

    Rows are paired as ``first_half[i]`` and ``second_half[i]``.  The official
    image programs use even batches; rejecting odd batches here makes that
    precondition visible instead of silently constructing different pairs.
    """

    if features.ndim != 2 or binary_codes.ndim != 2:
        raise ValueError("Bi-half similarity loss expects rank-2 tensors")
    if features.shape[0] != binary_codes.shape[0]:
        raise ValueError("Bi-half feature/code batch sizes differ")
    if features.shape[0] < 2 or features.shape[0] % 2:
        raise ValueError("Bi-half similarity loss requires an even batch")
    half = features.shape[0] // 2
    target_similarity = F.cosine_similarity(
        features[:half], features[half:], dim=1).detach()
    hash_similarity = F.cosine_similarity(
        binary_codes[:half], binary_codes[half:], dim=1)
    return F.mse_loss(hash_similarity, target_similarity)


class BiHalfFeatureHash(nn.Module):
    """Frozen cached feature -> one released linear hash projection."""

    def __init__(self, d_in: int, bit: int) -> None:
        super().__init__()
        if int(d_in) <= 0 or int(bit) <= 0:
            raise ValueError("Bi-half input and code dimensions must be positive")
        self.backbone = nn.Identity()
        self.encoder_layers = nn.Linear(int(d_in), int(bit))

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = features.float()
        continuous = self.encoder_layers(features)
        # DeepHashBase extracts sign(continuous_code), exactly matching the
        # authors' cal_map_*.py inference path.
        return {
            "continuous_code": continuous,
            "cnn_feat": features,
            "backbone_last_output": features,
        }

    def training_codes(self, features: torch.Tensor, *, gamma: float
                       ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self(features)
        continuous = output["continuous_code"]
        binary = bihalf_assign(continuous, gamma=gamma)
        return output["cnn_feat"], continuous, binary


class BiHalf(DeepHashBase):
    """Repository runner for the visual-only Bi-half Net baseline."""

    def _get_default_config_dict(self) -> dict:
        # The public Flickr/CIFAR image programs use this optimizer recipe.
        # The common P0 driver may explicitly replace epochs/batch size while
        # retaining every override in config.json for a fair protocol audit.
        return {
            "batch_size": 32,
            "max_epoch": 100,
            "learning_rate": 1e-4,
            "optimizer_name": "sgd",
            "sgd_momentum": 0.9,
            "sgd_weight_decay": 5e-4,
            "lr_scheduler": "step",
            "step_size": 60,
            "step_gamma": 0.1,
            "bihalf_gamma": 6.0,
            "bihalf_odd_batch_policy": "trim_last",
        }

    #: The public release ships one script per dataset, and they differ in BOTH
    #: the horizon and the LR decay period. D6 trains to the author horizon and
    #: keeps the last checkpoint, so the decay period is now load-bearing:
    #: discarding it ran CIFAR-10 300 epochs with a 60-epoch period, decaying
    #: four times to 1e-8 instead of the author's twice to 1e-6.
    #: NUS-WIDE has no upstream trainer; it keeps the Flickr profile and is
    #: declared as an adaptation in the manifest.
    SOURCE_SCRIPT_PROFILE = {
        "CIFAR10":   {"max_epoch": 300, "step_size": 120,
                      "source": "ImageHashing/Cifar10_I.py:14-16"},
        "Flickr25k": {"max_epoch": 100, "step_size": 60,
                      "source": "ImageHashing/Flickr25k.py:11-13"},
        "MSCOCO":    {"max_epoch": 150, "step_size": 60,
                      "source": "ImageHashing/Mscoco.py:16-18"},
        "NUSWIDE":   {"max_epoch": 100, "step_size": 60,
                      "source": "adaptation: no upstream NUS-WIDE trainer; "
                                "Flickr25k profile"},
    }

    def _get_config_dict_for_dataset(self, default_config: dict,
                                     dataset: str) -> dict:
        profile = self.SOURCE_SCRIPT_PROFILE.get(str(dataset))
        if profile is None:
            return default_config
        config = dict(default_config)
        config["max_epoch"] = int(profile["max_epoch"])
        config["step_size"] = int(profile["step_size"])
        config["bihalf_source_script"] = str(profile["source"])
        return config

    def _get_fixed_config_dict(self) -> dict:
        return {
            "gen_code_method": "sign",
            "finetune": False,
            "dataset_return_index": False,
            "dataset_return_paired_aug_img": False,
            "dataset_return_visual_tokens": False,
            "transform": "none",
            "encoder_layers": "bihalf_single_linear_projection",
            "batch_norm": False,
        }

    def _add_model_specific_args_into_parser(
            self, parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument(
            "--bihalf_gamma", type=float, default=6.0,
            help=("Release proxy-gradient multiplier before division by "
                  "batch_size*bit."),
        )
        parser.add_argument(
            "--bihalf_odd_batch_policy", choices=("trim_last", "error"),
            default="trim_last",
            help=("The published pair loss requires even batches. trim_last "
                  "handles arbitrary P0 split remainders explicitly."),
        )
        return parser

    def init_experiment(self, config: dict, rename: bool = True) -> None:
        gamma = float(config.get("bihalf_gamma", 6.0))
        if gamma < 0:
            raise ValueError("--bihalf_gamma must be non-negative")
        config["information_condition"] = (
            "U0 visual-only; no labels; frozen cached global features")
        config["information_tier"] = "U0"
        config["implementation_variant"] = (
            "official-image-release-matched-cache-adapter")
        config["official_source_commit"] = OFFICIAL_COMMIT
        config["official_source_doi"] = OFFICIAL_DOI
        config["bihalf_training_quantizer"] = (
            "per-bit descending mini-batch rank; floor(N/2) +1, rest -1")
        config["bihalf_inference_quantizer"] = "sign(continuous_code)"
        config["bihalf_proxy_gradient"] = (
            "dL/dB + bihalf_gamma*(U-B)/(batch_size*bit)")
        config["matched_protocol_adaptations"] = [
            "frozen audited visual cache replaces frozen ImageNet VGG-16 features",
            "canonical cached features replace online pixel preprocessing",
            "experiment seed replaces the release's hard-coded torch seed 0",
            "an odd final P0 batch is explicitly trimmed or rejected",
        ]
        config["paper_release_discrepancies"] = [
            "paper reports gamma=3/(N*K); image release implements 6*(U-B)/(N*K)",
            "paper states image LR 1e-4; MSCOCO release script sets LR 1e-3",
            "paper decays LR on loss saturation; release scripts use fixed epoch steps",
        ]
        super().init_experiment(config, rename=rename)

    def _build_model_from_config(self, d_in: int, config: dict) -> Module:
        return BiHalfFeatureHash(d_in=d_in, bit=int(config["bit"]))

    def _train_model(self, backbone_with_encoder: Module,
                     optimizer: Optimizer, scheduler: _LRScheduler,
                     trainset: Dataset, testset: Dataset, dbset: Dataset,
                     model_dir: str, result_dir: str, device: str,
                     batch_size: int, eval_period: int, config: dict) -> None:
        del testset, dbset, model_dir, result_dir, batch_size
        model = backbone_with_encoder.to(device)
        if not isinstance(model, BiHalfFeatureHash):
            raise TypeError("BiHalf requires BiHalfFeatureHash")
        loader = DataLoader(
            trainset, batch_size=int(config["batch_size"]), shuffle=True,
            num_workers=int(config.get("num_workers", 4)), drop_last=False,
            pin_memory=str(device).startswith("cuda"),
        )
        gamma = float(config["bihalf_gamma"])
        odd_policy = str(config["bihalf_odd_batch_policy"])

        for epoch in range(int(config["max_epoch"])):
            model.train()
            for batch in loader:
                features = batch["img"].to(device)
                if features.shape[0] % 2:
                    if odd_policy == "error":
                        raise ValueError(
                            "Bi-half received an odd final batch; use an even "
                            "batch/split or --bihalf_odd_batch_policy trim_last")
                    features = features[:-1]
                if features.shape[0] < 2:
                    # This can only occur for a singleton final remainder.
                    continue

                target_features, continuous, binary = model.training_codes(
                    features, gamma=gamma)
                loss = bihalf_similarity_loss(target_features, binary)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                # The proxy penalty is logged, not added: its derivative is
                # already injected exactly once by _BiHalfFunction.backward.
                proxy_penalty = 0.5 * gamma * F.mse_loss(
                    continuous.detach(), binary.detach())
                self._insert_iter_loss_sum(
                    {"loss": loss.item(),
                     "similarity_reconstruction": loss.item(),
                     "proxy_transport_penalty": proxy_penalty.item()},
                    current_batch_size=int(features.shape[0]),
                )

            scheduler.step()
            self._compute_loss_per_epoch()
            model_id = f"{epoch:03d}"
            self._save_train_model_log(model_id, "epoch", True)

            # Stage 1 may monitor held-out validation.  Stage 2 has a fixed
            # horizon and touches test only at its terminal checkpoint.
            is_validation_stage = self.valset is not None
            should_eval = (
                epoch + 1 == int(config["max_epoch"])
                or (is_validation_stage
                    and (epoch + 1) % int(eval_period) == 0)
            )
            if should_eval:
                self.start_eval_process(model_id, "epoch")
                self._save_train_model_params(model_id, "epoch")
