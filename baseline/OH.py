"""Overview Hashing (OH), clean-room matched-cache adapter.

This module follows the executable core released for:

    Jiaguo Yu, Yuming Shen, and Haofeng Zhang,
    "Hashing One With All", ACM Multimedia 2023, pp. 6420--6431.

The authors' repository is a CIFAR-10/TensorFlow program built around an
online ResNet-50.  GroundedDNA instead fixes a common, precomputed CLIP image
cache.  Consequently this runner is *not* a published-table reproduction: it
replaces the online image tower and stochastic pixel transforms with the same
two cached augmented CLIP views used by the other matched baselines.  The OH
mechanism itself is preserved: a shared hidden layer, a hard {0,1} hash head
with a sigmoid straight-through estimator (STE), a normalized continuous
head, an EMA key encoder, three FIFO queues, overview aggregation, and the two
cross-entropy objectives.

The public release at commit ``181691b2adc6fa0767a27a589e034de871764e7b``
has no LICENSE/COPYING file.  No source is vendored or translated here; this is
an independent implementation audited equation-by-equation and
operation-by-operation against that snapshot.

Important representation boundary
---------------------------------
Internally OH must keep its hash values in {0,1}, because overview similarity
is a soft Hamming match count

    b B^T + (1-b)(1-B)^T.

The repository-wide extractor, however, applies ``sign(continuous_code)`` and
expects a signed binary code.  Returning OH's internal 0/1 values directly
would collapse every bit to +1.  ``forward`` therefore exposes the exactly
equivalent signed representation ``2*b-1`` while the training path retains
the source-faithful 0/1 values.
"""

from __future__ import annotations

from argparse import ArgumentParser, BooleanOptionalAction
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, Dataset

from .base_model import DeepHashBase


OFFICIAL_COMMIT = "181691b2adc6fa0767a27a589e034de871764e7b"
OFFICIAL_DOI = "10.1145/3581783.3611977"


def hard_binary_ste(logits: torch.Tensor) -> torch.Tensor:
    """Released sigmoid + threshold-0.5 straight-through binary activation.

    The forward value is exactly in ``{0,1}``; its backward derivative is the
    derivative of ``sigmoid(logits)``.  Keeping this helper explicit prevents
    accidental replacement by a signed STE or a soft sigmoid code.
    """

    probabilities = torch.sigmoid(logits)
    hard = (probabilities > 0.5).to(probabilities.dtype)
    return probabilities + (hard - probabilities).detach()


class OHEncoder(nn.Module):
    """OH's shared trunk plus binary and continuous heads.

    The released image model is ``ResNet50 -> Dense(2048, ReLU)`` followed by
    a binary sigmoid/STE head and a 1024-D L2-normalized continuous head.  In
    the matched adapter the already frozen CLIP vector replaces ResNet50, but
    the released post-backbone head is unchanged.
    """

    def __init__(self, d_in: int, bit: int, middle_dim: int,
                 continuous_dim: int) -> None:
        super().__init__()
        if min(d_in, bit, middle_dim, continuous_dim) <= 0:
            raise ValueError("all OH dimensions must be positive")
        self.trunk = nn.Sequential(
            nn.Linear(int(d_in), int(middle_dim)),
            nn.ReLU(),
        )
        self.binary_head = nn.Linear(int(middle_dim), int(bit))
        self.continuous_head = nn.Linear(
            int(middle_dim), int(continuous_dim))
        # tf.keras.layers.Dense defaults to Glorot-uniform kernels and zero
        # biases; PyTorch Linear's default is Kaiming-uniform, so initialize
        # explicitly rather than silently changing the released head.
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(features.float())
        binary_01 = hard_binary_ste(self.binary_head(hidden))
        continuous = F.normalize(self.continuous_head(hidden), dim=-1)
        return binary_01, continuous


class OHFeatureHash(nn.Module):
    """Complete OH state, including EMA encoder and all three queues."""

    def __init__(self, d_in: int, bit: int, *, middle_dim: int = 2048,
                 continuous_dim: int = 1024, momentum: float = 0.999,
                 temperature: float = 0.2, queue_length: int = 4096,
                 context_weight: float = 0.5,
                 shuffle_key_batch: bool = True) -> None:
        super().__init__()
        if not 0.0 <= float(momentum) < 1.0:
            raise ValueError("OH EMA momentum must lie in [0,1)")
        if float(temperature) <= 0:
            raise ValueError("OH temperature must be positive")
        if int(queue_length) <= 0:
            raise ValueError("OH queue length must be positive")
        if float(context_weight) < 0:
            raise ValueError("OH context weight must be non-negative")

        self.bit = int(bit)
        self.continuous_dim = int(continuous_dim)
        self.momentum = float(momentum)
        self.temperature = float(temperature)
        self.queue_length = int(queue_length)
        self.context_weight = float(context_weight)
        self.shuffle_key_batch = bool(shuffle_key_batch)

        # Instantiate both networks independently, as the release does, then
        # copy q -> k.  Only q is optimized; k evolves exclusively by EMA.
        self.encoder_q = OHEncoder(
            d_in, bit, middle_dim, continuous_dim)
        self.encoder_k = OHEncoder(
            d_in, bit, middle_dim, continuous_dim)
        self.encoder_k.load_state_dict(self.encoder_q.state_dict())
        for parameter in self.encoder_k.parameters():
            parameter.requires_grad_(False)

        # The source initializes all queues with unnormalized N(0,1) values,
        # in P/B/C order.  In particular B is *not* initially binary and P/C
        # are not initially normalized.  Buffers ensure complete checkpoints.
        self.register_buffer(
            "p_queue", torch.randn(self.queue_length, self.continuous_dim))
        self.register_buffer(
            "b_queue", torch.randn(self.queue_length, self.bit))
        self.register_buffer(
            "c_queue", torch.randn(self.queue_length, self.continuous_dim))

    @property
    def encoder_layers(self) -> OHEncoder:
        """Compatibility view used by ``DeepHashBase``'s legacy save field.

        This is a property rather than a second registered module alias, so the
        full state dict has one unambiguous ``encoder_q.*`` namespace.
        """

        return self.encoder_q

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Canonical inference path with a signed extractor-facing code."""

        features = features.float()
        binary_01, continuous = self.encoder_q(features)
        return {
            "continuous_code": 2.0 * binary_01 - 1.0,
            "oh_binary_01": binary_01,
            "oh_continuous_head": continuous,
            "cnn_feat": features,
            "backbone_last_output": features,
        }

    def overview(self, binary_01: torch.Tensor,
                 continuous: torch.Tensor) -> torch.Tensor:
        """Aggregate the continuous queue using hash-space match attention.

        This is the public program's exact sequence: softmax over the
        bit-normalized Hamming-match score, L2-normalize the queue aggregate,
        add ``0.5*c`` (configurable only for explicit audits), then L2-normalize
        the final representation.
        """

        if binary_01.ndim != 2 or continuous.ndim != 2:
            raise ValueError("OH overview expects rank-2 tensors")
        if binary_01.shape[0] != continuous.shape[0]:
            raise ValueError("OH binary/continuous batch sizes differ")
        if binary_01.shape[1] != self.bit:
            raise ValueError("OH binary width differs from configured bit")
        if continuous.shape[1] != self.continuous_dim:
            raise ValueError("OH continuous width differs from configuration")

        # q is differentiated after the queues are updated later in the same
        # forward.  Snapshotting prevents an in-place version error and, more
        # importantly, ensures q's gradient uses the old dictionary values it
        # actually saw.  Key/q2 branches run under no_grad and need no copy.
        needs_snapshot = torch.is_grad_enabled() and (
            binary_01.requires_grad or continuous.requires_grad)
        b_memory = self.b_queue.detach().clone() if needs_snapshot else self.b_queue
        c_memory = self.c_queue.detach().clone() if needs_snapshot else self.c_queue
        similarity = (
            binary_01 @ b_memory.transpose(0, 1)
            + (1.0 - binary_01) @ (1.0 - b_memory).transpose(0, 1)
        )
        attention = torch.softmax(
            similarity / (self.bit * self.temperature), dim=-1)
        aggregate = F.normalize(attention @ c_memory, dim=-1)
        return F.normalize(
            aggregate + self.context_weight * continuous, dim=-1)

    @torch.no_grad()
    def momentum_update_key_encoder(self) -> None:
        """Apply the source EMA update immediately before key encoding."""

        for query, key in zip(
                self.encoder_q.parameters(), self.encoder_k.parameters()):
            key.mul_(self.momentum).add_(query, alpha=1.0 - self.momentum)

    @staticmethod
    @torch.no_grad()
    def _fifo_prepend(queue: torch.Tensor, values: torch.Tensor) -> None:
        n = int(values.shape[0])
        if n <= 0:
            raise ValueError("cannot enqueue an empty OH batch")
        if n > queue.shape[0]:
            raise ValueError(
                f"OH batch {n} exceeds queue length {queue.shape[0]}")
        replacement = torch.cat(
            (values.detach(), queue[:-n].clone()), dim=0)
        queue.copy_(replacement)

    @torch.no_grad()
    def update_queues(self, overview_key: torch.Tensor,
                      binary_key: torch.Tensor,
                      continuous_key: torch.Tensor) -> None:
        """Prepend one batch to P/B/C in the release's FIFO order."""

        sizes = {
            int(overview_key.shape[0]), int(binary_key.shape[0]),
            int(continuous_key.shape[0]),
        }
        if len(sizes) != 1:
            raise ValueError("OH queue update batch sizes differ")
        self._fifo_prepend(self.p_queue, overview_key)
        self._fifo_prepend(self.b_queue, binary_key)
        self._fifo_prepend(self.c_queue, continuous_key)

    def training_logits(self, first_view: torch.Tensor,
                        second_view: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run one source-ordered OH training forward.

        Ordering is method-defining:

        1. q from view 1 attends to the old queues;
        2. EMA-update k, encode shuffled view 2, and attend to old queues;
        3. form queue-positive/negative logits;
        4. enqueue k/b_k/c_k;
        5. recompute q2 from view 2 against the *updated* queues and stop grad;
        6. form in-batch q-vs-q2 logits.
        """

        if first_view.shape != second_view.shape or first_view.ndim != 2:
            raise ValueError(
                "OH expects two same-shaped [batch, feature] cached views")
        batch_size = int(first_view.shape[0])
        if batch_size <= 0:
            raise ValueError("OH requires a non-empty batch")
        if batch_size > self.queue_length:
            raise ValueError("OH batch exceeds the FIFO queue")

        binary_query, continuous_query = self.encoder_q(first_view.float())
        overview_query = self.overview(binary_query, continuous_query)

        with torch.no_grad():
            self.momentum_update_key_encoder()
            if self.shuffle_key_batch:
                shuffle = torch.randperm(
                    batch_size, device=second_view.device)
                unshuffle = torch.argsort(shuffle)
                binary_key, continuous_key = self.encoder_k(
                    second_view[shuffle].float())
                binary_key = binary_key[unshuffle]
                continuous_key = continuous_key[unshuffle]
            else:
                binary_key, continuous_key = self.encoder_k(
                    second_view.float())
            overview_key = self.overview(binary_key, continuous_key)

        # q must retain its gradient in both positive and negative logits.
        positive = torch.einsum(
            "nc,nc->n", overview_query, overview_key
        ).unsqueeze(-1)
        # P is mutated below as well, and its old values are needed to compute
        # d(logits)/d(q) during backward.  Clone for the same reason as B/C in
        # ``overview``; otherwise an in-place queue update invalidates autograd.
        p_memory = self.p_queue.detach().clone()
        negative = overview_query @ p_memory.transpose(0, 1)
        queue_logits = torch.cat((positive, negative), dim=-1)
        queue_logits = queue_logits / self.temperature
        queue_labels = torch.zeros(
            batch_size, dtype=torch.long, device=first_view.device)

        # The queue mutation is deliberately between the two logit blocks.
        self.update_queues(overview_key, binary_key, continuous_key)

        with torch.no_grad():
            binary_second, continuous_second = self.encoder_q(
                second_view.float())
            overview_second = self.overview(
                binary_second, continuous_second).detach()
        batch_logits = overview_query @ overview_second.transpose(0, 1)
        batch_logits = batch_logits / self.temperature
        batch_labels = torch.arange(
            batch_size, dtype=torch.long, device=first_view.device)

        return {
            "queue_logits": queue_logits,
            "queue_labels": queue_labels,
            "batch_logits": batch_logits,
            "batch_labels": batch_labels,
            "binary_query_01": binary_query,
            "continuous_query": continuous_query,
            "overview_query": overview_query,
            "overview_key": overview_key,
            "overview_second": overview_second,
        }


def oh_loss(outputs: Dict[str, torch.Tensor]
            ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Released equal-weight sum of queue and in-batch cross entropy."""

    queue = F.cross_entropy(outputs["queue_logits"], outputs["queue_labels"])
    batch = F.cross_entropy(outputs["batch_logits"], outputs["batch_labels"])
    return queue + batch, {"queue_contrastive": queue,
                           "batch_contrastive": batch}


class OH(DeepHashBase):
    """Target-label-free OH under GroundedDNA's fair P0 cache protocol."""

    def _get_default_config_dict(self) -> dict:
        # Executable CIFAR-10 release settings.  The shared P0 driver changes
        # only validation cadence (to five epochs) and the 36-bit storage budget.
        return {
            "batch_size": 50,
            "max_epoch": 200,
            "learning_rate": 1e-5,
            "optimizer_name": "adam",
            "adam_weight_decay": 0.0,
            "oh_adam_epsilon": 1e-7,
            "lr_scheduler": "none",
            "oh_middle_dim": 2048,
            "oh_continuous_dim": 1024,
            "oh_momentum": 0.999,
            "oh_temperature": 0.2,
            "oh_queue_length": 4096,
            "oh_context_weight": 0.5,
            "oh_shuffle_key_batch": True,
        }

    def _get_config_dict_for_dataset(self, default_config: dict,
                                     dataset: str) -> dict:
        # Upstream reports only a CIFAR-10 runner.  Other dataset runs retain
        # the exact executable hyperparameters and are explicitly adapters.
        return default_config

    def _get_fixed_config_dict(self) -> dict:
        return {
            "gen_code_method": "sign",
            "finetune": False,
            "dataset_return_index": False,
            "dataset_return_paired_aug_img": True,
            "dataset_return_visual_tokens": False,
            "transform": "cached_two_view",
            "encoder_layers": "oh_shared2048_binarySTE_continuous1024",
            "batch_norm": False,
            "implementation_variant": "official-release-core-matched-cache-adapter",
            "information_condition": "U0: target-label-free visual-only optimization",
            "information_tier": "U0",
            "source_paper": (
                "Yu et al., Hashing One With All, ACM MM 2023, "
                f"doi:{OFFICIAL_DOI}"
            ),
            "source_code_commit": OFFICIAL_COMMIT,
            "upstream_license_at_audit": "none-detected",
            "backbone_training": "frozen-precomputed-matched-cache",
            "official_result_difference": (
                "release is CIFAR-10 online ResNet50 with stochastic pixel "
                "views and 64 bits; matched runner uses fixed CLIP cached "
                "views, repository splits, and 36 bits"
            ),
            "paper_release_boundary": (
                "paper's one-with-all overview is instantiated by the "
                "release as a length-4096 FIFO approximation"
            ),
            "internal_hash_alphabet": "hard_0_1_sigmoid_STE",
            "extractor_hash_alphabet": "signed_2b_minus_1",
            "queue_initialization": "unnormalized_standard_normal_P_B_C",
            "dense_initialization": "tensorflow_glorot_uniform_zero_bias",
            "queue_update_order": (
                "after queue logits, before stopped-gradient q2 overview"
            ),
            "epoch_definition_difference": (
                "release consumes 50000 repeated CIFAR rows per epoch; matched "
                "P0 consumes one finite pass over its designated train split"
            ),
            "numeric_precision": "float32",
        }

    def _add_model_specific_args_into_parser(
            self, parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument("--oh_middle_dim", type=int, default=2048)
        parser.add_argument("--oh_continuous_dim", type=int, default=1024)
        parser.add_argument("--oh_momentum", type=float, default=0.999)
        parser.add_argument("--oh_temperature", type=float, default=0.2)
        parser.add_argument("--oh_queue_length", type=int, default=4096)
        parser.add_argument("--oh_context_weight", type=float, default=0.5)
        parser.add_argument("--oh_adam_epsilon", type=float, default=1e-7)
        parser.add_argument(
            "--oh_shuffle_key_batch",
            action=BooleanOptionalAction,
            default=True,
        )
        return parser

    def _build_model_from_config(self, d_in: int, config: dict) -> Module:
        return OHFeatureHash(
            d_in=d_in,
            bit=int(config["bit"]),
            middle_dim=int(config["oh_middle_dim"]),
            continuous_dim=int(config["oh_continuous_dim"]),
            momentum=float(config["oh_momentum"]),
            temperature=float(config["oh_temperature"]),
            queue_length=int(config["oh_queue_length"]),
            context_weight=float(config["oh_context_weight"]),
            shuffle_key_batch=bool(config["oh_shuffle_key_batch"]),
        )

    def _init_optimizer(self, model: Module, name: str,
                        learning_rate: float, **opts) -> Optimizer:
        if str(name).lower() != "adam":
            raise ValueError("OH's audited optimizer is Adam")
        if float(opts.get("adam_weight_decay", 0.0)) != 0.0:
            raise ValueError("OH release uses Adam with zero weight decay")
        epsilon = float(opts.get("oh_adam_epsilon", 1e-7))
        if epsilon <= 0:
            raise ValueError("OH Adam epsilon must be positive")
        if not isinstance(model, OHFeatureHash):
            raise TypeError("OH optimizer received an incompatible model")
        # Official train_step requests gradients only for encoder_q.
        return torch.optim.Adam(
            model.encoder_q.parameters(), lr=float(learning_rate),
            betas=(0.9, 0.999), eps=epsilon, weight_decay=0.0,
        )

    def _train_model(self, backbone_with_encoder: Module,
                     optimizer: Optimizer, scheduler: _LRScheduler,
                     trainset: Dataset, testset: Dataset, dbset: Dataset,
                     model_dir: str, result_dir: str, device: str,
                     batch_size: int, eval_period: int,
                     config: dict) -> None:
        del scheduler, testset, dbset, model_dir, result_dir, batch_size
        loader = DataLoader(
            trainset,
            batch_size=int(config["batch_size"]),
            shuffle=True,
            num_workers=int(config.get("num_workers", 4)),
            # Upstream's only released setting has equal-sized batches.  The
            # matched datasets use one finite pass per epoch; dropping a short
            # tail keeps the in-batch objective and queue cadence fixed.
            drop_last=True,
        )
        if len(loader) == 0:
            raise ValueError(
                "OH uses fixed full batches, but train split < batch_size")
        if int(config["batch_size"]) > int(config["oh_queue_length"]):
            raise ValueError("OH batch_size cannot exceed queue_length")

        model = backbone_with_encoder.to(device)
        if not isinstance(model, OHFeatureHash):
            raise TypeError("OH training received an incompatible model")
        for epoch in range(int(config["max_epoch"])):
            model.train()
            for batch in loader:
                # Deliberately never read batch['label'].
                outputs = model.training_logits(
                    batch["img_tr1"].to(device),
                    batch["img_tr2"].to(device),
                )
                loss, parts = oh_loss(outputs)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                self._insert_iter_loss_sum(
                    {
                        "loss": loss.item(),
                        "queue_contrastive": parts[
                            "queue_contrastive"].item(),
                        "batch_contrastive": parts[
                            "batch_contrastive"].item(),
                    },
                    current_batch_size=outputs["queue_logits"].shape[0],
                )

            self._compute_loss_per_epoch()
            model_id = f"{epoch:03d}"
            self._save_train_model_log(model_id, "epoch", True)
            is_validation_stage = self.valset is not None
            should_eval = (
                epoch + 1 == int(config["max_epoch"])
                or (is_validation_stage
                    and (epoch + 1) % int(eval_period) == 0)
            )
            if should_eval:
                self.start_eval_process(model_id, "epoch")
                self._save_train_model_params(model_id, "epoch")
