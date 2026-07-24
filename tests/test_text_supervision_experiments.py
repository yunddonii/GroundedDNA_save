import ast
import inspect
import math
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from loss_siglip2 import DNACodonHashLoss
from model_siglip2 import SigLIP2SemanticOTModel
from scripts.codebook_drop_ablation_fast import codebook_slices


NUM_CODEBOOKS = 6


class CodebookDropGeometryTest(unittest.TestCase):
    def test_l4_uses_six_four_base_codebook_slices(self):
        slices = codebook_slices(num_bases=24, num_codebooks=6)
        self.assertEqual(
            [(part.start, part.stop) for part in slices],
            [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 24)],
        )

    def test_non_divisible_geometry_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not divisible"):
            codebook_slices(num_bases=23, num_codebooks=6)


class _TinyQuantizer(nn.Module):
    """Deterministic VQ stub with a trainable path for gradient parity."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.75))

    def forward(self, tokens: torch.Tensor) -> dict:
        raw = tokens * self.scale
        # Quantized forward value equals tokens while its gradient follows raw.
        quantized = raw + (tokens - raw).detach()
        return {
            "quantized_tokens": quantized,
            "quantized_tokens_raw": raw,
            "codebook_indices": (tokens[..., 0] > 0).to(torch.long),
        }


class _TinyCodonHead(nn.Module):
    """Head stub that consumes RNG only while it is in training mode."""

    def __init__(self, d_model: int, codon_length: int) -> None:
        super().__init__()
        self.codon_length = codon_length
        self.projection = nn.Linear(
            d_model, codon_length * 4, bias=False,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        residual: torch.Tensor | None,
        gamma: float,
        text_chunks: torch.Tensor | None,
    ) -> dict:
        del text_chunks
        if self.training:
            # Mirrors CodonHead's train-time Gumbel draw without making the
            # continuous probabilities themselves stochastic.
            torch.rand((), device=tokens.device)
        logits = self.projection(tokens)
        if residual is not None:
            logits = logits + float(gamma) * residual[
                :, : self.codon_length * 4
            ]
        return {
            "continuous_code": logits.view(
                tokens.shape[0], self.codon_length, 4,
            ).softmax(dim=-1),
        }


def _tiny_text_dna_model(
    *,
    local_residual: bool,
) -> SigLIP2SemanticOTModel:
    """Allocate only the attributes used by _encode_text_tokens_to_dna."""
    model = SigLIP2SemanticOTModel.__new__(SigLIP2SemanticOTModel)
    nn.Module.__init__(model)
    model.d_model = 12
    model.num_codons_per_codebook = 3
    model.local_residual_quant = local_residual
    model.local_residual_text = local_residual
    model.local_residual_gamma = 0.6
    model.local_residual_detach_global = False
    model.mm_ema = False
    model.codon_residual_gamma = 0.25
    model.codon_input_source = "quantized"
    model.codon_text_anchor = False
    model.quantizer = _TinyQuantizer()
    model.codon_heads = nn.ModuleList([
        _TinyCodonHead(model.d_model, model.num_codons_per_codebook)
        for _ in range(NUM_CODEBOOKS)
    ])
    return model


def _loss_config(**overrides) -> SimpleNamespace:
    """Small, deterministic loss config with unrelated weighted terms off."""
    values = {
        "num_codebooks": NUM_CODEBOOKS,
        "lambda_hash": 0.0,
        "lambda_hash_hard": 0.0,
        "lambda_vq": 0.0,
        "lambda_quant": 0.0,
        "lambda_anchor": 0.0,
        "lambda_dna": 0.0,
        "lambda_bu": 0.0,
        "lambda_text_hash": 0.0,
        "lambda_text_hash_ntxent": 0.0,
        "lambda_xmodal_commit": 0.0,
        "lambda_cibhash_ntxent": 0.0,
        "lambda_cibhash_kl": 0.0,
        "text_hash_ntxent_mode": "per_codebook",
        "text_hash_ntxent_skip_global": True,
        "text_hash_ntxent_temperature": 0.5,
        "text_hash_counterfactual_weight": 0.0,
        "text_hash_counterfactual_margin": 0.1,
        "text_hash_counterfactual_warmup_epochs": 0,
        "xmodal_commit_skip_global": False,
        "cibhash_temperature": 0.3,
        "cibhash_mode": "per_codebook",
        "cibhash_dynamic_tau": False,
        "cibhash_dynamic_tau_alpha": 0.3,
        "cibhash_dynamic_tau_skip_global": False,
        "cibhash_ntxent_source": "continuous_code",
        "cibhash_ntxent_continuous": True,
        "cibhash_visual_token_bit_kl": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _core_outputs(
    continuous_code: torch.Tensor,
    *,
    d_model: int = 5,
    codebook_size: int = 4,
) -> dict:
    """Required tensors for a loss-only synthetic forward."""
    batch = continuous_code.shape[0]
    generator = torch.Generator().manual_seed(20260723)
    return {
        "continuous_code": continuous_code,
        "semantic_visual_tokens": torch.randn(
            batch, NUM_CODEBOOKS, d_model, generator=generator,
        ),
        "quantized_tokens_raw": torch.randn(
            batch, NUM_CODEBOOKS, d_model, generator=generator,
        ),
        "codebook_distances": torch.randn(
            batch, NUM_CODEBOOKS, codebook_size, generator=generator,
        ),
        "local_codebook_mean_anchors": torch.randn(
            NUM_CODEBOOKS - 1, d_model, generator=generator,
        ),
    }


def _labels(batch: int) -> torch.Tensor:
    return torch.eye(batch)


class StrictGlobalCaptionExclusionTest(unittest.TestCase):
    @staticmethod
    def _global_caption_pair(
        batch: int = 3,
        d_text: int = 5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        generator = torch.Generator().manual_seed(17)
        first = torch.randn(
            batch, NUM_CODEBOOKS, d_text, generator=generator,
        )
        # C0 pairwise cosine geometry: orthogonal rows in the first tensor,
        # identical rows in the second. This is a strong, deterministic
        # negative control for text-conditioned dynamic temperature.
        first[:, 0, :] = 0.0
        for index in range(batch):
            first[index, 0, index] = 1.0
        second = first.clone()
        second[:, 0, :] = 0.0
        second[:, 0, 0] = 1.0
        return first, second

    def test_visual_token_dynamic_tau_skip_is_value_and_gradient_invariant(
        self,
    ) -> None:
        criterion = DNACodonHashLoss(_loss_config())
        generator = torch.Generator().manual_seed(7)
        base_view1 = torch.randn(
            3, NUM_CODEBOOKS, 4, generator=generator,
        )
        base_view2 = (
            base_view1
            + 0.25 * torch.randn(
                3, NUM_CODEBOOKS, 4, generator=generator,
            )
        )
        text_first, text_second = self._global_caption_pair()

        losses = []
        gradients = []
        for text in (text_first, text_second):
            view1 = base_view1.clone().requires_grad_()
            view2 = base_view2.clone().requires_grad_()
            loss, _ = criterion._loss_cibhash_visual_per_codebook(
                view1,
                view2,
                temperature=0.3,
                text_part_raw=text,
                dynamic_tau_alpha=0.3,
                dynamic_tau_skip_global=True,
            )
            losses.append(loss.detach())
            gradients.append(torch.autograd.grad(loss, (view1, view2)))

        self.assertTrue(torch.equal(losses[0], losses[1]))
        for first, second in zip(gradients[0], gradients[1]):
            self.assertTrue(torch.equal(first, second))

        # Negative control: without the new flag the same C0 perturbation must
        # alter the objective, proving that the test exercises dynamic tau.
        legacy = []
        for text in (text_first, text_second):
            loss, _ = criterion._loss_cibhash_visual_per_codebook(
                base_view1,
                base_view2,
                temperature=0.3,
                text_part_raw=text,
                dynamic_tau_alpha=0.3,
                dynamic_tau_skip_global=False,
            )
            legacy.append(loss)
        self.assertGreater(float((legacy[0] - legacy[1]).abs()), 1e-5)

    def test_continuous_code_dynamic_tau_skip_is_value_and_gradient_invariant(
        self,
    ) -> None:
        criterion = DNACodonHashLoss(
            _loss_config(cibhash_ntxent_continuous=True),
        )
        generator = torch.Generator().manual_seed(19)
        base_logits1 = torch.randn(
            3, NUM_CODEBOOKS * 3, 4, generator=generator,
        )
        base_logits2 = (
            base_logits1
            + 0.4 * torch.randn(
                3, NUM_CODEBOOKS * 3, 4, generator=generator,
            )
        )
        text_first, text_second = self._global_caption_pair()

        losses = []
        gradients = []
        for text in (text_first, text_second):
            logits1 = base_logits1.clone().requires_grad_()
            logits2 = base_logits2.clone().requires_grad_()
            loss, _ = criterion._loss_cibhash_per_codebook(
                logits1.softmax(dim=-1),
                logits2.softmax(dim=-1),
                temperature=0.3,
                mode="per_codebook",
                text_part_raw=text,
                dynamic_tau_alpha=0.3,
                dynamic_tau_skip_global=True,
            )
            losses.append(loss.detach())
            gradients.append(torch.autograd.grad(loss, (logits1, logits2)))

        self.assertTrue(torch.equal(losses[0], losses[1]))
        for first, second in zip(gradients[0], gradients[1]):
            self.assertTrue(torch.equal(first, second))

        legacy = []
        for text in (text_first, text_second):
            loss, _ = criterion._loss_cibhash_per_codebook(
                base_logits1.softmax(dim=-1),
                base_logits2.softmax(dim=-1),
                temperature=0.3,
                mode="per_codebook",
                text_part_raw=text,
                dynamic_tau_alpha=0.3,
                dynamic_tau_skip_global=False,
            )
            legacy.append(loss)
        self.assertGreater(float((legacy[0] - legacy[1]).abs()), 1e-5)

    def test_xmodal_skip_ignores_both_c0_text_tensors_and_their_gradients(
        self,
    ) -> None:
        config = _loss_config(
            lambda_xmodal_commit=0.1,
            xmodal_commit_skip_global=True,
        )
        generator = torch.Generator().manual_seed(23)
        batch, d_model = 3, 4
        continuous = torch.softmax(
            torch.randn(
                batch, NUM_CODEBOOKS * 3, 4, generator=generator,
            ),
            dim=-1,
        )
        base_visual = torch.randn(
            batch, NUM_CODEBOOKS, d_model, generator=generator,
        )
        base_visual_quantized = torch.randn(
            batch, NUM_CODEBOOKS, d_model, generator=generator,
        )
        base_text = torch.randn(
            batch, NUM_CODEBOOKS, d_model, generator=generator,
        )
        base_text_quantized = torch.randn(
            batch, NUM_CODEBOOKS, d_model, generator=generator,
        )

        losses = []
        gradients = []
        for perturb in (0.0, 50.0):
            visual = base_visual.clone().requires_grad_()
            text = base_text.clone()
            text[:, 0, :] += perturb
            text.requires_grad_()
            text_quantized = base_text_quantized.clone()
            text_quantized[:, 0, :] -= perturb
            outputs = _core_outputs(continuous, d_model=d_model)
            outputs.update({
                "semantic_visual_tokens": visual,
                "quantized_tokens_raw": base_visual_quantized,
                "text_part_tokens": text,
                "text_quantized_tokens": text_quantized,
            })
            result = DNACodonHashLoss(config)(
                outputs,
                multi_hot_labels=_labels(batch),
            )
            loss = result["loss_xmodal_commit"]
            losses.append(loss.detach())
            gradients.append(torch.autograd.grad(loss, (visual, text)))

        self.assertTrue(torch.equal(losses[0], losses[1]))
        for first, second in zip(gradients[0], gradients[1]):
            self.assertTrue(torch.equal(first, second))
            self.assertEqual(float(first[:, 0, :].abs().sum()), 0.0)
            self.assertGreater(float(first[:, 1:, :].abs().sum()), 0.0)

        legacy_config = _loss_config(
            lambda_xmodal_commit=0.1,
            xmodal_commit_skip_global=False,
        )
        legacy_losses = []
        for perturb in (0.0, 50.0):
            text = base_text.clone()
            text[:, 0, :] += perturb
            text_quantized = base_text_quantized.clone()
            text_quantized[:, 0, :] -= perturb
            outputs = _core_outputs(continuous, d_model=d_model)
            outputs.update({
                "semantic_visual_tokens": base_visual,
                "quantized_tokens_raw": base_visual_quantized,
                "text_part_tokens": text,
                "text_quantized_tokens": text_quantized,
            })
            legacy_losses.append(DNACodonHashLoss(legacy_config)(
                outputs,
                multi_hot_labels=_labels(batch),
            )["loss_xmodal_commit"])
        self.assertGreater(
            float((legacy_losses[0] - legacy_losses[1]).abs()),
            1e-3,
        )


class CounterfactualTextDNALossTest(unittest.TestCase):
    def setUp(self) -> None:
        self.batch = 3
        self.codons_per_codebook = 3
        generator = torch.Generator().manual_seed(31)
        shape = (
            self.batch,
            NUM_CODEBOOKS * self.codons_per_codebook,
            4,
        )
        self.visual_logits = torch.randn(shape, generator=generator)
        self.text_logits = torch.randn(shape, generator=generator)
        self.foil_logits = torch.randn(shape, generator=generator)
        self.valid = torch.zeros(
            self.batch, NUM_CODEBOOKS, dtype=torch.bool,
        )
        self.valid[0, 2] = True
        self.valid[2, 5] = True

    def _criterion(self, **overrides) -> DNACodonHashLoss:
        values = {
            "lambda_text_hash_ntxent": 1.0,
            "text_hash_ntxent_mode": "per_codebook",
            "text_hash_ntxent_skip_global": True,
            "text_hash_ntxent_temperature": 0.5,
            "text_hash_counterfactual_weight": 0.7,
            "text_hash_counterfactual_margin": 0.1,
            "text_hash_counterfactual_warmup_epochs": 0,
        }
        values.update(overrides)
        return DNACodonHashLoss(_loss_config(**values))

    def _outputs(
        self,
        visual_logits: torch.Tensor,
        text_logits: torch.Tensor,
        *,
        include_foil: bool = True,
        include_mask: bool = True,
        foil_code: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> dict:
        outputs = _core_outputs(visual_logits.softmax(dim=-1))
        outputs["text_continuous_code"] = text_logits.softmax(dim=-1)
        if include_foil:
            outputs["text_foil_continuous_code"] = (
                self.foil_logits.softmax(dim=-1)
                if foil_code is None
                else foil_code
            )
        if include_mask:
            outputs["text_foil_valid_mask"] = (
                self.valid if valid_mask is None else valid_mask
            )
        return outputs

    def test_own_foil_denominator_matches_manual_formula_and_stops_gradient(
        self,
    ) -> None:
        visual_logits = self.visual_logits.clone().requires_grad_()
        text_logits = self.text_logits.clone().requires_grad_()
        foil_logits = self.foil_logits.clone().requires_grad_()
        outputs = self._outputs(
            visual_logits,
            text_logits,
            foil_code=foil_logits.softmax(dim=-1),
        )
        result = self._criterion()(
            outputs,
            multi_hot_labels=_labels(self.batch),
            epoch=0,
        )

        tau, rho, margin = 0.5, 0.7, 0.1
        length = self.codons_per_codebook
        image = outputs["continuous_code"].view(
            self.batch, NUM_CODEBOOKS, length, 4,
        ).reshape(self.batch, NUM_CODEBOOKS, length * 4)[:, 1:, :]
        text = outputs["text_continuous_code"].view(
            self.batch, NUM_CODEBOOKS, length, 4,
        ).reshape(self.batch, NUM_CODEBOOKS, length * 4)[:, 1:, :]
        foil = outputs["text_foil_continuous_code"].view(
            self.batch, NUM_CODEBOOKS, length, 4,
        ).reshape(self.batch, NUM_CODEBOOKS, length * 4)[:, 1:, :]
        image_n = F.normalize(image, dim=-1)
        text_n = F.normalize(text, dim=-1)
        foil_n = F.normalize(foil.detach(), dim=-1)
        logits = torch.einsum("bmd,cmd->mbc", image_n, text_n) / tau
        foil_cos = torch.einsum("bmd,bmd->mb", image_n, foil_n)
        foil_logits_manual = (
            (foil_cos + margin) / tau + math.log(rho)
        )
        valid_local_mb = self.valid[:, 1:].transpose(0, 1)
        foil_logits_manual = foil_logits_manual.masked_fill(
            ~valid_local_mb, -float("inf"),
        )
        image_to_text_logits = torch.cat(
            (logits, foil_logits_manual.unsqueeze(-1)),
            dim=-1,
        )
        labels = torch.arange(self.batch)
        labels_m = labels.unsqueeze(0).expand(
            NUM_CODEBOOKS - 1, self.batch,
        ).reshape(-1)
        base_image_to_text_per = F.cross_entropy(
            logits.reshape(
                (NUM_CODEBOOKS - 1) * self.batch,
                self.batch,
            ),
            labels_m,
            reduction="none",
        ).reshape(NUM_CODEBOOKS - 1, self.batch)
        foil_image_to_text_per = F.cross_entropy(
            image_to_text_logits.reshape(
                (NUM_CODEBOOKS - 1) * self.batch,
                self.batch + 1,
            ),
            labels_m,
            reduction="none",
        ).reshape(NUM_CODEBOOKS - 1, self.batch)
        # The factual batch InfoNCE keeps its dense mean. The own-foil
        # increment is normalized over valid anchors only, so sparse foil
        # coverage does not silently dilute rho by B*M.
        expected_image_to_text = (
            base_image_to_text_per.mean()
            + (
                foil_image_to_text_per - base_image_to_text_per
            )[valid_local_mb].mean()
        )
        expected_text_to_image = F.cross_entropy(
            logits.transpose(1, 2).reshape(
                (NUM_CODEBOOKS - 1) * self.batch,
                self.batch,
            ),
            labels_m,
        )
        expected = 0.5 * (
            expected_image_to_text + expected_text_to_image
        )
        self.assertTrue(torch.allclose(
            result["loss_text_hash_ntxent_add"],
            expected,
            atol=1e-7,
            rtol=1e-7,
        ))

        visual_grad, text_grad, foil_grad = torch.autograd.grad(
            result["loss_text_hash_ntxent_add"],
            (visual_logits, text_logits, foil_logits),
            allow_unused=True,
        )
        self.assertGreater(float(visual_grad.abs().sum()), 0.0)
        self.assertGreater(float(text_grad.abs().sum()), 0.0)
        self.assertTrue(
            foil_grad is None or float(foil_grad.abs().sum()) == 0.0,
        )

    def test_zero_weight_and_all_invalid_masks_are_exact_legacy(self) -> None:
        outputs = self._outputs(self.visual_logits, self.text_logits)
        zero_weight = self._criterion(
            text_hash_counterfactual_weight=0.0,
        )(
            outputs,
            multi_hot_labels=_labels(self.batch),
            epoch=0,
        )["loss_text_hash_ntxent_add"]
        no_sidecars = self._criterion(
            text_hash_counterfactual_weight=0.0,
        )(
            self._outputs(
                self.visual_logits,
                self.text_logits,
                include_foil=False,
                include_mask=False,
            ),
            multi_hot_labels=_labels(self.batch),
            epoch=0,
        )["loss_text_hash_ntxent_add"]
        all_invalid = self._criterion()(
            self._outputs(
                self.visual_logits,
                self.text_logits,
                valid_mask=torch.zeros_like(self.valid),
            ),
            multi_hot_labels=_labels(self.batch),
            epoch=0,
        )["loss_text_hash_ntxent_add"]
        self.assertTrue(torch.equal(zero_weight, no_sidecars))
        self.assertTrue(torch.equal(zero_weight, all_invalid))

    def test_only_own_valid_local_foil_can_change_the_loss(self) -> None:
        one_valid = torch.zeros_like(self.valid)
        one_valid[0, 2] = True
        base_foil = self.foil_logits.softmax(dim=-1)
        perturbed = base_foil.clone()
        # Perturb C0, every other sample, and every invalid local slot while
        # preserving sample 0's slot-2 codon block exactly.
        keep = torch.zeros_like(perturbed, dtype=torch.bool)
        start = 2 * self.codons_per_codebook
        stop = start + self.codons_per_codebook
        keep[0, start:stop, :] = True
        perturbed[~keep] = torch.flip(
            perturbed[~keep],
            dims=(0,),
        )
        first = self._criterion()(
            self._outputs(
                self.visual_logits,
                self.text_logits,
                foil_code=base_foil,
                valid_mask=one_valid,
            ),
            multi_hot_labels=_labels(self.batch),
        )["loss_text_hash_ntxent_add"]
        second = self._criterion()(
            self._outputs(
                self.visual_logits,
                self.text_logits,
                foil_code=perturbed,
                valid_mask=one_valid,
            ),
            multi_hot_labels=_labels(self.batch),
        )["loss_text_hash_ntxent_add"]
        self.assertTrue(torch.equal(first, second))

        # An erroneously marked C0 foil must fail closed rather than silently
        # weakening the global-caption exclusion contract.
        global_only = torch.zeros_like(self.valid)
        global_only[:, 0] = True
        with self.assertRaisesRegex(ValueError, "C_global|slot 0"):
            self._criterion()(
                self._outputs(
                    self.visual_logits,
                    self.text_logits,
                    foil_code=perturbed,
                    valid_mask=global_only,
                ),
                multi_hot_labels=_labels(self.batch),
            )

    def test_linear_warmup_boundaries(self) -> None:
        criterion = self._criterion(
            text_hash_counterfactual_weight=1.0,
            text_hash_counterfactual_warmup_epochs=4,
        )
        outputs = self._outputs(self.visual_logits, self.text_logits)
        values = [
            criterion(
                outputs,
                multi_hot_labels=_labels(self.batch),
                epoch=epoch,
            )["loss_text_hash_ntxent_add"]
            for epoch in (0, 2, 4, 8)
        ]
        self.assertLess(float(values[0]), float(values[1]))
        self.assertLess(float(values[1]), float(values[2]))
        self.assertTrue(torch.equal(values[2], values[3]))

    def test_enabled_counterfactual_requires_both_foil_outputs(self) -> None:
        criterion = self._criterion()
        cases = (
            (False, False),
            (True, False),
            (False, True),
        )
        for include_foil, include_mask in cases:
            with self.subTest(
                include_foil=include_foil,
                include_mask=include_mask,
            ):
                with self.assertRaisesRegex(
                    (TypeError, ValueError, RuntimeError),
                    "counterfactual|foil",
                ):
                    criterion(
                        self._outputs(
                            self.visual_logits,
                            self.text_logits,
                            include_foil=include_foil,
                            include_mask=include_mask,
                        ),
                        multi_hot_labels=_labels(self.batch),
                        epoch=0,
                    )

    def test_disabled_counterfactual_is_exact_legacy_without_sidecars(
        self,
    ) -> None:
        # Validation calls retain the training recipe's rho but explicitly
        # disable counterfactual execution. They must neither request foil
        # outputs nor perturb the factual text-DNA objective.
        factual_only = self._outputs(
            self.visual_logits,
            self.text_logits,
            include_foil=False,
            include_mask=False,
        )
        disabled = self._criterion()(
            factual_only,
            multi_hot_labels=_labels(self.batch),
            epoch=4,
            enable_counterfactual=False,
        )
        legacy = self._criterion(
            text_hash_counterfactual_weight=0.0,
        )(
            factual_only,
            multi_hot_labels=_labels(self.batch),
            epoch=4,
            enable_counterfactual=True,
        )
        for key in (
            "loss",
            "loss_text_hash_ntxent_add",
            "loss_text_hash_counterfactual",
            "counterfactual_pos_sim",
            "counterfactual_foil_sim",
            "counterfactual_margin_violation",
            "counterfactual_codeword_flip",
            "counterfactual_valid_ratio",
        ):
            with self.subTest(key=key):
                self.assertTrue(torch.equal(disabled[key], legacy[key]))

    def test_foil_valid_mask_must_be_boolean(self) -> None:
        with self.assertRaisesRegex(
            (TypeError, ValueError),
            "bool",
        ):
            self._criterion()(
                self._outputs(
                    self.visual_logits,
                    self.text_logits,
                    valid_mask=self.valid.to(torch.float32),
                ),
                multi_hot_labels=_labels(self.batch),
            )

    def test_valid_foil_values_must_be_finite(self) -> None:
        foil = self.foil_logits.softmax(dim=-1)
        slot = 2
        foil = foil.clone()
        foil[
            0,
            slot * self.codons_per_codebook,
            0,
        ] = float("nan")
        with self.assertRaisesRegex(
            (TypeError, ValueError),
            "finite|NaN|Inf",
        ):
            self._criterion()(
                self._outputs(
                    self.visual_logits,
                    self.text_logits,
                    foil_code=foil,
                ),
                multi_hot_labels=_labels(self.batch),
            )

    def test_nonfinite_values_in_invalid_foil_slots_are_ignored(self) -> None:
        one_valid = torch.zeros_like(self.valid)
        one_valid[0, 2] = True
        foil = self.foil_logits.softmax(dim=-1).clone()
        # Slot 3 is invalid and therefore must not poison the loss.
        invalid_slot = 3
        foil[
            1,
            invalid_slot * self.codons_per_codebook,
            0,
        ] = float("nan")
        result = self._criterion()(
            self._outputs(
                self.visual_logits,
                self.text_logits,
                foil_code=foil,
                valid_mask=one_valid,
            ),
            multi_hot_labels=_labels(self.batch),
        )
        self.assertTrue(torch.isfinite(
            result["loss_text_hash_ntxent_add"],
        ))


class TextDNAHelperRegressionTest(unittest.TestCase):
    @staticmethod
    def _manual_legacy_text_dna(
        model: SigLIP2SemanticOTModel,
        text_tokens: torch.Tensor,
    ) -> dict:
        """The factual pre-refactor text VQ/codon block, kept test-local."""
        batch = text_tokens.shape[0]
        quantizer_tokens = text_tokens
        if (
            model.local_residual_quant
            and model.local_residual_text
            and model.local_residual_gamma > 0.0
        ):
            quantizer_tokens = model._remove_global_projection(
                text_tokens,
                gamma=model.local_residual_gamma,
                detach_global=model.local_residual_detach_global,
                name="text_tokens",
            )

        previous_quantizer_mode = model.quantizer.training
        if not model.mm_ema:
            model.quantizer.eval()
        try:
            quantized = model.quantizer(quantizer_tokens)
        finally:
            if previous_quantizer_mode and not model.mm_ema:
                model.quantizer.train()

        text_q_st = quantized["quantized_tokens"]
        text_q_raw = quantized["quantized_tokens_raw"]
        residual = (
            quantizer_tokens - text_q_raw
            if model.codon_residual_gamma > 0.0
            else None
        )
        head_inputs = (
            quantizer_tokens
            if model.codon_input_source == "routed"
            else text_q_st
        )
        continuous = []
        for slot, head in enumerate(model.codon_heads):
            residual_slot = (
                residual[:, slot, :] if residual is not None else None
            )
            text_chunk = (
                quantizer_tokens[:, slot, :]
                if model.codon_text_anchor
                else None
            )
            continuous.append(head(
                head_inputs[:, slot, :],
                residual=residual_slot,
                gamma=model.codon_residual_gamma,
                text_chunks=text_chunk,
            )["continuous_code"])
        return {
            "continuous_code": torch.stack(
                continuous, dim=1,
            ).reshape(
                batch,
                NUM_CODEBOOKS * model.num_codons_per_codebook,
                4,
            ),
            "quantized_tokens": text_q_st,
            "codebook_indices": quantized["codebook_indices"],
            "quantizer_input": quantizer_tokens,
        }

    def test_factual_helper_matches_legacy_block_with_and_without_residual(
        self,
    ) -> None:
        for local_residual in (False, True):
            with self.subTest(local_residual=local_residual):
                torch.manual_seed(61)
                model = _tiny_text_dna_model(
                    local_residual=local_residual,
                )
                base = torch.randn(
                    2, NUM_CODEBOOKS, model.d_model,
                )
                manual_input = base.clone().requires_grad_()
                helper_input = base.clone().requires_grad_()
                parameters = tuple(model.parameters())
                rng_state = torch.random.get_rng_state()

                manual = self._manual_legacy_text_dna(
                    model, manual_input,
                )
                torch.random.set_rng_state(rng_state)
                helper = model._encode_text_tokens_to_dna(
                    helper_input,
                    allow_mm_ema=True,
                    deterministic_codon=False,
                )

                for key in (
                    "continuous_code",
                    "quantized_tokens",
                    "codebook_indices",
                    "quantizer_input",
                ):
                    self.assertTrue(torch.equal(
                        manual[key],
                        helper[key],
                    ))

                weights = torch.linspace(
                    -1.0,
                    1.0,
                    manual["continuous_code"].numel(),
                ).reshape_as(manual["continuous_code"])
                manual_score = (
                    manual["continuous_code"] * weights
                ).sum()
                helper_score = (
                    helper["continuous_code"] * weights
                ).sum()
                manual_gradients = torch.autograd.grad(
                    manual_score,
                    (manual_input,) + parameters,
                    allow_unused=True,
                )
                helper_gradients = torch.autograd.grad(
                    helper_score,
                    (helper_input,) + parameters,
                    allow_unused=True,
                )
                for manual_gradient, helper_gradient in zip(
                    manual_gradients, helper_gradients,
                ):
                    if manual_gradient is None or helper_gradient is None:
                        self.assertIsNone(manual_gradient)
                        self.assertIsNone(helper_gradient)
                    else:
                        self.assertTrue(torch.equal(
                            manual_gradient,
                            helper_gradient,
                        ))

    def test_deterministic_codon_preserves_rng_and_restores_modes(
        self,
    ) -> None:
        torch.manual_seed(67)
        model = _tiny_text_dna_model(local_residual=True)
        model.quantizer.train()
        for slot, head in enumerate(model.codon_heads):
            head.train(slot % 2 == 0)
        original_head_modes = [
            head.training for head in model.codon_heads
        ]
        tokens = torch.randn(
            2, NUM_CODEBOOKS, model.d_model,
        )
        rng_state = torch.random.get_rng_state().clone()

        model._encode_text_tokens_to_dna(
            tokens,
            allow_mm_ema=False,
            deterministic_codon=True,
        )
        self.assertTrue(torch.equal(
            rng_state,
            torch.random.get_rng_state(),
        ))
        self.assertEqual(
            [head.training for head in model.codon_heads],
            original_head_modes,
        )
        self.assertTrue(model.quantizer.training)

        # Negative control: ordinary factual encoding leaves training heads in
        # train mode, so the fake heads consume RNG exactly as real Gumbel
        # CodonHeads do.
        torch.random.set_rng_state(rng_state)
        model._encode_text_tokens_to_dna(
            tokens,
            allow_mm_ema=False,
            deterministic_codon=False,
        )
        self.assertFalse(torch.equal(
            rng_state,
            torch.random.get_rng_state(),
        ))
        self.assertEqual(
            [head.training for head in model.codon_heads],
            original_head_modes,
        )
        self.assertTrue(model.quantizer.training)

    def test_view2_compute_false_guards_missing_foil_validation(self) -> None:
        # Full model construction would download a backbone. Instead, verify
        # the control dependency in its compiled Python AST: the missing-cache
        # RuntimeError is inside the same `if` whose predicate includes
        # compute_text_foil. Runtime validation-side behavior is covered by
        # test_disabled_counterfactual_is_exact_legacy_without_sidecars.
        source = textwrap.dedent(inspect.getsource(
            SigLIP2SemanticOTModel.forward,
        ))
        tree = ast.parse(source)
        guarded = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            predicate = ast.unparse(node.test)
            body = ast.unparse(ast.Module(
                body=node.body,
                type_ignores=[],
            ))
            if (
                "compute_text_foil" in predicate
                and "cached_text_foil_raw is None" in body
                and "cached_text_foil_valid is None" in body
            ):
                guarded = True
                break
        self.assertTrue(
            guarded,
            "missing-foil validation must be dominated by compute_text_foil",
        )

        trainer_source = (
            Path(__file__).resolve().parents[1] / "train_siglip2.py"
        ).read_text(encoding="utf-8")
        self.assertGreaterEqual(
            trainer_source.count("compute_text_foil=False"),
            2,
            "both paired-augmentation view-2 forwards must disable foil work",
        )


class HybridCIBHashBitKLTest(unittest.TestCase):
    def test_bit_kl_is_zero_for_identical_views_and_has_two_sided_gradients(
        self,
    ) -> None:
        criterion = DNACodonHashLoss(_loss_config())
        for codons_per_codebook in (3, 4):
            with self.subTest(codons_per_codebook=codons_per_codebook):
                generator = torch.Generator().manual_seed(
                    100 + codons_per_codebook,
                )
                shape = (
                    3,
                    NUM_CODEBOOKS * codons_per_codebook,
                    4,
                )
                base_logits1 = torch.randn(shape, generator=generator)
                base_logits2 = (
                    base_logits1
                    + 0.5 * torch.randn(shape, generator=generator)
                )
                identical = criterion._loss_cibhash_bit_kl_only(
                    base_logits1.softmax(dim=-1),
                    base_logits1.softmax(dim=-1),
                    mode="per_codebook",
                )
                self.assertEqual(float(identical), 0.0)

                logits1 = base_logits1.clone().requires_grad_()
                logits2 = base_logits2.clone().requires_grad_()
                kl = criterion._loss_cibhash_bit_kl_only(
                    logits1.softmax(dim=-1),
                    logits2.softmax(dim=-1),
                    mode="per_codebook",
                )
                grad1, grad2 = torch.autograd.grad(kl, (logits1, logits2))
                self.assertGreater(float(kl), 0.0)
                self.assertTrue(torch.isfinite(kl))
                self.assertTrue(torch.isfinite(grad1).all())
                self.assertTrue(torch.isfinite(grad2).all())
                self.assertGreater(float(grad1.abs().sum()), 0.0)
                self.assertGreater(float(grad2.abs().sum()), 0.0)

    def test_visual_token_flag_preserves_ntxent_and_enables_bit_kl(self) -> None:
        generator = torch.Generator().manual_seed(47)
        batch, length, d_model = 3, 3, 5
        logits1 = torch.randn(
            batch, NUM_CODEBOOKS * length, 4, generator=generator,
        )
        logits2 = (
            logits1
            + 0.4 * torch.randn(
                batch, NUM_CODEBOOKS * length, 4, generator=generator,
            )
        )
        visual1 = torch.randn(
            batch, NUM_CODEBOOKS, d_model, generator=generator,
        )
        visual2 = torch.randn(
            batch, NUM_CODEBOOKS, d_model, generator=generator,
        )
        outputs1 = _core_outputs(logits1.softmax(dim=-1), d_model=d_model)
        outputs2 = _core_outputs(logits2.softmax(dim=-1), d_model=d_model)
        outputs1["cibhash_visual_tokens"] = visual1
        outputs2["cibhash_visual_tokens"] = visual2

        common = {
            "lambda_cibhash_ntxent": 1.0,
            "lambda_cibhash_kl": 0.001,
            "cibhash_ntxent_source": "visual_token",
            "cibhash_mode": "per_codebook",
        }
        disabled = DNACodonHashLoss(_loss_config(
            **common,
            cibhash_visual_token_bit_kl=False,
        ))(
            outputs1,
            multi_hot_labels=_labels(batch),
            outputs_view2=outputs2,
        )
        enabled = DNACodonHashLoss(_loss_config(
            **common,
            cibhash_visual_token_bit_kl=True,
        ))(
            outputs1,
            multi_hot_labels=_labels(batch),
            outputs_view2=outputs2,
        )
        self.assertTrue(torch.equal(
            disabled["loss_cibhash_ntxent"],
            enabled["loss_cibhash_ntxent"],
        ))
        self.assertEqual(float(disabled["loss_cibhash_kl"]), 0.0)
        self.assertGreater(float(enabled["loss_cibhash_kl"]), 0.0)
        expected_delta = 0.001 * enabled["loss_cibhash_kl"]
        self.assertTrue(torch.allclose(
            enabled["loss"] - disabled["loss"],
            expected_delta,
            atol=2e-7,
            rtol=2e-4,
        ))

    def test_bit_kl_is_finite_at_probability_boundaries(self) -> None:
        criterion = DNACodonHashLoss(_loss_config())
        first = torch.zeros(2, NUM_CODEBOOKS * 3, 4)
        second = torch.zeros_like(first)
        first[..., 0] = 1.0
        second[..., 3] = 1.0
        for mode in ("per_codebook", "global"):
            with self.subTest(mode=mode):
                kl = criterion._loss_cibhash_bit_kl_only(
                    first,
                    second,
                    mode=mode,
                )
                self.assertTrue(torch.isfinite(kl))
                self.assertGreater(float(kl), 0.0)


if __name__ == "__main__":
    unittest.main()
