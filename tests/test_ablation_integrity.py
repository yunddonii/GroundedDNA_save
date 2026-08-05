from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest import mock

import numpy as np
import torch
from torch import nn

import model_siglip2 as model_module
from loss_siglip2 import DNACodonHashLoss
from model_siglip2 import SemanticCodebookQuantizer, SigLIP2SemanticOTModel


class _FakeClip(nn.Module):
    """Backbone shell sufficient for constructing the hashing model offline."""

    def __init__(self) -> None:
        super().__init__()
        self.vision_hidden_dim = 6
        self.text_hidden_dim = 4
        self.projection_dim = 4
        core = nn.Module()
        core.visual_projection = nn.Linear(6, 4, bias=False)
        core.logit_scale = nn.Parameter(torch.tensor(0.0))
        core.vision_model = nn.Identity()
        core.text_model = nn.Identity()
        self.model = core

    @property
    def vision_model(self) -> nn.Module:
        return self.model.vision_model

    @property
    def text_model(self) -> nn.Module:
        return self.model.text_model


def _ablation_args(**overrides) -> SimpleNamespace:
    values = {
        "device": torch.device("cpu"),
        "batch_size": 4,
        "backbone_type": "clip",
        "clip_backbone": "fake",
        "d_model": 6,
        "num_codebooks": 6,
        "codebook_size": 4,
        "num_codons_per_codebook": 3,
        "codebook_update": "gradient",
        "freeze_backbone": True,
        "router_type": "sinkhorn",
        "c_global_source": "siglip2_global",
        "disable_text_supervision": True,
        "lambda_text_hash_ntxent": 0.05,
        "lambda_xmodal_commit": 0.05,
        "use_gumbel_softmax": False,
        "hash_target_mode": "siglip_cos",
        "lambda_hash": 0.0,
        "lambda_hash_hard": 0.0,
        "lambda_vq": 0.0,
        "lambda_quant": 0.0,
        "lambda_anchor": 0.0,
        "lambda_dna": 0.0,
        "lambda_bu": 0.0,
        "lambda_wasserstein": 0.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _build_ablation_model(args: SimpleNamespace) -> SigLIP2SemanticOTModel:
    with mock.patch.object(
        model_module, "build_clip_backbone", return_value=_FakeClip(),
    ):
        return SigLIP2SemanticOTModel(args).train()


def _assert_text_outputs_absent(
    test: unittest.TestCase,
    outputs: dict,
) -> None:
    test.assertEqual(outputs["routing_mode"], "codebook_mean")
    for key in (
        "text_part_tokens",
        "text_continuous_code",
        "text_quantized_tokens",
        "text_codebook_indices",
        "text_foil_continuous_code",
    ):
        test.assertIsNone(outputs[key], key)


class NoTextSupervisionAblationTest(unittest.TestCase):
    def test_cached_text_cannot_reenter_through_text_dna_fallback(self) -> None:
        args = _ablation_args()
        torch.manual_seed(0)
        model = _build_ablation_model(args)
        outputs = model(
            pixel_values=None,
            part_input_ids=None,
            part_attention_mask=None,
            return_routing=True,
            cached_visual_tokens_raw=torch.randn(4, 3, 6),
            cached_visual_global=torch.randn(4, 4),
            cached_text_part_raw=torch.randn(4, 6, 4),
            cached_has_text=torch.ones(4, dtype=torch.bool),
        )

        _assert_text_outputs_absent(self, outputs)
        losses = DNACodonHashLoss(args)(outputs)
        self.assertEqual(float(losses["loss_text_hash_ntxent_add"]), 0.0)
        self.assertEqual(float(losses["loss_xmodal_commit"]), 0.0)

        # A visual path still backpropagates, but no parameter in the text
        # adapter may receive a gradient in the no-text causal ablation.
        outputs["continuous_code"].sum().backward()
        for parameter in model.text_adapter.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.equal(
                    parameter.grad, torch.zeros_like(parameter.grad),
                ))

    def test_live_text_ids_are_removed_before_feature_extraction(self) -> None:
        args = _ablation_args()
        model = _build_ablation_model(args)
        observed: dict[str, object] = {}

        def fake_feature_extraction(
            this,
            pixel_values,
            part_input_ids=None,
            part_attention_mask=None,
            visual_attention_mask=None,
        ):
            del this, pixel_values, part_attention_mask, visual_attention_mask
            observed["part_input_ids"] = part_input_ids
            return {
                "visual_tokens_raw": torch.randn(4, 3, 6),
                "visual_global": torch.randn(4, 4),
                "text_part_raw": (
                    torch.randn(4, 6, 4)
                    if part_input_ids is not None else None
                ),
            }

        model.feature_extraction = MethodType(fake_feature_extraction, model)
        outputs = model(
            pixel_values=torch.randn(4, 3, 2, 2),
            part_input_ids=torch.ones(4, 6, 3, dtype=torch.long),
            part_attention_mask=torch.ones(4, 6, 3, dtype=torch.long),
        )

        self.assertIsNone(observed["part_input_ids"])
        _assert_text_outputs_absent(self, outputs)

    def test_text_whitening_bundle_cannot_affect_no_text_outputs_or_losses(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            first_path = Path(tmp_dir) / "first_whiten.npz"
            second_path = Path(tmp_dir) / "second_whiten.npz"
            np.savez(
                first_path,
                mu=np.zeros(4, dtype=np.float32),
                U=np.eye(4, dtype=np.float32),
                S=np.ones(4, dtype=np.float32),
            )
            np.savez(
                second_path,
                mu=np.asarray([100.0, -50.0, 7.0, 3.0], dtype=np.float32),
                U=np.eye(4, dtype=np.float32),
                S=np.asarray([0.01, 0.25, 4.0, 100.0], dtype=np.float32),
            )

            def build(path: Path):
                args = _ablation_args(
                    text_embed_transform="partial_whiten",
                    text_whiten_npz=str(path),
                    text_whiten_gamma=1.0,
                    lambda_text_code_kl=0.3,
                    text_hash_counterfactual_weight=1.0,
                    text_hash_ntxent_mode="per_codebook",
                    text_hash_ntxent_skip_global=True,
                    bidirectional_token_prune=True,
                    foreground_text_mask_topk_ratio=0.5,
                    routing_token_ot_evidence=True,
                    routing_token_ot_beta=1.0,
                )
                torch.manual_seed(818)
                return args, _build_ablation_model(args)

            first_args, first_model = build(first_path)
            second_args, second_model = build(second_path)
            self.assertFalse(torch.equal(
                first_model.text_whiten_mu,
                second_model.text_whiten_mu,
            ))
            self.assertFalse(torch.equal(
                first_model.text_whiten_W,
                second_model.text_whiten_W,
            ))

            batch_size, num_slots, num_tokens = 4, 6, 3
            foil_valid = torch.ones(
                batch_size, num_slots, dtype=torch.bool,
            )
            foil_valid[:, 0] = False
            inputs = {
                "pixel_values": torch.randn(batch_size, 3, 2, 2),
                "part_input_ids": torch.randint(
                    0, 30, (batch_size, num_slots, num_tokens),
                ),
                "part_attention_mask": torch.ones(
                    batch_size, num_slots, num_tokens, dtype=torch.long,
                ),
                "part_mask": torch.ones(
                    batch_size, num_slots, dtype=torch.bool,
                ),
                "cached_visual_tokens_raw": torch.randn(batch_size, 5, 6),
                "cached_visual_global": torch.randn(batch_size, 4),
                "cached_text_part_raw": torch.randn(
                    batch_size, num_slots, 4,
                ),
                "cached_has_text": torch.ones(batch_size, dtype=torch.bool),
                "cached_text_tokens": torch.randn(
                    batch_size, num_slots, num_tokens, 4,
                ),
                "cached_text_token_mask": torch.ones(
                    batch_size, num_slots, num_tokens, dtype=torch.bool,
                ),
                "cached_text_foil_raw": torch.randn(
                    batch_size, num_slots, 4,
                ),
                "cached_text_foil_valid": foil_valid,
                "cached_text_foil_tokens": torch.randn(
                    batch_size, num_slots, num_tokens, 4,
                ),
                "cached_text_foil_token_mask": torch.ones(
                    batch_size, num_slots, num_tokens, dtype=torch.bool,
                ),
                "compute_text_foil": True,
            }

            torch.manual_seed(919)
            first_outputs = first_model(**inputs)
            torch.manual_seed(919)
            second_outputs = second_model(**inputs)
            _assert_text_outputs_absent(self, first_outputs)
            _assert_text_outputs_absent(self, second_outputs)
            self.assertEqual(first_outputs.keys(), second_outputs.keys())
            for key, first_value in first_outputs.items():
                second_value = second_outputs[key]
                if torch.is_tensor(first_value):
                    torch.testing.assert_close(
                        first_value, second_value, rtol=0.0, atol=0.0,
                        msg=key,
                    )
                else:
                    self.assertEqual(first_value, second_value, key)

            first_losses = DNACodonHashLoss(first_args)(first_outputs)
            second_losses = DNACodonHashLoss(second_args)(second_outputs)
            for key in (
                "loss_text_hash_ntxent_add",
                "loss_xmodal_commit",
                "loss_text_code_kl",
            ):
                self.assertEqual(float(first_losses[key]), 0.0, key)
                self.assertEqual(float(second_losses[key]), 0.0, key)
            self.assertEqual(first_losses.keys(), second_losses.keys())
            for key, first_value in first_losses.items():
                second_value = second_losses[key]
                if torch.is_tensor(first_value):
                    torch.testing.assert_close(
                        first_value, second_value, rtol=0.0, atol=0.0,
                        msg=key,
                    )
                else:
                    self.assertEqual(first_value, second_value, key)


class SharedCodebookAblationTest(unittest.TestCase):
    def test_ema_aggregates_every_slot_into_the_one_used_bank(self) -> None:
        quantizer = SemanticCodebookQuantizer(
            num_codebooks=2,
            codebook_size=1,
            d_model=2,
            update_mode="ema",
            ema_decay=0.0,
            ema_eps=0.0,
            revive_dead=False,
            share_codebook=True,
        ).train()
        with torch.no_grad():
            quantizer.codebooks.zero_()
            quantizer.cluster_size.fill_(1.0)
            quantizer.embed_avg.zero_()
            quantizer.embed_sqavg.zero_()

        outputs = quantizer(torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]))

        self.assertTrue(torch.equal(
            outputs["codebook_indices"], torch.zeros(1, 2, dtype=torch.long),
        ))
        expected = torch.tensor([0.5, 0.5])
        torch.testing.assert_close(quantizer.codebooks[0, 0], expected)
        torch.testing.assert_close(quantizer.codebooks[1, 0], expected)
        torch.testing.assert_close(
            quantizer.cluster_size[0], quantizer.cluster_size[1],
        )
        torch.testing.assert_close(
            quantizer.embed_avg[0], quantizer.embed_avg[1],
        )

    def test_local_slot_observations_change_subsequent_shared_lookup(self) -> None:
        def updated_bank(local: torch.Tensor) -> torch.Tensor:
            quantizer = SemanticCodebookQuantizer(
                2, 1, 2,
                update_mode="ema",
                ema_decay=0.0,
                ema_eps=0.0,
                revive_dead=False,
                share_codebook=True,
            ).eval()
            with torch.no_grad():
                quantizer.cluster_size.fill_(1.0)
                quantizer.embed_avg.zero_()
                quantizer.embed_sqavg.zero_()
                z = torch.stack((torch.tensor([1.0, 0.0]), local)).view(1, 2, 2)
                quantizer._ema_update(
                    z, torch.zeros(1, 2, dtype=torch.long),
                )
            return quantizer.codebooks[0, 0].clone()

        first = updated_bank(torch.tensor([0.0, 1.0]))
        second = updated_bank(torch.tensor([0.0, 3.0]))
        self.assertFalse(torch.equal(first, second))
        torch.testing.assert_close(first, torch.tensor([0.5, 0.5]))
        torch.testing.assert_close(second, torch.tensor([0.5, 1.5]))

    def test_effective_anchors_and_gradient_views_repeat_bank_zero(self) -> None:
        quantizer = SemanticCodebookQuantizer(
            3, 2, 2,
            update_mode="gradient",
            revive_dead=False,
            share_codebook=True,
        )
        with torch.no_grad():
            quantizer.codebooks[0].copy_(torch.tensor([
                [1.0, 0.0],
                [0.0, 1.0],
            ]))
            quantizer.codebooks[1:].fill_(-99.0)

        effective = quantizer.get_effective_codebooks()
        self.assertTrue(torch.equal(effective[0], effective[1]))
        self.assertTrue(torch.equal(effective[0], effective[2]))
        anchors = quantizer.get_codebook_mean_anchors(exclude_global=False)
        torch.testing.assert_close(anchors[0], anchors[1])
        torch.testing.assert_close(anchors[0], anchors[2])

        effective.sum().backward()
        self.assertIsNotNone(quantizer.codebooks.grad)
        self.assertGreater(float(quantizer.codebooks.grad[0].abs().sum()), 0.0)
        self.assertEqual(float(quantizer.codebooks.grad[1:].abs().sum()), 0.0)

    def test_shared_adaptive_k_is_rejected_instead_of_silently_diverging(self) -> None:
        with self.assertRaisesRegex(ValueError, "adaptive-K"):
            SemanticCodebookQuantizer(
                2, 2, 2, K_max=3, share_codebook=True,
            )

    def test_gradient_shared_checkpoint_serializes_only_canonical_bank(self) -> None:
        quantizer = SemanticCodebookQuantizer(
            3, 2, 2,
            update_mode="gradient",
            revive_dead=False,
            share_codebook=True,
        )
        with torch.no_grad():
            quantizer.codebooks[0].fill_(7.0)
            quantizer.codebooks[1:].fill_(-3.0)
        saved = quantizer.state_dict()["codebooks"]
        self.assertTrue(torch.equal(saved[0], saved[1]))
        self.assertTrue(torch.equal(saved[0], saved[2]))
        self.assertTrue(torch.equal(saved[0], torch.full_like(saved[0], 7.0)))


if __name__ == "__main__":
    unittest.main()
