"""Focused paper/release-boundary tests for the MLS3RDUH cache adapter."""

from __future__ import annotations

import math
import unittest

import torch

from baseline.MLS3RDUH import (
    MLS3RDUH,
    OFFICIAL_RELEASE_COMMIT,
    PAPER_DOI,
    PAPER_HASH_INIT_POLICY,
    PAPER_NEIGHBOR_POLICY,
    PAPER_SOURCE_PROFILE,
    RELEASE_HASH_INIT_POLICY,
    RELEASE_NEIGHBOR_POLICY,
    RELEASE_SOURCE_PROFILE,
    initialize_mls3rduh_hash_head,
    mls3rduh_neighbor_counts,
)


class MLS3RDUHPaperCacheTest(unittest.TestCase):
    def test_paper_and_release_effective_o_are_explicit(self) -> None:
        self.assertEqual(
            mls3rduh_neighbor_counts(5000, 0.06, 0.06),
            (300, 300),
        )
        self.assertEqual(
            mls3rduh_neighbor_counts(
                5000, 0.06, 0.06, policy=PAPER_NEIGHBOR_POLICY),
            (300, 300),
        )
        self.assertEqual(
            mls3rduh_neighbor_counts(
                5000, 0.06, 0.06, policy=RELEASE_NEIGHBOR_POLICY),
            (300, 450),
        )
        with self.assertRaisesRegex(ValueError, "neighbor policy"):
            mls3rduh_neighbor_counts(5000, 0.06, 0.06, policy="implicit")

    def test_fixed_metadata_records_every_paper_release_choice(self) -> None:
        runner = MLS3RDUH()
        defaults = runner._get_default_config_dict()
        self.assertEqual(defaults["batch_size"], 128)
        self.assertEqual(defaults["max_epoch"], 150)
        self.assertEqual(defaults["learning_rate"], 0.04)
        self.assertEqual(defaults["optimizer_name"], "sgd")
        self.assertEqual(defaults["sgd_momentum"], 0.9)
        self.assertEqual(defaults["sgd_weight_decay"], 1e-5)
        self.assertEqual(defaults["k_nn"], 0.06)
        self.assertEqual(defaults["o_nn"], 0.06)
        self.assertEqual(defaults["alpha"], 0.99)

        fixed = runner._get_fixed_config_dict()
        self.assertEqual(fixed["implementation_variant"], PAPER_SOURCE_PROFILE)
        self.assertEqual(
            fixed["release_implementation_variant"], RELEASE_SOURCE_PROFILE)
        self.assertEqual(fixed["source_paper_doi"], PAPER_DOI)
        self.assertEqual(fixed["source_code_commit"], OFFICIAL_RELEASE_COMMIT)
        self.assertEqual(fixed["mls3rduh_neighbor_policy"], PAPER_NEIGHBOR_POLICY)
        self.assertEqual(fixed["mls3rduh_effective_o_formula"], "int(N * o_nn)")
        self.assertEqual(
            fixed["mls3rduh_release_effective_o_formula"],
            "int(N * o_nn * 1.5)",
        )
        self.assertEqual(
            fixed["mls3rduh_hash_head_initialization"],
            PAPER_HASH_INIT_POLICY,
        )
        self.assertEqual(
            fixed["mls3rduh_release_hash_head_initialization"],
            RELEASE_HASH_INIT_POLICY,
        )
        self.assertEqual(fixed["sgd_momentum"], 0.9)
        self.assertIn("momentum_0.9", fixed["mls3rduh_optimizer_choice"])
        self.assertIn("momentum_0.0", fixed["mls3rduh_release_optimizer_choice"])
        self.assertIn("2cosine_minus_1", fixed["mls3rduh_common_similarity"])
        self.assertIn("log_cosh", fixed["mls3rduh_common_objective"])

        model = runner._build_model_from_config(
            16,
            {"bit": 8, "encoder_layers": "layer=1", "batch_norm": False},
        )
        optimizer = runner._init_optimizer(
            model,
            name=defaults["optimizer_name"],
            **defaults,
        )
        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertEqual(optimizer.param_groups[0]["momentum"], 0.9)
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 1e-5)

    def test_default_builder_uses_deterministic_xavier_and_zero_bias(self) -> None:
        config = {"bit": 36, "encoder_layers": "layer=1", "batch_norm": False}
        torch.manual_seed(2020)
        first = MLS3RDUH()._build_model_from_config(512, config)
        torch.manual_seed(2020)
        second = MLS3RDUH()._build_model_from_config(512, config)
        first_projection = list(first.encoder_layers.modules())[-1]
        second_projection = list(second.encoder_layers.modules())[-1]

        torch.testing.assert_close(
            first_projection.weight, second_projection.weight,
            rtol=0.0, atol=0.0,
        )
        torch.testing.assert_close(
            first_projection.bias,
            torch.zeros_like(first_projection.bias),
            rtol=0.0, atol=0.0,
        )
        fan_in, fan_out = 512, 36
        expected_bound = math.sqrt(6.0 / (fan_in + fan_out))
        expected_std = math.sqrt(2.0 / (fan_in + fan_out))
        self.assertLessEqual(
            float(first_projection.weight.abs().max()), expected_bound)
        self.assertAlmostEqual(
            float(first_projection.weight.std()), expected_std,
            delta=expected_std * 0.03,
        )

    def test_release_initialization_remains_an_explicit_callable_noop(self) -> None:
        config = {
            "bit": 36,
            "encoder_layers": "layer=1",
            "batch_norm": False,
            "mls3rduh_hash_head_initialization": RELEASE_HASH_INIT_POLICY,
        }
        torch.manual_seed(17)
        model = MLS3RDUH()._build_model_from_config(32, config)
        projection = list(model.encoder_layers.modules())[-1]
        # torch.nn.Linear's release-default bias is random, unlike paper zero.
        self.assertFalse(torch.equal(
            projection.bias, torch.zeros_like(projection.bias)))
        before_weight = projection.weight.detach().clone()
        before_bias = projection.bias.detach().clone()
        returned = initialize_mls3rduh_hash_head(
            model, policy=RELEASE_HASH_INIT_POLICY)
        self.assertIs(returned, projection)
        torch.testing.assert_close(
            projection.weight, before_weight, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            projection.bias, before_bias, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
