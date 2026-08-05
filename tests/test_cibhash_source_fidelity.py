from __future__ import annotations

import inspect
import unittest

import torch
from torch import nn

from baseline.CIBHash import (
    CIBHASH_ENCODER_SPEC,
    CIBHASH_SOURCE_HORIZON,
    CIBHash,
    CIBHashCachedEncoder,
)


class CIBHashSourceFidelityTest(unittest.TestCase):
    def test_canonical_config_uses_fixed_lr_and_official_horizon(self) -> None:
        method = CIBHash()
        defaults = method._get_default_config_dict()
        fixed = method._get_fixed_config_dict()
        self.assertEqual(defaults["max_epoch"], CIBHASH_SOURCE_HORIZON)
        self.assertEqual(CIBHASH_SOURCE_HORIZON, 60)
        self.assertEqual(defaults["lr_scheduler"], "none")
        self.assertEqual(fixed["lr_scheduler"], "none")
        self.assertEqual(fixed["encoder_layers"], CIBHASH_ENCODER_SPEC)

    def test_exact_relu_head_replaces_the_historical_single_linear_adapter(self) -> None:
        method = CIBHash()
        model = method._build_model_from_config(512, {
            "bit": 36,
            "encoder_layers": CIBHASH_ENCODER_SPEC,
        })
        self.assertIsInstance(model, CIBHashCachedEncoder)
        self.assertEqual(
            [type(layer) for layer in model.encoder_layers],
            [nn.Linear, nn.ReLU, nn.Linear],
        )
        self.assertEqual(tuple(model.encoder_layers[0].weight.shape), (1024, 512))
        self.assertEqual(tuple(model.encoder_layers[2].weight.shape), (36, 1024))
        self.assertFalse(any(
            isinstance(layer, (nn.Dropout, nn.GELU))
            for layer in model.modules()
        ))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 562_212)
        self.assertEqual(
            tuple(model(torch.randn(3, 512))["continuous_code"].shape),
            (3, 36),
        )

    def test_legacy_single_linear_checkpoint_shape_remains_reconstructable(self) -> None:
        method = CIBHash()
        legacy = method._build_model_from_config(512, {
            "bit": 36,
            "encoder_layers": "none",
            "batch_norm": False,
        })
        self.assertEqual(sum(p.numel() for p in legacy.parameters()), 18_468)
        self.assertNotIsInstance(legacy, CIBHashCachedEncoder)

    def test_training_loop_never_steps_a_scheduler(self) -> None:
        source = inspect.getsource(CIBHash._train_model)
        self.assertNotIn("scheduler.step", source)
        self.assertIn("model.train()", source)


if __name__ == "__main__":
    unittest.main()
