"""Golden tests for the Bi-half Net paper/release implementation."""

import unittest
from unittest.mock import Mock

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from baseline.BiHalf import (
    BiHalf,
    BiHalfFeatureHash,
    bihalf_assign,
    bihalf_similarity_loss,
)


class BiHalfBaselineTest(unittest.TestCase):
    def test_assignment_is_independent_per_bit_and_exactly_balanced(self) -> None:
        continuous = torch.tensor([
            [4.0, -3.0, 0.2],
            [1.0, 8.0, -0.5],
            [3.0, 2.0, 0.7],
            [-2.0, 1.0, 0.1],
        ])
        actual = bihalf_assign(continuous)
        expected = torch.tensor([
            [1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
            [-1.0, -1.0, -1.0],
        ])
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(actual.sum(dim=0), torch.zeros(3)))

    def test_odd_assignment_matches_release_floor_half_semantics(self) -> None:
        continuous = torch.tensor([[5.0], [4.0], [3.0], [2.0], [1.0]])
        actual = bihalf_assign(continuous)
        self.assertEqual(int((actual == 1).sum()), 2)
        self.assertEqual(int((actual == -1).sum()), 3)

    def test_proxy_backward_matches_official_image_release(self) -> None:
        continuous = torch.tensor([
            [0.2, -0.8], [1.1, 0.4], [-0.1, 1.7], [-2.0, -0.3],
        ], requires_grad=True)
        upstream = torch.tensor([
            [0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8],
        ])
        gamma = 6.0
        binary = bihalf_assign(continuous, gamma=gamma)
        binary.backward(upstream)
        expected = upstream + gamma * (
            continuous.detach() - binary.detach()) / binary.numel()
        torch.testing.assert_close(continuous.grad, expected)

    def test_similarity_loss_matches_released_half_pair_equation(self) -> None:
        features = torch.tensor([
            [1.0, 0.0], [1.0, 1.0], [-1.0, 0.0], [1.0, -1.0],
        ])
        binary = torch.tensor([
            [1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0],
        ], requires_grad=True)
        actual = bihalf_similarity_loss(features, binary)
        expected = F.mse_loss(
            F.cosine_similarity(binary[:2], binary[2:]),
            F.cosine_similarity(features[:2], features[2:]),
        )
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertIsNotNone(binary.grad)
        with self.assertRaisesRegex(ValueError, "even batch"):
            bihalf_similarity_loss(features[:3], binary.detach()[:3])

    def test_inference_exposes_continuous_codes_not_batch_ranks(self) -> None:
        torch.manual_seed(4)
        model = BiHalfFeatureHash(d_in=3, bit=4).eval()
        sample = torch.tensor([[0.1, 0.2, 0.3]])
        companions = torch.randn(7, 3)
        alone = model(sample)["continuous_code"]
        together = model(torch.cat((sample, companions)))["continuous_code"][:1]
        torch.testing.assert_close(alone, together)
        self.assertIs(model.encoder_layers, model.encoder_layers)

    def test_runner_defaults_encode_release_recipe_and_u0_boundary(self) -> None:
        runner = BiHalf()
        defaults = runner._get_default_config_dict()
        self.assertEqual(defaults["optimizer_name"], "sgd")
        self.assertEqual(defaults["batch_size"], 32)
        self.assertEqual(defaults["bihalf_gamma"], 6.0)
        fixed = runner._get_fixed_config_dict()
        self.assertFalse(fixed["dataset_return_paired_aug_img"])
        self.assertEqual(fixed["gen_code_method"], "sign")

    def test_one_epoch_runner_smoke_handles_singleton_remainder(self) -> None:
        class ToyDataset(Dataset):
            def __len__(self) -> int:
                return 5

            def __getitem__(self, index: int) -> dict:
                return {
                    "img": torch.tensor([
                        float(index), float(index + 1), float(index - 2)]),
                    "label": torch.tensor([1.0]),
                }

        runner = BiHalf()
        runner.valset = None
        runner._insert_iter_loss_sum = Mock()
        runner._compute_loss_per_epoch = Mock()
        runner._save_train_model_log = Mock()
        runner.start_eval_process = Mock()
        runner._save_train_model_params = Mock()
        model = BiHalfFeatureHash(d_in=3, bit=4)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60)
        config = {
            "batch_size": 4,
            "num_workers": 0,
            "max_epoch": 1,
            "bihalf_gamma": 6.0,
            "bihalf_odd_batch_policy": "trim_last",
        }
        before = model.encoder_layers.weight.detach().clone()
        runner._train_model(
            model, optimizer, scheduler, ToyDataset(), None, None,
            "", "", "cpu", 4, 1, config)
        self.assertFalse(torch.equal(before, model.encoder_layers.weight))
        runner._insert_iter_loss_sum.assert_called_once()
        runner.start_eval_process.assert_called_once_with("000", "epoch")
        runner._save_train_model_params.assert_called_once_with("000", "epoch")


if __name__ == "__main__":
    unittest.main()
