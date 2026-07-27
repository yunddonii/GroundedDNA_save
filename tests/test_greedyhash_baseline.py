"""Equation, gradient, protocol, and runner tests for GreedyHash U0."""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from baseline.GreedyHash import (
    OFFICIAL_COMMIT,
    GreedyHash,
    GreedyHashFeatureHash,
    greedy_sign_ste,
    greedyhash_unsupervised_loss,
)


def test_released_sign_layer_is_hard_forward_identity_backward() -> None:
    logits = torch.tensor(
        [-2.0, 0.0, 3.0], dtype=torch.float64, requires_grad=True
    )
    upstream = torch.tensor([1.5, -2.0, 4.0], dtype=torch.float64)
    binary = greedy_sign_ste(logits)
    torch.testing.assert_close(
        binary,
        torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64),
    )
    (binary * upstream).sum().backward()
    # This is the method-defining greedy backward rule, not d(sign)/dh.
    torch.testing.assert_close(logits.grad, upstream)


def test_unsupervised_loss_matches_official_half_pair_equations() -> None:
    # For B=4, official pairs are (row 0,row 2) and (row 1,row 3).
    # The deliberately different adjacent relationships catch accidental
    # conversion to (0,1)/(2,3) pairing.
    features = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    logits = torch.tensor(
        [[0.2, -0.8], [1.3, 0.7], [-0.4, -1.2], [0.6, -0.3]],
        dtype=torch.float64,
        requires_grad=True,
    )
    binary = greedy_sign_ste(logits)
    actual, parts = greedyhash_unsupervised_loss(
        features, logits, binary, quantization_weight=0.1
    )

    expected_code_similarity = F.cosine_similarity(
        binary[:2], binary[2:], dim=1
    )
    expected_feature_similarity = F.cosine_similarity(
        features.detach()[:2], features.detach()[2:], dim=1
    )
    expected_similarity = F.mse_loss(
        expected_code_similarity, expected_feature_similarity
    )
    expected_quantization = (
        (logits.abs() - 1.0).pow(3).abs().mean()
    )
    expected = expected_similarity + 0.1 * expected_quantization

    torch.testing.assert_close(
        parts["similarity_preservation"], expected_similarity
    )
    torch.testing.assert_close(parts["quantization"], expected_quantization)
    torch.testing.assert_close(actual, expected)

    # An adjacent-pair implementation would give a different similarity loss.
    adjacent_similarity = F.mse_loss(
        F.cosine_similarity(binary[::2], binary[1::2], dim=1),
        F.cosine_similarity(
            features.detach()[::2], features.detach()[1::2], dim=1
        ),
    )
    assert not torch.isclose(expected_similarity, adjacent_similarity)

    actual.backward()
    assert logits.grad is not None
    assert bool(torch.isfinite(logits.grad).all())
    # Source similarities are fixed targets, just as frozen VGG fc7 was in
    # the release; optimization cannot leak into the cached representation.
    assert features.grad is None


def test_randomized_value_and_gradient_parity_with_literal_release_code() -> None:
    """Cross-check against an independent transcription of lines 132--141."""

    torch.manual_seed(29)
    features = torch.randn(8, 5, dtype=torch.float64)
    ours_logits = torch.randn(8, 6, dtype=torch.float64, requires_grad=True)
    reference_logits = ours_logits.detach().clone().requires_grad_(True)

    ours, _ = greedyhash_unsupervised_loss(features, ours_logits)

    # Literal official implementation, modernized only for Python-3 integer
    # division and expressed as detach-based STE to keep this oracle
    # independent of ``greedy_sign_ste``.
    hard = torch.sign(reference_logits)
    reference_binary = reference_logits + (hard - reference_logits).detach()
    half = reference_logits.shape[0] // 2
    target_b = F.cosine_similarity(
        reference_binary[:half], reference_binary[half:]
    )
    target_x = F.cosine_similarity(features[:half], features[half:])
    loss_one = F.mse_loss(target_b, target_x)
    loss_two = torch.mean(
        torch.abs(
            torch.pow(
                torch.abs(reference_logits)
                - torch.ones(reference_logits.size(), dtype=torch.float64),
                3,
            )
        )
    )
    reference = loss_one + 0.1 * loss_two

    torch.testing.assert_close(ours, reference, rtol=1e-12, atol=1e-12)
    ours.backward()
    reference.backward()
    torch.testing.assert_close(
        ours_logits.grad, reference_logits.grad, rtol=1e-11, atol=1e-12
    )


def test_loss_rejects_batches_that_cannot_form_equal_halves() -> None:
    with unittest.TestCase().assertRaisesRegex(ValueError, "even mini-batch"):
        greedyhash_unsupervised_loss(
            torch.randn(3, 4), torch.randn(3, 2)
        )


def test_model_is_single_linear_head_and_retrieves_with_hard_codes() -> None:
    model = GreedyHashFeatureHash(d_in=3, bit=2)
    assert isinstance(model.encoder_layers, torch.nn.Linear)
    assert list(model.modules())[1:] == [model.encoder_layers]
    with torch.no_grad():
        model.encoder_layers.weight.copy_(
            torch.tensor([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
        )
        model.encoder_layers.bias.copy_(torch.tensor([0.25, -0.25]))
    features = torch.tensor([[1.0, 2.0, 9.0], [-1.0, -2.0, 4.0]])
    output = model(features)
    assert set(output) == {
        "continuous_code",
        "greedyhash_binary",
        "greedyhash_logits",
        "cnn_feat",
        "backbone_last_output",
    }
    expected_logits = torch.tensor([[1.25, -2.25], [-0.75, 1.75]])
    expected_binary = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
    torch.testing.assert_close(output["greedyhash_logits"], expected_logits)
    torch.testing.assert_close(output["greedyhash_binary"], expected_binary)
    torch.testing.assert_close(output["continuous_code"], expected_binary)


def test_runner_records_exact_u0_release_core_and_optimizer() -> None:
    runner = GreedyHash()
    assert runner._get_default_config_dict() == {
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
    fixed = runner._get_fixed_config_dict()
    assert fixed["source_code_commit"] == OFFICIAL_COMMIT
    assert fixed["dataset_return_paired_aug_img"] is False
    assert fixed["dataset_return_index"] is False
    assert fixed["dataset_return_visual_tokens"] is False
    assert "U0" in fixed["information_condition"]
    assert "target-label-free" in fixed["information_condition"]
    assert "matched-cache-adapter" in fixed["implementation_variant"]
    assert fixed["baseline_display_name"].startswith("UGH (GreedyHash")
    assert "excludes supervised" in fixed["official_task_variant"]
    assert "label" not in fixed

    model = GreedyHashFeatureHash(4, 3)
    optimizer = runner._init_optimizer(
        model,
        name="sgd",
        learning_rate=1e-4,
        sgd_momentum=0.9,
        sgd_weight_decay=5e-4,
    )
    group = optimizer.param_groups[0]
    assert group["lr"] == 1e-4
    assert group["momentum"] == 0.9
    assert group["weight_decay"] == 5e-4
    optimized = {id(parameter) for parameter in group["params"]}
    assert optimized == {
        id(parameter) for parameter in model.encoder_layers.parameters()
    }


class _FeatureOnlyDataset(Dataset):
    """A label-less training set proving the runner never consumes labels."""

    def __init__(self) -> None:
        self.features = torch.tensor(
            [
                [1.0, 0.0, 0.5],
                [0.0, 1.0, -0.5],
                [0.5, 0.5, 1.0],
                [-0.5, 1.0, 0.5],
                [0.25, -0.75, 0.5],
                [-1.0, 0.5, 0.25],
            ]
        )

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> dict:
        return {"img": self.features[index]}


def test_training_loop_accepts_a_dataset_with_no_label_field() -> None:
    runner = GreedyHash()
    model = GreedyHashFeatureHash(3, 2)
    optimizer = runner._init_optimizer(
        model,
        name="sgd",
        learning_rate=1e-4,
        sgd_momentum=0.9,
        sgd_weight_decay=5e-4,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda _: 1.0
    )
    before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    logged = []
    evaluated = []
    runner._insert_iter_loss_sum = (  # type: ignore[method-assign]
        lambda values, current_batch_size: logged.append(
            (values, current_batch_size)
        )
    )
    runner._compute_loss_per_epoch = lambda: None  # type: ignore[method-assign]
    runner._save_train_model_log = lambda *args, **kwargs: None  # type: ignore[method-assign]
    runner._save_train_model_params = lambda *args, **kwargs: None  # type: ignore[method-assign]
    runner.start_eval_process = (  # type: ignore[method-assign]
        lambda *args, **kwargs: evaluated.append(args)
    )

    runner._train_model(
        model,
        optimizer,
        scheduler,
        _FeatureOnlyDataset(),
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        "unused",
        "unused",
        "cpu",
        4,
        5,
        {
            "batch_size": 4,
            "num_workers": 0,
            "max_epoch": 1,
            "greedyhash_quantization_weight": 0.1,
        },
    )
    # Match the official DataLoader contract: the final even short batch is
    # trained rather than dropped.
    assert [batch_size for _, batch_size in logged] == [4, 2]
    assert set(logged[0][0]) == {
        "loss", "similarity_preservation", "quantization"
    }
    assert evaluated
    assert any(
        not torch.equal(before[name], value)
        for name, value in model.state_dict().items()
    )


class GreedyHashBaselineTest(unittest.TestCase):
    def test_sign_ste(self):
        test_released_sign_layer_is_hard_forward_identity_backward()

    def test_loss_equations(self):
        test_unsupervised_loss_matches_official_half_pair_equations()

    def test_release_parity(self):
        test_randomized_value_and_gradient_parity_with_literal_release_code()

    def test_odd_batch_rejected(self):
        test_loss_rejects_batches_that_cannot_form_equal_halves()

    def test_model_contract(self):
        test_model_is_single_linear_head_and_retrieves_with_hard_codes()

    def test_runner_protocol(self):
        test_runner_records_exact_u0_release_core_and_optimizer()

    def test_label_free_training_loop(self):
        test_training_loop_accepts_a_dataset_with_no_label_field()


if __name__ == "__main__":
    unittest.main()
