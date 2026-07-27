"""Equation, architecture, and protocol tests for clean-room CroVCA."""

from __future__ import annotations

import math
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from baseline.CroVCA import (
    OFFICIAL_COMMIT,
    CroVCA,
    CroVCAFeatureHash,
    CroVCAHashCoder,
    crovca_alignment_loss,
    crovca_coding_rate_loss,
    crovca_cosine_learning_rate,
    crovca_loss,
)


def test_small_hashcoder_matches_published_architecture_and_interface():
    head = CroVCAHashCoder(d_in=7, bit=36)
    assert [type(layer) for layer in head.net] == [
        nn.Linear, nn.ReLU, nn.Linear, nn.BatchNorm1d,
    ]
    assert head.net[0].in_features == 7
    assert head.net[0].out_features == 7
    assert head.net[2].in_features == 7
    assert head.net[2].out_features == 36
    torch.testing.assert_close(head.net[0].bias, torch.zeros(7))
    torch.testing.assert_close(head.net[2].bias, torch.zeros(36))

    model = CroVCAFeatureHash(d_in=7, bit=36).eval()
    features = torch.randn(5, 7)
    output = model(features)
    assert set(output) == {
        "continuous_code", "cnn_feat", "backbone_last_output",
    }
    assert output["continuous_code"].shape == (5, 36)
    torch.testing.assert_close(output["cnn_feat"], features)
    assert model.encoder_layers is not None


def test_alignment_matches_symmetric_stop_gradient_bce_exactly():
    first = torch.tensor(
        [[2.0, -0.5, 0.1], [-1.0, 3.0, -2.0]],
        dtype=torch.float64, requires_grad=True,
    )
    second = torch.tensor(
        [[1.0, -2.0, -0.2], [0.4, 2.0, -3.0]],
        dtype=torch.float64, requires_grad=True,
    )
    actual = crovca_alignment_loss(first, second)
    target_first = (first.detach() > 0).to(first.dtype)
    target_second = (second.detach() > 0).to(second.dtype)
    expected = 0.5 * (
        F.binary_cross_entropy_with_logits(first, target_second)
        + F.binary_cross_entropy_with_logits(second, target_first)
    )
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    actual.backward()
    scale = 0.5 / first.numel()
    expected_first_grad = scale * (first.detach().sigmoid() - target_second)
    expected_second_grad = scale * (second.detach().sigmoid() - target_first)
    torch.testing.assert_close(first.grad, expected_first_grad,
                               rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(second.grad, expected_second_grad,
                               rtol=1e-12, atol=1e-12)


def test_alignment_rewards_agreeing_confident_codes():
    logits = torch.tensor([[5.0, -5.0], [-5.0, 5.0]])
    agreeing = crovca_alignment_loss(logits, logits.clone())
    disagreeing = crovca_alignment_loss(logits, -logits)
    assert agreeing < 0.01
    assert disagreeing > 4.0


def _coding_rate_slogdet_oracle(logits: torch.Tensor, eps: float) -> torch.Tensor:
    normalized = F.normalize(logits, dim=1)
    batch_size, bit = logits.shape
    matrix = (torch.eye(bit, dtype=logits.dtype)
              + bit / (batch_size * eps) * normalized.T @ normalized)
    sign, logabsdet = torch.linalg.slogdet(matrix)
    assert sign > 0
    return -0.5 * logabsdet * ((bit + batch_size) / (bit * batch_size))


def test_coding_rate_matches_algorithm_one_and_discourages_rank_collapse():
    logits = torch.tensor(
        [[0.2, -0.7, 1.1], [1.2, 0.4, -0.3], [-0.9, 0.8, 0.2],
         [0.5, 1.4, -0.6]],
        dtype=torch.float64, requires_grad=True,
    )
    actual = crovca_coding_rate_loss(logits, eps=0.05)
    expected = _coding_rate_slogdet_oracle(logits, eps=0.05)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    scaled = crovca_coding_rate_loss(logits.detach() * 7.3, eps=0.05)
    torch.testing.assert_close(actual.detach(), scaled, rtol=1e-12, atol=1e-12)
    actual.backward()
    assert logits.grad is not None
    assert bool(torch.isfinite(logits.grad).all())

    diverse = torch.eye(4, dtype=torch.float64)
    collapsed = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 4,
                             dtype=torch.float64)
    # The loss is the *negative* coding rate, hence more diverse is lower.
    assert (crovca_coding_rate_loss(diverse)
            < crovca_coding_rate_loss(collapsed))


def test_complete_objective_has_paper_weights_and_finite_gradients():
    torch.manual_seed(9)
    first = torch.randn(8, 36, requires_grad=True)
    second = torch.randn(8, 36, requires_grad=True)
    total, parts = crovca_loss(
        first, second, eps=0.05,
        alignment_weight=1.0, diversity_weight=0.1,
    )
    assert set(parts) == {"alignment", "diversity"}
    torch.testing.assert_close(
        total, parts["alignment"] + 0.1 * parts["diversity"])
    total.backward()
    assert first.grad is not None and second.grad is not None
    assert bool(torch.isfinite(first.grad).all())
    assert bool(torch.isfinite(second.grad).all())


def test_iteration_cosine_schedule_matches_public_formula():
    base, final, total = 1e-3, 1e-6, 20
    rates = [crovca_cosine_learning_rate(i, total, base, final)
             for i in range(total)]
    assert math.isclose(rates[0], base, rel_tol=0.0, abs_tol=1e-15)
    assert all(first >= second for first, second in zip(rates, rates[1:]))
    expected_last = final + 0.5 * (base - final) * (
        1.0 + math.cos(math.pi * (total - 1) / total))
    assert math.isclose(rates[-1], expected_last, rel_tol=0.0, abs_tol=1e-15)
    # The release divides by total_steps, so the final *visited* value is
    # slightly above min_lr.  This catches a common CosineAnnealingLR rewrite.
    assert rates[-1] > final
    with unittest.TestCase().assertRaises(ValueError):
        crovca_cosine_learning_rate(total, total, base, final)


def test_runner_is_explicitly_target_free_matched_cache_not_lora_claim():
    runner = CroVCA()
    defaults = runner._get_default_config_dict()
    fixed = runner._get_fixed_config_dict()
    assert defaults == {
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
    assert fixed["dataset_return_paired_aug_img"] is True
    assert fixed["dataset_return_visual_tokens"] is False
    assert fixed["dataset_return_index"] is False
    assert fixed["finetune"] is False
    assert "target-label-free" in fixed["information_condition"]
    assert "matched-cache-probing" in fixed["implementation_variant"]
    assert "LoRA" in fixed["official_result_difference"]
    assert fixed["source_code_commit"] == OFFICIAL_COMMIT
    assert "label" not in fixed


def test_runner_builds_adamw_and_rejects_optimizer_drift():
    runner = CroVCA()
    model = CroVCAFeatureHash(5, 36)
    optimizer = runner._init_optimizer(
        model, name="adamw", learning_rate=1e-3,
        adam_weight_decay=1e-2,
    )
    assert isinstance(optimizer, torch.optim.AdamW)
    assert math.isclose(
        optimizer.param_groups[0]["weight_decay"], 1e-2,
        rel_tol=0.0, abs_tol=1e-15)
    with unittest.TestCase().assertRaises(ValueError):
        runner._init_optimizer(model, name="adam", learning_rate=1e-3)


class CroVCABaselineTest(unittest.TestCase):
    def test_small_hashcoder(self):
        test_small_hashcoder_matches_published_architecture_and_interface()

    def test_alignment_equation_and_gradient(self):
        test_alignment_matches_symmetric_stop_gradient_bce_exactly()

    def test_alignment_behavior(self):
        test_alignment_rewards_agreeing_confident_codes()

    def test_coding_rate(self):
        test_coding_rate_matches_algorithm_one_and_discourages_rank_collapse()

    def test_complete_objective(self):
        test_complete_objective_has_paper_weights_and_finite_gradients()

    def test_schedule(self):
        test_iteration_cosine_schedule_matches_public_formula()

    def test_protocol_metadata(self):
        test_runner_is_explicitly_target_free_matched_cache_not_lora_claim()

    def test_optimizer(self):
        test_runner_builds_adamw_and_rejects_optimizer_drift()


if __name__ == "__main__":
    unittest.main()
