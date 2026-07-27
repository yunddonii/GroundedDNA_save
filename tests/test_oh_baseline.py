"""Equation, state, and protocol tests for the clean-room OH adapter."""

from __future__ import annotations

import copy
import math
import unittest

import torch
import torch.nn.functional as F

from baseline.OH import (
    OFFICIAL_COMMIT,
    OH,
    OHFeatureHash,
    hard_binary_ste,
    oh_loss,
)
from baseline.base_model import build_model_from_checkpoint_payload
from scripts.baseline_val_select_p0 import (
    _assert_compatible_full_config,
    _assert_same_stage1_config,
)


def _tiny_model(*, momentum: float = 0.75) -> OHFeatureHash:
    torch.manual_seed(17)
    return OHFeatureHash(
        d_in=3,
        bit=2,
        middle_dim=4,
        continuous_dim=3,
        momentum=momentum,
        temperature=0.2,
        queue_length=5,
        context_weight=0.5,
        shuffle_key_batch=False,
    )


def test_sigmoid_threshold_ste_has_exact_01_forward_and_soft_gradient():
    logits = torch.tensor([-2.0, 0.0, 2.0],
                          dtype=torch.float64, requires_grad=True)
    binary = hard_binary_ste(logits)
    torch.testing.assert_close(
        binary, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64))
    binary.sum().backward()
    probabilities = logits.detach().sigmoid()
    torch.testing.assert_close(
        logits.grad, probabilities * (1.0 - probabilities),
        rtol=1e-12, atol=1e-12)


def test_inference_converts_internal_01_to_signed_without_bit_collapse():
    model = _tiny_model().eval()
    for encoder in (model.encoder_q, model.encoder_k):
        for module in encoder.modules():
            if isinstance(module, torch.nn.Linear):
                torch.testing.assert_close(
                    module.bias, torch.zeros_like(module.bias))
                fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(
                    module.weight)
                bound = math.sqrt(6.0 / (fan_in + fan_out))
                assert float(module.weight.abs().max()) <= bound + 1e-7
    with torch.no_grad():
        model.encoder_q.trunk[0].weight.zero_()
        model.encoder_q.trunk[0].bias.fill_(1.0)
        model.encoder_q.binary_head.weight.zero_()
        model.encoder_q.binary_head.bias.copy_(torch.tensor([-1.0, 1.0]))
    output = model(torch.randn(4, 3))
    assert set(output) == {
        "continuous_code", "oh_binary_01", "oh_continuous_head",
        "cnn_feat", "backbone_last_output",
    }
    torch.testing.assert_close(
        output["oh_binary_01"],
        torch.tensor([[0.0, 1.0]]).expand(4, -1))
    torch.testing.assert_close(
        output["continuous_code"],
        torch.tensor([[-1.0, 1.0]]).expand(4, -1))
    assert set(output["continuous_code"].unique().tolist()) == {-1.0, 1.0}


def test_overview_matches_released_hamming_attention_equation():
    model = OHFeatureHash(
        d_in=2, bit=2, middle_dim=2, continuous_dim=2,
        momentum=0.9, temperature=0.25, queue_length=3,
        context_weight=0.5, shuffle_key_batch=False,
    )
    with torch.no_grad():
        # Deliberately include non-binary B values and non-normalized C values:
        # this is the release's initial/general queue contract.
        model.b_queue.copy_(torch.tensor([
            [1.0, 0.0], [0.0, 1.0], [0.2, -0.4],
        ]))
        model.c_queue.copy_(torch.tensor([
            [2.0, 0.0], [0.0, 4.0], [1.0, 1.0],
        ]))
    binary = torch.tensor([[1.0, 0.0]])
    continuous = F.normalize(torch.tensor([[1.0, 2.0]]), dim=1)
    actual = model.overview(binary, continuous)

    similarity = (
        binary @ model.b_queue.T
        + (1.0 - binary) @ (1.0 - model.b_queue).T
    )
    attention = torch.softmax(similarity / (2 * 0.25), dim=1)
    aggregate = F.normalize(attention @ model.c_queue, dim=1)
    expected = F.normalize(aggregate + 0.5 * continuous, dim=1)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


def test_ema_and_fifo_updates_match_the_public_operation_order():
    model = _tiny_model(momentum=0.75)
    with torch.no_grad():
        for parameter in model.encoder_q.parameters():
            parameter.fill_(2.0)
        for parameter in model.encoder_k.parameters():
            parameter.fill_(-3.0)
        model.momentum_update_key_encoder()
    for parameter in model.encoder_k.parameters():
        torch.testing.assert_close(
            parameter, torch.full_like(parameter, -1.75))

    old_p = model.p_queue.clone()
    old_b = model.b_queue.clone()
    old_c = model.c_queue.clone()
    new_p = torch.tensor([[9.0, 8.0, 7.0], [6.0, 5.0, 4.0]])
    new_b = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    new_c = torch.tensor([[3.0, 2.0, 1.0], [1.0, 2.0, 3.0]])
    model.update_queues(new_p, new_b, new_c)
    torch.testing.assert_close(
        model.p_queue, torch.cat((new_p, old_p[:-2]), dim=0))
    torch.testing.assert_close(
        model.b_queue, torch.cat((new_b, old_b[:-2]), dim=0))
    torch.testing.assert_close(
        model.c_queue, torch.cat((new_c, old_c[:-2]), dim=0))


def test_training_forward_uses_old_then_updated_queues_and_stops_q2_gradient():
    model = _tiny_model(momentum=0.6).train()
    before = copy.deepcopy(model)
    first = torch.tensor([[0.2, -0.3, 0.7], [1.1, 0.4, -0.5]])
    second = torch.tensor([[-0.6, 0.8, 0.2], [0.5, -0.9, 1.3]])

    # Query overview must use the pre-update queues.
    expected_bq, expected_cq = before.encoder_q(first)
    expected_q = before.overview(expected_bq, expected_cq)
    outputs = model.training_logits(first, second)
    torch.testing.assert_close(outputs["overview_query"], expected_q)

    # The key parameters are updated before key encoding.
    with torch.no_grad():
        before.momentum_update_key_encoder()
        expected_bk, expected_ck = before.encoder_k(second)
        expected_k = before.overview(expected_bk, expected_ck)
    torch.testing.assert_close(outputs["overview_key"], expected_k)
    for actual, expected in zip(
            model.encoder_k.parameters(), before.encoder_k.parameters()):
        torch.testing.assert_close(actual, expected)

    # Enqueue occurs before q2, so the stopped q2 is reproducible from the
    # now-updated queues, while the prepended rows are the current key batch.
    torch.testing.assert_close(model.p_queue[:2], expected_k)
    torch.testing.assert_close(model.b_queue[:2], expected_bk)
    torch.testing.assert_close(model.c_queue[:2], expected_ck)
    with torch.no_grad():
        second_b, second_c = model.encoder_q(second)
        expected_q2 = model.overview(second_b, second_c)
    torch.testing.assert_close(outputs["overview_second"], expected_q2)
    assert outputs["overview_second"].requires_grad is False

    loss, parts = oh_loss(outputs)
    torch.testing.assert_close(
        loss, parts["queue_contrastive"] + parts["batch_contrastive"])
    loss.backward()
    assert any(parameter.grad is not None
               for parameter in model.encoder_q.parameters())
    assert all(parameter.grad is None
               for parameter in model.encoder_k.parameters())
    assert all(buffer.grad is None
               for buffer in (model.p_queue, model.b_queue, model.c_queue))


def test_complete_checkpoint_restores_query_key_and_all_queues():
    model = _tiny_model().train()
    outputs = model.training_logits(
        torch.randn(2, 3), torch.randn(2, 3))
    loss, _ = oh_loss(outputs)
    optimizer = torch.optim.Adam(model.encoder_q.parameters(), lr=1e-5)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    config = {
        "method": "oh", "bit": 2, "resolved_projection_dim": 3,
        "oh_middle_dim": 4, "oh_continuous_dim": 3,
        "oh_momentum": 0.75, "oh_temperature": 0.2,
        "oh_queue_length": 5, "oh_context_weight": 0.5,
        "oh_shuffle_key_batch": False,
    }
    state = model.state_dict()
    assert "p_queue" in state and "b_queue" in state and "c_queue" in state
    assert any(name.startswith("encoder_q.") for name in state)
    assert any(name.startswith("encoder_k.") for name in state)
    assert not any(name.startswith("encoder_layers.") for name in state)

    restored = build_model_from_checkpoint_payload(
        {"config": config, "model_state_dict": state},
        d_in=3, device="cpu",
    )
    assert restored._checkpoint_load_format == "model_state_dict"
    for name, expected in state.items():
        torch.testing.assert_close(restored.state_dict()[name], expected)
    features = torch.randn(4, 3)
    model.eval()
    torch.testing.assert_close(
        restored(features)["continuous_code"],
        model(features)["continuous_code"],
    )


def test_runner_records_u0_adapter_boundary_and_optimizes_query_only():
    runner = OH()
    defaults = runner._get_default_config_dict()
    fixed = runner._get_fixed_config_dict()
    assert defaults == {
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
    assert fixed["dataset_return_paired_aug_img"] is True
    assert fixed["dataset_return_visual_tokens"] is False
    assert "U0" in fixed["information_condition"]
    assert "matched-cache-adapter" in fixed["implementation_variant"]
    assert fixed["source_code_commit"] == OFFICIAL_COMMIT
    assert fixed["internal_hash_alphabet"] == "hard_0_1_sigmoid_STE"
    assert fixed["extractor_hash_alphabet"] == "signed_2b_minus_1"
    assert "label" not in fixed

    model = _tiny_model()
    optimizer = runner._init_optimizer(
        model, name="adam", learning_rate=1e-5,
        adam_weight_decay=0.0,
    )
    optimized = {
        id(parameter)
        for group in optimizer.param_groups for parameter in group["params"]
    }
    assert optimized == {id(parameter)
                         for parameter in model.encoder_q.parameters()}
    assert optimized.isdisjoint(
        {id(parameter) for parameter in model.encoder_k.parameters()})
    assert math.isclose(optimizer.param_groups[0]["weight_decay"], 0.0)
    assert math.isclose(optimizer.param_groups[0]["eps"], 1e-7,
                        rel_tol=0.0, abs_tol=1e-15)
    with unittest.TestCase().assertRaises(ValueError):
        runner._init_optimizer(
            model, name="adamw", learning_rate=1e-5,
            adam_weight_decay=0.0)


def test_selector_binds_all_method_defining_oh_hyperparameters():
    config = {
        "method": "oh", "dataset": "Flickr25k", "setting": "setting1",
        "bit": 36, "trial_name": "oh-stage1", "seed": 1,
        "batch_size": 50, "learning_rate": 1e-5,
        "optimizer_name": "adam", "adam_weight_decay": 0.0,
        "sgd_weight_decay": 1e-5, "sgd_momentum": 0.99,
        "lr_scheduler": "none", "gen_code_method": "sign",
        "finetune": False, "dataset_return_paired_aug_img": True,
        "dataset_return_visual_tokens": False,
        "val_split_ratio": 0.1, "val_split_seed": 42,
        "max_epoch": 200, "eval_period": 5, "schedule_horizon": 200,
        "resolved_backbone": "clip", "resolved_projection_dim": 512,
        "resolved_cache_dir": "/tmp/cache",
        "resolved_cache_artifact_sha256": {
            "visual_global.f16.npy": "b" * 64},
        "cache_artifact_binding_status": "verified_against_driver",
        "protocol_identity_sha256": "a" * 64,
        "implementation_variant": "official-release-core-matched-cache-adapter",
        "information_tier": "U0", "source_code_commit": OFFICIAL_COMMIT,
        "internal_hash_alphabet": "hard_0_1_sigmoid_STE",
        "extractor_hash_alphabet": "signed_2b_minus_1",
        "numeric_precision": "float32",
        "oh_middle_dim": 2048, "oh_continuous_dim": 1024,
        "oh_momentum": 0.999, "oh_temperature": 0.2,
        "oh_queue_length": 4096, "oh_context_weight": 0.5,
        "oh_shuffle_key_batch": True, "oh_adam_epsilon": 1e-7,
    }
    drifted = {**config, "oh_queue_length": 1024}
    with unittest.TestCase().assertRaisesRegex(ValueError, "config drift"):
        _assert_same_stage1_config(config, drifted, "epoch_005.pth")
    refit = {**config, "trial_name": "oh-refit", "oh_momentum": 0.9}
    with unittest.TestCase().assertRaisesRegex(ValueError, "incompatible"):
        _assert_compatible_full_config(config, refit, "epoch_005.pth")


class OHBaselineTest(unittest.TestCase):
    def test_ste(self):
        test_sigmoid_threshold_ste_has_exact_01_forward_and_soft_gradient()

    def test_signed_inference(self):
        test_inference_converts_internal_01_to_signed_without_bit_collapse()

    def test_overview(self):
        test_overview_matches_released_hamming_attention_equation()

    def test_ema_fifo(self):
        test_ema_and_fifo_updates_match_the_public_operation_order()

    def test_training_order(self):
        test_training_forward_uses_old_then_updated_queues_and_stops_q2_gradient()

    def test_checkpoint(self):
        test_complete_checkpoint_restores_query_key_and_all_queues()

    def test_protocol(self):
        test_runner_records_u0_adapter_boundary_and_optimizes_query_only()

    def test_selector_binding(self):
        test_selector_binds_all_method_defining_oh_hyperparameters()


if __name__ == "__main__":
    unittest.main()
