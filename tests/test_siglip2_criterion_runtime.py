from types import SimpleNamespace

import pytest
import torch

from loss_siglip2 import DNACodonHashLoss
from train_siglip2 import (
    _raise_mid_eval_if_campaign,
    _set_epoch_training_mode,
)


def _criterion():
    return DNACodonHashLoss(SimpleNamespace(
        num_codebooks=5, anchor_ema_momentum=0.5))


def test_validation_never_initialises_or_updates_persistent_text_anchor():
    criterion = _criterion()
    first = torch.nn.functional.normalize(torch.randn(5, 7), dim=-1)
    second = torch.nn.functional.normalize(torch.randn(5, 7), dim=-1)

    criterion.train()
    criterion._update_ema_text_anchor(first)
    sealed = criterion.ema_text_anchor.clone()
    criterion.eval()
    target = criterion._update_ema_text_anchor(second)

    assert torch.equal(criterion.ema_text_anchor, sealed)
    assert target.data_ptr() == criterion.ema_text_anchor.data_ptr()

    fresh = _criterion().eval()
    ephemeral = fresh._update_ema_text_anchor(second)
    assert fresh.ema_text_anchor.numel() == 0
    assert torch.equal(ephemeral, second)


def test_epoch_mode_switches_model_and_stateful_criterion_together():
    model, criterion = torch.nn.Linear(2, 2), _criterion()
    _set_epoch_training_mode(model, criterion, False)
    assert not model.training and not criterion.training
    _set_epoch_training_mode(model, criterion, True)
    assert model.training and criterion.training


def test_phase3_mid_eval_error_is_fatal_but_legacy_policy_is_unchanged():
    error = RuntimeError("synthetic mid-eval failure")
    _raise_mid_eval_if_campaign(SimpleNamespace(), error)
    with pytest.raises(RuntimeError) as caught:
        _raise_mid_eval_if_campaign(
            SimpleNamespace(_phase3_campaign_binding={"cell_id": "x"}),
            error,
        )
    assert caught.value is error
