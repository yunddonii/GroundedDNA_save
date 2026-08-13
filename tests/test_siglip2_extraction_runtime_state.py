"""F01: a reloaded checkpoint must run inference at the epoch it was trained to.

`_current_epoch` is a plain Python int on the model, not a buffer, so it is
absent from the state dict. `extraction_siglip2.extract_code` builds a fresh
model and loads raw weights, which leaves it at 0. Every standalone extraction
therefore ran the Sinkhorn router at the INITIAL epsilon -- 1.0 on
CIFAR/Flickr/MS-COCO, 0.5 on NUS-WIDE -- instead of the annealed value the
weights were trained with. Measured on the CIFAR final checkpoint, 24 of the
first 32 rows change `base_indices` between epoch 0 and epoch 4.

The failure is silent: nothing raises, the codes just come from a router the
training never used. So the contract is fail-closed. If epsilon annealing is
active and the epoch cannot be established from an explicit flag or from
checkpoint metadata, extraction must ABORT rather than quietly use 0.

D2 (2026-08-13) additionally separates three horizons that a single `-e` used to
control at once: the training stop, the LR cosine horizon (fixed at 60) and the
Sinkhorn horizon (N+1). These tests pin that they are independent, so that
choosing N no longer silently retunes the learning-rate schedule.
"""
from __future__ import annotations

import json
import os
import sys
import types

import pytest
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from dna_utils.runtime_state import (  # noqa: E402
    CheckpointMetadata,
    InferenceEpochUnresolved,
    resolve_horizons,
    resolve_inference_epoch,
    write_checkpoint_metadata,
)


def _args(**kw):
    a = types.SimpleNamespace(
        epoch=60, stop_after_epoch=None,
        lr_schedule_horizon=None, sinkhorn_schedule_horizon=None,
        sinkhorn_epsilon_init=1.0, sinkhorn_epsilon_final=0.1,
        inference_epoch=None,
    )
    for k, v in kw.items():
        setattr(a, k, v)
    return a


# --------------------------------------------------------------- horizons (D2)

def test_horizons_default_to_the_nominal_epoch_budget():
    """Unset horizons must reproduce the historical behaviour exactly, so that
    re-running an old config does not silently change the model."""
    h = resolve_horizons(_args(epoch=60))
    assert h.training_stop_epoch == 59
    assert h.lr_schedule_horizon == 60
    assert h.sinkhorn_schedule_horizon == 60


def test_stop_after_epoch_shortens_training_but_not_the_lr_horizon():
    """The decided protocol: stop at N, keep the LR cosine on its 60-epoch
    prefix, and let epsilon complete over N+1."""
    h = resolve_horizons(_args(epoch=60, stop_after_epoch=4,
                               lr_schedule_horizon=60,
                               sinkhorn_schedule_horizon=5))
    assert h.training_stop_epoch == 4
    assert h.lr_schedule_horizon == 60       # NOT 5
    assert h.sinkhorn_schedule_horizon == 5


def test_horizons_are_independent():
    h = resolve_horizons(_args(epoch=40, stop_after_epoch=9,
                               lr_schedule_horizon=60,
                               sinkhorn_schedule_horizon=10))
    assert (h.training_stop_epoch, h.lr_schedule_horizon,
            h.sinkhorn_schedule_horizon) == (9, 60, 10)


# ------------------------------------------------------- epoch resolution (F01)

def test_metadata_round_trip(tmp_path):
    ck = tmp_path / "model_state_dict.pth"
    torch.save({"w": torch.zeros(1)}, ck)
    write_checkpoint_metadata(
        str(ck), checkpoint_epoch_zero_based=4, training_epoch_budget=60,
        stop_after_epoch=4, lr_schedule_horizon=60, sinkhorn_schedule_horizon=5,
        sinkhorn_epsilon_init=1.0, sinkhorn_epsilon_final=0.1,
        lr_scheduler="cosine", extra={"protocol": "fixedN"},
    )
    md = CheckpointMetadata.load(str(ck))
    assert md is not None
    assert md.checkpoint_epoch_zero_based == 4
    assert md.sinkhorn_schedule_horizon == 5
    assert md.checkpoint_sha256 and len(md.checkpoint_sha256) == 64
    assert md.effective_sinkhorn_epsilon == pytest.approx(0.1, abs=1e-9)


def test_resolved_epoch_comes_from_metadata(tmp_path):
    ck = tmp_path / "model_state_dict.pth"
    torch.save({"w": torch.zeros(1)}, ck)
    write_checkpoint_metadata(
        str(ck), checkpoint_epoch_zero_based=4, training_epoch_budget=60,
        stop_after_epoch=4, lr_schedule_horizon=60, sinkhorn_schedule_horizon=5,
        sinkhorn_epsilon_init=1.0, sinkhorn_epsilon_final=0.1)
    got = resolve_inference_epoch(str(ck), _args(epoch=60, stop_after_epoch=4))
    assert got.epoch == 4
    assert got.source == "checkpoint_metadata"


def test_annealed_checkpoint_without_metadata_fails_closed(tmp_path):
    """The whole point of F01: never fall back to epoch 0."""
    ck = tmp_path / "model_state_dict.pth"
    torch.save({"w": torch.zeros(1)}, ck)
    with pytest.raises(InferenceEpochUnresolved):
        resolve_inference_epoch(str(ck), _args(epoch=60, stop_after_epoch=None))


def test_explicit_flag_wins_but_must_agree_with_metadata(tmp_path):
    ck = tmp_path / "model_state_dict.pth"
    torch.save({"w": torch.zeros(1)}, ck)
    write_checkpoint_metadata(
        str(ck), checkpoint_epoch_zero_based=4, training_epoch_budget=60,
        stop_after_epoch=4, lr_schedule_horizon=60, sinkhorn_schedule_horizon=5,
        sinkhorn_epsilon_init=1.0, sinkhorn_epsilon_final=0.1)
    ok = resolve_inference_epoch(str(ck), _args(inference_epoch=4))
    assert ok.epoch == 4 and ok.source == "explicit_flag"
    with pytest.raises(InferenceEpochUnresolved):
        resolve_inference_epoch(str(ck), _args(inference_epoch=9))


def test_no_annealing_configured_does_not_require_metadata(tmp_path):
    """With epsilon annealing off the router uses its static epsilon, so an
    unknown epoch is harmless and must not block extraction."""
    ck = tmp_path / "model_state_dict.pth"
    torch.save({"w": torch.zeros(1)}, ck)
    got = resolve_inference_epoch(
        str(ck), _args(sinkhorn_epsilon_init=None, sinkhorn_epsilon_final=None))
    assert got.source == "no_annealing"


def test_stale_metadata_is_rejected(tmp_path):
    """A sidecar left over from a previous run in the same directory must not
    be trusted; the collision incident of 2026-08-12 made that concrete."""
    ck = tmp_path / "model_state_dict.pth"
    torch.save({"w": torch.zeros(1)}, ck)
    write_checkpoint_metadata(
        str(ck), checkpoint_epoch_zero_based=4, training_epoch_budget=60,
        stop_after_epoch=4, lr_schedule_horizon=60, sinkhorn_schedule_horizon=5,
        sinkhorn_epsilon_init=1.0, sinkhorn_epsilon_final=0.1)
    torch.save({"w": torch.ones(1)}, ck)          # checkpoint changed, sidecar did not
    with pytest.raises(InferenceEpochUnresolved):
        resolve_inference_epoch(str(ck), _args(epoch=60, stop_after_epoch=4))


# ------------------------------------------------- the model actually sees it

class _EpochSensitiveModel(torch.nn.Module):
    """Minimal stand-in with the same failure mode: an epoch held as a plain
    int, so it cannot ride along in the state dict."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self._current_epoch = 0

    def set_current_epoch(self, e):
        self._current_epoch = int(e)

    def forward(self):
        return self._current_epoch


def test_reloaded_model_observes_the_saved_epoch(tmp_path):
    from dna_utils.runtime_state import apply_inference_epoch
    ck = tmp_path / "model_state_dict.pth"
    m = _EpochSensitiveModel()
    m.set_current_epoch(4)
    torch.save(m.state_dict(), ck)
    write_checkpoint_metadata(
        str(ck), checkpoint_epoch_zero_based=4, training_epoch_budget=60,
        stop_after_epoch=4, lr_schedule_horizon=60, sinkhorn_schedule_horizon=5,
        sinkhorn_epsilon_init=1.0, sinkhorn_epsilon_final=0.1)

    m2 = _EpochSensitiveModel()
    m2.load_state_dict(torch.load(ck, weights_only=False))
    assert m2() == 0, "precondition: the raw reload is what loses the epoch"

    resolved = apply_inference_epoch(m2, str(ck), _args(epoch=60, stop_after_epoch=4))
    assert m2() == 4
    assert resolved.epoch == 4


def test_extraction_manifest_records_epoch_and_epsilon(tmp_path):
    from dna_utils.runtime_state import write_extraction_manifest
    ck = tmp_path / "model_state_dict.pth"
    torch.save({"w": torch.zeros(1)}, ck)
    write_checkpoint_metadata(
        str(ck), checkpoint_epoch_zero_based=4, training_epoch_budget=60,
        stop_after_epoch=4, lr_schedule_horizon=60, sinkhorn_schedule_horizon=5,
        sinkhorn_epsilon_init=1.0, sinkhorn_epsilon_final=0.1)
    resolved = resolve_inference_epoch(str(ck), _args(epoch=60, stop_after_epoch=4))
    out = tmp_path / "extraction_manifest.json"
    write_extraction_manifest(
        str(out), checkpoint_path=str(ck), resolved=resolved,
        num_slots=5, bases_per_slot=3, split="db", n_rows=100)
    d = json.load(open(out))
    assert d["inference_epoch"] == 4
    assert d["effective_sinkhorn_epsilon"] == pytest.approx(0.1, abs=1e-9)
    assert d["checkpoint_sha256"] == CheckpointMetadata.load(str(ck)).checkpoint_sha256
    assert d["num_slots"] == 5 and d["bases_per_slot"] == 3
    assert d["total_bases"] == 15 and d["total_bits"] == 30
