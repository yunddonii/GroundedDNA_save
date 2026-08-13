"""The standalone extraction entry point must not discard its CLI flags.

`extraction_siglip2.main` builds `args = Config()`, which is the raw object with
NO argv parsing -- the parsed values live on `Config.get_config()`. Restoring the
saved `config.pt` then overwrote `args` wholesale, so every flag except
`--config_path` was dropped, including `--inference_epoch`.

That flag is F01's PRIMARY way to resolve the extraction epoch (explicit flag ->
metadata sidecar -> fail). Legacy checkpoints have no sidecar, so the documented
escape hatch was the only route available for them, and it could not be used at
all: the resolver failed closed with "no --inference_epoch and no metadata
sidecar" even when `--inference_epoch 4` was on the command line.

Fail-closed meant no wrong numbers were produced. It also meant Phase 2's
diagnostic re-inference of the 15 seed-42 candidates could not run.
"""
from __future__ import annotations

import os
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from extraction_siglip2 import (  # noqa: E402
    _INFERENCE_TIME_OVERRIDES,
    _reapply_explicit_cli,
)


class _Args:
    """Stands in for the restored Config: whatever config.pt happened to hold."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_inference_epoch_survives_the_config_restore():
    args = _Args(inference_epoch=None)
    _reapply_explicit_cli(args, _Args(inference_epoch=4))
    assert args.inference_epoch == 4


def test_absent_flag_does_not_overwrite_the_restored_value():
    """A flag the user did not pass must not blank out the saved config."""
    args = _Args(inference_epoch=9)
    _reapply_explicit_cli(args, _Args(inference_epoch=None))
    assert args.inference_epoch == 9


def test_device_is_not_clobbered():
    """`device` is derived from `num_devices` during the restore; re-applying a
    stale CLI value here would undo that."""
    args = _Args(device="cuda:3")
    _reapply_explicit_cli(args, _Args(device="cuda:0"))
    assert args.device == "cuda:3"


def test_selection_mode_is_carried_too():
    args = _Args(selection_mode=None)
    _reapply_explicit_cli(args, _Args(selection_mode="train_only"))
    assert args.selection_mode == "train_only"


def test_inference_epoch_is_in_the_override_set():
    """If this drops out, F01's escape hatch silently stops working again."""
    assert "inference_epoch" in _INFERENCE_TIME_OVERRIDES


def test_unknown_attributes_are_left_alone():
    args = _Args(epoch=40, random_seed=42)
    _reapply_explicit_cli(args, _Args(epoch=1, random_seed=7))
    assert (args.epoch, args.random_seed) == (40, 42)
