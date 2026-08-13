"""D6: baselines train to their author-prescribed horizons, so the schedule
that runs over those horizons has to be the author's schedule too.

Each expectation below was read from the upstream source, not from memory:

  Bi-half   liyunqianggyn/Deep-Unsupervised-Image-Hashing
            ImageHashing/Cifar10_I.py:14-16   num_epochs=300, epoch_lr_decrease=120
            ImageHashing/Flickr25k.py:11-13   num_epochs=100, epoch_lr_decrease=60
            ImageHashing/Mscoco.py:16-18      num_epochs=150, epoch_lr_decrease=60
            (no NUS-WIDE trainer upstream -> Flickr profile, declared adaptation)
  SDC       kamwoh/sdc configs/train.yaml     batch 64, epochs 100
            configs/scheduler/step.yaml       step_size = int(0.8 * epochs), gamma 0.1
            configs/optim/adam.yaml           lr 1e-4, weight_decay 1e-5
  MLS3RDUH  rongchengtu1/MLS3RDUH/MLS3RDUH.py batch 128, epochs 150, lr 0.04,
            weight_decay 1e-5; AdjustLearningRate is defined but never called and
            StepLR(step_size=500) never fires in 150 epochs -> no decay
  CIMON     luoxiao12/CIMON run.py            max-iter 150, lr 1e-3, batch 24,
            SGD(momentum 0.9, weight_decay 1e-5), no scheduler
  GreedyHash ssppp/GreedyHash unsupervised_vgg.py:15-23  60 epochs, batch 32,
            lr 1e-4, SGD(0.9, 5e-4), adjust_learning_rate commented out

The defect this file pins: `BiHalf._get_config_dict_for_dataset` discarded the
dataset and returned one config, so `step_size` was 60 everywhere. At the
author's 300-epoch CIFAR-10 horizon that decays at 60/120/180/240 instead of
120/240, ending at LR 1e-8 instead of 1e-6 -- a 100x difference in the final
learning rate, which under "train to the horizon and keep the last checkpoint"
IS the reported number.
"""
from __future__ import annotations

import os
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from baseline.BiHalf import BiHalf  # noqa: E402
from baseline.SDC import SDC  # noqa: E402


def _config(method, dataset):
    base = method()._get_default_config_dict()
    return method()._get_config_dict_for_dataset(dict(base), dataset)


# ------------------------------------------------------------------ Bi-half

@pytest.mark.parametrize("dataset,step_size", [
    ("CIFAR10", 120),      # Cifar10_I.py: epoch_lr_decrease = 120
    ("Flickr25k", 60),     # Flickr25k.py: epoch_lr_decrease = 60
    ("MSCOCO", 60),        # Mscoco.py:    epoch_lr_decrease = 60
    ("NUSWIDE", 60),       # no upstream trainer; declared Flickr adaptation
])
def test_bihalf_step_size_is_per_dataset(dataset, step_size):
    assert _config(BiHalf, dataset)["step_size"] == step_size


@pytest.mark.parametrize("dataset,max_epoch", [
    ("CIFAR10", 300), ("Flickr25k", 100), ("MSCOCO", 150), ("NUSWIDE", 100),
])
def test_bihalf_horizon_matches_the_author_script(dataset, max_epoch):
    assert _config(BiHalf, dataset)["max_epoch"] == max_epoch


def test_bihalf_cifar_decays_twice_not_four_times():
    """The defect, stated as the quantity that changed: with step_size 60 the
    300-epoch CIFAR run decays four times and ends at 1e-8; the author's 120
    decays twice and ends at 1e-6."""
    cfg = _config(BiHalf, "CIFAR10")
    decays = (cfg["max_epoch"] - 1) // cfg["step_size"]
    assert decays == 2
    final_lr = cfg["learning_rate"] * cfg["step_gamma"] ** decays
    assert final_lr == pytest.approx(1e-6)
    wrong = cfg["learning_rate"] * cfg["step_gamma"] ** ((300 - 1) // 60)
    assert wrong == pytest.approx(1e-8)


def test_bihalf_optimizer_matches_upstream():
    cfg = BiHalf()._get_default_config_dict()
    assert cfg["optimizer_name"] == "sgd"
    assert cfg["learning_rate"] == pytest.approx(1e-4)
    assert cfg["sgd_momentum"] == pytest.approx(0.9)
    assert cfg["sgd_weight_decay"] == pytest.approx(5e-4)
    assert cfg["batch_size"] == 32
    assert cfg["step_gamma"] == pytest.approx(0.1)


# ---------------------------------------------------------------------- SDC

def test_sdc_step_size_is_derived_from_the_horizon():
    """`configs/scheduler/step.yaml` defines step_size as int(0.8 * epochs),
    so a hard-coded 80 silently stops matching if the horizon ever moves."""
    for horizon, expected in ((100, 80), (150, 120), (50, 40)):
        cfg = SDC()._get_default_config_dict()
        cfg["max_epoch"] = horizon
        cfg = SDC()._get_config_dict_for_dataset(cfg, "Flickr25k")
        assert cfg["step_size"] == expected


def test_sdc_defaults_match_upstream_configs():
    cfg = SDC()._get_default_config_dict()
    assert (cfg["batch_size"], cfg["max_epoch"]) == (64, 100)
    assert cfg["optimizer_name"] == "adam"
    assert cfg["learning_rate"] == pytest.approx(1e-4)
    assert cfg["adam_weight_decay"] == pytest.approx(1e-5)
    assert cfg["step_gamma"] == pytest.approx(0.1)
    # 0.8 * 100 = 80, the value the upstream expression yields at this horizon.
    assert _config(SDC, "Flickr25k")["step_size"] == 80
