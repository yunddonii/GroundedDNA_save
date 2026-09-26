"""Anchor confirmation v1: the axis-centring port onto the approved P3 source generation.

The generation starts from commit 88c3a25 (the approved P3 model/trainer bytes of 5304005 plus the
lambda launcher extension) and adds ONE recipe axis, `--axis_center {none, anchors}`. These tests
pin three things without loading CLIP or any real artifact:

1. the centring arithmetic (the six controls the audit applied to the extracted method);
2. the flag: default `none`, exactly two choices;
3. the port is additive and gated: relative to the historical commit, config.py and
   model_siglip2.py only gain lines, and the only call of the centring is under
   `self.axis_center == "anchors"` -- so `none` executes the historical routing path.
"""
from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

HISTORICAL_BASE = "88c3a25b1b309550eafc276c2ce5be7575507173"


def centre(x, n_global=1, mask=None):
    from model_siglip2 import SigLIP2SemanticOTModel
    return SigLIP2SemanticOTModel._axis_center_local(x, n_global, mask)


def anchors(b=3, m=5, d=6, seed=0):
    return torch.randn(b, m, d, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


# ---- 1. arithmetic -----------------------------------------------------------------------------
def test_global_prefix_is_untouched():
    x = anchors()
    assert torch.equal(centre(x, 1)[:, :1], x[:, :1])


def test_local_columns_have_zero_mean():
    y = centre(anchors(), 1)
    assert torch.allclose(y[:, 1:].mean(dim=1), torch.zeros(3, 6, dtype=torch.float64), atol=1e-12)


def test_a_shared_local_translation_is_removed():
    x = anchors()
    shifted = x.clone()
    shifted[:, 1:] += torch.randn(3, 1, 6, dtype=torch.float64)       # same vector added to every local anchor
    assert torch.allclose(centre(shifted, 1), centre(x, 1) + torch.cat(
        [shifted[:, :1] - x[:, :1], torch.zeros(3, 4, 6, dtype=torch.float64)], dim=1), atol=1e-12)


def test_masked_mean_uses_only_active_local_columns():
    x = anchors()
    mask = torch.tensor([[1, 1, 1, 0, 0]] * 3)
    y = centre(x, 1, mask)
    want = x[:, 1:] - x[:, 1:3].mean(dim=1, keepdim=True)
    assert torch.allclose(y[:, 1:], want, atol=1e-12)


def test_identical_local_anchors_collapse_to_zero():
    x = anchors()
    x[:, 1:] = x[:, 1:2]
    assert torch.allclose(centre(x, 1)[:, 1:], torch.zeros(3, 4, 6, dtype=torch.float64), atol=1e-12)


def test_gradients_are_finite():
    x = anchors().requires_grad_(True)
    centre(x, 1).pow(2).sum().backward()
    assert torch.isfinite(x.grad).all()


def test_without_a_global_column_every_column_is_centred():
    y = centre(anchors(m=4), 0)
    assert torch.allclose(y.mean(dim=1), torch.zeros(3, 6, dtype=torch.float64), atol=1e-12)


# ---- 2. the flag -------------------------------------------------------------------------------
def parser():
    from config import Config
    return Config.build_parser()


def test_axis_center_defaults_to_none_with_exactly_two_choices():
    action = next(a for a in parser()._actions if a.dest == "axis_center")
    assert action.default == "none" and tuple(action.choices) == ("none", "anchors")


@pytest.mark.parametrize("value", ["readout", "both", "Anchors", ""])
def test_exploratory_centring_modes_are_refused_by_the_parser(value):
    with pytest.raises(SystemExit):
        parser().parse_args(["--axis_center", value])


# ---- 3. additive, gated port -------------------------------------------------------------------
def _diff(path):
    return subprocess.run(["git", "diff", HISTORICAL_BASE, "--", path], cwd=REPO,
                          capture_output=True, text=True, check=True).stdout


@pytest.mark.parametrize("path", ["config.py", "model_siglip2.py"])
def test_port_only_adds_lines_to_the_historical_source(path):
    removed = [line for line in _diff(path).splitlines()
               if line.startswith("-") and not line.startswith("---")]
    assert removed == [], removed


def test_the_only_centring_call_is_gated_on_anchors():
    tree = ast.parse((REPO / "model_siglip2.py").read_text())
    calls, gated = [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            test = ast.unparse(node.test)
            for inner in ast.walk(node):
                if isinstance(inner, ast.Call) and ast.unparse(inner.func).endswith("_axis_center_local"):
                    gated.append((test, inner.lineno))
        if isinstance(node, ast.Call) and ast.unparse(node.func).endswith("_axis_center_local"):
            calls.append(node.lineno)
    assert len(calls) == 1
    assert [line for _, line in gated] == calls
    assert gated[0][0].startswith("self.axis_center == 'anchors'")


def test_no_other_historical_line_mentions_the_new_axis():
    added = {line[1:].strip() for line in _diff("model_siglip2.py").splitlines()
             if line.startswith("+") and not line.startswith("+++")}
    mentions = [i for i, line in enumerate((REPO / "model_siglip2.py").read_text().splitlines(), 1)
                if "axis_center" in line and line.strip() not in added]
    assert mentions == []
