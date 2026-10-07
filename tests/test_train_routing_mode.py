"""(2026-10-07) --train_routing_mode {text,codebook_mean}.

CPU-only tests that drive the real parser, the real SigLIP2SemanticOTModel
(fake CLIP backbone shell, cached-feature path, train mode) and the real
trainer startup validator. No dataset, cache or GPU is touched.

What is pinned:
  (1) parser default is "text".
  (2) default flag is bit-identical: a model built WITHOUT the attribute and a
      model built with train_routing_mode="text" give torch.equal outputs.
  (3) "codebook_mean" keeps the text path (text_part_tokens, text-side
      quantisation) and routes exactly as the no-text (deployment) forward.
  (4) the validator refuses the text-only routing features and
      --disable_text_supervision, and passes on defaults.
"""
from __future__ import annotations

import ast
import copy
import inspect
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch import nn

import model_siglip2 as model_module
import train_siglip2
from config import Config
from model_siglip2 import (
    NUM_SEMANTIC_PARTS,
    SigLIP2SemanticOTModel,
    train_routing_mode_conflicts,
)
from train_siglip2 import (
    _validate_path_consistency_args,
    _validate_train_routing_mode_args,
)

_D_PROJ = 4       # fake CLIP projection dim (cached text_part_raw last dim)
_H_VIS = 6        # fake CLIP vision hidden dim (cached visual_tokens_raw last dim)
_B = 4
_N = 3


class _FakeClip(nn.Module):
    """Backbone shell sufficient for constructing the hashing model offline
    (same shape contract as tests/test_ablation_integrity.py)."""

    def __init__(self) -> None:
        super().__init__()
        self.vision_hidden_dim = _H_VIS
        self.text_hidden_dim = _D_PROJ
        self.projection_dim = _D_PROJ
        core = nn.Module()
        core.visual_projection = nn.Linear(_H_VIS, _D_PROJ, bias=False)
        core.logit_scale = nn.Parameter(torch.tensor(0.0))
        core.vision_model = nn.Identity()
        core.text_model = nn.Identity()
        self.model = core

    @property
    def vision_model(self) -> nn.Module:
        return self.model.vision_model

    @property
    def text_model(self) -> nn.Module:
        return self.model.text_model


def _model_args(**overrides) -> SimpleNamespace:
    values = {
        "device": torch.device("cpu"),
        "batch_size": _B,
        "backbone_type": "clip",
        "clip_backbone": "fake",
        "d_model": 6,
        "num_semantic_parts": NUM_SEMANTIC_PARTS,
        "num_codebooks": NUM_SEMANTIC_PARTS,
        "codebook_size": 4,
        "num_codons_per_codebook": 3,
        "codebook_update": "gradient",
        "freeze_backbone": True,
        "router_type": "sinkhorn",
        "c_global_source": "siglip2_global",
        "disable_text_supervision": False,
        "lambda_text_hash_ntxent": 0.05,
        "lambda_xmodal_commit": 0.05,
        "use_gumbel_softmax": False,
        "hash_target_mode": "siglip_cos",
        "lambda_hash": 0.0,
        "lambda_hash_hard": 0.0,
        "lambda_vq": 0.0,
        "lambda_quant": 0.0,
        "lambda_anchor": 0.0,
        "lambda_dna": 0.0,
        "lambda_bu": 0.0,
        "lambda_wasserstein": 0.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _build(args: SimpleNamespace, seed: int = 0) -> SigLIP2SemanticOTModel:
    torch.manual_seed(seed)
    with mock.patch.object(
        model_module, "build_clip_backbone", return_value=_FakeClip(),
    ):
        return SigLIP2SemanticOTModel(args).train()


def _inputs(seed: int = 1234) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {
        "cached_visual_tokens_raw": torch.randn(_B, _N, _H_VIS, generator=g),
        "cached_visual_global": torch.randn(_B, _D_PROJ, generator=g),
        "cached_text_part_raw": torch.randn(_B, NUM_SEMANTIC_PARTS, _D_PROJ, generator=g),
    }


def _forward(model: SigLIP2SemanticOTModel, inputs: dict, *, with_text: bool,
             seed: int = 7) -> dict:
    """One cached-path training forward. `with_text=False` is the deployment
    (no-caption) routing: every row is flagged as having no text."""
    torch.manual_seed(seed)
    has_text = torch.full((_B,), bool(with_text), dtype=torch.bool)
    return model(
        pixel_values=None,
        part_input_ids=None,
        part_attention_mask=None,
        return_routing=True,
        cached_visual_tokens_raw=inputs["cached_visual_tokens_raw"],
        cached_visual_global=inputs["cached_visual_global"],
        cached_text_part_raw=inputs["cached_text_part_raw"] if with_text else None,
        cached_has_text=has_text,
    )


def _assert_same_tree(a, b, path: str = "out") -> int:
    """torch.equal on every tensor leaf, == on everything else; returns the
    number of tensor leaves compared so a vacuous pass is visible."""
    n = 0
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor), path
        assert a.shape == b.shape and a.dtype == b.dtype, path
        assert torch.equal(a, b), path
        return 1
    if isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b), path
        for k in a:
            n += _assert_same_tree(a[k], b[k], f"{path}[{k!r}]")
        return n
    if isinstance(a, (list, tuple)):
        assert isinstance(b, (list, tuple)) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            n += _assert_same_tree(x, y, f"{path}[{i}]")
        return n
    assert a == b, path
    return 0


# ------------------------------------------------------------ (1) parser

def _args(*argv: str):
    return Config.build_parser().parse_args(list(argv))


def test_parser_default_is_text_and_choices_are_closed():
    assert _args().train_routing_mode == "text"
    assert _args("--train_routing_mode", "codebook_mean").train_routing_mode == "codebook_mean"
    with pytest.raises(SystemExit):
        _args("--train_routing_mode", "text_prototype")


# ------------------------------------------- (2) default flag bit-identity

def test_default_flag_is_bit_identical_to_a_model_built_without_the_attribute():
    inputs = _inputs()
    legacy = _build(_model_args())                              # attribute absent
    assert not hasattr(legacy, "_sentinel")
    flagged = _build(_model_args(train_routing_mode="text"))
    assert legacy.train_routing_mode == "text" == flagged.train_routing_mode
    # same seed at construction -> same parameters
    for (ka, pa), (kb, pb) in zip(legacy.state_dict().items(), flagged.state_dict().items()):
        assert ka == kb and torch.equal(pa, pb), ka

    out_legacy = _forward(legacy, inputs, with_text=True)
    out_flag = _forward(flagged, inputs, with_text=True)
    assert out_legacy["routing_mode"] == "text"
    assert out_flag["routing_mode"] == "text"
    assert out_legacy["text_part_tokens"] is not None
    n_leaves = _assert_same_tree(out_legacy, out_flag)
    assert n_leaves > 20, n_leaves                               # not a vacuous compare


# ------------------------------- (3) codebook_mean: text path on, router off

def test_codebook_mean_keeps_text_path_and_routes_as_deployment():
    inputs = _inputs()
    text_model = _build(_model_args(train_routing_mode="text"))
    cbm_model = _build(_model_args(train_routing_mode="codebook_mean"))
    # identical state (same construction seed)
    for (ka, pa), (kb, pb) in zip(text_model.state_dict().items(), cbm_model.state_dict().items()):
        assert ka == kb and torch.equal(pa, pb), ka

    out_text = _forward(copy.deepcopy(text_model), inputs, with_text=True)
    out_cbm = _forward(copy.deepcopy(cbm_model), inputs, with_text=True)
    out_notext = _forward(copy.deepcopy(cbm_model), inputs, with_text=False)

    assert out_text["routing_mode"] == "text"
    assert out_cbm["routing_mode"] == "codebook_mean"
    assert out_notext["routing_mode"] == "codebook_mean"
    assert cbm_model.training and text_model.training

    # text path stays on: adapter output present and identical to the
    # text-routed forward (same adapter, same raw caption features)
    assert out_cbm["text_part_tokens"] is not None
    assert out_cbm["text_part_tokens"].shape == (_B, NUM_SEMANTIC_PARTS, 6)
    assert torch.equal(out_cbm["text_part_tokens"], out_text["text_part_tokens"])
    assert torch.equal(out_cbm["global_text_token"], out_text["global_text_token"])
    assert torch.equal(out_cbm["local_text_tokens"], out_text["local_text_tokens"])
    assert out_cbm["text_global_feat"] is not None
    # text-side quantisation outputs present under codebook_mean
    for key in ("text_continuous_code", "text_quantized_tokens", "codeword_text_tokens"):
        assert out_cbm[key] is not None, key
        assert out_text[key] is not None, key
    # ... and absent in the no-text forward (so the presence check is not vacuous)
    assert out_notext["text_part_tokens"] is None
    assert out_notext["text_continuous_code"] is None

    # routing is exactly the deployment routing of the same state
    assert out_cbm["local_routing_matrix"] is not None
    assert out_notext["local_routing_matrix"] is not None
    assert torch.allclose(
        out_cbm["local_routing_matrix"], out_notext["local_routing_matrix"],
        atol=1e-6, rtol=1e-5,
    )
    assert torch.allclose(out_cbm["routing_matrix"], out_notext["routing_matrix"],
                          atol=1e-6, rtol=1e-5)
    assert torch.allclose(out_cbm["local_anchor_tokens"], out_notext["local_anchor_tokens"])
    assert torch.allclose(out_cbm["semantic_visual_tokens"],
                          out_notext["semantic_visual_tokens"], atol=1e-6, rtol=1e-5)
    # and it genuinely differs from text routing (positive control: the
    # centroids differ, so the test above could have failed)
    assert not torch.allclose(out_cbm["local_routing_matrix"],
                              out_text["local_routing_matrix"], atol=1e-6)


def test_codebook_mean_text_losses_receive_gradient_through_text_adapter():
    cbm_model = _build(_model_args(train_routing_mode="codebook_mean"))
    out = _forward(cbm_model, _inputs(), with_text=True)
    out["text_continuous_code"].sum().backward()
    grads = [p.grad for p in cbm_model.text_adapter.parameters() if p.grad is not None]
    assert grads, "text adapter received no gradient under codebook_mean"
    assert any(bool((g != 0).any()) for g in grads)


def test_model_constructor_refuses_text_only_routing_features_under_codebook_mean():
    for extra in (
        {"route_global_text": True},
        {"bidirectional_token_prune": True},
        {"routing_text_evidence_beta": 0.1},
        {"routing_token_ot_evidence": True},
        {"foreground_text_mask_topk_ratio": 0.5},
        {"routing_cls_verified_consensus_mask": True},
        {"text_transform_routing_only": True},
    ):
        with pytest.raises(ValueError, match="train_routing_mode codebook_mean"):
            _build(_model_args(train_routing_mode="codebook_mean", **extra))
        # the same flag builds under the default mode
        _build(_model_args(train_routing_mode="text", **extra))
    with pytest.raises(ValueError, match="train_routing_mode"):
        _build(_model_args(train_routing_mode="nucleus"))


# -------------------------------------------------------- (4) validator

def test_validator_passes_on_defaults_and_on_plain_codebook_mean():
    _validate_train_routing_mode_args(_args())
    _validate_train_routing_mode_args(SimpleNamespace())          # attribute absent
    _validate_train_routing_mode_args(_args("--train_routing_mode", "codebook_mean"))
    assert train_routing_mode_conflicts(_args()) == []
    assert train_routing_mode_conflicts(_args("--train_routing_mode", "codebook_mean")) == []
    # under the default mode every otherwise-refused companion passes
    for argv in (
        ("--disable_text_supervision",),
        ("--route_global_text",),
        ("--bidirectional_token_prune",),
        ("--routing_text_evidence_beta", "0.1"),
        ("--routing_token_ot_evidence", "--routing_token_ot_beta", "0.1"),
    ):
        _validate_train_routing_mode_args(_args(*argv))


@pytest.mark.parametrize("argv, needle", [
    (("--disable_text_supervision",), "disable_text_supervision"),
    (("--routing_text_evidence_beta", "0.1"), "routing_text_evidence_beta"),
    (("--routing_text_evidence_penalty", "0.1"), "routing_text_evidence_penalty"),
    (("--routing_token_ot_evidence",), "routing_token_ot"),
    (("--routing_token_ot_beta", "0.2"), "routing_token_ot"),
    (("--route_global_text",), "route_global_text"),
    (("--bidirectional_token_prune",), "bidirectional_token_prune"),
    (("--foreground_text_mask_topk_ratio", "0.5"), "foreground_text_mask_topk_ratio"),
    (("--routing_cls_verified_consensus_mask",), "routing_cls_verified_consensus_mask"),
    (("--text_transform_routing_only",), "text_transform_routing_only"),
    (("--router_type", "cross_attn"), "cross_attn"),
])
def test_validator_refuses_codebook_mean_with(argv, needle):
    args = _args("--train_routing_mode", "codebook_mean", *argv)
    with pytest.raises(ValueError, match=needle):
        _validate_train_routing_mode_args(args)


def test_validator_refuses_unknown_mode_from_a_namespace():
    with pytest.raises(ValueError, match="train_routing_mode"):
        _validate_train_routing_mode_args(SimpleNamespace(train_routing_mode="nucleus"))


def test_validator_reports_every_conflict_not_only_the_first():
    args = _args("--train_routing_mode", "codebook_mean",
                 "--route_global_text", "--bidirectional_token_prune")
    with pytest.raises(ValueError) as excinfo:
        _validate_train_routing_mode_args(args)
    msg = str(excinfo.value)
    assert "route_global_text" in msg and "bidirectional_token_prune" in msg


# --------------------------------------------------- main() ordering

def test_main_calls_the_validator_right_after_path_consistency_and_before_the_model():
    source = inspect.getsource(train_siglip2.main)
    tree = ast.parse(source)
    named: dict = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            named.setdefault(node.func.id, []).append(node.lineno)
    assert named.get("_validate_train_routing_mode_args", []) , "validator not called in main()"
    assert len(named["_validate_train_routing_mode_args"]) == 1
    pc = min(named["_validate_path_consistency_args"])
    trm = min(named["_validate_train_routing_mode_args"])
    assert pc < trm < min(named["set_random_seed"])
    assert trm < min(named["SigLIP2SemanticOTModel"])
    # no other named call sits between the two validators
    between = [n for ls in named.values() for n in ls if pc < n < trm]
    assert between == [], between
    # the path-consistency validator is unchanged by this addition
    _validate_path_consistency_args(_args())
