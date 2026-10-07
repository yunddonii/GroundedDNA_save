"""TD (2026-10-07): text-dropout path consistency (Stage-3 arm N1).

CPU-only tests that drive the real parser, the real DNACodonHashLoss
constructor / forward, the real log-column gate, the real startup validator
and the real per-step gate helper.
"""
import ast
import inspect
import math
from types import SimpleNamespace

import pytest
import torch

import train_siglip2
from config import Config
from loss_siglip2 import DNACodonHashLoss
from train_siglip2 import (
    _build_active_loss_types,
    _refuse_path_consistency_without_text,
    _td_should_run,
    _validate_path_consistency_args,
)


def _args(*argv: str):
    return Config.build_parser().parse_args(list(argv))


def _outputs(num_codebooks: int, *, batch: int = 3, d_model: int = 8,
             codebook_size: int = 4, seed: int = 20261007,
             routing_mode: str = "text") -> dict:
    """Minimal synthetic forward output accepted by DNACodonHashLoss.forward.

    routing_mode defaults to "text" (a caption-routed teacher) because the
    term is gated on it; tests that want a no-text teacher pass it explicitly.
    """
    g = torch.Generator().manual_seed(seed)
    M, L = num_codebooks, 3
    return {
        "continuous_code": torch.rand(batch, M * L, 4, generator=g).softmax(-1),
        "semantic_visual_tokens": torch.randn(batch, M, d_model, generator=g),
        "quantized_tokens_raw": torch.randn(batch, M, d_model, generator=g),
        "codebook_distances": torch.rand(batch, M, codebook_size, generator=g),
        "local_codebook_mean_anchors": torch.randn(M - 1, d_model, generator=g),
        "routing_mode": routing_mode,
    }


def _labels(batch: int) -> torch.Tensor:
    return torch.arange(batch) % 2


def _hand_kl(d_text, d_notext, tau, slots, active=None) -> float:
    """Pure-python KL(p_text || p_notext), mean over batch x slots.

    `active` is either None, a [M][K] list (shared over the batch) or a
    [B][M][K] list (per sample); non-finite teacher distances are always
    excluded from the support, mirroring the loss.
    """
    total, count = 0.0, 0
    for b in range(len(d_text)):
        for m in slots:
            def _on(k):
                if not math.isfinite(d_text[b][m][k]):
                    return False
                if active is None:
                    return True
                row = active[b][m] if isinstance(active[0][0], list) else active[m]
                return bool(row[k])
            keep = [k for k in range(len(d_text[b][m])) if _on(k)]
            lt = [-d_text[b][m][k] / tau for k in keep]
            ln = [-d_notext[b][m][k] / tau for k in keep]
            zt = sum(math.exp(x) for x in lt)
            zn = sum(math.exp(x) for x in ln)
            pt = [math.exp(x) / zt for x in lt]
            pn = [math.exp(x) / zn for x in ln]
            total += sum(p * (math.log(p) - math.log(q)) for p, q in zip(pt, pn))
            count += 1
    return total / count


# ------------------------------------------------------------------ (c) flags

def test_parser_declares_the_new_flags_with_the_stated_defaults():
    args = _args()
    assert args.lambda_path_consistency == 0.0
    assert args.path_consistency_tau == 0.1
    assert args.path_consistency_include_global is False
    assert args.text_dropout_p == 0.0
    on = _args("--lambda_path_consistency", "0.05", "--path_consistency_tau", "0.5",
               "--path_consistency_include_global", "--text_dropout_p", "1.0")
    assert on.lambda_path_consistency == 0.05
    assert on.path_consistency_tau == 0.5
    assert on.path_consistency_include_global is True
    assert on.text_dropout_p == 1.0


# -------------------------------------------------------- (a) default = no-op

def test_default_flags_ignore_outputs_notext_and_return_zero_term():
    args = _args()
    M = int(args.num_codebooks)
    labels = _labels(3)

    torch.manual_seed(0)
    base = DNACodonHashLoss(args)(_outputs(M), labels=labels)
    torch.manual_seed(0)
    with_notext = DNACodonHashLoss(args)(
        _outputs(M), labels=labels,
        outputs_notext=_outputs(M, seed=99),
    )

    assert "path_consistency" in base
    assert torch.equal(base["path_consistency"], torch.zeros(()))
    assert set(base) == set(with_notext)
    for key in base:
        assert torch.equal(base[key], with_notext[key]), key
    assert torch.equal(base["loss"], with_notext["loss"])
    assert "path_consistency" not in _build_active_loss_types(args)


def test_lambda_on_but_no_notext_forward_is_still_zero():
    args = _args("--lambda_path_consistency", "0.05")
    M = int(args.num_codebooks)
    out = DNACodonHashLoss(args)(_outputs(M), labels=_labels(3))
    assert torch.equal(out["path_consistency"], torch.zeros(()))
    assert "path_consistency" in _build_active_loss_types(args)


# --------------------------------------------------- (b) value and gradients

def test_two_slot_toy_matches_hand_computed_kl_and_gradient_routing():
    tau = 0.5
    args = _args("--lambda_path_consistency", "0.05",
                 "--path_consistency_tau", str(tau), "--num_codebooks", "3")
    M = 3                       # slot 0 = global (skipped by default) + 2 local
    out = _outputs(M, batch=2, codebook_size=3)
    d_text = out["codebook_distances"].clone().requires_grad_(True)
    out["codebook_distances"] = d_text
    d_notext = torch.rand(2, M, 3, generator=torch.Generator().manual_seed(5))
    d_notext.requires_grad_(True)
    out_notext = {"codebook_distances": d_notext}

    torch.manual_seed(0)
    r = DNACodonHashLoss(args)(out, labels=_labels(2), outputs_notext=out_notext)
    torch.manual_seed(0)
    base = DNACodonHashLoss(args)(out, labels=_labels(2))   # no notext -> 0

    term = r["path_consistency"]
    expected = _hand_kl(d_text.tolist(), d_notext.tolist(), tau, slots=[1, 2])
    assert float(term) > 0.0
    assert float(term) == pytest.approx(expected, rel=1e-5, abs=1e-7)
    assert float(r["loss"]) == pytest.approx(
        float(base["loss"]) + 0.05 * expected, rel=1e-5, abs=1e-6)

    g_notext, g_text = torch.autograd.grad(
        term, (d_notext, d_text), allow_unused=True)
    assert g_notext is not None and float(g_notext.abs().sum()) > 0.0
    assert g_text is None or float(g_text.abs().sum()) == 0.0
    # slot 0 is excluded by default: no gradient reaches its distances
    assert float(g_notext[:, 0, :].abs().sum()) == 0.0
    assert float(g_notext[:, 1:, :].abs().sum()) > 0.0


def test_include_global_and_active_mask_follow_the_hand_formula():
    tau = 0.25
    out = _outputs(3, batch=2, codebook_size=3)
    d_notext = torch.rand(2, 3, 3, generator=torch.Generator().manual_seed(6))
    d_text = out["codebook_distances"]

    args = _args("--lambda_path_consistency", "1.0", "--path_consistency_tau",
                 str(tau), "--num_codebooks", "3",
                 "--path_consistency_include_global")
    r = DNACodonHashLoss(args)(out, labels=_labels(2),
                               outputs_notext={"codebook_distances": d_notext})
    expected = _hand_kl(d_text.tolist(), d_notext.tolist(), tau, slots=[0, 1, 2])
    assert float(r["path_consistency"]) == pytest.approx(expected, rel=1e-5)

    active = torch.tensor([[True, True, True],
                           [True, False, True],
                           [False, True, True]])
    out_masked = dict(out, codebook_active_mask=active)
    args = _args("--lambda_path_consistency", "1.0", "--path_consistency_tau",
                 str(tau), "--num_codebooks", "3")
    r = DNACodonHashLoss(args)(out_masked, labels=_labels(2),
                               outputs_notext={"codebook_distances": d_notext})
    expected = _hand_kl(d_text.tolist(), d_notext.tolist(), tau, slots=[1, 2],
                        active=active.tolist())
    assert float(r["path_consistency"]) == pytest.approx(expected, rel=1e-5)
    assert torch.isfinite(r["path_consistency"])


def test_plus_inf_inactive_distances_without_active_mask_stay_finite():
    """(iv) The quantizer fills inactive codewords with +inf distances. With
    NO codebook_active_mask supplied, the term must still be finite, match
    the hand formula over the finite support, and have a finite gradient
    (zero at the +inf positions)."""
    tau = 0.5
    M, K = 3, 4
    out = _outputs(M, batch=2, codebook_size=K)
    d_text = out["codebook_distances"].clone()
    d_notext = torch.rand(2, M, K, generator=torch.Generator().manual_seed(7))
    inf = float("inf")
    # inactive codewords: slot 1 / codeword 3, slot 2 / codeword 0 (all samples)
    d_text[:, 1, 3] = inf
    d_text[:, 2, 0] = inf
    d_notext[:, 1, 3] = inf
    d_notext[:, 2, 0] = inf
    d_notext.requires_grad_(True)
    out["codebook_distances"] = d_text
    assert "codebook_active_mask" not in out

    args = _args("--lambda_path_consistency", "1.0", "--path_consistency_tau",
                 str(tau), "--num_codebooks", str(M))
    r = DNACodonHashLoss(args)(out, labels=_labels(2),
                               outputs_notext={"codebook_distances": d_notext})
    term = r["path_consistency"]
    assert torch.isfinite(term)
    assert float(term) > 0.0
    expected = _hand_kl(d_text.tolist(), d_notext.tolist(), tau, slots=[1, 2])
    assert float(term) == pytest.approx(expected, rel=1e-5)
    assert torch.isfinite(r["loss"])

    (g,) = torch.autograd.grad(term, (d_notext,))
    assert torch.isfinite(g).all()
    assert float(g[:, 1, 3].abs().sum()) == 0.0
    assert float(g[:, 2, 0].abs().sum()) == 0.0
    assert float(g[:, 1:, :].abs().sum()) > 0.0


def test_plus_inf_combined_with_explicit_active_mask():
    """+inf in the teacher AND an explicit active mask: the support is the
    intersection (OR of the two inactive masks)."""
    tau = 0.5
    M, K = 3, 4
    out = _outputs(M, batch=2, codebook_size=K)
    d_text = out["codebook_distances"].clone()
    d_notext = torch.rand(2, M, K, generator=torch.Generator().manual_seed(8))
    d_text[:, 1, 3] = float("inf")
    d_notext[:, 1, 3] = float("inf")
    active = torch.ones(M, K, dtype=torch.bool)
    active[2, 1] = False            # masked only via the explicit mask
    out["codebook_distances"] = d_text
    out["codebook_active_mask"] = active

    args = _args("--lambda_path_consistency", "1.0", "--path_consistency_tau",
                 str(tau), "--num_codebooks", str(M))
    r = DNACodonHashLoss(args)(out, labels=_labels(2),
                               outputs_notext={"codebook_distances": d_notext})
    term = r["path_consistency"]
    assert torch.isfinite(term)
    expected = _hand_kl(d_text.tolist(), d_notext.tolist(), tau, slots=[1, 2],
                        active=active.tolist())
    assert float(term) == pytest.approx(expected, rel=1e-5)


def test_shape_mismatch_is_refused():
    args = _args("--lambda_path_consistency", "0.05", "--num_codebooks", "3")
    out = _outputs(3, batch=2, codebook_size=3)
    with pytest.raises(ValueError, match="path_consistency"):
        DNACodonHashLoss(args)(out, labels=_labels(2),
                               outputs_notext={"codebook_distances": torch.rand(2, 3, 4)})


def test_routing_mode_not_text_gives_zero_term():
    """(v) A teacher that was NOT caption-routed (routing_mode != "text")
    yields the zero tensor even with lambda > 0 and outputs_notext given,
    and the total is bit-identical to the no-notext call."""
    args = _args("--lambda_path_consistency", "0.05", "--num_codebooks", "3")
    d_notext = torch.rand(2, 3, 3, generator=torch.Generator().manual_seed(9))
    for mode in ("codebook_mean", None, "text_prototype"):
        out = _outputs(3, batch=2, codebook_size=3, routing_mode=mode)
        if mode is None:
            out.pop("routing_mode")
        torch.manual_seed(0)
        r = DNACodonHashLoss(args)(out, labels=_labels(2),
                                   outputs_notext={"codebook_distances": d_notext})
        torch.manual_seed(0)
        base = DNACodonHashLoss(args)(out, labels=_labels(2))
        assert torch.equal(r["path_consistency"], torch.zeros(())), mode
        assert torch.equal(r["loss"], base["loss"]), mode
    # positive control: the same inputs with routing_mode == "text" are nonzero
    out = _outputs(3, batch=2, codebook_size=3, routing_mode="text")
    r = DNACodonHashLoss(args)(out, labels=_labels(2),
                               outputs_notext={"codebook_distances": d_notext})
    assert float(r["path_consistency"]) > 0.0


# ---------------------------------------------------------- (d) validator

def _ns(**kw):
    base = dict(disable_text_supervision=False, lambda_path_consistency=0.0,
                text_dropout_p=0.0, path_consistency_tau=0.1,
                eval_routing_mode="codebook_mean")
    base.update(kw)
    return SimpleNamespace(**base)


def test_validator_passes_on_defaults_and_on_a_valid_active_arm():
    _validate_path_consistency_args(_args())
    _validate_path_consistency_args(_ns())
    _validate_path_consistency_args(_ns(lambda_path_consistency=0.05,
                                        text_dropout_p=1.0))
    _validate_path_consistency_args(_ns(lambda_path_consistency=0.05,
                                        text_dropout_p=0.3))
    _validate_path_consistency_args(_args(
        "--lambda_path_consistency", "0.05", "--text_dropout_p", "0.5"))
    # lambda 0 tolerates every otherwise-bad companion flag except tau <= 0
    _validate_path_consistency_args(_ns(disable_text_supervision=True))
    _validate_path_consistency_args(_ns(text_dropout_p=0.0))
    _validate_path_consistency_args(_ns(text_dropout_p=7.0))
    _validate_path_consistency_args(_ns(eval_routing_mode="text_prototype"))
    # alias still resolves to the same callable
    assert _refuse_path_consistency_without_text is _validate_path_consistency_args


def test_validator_refuses_disable_text_supervision():
    with pytest.raises(ValueError, match="disable_text_supervision"):
        _validate_path_consistency_args(_ns(
            disable_text_supervision=True, lambda_path_consistency=0.05,
            text_dropout_p=1.0))
    with pytest.raises(ValueError, match="disable_text_supervision"):
        _refuse_path_consistency_without_text(_args(
            "--disable_text_supervision", "--lambda_path_consistency", "0.05",
            "--text_dropout_p", "1.0"))


@pytest.mark.parametrize("p", [0.0, -0.1, 1.5])
def test_validator_refuses_text_dropout_p_outside_unit_interval(p):
    with pytest.raises(ValueError, match="text_dropout_p"):
        _validate_path_consistency_args(_ns(lambda_path_consistency=0.05,
                                            text_dropout_p=p))
    with pytest.raises(ValueError, match="text_dropout_p"):
        _validate_path_consistency_args(_args(
            "--lambda_path_consistency", "0.05", "--text_dropout_p", str(p)))


@pytest.mark.parametrize("tau", [0.0, -0.1])
def test_validator_refuses_nonpositive_tau_even_with_lambda_zero(tau):
    with pytest.raises(ValueError, match="path_consistency_tau"):
        _validate_path_consistency_args(_ns(path_consistency_tau=tau))
    with pytest.raises(ValueError, match="path_consistency_tau"):
        _validate_path_consistency_args(_args("--path_consistency_tau", str(tau)))


def test_validator_refuses_text_prototype_eval_routing():
    with pytest.raises(ValueError, match="text_prototype"):
        _validate_path_consistency_args(_ns(lambda_path_consistency=0.05,
                                            text_dropout_p=1.0,
                                            eval_routing_mode="text_prototype"))
    with pytest.raises(ValueError, match="text_prototype"):
        _validate_path_consistency_args(_args(
            "--lambda_path_consistency", "0.05", "--text_dropout_p", "1.0",
            "--eval_routing_mode", "text_prototype"))


# ------------------------------------------------- (i) per-step gate + RNG

def _gen(seed: int = 123) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


@pytest.mark.parametrize("p", [0.0, 0.3, 1.0, 5.0])
def test_gate_lambda_zero_draws_nothing_and_returns_false(p):
    gen = _gen()
    gen_state = gen.get_state().clone()
    torch.manual_seed(31)
    before = torch.get_rng_state().clone()
    assert _td_should_run(True, True, 0.0, p, gen) is False
    assert torch.equal(torch.get_rng_state(), before)
    assert torch.equal(gen.get_state(), gen_state)


@pytest.mark.parametrize("p", [1.0, 1.5])
def test_gate_p_at_least_one_draws_nothing_and_returns_true(p):
    gen = _gen()
    gen_state = gen.get_state().clone()
    torch.manual_seed(31)
    before = torch.get_rng_state().clone()
    assert _td_should_run(True, True, 0.05, p, gen) is True
    assert torch.equal(torch.get_rng_state(), before)
    assert torch.equal(gen.get_state(), gen_state)


def test_gate_not_train_or_not_cache_draws_nothing_and_returns_false():
    gen = _gen()
    gen_state = gen.get_state().clone()
    torch.manual_seed(31)
    before = torch.get_rng_state().clone()
    assert _td_should_run(False, True, 0.05, 0.5, gen) is False
    assert _td_should_run(True, False, 0.05, 0.5, gen) is False
    assert torch.equal(torch.get_rng_state(), before)
    assert torch.equal(gen.get_state(), gen_state)


def test_gate_fractional_p_draws_exactly_once_from_the_dedicated_generator():
    p = 0.3
    torch.manual_seed(31)
    before = torch.get_rng_state().clone()
    gen = _gen(777)
    ref = _gen(777)
    results = [_td_should_run(True, True, 0.05, p, gen) for _ in range(50)]
    # global RNG stream untouched across 50 gated steps
    assert torch.equal(torch.get_rng_state(), before)
    # exactly one scalar draw per call: replaying the same seed reproduces
    # the decisions, and the generator advanced by exactly 50 draws
    expected = [bool(torch.rand((), generator=ref).item() < p) for _ in range(50)]
    assert results == expected
    assert torch.equal(gen.get_state(), ref.get_state())
    assert any(results) and not all(results)   # positive control for p=0.3
    # the dedicated stream is reproducible from (random_seed + 1009)
    g1 = torch.Generator().manual_seed(42 + 1009)
    g2 = torch.Generator().manual_seed(42 + 1009)
    assert ([_td_should_run(True, True, 0.05, p, g1) for _ in range(20)]
            == [_td_should_run(True, True, 0.05, p, g2) for _ in range(20)])


# --------------------------------------------------- (iii) main() ordering

def _calls_in_main():
    source = inspect.getsource(train_siglip2.main)
    tree = ast.parse(source)
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call)]


def test_main_validates_before_seed_model_and_loaders():
    calls = _calls_in_main()
    named = {}
    for node in calls:
        if isinstance(node.func, ast.Name):
            named.setdefault(node.func.id, []).append(node.lineno)
    assert "_validate_path_consistency_args" in named
    assert "SigLIP2SemanticOTModel" in named
    assert "set_random_seed" in named
    validator_line = min(named["_validate_path_consistency_args"])
    assert validator_line < min(named["SigLIP2SemanticOTModel"])
    assert validator_line < min(named["set_random_seed"])
    # torch.load / DataLoader are attribute calls; they must also come later
    attr_lines = {}
    for node in calls:
        if isinstance(node.func, ast.Attribute):
            attr_lines.setdefault(node.func.attr, []).append(node.lineno)
    for later in ("load",):
        if later in attr_lines:
            assert validator_line < min(attr_lines[later]), later
    loader_lines = named.get("DataLoader", []) + attr_lines.get("DataLoader", [])
    assert loader_lines, "main() builds no DataLoader?"
    assert validator_line < min(loader_lines)
    # the legacy name is no longer called inside main (single call site)
    assert "_refuse_path_consistency_without_text" not in named
    assert len(named["_validate_path_consistency_args"]) == 1


def test_main_uses_the_gate_helper_and_passes_outputs_notext():
    calls = _calls_in_main()
    named = {node.func.id for node in calls if isinstance(node.func, ast.Name)}
    assert "_td_should_run" in named
    kw = {k.arg for node in calls for k in node.keywords}
    assert "outputs_notext" in kw
    assert "outputs_notext" in inspect.signature(DNACodonHashLoss.forward).parameters
    # the only torch.rand inside main() must go through a generator kwarg
    for node in calls:
        f = node.func
        if (isinstance(f, ast.Attribute) and f.attr == "rand"
                and isinstance(f.value, ast.Name) and f.value.id == "torch"):
            assert any(k.arg == "generator" for k in node.keywords), \
                f"bare torch.rand at main() line {node.lineno}"
