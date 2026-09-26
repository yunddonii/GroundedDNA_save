"""Anchor confirmation v1: the typed scientific recipe and its trainer-boundary check.

Audit section 659.4 asks that a declared-anchors cell whose trainer parses `none`, a missing axis,
an unsupported value, repeated conflicting flags, a changed gate, an undeclared feature or default
and forged display-only args all refuse before any claim, load or training step. Each case below
perturbs one thing of a sealed payload that the positive control accepts. No CLIP, cache or
checkpoint is touched.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from config import Config                                     # noqa: E402
import dna_utils.scientific_recipe as R                      # noqa: E402

ARGV = ["--tag", "cell_s42", "--dataset", "Flickr25k", "--setting", "1",
        "--axis_center", "anchors", "--global_gate_init_logit", "-3.0",
        "-e", "60", "--lambda_wasserstein", "0.15", "-e", "60",          # the approved repeat
        "--post_eval_compositional", "--no-post_eval_compositional"]     # the approved override


@pytest.fixture
def parser():
    return Config.build_parser()


def post_processed(parser, argv):
    """What Config.set_args leaves on the trainer's args: every parsed value, setting rewritten."""
    values = vars(parser.parse_args(argv))
    values["setting"] = "setting" + str(values["setting"])
    values.pop("log_dir", None)
    return SimpleNamespace(**values)


def seal(monkeypatch, payload):
    monkeypatch.setenv(R.EXPECTED_RECIPE_JSON_ENV, json.dumps(payload))
    monkeypatch.setenv(R.EXPECTED_RECIPE_DIGEST_ENV, R.digest(payload))


# ---- the positive control every refusal below perturbs ------------------------------------------
def test_sealed_recipe_is_accepted_and_bound(parser, monkeypatch):
    payload = R.build_payload(parser, ARGV)
    seal(monkeypatch, payload)
    bound = R.verify_trainer_recipe(parser, ARGV, post_processed(parser, ARGV))
    assert bound == {"scientific_recipe_schema": R.RECIPE_SCHEMA,
                     "scientific_recipe_sha256": R.digest(payload)}
    assert payload["fields"]["axis_center"] == "anchors"
    assert payload["fields"]["global_gate_init_logit"] == -3.0
    assert "tag" not in payload["fields"] and len(payload["fields"]) == len(R.destinations(parser)) - 2


def test_ordinary_run_without_the_binding_is_untouched(parser, monkeypatch):
    monkeypatch.delenv(R.EXPECTED_RECIPE_JSON_ENV, raising=False)
    monkeypatch.delenv(R.EXPECTED_RECIPE_DIGEST_ENV, raising=False)
    assert R.verify_trainer_recipe(parser, ARGV, post_processed(parser, ARGV)) is None


# ---- 659.4 refusals ----------------------------------------------------------------------------
def replace(argv, old, new):
    out = list(argv)
    out[out.index(old) + 1] = new
    return out


REFUSALS = {
    "declared_anchors_runtime_none": lambda a: [t for i, t in enumerate(a)
                                                if t != "--axis_center" and (i == 0 or a[i - 1] != "--axis_center")],
    "unsupported_axis_value": lambda a: replace(a, "--axis_center", "readout"),
    "repeat_injected_at_runtime": lambda a: a + ["--axis_center", "none"],
    "changed_gate_sign": lambda a: replace(a, "--global_gate_init_logit", "3.0"),
    "undeclared_feature": lambda a: a + ["--use_null_centroid"],
    "abbreviated_option": lambda a: replace(a, "--lambda_wasserstein", "0.15")[:-2] + ["--lambda_wasser", "0.15"],
    "declared_repeat_dropped": lambda a: a[:a.index("-e", a.index("-e") + 1)] + a[a.index("-e", a.index("-e") + 1) + 2:],
}


@pytest.mark.parametrize("case", REFUSALS)
def test_trainer_argv_other_than_the_sealed_one_is_refused(parser, monkeypatch, case):
    seal(monkeypatch, R.build_payload(parser, ARGV))
    runtime = REFUSALS[case](ARGV)
    assert runtime != ARGV
    with pytest.raises(R.RecipeMismatch):
        R.verify_trainer_recipe(parser, runtime, SimpleNamespace())


def test_missing_axis_in_the_sealed_recipe_is_refused(parser, monkeypatch):
    payload = R.build_payload(parser, ARGV)
    del payload["fields"]["axis_center"]
    seal(monkeypatch, payload)
    with pytest.raises(R.RecipeMismatch, match="axis_center"):
        R.verify_trainer_recipe(parser, ARGV, post_processed(parser, ARGV))


def test_a_changed_default_is_refused_with_an_identical_argv(parser, monkeypatch):
    """The source generation's default moved; the command did not."""
    seal(monkeypatch, R.build_payload(parser, ARGV))
    drifted = Config.build_parser()
    drifted.set_defaults(lambda_bu=0.5)
    with pytest.raises(R.RecipeMismatch, match="lambda_bu"):
        R.verify_trainer_recipe(drifted, ARGV, post_processed(drifted, ARGV))


def test_args_mutated_before_the_boundary_are_refused(parser, monkeypatch):
    seal(monkeypatch, R.build_payload(parser, ARGV))
    args = post_processed(parser, ARGV)
    args.lambda_wasserstein = 0.3
    with pytest.raises(R.RecipeMismatch, match="lambda_wasserstein"):
        R.verify_trainer_recipe(parser, ARGV, args)


def test_display_only_args_txt_cannot_tell_the_gate_signs_apart_but_the_recipe_can(parser):
    line = lambda v: f"{'global_gate_init_logit':-<30s}{str(v):->70s}"   # Config.print_info  # noqa: E731
    assert line(-3.0) == line(3.0)
    assert R.digest(R.build_payload(parser, ARGV)) != R.digest(
        R.build_payload(parser, replace(ARGV, "--global_gate_init_logit", "3.0")))


@pytest.mark.parametrize("which", ["digest_only", "json_only"])
def test_half_a_binding_is_refused(parser, monkeypatch, which):
    payload = R.build_payload(parser, ARGV)
    monkeypatch.delenv(R.EXPECTED_RECIPE_JSON_ENV, raising=False)
    monkeypatch.delenv(R.EXPECTED_RECIPE_DIGEST_ENV, raising=False)
    if which == "digest_only":
        monkeypatch.setenv(R.EXPECTED_RECIPE_DIGEST_ENV, R.digest(payload))
    else:
        monkeypatch.setenv(R.EXPECTED_RECIPE_JSON_ENV, json.dumps(payload))
    with pytest.raises(R.RecipeMismatch, match="incomplete"):
        R.verify_trainer_recipe(parser, ARGV, post_processed(parser, ARGV))


def test_a_sealed_json_edited_after_sealing_is_refused(parser, monkeypatch):
    payload = R.build_payload(parser, ARGV)
    monkeypatch.setenv(R.EXPECTED_RECIPE_DIGEST_ENV, R.digest(payload))
    payload["fields"]["axis_center"] = "none"
    monkeypatch.setenv(R.EXPECTED_RECIPE_JSON_ENV, json.dumps(payload))
    with pytest.raises(R.RecipeMismatch, match="does not match its digest"):
        R.verify_trainer_recipe(parser, ARGV, post_processed(parser, ARGV))


# ---- config.pt round trip (the reducer's side) ---------------------------------------------------
def saved_config(parser, argv):
    values = vars(post_processed(parser, argv))
    values.update(save_result_path="/x", device="cpu", num_classes=24, date="260926",
                  _phase3_campaign_binding={"cell_id": "c"})
    return values


def test_config_round_trip_reproduces_the_sealed_fields(parser):
    payload = R.build_payload(parser, ARGV)
    assert R.fields_from_config(saved_config(parser, ARGV), parser) == payload["fields"]


def test_a_value_changed_during_training_shows_in_the_config(parser):
    cfg = saved_config(parser, ARGV)
    cfg["axis_center"] = "none"
    assert R.field_differences(R.fields_from_config(cfg, parser),
                               R.build_payload(parser, ARGV)["fields"]) == ["axis_center"]


def test_only_the_declared_new_axis_may_be_absent_from_a_historical_config(parser):
    cfg = saved_config(parser, [t for t in ARGV if t not in ("--axis_center", "anchors")])
    del cfg["axis_center"]
    assert R.fields_from_config(cfg, parser, historical=True)["axis_center"] == "none"
    with pytest.raises(R.RecipeMismatch, match="axis_center"):
        R.fields_from_config(cfg, parser)
    del cfg["lambda_wasserstein"]
    with pytest.raises(R.RecipeMismatch, match="lambda_wasserstein"):
        R.fields_from_config(cfg, parser, historical=True)


def test_non_finite_values_cannot_enter_a_recipe(parser):
    cfg = saved_config(parser, ARGV)
    cfg["lambda_wasserstein"] = float("nan")
    with pytest.raises(R.RecipeMismatch, match="non-finite"):
        R.fields_from_config(cfg, parser)


# ---- the real trainer boundary -------------------------------------------------------------------
TRAINER_ARGV = ARGV + ["--keep_final_checkpoint"]     # a campaign must keep its terminal checkpoint

def test_resolve_save_path_refuses_before_any_directory_or_claim(parser, monkeypatch, tmp_path):
    """Drive train_siglip2._resolve_save_path itself: a sealed recipe the argv does not match must
    refuse before os.makedirs and claim_run_dir are reached."""
    import train_siglip2 as T
    import dna_utils.run_identity as RI
    events = []
    monkeypatch.setattr(T, "_phase3_input_authority_from_args", lambda args: None)
    monkeypatch.setattr(RI.RunIdentity, "from_args", staticmethod(lambda args: SimpleNamespace(digest="d")))
    monkeypatch.setattr(RI, "phase3_campaign_binding_from_env",
                        lambda identity, actual_tag: {"result_root": str(tmp_path)})
    monkeypatch.setattr(RI, "claim_run_dir", lambda *a, **k: events.append("claim"))
    monkeypatch.setattr(T.os, "makedirs", lambda *a, **k: events.append("makedirs"))
    seal(monkeypatch, R.build_payload(parser, TRAINER_ARGV))
    runtime = replace(TRAINER_ARGV, "--axis_center", "none")
    monkeypatch.setattr(sys, "argv", ["train_siglip2.py", *runtime])
    args = post_processed(parser, runtime)
    args.date = "260926"
    with pytest.raises(R.RecipeMismatch):
        T._resolve_save_path(args)
    assert events == []


def test_resolve_save_path_carries_the_recipe_digest_into_the_campaign_binding(parser, monkeypatch, tmp_path):
    import train_siglip2 as T
    import dna_utils.run_identity as RI
    written = {}
    monkeypatch.setattr(T, "_phase3_input_authority_from_args", lambda args: None)
    monkeypatch.setattr(RI.RunIdentity, "from_args", staticmethod(lambda args: SimpleNamespace(digest="d")))
    monkeypatch.setattr(RI, "phase3_campaign_binding_from_env",
                        lambda identity, actual_tag: {"result_root": str(tmp_path)})
    monkeypatch.setattr(RI, "claim_run_dir", lambda *a, **k: None)
    monkeypatch.setattr(RI, "write_phase3_campaign_binding", lambda base, b: written.update(b))
    payload = R.build_payload(parser, TRAINER_ARGV)
    seal(monkeypatch, payload)
    monkeypatch.setattr(sys, "argv", ["train_siglip2.py", *TRAINER_ARGV])
    args = post_processed(parser, TRAINER_ARGV)
    args.date = "260926"
    T._resolve_save_path(args)
    assert written["scientific_recipe_sha256"] == R.digest(payload)
    assert args._phase3_campaign_binding["scientific_recipe_sha256"] == R.digest(payload)
