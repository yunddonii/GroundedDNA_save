"""Anchor confirmation v1: the typed scientific recipe and its trainer-boundary check (v2).

Audit sections 659.4, 669 and 670. Every refusal perturbs one thing of a sealed payload that the
positive control accepts, and asserts its reason. No CLIP, cache, seal or checkpoint is touched.
"""
from __future__ import annotations

import json
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from config import Config                                     # noqa: E402
import dna_utils.scientific_recipe as R                      # noqa: E402

CELL = "flickr25k|N=4|P=0.6,0.95|JD=0.02|stage=select|seed=42|axis_center=anchors"
CAMPAIGN = {"cell_id": CELL}
TOKENIZERS = {"vocab.json": "b" * 64, "merges.txt": "a" * 64}
QUOTED_TOKENIZERS = shlex.quote(json.dumps(TOKENIZERS, sort_keys=True, separators=(",", ":")))
ARGV = ["--tag", "cell_s42", "--dataset", "Flickr25k", "--setting", "1",
        "--axis_center", "anchors", "--global_gate_init_logit", "-3.0",
        "-e", "60", "--lambda_wasserstein", "0.15", "-e", "60",          # reviewed repeat
        "--post_eval_compositional", "--no-post_eval_compositional",     # reviewed override
        # the launcher shell-quotes this JSON and the wrapper does not re-parse EXTRA_ARGS,
        # so the trainer receives the literal quote layer (audit section 670)
        "--clip_snapshot_tokenizers_sha256_json", QUOTED_TOKENIZERS]


@pytest.fixture
def parser():
    return Config.build_parser()


def post_processed(parser, argv):
    """What the trainer holds when the recipe check runs: Config.set_args (setting rewritten,
    log_dir removed) and _phase3_input_authority_from_args (tokenizer JSON normalised)."""
    import train_siglip2
    values = vars(parser.parse_args(argv))
    values["setting"] = "setting" + str(values["setting"])
    values.pop("log_dir", None)
    if values.get("clip_snapshot_tokenizers_sha256_json"):
        values["clip_snapshot_tokenizers_sha256_json"] = train_siglip2._canonical_phase3_tokenizer_json(
            values["clip_snapshot_tokenizers_sha256_json"])
    return SimpleNamespace(**values)


def seal(monkeypatch, payload):
    """Seal a payload under its own digest (self-consistent even when deliberately altered)."""
    monkeypatch.setenv(R.EXPECTED_RECIPE_JSON_ENV, json.dumps(payload))
    monkeypatch.setenv(R.EXPECTED_RECIPE_DIGEST_ENV, R.digest(payload))


def unseal(monkeypatch):
    for name in (R.EXPECTED_RECIPE_JSON_ENV, R.EXPECTED_RECIPE_DIGEST_ENV):
        monkeypatch.delenv(name, raising=False)


def replace(argv, old, new):
    out = list(argv)
    out[out.index(old) + 1] = new
    return out


# ---- positive control ------------------------------------------------------------------------------
def test_sealed_recipe_with_the_real_quoted_tokenizer_path_is_accepted(parser, monkeypatch):
    payload = R.build_payload(parser, ARGV, planned_arm="anchors")
    seal(monkeypatch, payload)
    bound = R.verify_trainer_recipe(parser, ARGV, post_processed(parser, ARGV), CAMPAIGN)
    assert bound == {"scientific_recipe_schema": R.RECIPE_SCHEMA,
                     "scientific_recipe_sha256": R.digest(payload)}
    assert payload["fields"]["clip_snapshot_tokenizers_sha256_json"] == \
        json.dumps(TOKENIZERS, sort_keys=True, separators=(",", ":"))
    assert QUOTED_TOKENIZERS in payload["argv"]                    # the raw token stays bound
    assert tuple(sorted(payload)) == R.PAYLOAD_KEYS


def test_the_trainer_normaliser_is_the_recipe_normaliser():
    import train_siglip2
    for value in (QUOTED_TOKENIZERS, json.dumps(TOKENIZERS), '{ "a" : "1" }'):
        assert train_siglip2._canonical_phase3_tokenizer_json(value) == R.canonical_tokenizer_json(value)


def test_ordinary_run_without_a_campaign_is_untouched(parser, monkeypatch):
    unseal(monkeypatch)
    assert R.verify_trainer_recipe(parser, ARGV, post_processed(parser, ARGV), None) is None


# ---- 669.2: presence of the binding and of each field ---------------------------------------------
@pytest.mark.parametrize("arm", ["anchors", "none"])
def test_an_anchor_cell_without_its_sealed_recipe_refuses(parser, monkeypatch, arm):
    """Both arms: the control arm of an anchor campaign is bound exactly like the candidate."""
    unseal(monkeypatch)
    argv = replace(ARGV, "--axis_center", arm)
    cell = {"cell_id": CELL.replace("axis_center=anchors", f"axis_center={arm}")}
    with pytest.raises(R.RecipeMismatch, match="must carry its sealed scientific recipe"):
        R.verify_trainer_recipe(parser, argv, post_processed(parser, argv), cell)


def test_anchors_inside_a_non_anchor_campaign_refuses(parser, monkeypatch):
    unseal(monkeypatch)
    legacy = {"cell_id": "flickr25k|N=4|P=0.6,0.95|JD=0.02|stage=select|seed=42"}
    with pytest.raises(R.RecipeMismatch, match="requires an anchor-confirmation cell"):
        R.verify_trainer_recipe(parser, ARGV, post_processed(parser, ARGV), legacy)


def test_a_sealed_recipe_outside_an_anchor_cell_refuses(parser, monkeypatch):
    seal(monkeypatch, R.build_payload(parser, ARGV, planned_arm="anchors"))
    with pytest.raises(R.RecipeMismatch, match="only valid inside an anchor-confirmation"):
        R.verify_trainer_recipe(parser, ARGV, post_processed(parser, ARGV), None)


@pytest.mark.parametrize("which", ["digest_only", "json_only"])
def test_half_a_binding_refuses(parser, monkeypatch, which):
    payload = R.build_payload(parser, ARGV, planned_arm="anchors")
    unseal(monkeypatch)
    if which == "digest_only":
        monkeypatch.setenv(R.EXPECTED_RECIPE_DIGEST_ENV, R.digest(payload))
    else:
        monkeypatch.setenv(R.EXPECTED_RECIPE_JSON_ENV, json.dumps(payload))
    with pytest.raises(R.RecipeMismatch, match="incomplete"):
        R.verify_trainer_recipe(parser, ARGV, post_processed(parser, ARGV), CAMPAIGN)


def test_a_post_processed_field_that_is_absent_refuses(parser, monkeypatch):
    seal(monkeypatch, R.build_payload(parser, ARGV, planned_arm="anchors"))
    args = post_processed(parser, ARGV)
    del args.config_path                                   # parsed default None, now absent
    with pytest.raises(R.RecipeMismatch, match="config_path: absent"):
        R.verify_trainer_recipe(parser, ARGV, args, CAMPAIGN)


# ---- 669.1: exact, type-sensitive comparison --------------------------------------------------------
FLOAT_ARGV = replace(ARGV, "--lambda_wasserstein", "1.0")


@pytest.mark.parametrize("sealed_value", [True, 1])
def test_a_sealed_value_of_another_type_refuses(parser, monkeypatch, sealed_value):
    payload = R.build_payload(parser, FLOAT_ARGV, planned_arm="anchors")
    assert payload["fields"]["lambda_wasserstein"] == 1.0
    payload["fields"]["lambda_wasserstein"] = sealed_value
    seal(monkeypatch, payload)
    with pytest.raises(R.RecipeMismatch, match="lambda_wasserstein"):
        R.verify_trainer_recipe(parser, FLOAT_ARGV, post_processed(parser, FLOAT_ARGV), CAMPAIGN)


@pytest.mark.parametrize("runtime_value", [True, 1])
def test_a_post_processed_value_of_another_type_refuses(parser, monkeypatch, runtime_value):
    seal(monkeypatch, R.build_payload(parser, FLOAT_ARGV, planned_arm="anchors"))
    args = post_processed(parser, FLOAT_ARGV)
    args.lambda_wasserstein = runtime_value
    with pytest.raises(R.RecipeMismatch, match="lambda_wasserstein"):
        R.verify_trainer_recipe(parser, FLOAT_ARGV, args, CAMPAIGN)


def test_an_undeclared_payload_key_refuses(parser, monkeypatch):
    payload = R.build_payload(parser, ARGV, planned_arm="anchors")
    payload["note"] = "extra"
    seal(monkeypatch, payload)
    with pytest.raises(R.RecipeMismatch, match="exactly the keys"):
        R.verify_trainer_recipe(parser, ARGV, post_processed(parser, ARGV), CAMPAIGN)


@pytest.mark.parametrize("mutate,reason", [
    ("schema", "unsupported recipe schema"), ("argv_type", "argv must be a list"),
    ("fields_type", "fields must be an object"), ("nested_field", "recipe differs")])
def test_malformed_or_nested_changes_refuse(parser, monkeypatch, mutate, reason):
    payload = R.build_payload(parser, ARGV, planned_arm="anchors")
    if mutate == "schema":
        payload["schema"] = "groundeddna-scientific-recipe/1"
    elif mutate == "argv_type":
        payload["argv"] = " ".join(payload["argv"])
    elif mutate == "fields_type":
        payload["fields"] = list(payload["fields"].items())
    else:
        payload["fields"]["tag_suffix"] = ["x"]                     # an undeclared nested field
    seal(monkeypatch, payload)
    with pytest.raises(R.RecipeMismatch, match=reason):
        R.verify_trainer_recipe(parser, ARGV, post_processed(parser, ARGV), CAMPAIGN)


def test_a_bool_saved_for_a_float_field_is_a_difference(parser):
    cfg = vars(post_processed(parser, FLOAT_ARGV))
    cfg["lambda_wasserstein"] = True
    expected = R.build_payload(parser, FLOAT_ARGV, planned_arm="anchors")["fields"]
    assert R.field_differences(R.fields_from_config(cfg, parser), expected) == ["lambda_wasserstein"]


# ---- 659.4 / 669.3: runtime drift and conflicting repeats --------------------------------------------
ARGV_DIFFERS = "trainer argv differs from the sealed argv"
REFUSALS = {   # case: (runtime argv from the sealed one, the refusal it must produce)
    "declared_anchors_runtime_none": (lambda a: [t for i, t in enumerate(a) if t != "--axis_center"
                                                 and (i == 0 or a[i - 1] != "--axis_center")],
                                      "exactly once"),
    "unsupported_axis_value": (lambda a: replace(a, "--axis_center", "readout"), "rejected the argv"),
    "axis_repeated_at_runtime": (lambda a: a + ["--axis_center", "none"], "more than once"),
    "changed_gate_sign": (lambda a: replace(a, "--global_gate_init_logit", "3.0"), ARGV_DIFFERS),
    "undeclared_feature": (lambda a: a + ["--use_null_centroid"], ARGV_DIFFERS),
    "abbreviated_option": (lambda a: replace(a, "--lambda_wasserstein", "0.15")[:-2]
                           + ["--lambda_wasser", "0.15"], "abbreviated or unknown"),
    "declared_repeat_dropped": (lambda a: a[:a.index("-e", a.index("-e") + 1)]
                                + a[a.index("-e", a.index("-e") + 1) + 2:], ARGV_DIFFERS),
    "tokenizer_map_changed": (lambda a: replace(a, "--clip_snapshot_tokenizers_sha256_json",
                                                shlex.quote(json.dumps({**TOKENIZERS, "vocab.json": "c" * 64},
                                                                       sort_keys=True, separators=(",", ":")))),
                              ARGV_DIFFERS),
}


@pytest.mark.parametrize("case", REFUSALS)
def test_trainer_argv_other_than_the_sealed_one_refuses(parser, monkeypatch, case):
    seal(monkeypatch, R.build_payload(parser, ARGV, planned_arm="anchors"))
    make, reason = REFUSALS[case]
    runtime = make(ARGV)
    assert runtime != ARGV
    try:
        args = post_processed(parser, runtime)
    except SystemExit:
        args = SimpleNamespace()
    with pytest.raises(R.RecipeMismatch, match=reason):
        R.verify_trainer_recipe(parser, runtime, args, CAMPAIGN)


def test_a_dropped_repeat_leaves_the_typed_fields_unchanged(parser):
    """Why the exact argv is sealed at all: this runtime command parses to the same fields."""
    dropped = REFUSALS["declared_repeat_dropped"][0](ARGV)
    assert R.build_payload(parser, dropped)["fields"] == R.build_payload(parser, ARGV)["fields"]


@pytest.mark.parametrize("extra,reason", [
    (["--axis_center", "none"], "more than once"),                  # conflict inside the SEALED argv
    (["-bs", "64", "--batch_size", "32"], "outside the reviewed"),  # alias of a non-reviewed dest
    (["--use_stop_grad_global", "--no-use_stop_grad_global"], "outside the reviewed"),  # boolean pair
    (["--random_seed", "42", "--random_seed", "43"], "outside the reviewed"),
])
def test_an_unreviewed_or_conflicting_repeat_cannot_be_sealed(parser, extra, reason):
    with pytest.raises(R.RecipeMismatch, match=reason):
        R.build_payload(parser, ARGV + extra, planned_arm="anchors")


# ---- the reviewed override policy (audit 669.3, 677.2) ---------------------------------------------
WRAPPER_LITERALS = [tok for span in R.REVIEWED_OVERRIDES.values() for tok in span]
LAUNCHER_OVERRIDES = ["-e", "60", "--routing_adaptive_topp", "--routing_adaptive_topp_min", "0.6",
                      "--routing_adaptive_topp_max", "0.95", "--lambda_codeword_codon_sinkhorn", "0.0",
                      "--no-post_eval_compositional", "--dna_distance_mode", "base"]
COMPOSED = ["--tag", "t", "--dataset", "Flickr25k", "--setting", "1", *WRAPPER_LITERALS,
            "--hash_target_mode", "siglip_cos", *LAUNCHER_OVERRIDES, "--axis_center", "anchors"]


def test_the_wrapper_literal_then_one_override_is_admitted_for_every_reviewed_destination(parser):
    admitted = R.admitted_overrides(parser, COMPOSED)
    assert sorted(admitted) == sorted(R.REVIEWED_OVERRIDES)
    assert all(tuple(spans[0]) == R.REVIEWED_OVERRIDES[d] for d, spans in admitted.items())
    fields = R.build_payload(parser, COMPOSED, planned_arm="anchors")["fields"]
    assert (fields["routing_adaptive_topp_min"], fields["routing_adaptive_topp_max"]) == (0.6, 0.95)
    assert fields["post_eval_compositional"] is False and fields["epoch"] == 60


def test_the_override_may_use_an_alias_of_its_destination(parser):
    argv = list(ARGV)
    argv[argv.index("-e", argv.index("-e") + 1)] = "--epoch"         # -e and --epoch share a dest
    assert R.build_payload(parser, argv, planned_arm="anchors")["fields"]["epoch"] == 60


@pytest.mark.parametrize("argv,reason", [
    (COMPOSED + ["--routing_adaptive_topp_min", "0.5"], "given 3 times"),          # a third occurrence
    (ARGV + ["--epoch", "60"], "given 3 times"),                                   # third through an alias
    (["-e", "30"] + COMPOSED[COMPOSED.index("-e") + 2:], "not the reviewed wrapper literal"),
    (["--epoch", "60"] + COMPOSED[COMPOSED.index("-e") + 2:], "not the reviewed wrapper literal"),
    (["--dna_distance_mode", "codeword", "--tag", "t", "--dna_distance_mode", "base"],
     "not the reviewed wrapper literal"),                                          # override placed first
    (["--no-post_eval_compositional", "--post_eval_compositional"], "not the reviewed wrapper literal"),
    (["--routing_adaptive_topp_min=0.3", "--routing_adaptive_topp_min", "0.6"],
     "not the reviewed wrapper literal"),                                          # inline literal form
])
def test_any_other_override_sequence_refuses(parser, argv, reason):
    with pytest.raises(R.RecipeMismatch, match=reason):
        R.build_payload(parser, argv)


def test_a_valid_candidate_control_pair_differs_in_the_axis_alone(parser):
    anchors = R.build_payload(parser, ARGV, planned_arm="anchors")
    none = R.build_payload(parser, replace(ARGV, "--axis_center", "none"), planned_arm="none")
    assert R.field_differences(anchors["fields"], none["fields"]) == ["axis_center"]


def test_the_planned_arm_must_be_the_rendered_arm(parser):
    with pytest.raises(R.RecipeMismatch, match="exactly once to 'none'"):
        R.build_payload(parser, ARGV, planned_arm="none")


@pytest.mark.parametrize("token", ["--lambda_wasser", "--axis_cent", "--lambda_wasser=0.15"])
def test_an_abbreviated_option_cannot_be_sealed(parser, token):
    argv = ["--tag", "t", token] + ([] if "=" in token else ["0.15" if "wasser" in token else "anchors"])
    with pytest.raises(R.RecipeMismatch, match="abbreviated or unknown"):
        R.build_payload(parser, argv)


def test_negative_numbers_are_values_not_options(parser):
    payload = R.build_payload(parser, ["--global_gate_init_logit", "-3.0", "--lambda_bu", "-0.001"])
    assert payload["fields"]["global_gate_init_logit"] == -3.0 and payload["fields"]["lambda_bu"] == -0.001


def test_a_negative_value_in_exponent_notation_is_refused_at_sealing(parser):
    with pytest.raises(R.RecipeMismatch, match="rejected the argv"):
        R.build_payload(parser, ["--lambda_bu", "-1e-3"])


@pytest.mark.parametrize("value", ["'[1,2]'", "'not json'", "'{\"a\":1'"])
def test_a_malformed_tokenizer_authority_cannot_be_sealed(parser, value):
    with pytest.raises(R.RecipeMismatch, match="clip_snapshot_tokenizers_sha256_json"):
        R.build_payload(parser, replace(ARGV, "--clip_snapshot_tokenizers_sha256_json", value))


def test_whitespace_in_direct_tokenizer_json_normalises_to_the_same_field(parser):
    spaced = replace(ARGV, "--clip_snapshot_tokenizers_sha256_json", json.dumps(TOKENIZERS, indent=2))
    assert R.build_payload(parser, spaced)["fields"] == R.build_payload(parser, ARGV)["fields"]


# ---- other sealed-value refusals ----------------------------------------------------------------------
def test_a_missing_axis_in_the_sealed_recipe_refuses(parser, monkeypatch):
    payload = R.build_payload(parser, ARGV, planned_arm="anchors")
    del payload["fields"]["axis_center"]
    seal(monkeypatch, payload)
    with pytest.raises(R.RecipeMismatch, match="axis_center"):
        R.verify_trainer_recipe(parser, ARGV, post_processed(parser, ARGV), CAMPAIGN)


def test_a_changed_default_refuses_with_an_identical_argv(parser, monkeypatch):
    seal(monkeypatch, R.build_payload(parser, ARGV, planned_arm="anchors"))
    drifted = Config.build_parser()
    drifted.set_defaults(lambda_bu=0.5)
    with pytest.raises(R.RecipeMismatch, match="lambda_bu"):
        R.verify_trainer_recipe(drifted, ARGV, post_processed(drifted, ARGV), CAMPAIGN)


def test_args_mutated_before_the_boundary_refuse(parser, monkeypatch):
    seal(monkeypatch, R.build_payload(parser, ARGV, planned_arm="anchors"))
    args = post_processed(parser, ARGV)
    args.lambda_wasserstein = 0.3
    with pytest.raises(R.RecipeMismatch, match="lambda_wasserstein"):
        R.verify_trainer_recipe(parser, ARGV, args, CAMPAIGN)


def test_display_only_args_txt_cannot_tell_the_gate_signs_apart_but_the_recipe_can(parser):
    line = lambda v: f"{'global_gate_init_logit':-<30s}{str(v):->70s}"   # noqa: E731
    assert line(-3.0) == line(3.0)
    assert R.digest(R.build_payload(parser, ARGV)) != R.digest(
        R.build_payload(parser, replace(ARGV, "--global_gate_init_logit", "3.0")))


def test_a_sealed_json_edited_after_sealing_refuses(parser, monkeypatch):
    payload = R.build_payload(parser, ARGV, planned_arm="anchors")
    monkeypatch.setenv(R.EXPECTED_RECIPE_DIGEST_ENV, R.digest(payload))
    payload["fields"]["axis_center"] = "none"
    monkeypatch.setenv(R.EXPECTED_RECIPE_JSON_ENV, json.dumps(payload))
    with pytest.raises(R.RecipeMismatch, match="does not match its digest"):
        R.verify_trainer_recipe(parser, ARGV, post_processed(parser, ARGV), CAMPAIGN)


# ---- config.pt round trip (the reducer's side) ----------------------------------------------------
def saved_config(parser, argv):
    values = vars(post_processed(parser, argv))
    values.update(save_result_path="/x", device="cpu", num_classes=24, date="260926",
                  _phase3_campaign_binding={"cell_id": CELL})
    return values


def test_config_round_trip_reproduces_the_sealed_fields_including_the_tokenizer(parser):
    payload = R.build_payload(parser, ARGV, planned_arm="anchors")
    assert R.field_differences(R.fields_from_config(saved_config(parser, ARGV), parser),
                               payload["fields"]) == []


def test_only_the_declared_new_axis_may_be_absent_from_a_historical_config(parser):
    argv = [t for t in ARGV if t not in ("--axis_center", "anchors")]
    cfg = saved_config(parser, argv)
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


# ---- the real trainer boundary --------------------------------------------------------------------
TRAINER_ARGV = ARGV + ["--keep_final_checkpoint"]


def boundary(monkeypatch, tmp_path, events, binding):
    import train_siglip2 as T
    import dna_utils.run_identity as RI
    monkeypatch.setattr(T, "_phase3_input_authority_from_args", lambda args: None)
    monkeypatch.setattr(RI.RunIdentity, "from_args", staticmethod(lambda args: SimpleNamespace(digest="d")))
    monkeypatch.setattr(RI, "phase3_campaign_binding_from_env",
                        lambda identity, actual_tag: dict(binding, result_root=str(tmp_path)))
    monkeypatch.setattr(RI, "claim_run_dir", lambda *a, **k: events.append("claim"))
    monkeypatch.setattr(RI, "write_phase3_campaign_binding", lambda base, b: events.append(("binding", b)))
    monkeypatch.setattr(T.os, "makedirs", lambda *a, **k: events.append("makedirs"))
    return T


@pytest.mark.parametrize("case", ["conflict", "no_recipe", "runtime_none"])
def test_resolve_save_path_refuses_before_any_directory_or_claim(parser, monkeypatch, tmp_path, case):
    events = []
    T = boundary(monkeypatch, tmp_path, events, CAMPAIGN)
    if case == "no_recipe":
        unseal(monkeypatch)
        runtime = TRAINER_ARGV
    else:
        seal(monkeypatch, R.build_payload(parser, TRAINER_ARGV, planned_arm="anchors"))
        runtime = (TRAINER_ARGV + ["--axis_center", "none"] if case == "conflict"
                   else replace(TRAINER_ARGV, "--axis_center", "none"))
    monkeypatch.setattr(sys, "argv", ["train_siglip2.py", *runtime])
    args = post_processed(parser, runtime)
    args.date = "260926"
    with pytest.raises(R.RecipeMismatch):
        T._resolve_save_path(args)
    assert events == []


def test_resolve_save_path_carries_the_recipe_digest_into_the_campaign_binding(parser, monkeypatch, tmp_path):
    events = []
    T = boundary(monkeypatch, tmp_path, events, CAMPAIGN)
    payload = R.build_payload(parser, TRAINER_ARGV, planned_arm="anchors")
    seal(monkeypatch, payload)
    monkeypatch.setattr(sys, "argv", ["train_siglip2.py", *TRAINER_ARGV])
    args = post_processed(parser, TRAINER_ARGV)
    args.date = "260926"
    T._resolve_save_path(args)
    written = next(e[1] for e in events if isinstance(e, tuple))
    assert written["scientific_recipe_sha256"] == R.digest(payload)
    assert args._phase3_campaign_binding["scientific_recipe_sha256"] == R.digest(payload)
