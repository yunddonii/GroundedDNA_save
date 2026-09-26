"""Anchor confirmation v1: the launcher mode (audit section 659.4).

Covered here with synthetic inputs and every load/claim/launch boundary stubbed:

* none/anchors commands for all four dataset profiles differ in the axis flag and tag alone;
* the refit path and exploratory arms refuse; so do lambda overrides on an anchor cell;
* the ledger-pinned incumbent authorities refuse wrong bytes and disagreement;
* the frozen stage-S record for `decide` is pinned and typed;
* `--plan` writes nothing, reserves nothing and launches nothing, and refused requests never
  reach those boundaries;
* completion refuses evidence or a saved config that is not the sealed recipe.

One integration test (skipped where the approved artifacts are absent) runs the real `--plan`
against the real approved aggregate and approved config.pt, read-only.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import scripts.phase3_selection_matrix as M                 # noqa: E402
from scripts.phase3_selection_matrix import CellRefused     # noqa: E402
import dna_utils.scientific_recipe as R                     # noqa: E402

INCUMBENT = {"flickr25k": {"N": 4, "topp": ("0.6", "0.95"), "joint": "0.02"},
             "nuswide": {"N": 4, "topp": ("0.4", "0.8"), "joint": "0.05"},
             "mscoco": {"N": 39, "topp": ("0.6", "0.95"), "joint": "0.03"}}


def extra(env):
    return env["EXTRA_ARGS"].split()


# ---- commands ------------------------------------------------------------------------------------
@pytest.mark.parametrize("dataset", ["flickr25k", "nuswide", "mscoco", "cifar10"])
def test_arms_differ_in_the_axis_flag_and_tag_alone(dataset):
    built = {arm: M.build_command(dataset, 4, 0, topp=("0.6", "0.95"), joint="0.02", anchor_arm=arm)
             for arm in M.ANCHOR_ARMS}
    legacy = M.build_command(dataset, 4, 0, topp=("0.6", "0.95"), joint="0.02")
    for arm, (cmd, env, tag) in built.items():
        assert cmd == legacy[0]
        assert extra(env)[-2:] == ["--axis_center", arm]
        assert extra(env)[:-2] == extra(legacy[1])
        assert {k: v for k, v in env.items() if k not in ("EXTRA_ARGS", "TAG")} == \
               {k: v for k, v in legacy[1].items() if k not in ("EXTRA_ARGS", "TAG")}
    tags = [built["none"][2], built["anchors"][2], legacy[2]]
    assert len(set(tags)) == 3
    assert not any(a in b for a in tags for b in tags if a != b)      # `*{tag}*` globs


@pytest.mark.parametrize("arm", ["readout", "both", "ancsoft", "", "Anchors"])
def test_exploratory_or_unknown_arms_refuse(arm):
    with pytest.raises(CellRefused, match="not an anchor-confirmation arm"):
        M.build_command("flickr25k", 4, 0, anchor_arm=arm)


def test_the_refit_path_refuses_an_anchor_arm():
    with pytest.raises(CellRefused, match="separate authorization"):
        M.build_command("flickr25k", 4, 0, stage="refit", anchor_arm="anchors")


def test_an_anchor_cell_moves_axis_center_alone():
    with pytest.raises(CellRefused, match="axis_center alone"):
        M.build_command("flickr25k", 4, 0, anchor_arm="anchors",
                        overrides=(("lambda_wasserstein", "0.30"),))


def test_cell_ids_carry_the_arm_and_leave_legacy_ids_unchanged():
    kw = dict(topp=("0.6", "0.95"), joint="0.02")
    legacy = M.campaign_cell_id("flickr25k", 4, **kw)
    assert "axis_center" not in legacy
    ids = {M.campaign_cell_id("flickr25k", 4, anchor_arm=a, **kw) for a in M.ANCHOR_ARMS}
    assert ids == {legacy + "|axis_center=none", legacy + "|axis_center=anchors"}


def test_eight_tuples_are_anchor_cells_and_nothing_else_changes():
    cell = ("flickr25k", 4, ("0.6", "0.95"), "0.02", "select", 42, (), "anchors")
    assert M._campaign_cell_parts(cell) == cell[:6]
    assert M._cell_overrides(cell) == () and M._cell_anchor_arm(cell) == "anchors"
    assert M._cell_anchor_arm(cell[:7]) is None and M._cell_anchor_arm(cell[:6]) is None
    with pytest.raises(CellRefused):
        M._cell_overrides(cell[:6] + ((("lambda_bu", "0"),), "anchors"))
    with pytest.raises(CellRefused):
        M._cell_anchor_arm(cell[:7] + ("both",))


# ---- cells -----------------------------------------------------------------------------------------
def test_select_cells_cover_the_three_multi_label_datasets_and_the_grid():
    cells = M.anchor_confirmation_cells("select", incumbent=INCUMBENT, arms=("anchors",))
    assert len(cells) == 12 and {c[0] for c in cells} == set(M.ANCHOR_DATASETS)
    assert "cifar10" not in {c[0] for c in cells}
    assert sorted({c[1] for c in cells}) == list(M.CANDIDATE_N)
    assert {c[5] for c in cells} == {42} and {c[4] for c in cells} == {"select"}
    assert all((c[2], c[3]) == (INCUMBENT[c[0]]["topp"], INCUMBENT[c[0]]["joint"]) for c in cells)
    assert len(M.anchor_confirmation_cells("select", incumbent=INCUMBENT, arms=M.ANCHOR_ARMS)) == 24


def test_decide_cells_take_each_arm_at_its_frozen_n():
    selection = {"flickr25k": {"anchors": 9}, "nuswide": {"anchors": 4}, "mscoco": {"anchors": 39}}
    cells = M.anchor_confirmation_cells("decide", incumbent=INCUMBENT, arms=("anchors",),
                                        selection=selection)
    assert sorted((c[0], c[1], c[5]) for c in cells) == sorted(
        (ds, n["anchors"], s) for ds, n in selection.items() for s in (43, 44))


@pytest.mark.parametrize("arms,selection,match", [
    ((), None, "nonempty"), (("anchors", "anchors"), None, "nonempty"), (("both",), None, "nonempty"),
])
def test_invalid_arm_lists_refuse(arms, selection, match):
    with pytest.raises(CellRefused, match=match):
        M.anchor_confirmation_cells("select", incumbent=INCUMBENT, arms=arms)


@pytest.mark.parametrize("n", [None, 5, "4", 4.0, True])
def test_decide_refuses_a_missing_or_off_grid_frozen_n(n):
    selection = {ds: {"anchors": 4} for ds in M.ANCHOR_DATASETS}
    selection["mscoco"]["anchors"] = n
    with pytest.raises(CellRefused, match="no frozen N"):
        M.anchor_confirmation_cells("decide", incumbent=INCUMBENT, arms=("anchors",),
                                    selection=selection)


# ---- pinned authorities ------------------------------------------------------------------------
def write_json(path, obj):
    raw = json.dumps(obj).encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def fake_authorities(tmp_path, monkeypatch, *, disagree=False):
    aggregate = {"datasets": {ds: {"N": v["N"], "recipe": {
        "routing_adaptive_topp_min": v["topp"][0], "routing_adaptive_topp_max": v["topp"][1],
        "lambda_codon_joint": v["joint"]}} for ds, v in INCUMBENT.items()}}
    selected = {"recipe_authority": {"choices": {ds: {"topp": list(v["topp"]), "joint": v["joint"]}
                                                 for ds, v in INCUMBENT.items()}},
                "selected": {ds: {"selected_N": v["N"] + (5 if disagree and ds == "nuswide" else 0)}
                             for ds, v in INCUMBENT.items()}}
    agg, sel = tmp_path / "aggregate.json", tmp_path / "selected_n.json"
    monkeypatch.setattr(M, "APPROVED_P3_REFIT_AGGREGATE", agg)
    monkeypatch.setattr(M, "APPROVED_P3_REFIT_AGGREGATE_SHA256", write_json(agg, aggregate))
    monkeypatch.setattr(M, "APPROVED_SELECTED_N", sel)
    monkeypatch.setattr(M, "APPROVED_SELECTED_N_SHA256", write_json(sel, selected))
    return agg, sel


def test_incumbent_is_read_from_the_pinned_bytes(tmp_path, monkeypatch):
    fake_authorities(tmp_path, monkeypatch)
    assert M.anchor_incumbent() == INCUMBENT


def test_the_constants_are_the_ledger_pins():
    assert M.APPROVED_P3_REFIT_AGGREGATE_SHA256.startswith("b4f3b0dff467f7c1")
    assert M.APPROVED_SELECTED_N_SHA256.startswith("2bf6133d8cdc7471")


@pytest.mark.parametrize("which", ["aggregate", "selected"])
def test_replaced_authority_bytes_refuse(tmp_path, monkeypatch, which):
    agg, sel = fake_authorities(tmp_path, monkeypatch)
    target = agg if which == "aggregate" else sel
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(CellRefused, match="pinned bytes"):
        M.anchor_incumbent()


def test_absent_authority_refuses(tmp_path, monkeypatch):
    agg, _ = fake_authorities(tmp_path, monkeypatch)
    agg.unlink()
    with pytest.raises(CellRefused, match="unreadable"):
        M.anchor_incumbent()


def test_disagreeing_authorities_refuse(tmp_path, monkeypatch):
    fake_authorities(tmp_path, monkeypatch, disagree=True)
    with pytest.raises(CellRefused, match="disagree"):
        M.anchor_incumbent()


def test_frozen_selection_record_is_pinned_and_typed(tmp_path):
    record = {"artifact_kind": "anchor_confirmation_n_selection", "version": M.ANCHOR_CONFIRM_VERSION,
              "n_selected": {"flickr25k": {"anchors": 4}}}
    path = tmp_path / "sel.json"
    sha = write_json(path, record)
    assert M._load_anchor_selection(path, sha) == record["n_selected"]
    with pytest.raises(CellRefused, match="needs"):
        M._load_anchor_selection(path, None)
    with pytest.raises(CellRefused, match="pinned bytes"):
        M._load_anchor_selection(path, "0" * 64)
    record["artifact_kind"] = "phase3_selected_n"
    with pytest.raises(CellRefused, match="not an"):
        M._load_anchor_selection(path, write_json(path, record))


# ---- the entry point: refusals and a side-effect-free plan ---------------------------------------
@pytest.fixture
def boundaries(monkeypatch):
    """Every boundary that reserves, writes, leases or launches records the call instead."""
    calls = []
    for name in ("_with_campaign_gpu_leases", "_run_sweep", "_run_managed_process",
                 "reserve_sweep_namespace", "_publish_json_exclusive"):
        monkeypatch.setattr(M, name, lambda *a, _n=name, **k: calls.append(_n) or 0)
    return calls


def fake_render(cmd, env):
    """Stand-in for the dataset-script render: the protocol flags as the trainer argv."""
    return ["--tag", env["TAG"], "--dataset", "Flickr25k", "--setting", "1", *extra(env)]


def run_main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["phase3_selection_matrix.py", *argv])
    return M.main()


@pytest.mark.parametrize("argv", [
    ["--anchor-confirm", "select", "--namespace", "ancT", "--refit"],
    ["--anchor-confirm", "select", "--namespace", "ancT", "--sweep", "lambda"],
    ["--anchor-confirm", "select", "--namespace", "ancT", "--at-topp", "0.6,0.95"],
    ["--anchor-confirm", "select", "--namespace", "ancT", "--recipe", "/x"],
    ["--anchor-confirm", "select", "--namespace", "ancT", "--stability-plan", "/x"],
    ["--anchor-confirm", "select", "--namespace", "p3gE", "--plan"],
    ["--anchor-confirm", "select", "--namespace", "ancT", "--run", "--only", "flickr25k:4:anchors:42"],
    ["--anchor-confirm", "select", "--namespace", "ancT", "--smoke"],
    ["--anchor-confirm", "decide", "--namespace", "ancT", "--plan"],
    ["--anchor-confirm", "select", "--namespace", "ancT", "--anchor-selection", "/x", "--plan"],
    ["--anchor-confirm", "select", "--namespace", "ancT", "--anchor-arms", "anchors,both", "--plan"],
])
def test_refused_requests_never_reach_a_boundary(tmp_path, monkeypatch, boundaries, argv):
    fake_authorities(tmp_path, monkeypatch)
    monkeypatch.setattr(M, "render_trainer_argv", fake_render)
    assert run_main(monkeypatch, *argv) == 2
    assert boundaries == []


def test_plan_renders_and_checks_but_writes_reserves_and_launches_nothing(
        tmp_path, monkeypatch, boundaries, capsys):
    fake_authorities(tmp_path, monkeypatch)
    monkeypatch.setattr(M, "render_trainer_argv", fake_render)
    monkeypatch.setattr(M, "anchor_incumbent_recipe_check", lambda ds, inc: {
        "dataset": ds, "approved_record": "r.json", "approved_config_sha256": "0" * 64,
        "approved_input_authority": {}, "differences": {}})
    before = sorted(p.name for p in tmp_path.iterdir())
    assert run_main(monkeypatch, "--anchor-confirm", "select", "--namespace", "ancT", "--plan") == 0
    out = capsys.readouterr().out
    assert "NON-EXECUTABLE" in out and "axis_center alone at all 12 coordinates" in out
    assert boundaries == [] and sorted(p.name for p in tmp_path.iterdir()) == before


def test_plan_refuses_arms_that_differ_in_more_than_the_axis(tmp_path, monkeypatch, boundaries):
    fake_authorities(tmp_path, monkeypatch)
    def drifting(cmd, env):
        argv = fake_render(cmd, env)
        return argv + (["--lambda_bu", "0.5"] if "anchors" in argv else [])
    monkeypatch.setattr(M, "render_trainer_argv", drifting)
    assert run_main(monkeypatch, "--anchor-confirm", "select", "--namespace", "ancT", "--plan") == 2
    assert boundaries == []


def test_plan_refuses_a_control_recipe_that_drifted_from_the_approved_one(
        tmp_path, monkeypatch, boundaries):
    fake_authorities(tmp_path, monkeypatch)
    monkeypatch.setattr(M, "render_trainer_argv", fake_render)
    monkeypatch.setattr(M, "anchor_incumbent_recipe_check", lambda ds, inc: {
        "dataset": ds, "approved_record": "r.json", "approved_config_sha256": "0" * 64,
        "approved_input_authority": {},
        "differences": {"lambda_wasserstein": {"rendered": 0.3, "approved": 0.15}}})
    assert run_main(monkeypatch, "--anchor-confirm", "select", "--namespace", "ancT", "--plan") == 2
    assert boundaries == []


def test_run_cell_refuses_an_anchor_cell_without_a_sealed_recipe(monkeypatch, boundaries):
    with pytest.raises(CellRefused, match="sealed scientific recipe"):
        M.run_cell("flickr25k", 4, 0, namespace="ancT", topp=("0.6", "0.95"), joint="0.02",
                   anchor_arm="anchors", campaign_binding=None, scientific_recipe=None)
    assert boundaries == []


# ---- completion --------------------------------------------------------------------------------
@pytest.fixture
def sealed():
    from config import Config
    argv = ["--tag", "t", "--dataset", "Flickr25k", "--setting", "1", "--axis_center", "anchors"]
    return R.build_payload(Config.build_parser(), argv), argv


def completed_run(tmp_path, payload, argv, *, evidence_digest=None, mutate=None):
    import torch
    from config import Config
    values = vars(Config.build_parser().parse_args(argv))
    values["setting"] = "setting" + str(values["setting"])
    values.pop("log_dir", None)
    values["_phase3_campaign_binding"] = {"cell_id": "c"}
    if mutate:
        mutate(values)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    torch.save(values, run_dir / "config.pt")
    from dna_utils.run_identity import PHASE3_CAMPAIGN_BINDING_NAME
    (run_dir / PHASE3_CAMPAIGN_BINDING_NAME).write_text(json.dumps({
        "scientific_recipe_schema": payload["schema"],
        "scientific_recipe_sha256": evidence_digest or R.digest(payload)}))
    return run_dir


def test_completion_accepts_the_sealed_recipe(tmp_path, sealed):
    payload, argv = sealed
    got = M.assert_anchor_recipe(completed_run(tmp_path, payload, argv), scientific_recipe=payload,
                                 arm="anchors")
    assert got["scientific_recipe_sha256"] == R.digest(payload) and got["arm"] == "anchors"


@pytest.mark.parametrize("case", ["evidence_digest", "config_value", "arm", "no_config"])
def test_completion_refuses_what_is_not_the_sealed_recipe(tmp_path, sealed, case):
    payload, argv = sealed
    kw = {"evidence_digest": "f" * 64} if case == "evidence_digest" else {}
    if case == "config_value":
        kw["mutate"] = lambda v: v.update(lambda_bu=0.5)
    run_dir = completed_run(tmp_path, payload, argv, **kw)
    if case == "no_config":
        (run_dir / "config.pt").unlink()
    with pytest.raises(CellRefused):
        M.assert_anchor_recipe(run_dir, scientific_recipe=payload,
                               arm="none" if case == "arm" else "anchors")


# ---- integration against the real approved artifacts (read-only) --------------------------------
@pytest.mark.skipif(not M.APPROVED_P3_REFIT_AGGREGATE.exists(), reason="approved artifacts absent")
def test_real_plan_reproduces_the_approved_control_recipe(monkeypatch, boundaries, capsys):
    assert run_main(monkeypatch, "--anchor-confirm", "select", "--namespace", "ancT", "--plan") == 0
    out = capsys.readouterr().out
    assert out.count("identical outside the 11 sealed-input fields") == 3
    assert boundaries == []


# ---- rendering refuses where the script would do real work -------------------------------------------
def test_render_refuses_a_missing_whitening_file_instead_of_letting_the_script_build_it(tmp_path):
    cmd, env, _ = M.build_command("flickr25k", 4, 0, topp=("0.6", "0.95"), joint="0.02",
                                  anchor_arm="anchors")
    env = dict(env, WHITEN_NPZ=str(tmp_path / "absent.npz"))
    with pytest.raises(CellRefused, match="missing"):
        M.render_trainer_argv(cmd, env)
