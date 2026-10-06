"""Anchor confirmation v2 (contract v3): the launcher mode (audit sections 659.4, 668, 671, 677.2, 678, 709).

No real input, config, checkpoint, model or GPU is touched. Leases, reservations, writers, process
launch and the historical inspection are sentinels. Most tests use a stand-in renderer; the
"actual composition" tests execute the PINNED dataset wrappers under bash with the capture shim, in
temporary directories, with every input path pointed into tmp_path (audit 675.2 describes what that
execution is). The one test that touches real approved artifacts is opt-in
(GDNA_ALLOW_REAL_ARTIFACT_TESTS=1) and is not part of CPU or mutation runs (audit 673.3).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import scripts.phase3_selection_matrix as M                 # noqa: E402
from scripts.phase3_selection_matrix import CellRefused     # noqa: E402
REAL_VERIFY_CAMPAIGN_INPUT_SEALS = M.verify_campaign_input_seals
import dna_utils.scientific_recipe as R                     # noqa: E402

INCUMBENT = {"cifar10": {"N": 19, "topp": ("0.3", "0.7"), "joint": "0.02"},
             "flickr25k": {"N": 4, "topp": ("0.6", "0.95"), "joint": "0.02"},
             "nuswide": {"N": 4, "topp": ("0.4", "0.8"), "joint": "0.05"},
             "mscoco": {"N": 39, "topp": ("0.6", "0.95"), "joint": "0.03"}}
ALL_ANCHORS = {ds: ("anchors",) for ds in M.ANCHOR_DATASETS}


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


def test_the_refit_path_takes_an_anchor_arm_only_without_a_lambda_override():
    # generation v9 (audits 743-744): rendering an anchor refit is stage R's, whose authority is the
    # stage-R admission (tests/test_anchor_refit_stage.py); it never carries a lambda override
    cmd, env, tag = M.build_command("flickr25k", 4, 0, stage="refit", anchor_arm="anchors")
    assert extra(env)[-2:] == ["--axis_center", "anchors"] and "_AXanchors" in tag
    with pytest.raises(CellRefused, match="no lambda override"):
        M.build_command("flickr25k", 4, 0, stage="refit", anchor_arm="anchors",
                        overrides=(("lambda_bu", "0"),))


@pytest.mark.parametrize("dataset,overrides", [
    ("flickr25k", (("lambda_wasserstein", "0.15"),)),                 # the incumbent value
    ("flickr25k", (("lambda_wasserstein", "0.40"),)),                 # an undeclared value
    ("flickr25k", (("lambda_bu", "0"), ("lambda_wasserstein", "0.30"))),   # two lambdas
    ("nuswide", (("lambda_wasserstein", "0.30"),)),                   # outside the stage-L scope
    ("mscoco", (("lambda_bu", "0"),)),
])
def test_an_anchor_cell_moves_axis_center_alone_or_one_declared_stage_l_lambda(dataset, overrides):
    """Generation v8: an anchor cell carries no override except one declared stage-L candidate of a
    scope dataset (contract L v1); every other override still refuses."""
    with pytest.raises(CellRefused, match="declared stage-L lambda candidate"):
        M.build_command(dataset, 4, 0, anchor_arm="anchors", overrides=overrides)


def test_cell_ids_carry_the_arm_and_leave_legacy_ids_unchanged():
    kw = dict(topp=("0.6", "0.95"), joint="0.02")
    legacy = M.campaign_cell_id("flickr25k", 4, **kw)
    assert "axis_center" not in legacy
    ids = {M.campaign_cell_id("flickr25k", 4, anchor_arm=a, **kw) for a in M.ANCHOR_ARMS}
    assert ids == {legacy + "|axis_center=none", legacy + "|axis_center=anchors"}
    assert {R.anchor_arm_of(i) for i in ids} == set(M.ANCHOR_ARMS) and R.anchor_arm_of(legacy) is None


def test_eight_tuples_are_anchor_cells_and_nothing_else_changes():
    cell = ("flickr25k", 4, ("0.6", "0.95"), "0.02", "select", 42, (), "anchors")
    assert M._campaign_cell_parts(cell) == cell[:6]
    assert M._cell_overrides(cell) == () and M._cell_anchor_arm(cell) == "anchors"
    assert M._cell_anchor_arm(cell[:7]) is None and M._cell_anchor_arm(cell[:6]) is None
    with pytest.raises(CellRefused, match="declared stage-L"):        # generation v8: undeclared
        M._cell_overrides(cell[:6] + ((("lambda_bu", "0.5"),), "anchors"))
    with pytest.raises(CellRefused, match="declared stage-L"):        # outside the stage-L scope
        M._cell_overrides(("nuswide",) + cell[1:6] + ((("lambda_bu", "0"),), "anchors"))
    with pytest.raises(CellRefused):
        M._cell_anchor_arm(cell[:7] + ("both",))


# ---- arm plans and cells ---------------------------------------------------------------------------
def test_the_arm_plan_is_the_anchor_arm_of_all_four_datasets():
    assert M.ANCHOR_DATASETS == ("cifar10", "flickr25k", "nuswide", "mscoco")
    assert M.ANCHOR_RUN_ARMS == ("anchors",)
    assert M.parse_arm_plan(None, "anchors") == ALL_ANCHORS


@pytest.mark.parametrize("cells,arms,reason", [
    ("cifar10:anchors", "anchors", "not part of contract v3"),       # a per-dataset pair list
    ("nuswide:none,mscoco:none", "anchors", "not part of contract v3"),
    (None, "none", "runs the arms"), (None, "none,anchors", "runs the arms"),   # a control arm
    (None, "anchors,anchors", "runs the arms"), (None, "readout", "runs the arms"), (None, "", "runs the arms")])
def test_invalid_arm_plans_refuse(cells, arms, reason):
    with pytest.raises(CellRefused, match=reason):
        M.parse_arm_plan(cells, arms)


@pytest.mark.parametrize("plan", [
    {ds: ("anchors",) for ds in ("flickr25k", "nuswide", "mscoco")},          # the v1 scope, CIFAR dropped
    {**{ds: ("anchors",) for ds in M.ANCHOR_DATASETS}, "cifar10": ("none",)},   # a control arm
    {**{ds: ("anchors",) for ds in M.ANCHOR_DATASETS}, "cifar10": ("anchors", "none")}])
def test_a_plan_that_drops_a_dataset_or_runs_a_control_refuses(plan):
    with pytest.raises(CellRefused, match="must run"):
        M.anchor_confirmation_cells("select", incumbent=INCUMBENT, arm_plan=plan)


def test_select_cells_cover_the_four_datasets_and_the_grid():
    cells = M.anchor_confirmation_cells("select", incumbent=INCUMBENT, arm_plan=ALL_ANCHORS)
    assert len(cells) == 16 and {c[0] for c in cells} == set(M.ANCHOR_DATASETS)
    assert {c[7] for c in cells} == {"anchors"}
    assert sorted({c[1] for c in cells}) == list(M.CANDIDATE_N)
    assert {c[5] for c in cells} == {42} and {c[4] for c in cells} == {"select"}
    assert all((c[2], c[3]) == (INCUMBENT[c[0]]["topp"], INCUMBENT[c[0]]["joint"]) for c in cells)


def test_decide_cells_take_the_anchor_arm_at_its_frozen_n():
    frozen = {"cifar10": {"anchors": 4}, "flickr25k": {"anchors": 9}, "nuswide": {"anchors": 4},
              "mscoco": {"anchors": 39}}
    cells = M.anchor_confirmation_cells("decide", incumbent=INCUMBENT, arm_plan=ALL_ANCHORS, selection=frozen)
    assert len(cells) == 8                                    # 4 datasets x seeds 43/44
    assert all(c[7] == "anchors" and c[1] == frozen[c[0]]["anchors"] and c[5] in (43, 44) for c in cells)


@pytest.mark.parametrize("n", [None, 5, "4", 4.0, True])
def test_decide_refuses_a_missing_or_off_grid_frozen_n(n):
    frozen = {ds: {"anchors": 4} for ds in M.ANCHOR_DATASETS}
    frozen["mscoco"]["anchors"] = n
    with pytest.raises(CellRefused, match="no frozen N"):
        M.anchor_confirmation_cells("decide", incumbent=INCUMBENT, arm_plan=ALL_ANCHORS, selection=frozen)


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


def test_disagreeing_authorities_refuse(tmp_path, monkeypatch):
    fake_authorities(tmp_path, monkeypatch, disagree=True)
    with pytest.raises(CellRefused, match="disagree"):
        M.anchor_incumbent()


# ---- rendering guards (audit 671.3) -----------------------------------------------------------------
@pytest.fixture
def no_process(monkeypatch):
    """An independent deny guard (audit 688.1): a refusal test must refuse before any wrapper
    process starts, so a deleted product guard is detected here and cannot reach a real builder."""
    started = []
    def deny(*a, **k):
        started.append(a[0] if a else k.get("args"))
        raise AssertionError("a process was started")
    monkeypatch.setattr(M.subprocess, "run", deny)
    return started


def test_render_refuses_a_script_that_is_not_the_pinned_bytes(monkeypatch, synthetic_paths, no_process):
    cmd, env, _ = M.build_command("flickr25k", 4, 0, topp=("0.6", "0.95"), joint="0.02",
                                  anchor_arm="anchors")
    monkeypatch.setitem(M.DATASET_SCRIPT_SHA256, cmd[1], "0" * 64)
    with pytest.raises(CellRefused, match="not a pinned dataset script"):
        M.render_trainer_argv(cmd, env)
    assert no_process == []


def test_render_refuses_a_missing_whitening_file_instead_of_letting_the_script_build_it(tmp_path,
                                                                                        synthetic_paths,
                                                                                        no_process):
    cmd, env, _ = M.build_command("flickr25k", 4, 0, topp=("0.6", "0.95"), joint="0.02",
                                  anchor_arm="anchors")
    with pytest.raises(CellRefused, match="missing"):
        M.render_trainer_argv(cmd, dict(env, WHITEN_NPZ=str(tmp_path / "absent.npz")))
    assert no_process == []


def test_the_pinned_scripts_are_the_current_bytes():
    for rel, want in M.DATASET_SCRIPT_SHA256.items():
        assert hashlib.sha256((REPO / rel).read_bytes()).hexdigest() == want


# ---- the generation manifest (audit 678.2, 683.2) ---------------------------------------------------------
def manifest(tmp_path, **changes):
    """A manifest of THIS tree: the exact generation closure, the designated contract, the process's
    environment. `changes` set "a::b::c" keys (file names contain dots)."""
    from importlib import metadata
    files = {rel: hashlib.sha256((REPO / rel).read_bytes()).hexdigest() for rel in M.anchor_generation_closure()}
    body = {"artifact_kind": M.ANCHOR_MANIFEST_KIND, "version": M.ANCHOR_CONFIRM_VERSION,
            "historical": {"approved_p3_refit_aggregate": {"sha256": M.APPROVED_P3_REFIT_AGGREGATE_SHA256},
                           "approved_selected_n": {"sha256": M.APPROVED_SELECTED_N_SHA256},
                           "historical_input_verifier": M.historical_input_verifier_pins(),
                           "anchor_v7": M.anchor_v7_pins()},
            "new_generation": {"commit": "c" * 40, "branch": "arch-exp-2026-09-anchor-confirm", "clean": True,
                               "files_sha256": files, "dataset_scripts_sha256": dict(M.DATASET_SCRIPT_SHA256),
                               "environment": {"python": ".".join(map(str, sys.version_info[:3])),
                                               "torch": metadata.version("torch"),
                                               "interpreter": sys.executable}},
            "contract": {"path": M.ANCHOR_CONTRACT_PATH, "sha256": files[M.ANCHOR_CONTRACT_PATH]},
            "lambda_contract": {"path": M.ANCHOR_LAMBDA_CONTRACT_PATH,
                                "sha256": files[M.ANCHOR_LAMBDA_CONTRACT_PATH]},
            "refit_contract": {"path": M.ANCHOR_REFIT_CONTRACT_PATH,
                               "sha256": files[M.ANCHOR_REFIT_CONTRACT_PATH]},
            "recovery_contract": {"path": M.ANCHOR_RECOVERY_CONTRACT_PATH,
                                  "sha256": files[M.ANCHOR_RECOVERY_CONTRACT_PATH]}}
    for key, value in changes.items():
        target = body
        *parents, leaf = key.split("::")
        for part in parents:
            target = target[part]
        if value is DELETE:
            del target[leaf]
        else:
            target[leaf] = value
    path = tmp_path / "manifest.json"
    return str(path), write_json(path, body)


DELETE = object()


def test_a_manifest_of_this_tree_loads(tmp_path):
    path, sha = manifest(tmp_path)
    got = M.load_anchor_manifest(path, sha)
    assert got["sha256"] == sha and got["contract"]["path"] == M.ANCHOR_CONTRACT_PATH


def test_the_closure_covers_wrappers_input_admission_measurement_tests_and_contract():
    closure = set(M.anchor_generation_closure())
    assert set(M.DATASET_SCRIPT_SHA256) <= closure and M.ANCHOR_CONTRACT_PATH in closure
    assert {"scripts/seal_phase3_inputs.py", "dna_utils/run_identity.py", "val_split.py",
            "scripts/anchor_confirm_code_axis.py", "scripts/anchor_confirm_decision.py",
            "tests/test_anchor_confirm_reducer.py", "tests/test_anchor_confirm_input_bridge.py",
            "tests/test_seal_phase3_inputs.py", "tests/test_anchor_confirm_env_handoff.py"} <= closure


@pytest.mark.parametrize("changes,reason", [
    ({"artifact_kind": "anchor_confirmation_reuse_proposal"}, "is not an"),
    ({"new_generation::files_sha256::config.py": "0" * 64}, "not the manifest's generation"),
    ({"new_generation::files_sha256::train_siglip2.py": DELETE}, "does not list the generation closure"),
    ({"new_generation::files_sha256::README.md": "0" * 64}, "does not list the generation closure"),
    ({"new_generation::files_sha256::scripts/train_mscoco_F2_sweep_clip.sh": DELETE},
     "does not list the generation closure"),                    # a wrapper cannot be left out
    ({"new_generation::dataset_scripts_sha256": {}}, "dataset-script pins"),
    ({"historical::approved_p3_refit_aggregate::sha256": "0" * 64}, "approved-aggregate pin"),
    ({"historical::historical_input_verifier::verifier::sha256": "0" * 64},
     "historical input-verifier pins"),                          # audit 715
    ({"historical::historical_input_verifier::root": "/data/yschoi/elsewhere"},
     "historical input-verifier pins"),
    ({"historical::historical_input_verifier": DELETE}, "historical input-verifier pins"),
    ({"historical::approved_selected_n::sha256": "0" * 64}, "selected-N pin"),
    ({"historical::anchor_v7::decision::sha256": "0" * 64}, "generation-v7 pins"),    # generation v8
    ({"historical::anchor_v7": DELETE}, "generation-v7 pins"),
    ({"lambda_contract::sha256": "0" * 64}, "stage-L contract"),
    ({"lambda_contract::path": M.ANCHOR_CONTRACT_PATH}, "stage-L contract"),
    ({"lambda_contract": DELETE}, "stage-L contract"),
    ({"contract::sha256": "0" * 64}, "not the designated"),
    ({"contract::path": "config.py"}, "not the designated"),   # a listed file substituted as contract
    ({"new_generation::commit": "not-a-commit"}, "commit, branch or clean flag"),
    ({"new_generation::clean": False}, "commit, branch or clean flag"),
    ({"new_generation::branch": ""}, "commit, branch or clean flag"),
    ({"new_generation::environment::python": "2.7.18"}, "environment"),
    ({"new_generation::environment": DELETE}, "environment"),
])
def test_a_manifest_that_is_not_this_generation_refuses(tmp_path, changes, reason):
    path, sha = manifest(tmp_path, **changes)
    with pytest.raises(CellRefused, match=reason):
        M.load_anchor_manifest(path, sha)


def test_both_historical_pins_wrong_refuses(tmp_path):
    path, sha = manifest(tmp_path, **{"historical::approved_p3_refit_aggregate::sha256": "0" * 64,
                                      "historical::approved_selected_n::sha256": "0" * 64})
    with pytest.raises(CellRefused, match="approved-aggregate pin"):
        M.load_anchor_manifest(path, sha)


def test_a_substituted_contract_at_its_correct_hash_refuses(tmp_path):
    """Audit 683.2's fixture: config.py named as the contract, at its own correct digest."""
    path, sha = manifest(tmp_path, **{"contract::path": "config.py", "contract::sha256":
                                      hashlib.sha256((REPO / "config.py").read_bytes()).hexdigest()})
    with pytest.raises(CellRefused, match="not the designated"):
        M.load_anchor_manifest(path, sha)


def test_a_manifest_is_read_at_its_pinned_bytes(tmp_path):
    path, _ = manifest(tmp_path)
    with pytest.raises(CellRefused, match="pinned bytes"):
        M.load_anchor_manifest(path, "0" * 64)


# ---- the approval authority: the audit ledger (audit 679.1, 683.2) -------------------------------------
def approval_line(scope, **pins):
    return " ".join([M.APPROVAL_TAG, f"version={M.ANCHOR_CONFIRM_VERSION}", f"scope={scope}",
                     *(f"{k}={v}" for k, v in pins.items())])


def ledger(tmp_path, monkeypatch, sections: dict) -> None:
    """A synthetic ledger: {section number: [lines]}; lines are written in backticks, as markdown."""
    text = ["# Synthetic audit ledger", ""]
    for n, lines in sections.items():
        text += [f"## {n}. Test decision", "", *(f"`{line}`" for line in lines), ""]
    path = tmp_path / "ledger.md"
    path.write_text("\n".join(text) + "\n")
    monkeypatch.setattr(M, "AUDIT_LEDGER", path)


A, Q = "a" * 64, "q" * 64          # a manifest digest and a request digest


def test_an_approval_line_is_found_and_bound(tmp_path, monkeypatch):
    ledger(tmp_path, monkeypatch, {5: ["prose", approval_line("stage-S-run", manifest=A, request=Q)]})
    got = M.audit_approval(5, "stage-S-run", manifest=A, request=Q)
    assert got["section"] == 5 and got["line"] == approval_line("stage-S-run", manifest=A, request=Q)
    assert got["ledger_sha256"] == hashlib.sha256((tmp_path / "ledger.md").read_bytes()).hexdigest()


RUN = approval_line("stage-S-run", manifest=A, request=Q)


@pytest.mark.parametrize("sections,section,scope,pins,reason", [
    ({5: [RUN]}, None, "stage-S-run", {"manifest": A, "request": Q}, "positive ledger section number"),
    ({5: [RUN]}, "5", "stage-S-run", {"manifest": A, "request": Q}, "positive ledger section number"),
    ({5: [RUN]}, True, "stage-S-run", {"manifest": A, "request": Q}, "positive ledger section number"),
    ({5: [RUN]}, 42, "stage-S-run", {"manifest": A, "request": Q}, "has 0 sections numbered 42"),
    ({5: []}, 5, "stage-S-run", {"manifest": A, "request": Q}, "has 0 approval lines"),
    ({5: [RUN, RUN]}, 5, "stage-S-run", {"manifest": A, "request": Q}, "has 2 approval lines"),
    ({5: [approval_line("stage-S-smoke", manifest=A, request=Q)]}, 5, "stage-S-run",
     {"manifest": A, "request": Q}, "has 0 approval lines for stage-S-run"),             # wrong scope
    ({5: [approval_line("stage-S-run", manifest="b" * 64, request=Q)]}, 5, "stage-S-run",
     {"manifest": A, "request": Q}, "approves stage-S-run for"),                         # stale generation
    ({5: [approval_line("stage-S-run", manifest=A, request="r" * 64)]}, 5, "stage-S-run",
     {"manifest": A, "request": Q}, "approves stage-S-run for"),                         # another request
    ({5: [approval_line("stage-D-run", manifest=A, selection="c" * 64, request=Q)]}, 5, "stage-D-run",
     {"manifest": A, "selection": "d" * 64, "request": Q}, "approves stage-D-run for"),  # another frozen N
    ({5: [approval_line("stage-D-run", manifest=A, request=Q)]}, 5, "stage-D-run",
     {"manifest": A, "selection": A, "request": Q}, "approves stage-D-run for"),         # missing field
    ({5: [approval_line("stage-S-run", manifest=A, request=Q, gpus="0")]}, 5, "stage-S-run",
     {"manifest": A, "request": Q}, "approves stage-S-run for"),                         # extra field
    ({5: [RUN.replace(M.ANCHOR_CONFIRM_VERSION, "anchor-confirm/1")]}, 5, "stage-S-run",
     {"manifest": A, "request": Q}, "approves stage-S-run for"),                         # a v1 (3-dataset) line
    ({5: [RUN + " approved"]}, 5, "stage-S-run", {"manifest": A, "request": Q}, "malformed approval line"),
    ({5: [RUN + " manifest=" + A]}, 5, "stage-S-run", {"manifest": A, "request": Q}, "a field repeats"),
    ({4: [RUN], 5: []}, 5, "stage-S-run", {"manifest": A, "request": Q}, "has 0 approval lines"),
    ({5: [RUN]}, 5, "stage-S-run", {"manifest": A}, "needs exactly"),                    # caller pins incomplete
    ({5: [RUN]}, 5, "stage-X", {"manifest": A, "request": Q}, "needs exactly"),
])
def test_an_approval_that_is_not_exactly_this_operation_refuses(tmp_path, monkeypatch, sections, section,
                                                                scope, pins, reason):
    ledger(tmp_path, monkeypatch, sections)
    with pytest.raises(CellRefused, match=reason):
        M.audit_approval(section, scope, **pins)


def test_a_duplicated_section_number_refuses(tmp_path, monkeypatch):
    path = tmp_path / "ledger.md"
    path.write_text(f"## 5. One\n`{RUN}`\n## 5. Two\n`{RUN}`\n")
    monkeypatch.setattr(M, "AUDIT_LEDGER", path)
    with pytest.raises(CellRefused, match="has 2 sections numbered 5"):
        M.audit_approval(5, "stage-S-run", manifest=A, request=Q)


# ---- the entry point: one admission path for plan, smoke and run (audit 671.1) ----------------------
@pytest.fixture
def boundaries(monkeypatch):
    """Every boundary that reserves, writes, leases, launches or opens a historical binary records
    the call instead."""
    calls = []
    for name in ("_with_campaign_gpu_leases", "_run_sweep", "_run_managed_process",
                 "reserve_sweep_namespace", "_publish_json_exclusive"):
        monkeypatch.setattr(M, name, lambda *a, _n=name, **k: calls.append(_n) or 0)
    # not a side effect, but its place in the order is asserted: before any lease (audit 671.1)
    monkeypatch.setattr(M, "verify_campaign_input_seals",
                        lambda *a, **k: calls.append("verify_campaign_input_seals") or {})
    return calls


#: the shape of a pinned wrapper body: the seven reviewed literals and the recipe invariants
WRAPPER_BODY = ["--codebook_size", "128", "-e", "60", "--routing_adaptive_topp",
                "--routing_adaptive_topp_min", "0.3", "--routing_adaptive_topp_max", "0.7",
                "--hash_target_mode", "siglip_cos", "--lambda_codeword_codon_sinkhorn", "0.0",
                "--post_eval_compositional", "--dna_distance_mode", "base"]


def wrapper_dataset(cmd):
    return next(ds for ds, spec in M.DATASETS.items() if spec["trainer"] == cmd[1])


def fake_render(cmd, env):
    """Stand-in for the dataset-script render, without bash: a wrapper body carrying the reviewed
    literals (the dataset's K; CIFAR-10's codon-Sinkhorn literal 0.1), the wrapper's env-driven
    flags, then EXTRA_ARGS."""
    ds = wrapper_dataset(cmd)
    body = list(WRAPPER_BODY)
    body[body.index("--codebook_size") + 1] = str(M.DATASETS[ds]["K"])
    if ds == "cifar10":
        body[body.index("--lambda_codeword_codon_sinkhorn") + 1] = "0.1"
    env_flags = ["--val_split_ratio", env["VAL_RATIO"], "--val_split_seed", env["VAL_SEED"],
                 "--stop_after_epoch", env["STOP_EP"], "--num_codons_per_codebook", env["NUM_CODONS"]]
    return ["--tag", env["TAG"], "--dataset", M.DATASETS[ds]["canon"], "--setting", "1", *body,
            *env_flags, *extra(env)]


def drifting_render(cmd, env):
    argv = fake_render(cmd, env)
    return argv + (["--lambda_bu", "0.5"] if "anchors" in argv else [])


def run_main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["phase3_selection_matrix.py", *argv])
    return M.main()


BASE = ["--anchor-confirm", "select", "--namespace", "ancT"]
#: one stream per dataset, one GPU per stream (audit 709.2)
RUNARGS = ["--run", "--gpus", "0,1,2,3"]


def with_manifest(tmp_path):
    path, sha = manifest(tmp_path)
    return ["--anchor-manifest", path, "--anchor-manifest-sha256", sha]


MODES = [["--plan"], ["--smoke", "--only", "flickr25k:4:anchors:42"], RUNARGS]


def approve_request(tmp_path, monkeypatch, capsys, argv, *, manifest_sha=None, section=700):
    """Preview the exact request with `--plan`, then write the ledger line approving that digest
    for the scope the command implies (what the audit would do)."""
    assert run_main(monkeypatch, *argv, "--plan") == 0
    digest = re.search(r"execution request sha256 ([0-9a-f]{64})", capsys.readouterr().out).group(1)
    sha = manifest_sha or argv[argv.index("--anchor-manifest-sha256") + 1]
    scope = "stage-S-smoke" if "--smoke" in argv else "stage-S-run"
    ledger(tmp_path, monkeypatch, {section: [approval_line(scope, manifest=sha, request=digest)]})
    return ["--anchor-approval-section", str(section)]


@pytest.mark.parametrize("mode", MODES)
def test_an_invalid_scientific_request_refuses_in_every_mode_before_any_boundary(
        tmp_path, monkeypatch, boundaries, capsys, mode):
    fake_authorities(tmp_path, monkeypatch)
    monkeypatch.setattr(M, "render_trainer_argv", drifting_render)
    assert run_main(monkeypatch, *BASE, *mode, *with_manifest(tmp_path)) == 2
    assert "not in axis_center alone" in capsys.readouterr().err
    assert boundaries == []


def unsupervised_breaking_render(cmd, env):
    argv = fake_render(cmd, env)
    i = argv.index("--hash_target_mode")
    return argv[:i] + argv[i + 2:]                          # the parser default is the tag-supervised jaccard


@pytest.mark.parametrize("mode", MODES)
def test_a_protocol_value_off_the_contract_refuses_in_every_mode(tmp_path, monkeypatch, boundaries,
                                                                  capsys, mode):
    fake_authorities(tmp_path, monkeypatch)
    monkeypatch.setattr(M, "render_trainer_argv", unsupervised_breaking_render)
    assert run_main(monkeypatch, *BASE, *mode, *with_manifest(tmp_path)) == 2
    assert "breaks the protocol in hash_target_mode='jaccard'" in capsys.readouterr().err
    assert boundaries == []


@pytest.mark.parametrize("mode", MODES[1:])
def test_an_approved_execution_verifies_its_inputs_then_reaches_only_the_lease(tmp_path, monkeypatch,
                                                                              boundaries, capsys, mode):
    fake_authorities(tmp_path, monkeypatch)
    monkeypatch.setattr(M, "render_trainer_argv", fake_render)
    argv = [*BASE, *mode, *with_manifest(tmp_path)]
    approval = approve_request(tmp_path, monkeypatch, capsys, argv)
    assert boundaries == []                                          # the preview touched nothing
    assert run_main(monkeypatch, *argv, *approval) == 0
    assert boundaries == ["verify_campaign_input_seals", "_with_campaign_gpu_leases"]


def test_the_plan_previews_the_request_it_would_need_approved(tmp_path, monkeypatch, boundaries, capsys):
    fake_authorities(tmp_path, monkeypatch)
    monkeypatch.setattr(M, "render_trainer_argv", fake_render)
    assert run_main(monkeypatch, *BASE, *RUNARGS, "--plan", *with_manifest(tmp_path)) == 0
    out = capsys.readouterr().out
    request = json.loads(out[out.index("{", out.index("would need approved")):out.index("execution request sha256")])
    assert request["mode"] == "run" and request["namespace"] == "ancT" and len(request["cells"]) == 16
    assert request["version"] == "anchor-confirm/2" and request["schema"] == "anchor-confirm-request/2"
    assert request["gpu_count"] == 4 and {c[0] for c in request["cells"]} == set(M.ANCHOR_DATASETS)
    assert {c[1] for c in request["cells"]} == {"anchors"}
    assert re.search(r"execution request sha256 ([0-9a-f]{64})", out).group(1) == M._json_digest(request)
    assert boundaries == []


@pytest.mark.parametrize("change,reason", [
    (["--namespace", "ancRepeat"], "approves stage-S-run for"),    # a new namespace, the old approval
    (["--result-root", "/data/elsewhere_result"], "approves stage-S-run for"),
    (["--anchor-arms", "none,anchors"], "runs the arms"),           # a control arm cannot be planned
    (["--gpus", "0,1,2"], "needs exactly that many distinct GPUs"),  # CIFAR's stream would be left out
    (["--gpus", "0,1,2,2"], "needs exactly that many distinct GPUs"),
    (["--admission-authority", "__TMP__/authority.json"], "approves stage-S-run for"),
    (["--input-seal", "flickr25k:stage1=__TMP__/seal.json"], "approves stage-S-run for"),
])
def test_a_changed_request_is_not_covered_by_the_old_approval(tmp_path, monkeypatch, boundaries, capsys,
                                                              change, reason):
    fake_authorities(tmp_path, monkeypatch)
    monkeypatch.setattr(M, "render_trainer_argv", fake_render)
    monkeypatch.setattr(M, "load_admission_authority", lambda path: {})   # its content is not the subject
    (tmp_path / "authority.json").write_text("{}")
    (tmp_path / "seal.json").write_text("{}")
    argv = [*BASE, *RUNARGS, *with_manifest(tmp_path)]
    approval = approve_request(tmp_path, monkeypatch, capsys, argv)
    changed = [a.replace("__TMP__", str(tmp_path)) for a in change]
    if changed[0] in ("--namespace", "--gpus"):
        argv[argv.index(changed[0]) + 1] = changed[1]
        changed = []
    assert run_main(monkeypatch, *argv, *changed, *approval) == 2
    assert reason in capsys.readouterr().err and boundaries == []


def test_the_superseded_three_dataset_request_cannot_be_approved(tmp_path, monkeypatch, boundaries, capsys):
    """The v4 package's line (version 1, request 033b6979, manifest 5a4481f4) approves nothing now."""
    fake_authorities(tmp_path, monkeypatch)
    monkeypatch.setattr(M, "render_trainer_argv", fake_render)
    ledger(tmp_path, monkeypatch, {709: [" ".join([
        M.APPROVAL_TAG, "version=anchor-confirm/1", "scope=stage-S-run",
        "manifest=5a4481f4898fda3df27250e0214e204495022a91546d7d78e2e1c2749e17647b",
        "request=033b697945067d587473c81551f05e145d124ea9f69914549fd7199d333eec54"])]})
    assert run_main(monkeypatch, *BASE, *RUNARGS, *with_manifest(tmp_path),
                    "--anchor-approval-section", "709") == 2
    assert "approves stage-S-run for" in capsys.readouterr().err and boundaries == []


@pytest.mark.parametrize("change", [["--epochs", "2"], ["--only", "nuswide:4:anchors:42"]])
def test_a_changed_smoke_is_not_covered_by_the_old_approval(tmp_path, monkeypatch, boundaries, capsys, change):
    fake_authorities(tmp_path, monkeypatch)
    monkeypatch.setattr(M, "render_trainer_argv", fake_render)
    argv = [*BASE, *MODES[1], *with_manifest(tmp_path)]
    approval = approve_request(tmp_path, monkeypatch, capsys, argv)
    if change[0] == "--only":
        argv[argv.index("--only") + 1] = change[1]
        change = []
    assert run_main(monkeypatch, *argv, *change, *approval) == 2
    assert "approves stage-S-smoke for" in capsys.readouterr().err and boundaries == []


@pytest.mark.parametrize("mode", MODES[1:])
def test_execution_without_an_approval_refuses_before_inputs_and_leases(tmp_path, monkeypatch, boundaries,
                                                                         capsys, mode):
    fake_authorities(tmp_path, monkeypatch)
    monkeypatch.setattr(M, "render_trainer_argv", fake_render)
    ledger(tmp_path, monkeypatch, {})
    assert run_main(monkeypatch, *BASE, *mode, *with_manifest(tmp_path)) == 2
    assert "positive ledger section number" in capsys.readouterr().err and boundaries == []


def test_a_smoke_approval_does_not_approve_a_run(tmp_path, monkeypatch, boundaries, capsys):
    fake_authorities(tmp_path, monkeypatch)
    monkeypatch.setattr(M, "render_trainer_argv", fake_render)
    manifest_args = with_manifest(tmp_path)
    smoke = approve_request(tmp_path, monkeypatch, capsys, [*BASE, *MODES[1], *manifest_args])
    assert run_main(monkeypatch, *BASE, *RUNARGS, *manifest_args, *smoke) == 2
    assert "has 0 approval lines for stage-S-run" in capsys.readouterr().err and boundaries == []


def test_a_run_approval_for_another_generation_refuses(tmp_path, monkeypatch, boundaries, capsys):
    fake_authorities(tmp_path, monkeypatch)
    monkeypatch.setattr(M, "render_trainer_argv", fake_render)
    argv = [*BASE, *RUNARGS, *with_manifest(tmp_path)]
    stale = approve_request(tmp_path, monkeypatch, capsys, argv, manifest_sha="b" * 64)
    assert run_main(monkeypatch, *argv, *stale) == 2
    assert "approves stage-S-run for" in capsys.readouterr().err and boundaries == []


@pytest.mark.parametrize("mode", MODES[1:])
def test_an_input_seal_that_does_not_verify_refuses_before_any_lease(tmp_path, monkeypatch, boundaries,
                                                                     capsys, mode):
    fake_authorities(tmp_path, monkeypatch)
    monkeypatch.setattr(M, "render_trainer_argv", fake_render)
    def refuse(*a, **k):
        raise CellRefused("input seal flickr25k:stage1 refused: synthetic")
    monkeypatch.setattr(M, "verify_campaign_input_seals", refuse)
    argv = [*BASE, *mode, *with_manifest(tmp_path)]
    assert run_main(monkeypatch, *argv, *approve_request(tmp_path, monkeypatch, capsys, argv)) == 2
    assert "input seal flickr25k:stage1 refused" in capsys.readouterr().err and boundaries == []


@pytest.mark.parametrize("mode", MODES[1:])
def test_execution_without_the_generation_manifest_refuses(tmp_path, monkeypatch, boundaries, capsys, mode):
    fake_authorities(tmp_path, monkeypatch)
    monkeypatch.setattr(M, "render_trainer_argv", fake_render)
    assert run_main(monkeypatch, *BASE, *mode) == 2
    assert "need --anchor-manifest" in capsys.readouterr().err and boundaries == []


@pytest.mark.parametrize("mode", MODES)
def test_a_manifest_of_another_tree_refuses_in_every_mode(tmp_path, monkeypatch, boundaries, capsys, mode):
    fake_authorities(tmp_path, monkeypatch)
    monkeypatch.setattr(M, "render_trainer_argv", fake_render)
    path, sha = manifest(tmp_path, **{"new_generation::files_sha256::model_siglip2.py": "0" * 64})
    assert run_main(monkeypatch, *BASE, *mode, "--anchor-manifest", path,
                    "--anchor-manifest-sha256", sha) == 2
    assert "not the manifest's generation" in capsys.readouterr().err and boundaries == []


def test_decide_refuses_a_stage_s_from_another_generation(tmp_path, monkeypatch, boundaries, capsys):
    """Every other gate is opened (the approval included), so the one-generation check is the
    only thing between this request and the lease."""
    import scripts.anchor_confirm_decision as reducer
    fake_authorities(tmp_path, monkeypatch)
    monkeypatch.setattr(M, "render_trainer_argv", fake_render)
    monkeypatch.setattr(M, "audit_approval", lambda section, scope, **pins: {"section": section, "scope": scope,
                                                                         "line": "stub"})
    monkeypatch.setattr(reducer, "verify_selection", lambda path, sha: {
        "n_selected": {ds: {"anchors": 9} for ds in M.ANCHOR_DATASETS},
        "record": {"path": path, "sha256": sha}, "anchor_manifest_sha256": "0" * 64})
    assert run_main(monkeypatch, "--anchor-confirm", "decide", "--namespace", "ancT", *RUNARGS,
                    "--anchor-selection", "/frozen.json", "--anchor-selection-sha256", "1" * 64,
                    *with_manifest(tmp_path)) == 2
    assert "stage S ran under another generation manifest" in capsys.readouterr().err
    assert boundaries == []


def test_plan_writes_reserves_launches_and_inspects_nothing(tmp_path, monkeypatch, boundaries, capsys):
    fake_authorities(tmp_path, monkeypatch)
    monkeypatch.setattr(M, "render_trainer_argv", fake_render)
    before = sorted(p.name for p in tmp_path.iterdir())
    assert run_main(monkeypatch, *BASE, "--plan") == 0
    out = capsys.readouterr().out
    assert "NON-EXECUTABLE" in out
    assert "carry the contract's protocol values at all 16 coordinates" in out
    assert "generation manifest: NOT SUPPLIED" in out
    assert boundaries == [] and sorted(p.name for p in tmp_path.iterdir()) == before


@pytest.mark.parametrize("argv", [
    BASE + ["--refit"], BASE + ["--sweep", "lambda"], BASE + ["--at-topp", "0.6,0.95"],
    BASE + ["--recipe", "/x"], BASE + ["--stability-plan", "/x"],
    ["--anchor-confirm", "select", "--namespace", "p3gE", "--plan"],
    BASE + ["--anchor-arms", "anchors,both", "--plan"],
    BASE + ["--anchor-cells", "cifar10:anchors", "--plan"],
    BASE + ["--anchor-selection", "/x", "--plan"],
    ["--anchor-confirm", "decide", "--namespace", "ancT", "--plan"],
])
def test_refused_requests_never_reach_a_boundary(tmp_path, monkeypatch, boundaries, argv):
    fake_authorities(tmp_path, monkeypatch)
    monkeypatch.setattr(M, "render_trainer_argv", fake_render)
    assert run_main(monkeypatch, *argv) == 2
    assert boundaries == []


def test_decide_refuses_a_self_pinned_minimal_selection_record(tmp_path, monkeypatch, boundaries, capsys):
    """Audit 671.2: kind, version and a matching digest are not selection evidence."""
    fake_authorities(tmp_path, monkeypatch)
    monkeypatch.setattr(M, "render_trainer_argv", fake_render)
    fake = tmp_path / "frozen.json"
    sha = write_json(fake, {"artifact_kind": "anchor_confirmation_n_selection",
                            "version": M.ANCHOR_CONFIRM_VERSION,
                            "n_selected": {ds: {"anchors": 9} for ds in M.ANCHOR_DATASETS}})
    assert run_main(monkeypatch, "--anchor-confirm", "decide", "--namespace", "ancT", "--plan",
                    "--anchor-selection", str(fake), "--anchor-selection-sha256", sha) == 2
    assert "does not replay" in capsys.readouterr().err and boundaries == []


def test_run_cell_refuses_an_anchor_cell_without_a_sealed_recipe(monkeypatch, boundaries):
    def launched(*a, **k):
        raise AssertionError("a trainer was launched")
    monkeypatch.setattr(M, "_run_managed_process", launched)
    with pytest.raises(CellRefused, match="sealed scientific recipe"):
        M.run_cell("flickr25k", 4, 0, namespace="ancT", topp=("0.6", "0.95"), joint="0.02",
                   anchor_arm="anchors", campaign_binding=None, scientific_recipe=None)
    assert boundaries == []


# ---- completion --------------------------------------------------------------------------------
@pytest.fixture
def sealed():
    from config import Config
    argv = ["--tag", "t", "--dataset", "Flickr25k", "--setting", "1", "--axis_center", "anchors"]
    return R.build_payload(Config.build_parser(), argv, planned_arm="anchors"), argv


def completed_run(tmp_path, payload, argv, *, evidence_digest=None, mutate=None):
    import torch
    from config import Config
    from dna_utils.run_identity import PHASE3_CAMPAIGN_BINDING_NAME
    values = vars(Config.build_parser().parse_args(argv))
    values["setting"] = "setting" + str(values["setting"])
    values.pop("log_dir", None)
    values["_phase3_campaign_binding"] = {"cell_id": "c"}
    if mutate:
        mutate(values)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    torch.save(values, run_dir / "config.pt")
    (run_dir / PHASE3_CAMPAIGN_BINDING_NAME).write_text(json.dumps({
        "scientific_recipe_schema": payload["schema"],
        "scientific_recipe_sha256": evidence_digest or R.digest(payload)}))
    return run_dir


def test_completion_accepts_the_sealed_recipe(tmp_path, sealed):
    payload, argv = sealed
    got = M.assert_anchor_recipe(completed_run(tmp_path, payload, argv), scientific_recipe=payload,
                                 arm="anchors")
    assert got["scientific_recipe_sha256"] == R.digest(payload) and got["arm"] == "anchors"


@pytest.mark.parametrize("case,reason", [("evidence_digest", "does not name the sealed recipe"),
                                         ("config_value", "differs from the sealed recipe"),
                                         ("config_type", "differs from the sealed recipe"),
                                         ("arm", "!= 'none'"), ("no_config", "missing anchor evidence")])
def test_completion_refuses_what_is_not_the_sealed_recipe(tmp_path, sealed, case, reason):
    payload, argv = sealed
    kw = {"evidence_digest": "f" * 64} if case == "evidence_digest" else {}
    if case == "config_value":
        kw["mutate"] = lambda v: v.update(lambda_bu=0.5)
    if case == "config_type":
        kw["mutate"] = lambda v: v.update(random_seed=float(v["random_seed"]))   # 42 -> 42.0
    run_dir = completed_run(tmp_path, payload, argv, **kw)
    if case == "no_config":
        (run_dir / "config.pt").unlink()
    with pytest.raises(CellRefused, match=reason):
        M.assert_anchor_recipe(run_dir, scientific_recipe=payload,
                               arm="none" if case == "arm" else "anchors")


# ---- approved inputs survive admission (audit 694.2) -------------------------------------------------------
def seal_files(tmp_path):
    """One private seal file per dataset, their CLI declarations and the authorities an admission of them
    returns (the low-level seal verifier itself is a stand-in here)."""
    args, authorities = [], {}
    for ds in M.ANCHOR_DATASETS:
        path = tmp_path / f"{ds}.stage1.input-seal.json"
        path.write_text(json.dumps({"seal": ds}))
        args += ["--input-seal", f"{ds}:stage1={path}"]
        authorities[f"{ds}:stage1"] = {"seal_path": str(path.resolve()),
                                       "seal_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return args, authorities


@pytest.mark.parametrize("change", ["digest", "path", "missing", "extra"])
def test_admitted_seals_other_than_the_approved_request_refuse(change):
    approved = {"flickr25k:stage1": {"path": "/s/f.json", "sha256": "a" * 64}}
    admitted = {"flickr25k:stage1": {"seal_path": "/s/f.json", "seal_file_sha256": "a" * 64}}
    M.assert_request_seals({"input_seals": approved}, admitted, "test")         # the positive control
    if change == "digest":
        admitted["flickr25k:stage1"]["seal_file_sha256"] = "b" * 64
    elif change == "path":
        admitted["flickr25k:stage1"]["seal_path"] = "/s/g.json"
    elif change == "missing":
        admitted = {}
    else:
        admitted["nuswide:stage1"] = {"seal_path": "/s/n.json", "seal_file_sha256": "c" * 64}
    with pytest.raises(CellRefused, match="not the approved request's"):
        M.assert_request_seals({"input_seals": approved}, admitted, "test")


def test_approved_seals_reach_the_lease(tmp_path, monkeypatch, boundaries, capsys):
    fake_authorities(tmp_path, monkeypatch)
    monkeypatch.setattr(M, "render_trainer_argv", fake_render)
    seal_args, admitted = seal_files(tmp_path)
    monkeypatch.setattr(M, "verify_campaign_input_seals",
                        lambda *a, **k: boundaries.append("verify_campaign_input_seals") or dict(admitted))
    argv = [*BASE, *RUNARGS, *seal_args, *with_manifest(tmp_path)]
    approval = approve_request(tmp_path, monkeypatch, capsys, argv)
    assert run_main(monkeypatch, *argv, *approval) == 0
    assert boundaries == ["verify_campaign_input_seals", "_with_campaign_gpu_leases"]


def test_the_anchor_admission_uses_the_historical_verifier_and_records_it(tmp_path, monkeypatch,
                                                                          boundaries, capsys):
    """Audit 715: the full admission asks for the historical verifier, and what it did reaches the
    campaign's authorities (the plan snapshot and receipt)."""
    fake_authorities(tmp_path, monkeypatch)
    monkeypatch.setattr(M, "render_trainer_argv", fake_render)
    seal_args, admitted = seal_files(tmp_path)
    seen, captured = {}, {}

    def admit(*a, **k):
        seen.update(k)
        k["evidence"]["flickr25k:stage1"] = {"report": "verified"}
        return dict(admitted)
    monkeypatch.setattr(M, "verify_campaign_input_seals", admit)
    monkeypatch.setattr(M, "_with_campaign_gpu_leases", lambda args, callback: callback())
    monkeypatch.setattr(M, "_run_sweep", lambda *a, **k: captured.update(k) or 0)
    argv = [*BASE, *RUNARGS, *seal_args, *with_manifest(tmp_path)]
    approval = approve_request(tmp_path, monkeypatch, capsys, argv)
    assert run_main(monkeypatch, *argv, *approval) == 0
    assert seen.get("full") is True and seen.get("historical") is True and seen.get("expected") is None
    assert captured["authorities"]["historical_input_admission"] == {"flickr25k:stage1": {"report": "verified"}}


def test_a_historical_verification_refusal_stops_before_the_lease(tmp_path, monkeypatch, boundaries, capsys):
    from scripts.seal_phase3_inputs import SealError
    fake_authorities(tmp_path, monkeypatch)
    monkeypatch.setattr(M, "render_trainer_argv", fake_render)
    seal_args, _ = seal_files(tmp_path)

    def refuse(path, **k):
        raise SealError("the historical verifier refused (rc 2): drifted")
    monkeypatch.setattr(M, "verify_campaign_input_seals", REAL_VERIFY_CAMPAIGN_INPUT_SEALS)
    monkeypatch.setattr(M, "verify_seal_historically", refuse)
    argv = [*BASE, *RUNARGS, *seal_args, *with_manifest(tmp_path)]
    approval = approve_request(tmp_path, monkeypatch, capsys, argv)
    assert run_main(monkeypatch, *argv, *approval) == 2
    assert "refused: the historical verifier refused (rc 2)" in capsys.readouterr().err
    assert boundaries == []


# ---- carrying a full historical admission forward (audit 723) ----------------------------------------
def sha256_of(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def carried_snapshot(tmp_path, admitted, **edits):
    """A plan snapshot whose seals were admitted by the historical verifier: input_seals plus the
    per-seal bridge evidence the launcher publishes. `edits` set "key:field" values in one row."""
    evidence = {key: {"returncode": 0, "root": str(M.HISTORICAL_INPUT_ROOT),
                      "verifier": {"path": M.HISTORICAL_SEAL_VERIFIER,
                                   "sha256": M.HISTORICAL_SEAL_VERIFIER_SHA256},
                      "seal": {"path": a["seal_path"], "sha256": a["seal_file_sha256"],
                               "aggregate_sha256": a["aggregate_sha256"]},
                      "report": f"verified {a['seal_path']} {a['aggregate_sha256']}"}
                for key, a in admitted.items()}
    for spec, value in edits.items():
        key, field = spec.rsplit(":", 1)
        row = evidence[key]
        if field in ("seal", "aggregate_sha256", "path"):
            row["seal"]["sha256" if field == "seal" else field] = value
        elif field == "verifier":
            row["verifier"]["sha256"] = value
        elif value is DELETE:
            evidence.pop(key)
        else:
            row[field] = value
    path = tmp_path / "carried_snapshot.json"
    path.write_text(json.dumps({"input_seals": admitted,
                                "plan": {"authorities": {"historical_input_admission": evidence}}}))
    return path


def full_authorities(admitted):
    return {k: {**v, "aggregate_sha256": "a" * 64} for k, v in admitted.items()}


def test_a_full_historical_admission_can_be_carried(tmp_path):
    _, admitted = seal_files(tmp_path)
    carried = full_authorities(admitted)
    snapshot = carried_snapshot(tmp_path, carried)
    assert M.anchor_carried_admission_refusal(snapshot, carried, expected_sha256=sha256_of(snapshot)) is None


@pytest.mark.parametrize("edit,reason", [
    ({"flickr25k:stage1:returncode": 2}, "returncode"),
    ({"flickr25k:stage1:root": "/data/yschoi/elsewhere"}, "root"),
    ({"flickr25k:stage1:verifier": "0" * 64}, "verifier"),
    ({"flickr25k:stage1:seal": "0" * 64}, "seal"),
    ({"flickr25k:stage1:aggregate_sha256": "0" * 64}, "aggregate"),
    ({"flickr25k:stage1:path": "/other/seal.json"}, "path"),
    ({"flickr25k:stage1:report": "verified elsewhere"}, "report"),
    ({"flickr25k:stage1:row": DELETE}, "are not exactly those"),
])
def test_a_carried_authority_that_is_not_a_full_historical_admission_refuses(tmp_path, edit, reason):
    _, admitted = seal_files(tmp_path)
    carried = full_authorities(admitted)
    (spec, value), = edit.items()
    key, field = spec.rsplit(":", 1)
    path = carried_snapshot(tmp_path, carried, **{f"{key}:{field}": value})
    refusal = M.anchor_carried_admission_refusal(path, carried, expected_sha256=sha256_of(path))
    assert refusal is not None and reason in refusal


def test_a_snapshot_without_historical_evidence_cannot_be_carried(tmp_path):
    _, admitted = seal_files(tmp_path)
    carried = full_authorities(admitted)
    path = tmp_path / "stats_only_snapshot.json"
    path.write_text(json.dumps({"input_seals": carried, "plan": {"authorities": {}}}))
    refusal = M.anchor_carried_admission_refusal(path, carried, expected_sha256=sha256_of(path))
    assert refusal is not None and "are not exactly those" in refusal


def test_the_carried_snapshot_is_read_at_its_approved_bytes(tmp_path):
    _, admitted = seal_files(tmp_path)
    carried = full_authorities(admitted)
    path = carried_snapshot(tmp_path, carried)
    refusal = M.anchor_carried_admission_refusal(path, carried, expected_sha256="0" * 64)
    assert refusal is not None and "changed after its approval pin" in refusal


@pytest.mark.parametrize("with_evidence", [True, False])
def test_the_anchor_admission_carries_only_a_full_historical_admission(tmp_path, monkeypatch, boundaries,
                                                                       capsys, with_evidence):
    """End to end in the launcher: an approved request naming the carried snapshot's bytes takes the
    stats-only check against it -- or refuses before the lease when the snapshot has no historical
    admission evidence."""
    fake_authorities(tmp_path, monkeypatch)
    monkeypatch.setattr(M, "render_trainer_argv", fake_render)
    seal_args, admitted = seal_files(tmp_path)
    carried = full_authorities(admitted)
    if with_evidence:
        snapshot = carried_snapshot(tmp_path, carried)
    else:
        snapshot = tmp_path / "stats_only_snapshot.json"
        snapshot.write_text(json.dumps({"input_seals": carried, "plan": {"authorities": {}}}))
    monkeypatch.setattr(M, "load_admission_authority", lambda path: dict(carried))
    seen = {}

    def admit(*a, **k):
        seen.update(k)
        boundaries.append("verify_campaign_input_seals")
        return dict(admitted)
    monkeypatch.setattr(M, "verify_campaign_input_seals", admit)
    argv = [*BASE, *RUNARGS, *seal_args, *with_manifest(tmp_path), "--admission-authority", str(snapshot)]
    approval = approve_request(tmp_path, monkeypatch, capsys, argv)
    rc = run_main(monkeypatch, *argv, *approval)
    if with_evidence:
        assert rc == 0 and seen.get("full") is False and seen.get("expected") == carried
        assert boundaries == ["verify_campaign_input_seals", "_with_campaign_gpu_leases"]
    else:
        assert rc == 2 and "not exactly those its snapshot admitted" in capsys.readouterr().err
        assert boundaries == []


def test_a_seal_changed_after_its_approval_refuses_before_the_lease(tmp_path, monkeypatch, boundaries, capsys):
    """The seal file changes after the approval check and before admission (audit 694.2's case)."""
    fake_authorities(tmp_path, monkeypatch)
    monkeypatch.setattr(M, "render_trainer_argv", fake_render)
    seal_args, admitted = seal_files(tmp_path)
    def admit_changed(*a, **k):
        boundaries.append("verify_campaign_input_seals")
        path = Path(admitted["flickr25k:stage1"]["seal_path"])
        path.write_text(path.read_text() + "\n")
        return dict(admitted, **{"flickr25k:stage1": {"seal_path": str(path),
                                 "seal_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}})
    monkeypatch.setattr(M, "verify_campaign_input_seals", admit_changed)
    argv = [*BASE, *RUNARGS, *seal_args, *with_manifest(tmp_path)]
    approval = approve_request(tmp_path, monkeypatch, capsys, argv)
    assert run_main(monkeypatch, *argv, *approval) == 2
    assert "not the approved request's" in capsys.readouterr().err
    assert boundaries == ["verify_campaign_input_seals"]                    # no lease


@pytest.mark.parametrize("changed", [False, True])
def test_the_carried_authority_is_the_approved_bytes_while_it_is_parsed(tmp_path, monkeypatch, boundaries,
                                                                        capsys, changed):
    fake_authorities(tmp_path, monkeypatch)
    monkeypatch.setattr(M, "render_trainer_argv", fake_render)
    authority = tmp_path / "authority.json"
    # a (vacuous) full historical admission, so the carried-provenance guard (audit 723, tested on
    # its own) passes and this test isolates the byte binding
    authority.write_text(json.dumps({"plan": {"authorities": {"historical_input_admission": {}}}}))
    def parse(path):
        if changed:
            Path(path).write_text('{"late": true}')
        return {}
    monkeypatch.setattr(M, "load_admission_authority", parse)
    argv = [*BASE, *RUNARGS, "--admission-authority", str(authority), *with_manifest(tmp_path)]
    approval = approve_request(tmp_path, monkeypatch, capsys, argv)
    rc = run_main(monkeypatch, *argv, *approval)
    if changed:
        assert rc == 2 and "carried admission authority is not the approved" in capsys.readouterr().err
        assert boundaries == []
    else:
        assert rc == 0 and boundaries == ["verify_campaign_input_seals", "_with_campaign_gpu_leases"]


# ---- the admitted generation, re-verified at every boundary (audit 697) -------------------------------------
def copy_of_the_tree(tmp_path, monkeypatch):
    """A private copy of the generation's files, with REPO pointed at it and its manifest."""
    root = tmp_path / "tree"
    for rel in M.anchor_generation_closure():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / rel, root / rel)
    monkeypatch.setattr(M, "REPO", root)
    from importlib import metadata
    files = {rel: hashlib.sha256((root / rel).read_bytes()).hexdigest() for rel in M.anchor_generation_closure()}
    body = {"artifact_kind": M.ANCHOR_MANIFEST_KIND, "version": M.ANCHOR_CONFIRM_VERSION,
            "historical": {"approved_p3_refit_aggregate": {"sha256": M.APPROVED_P3_REFIT_AGGREGATE_SHA256},
                           "approved_selected_n": {"sha256": M.APPROVED_SELECTED_N_SHA256},
                           "historical_input_verifier": M.historical_input_verifier_pins(),
                           "anchor_v7": M.anchor_v7_pins()},
            "new_generation": {"commit": "c" * 40, "branch": "b", "clean": True, "files_sha256": files,
                               "dataset_scripts_sha256": dict(M.DATASET_SCRIPT_SHA256),
                               "environment": {"python": ".".join(map(str, sys.version_info[:3])),
                                               "torch": metadata.version("torch"), "interpreter": sys.executable}},
            "contract": {"path": M.ANCHOR_CONTRACT_PATH, "sha256": files[M.ANCHOR_CONTRACT_PATH]},
            "lambda_contract": {"path": M.ANCHOR_LAMBDA_CONTRACT_PATH,
                                "sha256": files[M.ANCHOR_LAMBDA_CONTRACT_PATH]},
            "refit_contract": {"path": M.ANCHOR_REFIT_CONTRACT_PATH,
                               "sha256": files[M.ANCHOR_REFIT_CONTRACT_PATH]},
            "recovery_contract": {"path": M.ANCHOR_RECOVERY_CONTRACT_PATH,
                                  "sha256": files[M.ANCHOR_RECOVERY_CONTRACT_PATH]}}
    path = tmp_path / "manifest.json"
    return root, {"path": str(path), "sha256": write_json(path, body)}


def test_an_unchanged_generation_rechecks(tmp_path, monkeypatch):
    root, ref = copy_of_the_tree(tmp_path, monkeypatch)
    assert M.recheck_generation(ref, "test")["sha256"] == ref["sha256"]


@pytest.mark.parametrize("member", ["scripts/anchor_confirm_decision.py", "scripts/anchor_confirm_code_axis.py",
                                    "scripts/anchor_confirm_manifest.py", M.ANCHOR_CONTRACT_PATH,
                                    "tests/test_anchor_confirm_reducer.py", "config.py"])
def test_a_member_changed_after_admission_refuses(tmp_path, monkeypatch, member):
    """Audit 697's anchor-only members (and a legacy one) drift after the first admission."""
    root, ref = copy_of_the_tree(tmp_path, monkeypatch)
    M.recheck_generation(ref, "admission")
    (root / member).write_bytes((root / member).read_bytes() + b"\n")
    with pytest.raises(CellRefused, match="not the manifest's generation"):
        M.recheck_generation(ref, "later")


def test_a_module_imported_from_other_bytes_refuses(tmp_path, monkeypatch):
    import scripts.anchor_confirm_decision as reducer
    path, sha = manifest(tmp_path)
    M.recheck_generation({"path": path, "sha256": sha}, "positive")
    monkeypatch.setattr(reducer, "_IMPORTED_SOURCE_SHA256", "0" * 64)
    with pytest.raises(CellRefused, match="imported from other bytes"):
        M.recheck_generation({"path": path, "sha256": sha}, "test")


def test_a_generation_drifting_during_input_admission_refuses_before_the_lease(tmp_path, monkeypatch,
                                                                              boundaries, capsys):
    fake_authorities(tmp_path, monkeypatch)
    monkeypatch.setattr(M, "render_trainer_argv", fake_render)
    argv = [*BASE, *RUNARGS, *with_manifest(tmp_path)]
    approval = approve_request(tmp_path, monkeypatch, capsys, argv)
    def recheck(manifest, where):
        raise CellRefused(f"{where}: scripts/anchor_confirm_decision.py drifted")
    monkeypatch.setattr(M, "recheck_generation", recheck)
    assert run_main(monkeypatch, *argv, *approval) == 2
    assert "after input admission" in capsys.readouterr().err
    assert boundaries == ["verify_campaign_input_seals"]                    # no lease


@pytest.fixture
def sweep(tmp_path, monkeypatch):
    """The anchor-mode sweep after its lease, every side effect a recorder: seals, snapshot,
    reservation, cells and publications."""
    calls = []
    records = tmp_path / "records"
    records.mkdir()
    monkeypatch.setattr(M, "RECORD_DIR", records)
    cells = M.anchor_confirmation_cells("select", incumbent=INCUMBENT, arm_plan=ALL_ANCHORS)
    keys = [M.campaign_cell_id(c[0], c[1], topp=c[2], joint=c[3], stage=c[4], seed=c[5], anchor_arm=c[7])
            for c in cells]
    seals = {f"{ds}:stage1": {"seal_path": f"/seals/{ds}.json", "seal_file_sha256": "5" * 64}
             for ds in M.ANCHOR_DATASETS}
    path, sha = manifest(tmp_path)
    authorities = {"anchor_request": {"input_seals": {k: {"path": v["seal_path"], "sha256": v["seal_file_sha256"]}
                                                      for k, v in seals.items()}},
                   "anchor_manifest": {"path": path, "sha256": sha}}
    admitted = {"seals": dict(seals)}
    monkeypatch.setattr(M, "verify_campaign_input_seals",
                        lambda *a, **k: calls.append("seals") or dict(admitted["seals"]))
    def snapshot(*a, **k):
        calls.append("plan_snapshot")
        return {"sources": {}, "inputs": {}, "input_seals": k["input_seals"], "qwen_root": "q",
                "source_authority_sha256": "s", "environment_sha256": "e",
                "plan": {"result_root": "/r", "cell_bindings": {key: {} for key in keys}}}
    monkeypatch.setattr(M, "plan_snapshot", snapshot)
    for name in ("_assert_snapshot_gpu_leases", "assert_production_source_authority",
                 "assert_reservation_owner", "verify_snapshot_input_seals"):
        monkeypatch.setattr(M, name, lambda *a, **k: None)
    monkeypatch.setattr(M, "reserve_sweep_namespace", lambda *a, **k: calls.append("reserve") or "reservation")
    monkeypatch.setattr(M, "_publish_json_exclusive", lambda path, payload: calls.append(("publish", Path(path).name)))
    monkeypatch.setattr(M, "launch_binding_from_expected", lambda planned, digest: {})
    def run_cell(ds, n, gpu, **k):
        calls.append(("run_cell", ds, n))
        (records / f"t_{ds}_{n}.json").write_text("{}")
        return {"tag": f"t_{ds}_{n}", "run_dir": "/r/x", "geometry": {"identity_digest": "i"}, "recipe": {},
                "seed": k["seed"], "selection": {"selection_value": 0.5, "selection_epoch_zero_based": n},
                "campaign": {}, "completion": {}}
    monkeypatch.setattr(M, "run_cell", run_cell)
    reservation = tmp_path / "reservation.json"
    reservation.write_text("{}")
    monkeypatch.setattr(M, "campaign_reservation_path", lambda namespace: reservation)
    args = SimpleNamespace(only=None, sweep=None, anchor_confirm="select", plan=False, run=True, smoke=False,
                           epochs=1, gpus="0,1,2,3", gpu=0, namespace="ancT", admission_authority=None,
                           input_seal_specs={}, result_root="/r")
    def go():
        return M._run_sweep(args, None, full_plan=cells, authorities=authorities,
                            preverified_input_seals=dict(seals))
    return go, calls, admitted


def test_an_unchanged_generation_runs_every_cell_and_publishes_its_receipt(sweep):
    go, calls, _ = sweep
    assert go() == 0
    assert sum(1 for c in calls if c[0] == "run_cell") == 16
    assert ("publish", "ancT_sweep_complete.json") in calls


@pytest.mark.parametrize("where,rc", [("after the lease", 2), ("before ", 1), ("before the receipt", 1)])
def test_a_generation_drifting_inside_the_sweep_stops_before_its_next_boundary(sweep, monkeypatch, where, rc):
    go, calls, _ = sweep
    real = M.recheck_generation
    def recheck(manifest, at):
        if at.startswith(where) and not (where == "before " and at == "before the receipt"):
            raise CellRefused(f"{at}: scripts/anchor_confirm_code_axis.py drifted")
        return real(manifest, at)
    monkeypatch.setattr(M, "recheck_generation", recheck)
    assert go() == rc
    assert ("publish", "ancT_sweep_complete.json") not in calls
    if where == "after the lease":
        assert "plan_snapshot" not in calls and "reserve" not in calls
    elif where == "before ":
        assert not any(c[0] == "run_cell" for c in calls)                   # no cell dispatched
    else:
        assert sum(1 for c in calls if c[0] == "run_cell") == 16


def test_seals_readmitted_after_the_lease_must_still_be_the_approved_ones(sweep):
    go, calls, admitted = sweep
    admitted["seals"] = dict(admitted["seals"], **{"flickr25k:stage1": {"seal_path": "/seals/flickr25k.json",
                                                                         "seal_file_sha256": "e" * 64}})
    assert go() == 2
    assert "plan_snapshot" not in calls and "reserve" not in calls


# ---- the storage rule before every dispatch (audit 703.3) ---------------------------------------------

def test_no_cell_is_dispatched_when_the_result_filesystem_lacks_the_floor(sweep, monkeypatch):
    go, calls, _ = sweep
    monkeypatch.setattr(M, "ANCHOR_FREE_FLOOR_BYTES", 1 << 62)
    assert go() == 1
    assert not any(c[0] == "run_cell" for c in calls)
    assert ("publish", "ancT_sweep_complete.json") not in calls


def test_a_space_breach_mid_sweep_stops_every_later_dispatch(sweep, monkeypatch):
    go, calls, _ = sweep
    real_run_cell = M.run_cell

    def run_cell(*a, **k):
        record = real_run_cell(*a, **k)
        # ">=": four streams may add calls concurrently, so the count can pass 3 between two reads
        if sum(1 for c in calls if c[0] == "run_cell") >= 3:
            monkeypatch.setattr(M, "ANCHOR_FREE_FLOOR_BYTES", 1 << 62)
        return record

    monkeypatch.setattr(M, "run_cell", run_cell)
    assert go() == 1
    dispatched = sum(1 for c in calls if c[0] == "run_cell")
    # the first three, at most one racing call and one already past its check per other stream
    assert 3 <= dispatched <= 9 < len(M.ANCHOR_DATASETS) * len(M.CANDIDATE_N)
    assert ("publish", "ancT_sweep_complete.json") not in calls


def test_the_first_dispatch_reserves_space_for_every_unfinished_cell(sweep, monkeypatch):
    """Room for 15.5 cells of one TiB each: enough for any one stream's four, not for all sixteen."""
    go, calls, _ = sweep
    probe = Path("/r")
    while not probe.exists():
        probe = probe.parent
    usage = os.statvfs(probe)
    monkeypatch.setattr(M, "ANCHOR_CELL_OUTPUT_BYTES", 1 << 40)
    monkeypatch.setattr(M, "ANCHOR_FREE_FLOOR_BYTES", usage.f_bavail * usage.f_frsize - 31 * (1 << 39))
    assert go() == 1
    assert not any(c[0] == "run_cell" for c in calls)


def test_the_storage_rule_counts_every_unfinished_cell(tmp_path, monkeypatch):
    """One TiB per cell dwarfs any change of the real free space during the test."""
    free = os.statvfs(tmp_path).f_bavail * os.statvfs(tmp_path).f_frsize
    monkeypatch.setattr(M, "ANCHOR_CELL_OUTPUT_BYTES", 1 << 40)
    monkeypatch.setattr(M, "ANCHOR_FREE_FLOOR_BYTES", free - 5 * (1 << 39))
    assert M.anchor_dispatch_space_refusal(tmp_path / "not" / "yet", 2) is None
    refusal = M.anchor_dispatch_space_refusal(tmp_path / "not" / "yet", 3)
    assert refusal and "3 unfinished cells" in refusal and str(tmp_path) in refusal


# ---- the protocol values (contract section 6) -----------------------------------------------------------
def test_protocol_fields_fix_the_stage_1_horizons_and_the_smoke_shortens_them_alone():
    kw = dict(seed=43, arm="none", incumbent=INCUMBENT)
    full = M.anchor_protocol_fields("mscoco", 39, **kw)
    assert (full["epoch"], full["lr_schedule_horizon"], full["sinkhorn_schedule_horizon"],
            full["stop_after_epoch"]) == (60, 60, 40, 39)
    assert (full["routing_adaptive_topp_min"], full["routing_adaptive_topp_max"],
            full["lambda_codon_joint"]) == (0.6, 0.95, 0.03)
    assert full["hash_target_mode"] == "siglip_cos" and full["disable_text_supervision"] is False
    smoke = M.anchor_protocol_fields("mscoco", 39, epochs=2, **kw)
    assert (smoke["epoch"], smoke["lr_schedule_horizon"], smoke["sinkhorn_schedule_horizon"],
            smoke["stop_after_epoch"]) == (2, 2, 2, 1)
    assert {k: v for k, v in smoke.items() if k not in ("epoch", "lr_schedule_horizon",
            "sinkhorn_schedule_horizon", "stop_after_epoch")} == \
        {k: v for k, v in full.items() if k not in ("epoch", "lr_schedule_horizon",
            "sinkhorn_schedule_horizon", "stop_after_epoch")}


def test_a_cell_off_the_approved_recipe_refuses(monkeypatch):
    monkeypatch.setattr(M, "render_trainer_argv", fake_render)
    cell = ("flickr25k", 4, ("0.5", "0.9"), "0.02", "select", 42, (), "anchors")
    with pytest.raises(CellRefused, match="not the approved incumbent's"):
        M.anchor_admission([cell], namespace="ancT", incumbent=INCUMBENT)


@pytest.mark.parametrize("appended,field", [(["--disable_text_supervision"], "disable_text_supervision"),
                                            (["--no_routing_adaptive_topp"], "no_routing_adaptive_topp"),
                                            (["--final_epoch_eval"], "final_epoch_eval")])
def test_both_arms_off_the_protocol_refuse_although_they_differ_in_the_axis_alone(monkeypatch, appended, field):
    monkeypatch.setattr(M, "render_trainer_argv", lambda cmd, env: fake_render(cmd, env) + appended)
    cells = M.anchor_confirmation_cells("select", incumbent=INCUMBENT, arm_plan=ALL_ANCHORS)
    with pytest.raises(CellRefused, match=f"breaks the protocol in {field}"):
        M.anchor_admission(cells[:1], namespace="ancT", incumbent=INCUMBENT)


# ---- the actual command composition: the pinned wrappers, rendered (audit 677.2) ----------------------
@pytest.fixture
def synthetic_paths(tmp_path, monkeypatch):
    """Every input path the wrappers receive points into tmp_path; the whitening file exists (empty),
    so no wrapper reaches its builder, and so do the two token files the CIFAR-10 wrapper checks
    with `[ -f ]` (empty). Nothing under these paths is read."""
    whiten = tmp_path / "text_whiten_optTrain_localOnly.npz"
    whiten.write_bytes(b"")
    monkeypatch.setattr(M, "_whitening", lambda spec, stage="select": str(whiten))
    for ds in M.ANCHOR_DATASETS:
        cache = tmp_path / f"{ds}_cache"
        cache.mkdir()
        for name in ("text_tokens.f16.npy", "text_token_mask.bool.npy"):
            (cache / name).write_bytes(b"")
        monkeypatch.setitem(M.DATASETS[ds], "cache", str(cache))
        monkeypatch.setitem(M.DATASETS[ds], "qwen", str(tmp_path / f"{ds}_qwen.jsonl"))
    return tmp_path


@pytest.mark.parametrize("dataset", M.ANCHOR_DATASETS)
@pytest.mark.parametrize("arm", M.ANCHOR_ARMS)
def test_the_actual_wrapper_composition_repeats_exactly_the_reviewed_overrides(synthetic_paths, dataset, arm):
    from config import Config
    parser = Config.build_parser()
    inc = INCUMBENT[dataset]
    cmd, env, _ = M.build_command(dataset, inc["N"], 0, topp=inc["topp"], joint=inc["joint"],
                                  seed=43, anchor_arm=arm)
    argv = M.render_trainer_argv(cmd, env)
    repeated = {d for d, spans in R.option_occurrences(parser, argv).items() if len(spans) > 1}
    assert repeated == set(R.REVIEWED_OVERRIDES)          # all seven, the three routing ones included
    admitted = R.admitted_overrides(parser, argv)
    literal = {d: tuple(spans[0]) for d, spans in admitted.items()}
    # the wrapper's own first occurrence: the reviewed literal, except CIFAR-10's codon-Sinkhorn
    # literal, which its body takes from ${CCS:-0.1} (contract v3 section 6)
    want_literal = dict(R.REVIEWED_OVERRIDES)
    if dataset == "cifar10":
        want_literal["lambda_codeword_codon_sinkhorn"] = ("--lambda_codeword_codon_sinkhorn", "0.1")
    assert literal == want_literal
    assert admitted["lambda_codeword_codon_sinkhorn"][1] == ["--lambda_codeword_codon_sinkhorn", "0.0"]
    fields = R.build_payload(parser, argv, planned_arm=arm)["fields"]
    want = M.anchor_protocol_fields(dataset, inc["N"], seed=43, arm=arm, incumbent=INCUMBENT)
    assert {k: fields[k] for k in want} == want


def test_the_actual_composition_admits_every_stage_s_coordinate(synthetic_paths):
    cells = M.anchor_confirmation_cells("select", incumbent=INCUMBENT, arm_plan=ALL_ANCHORS)
    report = M.anchor_admission(cells, namespace="ancT", incumbent=INCUMBENT)
    assert len(report) == 16
    assert all(sorted(e["overrides"]) == sorted(R.REVIEWED_OVERRIDES) for e in report.values())


def test_the_real_plan_path_over_the_actual_composition(tmp_path, synthetic_paths, monkeypatch,
                                                        boundaries, capsys):
    fake_authorities(tmp_path, monkeypatch)
    assert run_main(monkeypatch, *BASE, "--plan") == 0
    assert "at all 16 coordinates" in capsys.readouterr().out and boundaries == []


def test_a_third_occurrence_injected_into_the_actual_composition_refuses(synthetic_paths):
    cmd, env, _ = M.build_command("nuswide", 4, 0, topp=("0.4", "0.8"), joint="0.05", anchor_arm="anchors")
    env["EXTRA_ARGS"] += " --routing_adaptive_topp_min 0.5"
    with pytest.raises(R.RecipeMismatch, match="given 3 times"):
        R.build_payload(__import__("config").Config.build_parser(), M.render_trainer_argv(cmd, env),
                        planned_arm="anchors")


# ---- opt-in, against the real approved artifacts ---------------------------------------------------
@pytest.mark.skipif(os.environ.get("GDNA_ALLOW_REAL_ARTIFACT_TESTS") != "1",
                    reason="real-artifact test; opt in with GDNA_ALLOW_REAL_ARTIFACT_TESTS=1")
def test_real_plan_renders_every_coordinate(monkeypatch, boundaries, capsys):
    assert run_main(monkeypatch, *BASE, "--plan") == 0
    assert "at all 16 coordinates" in capsys.readouterr().out
    assert boundaries == []


# ---- dataset-specific values (audit 709.1 item 4) ---------------------------------------------------
def test_the_dataset_specific_values_agree_across_their_sources():
    """K, the mAP@R cutoff and the train/validation populations, per dataset, from every source the
    campaign reads them from; CIFAR-10 is K=64 and mAP@1000, not the multi-label values."""
    import evaluation_siglip2 as E
    want = {"cifar10": (64, 1000, 5000), "flickr25k": (128, 5000, 5000),
            "nuswide": (128, 5000, 10500), "mscoco": (128, 5000, 10000)}
    for ds, (k, r, train) in want.items():
        assert M.DATASETS[ds]["K"] == k and M.MAP_R_CUTOFF[ds] == r
        assert E.MAP_AT_R_BY_DATASET[M.DATASETS[ds]["canon"]] == r
        assert M.PAPER_SPLIT_ROWS[ds]["train"] == train
        assert M.anchor_protocol_fields(ds, 4, seed=42, arm="anchors", incumbent=INCUMBENT)["codebook_size"] == k


def test_a_one_cell_smoke_needs_one_gpu_and_a_run_four(tmp_path, monkeypatch, boundaries, capsys):
    fake_authorities(tmp_path, monkeypatch)
    monkeypatch.setattr(M, "render_trainer_argv", fake_render)
    manifest_args = with_manifest(tmp_path)
    assert run_main(monkeypatch, *BASE, *MODES[1], "--gpus", "0,1", *manifest_args, "--plan") == 2
    assert "needs exactly that many distinct GPUs" in capsys.readouterr().err
    assert run_main(monkeypatch, *BASE, "--run", *manifest_args, "--plan") == 2     # one default GPU
    assert "needs exactly that many distinct GPUs" in capsys.readouterr().err and boundaries == []


def test_an_effective_cifar_codon_sinkhorn_other_than_zero_refuses(monkeypatch):
    """The alternate CIFAR-10 literal admits the ORDER (0.1, then the override); the protocol binds
    the effective value: an override left at 0.1 refuses at the admission."""
    def render(cmd, env):
        argv = fake_render(cmd, env)
        last = len(argv) - 1 - argv[::-1].index("--lambda_codeword_codon_sinkhorn")
        return argv[:last + 1] + ["0.1"] + argv[last + 2:]
    monkeypatch.setattr(M, "render_trainer_argv", render)
    cell = ("cifar10", 4, ("0.3", "0.7"), "0.02", "select", 42, (), "anchors")
    with pytest.raises(CellRefused, match="breaks the protocol in lambda_codeword_codon_sinkhorn"):
        M.anchor_admission([cell], namespace="ancT", incumbent=INCUMBENT)
