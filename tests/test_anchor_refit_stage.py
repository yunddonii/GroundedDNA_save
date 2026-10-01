"""Stages R and T of the fixed anchor model (generation v9; docs/ANCHOR_REFIT_CONTRACT_v1.md; audits
742-744).

Synthetic only: no real input, config, checkpoint, model, dataset, ledger or GPU is touched. Each test
says what kind of evidence it is:
  predicate  -- a pure function, checked on its inputs alone;
  structural -- the source wiring (AST or text), which a predicate test cannot see;
  composed   -- through a real entry point (the trainer's start-up and terminal decision, the
                launcher main() and run_cell, the stage-T campaign cell and entry main(), the real
                raw evaluator and BIO stage, the supervisor's ledger choice), only the external
                boundaries (processes, leases, model encoding) replaced by recorders.
The real-wrapper render is opt-in (GDNA_ALLOW_REAL_ARTIFACT_TESTS=1), like the other anchor suites.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

import scripts.phase3_selection_matrix as M                 # noqa: E402
from scripts.phase3_selection_matrix import CellRefused     # noqa: E402
import scripts.anchor_refit_stage as RT                     # noqa: E402
import scripts.anchor_terminal_test as TE                   # noqa: E402
import scripts.anchor_confirm_supervisor as S               # noqa: E402
import p0_protocol as P0                                    # noqa: E402
import dna_utils.scientific_recipe as R                     # noqa: E402
import test_anchor_confirm_launcher as LT                   # noqa: E402
import test_anchor_confirm_reducer as RW                    # noqa: E402
import test_anchor_confirm_recipe as RC                     # noqa: E402

REAL = os.environ.get("GDNA_ALLOW_REAL_ARTIFACT_TESTS") == "1"
INC = LT.INCUMBENT
FROZEN_N = {"cifar10": 4, "flickr25k": 4, "nuswide": 4, "mscoco": 39}
V8_COMMIT = "3dd1c02"            # the generation-v8 tip this branch starts from
#: the members generation v9 changes relative to v8 r2, and the members it adds
CHANGED_IN_V9 = {"scripts/phase3_selection_matrix.py", "scripts/anchor_confirm_supervisor.py",
                 "scripts/anchor_confirm_manifest.py", "train_siglip2.py", "p0_protocol.py",
                 "tests/test_anchor_confirm_launcher.py", "tests/test_anchor_lambda_stage.py"}
ADDED_IN_V9 = {"terminal_official_test.py", "scripts/anchor_refit_stage.py", "scripts/anchor_terminal_test.py",
               "docs/ANCHOR_REFIT_CONTRACT_v1.md", "tests/test_anchor_refit_stage.py"}
#: generation v8 r2 closure pins, generated from authority_manifest_v8r2.json (58e69ae1...) on disk
V8_FILES_SHA256 = {
    "config.py":
        "13079550490016a9600e6cfc6777cf01eb67335fa7f7e01de5f0018751aa1a8b",
    "dataloaders.py":
        "fa7545908664502a5a4f20fe8d3441b42d5f441653d1310dc4fbf08c1bb14e27",
    "dna_utils/__init__.py":
        "b20f32c8266c4bbdd4efa88fc1ce23bb17cacd7c76fa7ddcdd04437275c082e3",
    "dna_utils/bio_constraints.py":
        "f7863bd363ebe42989f232f550a0af837dde088abf01bada8a0c281f6f164030",
    "dna_utils/cache_provenance.py":
        "144f6c0e51517010898ab4d43cea1d0a06d6ca375d55e5e721550dd06453c73b",
    "dna_utils/csv_logger.py":
        "68450e316bf775989f547b2593131e767e24903b3bef0893bda7a27055cb8d6f",
    "dna_utils/dna_code_utils.py":
        "fa634b9a8cf05db7160f936c24ae72b51c39a828caed64343f64ea12c9251888",
    "dna_utils/extraction_validation.py":
        "bd126a04aac5d1b51eca0379180924d0ad9e80b8d6bae6c2784c86a12c88c8ca",
    "dna_utils/gc_policy.py":
        "3262c0b68eee02357259068efaaa05eb4c6ce0a357e901dd764b27eb96e3b22c",
    "dna_utils/gpu_lease.py":
        "01f1564258fc7ebe6d65226c961b579e7a88f063ce4814d89e75b490875ee976",
    "dna_utils/run_identity.py":
        "a6cfe0e902e889c4ccc35ada829fbd80063811457cf04a4c06c58939ec184c9f",
    "dna_utils/runtime_environment.py":
        "a78d51dbb134bcc400443a42b4c9fa46063b7bef356ab5b37b571a44f489a597",
    "dna_utils/runtime_state.py":
        "cdab32494012f3f7199c7f05af81af547f5ddcbeb2131e166787dab94d26e333",
    "dna_utils/scientific_recipe.py":
        "95ae775192b4806f604052f488a9feee089cfc9e9a144e2b0d76d7426d273cd4",
    "dna_utils/text_description_processor.py":
        "e2028a9d007de711eebcb1b00e7a5fcb37e593d06c28231f3fa9bbf0a0541221",
    "dna_utils/training_utils.py":
        "f1a6ef9a0383312e4d52e0d693656c778ea620e53e69dc61f75d2510946cddb8",
    "dna_utils/visualization.py":
        "f71796319f19915371ae1ba12e25f6fa37a30429860c6d84a15aa222607c25bd",
    "dna_utils/vlm_qwen25_descriptions.py":
        "3c5a865ed264612dd0ff4b9663cd6e7e27c0b4d7c1b02f5045d22c633719f39c",
    "docs/ANCHOR_CONFIRMATION_CONTRACT_v3.md":
        "e6c4978e6d30559e01c2ee34ae2b4d39c01717557ee16c9e96f85980aa8d3efc",
    "docs/ANCHOR_LAMBDA_CONTRACT_v1.md":
        "34a62bfa05600b76f316b3ad62f0da42b8aae8ae2be4fdad490647d834907dfe",
    "evaluation_siglip2.py":
        "7cca26a04b1edc156955fdb721237c6cc69b3b60265b70606c241fb33619911d",
    "extraction_siglip2.py":
        "1d10b3b0e065179f72e477edb98a37be77890040410c6b8a6f1fafc82ed9aab6",
    "loss_siglip2.py":
        "8a0275dd8504519888e68622c88a624685607ca68f05d0ae421dde7cc0a24d4b",
    "model_siglip2.py":
        "9c1cc71b7b40a366439495b3bfa510020a26ef6efd6c6a464b625a9982f3f11d",
    "models/__init__.py":
        "18106eb4dc331c9fbad2734a6e7bf5bc4d1870ba6cde514e9361baee10978444",
    "models/adapters.py":
        "0aa62a08472ef29662480023747caf661104d70b006f14311be7ffc165c68c4a",
    "models/cluster_attention_router.py":
        "2c69889d8d84b70fdf3fb2c571217da1ca06f76f0b49be8dd0d0485ffdc7c13e",
    "models/pretrained_backbone.py":
        "e86953f0760c8f46649b2152b0d2e812053f549da75b9c5a97de05fbd5801b3f",
    "models/pretrained_backbone_clip.py":
        "c0237e43c6fcbd3fb9221bd147648b2e42fa59df280e5919d1699f4c33d4580a",
    "models/semantic_router.py":
        "200a5a0d0f93b5dd05c9a3dcd412588f51a745fafb7d37b179bcc3f1a510460b",
    "models/text_cross_attention_router.py":
        "559677f2659ab1a543e1fed145891a67ffc5a7eb38ecb4f197b0cfa76e3302ec",
    "models/text_encoder.py":
        "0ed8d4f6db5ce87f0f5c18ccc804a631fe0c2502ff218630c91b7217938c0a7b",
    "models/visual_encoder.py":
        "9d38599dcd07377c3e4b8f89a1da4f5965515529023761c26dae587592cf8565",
    "p0_protocol.py":
        "da9aee20f221e896ca2cec3371bb1568568ec831b2c7eaf4f2b0f5f68fade399",
    "scripts/__init__.py":
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "scripts/anchor_confirm_code_axis.py":
        "31537dde1cc8dd338221bb7fd49241cc0c618c0de444a2adf3500e86f708b19e",
    "scripts/anchor_confirm_decision.py":
        "ae247d4a0fe7398198e0975cc81ddc0457b104b14e46faadd20d7d0f70bc7dd9",
    "scripts/anchor_confirm_manifest.py":
        "617eda4ab5ce5feb32e065c2c840862a27e1fae3f532ef3a2a0d8910570fa5de",
    "scripts/anchor_confirm_supervisor.py":
        "12119ff8fe4c822bd704c642acbc4f7dab2ac5a5fa3b0f5387989fbdcbe60364",
    "scripts/build_text_whiten_matrix.py":
        "76e9332662154b1cc3579aeea343c8ec4d917d6fea2007188909cafd456338a0",
    "scripts/eval_cell_bioproj.py":
        "d03ce36be19a0031b568ba1af6a583193308bfe434ae806ceffca426b74da122",
    "scripts/extract_train_split.py":
        "fcb8ea1991f2f3179c26e5fb878479538289c408070efb7b484e25a17c230370",
    "scripts/pairwise_nmi.py":
        "d9016784a76a59c28f7f58185df31363540e7002a1bda3561d720616f3ed9bb2",
    "scripts/phase3_launch_matrix.sh":
        "9ef51d79f9678ac0b2a5fb93d31ac1fe6029763628d991b3d290e6f0c5e8142e",
    "scripts/phase3_select_n.py":
        "1feaa90cd76077a7f53006c8905dd5db5478a700811eaf13edd9a7eaa449009e",
    "scripts/phase3_selection_matrix.py":
        "61dafdc8bf480e10386bb762e0cddf4bd828948377dbf054ca0c0b7c9e7ef4dc",
    "scripts/seal_cell_analysis.py":
        "65be44b77bf3a03a2afcee81c3bb24d3431f6d74d5e00debd878fe11a2a2e695",
    "scripts/seal_phase3_inputs.py":
        "12233f8e4967c90afcab131a1061fe76abfbaed86648e1388826538a6d42f214",
    "scripts/train_cifar10_v185_bidirTokenPrune05_ccs01_clip.sh":
        "a0dff871b7be004f9a614eae324c6613ca6713af591749239c7868ec6965abc9",
    "scripts/train_flickr25k_v185_bidirTokenPrune05_clip.sh":
        "7850783ff03b63035018fa217b05ee7f6fcc7b2778cdbb7e76faf61df7b06726",
    "scripts/train_mscoco_F2_sweep_clip.sh":
        "c4bfb0f8d172c3353b72605ab41d7c78d69b7fba3f4f0745ccaaf1fecfbad18f",
    "scripts/train_nuswide_v185_sweep_clip.sh":
        "d63476d81da232ce4676394116cc050511f7d7f0b8bda637211c538c6b4764a8",
    "tests/test_anchor_confirm_env_handoff.py":
        "54790086c5d01d48b0729072d028d338f936c08e2894f94bc5c2198a17334b4d",
    "tests/test_anchor_confirm_input_bridge.py":
        "80c78c36fbd016974e9bf143541c28074760f8de3ea0ad80375901b55674de17",
    "tests/test_anchor_confirm_launcher.py":
        "c7395d760b3c4bd980286360dc7f78f9fddf7712bce156c10c981fffdb346365",
    "tests/test_anchor_confirm_lifecycle.py":
        "93bca9a4099e6f5aa78eb90411b3bdc6aae2217b059d11fc04945d87a43d96fc",
    "tests/test_anchor_confirm_port.py":
        "d2d5f9550f3ff4202c617b4823f6605fd76cda0bdb7828bbb0ffd4977c13593f",
    "tests/test_anchor_confirm_recipe.py":
        "1dbcce64a3d0694114f00811ea3445cd379fbed35f59051e81d8a53b78ac1148",
    "tests/test_anchor_confirm_reducer.py":
        "b62635cd90a7a2a99d9b3cd0203f9bec3c194aea332da43c862815c8059097b9",
    "tests/test_anchor_confirm_supervisor.py":
        "1ed421d63b85b036ca2ec7650a43434d286a3cdbed5dd58165f5bb3ba2d2f561",
    "tests/test_anchor_lambda_stage.py":
        "2700082fdd67d6209a39ece12190d9cd97ec4b25a702810ffe13065bc13cc5e0",
    "tests/test_seal_phase3_inputs.py":
        "0e66852f6c7573c874ffdb6d75f4e74ccbf50842a163b03f4eed213fcc475773",
    "train_siglip2.py":
        "d23788cb9fea109aaffc191a6208f74869367756c933e985e2c3a1246050586f",
    "val_split.py":
        "95e415c6f43b714fc349c44c5d449e667429bc0f15fb4af9c4389e82ca9eecc2",
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write(path: Path, obj) -> str:
    raw = json.dumps(obj).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return sha(raw)


# =============================================================================================
# predicate: the trainer-side policy (p0_protocol)
# =============================================================================================
R_CELL = "flickr25k|N=4|P=0.6,0.95|JD=0.02|stage=refit|seed=42|axis_center=anchors"
S_CELL = "flickr25k|N=4|P=0.6,0.95|JD=0.02|stage=select|seed=42|axis_center=anchors"
LEGACY_R_CELL = "flickr25k|N=4|P=0.6,0.95|JD=0.02|stage=refit|seed=42"


@pytest.mark.parametrize("cell,stage", [(R_CELL, "refit"), (S_CELL, "select"), (LEGACY_R_CELL, "refit"),
                                        (None, None), ("", None)])
def test_predicate_the_cell_stage_is_read_from_the_cell_id(cell, stage):
    assert P0.campaign_cell_stage(cell) == stage


@pytest.mark.parametrize("cell", ["flickr25k|N=4|seed=42", "a|stage=refit|stage=select", "a|stage="])
def test_predicate_a_cell_id_without_exactly_one_stage_refuses(cell):
    with pytest.raises(ValueError, match="names no single stage"):
        P0.campaign_cell_stage(cell)


@pytest.mark.parametrize("axis", [None, "none"])
@pytest.mark.parametrize("p0_refit", [True, False])
@pytest.mark.parametrize("cell", [None, LEGACY_R_CELL, "flickr25k|N=4|P=0.6,0.95|JD=0.02|stage=select|seed=42"])
def test_predicate_a_model_without_anchors_is_never_withheld(axis, p0_refit, cell):
    assert P0.anchor_refit_withholds_official_test(axis_center=axis, p0_refit_active=p0_refit,
                                                   cell_id=cell, sealed_recipe=False) is False


def test_predicate_a_sealed_anchor_stage_r_cell_withholds_the_official_test():
    assert P0.anchor_refit_withholds_official_test(axis_center="anchors", p0_refit_active=True,
                                                   cell_id=R_CELL, sealed_recipe=True) is True


def test_predicate_stage_1_anchor_cells_are_unchanged():
    assert P0.anchor_refit_withholds_official_test(axis_center="anchors", p0_refit_active=False,
                                                   cell_id=S_CELL, sealed_recipe=True) is False


@pytest.mark.parametrize("axis,p0_refit,cell,sealed,reason", [
    ("anchors", True, None, False, "only as a sealed stage-R campaign cell"),        # marker deleted
    ("anchors", True, S_CELL, True, "only as a sealed stage-R campaign cell"),       # marker altered
    ("anchors", True, LEGACY_R_CELL, True, None),                                    # see below
    ("anchors", False, R_CELL, True, "must be a P0 refit"),                          # not a refit
    ("anchors", True, R_CELL, False, "sealed scientific recipe"),                    # recipe missing
    ("none", True, R_CELL, True, "must train the anchor model"),                     # arm swapped
], ids=["marker-deleted", "marker-altered", "legacy-cell-id", "not-refit", "no-recipe", "arm-swapped"])
def test_predicate_an_anchor_refit_outside_a_sealed_stage_r_cell_refuses(axis, p0_refit, cell, sealed, reason):
    if reason is None:
        # a legacy refit cell id carrying no arm marker is still stage "refit": the anchor model in
        # it is withheld only with a sealed recipe, which the trainer's recipe check already demands
        # of every anchor model inside a campaign (scientific_recipe.verify_trainer_recipe)
        assert P0.anchor_refit_withholds_official_test(axis_center=axis, p0_refit_active=p0_refit,
                                                       cell_id=cell, sealed_recipe=sealed) is True
        return
    with pytest.raises(ValueError, match=reason):
        P0.anchor_refit_withholds_official_test(axis_center=axis, p0_refit_active=p0_refit,
                                                cell_id=cell, sealed_recipe=sealed)


@pytest.mark.parametrize("n", [4, 39])
@pytest.mark.parametrize("final_epoch_eval,evaluation,loader,midtrain,terminal", [
    (True, True, False, False, True),       # the legacy refit: isolated, then the one evaluation
    (False, False, True, True, False),
    (True, False, False, False, False),
])
def test_predicate_audit_743_table(n, final_epoch_eval, evaluation, loader, midtrain, terminal):
    p0_refit = P0.is_p0_refit(stop_after_epoch=n, final_epoch_eval=final_epoch_eval)
    assert P0.should_load_official_test_split(val_protocol_active=False, p0_refit_active=p0_refit) is loader
    assert P0.should_run_training_time_evaluation(p0_refit_active=p0_refit) is midtrain
    assert P0.should_run_official_test_evaluation(evaluation_requested=evaluation,
                                                  val_protocol_active=False) is terminal


# =============================================================================================
# composed: the trainer's start-up boundary and its terminal decision
# =============================================================================================
R_ARGV = RC.ARGV + ["--selection_mode", "refit", "--stop_after_epoch", "4", "--final_epoch_eval",
                    "--val_split_ratio", "0.0", "-ev"]


def r_boundary(monkeypatch, tmp_path, events, binding):
    T = RC.boundary(monkeypatch, tmp_path, events, binding or {})
    if binding is None:                      # no campaign binding in the environment at all
        import dna_utils.run_identity as RI
        monkeypatch.setattr(RI, "phase3_campaign_binding_from_env", lambda identity, actual_tag: None)
    return T


def start(monkeypatch, tmp_path, argv, binding, *, sealed=True):
    from config import Config
    parser = Config.build_parser()
    events = []
    T = r_boundary(monkeypatch, tmp_path, events, binding)
    if sealed:
        RC.seal(monkeypatch, R.build_payload(parser, argv, planned_arm="anchors"))
    else:
        RC.unseal(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["train_siglip2.py", *argv])
    args = RC.post_processed(parser, argv)
    args.date = "261001"
    return T, args, events


def test_composed_a_sealed_stage_r_cell_starts(monkeypatch, tmp_path):
    T, args, events = start(monkeypatch, tmp_path, R_ARGV, {"cell_id": R_CELL})
    T._resolve_save_path(args)
    assert "claim" in events


@pytest.mark.parametrize("binding,sealed,error", [
    ({"cell_id": S_CELL}, True, ValueError),           # stage marker altered to select
    (None, False, ValueError),                         # no campaign at all: never the automatic path
    ({"cell_id": R_CELL}, False, R.RecipeMismatch),    # sealed recipe removed
], ids=["marker-altered", "no-campaign", "no-recipe"])
def test_composed_an_anchor_refit_without_its_stage_r_cell_refuses_before_the_run_directory(
        monkeypatch, tmp_path, binding, sealed, error):
    T, args, events = start(monkeypatch, tmp_path, R_ARGV, binding, sealed=sealed)
    with pytest.raises(error):
        T._resolve_save_path(args)
    assert events == []                       # no directory, no claim, no binding written


def decision_args(**overrides):
    base = dict(evaluation=True, _val_protocol=False, axis_center="anchors",
                _phase3_campaign_binding={"cell_id": R_CELL, "scientific_recipe_sha256": "f" * 64})
    base.update(overrides)
    return SimpleNamespace(**base)


def test_composed_the_terminal_decision_withholds_a_stage_r_cell(capsys):
    import train_siglip2 as T
    assert T._official_test_allowed(decision_args(), p0_refit=True) is False
    assert "WITHHELD" in capsys.readouterr().out


@pytest.mark.parametrize("args,p0_refit,allowed", [
    (decision_args(axis_center="none", _phase3_campaign_binding=None), True, True),      # legacy refit
    (decision_args(axis_center="none", _phase3_campaign_binding=None, evaluation=False), True, False),
    (decision_args(_val_protocol=True, _phase3_campaign_binding={"cell_id": S_CELL,
                                                                  "scientific_recipe_sha256": "f" * 64}),
     False, False),                                                                    # stage 1
])
def test_composed_the_terminal_decision_is_unchanged_elsewhere(args, p0_refit, allowed):
    import train_siglip2 as T
    assert T._official_test_allowed(args, p0_refit=p0_refit) is allowed


def _main_function():
    tree = ast.parse((REPO / "train_siglip2.py").read_text())
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")


def test_structural_the_official_test_runs_only_under_the_terminal_decision():
    main = _main_function()
    calls = [n for n in ast.walk(main) if isinstance(n, ast.Call)
             and getattr(n.func, "id", getattr(n.func, "attr", None)) == "run_official_test"]
    assert len(calls) == 1
    guards = [n for n in ast.walk(main) if isinstance(n, ast.If)
              and any(c is calls[0] for c in ast.walk(n))]
    assert any(isinstance(g.test, ast.Name) and g.test.id == "_official_test_eval_allowed" for g in guards)
    assigned = [n for n in ast.walk(main) if isinstance(n, ast.Assign)
                and any(getattr(t, "id", None) == "_official_test_eval_allowed" for t in n.targets)]
    assert len(assigned) == 1 and isinstance(assigned[0].value, ast.Call) \
        and assigned[0].value.func.id == "_official_test_allowed"


def test_structural_the_terminal_function_is_the_v8_block_moved_unchanged():
    old = subprocess.run(["git", "show", f"{V8_COMMIT}:train_siglip2.py"], cwd=REPO,
                         capture_output=True, text=True, check=True).stdout.splitlines()
    start_ = old.index("        from extraction_siglip2 import extract_code as _extract_code")
    end = next(i for i, line in enumerate(old) if line.startswith(
        "        # Restore (post-eval compositional uses _cache below"))
    old_block = [line[8:] if line.strip() else "" for line in old[start_:end]]
    new = (REPO / "terminal_official_test.py").read_text().splitlines()
    first = new.index("    from extraction_siglip2 import extract_code as _extract_code")
    new_block = [line[4:] if line.strip() else "" for line in new[first:]]
    new_block = [line.replace("codebook_size=codebook_size,", "codebook_size=codebook_size_cf,")
                 for line in new_block]
    while new_block and not new_block[-1]:
        new_block.pop()
    while old_block and not old_block[-1]:
        old_block.pop()
    assert new_block == old_block


def test_composed_the_terminal_function_runs_extraction_then_the_raw_evaluator(monkeypatch, tmp_path):
    import extraction_siglip2
    import evaluation_siglip2
    import terminal_official_test as TOT
    calls = []
    monkeypatch.setattr(extraction_siglip2, "extract_code", lambda args: calls.append(
        ("extract", args.siglip2_feature_cache_dir, args.text_whiten_npz)))
    monkeypatch.setattr(evaluation_siglip2, "evaluation", lambda path, **k: calls.append(("evaluate", path, k)))
    cache = tmp_path / "cache"
    cache.mkdir()
    args = SimpleNamespace(eval_cache_dir=str(cache), siglip2_feature_cache_dir="/train/cache",
                           text_whiten_npz=None, eval_text_whiten_npz=None,
                           save_result_path=str(tmp_path), dataset="Flickr25k")
    TOT.run_official_test(args, distance_mode="base", codebook_size=128)
    assert calls[0] == ("extract", str(cache), None)
    assert calls[1][0] == "evaluate" and calls[1][1] == str(tmp_path)
    assert calls[1][2] == {"distance_mode": "base", "codebook_size": 128,
                           "map_at_r": evaluation_siglip2.resolve_map_at_r("Flickr25k"),
                           "dataset_name": "Flickr25k"}          # raw: bio_project left False
    assert args.siglip2_feature_cache_dir == str(cache)           # as the inline block left it


# =============================================================================================
# composed: the raw evaluator before the BIO stage, through the real producers
# =============================================================================================
def _synthetic_run(tmp_path, monkeypatch, *, raw: bool):
    from tests.test_extraction_run_validation import _cell
    import extraction_siglip2
    import terminal_official_test as TOT
    cell = _cell(tmp_path, splits=("db", "query", "train"))
    if raw:
        # the stage-T function, with model encoding replaced by the synthetic extraction above
        monkeypatch.setattr(extraction_siglip2, "extract_code", lambda args: None)
        TOT.run_official_test(SimpleNamespace(eval_cache_dir=None, siglip2_feature_cache_dir=None,
                                              text_whiten_npz=None, eval_text_whiten_npz=None,
                                              save_result_path=str(cell), dataset="CIFAR10"),
                              distance_mode="base", codebook_size=64)
    return cell


def _bio_stage(monkeypatch, cell):
    import scripts.eval_cell_bioproj as bio
    monkeypatch.setattr(sys, "argv", [str(bio.__file__), "--dir", str(cell), "--dataset", "CIFAR10",
                                      "--K", "64", "--require-train"])
    bio.main()


def test_composed_the_bio_stage_refuses_without_the_raw_evaluation(tmp_path, monkeypatch):
    cell = _synthetic_run(tmp_path, monkeypatch, raw=False)
    with pytest.raises((SystemExit, Exception)):
        _bio_stage(monkeypatch, cell)
    assert not (cell / "evaluation_siglip2_base_bioproj.json").exists()
    assert not (cell / "cell_result.json").exists()


def test_composed_the_raw_evaluation_then_the_bio_stage_completes(tmp_path, monkeypatch):
    cell = _synthetic_run(tmp_path, monkeypatch, raw=True)
    raw = json.loads((cell / "evaluation_siglip2_base.json").read_text())
    assert raw.get("bio_project") in (False, None)
    _bio_stage(monkeypatch, cell)
    result = json.loads((cell / "cell_result.json").read_text())
    assert set(result["evaluation_artifacts"]) == {"raw", "bio_projected"}
    assert (cell / "evaluation_siglip2_base_bioproj.json").exists()


# =============================================================================================
# composed: the launcher -- commands, F authority, admission, main()
# =============================================================================================
def test_composed_an_anchor_refit_command_is_the_legacy_refit_plus_the_arm():
    cmd, env, tag = M.build_command("flickr25k", 4, 0, stage="refit", seed=43, topp=("0.6", "0.95"),
                                    joint="0.02", anchor_arm="anchors")
    lcmd, lenv, ltag = M.build_command("flickr25k", 4, 0, stage="refit", seed=43, topp=("0.6", "0.95"),
                                       joint="0.02")
    assert cmd == lcmd
    assert LT.extra(env)[-2:] == ["--axis_center", "anchors"] and LT.extra(env)[:-2] == LT.extra(lenv)
    assert {k: v for k, v in env.items() if k not in ("EXTRA_ARGS", "TAG")} == \
        {k: v for k, v in lenv.items() if k not in ("EXTRA_ARGS", "TAG")}
    assert env["VAL_RATIO"] == "0.0" and env["FINAL_EPOCH"] == "1" and env["STOP_EP"] == "4"
    assert env["WHITEN_NPZ"].endswith("text_whiten_trainOnly_localOnly.npz")
    assert LT.extra(env)[:2] == ["-e", "5"] and "--selection_mode" in LT.extra(env)
    assert tag.endswith("_refit_N4_s43_AXanchors_P06095_JD002") and ltag == tag.replace("_AXanchors", "")
    assert tag not in ltag and ltag not in tag


def test_composed_an_anchor_refit_carries_no_lambda_override():
    with pytest.raises(CellRefused, match="no lambda override"):
        M.build_command("flickr25k", 4, 0, stage="refit", anchor_arm="anchors",
                        overrides=(("lambda_bu", "0"),))


def refit_render(cmd, env):
    """The stand-in render with the composition of the real wrapper: its body passes
    `--text_whiten_npz "$WHITEN_NPZ"`, FINAL_EPOCH expands to --final_epoch_eval in the body, and
    `-ev -s` follow EXTRA_ARGS."""
    argv = LT.fake_render(cmd, env)
    tail = len(LT.extra(env))
    body = argv[:len(argv) - tail] + ["--text_whiten_npz", env["WHITEN_NPZ"]]
    if env.get("FINAL_EPOCH"):
        body = body + ["--final_epoch_eval"]
    return body + argv[len(argv) - tail:] + ["-ev", "-s"]


class FWorld:
    """A synthetic accepted F record: its validated recipes are the stage-S seed-42 renders of the
    same stand-in wrapper, in a synthetic v7 snapshot; ledger section 744 accepts it."""

    def __init__(self, root: Path, monkeypatch, *, ledger_text=None, mutate=None):
        root.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(M, "anchor_incumbent", lambda read=None: INC)
        monkeypatch.setattr(M, "render_trainer_argv", refit_render)
        bindings, entries = {}, {}
        for ds in M.ANCHOR_DATASETS:
            n = FROZEN_N[ds]
            topp, joint = INC[ds]["topp"], INC[ds]["joint"]
            payload = M.anchor_scientific_recipe(ds, n, namespace="ancS7", stage="select", seed=42, topp=topp,
                                                 joint=joint, anchor_arm="anchors", input_authority=RW.AUTH[ds])
            cid = M.campaign_cell_id(ds, n, topp=topp, joint=joint, stage="select", seed=42, anchor_arm="anchors")
            bindings[cid] = {"N": n, "scientific_recipe": payload,
                             "expected_scientific_recipe_sha256": R.digest(payload)}
            f = payload["fields"]
            entries[ds] = {"N": n, "routing_adaptive_topp": [f["routing_adaptive_topp_min"],
                                                             f["routing_adaptive_topp_max"]],
                           "lambda_codon_joint": f["lambda_codon_joint"],
                           "lambdas": {k: f[k] for k in ("lambda_wasserstein", "lambda_bu",
                                                         "lambda_text_hash_ntxent")},
                           "validated_recipe": {"cell_id": cid, "scientific_recipe_sha256": R.digest(payload)}}
        snapshot = root / "ancS7_snapshot.json"
        snap_sha = write(snapshot, {"plan": {"cell_bindings": bindings}})
        for e in entries.values():
            e["validated_recipe"]["snapshot"] = {"path": str(snapshot), "sha256": snap_sha}
        self.record = {"artifact_kind": RT.ANCHOR_F_KIND,
                       "fixed_architecture": {"axis_center": "anchors", "datasets": list(M.ANCHOR_DATASETS)},
                       "datasets": entries}
        if mutate:
            mutate(self)
        path = root / "ancF_candidate_v1.json"
        self.sha = write(path, self.record)
        monkeypatch.setattr(RT, "ANCHOR_F_RECORD", path)
        monkeypatch.setattr(RT, "ANCHOR_F_RECORD_SHA256", self.sha)
        text = ledger_text if ledger_text is not None else (
            f"## 744. Accepted\n\nAccepted record:\n{path},\nSHA256 `{self.sha}`.\n")
        ledger = root / "ledger.md"
        ledger.write_text("# ledger\n\n" + text)
        monkeypatch.setattr(M, "AUDIT_LEDGER", ledger)
        self.root, self.path, self.bindings = root, path, bindings


@pytest.fixture
def fworld(tmp_path, monkeypatch):
    return FWorld(tmp_path / "f", monkeypatch)


def test_composed_the_f_authority_is_the_record_and_its_acceptance(fworld):
    freeze = RT.anchor_freeze_authority()
    assert freeze["record"] == {"path": str(fworld.path), "sha256": fworld.sha}
    assert {ds: e["N"] for ds, e in freeze["datasets"].items()} == FROZEN_N
    assert freeze["datasets"]["mscoco"]["fields"]["axis_center"] == "anchors"


@pytest.mark.parametrize("ledger_text,reason", [
    ("## 744. Something else\n\nnothing accepted here\n", "does not accept"),
    ("## 743. Accepted record: elsewhere\n", "sections numbered 744"),
], ids=["no-acceptance", "no-section"])
def test_composed_an_f_record_without_its_acceptance_refuses(tmp_path, monkeypatch, ledger_text, reason):
    FWorld(tmp_path / "f", monkeypatch, ledger_text=ledger_text)
    with pytest.raises(CellRefused, match=reason):
        RT.anchor_freeze_authority()


def _drop_coco(world):
    del world.record["datasets"]["mscoco"]


def _copy_flickr_lambda_to_coco(world):
    world.record["datasets"]["mscoco"]["lambdas"]["lambda_wasserstein"] = 0.15


def _wrong_recipe_digest(world):
    world.record["datasets"]["nuswide"]["validated_recipe"]["scientific_recipe_sha256"] = "0" * 64


def _old_architecture(world):
    world.record["fixed_architecture"]["axis_center"] = "none"


@pytest.mark.parametrize("mutate,reason", [
    (_drop_coco, "exactly the four datasets"), (_copy_flickr_lambda_to_coco, "summary values"),
    (_wrong_recipe_digest, "not the F record's digest"), (_old_architecture, "exactly the four datasets"),
], ids=["dropped-dataset", "copied-lambda", "recipe-digest", "old-architecture"])
def test_composed_a_corrupt_f_record_refuses(tmp_path, monkeypatch, mutate, reason):
    FWorld(tmp_path / "f", monkeypatch, mutate=mutate)
    with pytest.raises(CellRefused, match=reason):
        RT.anchor_freeze_authority()


def test_composed_an_f_record_at_other_bytes_refuses(fworld, monkeypatch):
    monkeypatch.setattr(RT, "ANCHOR_F_RECORD_SHA256", "1" * 64)
    with pytest.raises(CellRefused, match="pinned bytes"):
        RT.anchor_freeze_authority()


def test_composed_the_refit_plan_is_twelve_cells_at_the_frozen_n(fworld):
    cells = RT.refit_cells(RT.anchor_freeze_authority(), INC)
    assert [(c[0], c[1], c[5]) for c in cells] == [(ds, FROZEN_N[ds], s) for ds in M.ANCHOR_DATASETS
                                                   for s in (42, 43, 44)]
    assert {(c[4], c[6], c[7]) for c in cells} == {("refit", (), "anchors")}


@pytest.mark.parametrize("epochs", [None, 1])
def test_composed_every_refit_recipe_is_the_f_recipe_with_the_contracted_fields_only(fworld, epochs):
    freeze = RT.anchor_freeze_authority()
    cells = RT.refit_cells(freeze, INC)
    report = RT.refit_admission(cells, namespace="ancRT", incumbent=INC, freeze=freeze, epochs=epochs)
    assert len(report) == 12
    for key, entry in report.items():
        # a preview renders without the admitted seals: the input-authority fields are not compared
        assert entry["input_authority_compared"] is False
        assert set(entry["differs_from_f"]) <= set(RT.REFIT_PROTOCOL_FIELDS) | set(M.INPUT_AUTHORITY_DESTS)
        assert "axis_center" not in entry["differs_from_f"]
    coco = report["mscoco|N=39|seed=43"]["differs_from_f"]
    assert {"epoch", "val_split_ratio", "selection_mode", "final_epoch_eval", "random_seed"} <= set(coco)


def _refit_seals(files=None):
    """Admitted refit seal authorities, as verify_campaign_input_seals returns them; `files` maps a
    dataset to (path, sha256) of the seal file the request pins."""
    files = files or {ds: (Path(f"/seals/{ds}.refit.input-seal.json"), sha(f"refit seal {ds}".encode()))
                      for ds in M.ANCHOR_DATASETS}
    return {f"{ds}:refit": dict(RW.AUTH[ds], stage="refit", seal_path=str(files[ds][0].resolve()),
                                seal_file_sha256=files[ds][1])
            for ds in M.ANCHOR_DATASETS}


def test_composed_with_the_admitted_seals_the_sealed_inputs_are_the_refit_seals(fworld):
    freeze = RT.anchor_freeze_authority()
    cells = RT.refit_cells(freeze, INC)
    report = RT.refit_admission(cells, namespace="ancRT", incumbent=INC, freeze=freeze, input_seals=_refit_seals())
    for e in report.values():
        assert e["input_authority_compared"] is True
        assert {"phase3_input_seal", "phase3_input_seal_sha256"} <= set(e["differs_from_f"])
        assert set(e["differs_from_f"]) <= set(RT.REFIT_PROTOCOL_FIELDS) | set(RT.REFIT_SEAL_FIELDS)


def test_composed_a_stage_1_seal_cannot_stand_in_for_the_refit_seal(fworld):
    freeze = RT.anchor_freeze_authority()
    seals = {k.replace(":refit", ":refit"): dict(v, stage="stage1") for k, v in _refit_seals().items()}
    with pytest.raises(CellRefused, match="not the admitted refit seal"):
        RT.refit_admission(RT.refit_cells(freeze, INC), namespace="ancRT", incumbent=INC, freeze=freeze,
                           input_seals=seals)


@pytest.mark.parametrize("render,reason", [
    (lambda cmd, env: refit_render(cmd, env) + ["--lambda_bu", "0.5"], "lambda_bu"),
    (lambda cmd, env: [t for t in refit_render(cmd, env) if t != "-ev"], "evaluation"),
    (lambda cmd, env: [t for t in refit_render(cmd, env) if t != "--final_epoch_eval"], "final_epoch_eval"),
], ids=["lambda", "no-ev", "no-final-epoch-eval"])
def test_composed_a_refit_render_off_the_mapping_refuses(fworld, monkeypatch, render, reason):
    monkeypatch.setattr(M, "render_trainer_argv", render)
    freeze = RT.anchor_freeze_authority()
    with pytest.raises(CellRefused, match=reason):
        RT.refit_admission(RT.refit_cells(freeze, INC), namespace="ancRT", incumbent=INC, freeze=freeze)


# ---- main(): the stage-R request, its approval and the boundaries it reaches -----------------------
@pytest.fixture
def rmain(tmp_path, monkeypatch, fworld):
    calls, kwargs = [], {}
    for name in ("_run_managed_process", "reserve_sweep_namespace", "_publish_json_exclusive"):
        monkeypatch.setattr(M, name, lambda *a, _n=name, **k: calls.append(_n) or 0)
    seal_files = {}
    for ds in M.ANCHOR_DATASETS:
        path = tmp_path / "seals" / f"{ds}.refit.input-seal.json"
        seal_files[ds] = (path, write(path, {"seal": ds}))
    seals = _refit_seals(seal_files)
    monkeypatch.setattr(M, "verify_campaign_input_seals",
                        lambda *a, **k: calls.append("verify_campaign_input_seals") or dict(seals))
    monkeypatch.setattr(M, "_with_campaign_gpu_leases",
                        lambda args, fn: calls.append("_with_campaign_gpu_leases") or fn())

    def run_sweep(*a, **k):
        calls.append("_run_sweep")
        kwargs.update(k)
        return 0
    monkeypatch.setattr(M, "_run_sweep", run_sweep)
    monkeypatch.setattr(RT, "REFIT_INPUT_SEALS", {ds: p for ds, (p, _) in seal_files.items()})
    monkeypatch.setattr(RT, "REFIT_INPUT_SEALS_SHA256", {ds: s for ds, (_, s) in seal_files.items()})
    path, digest = manifest(tmp_path)
    base = ["--anchor-confirm", "refit", "--namespace", "ancRT", "--anchor-manifest", path,
            "--anchor-manifest-sha256", digest,
            *[a for ds, (p, _) in seal_files.items() for a in ("--input-seal", f"{ds}:refit={p}")]]
    return SimpleNamespace(calls=calls, kwargs=kwargs, base=base, manifest=digest, fworld=fworld)


def manifest(tmp_path):
    path, digest = LT.manifest(tmp_path)
    return path, digest


RUN = ["--run", "--gpus", "0,1,2,3"]
SMOKE = ["--smoke", "--only", "flickr25k:4:anchors:42", "--epochs", "1", "--gpus", "0"]


def preview(monkeypatch, capsys, argv):
    assert LT.run_main(monkeypatch, *argv, "--plan") == 0
    out = capsys.readouterr().out
    request = json.loads(out[out.index("{", out.index("would need approved")):out.index("execution request sha256")])
    return request, M._json_digest(request)


def approve(tmp_path, monkeypatch, scope, manifest_sha, request_digest, *, freeze, section=750):
    LT.ledger(tmp_path, monkeypatch, {
        744: [], section: [LT.approval_line(scope, manifest=manifest_sha, freeze=freeze, request=request_digest)]})
    # keep the F acceptance in the synthetic ledger
    ledger = Path(M.AUDIT_LEDGER)
    ledger.write_text(ledger.read_text().replace(
        "## 744. Test decision\n", f"## 744. Test decision\n\nAccepted record:\n{RT.ANCHOR_F_RECORD},\n"
                                   f"SHA256 `{RT.ANCHOR_F_RECORD_SHA256}`.\n"))
    return ["--anchor-approval-section", str(section)]


def test_composed_the_plan_previews_the_stage_r_request_and_touches_nothing(rmain, monkeypatch, capsys):
    request, digest = preview(monkeypatch, capsys, [*rmain.base, *RUN])
    assert request["stage"] == "refit" and request["mode"] == "run"
    assert request["freeze"]["sha256"] == rmain.fworld.sha
    assert len(request["cells"]) == 12 and request["gpu_count"] == 4
    assert rmain.calls == []


@pytest.mark.parametrize("mode", [RUN, SMOKE])
def test_composed_an_approved_stage_r_verifies_inputs_then_leases_and_sweeps(tmp_path, rmain, monkeypatch,
                                                                            capsys, mode):
    request, digest = preview(monkeypatch, capsys, [*rmain.base, *mode])
    scope = "stage-R-smoke" if "--smoke" in mode else "stage-R-run"
    extra = approve(tmp_path, monkeypatch, scope, rmain.manifest, digest, freeze=rmain.fworld.sha)
    assert LT.run_main(monkeypatch, *rmain.base, *mode, *extra) == 0
    assert rmain.calls == ["verify_campaign_input_seals", "_with_campaign_gpu_leases", "_run_sweep"]
    auth = rmain.kwargs["authorities"]
    assert auth["anchor_request"] == request and auth["anchor_freeze"]["sha256"] == rmain.fworld.sha
    assert all(c[4] == "refit" and c[7] == "anchors" for c in rmain.kwargs["full_plan"])


@pytest.mark.parametrize("scope", ["stage-S-run", "stage-L-run", "stage-T-run", "stage-R-smoke"])
def test_composed_another_scope_does_not_approve_stage_r(tmp_path, rmain, monkeypatch, capsys, scope):
    _, digest = preview(monkeypatch, capsys, [*rmain.base, *RUN])
    extra = approve(tmp_path, monkeypatch, scope, rmain.manifest, digest, freeze=rmain.fworld.sha)
    assert LT.run_main(monkeypatch, *rmain.base, *RUN, *extra) == 2
    assert rmain.calls == []


def test_composed_a_run_scope_line_does_not_approve_a_stage_r_smoke(tmp_path, rmain, monkeypatch, capsys):
    """A stage-R-run line naming the smoke's own request digest still does not approve the smoke: the
    scope follows the mode (battery v13 attempt 1, RX13: a stage-S scope refuses every line because it
    carries no F pin, so it could not show this)."""
    _, digest = preview(monkeypatch, capsys, [*rmain.base, *SMOKE])
    extra = approve(tmp_path, monkeypatch, "stage-R-run", rmain.manifest, digest, freeze=rmain.fworld.sha)
    assert LT.run_main(monkeypatch, *rmain.base, *SMOKE, *extra) == 2
    assert rmain.calls == []


def test_composed_an_approval_naming_another_f_record_refuses(tmp_path, rmain, monkeypatch, capsys):
    _, digest = preview(monkeypatch, capsys, [*rmain.base, *RUN])
    extra = approve(tmp_path, monkeypatch, "stage-R-run", rmain.manifest, digest, freeze="2" * 64)
    assert LT.run_main(monkeypatch, *rmain.base, *RUN, *extra) == 2
    assert rmain.calls == []


def test_composed_stage_r_takes_only_the_approved_refit_seals(tmp_path, rmain, monkeypatch, capsys):
    other = tmp_path / "other.json"
    write(other, {"seal": "other"})
    base = [a if not a.startswith("flickr25k:refit=") else f"flickr25k:refit={other}" for a in rmain.base]
    _, digest = preview(monkeypatch, capsys, [*base, *RUN])
    extra = approve(tmp_path, monkeypatch, "stage-R-run", rmain.manifest, digest, freeze=rmain.fworld.sha)
    assert LT.run_main(monkeypatch, *base, *RUN, *extra) == 2
    assert "approved refit input seals" in capsys.readouterr().err
    assert rmain.calls == []


@pytest.mark.parametrize("swap", ["v7", "v8"])
def test_composed_stages_r_and_t_never_run_under_an_older_generation(rmain, monkeypatch, capsys, swap):
    pinned = M.ANCHOR_V7_MANIFEST_SHA256 if swap == "v7" else RT.ANCHOR_V8_MANIFEST_SHA256
    monkeypatch.setattr(M, "load_anchor_manifest", lambda path, digest: {"path": path, "sha256": pinned})
    assert LT.run_main(monkeypatch, *rmain.base, *RUN, "--plan") == 2
    assert "generation v9" in capsys.readouterr().err


def test_composed_the_refit_stage_takes_no_refit_receipt(rmain, monkeypatch, capsys):
    assert LT.run_main(monkeypatch, *rmain.base, "--anchor-refit-receipt", "/x", "--anchor-refit-receipt-sha256",
                       "0" * 64, "--plan") == 2
    assert "belongs to --anchor-confirm test" in capsys.readouterr().err


# ---- run_cell: a stage-R cell ends at its checkpoint ----------------------------------------------
@pytest.fixture
def rcell(tmp_path, monkeypatch):
    """run_cell with the trainer process and the post-trainer checks replaced by recorders; the
    command, binding projection, environment and record logic are the real ones."""
    calls = []
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    records = tmp_path / "records"
    records.mkdir()
    monkeypatch.setattr(M, "RECORD_DIR", records)
    monkeypatch.setattr(M, "_existing_artifacts", lambda tag, result_root=None: [])
    monkeypatch.setattr(M, "protocol_digests", lambda dataset=None: {"x": "y"})
    monkeypatch.setattr(M, "verify_snapshot", lambda *a, **k: None)
    monkeypatch.setattr(M, "_bind_campaign_gpu_selector", lambda cmd, env, **k: cmd)
    monkeypatch.setattr(M, "_resolve", lambda tag, **k: run_dir)
    monkeypatch.setattr(M, "assert_completed", lambda run_dir, **k: {"final_checkpoint_epoch_zero_based": 4})
    monkeypatch.setattr(M, "assert_geometry", lambda run_dir, **k: {"identity_digest": "i"})
    monkeypatch.setattr(M, "assert_anchor_recipe", lambda run_dir, **k: {"arm": "anchors"})
    monkeypatch.setattr(M, "_manifest_facts", lambda run_dir: {})
    monkeypatch.setattr(M, "_run_refit_postprocess", lambda *a, **k: calls.append("postprocess"))
    monkeypatch.setattr(M, "assert_refit_outputs", lambda *a, **k: calls.append("refit_outputs") or {})
    monkeypatch.setattr(M, "_run_managed_process",
                        lambda command, **k: calls.append(("trainer", command)) or SimpleNamespace(returncode=0))
    snapshot = {"plan": {"campaign_nonce": "n" * 64, "cell_bindings": {"c": {}}}, "environment_sha256": "e" * 64}
    snap_digest = M._json_digest(snapshot)

    def go(anchor_arm="anchors", stage="refit"):
        tag = M.refit_tag_for("flickr25k", 4, 42, namespace="ancRT", topp=("0.6", "0.95"), joint="0.02",
                              anchor_arm=anchor_arm)
        binding = {"cell_id": "c", "expected_tag": tag, "plan_snapshot_sha256": snap_digest,
                   "campaign_nonce": "n" * 64, "result_root": str(M._canonical_result_root(str(tmp_path))),
                   "environment_sha256": "e" * 64, "physical_gpu_index": 0,
                   "child_environment_sha256": "c" * 64, "expected_identity_digest": "i",
                   "expected_child_environment": {"physical_gpu": {"uuid": "GPU-x"}},
                   **{k: None for k in ("input_authority_sha256", "input_seal_sha256",
                                        "input_aggregate_sha256", "split_identity_sha256",
                                        "hf_identity_sha256")}}
        monkeypatch.setattr(M, "launch_binding_from_expected", lambda planned, digest: binding)
        return M.run_cell("flickr25k", 4, 0, namespace="ancRT", stage=stage, seed=42, topp=("0.6", "0.95"),
                          joint="0.02", snapshot=snapshot, campaign_binding=binding,
                          result_root=str(tmp_path), anchor_arm=anchor_arm,
                          scientific_recipe=({"fields": {}} if anchor_arm else None))
    return SimpleNamespace(go=go, calls=calls, run_dir=run_dir, records=records)


def test_composed_a_stage_r_cell_runs_no_post_chain_and_records_the_test_withheld(rcell):
    record = rcell.go()
    assert "postprocess" not in rcell.calls and "refit_outputs" not in rcell.calls
    assert record["completion"]["official_test"]["status"] == "withheld"
    assert record["tag"].endswith("_AXanchors_P06095_JD002")


@pytest.mark.parametrize("name", list(M.OFFICIAL_TEST_OUTPUTS))
def test_composed_a_stage_r_run_directory_holding_any_test_output_refuses(rcell, name):
    (rcell.run_dir / name).write_bytes(b"x")
    with pytest.raises(CellRefused, match="never reaches the official test"):
        rcell.go()
    assert list(rcell.records.iterdir()) == []


def test_composed_a_legacy_refit_still_runs_its_post_chain(rcell):
    rcell.go(anchor_arm=None)
    assert "postprocess" in rcell.calls and "refit_outputs" in rcell.calls


# =============================================================================================
# composed: stage T -- the receipt admission, the attempt, the entry, the chain
# =============================================================================================
class RWorld:
    """A completed synthetic stage-R campaign: receipt, snapshot (manifest, F, request, approval,
    cell bindings), twelve records and run directories with a checkpoint, trainer evidence and
    runtime witness; the ledger approves the stage-R request and accepts F."""

    def __init__(self, root: Path, monkeypatch, fworld, manifest_sha, *, mode="run", break_cell=None):
        from dna_utils.run_identity import PHASE3_CAMPAIGN_BINDING_NAME
        records = root / "records"
        records.mkdir(parents=True)
        monkeypatch.setattr(M, "ANCHOR_RECORD_DIR", records)
        monkeypatch.setattr(M, "RECORD_DIR", records)
        coords = [(ds, FROZEN_N[ds], s) for ds in M.ANCHOR_DATASETS for s in (42, 43, 44)]
        if mode == "smoke":
            coords = [("flickr25k", 4, 42)]
        request = {"stage": "refit", "mode": mode, "freeze": {"sha256": RT.ANCHOR_F_RECORD_SHA256},
                   "cells": [[ds, "anchors", n, s] for ds, n, s in coords]}
        line = LT.approval_line(f"stage-R-{mode}", manifest=manifest_sha, freeze=RT.ANCHOR_F_RECORD_SHA256,
                                request=M._json_digest(request))
        self.ledger_sections = {760: [line]}
        bindings, cells = {}, {}
        payload = {"argv": ["--axis_center", "anchors"], "fields": {"axis_center": "anchors"},
                   "schema": R.RECIPE_SCHEMA}
        for ds, n, seed in coords:
            cid = M.campaign_cell_id(ds, n, topp=INC[ds]["topp"], joint=INC[ds]["joint"], stage="refit",
                                     seed=seed, anchor_arm="anchors")
            tag = M.refit_tag_for(ds, n, seed, namespace="ancR9", topp=INC[ds]["topp"], joint=INC[ds]["joint"],
                                  anchor_arm="anchors")
            run = root / "result" / tag
            run.mkdir(parents=True)
            ckpt_sha = write(run / "model_state_dict.pth", {"weights": tag})
            evidence_sha = write(run / PHASE3_CAMPAIGN_BINDING_NAME, {"cell_id": cid})
            sidecar_sha = write(run / "model_state_dict.pth.runtime.json", {"checkpoint_epoch_zero_based": n})
            bindings[cid] = {"anchor_arm": "anchors", "stage": "refit", "scientific_recipe": payload,
                             "expected_scientific_recipe_sha256": R.digest(payload)}
            campaign = {"cell_id": cid}
            record = {"stage": "refit", "dataset": ds, "N": n, "seed": seed, "tag": tag, "run_dir": str(run),
                      "smoke": mode == "smoke", "campaign": campaign,
                      "anchor_confirmation": {"scientific_recipe_sha256": R.digest(payload)},
                      "completion": {"official_test": {"status": "withheld"},
                                     "final_checkpoint": "model_state_dict.pth",
                                     "final_checkpoint_epoch_zero_based": n,
                                     "final_checkpoint_sha256": ckpt_sha,
                                     "checkpoint_runtime_sha256": sidecar_sha,
                                     "phase3_campaign_evidence_sha256": evidence_sha}}
            if break_cell and (ds, seed) == break_cell[0]:
                break_cell[1](record, run)
            rec_sha = write(records / f"{tag}.json", record)
            cells[cid] = {"record": f"{tag}.json", "record_sha256": rec_sha, "campaign": campaign}
        snapshot = {"plan": {"campaign_nonce": "n" * 64, "cell_bindings": bindings, "authorities": {
            "anchor_manifest": {"sha256": manifest_sha}, "anchor_freeze": {"sha256": RT.ANCHOR_F_RECORD_SHA256},
            "anchor_request": request, "anchor_approval": {"section": 760, "line": line}}}}
        snap_file = "ancR9_snapshot_x.json"
        write(records / snap_file, snapshot)
        self.receipt = records / "ancR9_sweep_complete.json"
        self.receipt_sha = write(self.receipt, {
            "campaign_kind": M.ANCHOR_CAMPAIGN_KIND, "anchor_confirmation": {"stage": "refit"},
            "plan_snapshot_file": snap_file, "plan_snapshot_sha256": M._json_digest(snapshot),
            "campaign_nonce": "n" * 64, "cells": cells})
        self.root, self.records, self.coords, self.fworld = root, records, coords, fworld


def full_ledger(tmp_path, monkeypatch, sections):
    LT.ledger(tmp_path, monkeypatch, {744: [], **sections})
    ledger = Path(M.AUDIT_LEDGER)
    ledger.write_text(ledger.read_text().replace(
        "## 744. Test decision\n", f"## 744. Test decision\n\nAccepted record:\n{RT.ANCHOR_F_RECORD},\n"
                                   f"SHA256 `{RT.ANCHOR_F_RECORD_SHA256}`.\n"))


@pytest.fixture
def rworld(tmp_path, monkeypatch, fworld):
    world = RWorld(tmp_path / "rt", monkeypatch, fworld, "m" * 64)
    full_ledger(tmp_path, monkeypatch, world.ledger_sections)
    return world


def admit(world, mode="run", manifest_sha="m" * 64):
    return RT.admit_refit_receipt(world.receipt, world.receipt_sha, manifest_sha256=manifest_sha,
                                  freeze=RT.anchor_freeze_authority(), mode=mode)


def test_composed_the_t_admission_takes_exactly_the_twelve_stage_r_checkpoints(rworld):
    refit = admit(rworld)
    assert sorted((c["dataset"], c["N"], c["seed"]) for c in refit["cells"]) == sorted(rworld.coords)
    assert refit["approval"]["scope"] == "stage-R-run"


def _present_output(record, run):
    (run / "evaluation_siglip2_base.json").write_text("{}")


def _not_withheld(record, run):
    record["completion"]["official_test"] = {"status": "done"}


def _wrong_epoch(record, run):
    record["completion"]["final_checkpoint_epoch_zero_based"] = 3


@pytest.mark.parametrize("breaker,reason", [
    (_present_output, "never reaches the official test"), (_not_withheld, "test withheld"),
    (_wrong_epoch, "terminal checkpoint witness"),
], ids=["output-present", "not-withheld", "wrong-epoch"])
def test_composed_a_stage_r_cell_that_is_not_withheld_and_terminal_refuses(tmp_path, monkeypatch, fworld,
                                                                          breaker, reason):
    world = RWorld(tmp_path / "rt", monkeypatch, fworld, "m" * 64, break_cell=(("nuswide", 43), breaker))
    full_ledger(tmp_path, monkeypatch, world.ledger_sections)
    with pytest.raises(CellRefused, match=reason):
        admit(world)


def test_composed_a_stage_r_approval_no_longer_in_the_ledger_refuses(tmp_path, monkeypatch, rworld):
    full_ledger(tmp_path, monkeypatch, {760: []})
    with pytest.raises(CellRefused, match="approval lines"):
        admit(rworld)


@pytest.mark.parametrize("mode,manifest_sha,reason", [
    ("smoke", "m" * 64, "not a smoke"), ("run", "o" * 64, "not a run"),
])
def test_composed_a_receipt_of_another_mode_or_generation_refuses(rworld, mode, manifest_sha, reason):
    with pytest.raises(CellRefused, match=reason):
        admit(rworld, mode=mode, manifest_sha=manifest_sha)


# ---- the attempt reservation and the T cell ------------------------------------------------------
T_REQ = {"schema": RT.T_REQUEST_SCHEMA, "stage": "test", "mode": "run", "namespace": "ancT9"}


def t_request(world, manifest_path, manifest_sha):
    refit = admit(world)
    return dict(T_REQ, manifest=manifest_sha, manifest_path=manifest_path,
                freeze={"sha256": RT.ANCHOR_F_RECORD_SHA256}, cells=refit["cells"],
                refit_receipt=refit["receipt"]), refit


@pytest.fixture
def tcell(tmp_path, monkeypatch, rworld):
    """run_terminal_test_cell with the managed processes recorded (the T entry and the four post-chain
    producers); `fail_at` makes the k-th process exit 1."""
    path, digest = manifest(tmp_path)
    request, refit = t_request(rworld, path, digest)
    commands, state = [], {"fail_at": None, "attempt_seen": []}

    def managed(command, **k):
        name = Path(command[1] if command[1] != "-B" else command[2]).name
        if name == "anchor_terminal_test.py":
            state["attempt_seen"].append(Path(command[command.index("--attempt") + 1]).exists())
        commands.append(name)
        return SimpleNamespace(returncode=1 if len(commands) - 1 == state["fail_at"] else 0)
    monkeypatch.setattr(M, "_run_managed_process", managed)
    monkeypatch.setattr(M, "assert_refit_outputs", lambda run_dir, dataset: {"map_at_R": 0.5, "bio_map_at_R": 0.4})
    monkeypatch.setattr(M, "verify_snapshot_input_seals", lambda snap, full: None)
    approval = {"section": 770, "scope": "stage-T-run", "line": "L"}

    def go(cell=None):
        cell = cell or request["cells"][0]
        return RT.run_terminal_test_cell(cell, namespace="ancT9", request=request, approval=approval,
                                         campaign_nonce="t" * 64, gpu_uuid="GPU-x", seal_snapshot={})
    return SimpleNamespace(go=go, commands=commands, state=state, request=request, world=rworld)


CHAIN = ["anchor_terminal_test.py", "extract_train_split.py", "eval_cell_bioproj.py", "pairwise_nmi.py",
         "seal_cell_analysis.py"]


def test_composed_a_t_cell_reserves_then_runs_the_entry_and_the_chain_in_order(tcell):
    done = tcell.go()
    assert tcell.commands == CHAIN
    assert tcell.state["attempt_seen"] == [True]            # reserved BEFORE the entry started
    assert done["map_at_R"] == 0.5
    assert (tcell.world.records / done["record"]).exists()


@pytest.mark.parametrize("k", range(len(CHAIN)))
def test_composed_every_child_failure_stops_the_chain_and_keeps_the_attempt(tcell, k):
    tcell.state["fail_at"] = k
    with pytest.raises(CellRefused):
        tcell.go()
    assert tcell.commands == CHAIN[:k + 1]
    cell = tcell.request["cells"][0]
    assert RT.attempt_path("ancT9", cell).exists()
    assert not (tcell.world.records / f"ancT9_{cell['tag']}.json").exists()


@pytest.mark.parametrize("k", [None, 0, 3])
def test_composed_an_attempted_cell_is_never_retried(tcell, k):
    tcell.state["fail_at"] = k
    try:
        tcell.go()
    except CellRefused:
        pass
    seen = len(tcell.commands)
    tcell.state["fail_at"] = None
    with pytest.raises(CellRefused, match="already reserved"):
        tcell.go()
    assert len(tcell.commands) == seen                       # no process started for the retry


def test_composed_concurrent_attempts_reserve_once(tcell):
    results = []

    def attempt():
        try:
            results.append(("ok", tcell.go()))
        except CellRefused as error:
            results.append(("refused", str(error)))
    threads = [threading.Thread(target=attempt) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(status for status, _ in results) == ["ok", "refused", "refused", "refused"]
    assert tcell.commands.count("anchor_terminal_test.py") == 1


def test_composed_a_checkpoint_changed_during_t_publishes_nothing(tcell, monkeypatch):
    cell = tcell.request["cells"][0]
    real = M._run_managed_process

    def tamper(command, **k):
        if Path(command[2]).name == "anchor_terminal_test.py" if command[1] == "-B" else False:
            Path(cell["run_dir"], "model_state_dict.pth").write_text("changed")
        return real(command, **k)
    monkeypatch.setattr(M, "_run_managed_process", tamper)
    with pytest.raises(CellRefused, match="changed during stage T"):
        tcell.go()
    assert not (tcell.world.records / f"ancT9_{cell['tag']}.json").exists()


def test_composed_an_output_already_present_refuses_before_the_attempt(tcell):
    cell = tcell.request["cells"][0]
    Path(cell["run_dir"], "extract_query.npz").write_bytes(b"x")
    with pytest.raises(CellRefused, match="never reaches the official test"):
        tcell.go()
    assert not RT.attempt_path("ancT9", cell).exists() and tcell.commands == []


# ---- the T entry: admission before any test access ----------------------------------------------
@pytest.fixture
def entry(tmp_path, monkeypatch, rworld):
    path, digest = manifest(tmp_path)
    request, refit = t_request(rworld, path, digest)
    req_digest = M._json_digest(request)
    line = LT.approval_line("stage-T-run", manifest=digest, freeze=RT.ANCHOR_F_RECORD_SHA256, request=req_digest)
    full_ledger(tmp_path, monkeypatch, {**rworld.ledger_sections, 770: [line]})
    cell = request["cells"][0]
    attempt, attempt_sha = RT.reserve_attempt("ancT9", cell=cell, request=request,
                                              approval={"section": 770, "scope": "stage-T-run", "line": line},
                                              campaign_nonce="t" * 64)
    calls = []
    import extraction_siglip2
    import terminal_official_test

    def resume(args):
        calls.append("resume")
        for k, v in dict(axis_center="anchors", selection_mode="refit", final_epoch_eval=True,
                         _phase3_campaign_binding={"cell_id": cell["cell_id"]}, save_result_path=cell["run_dir"],
                         dna_distance_mode="base", codebook_size=128).items():
            setattr(args, k, v)
    monkeypatch.setattr(extraction_siglip2, "_resume_args_flat_or_legacy", resume)
    monkeypatch.setattr(terminal_official_test, "run_official_test",
                        lambda args, **k: calls.append(("official_test", k)))
    argv = ["--config_path", cell["run_dir"], "--attempt", str(attempt), "--attempt-sha256", attempt_sha]
    return SimpleNamespace(argv=argv, calls=calls, cell=cell, attempt=attempt, request=request, line=line,
                           tmp=tmp_path, world=rworld)


def test_composed_an_admitted_entry_runs_the_terminal_function_once(entry):
    assert TE.main(entry.argv) == 0
    assert entry.calls == ["resume", ("official_test", {"distance_mode": "base", "codebook_size": 128})]


def _edit_attempt(entry, change):
    data = json.loads(entry.attempt.read_text())
    change(data)
    entry.attempt.chmod(0o644)
    entry.attempt.write_text(json.dumps(data))
    return sha(entry.attempt.read_bytes())


@pytest.mark.parametrize("break_it,reason", [
    (lambda e, mp: e.argv.__setitem__(5, "0" * 64), "reserved attempt bytes"),
    (lambda e, mp: e.argv.__setitem__(5, _edit_attempt(e, lambda d: d["request"].update(mode="smoke"))),
     "not its digest"),
    (lambda e, mp: full_ledger(e.tmp, mp, e.world.ledger_sections), "no standing stage-T approval"),
    (lambda e, mp: Path(e.cell["run_dir"], "model_state_dict.pth").write_text("other"), "not the stage-R checkpoint"),
    (lambda e, mp: Path(e.cell["run_dir"], "extract_db.npz").write_bytes(b"x"), "never reaches the official test"),
    (lambda e, mp: e.argv.__setitem__(1, str(e.world.root)), "not "),
], ids=["attempt-bytes", "request-digest", "no-approval", "checkpoint", "output-present", "run-dir"])
def test_composed_an_entry_without_complete_admission_refuses_before_any_load(entry, monkeypatch, break_it, reason):
    break_it(entry, monkeypatch)
    assert TE.main(entry.argv) == 2
    assert entry.calls == []                 # neither the configuration nor the test was touched


def test_composed_an_entry_whose_saved_configuration_is_not_this_cell_refuses_before_the_test(entry, monkeypatch):
    import extraction_siglip2
    real = extraction_siglip2._resume_args_flat_or_legacy

    def other(args):
        real(args)
        args.axis_center = "none"
    monkeypatch.setattr(extraction_siglip2, "_resume_args_flat_or_legacy", other)
    assert TE.main(entry.argv) == 2
    assert entry.calls == ["resume"]


def test_composed_the_t_approval_cannot_come_from_the_caller_environment(entry, monkeypatch):
    for name in ("GDNA_PHASE3_EXPECTED_RECIPE_DIGEST", "ANCHOR_T_AUTHORITY", "GDNA_ANCHOR_T_APPROVED"):
        monkeypatch.setenv(name, "1")
    full_ledger(entry.tmp, monkeypatch, entry.world.ledger_sections)       # the stage-T line gone
    assert TE.main(entry.argv) == 2 and entry.calls == []


# =============================================================================================
# composed: the supervisor's stage-R/T ledger; the manifest's new pins
# =============================================================================================
@pytest.mark.parametrize("stage", S.RT_STAGES)
def test_composed_stages_r_and_t_keep_their_own_ledger(stage):
    assert S.stage_ledger(stage, None) == (S.RT_OPS_ROOT, S.RT_BUDGET_DEVICE_SECONDS)
    for other in (S.L_OPS_ROOT, S.DEFAULT_OPS_ROOT):
        with pytest.raises(S.Refused):
            S.stage_ledger(stage, str(other))


@pytest.mark.parametrize("stage", ["stage-S-run", "stage-D-run", "probe", "stage-L-run"])
def test_composed_no_earlier_stage_writes_the_stage_r_t_ledger(stage):
    with pytest.raises(S.Refused):
        S.stage_ledger(stage, str(S.RT_OPS_ROOT))


def test_composed_the_r_and_t_runs_use_one_gpu_per_dataset_stream():
    assert S.command_gpus("stage-R-run", ["--gpus", "0,1,2,3"]) == 4
    assert S.command_gpus("stage-T-smoke", ["--gpus", "5"]) == 1
    with pytest.raises(S.Refused):
        S.command_gpus("stage-R-smoke", ["--gpus", "0,1"])


def test_composed_a_manifest_without_the_refit_contract_refuses(tmp_path):
    path, digest = LT.manifest(tmp_path, refit_contract=LT.DELETE)
    with pytest.raises(CellRefused, match="stage-R/T contract"):
        M.load_anchor_manifest(path, digest)


def test_structural_the_v9_closure_is_v8_plus_the_declared_additions():
    assert set(M.anchor_generation_closure()) == set(V8_FILES_SHA256) | ADDED_IN_V9


def test_structural_every_v8_member_but_the_declared_v9_sources_is_byte_equal():
    current = {rel: sha((REPO / rel).read_bytes()) for rel in V8_FILES_SHA256}
    assert sorted(rel for rel, want in V8_FILES_SHA256.items() if current[rel] != want) == sorted(CHANGED_IN_V9)


def test_structural_the_approval_scopes_bind_the_f_record():
    for scope in ("stage-R-smoke", "stage-R-run", "stage-T-smoke", "stage-T-run"):
        assert M.APPROVAL_SCOPES[scope] == ("manifest", "freeze", "request")


# =============================================================================================
# opt-in: the real Flickr25K wrapper renders a stage-R cell through the reviewed composition
# =============================================================================================
@pytest.mark.skipif(not REAL, reason="reads the real pinned wrapper and whitening path (opt-in)")
def test_real_the_flickr25k_wrapper_renders_a_stage_r_cell():
    payload = M.anchor_scientific_recipe("flickr25k", 4, namespace="ancRT", stage="refit", seed=42,
                                         topp=("0.6", "0.95"), joint="0.02", anchor_arm="anchors")
    f = payload["fields"]
    assert (f["epoch"], f["lr_schedule_horizon"], f["sinkhorn_schedule_horizon"], f["stop_after_epoch"]) == \
        (5, None, None, 4)
    assert (f["val_split_ratio"], f["selection_mode"], f["final_epoch_eval"], f["evaluation"]) == \
        (0.0, "refit", True, True)
    assert f["axis_center"] == "anchors" and f["text_whiten_npz"].endswith("trainOnly_localOnly.npz")
