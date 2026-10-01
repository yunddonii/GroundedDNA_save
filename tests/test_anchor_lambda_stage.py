"""Stage L (contract L v1, generation v8; audit 731-733): the TODO 13-15 lambda checks of the fixed
anchor model, Flickr25K first.

No real input, config, checkpoint, model, dataset or GPU is touched. Renders are a stand-in that
carries the Flickr25K wrapper's three lambda literals ahead of EXTRA_ARGS (the real wrapper's
order); leases, reservations and launches are sentinels; the v7 history is a synthetic world bound
by monkeypatched pins; a stage-L campaign world mirrors the launcher's artifacts. The tests that read
the real v7 JSON metadata or the real wrappers are opt-in (GDNA_ALLOW_REAL_ARTIFACT_TESTS=1), like
the launcher suite's real-artifact test (audit 673.3).
"""
from __future__ import annotations

import csv
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

import scripts.phase3_selection_matrix as M                 # noqa: E402
from scripts.phase3_selection_matrix import CellRefused     # noqa: E402
import scripts.anchor_confirm_decision as D                 # noqa: E402
import scripts.anchor_confirm_supervisor as S               # noqa: E402
import dna_utils.scientific_recipe as R                     # noqa: E402
from dna_utils.run_identity import PHASE3_CAMPAIGN_BINDING_NAME   # noqa: E402
import test_anchor_confirm_launcher as LT                   # noqa: E402
import test_anchor_confirm_reducer as RW                    # noqa: E402

FLICKR = "flickr25k"
INC = LT.INCUMBENT
TOPP, JOINT = INC[FLICKR]["topp"], INC[FLICKR]["joint"]
AUTH = RW.AUTH[FLICKR]
CANDIDATES = (("lambda_wasserstein", "0.30"), ("lambda_wasserstein", "0.50"), ("lambda_bu", "0"),
              ("lambda_text_hash_ntxent", "0.025"), ("lambda_text_hash_ntxent", "0.10"))
LABELS = ("incumbent",) + tuple("=".join(c) for c in CANDIDATES)
#: the Flickr25K wrapper body's three lambda literals, in its order (lines 118, 121, 124)
WRAPPER_LAMBDAS = ["--lambda_text_hash_ntxent", "0.05", "--lambda_wasserstein", "0.15", "--lambda_bu", "0.02"]
REAL = os.environ.get("GDNA_ALLOW_REAL_ARTIFACT_TESTS") == "1"
#: the five sources generation v8 changes; every other v7 closure member stays byte-equal
CHANGED_IN_V8 = {"scripts/phase3_selection_matrix.py", "scripts/anchor_confirm_decision.py",
                 "scripts/anchor_confirm_supervisor.py", "scripts/anchor_confirm_manifest.py",
                 "dna_utils/scientific_recipe.py", "tests/test_anchor_confirm_launcher.py"}

#: generation v7 closure pins, generated from authority_manifest_v7.json (c0612963...) on disk
V7_FILES_SHA256 = {
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
        "b1ed8e4fc8a5e3f0b18b2c6dc37feb1360f7599281ea761972a09374375ca8ee",
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
        "79fbe1318a35e15ed501faec09ff9fafc9bd1fe4652938c850972e7351da3bf5",
    "scripts/anchor_confirm_manifest.py":
        "2a3b8ca47c72949118f9d1688873bc3af807c82fe31e5648a5edeab5bd5058ea",
    "scripts/anchor_confirm_supervisor.py":
        "b5eecf58cbbc9d7d03463e4ea7a35ac107842aa6657a3b66e3770528cbdd1dd0",
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
        "2cb8343dba923514d9c3fbc0ffffd79c66058d582c35486fab51a518ba541f53",
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
        "a542db5c097f5c377a7cb5a496adc8744d593b3cc62d24fe881784bd07d1147f",
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
    path.write_bytes(raw)
    return sha(raw)


def lambda_render(cmd, env):
    """The launcher test's wrapper stand-in, plus the Flickr25K body's three lambda literals before
    EXTRA_ARGS, as the pinned wrapper places them."""
    argv = LT.fake_render(cmd, env)
    tail = len(LT.extra(env))
    return argv[:len(argv) - tail] + WRAPPER_LAMBDAS + argv[len(argv) - tail:]


# ---- the synthetic v7 history ----------------------------------------------------------------------
#: binary-exact scores: seed range 0.25, so the threshold is 0.25 and the boundary is exact
V7_SEEDS = [0.5, 0.25, 0.375]


class History:
    """The three accepted v7 files and the stage-S plan snapshot, at monkeypatched pins."""

    def __init__(self, root: Path, monkeypatch, *, seeds=None):
        self.root, self.mp = root, monkeypatch
        root.mkdir(parents=True, exist_ok=True)
        self.seeds = list(V7_SEEDS if seeds is None else seeds)
        self.payload = M.anchor_scientific_recipe(FLICKR, 4, namespace="ancS7", stage="select", seed=42,
                                                  topp=TOPP, joint=JOINT, anchor_arm="anchors",
                                                  input_authority=AUTH)
        self.cid = M.campaign_cell_id(FLICKR, 4, topp=TOPP, joint=JOINT, stage="select", seed=42,
                                      anchor_arm="anchors")
        self.frozen = {"artifact_kind": "anchor_confirmation_n_selection",
                       "generation": {"anchor_manifest_sha256": M.ANCHOR_V7_MANIFEST_SHA256},
                       "n_selected": {ds: {"anchors": v["N"]} for ds, v in INC.items()},
                       "scores": {FLICKR: {"anchors": {"4": self.seeds[0]}}}}
        self.snapshot = {"input_seals": {f"{FLICKR}:stage1": AUTH},
                         "plan": {"authorities": {"anchor_manifest": {"sha256": M.ANCHOR_V7_MANIFEST_SHA256}},
                                  "cell_bindings": {self.cid: {"scientific_recipe": self.payload,
                                                               "expected_scientific_recipe_sha256":
                                                                   R.digest(self.payload)}}}}
        self.write()

    def write(self):
        sel = self.root / "ancS7_selected_n.json"
        sel_sha = write(sel, self.frozen)
        self.decision = getattr(self, "decision_override", None) or {
            "artifact_kind": "anchor_confirmation_decision",
            "generation": {"anchor_manifest_sha256": M.ANCHOR_V7_MANIFEST_SHA256},
            "selection": {"path": "artifacts/anchor_confirmation/ancS7_selected_n.json", "sha256": sel_sha},
            "summary": {FLICKR: {"frozen_N": 4, "retrieval": {"per_seed": self.seeds}}}}
        dec = self.root / "ancP7_decision.json"
        dec_sha = write(dec, self.decision)
        (self.root / "ancS7_snapshot_x.json").write_text(json.dumps(self.snapshot))
        self.receipt = getattr(self, "receipt_override", None) or {
            "campaign_kind": M.ANCHOR_CAMPAIGN_KIND, "namespace": "ancS7",
            "plan_snapshot_file": "ancS7_snapshot_x.json",
            "plan_snapshot_sha256": M._json_digest(self.snapshot), "cells": {self.cid: {}}}
        rec = self.root / "ancS7_sweep_complete.json"
        rec_sha = write(rec, self.receipt)
        for name, value in (("ANCHOR_V7_RECORD_DIR", self.root), ("ANCHOR_V7_SELECTION", sel),
                            ("ANCHOR_V7_SELECTION_SHA256", sel_sha), ("ANCHOR_V7_DECISION", dec),
                            ("ANCHOR_V7_DECISION_SHA256", dec_sha), ("ANCHOR_V7_S_RECEIPT", rec),
                            ("ANCHOR_V7_S_RECEIPT_SHA256", rec_sha)):
            self.mp.setattr(M, name, value)


@pytest.fixture
def renders(monkeypatch):
    monkeypatch.setattr(M, "anchor_incumbent", lambda: INC)
    monkeypatch.setattr(M, "render_trainer_argv", lambda_render)


@pytest.fixture
def v7(tmp_path, monkeypatch, renders):
    return History(tmp_path / "v7", monkeypatch)


LAMBDA_CELLS_SELECTION = {ds: {"anchors": v["N"]} for ds, v in INC.items()}


def lambda_cells():
    return M.anchor_confirmation_cells("lambda", incumbent=INC, arm_plan=LT.ALL_ANCHORS,
                                       selection=LAMBDA_CELLS_SELECTION)


# ---- the declared tables ---------------------------------------------------------------------------
def test_the_scope_is_flickr25k_first_by_the_users_decision():
    assert M.ANCHOR_LAMBDA_SCOPE == (FLICKR,) and set(M.ANCHOR_LAMBDA_CANDIDATES) == {FLICKR}
    assert M.ANCHOR_LAMBDA_STAGE == "lambda" and M.ANCHOR_LAMBDA_CONTROL == "incumbent"


def test_the_candidates_are_the_contracts_and_never_an_incumbent_value():
    assert M.ANCHOR_LAMBDA_CANDIDATES[FLICKR] == CANDIDATES
    incumbent = M.LAMBDA_INCUMBENT[FLICKR]
    assert incumbent == {"lambda_wasserstein": "0.15", "lambda_bu": "0.02", "lambda_text_hash_ntxent": "0.05"}
    for flag, value in CANDIDATES:
        assert flag in M.LAMBDA_AXES and value in M.LAMBDA_AXES[flag]["grid"]
        assert float(value) != float(incumbent[flag])
    assert sorted({f for f, _ in CANDIDATES}) == sorted(M.LAMBDA_AXES)


def test_the_v7_pins_are_the_digests_the_audit_accepted():
    assert M.ANCHOR_V7_MANIFEST_SHA256 == "c061296309fe41203205dff65de3a1afb843c950e93c0614dfd27a8216a90128"
    assert M.ANCHOR_V7_SELECTION_SHA256 == "5cda7adb055ed126efb0a8ec198e06d1176ccdc57e7d38fe35ff74b74a4a92ff"
    assert M.ANCHOR_V7_DECISION_SHA256 == "28b10a4c850f508bacf4d9402cda8c458fe92d2bfc8b616e14a1a6ce9866cc32"
    assert M.ANCHOR_V7_S_RECEIPT_SHA256 == "5915768767e3fd12b28c3f09e1a071c26fd5cbddbbac15662c495b80c4876f6b"
    assert M.ANCHOR_V7_RECORD_DIR == Path("/data/yschoi/gdna_anchor_confirm_v1/artifacts/anchor_confirmation")


def test_the_new_approval_scopes_name_the_frozen_n_record():
    assert M.APPROVAL_SCOPES["stage-L-smoke"] == M.APPROVAL_SCOPES["stage-L-run"] == \
        ("manifest", "selection", "request")


@pytest.mark.parametrize("label", LABELS)
def test_labels_and_overrides_round_trip(label):
    overrides = M.anchor_lambda_overrides(FLICKR, label)
    assert M.anchor_lambda_label(FLICKR, overrides) == label
    assert len(overrides) == (0 if label == "incumbent" else 1)


@pytest.mark.parametrize("dataset,overrides", [
    (FLICKR, (("lambda_bu", "0.02"),)),                                # the incumbent value
    (FLICKR, (("lambda_wasserstein", "0.3"),)),                        # not the declared spelling
    (FLICKR, (("lambda_bu", "0"), ("lambda_text_hash_ntxent", "0.10"))),   # two lambdas
    (FLICKR, (("lambda_anchor", "0.1"),)),                             # not a lambda axis
    ("mscoco", (("lambda_bu", "0"),)), ("cifar10", (("lambda_wasserstein", "0.30"),)),
])
def test_an_undeclared_label_refuses(dataset, overrides):
    with pytest.raises(CellRefused, match="declared stage-L lambda candidate"):
        M.anchor_lambda_label(dataset, overrides)


@pytest.mark.parametrize("label", ["lambda_bu", "lambda_bu=0.5", "=0", "control", ""])
def test_a_malformed_label_refuses(label):
    with pytest.raises(CellRefused):
        M.anchor_lambda_overrides(FLICKR, label)


# ---- commands and cells ----------------------------------------------------------------------------
@pytest.mark.parametrize("flag,value", CANDIDATES)
def test_a_candidate_command_appends_its_one_lambda_and_tags_it(flag, value):
    base = M.build_command(FLICKR, 4, 0, topp=TOPP, joint=JOINT, anchor_arm="anchors")
    moved = M.build_command(FLICKR, 4, 0, topp=TOPP, joint=JOINT, anchor_arm="anchors",
                            overrides=((flag, value),))
    extra = LT.extra(moved[1])
    assert extra[-2:] == ["--axis_center", "anchors"] and extra.count(f"--{flag}") == 1
    i = extra.index(f"--{flag}")
    assert extra[i + 1] == value
    assert extra[:i] + extra[i + 2:] == LT.extra(base[1])
    assert M.LAMBDA_AXES[flag]["tag"] + value.replace(".", "") in moved[1]["TAG"]
    assert moved[1]["TAG"] != base[1]["TAG"] and base[1]["TAG"] not in moved[1]["TAG"]


def test_the_lambda_plan_is_the_control_then_the_five_candidates_on_flickr25k_only():
    cells = lambda_cells()
    assert [M.anchor_lambda_label(c[0], M._cell_overrides(c)) for c in cells] == list(LABELS)
    assert {(c[0], c[1], c[2], c[3], c[4], c[5], c[7]) for c in cells} == \
        {(FLICKR, 4, TOPP, JOINT, "select", 42, "anchors")}


@pytest.mark.parametrize("n", [None, 5, 4.0, "4"])
def test_the_lambda_plan_refuses_a_missing_or_off_grid_frozen_n(n):
    selection = dict(LAMBDA_CELLS_SELECTION, **{FLICKR: {"anchors": n}})
    with pytest.raises(CellRefused, match="no frozen N"):
        M.anchor_confirmation_cells("lambda", incumbent=INC, arm_plan=LT.ALL_ANCHORS, selection=selection)


@pytest.mark.parametrize("stage", ["select", "decide"])
def test_s_and_d_plans_still_carry_no_override(stage):
    cells = M.anchor_confirmation_cells(stage, incumbent=INC, arm_plan=LT.ALL_ANCHORS,
                                        selection=LAMBDA_CELLS_SELECTION)
    assert all(c[6] == () for c in cells) and all(M._cell_overrides(c) == () for c in cells)


# ---- the v7 history --------------------------------------------------------------------------------
def test_the_history_reads_the_pinned_v7_metadata(v7):
    h = M.anchor_v7_history(INC)[FLICKR]
    assert (h["N"], h["incumbent_seed42"], h["seed_scores"]) == (4, 0.5, V7_SEEDS)
    assert h["seed_range"] == 0.25 and h["threshold"] == 0.25
    assert h["fields"] == v7.payload["fields"] and h["input_authority"] == AUTH and h["cell_id"] == v7.cid


def test_a_small_seed_range_takes_the_0002_floor(tmp_path, monkeypatch, renders):
    History(tmp_path / "v7", monkeypatch, seeds=[0.5, 0.4995, 0.4999])
    h = M.anchor_v7_history(INC)[FLICKR]
    assert h["seed_range"] == 0.5 - 0.4995 and h["threshold"] == 0.002


@pytest.mark.parametrize("which", ["ANCHOR_V7_SELECTION_SHA256", "ANCHOR_V7_DECISION_SHA256",
                                   "ANCHOR_V7_S_RECEIPT_SHA256"])
def test_a_v7_file_at_another_digest_refuses(v7, monkeypatch, which):
    monkeypatch.setattr(M, which, "0" * 64)
    with pytest.raises(CellRefused, match="not the pinned bytes"):
        M.anchor_v7_history(INC)


def mutate_history(v7, which, change):
    if which == "frozen":
        change(v7.frozen)
    elif which == "decision":
        v7.write()
        v7.decision_override = json.loads(json.dumps(v7.decision))
        change(v7.decision_override)
    elif which == "snapshot":
        change(v7.snapshot)
    elif which == "receipt":
        v7.write()
        v7.receipt_override = json.loads(json.dumps(v7.receipt))
        change(v7.receipt_override)
    v7.write()


@pytest.mark.parametrize("which,change,reason", [
    ("frozen", lambda f: f["generation"].update(anchor_manifest_sha256="1" * 64), "not the v7 frozen N"),
    ("decision", lambda d: d["selection"].update(sha256="1" * 64), "not the v7 stage-D summary"),
    ("decision", lambda d: d["generation"].update(anchor_manifest_sha256="1" * 64), "not the v7 stage-D"),
    ("decision", lambda d: d["summary"][FLICKR].update(frozen_N=9), "disagree"),
    ("decision", lambda d: d["summary"][FLICKR]["retrieval"].update(per_seed=[0.5, 0.25]), "three-seed"),
    ("decision", lambda d: d["summary"][FLICKR]["retrieval"].update(per_seed=[0.5, float("nan"), 0.3]),
     "not a finite score"),
    ("decision", lambda d: d["summary"][FLICKR]["retrieval"].update(per_seed=[0.5000001, 0.25, 0.375]),
     "not the frozen N record's"),
    ("frozen", lambda f: f["n_selected"][FLICKR].update(anchors=9), "disagree"),
    ("receipt", lambda r: r.update(namespace="ancD7"), "not the v7 stage-S campaign"),
    ("receipt", lambda r: r.update(plan_snapshot_sha256="1" * 64), "plan snapshot"),
    ("receipt", lambda r: r.update(cells={}), "no completed cell"),
    ("snapshot", lambda s: s["plan"]["authorities"]["anchor_manifest"].update(sha256="1" * 64),
     "plan snapshot"),
    ("snapshot", lambda s: next(iter(s["plan"]["cell_bindings"].values())).update(
        expected_scientific_recipe_sha256="1" * 64), "not its declared digest"),
    ("snapshot", lambda s: s.update(input_seals={}), "admitted no"),
])
def test_inconsistent_v7_history_refuses(v7, which, change, reason):
    mutate_history(v7, which, change)
    with pytest.raises(CellRefused, match=reason):
        M.anchor_v7_history(INC)


# ---- the stage-L admission -------------------------------------------------------------------------
def test_the_admission_admits_the_control_and_five_one_lambda_candidates(v7):
    history = M.anchor_v7_history(INC)
    report = M.anchor_lambda_admission(lambda_cells(), namespace="ancLT", incumbent=INC, history=history)
    assert [e["label"] for e in report.values()] == list(LABELS)
    assert [e["differs_in"] for e in report.values()] == [[]] + [[f] for f, _ in CANDIDATES]
    assert len({e["incumbent_digest"] for e in report.values()}) == 1
    assert report[next(iter(report))]["digest"] == report[next(iter(report))]["incumbent_digest"]
    assert {e["overrides"].get(f, [None])[0] == list(R.STAGE_L_REVIEWED_OVERRIDES[f])
            for e in list(report.values())[1:] for f in [e["differs_in"][0]]} == {True}


def test_a_render_that_moves_a_second_field_refuses(v7, monkeypatch):
    def drifting(cmd, env):
        argv = lambda_render(cmd, env)
        return argv + (["--lambda_dna", "0.07"] if "--lambda_bu" in LT.extra(env) else [])
    monkeypatch.setattr(M, "render_trainer_argv", drifting)
    with pytest.raises(CellRefused, match=r"lambda_bu=0\|axis_center=anchors: differs from the incumbent in "
                                          r"\['lambda_bu', 'lambda_dna'\]"):
        M.anchor_lambda_admission(lambda_cells(), namespace="ancLT", incumbent=INC,
                                  history=M.anchor_v7_history(INC))


def test_an_incumbent_that_is_not_the_v7_cells_recipe_refuses(v7):
    v7.snapshot["plan"]["cell_bindings"][v7.cid]["scientific_recipe"]["fields"]["lambda_anchor"] = 0.07
    payload = v7.snapshot["plan"]["cell_bindings"][v7.cid]["scientific_recipe"]
    v7.snapshot["plan"]["cell_bindings"][v7.cid]["expected_scientific_recipe_sha256"] = R.digest(payload)
    v7.write()
    with pytest.raises(CellRefused, match=r"not the v7 stage-S seed-42 cell's sealed recipe.*lambda_anchor"):
        M.anchor_lambda_admission(lambda_cells(), namespace="ancLT", incumbent=INC,
                                  history=M.anchor_v7_history(INC))


def test_a_wrong_incumbent_lambda_breaks_the_protocol(v7, monkeypatch):
    monkeypatch.setitem(M.LAMBDA_INCUMBENT, FLICKR, dict(M.LAMBDA_INCUMBENT[FLICKR], lambda_bu="0.05"))
    with pytest.raises(CellRefused, match="breaks the protocol in lambda_bu"):
        M.anchor_lambda_admission(lambda_cells(), namespace="ancLT", incumbent=INC,
                                  history=M.anchor_v7_history(INC))


@pytest.mark.parametrize("edit,reason", [
    (lambda cells: cells[1:], "lacks"),                                         # no control
    (lambda cells: cells[:-1], "lacks"),                                        # a candidate missing
    (lambda cells: cells + cells[:1], "planned twice"),
    (lambda cells: [cells[0][:5] + (43,) + cells[0][6:]] + cells[1:], "not a stage-L coordinate"),
    (lambda cells: [(cells[0][0], 9) + cells[0][2:]] + cells[1:], "not a stage-L coordinate"),
    (lambda cells: [cells[0][:2] + (("0.3", "0.7"),) + cells[0][3:]] + cells[1:], "top-p/joint"),
    (lambda cells: [cells[0][:7] + ("none",)] + cells[1:], "not a stage-L coordinate"),
])
def test_an_off_contract_plan_refuses(v7, edit, reason):
    with pytest.raises(CellRefused, match=reason):
        M.anchor_lambda_admission(edit(lambda_cells()), namespace="ancLT", incumbent=INC,
                                  history=M.anchor_v7_history(INC))


def test_the_rule_binds_the_unrounded_v7_numbers(v7):
    rule = M.anchor_lambda_rule(M.anchor_v7_history(INC))
    assert rule["datasets"] == {FLICKR: {"N": 4, "incumbent_seed42": 0.5, "seed_scores": V7_SEEDS,
                                         "seed_range": 0.25, "threshold": 0.25}}
    assert rule["candidates"] == {FLICKR: list(LABELS[1:])} and rule["history"] == M.anchor_v7_pins()


# ---- the entry point -------------------------------------------------------------------------------
@pytest.fixture
def entry(tmp_path, monkeypatch, v7):
    """Every boundary that reserves, writes, leases, launches or verifies inputs records the call;
    the lease wrapper calls through so the arguments the sweep would receive are visible."""
    calls, sweep_kwargs = [], {}
    for name in ("_run_managed_process", "reserve_sweep_namespace", "_publish_json_exclusive"):
        monkeypatch.setattr(M, name, lambda *a, _n=name, **k: calls.append(_n) or 0)
    monkeypatch.setattr(M, "verify_campaign_input_seals",
                        lambda *a, **k: calls.append("verify_campaign_input_seals") or {})
    monkeypatch.setattr(M, "_with_campaign_gpu_leases",
                        lambda args, fn: calls.append("_with_campaign_gpu_leases") or fn())
    def run_sweep(*a, **k):
        calls.append("_run_sweep")
        sweep_kwargs.update(k)
        return 0
    monkeypatch.setattr(M, "_run_sweep", run_sweep)
    path, digest = LT.manifest(tmp_path)
    base = ["--anchor-confirm", "lambda", "--namespace", "ancLT",
            "--anchor-selection", str(M.ANCHOR_V7_SELECTION),
            "--anchor-selection-sha256", M.ANCHOR_V7_SELECTION_SHA256,
            "--anchor-manifest", path, "--anchor-manifest-sha256", digest]
    return SimpleNamespace(calls=calls, kwargs=sweep_kwargs, base=base, manifest=digest)


RUN = ["--run", "--gpus", "0"]
SMOKE = ["--smoke", "--only", "flickr25k:4:anchors:42:lambda_bu=0", "--epochs", "1", "--gpus", "0"]


def preview(monkeypatch, capsys, argv):
    assert LT.run_main(monkeypatch, *argv, "--plan") == 0
    out = capsys.readouterr().out
    request = json.loads(out[out.index("{", out.index("would need approved")):out.index("execution request sha256")])
    digest = re.search(r"execution request sha256 ([0-9a-f]{64})", out).group(1)
    assert M._json_digest(request) == digest
    return request, digest


def approve(tmp_path, monkeypatch, entry, scope, request_digest, section=731):
    LT.ledger(tmp_path, monkeypatch, {section: [LT.approval_line(
        scope, manifest=entry.manifest, selection=M.ANCHOR_V7_SELECTION_SHA256, request=request_digest)]})
    return ["--anchor-approval-section", str(section)]


def test_the_plan_previews_the_stage_l_request_and_touches_nothing(entry, monkeypatch, capsys):
    request, _ = preview(monkeypatch, capsys, [*entry.base, *RUN])
    assert entry.calls == []
    assert request["stage"] == "lambda" and request["selection"] == M.ANCHOR_V7_SELECTION_SHA256
    assert request["cells"] == request["declared_cells"] == sorted(
        [FLICKR, "anchors", 4, 42, label] for label in LABELS)
    assert request["gpu_count"] == 1 and request["lambda"] == M.anchor_lambda_rule(M.anchor_v7_history(INC))


def test_an_approved_run_verifies_inputs_then_leases_with_the_control_gate(tmp_path, entry, monkeypatch, capsys):
    _, digest = preview(monkeypatch, capsys, [*entry.base, *RUN])
    approval = approve(tmp_path, monkeypatch, entry, "stage-L-run", digest)
    assert LT.run_main(monkeypatch, *entry.base, *RUN, *approval) == 0
    assert entry.calls == ["verify_campaign_input_seals", "_with_campaign_gpu_leases", "_run_sweep"]
    gate = entry.kwargs["control_gate"]
    control, candidate = lambda_cells()[0], lambda_cells()[1]
    assert gate(control, {"selection": {"selection_value": 0.5}}) is None
    assert "continuity control scored" in gate(control, {"selection": {"selection_value": math.nextafter(0.5, 1)}})
    assert gate(candidate, {"selection": {"selection_value": 0.9}}) is None
    assert [M.anchor_lambda_label(c[0], M._cell_overrides(c)) for c in entry.kwargs["full_plan"]] == list(LABELS)


def test_an_approved_smoke_runs_one_labelled_cell_without_the_gate(tmp_path, entry, monkeypatch, capsys):
    request, digest = preview(monkeypatch, capsys, [*entry.base, *SMOKE])
    assert request["mode"] == "smoke" and request["cells"] == [[FLICKR, "anchors", 4, 42, "lambda_bu=0"]]
    assert request["epochs"] == 1 and len(request["declared_cells"]) == 6
    approval = approve(tmp_path, monkeypatch, entry, "stage-L-smoke", digest)
    assert LT.run_main(monkeypatch, *entry.base, *SMOKE, *approval) == 0
    assert entry.kwargs["control_gate"] is None


@pytest.mark.parametrize("scope", ["stage-D-run", "stage-L-smoke", "stage-S-run"])
def test_another_scope_does_not_approve_the_run(tmp_path, entry, monkeypatch, capsys, scope):
    _, digest = preview(monkeypatch, capsys, [*entry.base, *RUN])
    pins = dict(manifest=entry.manifest, selection=M.ANCHOR_V7_SELECTION_SHA256, request=digest)
    if scope == "stage-S-run":
        pins.pop("selection")
    LT.ledger(tmp_path, monkeypatch, {731: [LT.approval_line(scope, **pins)]})
    assert LT.run_main(monkeypatch, *entry.base, *RUN, "--anchor-approval-section", "731") == 2
    assert "approval lines for stage-L-run" in capsys.readouterr().err and entry.calls == []


def test_a_stale_request_is_not_approved(tmp_path, entry, monkeypatch, capsys):
    approval = approve(tmp_path, monkeypatch, entry, "stage-L-run", "0" * 64)
    assert LT.run_main(monkeypatch, *entry.base, *RUN, *approval) == 2
    assert "approves stage-L-run" in capsys.readouterr().err and entry.calls == []


@pytest.mark.parametrize("swap,reason", [
    ({"--anchor-selection-sha256": "0" * 64}, "takes the v7 frozen N record"),
    ({"--anchor-selection": "/elsewhere/ancS7_selected_n.json"}, "takes the v7 frozen N record"),
])
def test_another_frozen_n_record_refuses(entry, monkeypatch, capsys, swap, reason):
    argv = list(entry.base)
    for flag, value in swap.items():
        argv[argv.index(flag) + 1] = value
    assert LT.run_main(monkeypatch, *argv, *RUN, "--plan") == 2
    assert reason in capsys.readouterr().err and entry.calls == []


def test_the_v7_manifest_cannot_carry_stage_l(entry, monkeypatch, capsys):
    monkeypatch.setattr(M, "load_anchor_manifest", lambda path, sha256: {
        "path": str(path), "sha256": M.ANCHOR_V7_MANIFEST_SHA256, "files_sha256": {}})
    assert LT.run_main(monkeypatch, *entry.base, *RUN, "--plan") == 2
    assert "never under the v7 manifest" in capsys.readouterr().err


@pytest.mark.parametrize("only", ["flickr25k:4:anchors:42", "flickr25k:4:anchors:42:lambda_bu=0.5",
                                  "flickr25k:4:anchors:43:incumbent"])
def test_a_smoke_must_name_one_planned_labelled_cell(entry, monkeypatch, capsys, only):
    argv = [*entry.base, "--smoke", "--only", only, "--epochs", "1", "--gpus", "0", "--plan"]
    assert LT.run_main(monkeypatch, *argv) == 2
    assert entry.calls == []


# ---- the stream: overrides reach the trainer and the control gates the candidates -----------------
@pytest.fixture
def stream(tmp_path, monkeypatch):
    calls = []
    records = tmp_path / "records"
    records.mkdir()
    monkeypatch.setattr(M, "RECORD_DIR", records)
    cells = lambda_cells()
    keys = [M.campaign_cell_id(c[0], c[1], topp=c[2], joint=c[3], stage=c[4], seed=c[5],
                               overrides=M._cell_overrides(c), anchor_arm=c[7]) for c in cells]
    seals = {f"{FLICKR}:stage1": {"seal_path": "/seals/f.json", "seal_file_sha256": "5" * 64}}
    path, digest = LT.manifest(tmp_path)
    authorities = {"anchor_request": {"input_seals": {k: {"path": v["seal_path"], "sha256": v["seal_file_sha256"]}
                                                      for k, v in seals.items()}},
                   "anchor_manifest": {"path": path, "sha256": digest}}
    monkeypatch.setattr(M, "verify_campaign_input_seals", lambda *a, **k: dict(seals))
    monkeypatch.setattr(M, "plan_snapshot", lambda *a, **k: {
        "sources": {}, "inputs": {}, "input_seals": k["input_seals"], "qwen_root": "q",
        "source_authority_sha256": "s", "environment_sha256": "e",
        "plan": {"result_root": "/r", "cell_bindings": {key: {} for key in keys}}})
    for name in ("_assert_snapshot_gpu_leases", "assert_production_source_authority",
                 "assert_reservation_owner", "verify_snapshot_input_seals"):
        monkeypatch.setattr(M, name, lambda *a, **k: None)
    monkeypatch.setattr(M, "reserve_sweep_namespace", lambda *a, **k: "reservation")
    monkeypatch.setattr(M, "_publish_json_exclusive", lambda path, payload: calls.append(("publish", Path(path).name)))
    monkeypatch.setattr(M, "launch_binding_from_expected", lambda planned, digest: {})
    scores = {label: 0.6 for label in LABELS}
    scores["incumbent"] = 0.5
    def run_cell(ds, n, gpu, **k):
        label = M.anchor_lambda_label(ds, k["overrides"] or ())
        calls.append(("run_cell", label, k["overrides"]))
        tag = f"t_{label}"
        (records / f"{tag}.json").write_text("{}")
        return {"tag": tag, "run_dir": "/r/x", "geometry": {"identity_digest": "i"}, "recipe": {},
                "seed": k["seed"], "selection": {"selection_value": scores[label],
                                                 "selection_epoch_zero_based": n},
                "campaign": {}, "completion": {}}
    monkeypatch.setattr(M, "run_cell", run_cell)
    reservation = tmp_path / "reservation.json"
    reservation.write_text("{}")
    monkeypatch.setattr(M, "campaign_reservation_path", lambda namespace: reservation)
    args = SimpleNamespace(only=None, sweep=None, anchor_confirm="lambda", plan=False, run=True, smoke=False,
                           epochs=1, gpus="0", gpu=0, namespace="ancLT", admission_authority=None,
                           input_seal_specs={}, result_root="/r")
    gate = functools.partial(M.anchor_lambda_control_refusal, {FLICKR: {"incumbent_seed42": 0.5}})
    def go():
        return M._run_sweep(args, None, full_plan=cells, authorities=authorities,
                            preverified_input_seals=dict(seals), control_gate=gate)
    return SimpleNamespace(go=go, calls=calls, scores=scores)


def test_every_cell_receives_its_own_override_and_the_campaign_completes(stream):
    assert stream.go() == 0
    ran = [c for c in stream.calls if c[0] == "run_cell"]
    assert [(label, overrides) for _, label, overrides in ran] == \
        [("incumbent", ())] + [("=".join(c), (c,)) for c in CANDIDATES]
    assert ("publish", "ancLT_sweep_complete.json") in stream.calls


@pytest.mark.parametrize("control", [math.nextafter(0.5, 1), math.nextafter(0.5, 0), 0.75])
def test_a_control_off_its_reference_stops_the_stream_before_any_candidate(stream, capsys, control):
    stream.scores["incumbent"] = control
    assert stream.go() == 1
    assert [c[1] for c in stream.calls if c[0] == "run_cell"] == ["incumbent"]
    assert ("publish", "ancLT_sweep_complete.json") not in stream.calls
    assert "continuity control scored" in capsys.readouterr().err


@pytest.mark.parametrize("record", [None, {}, {"selection": {}}, {"selection": {"selection_value": 1}},
                                    {"selection": {"selection_value": "0.5"}}])
def test_the_gate_judges_without_raising(record):
    control = lambda_cells()[0]
    refusal = M.anchor_lambda_control_refusal({FLICKR: {"incumbent_seed42": 0.5}}, control, record)
    assert isinstance(refusal, str) and refusal


def test_a_gate_without_history_refuses_rather_than_raising():
    assert "could not be judged" in M.anchor_lambda_control_refusal({}, lambda_cells()[0],
                                                                    {"selection": {"selection_value": 0.5}})


# ---- the recipe module: one reviewed stage-L lambda repeat -----------------------------------------
PARSER = RW.PARSER


def flickr_argv(*tail):
    base = RW.render_argv((FLICKR, "anchors", 4, 42))
    i = base.index("--axis_center")
    return base[:i] + WRAPPER_LAMBDAS + list(tail) + base[i:]


@pytest.mark.parametrize("flag,value", CANDIDATES)
def test_one_lambda_override_after_its_wrapper_literal_is_admitted(flag, value):
    payload = R.build_payload(PARSER, flickr_argv(f"--{flag}", value), planned_arm="anchors")
    assert payload["fields"][flag] == float(value)
    admitted = R.admitted_overrides(PARSER, payload["argv"])
    assert admitted == {flag: [list(R.STAGE_L_REVIEWED_OVERRIDES[flag]), [f"--{flag}", value]]}


@pytest.mark.parametrize("tail,reason", [
    (["--lambda_bu", "0", "--lambda_wasserstein", "0.30"], "moves one lambda"),
    (["--lambda_bu", "0", "--lambda_bu", "0.5"], "given 3 times"),
])
def test_two_lambda_repeats_cannot_be_sealed(tail, reason):
    with pytest.raises(R.RecipeMismatch, match=reason):
        R.build_payload(PARSER, flickr_argv(*tail), planned_arm="anchors")


def test_a_lambda_repeat_after_another_literal_cannot_be_sealed():
    argv = flickr_argv("--lambda_wasserstein", "0.30")
    argv[argv.index("0.15")] = "0.05"                                  # MS-COCO's value, not the reviewed literal
    with pytest.raises(R.RecipeMismatch, match="not the reviewed stage-L wrapper literal"):
        R.build_payload(PARSER, argv, planned_arm="anchors")


def test_the_reviewed_table_is_narrow_and_separate():
    assert R.STAGE_L_REVIEWED_OVERRIDES == {"lambda_wasserstein": ("--lambda_wasserstein", "0.15"),
                                            "lambda_bu": ("--lambda_bu", "0.02"),
                                            "lambda_text_hash_ntxent": ("--lambda_text_hash_ntxent", "0.05")}
    assert not set(R.STAGE_L_REVIEWED_OVERRIDES) & set(R.REVIEWED_OVERRIDES)
    with pytest.raises(R.RecipeMismatch, match="outside the reviewed"):     # other repeats still refuse
        R.build_payload(PARSER, flickr_argv("--lambda_anchor", "0.05", "--lambda_anchor", "0.1"),
                        planned_arm="anchors")


@pytest.mark.skipif(not REAL, reason="opt-in: renders the real pinned Flickr25K wrapper under bash")
def test_the_real_flickr25k_wrapper_renders_each_candidate_through_the_reviewed_repeat(tmp_path, monkeypatch):
    whiten = tmp_path / "w.npz"
    whiten.write_bytes(b"x")
    for flag, value in CANDIDATES:
        cmd, env, _ = M.build_command(FLICKR, 4, 0, topp=TOPP, joint=JOINT, anchor_arm="anchors",
                                      overrides=((flag, value),))
        argv = M.render_trainer_argv(cmd, dict(env, WHITEN_NPZ=str(whiten)))
        admitted = R.admitted_overrides(PARSER, argv)
        assert admitted[flag] == [list(R.STAGE_L_REVIEWED_OVERRIDES[flag]), [f"--{flag}", value]]


# ---- the stage-L reduction -------------------------------------------------------------------------
class LWorld:
    """One stage-L campaign (receipt + snapshot + six runs) under the synthetic v7 history, its exact
    request approved in the synthetic ledger as stage-L-run, its recipes sealed by the launcher's own
    renderer (the stand-in), records shaped as the launcher writes them."""

    def __init__(self, root: Path, ledger, *, scores, approval_scope="stage-L-run", rule=None,
                 manifest_sha=RW.MANIFEST_SHA, labels=LABELS):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.nonce = sha(str(self.root).encode())
        self.manifest_sha, self.scores, self.labels = manifest_sha, dict(scores), list(labels)
        history = M.anchor_v7_history(INC)
        self.seals = {f"{FLICKR}:stage1": AUTH}
        cells = sorted([FLICKR, "anchors", 4, 42, label] for label in self.labels)
        self.request = {"schema": M.REQUEST_SCHEMA, "version": M.ANCHOR_CONFIRM_VERSION, "stage": "lambda",
                        "mode": "run", "manifest": manifest_sha, "selection": M.ANCHOR_V7_SELECTION_SHA256,
                        "namespace": "ancLT", "record_dir": str(self.root / "records"),
                        "result_root": str(self.root / "runs"), "declared_cells": cells, "cells": cells,
                        "epochs": None, "admission_authority": None, "gpu_count": 1,
                        "input_seals": {k: {"path": a["seal_path"], "sha256": a["seal_file_sha256"]}
                                        for k, a in self.seals.items()},
                        "lambda": M.anchor_lambda_rule(history) if rule is None else rule}
        pins = {"manifest": manifest_sha, "selection": M.ANCHOR_V7_SELECTION_SHA256,
                "request": M._json_digest(self.request)}
        if approval_scope == "stage-S-run":
            pins.pop("selection")
        self.approval = ledger.approve(approval_scope, **pins)
        self.bindings, self.cells, self.entries = {}, {}, []
        for label in self.labels:
            overrides = M.anchor_lambda_overrides(FLICKR, label)
            payload = M.anchor_scientific_recipe(FLICKR, 4, namespace="ancLT", stage="select", seed=42,
                                                 topp=TOPP, joint=JOINT, anchor_arm="anchors",
                                                 input_authority=AUTH, overrides=overrides)
            self.bindings[self.cid(label)] = {"anchor_arm": "anchors", "scientific_recipe": payload,
                                              "expected_scientific_recipe_sha256": R.digest(payload),
                                              **RW.input_identity(FLICKR)}
        self.write_snapshot()
        for label in self.labels:
            self.entries.append(self.make(label))
        self.write_receipt()

    @staticmethod
    def cid(label):
        return M.campaign_cell_id(FLICKR, 4, topp=TOPP, joint=JOINT, stage="select", seed=42,
                                  overrides=M.anchor_lambda_overrides(FLICKR, label), anchor_arm="anchors")

    def write_snapshot(self):
        self.snapshot = {"input_seals": self.seals,
                         "plan": {"authorities": {"anchor_manifest": {"sha256": self.manifest_sha},
                                                  "anchor_approval": self.approval,
                                                  "anchor_request": self.request},
                                  "campaign_nonce": self.nonce, "cell_bindings": self.bindings}}
        (self.root / "snapshot.json").write_text(json.dumps(self.snapshot))
        self.snapshot_sha = M._json_digest(self.snapshot)

    def make(self, label):
        cid = self.cid(label)
        payload = self.bindings[cid]["scientific_recipe"]
        run = self.root / "runs" / label.replace("=", "_")
        run.mkdir(parents=True)
        torch.save(RW.config_values(payload["argv"], historical=False), run / "config.pt")
        config_sha = sha((run / "config.pt").read_bytes())
        (run / "model_state_dict.pth").write_bytes(label.encode())
        with (run / "log.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["epoch", "eval_mAP_at_R"])
            for e in range(4):
                writer.writerow([e, ""])
            writer.writerow([4, self.scores[label]])
        recipe = self.bindings[cid]["expected_scientific_recipe_sha256"]
        evidence = {"cell_id": cid, "scientific_recipe_sha256": recipe}
        (run / PHASE3_CAMPAIGN_BINDING_NAME).write_text(json.dumps(evidence))
        (run / "model_state_dict.pth.runtime.json").write_text(json.dumps(
            {"checkpoint_epoch_zero_based": 4, "extra": {"phase3_campaign": evidence}}))
        campaign = {"cell_id": cid, "plan_snapshot_sha256": self.snapshot_sha, "campaign_nonce": self.nonce,
                    **RW.input_identity(FLICKR)}
        completion = {"final_checkpoint_epoch_zero_based": 4, "final_checkpoint": "model_state_dict.pth",
                      "final_checkpoint_sha256": sha((run / "model_state_dict.pth").read_bytes()),
                      "log_csv_sha256": sha((run / "log.csv").read_bytes()),
                      "checkpoint_runtime_sha256": sha((run / "model_state_dict.pth.runtime.json").read_bytes()),
                      "phase3_campaign_evidence_sha256": sha((run / PHASE3_CAMPAIGN_BINDING_NAME).read_bytes())}
        record = {"dataset": FLICKR, "N": 4, "seed": 42, "stage": "select", "selection_mode": "select",
                  "smoke": False, "is_candidate_cell": True, "val_split_ratio": 0.1, "val_split_seed": 42,
                  "run_dir": str(run), "campaign": campaign, "completion": completion, "input_authority": AUTH,
                  "selection": {"selection_epoch_zero_based": 4, "selection_metric": "eval_mAP_at_R",
                                "distance_mode": "base", "selection_value": self.scores[label]},
                  "anchor_confirmation": {"version": M.ANCHOR_CONFIRM_VERSION, "arm": "anchors",
                                          "scientific_recipe_sha256": recipe,
                                          "campaign_evidence_sha256": completion["phase3_campaign_evidence_sha256"],
                                          "config_pt_sha256": config_sha}}
        path = run / "record.json"
        path.write_text(json.dumps(record))
        self.cells[cid] = {"record": path.name, "record_sha256": sha(path.read_bytes()), "campaign": campaign,
                           "completion": {k: completion[k] for k in D.RECEIPT_COMPLETION_KEYS}}
        return {"dataset": FLICKR, "arm": "anchors", "N": 4, "seed": 42, "lambda": label, "record": str(path),
                "record_sha256": sha(path.read_bytes()), "source": "receipt"}

    def write_receipt(self):
        receipt = {"campaign_kind": M.ANCHOR_CAMPAIGN_KIND,
                   "anchor_confirmation": {"version": M.ANCHOR_CONFIRM_VERSION},
                   "campaign_nonce": self.nonce, "namespace": "ancLT", "plan_snapshot_file": "snapshot.json",
                   "plan_snapshot_sha256": self.snapshot_sha, "cells": self.cells}
        path = self.root / "receipt.json"
        path.write_text(json.dumps(receipt))
        for e in self.entries:
            e.update(receipt=str(path), receipt_sha256=sha(path.read_bytes()))

    def sources(self, entries=None):
        path = self.root / "sources.json"
        path.write_text(json.dumps({"version": M.ANCHOR_CONFIRM_VERSION,
                                    "coordinates": self.entries if entries is None else entries}))
        return str(path), sha(path.read_bytes())


@pytest.fixture
def reducer(tmp_path, monkeypatch, v7):
    book = RW.Ledger(tmp_path / "ledger.md")
    monkeypatch.setattr(M, "AUDIT_LEDGER", book.path)
    monkeypatch.setattr(M, "load_anchor_manifest", RW.fake_manifest)
    monkeypatch.setattr(M, "recheck_generation",
                        lambda manifest, where: RW.fake_manifest(manifest["path"], manifest["sha256"]))
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("a binary was deserialised"))
    counter = iter(range(1000))
    def world(scores=None, **kw):
        base = {label: 0.6 for label in LABELS}
        base["incumbent"] = 0.5
        return LWorld(tmp_path / f"L{next(counter)}", book, scores={**base, **(scores or {})}, **kw)
    return SimpleNamespace(world=world, ledger=book, tmp=tmp_path)


def run_lambda(env, world, *, sources=None, manifest=RW.MANIFEST_SHA, extra=()):
    path, digest = sources or world.sources()
    out = env.tmp / f"decision_{len(list(env.tmp.glob('decision_*')))}.json"
    argv = ["lambda", "--sources", path, "--sources-sha256", digest, "--out", str(out),
            "--manifest", RW.MANIFESTS.get(manifest, {"path": "/m"})["path"], "--manifest-sha256", manifest,
            *extra]
    rc = D.main(argv)
    return rc, (json.loads(out.read_text()) if rc == 0 else None)


def test_no_candidate_above_the_threshold_keeps_the_incumbent(reducer):
    rc, result = run_lambda(reducer, reducer.world())
    assert rc == 0 and result["artifact_kind"] == D.LAMBDA_KIND and result["scope"] == [FLICKR]
    f = result["datasets"][FLICKR]
    assert f["changed_axes"] == [] and f["winners"] == {a: None for a in M.LAMBDA_AXES}
    assert f["recipe"] == {"lambda_wasserstein": 0.15, "lambda_bu": 0.02, "lambda_text_hash_ntxent": 0.05}
    assert f["control"]["score"] == 0.5 and f["threshold"] == 0.25 and f["incumbent_seed42"] == 0.5
    assert [c["qualifies"] for c in f["candidates"]] == [False] * 5
    assert "keep flickr25k's approved lambdas" in f["obligations"][0]
    assert result["history"] == M.anchor_v7_pins() and result["fixed_architecture"] == D.FIXED_ARCHITECTURE


def test_the_threshold_is_strict_on_unrounded_values(reducer):
    at, above = 0.75, math.nextafter(0.75, 1)               # delta 0.25 == T, delta 0.25 + 2**-53 > T
    rc, result = run_lambda(reducer, reducer.world({"lambda_bu=0": at, "lambda_wasserstein=0.30": above}))
    assert rc == 0
    by = {(c["axis"], c["value"]): c for c in result["datasets"][FLICKR]["candidates"]}
    assert by[("lambda_bu", 0.0)]["delta"] == 0.25 and by[("lambda_bu", 0.0)]["qualifies"] is False
    assert by[("lambda_wasserstein", 0.3)]["qualifies"] is True
    f = result["datasets"][FLICKR]
    assert f["winners"] == {"lambda_wasserstein": 0.3, "lambda_bu": None, "lambda_text_hash_ntxent": None}
    assert f["changed_axes"] == ["lambda_wasserstein"] and "reselect flickr25k's N" in f["obligations"][0]
    assert "never copied" in f["obligations"][2]


def test_the_highest_qualifier_wins_and_ties_go_to_the_value_nearer_the_incumbent(reducer):
    rc, result = run_lambda(reducer, reducer.world({
        "lambda_wasserstein=0.30": 0.8, "lambda_wasserstein=0.50": 0.8,                 # tie: 0.30 nearer
        "lambda_text_hash_ntxent=0.025": 0.8, "lambda_text_hash_ntxent=0.10": 0.9}))    # higher wins
    assert rc == 0
    assert result["datasets"][FLICKR]["winners"] == {"lambda_wasserstein": 0.3, "lambda_bu": None,
                                                     "lambda_text_hash_ntxent": 0.1}


def test_winners_on_two_axes_combine_and_state_the_reselection(reducer):
    rc, result = run_lambda(reducer, reducer.world({"lambda_bu=0": 0.8, "lambda_text_hash_ntxent=0.025": 0.85}))
    assert rc == 0
    f = result["datasets"][FLICKR]
    assert f["recipe"] == {"lambda_wasserstein": 0.15, "lambda_bu": 0.0, "lambda_text_hash_ntxent": 0.025}
    assert f["changed_axes"] == ["lambda_bu", "lambda_text_hash_ntxent"]


@pytest.mark.parametrize("value", [0.5, 0.25])
def test_the_tie_order_prefers_nearness_then_the_smaller_value(value):
    assert D.axis_winner([(0.3, 0.8), (0.5, 0.8)], 0.15) == 0.3
    assert D.axis_winner([(0.025, 0.8), (0.075, 0.8)], 0.05) == 0.025       # equally near: smaller
    assert D.axis_winner([(0.5, 0.9), (0.3, 0.8)], 0.15) == 0.5
    assert D.axis_winner([], value) is None


@pytest.mark.parametrize("control", [math.nextafter(0.5, 1), math.nextafter(0.5, 0), 0.25])
def test_a_control_off_the_v7_score_refuses_the_decision(reducer, control):
    rc, _ = run_lambda(reducer, reducer.world({"incumbent": control}))
    assert rc == 1


def test_a_control_whose_recipe_is_not_the_v7_cells_refuses(reducer, v7, monkeypatch, capsys):
    """The v7 cell's recipe is changed first and the campaign is issued under the resulting pins
    (its request binds them), so only the control-to-v7 field comparison can refuse."""
    v7.snapshot["plan"]["cell_bindings"][v7.cid]["scientific_recipe"]["fields"]["lambda_dna"] = 0.07
    payload = v7.snapshot["plan"]["cell_bindings"][v7.cid]["scientific_recipe"]
    v7.snapshot["plan"]["cell_bindings"][v7.cid]["expected_scientific_recipe_sha256"] = R.digest(payload)
    v7.write()
    world = reducer.world()
    rc, _ = run_lambda(reducer, world)
    assert rc == 1 and "continuity control's sealed recipe" in capsys.readouterr().err


@pytest.mark.parametrize("edit,reason", [
    (lambda es: es[1:], "membership"),
    (lambda es: es[:-1], "membership"),
    (lambda es: es + [dict(es[1])], "duplicate"),
    (lambda es: [dict(es[0], **{"lambda": "lambda_bu=0.5"})] + es[1:], "membership"),
    (lambda es: [dict(es[0], seed=43)] + es[1:], "membership"),
    (lambda es: [dict(es[0], N=9)] + es[1:], "membership"),
    (lambda es: [dict(es[1], record=es[0]["record"], record_sha256=es[0]["record_sha256"])] * 1
     + [es[0]] + es[2:], "reused"),
])
def test_off_contract_evidence_refuses(reducer, capsys, edit, reason):
    world = reducer.world()
    rc, _ = run_lambda(reducer, world, sources=world.sources(edit([dict(e) for e in world.entries])))
    assert rc == 1 and reason in capsys.readouterr().err


@pytest.mark.parametrize("score,reason", [(float("nan"), "not finite"), (float("inf"), "not finite"),
                                          (-0.1, "outside [0, 1]"), (1.5, "outside [0, 1]")])
def test_a_non_finite_or_out_of_range_score_refuses(reducer, capsys, score, reason):
    """The world is issued consistently around the bad score, so only the score check can refuse."""
    rc, _ = run_lambda(reducer, reducer.world({"lambda_bu=0": score}))
    assert rc == 1 and reason in capsys.readouterr().err


def test_a_candidate_sealed_with_a_second_change_refuses(reducer, capsys):
    world = reducer.world()
    cid = world.cid("lambda_bu=0")
    payload = world.bindings[cid]["scientific_recipe"]
    payload["fields"]["lambda_dna"] = 0.07
    # re-issue the whole chain consistently so only the one-lambda check can refuse
    world.bindings[cid]["expected_scientific_recipe_sha256"] = R.digest(payload)
    world.entries, world.cells = [], {}
    import shutil
    shutil.rmtree(world.root / "runs")
    world.write_snapshot()
    for label in world.labels:
        world.entries.append(world.make(label))
    world.write_receipt()
    rc, _ = run_lambda(reducer, world)
    assert rc == 1 and "differs from the control in ['lambda_bu', 'lambda_dna']" in capsys.readouterr().err


def test_a_campaign_approved_under_another_scope_refuses(reducer, capsys):
    rc, _ = run_lambda(reducer, reducer.world(approval_scope="stage-D-run"))
    assert rc == 1 and "approval scope 'stage-D-run', not stage-L-run" in capsys.readouterr().err


def test_a_request_without_the_preregistered_rule_refuses(reducer, capsys):
    rule = M.anchor_lambda_rule(M.anchor_v7_history(INC))
    rule["datasets"][FLICKR]["threshold"] = 0.2
    rc, _ = run_lambda(reducer, reducer.world(rule=rule))
    assert rc == 1 and "preregistered stage-L rule" in capsys.readouterr().err


def test_another_generation_refuses(reducer, capsys):
    rc, _ = run_lambda(reducer, reducer.world(manifest_sha=RW.OTHER_SHA))
    assert rc == 1


def test_the_v7_manifest_cannot_reduce_stage_l(reducer, monkeypatch, capsys):
    world = reducer.world()
    monkeypatch.setattr(M, "ANCHOR_V7_MANIFEST_SHA256", RW.MANIFEST_SHA)
    rc, _ = run_lambda(reducer, world)
    assert rc == 1 and "never the v7 manifest" in capsys.readouterr().err


def test_a_selection_argument_refuses(reducer, capsys):
    rc, _ = run_lambda(reducer, reducer.world(), extra=["--selection", "x", "--selection-sha256", "0" * 64])
    assert rc == 1 and "give no --selection" in capsys.readouterr().err


def test_a_v7_summary_at_another_digest_refuses(reducer, monkeypatch, capsys):
    world = reducer.world()
    monkeypatch.setattr(M, "ANCHOR_V7_DECISION_SHA256", "0" * 64)
    rc, _ = run_lambda(reducer, world)
    assert rc == 1 and "v7 history" in capsys.readouterr().err


def test_the_reduction_reads_the_v7_files_and_deserialises_nothing(reducer):
    rc, result = run_lambda(reducer, reducer.world())
    assert rc == 0
    read = result["consumed_sha256"]
    assert {str(M.ANCHOR_V7_SELECTION), str(M.ANCHOR_V7_DECISION), str(M.ANCHOR_V7_S_RECEIPT)} <= set(read)
    assert sum(1 for p in read if p.endswith("config.pt")) == 6
    assert not any(p.endswith(".pth") for p in read)


# ---- the S/D reducer is not opened to stage-L records ----------------------------------------------
def test_an_s_d_admission_refuses_an_override():
    with pytest.raises(D.NotReducible, match="carries no lambda override"):
        D.admit_metadata(D.Consumed(), (FLICKR, "anchors", 4, 42), {}, incumbent=INC,
                         manifest_sha256=RW.MANIFEST_SHA, overrides=(("lambda_bu", "0"),))


def test_a_stage_l_record_is_not_stage_s_evidence(reducer, capsys):
    world = reducer.world()
    control = dict(world.entries[0])
    control.pop("lambda")
    with pytest.raises(D.NotReducible):
        D.admit_metadata(D.Consumed(), (FLICKR, "anchors", 4, 42), control, incumbent=INC,
                         manifest_sha256=RW.MANIFEST_SHA)


def test_an_s_d_record_is_not_stage_l_evidence(reducer, tmp_path):
    s_world = RW.World(tmp_path / "S", [(FLICKR, "anchors", 4, 42)], ledger=reducer.ledger)
    with pytest.raises(D.NotReducible):
        D.admit_metadata(D.Consumed(), (FLICKR, "anchors", 4, 42), s_world.entries[0], incumbent=INC,
                         manifest_sha256=RW.MANIFEST_SHA, selection_sha256=M.ANCHOR_V7_SELECTION_SHA256,
                         role="lambda", overrides=(), lambda_rule=M.anchor_lambda_rule(M.anchor_v7_history(INC)))


# ---- the supervisor --------------------------------------------------------------------------------
def test_stage_l_keeps_its_own_ledger_and_budget():
    assert S.STAGES["stage-L-run"] == S.STAGES["stage-L-smoke"] == (2 * 3600.0, 1)
    assert S.L_BUDGET_DEVICE_SECONDS == 3600.0 and S.L_OPS_ROOT == Path("/home/yschoi/gdna_anchorL_ops")
    assert S.stage_ledger("stage-L-run", None) == (S.L_OPS_ROOT, 3600.0)
    assert S.stage_ledger("stage-L-smoke", str(S.L_OPS_ROOT)) == (S.L_OPS_ROOT, 3600.0)
    assert S.stage_ledger("stage-D-run", None) == (S.DEFAULT_OPS_ROOT, S.BUDGET_DEVICE_SECONDS)
    assert S.stage_ledger("probe", "/tmp/x") == (Path("/tmp/x"), S.BUDGET_DEVICE_SECONDS)


@pytest.mark.parametrize("stage,root", [("stage-L-run", str(S.DEFAULT_OPS_ROOT)), ("stage-L-smoke", "/tmp/x"),
                                        ("stage-S-run", str(S.L_OPS_ROOT)), ("probe", str(S.L_OPS_ROOT))])
def test_a_crossed_ledger_root_refuses(stage, root):
    with pytest.raises(S.Refused, match="ledger"):
        S.stage_ledger(stage, root)


def test_an_l_stage_supervises_one_gpu():
    assert S.command_gpus("stage-L-run", ["x", "--gpus", "3"]) == 1
    with pytest.raises(S.Refused, match="1 to 1"):
        S.command_gpus("stage-L-run", ["x", "--gpus", "0,1"])


# ---- the plan snapshot's seal and the request ------------------------------------------------------
@pytest.mark.parametrize("label", LABELS)
def test_the_snapshot_binding_seals_each_cells_own_override(v7, label):
    """expected_cell_binding (the plan snapshot's per-cell seal the trainer is held to) carries the
    cell's one lambda into its sealed recipe, identity and cell id. Without it a candidate whose
    stream also dropped the override would run the incumbent recipe under a candidate label."""
    overrides = M.anchor_lambda_overrides(FLICKR, label)
    binding = M.expected_cell_binding(FLICKR, 4, namespace="ancLT", campaign_nonce="n" * 64, topp=TOPP,
                                      joint=JOINT, seed=42, overrides=overrides, anchor_arm="anchors",
                                      input_authority=AUTH)
    fields = binding["scientific_recipe"]["fields"]
    want = M.anchor_lambda_protocol_fields(FLICKR, 4, overrides=overrides, incumbent=INC)
    assert {k: fields[k] for k in M.LAMBDA_AXES} == {k: want[k] for k in M.LAMBDA_AXES}
    assert binding["cell_id"] == LWorld.cid(label) and binding["lambda_overrides"] == dict(overrides)
    assert R.digest(binding["scientific_recipe"]) == binding["expected_scientific_recipe_sha256"]


def test_only_a_stage_l_request_carries_the_preregistered_rule():
    args = SimpleNamespace(anchor_confirm="select", smoke=False, only=None, gpus="0,1,2,3", gpu=0,
                           namespace="ancT", result_root="/r", epochs=None, input_seal_specs={},
                           admission_authority=None)
    cells = M.anchor_confirmation_cells("select", incumbent=INC, arm_plan=LT.ALL_ANCHORS)
    with pytest.raises(CellRefused, match="preregistered rule"):
        M.anchor_execution_request(args, cells, manifest_sha256="a" * 64, lambda_rule={"x": 1})
    args.anchor_confirm, args.gpus = "lambda", "0"
    with pytest.raises(CellRefused, match="preregistered rule"):
        M.anchor_execution_request(args, lambda_cells(), manifest_sha256="a" * 64,
                                   selection_sha256="b" * 64, lambda_rule=None)


@pytest.mark.parametrize("rel", ["scripts/phase3_selection_matrix.py", "scripts/anchor_confirm_decision.py",
                                 "scripts/anchor_confirm_supervisor.py", "scripts/anchor_confirm_manifest.py",
                                 "dna_utils/scientific_recipe.py"])
def test_each_changed_module_defines_each_top_level_name_once(rel):
    """A later definition silently replaces an earlier one; the first v8 draft shadowed the refit
    aggregation's `_finite_proportion` this way (caught by the full suite, then renamed)."""
    import ast
    names = [n.name for n in ast.parse((REPO / rel).read_text()).body
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    assert sorted({n for n in names if names.count(n) > 1}) == []


# ---- the closure: everything but the declared v8 sources is byte-equal to generation v7 -----------
def test_the_v8_closure_extends_v7_with_the_lambda_contract_and_this_file():
    closure = set(M.anchor_generation_closure())
    assert closure == set(V7_FILES_SHA256) | {M.ANCHOR_LAMBDA_CONTRACT_PATH, "tests/test_anchor_lambda_stage.py"}


def test_every_v7_member_but_the_declared_v8_sources_is_byte_equal():
    current = {rel: sha((REPO / rel).read_bytes()) for rel in V7_FILES_SHA256}
    assert sorted(rel for rel, want in V7_FILES_SHA256.items() if current[rel] != want) == sorted(CHANGED_IN_V8)


@pytest.mark.skipif(not REAL, reason="opt-in: reads the real v7 manifest and history (JSON only)")
def test_the_literal_v7_pins_and_history_are_the_real_files():
    raw = M.ANCHOR_V7_RECORD_DIR.joinpath("authority_manifest_v7.json").read_bytes()
    assert sha(raw) == M.ANCHOR_V7_MANIFEST_SHA256
    assert json.loads(raw)["new_generation"]["files_sha256"] == V7_FILES_SHA256
    h = M.anchor_v7_history(M.anchor_incumbent())[FLICKR]
    assert (h["N"], h["incumbent_seed42"], h["seed_range"], h["threshold"]) == \
        (4, 0.7641936888306327, 0.02819158958924184, 0.02819158958924184)
