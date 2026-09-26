"""Anchor confirmation v1: the reducer and the code-to-axis probe, on synthetic evidence.

Audit section 659.4 asks that the reducer validate exact planned membership and effective
recipes, the train-only split, terminal-epoch/checkpoint linkage and metric range/finiteness,
parse scores from verified bytes, and refuse partial/smoke/refit/test records. Each refusal
below perturbs ONE thing of a world the positive control reduces successfully. No real
checkpoint, cache or model is loaded; the rendered recipe is a stand-in built with the real
trainer parser.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import sys

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import scripts.phase3_selection_matrix as M                 # noqa: E402
import scripts.anchor_confirm_decision as D                 # noqa: E402
import scripts.anchor_confirm_code_axis as P                # noqa: E402
import dna_utils.scientific_recipe as R                     # noqa: E402
from config import Config                                   # noqa: E402

INCUMBENT = {"flickr25k": {"N": 4, "topp": ("0.6", "0.95"), "joint": "0.02"},
             "nuswide": {"N": 4, "topp": ("0.4", "0.8"), "joint": "0.05"},
             "mscoco": {"N": 39, "topp": ("0.6", "0.95"), "joint": "0.03"}}
NAMES = {"flickr25k": "Flickr25k", "nuswide": "NUSWIDE", "mscoco": "MSCOCO"}
PARSER = Config.build_parser()


def argv_for(coordinate):
    ds, arm, n, seed = coordinate
    return ["--tag", f"{ds}_{arm}_{n}_{seed}", "--dataset", NAMES[ds], "--setting", "1",
            "--axis_center", arm, "--stop_after_epoch", str(n), "--random_seed", str(seed)]


def score_of(coordinate):
    ds, arm, n, seed = coordinate
    base = {4: 0.70, 9: 0.72, 19: 0.71, 39: 0.69}[n]         # N=9 wins for every arm
    return round(base + (0.01 if arm == "anchors" else 0.0) + 0.001 * (seed - 42), 6)


class World:
    """Run directories, records and a sources file for a list of coordinates."""

    def __init__(self, root: Path, coordinates, *, probes=False):
        self.root, self.entries = root, []
        for c in coordinates:
            self.entries.append(self.make(c, probes=probes))

    def make(self, c, *, probes):
        ds, arm, n, seed = c
        run = self.root / "runs" / "_".join(map(str, c))
        run.mkdir(parents=True)
        values = vars(PARSER.parse_args(argv_for(c)))
        values["setting"] = "setting" + str(values["setting"])
        values.pop("log_dir", None)
        historical = arm == "none"
        if historical:
            del values["axis_center"]                          # the approved generation had no axis
        torch.save(values, run / "config.pt")
        config_sha = hashlib.sha256((run / "config.pt").read_bytes()).hexdigest()
        with (run / "log.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["epoch", "eval_mAP_at_R"])
            for e in range(n):
                writer.writerow([e, ""])
            writer.writerow([n, score_of(c)])
        record = {
            "dataset": ds, "N": n, "seed": seed, "stage": "select", "selection_mode": "select",
            "smoke": False, "is_candidate_cell": True, "val_split_ratio": 0.1,
            "run_dir": str(run),
            "completion": {"final_checkpoint_epoch_zero_based": n,
                           "final_checkpoint": "model_state_dict.pth",
                           "final_checkpoint_sha256": hashlib.sha256(str(c).encode()).hexdigest(),
                           "log_csv_sha256": hashlib.sha256((run / "log.csv").read_bytes()).hexdigest()},
            "selection": {"selection_epoch_zero_based": n, "selection_metric": "eval_mAP_at_R",
                          "distance_mode": "base", "selection_value": score_of(c)},
            **({} if historical else {"anchor_confirmation": {"arm": arm,
                                                              "config_pt_sha256": config_sha}})}
        path = run / "record.json"
        path.write_text(json.dumps(record))
        entry = {"dataset": ds, "arm": arm, "N": n, "seed": seed, "record": str(path),
                 "record_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        if probes:
            probe = {"artifact_kind": D.PROBE_KIND,
                     "checkpoint_sha256": record["completion"]["final_checkpoint_sha256"],
                     "config_pt_sha256": config_sha, "n_images": 512,
                     "code_picks_own_axis": 0.40 if arm == "anchors" else 0.33}
            probe_path = run / "probe.json"
            probe_path.write_text(json.dumps(probe))
            entry.update(probe=str(probe_path),
                         probe_sha256=hashlib.sha256(probe_path.read_bytes()).hexdigest())
        return entry

    def sources(self, entries=None):
        path = self.root / "sources.json"
        path.write_text(json.dumps({"version": M.ANCHOR_CONFIRM_VERSION,
                                    "coordinates": self.entries if entries is None else entries}))
        return str(path), hashlib.sha256(path.read_bytes()).hexdigest()

    def rewrite_record(self, index, **changes):
        entry = self.entries[index]
        record = json.loads(Path(entry["record"]).read_text())
        for key, value in changes.items():
            target = record
            *parents, leaf = key.split(".")
            for p in parents:
                target = target[p]
            target[leaf] = value
        Path(entry["record"]).write_text(json.dumps(record))
        entry["record_sha256"] = hashlib.sha256(Path(entry["record"]).read_bytes()).hexdigest()


@pytest.fixture(autouse=True)
def stand_ins(monkeypatch):
    monkeypatch.setattr(M, "anchor_incumbent", lambda: INCUMBENT)
    monkeypatch.setattr(D, "rendered_fields",
                        lambda c, inc: R.build_payload(PARSER, argv_for(c))["fields"])


def select_args(world, out, arms="none,anchors", sources=None):
    path, sha = sources or world.sources()
    return D.main(["select", "--sources", path, "--sources-sha256", sha, "--arms", arms,
                   "--out", str(out)])


@pytest.fixture
def select_world(tmp_path):
    return World(tmp_path, D.expected_coordinates("select", arms=M.ANCHOR_ARMS))


# ---- select: the positive control ---------------------------------------------------------------
def test_select_reduces_to_the_argmax_n_and_writes_once(select_world, tmp_path):
    out = tmp_path / "n.json"
    assert select_args(select_world, out) == 0
    result = json.loads(out.read_text())
    assert result["n_selected"] == {ds: {"none": 9, "anchors": 9} for ds in M.ANCHOR_DATASETS}
    assert result["artifact_kind"] == D.N_SELECTION_KIND
    assert len(result["consumed_sha256"]) == 1 + 24 * 3          # sources + record, log, config
    assert select_args(select_world, out) == 1                    # O_EXCL: never overwritten


def test_ties_go_to_the_smallest_n():
    assert D.select_n({4: 0.7, 9: 0.72, 19: 0.72, 39: 0.1}) == 9


# ---- select: membership --------------------------------------------------------------------------
@pytest.mark.parametrize("case", ["missing", "extra", "duplicate", "reused_record"])
def test_membership_must_be_exactly_the_contract(select_world, tmp_path, case):
    entries = [dict(e) for e in select_world.entries]
    if case == "missing":
        entries.pop()
    elif case == "extra":
        extra = dict(entries[0], N=5)
        entries.append(extra)
    elif case == "duplicate":
        entries.append(dict(entries[0]))
    else:
        entries[1]["record"], entries[1]["record_sha256"] = entries[0]["record"], entries[0]["record_sha256"]
    assert select_args(select_world, tmp_path / "o.json", sources=select_world.sources(entries)) == 1
    assert not (tmp_path / "o.json").exists()


# ---- select: one record perturbed ---------------------------------------------------------------
PERTURB = {   # case: (one change, the refusal it must produce)
    "smoke": ({"smoke": True}, "smoke or non-candidate"),
    "not_candidate": ({"is_candidate_cell": False}, "smoke or non-candidate"),
    "refit_stage": ({"stage": "refit"}, "not a stage-1 selection record"),
    "no_validation_split": ({"val_split_ratio": 0.0}, "90/10 train-only split"),
    "checkpoint_epoch": ({"completion.final_checkpoint_epoch_zero_based": 3}, "epoch is not N"),
    "selection_epoch": ({"selection.selection_epoch_zero_based": 3}, "epoch is not N"),
    "other_metric": ({"selection.selection_metric": "eval_mAP"}, "raw base-Hamming"),
    "other_distance": ({"selection.distance_mode": "codeword"}, "raw base-Hamming"),
    "selection_value": ({"selection.selection_value": 0.99}, "not the pinned log's"),
    "wrong_coordinate": ({"seed": 43}, "not the record of"),
}


@pytest.mark.parametrize("case", PERTURB)
def test_a_record_that_is_not_train_only_terminal_evidence_refuses(select_world, tmp_path, capsys, case):
    change, reason = PERTURB[case]
    select_world.rewrite_record(5, **change)
    assert select_args(select_world, tmp_path / "o.json") == 1
    assert reason in capsys.readouterr().err
    assert not (tmp_path / "o.json").exists()


@pytest.mark.parametrize("value", ["nan", "inf", "1.5", "-0.1", ""])
def test_a_terminal_score_that_is_not_a_proportion_refuses(select_world, tmp_path, value):
    entry = select_world.entries[3]
    run = Path(entry["record"]).parent
    lines = (run / "log.csv").read_text().splitlines()
    lines[-1] = lines[-1].rsplit(",", 1)[0] + "," + value
    (run / "log.csv").write_text("\n".join(lines) + "\n")
    select_world.rewrite_record(3, **{"completion.log_csv_sha256":
                                      hashlib.sha256((run / "log.csv").read_bytes()).hexdigest()})
    assert select_args(select_world, tmp_path / "o.json") == 1


def test_a_log_replaced_after_its_pin_refuses(select_world, tmp_path):
    run = Path(select_world.entries[2]["record"]).parent
    (run / "log.csv").write_text((run / "log.csv").read_text().replace("0.7", "0.9"))
    assert select_args(select_world, tmp_path / "o.json") == 1


def test_a_drifted_control_configuration_names_the_field(select_world, tmp_path, capsys):
    index = next(i for i, e in enumerate(select_world.entries) if e["arm"] == "none")
    run = Path(select_world.entries[index]["record"]).parent
    values = torch.load(run / "config.pt", weights_only=False)
    values["lambda_bu"] = 0.5
    torch.save(values, run / "config.pt")
    assert select_args(select_world, tmp_path / "o.json") == 1
    err = capsys.readouterr().err
    assert "is not the recipe of" in err and "lambda_bu" in err


def test_an_anchor_config_rewritten_after_training_is_caught_by_its_evidence(select_world, tmp_path, capsys):
    index = next(i for i, e in enumerate(select_world.entries) if e["arm"] == "anchors")
    run = Path(select_world.entries[index]["record"]).parent
    values = torch.load(run / "config.pt", weights_only=False)
    values["lambda_bu"] = 0.5
    torch.save(values, run / "config.pt")
    assert select_args(select_world, tmp_path / "o.json") == 1
    assert "anchor evidence disagrees" in capsys.readouterr().err


def test_an_anchor_record_whose_evidence_names_another_config_refuses(select_world, tmp_path):
    index = next(i for i, e in enumerate(select_world.entries) if e["arm"] == "anchors")
    select_world.rewrite_record(index, **{"anchor_confirmation.config_pt_sha256": "0" * 64})
    assert select_args(select_world, tmp_path / "o.json") == 1


def test_an_input_changed_after_it_was_read_refuses_the_write(select_world, tmp_path, monkeypatch):
    original = D.Consumed.reverify
    def tamper(self):
        Path(select_world.entries[0]["record"]).write_text("{}")
        return original(self)
    monkeypatch.setattr(D.Consumed, "reverify", tamper)
    assert select_args(select_world, tmp_path / "o.json") == 1
    assert not (tmp_path / "o.json").exists()


def test_sources_file_must_be_the_pinned_bytes(select_world, tmp_path):
    path, _ = select_world.sources()
    assert select_args(select_world, tmp_path / "o.json", sources=(path, "0" * 64)) == 1


# ---- decide ---------------------------------------------------------------------------------------
FROZEN = {ds: {"none": 4, "anchors": 9} for ds in M.ANCHOR_DATASETS}


def decide_world(tmp_path, probes=True):
    world = World(tmp_path, D.expected_coordinates("decide", arms=M.ANCHOR_ARMS, frozen=FROZEN),
                  probes=probes)
    frozen = tmp_path / "frozen.json"
    frozen.write_text(json.dumps({"artifact_kind": D.N_SELECTION_KIND,
                                  "version": M.ANCHOR_CONFIRM_VERSION, "n_selected": FROZEN}))
    return world, str(frozen), hashlib.sha256(frozen.read_bytes()).hexdigest()


def decide(world, frozen, frozen_sha, out, sources=None):
    path, sha = sources or world.sources()
    return D.main(["decide", "--sources", path, "--sources-sha256", sha, "--selection", frozen,
                   "--selection-sha256", frozen_sha, "--out", str(out)])


def test_decide_applies_the_frozen_train_only_rule(tmp_path):
    world, frozen, sha = decide_world(tmp_path)
    out = tmp_path / "decision.json"
    assert decide(world, frozen, sha, out) == 0
    result = json.loads(out.read_text())
    for ds in M.ANCHOR_DATASETS:
        d = result["decisions"][ds]
        assert d["adopted_axis_center"] == "anchors" and d["frozen_N"] == 9
        assert d["retrieval"]["passes"] and d["code_to_axis"]["passes"]


def test_equal_code_to_axis_does_not_adopt_anchors(tmp_path):
    world, frozen, sha = decide_world(tmp_path)
    for e in world.entries:
        probe = json.loads(Path(e["probe"]).read_text())
        probe["code_picks_own_axis"] = 0.33
        Path(e["probe"]).write_text(json.dumps(probe))
        e["probe_sha256"] = hashlib.sha256(Path(e["probe"]).read_bytes()).hexdigest()
    out = tmp_path / "decision.json"
    assert decide(world, frozen, sha, out) == 0
    assert {d["adopted_axis_center"] for d in json.loads(out.read_text())["decisions"].values()} == {"none"}


@pytest.mark.parametrize("case", ["other_checkpoint", "out_of_range", "no_images", "wrong_kind"])
def test_a_probe_that_is_not_this_checkpoints_valid_measurement_refuses(tmp_path, case):
    world, frozen, sha = decide_world(tmp_path)
    e = world.entries[0]
    probe = json.loads(Path(e["probe"]).read_text())
    probe.update({"other_checkpoint": {"checkpoint_sha256": "f" * 64},
                  "out_of_range": {"code_picks_own_axis": 1.2},
                  "no_images": {"n_images": 0},
                  "wrong_kind": {"artifact_kind": "quant_gap"}}[case])
    Path(e["probe"]).write_text(json.dumps(probe))
    e["probe_sha256"] = hashlib.sha256(Path(e["probe"]).read_bytes()).hexdigest()
    assert decide(world, frozen, sha, tmp_path / "d.json") == 1


def test_decide_refuses_evidence_off_the_frozen_n(tmp_path):
    world, frozen, sha = decide_world(tmp_path)
    shifted = {ds: {"none": 4, "anchors": 19} for ds in M.ANCHOR_DATASETS}
    Path(frozen).write_text(json.dumps({"artifact_kind": D.N_SELECTION_KIND,
                                        "version": M.ANCHOR_CONFIRM_VERSION, "n_selected": shifted}))
    assert decide(world, frozen, hashlib.sha256(Path(frozen).read_bytes()).hexdigest(),
                  tmp_path / "d.json") == 1


# ---- the probe's arithmetic and boundaries --------------------------------------------------------
def test_match_acc_scores_aligned_permuted_and_reports_chance():
    g = torch.Generator().manual_seed(0)
    t = torch.randn(64, 4, 16, generator=g)
    assert P.match_acc(t.clone(), t)["code_picks_own_axis"] == 1.0
    assert P.match_acc(t[:, [1, 2, 3, 0], :], t)["code_picks_own_axis"] == 0.0
    assert P.match_acc(t, t)["chance"] == 0.25


def test_probe_refuses_before_building_a_model_when_the_checkpoint_bytes_differ(tmp_path, monkeypatch):
    world = World(tmp_path, [("flickr25k", "anchors", 4, 42)])
    e = world.entries[0]
    run = Path(e["record"]).parent
    (run / "model_state_dict.pth").write_bytes(b"not the pinned checkpoint")
    import model_siglip2
    monkeypatch.setattr(model_siglip2, "SigLIP2SemanticOTModel",
                        lambda *a, **k: pytest.fail("a model was built from unverified bytes"))
    with pytest.raises(D.NotReducible, match="pinned bytes"):
        P.probe(e["record"], e["record_sha256"], device="cpu", n_images=512)


def test_probe_cli_fixes_the_image_count(tmp_path):
    assert P.main(["--record", "x", "--record-sha256", "0" * 64, "--out", str(tmp_path / "o"),
                   "--images", "256"]) == 1


def test_retrieval_exactly_at_the_margin_passes(tmp_path):
    """mean(anchors) == mean(none) - sd(none) is a pass (section 7.3: equality passes (i))."""
    world, frozen, sha = decide_world(tmp_path)
    for e in world.entries:                           # every seed of both arms scores 0.7
        run = Path(e["record"]).parent
        lines = (run / "log.csv").read_text().splitlines()
        lines[-1] = lines[-1].rsplit(",", 1)[0] + ",0.7"
        (run / "log.csv").write_text("\n".join(lines) + "\n")
        record = json.loads(Path(e["record"]).read_text())
        record["completion"]["log_csv_sha256"] = hashlib.sha256((run / "log.csv").read_bytes()).hexdigest()
        record["selection"]["selection_value"] = 0.7
        Path(e["record"]).write_text(json.dumps(record))
        e["record_sha256"] = hashlib.sha256(Path(e["record"]).read_bytes()).hexdigest()
    out = tmp_path / "decision.json"
    assert decide(world, frozen, sha, out) == 0
    decisions = json.loads(out.read_text())["decisions"]
    assert all(d["retrieval"]["passes"] and d["retrieval"]["control_sample_sd"] == 0.0
               for d in decisions.values())
