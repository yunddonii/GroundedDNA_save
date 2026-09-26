"""Anchor confirmation v1: reducer and probe, generation v2 (audit sections 665, 668, 671.2, 672, 673,
678-683).

A synthetic world mirrors the artifacts as the launcher writes them: per coordinate a run
directory (config.pt, checkpoint bytes, log.csv, trainer evidence, runtime sidecar) and a record
whose anchor block carries the completed recipe check; per campaign a receipt (campaign nonce,
complete campaign bindings and completion pins) and a plan snapshot that names the generation
manifest and the audit approval it ran under and seals each cell's recipe with the contract's
protocol values. A synthetic audit ledger holds the approval lines. Stage-D evidence joins the
stage-S records at the frozen N (seed 42) with a stage-D campaign (seeds 43/44) and v2 probes. The
generation-manifest loader is a stand-in (the launcher suite tests the real one). Each refusal
perturbs ONE thing of a world the positive control reduces, and asserts its reason. The reducer
deserialises nothing; probe tests stop at a torch.load or model-construction sentinel.
"""
from __future__ import annotations

import csv
import hashlib
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
from dna_utils.run_identity import PHASE3_CAMPAIGN_BINDING_NAME   # noqa: E402
from config import Config                                   # noqa: E402

INCUMBENT = {"flickr25k": {"N": 4, "topp": ("0.6", "0.95"), "joint": "0.02"},
             "nuswide": {"N": 4, "topp": ("0.4", "0.8"), "joint": "0.05"},
             "mscoco": {"N": 39, "topp": ("0.6", "0.95"), "joint": "0.03"}}
NAMES = {"flickr25k": "Flickr25k", "nuswide": "NUSWIDE", "mscoco": "MSCOCO"}
PARSER = Config.build_parser()
HF = {"identity_sha256": "1" * 64, "snapshot_dir": "/hf/snap", "revision": "r1",
      "weight_file": "pytorch_model.bin", "weight_sha256": "2" * 64, "config_sha256": "3" * 64,
      "tokenizer_files_sha256": {"vocab.json": "4" * 64}}
INPUT_AUTHORITY = {"seal_path": "/seals/x.json", "seal_file_sha256": "5" * 64,
                   "aggregate_sha256": "6" * 64, "split_identity_sha256": "7" * 64, "hf_runtime": HF}
ROWS = {ds: hashlib.sha256(ds.encode()).hexdigest() for ds in INCUMBENT}
MANIFEST_SHA, OTHER_SHA = "9" * 64, "8" * 64
MANIFESTS = {d: {"path": f"/manifests/{d[:4]}.json", "sha256": d, "commit": "c" * 40,
                 "contract": {"path": M.ANCHOR_CONTRACT_PATH, "sha256": "d" * 64}}
             for d in (MANIFEST_SHA, OTHER_SHA)}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def render_argv(c):
    """A stage-1 command carrying every protocol value of the contract (section 6)."""
    ds, arm, n, seed = c
    inc = INCUMBENT[ds]
    return ["--tag", f"{ds}_{arm}_{n}_{seed}", "--dataset", NAMES[ds], "--setting", "1",
            "-e", "60", "--lr_schedule_horizon", "60", "--sinkhorn_schedule_horizon", str(n + 1),
            "--stop_after_epoch", str(n), "--random_seed", str(seed), "--selection_mode", "select",
            "--keep_final_checkpoint", "--val_split_ratio", "0.1", "--val_split_seed", "42",
            "--routing_adaptive_topp", "--routing_adaptive_topp_min", inc["topp"][0],
            "--routing_adaptive_topp_max", inc["topp"][1], "--lambda_codon_joint", inc["joint"],
            "--lambda_codeword_codon_sinkhorn", "0.0", "--no-post_eval_compositional",
            "--dna_distance_mode", "base", "--num_semantic_parts", "5", "--num_codebooks", "5",
            "--num_codons_per_codebook", "3", "--codebook_size", "128", "--no_gumbel_softmax",
            "--text_hash_counterfactual_weight", "0.0", "--xmodal_commit_skip_global",
            "--cibhash_dynamic_tau_skip_global", "--no_visualize", "--hash_target_mode", "siglip_cos",
            "--axis_center", arm]


def run_argv(c):
    """What the run received: the render plus the campaign's sealed input flags."""
    return render_argv(c) + M._input_authority_flags(INPUT_AUTHORITY)


def score_of(c):
    ds, arm, n, seed = c
    base = {4: 0.70, 9: 0.72, 19: 0.71, 39: 0.69}.get(n, 0.5)   # N=9 wins; off-grid N scores low
    return round(base + (0.01 if arm == "anchors" else 0.0) + 0.001 * (seed - 42), 6)


def cell_id(c):
    ds, arm, n, seed = c
    return M.campaign_cell_id(ds, n, topp=INCUMBENT[ds]["topp"], joint=INCUMBENT[ds]["joint"],
                              stage="select", seed=seed, anchor_arm=arm)


def config_values(argv, *, historical):
    values = vars(PARSER.parse_args(argv))
    values["setting"] = "setting" + str(values["setting"])
    values.pop("log_dir", None)
    values["clip_snapshot_tokenizers_sha256_json"] = R.canonical_tokenizer_json(
        values["clip_snapshot_tokenizers_sha256_json"])
    if historical:
        del values["axis_center"]
    return values


def edit_json(path, **changes):
    """Set dotted keys in a JSON file; returns the new digest."""
    body = json.loads(Path(path).read_text())
    for key, value in changes.items():
        target = body
        *parents, leaf = key.split(".")
        for p in parents:
            target = target[p]
        target[leaf] = value
    Path(path).chmod(0o644)
    Path(path).write_text(json.dumps(body))
    return sha(Path(path).read_bytes())


class Ledger:
    """A synthetic audit ledger: numbered sections holding approval lines."""

    def __init__(self, path: Path):
        self.path, self.sections, self.next = path, {}, 900
        self.write()

    def line(self, scope, **pins):
        return " ".join([M.APPROVAL_TAG, f"version={M.ANCHOR_CONFIRM_VERSION}", f"scope={scope}",
                         *(f"{k}={v}" for k, v in pins.items())])

    def approve(self, scope, *, section=None, **pins):
        section = self.next if section is None else section
        self.next = max(self.next, section) + 1
        line = self.line(scope, **pins)
        self.sections.setdefault(section, []).append(line)
        self.write()
        return {"section": section, "scope": scope, "line": line}

    def write(self):
        text = ["# Synthetic audit ledger", ""]
        for n, lines in sorted(self.sections.items()):
            text += [f"## {n}. Test decision", ""] + [f"`{line}`" for line in lines] + [""]
        self.path.write_text("\n".join(text) + "\n")


@pytest.fixture(autouse=True)
def ledger(tmp_path, monkeypatch):
    book = Ledger(tmp_path / "ledger.md")
    monkeypatch.setattr(M, "AUDIT_LEDGER", book.path)
    return book


class World:
    """One campaign (receipt + snapshot) with its runs, or historical records for reuse. The
    campaign's canonical execution request is approved in the ledger unless `approval` is given
    (False: none); `request_changes` edit the request BEFORE it is approved."""

    def __init__(self, root: Path, coordinates, *, ledger: Ledger, approval=None, reuse_control=False,
                 manifest_sha=MANIFEST_SHA, stage="select", selection=None, approved_manifest=None,
                 request_changes=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.nonce = self.snapshot_nonce = sha(str(self.root).encode())
        self.manifest_sha = manifest_sha
        approved_manifest = approved_manifest or manifest_sha or MANIFEST_SHA
        cells = sorted(list(c) for c in coordinates)
        self.request = {"schema": M.REQUEST_SCHEMA, "version": M.ANCHOR_CONFIRM_VERSION, "stage": stage,
                        "mode": "run", "manifest": approved_manifest, "selection": selection,
                        "namespace": f"anc{stage.capitalize()}T", "record_dir": str(self.root / "records"),
                        "result_root": str(self.root / "runs"), "declared_cells": cells, "cells": cells,
                        "epochs": None, "input_seals": {}, "admission_authority": None, "gpu_count": 3,
                        **(request_changes or {})}
        scope = "stage-S-run" if stage == "select" else "stage-D-run"
        pins = {"manifest": approved_manifest, **({"selection": selection} if selection else {}),
                "request": M._json_digest(self.request)}
        self.approval = ledger.approve(scope, **pins) if approval is None else approval or None
        self.reuse_control = reuse_control
        self.entries, self.cells, self.bindings, self.reuse = [], {}, {}, []
        for c in coordinates:                                  # the sealed recipes first
            if not self.historical(c):
                payload = R.build_payload(PARSER, run_argv(c), planned_arm=c[1])
                self.bindings[cell_id(c)] = {"anchor_arm": c[1], "scientific_recipe": payload,
                                             "expected_scientific_recipe_sha256": R.digest(payload)}
        self.write_snapshot()
        for c in coordinates:
            self.entries.append(self.make(c))
        self.write_receipt()

    def historical(self, c):
        return self.reuse_control and c[1] == "none"

    def write_snapshot(self):
        authorities = {}
        if self.manifest_sha:
            authorities["anchor_manifest"] = {"sha256": self.manifest_sha}
        if self.approval:
            authorities["anchor_approval"] = self.approval
            if not getattr(self, "omit_request", False):
                authorities["anchor_request"] = self.request
        self.snapshot = {"plan": {"authorities": authorities, "campaign_nonce": self.snapshot_nonce,
                                  "cell_bindings": self.bindings}}
        (self.root / "snapshot.json").write_text(json.dumps(self.snapshot))
        self.snapshot_sha = M._json_digest(self.snapshot)

    def make(self, c):
        ds, arm, n, seed = c
        run = self.root / "runs" / "_".join(map(str, c))
        run.mkdir(parents=True)
        historical = self.historical(c)
        torch.save(config_values(run_argv(c), historical=historical), run / "config.pt")
        config_sha = sha((run / "config.pt").read_bytes())
        (run / "model_state_dict.pth").write_bytes(str(c).encode())
        with (run / "log.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["epoch", "eval_mAP_at_R"])
            for e in range(n):
                writer.writerow([e, ""])
            writer.writerow([n, score_of(c)])
        cid = cell_id(c)
        recipe = self.bindings[cid]["expected_scientific_recipe_sha256"] if not historical else "0" * 64
        evidence = {"cell_id": cid, "scientific_recipe_sha256": recipe}
        (run / PHASE3_CAMPAIGN_BINDING_NAME).write_text(json.dumps(evidence))
        sidecar = {"checkpoint_epoch_zero_based": n, "extra": {"phase3_campaign": evidence}}
        (run / "model_state_dict.pth.runtime.json").write_text(json.dumps(sidecar))
        campaign = {"cell_id": cid, "plan_snapshot_sha256": self.snapshot_sha, "campaign_nonce": self.nonce}
        completion = {"final_checkpoint_epoch_zero_based": n, "final_checkpoint": "model_state_dict.pth",
                      "final_checkpoint_sha256": sha((run / "model_state_dict.pth").read_bytes()),
                      "log_csv_sha256": sha((run / "log.csv").read_bytes()),
                      "checkpoint_runtime_sha256": sha((run / "model_state_dict.pth.runtime.json").read_bytes()),
                      "phase3_campaign_evidence_sha256": sha((run / PHASE3_CAMPAIGN_BINDING_NAME).read_bytes())}
        record = {"dataset": ds, "N": n, "seed": seed, "stage": "select", "selection_mode": "select",
                  "smoke": False, "is_candidate_cell": True, "val_split_ratio": 0.1, "val_split_seed": 42,
                  "run_dir": str(run), "campaign": campaign, "completion": completion,
                  "input_authority": INPUT_AUTHORITY,
                  "selection": {"selection_epoch_zero_based": n, "selection_metric": "eval_mAP_at_R",
                                "distance_mode": "base", "selection_value": score_of(c)}}
        if not historical:
            record["anchor_confirmation"] = {
                "version": M.ANCHOR_CONFIRM_VERSION, "arm": arm, "scientific_recipe_sha256": recipe,
                "campaign_evidence_sha256": completion["phase3_campaign_evidence_sha256"],
                "config_pt_sha256": config_sha}
        path = run / f"{cid.replace('|', '_')}.json"
        path.write_text(json.dumps(record))
        entry = {"dataset": ds, "arm": arm, "N": n, "seed": seed, "record": str(path),
                 "record_sha256": sha(path.read_bytes())}
        if historical:
            entry["source"] = "reuse"
            self.reuse.append({**{k: entry[k] for k in ("dataset", "arm", "N", "seed", "record",
                                                         "record_sha256")}, "config_pt_sha256": config_sha})
        else:
            entry["source"] = "receipt"
            self.cells[cid] = {"record": path.name, "record_sha256": entry["record_sha256"],
                               "campaign": campaign,
                               "completion": {k: completion[k] for k in D.RECEIPT_COMPLETION_KEYS}}
        return entry

    def write_receipt(self):
        receipt = {"campaign_kind": M.ANCHOR_CAMPAIGN_KIND,
                   "anchor_confirmation": {"version": M.ANCHOR_CONFIRM_VERSION},
                   "campaign_nonce": self.nonce, "namespace": self.request["namespace"],
                   "plan_snapshot_file": "snapshot.json",
                   "plan_snapshot_sha256": self.snapshot_sha, "cells": self.cells}
        path = self.root / "receipt.json"
        path.write_text(json.dumps(receipt))
        for e in self.entries:
            if e["source"] == "receipt":
                e.update(receipt=str(path), receipt_sha256=sha(path.read_bytes()))

    def reissue_snapshot(self):
        """Rewrite the snapshot and re-issue every record's campaign binding consistently, so a
        change to the snapshot is judged on its own."""
        self.write_snapshot()
        for i, e in enumerate(self.entries):
            if e["source"] == "receipt":
                self.rewrite_record(i, **{"campaign.plan_snapshot_sha256": self.snapshot_sha})

    def reseal(self, c, **changes):
        """Change one sealed recipe (fields, or `schema`) under a recomputed declared digest, and
        re-issue that cell's trainer evidence, sidecar and anchor block consistently, so only the
        recipe's own checks can refuse it."""
        cid = cell_id(c)
        payload = json.loads(json.dumps(self.bindings[cid]["scientific_recipe"]))
        for key, value in changes.items():
            if key == "schema":
                payload["schema"] = value
            else:
                payload["fields"][key] = value
        digest = R.digest(payload)
        self.bindings[cid]["scientific_recipe"] = payload
        self.bindings[cid]["expected_scientific_recipe_sha256"] = digest
        i = next(i for i, e in enumerate(self.entries) if (e["dataset"], e["arm"], e["N"], e["seed"]) == c)
        run = Path(self.entries[i]["record"]).parent
        evidence = {"cell_id": cid, "scientific_recipe_sha256": digest}
        (run / PHASE3_CAMPAIGN_BINDING_NAME).write_text(json.dumps(evidence))
        sidecar = run / "model_state_dict.pth.runtime.json"
        sidecar.write_text(json.dumps({"checkpoint_epoch_zero_based": c[2], "extra": {"phase3_campaign": evidence}}))
        evidence_sha = sha((run / PHASE3_CAMPAIGN_BINDING_NAME).read_bytes())
        self.rewrite_record(i, **{"completion.phase3_campaign_evidence_sha256": evidence_sha,
                                  "completion.checkpoint_runtime_sha256": sha(sidecar.read_bytes()),
                                  "anchor_confirmation.scientific_recipe_sha256": digest,
                                  "anchor_confirmation.campaign_evidence_sha256": evidence_sha})
        self.reissue_snapshot()

    def reuse_admission(self, **changes):
        body = {"artifact_kind": D.REUSE_KIND, "version": M.ANCHOR_CONFIRM_VERSION, "coordinates": self.reuse}
        body.update(changes)
        path = self.root / "reuse.json"
        path.write_text(json.dumps(body))
        return str(path), sha(path.read_bytes())

    def sources(self, entries=None):
        path = self.root / "sources.json"
        path.write_text(json.dumps({"version": M.ANCHOR_CONFIRM_VERSION,
                                    "coordinates": self.entries if entries is None else entries}))
        return str(path), sha(path.read_bytes())

    edit_json = staticmethod(edit_json)

    def rewrite_record(self, index, reissue=True, **changes):
        """Edit a record. By default everything downstream of it is re-issued consistently (the
        receipt's cell and the probe's record digest), so the edit is judged on its own merits;
        `reissue=False` leaves the receipt describing the old record."""
        e = self.entries[index]
        e["record_sha256"] = edit_json(e["record"], **changes)
        if reissue and "probe" in e:
            e["probe_sha256"] = edit_json(e["probe"], record_sha256=e["record_sha256"])
        if e["source"] == "receipt":
            cid = cell_id((e["dataset"], e["arm"], e["N"], e["seed"]))
            self.cells[cid]["record_sha256"] = e["record_sha256"]
            if reissue:
                record = json.loads(Path(e["record"]).read_text())
                self.cells[cid]["campaign"] = record["campaign"]
                self.cells[cid]["completion"] = {k: record["completion"][k]
                                                 for k in self.cells[cid]["completion"]}
            self.write_receipt()
        else:
            for r in self.reuse:
                if r["record"] == e["record"]:
                    r["record_sha256"] = e["record_sha256"]

    def add_probe(self, e, approval, request_sha256, *, manifest_sha=MANIFEST_SHA):
        ds, arm, n, seed = e["dataset"], e["arm"], e["N"], e["seed"]
        record = json.loads(Path(e["record"]).read_text())
        hits = 820 if arm == "anchors" else 680
        probe = {"artifact_kind": D.PROBE_KIND, "schema": D.PROBE_SCHEMA,
                 "producer_sha256": sha(D.PROBE_SOURCE.read_bytes()),
                 "manifest_sha256": manifest_sha, "approval": approval, "request_sha256": request_sha256,
                 "coordinate": [ds, arm, n, seed], "record_sha256": e["record_sha256"],
                 "checkpoint_sha256": record["completion"]["final_checkpoint_sha256"],
                 "config_pt_sha256": record["anchor_confirmation"]["config_pt_sha256"],
                 "split": dict(D.SPLIT), "split_identity_sha256": INPUT_AUTHORITY["split_identity_sha256"],
                 "routing": "deployment_no_text", "n_images": 512, "row_ids_sha256": ROWS[ds],
                 "caption_target": {**D.CAPTION_TARGET,
                                    "input_seal_sha256": INPUT_AUTHORITY["seal_file_sha256"]},
                 "hits": hits, "total": 2048, "ties_counted_as_misses": 3,
                 "code_picks_own_axis": hits / 2048,
                 "authority": {"kind": "receipt", "receipt": e["receipt"], "cell_id": cell_id((ds, arm, n, seed)),
                               "manifest_sha256": self.manifest_sha,
                               "approval": {**self.approval, "request_sha256": M._json_digest(self.request)}}}
        path = Path(e["record"]).parent / "probe.json"
        path.write_text(json.dumps(probe))
        e.update(probe=str(path), probe_sha256=sha(path.read_bytes()))


def fake_manifest(path, sha256):
    """Stand-in for M.load_anchor_manifest: the two known generations load, anything else refuses."""
    if sha256 not in MANIFESTS:
        raise M.CellRefused(f"{path} is not the pinned bytes {sha256[:12]}...")
    return dict(MANIFESTS[sha256])


@pytest.fixture(autouse=True)
def stand_ins(monkeypatch):
    monkeypatch.setattr(M, "anchor_incumbent", lambda: INCUMBENT)
    monkeypatch.setattr(M, "load_anchor_manifest", fake_manifest)
    monkeypatch.setattr(M, "APPROVED_REUSE_ADMISSION_SHA256", {})


def approved(monkeypatch, admission):
    """Approve a reuse admission the only way the reducer accepts: its digest listed in source."""
    monkeypatch.setattr(M, "APPROVED_REUSE_ADMISSION_SHA256", {admission[1]: "ledger section 999 (test)"})
    return admission


def run_select(world, out, *, sources=None, reuse=None, arms="none,anchors", manifest=MANIFEST_SHA):
    path, digest = sources or world.sources()
    argv = ["select", "--sources", path, "--sources-sha256", digest, "--arms", arms, "--out", str(out),
            "--manifest", MANIFESTS.get(manifest, {"path": "/m"})["path"], "--manifest-sha256", manifest]
    if reuse:
        argv += ["--reuse-admission", reuse[0], "--reuse-admission-sha256", reuse[1]]
    return D.main(argv)


SELECT = D.expected_coordinates("select", arms=M.ANCHOR_ARMS)


@pytest.fixture
def world(tmp_path, ledger):
    return World(tmp_path / "S", SELECT, ledger=ledger)


@pytest.fixture
def reuse_world(tmp_path, ledger):
    return World(tmp_path / "S", SELECT, ledger=ledger, reuse_control=True)


@pytest.fixture
def no_deserialisation(monkeypatch):
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("a binary was deserialised"))


def index_of(world, *, arm):
    return next(i for i, e in enumerate(world.entries) if e["arm"] == arm)


# ---- select: positive controls ----------------------------------------------------------------------
def test_select_reduces_new_evidence_to_the_argmax_n_and_writes_once(world, tmp_path, no_deserialisation):
    out = tmp_path / "n.json"
    assert run_select(world, out) == 0
    result = json.loads(out.read_text())
    assert result["n_selected"] == {ds: {"none": 9, "anchors": 9} for ds in M.ANCHOR_DATASETS}
    assert {e["authority"]["kind"] for e in result["evidence"].values()} == {"receipt"}
    assert {e["authority"]["approval"]["scope"] for e in result["evidence"].values()} == {"stage-S-run"}
    assert run_select(world, out) == 1                                  # O_EXCL


def test_the_sealed_recipes_of_the_fixture_are_the_contracts(world):
    """The positive control is valid under the contract, not merely self-consistent."""
    for cid, binding in world.bindings.items():
        c = next((e["dataset"], e["arm"], e["N"], e["seed"]) for e in world.entries
                 if cell_id((e["dataset"], e["arm"], e["N"], e["seed"])) == cid)
        want = M.anchor_protocol_fields(c[0], c[2], seed=c[3], arm=c[1], incumbent=INCUMBENT)
        assert {k: binding["scientific_recipe"]["fields"][k] for k in want} == want


def test_select_accepts_admitted_historical_control_reuse(reuse_world, tmp_path, monkeypatch):
    out = tmp_path / "n.json"
    assert run_select(reuse_world, out, reuse=approved(monkeypatch, reuse_world.reuse_admission())) == 0
    kinds = {k: v["authority"]["kind"] for k, v in json.loads(out.read_text())["evidence"].items()}
    assert {kinds[k] for k in kinds if "|none|" in k} == {"reuse"}
    assert {kinds[k] for k in kinds if "|anchors|" in k} == {"receipt"}


def test_ties_go_to_the_smallest_n():
    assert D.select_n({4: 0.7, 9: 0.72, 19: 0.72, 39: 0.1}) == 9


# ---- membership ---------------------------------------------------------------------------------------
MEMBERSHIP = {"missing": "membership differs from the contract", "extra": "membership differs",
              "duplicate": "duplicate evidence", "reused_record": "reused across coordinates"}


@pytest.mark.parametrize("case", MEMBERSHIP)
def test_membership_must_be_exactly_the_contract(world, tmp_path, capsys, case):
    entries = [dict(e) for e in world.entries]
    if case == "missing":
        entries.pop()
    elif case == "extra":
        entries.append(dict(entries[0], N=5))
    elif case == "duplicate":
        entries.append(dict(entries[0]))
    else:
        entries[1]["record"], entries[1]["record_sha256"] = entries[0]["record"], entries[0]["record_sha256"]
    assert run_select(world, tmp_path / "o.json", sources=world.sources(entries)) == 1
    assert MEMBERSHIP[case] in capsys.readouterr().err
    assert not (tmp_path / "o.json").exists()


# ---- JSON-level record admission ------------------------------------------------------------------------
PERTURB = {
    "smoke": ({"smoke": True}, "smoke or non-candidate"),
    "not_candidate": ({"is_candidate_cell": False}, "smoke or non-candidate"),
    "refit_stage": ({"stage": "refit"}, "not a stage-1 selection record"),
    "no_validation_split": ({"val_split_ratio": 0.0}, "90/10 train-only split"),
    "other_split_seed": ({"val_split_seed": 43}, "90/10 train-only split"),
    "checkpoint_epoch": ({"completion.final_checkpoint_epoch_zero_based": 3}, "epoch is not N"),
    "selection_epoch": ({"selection.selection_epoch_zero_based": 3}, "epoch is not N"),
    "other_metric": ({"selection.selection_metric": "eval_mAP"}, "raw base-Hamming"),
    "other_distance": ({"selection.distance_mode": "codeword"}, "raw base-Hamming"),
    "selection_value": ({"selection.selection_value": 0.99}, "not the pinned log's"),
    "wrong_coordinate": ({"seed": 43}, "not the record of"),
    "float_n": ({"N": 4.0}, "not the record of"),
    "float_seed": ({"seed": 42.0}, "not the record of"),
    "float_terminal_epoch": ({"completion.final_checkpoint_epoch_zero_based": 4.0}, "epoch is not N"),
    "string_split_ratio": ({"val_split_ratio": "0.1"}, "90/10 train-only split"),
    "bool_split_seed": ({"val_split_seed": True}, "90/10 train-only split"),
    "float_split_seed": ({"val_split_seed": 42.0}, "90/10 train-only split"),
}


@pytest.mark.parametrize("case", PERTURB)
def test_a_record_that_is_not_train_only_terminal_evidence_refuses(world, tmp_path, capsys, case):
    change, reason = PERTURB[case]
    world.rewrite_record(index_of(world, arm="anchors"), **change)
    assert run_select(world, tmp_path / "o.json") == 1
    assert reason in capsys.readouterr().err


@pytest.mark.parametrize("value", ["nan", "inf", "1.5", "-0.1", ""])
def test_a_terminal_score_that_is_not_a_proportion_refuses(world, tmp_path, capsys, value):
    i = index_of(world, arm="none")
    run = Path(world.entries[i]["record"]).parent
    lines = (run / "log.csv").read_text().splitlines()
    lines[-1] = lines[-1].rsplit(",", 1)[0] + "," + value
    (run / "log.csv").write_text("\n".join(lines) + "\n")
    world.rewrite_record(i, **{"completion.log_csv_sha256": sha((run / "log.csv").read_bytes())})
    assert run_select(world, tmp_path / "o.json") == 1
    err = capsys.readouterr().err
    assert ("not finite" in err or "outside [0, 1]" in err or "no mAP@R" in err)


# ---- receipt authority (audit 672.1, 679.2) ---------------------------------------------------------------
def test_a_record_the_receipt_does_not_list_refuses(world, tmp_path, capsys):
    cid = cell_id(("flickr25k", "anchors", 4, 42))
    del world.cells[cid]
    world.write_receipt()
    assert run_select(world, tmp_path / "o.json") == 1
    assert "no completed cell" in capsys.readouterr().err


@pytest.mark.parametrize("change,reason", [
    ({"campaign.cell_id": "other"}, "differs from the receipt"),
    ({"completion.final_checkpoint_sha256": "9" * 64}, "differs from the receipt"),
    ({"anchor_confirmation.arm": "none"}, "no anchor evidence for arm"),
])
def test_receipt_evidence_mismatch_refuses(world, tmp_path, capsys, change, reason):
    world.rewrite_record(index_of(world, arm="anchors"), reissue=False, **change)
    assert run_select(world, tmp_path / "o.json") == 1
    assert reason in capsys.readouterr().err


@pytest.mark.parametrize("field,value,reason", [
    ("campaign", {}, "campaign binding of"),                                 # 679.2: empty objects
    ("campaign", None, "campaign binding of"),
    ("completion", {}, "completion pins of"),
    ("completion", {"final_checkpoint_sha256": "a" * 64}, "completion pins of"),
])
def test_a_receipt_cell_without_its_campaign_proof_refuses(world, tmp_path, capsys, field, value, reason):
    cid = cell_id(("flickr25k", "anchors", 4, 42))
    world.cells[cid][field] = value
    world.write_receipt()
    assert run_select(world, tmp_path / "o.json") == 1
    assert reason in capsys.readouterr().err


def test_an_empty_campaign_binding_on_both_sides_refuses(world, tmp_path, capsys):
    """Record and receipt agree -- on an empty binding (audit 679.2): agreement is not proof."""
    world.rewrite_record(index_of(world, arm="anchors"), campaign={})
    assert run_select(world, tmp_path / "o.json") == 1
    assert "campaign binding of" in capsys.readouterr().err


def test_a_receipt_of_another_campaign_nonce_refuses(world, tmp_path, capsys):
    new = edit_json(world.root / "receipt.json", campaign_nonce="b" * 64)
    for e in world.entries:
        e["receipt_sha256"] = new
    assert run_select(world, tmp_path / "o.json") == 1
    assert "another campaign's" in capsys.readouterr().err


def test_a_snapshot_of_another_campaign_refuses(world, tmp_path, capsys):
    world.snapshot_nonce = "c" * 64
    world.reissue_snapshot()
    assert run_select(world, tmp_path / "o.json") == 1
    assert "the snapshot is another campaign's" in capsys.readouterr().err


def test_a_snapshot_that_is_not_the_receipts_refuses(world, tmp_path, capsys):
    snap = world.root / "snapshot.json"
    body = json.loads(snap.read_text())
    body["plan"]["extra"] = 1
    snap.write_text(json.dumps(body))
    assert run_select(world, tmp_path / "o.json") == 1
    assert "not the receipt's plan snapshot" in capsys.readouterr().err


def test_evidence_naming_another_recipe_refuses(world, tmp_path, capsys):
    i = index_of(world, arm="anchors")
    run = Path(world.entries[i]["record"]).parent
    new = edit_json(run / PHASE3_CAMPAIGN_BINDING_NAME, scientific_recipe_sha256="8" * 64)
    world.rewrite_record(i, **{"completion.phase3_campaign_evidence_sha256": new,
                               "anchor_confirmation.campaign_evidence_sha256": new})
    assert run_select(world, tmp_path / "o.json") == 1
    assert "does not name" in capsys.readouterr().err


def test_a_sidecar_that_is_not_the_terminal_witness_refuses(world, tmp_path, capsys):
    i = index_of(world, arm="anchors")
    run = Path(world.entries[i]["record"]).parent
    new = edit_json(run / "model_state_dict.pth.runtime.json", checkpoint_epoch_zero_based=0)
    world.rewrite_record(i, **{"completion.checkpoint_runtime_sha256": new})
    assert run_select(world, tmp_path / "o.json") == 1
    assert "terminal checkpoint witness" in capsys.readouterr().err


@pytest.mark.parametrize("change", [{"anchor_confirmation.config_pt_sha256": "Z" * 64},
                                    {"anchor_confirmation.scientific_recipe_sha256": "0" * 64},
                                    {"anchor_confirmation.version": "anchor-confirm/0"}])
def test_a_record_without_its_completed_recipe_check_refuses(world, tmp_path, capsys, change):
    world.rewrite_record(index_of(world, arm="anchors"), **change)
    assert run_select(world, tmp_path / "o.json") == 1
    assert "no anchor evidence for arm" in capsys.readouterr().err


def test_config_bytes_off_their_pin_refuse_without_deserialising(world, tmp_path, capsys, no_deserialisation):
    """Audit 679.2: byte identity before (here: instead of) any deserialisation."""
    run = Path(world.entries[index_of(world, arm="anchors")]["record"]).parent
    (run / "config.pt").write_bytes((run / "config.pt").read_bytes() + b"\0")
    assert run_select(world, tmp_path / "o.json") == 1
    assert "config.pt is not the pinned bytes" in capsys.readouterr().err


# ---- the sealed recipe itself (audit 679.2) --------------------------------------------------------------
@pytest.mark.parametrize("change,reason", [
    ({"schema": "groundeddna-scientific-recipe/1"}, "the sealed recipe is malformed"),
    ({"axis_center": "none"}, "axis_center"),                     # the sealed arm is not the outer arm
    ({"stop_after_epoch": 5}, "stop_after_epoch"),
    ({"hash_target_mode": "jaccard"}, "hash_target_mode"),
    ({"routing_adaptive_topp_min": 0.3}, "routing_adaptive_topp_min"),
])
def test_a_sealed_recipe_off_the_contract_refuses(world, tmp_path, capsys, change, reason):
    world.reseal(("flickr25k", "anchors", 4, 42), **change)
    assert run_select(world, tmp_path / "o.json") == 1
    assert reason in capsys.readouterr().err


def test_a_sealed_recipe_off_its_declared_digest_refuses(world, tmp_path, capsys):
    world.bindings[cell_id(("flickr25k", "anchors", 4, 42))]["expected_scientific_recipe_sha256"] = "0" * 64
    world.reissue_snapshot()
    assert run_select(world, tmp_path / "o.json") == 1
    assert "not its declared digest" in capsys.readouterr().err


# ---- approval authority (audit 679.1) ---------------------------------------------------------------------
def test_a_campaign_without_an_approval_refuses(tmp_path, ledger, capsys):
    world = World(tmp_path / "S", SELECT, ledger=ledger, approval=False)
    assert run_select(world, tmp_path / "o.json") == 1
    assert "names no audit approval" in capsys.readouterr().err


def test_a_withdrawn_approval_line_refuses(world, ledger, tmp_path, capsys):
    ledger.sections[world.approval["section"]] = []
    ledger.write()
    assert run_select(world, tmp_path / "o.json") == 1
    assert "has 0 approval lines for stage-S-run" in capsys.readouterr().err


def test_a_smoke_approval_is_not_a_run_approval(tmp_path, ledger, capsys):
    smoke = ledger.approve("stage-S-smoke", manifest=MANIFEST_SHA, request="a" * 64)
    world = World(tmp_path / "S", SELECT, ledger=ledger, approval=smoke)
    assert run_select(world, tmp_path / "o.json") == 1
    assert "ran under approval scope 'stage-S-smoke'" in capsys.readouterr().err


def test_an_approval_changed_after_the_run_refuses(world, ledger, tmp_path, capsys):
    ledger.sections[world.approval["section"]] = [
        ledger.line("stage-S-run", manifest=OTHER_SHA, request=M._json_digest(world.request))]
    ledger.write()
    assert run_select(world, tmp_path / "o.json") == 1
    assert "approves stage-S-run for" in capsys.readouterr().err


def test_a_recorded_approval_line_that_is_not_the_ledgers_refuses(world, tmp_path, capsys):
    world.approval = dict(world.approval, line=world.approval["line"] + " note=x")
    world.reissue_snapshot()
    assert run_select(world, tmp_path / "o.json") == 1
    assert "is not the ledger's" in capsys.readouterr().err


# ---- the approved execution request (audit 689.2) ----------------------------------------------------------
@pytest.mark.parametrize("request_changes", [
    {"cells": [["nuswide", "none", 4, 42]]},                       # the record's cell is not in the request
    {"cells": [["flickr25k", "anchors", 4.0, 42]]},                # nor a float-N lookalike
    {"result_root": "/elsewhere/runs"},                            # the runs are not under its root
    {"mode": "smoke"},
    {"stage": "decide"},
    {"schema": "anchor-confirm-request/0"},
])
def test_a_record_outside_its_approved_request_refuses(tmp_path, ledger, capsys, request_changes):
    world = World(tmp_path / "S", SELECT, ledger=ledger, request_changes=request_changes)
    assert run_select(world, tmp_path / "o.json") == 1
    assert "is not the one this record ran under" in capsys.readouterr().err


def test_a_receipt_of_another_namespace_than_its_request_refuses(world, tmp_path, capsys):
    new = edit_json(world.root / "receipt.json", namespace="ancOther")
    for e in world.entries:
        e["receipt_sha256"] = new
    assert run_select(world, tmp_path / "o.json") == 1
    assert "is not the one this record ran under" in capsys.readouterr().err


def test_a_request_edited_after_its_approval_refuses(world, tmp_path, capsys):
    world.request["namespace"] = "ancRepeat"                        # a new namespace, the old approval
    world.reissue_snapshot()
    assert run_select(world, tmp_path / "o.json") == 1
    assert "approves stage-S-run for" in capsys.readouterr().err


def test_a_campaign_without_its_request_refuses(world, tmp_path, capsys):
    world.omit_request = True
    world.reissue_snapshot()
    assert run_select(world, tmp_path / "o.json") == 1
    assert "names no execution request" in capsys.readouterr().err


# ---- reuse authority (audit 672.1, 683.1) -----------------------------------------------------------------
def test_historical_records_without_a_reuse_admission_refuse(reuse_world, tmp_path, capsys):
    assert run_select(reuse_world, tmp_path / "o.json") == 1
    assert "needs a pinned reuse admission" in capsys.readouterr().err


@pytest.mark.parametrize("claims", [{}, {"admitted_by": "ledger section 999 (test)"}, {"admitted_by": None}])
def test_a_self_pinned_reuse_admission_is_not_approval(reuse_world, tmp_path, capsys, claims):
    """Audit 678.2: the caller's digest plus a file naming its own approver admits nothing."""
    assert run_select(reuse_world, tmp_path / "o.json", reuse=reuse_world.reuse_admission(**claims)) == 1
    assert "no audit-approved reuse admission" in capsys.readouterr().err


def test_a_reuse_admission_must_be_one(reuse_world, tmp_path, capsys, monkeypatch):
    proposal = reuse_world.reuse_admission(artifact_kind="anchor_confirmation_reuse_proposal")
    assert run_select(reuse_world, tmp_path / "o.json", reuse=approved(monkeypatch, proposal)) == 1
    assert "not a reuse admission" in capsys.readouterr().err


def test_a_coordinate_the_reuse_admission_does_not_list_refuses(reuse_world, tmp_path, capsys, monkeypatch):
    reuse_world.reuse.pop()
    assert run_select(reuse_world, tmp_path / "o.json",
                      reuse=approved(monkeypatch, reuse_world.reuse_admission())) == 1
    assert "does not list this record" in capsys.readouterr().err


def test_a_reuse_admission_without_config_pins_refuses(reuse_world, tmp_path, capsys, monkeypatch):
    for r in reuse_world.reuse:
        r.pop("config_pt_sha256")
    assert run_select(reuse_world, tmp_path / "o.json",
                      reuse=approved(monkeypatch, reuse_world.reuse_admission())) == 1
    assert "no config.pt pin" in capsys.readouterr().err


def test_a_reused_config_off_its_approved_pin_refuses(reuse_world, tmp_path, capsys, monkeypatch,
                                                      no_deserialisation):
    run = Path(reuse_world.entries[index_of(reuse_world, arm="none")]["record"]).parent
    (run / "config.pt").write_bytes((run / "config.pt").read_bytes() + b"\0")
    assert run_select(reuse_world, tmp_path / "o.json",
                      reuse=approved(monkeypatch, reuse_world.reuse_admission())) == 1
    assert "config.pt is not the pinned bytes" in capsys.readouterr().err


def test_reuse_is_never_inferred_from_a_missing_anchor_block(world, tmp_path, capsys):
    entries = [dict(e) for e in world.entries]
    i = index_of(world, arm="none")
    entries[i]["source"] = None
    assert run_select(world, tmp_path / "o.json", sources=world.sources(entries)) == 1
    assert "must be 'receipt' or 'reuse'" in capsys.readouterr().err


# ---- one generation for the whole chain (audit 671.2, 678.2) ------------------------------------------------
def test_a_campaign_under_another_generation_refuses(tmp_path, ledger, capsys):
    """The snapshot names another manifest than the approval and the reducer's."""
    world = World(tmp_path / "S", SELECT, ledger=ledger, manifest_sha=OTHER_SHA, approved_manifest=MANIFEST_SHA)
    assert run_select(world, tmp_path / "o.json") == 1
    assert "ran under generation manifests" in capsys.readouterr().err


def test_a_campaign_approved_for_another_generation_refuses(tmp_path, ledger, capsys):
    world = World(tmp_path / "S", SELECT, ledger=ledger, manifest_sha=OTHER_SHA)
    assert run_select(world, tmp_path / "o.json") == 1
    assert "approves stage-S-run for" in capsys.readouterr().err


def test_a_campaign_that_names_no_generation_refuses(tmp_path, ledger, capsys):
    world = World(tmp_path / "S", SELECT, ledger=ledger, manifest_sha=None)
    assert run_select(world, tmp_path / "o.json") == 1
    assert "names no generation manifest" in capsys.readouterr().err


def test_the_reducer_needs_the_manifest_of_this_tree(world, tmp_path, capsys):
    assert run_select(world, tmp_path / "o.json", manifest="7" * 64) == 1
    assert "not the pinned bytes" in capsys.readouterr().err


def test_the_n_record_carries_its_generation(world, tmp_path):
    out = tmp_path / "n.json"
    assert run_select(world, out) == 0
    assert json.loads(out.read_text())["generation"] == {
        "anchor_manifest_sha256": MANIFEST_SHA, "commit": "c" * 40, "contract_sha256": "d" * 64}


# ---- whole-run properties ---------------------------------------------------------------------------
def test_a_log_replaced_after_its_pin_refuses(world, tmp_path, capsys):
    run = Path(world.entries[2]["record"]).parent
    (run / "log.csv").write_text((run / "log.csv").read_text().replace("0.7", "0.9"))
    assert run_select(world, tmp_path / "o.json") == 1
    assert "not the pinned bytes" in capsys.readouterr().err


def test_an_input_changed_after_it_was_read_refuses_the_write(world, tmp_path, monkeypatch, capsys):
    original = D.Consumed.reverify
    def tamper(self):
        Path(world.entries[0]["record"]).write_text("{}")
        return original(self)
    monkeypatch.setattr(D.Consumed, "reverify", tamper)
    assert run_select(world, tmp_path / "o.json") == 1
    assert "changed before the output was written" in capsys.readouterr().err
    assert not (tmp_path / "o.json").exists()


# ---- decide ---------------------------------------------------------------------------------------------
class Evidence:
    """Stage-D reduction input: the stage-S records at the frozen N (seed 42) and the stage-D
    campaign's records (seeds 43/44), each owned by its own campaign."""

    def __init__(self, root: Path, worlds, coordinates):
        self.root, self.owner, self.entries = root, {}, []
        for w in worlds:
            for e in w.entries:
                if (e["dataset"], e["arm"], e["N"], e["seed"]) in coordinates:
                    self.entries.append(e)
                    self.owner[id(e)] = w

    def sources(self, entries=None):
        path = self.root / "sources_decide.json"
        path.write_text(json.dumps({"version": M.ANCHOR_CONFIRM_VERSION,
                                    "coordinates": self.entries if entries is None else entries}))
        return str(path), sha(path.read_bytes())

    def rewrite_record(self, index, **changes):
        e = self.entries[index]
        w = self.owner[id(e)]
        w.rewrite_record(next(i for i, x in enumerate(w.entries) if x is e), **changes)


def decide_world(tmp_path, ledger, *, d_manifest=MANIFEST_SHA, d_selection=None, probe_request=None):
    """Stage S reduced to a real frozen N record (N=9 both arms), a stage-D campaign approved for
    it, and probes approved for exactly those records."""
    s_world = World(tmp_path / "S", SELECT, ledger=ledger)
    n_path = tmp_path / "n.json"
    assert run_select(s_world, n_path) == 0
    frozen_sha = sha(n_path.read_bytes())
    frozen = json.loads(n_path.read_text())["n_selected"]
    coords = D.expected_coordinates("decide", arms=M.ANCHOR_ARMS, frozen=frozen)
    d_world = World(tmp_path / "D", [c for c in coords if c[3] != M.SEED], ledger=ledger,
                    manifest_sha=d_manifest, stage="decide", selection=d_selection or frozen_sha)
    evidence = Evidence(tmp_path, (s_world, d_world), coords)
    request = probe_request or D.probe_request(
        MANIFEST_SHA, frozen_sha, {(e["dataset"], e["arm"], e["N"], e["seed"]): e["record_sha256"]
                                   for e in evidence.entries})
    request_sha = M._json_digest(request)
    probe_approval = ledger.approve("probe", manifest=MANIFEST_SHA, selection=frozen_sha, request=request_sha)
    for e in evidence.entries:
        evidence.owner[id(e)].add_probe(e, probe_approval, request_sha)
    return evidence, str(n_path), frozen_sha


def run_decide(evidence, frozen, frozen_sha, out, sources=None, manifest=MANIFEST_SHA):
    path, digest = sources or evidence.sources()
    return D.main(["decide", "--sources", path, "--sources-sha256", digest, "--selection", frozen,
                   "--selection-sha256", frozen_sha, "--out", str(out),
                   "--manifest", MANIFESTS[manifest]["path"], "--manifest-sha256", manifest])


def test_decide_replays_selection_and_applies_the_frozen_rule(tmp_path, ledger, no_deserialisation):
    evidence, frozen, frozen_sha = decide_world(tmp_path, ledger)
    assert {e["seed"] for e in evidence.entries} == {42, 43, 44}
    out = tmp_path / "decision.json"
    assert run_decide(evidence, frozen, frozen_sha, out) == 0
    result = json.loads(out.read_text())
    assert result["generation"]["anchor_manifest_sha256"] == MANIFEST_SHA
    for ds, d in result["decisions"].items():
        assert d["adopted_axis_center"] == "anchors" and d["frozen_N"] == 9
        assert d["retrieval"]["passes"] and d["code_to_axis"]["passes"]
        assert d["refit"] == {"stage_R": "scratch refit", "arm": "anchors", "N": 9}
        assert d["code_to_axis"]["hits_total"]["anchors"] == [[820, 2048]] * 3


def test_a_frozen_n_record_that_its_evidence_does_not_reproduce_refuses(tmp_path, ledger, capsys):
    evidence, frozen, _ = decide_world(tmp_path, ledger)
    copy = tmp_path / "tampered.json"                       # the original is written read-only
    copy.write_bytes(Path(frozen).read_bytes())
    tampered = edit_json(copy, **{"n_selected.flickr25k.anchors": 4})
    assert run_decide(evidence, str(copy), tampered, tmp_path / "d.json") == 1
    assert "not what its own evidence reduces to" in capsys.readouterr().err


def test_a_self_pinned_minimal_n_record_refuses(tmp_path, ledger, capsys):
    evidence, _, _ = decide_world(tmp_path, ledger)
    fake = tmp_path / "fake.json"
    fake.write_text(json.dumps({"artifact_kind": D.N_SELECTION_KIND, "version": M.ANCHOR_CONFIRM_VERSION,
                                "n_selected": {ds: {"none": 1, "anchors": 1} for ds in M.ANCHOR_DATASETS}}))
    assert run_decide(evidence, str(fake), sha(fake.read_bytes()), tmp_path / "d.json") == 1
    assert "not a anchor-confirm-reducer/2 N record" in capsys.readouterr().err


def test_probes_approved_for_other_records_refuse(tmp_path, ledger, capsys):
    """The probe approval covers exactly the stage-D records the reducer reads."""
    evidence, frozen, frozen_sha = decide_world(tmp_path, ledger, probe_request={"other": "records"})
    assert run_decide(evidence, frozen, frozen_sha, tmp_path / "d.json") == 1
    assert "approves probe for" in capsys.readouterr().err


def test_stage_d_records_approved_for_another_frozen_n_refuse(tmp_path, ledger, capsys):
    evidence, frozen, frozen_sha = decide_world(tmp_path, ledger, d_selection="e" * 64)
    assert run_decide(evidence, frozen, frozen_sha, tmp_path / "d.json") == 1
    assert "approves stage-D-run for" in capsys.readouterr().err


def edit_probe(evidence, index, **changes):
    e = evidence.entries[index]
    body = json.loads(Path(e["probe"]).read_text())
    body.update(changes)
    Path(e["probe"]).write_text(json.dumps(body))
    e["probe_sha256"] = sha(Path(e["probe"]).read_bytes())


@pytest.mark.parametrize("changes,reason", [
    ({"schema": "anchor-confirm-code-axis/1"}, "not a anchor-confirm-code-axis/2 probe"),
    ({"producer_sha256": "0" * 64}, "produced by another probe source"),
    ({"checkpoint_sha256": "f" * 64}, "measured another record"),
    ({"config_pt_sha256": "f" * 64}, "measured another record"),
    ({"split": {"val_split_ratio": 0.2, "val_split_seed": 42}}, "train-only validation rows"),
    ({"routing": "caption_routed"}, "train-only validation rows"),
    ({"n_images": 511, "total": 2044}, "512 images"),
    ({"total": 2047}, "512 images"),
    ({"hits": 2049}, "512 images"),
    ({"code_picks_own_axis": 0.5}, "not hits/total"),
    ({"hits": 820.0}, "not an integer"),
    ({"row_ids_sha256": None}, "no row-identity digest"),
    # audit 680.3: each reproduced mutation, refused for its own reason
    ({"row_ids_sha256": "z" * 64}, "no row-identity digest"),
    ({"coordinate": ["cifar10", "none", 9, 42]}, "measured another coordinate"),
    ({"coordinate": None}, "measured another coordinate"),
    ({"caption_target": None}, "caption target is not the record's admitted input"),
    ({"caption_target": {**D.CAPTION_TARGET, "input_seal_sha256": "e" * 64}}, "caption target"),
    ({"ties_counted_as_misses": None}, "not an integer"),
    ({"ties_counted_as_misses": 2049}, "ties do not fit"),
    ({"authority": None}, "not measured under this record's execution authority"),
    ({"hits": 2048, "ties_counted_as_misses": 0, "code_picks_own_axis": True}, "not hits/total"),
    ({"manifest_sha256": OTHER_SHA}, "produced under another generation"),
    ({"approval": None}, "names no probe approval"),
    ({"approval": {"section": 999999, "scope": "probe", "line": "x"}}, "sections numbered 999999"),
    ({"split_identity_sha256": "e" * 64}, "admitted split identity"),
    ({"split": {"val_split_ratio": 0.1, "val_split_seed": 42.0}}, "train-only validation rows"),   # 690.1
    ({"split": {"val_split_ratio": 0.1, "val_split_seed": 42, "extra": 1}}, "train-only validation rows"),
    ({"request_sha256": "e" * 64}, "approval line or request"),
])
def test_a_probe_that_is_not_this_records_contract_measurement_refuses(tmp_path, ledger, capsys, changes, reason):
    evidence, frozen, frozen_sha = decide_world(tmp_path, ledger)
    edit_probe(evidence, 0, **changes)
    assert run_decide(evidence, frozen, frozen_sha, tmp_path / "d.json") == 1
    assert reason in capsys.readouterr().err


def test_a_probe_coordinate_of_another_type_refuses(tmp_path, ledger, capsys):
    evidence, frozen, frozen_sha = decide_world(tmp_path, ledger)
    e = evidence.entries[0]
    edit_probe(evidence, 0, coordinate=[e["dataset"], e["arm"], float(e["N"]), e["seed"]])
    assert run_decide(evidence, frozen, frozen_sha, tmp_path / "d.json") == 1
    assert "measured another coordinate" in capsys.readouterr().err


def test_probes_must_share_one_row_population_per_dataset(tmp_path, ledger, capsys):
    evidence, frozen, frozen_sha = decide_world(tmp_path, ledger)
    edit_probe(evidence, 0, row_ids_sha256="a" * 64)
    assert run_decide(evidence, frozen, frozen_sha, tmp_path / "d.json") == 1
    assert "one shared row population" in capsys.readouterr().err


def test_equal_code_to_axis_keeps_the_incumbent_and_says_whether_the_old_refit_can_stand(tmp_path, ledger):
    evidence, frozen, frozen_sha = decide_world(tmp_path, ledger)
    for i in range(len(evidence.entries)):
        edit_probe(evidence, i, hits=680, code_picks_own_axis=680 / 2048)
    out = tmp_path / "decision.json"
    assert run_decide(evidence, frozen, frozen_sha, out) == 0
    for ds, d in json.loads(out.read_text())["decisions"].items():
        assert d["adopted_axis_center"] == "none" and not d["code_to_axis"]["passes"]
        # the control won at N=9 in this world, not the approved N, so the old refit cannot stand
        assert d["refit"]["stage_R"] == "scratch refit" and "cannot stand in" in d["refit"]["note"]


def test_a_kept_control_at_the_approved_n_names_the_old_refit_only_as_conditional(tmp_path, ledger,
                                                                                 monkeypatch):
    """Audit 668.1: equal N makes the old refit a candidate, never an automatic substitute."""
    approved_n = {ds: v["N"] for ds, v in INCUMBENT.items()}
    monkeypatch.setattr(sys.modules[__name__], "score_of",
                        lambda c: 0.8 if c[2] == approved_n[c[0]] else 0.5)
    evidence, frozen, frozen_sha = decide_world(tmp_path, ledger)
    for i in range(len(evidence.entries)):
        edit_probe(evidence, i, hits=680, code_picks_own_axis=680 / 2048)
    out = tmp_path / "decision.json"
    assert run_decide(evidence, frozen, frozen_sha, out) == 0
    for ds, d in json.loads(out.read_text())["decisions"].items():
        assert d["adopted_axis_center"] == "none" and d["frozen_N"] == approved_n[ds]
        assert d["refit"]["stage_R"] == "old refit only if admitted, otherwise scratch refit"
        assert d["refit"]["old_refit"]["N"] == approved_n[ds]


def test_decide_refuses_stage_s_from_another_generation(tmp_path, ledger, capsys):
    evidence, frozen, frozen_sha = decide_world(tmp_path, ledger, d_manifest=OTHER_SHA)
    assert run_decide(evidence, frozen, frozen_sha, tmp_path / "d.json", manifest=OTHER_SHA) == 1
    assert "stage S ran under another generation manifest" in capsys.readouterr().err


def test_decide_refuses_stage_d_records_from_another_generation(tmp_path, ledger, capsys):
    evidence, frozen, frozen_sha = decide_world(tmp_path, ledger, d_manifest=OTHER_SHA)
    assert run_decide(evidence, frozen, frozen_sha, tmp_path / "d.json") == 1
    assert "approves stage-D-run for" in capsys.readouterr().err


def test_retrieval_exactly_at_the_margin_passes(tmp_path, ledger, monkeypatch):
    """mean(anchors) == mean(none) - sd(none): every decide cell at 0.72 gives a zero control SD
    and equal means, so (i) passes (contract section 7.3: equality passes)."""
    monkeypatch.setattr(sys.modules[__name__], "score_of", lambda c: 0.72 if c[2] == 9 else 0.5)
    evidence, frozen, frozen_sha = decide_world(tmp_path, ledger)
    out = tmp_path / "decision.json"
    assert run_decide(evidence, frozen, frozen_sha, out) == 0
    for d in json.loads(out.read_text())["decisions"].values():
        assert d["retrieval"]["passes"] and d["retrieval"]["control_sample_sd"] == 0.0


# ---- the stage-D plan is metadata-only (audit 681) ------------------------------------------------------
def rendered_payload(dataset, n, *, namespace, stage, seed, topp, joint, epochs=None, anchor_arm=None,
                     input_authority=None):
    return R.build_payload(PARSER, render_argv((dataset, anchor_arm, n, seed)), planned_arm=anchor_arm)


@pytest.mark.parametrize("mode,rc", [(["--plan"], 0), ([], 0), (["--run"], 2),
                                     (["--smoke", "--only", "flickr25k:9:anchors:43"], 2)])
def test_the_stage_d_launcher_replays_stage_s_without_deserialising(tmp_path, ledger, monkeypatch, capsys,
                                                                     no_deserialisation, mode, rc):
    """The real call chain launcher -> verify_selection -> reduce_select, with a deserialiser
    sentinel: planning and every refusal stay metadata-only; no lease, reservation or dispatch."""
    s_world = World(tmp_path / "S", SELECT, ledger=ledger)
    n_path = tmp_path / "n.json"
    assert run_select(s_world, n_path) == 0
    calls = []
    for name in ("_with_campaign_gpu_leases", "_run_sweep", "reserve_sweep_namespace",
                 "_publish_json_exclusive", "verify_campaign_input_seals"):
        monkeypatch.setattr(M, name, lambda *a, _n=name, **k: calls.append(_n) or 0)
    monkeypatch.setattr(M, "anchor_scientific_recipe", rendered_payload)
    monkeypatch.setattr(sys, "argv", ["phase3_selection_matrix.py", "--anchor-confirm", "decide",
                                      "--namespace", "ancT", "--anchor-arms", "none,anchors", *mode,
                                      "--anchor-selection", str(n_path),
                                      "--anchor-selection-sha256", sha(n_path.read_bytes()),
                                      "--anchor-manifest", MANIFESTS[MANIFEST_SHA]["path"],
                                      "--anchor-manifest-sha256", MANIFEST_SHA])
    assert M.main() == rc
    assert calls == []
    if rc == 0:
        assert "stage decide, 12 cells" in capsys.readouterr().out


# ---- the probe's endpoint arithmetic (audit 673.2) --------------------------------------------------
def features(seed=0, b=64):
    return torch.randn(b, 4, 16, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


def test_strict_hits_count_aligned_permuted_and_tied_rows():
    t = features()
    aligned = P.strict_own_axis_hits(t.clone(), t)
    assert (aligned["hits"], aligned["total"]) == (256, 256) and aligned["ratio"] == 1.0
    assert P.strict_own_axis_hits(t[:, [1, 2, 3, 0], :], t)["hits"] == 0
    tied = t.clone()
    tied[:, 1, :] = tied[:, 0, :]                          # two captions identical -> slots 0,1 tie
    got = P.strict_own_axis_hits(t.clone()[:, [0, 0, 2, 3], :] * 0 + tied, tied)
    assert got["ties_counted_as_misses"] > 0 and got["hits"] < got["total"]


@pytest.mark.parametrize("case,reason", [
    ("three_slots", "matching nonempty"), ("shape_mismatch", "matching nonempty"),
    ("empty", "matching nonempty"), ("nan", "non-finite input"), ("inf", "non-finite input"),
    ("all_zero", "zero norm"), ("numpy", "must be tensors")])
def test_invalid_features_refuse_instead_of_scoring(case, reason):
    q, t = features(), features(1)
    if case == "three_slots":
        q, t = q[:, :3], t[:, :3]
    elif case == "shape_mismatch":
        t = t[:32]
    elif case == "empty":
        q, t = q[:0], t[:0]
    elif case == "nan":
        q[0, 0, 0] = float("nan")
    elif case == "inf":
        t[0, 0, 0] = float("inf")
    elif case == "all_zero":
        q, t = torch.zeros_like(q), torch.zeros_like(t)
    else:
        q = q.numpy()
    with pytest.raises(D.NotReducible, match=reason):
        P.strict_own_axis_hits(q, t)


# ---- the probe admits before it deserialises (audit 673.1, 679, 681) ---------------------------------------
SELECTION_N4 = {"n_selected": {ds: {"none": 4, "anchors": 4} for ds in M.ANCHOR_DATASETS},
                "record": {"path": "/n.json", "sha256": "f" * 64}, "anchor_manifest_sha256": MANIFEST_SHA}
PROBE_APPROVAL = {"section": 1, "scope": "probe", "line": "verified by main", "request_sha256": "b" * 64}


def run_probe(path, digest, c, **kw):
    return P.probe(path, digest, c, manifest=MANIFESTS[MANIFEST_SHA], approval=PROBE_APPROVAL,
                   selection=kw.pop("selection", SELECTION_N4), device="cpu", **kw)


@pytest.mark.parametrize("change,reason", [({"smoke": True}, "smoke or non-candidate"),
                                           ({"stage": "refit"}, "not a stage-1 selection record"),
                                           ({"val_split_ratio": 0.0}, "train-only split")])
def test_the_probe_refuses_forbidden_records_before_any_binary_load(world, change, reason, no_deserialisation):
    i = index_of(world, arm="anchors")
    world.rewrite_record(i, **change)
    path, digest = world.sources()
    e = world.entries[i]
    with pytest.raises(D.NotReducible, match=reason):
        run_probe(path, digest, (e["dataset"], e["arm"], e["N"], e["seed"]))


def test_the_probe_refuses_a_coordinate_outside_the_approved_selection(world, no_deserialisation):
    path, digest = world.sources()
    with pytest.raises(D.NotReducible, match="not a stage-D coordinate"):
        run_probe(path, digest, ("flickr25k", "anchors", 9, 42))


def test_the_probe_refuses_a_record_from_another_generation_before_any_binary_load(tmp_path, ledger,
                                                                                  no_deserialisation):
    world = World(tmp_path / "S", SELECT, ledger=ledger, manifest_sha=OTHER_SHA, approved_manifest=MANIFEST_SHA)
    path, digest = world.sources()
    e = world.entries[index_of(world, arm="anchors")]
    with pytest.raises(D.NotReducible, match="another generation manifest"):
        run_probe(path, digest, (e["dataset"], e["arm"], e["N"], e["seed"]))


def test_the_probe_refuses_config_bytes_off_their_pin_before_any_binary_load(world, no_deserialisation):
    i = index_of(world, arm="anchors")
    run = Path(world.entries[i]["record"]).parent
    (run / "config.pt").write_bytes((run / "config.pt").read_bytes() + b"\0")
    path, digest = world.sources()
    e = world.entries[i]
    with pytest.raises(D.NotReducible, match="config.pt is not the pinned bytes"):
        run_probe(path, digest, (e["dataset"], e["arm"], e["N"], e["seed"]))


def test_the_probe_refuses_inputs_the_trainer_admission_does_not_reproduce(world, monkeypatch):
    """After the pinned config is deserialised, the trainer's own input admission must reproduce
    the record's input authority before any model or dataset exists."""
    import model_siglip2
    monkeypatch.setattr(model_siglip2, "SigLIP2SemanticOTModel",
                        lambda *a, **k: pytest.fail("a model was constructed"))
    monkeypatch.setenv("GDNA_NUM_SEMANTIC_PARTS", "5")
    i = index_of(world, arm="anchors")
    path, digest = world.sources()
    e = world.entries[i]
    with pytest.raises(D.NotReducible, match="input admission refused"):
        run_probe(path, digest, (e["dataset"], e["arm"], e["N"], e["seed"]))


def pass_config_load(monkeypatch):
    """Let the probe deserialise its (synthetic, pinned) config and stop at model construction."""
    import model_siglip2
    monkeypatch.setattr(model_siglip2, "SigLIP2SemanticOTModel",
                        lambda *a, **k: pytest.fail("a model was constructed"))
    monkeypatch.setenv("GDNA_NUM_SEMANTIC_PARTS", "5")


def test_the_probe_refuses_inputs_other_than_the_records_admitted_ones(world, monkeypatch):
    import train_siglip2
    pass_config_load(monkeypatch)
    monkeypatch.setattr(train_siglip2, "_phase3_input_authority_from_args",
                        lambda args: dict(INPUT_AUTHORITY, seal_file_sha256="e" * 64))
    path, digest = world.sources()
    e = world.entries[index_of(world, arm="anchors")]
    with pytest.raises(D.NotReducible, match="not the record's admitted inputs"):
        run_probe(path, digest, (e["dataset"], e["arm"], e["N"], e["seed"]))


def test_matching_inputs_let_the_probe_reach_model_construction(world, monkeypatch):
    """Positive control of the input gate: the admitted inputs pass and the next boundary is the
    model, which the sentinel stops."""
    import train_siglip2
    pass_config_load(monkeypatch)
    monkeypatch.setattr(train_siglip2, "_phase3_input_authority_from_args", lambda args: dict(INPUT_AUTHORITY))
    path, digest = world.sources()
    e = world.entries[index_of(world, arm="anchors")]
    with pytest.raises(pytest.fail.Exception, match="a model was constructed"):
        run_probe(path, digest, (e["dataset"], e["arm"], e["N"], e["seed"]))


def test_the_probe_refuses_a_coordinate_its_sources_do_not_list(world, no_deserialisation):
    path, digest = world.sources([e for e in world.entries if e["arm"] == "none"])
    with pytest.raises(D.NotReducible, match="lists"):
        run_probe(path, digest, ("flickr25k", "anchors", 4, 42))


PROBE_CLI = ["--sources", "x", "--sources-sha256", "0" * 64, "--coordinate", "flickr25k:none:4:42",
             "--manifest", "m", "--manifest-sha256", MANIFEST_SHA, "--selection", "n.json",
             "--selection-sha256", "f" * 64, "--out"]


def test_probe_cli_needs_both_reuse_arguments(tmp_path, no_deserialisation):
    assert P.main([*PROBE_CLI, str(tmp_path / "o"), "--approval-section", "900",
                   "--reuse-admission", "r"]) == 1


@pytest.mark.parametrize("section,reason", [(999, "has 0 sections numbered 999"),
                                            (0, "positive ledger section number")])
def test_the_probe_cli_refuses_without_its_approval_line(tmp_path, ledger, capsys, section, reason,
                                                         no_deserialisation):
    frozen, frozen_sha, path, digest, _ = probe_cli_world(tmp_path, ledger)
    assert probe_cli(frozen, frozen_sha, path, digest, tmp_path / "o", section) == 1
    assert reason in capsys.readouterr().err
    assert not (tmp_path / "o").exists()


def probe_cli_world(tmp_path, ledger):
    """A frozen N record and a sources file listing exactly its stage-D records."""
    evidence, frozen, frozen_sha = decide_world(tmp_path, ledger)
    path, digest = evidence.sources()
    records = {(e["dataset"], e["arm"], e["N"], e["seed"]): e["record_sha256"] for e in evidence.entries}
    return frozen, frozen_sha, path, digest, records


def probe_cli(frozen, frozen_sha, path, digest, out, section, coordinate="flickr25k:none:9:43"):
    return P.main(["--sources", path, "--sources-sha256", digest, "--coordinate", coordinate,
                   "--manifest", MANIFESTS[MANIFEST_SHA]["path"], "--manifest-sha256", MANIFEST_SHA,
                   "--selection", frozen, "--selection-sha256", frozen_sha, "--out", str(out),
                   "--approval-section", str(section)])


@pytest.mark.parametrize("pins", [{"selection": "e" * 64}, {"request": "e" * 64}, {"manifest": OTHER_SHA}])
def test_the_probe_cli_refuses_an_approval_of_another_operation(tmp_path, ledger, capsys, no_deserialisation,
                                                                 pins):
    frozen, frozen_sha, path, digest, records = probe_cli_world(tmp_path, ledger)
    good = {"manifest": MANIFEST_SHA, "selection": frozen_sha,
            "request": M._json_digest(D.probe_request(MANIFEST_SHA, frozen_sha, records))}
    approval = ledger.approve("probe", **{**good, **pins})
    assert probe_cli(frozen, frozen_sha, path, digest, tmp_path / "o", approval["section"]) == 1
    assert "approves probe for" in capsys.readouterr().err


def test_the_probe_cli_reaches_its_first_load_only_under_its_exact_approval(tmp_path, ledger, capsys,
                                                                             monkeypatch):
    """Positive control: the approved probe request passes every metadata gate and stops at the
    first deserialisation (a sentinel), never at a refusal."""
    frozen, frozen_sha, path, digest, records = probe_cli_world(tmp_path, ledger)
    approval = ledger.approve("probe", manifest=MANIFEST_SHA, selection=frozen_sha,
                              request=M._json_digest(D.probe_request(MANIFEST_SHA, frozen_sha, records)))
    monkeypatch.setenv("GDNA_NUM_SEMANTIC_PARTS", "5")
    class Reached(Exception):
        pass
    def sentinel(*a, **k):
        raise Reached()
    monkeypatch.setattr(torch, "load", sentinel)
    with pytest.raises(Reached):
        probe_cli(frozen, frozen_sha, path, digest, tmp_path / "o", approval["section"])


def test_the_probe_cli_refuses_sources_that_are_not_the_stage_d_records(tmp_path, ledger, capsys,
                                                                         no_deserialisation):
    frozen, frozen_sha, path, digest, records = probe_cli_world(tmp_path, ledger)
    body = json.loads(Path(path).read_text())
    body["coordinates"] = body["coordinates"][:-1]
    Path(path).write_text(json.dumps(body))
    assert probe_cli(frozen, frozen_sha, path, sha(Path(path).read_bytes()), tmp_path / "o", 1) == 1
    assert "does not list exactly the stage-D coordinates" in capsys.readouterr().err
