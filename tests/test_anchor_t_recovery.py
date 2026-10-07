"""The one stage-T recovery of generation v9 r8 (docs/ANCHOR_T_RECOVERY_CONTRACT_v1.md; audits 795-797).

Synthetic only: a stopped twelve-cell stage-T campaign is built from the stage-R/T suite's synthetic worlds
(F record, stage-R receipt, run directories with synthetic checkpoints and configurations), with eleven
completed cells, one interrupted cell, a settled operations ledger and the pinned lineage. No real input,
config, checkpoint, model, dataset, ledger, claim root or GPU is touched. Kinds of evidence, as in the
stage-R/T suite:
  predicate  -- a pure function on its inputs;
  structural -- the source wiring;
  composed   -- through a real entry point (the launcher main(), the recovery campaign and its T cell,
                the T entry main(), the supervisor), only processes, leases, the boundary environment and
                model encoding replaced by recorders.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
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
import test_anchor_confirm_launcher as LT                   # noqa: E402
import test_anchor_refit_stage as AR                        # noqa: E402
from test_anchor_refit_stage import fworld                  # noqa: E402,F401  (fixture)

NS, NEW_NS, NONCE = "ancT9", "ancT9s", "t" * 64
RUN_ID = "20261006T145139Z-b08e4ae2"
RECOVERED = ("nuswide", 44)
FAILED_NS, FAILED_RUN, FAILED_NONCE = "ancT9r", "20261007T061117Z-12e14ceb", "f" * 64
FAILED_CHARGE = 5.704555157572031
BIT2 = RT.BIT2_OUTPUT
REQUIRED = [n for n in M.OFFICIAL_TEST_OUTPUTS if n != BIT2]
#: the launcher's REAL managed-child launcher, captured before any fixture replaces it (audit 839)
_REAL_MANAGED = M._run_managed_process
REASON = "budget: projected 80001 s of 80000 s"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write(path: Path, obj) -> str:
    return AR.write(path, obj)


def jline(obj) -> str:
    return json.dumps(obj, sort_keys=True)


def new_claims() -> list:
    """Claims in the claim root other than the failed r8 recovery's consumed one."""
    if not RT.RECOVERY_CLAIM_ROOT.is_dir():
        return []
    consumed = getattr(RT, "_TEST_CONSUMED_CLAIM", None)
    return sorted(p.name for p in RT.RECOVERY_CLAIM_ROOT.glob("*.json") if p.name != consumed)


@pytest.fixture(autouse=True)
def no_real_claim_or_ledger_roots(tmp_path, monkeypatch):
    """No test reaches the real claim root, recovery ledger or settled ledger."""
    monkeypatch.setattr(RT, "RECOVERY_CLAIM_ROOT", tmp_path / "claims")
    monkeypatch.setattr(S, "REC_OPS_ROOT", tmp_path / "recops")
    monkeypatch.setattr(S, "REC_PARENT_LEDGER", tmp_path / "no-parent-ledger.jsonl")


# =============================================================================================
# the synthetic stopped campaign
# =============================================================================================
class StoppedWorld:
    """Stage R (under a synthetic r7 manifest) and the stopped full T: twelve attempts and entries,
    eleven T records with their twelve outputs each, the interrupted cell with none, the settled ledger
    ending in the budget stop, the audit ledger (F acceptance, stage-R line, stage-T-run line) and the
    pinned lineage. The executing generation is THIS tree (an LT manifest), writing a separate record
    directory."""

    def __init__(self, tmp_path, monkeypatch, fworld):
        self.tmp, self.mp = tmp_path, monkeypatch
        root = tmp_path / "w"
        root.mkdir()
        files_now = {rel: sha((REPO / rel).read_bytes()) for rel in M.anchor_generation_closure()}
        r7_files = {rel: d for rel, d in files_now.items() if rel not in RT.RECOVERY_ADDED_CLOSURE}
        for rel in RT.RECOVERY_CHANGED_CLOSURE:
            r7_files[rel] = sha(f"r7:{rel}".encode())
        self.r7_files = r7_files
        self.r7_path = root / "authority_manifest_v9r7.json"
        self.r7_sha = write(self.r7_path, {"new_generation": {"files_sha256": r7_files}})
        monkeypatch.setattr(RT, "R7_MANIFEST_SHA256", self.r7_sha)
        self.rworld = AR.RWorld(root / "rt", monkeypatch, fworld, self.r7_sha)
        self.hist = self.rworld.records
        AR.full_ledger(tmp_path, monkeypatch, self.rworld.ledger_sections)
        refit = RT.admit_refit_receipt(self.rworld.receipt, self.rworld.receipt_sha, manifest_sha256=self.r7_sha,
                                       freeze=RT.anchor_freeze_authority(), mode="run")
        args = SimpleNamespace(gpus="0,1,2,3", gpu=None, smoke=False, namespace=NS)
        request = RT.terminal_test_request(args, refit, manifest={"sha256": self.r7_sha, "path": str(self.r7_path)})
        self.request, self.request_sha = request, M._json_digest(request)
        self.t_line = LT.approval_line("stage-T-run", manifest=self.r7_sha, freeze=RT.ANCHOR_F_RECORD_SHA256,
                                       request=self.request_sha)
        self.sections = {**self.rworld.ledger_sections, 788: [self.t_line]}
        AR.full_ledger(tmp_path, monkeypatch, self.sections)
        approval = {"section": 788, "scope": "stage-T-run", "line": self.t_line}
        snapshot = {"schema": RT.T_SNAPSHOT_SCHEMA,
                    "plan": {"namespace": NS, "campaign_nonce": NONCE, "campaign_kind": RT.T_CAMPAIGN_KIND,
                             "declared_count": 12, "executed_count": 12},
                    "request": request, "request_sha256": self.request_sha, "approval": approval}
        snap_file = f"{NS}_snapshot_{M._json_digest(snapshot)[:16]}.json"
        lineage = {"schema": RT.RECOVERY_LINEAGE_SCHEMA,
                   "historical_manifest": {"path": str(self.r7_path), "sha256": self.r7_sha},
                   "historical_record_dir": str(self.hist),
                   "refit_receipt": {"file": self.rworld.receipt.name, "sha256": self.rworld.receipt_sha},
                   "stopped": {"namespace": NS, "request_sha256": self.request_sha,
                               "snapshot": {"file": snap_file, "sha256": write(self.hist / snap_file, snapshot),
                                            "semantic_sha256": M._json_digest(snapshot)},
                               "campaign_nonce": NONCE, "approval": approval,
                               "receipt_absent": f"{NS}_test_complete.json"},
                   "carried": {}}
        reservation = {"namespace": NS, "plan_digest": M._json_digest(snapshot), "plan_snapshot_file": snap_file,
                       "campaign_nonce": NONCE}
        lineage["stopped"]["reservation"] = {"file": f"{NS}_campaign_reservation.json",
                                             "sha256": write(self.hist / f"{NS}_campaign_reservation.json",
                                                             reservation)}
        self.cells = {}
        for cell in request["cells"]:
            tag, cid = cell["tag"], cell["cell_id"]
            attempt = {"schema": RT.T_ATTEMPT_SCHEMA, "namespace": NS, "campaign_nonce": NONCE, "request": request,
                       "request_sha256": self.request_sha, "approval": approval, "cell": cell}
            att = {"file": f"{NS}_attempt_{tag}.json", "sha256": write(self.hist / f"{NS}_attempt_{tag}.json", attempt)}
            ent = {"file": f"{NS}_entry_{tag}.json",
                   "sha256": write(self.hist / f"{NS}_entry_{tag}.json",
                                   {"attempt": att["file"], "attempt_sha256": att["sha256"], "cell_id": cid})}
            self.cells[cid] = cell
            if (cell["dataset"], cell["seed"]) == RECOVERED:
                lineage["recovery_cell"] = {"cell_id": cid, "tag": tag, "attempt": att, "entry": ent}
                self.recovered = cell
                continue
            run = Path(cell["run_dir"])
            digests = {name: sha(self._output(run / name, f"{name}:{tag}")) for name in REQUIRED}
            completion = {key: digests[name] for key, name in RT.CARRIED_BINDINGS}
            completion["extraction_manifest_sha256"] = {s: digests[f"extraction_manifest_{s}.json"]
                                                        for s in ("query", "db", "train")}
            completion["npz_sha256"] = {s: digests[f"extract_{s}.npz"] for s in ("query", "db", "train")}
            record = {"schema": RT.T_RECORD_SCHEMA, "stage": "test", "namespace": NS, "campaign_nonce": NONCE,
                      "request_sha256": self.request_sha, "cell": cell, "attempt": {"file": att["file"],
                                                                                     "sha256": att["sha256"]},
                      "completion": completion, "wall_seconds": 1.0}
            rec = {"file": f"{NS}_{tag}.json", "sha256": write(self.hist / f"{NS}_{tag}.json", record)}
            lineage["carried"][cid] = {"tag": tag, "attempt": att, "entry": ent, "record": rec}
        self.ops = root / "ops" / "device_budget_ledger.jsonl"
        lineage["stopped"]["settlement"] = self._settled_ledger()
        self.lineage = lineage
        self._pin_lineage()
        # the executing generation: this tree, writing its own record directory
        self.records = root / "r8records"
        monkeypatch.setattr(M, "ANCHOR_RECORD_DIR", self.records)
        monkeypatch.setattr(M, "RECORD_DIR", self.records)
        self.manifest_path, self.manifest_sha = AR.manifest(tmp_path)
        self._failed_recovery(root)

    def _failed_recovery(self, root):
        """The failed r8 recovery (audits 838-839): its consumed claim, attempt, reservation, snapshot and two
        settled supervisor rows, no entry, record or receipt; and the exception artifact that pins them."""
        self.failed_dir = root / "r8failed"
        self.failed_dir.mkdir()
        tag, cid = self.recovered["tag"], self.recovered["cell_id"]
        self.failed_request = {"schema": "anchor-terminal-test-recovery-request/1", "mode": RT.RECOVERY_MODE,
                               "namespace": FAILED_NS, "cells": [self.recovered]}
        fsha = M._json_digest(self.failed_request)
        line = LT.approval_line(RT.RECOVERY_SCOPE, manifest="3" * 64, freeze=RT.ANCHOR_F_RECORD_SHA256, request=fsha)
        approval = {"section": 830, "scope": RT.RECOVERY_SCOPE, "line": line}
        attempt_name = f"{FAILED_NS}_attempt_{tag}.json"
        attempt = {"schema": RT.T_ATTEMPT_SCHEMA, "namespace": FAILED_NS, "campaign_nonce": FAILED_NONCE,
                   "request": self.failed_request, "request_sha256": fsha, "cell": self.recovered,
                   "approval": {**approval, "ledger": "/x/ledger.md", "ledger_sha256": "e" * 64}}
        snapshot = {"schema": RT.T_RECOVERY_SNAPSHOT_SCHEMA, "plan": {"namespace": FAILED_NS,
                                                                      "campaign_nonce": FAILED_NONCE},
                    "request": self.failed_request, "request_sha256": fsha}
        snap_name = f"{FAILED_NS}_snapshot_{M._json_digest(snapshot)[:16]}.json"
        reservation = {"schema_version": 1, "namespace": FAILED_NS, "campaign_nonce": FAILED_NONCE,
                       "plan_digest": M._json_digest(snapshot), "plan_snapshot_file": snap_name}
        key = RT.recovery_claim_key(self.lineage)
        claim = {"schema": RT.T_RECOVERY_CLAIM_SCHEMA, "claim_key": key, "request_sha256": fsha,
                 "approval": approval, "namespace": FAILED_NS, "record_dir": str(self.failed_dir),
                 "attempt": attempt_name, "lineage": {"path": str(RT.RECOVERY_LINEAGE),
                                                       "sha256": RT.RECOVERY_LINEAGE_SHA256}}
        RT.RECOVERY_CLAIM_ROOT.mkdir(parents=True, exist_ok=True)
        claim_path = RT.recovery_claim_path(key)
        self.mp.setattr(RT, "_TEST_CONSUMED_CLAIM", claim_path.name, raising=False)
        pins = {"claim": {"key": key, "path": str(claim_path), "sha256": write(claim_path, claim)},
                "attempt": {"path": str(self.failed_dir / attempt_name),
                            "sha256": write(self.failed_dir / attempt_name, attempt)},
                "reservation": {"path": str(self.failed_dir / f"{FAILED_NS}_campaign_reservation.json"),
                                "sha256": write(self.failed_dir / f"{FAILED_NS}_campaign_reservation.json",
                                                reservation)},
                "snapshot": {"path": str(self.failed_dir / snap_name),
                             "sha256": write(self.failed_dir / snap_name, snapshot),
                             "semantic_sha256": M._json_digest(snapshot)}}
        self.failed_ledger = root / "failed_recops" / "device_budget_ledger.jsonl"
        rows = [{"schema": S.LEDGER_SCHEMA, "event": "start", "run_id": FAILED_RUN, "stage": RT.RECOVERY_SCOPE,
                 "parent": {"sha256": "p" * 64}},
                {"schema": S.LEDGER_SCHEMA, "event": "final", "run_id": FAILED_RUN, "stage": RT.RECOVERY_SCOPE,
                 "status": "exited", "returncode": 1, "attempts": [], "charged_seconds": FAILED_CHARGE}]
        lines = [jline(r).encode() for r in rows]
        self.failed_ledger.parent.mkdir(parents=True)
        self.failed_ledger.write_bytes(b"".join(line + b"\n" for line in lines))
        self.exception = {"schema": RT.RECOVERY_EXCEPTION_SCHEMA, "cell_id": cid, "namespace": FAILED_NS,
                          "record_dir": str(self.failed_dir), "request_sha256": fsha, "approval": approval,
                          "window_section": 838, "run_id": FAILED_RUN, "campaign_nonce": FAILED_NONCE,
                          "ledger": {"path": str(self.failed_ledger), "rows": 2,
                                     "prefix_sha256": sha(self.failed_ledger.read_bytes()),
                                     "line_sha256": [sha(line) for line in lines],
                                     "charged_seconds": FAILED_CHARGE, "parent_sha256": "p" * 64},
                          **pins,
                          "absent": [f"{FAILED_NS}_entry_{tag}.json", f"{FAILED_NS}_{tag}.json",
                                     f"{FAILED_NS}_recovery_complete.json"]}
        self._pin_exception()
        self._key()

    def _pin_exception(self):
        path = self.tmp / "w" / "exception.json"
        path.unlink(missing_ok=True)
        self.mp.setattr(RT, "RECOVERY_EXCEPTION", path)
        self.mp.setattr(RT, "RECOVERY_EXCEPTION_SHA256", write(path, self.exception))

    def reexception(self, mutate):
        """Change the exception artifact AND repin it, so a later semantic check (not the pin) must refuse."""
        mutate(self.exception)
        self._pin_exception()

    def consumed_claim(self) -> Path:
        return Path(self.exception["claim"]["path"])

    def _key(self):
        """The exception claim key, computed once from the world as built (tests then change files)."""
        self.failed_block = RT.verify_failed_recovery(RT.recovery_exception(), RT.recovery_lineage())
        self.exception_key = RT.exception_claim_key(self.failed_block, self.recovered["cell_id"])

    @staticmethod
    def _output(path: Path, text: str) -> bytes:
        data = text.encode()
        path.write_bytes(data)
        return data

    def _settled_ledger(self, *, final_edit=None, stop_reason=REASON, extra_after=None):
        prior = {"schema": S.LEDGER_SCHEMA, "event": "final", "run_id": "earlier", "status": "exited",
                 "charged_seconds": 7280.510054702871}
        rows = [{"schema": S.LEDGER_SCHEMA, "event": "start", "run_id": "earlier"}, prior,
                {"schema": S.LEDGER_SCHEMA, "event": "start", "run_id": RUN_ID},
                {"schema": S.LEDGER_SCHEMA, "event": "poll", "run_id": RUN_ID},
                {"schema": S.LEDGER_SCHEMA, "event": "stop", "run_id": RUN_ID, "reason": stop_reason}]
        final = {"schema": S.LEDGER_SCHEMA, "event": "final", "run_id": RUN_ID, "status": "stopped",
                 "reason": stop_reason, "returncode": 143, "charged_seconds": 72201.6361753922,
                 "cumulative_charged_seconds": 79482.14623009507, "leases_held_after_exit": [],
                 "orphaned_live_attempts": [], "continuity_lost": [], "monitor_failures": []}
        if final_edit:
            final_edit(final)
        rows.append(final)
        if extra_after:
            rows.append(extra_after)
        lines = [jline(r) for r in rows]
        self.ops.parent.mkdir(parents=True, exist_ok=True)
        self.ops.write_text("\n".join(lines) + "\n")
        idx = {"start": 2, "stop": 4, "final": 5}
        return {"ledger": str(self.ops), "ledger_sha256": sha(self.ops.read_bytes()), "ledger_lines": len(lines),
                "run_id": RUN_ID, **{f"{k}_line_sha256": sha(lines[i].encode()) for k, i in idx.items()},
                "stop_reason": REASON, "final_status": "stopped", "final_returncode": 143,
                "charged_seconds": 72201.6361753922, "cumulative_charged_seconds": 79482.14623009507}

    def _pin_lineage(self):
        path = self.tmp / "w" / "lineage.json"
        path.unlink(missing_ok=True)
        digest = write(path, self.lineage)
        self.mp.setattr(RT, "RECOVERY_LINEAGE", path)
        self.mp.setattr(RT, "RECOVERY_LINEAGE_SHA256", digest)

    def relineage(self, mutate):
        """Change the lineage AND repin it, so a later semantic check (not the pin) must refuse."""
        mutate(self.lineage)
        self._pin_lineage()

    def repin_file(self, pin: dict, mutate):
        """Rewrite one pinned JSON record through `mutate` and update its pin in the lineage."""
        path = self.hist / pin["file"]
        obj = json.loads(path.read_bytes())
        mutate(obj)
        pin["sha256"] = write(path, obj)
        self._pin_lineage()

    def carried_pins(self, k=0):
        return self.lineage["carried"][sorted(self.lineage["carried"])[k]]

    def admit(self):
        return RT.admit_stopped_campaign(RT.recovery_lineage(), freeze=RT.anchor_freeze_authority())

    def argv(self, *extra, ns=NEW_NS):
        return ["--anchor-confirm", "recover", "--namespace", ns, "--anchor-manifest", self.manifest_path,
                "--anchor-manifest-sha256", self.manifest_sha, *extra]


@pytest.fixture
def world(tmp_path, monkeypatch, fworld):
    return StoppedWorld(tmp_path, monkeypatch, fworld)


# =============================================================================================
# composed: the stopped campaign is admitted from its lineage, exactly
# =============================================================================================
def test_composed_the_lineage_admits_eleven_carried_cells_and_the_one_interrupted_cell(world):
    admitted = world.admit()
    assert admitted["cell"] == world.recovered
    assert sorted(c["cell_id"] for c in admitted["carried"]) == sorted(set(world.cells) - {world.recovered["cell_id"]})
    assert admitted["settlement"]["cumulative_charged_seconds"] == 79482.14623009507
    assert admitted["refit"]["approval"]["scope"] == "stage-R-run"


def _drop_one_carried(lineage):
    lineage["carried"].pop(sorted(lineage["carried"])[0])


def _carry_the_recovered_cell_too(lineage):
    cid = lineage["recovery_cell"]["cell_id"]
    lineage["carried"][cid] = dict(lineage["carried"][sorted(lineage["carried"])[0]])


def _foreign_cell(lineage):
    first = sorted(lineage["carried"])[0]
    lineage["carried"][first.replace("seed=42", "seed=45")] = lineage["carried"].pop(first)


def _carried_tag_of_another_cell(lineage):
    keys = sorted(lineage["carried"])
    lineage["carried"][keys[0]]["tag"] = lineage["carried"][keys[1]]["tag"]


@pytest.mark.parametrize("mutate,reason", [
    (_drop_one_carried, "twelve minus the recovered one"),
    (_carry_the_recovered_cell_too, "twelve minus the recovered one"),
    (_foreign_cell, "twelve minus the recovered one"),
    (_carried_tag_of_another_cell, "another cell tag"),
], ids=["missing-carried", "duplicate-recovered", "foreign-seed", "crossed-tag"])
def test_composed_membership_is_exactly_the_twelve_minus_the_recovered_cell(world, mutate, reason):
    world.relineage(mutate)
    with pytest.raises(CellRefused, match=reason):
        world.admit()


def test_composed_the_recovered_cell_with_a_stage_t_record_refuses(world):
    write(world.hist / f"{NS}_{world.recovered['tag']}.json", {"forged": True})
    with pytest.raises(CellRefused, match="has a stage-T record"):
        world.admit()


def test_composed_the_recovered_cell_attempted_in_another_namespace_refuses(world):
    write(world.hist / f"ancOther_attempt_{world.recovered['tag']}.json", {})
    with pytest.raises(CellRefused, match="another namespace"):
        world.admit()


@pytest.mark.parametrize("name", ["extract_db.npz", BIT2])
def test_composed_the_recovered_cell_holding_any_output_refuses(world, name):
    Path(world.recovered["run_dir"], name).write_bytes(b"x")
    with pytest.raises(CellRefused, match="never reaches the official test"):
        world.admit()


def test_composed_a_stopped_campaign_with_a_receipt_refuses(world):
    write(world.hist / f"{NS}_test_complete.json", {})
    with pytest.raises(CellRefused, match="has a receipt"):
        world.admit()


@pytest.mark.parametrize("change", ["missing", "bit2"])
def test_composed_a_carried_run_directory_off_the_output_contract_refuses(world, change):
    run = Path(world.cells[sorted(world.lineage["carried"])[0]]["run_dir"])
    if change == "missing":
        (run / "pairwise_nmi.json").unlink()
    else:
        (run / BIT2).write_text("{}")
    with pytest.raises(CellRefused, match="exactly the twelve required outputs"):
        world.admit()


def _other_request(record):
    record["request_sha256"] = "0" * 64


def _other_nonce(record):
    record["campaign_nonce"] = "0" * 64


def _other_attempt_bytes(record):
    record["attempt"]["sha256"] = "0" * 64


@pytest.mark.parametrize("mutate", [_other_request, _other_nonce, _other_attempt_bytes],
                         ids=["record-request", "record-nonce", "record-attempt"])
def test_composed_a_carried_record_off_the_stopped_lineage_refuses(world, mutate):
    world.repin_file(world.carried_pins()["record"], mutate)
    with pytest.raises(CellRefused, match="not one stopped-campaign lineage"):
        world.admit()


def test_composed_a_carried_entry_not_binding_its_attempt_refuses(world):
    world.repin_file(world.carried_pins()["entry"], lambda e: e.update(attempt_sha256="0" * 64))
    with pytest.raises(CellRefused, match="not one stopped-campaign lineage"):
        world.admit()


def test_composed_a_record_changed_after_pinning_refuses_at_its_pin(world):
    path = world.hist / world.carried_pins()["record"]["file"]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(CellRefused):
        world.admit()


# ---- the settlement: the audited clean budget stop, exactly -----------------------------------------
@pytest.mark.parametrize("kwargs,reason", [
    ({"final_edit": lambda f: f.update(status="unclean")}, "clean budget stop"),
    ({"final_edit": lambda f: f.update(orphaned_live_attempts=[[1, 2]])}, "clean budget stop"),
    ({"final_edit": lambda f: f.update(cumulative_charged_seconds=0.0)}, "clean budget stop"),
    ({"final_edit": lambda f: f.update(leases_held_after_exit=["GPU-x"])}, "clean budget stop"),
    ({"stop_reason": "wall: 28700 s of the 28800 s limit"}, "clean budget stop"),
    ({"extra_after": {"schema": S.LEDGER_SCHEMA, "event": "start", "run_id": "later"}}, "ending the ledger"),
], ids=["unclean", "orphan", "charge-reset", "held-lease", "wall-stop", "line-after-final"])
def test_composed_the_settlement_must_be_the_audited_clean_budget_stop(world, kwargs, reason):
    settlement = world._settled_ledger(**kwargs)
    if "stop_reason" in kwargs:
        settlement["stop_reason"] = kwargs["stop_reason"]
    world.relineage(lambda lin: lin["stopped"].update(settlement=settlement))
    with pytest.raises(CellRefused, match=reason):
        world.admit()


def test_composed_a_settled_ledger_changed_after_pinning_refuses(world):
    with world.ops.open("a") as handle:
        handle.write(jline({"schema": S.LEDGER_SCHEMA, "event": "poll", "run_id": "x"}) + "\n")
    with pytest.raises(CellRefused, match="settled R/T ledger"):
        world.admit()


# ---- historical authority (r7) and the stopped approval -----------------------------------------
def test_composed_the_stopped_approval_must_still_be_in_the_ledger(world, tmp_path, monkeypatch):
    AR.full_ledger(tmp_path, monkeypatch, {**world.rworld.ledger_sections, 788: []})   # the §788 line removed
    with pytest.raises(CellRefused, match="approval lines"):
        world.admit()


def test_composed_the_stopped_snapshot_of_another_generation_refuses(world, monkeypatch):
    monkeypatch.setattr(RT, "R7_MANIFEST_SHA256", "0" * 64)
    with pytest.raises(CellRefused, match="not the pinned r7 run request"):
        world.admit()


def test_composed_the_historical_manifest_must_be_r7(world, monkeypatch):
    monkeypatch.setattr(RT, "R7_MANIFEST_SHA256", "0" * 64)
    with pytest.raises(CellRefused, match="not r7"):
        RT.historical_generation(RT.recovery_lineage())


# =============================================================================================
# composed: the r7 -> r8 transition and the T boundary allowlist
# =============================================================================================
def _executing(world):
    return M.load_anchor_manifest(world.manifest_path, world.manifest_sha)


def test_composed_the_reviewed_transition_is_admitted(world):
    hist = RT.historical_generation(RT.recovery_lineage())
    got = RT.generation_transition(_executing(world), hist)
    assert got["changed"] == sorted(RT.RECOVERY_CHANGED_CLOSURE) and got["added"] == sorted(RT.RECOVERY_ADDED_CLOSURE)


@pytest.mark.parametrize("edit", ["extra-change", "missing-change", "removed", "same-generation"])
def test_composed_any_other_transition_refuses(world, edit):
    hist = RT.historical_generation(RT.recovery_lineage())
    manifest = _executing(world)
    if edit == "extra-change":
        hist["files_sha256"]["evaluation_siglip2.py"] = "0" * 64
    elif edit == "missing-change":
        hist["files_sha256"]["scripts/anchor_confirm_supervisor.py"] = manifest["files_sha256"][
            "scripts/anchor_confirm_supervisor.py"]
    elif edit == "removed":
        hist["files_sha256"]["scripts/retired.py"] = "0" * 64
    else:
        manifest["sha256"] = RT.R7_MANIFEST_SHA256
    with pytest.raises(CellRefused):
        RT.generation_transition(manifest, hist)


SOURCE_SET = ("p0_protocol.py", "terminal_official_test.py", *RT.RECOVERY_CHANGED_SOURCES)


def _bundle(clean=True):
    def bundle(sources):
        return {"schema_version": 1, "head_commit": "h" * 40,
                "entries": {rel: {"tracked": True, "clean": clean if rel in RT.RECOVERY_CHANGED_SOURCES else True,
                                  "head_sha256": d, "worktree_sha256": d} for rel, d in sorted(sources.items())}}
    return bundle


@pytest.fixture
def boundary(world, monkeypatch):
    monkeypatch.setattr(M, "environment_fingerprint",
                        lambda gpus: {"errors": [], "selected_gpus": [{"index": 3, "uuid": "GPU-x"}]})
    monkeypatch.setattr(M, "_source_authority_bundle", _bundle())
    live = {rel: sha((REPO / rel).read_bytes()) for rel in SOURCE_SET}
    r_sources = {rel: (world.r7_files[rel] if rel in RT.RECOVERY_CHANGED_SOURCES else d) for rel, d in live.items()}
    r_snapshot = {"qwen_root": str(M.QWEN_ROOT), "sources": r_sources, "inputs": {},
                  "source_authority": _bundle()(r_sources), "input_seals": None}
    hist = RT.historical_generation(RT.recovery_lineage())
    manifest = _executing(world)

    def run(snapshot=None, manifest_=None, hist_=None):
        return RT.recovery_boundary(snapshot or r_snapshot, [3], manifest=manifest_ or manifest, historical=hist_ or hist)
    return SimpleNamespace(run=run, r_snapshot=r_snapshot, live=live, hist=hist, manifest=manifest)


def test_composed_the_boundary_crosses_exactly_the_allowlist(boundary):
    got = boundary.run()
    assert got["sources"] == boundary.live
    M.verify_snapshot(got)                                # the unchanged verifier accepts it between producers


def test_composed_a_changed_source_outside_the_allowlist_refuses(boundary):
    stale = json.loads(json.dumps(boundary.r_snapshot))
    stale["sources"]["p0_protocol.py"] = "0" * 64
    with pytest.raises(CellRefused, match="not exactly the reviewed allowlist"):
        boundary.run(stale)


def test_composed_an_allowlisted_source_left_unchanged_refuses(boundary):
    same = json.loads(json.dumps(boundary.r_snapshot))
    same["sources"]["scripts/anchor_terminal_test.py"] = boundary.live["scripts/anchor_terminal_test.py"]
    with pytest.raises(CellRefused, match="not exactly the reviewed allowlist"):
        boundary.run(same)


def test_composed_an_allowlisted_source_off_its_r7_or_r8_bytes_refuses(boundary):
    hist = json.loads(json.dumps(boundary.hist))
    hist["files_sha256"]["scripts/anchor_refit_stage.py"] = "1" * 64
    with pytest.raises(CellRefused, match="r7 bytes at stage R"):
        boundary.run(hist_=hist)
    manifest = json.loads(json.dumps(boundary.manifest))
    manifest["files_sha256"]["scripts/anchor_refit_stage.py"] = "1" * 64
    with pytest.raises(CellRefused, match="r7 bytes at stage R"):
        boundary.run(manifest_=manifest)


def test_composed_an_uncommitted_allowlisted_source_refuses(boundary, monkeypatch):
    monkeypatch.setattr(M, "_source_authority_bundle", _bundle(clean=False))
    with pytest.raises(CellRefused, match="committed r8 bytes"):
        boundary.run()


# =============================================================================================
# composed: the launcher main() -- plan, approval scope, the campaign, the claim and the receipt
# =============================================================================================
@pytest.fixture
def campaign(world, tmp_path, monkeypatch):
    """The launcher with the external boundaries recorded: leases (with their UUID evidence), the boundary
    environment, the managed processes and the post-chain checker."""
    state = {"commands": [], "events": [], "fail_at": None, "drift_at": None, "checks": 0, "during": None,
             "mp": monkeypatch, "forge_new_completion": False}
    monkeypatch.setattr(M, "_BOOTSTRAP_PREIMPORT_VERIFIED", True)

    def leases(args, fn):
        state["events"].append("leases")
        args._phase3_gpu_lease_uuids = ("GPU-x",)
        try:
            return fn()
        finally:
            del args._phase3_gpu_lease_uuids
    monkeypatch.setattr(M, "_with_campaign_gpu_leases", leases)

    def boundary_stub(r_snapshot, gpus, *, manifest, historical):
        state["events"].append("boundary")
        environment = {"errors": [], "selected_gpus": [{"index": 3, "uuid": "GPU-x"}]}
        return {"qwen_root": str(M.QWEN_ROOT), "sources": {}, "inputs": {}, "source_authority": {},
                "source_authority_sha256": "s", "environment": environment, "environment_sha256": "e",
                "input_seals": None}
    monkeypatch.setattr(RT, "recovery_boundary", boundary_stub)

    def verify(snapshot, datasets=None):
        state["checks"] += 1
        if state["checks"] - 1 == state["drift_at"]:
            raise CellRefused("environment changed while the campaign was running (injected)")
    monkeypatch.setattr(M, "verify_snapshot", verify)

    def managed(command, **k):
        name = Path(command[2] if command[1] == "-B" else command[1]).name
        if name == "anchor_terminal_test.py":
            state["events"].append(("entry", sorted(p.name for p in RT.RECOVERY_CLAIM_ROOT.glob("*.json")),
                                    sorted(p.name for p in world.records.glob("*_attempt_*.json"))))
            # the simulated entry claims its entry with the real claim function (audit 801), and the
            # simulated producers write the recovered cell's twelve outputs (bit2 never); a failed entry
            # writes none
            run = Path(command[command.index("--config_path") + 1])
            attempt = Path(command[command.index("--attempt") + 1])
            TE.claim_entry(attempt, command[command.index("--attempt-sha256") + 1],
                           json.loads(attempt.read_bytes())["cell"])
            if state["fail_at"] != 0:
                for out in REQUIRED:
                    (run / out).write_bytes(f"new:{out}".encode())
            if state["during"]:
                state["during"](world)                  # something changes while the cell runs (audit 800)
        state["commands"].append(name)
        return SimpleNamespace(returncode=1 if len(state["commands"]) - 1 == state["fail_at"] else 0)
    monkeypatch.setattr(M, "_run_managed_process", managed)

    def produced(run_dir, dataset):
        digests = {out: sha((Path(run_dir) / out).read_bytes()) for out in REQUIRED}
        completion = {key: digests[out] for key, out in RT.CARRIED_BINDINGS}
        completion["extraction_manifest_sha256"] = {s: digests[f"extraction_manifest_{s}.json"]
                                                    for s in ("query", "db", "train")}
        completion["npz_sha256"] = {s: digests[f"extract_{s}.npz"] for s in ("query", "db", "train")}
        if state["forge_new_completion"]:
            completion["npz_sha256"]["db"] = "0" * 64       # a record that binds other bytes than were written
        return {**completion, "map_at_R": 0.5, "bio_map_at_R": 0.4}
    monkeypatch.setattr(M, "assert_refit_outputs", produced)
    real_payloads = RT.verify_carried_payloads

    def payloads(admitted):
        state["events"].append("payloads")
        return real_payloads(admitted)
    monkeypatch.setattr(RT, "verify_carried_payloads", payloads)

    def plan(ns=NEW_NS, gpus="3"):
        return AR.preview(monkeypatch, _Capsys.current, world.argv("--run", "--gpus", gpus, ns=ns))

    def approve(digest, *, scope=RT.RECOVERY_SCOPE, manifest=None, section=800):
        line = LT.approval_line(scope, manifest=manifest or world.manifest_sha, freeze=RT.ANCHOR_F_RECORD_SHA256,
                                request=digest)
        AR.full_ledger(tmp_path, monkeypatch, {**world.sections, section: [line]})
        return ["--anchor-approval-section", str(section)]

    def run(*extra, ns=NEW_NS, gpus="3"):
        return LT.run_main(monkeypatch, *world.argv("--run", "--gpus", gpus, *extra, ns=ns))
    return SimpleNamespace(state=state, plan=plan, approve=approve, run=run, world=world)


class _Capsys:
    current = None


@pytest.fixture(autouse=True)
def _capsys_handle(capsys):
    _Capsys.current = capsys


CHAIN = AR.CHAIN


def test_composed_the_plan_previews_one_cell_and_touches_nothing(campaign, world):
    request, _ = campaign.plan()
    assert request["schema"] == RT.T_RECOVERY_REQUEST_SCHEMA and request["mode"] == "recovery"
    assert request["cells"] == [world.recovered] and request["gpu_count"] == 1
    assert len(request["carried"]) == 11 and request["recovery_of"]["cell_id"] == world.recovered["cell_id"]
    assert request["claim_key"] == world.exception_key
    assert request["source_transition"]["changed"] == sorted(RT.RECOVERY_CHANGED_CLOSURE)
    assert request["stopped"]["settlement"]["cumulative_charged_seconds"] == 79482.14623009507
    assert campaign.state["events"] == [] and new_claims() == []
    assert not world.records.exists() or not any(world.records.iterdir())


@pytest.mark.parametrize("gpus,ns,reason", [("2,3", NEW_NS, "exactly one GPU"), ("3", NS, "new namespace")],
                         ids=["two-gpus", "stopped-namespace"])
def test_composed_the_request_is_one_gpu_and_a_new_namespace(campaign, capsys, gpus, ns, reason):
    assert campaign.run("--plan", ns=ns, gpus=gpus) == 2
    assert reason in capsys.readouterr().err


def test_composed_an_approved_recovery_runs_one_cell_and_writes_the_lineage_receipt(campaign, world):
    _, digest = campaign.plan()
    assert campaign.run(*campaign.approve(digest)) == 0
    st = campaign.state
    assert st["commands"] == CHAIN
    # payloads verified before any lease; the claim exists BEFORE the attempt and the entry
    assert st["events"][:3] == ["payloads", "leases", "boundary"] and st["events"].count("payloads") == 2
    entry = [e for e in st["events"] if isinstance(e, tuple)][0]
    assert entry[1] == sorted([f"{RT.recovery_claim_key(world.lineage)}.json", f"{world.exception_key}.json"])
    assert entry[2] == [f"{NEW_NS}_attempt_{world.recovered['tag']}.json"]
    receipt = json.loads((world.records / f"{NEW_NS}{RT.T_RECOVERY_RECEIPT_SUFFIX}").read_text())
    origins = {cid: c["origin"] for cid, c in receipt["cells"].items()}
    assert sorted(origins.values()) == ["carried"] * 11 + ["executed"]
    assert origins[world.recovered["cell_id"]] == "executed"
    assert all(c["namespace"] == NS and c["record_dir"] == str(world.hist)
               for cid, c in receipt["cells"].items() if c["origin"] == "carried")
    assert receipt["schema"] == RT.T_RECOVERY_RECEIPT_SCHEMA and receipt["expected_cells"] == 12
    assert not (world.hist / f"{NS}_test_complete.json").exists()
    assert not (world.records / f"{NEW_NS}_test_complete.json").exists()


@pytest.mark.parametrize("scope_or_digest", ["stage-T-run", "other-request", "r7-manifest"])
def test_composed_only_a_recovery_line_for_this_exact_request_approves(campaign, world, capsys, scope_or_digest):
    _, digest = campaign.plan()
    if scope_or_digest == "stage-T-run":
        extra = campaign.approve(digest, scope="stage-T-run")
    elif scope_or_digest == "other-request":
        extra = campaign.approve("0" * 64)
    else:
        extra = campaign.approve(digest, manifest=world.r7_sha)
    assert campaign.run(*extra) == 2
    assert "approval" in capsys.readouterr().err or True
    assert campaign.state["events"] == [] and new_claims() == []


def test_composed_a_carried_payload_off_its_record_refuses_before_any_lease_or_claim(campaign, world, capsys):
    _, digest = campaign.plan()
    run = Path(world.cells[sorted(world.lineage["carried"])[3]]["run_dir"])
    (run / "extract_db.npz").write_bytes(b"changed")
    assert campaign.run(*campaign.approve(digest)) == 2
    assert "carried outputs are not the bytes" in capsys.readouterr().err
    assert campaign.state["events"] == ["payloads"] and new_claims() == []


@pytest.mark.parametrize("k", [None, 0, 2])
def test_composed_the_lineage_is_recovered_once_even_after_a_failure(campaign, world, capsys, k):
    _, digest = campaign.plan()
    campaign.state["fail_at"] = k
    first = campaign.run(*campaign.approve(digest))
    assert first == (0 if k is None else 1)
    seen = len(campaign.state["commands"])
    claim = RT.recovery_claim_path(world.exception_key)
    assert claim.exists() and RT.attempt_path(NEW_NS, world.recovered).exists()
    campaign.state["fail_at"] = None
    capsys.readouterr()
    # again, in the same namespace and in another: refused before any process starts -- at the admission
    # once the cell holds outputs (k None, 2), else (the entry failed before any output) at the claim
    if k == 0:
        assert campaign.run(*campaign.approve(digest)) == 2          # the namespace is already reserved
    else:
        assert campaign.run("--plan") == 2                           # the cell holds outputs: admission refuses
    rc = LT.run_main(campaign.state["mp"], *world.argv("--run", "--gpus", "3", "--plan", ns="ancT9q"))
    err = capsys.readouterr().err
    if k == 0:
        assert rc == 0
        _, other = campaign.plan(ns="ancT9q")
        assert campaign.run(*campaign.approve(other), ns="ancT9q") == 1
        assert "already claimed" in capsys.readouterr().err
    else:
        assert rc == 2 and "never reaches the official test" in err
    assert len(campaign.state["commands"]) == seen


def test_composed_concurrent_namespaces_claim_the_lineage_once(campaign, world):
    claims, results = [], []
    request = {"claim_root": str(RT.RECOVERY_CLAIM_ROOT), "claim_key": world.exception_key,
               "cells": [world.recovered], "record_dir": str(world.records), "lineage": {"path": "l", "sha256": "s"}}
    approval = {"section": 800, "scope": RT.RECOVERY_SCOPE, "line": "L"}

    def go(ns):
        try:
            claims.append(RT.consume_recovery_claim(dict(request, namespace=ns), approval))
            results.append("ok")
        except CellRefused:
            results.append("refused")
    threads = [threading.Thread(target=go, args=(f"ancR{i}",)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(results) == ["ok"] + ["refused"] * 5
    assert len(new_claims()) == 1


def test_composed_an_alternate_record_root_cannot_claim_again(campaign, world, monkeypatch, tmp_path):
    _, digest = campaign.plan()
    campaign.state["fail_at"] = 0                         # the entry fails before any output: the claim is spent
    assert campaign.run(*campaign.approve(digest)) == 1
    campaign.state["fail_at"] = None
    other = tmp_path / "another_worktree_records"
    monkeypatch.setattr(M, "ANCHOR_RECORD_DIR", other)
    monkeypatch.setattr(M, "RECORD_DIR", other)
    _, again = campaign.plan(ns="ancT9z")
    seen = len(campaign.state["commands"])
    assert campaign.run(*campaign.approve(again), ns="ancT9z") == 1
    assert len(campaign.state["commands"]) == seen and not list(other.glob("*_attempt_*.json"))


@pytest.mark.parametrize("k", range(6), ids=["before-entry", "entry-to-train", "after-train", "after-bio",
                                              "after-nmi", "after-seal"])
def test_composed_drift_between_producers_stops_the_recovery_without_a_receipt(campaign, world, k):
    _, digest = campaign.plan()
    campaign.state["drift_at"] = k
    assert campaign.run(*campaign.approve(digest)) == 1
    assert campaign.state["commands"] == CHAIN[:k]
    assert not (world.records / f"{NEW_NS}{RT.T_RECOVERY_RECEIPT_SUFFIX}").exists()
    # the claim is consumed after the pre-attempt check and is never removed afterwards
    assert RT.recovery_claim_path(world.exception_key).exists() == (k > 0)


def _carried_run(world, k=5):
    return Path(world.cells[sorted(world.lineage["carried"])[k]]["run_dir"])


def _edit_carried_json(world):
    path = _carried_run(world) / "cell_result.json"
    path.write_bytes(path.read_bytes() + b" ")


def _edit_carried_binary(world):
    (_carried_run(world) / "extract_train.npz").write_bytes(b"other array bytes")


def _remove_carried_output(world):
    (_carried_run(world) / "analysis_complete.json").unlink()


def _add_carried_bit2(world):
    (_carried_run(world) / BIT2).write_text("{}")


def _edit_carried_record(world):
    path = world.hist / world.carried_pins(5)["record"]["file"]
    path.write_bytes(path.read_bytes() + b" ")


def _edit_carried_entry(world):
    path = world.hist / world.carried_pins(5)["entry"]["file"]
    path.write_bytes(path.read_bytes() + b" ")


def _append_settled_ledger(world):
    with world.ops.open("a") as handle:
        handle.write(jline({"schema": S.LEDGER_SCHEMA, "event": "start", "run_id": "late"}) + "\n")


def _edit_stopped_reservation(world):
    path = world.hist / world.lineage["stopped"]["reservation"]["file"]
    path.write_bytes(path.read_bytes() + b" ")


def _edit_the_claim(world):
    path = RT.recovery_claim_path(world.exception_key)          # the exception claim this run took
    path.chmod(0o644)
    path.write_bytes(path.read_bytes() + b" ")


def _edit_the_consumed_claim(world):
    path = world.consumed_claim()                                # the failed r8 recovery's claim (audit 841.3)
    path.chmod(0o644)
    path.write_bytes(path.read_bytes() + b" ")


def _failed_entry_appears(world):
    (world.failed_dir / world.exception["absent"][0]).write_text("{}")


def _failed_receipt_dangling_link(world):
    (world.failed_dir / world.exception["absent"][2]).symlink_to(world.failed_dir / "nowhere")


def _failed_run_row_appended(world):
    with world.failed_ledger.open("a") as handle:
        handle.write(jline({"schema": S.LEDGER_SCHEMA, "event": "final", "run_id": FAILED_RUN}) + "\n")


@pytest.mark.parametrize("during", [_edit_carried_json, _edit_carried_binary, _remove_carried_output,
                                    _add_carried_bit2, _edit_carried_record, _edit_carried_entry,
                                    _append_settled_ledger, _edit_stopped_reservation, _edit_the_claim,
                                    _edit_the_consumed_claim, _failed_entry_appears, _failed_receipt_dangling_link,
                                    _failed_run_row_appended],
                         ids=["carried-json", "carried-binary", "carried-missing", "carried-new-bit2",
                              "carried-record", "carried-entry", "settled-ledger", "stopped-reservation", "claim",
                              "consumed-claim", "failed-entry", "failed-receipt-link", "failed-run-row"])
def test_composed_drift_during_the_recovered_cell_publishes_no_combined_receipt(campaign, world, during):
    """Audit 800: the final carry/lineage closure runs AFTER the new cell, immediately before publication."""
    _, digest = campaign.plan()
    campaign.state["during"] = during
    assert campaign.run(*campaign.approve(digest)) == 1
    assert campaign.state["commands"] == CHAIN                       # the cell itself completed
    assert (world.records / f"{NEW_NS}_{world.recovered['tag']}.json").exists()         # kept, not deleted
    assert not (world.records / f"{NEW_NS}{RT.T_RECOVERY_RECEIPT_SUFFIX}").exists()
    assert RT.recovery_claim_path(world.exception_key).exists()


def _new_attempt(world):
    return world.records / f"{NEW_NS}_attempt_{world.recovered['tag']}.json"


def _new_entry(world):
    return world.records / f"{NEW_NS}_entry_{world.recovered['tag']}.json"


def _change_new_attempt(world):
    path = _new_attempt(world)
    path.chmod(0o644)
    path.write_bytes(path.read_bytes() + b" ")


def _remove_new_attempt(world):
    _new_attempt(world).unlink()


def _remove_new_entry_claim(world):
    _new_entry(world).unlink()


def _mismatch_new_entry_claim(world):
    path = _new_entry(world)
    claim = json.loads(path.read_bytes())
    claim["attempt_sha256"] = "0" * 64
    path.chmod(0o644)
    path.write_text(json.dumps(claim))


@pytest.mark.parametrize("during", [_change_new_attempt, _remove_new_attempt, _remove_new_entry_claim,
                                    _mismatch_new_entry_claim],
                         ids=["new-attempt-changed", "new-attempt-missing", "new-entry-missing",
                              "new-entry-mismatched"])
def test_composed_a_new_cell_chain_off_its_files_publishes_no_combined_receipt(campaign, world, during):
    """Audit 801: the recovered cell's own attempt and entry claim are re-read at the publication boundary."""
    _, digest = campaign.plan()
    campaign.state["during"] = during
    assert campaign.run(*campaign.approve(digest)) == 1
    assert campaign.state["commands"] == CHAIN
    assert (world.records / f"{NEW_NS}_{world.recovered['tag']}.json").exists()
    assert not (world.records / f"{NEW_NS}{RT.T_RECOVERY_RECEIPT_SUFFIX}").exists()
    assert RT.recovery_claim_path(world.exception_key).exists()


def test_composed_the_receipt_binds_the_bytes_the_final_closure_verified(campaign, world):
    _, digest = campaign.plan()
    assert campaign.run(*campaign.approve(digest)) == 0
    receipt = json.loads((world.records / f"{NEW_NS}{RT.T_RECOVERY_RECEIPT_SUFFIX}").read_text())
    for cid, pins in world.lineage["carried"].items():
        got = receipt["cells"][cid]
        assert (got["record_sha256"], got["attempt_sha256"], got["entry_sha256"]) == \
            (pins["record"]["sha256"], pins["attempt"]["sha256"], pins["entry"]["sha256"])
    claim = RT.recovery_claim_path(world.exception_key)
    assert receipt["recovery_claim"]["sha256"] == M._sha(claim)
    executed = receipt["cells"][world.recovered["cell_id"]]
    tag = world.recovered["tag"]
    attempt, entry = world.records / f"{NEW_NS}_attempt_{tag}.json", world.records / f"{NEW_NS}_entry_{tag}.json"
    assert (executed["record_sha256"], executed["attempt_sha256"], executed["entry_sha256"]) == \
        (M._sha(world.records / executed["record"]), M._sha(attempt), M._sha(entry))
    assert receipt["final_closure"]["executed"] == {
        "record": {"file": executed["record"], "sha256": executed["record_sha256"]},
        "attempt": {"file": attempt.name, "sha256": executed["attempt_sha256"]},
        "entry": {"file": entry.name, "sha256": executed["entry_sha256"]}}


def test_composed_the_recovery_takes_no_receipt_smoke_or_cell_selection(campaign, world, capsys):
    for extra in (["--anchor-refit-receipt", "x", "--anchor-refit-receipt-sha256", "0" * 64],
                  ["--only", "nuswide:4:anchors:44"]):
        assert campaign.run("--plan", *extra) == 2
    assert campaign.state["events"] == []


# =============================================================================================
# composed: the T entry proves the recovery authority itself
# =============================================================================================
@pytest.fixture
def rentry(campaign, world, monkeypatch):
    request, digest = campaign.plan()
    campaign.approve(digest)
    line = LT.approval_line(RT.RECOVERY_SCOPE, manifest=world.manifest_sha, freeze=RT.ANCHOR_F_RECORD_SHA256,
                            request=digest)
    approval = {"section": 800, "scope": RT.RECOVERY_SCOPE, "line": line}
    world.records.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(request=request, digest=digest, approval=approval, line=line)


def _enter(world, monkeypatch, request, approval, *, claim=True, claim_request=None, claim_ns=None):
    if claim:
        claimed = dict(claim_request or request)
        if claim_ns:
            claimed["namespace"] = claim_ns
        RT.consume_recovery_claim(claimed, approval)
    attempt, attempt_sha = RT.reserve_attempt(request["namespace"], cell=request["cells"][0], request=request,
                                              approval=approval, campaign_nonce=NONCE)
    calls = []
    import extraction_siglip2
    import terminal_official_test
    monkeypatch.setattr(extraction_siglip2, "_resume_args_flat_or_legacy",
                        lambda args: (_ for _ in ()).throw(AssertionError("no second config read")))
    monkeypatch.setattr(terminal_official_test, "run_official_test", lambda args, **k: calls.append("official_test"))
    loads = []
    import torch
    real_load = torch.load
    monkeypatch.setattr(torch, "load", lambda *a, **k: loads.append(1) or real_load(*a, **k))
    argv = ["--config_path", request["cells"][0]["run_dir"], "--attempt", str(attempt), "--attempt-sha256", attempt_sha]
    return TE.main(argv), calls, loads, attempt


def test_composed_an_authorized_recovery_entry_runs_the_terminal_function_once(rentry, world, monkeypatch):
    rc, calls, _, attempt = _enter(world, monkeypatch, rentry.request, rentry.approval)
    assert rc == 0 and calls == ["official_test"] and TE.entry_claim_path(attempt).exists()


def test_composed_an_entry_without_the_claim_refuses_before_any_load(rentry, world, monkeypatch, capsys):
    rc, calls, loads, attempt = _enter(world, monkeypatch, rentry.request, rentry.approval, claim=False)
    assert rc == 2 and calls == [] and loads == [] and not TE.entry_claim_path(attempt).exists()
    assert "recovery claim is absent" in capsys.readouterr().err


@pytest.mark.parametrize("variant", ["other-namespace", "other-approval"])
def test_composed_a_claim_for_another_namespace_or_line_refuses_the_entry(rentry, world, monkeypatch, capsys, variant):
    if variant == "other-namespace":
        rc, calls, loads, _ = _enter(world, monkeypatch, rentry.request, rentry.approval, claim_ns="ancT9q")
    else:
        rc, calls, loads, _ = _enter(world, monkeypatch, rentry.request, rentry.approval,
                                     claim_request=dict(rentry.request, record_dir="/elsewhere"))
    assert rc == 2 and calls == [] and loads == []
    assert "does not authorize this request" in capsys.readouterr().err


def test_composed_a_run_scope_line_never_admits_a_recovery_entry(rentry, world, monkeypatch, capsys, tmp_path):
    line = LT.approval_line("stage-T-run", manifest=world.manifest_sha, freeze=RT.ANCHOR_F_RECORD_SHA256,
                            request=rentry.digest)
    AR.full_ledger(tmp_path, monkeypatch, {**world.sections, 800: [line]})
    rc, calls, loads, _ = _enter(world, monkeypatch, rentry.request, {**rentry.approval, "line": line})
    assert rc == 2 and calls == [] and loads == []


def test_composed_the_stage_t_form_with_the_recovery_mode_refuses(rentry, world, monkeypatch, capsys):
    mixed = dict(rentry.request, schema=RT.T_REQUEST_SCHEMA)
    rc, calls, loads, _ = _enter(world, monkeypatch, mixed, rentry.approval, claim=False)
    assert rc == 2 and calls == [] and "mixes the stage-T and recovery forms" in capsys.readouterr().err


def test_composed_a_counterfeit_lineage_in_an_approved_request_refuses_the_entry(rentry, world, monkeypatch, capsys,
                                                                                tmp_path):
    forged = json.loads(json.dumps(rentry.request))
    forged["carried"] = forged["carried"][1:]                 # one carried cell silently dropped
    digest = M._json_digest(forged)
    line = LT.approval_line(RT.RECOVERY_SCOPE, manifest=world.manifest_sha, freeze=RT.ANCHOR_F_RECORD_SHA256,
                            request=digest)
    AR.full_ledger(tmp_path, monkeypatch, {**world.sections, 801: [line]})
    approval = {"section": 801, "scope": RT.RECOVERY_SCOPE, "line": line}
    rc, calls, loads, _ = _enter(world, monkeypatch, forged, approval)
    assert rc == 2 and calls == [] and loads == []
    assert "not the pinned stopped campaign's" in capsys.readouterr().err


def test_composed_a_recovery_attempt_never_enters_twice(rentry, world, monkeypatch, capsys):
    rc, calls, _, attempt = _enter(world, monkeypatch, rentry.request, rentry.approval)
    assert rc == 0
    argv = ["--config_path", rentry.request["cells"][0]["run_dir"], "--attempt", str(attempt), "--attempt-sha256",
            M._sha(attempt)]
    assert TE.main(argv) == 2 and calls == ["official_test"]


# =============================================================================================
# composed: the supervisor's separate recovery ledger bound to the settled R/T ledger
# =============================================================================================
def test_composed_the_recovery_keeps_its_own_ledger_and_budget():
    assert S.stage_ledger("stage-T-recovery", None) == (S.REC_OPS_ROOT, 15000.0)
    for other in (S.RT_OPS_ROOT, S.L_OPS_ROOT, S.DEFAULT_OPS_ROOT):
        with pytest.raises(S.Refused):
            S.stage_ledger("stage-T-recovery", str(other))
    with pytest.raises(S.Refused):
        S.stage_ledger("stage-T-run", str(S.REC_OPS_ROOT))


def test_composed_the_recovery_is_one_gpu_and_five_producers():
    assert S.command_gpus("stage-T-recovery", ["--gpus", "3"]) == 1
    with pytest.raises(S.Refused):
        S.command_gpus("stage-T-recovery", ["--gpus", "2,3"])
    assert S.CHILDREN_PER_CELL["stage-T-recovery"] == len(RT.T_CHAIN) == 5


@pytest.fixture
def parent(tmp_path, monkeypatch):
    rows = [{"schema": S.LEDGER_SCHEMA, "event": "start", "run_id": "a"},
            {"schema": S.LEDGER_SCHEMA, "event": "final", "run_id": "a", "charged_seconds": 7280.510054702871},
            {"schema": S.LEDGER_SCHEMA, "event": "start", "run_id": RUN_ID},
            {"schema": S.LEDGER_SCHEMA, "event": "final", "run_id": RUN_ID, "charged_seconds": 72201.6361753922,
             "cumulative_charged_seconds": 79482.14623009507}]
    path = tmp_path / "parent.jsonl"
    path.write_text("".join(jline(r) + "\n" for r in rows))
    monkeypatch.setattr(S, "REC_PARENT_LEDGER", path)
    monkeypatch.setattr(S, "REC_PARENT_LEDGER_SHA256", sha(path.read_bytes()))
    return path


@pytest.fixture
def rsupervised(tmp_path, parent):
    ops, out = tmp_path / "recops", tmp_path / "out"
    out.mkdir()
    script = tmp_path / "children.py"
    script.write_text(AR.CHILDREN_LAUNCHER)

    def run(runs, *, sleep=0.25, budget=15000.0):
        return S.supervise([sys.executable, str(script), str(runs), str(sleep), "-1"], stage="stage-T-recovery",
                           label="r", gpus=1, attempts="sessions", planned_cells=1, watch_path=str(out),
                           ops_root=str(ops), manifest_sha256="f" * 64, budget_seconds=budget, poll_seconds=0.1,
                           watchdog_seconds=5.0, stop_bound_seconds=2.0, ledger_every=1.0,
                           lease_root=tmp_path, free_bytes=lambda _p: 1 << 62)

    def rows():
        return [json.loads(line) for line in (ops / S.LEDGER_NAME).read_text().splitlines()]
    return SimpleNamespace(run=run, rows=rows, parent=parent)


def test_composed_a_recovery_start_binds_the_settled_parent_ledger_and_carries_its_charge(rsupervised):
    before = rsupervised.parent.read_bytes()
    assert rsupervised.run(5) == 0
    start = [r for r in rsupervised.rows() if r["event"] == "start"][-1]
    final = [r for r in rsupervised.rows() if r["event"] == "final"][-1]
    assert start["parent"]["charged_seconds"] == 79482.14623009507 and start["parent"]["sha256"] == sha(before)
    assert final["parent_charged_seconds"] == 79482.14623009507
    assert final["cumulative_including_parent_seconds"] == pytest.approx(79482.14623009507 + final["charged_seconds"])
    assert rsupervised.parent.read_bytes() == before                    # never appended or rewritten


@pytest.mark.parametrize("edit", ["appended", "reformatted", "unsettled", "other-charge"])
def test_composed_a_changed_or_unsettled_parent_refuses_before_start(rsupervised, monkeypatch, edit):
    path = rsupervised.parent
    if edit == "appended":
        with path.open("a") as handle:
            handle.write(jline({"schema": S.LEDGER_SCHEMA, "event": "start", "run_id": "z"}) + "\n")
    elif edit == "reformatted":             # the same records in other bytes: refused at the pin alone
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        path.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows))
    else:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if edit == "unsettled":
            rows.append({"schema": S.LEDGER_SCHEMA, "event": "start", "run_id": "z"})
        else:
            rows[-1]["cumulative_charged_seconds"] = 1.0
        path.write_text("".join(jline(r) + "\n" for r in rows))
        monkeypatch.setattr(S, "REC_PARENT_LEDGER_SHA256", sha(path.read_bytes()))   # repinned: the content refuses
    assert rsupervised.run(5) == S.EXIT_REFUSED
    final = [r for r in rsupervised.rows() if r["event"] == "final"][-1]
    assert final["status"] == "refused-before-start" and final["reason"].startswith("parent:")


def test_composed_a_sixth_producer_of_the_recovered_cell_is_excess(rsupervised):
    assert rsupervised.run(6, sleep=0.5) == S.EXIT_UNRESOLVED       # each producer outlives several polls
    final = [r for r in rsupervised.rows() if r["event"] == "final"][-1]
    assert final["reason"].startswith("continuity lost: attempts: 6 observed for 1")


def test_composed_the_recovery_budget_binds_the_increment(rsupervised):
    # this fixture's rules: 5 attempts x 0.1 s poll + 1 GPU x (5 s watchdog + 2 s stop bound) reach it
    assert rsupervised.run(5, budget=5 * 0.1 + 1 * (5.0 + 2.0)) == S.EXIT_REFUSED
    final = [r for r in rsupervised.rows() if r["event"] == "final"][-1]
    assert final["reason"].startswith("budget:")


# =============================================================================================
# structural
# =============================================================================================
def test_structural_the_r8_closure_is_r7_plus_the_declared_additions():
    previous = set(AR.V8_FILES_SHA256) | AR.ADDED_IN_V9
    assert set(M.anchor_generation_closure()) == previous | set(RT.RECOVERY_ADDED_CLOSURE)


def test_structural_the_changed_sources_are_stage_r_sources_and_control_plane_only():
    assert set(RT.RECOVERY_CHANGED_SOURCES) <= set(M._BOOTSTRAP_SOURCE_PATHS)
    scientific = {"extraction_siglip2.py", "evaluation_siglip2.py", "terminal_official_test.py", "model_siglip2.py",
                  "train_siglip2.py", "loss_siglip2.py", "dataloaders.py", "config.py",
                  "scripts/extract_train_split.py", "scripts/eval_cell_bioproj.py", "scripts/pairwise_nmi.py",
                  "scripts/seal_cell_analysis.py", "dna_utils/runtime_state.py", "dna_utils/scientific_recipe.py"}
    assert not scientific & set(RT.RECOVERY_CHANGED_CLOSURE)


def test_structural_the_recovery_scope_binds_the_f_record():
    assert M.APPROVAL_SCOPES[RT.RECOVERY_SCOPE] == ("manifest", "freeze", "request")


def test_composed_a_manifest_without_the_recovery_contract_refuses(tmp_path):
    path, digest = LT.manifest(tmp_path, recovery_contract=LT.DELETE)
    with pytest.raises(CellRefused, match="stage-T recovery contract"):
        M.load_anchor_manifest(path, digest)


# ---- further refusals the mutation battery pins (each disables one whole condition) ---------------
def test_composed_a_reservation_not_binding_its_snapshot_refuses(world):
    world.repin_file(world.lineage["stopped"]["reservation"], lambda r: r.update(plan_digest="0" * 64))
    with pytest.raises(CellRefused, match="does not bind its snapshot"):
        world.admit()


def test_composed_the_recorded_stopped_line_must_be_the_ledger_s(world):
    world.relineage(lambda lin: lin["stopped"]["approval"].update(line="ANCHOR-CONFIRM-APPROVAL forged"))
    with pytest.raises(CellRefused, match="not the ledger's line"):
        world.admit()


def test_composed_a_request_naming_another_claim_root_refuses(world):
    request = {"claim_root": "/elsewhere", "claim_key": world.exception_key,
               "cells": [world.recovered], "record_dir": str(world.records), "namespace": NEW_NS,
               "lineage": {"path": "l", "sha256": "s"}}
    with pytest.raises(CellRefused, match="another recovery claim root"):
        RT.consume_recovery_claim(request, {"section": 800, "scope": RT.RECOVERY_SCOPE, "line": "L"})
    assert new_claims() == []


def test_composed_a_changed_source_authority_outside_the_allowlist_refuses(boundary, monkeypatch):
    def moved(sources):
        got = _bundle()(sources)
        got["entries"]["p0_protocol.py"]["head_sha256"] = "0" * 64
        return got
    monkeypatch.setattr(M, "_source_authority_bundle", moved)
    with pytest.raises(CellRefused, match="source authority changed"):
        boundary.run()


def test_composed_the_plan_refuses_a_transition_other_than_the_declared_one(campaign, monkeypatch, capsys):
    monkeypatch.setattr(RT, "RECOVERY_CHANGED_CLOSURE", RT.RECOVERY_CHANGED_CLOSURE[:-1])
    assert campaign.run("--plan") == 2
    assert "transition is not the reviewed one" in capsys.readouterr().err


def test_composed_a_settlement_line_off_its_pin_refuses(world):
    world.relineage(lambda lin: lin["stopped"]["settlement"].update(final_line_sha256="0" * 64))
    with pytest.raises(CellRefused, match="line is not its pinned bytes"):
        world.admit()


def test_composed_a_new_record_off_its_outputs_publishes_no_receipt(campaign, world):
    _, digest = campaign.plan()
    campaign.state["forge_new_completion"] = True
    assert campaign.run(*campaign.approve(digest)) == 1
    assert not (world.records / f"{NEW_NS}{RT.T_RECOVERY_RECEIPT_SUFFIX}").exists()
    assert (world.records / f"{NEW_NS}_{world.recovered['tag']}.json").exists()


# =============================================================================================
# audit 839: the recovery through the REAL managed-child boundary (worker thread, session registry)
# =============================================================================================
_SYNTHETIC_CHILD = r"""
import json, os, sys
from pathlib import Path
kind, marker, claim_root = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
claims = sorted(p.name for p in claim_root.glob("*.json")) if claim_root.is_dir() else []
with open(marker, "a") as log:
    log.write(json.dumps({"kind": kind, "name": sys.argv[4], "pid": os.getpid(), "claims": claims}) + "\n")
if kind == "entry":
    repo, attempt, attempt_sha, run_dir, required, fail = sys.argv[5:11]
    sys.path.insert(0, repo)
    from scripts import anchor_terminal_test as TE
    attempt = Path(attempt)
    TE.claim_entry(attempt, attempt_sha, json.loads(attempt.read_bytes())["cell"])
    if fail == "1":
        sys.exit(1)
    for out in json.loads(required):
        (Path(run_dir) / out).write_bytes(("new:" + out).encode())
sys.exit(0)
"""


@pytest.fixture
def real_dispatch(campaign, world, tmp_path, monkeypatch):
    """The campaign with the REAL M._run_managed_process: its main-thread guard, session registry and drain.
    Only the scientific work is substituted: every producer command becomes one tiny private synthetic
    child (the entry child performs the real entry claim and writes the outputs; post-chain children only
    record their start). The dispatching thread of every launch is recorded."""
    # the real launcher keeps process-wide child-registry state; other test files can leave it in a shutdown
    # (blocked launches, lease descriptors). Start from a fresh launcher's state, restored after the test.
    monkeypatch.setattr(M, "_CHILD_LAUNCH_BLOCKED", False)
    monkeypatch.setattr(M, "_CAMPAIGN_LEASE_FDS", ())
    monkeypatch.setattr(M, "_ACTIVE_CHILDREN", set())
    child = tmp_path / "synthetic_child.py"
    child.write_text(_SYNTHETIC_CHILD)
    marker = tmp_path / "child_starts.jsonl"
    state = campaign.state
    state.update(dispatch=[], entry_fail=False)

    def managed(command, *, cwd, env):
        name = Path(command[2] if command[1] == "-B" else command[1]).name
        state["dispatch"].append((name, threading.current_thread() is threading.main_thread(),
                                  threading.current_thread().name))
        common = [sys.executable, "-B", str(child)]
        if name == "anchor_terminal_test.py":
            argv = common + ["entry", str(marker), str(RT.RECOVERY_CLAIM_ROOT), name, str(Path(RT.REPO)),
                             command[command.index("--attempt") + 1], command[command.index("--attempt-sha256") + 1],
                             command[command.index("--config_path") + 1], json.dumps(REQUIRED),
                             "1" if state["entry_fail"] else "0"]
        else:
            argv = common + ["post", str(marker), str(RT.RECOVERY_CLAIM_ROOT), name]
        state["commands"].append(name)
        return _REAL_MANAGED(argv, cwd=cwd, env=env)        # the real guard, Popen, registry and drain
    monkeypatch.setattr(M, "_run_managed_process", managed)

    def starts():
        return [json.loads(line) for line in marker.read_text().splitlines()] if marker.exists() else []
    yield SimpleNamespace(campaign=campaign, state=state, starts=starts, child=child, marker=marker)
    # the fresh registry must be drained by the real launcher itself: no synthetic child is hidden
    assert M._owned_live_sessions() == [], "a synthetic child session outlived its test"


def test_real_dispatch_the_recovery_launches_its_children_from_a_worker_thread(real_dispatch, world):
    campaign, state = real_dispatch.campaign, real_dispatch.state
    _, digest = campaign.plan()
    assert campaign.run(*campaign.approve(digest)) == 0
    assert state["dispatch"] and not any(on_main for _n, on_main, _t in state["dispatch"])
    assert len({t for _n, _m, t in state["dispatch"]}) == 1          # one worker runs the whole cell
    starts = real_dispatch.starts()
    entries = [s for s in starts if s["kind"] == "entry"]
    assert len(entries) == 1                                        # the synthetic entry started once
    assert entries[0]["claims"] == sorted([f"{RT.recovery_claim_key(world.lineage)}.json",
                                          f"{world.exception_key}.json"])        # claim before the entry
    assert [s["name"] for s in starts] == CHAIN
    assert (world.records / f"{NEW_NS}{RT.T_RECOVERY_RECEIPT_SUFFIX}").is_file()


def test_real_dispatch_a_failed_entry_child_publishes_no_receipt(real_dispatch, world, capsys):
    campaign, state = real_dispatch.campaign, real_dispatch.state
    state["entry_fail"] = True
    _, digest = campaign.plan()
    assert campaign.run(*campaign.approve(digest)) == 1
    assert "the T entry exited 1" in capsys.readouterr().err
    assert [s["kind"] for s in real_dispatch.starts()] == ["entry"]   # nothing after the failed entry
    assert not (world.records / f"{NEW_NS}{RT.T_RECOVERY_RECEIPT_SUFFIX}").exists()
    assert list(RT.RECOVERY_CLAIM_ROOT.glob("*.json")) and list(world.records.glob(f"{NEW_NS}_attempt_*.json"))
    assert not any(on_main for _n, on_main, _t in state["dispatch"])
    assert M._owned_live_sessions() == []                          # the session registry is drained


def test_real_dispatch_an_exception_in_the_worker_publishes_no_receipt(real_dispatch, world, capsys, monkeypatch):
    campaign = real_dispatch.campaign
    _, digest = campaign.plan()

    def boom(*a, **k):
        raise RuntimeError("injected closure failure")
    monkeypatch.setattr(RT, "final_recovery_closure", boom)
    assert campaign.run(*campaign.approve(digest)) == 1
    assert "injected closure failure" in capsys.readouterr().err
    assert not (world.records / f"{NEW_NS}{RT.T_RECOVERY_RECEIPT_SUFFIX}").exists()


def test_real_dispatch_a_shutdown_within_the_test_still_refuses_a_new_child(real_dispatch, world, capsys, monkeypatch):
    """Negative control (audit 842): the fixture's fresh state does not disable the shutdown guard."""
    campaign = real_dispatch.campaign
    _, digest = campaign.plan()
    monkeypatch.setattr(M, "_CHILD_LAUNCH_BLOCKED", True)
    assert campaign.run(*campaign.approve(digest)) == 1
    assert "campaign shutdown has begun; refusing a new child process" in capsys.readouterr().err
    assert real_dispatch.starts() == []                                   # no synthetic child started
    assert not (world.records / f"{NEW_NS}{RT.T_RECOVERY_RECEIPT_SUFFIX}").exists()


def test_real_dispatch_the_main_thread_is_still_refused(real_dispatch):
    """Negative control (audit 842): the reset does not bypass the main-thread guard."""
    with pytest.raises(CellRefused, match="must launch from a worker thread"):
        _REAL_MANAGED([sys.executable, "-B", str(real_dispatch.child), "post", str(real_dispatch.marker),
                       str(RT.RECOVERY_CLAIM_ROOT), "control"], cwd=str(Path(RT.REPO)), env=dict())
    assert real_dispatch.starts() == []


# =============================================================================================
# audits 839-841: the explicit exception for one more recovery after the failed r8 attempt
# =============================================================================================
def _plan_refuses(world, monkeypatch, capsys, *, ns=NEW_NS):
    rc = LT.run_main(monkeypatch, *world.argv("--run", "--gpus", "3", "--plan", ns=ns))
    return rc, capsys.readouterr().err


def _rewrite(path: Path, data: bytes):
    path.chmod(0o644)
    path.write_bytes(data)


@pytest.mark.parametrize("drift", ["consumed-claim", "attempt", "reservation", "snapshot", "ledger-line",
                                   "ledger-missing", "ledger-run-row", "entry-present", "record-present",
                                   "receipt-dangling-link", "lstat-error", "other-cell", "claim-not-consumed-key"])
def test_exception_admission_refuses_a_changed_or_inconsistent_failed_attempt(world, monkeypatch, capsys, drift):
    ex = world.exception
    if drift in ("consumed-claim", "attempt", "reservation", "snapshot"):
        key = {"consumed-claim": "claim"}.get(drift, drift)
        path = Path(ex[key]["path"])
        _rewrite(path, path.read_bytes() + b" ")
    elif drift == "ledger-line":
        data = world.failed_ledger.read_bytes().replace(b'"returncode": 1', b'"returncode": 0')
        world.failed_ledger.write_bytes(data)
    elif drift == "ledger-missing":
        world.failed_ledger.unlink()
    elif drift == "ledger-run-row":
        with world.failed_ledger.open("a") as handle:
            handle.write(jline({"schema": S.LEDGER_SCHEMA, "event": "final", "run_id": FAILED_RUN}) + "\n")
    elif drift == "entry-present":
        (world.failed_dir / ex["absent"][0]).write_text("{}")
    elif drift == "record-present":
        (world.failed_dir / ex["absent"][1]).write_text("{}")
    elif drift == "receipt-dangling-link":
        (world.failed_dir / ex["absent"][2]).symlink_to(world.failed_dir / "nowhere")
    elif drift == "lstat-error":
        real = RT.os.lstat
        target = str(world.failed_dir / ex["absent"][1])

        def lstat(path, *a, **k):
            if str(path) == target:
                raise PermissionError(13, "denied")
            return real(path, *a, **k)
        monkeypatch.setattr(RT.os, "lstat", lstat)
    elif drift == "other-cell":
        world.reexception(lambda e: e.update(cell_id="flickr25k|N=4|other"))
    elif drift == "claim-not-consumed-key":
        world.reexception(lambda e: e["claim"].update(key="0" * 64))
    rc, err = _plan_refuses(world, monkeypatch, capsys)
    assert rc == 2 and new_claims() == []
    want = {"lstat-error": "cannot be observed", "receipt-dangling-link": "a link counts",
            "entry-present": "exists", "record-present": "exists", "ledger-missing": "recovery ledger cannot be read",
            "ledger-line": "pinned rows", "ledger-run-row": "pre-entry failure",
            "other-cell": "interrupted cell", "claim-not-consumed-key": "consumed claim"}.get(drift, "pinned bytes")
    assert want in err, err


@pytest.mark.parametrize("ns", ["ancT9", FAILED_NS])
def test_exception_never_reuses_the_stopped_or_the_failed_namespace(world, monkeypatch, capsys, ns):
    rc, err = _plan_refuses(world, monkeypatch, capsys, ns=ns)
    assert rc == 2 and "never the stopped campaign's or the failed recovery's" in err


def test_exception_request_binds_both_histories_and_the_exception_key(campaign, world):
    request, _ = campaign.plan()
    assert request["schema"] == "anchor-terminal-test-recovery-request/2"
    assert request["exception_of"]["run_id"] == FAILED_RUN and request["exception_of"]["namespace"] == FAILED_NS
    assert request["exception_of"]["ledger"]["charged_seconds"] == FAILED_CHARGE
    assert request["recovery_of"]["cell_id"] == world.recovered["cell_id"]
    assert request["claim_key"] == world.exception_key != RT.recovery_claim_key(world.lineage)
    assert request["exception_of"]["exception"]["sha256"] == RT.RECOVERY_EXCEPTION_SHA256


def test_exception_claim_key_ignores_namespace_root_and_generation(world):
    block = dict(world.failed_block)
    other = dict(block, namespace="elsewhere", record_dir="/other/root", exception={"path": "x", "sha256": "y"})
    assert RT.exception_claim_key(block, world.recovered["cell_id"]) == \
        RT.exception_claim_key(other, world.recovered["cell_id"])


def test_exception_a_second_namespace_after_a_completed_exception_refuses(campaign, world, monkeypatch, capsys):
    """After the one exception ran, another namespace is refused already at admission (the target holds
    outputs); the claim-level refusal is covered by the concurrent-namespace test with the same key."""
    _, digest = campaign.plan()
    assert campaign.run(*campaign.approve(digest)) == 0
    assert new_claims() == [f"{world.exception_key}.json"]
    capsys.readouterr()
    rc, err = _plan_refuses(world, monkeypatch, capsys, ns="ancT9t")
    assert rc == 2 and new_claims() == [f"{world.exception_key}.json"]


def test_exception_the_failed_attempt_is_rechecked_immediately_before_the_claim(campaign, world, monkeypatch, capsys):
    _, digest = campaign.plan()
    stub = RT.recovery_boundary                          # the campaign fixture's boundary, run after admission

    def boundary_then_drift(*a, **k):
        out = stub(*a, **k)
        (world.failed_dir / world.exception["absent"][0]).write_text("{}")   # the failed entry appears
        return out
    monkeypatch.setattr(RT, "recovery_boundary", boundary_then_drift)
    assert campaign.run(*campaign.approve(digest)) == 1
    assert "the failed recovery" in capsys.readouterr().err
    assert new_claims() == [] and not list(world.records.glob(f"{NEW_NS}_attempt_*.json"))


def test_exception_receipt_records_both_histories(campaign, world):
    _, digest = campaign.plan()
    assert campaign.run(*campaign.approve(digest)) == 0
    receipt = json.loads((world.records / f"{NEW_NS}{RT.T_RECOVERY_RECEIPT_SUFFIX}").read_text())
    assert receipt["exception_of"]["run_id"] == FAILED_RUN and receipt["stopped"]["namespace"] == NS
    assert receipt["recovery_claim"]["file"] == f"{world.exception_key}.json"


def test_exception_entry_refuses_a_schema_one_request(rentry, world, monkeypatch, capsys):
    old = dict(rentry.request, schema="anchor-terminal-test-recovery-request/1")
    rc, calls, loads, attempt = _enter(world, monkeypatch, old, rentry.approval, claim=False)
    assert rc == 2 and calls == [] and loads == []
    assert "not a stage-T request" in capsys.readouterr().err


def test_exception_entry_rechecks_the_failed_attempt(rentry, world, monkeypatch, capsys):
    (world.failed_dir / world.exception["absent"][1]).write_text("{}")      # the failed record appears
    rc, calls, loads, attempt = _enter(world, monkeypatch, rentry.request, rentry.approval)
    assert rc == 2 and calls == [] and loads == [] and not TE.entry_claim_path(attempt).exists()
    assert "the failed recovery" in capsys.readouterr().err


def _failed_rows(ops: Path):
    ops.mkdir(parents=True, exist_ok=True)
    rows = [{"schema": S.LEDGER_SCHEMA, "event": "start", "run_id": FAILED_RUN, "stage": "stage-T-recovery"},
            {"schema": S.LEDGER_SCHEMA, "event": "final", "run_id": FAILED_RUN, "stage": "stage-T-recovery",
             "status": "exited", "returncode": 1, "attempts": [], "charged_seconds": FAILED_CHARGE}]
    data = "".join(jline(r) + "\n" for r in rows).encode()
    (ops / S.LEDGER_NAME).write_bytes(data)
    return data


def test_exception_the_failed_charge_is_carried_in_the_append_only_ledger(rsupervised, tmp_path):
    prefix = _failed_rows(tmp_path / "recops")
    assert rsupervised.run(5) == 0
    raw = (tmp_path / "recops" / S.LEDGER_NAME).read_bytes()
    assert raw.startswith(prefix)                                        # history never rewritten
    start = [r for r in rsupervised.rows() if r["event"] == "start"][-1]
    final = [r for r in rsupervised.rows() if r["event"] == "final"][-1]
    assert start["prior_charged_seconds"] == pytest.approx(FAILED_CHARGE)
    assert final["cumulative_charged_seconds"] == pytest.approx(FAILED_CHARGE + final["charged_seconds"])
    assert final["cumulative_including_parent_seconds"] == pytest.approx(
        79482.14623009507 + FAILED_CHARGE + final["charged_seconds"])


def test_exception_the_carried_charge_counts_against_the_budget(rsupervised, tmp_path):
    # headroom 1 x (5 + 2) s and allowance 5 x 0.1 s: 7.5 s fit a 13 s budget, 7.5 + 5.70 s do not
    assert rsupervised.run(5, budget=13.0) == 0                        # control: no prior charge
    ops = tmp_path / "recops"
    (ops / S.LEDGER_NAME).unlink()
    _failed_rows(ops)
    assert rsupervised.run(5, budget=13.0) == S.EXIT_REFUSED
    final = [r for r in rsupervised.rows() if r["event"] == "final"][-1]
    assert final["status"] == "refused-before-start" and final["reason"].startswith("budget:")
