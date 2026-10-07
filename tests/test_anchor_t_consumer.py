"""Synthetic tests for the anchor stage-T consumer and its two formatters (proposal v2; audits 831, 833, 834).

A private synthetic world in tmp_path holds every artifact the real chain will have: F, the stage-R receipt
and records, the lineage, the settled parent ledger, the recovery ledger, receipt, snapshot, reservation and
claim, the approved section extracts, eleven carried r7 and one executed r8 record/attempt/entry, and per
cell the raw/BIO evaluation, pairwise NMI and DB extraction manifest. The World builds them in dependency
order and recomputes every downstream digest after a mutation hook, so each refusal case isolates one
check. Tests drive the formatters' real main(); no producer, checkpoint, NPZ, real record or GPU is used.
"""
import copy
import dataclasses
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import anchor_t_consumer as C  # noqa: E402
from scripts import make_anchor_bioproj_table as BIO  # noqa: E402
from scripts import make_anchor_nmi_table as NMI  # noqa: E402

PARAMS = {"cifar10": ("P=0.3,0.7|JD=0.02", 4), "flickr25k": ("P=0.6,0.95|JD=0.02", 4),
          "nuswide": ("P=0.4,0.8|JD=0.05", 4), "mscoco": ("P=0.6,0.95|JD=0.03", 39)}
EXEC = ("nuswide", 44)


@pytest.fixture(scope="session", autouse=True)
def imported_source_identities():
    """The modules under test import from the tree under test; record file and digest for the evidence."""
    rows = {m.__name__: {"file": m.__file__, "sha256": C.sha256(Path(m.__file__).read_bytes()),
                         "inside_tree": Path(m.__file__).resolve().is_relative_to(REPO)} for m in (C, BIO, NMI)}
    assert all(r["inside_tree"] for r in rows.values()), rows
    out = Path(os.environ.get("TMPDIR", "/tmp")) / "consumer_module_identities.json"
    try:
        out.write_text(json.dumps(rows, indent=1, sort_keys=True))
    except OSError:
        pass
    return rows


def dump(path: Path, obj) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(obj, indent=1, sort_keys=True).encode()
    path.write_bytes(data)
    return C.sha256(data)


class World:
    def __init__(self, root: Path, hook=None):
        self.root, self.hook = root, hook or (lambda kind, key, obj: obj)
        self.r7, self.r8 = root / "r7rec", root / "r8rec"
        self.results, self.claims, self.ext = root / "results", root / "claims", root / "extracts"
        self.out = root / "out"
        self.stopped_request = {"schema": "stopped-request", "cells": 12}
        self.request = {"schema": "anchor-terminal-test-recovery-request/1", "namespace": "ancT9r", "gpu_count": 1}
        self.nonce = "ab" * 32
        self.parent_run, self.run_id = "20261006T145139Z-b08e4ae2", "20261007T063000Z-0000beef"
        self.drop = set()

    def h(self, kind, key, obj):
        return self.hook(kind, key, obj)

    def cell_id(self, ds, seed):
        p, n = PARAMS[ds]
        return f"{ds}|N={n}|{p}|stage=refit|seed={seed}|axis_center=anchors"

    def build(self):
        auth0 = C.Authority()
        A = {"stopped_request_sha256": C.json_digest(self.stopped_request), "request_sha256": C.json_digest(self.request),
             "historical_record_dir": str(self.r7), "record_dir": str(self.r8), "claim_root": str(self.claims),
             "stopped_nonce": "cd" * 32, "recovery_attempt_sha256": "11" * 32, "recovery_entry_sha256": "22" * 32}
        self.auth = dataclasses.replace(auth0, **A)
        cells = {}
        for di, ds in enumerate(C.DATASETS):
            for si, seed in enumerate(C.SEEDS):
                cells[(ds, seed)] = self._cell(ds, seed, di, si)
        # F and the stage-R receipt
        freeze = self.h("freeze", None, {"datasets": {ds: {"N": PARAMS[ds][1], "axis_center": "anchors"}
                                                      for ds in C.DATASETS}})
        f_sha = dump(self.root / "F.json", freeze)
        rcells = {c["cell"]["cell_id"]: {"record": Path(c["cell"]["record"]).name, "record_sha256": c["cell"]["record_sha256"],
                                         "seed": c["cell"]["seed"], "run_dir": c["cell"]["run_dir"]} for c in cells.values()}
        refit = self.h("refit", None, {"cells": rcells})
        r_sha = dump(self.r7 / "ancR9_sweep_complete.json", refit)
        self.auth = dataclasses.replace(self.auth, freeze=(str(self.root / "F.json"), f_sha),
                                        refit_receipt=(str(self.r7 / "ancR9_sweep_complete.json"), r_sha))
        # carried and executed record / attempt / entry
        triples, lineage_carried = {}, {}
        for key, c in cells.items():
            executed = key == EXEC
            ns = self.auth.namespace if executed else self.auth.stopped_namespace
            root = self.r8 if executed else self.r7
            tag = c["cell"]["tag"]
            approval = ({"section": 830, "scope": "stage-T-recovery", "line": self.auth.launch_line,
                         "ledger": "/x/ledger.md", "ledger_sha256": "ee" * 32} if executed else
                        {"section": 788, "scope": "stage-T-run", "line": self.auth.stopped_line,
                         "ledger": "/x/ledger.md", "ledger_sha256": "e3" * 32})
            request = self.request if executed else self.stopped_request
            nonce = self.nonce if executed else self.auth.stopped_nonce
            attempt = self.h("attempt", key, {"schema": C.T_ATTEMPT_SCHEMA, "namespace": ns, "campaign_nonce": nonce,
                                             "request": request, "request_sha256": C.json_digest(request),
                                             "cell": copy.deepcopy(c["cell"]), "approval": approval})
            a_name = f"{ns}_attempt_{tag}.json"
            a_sha = dump(root / a_name, attempt)
            entry = self.h("entry", key, {"attempt": a_name, "attempt_sha256": a_sha, "cell_id": c["cell"]["cell_id"],
                                         "final_checkpoint_sha256": c["cell"].get("final_checkpoint_sha256"),
                                         "config_pt_sha256": c["cell"].get("config_pt_sha256")})
            e_name = f"{ns}_entry_{tag}.json"
            e_sha = dump(root / e_name, entry)
            record = self.h("record", key, {"schema": C.T_RECORD_SCHEMA, "stage": "test", "namespace": ns,
                                           "request_sha256": C.json_digest(request), "campaign_nonce": nonce,
                                           "attempt": {"file": a_name, "sha256": a_sha},
                                           "cell": copy.deepcopy(c["cell"]),
                                           "completion": copy.deepcopy(c["completion"])})
            r_name = f"{ns}_{tag}.json"
            rec_sha = dump(root / r_name, record)
            triples[key] = {"origin": "executed" if executed else "carried", "ns": ns, "root": root,
                            "record": (r_name, rec_sha), "attempt": (a_name, a_sha), "entry": (e_name, e_sha),
                            "cell_id": c["cell"]["cell_id"], "tag": tag}
            if not executed:
                lineage_carried[c["cell"]["cell_id"]] = {
                    "record": {"file": r_name, "sha256": rec_sha}, "attempt": {"file": a_name, "sha256": a_sha},
                    "entry": {"file": e_name, "sha256": e_sha}, "tag": tag}
        lineage = self.h("lineage", None, {"schema": C.LINEAGE_SCHEMA, "carried": lineage_carried,
                                          "recovery_cell": {"cell_id": self.auth.recovery_cell,
                                                            "attempt": {"file": "x", "sha256": self.auth.recovery_attempt_sha256},
                                                            "entry": {"file": "y", "sha256": self.auth.recovery_entry_sha256}}})
        l_sha = dump(self.r8 / "anchor_t_recovery_lineage_v1.json", lineage)
        self.auth = dataclasses.replace(self.auth, lineage=(str(self.r8 / "anchor_t_recovery_lineage_v1.json"), l_sha))
        # parent ledger (settled)
        parent_rows = self.h("parent_ledger", None, [
            {"schema": C.LEDGER_SCHEMA, "event": "start", "run_id": self.parent_run},
            {"schema": C.LEDGER_SCHEMA, "event": "final", "run_id": self.parent_run, "status": "stopped",
             "cumulative_charged_seconds": self.auth.parent_charged}])
        p_path = self.root / "parent_ops" / "device_budget_ledger.jsonl"
        p_sha = self._jsonl(p_path, parent_rows)
        self.auth = dataclasses.replace(self.auth, parent_ledger=(str(p_path), p_sha), parent_last_run_id=self.parent_run)
        # recovery ledger
        rec_rows = self.h("recovery_ledger", None, self._recovery_rows())
        rl_path = self.root / "rec_ops" / "device_budget_ledger.jsonl"
        rl_sha = self._jsonl(rl_path, rec_rows)
        # snapshot, reservation, claim, receipt
        executed_t = triples[EXEC]
        approval830 = {"section": 830, "scope": "stage-T-recovery", "line": self.auth.launch_line}
        snapshot = self.h("snapshot", None, {"schema": C.SNAPSHOT_SCHEMA,
                                            "plan": {"namespace": "ancT9r", "campaign_nonce": self.nonce, "declared_count": 1,
                                                     "executed_count": 1},
                                            "request": self.request, "request_sha256": C.json_digest(self.request),
                                            "approval": approval830})
        snap_digest = C.json_digest(snapshot)
        snap_name = f"ancT9r_snapshot_{snap_digest[:16]}.json"
        s_sha = dump(self.r8 / snap_name, snapshot)
        reservation = self.h("reservation", None, {"schema_version": 1, "namespace": "ancT9r", "campaign_nonce": self.nonce,
                                                  "plan_digest": snap_digest, "plan_snapshot_file": snap_name})
        v_sha = dump(self.r8 / "ancT9r_campaign_reservation.json", reservation)
        claim = self.h("claim", None, {"schema": C.CLAIM_SCHEMA, "claim_key": self.auth.claim_key,
                                      "request_sha256": C.json_digest(self.request), "approval": approval830,
                                      "namespace": "ancT9r", "record_dir": str(self.r8),
                                      "attempt": executed_t["attempt"][0],
                                      "lineage": {"path": self.auth.lineage[0], "sha256": l_sha}})
        c_path = self.claims / f"{self.auth.claim_key}.json"
        c_sha = dump(c_path, claim)
        rows = {t["cell_id"]: {"origin": t["origin"], "namespace": t["ns"], "record_dir": str(t["root"]),
                               "record": t["record"][0], "record_sha256": t["record"][1],
                               "attempt_sha256": t["attempt"][1], "entry_sha256": t["entry"][1]}
                for t in triples.values()}
        rows[executed_t["cell_id"]].update({"map_at_R": 0.123, "bio_map_at_R": 0.456})   # convenience, never read
        receipt = self.h("receipt", None, {
            "schema": C.RECEIPT_SCHEMA, "mode": "recovery", "namespace": "ancT9r", "campaign_nonce": self.nonce,
            "plan_snapshot_file": snap_name, "plan_snapshot_sha256": snap_digest,
            "campaign_reservation_file": "ancT9r_campaign_reservation.json", "campaign_reservation_sha256": v_sha,
            "request_sha256": C.json_digest(self.request),
            "refit_receipt": {"path": self.auth.refit_receipt[0], "sha256": r_sha},
            "lineage": {"path": self.auth.lineage[0], "sha256": l_sha},
            "stopped": {"namespace": "ancT9", "request_sha256": self.auth.stopped_request_sha256,
                        "approval": {"section": 788, "scope": "stage-T-run", "line": self.auth.stopped_line},
                        "settlement": {"ledger_sha256": p_sha}},
            "recovery_of": {"cell_id": self.auth.recovery_cell, "attempt": {"sha256": self.auth.recovery_attempt_sha256},
                            "entry": {"sha256": self.auth.recovery_entry_sha256}},
            "recovery_claim": {"file": c_path.name, "root": str(self.claims), "sha256": c_sha},
            "final_closure": {"settlement": {"ledger_sha256": p_sha},
                              "executed": {k: {"file": executed_t[k][0], "sha256": executed_t[k][1]}
                                           for k in ("record", "attempt", "entry")}},
            "expected_cells": 12, "cell_count": 12, "cells": rows})
        rc_path = self.r8 / "ancT9r_recovery_complete.json"
        rc_sha = dump(rc_path, receipt)
        # approved section extracts
        ext = {}
        for name, section, line in (("launch", 830, self.auth.launch_line), ("stopped_approval", 788, self.auth.stopped_line)):
            text = self.h("extract", name, f"## {section}. Synthetic section\n\nBody.\n\n```\n{line}\n```\n\nEnd.\n")
            path = self.ext / f"section_{section}.md"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            ext[name] = {"section": section, "line": line, "extract": {"path": str(path), "sha256": C.sha256(text.encode())}}
        inputs = self.h("inputs", None, {
            "schema": C.INPUTS_SCHEMA, **ext, "parent_ledger": {"path": str(p_path), "sha256": p_sha},
            "recovery": {"run_id": self.run_id, "receipt": {"path": str(rc_path), "sha256": rc_sha},
                         "ledger": {"path": str(rl_path), "sha256": rl_sha}, "claim": {"path": str(c_path), "sha256": c_sha},
                         "snapshot": {"path": str(self.r8 / snap_name), "sha256": s_sha},
                         "reservation": {"path": str(self.r8 / "ancT9r_campaign_reservation.json"), "sha256": v_sha}},
            "cells": [{"cell_id": t["cell_id"], "origin": t["origin"],
                       **{k: {"path": str(t["root"] / t[k][0]), "sha256": t[k][1]} for k in ("record", "attempt", "entry")}}
                      for t in triples.values()]})
        self.inputs = self.root / "inputs.json"
        i_sha = dump(self.inputs, inputs)
        self.auth = dataclasses.replace(self.auth, inputs_sha256=i_sha)
        for path in self.drop:
            Path(path).unlink()
        self.cells, self.triples = cells, triples
        return self

    def _jsonl(self, path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        data = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode()
        path.write_bytes(data)
        return C.sha256(data)

    def _recovery_rows(self):
        rules = dict(self.auth.rules)
        command = list(self.auth.expected_command)
        ident = {"schema": C.LEDGER_SCHEMA, "run_id": self.run_id, "stage": "stage-T-recovery", "label": ""}
        attempts = [{"id": [i], "start_boot": 100.0 + 1000 * i, "end_boot": 600.0 + 1000 * i, "seconds": 500.0}
                    for i in range(5)]
        device = sum(a["seconds"] for a in attempts)
        window = 2.0
        allowance = 5 * window
        charged = device + allowance
        start = {**ident, "event": "start", "command": command,
                 "command_sha256": C.sha256(json.dumps(command).encode()), "manifest_sha256": self.auth.manifest_sha256,
                 "supervisor_sha256": self.auth.supervisor_sha256, "prior_charged_seconds": 0.0, "free_bytes": 10 ** 12,
                 "refused": None, "parent": {"ledger": self.auth.parent_ledger[0], "sha256": self.auth.parent_ledger[1],
                                             "charged_seconds": self.auth.parent_charged,
                                             "last_run_id": self.auth.parent_last_run_id},
                 "rules": rules}
        poll = {**ident, "event": "poll", "device_seconds": 1.0}
        final = {**ident, "event": "final", "status": "exited", "reason": None, "returncode": 0, "attempts": attempts,
                 "device_seconds": device, "max_observation_window_seconds": window,
                 "unobserved_allowance_seconds": allowance, "charged_seconds": charged,
                 "cumulative_charged_seconds": charged, "parent_charged_seconds": self.auth.parent_charged,
                 "cumulative_including_parent_seconds": self.auth.parent_charged + charged, "wall_seconds": 9000.0,
                 "continuity_lost": [], "orphaned_live_attempts": [], "leases_held_after_exit": [], "monitor_failures": []}
        return [start, poll, final]

    def _cell(self, ds, seed, di, si):
        p, n = PARAMS[ds]
        tag = f"ancR9_{ds}_A_refit_N{n}_s{seed}"
        run = self.results / tag
        k = dict(C.Authority().codebook_size)[ds]
        rows = dict(C.Authority().db_rows)[ds]
        R = dict(C.Authority().r_cutoff)[ds]
        ckpt, cfg = C.sha256(f"ckpt{ds}{seed}".encode()), C.sha256(f"cfg{ds}{seed}".encode())
        npz = {s: C.sha256(f"npz{ds}{seed}{s}".encode()) for s in ("db", "query", "train")}
        key = (ds, seed)
        db = self.h("db_manifest", key, {"schema_version": 2, "split": "db", "n_rows": rows, "dataset": C.CANONICAL[ds],
                                         "random_seed": seed, "checkpoint_sha256": ckpt, "config_sha256": cfg,
                                         "inference_epoch": n, "npz_sha256": npz["db"], "backfilled": False})
        db_sha = dump(run / "extraction_manifest_db.json", db)
        manifests = {"db": db_sha, "query": "aa" * 32, "train": "bb" * 32}
        binding = {"schema_version": 1, "run_dir": str(run), "dataset": C.CANONICAL[ds], "random_seed": seed,
                   "inference_epoch": n, "checkpoint_sha256": ckpt, "config_sha256": cfg, "npz_sha256": npz,
                   "manifest_sha256": manifests, "codebook_size": k, "backfilled_inputs": False, "validator_sha256": "cc" * 32}
        pre = 0.5 + 0.05 * di + 0.01 * si
        post = pre + 0.002 * (si + 1)
        raw = self.h("raw", key, {"bio_project": False, "mAP_R_cutoff": R, "mAP_at_R": pre,
                                  "unique_code_ratio": 0.3 + 0.01 * si, "input_binding": copy.deepcopy(binding)})
        bio = self.h("bio", key, {"bio_project": True, "mAP_R_cutoff": R, "mAP_at_R": post, "unique_code_ratio": 0.35,
                                  "bio_stats": {"mAP_at_R_pre_projection": pre, "db_mean_edit_distance": 1.5 + 0.1 * si,
                                                "db_pre_compliance": 0.6 + 0.01 * si, "db_post_compliance": 1.0,
                                                "db_num_total": rows, "db_num_proj_failed": 0, "qy_num_proj_failed": 0},
                                  "input_binding": copy.deepcopy(binding)})
        m = [[1.0 if i == j else round(0.3 + 0.01 * (i + j) + 0.001 * si, 6) for j in range(5)] for i in range(5)]
        pairs = [m[i][j] for i in range(5) for j in range(i + 1, 5)]
        mean = math.fsum(pairs) / len(pairs)
        nmi = self.h("nmi", key, {"num_codebooks": 5, "nmi_average_method": "arithmetic", "nmi_matrix": m,
                                  "mean_off_diag_nmi": mean, "min_off_diag_nmi": min(pairs), "max_off_diag_nmi": max(pairs),
                                  "N": rows, "K_per_cb": [k] * 5, "unique_codewords_per_cb": [k - 1] * 5,
                                  "input_binding": copy.deepcopy(binding)})
        sums = {"evaluation_sha256": dump(run / "evaluation_siglip2_base.json", raw),
                "bio_evaluation_sha256": dump(run / "evaluation_siglip2_base_bioproj.json", bio),
                "pairwise_nmi_sha256": dump(run / "pairwise_nmi.json", nmi)}
        r_record = self.h("stage_r_record", key, {"tag": tag, "seed": seed})
        r_path = self.r7 / f"{tag}.json"
        r_sha = dump(r_path, r_record)
        cell = self.h("cell", key, {"cell_id": self.cell_id(ds, seed), "tag": tag, "dataset": ds, "seed": seed, "N": n,
                                    "terminal_epoch": n, "final_checkpoint": "model_state_dict.pth",
                                    "final_checkpoint_sha256": ckpt, "config_pt_sha256": cfg,
                                    "run_dir": str(run), "record": str(r_path), "record_sha256": r_sha})
        completion = self.h("completion", key, {**sums, "extraction_manifest_sha256": manifests, "npz_sha256": npz,
                                                "map_R_cutoff": R, "map_at_R": raw["mAP_at_R"],
                                                "bio_map_at_R": bio["mAP_at_R"],
                                                "mean_off_diag_nmi": nmi["mean_off_diag_nmi"],
                                                "analysis_protocol": {"codebook_size": k}})
        return {"cell": cell, "completion": completion}


def world(tmp_path, hook=None):
    return World(tmp_path / "w", hook).build()


def run(w, module, name="b1", before_recheck=None, out=None):
    return module.main(["--inputs", str(w.inputs), "--name", name, "--out-root", str(out or w.out)],
                       authority=w.auth, before_recheck=before_recheck)


@pytest.fixture(autouse=True)
def _protected_roots(tmp_path, monkeypatch):
    w = tmp_path / "w"
    monkeypatch.setattr(C, "FORBIDDEN_ROOTS", tuple(str(w / d) for d in ("r7rec", "r8rec", "results", "claims",
                                                                        "parent_ops", "rec_ops")))


# ---------------------------------------------------------------------------------------------- positive
def test_both_formatters_publish_and_read_exactly_the_declared_set(tmp_path):
    w = world(tmp_path)
    assert run(w, BIO, "b") == 0 and run(w, NMI, "n") == 0
    receipt = json.loads((w.out / "todo2_bioproj" / "b" / "bundle_receipt.json").read_text())
    consumed, kinds = receipt["consumed"], receipt["consumed_kinds"]
    expected = {str(w.inputs), str(w.root / "F.json"), w.auth.refit_receipt[0], w.auth.lineage[0], w.auth.parent_ledger[0],
                str(w.root / "rec_ops" / "device_budget_ledger.jsonl"), str(w.ext / "section_830.md"),
                str(w.ext / "section_788.md")}
    for t in w.triples.values():
        expected |= {str(t["root"] / t[k][0]) for k in ("record", "attempt", "entry")}
    for c in w.cells.values():
        run_dir = Path(c["cell"]["run_dir"])
        expected |= {c["cell"]["record"]} | {str(run_dir / f) for f in (
            "evaluation_siglip2_base.json", "evaluation_siglip2_base_bioproj.json", "pairwise_nmi.json",
            "extraction_manifest_db.json")}
    expected |= {str(w.r8 / "ancT9r_recovery_complete.json"), str(w.r8 / "ancT9r_campaign_reservation.json"),
                 str(next(w.r8.glob("ancT9r_snapshot_*.json"))), str(w.claims / f"{w.auth.claim_key}.json")}
    assert set(consumed) == expected and len(expected) == 108
    assert {kinds[str(w.ext / "section_830.md")], kinds[w.auth.parent_ledger[0]]} == {"markdown", "jsonl"}
    assert receipt["scope"] == "formatting of stored values" and receipt["paper_result_eligible"] is False
    assert not (w.out / "todo2_bioproj" / "b" / "INCOMPLETE").exists()


def test_table_values_come_from_the_metric_files_not_the_receipt_summary(tmp_path):
    w = world(tmp_path)
    assert run(w, BIO) == 0
    table = json.loads((w.out / "todo2_bioproj" / "b1" / "anchor_bioproj_15base_table.json").read_text())
    nus = table["rows"]["NUS-WIDE"]
    assert nus["per_seed"]["44"]["pre_map_at_R"] == pytest.approx(0.5 + 0.05 * 2 + 0.02)   # not the receipt's 0.123
    assert nus["per_seed"]["44"]["origin"] == "executed" and nus["per_seed"]["42"]["origin"] == "carried"
    assert nus["pre_map_at_R"]["sample_sd"] == pytest.approx(0.01)


def test_below_capacity_gapped_code_indices_are_valid(tmp_path):
    """Audit 834: K_per_cb is the largest observed index + 1, not capacity; gaps are allowed."""
    def hook(kind, key, obj):
        if kind == "nmi" and key[0] == "cifar10":
            obj["K_per_cb"], obj["unique_codewords_per_cb"] = [4, 6, 1, 64, 11], [3, 2, 1, 2, 3]
        return obj
    w = world(tmp_path, hook)
    assert run(w, NMI) == 0
    table = json.loads((w.out / "todo8_nmi" / "b1" / "anchor_nmi_15base_table.json").read_text())
    assert table["rows"]["CIFAR-10"]["per_seed"]["42"]["observed_max_code_index_plus_one_per_slot"] == [4, 6, 1, 64, 11]


def test_historical_approval_ledger_hashes_are_not_compared_with_any_current_ledger(tmp_path):
    """A resumed audit history: attempts record the ledger hash of their own time; the consumer binds the
    immutable section extract and its exact line instead."""
    def hook(kind, key, obj):
        if kind == "attempt":
            obj["approval"]["ledger_sha256"] = "0f" * 32
        return obj
    w = world(tmp_path, hook)
    assert run(w, NMI) == 0


def test_default_source_generation_refuses_without_a_reviewed_input_manifest(tmp_path):
    out = subprocess.run([sys.executable, str(REPO / "scripts" / "make_anchor_nmi_table.py"), "--inputs",
                          str(tmp_path / "nothing.json"), "--name", "x", "--out-root", str(tmp_path / "out")],
                         capture_output=True, text=True, cwd=tmp_path)
    assert out.returncode == 2 and "no reviewed consumer input manifest is pinned" in out.stderr
    assert not (tmp_path / "out").exists()


# ---------------------------------------------------------------------------------------------- refusals
def mutate(kind, key=None, fn=None):
    def hook(k, kk, obj):
        if k == kind and (key is None or kk == key):
            out = fn(obj)
            return obj if out is None else out
        return obj
    return hook


def final(fn):
    def edit(rows):
        fn(rows[-1])
    return mutate("recovery_ledger", None, edit)


def start(fn):
    def edit(rows):
        fn(rows[0])
    return mutate("recovery_ledger", None, edit)


def recompute(rows):
    f = rows[-1]
    f["device_seconds"] = sum(a["seconds"] for a in f["attempts"])
    f["charged_seconds"] = f["device_seconds"] + f["unobserved_allowance_seconds"]
    f["cumulative_charged_seconds"] = f["charged_seconds"]
    f["cumulative_including_parent_seconds"] = 79482.14623009507 + f["charged_seconds"]


def ledger(fn):
    def edit(rows):
        fn(rows)
        recompute(rows)
    return mutate("recovery_ledger", None, edit)


def _set(d, path, value):
    *head, last = path
    for p in head:
        d = d[p]
    d[last] = value


SETTLEMENT = {
    "start_only": (mutate("recovery_ledger", None, lambda rows: rows[:-1]), "single start and single final"),
    "status_stopped": (final(lambda f: f.update(status="stopped")), "final status is 'stopped'"),
    "status_unresolved": (final(lambda f: f.update(status="unresolved")), "final status is 'unresolved'"),
    "status_unclean": (final(lambda f: f.update(status="unclean")), "final status is 'unclean'"),
    "refused_before_start": (final(lambda f: f.update(status="refused-before-start")), "refused-before-start"),
    "failed_to_start": (final(lambda f: f.update(status="failed-to-start")), "failed-to-start"),
    "returncode_1": (final(lambda f: f.update(returncode=1)), "returned nonzero"),
    "returncode_bool": (final(lambda f: f.update(returncode=False)), "returned nonzero"),
    "leases_held": (final(lambda f: f.update(leases_held_after_exit=["GPU-x"])), "leases_held_after_exit"),
    "orphans": (final(lambda f: f.update(orphaned_live_attempts=[[1]])), "orphaned_live_attempts"),
    "continuity": (final(lambda f: f.update(continuity_lost=["gap"])), "continuity_lost"),
    "monitor_failures": (final(lambda f: f.update(monitor_failures=["x"])), "monitor_failures"),
    "over_budget": (ledger(lambda r: r[-1]["attempts"][0].update(end_boot=20100.0, seconds=20000.0)), "exceeded its budget"),
    "zero_allowance": (ledger(lambda r: r[-1].update(unobserved_allowance_seconds=0.0)), "planned attempts x the longest"),
    "manipulated_allowance": (ledger(lambda r: r[-1].update(unobserved_allowance_seconds=3.0)), "planned attempts x"),
    "window_over_watchdog": (ledger(lambda r: r[-1].update(max_observation_window_seconds=11.0,
                                                          unobserved_allowance_seconds=55.0)), "watchdog bound"),
    "attempt_seconds_mismatch": (ledger(lambda r: r[-1]["attempts"][1].update(seconds=400.0)), "end minus start"),
    "negative_time": (ledger(lambda r: r[-1]["attempts"][1].update(start_boot=-5.0, end_boot=495.0)), "negative time"),
    "device_not_sum": (final(lambda f: f.update(device_seconds=1.0)), "attempt sum"),
    "charge_not_sum": (final(lambda f: f.update(charged_seconds=1.0)), "device time plus allowance"),
    "parent_sum_wrong": (final(lambda f: f.update(cumulative_including_parent_seconds=5.0)), "does not add up"),
    "bool_number": (final(lambda f: f.update(wall_seconds=True)), "finite nonnegative"),
    "six_attempts": (ledger(lambda r: r[-1]["attempts"].append({"id": [9], "start_boot": 1.0, "end_boot": 2.0,
                                                                "seconds": 1.0})), "more than planned"),
    "two_runs": (mutate("recovery_ledger", None, lambda rows: rows + [dict(rows[0], run_id="other")]), "another run"),
    "line_after_final": (mutate("recovery_ledger", None, lambda rows: rows + [dict(rows[1])]), "after its final"),
    "parent_final_substituted": (mutate("recovery_ledger", None,
                                        lambda rows: [dict(r, run_id=r["run_id"]) for r in rows[:1]] + [
                                            {"schema": C.LEDGER_SCHEMA, "event": "final", "run_id": rows[0]["run_id"],
                                             "status": "stopped", "cumulative_charged_seconds": 79482.14623009507}]),
                                 "final status is 'stopped'"),
    "start_refused": (start(lambda s: s.update(refused="space")), "start was refused"),
    "prior_charge": (start(lambda s: s.update(prior_charged_seconds=5.0)), "prior charge"),
    "parent_digest": (start(lambda s: s["parent"].update(sha256="00" * 32)), "settled parent"),
    "other_manifest": (start(lambda s: s.update(manifest_sha256="00" * 32)), "another manifest"),
    "other_supervisor": (start(lambda s: s.update(supervisor_sha256="00" * 32)), "another supervisor"),
    "rule_headroom": (start(lambda s: s["rules"].update(headroom_seconds=0.0)), "rule headroom_seconds"),
    "rule_attempts": (start(lambda s: s["rules"].update(attempts="cells")), "rule attempts"),
    "rule_bool": (start(lambda s: s["rules"].update(gpus=True)), "rule gpus"),
    "other_namespace": (start(lambda s: (s["command"].__setitem__(5, "ancT9x"),
                                         s.update(command_sha256=C.sha256(json.dumps(s["command"]).encode())))),
                        "not the audited argv"),
    "other_gpus": (start(lambda s: (s["command"].__setitem__(12, "1"),
                                    s.update(command_sha256=C.sha256(json.dumps(s["command"]).encode())))),
                   "not the audited argv"),
    "other_section": (start(lambda s: (s["command"].__setitem__(14, "829"),
                                       s.update(command_sha256=C.sha256(json.dumps(s["command"]).encode())))),
                      "not the audited argv"),
    "digest_serialisation": (start(lambda s: s.update(command_sha256=C.sha256(
        json.dumps(s["command"], separators=(",", ":")).encode()))), "supervisor's serialisation"),
    "other_stage": (start(lambda s: s.update(stage="stage-T-run")), "not stage-T-recovery"),
}


@pytest.mark.parametrize("case", sorted(SETTLEMENT))
def test_recovery_settlement_refusals(tmp_path, case, capsys):
    hook, message = SETTLEMENT[case]
    w = world(tmp_path, hook)
    assert run(w, NMI) == 2
    assert message in capsys.readouterr().err
    assert not (w.out / "todo8_nmi" / "b1").exists()


def test_absent_recovery_ledger_refuses(tmp_path, capsys):
    w = world(tmp_path)
    (w.root / "rec_ops" / "device_budget_ledger.jsonl").unlink()
    assert run(w, NMI) == 2 and "recovery ledger" in capsys.readouterr().err


def test_non_finite_accounting_literal_refuses(tmp_path, capsys):
    w = world(tmp_path)
    path = w.root / "rec_ops" / "device_budget_ledger.jsonl"
    text = path.read_text().replace('"wall_seconds": 9000.0', '"wall_seconds": NaN')
    path.write_text(text)
    rec = json.loads(w.inputs.read_text())
    rec["recovery"]["ledger"]["sha256"] = C.sha256(text.encode())
    i_sha = dump(w.inputs, rec)
    w.auth = dataclasses.replace(w.auth, inputs_sha256=i_sha)
    assert run(w, NMI) == 2 and "not strict JSON" in capsys.readouterr().err


CHAIN = {
    "extract_missing_line": (mutate("extract", "launch", lambda t: t.replace("request=d34f", "request=e34f")),
                             "exact line once"),
    "extract_line_substring": (mutate("extract", "launch", lambda t: t.replace("\n```\n", "\n```\n> ", 1)),
                               "exact line once"),
    "extract_other_heading": (mutate("extract", "launch", lambda t: t.replace("## 830.", "## 831.")), "not section 830"),
    "extract_runs_on": (mutate("extract", "launch", lambda t: t + "\n## 831. Next\n"), "runs into another section"),
    "manifest_other_section": (mutate("inputs", None, lambda i: i["launch"].update(section=831)), "section 830"),
    "attempt_section": (mutate("attempt", EXEC, lambda a: a["approval"].update(section=829)), "approval"),
    "attempt_line": (mutate("attempt", EXEC, lambda a: a["approval"].update(line=C.STOPPED_LINE)), "approval"),
    "snapshot_nonce": (mutate("snapshot", None, lambda s: s["plan"].update(campaign_nonce="ef" * 32)), "nonce"),
    "snapshot_request": (mutate("snapshot", None, lambda s: s.update(request={"other": 1})), "request"),
    "snapshot_approval": (mutate("snapshot", None, lambda s: s["approval"].update(section=829)), "approval"),
    "reservation_nonce": (mutate("reservation", None, lambda r: r.update(campaign_nonce="ef" * 32)),
                          "reservation does not bind"),
    "reservation_plan": (mutate("reservation", None, lambda r: r.update(plan_digest="00" * 32)), "reservation does not bind"),
    "claim_approval": (mutate("claim", None, lambda c: c["approval"].update(scope="stage-T-run")), "claim does not bind"),
    "claim_lineage": (mutate("claim", None, lambda c: c["lineage"].update(sha256="00" * 32)), "claim does not bind"),
    "claim_attempt": (mutate("claim", None, lambda c: c.update(attempt="ancT9r_attempt_other.json")),
                      "claim authorizes another attempt"),
    "receipt_claim_digest": (mutate("receipt", None, lambda r: r["recovery_claim"].update(sha256="00" * 32)),
                             "receipt binds another claim"),
    "receipt_namespace": (mutate("receipt", None, lambda r: r.update(namespace="ancT9")), "not this recovery's"),
    "receipt_request": (mutate("receipt", None, lambda r: r.update(request_sha256="00" * 32)), "not this recovery's"),
    "receipt_stopped": (mutate("receipt", None, lambda r: r["stopped"].update(request_sha256="00" * 32)),
                        "another stopped campaign"),
    "receipt_recovery_of": (mutate("receipt", None, lambda r: r["recovery_of"].update(cell_id="x")), "recovers another"),
    "receipt_executed_triple": (mutate("receipt", None, lambda r: r["final_closure"]["executed"]["entry"].update(
        sha256="00" * 32)), "executed triple"),
    "lineage_recovery_cell": (mutate("lineage", None, lambda l: l["recovery_cell"].update(cell_id="x")),
                              "another recovered cell"),
    "parent_not_settled": (mutate("parent_ledger", None, lambda rows: rows[:1]), "does not end settled"),
    "parent_other_charge": (mutate("parent_ledger", None, lambda rows: rows[-1].update(cumulative_charged_seconds=1.0)),
                            "does not end settled"),
}


@pytest.mark.parametrize("case", sorted(CHAIN))
def test_chain_refusals(tmp_path, case, capsys):
    hook, message = CHAIN[case]
    w = world(tmp_path, hook)
    assert run(w, BIO) == 2
    assert message in capsys.readouterr().err


def _relabel(origin_cell, origin):
    def edit(inputs):
        for row in inputs["cells"]:
            if row["cell_id"] == origin_cell:
                row["origin"] = origin
    return mutate("inputs", None, edit)


CELLS = {
    "executed_labelled_carried": (_relabel("nuswide|N=4|P=0.4,0.8|JD=0.05|stage=refit|seed=44|axis_center=anchors",
                                           "carried"), "exactly the recovered one"),
    "carried_labelled_executed": (_relabel("cifar10|N=4|P=0.3,0.7|JD=0.02|stage=refit|seed=42|axis_center=anchors",
                                           "executed"), "exactly the recovered one"),
    "origin_unknown": (_relabel("cifar10|N=4|P=0.3,0.7|JD=0.02|stage=refit|seed=42|axis_center=anchors", "copied"),
                       "origin 'copied'"),
    "eleven_cells": (mutate("inputs", None, lambda i: i["cells"].__delitem__(-1)), "twelve cells"),
    "duplicate_cell": (mutate("inputs", None, lambda i: i["cells"].__setitem__(1, copy.deepcopy(i["cells"][0]))),
                       "missing or duplicated"),
    "record_nonce": (mutate("record", ("cifar10", 43), lambda r: r.update(campaign_nonce="ef" * 32)), "request, nonce"),
    "record_request": (mutate("record", ("cifar10", 43), lambda r: r.update(request_sha256="00" * 32)), "request, nonce"),
    "executed_record_nonce": (mutate("record", EXEC, lambda r: r.update(campaign_nonce="cd" * 32)), "request, nonce"),
    "attempt_cell": (mutate("attempt", ("mscoco", 42), lambda a: a["cell"].update(seed=43)), "attempt does not bind"),
    "entry_checkpoint": (mutate("entry", ("mscoco", 42), lambda e: e.update(final_checkpoint_sha256="00" * 32)),
                         "entry does not bind"),
    "lineage_triple": (mutate("lineage", None, lambda l: next(iter(l["carried"].values()))["entry"].update(
        sha256="00" * 32)), "carried triple"),
    "stage_r_not_in_receipt": (mutate("refit", None, lambda r: next(iter(r["cells"].values())).update(
        record_sha256="00" * 32)), "R receipt names"),
    "n_off_f": (mutate("freeze", None, lambda f: f["datasets"]["mscoco"].update(N=40)), "F does not freeze"),
    "seed_outside": (mutate("cell", ("flickr25k", 44), lambda c: c.update(seed=45)), "outside the design"),
    "seed_bool": (mutate("cell", ("cifar10", 42), lambda c: c.update(seed=True)), None),
    "legacy_shaped_record": (mutate("record", ("cifar10", 42), lambda r: {"dataset": "CIFAR10", "seed": 42,
                                                                          "run_dir": "/x", "completion": {}}),
                             "cell.cell_id is missing"),
}


@pytest.mark.parametrize("case", sorted(CELLS))
def test_cell_lineage_refusals(tmp_path, case, capsys):
    hook, message = CELLS[case]
    w = world(tmp_path, hook)
    assert run(w, NMI) == 2
    err = capsys.readouterr().err
    assert "REFUSED" in err and (message is None or message in err), err


def test_a_carried_record_in_the_r8_root_refuses(tmp_path, capsys):
    w = world(tmp_path)
    rec = json.loads(w.inputs.read_text())
    row = next(r for r in rec["cells"] if r["origin"] == "carried")
    for k in ("record", "attempt", "entry"):
        src = Path(row[k]["path"])
        dst = w.r8 / src.name
        dst.write_bytes(src.read_bytes())
        row[k]["path"] = str(dst)
    w.auth = dataclasses.replace(w.auth, inputs_sha256=dump(w.inputs, rec))
    assert run(w, NMI) == 2 and "outside its origin's record root" in capsys.readouterr().err


def at(kind, ds="flickr25k", seed=43):
    return lambda fn: mutate(kind, (ds, seed), fn)


METRICS = {
    "raw_bio_swapped": (at("raw")(lambda r: r.update(bio_project=True)), "not a raw/BIO pair"),
    "r_cutoff": (at("bio")(lambda b: b.update(mAP_R_cutoff=1000)), "R cutoff is not 5000"),
    "projection_failure": (at("bio")(lambda b: b["bio_stats"].update(db_num_proj_failed=1)), "projection failures"),
    "post_compliance": (at("bio")(lambda b: b["bio_stats"].update(db_post_compliance=0.99)), "compliance is not 1.0"),
    "pre_projection": (at("bio")(lambda b: b["bio_stats"].update(mAP_at_R_pre_projection=0.1)), "pre-projection"),
    "nmi_asymmetric": (at("nmi")(lambda n: n["nmi_matrix"][0].__setitem__(1, 0.9)), "NMI [0][1]"),
    "nmi_shape": (at("nmi")(lambda n: n.update(nmi_matrix=n["nmi_matrix"][:4])), "5-slot arithmetic"),
    "nmi_mean_file": (at("nmi")(lambda n: n.update(mean_off_diag_nmi=0.9)), "disagrees with the matrix"),
    "nmi_infinite": (at("nmi")(lambda n: n["nmi_matrix"][0].__setitem__(1, 1e999)), None),
    "completion_nmi": (at("completion")(lambda c: c.update(mean_off_diag_nmi=0.9)), "stored NMI mean"),
    "completion_map": (at("completion")(lambda c: c.update(map_at_R=0.9)), "stored mAP@R"),
    "completion_bio_map": (at("completion")(lambda c: c.update(bio_map_at_R=0.9)), "stored BIO mAP@R"),
    "binding_checkpoint": (at("raw")(lambda r: r["input_binding"].update(checkpoint_sha256="00" * 32)),
                           "input_binding checkpoint_sha256"),
    "binding_backfilled": (at("nmi")(lambda n: n["input_binding"].update(backfilled_inputs=True)), "backfilled_inputs"),
    "binding_epoch": (at("bio")(lambda b: b["input_binding"].update(inference_epoch=5)), "inference_epoch"),
    "binding_npz_map": (at("raw")(lambda r: r["input_binding"]["npz_sha256"].update(train="00" * 32)), "npz_sha256"),
    "binding_manifest_map": (at("nmi")(lambda n: n["input_binding"]["manifest_sha256"].update(db="00" * 32)),
                             "manifest_sha256"),
    "binding_config": (at("raw")(lambda r: r["input_binding"].update(config_sha256="00" * 32)), "config_sha256"),
    "binding_run_dir": (at("bio")(lambda b: b["input_binding"].update(run_dir="/elsewhere")), "run_dir"),
    "binding_seed_bool": (at("nmi")(lambda n: n["input_binding"].update(random_seed=True)), "random_seed"),
    "bindings_unequal": (at("nmi")(lambda n: n["input_binding"].update(validator_sha256="dd" * 32)), "bindings differ"),
    "db_rows_wrong": (at("db_manifest")(lambda d: d.update(n_rows=22999)), "reviewed full DB"),
    "db_rows_zero": (at("db_manifest")(lambda d: d.update(n_rows=0)), "schema-2 DB split"),
    "db_rows_bool": (at("db_manifest")(lambda d: d.update(n_rows=True)), "schema-2 DB split"),
    "db_schema_1": (at("db_manifest")(lambda d: d.update(schema_version=1)), "schema-2 DB split"),
    "db_split": (at("db_manifest")(lambda d: d.update(split="query")), "schema-2 DB split"),
    "db_checkpoint": (at("db_manifest")(lambda d: d.update(checkpoint_sha256="00" * 32)), "DB manifest checkpoint"),
    "db_backfilled": (at("db_manifest")(lambda d: d.update(backfilled=True)), "DB manifest backfilled"),
    "bio_total": (at("bio")(lambda b: b["bio_stats"].update(db_num_total=100)), "BIO DB total"),
    "nmi_rows": (at("nmi")(lambda n: n.update(N=100)), "NMI N"),
    "k_out_of_capacity": (at("nmi", "cifar10", 42)(lambda n: n.update(K_per_cb=[65, 64, 64, 64, 64],
                                                                      unique_codewords_per_cb=[60] * 5)), "occupancy"),
    "k_impossible_occupancy": (at("nmi")(lambda n: n.update(K_per_cb=[10] * 5, unique_codewords_per_cb=[11] * 5)),
                               "occupancy"),
    "k_zero_occupancy": (at("nmi")(lambda n: n.update(unique_codewords_per_cb=[0] * 5)), "occupancy"),
    "k_wrong_length": (at("nmi")(lambda n: n.update(K_per_cb=[128] * 4)), "five integers"),
    "k_boolean": (at("nmi")(lambda n: n.update(unique_codewords_per_cb=[True] * 5)), "five integers"),
    "record_codebook_size": (at("completion")(lambda c: c["analysis_protocol"].update(codebook_size=64)),
                             "codebook size is not 128"),
}


@pytest.mark.parametrize("case", sorted(METRICS))
def test_metric_binding_and_denominator_refusals(tmp_path, case, capsys):
    hook, message = METRICS[case]
    w = world(tmp_path, hook)
    assert run(w, BIO) == 2
    w2 = world(tmp_path / "again", hook)
    assert run(w2, NMI) == 2
    err = capsys.readouterr().err
    assert "REFUSED" in err and (message is None or message in err), err


@pytest.mark.parametrize("name", ["evaluation_siglip2_base.json", "pairwise_nmi.json", "extraction_manifest_db.json"])
def test_missing_or_changed_metric_file_refuses(tmp_path, name, capsys):
    w = world(tmp_path)
    path = Path(w.cells[("mscoco", 44)]["cell"]["run_dir"]) / name
    path.unlink()
    assert run(w, NMI) == 2 and "unreadable" in capsys.readouterr().err
    w2 = world(tmp_path / "again")
    path = Path(w2.cells[("mscoco", 44)]["cell"]["run_dir"]) / name
    path.write_text(path.read_text() + " ")
    assert run(w2, NMI) == 2 and "pinned" in capsys.readouterr().err


def test_a_parent_ledger_off_its_pin_refuses(tmp_path, capsys):
    w = world(tmp_path)
    w.auth = dataclasses.replace(w.auth, parent_ledger=(w.auth.parent_ledger[0], "11" * 32))
    assert run(w, NMI) == 2 and "not the settled pinned ledger" in capsys.readouterr().err


def test_input_manifest_absent_or_off_its_digest_refuses(tmp_path, capsys):
    w = world(tmp_path)
    w.inputs.write_text(w.inputs.read_text() + " ")
    assert run(w, NMI) == 2 and "consumer input manifest" in capsys.readouterr().err
    w.inputs.unlink()
    assert run(w, NMI) == 2 and "unreadable" in capsys.readouterr().err


def test_digests_that_agree_with_each_other_but_not_the_pin_refuse(tmp_path, capsys):
    """A caller-built manifest whose digests are internally consistent is not the reviewed manifest."""
    w = world(tmp_path)
    w2 = world(tmp_path / "other", mutate("bio", ("cifar10", 42), lambda b: b.update(unique_code_ratio=0.5)))
    assert w2.auth.inputs_sha256 != w.auth.inputs_sha256
    w2.auth = dataclasses.replace(w2.auth, inputs_sha256=w.auth.inputs_sha256)
    assert run(w2, NMI, "c") == 2 and "consumer input manifest" in capsys.readouterr().err


# ---------------------------------------------------------------------------------------------- publication
def test_an_existing_destination_refuses(tmp_path, capsys):
    w = world(tmp_path)
    assert run(w, NMI) == 0
    assert run(w, NMI) == 2 and "written once" in capsys.readouterr().err


def test_a_destination_inside_or_aliasing_a_protected_root_refuses(tmp_path, capsys):
    w = world(tmp_path)
    assert run(w, NMI, out=w.r8 / "bundles") == 2 and "protected root" in capsys.readouterr().err
    alias = tmp_path / "alias"
    alias.symlink_to(w.r8)
    assert run(w, NMI, out=alias) == 2 and "protected root" in capsys.readouterr().err
    assert run(w, NMI, out=tmp_path) == 2 and "protected root" in capsys.readouterr().err   # contains them


def test_drift_before_publication_keeps_an_incomplete_bundle_without_receipt(tmp_path, capsys):
    w = world(tmp_path)
    target = Path(w.cells[("cifar10", 42)]["cell"]["run_dir"]) / "pairwise_nmi.json"
    assert run(w, NMI, before_recheck=lambda: target.write_text(target.read_text() + " ")) == 2
    dest = w.out / "todo8_nmi" / "b1"
    assert "inputs changed before publication" in capsys.readouterr().err
    assert (dest / "INCOMPLETE").is_file() and not (dest / "bundle_receipt.json").exists()
    assert (dest / "anchor_nmi_15base_table.tex").is_file()


def test_every_file_is_read_once_before_the_final_check(tmp_path, monkeypatch):
    w = world(tmp_path)
    reads = []
    real = Path.read_bytes

    def counting(self):
        reads.append(str(self))
        return real(self)
    monkeypatch.setattr(Path, "read_bytes", counting)
    assert run(w, BIO) == 0
    consumed = json.loads((w.out / "todo2_bioproj" / "b1" / "bundle_receipt.json").read_text())["consumed"]
    for path in consumed:
        assert reads.count(path) == 2, (path, reads.count(path))     # one capture + the final no-drift check


# ---------------------------------------------------------------------------------------------- audit 835
PINS = {
    "evaluation_pin_absent": (at("completion")(lambda c: c.__delitem__("evaluation_sha256")), "completion.evaluation_sha256"),
    "bio_pin_null": (at("completion")(lambda c: c.update(bio_evaluation_sha256=None)), "completion.bio_evaluation_sha256"),
    "nmi_pin_malformed": (at("completion")(lambda c: c.update(pairwise_nmi_sha256="XYZ")), "completion.pairwise_nmi_sha256"),
    "nmi_pin_boolean": (at("completion")(lambda c: c.update(pairwise_nmi_sha256=True)), "completion.pairwise_nmi_sha256"),
    "evaluation_pin_uppercase": (at("completion")(lambda c: c.update(evaluation_sha256=c["evaluation_sha256"].upper())),
                                 "completion.evaluation_sha256"),
    "db_manifest_pin_absent": (at("completion")(lambda c: c["extraction_manifest_sha256"].__delitem__("db")),
                               "completion.extraction_manifest_sha256"),
    "npz_map_missing_member": (at("completion")(lambda c: c["npz_sha256"].__delitem__("train")), "completion.npz_sha256"),
    "npz_map_extra_member": (at("completion")(lambda c: c["npz_sha256"].update(val="00" * 32)), "completion.npz_sha256"),
    "npz_map_absent": (at("completion")(lambda c: c.__delitem__("npz_sha256")), "completion.npz_sha256"),
    "cell_checkpoint_absent": (at("cell")(lambda c: c.__delitem__("final_checkpoint_sha256")),
                               "cell.final_checkpoint_sha256"),
    "cell_epoch_boolean": (at("cell")(lambda c: c.update(terminal_epoch=True)), "cell.terminal_epoch"),
    "stage_r_pin_malformed": (at("cell")(lambda c: c.update(record_sha256="0" * 63)), "cell.record_sha256"),
}


@pytest.mark.parametrize("case", sorted(PINS))
def test_absent_null_malformed_or_boolean_pins_refuse_through_the_formatter(tmp_path, case, capsys):
    hook, message = PINS[case]
    w = world(tmp_path, hook)
    assert run(w, BIO) == 2
    err = capsys.readouterr().err
    assert "REFUSED" in err and message in err, err
    assert not (w.out / "todo2_bioproj" / "b1").exists()


def _counting(monkeypatch):
    reads = []
    real = Path.read_bytes

    def counting(self):
        reads.append(str(self))
        return real(self)
    monkeypatch.setattr(Path, "read_bytes", counting)
    return reads


def test_a_repeated_reference_is_parsed_from_the_first_capture(tmp_path, monkeypatch):
    path = tmp_path / "x.json"
    path.write_text('{"a": 1}')
    pin = C.sha256(path.read_bytes())
    reads = _counting(monkeypatch)
    reader = C.Reader()
    assert reader.json(path, pin, "first") == {"a": 1}
    path.write_text('{"a": 2}')                               # a later change is not re-read ...
    assert reader.json(path, pin, "second") == {"a": 1}       # ... the second reference uses the captured bytes
    assert reads.count(str(path)) == 1
    assert reader.recheck() == [str(path)] and reads.count(str(path)) == 2   # only the drift check reads again


def test_a_conflicting_pin_for_the_same_path_refuses(tmp_path):
    path = tmp_path / "x.json"
    path.write_text('{"a": 1}')
    reader = C.Reader()
    reader.json(path, C.sha256(path.read_bytes()), "first")
    with pytest.raises(C.Refused, match="was captured at"):
        reader.json(path, "00" * 32, "second")
    with pytest.raises(C.Refused, match="is referenced as"):
        reader.capture(path, C.sha256(path.read_bytes()), "markdown", "third")


@pytest.mark.parametrize("want", [None, True, "", "XYZ", "AB" * 32, 12])
def test_a_capture_without_a_valid_pin_refuses_before_any_read(tmp_path, monkeypatch, want):
    path = tmp_path / "x.json"
    path.write_text('{"a": 1}')
    reads = _counting(monkeypatch)
    with pytest.raises(C.Refused) as refused:
        C.Reader().json(path, want, "x")
    # audit 837: the pre-read contract is asserted directly, before the diagnostic text
    assert reads == [], f"the file was read before the pin was validated: {reads}"
    assert "no valid expected digest" in str(refused.value)
