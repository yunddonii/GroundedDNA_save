"""Admission of the anchor model's complete stage-T result set for downstream tables (proposal v2;
audits 798, 824, 831, 833). Metadata only: it reads JSON, JSONL and Markdown bytes, never a checkpoint,
NPZ, config or array, and never imports a producer.

What is admitted, in order (docs/ANCHOR_DOWNSTREAM_CONSUMER_PROPOSAL_v2.md):
  1. the consumer input manifest at the digest this source generation pins (none yet: refuses);
  2. the immutable approved audit-section extracts it names, each holding its exact line (section 830
     launch, 788 stopped T), never today's growing ledger compared with a historical ledger hash;
  3. the settled parent ledger at its pin, and the recovery's OWN supervisor ledger: exactly one start
     and one final, the exact start rules and argv, and the settlement arithmetic;
  4. the recovery receipt, snapshot, reservation and claim, joined by request, nonce and file digests;
  5. twelve cells through their own record -> attempt -> entry lineage (11 carried r7, 1 executed r8);
  6. per cell, on demand, the pinned raw/BIO evaluation, pairwise NMI and DB extraction manifest, with
     their input bindings joined to the cell and completion, and the full-DB denominator.

Every file is captured ONCE (hashed against its pin, parsed from that capture); the only second read is
the final no-drift check before a bundle's receipt. A refusal raises Refused with its reason.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import re
from pathlib import Path

INPUTS_SCHEMA = "anchor-consumer-inputs/1"
T_RECORD_SCHEMA = "anchor-terminal-test-record/1"
T_ATTEMPT_SCHEMA = "anchor-terminal-test-attempt/1"
RECEIPT_SCHEMA = "anchor-terminal-test-recovery-receipt/1"
SNAPSHOT_SCHEMA = "anchor-terminal-test-recovery-snapshot/1"
CLAIM_SCHEMA = "anchor-terminal-test-recovery-claim/1"
LEDGER_SCHEMA = "anchor-confirm-ops-ledger/1"
LINEAGE_SCHEMA = "anchor-t-recovery-lineage/1"
EXTRACTION_MANIFEST_SCHEMA_VERSION = 2
DATASETS = ("cifar10", "flickr25k", "nuswide", "mscoco")
SEEDS = (42, 43, 44)
SLOTS = 5
CANONICAL = {"cifar10": "CIFAR10", "flickr25k": "Flickr25k", "nuswide": "NUSWIDE", "mscoco": "MSCOCO"}
SHA = re.compile(r"[0-9a-f]{64}")

R8 = "/data/yschoi/gdna_anchor_refit_v9r8"
R7 = "/data/yschoi/gdna_anchor_refit_v9r6"
LAUNCH_LINE = ("ANCHOR-CONFIRM-APPROVAL version=anchor-confirm/2 scope=stage-T-recovery "
               "manifest=3537e297c0429c8ab5b02428651025903765a48c3c126eea172f30c37955c9f8 "
               "freeze=5165f5dc9fcfb8334270bc16aa9816d09db67b03a04abae7ff846d5235bdca1d "
               "request=d34f505b90db1ef02dce74bd0326350053a6708315f505c0c881fe89a91ad666")
STOPPED_LINE = ("ANCHOR-CONFIRM-APPROVAL version=anchor-confirm/2 scope=stage-T-run "
                "manifest=2f24fc80bb83e2c73b17ec473c2731ade6d34a356e02414e6719fe9e9bec947c "
                "freeze=5165f5dc9fcfb8334270bc16aa9816d09db67b03a04abae7ff846d5235bdca1d "
                "request=a74b68e1db10c0d71d911ce491a25a2fa97dda6258c88d9f2c063223e1b40612")
PY = "/home/yschoi/.conda/envs/dna_hashing/bin/python"
R8_MANIFEST = f"{R8}/artifacts/anchor_confirmation/authority_manifest_v9r8.json"


class Refused(RuntimeError):
    """The inputs are not the complete, settled, lineage-bound result set this consumer admits."""


def need(condition, message: str) -> None:
    if not condition:
        raise Refused(message)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_digest(obj) -> str:
    """The launcher's semantic digest (phase3_selection_matrix._json_digest)."""
    return sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode())


def is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _no_constant(name):
    raise ValueError(f"non-finite JSON constant {name}")


def strict_json(data: bytes, what: str):
    try:
        return json.loads(data.decode("utf-8"), parse_constant=_no_constant)
    except (ValueError, UnicodeDecodeError) as error:
        raise Refused(f"{what} is not strict JSON: {error}") from None


@dataclasses.dataclass(frozen=True)
class Authority:
    """Every identity this source generation pins. Tests replace it with a synthetic world's."""
    inputs_sha256: str | None = None            # the reviewed post-settlement input manifest: not yet pinned
    freeze: tuple = ("/data/yschoi/gdna_anchor_lambda_v8/artifacts/anchor_confirmation/ancF_candidate_v1.json",
                     "5165f5dc9fcfb8334270bc16aa9816d09db67b03a04abae7ff846d5235bdca1d")
    refit_receipt: tuple = (f"{R7}/artifacts/anchor_confirmation/ancR9_sweep_complete.json",
                            "2897aa880dddd07cbb0aeb49ef090b7e5ccc06e6e9b540922a9ca77a30d81d4a")
    lineage: tuple = (f"{R8}/artifacts/anchor_confirmation/anchor_t_recovery_lineage_v1.json",
                      "b767601961c489c469c8822988dcb66b9dc525f4539ef428456a2f8545ce8bda")
    parent_ledger: tuple = ("/home/yschoi/gdna_anchorRT_ops/device_budget_ledger.jsonl",
                            "316915141fdfa215940014dbeb5efb63ba2b9e98bb84b49146cab69a96fd25bf")
    parent_charged: float = 79482.14623009507
    parent_last_run_id: str = "20261006T145139Z-b08e4ae2"
    stopped_namespace: str = "ancT9"
    stopped_request_sha256: str = "a74b68e1db10c0d71d911ce491a25a2fa97dda6258c88d9f2c063223e1b40612"
    stopped_nonce: str = "f6234d0efc0bc527f718c3cba408cc43382d090e205ac54b7529127bddbfcfc0"
    stopped_section: int = 788
    stopped_line: str = STOPPED_LINE
    request_sha256: str = "d34f505b90db1ef02dce74bd0326350053a6708315f505c0c881fe89a91ad666"
    namespace: str = "ancT9r"
    recovery_cell: str = "nuswide|N=4|P=0.4,0.8|JD=0.05|stage=refit|seed=44|axis_center=anchors"
    recovery_attempt_sha256: str = "d626f950bcdc5a9131c317c23ecbe766e12913fc425c1c6a5d1d4cf66ac44143"
    recovery_entry_sha256: str = "7689d689c7cfe582ff559266ad837f6a0e89ed0799020a8437860923cb3a3f78"
    claim_root: str = "/home/yschoi/gdna_anchorRT_recovery_claims"
    claim_key: str = "dc4a8bc061c5d286d74e21b158091710c165737403e1c85e957f810740f92c6d"
    historical_record_dir: str = f"{R7}/artifacts/anchor_confirmation"
    record_dir: str = f"{R8}/artifacts/anchor_confirmation"
    manifest_sha256: str = "3537e297c0429c8ab5b02428651025903765a48c3c126eea172f30c37955c9f8"
    supervisor_sha256: str = "4732a99eaced36f655ff6fb3204580ef85036ea35a7ea13f85373bf4d72c0f6c"
    launch_section: int = 830
    launch_line: str = LAUNCH_LINE
    expected_command: tuple = (PY, "scripts/phase3_selection_matrix.py", "--anchor-confirm", "recover",
                               "--namespace", "ancT9r", "--anchor-manifest", R8_MANIFEST,
                               "--anchor-manifest-sha256",
                               "3537e297c0429c8ab5b02428651025903765a48c3c126eea172f30c37955c9f8",
                               "--run", "--gpus", "0", "--anchor-approval-section", "830")
    budget_seconds: float = 15000.0
    ceiling_seconds: float = 94482.14623009507
    rules: tuple = (("budget_seconds", 15000.0), ("gpus", 1), ("planned_cells", 1), ("children_per_cell", 5),
                    ("planned_attempts", 5), ("attempts", "sessions"), ("poll_seconds", 1.0),
                    ("watchdog_seconds", 10.0), ("stop_bound_seconds", 120.0), ("headroom_seconds", 130.0),
                    ("wall_limit_seconds", 28800.0), ("free_floor_bytes", 10 << 30),
                    ("cell_output_bytes", 3 << 28))
    n_by_dataset: tuple = (("cifar10", 4), ("flickr25k", 4), ("nuswide", 4), ("mscoco", 39))
    codebook_size: tuple = (("cifar10", 64), ("flickr25k", 128), ("nuswide", 128), ("mscoco", 128))
    #: full-DB rows and mAP@R cutoffs: the accepted literals PAPER_SPLIT_ROWS[*]["db"] and MAP_R_CUTOFF of
    #: scripts/phase3_selection_matrix.py SHA256 d80924365656f5286ce834a89525c2cca8a77349b2d3b093e7ec33ea68222c9e
    #: (r8 manifest 3537e297; audit 834), copied, never imported; the DB counts agree with audit 554's full-DB recount
    db_rows: tuple = (("cifar10", 59000), ("flickr25k", 23000), ("nuswide", 193734), ("mscoco", 107218))
    r_cutoff: tuple = (("cifar10", 1000), ("flickr25k", 5000), ("nuswide", 5000), ("mscoco", 5000))


def is_sha(x) -> bool:
    return isinstance(x, str) and SHA.fullmatch(x) is not None


class Reader:
    """One capture per file (audit 835.2): the first read is cached under the normalized path and every
    later reference is checked against it and parsed from those same bytes; recheck() is the only second
    read. Every capture needs an expected lowercase 64-hex pin from the authority chain (audit 835.1)."""

    def __init__(self):
        self.consumed, self._bytes = {}, {}

    def capture(self, path, want, kind: str, what: str) -> bytes:
        need(is_sha(want), f"{what}: no valid expected digest ({want!r}); the pin is required, never optional")
        p = os.path.abspath(str(path))
        if p in self._bytes:
            row = self.consumed[p]
            need(row["sha256"] == want, f"{what}: {p} was captured at {row['sha256']}, now pinned {want}")
            need(row["kind"] == kind, f"{what}: {p} is referenced as {kind} and {row['kind']}")
            return self._bytes[p]
        try:
            data = Path(p).read_bytes()
        except OSError as error:
            raise Refused(f"{what}: {p} is unreadable ({type(error).__name__})") from None
        got = sha256(data)
        need(got == want, f"{what}: {p} is {got}, pinned {want}")
        self.consumed[p] = {"sha256": got, "kind": kind}
        self._bytes[p] = data
        return data

    def json(self, path, want, what: str):
        obj = strict_json(self.capture(path, want, "json", what), what)
        need(isinstance(obj, dict), f"{what} is not a JSON object")
        return obj

    def recheck(self) -> list:
        drift = []
        for p, row in sorted(self.consumed.items()):
            try:
                if sha256(Path(p).read_bytes()) != row["sha256"]:
                    drift.append(p)
            except OSError:
                drift.append(p)
        return drift


def _pin(block, what):
    need(isinstance(block, dict) and isinstance(block.get("path"), str) and SHA.fullmatch(str(block.get("sha256"))),
         f"{what} is not a {{path, sha256}} pin")
    return block["path"], block["sha256"]


def _section_extract(reader, block, section: int, line: str, what: str) -> None:
    """An immutable approved section extract: its first line is the section heading, and it holds the exact
    approval line as a whole line (audit 833.2 item 4)."""
    need(isinstance(block, dict) and block.get("section") == section and block.get("line") == line,
         f"{what}: the manifest does not name section {section} with its exact line")
    path, digest = _pin(block.get("extract"), f"{what} extract")
    text = reader.capture(path, digest, "markdown", f"{what} extract").decode("utf-8")
    lines = text.splitlines()
    need(bool(lines) and lines[0].startswith(f"## {section}. "), f"{what}: the extract is not section {section}")
    need(sum(1 for x in lines if x == line) == 1, f"{what}: section {section} does not hold the exact line once")
    need(not any(x.startswith("## ") for x in lines[1:]), f"{what}: the extract runs into another section")


def _jsonl(reader, path, digest, what) -> list:
    raw = reader.capture(path, digest, "jsonl", what)
    rows = []
    for number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        row = strict_json(line.encode(), f"{what} line {number}")
        need(isinstance(row, dict) and row.get("schema") == LEDGER_SCHEMA, f"{what} line {number} is not a ledger row")
        rows.append(row)
    need(rows, f"{what} is empty")
    return rows


def _parent(reader, auth: Authority, block) -> dict:
    path, digest = _pin(block, "parent ledger")
    need((path, digest) == tuple(auth.parent_ledger), "the parent ledger is not the settled pinned ledger")
    rows = _jsonl(reader, path, digest, "parent ledger")
    starts = {r.get("run_id") for r in rows if r.get("event") == "start"}
    finals = {r.get("run_id"): r for r in rows if r.get("event") == "final"}
    last = rows[-1]
    need(last.get("event") == "final" and starts == set(finals) and last.get("run_id") == auth.parent_last_run_id
         and last.get("cumulative_charged_seconds") == auth.parent_charged,
         "the parent ledger does not end settled at its pinned charge")
    return {"sha256": digest, "charged_seconds": auth.parent_charged, "last_run_id": last["run_id"]}


def _recovery_settlement(reader, auth: Authority, block, run_id: str) -> dict:
    """Audit 824.1/831.2.4/833.2.2: the recovery's own supervisor run, not the receipt's settlement field."""
    path, digest = _pin(block, "recovery ledger")
    rows = _jsonl(reader, path, digest, "recovery ledger")
    need(all(r.get("run_id") == run_id for r in rows), "the recovery ledger holds another run")
    starts = [r for r in rows if r.get("event") == "start"]
    finals = [r for r in rows if r.get("event") == "final"]
    need(len(starts) == 1 and len(finals) == 1, "the recovery ledger has no single start and single final")
    need(rows[0] is starts[0] and rows[-1] is finals[0], "the recovery ledger has rows before its start or after its final")
    start, final = starts[0], finals[0]
    need(start.get("stage") == "stage-T-recovery" and start.get("label") == "", "the recovery run is not stage-T-recovery")
    need(start.get("refused") is None, "the recovery start was refused")
    need(start.get("manifest_sha256") == auth.manifest_sha256, "the recovery start names another manifest")
    need(start.get("supervisor_sha256") == auth.supervisor_sha256, "the recovery start names another supervisor source")
    need(start.get("parent") == {"ledger": auth.parent_ledger[0], "sha256": auth.parent_ledger[1],
                                 "charged_seconds": auth.parent_charged, "last_run_id": auth.parent_last_run_id},
         "the recovery start does not bind the settled parent")
    need(start.get("prior_charged_seconds") == 0.0 and not isinstance(start.get("prior_charged_seconds"), bool),
         "the recovery ledger had a prior charge")
    rules = start.get("rules")
    need(isinstance(rules, dict), "the recovery start records no rules")
    for key, value in auth.rules:
        got = rules.get(key)
        need(type(got) is type(value) and got == value, f"the recovery rule {key} is {got!r}, not {value!r}")
    command = start.get("command")
    need(isinstance(command, list) and command == list(auth.expected_command),
         "the recovery start command is not the audited argv")
    need(start.get("command_sha256") == sha256(json.dumps(list(command)).encode()),
         "the recovery start command digest is not the supervisor's serialisation of its argv")
    # the final settlement
    need(final.get("status") == "exited", f"the recovery final status is {final.get('status')!r}")
    need(is_int(final.get("returncode")) and final["returncode"] == 0, "the recovery returned nonzero")
    for key in ("continuity_lost", "monitor_failures", "orphaned_live_attempts", "leases_held_after_exit"):
        need(final.get(key) == [], f"the recovery final has {key}")
    numbers = ("device_seconds", "max_observation_window_seconds", "unobserved_allowance_seconds",
               "charged_seconds", "cumulative_charged_seconds", "parent_charged_seconds",
               "cumulative_including_parent_seconds", "wall_seconds")
    for key in numbers:
        need(is_num(final.get(key)) and final[key] >= 0, f"the recovery final {key} is not a finite nonnegative number")
    attempts = final.get("attempts")
    need(isinstance(attempts, list) and 0 < len(attempts) <= dict(auth.rules)["planned_attempts"],
         "the recovery final has no attempts or more than planned")
    total = 0.0
    for row in attempts:
        need(isinstance(row, dict) and all(is_num(row.get(k)) and row[k] >= 0
                                           for k in ("start_boot", "end_boot", "seconds")),
             "a recovery attempt has a non-finite or negative time")
        need(row["end_boot"] >= row["start_boot"] and abs(row["end_boot"] - row["start_boot"] - row["seconds"]) <= 1e-6,
             "a recovery attempt's seconds do not equal its end minus start")
        total += row["seconds"]
    rule = dict(auth.rules)
    tol = 1e-6
    need(abs(final["device_seconds"] - total) <= tol, "the recovery device seconds are not the attempt sum")
    need(final["max_observation_window_seconds"] <= rule["watchdog_seconds"],
         "the recovery observation window exceeds the watchdog bound")
    need(abs(final["unobserved_allowance_seconds"] - rule["planned_attempts"] * final["max_observation_window_seconds"]) <= tol,
         "the recovery allowance is not planned attempts x the longest observation window")
    need(abs(final["charged_seconds"] - final["device_seconds"] - final["unobserved_allowance_seconds"]) <= tol,
         "the recovery charge is not device time plus allowance")
    need(abs(final["cumulative_charged_seconds"] - final["charged_seconds"]) <= tol,
         "the recovery cumulative charge is not its own charge")
    need(final["parent_charged_seconds"] == auth.parent_charged, "the recovery final names another parent charge")
    need(abs(final["cumulative_including_parent_seconds"] - auth.parent_charged - final["cumulative_charged_seconds"]) <= tol,
         "the recovery cumulative including the parent does not add up")
    need(final["cumulative_charged_seconds"] <= auth.budget_seconds, "the recovery exceeded its budget")
    need(final["cumulative_including_parent_seconds"] <= auth.ceiling_seconds, "the recovery exceeded the ceiling")
    need(final["wall_seconds"] <= rule["wall_limit_seconds"], "the recovery exceeded its wall limit")
    return {"run_id": run_id, "sha256": digest, "status": "exited", "charged_seconds": final["charged_seconds"],
            "cumulative_including_parent_seconds": final["cumulative_including_parent_seconds"]}


def admit(auth: Authority, inputs_path, reader: Reader) -> dict:
    """The whole chain; returns the twelve admitted cells keyed by (dataset, seed)."""
    need(auth.inputs_sha256 is not None and SHA.fullmatch(auth.inputs_sha256),
         "no reviewed consumer input manifest is pinned in this source generation")
    inputs = reader.json(inputs_path, auth.inputs_sha256, "consumer input manifest")
    need(inputs.get("schema") == INPUTS_SCHEMA, "the consumer input manifest has another schema")
    _section_extract(reader, inputs.get("launch"), auth.launch_section, auth.launch_line, "launch approval")
    _section_extract(reader, inputs.get("stopped_approval"), auth.stopped_section, auth.stopped_line,
                     "stopped-T approval")
    freeze = reader.json(auth.freeze[0], auth.freeze[1], "F")
    refit = reader.json(auth.refit_receipt[0], auth.refit_receipt[1], "stage-R receipt")
    lineage = reader.json(auth.lineage[0], auth.lineage[1], "recovery lineage")
    need(lineage.get("schema") == LINEAGE_SCHEMA and isinstance(lineage.get("carried"), dict)
         and len(lineage["carried"]) == 11, "the lineage is not the eleven-carried stopped campaign")
    need((lineage.get("recovery_cell") or {}).get("cell_id") == auth.recovery_cell
         and lineage["recovery_cell"].get("attempt", {}).get("sha256") == auth.recovery_attempt_sha256
         and lineage["recovery_cell"].get("entry", {}).get("sha256") == auth.recovery_entry_sha256,
         "the lineage names another recovered cell")
    parent = _parent(reader, auth, inputs.get("parent_ledger"))
    rec = inputs.get("recovery")
    need(isinstance(rec, dict) and isinstance(rec.get("run_id"), str), "the input manifest names no recovery run")
    settlement = _recovery_settlement(reader, auth, rec.get("ledger"), rec["run_id"])
    # receipt, snapshot, reservation, claim
    rpath, rsha = _pin(rec.get("receipt"), "recovery receipt")
    need(rpath == os.path.join(auth.record_dir, f"{auth.namespace}_recovery_complete.json"),
         "the recovery receipt is not at its contracted path")
    receipt = reader.json(rpath, rsha, "recovery receipt")
    need(receipt.get("schema") == RECEIPT_SCHEMA and receipt.get("mode") == "recovery"
         and receipt.get("namespace") == auth.namespace and receipt.get("request_sha256") == auth.request_sha256,
         "the recovery receipt is not this recovery's")
    nonce = receipt.get("campaign_nonce")
    need(isinstance(nonce, str) and SHA.fullmatch(nonce), "the recovery receipt has no campaign nonce")
    need(receipt.get("refit_receipt") == {"path": auth.refit_receipt[0], "sha256": auth.refit_receipt[1]}
         and receipt.get("lineage") == {"path": auth.lineage[0], "sha256": auth.lineage[1]},
         "the recovery receipt binds another R receipt or lineage")
    stopped = receipt.get("stopped") or {}
    need(stopped.get("namespace") == auth.stopped_namespace and stopped.get("request_sha256") == auth.stopped_request_sha256
         and (stopped.get("approval") or {}).get("section") == auth.stopped_section
         and (stopped.get("approval") or {}).get("line") == auth.stopped_line
         and (stopped.get("settlement") or {}).get("ledger_sha256") == auth.parent_ledger[1],
         "the recovery receipt names another stopped campaign")
    recovery_of = receipt.get("recovery_of") or {}
    need(recovery_of.get("cell_id") == auth.recovery_cell
         and (recovery_of.get("attempt") or {}).get("sha256") == auth.recovery_attempt_sha256
         and (recovery_of.get("entry") or {}).get("sha256") == auth.recovery_entry_sha256,
         "the recovery receipt recovers another cell")
    spath, ssha = _pin(rec.get("snapshot"), "recovery snapshot")
    need(os.path.dirname(spath) == auth.record_dir and os.path.basename(spath) == receipt.get("plan_snapshot_file"),
         "the recovery snapshot is not the receipt's")
    snapshot = reader.json(spath, ssha, "recovery snapshot")
    need(json_digest(snapshot) == receipt.get("plan_snapshot_sha256")
         and os.path.basename(spath) == f"{auth.namespace}_snapshot_{json_digest(snapshot)[:16]}.json",
         "the recovery snapshot is not at the digest its receipt binds")
    plan = snapshot.get("plan") or {}
    need(snapshot.get("schema") == SNAPSHOT_SCHEMA and plan.get("namespace") == auth.namespace
         and plan.get("campaign_nonce") == nonce and plan.get("declared_count") == 1 and plan.get("executed_count") == 1
         and json_digest(snapshot.get("request")) == auth.request_sha256
         and snapshot.get("request_sha256") == auth.request_sha256
         and (snapshot.get("approval") or {}).get("section") == auth.launch_section
         and (snapshot.get("approval") or {}).get("line") == auth.launch_line,
         "the recovery snapshot does not bind this request, nonce and approval")
    vpath, vsha = _pin(rec.get("reservation"), "recovery reservation")
    need(vpath == os.path.join(auth.record_dir, receipt.get("campaign_reservation_file") or "-")
         and vsha == receipt.get("campaign_reservation_sha256"), "the recovery reservation is not the receipt's")
    reservation = reader.json(vpath, vsha, "recovery reservation")
    need(reservation.get("namespace") == auth.namespace and reservation.get("campaign_nonce") == nonce
         and reservation.get("plan_digest") == receipt.get("plan_snapshot_sha256")
         and reservation.get("plan_snapshot_file") == receipt.get("plan_snapshot_file"),
         "the recovery reservation does not bind this snapshot and nonce")
    cpath, csha = _pin(rec.get("claim"), "recovery claim")
    need(cpath == os.path.join(auth.claim_root, f"{auth.claim_key}.json"), "the claim is not at its lineage key")
    claim_row = receipt.get("recovery_claim") or {}
    need(claim_row == {"file": f"{auth.claim_key}.json", "root": auth.claim_root, "sha256": csha},
         "the receipt binds another claim")
    claim = reader.json(cpath, csha, "recovery claim")
    need(claim.get("schema") == CLAIM_SCHEMA and claim.get("claim_key") == auth.claim_key
         and claim.get("request_sha256") == auth.request_sha256 and claim.get("namespace") == auth.namespace
         and claim.get("record_dir") == auth.record_dir
         and claim.get("approval") == {"section": auth.launch_section, "scope": "stage-T-recovery",
                                       "line": auth.launch_line}
         and claim.get("lineage") == {"path": auth.lineage[0], "sha256": auth.lineage[1]},
         "the claim does not bind this request, approval and lineage")
    cells = _cells(reader, auth, inputs, receipt, lineage, refit, freeze, nonce, claim)
    return {"cells": cells, "settlement": settlement, "parent": parent, "receipt_sha256": rsha,
            "inputs_sha256": auth.inputs_sha256}


def _cells(reader, auth, inputs, receipt, lineage, refit, freeze, nonce, claim) -> dict:
    rows = inputs.get("cells")
    need(isinstance(rows, list) and len(rows) == 12, "the input manifest does not list twelve cells")
    rcells = receipt.get("cells")
    need(isinstance(rcells, dict) and len(rcells) == 12 and receipt.get("expected_cells") == 12
         and receipt.get("cell_count") == 12, "the receipt does not hold twelve cells")
    executed = (receipt.get("final_closure") or {}).get("executed") or {}
    n_of, admitted = dict(auth.n_by_dataset), {}
    seen_ids = set()
    for row in rows:
        need(isinstance(row, dict), "an input-manifest cell is not an object")
        cid, origin = row.get("cell_id"), row.get("origin")
        need(isinstance(cid, str) and cid not in seen_ids, f"cell {cid!r} is missing or duplicated")
        seen_ids.add(cid)
        carried = origin == "carried"
        need(origin in ("carried", "executed"), f"{cid}: origin {origin!r}")
        need((cid == auth.recovery_cell) == (not carried), f"{cid}: the executed cell must be exactly the recovered one")
        ns = auth.stopped_namespace if carried else auth.namespace
        root = auth.historical_record_dir if carried else auth.record_dir
        rec_p, rec_s = _pin(row.get("record"), f"{cid} record")
        att_p, att_s = _pin(row.get("attempt"), f"{cid} attempt")
        ent_p, ent_s = _pin(row.get("entry"), f"{cid} entry")
        need(all(os.path.dirname(p) == root for p in (rec_p, att_p, ent_p)), f"{cid}: a file is outside its origin's record root")
        rr = rcells.get(cid) or {}
        need(rr.get("origin") == origin and rr.get("namespace") == ns and rr.get("record_dir") == root
             and rr.get("record") == os.path.basename(rec_p) and rr.get("record_sha256") == rec_s
             and rr.get("attempt_sha256") == att_s and rr.get("entry_sha256") == ent_s,
             f"{cid}: the receipt row is not this cell's lineage")
        if carried:
            pinned = lineage["carried"].get(cid) or {}
            need({k: (pinned.get(k) or {}).get("sha256") for k in ("record", "attempt", "entry")}
                 == {"record": rec_s, "attempt": att_s, "entry": ent_s}
                 and {k: (pinned.get(k) or {}).get("file") for k in ("record", "attempt", "entry")}
                 == {"record": os.path.basename(rec_p), "attempt": os.path.basename(att_p),
                     "entry": os.path.basename(ent_p)}, f"{cid}: not the lineage's carried triple")
        else:
            need(executed.get("record") == {"file": os.path.basename(rec_p), "sha256": rec_s}
                 and executed.get("attempt") == {"file": os.path.basename(att_p), "sha256": att_s}
                 and executed.get("entry") == {"file": os.path.basename(ent_p), "sha256": ent_s},
                 f"{cid}: not the receipt's executed triple")
        record = reader.json(rec_p, rec_s, f"{cid} T record")
        cell, completion = record.get("cell") or {}, record.get("completion") or {}
        _require_fields(cell, completion, cid)
        tag = cell.get("tag")
        need(isinstance(tag, str) and os.path.basename(rec_p) == f"{ns}_{tag}.json"
             and os.path.basename(att_p) == f"{ns}_attempt_{tag}.json"
             and os.path.basename(ent_p) == f"{ns}_entry_{tag}.json", f"{cid}: file names are not the contracted ones")
        req_sha = auth.stopped_request_sha256 if carried else auth.request_sha256
        want_nonce = auth.stopped_nonce if carried else nonce
        need(record.get("schema") == T_RECORD_SCHEMA and record.get("stage") == "test" and record.get("namespace") == ns
             and record.get("request_sha256") == req_sha and record.get("campaign_nonce") == want_nonce
             and record.get("attempt") == {"file": os.path.basename(att_p), "sha256": att_s}
             and cell.get("cell_id") == cid, f"{cid}: the T record does not bind its request, nonce and attempt")
        attempt = reader.json(att_p, att_s, f"{cid} attempt")
        approval = attempt.get("approval") or {}
        want_section, want_line = ((auth.stopped_section, auth.stopped_line) if carried
                                   else (auth.launch_section, auth.launch_line))
        need(attempt.get("schema") == T_ATTEMPT_SCHEMA and attempt.get("namespace") == ns
             and attempt.get("campaign_nonce") == want_nonce and attempt.get("request_sha256") == req_sha
             and json_digest(attempt.get("request")) == req_sha and attempt.get("cell") == cell
             and approval.get("section") == want_section and approval.get("line") == want_line,
             f"{cid}: the attempt does not bind this request, nonce, cell and approval")
        if not carried:
            need(os.path.basename(att_p) == claim.get("attempt"), f"{cid}: the claim authorizes another attempt")
        entry = reader.json(ent_p, ent_s, f"{cid} entry")
        need(entry.get("attempt") == os.path.basename(att_p) and entry.get("attempt_sha256") == att_s
             and entry.get("cell_id") == cid and entry.get("final_checkpoint_sha256") == cell.get("final_checkpoint_sha256")
             and entry.get("config_pt_sha256") == cell.get("config_pt_sha256"), f"{cid}: the entry does not bind its attempt")
        ds, seed = cell.get("dataset"), cell.get("seed")
        need(ds in DATASETS and is_int(seed) and seed in SEEDS, f"{cid}: coordinate {ds}/{seed} is outside the design")
        need(is_int(cell.get("N")) and cell["N"] == n_of[ds], f"{cid}: N {cell.get('N')!r} is not F's {n_of[ds]}")
        fds = (freeze.get("datasets") or {}).get(ds) or {}
        need(fds.get("N") == n_of[ds] and fds.get("axis_center") == "anchors", f"{cid}: F does not freeze this N")
        rcell = (refit.get("cells") or {}).get(cid) or {}
        need(rcell.get("record") == os.path.basename(str(cell.get("record"))) and rcell.get("record_sha256") == cell.get("record_sha256")
             and rcell.get("seed") == seed and rcell.get("run_dir") == cell.get("run_dir"),
             f"{cid}: the stage-R record is not the one the R receipt names")
        reader.json(cell["record"], cell["record_sha256"], f"{cid} stage-R record")
        key = (ds, seed)
        need(key not in admitted, f"{cid}: duplicate coordinate {key}")
        admitted[key] = {"cell_id": cid, "origin": origin, "record": rec_p, "cell": cell, "completion": completion}
    need(sorted(admitted) == sorted((d, s) for d in DATASETS for s in SEEDS), "the twelve coordinates are not complete")
    need(set(rcells) == seen_ids, "the receipt and the input manifest list different cells")
    return admitted


SPLITS = ("db", "query", "train")


def _require_fields(cell: dict, completion: dict, cid: str) -> None:
    """Every field a later join compares must be present and well typed, so that two missing values can
    never satisfy an equality (audit 835.1)."""
    for key in ("cell_id", "tag", "dataset", "run_dir", "record", "final_checkpoint"):
        need(isinstance(cell.get(key), str) and cell[key], f"{cid}: cell.{key} is missing or not a string")
    need(os.path.isabs(cell["run_dir"]) and os.path.isabs(cell["record"]), f"{cid}: cell paths are not absolute")
    for key in ("seed", "N", "terminal_epoch"):
        need(is_int(cell.get(key)), f"{cid}: cell.{key} is not an integer")
    for key in ("final_checkpoint_sha256", "config_pt_sha256", "record_sha256"):
        need(is_sha(cell.get(key)), f"{cid}: cell.{key} is not a 64-hex digest")
    for key in ("evaluation_sha256", "bio_evaluation_sha256", "pairwise_nmi_sha256"):
        need(is_sha(completion.get(key)), f"{cid}: completion.{key} is not a 64-hex digest")
    for key in ("npz_sha256", "extraction_manifest_sha256"):
        block = completion.get(key)
        need(isinstance(block, dict) and sorted(block) == sorted(SPLITS) and all(is_sha(block[s]) for s in SPLITS),
             f"{cid}: completion.{key} is not a db/query/train map of 64-hex digests")


def _binding_join(auth, cell: dict, completion: dict, binding, what: str) -> dict:
    need(isinstance(binding, dict), f"{what} has no input_binding")
    ds = cell["dataset"]
    want = {"schema_version": 1, "run_dir": os.path.abspath(cell["run_dir"]), "dataset": CANONICAL[ds],
            "random_seed": cell["seed"], "inference_epoch": cell.get("terminal_epoch"),
            "checkpoint_sha256": cell.get("final_checkpoint_sha256"), "config_sha256": cell.get("config_pt_sha256"),
            "npz_sha256": completion.get("npz_sha256"), "manifest_sha256": completion.get("extraction_manifest_sha256"),
            "codebook_size": dict(auth.codebook_size)[ds], "backfilled_inputs": False}
    for key, value in want.items():
        got = binding.get(key)
        need(type(got) is type(value) and got == value, f"{what}: input_binding {key} is {got!r}, not {value!r}")
    return binding


def cell_inputs(auth: Authority, reader: Reader, admitted_cell: dict) -> dict:
    """The pinned metric JSONs of one cell and its full-DB denominator, joined to the cell (metadata only)."""
    cell, comp = admitted_cell["cell"], admitted_cell["completion"]
    ds, seed = cell["dataset"], cell["seed"]
    what = f"{ds}/seed{seed}"
    run = Path(cell["run_dir"])
    raw = reader.json(run / "evaluation_siglip2_base.json", comp.get("evaluation_sha256"), f"{what} raw evaluation")
    bio = reader.json(run / "evaluation_siglip2_base_bioproj.json", comp.get("bio_evaluation_sha256"),
                      f"{what} BIO evaluation")
    nmi = reader.json(run / "pairwise_nmi.json", comp.get("pairwise_nmi_sha256"), f"{what} pairwise NMI")
    db = reader.json(run / "extraction_manifest_db.json", (comp.get("extraction_manifest_sha256") or {}).get("db"),
                     f"{what} DB extraction manifest")
    bindings = [_binding_join(auth, cell, comp, x.get("input_binding"), f"{what} {name}")
                for name, x in (("raw evaluation", raw), ("BIO evaluation", bio), ("pairwise NMI", nmi))]
    need(bindings[0] == bindings[1] == bindings[2], f"{what}: the raw, BIO and NMI input bindings differ")
    rows = dict(auth.db_rows)[ds]
    need(db.get("schema_version") == EXTRACTION_MANIFEST_SCHEMA_VERSION and db.get("split") == "db"
         and is_int(db.get("n_rows")) and db["n_rows"] > 0, f"{what}: the DB manifest is not a schema-2 DB split")
    need(db["n_rows"] == rows, f"{what}: DB rows {db['n_rows']} are not the reviewed full DB {rows}")
    for key, value in (("dataset", CANONICAL[ds]), ("random_seed", seed),
                       ("checkpoint_sha256", cell.get("final_checkpoint_sha256")),
                       ("config_sha256", cell.get("config_pt_sha256")), ("inference_epoch", cell.get("terminal_epoch")),
                       ("npz_sha256", (comp.get("npz_sha256") or {}).get("db")), ("backfilled", False)):
        need(type(db.get(key)) is type(value) and db.get(key) == value, f"{what}: DB manifest {key} is {db.get(key)!r}")
    cutoff = dict(auth.r_cutoff)[ds]
    need(raw.get("bio_project") is False and bio.get("bio_project") is True, f"{what}: not a raw/BIO pair")
    for name, x in (("raw", raw), ("BIO", bio)):
        need(is_int(x.get("mAP_R_cutoff")) and x["mAP_R_cutoff"] == cutoff, f"{what}: {name} R cutoff is not {cutoff}")
        need(is_num(x.get("mAP_at_R")) and 0 <= x["mAP_at_R"] <= 1, f"{what}: {name} mAP@R is not a proportion")
        need(is_num(x.get("unique_code_ratio")) and 0 <= x["unique_code_ratio"] <= 1, f"{what}: {name} unique ratio")
    need(is_int(comp.get("map_R_cutoff")) and comp["map_R_cutoff"] == cutoff, f"{what}: the record's R cutoff differs")
    need(is_num(comp.get("map_at_R")) and abs(comp["map_at_R"] - raw["mAP_at_R"]) <= 1e-12,
         f"{what}: the record's stored mAP@R disagrees with the raw evaluation")
    need(is_num(comp.get("bio_map_at_R")) and abs(comp["bio_map_at_R"] - bio["mAP_at_R"]) <= 1e-12,
         f"{what}: the record's stored BIO mAP@R disagrees with the BIO evaluation")
    stats = bio.get("bio_stats")
    need(isinstance(stats, dict), f"{what}: no bio_stats")
    for key in ("mAP_at_R_pre_projection", "db_mean_edit_distance", "db_pre_compliance", "db_post_compliance"):
        need(is_num(stats.get(key)), f"{what}: bio_stats {key} is not finite")
    for key in ("db_num_total", "db_num_proj_failed", "qy_num_proj_failed"):
        need(is_int(stats.get(key)), f"{what}: bio_stats {key} is not an integer")
    need(abs(stats["mAP_at_R_pre_projection"] - raw["mAP_at_R"]) <= 1e-12, f"{what}: BIO pre-projection mAP@R differs")
    need(stats["db_num_proj_failed"] == 0 and stats["qy_num_proj_failed"] == 0, f"{what}: projection failures")
    need(stats["db_post_compliance"] == 1.0, f"{what}: post-projection compliance is not 1.0")
    need(stats["db_num_total"] == db["n_rows"], f"{what}: BIO DB total is not the DB manifest's rows")
    need(0 <= stats["db_mean_edit_distance"] <= 15 and 0 <= stats["db_pre_compliance"] <= 1,
         f"{what}: BIO statistics out of range")
    # pairwise NMI
    m = nmi.get("nmi_matrix")
    need(nmi.get("num_codebooks") == SLOTS and nmi.get("nmi_average_method") == "arithmetic"
         and isinstance(m, list) and len(m) == SLOTS and all(isinstance(r, list) and len(r) == SLOTS for r in m),
         f"{what}: not a 5-slot arithmetic NMI matrix")
    for i in range(SLOTS):
        for j in range(SLOTS):
            need(is_num(m[i][j]) and 0 <= m[i][j] <= 1 and abs(m[i][j] - m[j][i]) <= 1e-12, f"{what}: NMI [{i}][{j}]")
        need(m[i][i] == 1.0, f"{what}: NMI diagonal [{i}] is not 1")
    pairs = [m[i][j] for i in range(SLOTS) for j in range(i + 1, SLOTS)]
    mean = math.fsum(pairs) / len(pairs)
    for key, value in (("mean_off_diag_nmi", mean), ("min_off_diag_nmi", min(pairs)), ("max_off_diag_nmi", max(pairs))):
        need(is_num(nmi.get(key)) and abs(nmi[key] - value) <= 1e-12, f"{what}: NMI {key} disagrees with the matrix")
    need(is_num(comp.get("mean_off_diag_nmi")) and abs(comp["mean_off_diag_nmi"] - mean) <= 1e-12,
         f"{what}: the record's stored NMI mean disagrees with the file")
    need(is_int(nmi.get("N")) and nmi["N"] == db["n_rows"], f"{what}: NMI N is not the DB manifest's rows")
    # audit 834: K_per_cb is the largest observed code index + 1 per slot (pairwise_nmi.py), not capacity;
    # unique_codewords_per_cb is the distinct occupancy. Capacity is bound through input_binding.codebook_size.
    k = dict(auth.codebook_size)[ds]
    need(comp.get("analysis_protocol", {}).get("codebook_size") == k and not isinstance(
        comp.get("analysis_protocol", {}).get("codebook_size"), bool), f"{what}: the record's codebook size is not {k}")
    top, used = nmi.get("K_per_cb"), nmi.get("unique_codewords_per_cb")
    need(isinstance(top, list) and isinstance(used, list) and len(top) == SLOTS and len(used) == SLOTS
         and all(is_int(x) for x in top + used), f"{what}: K_per_cb / unique_codewords_per_cb are not five integers")
    need(all(1 <= u <= t <= k and u <= db["n_rows"] for u, t in zip(used, top)),
         f"{what}: per-slot occupancy is not 1 <= distinct <= max index + 1 <= {k} and <= DB rows")
    return {"raw": raw, "bio": bio, "nmi": nmi, "db_rows": db["n_rows"], "pairs": pairs, "mean_pair_nmi": mean,
            "max_index_plus_one": top, "distinct_codewords": used}


# ----- bundle publication (exclusive, contained, no drift, incomplete kept) -------------------------------
DEFAULT_OUT_ROOT = "/home/yschoi/gdna_anchor_downstream"
FORBIDDEN_ROOTS = ("/home/yschoi/GroundedDNA", R8, R7, "/home/yschoi/gdna_anchor_consumer",
                   "/data/yschoi/gdna_anchor_lambda_v8", "/data/yschoi/gdna_p3exec",
                   "/home/yschoi/gdna_anchorRT_result", "/home/yschoi/gdna_anchorRT_ops",
                   "/home/yschoi/gdna_anchorRTrec_ops", "/home/yschoi/gdna_anchorRT_recovery_claims")


def publish(out_root, todo: str, name: str, files: dict, receipt: dict, reader: Reader, *,
            forbidden=None, before_recheck=None) -> Path:
    forbidden = FORBIDDEN_ROOTS if forbidden is None else forbidden
    root = Path(os.path.realpath(out_root))
    need(re.fullmatch(r"[A-Za-z0-9_.-]+", todo) and re.fullmatch(r"[A-Za-z0-9_.-]+", name) and name not in (".", ".."),
         "bundle names must be plain")
    for bad in forbidden:
        b = os.path.realpath(bad)
        need(not (str(root) == b or str(root).startswith(b + "/") or b.startswith(str(root) + "/")),
             f"the bundle root {root} aliases or contains the protected root {b}")
    parent = root / todo
    parent.mkdir(parents=True, exist_ok=True)
    need(os.path.realpath(parent) == str(parent), "the bundle directory is reached through a symlink")
    dest = parent / name
    try:
        os.mkdir(dest, 0o755)
    except FileExistsError:
        raise Refused(f"{dest} exists; a bundle is written once, to a new path") from None
    try:
        written = {}
        for fname, text in files.items():
            fd = os.open(dest / fname, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            with os.fdopen(fd, "w") as handle:
                handle.write(text)
            written[fname] = sha256(text.encode())
        if before_recheck:
            before_recheck()
        drift = reader.recheck()
        need(not drift, f"inputs changed before publication: {drift[:3]}")
        body = {**receipt, "outputs": written,
                "consumed": {p: r["sha256"] for p, r in sorted(reader.consumed.items())},
                "consumed_kinds": {p: r["kind"] for p, r in sorted(reader.consumed.items())},
                "scope": "formatting of stored values", "independent_numerical_verification": False,
                "paper_result_eligible": False}
        fd = os.open(dest / "bundle_receipt.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(fd, "w") as handle:
            json.dump(body, handle, indent=1, sort_keys=True)
        return dest
    except BaseException as error:
        try:
            fd = os.open(dest / "INCOMPLETE", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            with os.fdopen(fd, "w") as handle:
                handle.write(f"{type(error).__name__}: {error}\n")
        except OSError:
            pass
        raise


def source_identity() -> dict:
    here = Path(__file__).resolve()
    return {"consumer": str(here), "consumer_sha256": sha256(here.read_bytes())}
