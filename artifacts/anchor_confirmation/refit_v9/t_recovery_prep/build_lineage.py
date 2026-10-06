"""Build the pinned recovery lineage of the stopped full T (audits 795, 797.2 item 2), metadata only.

Reads JSON records, the r7 manifest bytes and the settled R/T operations ledger (JSON lines); opens no
checkpoint, config, array or cache. Writes one JSON file once. The r8 launcher pins this file's digest
and re-verifies every member against disk at admission; this script is evidence, not a closure member.
Usage: python build_lineage.py <out.json>
"""
import hashlib
import json
import os
import sys
from pathlib import Path

R7_ROOT = Path("/data/yschoi/gdna_anchor_refit_v9r6")
RECORDS = R7_ROOT / "artifacts" / "anchor_confirmation"
R7_MANIFEST = RECORDS / "authority_manifest_v9r7.json"
LEDGER = Path("/home/yschoi/gdna_anchorRT_ops/device_budget_ledger.jsonl")
RUN_ID = "20261006T145139Z-b08e4ae2"
NAMESPACE = "ancT9"
SNAPSHOT = "ancT9_snapshot_da40a183d4ca9c85.json"
RECOVERY_TAG = "ancR9_nuswide_A_v4_refit_N4_s44_AXanchors_P0408_JD005"


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sem(payload) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def pin(name: str) -> dict:
    return {"file": name, "sha256": sha_file(RECORDS / name)}


def main(out: str) -> int:
    snapshot = json.loads((RECORDS / SNAPSHOT).read_bytes())
    request = snapshot["request"]
    reservation = json.loads((RECORDS / f"{NAMESPACE}_campaign_reservation.json").read_bytes())
    assert reservation["plan_digest"] == sem(snapshot) and reservation["namespace"] == NAMESPACE
    cells = {c["cell_id"]: c for c in request["cells"]}
    assert len(cells) == 12
    carried, recovery = {}, None
    for cid, cell in sorted(cells.items()):
        tag = cell["tag"]
        entry = {"tag": tag, "attempt": pin(f"{NAMESPACE}_attempt_{tag}.json"),
                 "entry": pin(f"{NAMESPACE}_entry_{tag}.json")}
        record = RECORDS / f"{NAMESPACE}_{tag}.json"
        if tag == RECOVERY_TAG:
            assert not record.exists()
            recovery = {"cell_id": cid, **entry}
        else:
            entry["record"] = pin(record.name)
            carried[cid] = entry
    assert recovery is not None and len(carried) == 11
    raw = LEDGER.read_bytes()
    lines = raw.decode("utf-8").splitlines()
    rows = [json.loads(line) for line in lines]
    mine = [i for i, r in enumerate(rows) if r.get("run_id") == RUN_ID and r["event"] in ("start", "stop", "final")]
    kinds = [rows[i]["event"] for i in mine]
    assert kinds == ["start", "stop", "final"] and mine[-1] == len(rows) - 1
    stop, final = rows[mine[1]], rows[mine[2]]
    lineage = {
        "schema": "anchor-t-recovery-lineage/1",
        "note": "the stopped full-T campaign the r8 recovery admits; every member is re-verified on disk",
        "historical_manifest": {"path": str(R7_MANIFEST), "sha256": sha_file(R7_MANIFEST)},
        "historical_record_dir": str(RECORDS),
        "refit_receipt": pin("ancR9_sweep_complete.json"),
        "stopped": {
            "namespace": NAMESPACE, "request_sha256": snapshot["request_sha256"],
            "snapshot": {**pin(SNAPSHOT), "semantic_sha256": sem(snapshot)},
            "reservation": pin(f"{NAMESPACE}_campaign_reservation.json"),
            "campaign_nonce": snapshot["plan"]["campaign_nonce"],
            "approval": {"section": snapshot["approval"]["section"], "scope": snapshot["approval"]["scope"],
                         "line": snapshot["approval"]["line"]},
            "receipt_absent": f"{NAMESPACE}_test_complete.json",
            "settlement": {"ledger": str(LEDGER), "ledger_sha256": hashlib.sha256(raw).hexdigest(),
                           "ledger_lines": len(lines), "run_id": RUN_ID,
                           "start_line_sha256": hashlib.sha256(lines[mine[0]].encode()).hexdigest(),
                           "stop_line_sha256": hashlib.sha256(lines[mine[1]].encode()).hexdigest(),
                           "final_line_sha256": hashlib.sha256(lines[mine[2]].encode()).hexdigest(),
                           "stop_reason": stop["reason"], "final_status": final["status"],
                           "final_returncode": final["returncode"],
                           "charged_seconds": final["charged_seconds"],
                           "cumulative_charged_seconds": final["cumulative_charged_seconds"]},
        },
        "recovery_cell": recovery,
        "carried": carried,
    }
    path = Path(out)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(lineage, indent=1, sort_keys=True) + "\n")
    print(f"wrote {path} sha256 {sha_file(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
