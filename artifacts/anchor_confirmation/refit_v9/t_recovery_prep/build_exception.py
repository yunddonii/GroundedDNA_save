"""Build the pinned exception artifact for the failed r8 recovery (audits 839, 841), metadata only.

Reads the failed recovery's JSON records, its supervisor ledger lines and the consumed claim; opens no
checkpoint, config, array or cache. The ledger is append-only and grows, so the failed run's rows are
pinned line by line (each line's SHA256) and by the digest of their byte prefix, never by the whole file.
Absence of the failed namespace's entry, record and receipt is NOT pinned here: it is re-observed by lstat
at every boundary (audit 841.3). Writes one JSON file once. The r9 stage source pins this file's digest.
Usage: python build_exception.py <out.json>
"""
import hashlib
import json
import os
import sys
from pathlib import Path

R8_RECORDS = Path("/data/yschoi/gdna_anchor_refit_v9r8/artifacts/anchor_confirmation")
LEDGER = Path("/home/yschoi/gdna_anchorRTrec_ops/device_budget_ledger.jsonl")
CLAIM_ROOT = Path("/home/yschoi/gdna_anchorRT_recovery_claims")
RUN_ID = "20261007T061117Z-12e14ceb"
NAMESPACE = "ancT9r"
TAG = "ancR9_nuswide_A_v4_refit_N4_s44_AXanchors_P0408_JD005"
CELL_ID = "nuswide|N=4|P=0.4,0.8|JD=0.05|stage=refit|seed=44|axis_center=anchors"
REQUEST_SHA256 = "d34f505b90db1ef02dce74bd0326350053a6708315f505c0c881fe89a91ad666"
CLAIM_KEY = "dc4a8bc061c5d286d74e21b158091710c165737403e1c85e957f810740f92c6d"
SNAPSHOT = "ancT9r_snapshot_7205adea6c6238e4.json"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sem(payload) -> str:
    return sha(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())


def main(out: str) -> int:
    raw = LEDGER.read_bytes()
    lines = raw.split(b"\n")
    assert lines[-1] == b"", "the ledger does not end with a newline"
    rows = [json.loads(line) for line in lines[:-1]]
    mine = [r for r in rows if r.get("run_id") == RUN_ID]
    assert [r["event"] for r in rows[:2]] == ["start", "final"] and mine == rows[:2], \
        "the ledger does not begin with exactly the failed run's start and final"
    start, final = rows[0], rows[1]
    assert final["status"] == "exited" and final["returncode"] == 1 and final["attempts"] == []
    prefix = lines[0] + b"\n" + lines[1] + b"\n"
    attempt_name = f"{NAMESPACE}_attempt_{TAG}.json"
    attempt = json.loads((R8_RECORDS / attempt_name).read_bytes())
    claim_path = CLAIM_ROOT / f"{CLAIM_KEY}.json"
    claim = json.loads(claim_path.read_bytes())
    assert attempt["request_sha256"] == REQUEST_SHA256 and sem(attempt["request"]) == REQUEST_SHA256
    assert claim["request_sha256"] == REQUEST_SHA256 and claim["attempt"] == attempt_name
    assert claim["claim_key"] == CLAIM_KEY and claim["approval"] == {k: attempt["approval"][k]
                                                                     for k in ("section", "scope", "line")}
    snapshot = json.loads((R8_RECORDS / SNAPSHOT).read_bytes())
    reservation = json.loads((R8_RECORDS / f"{NAMESPACE}_campaign_reservation.json").read_bytes())
    assert reservation["plan_digest"] == sem(snapshot) and reservation["namespace"] == NAMESPACE
    assert reservation["campaign_nonce"] == attempt["campaign_nonce"] == snapshot["plan"]["campaign_nonce"]

    def pin(path: Path) -> dict:
        return {"path": str(path), "sha256": sha(path.read_bytes())}
    artifact = {
        "schema": "anchor-t-recovery-exception/1",
        "note": ("the failed r8 recovery of the stopped campaign's interrupted cell (audits 838-839): "
                 "settled exited rc 1 before its T entry; it hashed the eleven carried cells' outputs and the "
                 "target checkpoint/config before the dispatch guard refused; no T-entry producer, outputs, "
                 "record or receipt. Absences are re-observed by lstat, never pinned."),
        "cell_id": CELL_ID, "namespace": NAMESPACE, "record_dir": str(R8_RECORDS),
        "request_sha256": REQUEST_SHA256,
        "approval": {k: attempt["approval"][k] for k in ("section", "scope", "line")},
        "window_section": 838,
        "run_id": RUN_ID,
        "ledger": {"path": str(LEDGER), "rows": 2, "prefix_sha256": sha(prefix),
                   "line_sha256": [sha(lines[0]), sha(lines[1])],
                   "charged_seconds": final["charged_seconds"],
                   "cumulative_including_parent_seconds": final["cumulative_including_parent_seconds"],
                   "parent_sha256": start["parent"]["sha256"]},
        "claim": {"key": CLAIM_KEY, **pin(claim_path)},
        "attempt": pin(R8_RECORDS / attempt_name),
        "reservation": pin(R8_RECORDS / f"{NAMESPACE}_campaign_reservation.json"),
        "snapshot": {**pin(R8_RECORDS / SNAPSHOT), "semantic_sha256": sem(snapshot)},
        "campaign_nonce": attempt["campaign_nonce"],
        "absent": [f"{NAMESPACE}_entry_{TAG}.json", f"{NAMESPACE}_{TAG}.json", f"{NAMESPACE}_recovery_complete.json"],
    }
    data = (json.dumps(artifact, indent=1, sort_keys=True) + "\n").encode()
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
    print(f"{out} {sha(data)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
