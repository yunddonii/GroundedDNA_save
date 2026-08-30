"""A campaign is 24 cells or it is not a campaign (F09).

The chain launched cells one at a time and collected exit codes at the end. That
is not a transaction, and four things could happen that nothing would notice:

  * a second chain, started while the first was running, planned the same 24
    cells and both wrote into the same result directories;
  * a cell that never started left no trace, so "23 of them succeeded" and "24
    of them succeeded" looked the same to every downstream reader;
  * the plan could be rebuilt between cell 1 and cell 24 -- a different N, a
    different cache -- and the campaign would be a mixture with one name;
  * `docs/newmodel_analysis/` was written from whatever cells happened to exist,
    so a partial campaign became a paper table.

The ledger is the missing transaction. `open_campaign` reserves the whole set at
once with `O_EXCL` -- so a second chain fails at reservation rather than at the
result directory -- and freezes the plan digest into the reservation. Each cell
records its own outcome against that digest. `seal_campaign` writes the
completion receipt only when all 24 are present and successful, and readers are
expected to require the receipt rather than to count directories.

Usage:
    python scripts/campaign_ledger.py open  --plan P --ledger L
    python scripts/campaign_ledger.py record --ledger L --tag T --status ok
    python scripts/campaign_ledger.py seal  --ledger L
    python scripts/campaign_ledger.py verify --ledger L --plan P
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import socket
from pathlib import Path
import sys

SCHEMA_VERSION = 1
RECEIPT_NAME = "campaign_complete.json"
STATUSES = ("ok", "failed")


class CampaignRefused(RuntimeError):
    """The campaign cannot be opened, recorded against, or sealed."""


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _atomic_write(path: Path, payload: dict) -> None:
    """Same filesystem, fsync, rename -- a torn ledger is worse than none."""
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    blob = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(blob)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _boot_id() -> str:
    try:
        with open("/proc/sys/kernel/random/boot_id", encoding="utf-8") as h:
            return h.read().strip()
    except OSError:
        return ""


def open_campaign(ledger_dir: Path, plan_path: Path) -> dict:
    """Reserve every cell at once, or refuse.

    `O_EXCL` on the reservation file is what makes a second chain fail here
    rather than 150 seconds later inside a result directory that already exists.
    """
    raw = plan_path.read_bytes()
    plan = json.loads(raw.decode("utf-8"))
    cells = plan.get("cells") or []
    expected = plan.get("expected_cells")
    if not isinstance(expected, int) or isinstance(expected, bool) \
            or len(cells) != expected:
        raise CampaignRefused(
            f"{plan_path}: {len(cells)} cells against expected_cells "
            f"{expected!r}; a campaign that cannot count itself cannot be "
            f"reserved")
    tags = [c["tag"] for c in cells]
    if len(set(tags)) != len(tags):
        raise CampaignRefused(f"{plan_path}: duplicate tags {sorted(tags)}")

    ledger_dir.mkdir(parents=True, exist_ok=True)
    reservation = ledger_dir / "campaign_reservation.json"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "plan_path": str(plan_path),
        "plan_file_sha256": _sha_bytes(raw),
        "plan_digest": plan.get("plan_digest"),
        "expected_cells": expected,
        "tags": sorted(tags),
        "selected_n": plan.get("selected_n"),
        "selection_sha256": plan.get("selection_sha256"),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "boot_id": _boot_id(),
    }
    blob = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        fd = os.open(str(reservation), os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                     0o644)
    except OSError as error:
        if error.errno != errno.EEXIST:
            raise
        held = json.loads(reservation.read_text(encoding="utf-8"))
        raise CampaignRefused(
            f"{reservation} is already held by pid {held.get('pid')} on "
            f"{held.get('host')} for plan {str(held.get('plan_digest'))[:12]}. "
            f"Two campaigns writing one result set is how a table becomes a "
            f"mixture; remove it only if that run is gone.") from None
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(blob)
        handle.flush()
        os.fsync(handle.fileno())
    return payload


def _reservation(ledger_dir: Path) -> dict:
    path = ledger_dir / "campaign_reservation.json"
    if not path.is_file():
        raise CampaignRefused(
            f"{path} does not exist; a cell cannot be recorded against a "
            f"campaign that was never opened")
    return json.loads(path.read_text(encoding="utf-8"))


def record_cell(ledger_dir: Path, tag: str, status: str, *,
                run_dir: str | None = None, detail: str | None = None) -> dict:
    if status not in STATUSES:
        raise CampaignRefused(f"status {status!r} is not one of {STATUSES}")
    # A success has to name what it produced. The first version accepted
    # `status=ok` with no `--run-dir`, and the launcher never passed one, so a
    # real three-cell run sealed a receipt whose every output was null -- the
    # receipt said "complete" and pointed at nothing.
    if status == "ok":
        if not run_dir:
            raise CampaignRefused(
                f"{tag}: a successful cell must name its result directory; "
                f"'ok' with no run_dir is a receipt that points at nothing")
        if not Path(run_dir).is_dir():
            raise CampaignRefused(
                f"{tag}: run_dir {run_dir!r} is not a directory that exists")
    held = _reservation(ledger_dir)
    if tag not in held["tags"]:
        raise CampaignRefused(
            f"{tag!r} is not one of the {len(held['tags'])} reserved cells; a "
            f"campaign cannot grow a cell after it was opened")
    entry = {
        "schema_version": SCHEMA_VERSION,
        "tag": tag, "status": status, "run_dir": run_dir, "detail": detail,
        "plan_digest": held["plan_digest"],
        "plan_file_sha256": held["plan_file_sha256"],
    }
    _atomic_write(ledger_dir / f"cell_{tag}.json", entry)
    return entry


def seal_campaign(ledger_dir: Path) -> dict:
    """The receipt exists only for a campaign that is entirely present."""
    held = _reservation(ledger_dir)
    entries, missing, failed, foreign = {}, [], [], []
    for tag in held["tags"]:
        path = ledger_dir / f"cell_{tag}.json"
        if not path.is_file():
            missing.append(tag)
            continue
        entry = json.loads(path.read_text(encoding="utf-8"))
        # A cell recorded against a DIFFERENT plan is not this campaign's cell,
        # even though its tag matches -- this is the rebuild-mid-run case.
        if entry.get("plan_file_sha256") != held["plan_file_sha256"]:
            foreign.append(tag)
            continue
        if entry.get("status") != "ok":
            failed.append(tag)
            continue
        if not entry.get("run_dir"):
            failed.append(tag)
            continue
        entries[tag] = entry

    if missing or failed or foreign:
        raise CampaignRefused(
            f"cannot seal: {len(entries)} of {len(held['tags'])} cells are "
            f"complete. missing={missing} failed={failed} "
            f"recorded_against_another_plan={foreign}")

    receipt = {
        "schema_version": SCHEMA_VERSION,
        "plan_digest": held["plan_digest"],
        "plan_file_sha256": held["plan_file_sha256"],
        "selected_n": held.get("selected_n"),
        "selection_sha256": held.get("selection_sha256"),
        "cells": {tag: entries[tag]["run_dir"] for tag in sorted(entries)},
        "cell_count": len(entries),
    }
    _atomic_write(ledger_dir / RECEIPT_NAME, receipt)
    return receipt


def require_sealed(ledger_dir: Path, *, expected_cells: int,
                   plan_file_sha256: str | None = None) -> dict:
    """What a downstream reader must call before tabulating anything."""
    path = ledger_dir / RECEIPT_NAME
    if not path.is_file():
        raise CampaignRefused(
            f"{path} does not exist; this campaign never completed, and a "
            f"partial one must not be read as a table")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("schema_version") != SCHEMA_VERSION:
        raise CampaignRefused(f"{path}: unknown schema_version")
    if receipt.get("cell_count") != expected_cells \
            or len(receipt.get("cells") or {}) != expected_cells:
        raise CampaignRefused(
            f"{path}: {receipt.get('cell_count')} cells, expected "
            f"{expected_cells}")
    empty = sorted(t for t, d in (receipt.get("cells") or {}).items() if not d)
    if empty:
        raise CampaignRefused(
            f"{path}: {len(empty)} cells name no result directory {empty[:3]}; "
            f"this receipt does not point at any output")
    if plan_file_sha256 is not None \
            and receipt.get("plan_file_sha256") != plan_file_sha256:
        raise CampaignRefused(
            f"{path} seals a different plan than the one being read")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("open", "record", "seal", "verify"):
        p = sub.add_parser(name)
        p.add_argument("--ledger", required=True)
        if name in ("open", "verify"):
            p.add_argument("--plan", required=True)
        if name == "record":
            p.add_argument("--tag", required=True)
            p.add_argument("--status", required=True, choices=STATUSES)
            p.add_argument("--run-dir", default=None)
            p.add_argument("--detail", default=None)
    args = parser.parse_args()

    ledger = Path(args.ledger)
    try:
        if args.action == "open":
            held = open_campaign(ledger, Path(args.plan))
            print(f"[ledger] reserved {len(held['tags'])} cells for plan "
                  f"{str(held['plan_digest'])[:12]}")
        elif args.action == "record":
            entry = record_cell(ledger, args.tag, args.status,
                                run_dir=args.run_dir, detail=args.detail)
            print(f"[ledger] {entry['tag']} {entry['status']}")
        elif args.action == "seal":
            receipt = seal_campaign(ledger)
            print(f"[ledger] sealed {receipt['cell_count']} cells")
        else:
            plan = json.loads(Path(args.plan).read_bytes().decode("utf-8"))
            receipt = require_sealed(
                ledger, expected_cells=plan["expected_cells"],
                plan_file_sha256=_sha_bytes(Path(args.plan).read_bytes()))
            print(f"[ledger] complete: {receipt['cell_count']} cells")
    except CampaignRefused as error:
        print(f"[ledger] REFUSED: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
