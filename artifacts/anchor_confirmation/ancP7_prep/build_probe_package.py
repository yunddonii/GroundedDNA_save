#!/usr/bin/env python
"""Probe handoff builder for audit section 729 (metadata only: no deserialisation, no GPU).

From pinned JSON inputs it writes, each once (O_EXCL):
  ancP7_probe_sources.json      -- exactly the twelve stage-D coordinates of the frozen N record:
                                   each dataset's selected stage-S seed-42 entry, copied unchanged
                                   from the audit's ancS7_select_sources.json, and the eight
                                   seed-43/44 cells of the ancD7 receipt;
  request_preview_ancP7_v7.txt  -- the canonical probe request, built by the probe's own
                                   `probe_records` and the reducer's `probe_request`, its SHA256 and
                                   the approval line it needs;
  ancP7_prep/metadata_check.json -- the reducer's JSON/CSV-level admission (`admit_metadata`) of the
                                   twelve records under this generation, and every file it read.
It reads no config.pt, checkpoint, dataset or feature cache, and refuses if the admission consumed
any file that is not JSON or CSV. It is not a manifest member and changes no source.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

WORKTREE = Path("/data/yschoi/gdna_anchor_confirm_v1")
REPO = Path(__file__).resolve().parents[3]
if REPO != WORKTREE:
    raise SystemExit(f"run from the anchor worktree copy, not {REPO}")
sys.path.insert(0, str(REPO))

import scripts.anchor_confirm_code_axis as P  # noqa: E402
import scripts.anchor_confirm_decision as D  # noqa: E402

M = D.M
RECORDS = REPO / "artifacts" / "anchor_confirmation"
MANIFEST = RECORDS / "authority_manifest_v7.json"
MANIFEST_SHA256 = "c061296309fe41203205dff65de3a1afb843c950e93c0614dfd27a8216a90128"
SELECTION = RECORDS / "ancS7_selected_n.json"
SELECTION_SHA256 = "5cda7adb055ed126efb0a8ec198e06d1176ccdc57e7d38fe35ff74b74a4a92ff"
S_SOURCES = RECORDS / "ancS7_select_sources.json"
S_SOURCES_SHA256 = "0cfab95106bfadb03e5e9750d8e5a17d7ff16523ff14b66bba1cab5519067656"
S_RECEIPT = RECORDS / "ancS7_sweep_complete.json"
S_RECEIPT_SHA256 = "5915768767e3fd12b28c3f09e1a071c26fd5cbddbbac15662c495b80c4876f6b"
D_RECEIPT = RECORDS / "ancD7_sweep_complete.json"
D_RECEIPT_SHA256 = "4272eeac4759354f94a58af13d84ab62cf78ca3457fa0d07cd379bd0ad0aa6e8"
OUT_SOURCES = RECORDS / "ancP7_probe_sources.json"
OUT_REQUEST = RECORDS / "request_preview_ancP7_v7.txt"
OUT_CHECK = RECORDS / "ancP7_prep" / "metadata_check.json"
TEXT_SUFFIXES = (".json", ".csv")


def write_once(path: Path, blob: bytes) -> str:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    with os.fdopen(fd, "wb") as handle:
        handle.write(blob)
    return hashlib.sha256(blob).hexdigest()


def main() -> int:
    manifest = M.recheck_generation({"path": str(MANIFEST), "sha256": MANIFEST_SHA256},
                                    "at the probe package build")
    pinned = D.Consumed()
    frozen = pinned.json(SELECTION, SELECTION_SHA256)       # read, not replayed: no config.pt read
    D.need(frozen["generation"]["anchor_manifest_sha256"] == MANIFEST_SHA256,
           "the frozen N record names another generation")
    n_selected = frozen["n_selected"]
    coords = D.expected_coordinates("decide", arms=M.ANCHOR_RUN_ARMS, frozen=n_selected)
    D.need(len(coords) == 12, f"{len(coords)} stage-D coordinates, not 12")

    s_sources = pinned.json(S_SOURCES, S_SOURCES_SHA256)
    pinned.json(S_RECEIPT, S_RECEIPT_SHA256)
    d_receipt = pinned.json(D_RECEIPT, D_RECEIPT_SHA256)
    D.need(d_receipt.get("namespace") == "ancD7" and len(d_receipt.get("cells") or {}) == 8,
           "the D receipt is not the eight-cell ancD7 campaign")
    incumbent = M.anchor_incumbent()

    entries, used_d = [], set()
    for ds, arm, n, seed in coords:
        if seed == M.SEED:
            found = [e for e in s_sources["coordinates"]
                     if (e["dataset"], e["arm"], e["N"], e["seed"]) == (ds, arm, n, seed)]
            D.need(len(found) == 1, f"{ds} N{n} s{seed}: {len(found)} stage-S entries")
            entry = found[0]
            D.need(entry["source"] == "receipt" and entry["receipt"] == str(S_RECEIPT)
                   and entry["receipt_sha256"] == S_RECEIPT_SHA256,
                   f"{ds} N{n} s{seed}: the S entry is not backed by the ancS7 receipt")
        else:
            cell_id = M.campaign_cell_id(ds, n, topp=incumbent[ds]["topp"], joint=incumbent[ds]["joint"],
                                         stage="select", seed=seed, anchor_arm=arm)
            cell = d_receipt["cells"].get(cell_id)
            D.need(isinstance(cell, dict), f"the D receipt has no cell {cell_id}")
            used_d.add(cell_id)
            entry = {"N": n, "arm": arm, "dataset": ds, "receipt": str(D_RECEIPT),
                     "receipt_sha256": D_RECEIPT_SHA256, "record": str(RECORDS / cell["record"]),
                     "record_sha256": cell["record_sha256"], "seed": seed, "source": "receipt"}
        pinned.read(entry["record"], entry["record_sha256"])
        entries.append(entry)
    D.need(used_d == set(d_receipt["cells"]), "the D receipt has cells outside the frozen N record")
    D.need(len({e["record_sha256"] for e in entries}) == 12, "a record digest repeats")

    blob = (json.dumps({"coordinates": entries, "version": M.ANCHOR_CONFIRM_VERSION},
                       indent=2, sort_keys=True) + "\n").encode()
    sources_sha256 = write_once(OUT_SOURCES, blob)

    # the request exactly as the probe's main() forms it from these sources
    records = P.probe_records(str(OUT_SOURCES), sources_sha256, {"n_selected": n_selected})
    request = D.probe_request(MANIFEST_SHA256, SELECTION_SHA256, records)
    request_sha256 = M._json_digest(request)
    line = (f"{M.APPROVAL_TAG} version={M.ANCHOR_CONFIRM_VERSION} scope=probe "
            f"manifest={MANIFEST_SHA256} selection={SELECTION_SHA256} request={request_sha256}")

    # the JSON/CSV-level admission each probe repeats before its first load
    consumed = D.Consumed()
    keyed = D.load_sources(consumed, OUT_SOURCES, sources_sha256, coords, with_probe=False)
    admitted = {c: D.admit_metadata(consumed, c, keyed[c], incumbent=incumbent,
                                    manifest_sha256=MANIFEST_SHA256, selection_sha256=SELECTION_SHA256)
                for c in coords}
    D.one_generation(admitted, MANIFEST_SHA256)
    read = sorted(consumed.digests) + sorted(pinned.digests)
    binary = [name for name in read if not name.endswith(TEXT_SUFFIXES)]
    D.need(not binary, f"the admission read non-JSON/CSV files: {binary[:4]}")
    per_dataset = {}
    for c in coords:
        per_dataset.setdefault(c[0], set()).add(admitted[c]["record"]["input_authority"]["split_identity_sha256"])
    D.need(all(len(v) == 1 for v in per_dataset.values()), "a dataset's records name two split identities")
    changed = consumed.reverify() + pinned.reverify()
    D.need(not changed, f"inputs changed during the build: {changed[:4]}")
    M.recheck_generation(manifest, "before the probe package is written")

    check = {
        "artifact_kind": "anchor_confirmation_probe_package_check",
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "generation": {"manifest_sha256": MANIFEST_SHA256, "commit": manifest.get("commit")},
        "selection_sha256": SELECTION_SHA256, "n_selected": n_selected,
        "receipts": {"ancS7": S_RECEIPT_SHA256, "ancD7": D_RECEIPT_SHA256},
        "sources_sha256": sources_sha256, "request_sha256": request_sha256,
        "population": dict(D.PROBE_POPULATION),
        "coordinates": [{
            "coordinate": list(c),
            "record": keyed[c]["record"], "record_sha256": keyed[c]["record_sha256"],
            "receipt_sha256": keyed[c]["receipt_sha256"],
            "cell_id": admitted[c]["authority"]["cell_id"],
            "campaign_approval": {k: admitted[c]["authority"]["approval"][k]
                                  for k in ("section", "scope", "request_sha256")},
            "run_dir": str(admitted[c]["run_dir"]),
            "config_pt_sha256_pin": admitted[c]["config_pt_sha256"],
            "final_checkpoint": admitted[c]["record"]["completion"]["final_checkpoint"],
            "final_checkpoint_sha256_pin": admitted[c]["record"]["completion"]["final_checkpoint_sha256"],
            "input_seal_sha256": admitted[c]["record"]["input_authority"]["seal_file_sha256"],
            "split_identity_sha256": admitted[c]["record"]["input_authority"]["split_identity_sha256"],
            "terminal_validation_mAP_at_R": admitted[c]["score"]} for c in coords],
        "files_read_sha256": {**consumed.digests, **pinned.digests},
        "non_json_csv_files_read": binary,
        "deserialised": [],
        "audit_ledger_sha256_at_check": hashlib.sha256(M.AUDIT_LEDGER.read_bytes()).hexdigest(),
    }
    check_sha256 = write_once(OUT_CHECK, (json.dumps(check, indent=1, sort_keys=True) + "\n").encode())

    preview = (f"anchor-confirm/2: probe, 12 coordinates, population "
               f"{D.PROBE_POPULATION['n_images']} rows x {D.PROBE_POPULATION['local_slots']} slots  "
               f"[NON-EXECUTABLE until audit approval]\n"
               + json.dumps(request, indent=1, sort_keys=True) + "\n"
               + f"request sha256 {request_sha256}\n"
               + f"needs: {line}\n")
    preview_sha256 = write_once(OUT_REQUEST, preview.encode())
    print(f"sources {OUT_SOURCES.name} {sources_sha256}")
    print(f"request {request_sha256} (preview file {preview_sha256})")
    print(f"check   {OUT_CHECK.name} {check_sha256}")
    print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
