"""Independent check of the ancP7 probe outputs (audit 730.3), plain JSON only, no project imports."""
import hashlib, json, sys
from pathlib import Path

REC = Path("/data/yschoi/gdna_anchor_confirm_v1/artifacts/anchor_confirmation")
OUT = REC / "ancP7_probes"
LEDGER = Path("/home/yschoi/gdna_anchor4_ops/device_budget_ledger.jsonl")
AUDIT = Path("/home/yschoi/GroundedDNA/docs/PHASE1_PHASE2_REAUDIT_2026-08-14.md")
PRIOR = 19033.569830481894
REQUEST = "748851142a8734703c076bbccd52af8c383dba14ee105d29235c55c92c476c44"
MANIFEST = "c061296309fe41203205dff65de3a1afb843c950e93c0614dfd27a8216a90128"
LINE = ("ANCHOR-CONFIRM-APPROVAL version=anchor-confirm/2 scope=probe manifest=" + MANIFEST
        + " selection=5cda7adb055ed126efb0a8ec198e06d1176ccdc57e7d38fe35ff74b74a4a92ff request=" + REQUEST)
PRODUCER = "31537dde1cc8dd338221bb7fd49241cc0c618c0de444a2adf3500e86f708b19e"
POP = {"rows": "first 500 of the train-only validation split, ascending dataset index",
       "n_images": 500, "local_slots": 4, "decisions": 2000}
H = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

sources = json.loads((REC / "ancP7_probe_sources.json").read_text())["coordinates"]
check = {tuple(c["coordinate"]): c for c in json.loads((REC / "ancP7_prep/metadata_check.json").read_text())["coordinates"]}
order = [(d, "anchors", n, s) for d, n in (("flickr25k", 4), ("cifar10", 4), ("nuswide", 4), ("mscoco", 39))
         for s in (42, 43, 44)]
problems, rows = [], []
files = sorted(OUT.glob("*.json")) if OUT.is_dir() else []
print(f"outputs present: {len(files)}")
by_ds = {}
for coord in order:
    d, a, n, s = coord
    f = OUT / f"{d}_{a}_N{n}_s{s}.json"
    if not f.is_file():
        rows.append((coord, None)); continue
    e = json.loads(f.read_text())
    src = next(x for x in sources if (x["dataset"], x["arm"], x["N"], x["seed"]) == coord)
    mc = check[coord]
    want = {
        "kind": e.get("artifact_kind") == "anchor_confirmation_code_axis" and e.get("schema") == "anchor-confirm-code-axis/3",
        "coordinate": e.get("coordinate") == list(coord),
        "record": e.get("record_sha256") == src["record_sha256"] == H(src["record"]),
        "checkpoint": e.get("checkpoint_sha256") == mc["final_checkpoint_sha256_pin"],
        "config": e.get("config_pt_sha256") == mc["config_pt_sha256_pin"],
        "manifest": e.get("manifest_sha256") == MANIFEST,
        "request": e.get("request_sha256") == REQUEST,
        "approval": e.get("approval") == {"section": 730, "scope": "probe", "line": LINE},
        "producer": e.get("producer_sha256") == PRODUCER == H("/data/yschoi/gdna_anchor_confirm_v1/scripts/anchor_confirm_code_axis.py"),
        "population": e.get("population") == POP and e.get("n_images") == 500,
        "split": e.get("split") == {"val_split_ratio": 0.1, "val_split_seed": 42}
                 and e.get("split_identity_sha256") == mc["split_identity_sha256"],
        "routing": e.get("routing") == "deployment_no_text",
        "caption_seal": (e.get("caption_target") or {}).get("input_seal_sha256") == mc["input_seal_sha256"],
        "counts": type(e.get("hits")) is int and e.get("total") == 2000 and 0 <= e["hits"] <= 2000
                  and type(e.get("ties_counted_as_misses")) is int
                  and 0 <= e["ties_counted_as_misses"] <= 2000 - e["hits"],
        "ratio": type(e.get("code_picks_own_axis")) is float and e["code_picks_own_axis"] == e["hits"] / e["total"],
        "authority": (e.get("authority") or {}).get("cell_id") == mc["cell_id"],
    }
    bad = [k for k, v in want.items() if not v]
    if bad:
        problems.append(f"{coord}: {bad}")
    by_ds.setdefault(d, set()).add((e.get("row_ids_sha256"), e.get("split_identity_sha256")))
    rows.append((coord, {"file_sha256": H(f), "hits": e.get("hits"), "ties": e.get("ties_counted_as_misses"),
                         "ratio": e.get("code_picks_own_axis"), "row": e.get("row_ids_sha256")}))
for d, ids in by_ds.items():
    if len(ids) != 1:
        problems.append(f"{d}: {len(ids)} distinct (row digest, split identity) pairs across seeds")
for coord, r in rows:
    print(coord, "MISSING" if r is None else f"{r['hits']}/2000 ties {r['ties']} ratio {r['ratio']:.6f} "
          f"row {str(r['row'])[:12]} file {r['file_sha256']}")

# operations ledger: the twelve probe runs after the D settlement
recs = [json.loads(l) for l in LEDGER.read_text().splitlines()]
starts = [r for r in recs if r.get("event") == "start" and r.get("stage") == "probe"]
finals = {r["run_id"]: r for r in recs if r.get("event") == "final"}
total = 0.0
print(f"ledger: {len(starts)} probe starts; ledger sha {H(LEDGER)}")
for s in starts:
    f = finals.get(s["run_id"])
    total += float(f["charged_seconds"]) if f else 0.0
    print(f"  {s['run_id']} {s['label']} prior {s['prior_charged_seconds']:.6f} -> "
          + (f"{f['status']} rc={f.get('returncode')} charged {f['charged_seconds']:.3f} device {f['device_seconds']:.3f} "
             f"allowance {f['unobserved_allowance_seconds']} cumulative {f['cumulative_charged_seconds']:.6f} "
             f"leases {f['leases_held_after_exit']} orphans {f['orphaned_live_attempts']} continuity {f['continuity_lost']}"
             if f else "NO FINAL RECORD"))
    if not f or f["status"] != "exited" or f.get("returncode") != 0:
        problems.append(f"{s['run_id']}: not a clean rc-0 exit")
last = [r for r in recs if r.get("event") == "final"][-1]
print(f"sum of probe charges {total:.6f}; prior + sum {PRIOR + total:.12f}; last cumulative {last['cumulative_charged_seconds']!r}")
if abs(PRIOR + total - last["cumulative_charged_seconds"]) > 1e-6:
    problems.append("cumulative charge does not reconcile")
print(f"audit ledger sha {H(AUDIT)}")
print("PROBLEMS:" if problems else "no problems found", *problems, sep="\n  ")
sys.exit(1 if problems else 0)
