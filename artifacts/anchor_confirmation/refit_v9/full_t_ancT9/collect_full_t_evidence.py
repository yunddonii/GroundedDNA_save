#!/usr/bin/env python3
"""Read-only evidence collector for the anchor full T (audit §788.3).

For every cell in the T snapshot's approved request it reports:
- the T record, attempt and entry files and their bindings;
- the twelve required outputs (sha256, bytes) and the absence of evaluation_siglip2_bit2.json;
- the R pins now (checkpoint, runtime json, config.pt) against the request and the full-R summary;
- the metrics read from the evaluation / NMI / cell_result files themselves, cross-checked with the T record.

Then it gives per-dataset mean/SD over the complete cells: descriptive only, no test.

It writes nothing except the --out JSON and --md table. It never imports the launcher: the output
contract is read from the launcher source with ast.
"""
import argparse, ast, hashlib, json, os, statistics, sys
from pathlib import Path

TREE = Path("/data/yschoi/gdna_anchor_refit_v9r6")
RECS = TREE / "artifacts/anchor_confirmation"
REQUEST = "a74b68e1db10c0d71d911ce491a25a2fa97dda6258c88d9f2c063223e1b40612"
BIT2 = "evaluation_siglip2_bit2.json"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def contract():
    tree = ast.parse((TREE / "scripts/phase3_selection_matrix.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "OFFICIAL_TEST_OUTPUTS" for t in node.targets):
            return tuple(ast.literal_eval(node.value))
    raise SystemExit("OFFICIAL_TEST_OUTPUTS not found")


def cell_evidence(c, r_summary, outputs, nonce, hash_pins):
    tag = c.get("tag") or Path(c["record"]).stem
    run = Path(c["run_dir"])
    ev = {"cell_id": c["cell_id"], "dataset": c["dataset"], "seed": c["seed"], "N": c["N"], "tag": tag,
          "run_dir": str(run), "problems": []}
    bad = ev["problems"].append
    files = {k: RECS / f"ancT9_{k + '_' if k != 'record' else ''}{tag}.json" for k in ("record", "attempt", "entry")}
    for k, p in files.items():
        ev[k] = {"file": p.name, "sha256": sha(p) if p.exists() else None}
    req = [n for n in outputs if n != BIT2]
    present = {n: {"sha256": sha(run / n), "bytes": (run / n).stat().st_size} for n in req if (run / n).exists()}
    ev["outputs"] = present
    ev["outputs_missing"] = [n for n in req if n not in present]
    ev["bit2_absent"] = not (run / BIT2).exists()
    if not ev["bit2_absent"]:
        bad("evaluation_siglip2_bit2.json exists")
    ev["complete"] = ev["record"]["sha256"] is not None and not ev["outputs_missing"]
    if ev["record"]["sha256"] is not None and ev["outputs_missing"]:
        bad(f"T record exists but outputs missing: {ev['outputs_missing']}")
    # R pins now, against the approved request and the full-R summary
    rs = r_summary.get(c["cell_id"], {})
    pins = {"model_state_dict.pth": ("final_checkpoint_sha256", "checkpoint_sha256"),
            "model_state_dict.pth.runtime.json": ("checkpoint_runtime_sha256", "runtime_sha256"),
            "config.pt": ("config_pt_sha256", "config_sha256")}
    ev["r_pins"] = {}
    for f, (rk, sk) in pins.items():
        now = sha(run / f) if hash_pins else None
        ev["r_pins"][f] = {"now": now, "request": c[rk], "full_r_summary": rs.get(sk)}
        if c[rk] != rs.get(sk):
            bad(f"{f}: request pin != full-R summary")
        if hash_pins and now != c[rk]:
            bad(f"{f}: changed since R")
    if ev["attempt"]["sha256"]:
        a = json.loads(files["attempt"].read_text())
        if a.get("request_sha256") != REQUEST or a.get("campaign_nonce") != nonce:
            bad("attempt request/nonce mismatch")
    if ev["entry"]["sha256"]:
        e = json.loads(files["entry"].read_text())
        if e.get("attempt_sha256") != ev["attempt"]["sha256"]:
            bad("entry does not bind the attempt bytes")
        if e.get("final_checkpoint_sha256") != c["final_checkpoint_sha256"] or e.get("config_pt_sha256") != c["config_pt_sha256"]:
            bad("entry pins differ from request")
    if not ev["complete"]:
        return ev
    t = json.loads(files["record"].read_text())
    comp = t["completion"]
    if t.get("request_sha256") != REQUEST or t.get("campaign_nonce") != nonce or t.get("namespace") != "ancT9":
        bad("T record request/nonce/namespace mismatch")
    if t["attempt"]["sha256"] != ev["attempt"]["sha256"]:
        bad("T record does not bind the attempt bytes")
    if t["cell"] != c:
        bad("T record cell differs from the approved request cell")
    bind = {"evaluation_sha256": "evaluation_siglip2_base.json",
            "bio_evaluation_sha256": "evaluation_siglip2_base_bioproj.json",
            "cell_result_sha256": "cell_result.json", "analysis_complete_sha256": "analysis_complete.json",
            "extraction_complete_sha256": "extraction_complete.json", "pairwise_nmi_sha256": "pairwise_nmi.json"}
    for k, n in bind.items():
        if comp.get(k) != present[n]["sha256"]:
            bad(f"T record {k} != {n} bytes")
    for s in ("query", "db", "train"):
        if comp["extraction_manifest_sha256"][s] != present[f"extraction_manifest_{s}.json"]["sha256"]:
            bad(f"T record manifest {s} mismatch")
        if comp["npz_sha256"][s] != present[f"extract_{s}.npz"]["sha256"]:
            bad(f"T record npz {s} mismatch")
    ld = lambda n: json.loads((run / n).read_text())
    base, bio, nmi, cr = ld("evaluation_siglip2_base.json"), ld("evaluation_siglip2_base_bioproj.json"), \
        ld("pairwise_nmi.json"), ld("cell_result.json")
    man = {s: ld(f"extraction_manifest_{s}.json") for s in ("query", "db", "train")}
    for s, m in man.items():
        if m["npz_sha256"] != present[f"extract_{s}.npz"]["sha256"]:
            bad(f"manifest {s} does not bind its npz")
        if m["checkpoint_sha256"] != c["final_checkpoint_sha256"] or m["config_sha256"] != c["config_pt_sha256"]:
            bad(f"manifest {s} pins differ from request")
        if m.get("inference_epoch") != c["terminal_epoch"]:
            bad(f"manifest {s} inference_epoch != terminal epoch")
    ev["splits"] = {"n_query": base["n_query"], "n_db": base["n_db"], "n_train": man["train"]["n_rows"],
                    "manifest_rows": {s: man[s]["n_rows"] for s in man}}
    ev["metrics"] = {
        "raw_base": {"mAP_at_R": base["mAP_at_R"], "R": base["mAP_R_cutoff"], "mAP": base["mAP"],
                     "P@1": base["precision_at_k"]["1"], "P@10": base["precision_at_k"]["10"],
                     "unique_code_ratio": base["unique_code_ratio"], "bio_project": base["bio_project"]},
        "base_bio": {"mAP_at_R": bio["mAP_at_R"], "R": bio["mAP_R_cutoff"], "mAP": bio["mAP"],
                     "P@1": bio["precision_at_k"]["1"], "P@10": bio["precision_at_k"]["10"],
                     "unique_code_ratio": bio["unique_code_ratio"], "bio_project": bio["bio_project"],
                     "db_num_proj_failed": bio["bio_stats"].get("db_num_proj_failed")},
        "nmi": {"mean_off_diag": nmi["mean_off_diag_nmi"], "min": nmi["min_off_diag_nmi"],
                "max": nmi["max_off_diag_nmi"], "N": nmi["N"]},
        "dna_unique_db": cr["DNA_unique_DB"],
    }
    if base["bio_project"] is not False or bio["bio_project"] is not True:
        bad("bio_project flags wrong")
    checks = [("map_at_R", base["mAP_at_R"]), ("map", base["mAP"]), ("bio_map_at_R", bio["mAP_at_R"]),
              ("bio_map", bio["mAP"]), ("mean_off_diag_nmi", nmi["mean_off_diag_nmi"])]
    for k, v in checks:
        if comp.get(k) != v:
            bad(f"T record {k} != file value")
    am = comp["analysis_metrics"]
    if am["full_map_bioproj"] != bio["mAP"] or am["map_at_R_bioproj"] != bio["mAP_at_R"] or \
            am["dna_unique_db"] != cr["DNA_unique_DB"] or am["mean_off_diag_nmi"] != nmi["mean_off_diag_nmi"]:
        bad("analysis metrics differ from the evaluation files")
    ev["wall_seconds"] = t.get("wall_seconds")
    return ev


def summarise(cells):
    out = {}
    keys = [("raw_base", "mAP_at_R"), ("base_bio", "mAP_at_R"), ("raw_base", "P@1"), ("base_bio", "P@1"),
            ("nmi", "mean_off_diag"), ("base_bio", "unique_code_ratio")]
    for ds in sorted({c["dataset"] for c in cells}):
        done = [c for c in cells if c["dataset"] == ds and c.get("metrics")]
        row = {"complete_seeds": [c["seed"] for c in done]}
        for g, k in keys:
            vals = [c["metrics"][g][k] for c in done]
            row[f"{g}.{k}"] = {"values": vals, "mean": statistics.mean(vals) if vals else None,
                               "sd": statistics.stdev(vals) if len(vals) > 1 else None}
        out[ds] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=str(RECS / "ancT9_snapshot_da40a183d4ca9c85.json"))
    ap.add_argument("--r-summary", default=str(TREE / "artifacts/anchor_confirmation/refit_v9/full_r_ancR9/cells_summary.json"))
    ap.add_argument("--only", default=None, help="restrict to one cell_id (testing)")
    ap.add_argument("--no-hash-pins", action="store_true", help="skip hashing the 0.65 GB checkpoints")
    ap.add_argument("--out", required=True)
    ap.add_argument("--md", default=None)
    a = ap.parse_args()
    snap = json.loads(Path(a.snapshot).read_text())
    if snap["request_sha256"] != REQUEST:
        raise SystemExit("snapshot is not the approved request")
    nonce = snap["plan"]["campaign_nonce"]
    outputs = contract()
    r_summary = json.loads(Path(a.r_summary).read_text())["cells"]
    cells = [c for c in snap["request"]["cells"] if a.only in (None, c["cell_id"])]
    ev = []
    for c in cells:
        try:
            ev.append(cell_evidence(c, r_summary, outputs, nonce, not a.no_hash_pins))
        except Exception as exc:  # an unreadable or malformed artifact is a finding, not a crash
            ev.append({"cell_id": c["cell_id"], "dataset": c["dataset"], "seed": c["seed"], "complete": False,
                       "problems": [f"unreadable: {type(exc).__name__}: {exc}"]})
    receipt = RECS / "ancT9_test_complete.json"
    res = {"request_sha256": REQUEST, "snapshot": Path(a.snapshot).name, "snapshot_sha256": sha(a.snapshot),
           "reservation_sha256": sha(RECS / "ancT9_campaign_reservation.json"),
           "receipt": {"file": receipt.name, "sha256": sha(receipt) if receipt.exists() else None},
           "contract": list(outputs), "cells_planned": len(snap["request"]["cells"]), "cells_examined": len(ev),
           "cells_complete": sum(c["complete"] for c in ev),
           "cells_with_problems": [c["cell_id"] for c in ev if c["problems"]],
           "cells": ev, "per_dataset_descriptive": summarise(ev),
           "note": "descriptive mean/SD over complete seeds; no statistical test was run"}
    Path(a.out).write_text(json.dumps(res, indent=1))
    if a.md:
        L = ["| Dataset | seeds done | raw mAP@R | BIO mAP@R | raw P@1 | BIO P@1 | NMI | BIO unique |", "|---|---|---|---|---|---|---|---|"]
        for ds, r in res["per_dataset_descriptive"].items():
            f = lambda k: "—" if r[k]["mean"] is None else (f"{r[k]['mean']:.4f}" + ("" if r[k]["sd"] is None else f" ± {r[k]['sd']:.4f}"))
            L.append(f"| {ds} | {r['complete_seeds']} | {f('raw_base.mAP_at_R')} | {f('base_bio.mAP_at_R')} | {f('raw_base.P@1')} | "
                     f"{f('base_bio.P@1')} | {f('nmi.mean_off_diag')} | {f('base_bio.unique_code_ratio')} |")
        Path(a.md).write_text("\n".join(L) + "\n")
    print(f"complete {res['cells_complete']}/{res['cells_planned']}, problems {len(res['cells_with_problems'])}, receipt {'present' if res['receipt']['sha256'] else 'absent'}")
    return 1 if res["cells_with_problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
