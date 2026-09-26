#!/usr/bin/env python
"""Anchor confirmation v1: the historical-to-new authority manifest (audit section 659.2 item 3).

Read-only except for its one output, written once. Starting from the ledger-pinned incumbent
constants in scripts/phase3_selection_matrix.py it records:

* historical authorities with their approval sections: the P3 refit aggregate (section 285), its
  selected-N authority and recipe authority, every refit record and the config.pt each one reaches
  through its query extraction manifest, the p3gE selection records, the p3lamA receipt (section
  536) and its incumbent seed records;
* dataset / N / recipe coordinates of the incumbent;
* the new source generation: branch, commit, a clean-tree check, and the digest of every file of
  the launcher's executable closure plus the anchor-confirmation files;
* the contract digest;
* two read-only checks, re-run here: this generation's control refit recipe against each approved
  config.pt, and the reducer's evidence rule applied to every proposed control-reuse record.
A pin proves byte identity, not correctness or permission to execute.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import scripts.phase3_selection_matrix as M                  # noqa: E402
import scripts.anchor_confirm_decision as D                  # noqa: E402

CONTRACT = REPO / "docs" / "ANCHOR_CONFIRMATION_CONTRACT_v1.md"
P3LAM_RECEIPT = M.APPROVED_P3_REFIT_AGGREGATE.parent / "p3lamA_sweep_complete.json"
ANCHOR_FILES = ("dna_utils/scientific_recipe.py", "scripts/phase3_selection_matrix.py",
                "scripts/anchor_confirm_decision.py", "scripts/anchor_confirm_code_axis.py",
                "scripts/anchor_confirm_manifest.py", "train_siglip2.py", "config.py",
                "model_siglip2.py", "docs/ANCHOR_CONFIRMATION_CONTRACT_v1.md",
                "tests/test_anchor_confirm_port.py", "tests/test_anchor_confirm_recipe.py",
                "tests/test_anchor_confirm_launcher.py", "tests/test_anchor_confirm_reducer.py")


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                          check=True).stdout.strip()


def build(p3lam_receipt_sha256: str) -> dict:
    consumed = D.Consumed()
    aggregate = consumed.json(M.APPROVED_P3_REFIT_AGGREGATE, M.APPROVED_P3_REFIT_AGGREGATE_SHA256)
    selected = consumed.json(M.APPROVED_SELECTED_N, M.APPROVED_SELECTED_N_SHA256)
    incumbent = M.anchor_incumbent()
    records_dir = M.APPROVED_P3_REFIT_AGGREGATE.parent

    refit = {}
    for ds in M.ANCHOR_DATASETS:
        for seed, want in sorted(aggregate["datasets"][ds]["record_sha256"].items()):
            names = [p for p in records_dir.glob("p3rfB_*.json")
                     if hashlib.sha256(p.read_bytes()).hexdigest() == want]
            D.need(len(names) == 1, f"{ds}/{seed}: {len(names)} refit records carry {want[:12]}")
            record = consumed.json(names[0], want)
            query = Path(record["run_dir"]) / "extraction_manifest_query.json"
            manifest = consumed.json(query, record["completion"]["extraction_manifest_sha256"]["query"])
            consumed.read(manifest["config_path"], manifest["config_sha256"])
            refit.setdefault(ds, {})[seed] = {
                "record": names[0].name, "record_sha256": want, "run_dir": record["run_dir"],
                "query_extraction_manifest_sha256": record["completion"]["extraction_manifest_sha256"]["query"],
                "config_pt_sha256": manifest["config_sha256"],
                "final_checkpoint_sha256": record["completion"]["final_checkpoint_sha256"]}

    reuse = {}
    for ds in M.ANCHOR_DATASETS:
        for n in M.CANDIDATE_N:
            name = next(k for k in selected["record_sha256"]
                        if k.startswith(f"p3gE_{M.DATASETS[ds]['exp']}_N{n}_s42"))
            entry = {"record": str(records_dir / name), "record_sha256": selected["record_sha256"][name]}
            reuse[f"{ds}|none|{n}|42"] = D.verify_record(consumed, (ds, "none", n, 42), entry, incumbent)
    lam = consumed.json(P3LAM_RECEIPT, p3lam_receipt_sha256)
    for seed in D.M.ANCHOR_DECIDE_SEEDS:
        cell = lam["cells"][f"flickr25k|N=4|P=0.6,0.95|JD=0.02|stage=select|seed={seed}"]
        entry = {"record": str(records_dir / cell["record"]), "record_sha256": cell["record_sha256"]}
        reuse[f"flickr25k|none|4|{seed}"] = D.verify_record(
            consumed, ("flickr25k", "none", 4, seed), entry, incumbent)

    incumbent_checks = {ds: M.anchor_incumbent_recipe_check(ds, incumbent)
                        for ds in M.ANCHOR_DATASETS}
    D.need(all(not c["differences"] for c in incumbent_checks.values()),
           "this generation's control refit recipe differs from an approved one")

    status = git("status", "--porcelain")
    D.need(status == "", f"the source generation has uncommitted changes:\n{status[:400]}")
    closure = sorted(set(M._BOOTSTRAP_SOURCE_PATHS) | set(ANCHOR_FILES))
    import torch
    manifest = {
        "artifact_kind": "anchor_confirmation_authority_manifest",
        "version": M.ANCHOR_CONFIRM_VERSION,
        "note": "byte identities and read-only checks; not approval to execute",
        "historical": {
            "approved_p3_refit_aggregate": {"path": str(M.APPROVED_P3_REFIT_AGGREGATE),
                                            "sha256": M.APPROVED_P3_REFIT_AGGREGATE_SHA256,
                                            "approval": "ledger section 285 (2026-09-09)"},
            "approved_selected_n": {"path": str(M.APPROVED_SELECTED_N),
                                    "sha256": M.APPROVED_SELECTED_N_SHA256,
                                    "namespace": selected["namespace"],
                                    "protocol_sources": selected["protocol_sources"],
                                    "records_sha256": selected["record_sha256"]},
            "recipe_authority": selected["recipe_authority"]["artifact"],
            "refit_records": refit,
            "lambda_campaign": {"receipt": str(P3LAM_RECEIPT), "sha256": p3lam_receipt_sha256,
                                "approval": "ledger section 536"},
            "p3_commit": "5304005cb6eaa6462a5450c30c25bc65f978af23",
            "base_commit": "88c3a25b1b309550eafc276c2ce5be7575507173",
            "not_authority": {"artifacts/phase3_selection/selected_n.json in the new worktree":
                              "f2218aa7 -- an older committed copy; the approved bytes are "
                              "APPROVED_SELECTED_N"},
        },
        "coordinates": {"datasets": list(M.ANCHOR_DATASETS), "arms": list(M.ANCHOR_ARMS),
                        "candidate_n": list(M.CANDIDATE_N), "select_seed": M.SEED,
                        "decide_seeds": list(M.ANCHOR_DECIDE_SEEDS),
                        "incumbent": {ds: {**v, "topp": list(v["topp"])} for ds, v in incumbent.items()}},
        "new_generation": {
            "worktree": str(REPO), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "commit": git("rev-parse", "HEAD"), "clean": True,
            "files_sha256": {rel: hashlib.sha256((REPO / rel).read_bytes()).hexdigest()
                             for rel in closure if (REPO / rel).is_file()},
            "environment": {"python": platform.python_version(), "torch": torch.__version__,
                            "interpreter": sys.executable}},
        "contract": {"path": str(CONTRACT.relative_to(REPO)),
                     "sha256": hashlib.sha256(consumed.read(CONTRACT)).hexdigest()},
        "evidence": {
            "incumbent_recipe_check": incumbent_checks,
            "control_reuse_check": {k: {kk: vv for kk, vv in v.items()} for k, v in reuse.items()},
        },
        "_consumed": consumed,
    }
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--p3lam-receipt-sha256", required=True,
                        help="full digest of the p3lamA receipt as the ledger (section 536) records it")
    parser.add_argument("--out", default=str(M.ANCHOR_RECORD_DIR / "authority_manifest_v1.json"))
    args = parser.parse_args(argv)
    try:
        payload = build(args.p3lam_receipt_sha256)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        digest = D.write_once(Path(args.out), payload)
    except (D.NotReducible, M.CellRefused, subprocess.CalledProcessError, FileExistsError) as error:
        print(f"[anchor-manifest] REFUSED: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(f"[anchor-manifest] wrote {args.out} sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
