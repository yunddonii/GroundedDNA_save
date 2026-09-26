#!/usr/bin/env python
"""Anchor confirmation v1: the versioned generation manifest (audit sections 659.2, 674, 678.2, 683.2).

inventory -- the manifest the audit reviews: byte identities of the historical JSON authorities
    (their full digests are this source's constants, taken from the ledger), of EVERY member of the
    generation closure (the launcher's executable closure, which includes the pinned wrappers, the
    input-admission, identity and split modules, plus the anchor reducer, probe, this script, the
    four test files and the designated contract), and the environment. A missing member refuses
    instead of being left out. It reads JSON and text only; no config.pt, checkpoint, array or
    cache is opened. The launcher's --smoke/--run, the reducer and the probe require the tree to
    match it; approval to run anything is a separate audit ledger line (launcher `audit_approval`).
The historical config inspection is not part of this generation (a stage-R proposal).
The output is written once. A pin proves byte identity, not correctness or permission to execute.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import platform
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import scripts.phase3_selection_matrix as M                  # noqa: E402
import scripts.anchor_confirm_decision as D                  # noqa: E402

#: the bytes this module was imported from (audit 697)
with open(__file__, "rb") as _source:
    _IMPORTED_SOURCE_SHA256 = hashlib.sha256(_source.read()).hexdigest()
CONTRACT = REPO / M.ANCHOR_CONTRACT_PATH
P3LAM_RECEIPT = M.APPROVED_P3_REFIT_AGGREGATE.parent / "p3lamA_sweep_complete.json"
#: Ledger section 536.1 (repeated in 561.1): the approved lambda-confirmation receipt.
P3LAM_RECEIPT_SHA256 = "5a8901b76c5e0b97e0daea5f1564bf1f1a7cf05a79212abd148f601df91894d2"


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                          check=True).stdout.strip()


def inventory() -> dict:
    consumed = D.Consumed()
    aggregate = consumed.json(M.APPROVED_P3_REFIT_AGGREGATE, M.APPROVED_P3_REFIT_AGGREGATE_SHA256)
    selected = consumed.json(M.APPROVED_SELECTED_N, M.APPROVED_SELECTED_N_SHA256)
    consumed.read(P3LAM_RECEIPT, P3LAM_RECEIPT_SHA256)
    status = git("status", "--porcelain")
    D.need(status == "", f"the source generation has uncommitted changes:\n{status[:400]}")
    closure = M.anchor_generation_closure()
    absent = [rel for rel in closure if not (REPO / rel).is_file()]
    D.need(not absent, f"closure members are absent: {absent}")
    from importlib import metadata
    return {
        "artifact_kind": M.ANCHOR_MANIFEST_KIND, "version": M.ANCHOR_CONFIRM_VERSION,
        "generation": "v2",
        "note": "byte identities from JSON and text only; not approval to execute",
        "historical": {
            "approved_p3_refit_aggregate": {"path": str(M.APPROVED_P3_REFIT_AGGREGATE),
                                            "sha256": M.APPROVED_P3_REFIT_AGGREGATE_SHA256,
                                            "approval": "ledger section 285 (2026-09-09)",
                                            "refit_record_sha256": {ds: aggregate["datasets"][ds]["record_sha256"]
                                                                    for ds in M.ANCHOR_DATASETS}},
            "approved_selected_n": {"path": str(M.APPROVED_SELECTED_N),
                                    "sha256": M.APPROVED_SELECTED_N_SHA256,
                                    "namespace": selected["namespace"],
                                    "protocol_sources": selected["protocol_sources"],
                                    "records_sha256": selected["record_sha256"]},
            "recipe_authority": selected["recipe_authority"]["artifact"],
            "lambda_campaign": {"receipt": str(P3LAM_RECEIPT), "sha256": P3LAM_RECEIPT_SHA256,
                                "approval": "ledger section 536.1"},
            "p3_commit": "5304005cb6eaa6462a5450c30c25bc65f978af23",
            "base_commit": "88c3a25b1b309550eafc276c2ce5be7575507173",
            "not_authority": {"artifacts/phase3_selection/selected_n.json in the new worktree":
                              "f2218aa7 -- an older committed copy"},
        },
        "coordinates": {"datasets": list(M.ANCHOR_DATASETS), "arms": list(M.ANCHOR_ARMS),
                        "candidate_n": list(M.CANDIDATE_N), "select_seed": M.SEED,
                        "decide_seeds": list(M.ANCHOR_DECIDE_SEEDS)},
        "control_reuse": {"proposed": False,
                          "reason": "stages S and D train fresh controls (contract section 5); the "
                                    "v1 proposal stays as history in authority_manifest_v1.json",
                          "approved_reuse_admissions": dict(M.APPROVED_REUSE_ADMISSION_SHA256)},
        "predecessor": {"authority_manifest_v1_sha256":
                        "c1eed986312ba9a017cc559813ce1d0d2b2dc2fc5aba1df3f032424cbda8b94e",
                        "contract_v1_sha256":
                        "418091eedff7ee8c09df149df6bc7027f6b4ac0d8ebbe3dd724facefa66a5a08"},
        "new_generation": {
            "worktree": str(REPO), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "commit": git("rev-parse", "HEAD"), "clean": True,
            "files_sha256": {rel: hashlib.sha256(consumed.read(REPO / rel)).hexdigest()
                             for rel in closure},
            "dataset_scripts_sha256": dict(M.DATASET_SCRIPT_SHA256),
            "environment": {"python": platform.python_version(), "torch": metadata.version("torch"),
                            "interpreter": sys.executable}},
        "contract": {"path": M.ANCHOR_CONTRACT_PATH,
                     "sha256": hashlib.sha256(consumed.read(CONTRACT)).hexdigest()},
        "approval": {"authority": str(M.AUDIT_LEDGER), "tag": M.APPROVAL_TAG,
                     "scopes": {k: list(v) for k, v in M.APPROVAL_SCOPES.items()},
                     "note": "not part of this manifest: the audit writes one ledger line per "
                             "approved operation, naming this manifest's digest"},
        "_consumed": consumed,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")
    inv = sub.add_parser("inventory")
    inv.add_argument("--out", default=str(M.ANCHOR_RECORD_DIR / "authority_manifest_v2.json"))
    args = parser.parse_args(argv)
    try:
        if args.command != "inventory":
            parser.error("the only command is inventory")
        payload = inventory()
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        digest = D.write_once(Path(args.out), payload)
    except (D.NotReducible, M.CellRefused, subprocess.CalledProcessError, FileExistsError) as error:
        print(f"[anchor-manifest] REFUSED: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(f"[anchor-manifest] wrote {args.out} sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
