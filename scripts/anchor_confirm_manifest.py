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
import scripts.anchor_refit_stage as RT                      # noqa: E402

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
    pins = M.historical_input_verifier_pins()          # audit 715: the historical verifier and sources
    consumed.read(M.HISTORICAL_INPUT_ROOT / pins["verifier"]["path"], pins["verifier"]["sha256"])
    for rel, digest in sorted(pins["sources_sha256"].items()):
        consumed.read(M.HISTORICAL_INPUT_ROOT / rel, digest)
    v7 = M.anchor_v7_pins()                           # audit 733.3: the accepted v7 digests, read
    for key in ("selection", "decision", "stage_s_receipt"):
        consumed.read(v7[key]["path"], v7[key]["sha256"])
    freeze = consumed.json(RT.ANCHOR_F_RECORD, RT.ANCHOR_F_RECORD_SHA256)   # audit 744.1, read
    seals = {ds: {"path": str(path), "sha256": hashlib.sha256(consumed.read(path)).hexdigest()}
             for ds, path in sorted(RT.REFIT_INPUT_SEALS.items())}
    D.need(all(seals[ds]["sha256"] == RT.REFIT_INPUT_SEALS_SHA256[ds] for ds in seals),
           f"the refit input seals are not the approved bytes: {seals}")
    status = git("status", "--porcelain")
    D.need(status == "", f"the source generation has uncommitted changes:\n{status[:400]}")
    closure = M.anchor_generation_closure()
    absent = [rel for rel in closure if not (REPO / rel).is_file()]
    D.need(not absent, f"closure members are absent: {absent}")
    from importlib import metadata
    return {
        "artifact_kind": M.ANCHOR_MANIFEST_KIND, "version": M.ANCHOR_CONFIRM_VERSION,
        "generation": "v9", "revision": 3,
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
            "historical_input_verifier": pins,
            "anchor_v7": v7,
            "p3_commit": "5304005cb6eaa6462a5450c30c25bc65f978af23",
            "base_commit": "88c3a25b1b309550eafc276c2ce5be7575507173",
            "not_authority": {"artifacts/phase3_selection/selected_n.json in the new worktree":
                              "f2218aa7 -- an older committed copy"},
        },
        "coordinates": {"datasets": list(M.ANCHOR_DATASETS), "arms": list(M.ANCHOR_ARMS),
                        "candidate_n": list(M.CANDIDATE_N), "select_seed": M.SEED,
                        "decide_seeds": list(M.ANCHOR_DECIDE_SEEDS)},
        "design": {"fixed_architecture": "axis_center=anchors for all four datasets (audit 709)",
                   "run_arms": list(M.ANCHOR_RUN_ARMS), "rendered_comparator": "none",
                   "datasets": list(M.ANCHOR_DATASETS), "reuse": "none: every S/D cell is fresh"},
        "lambda": {"stage": M.ANCHOR_LAMBDA_STAGE, "scope": list(M.ANCHOR_LAMBDA_SCOPE),
                   "control": M.ANCHOR_LAMBDA_CONTROL,
                   "candidates": {ds: ["=".join(c) for c in M.ANCHOR_LAMBDA_CANDIDATES[ds]]
                                  for ds in M.ANCHOR_LAMBDA_SCOPE},
                   "incumbent_lambdas": {ds: dict(M.LAMBDA_INCUMBENT[ds]) for ds in M.ANCHOR_LAMBDA_SCOPE},
                   "scope_decision": "the user's decision recorded under audit 733.1 (2026-10-01): "
                                     "Flickr25K first; expand only if its choice moves"},
        "freeze": {"path": str(RT.ANCHOR_F_RECORD), "sha256": RT.ANCHOR_F_RECORD_SHA256,
                   "acceptance": f"audit ledger section {RT.ANCHOR_F_ACCEPTANCE_SECTION}",
                   "datasets": {ds: {k: freeze["datasets"][ds][k]
                                     for k in ("N", "lambdas", "routing_adaptive_topp", "lambda_codon_joint")}
                                for ds in M.ANCHOR_DATASETS}},
        "refit": {"stages": [RT.REFIT_STAGE, RT.TEST_STAGE], "seeds": list(RT.REFIT_SEEDS),
                  "protocol_fields": list(RT.REFIT_PROTOCOL_FIELDS),
                  "seal_fields": list(RT.REFIT_SEAL_FIELDS), "input_seals": seals,
                  "official_test_outputs": list(M.OFFICIAL_TEST_OUTPUTS), "t_chain": list(RT.T_CHAIN)},
        "predecessor": {"authority_manifest_v8r2_sha256": RT.ANCHOR_V8_MANIFEST_SHA256,
                        "superseded_v9r2_manifest_sha256":
                        "b7e9264a7a933e3efd30125628f638bc5c5304f32ccae2055b730ad382b215f6",
                        "v9_revision_3_change": "audits 746-754: the T entry claims its entry exclusively, "
                                                "binds the config bytes, full typed recipe, campaign "
                                                "evidence and terminal epoch, and builds its arguments "
                                                "from the one verified read (extraction_siglip2 gains "
                                                "_apply_saved_config, its flat resume step); the F "
                                                "acceptance is the exact section-744 digest; T producers "
                                                "run under a boundary snapshot re-checked between them; "
                                                "the supervisor counts five managed children per T cell",
                        "superseded_v9r1_manifest_sha256":
                        "19acd584065d59050a4f25ead354b378fadb5d116808addc7cffd78bd1ff72fa",
                        "v9_revision_2_change": "mutation battery v13 attempt 1 (bdaba58) detected 22/24 "
                                                "as declared: tests/test_anchor_refit_stage.py gains the "
                                                "run-scope-line-does-not-approve-a-smoke test; no source "
                                                "behaviour changes",
                        "v9_change": "stages R and T (audits 742-744, contract R/T v1): the anchor "
                                     "refit of the accepted F record and its separately approved "
                                     "official test. The trainer withholds the official test in an "
                                     "anchor stage-R cell and refuses an anchor P0 refit outside one "
                                     "(p0_protocol); its terminal block moves unchanged into "
                                     "terminal_official_test.py, which the new T entry "
                                     "scripts/anchor_terminal_test.py also calls; the launcher admits "
                                     "anchor refits only through scripts/anchor_refit_stage.py, tags "
                                     "them with the arm, runs no post-chain for them and requires their "
                                     "run directories to hold no test output; the supervisor gains the "
                                     "stage-R/T ledger. Model, data, loss, wrappers, environment, "
                                     "seals and every S/D/L record are unchanged",
                        "authority_manifest_v7_sha256": M.ANCHOR_V7_MANIFEST_SHA256,
                        "superseded_v8_manifest_sha256":
                        "b7b5af9ed13473ed21773ddf824c951c6064a8cdeeb18d61c3a8a8777ab3572d",
                        "v8_revision_2_change": "audit 734.2: one approved campaign per stage-L "
                                                "decision; every historical read through the "
                                                "reduction's read-once record; synthetic identity "
                                                "artifacts in the binding test; this continuity "
                                                "statement corrected",
                        "v8_change": "stage L (audit 731-733, contract L v1): the TODO 13-15 lambda "
                                     "checks of the fixed anchor model, Flickr25K first, one declared "
                                     "lambda per anchor cell at the v7 frozen N, seed 42, with a "
                                     "continuity control first; the v7 frozen N record, stage-D summary "
                                     "and stage-S receipt are bound by their accepted digests and never "
                                     "replayed; launcher, reducer (lambda role), supervisor (stage-L "
                                     "ledger and budget) and this builder change, and the recipe "
                                     "parser dna_utils/scientific_recipe.py gains one separate table "
                                     "that admits a single reviewed lambda repeat per argv (the "
                                     "audit 734.1 exception; the parser does not authenticate a "
                                     "stage, its callers do); the trainer, model, data, wrappers, "
                                     "environment, seal and probe files are byte-equal to v7",
                        "authority_manifest_v6_sha256":
                        "a5ff2a0e93acd3a0bef7ecc47a69d550b28fc7ee57c1a25aa48e7e1169f739d7",
                        "stage_s_request_v6_sha256":
                        "1625b50faf74b311040a8f56f1d7103b7352e2733e8c8fe9b8860ed18a44793c",
                        "stage_s_snapshot_v6_sha256":
                        "ef5a3e8d857a7a6cc22d1e978854c3f721470ed19f426a020873718199a2b8ff",
                        "authority_manifest_v5_sha256":
                        "84e94e2f1897e9910df2aa0352e4737f87dd485f018a644b5be23c1b4e036284",
                        "stage_s_request_v5_sha256":
                        "bd3117a9108ee558dfb27f9b429806a62092d23736febd7c0c0f8c858e4882c7",
                        "authority_manifest_v4_sha256":
                        "5a4481f4898fda3df27250e0214e204495022a91546d7d78e2e1c2749e17647b",
                        "contract_v2_sha256":
                        "26ebe2c104e89702bbc9daef0a3470d123ef6f7c5c30eefc49222cb26e29b001",
                        "authority_manifest_v3_sha256":
                        "9e54bda303d6c34f080e8026492ff7eec7bfe432a6e53bd34a5a6056881a9b83",
                        "authority_manifest_v2_sha256":
                        "a5f7b8436e3e81b189deedb4d1f3e99eb993bce59d8792bada12f7c1a6a1bf0a",
                        "authority_manifest_v1_sha256":
                        "c1eed986312ba9a017cc559813ce1d0d2b2dc2fc5aba1df3f032424cbda8b94e",
                        "contract_v1_sha256":
                        "418091eedff7ee8c09df149df6bc7027f6b4ac0d8ebbe3dd724facefa66a5a08",
                        "v3_change": "campaign child lifecycle (audit 705/706), the pre-dispatch "
                                     "storage rule and the operational supervisor (703/704); the "
                                     "S/D scientific design and the contract are unchanged",
                        "v4_change": "the supervisor's accounting only (audit 707.1): observation "
                                     "windows, a watchdog and an unresolved state; the launcher "
                                     "and the S/D scientific design are unchanged",
                        "v5_change": "contract v3 (audit 709): axis_center=anchors fixed for all "
                                     "four datasets, anchor arm only, CIFAR-10 added, the "
                                     "adoption rule removed, a 500-row probe population, four "
                                     "GPUs; supersedes the three-dataset v2-v4 request",
                        "v6_change": "historical input verification (audit 715/716): the full "
                                     "input check runs the historical tree's pinned verifier in an "
                                     "isolated child after the six historical sources are checked "
                                     "in place; the v5 stage-S attempt (ancS5, run "
                                     "20260927T143532Z-4ada8a0d) was refused at that check and "
                                     "trained nothing; the S/D design is unchanged",
                        "v7_change": "child environment handoff (audit 722/723): build_command takes "
                                     "the four attested runtime variables from the launcher's "
                                     "start-up block (absent ones removed); a carried admission "
                                     "must be a full historical-verifier admission; the v6 "
                                     "attempt (ancS6, run 20260928T052252Z-35772b95) admitted all "
                                     "four seals, then every trainer refused on library_environment "
                                     "and nothing trained; the S/D design is unchanged"},
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
        "lambda_contract": {"path": M.ANCHOR_LAMBDA_CONTRACT_PATH,
                            "sha256": hashlib.sha256(consumed.read(REPO / M.ANCHOR_LAMBDA_CONTRACT_PATH)).hexdigest()},
        "refit_contract": {"path": M.ANCHOR_REFIT_CONTRACT_PATH,
                           "sha256": hashlib.sha256(consumed.read(REPO / M.ANCHOR_REFIT_CONTRACT_PATH)).hexdigest()},
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
    inv.add_argument("--out", default=str(M.ANCHOR_RECORD_DIR / "authority_manifest_v9r3.json"))
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
