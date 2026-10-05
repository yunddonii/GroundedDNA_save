"""Stages R and T of the fixed anchor model (generation v9; docs/ANCHOR_REFIT_CONTRACT_v1.md; audits 742-744).

Stage R -- `--anchor-confirm refit`: the twelve scratch full-train refits of the accepted F record
(anchors on all four datasets, seeds 42/43/44 at the frozen N), each the F record's validated
recipe with exactly the contracted full-train fields changed. R ENDS at the terminal checkpoint:
the trainer withholds the official test (p0_protocol.anchor_refit_withholds_official_test), and the
launcher runs no post-chain and requires the run directory to hold no official-test output.

Stage T -- `--anchor-confirm test`: the one official-test extraction and evaluation of each stage-R
checkpoint, a separate approval naming the completed stage-R receipt through its request. Per cell:
an exclusive, durable attempt reservation BEFORE any model or test construction (retained on
failure; a second attempt refuses), then the T entry `scripts/anchor_terminal_test.py` (which
re-verifies the approval and the checkpoint before it touches the test) runs the trainer's own
terminal function (terminal_official_test.run_official_test: query/DB extraction + RAW evaluation),
then the unchanged refit post-chain (train extraction, post-BIO evaluation, NMI, analysis seal),
fail-stop, and the legacy refit output check.

The F authority is the accepted record's bytes AND its acceptance in audit section 744: the record
is read at its pinned digest and the ledger section must name it. Neither a filename nor the
record's own status text is approval. Every execution also needs an approval line for its exact
request (scopes stage-R-smoke/run, stage-T-smoke/run; `freeze` names the F record).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import socket
import sys
import threading
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import scripts.phase3_selection_matrix as M                 # noqa: E402
from scripts.phase3_selection_matrix import CellRefused     # noqa: E402

#: the bytes this module was imported from (audit 697)
with open(__file__, "rb") as _source:
    _IMPORTED_SOURCE_SHA256 = hashlib.sha256(_source.read()).hexdigest()

REFIT_STAGE, TEST_STAGE = "refit", "test"
#: The accepted F record (audit 744.1): its bytes plus the acceptance in that ledger section.
ANCHOR_F_RECORD = Path("/data/yschoi/gdna_anchor_lambda_v8/artifacts/anchor_confirmation/"
                       "ancF_candidate_v1.json")
ANCHOR_F_RECORD_SHA256 = "5165f5dc9fcfb8334270bc16aa9816d09db67b03a04abae7ff846d5235bdca1d"
ANCHOR_F_ACCEPTANCE_SECTION = 744
#: The exact accepted text (audits 745-746.3): SHA256 of ledger section 744 from its heading line to
#: the next section heading, stripped of surrounding whitespace -- the corrected body digest stated in
#: section 745. Any other text in that section (a token-only paragraph, changed acceptance language)
#: is not this acceptance. The mutable ledger as a whole is NOT pinned.
ANCHOR_F_ACCEPTANCE_SHA256 = "13ef776bc1485b3917253e51ee4aa99879a8f1f30f5539e0d1e224d3faaa1c56"
ANCHOR_F_KIND = "anchor_confirmation_freeze_candidate"
#: generation v8's manifest: stage R/T never run under it (nor under v7's)
ANCHOR_V8_MANIFEST_SHA256 = "58e69ae1a3bed5366da61a4d61a6cdd90c338d9ac7b0474fcf9b6e2e2f7f5bbc"
REFIT_SEEDS = (42, 43, 44)
#: The contracted S/D -> R mapping (contract R/T v1 section 3; audit 744.1): the fields in which a
#: stage-R recipe may differ from the F record's validated (stage-S seed-42) recipe. Every other
#: typed field must be equal, axis_center=anchors included.
REFIT_PROTOCOL_FIELDS = ("epoch", "lr_schedule_horizon", "sinkhorn_schedule_horizon",
                         "stop_after_epoch", "val_split_ratio", "selection_mode",
                         "keep_final_checkpoint", "final_epoch_eval", "text_whiten_npz",
                         "random_seed")
#: ... and the sealed input authority, which is the refit seal's in stage R (stage-1's in S/D)
REFIT_SEAL_FIELDS = ("phase3_input_seal", "phase3_input_seal_sha256",
                     "phase3_input_aggregate_sha256", "phase3_split_identity_sha256")
#: the approved refit input seals (the incumbent refit's; built in the historical tree, never edited)
REFIT_INPUT_SEALS = {ds: Path(f"/data/yschoi/gdna_p3exec_seals/{ds}.refit.input-seal.json")
                     for ds in ("cifar10", "flickr25k", "nuswide", "mscoco")}
REFIT_INPUT_SEALS_SHA256 = {
    "cifar10": "943bb953f44a9d943e66944c15ec0d82bcef95d9769eea243c8b22e3993dec9f",
    "flickr25k": "71506fd19966216f2fd25f952db7e16e58d980eebc9eaa4d9a5488e63022c83c",
    "nuswide": "05b7d24b7ede41c989ad3c6d510769949c5cc6d33ec843c34cc52151ee2992a4",
    "mscoco": "a9d49e5db9d4b2b934a9340b69bf986fae435c7d712783c379d28ed8f0f177e4",
}
T_REQUEST_SCHEMA = "anchor-terminal-test-request/1"
T_SNAPSHOT_SCHEMA = "anchor-terminal-test-snapshot/1"
T_ATTEMPT_SCHEMA = "anchor-terminal-test-attempt/1"
T_RECORD_SCHEMA = "anchor-terminal-test-record/1"
T_RECEIPT_SUFFIX = "_test_complete.json"
T_CAMPAIGN_KIND = "anchor_terminal_test"
#: the T chain, in order: the entry's terminal function, then the unchanged refit post-chain
T_CHAIN = ("scripts/anchor_terminal_test.py: extraction_siglip2.extract_code (query, db) + "
           "evaluation_siglip2.evaluation (raw, bio_project=False)",
           "scripts/extract_train_split.py", "scripts/eval_cell_bioproj.py --require-train",
           "scripts/pairwise_nmi.py --require-train", "scripts/seal_cell_analysis.py --require-train")


def ledger_section_text(text: str, section: int) -> str:
    """Ledger section `section` from its own heading line to the next section heading, stripped of
    surrounding whitespace; exactly one such heading must exist."""
    starts = [m for m in re.finditer(r"^## (\d+)\. .*$", text, flags=re.M) if int(m.group(1)) == section]
    if len(starts) != 1:
        raise CellRefused(f"the audit ledger has {len(starts)} sections numbered {section}")
    following = re.compile(r"^## ", flags=re.M).search(text, starts[0].end())
    return text[starts[0].start():following.start() if following else len(text)].strip()


def anchor_freeze_acceptance() -> dict:
    """The F acceptance, bound exactly: ledger section 744's text must hash to the accepted digest and
    name the record's path and digest."""
    try:
        text = M.AUDIT_LEDGER.read_text(encoding="utf-8")
    except OSError as error:
        raise CellRefused(f"the audit ledger is unreadable: {error}") from None
    body = ledger_section_text(text, ANCHOR_F_ACCEPTANCE_SECTION)
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    flat = " ".join(body.split())
    if digest != ANCHOR_F_ACCEPTANCE_SHA256 or str(ANCHOR_F_RECORD) not in flat \
            or ANCHOR_F_RECORD_SHA256 not in flat:
        raise CellRefused(f"ledger section {ANCHOR_F_ACCEPTANCE_SECTION} is not the exact accepted text "
                          f"(sha256 {digest[:12]}..., want {ANCHOR_F_ACCEPTANCE_SHA256[:12]}...) naming "
                          f"{ANCHOR_F_RECORD} at {ANCHOR_F_RECORD_SHA256[:12]}...")
    return {"ledger": str(M.AUDIT_LEDGER), "section": ANCHOR_F_ACCEPTANCE_SECTION,
            "section_sha256": digest}


def _read_json(path, sha256, what, read=None):
    raw = (read or M._read_bytes)(path, sha256, what)
    try:
        return json.loads(raw), hashlib.sha256(raw).hexdigest()
    except ValueError as error:
        raise CellRefused(f"{what}: {path} is not JSON: {error}") from None


def anchor_freeze_authority(read=None) -> dict:
    """The accepted F record (audit 744.1), read at its pinned digest, AND its acceptance: ledger
    section 744 must name the record's path and digest as the accepted record. Per dataset: the
    frozen N, top-p window, joint weight and lambdas, and the validated typed recipe fields, read
    from the v7 stage-S plan snapshot the record binds (pinned bytes) and re-digested."""
    from dna_utils.scientific_recipe import RecipeMismatch, check_payload_shape, digest
    record, _ = _read_json(ANCHOR_F_RECORD, ANCHOR_F_RECORD_SHA256, "the accepted F record", read)
    acceptance = anchor_freeze_acceptance()
    if record.get("artifact_kind") != ANCHOR_F_KIND \
            or record.get("fixed_architecture", {}).get("axis_center") != "anchors" \
            or record.get("fixed_architecture", {}).get("datasets") != list(M.ANCHOR_DATASETS) \
            or set(record.get("datasets") or {}) != set(M.ANCHOR_DATASETS):
        raise CellRefused("the F record does not fix anchors for exactly the four datasets")
    snapshots, out = {}, {}
    for ds in M.ANCHOR_DATASETS:
        entry = record["datasets"][ds]
        recipe = entry["validated_recipe"]
        pin = (recipe["snapshot"]["path"], recipe["snapshot"]["sha256"])
        if pin not in snapshots:
            snapshots[pin] = _read_json(*pin, "the F record's validated-recipe snapshot", read)[0]
        binding = (snapshots[pin].get("plan") or {}).get("cell_bindings", {}).get(recipe["cell_id"])
        payload = (binding or {}).get("scientific_recipe")
        try:
            check_payload_shape(payload)
        except RecipeMismatch as error:
            raise CellRefused(f"{ds}: the validated recipe is malformed: {error}") from None
        if digest(payload) != recipe["scientific_recipe_sha256"] \
                or binding.get("expected_scientific_recipe_sha256") != recipe["scientific_recipe_sha256"]:
            raise CellRefused(f"{ds}: the validated recipe is not the F record's digest")
        fields = payload["fields"]
        n, topp = int(entry["N"]), entry["routing_adaptive_topp"]
        if fields.get("axis_center") != "anchors" or fields.get("use_gumbel_softmax") is not False \
                or fields.get("stop_after_epoch") != n or binding.get("N") != n \
                or [fields.get("routing_adaptive_topp_min"), fields.get("routing_adaptive_topp_max")] != topp \
                or fields.get("lambda_codon_joint") != entry["lambda_codon_joint"] \
                or any(fields.get(k) != v for k, v in entry["lambdas"].items()):
            raise CellRefused(f"{ds}: the F record's summary values are not its validated recipe's")
        out[ds] = {"N": n, "topp": (str(topp[0]), str(topp[1])), "joint": str(entry["lambda_codon_joint"]),
                   "fields": fields, "recipe_sha256": recipe["scientific_recipe_sha256"],
                   "cell_id": recipe["cell_id"]}
    return {"record": {"path": str(ANCHOR_F_RECORD), "sha256": ANCHOR_F_RECORD_SHA256},
            "acceptance": acceptance, "datasets": out}


def _check_freeze_against_incumbent(freeze: dict, incumbent: dict) -> None:
    """The F record's top-p/joint are the approved recipe authorities' (the launcher's incumbent)."""
    for ds, entry in freeze["datasets"].items():
        inc = incumbent[ds]
        if (tuple(map(float, entry["topp"])), float(entry["joint"])) != \
                (tuple(map(float, inc["topp"])), float(inc["joint"])):
            raise CellRefused(f"{ds}: the F record's top-p/joint are not the approved incumbent's")


def refit_cells(freeze: dict, incumbent: dict) -> list:
    """Eight-tuples `(dataset, N, topp, joint, "refit", seed, (), "anchors")`: every dataset at its
    frozen N, seeds 42/43/44 -- twelve cells, in dataset order."""
    return [(ds, freeze["datasets"][ds]["N"], tuple(incumbent[ds]["topp"]), str(incumbent[ds]["joint"]),
             REFIT_STAGE, seed, (), "anchors")
            for ds in M.ANCHOR_DATASETS for seed in REFIT_SEEDS]


def refit_protocol_fields(dataset: str, n: int, *, seed: int, arm: str, incumbent: dict,
                          epochs=None) -> dict:
    """The typed values the contract fixes for one stage-R cell: the stage-1 contract values of the
    approved recipe, with the full-train refit mapping (-e N+1, both horizons unset -> N+1, stop N,
    full designated train, refit mode, the terminal checkpoint kept by final_epoch_eval, the
    trainOnly whitening) and `-ev` (the wrapper's; the trainer withholds the test in stage R). A
    smoke shortens the horizons only."""
    fields = M.anchor_protocol_fields(dataset, n, seed=seed, arm=arm, incumbent=incumbent)
    stop = n if epochs is None else max(int(epochs) - 1, 0)
    fields.update({
        "epoch": n + 1 if epochs is None else int(epochs),
        "lr_schedule_horizon": None if epochs is None else int(epochs),
        "sinkhorn_schedule_horizon": None if epochs is None else stop + 1,
        "stop_after_epoch": stop, "selection_mode": "refit", "keep_final_checkpoint": False,
        "final_epoch_eval": True, "val_split_ratio": 0.0,
        "text_whiten_npz": M._whitening(M.DATASETS[dataset], stage=REFIT_STAGE),
        "evaluation": True,
    })
    return fields


def refit_admission(cells, *, namespace: str, incumbent: dict, freeze: dict, epochs=None,
                    input_seals=None) -> dict:
    """Before any lease, reservation or dispatch, at every stage-R coordinate: the anchor arm renders
    with the stage-R protocol values; its typed recipe differs from the F record's validated recipe
    only in the contracted refit fields (and, with the admitted seals, in the input-authority fields,
    which must then be the refit seal's); and the rendered control differs from it in axis_center
    alone. Nothing is reserved, leased or launched; the wrappers render in temporary directories."""
    from dna_utils.scientific_recipe import (admitted_overrides, canonical, digest,
                                             field_differences)
    from config import Config
    parser = Config.build_parser()
    report = {}
    for cell in cells:
        ds, n, topp, joint, stage, seed = M._campaign_cell_parts(cell)
        key = f"{ds}|N={n}|seed={seed}"
        if stage != REFIT_STAGE or M._cell_anchor_arm(cell) != "anchors" or M._cell_overrides(cell):
            raise CellRefused(f"{key}: not a stage-R anchor cell")
        if n != freeze["datasets"][ds]["N"] or seed not in REFIT_SEEDS:
            raise CellRefused(f"{key}: not the F record's frozen N or a refit seed")
        authority = (input_seals or {}).get(f"{ds}:{REFIT_STAGE}") if input_seals is not None else None
        if input_seals is not None and authority is None:
            raise CellRefused(f"{key}: no admitted {ds}:{REFIT_STAGE} input seal")
        payloads = {arm: M.anchor_scientific_recipe(ds, n, namespace=namespace, stage=REFIT_STAGE,
                                                    seed=seed, topp=topp, joint=joint, epochs=epochs,
                                                    anchor_arm=arm, input_authority=authority)
                    for arm in M.ANCHOR_ARMS}
        for arm, payload in payloads.items():
            want = refit_protocol_fields(ds, n, seed=seed, arm=arm, incumbent=incumbent, epochs=epochs)
            wrong = sorted(k for k, v in want.items()
                           if k not in payload["fields"] or canonical(payload["fields"][k]) != canonical(v))
            if wrong:
                raise CellRefused(f"{key} {arm}: the rendered refit breaks the protocol in "
                                  + ", ".join(f"{k}={payload['fields'].get(k)!r} (contract {want[k]!r})"
                                              for k in wrong[:6]))
        if field_differences(payloads["none"]["fields"], payloads["anchors"]["fields"]) != ["axis_center"]:
            raise CellRefused(f"{key}: the rendered arms differ in more than axis_center")
        fields = payloads["anchors"]["fields"]
        moved = field_differences(freeze["datasets"][ds]["fields"], fields)
        # Without admitted seals (a --plan preview) the render carries no input-authority flags, so
        # those fields are not compared; with them, only the four seal fields may differ (the CLIP
        # snapshot identity is the F recipe's) and they must be the refit seal's (below).
        exempt = set(REFIT_SEAL_FIELDS) if authority is not None else set(M.INPUT_AUTHORITY_DESTS)
        beyond = sorted(set(moved) - set(REFIT_PROTOCOL_FIELDS) - exempt)
        if beyond:
            raise CellRefused(f"{key}: the refit recipe differs from the F record's validated recipe "
                              f"in {beyond}, outside the contracted refit mapping")
        if authority is not None:
            sealed = {"phase3_input_seal": authority.get("seal_path"),
                      "phase3_input_seal_sha256": authority.get("seal_file_sha256"),
                      "phase3_input_aggregate_sha256": authority.get("aggregate_sha256"),
                      "phase3_split_identity_sha256": authority.get("split_identity_sha256")}
            off = sorted(k for k, v in sealed.items() if canonical(fields.get(k)) != canonical(v))
            if off or authority.get("stage") != REFIT_STAGE:
                raise CellRefused(f"{key}: the sealed input authority is not the admitted refit seal's: {off}")
        report[key] = {"digest": digest(payloads["anchors"]), "differs_from_f": moved,
                       "input_authority_compared": authority is not None,
                       "overrides": admitted_overrides(parser, payloads["anchors"]["argv"])}
    return report


# ----------------------------------------------------------------------------------------------- R

def _refit_main(args, manifest, execute: bool) -> int:
    incumbent = M.anchor_incumbent()
    freeze = anchor_freeze_authority()
    _check_freeze_against_incumbent(freeze, incumbent)
    cells = refit_cells(freeze, incumbent)
    epochs = args.epochs if args.smoke else None
    admission = refit_admission(cells, namespace=args.namespace, incumbent=incumbent,
                                freeze=freeze, epochs=epochs)
    approval = request = None
    input_seals, historical_admission = None, {}
    if execute:
        if args.smoke and not args.only:
            raise CellRefused("a stage-R smoke needs --only dataset:N:arms:seed")
        if args.run and args.only:
            raise CellRefused("--run takes the whole stage; --only is for --smoke")
        scope = f"stage-R-{'smoke' if args.smoke else 'run'}"
        request = refit_request(args, cells, manifest_sha256=manifest["sha256"], freeze=freeze)
        want_seals = {f"{ds}:{REFIT_STAGE}": {"path": str(path.resolve()),
                                             "sha256": REFIT_INPUT_SEALS_SHA256[ds]}
                      for ds, path in REFIT_INPUT_SEALS.items()}
        if request["input_seals"] != want_seals:
            raise CellRefused(f"stage R takes exactly the approved refit input seals {want_seals}, "
                              f"got {request['input_seals']}")
        approval = M.audit_approval(args.anchor_approval_section, scope, manifest=manifest["sha256"],
                                    freeze=ANCHOR_F_RECORD_SHA256, request=M._json_digest(request))
        carried = None
        if getattr(args, "admission_authority", None):
            before = M._file_pin(args.admission_authority)
            carried = M.load_admission_authority(args.admission_authority)
            if not (before == M._file_pin(args.admission_authority) == request["admission_authority"]):
                raise CellRefused("the carried admission authority is not the approved request's bytes")
            refusal = M.anchor_carried_admission_refusal(args.admission_authority, carried,
                                                         expected_sha256=before["sha256"])
            if refusal is not None:
                raise CellRefused(refusal)
        input_seals = M.verify_campaign_input_seals(
            args.input_seal_specs, cells, full=M.admission_is_full(carried), expected=carried,
            historical=True, evidence=historical_admission)
        M.assert_request_seals(request, input_seals, "at input admission")
        # the sealed recipes, rendered with the admitted refit seals, against the F record again
        admission = refit_admission(cells, namespace=args.namespace, incumbent=incumbent,
                                    freeze=freeze, epochs=epochs, input_seals=input_seals)
        M.recheck_generation(manifest, "after input admission")
    authorities = {
        "approved_p3_refit_aggregate": {"path": str(M.APPROVED_P3_REFIT_AGGREGATE),
                                        "sha256": M.APPROVED_P3_REFIT_AGGREGATE_SHA256},
        "approved_selected_n": {"path": str(M.APPROVED_SELECTED_N), "sha256": M.APPROVED_SELECTED_N_SHA256},
        "incumbent": {ds: {**v, "topp": list(v["topp"])} for ds, v in incumbent.items()},
        "anchor_freeze": {**freeze["record"], "acceptance": freeze["acceptance"],
                          "recipes_sha256": {ds: e["recipe_sha256"] for ds, e in freeze["datasets"].items()}},
        "admission_sha256": M._json_digest(admission),
        "anchor_manifest": manifest, "anchor_approval": approval, "anchor_request": request,
        **({"historical_input_admission": historical_admission} if historical_admission else {}),
    }
    if not execute:
        preview = (refit_request(args, cells, manifest_sha256=manifest["sha256"], freeze=freeze)
                   if (args.run or args.smoke) and manifest is not None else None)
        return _print_plan(args, cells, authorities, admission, preview)
    M.RECORD_DIR = M.ANCHOR_RECORD_DIR
    return M._with_campaign_gpu_leases(
        args, lambda: M._run_sweep(args, None, full_plan=cells, authorities=authorities,
                                   preverified_input_seals=input_seals))


def refit_request(args, cells, *, manifest_sha256: str, freeze: dict) -> dict:
    """The stage-R request: the anchor request of the cells (stage `refit`), plus the F authority and
    each dataset's validated-recipe digest. An approval names its digest."""
    request = M.anchor_execution_request(args, cells, manifest_sha256=manifest_sha256)
    request["freeze"] = {**freeze["record"], "acceptance_section": ANCHOR_F_ACCEPTANCE_SECTION,
                         "acceptance_sha256": ANCHOR_F_ACCEPTANCE_SHA256,
                         "recipes_sha256": {ds: e["recipe_sha256"] for ds, e in freeze["datasets"].items()}}
    return request


def _print_plan(args, cells, authorities, admission, preview=None) -> int:
    print(f"{M.ANCHOR_CONFIRM_VERSION}: stage {args.anchor_confirm}, {len(cells)} cells, namespace "
          f"{args.namespace}  [NON-EXECUTABLE until audit approval]")
    print(json.dumps(authorities, indent=1, sort_keys=True, default=str))
    if args.anchor_confirm == REFIT_STAGE:
        for cell in cells:
            ds, n, topp, joint, stage, seed = M._campaign_cell_parts(cell)
            print(f"  {ds:<10} N={n:<3} seed={seed} arm={cell[7]} -> "
                  f"{M.refit_tag_for(ds, n, seed, namespace=args.namespace, topp=topp, joint=joint, anchor_arm=cell[7])}")
        print("each refit recipe differs from the F record's validated recipe only in the contracted "
              "refit fields; the rendered control differs in axis_center alone")
        for key, entry in admission.items():
            print(f"  {key}: {entry['digest'][:16]} differs from F in {entry['differs_from_f']}")
    if preview is not None:
        print("execution request that this command without --plan would need approved:")
        print(json.dumps(preview, indent=1, sort_keys=True))
        print(f"execution request sha256 {M._json_digest(preview)}")
    return 0


# ----------------------------------------------------------------------------------------------- T

def admit_refit_receipt(path, sha256, *, manifest_sha256: str, freeze: dict, mode: str) -> dict:
    """The completed stage-R campaign stage T evaluates, from JSON metadata only (no checkpoint byte
    is read here; the T entry hashes each checkpoint before it loads it): the receipt at its digest;
    its snapshot (semantic digest), generation manifest, F authority and stage-R approval, re-verified
    in the ledger now; and per cell the record, its campaign binding and completion pins, the trainer
    evidence and runtime sidecar at their pinned digests (terminal epoch), the sealed recipe, and a
    run directory that still holds no official-test output. `mode` run: exactly the twelve F cells;
    smoke: one."""
    from dna_utils.run_identity import PHASE3_CAMPAIGN_BINDING_NAME
    from dna_utils.scientific_recipe import digest
    receipt, _ = _read_json(path, sha256, "the stage-R receipt")
    stage_block = receipt.get("anchor_confirmation") or {}
    if receipt.get("campaign_kind") != M.ANCHOR_CAMPAIGN_KIND or stage_block.get("stage") != REFIT_STAGE:
        raise CellRefused(f"{path}: not a stage-R campaign receipt")
    snapshot_path = Path(path).parent / receipt["plan_snapshot_file"]
    snapshot = json.loads(Path(snapshot_path).read_bytes())
    if M._json_digest(snapshot) != receipt.get("plan_snapshot_sha256") \
            or (snapshot.get("plan") or {}).get("campaign_nonce") != receipt.get("campaign_nonce"):
        raise CellRefused(f"{snapshot_path}: not the stage-R receipt's plan snapshot")
    auth = (snapshot.get("plan") or {}).get("authorities") or {}
    request = auth.get("anchor_request") or {}
    approval = auth.get("anchor_approval") or {}
    if (auth.get("anchor_manifest") or {}).get("sha256") != manifest_sha256 \
            or (auth.get("anchor_freeze") or {}).get("sha256") != ANCHOR_F_RECORD_SHA256 \
            or request.get("stage") != REFIT_STAGE or request.get("mode") != mode \
            or (request.get("freeze") or {}).get("sha256") != ANCHOR_F_RECORD_SHA256:
        raise CellRefused(f"{path}: the stage-R campaign is not a {mode} of this generation and F record")
    live = M.audit_approval(approval.get("section"), f"stage-R-{mode}", manifest=manifest_sha256,
                            freeze=ANCHOR_F_RECORD_SHA256, request=M._json_digest(request))
    if live["line"] != approval.get("line"):
        raise CellRefused(f"{path}: the stage-R approval line the campaign recorded is not the ledger's")
    cells = []
    for cell_id, cell in sorted((receipt.get("cells") or {}).items()):
        record_path = Path(path).parent / cell["record"]
        record, record_sha = _read_json(record_path, cell["record_sha256"], f"stage-R record {cell_id}")
        binding = (snapshot["plan"].get("cell_bindings") or {}).get(cell_id) or {}
        completion = record.get("completion") or {}
        ds, n, seed = record.get("dataset"), record.get("N"), record.get("seed")
        if record.get("stage") != REFIT_STAGE or record.get("campaign") != cell.get("campaign") \
                or binding.get("anchor_arm") != "anchors" or binding.get("stage") != REFIT_STAGE \
                or (completion.get("official_test") or {}).get("status") != "withheld" \
                or bool(record.get("smoke")) != (mode == "smoke"):
            raise CellRefused(f"{cell_id}: not a completed stage-R {mode} cell with the test withheld")
        if ds not in freeze["datasets"] or n != freeze["datasets"][ds]["N"] or seed not in REFIT_SEEDS:
            raise CellRefused(f"{cell_id}: not an F coordinate")
        if digest(binding["scientific_recipe"]) != binding.get("expected_scientific_recipe_sha256") \
                or (record.get("anchor_confirmation") or {}).get("scientific_recipe_sha256") \
                != binding["expected_scientific_recipe_sha256"]:
            raise CellRefused(f"{cell_id}: the sealed refit recipe does not bind its record")
        run_dir = Path(record["run_dir"])
        evidence, _ = _read_json(run_dir / PHASE3_CAMPAIGN_BINDING_NAME,
                                 completion.get("phase3_campaign_evidence_sha256"),
                                 f"{cell_id} trainer evidence")
        sidecar, _ = _read_json(run_dir / f"{completion.get('final_checkpoint')}.runtime.json",
                                completion.get("checkpoint_runtime_sha256"), f"{cell_id} runtime sidecar")
        terminal = completion.get("final_checkpoint_epoch_zero_based")
        # the terminal epoch the request allows: N for the run, the smoke horizon's last epoch for a smoke
        expected_terminal = n if mode == "run" else int(request.get("epochs") or 0) - 1
        anchor = record.get("anchor_confirmation") or {}
        if evidence.get("cell_id") != cell_id or sidecar.get("checkpoint_epoch_zero_based") != terminal \
                or type(terminal) is not int or terminal != expected_terminal \
                or (sidecar.get("extra") or {}).get("phase3_campaign") != evidence \
                or evidence.get("scientific_recipe_sha256") != binding["expected_scientific_recipe_sha256"] \
                or not M._is_sha256(completion.get("final_checkpoint_sha256")) \
                or not M._is_sha256(anchor.get("config_pt_sha256")) \
                or anchor.get("campaign_evidence_sha256") != completion.get("phase3_campaign_evidence_sha256"):
            raise CellRefused(f"{cell_id}: the trainer evidence or terminal checkpoint witness disagrees")
        M.assert_official_test_withheld(run_dir)
        cells.append({"cell_id": cell_id, "dataset": ds, "N": n, "seed": seed,
                      "tag": record["tag"], "run_dir": str(run_dir),
                      "record": str(record_path), "record_sha256": record_sha,
                      "final_checkpoint": completion["final_checkpoint"],
                      "final_checkpoint_sha256": completion["final_checkpoint_sha256"],
                      "checkpoint_runtime_sha256": completion["checkpoint_runtime_sha256"],
                      "phase3_campaign_evidence_sha256": completion["phase3_campaign_evidence_sha256"],
                      "config_pt_sha256": anchor["config_pt_sha256"],
                      "scientific_recipe_sha256": binding["expected_scientific_recipe_sha256"],
                      "terminal_epoch": terminal})
    want = sorted((ds, freeze["datasets"][ds]["N"], s) for ds in M.ANCHOR_DATASETS for s in REFIT_SEEDS)
    got = sorted((c["dataset"], c["N"], c["seed"]) for c in cells)
    if (mode == "run" and got != want) or (mode == "smoke" and len(got) != 1):
        raise CellRefused(f"{path}: the stage-R receipt holds {got}, not the {mode}'s cells")
    return {"receipt": {"path": str(path), "sha256": sha256}, "snapshot": str(snapshot_path),
            "epochs": request.get("epochs"),
            "approval": {"section": live["section"], "scope": live["scope"], "line": live["line"]},
            "cells": cells}


def terminal_test_request(args, refit: dict, *, manifest: dict) -> dict:
    """The stage-T request: the generation, the F authority, the stage-R receipt and approval, and
    per cell the run directory, record, terminal checkpoint and runtime witness it will evaluate,
    the outputs it may create (each once), the chain, roots and GPU count."""
    gpus = [g for g in (args.gpus.split(",") if args.gpus else [str(args.gpu)])]
    streams = len({c["dataset"] for c in refit["cells"]})
    if len(gpus) != streams or len(set(gpus)) != len(gpus):
        raise CellRefused(f"stage T runs {streams} dataset stream(s) and needs exactly that many "
                          f"distinct GPUs, got {gpus}")
    return {"schema": T_REQUEST_SCHEMA, "version": M.ANCHOR_CONFIRM_VERSION, "stage": TEST_STAGE,
            "mode": "smoke" if args.smoke else "run",
            "manifest": manifest["sha256"], "manifest_path": str(manifest["path"]),
            "freeze": {"path": str(ANCHOR_F_RECORD), "sha256": ANCHOR_F_RECORD_SHA256,
                       "acceptance_section": ANCHOR_F_ACCEPTANCE_SECTION,
                       "acceptance_sha256": ANCHOR_F_ACCEPTANCE_SHA256},
            "refit_receipt": refit["receipt"], "refit_approval": refit["approval"],
            "refit_epochs": refit["epochs"],
            "namespace": str(args.namespace), "record_dir": str(M.ANCHOR_RECORD_DIR),
            "cells": sorted(refit["cells"], key=lambda c: c["cell_id"]),
            "outputs": list(M.OFFICIAL_TEST_OUTPUTS), "chain": list(T_CHAIN), "gpu_count": len(gpus)}


def _snapshot_inputs_now(keys) -> dict:
    """The digests verify_snapshot recomputes for a snapshot's `inputs` keys, by the same mapping."""
    out = {}
    for key in keys:
        ds, label = key.split("/", 1)
        spec = M.DATASETS[ds]
        path = {"feature_cache_meta": Path(spec["cache"]) / "meta.json",
                "foils_meta": Path(spec["foils"]) / "meta.json",
                "whiten_select": Path(M._whitening(spec, stage="select")),
                "whiten_refit": Path(M._whitening(spec, stage=REFIT_STAGE)),
                "qwen": Path(spec["qwen"])}[label]
        out[key] = M._sha(path) if path.is_file() else "#absent"
    return out


def terminal_test_boundary(r_snapshot: dict, gpus) -> dict:
    """The authority every T producer runs under (audit 747.2): the stage-R campaign's sources, inputs
    and admitted input seals -- recomputed now and required unchanged since stage R -- with THIS
    stage's environment and leased physical GPUs. It has exactly the keys the launcher's own
    verify_snapshot re-checks (sources, inputs, source authority, environment and GPUs, stats-only
    seals), so the unchanged verifier runs between every two producers."""
    if r_snapshot.get("qwen_root") != str(M.QWEN_ROOT):
        raise CellRefused("the stage-R campaign used another Qwen root")
    sources = {rel: (M._sha(M.REPO / rel) if (M.REPO / rel).is_file() else "#absent")
               for rel in (r_snapshot.get("sources") or {})}
    inputs = _snapshot_inputs_now(r_snapshot.get("inputs") or {})
    authority = M._source_authority_bundle(sources)
    before = (r_snapshot.get("source_authority") or {}).get("entries") or {}
    if sources != r_snapshot.get("sources") or inputs != r_snapshot.get("inputs") \
            or {k: (v.get("head_sha256"), v.get("worktree_sha256")) for k, v in authority["entries"].items()} \
            != {k: (v.get("head_sha256"), v.get("worktree_sha256")) for k, v in before.items()}:
        raise CellRefused("the sources or inputs changed between stage R and stage T")
    environment = M.environment_fingerprint(gpus)
    if environment.get("errors"):
        raise CellRefused(f"the stage-T environment has errors: {environment['errors']}")
    return {"qwen_root": str(M.QWEN_ROOT), "sources": sources, "inputs": inputs,
            "source_authority": authority, "source_authority_sha256": M._semantic_digest(authority),
            "environment": environment, "environment_sha256": M._semantic_digest(environment),
            "input_seals": r_snapshot.get("input_seals")}


def terminal_test_env(gpu_uuid: str) -> dict:
    """The T entry's and post-chain's environment: the same clean base the trainers get (the
    launcher's passthrough set and START-UP runtime variables, no PYTHONPATH, no recipe variable),
    the fixed geometry and one physical GPU by UUID."""
    from dna_utils.runtime_environment import RUNTIME_ENV_KEYS, caller_environment
    env = {k: v for k, v in os.environ.items() if k in M._ENV_PASSTHROUGH}
    startup = caller_environment()
    for key in RUNTIME_ENV_KEYS:
        if startup.get(key) is None:
            env.pop(key, None)
        else:
            env[key] = startup[key]
    env.pop("PYTHONPATH", None)
    for name in M._RECIPE_ENV:
        env.pop(name, None)
    env.update(GDNA_NUM_SEMANTIC_PARTS=str(M.SLOTS), CUDA_VISIBLE_DEVICES=str(gpu_uuid))
    return env


def check_cell_inputs(cell: dict) -> None:
    """The consumed stage-R cell's own inputs at a T boundary (audit 756): config.pt, the terminal
    checkpoint and its runtime witness are still their pinned bytes. Scoped to this cell; the dataset,
    source, environment and seal checks are the T snapshot's."""
    run_dir = Path(cell["run_dir"])
    for name, pin in (("config.pt", cell["config_pt_sha256"]),
                      (cell["final_checkpoint"], cell["final_checkpoint_sha256"]),
                      (f"{cell['final_checkpoint']}.runtime.json", cell["checkpoint_runtime_sha256"])):
        if M._sha(run_dir / name) != pin:
            raise CellRefused(f"{cell['cell_id']}: {name} is not the stage-R cell's pinned bytes; no further "
                              "producer starts")


def cell_input_env(env: dict, cell: dict) -> dict:
    """The producers' environment plus the consumed cell's pins and admitted terminal epoch, which a
    producer that loads the configuration or the model verifies itself before deserializing and binds
    through epoch resolution (scripts/extract_train_split.py; audits 756, 759-760)."""
    return dict(env, GDNA_T_EXPECT_CONFIG_SHA256=cell["config_pt_sha256"],
                GDNA_T_EXPECT_CHECKPOINT_SHA256=cell["final_checkpoint_sha256"],
                GDNA_T_EXPECT_RUNTIME_SHA256=cell["checkpoint_runtime_sha256"],
                GDNA_T_EXPECT_TERMINAL_EPOCH=str(cell["terminal_epoch"]))


def attempt_path(namespace: str, cell: dict) -> Path:
    return M.RECORD_DIR / f"{namespace}_attempt_{cell['tag']}.json"


def reserve_attempt(namespace: str, *, cell: dict, request: dict, approval: dict,
                    campaign_nonce: str) -> tuple:
    """The durable, exclusive attempt reservation of one T cell, created BEFORE any model or test
    construction (audit 744.2 item 2). It is never removed: a failed or interrupted attempt keeps
    it, and a second attempt -- concurrent, duplicate or retry -- refuses, even if no metric or NPZ
    exists. Recovering an attempted cell needs a new authorization."""
    path = attempt_path(namespace, cell)
    payload = {"schema": T_ATTEMPT_SCHEMA, "namespace": namespace, "campaign_nonce": campaign_nonce,
               "request": request, "request_sha256": M._json_digest(request), "approval": approval,
               "cell": cell, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "owner_pid": os.getpid(), "owner_host": socket.gethostname(), "owner_boot_id": M._boot_id()}
    M.RECORD_DIR.mkdir(parents=True, exist_ok=True)
    try:
        M._publish_json_exclusive(path, payload)
    except CellRefused:
        raise CellRefused(f"{cell['cell_id']}: a stage-T attempt is already reserved at {path.name}; "
                          "an attempted cell is never retried without a new authorization") from None
    return path, M._sha(path)


def run_terminal_test_cell(cell: dict, *, namespace: str, request: dict, approval: dict,
                           campaign_nonce: str, gpu_uuid: str, snapshot: dict) -> dict:
    """One T cell, fail-stop: the T snapshot re-checked (sources, inputs, environment and GPUs,
    stats-only seals), the stage-R outputs re-checked, the attempt reserved, the T entry run, the
    snapshot re-checked at the entry-to-train-extraction transition, the unchanged post-chain run with
    the snapshot re-checked after every producer, every output checked with the legacy refit checker,
    the record published. A producer never starts after a failed check."""
    run_dir = Path(cell["run_dir"])
    M.verify_snapshot(snapshot)
    check_cell_inputs(cell)
    M.assert_official_test_withheld(run_dir)
    attempt, attempt_sha = reserve_attempt(namespace, cell=cell, request=request, approval=approval,
                                           campaign_nonce=campaign_nonce)
    env = cell_input_env(terminal_test_env(gpu_uuid), cell)
    started = time.time()
    proc = M._run_managed_process(
        [os.path.realpath(sys.executable), "-B", str(REPO / "scripts" / "anchor_terminal_test.py"),
         "--config_path", str(run_dir), "--attempt", str(attempt), "--attempt-sha256", attempt_sha],
        cwd=str(REPO), env=env)
    if proc.returncode != 0:
        raise CellRefused(f"{cell['cell_id']}: the T entry exited {proc.returncode}; the attempt "
                          f"{attempt.name} stays reserved")
    M.verify_snapshot(snapshot)
    check_cell_inputs(cell)                              # the entry-to-train-extraction transition
    M._run_refit_postprocess(run_dir, dataset=cell["dataset"], env=env, snapshot=snapshot,
                             boundary_check=lambda: check_cell_inputs(cell))
    completion = M.assert_refit_outputs(run_dir, dataset=cell["dataset"])
    if M._sha(run_dir / cell["final_checkpoint"]) != cell["final_checkpoint_sha256"]:
        raise CellRefused(f"{cell['cell_id']}: the terminal checkpoint changed during stage T")
    record = {"schema": T_RECORD_SCHEMA, "stage": TEST_STAGE, "namespace": namespace,
              "campaign_nonce": campaign_nonce, "request_sha256": M._json_digest(request),
              "cell": cell, "attempt": {"file": attempt.name, "sha256": attempt_sha},
              "completion": completion, "wall_seconds": round(time.time() - started, 1)}
    out = M.RECORD_DIR / f"{namespace}_{cell['tag']}.json"
    M._publish_json_exclusive(out, record)
    return {"record": out.name, "record_sha256": M._sha(out), "attempt_sha256": attempt_sha,
            "map_at_R": completion.get("map_at_R"), "bio_map_at_R": completion.get("bio_map_at_R")}


def _test_main(args, manifest, execute: bool) -> int:
    if not args.anchor_refit_receipt or not args.anchor_refit_receipt_sha256:
        raise CellRefused("--anchor-confirm test needs --anchor-refit-receipt and its digest")
    freeze = anchor_freeze_authority()
    _check_freeze_against_incumbent(freeze, M.anchor_incumbent())
    mode = "smoke" if args.smoke else "run"
    refit = admit_refit_receipt(args.anchor_refit_receipt, args.anchor_refit_receipt_sha256,
                                manifest_sha256=manifest["sha256"], freeze=freeze, mode=mode)
    request = terminal_test_request(args, refit, manifest=manifest)
    if not execute:
        print(f"{M.ANCHOR_CONFIRM_VERSION}: stage T, {len(request['cells'])} cells, namespace "
              f"{args.namespace}  [NON-EXECUTABLE until audit approval]")
        print("execution request that this command without --plan would need approved:")
        print(json.dumps(request, indent=1, sort_keys=True))
        print(f"execution request sha256 {M._json_digest(request)}")
        return 0
    if args.only:
        raise CellRefused("stage T takes the receipt's cells; --only does not apply")
    approval = M.audit_approval(args.anchor_approval_section, f"stage-T-{mode}",
                                manifest=manifest["sha256"], freeze=ANCHOR_F_RECORD_SHA256,
                                request=M._json_digest(request))
    M.recheck_generation(manifest, "after stage-T admission")
    M.RECORD_DIR = M.ANCHOR_RECORD_DIR
    return M._with_campaign_gpu_leases(
        args, lambda: _run_terminal_test(args, request=request, approval=approval, manifest=manifest))


def _run_terminal_test(args, *, request: dict, approval: dict, manifest: dict) -> int:
    import secrets
    gpus = [int(g) for g in (args.gpus.split(",") if args.gpus else [str(args.gpu)])]
    nonce = secrets.token_hex(32)
    receipt = json.loads(M._read_bytes(request["refit_receipt"]["path"], request["refit_receipt"]["sha256"],
                                       "the stage-R receipt"))
    r_snapshot = json.loads((Path(request["refit_receipt"]["path"]).parent
                             / receipt["plan_snapshot_file"]).read_bytes())
    if M._json_digest(r_snapshot) != receipt["plan_snapshot_sha256"]:
        print("[anchor-T] REFUSED: the stage-R snapshot is not its receipt's", file=sys.stderr)
        return 2
    try:
        boundary = terminal_test_boundary(r_snapshot, gpus)
    except CellRefused as error:
        print(f"[anchor-T] REFUSED: {error}", file=sys.stderr)
        return 2
    snapshot = {"schema": T_SNAPSHOT_SCHEMA,
                "plan": {"namespace": args.namespace, "campaign_nonce": nonce,
                         "campaign_kind": T_CAMPAIGN_KIND, "result_root": r_snapshot["plan"].get("result_root"),
                         "declared_count": len(request["cells"]),
                         "executed_count": len(request["cells"]),
                         "declared_cells": r_snapshot["plan"].get("declared_cells")},
                "request": request, "request_sha256": M._json_digest(request), "approval": approval,
                **boundary}
    try:
        M._assert_snapshot_gpu_leases(snapshot, args)
    except CellRefused as error:
        print(f"[anchor-T] REFUSED: {error}", file=sys.stderr)
        return 2
    snap_digest = M._json_digest(snapshot)
    snap_path = M.RECORD_DIR / f"{args.namespace}_snapshot_{snap_digest[:16]}.json"
    try:
        reservation = M.reserve_sweep_namespace(args.namespace, snapshot=snapshot, plan_digest=snap_digest,
                                                campaign_nonce=nonce, snapshot_file=snap_path.name)
        M._publish_json_exclusive(snap_path, snapshot)
    except CellRefused as error:
        print(f"[anchor-T] REFUSED: {error}", file=sys.stderr)
        return 2
    by_dataset = {}
    for cell in request["cells"]:
        by_dataset.setdefault(cell["dataset"], []).append(cell)
    results, lock = {}, threading.Lock()

    def _stream(cells, gpu_uuid):
        for cell in sorted(cells, key=lambda c: c["seed"]):
            try:
                M.assert_reservation_owner(args.namespace, reservation)
                M.recheck_generation(manifest, f"before stage T of {cell['cell_id']}")
                with lock:
                    unfinished = len(request["cells"]) - sum(1 for st, _ in results.values() if st == "ok")
                refusal = M.anchor_dispatch_space_refusal(str(Path(cell["run_dir"]).parent), unfinished)
                if refusal:
                    raise CellRefused(f"before {cell['cell_id']}: {refusal}")
                done = run_terminal_test_cell(cell, namespace=args.namespace, request=request,
                                              approval=approval, campaign_nonce=nonce, gpu_uuid=gpu_uuid,
                                              snapshot=snapshot)
                M.recheck_generation(manifest, f"after stage T of {cell['cell_id']}")
            except Exception as error:              # noqa: BLE001 -- the stream stops, fail-stop
                with lock:
                    results[cell["cell_id"]] = ("failed", str(error))
                return
            with lock:
                results[cell["cell_id"]] = ("ok", done)

    uuids = [str(row["uuid"]) for row in snapshot["environment"].get("selected_gpus") or []]
    if len(uuids) != len(by_dataset):
        print(f"[anchor-T] REFUSED: {len(uuids)} leased GPUs for {len(by_dataset)} dataset streams",
              file=sys.stderr)
        return 2
    threads = [threading.Thread(target=_stream, args=(cells, uuids[i]), daemon=False)
               for i, (ds, cells) in enumerate(sorted(by_dataset.items()))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for key in sorted(results):
        status, payload = results[key]
        print(f"[anchor-T] {key} {status}: {payload}", file=sys.stderr if status != "ok" else sys.stdout)
    done = sorted(k for k, (st, _) in results.items() if st == "ok")
    if len(done) != len(request["cells"]):
        print(f"[anchor-T] {len(done)} of {len(request['cells'])} stage-T cells complete; no receipt",
              file=sys.stderr)
        return 1
    M.recheck_generation(manifest, "before the stage-T receipt")
    M.assert_reservation_owner(args.namespace, reservation)
    receipt = {"schema": T_RECORD_SCHEMA, "campaign_kind": T_CAMPAIGN_KIND, "stage": TEST_STAGE,
               "namespace": args.namespace, "campaign_nonce": nonce,
               "plan_snapshot_file": snap_path.name, "plan_snapshot_sha256": snap_digest,
               "campaign_reservation_file": M.campaign_reservation_path(args.namespace).name,
               "campaign_reservation_sha256": M._sha(M.campaign_reservation_path(args.namespace)),
               "request_sha256": M._json_digest(request), "refit_receipt": request["refit_receipt"],
               "expected_cells": len(request["cells"]), "cell_count": len(done),
               "cells": {k: results[k][1] for k in done}}
    M._publish_json_exclusive(M.RECORD_DIR / f"{args.namespace}{T_RECEIPT_SUFFIX}", receipt)
    print(f"[anchor-T] {len(done)} of {len(done)} stage-T cells complete")
    return 0


def main(args) -> int:
    """`--anchor-confirm refit|test`: the admission runs first in every mode; --plan prints the
    request; --smoke/--run need the generation manifest and the audit's approval line."""
    for flag, value in (("--refit", args.refit), ("--recipe", args.recipe),
                        ("--stability-plan", args.stability_plan), ("--at-topp", args.at_topp),
                        ("--sweep", args.sweep), ("--anchor-selection", args.anchor_selection)):
        if value:
            print(f"[phase3] REFUSED: {flag} cannot be combined with --anchor-confirm "
                  f"{args.anchor_confirm}", file=sys.stderr)
            return 2
    if not re.fullmatch(r"anc[A-Za-z0-9]+", str(args.namespace)):
        print("[phase3] REFUSED: an anchor namespace must match anc[A-Za-z0-9]+", file=sys.stderr)
        return 2
    execute = bool(args.run or args.smoke) and not args.plan
    try:
        if bool(args.anchor_manifest) != bool(args.anchor_manifest_sha256):
            raise CellRefused("--anchor-manifest and --anchor-manifest-sha256 go together")
        if not args.anchor_manifest:
            raise CellRefused(f"stage {args.anchor_confirm} needs --anchor-manifest and its digest "
                              "(the reviewed generation this tree must be)")
        manifest = M.load_anchor_manifest(args.anchor_manifest, args.anchor_manifest_sha256)
        if manifest["sha256"] in (M.ANCHOR_V7_MANIFEST_SHA256, ANCHOR_V8_MANIFEST_SHA256):
            raise CellRefused("stages R and T run in generation v9, never under the v7 or v8 manifest")
        if args.anchor_confirm == REFIT_STAGE:
            if args.anchor_refit_receipt or args.anchor_refit_receipt_sha256:
                raise CellRefused("--anchor-refit-receipt belongs to --anchor-confirm test")
            return _refit_main(args, manifest, execute)
        return _test_main(args, manifest, execute)
    except CellRefused as error:
        print(f"[phase3] REFUSED: {error}", file=sys.stderr)
        return 2
