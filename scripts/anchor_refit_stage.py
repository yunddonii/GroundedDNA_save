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

def admit_refit_receipt(path, sha256, *, manifest_sha256: str, freeze: dict, mode: str,
                        carried_cells=frozenset()) -> dict:
    """The completed stage-R campaign stage T evaluates, from JSON metadata only (no checkpoint byte
    is read here; the T entry hashes each checkpoint before it loads it): the receipt at its digest;
    its snapshot (semantic digest), generation manifest, F authority and stage-R approval, re-verified
    in the ledger now; and per cell the record, its campaign binding and completion pins, the trainer
    evidence and runtime sidecar at their pinned digests (terminal epoch), the sealed recipe, and a
    run directory that still holds no official-test output. `mode` run: exactly the twelve F cells;
    smoke: one. `carried_cells` (the r8 recovery only, audit 797.2 item 2): those cells already hold
    their stage-T outputs, which must be exactly the required twelve with bit2 absent; every other
    cell is checked as before."""
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
        if cell_id in carried_cells:
            assert_carried_outputs(run_dir)
        else:
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
                           campaign_nonce: str, gpu_uuid: str, snapshot: dict, before_attempt=None) -> dict:
    """One T cell, fail-stop: the T snapshot re-checked (sources, inputs, environment and GPUs,
    stats-only seals), the stage-R outputs re-checked, the attempt reserved, the T entry run, the
    snapshot re-checked at the entry-to-train-extraction transition, the unchanged post-chain run with
    the snapshot re-checked after every producer, every output checked with the legacy refit checker,
    the record published. A producer never starts after a failed check. `before_attempt` (the r8
    recovery only) consumes the lineage's one recovery claim between those checks and the attempt."""
    run_dir = Path(cell["run_dir"])
    M.verify_snapshot(snapshot)
    check_cell_inputs(cell)
    M.assert_official_test_withheld(run_dir)
    if before_attempt is not None:
        before_attempt()
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


# --------------------------------------------------------------------------------------- recovery
# Generation v9 r8 (audits 795-797; docs/ANCHOR_T_RECOVERY_CONTRACT_v1.md): the ONE recovery of the
# stage-T cell the r7 campaign `ancT9` left unfinished at its budget stop (NUS-WIDE seed 44,
# 2026-10-06T20:50:07Z). It is not a resume flag or a generic subset: the stopped campaign is pinned
# member by member in the lineage file below, the eleven completed cells are CARRIED by their records
# (never evaluated again), and only the pinned cell runs, once, under a claim that no second namespace,
# record root or worktree can take again. The historical authority (F, the stage-R receipt and both
# approvals) is verified against generation r7; the executing sources are r8's, and the sources that
# differ from stage R are exactly the reviewed control-plane allowlist.
RECOVERY_STAGE = "recover"
RECOVERY_MODE = "recovery"
RECOVERY_SCOPE = "stage-T-recovery"
#: the pinned lineage of the stopped campaign (built from JSON and the settled ledger only)
RECOVERY_LINEAGE = REPO / "artifacts" / "anchor_confirmation" / "anchor_t_recovery_lineage_v1.json"
RECOVERY_LINEAGE_SHA256 = "b767601961c489c469c8822988dcb66b9dc525f4539ef428456a2f8545ce8bda"
RECOVERY_LINEAGE_SCHEMA = "anchor-t-recovery-lineage/1"
T_RECOVERY_REQUEST_SCHEMA = "anchor-terminal-test-recovery-request/2"     # r9: schema 1 (r8) is spent
T_RECOVERY_SNAPSHOT_SCHEMA = "anchor-terminal-test-recovery-snapshot/1"
T_RECOVERY_RECEIPT_SCHEMA = "anchor-terminal-test-recovery-receipt/1"
T_RECOVERY_CLAIM_SCHEMA = "anchor-terminal-test-recovery-claim/1"
T_RECOVERY_RECEIPT_SUFFIX = "_recovery_complete.json"
#: one persistent claim per interrupted lineage, outside every worktree, record root and namespace
#: (audit 797.2 item 3): its name is the lineage key alone
RECOVERY_CLAIM_ROOT = Path("/home/yschoi/gdna_anchorRT_recovery_claims")
# Generation v9 r9 (audits 839-841; the exception section of docs/ANCHOR_T_RECOVERY_CONTRACT_v1.md): the r8
# recovery's one attempt (namespace ancT9r) failed before its T entry, at the managed-child main-thread
# guard, after consuming the lineage claim. r9 runs the cell in a worker thread and admits ONE more recovery
# only as an explicit exception that binds the failed attempt's evidence (pinned below, from JSON and the
# ledger only) next to the stopped campaign; its claim key is derived from that failed evidence alone.
RECOVERY_EXCEPTION = REPO / "artifacts" / "anchor_confirmation" / "anchor_t_recovery_exception_v1.json"
RECOVERY_EXCEPTION_SHA256 = "25320aa63ebcb55b0a22c72432efffa543df66b6e288074b03849090c4f26d70"
RECOVERY_EXCEPTION_SCHEMA = "anchor-t-recovery-exception/1"
#: generation r7, under which stage R and the stopped campaign ran (historical authority only)
R7_MANIFEST_SHA256 = "2f24fc80bb83e2c73b17ec473c2731ade6d34a356e02414e6719fe9e9bec947c"
#: the exact stage-R snapshot sources r8 changes (control plane only; audit 797.2 item 1)
RECOVERY_CHANGED_SOURCES = ("scripts/anchor_refit_stage.py", "scripts/anchor_terminal_test.py",
                            "scripts/phase3_selection_matrix.py")
#: the exact generation-closure members r8 changes and adds relative to the r7 manifest
RECOVERY_CHANGED_CLOSURE = RECOVERY_CHANGED_SOURCES + (
    "scripts/anchor_confirm_manifest.py", "scripts/anchor_confirm_supervisor.py",
    "tests/test_anchor_confirm_launcher.py", "tests/test_anchor_refit_stage.py")
RECOVERY_ADDED_CLOSURE = ("docs/ANCHOR_T_RECOVERY_CONTRACT_v1.md", "tests/test_anchor_t_recovery.py")
#: the T record's completion digests, by the output each one binds
CARRIED_BINDINGS = (("evaluation_sha256", "evaluation_siglip2_base.json"),
                    ("bio_evaluation_sha256", "evaluation_siglip2_base_bioproj.json"),
                    ("cell_result_sha256", "cell_result.json"),
                    ("analysis_complete_sha256", "analysis_complete.json"),
                    ("extraction_complete_sha256", "extraction_complete.json"),
                    ("pairwise_nmi_sha256", "pairwise_nmi.json"))
BIT2_OUTPUT = "evaluation_siglip2_bit2.json"


def assert_carried_outputs(run_dir: Path) -> list:
    """A carried cell's run directory (stat only): the twelve required stage-T outputs exist and the
    bit2 evaluation does not (audit 776)."""
    required = [name for name in M.OFFICIAL_TEST_OUTPUTS if name != BIT2_OUTPUT]
    missing = [name for name in required if not (Path(run_dir) / name).is_file()]
    if missing or (Path(run_dir) / BIT2_OUTPUT).exists():
        raise CellRefused(f"{run_dir}: a carried stage-T cell must hold exactly the twelve required "
                          f"outputs with bit2 absent; missing {missing}")
    return required


def recovery_lineage(read=None) -> dict:
    """The pinned lineage file at its digest (JSON only)."""
    lineage, _ = _read_json(RECOVERY_LINEAGE, RECOVERY_LINEAGE_SHA256, "the recovery lineage", read)
    if lineage.get("schema") != RECOVERY_LINEAGE_SCHEMA:
        raise CellRefused(f"{RECOVERY_LINEAGE}: not a {RECOVERY_LINEAGE_SCHEMA} file")
    return lineage


def recovery_claim_key(lineage: dict) -> str:
    """The key of the interrupted lineage alone -- no namespace, root or caller in it."""
    stopped, cell = lineage["stopped"], lineage["recovery_cell"]
    return M._json_digest({"run_id": stopped["settlement"]["run_id"],
                           "stopped_request_sha256": stopped["request_sha256"],
                           "cell_id": cell["cell_id"], "attempt_sha256": cell["attempt"]["sha256"],
                           "entry_sha256": cell["entry"]["sha256"]})


def recovery_claim_path(key: str) -> Path:
    return RECOVERY_CLAIM_ROOT / f"{key}.json"


def recovery_exception(read=None) -> dict:
    """The pinned exception artifact (JSON only) of the failed r8 recovery."""
    exception, _ = _read_json(RECOVERY_EXCEPTION, RECOVERY_EXCEPTION_SHA256, "the recovery exception", read)
    if exception.get("schema") != RECOVERY_EXCEPTION_SCHEMA:
        raise CellRefused(f"{RECOVERY_EXCEPTION}: not a {RECOVERY_EXCEPTION_SCHEMA} file")
    return exception


def exception_claim_key(failed: dict, cell_id: str) -> str:
    """The ONE exception's claim key: the failed attempt's evidence alone -- no namespace, root, generation
    or caller -- so every alternate route to a second recovery derives the same key (audit 841.2)."""
    return M._json_digest({"failed_run_id": failed["run_id"], "failed_claim_sha256": failed["claim"]["sha256"],
                           "failed_attempt_sha256": failed["attempt"]["sha256"], "cell_id": cell_id})


def _observed_absent(path: Path, what: str) -> None:
    """Checked absence by lstat (audit 841.3): a link counts as present even when dangling, and an
    observation error refuses instead of reading as absence."""
    try:
        os.lstat(path)
    except FileNotFoundError:
        return
    except OSError as error:
        raise CellRefused(f"{what}: {path} cannot be observed ({type(error).__name__}); absence is not assumed") \
            from None
    raise CellRefused(f"{what}: {path} exists (a link counts, even a dangling one)")


def verify_failed_recovery(exception: dict, lineage: dict) -> dict:
    """The failed r8 recovery, exactly (audit 841): its consumed claim, attempt, reservation and snapshot at
    their pins and bound to one another, the lineage's interrupted cell and the failed request; its two
    supervisor rows as the unchanged prefix of the append-only recovery ledger (exited, rc 1, no attempts,
    the known charge), with no later row of that run; and the lstat-checked absence of its entry, record and
    receipt. Run at admission, immediately before the exception claim, in the T entry and before
    publication. Returns the request's `exception_of` block."""
    rec = lineage["recovery_cell"]
    cell_id = rec["cell_id"]
    if exception.get("cell_id") != cell_id or exception.get("namespace") == lineage["stopped"]["namespace"]:
        raise CellRefused("the exception does not name the stopped campaign's interrupted cell under its own namespace")
    consumed = recovery_claim_key(lineage)
    claim_pin = exception["claim"]
    if claim_pin.get("key") != consumed or Path(claim_pin["path"]) != recovery_claim_path(consumed):
        raise CellRefused("the exception does not name the lineage's consumed claim in the claim root")
    claim, _ = _read_json(claim_pin["path"], claim_pin["sha256"], "the consumed recovery claim")
    attempt, _ = _read_json(exception["attempt"]["path"], exception["attempt"]["sha256"],
                            "the failed recovery's attempt")
    reservation, _ = _read_json(exception["reservation"]["path"], exception["reservation"]["sha256"],
                                "the failed recovery's reservation")
    snapshot, _ = _read_json(exception["snapshot"]["path"], exception["snapshot"]["sha256"],
                             "the failed recovery's snapshot")
    rdir, ns, request_sha = Path(exception["record_dir"]), exception["namespace"], exception["request_sha256"]
    approval = exception["approval"]
    attempt_name = Path(exception["attempt"]["path"]).name
    if any(Path(exception[k]["path"]).parent != rdir for k in ("attempt", "reservation", "snapshot")) \
            or attempt_name != f"{ns}_attempt_{rec['tag']}.json" \
            or claim.get("claim_key") != consumed or claim.get("request_sha256") != request_sha \
            or claim.get("namespace") != ns or claim.get("attempt") != attempt_name \
            or claim.get("approval") != approval or claim.get("record_dir") != str(rdir) \
            or attempt.get("request_sha256") != request_sha or M._json_digest(attempt.get("request")) != request_sha \
            or attempt.get("namespace") != ns or attempt.get("campaign_nonce") != exception["campaign_nonce"] \
            or (attempt.get("cell") or {}).get("cell_id") != cell_id \
            or {k: (attempt.get("approval") or {}).get(k) for k in ("section", "scope", "line")} != approval \
            or approval.get("scope") != RECOVERY_SCOPE \
            or M._json_digest(snapshot) != exception["snapshot"]["semantic_sha256"] \
            or (snapshot.get("plan") or {}).get("campaign_nonce") != exception["campaign_nonce"] \
            or reservation.get("plan_digest") != exception["snapshot"]["semantic_sha256"] \
            or reservation.get("namespace") != ns or reservation.get("campaign_nonce") != exception["campaign_nonce"]:
        raise CellRefused("the failed recovery's claim, attempt, reservation and snapshot are not one failed attempt")
    ledger = exception["ledger"]
    try:
        raw = Path(ledger["path"]).read_bytes()
    except OSError as error:
        raise CellRefused(f"the recovery ledger cannot be read ({type(error).__name__})") from None
    lines = raw.split(b"\n")
    head, rows_n = lines[:ledger["rows"]], ledger["rows"]
    if len(lines) <= rows_n or lines[-1] != b"" \
            or [hashlib.sha256(line).hexdigest() for line in head] != ledger["line_sha256"] \
            or hashlib.sha256(b"".join(line + b"\n" for line in head)).hexdigest() != ledger["prefix_sha256"]:
        raise CellRefused("the recovery ledger does not begin with the failed run's pinned rows")
    try:
        rows = [json.loads(line) for line in lines[:-1]]
    except ValueError:
        raise CellRefused("the recovery ledger is not JSON lines") from None
    start, final = rows[0], rows[rows_n - 1]
    if start.get("event") != "start" or start.get("run_id") != exception["run_id"] \
            or final.get("event") != "final" or final.get("run_id") != exception["run_id"] \
            or final.get("status") != "exited" or final.get("returncode") != 1 or final.get("attempts") != [] \
            or final.get("charged_seconds") != ledger["charged_seconds"] \
            or (start.get("parent") or {}).get("sha256") != ledger["parent_sha256"] \
            or any(row.get("run_id") == exception["run_id"] for row in rows[rows_n:]):
        raise CellRefused("the failed run's settlement is not the audited pre-entry failure")
    for name in exception["absent"]:
        _observed_absent(rdir / name, "the failed recovery")
    return {"namespace": ns, "run_id": exception["run_id"], "request_sha256": request_sha, "approval": approval,
            "window_section": exception["window_section"], "record_dir": str(rdir),
            "claim": {"path": claim_pin["path"], "sha256": claim_pin["sha256"]},
            "attempt": dict(exception["attempt"]), "reservation": dict(exception["reservation"]),
            "snapshot": {"path": exception["snapshot"]["path"], "sha256": exception["snapshot"]["sha256"]},
            "ledger": {"path": ledger["path"], "prefix_sha256": ledger["prefix_sha256"],
                       "charged_seconds": ledger["charged_seconds"]},
            "exception": {"path": str(RECOVERY_EXCEPTION), "sha256": RECOVERY_EXCEPTION_SHA256}}


def historical_generation(lineage: dict) -> dict:
    """The r7 manifest's closure pins, read from its pinned bytes. It is never loaded as this tree's
    generation: it is the authority the historical approvals name."""
    hist = lineage["historical_manifest"]
    if hist.get("sha256") != R7_MANIFEST_SHA256:
        raise CellRefused("the recovery's historical generation is not r7")
    manifest = json.loads(M._read_bytes(hist["path"], hist["sha256"], "the r7 generation manifest"))
    files = (manifest.get("new_generation") or {}).get("files_sha256") or {}
    if not files:
        raise CellRefused("the r7 generation manifest lists no files")
    return {"path": hist["path"], "sha256": hist["sha256"], "files_sha256": files}


def generation_transition(manifest: dict, historical: dict) -> dict:
    """The r7 -> r8 closure transition is exactly the reviewed one: these members changed, these were
    added, none was removed, and every other member is its r7 bytes."""
    if manifest["sha256"] == R7_MANIFEST_SHA256:
        raise CellRefused("the recovery executes under r8, never under the r7 generation it recovers")
    old, new = historical["files_sha256"], manifest["files_sha256"]
    changed = sorted(rel for rel in set(old) & set(new) if old[rel] != new[rel])
    added, removed = sorted(set(new) - set(old)), sorted(set(old) - set(new))
    if changed != sorted(RECOVERY_CHANGED_CLOSURE) or added != sorted(RECOVERY_ADDED_CLOSURE) or removed:
        raise CellRefused(f"the r7 -> r8 transition is not the reviewed one: changed {changed}, added "
                          f"{added}, removed {removed}")
    return {"from": historical["sha256"], "to": manifest["sha256"], "changed": changed, "added": added}


def verify_settlement(settlement: dict) -> dict:
    """The stopped campaign's settlement, exactly (audit 797.2 items 2 and 4): the whole settled R/T
    ledger at its digest and line count; this run's start, stop and final lines at their digests and in
    that order; the final the ledger's last line; a budget stop with a clean termination (status
    stopped, no held lease, orphan, continuity loss or monitor failure) and the known charges."""
    raw = M._read_bytes(settlement["ledger"], settlement["ledger_sha256"], "the settled R/T ledger")
    lines = raw.decode("utf-8").splitlines()
    if len(lines) != settlement["ledger_lines"]:
        raise CellRefused("the settled R/T ledger does not have its pinned line count")
    rows = [json.loads(line) for line in lines]
    events = [(i, row["event"]) for i, row in enumerate(rows)
              if row.get("run_id") == settlement["run_id"] and row.get("event") in ("start", "stop", "final")]
    if [kind for _, kind in events] != ["start", "stop", "final"] or events[-1][0] != len(rows) - 1:
        raise CellRefused("the stopped run is not exactly one start, stop and final, ending the ledger")
    for i, kind in events:
        if hashlib.sha256(lines[i].encode("utf-8")).hexdigest() != settlement[f"{kind}_line_sha256"]:
            raise CellRefused(f"the stopped run's {kind} line is not its pinned bytes")
    stop, final = rows[events[1][0]], rows[events[2][0]]
    reason = settlement["stop_reason"]
    if not str(reason).startswith("budget: ") or stop.get("reason") != reason \
            or final.get("reason") != reason or settlement["final_status"] != "stopped" \
            or final.get("status") != "stopped" \
            or final.get("returncode") != settlement["final_returncode"] \
            or final.get("leases_held_after_exit") != [] or final.get("orphaned_live_attempts") != [] \
            or final.get("continuity_lost") != [] or final.get("monitor_failures") != [] \
            or final.get("charged_seconds") != settlement["charged_seconds"] \
            or final.get("cumulative_charged_seconds") != settlement["cumulative_charged_seconds"]:
        raise CellRefused("the stopped run's settlement is not the audited clean budget stop")
    return {"ledger": settlement["ledger"], "ledger_sha256": settlement["ledger_sha256"],
            "run_id": settlement["run_id"], "final_line_sha256": settlement["final_line_sha256"],
            "cumulative_charged_seconds": settlement["cumulative_charged_seconds"]}


def _pinned(record_dir: Path, pin: dict, what: str) -> dict:
    return _read_json(Path(record_dir) / pin["file"], pin["sha256"], what)[0]


def admit_stopped_campaign(lineage: dict, *, freeze: dict) -> dict:
    """Before execution: the immutable lineage (verify_stopped_lineage) AND the recovered cell's run
    directory still holding no official-test output (the pre-execution target absence)."""
    admitted = verify_stopped_lineage(lineage, freeze=freeze, target_executed=False)
    M.assert_official_test_withheld(Path(admitted["cell"]["run_dir"]))
    return admitted


def verify_stopped_lineage(lineage: dict, *, freeze: dict, target_executed: bool) -> dict:
    """The stopped campaign's immutable lineage, from JSON, the ledger and stat only (audit 797.2 items
    1-2): the stage-R receipt under r7 with the eleven carried cells holding exactly their twelve outputs
    (and, once `target_executed`, the recovered cell too; before, it must hold none); the stopped
    snapshot, request, reservation and approval (re-verified in the ledger, against r7); its settlement;
    exact membership (the carried set is the twelve minus the recovered cell); every carried record,
    attempt and entry at its pin and bound to the stopped request; the recovered cell's attempt and
    entry at their pins, no stage-T record and no attempt in another namespace of the historical record
    directory; the stopped campaign's receipt absent. Run before execution and again before the
    combined receipt is published (audit 800)."""
    rec_dir = Path(lineage["historical_record_dir"])
    stopped, recovery, carried = lineage["stopped"], lineage["recovery_cell"], lineage["carried"]
    snapshot = _pinned(rec_dir, stopped["snapshot"], "the stopped stage-T snapshot")
    request = snapshot.get("request") or {}
    if M._json_digest(snapshot) != stopped["snapshot"]["semantic_sha256"] \
            or snapshot.get("schema") != T_SNAPSHOT_SCHEMA \
            or M._json_digest(request) != stopped["request_sha256"] \
            or snapshot.get("request_sha256") != stopped["request_sha256"] \
            or request.get("schema") != T_REQUEST_SCHEMA or request.get("mode") != "run" \
            or request.get("namespace") != stopped["namespace"] \
            or request.get("manifest") != R7_MANIFEST_SHA256 \
            or (request.get("freeze") or {}).get("sha256") != ANCHOR_F_RECORD_SHA256 \
            or (snapshot.get("plan") or {}).get("campaign_nonce") != stopped["campaign_nonce"]:
        raise CellRefused("the stopped stage-T snapshot is not the pinned r7 run request")
    reservation = _pinned(rec_dir, stopped["reservation"], "the stopped campaign's reservation")
    if reservation.get("namespace") != stopped["namespace"] \
            or reservation.get("plan_digest") != stopped["snapshot"]["semantic_sha256"] \
            or reservation.get("plan_snapshot_file") != stopped["snapshot"]["file"] \
            or reservation.get("campaign_nonce") != stopped["campaign_nonce"]:
        raise CellRefused("the stopped campaign's reservation does not bind its snapshot")
    approval = stopped["approval"]
    live = M.audit_approval(approval["section"], "stage-T-run", manifest=R7_MANIFEST_SHA256,
                            freeze=ANCHOR_F_RECORD_SHA256, request=stopped["request_sha256"])
    if not (live["line"] == approval["line"] == (snapshot.get("approval") or {}).get("line")):
        raise CellRefused("the stopped campaign's stage-T approval is not the ledger's line")
    if (rec_dir / f"{stopped['namespace']}{T_RECEIPT_SUFFIX}").exists():
        raise CellRefused("the stopped campaign has a receipt: it is not a stopped campaign")
    settlement = verify_settlement(stopped["settlement"])
    cells = {c["cell_id"]: c for c in request.get("cells") or []}
    if len(cells) != len(request.get("cells") or []) or len(cells) != 12:
        raise CellRefused("the stopped request does not name twelve distinct cells")
    if recovery["cell_id"] in carried or set(carried) | {recovery["cell_id"]} != set(cells):
        raise CellRefused("the carried cells are not exactly the stopped request's twelve minus the "
                          "recovered one")
    receipt = lineage["refit_receipt"]
    holding = frozenset(carried) | (frozenset({recovery["cell_id"]}) if target_executed else frozenset())
    refit = admit_refit_receipt(rec_dir / receipt["file"], receipt["sha256"], manifest_sha256=R7_MANIFEST_SHA256,
                                freeze=freeze, mode="run", carried_cells=holding)
    if sorted(refit["cells"], key=lambda c: c["cell_id"]) != sorted(cells.values(), key=lambda c: c["cell_id"]) \
            or request.get("refit_receipt") != refit["receipt"]:
        raise CellRefused("the stage-R receipt no longer admits the stopped request's cells")
    nonce, ns = stopped["campaign_nonce"], stopped["namespace"]
    carried_out = []
    for cell_id in sorted(carried):
        pins, cell = carried[cell_id], cells[cell_id]
        if pins.get("tag") != cell["tag"]:
            raise CellRefused(f"{cell_id}: the carried pins name another cell tag")
        attempt = _pinned(rec_dir, pins["attempt"], f"{cell_id} stopped attempt")
        entry = _pinned(rec_dir, pins["entry"], f"{cell_id} stopped entry claim")
        record = _pinned(rec_dir, pins["record"], f"{cell_id} stage-T record")
        names = (f"{ns}_attempt_{cell['tag']}.json", f"{ns}_entry_{cell['tag']}.json", f"{ns}_{cell['tag']}.json")
        if (pins["attempt"]["file"], pins["entry"]["file"], pins["record"]["file"]) != names \
                or attempt.get("request_sha256") != stopped["request_sha256"] \
                or attempt.get("campaign_nonce") != nonce or attempt.get("cell") != cell \
                or entry.get("attempt_sha256") != pins["attempt"]["sha256"] or entry.get("cell_id") != cell_id \
                or record.get("schema") != T_RECORD_SCHEMA or record.get("namespace") != ns \
                or record.get("request_sha256") != stopped["request_sha256"] \
                or record.get("campaign_nonce") != nonce or record.get("cell") != cell \
                or record.get("attempt") != {"file": pins["attempt"]["file"], "sha256": pins["attempt"]["sha256"]}:
            raise CellRefused(f"{cell_id}: the carried record, attempt and entry are not one stopped-campaign "
                              "lineage")
        carried_out.append({"cell_id": cell_id, "run_dir": cell["run_dir"], "record": pins["record"],
                            "attempt": pins["attempt"], "entry": pins["entry"],
                            "completion": record.get("completion") or {}})
    cell = cells[recovery["cell_id"]]
    attempt = _pinned(rec_dir, recovery["attempt"], "the recovered cell's stopped attempt")
    entry = _pinned(rec_dir, recovery["entry"], "the recovered cell's stopped entry claim")
    if recovery.get("tag") != cell["tag"] \
            or (recovery["attempt"]["file"], recovery["entry"]["file"]) != (f"{ns}_attempt_{cell['tag']}.json",
                                                                           f"{ns}_entry_{cell['tag']}.json") \
            or attempt.get("request_sha256") != stopped["request_sha256"] or attempt.get("cell") != cell \
            or attempt.get("campaign_nonce") != nonce \
            or entry.get("attempt_sha256") != recovery["attempt"]["sha256"] \
            or entry.get("cell_id") != recovery["cell_id"]:
        raise CellRefused("the recovered cell's stopped attempt and entry are not its lineage")
    if (rec_dir / f"{ns}_{cell['tag']}.json").exists():
        raise CellRefused("the recovered cell has a stage-T record: it is not the interrupted cell")
    others = sorted(p.name for p in rec_dir.glob(f"*_attempt_{cell['tag']}.json")
                    if p.name != recovery["attempt"]["file"])
    if others:
        raise CellRefused(f"the recovered cell was attempted in another namespace: {others}")
    return {"request": request, "snapshot": snapshot, "settlement": settlement, "refit": refit,
            "cell": cell, "carried": carried_out}


def assert_record_outputs(run_dir: Path, completion: dict, cell_id: str) -> None:
    """A stage-T cell's twelve outputs are the bytes its record's completion binds (hashed, never
    deserialized) and bit2 is absent."""
    assert_carried_outputs(run_dir)
    want = {name: completion.get(key) for key, name in CARRIED_BINDINGS}
    for split in ("query", "db", "train"):
        want[f"extraction_manifest_{split}.json"] = (completion.get("extraction_manifest_sha256") or {}).get(split)
        want[f"extract_{split}.npz"] = (completion.get("npz_sha256") or {}).get(split)
    if sorted(want) != sorted(n for n in M.OFFICIAL_TEST_OUTPUTS if n != BIT2_OUTPUT):
        raise CellRefused(f"{cell_id}: the stage-T record does not bind all twelve outputs")
    wrong = sorted(name for name, pin in want.items() if not M._is_sha256(pin) or M._sha(Path(run_dir) / name) != pin)
    if wrong:
        raise CellRefused(f"{cell_id}: outputs are not the bytes its stage-T record binds: {wrong}")


def verify_carried_payloads(admitted: dict) -> None:
    """At the execution boundary only (after approval; audit 797.3), and again before the combined
    receipt (audit 800): every carried cell's twelve outputs are the bytes its stage-T record binds."""
    for carried in admitted["carried"]:
        try:
            assert_record_outputs(Path(carried["run_dir"]), carried["completion"], carried["cell_id"])
        except CellRefused as error:
            raise CellRefused(f"carried outputs are not the bytes its stage-T record binds: {error}") from None


def final_recovery_closure(request: dict, *, freeze: dict, claim: dict, executed: dict, approval: dict,
                           campaign_nonce: str) -> dict:
    """Audits 800-801: fail-closed, immediately before the combined receipt is published. The pinned
    lineage is re-verified with the recovered cell now executed (the eleven carried records, attempts
    and entries at their pins and bindings, their outputs present with bit2 absent and at their recorded
    bytes; the stopped snapshot, reservation, approval and settled ledger), the request's lineage part
    is still what that lineage determines, the claim is the bytes consumed, and the new cell's own chain
    is re-read from its files: the record at its digest, the attempt it names at that digest (its exact
    name in this record root, schema, namespace, nonce, request, approval and cell), the exclusive entry
    claim binding that attempt and cell, and the twelve outputs at the bytes the record binds. Metadata
    only: nothing is admitted, claimed or loaded again. The receipt binds what this returns."""
    lineage = recovery_lineage()
    admitted = verify_stopped_lineage(lineage, freeze=freeze, target_executed=True)
    failed = verify_failed_recovery(recovery_exception(), lineage)      # audit 841.3: publication recheck
    block = recovery_lineage_block(admitted, lineage, failed)
    if {k: request.get(k) for k in block} != block:
        raise CellRefused("the recovery request's lineage is no longer the pinned stopped campaign's and failed "
                          "attempt's")
    verify_carried_payloads(admitted)
    claim_path = recovery_claim_path(request["claim_key"])
    if not claim_path.is_file() or M._sha(claim_path) != claim.get("sha256"):
        raise CellRefused("the recovery claim is not the bytes this campaign consumed")
    cell = request["cells"][0]
    root = Path(request["record_dir"])
    if root.resolve() != Path(M.RECORD_DIR).resolve():
        raise CellRefused("the recovery wrote another record root than its request names")
    record_name = f"{request['namespace']}_{cell['tag']}.json"
    if executed["record"] != record_name:
        raise CellRefused("the recovered cell's record is not its contracted name")
    record, record_sha = _read_json(root / record_name, executed["record_sha256"],
                                    "the recovered cell's stage-T record")
    attempt_name = attempt_path(request["namespace"], cell).name
    if record.get("schema") != T_RECORD_SCHEMA or record.get("stage") != TEST_STAGE \
            or record.get("namespace") != request["namespace"] or record.get("campaign_nonce") != campaign_nonce \
            or record.get("request_sha256") != M._json_digest(request) or record.get("cell") != cell \
            or record.get("attempt") != {"file": attempt_name, "sha256": executed["attempt_sha256"]}:
        raise CellRefused("the recovered cell's record does not bind this recovery and its attempt")
    attempt, attempt_sha = _read_json(root / attempt_name, executed["attempt_sha256"],
                                      "the recovered cell's attempt reservation")
    if attempt.get("schema") != T_ATTEMPT_SCHEMA or attempt.get("namespace") != request["namespace"] \
            or attempt.get("campaign_nonce") != campaign_nonce or attempt.get("request") != request \
            or attempt.get("request_sha256") != M._json_digest(request) or attempt.get("cell") != cell \
            or attempt.get("approval") != approval:
        raise CellRefused("the recovered cell's attempt does not bind this recovery")
    entry_name = attempt_name.replace("_attempt_", "_entry_", 1)
    entry, entry_sha = _read_json(root / entry_name, None, "the recovered cell's entry claim")
    if entry.get("attempt") != attempt_name or entry.get("attempt_sha256") != attempt_sha \
            or entry.get("cell_id") != cell["cell_id"] \
            or entry.get("final_checkpoint_sha256") != cell["final_checkpoint_sha256"] \
            or entry.get("config_pt_sha256") != cell["config_pt_sha256"]:
        raise CellRefused("the recovered cell's entry claim does not bind its attempt and cell")
    assert_record_outputs(Path(cell["run_dir"]), record.get("completion") or {}, cell["cell_id"])
    return {"carried": {c["cell_id"]: {"record": c["record"], "attempt": c["attempt"], "entry": c["entry"]}
                        for c in admitted["carried"]},
            "settlement": admitted["settlement"], "claim_sha256": claim["sha256"], "exception_of": failed,
            "executed": {"record": {"file": record_name, "sha256": record_sha},
                         "attempt": {"file": attempt_name, "sha256": attempt_sha},
                         "entry": {"file": entry_name, "sha256": entry_sha}}}


def recovery_lineage_block(admitted: dict, lineage: dict, failed: dict) -> dict:
    """The part of the recovery request the lineage and the failed r8 attempt determine; the T entry
    recomputes it. r9: the claim is the exception's, keyed by the failed evidence (audit 841.2)."""
    stopped, recovery = lineage["stopped"], lineage["recovery_cell"]
    return {"historical_manifest": lineage["historical_manifest"],
            "lineage": {"path": str(RECOVERY_LINEAGE), "sha256": RECOVERY_LINEAGE_SHA256},
            "refit_receipt": admitted["request"]["refit_receipt"],
            "refit_approval": admitted["request"]["refit_approval"],
            "refit_epochs": admitted["request"]["refit_epochs"],
            "stopped": {"namespace": stopped["namespace"], "request_sha256": stopped["request_sha256"],
                        "snapshot": stopped["snapshot"], "reservation": stopped["reservation"],
                        "approval": stopped["approval"], "settlement": admitted["settlement"]},
            "recovery_of": {"cell_id": recovery["cell_id"], "attempt": recovery["attempt"],
                            "entry": recovery["entry"]},
            "carried": [{k: c[k] for k in ("cell_id", "record", "attempt", "entry")} for c in admitted["carried"]],
            "cells": [admitted["cell"]],
            "exception_of": failed,
            "claim_root": str(RECOVERY_CLAIM_ROOT),
            "claim_key": exception_claim_key(failed, recovery["cell_id"])}


def recovery_request(args, admitted: dict, *, manifest: dict, lineage: dict, transition: dict,
                     failed: dict) -> dict:
    """The recovery request: one cell on one GPU, the executing r8 generation and its reviewed
    transition from r7, the pinned lineage, the namespace and record root it writes, and the claim it
    consumes. An approval line of scope stage-T-recovery names its digest."""
    gpus = [g for g in (args.gpus.split(",") if args.gpus else [str(args.gpu)])]
    if len(gpus) != 1:
        raise CellRefused(f"the recovery runs one cell on exactly one GPU, got {gpus}")
    if str(args.namespace) in (lineage["stopped"]["namespace"], failed["namespace"]):
        raise CellRefused("the recovery writes a new namespace, never the stopped campaign's or the failed "
                          "recovery's")
    return {"schema": T_RECOVERY_REQUEST_SCHEMA, "version": M.ANCHOR_CONFIRM_VERSION, "stage": TEST_STAGE,
            "mode": RECOVERY_MODE, "manifest": manifest["sha256"], "manifest_path": str(manifest["path"]),
            "source_transition": {**transition,
                                  "changed_sources": {rel: manifest["files_sha256"][rel]
                                                      for rel in RECOVERY_CHANGED_SOURCES}},
            "freeze": {"path": str(ANCHOR_F_RECORD), "sha256": ANCHOR_F_RECORD_SHA256,
                       "acceptance_section": ANCHOR_F_ACCEPTANCE_SECTION,
                       "acceptance_sha256": ANCHOR_F_ACCEPTANCE_SHA256},
            **recovery_lineage_block(admitted, lineage, failed),
            "namespace": str(args.namespace), "record_dir": str(M.ANCHOR_RECORD_DIR),
            "outputs": list(M.OFFICIAL_TEST_OUTPUTS), "chain": list(T_CHAIN), "gpu_count": 1}


def recovery_claim_payload(request: dict, approval: dict) -> dict:
    """What the one recovery claim binds: the lineage key, the approved request and line, the namespace,
    the record root and the attempt name it authorizes (the T entry checks every field)."""
    return {"schema": T_RECOVERY_CLAIM_SCHEMA, "claim_key": request["claim_key"],
            "request_sha256": M._json_digest(request),
            "approval": {"section": approval["section"], "scope": approval["scope"], "line": approval["line"]},
            "namespace": request["namespace"], "record_dir": request["record_dir"],
            "attempt": attempt_path(request["namespace"], request["cells"][0]).name,
            "lineage": request["lineage"]}


def consume_recovery_claim(request: dict, approval: dict) -> tuple:
    """Atomically consume the lineage's ONE recovery authorization, before the attempt is reserved
    and before any test access (audit 797.2 item 3). The claim is O_EXCL in a root outside every
    worktree, keyed by the interrupted lineage alone, so a concurrent caller, a second namespace, an
    alternate record root or another worktree refuses; it is never removed, also after a failure."""
    if request["claim_root"] != str(RECOVERY_CLAIM_ROOT):
        raise CellRefused("the request names another recovery claim root")
    path = recovery_claim_path(request["claim_key"])
    payload = {**recovery_claim_payload(request, approval),
               "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "owner_pid": os.getpid(),
               "owner_host": socket.gethostname(), "owner_boot_id": M._boot_id()}
    RECOVERY_CLAIM_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        M._publish_json_exclusive(path, payload)
    except CellRefused:
        raise CellRefused(f"the interrupted lineage's one recovery is already claimed ({path.name}); "
                          "another recovery needs a new authorization") from None
    return path, M._sha(path)


def verify_recovery_authority(attempt_path: Path, attempt: dict, request: dict) -> dict:
    """The T entry's own proof of the recovery authority, before any configuration, model or test
    construction (audit 797.2 item 3): the pinned lineage re-admitted from its records (JSON, ledger
    and stat), the request's lineage part equal to what that admission determines, and the one
    recovery claim present, naming this request, approval line, namespace, record root and attempt."""
    lineage = recovery_lineage()
    freeze = anchor_freeze_authority()
    admitted = admit_stopped_campaign(lineage, freeze=freeze)
    failed = verify_failed_recovery(recovery_exception(), lineage)      # audit 841.3: entry recheck
    block = recovery_lineage_block(admitted, lineage, failed)
    if {k: request.get(k) for k in block} != block:
        raise CellRefused("the recovery request's lineage is not the pinned stopped campaign's and failed attempt's")
    try:
        claim = json.loads(recovery_claim_path(block["claim_key"]).read_bytes())
    except (OSError, ValueError) as error:
        raise CellRefused(f"the lineage's recovery claim is absent or unreadable: {error}") from None
    want = recovery_claim_payload(request, attempt.get("approval") or {})
    if {k: claim.get(k) for k in want} != want or attempt_path.name != want["attempt"]:
        raise CellRefused("the recovery claim does not authorize this request, namespace, record root and "
                          "attempt")
    return claim


def recovery_boundary(r_snapshot: dict, gpus, *, manifest: dict, historical: dict) -> dict:
    """terminal_test_boundary across the reviewed r7 -> r8 transition (audit 797.2 item 1): every
    stage-R source is its stage-R bytes except EXACTLY the allowlisted control-plane sources, each of
    which must be its r7 bytes in the stage-R snapshot and its r8 bytes now (clean in HEAD); the
    stage-R inputs unchanged; this run's own environment and leased GPU."""
    if r_snapshot.get("qwen_root") != str(M.QWEN_ROOT):
        raise CellRefused("the stage-R campaign used another Qwen root")
    before = r_snapshot.get("sources") or {}
    if not set(RECOVERY_CHANGED_SOURCES) <= set(before):
        raise CellRefused("the allowlisted sources are not stage-R sources")
    sources = {rel: (M._sha(M.REPO / rel) if (M.REPO / rel).is_file() else "#absent") for rel in before}
    changed = sorted(rel for rel in before if sources[rel] != before[rel])
    if changed != sorted(RECOVERY_CHANGED_SOURCES):
        raise CellRefused(f"the sources that differ from stage R are {changed}, not exactly the reviewed "
                          f"allowlist {sorted(RECOVERY_CHANGED_SOURCES)}")
    authority = M._source_authority_bundle(sources)
    entries = (r_snapshot.get("source_authority") or {}).get("entries") or {}
    for rel in before:
        now = authority["entries"][rel]
        if rel in RECOVERY_CHANGED_SOURCES:
            if before[rel] != historical["files_sha256"].get(rel) or sources[rel] != manifest["files_sha256"].get(rel) \
                    or not now.get("clean") or now.get("worktree_sha256") != sources[rel]:
                raise CellRefused(f"{rel}: not its r7 bytes at stage R and its committed r8 bytes now")
        elif (now.get("head_sha256"), now.get("worktree_sha256")) != \
                ((entries.get(rel) or {}).get("head_sha256"), (entries.get(rel) or {}).get("worktree_sha256")):
            raise CellRefused(f"{rel}: its source authority changed between stage R and the recovery")
    inputs = _snapshot_inputs_now(r_snapshot.get("inputs") or {})
    if inputs != r_snapshot.get("inputs"):
        raise CellRefused("the inputs changed between stage R and the recovery")
    environment = M.environment_fingerprint(gpus)
    if environment.get("errors"):
        raise CellRefused(f"the recovery environment has errors: {environment['errors']}")
    return {"qwen_root": str(M.QWEN_ROOT), "sources": sources, "inputs": inputs,
            "source_authority": authority, "source_authority_sha256": M._semantic_digest(authority),
            "environment": environment, "environment_sha256": M._semantic_digest(environment),
            "input_seals": r_snapshot.get("input_seals")}


def _recovery_main(args, manifest, execute: bool) -> int:
    if args.smoke or args.only:
        raise CellRefused("the recovery is one fixed cell: --run only, no --smoke or --only")
    if args.anchor_refit_receipt or args.anchor_refit_receipt_sha256:
        raise CellRefused("the recovery takes its stage-R receipt from the pinned lineage, not the command")
    lineage = recovery_lineage()
    freeze = anchor_freeze_authority()
    _check_freeze_against_incumbent(freeze, M.anchor_incumbent())
    historical = historical_generation(lineage)
    transition = generation_transition(manifest, historical)
    admitted = admit_stopped_campaign(lineage, freeze=freeze)
    failed = verify_failed_recovery(recovery_exception(), lineage)      # audit 841: the explicit exception
    request = recovery_request(args, admitted, manifest=manifest, lineage=lineage, transition=transition,
                               failed=failed)
    digest = M._json_digest(request)
    if not execute:
        print(f"{M.ANCHOR_CONFIRM_VERSION}: stage-T recovery of {request['recovery_of']['cell_id']} "
              f"({len(request['carried'])} cells carried), namespace {args.namespace}  "
              "[NON-EXECUTABLE until audit approval]")
        print("execution request that this command without --plan would need approved:")
        print(json.dumps(request, indent=1, sort_keys=True))
        print(f"execution request sha256 {digest}")
        return 0
    approval = M.audit_approval(args.anchor_approval_section, RECOVERY_SCOPE, manifest=manifest["sha256"],
                                freeze=ANCHOR_F_RECORD_SHA256, request=digest)
    verify_carried_payloads(admitted)
    M.recheck_generation(manifest, "after the recovery admission")
    M.RECORD_DIR = M.ANCHOR_RECORD_DIR
    return M._with_campaign_gpu_leases(
        args, lambda: _run_recovery(args, request=request, approval=approval, manifest=manifest,
                                    admitted=admitted, historical=historical))


def _run_recovery(args, *, request: dict, approval: dict, manifest: dict, admitted: dict,
                  historical: dict) -> int:
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
        boundary = recovery_boundary(r_snapshot, gpus, manifest=manifest, historical=historical)
    except CellRefused as error:
        print(f"[anchor-T] REFUSED: {error}", file=sys.stderr)
        return 2
    snapshot = {"schema": T_RECOVERY_SNAPSHOT_SCHEMA,
                "plan": {"namespace": args.namespace, "campaign_nonce": nonce,
                         "campaign_kind": T_CAMPAIGN_KIND, "result_root": r_snapshot["plan"].get("result_root"),
                         "declared_count": 1, "executed_count": 1,
                         "declared_cells": r_snapshot["plan"].get("declared_cells"),
                         "recovery_of": request["recovery_of"]},
                "request": request, "request_sha256": M._json_digest(request), "approval": approval,
                **boundary}
    try:
        M._assert_snapshot_gpu_leases(snapshot, args)
        snap_digest = M._json_digest(snapshot)
        snap_path = M.RECORD_DIR / f"{args.namespace}_snapshot_{snap_digest[:16]}.json"
        reservation = M.reserve_sweep_namespace(args.namespace, snapshot=snapshot, plan_digest=snap_digest,
                                                campaign_nonce=nonce, snapshot_file=snap_path.name)
        M._publish_json_exclusive(snap_path, snapshot)
    except CellRefused as error:
        print(f"[anchor-T] REFUSED: {error}", file=sys.stderr)
        return 2
    cell = request["cells"][0]
    uuid = str(snapshot["environment"]["selected_gpus"][0]["uuid"])
    claim, outcome = {}, {}

    def _recovery_worker():
        # Audit 839: managed campaign children launch only from a worker thread
        # (phase3_selection_matrix._run_managed_process refuses the main thread so a signal handler cannot
        # race child registration). Stage T runs its cells in _stream threads; the recovery's one cell runs
        # here, in one worker, from the space check to the final closure; the main thread only joins.
        try:
            M.assert_reservation_owner(args.namespace, reservation)
            M.recheck_generation(manifest, f"before the recovery of {cell['cell_id']}")
            refusal = M.anchor_dispatch_space_refusal(str(Path(cell["run_dir"]).parent), 1)
            if refusal:
                raise CellRefused(f"before {cell['cell_id']}: {refusal}")

            def claim_once():
                # audit 841.2-3: the failed attempt re-verified, and its consumed claim present, immediately
                # before the ONE exception claim is taken
                lineage_now = recovery_lineage()
                if verify_failed_recovery(recovery_exception(), lineage_now) != request["exception_of"]:
                    raise CellRefused("the failed recovery changed after admission")
                claim["path"], claim["sha256"] = consume_recovery_claim(request, approval)
            done = run_terminal_test_cell(cell, namespace=args.namespace, request=request, approval=approval,
                                          campaign_nonce=nonce, gpu_uuid=uuid, snapshot=snapshot,
                                          before_attempt=claim_once)
            M.recheck_generation(manifest, f"after the recovery of {cell['cell_id']}")
            M.assert_reservation_owner(args.namespace, reservation)
            outcome["closure"] = final_recovery_closure(request, freeze=anchor_freeze_authority(), claim=claim,
                                                        executed=done, approval=approval, campaign_nonce=nonce)
            outcome["done"] = done
        except Exception as error:                      # noqa: BLE001 -- fail-stop, no receipt
            outcome["error"] = error

    worker = threading.Thread(target=_recovery_worker, name=f"{args.namespace}-recovery", daemon=False)
    worker.start()
    worker.join()
    if "closure" not in outcome:                        # a failure, or a worker that ended without a closure
        reason = outcome.get("error", "the recovery worker ended without a final closure")
        print(f"[anchor-T] {cell['cell_id']} failed: {reason}; no receipt", file=sys.stderr)
        return 1
    done, closure = outcome["done"], outcome["closure"]
    hist_dir = Path(request["refit_receipt"]["path"]).parent
    cells = {cid: {"origin": "carried", "namespace": request["stopped"]["namespace"],
                   "record_dir": str(hist_dir), "record": c["record"]["file"],
                   "record_sha256": c["record"]["sha256"], "attempt_sha256": c["attempt"]["sha256"],
                   "entry_sha256": c["entry"]["sha256"]}
             for cid, c in closure["carried"].items()}
    executed = closure["executed"]
    cells[cell["cell_id"]] = {"origin": "executed", "namespace": args.namespace,
                              "record_dir": str(M.RECORD_DIR), "record": executed["record"]["file"],
                              "record_sha256": executed["record"]["sha256"],
                              "attempt_sha256": executed["attempt"]["sha256"],
                              "entry_sha256": executed["entry"]["sha256"],
                              "map_at_R": done.get("map_at_R"), "bio_map_at_R": done.get("bio_map_at_R")}
    receipt = {"schema": T_RECOVERY_RECEIPT_SCHEMA, "campaign_kind": T_CAMPAIGN_KIND, "stage": TEST_STAGE,
               "mode": RECOVERY_MODE, "namespace": args.namespace, "campaign_nonce": nonce,
               "plan_snapshot_file": snap_path.name, "plan_snapshot_sha256": snap_digest,
               "campaign_reservation_file": M.campaign_reservation_path(args.namespace).name,
               "campaign_reservation_sha256": M._sha(M.campaign_reservation_path(args.namespace)),
               "request_sha256": M._json_digest(request), "refit_receipt": request["refit_receipt"],
               "lineage": request["lineage"], "stopped": request["stopped"],
               "recovery_of": request["recovery_of"], "exception_of": request["exception_of"],
               "recovery_claim": {"file": Path(claim["path"]).name, "root": request["claim_root"],
                                  "sha256": closure["claim_sha256"]},
               "final_closure": {"settlement": closure["settlement"], "executed": executed},
               "expected_cells": len(cells), "cell_count": len(cells), "cells": cells}
    if len(cells) != 12:
        print("[anchor-T] the recovery does not complete twelve cells; no receipt", file=sys.stderr)
        return 1
    M._publish_json_exclusive(M.RECORD_DIR / f"{args.namespace}{T_RECOVERY_RECEIPT_SUFFIX}", receipt)
    print(f"[anchor-T] recovery complete: 1 executed + {len(cells) - 1} carried stage-T cells")
    return 0


def main(args) -> int:
    """`--anchor-confirm refit|test|recover`: the admission runs first in every mode; --plan prints the
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
        if execute:
            M.assert_preimport_handshake()   # before any admission: the verified launcher instance (r7)
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
        if args.anchor_confirm == RECOVERY_STAGE:
            return _recovery_main(args, manifest, execute)
        return _test_main(args, manifest, execute)
    except CellRefused as error:
        print(f"[phase3] REFUSED: {error}", file=sys.stderr)
        return 2
