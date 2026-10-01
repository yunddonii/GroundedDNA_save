#!/usr/bin/env python
"""Stage T entry (generation v9; docs/ANCHOR_REFIT_CONTRACT_v1.md; audits 743-744): the one
official-test extraction and RAW evaluation of one stage-R checkpoint.

    python -B scripts/anchor_terminal_test.py --config_path <stage-R run dir> \\
        --attempt <attempt reservation> --attempt-sha256 <hex>

The stage-T campaign (scripts/anchor_refit_stage.py) starts this, after it has reserved the attempt
exclusively. BEFORE any model, dataset or test construction this process proves the complete
admission itself, from records it does not take on trust from its caller's environment:
  1. the attempt reservation is the named bytes, of this schema, for exactly this run directory;
  2. its request is the digest it states, and names this cell (run directory, record, terminal
     checkpoint and runtime witness);
  3. the audit ledger still carries the stage-T approval line for exactly that request, generation
     manifest and F record, and the line is the one the reservation recorded;
  4. the tree is that generation (manifest re-verified, imported modules included);
  5. the terminal checkpoint's bytes and its runtime witness are the pinned ones, at the terminal
     epoch the request allows; the trainer's campaign evidence is the pinned bytes, names this cell
     and its sealed recipe, and is the witness's own; config.pt is the pinned bytes; the run
     directory holds no official-test output yet;
  6. it CLAIMS the entry exclusively and durably (<ns>_entry_<R tag>.json, O_EXCL; audit 746.1),
     before any configuration, model or test access: the same attempt cannot enter twice, after a
     failure before any output or concurrently, and the claim is never removed;
  7. config.pt is read ONCE: its bytes are hashed against the pin, deserialized from those bytes,
     and must carry every typed field of this cell's sealed recipe (read from the stage-R plan
     snapshot the request's receipt binds) and this cell's campaign binding (audit 746.2); the
     arguments are built from that same object by the shared resume helper's own flat-layout step
     (no second read; audit 754.2), their effective scientific fields are checked again, and they
     must be this anchor stage-R cell's (axis_center, refit mode, cell id, run dir).
Only then does it call terminal_official_test.run_official_test -- the trainer's own terminal block:
extract_code (query, db) and the raw evaluation. A refusal exits 2 (after step 6 the claim stays);
a failure after the test is touched exits nonzero; no retry without a new authorization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

#: the bytes this module was imported from (audit 697)
with open(__file__, "rb") as _source:
    _IMPORTED_SOURCE_SHA256 = hashlib.sha256(_source.read()).hexdigest()


class Refused(RuntimeError):
    pass


def need(condition, message: str) -> None:
    if not condition:
        raise Refused(message)


def _sha(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def entry_claim_path(attempt_path: Path) -> Path:
    name = attempt_path.name
    if "_attempt_" not in name:
        raise Refused(f"{attempt_path}: not an attempt reservation name")
    return attempt_path.with_name(name.replace("_attempt_", "_entry_", 1))


def admit(run_dir: Path, attempt_path: Path, attempt_sha256: str) -> tuple:
    """Steps 1-5 of the module docstring; returns (cell, request, attempt). Reads JSON, the ledger
    text and the checkpoint's and config's BYTES (hashed, not loaded)."""
    import scripts.phase3_selection_matrix as M
    import scripts.anchor_refit_stage as RT
    raw = attempt_path.read_bytes()
    need(hashlib.sha256(raw).hexdigest() == attempt_sha256,
         f"{attempt_path} is not the reserved attempt bytes")
    attempt = json.loads(raw)
    need(attempt.get("schema") == RT.T_ATTEMPT_SCHEMA, f"{attempt_path}: not a stage-T attempt reservation")
    need(attempt_path.resolve().parent == M.ANCHOR_RECORD_DIR.resolve(),
         f"{attempt_path}: an attempt reservation lives in the anchor record directory")
    request, cell = attempt.get("request") or {}, attempt.get("cell") or {}
    need(M._json_digest(request) == attempt.get("request_sha256"), "the attempt's request is not its digest")
    need(request.get("schema") == RT.T_REQUEST_SCHEMA and request.get("stage") == RT.TEST_STAGE,
         "the attempt's request is not a stage-T request")
    need(cell in (request.get("cells") or []), "the attempt's cell is not one of its request's cells")
    need(cell.get("config_pt_sha256") and cell.get("scientific_recipe_sha256")
         and cell.get("phase3_campaign_evidence_sha256"), "the request does not pin this cell's identity")
    need(Path(cell.get("run_dir", "")).resolve() == run_dir.resolve(),
         f"the attempt is for {cell.get('run_dir')}, not {run_dir}")
    need(attempt_path.name == RT.attempt_path(request["namespace"], cell).name,
         "the attempt reservation is not named for its cell")
    scope = f"stage-T-{request.get('mode')}"
    try:
        live = M.audit_approval((attempt.get("approval") or {}).get("section"), scope,
                                manifest=request.get("manifest"), freeze=RT.ANCHOR_F_RECORD_SHA256,
                                request=attempt["request_sha256"])
    except M.CellRefused as error:
        raise Refused(f"no standing stage-T approval for this request: {error}") from None
    need(live["line"] == (attempt.get("approval") or {}).get("line"),
         "the approval line the attempt recorded is not the ledger's")
    need((request.get("freeze") or {}).get("sha256") == RT.ANCHOR_F_RECORD_SHA256,
         "the request names another F record")
    try:
        M.recheck_generation({"path": request["manifest_path"], "sha256": request["manifest"]},
                             "in the stage-T entry")
    except M.CellRefused as error:
        raise Refused(str(error)) from None
    checkpoint = run_dir / cell["final_checkpoint"]
    need(_sha(checkpoint) == cell["final_checkpoint_sha256"], f"{checkpoint} is not the stage-R checkpoint")
    sidecar_path = run_dir / f"{cell['final_checkpoint']}.runtime.json"
    sidecar_raw = sidecar_path.read_bytes()
    need(hashlib.sha256(sidecar_raw).hexdigest() == cell["checkpoint_runtime_sha256"],
         f"{sidecar_path} is not the stage-R runtime witness")
    sidecar = json.loads(sidecar_raw)
    expected_terminal = cell["N"] if request.get("mode") == "run" else \
        int((request.get("refit_epochs") or 0)) - 1
    need(sidecar.get("checkpoint_epoch_zero_based") == cell["terminal_epoch"] == expected_terminal,
         f"the runtime witness is not at the terminal epoch {expected_terminal}")
    from dna_utils.run_identity import PHASE3_CAMPAIGN_BINDING_NAME
    evidence_raw = (run_dir / PHASE3_CAMPAIGN_BINDING_NAME).read_bytes()
    need(hashlib.sha256(evidence_raw).hexdigest() == cell["phase3_campaign_evidence_sha256"],
         "the trainer's campaign evidence is not the pinned bytes")
    evidence = json.loads(evidence_raw)
    need(evidence.get("cell_id") == cell["cell_id"]
         and evidence.get("scientific_recipe_sha256") == cell["scientific_recipe_sha256"]
         and (sidecar.get("extra") or {}).get("phase3_campaign") == evidence,
         "the trainer evidence, sealed recipe and runtime witness do not bind this cell")
    need(_sha(run_dir / "config.pt") == cell["config_pt_sha256"], "config.pt is not the pinned bytes")
    try:
        M.assert_official_test_withheld(run_dir)
    except M.CellRefused as error:
        raise Refused(str(error)) from None
    return cell, request, attempt


def claim_entry(attempt_path: Path, attempt_sha256: str, cell: dict) -> Path:
    """Step 6: the exclusive, durable entry claim (audit 746.1). Never removed."""
    import socket
    import time
    path = entry_claim_path(attempt_path)
    payload = {"attempt": attempt_path.name, "attempt_sha256": attempt_sha256, "cell_id": cell["cell_id"],
               "final_checkpoint_sha256": cell["final_checkpoint_sha256"],
               "config_pt_sha256": cell["config_pt_sha256"], "pid": os.getpid(),
               "host": socket.gethostname(), "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    except FileExistsError:
        raise Refused(f"{cell['cell_id']}: this attempt has already entered stage T ({path.name}); "
                      "a second entry needs a new authorization") from None
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=1, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return path


def sealed_recipe(request: dict, cell: dict) -> dict:
    """This cell's sealed recipe, from the stage-R plan snapshot of the request's pinned receipt."""
    import scripts.phase3_selection_matrix as M
    from dna_utils.scientific_recipe import digest
    raw = Path(request["refit_receipt"]["path"]).read_bytes()
    need(hashlib.sha256(raw).hexdigest() == request["refit_receipt"]["sha256"], "the stage-R receipt changed")
    receipt = json.loads(raw)
    snapshot = json.loads((Path(request["refit_receipt"]["path"]).parent
                           / receipt["plan_snapshot_file"]).read_bytes())
    need(M._json_digest(snapshot) == receipt["plan_snapshot_sha256"], "the stage-R snapshot is not its receipt's")
    payload = ((snapshot.get("plan") or {}).get("cell_bindings") or {}).get(cell["cell_id"], {}).get(
        "scientific_recipe")
    need(isinstance(payload, dict) and digest(payload) == cell["scientific_recipe_sha256"],
         "the sealed recipe of this cell is not the pinned one")
    return payload


def verified_configuration(run_dir: Path, cell: dict, payload: dict):
    """Step 7: config.pt read ONCE, its bytes hashed against the pin, deserialized from those bytes (as
    the shared resume helper loads it) and checked: every typed field of the sealed recipe and this
    cell's campaign binding. Returns (args, saved): the arguments are built from THIS object by the
    shared helper's own flat-layout step, never from a second read of the file (audit 754.2), and the
    effective scientific fields of those arguments are checked again before they are used."""
    import io
    import torch
    from config import Config
    from dna_utils.scientific_recipe import field_differences, fields_from_config
    from extraction_siglip2 import _apply_saved_config, _reapply_explicit_cli
    raw = (run_dir / "config.pt").read_bytes()
    need(hashlib.sha256(raw).hexdigest() == cell["config_pt_sha256"], "config.pt is not the pinned bytes")
    saved = torch.load(io.BytesIO(raw), map_location="cpu")
    parser = Config.build_parser()
    differing = field_differences(fields_from_config(saved, parser), payload["fields"])
    need(not differing, f"the saved configuration differs from the sealed recipe in {differing[:8]}")
    need((saved.get("_phase3_campaign_binding") or {}).get("cell_id") == cell["cell_id"],
         "the saved configuration's campaign binding names another cell")
    args = Config()
    # the shared helper's CLI is --config_path alone (as extract_train_split runs it)
    sys.argv = [sys.argv[0], "--config_path", str(run_dir)]
    cli = Config.get_config()
    _apply_saved_config(args, saved, str(run_dir))
    _reapply_explicit_cli(args, cli)
    effective = field_differences(fields_from_config(vars(args), parser), payload["fields"])
    need(not effective, f"the effective arguments differ from the sealed recipe in {effective[:8]}")
    return args, saved


def check_config(args, cell: dict, run_dir: Path) -> None:
    """Step 6, on the configuration the trainer saved (loaded by the shared resume helper)."""
    campaign = getattr(args, "_phase3_campaign_binding", None) or {}
    need(getattr(args, "axis_center", None) == "anchors", "the saved configuration is not the anchor model")
    need(getattr(args, "selection_mode", None) == "refit" and bool(getattr(args, "final_epoch_eval", False)),
         "the saved configuration is not a full-train refit")
    need(campaign.get("cell_id") == cell["cell_id"], "the saved campaign binding names another cell")
    need(Path(str(args.save_result_path)).resolve() == run_dir.resolve(),
         "the saved configuration resolves to another run directory")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--attempt-sha256", required=True)
    cli = parser.parse_args(argv)
    run_dir = Path(cli.config_path)
    try:
        cell, request, _attempt = admit(run_dir, Path(cli.attempt), cli.attempt_sha256)
        payload = sealed_recipe(request, cell)
        claim_entry(Path(cli.attempt), cli.attempt_sha256, cell)
    except (Refused, OSError, ValueError, KeyError) as error:
        print(f"[anchor-T] REFUSED before the official test: {type(error).__name__}: {error}",
              file=sys.stderr)
        return 2
    try:
        from config import set_random_seed
        args, _saved = verified_configuration(run_dir, cell, payload)
        check_config(args, cell, run_dir)
    except (Refused, OSError, ValueError, KeyError) as error:
        print(f"[anchor-T] REFUSED before the official test (entry claimed): {type(error).__name__}: {error}",
              file=sys.stderr)
        return 2
    set_random_seed(42)      # as the train-split extraction of the same chain
    from terminal_official_test import run_official_test
    print(f"[anchor-T] {cell['cell_id']}: admitted; official-test extraction and raw evaluation")
    run_official_test(args, distance_mode=str(getattr(args, "dna_distance_mode", "base")),
                      codebook_size=int(getattr(args, "codebook_size", 32)))
    print(f"[anchor-T] {cell['cell_id']}: done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
