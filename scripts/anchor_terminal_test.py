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
     epoch, and the run directory holds no official-test output yet;
  6. the saved configuration is this anchor stage-R cell's (axis_center, refit mode, cell id, run dir).
Only then does it call terminal_official_test.run_official_test -- the trainer's own terminal block:
extract_code (query, db) and the raw evaluation. A refusal exits 2 before the test is touched; a
failure after it exits nonzero, and the reservation stays (no retry without a new authorization).
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


def admit(run_dir: Path, attempt_path: Path, attempt_sha256: str) -> dict:
    """Steps 1-5 of the module docstring; returns the admitted cell. Reads JSON, the ledger text and
    the checkpoint's BYTES (hashed, not loaded)."""
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
    need(json.loads(sidecar_raw).get("checkpoint_epoch_zero_based") == cell["terminal_epoch"],
         "the runtime witness is not at the terminal epoch")
    try:
        M.assert_official_test_withheld(run_dir)
    except M.CellRefused as error:
        raise Refused(str(error)) from None
    return cell


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
        cell = admit(run_dir, Path(cli.attempt), cli.attempt_sha256)
    except (Refused, OSError, ValueError, KeyError) as error:
        print(f"[anchor-T] REFUSED before the official test: {type(error).__name__}: {error}",
              file=sys.stderr)
        return 2
    from config import Config, set_random_seed
    from extraction_siglip2 import _resume_args_flat_or_legacy
    args = Config()
    # the shared resume helper reads --config_path from sys.argv (as extract_train_split does)
    sys.argv = [sys.argv[0], "--config_path", str(run_dir)]
    _resume_args_flat_or_legacy(args)
    try:
        check_config(args, cell, run_dir)
    except Refused as error:
        print(f"[anchor-T] REFUSED before the official test: {error}", file=sys.stderr)
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
