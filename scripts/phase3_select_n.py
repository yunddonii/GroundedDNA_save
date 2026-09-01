"""Choose (N) per dataset from the completed 16-cell matrix (D1).

The rule is short and the enforcement is the whole point:

    argmax over the candidate's raw base-Hamming mAP@R at ITS OWN terminal
    epoch; ties go to the SMALLEST N; the chosen N is then reused unchanged for
    seeds 42, 43 and 44.

What this refuses, because each was reachable:

  * fewer than sixteen records, or more, or two for one cell -- a partial
    matrix picks N from whichever cells happened to finish;
  * records from more than one namespace or protocol -- a matrix assembled from
    two different launchers compares numbers that were not produced the same way;
  * smoke records -- they are short runs with the horizons collapsed, and the
    launcher marks them `is_candidate_cell: false`;
  * a record whose selection did not come from its own terminal epoch.

The output names every record digest it read and the protocol sources those
records were produced under, so the choice can be re-derived rather than
believed.

Usage:
    python scripts/phase3_select_n.py
    python scripts/phase3_select_n.py --records artifacts/phase3_selection
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dna_utils.run_identity import load_run_manifest  # noqa: E402
from scripts.phase3_selection_matrix import (  # noqa: E402
    BASES_PER_SLOT,
    CANDIDATE_N,
    DATASETS,
    LR_HORIZON,
    RECORD_SCHEMA,
    SEED,
    SLOTS,
    TOTAL_BASES,
    TOTAL_BITS,
    VAL_RATIO,
    VAL_SEED,
    cell_keys,
    protocol_digests,
)

DEFAULT_RECORDS = REPO / "artifacts" / "phase3_selection"
DEFAULT_OUT = DEFAULT_RECORDS / "selected_n.json"
SELECTED_N_SCHEMA = 2
REFIT_AGGREGATE_SCHEMA = 1
REFIT_AGGREGATE_SUFFIX = "_refit_aggregate.json"
N_SELECTION_REDUCTION = {
    "metric": "eval_mAP_at_R",
    "distance": "base_hamming",
    "split": "train_only_validation",
    "candidate_epoch": "own_terminal_epoch",
    "tie_break": "smallest_N",
    "candidate_n": list(CANDIDATE_N),
    "seed": SEED,
}


class SelectionRefused(RuntimeError):
    """The matrix cannot be reduced to a choice."""


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _reverify_run(payload: dict, path: Path) -> None:
    """Open the run the record describes and re-derive its claims.

    The record is a report, and every field in it was written by the same
    process that is now being trusted. A safe probe built sixteen records whose
    `run_dir` did not exist and whose completion digests were repeated
    characters, and they were accepted -- the checks were shape and truthiness.
    So the directory is opened, the manifest is re-read, the checkpoint and
    log.csv are re-hashed, and the selected value is re-read from the CSV.
    """
    run_dir = Path(payload["run_dir"])
    if not run_dir.is_dir():
        raise SelectionRefused(
            f"{path.name}: run_dir {run_dir} does not exist, so nothing in "
            f"this record can be re-derived")

    identity = load_run_manifest(str(run_dir))
    if identity is None:
        raise SelectionRefused(
            f"{path.name}: {run_dir} has no readable run manifest under the "
            f"current identity schema")
    if identity.digest != payload.get("identity_digest"):
        raise SelectionRefused(
            f"{path.name}: the run's identity is {identity.digest[:12]}..., "
            f"the record says {str(payload.get('identity_digest'))[:12]}...")

    completion = payload["completion"]
    manifest_path = run_dir / "run_identity.json"
    if completion.get("run_identity_sha256") is not None \
            and _sha(manifest_path) != completion.get("run_identity_sha256"):
        raise SelectionRefused(
            f"{path.name}: run manifest does not hash to completion identity")
    checkpoint = run_dir / str(completion.get("final_checkpoint") or
                               "model_state_dict.pth")
    if not checkpoint.is_file():
        raise SelectionRefused(
            f"{path.name}: {checkpoint.name} is not in {run_dir}")
    if _sha(checkpoint) != completion.get("final_checkpoint_sha256"):
        raise SelectionRefused(
            f"{path.name}: {checkpoint.name} does not hash to the digest the "
            f"record claims; the weights changed after the cell finished")

    # Recipe sweeps carry a trainer-owned launch witness both as a standalone
    # file and inside the checkpoint runtime sidecar.  Manifests, args, records,
    # receipts and directory names can all be rewritten after training; these
    # bytes say which prelaunch cell the checkpoint process itself received.
    campaign = payload.get("campaign")
    if campaign is not None:
        from dna_utils.run_identity import (
            PHASE3_CAMPAIGN_BINDING_NAME, RunCollision,
            phase3_campaign_from_checkpoint)
        from scripts.phase3_selection_matrix import CAMPAIGN_LAUNCH_FIELDS
        from scripts.phase3_selection_matrix import assert_completed

        try:
            reopened_completion = assert_completed(
                run_dir, terminal_epoch=int(payload["N"]),
                campaign_binding=campaign)
        except Exception as error:  # noqa: BLE001 - normalize reducer refusal
            raise SelectionRefused(
                f"{path.name}: terminal checkpoint/runtime refusal: {error}") \
                from None
        mismatch = {
            key: (completion.get(key), value)
            for key, value in reopened_completion.items()
            if completion.get(key) != value
        }
        if mismatch:
            raise SelectionRefused(
                f"{path.name}: completion differs from reopened terminal "
                f"checkpoint evidence: {mismatch}")

        runtime = Path(str(checkpoint) + ".runtime.json")
        evidence = run_dir / PHASE3_CAMPAIGN_BINDING_NAME
        for artifact, field in (
                (runtime, "checkpoint_runtime_sha256"),
                (evidence, "phase3_campaign_evidence_sha256"),
                (run_dir / "args.txt", "args_txt_sha256")):
            if not artifact.is_file():
                raise SelectionRefused(
                    f"{path.name}: campaign cell has no {artifact.name}")
            if _sha(artifact) != completion.get(field):
                raise SelectionRefused(
                    f"{path.name}: {artifact.name} does not hash to "
                    f"completion.{field}")
        try:
            runtime_payload = json.loads(runtime.read_text(encoding="utf-8"))
            trainer_evidence = json.loads(evidence.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise SelectionRefused(
                f"{path.name}: unreadable trainer campaign evidence: {error}")
        if runtime_payload.get("checkpoint_sha256") != _sha(checkpoint):
            raise SelectionRefused(
                f"{path.name}: checkpoint runtime sidecar names other weights")
        if (runtime_payload.get("extra") or {}).get("phase3_campaign") != \
                trainer_evidence:
            raise SelectionRefused(
                f"{path.name}: runtime sidecar and trainer launch witness differ")
        core = tuple(dict.fromkeys(
            ("plan_snapshot_sha256", *CAMPAIGN_LAUNCH_FIELDS)))
        wrong = {field: (trainer_evidence.get(field), campaign.get(field))
                 for field in core
                 if trainer_evidence.get(field) != campaign.get(field)}
        if wrong:
            raise SelectionRefused(
                f"{path.name}: trainer-produced checkpoint evidence belongs to "
                f"another planned cell: {wrong}")
        if trainer_evidence.get("actual_identity_digest") != identity.digest:
            raise SelectionRefused(
                f"{path.name}: trainer evidence identity "
                f"{trainer_evidence.get('actual_identity_digest')} differs from "
                f"the reopened manifest {identity.digest}")
        from dna_utils.runtime_environment import (
            EnvironmentAttestationError, semantic_digest,
            verify_child_environment)
        expected_child = trainer_evidence.get("expected_child_environment")
        actual_child = trainer_evidence.get("actual_child_environment")
        try:
            if not isinstance(expected_child, dict) \
                    or semantic_digest(expected_child) != \
                    campaign.get("child_environment_sha256"):
                raise EnvironmentAttestationError(
                    "child environment digest mismatch")
            verify_child_environment(expected_child, actual=actual_child)
        except EnvironmentAttestationError as error:
            raise SelectionRefused(
                f"{path.name}: child environment attestation is invalid: "
                f"{error}") from None
        try:
            checkpoint_campaign = phase3_campaign_from_checkpoint(
                str(checkpoint))
        except RunCollision as error:
            raise SelectionRefused(f"{path.name}: {error}") from None
        if checkpoint_campaign != trainer_evidence:
            raise SelectionRefused(
                f"{path.name}: checkpoint serialization commits another "
                "campaign cell")

    csv_path = run_dir / "log.csv"
    if not csv_path.is_file():
        raise SelectionRefused(f"{path.name}: {run_dir} has no log.csv")
    if _sha(csv_path) != completion.get("log_csv_sha256"):
        raise SelectionRefused(
            f"{path.name}: log.csv does not hash to completion.log_csv_sha256")
    if payload.get("stage") == "refit":
        _reverify_refit_outputs(payload, path, identity, checkpoint)
        return
    selection = payload["selection"]
    if _sha(csv_path) != selection.get("log_csv_sha256"):
        raise SelectionRefused(
            f"{path.name}: log.csv does not hash to the digest the record "
            f"claims; the metric was read from different bytes")

    # And the number itself, re-read from those bytes rather than copied.
    import csv as _csv
    with open(csv_path, newline="", encoding="utf-8") as handle:
        rows = {int(r["epoch"]): r for r in _csv.DictReader(handle)
                if (r.get("epoch") or "").strip().isdigit()}
    epoch = selection["selection_epoch_zero_based"]
    if epoch not in rows:
        raise SelectionRefused(
            f"{path.name}: log.csv has no row for epoch {epoch}")
    raw = (rows[epoch].get("eval_mAP_at_R") or "").strip()
    try:
        actual = float(raw)
    except ValueError:
        raise SelectionRefused(
            f"{path.name}: log.csv epoch {epoch} eval_mAP_at_R={raw!r} is not "
            f"a number") from None
    if abs(actual - float(selection["selection_value"])) > 1e-12:
        raise SelectionRefused(
            f"{path.name}: the record says {selection['selection_value']}, "
            f"log.csv epoch {epoch} says {actual}")


def _reverify_refit_outputs(payload: dict, path: Path, identity,
                            checkpoint: Path) -> None:
    """Reopen the official extraction/evaluation identity of a refit."""
    del identity, checkpoint  # re-opened by the single canonical verifier
    from scripts.phase3_selection_matrix import assert_refit_outputs

    completion = payload["completion"]
    try:
        reopened = assert_refit_outputs(
            Path(payload["run_dir"]), dataset=payload["dataset"])
    except Exception as error:  # noqa: BLE001 - normalize selector refusal
        raise SelectionRefused(
            f"{path.name}: final refit evidence is invalid: {error}") from None
    mismatch = {key: (completion.get(key), value)
                for key, value in reopened.items()
                if completion.get(key) != value}
    if mismatch:
        raise SelectionRefused(
            f"{path.name}: refit completion differs from strict reopened "
            f"outputs: {mismatch}")


def _assert_claimed_coordinate(payload: dict, path: Path) -> None:
    """§58.3: the record's `recipe` label must be what the RUN actually parsed.

    Three genuine Flickr runs -- one underlying identity between them, because
    the recipe axes were absent -- were relabelled `.3/.7`, `.4/.8`, `.6/.95`
    in their record JSONs and the reducer accepted all three AND changed its
    winner. Reopening the checkpoint and the CSV does not catch that: the bytes
    are real, it is the label that is wrong. So the effective values are read
    back out of `args.txt`, which is what the trainer parsed.
    """
    from scripts.phase3_selection_matrix import _arg_value

    run_dir = Path(payload["run_dir"])
    claimed = payload.get("recipe") or {}

    # The MANIFEST first. `args.txt` is a plain text file that nothing hashes,
    # so rotating it together with the record labels passed the earlier version
    # of this check. The run manifest's stored digest is verified against its
    # own fields by `load_run_manifest`, and schema 4 carries the recipe axes,
    # so it is the tamper-evident copy of what ran.
    identity = load_run_manifest(str(run_dir))
    if identity is None:
        raise SelectionRefused(
            f"{path.name}: {run_dir} has no readable run manifest")
    for field in ("routing_adaptive_topp_min", "routing_adaptive_topp_max",
                  "lambda_codon_joint"):
        want = claimed.get(field)
        if want is None:
            raise SelectionRefused(
                f"{path.name}: the record claims no {field}")
        actual = getattr(identity, field)
        if actual is None or abs(float(actual) - float(want)) > 1e-12:
            raise SelectionRefused(
                f"{path.name}: the record is labelled {field}={want} but the "
                f"run's sealed identity says {actual}; this is a genuine run "
                f"under someone else's coordinate")

    # And `args.txt` as a second witness, so a manifest and the arguments the
    # trainer parsed cannot disagree unnoticed.
    args_txt = run_dir / "args.txt"
    if not args_txt.is_file():
        raise SelectionRefused(f"{path.name}: {run_dir} has no args.txt")
    for field in ("routing_adaptive_topp_min", "routing_adaptive_topp_max",
                  "lambda_codon_joint"):
        parsed = float(_arg_value(args_txt, field))
        if abs(parsed - float(claimed[field])) > 1e-12:
            raise SelectionRefused(
                f"{path.name}: args.txt says {field}={parsed}, the record says "
                f"{claimed[field]}")


def _check_record(payload: dict, path: Path) -> tuple:
    if payload.get("schema_version") != RECORD_SCHEMA:
        raise SelectionRefused(
            f"{path.name}: unknown record schema_version "
            f"{payload.get('schema_version')!r}")
    if not payload.get("is_candidate_cell"):
        raise SelectionRefused(
            f"{path.name}: marked as not a candidate cell (a smoke run has "
            f"its horizons collapsed and cannot be compared)")

    dataset, n = payload.get("dataset"), payload.get("N")
    if (dataset, n) not in cell_keys():
        raise SelectionRefused(
            f"{path.name}: ({dataset}, {n}) is not one of the "
            f"{len(cell_keys())} cells")

    # A record must carry the evidence a cell produces, not merely the shape a
    # reader looks for. Sixteen hand-written JSONs with none of this -- and
    # `protocol_sources: null` throughout -- passed the first version.
    required = ("completion", "inputs", "matrix", "stage", "epoch_budget",
                "run_dir", "tag", "identity_digest", "protocol_sources")
    absent = [k for k in required if payload.get(k) in (None, "", {}, [])]
    if absent:
        raise SelectionRefused(
            f"{path.name}: no {', '.join(absent)} -- this is not a record a "
            f"completed cell wrote")
    if payload.get("stage") != "select":
        raise SelectionRefused(
            f"{path.name}: stage is {payload.get('stage')!r}, not a selection "
            f"cell")
    completion = payload["completion"]
    for field in ("final_checkpoint_sha256", "log_csv_sha256"):
        digest = completion.get(field)
        if not isinstance(digest, str) or len(digest) != 64:
            raise SelectionRefused(
                f"{path.name}: completion.{field} is {digest!r}, not a digest")
    if not (payload.get("selection") or {}).get("map_r_cutoff"):
        raise SelectionRefused(
            f"{path.name}: the metric records no mAP@R cutoff")

    expected = {
        "seed": SEED, "val_split_ratio": VAL_RATIO, "val_split_seed": VAL_SEED,
        "lr_schedule_horizon": LR_HORIZON,
        "sinkhorn_schedule_horizon": n + 1,
    }
    wrong = {k: (payload.get(k), v) for k, v in expected.items()
             if payload.get(k) != v}
    if wrong:
        raise SelectionRefused(f"{path.name}: protocol differs {wrong}")

    geometry = payload.get("geometry") or {}
    geo_wrong = {k: (geometry.get(k), v) for k, v in (
        ("num_semantic_parts", SLOTS), ("num_codebooks", SLOTS),
        ("num_codons_per_codebook", BASES_PER_SLOT)) if geometry.get(k) != v}
    if geo_wrong:
        raise SelectionRefused(f"{path.name}: geometry differs {geo_wrong}")

    selection = payload.get("selection") or {}
    if selection.get("selection_metric") != "eval_mAP_at_R":
        raise SelectionRefused(
            f"{path.name}: selection metric is "
            f"{selection.get('selection_metric')!r}, not raw base-Hamming "
            f"mAP@R")
    if selection.get("selection_epoch_zero_based") != n:
        raise SelectionRefused(
            f"{path.name}: the value comes from epoch "
            f"{selection.get('selection_epoch_zero_based')}, not the "
            f"candidate's own terminal epoch {n}")
    value = selection.get("selection_value")
    if not isinstance(value, (int, float)) or isinstance(value, bool) \
            or not 0.0 <= float(value) <= 1.0:
        raise SelectionRefused(
            f"{path.name}: selection value {value!r} is not a proportion")
    _reverify_run(payload, path)
    return (dataset, n), float(value)


def _campaign_rows(rows) -> list:
    from scripts.phase3_selection_matrix import _campaign_cell_parts

    out = []
    for row in rows:
        dataset, n, topp, joint, stage, seed = _campaign_cell_parts(row)
        out.append({"dataset": dataset, "N": n,
                    "topp": list(topp), "joint": joint,
                    "stage": stage, "seed": seed})
    return out


def _verify_production_source_admission(snapshot: dict, label: str) -> None:
    """Recheck the two source facts proven before any production import/run.

    A later selector must not infer admission merely from a self-consistent
    digest string.  Production requires the stdlib bootstrap digest to equal
    the snapshot digest and every executable source to have been a clean,
    tracked HEAD blob at launch.
    """
    authority = snapshot.get("source_authority")
    if not isinstance(authority, dict):
        raise SelectionRefused(f"{label} has no source-authority bundle")
    digest = hashlib.sha256(json.dumps(
        authority, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if snapshot.get("source_authority_sha256") != digest \
            or snapshot.get("preimport_source_authority_sha256") != digest:
        raise SelectionRefused(
            f"{label} was not admitted by one identical pre-import and "
            "snapshot source authority")
    entries = authority.get("entries")
    if not isinstance(entries, dict) or not entries:
        raise SelectionRefused(f"{label} source authority has no entries")
    dirty = sorted(
        rel for rel, state in entries.items()
        if not isinstance(state, dict)
        or state.get("tracked") is not True
        or state.get("clean") is not True)
    if dirty:
        raise SelectionRefused(
            f"{label} production sources were untracked or dirty: {dirty}")


def _campaign_envelope(records_dir: Path, *, namespace: str | None,
                       campaign_kind: str, receipt_suffix: str,
                       canonical_plan) -> dict:
    """Reopen one campaign solely through its exact immutable membership."""
    from scripts.phase3_selection_matrix import (
        CAMPAIGN_RESERVATION_SCHEMA, CAMPAIGN_RESERVATION_SUFFIX,
        SWEEP_RECEIPT_SCHEMA, SWEEP_SNAPSHOT_SCHEMA, _json_digest,
        campaign_cell_id, expected_cell_binding,
        launch_binding_from_expected, verify_snapshot)

    if namespace is None:
        matches = sorted(records_dir.glob(f"*{receipt_suffix}"))
        if len(matches) != 1:
            raise SelectionRefused(
                f"N reduction needs one explicit namespace; found "
                f"{len(matches)} {receipt_suffix} receipts")
        namespace = matches[0].name[:-len(receipt_suffix)]
    receipt_path = records_dir / f"{namespace}{receipt_suffix}"
    if not receipt_path.is_file():
        raise SelectionRefused(f"{receipt_path.name} does not exist")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise SelectionRefused(f"{receipt_path.name} is unreadable: {error}")
    axis = "n" if campaign_kind == "n_selection" else "refit"
    if receipt.get("schema_version") != SWEEP_RECEIPT_SCHEMA \
            or receipt.get("campaign_kind") != campaign_kind \
            or receipt.get("axis") != axis \
            or receipt.get("namespace") != namespace:
        raise SelectionRefused(
            f"{receipt_path.name} is not a schema-{SWEEP_RECEIPT_SCHEMA} "
            f"{campaign_kind} receipt for {namespace}")

    reservation_path = records_dir / f"{namespace}{CAMPAIGN_RESERVATION_SUFFIX}"
    if receipt.get("campaign_reservation_file") != reservation_path.name \
            or not reservation_path.is_file() \
            or _sha(reservation_path) != receipt.get(
                "campaign_reservation_sha256"):
        raise SelectionRefused(
            f"{receipt_path.name} does not bind the exact namespace reservation")
    try:
        reservation = json.loads(reservation_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise SelectionRefused(
            f"{reservation_path.name} is unreadable: {error}")
    if reservation.get("schema_version") != CAMPAIGN_RESERVATION_SCHEMA \
            or reservation.get("namespace") != namespace \
            or reservation.get("campaign_kind") != campaign_kind:
        raise SelectionRefused(
            f"{reservation_path.name} is not this {campaign_kind} reservation")
    if not isinstance(reservation.get("owner_pid"), int) \
            or isinstance(reservation.get("owner_pid"), bool) \
            or not reservation.get("owner_boot_id"):
        raise SelectionRefused(
            f"{reservation_path.name} has no owner PID/boot-id")
    nonce = reservation.get("campaign_nonce")
    if not isinstance(nonce, str) or len(nonce) < 32 \
            or any(c not in "0123456789abcdef" for c in nonce) \
            or receipt.get("campaign_nonce") != nonce:
        raise SelectionRefused(
            f"{reservation_path.name} and receipt have no one valid nonce")

    snapshot_path = records_dir / str(receipt.get("plan_snapshot_file") or "")
    if not snapshot_path.is_file():
        raise SelectionRefused(
            f"{receipt_path.name} names absent snapshot {snapshot_path.name}")
    try:
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise SelectionRefused(f"{snapshot_path.name} is unreadable: {error}")
    digest = _json_digest(snapshot)
    if digest != receipt.get("plan_snapshot_sha256") \
            or digest != reservation.get("plan_digest") \
            or reservation.get("plan_snapshot_file") != snapshot_path.name:
        raise SelectionRefused(
            f"{snapshot_path.name} is not the plan reserved before launch")
    if snapshot.get("schema_version") != SWEEP_SNAPSHOT_SCHEMA:
        raise SelectionRefused(
            f"{snapshot_path.name} has stale snapshot schema")
    _verify_production_source_admission(snapshot, snapshot_path.name)
    plan = snapshot.get("plan") or {}
    if plan.get("campaign_kind") != campaign_kind \
            or plan.get("axis") != axis \
            or plan.get("namespace") != namespace \
            or plan.get("campaign_nonce") != nonce \
            or plan.get("execution_kind") != "production":
        raise SelectionRefused(
            f"{snapshot_path.name} is not this production {campaign_kind} plan")
    raw_result_root = plan.get("result_root")
    result_root = Path(str(raw_result_root or ""))
    if not result_root.is_absolute() \
            or str(result_root.resolve()) != raw_result_root \
            or reservation.get("result_root") != raw_result_root \
            or receipt.get("result_root") != raw_result_root:
        raise SelectionRefused(
            f"{snapshot_path.name} does not bind one canonical result root")
    from scripts.phase3_selection_matrix import QWEN_ROOT
    if snapshot.get("qwen_root") != str(QWEN_ROOT) \
            or reservation.get("qwen_root") != str(QWEN_ROOT) \
            or receipt.get("qwen_root") != str(QWEN_ROOT):
        raise SelectionRefused(
            f"{snapshot_path.name} does not bind the canonical absolute Qwen root")
    source_digest = snapshot.get("source_authority_sha256")
    environment_digest = snapshot.get("environment_sha256")
    if not isinstance(source_digest, str) or len(source_digest) != 64 \
            or reservation.get("source_authority_sha256") != source_digest \
            or receipt.get("source_authority_sha256") != source_digest \
            or reservation.get("head_commit") != \
            (snapshot.get("source_authority") or {}).get("head_commit"):
        raise SelectionRefused(
            f"{snapshot_path.name} source authority is not sealed through "
            "reservation and receipt")
    if not isinstance(environment_digest, str) or len(environment_digest) != 64 \
            or reservation.get("environment_sha256") != environment_digest \
            or receipt.get("environment_sha256") != environment_digest:
        raise SelectionRefused(
            f"{snapshot_path.name} execution environment is not sealed through "
            "reservation and receipt")
    input_seals = snapshot.get("input_seals")
    input_seals_sha = _json_digest(input_seals or {})
    if not isinstance(input_seals, dict) or not input_seals \
            or receipt.get("input_seals") != input_seals \
            or receipt.get("input_seals_sha256") != input_seals_sha \
            or reservation.get("input_seals_sha256") != input_seals_sha:
        raise SelectionRefused(
            f"{snapshot_path.name} input seals are not exact through "
            "reservation and receipt")
    try:
        verify_snapshot(snapshot)
        rows = canonical_plan(plan.get("authorities") or {})
    except Exception as error:                         # noqa: BLE001
        raise SelectionRefused(
            f"{snapshot_path.name} authority/source verification failed: "
            f"{error}") from None
    want_rows = _campaign_rows(rows)
    if plan.get("declared_cells") != want_rows \
            or plan.get("executed_cells") != want_rows:
        raise SelectionRefused(
            f"{snapshot_path.name} is not the exact canonical "
            f"{len(want_rows)}-cell plan")
    if receipt.get("declared_cells") != len(want_rows) \
            or receipt.get("expected_cells") != len(want_rows) \
            or receipt.get("cell_count") != len(want_rows) \
            or reservation.get("declared_cells") != len(want_rows) \
            or reservation.get("executed_cells") != len(want_rows):
        raise SelectionRefused(
            f"{receipt_path.name} is not exact-{len(want_rows)} completion")

    bindings, keys = {}, set()
    from scripts.phase3_selection_matrix import _campaign_cell_parts
    from dna_utils.runtime_environment import expected_child_environment
    gpu_assignments = plan.get("dataset_gpu_assignments") or {}
    if set(gpu_assignments) != {row[0] for row in rows}:
        raise SelectionRefused(
            f"{snapshot_path.name} has no exact dataset-to-physical-GPU mapping")
    for row in rows:
        dataset, n, topp, joint, stage, seed = _campaign_cell_parts(row)
        binding = expected_cell_binding(
            dataset, n, namespace=namespace, campaign_nonce=nonce,
            topp=topp, joint=joint, stage=stage, seed=seed,
            result_root=result_root,
            environment_sha256=environment_digest,
            child_environment=expected_child_environment(
                snapshot["environment"], gpu_assignments[dataset]),
            input_authority=input_seals.get(
                f"{dataset}:{'refit' if stage == 'refit' else 'stage1'}"))
        bindings[binding["cell_id"]] = binding
        keys.add(campaign_cell_id(
            dataset, n, topp=topp, joint=joint, stage=stage, seed=seed))
    if plan.get("cell_bindings") != bindings:
        raise SelectionRefused(
            f"{snapshot_path.name} does not seal the canonical cell mapping")
    sealed = receipt.get("cells") or {}
    if set(sealed) != keys:
        raise SelectionRefused(
            f"{receipt_path.name} membership differs from the exact plan; "
            f"missing {sorted(keys - set(sealed))[:3]}, extra "
            f"{sorted(set(sealed) - keys)[:3]}")

    records, names = {}, set()
    for key in sorted(keys):
        entry = sealed[key]
        name = entry.get("record")
        if not isinstance(name, str) or Path(name).name != name or name in names:
            raise SelectionRefused(
                f"{receipt_path.name}: duplicate or unsafe record {name!r}")
        names.add(name)
        path = records_dir / name
        if not path.is_file() or _sha(path) != entry.get("record_sha256"):
            raise SelectionRefused(
                f"{receipt_path.name}: sealed record {name!r} is absent/changed")
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise SelectionRefused(f"{name} is unreadable: {error}")
        binding = bindings[key]
        launch = launch_binding_from_expected(binding, digest)
        for field, actual in (
                ("dataset", record.get("dataset")), ("N", record.get("N")),
                ("seed", record.get("seed")), ("stage", record.get("stage")),
                ("tag", record.get("tag")), ("run_dir", record.get("run_dir")),
                ("recipe", record.get("recipe")),
                ("campaign", record.get("campaign")),
                ("completion", record.get("completion"))):
            if entry.get(field) != actual:
                raise SelectionRefused(
                    f"{name}: receipt.{field} does not equal the record")
        if record.get("namespace") != namespace \
                or record.get("campaign") != launch \
                or record.get("result_root") != raw_result_root \
                or record.get("environment_sha256") != environment_digest \
                or record.get("input_authority") != input_seals.get(
                    f"{record.get('dataset')}:"
                    f"{'refit' if record.get('stage') == 'refit' else 'stage1'}") \
                or (record.get("geometry") or {}).get(
                    "identity_digest") != binding["expected_identity_digest"] \
                or entry.get("identity_digest") != \
                    binding["expected_identity_digest"]:
            raise SelectionRefused(
                f"{name}: record is not its prelaunch canonical cell")
        try:
            run_parent = Path(str(record.get("run_dir"))).resolve().parent
        except (OSError, RuntimeError):
            run_parent = None
        if run_parent != result_root:
            raise SelectionRefused(
                f"{name}: run_dir escapes its sealed result root")
        if binding["expected_tag"] not in Path(str(record.get("run_dir"))).name:
            raise SelectionRefused(
                f"{name}: result directory does not carry its expected tag")
        records[key] = (record, path, entry, binding)
    return {
        "namespace": namespace, "receipt": receipt,
        "receipt_path": receipt_path, "reservation": reservation,
        "snapshot": snapshot, "snapshot_path": snapshot_path,
        "plan_digest": digest, "records": records,
        "record_sha256": {path.name: _sha(path)
                          for _, path, _, _ in records.values()},
    }


def load_unsealed_record_matrix(records_dir: Path) -> dict:
    """Diagnostic parser for unit probes; never a production selection input.

    Production callers must use :func:`load_matrix`, whose authority starts at
    an exact namespace receipt.  Keeping this lower-level parser makes it
    possible to unit-test individual record refusals without fabricating a
    campaign trust root for every one-field counterexample.
    """
    paths = sorted(records_dir.glob("*.json"))
    cells, namespaces, digests, problems = {}, set(), {}, []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            key, value = _check_record(payload, path)
        except (ValueError, SelectionRefused) as error:
            problems.append(str(error))
            continue
        if key in cells:
            problems.append(
                f"{path.name}: {key[0]}/N{key[1]} is recorded twice")
            continue
        cells[key] = {"value": value, "record": path.name,
                      "run_dir": payload.get("run_dir"),
                      "tag": payload.get("tag")}
        namespaces.add(payload.get("namespace"))
        digests[path.name] = _sha(path)
        if payload.get("protocol_sources") != protocol_digests(
                payload.get("dataset")):
            problems.append(
                f"{path.name}: protocol sources differ from this tree")
    if problems:
        raise SelectionRefused("records refused:\n  " + "\n  ".join(problems))
    missing = [f"{dataset}/N{n}" for dataset, n in cell_keys()
               if (dataset, n) not in cells]
    if missing or len(cells) != len(cell_keys()):
        raise SelectionRefused(
            f"{len(cells)} of {len(cell_keys())} cells present; missing {missing}")
    if len(namespaces) != 1:
        raise SelectionRefused(
            f"records come from {len(namespaces)} namespaces; one matrix, one "
            "namespace")
    return {"cells": cells, "namespace": namespaces.pop(),
            "protocol_sources": protocol_digests(),
            "record_sha256": digests}


def load_matrix(records_dir: Path, namespace: str | None = None) -> dict:
    """Recompute D1 from the exact-16 N campaign receipt, never a JSON glob."""
    from scripts.phase3_selection_matrix import (
        N_SELECTION_RECEIPT_SUFFIX, n_selection_cells,
        verify_recipe_authority)

    def canonical(authorities):
        choices = verify_recipe_authority(authorities.get("recipe") or {})
        return n_selection_cells(choices)

    envelope = _campaign_envelope(
        records_dir, namespace=namespace, campaign_kind="n_selection",
        receipt_suffix=N_SELECTION_RECEIPT_SUFFIX,
        canonical_plan=canonical)
    cells = {}
    for record, path, _, _ in envelope["records"].values():
        key, value = _check_record(record, path)
        if key in cells:
            raise SelectionRefused(
                f"{path.name}: {key[0]}/N{key[1]} is recorded twice")
        cells[key] = {"value": value, "record": path.name,
                      "run_dir": record["run_dir"], "tag": record["tag"]}
        if record["protocol_sources"] != protocol_digests(record["dataset"]):
            raise SelectionRefused(
                f"{path.name}: protocol sources differ from this tree")
    if set(cells) != set(cell_keys()):
        raise SelectionRefused(
            f"exact-16 receipt resolved {len(cells)} distinct N cells")
    return {
        "cells": cells, "namespace": envelope["namespace"],
        "protocol_sources": protocol_digests(),
        "record_sha256": envelope["record_sha256"],
        "receipt_file": envelope["receipt_path"].name,
        "receipt_sha256": _sha(envelope["receipt_path"]),
        "plan_snapshot_file": envelope["snapshot_path"].name,
        "plan_snapshot_sha256": envelope["plan_digest"],
        "recipe_authority": envelope["snapshot"]["plan"]["authorities"]["recipe"],
        "records_dir": str(records_dir.resolve()),
    }


def verify_refit_campaign(records_dir: Path, *, namespace: str) -> dict:
    """Reopen the exact-12 refit receipt and every final paper output."""
    from scripts.phase3_selection_matrix import (
        DATASETS, REFIT_RECEIPT_SUFFIX, REFIT_SEEDS, refit_cells,
        verify_recipe_authority)

    def canonical(authorities):
        recipe = verify_recipe_authority(authorities.get("recipe") or {})
        selected = authorities.get("selected_n") or {}
        path = Path(str(selected.get("path") or ""))
        if not path.is_file() or _sha(path) != selected.get("sha256"):
            raise SelectionRefused(
                "refit selected-N authority is absent or changed")
        reopened = verify_selection_artifact(path)
        if reopened != selected.get("selected_n"):
            raise SelectionRefused(
                "refit snapshot copied a different selected-N decision")
        return refit_cells(reopened, recipe)

    envelope = _campaign_envelope(
        records_dir, namespace=namespace, campaign_kind="refit",
        receipt_suffix=REFIT_RECEIPT_SUFFIX, canonical_plan=canonical)
    seen = set()
    required_completion = {
        "final_checkpoint_sha256", "checkpoint_runtime_sha256",
        "run_identity_sha256", "args_txt_sha256",
        "phase3_campaign_evidence_sha256", "log_csv_sha256", "npz_sha256",
        "extraction_manifest_sha256", "extraction_complete_sha256",
        "evaluation_sha256", "bio_evaluation_sha256",
        "cell_result_sha256", "pairwise_nmi_sha256",
        "analysis_complete_sha256", "map", "map_at_R", "map_R_cutoff",
        "bio_map", "bio_map_at_R", "mean_off_diag_nmi",
        "analysis_metrics", "analysis_protocol", "required_splits",
        "checkpoint_exact_load",
    }
    admitted = {}
    for record, path, _, binding in envelope["records"].values():
        dataset, seed = record.get("dataset"), record.get("seed")
        key = (dataset, seed)
        if dataset not in DATASETS or seed not in REFIT_SEEDS or key in seen:
            raise SelectionRefused(
                f"{path.name}: duplicate/noncanonical refit cell {key}")
        seen.add(key)
        if record.get("stage") != "refit" \
                or record.get("selection_mode") != "refit" \
                or record.get("val_split_ratio") != 0.0 \
                or record.get("N") != binding["N"]:
            raise SelectionRefused(
                f"{path.name}: record is not the canonical scratch refit")
        if not required_completion <= set(record.get("completion") or {}):
            raise SelectionRefused(
                f"{path.name}: refit completion omits final output identity")
        _assert_claimed_coordinate(record, path)
        _reverify_run(record, path)
        if record.get("protocol_sources") != protocol_digests(dataset):
            raise SelectionRefused(
                f"{path.name}: protocol sources differ from this tree")
        admitted[key] = record
    expected = {(dataset, seed) for dataset in DATASETS for seed in REFIT_SEEDS}
    if seen != expected:
        raise SelectionRefused(
            f"exact-12 refit covers {len(seen)} dataset/seed cells")
    datasets = {}
    metric_fields = ("map_at_R", "bio_map_at_R", "mean_off_diag_nmi")
    for dataset in sorted(DATASETS):
        cells = [admitted[(dataset, seed)] for seed in sorted(REFIT_SEEDS)]
        if [cell["seed"] for cell in cells] != sorted(REFIT_SEEDS):
            raise SelectionRefused(
                f"{dataset}: refit aggregate does not contain three unique seeds")
        checkpoint_by_seed = {
            str(cell["seed"]): cell["completion"]["final_checkpoint_sha256"]
            for cell in cells
        }
        if len(set(checkpoint_by_seed.values())) != len(REFIT_SEEDS):
            raise SelectionRefused(
                f"{dataset}: three refit seeds do not have three distinct "
                "terminal checkpoint SHA-256 values")
        metrics = {}
        for field in metric_fields:
            values = [float(cell["completion"][field]) for cell in cells]
            if any(not math.isfinite(value)
                   or not 0.0 <= value <= 1.0 for value in values):
                raise SelectionRefused(
                    f"{dataset}: non-finite/out-of-range aggregate {field}")
            metrics[field] = {
                "values_by_seed": {
                    str(seed): value for seed, value in zip(
                        sorted(REFIT_SEEDS), values)},
                "mean": statistics.mean(values),
                "sample_sd": statistics.stdev(values),
            }
        datasets[dataset] = {
            "N": cells[0]["N"], "recipe": cells[0]["recipe"],
            "seeds": sorted(REFIT_SEEDS), "metrics": metrics,
            "record_sha256": {
                str(cell["seed"]): envelope["record_sha256"][
                    f"{cell['tag']}.json"] for cell in cells},
            "analysis_complete_sha256": {
                str(cell["seed"]): cell["completion"][
                    "analysis_complete_sha256"] for cell in cells},
            "final_checkpoint_sha256": checkpoint_by_seed,
        }
    return {
        "artifact_kind": "phase3_refit_aggregate",
        "schema_version": REFIT_AGGREGATE_SCHEMA,
        "namespace": namespace,
        "receipt_file": envelope["receipt_path"].name,
        "receipt_sha256": _sha(envelope["receipt_path"]),
        "plan_snapshot_file": envelope["snapshot_path"].name,
        "plan_snapshot_sha256": envelope["plan_digest"],
        "record_sha256": envelope["record_sha256"],
        "source_authority_sha256": envelope["snapshot"].get(
            "source_authority_sha256"),
        "environment_sha256": envelope["snapshot"].get(
            "environment_sha256"),
        "input_seals_sha256": envelope["receipt"].get("input_seals_sha256"),
        "reduction": {
            "seed_count_per_dataset": 3,
            "location": "arithmetic_mean",
            "dispersion": "sample_standard_deviation_ddof_1",
        },
        "datasets": datasets,
    }


def verify_refit_aggregate(path: Path, *, records_dir: Path,
                           namespace: str) -> dict:
    """Rebuild, rather than trust, one published paper-number aggregate."""
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value}")))
    except (OSError, ValueError, TypeError) as error:
        raise SelectionRefused(f"{path}: unreadable aggregate: {error}") from None
    rebuilt = verify_refit_campaign(records_dir, namespace=namespace)
    if payload != rebuilt:
        raise SelectionRefused(
            f"{path}: aggregate differs from exact-12 reopened evidence")
    return payload


#: The recipe reduction, declared BEFORE the cells run so the rule cannot be
#: picked after the numbers are seen.
#:
#:   metric      train-only validation raw base-Hamming mAP@R at the cell's own
#:               terminal epoch -- the same number D1 uses for N. Codon decoding
#:               is NOT in it: the held-out decoder reads the official query
#:               split, so using it would make the choice test-informed.
#:   winner      argmax of that metric within a dataset
#:   tie         the INCUMBENT coordinate first, then the smaller one. Ties go
#:               to not-moving, so a coin-flip cannot rewrite the recipe.
#:   fail        any missing, duplicated or non-finite cell refuses the WHOLE
#:               dataset. A recipe chosen from whichever cells finished is the
#:               partial-matrix defect under another name.
RECIPE_REDUCTION = {
    "metric": "eval_mAP_at_R",
    "distance": "base_hamming",
    "split": "train_only_validation",
    "tie_break": "incumbent_first_then_smaller",
    "fail_rule": "any_missing_or_nonfinite_refuses_the_dataset",
}

# The fixed-point protocol is deliberately data-independent.  In particular,
# changing this bound after looking at a trajectory cannot turn a non-converged
# search into a winner: the literal value is copied into every stage plan and
# the final authority, and every consumer replays the same transitions.
RECIPE_STABILITY_SCHEMA = 1
RECIPE_STABILITY_ARTIFACT_SCHEMA = 1
RECIPE_STAGE_PLAN_SCHEMA = 1
DEFAULT_MAX_UPDATE_ROUNDS = 3
RECIPE_STAGE_PHASES = {
    "bootstrap_topp": "topp",
    "bootstrap_joint": "joint",
    "bootstrap_n": "n",
    "confirm_topp": "topp",
    "confirm_joint": "joint",
    "update_n": "n",
}
PINNED_CIFAR_TOPP_POLICY = {
    "dataset": "cifar10",
    "topp": ["0.6", "0.95"],
    "source": "structural_empty_slot_policy",
    "selection_metric": None,
    "fake_candidate_record": False,
}


def _normalise_recipe_state(raw: dict) -> dict:
    """Return the one canonical JSON representation of a recipe state."""
    from scripts.phase3_selection_matrix import (
        DATASETS, JOINT_GRID, JOINT_INCUMBENT, TOPP_GRID, TOPP_PINNED)

    if not isinstance(raw, dict) or set(raw) != set(DATASETS):
        raise SelectionRefused(
            "recipe state must name exactly the four canonical datasets")
    out = {}
    for dataset in sorted(DATASETS):
        entry = raw.get(dataset)
        if not isinstance(entry, dict) or set(entry) != {"N", "topp", "joint"}:
            raise SelectionRefused(
                f"{dataset}: state must contain exactly N/topp/joint")
        n = entry.get("N")
        if isinstance(n, bool) or n not in CANDIDATE_N:
            raise SelectionRefused(f"{dataset}: N={n!r} is outside the grid")
        topp = entry.get("topp")
        if not isinstance(topp, (list, tuple)) or len(topp) != 2:
            raise SelectionRefused(f"{dataset}: malformed top-p window {topp!r}")
        topp = tuple(str(x) for x in topp)
        if topp not in tuple(tuple(x) for x in TOPP_GRID):
            raise SelectionRefused(
                f"{dataset}: top-p {topp!r} is outside the declared grid")
        if dataset in TOPP_PINNED and topp != tuple(TOPP_PINNED[dataset]):
            raise SelectionRefused(
                f"{dataset}: pinned top-p policy was changed to {topp!r}")
        joint = str(entry.get("joint"))
        if joint not in set(JOINT_GRID) | {JOINT_INCUMBENT}:
            raise SelectionRefused(
                f"{dataset}: JD={joint!r} is outside the declared policy")
        out[dataset] = {"N": int(n), "topp": list(topp), "joint": joint}
    return out


def initial_recipe_state() -> dict:
    """State before the first P sweep: incumbent N/J and explicit CIFAR pin."""
    from scripts.phase3_selection_matrix import (
        DATASETS, INCUMBENT_N, JOINT_INCUMBENT, TOPP_INCUMBENT, TOPP_PINNED)

    return _normalise_recipe_state({
        dataset: {
            "N": INCUMBENT_N[dataset],
            "topp": list(TOPP_PINNED.get(dataset, TOPP_INCUMBENT)),
            "joint": JOINT_INCUMBENT,
        }
        for dataset in DATASETS
    })


def stability_topp_cells(state: dict) -> list:
    """Exact 3x4 P grid at each dataset's *current* N and JD."""
    from scripts.phase3_selection_matrix import TOPP_GRID, TOPP_SWEEP_DATASETS

    state = _normalise_recipe_state(state)
    return [
        (dataset, state[dataset]["N"], tuple(topp),
         state[dataset]["joint"])
        for dataset in TOPP_SWEEP_DATASETS for topp in TOPP_GRID
    ]


def stability_joint_cells(state: dict) -> list:
    """Exact 4x5 positive JD grid at current per-dataset N and P."""
    from scripts.phase3_selection_matrix import DATASETS, JOINT_GRID

    state = _normalise_recipe_state(state)
    return [
        (dataset, state[dataset]["N"], tuple(state[dataset]["topp"]), jd)
        for dataset in sorted(DATASETS) for jd in JOINT_GRID
    ]


def _normalise_axis_decisions(raw: dict, *, axis: str) -> dict:
    """Validate reducer winners before applying one state transition."""
    from scripts.phase3_selection_matrix import (
        DATASETS, JOINT_GRID, TOPP_GRID, TOPP_SWEEP_DATASETS)

    expected = (set(TOPP_SWEEP_DATASETS) if axis == "topp" else set(DATASETS))
    if not isinstance(raw, dict) or set(raw) != expected:
        raise SelectionRefused(
            f"{axis} decisions must cover exactly {sorted(expected)}; got "
            f"{sorted(raw) if isinstance(raw, dict) else type(raw).__name__}")
    out = {}
    for dataset in sorted(expected):
        value = raw[dataset]
        if axis == "topp":
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise SelectionRefused(
                    f"{dataset}: malformed top-p decision {value!r}")
            value = tuple(str(x) for x in value)
            if value not in tuple(tuple(x) for x in TOPP_GRID):
                raise SelectionRefused(
                    f"{dataset}: top-p decision {value!r} is off-grid")
            out[dataset] = list(value)
        elif axis == "joint":
            value = str(value)
            # JD=0 is provenance for the first P sweep only.  It is never a
            # member of the joint candidate set, per the latest authority.
            if value not in JOINT_GRID:
                raise SelectionRefused(
                    f"{dataset}: JD decision {value!r} is not on the positive grid")
            out[dataset] = value
        elif axis == "n":
            if isinstance(value, bool) or value not in CANDIDATE_N:
                raise SelectionRefused(
                    f"{dataset}: N decision {value!r} is off-grid")
            out[dataset] = int(value)
        else:  # pragma: no cover - internal programming error
            raise AssertionError(axis)
    return out


def _apply_decisions(state: dict, decisions: dict, *, axis: str) -> dict:
    state = _normalise_recipe_state(state)
    decisions = _normalise_axis_decisions(decisions, axis=axis)
    out = json.loads(json.dumps(state))
    field = {"topp": "topp", "joint": "joint", "n": "N"}[axis]
    for dataset, value in decisions.items():
        out[dataset][field] = value
    return _normalise_recipe_state(out)


def _state_key(state: dict) -> str:
    return json.dumps(_normalise_recipe_state(state), sort_keys=True,
                      separators=(",", ":"))


def run_recipe_stability(bootstrap: dict, rounds: list, *,
                         max_update_rounds: int =
                         DEFAULT_MAX_UPDATE_ROUNDS) -> dict:
    """Replay P -> JD -> N -> P/J confirmations and require a fixed point.

    ``max_update_rounds`` counts drift rounds, not the final unchanged
    confirmation.  Thus three updates may be followed by one unchanged proof;
    a fourth update is refused and never returns the last tuple as a winner.
    """
    if isinstance(max_update_rounds, bool) \
            or not isinstance(max_update_rounds, int) \
            or max_update_rounds < 0:
        raise SelectionRefused("max_update_rounds must be a non-negative integer")
    if not isinstance(bootstrap, dict) or set(bootstrap) != {"topp", "joint", "n"}:
        raise SelectionRefused("bootstrap must contain exactly topp/joint/n")
    if not isinstance(rounds, list):
        raise SelectionRefused("confirmation rounds must be a list")

    state = initial_recipe_state()
    state = _apply_decisions(state, bootstrap["topp"], axis="topp")
    state = _apply_decisions(state, bootstrap["joint"], axis="joint")
    state = _apply_decisions(state, bootstrap["n"], axis="n")
    history = [{"event": "bootstrap", "state": state}]
    seen = {_state_key(state): "bootstrap"}
    updates = 0

    for index, evidence in enumerate(rounds, 1):
        if not isinstance(evidence, dict) \
                or not {"topp", "joint"} <= set(evidence) \
                or set(evidence) - {"topp", "joint", "n"}:
            raise SelectionRefused(
                f"round {index}: expected topp/joint and optional n only")
        before = state
        after_p = _apply_decisions(before, evidence["topp"], axis="topp")
        after_j = _apply_decisions(after_p, evidence["joint"], axis="joint")
        recipe_changed = any(
            after_j[d][axis] != before[d][axis]
            for d in after_j for axis in ("topp", "joint"))
        if not recipe_changed:
            if evidence.get("n") is not None:
                raise SelectionRefused(
                    f"round {index}: unchanged P/J must not rerun or replace N")
            state = after_j
            history.append({
                "event": "confirmed", "round": index,
                "recipe_changed": False, "n_rerun": False, "state": state,
            })
            if index != len(rounds):
                raise SelectionRefused(
                    f"round {index}: evidence continues after the fixed point")
            return {
                "schema_version": RECIPE_STABILITY_SCHEMA,
                "confirmed": True,
                "max_update_rounds": max_update_rounds,
                "update_rounds": updates,
                "history": history,
                "final_state": state,
                "pinned_policy": PINNED_CIFAR_TOPP_POLICY,
            }

        updates += 1
        if updates > max_update_rounds:
            raise SelectionRefused(
                f"round {index}: recipe drift exceeds max_update_rounds="
                f"{max_update_rounds}; no winner exists")
        if evidence.get("n") is None:
            raise SelectionRefused(
                f"round {index}: P/J changed, so the exact-16 N matrix must rerun")
        state = _apply_decisions(after_j, evidence["n"], axis="n")
        key = _state_key(state)
        if key in seen:
            raise SelectionRefused(
                f"round {index}: state cycles back to {seen[key]}; no winner exists")
        seen[key] = f"round {index}"
        history.append({
            "event": "updated", "round": index,
            "recipe_changed": True, "n_rerun": True, "state": state,
        })

    raise SelectionRefused(
        "stability evidence ended before an unchanged P/J confirmation; "
        "no winner exists")


def choose_recipe(cells: dict, *, axis: str, grid: list, incumbent,
                  datasets=None) -> dict:
    """Pick one coordinate per dataset on `RECIPE_REDUCTION`.

    `cells` maps (dataset, coordinate) -> value. `grid` is the exact set of
    coordinates every dataset must have run, so a dataset that is one cell
    short is refused rather than reduced.
    """
    seen = sorted({d for d, _ in cells})
    if datasets is None:
        raise SelectionRefused(
            "the declared dataset set is required: reducing over 'whatever "
            "turned up' accepted a Flickr-only matrix as a complete sweep")
    datasets = sorted(datasets)
    if seen != datasets:
        raise SelectionRefused(
            f"the sweep declares {datasets} but the records cover {seen}; a "
            f"recipe chosen from the datasets that happened to finish is a "
            f"partial matrix")
    chosen = {}
    for dataset in datasets:
        missing = [c for c in grid if (dataset, c) not in cells]
        if missing:
            raise SelectionRefused(
                f"{dataset}: {len(missing)} of {len(grid)} {axis} cells are "
                f"absent {missing}. A recipe chosen from whichever cells "
                f"finished is a partial matrix.")
        extra = sorted(c for d, c in cells if d == dataset and c not in grid)
        if extra:
            raise SelectionRefused(
                f"{dataset}: {extra} are not in the declared {axis} grid "
                f"{grid}")
        values = {}
        for c in grid:
            v = cells[(dataset, c)]
            if not isinstance(v, (int, float)) or isinstance(v, bool) \
                    or v != v or not 0.0 <= float(v) <= 1.0:
                raise SelectionRefused(
                    f"{dataset}/{c}: value {v!r} is not a finite proportion")
            values[c] = float(v)
        best = max(values.values())
        tied = [c for c in grid if values[c] == best]
        # Incumbent first, then the smaller coordinate. Both are positional in
        # `grid`, so the rule is part of the ordering rather than a branch.
        if incumbent in tied:
            winner = incumbent
        else:
            winner = sorted(tied, key=lambda c: grid.index(c))[0]
        chosen[dataset] = {
            "selected": list(winner) if isinstance(winner, tuple) else winner,
            "selection_value": values[winner],
            "tied_with": [list(c) if isinstance(c, tuple) else c for c in tied],
            "tie_broken": len(tied) > 1,
            "all_candidates": {
                str(c): values[c] for c in grid},
            # The incumbent is only ON the grid for the top-p sweep. The
            # lambda grid is the five positive values, and its zero control
            # comes from the top-p stage, so `values[incumbent]` raised
            # KeyError('0.0') and the whole lambda reduction was unrunnable.
            "delta_vs_incumbent": (
                round(values[winner] - values[incumbent], 6)
                if incumbent in values else None),
            "incumbent_on_grid": incumbent in values,
        }
    return chosen


def choose(cells: dict) -> dict:
    """argmax on the metric; ties to the smallest N."""
    chosen = {}
    for dataset in sorted(DATASETS):
        # Sorting by (-value, N) makes the tie-break part of the ordering
        # rather than a branch someone can forget.
        ranked = sorted(
            ((cells[(dataset, n)]["value"], n) for n in CANDIDATE_N),
            key=lambda pair: (-pair[0], pair[1]))
        best_value, best_n = ranked[0]
        tied = [n for value, n in ranked if value == best_value]
        chosen[dataset] = {
            "selected_N": best_n,
            "selection_value": best_value,
            "tied_with": sorted(tied),
            "tie_broken_by_smallest_N": len(tied) > 1,
            "all_candidates": {str(n): cells[(dataset, n)]["value"]
                               for n in CANDIDATE_N},
        }
    return chosen


def _recipe_matrix(records_dir: Path, *, axis: str,
                   namespace: str | None = None, at_topp=None,
                   expected_plan=None, expected_authorities=None) -> dict:
    """The same per-record authentication, keyed by the swept coordinate."""
    from scripts.phase3_selection_matrix import (
        JOINT_GRID, JOINT_INCUMBENT, TOPP_GRID, TOPP_INCUMBENT)

    grid = ([tuple(p) for p in TOPP_GRID] if axis == "topp"
            else list(JOINT_GRID))
    incumbent = (tuple(TOPP_INCUMBENT) if axis == "topp" else JOINT_INCUMBENT)

    cells, digests, problems, namespaces, snapshots = {}, {}, [], set(), set()
    for path in sorted(records_dir.glob("*.json")):
        if "_snapshot" in path.name or "_sweep_complete" in path.name \
                or path.name == DEFAULT_OUT.name:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not payload.get("is_candidate_cell"):
            continue
        # A recipe sweep and the N matrix share one directory, so the sweep
        # must be selected by its NAMESPACE rather than by "everything that
        # looks like a candidate". Reading all of them made the sixteen
        # N-selection records -- written under an older identity schema -- into
        # sixteen fatal problems, so the reducer could not run at all.
        if namespace is not None and payload.get("namespace") != namespace:
            continue
        recipe = payload.get("recipe") or {}
        coord = ((str(recipe.get("routing_adaptive_topp_min")),
                  str(recipe.get("routing_adaptive_topp_max")))
                 if axis == "topp" else str(recipe.get("lambda_codon_joint")))
        try:
            # Everything except the (dataset, N) membership test, which is
            # about the N matrix rather than a recipe sweep.
            _reverify_run(payload, path)
            _assert_claimed_coordinate(payload, path)
        except SelectionRefused as error:
            problems.append(str(error))
            continue
        key = (payload["dataset"], coord)
        if key in cells:
            problems.append(f"{path.name}: {key} is recorded twice")
            continue
        cells[key] = float(payload["selection"]["selection_value"])
        digests[path.name] = _sha(path)
        namespaces.add(payload.get("namespace"))
        snapshots.add(payload.get("plan_snapshot_sha256"))
    if problems:
        raise SelectionRefused("records refused:\n  " + "\n  ".join(problems))
    # ---- the sweep's own receipt is the authority on what ran -------------
    receipt_path = records_dir / f"{namespace}_sweep_complete.json"
    if not receipt_path.is_file():
        raise SelectionRefused(
            f"{receipt_path.name} does not exist; a reduction over records "
            f"nobody sealed cannot say the sweep finished")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    from scripts.phase3_selection_matrix import (
        CAMPAIGN_RESERVATION_SCHEMA, CAMPAIGN_RESERVATION_SUFFIX,
        SWEEP_RECEIPT_SCHEMA, SWEEP_SNAPSHOT_SCHEMA,
        _json_digest, expected_cell_binding, launch_binding_from_expected)

    if receipt.get("schema_version") != SWEEP_RECEIPT_SCHEMA:
        raise SelectionRefused(
            f"{receipt_path.name}: schema_version "
            f"{receipt.get('schema_version')!r}, not "
            f"{SWEEP_RECEIPT_SCHEMA}")
    campaign_kind = f"recipe_{axis}"
    if receipt.get("axis") != axis \
            or receipt.get("campaign_kind") != campaign_kind \
            or receipt.get("namespace") != namespace:
        raise SelectionRefused(
            f"{receipt_path.name} seals axis {receipt.get('axis')!r} in "
            f"namespace {receipt.get('namespace')!r}, not {axis!r}/{namespace!r}")
    if receipt.get("cell_count") != receipt.get("expected_cells"):
        raise SelectionRefused(
            f"{receipt_path.name}: {receipt.get('cell_count')} of "
            f"{receipt.get('expected_cells')} cells")

    # The reservation predates every trainer and is created with O_EXCL.  It is
    # the anchor a later self-consistent rewrite of snapshot/records/receipt
    # cannot silently replace.
    reservation_path = records_dir / f"{namespace}{CAMPAIGN_RESERVATION_SUFFIX}"
    if receipt.get("campaign_reservation_file") != reservation_path.name:
        raise SelectionRefused(
            f"{receipt_path.name} names reservation "
            f"{receipt.get('campaign_reservation_file')!r}, expected "
            f"{reservation_path.name!r}")
    if not reservation_path.is_file():
        raise SelectionRefused(
            f"{reservation_path.name} is absent; the namespace was never "
            "atomically reserved before launch")
    if _sha(reservation_path) != receipt.get("campaign_reservation_sha256"):
        raise SelectionRefused(
            f"{reservation_path.name} changed after the receipt was written")
    try:
        reservation = json.loads(reservation_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise SelectionRefused(
            f"{reservation_path.name} is unreadable: {error}")
    if reservation.get("schema_version") != CAMPAIGN_RESERVATION_SCHEMA \
            or reservation.get("namespace") != namespace \
            or reservation.get("campaign_kind") != campaign_kind:
        raise SelectionRefused(
            f"{reservation_path.name} is not this namespace's reservation")
    if not isinstance(reservation.get("owner_pid"), int) \
            or isinstance(reservation.get("owner_pid"), bool) \
            or not isinstance(reservation.get("owner_boot_id"), str) \
            or not reservation.get("owner_boot_id"):
        raise SelectionRefused(
            f"{reservation_path.name} has no owner PID/boot-id")
    nonce = reservation.get("campaign_nonce")
    if not isinstance(nonce, str) or len(nonce) < 32 \
            or any(c not in "0123456789abcdef" for c in nonce):
        raise SelectionRefused(
            f"{reservation_path.name} has no valid campaign nonce")
    if receipt.get("campaign_nonce") != nonce:
        raise SelectionRefused(
            f"{receipt_path.name} and reservation carry different campaigns")

    # ---- the snapshot it names, reopened and re-hashed --------------------
    snap_path = records_dir / str(receipt.get("plan_snapshot_file") or "")
    if not snap_path.is_file():
        raise SelectionRefused(
            f"the receipt names snapshot {snap_path.name}, which is not here")
    snapshot = json.loads(snap_path.read_text(encoding="utf-8"))
    snap_digest = hashlib.sha256(
        json.dumps(snapshot, sort_keys=True,
                   separators=(",", ":")).encode()).hexdigest()
    if snap_digest != receipt.get("plan_snapshot_sha256"):
        raise SelectionRefused(
            f"{snap_path.name} does not hash to the digest the receipt seals")
    if snapshot.get("schema_version") != SWEEP_SNAPSHOT_SCHEMA:
        raise SelectionRefused(
            f"{snap_path.name}: schema_version "
            f"{snapshot.get('schema_version')!r}, not {SWEEP_SNAPSHOT_SCHEMA}")
    _verify_production_source_admission(snapshot, snap_path.name)
    if reservation.get("plan_digest") != snap_digest \
            or reservation.get("plan_snapshot_file") != snap_path.name:
        raise SelectionRefused(
            f"{snap_path.name} is not the plan atomically reserved before "
            "training")
    plan = snapshot.get("plan") or {}
    if plan.get("axis") != axis \
            or plan.get("campaign_kind") != campaign_kind \
            or plan.get("namespace") != namespace:
        raise SelectionRefused(
            f"{snap_path.name} describes plan {plan.get('axis')!r}/"
            f"{plan.get('namespace')!r}, not {axis!r}/{namespace!r}")
    if plan.get("campaign_nonce") != nonce:
        raise SelectionRefused(
            f"{snap_path.name} belongs to a different campaign nonce")
    if expected_authorities is not None \
            and plan.get("authorities") != expected_authorities:
        raise SelectionRefused(
            f"{snap_path.name} was not launched from the expected stability "
            "stage authority")
    if plan.get("execution_kind") != "production":
        raise SelectionRefused(
            f"{snap_path.name} is {plan.get('execution_kind')!r}, not a "
            "production candidate sweep")
    raw_result_root = plan.get("result_root")
    result_root = Path(str(raw_result_root or ""))
    source_digest = snapshot.get("source_authority_sha256")
    environment_digest = snapshot.get("environment_sha256")
    if not result_root.is_absolute() \
            or str(result_root.resolve()) != raw_result_root \
            or reservation.get("result_root") != raw_result_root \
            or receipt.get("result_root") != raw_result_root:
        raise SelectionRefused(
            f"{snap_path.name} does not bind one canonical result root")
    from scripts.phase3_selection_matrix import QWEN_ROOT
    if snapshot.get("qwen_root") != str(QWEN_ROOT) \
            or reservation.get("qwen_root") != str(QWEN_ROOT) \
            or receipt.get("qwen_root") != str(QWEN_ROOT):
        raise SelectionRefused(
            f"{snap_path.name} does not bind the canonical absolute Qwen root")
    if not isinstance(source_digest, str) or len(source_digest) != 64 \
            or reservation.get("source_authority_sha256") != source_digest \
            or receipt.get("source_authority_sha256") != source_digest \
            or reservation.get("head_commit") != \
            (snapshot.get("source_authority") or {}).get("head_commit"):
        raise SelectionRefused(
            f"{snap_path.name} source authority is not sealed through "
            "reservation and receipt")
    if not isinstance(environment_digest, str) or len(environment_digest) != 64 \
            or reservation.get("environment_sha256") != environment_digest \
            or receipt.get("environment_sha256") != environment_digest:
        raise SelectionRefused(
            f"{snap_path.name} execution environment is not sealed through "
            "reservation and receipt")
    input_seals = snapshot.get("input_seals")
    input_seals_sha = _json_digest(input_seals or {})
    if not isinstance(input_seals, dict) or not input_seals \
            or receipt.get("input_seals") != input_seals \
            or receipt.get("input_seals_sha256") != input_seals_sha \
            or reservation.get("input_seals_sha256") != input_seals_sha:
        raise SelectionRefused(
            f"{snap_path.name} input seals are not exact through reservation "
            "and receipt")
    if not plan.get("declared_cells"):
        raise SelectionRefused(
            f"{snap_path.name} carries no declared plan; it was written before "
            f"the snapshot named what it was running")

    # The snapshot does NOT get to say what a complete sweep is. A Flickr-only
    # snapshot calling itself complete was accepted as one, and a snapshot whose
    # four coordinates had all been rewritten to 0.3/0.7 was too. Both are
    # compared against the plan this source declares for the axis.
    from scripts.phase3_selection_matrix import canonical_plan, verify_snapshot
    try:
        verify_snapshot(snapshot)
    except Exception as error:                         # noqa: BLE001
        raise SelectionRefused(
            f"{snap_path.name} source verification failed: {error}") from None
    declared_plan = (canonical_plan(axis, at_topp=at_topp)
                     if expected_plan is None else list(expected_plan))
    want = [{"dataset": ds, "N": n,
             "topp": list(topp) if topp is not None else None, "joint": jd,
             "stage": "select", "seed": SEED}
            for ds, n, topp, jd in declared_plan]
    if plan["declared_cells"] != want:
        raise SelectionRefused(
            f"{snap_path.name} declares a plan this tree does not: it names "
            f"{len(plan['declared_cells'])} cells over "
            f"{sorted({c['dataset'] for c in plan['declared_cells']})}, the "
            f"source declares {len(want)} over "
            f"{sorted({c['dataset'] for c in want})}")
    executed = plan.get("executed_cells")
    if executed != want:
        raise SelectionRefused(
            f"{snap_path.name} executed {len(executed or [])} of its "
            f"{len(want)} declared cells; a reduction needs the whole sweep")
    if receipt["expected_cells"] != len(want):
        raise SelectionRefused(
            f"{receipt_path.name} expects {receipt['expected_cells']} cells, "
            f"the declared plan has {len(want)}")
    if reservation.get("declared_cells") != len(want) \
            or reservation.get("executed_cells") != len(want):
        raise SelectionRefused(
            f"{reservation_path.name} reserved "
            f"{reservation.get('executed_cells')}/"
            f"{reservation.get('declared_cells')} cells, expected "
            f"{len(want)}/{len(want)}")
    declared = sorted({c["dataset"] for c in want})

    # Exact cell -> tag -> RunIdentity mapping, computed and sealed BEFORE a
    # child starts.  Merely checking the coordinate/tag SET is invariant under
    # a cyclic relabel of all four bundles; this mapping plus the trainer-owned
    # checkpoint witness is not.
    want_bindings = {}
    expected_tag = {}
    from dna_utils.runtime_environment import expected_child_environment
    gpu_assignments = plan.get("dataset_gpu_assignments") or {}
    if set(gpu_assignments) != set(declared):
        raise SelectionRefused(
            f"{snap_path.name} has no exact dataset-to-physical-GPU mapping")
    for cell in want:
        coord = tuple(cell["topp"]) if cell["topp"] is not None else None
        binding = expected_cell_binding(
            cell["dataset"], cell["N"], namespace=namespace,
            campaign_nonce=nonce, topp=coord, joint=cell["joint"], epochs=None,
            result_root=result_root,
            environment_sha256=environment_digest,
            child_environment=expected_child_environment(
                snapshot["environment"], gpu_assignments[cell["dataset"]]),
            input_authority=input_seals.get(f"{cell['dataset']}:stage1"))
        want_bindings[binding["cell_id"]] = binding
        expected_tag[binding["expected_tag"]] = (
            cell["dataset"], coord, cell["joint"], binding)
    if plan.get("cell_bindings") != want_bindings:
        raise SelectionRefused(
            f"{snap_path.name} does not seal this tree's canonical "
            "cell-to-tag/RunIdentity mapping")

    # ---- every record the receipt sealed, by name and by bytes ------------
    sealed = receipt.get("cells") or {}
    if len(sealed) != receipt["expected_cells"]:
        raise SelectionRefused(
            f"{receipt_path.name} seals {len(sealed)} cells, expected "
            f"{receipt['expected_cells']}")
    for key, entry in sorted(sealed.items()):
        name = entry.get("record")
        path = records_dir / str(name)
        if not path.is_file():
            raise SelectionRefused(f"the receipt seals {name}, which is gone")
        if _sha(path) != entry.get("record_sha256"):
            raise SelectionRefused(
                f"{name} does not hash to the digest the receipt seals; it "
                f"changed after the sweep finished")
        if name not in digests:
            raise SelectionRefused(
                f"the receipt seals {name}, which this reduction did not admit")
        # The receipt's own description of the cell has to be the record's.
        # Rewriting an entry's tag, run_dir, identity or recipe to foreign
        # values passed while the filename and its SHA still matched.
        rec = json.loads(path.read_text(encoding="utf-8"))
        for field, actual in (("tag", rec.get("tag")),
                              ("run_dir", rec.get("run_dir")),
                              ("recipe", rec.get("recipe")),
                              ("campaign", rec.get("campaign"))):
            if entry.get(field) != actual:
                raise SelectionRefused(
                    f"the receipt says {name} has {field}={entry.get(field)!r}, "
                    f"the record says {actual!r}")
        if key != (rec.get("campaign") or {}).get("cell_id"):
            raise SelectionRefused(
                f"receipt key {key!r} is not {name}'s sealed campaign cell id")
        receipt_completion = entry.get("completion") or {}
        record_completion = rec.get("completion") or {}
        for field in (
                "final_checkpoint_sha256", "log_csv_sha256",
                "args_txt_sha256", "checkpoint_runtime_sha256",
                "phase3_campaign_evidence_sha256"):
            if receipt_completion.get(field) != record_completion.get(field):
                raise SelectionRefused(
                    f"the receipt says {name} has completion.{field}="
                    f"{receipt_completion.get(field)!r}, the record says "
                    f"{record_completion.get(field)!r}")
        sealed_id = entry.get("identity_digest")
        if sealed_id != (rec.get("geometry") or {}).get("identity_digest"):
            raise SelectionRefused(
                f"the receipt seals identity {str(sealed_id)[:12]}... for "
                f"{name}, the record says "
                f"{str((rec.get('geometry') or {}).get('identity_digest'))[:12]}...")
    extra = sorted(set(digests) - {e.get("record") for e in sealed.values()})
    if extra:
        raise SelectionRefused(
            f"{extra} are being reduced but the receipt does not seal them")

    if not cells:
        raise SelectionRefused(
            f"no candidate {axis} records in {records_dir}"
            + (f" for namespace {namespace!r}" if namespace else ""))
    if len(namespaces) > 1:
        raise SelectionRefused(
            f"records come from {len(namespaces)} namespaces "
            f"{sorted(namespaces)}; one sweep, one namespace")
    # Every cell must have run against the SAME plan. Nine `--only` processes
    # each took their own snapshot, so nine cells carried up to nine different
    # digests and "these came from one plan" was not a statement anyone could
    # make about them.
    if len(snapshots) != 1 or None in snapshots:
        raise SelectionRefused(
            f"the cells carry {len(snapshots)} different plan snapshots "
            f"{sorted(str(x)[:12] for x in snapshots)}; a sweep reduced across "
            f"several plans is not one experiment")
    # Each admitted record's tag must be the tag its CLAIMED coordinate
    # implies, and the twelve declared tags must each be claimed exactly once.
    seen_tags = {}
    for name in sorted(digests):
        rec = json.loads((records_dir / name).read_text(encoding="utf-8"))
        tag = rec.get("tag")
        if tag not in expected_tag:
            raise SelectionRefused(
                f"{name} carries tag {tag!r}, which the declared plan does not "
                f"name")
        ds_want, coord_want, jd_want, binding = expected_tag[tag]
        recipe = rec.get("recipe") or {}
        coord_have = (str(recipe.get("routing_adaptive_topp_min")),
                      str(recipe.get("routing_adaptive_topp_max")))
        if rec.get("dataset") != ds_want or coord_have != tuple(coord_want) \
                or str(recipe.get("lambda_codon_joint")) != str(jd_want):
            raise SelectionRefused(
                f"{name} is tagged {tag!r}, which the plan says is "
                f"{ds_want}/{coord_want}/{jd_want}, but the record claims "
                f"{rec.get('dataset')}/{coord_have}/"
                f"{recipe.get('lambda_codon_joint')}")
        expected_launch = launch_binding_from_expected(binding, snap_digest)
        if rec.get("campaign") != expected_launch:
            raise SelectionRefused(
                f"{name}: campaign binding is not the prelaunch binding for "
                f"{binding['cell_id']}")
        if (rec.get("geometry") or {}).get("identity_digest") != \
                binding["expected_identity_digest"]:
            raise SelectionRefused(
                f"{name}: identity is not the RunIdentity sealed for "
                f"{binding['cell_id']}")
        if rec.get("result_root") != raw_result_root \
                or rec.get("environment_sha256") != environment_digest \
                or rec.get("input_authority") != input_seals.get(
                    f"{rec.get('dataset')}:stage1") \
                or Path(str(rec.get("run_dir"))).resolve().parent != result_root:
            raise SelectionRefused(
                f"{name}: record/run_dir is outside the sealed result root")
        if tag not in Path(str(rec.get("run_dir"))).name:
            raise SelectionRefused(
                f"{name}: run_dir {rec.get('run_dir')!r} does not carry its own "
                f"tag {tag!r}")
        if tag in seen_tags:
            raise SelectionRefused(
                f"{tag!r} is claimed by {seen_tags[tag]} and {name}")
        seen_tags[tag] = name
    missing_tags = sorted(set(expected_tag) - set(seen_tags))
    if missing_tags:
        raise SelectionRefused(
            f"the declared plan names {len(missing_tags)} tags no record "
            f"claims: {missing_tags[:3]}")

    only = snapshots.pop()
    if only != snap_digest:
        raise SelectionRefused(
            f"the records carry plan {only[:12]}... but the receipt seals "
            f"{snap_digest[:12]}...")
    return {"cells": cells, "grid": grid, "incumbent": incumbent,
            "record_sha256": digests,
            "plan_snapshot_sha256": only,
            "plan_snapshot_file": snap_path.name,
            "receipt_file": receipt_path.name,
            "receipt_sha256": _sha(receipt_path),
            "records_dir": str(records_dir.resolve()),
            "datasets": declared,
            "namespace": namespaces.pop() if namespaces else None,
            "authorities": plan.get("authorities") or {},
            "input_seals": input_seals,
            "environment_sha256": environment_digest,
            "source_authority_sha256": source_digest}


def build_stability_stage_plan(*, phase: str, state: dict, round_index: int,
                               max_update_rounds: int =
                               DEFAULT_MAX_UPDATE_ROUNDS) -> dict:
    """Build the immutable, prelaunch description of one stability campaign."""
    if phase not in RECIPE_STAGE_PHASES:
        raise SelectionRefused(f"unknown stability phase {phase!r}")
    if isinstance(round_index, bool) or not isinstance(round_index, int) \
            or round_index < 0:
        raise SelectionRefused("stability round_index must be non-negative")
    if phase.startswith("bootstrap_") and round_index != 0:
        raise SelectionRefused("bootstrap stage plans must use round_index=0")
    if not phase.startswith("bootstrap_") and round_index < 1:
        raise SelectionRefused("confirmation/update plans start at round 1")
    if isinstance(max_update_rounds, bool) \
            or not isinstance(max_update_rounds, int) \
            or max_update_rounds < 0:
        raise SelectionRefused("max_update_rounds must be a non-negative integer")
    state = _normalise_recipe_state(state)
    return {
        "schema_version": RECIPE_STAGE_PLAN_SCHEMA,
        "artifact_kind": "phase3_recipe_stability_stage_plan",
        "phase": phase,
        "axis": RECIPE_STAGE_PHASES[phase],
        "round_index": round_index,
        "max_update_rounds": max_update_rounds,
        "state": state,
        "reduction": RECIPE_REDUCTION,
        "pinned_policy": PINNED_CIFAR_TOPP_POLICY,
        "aggregator_sha256": _sha(Path(__file__)),
        "protocol_sources": protocol_digests(),
    }


def load_stability_stage_authority(
        path: Path, *, expected_phase: str | None = None,
        expected_state: dict | None = None, expected_round: int | None = None,
        expected_max_update_rounds: int | None = None) -> dict:
    """Reopen a stage plan and return the exact authority sealed by launch."""
    path = Path(path).resolve()
    if not path.is_file():
        raise SelectionRefused(f"stability stage plan does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise SelectionRefused(f"stability stage plan is unreadable: {error}")
    if payload.get("schema_version") != RECIPE_STAGE_PLAN_SCHEMA \
            or payload.get("artifact_kind") != \
            "phase3_recipe_stability_stage_plan":
        raise SelectionRefused("unknown stability stage-plan schema")
    phase = payload.get("phase")
    rebuilt = build_stability_stage_plan(
        phase=phase, state=payload.get("state"),
        round_index=payload.get("round_index"),
        max_update_rounds=payload.get("max_update_rounds"))
    if payload != rebuilt:
        raise SelectionRefused(
            "stability stage plan is stale, noncanonical, or was edited")
    if expected_phase is not None and phase != expected_phase:
        raise SelectionRefused(
            f"expected stability phase {expected_phase!r}, got {phase!r}")
    if expected_state is not None \
            and payload["state"] != _normalise_recipe_state(expected_state):
        raise SelectionRefused(
            f"{phase}: stage plan state is not the independently replayed state")
    if expected_round is not None and payload["round_index"] != expected_round:
        raise SelectionRefused(
            f"{phase}: stage round {payload['round_index']} != {expected_round}")
    if expected_max_update_rounds is not None \
            and payload["max_update_rounds"] != expected_max_update_rounds:
        raise SelectionRefused(
            f"{phase}: max-update policy changed between stages")
    return {
        "schema_version": RECIPE_STAGE_PLAN_SCHEMA,
        "path": str(path),
        "sha256": _sha(path),
        "phase": phase,
        "axis": payload["axis"],
        "round_index": payload["round_index"],
        "max_update_rounds": payload["max_update_rounds"],
        "state_sha256": hashlib.sha256(_state_key(payload["state"]).encode()).hexdigest(),
    }


def stability_stage_cells(stage_plan: dict) -> list:
    """Derive a launch plan; no global top-p/N value is accepted."""
    phase = stage_plan.get("phase")
    state = _normalise_recipe_state(stage_plan.get("state"))
    axis = RECIPE_STAGE_PHASES.get(phase)
    if axis == "topp":
        return stability_topp_cells(state)
    if axis == "joint":
        return stability_joint_cells(state)
    if axis == "n":
        from scripts.phase3_selection_matrix import n_selection_cells
        choices = {dataset: {
            "topp": state[dataset]["topp"],
            "joint": state[dataset]["joint"],
        } for dataset in sorted(state)}
        return n_selection_cells(choices)
    raise SelectionRefused(f"unknown stability phase {phase!r}")


def _stability_n_matrix(records_dir: Path, *, namespace: str,
                        state: dict, stage_authority: dict) -> dict:
    """Authenticate one provisional exact-16 N rerun without circular recipe."""
    from scripts.phase3_selection_matrix import (
        N_SELECTION_RECEIPT_SUFFIX, n_selection_cells)

    state = _normalise_recipe_state(state)
    choices = {dataset: {
        "topp": state[dataset]["topp"], "joint": state[dataset]["joint"]}
        for dataset in sorted(state)}
    expected_plan = n_selection_cells(choices)
    expected_authorities = {"stability_stage": stage_authority}

    def canonical(authorities):
        if authorities != expected_authorities:
            raise SelectionRefused(
                "N rerun was not launched from its exact stability stage plan")
        return expected_plan

    envelope = _campaign_envelope(
        records_dir, namespace=namespace, campaign_kind="n_selection",
        receipt_suffix=N_SELECTION_RECEIPT_SUFFIX, canonical_plan=canonical)
    cells = {}
    for record, path, _, _ in envelope["records"].values():
        key, value = _check_record(record, path)
        if key in cells:
            raise SelectionRefused(f"{path.name}: duplicate N cell {key}")
        cells[key] = {"value": value, "record": path.name,
                      "run_dir": record["run_dir"], "tag": record["tag"]}
    if set(cells) != set(cell_keys()):
        raise SelectionRefused(
            f"stability N campaign resolved {len(cells)} of 16 cells")
    return {
        "cells": cells,
        "namespace": envelope["namespace"],
        "records_dir": str(records_dir.resolve()),
        "receipt_file": envelope["receipt_path"].name,
        "receipt_sha256": _sha(envelope["receipt_path"]),
        "plan_snapshot_file": envelope["snapshot_path"].name,
        "plan_snapshot_sha256": envelope["plan_digest"],
        "record_sha256": envelope["record_sha256"],
        "stage_authority": stage_authority,
        "input_seals": envelope["snapshot"].get("input_seals"),
        "environment_sha256": envelope["snapshot"].get("environment_sha256"),
        "source_authority_sha256": envelope["snapshot"].get(
            "source_authority_sha256"),
    }


def _stability_ref(ref: dict, *, phase: str, round_index: int, state: dict,
                   max_update_rounds: int) -> tuple[dict, dict]:
    """Resolve one manifest reference into decisions and sealed provenance."""
    if not isinstance(ref, dict) or set(ref) != {
            "records_dir", "namespace", "stage_plan"}:
        raise SelectionRefused(
            f"{phase}: evidence ref must contain records_dir/namespace/stage_plan")
    records_dir = Path(str(ref["records_dir"])).resolve()
    namespace = ref["namespace"]
    if not records_dir.is_dir() or not isinstance(namespace, str) or not namespace:
        raise SelectionRefused(f"{phase}: invalid records_dir/namespace")
    stage_authority = load_stability_stage_authority(
        Path(str(ref["stage_plan"])), expected_phase=phase,
        expected_state=state, expected_round=round_index,
        expected_max_update_rounds=max_update_rounds)
    expected_authorities = {"stability_stage": stage_authority}
    axis = RECIPE_STAGE_PHASES[phase]
    if axis in ("topp", "joint"):
        expected_plan = (stability_topp_cells(state) if axis == "topp"
                         else stability_joint_cells(state))
        matrix = _recipe_matrix(
            records_dir, axis=axis, namespace=namespace,
            expected_plan=expected_plan,
            expected_authorities=expected_authorities)
        grid = matrix["grid"]
        decisions = {}
        expected_datasets = sorted({cell[0] for cell in expected_plan})
        for dataset in expected_datasets:
            incumbent = (tuple(state[dataset]["topp"])
                         if axis == "topp" else state[dataset]["joint"])
            subset = {(d, c): v for (d, c), v in matrix["cells"].items()
                      if d == dataset}
            picked = choose_recipe(
                subset, axis=axis, grid=grid, incumbent=incumbent,
                datasets=[dataset])[dataset]["selected"]
            decisions[dataset] = picked
    else:
        matrix = _stability_n_matrix(
            records_dir, namespace=namespace, state=state,
            stage_authority=stage_authority)
        decisions = {dataset: entry["selected_N"]
                     for dataset, entry in choose(matrix["cells"]).items()}
    summary_fields = (
        "records_dir", "namespace", "receipt_file", "receipt_sha256",
        "plan_snapshot_file", "plan_snapshot_sha256", "record_sha256",
        "input_seals", "environment_sha256", "source_authority_sha256")
    summary = {field: matrix.get(field) for field in summary_fields}
    summary["stage_authority"] = stage_authority
    summary["phase"] = phase
    summary["round_index"] = round_index
    summary["decisions"] = decisions
    return decisions, summary


def build_recipe_stability_artifact(spec: dict) -> dict:
    """Reopen every receipt/record and build the sole production recipe."""
    if not isinstance(spec, dict) \
            or spec.get("schema_version") != RECIPE_STABILITY_SCHEMA \
            or spec.get("artifact_kind") != \
            "phase3_recipe_stability_evidence":
        raise SelectionRefused("unknown recipe stability evidence schema")
    if set(spec) != {"schema_version", "artifact_kind",
                     "max_update_rounds", "bootstrap", "rounds"}:
        raise SelectionRefused("stability evidence has unexpected/missing fields")
    max_updates = spec.get("max_update_rounds")
    if isinstance(max_updates, bool) or not isinstance(max_updates, int) \
            or max_updates < 0:
        raise SelectionRefused("invalid max_update_rounds")
    bootstrap_refs = spec.get("bootstrap")
    rounds_refs = spec.get("rounds")
    if not isinstance(bootstrap_refs, dict) \
            or set(bootstrap_refs) != {"topp", "joint", "n"} \
            or not isinstance(rounds_refs, list):
        raise SelectionRefused("stability evidence needs bootstrap P/J/N and rounds")

    state = initial_recipe_state()
    resolved = {"bootstrap": {}, "rounds": []}
    decisions_boot = {}
    p, summary = _stability_ref(
        bootstrap_refs["topp"], phase="bootstrap_topp", round_index=0,
        state=state, max_update_rounds=max_updates)
    decisions_boot["topp"] = p
    resolved["bootstrap"]["topp"] = summary
    state = _apply_decisions(state, p, axis="topp")
    j, summary = _stability_ref(
        bootstrap_refs["joint"], phase="bootstrap_joint", round_index=0,
        state=state, max_update_rounds=max_updates)
    decisions_boot["joint"] = j
    resolved["bootstrap"]["joint"] = summary
    state = _apply_decisions(state, j, axis="joint")
    n, summary = _stability_ref(
        bootstrap_refs["n"], phase="bootstrap_n", round_index=0,
        state=state, max_update_rounds=max_updates)
    decisions_boot["n"] = n
    resolved["bootstrap"]["n"] = summary
    state = _apply_decisions(state, n, axis="n")

    decisions_rounds = []
    for index, refs in enumerate(rounds_refs, 1):
        if not isinstance(refs, dict) \
                or not {"topp", "joint"} <= set(refs) \
                or set(refs) - {"topp", "joint", "n"}:
            raise SelectionRefused(
                f"round {index}: evidence needs P/J and optional N refs")
        round_decisions, round_summary = {}, {}
        p, summary = _stability_ref(
            refs["topp"], phase="confirm_topp", round_index=index,
            state=state, max_update_rounds=max_updates)
        round_decisions["topp"] = p
        round_summary["topp"] = summary
        after_p = _apply_decisions(state, p, axis="topp")
        j, summary = _stability_ref(
            refs["joint"], phase="confirm_joint", round_index=index,
            state=after_p, max_update_rounds=max_updates)
        round_decisions["joint"] = j
        round_summary["joint"] = summary
        after_j = _apply_decisions(after_p, j, axis="joint")
        changed = any(
            after_j[d][axis] != state[d][axis]
            for d in state for axis in ("topp", "joint"))
        if changed:
            if "n" not in refs:
                raise SelectionRefused(
                    f"round {index}: P/J drift requires an exact-16 N ref")
            n, summary = _stability_ref(
                refs["n"], phase="update_n", round_index=index,
                state=after_j, max_update_rounds=max_updates)
            round_decisions["n"] = n
            round_summary["n"] = summary
            state = _apply_decisions(after_j, n, axis="n")
        else:
            if "n" in refs:
                raise SelectionRefused(
                    f"round {index}: unchanged P/J supplied an extraneous N ref")
            state = after_j
        decisions_rounds.append(round_decisions)
        resolved["rounds"].append(round_summary)

    replay = run_recipe_stability(
        decisions_boot, decisions_rounds,
        max_update_rounds=max_updates)
    final_state = replay["final_state"]
    selected = {dataset: {
        "topp": final_state[dataset]["topp"],
        "joint": final_state[dataset]["joint"],
    } for dataset in sorted(final_state)}
    return {
        "schema_version": RECIPE_STABILITY_ARTIFACT_SCHEMA,
        "artifact_kind": "phase3_recipe_stability",
        "reduction": RECIPE_REDUCTION,
        "aggregator_sha256": _sha(Path(__file__)),
        "protocol_sources": protocol_digests(),
        "pinned_policy": PINNED_CIFAR_TOPP_POLICY,
        "evidence_spec": json.loads(json.dumps(spec, sort_keys=True)),
        "resolved_evidence": resolved,
        "stability": replay,
        "selected": selected,
    }


def verify_recipe_stability_artifact(path: Path) -> dict:
    """Reopen all evidence and replay the state machine byte-for-byte."""
    path = Path(path).resolve()
    if not path.is_file():
        raise SelectionRefused(f"stable recipe does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise SelectionRefused(f"stable recipe is unreadable: {error}")
    if payload.get("schema_version") != RECIPE_STABILITY_ARTIFACT_SCHEMA \
            or payload.get("artifact_kind") != "phase3_recipe_stability":
        raise SelectionRefused("not a final recipe-stability authority")
    rebuilt = build_recipe_stability_artifact(payload.get("evidence_spec"))
    if payload != rebuilt:
        raise SelectionRefused(
            "stable recipe differs from reopened receipts/records or replay")
    if (payload.get("stability") or {}).get("confirmed") is not True:
        raise SelectionRefused("recipe stability is not confirmed")
    return payload["selected"]


def _publish_json_exclusive(path: Path, payload: dict) -> None:
    """Publish an authority once; replacing a prior decision is forbidden."""
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        fd = os.open(path, flags, 0o644)
    except FileExistsError:
        raise SelectionRefused(
            f"authority already exists and will not be replaced: {path}")
    try:
        blob = (json.dumps(payload, indent=2, sort_keys=True,
                           allow_nan=False) + "\n").encode()
        with os.fdopen(fd, "wb") as handle:
            handle.write(blob)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def verify_selection_artifact(path: Path) -> dict:
    """Reopen all 16 inputs and recompute a selected-N artefact from bytes."""
    if not path.is_file():
        raise SelectionRefused(f"{path} does not exist")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise SelectionRefused(f"{path} is unreadable: {error}")
    if payload.get("schema_version") != SELECTED_N_SCHEMA \
            or payload.get("artifact_kind") != "phase3_selected_n":
        raise SelectionRefused(
            f"{path}: not a schema-{SELECTED_N_SCHEMA} selected-N authority")
    if payload.get("aggregator_sha256") != _sha(Path(__file__)):
        raise SelectionRefused(
            f"{path}: selected by a stale phase3_select_n.py")
    if payload.get("reduction") != N_SELECTION_REDUCTION:
        raise SelectionRefused(
            f"{path}: N reduction rule/grid/tie-break differs from this tree")
    if payload.get("protocol_sources") != protocol_digests():
        raise SelectionRefused(
            f"{path}: protocol source seal is stale")
    records_dir = Path(str(payload.get("records_dir") or ""))
    namespace = payload.get("namespace")
    if not records_dir.is_dir() or not isinstance(namespace, str) or not namespace:
        raise SelectionRefused(
            f"{path}: records_dir/namespace authority is absent")
    matrix = load_matrix(records_dir, namespace=namespace)
    comparisons = {
        "receipt_file": matrix["receipt_file"],
        "receipt_sha256": matrix["receipt_sha256"],
        "plan_snapshot_file": matrix["plan_snapshot_file"],
        "plan_snapshot_sha256": matrix["plan_snapshot_sha256"],
        "recipe_authority": matrix["recipe_authority"],
        "record_sha256": matrix["record_sha256"],
    }
    wrong = {field: (payload.get(field), actual)
             for field, actual in comparisons.items()
             if payload.get(field) != actual}
    if wrong:
        raise SelectionRefused(
            f"{path}: sealed N campaign inputs changed: {sorted(wrong)}")
    recomputed = choose(matrix["cells"])
    if payload.get("selected") != recomputed:
        raise SelectionRefused(
            f"{path}: selected N is not raw terminal mAP@R argmax with the "
            "smallest-N tie-break over its sealed 16 records")
    return {dataset: entry["selected_N"]
            for dataset, entry in sorted(recomputed.items())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", default=str(DEFAULT_RECORDS))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--namespace", default=None,
                        help=("which sweep to reduce; the N matrix and a recipe "
                              "sweep share one records directory"))
    parser.add_argument("--axis", choices=("n", "topp", "joint"), default="n",
                        help="which coordinate this reduction chooses")
    parser.add_argument(
        "--stability-spec", default=None, metavar="PATH",
        help=("build the final production recipe by reopening the exact "
              "bootstrap/confirmation campaign references in this evidence "
              "manifest and replaying the fixed-point state machine"))
    parser.add_argument(
        "--emit-stage-plan", choices=tuple(RECIPE_STAGE_PHASES), default=None,
        help=("write a prelaunch P/J/N stability plan to --out; use --state "
              "for the current four-dataset state"))
    parser.add_argument("--state", default=None, metavar="PATH")
    parser.add_argument("--round-index", type=int, default=0)
    parser.add_argument("--max-update-rounds", type=int,
                        default=DEFAULT_MAX_UPDATE_ROUNDS)
    args = parser.parse_args()

    if args.stability_spec and args.emit_stage_plan:
        print("[phase3-select] choose --stability-spec or --emit-stage-plan, "
              "not both", file=sys.stderr)
        return 2
    if args.emit_stage_plan:
        if not args.state:
            print("[phase3-select] --emit-stage-plan requires --state",
                  file=sys.stderr)
            return 2
        try:
            if args.state == "initial":
                state = initial_recipe_state()
            else:
                state_payload = json.loads(
                    Path(args.state).read_text(encoding="utf-8"))
                state = state_payload.get("state", state_payload)
            payload = build_stability_stage_plan(
                phase=args.emit_stage_plan, state=state,
                round_index=args.round_index,
                max_update_rounds=args.max_update_rounds)
            _publish_json_exclusive(Path(args.out), payload)
        except (OSError, ValueError, SelectionRefused) as error:
            print(f"[phase3-select] REFUSED stage plan: {error}", file=sys.stderr)
            return 1
        print(f"wrote {args.out}")
        return 0
    if args.stability_spec:
        try:
            spec = json.loads(
                Path(args.stability_spec).read_text(encoding="utf-8"))
            payload = build_recipe_stability_artifact(spec)
            _publish_json_exclusive(Path(args.out), payload)
            # Reopen the published bytes as the downstream N/refit consumer
            # will.  Publication is success only if the replay still agrees.
            verify_recipe_stability_artifact(Path(args.out))
        except (OSError, ValueError, SelectionRefused) as error:
            print(f"[phase3-select] REFUSED stability: {error}", file=sys.stderr)
            return 1
        print(f"wrote confirmed recipe {args.out}")
        return 0

    if args.axis != "n":
        if not args.namespace:
            print("[phase3-select] --namespace is required for a recipe "
                  "reduction: the N matrix and every sweep share one records "
                  "directory", file=sys.stderr)
            return 2
        try:
            matrix = _recipe_matrix(Path(args.records), axis=args.axis,
                                    namespace=args.namespace)
            chosen = choose_recipe(matrix["cells"], axis=args.axis,
                                   grid=matrix["grid"],
                                   incumbent=matrix["incumbent"],
                                   datasets=matrix["datasets"])
        except SelectionRefused as error:
            print(f"[phase3-select] REFUSED: {error}", file=sys.stderr)
            return 1
        payload = {
            "schema_version": 1,
            "axis": args.axis,
            "reduction": RECIPE_REDUCTION,
            "grid": [list(c) if isinstance(c, tuple) else c
                     for c in matrix["grid"]],
            "incumbent": (list(matrix["incumbent"])
                          if isinstance(matrix["incumbent"], tuple)
                          else matrix["incumbent"]),
            "namespace": matrix["namespace"],
            "records_dir": matrix["records_dir"],
            "receipt_file": matrix["receipt_file"],
            "receipt_sha256": matrix["receipt_sha256"],
            "plan_snapshot_sha256": matrix["plan_snapshot_sha256"],
            "aggregator_sha256": _sha(Path(__file__)),
            "protocol_sources": protocol_digests(),
            "record_sha256": matrix["record_sha256"],
            "selected": chosen,
            "stability": {
                "confirmed": False,
                "state_machine_schema": None,
                "note": ("provisional: top-p -> JD -> N -> top-p confirmation "
                         "with predeclared repeat/cycle/fail policy has not "
                         "completed"),
            },
        }
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
        os.replace(tmp, out)
        for dataset in sorted(chosen):
            e = chosen[dataset]
            note = "  (tie -> incumbent)" if e["tie_broken"] else ""
            print(f"  {dataset:<10} {args.axis}={e['selected']} "
                  f"mAP@R={e['selection_value']:.4f} "
                  f"delta={e['delta_vs_incumbent']:+.4f}{note}")
        print(f"wrote {out}")
        return 0

    if not args.namespace:
        print("[phase3-select] --namespace is required for the exact-16 N "
              "campaign", file=sys.stderr)
        return 2
    try:
        matrix = load_matrix(Path(args.records), namespace=args.namespace)
    except SelectionRefused as error:
        print(f"[phase3-select] REFUSED: {error}", file=sys.stderr)
        return 1

    chosen = choose(matrix["cells"])
    payload = {
        "schema_version": SELECTED_N_SCHEMA,
        "artifact_kind": "phase3_selected_n",
        "what_this_is": (
            "D1's (N) choice per dataset: argmax of the raw base-Hamming mAP@R "
            "at each candidate's own terminal epoch, ties to the smallest N. "
            "The chosen N is reused unchanged for seeds 42, 43 and 44."),
        "selection_metric": "eval_mAP_at_R",
        "selection_distance": "base_hamming",
        "reduction": N_SELECTION_REDUCTION,
        "candidate_n": list(CANDIDATE_N),
        "seed": SEED,
        "val_split_ratio": VAL_RATIO,
        "val_split_seed": VAL_SEED,
        "geometry": {"num_slots": SLOTS, "bases_per_slot": BASES_PER_SLOT,
                     "total_bases": TOTAL_BASES, "total_bits": TOTAL_BITS},
        "namespace": matrix["namespace"],
        "records_dir": matrix["records_dir"],
        "protocol_sources": matrix["protocol_sources"],
        "aggregator_sha256": _sha(Path(__file__)),
        "receipt_file": matrix["receipt_file"],
        "receipt_sha256": matrix["receipt_sha256"],
        "plan_snapshot_file": matrix["plan_snapshot_file"],
        "plan_snapshot_sha256": matrix["plan_snapshot_sha256"],
        "recipe_authority": matrix["recipe_authority"],
        "record_sha256": matrix["record_sha256"],
        "selected": chosen,
    }

    out = Path(args.out)
    from scripts.phase3_selection_matrix import _publish_json_exclusive
    try:
        _publish_json_exclusive(out, payload)
    except Exception as error:                         # noqa: BLE001
        print(f"[phase3-select] REFUSED output: {error}", file=sys.stderr)
        return 1

    for dataset in sorted(chosen):
        entry = chosen[dataset]
        note = "  (tie -> smallest)" if entry["tie_broken_by_smallest_N"] else ""
        print(f"  {dataset:<10} N={entry['selected_N']:<3} "
              f"mAP@R={entry['selection_value']:.4f}{note}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
