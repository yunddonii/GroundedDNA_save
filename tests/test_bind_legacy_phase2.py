"""Focused authority tests for the Phase-2 inventory materializer."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import stat
import subprocess
import sys

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dna_utils import extraction_validation, runtime_state  # noqa: E402
from scripts import aggregate_phase2_f01 as aggregator  # noqa: E402
from scripts import bind_legacy_phase2 as binder  # noqa: E402

CELL = "cifar10_N4"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _two_bit(base: np.ndarray) -> np.ndarray:
    return np.stack((base // 2, base % 2), axis=-1).reshape(len(base), -1)


def _write_npz(path: Path, rows: int, *, label_offset: int = 0) -> None:
    base = np.arange(rows * 15, dtype=np.int64).reshape(rows, 15) % 4
    np.savez(
        path, base_indices=base, hash_2bit=_two_bit(base),
        codebook_indices=np.tile(np.arange(5, dtype=np.int64), (rows, 1)),
        labels=(np.arange(rows, dtype=np.int64) + label_offset) % 3,
        image_paths=np.asarray([f"sample-{index}" for index in range(rows)]),
    )


def _args() -> bytes:
    return (
        "num_semantic_parts----------------5\n"
        "num_codons_per_codebook-----------3\n"
        "codebook_size----------------------64\n"
        "epoch------------------------------5\n"
        "stop_after_epoch-------------------4\n"
        "sinkhorn_epsilon_init--------------1.0\n"
        "sinkhorn_epsilon_final-------------0.1\n"
        "random_seed------------------------42\n"
    ).encode()


def _runtime(side: str, checkpoint_sha: str, config_sha: str) -> dict:
    return {
        "backfilled": True, "bases_per_slot": 3,
        "checkpoint_sha256": checkpoint_sha, "codebook_size": 64,
        "config_sha256": config_sha, "dataset": "cifar10",
        "effective_sinkhorn_epsilon": 0.1 if side == "fixed" else 1.0,
        "inference_epoch": 4 if side == "fixed" else 0,
        "inference_epoch_source": (
            "explicit_flag" if side == "fixed" else "f01_unrestored"),
        "lr_schedule_horizon": 5, "num_slots": 5, "random_seed": 42,
        "sinkhorn_annealing_enabled": True, "sinkhorn_schedule_horizon": 5,
        "total_bases": 15, "total_bits": 30, "training_epoch_budget": 5,
        "training_stop_epoch": 4,
    }


@pytest.fixture
def authority(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    source = tmp_path / "preserved"
    fixed_dir = source / "fixed" / CELL
    legacy_dir = source / "result" / "preserved_cifar10_N4"
    fixed_dir.mkdir(parents=True)
    legacy_dir.mkdir(parents=True)
    for directory in (fixed_dir, legacy_dir):
        _write_npz(directory / "extract_db.npz", 4)
        _write_npz(directory / "extract_query.npz", 2)
        (directory / "args.txt").write_bytes(_args())
    (fixed_dir / "extract.log").write_bytes(b"fixed extraction\n")
    (legacy_dir / "model_state_dict.pth").write_bytes(b"checkpoint\n")
    (legacy_dir / "config.pt").write_bytes(b"config\n")
    checkpoint_sha = _sha(legacy_dir / "model_state_dict.pth")
    config_sha = _sha(legacy_dir / "config.pt")

    inventories, paths = {}, {}
    for side, data_dir in (("fixed", fixed_dir), ("legacy", legacy_dir)):
        runtime = _runtime(side, checkpoint_sha, config_sha)
        entry = {
            "cell": CELL, "dataset": "cifar10", "stop_epoch_zero_based": 4,
            "runtime_identity": runtime, "checkpoint_sha256": checkpoint_sha,
            "config_sha256": config_sha, "args_txt_sha256": _sha(data_dir / "args.txt"),
            "extract_log_sha256": (
                _sha(fixed_dir / "extract.log") if side == "fixed" else None),
            "canonical_legacy_source": {
                "dir": str(legacy_dir.relative_to(source)),
                "model_state_dict_sha256": checkpoint_sha,
                "config_sha256": config_sha,
            },
            "splits": {
                split: {
                    "npz_path_relative": str(
                        (data_dir / f"extract_{split}.npz").relative_to(source)),
                    "npz_sha256": _sha(data_dir / f"extract_{split}.npz"),
                    "n_rows": rows,
                }
                for split, rows in (("db", 4), ("query", 2))
            },
        }
        inventories[side] = {
            "schema_version": 2, "cells_validated": 1, "cells_refused": [],
            "eligibility": "diagnostic_only_never_promoted_to_paper_main",
            "cells": [entry],
        }
        paths[side] = tmp_path / f"{side}.inventory.json"

    monkeypatch.setattr(binder, "_EXPECTED_CELLS", frozenset({CELL}))

    def pin(side: str) -> None:
        paths[side].write_text(
            json.dumps(inventories[side], indent=2, sort_keys=True) + "\n")
        current = dict(binder._INVENTORIES)
        current[side] = (paths[side], _sha(paths[side]))
        monkeypatch.setattr(binder, "_INVENTORIES", current)

    pin("fixed")
    pin("legacy")
    source_state = {
        "git_head": "f" * 40,
        "source_sha256": {
            "scripts/bind_legacy_phase2.py": _sha(Path(binder.__file__)),
            "dna_utils/extraction_validation.py": _sha(
                Path(extraction_validation.__file__)),
            "dna_utils/runtime_state.py": _sha(Path(runtime_state.__file__)),
        },
    }
    monkeypatch.setattr(binder, "_source_state", lambda: copy.deepcopy(source_state))
    return SimpleNamespace(
        source=source, fixed_dir=fixed_dir, legacy_dir=legacy_dir,
        inventories=inventories, paths=paths, pin=pin,
        monkeypatch=monkeypatch, source_state=source_state,
    )


def test_both_sides_publish_one_common_exact_receipt_and_verify(
        authority: SimpleNamespace) -> None:
    roots, receipts = {}, {}
    for side in ("fixed", "legacy"):
        roots[side] = authority.source.parent / f"out-{side}"
        payload, receipt_sha = binder.materialize(
            authority.source, roots[side], side)
        reopened, reopened_sha = binder.verify_input_root(roots[side], side)
        assert (payload, receipt_sha) == (reopened, reopened_sha)
        assert set(payload["cells"]) == {CELL}
        assert set(payload["cells"][CELL]) == {
            "runtime_identity", "completion_marker_sha256", "npz_sha256"}
        assert "dna_utils/runtime_state.py" in payload["source_sha256"]
        receipts[side] = payload
        marker = roots[side] / binder.INPUT_RECEIPT_NAME
        assert stat.S_ISREG(marker.stat(follow_symlinks=False).st_mode)
        assert stat.S_IMODE(marker.stat().st_mode) == 0o600
        assert not list(roots[side].glob(f".{binder.INPUT_RECEIPT_NAME}.*.tmp"))
        for split in ("db", "query"):
            copied = roots[side] / CELL / f"extract_{split}.npz"
            original = (authority.fixed_dir if side == "fixed"
                        else authority.legacy_dir) / copied.name
            copied_stat = copied.stat(follow_symlinks=False)
            assert copied_stat.st_nlink == 1 and not copied.is_symlink()
            assert stat.S_IMODE(copied_stat.st_mode) == 0o600
            assert (copied_stat.st_dev, copied_stat.st_ino) != (
                original.stat().st_dev, original.stat().st_ino)
    assert receipts["fixed"]["pair_evidence"] == receipts["legacy"]["pair_evidence"]
    assert (roots["fixed"] / CELL / "extract.log").is_file()
    assert not (roots["legacy"] / CELL / "extract.log").exists()
    authority.monkeypatch.setattr(aggregator, "EXPECTED_CELLS", (CELL,))
    authority.monkeypatch.setattr(
        aggregator, "_git_head", lambda: authority.source_state["git_head"])
    pair_path = authority.source.parent / "phase2_input_pair_receipt.json"
    aggregator._publish_json_exclusive(
        pair_path, aggregator._pair_authority_payload(
            roots["fixed"], roots["legacy"]))
    pair, _ = aggregator._load_pair_receipt(
        pair_path, roots["fixed"], roots["legacy"])
    assert pair["pair_evidence"] == receipts["fixed"]["pair_evidence"]

    for root in roots.values():
        cell = root / CELL
        commands = [
            [sys.executable, str(REPO / "evaluation_siglip2.py"),
             "--extraction_path", str(cell), "--distance_mode", "base",
             "--codebook_size", "64", "--dataset", "CIFAR10",
             "--no-bio_project", "--query_chunk_size", "64"],
            [sys.executable, str(REPO / "scripts" / "eval_cell_bioproj.py"),
             "--dir", str(cell), "--dataset", "CIFAR10", "--K", "64",
             "--allow-backfilled"],
            [sys.executable, str(REPO / "scripts" / "pairwise_nmi.py"),
             "--results", str(cell), "--allow-backfilled"],
            [sys.executable, str(REPO / "scripts" / "seal_cell_analysis.py"),
             "--dir", str(cell), "--allow-backfilled"],
        ]
        for command in commands:
            completed = subprocess.run(
                command, cwd=REPO, capture_output=True, text=True, timeout=300)
            assert completed.returncode == 0, completed.stdout + completed.stderr

    authority.monkeypatch.setattr(
        aggregator, "LEGACY", {("cifar10", 4): "inventory-bound"})
    authority.monkeypatch.setattr(aggregator, "ORDER", ("cifar10",))
    refused = authority.source.parent / "refused-report"
    refused.mkdir()
    target = refused / "must-not-be-created.json"
    (refused / "impact.json").symlink_to(target)
    authority.monkeypatch.setattr(sys, "argv", [
        aggregator.__file__, "--phase2-root", str(roots["fixed"]),
        "--legacy-root", str(roots["legacy"]), "--pair-receipt", str(pair_path),
        "--out-json", str(refused / "impact.json"),
        "--out-md", str(refused / "impact.md"),
        "--out-receipt", str(refused / "receipt.json")])
    assert aggregator.main() == 2
    assert (refused / "impact.json").is_symlink() and not target.exists()

    report = authority.source.parent / "report"
    report.mkdir()
    authority.monkeypatch.setattr(sys, "argv", [
        aggregator.__file__, "--phase2-root", str(roots["fixed"]),
        "--legacy-root", str(roots["legacy"]), "--pair-receipt", str(pair_path),
        "--out-json", str(report / "impact.json"),
        "--out-md", str(report / "impact.md"),
        "--out-receipt", str(report / "receipt.json")])
    assert aggregator.main() == 0
    final = json.loads((report / "receipt.json").read_text())
    assert final["complete"] is True and final["paired_cells"] == 1
    assert "extraction_validation_sha256" in final["source_sha256"]
    verify_kwargs = {
        "expected_fixed_root": roots["fixed"],
        "expected_legacy_root": roots["legacy"],
        "expected_pair_receipt": pair_path,
        "expected_report_json": report / "impact.json",
        "expected_report_md": report / "impact.md",
    }
    verified, _ = aggregator.verify_report_receipt(
        report / "receipt.json", **verify_kwargs)
    assert verified == final

    original_report = (report / "impact.json").read_bytes()
    original_receipt = (report / "receipt.json").read_bytes()
    forged_report = json.loads(original_report)
    forged_report["cells"] = []
    (report / "impact.json").write_text(
        json.dumps(forged_report, indent=2, sort_keys=True) + "\n")
    forged_receipt = json.loads(original_receipt)
    forged_receipt["reports"]["json"]["sha256"] = _sha(
        report / "impact.json")
    (report / "receipt.json").write_text(
        json.dumps(forged_receipt, indent=2, sort_keys=True) + "\n")
    with pytest.raises(aggregator.ManifestMissing, match="semantics"):
        aggregator.verify_report_receipt(
            report / "receipt.json", **verify_kwargs)
    (report / "impact.json").write_bytes(original_report)
    (report / "receipt.json").write_bytes(original_receipt)

    with pytest.raises(aggregator.ManifestMissing, match="this invocation"):
        aggregator.verify_report_receipt(
            report / "receipt.json", **{
                **verify_kwargs,
                "expected_report_json": report / "different.json",
            })
    original_loader = extraction_validation._load_json_bound

    def replace_final_receipt(path, what):
        payload, digest = original_loader(path, what)
        if what == "Phase-2 report receipt final reopen":
            payload = {**payload, "paired_cells": 999}
        return payload, digest

    with authority.monkeypatch.context() as nested:
        nested.setattr(
            extraction_validation, "_load_json_bound", replace_final_receipt)
        with pytest.raises(aggregator.ManifestMissing, match="evidence changed"):
            aggregator.verify_report_receipt(
                report / "receipt.json", **verify_kwargs)

    events = []
    original_sources = aggregator._report_source_sha256
    original_validate = extraction_validation.validate_extraction_run
    validate_calls = 0

    def record_loader(path, what):
        payload, digest = original_loader(path, what)
        if what == "Phase-2 report receipt final reopen":
            events.append("receipt")
        return payload, digest

    def record_sources():
        events.append("source")
        return original_sources()

    def count_validate(*args, **kwargs):
        nonlocal validate_calls
        validate_calls += 1
        return original_validate(*args, **kwargs)

    def recursive_main_is_forbidden():
        pytest.fail("report verification must not recursively call main()")

    with authority.monkeypatch.context() as nested:
        nested.setattr(extraction_validation, "_load_json_bound", record_loader)
        nested.setattr(aggregator, "_report_source_sha256", record_sources)
        nested.setattr(
            extraction_validation, "validate_extraction_run", count_validate)
        nested.setattr(binder, "validate_extraction_run", count_validate)
        nested.setattr(aggregator, "main", recursive_main_is_forbidden)
        aggregator.verify_report_receipt(
            report / "receipt.json", **verify_kwargs)
    assert events[-1] == "receipt"
    # This pins ONE thing: the verifier no longer reaches its canonical payload
    # by re-entering `main()`, so it does not produce what it verifies.  It is
    # NOT evidence that the duplicate-read cost of §63.31 is resolved -- audit
    # §64.2 retracted that as a gate anyway, and §64.3 showed this count cannot
    # carry it.  The verifier still validates each cell-side FOUR times, and
    # `_inspect_npz` is not counted here at all; the dominant repetition is in
    # `verify_input_root`, which this refactor does not touch.  Writing the
    # duplication factor out loud keeps a later reader from mistaking a passing
    # test for an efficient one.
    validates_per_cell_side = 4
    sides = 2  # fixed + legacy
    assert validate_calls == (
        validates_per_cell_side * sides * len(aggregator.EXPECTED_CELLS))
    (report / "impact.md").write_text("changed after receipt\n")
    with pytest.raises(aggregator.ManifestMissing, match="report bytes"):
        aggregator.verify_report_receipt(
            report / "receipt.json", **verify_kwargs)


def test_either_hard_bound_inventory_drift_refuses_before_root(
        authority: SimpleNamespace) -> None:
    authority.paths["fixed"].write_bytes(
        authority.paths["fixed"].read_bytes() + b" \n")
    out = authority.source.parent / "never-created"
    with pytest.raises(binder.BindRefused, match="inventory SHA"):
        binder.materialize(authority.source, out, "legacy")
    assert not out.exists()


def test_full_identity_drift_refuses_before_root(authority: SimpleNamespace) -> None:
    authority.inventories["fixed"]["cells"][0]["runtime_identity"][
        "random_seed"] = 43
    authority.pin("fixed")
    out = authority.source.parent / "never-created"
    with pytest.raises(binder.BindRefused, match="runtime identity drift"):
        binder.materialize(authority.source, out, "fixed")
    assert not out.exists()


def test_pair_order_drift_refuses_before_root(authority: SimpleNamespace) -> None:
    target = authority.legacy_dir / "extract_query.npz"
    _write_npz(target, 2, label_offset=1)
    authority.inventories["legacy"]["cells"][0]["splits"]["query"][
        "npz_sha256"] = _sha(target)
    authority.pin("legacy")
    out = authority.source.parent / "never-created"
    with pytest.raises(binder.BindRefused, match="sample/order evidence differs"):
        binder.materialize(authority.source, out, "fixed")
    assert not out.exists()


def test_exclusive_publish_preserves_competing_destination(
        authority: SimpleNamespace) -> None:
    _, plans, _ = binder._preflight(authority.source)
    cell = authority.source.parent / "direct-cell"
    cell.mkdir()
    destination = cell / "extract_db.npz"
    destination.write_bytes(b"competitor\n")
    with pytest.raises(binder.BindRefused, match="refusing to overwrite"):
        binder._copy_npz(
            plans["fixed"][CELL]["splits"]["db"]["path"], destination,
            plan=plans["fixed"][CELL], split="db")
    assert destination.read_bytes() == b"competitor\n"
    assert not list(cell.glob(".*.tmp"))

    target = cell / "temp-target"
    target.write_bytes(b"keep-temp-target\n")
    temporary = cell / f".extract_query.npz.{os.getpid()}.tmp"
    temporary.symlink_to(target)
    with pytest.raises(binder.BindRefused, match="refusing to overwrite"):
        binder._copy_npz(
            plans["fixed"][CELL]["splits"]["query"]["path"],
            cell / "extract_query.npz", plan=plans["fixed"][CELL],
            split="query")
    assert temporary.is_symlink()
    assert target.read_bytes() == b"keep-temp-target\n"


def test_partial_materialization_has_no_root_receipt(
        authority: SimpleNamespace) -> None:
    out = authority.source.parent / "partial"
    authority.monkeypatch.setattr(
        binder, "_materialize_cell",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            binder.BindRefused("injected copy failure")))
    with pytest.raises(binder.BindRefused, match="injected copy failure"):
        binder.materialize(authority.source, out, "fixed")
    assert out.is_dir()
    assert not (out / binder.INPUT_RECEIPT_NAME).exists()


def test_source_drift_before_receipt_leaves_root_unsealed(
        authority: SimpleNamespace) -> None:
    out = authority.source.parent / "source-drift"
    changed = copy.deepcopy(authority.source_state)
    changed["git_head"] = "e" * 40
    states = iter((authority.source_state, changed))
    authority.monkeypatch.setattr(
        binder, "_source_state", lambda: copy.deepcopy(next(states)))
    with pytest.raises(binder.BindRefused, match="source HEAD/digests changed"):
        binder.materialize(authority.source, out, "fixed")
    assert out.is_dir() and not (out / binder.INPUT_RECEIPT_NAME).exists()
    assert not (out / CELL / "extraction_complete.json").exists()


@pytest.mark.parametrize(
    "target", ["validator", "args", "args_symlink", "support_retarget"])
def test_verify_refuses_receipt_or_copied_input_tampering(
        authority: SimpleNamespace, target: str) -> None:
    out = authority.source.parent / "tampered"
    binder.materialize(authority.source, out, "fixed")
    if target == "validator":
        path = out / binder.INPUT_RECEIPT_NAME
        payload = json.loads(path.read_text())
        payload["validator_sha256"] = "0" * 64
        path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    elif target == "args":
        (out / CELL / "args.txt").write_bytes(b"changed\n")
    elif target == "args_symlink":
        path = out / CELL / "args.txt"
        substitute = out.parent / "same-args.txt"
        substitute.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(substitute)
    else:
        path = out / CELL / "model_state_dict.pth"
        substitute = out.parent / "same-checkpoint.pth"
        substitute.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(substitute)
    with pytest.raises(binder.BindRefused):
        binder.verify_input_root(out, "fixed")


def test_failed_validation_and_receipt_reopen_leave_no_completion_authority(
        authority: SimpleNamespace) -> None:
    out = authority.source.parent / "invalid-cell"
    real_validate = binder.validate_extraction_run
    authority.monkeypatch.setattr(
        binder, "validate_extraction_run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            binder.ExtractionInvalid("injected validation failure")))
    with pytest.raises(binder.ExtractionInvalid):
        binder.materialize(authority.source, out, "fixed")
    assert not (out / CELL / "extraction_complete.json").exists()
    assert not (out / binder.INPUT_RECEIPT_NAME).exists()
    authority.monkeypatch.setattr(
        binder, "validate_extraction_run", real_validate)

    out = authority.source.parent / "invalid-receipt"
    authority.monkeypatch.setattr(
        binder, "verify_input_root",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            binder.BindRefused("injected receipt reopen failure")))
    with pytest.raises(binder.BindRefused, match="receipt reopen failure"):
        binder.materialize(authority.source, out, "fixed")
    assert not (out / binder.INPUT_RECEIPT_NAME).exists()


def test_final_receipt_reopen_detects_changed_bytes(
        authority: SimpleNamespace) -> None:
    out = authority.source.parent / "receipt-race"
    binder.materialize(authority.source, out, "fixed")
    original = binder._bound_json

    def changed(path: Path, what: str):
        payload, digest = original(path, what)
        if what == "Phase-2 input receipt final reopen":
            digest = "0" * 64
        return payload, digest

    authority.monkeypatch.setattr(binder, "_bound_json", changed)
    with pytest.raises(binder.BindRefused, match="changed during verification"):
        binder.verify_input_root(out, "fixed")


def test_verify_refuses_support_symlink_retarget_during_read(
        authority: SimpleNamespace) -> None:
    out = authority.source.parent / "link-race"
    binder.materialize(authority.source, out, "fixed")
    real_readlink = binder.os.readlink
    calls = 0

    def changing(path):
        nonlocal calls
        value = real_readlink(path)
        if Path(path).name == "model_state_dict.pth":
            calls += 1
            if calls == 2:
                return value + ".retargeted"
        return value

    authority.monkeypatch.setattr(binder.os, "readlink", changing)
    with pytest.raises(binder.BindRefused, match="symlink target changed"):
        binder.verify_input_root(out, "fixed")


def test_verify_cli_uses_the_same_authority_reader(
        authority: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    out = authority.source.parent / "cli"
    binder.materialize(authority.source, out, "legacy")
    monkeypatch.setattr(
        sys, "argv", [binder.__file__, "--side", "legacy",
                      "--verify-root", str(out)])
    assert binder.main() == 0
