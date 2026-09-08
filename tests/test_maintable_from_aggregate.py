"""F15: the main table may only be generated from an admissible aggregate.

The table was assembled by hand, so a diagnostic-only number could reach the
paper with nothing in the file recording that the matrix behind it was never
admissible. These tests pin the two properties that close that path: the
generator refuses when `paper_table_admission.eligible` is false and writes NO
file, and it never silently substitutes `diagnostic_mean_map_at_R_post` for the
eligible `mean_map_at_R_post` the aggregator deliberately leaves null.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

import pytest
import torch
import scripts.build_maintable_from_aggregate as maintable_script
from baseline.execution_environment import canonical_digest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from scripts.build_maintable_from_aggregate import (  # noqa: E402
    AGGREGATE_PRODUCER_SOURCE_FILES,
    AGGREGATE_PRODUCER_SOURCE_RELATIVE_PATHS,
    AGGREGATOR_SOURCE,
    AGGREGATOR_SOURCE_RELATIVE,
    BIO_PROJECTION_TIE_POLICY,
    BIO_RANKING_TIE_POLICY,
    CURRENT_AGGREGATE_SCHEMA_VERSION,
    COLUMN_ORDER,
    NotPaperEligible,
    ROW_ORDER,
    admission_failure_reasons,
    render_tex,
    _source_closure_digest,
)

_SCRIPT = os.path.join(_REPO, "scripts", "build_maintable_from_aggregate.py")
_AGGREGATOR = os.path.join(
    _REPO, "scripts", "aggregate_baseline_p0_matrix.py")


def _execution_environment() -> tuple[dict[str, object], str]:
    distributions = {
        "torch": "torch", "transformers": "transformers",
        "numpy": "numpy", "scipy": "scipy",
        "sklearn": "scikit-learn", "pandas": "pandas",
        "timm": "timm", "tensorboard": "tensorboard",
        "tqdm": "tqdm", "Pillow": "Pillow",
    }
    environment = {
        "schema": "groundeddna.baseline-child-execution-environment",
        "schema_version": 1,
        "python": {
            "executable": "/synthetic/python",
            "executable_sha256": "1" * 64,
            "implementation": "CPython", "version": "3.test",
            "version_info": [3, 11, 0, "final", 0],
        },
        "packages": {
            key: {"distribution": value, "installed": True,
                  "version": "test-version"}
            for key, value in distributions.items()
        },
        "cuda_runtime": {
            "torch_version": "test-version", "torch_cuda": "test",
            "cudnn_version": 1},
        "cuda_visible_devices": "GPU-TEST-0000",
        "torch_visible_device_count": 1,
        "logical_device_0": {
            "logical_index": 0, "name": "Synthetic GPU",
            "total_memory": 1, "major": 1, "minor": 0,
            "multi_processor_count": 1, "uuid": "GPU-TEST-0000",
        },
        "assigned_physical_gpu": {
            "physical_index": 0, "uuid": "GPU-TEST-0000",
            "name": "Synthetic GPU", "pci_bus_id": "0000:01:00.0",
            "driver_version": "test",
        },
    }
    return environment, canonical_digest(environment)


def _source_sha256() -> str:
    return hashlib.sha256(AGGREGATOR_SOURCE.read_bytes()).hexdigest()


def _source_closure_sha256() -> dict[str, str]:
    return {
        relative: hashlib.sha256(
            AGGREGATE_PRODUCER_SOURCE_FILES[relative].read_bytes()).hexdigest()
        for relative in AGGREGATE_PRODUCER_SOURCE_RELATIVE_PATHS
    }


def _record_key(variant: str, dataset: str, seed: int) -> str:
    return f"u0/{variant}/{dataset}/30b/seed{seed}"


def _decode_audit(dataset: str) -> dict[str, object]:
    if dataset == "MSCOCO":
        count = 16
        indices = list(maintable_script.MSCOCO_EXPECTED_FAILED_IMAGE_INDICES)
        image_ids = list(maintable_script.MSCOCO_EXPECTED_FAILED_IMAGE_IDS)
        indices_sha = maintable_script.MSCOCO_EXPECTED_FAILED_INDICES_SHA256
        image_ids_sha = (
            maintable_script.MSCOCO_EXPECTED_FAILED_IMAGE_IDS_SHA256)
        policy = "kept_and_disclosed_db_only"
        sensitivity = "later exclusion sensitivity; main inputs keep these rows"
        membership = {
            "train_count": 0, "query_count": 0, "database_count": 16,
            "database_only_count": 16, "unassigned_count": 0,
        }
    else:
        count = 0
        indices = []
        image_ids = []
        indices_sha = maintable_script.EMPTY_FAILED_INDICES_SHA256
        image_ids_sha = maintable_script.EMPTY_FAILED_IMAGE_IDS_SHA256
        policy = "no_decode_failures"
        sensitivity = "not_applicable"
        membership = {
            "train_count": 0, "query_count": 0, "database_count": 0,
            "database_only_count": 0, "unassigned_count": 0,
        }
    membership_sources = (
        {} if dataset == "CIFAR10" else {
            f"{dataset}/setting1/{name}.txt": "9" * 64
            for name in ("train", "test", "database")
        })
    return {
        "schema": "groundeddna.baseline-cache-decode-audit",
        "schema_version": 1,
        "status": "verified",
        "dataset": dataset,
        "setting": "setting1",
        "cache_dir": f"/synthetic-cache/{dataset}",
        "cache_meta_sha256": "7" * 64,
        "cache_image_ids_sha256": "8" * 64,
        "source_encoding": (
            "absent_means_zero_legacy_cifar_extractor"
            if dataset == "CIFAR10" else "explicit_meta_fields"),
        "policy": policy,
        "sensitivity_analysis": sensitivity,
        "count": count,
        "failed_image_indices": indices,
        "failed_indices_sha256": indices_sha,
        "failed_image_ids": image_ids,
        "failed_image_ids_sha256": image_ids_sha,
        "official_membership": membership,
        "official_train_query_overlap_count": 0,
        "membership_source_sha256": membership_sources,
    }


def _aggregate(*, eligible: bool, mean=0.7912, std=0.0021,
               diagnostic=False):
    """Construct a complete current-schema D6 U0 payload from records."""
    seeds = [42, 43, 44]
    offsets = (-std, 0.0, std)
    records = []
    aggregates = []
    execution_environment, execution_environment_sha256 = (
        _execution_environment())
    for variant in ROW_ORDER:
        for dataset in COLUMN_ORDER:
            keys = []
            values = []
            for seed, offset in zip(seeds, offsets):
                key = _record_key(variant, dataset, seed)
                value = mean + offset
                keys.append(key)
                values.append(value)
                records.append({
                    "key": key,
                    "panel": "u0",
                    "variant": variant,
                    "dataset": dataset,
                    "bit": 30,
                    "seed": seed,
                    "status": "complete_main_eligible",
                    "protocol_mode": "author_fixed_final",
                    "protocol_stage": "author_fixed_single_stage",
                    "protocol_digest_sha256": "b" * 64,
                    "checkpoint_protocol": {
                        "protocol_mode": "author_fixed_final",
                        "protocol_stage": "author_fixed_single_stage",
                        "protocol_identity_sha256": "b" * 64,
                    },
                    "final_checkpoint_sha256": "c" * 64,
                    "declared_main_protocol_eligible": True,
                    "main_protocol_eligible": True,
                    "implementation_comparison_eligible": True,
                    "paper_table_eligible": True,
                    "legacy_cache_diagnostic_only": False,
                    "validation_errors": [],
                    "cache_decode_failure_audit": _decode_audit(dataset),
                    "execution_environment": execution_environment,
                    "execution_environment_sha256": (
                        execution_environment_sha256),
                    "map_at_R_post": value,
                    "manifest": f"nonexistent/{variant}-{dataset}-{seed}.json",
                    "manifest_sha256": "a" * 64,
                    "bio_projection_manifest": (
                        f"nonexistent/bio-{variant}-{dataset}-{seed}.json"),
                    "bio_projection_manifest_sha256": "d" * 64,
                })
            cell_mean = statistics.mean(values)
            cell_std = statistics.stdev(values)
            aggregates.append({
                "panel": "u0",
                "variant": variant,
                "dataset": dataset,
                "bit": 30,
                "seeds_expected": seeds,
                "seeds_complete": seeds,
                "all_seeds_complete": True,
                "aggregate_status": "complete_paper_table_eligible",
                "all_records_main_protocol_eligible": True,
                "all_records_paper_table_eligible": True,
                "paper_table_eligible": True,
                "diagnostic_only": False,
                "mean_map_at_R_post": cell_mean,
                "sample_std_map_at_R_post": cell_std,
                "admitted_comparison_mean_map_at_R_post": cell_mean,
                "admitted_comparison_sample_std_map_at_R_post": cell_std,
                "diagnostic_mean_map_at_R_post": None,
                "diagnostic_sample_std_map_at_R_post": None,
                "records": keys,
            })

    expected = len(records)
    source_closure = _source_closure_sha256()
    payload = {
        "schema_version": CURRENT_AGGREGATE_SCHEMA_VERSION,
        "producer": {
            "source_path": AGGREGATOR_SOURCE_RELATIVE,
            "source_sha256": _source_sha256(),
            "source_closure_sha256": source_closure,
            "source_closure_digest_sha256": _source_closure_digest(
                source_closure),
        },
        "protocol": {
            "protocol_mode": "author_fixed_final",
            "final_checkpoint_protocol_stage": "author_fixed_single_stage",
            "checkpoint_policy": "author_horizon_last",
            "requested_bit_slice": [30],
            "train_seeds": seeds,
            "comparison_panels": ["u0"],
            "validation_seed": None,
            "validation_ratio": 0.0,
            "selection_metric_by_bit": None,
            "mandatory_bio_projection": True,
            "three_seed_mean_std_status": "candidate_aggregate",
        },
        "paper_table_admission": {
            "eligible": True,
            "reason_codes": [],
            "required_unique_train_seeds": 3,
            "observed_unique_train_seeds": 3,
            "eligible_records": expected,
            "expected_records": expected,
        },
        "implementation_audit": {
            "comparison_safe": True,
            "blocked_cells": {},
        },
        "summary": {
            "expected": expected,
            "complete": expected,
            "complete_main_eligible": expected,
            "complete_diagnostic_only": 0,
            "implementation_comparison_blocked": 0,
            "paper_table_eligible": expected,
            "missing": 0,
            "invalid": 0,
            "duplicate": 0,
            "malformed": 0,
            "extraneous": 0,
            "source_profile_excluded": 0,
        },
        "records": records,
        "aggregates": aggregates,
        "missing_cells": [],
        "invalid_cells": [],
        "duplicate_cells": [],
        "duplicate_manifests": {},
        "malformed_manifests": [],
        "extraneous_manifests": [],
        "source_profile_excluded_manifests": [],
    }
    if not eligible:
        record = records[0]
        record.update({
            "status": "complete_diagnostic_only",
            "declared_main_protocol_eligible": False,
            "main_protocol_eligible": False,
            "paper_table_eligible": False,
            "legacy_cache_diagnostic_only": True,
        })
        entry = aggregates[0]
        entry.update({
            "aggregate_status": "complete_diagnostic_only",
            "all_records_main_protocol_eligible": False,
            "all_records_paper_table_eligible": False,
            "paper_table_eligible": False,
            "diagnostic_only": True,
            "mean_map_at_R_post": None,
            "sample_std_map_at_R_post": None,
            "diagnostic_mean_map_at_R_post": mean,
            "diagnostic_sample_std_map_at_R_post": std,
        })
        payload["summary"].update({
            "complete_main_eligible": expected - 1,
            "complete_diagnostic_only": 1,
            "paper_table_eligible": expected - 1,
        })
        payload["paper_table_admission"].update({
            "eligible": False,
            "reason_codes": ["not_all_records_paper_table_eligible"],
            "eligible_records": expected - 1,
        })
    return payload


def _write(tmp_path, payload):
    manifest_root = tmp_path / "record-manifests"
    manifest_root.mkdir(exist_ok=True)
    for index, record in enumerate(payload.get("records", [])):
        protocol_identity = {
            "variant": record.get("variant"),
            "dataset": record.get("dataset"),
            "bit": record.get("bit"),
            "seed": record.get("seed"),
            "cache_decode_failure_audit": record.get(
                "cache_decode_failure_audit"),
            "execution_environment": record.get("execution_environment"),
            "execution_environment_sha256": record.get(
                "execution_environment_sha256"),
        }
        protocol_digest = hashlib.sha256(json.dumps(
            protocol_identity, sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        record["protocol_digest_sha256"] = protocol_digest
        record["checkpoint_protocol"][
            "protocol_identity_sha256"] = protocol_digest

        artifact_root = manifest_root / f"artifacts-{index:04d}"
        artifact_root.mkdir(exist_ok=True)
        checkpoint_path = artifact_root / "epoch_059.pth"
        torch.save({
            "config": {
                "protocol_mode": "author_fixed_final",
                "protocol_stage": "author_fixed_single_stage",
                "protocol_identity_sha256": protocol_digest,
                "execution_environment": record.get(
                    "execution_environment"),
                "execution_environment_sha256": record.get(
                    "execution_environment_sha256"),
            },
            "model_state_dict": {},
        }, checkpoint_path)
        record["final_checkpoint_sha256"] = hashlib.sha256(
            checkpoint_path.read_bytes()).hexdigest()
        input_records = {}
        output_records = {}
        for split in ("db", "query"):
            input_path = artifact_root / f"extract_{split}.npz"
            output_path = artifact_root / f"extract_{split}_bioproj.npz"
            input_path.write_bytes(f"input-{index}-{split}".encode())
            output_path.write_bytes(f"output-{index}-{split}".encode())
            input_records[split] = {
                "path": str(input_path.resolve()),
                "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
            }
            output_records[split] = {
                "path": str(output_path.resolve()),
                "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
            }

        bio_manifest = manifest_root / f"bio-{index:04d}.json"
        bio_payload = {
            "config": {
                "expected_length": 15,
                "gc_min_frac": 0.4,
                "gc_max_frac": 0.6,
                "max_homopolymer_run": 3,
                "projection_tie_policy": BIO_PROJECTION_TIE_POLICY,
                "ranking_tie_policy": BIO_RANKING_TIE_POLICY,
            },
            "cells": [{
                "name": record.get("variant"),
                "dataset": record.get("dataset"),
                "L": 15,
                "paper_result_eligible": True,
                "projection_invariant_satisfied": True,
                "post_compliance_db": 1.0,
                "post_compliance_qy": 1.0,
                "projection_failures_db": 0,
                "projection_failures_qy": 0,
                "projection_tie_policy": BIO_PROJECTION_TIE_POLICY,
                "ranking_tie_policy": BIO_RANKING_TIE_POLICY,
                "map_at_R_post": record.get("map_at_R_post"),
                "input_artifacts": input_records,
                "output_artifacts": output_records,
                "training_provenance": {
                    "protocol_mode": "author_fixed_final",
                    "protocol_stage": "author_fixed_single_stage",
                    "protocol_identity_sha256": protocol_digest,
                    "cache_decode_failure_audit": record.get(
                        "cache_decode_failure_audit"),
                    "execution_environment": record.get(
                        "execution_environment"),
                    "execution_environment_sha256": record.get(
                        "execution_environment_sha256"),
                    "checkpoint": {
                        "sha256": record.get("final_checkpoint_sha256"),
                        "checkpoint_protocol_mode": "author_fixed_final",
                        "checkpoint_protocol_stage": (
                            "author_fixed_single_stage"),
                        "protocol_identity_sha256": protocol_digest,
                        "execution_environment": record.get(
                            "execution_environment"),
                        "execution_environment_sha256": record.get(
                            "execution_environment_sha256"),
                    },
                },
            }],
        }
        bio_manifest.write_text(json.dumps(bio_payload), encoding="utf-8")
        bio_sha = hashlib.sha256(bio_manifest.read_bytes()).hexdigest()
        record["bio_projection_manifest"] = str(bio_manifest.resolve())
        record["bio_projection_manifest_sha256"] = bio_sha
        sidecar = manifest_root / f"bio-{index:04d}.json.sha256"
        sidecar.write_text(f"{bio_sha}  {bio_manifest.name}\n", encoding="utf-8")

        raw = record.get("manifest")
        existing = Path(raw).expanduser() if isinstance(raw, str) else None
        if existing is not None and existing.is_file():
            continue
        manifest = manifest_root / f"record-{index:04d}.json"
        manifest_payload = {
            "protocol_mode": record.get("protocol_mode"),
            "training_stage": record.get("protocol_stage"),
            "protocol_digest_sha256": record.get("protocol_digest_sha256"),
            "checkpoint_protocol": record.get("checkpoint_protocol"),
            "final_checkpoint_sha256": record.get("final_checkpoint_sha256"),
            "final_checkpoint": str(checkpoint_path.resolve()),
            "protocol_identity": protocol_identity,
            "cache_decode_failure_audit": record.get(
                "cache_decode_failure_audit"),
            "execution_environment": record.get("execution_environment"),
            "execution_environment_sha256": record.get(
                "execution_environment_sha256"),
            "bio_projection_manifest": str(bio_manifest.resolve()),
            "bio_projection_manifest_sha256": bio_sha,
            "bio_projection_sidecar": str(sidecar.resolve()),
            "bio_projection_sidecar_sha256": hashlib.sha256(
                sidecar.read_bytes()).hexdigest(),
            "extraction_artifact_sha256": {
                f"extract_{split}.npz": input_records[split]["sha256"]
                for split in ("db", "query")
            },
            "bio_projected_artifacts": output_records,
        }
        manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
        record["manifest"] = str(manifest.resolve())
        record["manifest_sha256"] = hashlib.sha256(
            manifest.read_bytes()).hexdigest()
    path = tmp_path / "aggregate.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ------------------------------------------------------------------ refusal

def test_ineligible_aggregate_is_refused(tmp_path):
    payload = _aggregate(eligible=False, diagnostic=True)
    with pytest.raises(NotPaperEligible):
        render_tex(payload, source=_write(tmp_path, payload))


def test_refusal_writes_no_file(tmp_path):
    """A partial or stale TeX left behind would be worse than none."""
    src = _write(tmp_path, _aggregate(eligible=False, diagnostic=True))
    out = tmp_path / "table.tex"
    proc = subprocess.run(
        [sys.executable, _SCRIPT, str(src), "--out", str(out)],
        capture_output=True, text=True, cwd=_REPO, timeout=120)
    assert proc.returncode == 3
    assert not out.exists()
    assert "REFUSED" in proc.stderr


def test_reason_codes_are_reported(tmp_path):
    payload = _aggregate(eligible=False, diagnostic=True)
    reasons = admission_failure_reasons(payload)
    assert reasons[0] == "not_all_records_paper_table_eligible"
    assert any("main_protocol_eligible" in reason for reason in reasons)


def test_missing_admission_block_is_a_refusal():
    assert admission_failure_reasons({"aggregates": []})


# ------------------------------------------------------- eligible generation

def test_eligible_aggregate_renders_the_value(tmp_path):
    payload = _aggregate(eligible=True)
    assert admission_failure_reasons(payload) == []
    tex = render_tex(payload, source=_write(tmp_path, payload))
    assert "0.7912 \\pm 0.0021" in tex
    assert "CIBHash" in tex


@pytest.mark.parametrize("mutation", [
    lambda record: record.pop("execution_environment"),
    lambda record: record["execution_environment"]["packages"]["numpy"]
    .__setitem__("version", "forged-package-drift"),
])
def test_execution_environment_missing_or_package_drift_is_refused(mutation):
    payload = _aggregate(eligible=True)
    mutation(payload["records"][0])
    reasons = admission_failure_reasons(payload)
    assert any("execution environment" in reason for reason in reasons)


def test_reopened_checkpoint_content_tamper_is_refused(tmp_path):
    payload = _aggregate(eligible=True)
    source = _write(tmp_path, payload)
    run_manifest = json.loads(Path(
        payload["records"][0]["manifest"]).read_text(encoding="utf-8"))
    Path(run_manifest["final_checkpoint"]).write_bytes(b"tampered-checkpoint")
    reasons = admission_failure_reasons(payload, source=source)
    assert any("checkpoint" in reason and "mismatch" in reason
               for reason in reasons)


def test_provenance_is_stamped(tmp_path):
    payload = _aggregate(eligible=True)
    src = _write(tmp_path, payload)
    tex = render_tex(payload, source=src)
    assert str(src) in tex
    assert "source_sha256:" in tex
    assert "[42, 43, 44]" in tex


def test_missing_cells_render_as_a_dash(tmp_path):
    payload = _aggregate(eligible=False, diagnostic=True)
    payload["aggregates"] = payload["aggregates"][:-1]
    tex = render_tex(
        payload, source=_write(tmp_path, payload), allow_diagnostic=True)
    mscoco_row = [line for line in tex.splitlines() if line.startswith("CIBHash")][0]
    assert "NOT PAPER-ELIGIBLE" in tex
    assert any("-" in line for line in tex.splitlines() if "CroVCA" in line)


def test_geometry_header_comes_from_the_declared_panel(tmp_path):
    payload = _aggregate(eligible=True)
    tex = render_tex(payload, source=_write(tmp_path, payload))
    assert "30 bit / 15 bases (5x3)" in tex


# --------------------------------------------- diagnostic must stay visible

def test_diagnostic_mean_is_never_substituted_silently(tmp_path):
    """`mean_map_at_R_post` is null exactly when the cell is diagnostic-only.
    Falling back to the diagnostic field without being asked would typeset a
    number the aggregator refused to certify."""
    payload = _aggregate(eligible=False, diagnostic=True)
    with pytest.raises(NotPaperEligible):
        render_tex(payload, source=_write(tmp_path, payload))


def test_allow_diagnostic_stamps_every_row(tmp_path):
    payload = _aggregate(eligible=False, diagnostic=True)
    tex = render_tex(payload, source=_write(tmp_path, payload),
                     allow_diagnostic=True)
    assert "0.7912" in tex
    assert "NOT PAPER-ELIGIBLE" in tex
    assert r"$^{\dagger}$" in tex


def test_allow_diagnostic_exits_zero(tmp_path):
    src = _write(tmp_path, _aggregate(eligible=False, diagnostic=True))
    out = tmp_path / "table.tex"
    proc = subprocess.run(
        [sys.executable, _SCRIPT, str(src), "--out", str(out),
         "--allow-diagnostic"],
        capture_output=True, text=True, cwd=_REPO, timeout=120)
    assert proc.returncode == 0
    assert "NOT PAPER-ELIGIBLE" in out.read_text(encoding="utf-8")


# ------------------------------------------------ adversarial reconstruction

def test_forged_true_with_empty_records_and_ineligible_aggregate_is_refused(tmp_path):
    payload = _aggregate(eligible=True)
    payload["records"] = []
    payload["aggregates"][0].update({
        "paper_table_eligible": False,
        "mean_map_at_R_post": 0.9999,
    })
    reasons = admission_failure_reasons(payload)
    assert any("records U0 key set mismatch" in reason for reason in reasons)
    with pytest.raises(NotPaperEligible):
        render_tex(payload, source=_write(tmp_path, payload))


def test_forged_top_level_true_cannot_promote_ineligible_record(tmp_path):
    payload = _aggregate(eligible=True)
    payload["records"][0].update({
        "status": "complete_diagnostic_only",
        "main_protocol_eligible": False,
        "paper_table_eligible": False,
    })
    reasons = admission_failure_reasons(payload)
    assert any("status is not complete_main_eligible" in reason for reason in reasons)
    assert any("main_protocol_eligible is not true" in reason for reason in reasons)
    with pytest.raises(NotPaperEligible):
        render_tex(payload, source=_write(tmp_path, payload))


def test_record_key_identity_and_finite_metric_are_rebuilt(tmp_path):
    identity_forgery = _aggregate(eligible=True)
    identity_forgery["records"][0]["dataset"] = "MSCOCO"
    assert any(
        "identity fields do not match" in reason
        for reason in admission_failure_reasons(identity_forgery))

    metric_forgery = _aggregate(eligible=True)
    metric_forgery["records"][0]["map_at_R_post"] = math.inf
    assert any(
        "map_at_R_post is not finite" in reason
        for reason in admission_failure_reasons(metric_forgery))


def test_record_protocol_mode_stage_digest_contract_is_rebuilt():
    mutations = (
        ("protocol_stage", "P0_stage2_refit_test", "protocol_stage"),
        ("protocol_digest_sha256", "0" * 64, "checkpoint_protocol"),
        ("checkpoint_protocol", {}, "checkpoint_protocol"),
        ("final_checkpoint_sha256", "tampered", "final_checkpoint_sha256"),
    )
    for field, value, needle in mutations:
        payload = _aggregate(eligible=True)
        payload["records"][0][field] = value
        assert any(
            needle in reason for reason in admission_failure_reasons(payload)
        ), (field, admission_failure_reasons(payload))


def test_reopened_manifest_protocol_contract_mismatch_is_refused(tmp_path):
    payload = _aggregate(eligible=True)
    source = _write(tmp_path, payload)
    record = payload["records"][0]
    manifest = Path(record["manifest"])
    reopened = json.loads(manifest.read_text(encoding="utf-8"))
    reopened["training_stage"] = "P0_stage2_refit_test"
    manifest.write_text(json.dumps(reopened), encoding="utf-8")
    record["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    reasons = admission_failure_reasons(payload, source=source)
    assert any(
        "reopened manifest training_stage disagrees" in reason
        for reason in reasons)


def test_forged_record_metric_with_genuine_manifests_is_refused(tmp_path):
    payload = _aggregate(eligible=True)
    source = _write(tmp_path, payload)
    payload["records"][0]["map_at_R_post"] += 0.01
    first_keys = payload["aggregates"][0]["records"]
    by_key = {record["key"]: record for record in payload["records"]}
    values = [by_key[key]["map_at_R_post"] for key in first_keys]
    mean = statistics.mean(values)
    std = statistics.stdev(values)
    payload["aggregates"][0].update({
        "mean_map_at_R_post": mean,
        "sample_std_map_at_R_post": std,
        "admitted_comparison_mean_map_at_R_post": mean,
        "admitted_comparison_sample_std_map_at_R_post": std,
    })
    reasons = admission_failure_reasons(payload, source=source)
    assert any("bio cell map_at_R_post mismatch" in reason for reason in reasons)


def test_forged_record_decode_audit_with_genuine_manifests_is_refused(tmp_path):
    payload = _aggregate(eligible=True)
    source = _write(tmp_path, payload)
    payload["records"][0]["cache_decode_failure_audit"]["policy"] = "forged"
    reasons = admission_failure_reasons(payload, source=source)
    assert any(
        "cache_decode_failure_audit disagrees" in reason
        or "cache decode audit mismatch" in reason
        or "bio provenance cache_decode_failure_audit mismatch" in reason
        for reason in reasons)


def test_bio_source_artifact_or_sidecar_tamper_is_refused(tmp_path):
    payload = _aggregate(eligible=True)
    source = _write(tmp_path, payload)
    record = payload["records"][0]
    bio = json.loads(Path(record["bio_projection_manifest"]).read_text(
        encoding="utf-8"))
    input_path = Path(bio["cells"][0]["input_artifacts"]["db"]["path"])
    input_path.write_bytes(b"tampered-source-npz")
    reasons = admission_failure_reasons(payload, source=source)
    assert any("bio input db artifact SHA mismatch" in reason for reason in reasons)

    run_manifest = json.loads(Path(record["manifest"]).read_text(
        encoding="utf-8"))
    sidecar = Path(run_manifest["bio_projection_sidecar"])
    sidecar.write_text("0" * 64 + "  bio.json\n", encoding="utf-8")
    reasons = admission_failure_reasons(payload, source=source)
    assert any("bio sidecar SHA mismatch" in reason for reason in reasons)


def test_missing_record_manifest_cannot_be_paper_admitted(tmp_path):
    payload = _aggregate(eligible=True)
    source = _write(tmp_path, payload)
    Path(payload["records"][0]["manifest"]).unlink()
    reasons = admission_failure_reasons(payload, source=source)
    assert any("manifest cannot be reopened" in reason for reason in reasons)


@pytest.mark.parametrize("field", [
    "mean_map_at_R_post",
    "sample_std_map_at_R_post",
    "admitted_comparison_mean_map_at_R_post",
    "admitted_comparison_sample_std_map_at_R_post",
])
def test_aggregate_mean_and_sample_std_are_recomputed(field):
    payload = _aggregate(eligible=True)
    payload["aggregates"][0][field] += 1e-6
    reasons = admission_failure_reasons(payload)
    assert any(field in reason and "record recomputation" in reason
               for reason in reasons)


def test_aggregate_record_keys_status_and_eligibility_are_recomputed():
    payload = _aggregate(eligible=True)
    aggregate = payload["aggregates"][0]
    aggregate["records"] = list(reversed(aggregate["records"]))
    aggregate["aggregate_status"] = "complete_diagnostic_only"
    aggregate["paper_table_eligible"] = False
    reasons = admission_failure_reasons(payload)
    assert any("records does not match records" in reason for reason in reasons)
    assert any("aggregate_status does not match records" in reason for reason in reasons)
    assert any("paper_table_eligible does not match records" in reason
               for reason in reasons)


def test_summary_and_admission_counts_are_recomputed():
    payload = _aggregate(eligible=True)
    payload["summary"]["complete"] = 0
    payload["paper_table_admission"]["eligible_records"] = 0
    reasons = admission_failure_reasons(payload)
    assert "summary.complete does not match record reconstruction" in reasons
    assert any("eligible_records" in reason for reason in reasons)


@pytest.mark.parametrize(("path", "value", "needle"), [
    (("schema_version",), 4, "schema_version"),
    (("protocol", "protocol_mode"), "validation_sensitivity", "protocol_mode"),
    (("protocol", "requested_bit_slice"), [36], "requested_bit_slice"),
    (("protocol", "train_seeds"), [42], "fewer_than_three_train_seeds"),
])
def test_current_d6_schema_mode_bit_and_seed_contract(path, value, needle):
    payload = _aggregate(eligible=True)
    target = payload
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = value
    assert any(needle in reason for reason in admission_failure_reasons(payload))


def test_aggregator_source_sha_seal_is_required_and_current():
    payload = _aggregate(eligible=True)
    assert admission_failure_reasons(payload) == []
    payload["producer"]["source_sha256"] = "0" * 64
    assert any(
        "producer.source_sha256" in reason
        for reason in admission_failure_reasons(payload))


def test_producer_source_closure_rejects_forged_dependency_digest():
    payload = _aggregate(eligible=True)
    closure = payload["producer"]["source_closure_sha256"]
    closure["baseline/cache_provenance.py"] = "0" * 64
    # Even an attacker who recomputes the self-consistent map digest cannot
    # make edited validator bytes equal the independently reopened source.
    payload["producer"]["source_closure_digest_sha256"] = (
        _source_closure_digest(closure))
    reasons = admission_failure_reasons(payload)
    assert any("source_closure_sha256" in reason for reason in reasons)


def test_local_validator_bytes_are_inside_the_sealed_producer(tmp_path, monkeypatch):
    source_text = AGGREGATOR_SOURCE.read_text(encoding="utf-8")
    assert "from scripts.run_modern_baseline_p0 import" not in source_text
    assert "def _verify_dataset_source_binding" in source_text
    payload = _aggregate(eligible=True)
    tampered = tmp_path / "aggregate_baseline_p0_matrix.py"
    tampered.write_text(
        source_text + "\n# forged validator-source closure\n", encoding="utf-8")
    monkeypatch.setattr(maintable_script, "AGGREGATOR_SOURCE", tampered)
    reasons = maintable_script.admission_failure_reasons(payload)
    assert any("producer.source_sha256" in reason for reason in reasons)

    cache_source = AGGREGATE_PRODUCER_SOURCE_FILES[
        "baseline/cache_provenance.py"]
    forged_cache_source = tmp_path / "cache_provenance.py"
    forged_cache_source.write_bytes(
        cache_source.read_bytes() + b"\n# forged decode validator\n")
    monkeypatch.setitem(
        maintable_script.AGGREGATE_PRODUCER_SOURCE_FILES,
        "baseline/cache_provenance.py", forged_cache_source)
    reasons = maintable_script.admission_failure_reasons(payload)
    assert any("source_closure_sha256" in reason for reason in reasons)


def test_producer_emits_current_schema_and_own_source_seal(tmp_path):
    empty_results = tmp_path / "empty-results"
    empty_results.mkdir()
    aggregate_json = tmp_path / "produced.json"
    aggregate_markdown = tmp_path / "produced.md"
    proc = subprocess.run([
        sys.executable, _AGGREGATOR,
        "--protocol-mode", "author_fixed_final",
        "--panels", "u0",
        "--bits", "30",
        "--seeds", "42", "43", "44",
        "--result-root", str(empty_results),
        "--out-json", str(aggregate_json),
        "--out-markdown", str(aggregate_markdown),
        "--no-verify-artifact-hashes",
    ], capture_output=True, text=True, cwd=_REPO, timeout=120)
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(aggregate_json.read_text(encoding="utf-8"))
    assert payload["schema_version"] == CURRENT_AGGREGATE_SCHEMA_VERSION
    source_closure = _source_closure_sha256()
    assert payload["producer"] == {
        "source_path": AGGREGATOR_SOURCE_RELATIVE,
        "source_sha256": _source_sha256(),
        "source_closure_sha256": source_closure,
        "source_closure_digest_sha256": _source_closure_digest(
            source_closure),
    }


def test_type_coercions_and_malformed_aggregate_identity_fail_closed():
    payload = _aggregate(eligible=True)
    payload["protocol"]["requested_bit_slice"] = [30.0]
    payload["summary"]["missing"] = False
    payload["aggregates"][0]["variant"] = {"forged": True}
    reasons = admission_failure_reasons(payload)
    assert any("requested_bit_slice" in reason for reason in reasons)
    assert any("summary.missing" in reason for reason in reasons)
    assert any("identity is invalid" in reason for reason in reasons)


def test_existing_manifest_is_reopened_and_content_hash_verified(tmp_path):
    payload = _aggregate(eligible=True)
    manifest = tmp_path / "run.json"
    manifest.write_text("{}\n", encoding="utf-8")
    payload["records"][0]["manifest"] = str(manifest)
    payload["records"][0]["manifest_sha256"] = "0" * 64
    reasons = admission_failure_reasons(
        payload, source=_write(tmp_path, payload))
    assert any("manifest content SHA-256 mismatch" in reason for reason in reasons)
