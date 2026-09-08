import json
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

import torch

from baseline.cache_provenance import audit_cache_decode_failures
from baseline.execution_environment import canonical_digest
from scripts.aggregate_baseline_p0_matrix import (
    CANONICAL_VARIANT_SOURCE_PROFILES,
    COMMON_REQUIRED_IMPLEMENTATION_PATHS as AGGREGATE_COMMON_SOURCE_PATHS,
    KNOWN_NON_SCIENTIFIC_IMPLEMENTATION_TRANSITIONS,
    Key,
    _apply_implementation_admission,
    _aggregate_cell,
    _audit_implementation_fingerprints,
    _implementation_fingerprint,
    _implementation_snapshot,
    _protocol_identity_digest as aggregate_protocol_identity_digest,
    _required_implementation_paths as aggregate_required_implementation_paths,
    _source_profile_exclusion,
    _validate_manifest,
)
from scripts.run_baseline_p0_matrix import (
    CANONICAL_VARIANT_SOURCE_PROFILES as RUNNER_SOURCE_PROFILES,
    COMMON_REQUIRED_IMPLEMENTATION_PATHS as RUNNER_COMMON_SOURCE_PATHS,
    SUPERVISED_VARIANTS,
    U0_VARIANTS,
    _completed_keys,
    _current_source_sha256,
    _matches_canonical_source_profile,
    _protocol_identity_digest as runner_protocol_identity_digest,
    _required_implementation_paths as runner_required_implementation_paths,
)
from scripts.run_modern_baseline_p0 import _dataset_source_binding


REPO = Path(__file__).resolve().parents[1]


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


def _record(key: Key, snapshot: dict[str, str], *,
            main_eligible: bool = False,
            legacy_cache: bool = True) -> dict[str, object]:
    return {
        "key": key.text,
        "panel": key.panel,
        "variant": key.variant,
        "dataset": key.dataset,
        "bit": key.bit,
        "seed": key.seed,
        "status": (
            "complete_main_eligible" if main_eligible
            else "complete_diagnostic_only"
        ),
        "main_protocol_eligible": main_eligible,
        "legacy_cache_diagnostic_only": legacy_cache,
        "implementation_sha256": snapshot,
        "implementation_fingerprint_sha256": _implementation_fingerprint(
            snapshot),
        "map_at_R_post": 0.5,
    }


def _write_json(path: Path, payload: object) -> str:
    data = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
    path.write_bytes(data)
    return sha256(data).hexdigest()


def _write_artifact(path: Path, data: bytes) -> str:
    path.write_bytes(data)
    return sha256(data).hexdigest()


def _minimal_resume_fixture(
        root: Path, *, author_fixed: bool = False,
        ) -> tuple[Path, dict[str, object]]:
    """Create a complete, self-contained legacy or D6 manifest fixture."""
    variant = "cibhash"
    dataset = "CIFAR10"
    bit = 30 if author_fixed else 36
    base_length = bit // 2
    seed = 42
    cell_dir = root / "cell"
    cell_dir.mkdir(parents=True)

    semantic = {
        "information_tier": "U0",
        "term_source_condition": "visual-only",
        "matched_official_taxonomy_dataset": "",
        "selection_verified": None,
    }
    implementation = {
        path: _current_source_sha256(path)
        for path in runner_required_implementation_paths(variant)
    }
    if not all(isinstance(digest, str) for digest in implementation.values()):
        raise AssertionError("test fixture requires every implementation source")
    identity = {
        "variant": variant,
        "dataset": dataset,
        "setting": "setting1",
        "bit": bit,
        "seed": seed,
        "val_seed": None if author_fixed else 42,
        "val_ratio": 0.0 if author_fixed else 0.1,
        "eval_period": 60 if author_fixed else 5,
        "horizon": 60,
        "batch_size_override": None,
        "implementation_sha256": implementation,
        "semantic_information_condition": semantic,
    }
    decode_failure_audit = None
    execution_environment, execution_environment_sha256 = (
        _execution_environment())
    if author_fixed:
        source_root = root / "dataset"
        (source_root / "CIFAR10" / "cifar-10-batches-py").mkdir(
            parents=True)
        for index in range(1, 6):
            _write_artifact(
                source_root / "CIFAR10" / "cifar-10-batches-py"
                / f"data_batch_{index}", f"batch-{index}".encode())
        _write_artifact(
            source_root / "CIFAR10" / "cifar-10-batches-py" / "test_batch",
            b"test-batch")
        _write_artifact(
            source_root / "CIFAR10" / "cifar-10-batches-py" / "batches.meta",
            b"metadata")
        dataset_root, split_sha256 = _dataset_source_binding(
            dataset, "setting1", source_root)
        cache = root / "eligible-cache"
        cache.mkdir()
        _write_json(cache / "meta.json", {})
        _write_json(cache / "image_ids.json", ["cifar-row-0"])
        decode_failure_audit = audit_cache_decode_failures(
            cache, dataset=dataset, dataset_root=source_root)
        identity.update({
            "protocol_mode": "author_fixed_final",
            "trainer_protocol_stages": ["author_fixed_single_stage"],
            "final_checkpoint_protocol_stage": (
                "author_fixed_single_stage"),
            "checkpoint_policy": "author_horizon_last",
            "author_horizon": 60,
            "training_epochs": 60,
            "final_epoch_zero_based": 59,
            "validation_selection": False,
            "designated_train_scope": "full_designated_train",
            "dataset_root": dataset_root,
            "split_sha256": split_sha256,
            "cache_dir": str(cache.resolve()),
            "cache_meta_sha256": (
                decode_failure_audit["cache_meta_sha256"]),
            "cache_image_ids_sha256": (
                decode_failure_audit["cache_image_ids_sha256"]),
            "cache_decode_failure_audit": decode_failure_audit,
            "cache_artifact_sha256": {
                "visual_global.f16.npy": "3" * 64,
                "visual_global_aug0.f16.npy": "4" * 64,
                "visual_global_aug1.f16.npy": "5" * 64,
            },
            "execution_environment": execution_environment,
            "execution_environment_sha256": execution_environment_sha256,
        })
    protocol_digest = runner_protocol_identity_digest(identity)

    extraction_hashes = {
        name: _write_artifact(cell_dir / name, name.encode("utf-8"))
        for name in ("extract_db.npz", "extract_query.npz", "args_extract.txt")
    }
    projected_hashes = {
        name: _write_artifact(
            cell_dir / f"extract_{name}_bioproj.npz",
            f"projected-{name}".encode("utf-8"),
        )
        for name in ("db", "query")
    }
    checkpoint = root / ("epoch_059.pth" if author_fixed else "checkpoint.pth")
    checkpoint_config = {
        "protocol_identity_sha256": protocol_digest,
    }
    if author_fixed:
        checkpoint_config.update({
            "protocol_mode": "author_fixed_final",
            "protocol_stage": "author_fixed_single_stage",
            "execution_environment": execution_environment,
            "execution_environment_sha256": execution_environment_sha256,
        })
    torch.save({"config": checkpoint_config, "model_state_dict": {}}, checkpoint)
    checkpoint_sha = sha256(checkpoint.read_bytes()).hexdigest()
    selection = root / "selection.json"
    selection_sha = None if author_fixed else _write_json(selection, {})
    protocol_manifest = cell_dir / "p0_protocol_manifest.json"
    protocol_manifest_sha = None

    bio = {
        "config": {
            "expected_length": base_length,
            "max_homopolymer_run": 3,
            "gc_min_frac": 0.4,
            "gc_max_frac": 0.6,
        },
        "cells": [{
            "name": variant,
            "dataset": dataset,
            "L": base_length,
            "post_compliance_db": 1.0,
            "post_compliance_qy": 1.0,
            "projection_failures_db": 0,
            "projection_failures_qy": 0,
            "projection_invariant_satisfied": True,
            "gc_count_range": (
                [6, 9] if author_fixed else [8, 10]),
            "paper_result_eligible": author_fixed,
            "map_at_R_pre": 0.5,
            "map_at_R_post": 0.5,
            "dna_unique_pre": 0.5,
            "dna_unique_post": 0.5,
            "R": 1,
            "input_artifacts": {
                "db": {
                    "path": str((cell_dir / "extract_db.npz").resolve()),
                    "sha256": extraction_hashes["extract_db.npz"],
                },
                "query": {
                    "path": str((cell_dir / "extract_query.npz").resolve()),
                    "sha256": extraction_hashes["extract_query.npz"],
                },
            },
            "output_artifacts": {
                name: {
                    "path": str(
                        (cell_dir / f"extract_{name}_bioproj.npz").resolve()),
                    "sha256": projected_hashes[name],
                }
                for name in ("db", "query")
            },
            "training_provenance": {
                "protocol_identity_sha256": protocol_digest,
                **({
                    "protocol_mode": "author_fixed_final",
                    "protocol_stage": "author_fixed_single_stage",
                    "cache_decode_failure_audit": decode_failure_audit,
                    "execution_environment": execution_environment,
                    "execution_environment_sha256": (
                        execution_environment_sha256),
                } if author_fixed else {}),
                "checkpoint": {
                    "sha256": checkpoint_sha,
                    **({
                        "checkpoint_protocol_mode": "author_fixed_final",
                        "checkpoint_protocol_stage": (
                            "author_fixed_single_stage"),
                        "protocol_identity_sha256": protocol_digest,
                        "execution_environment": execution_environment,
                        "execution_environment_sha256": (
                            execution_environment_sha256),
                    } if author_fixed else {}),
                },
            },
        }],
    }
    bio_path = cell_dir / "bio_projection.json"
    bio_sha = _write_json(bio_path, bio)
    sidecar = cell_dir / "bio_projection.json.sha256"
    sidecar_sha = _write_artifact(
        sidecar, f"{bio_sha}  bio_projection.json\n".encode("utf-8"))

    manifest = {
        "variant": variant,
        "method": variant,
        "dataset": dataset,
        "bit_length": bit,
        "base_length": base_length,
        "seed": seed,
        "val_seed": None if author_fixed else 42,
        "information_tier": "U0",
        "semantic_information_condition": semantic,
        "selection_metric": (
            None if author_fixed
            else "raw_18base_base_hamming_mAP_at_R"),
        "test_used_for_selection": False,
        "final_checkpoint_count": 1,
        "nominal_schedule_horizon": 60,
        "run_manifest_phase": "bio_projection_completed",
        "protocol_deviations": [],
        "main_eligibility_blockers": (
            [] if author_fixed else ["cache_test_provenance_missing"]),
        "main_protocol_eligible": author_fixed,
        "protocol_identity": identity,
        "protocol_digest_sha256": protocol_digest,
        "final_checkpoint": str(checkpoint.resolve()),
        "final_checkpoint_sha256": checkpoint_sha,
        "extraction_dir": str(cell_dir.resolve()),
        "extraction_artifact_sha256": extraction_hashes,
        "protocol_manifest": str(protocol_manifest.resolve()),
        "protocol_manifest_sha256": protocol_manifest_sha,
        "selection_artifact": (
            None if author_fixed else str(selection.resolve())),
        "selection_artifact_sha256": selection_sha,
        "bio_projection_manifest": str(bio_path.resolve()),
        "bio_projection_manifest_sha256": bio_sha,
        "bio_projection_sidecar": str(sidecar.resolve()),
        "bio_projection_sidecar_sha256": sidecar_sha,
        "bio_projection_status": (
            "paper_result_eligible" if author_fixed
            else "completed_not_paper_eligible"),
        "bio_projected_artifacts": {
            name: {
                "path": str(
                    (cell_dir / f"extract_{name}_bioproj.npz").resolve()),
                "sha256": projected_hashes[name],
            }
            for name in ("db", "query")
        },
    }
    if author_fixed:
        manifest.update({
            "protocol_mode": "author_fixed_final",
            "checkpoint_policy": "author_horizon_last",
            "author_horizon": 60,
            "training_epochs": 60,
            "final_epoch_zero_based": 59,
            "validation_selection": False,
            "refit_performed": False,
            "training_stage": "author_fixed_single_stage",
            "designated_train_scope": "full_designated_train",
            "checkpoint_protocol": {
                "protocol_mode": "author_fixed_final",
                "protocol_stage": "author_fixed_single_stage",
                "protocol_identity_sha256": protocol_digest,
            },
            "cache_decode_failure_audit": decode_failure_audit,
            "execution_environment": execution_environment,
            "execution_environment_sha256": execution_environment_sha256,
        })
    else:
        manifest.update({
            "best_epoch_zero_based": 0,
            "refit_epochs": 1,
        })
    pre_manifest = {
        key: value for key, value in manifest.items()
        if key not in {
            "protocol_manifest", "protocol_manifest_sha256",
            "bio_projection_manifest", "bio_projection_manifest_sha256",
            "bio_projection_sidecar", "bio_projection_sidecar_sha256",
            "bio_projected_artifacts",
        }
    }
    pre_manifest["run_manifest_phase"] = "pre_bio_projection"
    pre_manifest["bio_projection_status"] = "pending"
    protocol_manifest_sha = _write_json(protocol_manifest, pre_manifest)
    manifest["protocol_manifest_sha256"] = protocol_manifest_sha

    manifest_path = cell_dir / "p0_run_manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path, manifest


class BaselineMatrixImplementationAuditTest(unittest.TestCase):
    def test_d6_author_fixed_manifest_is_accepted_without_selection_or_refit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, manifest = _minimal_resume_fixture(
                root, author_fixed=True)
            key = Key("u0", "cibhash", "CIFAR10", 30, 42)
            record = _validate_manifest(
                manifest_path, key, verify_hashes=True,
                expected_protocol_mode="author_fixed_final")
            self.assertEqual(record["status"], "complete_main_eligible")
            self.assertEqual(record["validation_errors"], [])
            self.assertEqual(record["author_horizon"], 60)
            self.assertEqual(record["final_epoch_zero_based"], 59)
            self.assertIsNone(manifest["selection_artifact"])
            self.assertNotIn("best_epoch_zero_based", manifest)
            self.assertNotIn("refit_epochs", manifest)

    def test_d6_rejects_consumed_cifar_label_source_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, manifest = _minimal_resume_fixture(
                root, author_fixed=True)
            source_root = Path(manifest["protocol_identity"]["dataset_root"])
            label_source = (
                source_root / "CIFAR10" / "cifar-10-batches-py"
                / "data_batch_1")
            label_source.write_bytes(b"label-only-tamper")
            record = _validate_manifest(
                manifest_path, Key("u0", "cibhash", "CIFAR10", 30, 42),
                verify_hashes=False,
                expected_protocol_mode="author_fixed_final")
            self.assertEqual(record["status"], "invalid")
            self.assertTrue(any(
                "dataset source binding" in error
                and "split/source changed" in error
                for error in record["validation_errors"]),
                record["validation_errors"])

    def test_d6_rejects_omitted_null_or_present_legacy_fields(self) -> None:
        mutations = (
            ("missing-selection-key", lambda payload: payload.pop(
                "selection_artifact")),
            ("present-best-epoch", lambda payload: payload.__setitem__(
                "best_epoch_zero_based", None)),
            ("present-refit", lambda payload: payload.__setitem__(
                "refit_epochs", 60)),
        )
        for label, mutate in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest_path, manifest = _minimal_resume_fixture(
                    root, author_fixed=True)
                mutate(manifest)
                _write_json(manifest_path, manifest)
                record = _validate_manifest(
                    manifest_path,
                    Key("u0", "cibhash", "CIFAR10", 30, 42),
                    verify_hashes=False,
                    expected_protocol_mode="author_fixed_final")
                self.assertEqual(record["status"], "invalid")
                self.assertTrue(any(
                    "author_fixed_final" in error
                    for error in record["validation_errors"]),
                    record["validation_errors"])

    def test_d6_rejects_checkpoint_mode_stage_digest_forgery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, manifest = _minimal_resume_fixture(
                root, author_fixed=True)
            manifest["checkpoint_protocol"]["protocol_stage"] = (
                "P0_stage2_refit_test")
            _write_json(manifest_path, manifest)
            record = _validate_manifest(
                manifest_path, Key("u0", "cibhash", "CIFAR10", 30, 42),
                verify_hashes=False,
                expected_protocol_mode="author_fixed_final")
            self.assertEqual(record["status"], "invalid")
            self.assertTrue(any(
                "checkpoint_protocol" in error
                for error in record["validation_errors"]),
                record["validation_errors"])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, manifest = _minimal_resume_fixture(
                root, author_fixed=True)
            checkpoint = Path(manifest["final_checkpoint"])
            payload = torch.load(
                checkpoint, map_location="cpu", weights_only=True)
            payload["config"]["protocol_mode"] = "validation_sensitivity"
            torch.save(payload, checkpoint)
            record = _validate_manifest(
                manifest_path, Key("u0", "cibhash", "CIFAR10", 30, 42),
                verify_hashes=False,
                expected_protocol_mode="author_fixed_final")
            self.assertEqual(record["status"], "invalid")
            self.assertTrue(any(
                "final checkpoint protocol contract" in error
                for error in record["validation_errors"]),
                record["validation_errors"])

    def test_d6_rejects_pre_bio_or_bio_protocol_stage_forgery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, manifest = _minimal_resume_fixture(
                root, author_fixed=True)
            pre_path = Path(manifest["protocol_manifest"])
            pre = json.loads(pre_path.read_text(encoding="utf-8"))
            pre["checkpoint_protocol"]["protocol_stage"] = (
                "P0_stage2_refit_test")
            manifest["protocol_manifest_sha256"] = _write_json(pre_path, pre)
            _write_json(manifest_path, manifest)
            record = _validate_manifest(
                manifest_path, Key("u0", "cibhash", "CIFAR10", 30, 42),
                verify_hashes=True,
                expected_protocol_mode="author_fixed_final")
            self.assertEqual(record["status"], "invalid")
            self.assertTrue(any(
                "pre-bio/run checkpoint_protocol" in error
                for error in record["validation_errors"]),
                record["validation_errors"])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, manifest = _minimal_resume_fixture(
                root, author_fixed=True)
            bio_path = Path(manifest["bio_projection_manifest"])
            bio = json.loads(bio_path.read_text(encoding="utf-8"))
            bio["cells"][0]["training_provenance"]["protocol_stage"] = (
                "P0_stage2_refit_test")
            bio_sha = _write_json(bio_path, bio)
            manifest["bio_projection_manifest_sha256"] = bio_sha
            sidecar = Path(manifest["bio_projection_sidecar"])
            manifest["bio_projection_sidecar_sha256"] = _write_artifact(
                sidecar, f"{bio_sha}  bio_projection.json\n".encode("utf-8"))
            _write_json(manifest_path, manifest)
            record = _validate_manifest(
                manifest_path, Key("u0", "cibhash", "CIFAR10", 30, 42),
                verify_hashes=True,
                expected_protocol_mode="author_fixed_final")
            self.assertEqual(record["status"], "invalid")
            self.assertTrue(any(
                "bio training protocol stage" in error
                for error in record["validation_errors"]),
                record["validation_errors"])

    def test_d6_rejects_decode_audit_manifest_cache_or_bio_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, manifest = _minimal_resume_fixture(
                root, author_fixed=True)
            manifest["cache_decode_failure_audit"]["policy"] = "forged"
            _write_json(manifest_path, manifest)
            record = _validate_manifest(
                manifest_path, Key("u0", "cibhash", "CIFAR10", 30, 42),
                verify_hashes=False,
                expected_protocol_mode="author_fixed_final")
            self.assertEqual(record["status"], "invalid")
            self.assertTrue(any(
                "cache_decode_failure_audit" in error
                for error in record["validation_errors"]),
                record["validation_errors"])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, manifest = _minimal_resume_fixture(
                root, author_fixed=True)
            cache = Path(manifest["protocol_identity"]["cache_dir"])
            _write_json(cache / "image_ids.json", ["cifar-row-tampered"])
            record = _validate_manifest(
                manifest_path, Key("u0", "cibhash", "CIFAR10", 30, 42),
                verify_hashes=False,
                expected_protocol_mode="author_fixed_final")
            self.assertEqual(record["status"], "invalid")
            self.assertTrue(any(
                "cache decode-failure binding" in error
                for error in record["validation_errors"]),
                record["validation_errors"])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, manifest = _minimal_resume_fixture(
                root, author_fixed=True)
            bio_path = Path(manifest["bio_projection_manifest"])
            bio = json.loads(bio_path.read_text(encoding="utf-8"))
            bio["cells"][0]["training_provenance"].pop(
                "cache_decode_failure_audit")
            bio_sha = _write_json(bio_path, bio)
            manifest["bio_projection_manifest_sha256"] = bio_sha
            sidecar = Path(manifest["bio_projection_sidecar"])
            manifest["bio_projection_sidecar_sha256"] = _write_artifact(
                sidecar, f"{bio_sha}  bio_projection.json\n".encode("utf-8"))
            _write_json(manifest_path, manifest)
            record = _validate_manifest(
                manifest_path, Key("u0", "cibhash", "CIFAR10", 30, 42),
                verify_hashes=True,
                expected_protocol_mode="author_fixed_final")
            self.assertEqual(record["status"], "invalid")
            self.assertTrue(any(
                "bio training cache decode-failure audit" in error
                for error in record["validation_errors"]),
                record["validation_errors"])

    def test_d6_validator_rejects_non_30_bit_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, _ = _minimal_resume_fixture(
                root, author_fixed=True)
            record = _validate_manifest(
                manifest_path, Key("u0", "cibhash", "CIFAR10", 36, 42),
                verify_hashes=False,
                expected_protocol_mode="author_fixed_final")
            self.assertEqual(record["status"], "invalid")
            self.assertTrue(any(
                "matched 30-bit" in error
                for error in record["validation_errors"]))

    def test_d6_resume_binds_mode_and_every_cache_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, manifest = _minimal_resume_fixture(root, author_fixed=True)
            identity = manifest["protocol_identity"]
            self.assertIsInstance(identity, dict)
            binding = {
                field: identity[field]
                for field in (
                    "dataset_root", "split_sha256",
                    "cache_dir", "cache_meta_sha256",
                    "cache_image_ids_sha256", "cache_artifact_sha256",
                    "cache_decode_failure_audit")
            }
            key = ("cibhash", "CIFAR10", 30, 42)
            expected = {("cibhash", "CIFAR10"): binding}
            self.assertIn(key, _completed_keys(
                root, expected_protocol_mode="author_fixed_final",
                cache_bindings=expected))
            self.assertNotIn(key, _completed_keys(
                root, expected_protocol_mode="validation_sensitivity",
                cache_bindings=expected))

            for field in (
                    "split_sha256", "cache_meta_sha256", "cache_image_ids_sha256",
                    "cache_artifact_sha256", "cache_decode_failure_audit"):
                changed = dict(binding)
                changed[field] = (
                    {"visual_global.f16.npy": "9" * 64}
                    if field == "cache_artifact_sha256" else
                    {**binding["cache_decode_failure_audit"],
                     "policy": "forged"}
                    if field == "cache_decode_failure_audit" else
                    {**binding["split_sha256"],
                     "CIFAR10/cifar-10-batches-py/data_batch_1": "9" * 64}
                    if field == "split_sha256" else "9" * 64)
                self.assertNotIn(key, _completed_keys(
                    root, expected_protocol_mode="author_fixed_final",
                    cache_bindings={("cibhash", "CIFAR10"): changed}), field)

    def test_d6_resume_rejects_missing_or_drifted_execution_environment(self) -> None:
        mutations = (
            lambda manifest: manifest.pop("execution_environment"),
            lambda manifest: manifest["execution_environment"]["packages"]
            ["numpy"].__setitem__("version", "forged-package-drift"),
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest_path, manifest = _minimal_resume_fixture(
                    root, author_fixed=True)
                mutate(manifest)
                _write_json(manifest_path, manifest)
                self.assertNotIn(
                    ("cibhash", "CIFAR10", 30, 42),
                    _completed_keys(
                        root, expected_protocol_mode="author_fixed_final"))

    def test_legacy_two_stage_manifest_cannot_satisfy_d6_validator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, _ = _minimal_resume_fixture(root)
            record = _validate_manifest(
                manifest_path, Key("u0", "cibhash", "CIFAR10", 36, 42),
                verify_hashes=False,
                expected_protocol_mode="author_fixed_final")
            self.assertEqual(record["status"], "invalid")
            self.assertTrue(any(
                "protocol_mode" in error
                for error in record["validation_errors"]))

    def test_source_profile_filter_is_exact_and_fail_closed(self) -> None:
        key = Key("u0", "mls3rduh", "CIFAR10", 36, 42)
        profile = CANONICAL_VARIANT_SOURCE_PROFILES["mls3rduh"]
        source_path = str(profile["path"])
        canonical = {
            "protocol_identity": {
                "implementation_sha256": {
                    source_path: str(profile["sha256"]),
                },
            },
        }
        self.assertIsNone(_source_profile_exclusion(canonical, key))

        old_hybrid = {
            "protocol_identity": {
                "implementation_sha256": {source_path: "e" * 64},
            },
        }
        exclusion = _source_profile_exclusion(old_hybrid, key)
        self.assertIsNotNone(exclusion)
        self.assertEqual(exclusion["required_profile"], profile["profile"])
        self.assertEqual(exclusion["actual_sha256"], "e" * 64)

        missing = _source_profile_exclusion({}, key)
        self.assertIsNotNone(missing)
        self.assertEqual(missing["actual_sha256"], "missing")

        unrelated = Key("u0", "not-pinned", "CIFAR10", 36, 42)
        self.assertIsNone(_source_profile_exclusion({}, unrelated))

    def test_all_source_profiles_match_current_files_and_runner(self) -> None:
        aggregate_profiles = CANONICAL_VARIANT_SOURCE_PROFILES
        self.assertEqual(
            set(aggregate_profiles),
            set(U0_VARIANTS) | {"umrch"} | set(SUPERVISED_VARIANTS),
        )
        self.assertEqual(set(aggregate_profiles), set(RUNNER_SOURCE_PROFILES))
        for variant, profile in aggregate_profiles.items():
            source_path = str(profile["path"])
            expected = str(profile["sha256"])
            self.assertEqual(
                sha256((REPO / source_path).read_bytes()).hexdigest(), expected,
                variant,
            )
            self.assertEqual(
                RUNNER_SOURCE_PROFILES[variant], (source_path, expected), variant)
            snapshot = {
                path: _current_source_sha256(path)
                for path in runner_required_implementation_paths(variant)
            }
            self.assertTrue(all(isinstance(value, str)
                                for value in snapshot.values()))
            payload = {
                "protocol_identity": {
                    "implementation_sha256": snapshot,
                },
            }
            self.assertTrue(_matches_canonical_source_profile(payload, variant))
            removed = next(path for path in snapshot if path != source_path)
            removed_digest = snapshot.pop(removed)
            self.assertFalse(_matches_canonical_source_profile(payload, variant))
            snapshot[removed] = removed_digest
            payload["protocol_identity"]["implementation_sha256"][source_path] = (
                "0" * 64)
            self.assertFalse(_matches_canonical_source_profile(payload, variant))

    def test_required_implementation_paths_match_runner_and_aggregator(self) -> None:
        self.assertEqual(
            AGGREGATE_COMMON_SOURCE_PATHS, RUNNER_COMMON_SOURCE_PATHS)
        self.assertEqual(len(AGGREGATE_COMMON_SOURCE_PATHS), 14)
        for variant in CANONICAL_VARIANT_SOURCE_PROFILES:
            self.assertEqual(
                aggregate_required_implementation_paths(variant),
                runner_required_implementation_paths(variant),
                variant,
            )
            self.assertEqual(
                len(aggregate_required_implementation_paths(variant)), 15,
                variant,
            )

    def test_aggregator_rejects_method_only_implementation_closure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, manifest = _minimal_resume_fixture(root)
            key = Key("u0", "cibhash", "CIFAR10", 36, 42)
            baseline = _validate_manifest(
                manifest_path, key, verify_hashes=False)
            self.assertEqual(baseline["status"], "complete_diagnostic_only")
            self.assertEqual(baseline["validation_errors"], [])

            identity = manifest["protocol_identity"]
            self.assertIsInstance(identity, dict)
            snapshot = identity["implementation_sha256"]
            self.assertIsInstance(snapshot, dict)
            method_path = "baseline/CIBHash.py"
            identity["implementation_sha256"] = {
                method_path: snapshot[method_path],
            }
            manifest["protocol_digest_sha256"] = (
                aggregate_protocol_identity_digest(identity))
            _write_json(manifest_path, manifest)

            rejected = _validate_manifest(
                manifest_path, key, verify_hashes=False)
            self.assertEqual(rejected["status"], "invalid")
            self.assertTrue(any(
                "implementation_sha256 path-set mismatch" in error
                for error in rejected["validation_errors"]
            ))

    def test_resume_rejects_selection_metric_mutation_without_mock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, manifest = _minimal_resume_fixture(root)
            key = ("cibhash", "CIFAR10", 36, 42)
            self.assertIn(key, _completed_keys(root))

            manifest["selection_metric"] = "tampered_metric"
            _write_json(manifest_path, manifest)
            self.assertNotIn(key, _completed_keys(root))

    def test_resume_completion_rejects_semantic_condition_mutation(self) -> None:
        variant = "cibhash"
        dataset = "CIFAR10"
        semantic = {
            "information_tier": "U0",
            "term_source_condition": "visual-only",
            "matched_official_taxonomy_dataset": "",
            "selection_verified": None,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bio_path = root / "bio_projection.json"
            bio_path.write_text("{}\n", encoding="utf-8")
            manifest_path = root / "cell" / "p0_run_manifest.json"
            manifest_path.parent.mkdir()
            manifest = {
                "variant": variant,
                "dataset": dataset,
                "seed": 42,
                "bit_length": 36,
                "information_tier": "U0",
                "semantic_information_condition": semantic,
                "protocol_identity": {
                    "variant": variant,
                    "dataset": dataset,
                    "bit": 36,
                    "seed": 42,
                    "val_seed": 42,
                    "val_ratio": 0.1,
                    "eval_period": 5,
                    "horizon": 60,
                    "batch_size_override": None,
                    "implementation_sha256": {
                        path: _current_source_sha256(path)
                        for path in runner_required_implementation_paths(variant)
                    },
                    "semantic_information_condition": semantic,
                },
                "protocol_deviations": [],
                "test_used_for_selection": False,
                "final_checkpoint_count": 1,
                "run_manifest_phase": "bio_projection_completed",
                "bio_projection_status": "completed_not_paper_eligible",
                "bio_projection_manifest": str(bio_path),
                "bio_projection_manifest_sha256": sha256(
                    bio_path.read_bytes()).hexdigest(),
            }
            manifest["protocol_digest_sha256"] = runner_protocol_identity_digest(
                manifest["protocol_identity"])
            self.assertEqual(
                manifest["protocol_digest_sha256"],
                aggregate_protocol_identity_digest(manifest["protocol_identity"]),
            )
            manifest_path.write_text(
                json.dumps(manifest), encoding="utf-8")
            key = (variant, dataset, 36, 42)
            with patch(
                    "scripts.run_baseline_p0_matrix."
                    "_full_manifest_validation_passes", return_value=True):
                self.assertIn(key, _completed_keys(root))

            manifest["protocol_identity"]["device_request"] = "tampered"
            manifest_path.write_text(
                json.dumps(manifest), encoding="utf-8")
            with patch(
                    "scripts.run_baseline_p0_matrix."
                    "_full_manifest_validation_passes", return_value=True):
                self.assertNotIn(key, _completed_keys(root))

            manifest["protocol_identity"].pop("device_request")
            manifest["protocol_identity"]["semantic_information_condition"] = {
                **semantic,
                "information_tier": "U2",
            }
            manifest["protocol_digest_sha256"] = runner_protocol_identity_digest(
                manifest["protocol_identity"])
            manifest_path.write_text(
                json.dumps(manifest), encoding="utf-8")
            with patch(
                    "scripts.run_baseline_p0_matrix."
                    "_full_manifest_validation_passes", return_value=True):
                self.assertNotIn(key, _completed_keys(root))

    def test_diagnostic_aggregate_cannot_populate_unqualified_paper_mean(self) -> None:
        key = Key("u0", "cibhash", "CIFAR10", 36, 42)
        record = _record(key, {"baseline/CIBHash.py": "a" * 64})
        record["implementation_comparison_eligible"] = True
        record["paper_table_eligible"] = False
        aggregate = _aggregate_cell(
            {key: record}, panel="u0", variant="cibhash",
            dataset="CIFAR10", bit=36, seeds=(42,))
        self.assertEqual(aggregate["aggregate_status"], "complete_diagnostic_only")
        self.assertTrue(aggregate["diagnostic_only"])
        self.assertFalse(aggregate["paper_table_eligible"])
        self.assertIsNone(aggregate["mean_map_at_R_post"])
        self.assertEqual(aggregate["diagnostic_mean_map_at_R_post"], 0.5)
        self.assertEqual(aggregate["admitted_comparison_mean_map_at_R_post"], 0.5)

    def test_snapshot_requires_repo_relative_lowercase_sha256(self) -> None:
        errors: list[str] = []
        snapshot = _implementation_snapshot(
            errors, {"baseline/model.py": "a" * 64})
        self.assertEqual(snapshot, {"baseline/model.py": "a" * 64})
        self.assertEqual(errors, [])

        errors = []
        self.assertIsNone(_implementation_snapshot(
            errors, {"../outside.py": "A" * 64}))
        self.assertTrue(any("repository-relative" in error for error in errors))

    def test_reviewed_cache_memo_transition_is_warning_not_blocker(self) -> None:
        transition = KNOWN_NON_SCIENTIFIC_IMPLEMENTATION_TRANSITIONS[
            "baseline/cache_provenance.py"]
        old = str(transition["before_sha256"])
        new = str(transition["after_sha256"])
        first = Key("u0", "crovca", "Flickr25k", 36, 42)
        second = Key("u0", "crovca", "MSCOCO", 36, 42)
        records = {
            first: _record(first, {"baseline/cache_provenance.py": old}),
            second: _record(second, {"baseline/cache_provenance.py": new}),
        }

        audit = _audit_implementation_fingerprints(
            records,
            current_source_sha256={"baseline/cache_provenance.py": new},
        )
        self.assertTrue(audit["comparison_safe"])
        self.assertEqual(audit["blocked_cells"], {})
        self.assertEqual(
            audit["status"],
            "warning_known_non_scientific_performance_memo_transition",
        )
        self.assertEqual(
            audit["known_non_scientific_warnings"][0]["classification"],
            "non_scientific_performance_memo_only",
        )
        variant = audit["variant_fingerprints"][0]
        self.assertTrue(variant["heterogeneous"])
        self.assertTrue(variant["comparison_safe"])

    def test_unreviewed_cross_cell_hash_drift_blocks_both_cells(self) -> None:
        first = Key("u0", "crovca", "Flickr25k", 36, 42)
        second = Key("u0", "crovca", "MSCOCO", 36, 42)
        path = "scripts/run_modern_baseline_p0.py"
        records = {
            first: _record(first, {path: "a" * 64}),
            second: _record(second, {path: "b" * 64}),
        }
        audit = _audit_implementation_fingerprints(
            records, current_source_sha256={path: "b" * 64})
        self.assertFalse(audit["comparison_safe"])
        self.assertEqual(
            audit["status"],
            "blocked_unverified_implementation_source",
        )
        self.assertEqual(audit["unknown_cross_cell_drift_paths"], [path])
        self.assertEqual(set(audit["blocked_cells"]), {first.text, second.text})

        _apply_implementation_admission(records, audit)
        self.assertFalse(records[first]["implementation_comparison_eligible"])
        self.assertFalse(records[second]["implementation_comparison_eligible"])

    def test_cache_filename_does_not_exempt_an_unreviewed_third_digest(self) -> None:
        transition = KNOWN_NON_SCIENTIFIC_IMPLEMENTATION_TRANSITIONS[
            "baseline/cache_provenance.py"]
        old = str(transition["before_sha256"])
        new = str(transition["after_sha256"])
        first = Key("u0", "crovca", "Flickr25k", 36, 42)
        second = Key("u0", "crovca", "MSCOCO", 36, 42)
        path = "baseline/cache_provenance.py"
        records = {
            first: _record(first, {path: old}),
            second: _record(second, {path: "c" * 64}),
        }
        audit = _audit_implementation_fingerprints(
            records, current_source_sha256={path: new})
        self.assertFalse(audit["comparison_safe"])
        self.assertEqual(audit["known_non_scientific_warnings"], [])
        self.assertEqual(audit["unknown_cross_cell_drift_paths"], [path])

    def test_homogeneous_historical_snapshot_drift_is_blocked(self) -> None:
        key = Key("u0", "greedyhash", "Flickr25k", 36, 42)
        path = "baseline/GreedyHash.py"
        records = {key: _record(key, {path: "a" * 64})}
        audit = _audit_implementation_fingerprints(
            records, current_source_sha256={path: "b" * 64})
        self.assertFalse(audit["comparison_safe"])
        self.assertEqual(
            audit["historical_snapshot_current_drift_paths"], [path])
        self.assertEqual(
            audit["status"],
            "blocked_unverified_implementation_source",
        )
        self.assertIn(key.text, audit["blocked_cells"])

    def test_legacy_cache_record_can_never_be_paper_table_eligible(self) -> None:
        key = Key("u0", "crovca", "Flickr25k", 36, 42)
        path = "baseline/CroVCA.py"
        record = _record(
            key, {path: "a" * 64}, main_eligible=True, legacy_cache=True)
        records = {key: record}
        audit = _audit_implementation_fingerprints(
            records, current_source_sha256={path: "a" * 64})
        _apply_implementation_admission(records, audit)
        self.assertTrue(record["implementation_comparison_eligible"])
        self.assertFalse(record["paper_table_eligible"])


if __name__ == "__main__":
    unittest.main()
