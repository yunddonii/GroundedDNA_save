import json
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from dna_utils.bio_constraints import (
    PROJECTION_TIE_POLICY,
    batch_project_to_valid,
    is_valid,
    is_valid_batch,
    project_to_valid,
)
import scripts.apply_bio_projection as bio_script
from baseline.cache_provenance import audit_cache_decode_failures
from baseline.execution_environment import canonical_digest
from scripts.run_modern_baseline_p0 import _dataset_source_binding


def _pickle_load_bomb():
    raise AssertionError("object pickle must never be deserialized")


class _PickleBomb:
    def __reduce__(self):
        return (_pickle_load_bomb, ())


def _bits_from_bases(bases: np.ndarray) -> np.ndarray:
    bases = np.asarray(bases, dtype=np.int64)
    return np.stack((bases // 2, bases % 2), axis=-1).reshape(len(bases), -1).astype(np.uint8)


def _write_extraction(path: Path, bases: np.ndarray, labels: np.ndarray) -> None:
    np.savez(
        path,
        base_indices=np.asarray(bases, dtype=np.int64),
        hash_2bit=_bits_from_bases(bases),
        multi_hot_labels=np.asarray(labels, dtype=np.int64),
        image_paths=np.asarray([f"image-{index}" for index in range(len(bases))]),
    )


def _execution_environment() -> tuple[dict[str, object], str]:
    packages = {
        logical: {
            "distribution": distribution, "installed": True,
            "version": "test-version",
        }
        for logical, distribution in {
            "torch": "torch", "transformers": "transformers",
            "numpy": "numpy", "scipy": "scipy",
            "sklearn": "scikit-learn", "pandas": "pandas",
            "timm": "timm", "tensorboard": "tensorboard",
            "tqdm": "tqdm", "Pillow": "Pillow",
        }.items()
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
        "packages": packages,
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


class BioConstraintValidationTest(unittest.TestCase):
    def test_rejects_lossy_or_out_of_alphabet_base_arrays(self) -> None:
        invalid = (
            np.asarray([-1, 1] * 9, dtype=np.int64),
            np.asarray([256, 1] * 9, dtype=np.int64),
            np.asarray([0.9, 1.9] * 9, dtype=np.float64),
        )
        for code in invalid:
            with self.subTest(dtype=code.dtype, head=code[:2].tolist()):
                with self.assertRaises((TypeError, ValueError)):
                    is_valid(code)
                with self.assertRaises((TypeError, ValueError)):
                    project_to_valid(code)

    def test_infeasible_integer_gc_range_is_fail_closed_by_default(self) -> None:
        one_base = np.asarray([[0]], dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "no integer GC count"):
            bio_script.project_batch_memoized(one_base, 0.4, 0.6, 3)
        diagnostic = bio_script.project_batch_memoized(
            one_base, 0.4, 0.6, 3, allow_projection_failures=True)
        self.assertEqual(diagnostic["n_failed"], 1)
        self.assertEqual(diagnostic["edit_distances"].tolist(), [-1])
        self.assertEqual(diagnostic["compliance_rate"], 0.0)
        with self.assertRaisesRegex(RuntimeError, "refusing to return invalid DNA"):
            batch_project_to_valid(one_base, 0.4, 0.6, 3, progress=False)
        core_diagnostic = batch_project_to_valid(
            one_base, 0.4, 0.6, 3, progress=False,
            allow_projection_failures=True)
        self.assertEqual(core_diagnostic["n_failed"], 1)


class BioProjectionArtifactTest(unittest.TestCase):
    def _make_cell(
            self, root: Path, *, variant: str = "toy", bit: int = 30,
            author_fixed: bool = True) -> Path:
        base_length = bit // 2
        selection_metric = (
            f"raw_{base_length}base_base_hamming_mAP_at_R")
        cell = root / "toy_refit_dnaeval"
        cell.mkdir()
        protocol_payload = {
            "variant": variant, "dataset": "Flickr25k",
            "setting": "setting1", "bit": bit,
        }
        execution_environment, execution_environment_sha256 = (
            _execution_environment())
        if author_fixed:
            source_root = root / "dataset"
            split_root = source_root / "Flickr25k" / "setting1"
            split_root.mkdir(parents=True)
            image_ids = []
            for name in ("train.txt", "test.txt", "database.txt"):
                image_id = f"images/{name.removesuffix('.txt')}.jpg"
                image_ids.append(image_id)
                (split_root / name).write_text(
                    f"{image_id} 1 0\n", encoding="utf-8")
            cache = root / "cache"
            cache.mkdir()
            (cache / "meta.json").write_text(json.dumps({
                "n_failed_image_decode": 0,
                "failed_image_indices": [],
            }), encoding="utf-8")
            (cache / "image_ids.json").write_text(
                json.dumps(image_ids), encoding="utf-8")
            dataset_root, split_sha256 = _dataset_source_binding(
                "Flickr25k", "setting1", source_root)
            decode_audit = audit_cache_decode_failures(
                cache, dataset="Flickr25k", dataset_root=source_root)
            protocol_payload.update({
                "protocol_mode": "author_fixed_final",
                "trainer_protocol_stages": ["author_fixed_single_stage"],
                "final_checkpoint_protocol_stage": (
                    "author_fixed_single_stage"),
                "checkpoint_policy": "author_horizon_last",
                "author_horizon": 5,
                "training_epochs": 5,
                "final_epoch_zero_based": 4,
                "validation_selection": False,
                "designated_train_scope": "full_designated_train",
                "val_seed": None,
                "val_ratio": 0.0,
                "horizon": 5,
                "eval_period": 5,
                "dataset_root": dataset_root,
                "split_sha256": split_sha256,
                "cache_dir": str(cache.resolve()),
                "cache_meta_sha256": decode_audit["cache_meta_sha256"],
                "cache_image_ids_sha256": (
                    decode_audit["cache_image_ids_sha256"]),
                "cache_decode_failure_audit": decode_audit,
                "execution_environment": execution_environment,
                "execution_environment_sha256": (
                    execution_environment_sha256),
            })
        else:
            protocol_payload["selection"] = "heldout-validation-only"
        protocol = hashlib.sha256(json.dumps(
            protocol_payload, sort_keys=True,
            separators=(",", ":")).encode("utf-8")).hexdigest()
        db = np.asarray([
            [0] * base_length,
            np.resize([1, 2], base_length).tolist(),
            np.resize([0, 1, 2, 3], base_length).tolist(),
        ], dtype=np.int64)
        query = np.asarray([
            [0] * base_length,
            np.resize([1, 2], base_length).tolist(),
        ], dtype=np.int64)
        db_labels = np.asarray([[1, 0], [0, 1], [1, 0]], dtype=np.int64)
        query_labels = np.asarray([[1, 0], [0, 1]], dtype=np.int64)
        _write_extraction(cell / "extract_db.npz", db, db_labels)
        _write_extraction(cell / "extract_query.npz", query, query_labels)

        checkpoint = root / "epoch_004.pth"
        checkpoint_config = {
                "method": "toy", "dataset": "Flickr25k", "setting": "setting1",
                "bit": bit, "protocol_identity_sha256": protocol,
        }
        if author_fixed:
            checkpoint_config.update({
                "protocol_mode": "author_fixed_final",
                "protocol_stage": "author_fixed_single_stage",
                "execution_environment": execution_environment,
                "execution_environment_sha256": (
                    execution_environment_sha256),
            })
        torch.save({
            "config": checkpoint_config,
            "model_state_dict": {},
        }, checkpoint)
        (cell / "args_extract.txt").write_text(
            "\n".join((
                f"weights: {checkpoint}",
                "dataset: Flickr25k",
                "setting: setting1",
                f"bit: {bit}",
            )) + "\n",
            encoding="utf-8",
        )
        selection = root / "p0_selection_only.json"
        if not author_fixed:
            selection.write_text(json.dumps({
                "method": "toy", "dataset": "Flickr25k",
                "bit_length": bit, "base_length": base_length,
                "protocol_identity_sha256": protocol,
                "selection_only": True,
                "test_touched_during_selection": False,
                "selection_metric": selection_metric,
                "best_epoch": 4,
            }), encoding="utf-8")
        protocol_manifest = {
            "variant": variant, "method": "toy", "dataset": "Flickr25k",
            "bit_length": bit, "base_length": base_length,
            "protocol_digest_sha256": protocol,
            "protocol_identity": protocol_payload,
            "selection_metric": None if author_fixed else selection_metric,
            "selection_artifact": None if author_fixed else str(selection),
            "selection_artifact_sha256": (
                None if author_fixed else bio_script.sha256_file(selection)),
            "main_protocol_eligible": True,
            "protocol_deviations": [],
            "main_eligibility_blockers": [],
            "test_used_for_selection": False,
            "final_checkpoint_count": 1,
            "final_checkpoint": str(checkpoint),
            "final_checkpoint_sha256": bio_script.sha256_file(checkpoint),
            "extraction_dir": str(cell),
            "extraction_artifact_sha256": {
                "extract_db.npz": bio_script.sha256_file(cell / "extract_db.npz"),
                "extract_query.npz": bio_script.sha256_file(cell / "extract_query.npz"),
                "args_extract.txt": bio_script.sha256_file(cell / "args_extract.txt"),
            },
            "run_manifest_phase": "pre_bio_projection",
        }
        if author_fixed:
            protocol_manifest.update({
                "protocol_mode": "author_fixed_final",
                "checkpoint_policy": "author_horizon_last",
                "author_horizon": 5,
                "training_epochs": 5,
                "final_epoch_zero_based": 4,
                "validation_selection": False,
                "refit_performed": False,
                "training_stage": "author_fixed_single_stage",
                "designated_train_scope": "full_designated_train",
                "val_seed": None,
                "training_model_dir": str(checkpoint.parent),
                "checkpoint_protocol": {
                    "protocol_mode": "author_fixed_final",
                    "protocol_stage": "author_fixed_single_stage",
                    "protocol_identity_sha256": protocol,
                },
                "cache_decode_failure_audit": (
                    protocol_payload["cache_decode_failure_audit"]),
                "execution_environment": execution_environment,
                "execution_environment_sha256": (
                    execution_environment_sha256),
            })
        else:
            protocol_manifest.update({
                "best_epoch_zero_based": 4,
                "refit_model_dir": str(checkpoint.parent),
            })
        (cell / "p0_protocol_manifest.json").write_text(
            json.dumps(protocol_manifest), encoding="utf-8")
        return cell

    def test_strict_cell_records_both_splits_and_content_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cell = self._make_cell(Path(temporary))
            with mock.patch.object(bio_script, "SAVE_PROJECTED", True):
                result = bio_script.run_cell(
                    "toy", str(cell), "Flickr25k", 0.4, 0.6, 3,
                    expected_length=15, require_provenance=True, device="cpu")

            self.assertTrue(result["projection_invariant_satisfied"])
            self.assertEqual(result["projection_failures_db"], 0)
            self.assertEqual(result["projection_failures_qy"], 0)
            self.assertEqual(result["post_compliance_db"], 1.0)
            self.assertEqual(result["post_compliance_qy"], 1.0)
            self.assertEqual(result["projection_tie_policy"], PROJECTION_TIE_POLICY)
            run_manifest = json.loads(
                (cell / "p0_protocol_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(
                result["training_provenance"]["protocol_identity_sha256"],
                run_manifest["protocol_digest_sha256"])
            self.assertEqual(
                result["training_provenance"]["status"], "paper_protocol_eligible")
            self.assertTrue(result["training_provenance"]["paper_protocol_eligible"])
            self.assertEqual(
                result["training_provenance"]["protocol_mode"],
                "author_fixed_final")
            self.assertEqual(
                result["training_provenance"]["protocol_stage"],
                "author_fixed_single_stage")
            self.assertEqual(
                result["training_provenance"]["cache_decode_failure_audit"],
                run_manifest["cache_decode_failure_audit"])
            self.assertTrue(result["paper_result_eligible"])
            self.assertEqual(result["retrieval_device"], "cpu")
            protocol_record = result["training_provenance"]["protocol_manifest"]
            self.assertEqual(
                protocol_record["sha256"],
                bio_script.sha256_file(protocol_record["path"]))

            for split in ("db", "query"):
                input_record = result["input_artifacts"][split]
                output_record = result["output_artifacts"][split]
                self.assertEqual(
                    input_record["sha256"], bio_script.sha256_file(input_record["path"]))
                self.assertEqual(
                    output_record["sha256"], bio_script.sha256_file(output_record["path"]))
                with np.load(output_record["path"], allow_pickle=False) as projected:
                    self.assertEqual(
                        projected["projection_tie_policy"].item(), PROJECTION_TIE_POLICY)
                    self.assertEqual(
                        projected["source_npz_sha256"].item(), input_record["sha256"])
                    self.assertEqual(
                        projected["protocol_manifest_sha256"].item(),
                        protocol_record["sha256"])
                    self.assertEqual(
                        projected["checkpoint_sha256"].item(),
                        result["training_provenance"]["checkpoint"]["sha256"])
                    np.testing.assert_array_equal(
                        bio_script.base_from_hash(projected["hash_2bit"]),
                        projected["base_indices"],
                    )
                    self.assertTrue(is_valid_batch(projected["base_indices"]).all())

            leftovers = list(cell.glob(".*.tmp")) + list(cell.glob(".*.npz.npz"))
            self.assertEqual(leftovers, [])

    def test_48_bit_cell_uses_24_base_metric_and_default_gc_fractions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cell = self._make_cell(
                Path(temporary), bit=48, author_fixed=False)
            with mock.patch.object(bio_script, "SAVE_PROJECTED", True):
                result = bio_script.run_cell(
                    "toy", str(cell), "Flickr25k", 0.4, 0.6, 3,
                    expected_length=24, require_provenance=True, device="cpu")
            self.assertEqual(result["L"], 24)
            self.assertEqual(result["gc_count_range"], [10, 14])
            self.assertFalse(result["paper_result_eligible"])
            manifest = json.loads(
                (cell / "p0_protocol_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["selection_metric"],
                "raw_24base_base_hamming_mAP_at_R")
            self.assertNotIn("protocol_mode", manifest)

    def test_author_fixed_selection_null_and_best_epoch_absence_are_enforced(self) -> None:
        mutations = (
            ("omitted-selection-artifact", lambda payload: payload.pop(
                "selection_artifact"), "requires explicit null selection_artifact"),
            ("present-null-best-epoch", lambda payload: payload.__setitem__(
                "best_epoch_zero_based", None), "forbidden legacy field best_epoch"),
        )
        for label, mutate, expected_reason in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                cell = self._make_cell(Path(temporary))
                manifest_path = cell / "p0_protocol_manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                mutate(manifest)
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                result = bio_script.run_cell(
                    "toy", str(cell), "Flickr25k", 0.4, 0.6, 3,
                    expected_length=15, require_provenance=True, device="cpu")
                self.assertFalse(result["paper_result_eligible"])
                self.assertTrue(any(
                    expected_reason in reason
                    for reason in result["training_provenance"]["eligibility_reasons"]),
                    result["training_provenance"]["eligibility_reasons"])

    def test_author_fixed_reopens_decode_failure_cache_and_split_evidence(self) -> None:
        mutations = (
            ("missing-manifest-audit", lambda cell, manifest: manifest.pop(
                "cache_decode_failure_audit"), "cache_decode_failure_audit"),
            ("changed-manifest-audit", lambda cell, manifest: manifest[
                "cache_decode_failure_audit"].__setitem__("policy", "forged"),
             "cache_decode_failure_audit"),
            ("cache-meta-tamper", lambda cell, manifest: (
                Path(manifest["protocol_identity"]["cache_dir"])
                .joinpath("meta.json").write_text(json.dumps({
                    "n_failed_image_decode": 1,
                    "failed_image_indices": [0],
                }), encoding="utf-8")), "decode"),
            ("wrong-split-membership", lambda cell, manifest: (
                Path(manifest["protocol_identity"]["dataset_root"])
                .joinpath("Flickr25k", "setting1", "train.txt")
                .write_text("images/database.jpg 1 0\n", encoding="utf-8")),
             "split"),
        )
        for label, mutate, expected in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                cell = self._make_cell(Path(temporary))
                manifest_path = cell / "p0_protocol_manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                mutate(cell, manifest)
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, expected):
                    bio_script.run_cell(
                        "toy", str(cell), "Flickr25k", 0.4, 0.6, 3,
                        expected_length=15, require_provenance=True,
                        device="cpu")

    def test_validation_selection_refit_remains_diagnostic_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cell = self._make_cell(Path(temporary), author_fixed=False)
            result = bio_script.run_cell(
                "toy", str(cell), "Flickr25k", 0.4, 0.6, 3,
                expected_length=15, require_provenance=True, device="cpu")
            self.assertFalse(result["paper_result_eligible"])
            provenance = result["training_provenance"]
            self.assertEqual(provenance["protocol_mode"], "validation_sensitivity")
            self.assertTrue(any(
                "diagnostic-only" in reason
                for reason in provenance["eligibility_reasons"]))

    def test_author_fixed_rejects_non_d6_bit_or_extra_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cell = self._make_cell(Path(temporary), bit=36)
            result = bio_script.run_cell(
                "toy", str(cell), "Flickr25k", 0.4, 0.6, 3,
                expected_length=18, require_provenance=True, device="cpu")
            self.assertFalse(result["paper_result_eligible"])
            self.assertTrue(any(
                "matched 30-bit" in reason
                for reason in result["training_provenance"]["eligibility_reasons"]))

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cell = self._make_cell(root)
            (root / "epoch_000.pth").write_bytes(b"unexpected candidate")
            result = bio_script.run_cell(
                "toy", str(cell), "Flickr25k", 0.4, 0.6, 3,
                expected_length=15, require_provenance=True, device="cpu")
            self.assertFalse(result["paper_result_eligible"])
            self.assertTrue(any(
                "checkpoint set is not exactly" in reason
                for reason in result["training_provenance"]["eligibility_reasons"]))

    def test_inconsistent_base_and_hash_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary) / "bad.npz"
            bases = np.zeros((2, 18), dtype=np.int64)
            bits = _bits_from_bases(bases)
            bits[0, 0] = 1
            np.savez(
                artifact, base_indices=bases, hash_2bit=bits,
                multi_hot_labels=np.asarray([[1], [1]], dtype=np.int64))
            with self.assertRaisesRegex(ValueError, "different DNA codes"):
                bio_script._load_extraction(artifact, "bad")

    def test_object_image_paths_are_never_unpickled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary) / "legacy-object-paths.npz"
            bases = np.asarray([[0, 1] * 9], dtype=np.int64)
            np.savez(
                artifact,
                base_indices=bases,
                hash_2bit=_bits_from_bases(bases),
                multi_hot_labels=np.asarray([[1]], dtype=np.int64),
                image_paths=np.asarray([_PickleBomb()], dtype=object),
            )
            diagnostic = bio_script._load_extraction(
                artifact, "legacy", strict_image_paths=False)
            self.assertIsNone(diagnostic["image_paths"])
            self.assertEqual(
                diagnostic["image_paths_status"],
                "legacy_object_omitted_without_unpickling")
            with self.assertRaisesRegex(ValueError, "forbidden in paper-grade mode"):
                bio_script._load_extraction(
                    artifact, "legacy", strict_image_paths=True)

    def test_explicit_cell_requires_checkpoint_protocol_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cell = Path(temporary)
            bases = np.asarray([[0, 1] * 9], dtype=np.int64)
            labels = np.asarray([[1]], dtype=np.int64)
            _write_extraction(cell / "extract_db.npz", bases, labels)
            _write_extraction(cell / "extract_query.npz", bases, labels)
            with self.assertRaisesRegex(ValueError, "incomplete paper-result provenance"):
                bio_script.run_cell(
                    "toy", str(cell), "Flickr25k", 0.4, 0.6, 3,
                    require_provenance=True)

    def test_protocol_payload_hash_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cell = self._make_cell(Path(temporary))
            manifest_path = cell / "p0_protocol_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["protocol_identity"]["bit"] = 48
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "bit budget"):
                bio_script.run_cell(
                    "toy", str(cell), "Flickr25k", 0.4, 0.6, 3,
                    expected_length=15, require_provenance=True)

    def test_author_fixed_dataset_source_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cell = self._make_cell(Path(temporary))
            manifest = json.loads(
                (cell / "p0_protocol_manifest.json").read_text(
                    encoding="utf-8"))
            identity = manifest["protocol_identity"]
            source = (
                Path(identity["dataset_root"])
                / "Flickr25k" / "setting1" / "train.txt")
            source.write_text("/same-image 0 1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "split/source changed"):
                bio_script.run_cell(
                    "toy", str(cell), "Flickr25k", 0.4, 0.6, 3,
                    expected_length=15, require_provenance=True, device="cpu")

    def test_checkpoint_mode_stage_contract_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cell = self._make_cell(root)
            manifest_path = cell / "p0_protocol_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["checkpoint_protocol"]["protocol_stage"] = (
                "P0_stage2_refit_test")
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(
                    ValueError, "checkpoint_protocol contract mismatch"):
                bio_script.run_cell(
                    "toy", str(cell), "Flickr25k", 0.4, 0.6, 3,
                    expected_length=15, require_provenance=True, device="cpu")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cell = self._make_cell(root)
            checkpoint = root / "epoch_004.pth"
            payload = torch.load(
                checkpoint, map_location="cpu", weights_only=True)
            payload["config"]["protocol_stage"] = "P0_stage2_refit_test"
            torch.save(payload, checkpoint)
            with self.assertRaisesRegex(
                    ValueError, "mode/stage/digest disagrees"):
                bio_script.run_cell(
                    "toy", str(cell), "Flickr25k", 0.4, 0.6, 3,
                    expected_length=15, require_provenance=True, device="cpu")

    def test_unreadable_checkpoint_tamper_fails_strict_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cell = self._make_cell(root)
            (root / "epoch_004.pth").write_bytes(b"tampered")
            with self.assertRaisesRegex(
                    ValueError,
                    "incomplete paper-result provenance|final checkpoint SHA-256 mismatch"):
                bio_script.run_cell(
                    "toy", str(cell), "Flickr25k", 0.4, 0.6, 3,
                    expected_length=15, require_provenance=True, device="cpu")

    def test_post_manifest_extraction_substitution_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cell = self._make_cell(Path(temporary))
            replacement = np.resize([0, 1], (1, 15)).astype(np.int64)
            _write_extraction(
                cell / "extract_db.npz", replacement,
                np.asarray([[1, 0]], dtype=np.int64))
            with self.assertRaisesRegex(ValueError, "artifact SHA-256 map mismatch"):
                bio_script.run_cell(
                    "toy", str(cell), "Flickr25k", 0.4, 0.6, 3,
                    expected_length=15, require_provenance=True)

    def test_cli_infeasible_projection_raises_before_writing_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cell = Path(temporary) / "cell"
            cell.mkdir()
            bases = np.asarray([[0]], dtype=np.int64)
            labels = np.asarray([[1]], dtype=np.int64)
            _write_extraction(cell / "extract_db.npz", bases, labels)
            _write_extraction(cell / "extract_query.npz", bases, labels)
            output = Path(temporary) / "bio.json"
            argv = [
                "apply_bio_projection.py",
                "--out", str(output),
                "--cell", f"toy,Flickr25k,{cell}",
                "--expected_length", "1",
                "--allow_missing_provenance",
            ]
            with mock.patch.object(sys, "argv", argv):
                with self.assertRaisesRegex(ValueError, "no feasible integer GC count"):
                    bio_script.main()
            self.assertFalse(output.exists())
            self.assertFalse(Path(f"{output}.sha256").exists())

    def test_multi_cell_failure_does_not_leave_partial_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_root = root / "first"
            second_root = root / "second"
            first_root.mkdir(); second_root.mkdir()
            first = self._make_cell(first_root, variant="first")
            second = self._make_cell(second_root, variant="second")
            bad_query = np.asarray([[0] * 17], dtype=np.int64)
            _write_extraction(
                second / "extract_query.npz", bad_query,
                np.asarray([[1, 0]], dtype=np.int64))
            output = root / "partial-must-not-exist.json"
            argv = [
                "apply_bio_projection.py", "--out", str(output),
                "--cell", f"first,Flickr25k,{first}",
                "--cell", f"second,Flickr25k,{second}",
                "--expected_length", "15",
            ]
            with mock.patch.object(sys, "argv", argv):
                with self.assertRaisesRegex(ValueError, "lengths differ"):
                    bio_script.main()
            self.assertFalse(output.exists())
            self.assertFalse(Path(f"{output}.sha256").exists())

    def test_atomic_json_has_verifiable_sidecar_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "manifest.json"
            digest = bio_script._atomic_write_json(output, {"ok": True})
            sidecar = Path(f"{output}.sha256")
            bio_script._atomic_write_text(
                sidecar, f"{digest}  {output.name}\n")
            self.assertEqual(digest, bio_script.sha256_file(output))
            self.assertEqual(sidecar.read_text(encoding="utf-8"),
                             f"{digest}  {output.name}\n")
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), {"ok": True})

    def test_manifest_eligibility_is_aggregated_from_verified_cells(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cell = self._make_cell(root)
            output = root / "bio.json"
            argv = [
                "apply_bio_projection.py", "--out", str(output),
                "--cell", f"toy,Flickr25k,{cell}",
                "--expected_length", "15", "--save_projected",
            ]
            with mock.patch.object(sys, "argv", argv):
                bio_script.main()
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(payload["config"]["paper_result_eligible"])
            self.assertTrue(payload["cells"][0]["paper_result_eligible"])
            self.assertEqual(payload["config"]["paper_ineligibility_reasons"], [])
            expected = bio_script.sha256_file(output)
            self.assertEqual(
                Path(f"{output}.sha256").read_text(encoding="utf-8"),
                f"{expected}  {output.name}\n")


if __name__ == "__main__":
    unittest.main()
