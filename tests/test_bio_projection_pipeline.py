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
            self, root: Path, *, variant: str = "toy", bit: int = 36) -> Path:
        base_length = bit // 2
        selection_metric = (
            f"raw_{base_length}base_base_hamming_mAP_at_R")
        cell = root / "toy_refit_dnaeval"
        cell.mkdir()
        protocol_payload = {
            "variant": variant, "dataset": "Flickr25k", "bit": bit,
            "selection": "heldout-validation-only",
        }
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
        torch.save({
            "config": {
                "method": "toy", "dataset": "Flickr25k", "setting": "setting1",
                "bit": bit, "protocol_identity_sha256": protocol,
            },
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
        selection.write_text(json.dumps({
            "method": "toy", "dataset": "Flickr25k",
            "bit_length": bit, "base_length": base_length,
            "protocol_identity_sha256": protocol,
            "selection_only": True,
            "test_touched_during_selection": False,
            "selection_metric": selection_metric,
            "best_epoch": 4,
        }), encoding="utf-8")
        (cell / "p0_protocol_manifest.json").write_text(json.dumps({
            "variant": variant, "method": "toy", "dataset": "Flickr25k",
            "bit_length": bit, "base_length": base_length,
            "protocol_digest_sha256": protocol,
            "protocol_identity": protocol_payload,
            "selection_metric": selection_metric,
            "best_epoch_zero_based": 4,
            "selection_artifact": str(selection),
            "selection_artifact_sha256": bio_script.sha256_file(selection),
            "main_protocol_eligible": True,
            "protocol_deviations": [],
            "test_used_for_selection": False,
            "final_checkpoint_count": 1,
            "final_checkpoint": str(checkpoint),
            "final_checkpoint_sha256": bio_script.sha256_file(checkpoint),
            "refit_model_dir": str(checkpoint.parent),
            "extraction_dir": str(cell),
            "extraction_artifact_sha256": {
                "extract_db.npz": bio_script.sha256_file(cell / "extract_db.npz"),
                "extract_query.npz": bio_script.sha256_file(cell / "extract_query.npz"),
                "args_extract.txt": bio_script.sha256_file(cell / "args_extract.txt"),
            },
            "run_manifest_phase": "pre_bio_projection",
        }), encoding="utf-8")
        return cell

    def test_strict_cell_records_both_splits_and_content_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cell = self._make_cell(Path(temporary))
            with mock.patch.object(bio_script, "SAVE_PROJECTED", True):
                result = bio_script.run_cell(
                    "toy", str(cell), "Flickr25k", 0.4, 0.6, 3,
                    expected_length=18, require_provenance=True, device="cpu")

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
            cell = self._make_cell(Path(temporary), bit=48)
            with mock.patch.object(bio_script, "SAVE_PROJECTED", True):
                result = bio_script.run_cell(
                    "toy", str(cell), "Flickr25k", 0.4, 0.6, 3,
                    expected_length=24, require_provenance=True, device="cpu")
            self.assertEqual(result["L"], 24)
            self.assertEqual(result["gc_count_range"], [10, 14])
            self.assertTrue(result["paper_result_eligible"])
            manifest = json.loads(
                (cell / "p0_protocol_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["selection_metric"],
                "raw_24base_base_hamming_mAP_at_R")

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
                    require_provenance=True)

    def test_post_manifest_extraction_substitution_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cell = self._make_cell(Path(temporary))
            replacement = np.asarray([[0, 1] * 9], dtype=np.int64)
            _write_extraction(
                cell / "extract_db.npz", replacement,
                np.asarray([[1, 0]], dtype=np.int64))
            with self.assertRaisesRegex(ValueError, "artifact SHA-256 map mismatch"):
                bio_script.run_cell(
                    "toy", str(cell), "Flickr25k", 0.4, 0.6, 3,
                    require_provenance=True)

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
                "--cell", f"toy,Flickr25k,{cell}", "--save_projected",
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
