from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from dna_utils.bio_constraints import is_valid_batch
from scripts.slot_intervention_eval import (
    main as slot_intervention_main,
    run_arm,
    swap_segments,
    topk_by_hamming,
    valid_swap_candidates,
)


class StableHammingRankingTest(unittest.TestCase):
    def test_boundary_ties_use_database_row_order(self) -> None:
        query = np.array([[0, 0, 0]], dtype=np.int64)
        database = np.array([
            [1, 0, 0],  # distance 1, DB row 0
            [0, 1, 0],  # distance 1, DB row 1
            [0, 0, 1],  # distance 1, DB row 2
            [1, 1, 0],  # distance 2
        ], dtype=np.int64)
        np.testing.assert_array_equal(
            topk_by_hamming(query, database, 2),
            np.array([[0, 1]], dtype=np.int64),
        )

    def test_invalid_k_fails_closed(self) -> None:
        codes = np.zeros((2, 3), dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "k must be"):
            topk_by_hamming(codes, codes, 3)


class ExactBioValidSlotSwapTest(unittest.TestCase):
    def setUp(self) -> None:
        # Nine GC bases, no long homopolymer: valid for the script's 18-base
        # [ceil(.4444L), floor(.5556L)] == [8, 10] GC window.
        self.query = np.array([0, 1] * 9, dtype=np.int64)
        self.slices = [slice(3 * slot, 3 * (slot + 1)) for slot in range(6)]
        self.validity = lambda rows: is_valid_batch(
            rows, 0.4444, 0.5556, 3,
        )
        self.assertTrue(self.validity(self.query[None, :])[0])

    def test_candidate_filter_rejects_gc_invalid_exact_splice(self) -> None:
        donors = np.repeat(self.query[None, :], 2, axis=0)
        donors[0, :3] = [2, 3, 2]  # two GC -> total 10, valid
        donors[1, :3] = [1, 2, 1]  # three GC -> total 11, invalid
        np.testing.assert_array_equal(
            valid_swap_candidates(
                self.query, donors, 0, self.slices, self.validity,
            ),
            np.array([True, False]),
        )

    def test_exact_swap_never_changes_non_target_positions(self) -> None:
        donors = np.repeat(self.query[None, :], 1, axis=0)
        donors[0, 3:6] = [2, 3, 2]
        swapped = swap_segments(
            self.query[None, :], donors,
            np.array([0]), np.array([0]), np.array([1]), self.slices,
        )
        np.testing.assert_array_equal(swapped[0, :3], self.query[:3])
        np.testing.assert_array_equal(swapped[0, 6:], self.query[6:])
        np.testing.assert_array_equal(swapped[0, 3:6], donors[0, 3:6])

    def test_scoring_refuses_an_invalid_bio_intervention(self) -> None:
        database = np.repeat(self.query[None, :], 2, axis=0)
        database[0, :3] = [1, 2, 1]  # makes the query's GC count 11
        database_labels = np.array([[1, 0], [0, 1]], dtype=np.int64)
        with self.assertRaisesRegex(RuntimeError, "invalid DNA"):
            run_arm(
                self.query[None, :],
                database,
                database_labels,
                np.array([0]),
                np.array([0]),
                np.array([0]),
                np.array([0]),
                self.slices,
                np.array([[1]]),
                np.array([[0.0, 1.0]]),
                1,
                10,
                42,
                validity_fn=self.validity,
            )


class InterventionDoseMatchingEndToEndTest(unittest.TestCase):
    @staticmethod
    def _write_fixture(root: Path, *, baseline_noop: bool) -> tuple[Path, Path]:
        ours = root / "ours"
        baseline = root / "baseline"
        ours.mkdir()
        baseline.mkdir()

        n_db, n_query, n_train = 4, 2, 4
        db_paths = np.asarray([f"db_{i}.jpg" for i in range(n_db)])
        query_paths = np.asarray([f"query_{i}.jpg" for i in range(n_query)])
        db_labels = np.tile(np.asarray([[0, 1]], dtype=np.int64), (n_db, 1))
        query_labels = np.tile(
            np.asarray([[1, 0]], dtype=np.int64), (n_query, 1),
        )
        train_labels = np.tile(
            np.asarray([[0, 1]], dtype=np.int64), (n_train, 1),
        )
        db_bases = np.full((n_db, 18), 3, dtype=np.int64)
        query_bases = np.zeros((n_query, 18), dtype=np.int64)
        train_bases = np.full((n_train, 18), 3, dtype=np.int64)
        np.savez(
            ours / "extract_db.npz", base_indices=db_bases,
            multi_hot_labels=db_labels, image_paths=db_paths,
        )
        np.savez(
            ours / "extract_query.npz", base_indices=query_bases,
            multi_hot_labels=query_labels, image_paths=query_paths,
        )
        np.savez(
            ours / "extract_train.npz", base_indices=train_bases,
            multi_hot_labels=train_labels,
        )

        db_bits = np.zeros((n_db, 36), dtype=np.int64)
        if not baseline_noop:
            db_bits.fill(1)
        query_bits = np.zeros((n_query, 36), dtype=np.int64)
        np.savez(
            baseline / "extract_db.npz", hash_2bit=db_bits,
            image_paths=db_paths,
        )
        np.savez(
            baseline / "extract_query.npz", hash_2bit=query_bits,
            image_paths=query_paths,
        )
        return ours, baseline

    def _run(self, root: Path, *, baseline_noop: bool) -> dict:
        ours, baseline = self._write_fixture(
            root, baseline_noop=baseline_noop,
        )
        output = root / "result.json"
        argv = [
            "slot_intervention_eval.py",
            "--ours_dir", str(ours),
            "--baseline_dirs", str(baseline),
            "--baseline_names", "cibhash",
            "--dataset", "synthetic",
            "--out", str(output),
            "--k", "2",
            "--n_query", "2",
            "--n_boot", "10",
        ]
        with patch.object(sys, "argv", argv):
            slot_intervention_main()
        return json.loads(output.read_text(encoding="utf-8"))

    def test_every_compared_arm_has_exactly_matched_nonzero_dose(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = self._run(Path(directory), baseline_noop=False)
        self.assertEqual(
            payload["intervention_dose_policy"],
            "exact_per_query_changed_slot_fraction_matched",
        )
        for slot in payload["results"].values():
            self.assertGreater(slot["paired_common_subset_n"], 0)
            doses = [
                arm["fraction_changed"] for arm in slot["arms"].values()
            ]
            self.assertGreater(doses[0], 0.0)
            self.assertTrue(all(dose == doses[0] for dose in doses))

    def test_flat_noop_controls_are_excluded_not_compared(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = self._run(Path(directory), baseline_noop=True)
        for slot in payload["results"].values():
            self.assertEqual(slot["paired_common_subset_n"], 0)
            self.assertNotIn("arms", slot)


if __name__ == "__main__":
    unittest.main()
