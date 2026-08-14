"""The tracked inventory must cover the metric half too, and land atomically.

`docs/phase2_extraction_inventory.json` existed so a fresh clone could re-check
the Phase 2 snapshot without the ~37 GiB of untracked arrays. It recorded only
the extraction side, which leaves the question the audit actually asks -- which
numbers came out, under which evaluator, under which protocol constants --
unanswerable, since the metric JSONs are exactly as untracked as the NPZs
(§19.7). It also wrote in place, so an interrupted run left a shorter file that
still parses and still reads as complete.

The fixtures are built by copying a real cell's small files into a temp root:
the manifests carry absolute NPZ paths and digests, so the copy stays bound to
the same bytes without duplicating gigabytes, and a metric recompute running
concurrently cannot flip the fixture underneath the test.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dna_utils.extraction_validation import describe_failure  # noqa: E402

PHASE2_ROOT = REPO / "result_diagnostic" / "phase2_F01_only"
SCRIPT = REPO / "scripts" / "write_phase2_inventory.py"
_SMALL = {".json", ".txt", ".log"}


def _a_validating_cell() -> Path | None:
    if not PHASE2_ROOT.is_dir():
        return None
    for cell in sorted(PHASE2_ROOT.iterdir()):
        if cell.is_dir() and not describe_failure(str(cell),
                                                  allow_backfilled=True):
            return cell
    return None


class InventoryCoversBothHalves(unittest.TestCase):

    def setUp(self) -> None:
        source = _a_validating_cell()
        if source is None:
            self.skipTest("no validating Phase 2 cell to build a fixture from")
        if not (source / "analysis_complete.json").is_file():
            self.skipTest(f"{source.name} has no analysis marker yet")

        self.tmp = Path(os.environ.get("TMPDIR", "/tmp")) / f"p2inv{os.getpid()}"
        shutil.rmtree(self.tmp, ignore_errors=True)
        self.root = self.tmp / "root"
        self.root.mkdir(parents=True)

        def copy_small(dest_name: str) -> Path:
            dest = self.root / dest_name
            dest.mkdir()
            for path in sorted(source.iterdir()):
                if path.is_file() and path.suffix in _SMALL:
                    shutil.copy(path, dest / path.name)
            # The marker binds to its own absolute run_dir on purpose, so a
            # marker dropped into someone else's cell is refused. Relocating
            # the fixture therefore means restating where it lives; every other
            # digest in the binding is recomputed from these copied bytes and
            # so is unchanged.
            marker = dest / "analysis_complete.json"
            if marker.is_file():
                payload = json.loads(marker.read_text())
                payload["input_binding"]["run_dir"] = str(dest)
                marker.write_text(
                    json.dumps(payload, indent=2, sort_keys=True))
            return dest

        self.with_metrics = copy_small(source.name)
        self.source_name = source.name
        self.out = self.tmp / "inventory.json"

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _generate(self) -> dict:
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "--root", str(self.root),
             "--out", str(self.out)],
            capture_output=True, text=True, cwd=REPO, timeout=600)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        return json.loads(self.out.read_text())

    def _cell(self, payload: dict, name: str) -> dict:
        for entry in payload["cells"]:
            if entry["cell"] == name:
                return entry
        self.fail(f"{name} missing from the inventory")

    def test_records_the_metrics_evaluator_and_protocol(self) -> None:
        entry = self._cell(self._generate(), self.source_name)
        analysis = entry["analysis"]
        self.assertTrue(analysis["admitted"])
        # The numbers themselves, so a clone can compare without the arrays.
        self.assertIn("map_at_R_bioproj", analysis["metrics"])
        self.assertIn("dna_unique_db", analysis["metrics"])
        # Which code produced them.
        self.assertTrue(analysis["analysis_sources"])
        # Under which protocol constants -- a GC window change silently
        # rewrites every DNA number otherwise.
        self.assertEqual(analysis["protocol"]["gc_policy_version"],
                         "gc-40-60-inclusive-v1")
        # And the digests of the metric files themselves.
        self.assertTrue(
            any(v for v in analysis["metric_files"].values()),
            "no metric file digests recorded")

    def test_a_cell_without_a_marker_is_recorded_as_inadmissible(self) -> None:
        """It used to be indistinguishable from a cell with numbers."""
        stripped = self.root / self.source_name
        (stripped / "analysis_complete.json").unlink()
        payload = self._generate()
        analysis = self._cell(payload, self.source_name)["analysis"]
        self.assertFalse(analysis["admitted"])
        self.assertFalse(analysis["marker_present"])
        self.assertIsNotNone(analysis["refusal"])
        self.assertIsNone(analysis["metrics"])
        self.assertEqual(payload["cells_with_admissible_metrics"], 0)

    def test_records_the_execution_context_and_dirty_state(self) -> None:
        execution = self._generate()["execution"]
        self.assertTrue(execution["generated_at_utc"])
        self.assertIn("write_phase2_inventory.py", execution["command"])
        self.assertIsNotNone(execution["git_head"])
        # A digest set taken from a dirty tree is not reproducible from the
        # commit it names, and has to say so rather than imply otherwise.
        self.assertIsInstance(execution["git_tree_dirty"], bool)
        self.assertEqual(execution["reproducible_from_git_head"],
                         not execution["git_tree_dirty"])

    def test_the_write_is_atomic_and_leaves_no_temporary_behind(self) -> None:
        self._generate()
        leftovers = [p.name for p in self.tmp.iterdir() if ".tmp" in p.name]
        self.assertEqual(leftovers, [])

    def test_an_existing_inventory_is_replaced_whole(self) -> None:
        """os.replace, not truncate-then-write: a reader sees one or the other."""
        self.out.write_text(json.dumps({"schema_version": 1, "cells": []}))
        payload = self._generate()
        self.assertEqual(payload["schema_version"], 2)
        self.assertTrue(payload["cells"])


if __name__ == "__main__":
    unittest.main()
