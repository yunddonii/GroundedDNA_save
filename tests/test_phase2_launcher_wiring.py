"""The Phase 2 launcher must branch on the ANALYSIS state, not on the NPZs.

The launcher decided "already complete" from the extraction validator alone, so
a cell whose NPZs validated but whose bio-eval and NMI had never run was skipped
outright. That is how all 15 Phase 2 cells came to be bound to their inputs and
still carry no admissible metric. A second defect sat right after it: the NMI
step was invoked with `--out <dir>/pairwise_nmi.json`, which the script now
refuses by design, so under `pipefail` the whole launcher aborted there.

These are behavioural tests. Rather than grep the shell source -- the mistake
that let three earlier "fixes" pass while nothing was wired to production --
they execute the real launcher against stub tools that record their argv, and
assert which stages ran and with which flags.
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

LAUNCHER = REPO / "scripts" / "phase2_f01_reinference.sh"
#: The stages that write numbers, and so must clear the strict gate unless the
#: cell being re-analysed is one of the preserved retrospective ones.
_METRIC_STAGES = {"eval_cell_bioproj.py", "pairwise_nmi.py",
                  "seal_cell_analysis.py"}
CELL = "cifar10/4"
LEGACY_REL = ("result/260811+cifar10_setting1_promptAblA_cifar_A_v4_"
              "P0refit_e4+bs+64+e+5+proj_lr+0.001")

_STUB_PY = """#!/usr/bin/env python3
import json, os, sys
log = os.environ["STUB_LOG"]
with open(log, "a") as handle:
    handle.write(json.dumps(sys.argv) + "\\n")
name = os.path.basename(sys.argv[0])
if name == "_phase2_cell_state.py":
    print(os.environ["STUB_STATE"])
elif name == "_phase2_read_arg.py":
    print({"stop_after_epoch": "4", "num_semantic_parts": "5",
           "codebook_size": "64"}[sys.argv[2]])
"""


class LauncherStageSelection(unittest.TestCase):
    """Run the real launcher; stub only the tools it invokes."""

    def setUp(self) -> None:
        self.tmp = Path(os.environ.get("TMPDIR", "/tmp")) / f"p2launch{os.getpid()}"
        if self.tmp.exists():
            shutil.rmtree(self.tmp)
        (self.tmp / "scripts").mkdir(parents=True)
        (self.tmp / LEGACY_REL).mkdir(parents=True)
        for name in ("args.txt", "config.pt", "model_state_dict.pth"):
            (self.tmp / LEGACY_REL / name).write_text("stub")

        shutil.copy(LAUNCHER, self.tmp / "scripts" / LAUNCHER.name)
        for name in ("_phase2_cell_state.py", "_phase2_read_arg.py",
                     "eval_cell_bioproj.py", "pairwise_nmi.py",
                     "seal_cell_analysis.py"):
            path = self.tmp / "scripts" / name
            path.write_text(_STUB_PY)
            path.chmod(0o755)
        extraction = self.tmp / "extraction_siglip2.py"
        extraction.write_text(_STUB_PY)
        extraction.chmod(0o755)

        self.log = self.tmp / "calls.jsonl"
        self.root = self.tmp / "phase2root"

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, state: str) -> list[list[str]]:
        env = dict(os.environ)
        env.update(PY=sys.executable, STUB_LOG=str(self.log), STUB_STATE=state,
                   PHASE2_ROOT=str(self.root))
        proc = subprocess.run(
            ["bash", str(self.tmp / "scripts" / LAUNCHER.name), "0", CELL],
            capture_output=True, text=True, env=env, timeout=120)
        self.assertEqual(proc.returncode, 0,
                         f"launcher failed for state={state}: {proc.stderr}")
        if not self.log.exists():
            return []
        return [json.loads(line) for line in
                self.log.read_text().splitlines() if line.strip()]

    @staticmethod
    def _stages(calls: list[list[str]]) -> set[str]:
        return {os.path.basename(call[0]) for call in calls}

    def test_complete_runs_nothing_beyond_the_state_probe(self) -> None:
        stages = self._stages(self._run("complete"))
        self.assertEqual(stages, {"_phase2_cell_state.py"},
                         "a complete cell must not re-run any stage")

    def test_analysis_state_reruns_metrics_without_re_extracting(self) -> None:
        """The defect: this state used to be indistinguishable from complete."""
        calls = self._run("analysis")
        stages = self._stages(calls)
        self.assertNotIn("extraction_siglip2.py", stages,
                         "valid NPZs must not be re-inferred")
        self.assertIn("eval_cell_bioproj.py", stages,
                      "a cell without metrics must have its metrics computed")
        self.assertIn("pairwise_nmi.py", stages)
        self.assertIn("seal_cell_analysis.py", stages,
                      "the seal is what makes the cell complete")
        # The preserved cells carry retrospective manifests, so the metric
        # stages have to be told that explicitly or they refuse.
        for call in calls:
            if os.path.basename(call[0]) in _METRIC_STAGES:
                self.assertIn("--allow-backfilled", call)

    def test_absent_state_runs_every_stage_under_the_strict_gate(self) -> None:
        calls = self._run("absent")
        self.assertEqual(
            self._stages(calls),
            {"_phase2_cell_state.py", "_phase2_read_arg.py",
             "extraction_siglip2.py", "eval_cell_bioproj.py",
             "pairwise_nmi.py", "seal_cell_analysis.py"})
        # The probe is deliberately permissive -- it has to recognise the
        # preserved retrospective cells in order to classify them at all. The
        # gate that matters is on the stages that WRITE numbers.
        for call in calls:
            if os.path.basename(call[0]) in _METRIC_STAGES:
                self.assertNotIn(
                    "--allow-backfilled", call,
                    "a cell this launcher infers itself must clear the "
                    "strict gate")

    def test_nmi_is_not_invoked_with_the_path_it_refuses(self) -> None:
        """`--out <dir>/pairwise_nmi.json` aborts the launcher under pipefail."""
        for state in ("absent", "analysis"):
            with self.subTest(state=state):
                self.log.unlink(missing_ok=True)
                for call in self._run(state):
                    if os.path.basename(call[0]) == "pairwise_nmi.py":
                        self.assertNotIn("--out", call)


class RealNmiRefusesThatPath(unittest.TestCase):
    """Pin the refusal the launcher used to walk into."""

    def test_out_equal_to_the_per_result_file_is_refused(self) -> None:
        tmp = Path(os.environ.get("TMPDIR", "/tmp")) / f"p2nmi{os.getpid()}"
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            proc = subprocess.run(
                [sys.executable, str(REPO / "scripts" / "pairwise_nmi.py"),
                 "--results", str(tmp),
                 "--out", str(tmp / "pairwise_nmi.json")],
                capture_output=True, text=True, cwd=REPO, timeout=300)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("per-result", proc.stdout + proc.stderr)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class CellStateProbe(unittest.TestCase):
    """`_phase2_cell_state.py` must work from any cwd and separate the states."""

    def setUp(self) -> None:
        self.tmp = Path(os.environ.get("TMPDIR", "/tmp")) / f"p2state{os.getpid()}"
        self.tmp.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _probe(self, cell: Path, cwd: Path) -> str:
        proc = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "_phase2_cell_state.py"),
             str(cell), "--allow-backfilled"],
            capture_output=True, text=True, cwd=cwd, timeout=300)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return proc.stdout.strip()

    def test_importable_from_a_foreign_cwd(self) -> None:
        """The heredoc it replaced died with ModuleNotFoundError outside REPO."""
        missing = self.tmp / "nothing_here"
        self.assertEqual(self._probe(missing, cwd=self.tmp), "absent")

    def test_a_directory_with_junk_is_invalid_not_complete(self) -> None:
        cell = self.tmp / "cifar10_N4"
        cell.mkdir()
        (cell / "extract_db.npz").write_text("not-a-npz")
        (cell / "extraction_manifest_db.json").write_text("not-json")
        self.assertEqual(self._probe(cell, cwd=REPO), "invalid")

    def test_a_bound_cell_without_metrics_reports_analysis(self) -> None:
        """Exactly the state of the 15 preserved cells before the recompute.

        The fixture is built here rather than picked out of the live diagnostic
        root: a metric recompute writing markers concurrently would otherwise
        flip a chosen cell from `analysis` to `complete` mid-test.
        """
        from dna_utils.extraction_validation import describe_failure

        root = REPO / "result_diagnostic" / "phase2_F01_only"
        if not root.is_dir():
            self.skipTest("the Phase 2 diagnostic root is not present")
        source = next(
            (cell for cell in sorted(root.iterdir())
             if cell.is_dir()
             and not describe_failure(str(cell), allow_backfilled=True)),
            None)
        if source is None:
            self.skipTest("no validating cell to copy the binding from")

        # The manifests carry absolute NPZ/config paths and their digests, so a
        # copy of the small files alone is still bound to the same bytes.
        cell = self.tmp / source.name
        cell.mkdir()
        for path in sorted(source.iterdir()):
            if path.is_file() and path.suffix in {".json", ".txt", ".log"}:
                shutil.copy(path, cell / path.name)
        (cell / "analysis_complete.json").unlink(missing_ok=True)

        self.assertIsNone(
            describe_failure(str(cell), allow_backfilled=True),
            "the copied fixture must itself be a validly bound extraction")
        self.assertEqual(self._probe(cell, cwd=REPO), "analysis")


if __name__ == "__main__":
    unittest.main()
