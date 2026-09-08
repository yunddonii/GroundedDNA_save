"""Bi-half / NUS-WIDE is a disclosed adaptation, not an eligibility blocker.

Master audit :442 settled this pair: the public Bi-half release ships no
NUS-WIDE trainer, so NUS-WIDE runs the paper/Flickr profile and the source
boundary is DISCLOSED in the table and the text. The runner had instead treated
the same adaptation as a main-eligibility blocker and refused the cell before
training, which made the 108-cell expected matrix unreachable and
`--require-paper-eligible` permanently false.

These tests pin both halves of the fix so it cannot silently regress into
either failure mode: refusing the cell again, or running the adapter without
saying so.
"""
from __future__ import annotations

import hashlib
import pathlib
import re
import unittest


REPO = pathlib.Path(__file__).resolve().parents[1]
RUNNER = REPO / "scripts/run_modern_baseline_p0.py"
AGGREGATOR = REPO / "scripts/aggregate_baseline_p0_matrix.py"
ADAPTER_TOKEN = "bihalf_public_release_has_no_nuswide_training_script"


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class BiHalfNusWideSourceBoundary(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = RUNNER.read_text(encoding="utf-8")

    def test_the_pair_is_not_an_eligibility_blocker(self):
        """The adaptation must never re-enter `input_eligibility_blockers`.

        That is the exact line that refused three cells before training and
        made the strict gate unreachable.
        """
        for match in re.finditer(
                r"input_eligibility_blockers\.append\(\s*\n?\s*(.+?)\)",
                self.runner, re.S):
            self.assertNotIn(
                ADAPTER_TOKEN, match.group(1),
                "the Bi-half/NUS-WIDE adapter is a disclosed source boundary, "
                "not an eligibility blocker; appending it here refuses the "
                "cell before training and makes the 108-cell matrix "
                "unreachable")

    def test_the_pair_is_recorded_as_a_source_boundary(self):
        """It must still be declared -- silence would hide the adapter."""
        self.assertIn("source_boundary_adaptations", self.runner)
        self.assertIn(ADAPTER_TOKEN, self.runner)
        guard = re.search(
            r"if args\.variant == 'bihalf' and args\.dataset == 'NUSWIDE':\s*\n"
            r"\s*source_boundary_adaptations\.append\(", self.runner)
        self.assertIsNotNone(
            guard,
            "the (bihalf, NUSWIDE) branch must append to "
            "source_boundary_adaptations, so every manifest carries it")

    def test_the_manifest_carries_the_field(self):
        self.assertIn(
            '"source_boundary_adaptations": source_boundary_adaptations',
            self.runner,
            "a boundary nobody can read from the manifest is not disclosed")

    def test_the_current_runner_digest_is_reviewed(self):
        """A runner the aggregator has not reviewed blocks its whole variant.

        `_audit_implementation_fingerprints` treats an unreviewed differing
        path as unknown drift, which sets comparison_safe=False and blocks
        every Bi-half cell -- including the nine already completed.
        """
        aggregator = AGGREGATOR.read_text(encoding="utf-8")
        self.assertIn(
            _sha256(RUNNER), aggregator,
            "run_modern_baseline_p0.py changed without being registered in the "
            "aggregator's reviewed_sha256 allowlist; re-running the three "
            "Bi-half/NUS-WIDE cells against it would block all twelve")


if __name__ == "__main__":
    unittest.main()
