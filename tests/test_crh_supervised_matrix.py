import json
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path

from scripts.aggregate_baseline_p0_matrix import (
    CANONICAL_VARIANT_SOURCE_PROFILES as AGGREGATE_SOURCE_PROFILES,
    Key,
    KNOWN_NON_SCIENTIFIC_IMPLEMENTATION_TRANSITIONS,
    _audit_implementation_fingerprints,
    _expected_keys,
    _expected_semantic_condition,
    _implementation_fingerprint,
    _manifest_key,
    _markdown,
    _required_implementation_paths as aggregate_required_paths,
    _validate_manifest,
)
from scripts.run_baseline_p0_matrix import (
    BITS,
    CANONICAL_VARIANT_SOURCE_PROFILES as RUNNER_SOURCE_PROFILES,
    DATASETS,
    SUPERVISED_DATASETS,
    SUPERVISED_VARIANTS,
    U0_VARIANTS,
    _current_source_sha256,
    _jobs,
    _matches_canonical_source_profile,
    _matches_semantic_information_condition,
    _required_implementation_paths as runner_required_paths,
)


REPO = Path(__file__).resolve().parents[1]
CRH_VARIANT = "crh-supervised"


def _complete_record(key: Key, value: float = 0.5) -> dict[str, object]:
    snapshot = {"baseline/CRH.py": "a" * 64}
    return {
        "key": key.text,
        "panel": key.panel,
        "variant": key.variant,
        "dataset": key.dataset,
        "bit": key.bit,
        "seed": key.seed,
        "status": "complete_diagnostic_only",
        "main_protocol_eligible": False,
        "legacy_cache_diagnostic_only": True,
        "implementation_comparison_eligible": True,
        "implementation_sha256": snapshot,
        "implementation_fingerprint_sha256": _implementation_fingerprint(
            snapshot),
        "map_at_R_post": value,
    }


class CRHSupervisedMatrixTest(unittest.TestCase):
    def test_supervised_job_plan_has_eight_or_twenty_four_cells(self) -> None:
        single_seed = _jobs(
            "supervised", U0_VARIANTS, DATASETS, BITS, (42,))
        self.assertEqual(len(single_seed), 8)
        self.assertEqual(
            {
                (job.panel, job.variant, job.dataset, job.bit, job.seed)
                for job in single_seed
            },
            {
                ("supervised", CRH_VARIANT, dataset, bit, 42)
                for dataset in DATASETS for bit in BITS
            },
        )

        three_seed = _jobs(
            "supervised", U0_VARIANTS, DATASETS, BITS, (1, 2, 3))
        self.assertEqual(len(three_seed), 24)
        self.assertTrue(
            all(job.panel == "supervised" for job in three_seed))

    def test_historical_all_panel_stays_u0_plus_u2_only(self) -> None:
        jobs = _jobs("all", U0_VARIANTS, DATASETS, BITS, (42,))
        self.assertEqual(len(jobs), 78)
        self.assertFalse(any(job.variant == CRH_VARIANT for job in jobs))
        self.assertEqual({job.panel for job in jobs}, {"u0", "u2"})

    def test_aggregator_expected_key_counts_are_panel_scoped(self) -> None:
        self.assertEqual(len(_expected_keys((42,), ("supervised",))), 8)
        self.assertEqual(
            len(_expected_keys((1, 2, 3), ("supervised",))), 24)
        # Backward-compatible default: 72 U0 + 6 U2 cells.
        default = _expected_keys((42,))
        self.assertEqual(len(default), 78)
        self.assertFalse(any(key.panel == "supervised" for key in default))

    def test_semantic_condition_is_exact_and_cannot_spoof_u0(self) -> None:
        expected = _expected_semantic_condition("supervised", "MSCOCO")
        payload = {
            "variant": CRH_VARIANT,
            "dataset": "MSCOCO",
            "bit_length": 36,
            "seed": 42,
            "comparison_panel": "supervised",
            "information_tier": "S",
            "semantic_information_condition": expected,
            "protocol_identity": {
                "semantic_information_condition": expected,
            },
        }
        self.assertTrue(_matches_semantic_information_condition(
            payload, CRH_VARIANT, "MSCOCO"))
        self.assertEqual(
            _manifest_key(payload),
            Key("supervised", CRH_VARIANT, "MSCOCO", 36, 42),
        )

        spoofed = {
            **payload,
            "comparison_panel": "target-label-free",
            "information_tier": "U0",
        }
        self.assertFalse(_matches_semantic_information_condition(
            spoofed, CRH_VARIANT, "MSCOCO"))
        self.assertIsNone(_manifest_key(spoofed))

        mutated = dict(expected)
        mutated["uses_training_labels_in_center_reassignment"] = False
        payload["semantic_information_condition"] = mutated
        self.assertFalse(_matches_semantic_information_condition(
            payload, CRH_VARIANT, "MSCOCO"))

    def test_validator_rejects_direct_panel_variant_spoof(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "p0_run_manifest.json"
            path.write_text(json.dumps({
                "variant": CRH_VARIANT,
                "method": "crh",
                "dataset": "MSCOCO",
                "bit_length": 36,
                "seed": 42,
                "information_tier": "U0",
                "semantic_information_condition": (
                    _expected_semantic_condition("supervised", "MSCOCO")
                ),
            }), encoding="utf-8")
            record = _validate_manifest(
                path,
                Key("u0", CRH_VARIANT, "MSCOCO", 36, 42),
                verify_hashes=False,
            )
        self.assertEqual(record["status"], "invalid")
        self.assertTrue(any(
            "panel/variant mismatch" in error
            for error in record["validation_errors"]
        ))

    def test_crh_source_path_and_digest_are_identical_everywhere(self) -> None:
        aggregate = AGGREGATE_SOURCE_PROFILES[CRH_VARIANT]
        runner_path, runner_digest = RUNNER_SOURCE_PROFILES[CRH_VARIANT]
        source = REPO / runner_path
        actual = sha256(source.read_bytes()).hexdigest()
        self.assertEqual(runner_path, "baseline/CRH.py")
        self.assertEqual(str(aggregate["path"]), runner_path)
        self.assertEqual(str(aggregate["sha256"]), runner_digest)
        self.assertEqual(actual, runner_digest)
        self.assertEqual(
            aggregate_required_paths(CRH_VARIANT),
            runner_required_paths(CRH_VARIANT),
        )
        self.assertIn("baseline/CRH.py", runner_required_paths(CRH_VARIANT))

    def test_historical_common_dispatch_hashes_remain_resume_auditable(self) -> None:
        snapshot = {
            path: _current_source_sha256(path)
            for path in runner_required_paths("cibhash")
        }
        self.assertTrue(all(isinstance(value, str)
                            for value in snapshot.values()))
        for path in (
                "baseline/base_model.py",
                "scripts/run_modern_baseline_p0.py"):
            transition = KNOWN_NON_SCIENTIFIC_IMPLEMENTATION_TRANSITIONS[path]
            snapshot[path] = str(transition["before_sha256"])
        payload = {"protocol_identity": {"implementation_sha256": snapshot}}
        self.assertTrue(
            _matches_canonical_source_profile(payload, "cibhash"))

        snapshot["baseline/base_model.py"] = "f" * 64
        self.assertFalse(
            _matches_canonical_source_profile(payload, "cibhash"))

    def test_dispatch_only_cross_cell_transition_is_non_blocking(self) -> None:
        path = "scripts/run_modern_baseline_p0.py"
        transition = KNOWN_NON_SCIENTIFIC_IMPLEMENTATION_TRANSITIONS[path]
        first = Key("u0", "cibhash", "Flickr25k", 36, 42)
        second = Key("u0", "cibhash", "MSCOCO", 36, 42)
        records = {
            first: _complete_record(first),
            second: _complete_record(second),
        }
        for record, digest in (
                (records[first], transition["before_sha256"]),
                (records[second], transition["after_sha256"])):
            snapshot = {path: str(digest)}
            record["implementation_sha256"] = snapshot
            record["implementation_fingerprint_sha256"] = (
                _implementation_fingerprint(snapshot))

        audit = _audit_implementation_fingerprints(
            records,
            current_source_sha256={
                path: str(transition["after_sha256"])},
        )
        self.assertTrue(audit["comparison_safe"])
        self.assertEqual(audit["blocked_cells"], {})
        self.assertEqual(
            audit["known_non_scientific_warnings"][0]["classification"],
            "non_scientific_dispatch_extension_only",
        )
        self.assertTrue(
            audit["variant_fingerprints"][0]["comparison_safe"])

    def test_markdown_renders_crh_only_in_supervised_table(self) -> None:
        key = Key("supervised", CRH_VARIANT, "MSCOCO", 36, 42)
        payload = {
            "summary": {"expected": 8, "complete": 1, "missing": 7},
            "protocol": {"comparison_panels": ["supervised"]},
            "paper_table_admission": {
                "eligible": False,
                "reason_codes": ["matrix_incomplete_or_invalid"],
            },
            "implementation_audit": {
                "status": "consistent_with_current_source",
                "comparison_safe": True,
            },
            "source_profile_excluded_manifests": [],
            "invalid_cells": [],
            "duplicate_cells": [],
            "missing_cells": [],
        }
        markdown = _markdown(payload, {key: _complete_record(key)}, (42,))
        self.assertIn("## Supervised baseline — 36 bits / 18 bases", markdown)
        self.assertIn("| CRH (supervised) |", markdown)
        self.assertIn("benchmark training labels", markdown)
        self.assertNotIn("## U0 visual-only", markdown)
        self.assertNotIn("## U2 taxonomy-assisted", markdown)


if __name__ == "__main__":
    unittest.main()
