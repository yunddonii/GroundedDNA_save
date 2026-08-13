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


#: The counts below (8 supervised cells, 78 U0+U2 cells at one seed) describe a
#: TWO-budget slice. They broke when `BITS` grew from (36, 48) to
#: (30, 36, 40, 48) and observed exactly double. Doubling the fixtures would
#: have hidden the cause; F15 makes the requested slice decide the job plan and
#: the aggregate expected set, so the fixture states the slice it means.
LEGACY_SLICE = (36, 48)


class CRHSupervisedMatrixTest(unittest.TestCase):
    def test_supervised_job_plan_has_eight_or_twenty_four_cells(self) -> None:
        single_seed = _jobs(
            "supervised", U0_VARIANTS, DATASETS, LEGACY_SLICE, (42,))
        self.assertEqual(len(single_seed), 8)
        self.assertEqual(
            {
                (job.panel, job.variant, job.dataset, job.bit, job.seed)
                for job in single_seed
            },
            {
                ("supervised", CRH_VARIANT, dataset, bit, 42)
                for dataset in DATASETS for bit in LEGACY_SLICE
            },
        )

        three_seed = _jobs(
            "supervised", U0_VARIANTS, DATASETS, LEGACY_SLICE, (1, 2, 3))
        self.assertEqual(len(three_seed), 24)
        self.assertTrue(
            all(job.panel == "supervised" for job in three_seed))

    def test_historical_all_panel_stays_u0_plus_u2_only(self) -> None:
        jobs = _jobs("all", U0_VARIANTS, DATASETS, LEGACY_SLICE, (42,))
        self.assertEqual(len(jobs), 78)
        self.assertFalse(any(job.variant == CRH_VARIANT for job in jobs))
        self.assertEqual({job.panel for job in jobs}, {"u0", "u2"})

    def test_aggregator_expected_key_counts_are_panel_scoped(self) -> None:
        self.assertEqual(len(_expected_keys((42,), ("supervised",), LEGACY_SLICE)), 8)
        self.assertEqual(
            len(_expected_keys((1, 2, 3), ("supervised",), LEGACY_SLICE)), 24)
        # Backward-compatible default: 72 U0 + 6 U2 cells.
        default = _expected_keys((42,), bits=LEGACY_SLICE)
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
        runner_path = "scripts/run_modern_baseline_p0.py"
        runner_transition = KNOWN_NON_SCIENTIFIC_IMPLEMENTATION_TRANSITIONS[
            runner_path]
        runner_history = tuple(runner_transition["reviewed_sha256"])

        # Both reviewed historical runner snapshots are non-scientific for a
        # pre-existing unaffected U0 method.
        snapshot = {
            path: _current_source_sha256(path)
            for path in runner_required_paths("cimon")
        }
        self.assertTrue(all(isinstance(value, str)
                            for value in snapshot.values()))
        payload = {"protocol_identity": {"implementation_sha256": snapshot}}
        for digest in runner_history[:-1]:
            snapshot[runner_path] = str(digest)
            self.assertTrue(
                _matches_canonical_source_profile(payload, "cimon"))

        # The base-model CRH dispatch addition remains non-scientific for
        # Cimon and is still independently content-scoped.
        base_transition = KNOWN_NON_SCIENTIFIC_IMPLEMENTATION_TRANSITIONS[
            "baseline/base_model.py"]
        snapshot[runner_path] = str(runner_history[-1])
        snapshot["baseline/base_model.py"] = str(
            base_transition["before_sha256"])
        self.assertTrue(_matches_canonical_source_profile(payload, "cimon"))
        snapshot["baseline/base_model.py"] = "f" * 64
        self.assertFalse(_matches_canonical_source_profile(payload, "cimon"))

        # The same pre-dispatch base-model snapshot is scientific for CRH:
        # that source does not contain a CRH branch and therefore cannot have
        # produced a valid supervised CRH result.
        crh_snapshot = {
            path: _current_source_sha256(path)
            for path in runner_required_paths(CRH_VARIANT)
        }
        crh_payload = {
            "protocol_identity": {
                "implementation_sha256": crh_snapshot,
            },
        }
        crh_snapshot["baseline/base_model.py"] = str(
            base_transition["before_sha256"])
        self.assertFalse(
            _matches_canonical_source_profile(crh_payload, CRH_VARIANT))

        # Runner A predates CRH; runner A/B both carry the stale CIBHash
        # horizon (100, corrected to 60 by the 2026-08-05 auditfix).  Neither
        # may be admitted for those scientific variants.  Name them explicitly:
        # `runner_history[:-1]` meant "the stale ones" only while the history
        # ended at the auditfix, and it silently started rejecting the two
        # later SUPPORTED_BITS widenings, which are valid for CIBHash.
        stale_cibhash_horizon = runner_history[:2]
        for variant, accepted_digest, rejected_digests in (
            ("crh-supervised", runner_history[1], (runner_history[0],)),
            ("cibhash", runner_history[-1], stale_cibhash_horizon),
        ):
            scoped_snapshot = {
                path: _current_source_sha256(path)
                for path in runner_required_paths(variant)
            }
            scoped_payload = {
                "protocol_identity": {
                    "implementation_sha256": scoped_snapshot,
                },
            }
            scoped_snapshot[runner_path] = str(accepted_digest)
            self.assertTrue(
                _matches_canonical_source_profile(scoped_payload, variant))
            for digest in rejected_digests:
                scoped_snapshot[runner_path] = str(digest)
                self.assertFalse(
                    _matches_canonical_source_profile(scoped_payload, variant))

    def test_variant_scoped_runner_transition_is_non_blocking_for_cimon(self) -> None:
        path = "scripts/run_modern_baseline_p0.py"
        transition = KNOWN_NON_SCIENTIFIC_IMPLEMENTATION_TRANSITIONS[path]
        digests = tuple(transition["reviewed_sha256"])
        keys = (
            Key("u0", "cimon", "Flickr25k", 36, 42),
            Key("u0", "cimon", "MSCOCO", 36, 42),
            Key("u0", "cimon", "NUSWIDE", 36, 42),
        )
        records = {key: _complete_record(key) for key in keys}
        for key, digest in zip(keys, digests):
            record = records[key]
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
            "non_scientific_for_reviewed_variants_only",
        )
        self.assertTrue(
            audit["variant_fingerprints"][0]["comparison_safe"])

    def test_base_model_dispatch_transition_is_blocking_only_for_crh(self) -> None:
        path = "baseline/base_model.py"
        transition = KNOWN_NON_SCIENTIFIC_IMPLEMENTATION_TRANSITIONS[path]
        old = str(transition["before_sha256"])
        new = str(transition["after_sha256"])

        cimon = Key("u0", "cimon", "Flickr25k", 36, 42)
        cimon_record = _complete_record(cimon)
        cimon_snapshot = {path: old}
        cimon_record["implementation_sha256"] = cimon_snapshot
        cimon_record["implementation_fingerprint_sha256"] = (
            _implementation_fingerprint(cimon_snapshot))
        cimon_audit = _audit_implementation_fingerprints(
            {cimon: cimon_record}, current_source_sha256={path: new},
        )
        self.assertTrue(cimon_audit["comparison_safe"])
        self.assertEqual(cimon_audit["blocked_cells"], {})

        crh = Key("supervised", CRH_VARIANT, "MSCOCO", 36, 42)
        crh_record = _complete_record(crh)
        crh_snapshot = {path: old}
        crh_record["implementation_sha256"] = crh_snapshot
        crh_record["implementation_fingerprint_sha256"] = (
            _implementation_fingerprint(crh_snapshot))
        crh_audit = _audit_implementation_fingerprints(
            {crh: crh_record}, current_source_sha256={path: new},
        )
        self.assertFalse(crh_audit["comparison_safe"])
        self.assertEqual(set(crh_audit["blocked_cells"]), {crh.text})
        self.assertEqual(crh_audit["known_non_scientific_warnings"], [])

    def test_cibhash_runner_horizon_transition_is_scientific_and_blocking(self) -> None:
        path = "scripts/run_modern_baseline_p0.py"
        transition = KNOWN_NON_SCIENTIFIC_IMPLEMENTATION_TRANSITIONS[path]
        old = str(tuple(transition["reviewed_sha256"])[1])
        new = str(transition["after_sha256"])
        first = Key("u0", "cibhash", "Flickr25k", 36, 42)
        second = Key("u0", "cibhash", "MSCOCO", 36, 42)
        records = {first: _complete_record(first), second: _complete_record(second)}
        for key, digest in ((first, old), (second, new)):
            snapshot = {path: digest}
            records[key]["implementation_sha256"] = snapshot
            records[key]["implementation_fingerprint_sha256"] = (
                _implementation_fingerprint(snapshot))

        audit = _audit_implementation_fingerprints(
            records, current_source_sha256={path: new},
        )
        self.assertFalse(audit["comparison_safe"])
        self.assertEqual(audit["known_non_scientific_warnings"], [])
        self.assertEqual(set(audit["blocked_cells"]), {first.text, second.text})

    def test_current_cibhash_does_not_block_old_unaffected_cimon(self) -> None:
        path = "scripts/run_modern_baseline_p0.py"
        transition = KNOWN_NON_SCIENTIFIC_IMPLEMENTATION_TRANSITIONS[path]
        old = str(tuple(transition["reviewed_sha256"])[1])
        new = str(transition["after_sha256"])
        cib = Key("u0", "cibhash", "Flickr25k", 36, 42)
        cimon = Key("u0", "cimon", "MSCOCO", 36, 42)
        records = {cib: _complete_record(cib), cimon: _complete_record(cimon)}
        for key, digest in ((cib, new), (cimon, old)):
            snapshot = {path: digest}
            records[key]["implementation_sha256"] = snapshot
            records[key]["implementation_fingerprint_sha256"] = (
                _implementation_fingerprint(snapshot))

        audit = _audit_implementation_fingerprints(
            records, current_source_sha256={path: new},
        )
        self.assertTrue(audit["comparison_safe"])
        self.assertEqual(audit["blocked_cells"], {})
        self.assertEqual(
            audit["known_non_scientific_warnings"][0]["affected_cells"],
            [cimon.text],
        )

    def test_markdown_renders_crh_only_in_supervised_table(self) -> None:
        key = Key("supervised", CRH_VARIANT, "MSCOCO", 36, 42)
        payload = {
            "summary": {"expected": 8, "complete": 1, "missing": 7},
            # F15: the payload now states the slice it was aggregated over,
            # and the markdown sections follow it instead of a constant.
            "protocol": {"comparison_panels": ["supervised"],
                         "requested_bit_slice": [36, 48]},
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
