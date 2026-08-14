import copy
import json
import statistics
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path

from scripts.aggregate_native_dna_p0 import (
    COMMON_BIO_PROJECTION,
    DATASETS,
    DEFAULT_SEEDS,
    DISPLAY,
    Key,
    METHOD_PROTOCOL_LOCK_SHA256,
    method_protocol_lock_for,
    TEST_ACCESS_CONTRACT,
    _aggregate_cell,
    _canonical_digest,
    _markdown,
    _resolve_records,
    _validate_manifest,
    aggregate,
)
from dna_utils.native_protocol import resolve_native_protocol
from scripts.run_native_dna_p0 import (
    SOURCE_PROFILES,
    SOURCE_REPRODUCTION_AUDIT,
)

# F18: these fixtures encode the 18-base contract, which is what the historical
# native cells were run under. They used to say so only by hard-coding 18 while
# the validator read its length from the environment, so they failed outright
# once the paper moved to 15 bases. Naming the protocol keeps the fixture
# meaningful and lets the 15-base contract be exercised beside it.
PROTOCOL = resolve_native_protocol(18)


HISTORICAL_IMPLEMENTATION_SHA256 = {
    "baseline/base_model.py": "c9f39c05a27cca24ab0084bdd00462b634953499021a67c1cdc18dfc33837a47",
    "baseline/cache_provenance.py": "4fe40c862a23e265ecf0559232d8b6ac5b84b282df720670c3aed58e3daf588d",
    "baseline/native_dna.py": "3a0c49c4b49ba54db0eb1ce6a127c62279023aee2e6a5f90e2b5fa5943cf91b5",
    "dataloaders.py": "8632d4328e529c5b9967f4ffadcc5eeda6809c51fa844d3fdbdda2462e5f8315",
    "dna_utils/bio_constraints.py": "f7863bd363ebe42989f232f550a0af837dde088abf01bada8a0c281f6f164030",
    "scripts/run_native_dna_p0.py": "9a155dd1255163701a214d62fb380d6f1ef6c8d9c6500bb6b9a36c3d9ed444a7",
    "scripts/train_native_dna_baseline.py": "c85d8243047a6ed6a61b34de64d1bdda5b963a0a5b8dc1caab56692734c5f051",
    "val_split.py": "95e415c6f43b714fc349c44c5d449e667429bc0f15fb4af9c4389e82ca9eecc2",
}
OFFICIAL_PRIMO_PREDICTOR_SHA256 = (
    "a17d51f2d288472f5a4ddc7aefd81c4d4737f8ae2dd67ceb8bc064092a98133f"
)


def _stage_contract() -> dict[str, object]:
    return {
        "stage1": {
            "train_partition": "optimization-train_only",
            "official_test_loaded": False,
            "save_train_extract": False,
            "save_soft_probs": False,
        },
        "refit": {
            "initialization": "scratch",
            "train_partition": "full_designated_train",
            "epochs": "E_star_plus_1",
            "checkpoint_count": 1,
            "save_train_extract": True,
            "save_soft_probs": True,
        },
        "terminal_evaluation": {
            "official_test_extraction_passes": 1,
            "raw_and_post_dp_share_saved_extraction": True,
        },
    }


def _write(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return sha256(data).hexdigest()


def _json(path: Path, payload: object) -> str:
    return _write(
        path,
        (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8"),
    )


def _protocol_identity(
    *,
    method: str,
    dataset: str,
    seed: int,
    device: str = "cuda:0",
    cache_token: str = "cache-family-a",
) -> dict[str, object]:
    def artifact(token: str) -> dict[str, object]:
        return {
            "path": f"/sealed/{token}",
            "sha256": sha256(token.encode()).hexdigest(),
            "bytes": len(token),
        }

    horizon = int(SOURCE_PROFILES[method]["horizon"])
    implementation = {
        relative: {
            "path": f"/sealed/{relative}",
            "sha256": digest,
            "bytes": 1,
        }
        for relative, digest in HISTORICAL_IMPLEMENTATION_SHA256.items()
    }
    predictor = None
    if method == "bee2021":
        predictor = {
            "path": "/sealed/primo-predictor.npz",
            "sha256": OFFICIAL_PRIMO_PREDICTOR_SHA256,
            "bytes": 16878,
        }
    return {
        "schema_version": 1,
        "method": method,
        "dataset": dataset,
        "setting": "setting1",
        "seed": seed,
        "matched_length_bases": PROTOCOL.length_bases,
        "val_split_ratio": 0.1,
        "val_split_seed": 42,
        "selection_metric": "val_neural_raw_mAP_at_R",
        "selection_distance": "base_hamming",
        "source_profile": copy.deepcopy(SOURCE_PROFILES[method]),
        # This fixture stands for a HISTORICAL cell: it carries the historical
        # implementation SHAs above, so it must also carry the audit text those
        # runs recorded. H6 later replaced PRIMO's literal "18 nt" with a
        # length-agnostic phrase, and mixing the new text with the old SHAs
        # would produce an identity no real run ever had -- matching neither the
        # historical nor the current reviewed lock.
        "source_reproduction_audit": _historical_audit(
            method) if method == "bee2021" else copy.deepcopy(
            SOURCE_REPRODUCTION_AUDIT[method]
        ),
        "candidate_eval_period": 5,
        "candidate_epochs_zero_based": list(range(4, horizon, 5)),
        "map_at_R": 1000 if dataset == "CIFAR10" else 5000,
        "stage_contract": _stage_contract(),
        "common_bio_projection": dict(COMMON_BIO_PROJECTION),
        "cache": {
            "cache_dir": "/sealed/cache",
            "artifacts": {
                "visual_global.f16.npy": artifact(cache_token),
            },
        },
        "dataset_root": "/sealed/dataset",
        "split_artifacts": {"train.txt": artifact("split")},
        "implementation_artifacts": implementation,
        "primo_predictor": predictor,
        "execution": {
            "device_request": device,
            "num_workers": 4,
            "extract_batch_size": 512,
            "query_chunk": 64,
            "runtime_versions": {"python": "test", "numpy": "test", "torch": "test"},
        },
    }


#: The PRIMO adaptation text as the historical 18-base cells recorded it, before
#: H6 made it length-agnostic. Pinned here so this fixture keeps describing a
#: run that actually existed.
_HISTORICAL_PRIMO_ADAPTATION = (
    "The official 80-nt yield predictor is frozen and transferred to "
    "18 nt; PRIMO's per-epoch NUPACK relabeling/predictor refit is not "
    "performed."
)


def _historical_audit(method: str) -> dict:
    audit = copy.deepcopy(SOURCE_REPRODUCTION_AUDIT[method])
    audit["declared_adaptation"] = _HISTORICAL_PRIMO_ADAPTATION
    return audit


def _fixture(
    root: Path,
    *,
    method: str = "bee2018",
    dataset: str = "Flickr25k",
    seed: int = 42,
    post: float = 0.7,
    raw: float = 0.72,
    unique: float = 0.8,
    main_eligible: bool = True,
    blockers: list[str] | None = None,
    query_compliance: float = 1.0,
    db_compliance: float = 1.0,
    device: str = "cuda:0",
    cache_token: str = "cache-family-a",
) -> tuple[Path, dict[str, object]]:
    cell = root / f"{method}_{dataset}_{seed}"
    checkpoint = cell / "epoch_009.pth"
    checkpoint_sha = _write(checkpoint, f"checkpoint-{seed}".encode())
    supervision = (
        "ground-truth labels" if method.startswith("koike")
        else "feature-distance pairs"
    )
    evaluation = {
        "method": method,
        "dataset": dataset,
        "length": PROTOCOL.length_bases,
        "protocol": PROTOCOL.protocol_label,
        "supervision": supervision,
        "neural_raw": {
            "mAP_at_R": raw,
            "mAP_R_cutoff": 1000 if dataset == "CIFAR10" else 5000,
        },
        "projected": {
            "mAP_at_R": post,
            "mAP_R_cutoff": 1000 if dataset == "CIFAR10" else 5000,
        },
        "query": {"post_compliance": query_compliance},
        "database": {
            "post_compliance": db_compliance,
            "unique_ratio_projected": unique,
        },
    }
    if method == "bee2021":
        evaluation["method_variant"] = "frozen_predictor_length_transfer"
    evaluation_path = cell / "evaluation_native_dna.json"
    evaluation_sha = _json(evaluation_path, evaluation)
    resolved_blockers = [] if blockers is None else blockers
    identity = _protocol_identity(
        method=method,
        dataset=dataset,
        seed=seed,
        device=device,
        cache_token=cache_token,
    )
    manifest = {
        "schema_version": 1,
        "method": method,
        "dataset": dataset,
        "setting": "setting1",
        "seed": seed,
        "length": PROTOCOL.length_bases,
        "base_length": PROTOCOL.length_bases,
        "test_used_for_selection": False,
        "test_access_contract": dict(TEST_ACCESS_CONTRACT),
        "best_epoch_zero_based": 9,
        "refit_epochs": 10,
        "run_manifest_phase": "completed",
        "main_protocol_eligible": main_eligible,
        "main_eligibility_blockers": resolved_blockers,
        "evaluation_native_dna": {
            "path": "evaluation_native_dna.json",
            "sha256": evaluation_sha,
        },
        "final_checkpoint": {
            "path": "epoch_009.pth",
            "sha256": checkpoint_sha,
        },
        "protocol_identity": identity,
        "protocol_digest_sha256": _canonical_digest(identity),
    }
    manifest_path = cell / "native_p0_run_manifest.json"
    _json(manifest_path, manifest)
    return manifest_path, manifest


class NativeDNAP0ManifestValidationTest(unittest.TestCase):
    def test_valid_main_eligible_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, _ = _fixture(Path(directory))
            key = Key("bee2018", "Flickr25k", 42)
            record = _validate_manifest(path, key, protocol=PROTOCOL)
            self.assertEqual(record["status"], "complete_main_eligible")
            self.assertEqual(record["post_dp_map_at_R"], 0.7)
            self.assertEqual(record["raw_map_at_R"], 0.72)
            self.assertEqual(record["db_unique_post"], 0.8)
            self.assertEqual(record["best_epoch_zero_based"], 9)
            self.assertEqual(
                record["method_protocol_lock_sha256"],
                method_protocol_lock_for(PROTOCOL)["bee2018"],
            )
            self.assertEqual(record["validation_errors"], [])

    def test_historical_lock_does_not_rehash_the_live_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, _ = _fixture(Path(directory))
            current = sha256(
                (Path(__file__).parents[1] / "baseline/base_model.py").read_bytes()
            ).hexdigest()
            self.assertNotEqual(
                current,
                HISTORICAL_IMPLEMENTATION_SHA256["baseline/base_model.py"],
            )
            record = _validate_manifest(
                path, Key("bee2018", "Flickr25k", 42), protocol=PROTOCOL
            )
            self.assertEqual(record["status"], "complete_main_eligible")

    def test_stable_protocol_mutations_fail_after_outer_digest_is_recomputed(
        self,
    ) -> None:
        mutation_names = (
            "source_profile",
            "source_audit",
            "candidate_grid",
            "stage_contract",
            "bio_projection",
            "implementation_sha",
            "execution",
            "pipeline_variant",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in mutation_names:
                path, manifest = _fixture(root / name)
                identity = manifest["protocol_identity"]
                if name == "source_profile":
                    identity["source_profile"]["learning_rate"] = 0.123
                elif name == "source_audit":
                    identity["source_reproduction_audit"]["reference"] = (
                        "different-paper"
                    )
                elif name == "candidate_grid":
                    identity["candidate_epochs_zero_based"] = [4, 9]
                elif name == "stage_contract":
                    identity["stage_contract"]["refit"]["initialization"] = (
                        "selected_checkpoint"
                    )
                elif name == "bio_projection":
                    identity["common_bio_projection"]["max_run"] = 4
                elif name == "implementation_sha":
                    identity["implementation_artifacts"][
                        "baseline/native_dna.py"
                    ]["sha256"] = "f" * 64
                elif name == "execution":
                    identity["execution"]["query_chunk"] = 32
                elif name == "pipeline_variant":
                    identity["pipeline_variant"] = "forged-18-v1"
                manifest["protocol_digest_sha256"] = _canonical_digest(identity)
                _json(path, manifest)
                record = _validate_manifest(
                    path, Key("bee2018", "Flickr25k", 42), protocol=PROTOCOL
                )
                self.assertEqual(record["status"], "invalid", name)
                self.assertTrue(
                    any(
                        "method_protocol_lock_sha256" in error
                        for error in record["validation_errors"]
                    ),
                    name,
                )

    def test_primo_predictor_mutation_fails_after_digest_recompute(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, manifest = _fixture(
                Path(directory),
                method="bee2021",
                main_eligible=False,
                blockers=["primo_frozen_predictor_length_transfer"],
            )
            identity = manifest["protocol_identity"]
            identity["primo_predictor"]["sha256"] = "e" * 64
            manifest["protocol_digest_sha256"] = _canonical_digest(identity)
            _json(path, manifest)
            record = _validate_manifest(
                path, Key("bee2021", "Flickr25k", 42), protocol=PROTOCOL
            )
            self.assertEqual(record["status"], "invalid")
            self.assertTrue(any(
                "method_protocol_lock_sha256" in error
                for error in record["validation_errors"]
            ))

    def test_legacy_or_primo_adaptation_is_diagnostic_not_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, _ = _fixture(
                Path(directory),
                method="bee2021",
                main_eligible=False,
                blockers=["primo_frozen_predictor_length_transfer"],
            )
            record = _validate_manifest(
                path, Key("bee2021", "Flickr25k", 42), protocol=PROTOCOL
            )
            self.assertEqual(record["status"], "complete_diagnostic_only")
            self.assertFalse(record["main_protocol_eligible"])
            self.assertEqual(
                record["main_eligibility_blockers"],
                ["primo_frozen_predictor_length_transfer"],
            )

    def test_sha_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, manifest = _fixture(Path(directory))
            manifest["evaluation_native_dna"]["sha256"] = "0" * 64
            _json(path, manifest)
            record = _validate_manifest(
                path, Key("bee2018", "Flickr25k", 42), protocol=PROTOCOL
            )
            self.assertEqual(record["status"], "invalid")
            self.assertTrue(any(
                "mismatch" in error for error in record["validation_errors"]
            ))

    def test_post_compliance_below_one_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, _ = _fixture(
                Path(directory), query_compliance=0.999
            )
            record = _validate_manifest(
                path, Key("bee2018", "Flickr25k", 42), protocol=PROTOCOL
            )
            self.assertEqual(record["status"], "invalid")
            self.assertTrue(any(
                "exact value 1.0 required" in error
                for error in record["validation_errors"]
            ))

    def test_protocol_digest_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, manifest = _fixture(Path(directory))
            manifest["protocol_digest_sha256"] = "0" * 64
            _json(path, manifest)
            record = _validate_manifest(
                path, Key("bee2018", "Flickr25k", 42), protocol=PROTOCOL
            )
            self.assertEqual(record["status"], "invalid")
            self.assertTrue(any(
                "does not bind protocol_identity" in error
                for error in record["validation_errors"]
            ))

    def test_ineligible_without_blocker_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, _ = _fixture(
                Path(directory), main_eligible=False, blockers=[]
            )
            record = _validate_manifest(
                path, Key("bee2018", "Flickr25k", 42), protocol=PROTOCOL
            )
            self.assertEqual(record["status"], "invalid")
            self.assertTrue(any(
                "requires blocker" in error
                for error in record["validation_errors"]
            ))


class NativeDNAP0ResolutionAndAggregationTest(unittest.TestCase):
    def test_duplicate_is_never_score_or_timestamp_resolved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, _ = _fixture(root / "first")
            second, _ = _fixture(root / "second", post=0.99)
            key = Key("bee2018", "Flickr25k", 42)
            records = _resolve_records(
                {key: [first, second]},
                seeds=(42,),
                blocked_methods={},
                verify_hashes=True,
            )
            self.assertEqual(records[key]["status"], "duplicate")
            self.assertEqual(len(records[key]["duplicate_manifests"]), 2)

    def test_three_seed_diagnostic_and_strict_are_separate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = [0.70, 0.72, 0.74]
            for seed, score in zip(DEFAULT_SEEDS, expected):
                _fixture(
                    root,
                    method="bee2021",
                    seed=seed,
                    post=score,
                    raw=score + 0.01,
                    unique=0.5 + (seed - 42) * 0.1,
                    main_eligible=False,
                    blockers=["primo_frozen_predictor_length_transfer"],
                )
            payload, records = aggregate([root], protocol=PROTOCOL)
            diagnostic = _aggregate_cell(
                records,
                method="bee2021",
                dataset="Flickr25k",
                seeds=DEFAULT_SEEDS,
                mode="diagnostic",
            )
            strict = _aggregate_cell(
                records,
                method="bee2021",
                dataset="Flickr25k",
                seeds=DEFAULT_SEEDS,
                mode="strict_main",
            )
            self.assertEqual(diagnostic["aggregate_status"], "complete")
            self.assertAlmostEqual(
                diagnostic["mean_post_dp_map_at_R"],
                statistics.mean(expected),
            )
            self.assertAlmostEqual(
                diagnostic["sample_std_post_dp_map_at_R"],
                statistics.stdev(expected),
            )
            self.assertTrue(diagnostic["diagnostic_only"])
            self.assertEqual(
                diagnostic["best_epoch_zero_based_by_seed"],
                {"42": 9, "43": 9, "44": 9},
            )
            self.assertEqual(strict["aggregate_status"], "missing")
            self.assertIsNone(strict["mean_post_dp_map_at_R"])
            rendered = _markdown(payload)
            self.assertIn(DISPLAY["bee2021"], rendered)
            self.assertIn("0.7200 ± 0.0200†", rendered)
            self.assertIn(
                "`U0-FD` = unsupervised with respect to the target benchmark",
                rendered,
            )
            self.assertIn(
                "repository-local information-condition tags",
                rendered,
            )
            self.assertIn(
                "held-out labels are used only to score validation retrieval "
                "and select E*",
                rendered,
            )
            self.assertIn(
                "## Unsupervised (`U0-FD`; target-label-free encoder objective) "
                "feature-distance-pair direct predecessors",
                rendered,
            )
            self.assertIn(
                "## Supervised (`S`) direct-prior baselines",
                rendered,
            )

    def test_missing_seed_is_partial_and_has_no_unqualified_mean(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _fixture(root, seed=42)
            _fixture(root, seed=43)
            _, records = aggregate([root], protocol=PROTOCOL)
            item = _aggregate_cell(
                records,
                method="bee2018",
                dataset="Flickr25k",
                seeds=DEFAULT_SEEDS,
                mode="diagnostic",
            )
            self.assertEqual(item["aggregate_status"], "partial")
            self.assertEqual(item["seeds_admitted"], [42, 43])
            self.assertIsNone(item["mean_post_dp_map_at_R"])
            self.assertIsNone(item["sample_std_post_dp_map_at_R"])

    def test_mixed_cache_or_code_protocol_family_invalidates_all_present_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _fixture(root, seed=42)
            _fixture(root, seed=43)
            _fixture(root, seed=44, cache_token="different-cache")
            payload, records = aggregate([root], protocol=PROTOCOL)
            for seed in DEFAULT_SEEDS:
                record = records[Key("bee2018", "Flickr25k", seed)]
                self.assertEqual(record["status"], "invalid")
                self.assertTrue(any(
                    "mixed protocol families across seeds" in error
                    for error in record["validation_errors"]
                ))
            item = next(
                value
                for value in payload["diagnostic_aggregate"]
                if value["method"] == "bee2018"
                and value["dataset"] == "Flickr25k"
            )
            self.assertEqual(item["aggregate_status"], "missing")

    def test_seed_and_requested_device_are_the_only_allowed_family_differences(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed, device in zip(DEFAULT_SEEDS, ("cuda:0", "cuda:1", "cpu")):
                _fixture(root, seed=seed, device=device)
            _, records = aggregate([root], protocol=PROTOCOL)
            item = _aggregate_cell(
                records,
                method="bee2018",
                dataset="Flickr25k",
                seeds=DEFAULT_SEEDS,
                mode="diagnostic",
            )
            self.assertEqual(item["aggregate_status"], "complete")

    def test_explicit_primo_block_is_rendered(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload, records = aggregate(
                [Path(directory)],
                blocked_methods={"bee2021": "official predictor unavailable"},
            )
            for dataset in DATASETS:
                for seed in DEFAULT_SEEDS:
                    self.assertEqual(
                        records[Key("bee2021", dataset, seed)]["status"],
                        "blocked",
                    )
            markdown = _markdown(payload)
            self.assertIn("BLOCKED", markdown)
            self.assertIn("official predictor unavailable", markdown)


if __name__ == "__main__":
    unittest.main()
