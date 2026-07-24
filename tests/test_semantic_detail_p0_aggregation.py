import hashlib
import json
import shlex
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.aggregate_semantic_detail_p0 import (
    AB_RUNNER_SOURCE_FILES,
    ARM_ORDER,
    COMMON_SOURCE_FILES,
    DATASETS,
    DATASET_ORDER,
    EXTENDED_ARM_ORDER,
    FACTUAL_ARRAY_NAMES,
    FOIL_ARTIFACT_NAMES,
    LEGACY_RUNNER_SOURCE_FILES,
    REPO,
    _delta,
    _cross_validate,
    _expected_extra_args,
    _expected_source_files,
    _protocol_core,
    _validate_base_manifest,
    _validate_one,
    _validate_saved_args,
    main,
)


def _write_archived_incumbents(repo_root: Path) -> None:
    for spec in DATASETS.values():
        path = repo_root / spec["incumbent_result"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "dataset": spec["canonical"],
                    "K": 128,
                    "mAP_at_R_bioproj": spec["incumbent_map"],
                    "DNA_unique_DB": spec["incumbent_unique"],
                    "map_r_cutoff": spec["map_r"],
                }
            ),
            encoding="utf-8",
        )


def _write_stub_manifests(
    repo_root: Path, arms: tuple[str, ...]
) -> Path:
    manifest_dir = repo_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for dataset in DATASET_ORDER:
        for arm in arms:
            (manifest_dir / f"{dataset}_{arm}_K128L4_p0.json").write_text(
                "{}\n", encoding="utf-8"
            )
    return manifest_dir


def _fake_metrics(dataset: str, arm: str) -> dict[str, float | int]:
    dataset_offset = DATASET_ORDER.index(dataset) * 0.01
    map_value = {"A": 0.70, "AB": 0.72, "ABC": 0.75}[arm]
    unique_value = {"A": 0.20, "AB": 0.25, "ABC": 0.23}[arm]
    return {
        "mAP_at_R_bioproj": map_value + dataset_offset,
        "full_mAP_bioproj": map_value - 0.1 + dataset_offset,
        "full_mAP_pre_projection": map_value - 0.09 + dataset_offset,
        "DNA_unique_DB": unique_value + dataset_offset,
        "map_r_cutoff": DATASETS[dataset]["map_r"],
    }


def _fake_complete_record(
    manifest_path: Path,
    *,
    dataset: str,
    arm: str,
    repo_root: Path,
) -> dict:
    del repo_root
    return {
        "dataset": dataset,
        "arm": arm,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": "0" * 64,
        "manifest_status": "complete",
        "validation_status": "complete_valid",
        "errors": [],
        "stage1": {
            "selected_epoch": 4,
            "result_dir": f"/stage1/{dataset}/{arm}",
            "log": f"/logs/{dataset}_{arm}_stage1.log",
        },
        "stage2": {
            "result_dir": f"/stage2/{dataset}/{arm}",
            "log": f"/logs/{dataset}_{arm}_stage2.log",
        },
        "metrics": _fake_metrics(dataset, arm),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _base_manifest_fixture(
    dataset: str,
    arm: str,
    *,
    repo_root: Path = REPO,
) -> dict:
    digest = "0" * 64
    source_sha = {
        path: _sha256(repo_root / path)
        for path in _expected_source_files(
            dataset, arm, repo_root=repo_root
        )
    }
    cache = {
        "base": "/cache/base",
        "foil_overlay": "/cache/foil",
        "evaluation": "/cache/eval",
        "qwen_jsonl": "/cache/qwen.jsonl",
        "training": "/cache/base" if arm == "A" else "/cache/foil",
        "preparation_manifest": {
            "path": "/cache/preparation_manifest.json",
            "sha256": digest,
            "payload": {"schema_version": 1, "fixture": True},
        },
    }
    whitening_entry = {
        "path": "/cache/whiten.npz",
        "meta_path": "/cache/whiten.npz.meta.json",
        "sha256": digest,
        "meta_sha256": digest,
        "N_kept": 10,
        "vectors_used": 50,
    }
    return {
        "schema_version": 1,
        "protocol": {
            **_protocol_core(),
            "dataset": dataset,
            "canonical_dataset": DATASETS[dataset]["canonical"],
            "arm": arm,
        },
        "launcher": DATASETS[dataset]["launcher"],
        "stage1_tag": f"{dataset}_semantic_detail_{arm}_K128L4_P0val_s42",
        "extra_args": shlex.join(_expected_extra_args(arm)),
        "source_sha256": source_sha,
        "cache": cache,
        "factual_arrays": {
            name: {
                "path": f"/cache/base/{name}",
                "shape": [10],
                "dtype": "float16",
                "bytes": 20,
            }
            for name in FACTUAL_ARRAY_NAMES
        },
        "foil_artifacts": (
            {}
            if arm == "A"
            else {
                name: {
                    "path": f"/cache/foil/{name}",
                    "shape": [10],
                    "dtype": "float16",
                    "bytes": 20,
                }
                for name in FOIL_ARTIFACT_NAMES
            }
        ),
        "whitening": {
            "stage1_optTrain_localOnly": dict(whitening_entry),
            "stage2_trainOnly_localOnly": dict(whitening_entry),
        },
    }


def _write_source_fixture(
    repo_root: Path,
    *,
    dataset: str,
    arm: str,
) -> None:
    for relative in _expected_source_files(
        dataset, arm, repo_root=repo_root
    ):
        path = repo_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# fixture: {relative}\n", encoding="utf-8")
    for relative in ("dna_utils/fixture.py", "models/fixture.py"):
        path = repo_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# fixture: {relative}\n", encoding="utf-8")


def _write_complete_manifest_with_malformed_args(
    repo_root: Path,
    *,
    dataset: str = "flickr",
    arm: str = "AB",
) -> Path:
    _write_source_fixture(repo_root, dataset=dataset, arm=arm)
    manifest = _base_manifest_fixture(
        dataset, arm, repo_root=repo_root
    )
    selected_epoch = 4
    stage1_tag = manifest["stage1_tag"]
    stage2_tag = (
        f"{dataset}_semantic_detail_{arm}_K128L4_"
        f"P0refit_e{selected_epoch}_s42"
    )
    stage1_result = repo_root / "result" / "stage1"
    stage2_result = repo_root / "result" / "stage2"
    stage1_result.mkdir(parents=True)
    stage2_result.mkdir(parents=True)

    malformed = "this is not a saved-args key/value record\n"
    for result_dir, required in (
        (
            stage1_result,
            (
                "args.txt",
                "log.csv",
                "model_state_dict.pth",
                "model_state_dict_best.pth",
            ),
        ),
        (
            stage2_result,
            (
                "args.txt",
                "log.csv",
                "model_state_dict.pth",
                "extract_db.npz",
                "extract_query.npz",
                "evaluation_siglip2_base.json",
            ),
        ),
    ):
        for filename in required:
            value = malformed if filename == "args.txt" else "fixture\n"
            (result_dir / filename).write_text(value, encoding="utf-8")

    logs = repo_root / "logs"
    logs.mkdir()
    (logs / f"{stage1_tag}.log").write_text(
        "\n".join(
            (
                "Final checkpoint saved",
                "[p0-stage1] SKIP official-test extraction/evaluation",
                f"new best mid-eval mAP=0.5 at epoch {selected_epoch}",
                "",
            )
        ),
        encoding="utf-8",
    )
    (logs / f"{stage2_tag}.log").write_text(
        "\n".join(
            (
                "Final checkpoint saved",
                "[p0-stage2] OFFICIAL TEST ISOLATED",
                "[final-eval] PRESERVE training text_whiten across cache override",
                "[final-eval] running extraction",
                "[final-eval] running evaluation",
                "[CELL-RESULT]",
                f"[refit-stop] stopping after epoch {selected_epoch}",
                "",
            )
        ),
        encoding="utf-8",
    )

    cell_result = {
        "dataset": DATASETS[dataset]["canonical"],
        "K": 128,
        "map_r_cutoff": DATASETS[dataset]["map_r"],
        "gc_min_frac": 0.416,
        "gc_max_frac": 0.584,
        "mAP_at_R_bioproj": 0.8,
        "full_mAP_bioproj": 0.7,
        "full_mAP_pre_projection": 0.71,
        "DNA_unique_DB": 0.3,
    }
    (stage2_result / "cell_result.json").write_text(
        json.dumps(cell_result), encoding="utf-8"
    )
    (stage2_result / "evaluation_siglip2_base_bioproj.json").write_text(
        json.dumps(
            {
                "mAP_at_R": 0.8,
                "mAP": 0.7,
                "unique_code_ratio": 0.3,
                "mAP_R_cutoff": DATASETS[dataset]["map_r"],
                "bio_stats": {"mAP_pre_projection": 0.71},
            }
        ),
        encoding="utf-8",
    )
    manifest.update(
        {
            "status": "complete",
            "selected_epoch": selected_epoch,
            "stage1_result_dir": str(stage1_result),
            "stage2_tag": stage2_tag,
            "stage2_result_dir": str(stage2_result),
            "cell_result": cell_result,
        }
    )
    manifest_path = repo_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


class SemanticDetailP0AggregationTest(unittest.TestCase):
    def test_delta_reports_retrieval_and_unique_changes(self) -> None:
        actual = _delta(
            {"mAP_at_R_bioproj": 0.8, "DNA_unique_DB": 0.3},
            {"mAP_at_R_bioproj": 0.75, "DNA_unique_DB": 0.2},
        )
        self.assertIsNotNone(actual)
        self.assertAlmostEqual(actual["mAP_at_R_bioproj"], 0.05)
        self.assertAlmostEqual(actual["DNA_unique_DB"], 0.1)

    def test_default_refuses_incomplete_matrix_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_archived_incumbents(root)
            output_json = root / "summary.json"
            output_markdown = root / "summary.md"
            status = main(
                [
                    "--repo-root",
                    str(root),
                    "--manifest-dir",
                    "manifests",
                    "--out-json",
                    str(output_json),
                    "--out-markdown",
                    str(output_markdown),
                ]
            )
            self.assertEqual(status, 2)
            self.assertFalse(output_json.exists())
            self.assertFalse(output_markdown.exists())

    def test_allow_incomplete_writes_explicit_progress_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_archived_incumbents(root)
            output_json = root / "summary.json"
            output_markdown = root / "summary.md"
            status = main(
                [
                    "--repo-root",
                    str(root),
                    "--manifest-dir",
                    "manifests",
                    "--out-json",
                    str(output_json),
                    "--out-markdown",
                    str(output_markdown),
                    "--allow-incomplete",
                ]
            )
            self.assertEqual(status, 0)
            payload = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertFalse(payload["summary"]["table_ready"])
            self.assertEqual(payload["summary"]["complete_valid"], 0)
            self.assertEqual(payload["summary"]["missing"], 8)
            markdown = output_markdown.read_text(encoding="utf-8")
            self.assertIn("Table ready: `false`", markdown)
            self.assertIn("## Pending cells", markdown)

    def test_ab_contract_uses_foil_without_cibhash_bit_kl(self) -> None:
        ab_args = _expected_extra_args("AB")
        self.assertIn("--text_hash_counterfactual_weight", ab_args)
        self.assertNotIn("--cibhash_visual_token_bit_kl", ab_args)
        self.assertEqual(
            _expected_extra_args("ABC"),
            [*ab_args, "--cibhash_visual_token_bit_kl"],
        )

        errors: list[str] = []
        manifest = _base_manifest_fixture("flickr", "AB")
        _validate_base_manifest(
            manifest, dataset="flickr", arm="AB", errors=errors
        )
        self.assertEqual(errors, [])

    def test_malformed_saved_args_fails_direct_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "args.txt"
            path.write_text("not a parseable saved-args record\n", encoding="utf-8")
            errors: list[str] = []
            _validate_saved_args(
                path=path,
                label="stage1",
                dataset="flickr",
                arm="AB",
                stage=1,
                selected_epoch=4,
                manifest={},
                repo_root=Path(temporary),
                errors=errors,
            )
        self.assertTrue(
            any("no parseable key/value records" in error for error in errors)
        )

    def test_complete_manifest_with_malformed_saved_args_is_invalid(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo_root = Path(temporary)
            manifest_path = _write_complete_manifest_with_malformed_args(
                repo_root
            )
            record = _validate_one(
                manifest_path,
                dataset="flickr",
                arm="AB",
                repo_root=repo_root,
            )
        self.assertEqual(record["manifest_status"], "complete")
        self.assertEqual(record["validation_status"], "invalid")
        self.assertEqual(
            sum(
                "no parseable key/value records" in error
                for error in record["errors"]
            ),
            2,
        )

    def test_ab_and_legacy_runner_source_contracts_are_separate(self) -> None:
        launcher = DATASETS["flickr"]["launcher"]
        dynamic_sources = {
            path.relative_to(REPO).as_posix()
            for parent in ("dna_utils", "models")
            for path in (REPO / parent).glob("*.py")
            if path.is_file()
        }
        legacy_expected = (
            set(COMMON_SOURCE_FILES) | dynamic_sources | {launcher}
        )
        self.assertEqual(
            _expected_source_files("flickr", "A"), legacy_expected
        )
        self.assertEqual(
            _expected_source_files("flickr", "ABC"), legacy_expected
        )

        ab_expected = _expected_source_files("flickr", "AB")
        self.assertTrue(set(AB_RUNNER_SOURCE_FILES) <= ab_expected)
        self.assertTrue(set(LEGACY_RUNNER_SOURCE_FILES).isdisjoint(ab_expected))

        manifest = _base_manifest_fixture("flickr", "AB")
        del manifest["source_sha256"][AB_RUNNER_SOURCE_FILES[0]]
        errors: list[str] = []
        _validate_base_manifest(
            manifest, dataset="flickr", arm="AB", errors=errors
        )
        self.assertTrue(
            any("required path(s) missing" in error for error in errors)
        )

    def test_source_allowlist_rejects_missing_and_fake_dynamic_paths(
        self,
    ) -> None:
        manifest = _base_manifest_fixture("flickr", "AB")
        real_dynamic = sorted(
            path
            for path in manifest["source_sha256"]
            if path.startswith(("dna_utils/", "models/"))
        )
        self.assertGreater(len(real_dynamic), 2)
        del manifest["source_sha256"][real_dynamic[0]]
        manifest["source_sha256"]["dna_utils/not_real.py"] = "0" * 64
        errors: list[str] = []
        _validate_base_manifest(
            manifest, dataset="flickr", arm="AB", errors=errors
        )
        self.assertTrue(
            any("required path(s) missing" in error for error in errors)
        )
        self.assertTrue(
            any("unexpected path(s)" in error for error in errors)
        )

    def test_source_digest_must_match_current_repo_file(self) -> None:
        manifest = _base_manifest_fixture("flickr", "AB")
        manifest["source_sha256"]["config.py"] = "1" * 64
        errors: list[str] = []
        _validate_base_manifest(
            manifest, dataset="flickr", arm="AB", errors=errors
        )
        self.assertTrue(
            any(
                "current SHA-256" in error and "config.py" in error
                for error in errors
            )
        )

    def test_cross_validation_accepts_only_the_runner_provenance_split(
        self,
    ) -> None:
        records = {}
        for dataset in DATASET_ORDER:
            for arm in EXTENDED_ARM_ORDER:
                records[f"{dataset}:{arm}"] = {
                    "dataset": dataset,
                    "arm": arm,
                    "errors": [],
                    "validation_status": "complete_valid",
                    "_manifest_payload": _base_manifest_fixture(dataset, arm),
                }

        _cross_validate(records, arms=EXTENDED_ARM_ORDER)
        self.assertTrue(
            all(not record["errors"] for record in records.values())
        )

        records["flickr:AB"]["_manifest_payload"]["source_sha256"][
            "config.py"
        ] = "1" * 64
        _cross_validate(records, arms=EXTENDED_ARM_ORDER)
        self.assertEqual(
            records["flickr:AB"]["validation_status"], "invalid"
        )
        self.assertTrue(
            any(
                "source" in error
                for error in records["flickr:AB"]["errors"]
            )
        )

    def test_cross_validation_rejects_factual_array_drift(self) -> None:
        records = {}
        for dataset in DATASET_ORDER:
            for arm in EXTENDED_ARM_ORDER:
                records[f"{dataset}:{arm}"] = {
                    "dataset": dataset,
                    "arm": arm,
                    "errors": [],
                    "validation_status": "complete_valid",
                    "_manifest_payload": _base_manifest_fixture(dataset, arm),
                }
        records["flickr:AB"]["_manifest_payload"]["factual_arrays"][
            "text_part.f16.npy"
        ]["path"] = "/cache/different/text_part.f16.npy"
        _cross_validate(records, arms=EXTENDED_ARM_ORDER)
        self.assertEqual(
            records["flickr:AB"]["validation_status"], "invalid"
        )
        self.assertTrue(
            any(
                "protocol identity mismatch" in error
                for error in records["flickr:AB"]["errors"]
            )
        )

    def test_cross_validation_rejects_preparation_manifest_drift(
        self,
    ) -> None:
        records = {}
        for dataset in DATASET_ORDER:
            for arm in EXTENDED_ARM_ORDER:
                records[f"{dataset}:{arm}"] = {
                    "dataset": dataset,
                    "arm": arm,
                    "errors": [],
                    "validation_status": "complete_valid",
                    "_manifest_payload": _base_manifest_fixture(dataset, arm),
                }
        records["flickr:AB"]["_manifest_payload"]["cache"][
            "preparation_manifest"
        ]["payload"]["fixture"] = False
        _cross_validate(records, arms=EXTENDED_ARM_ORDER)
        self.assertEqual(
            records["flickr:AB"]["validation_status"], "invalid"
        )
        self.assertTrue(
            any(
                "protocol identity mismatch" in error
                for error in records["flickr:AB"]["errors"]
            )
        )

    def test_cross_validation_rejects_ab_abc_foil_artifact_drift(
        self,
    ) -> None:
        records = {}
        for dataset in DATASET_ORDER:
            for arm in EXTENDED_ARM_ORDER:
                records[f"{dataset}:{arm}"] = {
                    "dataset": dataset,
                    "arm": arm,
                    "errors": [],
                    "validation_status": "complete_valid",
                    "_manifest_payload": _base_manifest_fixture(dataset, arm),
                }
        records["flickr:AB"]["_manifest_payload"]["foil_artifacts"][
            "text_foil_part.f16.npy"
        ]["path"] = "/cache/different/text_foil_part.f16.npy"
        _cross_validate(records, arms=EXTENDED_ARM_ORDER)
        self.assertEqual(
            records["flickr:AB"]["validation_status"], "invalid"
        )
        self.assertEqual(
            records["flickr:ABC"]["validation_status"], "invalid"
        )
        self.assertTrue(
            any(
                "foil-artifact identity mismatch" in error
                for error in records["flickr:AB"]["errors"]
            )
        )
        self.assertEqual(
            records["flickr:A"]["validation_status"], "complete_valid"
        )

    def test_extended_mode_refuses_when_only_legacy_cells_exist(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_archived_incumbents(root)
            _write_stub_manifests(root, ARM_ORDER)
            output_json = root / "summary.json"
            output_markdown = root / "summary.md"
            arguments = [
                "--repo-root",
                str(root),
                "--manifest-dir",
                "manifests",
                "--out-json",
                str(output_json),
                "--out-markdown",
                str(output_markdown),
                "--include-ab",
            ]
            with patch(
                "scripts.aggregate_semantic_detail_p0._validate_one",
                side_effect=_fake_complete_record,
            ):
                status = main(arguments)
                self.assertEqual(status, 2)
                self.assertFalse(output_json.exists())
                self.assertFalse(output_markdown.exists())

                status = main([*arguments, "--allow-incomplete"])
            self.assertEqual(status, 0)
            payload = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["matrix"]["arms"], list(EXTENDED_ARM_ORDER))
            self.assertEqual(payload["summary"]["expected"], 12)
            self.assertEqual(payload["summary"]["complete_valid"], 8)
            self.assertEqual(payload["summary"]["missing"], 4)
            self.assertFalse(payload["summary"]["table_ready"])
            comparison = payload["comparisons"]["flickr"]
            self.assertIsNone(comparison["AB"])
            self.assertIsNone(comparison["AB_minus_A"])
            self.assertIsNone(comparison["ABC_minus_AB"])
            self.assertIsNotNone(comparison["ABC_minus_A"])
            self.assertIsNone(
                comparison["AB_minus_archived_incumbent"]
            )
            markdown = output_markdown.read_text(encoding="utf-8")
            self.assertIn("`flickr:AB`: `missing`", markdown)
            self.assertIn("## Pairwise arm deltas", markdown)

    def test_extended_complete_matrix_writes_all_deltas(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_archived_incumbents(root)
            _write_stub_manifests(root, EXTENDED_ARM_ORDER)
            output_json = root / "summary.json"
            output_markdown = root / "summary.md"
            with patch(
                "scripts.aggregate_semantic_detail_p0._validate_one",
                side_effect=_fake_complete_record,
            ):
                status = main(
                    [
                        "--repo-root",
                        str(root),
                        "--manifest-dir",
                        "manifests",
                        "--out-json",
                        str(output_json),
                        "--out-markdown",
                        str(output_markdown),
                        "--include-ab",
                    ]
                )
            self.assertEqual(status, 0)
            payload = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertTrue(payload["summary"]["table_ready"])
            self.assertEqual(payload["summary"]["complete_valid"], 12)
            comparison = payload["comparisons"]["flickr"]
            self.assertAlmostEqual(
                comparison["AB_minus_A"]["mAP_at_R_bioproj"], 0.02
            )
            self.assertAlmostEqual(
                comparison["AB_minus_A"]["DNA_unique_DB"], 0.05
            )
            self.assertAlmostEqual(
                comparison["ABC_minus_AB"]["mAP_at_R_bioproj"], 0.03
            )
            self.assertAlmostEqual(
                comparison["ABC_minus_AB"]["DNA_unique_DB"], -0.02
            )
            self.assertAlmostEqual(
                comparison["ABC_minus_A"]["mAP_at_R_bioproj"], 0.05
            )
            for arm in EXTENDED_ARM_ORDER:
                self.assertIn(
                    f"{arm}_minus_archived_incumbent", comparison
                )
            markdown = output_markdown.read_text(encoding="utf-8")
            self.assertIn("AB: bio mAP@R / DNA-unique", markdown)
            self.assertIn("ABC−AB mAP@R", markdown)
            self.assertIn(
                "Requested arms versus archived K=128", markdown
            )

    def test_extended_invalid_ab_is_not_admitted_with_allow_incomplete(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_archived_incumbents(root)
            _write_stub_manifests(root, EXTENDED_ARM_ORDER)
            output_json = root / "summary.json"
            output_markdown = root / "summary.md"

            def validate_with_one_invalid_ab(*args, **kwargs):
                record = _fake_complete_record(*args, **kwargs)
                if record["dataset"] == "flickr" and record["arm"] == "AB":
                    record["validation_status"] = "invalid"
                    record["errors"] = ["fixture protocol violation"]
                return record

            with patch(
                "scripts.aggregate_semantic_detail_p0._validate_one",
                side_effect=validate_with_one_invalid_ab,
            ):
                status = main(
                    [
                        "--repo-root",
                        str(root),
                        "--manifest-dir",
                        "manifests",
                        "--out-json",
                        str(output_json),
                        "--out-markdown",
                        str(output_markdown),
                        "--include-ab",
                        "--allow-incomplete",
                    ]
                )
            self.assertEqual(status, 1)
            payload = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["summary"]["invalid"], 1)
            self.assertFalse(payload["summary"]["table_ready"])
            comparison = payload["comparisons"]["flickr"]
            self.assertIsNone(comparison["AB"])
            self.assertIsNone(comparison["AB_minus_A"])
            self.assertIsNone(comparison["ABC_minus_AB"])
            self.assertIsNotNone(comparison["ABC_minus_A"])
            markdown = output_markdown.read_text(encoding="utf-8")
            self.assertIn("## Invalid cells", markdown)
            self.assertIn("fixture protocol violation", markdown)

    def test_default_mode_remains_eight_cell_a_abc_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_archived_incumbents(root)
            _write_stub_manifests(root, EXTENDED_ARM_ORDER)
            output_json = root / "summary.json"
            output_markdown = root / "summary.md"
            with patch(
                "scripts.aggregate_semantic_detail_p0._validate_one",
                side_effect=_fake_complete_record,
            ) as validator:
                status = main(
                    [
                        "--repo-root",
                        str(root),
                        "--manifest-dir",
                        "manifests",
                        "--out-json",
                        str(output_json),
                        "--out-markdown",
                        str(output_markdown),
                    ]
                )
            self.assertEqual(status, 0)
            self.assertEqual(validator.call_count, 8)
            payload = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["matrix"]["arms"], list(ARM_ORDER))
            self.assertEqual(payload["summary"]["expected"], 8)
            self.assertEqual(set(payload["cells"]), {
                f"{dataset}:{arm}"
                for dataset in DATASET_ORDER
                for arm in ARM_ORDER
            })
            self.assertNotIn("AB", payload["comparisons"]["flickr"])
            markdown = output_markdown.read_text(encoding="utf-8")
            self.assertNotIn("## Pairwise arm deltas", markdown)


if __name__ == "__main__":
    unittest.main()
