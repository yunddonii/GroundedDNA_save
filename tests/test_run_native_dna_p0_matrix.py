import copy
import contextlib
from hashlib import sha256
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest import mock
import uuid

from dna_utils.native_protocol import resolve_native_protocol
from scripts.run_native_dna_p0 import (
    SOURCE_PROFILES,
    SOURCE_REPRODUCTION_AUDIT,
    build_parser as build_cell_parser,
)
from scripts.run_native_dna_p0_matrix import (
    DEFAULT_DATASETS,
    DEFAULT_METHODS,
    DEFAULT_SEEDS,
    RUNNER,
    Job,
    LaunchOptions,
    MatrixStateError,
    RunResult,
    _canonical_protocol_digest,
    _command,
    _completed_cells,
    _jobs,
    _run_pool,
    _validate_data_root,
    _validate_inputs,
    main,
)
from scripts.aggregate_native_dna_p0 import (
    COMMON_BIO_PROJECTION,
    TEST_ACCESS_CONTRACT,
)


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


def _write_json(path: Path, payload: object) -> str:
    return _write(
        path,
        (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8"),
    )


def _completed_fixture(
    root: Path,
    job: Job,
    *,
    main_eligible: bool = True,
) -> Path:
    def artifact(token: str) -> dict[str, object]:
        return {
            "path": f"/sealed/{token}",
            "sha256": sha256(token.encode()).hexdigest(),
            "bytes": len(token),
        }

    horizon = int(SOURCE_PROFILES[job.method]["horizon"])
    implementation = {
        relative: {
            "path": f"/sealed/{relative}",
            "sha256": digest,
            "bytes": 1,
        }
        for relative, digest in HISTORICAL_IMPLEMENTATION_SHA256.items()
    }
    predictor = None
    if job.method == "bee2021":
        predictor = {
            "path": "/sealed/primo-predictor.npz",
            "sha256": OFFICIAL_PRIMO_PREDICTOR_SHA256,
            "bytes": 16878,
        }
    identity = {
        "schema_version": 1,
        "method": job.method,
        "dataset": job.dataset,
        "setting": "setting1",
        "seed": job.seed,
        "matched_length_bases": PROTOCOL.length_bases,
        "val_split_ratio": 0.1,
        "val_split_seed": 42,
        "selection_metric": "val_neural_raw_mAP_at_R",
        "selection_distance": "base_hamming",
        "source_profile": copy.deepcopy(SOURCE_PROFILES[job.method]),
        "source_reproduction_audit": copy.deepcopy(
            SOURCE_REPRODUCTION_AUDIT[job.method]
        ),
        "candidate_eval_period": 5,
        "candidate_epochs_zero_based": list(range(4, horizon, 5)),
        "map_at_R": 1000 if job.dataset == "CIFAR10" else 5000,
        "stage_contract": _stage_contract(),
        "common_bio_projection": dict(COMMON_BIO_PROJECTION),
        "cache": {
            "cache_dir": "/sealed/cache",
            "artifacts": {
                "visual_global.f16.npy": artifact("cache"),
            },
        },
        "dataset_root": "/sealed/dataset",
        "split_artifacts": {"train.txt": artifact("split")},
        "implementation_artifacts": implementation,
        "primo_predictor": predictor,
        "execution": {
            "device_request": "cuda:0",
            "num_workers": 4,
            "extract_batch_size": 512,
            "query_chunk": 64,
            "runtime_versions": {"python": "test", "numpy": "test", "torch": "test"},
        },
    }
    digest = _canonical_protocol_digest(identity)
    cell = root / f"{job.cell_prefix}{digest[:12]}"
    stage1 = cell / "stage1"
    refit = cell / "refit"
    stage1.mkdir(parents=True)
    refit.mkdir()

    checkpoint = refit / "epoch_009.pth"
    checkpoint_sha = _write(checkpoint, b"sealed-checkpoint")
    evaluation = {
        "method": job.method,
        "dataset": job.dataset,
        "length": PROTOCOL.length_bases,
        "protocol": PROTOCOL.protocol_label,
        "supervision": (
            "ground-truth labels"
            if job.method.startswith("koike")
            else "feature-distance pairs"
        ),
        "neural_raw": {
            "mAP_at_R": 0.71,
            "mAP_R_cutoff": 1000 if job.dataset == "CIFAR10" else 5000,
        },
        "projected": {
            "mAP_at_R": 0.69,
            "mAP_R_cutoff": 1000 if job.dataset == "CIFAR10" else 5000,
        },
        "query": {"post_compliance": 1.0},
        "database": {
            "post_compliance": 1.0,
            "unique_ratio_projected": 0.82,
        },
    }
    if job.method == "bee2021":
        evaluation["method_variant"] = "frozen_predictor_length_transfer"
    evaluation_path = refit / "evaluation_native_dna.json"
    evaluation_sha = _write_json(evaluation_path, evaluation)
    blockers = [] if main_eligible else ["diagnostic_fixture"]
    manifest = {
        "schema_version": 1,
        "method": job.method,
        "dataset": job.dataset,
        "setting": "setting1",
        "seed": job.seed,
        "length": PROTOCOL.length_bases,
        "base_length": PROTOCOL.length_bases,
        "best_epoch_zero_based": 9,
        "refit_epochs": 10,
        "run_manifest_phase": "completed",
        "main_protocol_eligible": main_eligible,
        "main_eligibility_blockers": blockers,
        "test_used_for_selection": False,
        "evaluation_native_dna": {
            "path": str(evaluation_path.resolve()),
            "sha256": evaluation_sha,
        },
        "final_checkpoint": {
            "path": str(checkpoint.resolve()),
            "sha256": checkpoint_sha,
        },
        "protocol_identity": identity,
        "protocol_digest_sha256": digest,
        "cell_root": str(cell.resolve()),
        "stage1_dir": str(stage1.resolve()),
        "refit_dir": str(refit.resolve()),
        "post_compliance": {"query": 1.0, "database": 1.0},
        "test_access_contract": dict(TEST_ACCESS_CONTRACT),
    }
    manifest_path = cell / "native_p0_run_manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path


def _expected_protocols(
    *manifests: Path,
) -> dict[tuple[str, str, int], tuple[dict[str, object], str]]:
    expected = {}
    for manifest in manifests:
        payload = json.loads(manifest.read_text())
        key = (
            str(payload["method"]),
            str(payload["dataset"]),
            int(payload["seed"]),
        )
        expected[key] = (
            payload["protocol_identity"],
            payload["protocol_digest_sha256"],
        )
    return expected


def _options(
    root: Path,
    *,
    predictor: Path | None = None,
    allow_diagnostic: bool = False,
) -> LaunchOptions:
    return LaunchOptions(
        python=Path(sys.executable).resolve(),
        driver=RUNNER.resolve(),
        dataset_root=root,
        result_root=root / "runs",
        log_root=root / "logs",
        predictor=predictor,
        num_workers=4,
        extract_batch_size=512,
        query_chunk=64,
        allow_diagnostic=allow_diagnostic,
    )


class NativeDNAMatrixStaticContractTest(unittest.TestCase):
    def test_default_matrix_has_48_unique_cells(self) -> None:
        jobs = _jobs(DEFAULT_METHODS, DEFAULT_DATASETS, DEFAULT_SEEDS)
        self.assertEqual(len(jobs), 4 * 4 * 3)
        self.assertEqual(len({job.key for job in jobs}), len(jobs))
        self.assertEqual(len({job.job_id for job in jobs}), len(jobs))
        self.assertEqual(jobs[0].method, "koike2026")
        self.assertEqual(jobs[0].dataset, "MSCOCO")

    def test_data_root_must_be_explicit_and_below_data(self) -> None:
        expected = Path("/data/yschoi/groundeddna_native_p0")
        self.assertEqual(_validate_data_root(expected), expected)
        for invalid in (
            "relative/path",
            "/tmp/native_p0",
            "/data",
            "/data/../tmp/native_p0",
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    _validate_data_root(invalid)

    def test_commands_parse_with_the_real_single_cell_driver(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictor = root / "primo.npz"
            predictor.write_bytes(b"frozen")
            options = _options(
                root, predictor=predictor, allow_diagnostic=True
            )
            bee = Job("bee2021", "MSCOCO", 43)
            command = _command(bee, options)
            parsed = build_cell_parser().parse_args(command[2:])
            self.assertEqual(parsed.method, "bee2021")
            self.assertEqual(parsed.dataset, "MSCOCO")
            self.assertEqual(parsed.seed, 43)
            self.assertEqual(parsed.device, "cuda:0")
            self.assertEqual(
                Path(parsed.result_root), options.result_root
            )
            self.assertEqual(
                Path(parsed.primo_predictor_npz), predictor
            )
            self.assertTrue(parsed.allow_main_ineligible_diagnostic)
            self.assertNotIn("--model-root", command)

            dna24 = _command(Job("bee2018", "CIFAR10", 42), options)
            self.assertNotIn("--primo-predictor-npz", dna24)
            parsed_dna24 = build_cell_parser().parse_args(dna24[2:])
            self.assertIsNone(parsed_dna24.primo_predictor_npz)

    def test_bee2021_fails_preflight_without_predictor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            options = _options(root)
            with self.assertRaisesRegex(
                FileNotFoundError, "bee2021 requires"
            ):
                _validate_inputs(
                    options, [Job("bee2021", "Flickr25k", 42)]
                )


# F18: these fixtures are 18-base cells, which is what the historical native
# matrix ran. Saying so explicitly is what lets them be validated in a process
# configured for the paper's 15 bases.
PROTOCOL = resolve_native_protocol(18)


class NativeDNAMatrixResumeTest(unittest.TestCase):
    def test_valid_completed_manifest_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = Job("koike2024", "NUSWIDE", 44)
            manifest = _completed_fixture(root, job)
            completed = _completed_cells(
                root,
                [job],
                allow_diagnostic=False,
                expected_protocols=_expected_protocols(manifest),
                protocol=PROTOCOL,
            )
            self.assertEqual(completed, {job.key: manifest.resolve()})

    def test_incomplete_existing_cell_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = Job("bee2018", "Flickr25k", 42)
            (root / f"{job.cell_prefix}{'a' * 12}").mkdir()
            with self.assertRaisesRegex(
                MatrixStateError, "incomplete existing cell"
            ):
                _completed_cells(root, [job], allow_diagnostic=False, protocol=PROTOCOL)

    def test_diagnostic_completion_requires_explicit_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = Job("bee2018", "CIFAR10", 42)
            manifest = _completed_fixture(
                root, job, main_eligible=False
            )
            with self.assertRaisesRegex(
                MatrixStateError,
                "--allow-main-ineligible-diagnostic",
            ):
                _completed_cells(
                    root,
                    [job],
                    allow_diagnostic=False,
                    expected_protocols=_expected_protocols(manifest),
                    protocol=PROTOCOL,
                )
            completed = _completed_cells(
                root,
                [job],
                allow_diagnostic=True,
                expected_protocols=_expected_protocols(manifest),
                protocol=PROTOCOL,
            )
            self.assertEqual(completed[job.key], manifest.resolve())

    def test_stale_or_different_protocol_is_never_resumed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = Job("koike2024", "CIFAR10", 42)
            manifest = _completed_fixture(root, job)
            expected = _expected_protocols(manifest)
            identity, _ = expected[job.key]
            changed = json.loads(json.dumps(identity))
            changed["cache"]["cache_dir"] = "/sealed/new-cache"
            changed_digest = _canonical_protocol_digest(changed)
            with self.assertRaisesRegex(
                MatrixStateError, "does not match current"
            ):
                _completed_cells(
                    root,
                    [job],
                    allow_diagnostic=False,
                    expected_protocols={
                        job.key: (changed, changed_digest),
                    },
                )


class NativeDNAMatrixExecutionIsolationTest(unittest.TestCase):
    def test_child_failure_does_not_cancel_remaining_queue(self) -> None:
        jobs = [
            Job("bee2018", "CIFAR10", seed)
            for seed in DEFAULT_SEEDS
        ]
        seen: list[tuple[tuple[str, str, int], str]] = []
        seen_lock = threading.Lock()

        def fake_run(job: Job, gpu: str) -> RunResult:
            with seen_lock:
                seen.append((job.key, gpu))
            return RunResult(
                job,
                gpu,
                7 if job.seed == 43 else 0,
                Path(f"/tmp/{job.job_id}.log"),
            )

        results = _run_pool(jobs, ["0", "1"], fake_run)
        self.assertEqual(
            {item[0] for item in seen}, {job.key for job in jobs}
        )
        self.assertEqual(len(results), len(jobs))
        self.assertEqual(
            [(item.job.seed, item.returncode) for item in results],
            [(42, 0), (43, 7), (44, 0)],
        )

    def test_main_returns_nonzero_only_after_all_children_finish(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seen: list[int] = []
            seen_lock = threading.Lock()

            def fake_cell(
                job: Job, gpu: str, _options: LaunchOptions
            ) -> RunResult:
                with seen_lock:
                    seen.append(job.seed)
                return RunResult(
                    job,
                    gpu,
                    9 if job.seed == 43 else 0,
                    root / f"{job.job_id}.log",
                )

            output = io.StringIO()
            with mock.patch(
                "scripts.run_native_dna_p0_matrix._validate_data_root",
                return_value=root,
            ), mock.patch(
                "scripts.run_native_dna_p0_matrix._run_cell",
                side_effect=fake_cell,
            ), contextlib.redirect_stdout(output):
                returncode = main(
                    [
                        "--data-root",
                        "/data/static-test",
                        "--gpus",
                        "0",
                        "1",
                        "--methods",
                        "bee2018",
                        "--datasets",
                        "CIFAR10",
                        "--seeds",
                        "42",
                        "43",
                        "44",
                        "--python",
                        sys.executable,
                        "--driver",
                        str(RUNNER),
                        "--dataset-root",
                        str(root),
                    ]
                )
            self.assertEqual(returncode, 1)
            self.assertEqual(sorted(seen), [42, 43, 44])
            self.assertIn("finished failures=1", output.getvalue())
            summaries = list((root / "logs").glob("matrix_summary_*.json"))
            self.assertEqual(len(summaries), 1)
            summary = json.loads(summaries[0].read_text())
            self.assertEqual(summary["scheduled"], 3)
            self.assertEqual(len(summary["failures"]), 1)

    def test_dry_run_launches_nothing_and_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset_root = Path(directory)
            data_root = Path(
                "/data"
            ) / f"codex_native_matrix_static_{uuid.uuid4().hex}"
            self.assertFalse(data_root.exists())
            output = io.StringIO()
            with mock.patch(
                "scripts.run_native_dna_p0_matrix.subprocess.run",
                side_effect=AssertionError("dry-run launched a child"),
            ), contextlib.redirect_stdout(output):
                returncode = main(
                    [
                        "--data-root",
                        str(data_root),
                        "--gpus",
                        "7",
                        "--methods",
                        "bee2018",
                        "--datasets",
                        "CIFAR10",
                        "--seeds",
                        "42",
                        "--python",
                        sys.executable,
                        "--driver",
                        str(RUNNER),
                        "--dataset-root",
                        str(dataset_root),
                        "--dry-run",
                    ]
                )
            self.assertEqual(returncode, 0)
            self.assertIn("CUDA_VISIBLE_DEVICES=7", output.getvalue())
            self.assertIn("--device cuda:0", output.getvalue())
            self.assertFalse(data_root.exists())


if __name__ == "__main__":
    unittest.main()
