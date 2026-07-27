import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.run_native_dna_p0 import (
    DEFAULT_CACHE,
    EVAL_PERIOD,
    MATCHED_LENGTH,
    METHODS,
    SOURCE_PROFILES,
    SOURCE_REPRODUCTION_AUDIT,
    STRICT_CACHE_BLOCKER,
    _audit_cache,
    _expected_split_cardinalities,
    _manifest,
    _require_complete_post_compliance,
    _training_command,
    _verify_aligned_extractions,
    _verify_selection,
)


def _option(command, name):
    return command[command.index(name) + 1]


def test_source_profiles_and_visual_only_default_caches_are_fixed():
    assert METHODS == ("bee2018", "bee2021", "koike2024", "koike2026")
    assert {
        method: profile["horizon"]
        for method, profile in SOURCE_PROFILES.items()
    } == {
        "bee2018": 65,
        "bee2021": 100,
        "koike2024": 150,
        "koike2026": 1000,
    }
    assert all(
        profile["grad_clip"] == 0.0 for profile in SOURCE_PROFILES.values()
    )
    assert "Eq. (2)" in SOURCE_REPRODUCTION_AUDIT["koike2024"][
        "equation_choice"
    ]
    assert "statistically" in SOURCE_REPRODUCTION_AUDIT["bee2021"][
        "pair_sampling"
    ]
    assert DEFAULT_CACHE == {
        "Flickr25k": "cache/flickr25k_clip_v4plus",
        "MSCOCO": "cache/mscoco_clip_v4plus",
        "NUSWIDE": "cache/nuswide_clip",
        "CIFAR10": "cache/cifar10_clip",
    }


def test_commands_encode_raw_stage1_and_scratch_refit_contract(tmp_path):
    common = dict(
        method="koike2024",
        dataset="Flickr25k",
        setting="setting1",
        dataset_root=tmp_path / "dataset",
        cache_dir=tmp_path / "cache",
        seed=43,
        device="cpu",
        num_workers=0,
        extract_batch_size=32,
        query_chunk=8,
        predictor=None,
        save_train_extract=False,
        save_soft_probs=False,
    )
    stage1 = _training_command(
        **common,
        output=tmp_path / "stage1",
        epochs=SOURCE_PROFILES["koike2024"]["horizon"],
        eval_period=EVAL_PERIOD,
        val_ratio=0.1,
    )
    assert _option(stage1, "--length") == str(MATCHED_LENGTH)
    assert _option(stage1, "--epochs") == "150"
    assert _option(stage1, "--eval_period") == "5"
    assert _option(stage1, "--val_split_ratio") == "0.1"
    assert _option(stage1, "--val_split_seed") == "42"
    assert _option(stage1, "--selection_metric") == "neural_raw"
    assert _option(stage1, "--grad_clip") == "0.0"
    assert "--weights" not in stage1
    assert "--allow_nonempty_out" not in stage1

    refit = _training_command(
        **{
            **common,
            "save_train_extract": True,
            "save_soft_probs": True,
        },
        output=tmp_path / "refit",
        epochs=15,
        eval_period=15,
        val_ratio=0.0,
    )
    assert _option(refit, "--epochs") == "15"
    assert _option(refit, "--eval_period") == "15"
    assert _option(refit, "--val_split_ratio") == "0.0"
    assert "--weights" not in refit
    assert "--save_train_extract" in refit
    assert "--save_soft_probs" in refit


def test_bee2021_requires_and_binds_predictor(tmp_path):
    common = dict(
        method="bee2021",
        dataset="CIFAR10",
        setting="setting1",
        dataset_root=tmp_path / "dataset",
        cache_dir=tmp_path / "cache",
        output=tmp_path / "stage1",
        seed=42,
        epochs=100,
        eval_period=5,
        val_ratio=0.1,
        device="cpu",
        num_workers=0,
        extract_batch_size=32,
        query_chunk=8,
        save_train_extract=False,
        save_soft_probs=False,
    )
    with unittest.TestCase().assertRaisesRegex(ValueError, "requires"):
        _training_command(**common, predictor=None)
    predictor = tmp_path / "predictor.npz"
    command = _training_command(**common, predictor=predictor)
    assert _option(command, "--primo_predictor_npz") == str(predictor)


def test_cache_audit_hashes_only_three_consumed_inputs_and_blocks_legacy(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "meta.json").write_text(
        json.dumps(
            {
                "N": 3,
                "D_proj": 2,
                "dtype": "float16",
                "backbone": "clip-fixture",
            }
        ),
        encoding="utf-8",
    )
    (cache / "image_ids.json").write_text(
        json.dumps(["a", "b", "c"]), encoding="utf-8"
    )
    np.save(
        cache / "visual_global.f16.npy",
        np.zeros((3, 2), dtype=np.float16),
    )
    # Unconsumed arrays must not silently enter the native protocol identity.
    np.save(cache / "visual_tokens.f16.npy", np.zeros((3, 4, 2), dtype=np.float16))

    audit = _audit_cache(cache)

    assert set(audit["artifacts"]) == {
        "meta.json",
        "image_ids.json",
        "visual_global.f16.npy",
    }
    assert audit["blockers"] == [STRICT_CACHE_BLOCKER]
    assert audit["strict_provenance_complete"] is False


def _stage1_fixture(root: Path):
    root.mkdir()
    config = {
        "method": "koike2024",
        "dataset": "Flickr25k",
        "setting": "setting1",
        "length": 18,
        "seed": 42,
        "epochs": 150,
        "eval_period": 5,
        "val_split_ratio": 0.1,
        "val_split_seed": 42,
        "selection_metric": "neural_raw",
        "gc_min": 0.4,
        "gc_max": 0.6,
        "max_run": 3,
        "map_at_r": 5000,
        "weights": None,
        "allow_nonempty_out": False,
        "allow_checkpoint_context_mismatch": False,
        "save_train_extract": False,
        "save_soft_probs": False,
        "batch_size": 128,
        "steps_per_epoch": None,
        "lr": 0.01,
        "optimizer": "adagrad",
        "grad_clip": 0.0,
        "bee_pair_sampling": None,
        "cache_dir": str((root.parent / "cache").resolve()),
        "dataset_root": str((root.parent / "dataset").resolve()),
        "out": str(root.resolve()),
        "primo_predictor_npz": None,
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    split = {
        "ratio": 0.1,
        "seed": 42,
        "strategy": "fixture",
        "n_full": 10,
        "n_opt": 9,
        "n_val": 1,
        "val_indices": [2],
    }
    (root / "val_split.json").write_text(json.dumps(split), encoding="utf-8")
    (root / "log.csv").write_text("epoch,loss\n", encoding="utf-8")
    scores = {}
    for epoch in range(4, 150, 5):
        checkpoint = root / f"epoch_{epoch:03d}.pth"
        checkpoint.write_bytes(f"checkpoint-{epoch}".encode())
        # E=9 and E=14 tie; earliest E=9 must win.
        score = 0.8 if epoch in (9, 14) else 0.1 + epoch / 10000
        scores[epoch] = score
        evaluation = {
            "epoch": epoch,
            "checkpoint": str(checkpoint),
            "neural_raw": {"mAP_at_R": score},
            "projected": {"mAP_at_R": 0.99 - epoch / 10000},
        }
        (root / f"eval_val_epoch_{epoch:03d}.json").write_text(
            json.dumps(evaluation), encoding="utf-8"
        )
    selection = {
        "selection_metric": "val_neural_raw_mAP_at_R",
        "best_epoch_zero_based": 9,
        "refit_epochs": 10,
        "best_score": scores[9],
        "checkpoint": str(root / "epoch_009.pth"),
        "split": split,
        "test_evaluated": False,
    }
    (root / "selection.json").write_text(
        json.dumps(selection), encoding="utf-8"
    )


def test_selection_verification_recomputes_earliest_raw_argmax_and_binds_set():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        stage1 = root / "stage1"
        _stage1_fixture(stage1)
        audit = _verify_selection(
            stage1,
            method="koike2024",
            dataset="Flickr25k",
            setting="setting1",
            seed=42,
            cache_dir=root / "cache",
            dataset_root=root / "dataset",
            predictor=None,
        )
        assert audit["best_epoch_zero_based"] == 9
        assert audit["refit_epochs"] == 10
        assert len(audit["stage1_checkpoint_sha256"]) == 30
        assert len(audit["stage1_checkpoint_set_sha256"]) == 64

        # A missing candidate makes the whole selection artifact invalid.
        (stage1 / "epoch_014.pth").unlink()
        with unittest.TestCase().assertRaisesRegex(ValueError, "checkpoint set"):
            _verify_selection(
                stage1,
                method="koike2024",
                dataset="Flickr25k",
                setting="setting1",
                seed=42,
                cache_dir=root / "cache",
                dataset_root=root / "dataset",
                predictor=None,
            )


def test_post_compliance_is_fail_closed():
    valid = {
        "query": {"post_compliance": 1.0},
        "database": {"post_compliance": 1.0},
    }
    assert _require_complete_post_compliance(valid) == {
        "query": 1.0,
        "database": 1.0,
    }
    invalid = {
        "query": {"post_compliance": 1.0},
        "database": {"post_compliance": 0.999},
    }
    with unittest.TestCase().assertRaisesRegex(
        ValueError, "required exactly 1.0"
    ):
        _require_complete_post_compliance(invalid)


def test_saved_terminal_extractions_are_aligned_and_bio_valid(tmp_path):
    labels = np.array([[1, 0], [0, 1]], dtype=np.int64)
    paths = np.array(["first", "second"])
    valid = np.array(
        [
            [0, 1] * 9,
            [3, 2] * 9,
        ],
        dtype=np.int64,
    )
    canonical_to_paper = np.asarray([0, 2, 3, 1], dtype=np.int64)
    soft = np.eye(4, dtype=np.float16)[canonical_to_paper[valid]]
    for split, count in (("query", 1), ("db", 2)):
        split_codes = valid[:count]
        common = {
            "multi_hot_labels": labels[:count],
            "image_paths": paths[:count],
        }
        np.savez(
            tmp_path / f"extract_{split}_neural_raw.npz",
            base_indices=split_codes,
            soft_probs_atcg=soft[:count],
            **common,
        )
        np.savez(
            tmp_path / f"extract_{split}.npz",
            base_indices=split_codes,
            neural_raw_base_indices=split_codes,
            soft_probs_atcg=soft[:count],
            **common,
        )
        np.savez(
            tmp_path / f"extract_{split}_bioproj.npz",
            base_indices=split_codes,
            **common,
        )
    np.savez(
        tmp_path / "extract_train.npz",
        base_indices=valid,
        neural_raw_base_indices=valid,
        soft_probs_atcg=soft,
        multi_hot_labels=labels,
        image_paths=paths,
    )
    artifacts = _verify_aligned_extractions(
        tmp_path,
        method="koike2024",
        expected_counts={"query": 1, "database": 2, "train": 2},
        evaluation_counts={"query": 1, "database": 2},
        cache_cardinality=2,
    )
    assert len(artifacts) == 7
    assert all(len(record["sha256"]) == 64 for record in artifacts.values())

    with unittest.TestCase().assertRaisesRegex(
        ValueError, "differs from saved evaluation"
    ):
        _verify_aligned_extractions(
            tmp_path,
            method="koike2024",
            expected_counts={"query": 1, "database": 2, "train": 2},
            evaluation_counts={"query": 2, "database": 2},
            cache_cardinality=2,
        )
    with unittest.TestCase().assertRaisesRegex(
        ValueError, "expected cache cardinality"
    ):
        _verify_aligned_extractions(
            tmp_path,
            method="koike2024",
            expected_counts={"query": 1, "database": 2, "train": 2},
            evaluation_counts={"query": 1, "database": 2},
            cache_cardinality=3,
        )

    alternative_valid = np.array([[0, 2] * 9], dtype=np.int64)
    np.savez(
        tmp_path / "extract_query_bioproj.npz",
        base_indices=alternative_valid,
        multi_hot_labels=labels[:1],
        image_paths=paths[:1],
    )
    with unittest.TestCase().assertRaisesRegex(ValueError, "exact common-DP"):
        _verify_aligned_extractions(tmp_path, method="koike2024")

    np.savez(
        tmp_path / "extract_query_bioproj.npz",
        base_indices=np.zeros((1, 18), dtype=np.int64),
        multi_hot_labels=labels[:1],
        image_paths=paths[:1],
    )
    with unittest.TestCase().assertRaisesRegex(ValueError, "100% compliant"):
        _verify_aligned_extractions(tmp_path, method="koike2024")


def test_expected_split_cardinalities_are_derived_without_loading_test_data(
    tmp_path,
):
    split_root = tmp_path / "Flickr25k" / "setting1"
    split_root.mkdir(parents=True)
    for filename, rows in (
        ("train.txt", ["a 1", "b 1"]),
        ("test.txt", ["c 1"]),
        ("database.txt", ["a 1", "b 1", "d 1"]),
    ):
        (split_root / filename).write_text("\n".join(rows) + "\n")
    assert _expected_split_cardinalities(
        "Flickr25k", "setting1", tmp_path
    ) == {"train": 2, "query": 1, "database": 3}
    assert _expected_split_cardinalities(
        "CIFAR10", "setting1", tmp_path
    ) == {"train": 5000, "query": 1000, "database": 59000}


def test_manifest_matches_native_aggregator_contract(tmp_path):
    selection = {
        "best_epoch_zero_based": 9,
        "refit_epochs": 10,
        "best_val_raw_mAP_at_R": 0.8,
        "selection_artifact": {
            "path": str(tmp_path / "selection.json"),
            "sha256": "d" * 64,
        },
        "stage1_checkpoint_sha256": {"epoch_009.pth": "e" * 64},
        "stage1_checkpoint_set_sha256": "f" * 64,
    }
    evaluation = {
        "path": str(tmp_path / "evaluation_native_dna.json"),
        "sha256": "a" * 64,
    }
    checkpoint = {
        "path": str(tmp_path / "epoch_009.pth"),
        "sha256": "b" * 64,
    }
    refit = {
        "evaluation_native_dna": evaluation,
        "final_checkpoint": checkpoint,
        "extraction_artifacts": {},
        "post_compliance": {"query": 1.0, "database": 1.0},
        "raw_mAP_at_R": 0.4,
        "post_dp_mAP_at_R": 0.3,
        "post_dp_database_unique_ratio": 0.9,
        "refit_config": {},
        "refit_log": {},
        "resolved_device": "cpu",
    }
    manifest = _manifest(
        method="koike2024",
        dataset="Flickr25k",
        setting="setting1",
        seed=42,
        cell_root=tmp_path,
        stage1_dir=tmp_path / "stage1",
        refit_dir=tmp_path / "refit",
        protocol_identity={
            "fixture": True,
            "cache": {"artifacts": {}},
            "implementation_artifacts": {},
            "source_reproduction_audit": {"equation_choice": "fixture"},
        },
        protocol_digest="c" * 64,
        blockers=[STRICT_CACHE_BLOCKER],
        selection_audit=selection,
        refit_audit=refit,
    )
    assert manifest["method"] == "koike2024"
    assert manifest["dataset"] == "Flickr25k"
    assert manifest["seed"] == 42
    assert manifest["best_epoch_zero_based"] == 9
    assert manifest["refit_epochs"] == 10
    assert manifest["run_manifest_phase"] == "completed"
    assert manifest["main_protocol_eligible"] is False
    assert manifest["blockers"] == [STRICT_CACHE_BLOCKER]
    assert manifest["evaluation_native_dna"] == evaluation
    assert manifest["final_checkpoint"] == checkpoint
    assert manifest["source_reproduction_audit"] == {
        "equation_choice": "fixture"
    }


class NativeP0DriverTest(unittest.TestCase):
    test_source_profiles_and_visual_only_default_caches_are_fixed = staticmethod(
        test_source_profiles_and_visual_only_default_caches_are_fixed
    )

    def test_commands_encode_raw_stage1_and_scratch_refit_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            test_commands_encode_raw_stage1_and_scratch_refit_contract(
                Path(directory)
            )

    def test_bee2021_requires_and_binds_predictor(self):
        with tempfile.TemporaryDirectory() as directory:
            test_bee2021_requires_and_binds_predictor(Path(directory))

    def test_cache_audit_hashes_only_three_consumed_inputs_and_blocks_legacy(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            test_cache_audit_hashes_only_three_consumed_inputs_and_blocks_legacy(
                Path(directory)
            )

    test_selection_verification_recomputes_earliest_raw_argmax_and_binds_set = (
        staticmethod(
            test_selection_verification_recomputes_earliest_raw_argmax_and_binds_set
        )
    )
    test_post_compliance_is_fail_closed = staticmethod(
        test_post_compliance_is_fail_closed
    )

    def test_saved_terminal_extractions_are_aligned_and_bio_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            test_saved_terminal_extractions_are_aligned_and_bio_valid(
                Path(directory)
            )

    def test_manifest_matches_native_aggregator_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            test_manifest_matches_native_aggregator_contract(Path(directory))


if __name__ == "__main__":
    unittest.main()
