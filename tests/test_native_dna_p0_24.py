import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

# F18: these are the LENGTH-PARAMETERISED shared modules, not 18-base ones.
# The old aliases (`aggregate18`, `driver18`, `matrix18`) asserted a seal at
# 18 that never existed -- the length came from the environment, so the
# assertion broke the moment the paper moved to 15 bases.
from scripts import aggregate_native_dna_p0 as shared_aggregate
from scripts import run_native_dna_p0 as shared_driver
from scripts import run_native_dna_p0_matrix as shared_matrix
from scripts.aggregate_native_dna_p0_24 import (
    COMMON_BIO_PROJECTION,
    Key,
    METHOD_PROTOCOL_LOCK_SHA256,
    _markdown as markdown24,
    _validate_manifest,
    aggregate as aggregate24,
)
from scripts.run_native_dna_p0_24 import (
    GC_COUNT_MAX,
    GC_COUNT_MIN,
    MATCHED_LENGTH,
    PIPELINE_VARIANT,
    PRIMO_TRANSFER_BLOCKER,
    TRAINER,
    _protocol_identity,
    _training_command,
    configured_canonical_driver,
)
from scripts.run_native_dna_p0_matrix_24 import (
    DEFAULT_DATASETS,
    DEFAULT_METHODS,
    DEFAULT_SEEDS,
    _validate_data_root_24,
    configured_canonical_matrix,
)
from scripts.train_native_dna_baseline_24 import (
    LEGACY_NON18_LABEL,
    PROTOCOL_LABEL,
    _normalized_evaluation,
)


IMPLEMENTATION_24_SHA256 = {
    "baseline/base_model.py": "b9d7e1a0cfab613fd72ce3dc38fafeca82ca99f6121d20138bf22239ef34f3be",
    "baseline/cache_provenance.py": "4fe40c862a23e265ecf0559232d8b6ac5b84b282df720670c3aed58e3daf588d",
    "baseline/native_dna.py": "3a0c49c4b49ba54db0eb1ce6a127c62279023aee2e6a5f90e2b5fa5943cf91b5",
    "dataloaders.py": "8632d4328e529c5b9967f4ffadcc5eeda6809c51fa844d3fdbdda2462e5f8315",
    "dna_utils/bio_constraints.py": "f7863bd363ebe42989f232f550a0af837dde088abf01bada8a0c281f6f164030",
    "scripts/run_native_dna_p0.py": "9a155dd1255163701a214d62fb380d6f1ef6c8d9c6500bb6b9a36c3d9ed444a7",
    "scripts/run_native_dna_p0_24.py": "bfb20e0aebee2579f8dcedeb8389d18d626a914096bfb6f4481e50d443089684",
    "scripts/train_native_dna_baseline.py": "c85d8243047a6ed6a61b34de64d1bdda5b963a0a5b8dc1caab56692734c5f051",
    "scripts/train_native_dna_baseline_24.py": "0ad4ed94a1fc5ac3d54077bdfa1455e49fdca5ee8bc87d4cc24e02397af55cc3",
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


def _source_audit(method: str) -> dict[str, object]:
    audit = copy.deepcopy(shared_driver.SOURCE_REPRODUCTION_AUDIT[method])
    if method == "bee2021":
        audit["declared_adaptation"] = (
            "The official 80-nt yield predictor is frozen and transferred to "
            "24 nt; PRIMO's per-epoch NUPACK relabeling/predictor refit is not "
            "performed."
        )
        audit["length_transfer"] = {
            "source_predictor_length_bases": 80,
            "target_length_bases": 24,
            "predictor_frozen": True,
            "alternating_predictor_refit": False,
        }
    return audit


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return _sha256(path)


def _artifact(token: str) -> dict[str, object]:
    return {
        "path": f"/sealed/{token}",
        "sha256": hashlib.sha256(token.encode()).hexdigest(),
        "bytes": len(token),
    }


def test_24_command_is_isolated_from_the_sealed_18_base_driver(tmp_path):
    command = _training_command(
        method="koike2024",
        dataset="Flickr25k",
        setting="setting1",
        dataset_root=tmp_path / "dataset",
        cache_dir=tmp_path / "cache",
        output=tmp_path / "stage1",
        seed=43,
        epochs=150,
        eval_period=5,
        val_ratio=0.1,
        device="cpu",
        num_workers=0,
        extract_batch_size=32,
        query_chunk=8,
        predictor=None,
        save_train_extract=False,
        save_soft_probs=False,
    )
    assert Path(command[1]).resolve() == TRAINER.resolve()
    assert command[command.index("--length") + 1] == "24"
    assert command[command.index("--gc_min") + 1] == "0.4"
    assert command[command.index("--gc_max") + 1] == "0.6"
    # The sealed 24-base driver must be a DIFFERENT object from the shared one,
    # and the shared one must not have been dragged to 24 by importing this.
    assert shared_driver._training_command is not _training_command
    assert shared_driver.MATCHED_LENGTH != MATCHED_LENGTH == 24


def test_protocol_identity_binds_exact_gc_counts_and_primo_transfer(tmp_path):
    split = tmp_path / "dataset" / "Flickr25k" / "setting1"
    split.mkdir(parents=True)
    for name in ("train.txt", "test.txt", "database.txt"):
        (split / name).write_text(f"{name}\n", encoding="utf-8")
    predictor = tmp_path / "predictor.npz"
    predictor.write_bytes(b"frozen-primo")
    identity, digest = _protocol_identity(
        method="bee2021",
        dataset="Flickr25k",
        setting="setting1",
        seed=42,
        cache_audit={
            "cache_dir": "/sealed/cache",
            "artifacts": {"visual_global.f16.npy": _artifact("cache")},
        },
        dataset_root=tmp_path / "dataset",
        predictor=predictor,
        device="cuda:0",
        num_workers=4,
        extract_batch_size=512,
        query_chunk=64,
    )
    assert identity["matched_length_bases"] == 24
    assert identity["pipeline_variant"] == PIPELINE_VARIANT
    assert identity["common_bio_projection"] == COMMON_BIO_PROJECTION
    assert identity["primo_frozen_predictor_length_transfer"] == {
        "source_predictor_length_bases": 80,
        "target_length_bases": 24,
        "predictor_frozen": True,
        "alternating_predictor_refit": False,
        "eligibility_blocker": PRIMO_TRANSFER_BLOCKER,
    }
    canonical = json.dumps(
        identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    assert digest == hashlib.sha256(canonical).hexdigest()
    assert shared_driver.MATCHED_LENGTH != MATCHED_LENGTH == 24


def test_terminal_metadata_relabel_is_exact_and_refuses_unexpected_input():
    normalized = _normalized_evaluation({
        "method": "bee2018",
        "length": 24,
        "protocol": LEGACY_NON18_LABEL,
        "neural_raw": {"mAP_at_R": 0.5},
    })
    assert normalized["protocol"] == PROTOCOL_LABEL
    assert normalized["matched_capacity_bases"] == 24
    assert normalized["common_dp_gc_count_range"] == [10, 14]
    with pytest.raises(ValueError, match="unexpected"):
        _normalized_evaluation({
            "length": 24,
            "protocol": "matched_18nt_adaptation",
        })
    with pytest.raises(ValueError, match="length mismatch"):
        _normalized_evaluation({
            "length": 18,
            "protocol": LEGACY_NON18_LABEL,
        })


def test_24_base_bio_validation_means_gc_10_to_14_and_run_at_most_three():
    from baseline.native_dna import project_codes_memoized

    valid_10 = np.asarray([[1, 0, 2, 3] * 5 + [0, 3, 0, 3]], dtype=np.int8)
    invalid_9 = valid_10.copy()
    invalid_9[0, 0] = 0
    valid_14 = np.asarray(
        [[1, 0] * 10 + [1, 2, 1, 2]], dtype=np.int8
    )
    invalid_15 = valid_14.copy()
    invalid_15[0, 1] = 1
    invalid_run = valid_10.copy()
    invalid_run[0, :4] = 0
    # F18: the 24-base window is passed in, not patched into a global that the
    # 15- and 18-base paths also read.
    check = shared_driver._valid_bio_codes
    assert check(valid_10, protocol=24).tolist() == [True]
    assert check(invalid_9, protocol=24).tolist() == [False]
    assert check(valid_14, protocol=24).tolist() == [True]
    assert check(invalid_15, protocol=24).tolist() == [False]
    assert check(invalid_run, protocol=24).tolist() == [False]
    # ...and the same codes are NOT valid under the paper's 15-base window,
    # which is the contamination this replaces.
    assert check(valid_10, protocol=15).tolist() == [False]
    projected = project_codes_memoized(
        np.zeros((1, 24), dtype=np.int8), 0.4, 0.6, 3
    )
    projected_code = np.asarray(projected["projected_codes"])
    gc_count = int(((projected_code == 1) | (projected_code == 2)).sum())
    assert GC_COUNT_MIN <= gc_count <= GC_COUNT_MAX
    assert projected["post_compliance"] == 1.0
    assert (GC_COUNT_MIN, GC_COUNT_MAX, MATCHED_LENGTH) == (10, 14, 24)


def _manifest_fixture(root: Path, method: str = "bee2018") -> Path:
    dataset = "Flickr25k"
    checkpoint = root / "epoch_009.pth"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"checkpoint")
    evaluation = {
        "method": method,
        "dataset": dataset,
        "length": 24,
        "protocol": PROTOCOL_LABEL,
        "matched_capacity_bases": 24,
        "common_dp_gc_count_range": [10, 14],
        "supervision": (
            "ground-truth labels"
            if method.startswith("koike")
            else "feature-distance pairs"
        ),
        "neural_raw": {"mAP_at_R": 0.72, "mAP_R_cutoff": 5000},
        "projected": {"mAP_at_R": 0.70, "mAP_R_cutoff": 5000},
        "query": {"post_compliance": 1.0},
        "database": {
            "post_compliance": 1.0,
            "unique_ratio_projected": 0.8,
        },
    }
    if method == "bee2021":
        evaluation["method_variant"] = "frozen_predictor_length_transfer"
    evaluation_path = root / "evaluation_native_dna.json"
    evaluation_sha = _json(evaluation_path, evaluation)
    horizon = int(shared_driver.SOURCE_PROFILES[method]["horizon"])
    implementation = {
        relative: {
            "path": f"/sealed/{relative}",
            "sha256": digest,
            "bytes": 1,
        }
        for relative, digest in IMPLEMENTATION_24_SHA256.items()
    }
    predictor = None
    transfer = None
    if method == "bee2021":
        predictor = {
            "path": "/sealed/primo-predictor.npz",
            "sha256": OFFICIAL_PRIMO_PREDICTOR_SHA256,
            "bytes": 16878,
        }
        transfer = {
            "source_predictor_length_bases": 80,
            "target_length_bases": 24,
            "predictor_frozen": True,
            "alternating_predictor_refit": False,
            "eligibility_blocker": PRIMO_TRANSFER_BLOCKER,
        }
    identity = {
        "schema_version": 1,
        "method": method,
        "dataset": dataset,
        "setting": "setting1",
        "seed": 42,
        "matched_length_bases": 24,
        "pipeline_variant": PIPELINE_VARIANT,
        "val_split_ratio": 0.1,
        "val_split_seed": 42,
        "selection_metric": "val_neural_raw_mAP_at_R",
        "selection_distance": "base_hamming",
        "source_profile": copy.deepcopy(shared_driver.SOURCE_PROFILES[method]),
        "source_reproduction_audit": _source_audit(method),
        "candidate_eval_period": 5,
        "candidate_epochs_zero_based": list(range(4, horizon, 5)),
        "map_at_R": 5000,
        "stage_contract": _stage_contract(),
        "common_bio_projection": dict(COMMON_BIO_PROJECTION),
        "cache": {
            "cache_dir": "/sealed/cache",
            "artifacts": {"visual_global.f16.npy": _artifact("cache")},
        },
        "dataset_root": "/sealed/dataset",
        "split_artifacts": {"train.txt": _artifact("split")},
        "implementation_artifacts": implementation,
        "primo_predictor": predictor,
        "execution": {
            "device_request": "cuda:0",
            "num_workers": 4,
            "extract_batch_size": 512,
            "query_chunk": 64,
            "runtime_versions": {
                "python": "test",
                "numpy": "test",
                "torch": "test",
            },
        },
    }
    if transfer is not None:
        identity["primo_frozen_predictor_length_transfer"] = transfer
    digest = hashlib.sha256(json.dumps(
        identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")).hexdigest()
    manifest = {
        "schema_version": 1,
        "method": method,
        "dataset": dataset,
        "setting": "setting1",
        "seed": 42,
        "length": 24,
        "base_length": 24,
        "pipeline_variant": PIPELINE_VARIANT,
        "common_dp_gc_count_range": [10, 14],
        "test_used_for_selection": False,
        "test_access_contract": dict(shared_aggregate.TEST_ACCESS_CONTRACT),
        "best_epoch_zero_based": 9,
        "refit_epochs": 10,
        "run_manifest_phase": "completed",
        "main_protocol_eligible": method != "bee2021",
        "main_eligibility_blockers": (
            []
            if method != "bee2021"
            else [PRIMO_TRANSFER_BLOCKER]
        ),
        "evaluation_native_dna": {
            "path": str(evaluation_path),
            "sha256": evaluation_sha,
        },
        "final_checkpoint": {
            "path": str(checkpoint),
            "sha256": _sha256(checkpoint),
        },
        "protocol_identity": identity,
        "protocol_digest_sha256": digest,
    }
    if transfer is not None:
        manifest["primo_frozen_predictor_length_transfer"] = {
            key: value
            for key, value in transfer.items()
            if key != "alternating_predictor_refit"
        }
    manifest_path = root / "native_p0_run_manifest.json"
    _json(manifest_path, manifest)
    return manifest_path


def test_24_aggregator_accepts_only_explicit_24_base_contract(tmp_path):
    manifest_path = _manifest_fixture(tmp_path)
    key = Key("bee2018", "Flickr25k", 42)
    record = _validate_manifest(manifest_path, key)
    assert record["status"] == "complete_main_eligible"
    assert (
        record["method_protocol_lock_sha256"]
        == METHOD_PROTOCOL_LOCK_SHA256["bee2018"]
    )
    assert record["validation_errors"] == []

    evaluation_path = tmp_path / "evaluation_native_dna.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation.pop("common_dp_gc_count_range")
    evaluation_sha = _json(evaluation_path, evaluation)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["evaluation_native_dna"]["sha256"] = evaluation_sha
    _json(manifest_path, manifest)
    record = _validate_manifest(manifest_path, key)
    assert record["status"] == "invalid"
    assert any(
        "common_dp_gc_count_range" in error
        for error in record["validation_errors"]
    )


def test_24_markdown_surfaces_unsupervised_and_supervised_regimes(tmp_path):
    _manifest_fixture(tmp_path)
    payload, _ = aggregate24([tmp_path])
    rendered = markdown24(payload)
    assert rendered.startswith("# Native-DNA 24-base common-P0 aggregation")
    assert "repository-local information-condition tags" in rendered
    assert "`U0-FD` = unsupervised with respect to the target benchmark" in rendered
    assert (
        "held-out labels are used only to score validation retrieval and select E*"
    ) in rendered
    assert (
        "## Unsupervised (`U0-FD`; target-label-free encoder objective) "
        "feature-distance-pair direct predecessors"
    ) in rendered
    assert "## Supervised (`S`) direct-prior baselines" in rendered


@pytest.mark.parametrize(
    "mutation",
    (
        "source_profile",
        "source_audit",
        "candidate_grid",
        "stage_contract",
        "bio_projection",
        "implementation_sha",
        "execution",
        "pipeline_variant",
    ),
)
def test_24_stable_protocol_mutations_fail_after_digest_recompute(
    tmp_path, mutation
):
    manifest_path = _manifest_fixture(tmp_path / mutation)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = manifest["protocol_identity"]
    if mutation == "source_profile":
        identity["source_profile"]["optimizer"] = "sgd"
    elif mutation == "source_audit":
        identity["source_reproduction_audit"]["reference"] = "different-paper"
    elif mutation == "candidate_grid":
        identity["candidate_epochs_zero_based"] = [4, 9]
    elif mutation == "stage_contract":
        identity["stage_contract"]["terminal_evaluation"][
            "official_test_extraction_passes"
        ] = 2
    elif mutation == "bio_projection":
        identity["common_bio_projection"]["gc_count_max_inclusive"] = 15
    elif mutation == "implementation_sha":
        identity["implementation_artifacts"]["baseline/native_dna.py"][
            "sha256"
        ] = "f" * 64
    elif mutation == "execution":
        identity["execution"]["extract_batch_size"] = 256
    elif mutation == "pipeline_variant":
        identity["pipeline_variant"] = "forged-24-v1"
    manifest["protocol_digest_sha256"] = shared_aggregate._canonical_digest(identity)
    _json(manifest_path, manifest)
    record = _validate_manifest(
        manifest_path, Key("bee2018", "Flickr25k", 42)
    )
    assert record["status"] == "invalid"
    assert any(
        "method_protocol_lock_sha256" in error
        for error in record["validation_errors"]
    )


@pytest.mark.parametrize("mutation", ("predictor_sha", "transfer"))
def test_24_primo_lock_pins_predictor_and_transfer(tmp_path, mutation):
    manifest_path = _manifest_fixture(tmp_path / mutation, method="bee2021")
    initial = _validate_manifest(
        manifest_path, Key("bee2021", "Flickr25k", 42)
    )
    assert initial["status"] == "complete_diagnostic_only"
    assert (
        initial["method_protocol_lock_sha256"]
        == METHOD_PROTOCOL_LOCK_SHA256["bee2021"]
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = manifest["protocol_identity"]
    if mutation == "predictor_sha":
        identity["primo_predictor"]["sha256"] = "e" * 64
    else:
        identity["primo_frozen_predictor_length_transfer"][
            "target_length_bases"
        ] = 23
    manifest["protocol_digest_sha256"] = shared_aggregate._canonical_digest(identity)
    _json(manifest_path, manifest)
    record = _validate_manifest(
        manifest_path, Key("bee2021", "Flickr25k", 42)
    )
    assert record["status"] == "invalid"
    assert any(
        "method_protocol_lock_sha256" in error
        for error in record["validation_errors"]
    )


def test_24_matrix_has_48_cells_and_rejects_a_mixed_18_base_root(
    tmp_path, monkeypatch
):
    with configured_canonical_driver(), configured_canonical_matrix():
        jobs = shared_matrix._jobs(
            DEFAULT_METHODS, DEFAULT_DATASETS, DEFAULT_SEEDS
        )
        assert len(jobs) == 48
        assert all("_24nt_P0_" in job.job_id for job in jobs)

    mixed = tmp_path / "matrix"
    runs = mixed / "runs"
    runs.mkdir(parents=True)
    (runs / "bee2018_flickr25k_18nt_P0_seed42_pabc").mkdir()
    import scripts.run_native_dna_p0_matrix_24 as matrix24

    monkeypatch.setattr(
        matrix24, "_ORIGINAL_VALIDATE_DATA_ROOT", lambda path: Path(path)
    )
    with pytest.raises(ValueError, match="separate data root"):
        _validate_data_root_24(mixed)
