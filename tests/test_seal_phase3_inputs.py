from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from scripts import seal_phase3_inputs as seal


def _write(path: Path, data: bytes | str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        path.write_text(data, encoding="utf-8")
    else:
        path.write_bytes(data)
    return path


def _json(path: Path, value: object) -> Path:
    return _write(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _npy(path: Path, value: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, value, allow_pickle=False)
    return path


def _npz(path: Path, **values: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **values)
    return path


def _relative_symlink(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(os.path.relpath(target, link.parent))


def _record(group: dict, name: str) -> dict:
    return next(row for row in group["files"] if row["logical_name"] == name)


def _effective_sha(record: dict) -> str:
    if record["file_type"] == "symlink":
        return record["resolved_target"]["content"]["sha256"]
    return record["content"]["sha256"]


def _fixture(tmp_path: Path, *, stage: str = "stage1") -> dict[str, Path | seal.SealRequest]:
    dataset_root = tmp_path / "dataset"
    split_dir = dataset_root / "Flickr25k" / "setting1"
    train_ids = [f"images/train-{index}.jpg" for index in range(10)]
    train_lines = [
        f"{image_id} {1 if index % 2 == 0 else 0} "
        f"{0 if index % 2 == 0 else 1}\n"
        for index, image_id in enumerate(train_ids)
    ]
    _write(split_dir / "train.txt", "".join(train_lines))
    _write(split_dir / "test.txt", "images/query.jpg 1 0\n")
    _write(
        split_dir / "database.txt",
        "".join(train_lines) + "images/db.jpg 1 0\n",
    )

    qwen = _write(
        tmp_path / "cache" / "flickr25k_qwen3_v4_trainset.jsonl",
        '{"image_id":"images/train-0.jpg","descriptions":{}}\n',
    )
    feature = tmp_path / "feature"
    overlay = tmp_path / "feature_foils"
    donor = tmp_path / "donor"
    feature.mkdir()
    overlay.mkdir()
    donor.mkdir()

    hf_root = tmp_path / "hf" / "models--fixture--model"
    blobs = hf_root / "blobs"
    snapshot = hf_root / "snapshots" / ("a" * 40)
    weight_blob = _write(blobs / "weight-blob", b"fixture model weight\n")
    _relative_symlink(snapshot / "pytorch_model.bin", weight_blob)
    config_blob = _write(blobs / "config-blob", '{"model_type":"clip"}\n')
    _relative_symlink(snapshot / "config.json", config_blob)
    tokenizer_hashes: dict[str, str] = {}
    for index, name in enumerate(seal._HF_TOKENIZER_FILES):
        blob = _write(blobs / f"tokenizer-{index}", f"{name}\n")
        _relative_symlink(snapshot / name, blob)
        tokenizer_hashes[str(blob)] = _sha(blob)

    meta = {
        "N": 12,
        "D_proj": 4,
        "save_aug_views": 2,
        "qwen_cache": qwen.name,
        "backbone": "fixture/model",
        "tokenizer": "fixture/model",
        "hf_provenance": {
            "checkpoint": "fixture/model",
            "model_revision": "a" * 40,
            "transformers_version": "fixture",
            "model_weight_file": str(weight_blob),
            "model_weight_sha256": _sha(weight_blob),
            "tokenizer_files_sha256": tokenizer_hashes,
        },
        "canonical_transform": {
            "resize": [224, 224],
            "interpolation": "bilinear",
            "crop": False,
            "normalization": "openai_clip",
        },
        "n_failed_image_decode": 0,
        "failed_image_indices": [],
    }
    _json(feature / "meta.json", meta)
    image_id_values = [*train_ids, "images/query.jpg", "images/db.jpg"]
    # The production feature cache uses compact JSON while foil generators use
    # pretty JSON.  The loader compares parsed lists, not raw formatting.
    image_ids = _write(
        feature / "image_ids.json",
        json.dumps(image_id_values, separators=(",", ":")) + "\n",
    )

    # Mirror the real cache: large base arrays resolve through symlinks to a
    # donor cache, while token text tensors are regular files in the feature
    # cache.  Every byte payload is tiny here.
    donor_backed = (
        "visual_tokens.f16.npy",
        "visual_global.f16.npy",
        "visual_tokens_aug0.f16.npy",
        "visual_global_aug0.f16.npy",
        "visual_tokens_aug1.f16.npy",
        "visual_global_aug1.f16.npy",
    )
    for index, name in enumerate(donor_backed):
        target = _write(donor / name, (f"donor-{index}-{name}\n").encode())
        _relative_symlink(feature / name, target)
    text_part_target = _npy(
        donor / "text_part.f16.npy",
        np.arange(12 * 6 * 4, dtype=np.float16).reshape(12, 6, 4),
    )
    has_text_target = _npy(
        donor / "has_text.bool.npy", np.ones(12, dtype=np.bool_)
    )
    _relative_symlink(feature / "text_part.f16.npy", text_part_target)
    _relative_symlink(feature / "has_text.bool.npy", has_text_target)
    _write(feature / "text_tokens.f16.npy", b"factual-token-features\n")
    _write(feature / "text_token_mask.bool.npy", b"factual-token-mask\n")
    from val_split import carve_val_indices

    labels = np.asarray(
        [[1, 0] if index % 2 == 0 else [0, 1] for index in range(10)],
        dtype=np.int64,
    )
    opt_idx, _, _ = carve_val_indices(labels, ratio=0.1, seed=42)
    opt_rows = _npy(feature / "opt_train_rows.npy", opt_idx.astype(np.int64))
    train_rows = _npy(
        feature / "train_all_rows.npy", np.arange(10, dtype=np.int64)
    )

    loader_names = (
        *seal._CACHE_CORE,
        *seal._FACTUAL_TOKEN_PAIR,
        "visual_tokens_aug0.f16.npy",
        "visual_global_aug0.f16.npy",
        "visual_tokens_aug1.f16.npy",
        "visual_global_aug1.f16.npy",
    )
    for name in loader_names:
        _relative_symlink(overlay / name, feature / name)
    _relative_symlink(overlay / "opt_train_rows.npy", opt_rows)
    _relative_symlink(overlay / "train_all_rows.npy", train_rows)

    _write(overlay / "text_foil_part.f16.npy", b"pooled-foil-features\n")
    _write(overlay / "text_foil_valid.bool.npy", b"pooled-foil-mask\n")
    _write(overlay / "text_foil_tokens.f16.npy", b"token-foil-features\n")
    _write(overlay / "text_foil_token_mask.bool.npy", b"token-foil-mask\n")
    # Same ordered IDs, deliberately different raw formatting.
    _json(overlay / "text_foil_image_ids.json", image_id_values)
    _json(overlay / "text_foil_token_image_ids.json", image_id_values)
    _write(overlay / "text_foil_edits.jsonl", b'{"row":0,"edits":{}}\n')

    image_ids_sha = _sha(image_ids)
    foil_source_sha = hashlib.sha256(b"fixture-foil-source").hexdigest()
    common_foil_meta = {
        "cache_dir": str(feature),
        "cache_image_ids_sha256": image_ids_sha,
        "foil_jsonl_sha256": foil_source_sha,
    }
    _json(overlay / "text_foil_meta.json", common_foil_meta)
    _json(overlay / "text_foil_token_meta.json", common_foil_meta)

    if stage == "stage1":
        split_rows = opt_rows
        whitening = _npz(
            overlay / "text_whiten_optTrain_localOnly.npz",
            mu=np.zeros(4, dtype=np.float32),
            U=np.eye(4, dtype=np.float32),
            S=np.ones(4, dtype=np.float32),
        )
    else:
        split_rows = train_rows
        whitening = _npz(
            overlay / "text_whiten_trainOnly_localOnly.npz",
            mu=np.zeros(4, dtype=np.float32),
            U=np.eye(4, dtype=np.float32),
            S=np.ones(4, dtype=np.float32),
        )
    n_kept = int(np.load(split_rows, allow_pickle=False).size)
    whitening_meta = {
        "cache_dir": str(feature),
        "leakage_free_fit": True,
        "local_slots_only": True,
        "residualize_first": False,
        "row_index_npy": str(split_rows),
        "D": 4,
        "N_kept": n_kept,
        "rows_used": n_kept * 5,
    }
    _json(Path(str(whitening) + ".meta.json"), whitening_meta)

    semantic = {
        "base_cache": str(feature),
        "dataset": "flickr",
        "foil_jsonl_sha256": foil_source_sha,
        "image_ids_sha256": image_ids_sha,
        "overlay": str(overlay),
        "qwen_jsonl_sha256": _sha(qwen),
        "whitening": {
            whitening.name: {
                "row_index": str(split_rows),
                "row_index_sha256": _sha(split_rows),
                "rows_used": n_kept * 5,
                "sha256": _sha(whitening),
            }
        },
    }
    _json(overlay / "semantic_detail_cache_manifest.json", semantic)

    request = seal.SealRequest.normalized(
        dataset="flickr25k",
        stage=stage,
        dataset_root=dataset_root,
        feature_cache=feature,
        foil_cache=overlay,
        whitening=whitening,
        qwen=qwen,
        split_rows=split_rows,
    )
    return {
        "request": request,
        "dataset_root": dataset_root,
        "split_dir": split_dir,
        "feature": feature,
        "overlay": overlay,
        "donor": donor,
        "qwen": qwen,
        "split_rows": split_rows,
        "whitening": whitening,
    }


def test_seal_verify_and_symlink_target_hash_memoization(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "phase3-inputs.json"
    sealed = seal.create_seal(fixture["request"], output)

    assert sealed["schema"] == seal.SCHEMA
    assert sealed["schema_version"] == seal.SCHEMA_VERSION
    feature = sealed["inputs"]["feature_cache"]
    overlay = sealed["inputs"]["foil_overlay"]
    base_visual = _record(feature, "visual_tokens.f16.npy")
    overlay_visual = _record(overlay, "visual_tokens.f16.npy")
    assert base_visual["file_type"] == "symlink"
    assert overlay_visual["file_type"] == "symlink"
    assert base_visual["resolved_target"]["path"] == overlay_visual["resolved_target"]["path"]
    assert _effective_sha(base_visual) == _effective_sha(overlay_visual)
    assert sealed["hash_summary"]["unique_resolved_content_objects"] < sealed["hash_summary"]["logical_file_records"]
    feature_ids = _record(feature, "image_ids.json")
    pooled_ids = _record(overlay, "text_foil_image_ids.json")
    assert _effective_sha(feature_ids) != _effective_sha(pooled_ids)
    alignment = sealed["inputs"]["image_id_alignment"]
    assert alignment["row_count"] == 12
    assert len({row["semantic_sha256"] for row in alignment["logical_files"]}) == 1
    roles = sealed["contract"]["consumption_roles"]
    assert "inputs.feature_cache" in roles["runtime_consumed"]
    assert "inputs.foil_overlay" in roles["derivation_provenance"]
    assert "inputs.hf_provenance" in roles["derivation_provenance"]
    assert sealed["inputs"]["decode_failure_audit"]["policy"] == "no_decode_failures"
    split_identity = sealed["inputs"]["split_identity"]
    assert split_identity["counts"] == {
        "designated_train": 10,
        "optimization_train": 8,
        "heldout_train_validation": 2,
    }
    whitening = sealed["inputs"]["whitening"]["validation"]
    assert whitening["fit_geometry"]["source_local_slots"] == [1, 2, 3, 4, 5]
    assert whitening["effective_runtime_preprocessing"][
        "runtime_local_slots_whitened"
    ] == [1, 2, 3, 4]
    assert sealed["inputs"]["hf_provenance"]["snapshot_contract"][
        "tokenizer_file_set"
    ] == list(seal._HF_TOKENIZER_FILES)
    assert sealed["inputs"]["hf_provenance"]["snapshot_contract"][
        "config_file"
    ] == "config.json"

    verified = seal.verify_seal(output)
    assert verified["aggregate_digest"] == sealed["aggregate_digest"]
    fast = seal.verify_seal_stats(
        output,
        expected_aggregate_sha256=sealed["aggregate_digest"]["sha256"],
    )
    assert fast["aggregate_digest"] == sealed["aggregate_digest"]

    authority = seal.verify_seal_authority(output)
    assert authority["dataset"] == "flickr25k"
    assert authority["stage"] == "stage1"
    assert authority["hf_runtime"]["snapshot_dir"] == str(snapshot := (
        tmp_path / "hf" / "models--fixture--model" / "snapshots" / ("a" * 40)
    ))
    assert authority["hf_runtime"]["config_sha256"] == _sha(snapshot / "config.json")
    assert seal.verify_seal_authority(
        output, expected=authority, full=False
    ) == authority


def test_runtime_authority_rejects_path_and_split_row_drift(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "phase3-inputs.json"
    seal.create_seal(fixture["request"], output)
    authority = seal.verify_seal_authority(output)
    request = fixture["request"]
    seal.assert_runtime_paths(
        authority,
        dataset="flickr25k", stage="stage1",
        dataset_root=request.dataset_root,
        feature_cache=request.feature_cache,
        qwen=request.qwen, whitening=request.whitening,
    )
    from val_split import carve_val_indices
    labels = np.asarray(
        [[1, 0] if index % 2 == 0 else [0, 1] for index in range(10)],
        dtype=np.int64,
    )
    opt_idx, val_idx, _ = carve_val_indices(labels, ratio=0.1, seed=42)
    seal.assert_runtime_split_rows(
        authority,
        dataset_to_cache_rows=np.arange(10, dtype=np.int64),
        optimization_indices=opt_idx,
        validation_indices=val_idx,
    )
    # A different logical alias was never sealed, even when it reaches the
    # exact same directory.  Realpath-only comparison used to admit this.
    feature_alias = tmp_path / "feature-alias"
    feature_alias.symlink_to(request.feature_cache, target_is_directory=True)
    with pytest.raises(seal.SealError, match="runtime inputs differ"):
        seal.assert_runtime_paths(
            authority,
            dataset="flickr25k", stage="stage1",
            dataset_root=request.dataset_root,
            feature_cache=str(feature_alias),
            qwen=request.qwen, whitening=request.whitening,
        )
    with pytest.raises(seal.SealError, match="runtime inputs differ"):
        seal.assert_runtime_paths(
            authority,
            dataset="flickr25k", stage="stage1",
            dataset_root=request.dataset_root,
            feature_cache=str(tmp_path / "other-cache"),
            qwen=request.qwen, whitening=request.whitening,
        )
    shifted = np.arange(10, dtype=np.int64) + 1
    with pytest.raises(seal.SealError, match="split/cache rows differ"):
        seal.assert_runtime_split_rows(
            authority,
            dataset_to_cache_rows=shifted,
            optimization_indices=opt_idx,
            validation_indices=val_idx,
        )
    # Preserve every sorted membership set while changing which dataset
    # example points at which cache row.  Only the ordered mapping digest can
    # distinguish this from the honest vector.
    permuted = np.arange(10, dtype=np.int64)
    first, second = map(int, opt_idx[:2])
    permuted[first], permuted[second] = permuted[second], permuted[first]
    with pytest.raises(seal.SealError, match="dataset_to_cache_rows_sha256"):
        seal.assert_runtime_split_rows(
            authority,
            dataset_to_cache_rows=permuted,
            optimization_indices=opt_idx,
            validation_indices=val_idx,
        )

    # Exact pathname equality is not enough if the named inode/link changes
    # after admission.  Replace the sealed regular Qwen file with a symlink to
    # identical bytes and require the lstat/link identity check to refuse it.
    qwen = Path(request.qwen)
    qwen_copy = tmp_path / "qwen-identical-copy.jsonl"
    qwen_copy.write_bytes(qwen.read_bytes())
    qwen.unlink()
    qwen.symlink_to(qwen_copy)
    with pytest.raises(seal.SealError, match="stat drifted|symlink"):
        seal.assert_runtime_paths(
            authority,
            dataset="flickr25k", stage="stage1",
            dataset_root=request.dataset_root,
            feature_cache=request.feature_cache,
            qwen=request.qwen, whitening=request.whitening,
        )


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("aug_half_pair", "orphan"),
        ("aug_gap", "gap/orphan"),
        ("foil_half_pair", "missing runtime foil"),
    ],
)
def test_missing_gap_and_orphan_groups_are_refused(
    tmp_path: Path, mutation: str, message: str
) -> None:
    fixture = _fixture(tmp_path)
    overlay = fixture["overlay"]
    if mutation == "aug_half_pair":
        (overlay / "visual_global_aug1.f16.npy").unlink()
    elif mutation == "aug_gap":
        _write(overlay / "visual_tokens_aug2.f16.npy", b"unpaired-extra-view")
    else:
        (overlay / "text_foil_token_mask.bool.npy").unlink()
    with pytest.raises(seal.SealError, match=message):
        seal.build_seal(fixture["request"])


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"not-json\n", "invalid UTF-8 JSON"),
        (json.dumps(["images/train.jpg", 7]), "non-empty strings"),
        (
            json.dumps(["images/train.jpg", "images/train.jpg"]),
            "duplicate image identifiers",
        ),
        (json.dumps(["", "images/db.jpg"]), "non-empty strings"),
    ],
)
def test_image_id_preflight_refuses_malformed_nonstring_empty_and_duplicate(
    tmp_path: Path, payload: bytes | str, message: str
) -> None:
    fixture = _fixture(tmp_path)
    _write(fixture["overlay"] / "text_foil_image_ids.json", payload)
    with pytest.raises(seal.SealError, match=message):
        seal.build_seal(fixture["request"])


def test_invalid_image_order_refuses_before_any_content_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    _json(
        fixture["overlay"] / "text_foil_image_ids.json",
        ["images/db.jpg", "images/train.jpg"],
    )

    def forbidden_digest(*args, **kwargs):  # pragma: no cover - failure path
        raise AssertionError("content hashing began before image-id preflight")

    monkeypatch.setattr(seal._HashMemo, "digest", forbidden_digest)
    with pytest.raises(seal.SealError, match="exactly match.*element/order"):
        seal.build_seal(fixture["request"])


def test_stage_specific_whitening_and_row_index_are_refused_on_swap(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, stage="stage1")
    bad = seal.SealRequest.normalized(
        dataset="flickr25k",
        stage="stage1",
        dataset_root=fixture["dataset_root"],
        feature_cache=fixture["feature"],
        foil_cache=fixture["overlay"],
        whitening=fixture["whitening"],
        qwen=fixture["qwen"],
        split_rows=fixture["feature"] / "train_all_rows.npy",
    )
    with pytest.raises(seal.SealError, match="requires opt_train_rows"):
        seal.build_seal(bad)


def test_consistently_rewritten_split_and_whitening_bundle_is_refused_pre_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, stage="stage1")
    wrong_rows = np.arange(1, 9, dtype=np.int64)
    _npy(fixture["split_rows"], wrong_rows)
    semantic_path = fixture["overlay"] / "semantic_detail_cache_manifest.json"
    semantic = json.loads(semantic_path.read_text(encoding="utf-8"))
    entry = semantic["whitening"][Path(fixture["whitening"]).name]
    entry["row_index_sha256"] = _sha(fixture["split_rows"])
    _json(semantic_path, semantic)

    def forbidden_digest(*args, **kwargs):  # pragma: no cover - failure path
        raise AssertionError("large hashing began before split identity refusal")

    monkeypatch.setattr(seal._HashMemo, "digest", forbidden_digest)
    with pytest.raises(seal.SealError, match="exact ordered canonical"):
        seal.build_seal(fixture["request"])


@pytest.mark.parametrize("mutation", ("missing_key", "nonorthonormal", "wrong_N_kept"))
def test_malformed_or_inconsistent_whitening_is_refused(
    tmp_path: Path, mutation: str
) -> None:
    fixture = _fixture(tmp_path)
    path = fixture["whitening"]
    if mutation == "missing_key":
        _npz(
            path,
            mu=np.zeros(4, dtype=np.float32),
            U=np.eye(4, dtype=np.float32),
        )
        message = "exact keys"
    elif mutation == "nonorthonormal":
        _npz(
            path,
            mu=np.zeros(4, dtype=np.float32),
            U=np.eye(4, dtype=np.float32) * 2,
            S=np.ones(4, dtype=np.float32),
        )
        message = "not orthonormal"
    else:
        meta_path = Path(str(path) + ".meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["N_kept"] += 1
        _json(meta_path, meta)
        message = "N_kept"
    with pytest.raises(seal.SealError, match=message):
        seal.build_seal(fixture["request"])


def test_hf_tokenizer_exact_set_and_bytes_are_fail_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    meta_path = fixture["feature"] / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    tokenizers = meta["hf_provenance"]["tokenizer_files_sha256"]
    tokenizers.pop(next(iter(tokenizers)))
    _json(meta_path, meta)
    with pytest.raises(seal.SealError, match="omits immutable snapshot file"):
        seal.build_seal(fixture["request"])


def test_invalid_decode_failure_index_is_refused(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    meta_path = fixture["feature"] / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["n_failed_image_decode"] = 1
    meta["failed_image_indices"] = [12]
    _json(meta_path, meta)
    with pytest.raises(seal.SealError, match="within image_ids bounds"):
        seal.build_seal(fixture["request"])


def test_mscoco_database_only_decode_failures_are_kept_and_disclosed(
    tmp_path: Path,
) -> None:
    split = tmp_path / "dataset" / "MSCOCO" / "setting1"
    _write(split / "train.txt", "images/train.jpg 1 0\n")
    _write(split / "test.txt", "images/query.jpg 0 1\n")
    _write(
        split / "database.txt",
        "images/train.jpg 1 0\nimages/db-a.jpg 1 0\nimages/db-b.jpg 0 1\n",
    )
    request = seal.SealRequest.normalized(
        dataset="mscoco",
        stage="stage1",
        dataset_root=tmp_path / "dataset",
        feature_cache=tmp_path / "feature",
        foil_cache=tmp_path / "foil",
        whitening=tmp_path / "whiten.npz",
        qwen=tmp_path / "qwen.jsonl",
        split_rows=tmp_path / "opt_train_rows.npy",
    )
    evidence = seal._decode_failure_audit_preflight(
        request,
        feature_meta={
            "n_failed_image_decode": 2,
            "failed_image_indices": [2, 3],
        },
        feature_image_ids=[
            "images/train.jpg", "images/query.jpg",
            "images/db-a.jpg", "images/db-b.jpg",
        ],
    )
    assert evidence["policy"] == "kept_and_disclosed_db_only"
    assert evidence["official_membership"] == {
        "train_count": 0,
        "query_count": 0,
        "database_count": 2,
        "database_only_count": 2,
        "unassigned_count": 0,
    }


def test_refit_requires_query_and_database_row_files(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, stage="refit")
    (fixture["split_dir"] / "test.txt").unlink()
    with pytest.raises(seal.SealError, match="No such file|required input is missing"):
        seal.build_seal(fixture["request"])


def test_refit_binds_exact_all_train_rows_and_runtime_dataset_splits(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, stage="refit")
    payload = seal.build_seal(fixture["request"])
    split = payload["inputs"]["split_identity"]
    assert split["counts"] == {
        "designated_train": 10,
        "optimization_train": 10,
        "heldout_train_validation": 0,
    }
    assert split["strategy"] == "all designated train rows (refit; no validation carve)"
    dataset_roles = {
        record["logical_name"]: record["consumption_role"]
        for record in payload["inputs"]["dataset_rows"]["files"]
    }
    assert dataset_roles == {
        "train_rows": "runtime_consumed",
        "query_rows": "runtime_consumed",
        "database_rows": "runtime_consumed",
    }


def test_exclusive_publish_never_overwrites_an_existing_seal(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "phase3-inputs.json"
    seal.create_seal(fixture["request"], output)
    before = output.read_bytes()
    with pytest.raises(seal.SealError, match="refusing to overwrite"):
        seal.create_seal(fixture["request"], output)
    assert output.read_bytes() == before


def test_full_verify_refuses_content_drift(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "phase3-inputs.json"
    seal.create_seal(fixture["request"], output)
    _write(fixture["qwen"], b'{"changed":true}\n')
    with pytest.raises(seal.SealError, match="drifted|does not match"):
        seal.verify_seal(output)


def test_fast_verify_refuses_stat_only_drift_without_rehash(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "phase3-inputs.json"
    sealed = seal.create_seal(fixture["request"], output)
    split_rows = fixture["split_rows"]
    current = split_rows.stat()
    os.utime(
        split_rows,
        ns=(current.st_atime_ns, current.st_mtime_ns + 1_000_000),
    )
    with pytest.raises(seal.SealError, match="stat drifted|target stat drifted"):
        seal.verify_seal_stats(
            output,
            expected_aggregate_sha256=sealed["aggregate_digest"]["sha256"],
        )


def test_fast_verify_refuses_a_new_consumed_orphan(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "phase3-inputs.json"
    sealed = seal.create_seal(fixture["request"], output)
    _write(fixture["feature"] / "visual_tokens_aug2.f16.npy", b"new-view")
    with pytest.raises(seal.SealError, match="inventory drifted"):
        seal.verify_seal_stats(
            output,
            expected_aggregate_sha256=sealed["aggregate_digest"]["sha256"],
        )


def test_fast_verify_never_invokes_the_content_hasher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "phase3-inputs.json"
    sealed = seal.create_seal(fixture["request"], output)

    def forbidden_digest(*args, **kwargs):  # pragma: no cover - failure path
        raise AssertionError("fast stat verification attempted to hash content")

    monkeypatch.setattr(seal._HashMemo, "digest", forbidden_digest)
    seal.verify_seal_stats(
        output,
        expected_aggregate_sha256=sealed["aggregate_digest"]["sha256"],
    )


def test_fast_verify_refuses_symlink_retarget_even_with_equal_bytes(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "phase3-inputs.json"
    sealed = seal.create_seal(fixture["request"], output)
    link = fixture["overlay"] / "visual_tokens.f16.npy"
    original_target = link.resolve(strict=True)
    replacement = _write(tmp_path / "replacement.npy", original_target.read_bytes())
    link.unlink()
    _relative_symlink(link, replacement)
    with pytest.raises(seal.SealError, match="symlink/target stat drifted"):
        seal.verify_seal_stats(
            output,
            expected_aggregate_sha256=sealed["aggregate_digest"]["sha256"],
        )


def test_verify_rejects_manifest_tampering_before_reading_inputs(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "phase3-inputs.json"
    seal.create_seal(fixture["request"], output)
    os.chmod(output, 0o644)
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["request"]["dataset"] = "mscoco"
    output.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(seal.SealError, match="aggregate digest"):
        seal.verify_seal(output)


def test_cli_seal_full_verify_and_fast_verify(tmp_path: Path, capsys) -> None:
    fixture = _fixture(tmp_path)
    request = fixture["request"]
    output = tmp_path / "cli-seal.json"
    assert seal.main([
        "seal",
        "--dataset", request.dataset,
        "--stage", request.stage,
        "--dataset-root", request.dataset_root,
        "--feature-cache", request.feature_cache,
        "--foil-cache", request.foil_cache,
        "--whitening", request.whitening,
        "--qwen", request.qwen,
        "--split-rows", request.split_rows,
        "--output", str(output),
    ]) == 0
    created = json.loads(output.read_text(encoding="utf-8"))
    digest = created["aggregate_digest"]["sha256"]
    assert seal.main(["verify", "--seal", str(output)]) == 0
    assert seal.main([
        "verify-stats",
        "--seal", str(output),
        "--expected-aggregate-sha256", digest,
    ]) == 0
    stdout = capsys.readouterr().out
    assert "sealed " in stdout
    assert "verified " in stdout
    assert "verified-stats " in stdout
