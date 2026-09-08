"""Fail-closed provenance for feature-cache image decode failures."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from baseline.cache_provenance import (
    MSCOCO_EXPECTED_FAILED_IMAGE_IDS,
    MSCOCO_EXPECTED_FAILED_IMAGE_IDS_SHA256,
    MSCOCO_EXPECTED_FAILED_IMAGE_INDICES,
    MSCOCO_EXPECTED_FAILED_INDICES_SHA256,
    audit_cache_decode_failures,
    verify_cache_decode_failure_binding,
)


def _write_splits(
        root: Path, dataset: str, *, train: list[str], query: list[str],
        database: list[str]) -> None:
    split = root / dataset / "setting1"
    split.mkdir(parents=True, exist_ok=True)
    for filename, rows in (
            ("train.txt", train), ("test.txt", query),
            ("database.txt", database)):
        (split / filename).write_text(
            "".join(f"{image_id} 1 0\n" for image_id in rows),
            encoding="utf-8")


def _write_cache(cache: Path, image_ids: list[str], *,
                 failed: list[int] | None) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    meta = {} if failed is None else {
        "n_failed_image_decode": len(failed),
        "failed_image_indices": failed,
    }
    (cache / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (cache / "image_ids.json").write_text(
        json.dumps(image_ids), encoding="utf-8")


def _identity(audit: dict[str, object], dataset_root: Path) -> dict[str, object]:
    return {
        "cache_dir": audit["cache_dir"],
        "dataset_root": str(dataset_root.resolve()),
        "cache_meta_sha256": audit["cache_meta_sha256"],
        "cache_image_ids_sha256": audit["cache_image_ids_sha256"],
        "split_sha256": audit["membership_source_sha256"],
        "cache_decode_failure_audit": audit,
    }


def test_zero_failure_non_cifar_cache_is_recomputed_and_reopen_bound(
        tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    cache = tmp_path / "cache"
    image_ids = ["images/train.jpg", "images/query.jpg", "images/db.jpg"]
    _write_splits(
        dataset_root, "Flickr25k", train=image_ids[:1],
        query=image_ids[1:2], database=image_ids[2:])
    _write_cache(cache, image_ids, failed=[])

    audit = audit_cache_decode_failures(
        cache, dataset="Flickr25k", dataset_root=dataset_root)
    assert audit["policy"] == "no_decode_failures"
    assert audit["count"] == 0
    assert audit["official_membership"] == {
        "train_count": 0,
        "query_count": 0,
        "database_count": 0,
        "database_only_count": 0,
        "unassigned_count": 0,
    }
    assert audit["official_train_query_overlap_count"] == 0
    assert verify_cache_decode_failure_binding(
        _identity(audit, dataset_root), dataset="Flickr25k") == audit


def test_non_cifar_missing_failure_metadata_and_any_nonzero_set_are_rejected(
        tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    image_ids = ["images/train.jpg", "images/query.jpg", "images/db.jpg"]
    _write_splits(
        dataset_root, "NUSWIDE", train=image_ids[:1],
        query=image_ids[1:2], database=image_ids[2:])

    missing = tmp_path / "missing"
    _write_cache(missing, image_ids, failed=None)
    with pytest.raises(ValueError, match="omits failed-decode provenance"):
        audit_cache_decode_failures(
            missing, dataset="NUSWIDE", dataset_root=dataset_root)

    nonzero = tmp_path / "nonzero"
    _write_cache(nonzero, image_ids, failed=[2])
    with pytest.raises(ValueError, match="zero image decode failures"):
        audit_cache_decode_failures(
            nonzero, dataset="NUSWIDE", dataset_root=dataset_root)


def test_official_train_query_overlap_is_rejected_even_with_zero_failures(
        tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    cache = tmp_path / "cache"
    image_ids = ["images/shared.jpg", "images/db.jpg"]
    _write_splits(
        dataset_root, "Flickr25k", train=image_ids[:1],
        query=image_ids[:1], database=image_ids[1:])
    _write_cache(cache, image_ids, failed=[])
    with pytest.raises(ValueError, match="train/query split overlap is nonzero"):
        audit_cache_decode_failures(
            cache, dataset="Flickr25k", dataset_root=dataset_root)


def test_mscoco_exact_database_only_16_and_tamper_cases(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    cache = tmp_path / "cache"
    size = max(MSCOCO_EXPECTED_FAILED_IMAGE_INDICES) + 1
    image_ids = [f"images/train2014/filler_{index:06d}.jpg"
                 for index in range(size)]
    for index, image_id in zip(
            MSCOCO_EXPECTED_FAILED_IMAGE_INDICES,
            MSCOCO_EXPECTED_FAILED_IMAGE_IDS):
        image_ids[index] = image_id
    train = [image_ids[0]]
    query = [image_ids[1]]
    database = list(MSCOCO_EXPECTED_FAILED_IMAGE_IDS)
    _write_splits(
        dataset_root, "MSCOCO", train=train, query=query,
        database=database)
    _write_cache(
        cache, image_ids, failed=list(MSCOCO_EXPECTED_FAILED_IMAGE_INDICES))

    audit = audit_cache_decode_failures(
        cache, dataset="MSCOCO", dataset_root=dataset_root)
    assert audit["policy"] == "kept_and_disclosed_db_only"
    assert audit["count"] == 16
    assert audit["failed_indices_sha256"] == (
        MSCOCO_EXPECTED_FAILED_INDICES_SHA256)
    assert audit["failed_image_ids_sha256"] == (
        MSCOCO_EXPECTED_FAILED_IMAGE_IDS_SHA256)
    assert audit["official_membership"] == {
        "train_count": 0,
        "query_count": 0,
        "database_count": 16,
        "database_only_count": 16,
        "unassigned_count": 0,
    }
    identity = _identity(audit, dataset_root)
    assert verify_cache_decode_failure_binding(
        identity, dataset="MSCOCO") == audit

    changed_rows = list(MSCOCO_EXPECTED_FAILED_IMAGE_INDICES)
    changed_rows[0] -= 1
    (cache / "meta.json").write_text(json.dumps({
        "n_failed_image_decode": 16,
        "failed_image_indices": changed_rows,
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="row set differs"):
        audit_cache_decode_failures(
            cache, dataset="MSCOCO", dataset_root=dataset_root)

    _write_cache(
        cache, image_ids, failed=list(MSCOCO_EXPECTED_FAILED_IMAGE_INDICES))
    _write_splits(
        dataset_root, "MSCOCO", train=train,
        query=[*query, MSCOCO_EXPECTED_FAILED_IMAGE_IDS[0]],
        database=database)
    with pytest.raises(ValueError, match="overlap designated train/query"):
        audit_cache_decode_failures(
            cache, dataset="MSCOCO", dataset_root=dataset_root)

    _write_splits(
        dataset_root, "MSCOCO", train=train, query=query,
        database=database[1:])
    with pytest.raises(ValueError, match="not exclusively official database-only"):
        audit_cache_decode_failures(
            cache, dataset="MSCOCO", dataset_root=dataset_root)


def test_reopen_rejects_declared_or_underlying_evidence_tamper(
        tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    cache = tmp_path / "cache"
    image_ids = ["images/train.jpg", "images/query.jpg", "images/db.jpg"]
    _write_splits(
        dataset_root, "Flickr25k", train=image_ids[:1],
        query=image_ids[1:2], database=image_ids[2:])
    _write_cache(cache, image_ids, failed=[])
    audit = audit_cache_decode_failures(
        cache, dataset="Flickr25k", dataset_root=dataset_root)
    identity = _identity(audit, dataset_root)

    missing = dict(identity)
    missing.pop("cache_decode_failure_audit")
    with pytest.raises(ValueError, match="audit is missing"):
        verify_cache_decode_failure_binding(missing, dataset="Flickr25k")

    forged = json.loads(json.dumps(identity))
    forged["cache_decode_failure_audit"]["policy"] = "forged"
    with pytest.raises(ValueError, match="differs from current"):
        verify_cache_decode_failure_binding(forged, dataset="Flickr25k")

    (cache / "image_ids.json").write_text(
        json.dumps([*image_ids, "images/extra.jpg"]), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from current"):
        verify_cache_decode_failure_binding(identity, dataset="Flickr25k")
