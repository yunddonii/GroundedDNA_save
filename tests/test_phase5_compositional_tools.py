from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

import compositional_eval as ce
import interpret_compositional_code as icc
from dna_utils.extraction_validation import base_indices_to_2bit
from dna_utils.runtime_state import ResolvedEpoch, sha256_file, write_extraction_manifest


def _json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _qwen_row(image_id: str, image_path: Path, *, version: str = "v4") -> dict:
    return {
        "image_id": image_id,
        "image_path": str(image_path),
        "codebook_texts": {
            key: f"{key} text for {image_id}" for key in icc.QWEN_SLOT_KEYS
        },
        "prompt_version": version,
        "vlm": "Qwen/Qwen3-VL-8B-Instruct",
    }


def _write_qwen(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _write_extraction_npz(
    path: Path, *, rows: int, M: int, L: int, K: int,
    image_paths: list[str] | None,
) -> None:
    base = (np.arange(rows * M * L, dtype=np.int64) % 4).reshape(rows, M * L)
    codebook = (np.arange(rows * M, dtype=np.int64) % K).reshape(rows, M)
    payload = {
        "base_indices": base,
        "hash_2bit": base_indices_to_2bit(base),
        "codebook_indices": codebook,
        "multi_hot_labels": np.eye(2, dtype=np.int64)[np.arange(rows) % 2],
    }
    if image_paths is not None:
        payload["image_paths"] = np.asarray(image_paths, dtype=np.str_)
    np.savez(path, **payload)


def _fixture(tmp_path: Path, *, backfilled: bool = False) -> SimpleNamespace:
    M, L, K, D = 5, 3, 8, 4
    dataset_root = tmp_path / "dataset" / "Flickr25k"
    ids = [f"images/group-{index % 2}/image-{index}.jpg" for index in range(5)]
    full_paths = []
    for index, image_id in enumerate(ids):
        image_path = dataset_root / image_id
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), (index * 20, 0, 0)).save(image_path)
        full_paths.append(str(image_path))

    cache = tmp_path / "cache" / "flickr"
    cache.mkdir(parents=True)
    _json(cache / "image_ids.json", ids)
    _json(cache / "meta.json", {"N": len(ids), "D_proj": D})
    np.save(
        cache / "text_part.f16.npy",
        np.arange(len(ids) * 6 * D, dtype=np.float16).reshape(len(ids), 6, D),
    )
    np.save(
        cache / "visual_global.f16.npy",
        np.arange(len(ids) * D, dtype=np.float16).reshape(len(ids), D),
    )
    np.save(cache / "has_text.bool.npy", np.ones(len(ids), dtype=np.bool_))

    # Official Flickr/MSCOCO files are exact train-ID sets whose JSONL order
    # differs from dataset order after shard concatenation.
    train_indices = (0, 2, 3)
    qwen = _write_qwen(
        tmp_path / "cache" / "qwen.jsonl",
        [_qwen_row(ids[index], Path(full_paths[index])) for index in (3, 0, 2)],
    )
    run = tmp_path / "run"
    run.mkdir()
    checkpoint = run / "model_state_dict.pth"
    checkpoint.write_bytes(b"fixture-checkpoint")
    config = {
        "num_semantic_parts": M,
        "num_codons_per_codebook": L,
        "codebook_size": K,
        "siglip2_feature_cache_dir": str(cache),
        "qwen_text_cache_path": str(qwen),
        "dataset_dir": str((tmp_path / "dataset").resolve()),
        "dataset": "Flickr25k",
    }
    torch.save(config, run / "config.pt")

    split_specs = {
        "db": (4, full_paths[:4]),
        "query": (1, full_paths[4:]),
        "train": (len(train_indices), [full_paths[index] for index in train_indices]),
    }
    manifests: dict[str, Path] = {}
    resolved = ResolvedEpoch(
        epoch=4,
        source="explicit_flag",
        effective_sinkhorn_epsilon=0.1,
        sinkhorn_schedule_horizon=5,
        checkpoint_sha256=sha256_file(str(checkpoint)),
        sinkhorn_annealing_enabled=True,
    )
    for split, (rows, paths) in split_specs.items():
        npz = run / f"extract_{split}.npz"
        _write_extraction_npz(npz, rows=rows, M=M, L=L, K=K, image_paths=paths)
        manifest = run / f"extraction_manifest_{split}.json"
        write_extraction_manifest(
            str(manifest),
            checkpoint_path=str(checkpoint),
            resolved=resolved,
            num_slots=M,
            bases_per_slot=L,
            split=split,
            n_rows=rows,
            lr_schedule_horizon=60,
            training_epoch_budget=5,
            training_stop_epoch=4,
            extra={
                "config_path": str((run / "config.pt").resolve()),
                "config_sha256": sha256_file(str(run / "config.pt")),
                "backfilled": backfilled,
                "dataset": "flickr25k",
                "random_seed": 42,
                "codebook_size": K,
                "npz_path": str(npz.resolve()),
                "npz_sha256": sha256_file(str(npz)),
            },
        )
        manifests[split] = manifest
    _json(
        run / "extraction_complete.json",
        {
            "schema_version": 1,
            "splits": ["db", "query", "train"],
            "manifest_sha256": {
                split: sha256_file(str(path)) for split, path in manifests.items()
            },
        },
    )
    return SimpleNamespace(
        run=run,
        cache=cache,
        qwen=qwen,
        dataset_root=dataset_root,
        ids=ids,
        paths=full_paths,
        M=M,
        L=L,
        K=K,
        checkpoint=checkpoint,
    )


def test_bound_extraction_geometry_and_cache_manifest(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    extraction = ce._load_phase5_extraction(str(fixture.run), allow_backfilled=False)
    assert extraction["geometry"] == {"M": 5, "L": 3, "K": 8, "N": 4}
    assert extraction["paper_eligible"] is True
    cache = ce._validate_cache_binding(str(fixture.cache), extraction=extraction)
    assert cache["binding"]["geometry"] == {
        "N": 5,
        "cache_text_slots": 6,
        "extraction_model_slots": 5,
        "D": 4,
    }
    assert len(cache["binding"]["aggregate_sha256"]) == 64
    rows = ce._build_path_to_cache_row(
        extraction["image_paths"], cache["image_ids"], str(fixture.dataset_root)
    )
    assert rows.tolist() == [0, 1, 2, 3]

    train = ce._load_phase5_extraction(
        str(fixture.run), split="train", allow_backfilled=False
    )
    assert train["geometry"] == {"M": 5, "L": 3, "K": 8, "N": 3}
    assert train["split"] == "train"


def test_image_mapping_has_no_basename_fallback_and_refuses_duplicates(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    path = root / "other" / "same.jpg"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"image")
    with pytest.raises(ce.CompositionalInputError, match="missing from the cache"):
        ce._build_path_to_cache_row(
            np.asarray([str(path)]), ["images/same.jpg"], str(root)
        )
    with pytest.raises(ce.CompositionalInputError, match="contains duplicates"):
        ce._build_path_to_cache_row(
            np.asarray([str(path)]), ["other/same.jpg", "other/same.jpg"], str(root)
        )
    with pytest.raises(ce.CompositionalInputError, match="canonical image ids contain duplicates"):
        ce._build_path_to_cache_row(
            np.asarray([str(path), str(path)]), ["other/same.jpg"], str(root)
        )


def test_canonical_ids_allow_dataset_image_symlink_but_reject_traversal(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    bulk = tmp_path / "bulk-images"
    root.mkdir()
    bulk.mkdir()
    (root / "images").symlink_to(bulk, target_is_directory=True)
    logical = root / "images" / "sample.jpg"
    logical.write_bytes(b"image")
    assert ce._canonical_ids_from_image_paths(
        np.asarray([str(logical)]), str(root)
    ) == ["images/sample.jpg"]
    with pytest.raises(ce.CompositionalInputError, match="normalized"):
        ce._strict_cache_image_ids(["images/../sample.jpg"])


def test_cache_must_be_the_manifest_bound_config_cache(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    extraction = ce._load_phase5_extraction(str(fixture.run), allow_backfilled=False)
    wrong = tmp_path / "wrong-cache"
    wrong.mkdir()
    with pytest.raises(ce.CompositionalInputError, match="differs from.*config"):
        ce._validate_cache_binding(str(wrong), extraction=extraction)


def test_dataset_root_and_cache_bytes_are_provenance_bound(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    extraction = ce._load_phase5_extraction(str(fixture.run), allow_backfilled=False)
    wrong_root = tmp_path / "wrong-dataset"
    wrong_root.mkdir()
    with pytest.raises(ce.CompositionalInputError, match="dataset_root differs"):
        ce._validate_dataset_root_binding(str(wrong_root), extraction=extraction)

    cache = ce._validate_cache_binding(str(fixture.cache), extraction=extraction)
    (fixture.cache / "meta.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ce.CompositionalInputError, match="drifted"):
        ce._assert_file_evidence_current(cache["binding"]["files"]["meta"])


@pytest.mark.parametrize("target", ("checkpoint", "manifest"))
def test_extraction_checkpoint_and_manifest_sha_tampering_is_refused(
    tmp_path: Path, target: str
) -> None:
    fixture = _fixture(tmp_path)
    if target == "checkpoint":
        fixture.checkpoint.write_bytes(b"changed-checkpoint")
        match = "checkpoint"
    else:
        manifest = fixture.run / "extraction_manifest_db.json"
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        payload["n_rows"] = 99
        _json(manifest, payload)
        match = "manifest digest"
    with pytest.raises(ce.CompositionalInputError, match=match):
        ce._load_phase5_extraction(str(fixture.run), allow_backfilled=False)


def test_backfilled_inputs_are_diagnostic_only(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, backfilled=True)
    with pytest.raises(ce.CompositionalInputError, match="backfilled"):
        ce._load_phase5_extraction(str(fixture.run), allow_backfilled=False)
    extraction = ce._load_phase5_extraction(
        str(fixture.run), allow_backfilled=True
    )
    assert extraction["paper_eligible"] is False


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("missing_key", "exact V4/V5b keys"),
        ("extra_key", "exact V4/V5b keys"),
        ("blank_text", "blank/non-string"),
        ("duplicate_id", "repeats image_id"),
        ("mixed_version", "mixes prompt versions"),
        ("wrong_path", "image_path/image_id disagreement"),
    ],
)
def test_qwen_v4_v5b_schema_is_exact_and_fail_closed(
    tmp_path: Path, mutation: str, message: str
) -> None:
    fixture = _fixture(tmp_path)
    rows = [
        _qwen_row(fixture.ids[0], Path(fixture.paths[0]), version="v4"),
        _qwen_row(fixture.ids[1], Path(fixture.paths[1]), version="v4"),
    ]
    if mutation == "missing_key":
        rows[0]["codebook_texts"].pop(icc.QWEN_SLOT_KEYS[-1])
    elif mutation == "extra_key":
        rows[0]["codebook_texts"]["unexpected"] = "text"
    elif mutation == "blank_text":
        rows[0]["codebook_texts"][icc.QWEN_SLOT_KEYS[0]] = ""
    elif mutation == "duplicate_id":
        rows[1] = dict(rows[1], image_id=rows[0]["image_id"], image_path=rows[0]["image_path"])
    elif mutation == "mixed_version":
        rows[1]["prompt_version"] = "v5b"
    else:
        rows[0]["image_path"] = fixture.paths[1]
    path = _write_qwen(tmp_path / "mutated.jsonl", rows)
    with pytest.raises(ce.CompositionalInputError, match=message):
        icc._load_qwen_jsonl(str(path), dataset_root=str(fixture.dataset_root))


def test_qwen_mapping_uses_full_canonical_id_not_basename(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    paths = [root / "one" / "same.jpg", root / "two" / "same.jpg"]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"image")
    rows = [
        _qwen_row("one/same.jpg", paths[0], version="v5b"),
        _qwen_row("two/same.jpg", paths[1], version="v5b"),
    ]
    qwen_path = _write_qwen(tmp_path / "qwen.jsonl", rows)
    qwen, evidence = icc._load_qwen_jsonl(
        str(qwen_path), dataset_root=str(root)
    )
    assert set(qwen) == {"one/same.jpg", "two/same.jpg"}
    assert evidence["prompt_version"] == "v5b"


@pytest.mark.parametrize("version", ("v4", "v5b"))
def test_actual_v4_v5b_scene_schema_fixture_is_admitted(
    tmp_path: Path, version: str
) -> None:
    root = tmp_path / "dataset"
    image = root / "images" / "sample.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    path = _write_qwen(
        tmp_path / f"{version}.jsonl",
        [_qwen_row("images/sample.jpg", image, version=version)],
    )
    qwen, evidence = icc._load_qwen_jsonl(str(path), dataset_root=str(root))
    assert set(qwen["images/sample.jpg"]) == set(icc.QWEN_SLOT_KEYS)
    assert evidence["slot_keys"] == icc.QWEN_SLOT_KEYS
    assert evidence["prompt_version"] == version
    assert evidence["vlm"] == "Qwen/Qwen3-VL-8B-Instruct"


def test_train_qwen_binding_requires_exact_set_and_seals_nonidentity_order() -> None:
    texts = {key: "text" for key in icc.QWEN_SLOT_KEYS}
    qwen = {"b.jpg": texts, "a.jpg": texts, "c.jpg": texts}
    evidence = icc._bind_qwen_to_train_extraction(
        qwen, ["a.jpg", "b.jpg", "c.jpg"]
    )
    assert evidence["exact_set_equal"] is True
    assert evidence["raw_order_equal"] is False
    assert len(evidence["qwen_row_for_extraction_order_sha256"]) == 64
    with pytest.raises(ce.CompositionalInputError, match="sets differ"):
        icc._bind_qwen_to_train_extraction(qwen, ["a.jpg", "b.jpg", "d.jpg"])


def test_closest_vs_sample_counterexample_is_not_mislabeled_as_center() -> None:
    code = np.zeros((4, 1), dtype=np.int64)
    ids = ["a", "b", "c", "d"]
    expected = sorted(
        range(4), key=lambda index: __import__("hashlib").sha256(
            ids[index].encode("utf-8")
        ).hexdigest()
    )[:2]
    first = icc._sample_indices_for_codeword(code, ids, 0, 0, 2)
    second = icc._sample_indices_for_codeword(code, ids, 0, 0, 2)
    assert first.tolist() == expected
    assert np.array_equal(first, second)
    # Synthetic model distances would choose rows 0 and 1. The extractor does
    # not store those distances, and the deterministic sample is different.
    closest_by_distance = np.argsort(np.asarray([0.01, 0.02, 8.0, 9.0]))[:2]
    assert first.tolist() != closest_by_distance.tolist()
    assert "sampled" in icc._draw_codeword_grid.__doc__
    assert "closest" not in icc._draw_codeword_grid.__doc__


def test_interpret_main_uses_manifest_M_and_preserves_summary_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)

    def fake_grid(*args, **kwargs):
        Path(args[4]).write_bytes(b"grid")

    monkeypatch.setattr(icc, "_draw_codeword_grid", fake_grid)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "interpret_compositional_code.py",
            "--run_dir", str(fixture.run),
            "--split", "train",
            "--qwen_jsonl", str(fixture.qwen),
            "--dataset_root", str(fixture.dataset_root),
            "--top_n_images", "2",
            "--top_n_words", "3",
        ],
    )
    icc.main()
    summary = json.loads(
        (fixture.run / "interpret" / "train" / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert all(f"C{m}_{icc.CODEBOOK_SHORT[m]}" in summary for m in range(5))
    assert "C5_scene" not in summary
    assert summary["_evidence"]["geometry"]["M"] == 5
    assert summary["_evidence"]["geometry"]["L"] == 3
    assert summary["_evidence"]["geometry"]["N"] == 3
    assert summary["_evidence"]["analysis_split"] == "train"
    assert summary["_evidence"]["slot_keys_used"] == icc.QWEN_SLOT_KEYS[:5]
    assert summary["_evidence"]["qwen_extraction_join"]["exact_set_equal"] is True
    assert summary["_evidence"]["qwen_extraction_join"]["raw_order_equal"] is False
    assert summary["_evidence"]["representative_selection"][
        "center_nearest_claim"
    ] is False


def test_db_report_is_split_separated_and_never_joins_train_qwen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)

    def fake_grid(*args, **kwargs):
        Path(args[4]).write_bytes(b"grid")

    monkeypatch.setattr(icc, "_draw_codeword_grid", fake_grid)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "interpret_compositional_code.py",
            "--run_dir", str(fixture.run),
            "--split", "db",
            "--dataset_root", str(fixture.dataset_root),
        ],
    )
    icc.main()
    db_dir = fixture.run / "interpret" / "db"
    summary = json.loads((db_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["_evidence"]["analysis_split"] == "db"
    assert summary["_evidence"]["semantic_word_tables_produced"] is False
    assert summary["_evidence"]["qwen_binding"] is None
    assert not list(db_dir.glob("codeword_qwen_words_*"))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "interpret_compositional_code.py",
            "--run_dir", str(fixture.run),
            "--split", "db",
            "--dataset_root", str(fixture.dataset_root),
            "--qwen_jsonl", str(fixture.qwen),
        ],
    )
    with pytest.raises(ce.CompositionalInputError, match="does not consume"):
        icc.main()


def test_compositional_main_writes_bound_paper_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compositional_eval.py",
            "--result_dir", str(fixture.run),
            "--cache_dir", str(fixture.cache),
            "--dataset_root", str(fixture.dataset_root),
            "--min_cluster", "1",
            "--skip_grids",
        ],
    )
    assert ce.main() == 0
    output = json.loads(
        (fixture.run / "compositional_eval.json").read_text(encoding="utf-8")
    )
    # Legacy output fields remain, with sealed evidence added alongside them.
    for key in ("result_dir", "cache_dir", "dataset_root", "metric_b", "metric_c"):
        assert key in output
    assert output["geometry"] == {"M": 5, "L": 3, "K": 8, "N": 4}
    assert output["paper_eligibility"]["eligible"] is True
    assert output["input_binding"]["checkpoint_sha256"] == sha256_file(
        str(fixture.checkpoint)
    )
    assert output["cache_binding"]["files"]["image_ids"]["sha256"] == sha256_file(
        str(fixture.cache / "image_ids.json")
    )
