from __future__ import annotations

import hashlib
import json
import math
import os
from unittest import mock

import numpy as np
import pytest

import evaluation_siglip2 as evaluation


def _bits(base: np.ndarray) -> np.ndarray:
    table = np.asarray(((0, 0), (0, 1), (1, 0), (1, 1)), dtype=np.uint8)
    return table[np.asarray(base, dtype=np.int64)].reshape(len(base), -1)


def _ap(relevance: np.ndarray, cutoff: int | None = None) -> float:
    ranked = relevance if cutoff is None else relevance[:cutoff]
    hits = np.flatnonzero(ranked == 1) + 1.0
    if len(hits) == 0:
        return 0.0
    return float((np.arange(1, len(hits) + 1, dtype=np.float64) / hits).mean())


def _dense_reference(
        query: dict[str, np.ndarray], db: dict[str, np.ndarray], *,
        distance_mode: str, precision_at_k_list: tuple[int, ...],
        threshold: float, remove_self_match: bool,
        map_at_r: int | None,
        ) -> dict[str, object]:
    """Independent full Nq x Nd golden implementation."""
    distances = evaluation._compute_distance(
        query["base_indices"], db["base_indices"],
        query.get("hash_2bit"), db.get("hash_2bit"), distance_mode)
    relevance = evaluation._compute_relevance(
        query.get("labels"), db.get("labels"),
        query.get("multi_hot_labels"), db.get("multi_hot_labels"), threshold)
    nq, nd = distances.shape
    aps: list[float] = []
    aps_at_r: list[float] = []
    precision = {k: [] for k in precision_at_k_list}
    recall = {k: [] for k in precision_at_k_list}
    positive_sum = 0
    negative_sum = 0
    positive_count = 0
    negative_count = 0
    positive_query_means: list[float] = []
    negative_query_means: list[float] = []
    for index in range(nq):
        d = distances[index]
        r = relevance[index]
        if remove_self_match and index < nd:
            keep = np.arange(nd) != index
            d = d[keep]
            r = r[keep]
        order = np.argsort(d, kind="stable")
        ranked = r[order]
        aps.append(_ap(ranked))
        if map_at_r is not None:
            aps_at_r.append(_ap(ranked, map_at_r))
        n_relevant = int(r.sum())
        for k in precision_at_k_list:
            kk = min(k, len(r))
            top = ranked[:kk]
            precision[k].append(float(top.mean()) if kk else 0.0)
            recall[k].append(float(top.sum() / max(n_relevant, 1)))
        positive = r == 1
        negative = r == 0
        if positive.any():
            positive_query_means.append(float(d[positive].mean()))
        if negative.any():
            negative_query_means.append(float(d[negative].mean()))
        positive_sum += int(d[positive].sum(dtype=np.int64))
        negative_sum += int(d[negative].sum(dtype=np.int64))
        positive_count += int(positive.sum())
        negative_count += int(negative.sum())

    result: dict[str, object] = {
        "mAP": float(np.mean(aps)) if aps else 0.0,
        "precision_at_k": {
            k: float(np.mean(values)) for k, values in precision.items()},
        "recall_at_k": {
            k: float(np.mean(values)) for k, values in recall.items()},
        "pr_curve": {
            "k": list(precision_at_k_list),
            "precision": [float(np.mean(precision[k]))
                          for k in precision_at_k_list],
            "recall": [float(np.mean(recall[k]))
                       for k in precision_at_k_list],
        },
        "mean_positive_distance": (
            float(np.mean(positive_query_means))
            if positive_query_means else 0.0),
        "mean_negative_distance": (
            float(np.mean(negative_query_means))
            if negative_query_means else 0.0),
        "pair_weighted_mean_positive_distance": (
            float(positive_sum / positive_count) if positive_count else 0.0),
        "pair_weighted_mean_negative_distance": (
            float(negative_sum / negative_count) if negative_count else 0.0),
    }
    if map_at_r is not None:
        result.update({
            "mAP_at_R": float(np.mean(aps_at_r)) if aps_at_r else 0.0,
            "mAP_R_cutoff": int(map_at_r),
        })
    return result


def _fixture(*, multi_label: bool) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    db_base = np.asarray([
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [2, 2, 2, 2],
        [3, 3, 3, 3],
        [0, 1, 2, 3],
    ], dtype=np.int64)
    query_base = np.asarray([
        [0, 0, 0, 0],
        [0, 0, 1, 1],
        [1, 1, 1, 1],
        [2, 2, 2, 3],
        [3, 3, 3, 2],
        [0, 1, 3, 2],
        [1, 2, 3, 0],
    ], dtype=np.int64)
    query = {"base_indices": query_base, "hash_2bit": _bits(query_base)}
    db = {"base_indices": db_base, "hash_2bit": _bits(db_base)}
    if multi_label:
        db["multi_hot_labels"] = np.asarray([
            [1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 0, 0],
            [0, 0, 1, 0], [1, 0, 1, 0], [0, 0, 0, 1],
            [0, 1, 0, 1], [0, 0, 1, 1], [1, 0, 0, 1],
        ], dtype=np.uint8)
        query["multi_hot_labels"] = np.asarray([
            [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
            [0, 0, 0, 1], [0, 0, 0, 0], [1, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.uint8)
    else:
        db["labels"] = np.asarray([0, 0, 1, 2, 0, 3, 1, 2, 3])
        query["labels"] = np.asarray([0, 1, 2, 3, 99, 0, 3])
    return query, db


def _assert_close(actual: object, expected: object) -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_close(actual[key], expected[key])
    elif isinstance(expected, list):
        assert isinstance(actual, list)
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _assert_close(left, right)
    elif isinstance(expected, float):
        assert isinstance(actual, float)
        assert math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
    else:
        assert actual == expected


@pytest.mark.parametrize("distance_mode", ["base", "bit2"])
@pytest.mark.parametrize("multi_label", [False, True])
@pytest.mark.parametrize("map_at_r", [None, 1, 3, 50])
def test_query_chunks_match_dense_disjoint_golden(
        distance_mode, multi_label, map_at_r):
    query, db = _fixture(multi_label=multi_label)
    ks = (1, 2, 3, 5, 20)
    threshold = 0.2 if multi_label else 0.0
    expected = _dense_reference(
        query, db, distance_mode=distance_mode,
        precision_at_k_list=ks, threshold=threshold,
        remove_self_match=False, map_at_r=map_at_r)
    first = None
    for chunk_size in (1, 2, 3, len(query["base_indices"]), 100):
        actual = evaluation.evaluate_retrieval(
            query, db, distance_mode=distance_mode,
            precision_at_k_list=ks,
            multi_label_relevance_threshold=threshold,
            remove_self_match=False, map_at_r=map_at_r,
            query_chunk_size=chunk_size)
        _assert_close(actual, expected)
        if first is None:
            first = actual
        else:
            # Chunk boundaries do not even change the serialized float values.
            assert actual == first


@pytest.mark.parametrize("distance_mode", ["base", "bit2"])
def test_tied_distances_keep_global_database_order(distance_mode):
    query = {
        "base_indices": np.asarray([[0]], dtype=np.int64),
        "hash_2bit": _bits(np.asarray([[0]], dtype=np.int64)),
        "labels": np.asarray([1]),
    }
    db = {
        "base_indices": np.asarray([[1], [1], [1], [1]], dtype=np.int64),
        "hash_2bit": _bits(np.asarray([[1], [1], [1], [1]], dtype=np.int64)),
        "labels": np.asarray([0, 1, 1, 0]),
    }
    result = evaluation.evaluate_retrieval(
        query, db, distance_mode=distance_mode,
        precision_at_k_list=(1, 2, 3, 4),
        map_at_r=3, query_chunk_size=1)
    assert result["precision_at_k"][1] == 0.0
    assert result["mAP"] == pytest.approx(7.0 / 12.0, abs=1e-12)
    assert result["mAP_at_R"] == pytest.approx(7.0 / 12.0, abs=1e-12)


def test_zero_positive_query_contributes_zero_ap_and_recall():
    query, db = _fixture(multi_label=False)
    result = evaluation.evaluate_retrieval(
        query, db, precision_at_k_list=(1, 9),
        map_at_r=5, query_chunk_size=2)
    expected = _dense_reference(
        query, db, distance_mode="base", precision_at_k_list=(1, 9),
        threshold=0.0, remove_self_match=False, map_at_r=5)
    _assert_close(result, expected)
    assert not np.any(db["labels"] == query["labels"][4])


def test_zero_positive_multihot_query_is_allowed_and_contributes_zero_ap():
    query, db = _fixture(multi_label=True)
    result = evaluation.evaluate_retrieval(
        query, db, precision_at_k_list=(1, 9), map_at_r=5,
        query_chunk_size=2)
    expected = _dense_reference(
        query, db, distance_mode="base", precision_at_k_list=(1, 9),
        threshold=0.0, remove_self_match=False, map_at_r=5)
    _assert_close(result, expected)
    assert query["multi_hot_labels"][4].sum() == 0


@pytest.mark.parametrize("bad", [
    np.asarray([[1.0, 0.0]], dtype=np.float32),
    np.asarray([[np.nan, 0.0]], dtype=np.float32),
    np.asarray([[2, 0]], dtype=np.int64),
])
def test_noninteger_nonfinite_or_nonbinary_multihot_is_refused(bad):
    base = np.zeros((1, 1), dtype=np.int64)
    query = {"base_indices": base, "multi_hot_labels": bad}
    db = {"base_indices": base, "multi_hot_labels": bad.copy()}
    with pytest.raises(ValueError, match="finite binary"):
        evaluation.evaluate_retrieval(query, db, precision_at_k_list=(1,))


def test_legacy_distance_means_remain_query_equal_and_pair_means_are_explicit():
    query = {
        "base_indices": np.asarray([[0], [0]], dtype=np.int64),
        "multi_hot_labels": np.asarray([[1, 0], [0, 1]], dtype=np.uint8),
    }
    db = {
        "base_indices": np.asarray([[0], [1], [2], [3]], dtype=np.int64),
        "multi_hot_labels": np.asarray(
            [[1, 0], [1, 0], [1, 0], [0, 1]], dtype=np.uint8),
    }
    result = evaluation.evaluate_retrieval(
        query, db, precision_at_k_list=(1,), query_chunk_size=1)
    legacy_positive_query_mean = ((2.0 / 3.0) + 1.0) / 2.0
    legacy_negative_query_mean = (1.0 + (2.0 / 3.0)) / 2.0
    assert result["mean_positive_distance"] == legacy_positive_query_mean
    assert result["mean_negative_distance"] == legacy_negative_query_mean
    assert result["pair_weighted_mean_positive_distance"] == 0.75
    assert result["pair_weighted_mean_negative_distance"] == 0.75
    assert (result["mean_positive_distance"]
            != result["pair_weighted_mean_positive_distance"])
    assert (result["mean_negative_distance"]
            != result["pair_weighted_mean_negative_distance"])


def test_remove_self_match_removes_candidate_from_stable_ranking():
    base = np.zeros((3, 1), dtype=np.int64)
    labels = np.asarray([0, 0, 1])
    split = {"base_indices": base, "labels": labels}
    result = evaluation.evaluate_retrieval(
        split, split, precision_at_k_list=(1, 2, 3),
        map_at_r=2, remove_self_match=True, query_chunk_size=2)
    expected = _dense_reference(
        split, split, distance_mode="base",
        precision_at_k_list=(1, 2, 3), threshold=0.0,
        remove_self_match=True, map_at_r=2)
    _assert_close(result, expected)
    assert result["mAP"] == pytest.approx(2.0 / 3.0, abs=1e-12)
    assert result["precision_at_k"][1] == pytest.approx(2.0 / 3.0, abs=1e-12)


def test_canonical_disjoint_path_keeps_every_database_candidate():
    # Equal row indices have no identity meaning for official disjoint query/db
    # splits.  The canonical default must therefore retain db row zero.
    query = {
        "base_indices": np.asarray([[0]], dtype=np.int64),
        "labels": np.asarray([7]),
    }
    db = {
        "base_indices": np.asarray([[0], [1]], dtype=np.int64),
        "labels": np.asarray([7, 8]),
    }
    canonical = evaluation.evaluate_retrieval(
        query, db, precision_at_k_list=(1, 2),
        remove_self_match=False, query_chunk_size=1)
    assert canonical["mAP"] == 1.0
    assert canonical["precision_at_k"][1] == 1.0
    _assert_close(canonical, _dense_reference(
        query, db, distance_mode="base", precision_at_k_list=(1, 2),
        threshold=0.0, remove_self_match=False, map_at_r=None))


def test_distance_and_relevance_are_called_on_query_chunks_only():
    query, db = _fixture(multi_label=True)
    original_distance = evaluation._compute_distance
    original_relevance = evaluation._compute_relevance
    with mock.patch.object(
            evaluation, "_compute_distance", wraps=original_distance) as distance_call, \
         mock.patch.object(
            evaluation, "_compute_relevance", wraps=original_relevance) as relevance_call:
        evaluation.evaluate_retrieval(
            query, db, query_chunk_size=3, precision_at_k_list=(1,))
    assert [call.args[0].shape[0] for call in distance_call.call_args_list] == [3, 3, 1]
    assert [call.args[2].shape[0] for call in relevance_call.call_args_list] == [3, 3, 1]
    assert all(call.args[1].shape[0] == len(db["base_indices"])
               for call in distance_call.call_args_list)
    assert all(call.args[3].shape[0] == len(db["base_indices"])
               for call in relevance_call.call_args_list)


@pytest.mark.parametrize("bad_chunk", [0, -1, True, 1.5])
def test_query_chunk_size_must_be_a_positive_integer(bad_chunk):
    query, db = _fixture(multi_label=False)
    with pytest.raises(ValueError, match="positive integer"):
        evaluation.evaluate_retrieval(
            query, db, query_chunk_size=bad_chunk,
            precision_at_k_list=(1,))


def test_bio_pre_projection_adds_map_at_r_without_redefining_full_map(tmp_path):
    base = np.asarray([[0, 1]], dtype=np.int64)
    np.savez(tmp_path / "extract_db.npz", base_indices=base)
    np.savez(tmp_path / "extract_query.npz", base_indices=base)

    pre = {"mAP": 0.4, "mAP_at_R": 0.3, "mAP_R_cutoff": 5}
    post = {"mAP": 0.5, "mAP_at_R": 0.45, "mAP_R_cutoff": 5}
    with mock.patch.object(
            evaluation, "evaluate_retrieval",
            side_effect=[pre, post]) as retrieval_call, \
         mock.patch.object(
            evaluation, "_apply_bio_projection", side_effect=[{}, {}]), \
         mock.patch.object(evaluation, "evaluate_code_collapse", return_value={
             "unique_code_ratio": 1.0,
             "duplicate_rate": 0.0,
         }):
        result = evaluation.evaluation(
            str(tmp_path), bio_project=True, map_at_r=5,
            precision_at_k_list=(1,), query_chunk_size=2)

    assert retrieval_call.call_count == 2
    assert retrieval_call.call_args_list[0].kwargs["map_at_r"] == 5
    stats = result["bio_stats"]
    # Existing consumers interpret this field as full-ranking mAP.
    assert stats["mAP_pre_projection"] == 0.4
    assert stats["mAP_at_R_pre_projection"] == 0.3
    assert stats["mAP_R_cutoff_pre_projection"] == 5


def _write_evaluation_fixture(path) -> None:
    db_base = np.asarray([[0, 0], [1, 1], [2, 2]], dtype=np.int64)
    query_base = np.asarray([[0, 0], [2, 2]], dtype=np.int64)
    np.savez(
        path / "extract_db.npz",
        base_indices=db_base,
        hash_2bit=_bits(db_base),
        codebook_indices=np.asarray([[0], [1], [2]], dtype=np.int64),
        labels=np.asarray([0, 1, 2], dtype=np.int64),
        image_paths=np.asarray(["db0", "db1", "db2"]),
    )
    np.savez(
        path / "extract_query.npz",
        base_indices=query_base,
        hash_2bit=_bits(query_base),
        codebook_indices=np.asarray([[0], [2]], dtype=np.int64),
        labels=np.asarray([0, 2], dtype=np.int64),
        image_paths=np.asarray(["qy0", "qy1"]),
    )


def test_evaluation_seals_consumed_npz_and_metric_policy(tmp_path):
    _write_evaluation_fixture(tmp_path)
    result = evaluation.evaluation(
        str(tmp_path), codebook_size=4, map_at_r=1000,
        dataset_name="CIFAR10", precision_at_k_list=(1, 3),
        query_chunk_size=1)
    output = tmp_path / "evaluation_siglip2_base.json"
    stored = json.loads(
        output.read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )
    assert stored == json.loads(json.dumps(result, allow_nan=False))
    assert stored["evaluation_schema_version"] == 2
    assert stored["dataset"] == "CIFAR10"
    assert stored["mAP_R_cutoff"] == 1000
    assert stored["metric_policy"] == {
        "ap_normalization": evaluation.AP_NORMALIZATION_POLICY,
        "map_at_r": 1000,
        "multi_label_relevance_threshold": 0.0,
        "query_chunk_size": 1,
        "ranking": evaluation.RANKING_POLICY,
        "relevance": evaluation.RELEVANCE_POLICY,
        "remove_self_match": False,
    }
    for split, filename in (("db", "extract_db.npz"),
                            ("query", "extract_query.npz")):
        expected = hashlib.sha256((tmp_path / filename).read_bytes()).hexdigest()
        binding = stored["input_artifacts"][split]
        assert binding["sha256"] == expected
        assert binding["size_bytes"] == (tmp_path / filename).stat().st_size


def test_dataset_name_enforces_exact_paper_cutoff(tmp_path):
    _write_evaluation_fixture(tmp_path)
    with pytest.raises(ValueError, match="requires mAP@R cutoff 5000"):
        evaluation.evaluation(
            str(tmp_path), codebook_size=4, map_at_r=1000,
            dataset_name="MSCOCO", precision_at_k_list=(1,))
    assert not (tmp_path / "evaluation_siglip2_base.json").exists()


def test_object_array_is_rejected_without_pickle(tmp_path):
    _write_evaluation_fixture(tmp_path)
    with np.load(tmp_path / "extract_db.npz", allow_pickle=False) as stored:
        payload = {key: stored[key] for key in stored.files}
    payload["image_paths"] = np.asarray(
        [object(), object(), object()], dtype=object)
    np.savez(tmp_path / "extract_db.npz", **payload)
    with pytest.raises(ValueError, match="Object arrays cannot be loaded"):
        evaluation.evaluation(
            str(tmp_path), codebook_size=4, map_at_r=1000,
            dataset_name="CIFAR10", precision_at_k_list=(1,))
    assert not (tmp_path / "evaluation_siglip2_base.json").exists()


def test_bound_loader_refuses_path_replacement_during_load(tmp_path):
    original_path = tmp_path / "extract_db.npz"
    replacement_path = tmp_path / "replacement.npz"
    np.savez(original_path, base_indices=np.asarray([[0]], dtype=np.int64))
    np.savez(replacement_path, base_indices=np.asarray([[0]], dtype=np.int64))
    original_load = np.load

    def replace_then_load(handle, *args, **kwargs):
        os.replace(replacement_path, original_path)
        return original_load(handle, *args, **kwargs)

    with mock.patch.object(evaluation.np, "load", side_effect=replace_then_load):
        with pytest.raises(
                RuntimeError,
                match=(r"extraction (?:changed|pathname was replaced)"
                       r".*being read")):
            evaluation._load_npz_bound(str(original_path))


def test_nonfinite_metric_cannot_replace_prior_valid_json(tmp_path):
    _write_evaluation_fixture(tmp_path)
    output = tmp_path / "evaluation_siglip2_base.json"
    output.write_text('{"prior":"valid"}\n', encoding="utf-8")
    bad_retrieval = {
        "mAP": float("inf"), "mAP_at_R": 0.5, "mAP_R_cutoff": 1000}
    with mock.patch.object(
            evaluation, "evaluate_retrieval", return_value=bad_retrieval), \
         mock.patch.object(
            evaluation, "evaluate_code_collapse", return_value={
                "unique_code_ratio": 1.0,
                "duplicate_rate": 0.0,
            }):
        with pytest.raises(ValueError, match="not finite"):
            evaluation.evaluation(
                str(tmp_path), codebook_size=4, map_at_r=1000,
                dataset_name="CIFAR10", precision_at_k_list=(1,))
    assert output.read_text(encoding="utf-8") == '{"prior":"valid"}\n'
    assert not list(tmp_path.glob(".evaluation_siglip2_base.json.*.tmp"))
