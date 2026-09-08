"""Compositional faithfulness evaluation for a trained GroundedDNA model.

Metrics produced (writes alongside the result dir):

B. codeword-caption semantic concentration
   For each (codebook m, codeword k), gather the DB samples that selected
   codeword k in codebook m, and measure how semantically coherent their
   per-part Qwen captions are. We use the cached SigLIP2 text_part feature
   for the same part m as a proxy for the caption: intra-cluster mean
   cosine similarity tells us whether a codeword corresponds to a
   consistent semantic concept. A random baseline (shuffled codeword
   assignment) is also reported so the absolute number is interpretable.

   Output keys:
     codeword_intra_text_sim[m][k]    -- per-codeword score
     codebook_mean_intra_text_sim[m]  -- per-codebook average (active codewords)
     random_baseline_per_codebook[m]  -- shuffled-assignment baseline
     compositional_lift_per_codebook[m] = above - baseline   -- "concentration" amount

C. top-K image visualization
   For each codebook m, take the top `viz_top_codewords` populated
   codewords; for each such (m, k), grab `viz_per_grid` sample images
   and tile them as a 3x3 grid PNG. Useful for human-reviewer evidence
   that the codeword captures a coherent visual concept.

Usage:
    python compositional_eval.py \
        --result_dir result/260518+flickr25k_setting1_v34_*/ \
        --cache_dir  cache/flickr25k_siglip2 \
        --dataset_root dataset/Flickr25k \
        --viz_top_codewords 5 --viz_per_grid 9
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import pickle
import sys
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np


class CompositionalInputError(RuntimeError):
    """Phase-5 evidence cannot be bound to its claimed inputs."""


def _sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_file_evidence(path: os.PathLike[str] | str) -> dict[str, Any]:
    logical = os.path.abspath(os.fspath(path))
    try:
        before = os.stat(logical)
        digest = _sha256_file(logical)
        after = os.stat(logical)
    except OSError as error:
        raise CompositionalInputError(
            f"cannot seal Phase-5 input {logical}: {error}"
        ) from error
    identity_before = (
        before.st_dev, before.st_ino, before.st_size,
        before.st_mtime_ns, before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev, after.st_ino, after.st_size,
        after.st_mtime_ns, after.st_ctime_ns,
    )
    if identity_before != identity_after:
        raise CompositionalInputError(
            f"Phase-5 input changed while hashing: {logical}"
        )
    return {
        "path": logical,
        "resolved_path": os.path.realpath(logical),
        "size": int(before.st_size),
        "device": int(before.st_dev),
        "inode": int(before.st_ino),
        "mtime_ns": int(before.st_mtime_ns),
        "ctime_ns": int(before.st_ctime_ns),
        "sha256": digest,
    }


def _assert_file_evidence_current(evidence: Mapping[str, Any]) -> None:
    path = str(evidence["path"])
    try:
        current = os.stat(path)
    except OSError as error:
        raise CompositionalInputError(
            f"sealed Phase-5 input disappeared: {path}: {error}"
        ) from error
    actual = {
        "resolved_path": os.path.realpath(path),
        "size": int(current.st_size),
        "device": int(current.st_dev),
        "inode": int(current.st_ino),
        "mtime_ns": int(current.st_mtime_ns),
        "ctime_ns": int(current.st_ctime_ns),
    }
    expected = {key: evidence[key] for key in actual}
    if actual != expected:
        raise CompositionalInputError(
            f"sealed Phase-5 input stat/path drifted: {path}"
        )


def _strict_json(path: str, description: str) -> tuple[Any, dict[str, Any]]:
    evidence = _stable_file_evidence(path)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CompositionalInputError(
            f"invalid UTF-8 JSON {description} {path}: {error}"
        ) from error
    if _sha256_file(path) != evidence["sha256"]:
        raise CompositionalInputError(
            f"{description} changed between hashing and parsing: {path}"
        )
    return value, evidence


# ----------------------- helpers -----------------------

def _l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)


def _intra_cluster_cos_sim(feats: np.ndarray) -> float:
    """Mean off-diagonal cosine similarity of L2-normalised rows.
    Returns 1.0 if N<2 (single-sample cluster is trivially "coherent")."""
    if feats.shape[0] < 2:
        return 1.0
    f = _l2_normalize(feats.astype(np.float32), axis=-1)
    sim = f @ f.T
    # mean of upper triangle (off diag)
    N = sim.shape[0]
    iu = np.triu_indices(N, k=1)
    return float(sim[iu].mean())


def _strict_canonical_image_id(value: Any, *, what: str) -> str:
    """Admit one portable, root-relative image identifier.

    ``pathlib.Path.as_posix()`` does not collapse ``a/../b`` and treats a
    backslash as an ordinary character on Linux.  Accepting either makes the
    supposedly exact mapping platform-dependent, so validate the wire-format
    directly.
    """
    if not isinstance(value, str) or not value:
        raise CompositionalInputError(f"{what} must be a non-empty string")
    if (
        "\x00" in value
        or "\\" in value
        or os.path.isabs(value)
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise CompositionalInputError(
            f"{what} is not a normalized root-relative image ID: {value!r}"
        )
    return value


def _strict_cache_image_ids(value: Any) -> List[str]:
    if not isinstance(value, list) or not value:
        raise CompositionalInputError("cache image_ids.json must be non-empty list[str]")
    if any(not isinstance(item, str) or not item for item in value):
        raise CompositionalInputError(
            "cache image_ids.json contains a non-string or empty identifier"
        )
    for index, item in enumerate(value):
        _strict_canonical_image_id(item, what=f"cache image ID row {index}")
    if len(set(value)) != len(value):
        raise CompositionalInputError("cache image_ids.json contains duplicates")
    return value


def _canonical_ids_from_image_paths(
    image_paths: np.ndarray, dataset_root: str
) -> List[str]:
    paths = np.asarray(image_paths)
    if paths.ndim != 1 or paths.dtype.kind not in {"U", "S"}:
        raise CompositionalInputError(
            "extract_db image_paths must be a non-pickled string [N] array"
        )
    # Keep lexical paths here. Canonical datasets intentionally place `images/`
    # behind a symlink to bulk storage; realpath-containment would reject every
    # valid Flickr/MSCOCO/NUS-WIDE row as an escape from the small metadata root.
    # Traversal is still impossible because IDs are derived only after strict
    # lexical containment under the manifest-bound dataset root.
    root = os.path.abspath(dataset_root)
    if not os.path.isdir(root):
        raise CompositionalInputError(f"dataset_root is not a directory: {root}")
    result: List[str] = []
    for index, raw_path in enumerate(paths.tolist()):
        if isinstance(raw_path, bytes):
            try:
                raw_path = raw_path.decode("utf-8")
            except UnicodeDecodeError as error:
                raise CompositionalInputError(
                    f"extract_db image path {index} is not UTF-8"
                ) from error
        if not isinstance(raw_path, str) or not raw_path:
            raise CompositionalInputError(
                f"extract_db image path {index} is empty/non-string"
            )
        full = (
            os.path.abspath(os.path.normpath(raw_path))
            if os.path.isabs(raw_path)
            else os.path.abspath(os.path.normpath(os.path.join(root, raw_path)))
        )
        try:
            inside = os.path.commonpath((root, full)) == root
        except ValueError:
            inside = False
        if not inside:
            raise CompositionalInputError(
                f"extract_db image path escapes dataset_root: {raw_path!r}"
            )
        canonical = os.path.relpath(full, root).replace(os.sep, "/")
        if canonical == ".." or canonical.startswith("../"):
            raise CompositionalInputError(
                f"cannot canonicalize extraction image path {raw_path!r}"
            )
        result.append(canonical)
    if len(set(result)) != len(result):
        raise CompositionalInputError(
            "extract_db canonical image ids contain duplicates"
        )
    return result


def _build_path_to_cache_row(
    image_paths: np.ndarray,
    cache_image_ids: List[str],
    dataset_root: str,
) -> np.ndarray:
    """Exact canonical image-ID map from extraction rows to cache rows.

    Missing rows and duplicate IDs are fatal. Basename/suffix fallbacks are
    intentionally absent because they silently bind different images that share
    a filename.
    """
    cache_ids = _strict_cache_image_ids(cache_image_ids)
    extraction_ids = _canonical_ids_from_image_paths(image_paths, dataset_root)
    id_to_row = {image_id: row for row, image_id in enumerate(cache_ids)}
    missing = [image_id for image_id in extraction_ids if image_id not in id_to_row]
    if missing:
        preview = missing[:3]
        raise CompositionalInputError(
            f"{len(missing)}/{len(extraction_ids)} extraction image IDs are "
            f"missing from the cache (first={preview})"
        )
    rows = np.asarray([id_to_row[image_id] for image_id in extraction_ids], dtype=np.int64)
    if np.unique(rows).size != len(rows):
        raise CompositionalInputError(
            "extraction rows do not map one-to-one onto cache image IDs"
        )
    return rows


def _save_image_grid(image_paths: List[str], out_path: str,
                     cols: int = 3, thumb: int = 224) -> None:
    """Tile up to len(image_paths) images into a square-ish grid PNG."""
    from PIL import Image
    n = len(image_paths)
    rows = (n + cols - 1) // cols
    canvas = Image.new("RGB", (cols * thumb, rows * thumb), color=(245, 245, 245))
    for i, p in enumerate(image_paths):
        try:
            im = Image.open(p).convert("RGB").resize((thumb, thumb))
        except Exception:
            continue
        r, c = i // cols, i % cols
        canvas.paste(im, (c * thumb, r * thumb))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path)


def _load_phase5_extraction(
    result_dir: str, *, split: str = "db", allow_backfilled: bool
) -> dict[str, Any]:
    """Validate one complete extraction transaction and load ``split`` arrays.

    A run with an additive train extraction commits all three manifests in the
    completion marker.  Validating only DB/query would reject that transaction;
    validating only train would ignore stale siblings.  Re-open the marker and
    validate its exact declared transaction, then select the requested split.
    """
    if split not in {"db", "train"}:
        raise CompositionalInputError(f"unsupported Phase-5 split {split!r}")
    result_dir = os.path.abspath(result_dir)
    marker_path = os.path.join(result_dir, "extraction_complete.json")
    marker, marker_evidence = _strict_json(marker_path, "extraction completion marker")
    if not isinstance(marker, Mapping):
        raise CompositionalInputError("extraction completion marker is not an object")
    declared_value = marker.get("splits")
    if not isinstance(declared_value, list) or any(
        not isinstance(item, str) for item in declared_value
    ):
        raise CompositionalInputError("completion marker splits must be list[str]")
    declared = tuple(declared_value)
    if len(set(declared)) != len(declared) or set(declared) not in (
        {"db", "query"}, {"db", "query", "train"},
    ):
        raise CompositionalInputError(
            f"unsupported/incomplete extraction transaction splits={list(declared)!r}"
        )
    if split not in declared:
        raise CompositionalInputError(
            f"requested split {split!r} is not sealed by the completion marker"
        )
    try:
        from dna_utils.extraction_validation import (
            ExtractionInvalid,
            metric_input_binding,
        )

        binding = metric_input_binding(
            result_dir,
            required_splits=declared,
            allow_backfilled=allow_backfilled,
        )
    except (ExtractionInvalid, KeyError, TypeError, ValueError) as error:
        raise CompositionalInputError(
            f"Phase-5 extraction is not admissible: {error}"
        ) from error

    manifest_path = os.path.join(result_dir, f"extraction_manifest_{split}.json")
    manifest, manifest_evidence = _strict_json(
        manifest_path, f"{split} extraction manifest"
    )
    if not isinstance(manifest, Mapping):
        raise CompositionalInputError(f"{split} extraction manifest is not an object")
    if manifest_evidence["sha256"] != binding["manifest_sha256"][split]:
        raise CompositionalInputError(
            f"{split} extraction manifest changed after validator admission"
        )
    db_path = os.path.abspath(str(manifest.get("npz_path") or ""))
    expected_db = os.path.abspath(os.path.join(result_dir, f"extract_{split}.npz"))
    if db_path != expected_db:
        raise CompositionalInputError(
            f"{split} manifest names {db_path}, not result split NPZ {expected_db}"
        )
    db_evidence = _stable_file_evidence(db_path)
    if db_evidence["sha256"] != binding["npz_sha256"][split]:
        raise CompositionalInputError(
            f"extract_{split}.npz changed after extraction validator admission"
        )
    try:
        with np.load(db_path, allow_pickle=False) as stored:
            if "image_paths" not in stored:
                raise CompositionalInputError(
                    f"extract_{split}.npz has no canonical image_paths; basename/order "
                    "inference is not paper-eligible"
                )
            codebook_indices = np.asarray(stored["codebook_indices"])
            base_indices = np.asarray(stored["base_indices"])
            image_paths = np.asarray(stored["image_paths"])
    except CompositionalInputError:
        raise
    except (OSError, ValueError) as error:
        raise CompositionalInputError(
            f"extract_{split}.npz is not safe/readable without pickle: {error}"
        ) from error

    M = manifest.get("num_slots")
    L = manifest.get("bases_per_slot")
    K = manifest.get("codebook_size")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
           for value in (M, L, K)):
        raise CompositionalInputError(
            f"invalid manifest geometry M/L/K={M!r}/{L!r}/{K!r}"
        )
    n_rows = int(manifest["n_rows"])
    if codebook_indices.shape != (n_rows, M):
        raise CompositionalInputError(
            f"codebook_indices shape {codebook_indices.shape} != {(n_rows, M)}"
        )
    if base_indices.shape != (n_rows, M * L):
        raise CompositionalInputError(
            f"base_indices shape {base_indices.shape} != {(n_rows, M * L)}"
        )
    if image_paths.shape != (n_rows,) or image_paths.dtype.kind not in {"U", "S"}:
        raise CompositionalInputError(
            "extract_db image_paths must be a non-pickled string [N] array"
        )

    config_path = str(manifest["config_path"])
    config_evidence = _stable_file_evidence(config_path)
    if config_evidence["sha256"] != binding["config_sha256"]:
        raise CompositionalInputError(
            "manifest-bound config changed after extraction validator admission"
        )
    checkpoint_path = str(manifest["checkpoint_path"])
    checkpoint_evidence = _stable_file_evidence(checkpoint_path)
    if checkpoint_evidence["sha256"] != binding["checkpoint_sha256"]:
        raise CompositionalInputError(
            "manifest-bound checkpoint changed after extraction validator admission"
        )
    try:
        import torch

        config = torch.load(config_path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError, pickle.UnpicklingError) as error:
        raise CompositionalInputError(
            f"cannot safely read manifest-bound config {config_path}: {error}"
        ) from error
    if not isinstance(config, Mapping):
        raise CompositionalInputError("manifest-bound config.pt is not a mapping")
    for key, expected in (
        ("num_semantic_parts", M),
        ("num_codons_per_codebook", L),
        ("codebook_size", K),
    ):
        if config.get(key) != expected:
            raise CompositionalInputError(
                f"config {key}={config.get(key)!r} differs from extraction "
                f"manifest value {expected!r}"
            )

    return {
        "input_binding": binding,
        "manifest": dict(manifest),
        "config": dict(config),
        "codebook_indices": codebook_indices,
        "base_indices": base_indices,
        "image_paths": image_paths,
        "geometry": {"M": M, "L": L, "K": K, "N": n_rows},
        "split": split,
        "paper_eligible": not bool(binding["backfilled_inputs"]),
        "db_evidence": db_evidence,
        "manifest_evidence": manifest_evidence,
        "transaction_evidence": {
            "completion_marker": marker_evidence,
            "selected_manifest": manifest_evidence,
            "selected_npz": db_evidence,
            "checkpoint": checkpoint_evidence,
            "config": config_evidence,
        },
    }


def _resolve_config_path(value: str) -> str:
    if os.path.isabs(value):
        return os.path.realpath(value)
    # Training launchers are repository-root relative. Binding relative paths
    # to the tool's source root avoids dependence on the caller's current cwd.
    return os.path.realpath(os.path.join(os.path.dirname(__file__), value))


def _validate_dataset_root_binding(
    dataset_root: str, *, extraction: Mapping[str, Any]
) -> str:
    """Bind canonical image IDs to the dataset root recorded by config.pt."""
    config = extraction["config"]
    dataset_dir = config.get("dataset_dir")
    dataset_name = config.get("dataset")
    if not isinstance(dataset_dir, str) or not dataset_dir:
        raise CompositionalInputError("manifest-bound config names no dataset_dir")
    if not isinstance(dataset_name, str) or not dataset_name:
        raise CompositionalInputError("manifest-bound config names no dataset")
    configured_root = _resolve_config_path(os.path.join(dataset_dir, dataset_name))
    explicit_root = os.path.realpath(os.path.abspath(dataset_root))
    if configured_root != explicit_root:
        raise CompositionalInputError(
            "explicit dataset_root differs from the manifest-bound config: "
            f"{explicit_root!r} != {configured_root!r}"
        )
    if not os.path.isdir(explicit_root):
        raise CompositionalInputError(
            f"manifest-bound dataset_root is not a directory: {explicit_root}"
        )
    return os.path.abspath(dataset_root)


def _validate_cache_binding(
    cache_dir: str,
    *,
    extraction: Mapping[str, Any],
) -> dict[str, Any]:
    explicit_cache = os.path.realpath(os.path.abspath(cache_dir))
    configured = extraction["config"].get("siglip2_feature_cache_dir")
    if not isinstance(configured, str) or not configured:
        raise CompositionalInputError(
            "manifest-bound config names no siglip2_feature_cache_dir"
        )
    configured_cache = _resolve_config_path(configured)
    if configured_cache != explicit_cache:
        raise CompositionalInputError(
            "explicit cache_dir differs from the manifest-bound config: "
            f"{explicit_cache!r} != {configured_cache!r}"
        )
    if not os.path.isdir(explicit_cache):
        raise CompositionalInputError(f"cache_dir is not a directory: {explicit_cache}")

    required = {
        "meta": "meta.json",
        "image_ids": "image_ids.json",
        "text_part": "text_part.f16.npy",
        "visual_global": "visual_global.f16.npy",
        "has_text": "has_text.bool.npy",
    }
    files = {
        role: _stable_file_evidence(os.path.join(explicit_cache, name))
        for role, name in required.items()
    }
    meta, meta_evidence = _strict_json(
        os.path.join(explicit_cache, "meta.json"), "feature-cache metadata"
    )
    image_ids_value, image_ids_evidence = _strict_json(
        os.path.join(explicit_cache, "image_ids.json"), "feature-cache image IDs"
    )
    if meta_evidence["sha256"] != files["meta"]["sha256"]:
        raise CompositionalInputError("cache meta changed during binding")
    if image_ids_evidence["sha256"] != files["image_ids"]["sha256"]:
        raise CompositionalInputError("cache image IDs changed during binding")
    if not isinstance(meta, Mapping):
        raise CompositionalInputError("feature-cache meta.json is not an object")
    image_ids = _strict_cache_image_ids(image_ids_value)
    n_cache = len(image_ids)
    M = int(extraction["geometry"]["M"])
    try:
        text_part = np.load(files["text_part"]["path"], mmap_mode="r", allow_pickle=False)
        visual_global = np.load(
            files["visual_global"]["path"], mmap_mode="r", allow_pickle=False
        )
        has_text = np.load(files["has_text"]["path"], mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as error:
        raise CompositionalInputError(f"invalid cache NPY input: {error}") from error
    if text_part.ndim != 3 or text_part.shape[0] != n_cache or text_part.shape[1] < M:
        raise CompositionalInputError(
            f"text_part geometry {text_part.shape} cannot serve N={n_cache}, M={M}"
        )
    if text_part.dtype != np.float16:
        raise CompositionalInputError(
            f"text_part dtype {text_part.dtype} is not canonical float16"
        )
    D = int(text_part.shape[2])
    if visual_global.shape != (n_cache, D) or visual_global.dtype != np.float16:
        raise CompositionalInputError(
            f"visual_global geometry/dtype {visual_global.shape}/{visual_global.dtype} "
            f"!= {(n_cache, D)}/float16"
        )
    if has_text.shape != (n_cache,) or has_text.dtype != np.bool_:
        raise CompositionalInputError(
            f"has_text geometry/dtype {has_text.shape}/{has_text.dtype} "
            f"!= {(n_cache,)}/bool"
        )
    if meta.get("N") != n_cache:
        raise CompositionalInputError(
            f"cache meta N={meta.get('N')!r} != image_ids rows {n_cache}"
        )
    if meta.get("D_proj") != D:
        raise CompositionalInputError(
            f"cache meta D_proj={meta.get('D_proj')!r} != text width {D}"
        )

    payload: dict[str, Any] = {
        "schema_version": 1,
        "cache_dir": explicit_cache,
        "config_declared_cache_dir": configured,
        "files": files,
        "geometry": {
            "N": n_cache,
            "cache_text_slots": int(text_part.shape[1]),
            "extraction_model_slots": M,
            "D": D,
        },
    }
    payload["aggregate_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "binding": payload,
        "image_ids": image_ids,
        "text_part": text_part,
        "visual_global": visual_global,
        "has_text": has_text,
    }


def _atomic_json(path: str, payload: Mapping[str, Any]) -> None:
    temporary = f"{path}.{os.getpid()}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


# ----------------------- main metrics -----------------------

def _per_codebook_concentration(
    cb_idx_v: np.ndarray,           # [Nv, M]
    feats_per_slot: np.ndarray,     # [Nv, M, D]  for text_part; or [Nv, D] for global
    min_cluster: int,
    rng_seed: int,
    use_per_slot_feature: bool,
    centered: bool,
) -> Tuple[List[List[Optional[float]]], List[List[int]], List[float], List[float], List[float]]:
    """Generic core: per-codebook intra-cluster cosine of `feats_per_slot`.
    If use_per_slot_feature: feats[:, m, :] is the feature seen by codebook m.
    Else: feats[:] is global and shared across codebooks.
    If centered: subtract per-slot mean from features first.
    """
    Nv, M = cb_idx_v.shape
    K = int(cb_idx_v.max()) + 1
    rng = np.random.default_rng(rng_seed)

    out_per_codeword: List[List[Optional[float]]] = [[None] * K for _ in range(M)]
    out_per_codeword_pop: List[List[int]] = [[0] * K for _ in range(M)]
    cb_means: List[float] = []
    cb_baselines: List[float] = []

    for m in range(M):
        if use_per_slot_feature:
            ev_all = feats_per_slot[:, m, :]            # [Nv, D]
        else:
            ev_all = feats_per_slot                     # [Nv, D]
        if centered:
            ev_all = ev_all - ev_all.mean(axis=0, keepdims=True)

        intra_vals, weights = [], []
        for k in range(K):
            idxs = np.where(cb_idx_v[:, m] == k)[0]
            out_per_codeword_pop[m][k] = int(idxs.size)
            if idxs.size < min_cluster:
                continue
            s = _intra_cluster_cos_sim(ev_all[idxs])
            out_per_codeword[m][k] = s
            intra_vals.append(s)
            weights.append(idxs.size)
        cb_means.append(
            float(np.average(intra_vals, weights=weights)) if intra_vals else float("nan")
        )

        shuffled = rng.permutation(cb_idx_v[:, m])
        intra_vals_b, weights_b = [], []
        for k in range(K):
            idxs = np.where(shuffled == k)[0]
            if idxs.size < min_cluster:
                continue
            intra_vals_b.append(_intra_cluster_cos_sim(ev_all[idxs]))
            weights_b.append(idxs.size)
        cb_baselines.append(
            float(np.average(intra_vals_b, weights=weights_b)) if intra_vals_b else float("nan")
        )

    lifts = [
        (m - b) if (not np.isnan(m) and not np.isnan(b)) else float("nan")
        for m, b in zip(cb_means, cb_baselines)
    ]
    return out_per_codeword, out_per_codeword_pop, cb_means, cb_baselines, lifts


def metric_b_text_concentration(
    codebook_indices: np.ndarray,        # [N, M]
    text_part: np.ndarray,               # [N_cache, M, D_proj]
    visual_global: Optional[np.ndarray], # [N_cache, D_proj] or None
    cache_rows: np.ndarray,              # [N] -> cache row (-1 = miss)
    has_text: np.ndarray,                # [N_cache] bool
    min_cluster: int = 5,
    rng_seed: int = 0,
) -> Dict[str, object]:
    """Three variants of intra-cluster semantic concentration:

    B0: raw text_part per-slot (legacy; SigLIP2 text encoder has a tight
        baseline ~0.88 so lift is near-zero noise)
    B1: text_part per-slot, centered by per-slot mean (removes the slot
        baseline so any non-zero lift reflects true within-slot concept clusters)
    B2: visual_global, SHARED across codebooks (uses *all* DB samples,
        not just the 5K with captions; baseline is meaningful)

    For each variant we report the per-codebook clustered cosine vs a
    shuffled-assignment baseline; the "lift" is real - baseline.
    """
    N, M = codebook_indices.shape
    # B0/B1 use text features -> only samples with cached real captions
    valid_text = (cache_rows >= 0) & has_text[np.clip(cache_rows, 0, len(has_text) - 1)]
    valid_text_idx = np.where(valid_text)[0]
    print(f"[B0/B1] text-based: using {valid_text_idx.size}/{N} db samples (has_text=True).")
    out = {
        "n_text_samples": int(valid_text_idx.size),
        "min_cluster":    int(min_cluster),
    }
    if valid_text_idx.size > 0:
        cb_idx_t = codebook_indices[valid_text_idx]
        text_feats = np.asarray(text_part[cache_rows[valid_text_idx]], dtype=np.float32)  # [Nv, M, D]
        for tag, centered in [("b0_raw_text", False), ("b1_centered_text", True)]:
            cw, pop, means, baselines, lifts = _per_codebook_concentration(
                cb_idx_t, text_feats, min_cluster, rng_seed,
                use_per_slot_feature=True, centered=centered,
            )
            out[tag] = {
                "codeword_intra_sim":           cw,
                "codeword_population":          pop,
                "codebook_mean_intra_sim":      means,
                "random_baseline_per_codebook": baselines,
                "compositional_lift_per_codebook": lifts,
                "mean_compositional_lift":      float(np.nanmean(lifts)),
            }
    else:
        print("[B0/B1] skipped: no db rows have cached captions (text path unavailable for this cache).")

    # B2: visual_global on ALL DB samples
    if visual_global is not None:
        valid_v = cache_rows >= 0
        valid_v_idx = np.where(valid_v)[0]
        print(f"[B2] visual_global-based: using {valid_v_idx.size}/{N} db samples.")
        cb_idx_v = codebook_indices[valid_v_idx]
        vg_feats = np.asarray(visual_global[cache_rows[valid_v_idx]], dtype=np.float32)  # [Nv, D]
        cw, pop, means, baselines, lifts = _per_codebook_concentration(
            cb_idx_v, vg_feats, min_cluster, rng_seed,
            use_per_slot_feature=False, centered=False,
        )
        out["b2_visual_global"] = {
            "n_samples":                    int(valid_v_idx.size),
            "codeword_intra_sim":           cw,
            "codeword_population":          pop,
            "codebook_mean_intra_sim":      means,
            "random_baseline_per_codebook": baselines,
            "compositional_lift_per_codebook": lifts,
            "mean_compositional_lift":      float(np.nanmean(lifts)),
        }
    return out


def metric_c_grids(
    codebook_indices: np.ndarray,        # [N, M]
    image_paths: np.ndarray,             # [N]
    out_root: str,
    viz_top_codewords: int = 5,
    viz_per_grid: int = 9,
    cols: int = 3,
) -> Dict[str, object]:
    """For each codebook, pick top-`viz_top_codewords` populated codewords
    and tile their top-`viz_per_grid` sample images as a 3x3 grid."""
    N, M = codebook_indices.shape
    out: List[Dict] = []
    for m in range(M):
        K = int(codebook_indices[:, m].max()) + 1
        counts = np.bincount(codebook_indices[:, m].astype(np.int64), minlength=K)
        order = np.argsort(-counts)        # descending populated
        chosen: List[Dict] = []
        for k in order[:viz_top_codewords]:
            idxs = np.where(codebook_indices[:, m] == k)[0]
            sample_paths = [str(image_paths[i]) for i in idxs[:viz_per_grid]]
            grid_path = os.path.join(
                out_root, f"codebook_grids/cb{m}_cw{int(k):03d}.png"
            )
            _save_image_grid(sample_paths, grid_path, cols=cols)
            chosen.append({
                "codeword":      int(k),
                "population":    int(counts[k]),
                "grid_path":     os.path.relpath(grid_path, out_root),
                "sample_paths":  sample_paths,
            })
        out.append({"codebook": m, "top_codewords": chosen})
    return {
        "viz_top_codewords": int(viz_top_codewords),
        "viz_per_grid":      int(viz_per_grid),
        "per_codebook":      out,
    }


# ----------------------- driver -----------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", required=True,
        help="Result directory with extract_db.npz")
    ap.add_argument("--cache_dir",   default="cache/flickr25k_siglip2")
    ap.add_argument("--dataset_root", default="dataset/Flickr25k")
    ap.add_argument("--min_cluster", type=int, default=5)
    ap.add_argument("--viz_top_codewords", type=int, default=5)
    ap.add_argument("--viz_per_grid",      type=int, default=9)
    ap.add_argument("--skip_grids", action="store_true", default=False,
        help="Skip metric C (image grids). Just compute metric B.")
    ap.add_argument(
        "--allow_backfilled", action="store_true", default=False,
        help=(
            "admit retrospectively bound extraction for diagnostics; output is "
            "explicitly paper_eligible=false"
        ),
    )
    args = ap.parse_args()

    extraction = _load_phase5_extraction(
        args.result_dir, split="db", allow_backfilled=args.allow_backfilled
    )
    dataset_root = _validate_dataset_root_binding(
        args.dataset_root, extraction=extraction
    )
    cb_idx = np.asarray(extraction["codebook_indices"], dtype=np.int64)
    image_paths = np.asarray(extraction["image_paths"])
    N, M = cb_idx.shape
    L, K = extraction["geometry"]["L"], extraction["geometry"]["K"]
    print(
        f"[compositional] admitted N={N}, M={M}, L={L}, K={K} from "
        "extraction/checkpoint manifests"
    )

    cache = _validate_cache_binding(args.cache_dir, extraction=extraction)
    cache_image_ids = cache["image_ids"]
    text_part = cache["text_part"]
    visual_global = cache["visual_global"]
    has_text = np.asarray(cache["has_text"])
    cache_rows = _build_path_to_cache_row(
        image_paths, cache_image_ids, dataset_root,
    )

    # --- metric B (three variants: B0 raw text, B1 centered text, B2 visual_global)
    print("[compositional] computing metric B (text+visual concentration) ...")
    metric_b = metric_b_text_concentration(
        cb_idx, text_part, visual_global, cache_rows, has_text,
        min_cluster=args.min_cluster,
    )
    for tag in ("b0_raw_text", "b1_centered_text", "b2_visual_global"):
        if tag not in metric_b:
            continue
        s = metric_b[tag]
        print(f"[{tag}] mean lift = {s['mean_compositional_lift']:.4f}  "
              f"means={[f'{x:.3f}' for x in s['codebook_mean_intra_sim']]}  "
              f"base ={[f'{x:.3f}' for x in s['random_baseline_per_codebook']]}")

    # --- metric C
    metric_c: Dict[str, object] = {}
    if not args.skip_grids:
        print("[compositional] computing metric C (image grids) ...")
        metric_c = metric_c_grids(
            cb_idx, image_paths,
            out_root=args.result_dir,
            viz_top_codewords=args.viz_top_codewords,
            viz_per_grid=args.viz_per_grid,
        )
        print(f"[C] saved {sum(len(cb['top_codewords']) for cb in metric_c['per_codebook'])} grids.")

    out = {
        "result_dir":     args.result_dir,
        "cache_dir":      args.cache_dir,
        "dataset_root":   args.dataset_root,
        "metric_b":       metric_b,
        "metric_c":       metric_c if metric_c else None,
        "input_binding": extraction["input_binding"],
        "extraction_file_evidence": extraction["transaction_evidence"],
        "cache_binding": cache["binding"],
        "geometry": extraction["geometry"],
        "analysis_split": "db",
        "paper_eligibility": {
            "eligible": bool(extraction["paper_eligible"]),
            "reason": (
                "fresh extraction/checkpoint manifests and exact cache/image-ID binding"
                if extraction["paper_eligible"]
                else "diagnostic only: extraction manifests were backfilled"
            ),
        },
        "producer": {
            "path": os.path.abspath(__file__),
            "sha256": _sha256_file(__file__),
        },
    }
    for evidence in extraction["transaction_evidence"].values():
        _assert_file_evidence_current(evidence)
    for evidence in cache["binding"]["files"].values():
        _assert_file_evidence_current(evidence)
    out_p = os.path.join(args.result_dir, "compositional_eval.json")
    _atomic_json(out_p, out)
    print(f"[compositional] saved -> {out_p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
