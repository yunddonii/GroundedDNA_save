#!/usr/bin/env python3
"""Prepare a provenance-checked semantic-detail cache overlay.

The A/B/ABC semantic-detail experiments need three additions to an existing
factual CLIP cache:

* deterministic local-caption minimal-pair foils;
* pooled and token-level CLIP foil sidecars;
* leakage-free, local-slot-only whitening statistics for both optTrain and
  trainOnly row sets.

This orchestrator deliberately treats each multi-file output as a transaction:
an entirely missing group is built, an entirely complete group is validated
and reused, and a partially present group is rejected.  It never passes an
overwrite flag to the underlying extractors and never replaces donor-cache
files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).absolute().parent.parent
SCHEMA_VERSION = "groundeddna-semantic-detail-cache-overlay-v1"


@dataclass(frozen=True)
class DatasetDefaults:
    base_cache: str
    qwen: str
    overlay: str
    foil_jsonl: str


DATASETS: Mapping[str, DatasetDefaults] = {
    "flickr": DatasetDefaults(
        base_cache="cache/flickr25k_clip_v4plus_qwen3_tokens",
        qwen="cache/flickr25k_qwen3_v4_trainset.jsonl",
        overlay="cache/flickr25k_clip_v4plus_qwen3_tokens_foils",
        # This is the already audited Flickr foil set used by the 2^3 screen.
        foil_jsonl=(
            "artifacts/semantic_detail_ablation/"
            "flickr25k_qwen3_v4_trainset.foils.jsonl"
        ),
    ),
    "mscoco": DatasetDefaults(
        base_cache="cache/mscoco_clip_v5b_tokens",
        qwen="cache/mscoco_qwen3_v5b_trainset.jsonl",
        overlay="cache/mscoco_clip_v5b_tokens_foils",
        foil_jsonl="artifacts/semantic_detail_multidataset/mscoco.foils.jsonl",
    ),
    "nuswide": DatasetDefaults(
        base_cache="cache/nuswide_clip_tokens",
        qwen="cache/nuswide_qwen3_v4_trainset.jsonl",
        overlay="cache/nuswide_clip_tokens_foils",
        foil_jsonl="artifacts/semantic_detail_multidataset/nuswide.foils.jsonl",
    ),
    "cifar10": DatasetDefaults(
        base_cache="cache/cifar10_clip",
        qwen="cache/cifar10_qwen.jsonl",
        overlay="cache/cifar10_clip_foils",
        foil_jsonl="artifacts/semantic_detail_multidataset/cifar10.foils.jsonl",
    ),
}

FOIL_JSON_GROUP = ("foil_jsonl", "foil_meta")
POOLED_GROUP = (
    "text_foil_part.f16.npy",
    "text_foil_valid.bool.npy",
    "text_foil_image_ids.json",
    "text_foil_edits.jsonl",
    "text_foil_meta.json",
)
TOKEN_GROUP = (
    "text_foil_tokens.f16.npy",
    "text_foil_token_mask.bool.npy",
    "text_foil_token_image_ids.json",
    "text_foil_token_meta.json",
)
WHITEN_OUTPUTS = (
    "text_whiten_optTrain_localOnly.npz",
    "text_whiten_trainOnly_localOnly.npz",
)
MANIFEST_NAME = "semantic_detail_cache_manifest.json"

# These names belong to the overlay.  Everything else at the top level of the
# factual cache is linked into the overlay, so a launcher can use it as a
# drop-in cache directory.
OVERLAY_OWNED_NAMES = frozenset(
    POOLED_GROUP
    + TOKEN_GROUP
    + WHITEN_OUTPUTS
    + tuple(name + ".meta.json" for name in WHITEN_OUTPUTS)
    + (MANIFEST_NAME,)
)


def _abspath(path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return Path(os.path.abspath(candidate))


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path):
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"[semantic-cache] invalid JSON {path}: {exc}") from exc


def _same_file_or_target(left: Path, right: Path) -> bool:
    return os.path.realpath(left) == os.path.realpath(right)


def _group_state(label: str, paths: Sequence[Path]) -> str:
    present = [_lexists(path) for path in paths]
    if all(present):
        return "complete"
    if not any(present):
        return "missing"
    found = [str(path) for path, exists in zip(paths, present) if exists]
    absent = [str(path) for path, exists in zip(paths, present) if not exists]
    raise SystemExit(
        f"[semantic-cache] refusing partial {label} group.\n"
        f"  present: {', '.join(found)}\n"
        f"  missing: {', '.join(absent)}\n"
        "Audit/move the partial group aside before retrying; this preparer "
        "never overwrites it."
    )


def _run(command: Sequence[str]) -> None:
    print("[semantic-cache] run:", shlex.join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def _nearest_existing_parent(path: Path) -> Path:
    current = path
    while not current.exists():
        if current.parent == current:
            raise SystemExit(
                f"[semantic-cache] cannot locate an existing parent for {path}"
            )
        current = current.parent
    return current


def _device_from_gpu(gpu: str) -> str:
    lowered = gpu.strip().lower()
    if lowered in {"cpu", "-1"}:
        return "cpu"
    try:
        index = int(gpu)
    except ValueError as exc:
        raise SystemExit(
            f"[semantic-cache] GPU must be a non-negative integer, cpu, or -1; "
            f"got {gpu!r}"
        ) from exc
    if index < 0:
        raise SystemExit(
            f"[semantic-cache] GPU must be non-negative (or exactly -1); got {index}"
        )
    return f"cuda:{index}"


def _load_base_contract(base: Path) -> dict:
    required = (
        "image_ids.json",
        "text_part.f16.npy",
        "has_text.bool.npy",
        "text_tokens.f16.npy",
        "text_token_mask.bool.npy",
        "opt_train_rows.npy",
        "train_all_rows.npy",
    )
    missing = [str(base / name) for name in required if not (base / name).is_file()]
    if missing:
        raise SystemExit(
            "[semantic-cache] factual cache is incomplete: " + ", ".join(missing)
        )

    image_ids = _read_json(base / "image_ids.json")
    if (
        not isinstance(image_ids, list)
        or any(not isinstance(item, str) for item in image_ids)
        or len(set(image_ids)) != len(image_ids)
    ):
        raise SystemExit(
            f"[semantic-cache] {base / 'image_ids.json'} must contain unique strings"
        )
    n_rows = len(image_ids)
    text_part = np.load(base / "text_part.f16.npy", mmap_mode="r")
    text_tokens = np.load(base / "text_tokens.f16.npy", mmap_mode="r")
    text_mask = np.load(base / "text_token_mask.bool.npy", mmap_mode="r")
    has_text = np.load(base / "has_text.bool.npy", mmap_mode="r")
    if (
        text_part.ndim != 3
        or text_part.shape[:2] != (n_rows, 6)
        or text_part.dtype != np.float16
    ):
        raise SystemExit(
            f"[semantic-cache] invalid factual pooled-text contract: "
            f"{text_part.shape}/{text_part.dtype}"
        )
    if (
        text_tokens.ndim != 4
        or text_tokens.shape[:2] != (n_rows, 6)
        or text_tokens.shape[3] != text_part.shape[2]
        or text_tokens.dtype != np.float16
    ):
        raise SystemExit(
            f"[semantic-cache] invalid factual token contract: "
            f"{text_tokens.shape}/{text_tokens.dtype}"
        )
    if text_mask.shape != text_tokens.shape[:3] or text_mask.dtype != np.bool_:
        raise SystemExit(
            f"[semantic-cache] invalid factual token mask: "
            f"{text_mask.shape}/{text_mask.dtype}"
        )
    if has_text.shape != (n_rows,) or has_text.dtype != np.bool_:
        raise SystemExit(
            f"[semantic-cache] invalid has_text mask: "
            f"{has_text.shape}/{has_text.dtype}"
        )

    row_contract: Dict[str, dict] = {}
    for filename in ("opt_train_rows.npy", "train_all_rows.npy"):
        rows = np.asarray(np.load(base / filename)).astype(np.int64).ravel()
        if (
            rows.size == 0
            or len(np.unique(rows)) != rows.size
            or int(rows.min()) < 0
            or int(rows.max()) >= n_rows
        ):
            raise SystemExit(
                f"[semantic-cache] invalid row-index contract in {base / filename}"
            )
        kept = rows[np.asarray(has_text[rows], dtype=bool)]
        row_contract[filename] = {
            "count": int(rows.size),
            "captioned_count": int(kept.size),
            "rows_used": int(kept.size * 5),
            "sha256": _sha256(base / filename),
        }

    return {
        "image_ids": image_ids,
        "n_rows": n_rows,
        "dimension": int(text_part.shape[2]),
        "max_length": int(text_tokens.shape[2]),
        "token_shape": tuple(int(item) for item in text_tokens.shape),
        "mask_shape": tuple(int(item) for item in text_mask.shape),
        "row_contract": row_contract,
        "image_ids_sha256": _sha256(base / "image_ids.json"),
    }


def _foil_paths(foil_jsonl: Path) -> Tuple[Path, Path]:
    return foil_jsonl, Path(str(foil_jsonl) + ".meta.json")


def _validate_foil_jsonl(foil_jsonl: Path, qwen: Path, seed: int) -> dict:
    foil_meta_path = Path(str(foil_jsonl) + ".meta.json")
    metadata = _read_json(foil_meta_path)
    expected = {
        "schema_version": "groundeddna-local-caption-foil-v1",
        "seed": seed,
        "source_sha256": _sha256(qwen),
        "output_sha256": _sha256(foil_jsonl),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise SystemExit(
                f"[semantic-cache] stale/incompatible foil JSONL metadata: "
                f"{foil_meta_path} key {key!r} is {metadata.get(key)!r}, "
                f"expected {value!r}"
            )
    contract = metadata.get("contract", {})
    if contract.get("global_slot_valid") is not False:
        raise SystemExit("[semantic-cache] foil metadata does not exclude C_global")
    if int(metadata.get("rows", 0)) <= 0:
        raise SystemExit("[semantic-cache] foil JSONL reports no source rows")
    return metadata


def _validate_pooled(
    overlay: Path, foil_jsonl: Path, base: Path, contract: dict
) -> dict:
    metadata = _read_json(overlay / "text_foil_meta.json")
    expected_shape = (contract["n_rows"], 6, contract["dimension"])
    pooled = np.load(overlay / "text_foil_part.f16.npy", mmap_mode="r")
    valid = np.load(overlay / "text_foil_valid.bool.npy", mmap_mode="r")
    if pooled.shape != expected_shape or pooled.dtype != np.float16:
        raise SystemExit(
            f"[semantic-cache] invalid pooled foil array: "
            f"{pooled.shape}/{pooled.dtype}, expected {expected_shape}/float16"
        )
    if valid.shape != expected_shape[:2] or valid.dtype != np.bool_:
        raise SystemExit(
            f"[semantic-cache] invalid foil-valid array: "
            f"{valid.shape}/{valid.dtype}"
        )
    if bool(valid[:, 0].any()):
        raise SystemExit("[semantic-cache] pooled foil mask marks C_global valid")
    if _read_json(overlay / "text_foil_image_ids.json") != contract["image_ids"]:
        raise SystemExit("[semantic-cache] pooled foil image order differs from donor")
    checks = {
        "schema_version": "groundeddna-local-caption-foil-features-v1",
        "foil_jsonl_sha256": _sha256(foil_jsonl),
        "cache_image_ids_sha256": contract["image_ids_sha256"],
        "global_slot_valid": False,
        "shape": list(expected_shape),
    }
    for key, value in checks.items():
        if metadata.get(key) != value:
            raise SystemExit(
                f"[semantic-cache] incompatible pooled foil metadata key {key}: "
                f"{metadata.get(key)!r} != {value!r}"
            )
    if int(valid.sum()) <= 0:
        raise SystemExit("[semantic-cache] pooled foil cache has zero valid pairs")
    return metadata


def _validate_tokens(
    overlay: Path, foil_jsonl: Path, contract: dict
) -> dict:
    metadata = _read_json(overlay / "text_foil_token_meta.json")
    tokens = np.load(overlay / "text_foil_tokens.f16.npy", mmap_mode="r")
    mask = np.load(overlay / "text_foil_token_mask.bool.npy", mmap_mode="r")
    if tokens.shape != contract["token_shape"] or tokens.dtype != np.float16:
        raise SystemExit(
            f"[semantic-cache] invalid foil token array: "
            f"{tokens.shape}/{tokens.dtype}, expected "
            f"{contract['token_shape']}/float16"
        )
    if mask.shape != contract["mask_shape"] or mask.dtype != np.bool_:
        raise SystemExit(
            f"[semantic-cache] invalid foil token mask: {mask.shape}/{mask.dtype}"
        )
    if bool(mask[:, 0].any()):
        raise SystemExit("[semantic-cache] foil token mask marks C_global valid")
    if _read_json(overlay / "text_foil_token_image_ids.json") != contract["image_ids"]:
        raise SystemExit("[semantic-cache] foil token image order differs from donor")
    checks = {
        "schema_version": "groundeddna-local-caption-foil-token-features-v1",
        "foil_jsonl_sha256": _sha256(foil_jsonl),
        "cache_image_ids_sha256": contract["image_ids_sha256"],
        "global_slot_valid": False,
        "shape": list(contract["token_shape"]),
        "mask_shape": list(contract["mask_shape"]),
    }
    for key, value in checks.items():
        if metadata.get(key) != value:
            raise SystemExit(
                f"[semantic-cache] incompatible foil token metadata key {key}: "
                f"{metadata.get(key)!r} != {value!r}"
            )
    return metadata


def _whiten_paths(overlay: Path, filename: str) -> Tuple[Path, Path]:
    output = overlay / filename
    return output, Path(str(output) + ".meta.json")


def _validate_whiten(
    output: Path, row_index: Path, base: Path, contract: dict
) -> dict:
    metadata = _read_json(Path(str(output) + ".meta.json"))
    with np.load(output) as whitening:
        keys = set(whitening.files)
        if keys != {"mu", "U", "S"}:
            raise SystemExit(
                f"[semantic-cache] {output} has keys {sorted(keys)}, "
                "expected ['S', 'U', 'mu']"
            )
        dimension = contract["dimension"]
        if (
            whitening["mu"].shape != (dimension,)
            or whitening["U"].shape != (dimension, dimension)
            or whitening["S"].shape != (dimension,)
        ):
            raise SystemExit(f"[semantic-cache] invalid whitening shapes in {output}")
    expected_rows = contract["row_contract"][row_index.name]["rows_used"]
    if (
        metadata.get("local_slots_only") is not True
        or metadata.get("leakage_free_fit") is not True
        or metadata.get("residualize_first") is not False
        or int(metadata.get("D", -1)) != contract["dimension"]
        or int(metadata.get("rows_used", -1)) != expected_rows
    ):
        raise SystemExit(
            f"[semantic-cache] incompatible local-only whitening metadata: "
            f"{output}.meta.json"
        )
    metadata_index = metadata.get("row_index_npy")
    if not metadata_index or not _same_file_or_target(
        Path(metadata_index), row_index
    ):
        raise SystemExit(
            f"[semantic-cache] whitening row index mismatch in "
            f"{output}.meta.json: {metadata_index!r} != {row_index}"
        )
    metadata_cache = metadata.get("cache_dir")
    if not metadata_cache or not _same_file_or_target(Path(metadata_cache), base):
        raise SystemExit(
            f"[semantic-cache] whitening donor cache mismatch in "
            f"{output}.meta.json"
        )
    return metadata


def _validate_overlay_links(base: Path, overlay: Path) -> Tuple[int, int]:
    if _lexists(overlay) and not overlay.is_dir():
        raise SystemExit(f"[semantic-cache] overlay is not a directory: {overlay}")
    if os.path.realpath(base) == os.path.realpath(overlay):
        raise SystemExit("[semantic-cache] overlay must differ from factual cache")
    linked = 0
    missing = 0
    if not overlay.exists():
        return linked, sum(
            1
            for source in base.iterdir()
            if source.name not in OVERLAY_OWNED_NAMES
            and (source.is_file() or source.is_symlink())
        )
    for source in sorted(base.iterdir()):
        if source.name in OVERLAY_OWNED_NAMES:
            continue
        if not (source.is_file() or source.is_symlink()):
            continue
        target = overlay / source.name
        if not _lexists(target):
            missing += 1
            continue
        if not target.is_symlink() or not _same_file_or_target(target, source):
            raise SystemExit(
                f"[semantic-cache] overlay donor collision at {target}; expected "
                f"a symlink resolving to {source}"
            )
        linked += 1
    return linked, missing


def _create_missing_overlay_links(base: Path, overlay: Path) -> None:
    overlay.mkdir(parents=True, exist_ok=True)
    for source in sorted(base.iterdir()):
        if source.name in OVERLAY_OWNED_NAMES:
            continue
        if not (source.is_file() or source.is_symlink()):
            continue
        target = overlay / source.name
        if _lexists(target):
            if not target.is_symlink() or not _same_file_or_target(target, source):
                raise SystemExit(
                    f"[semantic-cache] refusing overlay collision: {target}"
                )
            continue
        os.symlink(str(source.absolute()), target)
        print(f"[semantic-cache] linked {target.name} -> {source}")


def _disk_preflight(
    overlay: Path,
    foil_jsonl: Path,
    qwen: Path,
    contract: dict,
    states: Mapping[str, str],
) -> None:
    n_rows = contract["n_rows"]
    dimension = contract["dimension"]
    pooled_bytes = n_rows * 6 * dimension * np.dtype(np.float16).itemsize
    pooled_bytes += n_rows * 6 * np.dtype(np.bool_).itemsize
    token_bytes = int(np.prod(contract["token_shape"])) * np.dtype(
        np.float16
    ).itemsize
    token_bytes += int(np.prod(contract["mask_shape"])) * np.dtype(
        np.bool_
    ).itemsize
    required_overlay = 0
    if states["pooled"] == "missing":
        required_overlay += pooled_bytes
    if states["token"] == "missing":
        required_overlay += token_bytes
    for key in ("whiten_opt", "whiten_train"):
        if states[key] == "missing":
            required_overlay += 2 * 1024 * 1024
    # Leave room for metadata and filesystem bookkeeping.  The token extractor
    # itself independently performs an exact free-space check before encoding.
    reserve = 256 * 1024 * 1024 if required_overlay else 0
    output_fs = _nearest_existing_parent(overlay.parent)
    free_overlay = shutil.disk_usage(output_fs).free
    print(
        "[semantic-cache] planned overlay allocation: "
        f"{required_overlay / 1e9:.2f} GB "
        f"(pooled={pooled_bytes / 1e9:.2f}, token={token_bytes / 1e9:.2f}); "
        f"free={free_overlay / 1e9:.2f} GB"
    )
    if free_overlay < required_overlay + reserve:
        raise SystemExit(
            "[semantic-cache] insufficient disk for the missing overlay groups: "
            f"need {(required_overlay + reserve) / 1e9:.2f} GB including reserve, "
            f"have {free_overlay / 1e9:.2f} GB"
        )
    if states["foil"] == "missing":
        # JSONL expansion is data-dependent; 4x the source size is a
        # conservative bound for the deterministic edit/audit payload.
        required_foil = qwen.stat().st_size * 4
        foil_fs = _nearest_existing_parent(foil_jsonl.parent)
        free_foil = shutil.disk_usage(foil_fs).free
        print(
            "[semantic-cache] planned foil JSONL allowance: "
            f"{required_foil / 1e9:.2f} GB; free={free_foil / 1e9:.2f} GB"
        )
        if free_foil < required_foil + 16 * 1024 * 1024:
            raise SystemExit("[semantic-cache] insufficient disk for foil JSONL")


def _build_whiten(
    python: str,
    base: Path,
    overlay: Path,
    filename: str,
    row_filename: str,
    contract: dict,
) -> None:
    final, final_meta = _whiten_paths(overlay, filename)
    if _group_state(f"{filename} whitening", (final, final_meta)) != "missing":
        raise AssertionError("_build_whiten called for a complete group")
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{filename}.", suffix=".npz", dir=overlay
    )
    os.close(fd)
    temporary = Path(temporary_name)
    temporary.unlink()
    temporary_meta = Path(str(temporary) + ".meta.json")
    try:
        _run(
            (
                python,
                str(ROOT / "scripts/build_text_whiten_matrix.py"),
                "--cache_dir",
                str(base),
                "--out",
                str(temporary),
                "--row_index_npy",
                str(base / row_filename),
                "--local_slots_only",
            )
        )
        _validate_whiten(
            temporary, base / row_filename, base, contract
        )
        if _lexists(final) or _lexists(final_meta):
            raise SystemExit(
                f"[semantic-cache] whitening destination appeared concurrently: "
                f"{final}"
            )
        os.rename(temporary, final)
        os.rename(temporary_meta, final_meta)
    finally:
        for path in (temporary, temporary_meta):
            if _lexists(path):
                path.unlink()


def _source_hashes() -> dict:
    files = (
        Path(__file__).absolute(),
        ROOT / "scripts/build_counterfactual_caption_foils.py",
        ROOT / "scripts/extract_counterfactual_foil_features.py",
        ROOT / "scripts/extract_counterfactual_foil_token_features.py",
        ROOT / "scripts/build_text_whiten_matrix.py",
    )
    return {
        str(path.relative_to(ROOT)): _sha256(path)
        for path in files
    }


def _manifest_payload(
    dataset: str,
    base: Path,
    qwen: Path,
    overlay: Path,
    foil_jsonl: Path,
    seed: int,
    contract: dict,
    pooled_meta: dict,
    token_meta: dict,
) -> dict:
    whitening = {}
    for filename, row_filename in zip(
        WHITEN_OUTPUTS, ("opt_train_rows.npy", "train_all_rows.npy")
    ):
        whitening[filename] = {
            "sha256": _sha256(overlay / filename),
            "row_index": os.path.realpath(base / row_filename),
            "row_index_sha256": contract["row_contract"][row_filename]["sha256"],
            "rows_used": contract["row_contract"][row_filename]["rows_used"],
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "base_cache": os.path.realpath(base),
        "overlay": os.path.realpath(overlay),
        "qwen_jsonl": os.path.realpath(qwen),
        "qwen_jsonl_sha256": _sha256(qwen),
        "foil_jsonl": os.path.realpath(foil_jsonl),
        "foil_jsonl_sha256": _sha256(foil_jsonl),
        "foil_seed": seed,
        "image_ids_sha256": contract["image_ids_sha256"],
        "pooled_shape": pooled_meta["shape"],
        "token_shape": token_meta["shape"],
        "valid_per_slot": pooled_meta["valid_per_slot"],
        "whitening": whitening,
        "source_sha256": _source_hashes(),
    }


def _write_or_validate_manifest(path: Path, payload: dict) -> None:
    if _lexists(path):
        if _read_json(path) != payload:
            raise SystemExit(
                f"[semantic-cache] existing manifest does not match the validated "
                f"cache: {path}. Refusing to overwrite provenance."
            )
        print(f"[semantic-cache] reused identical manifest: {path}")
        return
    fd, temporary_name = tempfile.mkstemp(
        prefix=".semantic-detail-manifest-", suffix=".json", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if _lexists(path):
            raise SystemExit(
                f"[semantic-cache] manifest appeared concurrently: {path}"
            )
        os.rename(temporary_name, path)
        temporary_name = ""
    finally:
        if temporary_name and os.path.lexists(temporary_name):
            os.unlink(temporary_name)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=tuple(DATASETS))
    parser.add_argument(
        "--gpu",
        default="0",
        help="CUDA index used for both CLIP encoding passes; use cpu or -1 for CPU.",
    )
    parser.add_argument("--base-cache", default=None)
    parser.add_argument("--qwen", default=None)
    parser.add_argument("--overlay", default=None)
    parser.add_argument("--foil-jsonl", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pooled-batch-size", type=int, default=256)
    parser.add_argument("--token-batch-size", type=int, default=256)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate inputs/state/capacity without creating any file or directory.",
    )
    args = parser.parse_args()
    if args.pooled_batch_size <= 0 or args.token_batch_size <= 0:
        parser.error("batch sizes must be positive")
    return args


def main() -> int:
    args = _parse_args()
    defaults = DATASETS[args.dataset]
    base = _abspath(args.base_cache or defaults.base_cache)
    qwen = _abspath(args.qwen or defaults.qwen)
    overlay = _abspath(args.overlay or defaults.overlay)
    foil_jsonl = _abspath(args.foil_jsonl or defaults.foil_jsonl)
    device = _device_from_gpu(args.gpu)

    if not base.is_dir():
        raise SystemExit(f"[semantic-cache] missing factual cache: {base}")
    if not qwen.is_file():
        raise SystemExit(f"[semantic-cache] missing Qwen JSONL: {qwen}")
    if os.path.realpath(base) == os.path.realpath(overlay):
        raise SystemExit("[semantic-cache] --overlay must differ from --base-cache")

    contract = _load_base_contract(base)
    print(
        f"[semantic-cache] dataset={args.dataset}, N={contract['n_rows']}, "
        f"T={contract['max_length']}, D={contract['dimension']}, device={device}"
    )

    foil_state = _group_state(
        "foil JSONL", _foil_paths(foil_jsonl)
    )
    pooled_state = _group_state(
        "pooled foil sidecar",
        tuple(overlay / name for name in POOLED_GROUP),
    )
    token_state = _group_state(
        "token foil sidecar",
        tuple(overlay / name for name in TOKEN_GROUP),
    )
    whiten_states = {}
    for key, filename in (
        ("whiten_opt", WHITEN_OUTPUTS[0]),
        ("whiten_train", WHITEN_OUTPUTS[1]),
    ):
        whiten_states[key] = _group_state(
            f"{filename} whitening", _whiten_paths(overlay, filename)
        )
    states = {
        "foil": foil_state,
        "pooled": pooled_state,
        "token": token_state,
        **whiten_states,
    }
    print(
        "[semantic-cache] state: "
        + ", ".join(f"{key}={value}" for key, value in states.items())
    )
    linked, links_missing = _validate_overlay_links(base, overlay)
    print(
        f"[semantic-cache] overlay donor links: valid={linked}, "
        f"missing/planned={links_missing}"
    )

    if foil_state == "complete":
        _validate_foil_jsonl(foil_jsonl, qwen, args.seed)
    pooled_meta = None
    token_meta = None
    if pooled_state == "complete":
        if foil_state != "complete":
            raise SystemExit(
                "[semantic-cache] pooled sidecars exist without their foil JSONL"
            )
        pooled_meta = _validate_pooled(
            overlay, foil_jsonl, base, contract
        )
    if token_state == "complete":
        if pooled_state != "complete":
            raise SystemExit(
                "[semantic-cache] token sidecars exist without pooled sidecars"
            )
        token_meta = _validate_tokens(overlay, foil_jsonl, contract)
    for state_key, filename, row_filename in (
        (
            "whiten_opt",
            WHITEN_OUTPUTS[0],
            "opt_train_rows.npy",
        ),
        (
            "whiten_train",
            WHITEN_OUTPUTS[1],
            "train_all_rows.npy",
        ),
    ):
        if states[state_key] == "complete":
            _validate_whiten(
                overlay / filename, base / row_filename, base, contract
            )

    all_groups_complete = all(value == "complete" for value in states.values())
    manifest_path = overlay / MANIFEST_NAME
    if _lexists(manifest_path) and not all_groups_complete:
        raise SystemExit(
            f"[semantic-cache] completion manifest exists while one or more "
            f"artifact groups are missing: {manifest_path}"
        )
    if all_groups_complete:
        assert pooled_meta is not None and token_meta is not None
        expected_manifest = _manifest_payload(
            args.dataset,
            base,
            qwen,
            overlay,
            foil_jsonl,
            args.seed,
            contract,
            pooled_meta,
            token_meta,
        )
        if _lexists(manifest_path):
            if _read_json(manifest_path) != expected_manifest:
                raise SystemExit(
                    f"[semantic-cache] stale/incompatible completion manifest: "
                    f"{manifest_path}"
                )
            print(
                f"[semantic-cache] completion manifest validated: {manifest_path}"
            )
        else:
            print(
                f"[semantic-cache] completion manifest is missing and will be "
                f"created in build mode: {manifest_path}"
            )

    _disk_preflight(overlay, foil_jsonl, qwen, contract, states)

    foil_builder = str(ROOT / "scripts/build_counterfactual_caption_foils.py")
    pooled_extractor = str(
        ROOT / "scripts/extract_counterfactual_foil_features.py"
    )
    token_extractor = str(
        ROOT / "scripts/extract_counterfactual_foil_token_features.py"
    )

    if args.preflight_only:
        _run(
            (
                sys.executable,
                foil_builder,
                "--input",
                str(qwen),
                "--output",
                str(foil_jsonl),
                "--seed",
                str(args.seed),
                "--dry-run",
            )
        )
        if foil_state == "complete":
            _run(
                (
                    sys.executable,
                    pooled_extractor,
                    "--foil-jsonl",
                    str(foil_jsonl),
                    "--cache-dir",
                    str(base),
                    "--out-dir",
                    str(overlay),
                    "--encoder",
                    "clip",
                    "--dry-run",
                )
            )
            _run(
                (
                    sys.executable,
                    token_extractor,
                    "--foil-jsonl",
                    str(foil_jsonl),
                    "--cache-dir",
                    str(base),
                    "--out-dir",
                    str(overlay),
                    "--dry-run",
                )
            )
        else:
            print(
                "[semantic-cache] foil alignment dry-runs are deferred until the "
                "missing foil JSONL is built"
            )
        print("[semantic-cache] PREFLIGHT_ONLY complete; no files written")
        return 0

    if foil_state == "missing":
        _run(
            (
                sys.executable,
                foil_builder,
                "--input",
                str(qwen),
                "--output",
                str(foil_jsonl),
                "--seed",
                str(args.seed),
            )
        )
        _validate_foil_jsonl(foil_jsonl, qwen, args.seed)

    _create_missing_overlay_links(base, overlay)
    _validate_overlay_links(base, overlay)

    if pooled_state == "missing":
        _run(
            (
                sys.executable,
                pooled_extractor,
                "--foil-jsonl",
                str(foil_jsonl),
                "--cache-dir",
                str(base),
                "--out-dir",
                str(overlay),
                "--encoder",
                "clip",
                "--device",
                device,
                "--batch-size",
                str(args.pooled_batch_size),
            )
        )
    pooled_meta = _validate_pooled(overlay, foil_jsonl, base, contract)

    if token_state == "missing":
        _run(
            (
                sys.executable,
                token_extractor,
                "--foil-jsonl",
                str(foil_jsonl),
                "--cache-dir",
                str(base),
                "--out-dir",
                str(overlay),
                "--device",
                device,
                "--batch-size",
                str(args.token_batch_size),
            )
        )
    token_meta = _validate_tokens(overlay, foil_jsonl, contract)

    for state_key, filename, row_filename in (
        ("whiten_opt", WHITEN_OUTPUTS[0], "opt_train_rows.npy"),
        ("whiten_train", WHITEN_OUTPUTS[1], "train_all_rows.npy"),
    ):
        if states[state_key] == "missing":
            _build_whiten(
                sys.executable,
                base,
                overlay,
                filename,
                row_filename,
                contract,
            )
        _validate_whiten(
            overlay / filename, base / row_filename, base, contract
        )

    manifest = _manifest_payload(
        args.dataset,
        base,
        qwen,
        overlay,
        foil_jsonl,
        args.seed,
        contract,
        pooled_meta,
        token_meta,
    )
    _write_or_validate_manifest(overlay / MANIFEST_NAME, manifest)
    print("[semantic-cache] complete")
    print(f"[semantic-cache] CACHE={overlay}")
    print(
        "[semantic-cache] WHITEN_OPT="
        f"{overlay / WHITEN_OUTPUTS[0]}"
    )
    print(
        "[semantic-cache] WHITEN_TRAIN="
        f"{overlay / WHITEN_OUTPUTS[1]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
