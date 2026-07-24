#!/usr/bin/env python
"""Validate and aggregate semantic-detail P0 matrices.

The default input contract is the eight A/ABC manifests written by
``scripts/run_semantic_detail_p0_cell.sh``:

    artifacts/semantic_detail_multidataset/p0/
        {flickr,mscoco,nuswide,cifar10}_{A,ABC}_K128L4_p0.json

``--include-ab`` opts into the twelve-cell A/AB/ABC contract.  AB provenance
uses the dedicated AB cell/matrix runners while the legacy A/ABC source
contract remains unchanged.

By default this command fails closed and writes no summary unless every
requested cell is complete and valid.  ``--allow-incomplete`` is only a
progress-report escape hatch: it permits missing/preflight/stage-1 cells, but
it does not turn a protocol-invalid completed cell into an admitted result.

Only the requested JSON and Markdown summaries are written.  Training
artifacts and source manifests are read-only.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import shlex
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
DATASET_ORDER = ("flickr", "mscoco", "nuswide", "cifar10")
ARM_ORDER = ("A", "ABC")
EXTENDED_ARM_ORDER = ("A", "AB", "ABC")
STATUS_ORDER = ("preflight_passed", "stage1_complete", "complete")
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")

DATASETS: dict[str, dict[str, Any]] = {
    "flickr": {
        "canonical": "Flickr25k",
        "display": "Flickr25k",
        "map_r": 5000,
        "launcher": "scripts/train_flickr25k_v185_bidirTokenPrune05_clip.sh",
        "incumbent_result": (
            "result/260722+flickr25k_setting1_flickr_K128_L4_P0refit_e4"
            "+bs+64+e+60+proj_lr+0.001/cell_result.json"
        ),
        "incumbent_map": 0.8742346325610938,
        "incumbent_unique": 0.49817391304347824,
    },
    "mscoco": {
        "canonical": "MSCOCO",
        "display": "MSCOCO",
        "map_r": 5000,
        "launcher": "scripts/train_mscoco_F2_sweep_clip.sh",
        "incumbent_result": (
            "result/260721+mscoco_setting1_mscoco_K128_L4_P0refit_e24"
            "+bs+64+e+60+proj_lr+0.001/cell_result.json"
        ),
        "incumbent_map": 0.8256747046303248,
        "incumbent_unique": 0.2184427987837863,
    },
    "nuswide": {
        "canonical": "NUSWIDE",
        "display": "NUS-WIDE",
        "map_r": 5000,
        "launcher": "scripts/train_nuswide_v185_sweep_clip.sh",
        "incumbent_result": (
            "result/260721+nuswide_setting1_nuswide_K128_L4_P0refit_e4"
            "+bs+64+e+60+proj_lr+0.001/cell_result.json"
        ),
        "incumbent_map": 0.8328272693570413,
        "incumbent_unique": 0.2367059989470098,
    },
    "cifar10": {
        "canonical": "CIFAR10",
        "display": "CIFAR10",
        "map_r": 1000,
        "launcher": (
            "scripts/train_cifar10_v185_bidirTokenPrune05_ccs01_clip.sh"
        ),
        "incumbent_result": (
            "result/260722+cifar10_setting1_cifar10_K128_L4_P0refit_e14"
            "+bs+64+e+60+proj_lr+0.001/cell_result.json"
        ),
        "incumbent_map": 0.9032606579362007,
        "incumbent_unique": 0.2571864406779661,
    },
}

COMMON_SOURCE_FILES = (
    "config.py",
    "dataloaders.py",
    "evaluation_siglip2.py",
    "extraction_siglip2.py",
    "loss_siglip2.py",
    "model_siglip2.py",
    "p0_protocol.py",
    "train_siglip2.py",
    "val_split.py",
    "scripts/build_text_whiten_matrix.py",
    "scripts/eval_cell_bioproj.py",
    "scripts/prepare_semantic_detail_cache.py",
    "scripts/prepare_semantic_detail_cache.sh",
    "scripts/run_semantic_detail_multidataset_p0.sh",
    "scripts/run_semantic_detail_p0_cell.sh",
)
LEGACY_RUNNER_SOURCE_FILES = (
    "scripts/run_semantic_detail_multidataset_p0.sh",
    "scripts/run_semantic_detail_p0_cell.sh",
)
AB_RUNNER_SOURCE_FILES = (
    "scripts/run_semantic_detail_ab_multidataset_p0.sh",
    "scripts/run_semantic_detail_ab_p0_cell.sh",
)
SHARED_SOURCE_FILES = tuple(
    path for path in COMMON_SOURCE_FILES if path not in LEGACY_RUNNER_SOURCE_FILES
)
DYNAMIC_SOURCE_GLOBS = ("dna_utils/*.py", "models/*.py")
FACTUAL_ARRAY_NAMES = (
    "has_text.bool.npy",
    "text_part.f16.npy",
    "text_tokens.f16.npy",
    "text_token_mask.bool.npy",
    "visual_tokens_aug0.f16.npy",
    "visual_tokens_aug1.f16.npy",
)
FOIL_ARTIFACT_NAMES = (
    "text_foil_part.f16.npy",
    "text_foil_valid.bool.npy",
    "text_foil_tokens.f16.npy",
    "text_foil_token_mask.bool.npy",
    "text_foil_image_ids.json",
    "text_foil_token_image_ids.json",
)

STAGE1_REQUIRED = (
    "args.txt",
    "log.csv",
    "model_state_dict.pth",
    "model_state_dict_best.pth",
)
STAGE1_FORBIDDEN_PATTERNS = (
    "extract_db*.npz",
    "extract_query*.npz",
    "evaluation_siglip2_*.json",
    "cell_result*.json",
)
STAGE1_REQUIRED_MARKERS = (
    "Final checkpoint saved",
    "[p0-stage1] SKIP official-test extraction/evaluation",
)
STAGE1_FORBIDDEN_MARKERS = (
    "[p0-stage2] OFFICIAL TEST ISOLATED",
    "[final-eval] running extraction",
    "[final-eval] running evaluation",
    "[final-eval] paper mAP@R cutoff",
    "[CELL-RESULT]",
)
STAGE2_REQUIRED = (
    "args.txt",
    "log.csv",
    "model_state_dict.pth",
    "extract_db.npz",
    "extract_query.npz",
    "evaluation_siglip2_base.json",
    "evaluation_siglip2_base_bioproj.json",
    "cell_result.json",
)


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _atomic_write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _expect(
    errors: list[str], field: str, actual: Any, expected: Any
) -> None:
    if actual != expected:
        errors.append(f"{field}: expected {expected!r}, found {actual!r}")


def _expect_close(
    errors: list[str],
    field: str,
    actual: Any,
    expected: float,
    *,
    atol: float = 1e-12,
) -> None:
    try:
        value = float(actual)
    except (TypeError, ValueError):
        errors.append(f"{field}: expected finite float {expected!r}, found {actual!r}")
        return
    if not math.isfinite(value) or not math.isclose(
        value, expected, rel_tol=0.0, abs_tol=atol
    ):
        errors.append(f"{field}: expected {expected!r}, found {actual!r}")


def _resolve_declared_path(value: Any, repo_root: Path) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _parse_saved_args(path: Path, errors: list[str], label: str) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as error:
        errors.append(f"{label}: cannot read {path}: {error}")
        return {}
    parsed: dict[str, str] = {}
    for line in lines:
        match = re.match(r"^([A-Za-z][A-Za-z0-9_]*)-+(.*)$", line)
        if match:
            parsed[match.group(1)] = match.group(2).strip()
    return parsed


def _expect_arg_text(
    errors: list[str],
    args: Mapping[str, str],
    key: str,
    expected: str,
    label: str,
) -> None:
    actual = args.get(key)
    if actual != expected:
        errors.append(
            f"{label}.args.{key}: expected {expected!r}, found {actual!r}"
        )


def _expect_arg_float(
    errors: list[str],
    args: Mapping[str, str],
    key: str,
    expected: float,
    label: str,
) -> None:
    actual = args.get(key)
    try:
        value = float(actual) if actual is not None else None
    except ValueError:
        value = None
    if value is None or not math.isclose(
        value, expected, rel_tol=0.0, abs_tol=1e-12
    ):
        errors.append(
            f"{label}.args.{key}: expected {expected!r}, found {actual!r}"
        )


def _expect_arg_path(
    errors: list[str],
    args: Mapping[str, str],
    key: str,
    expected: Any,
    repo_root: Path,
    label: str,
) -> None:
    actual_path = _resolve_declared_path(args.get(key), repo_root)
    expected_path = _resolve_declared_path(expected, repo_root)
    if actual_path != expected_path:
        errors.append(
            f"{label}.args.{key}: expected path {expected_path}, "
            f"found {actual_path}"
        )


def _validate_saved_args(
    *,
    path: Path,
    label: str,
    dataset: str,
    arm: str,
    stage: int,
    selected_epoch: int | None,
    manifest: Mapping[str, Any],
    repo_root: Path,
    errors: list[str],
) -> None:
    errors_before_parse = len(errors)
    args = _parse_saved_args(path, errors, label)
    if not args:
        if len(errors) == errors_before_parse:
            errors.append(
                f"{label}.args: no parseable key/value records in {path}"
            )
        return
    dataset_spec = DATASETS[dataset]
    cache = _as_mapping(manifest.get("cache")) or {}
    whitening = _as_mapping(manifest.get("whitening")) or {}
    whitening_key = (
        "stage1_optTrain_localOnly"
        if stage == 1
        else "stage2_trainOnly_localOnly"
    )
    whitening_entry = _as_mapping(whitening.get(whitening_key)) or {}

    _expect_arg_text(
        errors, args, "dataset", dataset_spec["canonical"], label
    )
    _expect_arg_text(errors, args, "codebook_size", "128", label)
    _expect_arg_text(errors, args, "num_codons_per_codebook", "4", label)
    _expect_arg_text(errors, args, "evaluation", "True", label)
    _expect_arg_text(errors, args, "post_eval_compositional", "False", label)
    _expect_arg_text(errors, args, "xmodal_commit_skip_global", "True", label)
    _expect_arg_text(
        errors, args, "cibhash_dynamic_tau_skip_global", "True", label
    )
    _expect_arg_text(
        errors, args, "cibhash_ntxent_source", "visual_token", label
    )
    _expect_arg_float(errors, args, "lambda_cibhash_kl", 0.001, label)
    _expect_arg_path(
        errors,
        args,
        "siglip2_feature_cache_dir",
        cache.get("training"),
        repo_root,
        label,
    )
    _expect_arg_path(
        errors,
        args,
        "eval_cache_dir",
        cache.get("evaluation"),
        repo_root,
        label,
    )
    _expect_arg_path(
        errors,
        args,
        "text_whiten_npz",
        whitening_entry.get("path"),
        repo_root,
        label,
    )

    if stage == 1:
        _expect_arg_float(errors, args, "val_split_ratio", 0.1, label)
        _expect_arg_text(errors, args, "val_split_seed", "42", label)
        _expect_arg_text(errors, args, "final_epoch_eval", "False", label)
        _expect_arg_text(errors, args, "stop_after_epoch", "None", label)
    else:
        _expect_arg_float(errors, args, "val_split_ratio", 0.0, label)
        _expect_arg_text(errors, args, "val_split_seed", "42", label)
        _expect_arg_text(errors, args, "final_epoch_eval", "True", label)
        _expect_arg_text(
            errors, args, "stop_after_epoch", str(selected_epoch), label
        )

    if arm == "A":
        _expect_arg_float(
            errors, args, "text_hash_counterfactual_weight", 0.0, label
        )
        _expect_arg_text(
            errors,
            args,
            "text_hash_counterfactual_warmup_epochs",
            "0",
            label,
        )
        _expect_arg_text(
            errors, args, "cibhash_visual_token_bit_kl", "False", label
        )
    else:
        _expect_arg_float(
            errors, args, "text_hash_counterfactual_weight", 0.10, label
        )
        _expect_arg_float(
            errors, args, "text_hash_counterfactual_margin", 0.02, label
        )
        _expect_arg_text(
            errors,
            args,
            "text_hash_counterfactual_warmup_epochs",
            "1",
            label,
        )
        _expect_arg_text(
            errors,
            args,
            "cibhash_visual_token_bit_kl",
            "True" if arm == "ABC" else "False",
            label,
        )


def _protocol_core() -> dict[str, Any]:
    return {
        "name": "semantic-detail cross-dataset P0",
        "seed": 42,
        "K": 128,
        "num_codons_per_codebook": 4,
        "dna_bases": 24,
        "stage1": "90% opt-train; 10% held-out train validation; select E*",
        "stage2": "scratch refit on 100% train through E*",
        "official_test_evaluations": 1,
        "bio_projection": {
            "mandatory": True,
            "gc_min_frac": 0.416,
            "gc_max_frac": 0.584,
        },
    }


def _expected_extra_args(arm: str) -> list[str]:
    arguments = [
        "--no-post_eval_compositional",
        "--no_visualize",
        "--xmodal_commit_skip_global",
        "--cibhash_dynamic_tau_skip_global",
    ]
    if arm in ("AB", "ABC"):
        arguments.extend(
            [
                "--text_hash_counterfactual_weight",
                "0.10",
                "--text_hash_counterfactual_margin",
                "0.02",
                "--text_hash_counterfactual_warmup_epochs",
                "1",
            ]
        )
    if arm == "ABC":
        arguments.append("--cibhash_visual_token_bit_kl")
    return arguments


def _expected_source_files(
    dataset: str,
    arm: str,
    *,
    repo_root: Path = REPO,
) -> set[str]:
    runners = (
        AB_RUNNER_SOURCE_FILES
        if arm == "AB"
        else LEGACY_RUNNER_SOURCE_FILES
    )
    dynamic_sources = {
        path.relative_to(repo_root).as_posix()
        for pattern in DYNAMIC_SOURCE_GLOBS
        for path in repo_root.glob(pattern)
        if path.is_file()
    }
    return (
        set(SHARED_SOURCE_FILES)
        | set(runners)
        | dynamic_sources
        | {DATASETS[dataset]["launcher"]}
    )


def _validate_base_manifest(
    manifest: Mapping[str, Any],
    *,
    dataset: str,
    arm: str,
    repo_root: Path = REPO,
    errors: list[str],
) -> None:
    _expect(errors, "schema_version", manifest.get("schema_version"), 1)
    protocol = _as_mapping(manifest.get("protocol"))
    if protocol is None:
        errors.append("protocol: missing object")
    else:
        expected = _protocol_core()
        for key, value in expected.items():
            _expect(errors, f"protocol.{key}", protocol.get(key), value)
        _expect(errors, "protocol.dataset", protocol.get("dataset"), dataset)
        _expect(
            errors,
            "protocol.canonical_dataset",
            protocol.get("canonical_dataset"),
            DATASETS[dataset]["canonical"],
        )
        _expect(errors, "protocol.arm", protocol.get("arm"), arm)

    _expect(errors, "launcher", manifest.get("launcher"), DATASETS[dataset]["launcher"])
    expected_s1_tag = f"{dataset}_semantic_detail_{arm}_K128L4_P0val_s42"
    _expect(errors, "stage1_tag", manifest.get("stage1_tag"), expected_s1_tag)

    extra_args = manifest.get("extra_args")
    if not isinstance(extra_args, str):
        errors.append("extra_args: expected shell-escaped string")
    else:
        try:
            parsed = shlex.split(extra_args)
        except ValueError as error:
            errors.append(f"extra_args: cannot parse: {error}")
        else:
            _expect(errors, "extra_args tokens", parsed, _expected_extra_args(arm))

    source_sha = _as_mapping(manifest.get("source_sha256"))
    repo_root = repo_root.resolve()
    expected_sources = _expected_source_files(
        dataset, arm, repo_root=repo_root
    )
    if source_sha is None:
        errors.append("source_sha256: missing object")
    else:
        missing_sources = expected_sources - set(source_sha)
        if missing_sources:
            errors.append(
                "source_sha256: required path(s) missing: "
                f"{sorted(missing_sources)!r}"
            )
        unknown_sources = set(source_sha) - expected_sources
        if unknown_sources:
            errors.append(
                "source_sha256: unexpected path(s): "
                f"{sorted(unknown_sources)!r}"
            )
        for source_path, digest in source_sha.items():
            if not isinstance(source_path, str) or not isinstance(digest, str):
                errors.append("source_sha256: paths and digests must be strings")
                continue
            if not HEX_SHA256.fullmatch(digest):
                errors.append(
                    f"source_sha256.{source_path}: invalid SHA-256 {digest!r}"
                )
                continue
            if source_path in expected_sources:
                current_path = repo_root / source_path
                if not current_path.is_file():
                    errors.append(
                        f"source_sha256.{source_path}: current source file is missing"
                    )
                else:
                    current_digest = _file_sha256(current_path)
                    if current_digest != digest:
                        errors.append(
                            f"source_sha256.{source_path}: current SHA-256 "
                            f"{current_digest!r} does not match manifest {digest!r}"
                        )

    cache = _as_mapping(manifest.get("cache"))
    if cache is None:
        errors.append("cache: missing object")
    else:
        base = cache.get("base")
        foil = cache.get("foil_overlay")
        training = cache.get("training")
        expected_training = base if arm == "A" else foil
        _expect(errors, "cache.training", training, expected_training)
        if not isinstance(base, str) or not base:
            errors.append("cache.base: missing path")
        if not isinstance(foil, str) or not foil:
            errors.append("cache.foil_overlay: missing path")
        if not isinstance(cache.get("evaluation"), str):
            errors.append("cache.evaluation: missing path")
        if not isinstance(cache.get("qwen_jsonl"), str):
            errors.append("cache.qwen_jsonl: missing path")
        preparation = _as_mapping(cache.get("preparation_manifest"))
        if preparation is None:
            errors.append("cache.preparation_manifest: missing object")
        else:
            expected_preparation_keys = {"path", "sha256", "payload"}
            if set(preparation) != expected_preparation_keys:
                errors.append(
                    "cache.preparation_manifest: expected exactly "
                    f"{sorted(expected_preparation_keys)!r}, found "
                    f"{sorted(preparation)!r}"
                )
            if not isinstance(preparation.get("path"), str) or not preparation.get(
                "path"
            ):
                errors.append("cache.preparation_manifest.path: missing path")
            preparation_sha = preparation.get("sha256")
            if not isinstance(preparation_sha, str) or not HEX_SHA256.fullmatch(
                preparation_sha
            ):
                errors.append(
                    "cache.preparation_manifest.sha256: invalid SHA-256"
                )
            if _as_mapping(preparation.get("payload")) is None:
                errors.append(
                    "cache.preparation_manifest.payload: missing object"
                )

    factual_arrays = _as_mapping(manifest.get("factual_arrays"))
    if factual_arrays is None:
        errors.append("factual_arrays: missing object")
    else:
        expected_factual = set(FACTUAL_ARRAY_NAMES)
        if set(factual_arrays) != expected_factual:
            errors.append(
                "factual_arrays: expected exactly "
                f"{sorted(expected_factual)!r}, found "
                f"{sorted(factual_arrays)!r}"
            )
        for name, entry in factual_arrays.items():
            if _as_mapping(entry) is None:
                errors.append(f"factual_arrays.{name}: expected object")

    foil_artifacts = _as_mapping(manifest.get("foil_artifacts"))
    if foil_artifacts is None:
        errors.append("foil_artifacts: missing object")
    else:
        expected_foil = set() if arm == "A" else set(FOIL_ARTIFACT_NAMES)
        if set(foil_artifacts) != expected_foil:
            errors.append(
                "foil_artifacts: expected exactly "
                f"{sorted(expected_foil)!r}, found "
                f"{sorted(foil_artifacts)!r}"
            )
        for name, entry in foil_artifacts.items():
            if _as_mapping(entry) is None:
                errors.append(f"foil_artifacts.{name}: expected object")

    whitening = _as_mapping(manifest.get("whitening"))
    expected_whitening_keys = {
        "stage1_optTrain_localOnly",
        "stage2_trainOnly_localOnly",
    }
    if whitening is None:
        errors.append("whitening: missing object")
    else:
        if set(whitening) != expected_whitening_keys:
            errors.append(
                "whitening: expected exactly "
                f"{sorted(expected_whitening_keys)!r}, found "
                f"{sorted(whitening)!r}"
            )
        for label in expected_whitening_keys:
            entry = _as_mapping(whitening.get(label))
            if entry is None:
                continue
            for digest_key in ("sha256", "meta_sha256"):
                digest = entry.get(digest_key)
                if not isinstance(digest, str) or not HEX_SHA256.fullmatch(digest):
                    errors.append(
                        f"whitening.{label}.{digest_key}: invalid SHA-256"
                    )
            try:
                n_kept = int(entry.get("N_kept", -1))
            except (TypeError, ValueError):
                n_kept = -1
            if n_kept <= 0:
                errors.append(f"whitening.{label}.N_kept: must be positive")
            if entry.get("vectors_used") != n_kept * 5:
                errors.append(
                    f"whitening.{label}.vectors_used: must equal 5*N_kept"
                )


def _validate_stage1(
    manifest: Mapping[str, Any],
    *,
    dataset: str,
    arm: str,
    repo_root: Path,
    errors: list[str],
) -> dict[str, Any]:
    selected_epoch = manifest.get("selected_epoch")
    if (
        isinstance(selected_epoch, bool)
        or not isinstance(selected_epoch, int)
        or selected_epoch < 0
        or selected_epoch > 59
    ):
        errors.append(f"selected_epoch: expected integer in [0,59], found {selected_epoch!r}")
        selected_epoch_value: int | None = None
    else:
        selected_epoch_value = selected_epoch

    stage1_result = _resolve_declared_path(
        manifest.get("stage1_result_dir"), repo_root
    )
    if stage1_result is None or not stage1_result.is_dir():
        errors.append(f"stage1_result_dir: missing directory {stage1_result}")
    else:
        for filename in STAGE1_REQUIRED:
            if not (stage1_result / filename).is_file():
                errors.append(f"stage1 artifact missing: {stage1_result / filename}")
        forbidden = sorted(
            {
                path
                for pattern in STAGE1_FORBIDDEN_PATTERNS
                for path in stage1_result.glob(pattern)
            }
        )
        for path in forbidden:
            errors.append(f"stage1 official-test artifact forbidden: {path}")
        args_path = stage1_result / "args.txt"
        if args_path.is_file():
            _validate_saved_args(
                path=args_path,
                label="stage1",
                dataset=dataset,
                arm=arm,
                stage=1,
                selected_epoch=selected_epoch_value,
                manifest=manifest,
                repo_root=repo_root,
                errors=errors,
            )

    stage1_tag = manifest.get("stage1_tag")
    stage1_log = (
        repo_root / "logs" / f"{stage1_tag}.log"
        if isinstance(stage1_tag, str)
        else None
    )
    if stage1_log is None or not stage1_log.is_file():
        errors.append(f"stage1 log missing: {stage1_log}")
    else:
        text = stage1_log.read_text(encoding="utf-8", errors="replace")
        for marker in STAGE1_REQUIRED_MARKERS:
            if text.count(marker) != 1:
                errors.append(
                    f"stage1 log marker must occur exactly once: {marker!r}; "
                    f"count={text.count(marker)}"
                )
        for marker in STAGE1_FORBIDDEN_MARKERS:
            if marker in text:
                errors.append(f"stage1 official-test marker forbidden: {marker!r}")
        best_epochs = [
            int(value)
            for value in re.findall(
                r"new best mid-eval mAP=[0-9.]+ at epoch ([0-9]+)", text
            )
        ]
        if not best_epochs:
            errors.append("stage1 log has no held-out validation best-epoch marker")
        elif selected_epoch_value is not None and best_epochs[-1] != selected_epoch_value:
            errors.append(
                "selected_epoch does not match the last stage1 best-epoch marker: "
                f"{selected_epoch_value} != {best_epochs[-1]}"
            )

    if selected_epoch_value is not None:
        expected_s2_tag = (
            f"{dataset}_semantic_detail_{arm}_K128L4_"
            f"P0refit_e{selected_epoch_value}_s42"
        )
        _expect(
            errors, "stage2_tag", manifest.get("stage2_tag"), expected_s2_tag
        )
    return {
        "selected_epoch": selected_epoch_value,
        "result_dir": str(stage1_result) if stage1_result else None,
        "log": str(stage1_log) if stage1_log else None,
    }


def _metric(
    errors: list[str],
    payload: Mapping[str, Any],
    key: str,
    *,
    allow_none: bool = False,
) -> float | None:
    value = payload.get(key)
    if value is None and allow_none:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        errors.append(f"cell_result.{key}: expected finite number, found {value!r}")
        return None
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        errors.append(f"cell_result.{key}: expected value in [0,1], found {value!r}")
        return None
    return parsed


def _validate_stage2(
    manifest: Mapping[str, Any],
    *,
    dataset: str,
    arm: str,
    repo_root: Path,
    selected_epoch: int | None,
    errors: list[str],
) -> tuple[dict[str, Any], dict[str, float | int | None] | None]:
    stage2_result = _resolve_declared_path(
        manifest.get("stage2_result_dir"), repo_root
    )
    if stage2_result is None or not stage2_result.is_dir():
        errors.append(f"stage2_result_dir: missing directory {stage2_result}")
    else:
        for filename in STAGE2_REQUIRED:
            if not (stage2_result / filename).is_file():
                errors.append(f"stage2 artifact missing: {stage2_result / filename}")
        args_path = stage2_result / "args.txt"
        if args_path.is_file():
            _validate_saved_args(
                path=args_path,
                label="stage2",
                dataset=dataset,
                arm=arm,
                stage=2,
                selected_epoch=selected_epoch,
                manifest=manifest,
                repo_root=repo_root,
                errors=errors,
            )

    stage2_tag = manifest.get("stage2_tag")
    stage2_log = (
        repo_root / "logs" / f"{stage2_tag}.log"
        if isinstance(stage2_tag, str)
        else None
    )
    if stage2_log is None or not stage2_log.is_file():
        errors.append(f"stage2 log missing: {stage2_log}")
    else:
        text = stage2_log.read_text(encoding="utf-8", errors="replace")
        required_markers = [
            "Final checkpoint saved",
            "[p0-stage2] OFFICIAL TEST ISOLATED",
            "[final-eval] PRESERVE training text_whiten across cache override",
            "[final-eval] running extraction",
            "[final-eval] running evaluation",
            "[CELL-RESULT]",
        ]
        if selected_epoch is not None:
            required_markers.append(
                f"[refit-stop] stopping after epoch {selected_epoch}"
            )
        for marker in required_markers:
            if text.count(marker) != 1:
                errors.append(
                    f"stage2 log marker must occur exactly once: {marker!r}; "
                    f"count={text.count(marker)}"
                )
        if "[p0-stage1] SKIP official-test extraction/evaluation" in text:
            errors.append("stage2 log contains the stage1 official-test skip marker")
        if "new best mid-eval mAP=" in text:
            errors.append("stage2 log contains a checkpoint-selection marker")

    metrics: dict[str, float | int | None] | None = None
    if stage2_result is not None and stage2_result.is_dir():
        cell_path = stage2_result / "cell_result.json"
        bio_path = stage2_result / "evaluation_siglip2_base_bioproj.json"
        if cell_path.is_file():
            try:
                cell_raw = _load_json(cell_path)
            except (OSError, json.JSONDecodeError) as error:
                errors.append(f"cannot load {cell_path}: {error}")
                cell = None
            else:
                cell = _as_mapping(cell_raw)
                if cell is None:
                    errors.append(f"{cell_path}: root must be an object")
            if cell is not None:
                _expect(
                    errors,
                    "cell_result.dataset",
                    cell.get("dataset"),
                    DATASETS[dataset]["canonical"],
                )
                _expect(errors, "cell_result.K", cell.get("K"), 128)
                _expect(
                    errors,
                    "cell_result.map_r_cutoff",
                    cell.get("map_r_cutoff"),
                    DATASETS[dataset]["map_r"],
                )
                _expect_close(
                    errors,
                    "cell_result.gc_min_frac",
                    cell.get("gc_min_frac"),
                    0.416,
                )
                _expect_close(
                    errors,
                    "cell_result.gc_max_frac",
                    cell.get("gc_max_frac"),
                    0.584,
                )
                map_r = _metric(errors, cell, "mAP_at_R_bioproj")
                full_map = _metric(errors, cell, "full_mAP_bioproj")
                pre_map = _metric(
                    errors, cell, "full_mAP_pre_projection", allow_none=False
                )
                unique = _metric(errors, cell, "DNA_unique_DB")
                metrics = {
                    "mAP_at_R_bioproj": map_r,
                    "full_mAP_bioproj": full_map,
                    "full_mAP_pre_projection": pre_map,
                    "DNA_unique_DB": unique,
                    "map_r_cutoff": DATASETS[dataset]["map_r"],
                }
                embedded = _as_mapping(manifest.get("cell_result"))
                if embedded is None:
                    errors.append("manifest.cell_result: missing object")
                elif dict(embedded) != dict(cell):
                    errors.append(
                        "manifest.cell_result does not exactly match "
                        "stage2 cell_result.json"
                    )

                if bio_path.is_file():
                    try:
                        bio_raw = _load_json(bio_path)
                    except (OSError, json.JSONDecodeError) as error:
                        errors.append(f"cannot load {bio_path}: {error}")
                        bio = None
                    else:
                        bio = _as_mapping(bio_raw)
                        if bio is None:
                            errors.append(f"{bio_path}: root must be an object")
                    if bio is not None:
                        _expect_close(
                            errors,
                            "bio.mAP_at_R",
                            bio.get("mAP_at_R"),
                            float(map_r) if map_r is not None else math.nan,
                        )
                        _expect_close(
                            errors,
                            "bio.mAP",
                            bio.get("mAP"),
                            float(full_map) if full_map is not None else math.nan,
                        )
                        _expect_close(
                            errors,
                            "bio.unique_code_ratio",
                            bio.get("unique_code_ratio"),
                            float(unique) if unique is not None else math.nan,
                        )
                        _expect(
                            errors,
                            "bio.mAP_R_cutoff",
                            bio.get("mAP_R_cutoff"),
                            DATASETS[dataset]["map_r"],
                        )
                        bio_stats = _as_mapping(bio.get("bio_stats"))
                        if bio_stats is None:
                            errors.append("bio.bio_stats: missing object")
                        elif pre_map is not None:
                            _expect_close(
                                errors,
                                "bio.bio_stats.mAP_pre_projection",
                                bio_stats.get("mAP_pre_projection"),
                                float(pre_map),
                            )

    return (
        {
            "result_dir": str(stage2_result) if stage2_result else None,
            "log": str(stage2_log) if stage2_log else None,
        },
        metrics,
    )


def _validate_one(
    manifest_path: Path,
    *,
    dataset: str,
    arm: str,
    repo_root: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        raw = _load_json(manifest_path)
    except (OSError, json.JSONDecodeError) as error:
        return {
            "dataset": dataset,
            "arm": arm,
            "manifest": str(manifest_path),
            "manifest_status": "unreadable",
            "validation_status": "invalid",
            "errors": [f"cannot load manifest: {error}"],
            "metrics": None,
        }
    manifest = _as_mapping(raw)
    if manifest is None:
        return {
            "dataset": dataset,
            "arm": arm,
            "manifest": str(manifest_path),
            "manifest_status": "malformed",
            "validation_status": "invalid",
            "errors": ["manifest root must be an object"],
            "metrics": None,
        }

    _validate_base_manifest(
        manifest,
        dataset=dataset,
        arm=arm,
        repo_root=repo_root,
        errors=errors,
    )
    status = manifest.get("status")
    if status not in STATUS_ORDER:
        errors.append(
            f"status: expected one of {STATUS_ORDER!r}, found {status!r}"
        )

    stage1: dict[str, Any] | None = None
    stage2: dict[str, Any] | None = None
    metrics = None
    if status in ("stage1_complete", "complete"):
        stage1 = _validate_stage1(
            manifest,
            dataset=dataset,
            arm=arm,
            repo_root=repo_root,
            errors=errors,
        )
    if status == "complete":
        stage2, metrics = _validate_stage2(
            manifest,
            dataset=dataset,
            arm=arm,
            repo_root=repo_root,
            selected_epoch=None if stage1 is None else stage1["selected_epoch"],
            errors=errors,
        )

    validation_status = (
        "invalid"
        if errors
        else ("complete_valid" if status == "complete" else "incomplete_valid")
    )
    return {
        "dataset": dataset,
        "arm": arm,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _file_sha256(manifest_path),
        "manifest_status": status,
        "validation_status": validation_status,
        "errors": errors,
        "stage1": stage1,
        "stage2": stage2,
        "metrics": metrics,
        "_manifest_payload": dict(manifest),
    }


def _pair_comparable_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    cache = _as_mapping(manifest.get("cache")) or {}
    return {
        "launcher": manifest.get("launcher"),
        "cache": {
            key: cache.get(key)
            for key in ("base", "foil_overlay", "evaluation", "qwen_jsonl")
        },
        "cache_preparation": cache.get("preparation_manifest"),
        "factual_arrays": manifest.get("factual_arrays"),
        "whitening": manifest.get("whitening"),
        "source_sha256": manifest.get("source_sha256"),
    }


def _ab_pair_comparable_payload(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize only the intentional legacy/AB runner provenance split."""

    payload = _pair_comparable_payload(manifest)
    source_sha = dict(_as_mapping(payload.get("source_sha256")) or {})
    for runner in LEGACY_RUNNER_SOURCE_FILES + AB_RUNNER_SOURCE_FILES:
        source_sha.pop(runner, None)
    payload["source_sha256"] = source_sha
    return payload


def _cross_validate(
    records: dict[str, dict[str, Any]],
    *,
    arms: Sequence[str] = ARM_ORDER,
) -> None:
    loaded = [
        record
        for record in records.values()
        if isinstance(record.get("_manifest_payload"), Mapping)
    ]
    source_profiles = {
        "legacy": [
            record for record in loaded if record.get("arm") in ("A", "ABC")
        ],
        "AB": [record for record in loaded if record.get("arm") == "AB"],
    }
    for profile, profile_records in source_profiles.items():
        common_snapshots: dict[str, list[dict[str, Any]]] = {}
        for record in profile_records:
            manifest = record["_manifest_payload"]
            source_sha = dict(
                _as_mapping(manifest.get("source_sha256")) or {}
            )
            launcher = manifest.get("launcher")
            if isinstance(launcher, str):
                source_sha.pop(launcher, None)
            fingerprint = _canonical_sha256(source_sha)
            common_snapshots.setdefault(fingerprint, []).append(record)
        if len(common_snapshots) > 1:
            versions = sorted(common_snapshots)
            message = (
                f"cross-cell {profile} source path-set/SHA drift: "
                f"fingerprints={versions!r}"
            )
            for record in profile_records:
                record["errors"].append(message)

    cross_cell_source_files = (
        set(SHARED_SOURCE_FILES)
        | set(LEGACY_RUNNER_SOURCE_FILES)
        | set(AB_RUNNER_SOURCE_FILES)
    )
    for source_path in sorted(cross_cell_source_files):
        by_digest: dict[str, list[dict[str, Any]]] = {}
        for record in loaded:
            manifest = record["_manifest_payload"]
            source_sha = _as_mapping(manifest.get("source_sha256")) or {}
            digest = source_sha.get(source_path)
            if isinstance(digest, str):
                by_digest.setdefault(digest, []).append(record)
        if len(by_digest) > 1:
            versions = sorted(by_digest)
            message = (
                f"cross-cell source drift for {source_path}: {versions!r}"
            )
            for records_with_digest in by_digest.values():
                for record in records_with_digest:
                    record["errors"].append(message)

    requested_arms = tuple(arms)
    for dataset in DATASET_ORDER:
        if "A" not in requested_arms or "ABC" not in requested_arms:
            continue
        a = records[f"{dataset}:A"]
        abc = records[f"{dataset}:ABC"]
        if not (
            isinstance(a.get("_manifest_payload"), Mapping)
            and isinstance(abc.get("_manifest_payload"), Mapping)
        ):
            continue
        left = _pair_comparable_payload(a["_manifest_payload"])
        right = _pair_comparable_payload(abc["_manifest_payload"])
        if left != right:
            message = (
                "A/ABC protocol identity mismatch outside the intended "
                "semantic-detail arm"
            )
            a["errors"].append(message)
            abc["errors"].append(message)

    if "AB" in requested_arms:
        for dataset in DATASET_ORDER:
            arm_records = [
                records[f"{dataset}:{arm}"] for arm in EXTENDED_ARM_ORDER
            ]
            if not all(
                isinstance(record.get("_manifest_payload"), Mapping)
                for record in arm_records
            ):
                continue
            comparable = [
                _ab_pair_comparable_payload(record["_manifest_payload"])
                for record in arm_records
            ]
            if any(payload != comparable[0] for payload in comparable[1:]):
                message = (
                    "A/AB/ABC protocol identity mismatch outside the "
                    "intended semantic-detail arm and runner provenance"
                )
                for record in arm_records:
                    record["errors"].append(message)

            ab = records[f"{dataset}:AB"]
            abc = records[f"{dataset}:ABC"]
            ab_manifest = ab["_manifest_payload"]
            abc_manifest = abc["_manifest_payload"]
            if ab_manifest.get("foil_artifacts") != abc_manifest.get(
                "foil_artifacts"
            ):
                message = (
                    "AB/ABC foil-artifact identity mismatch outside the "
                    "intended C objective"
                )
                ab["errors"].append(message)
                abc["errors"].append(message)

    for record in loaded:
        if record["errors"]:
            record["validation_status"] = "invalid"


def _load_incumbent(
    dataset: str, repo_root: Path, errors: list[str]
) -> dict[str, Any]:
    spec = DATASETS[dataset]
    path = (repo_root / spec["incumbent_result"]).resolve()
    result = {
        "source": str(path),
        "mAP_at_R_bioproj": spec["incumbent_map"],
        "DNA_unique_DB": spec["incumbent_unique"],
        "map_r_cutoff": spec["map_r"],
    }
    if not path.is_file():
        errors.append(f"archived incumbent is missing: {path}")
        return result
    try:
        raw = _load_json(path)
    except (OSError, json.JSONDecodeError) as error:
        errors.append(f"cannot load archived incumbent {path}: {error}")
        return result
    payload = _as_mapping(raw)
    if payload is None:
        errors.append(f"archived incumbent root is not an object: {path}")
        return result
    _expect(
        errors,
        f"incumbent.{dataset}.dataset",
        payload.get("dataset"),
        spec["canonical"],
    )
    _expect(errors, f"incumbent.{dataset}.K", payload.get("K"), 128)
    _expect(
        errors,
        f"incumbent.{dataset}.map_r_cutoff",
        payload.get("map_r_cutoff"),
        spec["map_r"],
    )
    _expect_close(
        errors,
        f"incumbent.{dataset}.mAP_at_R_bioproj",
        payload.get("mAP_at_R_bioproj"),
        spec["incumbent_map"],
    )
    _expect_close(
        errors,
        f"incumbent.{dataset}.DNA_unique_DB",
        payload.get("DNA_unique_DB"),
        spec["incumbent_unique"],
    )
    result["source_sha256"] = _file_sha256(path)
    return result


def _delta(
    left: Mapping[str, Any] | None,
    right: Mapping[str, Any] | None,
) -> dict[str, float] | None:
    if left is None or right is None:
        return None
    left_map = left.get("mAP_at_R_bioproj")
    right_map = right.get("mAP_at_R_bioproj")
    left_unique = left.get("DNA_unique_DB")
    right_unique = right.get("DNA_unique_DB")
    if any(
        value is None
        for value in (left_map, right_map, left_unique, right_unique)
    ):
        return None
    return {
        "mAP_at_R_bioproj": float(left_map) - float(right_map),
        "DNA_unique_DB": float(left_unique) - float(right_unique),
    }


def _public_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if not key.startswith("_")
    }


def aggregate(
    *,
    manifest_dir: Path,
    repo_root: Path,
    arms: Sequence[str] = ARM_ORDER,
) -> tuple[dict[str, Any], bool, bool]:
    requested_arms = tuple(arms)
    if requested_arms not in (ARM_ORDER, EXTENDED_ARM_ORDER):
        raise ValueError(
            "arms must be exactly "
            f"{ARM_ORDER!r} or {EXTENDED_ARM_ORDER!r}, "
            f"found {requested_arms!r}"
        )
    records: dict[str, dict[str, Any]] = {}
    for dataset in DATASET_ORDER:
        for arm in requested_arms:
            key = f"{dataset}:{arm}"
            path = manifest_dir / f"{dataset}_{arm}_K128L4_p0.json"
            if not path.is_file():
                records[key] = {
                    "dataset": dataset,
                    "arm": arm,
                    "manifest": str(path.resolve()),
                    "manifest_status": "missing",
                    "validation_status": "missing",
                    "errors": [],
                    "stage1": None,
                    "stage2": None,
                    "metrics": None,
                }
            else:
                records[key] = _validate_one(
                    path,
                    dataset=dataset,
                    arm=arm,
                    repo_root=repo_root,
                )

    _cross_validate(records, arms=requested_arms)

    incumbent_errors: list[str] = []
    incumbents = {
        dataset: _load_incumbent(dataset, repo_root, incumbent_errors)
        for dataset in DATASET_ORDER
    }

    comparisons: dict[str, Any] = {}
    for dataset in DATASET_ORDER:
        arm_metrics = {}
        for arm in requested_arms:
            record = records[f"{dataset}:{arm}"]
            arm_metrics[arm] = (
                record.get("metrics")
                if record.get("validation_status") == "complete_valid"
                else None
            )
        incumbent = incumbents[dataset]
        if requested_arms == ARM_ORDER:
            comparisons[dataset] = {
                "A": arm_metrics["A"],
                "ABC": arm_metrics["ABC"],
                "ABC_minus_A": _delta(
                    arm_metrics["ABC"], arm_metrics["A"]
                ),
                "archived_incumbent": incumbent,
                "ABC_minus_archived_incumbent": _delta(
                    arm_metrics["ABC"], incumbent
                ),
            }
        else:
            comparisons[dataset] = {
                **arm_metrics,
                "AB_minus_A": _delta(
                    arm_metrics["AB"], arm_metrics["A"]
                ),
                "ABC_minus_AB": _delta(
                    arm_metrics["ABC"], arm_metrics["AB"]
                ),
                "ABC_minus_A": _delta(
                    arm_metrics["ABC"], arm_metrics["A"]
                ),
                "archived_incumbent": incumbent,
                "A_minus_archived_incumbent": _delta(
                    arm_metrics["A"], incumbent
                ),
                "AB_minus_archived_incumbent": _delta(
                    arm_metrics["AB"], incumbent
                ),
                "ABC_minus_archived_incumbent": _delta(
                    arm_metrics["ABC"], incumbent
                ),
            }

    complete = sum(
        record["validation_status"] == "complete_valid"
        for record in records.values()
    )
    missing = sum(
        record["validation_status"] == "missing"
        for record in records.values()
    )
    incomplete = sum(
        record["validation_status"] == "incomplete_valid"
        for record in records.values()
    )
    invalid = sum(
        record["validation_status"] == "invalid"
        for record in records.values()
    )
    has_validation_errors = invalid > 0 or bool(incumbent_errors)
    expected_cells = len(DATASET_ORDER) * len(requested_arms)
    table_ready = (
        complete == expected_cells and not has_validation_errors
    )

    protocol_identity = _protocol_core()
    common_source_paths = sorted(
        {
            path
            for record in records.values()
            if isinstance(record.get("_manifest_payload"), Mapping)
            for path in (
                _as_mapping(
                    (_as_mapping(record.get("_manifest_payload")) or {}).get(
                        "source_sha256"
                    )
                )
                or {}
            )
            if path
            != (_as_mapping(record.get("_manifest_payload")) or {}).get(
                "launcher"
            )
        }
    )
    common_source_versions = {
        source_path: sorted(
            {
                str(
                    (_as_mapping(record.get("_manifest_payload")) or {})
                    .get("source_sha256", {})
                    .get(source_path)
                )
                for record in records.values()
                if isinstance(record.get("_manifest_payload"), Mapping)
                and isinstance(
                    (_as_mapping(record.get("_manifest_payload")) or {})
                    .get("source_sha256"),
                    Mapping,
                )
                and (
                    (_as_mapping(record.get("_manifest_payload")) or {})
                    .get("source_sha256", {})
                    .get(source_path)
                    is not None
                )
            }
        )
        for source_path in common_source_paths
    }
    payload = {
        "schema_version": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "matrix": {
            "datasets": list(DATASET_ORDER),
            "arms": list(requested_arms),
            "K": 128,
            "num_codons_per_codebook": 4,
            "dna_bases": 24,
            "seed": 42,
            "manifest_dir": str(manifest_dir.resolve()),
        },
        "protocol_identity": {
            "fixed_contract": protocol_identity,
            "fixed_contract_sha256": _canonical_sha256(protocol_identity),
            "common_source_versions": common_source_versions,
        },
        "summary": {
            "expected": expected_cells,
            "complete_valid": complete,
            "incomplete_valid": incomplete,
            "missing": missing,
            "invalid": invalid,
            "archived_incumbent_errors": incumbent_errors,
            "table_ready": table_ready,
        },
        "cells": {
            key: _public_record(record) for key, record in records.items()
        },
        "comparisons": comparisons,
    }
    return payload, table_ready, has_validation_errors


def _fmt_metric(value: Any) -> str:
    return "—" if value is None else f"{float(value):.4f}"


def _fmt_delta(value: Any) -> str:
    return "—" if value is None else f"{float(value):+.4f}"


def _cell_text(record: Mapping[str, Any]) -> str:
    metrics = _as_mapping(record.get("metrics"))
    if record.get("validation_status") != "complete_valid" or metrics is None:
        return f"`{record.get('manifest_status')}`"
    epoch = (_as_mapping(record.get("stage1")) or {}).get("selected_epoch")
    return (
        f"{_fmt_metric(metrics.get('mAP_at_R_bioproj'))} / "
        f"{_fmt_metric(metrics.get('DNA_unique_DB'))} (E*={epoch})"
    )


def _markdown(payload: Mapping[str, Any]) -> str:
    summary = _as_mapping(payload.get("summary")) or {}
    cells = _as_mapping(payload.get("cells")) or {}
    comparisons = _as_mapping(payload.get("comparisons")) or {}
    matrix = _as_mapping(payload.get("matrix")) or {}
    arms = tuple(matrix.get("arms") or ARM_ORDER)
    include_ab = arms == EXTENDED_ARM_ORDER
    lines = [
        "# Semantic-detail multi-dataset P0 aggregation",
        "",
        f"- Table ready: `{str(bool(summary.get('table_ready'))).lower()}`",
        f"- Complete-valid: {summary.get('complete_valid', 0)} / "
        f"{summary.get('expected', 8)}; incomplete: "
        f"{summary.get('incomplete_valid', 0)}; missing: "
        f"{summary.get('missing', 0)}; invalid: {summary.get('invalid', 0)}",
        "- Fixed contract: seed 42, K=128, 4 codons/codebook (24 bases), "
        "held-out-train E* selection, scratch full-train refit, mandatory "
        "bio-projection GC [0.416, 0.584].",
    ]
    if include_ab:
        lines.append(
            "- A is strict global-caption-free supervision; AB adds the own "
            "local minimal-pair foil; ABC additionally enables post-VQ "
            "DNA-bit CIBHash KL."
        )
        lines.extend(
            [
                "",
                "## Validated cells",
                "",
                "| Dataset | A: bio mAP@R / DNA-unique | "
                "AB: bio mAP@R / DNA-unique | "
                "ABC: bio mAP@R / DNA-unique |",
                "|---|---:|---:|---:|",
            ]
        )
        for dataset in DATASET_ORDER:
            a = _as_mapping(cells.get(f"{dataset}:A")) or {}
            ab = _as_mapping(cells.get(f"{dataset}:AB")) or {}
            abc = _as_mapping(cells.get(f"{dataset}:ABC")) or {}
            lines.append(
                f"| {DATASETS[dataset]['display']} | {_cell_text(a)} | "
                f"{_cell_text(ab)} | {_cell_text(abc)} |"
            )

        lines.extend(
            [
                "",
                "## Pairwise arm deltas",
                "",
                "| Dataset | AB−A mAP@R | AB−A unique | "
                "ABC−AB mAP@R | ABC−AB unique | "
                "ABC−A mAP@R | ABC−A unique |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for dataset in DATASET_ORDER:
            comparison = _as_mapping(comparisons.get(dataset)) or {}
            ab_a = _as_mapping(comparison.get("AB_minus_A")) or {}
            abc_ab = _as_mapping(comparison.get("ABC_minus_AB")) or {}
            abc_a = _as_mapping(comparison.get("ABC_minus_A")) or {}
            lines.append(
                f"| {DATASETS[dataset]['display']} | "
                f"{_fmt_delta(ab_a.get('mAP_at_R_bioproj'))} | "
                f"{_fmt_delta(ab_a.get('DNA_unique_DB'))} | "
                f"{_fmt_delta(abc_ab.get('mAP_at_R_bioproj'))} | "
                f"{_fmt_delta(abc_ab.get('DNA_unique_DB'))} | "
                f"{_fmt_delta(abc_a.get('mAP_at_R_bioproj'))} | "
                f"{_fmt_delta(abc_a.get('DNA_unique_DB'))} |"
            )

        lines.extend(
            [
                "",
                "## Requested arms versus archived K=128 · 24-base incumbent",
                "",
                "| Dataset | Arm | Archived: bio mAP@R / DNA-unique | "
                "Arm: bio mAP@R / DNA-unique | Δ mAP@R | Δ unique |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for dataset in DATASET_ORDER:
            comparison = _as_mapping(comparisons.get(dataset)) or {}
            incumbent = _as_mapping(
                comparison.get("archived_incumbent")
            ) or {}
            for arm in EXTENDED_ARM_ORDER:
                metrics = _as_mapping(comparison.get(arm)) or {}
                delta = _as_mapping(
                    comparison.get(f"{arm}_minus_archived_incumbent")
                ) or {}
                lines.append(
                    f"| {DATASETS[dataset]['display']} | {arm} | "
                    f"{_fmt_metric(incumbent.get('mAP_at_R_bioproj'))} / "
                    f"{_fmt_metric(incumbent.get('DNA_unique_DB'))} | "
                    f"{_fmt_metric(metrics.get('mAP_at_R_bioproj'))} / "
                    f"{_fmt_metric(metrics.get('DNA_unique_DB'))} | "
                    f"{_fmt_delta(delta.get('mAP_at_R_bioproj'))} | "
                    f"{_fmt_delta(delta.get('DNA_unique_DB'))} |"
                )
    else:
        lines.extend(
            [
                "- A is strict global-caption-free supervision. ABC adds the own "
                "local minimal-pair foil and post-VQ DNA-bit CIBHash KL.",
                "",
                "## Validated cells",
                "",
                "| Dataset | A: bio mAP@R / DNA-unique | "
                "ABC: bio mAP@R / DNA-unique | "
                "ABC−A mAP@R | ABC−A unique |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for dataset in DATASET_ORDER:
            comparison = _as_mapping(comparisons.get(dataset)) or {}
            delta = _as_mapping(comparison.get("ABC_minus_A")) or {}
            a = _as_mapping(cells.get(f"{dataset}:A")) or {}
            abc = _as_mapping(cells.get(f"{dataset}:ABC")) or {}
            lines.append(
                f"| {DATASETS[dataset]['display']} | {_cell_text(a)} | "
                f"{_cell_text(abc)} | "
                f"{_fmt_delta(delta.get('mAP_at_R_bioproj'))} | "
                f"{_fmt_delta(delta.get('DNA_unique_DB'))} |"
            )

        lines.extend(
            [
                "",
                "## ABC versus archived K=128 · 24-base incumbent",
                "",
                "| Dataset | Archived incumbent: bio mAP@R / DNA-unique | "
                "ABC: bio mAP@R / DNA-unique | ABC−incumbent mAP@R | "
                "ABC−incumbent unique |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for dataset in DATASET_ORDER:
            comparison = _as_mapping(comparisons.get(dataset)) or {}
            incumbent = _as_mapping(
                comparison.get("archived_incumbent")
            ) or {}
            abc = _as_mapping(comparison.get("ABC")) or {}
            delta = _as_mapping(
                comparison.get("ABC_minus_archived_incumbent")
            ) or {}
            lines.append(
                f"| {DATASETS[dataset]['display']} | "
                f"{_fmt_metric(incumbent.get('mAP_at_R_bioproj'))} / "
                f"{_fmt_metric(incumbent.get('DNA_unique_DB'))} | "
                f"{_fmt_metric(abc.get('mAP_at_R_bioproj'))} / "
                f"{_fmt_metric(abc.get('DNA_unique_DB'))} | "
                f"{_fmt_delta(delta.get('mAP_at_R_bioproj'))} | "
                f"{_fmt_delta(delta.get('DNA_unique_DB'))} |"
            )

    invalid = [
        (key, record)
        for key, record in cells.items()
        if isinstance(record, Mapping) and record.get("validation_status") == "invalid"
    ]
    pending = [
        (key, record)
        for key, record in cells.items()
        if isinstance(record, Mapping)
        and record.get("validation_status") in ("missing", "incomplete_valid")
    ]
    if invalid:
        lines.extend(["", "## Invalid cells", ""])
        for key, record in invalid:
            lines.append(f"- `{key}`")
            for error in record.get("errors", []):
                lines.append(f"  - {error}")
    if pending:
        lines.extend(["", "## Pending cells", ""])
        for key, record in pending:
            lines.append(
                f"- `{key}`: `{record.get('manifest_status')}`"
            )
    incumbent_errors = summary.get("archived_incumbent_errors")
    if isinstance(incumbent_errors, list) and incumbent_errors:
        lines.extend(["", "## Archived-incumbent validation errors", ""])
        lines.extend(f"- {error}" for error in incumbent_errors)
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the eight-cell A/ABC or twelve-cell A/AB/ABC "
            "semantic-detail P0 matrix and write JSON plus Markdown summaries."
        )
    )
    parser.add_argument(
        "--manifest-dir",
        default="artifacts/semantic_detail_multidataset/p0",
        help="Directory containing the per-cell manifests.",
    )
    parser.add_argument(
        "--include-ab",
        action="store_true",
        help=(
            "Require and aggregate A/AB/ABC (12 cells). By default only the "
            "legacy A/ABC matrix (8 cells) is requested."
        ),
    )
    parser.add_argument(
        "--out-json",
        default=None,
        help="Output JSON (default: MANIFEST_DIR/aggregate.json).",
    )
    parser.add_argument(
        "--out-markdown",
        default=None,
        help="Output Markdown (default: MANIFEST_DIR/aggregate.md).",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help=(
            "Write a progress report when cells are missing or unfinished. "
            "Protocol-invalid cells still produce a nonzero exit status."
        ),
    )
    parser.add_argument(
        "--repo-root",
        default=str(REPO),
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    requested_arms = EXTENDED_ARM_ORDER if args.include_ab else ARM_ORDER
    expected_cells = len(DATASET_ORDER) * len(requested_arms)
    manifest_dir = Path(args.manifest_dir)
    if not manifest_dir.is_absolute():
        manifest_dir = repo_root / manifest_dir
    manifest_dir = manifest_dir.resolve()
    out_json = (
        Path(args.out_json)
        if args.out_json
        else manifest_dir
        / ("aggregate_a_ab_abc.json" if args.include_ab else "aggregate.json")
    )
    out_markdown = (
        Path(args.out_markdown)
        if args.out_markdown
        else manifest_dir
        / ("aggregate_a_ab_abc.md" if args.include_ab else "aggregate.md")
    )
    if not out_json.is_absolute():
        out_json = (repo_root / out_json).resolve()
    if not out_markdown.is_absolute():
        out_markdown = (repo_root / out_markdown).resolve()
    if out_json == out_markdown:
        parser.error("--out-json and --out-markdown must be different paths")

    payload, table_ready, has_validation_errors = aggregate(
        manifest_dir=manifest_dir,
        repo_root=repo_root,
        arms=requested_arms,
    )
    summary = payload["summary"]
    if not table_ready and not args.allow_incomplete:
        required_label = "all twelve" if args.include_ab else "all eight"
        print(
            f"[semantic-p0-aggregate] REFUSED: {required_label} cells must be "
            "complete and valid; "
            f"complete={summary['complete_valid']}/{expected_cells} "
            f"incomplete={summary['incomplete_valid']} "
            f"missing={summary['missing']} invalid={summary['invalid']}. "
            "Use --allow-incomplete only for a progress report.",
            file=sys.stderr,
        )
        return 2

    json_text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    _atomic_write(out_json, json_text)
    _atomic_write(out_markdown, _markdown(payload))
    print(
        "[semantic-p0-aggregate] wrote "
        f"{out_json} and {out_markdown}; "
        f"complete={summary['complete_valid']}/{expected_cells} "
        f"table_ready={str(table_ready).lower()}"
    )
    if has_validation_errors:
        print(
            "[semantic-p0-aggregate] one or more completed/provenance "
            "records are invalid; no invalid metric was admitted.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
