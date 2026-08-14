#!/usr/bin/env python3
"""Run one native-DNA predecessor under the sealed P0 comparison protocol.

The driver deliberately has no resume mode.  One invocation owns a fresh cell
directory and performs exactly two trainer invocations:

1. a 90/10 train-only validation run that selects zero-based ``E*`` with raw
   18-base Hamming mAP@R; and
2. a scratch, full-train refit for exactly ``E* + 1`` epochs whose trainer
   invocation performs the sole terminal official-test extraction.

All raw, author-postprocessed, and common-DP scores are produced from that one
terminal extraction.  The driver only validates and hashes saved arrays after
the trainer exits; it never constructs an official test dataset itself.

The four historical comparison caches currently lack immutable transform/model
provenance.  They may therefore be run only with
``--allow-main-ineligible-diagnostic`` and remain ineligible for a paper main
table until provenance-complete caches are rebuilt.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from baseline.cache_provenance import memoized_sha256_file  # noqa: E402


DATASETS = ("Flickr25k", "MSCOCO", "NUSWIDE", "CIFAR10")
METHODS = ("bee2018", "bee2021", "koike2024", "koike2026")
PAPER_SEEDS = (42, 43, 44)
# Matched code length in bases. 18 was the 6-slot / 36-bit budget; the paper
# is 5 slots x 3 bases = 15 (30 bit) since 2026-08-11. Overridden by the env
# var rather than a flag because the constant is read at module scope by both
# this file and the matrix launcher, before argparse runs, and the two MUST
# agree or the matrix gate rejects every cell for base_length mismatch.
# F18: the length is a property of the ARTEFACT, not of this process. The
# module constant stays as the CLI default, but every validator takes a
# protocol argument so an 18-base manifest can be checked in a process
# configured for 15 -- which is what 18 native cases were failing on.
from dna_utils.native_protocol import (  # noqa: E402
    NativeProtocol, coerce_protocol, resolve_native_protocol)

DEFAULT_PROTOCOL = resolve_native_protocol()
MATCHED_LENGTH = DEFAULT_PROTOCOL.length_bases

#: 24 bases run through `run_native_dna_p0_24`, which injects a pipeline variant
#: and extra implementation paths into the protocol identity. The canonical
#: driver has neither, so a canonical 24-base run produces an identity whose
#: method lock can never be in the reviewed 24 set: the child trains and
#: extracts, and only then does the gate flip its return code. Refuse before
#: anything is launched.
_WRAPPER_ONLY_LENGTHS = {24: "scripts/run_native_dna_p0_24.py"}


def _refuse_wrapper_only_length() -> None:
    """Called from `main()`, before any child is launched.

    Import must stay cheap and total -- guarding at import is what made the
    env20 aggregator unable to print `--help`.
    """
    if WRAPPER_DISPATCH:
        return
    wrapper = _WRAPPER_ONLY_LENGTHS.get(MATCHED_LENGTH)
    if wrapper is None:
        return
    raise SystemExit(
        f"{MATCHED_LENGTH} bases must run through {wrapper}, not this "
        f"entrypoint. The canonical identity carries no pipeline variant, so a "
        f"canonical {MATCHED_LENGTH}-base run would finish training and "
        f"extraction and then be rejected by the method-protocol lock. Use "
        f"{wrapper} (or {wrapper.replace('.py', '_matrix.py')} for a matrix).")


#: Set by `run_native_dna_p0_24` / `run_native_dna_p0_matrix_24` on the module
#: object, not through the environment: an env var leaks into every child
#: process, which would silently disable this guard for anything the wrapper
#: launches.
WRAPPER_DISPATCH = False



def effective_protocol() -> "NativeProtocol":
    """The protocol THIS call runs under.

    `run_native_dna_p0_24` executes the canonical driver inside a context that
    patches `MATCHED_LENGTH` to 24. Binding to the import-time
    `DEFAULT_PROTOCOL` therefore reported length 24 and validated against 15 at
    the same time, so a fresh 24-base run checked its artefacts against the
    wrong contract. Reading the module attribute at call time is what makes the
    wrapper's patch visible.
    """
    return coerce_protocol(MATCHED_LENGTH)
VAL_RATIO = 0.1
VAL_SEED = 42
EVAL_PERIOD = 5
GC_MIN = 0.4
GC_MAX = 0.6
MAX_RUN = 3

DEFAULT_CACHE = {
    "Flickr25k": "cache/flickr25k_clip_v4plus",
    "MSCOCO": "cache/mscoco_clip_v4plus",
    "NUSWIDE": "cache/nuswide_clip",
    "CIFAR10": "cache/cifar10_clip",
}
MAP_AT_R = {
    "CIFAR10": 1000,
    "Flickr25k": 5000,
    "MSCOCO": 5000,
    "NUSWIDE": 5000,
}
SOURCE_PROFILES: dict[str, dict[str, Any]] = {
    "bee2018": {
        "horizon": 65,
        "batch_size": 500,
        "steps_per_epoch": 1000,
        "learning_rate": 1e-3,
        "optimizer": "adam",
        "grad_clip": 0.0,
        "pair_sampling": "random",
        "supervision": "train-only feature-distance pairs",
    },
    "bee2021": {
        "horizon": 100,
        "batch_size": 100,
        "steps_per_epoch": 1000,
        "learning_rate": 1e-3,
        "optimizer": "adagrad",
        "grad_clip": 0.0,
        "pair_sampling": "balanced",
        "supervision": "train-only feature-distance pairs",
    },
    "koike2024": {
        "horizon": 150,
        "batch_size": 128,
        "steps_per_epoch": None,
        "learning_rate": 1e-2,
        "optimizer": "adagrad",
        "grad_clip": 0.0,
        "pair_sampling": None,
        "supervision": "ground-truth train labels",
    },
    "koike2026": {
        "horizon": 1000,
        "batch_size": 128,
        "steps_per_epoch": None,
        "learning_rate": 1e-2,
        "optimizer": "adagrad",
        "grad_clip": 0.0,
        "pair_sampling": None,
        "supervision": "ground-truth train labels",
    },
}
SOURCE_REPRODUCTION_AUDIT: dict[str, dict[str, Any]] = {
    "bee2018": {
        "reference": "Stewart_et_al_DNA24_2018_paper",
        "implementation_basis": "paper_equations_no_official_source_release",
        "source_underspecified": [
            "optimizer",
            "learning_rate",
            "random_seed",
        ],
        "declared_choice": (
            "Adam at 1e-3 is a reimplementation choice; gradient clipping is "
            "disabled because the paper does not specify it."
        ),
    },
    "bee2021": {
        "reference": "Bee_et_al_Nature_Communications_2021_PRIMO",
        "official_source": {
            "url": "https://github.com/uwmisl/primo-similarity-search",
            "revision": "5cf3656b163e1ae49f01b39e30a7fba985552cbf",
        },
        "pair_sampling": (
            "official balanced-pool sampling with random truncation; a final "
            "batch is statistically rather than forcibly 50/50 balanced"
        ),
        # H6: a hard-coded "18 nt" made a 15-base manifest declare length=15
        # and a transfer to 18 nt at the same time. Deriving it from the process
        # default would be worse -- this string enters the method-protocol lock,
        # so the lock would then depend on an environment variable. The length
        # is already recorded per cell as `length` and
        # `protocol_identity.matched_length_bases`, so the adaptation is stated
        # without repeating it.
        "declared_adaptation": (
            "The official 80-nt yield predictor is frozen and transferred to "
            "the matched code length declared by this run; PRIMO's per-epoch "
            "NUPACK relabeling/predictor refit is not performed."
        ),
    },
    "koike2024": {
        "reference": "Koike_et_al_DATE_and_DAC_2024",
        "official_source": {
            "url": "https://github.com/tkoike-kuee/dna-triplet-network",
            "revision": "cb42f6b51cfab6e8e9f403f24e32643692fcb18c",
        },
        "equation_choice": (
            "Use the fully symmetric normalized L1 distance in DATE Eq. (2). "
            "The released helper's lower-triangle/empty-upper-triangle "
            "symmetrization defect is intentionally not reproduced."
        ),
    },
    "koike2026": {
        "reference": "Koike_et_al_IEEE_TCBB_2026",
        "official_source": {
            "url": (
                "https://github.com/tkoike-kuee/"
                "DNA-Encoder-under-Bioconstraints"
            ),
            "revision": "24e3f1b9fe0a522bbaa8e996c28c4a42f6993dcd",
        },
        "declared_adaptation": (
            "The source's validation-on-test EarlyStopping(patience=10, "
            "min_delta=1e-4) is replaced by the common train-only P0 "
            "candidate selection and scratch refit contract."
        ),
    },
}

IMPLEMENTATION_PATHS = (
    "scripts/run_native_dna_p0.py",
    "scripts/train_native_dna_baseline.py",
    "baseline/native_dna.py",
    "baseline/base_model.py",
    "baseline/cache_provenance.py",
    "dna_utils/bio_constraints.py",
    "val_split.py",
    "dataloaders.py",
)
EXTRACTION_NAMES = (
    "extract_query_neural_raw.npz",
    "extract_db_neural_raw.npz",
    "extract_query.npz",
    "extract_db.npz",
    "extract_query_bioproj.npz",
    "extract_db_bioproj.npz",
    "extract_train.npz",
)
STRICT_CACHE_BLOCKER = "legacy_cache_missing_strict_provenance"
PRIMO_TRANSFER_BLOCKER = "primo_frozen_predictor_length_transfer"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_REVISION_RE = re.compile(r"[0-9a-f]{40}")


def _absolute(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (REPO / candidate).resolve()


def _require_file(path: Path, purpose: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{purpose} is missing: {path}")


def _sha256(path: Path, *, memoized: bool = False) -> str:
    _require_file(path, "artifact")
    if memoized:
        return memoized_sha256_file(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_record(path: Path, *, memoized: bool = False) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved, memoized=memoized),
        "bytes": int(resolved.stat().st_size),
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return _sha256(path)


def _load_json(path: Path, purpose: str) -> dict[str, Any]:
    _require_file(path, purpose)
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{purpose} must be a JSON object: {path}")
    return payload


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _split_paths(dataset: str, setting: str, dataset_root: Path) -> list[Path]:
    if dataset == "CIFAR10":
        paths = [dataset_root / "CIFAR10" / "cifar-10-python.tar.gz"]
    else:
        root = dataset_root / dataset / setting
        paths = [root / name for name in ("train.txt", "test.txt", "database.txt")]
    for path in paths:
        _require_file(path, "dataset split/source artifact")
    return paths


def _strict_cache_provenance_reasons(meta: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not isinstance(meta.get("canonical_transform"), dict):
        reasons.append("meta.canonical_transform is absent")
    hf = meta.get("hf_provenance")
    if not isinstance(hf, dict):
        reasons.append("meta.hf_provenance is absent")
        return reasons
    revision = hf.get("model_revision")
    weight_path = hf.get("model_weight_file")
    weight_hash = hf.get("model_weight_sha256")
    if not isinstance(revision, str) or _REVISION_RE.fullmatch(revision) is None:
        reasons.append("hf_provenance.model_revision is not an immutable commit")
    if not isinstance(weight_path, str) or not weight_path:
        reasons.append("hf_provenance.model_weight_file is absent")
    if not isinstance(weight_hash, str) or _SHA256_RE.fullmatch(weight_hash) is None:
        reasons.append("hf_provenance.model_weight_sha256 is absent or malformed")
    elif isinstance(weight_path, str) and weight_path:
        resolved_weight = Path(weight_path).expanduser()
        if not resolved_weight.is_file():
            reasons.append("hf_provenance.model_weight_file is unavailable")
        elif _sha256(resolved_weight) != weight_hash:
            raise ValueError("cache HF model-weight SHA-256 mismatch")
    return reasons


def _audit_cache(cache_dir: Path) -> dict[str, Any]:
    """Validate and bind the three visual-only inputs consumed by native DNA."""
    paths = {
        "meta.json": cache_dir / "meta.json",
        "image_ids.json": cache_dir / "image_ids.json",
        "visual_global.f16.npy": cache_dir / "visual_global.f16.npy",
    }
    for name, path in paths.items():
        _require_file(path, f"cache {name}")
    meta = _load_json(paths["meta.json"], "cache metadata")
    with paths["image_ids.json"].open(encoding="utf-8") as handle:
        image_ids = json.load(handle)
    if (
        not isinstance(image_ids, list)
        or not all(isinstance(item, str) for item in image_ids)
        or len(image_ids) != len(set(image_ids))
    ):
        raise ValueError("cache image_ids.json must be a unique ordered string list")
    try:
        n = int(meta["N"])
        d_proj = int(meta["D_proj"])
        dtype = np.dtype(str(meta["dtype"]))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("cache meta lacks valid N/D_proj/dtype") from error
    if n <= 0 or d_proj <= 0 or n != len(image_ids):
        raise ValueError(
            f"cache dimensions/image order disagree: N={n}, D={d_proj}, "
            f"image_ids={len(image_ids)}"
        )
    visual = np.load(paths["visual_global.f16.npy"], mmap_mode="r")
    if visual.shape != (n, d_proj) or visual.dtype != dtype:
        raise ValueError(
            "cache visual_global has "
            f"{visual.shape}/{visual.dtype}, expected {(n, d_proj)}/{dtype}"
        )
    del visual

    provenance_reasons = _strict_cache_provenance_reasons(meta)
    blockers = [STRICT_CACHE_BLOCKER] if provenance_reasons else []
    artifacts = {
        name: _artifact_record(path, memoized=True)
        for name, path in paths.items()
    }
    return {
        "cache_dir": str(cache_dir.resolve()),
        "shape": [n, d_proj],
        "dtype": str(dtype),
        "backbone": meta.get("backbone"),
        "strict_provenance_complete": not provenance_reasons,
        "strict_provenance_reasons": provenance_reasons,
        "blockers": blockers,
        "artifacts": artifacts,
    }


def _protocol_identity(
    *,
    method: str,
    dataset: str,
    setting: str,
    seed: int,
    cache_audit: Mapping[str, Any],
    dataset_root: Path,
    predictor: Path | None,
    device: str,
    num_workers: int,
    extract_batch_size: int,
    query_chunk: int,
) -> tuple[dict[str, Any], str]:
    implementation = {
        relative: _artifact_record(REPO / relative)
        for relative in IMPLEMENTATION_PATHS
    }
    splits = {
        path.name: _artifact_record(path)
        for path in _split_paths(dataset, setting, dataset_root)
    }
    predictor_record = (
        None if predictor is None else _artifact_record(predictor)
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "method": method,
        "dataset": dataset,
        "setting": setting,
        "seed": seed,
        "matched_length_bases": MATCHED_LENGTH,
        "val_split_ratio": VAL_RATIO,
        "val_split_seed": VAL_SEED,
        "selection_metric": "val_neural_raw_mAP_at_R",
        "selection_distance": "base_hamming",
        "source_profile": SOURCE_PROFILES[method],
        "source_reproduction_audit": SOURCE_REPRODUCTION_AUDIT[method],
        "candidate_eval_period": EVAL_PERIOD,
        "candidate_epochs_zero_based": list(
            range(EVAL_PERIOD - 1, SOURCE_PROFILES[method]["horizon"], EVAL_PERIOD)
        ),
        "map_at_R": MAP_AT_R[dataset],
        "stage_contract": {
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
        },
        "common_bio_projection": {
            "algorithm": "minimum_hamming_dynamic_programming",
            "applied_to": ["query", "database"],
            "gc_min": GC_MIN,
            "gc_max": GC_MAX,
            "max_run": MAX_RUN,
        },
        "cache": cache_audit,
        "dataset_root": str(dataset_root),
        "split_artifacts": splits,
        "implementation_artifacts": implementation,
        "primo_predictor": predictor_record,
        "execution": {
            "device_request": device,
            "num_workers": num_workers,
            "extract_batch_size": extract_batch_size,
            "query_chunk": query_chunk,
            "runtime_versions": {
                "python": ".".join(map(str, sys.version_info[:3])),
                "numpy": _package_version("numpy"),
                "torch": _package_version("torch"),
            },
        },
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return payload, hashlib.sha256(canonical).hexdigest()


def _training_command(
    *,
    method: str,
    dataset: str,
    setting: str,
    dataset_root: Path,
    cache_dir: Path,
    output: Path,
    seed: int,
    epochs: int,
    eval_period: int,
    val_ratio: float,
    device: str,
    num_workers: int,
    extract_batch_size: int,
    query_chunk: int,
    predictor: Path | None,
    save_train_extract: bool = False,
    save_soft_probs: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        str(REPO / "scripts/train_native_dna_baseline.py"),
        "--method",
        method,
        "--dataset",
        dataset,
        "--setting",
        setting,
        "--dataset_root",
        str(dataset_root),
        "--cache_dir",
        str(cache_dir),
        "--out",
        str(output),
        "--length",
        str(MATCHED_LENGTH),
        "--epochs",
        str(epochs),
        "--eval_period",
        str(eval_period),
        "--val_split_ratio",
        str(val_ratio),
        "--val_split_seed",
        str(VAL_SEED),
        "--selection_metric",
        "neural_raw",
        "--gc_min",
        str(GC_MIN),
        "--gc_max",
        str(GC_MAX),
        "--max_run",
        str(MAX_RUN),
        "--device",
        device,
        "--seed",
        str(seed),
        "--num_workers",
        str(num_workers),
        "--extract_batch_size",
        str(extract_batch_size),
        "--query_chunk",
        str(query_chunk),
        "--grad_clip",
        str(SOURCE_PROFILES[method]["grad_clip"]),
    ]
    if method == "bee2021":
        if predictor is None:
            raise ValueError("bee2021 requires a frozen --primo-predictor-npz")
        command += ["--primo_predictor_npz", str(predictor)]
    elif predictor is not None:
        raise ValueError("--primo-predictor-npz is valid only for bee2021")
    if save_train_extract:
        command.append("--save_train_extract")
    if save_soft_probs:
        command.append("--save_soft_probs")
    return command


def _run(command: Sequence[str], *, dry_run: bool = False) -> None:
    print("+", shlex.join([str(item) for item in command]), flush=True)
    if not dry_run:
        subprocess.run([str(item) for item in command], cwd=REPO, check=True)


def _same_path(actual: object, expected: Path) -> bool:
    return isinstance(actual, str) and Path(actual).expanduser().resolve() == expected.resolve()


def _assert_config_value(
    config: Mapping[str, Any], key: str, expected: Any, stage: str
) -> None:
    actual = config.get(key)
    if isinstance(expected, float):
        matches = (
            isinstance(actual, (int, float))
            and math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
        )
    else:
        matches = actual == expected
    if not matches:
        raise ValueError(
            f"{stage} config {key}={actual!r}, expected {expected!r}"
        )


def _verify_common_config(
    config: Mapping[str, Any],
    *,
    stage: str,
    method: str,
    dataset: str,
    setting: str,
    seed: int,
    cache_dir: Path,
    dataset_root: Path,
    output: Path,
    epochs: int,
    eval_period: int,
    val_ratio: float,
    predictor: Path | None,
    save_train_extract: bool,
    save_soft_probs: bool,
    protocol: "NativeProtocol | int | None" = None,
) -> None:
    protocol = coerce_protocol(protocol)
    expected = {
        "method": method,
        "dataset": dataset,
        "setting": setting,
        "length": protocol.length_bases,
        "seed": seed,
        "epochs": epochs,
        "eval_period": eval_period,
        "val_split_ratio": val_ratio,
        "val_split_seed": VAL_SEED,
        "selection_metric": "neural_raw",
        "gc_min": GC_MIN,
        "gc_max": GC_MAX,
        "max_run": protocol.max_homopolymer_run,
        "map_at_r": MAP_AT_R[dataset],
        "weights": None,
        "allow_nonempty_out": False,
        "allow_checkpoint_context_mismatch": False,
        "save_train_extract": save_train_extract,
        "save_soft_probs": save_soft_probs,
    }
    profile = SOURCE_PROFILES[method]
    expected.update(
        {
            "batch_size": profile["batch_size"],
            "steps_per_epoch": profile["steps_per_epoch"],
            "lr": profile["learning_rate"],
            "optimizer": profile["optimizer"],
            "grad_clip": profile["grad_clip"],
            "bee_pair_sampling": profile["pair_sampling"],
        }
    )
    if method == "koike2026":
        expected.update(
            {
                "paper_hp_postprocess": True,
                "paper_hp_max_run": 4,
            }
        )
    for key, value in expected.items():
        _assert_config_value(config, key, value, stage)
    for key, actual, expected_path in (
        ("cache_dir", config.get("cache_dir"), cache_dir),
        ("dataset_root", config.get("dataset_root"), dataset_root),
        ("out", config.get("out"), output),
    ):
        if not _same_path(actual, expected_path):
            raise ValueError(
                f"{stage} config {key}={actual!r}, expected {expected_path}"
            )
    saved_predictor = config.get("primo_predictor_npz")
    if predictor is None:
        if saved_predictor not in (None, ""):
            raise ValueError(f"{stage} unexpectedly records a PRIMO predictor")
    elif not _same_path(saved_predictor, predictor):
        raise ValueError(f"{stage} PRIMO predictor path mismatch")


def _checkpoint_set_digest(hashes: Mapping[str, str]) -> str:
    canonical = json.dumps(
        hashes, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def _verify_split(split: Mapping[str, Any]) -> None:
    _assert_config_value(split, "ratio", VAL_RATIO, "stage1 split")
    _assert_config_value(split, "seed", VAL_SEED, "stage1 split")
    for key in ("n_full", "n_opt", "n_val"):
        if isinstance(split.get(key), bool) or not isinstance(split.get(key), int):
            raise ValueError(f"stage1 split {key} must be an integer")
    if split["n_full"] != split["n_opt"] + split["n_val"]:
        raise ValueError("stage1 split counts do not partition the train set")
    indices = split.get("val_indices")
    if (
        not isinstance(indices, list)
        or len(indices) != split["n_val"]
        or any(isinstance(value, bool) or not isinstance(value, int) for value in indices)
        or len(indices) != len(set(indices))
        or any(value < 0 or value >= split["n_full"] for value in indices)
    ):
        raise ValueError("stage1 val_indices are malformed")
    if not isinstance(split.get("strategy"), str) or not split["strategy"]:
        raise ValueError("stage1 split strategy is missing")


def _verify_selection(
    stage1_dir: Path,
    *,
    method: str,
    dataset: str,
    setting: str,
    seed: int,
    cache_dir: Path,
    dataset_root: Path,
    predictor: Path | None,
    protocol: "NativeProtocol | int | None" = None,
) -> dict[str, Any]:
    protocol = coerce_protocol(protocol)
    """Verify trainer-produced E* and bind every candidate checkpoint."""
    horizon = int(SOURCE_PROFILES[method]["horizon"])
    config_path = stage1_dir / "config.json"
    split_path = stage1_dir / "val_split.json"
    selection_path = stage1_dir / "selection.json"
    config = _load_json(config_path, "stage1 config")
    _verify_common_config(
        config,
        stage="stage1",
        method=method,
        dataset=dataset,
        setting=setting,
        seed=seed,
        cache_dir=cache_dir,
        dataset_root=dataset_root,
        output=stage1_dir,
        epochs=horizon,
        eval_period=EVAL_PERIOD,
        val_ratio=VAL_RATIO,
        predictor=predictor,
        save_train_extract=False,
        save_soft_probs=False,
        protocol=protocol,
    )
    split = _load_json(split_path, "stage1 validation split")
    _verify_split(split)
    selection = _load_json(selection_path, "stage1 selection")
    if selection.get("selection_metric") != "val_neural_raw_mAP_at_R":
        raise ValueError("stage1 selection did not use raw base-Hamming mAP@R")
    if selection.get("test_evaluated") is not False:
        raise ValueError("stage1 selection artifact does not prove test was untouched")
    if selection.get("split") != split:
        raise ValueError("stage1 selection split differs from val_split.json")
    if (stage1_dir / "evaluation_native_dna.json").exists() or any(
        stage1_dir.glob("extract_*.npz")
    ):
        raise ValueError("stage1 contains official-test evaluation/extraction artifacts")

    expected_epochs = list(range(EVAL_PERIOD - 1, horizon, EVAL_PERIOD))
    checkpoint_paths = sorted(stage1_dir.glob("epoch_*.pth"))
    evaluation_paths = sorted(stage1_dir.glob("eval_val_epoch_*.json"))
    expected_checkpoint_names = [f"epoch_{epoch:03d}.pth" for epoch in expected_epochs]
    expected_evaluation_names = [
        f"eval_val_epoch_{epoch:03d}.json" for epoch in expected_epochs
    ]
    if [path.name for path in checkpoint_paths] != expected_checkpoint_names:
        raise ValueError("stage1 checkpoint set does not match the fixed 5-epoch grid")
    if [path.name for path in evaluation_paths] != expected_evaluation_names:
        raise ValueError("stage1 validation artifacts do not match the checkpoint grid")

    scores: list[tuple[int, float]] = []
    evaluation_artifacts: dict[str, dict[str, Any]] = {}
    for epoch, evaluation_path, checkpoint_path in zip(
        expected_epochs, evaluation_paths, checkpoint_paths
    ):
        evaluation = _load_json(evaluation_path, "stage1 validation evaluation")
        if evaluation.get("epoch") != epoch:
            raise ValueError(f"validation artifact epoch mismatch: {evaluation_path}")
        if not _same_path(evaluation.get("checkpoint"), checkpoint_path):
            raise ValueError(f"validation checkpoint path mismatch: {evaluation_path}")
        raw = evaluation.get("neural_raw")
        if not isinstance(raw, dict):
            raise ValueError(f"validation artifact lacks neural_raw metrics: {evaluation_path}")
        score = raw.get("mAP_at_R")
        if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            raise ValueError(f"validation raw mAP@R is invalid: {evaluation_path}")
        scores.append((epoch, float(score)))
        evaluation_artifacts[evaluation_path.name] = _artifact_record(evaluation_path)

    expected_best_epoch, expected_best_score = max(
        scores, key=lambda item: item[1]
    )
    if selection.get("best_epoch_zero_based") != expected_best_epoch:
        raise ValueError("selection E* is not the earliest maximum raw mAP@R candidate")
    if selection.get("refit_epochs") != expected_best_epoch + 1:
        raise ValueError("selection refit_epochs is not E*+1")
    best_score = selection.get("best_score")
    if (
        not isinstance(best_score, (int, float))
        or not math.isclose(
            float(best_score), expected_best_score, rel_tol=0.0, abs_tol=1e-12
        )
    ):
        raise ValueError("selection best_score differs from the selected raw mAP@R")
    selected_checkpoint = stage1_dir / f"epoch_{expected_best_epoch:03d}.pth"
    if not _same_path(selection.get("checkpoint"), selected_checkpoint):
        raise ValueError("selection checkpoint path does not bind E*")

    checkpoint_hashes = {
        path.name: _sha256(path) for path in checkpoint_paths
    }
    return {
        "best_epoch_zero_based": expected_best_epoch,
        "refit_epochs": expected_best_epoch + 1,
        "best_val_raw_mAP_at_R": expected_best_score,
        "selection_artifact": _artifact_record(selection_path),
        "stage1_config": _artifact_record(config_path),
        "val_split_artifact": _artifact_record(split_path),
        "stage1_log": _artifact_record(stage1_dir / "log.csv"),
        "stage1_validation_artifacts": evaluation_artifacts,
        "stage1_checkpoint_sha256": checkpoint_hashes,
        "stage1_checkpoint_set_sha256": _checkpoint_set_digest(checkpoint_hashes),
        "selected_stage1_checkpoint": _artifact_record(selected_checkpoint),
    }


def _base_codes_from_npz(
    path: Path, *, protocol: "NativeProtocol | int | None" = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    protocol = coerce_protocol(protocol)
    _require_file(path, "extraction artifact")
    with np.load(path, allow_pickle=False) as payload:
        required = ("base_indices", "multi_hot_labels", "image_paths")
        missing = [name for name in required if name not in payload]
        if missing:
            raise ValueError(f"{path} lacks extraction fields {missing}")
        codes = np.asarray(payload["base_indices"])
        labels = np.asarray(payload["multi_hot_labels"])
        paths = np.asarray(payload["image_paths"])
    if codes.ndim != 2 or codes.shape[1] != protocol.length_bases:
        raise ValueError(
            f"{path} has invalid base-code shape {codes.shape}; expected "
            f"{protocol.length_bases} bases ({protocol.protocol_label})")
    if codes.shape[0] == 0:
        raise ValueError(f"{path} contains no extracted rows")
    if (
        not np.issubdtype(codes.dtype, np.integer)
        or np.any(codes < 0)
        or np.any(codes > 3)
    ):
        raise ValueError(f"{path} contains non-canonical base indices")
    if labels.ndim != 2 or labels.shape[0] != codes.shape[0]:
        raise ValueError(f"{path} labels do not align with codes")
    if paths.ndim != 1 or paths.shape[0] != codes.shape[0]:
        raise ValueError(f"{path} paths do not align with codes")
    return codes.astype(np.int8, copy=False), labels, paths


def _valid_bio_codes(
    codes: np.ndarray, *, protocol: "NativeProtocol | int | None" = None,
) -> np.ndarray:
    # F07/F18: counts come from the one central policy, keyed by the protocol's
    # own length rather than by whatever this process was configured for.
    protocol = coerce_protocol(protocol)
    gc_count = ((codes == 1) | (codes == 2)).sum(axis=1)
    lower = protocol.gc_min_count
    upper = protocol.gc_max_count
    max_run = protocol.max_homopolymer_run
    valid = (gc_count >= lower) & (gc_count <= upper)
    for row_index, row in enumerate(codes):
        run = longest = 1
        for position in range(1, len(row)):
            run = run + 1 if row[position] == row[position - 1] else 1
            longest = max(longest, run)
        if longest > max_run:
            valid[row_index] = False
    return valid


def _soft_probs_from_npz(
    path: Path, neural_raw_codes: np.ndarray
) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        if "soft_probs_atcg" not in payload:
            raise ValueError(f"{path} lacks required soft_probs_atcg")
        probabilities = np.asarray(payload["soft_probs_atcg"])
    if probabilities.shape != (*neural_raw_codes.shape, 4):
        raise ValueError(f"{path} soft probabilities have invalid shape")
    if (
        not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or not np.allclose(
            probabilities.astype(np.float32).sum(axis=-1),
            1.0,
            rtol=0.0,
            atol=2e-3,
        )
    ):
        raise ValueError(f"{path} soft probabilities are not finite simplices")
    # Encoder probabilities retain the predecessor A,T,C,G order.  Saved base
    # indices use the repository A,C,G,T order.
    canonical_to_paper = np.asarray([0, 2, 3, 1], dtype=np.int8)
    raw_paper = canonical_to_paper[neural_raw_codes]
    selected = np.take_along_axis(
        probabilities,
        raw_paper[..., None],
        axis=-1,
    )[..., 0]
    # Probabilities are saved as float16 after the float32 argmax was taken.
    # Permit only the possible half-precision tie/rounding gap.
    if np.any(
        probabilities.max(axis=-1).astype(np.float32)
        - selected.astype(np.float32)
        > 2e-3
    ):
        raise ValueError(f"{path} soft-probability maximum differs from neural raw")
    return probabilities


def _verify_aligned_extractions(
    refit_dir: Path,
    *,
    method: str,
    expected_counts: Mapping[str, int] | None = None,
    evaluation_counts: Mapping[str, int] | None = None,
    cache_cardinality: int | None = None,
    protocol: "NativeProtocol | int | None" = None,
) -> dict[str, dict[str, Any]]:
    protocol = coerce_protocol(protocol)
    from baseline.native_dna import project_codes_memoized

    paths_by_split: dict[str, np.ndarray] = {}
    for split in ("query", "db"):
        logical_split = "query" if split == "query" else "database"
        raw_path = refit_dir / f"extract_{split}_neural_raw.npz"
        deployment_path = refit_dir / f"extract_{split}.npz"
        projected_path = refit_dir / f"extract_{split}_bioproj.npz"
        raw_codes, raw_labels, raw_paths = _base_codes_from_npz(raw_path, protocol=protocol)
        deployment_codes, deployment_labels, deployment_paths = _base_codes_from_npz(
            deployment_path, protocol=protocol
        )
        projected_codes, projected_labels, projected_paths = _base_codes_from_npz(
            projected_path, protocol=protocol
        )
        if not (
            np.array_equal(raw_labels, deployment_labels)
            and np.array_equal(raw_labels, projected_labels)
            and np.array_equal(raw_paths, deployment_paths)
            and np.array_equal(raw_paths, projected_paths)
        ):
            raise ValueError(f"{split} raw/deployment/projected rows are not aligned")
        with np.load(deployment_path, allow_pickle=False) as payload:
            if "neural_raw_base_indices" not in payload:
                raise ValueError(
                    f"{deployment_path} lacks embedded neural_raw_base_indices"
                )
            embedded_raw = np.asarray(payload["neural_raw_base_indices"])
        if not np.array_equal(raw_codes, embedded_raw):
            raise ValueError(
                f"{split} neural-raw artifact differs from the terminal extraction"
            )
        raw_probabilities = _soft_probs_from_npz(raw_path, raw_codes)
        deployment_probabilities = _soft_probs_from_npz(
            deployment_path, raw_codes
        )
        if not np.array_equal(raw_probabilities, deployment_probabilities):
            raise ValueError(
                f"{split} raw/deployment soft probabilities differ"
            )
        if method != "koike2026" and not np.array_equal(
            raw_codes, deployment_codes
        ):
            raise ValueError(
                f"{split} deployment differs from neural raw outside TCBB"
            )
        if not _valid_bio_codes(projected_codes, protocol=protocol).all():
            raise ValueError(f"{split} projected extraction is not 100% compliant")
        recomputed = project_codes_memoized(
            deployment_codes,
            GC_MIN,
            GC_MAX,
            MAX_RUN,
        )["projected_codes"]
        if not np.array_equal(
            projected_codes,
            np.asarray(recomputed, dtype=np.int8),
        ):
            raise ValueError(
                f"{split} projected extraction is not the exact common-DP "
                "projection of deployment codes"
            )
        paths_by_split[logical_split] = raw_paths
        if expected_counts is not None:
            expected = expected_counts.get(logical_split)
            if expected is None or len(raw_codes) != expected:
                raise ValueError(
                    f"{logical_split} extraction row count={len(raw_codes)}, "
                    f"expected {expected!r}"
                )
        if evaluation_counts is not None:
            evaluated = evaluation_counts.get(logical_split)
            if evaluated is None or len(raw_codes) != evaluated:
                raise ValueError(
                    f"{logical_split} extraction row count={len(raw_codes)} "
                    f"differs from saved evaluation n={evaluated!r}"
                )

    train_path = refit_dir / "extract_train.npz"
    train_codes, _, train_paths = _base_codes_from_npz(train_path, protocol=protocol)
    with np.load(train_path, allow_pickle=False) as payload:
        if "neural_raw_base_indices" not in payload:
            raise ValueError(
                "train extraction lacks embedded neural_raw_base_indices"
            )
        train_raw = np.asarray(payload["neural_raw_base_indices"])
    if train_raw.shape != train_codes.shape:
        raise ValueError("train neural-raw/deployment shapes differ")
    _soft_probs_from_npz(train_path, train_raw)
    if method != "koike2026" and not np.array_equal(train_raw, train_codes):
        raise ValueError("train deployment differs from neural raw outside TCBB")
    paths_by_split["train"] = train_paths
    if expected_counts is not None:
        expected_train = expected_counts.get("train")
        if expected_train is None or len(train_codes) != expected_train:
            raise ValueError(
                f"train extraction row count={len(train_codes)}, "
                f"expected {expected_train!r}"
            )
    for split, paths in paths_by_split.items():
        if len(np.unique(paths)) != len(paths):
            raise ValueError(f"{split} extraction contains duplicate row identifiers")
    if cache_cardinality is not None:
        if (
            isinstance(cache_cardinality, bool)
            or not isinstance(cache_cardinality, int)
            or cache_cardinality <= 0
        ):
            raise ValueError("cache cardinality must be a positive integer")
        covered = set()
        for paths in paths_by_split.values():
            covered.update(map(str, paths.tolist()))
        if len(covered) != cache_cardinality:
            raise ValueError(
                "train/query/database extraction union covers "
                f"{len(covered)} unique rows, expected cache cardinality "
                f"{cache_cardinality}"
            )
    return {
        name: _artifact_record(refit_dir / name) for name in EXTRACTION_NAMES
    }


def _expected_split_cardinalities(
    dataset: str,
    setting: str,
    dataset_root: Path,
) -> dict[str, int]:
    if dataset == "CIFAR10":
        if setting != "setting1":
            raise ValueError(
                f"native-DNA P0 has no sealed CIFAR10 cardinality for {setting!r}"
            )
        return {"train": 5000, "query": 1000, "database": 59000}
    split_root = dataset_root / dataset / setting
    counts: dict[str, int] = {}
    for logical, filename in (
        ("train", "train.txt"),
        ("query", "test.txt"),
        ("database", "database.txt"),
    ):
        path = split_root / filename
        _require_file(path, f"{logical} split")
        with path.open(encoding="utf-8") as handle:
            counts[logical] = sum(1 for line in handle if line.strip())
        if counts[logical] <= 0:
            raise ValueError(f"{logical} split is empty: {path}")
    return counts


def _require_complete_post_compliance(evaluation: Mapping[str, Any]) -> dict[str, float]:
    compliance: dict[str, float] = {}
    for split in ("query", "database"):
        record = evaluation.get(split)
        if not isinstance(record, dict):
            raise ValueError(f"terminal evaluation lacks {split} metrics")
        value = record.get("post_compliance")
        if (
            not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not math.isclose(float(value), 1.0, rel_tol=0.0, abs_tol=1e-12)
        ):
            raise ValueError(
                f"{split} post-DP compliance is {value!r}, required exactly 1.0"
            )
        compliance[split] = float(value)
    return compliance


def _verify_refit(
    refit_dir: Path,
    *,
    method: str,
    dataset: str,
    setting: str,
    seed: int,
    cache_dir: Path,
    dataset_root: Path,
    predictor: Path | None,
    best_epoch: int,
    protocol: "NativeProtocol | int | None" = None,
) -> dict[str, Any]:
    protocol = coerce_protocol(protocol)
    refit_epochs = best_epoch + 1
    config_path = refit_dir / "config.json"
    config = _load_json(config_path, "refit config")
    _verify_common_config(
        config,
        stage="refit",
        method=method,
        dataset=dataset,
        setting=setting,
        seed=seed,
        cache_dir=cache_dir,
        dataset_root=dataset_root,
        output=refit_dir,
        epochs=refit_epochs,
        eval_period=refit_epochs,
        val_ratio=0.0,
        predictor=predictor,
        save_train_extract=True,
        save_soft_probs=True,
        protocol=protocol,
    )
    if (refit_dir / "selection.json").exists() or (
        refit_dir / "val_split.json"
    ).exists():
        raise ValueError("refit unexpectedly contains validation-selection artifacts")
    if any(refit_dir.glob("eval_val_epoch_*.json")):
        raise ValueError("refit performed validation-time model selection")

    checkpoints = sorted(refit_dir.glob("epoch_*.pth"))
    expected_checkpoint = refit_dir / f"epoch_{best_epoch:03d}.pth"
    if checkpoints != [expected_checkpoint]:
        raise ValueError(
            "refit must contain exactly the fixed final E* checkpoint; "
            f"found {[path.name for path in checkpoints]}"
        )
    evaluation_path = refit_dir / "evaluation_native_dna.json"
    evaluation = _load_json(evaluation_path, "terminal native-DNA evaluation")
    for key, expected in (
        ("method", method),
        ("dataset", dataset),
        ("length", protocol.length_bases),
    ):
        if evaluation.get(key) != expected:
            raise ValueError(
                f"terminal evaluation {key}={evaluation.get(key)!r}, "
                f"expected {expected!r}"
            )
    raw = evaluation.get("neural_raw")
    projected = evaluation.get("projected")
    if not isinstance(raw, dict) or not isinstance(projected, dict):
        raise ValueError("terminal evaluation lacks raw/projected retrieval metrics")
    for name, record in (("raw", raw), ("post-DP", projected)):
        value = record.get("mAP_at_R")
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"terminal {name} mAP@R is invalid")
    compliance = _require_complete_post_compliance(evaluation)
    evaluation_counts: dict[str, int] = {}
    for split in ("query", "database"):
        record = evaluation.get(split)
        value = None if not isinstance(record, dict) else record.get("n")
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"terminal evaluation {split}.n is invalid: {value!r}")
        evaluation_counts[split] = value
    cache_meta = _load_json(cache_dir / "meta.json", "cache metadata")
    cache_cardinality = cache_meta.get("N")
    if (
        isinstance(cache_cardinality, bool)
        or not isinstance(cache_cardinality, int)
        or cache_cardinality <= 0
    ):
        raise ValueError(
            f"cache metadata N is invalid: {cache_cardinality!r}"
        )
    extraction_artifacts = _verify_aligned_extractions(
        refit_dir,
        method=method,
        expected_counts=_expected_split_cardinalities(
            dataset,
            setting,
            dataset_root,
        ),
        evaluation_counts=evaluation_counts,
        cache_cardinality=cache_cardinality,
        protocol=protocol,
    )
    return {
        "refit_config": _artifact_record(config_path),
        "refit_log": _artifact_record(refit_dir / "log.csv"),
        "evaluation_native_dna": _artifact_record(evaluation_path),
        "final_checkpoint": _artifact_record(expected_checkpoint),
        "extraction_artifacts": extraction_artifacts,
        "post_compliance": compliance,
        "raw_mAP_at_R": float(raw["mAP_at_R"]),
        "post_dp_mAP_at_R": float(projected["mAP_at_R"]),
        "post_dp_database_unique_ratio": float(
            evaluation["database"]["unique_ratio_projected"]
        ),
        "resolved_device": config.get("device"),
    }


def _verify_immutable_inputs(
    protocol_identity: Mapping[str, Any],
    *,
    cache_dir: Path,
) -> None:
    """Detect cache/source mutation across the two long trainer invocations."""
    expected_cache = protocol_identity["cache"]["artifacts"]
    for name, record in expected_cache.items():
        actual = _sha256(cache_dir / name, memoized=True)
        if actual != record["sha256"]:
            raise ValueError(f"cache artifact changed during the cell: {name}")
    for record in protocol_identity["implementation_artifacts"].values():
        path = Path(record["path"])
        if _sha256(path) != record["sha256"]:
            raise ValueError(f"implementation changed during the cell: {path}")
    for record in protocol_identity["split_artifacts"].values():
        path = Path(record["path"])
        if _sha256(path) != record["sha256"]:
            raise ValueError(f"dataset split/source changed during the cell: {path}")
    predictor = protocol_identity.get("primo_predictor")
    if isinstance(predictor, dict):
        path = Path(predictor["path"])
        if _sha256(path) != predictor["sha256"]:
            raise ValueError("PRIMO predictor changed during the cell")


def _manifest(
    *,
    method: str,
    dataset: str,
    setting: str,
    seed: int,
    cell_root: Path,
    stage1_dir: Path,
    refit_dir: Path,
    protocol_identity: Mapping[str, Any],
    protocol_digest: str,
    blockers: Sequence[str],
    selection_audit: Mapping[str, Any],
    refit_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the aggregator-facing completed-run contract."""
    blocker_list = list(dict.fromkeys(str(item) for item in blockers))
    return {
        "schema_version": 1,
        "method": method,
        "dataset": dataset,
        "setting": setting,
        "seed": seed,
        "length": MATCHED_LENGTH,
        "base_length": MATCHED_LENGTH,
        "best_epoch_zero_based": selection_audit["best_epoch_zero_based"],
        "refit_epochs": selection_audit["refit_epochs"],
        "run_manifest_phase": "completed",
        "main_protocol_eligible": not blocker_list,
        "blockers": blocker_list,
        "has_blockers": bool(blocker_list),
        "main_eligibility_blockers": blocker_list,
        "test_used_for_selection": False,
        "evaluation_native_dna": refit_audit["evaluation_native_dna"],
        "final_checkpoint": refit_audit["final_checkpoint"],
        "protocol_digest_sha256": protocol_digest,
        "protocol_identity": protocol_identity,
        "source_reproduction_audit": protocol_identity.get(
            "source_reproduction_audit"
        ),
        "cache_artifacts": protocol_identity["cache"]["artifacts"],
        "implementation_artifacts": protocol_identity[
            "implementation_artifacts"
        ],
        "selection_artifact": selection_audit["selection_artifact"],
        "stage1_checkpoint_sha256": selection_audit[
            "stage1_checkpoint_sha256"
        ],
        "stage1_checkpoint_set_sha256": selection_audit[
            "stage1_checkpoint_set_sha256"
        ],
        "cell_root": str(cell_root.resolve()),
        "stage1_dir": str(stage1_dir.resolve()),
        "refit_dir": str(refit_dir.resolve()),
        "selection": selection_audit,
        "extraction_artifacts": refit_audit["extraction_artifacts"],
        "post_compliance": refit_audit["post_compliance"],
        "metrics": {
            "validation_raw_mAP_at_R": selection_audit[
                "best_val_raw_mAP_at_R"
            ],
            "test_neural_raw_mAP_at_R": refit_audit["raw_mAP_at_R"],
            "test_post_dp_mAP_at_R": refit_audit["post_dp_mAP_at_R"],
            "test_post_dp_database_unique_ratio": refit_audit[
                "post_dp_database_unique_ratio"
            ],
        },
        "refit_artifacts": {
            "config": refit_audit["refit_config"],
            "log": refit_audit["refit_log"],
        },
        "resolved_device": refit_audit["resolved_device"],
        "test_access_contract": {
            "stage1_test_evaluated": False,
            "refit_from_scratch": True,
            "refit_weights_argument": None,
            "terminal_trainer_invocations": 1,
            "terminal_official_test_extraction_passes": 1,
            "post_run_validation_reopens_official_test": False,
            "raw_and_post_dp_share_terminal_extraction": True,
            "full_train_extraction_saved_after_terminal_test": True,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one sealed native-DNA P0 selection/refit/test cell."
    )
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--setting", default="setting1")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument(
        "--result-root",
        default="result_native_dna/p0",
        help="Parent for the fresh cell directory; absolute /data paths are allowed.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, choices=PAPER_SEEDS, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--extract-batch-size", type=int, default=512)
    parser.add_argument("--query-chunk", type=int, default=64)
    parser.add_argument(
        "--primo-predictor-npz",
        default=None,
        help="Required frozen official predictor conversion for bee2021 only.",
    )
    parser.add_argument(
        "--allow-main-ineligible-diagnostic",
        action="store_true",
        help=(
            "Allow an explicitly diagnostic run with a legacy cache. "
            "Blockers remain recorded and main_protocol_eligible stays false."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Audit inputs and print the stage-1 command without creating outputs.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _refuse_wrapper_only_length()
    args = build_parser().parse_args(argv)
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.extract_batch_size <= 0 or args.query_chunk <= 0:
        raise ValueError("extraction batch/chunk sizes must be positive")

    cache_dir = _absolute(args.cache_dir or DEFAULT_CACHE[args.dataset])
    dataset_root = _absolute(args.dataset_root)
    result_root = _absolute(args.result_root)
    predictor = (
        None
        if args.primo_predictor_npz is None
        else _absolute(args.primo_predictor_npz)
    )
    if args.method == "bee2021":
        if predictor is None:
            raise ValueError("bee2021 requires --primo-predictor-npz")
        _require_file(predictor, "frozen PRIMO predictor")
    elif predictor is not None:
        raise ValueError("--primo-predictor-npz is valid only for bee2021")

    cache_audit = _audit_cache(cache_dir)
    blockers = list(cache_audit["blockers"])
    if args.method == "bee2021":
        blockers.append(PRIMO_TRANSFER_BLOCKER)
    if blockers and not args.allow_main_ineligible_diagnostic:
        raise ValueError(
            "native-DNA main-comparison eligibility checks failed: "
            + "; ".join(blockers)
            + ". Rebuild a provenance-complete visual cache, or pass "
            "--allow-main-ineligible-diagnostic for a labelled diagnostic run."
        )
    protocol_identity, protocol_digest = _protocol_identity(
        method=args.method,
        dataset=args.dataset,
        setting=args.setting,
        seed=args.seed,
        cache_audit=cache_audit,
        dataset_root=dataset_root,
        predictor=predictor,
        device=args.device,
        num_workers=args.num_workers,
        extract_batch_size=args.extract_batch_size,
        query_chunk=args.query_chunk,
    )
    dataset_slug = args.dataset.lower().replace("_", "")
    cell_name = (
        f"{args.method}_{dataset_slug}_{MATCHED_LENGTH}nt_"
        f"P0_seed{args.seed}_p{protocol_digest[:12]}"
    )
    cell_root = result_root / cell_name
    stage1_dir = cell_root / "stage1"
    refit_dir = cell_root / "refit"
    if cell_root.exists():
        raise FileExistsError(
            f"native-DNA P0 cell already exists: {cell_root}. "
            "Refusing to resume or mix artifacts; choose a fresh --result-root."
        )

    horizon = int(SOURCE_PROFILES[args.method]["horizon"])
    stage1_command = _training_command(
        method=args.method,
        dataset=args.dataset,
        setting=args.setting,
        dataset_root=dataset_root,
        cache_dir=cache_dir,
        output=stage1_dir,
        seed=args.seed,
        epochs=horizon,
        eval_period=EVAL_PERIOD,
        val_ratio=VAL_RATIO,
        device=args.device,
        num_workers=args.num_workers,
        extract_batch_size=args.extract_batch_size,
        query_chunk=args.query_chunk,
        predictor=predictor,
        save_train_extract=False,
        save_soft_probs=False,
    )
    if args.dry_run:
        print(json.dumps(
            {
                "cell_root": str(cell_root),
                "protocol_digest_sha256": protocol_digest,
                "blockers": blockers,
                "stage1_command": stage1_command,
                "refit_command": "data-dependent on validated E*",
            },
            indent=2,
        ))
        return 0

    result_root.mkdir(parents=True, exist_ok=True)
    cell_root.mkdir(parents=False, exist_ok=False)
    _atomic_write_json(
        cell_root / "native_p0_protocol_identity.json",
        {
            "protocol_digest_sha256": protocol_digest,
            "protocol_identity": protocol_identity,
            "blockers": blockers,
        },
    )
    _run(stage1_command)
    selection_audit = _verify_selection(
        stage1_dir,
        method=args.method,
        dataset=args.dataset,
        setting=args.setting,
        seed=args.seed,
        cache_dir=cache_dir,
        dataset_root=dataset_root,
        predictor=predictor,
        protocol=effective_protocol(),
    )
    best_epoch = int(selection_audit["best_epoch_zero_based"])
    refit_epochs = best_epoch + 1
    if refit_dir.exists():
        raise FileExistsError(
            f"refit output unexpectedly exists: {refit_dir}; refusing reuse"
        )
    refit_command = _training_command(
        method=args.method,
        dataset=args.dataset,
        setting=args.setting,
        dataset_root=dataset_root,
        cache_dir=cache_dir,
        output=refit_dir,
        seed=args.seed,
        epochs=refit_epochs,
        eval_period=refit_epochs,
        val_ratio=0.0,
        device=args.device,
        num_workers=args.num_workers,
        extract_batch_size=args.extract_batch_size,
        query_chunk=args.query_chunk,
        predictor=predictor,
        save_train_extract=True,
        save_soft_probs=True,
    )
    # This is the only subprocess in the entire driver that is permitted to
    # construct the official test/database datasets.
    _run(refit_command)
    refit_audit = _verify_refit(
        refit_dir,
        method=args.method,
        dataset=args.dataset,
        setting=args.setting,
        seed=args.seed,
        cache_dir=cache_dir,
        dataset_root=dataset_root,
        predictor=predictor,
        best_epoch=best_epoch,
        protocol=effective_protocol(),
    )
    selection_reaudit = _verify_selection(
        stage1_dir,
        method=args.method,
        dataset=args.dataset,
        setting=args.setting,
        seed=args.seed,
        cache_dir=cache_dir,
        dataset_root=dataset_root,
        predictor=predictor,
        protocol=effective_protocol(),
    )
    if selection_reaudit != selection_audit:
        raise ValueError(
            "stage1 selection/checkpoint artifacts changed during the refit"
        )
    _verify_immutable_inputs(protocol_identity, cache_dir=cache_dir)
    manifest = _manifest(
        method=args.method,
        dataset=args.dataset,
        setting=args.setting,
        seed=args.seed,
        cell_root=cell_root,
        stage1_dir=stage1_dir,
        refit_dir=refit_dir,
        protocol_identity=protocol_identity,
        protocol_digest=protocol_digest,
        blockers=blockers,
        selection_audit=selection_audit,
        refit_audit=refit_audit,
    )
    manifest_path = cell_root / "native_p0_run_manifest.json"
    _atomic_write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
