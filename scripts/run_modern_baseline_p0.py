#!/usr/bin/env python3
"""Run one audited modern hashing baseline under an explicit P0 protocol.

The paper-main default implements decision D6: train once on the complete
designated train split for the author-prescribed horizon ``H`` and use only the
terminal ``H-1`` checkpoint.  It has no validation selector and no scratch
refit.  The former held-out-selection + refit protocol remains available only
as the explicitly labelled ``validation_sensitivity`` mode and can never be
promoted to the paper-main pool.

Published 16/32/64-bit numbers are never copied or interpolated.  The runnable
variants are deliberately explicit because several papers and public releases
do not describe the same model.

Examples
--------
python scripts/run_modern_baseline_p0.py \
  --variant sdc-paper --dataset Flickr25k --device cuda:0

python scripts/run_modern_baseline_p0.py \
  --variant hhch --dataset MSCOCO --device cuda:0

CRH uses target training labels and is reported only in the supervised panel:

python scripts/run_modern_baseline_p0.py \
  --variant crh-supervised --dataset MSCOCO --device cuda:0

python scripts/run_modern_baseline_p0.py \
  --variant duheg --dataset Flickr25k --device cuda:0 \
  --duheg-noun-embeddings artifacts/duheg_selected_nouns_embeddings.npy \
  --duheg-asset-manifest artifacts/duheg_selected_nouns_manifest.json \
  --duheg-allow-unverified-selection

UMRCH is taxonomy-assisted (U2), not a strict visual-only U0 comparison:

python scripts/run_modern_baseline_p0.py \
  --variant umrch --dataset Flickr25k --device cuda:0 \
  --umrch-concept-embeddings artifacts/umrch_flickr_embeddings.npy \
  --umrch-vision-adapter artifacts/umrch_flickr_vision_adapter.npz \
  --umrch-asset-manifest artifacts/umrch_flickr_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import re
from typing import Mapping, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from baseline.asset_provenance import (
    DUHEG_PROMPT_TEMPLATES,
    load_and_verify_text_asset_manifest,
    verify_manifest_cache,
)
from baseline.cache_provenance import (
    DECODE_FAILURE_AUDIT_SCHEMA,
    DECODE_FAILURE_AUDIT_SCHEMA_VERSION,
    audit_cache_decode_failures,
    consumed_cache_artifact_names,
    hash_cache_artifacts,
    memoized_sha256_file,
    verify_cache_decode_failure_binding,
)
from baseline.execution_environment import (
    ExecutionEnvironmentError,
    capture_child_execution_environment,
    parse_gpu_assignment_json,
    require_exact_environment,
    verify_execution_environment,
)


DATASETS = ("Flickr25k", "MSCOCO", "NUSWIDE", "CIFAR10")
AUTHOR_FIXED_FINAL = "author_fixed_final"
VALIDATION_SENSITIVITY = "validation_sensitivity"
PROTOCOL_MODES = (AUTHOR_FIXED_FINAL, VALIDATION_SENSITIVITY)
AUTHOR_CHECKPOINT_POLICY = "author_horizon_last"
LEGACY_CHECKPOINT_POLICY = "validation_selected_scratch_refit"
AUTHOR_FIXED_SINGLE_STAGE = "author_fixed_single_stage"
P0_STAGE1_VAL_SELECTION = "P0_stage1_val_selection"
P0_STAGE2_REFIT_TEST = "P0_stage2_refit_test"
CIFAR10_CONSUMED_SOURCE_RELATIVE_PATHS = (
    *(f"CIFAR10/cifar-10-batches-py/data_batch_{index}"
      for index in range(1, 6)),
    "CIFAR10/cifar-10-batches-py/test_batch",
    "CIFAR10/cifar-10-batches-py/batches.meta",
)
# 30 added 2026-08-10 to match the 5-slot GroundedDNA variant (15 bases).
# Everything downstream is length-generic: `_base_length` is bit // 2 and
# apply_bio_projection takes GC as a FRACTION, so the window scales.
SUPPORTED_BITS = (30, 36, 40, 48)
DEFAULT_CACHE = {
    "Flickr25k": "cache/flickr25k_clip_v4plus_qwen3_tokens",
    "MSCOCO": "cache/mscoco_clip_v5b",
    "NUSWIDE": "cache/nuswide_clip_tokens",
    "CIFAR10": "cache/cifar10_clip",
}
VARIANTS = {
    "cibhash": {
        "method": "cibhash", "horizon": 60, "source_batch_size": 64,
        "eval_period": 5, "extra": (),
        "needs_aug": True, "needs_tokens": False, "tier": "U0",
    },
    "cimon": {
        "method": "cimon", "horizon": 150, "source_batch_size": 24,
        "eval_period": 5, "extra": (),
        "needs_aug": True, "needs_tokens": False, "tier": "U0",
    },
    "mls3rduh": {
        "method": "mls3rduh", "horizon": 150, "source_batch_size": 128,
        "eval_period": 5, "extra": (),
        "needs_aug": False, "needs_tokens": False, "tier": "U0",
    },
    "greedyhash": {
        "method": "greedyhash", "horizon": 60, "eval_period": 5,
        "extra": (), "needs_aug": False, "needs_tokens": False, "tier": "U0",
    },
    "bihalf": {
        "method": "bihalf",
        # The public image release provides dataset-specific scripts.  Its
        # repository contains no NUS-WIDE trainer, so NUS-WIDE uses the paper/
        # Flickr profile and is identified as an adaptation in the manifest.
        "horizon": {
            "Flickr25k": 100, "MSCOCO": 150,
            "CIFAR10": 300, "NUSWIDE": 100,
        },
        "eval_period": 5,
        "extra": (), "needs_aug": False, "needs_tokens": False, "tier": "U0",
    },
    "sdc-release": {
        "method": "sdc", "horizon": 100, "eval_period": 5,
        "extra": ("--sdc_variant", "release_no_cl"),
        "needs_aug": False, "needs_tokens": False, "tier": "U0",
    },
    "sdc-paper": {
        "method": "sdc", "horizon": 100, "eval_period": 5,
        "extra": ("--sdc_variant", "paper_cache2v"),
        "needs_aug": True, "needs_tokens": False, "tier": "U0",
    },
    "sdc-release-simclr": {
        "method": "sdc", "horizon": 100, "eval_period": 5,
        "extra": ("--sdc_variant", "release_simclr"),
        "needs_aug": True, "needs_tokens": False, "tier": "U0",
    },
    "hhch": {
        "method": "hhch", "horizon": 80, "eval_period": 5,
        "extra": (), "needs_aug": True, "needs_tokens": False, "tier": "U0",
    },
    "crovca": {
        "method": "crovca", "horizon": 5, "eval_period": 5,
        "extra": (), "needs_aug": True, "needs_tokens": False, "tier": "U0",
    },
    "oh": {
        "method": "oh", "horizon": 200, "eval_period": 5,
        "extra": (), "needs_aug": True, "needs_tokens": False, "tier": "U0",
    },
    "duheg": {
        "method": "duheg", "horizon": 60, "eval_period": 5,
        "extra": (), "needs_aug": True, "needs_tokens": False, "tier": "U?",
    },
    "umrch": {
        "method": "umrch", "horizon": 100, "eval_period": 5,
        "extra": (), "needs_aug": True, "needs_tokens": True, "tier": "U2",
    },
    "crh-supervised": {
        "method": "crh",
        "horizon": {
            "Flickr25k": 30, "MSCOCO": 30,
            "NUSWIDE": 30, "CIFAR10": 300,
        },
        "source_batch_size": 128,
        "eval_period": 5,
        "extra": (),
        "needs_aug": False,
        "needs_tokens": False,
        "tier": "S",
    },
}


def _base_length(bit: int) -> int:
    """Return the DNA length for one of the registered comparison budgets."""
    if bit not in SUPPORTED_BITS:
        raise ValueError(f"bit must be one of {SUPPORTED_BITS}; got {bit}")
    return bit // 2


def _selection_metric(bit: int) -> str:
    return f"raw_{_base_length(bit)}base_base_hamming_mAP_at_R"


def _source_horizon(specification: Mapping[str, object], dataset: str) -> int:
    """Resolve a scalar or source-provided per-dataset training horizon."""
    raw = specification["horizon"]
    if isinstance(raw, Mapping):
        if dataset not in raw:
            raise KeyError(f"no source/default horizon for {dataset!r}")
        raw = raw[dataset]
    horizon = int(raw)
    if horizon <= 0:
        raise ValueError(f"source/default horizon must be positive, got {horizon}")
    return horizon


def _absolute(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else (REPO / path).resolve()


def _run(command: Sequence[str], *, dry_run: bool) -> None:
    print("+", shlex.join([str(item) for item in command]), flush=True)
    if not dry_run:
        subprocess.run([str(item) for item in command], cwd=REPO, check=True)


def _newest_trial(root: Path, trial: str) -> Path:
    candidates = [path for path in root.glob(f"*/{trial}") if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"no generated trial directory {root}/*/{trial}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _require_file(path: Path, purpose: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{purpose} is missing: {path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_source_relative_paths(dataset: str, setting: str) -> tuple[str, ...]:
    """Return every file that defines the labels/splits consumed by training.

    Torchvision loads the extracted CIFAR pickle batches, not the downloaded
    tarball. Binding only the tarball therefore failed to bind the labels and
    train/query/database membership actually seen by ``CachedFeatureDataset``.
    """
    if dataset == "CIFAR10":
        if setting != "setting1":
            raise ValueError(f"CIFAR10 supports only setting1, got {setting!r}")
        return CIFAR10_CONSUMED_SOURCE_RELATIVE_PATHS
    return tuple(
        f"{dataset}/{setting}/{name}"
        for name in ("train.txt", "test.txt", "database.txt")
    )


def _dataset_source_binding(dataset: str, setting: str,
                            dataset_root: str | Path) -> tuple[str, dict[str, str]]:
    root = _absolute(dataset_root)
    hashes: dict[str, str] = {}
    for relative in _dataset_source_relative_paths(dataset, setting):
        path = root / relative
        _require_file(path, "dataset split/source artifact")
        hashes[relative] = _sha256(path)
    return str(root), hashes


def _verify_dataset_source_binding(identity: Mapping[str, object], *,
                                   dataset: str, setting: str) -> None:
    """Reopen the exact declared dataset sources and fail on substitution."""
    root_raw = identity.get("dataset_root")
    snapshot = identity.get("split_sha256")
    if not isinstance(root_raw, str) or not root_raw:
        raise ValueError("protocol_identity.dataset_root is missing")
    if not isinstance(snapshot, Mapping):
        raise ValueError("protocol_identity.split_sha256 is missing")
    expected_paths = _dataset_source_relative_paths(dataset, setting)
    if set(snapshot) != set(expected_paths):
        raise ValueError(
            "protocol_identity.split_sha256 path-set mismatch; "
            f"expected={list(expected_paths)!r}, found={sorted(map(str, snapshot))!r}")
    root = Path(root_raw).expanduser().resolve()
    for relative in expected_paths:
        declared = snapshot.get(relative)
        if not isinstance(declared, str) or re.fullmatch(
                r"[0-9a-f]{64}", declared) is None:
            raise ValueError(
                f"protocol_identity.split_sha256[{relative!r}] is invalid")
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(
                f"dataset source escapes declared root: {relative!r}") from error
        _require_file(path, "dataset split/source artifact")
        actual = _sha256(path)
        if actual != declared:
            raise ValueError(
                "dataset split/source changed after protocol identity was "
                f"sealed: {path}")


def _atomic_write_json(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
                mode='w', encoding='utf-8', dir=path.parent,
                prefix=f'.{path.name}.', suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return _sha256(path)


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return 'not-installed'


def _protocol_identity(args: argparse.Namespace, *, cache_dir: Path,
                       horizon: int, eval_period: int,
                       extra: Sequence[str],
                       cache_artifact_hashes: dict[str, str],
                       cache_decode_failure_audit: Mapping[str, object],
                       semantic_condition: dict,
                       execution_environment: Mapping[str, object] | None = None,
                       execution_environment_sha256: str | None = None,
                       ) -> tuple[dict, str]:
    """Bind a trial name to code, cache/splits, assets, and P0 settings."""
    method_file = {
        'cibhash': 'baseline/CIBHash.py',
        'cimon': 'baseline/CIMON.py',
        'mls3rduh': 'baseline/MLS3RDUH.py',
        'greedyhash': 'baseline/GreedyHash.py',
        'bihalf': 'baseline/BiHalf.py',
        'sdc-release': 'baseline/SDC.py',
        'sdc-release-simclr': 'baseline/SDC.py',
        'sdc-paper': 'baseline/SDC.py',
        'hhch': 'baseline/HHCH.py',
        'crovca': 'baseline/CroVCA.py',
        'oh': 'baseline/OH.py',
        'duheg': 'baseline/DUHEG.py',
        'umrch': 'baseline/UMRCH.py',
        'crh-supervised': 'baseline/CRH.py',
    }[args.variant]
    implementation_paths = [
        REPO / 'scripts/run_modern_baseline_p0.py',
        REPO / 'scripts/run_baseline_p0_matrix.py',
        REPO / method_file,
        REPO / 'baseline/base_model.py',
        REPO / 'baseline/modern_unsupervised.py',
        REPO / 'baseline/asset_provenance.py',
        REPO / 'baseline/cache_provenance.py',
        REPO / 'baseline/execution_environment.py',
        REPO / 'dna_utils/runtime_environment.py',
        REPO / 'dna_utils/gpu_lease.py',
        REPO / 'scripts/baseline_val_select_p0.py',
        REPO / 'scripts/extract_flat_baseline.py',
        REPO / 'scripts/apply_bio_projection.py',
        REPO / 'dna_utils/bio_constraints.py',
        REPO / 'val_split.py',
    ]
    dataset_root, split_sha256 = _dataset_source_binding(
        args.dataset, args.setting, args.dataset_root)
    author_fixed = args.protocol_mode == AUTHOR_FIXED_FINAL
    final_checkpoint_stage = (
        AUTHOR_FIXED_SINGLE_STAGE if author_fixed else P0_STAGE2_REFIT_TEST)
    payload = {
        'variant': args.variant,
        'dataset': args.dataset,
        'setting': args.setting,
        'bit': args.bit,
        'seed': args.seed,
        'device_request': args.device,
        'num_workers': args.num_workers,
        'runtime_versions': {
            'python': '.'.join(map(str, sys.version_info[:3])),
            'torch': _package_version('torch'),
            'numpy': _package_version('numpy'),
            'scipy': _package_version('scipy'),
        },
        'protocol_mode': args.protocol_mode,
        'trainer_protocol_stages': (
            [AUTHOR_FIXED_SINGLE_STAGE]
            if author_fixed else
            [P0_STAGE1_VAL_SELECTION, P0_STAGE2_REFIT_TEST]),
        'final_checkpoint_protocol_stage': final_checkpoint_stage,
        'checkpoint_policy': (
            AUTHOR_CHECKPOINT_POLICY if author_fixed
            else LEGACY_CHECKPOINT_POLICY),
        'author_horizon': horizon if author_fixed else None,
        'training_epochs': horizon if author_fixed else None,
        'final_epoch_zero_based': horizon - 1 if author_fixed else None,
        'validation_selection': not author_fixed,
        'designated_train_scope': (
            'full_designated_train' if author_fixed
            else 'optimization_train_then_full_designated_train_refit'),
        'val_seed': None if author_fixed else args.val_seed,
        'val_ratio': 0.0 if author_fixed else args.val_ratio,
        'horizon': horizon,
        'eval_period': eval_period,
        'batch_size_override': args.batch_size,
        'source_batch_size': VARIANTS[args.variant].get('source_batch_size'),
        'dataset_root': dataset_root,
        'cache_dir': str(cache_dir),
        'cache_meta_sha256': _sha256(cache_dir / 'meta.json'),
        'cache_image_ids_sha256': _sha256(cache_dir / 'image_ids.json'),
        'cache_artifact_sha256': cache_artifact_hashes,
        'cache_decode_failure_audit': dict(cache_decode_failure_audit),
        'split_sha256': split_sha256,
        'implementation_sha256': {
            str(path.relative_to(REPO)): _sha256(path)
            for path in implementation_paths
        },
        'method_extra': list(extra),
        'semantic_information_condition': semantic_condition,
    }
    if execution_environment is not None:
        payload['execution_environment'] = dict(execution_environment)
        payload['execution_environment_sha256'] = (
            execution_environment_sha256)
    for name in ('duheg_asset_manifest', 'umrch_asset_manifest'):
        value = getattr(args, name, None)
        if value:
            path = _absolute(value)
            payload[f'{name}_sha256'] = _sha256(path)
    canonical = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()
    return payload, hashlib.sha256(canonical).hexdigest()


def _assert_trial_absent(root: Path, trial: str, purpose: str) -> None:
    existing = [path for path in root.glob(f'*/{trial}') if path.exists()]
    if existing:
        raise FileExistsError(
            f'{purpose} trial already exists: {existing[0]}. Refusing to mix '
            'stale checkpoints/results; use a fresh output root.')


def _check_cache(cache_dir: Path, *, dataset: str, needs_aug: bool,
                 needs_tokens: bool, dataset_root: str | Path = REPO / 'dataset',
                 setting: str = 'setting1') -> dict:
    artifact_names = consumed_cache_artifact_names(
        paired_aug=needs_aug, visual_tokens=needs_tokens)
    required = ["meta.json", "image_ids.json", *artifact_names]
    missing = [name for name in required if not (cache_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"cache {cache_dir} lacks method-defining inputs: {missing}")
    with (cache_dir / 'meta.json').open(encoding='utf-8') as handle:
        meta = json.load(handle)
    with (cache_dir / 'image_ids.json').open(encoding='utf-8') as handle:
        image_ids = json.load(handle)
    if (not isinstance(image_ids, list)
            or not all(isinstance(item, str) for item in image_ids)
            or len(image_ids) != len(set(image_ids))):
        raise ValueError('cache image_ids must be a unique ordered string list')
    try:
        n = int(meta['N'])
        d_proj = int(meta['D_proj'])
        n_tokens = int(meta['num_tokens'])
        h_v = int(meta['H_v'])
        dtype = np.dtype(str(meta['dtype']))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError('cache meta lacks valid N/D_proj/num_tokens/H_v/dtype') from error
    if n != len(image_ids):
        raise ValueError(f'cache N={n} != image_ids count={len(image_ids)}')
    expected_shapes = {'visual_global.f16.npy': (n, d_proj)}
    if needs_aug:
        expected_shapes.update({
            'visual_global_aug0.f16.npy': (n, d_proj),
            'visual_global_aug1.f16.npy': (n, d_proj),
        })
    if needs_tokens:
        expected_shapes['visual_tokens.f16.npy'] = (n, n_tokens, h_v)
        if needs_aug:
            expected_shapes.update({
                'visual_tokens_aug0.f16.npy': (n, n_tokens, h_v),
                'visual_tokens_aug1.f16.npy': (n, n_tokens, h_v),
            })
    for name, shape in expected_shapes.items():
        array = np.load(cache_dir / name, mmap_mode='r')
        if array.shape != shape or array.dtype != dtype:
            raise ValueError(
                f'cache {name} has {array.shape}/{array.dtype}, expected '
                f'{shape}/{dtype}')

    eligibility_blockers: list[str] = []
    canonical = meta.get('canonical_transform')
    expected_canonical = (
        {
            'resize': [224, 224], 'interpolation': 'bicubic',
            'normalization': 'openai_clip',
        }
        if dataset == 'CIFAR10' else
        {
            'resize': [224, 224], 'interpolation': 'bilinear', 'crop': False,
            'normalization': 'openai_clip',
        }
    )
    if not isinstance(canonical, dict):
        eligibility_blockers.append('cache_meta_missing_canonical_transform')
    elif canonical != expected_canonical:
        eligibility_blockers.append('cache_canonical_transform_not_audited_clip_spec')
    augmentation = meta.get('augmentation_transform')
    expected_augmentation = (
        {
            'random_resized_crop': {
                'size': 224, 'scale': [0.5, 1.0],
                'interpolation': 'bicubic'},
            'horizontal_flip_probability': 0.5,
            'color_jitter': [0.4, 0.4, 0.4, 0.1],
            'color_jitter_probability': 1.0,
            'grayscale_probability': 0.2,
            'normalization': 'openai_clip',
            'gaussian_blur': False,
        }
        if dataset == 'CIFAR10' else
        {
            'random_resized_crop': {
                'size': 224, 'scale': [0.5, 1.0],
                'interpolation': 'bilinear'},
            'horizontal_flip_probability': 0.5,
            'color_jitter': [0.4, 0.4, 0.4, 0.1],
            'color_jitter_probability': 0.8,
            'grayscale_probability': 0.2,
            'normalization': 'openai_clip',
            'gaussian_blur': False,
        }
    )
    if needs_aug and not isinstance(augmentation, dict):
        eligibility_blockers.append('cache_meta_missing_augmentation_transform')
    elif needs_aug and augmentation != expected_augmentation:
        eligibility_blockers.append(
            'cache_augmentation_transform_not_audited_two_view_spec')
    if needs_aug and (
            isinstance(meta.get('augmentation_seed'), bool)
            or not isinstance(meta.get('augmentation_seed'), int)):
        eligibility_blockers.append('cache_meta_missing_augmentation_seed')
    if needs_aug and int(meta.get('save_aug_views', 0)) < 2:
        eligibility_blockers.append('cache_meta_declares_fewer_than_two_aug_views')

    hf = meta.get('hf_provenance')
    if not isinstance(hf, dict):
        eligibility_blockers.append('cache_meta_missing_immutable_hf_provenance')
    else:
        revision = hf.get('model_revision')
        weight_path = hf.get('model_weight_file')
        weight_hash = hf.get('model_weight_sha256')
        if (hf.get('checkpoint') != meta.get('backbone')
                or not isinstance(revision, str)
                or re.fullmatch(r'[0-9a-f]{40}', revision) is None
                or not isinstance(weight_path, str)
                or not isinstance(weight_hash, str)):
            eligibility_blockers.append('cache_hf_provenance_is_not_immutable')
        elif not Path(weight_path).is_file():
            eligibility_blockers.append('cache_hf_weight_provenance_file_missing')
        elif memoized_sha256_file(weight_path) != weight_hash:
            eligibility_blockers.append('cache_hf_weight_sha256_mismatch')

    try:
        decode_failure_audit = audit_cache_decode_failures(
            cache_dir, dataset=dataset, dataset_root=dataset_root,
            setting=setting)
    except (OSError, ValueError) as error:
        # Diagnostic smoke remains possible only through the caller's explicit
        # ineligible-smoke opt-in.  Paper production sees this blocker and stops
        # before a trainer/GPU is launched.
        decode_failure_audit = {
            'schema': DECODE_FAILURE_AUDIT_SCHEMA,
            'schema_version': DECODE_FAILURE_AUDIT_SCHEMA_VERSION,
            'status': 'invalid',
            'dataset': dataset,
            'setting': setting,
            'cache_dir': str(cache_dir.resolve()),
            'error': str(error),
        }
        eligibility_blockers.append(
            f'cache_decode_failure_audit_invalid: {error}')

    return {
        'meta': meta,
        'artifact_hashes': hash_cache_artifacts(cache_dir, artifact_names),
        'decode_failure_audit': decode_failure_audit,
        'eligibility_blockers': eligibility_blockers,
    }


def _method_extra(args: argparse.Namespace) -> list[str]:
    specification = VARIANTS[args.variant]
    extra = list(specification["extra"])
    if args.variant == "bihalf":
        # Preserve the executable release profiles instead of silently using
        # the Flickr defaults everywhere.  MSCOCO's released 1e-3 differs
        # from the paper's stated 1e-4 and remains visible in the checkpoint.
        profile = {
            "Flickr25k": (1e-4, 60),
            "MSCOCO": (1e-3, 60),
            "CIFAR10": (1e-4, 120),
            # No public NUS-WIDE trainer exists; use the paper/Flickr profile.
            "NUSWIDE": (1e-4, 60),
        }[args.dataset]
        extra += [
            "--learning_rate", str(profile[0]),
            "--step_size", str(profile[1]),
        ]
    if args.variant == "hhch" and args.hhch_clusters:
        extra += ["--hhch_clusters", args.hhch_clusters]
    if args.variant == "duheg":
        if not args.duheg_allow_unverified_selection:
            raise ValueError(
                "the public DUH-EG release does not provide a verifiable "
                "ordered selected-noun bank; pass "
                "--duheg-allow-unverified-selection only for the explicitly "
                "U?/U2 released-objective adapter")
        if not args.duheg_noun_embeddings or not args.duheg_asset_manifest:
            raise ValueError(
                "duheg requires --duheg-noun-embeddings and "
                "--duheg-asset-manifest")
        noun_path = _absolute(args.duheg_noun_embeddings)
        manifest_path = _absolute(args.duheg_asset_manifest)
        _require_file(noun_path, "audited selected-WordNet noun embedding bank")
        _require_file(manifest_path, "DUH-EG semantic asset manifest")
        extra += [
            "--duheg_noun_embeddings", str(noun_path),
            "--duheg_asset_manifest", str(manifest_path),
        ]
        extra.append("--duheg_allow_unverified_selection")
    if args.variant == "umrch":
        if args.dataset == "CIFAR10":
            raise ValueError("the audited UMRCH implementation does not report CIFAR-10")
        if (not args.umrch_concept_embeddings
                or not args.umrch_vision_adapter
                or not args.umrch_asset_manifest):
            raise ValueError(
                "umrch requires --umrch-concept-embeddings and "
                "--umrch-vision-adapter and --umrch-asset-manifest")
        concept_path = _absolute(args.umrch_concept_embeddings)
        adapter_path = _absolute(args.umrch_vision_adapter)
        manifest_path = _absolute(args.umrch_asset_manifest)
        _require_file(concept_path, "UMRCH benchmark-taxonomy embeddings")
        _require_file(adapter_path, "UMRCH CLIP local-token adapter")
        _require_file(manifest_path, "UMRCH semantic asset manifest")
        extra += [
            "--umrch_concept_embeddings", str(concept_path),
            "--umrch_vision_adapter", str(adapter_path),
            "--umrch_asset_manifest", str(manifest_path),
        ]
    return extra


def _audit_semantic_condition(
        args: argparse.Namespace, cache_dir: Path) -> tuple[dict, list[str]]:
    """Validate semantic assets and derive, never assume, the U-tier."""
    if args.variant == 'duheg':
        manifest = load_and_verify_text_asset_manifest(
            str(_absolute(args.duheg_asset_manifest)),
            expected_mode='duheg-selected',
            expected_embeddings=str(_absolute(args.duheg_noun_embeddings)),
            expected_prompt=list(DUHEG_PROMPT_TEMPLATES),
        )
        verify_manifest_cache(manifest, str(cache_dir))
        return {
            'information_tier': manifest['information_tier'],
            'term_source_condition': manifest['term_source_condition'],
            'matched_official_taxonomy_dataset': manifest[
                'matched_official_taxonomy_dataset'],
            'selection_verified': False,
        }, ['duheg_wordnet_term_selection_unverified']
    if args.variant == 'umrch':
        manifest = load_and_verify_text_asset_manifest(
            str(_absolute(args.umrch_asset_manifest)),
            expected_mode='umrch',
            expected_embeddings=str(_absolute(args.umrch_concept_embeddings)),
            expected_prompt='a photo of the {class}',
            expected_extra_outputs={
                'vision_adapter': str(_absolute(args.umrch_vision_adapter))},
        )
        verify_manifest_cache(manifest, str(cache_dir))
        if manifest['information_tier'] != 'U2':
            raise ValueError(
                'UMRCH requires an exact target benchmark taxonomy (U2)')
        return {
            'information_tier': manifest['information_tier'],
            'term_source_condition': manifest['term_source_condition'],
            'matched_official_taxonomy_dataset': manifest[
                'matched_official_taxonomy_dataset'],
            'selection_verified': None,
        }, []
    if args.variant == 'crh-supervised':
        return {
            'information_tier': 'S',
            'term_source_condition': (
                'benchmark training labels and target class taxonomy'
            ),
            'matched_official_taxonomy_dataset': args.dataset,
            'selection_verified': None,
            'uses_training_labels_in_objective': True,
            'uses_training_labels_in_center_reassignment': True,
            'comparison_panel': 'supervised',
        }, []
    return {
        'information_tier': VARIANTS[args.variant]['tier'],
        'term_source_condition': 'visual-only',
        'matched_official_taxonomy_dataset': '',
        'selection_verified': None,
    }, []


def _require_main_eligibility_or_smoke(
        blockers: Sequence[str], *, allow_smoke: bool) -> None:
    """Do not spend a full run on artifacts known to be paper-ineligible."""
    if blockers and not allow_smoke:
        formatted = '; '.join(str(blocker) for blocker in blockers)
        raise ValueError(
            'main-comparison eligibility checks failed: ' + formatted
            + '. Regenerate/fix the audited inputs, or pass '
              '--allow-main-ineligible-smoke only for a labelled integration '
              'smoke whose metrics will not enter the paper.')


def _training_command(args: argparse.Namespace, *, method: str,
                      cache_dir: Path, trial: str, epochs: int,
                      eval_period: int, val_ratio: float,
                      schedule_horizon: int,
                      extra: Sequence[str],
                      protocol_digest: str | None = None,
                      protocol_mode: str | None = None,
                      protocol_stage: str | None = None,
                      cache_artifact_hashes: dict[str, str] | None = None,
                      execution_environment: Mapping[str, object] | None = None,
                      execution_environment_sha256: str | None = None,
                      val_seed: int | None = None,
                      ) -> list[str]:
    command = [
        sys.executable, "-m", "baseline.base_model",
        "--method", method,
        "--dataset", args.dataset,
        "--setting", args.setting,
        "--bit", str(args.bit),
        "--cache_dir", str(cache_dir),
        "--dataset_root", str(_absolute(args.dataset_root)),
        "--device", args.device,
        "--seed", str(args.seed),
        "--max_epoch", str(epochs),
        "--schedule_horizon", str(schedule_horizon),
        "--eval_period", str(eval_period),
        "--val_split_ratio", str(val_ratio),
        "--trial_name", trial,
        "--model_root", str(_absolute(args.model_root)),
        "--result_root", str(_absolute(args.result_root)),
        "--compress_root", str(_absolute(args.compress_root)),
        "--num_workers", str(args.num_workers),
    ]
    if val_seed is not None:
        command += ["--val_split_seed", str(val_seed)]
    if protocol_digest is not None:
        command += ["--protocol_identity_sha256", protocol_digest]
    if (protocol_mode is None) != (protocol_stage is None):
        raise ValueError(
            'protocol_mode and protocol_stage must be supplied together')
    if protocol_mode is not None:
        command += [
            "--protocol_mode", protocol_mode,
            "--protocol_stage", str(protocol_stage),
        ]
    if cache_artifact_hashes is not None:
        command += [
            '--expected_cache_artifact_sha256_json',
            json.dumps(cache_artifact_hashes, sort_keys=True, separators=(',', ':')),
        ]
    if ((execution_environment is None)
            != (execution_environment_sha256 is None)):
        raise ValueError(
            'execution environment payload and digest must be supplied together')
    if execution_environment is not None:
        command += [
            '--expected_execution_environment_json',
            json.dumps(
                execution_environment, sort_keys=True, separators=(',', ':'),
                allow_nan=False),
            '--expected_execution_environment_sha256',
            str(execution_environment_sha256),
        ]
    if args.batch_size is not None:
        command += ["--batch_size", str(args.batch_size)]
    command += list(extra)
    return command


def _checkpoint_set_digest(hashes: dict[str, str]) -> str:
    encoded = json.dumps(
        hashes, sort_keys=True, separators=(',', ':'),
    ).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _load_verified_selection_artifact(
        path: Path, *, method: str, dataset: str,
        protocol_digest: str, bit: int) -> dict:
    _require_file(path, 'P0 selection-only JSON')
    with path.open(encoding='utf-8') as handle:
        selection = json.load(handle)
    expected_selection = {
        'method': method,
        'dataset': dataset,
        'bit_length': bit,
        'base_length': _base_length(bit),
        'selection_metric': _selection_metric(bit),
        'selection_only': True,
        'test_touched_during_selection': False,
        'protocol_identity_sha256': protocol_digest,
    }
    mismatches = {
        key: (selection.get(key), expected)
        for key, expected in expected_selection.items()
        if selection.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            f'selection artifact is incompatible with this refit: {mismatches}')
    directory_raw = selection.get('stage1_checkpoint_dir')
    hashes = selection.get('stage1_checkpoint_sha256')
    set_digest = selection.get('stage1_checkpoint_set_sha256')
    if not isinstance(directory_raw, str) or not isinstance(hashes, dict) or not hashes:
        raise ValueError('selection artifact lacks stage-1 checkpoint-set binding')
    directory = Path(directory_raw).expanduser().resolve()
    actual_paths = sorted(directory.glob('epoch_*.pth'))
    actual_names = [candidate.name for candidate in actual_paths]
    if set(actual_names) != set(hashes):
        raise ValueError(
            'selection checkpoint candidate set changed after E* selection')
    actual_hashes = {candidate.name: _sha256(candidate)
                     for candidate in actual_paths}
    if actual_hashes != hashes or _checkpoint_set_digest(hashes) != set_digest:
        raise ValueError(
            'selection checkpoint content hashes do not match the bound set')
    best_epoch = int(selection['best_epoch'])
    selected_name = f'epoch_{best_epoch:03d}.pth'
    selected = selection.get('selected_stage1_checkpoint')
    if (not isinstance(selected, dict)
            or Path(str(selected.get('path', ''))).resolve()
            != (directory / selected_name).resolve()
            or selected.get('sha256') != hashes.get(selected_name)):
        raise ValueError('selection artifact does not bind the selected E* checkpoint')
    return selection


def _selector_command(args: argparse.Namespace, *, method: str,
                      stage1_parent: Path, stage1_trial: str,
                      full_param_parent: Path, full_result_parent: Path,
                      full_trial: str, out: Path,
                      selection_only: bool) -> list[str]:
    command = [
        sys.executable, "scripts/baseline_val_select_p0.py",
        "--methods", method,
        "--datasets", args.dataset,
        "--bit", str(args.bit),
        "--device", args.device,
        "--num-workers", str(args.num_workers),
        "--stage1-root", str(stage1_parent),
        "--stage1-template", stage1_trial,
        "--full-param-root", str(full_param_parent),
        "--full-result-root", str(full_result_parent),
        "--full-template", full_trial,
        "--out", str(out),
    ]
    if selection_only:
        command.append("--selection-only")
    return command


def _trial_names(args: argparse.Namespace, protocol_digest: str) -> tuple[str, str]:
    """Create bit-explicit stage-1 and refit trial labels."""
    slug = args.variant.replace("-", "_")
    dataset_slug = args.dataset.lower()
    identity = protocol_digest[:12]
    stage1 = (
        f"{slug}_{dataset_slug}_{args.bit}b_P0s1_seed{args.seed}_p{identity}")
    refit_prefix = (
        f"{slug}_{dataset_slug}_{args.bit}b_P0refit_seed{args.seed}_p{identity}")
    return stage1, refit_prefix


def _author_trial_name(args: argparse.Namespace, protocol_digest: str) -> str:
    """Create a protocol-explicit label for the one-stage D6 run."""
    slug = args.variant.replace("-", "_")
    dataset_slug = args.dataset.lower()
    identity = protocol_digest[:12]
    return (
        f"{slug}_{dataset_slug}_{args.bit}b_D6authorLAST_seed{args.seed}_p{identity}")


def _require_single_terminal_checkpoint(
        directory: Path, *, final_epoch_zero_based: int) -> Path:
    """Fail closed unless the author-fixed run emitted only its LAST checkpoint."""
    expected = directory / f"epoch_{final_epoch_zero_based:03d}.pth"
    actual = sorted(directory.glob("epoch_*.pth"))
    if actual != [expected]:
        raise ValueError(
            "author-fixed checkpoint set must contain exactly terminal LAST "
            f"{expected.name}; found {[path.name for path in actual]}")
    _require_file(expected, "D6 author-fixed terminal checkpoint")
    return expected


def _read_verified_checkpoint_protocol(
        checkpoint: Path, *, protocol_mode: str, protocol_stage: str,
        protocol_digest: str,
        execution_environment: Mapping[str, object] | None = None,
        execution_environment_sha256: str | None = None,
        ) -> dict[str, object]:
    """Reopen the final checkpoint and bind its trainer protocol to the run."""
    import torch

    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError(
            f"cannot reopen final checkpoint protocol metadata: {checkpoint}") \
            from error
    if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
        raise ValueError(f"final checkpoint lacks config metadata: {checkpoint}")
    config = payload["config"]
    actual = {
        "protocol_mode": config.get("protocol_mode"),
        "protocol_stage": config.get("protocol_stage"),
        "protocol_identity_sha256": config.get("protocol_identity_sha256"),
    }
    expected = {
        "protocol_mode": protocol_mode,
        "protocol_stage": protocol_stage,
        "protocol_identity_sha256": protocol_digest,
    }
    if actual != expected:
        raise ValueError(
            "final checkpoint protocol contract mismatch: "
            f"expected {expected}, found {actual}")
    if ((execution_environment is None)
            != (execution_environment_sha256 is None)):
        raise ValueError(
            'expected checkpoint execution environment contract is incomplete')
    if execution_environment is not None:
        checkpoint_environment = config.get('execution_environment')
        checkpoint_environment_sha256 = config.get(
            'execution_environment_sha256')
        require_exact_environment(
            execution_environment, execution_environment_sha256,
            actual=checkpoint_environment,
            actual_digest=checkpoint_environment_sha256)
    return actual


def _bio_projection_command(
        args: argparse.Namespace, *, extraction_dir: Path,
        bio_out: Path) -> list[str]:
    """Build projection command while retaining canonical GC/run defaults."""
    return [
        sys.executable, "scripts/apply_bio_projection.py",
        "--out", str(bio_out),
        "--cell", f"{args.variant},{args.dataset},{extraction_dir}",
        "--expected_length", str(_base_length(args.bit)),
        "--device", args.device,
        "--save_projected",
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol-mode", choices=PROTOCOL_MODES,
        default=AUTHOR_FIXED_FINAL,
        help=(
            "author_fixed_final is the D6 paper-main contract. "
            "validation_sensitivity preserves the former E* selection/refit "
            "workflow as diagnostic supplementary analysis only."),
    )
    parser.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument(
        "--bit", type=int, choices=SUPPORTED_BITS, default=None,
        help=("Hash budget. D6 paper-main requires an explicit --bit 30; "
              "diagnostic sensitivity mode retains the other registered budgets."),
    )
    parser.add_argument("--setting", default="setting1")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--matrix-assigned-gpu-json", default=None,
        help=("Exact physical index/UUID assignment emitted by the matrix. "
              "Mandatory for D6; the child independently reopens it."),
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--val-seed", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--max-epoch", type=int, default=None,
                        help="Stage-1 search horizon; defaults to the audited method setting.")
    parser.add_argument("--eval-period", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--model-root", default="params_baseline")
    parser.add_argument("--result-root", default="result_baseline")
    parser.add_argument("--compress-root", default="compress_baseline")
    parser.add_argument("--stage", choices=("all", "selection", "refit"),
                        default="all")
    parser.add_argument("--best-epoch", type=int, default=None,
                        help="Optional cross-check for --stage refit; zero-based E*.")
    parser.add_argument(
        "--selection-json", default=None,
        help=("P0 selection-only JSON required when resuming --stage refit. "
              "Its method/dataset/E*/protocol digest are verified."),
    )
    parser.add_argument("--hhch-clusters", default=None,
                        help="Explicit bottom-to-top counts for a paper-unreported adaptation.")
    parser.add_argument("--duheg-noun-embeddings", default=None)
    parser.add_argument("--duheg-asset-manifest", default=None)
    parser.add_argument("--duheg-allow-unverified-selection", action="store_true")
    parser.add_argument("--umrch-concept-embeddings", default=None)
    parser.add_argument("--umrch-vision-adapter", default=None)
    parser.add_argument("--umrch-asset-manifest", default=None)
    parser.add_argument("--skip-bio-projection", action="store_true")
    parser.add_argument(
        "--allow-main-ineligible-smoke", action="store_true",
        help=("Explicitly run despite cache/semantic/projection eligibility "
              "blockers. The manifest remains main-ineligible."),
    )
    parser.add_argument(
        "--allow-nonstandard-protocol", action="store_true",
        help=("Permit smoke/ablation overrides of the shared .1/seed42/5-epoch "
              "P0 grid or source batch/horizon. Such runs are marked ineligible "
              "for the main comparison."),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    author_fixed = args.protocol_mode == AUTHOR_FIXED_FINAL
    if args.bit is None:
        if author_fixed:
            parser.error("D6 paper-main requires an explicit --bit 30")
        args.bit = 36
    if author_fixed and args.bit != 30:
        parser.error(
            "D6 paper-main is the matched 30-bit panel; pass exactly --bit 30. "
            "Use --protocol-mode validation_sensitivity for non-main budgets.")
    if author_fixed and args.cache_dir is None:
        parser.error(
            "D6 paper-main requires an explicit --cache-dir; legacy defaults "
            "are forbidden")
    if author_fixed and args.matrix_assigned_gpu_json is None:
        parser.error(
            "D6 paper-main requires --matrix-assigned-gpu-json so the actual "
            "child GPU can be bound to its matrix assignment")
    if author_fixed and args.device != "cuda:0":
        parser.error(
            "D6 paper-main requires logical --device cuda:0 under an exact "
            "single-GPU CUDA_VISIBLE_DEVICES assignment")
    if author_fixed and args.stage != "all":
        parser.error(
            "--stage selection/refit belongs only to validation_sensitivity; "
            "D6 author_fixed_final is one indivisible training stage")
    if author_fixed and (args.best_epoch is not None
                         or args.selection_json is not None):
        parser.error(
            "D6 author_fixed_final forbids --best-epoch and --selection-json")
    if author_fixed and args.val_ratio is not None:
        parser.error(
            "D6 author_fixed_final has no validation split; omit --val-ratio")
    if author_fixed and args.val_seed is not None:
        parser.error(
            "D6 author_fixed_final has no validation selector; omit --val-seed")

    if not author_fixed:
        args.val_ratio = 0.1 if args.val_ratio is None else args.val_ratio
        args.val_seed = 42 if args.val_seed is None else args.val_seed

    execution_environment: dict[str, object] | None = None
    execution_environment_sha256: str | None = None
    if args.matrix_assigned_gpu_json is not None:
        try:
            gpu_assignment = parse_gpu_assignment_json(
                args.matrix_assigned_gpu_json)
            execution_environment, execution_environment_sha256 = (
                capture_child_execution_environment(gpu_assignment))
        except ExecutionEnvironmentError as error:
            parser.error(str(error))

    base_length = _base_length(args.bit)
    selection_metric = None if author_fixed else _selection_metric(args.bit)
    specification = VARIANTS[args.variant]
    method = str(specification["method"])
    source_horizon = _source_horizon(specification, args.dataset)
    if author_fixed:
        if args.max_epoch is not None and int(args.max_epoch) != source_horizon:
            parser.error(
                "D6 author_fixed_final forbids horizon overrides: expected "
                f"H={source_horizon}, found {args.max_epoch}")
        if args.eval_period is not None and int(args.eval_period) != source_horizon:
            parser.error(
                "D6 author_fixed_final evaluates/saves only LAST: expected "
                f"--eval-period {source_horizon}, found {args.eval_period}")
        horizon = source_horizon
        eval_period = source_horizon
    else:
        horizon = int(args.max_epoch or source_horizon)
        eval_period = int(args.eval_period or specification["eval_period"])
    if horizon <= 0 or eval_period <= 0:
        raise ValueError("max epoch and eval period must be positive")
    if not author_fixed and not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be strictly between 0 and 1")
    protocol_deviations = []
    if not author_fixed:
        if args.val_ratio != 0.1:
            protocol_deviations.append(
                f'val_ratio={args.val_ratio} (required .1)')
        if args.val_seed != 42:
            protocol_deviations.append(
                f'val_seed={args.val_seed} (required 42)')
        if eval_period != 5:
            protocol_deviations.append(
                f'eval_period={eval_period} (required shared cadence 5)')
        if horizon != source_horizon:
            protocol_deviations.append(
                f'horizon={horizon} (source/default {source_horizon})')
    if args.batch_size is not None:
        protocol_deviations.append(
            f'batch_size override={args.batch_size} (source method default required)')
    if protocol_deviations and not args.allow_nonstandard_protocol:
        raise ValueError(
            'nonstandard comparison protocol: ' + '; '.join(protocol_deviations)
            + '. Pass --allow-nonstandard-protocol only for a labelled smoke/ablation run.')

    cache_dir = _absolute(args.cache_dir or DEFAULT_CACHE[args.dataset])
    cache_audit = _check_cache(
        cache_dir, dataset=args.dataset,
        needs_aug=bool(specification["needs_aug"]),
        needs_tokens=bool(specification["needs_tokens"]),
        dataset_root=args.dataset_root, setting=args.setting,
    )
    extra = _method_extra(args)
    semantic_condition, semantic_blockers = _audit_semantic_condition(
        args, cache_dir)
    input_eligibility_blockers = [
        *cache_audit['eligibility_blockers'], *semantic_blockers,
    ]
    #: Master audit :442 already settled this pair: the public Bi-half release
    #: has no NUS-WIDE trainer, so NUS-WIDE runs the paper/Flickr profile and
    #: the source boundary is DISCLOSED in the table and the text. Treating the
    #: same adaptation as an eligibility blocker refused the cell before
    #: training and made the 108-cell expected matrix unreachable, so
    #: `--require-paper-eligible` could never pass. The adaptation is recorded
    #: as a source boundary instead -- visible in every manifest, and named in
    #: the paper -- rather than silently run or silently refused.
    source_boundary_adaptations: list[str] = []
    if args.variant == 'bihalf' and args.dataset == 'NUSWIDE':
        source_boundary_adaptations.append(
            'bihalf_public_release_has_no_nuswide_training_script; '
            'paper_flickr_profile_adapter')
    if args.skip_bio_projection:
        input_eligibility_blockers.append('mandatory_bio_projection_skipped')
    _require_main_eligibility_or_smoke(
        input_eligibility_blockers,
        allow_smoke=bool(args.allow_main_ineligible_smoke),
    )
    eligibility_blockers = list(input_eligibility_blockers)
    if not author_fixed:
        eligibility_blockers.append(
            'validation_sensitivity_selection_refit_not_paper_main')
    protocol_payload, protocol_digest = _protocol_identity(
        args, cache_dir=cache_dir, horizon=horizon,
        eval_period=eval_period, extra=extra,
        cache_artifact_hashes=cache_audit['artifact_hashes'],
        cache_decode_failure_audit=cache_audit['decode_failure_audit'],
        semantic_condition=semantic_condition,
        execution_environment=execution_environment,
        execution_environment_sha256=execution_environment_sha256)

    model_root = _absolute(args.model_root)
    result_root = _absolute(args.result_root)
    compress_root = _absolute(args.compress_root)

    best_epoch: int | None = args.best_epoch
    selection_artifact: Path | None = None
    stage1_model_dir: Path | None = None
    stage1_result_dir: Path | None = None
    if author_fixed:
        training_trial = _author_trial_name(args, protocol_digest)
        if not args.dry_run:
            for root, purpose in (
                    (model_root, 'D6 author-fixed model'),
                    (result_root, 'D6 author-fixed result'),
                    (compress_root, 'D6 author-fixed compression')):
                _assert_trial_absent(root, training_trial, purpose)
        command = _training_command(
            args, method=method, cache_dir=cache_dir, trial=training_trial,
            epochs=horizon, eval_period=horizon, val_ratio=0.0,
            schedule_horizon=horizon, extra=extra,
            protocol_digest=protocol_digest,
            protocol_mode=AUTHOR_FIXED_FINAL,
            protocol_stage=AUTHOR_FIXED_SINGLE_STAGE,
            cache_artifact_hashes=cache_audit['artifact_hashes'],
            execution_environment=execution_environment,
            execution_environment_sha256=execution_environment_sha256,
            val_seed=None,
        )
        _run(command, dry_run=args.dry_run)
        if args.dry_run:
            print(
                f"[dry-run] D6: full designated train, H={horizon}, "
                f"LAST=epoch_{horizon - 1:03d}; no selector/refit.", flush=True)
            return 0
        training_model_dir = _newest_trial(model_root, training_trial)
        training_result_dir = _newest_trial(result_root, training_trial)
        checkpoint = _require_single_terminal_checkpoint(
            training_model_dir, final_epoch_zero_based=horizon - 1)
    else:
        stage1_trial, full_trial_prefix = _trial_names(args, protocol_digest)
        if args.stage in ("all", "selection"):
            if not args.dry_run:
                for root, purpose in (
                        (model_root, 'stage-1 model'),
                        (result_root, 'stage-1 result'),
                        (compress_root, 'stage-1 compression')):
                    _assert_trial_absent(root, stage1_trial, purpose)
            stage1_command = _training_command(
                args, method=method, cache_dir=cache_dir, trial=stage1_trial,
                epochs=horizon, eval_period=eval_period,
                val_ratio=args.val_ratio, schedule_horizon=horizon, extra=extra,
                protocol_digest=protocol_digest,
                protocol_mode=VALIDATION_SENSITIVITY,
                protocol_stage=P0_STAGE1_VAL_SELECTION,
                cache_artifact_hashes=cache_audit['artifact_hashes'],
                execution_environment=execution_environment,
                execution_environment_sha256=execution_environment_sha256,
                val_seed=args.val_seed,
            )
            _run(stage1_command, dry_run=args.dry_run)
            if args.dry_run:
                print(
                    "[dry-run][validation_sensitivity] E* is data-dependent; "
                    "refit/extraction commands are omitted.")
                return 0
            stage1_model_dir = _newest_trial(model_root, stage1_trial)
            stage1_result_dir = _newest_trial(result_root, stage1_trial)
            selection_out = stage1_result_dir / "p0_selection_only.json"
            selection_artifact = selection_out
            selector = _selector_command(
                args, method=method,
                stage1_parent=stage1_model_dir.parent,
                stage1_trial=stage1_model_dir.name,
                full_param_parent=stage1_model_dir.parent,
                full_result_parent=stage1_result_dir.parent,
                full_trial="unused_until_refit", out=selection_out,
                selection_only=True,
            )
            _run(selector, dry_run=False)
            selection = _load_verified_selection_artifact(
                selection_out, method=method, dataset=args.dataset,
                protocol_digest=protocol_digest, bit=args.bit)
            best_epoch = int(selection["best_epoch"])
            print(
                f"[validation_sensitivity] fixed E*={best_epoch} from raw "
                f"{base_length}-base held-out mAP@R="
                f"{selection['best_val_base_mAP_at_R']:.6f}", flush=True)
            if args.stage == "selection":
                return 0

        if args.stage == "refit":
            if args.selection_json is None:
                if not args.allow_nonstandard_protocol:
                    raise ValueError(
                        '--stage refit requires --selection-json so E* and the '
                        'protocol fingerprint cannot be supplied by hand')
                if best_epoch is None:
                    raise ValueError(
                        'a nonstandard manual refit still requires --best-epoch')
                eligibility_blockers.append(
                    'manual_refit_without_verified_selection_artifact')
            else:
                selection_artifact = _absolute(args.selection_json)
                selection = _load_verified_selection_artifact(
                    selection_artifact, method=method, dataset=args.dataset,
                    protocol_digest=protocol_digest, bit=args.bit)
                selected_epoch = int(selection['best_epoch'])
                if best_epoch is not None and int(best_epoch) != selected_epoch:
                    raise ValueError(
                        f'--best-epoch={best_epoch} disagrees with selection JSON '
                        f'E*={selected_epoch}')
                best_epoch = selected_epoch
                stage1_model_dir = Path(selection['stage1_checkpoint_dir'])
                stage1_result_dir = selection_artifact.parent
        assert best_epoch is not None
        training_trial = f"{full_trial_prefix}_e{best_epoch}"
        if not args.dry_run:
            for root, purpose in (
                    (model_root, 'sensitivity refit model'),
                    (result_root, 'sensitivity refit result'),
                    (compress_root, 'sensitivity refit compression')):
                _assert_trial_absent(root, training_trial, purpose)
        refit_command = _training_command(
            args, method=method, cache_dir=cache_dir, trial=training_trial,
            epochs=best_epoch + 1, eval_period=best_epoch + 1,
            val_ratio=0.0, schedule_horizon=horizon, extra=extra,
            protocol_digest=protocol_digest,
            protocol_mode=VALIDATION_SENSITIVITY,
            protocol_stage=P0_STAGE2_REFIT_TEST,
            cache_artifact_hashes=cache_audit['artifact_hashes'],
            execution_environment=execution_environment,
            execution_environment_sha256=execution_environment_sha256,
            val_seed=args.val_seed,
        )
        _run(refit_command, dry_run=args.dry_run)
        if args.dry_run:
            return 0
        training_model_dir = _newest_trial(model_root, training_trial)
        training_result_dir = _newest_trial(result_root, training_trial)
        checkpoint = training_model_dir / f"epoch_{best_epoch:03d}.pth"
        _require_file(checkpoint, "validation-sensitivity refit checkpoint")
        if stage1_model_dir is not None and stage1_result_dir is not None:
            protocol_out = training_result_dir / "p0_protocol_summary.json"
            selector = _selector_command(
                args, method=method,
                stage1_parent=stage1_model_dir.parent,
                stage1_trial=stage1_model_dir.name,
                full_param_parent=training_model_dir.parent,
                full_result_parent=training_result_dir.parent,
                full_trial=training_trial, out=protocol_out,
                selection_only=False,
            )
            _run(selector, dry_run=False)

    extraction_dir = training_result_dir.parent / f"{training_trial}_dnaeval"
    if extraction_dir.exists():
        raise FileExistsError(
            f'extraction output already exists: {extraction_dir}; refusing '
            'to mix artifacts from different checkpoints')
    extract = [
        sys.executable, "scripts/extract_flat_baseline.py",
        "--weights", str(checkpoint),
        "--dataset", args.dataset,
        "--setting", args.setting,
        "--cache_dir", str(cache_dir),
        "--dataset_root", str(_absolute(args.dataset_root)),
        "--out", str(extraction_dir),
        "--num_workers", str(args.num_workers),
        "--device", args.device,
    ]
    _run(extract, dry_run=False)

    # Reopen immediately before publishing the immutable pre-BIO manifest.
    # This also catches any checkpoint substitution during extraction.
    final_checkpoint_protocol_stage = (
        AUTHOR_FIXED_SINGLE_STAGE if author_fixed else P0_STAGE2_REFIT_TEST)
    checkpoint_protocol = _read_verified_checkpoint_protocol(
        checkpoint,
        protocol_mode=args.protocol_mode,
        protocol_stage=final_checkpoint_protocol_stage,
        protocol_digest=protocol_digest,
        execution_environment=execution_environment,
        execution_environment_sha256=execution_environment_sha256,
    )
    # Dataset labels/splits are scientific inputs just like the checkpoint.
    # Reopen them immediately before publishing the immutable pre-BIO manifest
    # so an in-flight extracted-file edit cannot inherit the earlier digest.
    _verify_dataset_source_binding(
        protocol_payload, dataset=args.dataset, setting=args.setting)
    if cache_audit['decode_failure_audit'].get('status') == 'verified':
        verify_cache_decode_failure_binding(
            protocol_payload, dataset=args.dataset, setting=args.setting)

    main_protocol_eligible = bool(
        author_fixed and not protocol_deviations and not eligibility_blockers)
    manifest = {
        "protocol_mode": args.protocol_mode,
        "checkpoint_policy": (
            AUTHOR_CHECKPOINT_POLICY if author_fixed
            else LEGACY_CHECKPOINT_POLICY),
        "variant": args.variant,
        "method": method,
        "information_tier": semantic_condition['information_tier'],
        "comparison_panel": (
            "supervised"
            if semantic_condition["information_tier"] == "S"
            else "target-label-free"
        ),
        "semantic_information_condition": semantic_condition,
        "dataset": args.dataset,
        "bit_length": args.bit,
        "base_length": base_length,
        "seed": args.seed,
        "val_seed": None if author_fixed else args.val_seed,
        "selection_metric": selection_metric,
        "nominal_schedule_horizon": horizon,
        "protocol_digest_sha256": protocol_digest,
        "protocol_identity": protocol_payload,
        "checkpoint_protocol": checkpoint_protocol,
        "execution_environment": execution_environment,
        "execution_environment_sha256": execution_environment_sha256,
        "selection_artifact": (
            None if selection_artifact is None else str(selection_artifact)),
        "selection_artifact_sha256": (
            None if selection_artifact is None else _sha256(selection_artifact)),
        "main_protocol_eligible": main_protocol_eligible,
        "protocol_deviations": protocol_deviations,
        "main_eligibility_blockers": eligibility_blockers,
        "source_boundary_adaptations": source_boundary_adaptations,
        "published_table_reproduction_eligible": False,
        "matched_comparison_core_status": (
            "released-objective adapter; noun selection unverified"
            if args.variant == 'duheg' else
            (
                "clean-room published-equation supervised core under shared "
                "frozen-cache adaptation"
                if args.variant == 'crh-supervised' else
                "audited method core under shared frozen-cache adaptation"
            )
        ),
        "training_model_dir": str(training_model_dir),
        "training_result_dir": str(training_result_dir),
        "extraction_dir": str(extraction_dir),
        "test_used_for_selection": False,
        "final_checkpoint_count": 1,
        "final_checkpoint": str(checkpoint.resolve()),
        "final_checkpoint_sha256": _sha256(checkpoint),
        "cache_artifact_sha256": cache_audit['artifact_hashes'],
        "cache_decode_failure_audit": cache_audit['decode_failure_audit'],
        "extraction_artifact_sha256": {
            name: _sha256(extraction_dir / name)
            for name in (
                'extract_db.npz', 'extract_query.npz', 'args_extract.txt')
        },
        "run_manifest_phase": "pre_bio_projection",
        "bio_projection_status": "pending",
        "test_metric_passes": (
            "raw final-checkpoint metric, then post-projection metric from the "
            "same fixed checkpoint; no test-dependent choice"
        ),
    }
    if author_fixed:
        manifest.update({
            "author_horizon": horizon,
            "training_epochs": horizon,
            "final_epoch_zero_based": horizon - 1,
            "validation_selection": False,
            "refit_performed": False,
            "training_stage": "author_fixed_single_stage",
            "designated_train_scope": "full_designated_train",
        })
    else:
        assert best_epoch is not None
        manifest.update({
            "best_epoch_zero_based": best_epoch,
            "refit_epochs": best_epoch + 1,
            "validation_selection": True,
            "refit_performed": True,
            "training_stage": "validation_selection_scratch_refit",
            "designated_train_scope": (
                "optimization_train_then_full_designated_train_refit"),
            "stage1_model_dir": (
                None if stage1_model_dir is None else str(stage1_model_dir)),
            "refit_model_dir": str(training_model_dir),
            "refit_result_dir": str(training_result_dir),
        })
    protocol_manifest_path = extraction_dir / 'p0_protocol_manifest.json'
    protocol_manifest_sha256 = _atomic_write_json(
        protocol_manifest_path, manifest)

    if not args.skip_bio_projection:
        bio_out = extraction_dir / "bio_projection.json"
        bio = _bio_projection_command(
            args, extraction_dir=extraction_dir, bio_out=bio_out)
        _run(bio, dry_run=False)
        with bio_out.open(encoding='utf-8') as handle:
            bio_payload = json.load(handle)
        if len(bio_payload.get('cells', [])) != 1:
            raise ValueError('bio projection manifest must contain exactly one cell')
        bio_cell = bio_payload['cells'][0]
        if (bio_cell.get('paper_result_eligible') is True) != main_protocol_eligible:
            raise ValueError(
                'bio projection and runner disagree on paper-main eligibility; '
                'refusing to publish a mixed protocol manifest')
        output_artifacts = bio_cell.get('output_artifacts')
        if (not isinstance(output_artifacts, dict)
                or set(output_artifacts) != {'db', 'query'}):
            raise ValueError('bio projection manifest lacks projected outputs')
        for name, record in output_artifacts.items():
            if not isinstance(record, dict) or not isinstance(record.get('path'), str):
                raise ValueError(f'invalid projected {name} artifact record')
            projected_path = Path(record['path']).expanduser()
            _require_file(projected_path, f'projected {name} artifact')
            if record.get('sha256') != _sha256(projected_path):
                raise ValueError(
                    f'projected {name} artifact SHA-256 does not match bio manifest')
        sidecar = Path(str(bio_out) + '.sha256')
        _require_file(sidecar, 'bio projection SHA-256 sidecar')
        bio_manifest_sha256 = _sha256(bio_out)
        sidecar_parts = sidecar.read_text(encoding='utf-8').strip().split()
        if not sidecar_parts or sidecar_parts[0] != bio_manifest_sha256:
            raise ValueError('bio projection SHA-256 sidecar does not match manifest')
        manifest.update({
            'run_manifest_phase': 'bio_projection_completed',
            'protocol_manifest': str(protocol_manifest_path.resolve()),
            'protocol_manifest_sha256': protocol_manifest_sha256,
            'bio_projection_status': (
                'paper_result_eligible'
                if bio_cell.get('paper_result_eligible') is True
                else 'completed_not_paper_eligible'),
            'bio_projection_manifest': str(bio_out.resolve()),
            'bio_projection_manifest_sha256': bio_manifest_sha256,
            'bio_projection_sidecar': str(sidecar.resolve()),
            'bio_projection_sidecar_sha256': _sha256(sidecar),
            'bio_projected_artifacts': output_artifacts,
        })
    else:
        manifest.update({
            'run_manifest_phase': 'bio_projection_skipped',
            'protocol_manifest': str(protocol_manifest_path.resolve()),
            'protocol_manifest_sha256': protocol_manifest_sha256,
            'bio_projection_status': 'skipped_non_main',
        })
    manifest_path = extraction_dir / 'p0_run_manifest.json'
    _atomic_write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
