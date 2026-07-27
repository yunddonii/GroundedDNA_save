"""P0 stage 2 for the baselines: pick E* on held-out val, report test at E*.

Stage 1 retrains each baseline on the optimization-train 90% and saves periodic
checkpoints. This script independently re-scores every stage-1 checkpoint on
held-out validation rows (val_query vs opt-train DB) and takes E* = argmax
validation mAP@R under raw L-base Hamming distance (L = bit/2). It never
constructs the
official test/database datasets until after E* has been fixed.

The stage-1 model is never reported. The reported number is the 100%-train
run's test mAP@R at E*, exactly mirroring our own stage 2 ("refit on 100% of
train, stop at the val-selected epoch"). A modern result JSON's `base_2bit`
block is reused; historical binary-only result files are re-evaluated from the
full-train checkpoint with the base metric.

Protocol invariants (must stay identical to train_siglip2.py):
  * the split comes from `val_split.carve_val_indices(labels, 0.1, 42)` -- imported,
    never reimplemented;
  * mid-eval is val_query vs opt-train DB, disjoint splits, no self-match removal;
  * the selection metric is raw Hamming over L bases obtained by packing each
    adjacent pair of the 36/48 signed bits; it is not binary Hamming;
  * mAP@R uses the dataset cutoff in `MAP_AT_R_BY_DATASET`
    (CIFAR10@1000, others@5000), CalcTopMap convention;
  * codes are `sign(model(cached_feature))`, and checkpoints are reconstructed
    through their registered method class, including custom modern heads.

Usage:
    python scripts/baseline_val_select_p0.py --methods cibhash --datasets Flickr25k \
        --out docs/baseline_p0_stage2_partial/cibhash_Flickr25k.json
    python scripts/baseline_val_select_p0.py --merge 'docs/baseline_p0_stage2_partial/*.json' \
        --out docs/baseline_p0_stage2.json --markdown docs/baseline_p0_stage2.md
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from baseline.base_model import (  # noqa: E402
    MAP_AT_R_BY_DATASET,
    CachedFeatureDataset,
    _extract_codes,
    build_model_from_checkpoint_payload,
    evaluate_base_retrieval_model,
    evaluate_retrieval_model,
    verify_checkpoint_data_context,
)
from val_split import carve_val_indices  # noqa: E402

METHODS = ('cibhash', 'cimon', 'mls3rduh')
DATASETS = ('Flickr25k', 'MSCOCO', 'NUSWIDE', 'CIFAR10')
SUPPORTED_BITS = (36, 48)
SHORT = {'CIFAR10': 'cifar10', 'Flickr25k': 'flickr25k',
         'MSCOCO': 'mscoco', 'NUSWIDE': 'nuswide'}

STAGE1_ROOT = os.path.join(_REPO, 'params_baseline', '260718')
FULL_PARAM_ROOT = os.path.join(_REPO, 'params_baseline', '260714')
FULL_RESULT_ROOT = os.path.join(_REPO, 'result_baseline', '260714')
DEFAULT_STAGE1_TEMPLATE = '{method}_{short}_clip_P0s1_unsup60'
DEFAULT_FULL_TEMPLATE = '{method}_{short}_clip_mapr_unsup60'


def _base_length(bit: int) -> int:
    if bit not in SUPPORTED_BITS:
        raise ValueError(f'bit must be one of {SUPPORTED_BITS}; got {bit}')
    return bit // 2


def _selection_metric(bit: int) -> str:
    return f'raw_{_base_length(bit)}base_base_hamming_mAP_at_R'

_STAGE_SPECIFIC_CONFIG_KEYS = {
    'trial_name', 'max_epoch', 'eval_period', 'val_split_ratio',
    'n_opt_train', 'n_val_query', 'val_split_strategy',
    'val_split_indices_sha256', 'model_dir', 'result_dir', 'compress_dir',
    'day_info', 'protocol_stage',
}
_CRH_STAGE_OBSERVATION_CONFIG_KEYS = {
    # CRH excludes zero-positive-label rows from the optimization loader
    # because its label-normalized loss is undefined for those rows.  Stage 1
    # and the 100%-train refit intentionally see different training
    # partitions, so these audit-only counts differ even when every
    # scientific CRH setting is identical.  Keep the allowlist exact: the
    # filtering rule and all CRH objective/codebook knobs still fail closed.
    'crh_training_partition_stage',
    'crh_training_rows_total',
    'crh_training_rows_retained',
    'crh_training_zero_label_rows_dropped',
    'crh_training_class_positive_counts',
}
_COMMON_REQUIRED_CONFIG_KEYS = {
    'method', 'dataset', 'setting', 'bit', 'seed', 'batch_size',
    'learning_rate', 'optimizer_name', 'adam_weight_decay',
    'sgd_weight_decay', 'sgd_momentum', 'lr_scheduler', 'schedule_horizon',
    'gen_code_method', 'finetune', 'dataset_return_paired_aug_img',
    'dataset_return_visual_tokens', 'resolved_backbone',
    'resolved_projection_dim', 'protocol_identity_sha256',
    'resolved_cache_artifact_sha256', 'cache_artifact_binding_status',
}
_METHOD_REQUIRED_CONFIG_KEYS = {
    'crh': {
        # Supervision and implementation identity.
        'supervision_regime', 'uses_train_labels_in_objective',
        'uses_train_labels_in_center_reassignment', 'implementation_variant',
        'information_tier', 'source_code_commit', 'numeric_precision',
        # Paper objective, codebook construction/update, and optimizer audit.
        'crh_margin', 'crh_scale', 'crh_quantization_weight',
        'crh_codebook_scale', 'crh_codebook_size', 'crh_num_classes',
        'crh_head_dim', 'crh_num_heads', 'crh_update_rounds',
        'crh_update_interval', 'crh_gradient_clip_norm',
        'crh_source_horizon', 'crh_source_optimizer',
        'crh_source_scheduler',
        # The five stage observations below may differ between opt-train and
        # full-train, but they must be present on both sides.  Filter policy
        # and scope are never ignored.
        'crh_training_partition_stage', 'crh_training_rows_total',
        'crh_training_rows_retained',
        'crh_training_zero_label_rows_dropped',
        'crh_training_class_positive_counts',
        'crh_training_all_classes_present', 'crh_zero_label_policy',
        'crh_training_filter_scope',
    },
    'greedyhash': {
        'greedyhash_quantization_weight', 'implementation_variant',
        'information_tier', 'source_code_commit',
        'internal_hash_alphabet', 'extractor_hash_alphabet',
    },
    'bihalf': {
        'bihalf_gamma', 'bihalf_odd_batch_policy', 'implementation_variant',
        'information_tier', 'official_source_commit',
        'bihalf_training_quantizer', 'bihalf_inference_quantizer',
    },
    'sdc': {
        'sdc_variant', 'sdc_beta', 'sdc_reconstruction', 'sdc_quantization',
        'sdc_reconstruction_weight', 'sdc_quantization_weight',
        'sdc_orthogonal_only', 'sdc_contrastive_temperature',
        'sdc_contrastive_weight',
        'implementation_variant', 'sdc_calibration_scope',
        'information_tier', 'audited_release_commit',
    },
    'hhch': {
        'hhch_variant', 'hhch_temperature', 'hhch_curvature',
        'hhch_hyper_dim', 'hhch_quantization_weight', 'hhch_clusters',
        'hhch_kmeans_max_iter', 'hhch_kmeans_tolerance',
        'hhch_cluster_chunk_size', 'hhch_dropout', 'hhch_clip_radius',
        'hhch_cluster_counts', 'hhch_cluster_source',
        'hhch_learning_rate_schedule_source',
        'implementation_variant', 'information_tier',
        'audited_release_commit',
    },
    'crovca': {
        'crovca_min_lr', 'crovca_eps', 'crovca_alignment_weight',
        'crovca_diversity_weight',
        'implementation_variant', 'source_code_commit',
        'numeric_precision', 'information_tier',
    },
    'duheg': {
        'duheg_temperature', 'duheg_positive_threshold',
        'duheg_guidance_temperature', 'duheg_guidance_precision',
        'duheg_warmup_epochs', 'duheg_min_lr', 'duheg_noun_bank_sha256',
        'duheg_asset_manifest_sha256', 'duheg_selection_verified',
        'duheg_allow_unverified_selection', 'information_tier',
        'duheg_term_source_condition',
        'implementation_variant', 'duheg_n_nouns', 'duheg_text_dim',
        'publishable_as_exact_duheg',
        'duheg_matched_official_taxonomy_dataset',
    },
    'umrch': {
        'umrch_vl_temperature', 'umrch_concept_threshold',
        'umrch_negative_threshold', 'umrch_contrastive_temperature',
        'umrch_distribution_weight', 'umrch_contrastive_weight',
        'umrch_concept_sha256', 'umrch_adapter_sha256',
        'umrch_asset_manifest_sha256', 'umrch_official_taxonomy_sha256',
        'umrch_n_concepts', 'umrch_taxonomy_source_commit',
        'information_tier', 'umrch_term_source_condition',
        'implementation_variant',
        'umrch_matched_official_taxonomy_dataset',
    },
    'oh': {
        'oh_middle_dim', 'oh_continuous_dim', 'oh_momentum',
        'oh_temperature', 'oh_queue_length', 'oh_context_weight',
        'oh_shuffle_key_batch', 'oh_adam_epsilon',
        'implementation_variant', 'source_code_commit',
        'internal_hash_alphabet', 'extractor_hash_alphabet',
        'numeric_precision', 'information_tier',
    },
}


def _trial_dir(root: str, template: str, method: str, dataset: str) -> str:
    return os.path.join(root, template.format(
        method=method, dataset=dataset, short=SHORT[dataset]))


def _resolve_repo_path(path: str | None) -> str | None:
    if path is None or os.path.isabs(path):
        return path
    return os.path.join(_REPO, path)


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _stage1_checkpoint_artifacts(paths: list[str]) -> dict:
    """Content-bind the exact candidate set used for E* selection."""
    if not paths:
        raise ValueError('cannot bind an empty stage-1 checkpoint set')
    resolved = [os.path.realpath(path) for path in paths]
    directories = {os.path.dirname(path) for path in resolved}
    if len(directories) != 1:
        raise ValueError('stage-1 checkpoints do not share one directory')
    names = [os.path.basename(path) for path in resolved]
    if len(names) != len(set(names)):
        raise ValueError('stage-1 checkpoint basenames are not unique')
    hashes = {
        name: _sha256_file(path)
        for name, path in sorted(zip(names, resolved))
    }
    encoded = json.dumps(
        hashes, sort_keys=True, separators=(',', ':'),
    ).encode('utf-8')
    return {
        'stage1_checkpoint_dir': next(iter(directories)),
        'stage1_checkpoint_sha256': hashes,
        'stage1_checkpoint_set_sha256': hashlib.sha256(encoded).hexdigest(),
    }


def _assert_required_config(config: dict, checkpoint_path: str) -> None:
    method = str(config.get('method', '')).lower()
    required = _COMMON_REQUIRED_CONFIG_KEYS | _METHOD_REQUIRED_CONFIG_KEYS.get(
        method, set())
    missing = sorted(key for key in required if key not in config)
    if missing:
        raise ValueError(
            f'{checkpoint_path} lacks selection-critical config keys: {missing}')


def _scientific_config(config: dict) -> dict:
    """Retain every non-stage field so future loss knobs fail closed."""
    ignored = set(_STAGE_SPECIFIC_CONFIG_KEYS)
    if str(config.get('method', '')).lower() == 'crh':
        ignored.update(_CRH_STAGE_OBSERVATION_CONFIG_KEYS)
    return {
        key: value for key, value in config.items()
        if key not in ignored
    }


def _assert_same_stage1_config(reference: dict, candidate: dict,
                               checkpoint_path: str) -> None:
    _assert_required_config(reference, 'reference stage-1 checkpoint')
    _assert_required_config(candidate, checkpoint_path)
    keys = set(reference) | set(candidate)
    mismatches = {
        key: (reference.get(key), candidate.get(key))
        for key in keys if reference.get(key) != candidate.get(key)
    }
    reference_cache = os.path.realpath(str(_resolve_repo_path(
        reference.get('resolved_cache_dir') or reference.get('cache_dir'))))
    candidate_cache = os.path.realpath(str(_resolve_repo_path(
        candidate.get('resolved_cache_dir') or candidate.get('cache_dir'))))
    if reference_cache != candidate_cache:
        mismatches['cache_dir'] = (reference_cache, candidate_cache)
    if mismatches:
        raise ValueError(
            f'stage-1 checkpoint config drift at {checkpoint_path}: {mismatches}')


def _assert_compatible_full_config(stage1: dict, full: dict,
                                   checkpoint_path: str,
                                   best_epoch: int | None = None) -> None:
    _assert_required_config(stage1, 'selected stage-1 checkpoint')
    _assert_required_config(full, checkpoint_path)
    stage_scientific = _scientific_config(stage1)
    full_scientific = _scientific_config(full)
    keys = set(stage_scientific) | set(full_scientific)
    mismatches = {
        key: (stage_scientific.get(key), full_scientific.get(key))
        for key in keys
        if stage_scientific.get(key) != full_scientific.get(key)
    }
    stage1_cache = os.path.realpath(str(_resolve_repo_path(
        stage1.get('resolved_cache_dir') or stage1.get('cache_dir'))))
    full_cache = os.path.realpath(str(_resolve_repo_path(
        full.get('resolved_cache_dir') or full.get('cache_dir'))))
    if stage1_cache != full_cache:
        mismatches['cache_dir'] = (stage1_cache, full_cache)
    if float(full.get('val_split_ratio', -1.0)) != 0.0:
        mismatches['full.val_split_ratio'] = (
            'required 0.0', full.get('val_split_ratio'))
    if best_epoch is not None:
        expected_stop = int(best_epoch) + 1
        if int(full.get('max_epoch', -1)) != expected_stop:
            mismatches['full.max_epoch'] = (
                f'required E*+1={expected_stop}', full.get('max_epoch'))
        if int(full.get('eval_period', -1)) != expected_stop:
            mismatches['full.eval_period'] = (
                f'required E*+1={expected_stop}', full.get('eval_period'))
    if mismatches:
        raise ValueError(
            f'selected full checkpoint is incompatible with stage 1 at '
            f'{checkpoint_path}: {mismatches}')


def evaluate_pair(method: str, dataset: str, device: str,
                  stage1_root: str = STAGE1_ROOT,
                  full_param_root: str = FULL_PARAM_ROOT,
                  full_result_root: str = FULL_RESULT_ROOT,
                  stage1_template: str = DEFAULT_STAGE1_TEMPLATE,
                  full_template: str = DEFAULT_FULL_TEMPLATE,
                  num_workers: int = 4,
                  selection_only: bool = False,
                  bit: int = 36) -> dict:
    base_length = _base_length(bit)
    selection_metric = _selection_metric(bit)
    ckpt_dir = _trial_dir(stage1_root, stage1_template, method, dataset)
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, 'epoch_*.pth')))
    if not ckpts:
        raise FileNotFoundError(f'no stage-1 checkpoints in {ckpt_dir}')

    checkpoint_artifacts = _stage1_checkpoint_artifacts(ckpts)
    cfg = torch.load(ckpts[0], map_location='cpu', weights_only=False)['config']
    _assert_required_config(cfg, ckpts[0])
    assert cfg['dataset'] == dataset, f"{ckpts[0]} is for {cfg['dataset']}, not {dataset}"
    # Sanity: the stage-1 run must actually have held rows out.
    assert float(cfg.get('val_split_ratio', 0.0)) > 0.0, \
        f'{ckpt_dir} was NOT trained with --val_split_ratio > 0; E* would leak.'
    ratio = float(cfg['val_split_ratio'])
    seed = int(cfg.get('val_split_seed', 42))

    # Paths in the stored config are relative to the repo root (the runs were
    # launched from there), so resolve them against it rather than the cwd.
    dataset_root = _resolve_repo_path(cfg['dataset_root'])
    cache_dir = _resolve_repo_path(
        cfg.get('resolved_cache_dir') or cfg.get('cache_dir'))
    verify_checkpoint_data_context(
        {'config': cfg}, dataset=dataset, setting=cfg['setting'],
        cache_dir=cache_dir, expected_bit=bit)

    def _train_split():
        return CachedFeatureDataset(dataset, cfg['setting'], 'train',
                                    dataset_root, cache_dir)

    full = _train_split()
    # Bit-identical to stage 1 (base_model.init_experiment) and to our trainer:
    # for CIFAR10 the baseline loader hands carve_val_indices a one-hot [N, 10]
    # matrix while train_siglip2.py hands it the [N] integer `targets`;
    # carve_val_indices collapses the one-hot with argmax, so both paths take the
    # identical branch on identical class ids and yield the identical rows.
    opt_idx, val_idx, strat = carve_val_indices(np.asarray(full.labels),
                                                ratio=ratio, seed=seed)

    val_ds = _train_split(); val_ds.restrict_to(val_idx)
    opt_ds = _train_split(); opt_ds.restrict_to(opt_idx)

    checkpoint_bit = int(cfg['bit'])
    if checkpoint_bit != bit:
        raise ValueError(
            f'{ckpt_dir} uses {checkpoint_bit} bits; this selector invocation '
            f'requires {bit} bits packed as {base_length} DNA bases'
        )
    d_in = int(full.visual_global.shape[1])

    bs = int(cfg.get('batch_size', 64))
    q_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False, num_workers=num_workers)
    d_loader = DataLoader(
        opt_ds, batch_size=bs, shuffle=False, num_workers=num_workers)
    map_r = MAP_AT_R_BY_DATASET[dataset]

    per_epoch = {}
    for path in ckpts:
        ep = int(re.search(r'epoch_(\d+)\.pth$', path).group(1))
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        candidate_cfg = checkpoint.get('config')
        if not isinstance(candidate_cfg, dict):
            raise ValueError(f'{path} lacks checkpoint config')
        _assert_same_stage1_config(cfg, candidate_cfg, path)
        verify_checkpoint_data_context(
            checkpoint, dataset=dataset, setting=cfg['setting'],
            cache_dir=cache_dir, expected_bit=bit)
        model = build_model_from_checkpoint_payload(
            checkpoint, d_in=d_in, device=device)
        cont_q, bin_q, lbl_q = _extract_codes(model, q_loader, device)
        cont_d, bin_d, lbl_d = _extract_codes(model, d_loader, device)
        binary_res = evaluate_retrieval_model(
            query_codes={'B': bin_q, 'C': cont_q},
            retrieval_codes={'B': bin_d, 'C': cont_d},
            query_labels=lbl_q, retrieval_labels=lbl_d,
            precision_at_k_list=(1, 10, 100),
            map_at_r=map_r,
        )
        base_res = evaluate_base_retrieval_model(
            bin_q, bin_d, lbl_q, lbl_d,
            precision_at_k_list=(1, 10, 100), map_at_r=map_r,
        )
        per_epoch[ep] = {
            # Compatibility aliases now intentionally mean the selection
            # metric, i.e. raw L-base Hamming rather than binary Hamming.
            'val_mAP_at_R': base_res['mAP_at_R'],
            'val_mAP': base_res['mAP'],
            'val_P_at_1': base_res['precision_at_k'][1],
            'val_base_mAP_at_R': base_res['mAP_at_R'],
            'val_base_mAP': base_res['mAP'],
            'val_base_P_at_1': base_res['precision_at_k'][1],
            'val_binary_mAP_at_R': binary_res['mAP_at_R'],
            'val_binary_mAP': binary_res['mAP'],
            'checkpoint_load_format': getattr(
                model, '_checkpoint_load_format', 'unknown'),
        }
        print(f'[{method}/{dataset}] epoch {ep:03d}  '
              f'val base-mAP@{map_r}={base_res["mAP_at_R"]:.4f}  '
              f'(binary={binary_res["mAP_at_R"]:.4f})',
              flush=True)

    best_ep = max(per_epoch, key=lambda e: per_epoch[e]['val_base_mAP_at_R'])

    common = {
        'method': method, 'dataset': dataset,
        'val_split': {'ratio': ratio, 'seed': seed, 'strategy': strat,
                      'n_train': int(len(full)), 'n_opt_train': int(len(opt_ds)),
                      'n_val_query': int(len(val_ds)),
                      'val_idx_sha': int(np.sum(val_idx)),
                      'val_idx_head': val_idx[:5].tolist()},
        'map_at_r_cutoff': map_r,
        'bit_length': bit,
        'base_length': base_length,
        'selection_metric': selection_metric,
        'bit_pair_mapping': '00,01,10,11 -> 0,1,2,3',
        'test_touched_during_selection': False,
        'per_epoch_val': {str(k): v for k, v in sorted(per_epoch.items())},
        'best_epoch': int(best_ep),
        'best_val_mAP_at_R': per_epoch[best_ep]['val_base_mAP_at_R'],
        'best_val_base_mAP_at_R': per_epoch[best_ep]['val_base_mAP_at_R'],
        'best_val_binary_mAP_at_R': per_epoch[best_ep]['val_binary_mAP_at_R'],
        'protocol_identity_sha256': cfg.get('protocol_identity_sha256'),
        **checkpoint_artifacts,
        'selected_stage1_checkpoint': {
            'path': os.path.realpath(os.path.join(
                checkpoint_artifacts['stage1_checkpoint_dir'],
                f'epoch_{best_ep:03d}.pth')),
            'sha256': checkpoint_artifacts['stage1_checkpoint_sha256'][
                f'epoch_{best_ep:03d}.pth'],
        },
    }
    if selection_only:
        # This branch deliberately returns before constructing or probing any
        # official test/database path.  It is the safe hand-off between P0
        # stage 1 and a fresh full-train refit at the fixed E*.
        return {
            **common,
            'selection_only': True,
            'test_mAP_at_R': None,
            'test_base_mAP_at_R': None,
        }

    # --- E* is fixed. Only now may the official test/database be touched. ---
    full_trial_result = _trial_dir(
        full_result_root, full_template, method, dataset)
    full_trial_params = _trial_dir(
        full_param_root, full_template, method, dataset)
    test_json = os.path.join(full_trial_result, f'eval_epoch_{best_ep:03d}.json')
    full_ckpt = os.path.join(full_trial_params, f'epoch_{best_ep:03d}.pth')

    persisted = None
    if os.path.isfile(test_json):
        with open(test_json) as handle:
            persisted = json.load(handle)
    if os.path.isfile(full_ckpt):
        # Recompute from the selected checkpoint even when a JSON exists. This
        # prevents a stale result file from a prior trial being silently mixed
        # into a new protocol run.
        full_payload = torch.load(full_ckpt, map_location='cpu', weights_only=False)
        full_cfg = full_payload['config']
        _assert_compatible_full_config(
            cfg, full_cfg, full_ckpt, best_epoch=best_ep)
        full_dataset_root = _resolve_repo_path(full_cfg['dataset_root'])
        full_cache_dir = _resolve_repo_path(
            full_cfg.get('resolved_cache_dir') or full_cfg.get('cache_dir'))
        verify_checkpoint_data_context(
            full_payload, dataset=dataset, setting=full_cfg['setting'],
            cache_dir=full_cache_dir, expected_bit=bit)
        test_ds = CachedFeatureDataset(
            dataset, full_cfg['setting'], 'test',
            full_dataset_root, full_cache_dir)
        database_ds = CachedFeatureDataset(
            dataset, full_cfg['setting'], 'database',
            full_dataset_root, full_cache_dir)
        full_d_in = int(test_ds.visual_global.shape[1])
        full_model = build_model_from_checkpoint_payload(
            full_payload, d_in=full_d_in, device=device)
        full_bs = int(full_cfg.get('batch_size', bs))
        test_loader = DataLoader(
            test_ds, batch_size=full_bs, shuffle=False,
            num_workers=num_workers)
        database_loader = DataLoader(
            database_ds, batch_size=full_bs, shuffle=False,
            num_workers=num_workers)
        cont_q, bin_q, lbl_q = _extract_codes(full_model, test_loader, device)
        cont_d, bin_d, lbl_d = _extract_codes(full_model, database_loader, device)
        test_binary = evaluate_retrieval_model(
            query_codes={'B': bin_q, 'C': cont_q},
            retrieval_codes={'B': bin_d, 'C': cont_d},
            query_labels=lbl_q, retrieval_labels=lbl_d,
            precision_at_k_list=(1, 10, 100), map_at_r=map_r,
        )
        test_base = evaluate_base_retrieval_model(
            bin_q, bin_d, lbl_q, lbl_d,
            precision_at_k_list=(1, 10, 100), map_at_r=map_r,
        )
        persisted_base = (
            persisted.get('base_2bit') if isinstance(persisted, dict) else None)
        if isinstance(persisted_base, dict):
            saved = persisted_base.get('mAP_at_R')
            if saved is None or not np.isclose(
                    float(saved), float(test_base['mAP_at_R']),
                    rtol=0.0, atol=1e-10):
                raise ValueError(
                    f'stale/mismatched test JSON {test_json}: saved base mAP@R '
                    f'{saved!r}, recomputed {test_base["mAP_at_R"]!r}')
        test = {
            'test_mAP_at_R': test_base['mAP_at_R'],
            'test_mAP': test_base['mAP'],
            'test_mAP_R_cutoff': test_base['mAP_R_cutoff'],
            'test_P_at_1': test_base['precision_at_k'][1],
            'test_base_mAP_at_R': test_base['mAP_at_R'],
            'test_binary_mAP_at_R': test_binary['mAP_at_R'],
            'source': os.path.relpath(full_ckpt, _REPO),
            'source_metric': 'recomputed_base_2bit_from_selected_checkpoint',
            'persisted_json_cross_checked': isinstance(persisted_base, dict),
        }
    else:
        reason = 'the selected full-train checkpoint is missing'
        test = {
            'test_mAP_at_R': None,
            'test_base_mAP_at_R': None,
            'MISSING': os.path.relpath(full_ckpt, _REPO),
            'missing_reason': reason,
        }
        print(f'!! MISSING fair base test result for E*={best_ep}: {reason}', flush=True)

    return {**common, 'selection_only': False, **test}


def write_markdown(merged: dict, path: str, bit: int = 36) -> None:
    base_length = _base_length(bit)
    mismatched = {
        key: result.get('bit_length')
        for key, result in merged.items()
        if isinstance(result, dict) and result.get('bit_length') != bit
    }
    if mismatched:
        raise ValueError(
            f'merged results do not match requested bit={bit}: {mismatched}')
    found_methods = {key.split('|', 1)[0] for key in merged if '|' in key}
    methods = [m for m in METHODS if m in found_methods]
    methods += sorted(found_methods.difference(methods))
    lines = [
        f'# P0 stage 2 -- baselines ({bit}-bit / {base_length}-base)',
        '',
        f'E* selected on held-out val by raw {base_length}-base Hamming mAP@R '
        '(val_query vs opt-train DB, 10% carve, seed 42).',
        'Cell = **base-Hamming test mAP@R of the 100%-train run at that E\\***; '
        '(E*) in parentheses.',
        '',
        '| Dataset | ' + ' | '.join(m.upper() for m in methods) + ' |',
        '|---|' + '---|' * len(methods),
    ]
    for d in DATASETS:
        cells = []
        for m in methods:
            r = merged.get(f'{m}|{d}')
            if r is None:
                cells.append('n/a')
            elif r.get('test_mAP_at_R') is None:
                cells.append(f"MISSING (E*={r['best_epoch']})")
            else:
                cells.append(f"{r['test_mAP_at_R']:.4f} (E*={r['best_epoch']})")
        lines.append(f'| {d} | ' + ' | '.join(cells) + ' |')
    lines += ['', '## Selected epochs (E*) and base-Hamming mAP@R', '',
              '| Method | Dataset | E* | val base-mAP@R | test base-mAP@R |',
              '|---|---|---|---|---|']
    for m in methods:
        for d in DATASETS:
            r = merged.get(f'{m}|{d}')
            if r is None:
                continue
            t = r.get('test_mAP_at_R')
            lines.append(f"| {m} | {d} | {r['best_epoch']} | "
                         f"{r['best_val_mAP_at_R']:.4f} | "
                         f"{'MISSING' if t is None else f'{t:.4f}'} |")
    lines += [
        '',
        '## Protocol notes',
        '',
        '* E* is selected **only** on val_query vs opt-train DB using raw '
        f'{base_length}-base Hamming. No test metric enters the selection.',
        f'* Every {bit}-bit code is packed in adjacent pairs with '
        '`00,01,10,11 -> 0,1,2,3`; binary-Hamming values are retained only as '
        'diagnostics and never select E*.',
        '* The val DB is the opt-train split, which is smaller than the official '
        'database. For Flickr25k (val DB = 4500 rows) the R=5000 cutoff never '
        'binds, so the selection metric degenerates to full mAP there. This is '
        'symmetric with our own model (same val DB), but it means the *selection* '
        'metric and the *reported* metric are not the identical statistic on '
        'Flickr25k.',
        '* Absolute val mAP@R is not comparable to test mAP@R: different query '
        'set, much smaller DB, and a different effective cutoff.',
    ]
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--methods', nargs='+', default=list(METHODS))
    p.add_argument('--datasets', nargs='+', default=list(DATASETS))
    p.add_argument(
        '--bit', type=int, choices=SUPPORTED_BITS, default=36,
        help='Hash budget: 36 bits/18 bases or 48 bits/24 bases.',
    )
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--stage1-root', default=STAGE1_ROOT)
    p.add_argument('--full-param-root', default=FULL_PARAM_ROOT)
    p.add_argument('--full-result-root', default=FULL_RESULT_ROOT)
    p.add_argument('--stage1-template', default=DEFAULT_STAGE1_TEMPLATE,
                   help='trial-dir format with {method}, {dataset}, and {short}')
    p.add_argument('--full-template', default=DEFAULT_FULL_TEMPLATE,
                   help='full-train trial-dir format with {method}, {dataset}, and {short}')
    p.add_argument('--out', required=True)
    p.add_argument('--merge', default=None,
                   help='glob of per-pair jsons; merge into --out instead of evaluating')
    p.add_argument('--markdown', default=None)
    p.add_argument('--selection-only', action='store_true',
                   help=('Select E* from held-out validation and return before '
                         'looking for any full-run test artifact.'))
    a = p.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or '.', exist_ok=True)

    if a.merge:
        merged = {}
        for f in sorted(glob.glob(a.merge)):
            r = json.load(open(f))
            merged[f"{r['method']}|{r['dataset']}"] = r
        json.dump(merged, open(a.out, 'w'), indent=2)
        print(f'merged {len(merged)} pairs -> {a.out}')
        if a.markdown:
            write_markdown(merged, a.markdown, bit=a.bit)
            print(f'markdown -> {a.markdown}')
        return 0

    device = a.device if torch.cuda.is_available() else 'cpu'
    results = {}
    for m in a.methods:
        for d in a.datasets:
            results[f'{m}|{d}'] = evaluate_pair(
                m, d, device,
                stage1_root=_resolve_repo_path(a.stage1_root),
                full_param_root=_resolve_repo_path(a.full_param_root),
                full_result_root=_resolve_repo_path(a.full_result_root),
                stage1_template=a.stage1_template,
                full_template=a.full_template,
                num_workers=a.num_workers,
                selection_only=a.selection_only,
                bit=a.bit,
            )
    json.dump(results if len(results) > 1 else list(results.values())[0],
              open(a.out, 'w'), indent=2)
    print(f'wrote {a.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
