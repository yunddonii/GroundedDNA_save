import argparse
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch import nn

import baseline.base_model as bm
from scripts.baseline_val_select_p0 import (
    _assert_compatible_full_config,
    _assert_same_stage1_config,
    _selection_metric as selector_selection_metric,
    _stage1_checkpoint_artifacts,
    write_markdown,
)


class _ToyHash(nn.Module):
    def __init__(self, d_in: int, bit: int):
        super().__init__()
        self.encoder_layers = nn.Linear(d_in, bit, bias=False)
        self.output_gain = nn.Parameter(torch.ones(bit))

    def forward(self, features: torch.Tensor) -> dict:
        code = self.encoder_layers(features.float()) * self.output_gain
        return {'continuous_code': code}


class _ToyBuilder:
    config = {}

    def _build_model_from_config(self, d_in: int, config: dict) -> nn.Module:
        return _ToyHash(d_in, int(config['bit']))


def test_full_checkpoint_uses_registered_custom_builder_and_all_weights():
    source = _ToyHash(5, 4)
    with torch.no_grad():
        source.encoder_layers.weight.copy_(torch.arange(20).reshape(4, 5))
        source.output_gain.copy_(torch.tensor([0.5, 1.5, 2.0, 3.0]))
    payload = {
        'config': {'method': 'toy', 'bit': 4},
        'model_state_dict': source.state_dict(),
        # Deliberately include an encoder-only state as well: full state must win.
        'encoder_layers': source.encoder_layers.state_dict(),
    }

    with mock.patch.object(bm, '_build_method', lambda method: _ToyBuilder()):
        restored = bm.build_model_from_checkpoint_payload(payload, d_in=5)
    assert restored._checkpoint_load_format == 'model_state_dict'
    torch.testing.assert_close(restored.output_gain, source.output_gain)
    x = torch.randn(3, 5)
    torch.testing.assert_close(
        restored(x)['continuous_code'], source(x)['continuous_code'])


def test_legacy_encoder_only_checkpoint_remains_loadable():
    source = bm.BackboneWithEncoder(
        d_in=7, bit=6, hidden_nodes=[], batch_norm=False)
    payload = {
        'config': {
            'method': 'cibhash', 'bit': 6,
            'encoder_layers': 'layer=1', 'batch_norm': False,
        },
        'encoder_layers': source.encoder_layers.state_dict(),
    }

    restored = bm.build_model_from_checkpoint_payload(payload, d_in=7)
    assert restored._checkpoint_load_format == 'encoder_layers'
    x = torch.randn(2, 7)
    torch.testing.assert_close(
        restored(x)['continuous_code'], source(x)['continuous_code'])


def test_base_hamming_is_the_selection_metric_not_binary_hamming():
    # The relevant database item is first. Relative to query 00, relevant 11
    # has two bit flips while irrelevant 01 has one; binary Hamming ranks the
    # irrelevant item first. Base Hamming treats both as one substituted base,
    # then stable index order ranks the relevant item first.
    query_labels = np.array([[1, 0]], dtype=np.int64)
    database_labels = np.array([[1, 0], [0, 1]], dtype=np.int64)
    for bit in (36, 48):
        query = -np.ones((1, bit), dtype=np.int8)  # 00 repeated
        relevant = query.copy()
        relevant[0, :2] = 1                       # first base: 00 -> 11
        irrelevant = query.copy()
        irrelevant[0, 1] = 1                     # first base: 00 -> 01
        database = np.concatenate((relevant, irrelevant), axis=0)

        binary = bm.evaluate_retrieval_model(
            {'B': query}, {'B': database}, query_labels, database_labels,
            precision_at_k_list=(1,), map_at_r=2)
        base = bm.evaluate_base_retrieval_model(
            query, database, query_labels, database_labels,
            precision_at_k_list=(1,), map_at_r=2)
        assert binary['mAP_at_R'] == 0.5
        assert base['mAP_at_R'] == 1.0
        assert bm.signed_bits_to_base_indices(database).shape == (2, bit // 2)


def test_selector_labels_48_bit_results_as_24_base(tmp_path=None):
    assert selector_selection_metric(36) == 'raw_18base_base_hamming_mAP_at_R'
    assert selector_selection_metric(48) == 'raw_24base_base_hamming_mAP_at_R'
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / 'summary.md'
        write_markdown({
            'cibhash|Flickr25k': {
                'bit_length': 48, 'best_epoch': 4,
                'best_val_mAP_at_R': 0.5, 'test_mAP_at_R': 0.4,
            },
        }, str(output), bit=48)
        text = output.read_text(encoding='utf-8')
        assert '48-bit / 24-base' in text
        assert 'raw 24-base Hamming' in text


class _TinyDataset:
    def __init__(self, cache_dir):
        self.cache_dir = str(cache_dir)
        self.visual_global = np.zeros((20, 5), dtype=np.float32)
        self.labels = np.eye(10, dtype=np.int64)[np.arange(20) % 10]
        self.rows = np.arange(20)

    def restrict_to(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        self.rows = self.rows[indices]
        self.labels = self.labels[indices]

    def __len__(self):
        return len(self.rows)


class _TinyMethod(bm.DeepHashBase):
    def _get_default_config_dict(self):
        return {}

    def _get_config_dict_for_dataset(self, default_config, dataset):
        return default_config

    def _get_fixed_config_dict(self):
        return {}

    def _add_model_specific_args_into_parser(
            self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        return parser

    def _train_model(self, *args, **kwargs):
        raise AssertionError('not used')


def test_stage1_initialization_does_not_load_test_or_database():
    calls = []

    temporary = tempfile.TemporaryDirectory()
    tmp_path = Path(temporary.name)
    np.save(
        tmp_path / 'visual_global.f16.npy',
        np.zeros((20, 5), dtype=np.float32))

    def fake_load_dataset(*args, **kwargs):
        calls.append(kwargs.copy())
        return _TinyDataset(tmp_path), None, None

    with mock.patch.object(bm, 'load_dataset', fake_load_dataset), \
            mock.patch.object(
                bm, 'CachedFeatureDataset',
                lambda *args, **kwargs: _TinyDataset(tmp_path)):
        method = _TinyMethod()
        config = {
            'trial_name': 'stage1-no-test',
            'dataset': 'CIFAR10', 'setting': 'setting1',
            'dataset_root': str(tmp_path), 'cache_dir': str(tmp_path),
            'dataset_return_index': False,
            'dataset_return_paired_aug_img': False,
            'dataset_return_visual_tokens': False,
            'bit': 36, 'backbone': 'cached', 'encoder_layers': 'layer=1',
            'encoder_type': 'linear', 'batch_norm': False,
            'optimizer_name': 'adam', 'learning_rate': 1e-4,
            'adam_weight_decay': 0.0, 'lr_scheduler': 'none',
            'finetune': False, 'gen_code_method': 'sign',
            'seed': 1, 'device': 'cpu', 'batch_size': 4, 'eval_period': 1,
            'val_split_ratio': 0.2, 'val_split_seed': 42,
            'model_root': str(tmp_path / 'params'),
            'result_root': str(tmp_path / 'results'),
            'compress_root': str(tmp_path / 'compress'),
        }
        method.init_experiment(config)

    assert calls and calls[0]['load_test'] is False
    assert calls[0]['load_database'] is False
    assert method.testset is None and method.dbset is None
    assert method.valset is not None
    temporary.cleanup()


def test_checkpoint_cache_identity_rejects_same_width_stale_cache():
    temporary = tempfile.TemporaryDirectory()
    root = Path(temporary.name)
    meta = root / 'meta.json'
    image_ids = root / 'image_ids.json'
    meta.write_text(
        '{"backbone":"clip-a","D_proj":5}', encoding='utf-8')
    image_ids.write_text('["a","b"]', encoding='utf-8')
    visual = root / 'visual_global.f16.npy'
    np.save(visual, np.zeros((2, 5), dtype=np.float16))
    checkpoint = {
        'config': {
            'dataset': 'Flickr25k', 'setting': 'setting1', 'bit': 36,
            'resolved_cache_dir': str(root),
            'resolved_backbone': 'clip-a',
            'resolved_projection_dim': 5,
            'resolved_cache_meta_sha256': bm._sha256_file(str(meta)),
            'resolved_cache_image_ids_sha256': bm._sha256_file(str(image_ids)),
            'resolved_cache_artifact_sha256': {
                visual.name: bm._sha256_file(str(visual)),
            },
        },
    }
    bm.verify_checkpoint_data_context(
        checkpoint, dataset='Flickr25k', setting='setting1',
        cache_dir=str(root), expected_bit=36)
    meta.write_text(
        '{"backbone":"clip-a","D_proj":5,"changed":true}', encoding='utf-8')
    with unittest.TestCase().assertRaisesRegex(ValueError, 'meta SHA-256'):
        bm.verify_checkpoint_data_context(
            checkpoint, dataset='Flickr25k', setting='setting1',
            cache_dir=str(root), expected_bit=36)
    meta.write_text(
        '{"backbone":"clip-a","D_proj":5}', encoding='utf-8')
    image_ids.write_text('["b","a"]', encoding='utf-8')
    with unittest.TestCase().assertRaisesRegex(ValueError, 'image order SHA-256'):
        bm.verify_checkpoint_data_context(
            checkpoint, dataset='Flickr25k', setting='setting1',
            cache_dir=str(root), expected_bit=36)
    image_ids.write_text('["a","b"]', encoding='utf-8')
    np.save(visual, np.ones((2, 5), dtype=np.float16))
    stat = visual.stat()
    os.utime(visual, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    with unittest.TestCase().assertRaisesRegex(
            ValueError, 'consumed-array SHA-256'):
        bm.verify_checkpoint_data_context(
            checkpoint, dataset='Flickr25k', setting='setting1',
            cache_dir=str(root), expected_bit=36)
    temporary.cleanup()


def test_selector_rejects_protocol_or_semantic_asset_drift():
    common = {
        'method': 'duheg', 'dataset': 'Flickr25k', 'setting': 'setting1',
        'bit': 36, 'trial_name': 'trial', 'seed': 1,
        'val_split_ratio': 0.1, 'val_split_seed': 42,
        'max_epoch': 60, 'eval_period': 5, 'schedule_horizon': 60,
        'batch_size': 64, 'learning_rate': 1e-4,
        'optimizer_name': 'adam', 'adam_weight_decay': 0.0,
        'sgd_weight_decay': 1e-5, 'sgd_momentum': 0.99,
        'lr_scheduler': 'none', 'gen_code_method': 'sign',
        'finetune': False, 'dataset_return_paired_aug_img': True,
        'dataset_return_visual_tokens': False,
        'resolved_backbone': 'clip-a', 'resolved_projection_dim': 5,
        'resolved_cache_dir': '/tmp/cache-a',
        'resolved_cache_artifact_sha256': {
            'visual_global.f16.npy': 'f' * 64},
        'cache_artifact_binding_status': 'verified_against_driver',
        'protocol_identity_sha256': 'a' * 64,
        'duheg_noun_bank_sha256': 'b' * 64,
        'duheg_asset_manifest_sha256': 'e' * 64,
        'duheg_selection_verified': False,
        'duheg_allow_unverified_selection': True,
        'duheg_matched_official_taxonomy_dataset': '',
        'duheg_n_nouns': 1000, 'duheg_text_dim': 5,
        'implementation_variant': 'released-objective-fixed-cache-2view',
        'publishable_as_exact_duheg': False,
        'duheg_temperature': 0.8, 'duheg_positive_threshold': 0.97,
        'duheg_guidance_temperature': 0.004,
        'duheg_guidance_precision': 'release_fp16',
        'duheg_warmup_epochs': 10, 'duheg_min_lr': 1e-5,
        'information_tier': 'U1',
        'duheg_term_source_condition': 'external-term-bank-selection-unverified',
    }
    drifted_asset = {**common, 'duheg_noun_bank_sha256': 'c' * 64}
    with unittest.TestCase().assertRaisesRegex(ValueError, 'config drift'):
        _assert_same_stage1_config(common, drifted_asset, 'epoch_004.pth')

    refit = {**common, 'trial_name': 'refit'}
    refit['protocol_identity_sha256'] = 'd' * 64
    with unittest.TestCase().assertRaisesRegex(ValueError, 'incompatible'):
        _assert_compatible_full_config(common, refit, 'epoch_004.pth')

    refit = {
        **common, 'trial_name': 'refit', 'val_split_ratio': 0.0,
        'max_epoch': 5, 'eval_period': 5,
    }
    _assert_compatible_full_config(
        common, refit, 'epoch_004.pth', best_epoch=4)
    refit['duheg_positive_threshold'] = 0.5
    with unittest.TestCase().assertRaisesRegex(ValueError, 'incompatible'):
        _assert_compatible_full_config(
            common, refit, 'epoch_004.pth', best_epoch=4)


def test_selector_allows_only_crh_stage_partition_audit_counts_to_differ():
    common = {
        'method': 'crh', 'dataset': 'Flickr25k', 'setting': 'setting1',
        'bit': 36, 'trial_name': 'stage1', 'seed': 42,
        'val_split_ratio': 0.1, 'val_split_seed': 42,
        'max_epoch': 30, 'eval_period': 1, 'schedule_horizon': 30,
        'batch_size': 128, 'learning_rate': 1e-4,
        'optimizer_name': 'adam', 'adam_weight_decay': 1e-5,
        'sgd_weight_decay': 1e-5, 'sgd_momentum': 0.99,
        'lr_scheduler': 'cosine', 'gen_code_method': 'sign',
        'finetune': False, 'dataset_return_paired_aug_img': False,
        'dataset_return_visual_tokens': False,
        'resolved_backbone': 'clip-a', 'resolved_projection_dim': 512,
        'resolved_cache_dir': '/tmp/cache-a',
        'resolved_cache_artifact_sha256': {
            'visual_global.f16.npy': 'f' * 64},
        'cache_artifact_binding_status': 'verified_against_driver',
        'protocol_identity_sha256': 'a' * 64,
        'implementation_variant': 'paper-equation-matched-cache-adapter',
        'information_tier': 'S',
        'supervision_regime': 'supervised',
        'uses_train_labels_in_objective': True,
        'uses_train_labels_in_center_reassignment': True,
        'source_code_commit': 'b' * 40,
        'numeric_precision': 'float32',
        'crh_margin': 0.2,
        'crh_scale': 4.0,
        'crh_quantization_weight': 0.0,
        'crh_codebook_scale': 2,
        'crh_codebook_size': 48,
        'crh_num_classes': 24,
        'crh_head_dim': 6,
        'crh_num_heads': 6,
        'crh_update_rounds': 20,
        'crh_update_interval': 5,
        'crh_gradient_clip_norm': 1.0,
        'crh_source_horizon': 30,
        'crh_source_optimizer': (
            'Adam(lr=1e-4, betas=(0.5,0.999), weight_decay=1e-5)'
        ),
        'crh_source_scheduler': (
            'CosineAnnealingLR(T_max=source_horizon, eta_min=1e-7)'
        ),
        'crh_training_partition_stage': 'stage1_opt_train',
        'crh_training_rows_total': 4500,
        'crh_training_rows_retained': 4424,
        'crh_training_zero_label_rows_dropped': 76,
        'crh_training_class_positive_counts': [100, 200],
        'crh_training_all_classes_present': True,
        'crh_zero_label_policy': (
            'drop_from_current_stage_training_loader_only'
        ),
        'crh_training_filter_scope': (
            'training-loader-only; validation/test/database datasets unchanged'
        ),
    }
    refit = {
        **common,
        'trial_name': 'refit',
        'val_split_ratio': 0.0,
        'max_epoch': 5,
        'eval_period': 5,
        'crh_training_partition_stage': 'stage2_full_train',
        'crh_training_rows_total': 5000,
        'crh_training_rows_retained': 4919,
        'crh_training_zero_label_rows_dropped': 81,
        'crh_training_class_positive_counts': [120, 220],
    }
    _assert_compatible_full_config(
        common, refit, 'epoch_004.pth', best_epoch=4)

    # The filtering policy itself is scientific configuration and must remain
    # fail-closed; only the stage-dependent audit observations are ignored.
    for key, changed in (
        ('crh_zero_label_policy', 'silently-keep-invalid-rows'),
        ('crh_margin', 0.3),
        ('crh_codebook_scale', 3),
        ('resolved_cache_dir', '/tmp/cache-b'),
        ('protocol_identity_sha256', 'c' * 64),
    ):
        drifted = {**refit, key: changed}
        with unittest.TestCase().assertRaisesRegex(ValueError, 'incompatible'):
            _assert_compatible_full_config(
                common, drifted, 'epoch_004.pth', best_epoch=4)

    # Required CRH fields fail closed even when the same omission would
    # otherwise disappear from the union-based drift comparison.
    for key in (
        'crh_margin',
        'crh_codebook_scale',
        'crh_zero_label_policy',
        'crh_training_rows_total',
        'source_code_commit',
    ):
        incomplete_stage1 = {k: v for k, v in common.items() if k != key}
        incomplete_refit = {k: v for k, v in refit.items() if k != key}
        with unittest.TestCase().assertRaisesRegex(
            ValueError, 'lacks selection-critical config keys'
        ):
            _assert_compatible_full_config(
                incomplete_stage1,
                incomplete_refit,
                'epoch_004.pth',
                best_epoch=4,
            )

    # The exception is method-scoped: CRH-shaped keys accidentally attached to
    # another baseline remain scientific config and cannot drift silently.
    foreign_stage1 = {**common, 'method': 'toy'}
    foreign_refit = {**refit, 'method': 'toy', 'crh_training_rows_total': 5001}
    with unittest.TestCase().assertRaisesRegex(ValueError, 'incompatible'):
        _assert_compatible_full_config(
            foreign_stage1, foreign_refit, 'epoch_004.pth', best_epoch=4)


def test_selection_manifest_binds_every_candidate_checkpoint():
    temporary = tempfile.TemporaryDirectory()
    root = Path(temporary.name)
    first = root / 'epoch_004.pth'
    second = root / 'epoch_009.pth'
    first.write_bytes(b'first')
    second.write_bytes(b'second')
    artifacts = _stage1_checkpoint_artifacts([str(first), str(second)])
    assert set(artifacts['stage1_checkpoint_sha256']) == {
        first.name, second.name}
    original_set_hash = artifacts['stage1_checkpoint_set_sha256']
    second.write_bytes(b'changed')
    mutated = _stage1_checkpoint_artifacts([str(first), str(second)])
    assert mutated['stage1_checkpoint_set_sha256'] != original_set_hash
    temporary.cleanup()


def test_cifar_split_paths_use_global_cache_image_ids():
    from dataloaders import _cifar10_image_id

    train_data = np.stack((
        np.zeros((2, 2, 3), dtype=np.uint8),
        np.ones((2, 2, 3), dtype=np.uint8),
    ))
    test_data = np.stack((
        np.full((2, 2, 3), 2, dtype=np.uint8),
        np.full((2, 2, 3), 3, dtype=np.uint8),
    ))

    class FakeCIFAR10:
        def __init__(self, root, train, download):
            del root, download
            self.data = train_data if train else test_data
            self.targets = [0, 1]

    all_ids = [_cifar10_image_id(array)
               for array in np.concatenate((train_data, test_data))]
    dataset = bm.CachedFeatureDataset.__new__(bm.CachedFeatureDataset)
    dataset.image_ids = all_ids
    dataset.id_to_row = {identifier: row
                         for row, identifier in enumerate(all_ids)}
    with mock.patch('torchvision.datasets.CIFAR10', FakeCIFAR10), \
            mock.patch(
                'dataloaders.get_idx_for_uniform_sampling',
                return_value=np.array([1, 0], dtype=np.int64)):
        dataset._build_cifar10('setting1', 'test', '/tmp/unused')
    self_expected = [
        _cifar10_image_id(test_data[1]), _cifar10_image_id(test_data[0])]
    assert dataset.paths == self_expected
    assert all(not value.startswith('cifar10:') for value in dataset.paths)


def test_all_registered_modern_heads_restore_full_state():
    configurations = {
        'greedyhash': {'bit': 36},
        'bihalf': {'bit': 36},
        'sdc': {'bit': 36, 'sdc_variant': 'paper_cache2v'},
        'hhch': {
            'bit': 36, 'hhch_hyper_dim': 8, 'hhch_curvature': 0.01,
            'hhch_dropout': 0.0, 'hhch_clip_radius': 2.3,
        },
        'crovca': {'bit': 36},
        'duheg': {'bit': 36, 'duheg_text_dim': 5},
        'umrch': {'bit': 36},
    }
    features = torch.randn(3, 5)
    for method_name, method_config in configurations.items():
        config = {
            'method': method_name, 'resolved_projection_dim': 5,
            **method_config,
        }
        builder = bm._build_method(method_name)
        builder.config = dict(config)
        source = builder._build_model_from_config(5, config).eval()
        payload = {'config': config, 'model_state_dict': source.state_dict()}
        restored = bm.build_model_from_checkpoint_payload(
            payload, d_in=5, device='cpu').eval()
        self_state = source.state_dict()
        restored_state = restored.state_dict()
        assert self_state.keys() == restored_state.keys()
        for name in self_state:
            torch.testing.assert_close(self_state[name], restored_state[name])
        torch.testing.assert_close(
            source(features)['continuous_code'],
            restored(features)['continuous_code'],
        )


class BaselineCheckpointProtocolTest(unittest.TestCase):
    def test_full_custom_checkpoint(self):
        test_full_checkpoint_uses_registered_custom_builder_and_all_weights()

    def test_legacy_checkpoint(self):
        test_legacy_encoder_only_checkpoint_remains_loadable()

    def test_base_hamming_selection(self):
        test_base_hamming_is_the_selection_metric_not_binary_hamming()

    def test_stage1_no_test_loading(self):
        test_stage1_initialization_does_not_load_test_or_database()

    def test_selector_48_bit_labels(self):
        test_selector_labels_48_bit_results_as_24_base()

    def test_checkpoint_cache_identity(self):
        test_checkpoint_cache_identity_rejects_same_width_stale_cache()

    def test_all_modern_checkpoint_heads(self):
        test_all_registered_modern_heads_restore_full_state()

    def test_selector_provenance_drift(self):
        test_selector_rejects_protocol_or_semantic_asset_drift()

    def test_selection_checkpoint_artifact_binding(self):
        test_selection_manifest_binds_every_candidate_checkpoint()

    def test_cifar_cache_identity_paths(self):
        test_cifar_split_paths_use_global_cache_image_ids()


if __name__ == '__main__':
    unittest.main()
