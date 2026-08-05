import argparse
import json
import tempfile
import unittest
from pathlib import Path

from baseline.DUHEG import DUHEG
from scripts.run_modern_baseline_p0 import (
    VARIANTS,
    _base_length,
    _bio_projection_command,
    _checkpoint_set_digest,
    _check_cache,
    _load_verified_selection_artifact,
    _method_extra,
    _require_main_eligibility_or_smoke,
    _selection_metric,
    _sha256,
    _source_horizon,
    _training_command,
    _trial_names,
)


class ModernDriverProtocolTest(unittest.TestCase):
    def test_all_variants_use_the_common_five_epoch_candidate_grid(self) -> None:
        self.assertTrue(VARIANTS)
        self.assertEqual(
            {name: spec['eval_period'] for name, spec in VARIANTS.items()},
            {name: 5 for name in VARIANTS},
        )

    def test_bihalf_release_profiles_are_dataset_specific(self) -> None:
        spec = VARIANTS['bihalf']
        self.assertEqual(_source_horizon(spec, 'Flickr25k'), 100)
        self.assertEqual(_source_horizon(spec, 'MSCOCO'), 150)
        self.assertEqual(_source_horizon(spec, 'CIFAR10'), 300)
        self.assertEqual(_source_horizon(spec, 'NUSWIDE'), 100)

        base = dict(variant='bihalf', hhch_clusters=None)
        flickr = _method_extra(argparse.Namespace(
            **base, dataset='Flickr25k'))
        coco = _method_extra(argparse.Namespace(**base, dataset='MSCOCO'))
        cifar = _method_extra(argparse.Namespace(**base, dataset='CIFAR10'))
        self.assertEqual(
            flickr, ['--learning_rate', '0.0001', '--step_size', '60'])
        self.assertEqual(
            coco, ['--learning_rate', '0.001', '--step_size', '60'])
        self.assertEqual(
            cifar, ['--learning_rate', '0.0001', '--step_size', '120'])

    def test_canonical_variants_retain_source_horizons_and_batches(self) -> None:
        expected = {
            'cibhash': (60, 64, True),
            'cimon': (150, 24, True),
            'mls3rduh': (150, 128, False),
        }
        for variant, (horizon, batch, needs_aug) in expected.items():
            specification = VARIANTS[variant]
            self.assertEqual(_source_horizon(specification, 'Flickr25k'), horizon)
            self.assertEqual(specification['source_batch_size'], batch)
            self.assertEqual(specification['needs_aug'], needs_aug)
            self.assertEqual(specification['tier'], 'U0')

    def test_bit_budget_controls_metric_trials_and_projection_length(self) -> None:
        self.assertEqual(_base_length(36), 18)
        self.assertEqual(_base_length(48), 24)
        self.assertEqual(
            _selection_metric(48),
            'raw_24base_base_hamming_mAP_at_R')
        with self.assertRaisesRegex(ValueError, 'one of'):
            _base_length(32)

        args = argparse.Namespace(
            variant='cibhash', dataset='Flickr25k', bit=48, seed=3,
            device='cuda:2')
        stage1, refit = _trial_names(args, 'a' * 64)
        self.assertIn('_48b_P0s1_', stage1)
        self.assertIn('_48b_P0refit_', refit)
        command = _bio_projection_command(
            args, extraction_dir=Path('/tmp/cell'),
            bio_out=Path('/tmp/bio.json'))
        self.assertEqual(
            command[command.index('--expected_length') + 1], '24')
        self.assertEqual(command[command.index('--device') + 1], 'cuda:2')
        for option in ('--gc_min', '--gc_max', '--max_run'):
            self.assertNotIn(option, command)

    def test_duheg_unverified_adapter_requires_explicit_opt_in(self) -> None:
        args = argparse.Namespace(
            variant='duheg', dataset='Flickr25k', hhch_clusters=None,
            duheg_allow_unverified_selection=False,
            duheg_noun_embeddings='unused.npy',
            duheg_asset_manifest='unused.json',
        )
        with self.assertRaisesRegex(ValueError, 'selected-noun bank'):
            _method_extra(args)

    def test_main_ineligible_inputs_require_explicit_smoke_opt_in(self) -> None:
        with self.assertRaisesRegex(ValueError, 'allow-main-ineligible-smoke'):
            _require_main_eligibility_or_smoke(
                ['cache_meta_missing_canonical_transform'], allow_smoke=False)
        _require_main_eligibility_or_smoke(
            ['cache_meta_missing_canonical_transform'], allow_smoke=True)
        _require_main_eligibility_or_smoke([], allow_smoke=False)

    def test_refit_command_retains_nominal_schedule_horizon(self) -> None:
        args = argparse.Namespace(
            dataset='Flickr25k', setting='setting1', dataset_root='dataset',
            device='cpu', seed=1, val_seed=42, model_root='params',
            result_root='results', compress_root='compress', num_workers=0,
            batch_size=64, bit=48,
        )
        command = _training_command(
            args, method='duheg', cache_dir='cache', trial='refit',
            epochs=15, eval_period=15, val_ratio=0.0,
            schedule_horizon=60, extra=(),
            protocol_digest='a' * 64,
            cache_artifact_hashes={'visual_global.f16.npy': 'b' * 64},
        )
        self.assertEqual(command[command.index('--max_epoch') + 1], '15')
        self.assertEqual(command[command.index('--bit') + 1], '48')
        self.assertEqual(
            command[command.index('--schedule_horizon') + 1], '60')
        self.assertEqual(
            command[command.index('--protocol_identity_sha256') + 1],
            'a' * 64)
        self.assertEqual(
            command[
                command.index('--expected_cache_artifact_sha256_json') + 1],
            '{"visual_global.f16.npy":"' + 'b' * 64 + '"}')

    def test_duheg_cosine_is_not_compressed_to_early_stop_epoch(self) -> None:
        retained = DUHEG._warmup_cosine_lr(30, 60, 10, 1e-5, 1e-4)
        compressed = DUHEG._warmup_cosine_lr(30, 31, 10, 1e-5, 1e-4)
        self.assertGreater(retained, compressed)

    def test_refit_selection_json_is_bound_to_checkpoint_set(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / 'epoch_004.pth'
            second = root / 'epoch_009.pth'
            first.write_bytes(b'first')
            second.write_bytes(b'second')
            hashes = {first.name: _sha256(first), second.name: _sha256(second)}
            selection_path = root / 'selection.json'
            selection = {
                'method': 'sdc', 'dataset': 'Flickr25k',
                'bit_length': 36, 'base_length': 18,
                'selection_metric': 'raw_18base_base_hamming_mAP_at_R',
                'selection_only': True, 'test_touched_during_selection': False,
                'protocol_identity_sha256': 'a' * 64, 'best_epoch': 4,
                'stage1_checkpoint_dir': str(root),
                'stage1_checkpoint_sha256': hashes,
                'stage1_checkpoint_set_sha256': _checkpoint_set_digest(hashes),
                'selected_stage1_checkpoint': {
                    'path': str(first), 'sha256': hashes[first.name]},
            }
            selection_path.write_text(json.dumps(selection), encoding='utf-8')
            verified = _load_verified_selection_artifact(
                selection_path, method='sdc', dataset='Flickr25k',
                protocol_digest='a' * 64, bit=36)
            self.assertEqual(verified['best_epoch'], 4)
            second.write_bytes(b'mutated')
            with self.assertRaisesRegex(ValueError, 'content hashes'):
                _load_verified_selection_artifact(
                    selection_path, method='sdc', dataset='Flickr25k',
                    protocol_digest='a' * 64, bit=36)

            selection['selection_metric'] = 'raw_18base_base_hamming_mAP_at_R'
            selection_path.write_text(json.dumps(selection), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'selection_metric'):
                _load_verified_selection_artifact(
                    selection_path, method='sdc', dataset='Flickr25k',
                    protocol_digest='a' * 64, bit=48)

    def test_legacy_cache_can_run_but_is_main_ineligible(self) -> None:
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'image_ids.json').write_text(
                json.dumps(['a', 'b']), encoding='utf-8')
            (root / 'meta.json').write_text(json.dumps({
                'N': 2, 'D_proj': 3, 'num_tokens': 4, 'H_v': 5,
                'dtype': 'float16', 'backbone': 'clip-test',
            }), encoding='utf-8')
            np.save(
                root / 'visual_global.f16.npy',
                np.zeros((2, 3), dtype=np.float16))
            audit = _check_cache(
                root, dataset='Flickr25k', needs_aug=False,
                needs_tokens=False)
            self.assertIn(
                'cache_meta_missing_canonical_transform',
                audit['eligibility_blockers'])
            self.assertIn(
                'cache_meta_missing_immutable_hf_provenance',
                audit['eligibility_blockers'])
            self.assertIn(
                'visual_global.f16.npy', audit['artifact_hashes'])


if __name__ == '__main__':
    unittest.main()
