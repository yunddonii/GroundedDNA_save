import argparse
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from baseline.DUHEG import DUHEG
from scripts.run_modern_baseline_p0 import (
    VARIANTS,
    AUTHOR_FIXED_FINAL,
    AUTHOR_FIXED_SINGLE_STAGE,
    P0_STAGE2_REFIT_TEST,
    VALIDATION_SENSITIVITY,
    _author_trial_name,
    _base_length,
    _bio_projection_command,
    _checkpoint_set_digest,
    _check_cache,
    _dataset_source_binding,
    _dataset_source_relative_paths,
    _load_verified_selection_artifact,
    _method_extra,
    _require_main_eligibility_or_smoke,
    _require_single_terminal_checkpoint,
    _read_verified_checkpoint_protocol,
    _selection_metric,
    _sha256,
    _source_horizon,
    _training_command,
    _trial_names,
    _verify_dataset_source_binding,
)
from scripts.run_baseline_p0_matrix import (
    Attempt,
    Job,
    _command as _matrix_command,
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
            protocol_mode=VALIDATION_SENSITIVITY,
            protocol_stage=P0_STAGE2_REFIT_TEST,
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

    def test_author_fixed_command_is_full_train_horizon_and_last_only(self) -> None:
        args = argparse.Namespace(
            dataset='Flickr25k', setting='setting1', dataset_root='dataset',
            device='cpu', seed=42, model_root='params',
            result_root='results', compress_root='compress', num_workers=0,
            batch_size=None, bit=30, variant='cibhash',
        )
        trial = _author_trial_name(args, 'a' * 64)
        self.assertIn('_30b_D6authorLAST_', trial)
        command = _training_command(
            args, method='cibhash', cache_dir=Path('/cache'), trial=trial,
            epochs=60, eval_period=60, val_ratio=0.0,
            schedule_horizon=60, extra=(), val_seed=None,
            protocol_digest='a' * 64,
            protocol_mode=AUTHOR_FIXED_FINAL,
            protocol_stage=AUTHOR_FIXED_SINGLE_STAGE,
        )
        self.assertEqual(command[command.index('--max_epoch') + 1], '60')
        self.assertEqual(command[command.index('--schedule_horizon') + 1], '60')
        self.assertEqual(command[command.index('--eval_period') + 1], '60')
        self.assertEqual(command[command.index('--val_split_ratio') + 1], '0.0')
        self.assertEqual(
            command[command.index('--protocol_mode') + 1],
            AUTHOR_FIXED_FINAL)
        self.assertEqual(
            command[command.index('--protocol_stage') + 1],
            AUTHOR_FIXED_SINGLE_STAGE)
        self.assertNotIn('--val_split_seed', command)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory)
            terminal = checkpoint_dir / 'epoch_059.pth'
            terminal.write_bytes(b'last')
            self.assertEqual(
                _require_single_terminal_checkpoint(
                    checkpoint_dir, final_epoch_zero_based=59),
                terminal,
            )
            (checkpoint_dir / 'epoch_004.pth').write_bytes(b'candidate')
            with self.assertRaisesRegex(ValueError, 'exactly terminal LAST'):
                _require_single_terminal_checkpoint(
                    checkpoint_dir, final_epoch_zero_based=59)

    def test_final_checkpoint_protocol_is_reopened_and_tamper_rejected(self) -> None:
        expected = {
            'protocol_mode': AUTHOR_FIXED_FINAL,
            'protocol_stage': AUTHOR_FIXED_SINGLE_STAGE,
            'protocol_identity_sha256': 'a' * 64,
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'epoch_059.pth'
            torch.save({'config': dict(expected), 'model_state_dict': {}}, checkpoint)
            self.assertEqual(
                _read_verified_checkpoint_protocol(
                    checkpoint,
                    protocol_mode=AUTHOR_FIXED_FINAL,
                    protocol_stage=AUTHOR_FIXED_SINGLE_STAGE,
                    protocol_digest='a' * 64),
                expected)

            payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
            payload['config']['protocol_stage'] = P0_STAGE2_REFIT_TEST
            torch.save(payload, checkpoint)
            with self.assertRaisesRegex(ValueError, 'protocol contract mismatch'):
                _read_verified_checkpoint_protocol(
                    checkpoint,
                    protocol_mode=AUTHOR_FIXED_FINAL,
                    protocol_stage=AUTHOR_FIXED_SINGLE_STAGE,
                    protocol_digest='a' * 64)

            checkpoint.write_bytes(b'tampered-not-a-checkpoint')
            with self.assertRaisesRegex(ValueError, 'cannot reopen'):
                _read_verified_checkpoint_protocol(
                    checkpoint,
                    protocol_mode=AUTHOR_FIXED_FINAL,
                    protocol_stage=AUTHOR_FIXED_SINGLE_STAGE,
                    protocol_digest='a' * 64)

    def test_cifar_identity_binds_consumed_batches_and_detects_label_tamper(self) -> None:
        expected = tuple(
            [f"CIFAR10/cifar-10-batches-py/data_batch_{index}"
             for index in range(1, 6)]
            + ["CIFAR10/cifar-10-batches-py/test_batch",
               "CIFAR10/cifar-10-batches-py/batches.meta"])
        self.assertEqual(
            _dataset_source_relative_paths("CIFAR10", "setting1"), expected)
        self.assertNotIn("CIFAR10/cifar-10-python.tar.gz", expected)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, relative in enumerate(expected):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"source-{index}".encode())
            dataset_root, snapshot = _dataset_source_binding(
                "CIFAR10", "setting1", root)
            identity = {
                "dataset_root": dataset_root,
                "split_sha256": snapshot,
            }
            digest_before = hashlib.sha256(json.dumps(
                identity, sort_keys=True, separators=(",", ":"),
            ).encode()).hexdigest()
            self.assertEqual(set(snapshot), set(expected))
            _verify_dataset_source_binding(
                identity, dataset="CIFAR10", setting="setting1")

            # A label-only pickle edit must invalidate the sealed identity even
            # when the feature-cache image mapping remains byte-identical.
            (root / expected[0]).write_bytes(b"source-0-label-tampered")
            with self.assertRaisesRegex(ValueError, "split/source changed"):
                _verify_dataset_source_binding(
                    identity, dataset="CIFAR10", setting="setting1")
            _, changed_snapshot = _dataset_source_binding(
                "CIFAR10", "setting1", root)
            changed_identity = {**identity, "split_sha256": changed_snapshot}
            digest_after = hashlib.sha256(json.dumps(
                changed_identity, sort_keys=True, separators=(",", ":"),
            ).encode()).hexdigest()
            self.assertNotEqual(digest_before, digest_after)

    def test_matrix_author_command_passes_cache_without_forced_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempt = Attempt(
                job=Job('u0', 'cibhash', 'CIFAR10', 30, 42),
                number=1,
                model_root=root / 'model',
                result_root=root / 'result',
                compress_root=root / 'compress',
                log_path=root / 'attempt.log',
                status_path=root / 'attempt.json',
            )
            cache = root / 'cache'
            assignment = {
                'physical_index': 0, 'uuid': 'GPU-TEST-0000',
                'name': 'Synthetic GPU', 'pci_bus_id': '0000:01:00.0',
                'driver_version': 'test',
            }
            command = _matrix_command(
                attempt, python=Path('/python'),
                gpu_assignment=assignment, num_workers=0,
                protocol_mode='author_fixed_final',
                cache_dirs={'CIFAR10': cache},
                allow_main_ineligible_smoke=False,
            )
            self.assertEqual(
                command[command.index('--protocol-mode') + 1],
                'author_fixed_final')
            self.assertEqual(command[command.index('--bit') + 1], '30')
            self.assertEqual(command[command.index('--cache-dir') + 1], str(cache))
            self.assertEqual(
                json.loads(command[
                    command.index('--matrix-assigned-gpu-json') + 1]),
                assignment)
            self.assertNotIn('--allow-main-ineligible-smoke', command)

            smoke_command = _matrix_command(
                attempt, python=Path('/python'),
                gpu_assignment=assignment, num_workers=0,
                protocol_mode='author_fixed_final',
                cache_dirs={'CIFAR10': cache},
                allow_main_ineligible_smoke=True,
            )
            self.assertIn('--allow-main-ineligible-smoke', smoke_command)

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
            self.assertTrue(any(
                blocker.startswith('cache_decode_failure_audit_invalid:')
                for blocker in audit['eligibility_blockers']))
            self.assertEqual(
                audit['decode_failure_audit']['status'], 'invalid')
            self.assertIn(
                'visual_global.f16.npy', audit['artifact_hashes'])


if __name__ == '__main__':
    unittest.main()
