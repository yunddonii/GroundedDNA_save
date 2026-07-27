"""Similarity Distribution Calibration (BMVC 2023) baseline.

This is a clean-room implementation of the published SDC objective and hash
head.  The paper freezes its pretrained backbone for SDC; consequently the
repository's frozen-cache protocol is a direct feature-level adaptation rather
than a replacement of a trainable image tower.
"""

from __future__ import annotations

from argparse import ArgumentParser

import torch
from torch.nn import Module
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset

from .base_model import DeepHashBase
from .modern_unsupervised import SDCFeatureHash, sdc_loss, simclr_nt_xent


class SDC(DeepHashBase):
    def _get_default_config_dict(self) -> dict:
        return {
            'batch_size': 64,
            'max_epoch': 100,
            'learning_rate': 1e-4,
            'optimizer_name': 'adam',
            'adam_weight_decay': 1e-5,
            'lr_scheduler': 'step',
            'step_size': 80,
            'step_gamma': 0.1,
            'sdc_beta': 5.0,
            'sdc_reconstruction': 'l1',
            'sdc_quantization': 'cosine',
            'sdc_reconstruction_weight': 1.0,
            'sdc_quantization_weight': 1.0,
            'sdc_variant': 'release_no_cl',
            'sdc_contrastive_temperature': 0.3,
            'sdc_contrastive_weight': 1.0,
        }

    def _get_config_dict_for_dataset(self, default_config: dict, dataset: str) -> dict:
        return default_config

    def _get_fixed_config_dict(self) -> dict:
        return {
            'gen_code_method': 'sign',
            'finetune': False,
            'dataset_return_index': False,
            'dataset_return_visual_tokens': False,
            'transform': 'none',
            'encoder_layers': 'sdc_variant_specific_two_linear_head_with_output_bn',
            'batch_norm': True,
        }

    def _add_model_specific_args_into_parser(self, parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument('--sdc_beta', type=float, default=5.0)
        parser.add_argument('--sdc_reconstruction', choices=('l1', 'l2', 'hinge'), default='l1')
        parser.add_argument('--sdc_quantization', choices=('cosine', 'l1', 'l2'), default='cosine')
        parser.add_argument('--sdc_reconstruction_weight', type=float, default=1.0)
        parser.add_argument('--sdc_quantization_weight', type=float, default=1.0)
        parser.add_argument('--sdc_orthogonal_only', action='store_true')
        parser.add_argument(
            '--sdc_variant',
            choices=('release_no_cl', 'release_simclr', 'paper_cache2v'),
            default='release_no_cl',
            help=('Keep the public no-CL, public SimCLR, and paper Algorithm-1 '
                  'configurations separate.'),
        )
        parser.add_argument('--sdc_contrastive_temperature', type=float, default=0.3)
        parser.add_argument('--sdc_contrastive_weight', type=float, default=1.0)
        return parser

    def _build_model_from_config(self, d_in: int, config: dict) -> Module:
        variant = str(config.get('sdc_variant', 'release_no_cl'))
        hidden = 4096 if variant == 'paper_cache2v' else d_in
        return SDCFeatureHash(d_in=d_in, bit=int(config['bit']), hidden_dim=hidden)

    def init_experiment(self, config: dict, rename: bool = True) -> None:
        """Resolve mutually exclusive, source-traceable SDC variants."""
        variant = str(config.get('sdc_variant', 'release_no_cl'))
        if variant == 'release_no_cl':
            config['dataset_return_paired_aug_img'] = False
            config['adam_weight_decay'] = 1e-5
            config['sdc_orthogonal_only'] = False
            config['implementation_variant'] = 'official-release-default-noCL'
            config['sdc_calibration_scope'] = 'canonical-view random half-pairs'
        elif variant == 'release_simclr':
            config['dataset_return_paired_aug_img'] = True
            config['adam_weight_decay'] = 1e-5
            config['sdc_orthogonal_only'] = True
            config['implementation_variant'] = 'official-release-simclr'
            config['sdc_calibration_scope'] = (
                'both views; within-view half-pairs via release 4-block layout')
        elif variant == 'paper_cache2v':
            config['dataset_return_paired_aug_img'] = True
            config['adam_weight_decay'] = 5e-4
            config['sdc_orthogonal_only'] = False
            config['implementation_variant'] = 'paper-algorithm1-fixed-cache-2view'
            config['sdc_calibration_scope'] = (
                'first view only; random half-pairs; second view enters NT-Xent')
        else:  # parser normally catches this; retain protection for API use.
            raise ValueError(f'unknown SDC variant: {variant!r}')
        config['information_tier'] = 'U0'
        config['audited_release_commit'] = (
            '9981beb206f6be8199185bd6fcd92eb1b597a00d')
        config['information_condition'] = 'visual-only; no labels; frozen cached views'
        super().init_experiment(config, rename=rename)

    def _train_model(self, backbone_with_encoder: Module, optimizer: Optimizer,
                     scheduler: _LRScheduler, trainset: Dataset,
                     testset: Dataset, dbset: Dataset, model_dir: str,
                     result_dir: str, device: str, batch_size: int,
                     eval_period: int, config: dict) -> None:
        loader = DataLoader(
            trainset, batch_size=int(config['batch_size']), shuffle=True,
            num_workers=int(config.get('num_workers', 4)), drop_last=True,
        )
        model = backbone_with_encoder.to(device)
        for epoch in range(int(config['max_epoch'])):
            model.train()
            for batch in loader:
                variant = str(config['sdc_variant'])
                if variant in ('paper_cache2v', 'release_simclr'):
                    first = batch['img_tr1'].to(device)
                    second = batch['img_tr2'].to(device)
                    features = torch.cat((first, second), dim=0)
                else:
                    features = batch['img'].to(device)
                output = model(features)
                if variant == 'paper_cache2v':
                    n = first.shape[0]
                    calibration_features = output['cnn_feat'][:n]
                    calibration_codes = output['continuous_code'][:n]
                    release_four_block_layout = False
                else:
                    calibration_features = output['cnn_feat']
                    calibration_codes = output['continuous_code']
                    release_four_block_layout = (variant == 'release_simclr')
                loss, parts = sdc_loss(
                    calibration_features, calibration_codes,
                    beta=float(config['sdc_beta']),
                    reconstruction=str(config['sdc_reconstruction']),
                    quantization=str(config['sdc_quantization']),
                    reconstruction_weight=float(config['sdc_reconstruction_weight']),
                    quantization_weight=float(config['sdc_quantization_weight']),
                    orthogonal_only=bool(config.get('sdc_orthogonal_only', False)),
                    contrastive_pair_layout=release_four_block_layout,
                )
                if variant in ('paper_cache2v', 'release_simclr'):
                    n = first.shape[0]
                    contrastive = simclr_nt_xent(
                        output['continuous_code'][:n],
                        output['continuous_code'][n:],
                        temperature=float(config['sdc_contrastive_temperature']),
                    )
                    loss = loss + float(config['sdc_contrastive_weight']) * contrastive
                    parts['contrastive'] = contrastive
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                self._insert_iter_loss_sum(
                    {'loss': loss.item(), **{k: v.item() for k, v in parts.items()}},
                    current_batch_size=batch['img'].shape[0],
                )
            scheduler.step()
            self._compute_loss_per_epoch()
            model_id = f'{epoch:03d}'
            self._save_train_model_log(model_id, 'epoch', True)
            # During stage 1, periodic held-out validation supplies E*.  The
            # stage-2 horizon is already fixed to E*, so this training process
            # evaluates only its final checkpoint and makes no test-led choice.
            is_validation_stage = self.valset is not None
            should_eval = (epoch + 1 == int(config['max_epoch'])
                           or (is_validation_stage
                               and (epoch + 1) % int(eval_period) == 0))
            if should_eval:
                self.start_eval_process(model_id, 'epoch')
                self._save_train_model_params(model_id, 'epoch')
