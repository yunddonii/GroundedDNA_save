"""Hierarchical Hyperbolic Contrastive Hashing (IEEE TIP 2024).

This is a clean-room ``paper_cache3v`` implementation.  It follows the
published HHCH equations while replacing the frozen VGG19 image tower with the
repository's frozen visual-feature cache.  Each training row must therefore
provide one canonical feature and two independently augmented cached features.

The variant name is intentional.  The public HHCH release (commit
47eac18aedf3a2f8623002a702da28aec8aafd0b) differs from the paper in material
ways: random rather than K-Means++ initialization, MSE rather than LogCosh
quantization, default tau=0.3 rather than 0.2, and an asymmetric scaling of the
two instance-anchor directions.  This module implements the paper choices and
records those discrepancies in every experiment config; it does not silently
combine the two objectives.
"""

from __future__ import annotations

from argparse import ArgumentParser
from typing import Any

import torch
from torch.nn import Module
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset

from .base_model import DeepHashBase
from .modern_unsupervised import (
    HHCHFeatureHash,
    hhch_hierarchical_kmeans,
    hhch_loss,
)


_PAPER_CLUSTER_COUNTS = {
    'CIFAR10': (100, 80, 50),
    'Flickr25k': (200, 150, 80),
    'NUSWIDE': (200, 150, 80),
}


def _parse_cluster_counts(specification: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in specification.split(','))
    except ValueError as error:
        raise ValueError(
            'HHCH cluster counts must be comma-separated integers') from error
    if not values or any(value <= 0 for value in values):
        raise ValueError('HHCH cluster counts must all be positive')
    if any(next_value > value for value, next_value in zip(values, values[1:])):
        raise ValueError('HHCH hierarchy must be non-increasing from bottom to top')
    return values


class HHCH(DeepHashBase):
    """Visual-only self-supervised HHCH under the matched cache protocol."""

    def _get_default_config_dict(self) -> dict:
        return {
            'batch_size': 64,
            # The journal paper does not state T; 80 is the public runner's
            # default and is recorded as such rather than attributed to Eq. 16.
            'max_epoch': 80,
            'eval_period': 2,
            'learning_rate': 1e-3,
            'optimizer_name': 'adam',
            'adam_weight_decay': 0.0,
            'lr_scheduler': 'hhch_release_piecewise',
            'hhch_variant': 'paper_cache3v',
            'hhch_temperature': 0.2,
            'hhch_curvature': 0.01,
            'hhch_hyper_dim': 128,
            'hhch_quantization_weight': 0.01,
            'hhch_clusters': 'auto',
            'hhch_kmeans_max_iter': 30,
            'hhch_kmeans_tolerance': 1e-3,
            'hhch_cluster_chunk_size': 4096,
            'hhch_dropout': 0.0,
            # The released projection uses this stability clip, which is not
            # specified in the paper prose.  It remains explicit and tunable.
            'hhch_clip_radius': 2.3,
            'hhch_num_workers': 4,
        }

    def _get_config_dict_for_dataset(self, default_config: dict,
                                     dataset: str) -> dict:
        return default_config

    def _get_fixed_config_dict(self) -> dict:
        return {
            'gen_code_method': 'sign',
            'finetune': False,
            'dataset_return_index': True,
            'dataset_return_paired_aug_img': True,
            'dataset_return_visual_tokens': False,
            'transform': 'HHCH-cache3v',
            'encoder_layers': 'hhch_d_to_512_to_bit',
            'batch_norm': False,
        }

    def _add_model_specific_args_into_parser(
            self, parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument('--hhch_variant', choices=('paper_cache3v',),
                            default='paper_cache3v')
        parser.add_argument('--hhch_temperature', type=float, default=0.2)
        parser.add_argument('--hhch_curvature', type=float, default=0.01)
        parser.add_argument('--hhch_hyper_dim', type=int, default=128)
        parser.add_argument('--hhch_quantization_weight', type=float, default=0.01)
        parser.add_argument(
            '--hhch_clusters', type=str, default='auto',
            help=('Bottom-to-top cluster counts. auto uses the paper values '
                  'for CIFAR-10/Flickr25k/NUS-WIDE and a clearly tagged '
                  '200,150,80 matched adaptation for MSCOCO. CUB requires an '
                  'explicit value because HHCH reports no CUB setting.'),
        )
        parser.add_argument('--hhch_kmeans_max_iter', type=int, default=30)
        parser.add_argument('--hhch_kmeans_tolerance', type=float, default=1e-3)
        parser.add_argument('--hhch_cluster_chunk_size', type=int, default=4096)
        parser.add_argument('--hhch_dropout', type=float, default=0.0)
        parser.add_argument('--hhch_clip_radius', type=float, default=2.3)
        parser.add_argument('--hhch_num_workers', type=int, default=4)
        return parser

    def init_experiment(self, config: dict, rename: bool = True) -> None:
        if config.get('hhch_variant', 'paper_cache3v') != 'paper_cache3v':
            raise ValueError('Only the audited paper_cache3v HHCH variant is implemented')
        dataset = str(config['dataset'])
        cluster_spec = str(config.get('hhch_clusters', 'auto'))
        if cluster_spec.lower() == 'auto':
            if dataset in _PAPER_CLUSTER_COUNTS:
                cluster_counts = _PAPER_CLUSTER_COUNTS[dataset]
                cluster_source = 'HHCH TIP 2024 paper setting'
            elif dataset == 'MSCOCO':
                # HHCH includes no COCO experiment or cluster setting in the
                # final paper.  Reusing its two multi-label settings is a
                # protocol adaptation, never represented as an original result.
                cluster_counts = (200, 150, 80)
                cluster_source = (
                    'GroundedDNA matched MSCOCO adaptation; HHCH paper does not '
                    'report an MSCOCO cluster setting')
            else:
                raise ValueError(
                    f'HHCH reports no cluster hierarchy for {dataset}; pass '
                    '--hhch_clusters explicitly and report it as an adaptation')
        else:
            cluster_counts = _parse_cluster_counts(cluster_spec)
            cluster_source = ('user-specified protocol adaptation'
                              if dataset not in _PAPER_CLUSTER_COUNTS
                              else 'user override of HHCH paper setting')

        config['hhch_cluster_counts'] = list(cluster_counts)
        config['hhch_cluster_source'] = cluster_source
        config['implementation_variant'] = 'paper_cache3v'
        config['paper_doi'] = '10.1109/TIP.2024.3371358'
        config['audited_release_commit'] = (
            '47eac18aedf3a2f8623002a702da28aec8aafd0b')
        config['upstream_license_audit'] = (
            'no LICENSE/COPYING detected; clean-room equation implementation')
        config['supervision_regime'] = 'visual-only self-supervised'
        config['information_tier'] = 'U0'
        config['instance_labels_used_for_training'] = False
        config['test_labels_used_for_selection'] = False
        config['protocol_adaptation'] = (
            'frozen shared visual backbone; fixed cached canonical+two augmented '
            'views replace frozen VGG19 plus online augmentation')
        config['paper_release_discrepancies'] = [
            'paper K-Means++ vs release random center initialization',
            'paper LogCosh quantization vs release MSE quantization',
            'paper tau=0.2 vs release parser default tau=0.3',
            'paper sums both HIC anchor directions vs release averages them',
            'paper head omits dropout in prose vs release Dropout(0.1)',
            ('paper does not specify an epoch LR schedule; release applies '
             'lr0*0.9**(epoch//20) from epoch 30; paper_cache3v adopts that '
             'released schedule as the auditable missing-detail choice'),
        ]
        config['hhch_learning_rate_schedule_source'] = (
            'official release: lr0 through epoch 29, then '
            'lr0*0.9**(epoch//20); paper does not specify the schedule')
        config['epoch_count_source'] = (
            'official release default; journal paper does not state total epochs')
        config['projection_clip_source'] = (
            'official release stability default; not stated in journal prose')
        super().init_experiment(config, rename=rename)

    def _build_model_from_config(self, d_in: int, config: dict) -> Module:
        clip_radius = config.get('hhch_clip_radius')
        if clip_radius is not None and float(clip_radius) <= 0:
            clip_radius = None
        return HHCHFeatureHash(
            d_in=d_in,
            bit=int(config['bit']),
            hyper_dim=int(config['hhch_hyper_dim']),
            curvature=float(config['hhch_curvature']),
            dropout=float(config['hhch_dropout']),
            clip_radius=None if clip_radius is None else float(clip_radius),
        )

    @staticmethod
    def _release_lr_multiplier(epoch: int) -> float:
        """Official HHCH epoch schedule used for the paper's missing detail."""
        if int(epoch) < 0:
            raise ValueError('HHCH scheduler epoch must be non-negative')
        return 1.0 if int(epoch) < 30 else 0.9 ** (int(epoch) // 20)

    def _init_scheduler(self, optimizer: Optimizer, name: str,
                        **opts) -> _LRScheduler:
        del opts
        if str(name) != 'hhch_release_piecewise':
            raise ValueError(
                'HHCH paper_cache3v requires the audited release LR schedule')
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=self._release_lr_multiplier)

    @staticmethod
    @torch.no_grad()
    def _canonical_embeddings(model: HHCHFeatureHash, dataset: Dataset, *,
                              device: str, batch_size: int,
                              num_workers: int) -> torch.Tensor:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=str(device).startswith('cuda'))
        model.eval()
        embeddings = None
        seen = torch.zeros(len(dataset), dtype=torch.bool, device=device)
        for batch in loader:
            index = batch['idx'].to(device=device, dtype=torch.long)
            projected = model.project_features(batch['img'].to(device))
            if embeddings is None:
                embeddings = torch.empty(
                    len(dataset), projected.shape[1], dtype=projected.dtype,
                    device=device)
            embeddings.index_copy_(0, index, projected)
            seen[index] = True
        if embeddings is None or not bool(seen.all()):
            raise RuntimeError('HHCH failed to compute every canonical embedding')
        return embeddings

    def _train_model(self, backbone_with_encoder: Module, optimizer: Optimizer,
                     scheduler: _LRScheduler, trainset: Dataset,
                     testset: Dataset, dbset: Dataset, model_dir: str,
                     result_dir: str, device: str, batch_size: int,
                     eval_period: int, config: dict) -> None:
        del testset, dbset, model_dir, result_dir, batch_size
        model = backbone_with_encoder.to(device)
        if not isinstance(model, HHCHFeatureHash):
            raise TypeError('HHCH requires HHCHFeatureHash')
        num_workers = int(config['hhch_num_workers'])
        train_loader = DataLoader(
            trainset, batch_size=int(config['batch_size']), shuffle=True,
            num_workers=num_workers, drop_last=False,
            pin_memory=str(device).startswith('cuda'))
        cluster_counts = tuple(int(value)
                               for value in config['hhch_cluster_counts'])

        for epoch in range(int(config['max_epoch'])):
            # Algorithm 1 constructs the complete hierarchy on canonical
            # embeddings before every epoch. Labels are neither read nor used.
            canonical = self._canonical_embeddings(
                model, trainset, device=device,
                batch_size=int(config['batch_size']),
                num_workers=num_workers)
            hierarchy = hhch_hierarchical_kmeans(
                canonical, cluster_counts,
                curvature=float(config['hhch_curvature']),
                max_iter=int(config['hhch_kmeans_max_iter']),
                tolerance=float(config['hhch_kmeans_tolerance']),
                seed=int(config['seed']) + epoch * len(cluster_counts),
                chunk_size=int(config['hhch_cluster_chunk_size']),
            )
            ancestors_all = hierarchy['assignments']
            centers_by_level = hierarchy['centers']
            if not isinstance(ancestors_all, torch.Tensor):
                raise TypeError('invalid HHCH hierarchy assignments')

            model.train()
            for batch in train_loader:
                first_codes = model.encode_hash(batch['img_tr1'].to(device))
                second_codes = model.encode_hash(batch['img_tr2'].to(device))
                first_hyperbolic = model.project_hash(first_codes)
                second_hyperbolic = model.project_hash(second_codes)
                index = batch['idx'].to(device=device, dtype=torch.long)
                ancestors = ancestors_all.index_select(0, index)
                loss, parts = hhch_loss(
                    first_codes, second_codes,
                    first_hyperbolic, second_hyperbolic,
                    ancestors, centers_by_level,
                    temperature=float(config['hhch_temperature']),
                    curvature=float(config['hhch_curvature']),
                    quantization_weight=float(
                        config['hhch_quantization_weight']),
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                self._insert_iter_loss_sum(
                    {'loss': loss.item(),
                     **{name: value.item() for name, value in parts.items()}},
                    current_batch_size=int(first_codes.shape[0]),
                )

            scheduler.step()
            self._compute_loss_per_epoch()
            model_id: Any = f'{epoch:03d}'
            self._save_train_model_log(model_id, 'epoch', True)
            # Validation may be monitored for epoch selection in stage 1.
            # Stage 2 has an already fixed epoch and evaluates only its terminal
            # checkpoint here, preventing test-led selection by construction.
            is_validation_stage = self.valset is not None
            should_eval = (epoch + 1 == int(config['max_epoch'])
                           or (is_validation_stage
                               and (epoch + 1) % int(eval_period) == 0))
            if should_eval:
                # DeepHashBase uses held-out validation during stage 1 and the
                # test split only after a fixed epoch is supplied for stage 2.
                self.start_eval_process(model_id, 'epoch')
                self._save_train_model_params(
                    model_id, 'epoch',
                    etc_info={'hhch_cluster_counts': list(cluster_counts)})
