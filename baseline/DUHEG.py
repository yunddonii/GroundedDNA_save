"""Deep Unsupervised Hashing via External Guidance (ICML 2025).

Clean-room implementation of the released feature-level objective.  Training
labels are never read.  The method is *not* visual-only: its released objective
requires CLIP-encoded external terms supplied with
``--duheg_noun_embeddings``.  Because the official selected WordNet artifact
is unavailable, this module is an explicitly unverified noun-bank adapter, not
an exact paper reproduction.  The same frozen CLIP space as the selected
visual cache must be used.
"""

from __future__ import annotations

from argparse import ArgumentParser
from math import cos, pi

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset

from .base_model import DeepHashBase
from .asset_provenance import (
    DUHEG_PROMPT_TEMPLATES,
    load_and_verify_text_asset_manifest,
    sha256_file,
    verify_manifest_cache,
)
from .modern_unsupervised import (
    DUHEGFeatureHash,
    duheg_external_guidance,
    duheg_multi_positive_nce,
)


class DUHEG(DeepHashBase):
    def __init__(self) -> None:
        super().__init__()
        self._guidance: torch.Tensor | None = None

    def _get_default_config_dict(self) -> dict:
        return {
            'batch_size': 512,
            'max_epoch': 60,
            'learning_rate': 1e-4,
            'optimizer_name': 'adam',
            'adam_weight_decay': 0.0,
            'lr_scheduler': 'none',
            'duheg_temperature': -1.0,
            'duheg_positive_threshold': 0.97,
            'duheg_guidance_temperature': 0.004,
            'duheg_guidance_precision': 'release_fp16',
            'duheg_warmup_epochs': 10,
            'duheg_min_lr': 1e-5,
        }

    def _get_config_dict_for_dataset(self, default_config: dict, dataset: str) -> dict:
        return default_config

    def _get_fixed_config_dict(self) -> dict:
        return {
            'gen_code_method': 'sign',
            'finetune': False,
            'dataset_return_index': True,
            'dataset_return_paired_aug_img': True,
            'dataset_return_visual_tokens': False,
            'transform': 'fixed-cached-two-view',
            'encoder_layers': 'duheg_d_to_512_to_bit',
            'batch_norm': True,
        }

    def _add_model_specific_args_into_parser(self, parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument(
            '--duheg_noun_embeddings', required=True,
            help='[N, D] selected WordNet noun CLIP embeddings (.npy).',
        )
        parser.add_argument(
            '--duheg_asset_manifest', required=True,
            help='Manifest emitted with the selected noun embedding asset.',
        )
        parser.add_argument(
            '--duheg_allow_unverified_selection', action='store_true',
            help=('Explicitly run the released-objective adapter even though '
                  'the public noun-selection artifact is unavailable. Such a '
                  'run is not an exact paper baseline.'),
        )
        parser.add_argument(
            '--duheg_temperature', type=float, default=-1.0,
            help='-1 selects the released dataset default (.2 COCO, .8 otherwise).',
        )
        parser.add_argument('--duheg_positive_threshold', type=float, default=0.97)
        parser.add_argument('--duheg_guidance_temperature', type=float, default=0.004)
        parser.add_argument(
            '--duheg_guidance_precision',
            choices=('release_fp16', 'stable_fp32'), default='release_fp16',
            help=('release_fp16 reproduces the public guidance aggregation; '
                  'stable_fp32 is a separately reported numerical ablation.'),
        )
        parser.add_argument('--duheg_warmup_epochs', type=int, default=10)
        parser.add_argument('--duheg_min_lr', type=float, default=1e-5)
        return parser

    def _build_model_from_config(self, d_in: int, config: dict) -> Module:
        text_dim = config.get('duheg_text_dim')
        if text_dim is None:
            noun_path = str(config['duheg_noun_embeddings'])
            nouns = np.load(noun_path, mmap_mode='r')
            if nouns.ndim != 2:
                raise ValueError(f'DUH-EG noun bank must be [N,D], got {nouns.shape}')
            text_dim = int(nouns.shape[1])
        if int(text_dim) != int(d_in):
            raise ValueError(
                f'DUH-EG image/noun dimensions differ ({d_in} vs {text_dim}). '
                'Encode nouns with the exact CLIP checkpoint recorded by cache/meta.json.'
            )
        return DUHEGFeatureHash(
            image_dim=d_in, text_dim=int(text_dim),
            bit=int(config['bit']), hidden_dim=512,
        )

    def init_experiment(self, config: dict, rename: bool = True) -> None:
        if float(config.get('duheg_temperature', -1.0)) < 0:
            config['duheg_temperature'] = (
                0.2 if str(config['dataset']) == 'MSCOCO' else 0.8)
        noun_array = np.asarray(
            np.load(config['duheg_noun_embeddings']), dtype=np.float32)
        manifest = load_and_verify_text_asset_manifest(
            config['duheg_asset_manifest'],
            expected_mode='duheg-selected',
            expected_embeddings=config['duheg_noun_embeddings'],
            expected_prompt=list(DUHEG_PROMPT_TEMPLATES),
        )
        # There is currently no approved ordered noun-bank hash to compare
        # against: the paper and public selection program disagree, and the
        # release contains no selected artifact.  A manifest-authored boolean
        # is therefore not accepted as proof of an exact reproduction.
        selection_verified = False
        if not bool(config.get('duheg_allow_unverified_selection', False)):
            raise ValueError(
                'DUH-EG public code does not provide an unambiguous selected-noun '
                'artifact. Pass --duheg_allow_unverified_selection only for an '
                'explicit released-objective adapter run, not an exact baseline.')
        import hashlib
        config['duheg_noun_bank_sha256'] = hashlib.sha256(
            np.ascontiguousarray(noun_array).view(np.uint8)).hexdigest()
        config['duheg_n_nouns'] = int(noun_array.shape[0])
        config['duheg_text_dim'] = int(noun_array.shape[1])
        config['duheg_asset_manifest_sha256'] = sha256_file(
            config['duheg_asset_manifest'])
        config['duheg_selection_verified'] = selection_verified
        config['information_tier'] = manifest['information_tier']
        config['duheg_term_source_condition'] = manifest[
            'term_source_condition']
        config['duheg_matched_official_taxonomy_dataset'] = manifest[
            'matched_official_taxonomy_dataset']
        config['publishable_as_exact_duheg'] = False
        config['implementation_variant'] = 'released-objective-fixed-cache-2view'
        config['information_condition'] = (
            f'{manifest["information_tier"]}: '
            f'{manifest["term_source_condition"]}; no instance labels; '
            'WordNet selection is not verified')
        config['hash_head_input_normalization'] = (
            'L2 normalization of canonical/augmented/query/database CLIP features')
        config['duheg_release_guidance_precision_matched'] = (
            config.get('duheg_guidance_precision') == 'release_fp16')
        super().init_experiment(config, rename=rename)
        verify_manifest_cache(manifest, self.trainset.cache_dir)

        nouns = torch.from_numpy(noun_array).to(self.device)
        image_array = np.asarray(
            self.trainset.visual_global[self.trainset.rows], dtype=np.float32)
        images = torch.from_numpy(image_array).to(self.device)
        self._guidance = duheg_external_guidance(
            images, nouns,
            temperature=float(config['duheg_guidance_temperature']),
            compute_dtype=(
                torch.float16
                if config.get('duheg_guidance_precision') == 'release_fp16'
                else torch.float32),
        ).cpu()

    @staticmethod
    def _warmup_cosine_lr(epoch: int, max_epoch: int, warmup_epoch: int,
                          lr_min: float, lr_max: float) -> float:
        if epoch < warmup_epoch:
            return lr_min + (lr_max - lr_min) * epoch / max(warmup_epoch, 1)
        span = max(max_epoch - warmup_epoch, 1)
        return lr_min + (lr_max - lr_min) * (
            1.0 + cos(pi * (epoch - warmup_epoch) / span)) / 2.0

    def _train_model(self, backbone_with_encoder: Module, optimizer: Optimizer,
                     scheduler: _LRScheduler, trainset: Dataset,
                     testset: Dataset, dbset: Dataset, model_dir: str,
                     result_dir: str, device: str, batch_size: int,
                     eval_period: int, config: dict) -> None:
        if self._guidance is None:
            raise RuntimeError('DUH-EG guidance was not initialized')
        loader = DataLoader(
            trainset, batch_size=int(config['batch_size']), shuffle=True,
            num_workers=int(config.get('num_workers', 4)), drop_last=True,
        )
        model = backbone_with_encoder.to(device)
        max_epoch = int(config['max_epoch'])
        schedule_horizon = int(config.get('schedule_horizon') or max_epoch)
        if schedule_horizon < max_epoch:
            raise ValueError('DUH-EG schedule_horizon cannot be shorter than max_epoch')
        for epoch in range(max_epoch):
            lr = self._warmup_cosine_lr(
                epoch, schedule_horizon, int(config['duheg_warmup_epochs']),
                float(config['duheg_min_lr']), float(config['learning_rate']))
            for group in optimizer.param_groups:
                group['lr'] = lr
            model.train()
            for batch in loader:
                index = batch['idx'].long()
                guidance = self._guidance[index].to(device)
                image_first = model(batch['img_tr1'].to(device))['continuous_code']
                image_second = model(batch['img_tr2'].to(device))['continuous_code']
                text_code = model.encode_guidance(guidance)
                semantic = F.normalize(guidance, dim=1) @ F.normalize(guidance, dim=1).T
                positives = (semantic > float(config['duheg_positive_threshold'])).float()
                temperature = float(config['duheg_temperature'])
                image_text = duheg_multi_positive_nce(
                    image_first, text_code, positives, temperature)
                aug_text = duheg_multi_positive_nce(
                    image_second, text_code, positives, temperature)
                aug_image = duheg_multi_positive_nce(
                    image_first, image_second, positives, temperature)
                loss = image_text + aug_text + aug_image
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                self._insert_iter_loss_sum(
                    {'loss': loss.item(), 'image_text': image_text.item(),
                     'aug_text': aug_text.item(), 'aug_image': aug_image.item(),
                     'lr': lr},
                    current_batch_size=batch['img'].shape[0],
                )
            scheduler.step()
            self._compute_loss_per_epoch()
            model_id = f'{epoch:03d}'
            self._save_train_model_log(model_id, 'epoch', True)
            is_validation_stage = self.valset is not None
            should_eval = (epoch + 1 == max_epoch
                           or (is_validation_stage
                               and (epoch + 1) % int(eval_period) == 0))
            if should_eval:
                self.start_eval_process(model_id, 'epoch')
                self._save_train_model_params(model_id, 'epoch')
