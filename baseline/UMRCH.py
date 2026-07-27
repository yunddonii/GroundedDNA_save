"""UMRCH (Expert Systems with Applications 2026) matched-cache adapter.

UMRCH is label-free at the instance level, but its released method consumes
the benchmark's class-name taxonomy through CLIP.  It is therefore reported
as a taxonomy/VLM-assisted baseline, not as strict visual-only unsupervised
hashing.  The hash objective below is a clean-room transcription of the
released implementation; only image encoding is replaced by the repository's
fixed two-view CLIP cache.
"""

from __future__ import annotations

from argparse import ArgumentParser
import hashlib

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset

from .base_model import DeepHashBase
from .asset_provenance import (
    OFFICIAL_UMRCH_TAXONOMY_SHA256,
    load_and_verify_text_asset_manifest,
    sha256_file,
    verify_manifest_cache,
)
from .modern_unsupervised import (
    UMRCHFeatureHash,
    umrch_loss,
    umrch_semantic_similarity,
)


def _sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


class UMRCH(DeepHashBase):
    def __init__(self) -> None:
        super().__init__()
        self._concepts: torch.Tensor | None = None
        self._ln_weight: torch.Tensor | None = None
        self._ln_bias: torch.Tensor | None = None
        self._local_projection: torch.Tensor | None = None

    def _get_default_config_dict(self) -> dict:
        return {
            'batch_size': 32,
            'max_epoch': 100,
            'learning_rate': 0.005,
            'optimizer_name': 'sgd',
            'sgd_momentum': 0.9,
            'sgd_weight_decay': 5e-4,
            'lr_scheduler': 'umrch_epoch_onecycle',
            'umrch_vl_temperature': 0.01,
            'umrch_concept_threshold': 0.3,
            'umrch_negative_threshold': 0.0,
            'umrch_contrastive_temperature': 1.0,
            'umrch_distribution_weight': 1.0,
            'umrch_contrastive_weight': 150.0,
        }

    def _get_config_dict_for_dataset(self, default_config: dict, dataset: str) -> dict:
        return default_config

    def _get_fixed_config_dict(self) -> dict:
        return {
            'gen_code_method': 'sign',
            'finetune': False,
            'dataset_return_index': False,
            'dataset_return_paired_aug_img': True,
            'dataset_return_visual_tokens': True,
            'transform': 'fixed-cached-two-view-local',
            'encoder_layers': 'umrch_l2norm_linear_to_bit',
            'batch_norm': False,
        }

    def _add_model_specific_args_into_parser(self, parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument(
            '--umrch_concept_embeddings', required=True,
            help='[C,D] CLIP embeddings of "a photo of the {class}" (.npy).',
        )
        parser.add_argument(
            '--umrch_vision_adapter', required=True,
            help='NPZ with ln_weight, ln_bias, projection for cached patch tokens.',
        )
        parser.add_argument(
            '--umrch_asset_manifest', required=True,
            help='Manifest emitted with concepts and CLIP vision adapter.',
        )
        parser.add_argument('--umrch_vl_temperature', type=float, default=0.01)
        parser.add_argument('--umrch_concept_threshold', type=float, default=0.3)
        parser.add_argument('--umrch_negative_threshold', type=float, default=0.0)
        parser.add_argument('--umrch_contrastive_temperature', type=float, default=1.0)
        parser.add_argument('--umrch_distribution_weight', type=float, default=1.0)
        parser.add_argument('--umrch_contrastive_weight', type=float, default=150.0)
        return parser

    def _build_model_from_config(self, d_in: int, config: dict) -> Module:
        return UMRCHFeatureHash(d_in=d_in, bit=int(config['bit']))

    def _init_scheduler(self, optimizer: Optimizer, name: str,
                        **opts) -> _LRScheduler:
        if str(name) != 'umrch_epoch_onecycle':
            raise ValueError('UMRCH requires its released epoch-wise OneCycleLR')
        # The release creates a 100-step OneCycleLR and (unusually) advances it
        # once per epoch, rather than once per minibatch.
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=0.1, total_steps=100, pct_start=0.3,
            anneal_strategy='cos', final_div_factor=100,
        )

    def init_experiment(self, config: dict, rename: bool = True) -> None:
        if str(config['dataset']) not in ('Flickr25k', 'MSCOCO', 'NUSWIDE'):
            raise ValueError(
                'The released UMRCH protocol defines benchmark taxonomies only '
                'for Flickr25k, MSCOCO, and NUSWIDE. Do not invent a CIFAR/CUB '
                'variant and call it an exact baseline.'
            )
        concepts = np.asarray(
            np.load(config['umrch_concept_embeddings']), dtype=np.float32)
        manifest = load_and_verify_text_asset_manifest(
            config['umrch_asset_manifest'],
            expected_mode='umrch',
            expected_embeddings=config['umrch_concept_embeddings'],
            expected_prompt='a photo of the {class}',
            expected_extra_outputs={
                'vision_adapter': config['umrch_vision_adapter']},
        )
        adapter = np.load(config['umrch_vision_adapter'])
        required = {'ln_weight', 'ln_bias', 'projection'}
        if not required.issubset(adapter.files):
            raise ValueError(
                f'UMRCH adapter lacks {sorted(required - set(adapter.files))}')
        ln_weight = np.asarray(adapter['ln_weight'], dtype=np.float32)
        ln_bias = np.asarray(adapter['ln_bias'], dtype=np.float32)
        projection = np.asarray(adapter['projection'], dtype=np.float32)
        if concepts.ndim != 2 or projection.ndim != 2:
            raise ValueError('UMRCH concepts/projection must be rank-2 arrays')
        if ln_weight.shape != ln_bias.shape or projection.shape[0] != ln_weight.size:
            raise ValueError('UMRCH layernorm/projection dimensions are inconsistent')
        if projection.shape[1] != concepts.shape[1]:
            raise ValueError('UMRCH projected patch and concept dimensions differ')
        expected_concepts = {'Flickr25k': 24, 'MSCOCO': 80, 'NUSWIDE': 21}
        expected_count = expected_concepts[str(config['dataset'])]
        if concepts.shape[0] != expected_count:
            raise ValueError(
                f'UMRCH {config["dataset"]} requires the ordered {expected_count}-'
                f'class taxonomy, got {concepts.shape[0]} concepts')
        expected_terms_hash = OFFICIAL_UMRCH_TAXONOMY_SHA256[
            str(config['dataset'])]
        if manifest.get('terms_file_sha256') != expected_terms_hash:
            raise ValueError(
                f'UMRCH {config["dataset"]} requires the exact ordered class-name '
                'file from audited release commit '
                '5e60e8d3d35e435506f86dd135e90763c07b5592')
        if manifest.get('information_tier') != 'U2':
            raise ValueError(
                'UMRCH must be recorded as U2 because it consumes the exact '
                'target benchmark taxonomy')

        config['umrch_concept_sha256'] = _sha256_array(concepts)
        config['umrch_adapter_sha256'] = hashlib.sha256(
            b''.join(np.ascontiguousarray(x).view(np.uint8).tobytes()
                     for x in (ln_weight, ln_bias, projection))).hexdigest()
        config['umrch_n_concepts'] = int(concepts.shape[0])
        config['umrch_official_taxonomy_sha256'] = expected_terms_hash
        config['umrch_taxonomy_source_commit'] = (
            '5e60e8d3d35e435506f86dd135e90763c07b5592')
        config['umrch_asset_manifest_sha256'] = sha256_file(
            config['umrch_asset_manifest'])
        config['information_tier'] = 'U2'
        config['umrch_term_source_condition'] = manifest[
            'term_source_condition']
        config['umrch_matched_official_taxonomy_dataset'] = manifest[
            'matched_official_taxonomy_dataset']
        config['implementation_variant'] = 'released-objective-clip-cache-adaptation'
        config['information_condition'] = (
            'no instance labels; benchmark class-name taxonomy encoded by frozen CLIP')
        config['hash_head_input_normalization'] = (
            'L2 normalization matching released CLIP get_feat before linear head')
        super().init_experiment(config, rename=rename)
        verify_manifest_cache(manifest, self.trainset.cache_dir)

        global_dim = int(self.trainset.visual_global.shape[1])
        token_dim = int(self.trainset.visual_tokens.shape[-1])
        if global_dim != int(concepts.shape[1]):
            raise ValueError(
                f'cache global dim {global_dim} != concept dim {concepts.shape[1]}')
        if token_dim != int(projection.shape[0]):
            raise ValueError(
                f'cache token dim {token_dim} != adapter dim {projection.shape[0]}')
        self._concepts = torch.from_numpy(concepts).to(self.device)
        self._ln_weight = torch.from_numpy(ln_weight).to(self.device)
        self._ln_bias = torch.from_numpy(ln_bias).to(self.device)
        self._local_projection = torch.from_numpy(projection).to(self.device)

    def _project_local(self, tokens: torch.Tensor) -> torch.Tensor:
        if self._ln_weight is None or self._local_projection is None:
            raise RuntimeError('UMRCH vision adapter was not initialized')
        tokens = F.layer_norm(
            tokens.float(), (tokens.shape[-1],),
            self._ln_weight, self._ln_bias,
        )
        return tokens @ self._local_projection

    def _train_model(self, backbone_with_encoder: Module, optimizer: Optimizer,
                     scheduler: _LRScheduler, trainset: Dataset,
                     testset: Dataset, dbset: Dataset, model_dir: str,
                     result_dir: str, device: str, batch_size: int,
                     eval_period: int, config: dict) -> None:
        if self._concepts is None:
            raise RuntimeError('UMRCH concepts were not initialized')
        loader = DataLoader(
            trainset, batch_size=int(config['batch_size']), shuffle=True,
            num_workers=int(config.get('num_workers', 4)), drop_last=True,
        )
        model = backbone_with_encoder.to(device)
        max_epoch = int(config['max_epoch'])
        for epoch in range(max_epoch):
            model.train()
            for batch in loader:
                global_first = batch['img_tr1'].to(device)
                global_second = batch['img_tr2'].to(device)
                local_first = self._project_local(
                    batch['visual_tokens_tr1'].to(device))
                local_second = self._project_local(
                    batch['visual_tokens_tr2'].to(device))
                output_first = model(global_first)['continuous_code']
                output_second = model(global_second)['continuous_code']
                global_both = torch.cat((global_first, global_second), dim=0)
                local_both = torch.cat((local_first, local_second), dim=0)
                code_both = torch.cat((output_first, output_second), dim=0)
                with torch.no_grad():
                    semantic = umrch_semantic_similarity(
                        global_both, local_both, self._concepts,
                        vl_temperature=float(config['umrch_vl_temperature']),
                        concept_threshold=float(config['umrch_concept_threshold']),
                        negative_threshold=float(config['umrch_negative_threshold']),
                    )
                loss, parts = umrch_loss(
                    code_both, semantic,
                    contrastive_temperature=float(
                        config['umrch_contrastive_temperature']),
                    distribution_weight=float(config['umrch_distribution_weight']),
                    contrastive_weight=float(config['umrch_contrastive_weight']),
                )
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
            is_validation_stage = self.valset is not None
            should_eval = (epoch + 1 == max_epoch
                           or (is_validation_stage
                               and (epoch + 1) % int(eval_period) == 0))
            if should_eval:
                self.start_eval_process(model_id, 'epoch')
                self._save_train_model_params(model_id, 'epoch')
