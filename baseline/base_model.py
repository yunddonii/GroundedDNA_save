"""Self-contained binary deep-hashing baseline runner.

Adapts the original abstract `DeepHashBase` framework (which expected a sister
`lib/` package -- not present in this repo) to the GroundedDNA project layout:

    - The vision backbone is FROZEN. We do not run it
      online; instead we read its already-extracted `visual_global` embeddings
      from the explicitly selected feature cache. The cache provenance is
      persisted in each run config so matched-backbone comparisons can be
      audited.
    - The trainable hash model is method-specific. Legacy methods use the
      Linear/GELU encoder below; modern methods can override
      `DeepHashBase._build_model_from_config` while preserving the common
      `output['continuous_code']` convention.
    - Hash extraction is `sign(continuous_code)` (matches `gen_code_method=sign`).
    - Evaluation reports both binary Hamming and the paper comparison metric:
      adjacent signed-bit pairs are packed into 18 DNA bases and ranked by raw
      base-Hamming distance.

Each concrete baseline (DPSH, HashNet, CSQ, OrthoHash, ...) overrides:
    _get_default_config_dict, _get_config_dict_for_dataset, _get_fixed_config_dict,
    _add_model_specific_args_into_parser, _train_model.
The runner main() at the bottom dispatches to the chosen class via `--method`.

Fair-comparison sketch (DNA vs binary):
    Our DNA hash : 18 positions x 4 bases -> 36 bits effective storage.
                   Distance: raw Hamming over 18 base symbols.
    Binary base  : 36 signed bits -> consecutive bit pairs -> 18 base symbols.
                   Distance: the same raw base-Hamming metric.
    Backbone/cache, train/val/test splits, bit budget, and selection metric are
    recorded; method-required auxiliary inputs are disclosed separately.
"""

from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset, DataLoader

from .cache_provenance import (
    consumed_cache_artifact_names,
    hash_cache_artifacts,
    verify_cache_artifact_hashes,
)
from tqdm import tqdm


# ============================================================ project-wide constants

NUM_CLASS: Dict[str, Dict[str, int]] = {
    'CIFAR10':   {'setting1': 10},
    'Flickr25k': {'setting1': 24},
    'MSCOCO':    {'setting1': 80},
    'NUSWIDE':   {'setting1': 21},
    'CUB_200':   {'setting1': 200},
}
MULTI_LABEL: Dict[str, bool] = {
    'CIFAR10':   False,
    'Flickr25k': True,
    'MSCOCO':    True,
    'NUSWIDE':   True,
    'CUB_200':   False,
}
DEFAULT_CACHE_DIR: Dict[str, str] = {
    'CIFAR10':   './cache/cifar10_siglip2',
    'Flickr25k': './cache/flickr25k_siglip2',
    'MSCOCO':    './cache/mscoco_siglip2',
    'NUSWIDE':   './cache/nuswide_siglip2',
    'CUB_200':   './cache/cub200_clip',
}


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint_data_context(checkpoint: dict, *, dataset: str,
                                   setting: str, cache_dir: str,
                                   expected_bit: int | None = None) -> dict:
    """Fail before extraction if checkpoint/cache/split identities disagree."""
    config = checkpoint.get('config')
    if not isinstance(config, dict):
        raise ValueError('checkpoint lacks config metadata')
    if config.get('dataset') != dataset or config.get('setting') != setting:
        raise ValueError(
            f'checkpoint dataset context is '
            f'{config.get("dataset")}/{config.get("setting")}, requested '
            f'{dataset}/{setting}')
    if expected_bit is not None and int(config.get('bit', -1)) != int(expected_bit):
        raise ValueError(
            f'checkpoint bit={config.get("bit")} != required {expected_bit}')
    meta_path = os.path.join(cache_dir, 'meta.json')
    ids_path = os.path.join(cache_dir, 'image_ids.json')
    with open(meta_path, encoding='utf-8') as handle:
        meta = json.load(handle)
    saved_backbone = config.get('resolved_backbone')
    if saved_backbone is not None and saved_backbone != meta.get('backbone'):
        raise ValueError('checkpoint/cache backbone mismatch')
    saved_dim = config.get('resolved_projection_dim')
    if saved_dim is not None and int(saved_dim) != int(meta.get('D_proj', -1)):
        raise ValueError('checkpoint/cache projection dimension mismatch')
    saved_meta_hash = config.get('resolved_cache_meta_sha256')
    if saved_meta_hash is not None:
        if saved_meta_hash != _sha256_file(meta_path):
            raise ValueError('checkpoint/cache meta SHA-256 mismatch')
    else:
        saved_cache = config.get('resolved_cache_dir') or config.get('cache_dir')
        if saved_cache is not None and not os.path.isabs(str(saved_cache)):
            repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            saved_cache = os.path.join(repo, str(saved_cache))
        if saved_cache is not None and os.path.realpath(str(saved_cache)) \
                != os.path.realpath(cache_dir):
            raise ValueError(
                'legacy checkpoint/cache path mismatch and no cache hash is available')
    saved_ids_hash = config.get('resolved_cache_image_ids_sha256')
    if saved_ids_hash is not None and saved_ids_hash != _sha256_file(ids_path):
        raise ValueError('checkpoint/cache image order SHA-256 mismatch')
    saved_artifact_hashes = config.get('resolved_cache_artifact_sha256')
    if saved_artifact_hashes is not None:
        expected_artifact_names = set(consumed_cache_artifact_names(
            paired_aug=bool(
                config.get('dataset_return_paired_aug_img', False)),
            visual_tokens=bool(
                config.get('dataset_return_visual_tokens', False)),
        ))
        if set(saved_artifact_hashes) != expected_artifact_names:
            raise ValueError(
                'checkpoint consumed cache-artifact set does not match its '
                f'loader flags: {set(saved_artifact_hashes)} != '
                f'{expected_artifact_names}')
        verify_cache_artifact_hashes(cache_dir, saved_artifact_hashes)
    return meta


# ============================================================ inline utilities (replaces lib/*)

def fix_random_seed(seed: int = 1) -> None:
    """Replaces lib.utils.fix_random_seed."""
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Logger:
    """Replaces lib.utils.Logger.

    Per-iter loss accumulation -> per-epoch mean -> CSV row + console print.
    Eval metrics are merged in via `save_eval_model_log`.
    """
    def __init__(self, result_dir: str, config: dict):
        self.result_dir = result_dir
        os.makedirs(result_dir, exist_ok=True)
        self.config = config
        self.csv_path = os.path.join(result_dir, 'log.csv')
        self._iter_buf: Dict[str, float] = {}
        self._iter_n:   int = 0
        self._latest_means: Dict[str, float] = {}
        self._fields:   List[str] = ['model_id']
        self._wrote_header = False

        # snapshot config (only json-serializable scalars)
        cfg_snap = {k: v for k, v in config.items()
                    if isinstance(v, (str, int, float, bool, list, type(None)))}
        with open(os.path.join(result_dir, 'config.json'), 'w') as f:
            json.dump(cfg_snap, f, indent=2)

    # -- iter / epoch loss accumulators (called from each baseline's _train_model)
    def insert_iter_loss_sum(self, loss_sum_info: dict, current_batch_size: int) -> None:
        for k, v in loss_sum_info.items():
            self._iter_buf[k] = self._iter_buf.get(k, 0.0) + float(v) * current_batch_size
        self._iter_n += int(current_batch_size)

    def compute_loss_per_epoch(self) -> None:
        if self._iter_n == 0:
            self._latest_means = {}
            return
        self._latest_means = {k: v / self._iter_n for k, v in self._iter_buf.items()}
        self._iter_buf = {}; self._iter_n = 0

    # -- per-epoch row writer (flushes accumulated train losses)
    def save_train_model_log(self, model_id: Any, id_type: str,
                             enable_print: bool = False, print_message: str = "") -> None:
        row = {'model_id': str(model_id), **{f'train_{k}': v for k, v in self._latest_means.items()}}
        self._write_row(row)
        if enable_print:
            msg = ' '.join(f'{k}={v:.4f}' if isinstance(v, float) else f'{k}={v}'
                           for k, v in row.items())
            print(f'[{self.config.get("trial_name","baseline")}] {print_message} {msg}')

    def save_eval_model_log(self, model_id: Any, id_type: str, **eval_metrics) -> None:
        row = {'model_id': str(model_id),
               **{f'eval_{k}': v for k, v in eval_metrics.items()
                  if isinstance(v, (int, float))}}
        self._write_row(row)
        msg = ' '.join(f'{k}={v:.4f}' if isinstance(v, float) else f'{k}={v}'
                       for k, v in row.items())
        print(f'[{self.config.get("trial_name","baseline")}] eval -> {msg}')

    def _write_row(self, row: dict) -> None:
        for k in row:
            if k not in self._fields:
                self._fields.append(k)
        if not self._wrote_header or not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w') as f:
                f.write(','.join(self._fields) + '\n')
            self._wrote_header = True
        else:
            # rewrite if schema grew (rare; cheap for our sizes)
            existing = []
            with open(self.csv_path, 'r') as f:
                rd = csv.reader(f); existing = list(rd)
            if existing and existing[0] != self._fields:
                with open(self.csv_path, 'w') as f:
                    wr = csv.writer(f); wr.writerow(self._fields)
                    for old in existing[1:]:
                        d = dict(zip(existing[0], old))
                        wr.writerow([d.get(k, '') for k in self._fields])
        with open(self.csv_path, 'a') as f:
            f.write(','.join(str(row.get(k, '')) for k in self._fields) + '\n')


def load_transform_by_transform_name(name: str) -> Optional[Any]:
    """Replaces lib.transforms.load_transform_by_transform_name.

    Backbone is frozen + cached -- there is no online image augmentation. We
    return None and rely on cached features. Kept for interface compatibility.
    """
    return None


class _NoOpScheduler(_LRScheduler):
    def get_lr(self): return [g['lr'] for g in self.optimizer.param_groups]


# ============================================================ frozen cached-feature backbone

class _CachedFeatureBackbone(nn.Module):
    """Identity-like wrapper that just returns the cached visual_global tensor.

    The dataset already returns features in `batch['img']` (shape [B, D_proj]),
    so this module's forward is essentially a passthrough -- we keep it as a
    Module so each baseline's `model = backbone_with_encoder` pattern works.
    """
    def __init__(self, d_proj: int):
        super().__init__()
        self.d_proj = d_proj
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Already [B, D_proj]; just ensure float32 on the right device.
        return x.float()


class _LinearEncoder(nn.Module):
    """Trainable head: cached global feature [B, D_proj] -> code [B, bit]."""
    def __init__(self, d_in: int, bit: int, hidden_nodes: Optional[List[int]] = None,
                 batch_norm: bool = False):
        super().__init__()
        layers: List[nn.Module] = []
        last = d_in
        if hidden_nodes:
            for h in hidden_nodes:
                layers += [nn.Linear(last, h), nn.GELU(), nn.Dropout(0.1)]
                last = h
        layers.append(nn.Linear(last, bit))
        if batch_norm:
            layers.append(nn.BatchNorm1d(bit))
        self.net = nn.Sequential(*layers)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BackboneWithEncoder(nn.Module):
    """Convention-compatible model: forward(img) -> {'continuous_code': [B, bit], ...}.

    Mirrors what the original `lib.layers.backbone_dict[backbone]` instances
    returned, so existing baseline _train_model code (`output['continuous_code']`)
    runs unchanged.
    """
    def __init__(self, d_in: int, bit: int, hidden_nodes: Optional[List[int]] = None,
                 batch_norm: bool = False, finetune: bool = False):
        super().__init__()
        # Backbone is frozen by definition (cached features). `finetune=True`
        # is rejected -- we don't have raw pixels here.
        if finetune:
            print('[base_model] WARNING: finetune=True ignored; cached frozen '
                  'features cannot be back-propagated into.')
        self.backbone = _CachedFeatureBackbone(d_in)
        self.encoder_layers = _LinearEncoder(d_in, bit, hidden_nodes, batch_norm)
        self.bit = bit

    def forward(self, img: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.backbone(img)
        cont = self.encoder_layers(feat)
        # `backbone_last_output` is an alias used by some baselines (MLS3RDUH's
        # HashNet wrapper). Same tensor as `cnn_feat`; both keys provided for
        # cross-baseline compatibility.
        return {'continuous_code': cont, 'cnn_feat': feat, 'backbone_last_output': feat}


# Mimic the original `backbone_dict[name]` factory signature.
backbone_dict = {
    name: lambda encode_length, hidden_nodes, finetune, batch_norm, encoder_type='linear':
        BackboneWithEncoder(d_in=768, bit=int(encode_length),
                            hidden_nodes=hidden_nodes, batch_norm=batch_norm,
                            finetune=False)
    for name in ('SigLIP2', 'siglip2', 'SigLIP2-cached')
}


# ============================================================ cached-features dataset

class CachedFeatureDataset(Dataset):
    """Returns cached frozen-backbone features and evaluation labels.

    Output dict (keys match what each baseline `_train_model` reads):
        'img'   : float32 [D_proj]       (the cached visual_global)
        'label' : int64  [n_class]       (multi-hot; one-hot for single-class)
        'image_path' : str              (cache image_id, for reference)

    Optional augmented-view paths (set via `paired_aug=True`) populate two
    extra keys with cached auxiliary `visual_global_aug{0,1}.f16.npy` views:
        'img_tr1' : float32 [D_proj]
        'img_tr2' : float32 [D_proj]
    These are required by CIBHash / CIMON's NtXent training. Built by
    `extract_siglip2_features.py --save_aug_views 2`.

    `return_index=True` adds:
        'idx'   : int       row index within this split (0-based)
    Required by MLS3RDUH for its kNN-graph index lookup.
    """
    def __init__(self, dataset_name: str, setting: str, mode: str,
                 dataset_dir: str, cache_dir: Optional[str] = None,
                 paired_aug: bool = False, return_index: bool = False,
                 return_visual_tokens: bool = False):
        self.dataset_name = dataset_name
        self.setting = setting
        self.mode = mode
        if cache_dir is None:
            cache_dir = DEFAULT_CACHE_DIR.get(dataset_name)
            if cache_dir is None:
                raise KeyError(f'no default cache_dir for {dataset_name}')
        self.cache_dir = cache_dir
        self.paired_aug = bool(paired_aug)
        self.return_index = bool(return_index)
        self.return_visual_tokens = bool(return_visual_tokens)
        # cache memmaps
        ids = json.load(open(os.path.join(cache_dir, 'image_ids.json')))
        if not isinstance(ids, list) or not all(isinstance(iid, str) for iid in ids):
            raise ValueError('cache image_ids.json must contain a list of strings')
        if len(ids) != len(set(ids)):
            raise ValueError('cache image_ids.json contains duplicate identifiers')
        self.image_ids = list(ids)
        self.id_to_row = {iid: i for i, iid in enumerate(ids)}
        self.visual_global = np.load(
            os.path.join(cache_dir, 'visual_global.f16.npy'), mmap_mode='r')
        if self.visual_global.ndim != 2 or self.visual_global.shape[0] != len(ids):
            raise ValueError(
                'visual_global row count/shape does not match image_ids.json')
        # Optional augmented views (lazily; only required when paired_aug=True).
        self.visual_global_aug0 = None
        self.visual_global_aug1 = None
        if self.paired_aug:
            for k in ('aug0', 'aug1'):
                p = os.path.join(cache_dir, f'visual_global_{k}.f16.npy')
                if not os.path.exists(p):
                    raise FileNotFoundError(
                        f"[CachedFeatureDataset] paired_aug=True but {p} is "
                        f"missing. Build it with "
                        f"`extract_siglip2_features.py --save_aug_views 2 --aug_only`."
                    )
            self.visual_global_aug0 = np.load(
                os.path.join(cache_dir, 'visual_global_aug0.f16.npy'), mmap_mode='r')
            self.visual_global_aug1 = np.load(
                os.path.join(cache_dir, 'visual_global_aug1.f16.npy'), mmap_mode='r')
            for name, array in (
                    ('visual_global_aug0', self.visual_global_aug0),
                    ('visual_global_aug1', self.visual_global_aug1)):
                if array.shape != self.visual_global.shape:
                    raise ValueError(
                        f'{name} shape {array.shape} != canonical global '
                        f'{self.visual_global.shape}')
        self.visual_tokens = None
        self.visual_tokens_aug0 = None
        self.visual_tokens_aug1 = None
        if self.return_visual_tokens:
            token_path = os.path.join(cache_dir, 'visual_tokens.f16.npy')
            if not os.path.exists(token_path):
                raise FileNotFoundError(
                    f"[CachedFeatureDataset] visual tokens requested but {token_path} "
                    "is missing. Use a cache built with local/token features."
                )
            self.visual_tokens = np.load(token_path, mmap_mode='r')
            if self.visual_tokens.ndim != 3 \
                    or self.visual_tokens.shape[0] != len(ids):
                raise ValueError(
                    'visual_tokens row count/shape does not match image_ids.json')
            if self.paired_aug:
                for k in ('aug0', 'aug1'):
                    path = os.path.join(cache_dir, f'visual_tokens_{k}.f16.npy')
                    if not os.path.exists(path):
                        raise FileNotFoundError(
                            f"[CachedFeatureDataset] paired token view {path} is missing."
                        )
                self.visual_tokens_aug0 = np.load(
                    os.path.join(cache_dir, 'visual_tokens_aug0.f16.npy'), mmap_mode='r')
                self.visual_tokens_aug1 = np.load(
                    os.path.join(cache_dir, 'visual_tokens_aug1.f16.npy'), mmap_mode='r')
                for name, array in (
                        ('visual_tokens_aug0', self.visual_tokens_aug0),
                        ('visual_tokens_aug1', self.visual_tokens_aug1)):
                    if array.shape != self.visual_tokens.shape:
                        raise ValueError(
                            f'{name} shape {array.shape} != canonical tokens '
                            f'{self.visual_tokens.shape}')
        # build (rows, labels, paths) for this split
        if dataset_name == 'CIFAR10':
            self._build_cifar10(setting, mode, dataset_dir)
        else:
            self._build_pathlist(dataset_name, setting, mode, dataset_dir)
        n = self.rows.shape[0]
        n_cls = NUM_CLASS[dataset_name][setting]
        assert self.labels.shape == (n, n_cls), \
            f'labels shape {self.labels.shape} != ({n}, {n_cls})'

    def _build_cifar10(self, setting, mode, dataset_dir):
        from torchvision.datasets import CIFAR10
        # reuse our existing samplers + image hasher
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from dataloaders import _cifar10_image_id, get_idx_for_uniform_sampling

        if setting != 'setting1':
            raise NotImplementedError(setting)
        if mode == 'train':
            ds = CIFAR10(os.path.join(dataset_dir, 'CIFAR10'), train=True, download=True)
            idx = get_idx_for_uniform_sampling(ds, 10, 500)
            data = ds.data[idx]; targets = np.array(ds.targets)[idx]
        elif mode in ('test', 'query'):
            ds = CIFAR10(os.path.join(dataset_dir, 'CIFAR10'), train=False, download=True)
            idx = get_idx_for_uniform_sampling(ds, 10, 100)
            data = ds.data[idx]; targets = np.array(ds.targets)[idx]
        elif mode == 'database':
            trainset = CIFAR10(os.path.join(dataset_dir, 'CIFAR10'), train=True, download=True)
            testset  = CIFAR10(os.path.join(dataset_dir, 'CIFAR10'), train=False, download=True)
            data    = np.concatenate((trainset.data, testset.data))
            targets = np.concatenate((np.array(trainset.targets), np.array(testset.targets)))
            idx     = get_idx_for_uniform_sampling(testset, 10, 900, offset=100)
            idx     = np.concatenate((np.arange(0, len(trainset)), idx + len(trainset)))
            data = data[idx]; targets = targets[idx]
        else:
            raise KeyError(mode)
        rows = np.fromiter(
            (self.id_to_row[_cifar10_image_id(data[i])] for i in range(len(data))),
            dtype=np.int64, count=len(data),
        )
        n_cls = NUM_CLASS['CIFAR10']['setting1']
        labels = np.eye(n_cls, dtype=np.int64)[targets]
        self.rows = rows; self.labels = labels
        # Use the immutable cache IDs, not split-local positions. Query and
        # database overlap detection must identify the same CIFAR image.
        self.paths = [self.image_ids[int(row)] for row in rows]

    def _build_pathlist(self, dataset_name, setting, mode, dataset_dir):
        if mode == 'train':           base = 'train.txt'
        elif mode in ('test','query'): base = 'test.txt'
        elif mode == 'database':       base = 'database.txt'
        else: raise KeyError(mode)
        split_txt = os.path.join(dataset_dir, dataset_name, setting, base)
        rows: List[int] = []; labels: List[List[int]] = []; paths: List[str] = []
        with open(split_txt) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.split()
                relpath = parts[0]
                r = self.id_to_row.get(relpath)
                if r is None:
                    raise KeyError(f'cache miss for {relpath}')
                rows.append(r)
                labels.append([int(x) for x in parts[1:]])
                paths.append(relpath)
        self.rows = np.array(rows, dtype=np.int64)
        self.labels = np.array(labels, dtype=np.int64)
        self.paths = paths

    def restrict_to(self, indices) -> None:
        """Shrink this split in place to `indices` (P0 optimization-train).

        Restricting the dataset's own rows/labels/paths -- rather than wrapping
        it in a torch Subset -- is what keeps `idx` correct: `__getitem__`
        returns its positional index, and MLS3RDUH (`x_feature[batch['idx']]`)
        and CIMON (`S_1[batch['idx']][:, batch['idx']]`) use it to address
        train-sized buffers. A Subset would pass through the ORIGINAL indices
        while those buffers were sized to the subset, indexing out of range.
        """
        idx = np.asarray(indices, dtype=np.int64)
        self.rows   = self.rows[idx]
        self.labels = self.labels[idx]
        self.paths  = [self.paths[int(i)] for i in idx]

    def __len__(self): return int(self.rows.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = int(self.rows[idx])
        feat = torch.from_numpy(np.asarray(self.visual_global[r], dtype=np.float32))   # [D_proj]
        lbl  = torch.from_numpy(self.labels[idx]).float()                               # [n_class]
        out: Dict[str, Any] = {'img': feat, 'label': lbl, 'image_path': self.paths[idx]}
        if self.paired_aug and self.visual_global_aug0 is not None:
            out['img_tr1'] = torch.from_numpy(np.asarray(self.visual_global_aug0[r], dtype=np.float32))
            out['img_tr2'] = torch.from_numpy(np.asarray(self.visual_global_aug1[r], dtype=np.float32))
        if self.return_visual_tokens and self.visual_tokens is not None:
            out['visual_tokens'] = torch.from_numpy(
                np.asarray(self.visual_tokens[r], dtype=np.float32))
            if self.paired_aug and self.visual_tokens_aug0 is not None:
                out['visual_tokens_tr1'] = torch.from_numpy(
                    np.asarray(self.visual_tokens_aug0[r], dtype=np.float32))
                out['visual_tokens_tr2'] = torch.from_numpy(
                    np.asarray(self.visual_tokens_aug1[r], dtype=np.float32))
        if self.return_index:
            out['idx'] = idx
        return out


def load_dataset(dataset_root: str, dataset_name: str, setting: str,
                 train_transform=None, test_transform=None,
                 load_train: bool = True, load_test: bool = True, load_database: bool = True,
                 return_index: bool = False, return_paired_aug_img: bool = False,
                 return_visual_tokens: bool = False,
                 cache_dir: Optional[str] = None) -> Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset]]:
    """Replaces lib.dataloaders.load_dataset. Cached-feature backed.

    Train split honors `return_index` and `return_paired_aug_img` so
    unsupervised baselines (CIBHash / CIMON NtXent, MLS3RDUH kNN graph) get
    the auxiliary fields they expect. Test/database splits stay deterministic
    (single-feature, no aug pairs) so retrieval evaluation is unaffected.
    """
    train = test = db = None
    if load_train:
        train = CachedFeatureDataset(
            dataset_name, setting, 'train', dataset_root, cache_dir,
            paired_aug=return_paired_aug_img, return_index=return_index,
            return_visual_tokens=return_visual_tokens,
        )
    if load_test:
        test = CachedFeatureDataset(dataset_name, setting, 'test', dataset_root, cache_dir)
    if load_database:
        db = CachedFeatureDataset(dataset_name, setting, 'database', dataset_root, cache_dir)
    return train, test, db


# ============================================================ extraction + evaluation

@torch.no_grad()
def _extract_codes(model: Module, loader: DataLoader, device: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the encoder over a split. Returns continuous codes + sign-binary codes + labels.

    Shapes:
        cont : [N, bit] float32
        bin  : [N, bit] int8 in {-1, +1}
        lbl  : [N, n_class] int64
    """
    model.eval()
    cont_chunks, lbl_chunks = [], []
    for batch in loader:
        feat = batch['img'].to(device)
        out  = model(feat)
        cont_chunks.append(out['continuous_code'].detach().cpu().numpy())
        lbl_chunks.append(batch['label'].numpy().astype(np.int64))
    cont = np.concatenate(cont_chunks, axis=0).astype(np.float32)
    lbls = np.concatenate(lbl_chunks,  axis=0)
    bin_ = np.sign(cont).astype(np.int8)
    bin_[bin_ == 0] = 1     # tie-break to +1
    return cont, bin_, lbls


def _hamming_distance_signed(qb: np.ndarray, db: np.ndarray) -> np.ndarray:
    """Signed binary Hamming distance.
    qb, db : [Nq, bit] / [Nd, bit] in {-1, +1} int8
    -> dist [Nq, Nd] int32, range [0, bit]. d = (bit - q.dot(d)) / 2.
    """
    inner = qb.astype(np.int32) @ db.astype(np.int32).T
    bit = qb.shape[1]
    return ((bit - inner) // 2).astype(np.int32)


def signed_bits_to_base_indices(bits_pm: np.ndarray) -> np.ndarray:
    """Pack each consecutive bit pair into one A/C/G/T-style base index.

    The numeric alphabet is irrelevant to base-Hamming distance; the mapping
    is the repository convention ``00,01,10,11 -> 0,1,2,3``.
    """
    bits_pm = np.asarray(bits_pm)
    if bits_pm.ndim != 2 or bits_pm.shape[1] % 2:
        raise ValueError(f'expected [N, even_bits], got {bits_pm.shape}')
    bits01 = (bits_pm > 0).astype(np.int8)
    paired = bits01.reshape(bits01.shape[0], -1, 2)
    return (2 * paired[..., 0] + paired[..., 1]).astype(np.int8)


def _base_hamming_distance(query_bases: np.ndarray,
                           database_bases: np.ndarray) -> np.ndarray:
    query_bases = np.asarray(query_bases, dtype=np.int8)
    database_bases = np.asarray(database_bases, dtype=np.int8)
    if query_bases.ndim != 2 or database_bases.ndim != 2:
        raise ValueError('base codes must be rank-2 arrays')
    if query_bases.shape[1] != database_bases.shape[1]:
        raise ValueError('query/database base lengths differ')
    # Avoid materialising [Nq, Nd, L]. Four indicator GEMMs need only the
    # final [Nq, Nd] matrix and are exact for the four-symbol alphabet.
    matches = np.zeros(
        (query_bases.shape[0], database_bases.shape[0]), dtype=np.int16)
    for base in range(4):
        q = (query_bases == base).astype(np.int16)
        d = (database_bases == base).astype(np.int16)
        matches += q @ d.T
    return (query_bases.shape[1] - matches).astype(np.int16)


def _multi_hot_relevance(qy: np.ndarray, db: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """Pairwise binary relevance = 1 iff Jaccard(query, db) > threshold.
    Mirrors evaluation_siglip2._compute_relevance.
    """
    Lq = qy.astype(np.float32); Ld = db.astype(np.float32)
    inter = Lq @ Ld.T
    union = Lq.sum(axis=1, keepdims=True) + Ld.sum(axis=1, keepdims=True).T - inter
    sim = np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)
    return (sim > threshold).astype(np.uint8)


# Paper-standard mAP@R cutoffs (must match evaluation_siglip2.MAP_AT_R_BY_DATASET)
MAP_AT_R_BY_DATASET: Dict[str, int] = {
    'CIFAR10': 1000, 'NUSWIDE': 5000, 'MSCOCO': 5000, 'Flickr25k': 5000,
}


def _ap_at_r(rel_sorted: np.ndarray, R: Optional[int] = None) -> float:
    """Truncated AP (canonical deep-hashing CalcTopMap; R=None => full AP)."""
    tgnd = rel_sorted if R is None else rel_sorted[:R]
    tsum = int(tgnd.sum())
    if tsum == 0: return 0.0
    tindex = np.where(tgnd == 1)[0] + 1.0
    counts = np.arange(1, tsum + 1, dtype=np.float64)
    return float((counts / tindex).mean())


def _ap_from_sorted_relevance(rel_sorted: np.ndarray) -> float:
    return _ap_at_r(rel_sorted, R=None)


def evaluate_retrieval_model(query_codes: dict, retrieval_codes: dict,
                             query_labels: np.ndarray, retrieval_labels: np.ndarray,
                             precision_at_k_list=(1, 5, 10, 20, 50, 100, 500, 1000),
                             multi_label_relevance_threshold: float = 0.0,
                             map_at_r: Optional[int] = None,
                             **_unused) -> dict:
    """Replaces lib.evaluation.evaluate_retrieval_model.

    Computes mAP + P@K + R@K via signed binary Hamming distance on `code['B']`.
    `query_codes` / `retrieval_codes` are dicts at minimum containing key 'B'
    (signed-binary codes [N, bit] in {-1, +1}). Other keys ignored.
    """
    qb = np.asarray(query_codes['B'])
    db = np.asarray(retrieval_codes['B'])
    distances = _hamming_distance_signed(qb, db)                   # [Nq, Nd]
    return _evaluate_distance_matrix(
        distances, query_labels, retrieval_labels,
        precision_at_k_list=precision_at_k_list,
        multi_label_relevance_threshold=multi_label_relevance_threshold,
        map_at_r=map_at_r, description='eval[binary]',
    )


def evaluate_base_retrieval_model(query_bits: np.ndarray, database_bits: np.ndarray,
                                  query_labels: np.ndarray,
                                  database_labels: np.ndarray,
                                  precision_at_k_list=(1, 5, 10, 20, 50, 100, 500, 1000),
                                  multi_label_relevance_threshold: float = 0.0,
                                  map_at_r: Optional[int] = None) -> dict:
    """Evaluate the exact 2-bit-to-base control used by the paper protocol."""
    query_bases = signed_bits_to_base_indices(query_bits)
    database_bases = signed_bits_to_base_indices(database_bits)
    distances = _base_hamming_distance(query_bases, database_bases)
    return _evaluate_distance_matrix(
        distances, query_labels, database_labels,
        precision_at_k_list=precision_at_k_list,
        multi_label_relevance_threshold=multi_label_relevance_threshold,
        map_at_r=map_at_r, description='eval[base-2bit]',
    )


def _evaluate_distance_matrix(distances: np.ndarray,
                              query_labels: np.ndarray,
                              retrieval_labels: np.ndarray,
                              precision_at_k_list,
                              multi_label_relevance_threshold: float,
                              map_at_r: Optional[int],
                              description: str) -> dict:
    relevance = _multi_hot_relevance(query_labels, retrieval_labels,
                                     threshold=multi_label_relevance_threshold)
    Nq, Nd = distances.shape
    aps = []
    aps_at_r = []
    p_at_k = {k: [] for k in precision_at_k_list}
    r_at_k = {k: [] for k in precision_at_k_list}
    for i in tqdm(range(Nq), desc=description):
        d = distances[i]; r = relevance[i]
        order = np.argsort(d, kind='stable')
        rs = r[order]
        aps.append(_ap_from_sorted_relevance(rs))
        if map_at_r is not None:
            aps_at_r.append(_ap_at_r(rs, R=int(map_at_r)))
        nrel = int(r.sum())
        for k in precision_at_k_list:
            kk = min(k, Nd); top = rs[:kk]
            p_at_k[k].append(float(top.mean()) if kk > 0 else 0.0)
            r_at_k[k].append(float(top.sum() / max(nrel, 1)))
    return {
        'mAP':            float(np.mean(aps)) if aps else 0.0,
        'precision_at_k': {int(k): float(np.mean(v)) for k, v in p_at_k.items()},
        'recall_at_k':    {int(k): float(np.mean(v)) for k, v in r_at_k.items()},
        **({'mAP_at_R': float(np.mean(aps_at_r)) if aps_at_r else 0.0,
            'mAP_R_cutoff': int(map_at_r)} if map_at_r is not None else {}),
    }


def calculate_affinity(query_labels, retrieval_labels):
    """Replaces lib.evaluation.calculate_affinity. Currently unused by our flow."""
    return None


def compress_images(*args, **kwargs):
    """Stub. Original lib.compress.compress_images ran the CNN backbone -- we
    do extraction inline via `_extract_codes` instead. Kept for interface symmetry."""
    raise NotImplementedError("Use _extract_codes (cached-feature path) instead.")


def make_hash_table(*args, **kwargs):
    return None


def convert_all_type_codes(*args, **kwargs):
    """Stub. Quaternary-code conversion (DNA-style constraints) not used by binary baselines."""
    raise NotImplementedError


# ============================================================ DeepHashBase (slimmed)

BOLD = '\033[1m'; END = '\033[0m'
HIGH = '\x1b[6;30;42m'; EHIGH = '\x1b[0m'


class DeepHashBase(metaclass=ABCMeta):
    """Slim abstract base for binary deep-hashing baselines on cached SigLIP2.

    Subclasses must implement: _get_default_config_dict, _get_config_dict_for_dataset,
    _get_fixed_config_dict, _add_model_specific_args_into_parser, _train_model.
    Everything else (logger, dataset, encoder build, eval) is provided here and
    replaces the missing `lib/*` modules with cached-feature equivalents.
    """
    def __init__(self) -> None:
        self._n_class:    int = -1
        self._is_multi_label: bool = False
        self.backbone_with_encoder: Module = None  # type: ignore
        self.optimizer: Optimizer = None           # type: ignore
        self.scheduler: _LRScheduler = None        # type: ignore
        self.config: dict = {}
        self.device: str = 'cpu'
        self.logger: Logger = None                 # type: ignore
        self.trainset: Dataset = None              # type: ignore
        self.valset: Dataset = None                # type: ignore
        self.testset:  Dataset = None              # type: ignore
        self.dbset:    Dataset = None              # type: ignore
        self.model_dir: str = ''
        self.result_dir: str = ''
        self.compress_dir: str = ''
        self.required_args = ['dataset', 'setting', 'bit', 'optimizer_name',
                              'lr_scheduler', 'finetune', 'gen_code_method']
        self.dataset_name_list = list(NUM_CLASS.keys())

    # ---- abstract methods (per-baseline overrides) --------------------
    @abstractmethod
    def _get_default_config_dict(self) -> dict: ...
    @abstractmethod
    def _get_config_dict_for_dataset(self, default_config: dict, dataset: str) -> dict: ...
    @abstractmethod
    def _get_fixed_config_dict(self) -> dict: ...
    @abstractmethod
    def _add_model_specific_args_into_parser(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser: ...
    @abstractmethod
    def _train_model(self, backbone_with_encoder: Module, optimizer: Optimizer,
                     scheduler: _LRScheduler, trainset: Dataset, testset: Dataset,
                     dbset: Dataset, model_dir: str, result_dir: str, device: str,
                     batch_size: int, eval_period: int, config: dict) -> None: ...

    # ---- argparse plumbing --------------------------------------------
    def _add_parent_args_into_parser(self, parser: argparse.ArgumentParser):
        g = parser.add_argument_group(BOLD + 'Experiment Setting' + END)
        g.add_argument('-tn', '--trial_name', type=str, default='', help='Trial name (auto-built if blank)')
        g.add_argument(
            '--protocol_identity_sha256', default=None, type=str,
            help=('Optional immutable protocol fingerprint supplied by an '
                  'orchestrating comparison driver and persisted in checkpoints.'),
        )
        g.add_argument(
            '--expected_cache_artifact_sha256_json', default=None, type=str,
            help=('Canonical JSON mapping of every consumed cache .npy to its '
                  'driver-computed SHA-256.'),
        )

        g = parser.add_argument_group(BOLD + 'Model' + END)
        g.add_argument('-gcm', '--gen_code_method', type=str, default='sign')
        g.add_argument('-mr', '--model_root', default='./params/', type=str)
        g.add_argument('--seed', type=int, default=1)
        g.add_argument('--finetune', action=argparse.BooleanOptionalAction, default=False,
                       help='Ignored: backbone is cached SigLIP2 (frozen).')

        g = parser.add_argument_group(BOLD + 'Encoder head' + END)
        g.add_argument('-bb', '--backbone', type=str, default='SigLIP2',
                       help='Fixed: cached SigLIP2 features.')
        g.add_argument('-el', '--encoder_layers', type=str, default='none',
                       help='version1=2-layer 4096-hidden / version2=1-layer / layer=N+hidden=H / none=1-layer')
        g.add_argument('-et', '--encoder_type', type=str, default='linear')
        g.add_argument('-bit', '--bit', type=int, default=36,
                       help='Hash bit length (default matches DNA storage 18 base x 2 bit).')
        g.add_argument('--batch_norm', action=argparse.BooleanOptionalAction, default=False)

        g = parser.add_argument_group(BOLD + 'Dataset' + END)
        g.add_argument('-d', '--dataset', type=str, required=True,
                       choices=list(NUM_CLASS.keys()))
        g.add_argument('-s', '--setting', type=str, default='setting1')
        g.add_argument('-t', '--transform', default='default', type=str)
        g.add_argument('--dataset_return_index',         action=argparse.BooleanOptionalAction, default=False)
        g.add_argument('--dataset_return_paired_aug_img', action=argparse.BooleanOptionalAction, default=False)
        g.add_argument('--dataset_return_visual_tokens', action=argparse.BooleanOptionalAction, default=False)
        g.add_argument('-dr', '--dataset_root', default='./dataset/', type=str)
        g.add_argument('--cache_dir', default=None, type=str,
                       help='Override SigLIP2 feature cache dir (default: ./cache/<dataset>_siglip2/).')

        g = parser.add_argument_group(BOLD + 'Device' + END)
        g.add_argument('-device', '--device', default='cuda:0', type=str)
        g.add_argument('-batch', '--batch_size', default=64, type=int)
        g.add_argument('--num_workers', default=4, type=int)

        g = parser.add_argument_group(BOLD + 'Train Loop' + END)
        g.add_argument('-me', '--max_epoch', default=60, type=int)
        g.add_argument('-ep', '--eval_period', default=20, type=int)
        g.add_argument(
            '--schedule_horizon', default=None, type=int,
            help=('Nominal full training horizon used by schedules.  P0 stage 2 '
                  'may stop at E* while retaining the stage-1 schedule instead '
                  'of compressing cosine decay into E*+1 epochs.'),
        )
        # P0 stage 1: train on the optimization-train split only, so the epoch
        # can later be selected on val rows the model never trained on. Uses the
        # same carve-out as train_siglip2.py (val_split.py) -- both sides of the
        # comparison must hold out the identical rows.
        g.add_argument('--val_split_ratio', default=0.0, type=float,
                       help='Hold out this fraction of train (P0 stage 1). '
                            '0.0 = train on the full split (stage 2 / legacy).')
        g.add_argument('--val_split_seed', default=42, type=int)

        g = parser.add_argument_group(BOLD + 'Optimizer' + END)
        g.add_argument('-opt', '--optimizer_name', type=str, default='adam')
        g.add_argument('--sgd_weight_decay', type=float, default=1e-5)
        g.add_argument('--sgd_momentum',     type=float, default=0.99)
        g.add_argument('--adam_weight_decay', type=float, default=0.0)

        g = parser.add_argument_group(BOLD + 'Scheduler' + END)
        g.add_argument('-lrschd', '--lr_scheduler', type=str, default='none')
        g.add_argument('-lr', '--learning_rate', default=1e-4, type=float)
        g.add_argument('--coslr_T_max', type=int, default=20)
        g.add_argument('--explr_gamma', type=float, default=0.95)
        g.add_argument('--step_size', type=int, default=80)
        g.add_argument('--step_gamma', type=float, default=0.1)

        g = parser.add_argument_group(BOLD + 'Output' + END)
        g.add_argument('-rr', '--result_root', default='./result_baseline/', type=str)
        g.add_argument('--compress_root',      default='./compress_baseline/', type=str)
        # eval flags retained for legacy interface compatibility (always-on mAP)
        g.add_argument('-mAP_N', '--eval_mAP_topN', action=argparse.BooleanOptionalAction, default=True)
        g.add_argument('-mAPt',  '--eval_mAPt',     action=argparse.BooleanOptionalAction, default=False)
        g.add_argument('-mNDCG', '--eval_mNDCG',    action=argparse.BooleanOptionalAction, default=False)
        g.add_argument('-mAP_R', '--eval_mAP_R',    action=argparse.BooleanOptionalAction, default=False)
        g.add_argument('-RAMAP', '--eval_RAMAP',    action=argparse.BooleanOptionalAction, default=False)
        g.add_argument('-mLGAP', '--eval_mLGAP',    action=argparse.BooleanOptionalAction, default=False)
        g.add_argument('--eval_max_radius', type=int, default=2)
        g.add_argument('-gc', '--eval_gc_content', action=argparse.BooleanOptionalAction, default=False)
        g.add_argument('-seq_r', '--eval_seq_r',   action=argparse.BooleanOptionalAction, default=False)
        g.add_argument('--save_code',     action=argparse.BooleanOptionalAction, default=False)
        g.add_argument('--save_affinity', action=argparse.BooleanOptionalAction, default=False)
        g.add_argument('--save_table',    action=argparse.BooleanOptionalAction, default=False)
        g.add_argument('--convert_quat_code_with_constraints',
                       action=argparse.BooleanOptionalAction, default=False)
        g.add_argument('--max_seq_r', default=3, type=int)

    def get_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
        self._add_parent_args_into_parser(parser)
        self._add_model_specific_args_into_parser(parser)
        # apply baseline defaults
        default_cfg = self._get_default_config_dict()
        fixed_cfg   = self._get_fixed_config_dict()
        merged = {**default_cfg, **fixed_cfg}
        # only set defaults for known argparse keys (skip keys baseline added themselves)
        actions = parser._get_optional_actions()
        known_dests = {a.dest for a in actions}
        merged = {k: v for k, v in merged.items() if k in known_dests}
        parser.set_defaults(**merged)
        return parser

    # ---- experiment plumbing ------------------------------------------
    def _make_and_register_exp_dirs(self, dir_tag: str, root: str, day_info: str, trial_name: str) -> str:
        d = os.path.join(root, day_info, trial_name)
        self.config[dir_tag] = d
        os.makedirs(d, exist_ok=True)
        return d

    def _check_required_args_not_None(self, config: dict):
        for key in self.required_args:
            assert config.get(key) is not None, f'config missing: {key}'

    def _modify_trial_name_from_config(self, config: dict) -> None:
        if config.get('trial_name'):
            return
        # method + dataset + bit + lr -> trial name
        method = self.__class__.__name__
        config['trial_name'] = f"{method}-d-{config['dataset']}-bit-{config['bit']}-lr-{config['learning_rate']}"

    def _set_n_class(self, dataset_name: str, setting_name: str) -> None:
        self._n_class = NUM_CLASS[dataset_name][setting_name]

    def _set_multi_label(self, dataset_name: str) -> None:
        self._is_multi_label = MULTI_LABEL[dataset_name]

    def _set_device(self, name: str) -> None:
        self.device = str(name)

    def _build_model_from_config(self, d_in: int, config: dict) -> Module:
        """Construct the method model for training or checkpoint evaluation.

        Modern baselines override this hook when the paper's hashing head is
        not the legacy Linear/GELU stack.  Keeping construction in one hook is
        what lets validation and extraction rebuild the exact same module.
        """
        encoder_layers = str(config.get('encoder_layers', 'none'))
        # parse encoder_layers spec same as the original
        if encoder_layers in ('version1', 'layer=2+hidden=4096'):
            hidden_nodes = [4096]
        elif encoder_layers in ('version2', 'layer=1', 'none'):
            hidden_nodes = []
        else:
            options = encoder_layers.split('+')
            cfg = {opt.split('=')[0]: int(opt.split('=')[1]) for opt in options}
            hidden_nodes = [cfg['hidden']] * (cfg['layer'] - 1)
        return BackboneWithEncoder(
            d_in=d_in,
            bit=int(config['bit']),
            hidden_nodes=hidden_nodes,
            batch_norm=bool(config.get('batch_norm', False)),
            finetune=False,
        )

    def _init_backbone_with_encoder(self, backbone, encoder_layers, encode_length,
                                    finetune, batch_norm, encoder_type='linear') -> Module:
        # Auto-detect feature dim from cached visual_global. SigLIP2 has
        # D_proj=768, CLIP-vit-base-patch16 has D_proj=512. Falls back to
        # 768 if the dataset attribute is unavailable for any reason.
        d_in = 768
        if hasattr(self, "trainset") and hasattr(self.trainset, "visual_global"):
            try:
                d_in = int(self.trainset.visual_global.shape[1])
            except Exception:
                pass
        return self._build_model_from_config(d_in, self.config)

    def _init_optimizer(self, model: Module, name: str, learning_rate: float, **opts) -> Optimizer:
        if name == 'sgd':
            return torch.optim.SGD(model.parameters(), lr=learning_rate,
                                   momentum=opts.get('sgd_momentum', 0.99),
                                   weight_decay=opts.get('sgd_weight_decay', 1e-5))
        if name == 'adam':
            return torch.optim.Adam(
                model.parameters(), lr=learning_rate,
                weight_decay=float(opts.get('adam_weight_decay', 0.0)),
            )
        raise KeyError(name)

    def _init_scheduler(self, optimizer: Optimizer, name: str, **opts) -> _LRScheduler:
        if name == 'coslr':
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                T_max=int(opts.get('coslr_T_max', 20)))
        if name == 'explr':
            return torch.optim.lr_scheduler.ExponentialLR(optimizer,
                gamma=float(opts.get('explr_gamma', 0.95)))
        if name == 'step':
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=int(opts.get('step_size', 80)),
                gamma=float(opts.get('step_gamma', 0.1)),
            )
        if name == 'none':
            return _NoOpScheduler(optimizer)
        raise NotImplementedError(name)

    def _get_n_class(self) -> int: return self._n_class
    def is_multi_label_class(self) -> bool: return self._is_multi_label

    # logger helpers (called by baselines)
    def _insert_iter_loss_sum(self, loss_sum_info: dict, current_batch_size: int) -> None:
        self.logger.insert_iter_loss_sum(loss_sum_info, current_batch_size)
    def _compute_loss_per_epoch(self) -> None:
        self.logger.compute_loss_per_epoch()
    def _save_train_model_log(self, model_id, id_type, enable_print=False, print_message='') -> None:
        self.logger.save_train_model_log(model_id, id_type, enable_print, print_message)
    def _save_train_model_params(self, model_id, id_type, etc_info: dict | None = None) -> None:
        info = {'encoder_layers': self.backbone_with_encoder.encoder_layers.state_dict(),
                'model_state_dict': self.backbone_with_encoder.state_dict(),
                'backbone': None,
                'optimizer': self.optimizer.state_dict(),
                'config': self.config}
        if etc_info:
            for k, v in etc_info.items():
                info[k] = v
        path = os.path.join(self.model_dir, f'{id_type}_{model_id}.pth')
        os.makedirs(self.model_dir, exist_ok=True)
        torch.save(info, path)

    # ---- experiment lifecycle -----------------------------------------
    def init_experiment(self, config: dict, rename: bool = True) -> None:
        if rename:
            self._modify_trial_name_from_config(config)
        config['day_info'] = time.strftime('%y%m%d', time.localtime())
        config['runtime_versions'] = {
            'python': '.'.join(map(str, sys.version_info[:3])),
            'torch': str(torch.__version__),
            'numpy': str(np.__version__),
            'cuda_runtime': None if torch.version.cuda is None
            else str(torch.version.cuda),
        }
        requested_device = torch.device(str(config['device']))
        if requested_device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError(
                f'CUDA device {config["device"]!r} was requested but CUDA is '
                'not available; choose --device cpu explicitly')
        config['runtime_accelerator'] = (
            torch.cuda.get_device_name(requested_device)
            if requested_device.type == 'cuda' else 'cpu')
        # apply fixed config last (overrides any default)
        for k, v in self._get_fixed_config_dict().items():
            config[k] = v
        self._check_required_args_not_None(config)
        self.config = config
        fix_random_seed(config['seed'])
        self.model_dir    = self._make_and_register_exp_dirs('model_dir',    config['model_root'],   config['day_info'], config['trial_name'])
        self.result_dir   = self._make_and_register_exp_dirs('result_dir',   config['result_root'],  config['day_info'], config['trial_name'])
        self.compress_dir = self._make_and_register_exp_dirs('compress_dir', config['compress_root'], config['day_info'], config['trial_name'])
        # cached-feature datasets only -- no transforms
        # Honor each baseline's `_get_fixed_config_dict()` request for
        # auxiliary fields (paired-aug pairs for CIBHash/CIMON, idx for
        # CIMON/MLS3RDUH). The keys come from the parent argparse
        # (`--dataset_return_index`, `--dataset_return_paired_aug_img`)
        # and are overridden last by the fixed_config dict.
        _vr = float(config.get('val_split_ratio', 0.0) or 0.0)
        _stage1 = _vr > 0.0
        # Keep repeated initialization on one object honest as well: stage 2
        # must not inherit an earlier stage-1 validation dataset.
        self.valset = None
        self.trainset, self.testset, self.dbset = load_dataset(
            config['dataset_root'], config['dataset'], config['setting'],
            None, None,
            load_test=not _stage1,
            load_database=not _stage1,
            return_index=bool(config.get('dataset_return_index', False)),
            return_paired_aug_img=bool(config.get('dataset_return_paired_aug_img', False)),
            return_visual_tokens=bool(config.get('dataset_return_visual_tokens', False)),
            cache_dir=config.get('cache_dir'),
        )
        config['resolved_cache_dir'] = os.path.realpath(str(self.trainset.cache_dir))
        consumed_artifacts = consumed_cache_artifact_names(
            paired_aug=bool(config.get('dataset_return_paired_aug_img', False)),
            visual_tokens=bool(config.get('dataset_return_visual_tokens', False)),
        )
        actual_artifact_hashes = hash_cache_artifacts(
            self.trainset.cache_dir, consumed_artifacts)
        expected_artifact_json = config.get(
            'expected_cache_artifact_sha256_json')
        if expected_artifact_json is not None:
            try:
                expected_artifact_hashes = json.loads(expected_artifact_json)
            except (TypeError, json.JSONDecodeError) as error:
                raise ValueError(
                    'invalid --expected_cache_artifact_sha256_json') from error
            if expected_artifact_hashes != actual_artifact_hashes:
                raise ValueError(
                    'trainer cache artifact hashes differ from driver protocol: '
                    f'{expected_artifact_hashes} != {actual_artifact_hashes}')
            config['cache_artifact_binding_status'] = 'verified_against_driver'
        else:
            config['cache_artifact_binding_status'] = (
                'locally_hashed_without_driver_expectation')
        config['resolved_cache_artifact_sha256'] = actual_artifact_hashes
        meta_path = os.path.join(self.trainset.cache_dir, 'meta.json')
        if os.path.isfile(meta_path):
            with open(meta_path) as handle:
                cache_meta = json.load(handle)
            config['resolved_backbone'] = cache_meta.get('backbone')
            config['resolved_projection_dim'] = cache_meta.get(
                'D_proj', int(self.trainset.visual_global.shape[1]))
            config['resolved_cache_meta_sha256'] = _sha256_file(meta_path)
            config['resolved_cache_image_ids_sha256'] = _sha256_file(
                os.path.join(self.trainset.cache_dir, 'image_ids.json'))
            config['resolved_augmentation_transform'] = cache_meta.get(
                'augmentation_transform')
            config['augmentation_provenance_status'] = (
                'recorded_in_cache_meta'
                if cache_meta.get('augmentation_transform') is not None
                else 'legacy_cache_meta_omits_transform_spec; matched cache only')
        else:
            config['resolved_backbone'] = 'unknown-cache-backbone'
            config['resolved_projection_dim'] = int(self.trainset.visual_global.shape[1])
        config['protocol_stage'] = 'P0_stage1_val_selection' if _stage1 else 'P0_stage2_refit_test'
        # ---- P0 stage 1: restrict training to the optimization-train rows ----
        if _stage1:
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from val_split import carve_val_indices
            _n = len(self.trainset)
            _opt, _val, _strat = carve_val_indices(
                np.asarray(self.trainset.labels), ratio=_vr,
                seed=int(config.get('val_split_seed', 42)))
            self.valset = CachedFeatureDataset(
                config['dataset'], config['setting'], 'train',
                config['dataset_root'], self.trainset.cache_dir,
                paired_aug=False, return_index=False,
                return_visual_tokens=False,
            )
            self.valset.restrict_to(_val)
            self.trainset.restrict_to(_opt)
            config['val_split_strategy'] = _strat
            config['n_opt_train'] = len(self.trainset)
            config['n_val_query'] = len(self.valset)
            print(f"[val-protocol] P0 stage 1: train {_n} -> opt-train "
                  f"{len(self.trainset)} (+ {len(_val)} val held out; {_strat}, "
                  f"seed={config.get('val_split_seed', 42)}). Epoch will be "
                  f"selected on those held-out rows, not on test.")
        print(pd.Series(config))
        self._set_n_class(config['dataset'], config['setting'])
        self._set_multi_label(config['dataset'])
        self.logger = Logger(self.result_dir, config=config)
        self.backbone_with_encoder = self._init_backbone_with_encoder(
            config['backbone'], config['encoder_layers'], config['bit'],
            config['finetune'], config['batch_norm'], config['encoder_type'])
        self.optimizer = self._init_optimizer(model=self.backbone_with_encoder,
                                              name=config['optimizer_name'], **config)
        self.scheduler = self._init_scheduler(optimizer=self.optimizer,
                                              name=config['lr_scheduler'], **config)
        self._set_device(config['device'])
        self.batch_size = config['batch_size']
        self.eval_period = config['eval_period']

    def start_train_process(self) -> None:
        self._train_model(self.backbone_with_encoder, self.optimizer, self.scheduler,
                          self.trainset, self.testset, self.dbset,
                          self.model_dir, self.result_dir, self.device,
                          self.batch_size, self.eval_period, self.config)

    def start_eval_process(self, model_id: Any, id_type: str = 'epoch') -> None:
        # P0 stage 1 never loads or evaluates test. Validation queries are
        # ranked against the disjoint optimization-train database. Stage 2
        # uses the official test/database split after the epoch is fixed.
        is_stage1 = self.valset is not None
        query_set = self.valset if is_stage1 else self.testset
        database_set = self.trainset if is_stage1 else self.dbset
        if query_set is None or database_set is None:
            raise RuntimeError('evaluation datasets were not initialized')
        num_workers = int(self.config.get('num_workers', 4))
        qy_loader = DataLoader(
            query_set, batch_size=self.batch_size, shuffle=False,
            num_workers=num_workers)
        db_loader = DataLoader(
            database_set, batch_size=self.batch_size, shuffle=False,
            num_workers=num_workers)
        model = self.backbone_with_encoder.to(self.device)
        cont_q, bin_q, lbl_q = _extract_codes(model, qy_loader, self.device)
        cont_d, bin_d, lbl_d = _extract_codes(model, db_loader, self.device)
        _map_r = MAP_AT_R_BY_DATASET.get(str(self.config.get('dataset')))
        binary_result = evaluate_retrieval_model(
            query_codes={'B': bin_q, 'C': cont_q},
            retrieval_codes={'B': bin_d, 'C': cont_d},
            query_labels=lbl_q, retrieval_labels=lbl_d,
            map_at_r=_map_r,
        )
        base_result = evaluate_base_retrieval_model(
            bin_q, bin_d, lbl_q, lbl_d, map_at_r=_map_r)
        result = {
            **binary_result,
            'evaluation_split': 'val_query_vs_opt_train' if is_stage1 else 'test_vs_database',
            'test_evaluated': not is_stage1,
            'binary': binary_result,
            'base_2bit': base_result,
        }
        # flatten precision_at_k / recall_at_k for csv/console
        flat = {'binary_mAP': binary_result['mAP'], 'base_mAP': base_result['mAP']}
        if 'mAP_at_R' in binary_result:
            cutoff = binary_result['mAP_R_cutoff']
            flat[f"binary_mAP@{cutoff}"] = binary_result['mAP_at_R']
            flat[f"base_mAP@{cutoff}"] = base_result['mAP_at_R']
        for k, v in binary_result['precision_at_k'].items(): flat[f'binary_P@{k}'] = v
        for k, v in base_result['precision_at_k'].items(): flat[f'base_P@{k}'] = v
        self.logger.save_eval_model_log(model_id, id_type, **flat)
        # persist the full result json (handy for the comparison table)
        prefix = 'eval_val' if is_stage1 else 'eval'
        out_json = os.path.join(self.result_dir, f'{prefix}_{id_type}_{model_id}.json')
        with open(out_json, 'w') as f:
            json.dump(result, f, indent=2)


# ============================================================ checkpoint reconstruction

def _legacy_hidden_nodes_from_config(config: dict) -> List[int]:
    """Parse the historical ``encoder_layers`` mini-language.

    This standalone form is deliberately kept for checkpoints that predate
    ``model_state_dict`` and, in rare cases, also predate the ``method`` key.
    New method-specific heads must be rebuilt through their class hook below.
    """
    spec = str(config.get('encoder_layers', 'none'))
    if spec in ('version1', 'layer=2+hidden=4096'):
        return [4096]
    if spec in ('version2', 'layer=1', 'none'):
        return []
    try:
        options = spec.split('+')
        parsed = {opt.split('=')[0]: int(opt.split('=')[1]) for opt in options}
        return [parsed['hidden']] * (parsed['layer'] - 1)
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError(
            f'cannot reconstruct legacy encoder_layers={spec!r}; the checkpoint '
            'must contain a registered config["method"] for a custom head'
        ) from exc


def _infer_checkpoint_input_dim(checkpoint: dict) -> int:
    config = checkpoint.get('config', {})
    saved_dim = config.get('resolved_projection_dim')
    if saved_dim is not None:
        return int(saved_dim)

    # Historical checkpoints did not persist cache metadata. Infer the input
    # dimension from the first rank-2 encoder weight as a last resort.
    for state_key in ('model_state_dict', 'encoder_layers'):
        state = checkpoint.get(state_key)
        if not isinstance(state, dict):
            continue
        for name, value in state.items():
            if ('encoder_layers' in name or state_key == 'encoder_layers') \
                    and name.endswith('weight') and getattr(value, 'ndim', 0) == 2:
                return int(value.shape[1])
    raise ValueError(
        'cannot infer cached feature dimension from checkpoint; pass d_in explicitly'
    )


def build_model_from_checkpoint_payload(checkpoint: dict,
                                        d_in: Optional[int] = None,
                                        device: str | torch.device = 'cpu',
                                        strict: bool = True) -> Module:
    """Rebuild and load the exact hash model described by a checkpoint.

    New checkpoints carry ``model_state_dict`` and are loaded as complete
    method-specific modules. Historical checkpoints carry only the state of
    ``encoder_layers``; that format remains supported for the legacy methods.
    No optimizer, dataset, output directory, or experiment state is created.

    Args:
        checkpoint: Mapping returned by ``torch.load``.
        d_in: Actual cached global-feature width. If omitted, use persisted
            cache metadata or infer it from a historical encoder weight.
        device: Destination device.
        strict: Forwarded to ``load_state_dict``.
    """
    if not isinstance(checkpoint, dict):
        raise TypeError('checkpoint payload must be a dict')
    config = checkpoint.get('config')
    if not isinstance(config, dict):
        raise KeyError('checkpoint is missing a dict-valued "config"')
    resolved_d_in = _infer_checkpoint_input_dim(checkpoint) if d_in is None else int(d_in)
    if resolved_d_in <= 0:
        raise ValueError(f'd_in must be positive, got {resolved_d_in}')

    method_name = config.get('method')
    if method_name:
        try:
            method = _build_method(str(method_name))
        except SystemExit as exc:
            raise ValueError(
                f'checkpoint method {method_name!r} is not registered in '
                'baseline.base_model._build_method'
            ) from exc
        # Some custom builders consult self.config in addition to the explicit
        # argument. Do not call init_experiment: reconstruction must be read-only.
        method.config = dict(config)
        model = method._build_model_from_config(resolved_d_in, method.config)
    else:
        model = BackboneWithEncoder(
            d_in=resolved_d_in,
            bit=int(config['bit']),
            hidden_nodes=_legacy_hidden_nodes_from_config(config),
            batch_norm=bool(config.get('batch_norm', False)),
            finetune=False,
        )

    full_state = checkpoint.get('model_state_dict')
    if isinstance(full_state, dict):
        model.load_state_dict(full_state, strict=strict)
        load_format = 'model_state_dict'
    else:
        encoder_state = checkpoint.get('encoder_layers')
        if not isinstance(encoder_state, dict):
            raise KeyError(
                'checkpoint contains neither "model_state_dict" nor the legacy '
                '"encoder_layers" state dict'
            )
        if not hasattr(model, 'encoder_layers'):
            raise TypeError(
                f'{type(model).__name__} cannot load a legacy encoder-only checkpoint'
            )
        model.encoder_layers.load_state_dict(encoder_state, strict=strict)
        load_format = 'encoder_layers'

    model.to(device)
    model.eval()
    # Useful for audit logs without changing the public return type.
    model._checkpoint_load_format = load_format  # type: ignore[attr-defined]
    return model


# ============================================================ method dispatch + main()

def _build_method(method: str) -> 'DeepHashBase':
    """Lazy-import the chosen baseline class. Avoids circular imports during file modify."""
    method = method.lower()
    # Supervised (label-driven) baselines.
    if method == 'dpsh':
        from baseline.DPSH import DPSH; return DPSH()
    if method == 'hashnet':
        from baseline.HashNet import HashNet; return HashNet()
    if method == 'csq':
        from baseline.CSQ import CSQ; return CSQ()
    if method == 'orthohash':
        from baseline.OrthoHash import OrthoHash; return OrthoHash()
    if method in ('crh', 'crh-supervised', 'crh_supervised'):
        from baseline.CRH import CRH; return CRH()
    # Unsupervised baselines (v27b comparison set, 2026-05-14).
    # SPQ is intentionally NOT registered: baseline/SPQ.py is incomplete
    # (`_get_fixed_config_dict` syntax error at line 176, `_train_model`
    # never implemented). Reimplementing it properly would require porting
    # the original PQ+CQC training loop -- left as future work.
    if method == 'cibhash':
        from baseline.CIBHash import CIBHash; return CIBHash()
    if method == 'cimon':
        from baseline.CIMON import CIMON; return CIMON()
    if method == 'mls3rduh':
        from baseline.MLS3RDUH import MLS3RDUH; return MLS3RDUH()
    # Recent target-label-free methods. Their U0/U1/U2 information condition
    # is persisted by each class and must remain visible in paper tables.
    if method == 'sdc':
        from baseline.SDC import SDC; return SDC()
    if method == 'hhch':
        from baseline.HHCH import HHCH; return HHCH()
    if method == 'crovca':
        from baseline.CroVCA import CroVCA; return CroVCA()
    if method == 'oh':
        from baseline.OH import OH; return OH()
    if method in ('greedyhash', 'greedy-hash', 'greedy_hash'):
        from baseline.GreedyHash import GreedyHash; return GreedyHash()
    if method in ('bihalf', 'bi-half', 'bi_half'):
        from baseline.BiHalf import BiHalf; return BiHalf()
    if method in ('duheg', 'duh-eg'):
        from baseline.DUHEG import DUHEG; return DUHEG()
    if method == 'umrch':
        from baseline.UMRCH import UMRCH; return UMRCH()
    raise SystemExit(
        f'unknown method: {method!r}. Available supervised: dpsh hashnet csq '
        f'orthohash crh. '
        f'Available target-label-free: cibhash cimon mls3rduh greedyhash bihalf '
        f'sdc hhch crovca oh duheg umrch '
        f'(information conditions differ).'
    )


def main() -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--method', type=str, required=True,
                     help='Supervised: dpsh / hashnet / csq / orthohash / crh. '
                          'Target-label-free: cibhash / cimon / mls3rduh / '
                          'greedyhash / bihalf / sdc / hhch / crovca / oh / '
                          'duheg / umrch.')
    pre_args, remaining = pre.parse_known_args()
    obj = _build_method(pre_args.method)
    parser = obj.get_parser()
    args = parser.parse_args(remaining)
    config = vars(args)
    config['method'] = pre_args.method
    obj.init_experiment(config)
    obj.start_train_process()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
