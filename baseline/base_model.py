"""Self-contained binary deep-hashing baseline runner.

Adapts the original abstract `DeepHashBase` framework (which expected a sister
`lib/` package -- not present in this repo) to the GroundedDNA project layout:

    - The "backbone" is the SigLIP2 vision tower, FROZEN. We do not run it
      online; instead we read its already-extracted `visual_global` embeddings
      from `./cache/<dataset>_siglip2/` (built by extract_siglip2_features.py).
      This is the SAME backbone our DNA model uses, so binary baselines and
      GroundedDNA see identical inputs -- enabling fair comparison.
    - The trainable "encoder" is a tiny Linear(D_proj=768, bit) head per
      baseline, optionally with BatchNorm. Each baseline keeps its loss code
      verbatim; we just plug their `output['continuous_code']` into the same
      shape convention.
    - Hash extraction is `sign(continuous_code)` (matches `gen_code_method=sign`).
    - Evaluation reuses our `evaluate_retrieval` from evaluation_siglip2 (binary
      Hamming on the {-1,+1} -> {0,1} packed code), so the metric matches what
      we report for GroundedDNA at `--dna_distance_mode bit2`.

Each concrete baseline (DPSH, HashNet, CSQ, OrthoHash, ...) overrides:
    _get_default_config_dict, _get_config_dict_for_dataset, _get_fixed_config_dict,
    _add_model_specific_args_into_parser, _train_model.
The runner main() at the bottom dispatches to the chosen class via `--method`.

Fair-comparison sketch (DNA vs binary):
    Our DNA hash : 18 codon positions x 4 bases  -> 36 bits effective storage.
                   Distance: 2-bit packed Hamming on 36 bits (`bit2` mode).
    Binary base  : 36 bits straight, signed continuous head -> sign() -> {-1,+1}.
                   Distance: bit Hamming on 36 bits (identical metric).
    Same input features (SigLIP2 visual_global frozen), same train/test/db splits,
    same K=36-bit storage. Only the hashing head + loss differ.
"""

from __future__ import annotations
import argparse
import csv
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


# ============================================================ SigLIP2-feature backbone

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
    """Trainable head: cached SigLIP2 feature [B, D_proj] -> continuous_code [B, bit]."""
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
            print('[base_model] WARNING: finetune=True ignored; cached SigLIP2 '
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
    """Returns cached SigLIP2 features + labels for the requested split.

    Output dict (keys match what each baseline `_train_model` reads):
        'img'   : float32 [D_proj=768]   (the cached SigLIP2 visual_global)
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
                 paired_aug: bool = False, return_index: bool = False):
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
        # cache memmaps
        ids = json.load(open(os.path.join(cache_dir, 'image_ids.json')))
        self.id_to_row = {iid: i for i, iid in enumerate(ids)}
        self.visual_global = np.load(
            os.path.join(cache_dir, 'visual_global.f16.npy'), mmap_mode='r')
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
        self.paths = [f'cifar10:{i}' for i in range(len(rows))]

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

    def __len__(self): return int(self.rows.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = int(self.rows[idx])
        feat = torch.from_numpy(np.asarray(self.visual_global[r], dtype=np.float32))   # [D_proj]
        lbl  = torch.from_numpy(self.labels[idx]).float()                               # [n_class]
        out: Dict[str, Any] = {'img': feat, 'label': lbl, 'image_path': self.paths[idx]}
        if self.paired_aug and self.visual_global_aug0 is not None:
            out['img_tr1'] = torch.from_numpy(np.asarray(self.visual_global_aug0[r], dtype=np.float32))
            out['img_tr2'] = torch.from_numpy(np.asarray(self.visual_global_aug1[r], dtype=np.float32))
        if self.return_index:
            out['idx'] = idx
        return out


def load_dataset(dataset_root: str, dataset_name: str, setting: str,
                 train_transform=None, test_transform=None,
                 load_train: bool = True, load_test: bool = True, load_database: bool = True,
                 return_index: bool = False, return_paired_aug_img: bool = False,
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


def _multi_hot_relevance(qy: np.ndarray, db: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """Pairwise binary relevance = 1 iff Jaccard(query, db) > threshold.
    Mirrors evaluation_siglip2._compute_relevance.
    """
    Lq = qy.astype(np.float32); Ld = db.astype(np.float32)
    inter = Lq @ Ld.T
    union = Lq.sum(axis=1, keepdims=True) + Ld.sum(axis=1, keepdims=True).T - inter
    sim = np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)
    return (sim > threshold).astype(np.uint8)


def _ap_from_sorted_relevance(rel_sorted: np.ndarray) -> float:
    nrel = int(rel_sorted.sum())
    if nrel == 0: return 0.0
    ranks = np.where(rel_sorted == 1)[0] + 1.0
    counts = np.arange(1, nrel + 1, dtype=np.float64)
    return float((counts / ranks).mean())


def evaluate_retrieval_model(query_codes: dict, retrieval_codes: dict,
                             query_labels: np.ndarray, retrieval_labels: np.ndarray,
                             precision_at_k_list=(1, 5, 10, 20, 50, 100, 500, 1000),
                             multi_label_relevance_threshold: float = 0.0,
                             **_unused) -> dict:
    """Replaces lib.evaluation.evaluate_retrieval_model.

    Computes mAP + P@K + R@K via signed binary Hamming distance on `code['B']`.
    `query_codes` / `retrieval_codes` are dicts at minimum containing key 'B'
    (signed-binary codes [N, bit] in {-1, +1}). Other keys ignored.
    """
    qb = np.asarray(query_codes['B'])
    db = np.asarray(retrieval_codes['B'])
    distances = _hamming_distance_signed(qb, db)                   # [Nq, Nd]
    relevance = _multi_hot_relevance(query_labels, retrieval_labels,
                                     threshold=multi_label_relevance_threshold)
    Nq, Nd = distances.shape
    aps = []
    p_at_k = {k: [] for k in precision_at_k_list}
    r_at_k = {k: [] for k in precision_at_k_list}
    for i in tqdm(range(Nq), desc='eval[binary]'):
        d = distances[i]; r = relevance[i]
        order = np.argsort(d, kind='stable')
        rs = r[order]
        aps.append(_ap_from_sorted_relevance(rs))
        nrel = int(r.sum())
        for k in precision_at_k_list:
            kk = min(k, Nd); top = rs[:kk]
            p_at_k[k].append(float(top.mean()) if kk > 0 else 0.0)
            r_at_k[k].append(float(top.sum() / max(nrel, 1)))
    return {
        'mAP':            float(np.mean(aps)) if aps else 0.0,
        'precision_at_k': {int(k): float(np.mean(v)) for k, v in p_at_k.items()},
        'recall_at_k':    {int(k): float(np.mean(v)) for k, v in r_at_k.items()},
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
        g.add_argument('-dr', '--dataset_root', default='./dataset/', type=str)
        g.add_argument('--cache_dir', default=None, type=str,
                       help='Override SigLIP2 feature cache dir (default: ./cache/<dataset>_siglip2/).')

        g = parser.add_argument_group(BOLD + 'Device' + END)
        g.add_argument('-device', '--device', default='cuda:0', type=str)
        g.add_argument('-batch', '--batch_size', default=64, type=int)

        g = parser.add_argument_group(BOLD + 'Train Loop' + END)
        g.add_argument('-me', '--max_epoch', default=60, type=int)
        g.add_argument('-ep', '--eval_period', default=20, type=int)

        g = parser.add_argument_group(BOLD + 'Optimizer' + END)
        g.add_argument('-opt', '--optimizer_name', type=str, default='adam')
        g.add_argument('--sgd_weight_decay', type=float, default=1e-5)
        g.add_argument('--sgd_momentum',     type=float, default=0.99)

        g = parser.add_argument_group(BOLD + 'Scheduler' + END)
        g.add_argument('-lrschd', '--lr_scheduler', type=str, default='none')
        g.add_argument('-lr', '--learning_rate', default=1e-4, type=float)
        g.add_argument('--coslr_T_max', type=int, default=20)
        g.add_argument('--explr_gamma', type=float, default=0.95)

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

    def _init_backbone_with_encoder(self, backbone, encoder_layers, encode_length,
                                    finetune, batch_norm, encoder_type='linear') -> Module:
        # parse encoder_layers spec same as the original
        if encoder_layers in ('version1', 'layer=2+hidden=4096'):
            hidden_nodes = [4096]
        elif encoder_layers in ('version2', 'layer=1', 'none'):
            hidden_nodes = []
        else:
            options = encoder_layers.split('+')
            cfg = {opt.split('=')[0]: int(opt.split('=')[1]) for opt in options}
            hidden_nodes = [cfg['hidden']] * (cfg['layer'] - 1)
        # Auto-detect feature dim from cached visual_global. SigLIP2 has
        # D_proj=768, CLIP-vit-base-patch16 has D_proj=512. Falls back to
        # 768 if the dataset attribute is unavailable for any reason.
        d_in = 768
        if hasattr(self, "trainset") and hasattr(self.trainset, "visual_global"):
            try:
                d_in = int(self.trainset.visual_global.shape[1])
            except Exception:
                pass
        return BackboneWithEncoder(d_in=d_in, bit=int(encode_length),
                                   hidden_nodes=hidden_nodes, batch_norm=batch_norm)

    def _init_optimizer(self, model: Module, name: str, learning_rate: float, **opts) -> Optimizer:
        if name == 'sgd':
            return torch.optim.SGD(model.parameters(), lr=learning_rate,
                                   momentum=opts.get('sgd_momentum', 0.99),
                                   weight_decay=opts.get('sgd_weight_decay', 1e-5))
        if name == 'adam':
            return torch.optim.Adam(model.parameters(), lr=learning_rate)
        raise KeyError(name)

    def _init_scheduler(self, optimizer: Optimizer, name: str, **opts) -> _LRScheduler:
        if name == 'coslr':
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                T_max=int(opts.get('coslr_T_max', 20)))
        if name == 'explr':
            return torch.optim.lr_scheduler.ExponentialLR(optimizer,
                gamma=float(opts.get('explr_gamma', 0.95)))
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
        # apply fixed config last (overrides any default)
        for k, v in self._get_fixed_config_dict().items():
            config[k] = v
        self._check_required_args_not_None(config)
        print(pd.Series(config))
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
        self.trainset, self.testset, self.dbset = load_dataset(
            config['dataset_root'], config['dataset'], config['setting'],
            None, None,
            return_index=bool(config.get('dataset_return_index', False)),
            return_paired_aug_img=bool(config.get('dataset_return_paired_aug_img', False)),
            cache_dir=config.get('cache_dir'),
        )
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
        self.config = config
        self.batch_size = config['batch_size']
        self.eval_period = config['eval_period']

    def start_train_process(self) -> None:
        self._train_model(self.backbone_with_encoder, self.optimizer, self.scheduler,
                          self.trainset, self.testset, self.dbset,
                          self.model_dir, self.result_dir, self.device,
                          self.batch_size, self.eval_period, self.config)

    def start_eval_process(self, model_id: Any, id_type: str = 'epoch') -> None:
        # build query / db loaders
        qy_loader = DataLoader(self.testset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        db_loader = DataLoader(self.dbset,   batch_size=self.batch_size, shuffle=False, num_workers=4)
        model = self.backbone_with_encoder.to(self.device)
        cont_q, bin_q, lbl_q = _extract_codes(model, qy_loader, self.device)
        cont_d, bin_d, lbl_d = _extract_codes(model, db_loader, self.device)
        result = evaluate_retrieval_model(
            query_codes={'B': bin_q, 'C': cont_q},
            retrieval_codes={'B': bin_d, 'C': cont_d},
            query_labels=lbl_q, retrieval_labels=lbl_d,
        )
        # flatten precision_at_k / recall_at_k for csv/console
        flat = {'mAP': result['mAP']}
        for k, v in result['precision_at_k'].items(): flat[f'P@{k}'] = v
        for k, v in result['recall_at_k'   ].items(): flat[f'R@{k}'] = v
        self.logger.save_eval_model_log(model_id, id_type, **flat)
        # persist the full result json (handy for the comparison table)
        out_json = os.path.join(self.result_dir, f'eval_{id_type}_{model_id}.json')
        with open(out_json, 'w') as f:
            json.dump(result, f, indent=2)


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
    raise SystemExit(
        f'unknown method: {method!r}. Available supervised: dpsh hashnet csq orthohash. '
        f'Available unsupervised: cibhash cimon mls3rduh (spq stub is incomplete, see baseline/SPQ.py).'
    )


def main() -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--method', type=str, required=True,
                     help='Supervised: dpsh / hashnet / csq / orthohash. '
                          'Unsupervised: cibhash / cimon / mls3rduh.')
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
