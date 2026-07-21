#!/usr/bin/env python
"""48-bit (24-base DNA) baseline evaluation for the main table, P0 leak-free.

For each (method, dataset) at --bit 48:
  1. E* SELECTION (leak-free): score every stage-1 (90%-train) checkpoint on
     val_query vs opt-train DB in NON-projected 24-base Hamming mAP@R; E* =
     argmax. Test is never touched here. The val split is the imported
     val_split.carve_val_indices(labels, 0.1, 42) -- bit-identical to how the
     stage-1 run carved it (same labels/ratio/seed), so val is genuinely held
     out.
  2. REPORT: extract test-query + official-DB codes from the 100%-train
     checkpoint at E*, convert 48-bit sign codes -> 24 DNA bases, apply the
     mandatory bio-projection (GC frac [gc_min,gc_max] -> count window by length,
     homopolymer <= max_run), and compute bio-projected 24-base mAP@R. Also
     reports the pre-projection number, the projection delta, and DB DNA-unique.

Selection metric = non-projected base mAP@R and report metric = bio-projected
base mAP@R -- symmetric with our model's P0 cells (eval_cell_bioproj.py).

Full-train checkpoints: Flickr25k/MSCOCO were trained 2026-07-15
(result_baseline/260715), NUSWIDE/CIFAR10 here; both resolved by globbing the
tag across date dirs, so no single-root assumption.

Usage:
  python scripts/baseline_48bit_dnaeval.py \
      --methods cibhash cimon mls3rduh \
      --datasets Flickr25k MSCOCO NUSWIDE CIFAR10 \
      --gc_min 0.416 --gc_max 0.584 --out docs/baseline_24base_dnaeval.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from baseline.base_model import (  # noqa: E402
    CachedFeatureDataset, _extract_codes, MAP_AT_R_BY_DATASET,
)
from scripts.baseline_extract_splits import build_model  # noqa: E402
from scripts.eval_baseline_dna_space import (  # noqa: E402
    _two_bit_to_bases, _map_at_r_from_codes,
)
from dna_utils.bio_constraints import batch_project_to_valid  # noqa: E402
from val_split import carve_val_indices  # noqa: E402

SHORT = {'CIFAR10': 'cifar10', 'Flickr25k': 'flickr25k',
         'MSCOCO': 'mscoco', 'NUSWIDE': 'nuswide'}
CACHE = {'CIFAR10': './cache/cifar10_clip',
         'Flickr25k': './cache/flickr25k_clip_v4plus',
         'MSCOCO': './cache/mscoco_clip_v4plus',
         'NUSWIDE': './cache/nuswide_clip'}


def _find_ckpt_dir(tag: str) -> str | None:
    hits = sorted(glob.glob(os.path.join(_REPO, 'params_baseline', '*', tag)))
    return hits[-1] if hits else None


def _bits01(bin_signed: np.ndarray) -> np.ndarray:
    """[N, bit] int8 in {-1,+1} -> {0,1} uint8 (same convention as hash_2bit)."""
    return (bin_signed > 0).astype(np.uint8)


def _project_memoized(bases, gc_min, gc_max, max_run):
    """Bio-project only the UNIQUE base rows (DP per row is expensive; large DBs
    have many repeats -- and baselines/low-uniq collapse to few codes), then map
    back. Returns (projected [N,L] int64, db_pre_compliance float)."""
    uniq, inv = np.unique(np.asarray(bases), axis=0, return_inverse=True)
    out = batch_project_to_valid(uniq, gc_min, gc_max, max_run, progress=False)
    proj = out['projected_codes'].astype(np.int64)[inv]
    pre_compliance = float(np.mean(out['was_valid'][inv]))
    return proj, pre_compliance


def _extract_bases(model, ds, device, bs, nw) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw)
    _, bin_signed, lbl = _extract_codes(model, loader, device)
    bases = _two_bit_to_bases(_bits01(bin_signed))            # [N, bit/2]
    return bases, lbl


def _select_estar(method, dataset, setting, cache_dir, dataset_root,
                  device, bs, nw, ratio, seed):
    s1_dir = _find_ckpt_dir(f'{method}_{SHORT[dataset]}_clip_P0s1_48bit')
    if s1_dir is None:
        raise FileNotFoundError(f'no stage-1 48bit dir for {method}/{dataset}')
    ckpts = sorted(glob.glob(os.path.join(s1_dir, 'epoch_*.pth')))
    if not ckpts:
        raise FileNotFoundError(f'no epoch_*.pth in {s1_dir}')

    full = CachedFeatureDataset(dataset, setting, 'train', dataset_root, cache_dir)
    opt_idx, val_idx, strat = carve_val_indices(np.asarray(full.labels),
                                                ratio=ratio, seed=seed)
    val_ds = CachedFeatureDataset(dataset, setting, 'train', dataset_root, cache_dir)
    val_ds.restrict_to(val_idx)
    opt_ds = CachedFeatureDataset(dataset, setting, 'train', dataset_root, cache_dir)
    opt_ds.restrict_to(opt_idx)
    cutoff = MAP_AT_R_BY_DATASET[dataset]

    per_epoch = {}
    for path in ckpts:
        ep = int(re.search(r'epoch_(\d+)\.pth$', path).group(1))
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        model = build_model(ckpt, device)
        vq_b, vq_l = _extract_bases(model, val_ds, device, bs, nw)
        db_b, db_l = _extract_bases(model, opt_ds, device, bs, nw)
        m = _map_at_r_from_codes(vq_b, db_b, vq_l, db_l, cutoff, device, 256)
        per_epoch[ep] = float(m)
        print(f'  [{method}/{dataset}] s1 ep{ep:03d} val base-mAP@{cutoff}={m:.4f}',
              flush=True)
    estar = max(per_epoch, key=lambda e: per_epoch[e])
    return estar, per_epoch, strat, s1_dir


def _report_at_estar(method, dataset, setting, cache_dir, dataset_root,
                     device, bs, nw, estar, gc_min, gc_max, max_run):
    full_dir = _find_ckpt_dir(f'{method}_{SHORT[dataset]}_clip_48bit_unsup60')
    if full_dir is None:
        raise FileNotFoundError(f'no full 48bit dir for {method}/{dataset}')
    full_ckpt = os.path.join(full_dir, f'epoch_{estar:03d}.pth')
    if not os.path.exists(full_ckpt):
        raise FileNotFoundError(f'missing {full_ckpt}')
    ckpt = torch.load(full_ckpt, map_location='cpu', weights_only=False)
    model = build_model(ckpt, device)

    q_ds = CachedFeatureDataset(dataset, setting, 'test', dataset_root, cache_dir)
    d_ds = CachedFeatureDataset(dataset, setting, 'database', dataset_root, cache_dir)
    q_base, q_lbl = _extract_bases(model, q_ds, device, bs, nw)
    d_base, d_lbl = _extract_bases(model, d_ds, device, bs, nw)
    cutoff = MAP_AT_R_BY_DATASET[dataset]

    pre = _map_at_r_from_codes(q_base, d_base, q_lbl, d_lbl, cutoff, device, 256)
    q_proj, _ = _project_memoized(q_base, gc_min, gc_max, max_run)
    d_proj, db_pre_compliance = _project_memoized(d_base, gc_min, gc_max, max_run)
    post = _map_at_r_from_codes(q_proj, d_proj, q_lbl, d_lbl, cutoff, device, 256)

    dna_uniq_db = len({tuple(r) for r in d_proj.tolist()}) / max(len(d_proj), 1)
    return {
        'best_epoch': int(estar),
        'map_at_R_pre': float(pre),
        'map_at_R_post': float(post),
        'delta_proj': float(post - pre),
        'db_pre_compliance': db_pre_compliance,
        'dna_unique_db_post': float(dna_uniq_db),
        'full_ckpt': full_ckpt,
        'n_query': int(q_base.shape[0]), 'n_db': int(d_base.shape[0]),
        'base_length': int(q_base.shape[1]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--methods', nargs='+',
                    default=['cibhash', 'cimon', 'mls3rduh'])
    ap.add_argument('--datasets', nargs='+',
                    default=['Flickr25k', 'MSCOCO', 'NUSWIDE', 'CIFAR10'])
    ap.add_argument('--setting', default='setting1')
    ap.add_argument('--dataset_root', default=os.path.join(_REPO, 'dataset'))
    ap.add_argument('--gc_min', type=float, default=0.416)   # 24-base -> GC[10,14]
    ap.add_argument('--gc_max', type=float, default=0.584)
    ap.add_argument('--max_run', type=int, default=3)
    ap.add_argument('--ratio', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--batch_size', type=int, default=512)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--out', default='docs/baseline_24base_dnaeval.json')
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    results = []
    cfg = {'bit': 48, 'base_length': 24, 'gc_min_frac': args.gc_min,
           'gc_max_frac': args.gc_max, 'max_run': args.max_run,
           'ratio': args.ratio, 'seed': args.seed}
    print(f"{'method':10s}{'dataset':10s}{'E*':>4s}{'mAP@R pre':>11s}"
          f"{'mAP@R post':>12s}{'d_proj':>9s}{'DNAuniq':>9s}", flush=True)
    for ds in args.datasets:
        cache_dir = CACHE[ds]
        for m in args.methods:
            try:
                estar, per_epoch, strat, s1_dir = _select_estar(
                    m, ds, args.setting, cache_dir, args.dataset_root,
                    args.device, args.batch_size, args.num_workers,
                    args.ratio, args.seed)
                rep = _report_at_estar(
                    m, ds, args.setting, cache_dir, args.dataset_root,
                    args.device, args.batch_size, args.num_workers, estar,
                    args.gc_min, args.gc_max, args.max_run)
                row = {'method': m, 'dataset': ds, 'val_strategy': strat,
                       'stage1_dir': s1_dir, 'per_epoch_val': per_epoch, **rep}
            except Exception as ex:  # noqa: BLE001
                row = {'method': m, 'dataset': ds, 'error': repr(ex)}
                print(f'  !! {m}/{ds} FAILED: {ex}', flush=True)
            results.append(row)
            if 'error' not in row:
                print(f"{m:10s}{ds:10s}{row['best_epoch']:>4d}"
                      f"{row['map_at_R_pre']:>11.4f}{row['map_at_R_post']:>12.4f}"
                      f"{row['delta_proj']:>+9.4f}{row['dna_unique_db_post']:>9.4f}",
                      flush=True)
            with open(args.out, 'w') as f:
                json.dump({'config': cfg, 'cells': results}, f, indent=2)
    print(f'\nwrote {args.out}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
