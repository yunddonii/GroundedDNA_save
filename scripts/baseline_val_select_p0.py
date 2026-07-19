"""P0 stage 2 for the baselines: pick E* on held-out val, report test at E*.

Stage 1 (`scripts/run_baselines_p0_stage1.sh`) retrained every baseline on the
optimization-train 90% and dumped one checkpoint every 5 epochs. Those runs'
`eval_epoch_*.json` files contain ONLY test metrics, so they cannot be used to
select an epoch without leaking test into the selection. This script closes that
gap: it re-scores every stage-1 checkpoint on the held-out val rows
(val_query vs opt-train DB) and takes E* = argmax val mAP@R.

The stage-1 model is never reported. The reported number is the EXISTING
100%-train run's `eval_epoch_{E*}.json` test mAP@R -- exactly mirroring our own
stage 2 ("refit on 100% of train, stop at the val-selected epoch").

Protocol invariants (must stay identical to train_siglip2.py):
  * the split comes from `val_split.carve_val_indices(labels, 0.1, 42)` -- imported,
    never reimplemented;
  * mid-eval is val_query vs opt-train DB, disjoint splits, no self-match removal;
  * the metric is mAP@R with the dataset cutoff in `MAP_AT_R_BY_DATASET`
    (CIFAR10@1000, others@5000), CalcTopMap convention;
  * codes are `sign(encoder(cached_feature))`, via `_extract_codes` -- the same
    function `DeepHashBase.start_eval_process` uses for the test numbers.

Usage:
    python scripts/baseline_val_select_p0.py --methods cibhash --datasets Flickr25k \
        --out docs/baseline_p0_stage2_partial/cibhash_Flickr25k.json
    python scripts/baseline_val_select_p0.py --merge 'docs/baseline_p0_stage2_partial/*.json' \
        --out docs/baseline_p0_stage2.json --markdown docs/baseline_p0_stage2.md
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
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from baseline.base_model import (  # noqa: E402
    MAP_AT_R_BY_DATASET,
    BackboneWithEncoder,
    CachedFeatureDataset,
    _extract_codes,
    evaluate_retrieval_model,
)
from val_split import carve_val_indices  # noqa: E402

METHODS = ('cibhash', 'cimon', 'mls3rduh')
DATASETS = ('Flickr25k', 'MSCOCO', 'NUSWIDE', 'CIFAR10')
SHORT = {'CIFAR10': 'cifar10', 'Flickr25k': 'flickr25k',
         'MSCOCO': 'mscoco', 'NUSWIDE': 'nuswide'}

STAGE1_ROOT = os.path.join(_REPO, 'params_baseline', '260718')
FULL_RESULT_ROOT = os.path.join(_REPO, 'result_baseline', '260714')


def stage1_dir(method: str, dataset: str) -> str:
    return os.path.join(STAGE1_ROOT, f'{method}_{SHORT[dataset]}_clip_P0s1_unsup60')


def full_result_dir(method: str, dataset: str) -> str:
    return os.path.join(FULL_RESULT_ROOT, f'{method}_{SHORT[dataset]}_clip_mapr_unsup60')


def _hidden_nodes(spec: str):
    """Mirror DeepHashBase._init_backbone_with_encoder's encoder_layers parsing."""
    if spec in ('version1', 'layer=2+hidden=4096'):
        return [4096]
    if spec in ('version2', 'layer=1', 'none'):
        return []
    cfg = {o.split('=')[0]: int(o.split('=')[1]) for o in spec.split('+')}
    return [cfg['hidden']] * (cfg['layer'] - 1)


def evaluate_pair(method: str, dataset: str, device: str) -> dict:
    ckpt_dir = stage1_dir(method, dataset)
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, 'epoch_*.pth')))
    if not ckpts:
        raise FileNotFoundError(f'no stage-1 checkpoints in {ckpt_dir}')

    cfg = torch.load(ckpts[0], map_location='cpu', weights_only=False)['config']
    assert cfg['dataset'] == dataset, f"{ckpts[0]} is for {cfg['dataset']}, not {dataset}"
    # Sanity: the stage-1 run must actually have held rows out.
    assert float(cfg.get('val_split_ratio', 0.0)) > 0.0, \
        f'{ckpt_dir} was NOT trained with --val_split_ratio > 0; E* would leak.'
    ratio = float(cfg['val_split_ratio'])
    seed = int(cfg.get('val_split_seed', 42))

    # Paths in the stored config are relative to the repo root (the runs were
    # launched from there), so resolve them against it rather than the cwd.
    dataset_root = os.path.join(_REPO, cfg['dataset_root'])
    cache_dir = os.path.join(_REPO, cfg['cache_dir'])

    def _train_split():
        return CachedFeatureDataset(dataset, cfg['setting'], 'train',
                                    dataset_root, cache_dir)

    full = _train_split()
    # Bit-identical to stage 1 (base_model.init_experiment) and to our trainer:
    # for CIFAR10 the baseline loader hands carve_val_indices a one-hot [N, 10]
    # matrix while train_siglip2.py hands it the [N] integer `targets`;
    # carve_val_indices collapses the one-hot with argmax, so both paths take the
    # identical branch on identical class ids and yield the identical rows.
    opt_idx, val_idx, strat = carve_val_indices(np.asarray(full.labels),
                                                ratio=ratio, seed=seed)

    val_ds = _train_split(); val_ds.restrict_to(val_idx)
    opt_ds = _train_split(); opt_ds.restrict_to(opt_idx)

    d_in = int(full.visual_global.shape[1])
    model = BackboneWithEncoder(
        d_in=d_in, bit=int(cfg['bit']),
        hidden_nodes=_hidden_nodes(str(cfg['encoder_layers'])),
        batch_norm=bool(cfg['batch_norm']),
    ).to(device)

    bs = int(cfg.get('batch_size', 64))
    q_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=4)
    d_loader = DataLoader(opt_ds, batch_size=bs, shuffle=False, num_workers=4)
    map_r = MAP_AT_R_BY_DATASET[dataset]

    per_epoch = {}
    for path in ckpts:
        ep = int(re.search(r'epoch_(\d+)\.pth$', path).group(1))
        sd = torch.load(path, map_location='cpu', weights_only=False)
        model.encoder_layers.load_state_dict(sd['encoder_layers'])
        cont_q, bin_q, lbl_q = _extract_codes(model, q_loader, device)
        cont_d, bin_d, lbl_d = _extract_codes(model, d_loader, device)
        res = evaluate_retrieval_model(
            query_codes={'B': bin_q, 'C': cont_q},
            retrieval_codes={'B': bin_d, 'C': cont_d},
            query_labels=lbl_q, retrieval_labels=lbl_d,
            precision_at_k_list=(1, 10, 100),
            map_at_r=map_r,
        )
        per_epoch[ep] = {'val_mAP_at_R': res['mAP_at_R'], 'val_mAP': res['mAP'],
                         'val_P_at_1': res['precision_at_k'][1]}
        print(f'[{method}/{dataset}] epoch {ep:03d}  '
              f'val mAP@{map_r}={res["mAP_at_R"]:.4f}  val mAP={res["mAP"]:.4f}',
              flush=True)

    best_ep = max(per_epoch, key=lambda e: per_epoch[e]['val_mAP_at_R'])

    # --- the reported number: 100%-train run's test metrics at E* ---
    test_json = os.path.join(full_result_dir(method, dataset),
                             f'eval_epoch_{best_ep:03d}.json')
    if os.path.exists(test_json):
        tj = json.load(open(test_json))
        test = {'test_mAP_at_R': tj.get('mAP_at_R'), 'test_mAP': tj.get('mAP'),
                'test_mAP_R_cutoff': tj.get('mAP_R_cutoff'),
                'test_P_at_1': tj.get('precision_at_k', {}).get('1'),
                'source': os.path.relpath(test_json, _REPO)}
    else:
        test = {'test_mAP_at_R': None, 'MISSING': os.path.relpath(test_json, _REPO)}
        print(f'!! MISSING 100%-train eval for E*={best_ep}: {test_json}', flush=True)

    return {
        'method': method, 'dataset': dataset,
        'val_split': {'ratio': ratio, 'seed': seed, 'strategy': strat,
                      'n_train': int(len(full)), 'n_opt_train': int(len(opt_ds)),
                      'n_val_query': int(len(val_ds)),
                      'val_idx_sha': int(np.sum(val_idx)),  # cheap cross-run identity check
                      'val_idx_head': val_idx[:5].tolist()},
        'map_at_r_cutoff': map_r,
        'per_epoch_val': {str(k): v for k, v in sorted(per_epoch.items())},
        'best_epoch': int(best_ep),
        'best_val_mAP_at_R': per_epoch[best_ep]['val_mAP_at_R'],
        **test,
    }


def write_markdown(merged: dict, path: str) -> None:
    methods = [m for m in METHODS if any(k.endswith(f'|{m}') or k.startswith(f'{m}|')
                                         for k in merged)]
    methods = list(METHODS)
    lines = [
        '# P0 stage 2 -- baselines (36-bit, CLIP features, unsupervised)',
        '',
        'E* selected on held-out val (val_query vs opt-train DB, 10% carve, seed 42).',
        'Cell = **test mAP@R of the 100%-train run at that E\\***; (E*) in parentheses.',
        '',
        '| Dataset | ' + ' | '.join(m.upper() for m in methods) + ' |',
        '|---|' + '---|' * len(methods),
    ]
    for d in DATASETS:
        cells = []
        for m in methods:
            r = merged.get(f'{m}|{d}')
            if r is None:
                cells.append('n/a')
            elif r.get('test_mAP_at_R') is None:
                cells.append(f"MISSING (E*={r['best_epoch']})")
            else:
                cells.append(f"{r['test_mAP_at_R']:.4f} (E*={r['best_epoch']})")
        lines.append(f'| {d} | ' + ' | '.join(cells) + ' |')
    lines += ['', '## Selected epochs (E*) and val mAP@R', '',
              '| Method | Dataset | E* | val mAP@R | test mAP@R |', '|---|---|---|---|---|']
    for m in methods:
        for d in DATASETS:
            r = merged.get(f'{m}|{d}')
            if r is None:
                continue
            t = r.get('test_mAP_at_R')
            lines.append(f"| {m} | {d} | {r['best_epoch']} | "
                         f"{r['best_val_mAP_at_R']:.4f} | "
                         f"{'MISSING' if t is None else f'{t:.4f}'} |")
    lines += [
        '',
        '## Protocol notes',
        '',
        '* E* is selected **only** on val_query vs opt-train DB. No test metric '
        'enters the selection.',
        '* The val DB is the opt-train split, which is smaller than the official '
        'database. For Flickr25k (val DB = 4500 rows) the R=5000 cutoff never '
        'binds, so the selection metric degenerates to full mAP there. This is '
        'symmetric with our own model (same val DB), but it means the *selection* '
        'metric and the *reported* metric are not the identical statistic on '
        'Flickr25k.',
        '* Absolute val mAP@R is not comparable to test mAP@R: different query '
        'set, much smaller DB, and a different effective cutoff.',
    ]
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--methods', nargs='+', default=list(METHODS))
    p.add_argument('--datasets', nargs='+', default=list(DATASETS))
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--out', required=True)
    p.add_argument('--merge', default=None,
                   help='glob of per-pair jsons; merge into --out instead of evaluating')
    p.add_argument('--markdown', default=None)
    a = p.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or '.', exist_ok=True)

    if a.merge:
        merged = {}
        for f in sorted(glob.glob(a.merge)):
            r = json.load(open(f))
            merged[f"{r['method']}|{r['dataset']}"] = r
        json.dump(merged, open(a.out, 'w'), indent=2)
        print(f'merged {len(merged)} pairs -> {a.out}')
        if a.markdown:
            write_markdown(merged, a.markdown)
            print(f'markdown -> {a.markdown}')
        return 0

    device = a.device if torch.cuda.is_available() else 'cpu'
    results = {}
    for m in a.methods:
        for d in a.datasets:
            results[f'{m}|{d}'] = evaluate_pair(m, d, device)
    json.dump(results if len(results) > 1 else list(results.values())[0],
              open(a.out, 'w'), indent=2)
    print(f'wrote {a.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
