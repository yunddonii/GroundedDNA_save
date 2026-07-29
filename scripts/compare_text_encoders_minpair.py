#!/usr/bin/env python
"""Do any text encoder actually separate our minimal pairs?

Diagnosis (2026-07-29): with the frozen CLIP text encoder, a caption and its
own single-atom foil sit at cosine 0.94-0.98, while *unrelated* captions sit at
0.93-0.95 — a separation of only 0.003-0.029. That caps the held-out
minimal-pair metric near chance no matter what the visual side does. Caption
length is NOT the cause (Spearman(len, cos) = +0.004).

This script asks the remaining legitimate question: is that a property of CLIP,
or of sentence embeddings in general? It re-encodes the SAME held-out foil pairs
with several encoders and reports, per encoder:

    own      = cos(factual, its own single-atom foil)      (want LOW)
    unrel    = cos(factual, an unrelated image's caption)  (reference)
    sep      = own - unrel                                 (want LARGE)
    margin   = fraction of pairs with cos(f, own foil) > cos(f, unrelated)

`sep` is the quantity that matters: how much closer a minimal pair is than two
unrelated sentences. If it stays ~0.02-0.05 everywhere, the limitation is a
property of sentence embeddings and belongs in the paper as a measured boundary.

NOTE: this is a diagnostic only. Nothing here is fitted to the foils; we merely
encode them. Fitting a projection to factual-foil differences would be
teaching-to-the-test, since the foils come from our own edit templates.

Usage:
  python scripts/compare_text_encoders_minpair.py --n 2000
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

SLOTS = ['C_global', 'C_primary_object', 'C_secondary_object',
         'C_activity_or_relation', 'C_color_texture', 'C_scene_type']
FOIL_CACHE = 'cache/mscoco_clip_v5b_tokens_foils'
QWEN = 'cache/mscoco_qwen3_v5b_trainset.jsonl'


def load_pairs(n: int, seed: int = 0):
    cap = {}
    for line in open(QWEN):
        r = json.loads(line)
        cap[r.get('image_id') or r.get('image_path')] = r.get('codebook_texts', {})
    fact, foil, slot = [], [], []
    for line in open(os.path.join(FOIL_CACHE, 'text_foil_edits.jsonl')):
        e = json.loads(line)
        ct = cap.get(e['image_id'])
        if not ct:
            continue
        for si, sn in enumerate(SLOTS):
            ed = e['edits'].get(sn)
            if not (ed and ed.get('valid')):
                continue
            t = str(ct.get(sn, ''))
            a, b = ed['source_span']
            if not t or a >= len(t):
                continue
            fact.append(t)
            foil.append(t[:a] + ed['target_atom'] + t[b:])
            slot.append(si)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(fact), min(n, len(fact)), replace=False)
    return ([fact[i] for i in idx], [foil[i] for i in idx],
            np.array([slot[i] for i in idx]))


def _norm(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def enc_clip(name, texts, device, bs=256):
    from transformers import CLIPModel, CLIPTokenizer
    m = CLIPModel.from_pretrained(name).to(device).eval()
    tk = CLIPTokenizer.from_pretrained(name)
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            b = tk(texts[i:i + bs], padding=True, truncation=True,
                   max_length=77, return_tensors='pt').to(device)
            r = m.get_text_features(**b)
            if not torch.is_tensor(r):
                r = getattr(r, 'text_embeds', None) or getattr(r, 'pooler_output')
            out.append(r.cpu().numpy())
    del m
    torch.cuda.empty_cache()
    return np.concatenate(out)


def enc_siglip(name, texts, device, bs=256):
    from transformers import AutoModel, AutoTokenizer
    m = AutoModel.from_pretrained(name).to(device).eval()
    tk = AutoTokenizer.from_pretrained(name)
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            b = tk(texts[i:i + bs], padding='max_length', truncation=True,
                   max_length=64, return_tensors='pt').to(device)
            r = m.get_text_features(**b)
            if not torch.is_tensor(r):
                r = getattr(r, 'text_embeds', None) or getattr(r, 'pooler_output')
            out.append(r.cpu().numpy())
    del m
    torch.cuda.empty_cache()
    return np.concatenate(out)


def enc_meanpool(name, texts, device, bs=128):
    """Generic masked-mean-pooled encoder (SBERT-style)."""
    from transformers import AutoModel, AutoTokenizer
    m = AutoModel.from_pretrained(name).to(device).eval()
    tk = AutoTokenizer.from_pretrained(name)
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            b = tk(texts[i:i + bs], padding=True, truncation=True,
                   max_length=128, return_tensors='pt').to(device)
            h = m(**b).last_hidden_state
            mask = b['attention_mask'].unsqueeze(-1).float()
            out.append(((h * mask).sum(1) / mask.sum(1).clamp_min(1e-6)).cpu().numpy())
    del m
    torch.cuda.empty_cache()
    return np.concatenate(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=2000)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--out', default='docs/text_encoder_minpair_comparison.json')
    a = ap.parse_args()

    fact, foil, slot = load_pairs(a.n)
    print(f"pairs={len(fact)}  (MSCOCO V5b held-out foils)")
    rng = np.random.default_rng(1)
    perm = rng.permutation(len(fact))

    encoders = [
        ('CLIP-B/16 (current)', enc_clip,     'openai/clip-vit-base-patch16'),
        ('SigLIP2-base',        enc_siglip,   'google/siglip2-base-patch16-224'),
        ('FG-CLIP-base',        enc_clip,     'qihoo360/fg-clip-base'),
        ('all-mpnet-base-v2',   enc_meanpool, 'sentence-transformers/all-mpnet-base-v2'),
        ('all-MiniLM-L6-v2',    enc_meanpool, 'sentence-transformers/all-MiniLM-L6-v2'),
    ]
    rows = []
    print(f"\n{'encoder':24s}{'own-foil cos':>14s}{'unrelated cos':>15s}"
          f"{'separation':>12s}{'foil-closer%':>14s}")
    for label, fn, name in encoders:
        try:
            F = _norm(fn(name, fact, a.device).astype(np.float32))
            L = _norm(fn(name, foil, a.device).astype(np.float32))
        except Exception as e:                                   # noqa: BLE001
            print(f"{label:24s}  SKIPPED ({type(e).__name__}: {str(e)[:60]})")
            continue
        own = (F * L).sum(-1)
        unrel = (F * F[perm]).sum(-1)
        sep = float(own.mean() - unrel.mean())
        closer = float((own > unrel).mean())
        rows.append({'encoder': label, 'model': name, 'own_foil_cos': float(own.mean()),
                     'unrelated_cos': float(unrel.mean()), 'separation': sep,
                     'foil_closer_frac': closer, 'n': int(len(F))})
        print(f"{label:24s}{own.mean():>14.4f}{unrel.mean():>15.4f}"
              f"{sep:>12.4f}{closer*100:>13.1f}%")

    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    json.dump({'note': 'diagnostic only; nothing fitted to the foils',
               'dataset': 'MSCOCO V5b held-out foils', 'cells': rows},
              open(a.out, 'w'), indent=2)
    print(f"\nwrote {a.out}")
    print("[read] separation = own_foil_cos - unrelated_cos. Larger = the encoder "
          "places a minimal pair much closer than two unrelated captions, i.e. it "
          "encodes the atom substitution.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
