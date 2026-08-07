#!/usr/bin/env python
"""PER-IMAGE: does any slot receive zero visual tokens, making its codon content-free?

Every earlier dead-slot measurement averaged the transported mass over the
batch, which cannot separate two very different situations for a slot at 0.9 %
mean share:

    (a) it receives a thin slice of mass on EVERY image  -> pooled feature is a
        real, image-varying direction; the codon means something.
    (b) it receives EXACTLY zero on most images          -> `denom.clamp_min(1e-12)`
        (semantic_router.py:498) makes the pooled feature the ZERO VECTOR, so the
        codon is a constant filler carrying no information about that image.

(b) would be a design defect: the slot's three bases would be reserved capacity
spent on a symbol that says nothing. The adaptive top-p mask makes it possible,
because `keep_sorted[...,0] = True` only guarantees that each PATCH keeps its
own rank-1 slot — nothing guarantees a given SLOT is anyone's rank-1.

Reports per (image, slot):
  n_tokens  number of patches with strictly positive routing weight
  mass      column mass after masking
and aggregates the fraction of images where a slot is completely empty.
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
sys.path.insert(0, os.path.join(_REPO, "scripts"))

SLOTS = ["global", "primary_object", "secondary_object",
         "activity_relation", "color_texture", "scene_type"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", required=True)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--epoch", type=int, default=None)
    ap.add_argument("--sample_seed", type=int, default=1234)
    ap.add_argument("--split", default="train", choices=("train", "test"))
    ap.add_argument("--disable_adaptive_topp", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from regen_viz_routing import _parse_args_txt
    args = _parse_args_txt(os.path.join(a.result_dir, "args.txt"))
    if a.disable_adaptive_topp:
        args.routing_adaptive_topp = False
    from model_siglip2 import SigLIP2SemanticOTModel
    from dataloaders import load_dataset
    from dna_utils.visualization import _forward_text_routed

    model = SigLIP2SemanticOTModel(args).to(a.device).eval()
    sd = torch.load(os.path.join(a.result_dir, "model_state_dict.pth"),
                    map_location="cpu", weights_only=False)
    model.load_state_dict(sd, strict=False)
    if a.epoch is not None:
        model.set_current_epoch(a.epoch)

    want_train = (a.split == "train")
    tr, te, _ = load_dataset(
        getattr(args, "dataset_dir", "dataset"), args.dataset,
        setting=getattr(args, "setting", "setting1"),
        train_transform=None, test_transform=None,
        load_train=want_train, load_database=False, load_test=not want_train,
        return_index=True,
        qwen_text_cache_path=getattr(args, "qwen_text_cache_path", None),
        siglip2_feature_cache_dir=getattr(args, "siglip2_feature_cache_dir", None))
    ds = tr if want_train else te

    g = np.random.default_rng(a.sample_seed)
    order = g.permutation(len(ds))[:min(a.n, len(ds))].tolist()

    ntok, mass, codons = [], [], []
    with torch.no_grad():
        for i0 in range(0, len(order), 32):
            samples = [ds[i] for i in order[i0:i0 + 32]]
            batch = {}
            for k in samples[0]:
                vals = [s_[k] for s_ in samples]
                if torch.is_tensor(vals[0]):
                    batch[k] = torch.stack(vals)
                elif isinstance(vals[0], np.ndarray):
                    batch[k] = torch.from_numpy(np.stack(vals))
                else:
                    batch[k] = vals
            o = _forward_text_routed(model, batch, a.device)
            P = o["routing_matrix"].float()
            ntok.append((P > 0).sum(dim=1).cpu().numpy())      # [B, M]
            mass.append(P.sum(dim=1).cpu().numpy())            # [B, M]
            # codon symbol per (image, slot), to test whether the images an
            # empty slot cannot see all collapse onto ONE constant filler codon
            for k in ("base_indices", "dna_base_indices", "bases", "codon_indices"):
                if k in o and o[k] is not None:
                    bi = o[k]
                    bi = bi.argmax(-1) if bi.dim() == 3 else bi
                    codons.append(bi.long().cpu().numpy()); break
    ntok = np.concatenate(ntok); mass = np.concatenate(mass)
    codons = np.concatenate(codons) if codons else None
    B, M = ntok.shape
    N_patch = None

    print(f"{args.dataset} [{a.split}] images={B} slots={M} "
          f"topp={'OFF' if a.disable_adaptive_topp else 'ON'} epoch={a.epoch}")
    print(f"  {'slot':20s}{'empty imgs %':>14s}{'n_tok med':>11s}"
          f"{'n_tok p10':>11s}{'mass med':>11s}{'mass<1e-6 %':>13s}")
    rows = {}
    for m in range(M):
        empty = float((ntok[:, m] == 0).mean() * 100)
        tiny = float((mass[:, m] < 1e-6).mean() * 100)
        name = SLOTS[m] if m < len(SLOTS) else str(m)
        print(f"  {name:20s}{empty:13.2f}%{np.median(ntok[:, m]):11.1f}"
              f"{np.percentile(ntok[:, m], 10):11.1f}"
              f"{np.median(mass[:, m]):11.5f}{tiny:12.2f}%")
        rows[name] = {"empty_image_pct": round(empty, 4),
                      "mass_below_1e-6_pct": round(tiny, 4),
                      "n_tokens_median": float(np.median(ntok[:, m])),
                      "n_tokens_p10": float(np.percentile(ntok[:, m], 10)),
                      "n_tokens_min": int(ntok[:, m].min()),
                      "mass_median": float(np.median(mass[:, m])),
                      "mass_min": float(mass[:, m].min())}
    print(f"  (patches per image = {int(ntok.sum(axis=1).max())} at most across slots)")

    if codons is not None and codons.shape[1] >= 3 * M:
        print(f"\n  constant-filler check: among images a slot CANNOT see, do the "
              f"codons collapse to one symbol?")
        print(f"  {'slot':20s}{'empty n':>9s}{'codons(empty)':>15s}{'top1%':>8s}"
              f"{'codons(seen)':>14s}{'top1%':>8s}")
        for m in range(M):
            e = ntok[:, m] == 0
            if e.sum() < 5:
                continue
            c = codons[:, 3*m]*16 + codons[:, 3*m+1]*4 + codons[:, 3*m+2]
            ue, ce = np.unique(c[e], return_counts=True)
            us, cs = np.unique(c[~e], return_counts=True)
            name = SLOTS[m] if m < len(SLOTS) else str(m)
            print(f"  {name:20s}{int(e.sum()):9d}{len(ue):15d}"
                  f"{ce.max()/ce.sum()*100:7.1f}%{len(us):14d}"
                  f"{cs.max()/cs.sum()*100:7.1f}%")
            rows[name]["codons_when_empty"] = int(len(ue))
            rows[name]["top1_codon_when_empty_pct"] = round(float(ce.max()/ce.sum()*100), 2)
            rows[name]["codons_when_seen"] = int(len(us))

    out = {"dataset": args.dataset, "dir": a.result_dir, "split": a.split,
           "epoch": a.epoch, "images": int(B),
           "adaptive_topp_disabled": bool(a.disable_adaptive_topp), "slots": rows}
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=2)
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
