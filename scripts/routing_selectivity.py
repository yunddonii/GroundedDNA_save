#!/usr/bin/env python
"""Does routing SELECT patches, or does it average them?

Chain established so far (rho = Spearman(code/feature distance, label distance),
image pairs, train rows):

    dataset      CLIP CLS   patch-mean    TEXT     z
    Flickr25k       0.183        0.021    0.226    0.338
    NUS-WIDE        0.038        0.063    0.091    0.332
    MSCOCO          0.131       -0.115    0.069    0.080

Mean-pooled patch tokens are a poor semantic representation everywhere, and on
MSCOCO they are ANTI-correlated with label similarity (-0.115): COCO images carry
~2.9 labels over diverse backgrounds, so averaging 196 patches yields a
scene-texture vector whose similarity is driven by background, not objects.

Hypothesis: z inherits patch-mean's defect exactly when routing is diffuse.
Sharp routing (few patches per slot) escapes it; near-uniform routing does not.

Measured per slot, from the model's own `routing_matrix` [B, N, 6]:
    eff_k         participation ratio 1 / sum_n p_n^2 over the patch axis
                  (= effective number of patches the slot actually reads;
                  N = uniform, 1 = single patch)
    eff_k_frac    eff_k / N -- comparable across datasets
    top1_mass     weight on the single highest-scoring patch
    top10_mass    weight on the top 10 patches
    slot_overlap  mean cosine between two slots' routing distributions
                  (1 = every slot reads the same patches)

If MSCOCO's eff_k_frac is much higher than Flickr/NUS-WIDE, routing is diffuse
and z ~ patch-mean ~ anti-correlated -- the fix is routing sharpening
(adaptive top-p range, Sinkhorn epsilon). If MSCOCO is already sharp, patch
selection is not the mechanism and the hypothesis is refuted.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config, set_random_seed
from dataloaders import load_dataset
from dna_utils import get_transform
from extraction_siglip2 import _find_model_checkpoint, _resume_args_flat_or_legacy
from model_siglip2 import SigLIP2SemanticOTModel

N_SLOTS = 6


def main() -> None:
    set_random_seed(42)
    args = Config()
    _resume_args_flat_or_legacy(args)
    n_img = int(os.environ.get("R_MAX", "2048"))

    model = SigLIP2SemanticOTModel(args).to(args.device)
    model.load_state_dict(
        torch.load(_find_model_checkpoint(args.save_model_state_path),
                   map_location=args.device), strict=False)
    model.eval()

    tf = get_transform("test")
    _, _, dbs = load_dataset(
        args.dataset_dir, args.dataset, setting=args.setting,
        train_transform=tf, test_transform=tf,
        load_train=False, load_database=True, load_test=False, return_index=True,
        qwen_text_cache_path=getattr(args, "qwen_text_cache_path", None),
        siglip2_feature_cache_dir=getattr(args, "siglip2_feature_cache_dir", None))

    # contiguous head is fine here: routing statistics are per-image and do not
    # depend on the label mix (unlike the codeword-prototype analyses)
    subset = torch.utils.data.Subset(dbs, list(range(min(n_img, len(dbs)))))
    loader = torch.utils.data.DataLoader(subset, batch_size=128, num_workers=4)

    eff_k, top1, top10, overlap, silent = [], [], [], [], []
    with torch.no_grad():
        for b in loader:
            kw = dict(pixel_values=None, part_input_ids=None,
                      part_attention_mask=None, return_routing=True,
                      cached_text_part_raw=None, cached_has_text=None)
            cvt = b.get("cached_visual_tokens_raw")
            if cvt is not None:
                kw["cached_visual_tokens_raw"] = cvt.to(args.device)
                cvg = b.get("cached_visual_global")
                kw["cached_visual_global"] = cvg.to(args.device) if cvg is not None else None
            else:
                from extraction_siglip2 import _get_pixel_values
                kw["pixel_values"] = _get_pixel_values(b).to(args.device)
            out = model(**kw)
            R = out["routing_matrix"]                      # [B, N, 6]
            if R is None:
                raise SystemExit("routing_matrix is None for this run/config")
            # Slot 0 is excluded throughout: under c_global_source=siglip2_global
            # the global slot is built from the CLS feature and its
            # routing_matrix column carries no functional role (it reads as
            # uniform, eff_k = N exactly).
            R = R.float().clamp_min(0.0)[:, :, 1:]              # [B, N, 5]
            mass = R.sum(dim=1)                                  # [B, 5]
            live = mass > 1e-8       # adaptive top-p can zero a slot entirely
            P = R / mass.clamp_min(1e-9).unsqueeze(1)
            k = 1.0 / (P ** 2).sum(dim=1).clamp_min(1e-12)       # [B, 5]
            eff_k.append(torch.where(live, k, torch.nan).cpu().numpy())
            s, _ = P.sort(dim=1, descending=True)
            top1.append(torch.where(live, s[:, 0, :], torch.nan).cpu().numpy())
            top10.append(torch.where(live, s[:, :10, :].sum(dim=1), torch.nan).cpu().numpy())
            silent.append((~live).float().cpu().numpy())
            Pn = P / P.norm(dim=1, keepdim=True).clamp_min(1e-9)
            G = torch.einsum("bnm,bnk->bmk", Pn, Pn)
            iu = torch.triu_indices(N_SLOTS - 1, N_SLOTS - 1, offset=1)
            overlap.append(G[:, iu[0], iu[1]].mean(dim=1).cpu().numpy())

    eff_k = np.concatenate(eff_k); top1 = np.concatenate(top1)
    top10 = np.concatenate(top10); overlap = np.concatenate(overlap)
    silent = np.concatenate(silent)
    N = int(R.shape[1])

    res = {
        "dir": os.path.basename(str(args.save_result_path).rstrip("/")),
        "n_images": int(len(eff_k)), "n_patches": N,
        "note": "slot 0 (global) excluded; stats over the 5 routed local slots",
        "eff_k_per_slot": np.nanmean(eff_k, 0).tolist(),
        "eff_k_mean": float(np.nanmean(eff_k)),
        "eff_k_frac_mean": float(np.nanmean(eff_k) / N),
        "top1_mass_mean": float(np.nanmean(top1)),
        "top10_mass_mean": float(np.nanmean(top10)),
        "silent_slot_frac": float(silent.mean()),
        "slot_routing_overlap": float(np.nanmean(overlap)),
    }
    print(f"[routing] {res['dir'][:52]}")
    print(f"  patches N={N}  images={res['n_images']}")
    print(f"  eff_k mean={res['eff_k_mean']:.1f}  ({res['eff_k_frac_mean']*100:.1f}% of patches)"
          f"   silent-slot frac={res['silent_slot_frac']:.3f}")
    print(f"  per-slot eff_k: " + " ".join(f"{v:.1f}" for v in res["eff_k_per_slot"]))
    print(f"  top1 mass={res['top1_mass_mean']:.4f}  top10 mass={res['top10_mass_mean']:.4f}")
    print(f"  slot routing overlap={res['slot_routing_overlap']:.3f}")

    dest = os.environ.get("R_OUT")
    if dest:
        prev = json.load(open(dest)) if os.path.exists(dest) else []
        prev.append(res)
        with open(dest, "w") as f:
            json.dump(prev, f, indent=2)
        print(f"  -> {dest}")


if __name__ == "__main__":
    main()
