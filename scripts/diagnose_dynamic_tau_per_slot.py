#!/usr/bin/env python
"""Does the adaptive (dynamic) temperature discriminate between slots?

The adaptive-tau in the contrastive losses is NOT the OT router's slot-marginal
strength; it is a per-pair temperature on the per-slot NT-Xent / CIBHash terms
(loss_siglip2.py:1429):

    cos_tt = cos( text_part_raw[i, m] , text_part_raw[j, m] )     # [B, B]
    tau_ij = T * (1 + alpha * cos_tt)
    sim    = (v_i . v_j) / tau_ij

so a slot whose captions are near-identical across images (cos_tt -> 1) is
trained at tau up to T*(1+alpha), i.e. a HOTTER temperature and therefore a
weaker contrastive gradient, while a slot with diverse captions (cos_tt -> 0)
keeps tau ~ T.

That matters for the dead-slot question: on CIFAR-10 32x32 there is no real
"secondary object" and no real "scene", so those captions should be nearly
constant -- and if they are, adaptive-tau softens the loss on exactly the two
slots that die. This measures cos_tt per slot to test that.

It cannot show causation (the router, not the loss, decides transported mass),
only whether the amplification path is open.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts"))

SLOTS = ["global", "primary_object", "secondary_object",
         "activity_relation", "color_texture", "scene_type"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", required=True)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--sample_seed", type=int, default=1234)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from regen_viz_routing import _parse_args_txt
    from dataloaders import load_dataset
    args = _parse_args_txt(os.path.join(a.result_dir, "args.txt"))
    ds_name = args.dataset

    trainset, _, _ = load_dataset(
        getattr(args, "dataset_dir", "dataset"), ds_name,
        setting=getattr(args, "setting", "setting1"),
        train_transform=None, test_transform=None,
        load_train=True, load_database=False, load_test=False, return_index=True,
        qwen_text_cache_path=getattr(args, "qwen_text_cache_path", None),
        siglip2_feature_cache_dir=getattr(args, "siglip2_feature_cache_dir", None))

    g = np.random.default_rng(a.sample_seed)
    idx = g.permutation(len(trainset))[:min(a.n, len(trainset))].tolist()
    tps = []
    for i in idx:
        s = trainset[i]
        tp = s.get("cached_text_part_raw")
        if tp is None:
            raise SystemExit("cached_text_part_raw absent -- this run does not "
                             "use the cached text path, nothing to measure")
        tps.append(torch.as_tensor(np.asarray(tp)))
    T = torch.stack(tps).float()                       # [B, M, D]
    B, M, D = T.shape

    n_alpha = float(getattr(args, "ntxent_dynamic_tau_alpha", 0.0))
    c_alpha = float(getattr(args, "cibhash_dynamic_tau_alpha", 0.0))
    n_skip = bool(getattr(args, "ntxent_dynamic_tau_skip_global", False))
    c_skip = bool(getattr(args, "cibhash_dynamic_tau_skip_global", False))

    print(f"{ds_name}  B={B} M={M} D={D}   "
          f"ntxent_alpha={n_alpha} (skip_global={n_skip})  "
          f"cibhash_alpha={c_alpha} (skip_global={c_skip})")
    print(f"  {'slot':20s}{'mean cos_tt':>13s}{'p90':>8s}"
          f"{'tau x (ntxent)':>16s}{'tau x (cibhash)':>17s}")

    rows = {}
    off = ~torch.eye(B, dtype=torch.bool)
    for m in range(M):
        tn = F.normalize(T[:, m, :], dim=-1, eps=1e-8)
        cos = (tn @ tn.T).clamp(-1.0, 1.0)[off]
        mu = float(cos.mean())
        p90 = float(cos.quantile(0.90))
        nx = 1.0 if (n_skip and m == 0) else 1.0 + n_alpha * mu
        cb = 1.0 if (c_skip and m == 0) else 1.0 + c_alpha * mu
        print(f"  {SLOTS[m] if m < len(SLOTS) else m:20s}{mu:13.4f}{p90:8.4f}"
              f"{nx:16.4f}{cb:17.4f}")
        rows[SLOTS[m] if m < len(SLOTS) else str(m)] = {
            "mean_cos_tt": round(mu, 5), "p90_cos_tt": round(p90, 5),
            "tau_mult_ntxent": round(nx, 5), "tau_mult_cibhash": round(cb, 5)}

    loc = [v["mean_cos_tt"] for k, v in rows.items() if k != "global"]
    print(f"  local-slot cos_tt spread: min={min(loc):.4f} max={max(loc):.4f} "
          f"range={max(loc)-min(loc):.4f}")

    out = {"dataset": ds_name, "dir": a.result_dir, "n": B,
           "ntxent_alpha": n_alpha, "cibhash_alpha": c_alpha,
           "ntxent_skip_global": n_skip, "cibhash_skip_global": c_skip,
           "slots": rows}
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=2)
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
