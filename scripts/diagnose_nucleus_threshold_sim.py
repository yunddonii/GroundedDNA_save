#!/usr/bin/env python
"""Why does the SAME nucleus range behave oppositely on NUS-WIDE and CIFAR-10?

On CIFAR-10 widening the nucleus 0.3-0.7 -> 0.6-0.95 costs less than removing
the mask entirely (mAP -0.0130 vs -0.0262). On NUS-WIDE it costs MORE than
removing it (-0.0107 vs -0.0066) and wrecks DNA-uniqueness (-0.0616 vs -0.0342)
— non-monotone in nucleus width, which a pure "more slots = smoother" story
cannot produce.

The mask threshold is patch-adaptive (semantic_router.py:410-419):

    tau = tau_min + (1 - p_max) * (tau_max - tau_min)
    keep while prev_cum < tau, rank-1 always kept

so the SAME (tau_min, tau_max) lands somewhere different depending on where the
router's p_max mass sits. This takes ONE unmasked forward per model and replays
every candidate threshold on the identical transport plan, which isolates the
threshold from training differences (all three NUS cells share E*=4, but the
CIFAR cells do not, so replaying on one plan is the only paired comparison).

Reports, per setting: the distribution of how many slots survive per patch, and
how much that count VARIES across patches — an unstable kept-set size means a
slot pools from an inconsistent patch population from image to image, which
shows up downstream as lost code diversity.
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

SETTINGS = [("baseline", 0.3, 0.7), ("topp69", 0.6, 0.95), ("noTOPP", None, None)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", required=True)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--epoch", type=int, default=None)
    ap.add_argument("--sample_seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from regen_viz_routing import _parse_args_txt
    args = _parse_args_txt(os.path.join(a.result_dir, "args.txt"))
    args.routing_adaptive_topp = False          # get the UNMASKED plan
    from model_siglip2 import SigLIP2SemanticOTModel
    from dataloaders import load_dataset
    from dna_utils.visualization import _forward_text_routed

    model = SigLIP2SemanticOTModel(args).to(a.device).eval()
    model.load_state_dict(torch.load(os.path.join(a.result_dir, "model_state_dict.pth"),
                                     map_location="cpu", weights_only=False), strict=False)
    if a.epoch is not None:
        model.set_current_epoch(a.epoch)

    trainset, _, _ = load_dataset(
        getattr(args, "dataset_dir", "dataset"), args.dataset,
        setting=getattr(args, "setting", "setting1"),
        train_transform=None, test_transform=None,
        load_train=True, load_database=False, load_test=False, return_index=True,
        qwen_text_cache_path=getattr(args, "qwen_text_cache_path", None),
        siglip2_feature_cache_dir=getattr(args, "siglip2_feature_cache_dir", None))
    g = np.random.default_rng(a.sample_seed)
    order = g.permutation(len(trainset))[:min(a.n, len(trainset))].tolist()

    # per-(image, slot) fraction of patches on which the slot is DROPPED.
    # Near 0 or 1 => the exclusion is consistent within the image and builds a
    # stable specialisation; near 0.5 => the mask is picking on noise and the
    # slot pools from a different patch population every time.
    pmax_all, keep_all = [], {s[0]: [] for s in SETTINGS}
    drop_frac = {s[0]: [] for s in SETTINGS if s[1] is not None}
    with torch.no_grad():
        for i0 in range(0, len(order), 32):
            samples = [trainset[i] for i in order[i0:i0 + 32]]
            batch = {}
            for k in samples[0]:
                vals = [s_[k] for s_ in samples]
                if torch.is_tensor(vals[0]):
                    batch[k] = torch.stack(vals)
                elif isinstance(vals[0], np.ndarray):
                    batch[k] = torch.from_numpy(np.stack(vals))
                else:
                    batch[k] = vals
            P = _forward_text_routed(model, batch, a.device)["routing_matrix"].float()
            Pp = P / P.sum(-1, keepdim=True).clamp_min(1e-12)     # [B, N, M]
            sp, _ = Pp.sort(-1, descending=True)
            cum = sp.cumsum(-1)
            prev = torch.cat([torch.zeros_like(cum[..., :1]), cum[..., :-1]], -1)
            pm = sp[..., 0]                                        # [B, N]
            pmax_all.append(pm.flatten().cpu().numpy())
            M = Pp.shape[-1]
            for name, lo, hi in SETTINGS:
                if lo is None:
                    keep_all[name].append(np.full(pm.numel(), M, dtype=np.int64))
                    continue
                tau = lo + (1.0 - pm) * (hi - lo)                  # [B, N]
                keep_sorted = prev < tau.unsqueeze(-1)
                keep_sorted[..., 0] = True
                k = keep_sorted.sum(-1).clamp_min(1)
                keep_all[name].append(k.flatten().cpu().numpy())
                keep = torch.zeros_like(Pp, dtype=torch.bool).scatter_(
                    -1, Pp.sort(-1, descending=True)[1], keep_sorted)
                drop_frac[name].append((~keep).float().mean(1).cpu().numpy())  # [B, M]

    pmax = np.concatenate(pmax_all)
    print(f"{args.dataset}  patches={len(pmax)}  slots={M}  epoch={a.epoch}")
    print(f"  p_max (router confidence, unmasked):  "
          f"p10={np.percentile(pmax,10):.4f}  med={np.median(pmax):.4f}  "
          f"p90={np.percentile(pmax,90):.4f}  (uniform would be {1/M:.4f})")
    print()
    print(f"  {'setting':12s}{'mean k':>9s}{'std k':>8s}{'CV':>7s}"
          f"{'k=1 %':>8s}{'k=M %':>8s}   histogram k=1..M")
    rows = {}
    for name, lo, hi in SETTINGS:
        k = np.concatenate(keep_all[name])
        h = np.bincount(k, minlength=M + 1)[1:M + 1] / len(k) * 100
        cv = k.std() / max(k.mean(), 1e-9)
        print(f"  {name:12s}{k.mean():9.3f}{k.std():8.3f}{cv:7.3f}"
              f"{(k==1).mean()*100:7.1f}%{(k==M).mean()*100:7.1f}%   "
              + " ".join(f"{x:5.1f}" for x in h))
        rows[name] = {"mean_k": round(float(k.mean()), 4),
                      "std_k": round(float(k.std()), 4),
                      "cv_k": round(float(cv), 4),
                      "pct_k1": round(float((k == 1).mean() * 100), 3),
                      "pct_kM": round(float((k == M).mean() * 100), 3),
                      "hist_pct": [round(float(x), 3) for x in h]}

    print()
    print("  per-(image, slot) DROP fraction: 0 = always kept, 1 = always dropped,")
    print("  ~0.5 = the mask flips its mind patch to patch (selecting on noise)")
    print(f"  {'setting':12s}{'mean|f-0.5|':>13s}{'f in .2-.8 %':>14s}"
          f"{'f<.05 %':>10s}{'f>.95 %':>10s}")
    for name, lo, hi in SETTINGS:
        if lo is None:
            continue
        f = np.concatenate(drop_frac[name]).ravel()
        mid = float(((f > 0.2) & (f < 0.8)).mean() * 100)
        print(f"  {name:12s}{float(np.abs(f-0.5).mean()):13.4f}{mid:13.1f}%"
              f"{float((f<0.05).mean()*100):9.1f}%{float((f>0.95).mean()*100):9.1f}%")
        rows[name]["drop_frac_mean_abs_dev"] = round(float(np.abs(f-0.5).mean()), 4)
        rows[name]["drop_frac_ambiguous_pct"] = round(mid, 3)

    out = {"dataset": args.dataset, "dir": a.result_dir, "epoch": a.epoch,
           "n_patches": int(len(pmax)), "M": int(M),
           "pmax": {"p10": float(np.percentile(pmax, 10)),
                    "median": float(np.median(pmax)),
                    "p90": float(np.percentile(pmax, 90))},
           "settings": rows}
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=2)
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
