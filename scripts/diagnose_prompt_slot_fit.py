#!/usr/bin/env python
"""Does the 6-axis V4 prompt actually fit each dataset?

Reading CIFAR-10 V4 captions raises two concrete worries that the earlier
per-slot statistics cannot separate:

  (A) HALLUCINATED DETAIL. CIFAR images are 32x32 upsampled. A caption like
      "Two bright yellow landing gear wheels are visible beneath the plane"
      describes detail that is not in the pixels. The router scores patches by
      similarity to the text anchor, so an anchor with no visual referent loses
      every patch competition and the slot starves.
      -> measure, per slot, the anchor-to-patch similarity (the routing cost).

  (B) AXIS COLLAPSE. One sampled CIFAR `secondary_object` caption reads
      "Distant mountainous landscape and soft atmospheric haze form the
      background" — that is a SCENE description occupying the secondary-object
      slot. If two axes describe the same thing, the router cannot separate
      them and one must lose.
      -> measure the cross-slot anchor similarity WITHIN an image.

(A) predicts the dying slots have uniformly low anchor-patch similarity.
(B) predicts the dying slots are the ones most redundant with another axis.
The two are distinguishable and both are reported here.
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
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--epoch", type=int, default=None)
    ap.add_argument("--sample_seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from regen_viz_routing import _parse_args_txt
    args = _parse_args_txt(os.path.join(a.result_dir, "args.txt"))
    args.routing_adaptive_topp = False
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

    xsim_sum = None          # cross-slot anchor similarity, within image
    apsim = []               # [B, M] anchor-to-patch similarity summary
    apmax = []
    seen = 0
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
            out = _forward_text_routed(model, batch, a.device)
            tp = batch.get("cached_text_part_raw")
            if tp is None:
                raise SystemExit("cached_text_part_raw absent")
            t = F.normalize(torch.as_tensor(tp).float().to(a.device), dim=-1)  # [B,M,D]
            xs = torch.einsum("bmd,bnd->bmn", t, t)                            # [B,M,M]
            xsim_sum = xs.sum(0) if xsim_sum is None else xsim_sum + xs.sum(0)

            # anchor-to-patch similarity: the quantity the router ranks on.
            # semantic_visual_tokens is post-routing, so use the raw patch bank.
            vt = out.get("visual_tokens_raw", out.get("visual_tokens"))
            if vt is None:
                raise SystemExit("no raw visual tokens in output")
            v = F.normalize(vt.float(), dim=-1)                                # [B,N,Dv]
            if v.shape[-1] != t.shape[-1]:
                # project text anchors through whatever the router uses; fall
                # back to skipping this half rather than comparing wrong spaces
                apsim = None
            if apsim is not None:
                s = torch.einsum("bmd,bnd->bmn", t, v)                         # [B,M,N]
                apsim.append(s.mean(-1).cpu().numpy())
                apmax.append(s.max(-1).values.cpu().numpy())
            seen += t.shape[0]

    X = (xsim_sum / seen).cpu().numpy()
    print(f"{args.dataset}  images={seen}")
    print("\n  CROSS-SLOT anchor similarity within an image "
          "(high off-diagonal = two axes describe the same thing)")
    print(f"  {'':20s}" + "".join(f"{s[:9]:>10s}" for s in SLOTS))
    for m in range(6):
        print(f"  {SLOTS[m]:20s}" + "".join(
            f"{X[m][n]:10.4f}" if n != m else f"{'--':>10s}" for n in range(6)))
    off = [(X[m][n], SLOTS[m], SLOTS[n]) for m in range(1, 6) for n in range(m + 1, 6)]
    off.sort(reverse=True)
    print("  most redundant LOCAL pairs:")
    for v, x, y in off[:3]:
        print(f"    {x:20s} <-> {y:20s} {v:.4f}")
    rows = {"cross_slot_sim": [[round(float(X[m][n]), 5) for n in range(6)] for m in range(6)],
            "top_redundant_local_pairs": [[x, y, round(float(v), 5)] for v, x, y in off[:3]]}

    if apsim:
        A = np.concatenate(apsim); Am = np.concatenate(apmax)
        print("\n  ANCHOR-to-PATCH similarity (what the router ranks on)")
        print(f"  {'slot':20s}{'mean':>10s}{'max':>10s}")
        for m in range(6):
            print(f"  {SLOTS[m]:20s}{A[:, m].mean():10.4f}{Am[:, m].mean():10.4f}")
        rows["anchor_patch_mean"] = [round(float(A[:, m].mean()), 5) for m in range(6)]
        rows["anchor_patch_max"] = [round(float(Am[:, m].mean()), 5) for m in range(6)]
    else:
        print("\n  ANCHOR-to-PATCH similarity SKIPPED: text and visual banks live "
              "in different dimensions here, so a raw cosine would be meaningless.")

    out_d = {"dataset": args.dataset, "dir": a.result_dir, "images": int(seen), **rows}
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(out_d, open(a.out, "w"), indent=2)
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
