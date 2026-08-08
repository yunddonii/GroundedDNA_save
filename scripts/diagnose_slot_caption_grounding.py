#!/usr/bin/env python
"""Does slot m actually look at what caption m describes?

None of the three metrics we report answers this. Codon decoding measures
code-vs-dataset-label, routing mass measures how MUCH a slot sees, and slot
intervention measures causal effect on retrieval -- none says WHERE the slot
looks or whether that matches its own caption. The gap is not academic: on
CIFAR-10 the images `secondary_object` could not see at all decoded BETTER
(0.9856) than the ones it could (0.8536), because a starved slot's codon comes
from the CLIP global embedding through the saturated global gate.

Two tests, run on the same forward passes.

(1) SIBLING-CAPTION DISCRIMINATION
    Every image carries six captions. Pool the patches with slot m's routing
    column and ask whether the result matches caption m better than the five
    OTHER captions of the SAME image:

        c_m = sum_n P[n, m] x_n ,   j* = argmax_j cos(c_m, t_j) ,   acc = Pr[j* = m]

    Chance is 1/6. Comparing within an image removes "CLIP just matches images
    to text well" as an explanation. PARTIALLY CIRCULAR: xmodal_commit,
    text_code_kl and text_hash_ntxent all train c_m towards t_m, so the absolute
    value is not proof of grounding -- it is useful held-out and for comparing
    cells/ablations. The 6x6 confusion matrix says WHICH axes collapse into
    each other.

(2) CAPTION-SWAP COUNTERFACTUAL  (not circular)
    Replace ONLY slot m's caption and re-run the router:

        delta_m = 1 - cos( P_m(original) , P_m(swapped) )

    delta ~ 0 means the slot ignores its caption and stares at a fixed region;
    large delta means the caption actually steers it. The alignment losses teach
    "be close to the RIGHT caption"; they never teach "look elsewhere when given
    a WRONG one", so this probes causal dependence rather than fit.

    Two swap sources, because they mean different things:
      random  : another image's slot-m caption   (is the slot caption-driven at all?)
      sibling : the same image's slot-(m+1) caption (does it distinguish AXES?)

    Control -- Sinkhorn couples the columns, so perturbing one slot moves the
    others too. `specificity` = delta on the swapped slot minus mean delta on the
    untouched slots. Without it a large delta proves nothing.
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
TEXT_KEYS = ("cached_text_part_raw", "cached_text_tokens", "cached_text_token_mask")


def _collate(samples):
    batch = {}
    for k in samples[0]:
        vals = [s[k] for s in samples]
        if torch.is_tensor(vals[0]):
            batch[k] = torch.stack(vals)
        elif isinstance(vals[0], np.ndarray):
            batch[k] = torch.from_numpy(np.stack(vals))
        else:
            batch[k] = vals
    return batch


def _swap_slot(batch, m, mode):
    """Copy of `batch` with ONLY slot m's caption replaced, on every text key."""
    out = dict(batch)
    for k in TEXT_KEYS:
        v = batch.get(k)
        if v is None or not torch.is_tensor(v) or v.dim() < 2:
            continue
        w = v.clone()
        if mode == "random":                       # another image, same slot
            w[:, m] = v.roll(1, dims=0)[:, m]
        else:                                      # same image, next local slot
            src = 1 + (m % 5) if m >= 1 else 0
            w[:, m] = v[:, src]
        out[k] = w
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", required=True)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--epoch", type=int, default=None)
    ap.add_argument("--split", default="train", choices=("train", "test"))
    ap.add_argument("--sample_seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from regen_viz_routing import _parse_args_txt
    from model_siglip2 import SigLIP2SemanticOTModel
    from dataloaders import load_dataset
    from dna_utils.visualization import _forward_text_routed
    args = _parse_args_txt(os.path.join(a.result_dir, "args.txt"))

    model = SigLIP2SemanticOTModel(args).to(a.device).eval()
    model.load_state_dict(torch.load(os.path.join(a.result_dir, "model_state_dict.pth"),
                                     map_location="cpu", weights_only=False), strict=False)
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

    conf = None                       # [M, M] argmax counts
    seen = 0
    dsw = {"random": {}, "sibling": {}}   # mode -> swapped slot -> [delta per slot]

    with torch.no_grad():
        for i0 in range(0, len(order), 32):
            batch = _collate([ds[i] for i in order[i0:i0 + 32]])
            out0 = _forward_text_routed(model, batch, a.device)
            P0 = out0["routing_matrix"].float()                  # [B, N, M]
            c = out0["semantic_visual_tokens"].float()           # [B, M, D]
            # `cached_text_part_raw` is the RAW CLIP text embedding (512-d) while
            # the pooled visual token is 768-d, so the two are not comparable.
            # `text_part_tokens` is the same caption after the per-slot text
            # adapter, i.e. the representation the model itself aligns c_m to,
            # and it lives in the visual 768-d space.
            t = out0.get("text_part_tokens")
            if t is None:
                raise SystemExit("text_part_tokens absent -- no text representation "
                                 "in the pooled-visual space, discrimination test "
                                 "cannot be run for this config")
            t = t.float()                                        # [B, M, D]
            B, M = c.shape[0], c.shape[1]

            # (1) sibling-caption discrimination
            if c.shape[-1] == t.shape[-1]:
                S = torch.einsum("bmd,bjd->bmj",
                                 F.normalize(c, dim=-1), F.normalize(t, dim=-1))
                pred = S.argmax(-1)                              # [B, M]
                cm = torch.zeros(M, M, device=c.device)
                for m in range(M):
                    cm[m] = torch.bincount(pred[:, m], minlength=M).float()
                conf = cm if conf is None else conf + cm

            # (2) caption-swap counterfactual, one slot at a time
            P0n = F.normalize(P0.transpose(1, 2), dim=-1, eps=1e-8)   # [B, M, N]
            for mode in ("random", "sibling"):
                for m in range(1, M):                 # slot 0 bypasses the router
                    o = _forward_text_routed(model, _swap_slot(batch, m, mode), a.device)
                    Pn = F.normalize(o["routing_matrix"].float().transpose(1, 2),
                                     dim=-1, eps=1e-8)
                    d = (1.0 - (P0n * Pn).sum(-1)).mean(0)        # [M] delta per slot
                    dsw[mode].setdefault(m, []).append(d.cpu().numpy())
            seen += B

    print(f"{args.dataset} [{a.split}] images={seen} epoch={a.epoch}\n")
    rows = {}
    if conf is not None:
        acc = (conf.diag() / conf.sum(-1).clamp_min(1)).cpu().numpy()
        C = (conf / conf.sum(-1, keepdim=True).clamp_min(1)).cpu().numpy()
        print("(1) sibling-caption discrimination   chance = "
              f"{1.0/conf.shape[0]:.3f}")
        print(f"  {'slot (pooled by)':22s}{'acc':>8s}   row = argmax caption share")
        for m in range(conf.shape[0]):
            print(f"  {SLOTS[m]:22s}{acc[m]:8.3f}   "
                  + " ".join(f"{x:.2f}" for x in C[m]))
        print(f"  mean acc (local slots) = {acc[1:].mean():.3f}\n")
        rows["discrimination_acc"] = [round(float(x), 4) for x in acc]
        rows["confusion_row_normalised"] = [[round(float(v), 4) for v in r] for r in C]

    print("(2) caption-swap counterfactual   delta = 1 - cos(P_m before, after)")
    print(f"  {'swapped slot':22s}{'random: own':>13s}{'others':>9s}{'spec':>8s}"
          f"{'sibling: own':>14s}{'others':>9s}{'spec':>8s}")
    for m in range(1, len(SLOTS)):
        line = f"  {SLOTS[m]:22s}"
        rec = {}
        for mode in ("random", "sibling"):
            D = np.mean(np.stack(dsw[mode][m]), axis=0)          # [M]
            own = float(D[m])
            oth = float(np.mean([D[k] for k in range(1, len(D)) if k != m]))
            line += f"{own:13.4f}{oth:9.4f}{own - oth:8.4f}"
            rec[mode] = {"own": round(own, 5), "others": round(oth, 5),
                         "specificity": round(own - oth, 5)}
        print(line)
        rows.setdefault("swap", {})[SLOTS[m]] = rec

    out_d = {"dataset": args.dataset, "dir": a.result_dir, "split": a.split,
             "epoch": a.epoch, "images": int(seen), **rows}
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(out_d, open(a.out, "w"), indent=2)
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
