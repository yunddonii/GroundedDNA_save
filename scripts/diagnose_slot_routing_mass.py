#!/usr/bin/env python
"""Per-slot transported mass in the UOT router — are any local slots dead?

Motivation (2026-08-05): the regenerated `viz_routing_heatmap.png` shows some
slot columns with no visible activation. The heatmap only draws the FIVE LOCAL
slots (dna_utils/visualization.py: "col 1..5 : routing-weight heatmap for the 5
local parts") because slot 0 bypasses the router by construction
(`c_global_source=siglip2_global`), so an empty column is a *local* slot and
would be a real defect, not a plotting artifact.

Eyeballing a heatmap cannot distinguish "small but non-zero mass rendered under
the colour floor" from "exactly zero mass". This measures the transport plan
directly:

    P            [B, N, M]   routing matrix from the UOT router
    col_mass[m]  = sum_n P[b,n,m]          how much patch mass slot m receives
    share[m]     = col_mass[m] / sum_m col_mass[m]
    top1[m]      = fraction of patches whose argmax slot is m
    eff_patch[m] = exp(H(P[:,m] normalised))   effective #patches feeding slot m

A slot is reported DEAD if its share is below `--dead_frac` (default 1 %) of the
uniform expectation, i.e. it receives essentially none of the transported mass.

Usage:
  python scripts/diagnose_slot_routing_mass.py --result_dir <dir> [--n 256]
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

SLOTS = ["global", "primary_object", "secondary_object",
         "activity_relation", "color_texture", "scene_type"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", required=True)
    ap.add_argument("--n", type=int, default=256, help="images to average over")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dead_frac", type=float, default=0.01,
                    help="share below this fraction of uniform counts as dead")
    ap.add_argument("--disable_adaptive_topp", action="store_true",
                    help="turn OFF the per-patch nucleus mask (semantic_router.py "
                         "4e) before building the model. The mask is HARD: it "
                         "zeroes every slot outside the nucleus and renormalises "
                         "the row, so a slot that is never in the nucleus gets "
                         "exactly 0 mass. This isolates it from the soft tau_b "
                         "relaxation as a cause of dead slots.")
    ap.add_argument("--epoch", type=int, default=None,
                    help="value to feed model.set_current_epoch(). The Sinkhorn "
                         "epsilon is annealed as a function of the CURRENT EPOCH, "
                         "and _current_epoch is a plain int that is NOT in the "
                         "state dict -- so a fresh model + load_state_dict runs at "
                         "epoch 0 (eps=eps_init), not at the epoch the checkpoint "
                         "was trained to. Pass the training E* to reproduce the "
                         "operating point the weights were actually fitted at.")
    ap.add_argument("--sample_seed", type=int, default=1234,
                    help="RNG for the random subsample; fixed across the matrix "
                         "so every seed/split cell sees the same images")
    ap.add_argument("--split", default="train", choices=("train", "test"),
                    help="which split to measure; test checks that the dead-slot "
                         "pattern is not an artifact of the training rows")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    # config.pt can disagree with the checkpoint (observed 6x64 vs 6x128
    # codebooks); args.txt is the authoritative record, which is why
    # regen_viz_routing.py parses it. Reuse that parser.
    sys.path.insert(0, os.path.join(_REPO, "scripts"))
    from regen_viz_routing import _parse_args_txt
    args = _parse_args_txt(os.path.join(a.result_dir, "args.txt"))
    if a.disable_adaptive_topp:
        # the flag alone gates the kwargs (model_siglip2.py:4395); min/max are
        # still float()-cast unconditionally at construction, so leave them be
        args.routing_adaptive_topp = False
        print("[router] adaptive top-p DISABLED for this measurement")
    from model_siglip2 import SigLIP2SemanticOTModel
    from dataloaders import load_dataset

    model = SigLIP2SemanticOTModel(args).to(a.device).eval()
    sd = torch.load(os.path.join(a.result_dir, "model_state_dict.pth"),
                    map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    bad = [k for k in missing if k.startswith(("quantizer.", "codon_heads.", "router."))]
    if bad:
        raise SystemExit(f"checkpoint/model mismatch on core modules: {bad[:5]}")

    if a.epoch is not None and hasattr(model, "set_current_epoch"):
        model.set_current_epoch(a.epoch)
    if hasattr(model, "_current_sinkhorn_epsilon"):
        _e = model._current_sinkhorn_epsilon()
        _lb = getattr(args, "sinkhorn_lambda_b", None)
        _tb = None if (_e is None or _lb is None) else float(_lb) / (float(_lb) + _e)
        print(f"[router] _current_epoch={getattr(model, '_current_epoch', None)} "
              f"sinkhorn_eps={_e}  lambda_b={_lb}  tau_b={_tb}")

    ds_name = getattr(args, "dataset")
    want_train = (a.split == "train")
    tr, te, _ = load_dataset(
        getattr(args, "dataset_dir", "dataset"), ds_name,
        setting=getattr(args, "setting", "setting1"),
        train_transform=None, test_transform=None,
        load_train=want_train, load_database=False, load_test=not want_train,
        return_index=True,
        qwen_text_cache_path=getattr(args, "qwen_text_cache_path", None),
        siglip2_feature_cache_dir=getattr(args, "siglip2_feature_cache_dir", None))
    trainset = tr if want_train else te
    if trainset is None:
        raise SystemExit(f"load_dataset returned None for split={a.split}")
    from dna_utils.visualization import _forward_text_routed
    from torch.utils.data import default_collate

    # Sample uniformly at random, NOT the first n rows: several splits are
    # stored class-ordered, so a sequential head is one or two classes and the
    # router legitimately concentrates on a couple of slots for it. `sample_seed`
    # is fixed and independent of the model seed so every cell in the matrix
    # scores the SAME images and the seeds stay comparable.
    tot = None; top1 = None; seen = 0
    n_take = min(a.n, len(trainset))
    g = np.random.default_rng(a.sample_seed)
    order = g.permutation(len(trainset))[:n_take].tolist()
    with torch.no_grad():
        for i0 in range(0, n_take, 32):
            idxs = order[i0:i0 + 32]
            samples = [trainset[i] for i in idxs]
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
            P = out.get("routing_matrix")
            if P is None:
                raise SystemExit("routing_matrix absent from model output")
            P = P.float()
            cm = P.sum(dim=1)
            am = P.argmax(dim=-1)
            oh = torch.nn.functional.one_hot(am, P.shape[-1]).float().mean(1)
            tot = cm.sum(0) if tot is None else tot + cm.sum(0)
            top1 = oh.sum(0) if top1 is None else top1 + oh.sum(0)
            seen += P.shape[0]

    tot = (tot / seen).cpu().numpy()
    top1 = (top1 / seen).cpu().numpy()
    share = tot / max(tot.sum(), 1e-12)
    M = len(share)
    uniform = 1.0 / M
    dead = [m for m in range(M) if share[m] < a.dead_frac * uniform]

    print(f"{ds_name} [{a.split}]  images={seen}  slots={M}")
    print(f"  {'slot':20s}{'mass':>10s}{'share':>9s}{'top1%':>8s}   status")
    for m in range(M):
        st = "DEAD" if m in dead else ("routing-free (by design)" if m == 0 else "")
        print(f"  {SLOTS[m] if m < len(SLOTS) else m:20s}"
              f"{tot[m]:10.4f}{share[m]*100:8.2f}%{top1[m]*100:7.1f}%   {st}")
    print(f"  uniform share would be {uniform*100:.2f}%")

    row = {"dataset": ds_name, "split": a.split, "sample_seed": a.sample_seed, "epoch": a.epoch,
           "adaptive_topp_disabled": bool(a.disable_adaptive_topp), "dir": a.result_dir, "images": int(seen),
           "mass": [float(x) for x in tot], "share": [float(x) for x in share],
           "top1_frac": [float(x) for x in top1],
           "dead_slots": dead, "dead_frac_threshold": a.dead_frac}
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(row, open(a.out, "w"), indent=2)
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
