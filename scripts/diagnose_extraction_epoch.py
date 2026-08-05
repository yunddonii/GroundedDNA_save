#!/usr/bin/env python
"""Does the checkpoint-reload extraction run at the WRONG Sinkhorn epsilon?

`model._current_epoch` is a plain int with default 0 and is NOT part of the
state dict (model_siglip2.py:2331); only the trainer ever sets it
(train_siglip2.py:1034).  `extraction_siglip2.extract_code` builds a FRESH
`SigLIP2SemanticOTModel(args)` and calls `load_state_dict`, so every
checkpoint-reload extraction runs at epoch 0.

The router epsilon is annealed as a function of that epoch
(`_current_sinkhorn_epsilon`, cosine 1.0 -> 0.1 over `total_epochs`), and it
also sets the unbalanced-OT slot-marginal strength
`tau_b = lambda_b / (lambda_b + eps)` (semantic_router.py:78).  So a run that
stopped at E*=39 was fitted at eps=0.33 / tau_b=0.75 but has its codes
extracted at eps=1.00 / tau_b=0.50.

This re-extracts the SAME checkpoint at epoch 0 and at the training E*, and
reports mAP@R for both.  Equal mAP => the mismatch is cosmetic (per-slot pooled
features are renormalised downstream).  Different mAP => every reported number
comes from a different operating point than the one that was trained.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", required=True)
    ap.add_argument("--epochs", default="0",
                    help="comma list of epochs to set before extracting; the "
                         "training E* is always appended (parsed from the dir name)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from regen_viz_routing import _parse_args_txt
    args = _parse_args_txt(os.path.join(a.result_dir, "args.txt"))
    args.device = a.device

    # E* is encoded in the stage-2 dir name as ..._P0refit_e<E*>+bs+...
    # NB: the dir name is ..._P0refit_e19+bs+64+..., so splitting on "_" puts
    # "P0refit" and "e19+bs+64+e+60+proj" in SEPARATE tokens -- an earlier
    # per-token parse silently found nothing and only epoch 0 ever ran.
    base = os.path.basename(a.result_dir.rstrip("/"))
    m_ = re.search(r"P0refit_e(\d+)", base)
    e_star = int(m_.group(1)) if m_ else None
    if e_star is None:
        raise SystemExit(f"could not parse E* from {base!r}; refusing to run a "
                         f"single-epoch comparison that proves nothing")
    epochs = [int(x) for x in a.epochs.split(",") if x != ""]
    if e_star is not None and e_star not in epochs:
        epochs.append(e_star)

    from model_siglip2 import SigLIP2SemanticOTModel
    from dataloaders import load_dataset
    from dna_utils.training_utils import get_transform
    from extraction_siglip2 import encode_split
    from evaluation_siglip2 import evaluate_retrieval, resolve_map_at_r

    model = SigLIP2SemanticOTModel(args).to(a.device).eval()
    sd = torch.load(os.path.join(a.result_dir, "model_state_dict.pth"),
                    map_location=a.device, weights_only=False)
    model.load_state_dict(sd, strict=False)

    tf = get_transform("test")
    _, queryset, dbset = load_dataset(
        getattr(args, "dataset_dir", "dataset"), args.dataset,
        setting=getattr(args, "setting", "setting1"),
        train_transform=tf, test_transform=tf,
        load_train=False, load_database=True, load_test=True, return_index=True,
        qwen_text_cache_path=getattr(args, "qwen_text_cache_path", None),
        siglip2_feature_cache_dir=getattr(args, "siglip2_feature_cache_dir", None))

    bs = int(getattr(args, "extract_batch_size", 256))
    nw = int(getattr(args, "num_workers", 4))
    db_loader = torch.utils.data.DataLoader(dbset, batch_size=bs, shuffle=False,
                                            num_workers=nw, drop_last=False)
    qy_loader = torch.utils.data.DataLoader(queryset, batch_size=bs, shuffle=False,
                                            num_workers=nw, drop_last=False)
    R = resolve_map_at_r(args.dataset)

    rows = []
    codes = {}
    for e in epochs:
        model.set_current_epoch(e)
        eps = model._current_sinkhorn_epsilon()
        lb = getattr(args, "sinkhorn_lambda_b", None)
        tau_b = None if (eps is None or lb is None) else float(lb) / (float(lb) + eps)
        with torch.no_grad():
            db = encode_split(model, db_loader, a.device, split_name="db")
            qy = encode_split(model, qy_loader, a.device, split_name="query")
        res = evaluate_retrieval(qy, db, distance_mode="base", map_at_r=R)
        key = f"mAP@{R}" if R else "mAP"
        m = res.get("mAP_at_R", res.get("mAP"))
        uniq = len(np.unique(db["base_indices"], axis=0)) / len(db["base_indices"])
        codes[e] = db["base_indices"]
        rows.append({"epoch": e, "sinkhorn_eps": eps, "tau_b": tau_b,
                     "map_at_R": float(m), "R": R,
                     "unique_code_ratio_DB": float(uniq)})
        print(f"[epoch {e:3d}] eps={eps}  tau_b={tau_b}  {key}={m:.4f}  "
              f"DB-unique={uniq:.4f}")

    # how many DB codes actually change between the two operating points
    if len(epochs) >= 2:
        a0, a1 = codes[epochs[0]], codes[epochs[-1]]
        frac = float((a0 != a1).any(axis=1).mean())
        print(f"\nDB codes differing between epoch {epochs[0]} and {epochs[-1]}: "
              f"{frac*100:.2f}%")
    else:
        frac = None

    out = {"dir": a.result_dir, "dataset": args.dataset, "e_star": e_star,
           "rows": rows, "frac_db_codes_changed": frac}
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=2)
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
