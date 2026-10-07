#!/usr/bin/env python
"""Which loss terms actually move a weight?

A term can be listed in the config, carry a positive lambda, appear in the
training log, and still contribute NOTHING: its inputs may be fully detached,
gated off by a flag, or produced only on a code path this recipe does not take.
Reading lambdas cannot tell them apart -- on 2026-06 the anchor term was found
to have exactly zero gradient while sitting at lambda 0.05 in every launcher.

So this measures it. One real batch, one forward, then for each term
individually:

    total_grad_norm(term) = || d(term) / d(theta) ||   over ALL trainable params

computed with `torch.autograd.grad(..., allow_unused=True, retain_graph=True)`.
A term whose norm is exactly 0 (or which raises "does not require grad") is
inert in this configuration and must not be described in the paper as part of
the objective.

Also reports each term's contribution to the total, lambda * value, so a term
can be judged on whether it is inert (no gradient) versus merely small.

Runs on CPU by default: it is a single batch and the GPUs are usually busy.
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epoch", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from regen_viz_routing import _parse_args_txt
    from model_siglip2 import SigLIP2SemanticOTModel
    from loss_siglip2 import DNACodonHashLoss
    from dataloaders import load_dataset

    args = _parse_args_txt(os.path.join(a.result_dir, "args.txt"))
    args.device = a.device
    model = SigLIP2SemanticOTModel(args).to(a.device)
    model.load_state_dict(torch.load(os.path.join(a.result_dir, "model_state_dict.pth"),
                                     map_location="cpu", weights_only=False), strict=False)
    model.train()
    model.set_current_epoch(a.epoch)
    crit = DNACodonHashLoss(args).to(a.device)

    tr, _, _ = load_dataset(
        getattr(args, "dataset_dir", "dataset"), args.dataset,
        setting=getattr(args, "setting", "setting1"),
        train_transform=None, test_transform=None,
        load_train=True, load_database=False, load_test=False, return_index=True,
        qwen_text_cache_path=getattr(args, "qwen_text_cache_path", None),
        siglip2_feature_cache_dir=getattr(args, "siglip2_feature_cache_dir", None))

    samples = [tr[i] for i in range(a.batch)]
    batch = {}
    for k in samples[0]:
        v = [s[k] for s in samples]
        if torch.is_tensor(v[0]):
            batch[k] = torch.stack(v).to(a.device)
        elif isinstance(v[0], np.ndarray):
            batch[k] = torch.from_numpy(np.stack(v)).to(a.device)
        elif isinstance(v[0], (bool, np.bool_, int, float)):
            # `has_text` arrives as a python bool per sample; the model calls
            # `.any()` on it, so it has to be a tensor, not a list.
            batch[k] = torch.as_tensor(v).to(a.device)
        else:
            batch[k] = v

    def _fwd(b):
        # Mirror the training call (train_siglip2.py:872) so the same code paths
        # fire; a term gated on an input this misses would look inert here.
        return model(
            pixel_values=None,
            part_input_ids=None,
            part_attention_mask=None,
            return_routing=True,
            cached_visual_tokens_raw=b.get("cached_visual_tokens_raw"),
            cached_visual_global=b.get("cached_visual_global"),
            cached_text_part_raw=b.get("cached_text_part_raw"),
            cached_has_text=b.get("has_text"),
            cached_text_tokens=b.get("cached_text_tokens"),
            cached_text_token_mask=b.get("cached_text_token_mask"),
        )

    out = _fwd(batch)
    # The paired-aug second view is NOT optional for this audit. cibhash_ntxent
    # and ntxent take their negatives from it, so omitting it reports the
    # largest-lambda term in the recipe as inert. train_siglip2.py:899 builds it
    # from the `*_aug1` rows of the same cache, with the SAME text inputs.
    out_v2 = None
    if "cached_visual_tokens_aug1" in batch:
        out_v2 = model(
            pixel_values=None, part_input_ids=None, part_attention_mask=None,
            return_routing=True,
            cached_visual_tokens_raw=batch["cached_visual_tokens_aug1"],
            cached_visual_global=batch["cached_visual_global_aug1"],
            cached_text_part_raw=batch.get("cached_text_part_raw"),
            cached_has_text=batch.get("has_text"),
            cached_text_tokens=batch.get("cached_text_tokens"),
            cached_text_token_mask=batch.get("cached_text_token_mask"),
        )
    else:
        print("  WARNING: no paired-aug view in this cache; two-view terms "
              "(cibhash_ntxent, ntxent) will read as inert and must not be "
              "reported as such")
    mh = batch.get("multi_hot_labels")
    lbl = batch.get("label", batch.get("labels"))
    # TD (2026-10-07): path_consistency needs a THIRD (no-text) forward passed
    # as outputs_notext; this audit does not build it, so the term reads as
    # inert here by construction, not by recipe.
    if float(getattr(args, "lambda_path_consistency", 0.0)) > 0.0:
        print("  NOTE: lambda_path_consistency > 0 but this audit passes no "
              "outputs_notext (no-text student forward); 'path_consistency' "
              "will read as inert here and must not be reported as such")
    loss_dict = crit(outputs=out, labels=lbl, multi_hot_labels=mh,
                     epoch=a.epoch, outputs_view2=out_v2,
                     enable_counterfactual=True)

    params = [p for p in model.parameters() if p.requires_grad]
    print(f"{args.dataset}  batch={a.batch}  trainable tensors={len(params)}\n")
    print(f"  {'term':34s}{'value':>11s}{'grad norm':>14s}   verdict")

    rows, inert = {}, []
    for name, val in sorted(loss_dict.items()):
        if name == "loss" or not torch.is_tensor(val) or val.dim() != 0:
            continue
        v = float(val.detach())
        if not val.requires_grad:
            gn, verdict = 0.0, "INERT (no graph)"
        else:
            g = torch.autograd.grad(val, params, retain_graph=True,
                                    allow_unused=True)
            gn = float(sum((x.detach() ** 2).sum() for x in g if x is not None) ** 0.5)
            verdict = "INERT (zero grad)" if gn == 0.0 else ""
        if verdict:
            inert.append(name)
        print(f"  {name:34s}{v:11.5f}{gn:14.6g}   {verdict}")
        rows[name] = {"value": round(v, 6), "grad_norm": gn,
                      "active": bool(gn > 0.0)}

    print(f"\n  ACTIVE : {sum(1 for r in rows.values() if r['active'])}")
    print(f"  INERT  : {len(inert)}  -> {', '.join(inert) if inert else '(none)'}")
    print("\n  'Inert' means this configuration; a term can be live under another"
          "\n  recipe. Do not describe an inert term as part of the objective.")

    out_d = {"dataset": args.dataset, "dir": a.result_dir, "epoch": a.epoch,
             "batch": a.batch, "terms": rows, "inert": inert}
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(out_d, open(a.out, "w"), indent=2)
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
