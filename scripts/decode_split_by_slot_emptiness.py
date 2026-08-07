#!/usr/bin/env python
"""Split held-out codon decoding by whether the slot actually SAW the image.

Motivation (2026-08-05): on the CIFAR-10 champion, `secondary_object` receives
zero visual tokens on 51.76 % of images and `scene_type` on 45.12 %. For those
images `denom.clamp_min(1e-12)` makes the pooled feature the zero vector, so the
slot's codeword is a fixed constant -- yet the emitted codon still varies (22
and 18 distinct symbols). The variation comes from the global gate
(model_siglip2.py:4919):

    q_conditioned_local[m] = q_local_cw[m] + sigmoid(gate_m) * q_global_cw

with the measured gates at 0.9948-0.9978, i.e. saturated. `q_global_cw` is
slot 0's codeword and `c_global_source=siglip2_global` routes the raw CLIP
global embedding into it. So on an image the slot cannot see, its three bases
are a deterministic function of the WHOLE-IMAGE embedding.

If the reported per-slot decoding score is partly earned by that leak, then
restricting the evaluation to images the slot actually saw must LOWER it toward
what the fixed cells (`topp69`, `noTOPP`, both 0.00 % empty) report. If the
score is unchanged, the leak hypothesis is wrong.

The test images are scored in `extract_query.npz` order, which is the order the
dataloader yields, so the emptiness mask is computed in that same sequential
order (NOT the random subsample used by the other diagnostics).
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
    ap.add_argument("--result_dir", required=True,
                    help="model dir (holds args.txt + model_state_dict.pth)")
    ap.add_argument("--ours_dir", default=None,
                    help="extraction dir; defaults to result_dir, or its "
                         "withids/ overlay when present (CIFAR)")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--train_manifest", default=None)
    ap.add_argument("--epoch", type=int, default=None)
    ap.add_argument("--bio_project", action="store_true")
    ap.add_argument("--gc_min_frac", type=float, default=0.4444)
    ap.add_argument("--gc_max_frac", type=float, default=0.5556)
    ap.add_argument("--max_run", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--min_support", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import heldout_codon_decoding as H
    alpha = a.alpha if a.alpha is not None else H.DEFAULT_ALPHA
    min_support = a.min_support if a.min_support is not None else H.DEFAULT_MIN_SUPPORT

    ours = a.ours_dir
    if ours is None:
        w = os.path.join(a.result_dir, "withids")
        ours = w if os.path.exists(os.path.join(w, "extract_query.npz")) else a.result_dir

    # ---- 1) same codes / labels the decoding script uses ------------------
    d = H.load_split(ours, a.train_manifest)
    tr_lab, te_lab = d["train_labels"], d["test_labels"]
    tr_idx, src, qy = d["train_idx"], d["train_src"], d["qy"]
    n_bases = src["base_indices"].shape[1] // H.N_SLOTS

    def _bioproj(arr):
        if not a.bio_project:
            return arr
        from dna_utils.bio_constraints import project_to_valid, is_valid_batch
        arr = np.ascontiguousarray(arr).astype(np.int8)
        uniq, inv = np.unique(arr, axis=0, return_inverse=True)
        valid = is_valid_batch(uniq, a.gc_min_frac, a.gc_max_frac, a.max_run)
        out = uniq.copy()
        for i in np.where(~valid)[0]:
            out[i], _ = project_to_valid(uniq[i], a.gc_min_frac, a.gc_max_frac, a.max_run)
        return out[inv].astype(np.int64)

    tr_u = H.codon_ids(_bioproj(src["base_indices"][tr_idx]), n_bases)
    qy_u = H.codon_ids(_bioproj(qy["base_indices"]), n_bases)
    n_test = len(te_lab)

    # ---- 2) per-test-image emptiness, in extract_query.npz order ----------
    from regen_viz_routing import _parse_args_txt
    margs = _parse_args_txt(os.path.join(a.result_dir, "args.txt"))
    from model_siglip2 import SigLIP2SemanticOTModel
    from dataloaders import load_dataset
    from dna_utils.visualization import _forward_text_routed

    model = SigLIP2SemanticOTModel(margs).to(a.device).eval()
    model.load_state_dict(torch.load(os.path.join(a.result_dir, "model_state_dict.pth"),
                                     map_location="cpu", weights_only=False), strict=False)
    if a.epoch is not None:
        model.set_current_epoch(a.epoch)

    _, testset, _ = load_dataset(
        getattr(margs, "dataset_dir", "dataset"), margs.dataset,
        setting=getattr(margs, "setting", "setting1"),
        train_transform=None, test_transform=None,
        load_train=False, load_database=False, load_test=True, return_index=True,
        qwen_text_cache_path=getattr(margs, "qwen_text_cache_path", None),
        siglip2_feature_cache_dir=getattr(margs, "siglip2_feature_cache_dir", None))
    if len(testset) != n_test:
        raise SystemExit(f"test split size {len(testset)} != decoding rows {n_test}; "
                         f"the emptiness mask would not align")

    ntok = []
    with torch.no_grad():
        for i0 in range(0, n_test, 32):
            samples = [testset[i] for i in range(i0, min(i0 + 32, n_test))]
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
            ntok.append((P > 0).sum(dim=1).cpu().numpy())
    ntok = np.concatenate(ntok)                                   # [n_test, M]

    # ---- 3) decode, then split the per-image AP by the mask ---------------
    print(f"{a.dataset}  test={n_test}  train={len(tr_lab)}  "
          f"bio_project={a.bio_project}  epoch={a.epoch}")
    print(f"  {'slot':20s}{'empty %':>9s}{'mAP all':>10s}{'mAP SEEN':>10s}"
          f"{'mAP EMPTY':>11s}{'seen-all':>10s}")
    rows = {}
    for m in range(H.N_SLOTS):
        r = H.decode_slot(tr_u[:, m], tr_lab, qy_u[:, m], te_lab,
                          4 ** n_bases, alpha, min_support)
        apv = r["_ap"]
        empty = ntok[:, m] == 0
        v = ~np.isnan(apv)
        all_m = float(np.nanmean(apv))
        seen = float(np.nanmean(apv[~empty])) if (v & ~empty).any() else float("nan")
        emp = float(np.nanmean(apv[empty])) if (v & empty).any() else float("nan")
        print(f"  {SLOTS[m]:20s}{empty.mean()*100:8.2f}%{all_m:10.4f}{seen:10.4f}"
              f"{emp:11.4f}{seen-all_m:+10.4f}")
        rows[SLOTS[m]] = {"empty_pct": round(float(empty.mean()*100), 3),
                          "n_empty": int(empty.sum()),
                          "concept_mAP_all": round(all_m, 5),
                          "concept_mAP_seen_only": round(seen, 5),
                          "concept_mAP_empty_only": round(emp, 5),
                          "delta_seen_minus_all": round(seen - all_m, 5)}

    out = {"dataset": a.dataset, "result_dir": a.result_dir, "ours_dir": ours,
           "epoch": a.epoch, "bio_project": a.bio_project,
           "n_test": int(n_test), "slots": rows}
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=2)
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
