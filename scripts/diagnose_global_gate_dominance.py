#!/usr/bin/env python
"""How much of each slot's codon is decided by the GLOBAL codeword, not its own?

The codon head does not see a slot's own codeword alone (model_siglip2.py:4919):

    q_conditioned_local[m] = q_local_cw[m] + sigmoid(gate_m) * q_global_cw
    codon[m]               = head_m(q_conditioned_local[m])

and the measured gates are saturated (0.9940-0.9990 on all four champions), so
the global codeword is added at essentially full weight. `c_global_source =
siglip2_global` routes the raw CLIP global embedding into slot 0, so whatever
share of the codon it decides is whole-image information wearing that slot's
label.

Counterfactual, on the same batch:
  swap-GLOBAL : replace q_global with another image's q_global (a fixed
                derangement of the batch), keep q_local          -> codon flips?
  swap-LOCAL  : replace q_local[m] with another image's q_local[m], keep
                q_global                                          -> codon flips?

flip(swap-GLOBAL) >> flip(swap-LOCAL) means the codon is mostly a re-encoding
of the whole image. Both are reported per slot; `n_codons_when_global_fixed`
additionally holds q_global at the batch mean to show how much codon diversity
survives without it.

The reconstruction is checked against the model's own `base_indices` before any
counterfactual is reported; a mismatch aborts.
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


def _codon(model, m, hi):
    return model.codon_heads[m](hi, residual=None, gamma=0.0,
                                text_chunks=None)["base_indices"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", required=True)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--epoch", type=int, default=None)
    ap.add_argument("--sample_seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from regen_viz_routing import _parse_args_txt
    args = _parse_args_txt(os.path.join(a.result_dir, "args.txt"))
    from model_siglip2 import SigLIP2SemanticOTModel
    from dataloaders import load_dataset
    from dna_utils.visualization import _forward_text_routed

    model = SigLIP2SemanticOTModel(args).to(a.device).eval()
    model.load_state_dict(torch.load(os.path.join(a.result_dir, "model_state_dict.pth"),
                                     map_location="cpu", weights_only=False), strict=False)
    if a.epoch is not None:
        model.set_current_epoch(a.epoch)
    if model.codon_residual_gamma != 0.0 or model.codon_input_source != "quantized":
        raise SystemExit("this reconstruction assumes codon_residual_gamma=0 and "
                         "codon_input_source=quantized")

    gate = torch.sigmoid(model.global_gate_logits.float())          # [5]
    disabled = bool(getattr(model, "disable_global_gate", False))

    trainset, _, _ = load_dataset(
        getattr(args, "dataset_dir", "dataset"), args.dataset,
        setting=getattr(args, "setting", "setting1"),
        train_transform=None, test_transform=None,
        load_train=True, load_database=False, load_test=False, return_index=True,
        qwen_text_cache_path=getattr(args, "qwen_text_cache_path", None),
        siglip2_feature_cache_dir=getattr(args, "siglip2_feature_cache_dir", None))
    g = np.random.default_rng(a.sample_seed)
    order = g.permutation(len(trainset))[:min(a.n, len(trainset))].tolist()

    flip_g = np.zeros(6); flip_l = np.zeros(6); seen = 0
    nrm_g = np.zeros(6); nrm_l = np.zeros(6)
    codons_fixed, codons_real = [[] for _ in range(6)], [[] for _ in range(6)]
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
            qt = out["quantized_tokens"].float()                    # [B, 6, D]
            ref = out["base_indices"]                               # [B, 18]
            B = qt.shape[0]
            if B < 4:
                continue
            roll = torch.roll(torch.arange(B, device=qt.device), 1)  # derangement

            qg = qt[:, 0, :]                                        # [B, D]
            for m in range(6):
                gm = 1.0 if m == 0 else (0.0 if disabled else float(gate[m - 1]))
                base = qt[:, m, :] if m == 0 else qt[:, m, :] + gm * qg
                c0 = _codon(model, m, base)
                r = ref[:, 3 * m:3 * m + 3]
                if not torch.equal(c0, r):
                    raise SystemExit(
                        f"reconstruction mismatch on slot {m}: "
                        f"{(c0 != r).any(-1).float().mean():.3f} of rows differ")
                if m == 0:
                    codons_real[m].append(c0.cpu().numpy()); continue
                cg = _codon(model, m, qt[:, m, :] + gm * qg[roll])
                cl = _codon(model, m, qt[roll, m, :] + gm * qg)
                cf = _codon(model, m, qt[:, m, :] + gm * qg.mean(0, keepdim=True))
                flip_g[m] += float((cg != c0).any(-1).float().sum())
                flip_l[m] += float((cl != c0).any(-1).float().sum())
                nrm_g[m] += float((gm * qg).norm(dim=-1).sum())
                nrm_l[m] += float(qt[:, m, :].norm(dim=-1).sum())
                codons_fixed[m].append(cf.cpu().numpy())
                codons_real[m].append(c0.cpu().numpy())
            seen += B

    print(f"{args.dataset}  images={seen}  global_gate="
          f"{'DISABLED' if disabled else '[' + ', '.join(f'{float(x):.4f}' for x in gate) + ']'}")
    print(f"  {'slot':20s}{'flip if GLOBAL':>16s}{'flip if LOCAL':>15s}"
          f"{'||g.q_glob||':>14s}{'||q_loc||':>11s}{'codons':>8s}{'codons|g fixed':>16s}")
    rows = {}
    for m in range(1, 6):
        fg, fl = flip_g[m] / seen * 100, flip_l[m] / seen * 100
        ng, nl = nrm_g[m] / seen, nrm_l[m] / seen
        cr = np.concatenate(codons_real[m]); cf = np.concatenate(codons_fixed[m])
        nr = len(np.unique(cr[:, 0] * 16 + cr[:, 1] * 4 + cr[:, 2]))
        nf = len(np.unique(cf[:, 0] * 16 + cf[:, 1] * 4 + cf[:, 2]))
        print(f"  {SLOTS[m]:20s}{fg:15.1f}%{fl:14.1f}%{ng:14.3f}{nl:11.3f}"
              f"{nr:8d}{nf:16d}")
        rows[SLOTS[m]] = {"flip_pct_if_global_swapped": round(fg, 3),
                          "flip_pct_if_local_swapped": round(fl, 3),
                          "norm_gated_global": round(ng, 4),
                          "norm_local": round(nl, 4),
                          "n_codons": int(nr), "n_codons_global_fixed": int(nf)}
    out = {"dataset": args.dataset, "dir": a.result_dir, "images": int(seen),
           "global_gate": None if disabled else [float(x) for x in gate],
           "global_gate_disabled": disabled, "slots": rows}
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=2)
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
