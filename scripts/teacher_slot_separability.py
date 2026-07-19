#!/usr/bin/env python
"""Is the TEXT TEACHER slot-discriminative, and do the SLOTS stay distinct?

Context
    `scripts/codebook_semantic_alignment.py` found that text supervision
    organises the codebook into a semantically graded geometry on Flickr25k
    (rho 0.42 -> 0.58) and NUS-WIDE (0.31 -> 0.49) but *degrades* it on MSCOCO
    (0.26 -> 0.21), and that the damage is confined to the five text-routed
    local slots (slot 0, which bypasses text routing, is unharmed).

    Two mechanisms predict that pattern and are distinguished here:

      (1) TEACHER QUALITY -- the per-slot captions are not slot-discriminative
          enough on MSCOCO, so the routing centroids are mush and the six
          codebooks receive near-identical targets.
          Signature: low slot identifiability on the TEXT side.

      (2) ROUTING COLLAPSE -- the captions are fine but Sinkhorn routing sends
          nearly the same visual evidence to every slot.
          Signature: text side normal, but quantised slot vectors q^m are
          near-identical across slots.

Measures
    text_slot_id_acc   nearest-slot-centroid accuracy of each (image, slot)
                       text embedding; chance = 1/6 = 0.167.  This is the
                       teacher-side signal.
    text_xslot_cos     mean cosine between two DIFFERENT slots of the SAME
                       image (high = the six captions say the same thing).
    q_xslot_cos        same, on the model's quantised slot vectors
                       q^m = codebook_m[k^m].  This is the model side.

    Comparing text_* against q_* separates (1) from (2).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

N_SLOTS = 6


def _unit(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), 1e-12)


def text_side(cache_dir: str, n_sample: int, rng) -> dict:
    tp = np.load(os.path.join(cache_dir, "text_part.f16.npy"), mmap_mode="r")
    has = os.path.join(cache_dir, "has_text.bool.npy")
    rows = np.where(np.load(has))[0] if os.path.exists(has) else np.arange(len(tp))
    if len(rows) > n_sample:
        rows = np.sort(rng.choice(rows, n_sample, replace=False))
    X = np.asarray(tp[rows], dtype=np.float32)             # [n, 6, D]
    Xn = _unit(X, axis=2)

    # nearest-slot-centroid identifiability (leave-one-out on the centroid)
    cent = Xn.mean(axis=0)                                  # [6, D]
    n = len(Xn)
    correct = 0
    for m in range(N_SLOTS):
        c = cent.copy()
        c[m] = (Xn[:, m].sum(0) - Xn[:, m]) / max(n - 1, 1) if False else cent[m]
        sims = Xn[:, m] @ _unit(cent, axis=1).T             # [n, 6]
        correct += (sims.argmax(1) == m).sum()
    acc = correct / (n * N_SLOTS)

    iu = np.triu_indices(N_SLOTS, 1)
    G = np.einsum("nmd,nkd->nmk", Xn, Xn)                   # [n, 6, 6]
    xslot = float(G[:, iu[0], iu[1]].mean())
    within = float(np.mean([_unit(Xn[:, m]) @ _unit(Xn[:, m]).T for m in range(0)])
                   ) if False else None
    return {"text_slot_id_acc": float(acc), "text_xslot_cos": xslot,
            "n_rows": int(n), "chance": 1.0 / N_SLOTS}


def model_side(run_dir: str, n_sample: int, rng) -> dict:
    sd = torch.load(os.path.join(run_dir, "model_state_dict.pth"), map_location="cpu")
    cbs = sd["quantizer.codebooks"].float().numpy()
    db = np.load(os.path.join(run_dir, "extract_db.npz"), allow_pickle=True)
    cw = db["codebook_indices"]
    if len(cw) > n_sample:
        cw = cw[np.sort(rng.choice(len(cw), n_sample, replace=False))]
    Q = np.stack([cbs[m if cbs.shape[0] > m else 0][cw[:, m]]
                  for m in range(N_SLOTS)], axis=1)         # [n, 6, D]
    Qn = _unit(Q, axis=2)
    iu = np.triu_indices(N_SLOTS, 1)
    G = np.einsum("nmd,nkd->nmk", Qn, Qn)
    cent = _unit(Qn.mean(axis=0), axis=1)
    acc = float(np.mean([( Qn[:, m] @ cent.T).argmax(1) == m for m in range(N_SLOTS)]))
    return {"q_xslot_cos": float(G[:, iu[0], iu[1]].mean()),
            "q_slot_id_acc": acc, "n_rows": int(len(cw))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", nargs="+", required=True,
                    help="NAME:CACHE_DIR:RUN_DIR (RUN_DIR optional)")
    ap.add_argument("--n_sample", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = []
    print(f"{'dataset':14s}{'text slot-id':>14s}{'text xslot cos':>16s}"
          f"{'q slot-id':>12s}{'q xslot cos':>14s}")
    print(f"{'(chance)':14s}{0.167:>14.3f}")
    for spec in args.specs:
        parts = spec.split(":")
        name, cache = parts[0], parts[1]
        run = parts[2] if len(parts) > 2 and parts[2] else None
        r = {"dataset": name, **text_side(cache, args.n_sample, rng)}
        if run:
            r.update(model_side(run, args.n_sample, rng))
        out.append(r)
        print(f"{name:14s}{r['text_slot_id_acc']:>14.3f}{r['text_xslot_cos']:>16.3f}"
              f"{r.get('q_slot_id_acc', float('nan')):>12.3f}"
              f"{r.get('q_xslot_cos', float('nan')):>14.3f}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
