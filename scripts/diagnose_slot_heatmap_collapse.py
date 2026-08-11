#!/usr/bin/env python
"""WHERE do the per-slot routing heatmaps become identical?

The 5-slot qualitative figures show four local-slot heatmaps that look the same
on most Flickr25k images even though their captions differ completely. "The
slots collapse" is not an actionable statement -- the collapse has to be located
on the path that produces the plan, because a different stage implies a
different fix:

    caption text -> CLIP text embedding -> per-slot text adapter -> t_m
      -> sim[n, m] = cos(x_n, t_m) -> cost -> Sinkhorn(epsilon) -> P -> top-p mask

If the text anchors t_m are already near-collinear, NO OT setting can separate
the columns: the cost matrix is essentially rank-1 and every slot is asking the
same question of the image. If the anchors are well separated but the plan is
not, the entropic term is flattening a signal that exists, and epsilon (or the
mask) is the lever. This measures each stage on the same forward pass so the
stages are directly comparable.

METRIC. Raw column cosine is misleading here: every column of P inherits the
same "which patches carry any mass at all" envelope, which inflates similarity
regardless of specialisation. The number that matters is the cosine AFTER
subtracting the per-patch mean across slots -- that residual is exactly what
makes one slot's heatmap look different from another's, since each patch
distributes a fixed amount of mass among the slots. Both are reported; read the
centered one.

EPSILON REPLAY. The plan is recomputed from the SAME cost at several epsilon
values, so any change in column similarity is attributable to the entropic term
alone and not to a different model. This is what says whether tuning epsilon
could work before spending a training run on it.
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


def _pair_cos(A: torch.Tensor) -> float:
    """Mean off-diagonal cosine between the columns of [B, N, M]."""
    An = F.normalize(A.transpose(1, 2), dim=-1, eps=1e-8)      # [B, M, N]
    G = torch.bmm(An, An.transpose(1, 2))                      # [B, M, M]
    M = G.shape[-1]
    off = ~torch.eye(M, dtype=torch.bool, device=G.device)
    return float(G[:, off].mean())


def _centered(A: torch.Tensor) -> torch.Tensor:
    """Remove the per-patch mean across slots: the part that can differentiate."""
    return A - A.mean(dim=-1, keepdim=True)


def _sinkhorn(cost, eps, n_iter=50):
    """Plain entropic OT with uniform marginals, on the given cost."""
    B, N, M = cost.shape
    log_K = -cost / max(eps, 1e-6)
    log_a = torch.full((B, N), -np.log(N), device=cost.device, dtype=cost.dtype)
    log_b = torch.full((B, M), -np.log(M), device=cost.device, dtype=cost.dtype)
    f = torch.zeros_like(log_a)
    g = torch.zeros_like(log_b)
    for _ in range(n_iter):
        f = log_a - torch.logsumexp(log_K + g.unsqueeze(1), dim=2)
        g = log_b - torch.logsumexp(log_K + f.unsqueeze(2), dim=1)
    return (log_K + f.unsqueeze(2) + g.unsqueeze(1)).exp()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", required=True)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--split", default="train", choices=("train", "test"))
    ap.add_argument("--epsilons", type=float, nargs="*",
                    default=[1.0, 0.5, 0.3, 0.1, 0.05, 0.02])
    ap.add_argument("--sample_seed", type=int, default=1234)
    ap.add_argument("--epoch", type=int, default=None)
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
    rng = np.random.default_rng(a.sample_seed)
    order = rng.permutation(len(ds))[:min(a.n, len(ds))].tolist()

    acc = {k: [] for k in ("t_raw", "t_adapt", "sim", "sim_c",
                           "P_pre", "P_pre_c", "P_post", "P_post_c")}
    eps_acc = {e: [] for e in a.epsilons}
    eps_acc_c = {e: [] for e in a.epsilons}
    n_seen = 0

    with torch.no_grad():
        for i0 in range(0, len(order), 32):
            samples = [ds[i] for i in order[i0:i0 + 32]]
            batch = {}
            for k in samples[0]:
                v = [s[k] for s in samples]
                if torch.is_tensor(v[0]):
                    batch[k] = torch.stack(v)
                elif isinstance(v[0], np.ndarray):
                    batch[k] = torch.from_numpy(np.stack(v))
                else:
                    batch[k] = v
            o = _forward_text_routed(model, batch, a.device)

            # --- stage 1/2: the text anchors, before and after the adapter ---
            traw = batch.get("cached_text_part_raw")
            if traw is not None and torch.is_tensor(traw):
                # drop axis 0 (C_global) so stages 1 and 2 count the same slots
                t0 = traw.to(a.device).float()[:, 1:, :]           # [B, M_loc, D]
                acc["t_raw"].append(_pair_cos(t0.transpose(1, 2)))
            # LOCAL slots only: C_global bypasses the router entirely, so
            # including it would score a column that no heatmap panel shows.
            t = o.get("local_text_tokens")
            if t is None:
                raise SystemExit("local_text_tokens absent; nothing to analyse")
            t = t.float()                                          # [B, M_loc, D]
            acc["t_adapt"].append(_pair_cos(t.transpose(1, 2)))

            # --- stage 3: the cost the router actually sees -------------------
            x = o.get("visual_tokens")
            if x is None:
                raise SystemExit("no patch-token tensor in the model output; "
                                 "cannot rebuild the cost matrix")
            x = x.float()                                          # [B, N, D]
            sim = torch.matmul(F.normalize(x, dim=-1),
                               F.normalize(t, dim=-1).transpose(-1, -2))   # [B,N,M]
            acc["sim"].append(_pair_cos(sim))
            acc["sim_c"].append(_pair_cos(_centered(sim)))

            # --- stage 4/5: the plan, before and after the nucleus mask -------
            Ppre = o.get("routing_matrix_premask")
            Ppost = o.get("local_routing_matrix")
            if Ppre is not None:
                Ppre = Ppre.float()
                acc["P_pre"].append(_pair_cos(Ppre))
                acc["P_pre_c"].append(_pair_cos(_centered(Ppre)))
            if Ppost is not None:
                Ppost = Ppost.float()
                acc["P_post"].append(_pair_cos(Ppost))
                acc["P_post_c"].append(_pair_cos(_centered(Ppost)))

            # --- epsilon replay on the identical cost ------------------------
            cost = 1.0 - sim
            for e in a.epsilons:
                P = _sinkhorn(cost, e)
                eps_acc[e].append(_pair_cos(P))
                eps_acc_c[e].append(_pair_cos(_centered(P)))
            n_seen += x.shape[0]

    m = {k: (float(np.mean(v)) if v else None) for k, v in acc.items()}
    print(f"{args.dataset} [{a.split}]  images={n_seen}  slots={t.shape[1]}\n")
    print("PIPELINE — mean pairwise cosine between slots (1.000 = indistinguishable)")
    print(f"  {'stage':38s}{'raw':>9s}{'centered':>11s}")
    rows = [("1  CLIP text embedding (pre-adapter)", "t_raw", None),
            ("2  text anchor t_m (post-adapter)",    "t_adapt", None),
            ("3  cost / similarity  cos(x_n, t_m)",  "sim", "sim_c"),
            ("4  Sinkhorn plan, BEFORE top-p mask",  "P_pre", "P_pre_c"),
            ("5  Sinkhorn plan, AFTER top-p mask",   "P_post", "P_post_c")]
    for label, kr, kc in rows:
        r = m.get(kr); c = m.get(kc) if kc else None
        print(f"  {label:38s}{'-' if r is None else f'{r:9.4f}'}"
              f"{'         -' if c is None else f'{c:11.4f}'}")

    print(f"\nEPSILON REPLAY — same cost, plan recomputed (centered is the one to read)")
    print(f"  {'epsilon':>10s}{'raw':>10s}{'centered':>11s}")
    eps_rows = {}
    for e in a.epsilons:
        r = float(np.mean(eps_acc[e])); c = float(np.mean(eps_acc_c[e]))
        print(f"  {e:10.3f}{r:10.4f}{c:11.4f}")
        eps_rows[str(e)] = {"raw": round(r, 5), "centered": round(c, 5)}

    print("\n  If stage 2 is already near 1.0 the anchors are the bottleneck and no")
    print("  epsilon helps. If stage 3 is low but 4 is high, the entropic term is")
    print("  flattening a signal that exists and the replay says how far it can go.")

    out = {"dataset": args.dataset, "dir": a.result_dir, "split": a.split,
           "images": n_seen, "slots": int(t.shape[1]),
           "pipeline": {k: (None if v is None else round(v, 5)) for k, v in m.items()},
           "epsilon_replay": eps_rows}
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=2)
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
