"""Helpers for the off-protocol architecture experiments (branch arch-exp-2026-09).

Called from train_siglip2.py only when the corresponding flag is on, so the
default training path never imports this module.

  init_routing_mass_stats   (a-1) fixed statistics for the content-dependent slot mass
  init_codebook_from_routed_visual   (c-1) k-means codebook at the end of the VQ-free stage
"""
from __future__ import annotations

import os
import random
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F


def _training_rows(args) -> np.ndarray:
    """Cache rows the run optimises on: the stage-1 optimisation split when a
    held-out validation split is carved, otherwise all designated-train rows.
    The validation rows never enter these statistics."""
    cache = str(args.siglip2_feature_cache_dir)
    name = ("opt_train_rows.npy"
            if float(getattr(args, "val_split_ratio", 0.0) or 0.0) > 0.0
            else "train_all_rows.npy")
    rows = np.load(os.path.join(cache, name))
    return np.asarray(rows, dtype=np.int64), name


def init_routing_mass_stats(model, args) -> Dict[str, Any]:
    """(a-1) p_m, mu_m and tau from the frozen CLIP caches -- no DataLoader,
    so the global RNG stream is untouched and the run stays comparable."""
    cache = str(args.siglip2_feature_cache_dir)
    rows, rows_name = _training_rows(args)
    M_loc = int(model.routing_mass_proto.shape[0])
    tp = np.load(os.path.join(cache, "text_part.f16.npy"), mmap_mode="r")      # [R, 6, D]
    vg = np.load(os.path.join(cache, "visual_global.f16.npy"), mmap_mode="r")  # [R, D]
    has = np.load(os.path.join(cache, "has_text.bool.npy"), mmap_mode="r")
    rows = rows[np.asarray(has[rows], dtype=bool)]
    T = torch.from_numpy(np.asarray(tp[rows][:, 1:1 + M_loc], dtype=np.float32))  # local slots
    G = torch.from_numpy(np.asarray(vg[rows], dtype=np.float32))
    T = F.normalize(T, dim=-1)
    G = F.normalize(G, dim=-1)
    P = F.normalize(T.mean(dim=0), dim=-1)                                      # [M_loc, D]
    S = G @ P.t()                                                               # [N, M_loc]
    mu = S.mean(dim=0)
    Sc = S - mu
    tau_arg = float(getattr(args, "routing_mass_tau", 0.0) or 0.0)
    tau = tau_arg if tau_arg > 0.0 else float(Sc.std())
    alpha = float(model.routing_mass_alpha)
    W = torch.softmax(Sc / tau, dim=-1)
    mass = (1.0 - alpha) / M_loc + alpha * W                                    # [N, M_loc]
    with torch.no_grad():
        dev = model.routing_mass_proto.device
        model.routing_mass_proto.copy_(P.to(dev))
        model.routing_mass_center.copy_(mu.to(dev))
        model.routing_mass_tau.fill_(tau)
        model.routing_mass_ready.fill_(True)
    uni = 1.0 / M_loc
    return {
        "rows_file": rows_name, "N": int(len(rows)), "alpha": alpha,
        "tau": round(tau, 6), "tau_source": "given" if tau_arg > 0 else "pooled SD of centred cos",
        "mu": [round(float(x), 5) for x in mu],
        "prototype_pairwise_cos_mean": round(float(((P @ P.t()).sum() - M_loc) / (M_loc * (M_loc - 1))), 5),
        "mass_mean_per_slot": [round(float(x), 4) for x in mass.mean(dim=0)],
        "mass_min_over_images": round(float(mass.min()), 4),
        "mass_max_over_images": round(float(mass.max()), 4),
        "floor": round((1.0 - alpha) / M_loc, 4), "uniform": round(uni, 4),
        "per_image_max_over_min_median": round(float((mass.max(1).values / mass.min(1).values).median()), 3),
    }


def _collate(samples):
    batch = {}
    for k in samples[0]:
        v = [s[k] for s in samples]
        if torch.is_tensor(v[0]):
            batch[k] = torch.stack(v)
        elif isinstance(v[0], np.ndarray):
            batch[k] = torch.from_numpy(np.stack(v))
        else:
            batch[k] = v
    return batch


def init_codebook_from_routed_visual(model, dataset, args) -> Dict[str, Any]:
    """(c-1) k-means the routed visual tokens (pre-VQ z, text-routed as in
    training) of a fixed subset of the optimisation rows into the codebook.

    Unlike --text_init_codebook this does NOT rescale the centres to the
    1/sqrt(D) init norm: the codewords must sit where z lives, which is where
    EMA would have put them. Every RNG state is saved and restored, so the
    training stream after the stage boundary differs from a run without the
    initialisation only through the codebook itself.
    """
    from sklearn.cluster import KMeans
    from dna_utils.visualization import _forward_text_routed

    states = (torch.get_rng_state(),
              torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
              np.random.get_state(), random.getstate())
    was_training = model.training
    q = model.quantizer
    try:
        n_target = int(getattr(args, "text_init_subset", 4096) or 4096)
        seed = int(getattr(args, "text_init_seed", 42) or 42)
        order = np.random.default_rng(seed).permutation(len(dataset))[:min(n_target, len(dataset))]
        zs = []
        with torch.no_grad():
            for i0 in range(0, len(order), 64):
                batch = _collate([dataset[int(i)] for i in order[i0:i0 + 64]])
                o = _forward_text_routed(model, batch, args.device)
                zs.append(o["semantic_visual_tokens"].detach().float().cpu())
        Z = torch.cat(zs, dim=0)                                               # [N, M, D]
        # (R1, arch-exp-3) The quantiser may not see these tokens as they are.
        # With --quant_center_local it receives the per-image deviation across
        # the LOCAL slots, and optionally that deviation rescaled to the token's
        # original norm. k-means must cluster the SAME thing, or the codewords
        # are placed where the uncentred tokens live while the inputs are
        # deviations -- which is a worse mismatch than no initialisation at all.
        _qc_centered = bool(getattr(model, "quant_center_local", False))
        if _qc_centered and Z.shape[1] > 1:
            _loc = Z[:, 1:, :]
            _dev = _loc - _loc.mean(dim=1, keepdim=True)
            if bool(getattr(model, "quant_center_rescale", False)):
                _dev = _dev * (_loc.norm(dim=-1, keepdim=True)
                               / _dev.norm(dim=-1, keepdim=True).clamp_min(1e-6))
            Z = torch.cat([Z[:, :1, :], _dev], dim=1)
        M_cb, K, D = q.codebooks.shape
        if tuple(Z.shape[1:]) != (M_cb, D):
            raise ValueError(f"routed tokens {tuple(Z.shape)} vs codebooks {tuple(q.codebooks.shape)}")
        inertia = []
        with torch.no_grad():
            for m in range(M_cb):
                km = KMeans(n_clusters=int(q.codebook_size), random_state=seed + m, n_init=4, max_iter=100)
                km.fit(Z[:, m, :].numpy())
                c = torch.from_numpy(km.cluster_centers_).to(q.codebooks.device, q.codebooks.dtype)
                k_act = int(c.shape[0])
                q.codebooks[m, :k_act].copy_(c)
                if q.update_mode == "ema":
                    cs = max(1.0, float(Z.shape[0]) / float(k_act))
                    q.cluster_size[m, :k_act].fill_(cs)
                    q.embed_avg[m, :k_act].copy_(c * cs)
                    q.embed_sqavg[m, :k_act].copy_((c ** 2) * cs)
                inertia.append(round(float(km.inertia_), 2))
        z_norm = float(Z.norm(dim=-1).mean())
        return {"N": int(Z.shape[0]), "M": int(M_cb), "K": int(q.codebook_size),
                "z_mean_norm": round(z_norm, 4), "inertia": inertia, "rescaled": False,
                "centered_like_quantiser": _qc_centered}
    finally:
        torch.set_rng_state(states[0])
        if states[1] is not None:
            torch.cuda.set_rng_state_all(states[1])
        np.random.set_state(states[2])
        random.setstate(states[3])
        model.train(was_training)
