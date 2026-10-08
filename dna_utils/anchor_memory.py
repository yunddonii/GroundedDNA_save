"""Image-conditioned routing anchors from a memory of TRAINING captions (Stage 3-C, 2026-10-08).

Problem: during training the Sinkhorn router's per-slot query is the image's own caption
embedding (per axis); at deployment there is no caption and every image gets the same
codebook-mean query. The deployment path has therefore never seen a loss and the
slot-specific signal of the caption-routed codes does not reach the deployed codes.

This module gives the deployment path an image-conditioned query that lives in the CAPTION
space, without any training (DeCap, ICLR 2023, projection-based decoding): for image feature
v and the memory of raw per-axis caption features {t_j,m} of the training rows,

    anchor_m(v) = sum_j softmax_j( cos(v, t_j,m) / tau ) * t_j,m

The result is a convex combination of real caption embeddings, so the model's whitening and
per-slot text adapter (fitted/trained on captions) apply to it unchanged. The memory holds
TRAINING rows only (the stage-1 optimisation split when a validation split is carved); query
and database captions are never read. Everything here is a pure function so it can be unit
tested against a hand computation.
"""
from __future__ import annotations

import hashlib
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Two caption vectors whose cosine exceeds this are treated as the same caption
# (vector-match self-exclusion: training batches carry no cache row index).
SELF_MATCH_COS = 1.0 - 1e-4


def training_rows(args) -> Tuple[np.ndarray, str]:
    """Rows the run optimises on (same rule as dna_utils.arch_exp._training_rows)."""
    cache = str(args.siglip2_feature_cache_dir)
    name = ("opt_train_rows.npy"
            if float(getattr(args, "val_split_ratio", 0.0) or 0.0) > 0.0
            else "train_all_rows.npy")
    rows = np.load(os.path.join(cache, name))
    return np.asarray(rows, dtype=np.int64), name


def build_anchor_memory(args, num_local: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Raw per-axis caption features of the training rows.

    Returns T [L, R, D] float32 (L = num_local axes = cache slots 1..L, R = rows with text)
    and a provenance dict (rows file, N, sha256 of the fp16 bytes that were read).
    """
    cache = str(args.siglip2_feature_cache_dir)
    rows, rows_name = training_rows(args)
    tp = np.load(os.path.join(cache, "text_part.f16.npy"), mmap_mode="r")      # [R_all, 6, D]
    has = np.load(os.path.join(cache, "has_text.bool.npy"), mmap_mode="r")
    rows = rows[np.asarray(has[rows], dtype=bool)]
    if tp.shape[1] < 1 + num_local:
        raise ValueError(f"text_part has {tp.shape[1]} slots, need 1 + {num_local}")
    sub16 = np.ascontiguousarray(np.asarray(tp[rows][:, 1:1 + num_local]))     # [R, L, D] fp16
    T = torch.from_numpy(np.asarray(sub16, dtype=np.float32)).permute(1, 0, 2).contiguous()  # [L, R, D]
    meta = {
        "cache_dir": cache, "rows_file": rows_name, "N": int(len(rows)), "num_local": int(num_local),
        "dim": int(T.shape[-1]), "dtype_on_disk": str(sub16.dtype),
        "sha256_fp16": hashlib.sha256(sub16.tobytes()).hexdigest(),
    }
    return T, meta


def memory_anchor_raw(
    T: torch.Tensor,                 # [L, R, D] raw caption features (memory)
    v: torch.Tensor,                 # [B, D]    raw image global feature
    tau: float = 0.01,
    topk: int = 0,
    exclude_raw: Optional[torch.Tensor] = None,   # [B, L, D] the batch's own raw captions
) -> Tuple[torch.Tensor, torch.Tensor]:
    """DeCap-style projection of an image feature into the caption space of each axis.

    weights = softmax_j(cos(v_b, T[m, j]) / tau); anchor[b, m] = sum_j weights * T[m, j]
    (raw, unnormalised: the downstream whitening expects the raw caption distribution).
    `topk` > 0 keeps the k largest cosines per (b, m) before the softmax.
    `exclude_raw` masks memory rows that ARE the row's own caption (cos > SELF_MATCH_COS);
    duplicate captions of other images are masked too (documented limitation).
    Returns (anchor [B, L, D], max_weight [B, L]).
    """
    if T.dim() != 3 or v.dim() != 2:
        raise ValueError(f"memory_anchor_raw: T {tuple(T.shape)} must be [L,R,D], v {tuple(v.shape)} [B,D]")
    if float(tau) <= 0.0:
        raise ValueError(f"memory_anchor_raw: tau must be > 0 (got {tau})")
    L, R, D = T.shape
    Tn = F.normalize(T.float(), dim=-1)                                  # [L, R, D]
    vn = F.normalize(v.float(), dim=-1)                                  # [B, D]
    sim = torch.einsum("bd,lrd->blr", vn, Tn)                            # [B, L, R]
    if exclude_raw is not None:
        if tuple(exclude_raw.shape[:2]) != (v.shape[0], L):
            raise ValueError(f"exclude_raw {tuple(exclude_raw.shape)} must be [B, {L}, D]")
        en = F.normalize(exclude_raw.float(), dim=-1)                    # [B, L, D]
        own = torch.einsum("bld,lrd->blr", en, Tn) > SELF_MATCH_COS      # [B, L, R]
        sim = sim.masked_fill(own, float("-inf"))
    if int(topk) > 0 and int(topk) < R:
        kth = sim.topk(int(topk), dim=-1).values[..., -1:]               # [B, L, 1]
        sim = sim.masked_fill(sim < kth, float("-inf"))
    w = torch.softmax(sim / float(tau), dim=-1)                          # [B, L, R]
    w = torch.nan_to_num(w, nan=0.0)                                     # all-masked rows -> zero anchor
    anchor = torch.einsum("blr,lrd->bld", w, T.float())                  # [B, L, D]
    return anchor.to(v.dtype), w.max(dim=-1).values.to(v.dtype)
