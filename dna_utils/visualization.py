"""End-of-run diagnostic plots for the SigLIP2 + DNA hashing model.

Two visualizations are produced and saved as PNG files inside the run's
result directory:

    routing_heatmap.png    -- random sampled images with the 5 local-part
                              Sinkhorn routing weights overlaid as heatmaps,
                              optionally annotated with the Qwen text used
                              for that part. Verifies that text supervision
                              attached to the right visual patches.
    codebook_tsne.png      -- per-codebook t-SNE scatter of the (post-router)
                              semantic visual tokens, colored by the
                              quantizer's chosen codeword index. Shows
                              whether the codebook learned a discrete
                              partition vs collapsing onto a few clusters.

Conventions:
    - All forward passes are run with ``torch.no_grad()``; the EMA codebook
      update side-effect is suppressed by temporarily flipping
      ``model.quantizer.update_mode`` to 'gradient' for the viz pass and
      restoring it afterwards.
    - We import matplotlib lazily so the rest of the project does not pay the
      import cost when visualization is disabled.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F


# Same fixed CIFAR10 default-transform ImageNet stats used by `dataloaders.ImgRtvCIFAR10`.
# We invert them here only to denormalize for plotting.
_NORM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_NORM_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _denormalize_pil(pixel_values: torch.Tensor) -> np.ndarray:
    """[3, H, W] float -> [H, W, 3] uint8 with the dataloader's normalization undone."""
    img = pixel_values.detach().cpu().float().numpy().transpose(1, 2, 0)   # [H, W, 3]
    img = img * _NORM_STD[None, None, :] + _NORM_MEAN[None, None, :]
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


def _load_qwen_text_lookup(jsonl_path: Optional[str]) -> Dict[str, List[str]]:
    """Return ``image_id -> [6 part strings]`` from the Qwen JSONL cache."""
    if jsonl_path is None or not os.path.exists(jsonl_path):
        return {}
    from .text_description_processor import (
        CODEBOOK_TEXT_KEYS, DEFAULT_FALLBACK_TEXT,
    )
    out: Dict[str, List[str]] = {}
    with open(jsonl_path, "r") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            iid = row.get("image_id")
            cb  = row.get("codebook_texts")
            if not iid or not isinstance(cb, dict):
                continue
            out[iid] = [
                (cb.get(k, "") or DEFAULT_FALLBACK_TEXT) for k in CODEBOOK_TEXT_KEYS
            ]
    return out


def _cifar10_image_id(arr: np.ndarray) -> str:
    """Mirror of ``dataloaders._cifar10_image_id``."""
    import hashlib
    return hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


# ----------------------------------------------------------------- forward helpers

def _move_to(batch: Dict[str, Any], device) -> Dict[str, Any]:
    """Move tensor entries to device. Coerce bool / list-of-bool entries
    (e.g. ``has_text`` from the manual single-sample collate path) into a
    1-D bool tensor on the same device.
    """
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif isinstance(v, (bool, list)):
            try:
                out[k] = torch.as_tensor(v, dtype=torch.bool, device=device)
            except (TypeError, ValueError):
                out[k] = v
        else:
            out[k] = v
    return out


@torch.no_grad()
def _forward_text_routed(model, batch: Dict[str, Any], device) -> Dict[str, Any]:
    """Run model.forward in 'training-style' (text routing) without altering
    state. Used by both routing-viz and t-SNE viz so we can see the routes
    that text *would* drive even at viz time.
    """
    batch = _move_to(batch, device)
    cached_vt = batch.get("cached_visual_tokens_raw")
    cached_vg = batch.get("cached_visual_global")
    cached_tp = batch.get("cached_text_part_raw")
    cached_ht = batch.get("has_text")

    # Suppress EMA mutation. SemanticCodebookQuantizer._ema_update only fires
    # when (update_mode == 'ema' AND self.training); we just keep model.eval()
    # but force training=True on the quantizer? Easier: temporarily flip
    # update_mode to 'gradient' which guarantees no EMA write. We DO need
    # `model.training=True` so the forward picks the text-routing branch
    # (see model_siglip2.forward use_text_routing rule).
    quant     = model.quantizer
    saved_um  = getattr(quant, "update_mode", None)
    saved_tr  = model.training
    if saved_um == "ema":
        quant.update_mode = "gradient"
    model.train()
    try:
        out = model(
            pixel_values=None if cached_vt is not None else batch.get("img"),
            part_input_ids=batch.get("part_input_ids"),
            part_attention_mask=batch.get("part_attention_mask"),
            return_routing=True,
            cached_visual_tokens_raw=cached_vt,
            cached_visual_global=cached_vg,
            cached_text_part_raw=cached_tp,
            cached_has_text=cached_ht,
        )
    finally:
        if saved_um is not None:
            quant.update_mode = saved_um
        if saved_tr:
            model.train()
        else:
            model.eval()
    return out


# ----------------------------------------------------------------- routing viz

LOCAL_PART_LABELS: Tuple[str, ...] = (
    "C_head", "C_body", "C_limb", "C_color/tex", "C_background",
)


def visualize_routing(
    model,
    dataset,
    save_path: str,
    qwen_jsonl_path: Optional[str] = None,
    num_samples: int = 12,
    device=None,
    seed: int = 0,
) -> Optional[str]:
    """Plot Sinkhorn routing weights overlaid on a sample of test images.

    Layout: (num_samples) rows x 6 columns
        col 0 : original image (with image_id title)
        col 1..5 : routing-weight heatmap for the 5 local parts overlaid
                   on the image; subtitle is the per-part Qwen text when
                   available.

    The routing matrix is reshaped from ``[num_tokens]`` to ``[h, w]`` assuming
    a square patch grid (``h*w == num_tokens``). Returns the saved path.
    """
    import matplotlib.pyplot as plt
    if device is None:
        device = next(model.parameters()).device

    N_total = len(dataset)
    if N_total == 0:
        print("[visualize_routing] empty dataset, skipping.")
        return None
    rng = random.Random(seed)
    indices = rng.sample(range(N_total), k=min(num_samples, N_total))
    qwen_lookup = _load_qwen_text_lookup(qwen_jsonl_path)

    # collect samples
    img_arrs: List[np.ndarray] = []           # [H, W, 3] uint8, displayable
    img_ids:  List[str]        = []
    routings: List[np.ndarray] = []           # each [num_tokens, 5]
    qwen_texts: List[List[str]] = []          # each [6 strings]

    for idx in indices:
        sample = dataset[idx]
        # Three image-source paths, in priority:
        #   (1) CIFAR10  -> dataset.data[idx] is the raw uint8 [32,32,3]
        #   (2) ImgRtvDataset (Flickr/MSCOCO/...) -> sample['image_path'] is
        #       the absolute file path; load PIL on the fly. This is the only
        #       path that works when the dataset uses cached SigLIP2 features
        #       (so sample['img'] is skipped to avoid repeated PIL decode).
        #   (3) fallback: denormalize sample['img'] (legacy path).
        if hasattr(dataset, "data") and getattr(dataset, "data", None) is not None:
            raw = dataset.data[idx]
            iid = _cifar10_image_id(raw)
            disp_img = raw                                              # [32, 32, 3] uint8
        elif "image_path" in sample and isinstance(sample["image_path"], str):
            from PIL import Image as _PIL
            try:
                pil = _PIL.open(sample["image_path"]).convert("RGB")
                disp_img = np.asarray(pil)                              # [H, W, 3] uint8
            except Exception as _e:
                # raw image unreadable (e.g. /data link broken); fall back
                disp_img = np.zeros((224, 224, 3), dtype=np.uint8)
            iid = sample["image_path"]
            if hasattr(dataset, "root"):
                # cache key in JSONL is relpath from dataset root
                try:
                    import os as _os
                    iid = _os.path.relpath(sample["image_path"], dataset.root)
                except Exception:
                    pass
        else:
            disp_img = _denormalize_pil(sample["img"])
            iid = ""

        # build a 1-element batch (use default_collate-like stacking)
        batch1: Dict[str, Any] = {}
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                batch1[k] = v.unsqueeze(0)
            elif isinstance(v, np.ndarray):
                batch1[k] = torch.from_numpy(v).unsqueeze(0)
            else:
                batch1[k] = [v]
        out = _forward_text_routed(model, batch1, device)
        local_routing = out.get("local_routing_matrix")
        if local_routing is None:
            print("[visualize_routing] local_routing_matrix is None, "
                  "model probably has no text -- skipping that sample.")
            continue
        # local_routing : [1, num_tokens, 5]
        assert local_routing.dim() == 3 and local_routing.shape[0] == 1
        rout = local_routing[0].detach().cpu().float().numpy()           # [num_tokens, 5]
        img_arrs.append(disp_img)
        img_ids.append(iid)
        routings.append(rout)
        qwen_texts.append(qwen_lookup.get(iid, []))

    n = len(img_arrs)
    if n == 0:
        print("[visualize_routing] no usable samples; skipping plot.")
        return None

    fig, axes = plt.subplots(
        n, 6, figsize=(2.4 * 6, 2.6 * n),
        squeeze=False,
    )
    for i in range(n):
        # original
        ax = axes[i, 0]
        ax.imshow(img_arrs[i])
        ax.set_title(f"img\n{img_ids[i]}", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
        # routing heatmaps (parts 1..5; index 0 is C_global, drawn from uniform)
        rout = routings[i]                           # [num_tokens, 5]
        num_tokens = rout.shape[0]
        side = int(round(num_tokens ** 0.5))
        assert side * side == num_tokens, (
            f"num_tokens={num_tokens} is not a square grid (side={side})"
        )
        # part m: [side, side]
        for m in range(5):
            heat = rout[:, m].reshape(side, side)
            ax = axes[i, m + 1]
            ax.imshow(img_arrs[i])
            ax.imshow(
                heat, alpha=0.55, cmap="jet",
                interpolation="bilinear",
                extent=(0, img_arrs[i].shape[1], img_arrs[i].shape[0], 0),
            )
            txt = qwen_texts[i][m + 1] if len(qwen_texts[i]) >= 6 else ""
            label = LOCAL_PART_LABELS[m]
            # truncate text to fit; matplotlib will ignore lines too long
            if txt:
                txt = txt if len(txt) <= 36 else txt[:33] + "..."
                ax.set_title(f"{label}\n{txt}", fontsize=6)
            else:
                ax.set_title(label, fontsize=7)
            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("Text-guided Sinkhorn routing (heatmap = routing weight per patch)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize_routing] saved -> {save_path}")
    return save_path


# ----------------------------------------------------------------- t-SNE viz

def visualize_codebook_tsne(
    model,
    dataset,
    save_path: str,
    num_samples: int = 2000,
    perplexity: float = 30.0,
    batch_size: int = 256,
    device=None,
    seed: int = 0,
) -> Optional[str]:
    """For each of the 6 codebooks, plot a 2-D t-SNE of its post-router
    semantic visual tokens, colored by the codeword the quantizer chose.

    A collapsed codebook will show one or two big blobs in one color; a
    healthy codebook shows many separable clusters.
    """
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    if device is None:
        device = next(model.parameters()).device

    N_total = len(dataset)
    if N_total == 0:
        print("[visualize_codebook_tsne] empty dataset, skipping.")
        return None
    rng = random.Random(seed + 1)
    indices = rng.sample(range(N_total), k=min(num_samples, N_total))

    M = int(getattr(model, "num_codebooks", 6))
    D = int(getattr(model, "d_model", 768))
    K = int(getattr(model, "codebook_size", 32))

    sem_tokens = np.zeros((len(indices), M, D), dtype=np.float32)        # [N, M, D]
    cb_indices = np.zeros((len(indices), M),    dtype=np.int64)          # [N, M]
    write_at = 0

    quant    = model.quantizer
    saved_um = getattr(quant, "update_mode", None)
    if saved_um == "ema":
        quant.update_mode = "gradient"
    model.eval()
    try:
        for start in range(0, len(indices), batch_size):
            chunk = indices[start:start + batch_size]
            samples = [dataset[i] for i in chunk]
            # collate manually (dataset returns dicts of tensors+ndarrays)
            batch: Dict[str, Any] = {}
            keys = samples[0].keys()
            for k in keys:
                if isinstance(samples[0][k], torch.Tensor):
                    batch[k] = torch.stack([s[k] for s in samples], dim=0)
                elif isinstance(samples[0][k], np.ndarray):
                    batch[k] = torch.from_numpy(np.stack([s[k] for s in samples], axis=0))
            # at viz time we want the EVAL routing path (codebook_mean) so
            # that the t-SNE reflects how the model actually quantizes at
            # inference; do NOT pass text features into the model.
            cached_vt = batch.get("cached_visual_tokens_raw")
            cached_vg = batch.get("cached_visual_global")
            cached_vt = cached_vt.to(device) if cached_vt is not None else None
            cached_vg = cached_vg.to(device) if cached_vg is not None else None
            with torch.no_grad():
                out = model(
                    pixel_values=None if cached_vt is not None else batch.get("img").to(device),
                    part_input_ids=None,
                    part_attention_mask=None,
                    return_routing=True,
                    cached_visual_tokens_raw=cached_vt,
                    cached_visual_global=cached_vg,
                    cached_text_part_raw=None,
                    cached_has_text=None,
                )
            sv = out["semantic_visual_tokens"]                            # [B, M, D]
            ci = out["codebook_indices"]                                  # [B, M]
            assert sv.shape[1] == M and sv.shape[2] == D, sv.shape
            assert ci.shape[1] == M, ci.shape
            B = sv.shape[0]
            sem_tokens[write_at:write_at + B] = sv.detach().cpu().numpy().astype(np.float32)
            cb_indices[write_at:write_at + B] = ci.detach().cpu().numpy().astype(np.int64)
            write_at += B
    finally:
        if saved_um is not None:
            quant.update_mode = saved_um

    sem_tokens = sem_tokens[:write_at]
    cb_indices = cb_indices[:write_at]
    if write_at == 0:
        print("[visualize_codebook_tsne] no samples; skipping plot.")
        return None

    # 2 rows x 3 cols (M = 6)
    cols = 3
    rows = (M + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(4.2 * cols, 4.0 * rows), squeeze=False,
    )
    for m in range(M):
        ax = axes[m // cols, m % cols]
        # t-SNE may be slow; cap at 2000 samples (already capped by num_samples)
        # use random_state=seed to keep plots comparable across runs
        try:
            tsne = TSNE(
                n_components=2,
                perplexity=min(perplexity, max(5, write_at // 8)),
                init="pca",
                random_state=seed,
                n_iter=750,
            )
            pts = tsne.fit_transform(sem_tokens[:, m, :])                # [N, 2]
        except TypeError:
            tsne = TSNE(
                n_components=2,
                perplexity=min(perplexity, max(5, write_at // 8)),
                init="pca",
                random_state=seed,
            )
            pts = tsne.fit_transform(sem_tokens[:, m, :])
        labels = cb_indices[:, m]
        # colored scatter (tab20 cycles every 20; with K up to 64 we re-cycle).
        ax.scatter(pts[:, 0], pts[:, 1],
                   c=labels, cmap="tab20", s=6, alpha=0.7)
        n_used = int(np.unique(labels).size)
        ax.set_title(f"C_{m} -- {n_used}/{K} codewords used", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    # hide leftover axes if M < rows*cols
    for j in range(M, rows * cols):
        axes[j // cols, j % cols].axis("off")
    fig.suptitle(f"per-codebook t-SNE (n={write_at} samples)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    if os.path.dirname(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize_codebook_tsne] saved -> {save_path}")
    return save_path
