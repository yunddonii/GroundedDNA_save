"""Diagnostic viz: show per-slot text->visual alignment INDEPENDENT of routing.

For each image, computes cos(text_part_adapted[m], visual_tokens[p]) per local
slot (m=1..5) and renders as heatmap overlay. Answers: "if text points to
anatomy region m, does it actually align with the image's anatomy region?"

Compares directly against `viz_routing_heatmap.png` which shows the routing
DECISION (using codebook_mean at inference). Divergence between these two
tells us whether the problem is:
  (a) codebook_mean loses text signal at inference (fixable in Path B), or
  (b) CLIP text-visual alignment is intrinsically weak at patch level.

Uses model.train() forward with no_grad to force text_part_tokens computation.

Usage:
    python scripts/diagnostic_text_alignment_viz.py \
        --result_dir <path> \
        --num_samples 8 \
        --device cuda:0
"""
from __future__ import annotations
import argparse, json, os, sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path: sys.path.insert(0, _REPO)


def _parse_args_txt(path: str) -> SimpleNamespace:
    ns = SimpleNamespace()
    with open(path) as f:
        for raw in f:
            line = raw.rstrip("\n")
            for k_end in range(len(line)):
                if line[k_end] == "-": break
            else: continue
            v_start = k_end
            while v_start < len(line) and line[v_start] == "-": v_start += 1
            key = line[:k_end]; val = line[v_start:].strip()
            if not key: continue
            if val == "None": val = None
            elif val == "True": val = True
            elif val == "False": val = False
            else:
                try: val = int(val)
                except:
                    try: val = float(val)
                    except:
                        if val.startswith("[") and val.endswith("]"):
                            inner = val[1:-1].strip()
                            if inner.startswith("'") and inner.endswith("'"):
                                val = [inner[1:-1]]
            setattr(ns, key, val)
    return ns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", required=True)
    ap.add_argument("--num_samples", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--save_name", default="viz_text_alignment_diagnostic.png")
    ap.add_argument("--cache_dir_override", default=None,
                    help="Override cache (typically to whole-image variant).")
    cli = ap.parse_args()

    args = _parse_args_txt(os.path.join(cli.result_dir, "args.txt"))
    args.device = cli.device
    if cli.cache_dir_override:
        args.siglip2_feature_cache_dir = cli.cache_dir_override
        _mw = os.path.join(cli.cache_dir_override, "text_whiten.npz")
        if os.path.exists(_mw):
            args.text_whiten_npz = _mw
    print(f"[diag-viz] cache: {args.siglip2_feature_cache_dir}")

    from model_siglip2 import SigLIP2SemanticOTModel
    model = SigLIP2SemanticOTModel(args).to(cli.device)
    ckpt = os.path.join(cli.result_dir, "model_state_dict.pth")
    sd = torch.load(ckpt, map_location=cli.device, weights_only=False)
    model.load_state_dict(sd, strict=False)
    # NOTE: train() mode forces text_part_tokens computation
    model.train()

    from dataloaders import ImgRtvCUB2011, ImgRtvDataset
    dataset_name = getattr(args, "dataset")
    setting = getattr(args, "setting", "setting1")
    dataset_dir = getattr(args, "dataset_dir", "dataset")
    qwen_path = getattr(args, "qwen_text_cache_path", None)
    cache_dir = args.siglip2_feature_cache_dir

    if dataset_name == "CUB_200":
        ds = ImgRtvCUB2011(root=os.path.join(dataset_dir, "CUB_200"),
                           mode="train", setting_name=setting,
                           qwen_text_cache_path=qwen_path,
                           siglip2_feature_cache_dir=cache_dir)
    else:
        ds_root = os.path.join(dataset_dir, dataset_name)
        ds = ImgRtvDataset(root=ds_root, mode="train", setting_name=setting,
                           qwen_text_cache_path=qwen_path,
                           siglip2_feature_cache_dir=cache_dir)

    # Load Qwen captions for display
    caps_lookup = {}
    if qwen_path and os.path.exists(qwen_path):
        with open(qwen_path) as f:
            for line in f:
                try:
                    r = json.loads(line.strip())
                    if "image_id" in r and "codebook_texts" in r:
                        caps_lookup[r["image_id"]] = r["codebook_texts"]
                except: pass

    # Sample images
    np.random.seed(0)
    N = min(cli.num_samples, len(ds))
    idx = np.random.choice(len(ds), N, replace=False)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(ds, idx.tolist()),
        batch_size=N, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    allowed = {"cached_visual_tokens_raw", "cached_visual_global", "cached_text_part_raw",
               "cached_has_text", "cached_text_tokens", "cached_text_token_mask"}
    in_kw = {}
    for k, v in batch.items():
        if k in allowed:
            in_kw[k] = v.to(cli.device) if torch.is_tensor(v) else v
        elif k == "has_text":
            in_kw["cached_has_text"] = v.to(cli.device) if torch.is_tensor(v) else v

    with torch.no_grad():
        out = model(**in_kw)
    v_tokens = out.get("visual_tokens")           # [B, N, D]
    t_part   = out.get("text_part_tokens")        # [B, M, D]
    routing_matrix = out.get("routing_matrix")    # [B, N, M]

    # Compute per-slot text->visual alignment
    v_n = torch.nn.functional.normalize(v_tokens, dim=-1)
    t_n = torch.nn.functional.normalize(t_part, dim=-1)
    # sim[b, m, p] = cos(text[b, m], visual[b, p])
    sim = torch.einsum('bmd,bpd->bmp', t_n, v_n).cpu().numpy()   # [B, M, N_patch]

    # Get routing_matrix for comparison (also inference codebook_mean version)
    model.eval()
    with torch.no_grad():
        out_eval = model(**in_kw)
    routing_eval = out_eval.get("routing_matrix").cpu().numpy()  # [B, N, M] or [B, M, N]
    # Handle shape [B, N, M] vs [B, M, N]
    B, dim2, dim3 = routing_eval.shape
    if dim3 <= 8:  # M is last (routing_matrix [B, N, M])
        routing_eval = routing_eval.transpose(0, 2, 1)  # → [B, M, N]

    # Render heatmap comparison
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    M_local = sim.shape[1] - 1   # skip cb0 global
    fig, axes = plt.subplots(N, 2 * M_local + 1, figsize=(3 * (2 * M_local + 1), 3 * N))
    if N == 1:
        axes = axes.reshape(1, -1)

    side = int(np.sqrt(sim.shape[2]))  # 14 for 196
    print(f"[diag-viz] side={side}, N patches={sim.shape[2]}")

    # Sample image paths
    for i in range(N):
        sample = ds[idx[i]]
        img_path = sample.get("image_path", None)
        img_id = sample.get("image_id", str(idx[i]))
        # Get captions
        cb_texts = caps_lookup.get(img_id, {})
        caption_display = "; ".join(list(cb_texts.values())[:3]) if cb_texts else img_id

        # Column 0: image
        if img_path and os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
            axes[i, 0].imshow(img)
        axes[i, 0].axis("off")
        axes[i, 0].set_title(f"[{i}] {img_id[:30]}", fontsize=6)

        for m_local in range(M_local):
            m = m_local + 1  # skip cb0
            slot_key = list(cb_texts.keys())[m] if len(cb_texts) > m else f"C_{m}"
            cap_txt = cb_texts.get(slot_key, "")[:60] if cb_texts else ""

            # (m*2+1): text alignment heatmap
            heat_text = sim[i, m].reshape(side, side)
            ax = axes[i, m_local * 2 + 1]
            ax.imshow(heat_text, cmap="jet", vmin=heat_text.min(), vmax=heat_text.max())
            ax.axis("off")
            ax.set_title(f"TEXT slot {m}\n{cap_txt[:40]}", fontsize=5)

            # (m*2+2): routing_matrix at eval
            heat_route = routing_eval[i, m].reshape(side, side)
            ax = axes[i, m_local * 2 + 2]
            ax.imshow(heat_route, cmap="jet", vmin=heat_route.min(), vmax=heat_route.max())
            ax.axis("off")
            ax.set_title(f"ROUTE slot {m}\n(codebook_mean)", fontsize=5)

    save_path = os.path.join(cli.result_dir, cli.save_name)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    print(f"[diag-viz] saved -> {save_path}")


if __name__ == "__main__":
    main()
