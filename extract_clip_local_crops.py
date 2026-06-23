"""Extract CLIP visual features from L random local crops + select top-K by
text-anchor cosine similarity.

For each image:
  1. Generate L deterministic RandomResizedCrop's (seeded with image index +
     view index for reproducibility across re-runs).
  2. Forward each crop through frozen CLIP-ViT-B/16 → [L, 196, H_v=768]
     patch tokens + [L, D_proj=512] CLS-pooled globals.
  3. Compute text anchor = mean over the 5 local v6b slot embeddings
     (cached `text_part[:, 1:6, :].mean(dim=1)`), normalize.
  4. Compute cos(crop_global_normalized, text_anchor_normalized) per crop → [L].
  5. Top-K crops by similarity → concatenate their patch tokens into
     [K*196, H_v] → save as the new `visual_tokens.f16.npy`.

`visual_global` stays as the full-image CLS-pooled global (single, no crop).
Augmented views `visual_tokens_aug{0,1}` go through the same L-crops + top-K
pipeline with different RNG seeds (one per view), preserving paired-aug NtXent
compatibility.

This script is a SUPPLEMENT to `extract_clip_features.py`; it expects a donor
cache (e.g. `cache/cub200_clip_v6bplus`) containing the precomputed
`text_part.f16.npy`, `image_ids.json`, and `meta.json`, plus the original
dataset path-list so it can re-load images for the local crops.

Output cache layout (mirror of `cub200_clip_v6bplus` but with `visual_tokens`
replaced):
    visual_tokens.f16.npy          [N, K*196, H_v]     (local-crop top-K concat)
    visual_global.f16.npy          [N, D_proj]         (full-image CLS, single)
    visual_tokens_aug0.f16.npy     [N, K*196, H_v]
    visual_tokens_aug1.f16.npy     [N, K*196, H_v]
    visual_global_aug0.f16.npy     [N, D_proj]
    visual_global_aug1.f16.npy     [N, D_proj]
    text_part.f16.npy              symlink from donor
    has_text.bool.npy              symlink from donor
    text_whiten.npz                symlink from donor (recomputed if missing)
    image_ids.json                 symlink from donor
    meta.json                      (new, records L/K/scale/seed)

Usage:
    python extract_clip_local_crops.py \\
        --donor_dir   cache/cub200_clip_v6bplus \\
        --out_dir     cache/cub200_clip_v6bplus_localL8K3 \\
        --pathlist_root dataset/CUB_200 \\
        --pathlist_setting setting1 \\
        --num_local_crops 8 \\
        --local_crops_top_k 3 \\
        --crop_scale_min 0.25 --crop_scale_max 0.6 \\
        --clip_backbone openai/clip-vit-base-patch16
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from models.pretrained_backbone_clip import build_clip_backbone, DEFAULT_CLIP_BACKBONE
from models.pretrained_backbone import coerce_pooled_to_tensor


CLIP_PIXEL_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_PIXEL_STD  = [0.26862954, 0.26130258, 0.27577711]


def _read_split_paths(split_txt: str) -> List[str]:
    out: List[str] = []
    with open(split_txt, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(line.split()[0])
    return out


def _iter_pathlist(root: str, setting: str = "setting1") -> List[Tuple[str, str]]:
    base = root.rstrip("/")
    rows: List[Tuple[str, str]] = []
    seen = set()
    for split in ("train", "test", "database"):
        split_txt = os.path.join(base, setting, f"{split}.txt")
        if not os.path.exists(split_txt):
            continue
        for relpath in _read_split_paths(split_txt):
            if relpath in seen:
                continue
            seen.add(relpath)
            rows.append((relpath, os.path.join(base, relpath)))
    return rows


def _crop_then_resize(pil_img: Image.Image, top: int, left: int,
                      h: int, w: int, size: int) -> torch.Tensor:
    """Deterministic crop + resize -> tensor (CLIP-normalized).
    Mirrors transforms.functional.resized_crop semantics."""
    from torchvision.transforms import functional as TF
    cropped = TF.resized_crop(pil_img, top, left, h, w, [size, size],
                              interpolation=transforms.InterpolationMode.BICUBIC)
    t = TF.to_tensor(cropped)
    t = TF.normalize(t, CLIP_PIXEL_MEAN, CLIP_PIXEL_STD)
    return t


def _gen_crop_params(pil_img: Image.Image, scale_min: float, scale_max: float,
                     rng: np.random.Generator) -> Tuple[int, int, int, int]:
    """Sample (top, left, h, w) for a RandomResizedCrop with the given scale
    range. Aspect ratio fixed at 1.0 (square crops); CLIP input is square."""
    W, H = pil_img.size
    img_area = float(H * W)
    for _ in range(10):
        s = float(rng.uniform(scale_min, scale_max))
        target_area = s * img_area
        side = int(round(target_area ** 0.5))
        side = max(8, min(side, min(H, W)))
        top  = int(rng.integers(0, H - side + 1))
        left = int(rng.integers(0, W - side + 1))
        return top, left, side, side
    # fallback: center crop
    side = min(H, W)
    return (H - side) // 2, (W - side) // 2, side, side


def _gen_grid_crops(pil_img: Image.Image, grid_n: int, crop_frac: float):
    """Deterministic N×N grid of overlapping square crops covering the image.

    grid_n=3, crop_frac=0.5 -> 9 crops of 50%-side at positions {0, 0.25, 0.5}
                              of (W - crop_side) and (H - crop_side).
    grid_n=4, crop_frac=0.4 -> 16 crops of 40%-side at positions {0, 0.2, 0.4, 0.6}.
    Each (top, left, side, side) entry resizes to the encoder input size.
    """
    W, H = pil_img.size
    short = min(H, W)
    side = max(8, int(round(crop_frac * short)))
    if grid_n == 1:
        return [((H - side) // 2, (W - side) // 2, side, side)]
    crops = []
    # positions span [0, 1 - crop_frac], i.e. crop top-lefts stay inside the
    # image. grid_n positions per axis -> grid_n^2 total crops.
    for i in range(grid_n):
        for j in range(grid_n):
            ty = (i / (grid_n - 1)) if grid_n > 1 else 0.0
            tx = (j / (grid_n - 1)) if grid_n > 1 else 0.0
            top  = int(round(ty * (H - side)))
            left = int(round(tx * (W - side)))
            crops.append((top, left, side, side))
    return crops


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--donor_dir",  required=True,
                    help="Source cache dir (must contain text_part.f16.npy + "
                         "has_text.bool.npy + image_ids.json + meta.json).")
    ap.add_argument("--out_dir",    required=True,
                    help="Output cache dir (will be created).")
    ap.add_argument("--pathlist_root",    required=True)
    ap.add_argument("--pathlist_setting", default="setting1")
    ap.add_argument("--num_local_crops",  type=int, default=8)
    ap.add_argument("--local_crops_top_k", type=int, default=3)
    ap.add_argument("--crop_scale_min",   type=float, default=0.25)
    ap.add_argument("--crop_scale_max",   type=float, default=0.6)
    ap.add_argument("--crop_mode",        default="random",
                    choices=["random", "grid"],
                    help="random = stochastic L crops with scale in "
                         "[scale_min, scale_max] (sampled per crop); grid = "
                         "deterministic grid_n x grid_n overlapping crops, "
                         "each crop covers --crop_grid_frac of the shorter "
                         "image side. With grid, --num_local_crops is forced "
                         "to grid_n**2.")
    ap.add_argument("--crop_grid_n",      type=int, default=3,
                    help="When --crop_mode=grid, the grid is N x N. (3 -> "
                         "L=9, 4 -> L=16.)")
    ap.add_argument("--crop_grid_frac",   type=float, default=0.5,
                    help="When --crop_mode=grid, each crop's side equals this "
                         "fraction of the shorter image side. Default 0.5 = "
                         "50% side = 25% area, overlapping.")
    ap.add_argument("--num_aug_views",    type=int, default=2,
                    help="Number of paired-aug views (each is its own L-crop "
                         "sampling with different RNG seed).")
    ap.add_argument("--seed",             type=int, default=42)
    ap.add_argument("--clip_backbone",    default=DEFAULT_CLIP_BACKBONE)
    ap.add_argument("--image_size",       type=int, default=224)
    ap.add_argument("--device",           default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size",       type=int, default=128)
    ap.add_argument("--text_anchor_slots", default="1,2,3,4,5",
                    help="Comma-separated indices of text_part slots whose mean "
                         "is used as text anchor. Default 1..5 = the 5 local "
                         "anatomy slots (v6b: head, body, wing, tail, "
                         "pattern_markings). Slot 0 = C_global, usually excluded.")
    ap.add_argument("--anchor_mode", default="text",
                    choices=["text", "image_global"],
                    help="Ranking signal for top-K crop selection. "
                         "'text' (default): cos(crop_CLS, mean(text_part[slots])). "
                         "'image_global': cos(crop_CLS, donor_visual_global) — "
                         "FAIR / WCA style image-self-similarity ranking. Use "
                         "image_global on multi-object scene datasets where the "
                         "averaged text anchor collapses to one dominant object.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.crop_mode == "grid":
        L = int(args.crop_grid_n) ** 2
    else:
        L = int(args.num_local_crops)
    K = int(args.local_crops_top_k)
    assert 1 <= K <= L, f"K={K} must be in [1, L={L}]"

    # ---- load donor cache ------------------------------------------------
    img_ids: List[str] = json.load(open(os.path.join(args.donor_dir, "image_ids.json")))
    text_part = np.load(os.path.join(args.donor_dir, "text_part.f16.npy"), mmap_mode="r")
    donor_meta = json.load(open(os.path.join(args.donor_dir, "meta.json")))
    N = len(img_ids)
    assert text_part.shape[0] == N, f"text_part {text_part.shape} vs N {N}"
    M = int(text_part.shape[1])
    D_proj = int(text_part.shape[2])
    print(f"[local-crops] donor cache: N={N}, M={M}, D_proj={D_proj}")

    # ---- anchor: text-derived (default) OR image-self-similarity (FAIR) --
    slots = [int(x) for x in args.text_anchor_slots.split(",") if x.strip()]
    if args.anchor_mode == "text":
        assert all(0 <= s < M for s in slots), f"slot indices {slots} out of range [0, {M})"
        anchor_np = text_part[:, slots, :].astype(np.float32).mean(axis=1)   # [N, D_proj]
        print(f"[local-crops] anchor=text  slots={slots}  shape={anchor_np.shape}")
    else:
        # FAIR-style: rank crops by cos(crop_CLS, full_image_CLS). Read donor's
        # visual_global cache (the CLIP-projected pooled feature for the full
        # image; same projection layer that produces per-crop globals here, so
        # the dot-product is in a consistent space).
        donor_vg_path = os.path.join(args.donor_dir, "visual_global.f16.npy")
        if not os.path.exists(donor_vg_path):
            raise FileNotFoundError(
                f"--anchor_mode image_global needs donor_dir to contain "
                f"visual_global.f16.npy ({donor_vg_path} missing)."
            )
        anchor_np = np.load(donor_vg_path, mmap_mode="r")[:].astype(np.float32)
        assert anchor_np.shape[0] == N and anchor_np.shape[1] == D_proj, (
            f"donor visual_global shape {anchor_np.shape} mismatch (expected "
            f"({N},{D_proj}))."
        )
        print(f"[local-crops] anchor=image_global (FAIR-style)  shape={anchor_np.shape}")
    text_anchor = torch.from_numpy(anchor_np).to(args.device)
    text_anchor_n = F.normalize(text_anchor, dim=-1)

    # ---- image path lookup ------------------------------------------------
    path_rows = _iter_pathlist(args.pathlist_root, args.pathlist_setting)
    relpath_to_full = {rp: fp for rp, fp in path_rows}
    image_paths: List[str] = []
    for iid in img_ids:
        if iid not in relpath_to_full:
            raise KeyError(f"image_id '{iid}' not found in {args.pathlist_root}/{args.pathlist_setting}")
        image_paths.append(relpath_to_full[iid])
    assert len(image_paths) == N

    # ---- backbone ---------------------------------------------------------
    print(f"[local-crops] loading {args.clip_backbone} ...")
    backbone = build_clip_backbone(args.clip_backbone).to(args.device).eval()
    H_v = int(backbone.vision_hidden_dim)
    patch = (args.image_size // 16) ** 2   # 196 for 224 / patch16
    _interp = (args.image_size != 224)
    print(f"[local-crops] backbone dims: H_v={H_v}, D_proj={D_proj}, patches/crop={patch}")

    # ---- allocate output memmaps -----------------------------------------
    tok_out_shape = (N, K * patch, H_v)
    glob_out_shape = (N, D_proj)

    def _alloc(name, shape):
        p = os.path.join(args.out_dir, name)
        gb = float(np.prod(shape)) * 2 / 1e9
        print(f"[local-crops] allocate {p} {shape} = {gb:.1f} GB at fp16")
        return np.lib.format.open_memmap(p, mode="w+", dtype=np.float16, shape=shape)

    tok_main = _alloc("visual_tokens.f16.npy", tok_out_shape)
    glob_main = _alloc("visual_global.f16.npy", glob_out_shape)
    aug_tok = [
        _alloc(f"visual_tokens_aug{i}.f16.npy", tok_out_shape)
        for i in range(args.num_aug_views)
    ]
    aug_glob = [
        _alloc(f"visual_global_aug{i}.f16.npy", glob_out_shape)
        for i in range(args.num_aug_views)
    ]

    # ---- per-image processing --------------------------------------------
    # View v=0 = main (top-K kept), v=1..= aug paired views (top-K with different
    # crop sampling). Each view uses its own seed sequence.
    n_views = 1 + args.num_aug_views
    print(f"[local-crops] L={L} crops per view, K={K} top-K kept, "
          f"crop scale=({args.crop_scale_min}, {args.crop_scale_max}), views={n_views}")

    bs = max(1, args.batch_size // L)   # process bs images at a time, each unrolled to L crops
    for view_idx in range(n_views):
        view_seed = int(args.seed) * 1009 + view_idx * 31337
        rng_global = np.random.default_rng(view_seed)
        per_image_seeds = rng_global.integers(0, 2**31 - 1, size=N)
        desc = ("main" if view_idx == 0 else f"aug{view_idx-1}")
        out_tok = tok_main if view_idx == 0 else aug_tok[view_idx - 1]
        out_glob = glob_main if view_idx == 0 else aug_glob[view_idx - 1]
        for i_start in tqdm(range(0, N, bs), total=(N + bs - 1) // bs, desc=desc):
            i_end = min(i_start + bs, N)
            B_eff = i_end - i_start
            # build (B*L) crops
            crops_list = []
            for i in range(i_start, i_end):
                pil = Image.open(image_paths[i]).convert("RGB")
                if args.crop_mode == "grid":
                    # deterministic grid; per-view RNG used only for aug
                    # color/flip placeholder (currently not applied).
                    crop_params = _gen_grid_crops(
                        pil, int(args.crop_grid_n), float(args.crop_grid_frac),
                    )
                else:
                    img_rng = np.random.default_rng(int(per_image_seeds[i]))
                    crop_params = [
                        _gen_crop_params(pil, args.crop_scale_min, args.crop_scale_max, img_rng)
                        for _ in range(L)
                    ]
                for (top, left, h, w) in crop_params:
                    crops_list.append(_crop_then_resize(pil, top, left, h, w, args.image_size))
            crops = torch.stack(crops_list, dim=0).to(args.device)    # [B*L, 3, S, S]
            # forward
            v_out = backbone.vision_model(pixel_values=crops, interpolate_pos_encoding=_interp)
            tok_BL = v_out.last_hidden_state[:, 1:, :]                # [B*L, 196, H_v]
            g_out = backbone.model.get_image_features(pixel_values=crops, interpolate_pos_encoding=_interp)
            g_BL = coerce_pooled_to_tensor(g_out)                      # [B*L, D_proj]
            tok_BL = tok_BL.view(B_eff, L, patch, H_v)                # [B, L, 196, H_v]
            g_BL = g_BL.view(B_eff, L, D_proj)                        # [B, L, D_proj]

            # top-K by cos sim to text anchor (per image)
            g_n = F.normalize(g_BL, dim=-1)                           # [B, L, D]
            anchor_n = text_anchor_n[i_start:i_end].unsqueeze(1)      # [B, 1, D]
            sim = (g_n * anchor_n).sum(-1)                            # [B, L]
            topk_idx = sim.topk(K, dim=-1).indices                    # [B, K]
            # gather
            sel_tok = torch.gather(
                tok_BL, dim=1,
                index=topk_idx.view(B_eff, K, 1, 1).expand(B_eff, K, patch, H_v),
            )                                                          # [B, K, 196, H_v]
            sel_tok = sel_tok.reshape(B_eff, K * patch, H_v)           # [B, K*196, H_v]
            sel_g = torch.gather(
                g_BL, dim=1,
                index=topk_idx.view(B_eff, K, 1).expand(B_eff, K, D_proj),
            ).mean(dim=1)                                             # [B, D_proj] (mean of selected K globals)

            out_tok[i_start:i_end] = sel_tok.detach().cpu().numpy().astype(np.float16)
            out_glob[i_start:i_end] = sel_g.detach().cpu().numpy().astype(np.float16)
        out_tok.flush(); out_glob.flush()
    del tok_main, glob_main, aug_tok, aug_glob

    # ---- symlink text-side files from donor ------------------------------
    for f in ("text_part.f16.npy", "has_text.bool.npy",
              "text_whiten.npz", "text_whiten.npz.meta.json",
              "image_ids.json"):
        src = os.path.join(args.donor_dir, f)
        dst = os.path.join(args.out_dir, f)
        if os.path.exists(src) and not os.path.exists(dst):
            os.symlink(os.path.abspath(src), dst)
            print(f"[local-crops] symlink {f}")

    # ---- write new meta --------------------------------------------------
    meta = dict(donor_meta)
    meta.update({
        "num_tokens": K * patch,
        "local_crops_L": L,
        "local_crops_K": K,
        "local_crops_scale": [args.crop_scale_min, args.crop_scale_max],
        "local_crops_seed": int(args.seed),
        "text_anchor_slots": slots,
        "donor_cache": args.donor_dir,
    })
    json.dump(meta, open(os.path.join(args.out_dir, "meta.json"), "w"), indent=2)
    print(f"[local-crops] DONE -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
