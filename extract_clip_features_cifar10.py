"""Extract CLIP visual + text features for CIFAR10.

Mirrors extract_clip_features.py but reads from torchvision CIFAR10 dataset
(no filesystem paths — image_ids are md5 hashes of raw uint8 bytes, matching
the SigLIP2 extractor + dataloaders.ImgRtvCIFAR10 convention).

Usage:
    python extract_clip_features_cifar10.py \\
        --cifar10_root ./dataset/CIFAR10 \\
        --qwen_cache_path ./cache/cifar10_qwen.jsonl \\
        --cache_dir ./cache/cifar10_clip \\
        --save_aug_views 2 \\
        --batch_size 128

Outputs (under --cache_dir): matches extract_clip_features.py cache layout.
"""
from __future__ import annotations
import argparse, hashlib, json, os, random, sys
from typing import List, Tuple
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

_REPO = os.path.dirname(os.path.abspath(__file__))
if _REPO not in sys.path: sys.path.insert(0, _REPO)

from models.pretrained_backbone import coerce_pooled_to_tensor
from models.pretrained_backbone_clip import CLIPBackbone, DEFAULT_CLIP_BACKBONE
from dna_utils import build_clip_text_tokenizer, DEFAULT_CLIP_TOKENIZER_NAME
from extract_clip_features import (
    _atomic_write_json,
    _resolve_hf_provenance,
    _seed_worker,
)


SLOT_KEYS_V3V4 = (
    "C_global", "C_primary_object", "C_secondary_object",
    "C_activity_or_relation", "C_color_texture", "C_scene_type",
)

# CIFAR10 Qwen jsonl uses different keys. Map them to V3V4 positions.
CIFAR10_KEY_TO_V3V4 = {
    "C_global": "C_global",
    "C_head_or_main_part": "C_primary_object",
    "C_body_or_secondary_part": "C_secondary_object",
    "C_limb_or_detail_part": "C_activity_or_relation",
    "C_color_texture": "C_color_texture",
    "C_background_null": "C_scene_type",
}


def _cifar10_image_id(arr: np.ndarray) -> str:
    return hashlib.md5(arr.tobytes()).hexdigest()[:16]


def _iter_cifar10(root: str) -> List[Tuple[str, np.ndarray]]:
    from torchvision.datasets import CIFAR10
    rows = []
    for split, train in [("train", True), ("test", False)]:
        ds = CIFAR10(root, train=train, download=True)
        for i in range(len(ds.data)):
            arr = ds.data[i]
            iid = _cifar10_image_id(arr)
            rows.append((iid, arr))
    return rows


def _default_transform(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                             [0.26862954, 0.26130258, 0.27577711]),
    ])


def _build_aug_transform(image_size: int = 224):
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0),
                                      interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                             [0.26862954, 0.26130258, 0.27577711]),
    ])


def _load_qwen_cache(path: str) -> dict:
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                r = json.loads(line)
                iid = r.get("image_id")
                cb = r.get("codebook_texts", {})
                if iid and cb:
                    # Remap CIFAR10 keys to V3V4 positions if not already
                    remapped = {}
                    for k, v in cb.items():
                        vk = CIFAR10_KEY_TO_V3V4.get(k, k)
                        remapped[vk] = v
                    out[iid] = remapped
            except json.JSONDecodeError:
                continue
    return out


def _assert_fresh_cache_targets(
        cache_dir: str, save_aug_views: int, *, overwrite_cache: bool) -> None:
    names = [
        'visual_tokens.f16.npy', 'visual_global.f16.npy', 'text_part.f16.npy',
        'has_text.bool.npy', 'image_ids.json', 'meta.json',
    ]
    names += [
        f'visual_{kind}_aug{view}.f16.npy'
        for view in range(int(save_aug_views))
        for kind in ('global', 'tokens')
    ]
    existing = [os.path.join(cache_dir, name) for name in names
                if os.path.exists(os.path.join(cache_dir, name))]
    if existing and not overwrite_cache:
        raise FileExistsError(
            'refusing to overwrite an existing cache without --overwrite_cache: '
            f'{existing}')


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cifar10_root", default="./dataset/CIFAR10")
    ap.add_argument("--qwen_cache_path", default=None)
    ap.add_argument("--cache_dir", default="./cache/cifar10_clip")
    ap.add_argument("--clip_backbone", default=DEFAULT_CLIP_BACKBONE)
    ap.add_argument("--tokenizer_name", default=DEFAULT_CLIP_TOKENIZER_NAME)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--text_max_length", type=int, default=64)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--save_aug_views", type=int, default=0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--overwrite_cache', action='store_true', default=False)
    args = ap.parse_args()
    if args.seed < 0:
        raise ValueError('--seed must be non-negative')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    augmentation_generator = torch.Generator()
    augmentation_generator.manual_seed(args.seed)

    os.makedirs(args.cache_dir, exist_ok=True)
    np_dtype = np.float16 if args.dtype == "float16" else np.float32

    # ---- 1. enumerate CIFAR10 --------------------------------
    print(f"[extract-clip-cifar10] enumerating CIFAR10 from {args.cifar10_root}")
    rows = _iter_cifar10(args.cifar10_root)
    N = len(rows)
    image_ids = [iid for iid, _ in rows]
    image_arrs = [arr for _, arr in rows]
    print(f"[extract-clip-cifar10] N={N}")
    _assert_fresh_cache_targets(
        args.cache_dir, args.save_aug_views,
        overwrite_cache=args.overwrite_cache)

    # ---- 2. load backbone (frozen) -----------------------------
    print(f"[extract-clip-cifar10] loading {args.clip_backbone}")
    backbone = CLIPBackbone(args.clip_backbone).to(args.device).eval()
    for p in backbone.parameters(): p.requires_grad = False
    H_v = backbone.vision_hidden_dim
    D_proj = backbone.projection_dim
    hf_provenance = _resolve_hf_provenance(args.clip_backbone, backbone.model)

    # Warm-up to get num_tokens
    dummy = _default_transform(args.image_size)(
        Image.fromarray(image_arrs[0])
    ).unsqueeze(0).to(args.device)
    _interp = (args.image_size != 224)
    v_out = backbone.vision_model(pixel_values=dummy, interpolate_pos_encoding=_interp)
    num_tokens = int(v_out.last_hidden_state.shape[1]) - 1
    print(f"[extract-clip-cifar10] num_tokens={num_tokens} (CLS stripped), H_v={H_v}, D_proj={D_proj}")

    # ---- 3. allocate output memmaps -------------------
    paths = {
        "visual_tokens": os.path.join(args.cache_dir, "visual_tokens.f16.npy"),
        "visual_global": os.path.join(args.cache_dir, "visual_global.f16.npy"),
        "text_part":     os.path.join(args.cache_dir, "text_part.f16.npy"),
        "has_text":      os.path.join(args.cache_dir, "has_text.bool.npy"),
        "image_ids":     os.path.join(args.cache_dir, "image_ids.json"),
        "meta":          os.path.join(args.cache_dir, "meta.json"),
    }
    visual_tokens_mm = np.lib.format.open_memmap(
        paths["visual_tokens"], mode="w+", dtype=np_dtype,
        shape=(N, num_tokens, H_v),
    )
    visual_global_mm = np.lib.format.open_memmap(
        paths["visual_global"], mode="w+", dtype=np_dtype,
        shape=(N, D_proj),
    )

    # ---- 4. extract visual features -------------------------
    from torch.utils.data import Dataset, DataLoader

    class _ArrDS(Dataset):
        def __init__(self, arrs, transform):
            self.arrs = arrs
            self.transform = transform
        def __len__(self): return len(self.arrs)
        def __getitem__(self, idx):
            im = Image.fromarray(self.arrs[idx])
            return idx, self.transform(im)

    tr = _default_transform(args.image_size)
    ds = _ArrDS(image_arrs, tr)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=int(args.num_workers), pin_memory=True,
                    persistent_workers=(int(args.num_workers) > 0))
    for idx_batch, pix_batch in tqdm(dl, total=(N + args.batch_size - 1) // args.batch_size, desc="visual"):
        pix_batch = pix_batch.to(args.device, non_blocking=True)
        v_feat = backbone.vision_model(pixel_values=pix_batch, interpolate_pos_encoding=_interp)
        token_feat = v_feat.last_hidden_state[:, 1:, :]
        g_feat = backbone.model.get_image_features(pixel_values=pix_batch, interpolate_pos_encoding=_interp)
        g_feat = coerce_pooled_to_tensor(g_feat)
        idx_np = idx_batch.numpy()
        visual_tokens_mm[idx_np] = token_feat.cpu().numpy().astype(np_dtype)
        visual_global_mm[idx_np] = g_feat.cpu().numpy().astype(np_dtype)

    visual_tokens_mm.flush(); visual_global_mm.flush()
    del visual_tokens_mm, visual_global_mm
    print(f"[extract-clip-cifar10] visual extracted.")

    # ---- 5. extract text features ---------------------------
    print(f"[extract-clip-cifar10] loading tokenizer {args.tokenizer_name}")
    tokenizer = build_clip_text_tokenizer(args.tokenizer_name)
    qwen = _load_qwen_cache(args.qwen_cache_path) if args.qwen_cache_path else {}
    print(f"[extract-clip-cifar10] qwen entries: {len(qwen)}")

    text_part = np.zeros((N, 6, D_proj), dtype=np_dtype)
    has_text = np.zeros((N,), dtype=bool)
    M = 6
    for start in tqdm(range(0, N, args.batch_size), desc="text"):
        end = min(start + args.batch_size, N)
        b = end - start
        prompts = []
        keep_mask = []
        for i in range(start, end):
            entry = qwen.get(image_ids[i], {})
            if entry:
                # Get all 6 slots in order
                per_slot = [entry.get(k, "") for k in SLOT_KEYS_V3V4]
                if all(p.strip() for p in per_slot):
                    prompts.extend(per_slot)
                    keep_mask.append(True)
                    continue
            prompts.extend([""] * M)
            keep_mask.append(False)
        if not any(keep_mask):
            continue
        enc = tokenizer(prompts, padding="max_length", truncation=True,
                        max_length=args.text_max_length, return_tensors="pt").to(args.device)
        with torch.no_grad():
            t_feat = backbone.model.get_text_features(**enc)   # tensor OR BaseModelOutputWithPooling
        t_feat = coerce_pooled_to_tensor(t_feat)               # [b*M, D_proj]
        t_feat = t_feat.reshape(b, M, D_proj).cpu().numpy().astype(np_dtype)
        for j, ok in enumerate(keep_mask):
            if ok:
                text_part[start + j] = t_feat[j]
                has_text[start + j] = True

    np.save(paths["text_part"], text_part)
    np.save(paths["has_text"], has_text)
    print(f"[extract-clip-cifar10] text: {has_text.sum()}/{N} valid")

    # ---- 6. save image_ids + meta ----------------------------
    with open(paths["image_ids"], "w") as f:
        json.dump(image_ids, f)
    meta = {
        "N": N, "num_tokens": num_tokens, "H_v": H_v, "D_proj": D_proj,
        "dtype": args.dtype, "backbone": args.clip_backbone,
        "tokenizer": args.tokenizer_name, "text_max_length": args.text_max_length,
        "source": "CIFAR10 via torchvision",
        "hf_provenance": hf_provenance,
        "augmentation_seed": (
            int(args.seed) if int(args.save_aug_views) > 0 else None),
        "canonical_transform": {
            "resize": [int(args.image_size), int(args.image_size)],
            "interpolation": "bicubic",
            "normalization": "openai_clip",
        },
        "augmentation_transform": ({
            "random_resized_crop": {"size": int(args.image_size),
                                     "scale": [0.5, 1.0],
                                     "interpolation": "bicubic"},
            "horizontal_flip_probability": 0.5,
            "color_jitter": [0.4, 0.4, 0.4, 0.1],
            "color_jitter_probability": 1.0,
            "grayscale_probability": 0.2,
            "normalization": "openai_clip",
            "gaussian_blur": False,
        } if args.save_aug_views > 0 else None),
    }
    # ---- 7. Optional aug views --------------------------------
    if args.save_aug_views > 0:
        for k in range(args.save_aug_views):
            print(f"[extract-clip-cifar10] extracting aug view {k}")
            aug_tr = _build_aug_transform(args.image_size)
            aug_tokens_mm = np.lib.format.open_memmap(
                os.path.join(
                    args.cache_dir, f"visual_tokens_aug{k}.f16.npy.partial"),
                mode="w+", dtype=np_dtype, shape=(N, num_tokens, H_v))
            aug_global_mm = np.lib.format.open_memmap(
                os.path.join(
                    args.cache_dir, f"visual_global_aug{k}.f16.npy.partial"),
                mode="w+", dtype=np_dtype, shape=(N, D_proj))
            ds_aug = _ArrDS(image_arrs, aug_tr)
            dl_aug = DataLoader(ds_aug, batch_size=args.batch_size, shuffle=False,
                                num_workers=int(args.num_workers), pin_memory=True,
                                worker_init_fn=_seed_worker,
                                generator=augmentation_generator)
            for idx_batch, pix_batch in tqdm(dl_aug, total=(N + args.batch_size - 1) // args.batch_size,
                                              desc=f"aug{k}"):
                pix_batch = pix_batch.to(args.device, non_blocking=True)
                v_feat = backbone.vision_model(pixel_values=pix_batch, interpolate_pos_encoding=_interp)
                token_feat = v_feat.last_hidden_state[:, 1:, :]
                g_feat = backbone.model.get_image_features(pixel_values=pix_batch, interpolate_pos_encoding=_interp)
                g_feat = coerce_pooled_to_tensor(g_feat)
                idx_np = idx_batch.numpy()
                aug_tokens_mm[idx_np] = token_feat.cpu().numpy().astype(np_dtype)
                aug_global_mm[idx_np] = g_feat.cpu().numpy().astype(np_dtype)
            aug_tokens_mm.flush(); aug_global_mm.flush()
            del aug_tokens_mm, aug_global_mm
            os.replace(
                os.path.join(
                    args.cache_dir, f"visual_tokens_aug{k}.f16.npy.partial"),
                os.path.join(args.cache_dir, f"visual_tokens_aug{k}.f16.npy"))
            os.replace(
                os.path.join(
                    args.cache_dir, f"visual_global_aug{k}.f16.npy.partial"),
                os.path.join(args.cache_dir, f"visual_global_aug{k}.f16.npy"))

    # Publish metadata only after every declared augmentation array completed.
    meta['save_aug_views'] = int(args.save_aug_views)
    _atomic_write_json(paths["meta"], meta)

    print(f"[extract-clip-cifar10] DONE. cache -> {args.cache_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
