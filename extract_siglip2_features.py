"""Pre-extract SigLIP2 visual + text features for CIFAR10.

Why:
    SigLIP2 is FROZEN (default `freeze_backbone=True`). The encoder pass is
    deterministic given (pixel_values, part_input_ids), so running it every
    epoch is wasted compute. Cache once, then training/extraction load the
    cached tensors and skip the encoder entirely.

Outputs (under ``--cache_dir``, default ``./cache/cifar10_siglip2/``):
    image_ids.json                list[str]  md5 hex per row    (length N)
    visual_tokens.f16.npy         fp16       [N, num_patches, H_v]
    visual_global.f16.npy         fp16       [N, D_proj]
    text_part.f16.npy             fp16       [N, 6, D_proj]
    has_text.bool.npy             bool       [N]   True = real Qwen text
    meta.json                     schema {N, num_patches, H_v, D_proj, ...}

Row-major order mirrors `iter_cifar10` (train 0..49999 then test 0..9999), so
row index = (split_offset + raw_index). The dataloader looks up by md5 hash
via image_ids.json for safety, not by index.

Usage:
    python extract_siglip2_features.py \\
        --cifar10_root ./dataset/CIFAR10 \\
        --qwen_cache_path ./cache/cifar10_qwen.jsonl \\
        --cache_dir ./cache/cifar10_siglip2 \\
        --batch_size 128
"""

from __future__ import annotations
import argparse
import json
import os
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from dna_utils import (
    extract_codebook_texts,
    build_siglip2_text_tokenizer,
    DEFAULT_FALLBACK_TEXT,
)
from models.pretrained_backbone import (
    SigLIP2Backbone,
    coerce_pooled_to_tensor,
    DEFAULT_BACKBONE,
)


# Mirror of preprocess_qwen_codebook_texts._cifar10_image_id and
# dataloaders._cifar10_image_id. Kept intentionally redundant so this script
# does not need to import a shared helper.
def _cifar10_image_id(arr: np.ndarray) -> str:
    import hashlib
    return hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


def _load_qwen_cache(path: str) -> dict:
    """Return {image_id: codebook_texts dict}."""
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, "r") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            iid = row.get("image_id")
            cb  = row.get("codebook_texts")
            if iid is not None and cb is not None:
                out[iid] = cb
    return out


def _iter_cifar10(root: str):
    from torchvision.datasets import CIFAR10
    rows: List[Tuple[str, np.ndarray]] = []
    for split, train in [("train", True), ("test", False)]:
        ds = CIFAR10(root, train=train, download=True)
        for i in range(len(ds.data)):
            arr = ds.data[i]
            iid = _cifar10_image_id(arr)
            rows.append((iid, arr))
    return rows


def _read_split_paths(split_txt: str) -> List[str]:
    """Read the leading relpath token from each non-empty line of `split_txt`."""
    out: List[str] = []
    with open(split_txt, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(line.split()[0])
    return out


def _iter_imagenet100(root: str, setting: str = "setting1") -> List[Tuple[str, str]]:
    """Return the union of (relpath, full_path) for setting1's train+test+database
    splits. Order: train, then test, then any database rows not already seen.

    Uses each split's leading relpath token as image_id (matches
    `dataloaders.ImgRtvDataset` and the Qwen JSONL cache key convention).
    """
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


# Match dataloaders.ImgRtvCIFAR10.default_transform.
_default_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="cifar10",
                    choices=["cifar10", "imagenet100", "pathlist"],
                    help="Dataset layout. cifar10 = md5-keyed in-memory uint8. "
                         "imagenet100 / pathlist = path-keyed; reads union of "
                         "<root>/<setting>/{train,test,database}.txt and decodes "
                         "image files on the fly. pathlist is the generic name; "
                         "imagenet100 is kept as a back-compat alias.")
    ap.add_argument("--cifar10_root", default="./dataset/CIFAR10")
    ap.add_argument("--imagenet100_root", default="./dataset/ImageNet100",
                    help="ImageNet100 root containing setting1/*.txt and image/*.JPEG.")
    ap.add_argument("--imagenet100_setting", default="setting1")
    ap.add_argument("--pathlist_root", default=None,
                    help="Generic path-keyed dataset root (Flickr25k / MSCOCO / etc.) "
                         "containing setting1/{train,test,database}.txt.")
    ap.add_argument("--pathlist_setting", default="setting1")
    ap.add_argument("--qwen_cache_path", default=None,
                    help="JSONL with codebook_texts per image_id (set to '' or omit "
                         "to skip text encoding). Default depends on --mode.")
    ap.add_argument("--cache_dir", default=None,
                    help="Output dir. Default depends on --mode.")
    ap.add_argument("--backbone_name", default=DEFAULT_BACKBONE)
    ap.add_argument("--tokenizer_name", default="google/siglip2-base-patch16-224")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--text_max_length", type=int, default=64)
    ap.add_argument("--save_text_tokens", action="store_true", default=False,
        help="Additionally save token-level text features ([N, 6, T, D_proj]) "
             "to `text_tokens.f16.npy` + token-level attention masks. Used by "
             "the model's visual-attention text-pooling block. ~6x the size of "
             "the pooled text cache so off by default.")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--num_workers", type=int, default=4,
                    help="Parallel workers for ImageNet PIL decode + transform.")
    args = ap.parse_args()

    # mode-aware defaults
    if args.mode == "pathlist" and args.pathlist_root is None:
        raise SystemExit("[extract --mode pathlist] need --pathlist_root.")
    if args.qwen_cache_path is None:
        if args.mode == "cifar10":
            args.qwen_cache_path = "./cache/cifar10_qwen.jsonl"
        elif args.mode == "imagenet100":
            args.qwen_cache_path = "./cache/imagenet100_qwen.jsonl"
        else:
            args.qwen_cache_path = ""    # caller must pass explicitly for pathlist
    if args.cache_dir is None:
        if args.mode == "cifar10":
            args.cache_dir = "./cache/cifar10_siglip2"
        elif args.mode == "imagenet100":
            args.cache_dir = "./cache/imagenet100_siglip2"
        else:
            # derive from pathlist_root: ./dataset/Flickr25k -> ./cache/Flickr25k_siglip2
            ds_name = os.path.basename(os.path.normpath(args.pathlist_root)).lower()
            args.cache_dir = f"./cache/{ds_name}_siglip2"

    os.makedirs(args.cache_dir, exist_ok=True)
    np_dtype = np.float16 if args.dtype == "float16" else np.float32

    # ---- 1. enumerate images (mode-specific) ------------------------------
    if args.mode == "cifar10":
        print("[extract] enumerating CIFAR10 ...")
        rows = _iter_cifar10(args.cifar10_root)
        N = len(rows)
        image_ids = [iid for iid, _ in rows]
        arrays    = [arr for _, arr in rows]                    # in-memory uint8 [32,32,3]
        image_paths = None
    elif args.mode == "imagenet100":
        print(f"[extract] enumerating ImageNet100 {args.imagenet100_setting} (train+test+database) ...")
        rows_in = _iter_imagenet100(args.imagenet100_root, args.imagenet100_setting)
        N = len(rows_in)
        image_ids   = [iid for iid, _ in rows_in]
        image_paths = [p   for _, p   in rows_in]
        arrays = None
    else:  # pathlist
        print(f"[extract] enumerating {args.pathlist_root} {args.pathlist_setting} (train+test+database) ...")
        rows_in = _iter_imagenet100(args.pathlist_root, args.pathlist_setting)
        N = len(rows_in)
        image_ids   = [iid for iid, _ in rows_in]
        image_paths = [p   for _, p   in rows_in]               # decode on-the-fly
        arrays = None
    print(f"[extract] {N} images total")

    # ---- 2. load backbone -------------------------------------------------
    print(f"[extract] loading backbone {args.backbone_name} on {args.device} ...")
    backbone = SigLIP2Backbone(args.backbone_name).to(args.device).eval()
    H_v    = backbone.vision_hidden_dim
    D_proj = backbone.projection_dim

    # warm-up forward to discover patch count for this checkpoint.
    # Path-keyed datasets may contain unreadable files (storage I/O errors);
    # try a few until one succeeds, fall back to a black image otherwise.
    if args.mode == "cifar10":
        dummy_pil = Image.fromarray(arrays[0])
    else:
        dummy_pil = None
        for _p in image_paths[:50]:
            try:
                dummy_pil = Image.open(_p).convert("RGB")
                break
            except Exception:
                continue
        if dummy_pil is None:
            print(f"[extract] WARNING: first 50 images all unreadable, "
                  f"falling back to a black 224x224 for warm-up.")
            dummy_pil = Image.new("RGB", (224, 224), 0)
    dummy = _default_transform(dummy_pil).unsqueeze(0).to(args.device)
    v_out = backbone.vision_model(pixel_values=dummy)
    num_tokens = int(v_out.last_hidden_state.shape[1])
    print(f"[extract] vision dims: num_tokens={num_tokens}, H_v={H_v}, D_proj={D_proj}")

    # ---- 3. allocate output memmaps --------------------------------------
    paths = {
        "visual_tokens": os.path.join(args.cache_dir, "visual_tokens.f16.npy"),
        "visual_global": os.path.join(args.cache_dir, "visual_global.f16.npy"),
        "text_part":     os.path.join(args.cache_dir, "text_part.f16.npy"),
        "has_text":      os.path.join(args.cache_dir, "has_text.bool.npy"),
        "image_ids":     os.path.join(args.cache_dir, "image_ids.json"),
        "meta":          os.path.join(args.cache_dir, "meta.json"),
    }
    print(f"[extract] visual_tokens shape -> [{N}, {num_tokens}, {H_v}]"
          f" ~ {N * num_tokens * H_v * 2 / 1e9:.1f} GB at fp16")
    visual_tokens_mm = np.lib.format.open_memmap(
        paths["visual_tokens"], mode="w+", dtype=np_dtype,
        shape=(N, num_tokens, H_v),
    )
    visual_global_mm = np.lib.format.open_memmap(
        paths["visual_global"], mode="w+", dtype=np_dtype,
        shape=(N, D_proj),
    )

    # ---- 4. extract VISUAL features --------------------------------------
    print(f"[extract] running vision tower (bs={args.batch_size}, workers={args.num_workers}) ...")
    bs = args.batch_size

    # For ImageNet we use a torch DataLoader to overlap PIL decode + transform
    # with GPU forward via num_workers. For CIFAR everything is already in
    # memory so the simple batch loop is faster.
    if args.mode == "cifar10":
        for start in tqdm(range(0, N, bs), desc="visual"):
            end = min(start + bs, N)
            batch_pil  = [Image.fromarray(arrays[i]) for i in range(start, end)]
            batch_pix  = torch.stack([_default_transform(im) for im in batch_pil]).to(args.device)
            v_feat = backbone.vision_model(pixel_values=batch_pix)
            token_feat = v_feat.last_hidden_state                            # [b, num_tokens, H_v]
            g_feat = backbone.model.get_image_features(pixel_values=batch_pix)
            g_feat = coerce_pooled_to_tensor(g_feat)
            visual_tokens_mm[start:end] = token_feat.detach().cpu().numpy().astype(np_dtype)
            visual_global_mm[start:end] = g_feat.detach().cpu().numpy().astype(np_dtype)
    else:
        from torch.utils.data import Dataset as _TDS, DataLoader as _TDL

        # MSCOCO storage on /data has reproducible per-file Input/output
        # errors on a small fraction of JPEGs. We tolerate them by returning
        # a black image + an `ok=False` flag; downstream the row gets zero
        # features and is recorded in meta.json's `failed_image_indices`.
        class _PathDS(_TDS):
            def __init__(self, paths, transform):
                self.paths = paths
                self.transform = transform
            def __len__(self): return len(self.paths)
            def __getitem__(self, idx):
                try:
                    im = Image.open(self.paths[idx]).convert("RGB")
                    return idx, self.transform(im), 1
                except Exception:
                    # 224x224 black tensor with same normalization as a real image
                    return idx, self.transform(Image.new("RGB", (224, 224), 0)), 0

        ds = _PathDS(image_paths, _default_transform)
        dl = _TDL(ds, batch_size=bs, shuffle=False,
                  num_workers=int(args.num_workers), pin_memory=True,
                  persistent_workers=(int(args.num_workers) > 0))
        n_failed = 0
        failed_indices = []
        for idx_batch, pix_batch, ok_batch in tqdm(dl, total=(N + bs - 1) // bs, desc="visual"):
            pix_batch = pix_batch.to(args.device, non_blocking=True)
            v_feat = backbone.vision_model(pixel_values=pix_batch)
            token_feat = v_feat.last_hidden_state                            # [b, num_tokens, H_v]
            g_feat = backbone.model.get_image_features(pixel_values=pix_batch)
            g_feat = coerce_pooled_to_tensor(g_feat)
            tok_np = token_feat.detach().cpu().numpy().astype(np_dtype)
            g_np   = g_feat   .detach().cpu().numpy().astype(np_dtype)
            # zero out rows where decode failed -- prevents black-image features
            # from polluting retrieval (they end up as outliers)
            ok_np = ok_batch.numpy().astype(bool)
            if (~ok_np).any():
                tok_np[~ok_np] = 0
                g_np  [~ok_np] = 0
                bad = idx_batch.numpy()[~ok_np]
                failed_indices.extend(int(x) for x in bad)
                n_failed += int((~ok_np).sum())
            ids = idx_batch.numpy()
            visual_tokens_mm[ids] = tok_np
            visual_global_mm[ids] = g_np
    visual_tokens_mm.flush(); del visual_tokens_mm
    visual_global_mm.flush(); del visual_global_mm

    # ---- 5. text features (per-part global, 6 per image) -----------------
    text_part_mm = np.lib.format.open_memmap(
        paths["text_part"], mode="w+", dtype=np_dtype,
        shape=(N, 6, D_proj),
    )
    has_text_mm = np.lib.format.open_memmap(
        paths["has_text"], mode="w+", dtype=np.bool_, shape=(N,),
    )
    # Optional token-level text cache (Option B / v22b cross-attention path).
    text_tokens_mm = None
    text_token_mask_mm = None
    if args.save_text_tokens:
        T_max = int(args.text_max_length)
        paths["text_tokens"]      = os.path.join(args.cache_dir, "text_tokens.f16.npy")
        paths["text_token_mask"]  = os.path.join(args.cache_dir, "text_token_mask.bool.npy")
        print(f"[extract] text_tokens shape -> [{N}, 6, {T_max}, {D_proj}] "
              f"~ {N * 6 * T_max * D_proj * 2 / 1e9:.2f} GB at fp16")
        text_tokens_mm = np.lib.format.open_memmap(
            paths["text_tokens"], mode="w+", dtype=np_dtype,
            shape=(N, 6, T_max, D_proj),
        )
        text_token_mask_mm = np.lib.format.open_memmap(
            paths["text_token_mask"], mode="w+", dtype=np.bool_,
            shape=(N, 6, T_max),
        )

    qwen = _load_qwen_cache(args.qwen_cache_path) if args.qwen_cache_path else {}
    print(f"[extract] qwen cache rows: {len(qwen)}")

    tokenizer = build_siglip2_text_tokenizer(args.tokenizer_name)
    fallback_texts = [DEFAULT_FALLBACK_TEXT] * 6

    # build flat text list and a "real" mask, then encode in batches.
    print("[extract] building text rows ...")
    flat_texts: List[str] = []
    has_real:   List[bool] = []
    for iid in image_ids:
        cb = qwen.get(iid)
        if cb is None:
            flat_texts.extend(fallback_texts)
            has_real.append(False)
        else:
            try:
                texts = extract_codebook_texts({"codebook_texts": cb})
            except Exception:
                texts = list(fallback_texts)
                has_real.append(False)
                flat_texts.extend(texts)
                continue
            flat_texts.extend(texts)
            has_real.append(True)
    has_text_mm[:] = np.array(has_real, dtype=np.bool_)
    has_text_mm.flush(); del has_text_mm

    print(f"[extract] running text tower over {len(flat_texts)} rows (bs={bs * 6}) ...")
    enc_bs = bs * 6  # one text-batch per image-batch
    for start_t in tqdm(range(0, len(flat_texts), enc_bs), desc="text"):
        end_t = min(start_t + enc_bs, len(flat_texts))
        chunk = flat_texts[start_t:end_t]
        enc = tokenizer(
            chunk, padding="max_length", truncation=True,
            max_length=args.text_max_length, return_tensors="pt",
            return_attention_mask=True,
        )
        input_ids = enc["input_ids"].to(args.device)
        if "attention_mask" in enc:
            attn = enc["attention_mask"].to(args.device)
        else:
            pad_id = getattr(tokenizer, "pad_token_id", None)
            attn = (input_ids != pad_id).long() if pad_id is not None else torch.ones_like(input_ids)
        t_feat = backbone.model.get_text_features(input_ids=input_ids, attention_mask=attn)
        t_feat = coerce_pooled_to_tensor(t_feat)
        # write back to [N, 6, D_proj] view
        for k, j in enumerate(range(start_t, end_t)):
            row, part = divmod(j, 6)
            text_part_mm[row, part, :] = t_feat[k].detach().cpu().numpy().astype(np_dtype)

        # Optional: token-level features (last_hidden_state of text encoder
        # projected to D_proj). Saves a per-token [T, D_proj] tensor for each
        # of the 6 slots so the model can cross-attend visual <- text tokens
        # at training time. The projection used here matches SigLIP2's
        # visual / text shared embedding space.
        if text_tokens_mm is not None:
            with torch.no_grad():
                tm_out = backbone.model.text_model(
                    input_ids=input_ids, attention_mask=attn,
                )
                last_hidden = tm_out.last_hidden_state                 # [bs6, T, H_t]
                # Project each token into the SigLIP2 shared D_proj space
                # using the same text projection as `get_text_features`.
                # SigLIP2 has `text_projection` attribute on the wrapping
                # SiglipModel; fall back to the text_model's own head.
                proj = getattr(backbone.model, "text_projection", None)
                if proj is not None:
                    tok_feat = proj(last_hidden)                        # [bs6, T, D_proj]
                else:
                    tok_feat = last_hidden                              # already shared dim in some impls
            tok_np  = tok_feat.detach().cpu().numpy().astype(np_dtype)  # [bs6, T, D]
            attn_np = attn.detach().cpu().numpy().astype(np.bool_)      # [bs6, T]
            for k, j in enumerate(range(start_t, end_t)):
                row, part = divmod(j, 6)
                text_tokens_mm[row, part]     = tok_np[k]
                text_token_mask_mm[row, part] = attn_np[k]
    text_part_mm.flush(); del text_part_mm
    if text_tokens_mm is not None:
        text_tokens_mm.flush(); del text_tokens_mm
        text_token_mask_mm.flush(); del text_token_mask_mm

    # ---- 6. write tiny side-files ----------------------------------------
    with open(paths["image_ids"], "w") as f:
        json.dump(image_ids, f)
    meta = {
        "N":           int(N),
        "num_tokens":  int(num_tokens),
        "H_v":         int(H_v),
        "D_proj":      int(D_proj),
        "dtype":       str(np_dtype.__name__),
        "backbone":    args.backbone_name,
        "tokenizer":   args.tokenizer_name,
        "text_max_length": int(args.text_max_length),
    }
    if args.mode != "cifar10":
        meta["n_failed_image_decode"] = int(locals().get("n_failed", 0))
        meta["failed_image_indices"]  = list(locals().get("failed_indices", []))[:5000]
        if int(locals().get("n_failed", 0)) > 0:
            print(f"[extract] WARNING: {meta['n_failed_image_decode']} images "
                  f"failed PIL decode (zeroed out, recorded in meta.json).")
    with open(paths["meta"], "w") as f:
        json.dump(meta, f, indent=2)

    print("[extract] done:")
    for k, p in paths.items():
        sz_mb = os.path.getsize(p) / 1e6 if os.path.exists(p) else 0.0
        print(f"  {k:<14s} {p}  ({sz_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
