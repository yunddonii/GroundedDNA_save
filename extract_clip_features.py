"""Pre-extract OpenAI-CLIP visual + text features for path-keyed datasets.

This is the CLIP-backbone twin of ``extract_siglip2_features.py``. Same cache
schema, same image_ids convention, same V4-caption text path, so the
downstream dataloader (``_SigLIP2FeatureCache``) and the training pipeline
work for either backbone family.

Why a parallel script (not a `--backbone {siglip2, clip}` switch inside
the existing extractor):
    - Image normalization differs: CLIP uses its own mean/std (close to but
      not identical to ImageNet stats SigLIP2 uses here). Embedding the
      backbone-specific preprocessing into a separate file matches the
      project's "new file centric" change policy and avoids a hidden
      regression on SigLIP2 runs that already exist on disk.
    - CLIP's vision tower returns `[B, 197, H_v]` (196 patches + 1 [CLS]).
      We strip the CLS token so the cached `visual_tokens` shape is
      `[N, 196, H_v]`, matching the SigLIP2 cache schema. This keeps every
      shape downstream identical to the SigLIP2 path (the Sinkhorn router
      and codebook quantizer are dim-agnostic but the consistency matters
      for ablation comparisons).

Outputs (under ``--cache_dir``, default derived from --pathlist_root):
    image_ids.json                list[str]    relpath per row    (length N)
    visual_tokens.f16.npy         fp16         [N, 196, H_v=768]
    visual_global.f16.npy         fp16         [N, D_proj=512]
    text_part.f16.npy             fp16         [N, 6, D_proj=512]
    has_text.bool.npy             bool         [N]   True = real Qwen text
    meta.json                     schema dict with backbone="openai/clip-vit-base-patch16"
    visual_tokens_aug{i}.f16.npy  fp16 (when --save_aug_views > 0)
    visual_global_aug{i}.f16.npy  fp16 (when --save_aug_views > 0)

Row-major order mirrors ``extract_siglip2_features.py`` (union of
``setting/{train, test, database}.txt`` in that order, dedup). Lookup keys
are ``os.path.relpath(p, pathlist_root)`` — same convention as
``_build_pathkeyed_cache_row_map`` in ``dataloaders.py``.

Usage (Flickr25k V4):
    python extract_clip_features.py \\
        --mode pathlist \\
        --pathlist_root ./dataset/Flickr25k \\
        --pathlist_setting setting1 \\
        --qwen_cache_path ./cache/flickr25k_qwen_v4.jsonl \\
        --cache_dir ./cache/flickr25k_clip_v4plus \\
        --save_aug_views 2 \\
        --batch_size 128
"""

from __future__ import annotations
import argparse
import json
import os
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from dna_utils import (
    extract_codebook_texts,
    build_clip_text_tokenizer,
    DEFAULT_CLIP_TOKENIZER_NAME,
    DEFAULT_FALLBACK_TEXT,
)
from models.pretrained_backbone import coerce_pooled_to_tensor
from models.pretrained_backbone_clip import (
    CLIPBackbone,
    DEFAULT_CLIP_BACKBONE,
)


# --------------------------------------------------------------- CLIP preproc
# These are the official OpenAI CLIP image normalization stats — different
# from ImageNet (which SigLIP2 happened to inherit) and what every CLIP-based
# unsupervised hashing baseline (CIBHash / CIMON / MLS3RDUH) uses.
CLIP_PIXEL_MEAN = [0.48145466, 0.4578275,  0.40821073]
CLIP_PIXEL_STD  = [0.26862954, 0.26130258, 0.27577711]


def _default_transform(image_size: int = 224) -> "transforms.Compose":
    # Match the SigLIP2 extractor's deterministic transform shape (resize to
    # NxN, no crop / aug) so the cache geometry stays consistent across
    # backbones; only the normalization (and resolution) changes.
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_PIXEL_MEAN, CLIP_PIXEL_STD),
    ])


def _build_aug_transform(image_size: int = 224) -> "transforms.Compose":
    """SimCLR / CIBHash-style augmentation for paired-aug NtXent views.

    Mirror of ``extract_siglip2_features._build_aug_transform`` but with the
    CLIP normalization stats. Used by ``--save_aug_views K``.
    """
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_PIXEL_MEAN, CLIP_PIXEL_STD),
    ])


# --------------------------------------------------------------- IO helpers

def _load_qwen_cache(path: str) -> dict:
    """Return {image_id: codebook_texts dict}.

    Mirror of ``extract_siglip2_features._load_qwen_cache``; copied here so
    this script stays usable without modifying the SigLIP2 extractor.
    """
    if not path or not os.path.exists(path):
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


def _iter_pathlist(root: str, setting: str = "setting1") -> List[Tuple[str, str]]:
    """Return the union of (relpath, full_path) for setting's train+test+database.

    Order: train, then test, then any database rows not already seen.
    Uses each split's leading relpath token as image_id (same convention as
    ``dataloaders._build_pathkeyed_cache_row_map``).
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


# --------------------------------------------------------------- main

@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="pathlist",
                    choices=["pathlist"],
                    help="Dataset layout. Only pathlist (Flickr25k / MSCOCO / "
                         "ImageNet100) is supported here; the CIFAR10 path is "
                         "covered by the SigLIP2 extractor only.")
    ap.add_argument("--pathlist_root", required=True,
                    help="Dataset root containing setting1/{train,test,database}.txt "
                         "(e.g. ./dataset/Flickr25k).")
    ap.add_argument("--pathlist_setting", default="setting1")
    ap.add_argument("--qwen_cache_path", default=None,
                    help="JSONL with codebook_texts per image_id (V4 prompt). "
                         "Empty / omitted -> only visual features written, all "
                         "has_text rows are False.")
    ap.add_argument("--cache_dir", default=None,
                    help="Output dir. Default = ./cache/<dsname>_clip.")
    ap.add_argument("--clip_backbone",
                    default=DEFAULT_CLIP_BACKBONE,
                    help="HuggingFace CLIP model checkpoint name (default: %(default)s).")
    ap.add_argument("--tokenizer_name",
                    default=DEFAULT_CLIP_TOKENIZER_NAME,
                    help="HuggingFace tokenizer name (default: %(default)s).")
    ap.add_argument("--batch_size",  type=int, default=128)
    ap.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--text_max_length", type=int, default=64,
                    help="Max token length for the CLIP text tokenizer. CLIP's "
                         "context window is 77; we keep 64 by default to match "
                         "the SigLIP2 cache.")
    ap.add_argument("--image_size", type=int, default=224,
                    help="Input image side (square). Default 224 matches CLIP "
                         "ViT-B/16's pretraining resolution. Set to 336 for "
                         "higher-resolution extraction (more patches per image, "
                         "better for fine-grained datasets). Position embeddings "
                         "are bicubic-interpolated by CLIPVisionModel when the "
                         "input grid differs from the pretrained 14x14.")
    ap.add_argument("--dtype",       default="float16", choices=["float16", "float32"])
    ap.add_argument("--num_workers", type=int, default=4,
                    help="Parallel workers for PIL decode + transform.")
    ap.add_argument("--save_aug_views", type=int, default=0,
                    help="If > 0, additionally cache K paired-aug views per "
                         "image (SimCLR/CIBHash transform). Saved as "
                         "visual_{tokens,global}_aug{0..K-1}.f16.npy. Set to 2 "
                         "for v81a paired-aug NtXent.")
    ap.add_argument("--aug_only", action="store_true", default=False,
                    help="Skip the deterministic visual + text extraction and "
                         "ONLY produce the aug views (assumes the deterministic "
                         "cache already exists).")
    args = ap.parse_args()

    if not args.cache_dir:
        ds_name = os.path.basename(os.path.normpath(args.pathlist_root)).lower()
        args.cache_dir = f"./cache/{ds_name}_clip"

    os.makedirs(args.cache_dir, exist_ok=True)
    np_dtype = np.float16 if args.dtype == "float16" else np.float32

    # ---- 1. enumerate images ---------------------------------------------
    print(f"[extract-clip] enumerating {args.pathlist_root} {args.pathlist_setting} "
          f"(train+test+database) ...")
    rows_in = _iter_pathlist(args.pathlist_root, args.pathlist_setting)
    N = len(rows_in)
    if N == 0:
        raise SystemExit(f"[extract-clip] no images discovered under {args.pathlist_root}/"
                         f"{args.pathlist_setting}")
    image_ids   = [iid for iid, _ in rows_in]
    image_paths = [p   for _, p   in rows_in]
    print(f"[extract-clip] {N} images total")

    # ---- 2. load backbone (FROZEN) ---------------------------------------
    print(f"[extract-clip] loading backbone {args.clip_backbone} on {args.device} ...")
    backbone = CLIPBackbone(args.clip_backbone).to(args.device).eval()
    for p in backbone.parameters():                                  # FROZEN
        p.requires_grad = False
    H_v    = backbone.vision_hidden_dim
    D_proj = backbone.projection_dim
    print(f"[extract-clip] backbone dims: vision_hidden={H_v}, projection={D_proj}")

    # ---- 2b. warm-up forward to discover num_tokens (= patches; CLS dropped) ---
    dummy_pil = None
    for _p in image_paths[:50]:
        try:
            dummy_pil = Image.open(_p).convert("RGB")
            break
        except Exception:
            continue
    if dummy_pil is None:
        print("[extract-clip] WARNING: first 50 images all unreadable, "
              "falling back to a black 224x224 for warm-up.")
        dummy_pil = Image.new("RGB", (224, 224), 0)
    pre = _default_transform(args.image_size)
    dummy = pre(dummy_pil).unsqueeze(0).to(args.device)               # [1, 3, S, S]
    # interpolate_pos_encoding=True so CLIPVisionModel resamples its 14x14
    # position embeddings to the actual patch grid when --image_size != 224.
    _interp = (args.image_size != 224)
    v_out = backbone.vision_model(pixel_values=dummy, interpolate_pos_encoding=_interp)
    raw_n_tokens = int(v_out.last_hidden_state.shape[1])              # 1 CLS + N patches
    num_tokens   = int(raw_n_tokens - 1)
    _expected = (args.image_size // 16) ** 2
    assert num_tokens == _expected, (
        f"[extract-clip] expected {_expected} patches after stripping CLS for "
        f"image_size={args.image_size} on patch16 backbone, got {num_tokens}."
    )
    print(f"[extract-clip] vision tokens: raw={raw_n_tokens}, after CLS-strip={num_tokens} "
          f"(H_v={H_v}, D_proj={D_proj})")

    # ---- 3. allocate output memmaps --------------------------------------
    paths = {
        "visual_tokens": os.path.join(args.cache_dir, "visual_tokens.f16.npy"),
        "visual_global": os.path.join(args.cache_dir, "visual_global.f16.npy"),
        "text_part":     os.path.join(args.cache_dir, "text_part.f16.npy"),
        "has_text":      os.path.join(args.cache_dir, "has_text.bool.npy"),
        "image_ids":     os.path.join(args.cache_dir, "image_ids.json"),
        "meta":          os.path.join(args.cache_dir, "meta.json"),
    }
    bs = args.batch_size
    if not args.aug_only:
        print(f"[extract-clip] visual_tokens shape -> [{N}, {num_tokens}, {H_v}]"
              f" ~ {N * num_tokens * H_v * 2 / 1e9:.1f} GB at fp16")
        visual_tokens_mm = np.lib.format.open_memmap(
            paths["visual_tokens"], mode="w+", dtype=np_dtype,
            shape=(N, num_tokens, H_v),
        )
        visual_global_mm = np.lib.format.open_memmap(
            paths["visual_global"], mode="w+", dtype=np_dtype,
            shape=(N, D_proj),
        )
    else:
        print(f"[extract-clip] aug_only=True -> reusing existing visual_global cache.")
        visual_tokens_mm = None
        visual_global_mm = None

    # ---- 4. extract VISUAL features --------------------------------------
    # Same _PathDS loader pattern as extract_siglip2_features.py: tolerant of
    # unreadable JPEGs by returning a black image + ok=0 flag. The CLS token
    # is stripped here so the cached shape is [N, 196, H_v].
    from torch.utils.data import Dataset as _TDS, DataLoader as _TDL

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
                return idx, self.transform(Image.new("RGB", (224, 224), 0)), 0

    n_failed = 0
    failed_indices: List[int] = []
    if not args.aug_only:
        ds = _PathDS(image_paths, _default_transform(args.image_size))
        dl = _TDL(ds, batch_size=bs, shuffle=False,
                  num_workers=int(args.num_workers), pin_memory=True,
                  persistent_workers=(int(args.num_workers) > 0))
        for idx_batch, pix_batch, ok_batch in tqdm(
            dl, total=(N + bs - 1) // bs, desc="visual",
        ):
            pix_batch = pix_batch.to(args.device, non_blocking=True)
            v_feat = backbone.vision_model(pixel_values=pix_batch, interpolate_pos_encoding=_interp)
            # [b, 197, H_v] -> drop CLS -> [b, 196, H_v]
            token_feat = v_feat.last_hidden_state[:, 1:, :]
            assert token_feat.shape[1] == num_tokens, (
                f"[extract-clip] unexpected token count after CLS strip: "
                f"{tuple(token_feat.shape)}, expected ({pix_batch.shape[0]}, "
                f"{num_tokens}, {H_v})."
            )
            g_feat = backbone.model.get_image_features(pixel_values=pix_batch, interpolate_pos_encoding=_interp)
            g_feat = coerce_pooled_to_tensor(g_feat)                  # [b, D_proj]
            tok_np = token_feat.detach().cpu().numpy().astype(np_dtype)
            g_np   = g_feat   .detach().cpu().numpy().astype(np_dtype)
            ok_np  = ok_batch.numpy().astype(bool)
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

    # ---- 4b. optional augmented visual views -----------------------------
    if int(args.save_aug_views) > 0:
        K = int(args.save_aug_views)
        aug_paths: list = []
        aug_global_mms: list = []
        aug_token_mms:  list = []
        for i in range(K):
            pg = os.path.join(args.cache_dir, f"visual_global_aug{i}.f16.npy")
            pt = os.path.join(args.cache_dir, f"visual_tokens_aug{i}.f16.npy")
            mm_g = np.lib.format.open_memmap(
                pg, mode="w+", dtype=np_dtype, shape=(N, D_proj),
            )
            mm_t = np.lib.format.open_memmap(
                pt, mode="w+", dtype=np_dtype, shape=(N, num_tokens, H_v),
            )
            aug_paths.append(pg); aug_paths.append(pt)
            aug_global_mms.append(mm_g); aug_token_mms.append(mm_t)

        aug_transform = _build_aug_transform(args.image_size)
        print(f"[extract-clip] running vision tower {K} extra times with aug transform "
              f"(bs={bs}, workers={args.num_workers}) -- saving BOTH "
              f"visual_global_aug* and visual_tokens_aug* ...")

        class _AugPathDS(_TDS):
            def __init__(self, paths, transform):
                self.paths = paths
                self.transform = transform
            def __len__(self): return len(self.paths)
            def __getitem__(self, idx):
                try:
                    im = Image.open(self.paths[idx]).convert("RGB")
                    return idx, self.transform(im), 1
                except Exception:
                    return idx, self.transform(Image.new("RGB", (224, 224), 0)), 0

        for view_idx in range(K):
            aug_ds = _AugPathDS(image_paths, aug_transform)
            aug_dl = _TDL(aug_ds, batch_size=bs, shuffle=False,
                          num_workers=int(args.num_workers), pin_memory=True,
                          persistent_workers=(int(args.num_workers) > 0))
            for idx_batch, pix_batch, ok_batch in tqdm(
                aug_dl, total=(N + bs - 1) // bs, desc=f"aug{view_idx}",
            ):
                pix_batch = pix_batch.to(args.device, non_blocking=True)
                v_feat = backbone.vision_model(pixel_values=pix_batch, interpolate_pos_encoding=_interp)
                token_feat = v_feat.last_hidden_state[:, 1:, :]      # [b, 196, H_v]
                g_feat = backbone.model.get_image_features(pixel_values=pix_batch, interpolate_pos_encoding=_interp)
                g_feat = coerce_pooled_to_tensor(g_feat)
                g_np = g_feat   .detach().cpu().numpy().astype(np_dtype)
                t_np = token_feat.detach().cpu().numpy().astype(np_dtype)
                ok_np = ok_batch.numpy().astype(bool)
                if (~ok_np).any():
                    g_np[~ok_np] = 0
                    t_np[~ok_np] = 0
                ids = idx_batch.numpy()
                aug_global_mms[view_idx][ids] = g_np
                aug_token_mms [view_idx][ids] = t_np

        for mm in aug_global_mms + aug_token_mms:
            mm.flush()
        del aug_global_mms, aug_token_mms
        for p in aug_paths:
            paths[os.path.basename(p).replace(".f16.npy", "")] = p

    # ---- 4c. aug_only short-circuit --------------------------------------
    if args.aug_only:
        print("[extract-clip] aug_only=True -> skipping text extraction.")
        if int(args.save_aug_views) > 0:
            meta_p = os.path.join(args.cache_dir, "meta.json")
            if os.path.exists(meta_p):
                with open(meta_p, "r") as f:
                    meta = json.load(f)
            else:
                meta = {}
            meta["save_aug_views"] = int(args.save_aug_views)
            with open(meta_p, "w") as f:
                json.dump(meta, f, indent=2)
        print("[extract-clip] done (aug_only):")
        for k, p in paths.items():
            sz_mb = os.path.getsize(p) / 1e6 if os.path.exists(p) else 0.0
            print(f"  {k:<22s} {p}  ({sz_mb:.1f} MB)")
        return 0

    # ---- 5. text features (per-part global, 6 per image) -----------------
    text_part_mm = np.lib.format.open_memmap(
        paths["text_part"], mode="w+", dtype=np_dtype,
        shape=(N, 6, D_proj),
    )
    has_text_mm  = np.lib.format.open_memmap(
        paths["has_text"], mode="w+", dtype=np.bool_, shape=(N,),
    )

    qwen = _load_qwen_cache(args.qwen_cache_path) if args.qwen_cache_path else {}
    print(f"[extract-clip] qwen cache rows: {len(qwen)}")

    tokenizer = build_clip_text_tokenizer(args.tokenizer_name)
    fallback_texts = [DEFAULT_FALLBACK_TEXT] * 6

    print("[extract-clip] building text rows ...")
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

    print(f"[extract-clip] running text tower over {len(flat_texts)} rows "
          f"(bs={bs * 6}) ...")
    enc_bs = bs * 6                                                   # one text-batch per image-batch
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
        t_feat = coerce_pooled_to_tensor(t_feat)                      # [bs6, D_proj]
        # write back to [N, 6, D_proj] view
        for k, j in enumerate(range(start_t, end_t)):
            row, part = divmod(j, 6)
            text_part_mm[row, part, :] = t_feat[k].detach().cpu().numpy().astype(np_dtype)

    text_part_mm.flush(); del text_part_mm

    # ---- 6. write tiny side-files ----------------------------------------
    with open(paths["image_ids"], "w") as f:
        json.dump(image_ids, f)
    meta = {
        "N":           int(N),
        "num_tokens":  int(num_tokens),
        "H_v":         int(H_v),
        "D_proj":      int(D_proj),
        "dtype":       str(np_dtype.__name__),
        "backbone":    args.clip_backbone,
        "tokenizer":   args.tokenizer_name,
        "text_max_length": int(args.text_max_length),
        "n_failed_image_decode": int(n_failed),
        "failed_image_indices":  list(failed_indices)[:5000],
        "save_aug_views": int(args.save_aug_views),
        "text_part_source": "clip",
        "qwen_cache":  os.path.basename(args.qwen_cache_path) if args.qwen_cache_path else "",
        "image_normalize_mean": CLIP_PIXEL_MEAN,
        "image_normalize_std":  CLIP_PIXEL_STD,
        "cls_token_stripped":   True,
        "backbone_family":      "clip",
    }
    if n_failed > 0:
        print(f"[extract-clip] WARNING: {n_failed} images failed PIL decode "
              f"(zeroed out, recorded in meta.json).")
    with open(paths["meta"], "w") as f:
        json.dump(meta, f, indent=2)

    print("[extract-clip] done:")
    for k, p in paths.items():
        sz_mb = os.path.getsize(p) / 1e6 if os.path.exists(p) else 0.0
        print(f"  {k:<22s} {p}  ({sz_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
