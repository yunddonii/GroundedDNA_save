from typing import Callable, Optional, T_co, Any
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import os
import json
import re
from PIL import Image
import pandas as pd

import cv2


# ---------------------------------------------------------------- siglip2 feature cache

class _SigLIP2FeatureCache:
    """Memmap-backed reader for SigLIP2 visual + text feature caches.

    Built once from a directory produced by `extract_siglip2_features.py`.
    Looks up by md5 image_id (md5 of raw uint8 CIFAR10 bytes, 16 hex chars)
    and yields per-row tensors as torch.float32 for downstream consumers.
    """

    def __init__(self, cache_dir: str):
        if not os.path.isdir(cache_dir):
            raise FileNotFoundError(f"[siglip2-cache] dir not found: {cache_dir}")
        meta_p   = os.path.join(cache_dir, "meta.json")
        ids_p    = os.path.join(cache_dir, "image_ids.json")
        vt_p     = os.path.join(cache_dir, "visual_tokens.f16.npy")
        vg_p     = os.path.join(cache_dir, "visual_global.f16.npy")
        tp_p     = os.path.join(cache_dir, "text_part.f16.npy")
        ht_p     = os.path.join(cache_dir, "has_text.bool.npy")
        for p in (meta_p, ids_p, vt_p, vg_p, tp_p, ht_p):
            if not os.path.exists(p):
                raise FileNotFoundError(f"[siglip2-cache] missing {p}")
        with open(meta_p, "r") as f:
            self.meta = json.load(f)
        with open(ids_p, "r") as f:
            ids = json.load(f)
        self.image_ids = list(ids)
        self.id_to_row = {iid: i for i, iid in enumerate(self.image_ids)}
        # mmap_mode='r' -> file stays on disk, no full RAM load
        self.visual_tokens = np.load(vt_p, mmap_mode="r")
        self.visual_global = np.load(vg_p, mmap_mode="r")
        self.text_part     = np.load(tp_p, mmap_mode="r")
        self.has_text      = np.load(ht_p, mmap_mode="r")
        # Cache metadata is part of the experiment definition.  In particular,
        # historical FAIR/local-crop caches claimed a token-only intervention
        # while their generator silently replaced the full-image global with a
        # selected-crop mean.  Reject those caches rather than continuing to
        # report a confounded single-delta experiment.
        declared_geometry = {
            "N": len(self.image_ids),
            "num_tokens": int(self.visual_tokens.shape[1]),
            "H_v": int(self.visual_tokens.shape[2]),
            "D_proj": int(self.visual_global.shape[1]),
        }
        geometry_mismatches = {
            key: (self.meta.get(key), actual)
            for key, actual in declared_geometry.items()
            if key in self.meta and int(self.meta[key]) != actual
        }
        if geometry_mismatches:
            raise ValueError(
                "[siglip2-cache] tensor geometry differs from meta.json: "
                f"{geometry_mismatches}"
            )
        first_dims = {
            "visual_tokens": int(self.visual_tokens.shape[0]),
            "visual_global": int(self.visual_global.shape[0]),
            "text_part": int(self.text_part.shape[0]),
            "has_text": int(self.has_text.shape[0]),
        }
        bad_first_dims = {
            name: size for name, size in first_dims.items()
            if size != len(self.image_ids)
        }
        if bad_first_dims:
            raise ValueError(
                "[siglip2-cache] feature rows differ from image_ids.json: "
                f"{bad_first_dims} vs {len(self.image_ids)}"
            )
        declared_local_aug_views = None
        if (
            "local_crops_L" in self.meta or "local_crops_K" in self.meta
        ):
            if self.meta.get("visual_global_source") != "donor_full_image":
                raise ValueError(
                    "[siglip2-cache] refusing legacy local-crop cache whose "
                    "global feature is not provenance-bound to the donor "
                    "full image; rebuild it with extract_clip_local_crops.py"
                )
            try:
                declared_local_aug_views = int(
                    self.meta.get("save_aug_views", 0)
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "[siglip2-cache] local-crop save_aug_views must be an "
                    "integer"
                ) from exc
            if declared_local_aug_views < 0:
                raise ValueError(
                    "[siglip2-cache] local-crop save_aug_views must be "
                    "non-negative"
                )
            expected_global_files = ["visual_global.f16.npy"] + [
                f"visual_global_aug{view}.f16.npy"
                for view in range(declared_local_aug_views)
            ]
            if self.meta.get("visual_global_files") != expected_global_files:
                raise ValueError(
                    "[siglip2-cache] local-crop global-file provenance is "
                    "missing or inconsistent; rebuild the cache"
                )
            donor_ref = self.meta.get("donor_cache")
            if not isinstance(donor_ref, str) or not donor_ref.strip():
                raise ValueError(
                    "[siglip2-cache] local-crop donor_cache provenance is "
                    "missing"
                )
            donor_dir = os.path.realpath(donor_ref)
            cache_real = os.path.realpath(cache_dir)
            if donor_dir == cache_real or not os.path.isdir(donor_dir):
                raise ValueError(
                    "[siglip2-cache] local-crop donor_cache does not resolve "
                    f"to a distinct cache directory: {donor_ref!r}"
                )
            donor_ids_p = os.path.join(donor_dir, "image_ids.json")
            if not os.path.isfile(donor_ids_p):
                raise FileNotFoundError(
                    "[siglip2-cache] local-crop donor image_ids.json is missing: "
                    f"{donor_ids_p}"
                )
            with open(donor_ids_p, "r") as f:
                donor_ids = json.load(f)
            if list(donor_ids) != self.image_ids:
                raise ValueError(
                    "[siglip2-cache] local-crop image_ids do not match the "
                    "declared donor cache"
                )

            # Local crops are a visual-token-only intervention.  Content
            # equality is insufficient here: a copied or stale text tensor can
            # silently preserve rows from a previous donor while spoofing the
            # same metadata.  Require the cache's row/text sidecars to resolve
            # to the actual declared donor files.
            for name in (
                "image_ids.json",
                "text_part.f16.npy",
                "has_text.bool.npy",
            ):
                cache_sidecar = os.path.join(cache_dir, name)
                donor_sidecar = os.path.join(donor_dir, name)
                if not os.path.isfile(donor_sidecar):
                    raise FileNotFoundError(
                        "[siglip2-cache] declared donor text sidecar is "
                        f"missing: {donor_sidecar}"
                    )
                try:
                    bound_to_donor = os.path.samefile(
                        cache_sidecar, donor_sidecar,
                    )
                except FileNotFoundError as exc:
                    raise FileNotFoundError(
                        "[siglip2-cache] local-crop text provenance file is "
                        f"missing: {name}"
                    ) from exc
                if not bound_to_donor:
                    raise ValueError(
                        "[siglip2-cache] local-crop text/row sidecar is not "
                        f"bound to the declared donor: {name}"
                    )

            aug_pattern = re.compile(
                r"^visual_(tokens|global)_aug([0-9]+)\.f16\.npy$"
            )
            actual_aug = {"tokens": set(), "global": set()}
            for name in os.listdir(cache_dir):
                match = aug_pattern.fullmatch(name)
                if match is not None:
                    actual_aug[match.group(1)].add(int(match.group(2)))
            expected_aug = set(range(declared_local_aug_views))
            if (
                actual_aug["tokens"] != expected_aug
                or actual_aug["global"] != expected_aug
            ):
                raise ValueError(
                    "[siglip2-cache] local-crop augmented visual files do not "
                    "exactly match save_aug_views: "
                    f"expected={sorted(expected_aug)}, actual={actual_aug}"
                )

            expected_token_shape = tuple(self.visual_tokens.shape)
            expected_global_shape = tuple(self.visual_global.shape)
            for view in range(declared_local_aug_views):
                token_path = os.path.join(
                    cache_dir, f"visual_tokens_aug{view}.f16.npy"
                )
                global_path = os.path.join(
                    cache_dir, f"visual_global_aug{view}.f16.npy"
                )
                token_array = np.load(token_path, mmap_mode="r")
                global_array = np.load(global_path, mmap_mode="r")
                if (
                    tuple(token_array.shape) != expected_token_shape
                    or token_array.dtype != np.float16
                ):
                    raise ValueError(
                        f"[siglip2-cache] visual_tokens_aug{view} must be "
                        f"float16 {expected_token_shape}, got "
                        f"{token_array.dtype} {token_array.shape}"
                    )
                if (
                    tuple(global_array.shape) != expected_global_shape
                    or global_array.dtype != np.float16
                ):
                    raise ValueError(
                        f"[siglip2-cache] visual_global_aug{view} must be "
                        f"float16 {expected_global_shape}, got "
                        f"{global_array.dtype} {global_array.shape}"
                    )

            for name in expected_global_files:
                cache_global = os.path.join(cache_dir, name)
                donor_global = os.path.join(donor_dir, name)
                if not os.path.isfile(donor_global):
                    raise FileNotFoundError(
                        "[siglip2-cache] declared donor full-image global is "
                        f"missing: {donor_global}"
                    )
                try:
                    bound_to_donor = os.path.samefile(
                        cache_global, donor_global,
                    )
                except FileNotFoundError as exc:
                    raise FileNotFoundError(
                        "[siglip2-cache] local-crop global provenance file is "
                        f"missing: {name}"
                    ) from exc
                if not bound_to_donor:
                    raise ValueError(
                        "[siglip2-cache] local-crop global is not bound to "
                        f"the declared donor full-image tensor: {name}"
                    )
        # Optional token-level text cache (Option B / v22b cross-attention path).
        # Files are produced by `extract_siglip2_features.py --save_text_tokens`.
        tt_p  = os.path.join(cache_dir, "text_tokens.f16.npy")
        ttm_p = os.path.join(cache_dir, "text_token_mask.bool.npy")
        self.text_tokens = None
        self.text_token_mask = None
        if os.path.exists(tt_p) and os.path.exists(ttm_p):
            self.text_tokens     = np.load(tt_p,  mmap_mode="r")        # [N, 6, T, D_proj]
            self.text_token_mask = np.load(ttm_p, mmap_mode="r")        # [N, 6, T]
            print(f"[siglip2-cache] loaded token-level text cache from {cache_dir} "
                  f"(text_tokens {self.text_tokens.shape}).")

        # Optional local counterfactual-caption features. Both files are an
        # atomic sidecar contract; loading just one could silently turn the
        # enabled hard-negative experiment into a no-op.
        tf_p = os.path.join(cache_dir, "text_foil_part.f16.npy")
        tfv_p = os.path.join(cache_dir, "text_foil_valid.bool.npy")
        tfi_p = os.path.join(cache_dir, "text_foil_image_ids.json")
        self.text_foil_part = None
        self.text_foil_valid = None
        if os.path.exists(tf_p) != os.path.exists(tfv_p):
            raise FileNotFoundError(
                "[siglip2-cache] counterfactual sidecars are incomplete; "
                "text_foil_part.f16.npy and text_foil_valid.bool.npy must "
                "either both exist or both be absent."
            )
        if os.path.exists(tf_p):
            self.text_foil_part = np.load(tf_p, mmap_mode="r")
            self.text_foil_valid = np.load(tfv_p, mmap_mode="r")
            if self.text_foil_part.shape != self.text_part.shape:
                raise ValueError(
                    "[siglip2-cache] text foil/factual feature shapes differ: "
                    f"{self.text_foil_part.shape} vs {self.text_part.shape}"
                )
            if self.text_foil_valid.shape != self.text_part.shape[:2]:
                raise ValueError(
                    "[siglip2-cache] text_foil_valid must be [N, M], got "
                    f"{self.text_foil_valid.shape}"
                )
            if self.text_foil_valid.dtype != np.bool_:
                raise TypeError(
                    "[siglip2-cache] text_foil_valid must be bool, got "
                    f"{self.text_foil_valid.dtype}"
                )
            if bool(np.asarray(self.text_foil_valid[:, 0]).any()):
                raise ValueError(
                    "[siglip2-cache] counterfactual C_global must be invalid"
                )
            if os.path.exists(tfi_p):
                with open(tfi_p, "r") as f:
                    foil_image_ids = json.load(f)
                if foil_image_ids != self.image_ids:
                    raise ValueError(
                        "[siglip2-cache] text foil row order does not match "
                        "image_ids.json"
                    )
            print(
                f"[siglip2-cache] loaded local counterfactual text sidecars "
                f"from {cache_dir} (valid={int(self.text_foil_valid.sum())})."
            )

        tft_p = os.path.join(cache_dir, "text_foil_tokens.f16.npy")
        tftm_p = os.path.join(cache_dir, "text_foil_token_mask.bool.npy")
        tfti_p = os.path.join(cache_dir, "text_foil_token_image_ids.json")
        self.text_foil_tokens = None
        self.text_foil_token_mask = None
        if os.path.exists(tft_p) != os.path.exists(tftm_p):
            raise FileNotFoundError(
                "[siglip2-cache] token-level counterfactual sidecars are "
                "incomplete; feature and mask files must be paired."
            )
        if os.path.exists(tft_p):
            if self.text_foil_part is None:
                raise FileNotFoundError(
                    "[siglip2-cache] token-level foils require pooled foil "
                    "features and validity mask"
                )
            if self.text_tokens is None or self.text_token_mask is None:
                raise FileNotFoundError(
                    "[siglip2-cache] token-level foils require the factual "
                    "text token cache"
                )
            self.text_foil_tokens = np.load(tft_p, mmap_mode="r")
            self.text_foil_token_mask = np.load(tftm_p, mmap_mode="r")
            if self.text_foil_tokens.shape != self.text_tokens.shape:
                raise ValueError(
                    "[siglip2-cache] factual/foil token shapes differ: "
                    f"{self.text_tokens.shape} vs "
                    f"{self.text_foil_tokens.shape}"
                )
            if self.text_foil_token_mask.shape != self.text_token_mask.shape:
                raise ValueError(
                    "[siglip2-cache] factual/foil token-mask shapes differ"
                )
            if self.text_foil_token_mask.dtype != np.bool_:
                raise TypeError(
                    "[siglip2-cache] token-level foil mask must be bool"
                )
            if bool(np.asarray(self.text_foil_token_mask[:, 0, :]).any()):
                raise ValueError(
                    "[siglip2-cache] C_global foil token mask must be empty"
                )
            token_slot_valid = np.asarray(
                self.text_foil_token_mask,
            ).any(axis=-1)
            if not np.array_equal(
                token_slot_valid,
                np.asarray(self.text_foil_valid),
            ):
                raise ValueError(
                    "[siglip2-cache] token-mask slot coverage does not match "
                    "text_foil_valid"
                )
            if not os.path.exists(tfti_p):
                raise FileNotFoundError(
                    "[siglip2-cache] token-level foil image-id contract is missing"
                )
            with open(tfti_p, "r") as f:
                foil_token_image_ids = json.load(f)
            if foil_token_image_ids != self.image_ids:
                raise ValueError(
                    "[siglip2-cache] token-level foil row order does not "
                    "match image_ids.json"
                )
            print(
                f"[siglip2-cache] loaded token-level counterfactual text "
                f"sidecars from {cache_dir} "
                f"(tokens {self.text_foil_tokens.shape})."
            )

        # Optional paired-augmentation visual feature cache. Files are
        # produced by `extract_siglip2_features.py --save_aug_views K`.
        # Used by v29+ paired-aug NtXent so the model doesn't have to run
        # the SigLIP2 backbone live on each augmented view at train time.
        self.visual_tokens_aug = []   # list of memmaps [N, num_patches, H_v]
        self.visual_global_aug = []   # list of memmaps [N, D_proj]
        aug_indices = (
            range(declared_local_aug_views)
            if declared_local_aug_views is not None else range(8)
        )
        for i in aug_indices:  # legacy caches stop at the first missing pair
            t_p = os.path.join(cache_dir, f"visual_tokens_aug{i}.f16.npy")
            g_p = os.path.join(cache_dir, f"visual_global_aug{i}.f16.npy")
            if not (os.path.exists(t_p) and os.path.exists(g_p)):
                break
            self.visual_tokens_aug.append(np.load(t_p, mmap_mode="r"))
            self.visual_global_aug.append(np.load(g_p, mmap_mode="r"))
        if self.visual_tokens_aug:
            print(f"[siglip2-cache] loaded {len(self.visual_tokens_aug)} "
                  f"paired-aug view(s) from {cache_dir}.")

    def lookup_row(self, image_id: str) -> Optional[int]:
        return self.id_to_row.get(image_id)

    def get(self, row_idx: int) -> dict:
        # fp16 -> fp32 here; one row is small (e.g. ~300KB at SigLIP2-base).
        out = {
            "cached_visual_tokens_raw": torch.from_numpy(np.asarray(self.visual_tokens[row_idx], dtype=np.float32)),
            "cached_visual_global":     torch.from_numpy(np.asarray(self.visual_global[row_idx], dtype=np.float32)),
            "cached_text_part_raw":     torch.from_numpy(np.asarray(self.text_part[row_idx],     dtype=np.float32)),
            "has_text":                 bool(self.has_text[row_idx]),
        }
        if self.text_tokens is not None:
            out["cached_text_tokens"]     = torch.from_numpy(np.asarray(self.text_tokens    [row_idx], dtype=np.float32))
            out["cached_text_token_mask"] = torch.from_numpy(np.asarray(self.text_token_mask[row_idx], dtype=np.bool_))
        if self.text_foil_part is not None:
            out["cached_text_foil_raw"] = torch.from_numpy(
                np.asarray(self.text_foil_part[row_idx], dtype=np.float32)
            )
            out["cached_text_foil_valid"] = torch.from_numpy(
                np.asarray(self.text_foil_valid[row_idx], dtype=np.bool_)
            )
        if self.text_foil_tokens is not None:
            out["cached_text_foil_tokens"] = torch.from_numpy(
                np.asarray(self.text_foil_tokens[row_idx], dtype=np.float32)
            )
            out["cached_text_foil_token_mask"] = torch.from_numpy(
                np.asarray(
                    self.text_foil_token_mask[row_idx], dtype=np.bool_,
                )
            )
        # Paired-augmentation views (built by `--save_aug_views K`).
        # Exposed as cached_visual_{tokens,global}_aug{i}. Trainer can
        # use these as view-1/view-2 inputs to skip the live backbone
        # during paired-aug NtXent training.
        for i, (tk, gl) in enumerate(zip(self.visual_tokens_aug, self.visual_global_aug)):
            out[f"cached_visual_tokens_aug{i}"] = torch.from_numpy(np.asarray(tk[row_idx], dtype=np.float32))
            out[f"cached_visual_global_aug{i}"] = torch.from_numpy(np.asarray(gl[row_idx], dtype=np.float32))
        return out


def _build_cifar10_cache_row_map(data: np.ndarray, cache: "_SigLIP2FeatureCache") -> np.ndarray:
    """Map dataset row -> feature-cache row using md5 of image bytes."""
    rows = np.full(len(data), -1, dtype=np.int64)
    miss = 0
    for i in range(len(data)):
        iid = _cifar10_image_id(data[i])
        r = cache.lookup_row(iid)
        if r is None:
            miss += 1
            continue
        rows[i] = r
    if miss > 0:
        # Hard error -- the cache is keyed by content md5 so a miss means the
        # cache was built from a different CIFAR10 dump. Better to fail loudly.
        raise RuntimeError(
            f"[siglip2-cache] {miss}/{len(data)} dataset rows not found in cache "
            f"(meta N={cache.meta.get('N')}). Was the cache built for a different "
            f"CIFAR10 root?"
        )
    return rows


def _build_pathkeyed_cache_row_map(
    img_paths: list, root: str, cache: "_SigLIP2FeatureCache",
) -> np.ndarray:
    """Map dataset row -> feature-cache row using ``relpath(img_path, root)``.

    For ImgRtvDataset (filesystem-walk datasets like ImageNet100) the cache
    is keyed by the same relpath that `preprocess_qwen_codebook_texts.py`
    uses, e.g. ``image/n02xxx_yyy.JPEG``.
    """
    rows = np.full(len(img_paths), -1, dtype=np.int64)
    miss = 0
    for i, p in enumerate(img_paths):
        iid = os.path.relpath(str(p), str(root))
        r = cache.lookup_row(iid)
        if r is None:
            miss += 1
            continue
        rows[i] = r
    if miss > 0:
        raise RuntimeError(
            f"[siglip2-cache] {miss}/{len(img_paths)} dataset rows not found in cache "
            f"(meta N={cache.meta.get('N')}). Was the cache built with a matching "
            f"--imagenet100_root / split list?"
        )
    return rows


# ---------------------------------------------------------------- qwen cache

def _attention_mask_from_enc(enc, tokenizer) -> "torch.Tensor":
    """Return the attention_mask tensor whether or not the tokenizer emitted it.

    SigLIP2's tokenizer has ``model_input_names = ['input_ids']`` so it does NOT
    include ``attention_mask`` by default — we have to either pass
    ``return_attention_mask=True`` (which is also done at call sites) or
    reconstruct it from the pad token id. This helper handles both paths so
    callers don't crash on a KeyError.
    """
    import torch as _torch
    if "attention_mask" in enc:
        return enc["attention_mask"]
    input_ids = enc["input_ids"]
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        return _torch.ones_like(input_ids)
    return (input_ids != pad_id).long()


def _load_qwen_text_cache(jsonl_path: str) -> dict:
    """Read a JSONL cache produced by `preprocess_qwen_codebook_texts.py`.

    Each line: ``{"image_id": ..., "codebook_texts": {...}, ...}``.
    Returns dict ``image_id -> codebook_texts (dict)``.
    Lines without a valid ``codebook_texts`` are skipped.
    """
    cache: dict = {}
    if not os.path.exists(jsonl_path):
        return cache
    with open(jsonl_path, "r") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            iid = row.get("image_id")
            cb  = row.get("codebook_texts")
            if iid is None or not isinstance(cb, dict):
                continue
            cache[iid] = cb
    return cache


def _build_part_token_tensors(
    img_paths: list,
    root: str,
    cache_path: str,
    tokenizer_name: str,
    max_length: int,
) -> Optional[dict]:
    """Eagerly tokenize the 6 codebook texts for every dataset sample.

    Returns:
        {"input_ids":      LongTensor [N, 6, L],
         "attention_mask": Tensor     [N, 6, L]}
        or None if the cache is missing / empty.
    """
    if not cache_path or not os.path.exists(cache_path):
        return None

    # imports kept local so dataloaders.py stays usable without these deps
    from dna_utils import (
        build_siglip2_text_tokenizer,
        extract_codebook_texts,
        CODEBOOK_TEXT_KEYS,
        DEFAULT_FALLBACK_TEXT,
    )

    cache = _load_qwen_text_cache(cache_path)
    if not cache:
        print(f"[qwen-cache] WARNING: cache file at `{cache_path}` is empty / unparseable.")
        return None

    flat: list = []
    miss = 0
    fallback = DEFAULT_FALLBACK_TEXT
    for p in img_paths:
        iid = os.path.relpath(p, root) if root else p
        cb = cache.get(iid)
        if cb is None:
            miss += 1
            texts = [fallback] * 6
        else:
            texts = extract_codebook_texts({"codebook_texts": cb})
        flat.extend(texts)
    if miss > 0:
        print(f"[qwen-cache] {miss}/{len(img_paths)} images missing in `{cache_path}` "
              f"-- using `{fallback}` fallback for those rows.")

    tokenizer = build_siglip2_text_tokenizer(tokenizer_name)
    enc = tokenizer(
        flat,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_attention_mask=True,            # SigLIP2 tokenizer omits this by default
    )
    input_ids      = enc["input_ids"]
    attention_mask = _attention_mask_from_enc(enc, tokenizer)
    N = len(img_paths)
    L = input_ids.shape[-1]
    return {
        "input_ids":      input_ids     .view(N, 6, L),
        "attention_mask": attention_mask.view(N, 6, L),
    }


def _cifar10_image_id(img_arr: np.ndarray) -> str:
    """Stable image_id for CIFAR10 keyed by md5 of the raw uint8 bytes.

    CIFAR10 has no per-image filesystem path, so we hash the image content
    instead. This makes the cache key independent of mode / setting / sampling
    — the same image gets the same id whether it appears in train, test, or
    database. The preprocessing script must use the SAME hashing scheme.
    """
    import hashlib
    return hashlib.md5(np.ascontiguousarray(img_arr).tobytes()).hexdigest()[:16]


def _build_part_token_tensors_cifar10(
    data: np.ndarray,
    cache_path: str,
    tokenizer_name: str,
    max_length: int,
) -> Optional[dict]:
    """CIFAR10 equivalent of `_build_part_token_tensors`, keyed by md5 of image
    bytes (see `_cifar10_image_id`).

    Args:
        data : numpy array of shape [N, H, W, 3] uint8 (CIFAR10's `self.data`)
    """
    if not cache_path or not os.path.exists(cache_path):
        return None

    from dna_utils import (
        build_siglip2_text_tokenizer,
        extract_codebook_texts,
        DEFAULT_FALLBACK_TEXT,
    )

    cache = _load_qwen_text_cache(cache_path)
    if not cache:
        print(f"[qwen-cache CIFAR10] WARNING: cache `{cache_path}` empty / unparseable.")
        return None

    flat: list = []
    miss = 0
    fallback = DEFAULT_FALLBACK_TEXT
    for i in range(len(data)):
        iid = _cifar10_image_id(data[i])
        cb = cache.get(iid)
        if cb is None:
            miss += 1
            texts = [fallback] * 6
        else:
            texts = extract_codebook_texts({"codebook_texts": cb})
        flat.extend(texts)
    if miss > 0:
        print(f"[qwen-cache CIFAR10] {miss}/{len(data)} images missing in "
              f"`{cache_path}` -- using `{fallback}` fallback for those rows.")

    tokenizer = build_siglip2_text_tokenizer(tokenizer_name)
    enc = tokenizer(
        flat,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_attention_mask=True,            # SigLIP2 tokenizer omits this by default
    )
    input_ids      = enc["input_ids"]
    attention_mask = _attention_mask_from_enc(enc, tokenizer)
    N = len(data)
    L = input_ids.shape[-1]
    return {
        "input_ids":      input_ids     .view(N, 6, L),
        "attention_mask": attention_mask.view(N, 6, L),
    }


def gen_hash_table(hash_code_path='/home/yschoi/DNAemb_quat_hashing/converted_data.csv', num_bit=20):

    base = {
        'A' : 0,
        'C' : 1,
        'G' : 2, 
        'T' : 3,
    }

    data = pd.read_csv(hash_code_path)
    data = data.to_numpy()
    data = data.squeeze(-1)

    hash_code_set = np.zeros((len(data), num_bit, len(base) // 2))
    # ii = torch.eye(4)
    ii = torch.tensor([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    
    for n in range(len(data)):
        for b in range(len(data[n])):
            hash_code_set[n, b, :] = ii[int(base[data[n][b]])]
            
    _hash_code_set = hash_code_set.reshape(len(hash_code_set), -1)
    
    return _hash_code_set
    

class GaussianBlur(object):
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

    

class ImgRtvDataset(Dataset):
    def __init__(self, root,
                 img_transform=None, target_transform=None, mode ="train", setting_name = "setting1", return_index = False, return_paired_aug_img=False,
                 qwen_text_cache_path: Optional[str] = None,
                 siglip2_tokenizer_name: str = "google/siglip2-base-patch16-224",
                 siglip2_text_max_length: int = 64,
                 siglip2_feature_cache_dir: Optional[str] = None,
                 force_pixel_decode: bool = False):
        # path-keyed SigLIP2 feature cache (built by extract_siglip2_features.py
        # in --imagenet100 mode). Loaded after self.img_paths is populated.
        self._siglip2_feature_cache_dir = siglip2_feature_cache_dir
        # `force_pixel_decode=True` keeps the PIL decode active even when the
        # SigLIP2 feature cache is loaded -- needed when the model has a pixel
        # reconstruction decoder (v28a) and requires the raw image as target.
        # When False (default), we skip the decode for speed.
        self._force_pixel_decode = bool(force_pixel_decode)
        self._feat_cache: Optional[_SigLIP2FeatureCache] = None
        self._feat_cache_rows: Optional[np.ndarray] = None
        self.loader = self.default_loader
        self.root = os.path.expanduser(root)
        
        self.default_transform = transforms.Compose([transforms.Resize((224,224)),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                        ])    
        self.transform = img_transform
        self.target_transform = target_transform

        if mode == "train":
            self.base_folder = 'train.txt'
        elif mode == "test" or mode == "query":
            self.base_folder = 'test.txt'
        elif mode == "database":
            self.base_folder = 'database.txt'
        else:
            raise ValueError

        self.img_paths = []
        self.img_labels = []

        filename = os.path.join(self.root, setting_name, self.base_folder)

        with open(filename, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline()
                if not lines:
                    break
                pos_tmp = lines.split()[0]    
                pos_tmp = os.path.join(self.root, pos_tmp)
                label_tmp = lines.split()[1:]
                self.img_paths.append(pos_tmp)
                self.img_labels.append( list(map(int,label_tmp)))
        
        self.img_paths = np.array(self.img_paths)

        num_classes = len(self.img_labels[0])

        # self.img_labels = torch.tensor(self.img_labels)
        self.img_labels = np.array(self.img_labels, dtype=np.int64) #np.float
        self.img_labels.reshape((-1, num_classes))
        
        self.return_index = return_index
        if return_index:
            self.index_table = torch.arange(0, len(self))
        self.return_paired_aug_img = return_paired_aug_img
        if return_paired_aug_img:
            assert img_transform != None

        # ---- optional Qwen-derived part text tokens ----
        # When a cache is supplied, we pre-tokenize every sample's 6 sentences
        # with the SigLIP2 tokenizer so __getitem__ can just slice tensors.
        # Skipped when a SigLIP2 feature cache is supplied -- the cached
        # `text_part_raw` already encapsulates the encoded text for each row.
        self.qwen_text_cache_path = qwen_text_cache_path
        self._part_tokens = None
        if self._siglip2_feature_cache_dir is None:
            self._part_tokens = _build_part_token_tensors(
                img_paths=list(self.img_paths),
                root=self.root,
                cache_path=qwen_text_cache_path,
                tokenizer_name=siglip2_tokenizer_name,
                max_length=siglip2_text_max_length,
            )
            if self._part_tokens is not None:
                assert self._part_tokens["input_ids"].shape[0] == len(self.img_paths)

        # ---- optional SigLIP2 visual+text feature cache (path-keyed) ----
        if (self._siglip2_feature_cache_dir is not None
                and os.path.isdir(self._siglip2_feature_cache_dir)):
            self._feat_cache = _SigLIP2FeatureCache(self._siglip2_feature_cache_dir)
            self._feat_cache_rows = _build_pathkeyed_cache_row_map(
                list(self.img_paths), self.root, self._feat_cache,
            )
            print(f"[ImgRtvDataset] siglip2 feature cache loaded from "
                  f"{self._siglip2_feature_cache_dir} -- "
                  f"N={len(self.img_paths)} images mapped to cache rows.")

    def default_loader(self, path):
        from torchvision import get_image_backend
        if get_image_backend() == 'accimage':
            return accimage_loader(path)
        else:
            return pil_loader(path)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        img_name, target = self.img_paths[index], self.img_labels[index]

        # When a SigLIP2 feature cache is loaded we skip the expensive PIL
        # decode + resize entirely -- pixel_values won't be consumed and the
        # SigLIP2 trainer does not read img_tr1/img_tr2 either (legacy fields).
        # For ImageNet 224x224 JPEGs this avoids 100K+ disk reads per epoch.
        # Override: `force_pixel_decode=True` is set when v28a (pixel decoder)
        # needs the raw image as a reconstruction target.
        skip_image_decode = (self._feat_cache is not None) and not self._force_pixel_decode
        if not skip_image_decode:
            img = self.loader(img_name)
            img_i = self.default_transform(img)
        else:
            img = None
            img_i = None

        if self.target_transform is not None:
            target = self.target_transform(target)

        out = {}
        out['label'] = target
        if img_i is not None:
            out['img'] = img_i

        if self.return_index:
            out['idx'] = self.index_table[index]

        if self.return_paired_aug_img and not skip_image_decode:
            assert self.transform  is not None
            out['img_tr1'] = self.transform(img)
            out['img_tr2'] = self.transform(img)

        if self._part_tokens is not None:
            out['part_input_ids']      = self._part_tokens['input_ids'][index]       # [6, L]
            out['part_attention_mask'] = self._part_tokens['attention_mask'][index]  # [6, L]

        # cached SigLIP2 features when configured -- training / extraction
        # loops detect these and skip the encoder pass entirely.
        if self._feat_cache is not None and self._feat_cache_rows is not None:
            row = int(self._feat_cache_rows[index])
            cached = self._feat_cache.get(row)
            out.update(cached)

        out['image_path'] = str(img_name)
        return out

    def __len__(self):
        return len(self.img_paths)
        
def get_idx_for_uniform_sampling(dataset, n_class, n_samples_per_class,offset=0): # convert_train_test=False
    # train, test data를 모두 합침
    L = np.array(dataset.targets)
    first = True

    for label in range(n_class):
        # 레이블이 "label"에 해당하는 데이터의 인덱스를 찾음
        index = np.where(L == label)[0]

        # 인덱스를 랜덤으로 섞음
        N = index.shape[0]
        np.random.seed(0)
        perm = np.random.permutation(N)
        index = index[perm]

        # 레이블이 "label"에 해당하는 데이터 N개를 무작위로 선택 
        index = index[offset:offset+n_samples_per_class]

        # array에 추가
        if first:
            index_array = index
        else:
            index_array = np.concatenate((index_array, index))

        first = False

    return index_array


class ImgRtvCIFAR10(CIFAR10):
    def __init__(self,  root,
                 img_transform=None, target_transform=None, mode ="train", setting_name = "setting1", return_index=False,return_paired_aug_img=False,
                 qwen_text_cache_path: Optional[str] = None,
                 siglip2_tokenizer_name: str = "google/siglip2-base-patch16-224",
                 siglip2_text_max_length: int = 64,
                 siglip2_feature_cache_dir: Optional[str] = None,
                 force_pixel_decode: bool = False) -> None:

        # CIFAR10 has no stable per-image filesystem path, so the Qwen cache
        # is keyed by md5 of the raw image bytes instead (see `_cifar10_image_id`).
        # We defer the actual load until after `self.data` is finalized below.
        self._qwen_text_cache_path     = qwen_text_cache_path
        self._siglip2_tokenizer_name   = siglip2_tokenizer_name
        self._siglip2_text_max_length  = siglip2_text_max_length
        self._siglip2_feature_cache_dir = siglip2_feature_cache_dir
        # Mirror of ImgRtvDataset.force_pixel_decode (v28a path); CIFAR10 in
        # cached mode also short-circuits the PIL decode without it.
        self._force_pixel_decode = bool(force_pixel_decode)
        self._part_tokens = None
        self._feat_cache: Optional[_SigLIP2FeatureCache] = None
        self._feat_cache_rows: Optional[np.ndarray] = None

        self.default_transform = transforms.Compose([transforms.Resize((224,224)),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                ])

        if setting_name == "setting1":
            # Setting 1 - (MBE 논문 setting 2)
            # Train: 클래스당 500개를 Trainset에서 샘플링 (5k)
            # Test:  클래스당 100개를 Testset에서 샘플링 (1k)
            # Database: Testset 1k개 제외한 전체 (5.8k)

            if mode == "train":
                train = True; transform=img_transform
            elif mode == "test":
                train= False; transform=img_transform
            elif mode == "database":
                train= False; transform=img_transform
            else:
                raise KeyError
            
            super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=True)
            
            if mode == 'train':
                idx = get_idx_for_uniform_sampling(self, 10, n_samples_per_class=500)
            elif mode == "test": 
                idx = get_idx_for_uniform_sampling(self, 10, n_samples_per_class=100)
            elif mode == "database":
                trainset = CIFAR10(root, train=True, transform=img_transform, target_transform=target_transform)
                testset = CIFAR10(root, train=False, transform=img_transform, target_transform=target_transform)
                self.data = np.concatenate((trainset.data, testset.data))
                self.targets = np.concatenate(( np.array(trainset.targets), np.array(testset.targets)))
                
                # Test에서 뽑힌 N개 제외한 인덱스를 구함
                idx = get_idx_for_uniform_sampling(testset, 10, n_samples_per_class= 900, offset=100)
                idx = np.concatenate( (np.arange(0, len(trainset)), idx+len(trainset)))
                
                
            self.data = self.data[idx]
            self.targets = np.array(self.targets)[idx]
            
            

        elif setting_name == "setting2":         
            # Setting 2 -  (MBE 논문 setting 1)
            # Test: 전체 Testset (10k)/ Training: 전체 Trainset (50k)/ Database: 전체 Trainset(50k)
            if mode == "train":
                train=True; transform=img_transform
            elif mode == "test":
                train=False; transform=img_transform
            elif mode == "database":
                train=True; transform=img_transform
            else:
                raise KeyError
            super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=True)
        else:
            raise KeyError
        
        self.return_index = return_index

        if return_index:
            self.index_table = torch.arange(0, len(self))

        self.return_paired_aug_img = return_paired_aug_img
        if return_paired_aug_img:
            assert img_transform != None

        # ---- optional Qwen-derived part text tokens (md5-keyed) ----
        # `self.data` is now the final post-sampling array of shape [N, H, W, 3].
        # Skipped when a SigLIP2 feature cache is supplied -- the cached
        # text_part embeddings already encapsulate the tokenized + encoded
        # form, so the runtime tokenizer + text encoder are not needed.
        if (self._qwen_text_cache_path is not None
                and os.path.exists(self._qwen_text_cache_path)
                and self._siglip2_feature_cache_dir is None):
            self._part_tokens = _build_part_token_tensors_cifar10(
                data=self.data,
                cache_path=self._qwen_text_cache_path,
                tokenizer_name=self._siglip2_tokenizer_name,
                max_length=self._siglip2_text_max_length,
            )
            if self._part_tokens is not None:
                assert self._part_tokens["input_ids"].shape[0] == len(self.data)

        # ---- optional SigLIP2 visual+text feature cache (md5-keyed) ----
        # When supplied, __getitem__ returns precomputed encoder outputs and
        # the model can skip the encoder pass entirely. Built once in
        # `extract_siglip2_features.py`.
        if (self._siglip2_feature_cache_dir is not None
                and os.path.isdir(self._siglip2_feature_cache_dir)):
            self._feat_cache = _SigLIP2FeatureCache(self._siglip2_feature_cache_dir)
            self._feat_cache_rows = _build_cifar10_cache_row_map(self.data, self._feat_cache)
            print(f"[ImgRtvCIFAR10] siglip2 feature cache loaded from "
                  f"{self._siglip2_feature_cache_dir} -- "
                  f"N={len(self.data)} images mapped to cache rows.")


    def __getitem__(self, index: Any) -> T_co:
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        img_i = self.default_transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        out = {}
        out['label'] = np.eye(10,dtype=np.int64)[target]
        out['img'] = img_i

        if self.return_index:
            out['idx'] = self.index_table[index]

        if self.return_paired_aug_img:
            assert self.transform  is not None
            out['img_tr1'] = self.transform(img)
            out['img_tr2'] = self.transform(img)

        if self._part_tokens is not None:
            out['part_input_ids']      = self._part_tokens['input_ids'][index]       # [6, L]
            out['part_attention_mask'] = self._part_tokens['attention_mask'][index]  # [6, L]

        # Cached SigLIP2 features (when configured) -- the training / extraction
        # loops use these to bypass the visual + text encoder pass.
        if self._feat_cache is not None and self._feat_cache_rows is not None:
            row = int(self._feat_cache_rows[index])
            cached = self._feat_cache.get(row)
            out.update(cached)

        return out






def _encode_onehot(labels, num_classes: int) -> np.ndarray:
    """Encode integer labels as one-hot int64 array of shape [N, num_classes].

    Mirrors `data.transform.encode_onehot` used by the upstream CUB-2011
    pipeline so the label format stays identical.
    """
    n = len(labels)
    out = np.zeros((n, num_classes), dtype=np.int64)
    for i, lab in enumerate(labels):
        out[i, int(lab)] = 1
    return out


class ImgRtvCUB2011(Dataset):
    """CUB-200-2011 wrapper.

    Mirrors the upstream Cub2011 pre-processing exactly:
      - reads `images.txt`, `image_class_labels.txt`, `train_test_split.txt`
        from the dataset root (no setting1/setting2 split files needed);
      - train_test_split.txt: is_training_img == 1 -> TRAIN_DATA + RETRIEVAL_DATA,
                              is_training_img == 0 -> QUERY_DATA;
      - retrieval database == trainset (5994 images), query == test split (5794);
      - labels encoded as int64 one-hot [N, 200] (target - 1, 0-indexed).

    Output dict matches ImgRtvDataset's interface so the rest of the
    training / extraction pipeline does not need to special-case CUB:
        'label'             : int64 one-hot [200]
        'img'               : default-transformed tensor (skipped when feat cache present)
        'idx'               : (when return_index)
        'img_tr1' / 'img_tr2': augmented views (when return_paired_aug_img)
        'part_input_ids' / 'part_attention_mask': Qwen-derived 6-codebook tokens
        'cached_*'          : SigLIP2 / CLIP feature cache rows
        'image_path'        : str

    `mode` accepts the upstream names ('train', 'query', 'retrieval') and
    the existing pipeline names ('train', 'test', 'database') interchangeably.
    `setting_name` is accepted for API compatibility but ignored (CUB has a
    canonical split baked into train_test_split.txt).
    """

    _INITIALIZED_FOR: Optional[str] = None  # cache root path of last init

    @staticmethod
    def _init_cls(root: str) -> None:
        images = pd.read_csv(os.path.join(root, 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(root, 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(root, 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])
        data = images.merge(image_class_labels, on='img_id')
        all_data = data.merge(train_test_split, on='img_id')
        all_data['filepath'] = 'images/' + all_data['filepath']
        train_data = all_data[all_data['is_training_img'] == 1]
        test_data = all_data[all_data['is_training_img'] == 0]

        ImgRtvCUB2011.QUERY_DATA      = test_data['filepath'].to_numpy()
        ImgRtvCUB2011.QUERY_TARGETS   = _encode_onehot((test_data['target'] - 1).tolist(), 200)
        ImgRtvCUB2011.TRAIN_DATA      = train_data['filepath'].to_numpy()
        ImgRtvCUB2011.TRAIN_TARGETS   = _encode_onehot((train_data['target'] - 1).tolist(), 200)
        ImgRtvCUB2011.RETRIEVAL_DATA    = ImgRtvCUB2011.TRAIN_DATA
        ImgRtvCUB2011.RETRIEVAL_TARGETS = ImgRtvCUB2011.TRAIN_TARGETS
        ImgRtvCUB2011._INITIALIZED_FOR = root

    def __init__(self, root,
                 img_transform=None, target_transform=None,
                 mode="train", setting_name="setting1",
                 return_index=False, return_paired_aug_img=False,
                 qwen_text_cache_path: Optional[str] = None,
                 siglip2_tokenizer_name: str = "google/siglip2-base-patch16-224",
                 siglip2_text_max_length: int = 64,
                 siglip2_feature_cache_dir: Optional[str] = None,
                 force_pixel_decode: bool = False):
        self.root = os.path.expanduser(root)
        self.transform = img_transform               # augmentation for paired-aug NtXent
        self.target_transform = target_transform
        self.return_index = return_index
        self.return_paired_aug_img = return_paired_aug_img
        self.qwen_text_cache_path = qwen_text_cache_path
        self._siglip2_feature_cache_dir = siglip2_feature_cache_dir
        self._force_pixel_decode = bool(force_pixel_decode)

        self.default_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Build the per-split paths/labels exactly the way upstream Cub2011 does.
        if ImgRtvCUB2011._INITIALIZED_FOR != self.root:
            ImgRtvCUB2011._init_cls(self.root)

        if mode == "train":
            data, targets = ImgRtvCUB2011.TRAIN_DATA, ImgRtvCUB2011.TRAIN_TARGETS
        elif mode in ("test", "query"):
            data, targets = ImgRtvCUB2011.QUERY_DATA, ImgRtvCUB2011.QUERY_TARGETS
        elif mode in ("database", "retrieval"):
            data, targets = ImgRtvCUB2011.RETRIEVAL_DATA, ImgRtvCUB2011.RETRIEVAL_TARGETS
        else:
            raise ValueError(f"[ImgRtvCUB2011] invalid mode: {mode}")

        # `img_paths` is absolute filesystem paths (consistent with ImgRtvDataset)
        # so cache row mapping and image_path output stay uniform.
        self.img_paths = np.array([os.path.join(self.root, fp) for fp in data])
        self.img_labels = np.asarray(targets, dtype=np.int64)

        if return_index:
            self.index_table = torch.arange(0, len(self))
        if return_paired_aug_img:
            assert img_transform is not None, (
                "[ImgRtvCUB2011] return_paired_aug_img requires img_transform"
            )

        # ---- optional Qwen-derived part text tokens ----
        # Same gate as ImgRtvDataset: only build if no SigLIP2 feature cache.
        self._part_tokens = None
        if self._siglip2_feature_cache_dir is None:
            self._part_tokens = _build_part_token_tensors(
                img_paths=list(self.img_paths),
                root=self.root,
                cache_path=qwen_text_cache_path,
                tokenizer_name=siglip2_tokenizer_name,
                max_length=siglip2_text_max_length,
            )
            if self._part_tokens is not None:
                assert self._part_tokens["input_ids"].shape[0] == len(self.img_paths)

        # ---- optional SigLIP2 / CLIP visual+text feature cache ----
        self._feat_cache: Optional[_SigLIP2FeatureCache] = None
        self._feat_cache_rows: Optional[np.ndarray] = None
        if (self._siglip2_feature_cache_dir is not None
                and os.path.isdir(self._siglip2_feature_cache_dir)):
            self._feat_cache = _SigLIP2FeatureCache(self._siglip2_feature_cache_dir)
            self._feat_cache_rows = _build_pathkeyed_cache_row_map(
                list(self.img_paths), self.root, self._feat_cache,
            )
            print(f"[ImgRtvCUB2011] siglip2 feature cache loaded from "
                  f"{self._siglip2_feature_cache_dir} -- "
                  f"N={len(self.img_paths)} images mapped to cache rows.")

    def get_onehot_targets(self):
        """Mirror upstream Cub2011.get_onehot_targets (returns float32 tensor)."""
        return torch.from_numpy(self.img_labels).float()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_name = self.img_paths[index]
        target = self.img_labels[index]

        skip_image_decode = (self._feat_cache is not None) and not self._force_pixel_decode
        if not skip_image_decode:
            img = Image.open(img_name).convert('RGB')
            img_i = self.default_transform(img)
        else:
            img = None
            img_i = None

        if self.target_transform is not None:
            target = self.target_transform(target)

        out = {}
        out['label'] = target
        if img_i is not None:
            out['img'] = img_i

        if self.return_index:
            out['idx'] = self.index_table[index]

        if self.return_paired_aug_img and not skip_image_decode:
            assert self.transform is not None
            out['img_tr1'] = self.transform(img)
            out['img_tr2'] = self.transform(img)

        if self._part_tokens is not None:
            out['part_input_ids']      = self._part_tokens['input_ids'][index]       # [6, L]
            out['part_attention_mask'] = self._part_tokens['attention_mask'][index]  # [6, L]

        if self._feat_cache is not None and self._feat_cache_rows is not None:
            row = int(self._feat_cache_rows[index])
            cached = self._feat_cache.get(row)
            out.update(cached)

        out['image_path'] = str(img_name)
        return out


def load_dataset(dataset_dir, dataset_name, setting,
                 train_transform=None, test_transform=None,
                 load_train=True, load_test=True, load_database=True,
                 return_index=False, return_paired_aug_img=False,
                 qwen_text_cache_path: Optional[str] = None,
                 siglip2_tokenizer_name: str = "google/siglip2-base-patch16-224",
                 siglip2_text_max_length: int = 64,
                 siglip2_feature_cache_dir: Optional[str] = None,
                 force_pixel_decode: bool = False):
    root = os.path.join(dataset_dir, dataset_name)
    extra_kwargs = dict(
        qwen_text_cache_path=qwen_text_cache_path,
        siglip2_tokenizer_name=siglip2_tokenizer_name,
        siglip2_text_max_length=siglip2_text_max_length,
        siglip2_feature_cache_dir=siglip2_feature_cache_dir,
        force_pixel_decode=force_pixel_decode,
    )
    if load_train:
        train_dataset = DATASET[dataset_name](
            root, train_transform, None, "train", setting,
            return_index=return_index, return_paired_aug_img=return_paired_aug_img,
            **extra_kwargs,
        )
    else:
        train_dataset = None

    if load_test:
        test_dataset = DATASET[dataset_name](
            root, test_transform, None, "test", setting,
            return_index=return_index, return_paired_aug_img=return_paired_aug_img,
            **extra_kwargs,
        )
    else:
        test_dataset = None

    if load_database:
        database_dataset = DATASET[dataset_name](
            root, test_transform, None, "database", setting,
            return_index=return_index, return_paired_aug_img=return_paired_aug_img,
            **extra_kwargs,
        )
    else:
        database_dataset = None

    return train_dataset, test_dataset, database_dataset

        

DATASET = {
           'DEFAULT': None,
           'CIFAR10': ImgRtvCIFAR10,
            'Flickr25k': ImgRtvDataset,
            'ImageNet100': ImgRtvDataset,
            'MSCOCO': ImgRtvDataset,
            'NUSWIDE': ImgRtvDataset,
            'CUB_200': ImgRtvCUB2011,
            'CUB_200_2011': ImgRtvCUB2011,
            }

NUM_CLASSES = { 
             'DEFAULT': {'setting1' : None},
             'CIFAR10': {'setting1' : 10,
                         'setting2'  : 10}, 
            'Flickr25k': {'setting1': 24}, 
            'ImageNet100':{'setting1': 100}, 
            'MSCOCO': {'setting1': 80}, 
            'NUSWIDE': {'setting1'  : 21,
                        'setting2'  : 10},
            'CUB_200': {'setting1': 200},
            'CUB_200_2011': {'setting1': 200},

            }

MULTI_LABEL = {
            'DEFAULT'   : None,
            'CIFAR10'   : False,
            'Flickr25k' : True,
            'ImageNet100': False,
            'MSCOCO'    : True,
            'NUSWIDE'   : True,
            'CUB_200'   : False,
            'CUB_200_2011': False,
            }

if __name__ == "__main__":
    # dataset = ImgRtvDataset("./ImageHashing/dataset/MSCOCO", img_transform=None,mode="test", setting_name="setting1")
    # dataset = ImgRtvCIFAR10("./ImageHashing/dataset/CIFAR10", img_transform=None,mode="database", setting_name="setting1")
    
    for dataset in ['Flickr25k', 'ImageNet100', 'MSCOCO', 'NUSWIDE']:
        print(dataset)
        trainset, testset, databaseset = load_dataset("dataset", dataset, setting="setting1", train_transform=None, test_transform=None)
        
        trainloader = DataLoader(trainset, batch_size=64, pin_memory=True)
        testloader = DataLoader(testset, batch_size=64, pin_memory=True)
        databaseloader = DataLoader(databaseset, batch_size=64, pin_memory=True)


        for epoch in range(1):
            print(f"Train:{len(trainloader.dataset)}")
            for i in trainloader:
                print(i[-1].shape)
                break
            print(f"Test:{len(testloader.dataset)}")
            for i in testloader:
                break
            print(f"Database:{len(databaseloader.dataset)}")
            for i in databaseloader:
                break

            print(epoch)
