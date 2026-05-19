"""Offline Qwen2.5-VL preprocessing script — produces a JSONL cache of codebook texts.

Run BEFORE training. Each output line is a JSON object with ``image_id``
and ``codebook_texts`` (the six per-codebook short sentences). The script can
be re-run safely — already-processed ``image_id``s are skipped.

Two modes:

1) Filesystem walk mode (default — Flickr25k / MSCOCO / NUSWIDE / ImageNet100):

    python preprocess_qwen_codebook_texts.py \\
        --image_dir /path/to/dataset/Flickr25k \\
        --root      /path/to/dataset/Flickr25k \\
        --cache_path /path/to/cache/flickr25k_qwen.jsonl

   image_id = os.path.relpath(image_path, root)
   This MUST match `dataloaders.ImgRtvDataset`'s lookup convention.

2) CIFAR10 mode (``--cifar10``):

    python preprocess_qwen_codebook_texts.py \\
        --cifar10 \\
        --cifar10_root ./dataset/CIFAR10 \\
        --cache_path /path/to/cache/cifar10_qwen.jsonl \\
        --limit 100      # smoke test

   image_id = md5(uint8_image_bytes)[:16]
   This MUST match `dataloaders._cifar10_image_id`'s hashing scheme.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from typing import Set

import torch
from PIL import Image
from tqdm import tqdm

from dna_utils import (
    CODEBOOK_TEXT_KEYS,
    extract_codebook_texts,
    build_qwen25_vl_generator,
    generate_object_centric_scene_graph,
)
# Direct import so we can dispatch between V1 / V2 prompts without touching
# the dna_utils public surface.
from dna_utils.vlm_qwen25_descriptions import (
    _PROMPT       as _PROMPT_V1,
    _PROMPT_V2,
    _PROMPT_V3,
    _PROMPT_V4,
    CODEBOOK_KEYS    as _CB_KEYS_V1,
    CODEBOOK_KEYS_V2 as _CB_KEYS_V2,
)
# V3 uses the same 6-key schema as V2 (only the prompt content differs --
# captions instead of bare nouns), so reuse the V2 key tuple.
_CB_KEYS_V3 = _CB_KEYS_V2
# V4 reuses the V2/V3 key schema with reframed "evidence-axis" prompt.
_CB_KEYS_V4 = _CB_KEYS_V2


_DEFAULT_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


# ----------------------------- filesystem walk mode helpers

def list_images(image_dir: str, exts: tuple = _DEFAULT_EXTS) -> list:
    paths = []
    for root, _, files in os.walk(image_dir):
        for f in files:
            if f.lower().endswith(exts):
                paths.append(os.path.join(root, f))
    return sorted(paths)


# ----------------------------- CIFAR10 mode helpers

def _cifar10_image_id(img_arr) -> str:
    """Mirror of `dataloaders._cifar10_image_id`. Keep in sync."""
    import numpy as np
    return hashlib.md5(np.ascontiguousarray(img_arr).tobytes()).hexdigest()[:16]


def _read_imagenet_split_paths(split_txt: str) -> list:
    """Return list of relpaths from a setting1 split txt file.

    Each line is of the form ``<relpath> <label1> <label2> ...``. We keep only
    the leading relpath token (matches ``ImgRtvDataset``'s loader convention).
    """
    out: list = []
    with open(split_txt, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(line.split()[0])
    return out


def iter_imagenet100_setting1(
    imagenet_root: str,
    setting_dir: str = "setting1",
    splits: tuple = ("train",),
):
    """Yield only the images referenced in ImageNet100 setting1's train.txt.

    image_id matches ``ImgRtvDataset``'s `os.path.relpath(img_path, root)`,
    which is just the leading token in train.txt (e.g. ``image/n02xxx_yyy.JPEG``).

    Captioning the train split is sufficient: text-guided routing only fires
    in `model.train()`. Test/database splits use codebook_mean fallback at
    extraction time.
    """
    base = imagenet_root.rstrip("/")
    seen = set()
    for split in splits:
        split_txt = os.path.join(base, setting_dir, f"{split}.txt")
        if not os.path.exists(split_txt):
            raise FileNotFoundError(f"[iter_imagenet100_setting1] missing {split_txt}")
        for relpath in _read_imagenet_split_paths(split_txt):
            if relpath in seen:
                continue
            seen.add(relpath)
            full = os.path.join(base, relpath)
            yield relpath, split, full


def iter_cifar10(cifar10_root: str):
    """Yield (image_id, split, raw_index, PIL.Image) for every CIFAR10 sample.

    Both train (50k) and test (10k) are emitted with stable md5 ids so the
    same id resolves regardless of how the runtime dataloader sub-samples.
    """
    try:
        from torchvision.datasets import CIFAR10
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "[preprocess --cifar10] torchvision is required to load CIFAR10."
        ) from e
    for split, train in [("train", True), ("test", False)]:
        ds = CIFAR10(cifar10_root, train=train, download=True)
        for i in range(len(ds.data)):
            arr = ds.data[i]                    # uint8 [32, 32, 3]
            iid = _cifar10_image_id(arr)
            pil = Image.fromarray(arr)          # in-memory; no disk dump
            yield iid, split, int(i), pil


def _setting1_uniform_sampling(targets, n_class: int, n_samples_per_class: int, offset: int = 0):
    """Mirror of `dataloaders.get_idx_for_uniform_sampling`. Keep in sync.

    Deterministic with `np.random.seed(0)`. Picks `n_samples_per_class` indices
    per class from `targets`, returning a flat 1D array.
    """
    import numpy as np
    L = np.array(targets)
    out = []
    for label in range(n_class):
        index = np.where(L == label)[0]
        N = index.shape[0]
        np.random.seed(0)
        perm = np.random.permutation(N)
        index = index[perm]
        index = index[offset:offset + n_samples_per_class]
        out.append(index)
    return np.concatenate(out)


def iter_cifar10_setting1(cifar10_root: str):
    """Yield only the (5K train + 1K test) subset that setting1 actually uses.

    Mirrors `ImgRtvCIFAR10` setting1 sampling so we don't waste hours
    captioning images the dataloader will never read.
    """
    try:
        from torchvision.datasets import CIFAR10
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "[preprocess --cifar10_setting1] torchvision is required to load CIFAR10."
        ) from e
    # train subset: 500 / class, seed=0
    trainset = CIFAR10(cifar10_root, train=True, download=True)
    train_idx = _setting1_uniform_sampling(trainset.targets, 10, 500, offset=0)
    for i in train_idx:
        arr = trainset.data[int(i)]
        iid = _cifar10_image_id(arr)
        yield iid, "train", int(i), Image.fromarray(arr)
    # test subset: 100 / class, seed=0  (the query split)
    testset = CIFAR10(cifar10_root, train=False, download=True)
    test_idx = _setting1_uniform_sampling(testset.targets, 10, 100, offset=0)
    for i in test_idx:
        arr = testset.data[int(i)]
        iid = _cifar10_image_id(arr)
        yield iid, "test", int(i), Image.fromarray(arr)


# ----------------------------- shared helpers

def already_done(cache_path: str) -> Set[str]:
    """Read the cache and return the set of already-processed image_ids."""
    done: Set[str] = set()
    if not os.path.exists(cache_path):
        return done
    with open(cache_path, "r") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            iid = row.get("image_id")
            if iid is not None and "codebook_texts" in row:
                done.add(iid)
    return done


def main() -> int:
    parser = argparse.ArgumentParser()
    # mode select
    parser.add_argument("--cifar10", action="store_true",
                        help="Use CIFAR10 (md5-keyed) mode -- caption ALL 60K images.")
    parser.add_argument("--cifar10_setting1", action="store_true",
                        help="Use CIFAR10 (md5-keyed) mode -- caption only the deterministic "
                             "5K train + 1K test subset that `ImgRtvCIFAR10` setting1 reads.")
    parser.add_argument("--cifar10_root", default="./dataset/CIFAR10",
                        help="torchvision.datasets.CIFAR10 download / cache root.")
    parser.add_argument("--imagenet100_setting1", action="store_true",
                        help="Caption only the images listed in setting1/train.txt of "
                             "the ImageNet100 layout (path-keyed). Test/database "
                             "are skipped because text routing is train-only.")
    parser.add_argument("--imagenet100_root", default="./dataset/ImageNet100",
                        help="ImageNet100 root containing setting1/{train,test,database}.txt "
                             "and image/*.JPEG (matches ImgRtvDataset).")
    parser.add_argument("--imagenet100_splits", default="train",
                        help="Comma-separated subset of {train,test,database} to caption "
                             "(default: train).")
    parser.add_argument("--split_txt", default=None,
                        help="Generic mode: read image relpaths from this split file "
                             "(e.g. setting1/train.txt). Each line's first whitespace "
                             "token is treated as image_id (matches ImgRtvDataset). "
                             "Used together with --root for resolving full paths. "
                             "Use this for MSCOCO / Flickr25k / any txt-driven dataset.")
    # walk-mode args (optional when --cifar10 is set)
    parser.add_argument("--image_dir", default=None,
                        help="root directory to walk for images (filesystem-walk mode).")
    parser.add_argument("--root", default=None,
                        help="path prefix to strip when computing image_id "
                             "(default: --image_dir).")
    # shared args
    parser.add_argument("--cache_path", required=True,
                        help="output JSONL path (append-mode safe).")
    parser.add_argument("--prompt_version", choices=("v1", "v2", "v3", "v4"), default="v1",
                        help="Which prompt schema to use. v1 = legacy "
                             "head/body/limb decomposition. v2 = scene-aware "
                             "with BARE NOUN PHRASES per slot. v3 = scene-aware "
                             "(same six keys as v2) but each slot is a short "
                             "CAPTION-STYLE SENTENCE (~15-25 words) -- better "
                             "match to the SigLIP2 text encoder's training "
                             "distribution. Cache files should be kept distinct "
                             "between versions.")
    parser.add_argument("--vlm_name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_new_tokens", type=int, default=384)
    parser.add_argument("--limit", type=int, default=-1,
                        help="only process the first N items (smoke test).")
    args = parser.parse_args()

    has_any_mode = (args.cifar10 or args.cifar10_setting1
                    or args.imagenet100_setting1 or args.image_dir
                    or args.split_txt)
    if not has_any_mode:
        raise SystemExit(
            "[preprocess] need one of --cifar10 / --cifar10_setting1 / "
            "--imagenet100_setting1 / --split_txt / --image_dir."
        )
    if args.cifar10 and args.cifar10_setting1:
        raise SystemExit("[preprocess] --cifar10 and --cifar10_setting1 are mutually exclusive.")
    if args.imagenet100_setting1 and (args.cifar10 or args.cifar10_setting1):
        raise SystemExit("[preprocess] --imagenet100_setting1 cannot combine with cifar10 modes.")

    os.makedirs(os.path.dirname(os.path.abspath(args.cache_path)) or ".", exist_ok=True)
    done = already_done(args.cache_path)

    # Build the iteration of (image_id, payload, image_or_path).
    # `payload` is extra metadata to embed in the JSONL row.
    if args.split_txt is not None:
        if args.root is None:
            raise SystemExit("[preprocess --split_txt] need --root for path resolution.")
        relpaths: list = []
        with open(args.split_txt, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                relpaths.append(line.split()[0])
        if args.limit > 0:
            relpaths = relpaths[: args.limit]
        # de-dup while preserving order
        seen = set()
        relpaths = [rp for rp in relpaths if not (rp in seen or seen.add(rp))]
        total = len(relpaths)
        print(f"[preprocess split_txt={args.split_txt}] {total} candidate images, "
              f"{len(done)} already cached")

        def gen_items():
            for rp in relpaths:
                full = os.path.join(args.root, rp)
                yield rp, {"image_path": full}, full

    elif args.imagenet100_setting1:
        splits = tuple(s.strip() for s in args.imagenet100_splits.split(",") if s.strip())
        candidates_in = list(iter_imagenet100_setting1(
            args.imagenet100_root, "setting1", splits=splits,
        ))
        if args.limit > 0:
            candidates_in = candidates_in[: args.limit]
        total = len(candidates_in)
        print(f"[preprocess ImageNet100 setting1 splits={splits}] "
              f"{total} candidate images, {len(done)} already cached")

        def gen_items():
            for iid, split, full_path in candidates_in:
                yield iid, {"split": split, "image_path": full_path}, full_path

    elif args.cifar10 or args.cifar10_setting1:
        # eagerly materialize candidate list so we can apply --limit and report
        # progress against a known total (CIFAR10 is small, so OK in memory).
        if args.cifar10_setting1:
            candidates = list(iter_cifar10_setting1(args.cifar10_root))
            mode_label = "CIFAR10 setting1"
        else:
            candidates = list(iter_cifar10(args.cifar10_root))
            mode_label = "CIFAR10 (full)"
        if args.limit > 0:
            candidates = candidates[: args.limit]
        total = len(candidates)
        print(f"[preprocess {mode_label}] {total} candidate images, {len(done)} already cached")

        def gen_items():
            for iid, split, idx, pil in candidates:
                yield iid, {"split": split, "index": idx}, pil
    else:
        root = args.root or args.image_dir
        paths = list_images(args.image_dir)
        if args.limit > 0:
            paths = paths[: args.limit]
        total = len(paths)
        print(f"[preprocess walk] {total} candidate images, {len(done)} already cached")

        def gen_items():
            for path in paths:
                iid = os.path.relpath(path, root)
                yield iid, {"image_path": path}, path

    # ---- dispatch to V1 / V2 / V3 prompt + key schema -------------------
    if args.prompt_version == "v4":
        prompt_str = _PROMPT_V4
        out_keys   = _CB_KEYS_V4
    elif args.prompt_version == "v3":
        prompt_str = _PROMPT_V3
        out_keys   = _CB_KEYS_V3
    elif args.prompt_version == "v2":
        prompt_str = _PROMPT_V2
        out_keys   = _CB_KEYS_V2
    else:
        prompt_str = _PROMPT_V1
        out_keys   = _CB_KEYS_V1
    print(f"[preprocess] prompt_version={args.prompt_version}  keys={out_keys}")

    print(f"[preprocess] loading VLM {args.vlm_name} on {args.device} ...")
    model, processor = build_qwen25_vl_generator(
        model_name=args.vlm_name, device=args.device,
    )

    f = open(args.cache_path, "a")
    try:
        for iid, payload, image_or_path in tqdm(gen_items(), total=total, desc="qwen-extract"):
            if iid in done:
                continue
            row = {"image_id": iid, **payload}
            try:
                desc = generate_object_centric_scene_graph(
                    image_or_path, model=model, processor=processor,
                    prompt=prompt_str,
                    max_new_tokens=args.max_new_tokens,
                )
                # Pull texts in the SCHEMA ORDER for this prompt version (the
                # generic extractor auto-detects which schema the JSON uses).
                cb = desc.get("codebook_texts", {}) or {}
                texts = [str(cb.get(k, "") or "").strip() for k in out_keys]
                row["codebook_texts"] = {k: t for k, t in zip(out_keys, texts)}
                row["prompt_version"] = args.prompt_version
            except Exception as e:  # pragma: no cover
                row["error"] = str(e)
                print(f"[preprocess] ERROR on {iid}: {e}", file=sys.stderr)
                traceback.print_exc()
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
    finally:
        f.close()
    print(f"[preprocess] done -> {args.cache_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
