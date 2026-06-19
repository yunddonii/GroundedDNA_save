"""Generate V6 (anatomical) Qwen3-VL captions for the CUB-200 trainset.

v6_cub = single-bird fine-grained captions with STRICTLY DISJOINT
ANATOMICAL regions (head/bill, upperparts/wing, underparts,
tail/appendages, background) instead of the V2/V4/V5b scene axes.

KEY REMAP (critical for downstream compatibility):
The VLM is prompted with the V6 CUB schema (C_head_bill, ...), but the
CLIP text/token feature extractors and the dataloader are POSITION-based
on the V3/V4 key tuple. So before caching we re-map V6 keys onto the
V3V4 positions:

    C_global          -> C_global
    C_head_bill       -> C_primary_object
    C_upperparts_wing -> C_secondary_object
    C_underparts      -> C_activity_or_relation
    C_tail_appendages -> C_color_texture
    C_background      -> C_scene_type

`codebook_texts`     : V3V4-keyed (consumed by extract_clip_text*_features.py)
`codebook_texts_v6`  : original V6-keyed (interpretability / inspection only)

image_id == the leading relpath token of setting1/train.txt
(e.g. "images/001.Black_footed_Albatross/...jpg"), matching
extract_clip_features._iter_pathlist and ImgRtvDataset.

Usage (sample validation, 24 images on one GPU):
    CUDA_VISIBLE_DEVICES=0 python tools/qwen3_v6_cub_trainset.py \\
        --sample 24 \\
        --out_path cache/cub200_qwen_v6b_sample24.jsonl

Usage (full 5994 trainset, single GPU):
    CUDA_VISIBLE_DEVICES=0 python tools/qwen3_v6_cub_trainset.py \\
        --out_path cache/cub200_qwen_v6b_trainset.jsonl

Usage (multi-GPU 6-shard):
    for i in 0 1 2 3 4 5; do
      CUDA_VISIBLE_DEVICES=$i python tools/qwen3_v6_cub_trainset.py \\
          --shard_id $i --n_shards 6 \\
          > logs/qwen3v6b_cub_shard$i.log 2>&1 &
    done
    wait
    cat cache/cub200_qwen_v6b_trainset.shard*.jsonl \\
        > cache/cub200_qwen_v6b_trainset.jsonl
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

sys.path.insert(0, "/home/yschoi/GroundedDNA")
from dna_utils.vlm_qwen25_descriptions import (
    _PROMPT_V6B_CUB,
    CODEBOOK_KEYS_V6B_CUB,
    _strip_code_fences,
)

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
BATCH = 4
MAX_NEW_TOKENS = 384
PROMPT_VERSION = "v6b_cub"

CUB_ROOT = "/home/yschoi/GroundedDNA/dataset/CUB_200"
TRAIN_TXT = os.path.join(CUB_ROOT, "setting1", "train.txt")

# V6 anatomical key -> V3V4 positional key (downstream extractors expect V3V4).
V6_TO_V3V4 = {
    "C_global":          "C_global",
    "C_head_bill":       "C_primary_object",
    "C_upperparts_wing": "C_secondary_object",
    "C_underparts":      "C_activity_or_relation",
    "C_tail_appendages": "C_color_texture",
    "C_pattern_markings": "C_scene_type",
}


def _load_train_pairs():
    """Return list of (image_id, full_path) for CUB setting1 train split.

    image_id = leading relpath token of train.txt (matches the convention in
    extract_clip_features._iter_pathlist / ImgRtvDataset).
    """
    pairs = []
    with open(TRAIN_TXT) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            relpath = line.split()[0]
            pairs.append((relpath, os.path.join(CUB_ROOT, relpath)))
    return pairs


def _already_done(jsonl_path):
    if not jsonl_path.exists():
        return set()
    done = set()
    with open(jsonl_path) as f:
        for line in f:
            try:
                r = json.loads(line)
                if "image_id" in r and r.get("codebook_texts"):
                    done.add(r["image_id"])
            except json.JSONDecodeError:
                continue
    return done


def _remap_to_v3v4(cb_v6: dict) -> dict:
    """Re-key a V6 codebook_texts dict onto the V3V4 positional keys."""
    return {
        V6_TO_V3V4[k]: str(cb_v6.get(k, "") or "").strip()
        for k in CODEBOOK_KEYS_V6B_CUB
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--out_path", default=None)
    ap.add_argument("--sample", type=int, default=None,
                    help="If set, process only first N images (prompt validation).")
    args = ap.parse_args()

    if args.out_path is None:
        if args.n_shards > 1:
            args.out_path = f"/home/yschoi/GroundedDNA/cache/cub200_qwen_v6b_trainset.shard{args.shard_id}.jsonl"
        else:
            args.out_path = "/home/yschoi/GroundedDNA/cache/cub200_qwen_v6b_trainset.jsonl"
    out_path = Path(args.out_path)

    all_pairs = _load_train_pairs()
    shard = [p for i, p in enumerate(all_pairs) if i % args.n_shards == args.shard_id]
    if args.sample is not None:
        shard = shard[:args.sample]
    print(f"CUB trainset total: {len(all_pairs)}, shard {args.shard_id}/{args.n_shards}: "
          f"{len(shard)} images"
          + (f"  [SAMPLE MODE, n={args.sample}]" if args.sample else ""))

    done = _already_done(out_path)
    pairs = shard
    if done:
        print(f"resuming: {len(done)} already cached")
        pairs = [(iid, p) for (iid, p) in pairs if iid not in done]
        print(f"  remaining: {len(pairs)}")
    if not pairs:
        print("nothing to do, all cached.")
        return

    print(f"loading {MODEL_ID} on cuda:0 (bfloat16)...")
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    if hasattr(proc, "tokenizer"):
        proc.tokenizer.padding_side = "left"
    mdl = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16,
    ).cuda().eval()
    print(f"  loaded in {time.time()-t0:.0f}s")

    chat_template = proc.apply_chat_template(
        [{"role": "user", "content": [
            {"type": "image", "image": None},
            {"type": "text",  "text": _PROMPT_V6B_CUB},
        ]}],
        tokenize=False, add_generation_prompt=True,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fp = open(out_path, "a")
    t_start = time.time()
    n_parse_fail = 0

    for s in range(0, len(pairs), BATCH):
        batch = pairs[s:s + BATCH]
        batch_imgs = [Image.open(p).convert("RGB") for _, p in batch]
        texts = [chat_template] * len(batch_imgs)
        inp = proc(text=texts, images=batch_imgs,
                   return_tensors="pt", padding=True).to(mdl.device)
        with torch.no_grad():
            out = mdl.generate(**inp, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        gen = out[:, inp.input_ids.shape[1]:]
        raws = proc.batch_decode(gen, skip_special_tokens=True)
        for (iid, p), raw in zip(batch, raws):
            try:
                d = json.loads(_strip_code_fences(raw))
                cb_v6 = d.get("codebook_texts", {}) or {}
                cb_v3v4 = _remap_to_v3v4(cb_v6)
                rec = {
                    "image_id": iid,
                    "image_path": p,
                    "codebook_texts": cb_v3v4,
                    "codebook_texts_v6b": cb_v6,
                    "prompt_version": PROMPT_VERSION,
                    "vlm": MODEL_ID,
                }
            except Exception as e:
                n_parse_fail += 1
                rec = {
                    "image_id": iid,
                    "image_path": p,
                    "codebook_texts": {},
                    "prompt_version": PROMPT_VERSION,
                    "vlm": MODEL_ID,
                    "_parse_error": str(e),
                    "_raw": raw[:500],
                }
            fp.write(json.dumps(rec) + "\n")
        fp.flush()
        done_n = s + len(batch_imgs)
        if done_n % 20 == 0 or done_n == len(pairs):
            elapsed = time.time() - t_start
            rate = done_n / elapsed
            eta = (len(pairs) - done_n) / rate if rate > 0 else float("inf")
            print(f"  [{done_n}/{len(pairs)}] {elapsed:.0f}s, {rate:.2f} img/s, "
                  f"ETA {eta/60:.0f} min, parse_fail={n_parse_fail}")

    fp.close()
    print(f"\nDONE. saved -> {out_path}")
    print(f"total parse failures: {n_parse_fail}/{len(pairs)}")


if __name__ == "__main__":
    main()
