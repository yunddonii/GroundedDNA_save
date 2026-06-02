"""Generate V4 Qwen3-VL captions for MSCOCO trainset (10K images).

Same design as `qwen3_v4_flickr25k_trainset.py` — batched (batch=4) +
multi-GPU sharded. Trainset is 10K, so 5-GPU shards = 2K each → ~60 min.

Usage (single shard):
    CUDA_VISIBLE_DEVICES=1 python tools/qwen3_v4_mscoco_trainset.py \\
        --shard_id 0 --n_shards 4

Usage (4-GPU launcher, GPU 1..4 since GPU 0 is occupied by Task A):
    for i in 0 1 2 3; do
      gpu=$((i+1))
      CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=2 \\
        python tools/qwen3_v4_mscoco_trainset.py \\
            --shard_id $i --n_shards 4 \\
            > logs/qwen3_mscoco_shard$i.log 2>&1 &
    done
    wait
    cat cache/mscoco_qwen3_v4_trainset.shard*.jsonl \\
        > cache/mscoco_qwen3_v4_trainset.jsonl
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

sys.path.insert(0, "/home/yschoi/GroundedDNA")
from dna_utils.vlm_qwen25_descriptions import _PROMPT_V4, _strip_code_fences

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
BATCH = 4
MAX_NEW_TOKENS = 384


def _load_trainset_paths():
    from dataloaders import load_dataset
    from dna_utils import get_transform
    transform = get_transform("test")
    trainset, _, _ = load_dataset(
        "/home/yschoi/GroundedDNA/dataset", "MSCOCO", setting="setting1",
        train_transform=transform, test_transform=transform,
        load_train=True, load_database=False, load_test=False, return_index=True,
        qwen_text_cache_path=None,
        siglip2_feature_cache_dir=None,
    )
    return [trainset[i]["image_path"] for i in range(len(trainset))]


def _already_done(jsonl_path):
    if not jsonl_path.exists():
        return set()
    done = set()
    with open(jsonl_path) as f:
        for line in f:
            try:
                r = json.loads(line)
                if "image_id" in r:
                    done.add(r["image_id"])
            except json.JSONDecodeError:
                continue
    return done


def _id_for_path(p: str) -> str:
    """image_id = relpath from dataset root (matches legacy MSCOCO V4 cache convention)."""
    root = "/home/yschoi/GroundedDNA/dataset/MSCOCO/"
    return p[len(root):] if p.startswith(root) else os.path.basename(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--out_path", default=None)
    args = ap.parse_args()

    if args.out_path is None:
        if args.n_shards > 1:
            args.out_path = f"/home/yschoi/GroundedDNA/cache/mscoco_qwen3_v4_trainset.shard{args.shard_id}.jsonl"
        else:
            args.out_path = "/home/yschoi/GroundedDNA/cache/mscoco_qwen3_v4_trainset.jsonl"
    out_path = Path(args.out_path)

    all_paths = _load_trainset_paths()
    shard = [p for i, p in enumerate(all_paths) if i % args.n_shards == args.shard_id]
    print(f"MSCOCO trainset total: {len(all_paths)}, shard {args.shard_id}/{args.n_shards}: {len(shard)} images")

    done = _already_done(out_path)
    paths = shard
    if done:
        print(f"resuming: {len(done)} already cached")
        paths = [p for p in paths if _id_for_path(p) not in done]
        print(f"  remaining: {len(paths)}")
    if not paths:
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
            {"type": "text",  "text": _PROMPT_V4},
        ]}],
        tokenize=False, add_generation_prompt=True,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fp = open(out_path, "a")
    t_start = time.time()
    n_parse_fail = 0

    for s in range(0, len(paths), BATCH):
        batch_paths = paths[s:s + BATCH]
        batch_imgs = [Image.open(p).convert("RGB") for p in batch_paths]
        texts = [chat_template] * len(batch_imgs)
        inp = proc(text=texts, images=batch_imgs,
                   return_tensors="pt", padding=True).to(mdl.device)
        with torch.no_grad():
            out = mdl.generate(**inp, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        gen = out[:, inp.input_ids.shape[1]:]
        raws = proc.batch_decode(gen, skip_special_tokens=True)
        for p, raw in zip(batch_paths, raws):
            iid = _id_for_path(p)
            try:
                d = json.loads(_strip_code_fences(raw))
                rec = {
                    "image_id": iid,
                    "image_path": p,
                    "codebook_texts": d.get("codebook_texts", {}),
                    "prompt_version": "v4",
                    "vlm": MODEL_ID,
                }
            except Exception as e:
                n_parse_fail += 1
                rec = {
                    "image_id": iid,
                    "image_path": p,
                    "codebook_texts": {},
                    "prompt_version": "v4",
                    "vlm": MODEL_ID,
                    "_parse_error": str(e),
                    "_raw": raw[:500],
                }
            fp.write(json.dumps(rec) + "\n")
        fp.flush()
        done_n = s + len(batch_imgs)
        if done_n % 100 == 0 or done_n == len(paths):
            elapsed = time.time() - t_start
            rate = done_n / elapsed
            eta = (len(paths) - done_n) / rate if rate > 0 else float("inf")
            print(f"  [{done_n}/{len(paths)}] {elapsed:.0f}s, {rate:.2f} img/s, ETA {eta/60:.0f} min, parse_fail={n_parse_fail}")

    fp.close()
    print(f"\nDONE. saved -> {out_path}")
    print(f"total parse failures: {n_parse_fail}/{len(paths)}")


if __name__ == "__main__":
    main()
