"""Qwen3-VL CUB trainset captioner using PROMPT_V7 (6 anatomy-only slots).

V7 changes vs V6b:
- REMOVES C_pattern_markings (20.7% hedged in v6b) and redistributes
  markings into their natural anatomy slots.
- SPLITS the v6b C_head_bill into C_head_face_eye + C_bill.
- ABSOLUTELY bans hedging language ("no distinct", "not visible", etc.)
- ABSOLUTELY bans "bird" family words in ALL slots including C_global
  (v6b had 55% "bird" occurrence in C_global — massive text_part cos-sim
  inflation source).

Positional mapping onto V3V4 downstream keys (backward-compat with
extract_clip_text*_features.py and dataloaders):

    V7 key            -> V3V4 positional key
    C_global          -> C_global
    C_head_face_eye   -> C_primary_object
    C_bill            -> C_secondary_object
    C_wing_back       -> C_activity_or_relation
    C_underparts      -> C_color_texture
    C_tail_legs       -> C_scene_type

image_id == leading relpath token of setting1/train.txt.

Usage (sample 24 for prompt validation):
    CUDA_VISIBLE_DEVICES=0 python tools/qwen3_v7_cub_trainset.py \\
        --sample 24 \\
        --out_path cache/cub200_qwen_v7_1_sample24.jsonl

Usage (full 5994, single GPU):
    CUDA_VISIBLE_DEVICES=0 python tools/qwen3_v7_cub_trainset.py \\
        --out_path cache/cub200_qwen_v7_1_trainset.jsonl

Usage (6-shard parallel):
    for i in 0 1 2 3 4 5; do
      CUDA_VISIBLE_DEVICES=$i python tools/qwen3_v7_cub_trainset.py \\
          --shard_id $i --n_shards 6 \\
          > logs/qwen3v7_1_cub_shard$i.log 2>&1 &
    done
    wait
    cat cache/cub200_qwen_v7_1_trainset.shard*.jsonl \\
        > cache/cub200_qwen_v7_1_trainset.jsonl
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

sys.path.insert(0, "/home/yschoi/GroundedDNA")
from dna_utils.vlm_qwen25_descriptions import (
    _PROMPT_V7_1_CUB,
    CODEBOOK_KEYS_V7_1_CUB,
    _strip_code_fences,
)

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
BATCH = 4
MAX_NEW_TOKENS = 384
PROMPT_VERSION = "v7_1_cub"

CUB_ROOT = "/home/yschoi/GroundedDNA/dataset/CUB_200"
TRAIN_TXT = os.path.join(CUB_ROOT, "setting1", "train.txt")

# V7 anatomical key -> V3V4 positional key (downstream extractors expect V3V4).
V7_TO_V3V4 = {
    "C_global":         "C_global",
    "C_head_face_eye":  "C_primary_object",
    "C_bill":           "C_secondary_object",
    "C_wing_back":      "C_activity_or_relation",
    "C_underparts":     "C_color_texture",
    "C_tail_legs":      "C_scene_type",
}


def _load_train_pairs():
    pairs = []
    with open(TRAIN_TXT) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            relpath = line.split()[0]
            pairs.append((relpath, os.path.join(CUB_ROOT, relpath)))
    return pairs


def _already_done(jsonl_path):
    if not jsonl_path.exists(): return set()
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


def _remap_to_v3v4(cb_v7: dict) -> dict:
    """Re-key a V7 codebook_texts dict onto the V3V4 positional keys."""
    return {V7_TO_V3V4[k]: v for k, v in cb_v7.items() if k in V7_TO_V3V4}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_path", type=str, default=None)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--sample", type=int, default=None,
                    help="If set, process only first N images (prompt validation).")
    args = ap.parse_args()

    if args.out_path is None:
        if args.n_shards > 1:
            args.out_path = f"/home/yschoi/GroundedDNA/cache/cub200_qwen_v7_1_trainset.shard{args.shard_id}.jsonl"
        else:
            args.out_path = "/home/yschoi/GroundedDNA/cache/cub200_qwen_v7_1_trainset.jsonl"
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
            {"type": "text",  "text": _PROMPT_V7_1_CUB},
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
                cb_v7 = d.get("codebook_texts", {}) or {}
                cb_v3v4 = _remap_to_v3v4(cb_v7)
                rec = {
                    "image_id": iid,
                    "image_path": p,
                    "codebook_texts": cb_v3v4,
                    "codebook_texts_v7_1": cb_v7,
                    "prompt_version": PROMPT_VERSION,
                    "vlm": MODEL_ID,
                }
            except Exception as e:
                n_parse_fail += 1
                rec = {
                    "image_id": iid,
                    "image_path": p,
                    "codebook_texts": {},
                    "codebook_texts_v7_1": {},
                    "prompt_version": PROMPT_VERSION,
                    "vlm": MODEL_ID,
                    "parse_error": str(e),
                    "raw_output": raw,
                }
            fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fp.flush()

        elapsed = time.time() - t_start
        done_so_far = s + len(batch)
        rate = done_so_far / max(elapsed, 1e-3)
        eta_sec = max(0.0, (len(pairs) - done_so_far)) / max(rate, 1e-6)
        print(f"  [{done_so_far}/{len(pairs)}] elapsed={elapsed:.0f}s "
              f"rate={rate:.2f}img/s eta={eta_sec/60:.1f}min "
              f"parse_fail={n_parse_fail}")

    fp.close()
    print(f"[done] wrote {out_path}. parse_fail={n_parse_fail}/{len(pairs)}")


if __name__ == "__main__":
    main()
