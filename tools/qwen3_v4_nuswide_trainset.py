"""Generate V4 Qwen3-VL captions for NUS-WIDE trainset (10,500 balanced subset).

Single-GPU execution (no sharding).

The 10,500 balanced subset was built by scripts/build_nuswide_10500_subset.py
(500 per tag × 21 tags → dedup + refill uniform → 10,500 rows).

PROMPT_V4 is recommended: NUS-WIDE = Flickr photos = same domain as
Flickr25k (where V4 is validated). V5b's strict disjoint-vocab was designed
for the redundant MSCOCO baseline and over-sharpens on already-disjoint
photo captions.

Usage:
    CUDA_VISIBLE_DEVICES=0 python tools/qwen3_v4_nuswide_trainset.py

ETA: ~5 hours single GPU, batch=4.
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
DATASET_ROOT = "/home/yschoi/GroundedDNA/dataset/NUSWIDE"
SUBSET_FILE  = f"{DATASET_ROOT}/setting1/train_10500.txt"
DEFAULT_OUT  = "/home/yschoi/GroundedDNA/cache/nuswide_qwen3_v4_trainset.jsonl"


def _load_subset_paths():
    paths = []
    with open(SUBSET_FILE) as f:
        for ln in f:
            ln = ln.strip()
            if not ln: continue
            rel = ln.split()[0]                          # e.g. images/actor/0001_...jpg
            paths.append(os.path.join(DATASET_ROOT, rel))
    return paths


def _id_for_path(p: str) -> str:
    root = DATASET_ROOT + "/"
    return p[len(root):] if p.startswith(root) else os.path.basename(p)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_path", default=DEFAULT_OUT)
    args = ap.parse_args()
    out_path = Path(args.out_path)

    paths = _load_subset_paths()
    print(f"NUS-WIDE 10,500 subset: {len(paths)} images")

    done = _already_done(out_path)
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
    print(f"  loaded in {time.time()-t0:.0f}s "
          f"({sum(p.numel() for p in mdl.parameters())/1e9:.1f}B)")

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
        try:
            batch_imgs = [Image.open(p).convert("RGB") for p in batch_paths]
        except Exception as e:
            print(f"[skip-batch {s}] image load failed: {e}")
            for p in batch_paths:
                fp.write(json.dumps({"image_id": _id_for_path(p), "image_path": p,
                                      "codebook_texts": {}, "prompt_version": "v4",
                                      "vlm": MODEL_ID, "_load_error": str(e)}) + "\n")
            continue
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
                rec = {"image_id": iid, "image_path": p,
                        "codebook_texts": d.get("codebook_texts", {}),
                        "prompt_version": "v4", "vlm": MODEL_ID}
            except Exception as e:
                n_parse_fail += 1
                rec = {"image_id": iid, "image_path": p,
                        "codebook_texts": {}, "prompt_version": "v4",
                        "vlm": MODEL_ID, "_parse_error": str(e),
                        "_raw": raw[:500]}
            fp.write(json.dumps(rec) + "\n")
        fp.flush()
        done_n = s + len(batch_imgs)
        if done_n % 100 == 0 or done_n == len(paths):
            elapsed = time.time() - t_start
            rate = done_n / elapsed
            eta = (len(paths) - done_n) / rate if rate > 0 else float("inf")
            print(f"  [{done_n}/{len(paths)}] {elapsed:.0f}s elapsed, "
                  f"{rate:.2f} img/s, ETA {eta/60:.0f} min, "
                  f"parse_fail={n_parse_fail}")

    fp.close()
    print(f"\nDONE. saved -> {out_path}")
    print(f"total parse failures: {n_parse_fail}/{len(paths)}")


if __name__ == "__main__":
    main()
