"""Evaluation-only V4 captions for a fixed sample of Flickr25k DATABASE images (text-path line).

Purpose: enlarge the A3 evaluation sample (Stage 1 finding: 500 validation rows cannot resolve an S
below ~.2 on Flickr). These captions are labels for the lexical pair rule ONLY; they are never a
training input, never go into a text cache used by a trainer, and the sampled image_ids are written
to a sidecar so the exclusion can be checked.

Sample = `--n` database images, excluding every training image (setting1 trainset), drawn with a
fixed seed from the database split in its loader order. Prompt = _PROMPT_V4 from THIS worktree.
Usage (one shard per GPU):
    CUDA_VISIBLE_DEVICES=4 python tools/qwen3_v4_flickr25k_evaldb.py --shard_id 0 --n_shards 2 --out_dir <dir>
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

WORKTREE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, WORKTREE)
from dna_utils.vlm_qwen25_descriptions import _PROMPT_V4, _strip_code_fences  # noqa: E402

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
BATCH = 4
MAX_NEW_TOKENS = 384
DATASET_DIR = "/home/yschoi/GroundedDNA/dataset"
ROOT = DATASET_DIR + "/Flickr25k/"


def _id_for_path(p):
    return p[len(ROOT):] if p.startswith(ROOT) else os.path.basename(p)


def _paths(mode):
    from dataloaders import load_dataset
    from dna_utils import get_transform
    t = get_transform("test")
    tr, te, db = load_dataset(DATASET_DIR, "Flickr25k", setting="setting1", train_transform=t, test_transform=t,
                              load_train=(mode == "train"), load_database=(mode == "database"), load_test=False,
                              return_index=True, qwen_text_cache_path=None, siglip2_feature_cache_dir=None)
    ds = tr if mode == "train" else db
    return [ds[i]["image_path"] for i in range(len(ds))]


def sample_paths(n, seed, out_dir):
    side = Path(out_dir) / "evaldb_sample.json"
    if side.exists():
        return json.load(open(side))["paths"]
    train_ids = set(_id_for_path(p) for p in _paths("train"))
    db = [p for p in _paths("database") if _id_for_path(p) not in train_ids]
    rng = np.random.default_rng(seed)
    pick = sorted(rng.choice(len(db), size=n, replace=False).tolist())
    paths = [db[i] for i in pick]
    side.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"n": n, "seed": seed, "database_size_excl_train": len(db), "train_excluded": len(train_ids),
               "image_ids": [_id_for_path(p) for p in paths], "paths": paths,
               "prompt_sha256": hashlib.sha256(_PROMPT_V4.encode()).hexdigest(), "prompt_version": "v4", "vlm": MODEL_ID},
              open(side, "w"), indent=1)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard_id", type=int, default=0); ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--n", type=int, default=1500); ap.add_argument("--seed", type=int, default=20261007)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    paths_all = sample_paths(a.n, a.seed, a.out_dir)
    shard = [p for i, p in enumerate(paths_all) if i % a.n_shards == a.shard_id]
    out_path = Path(a.out_dir) / f"flickr25k_qwen3_v4_evaldb{a.n}.shard{a.shard_id}.jsonl"
    done = set()
    if out_path.exists():
        for line in open(out_path):
            try:
                done.add(json.loads(line)["image_id"])
            except Exception:
                pass
    paths = [p for p in shard if _id_for_path(p) not in done]
    print(f"sample {len(paths_all)}, shard {a.shard_id}/{a.n_shards}: {len(shard)}, remaining {len(paths)}", flush=True)
    if not paths:
        print("nothing to do"); return
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    if hasattr(proc, "tokenizer"):
        proc.tokenizer.padding_side = "left"
    mdl = Qwen3VLForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).cuda().eval()
    chat = proc.apply_chat_template([{"role": "user", "content": [{"type": "image", "image": None}, {"type": "text", "text": _PROMPT_V4}]}],
                                    tokenize=False, add_generation_prompt=True)
    fp = open(out_path, "a"); t0 = time.time(); fail = 0
    for s in range(0, len(paths), BATCH):
        bp = paths[s:s + BATCH]
        imgs = [Image.open(p).convert("RGB") for p in bp]
        inp = proc(text=[chat] * len(imgs), images=imgs, return_tensors="pt", padding=True).to(mdl.device)
        with torch.no_grad():
            out = mdl.generate(**inp, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        raws = proc.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)
        for p, raw in zip(bp, raws):
            rec = {"image_id": _id_for_path(p), "image_path": p, "prompt_version": "v4", "vlm": MODEL_ID, "split": "database_eval_only"}
            try:
                rec["codebook_texts"] = json.loads(_strip_code_fences(raw)).get("codebook_texts", {})
            except Exception as e:
                fail += 1; rec.update(codebook_texts={}, _parse_error=str(e), _raw=raw[:500])
            fp.write(json.dumps(rec) + "\n")
        fp.flush()
        n = s + len(bp)
        if n % 100 == 0 or n == len(paths):
            el = time.time() - t0; print(f"  [{n}/{len(paths)}] {el:.0f}s {n / el:.2f} img/s parse_fail={fail}", flush=True)
    fp.close(); print(f"DONE -> {out_path}")


if __name__ == "__main__":
    main()
