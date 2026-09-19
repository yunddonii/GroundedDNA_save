#!/usr/bin/env python
"""Build the (b-1) masked-entity-completion cache (branch arch-exp-2026-09).

Adapts OVSegmentor's question/answer construction
(datasets/clip_dataset.py::build_question_and_answer @ cfcbc3a) to the four
local caption axes of this project:

  entities   OVSegmentor masks words from a fixed list of 73 object classes.
             The axes here are not all objects (colour/texture, activity), so
             each axis gets its own list: the TOP_K most frequent content words
             of that axis's captions over the OPTIMISATION rows only (the
             held-out validation rows never shape the vocabulary).
  question   the axis caption with every entity word replaced by "M", which the
             CLIP tokenizer maps to id 332 ('m</w>') -- the id OVSegmentor uses.
  answer     for every one of OVSegmentor's 80 ImageNet templates:
             template.split('{}')[0] + ' and '.join(entities), entities in order
             of first appearance (OVSegmentor uses set order, which varies run to
             run).
  encoding   frozen CLIP ViT-B/16 from the pinned snapshot. Question tokens use
             exactly the feature cache's recipe (text_projection applied to
             text_model.last_hidden_state, max_length 32); answers use
             get_text_features (the 512-d joint space).
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
from collections import Counter

import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.pretrained_backbone import coerce_pooled_to_tensor  # same conversion as the feature cache

AXES = ("C_primary_object", "C_secondary_object", "C_activity_or_relation", "C_color_texture")
BOILERPLATE = {
    "image", "images", "photo", "picture", "scene", "frame", "visible", "appears", "appear",
    "shows", "show", "showing", "shown", "features", "featuring", "feature", "seen",
    "suggests", "suggesting", "suggest", "indicating", "indicates", "possibly", "likely",
    "slightly", "partially", "various", "several", "another", "also", "within",
    "background", "foreground",
}
WORD = re.compile(r"[A-Za-z]+")


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen_jsonl", required=True)
    ap.add_argument("--feature_cache_dir", required=True)
    ap.add_argument("--clip_snapshot_dir", required=True)
    ap.add_argument("--ovseg_templates_py", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--top_k", type=int, default=100)
    ap.add_argument("--min_df", type=int, default=10)
    ap.add_argument("--max_len", type=int, default=32)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    if os.path.exists(os.path.join(a.out_dir, "meta.json")):
        raise SystemExit(f"{a.out_dir} already holds a cache; refusing to overwrite")
    os.makedirs(a.out_dir, exist_ok=True)

    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    stop = set(ENGLISH_STOP_WORDS) | BOILERPLATE

    src = open(a.ovseg_templates_py).read()
    templates = ast.literal_eval(re.search(r"full_imagenet_templates = (\[.*?\])", src, re.S).group(1))

    caps = {}
    for line in open(a.qwen_jsonl):
        d = json.loads(line)
        caps[str(d["image_id"])] = {k: str(d["codebook_texts"].get(k, "") or "").strip() for k in AXES}
    ids = sorted(caps)

    fc_ids = json.load(open(os.path.join(a.feature_cache_dir, "image_ids.json")))
    opt_rows = np.load(os.path.join(a.feature_cache_dir, "opt_train_rows.npy"))
    opt_ids = {str(fc_ids[int(r)]) for r in opt_rows}
    vocab_ids = [i for i in ids if i in opt_ids]

    vocab, df_tab = {}, {}
    for ax in AXES:
        df = Counter()
        for i in vocab_ids:
            words = {w.lower() for w in WORD.findall(caps[i][ax])}
            df.update(w for w in words if len(w) >= 3 and w not in stop)
        top = [(w, c) for w, c in df.most_common() if c >= a.min_df][: a.top_k]
        vocab[ax] = [w for w, _ in top]
        df_tab[ax] = {w: int(c) for w, c in top}

    R, M = len(ids), len(AXES)
    questions, entities = [], []
    valid = np.zeros((R, M), dtype=np.bool_)
    for r, i in enumerate(ids):
        for m, ax in enumerate(AXES):
            V = set(vocab[ax])
            ents = []
            def sub(mt):
                w = mt.group(0)
                if w.lower() in V:
                    if w.lower() not in ents:
                        ents.append(w.lower())
                    return "M"
                return w
            q = WORD.sub(sub, caps[i][ax])
            questions.append(q)
            entities.append(ents)
            valid[r, m] = len(ents) > 0

    from transformers import CLIPModel, CLIPTokenizer
    tok = CLIPTokenizer.from_pretrained(a.clip_snapshot_dir, local_files_only=True)
    model = CLIPModel.from_pretrained(a.clip_snapshot_dir, local_files_only=True).to(a.device).eval()
    D = int(model.config.projection_dim)

    q_tokens = np.zeros((R * M, a.max_len, D), dtype=np.float16)
    q_mask = np.zeros((R * M, a.max_len), dtype=np.bool_)
    with torch.no_grad():
        for s0 in range(0, R * M, 512):
            enc = tok(questions[s0:s0 + 512], padding="max_length", truncation=True,
                      max_length=a.max_len, return_tensors="pt", return_attention_mask=True)
            ii, am = enc["input_ids"].to(a.device), enc["attention_mask"].to(a.device)
            hid = model.text_model(input_ids=ii, attention_mask=am).last_hidden_state
            q_tokens[s0:s0 + len(ii)] = model.text_projection(hid).float().cpu().numpy().astype(np.float16)
            q_mask[s0:s0 + len(ii)] = am.bool().cpu().numpy()
        n_t = len(templates)
        answers = np.zeros((R * M, n_t, D), dtype=np.float16)
        flat = [(k, t) for k in range(R * M) for t in range(n_t)]
        for s0 in range(0, len(flat), 4096):
            chunk = flat[s0:s0 + 4096]
            texts = [templates[t].split("{}")[0] + " and ".join(entities[k]) for k, t in chunk]
            enc = tok(texts, padding=True, truncation=True, max_length=77, return_tensors="pt")
            f = coerce_pooled_to_tensor(model.get_text_features(
                input_ids=enc["input_ids"].to(a.device), attention_mask=enc["attention_mask"].to(a.device)))
            f = f.float().cpu().numpy().astype(np.float16)
            for j, (k, t) in enumerate(chunk):
                answers[k, t] = f[j]

    tp = np.load(os.path.join(a.feature_cache_dir, "text_part.f16.npy"), mmap_mode="r")
    row_of_fc = {str(x): r for r, x in enumerate(fc_ids)}
    chk = [i for i in ids[:: max(1, R // 24)]][:24]
    with torch.no_grad():
        enc = tok([caps[i][AXES[0]] for i in chk], padding="max_length", truncation=True,
                  max_length=64, return_tensors="pt")
        f = coerce_pooled_to_tensor(model.get_text_features(
            input_ids=enc["input_ids"].to(a.device), attention_mask=enc["attention_mask"].to(a.device)))
        f = torch.nn.functional.normalize(f.float(), dim=-1).cpu()
    ref = torch.nn.functional.normalize(torch.from_numpy(
        np.stack([np.asarray(tp[row_of_fc[i], 1], dtype=np.float32) for i in chk])), dim=-1)
    space_cos = float((f * ref).sum(-1).min())
    print(f"[space check] min cosine(recomputed, cached text_part) over {len(chk)} captions = {space_cos:.6f}")
    if space_cos < 0.999:
        raise SystemExit("answers would not live in the cached text space; refusing")

    np.save(os.path.join(a.out_dir, "q_tokens.f16.npy"), q_tokens.reshape(R, M, a.max_len, D))
    np.save(os.path.join(a.out_dir, "q_mask.bool.npy"), q_mask.reshape(R, M, a.max_len))
    np.save(os.path.join(a.out_dir, "answers.f16.npy"), answers.reshape(R, M, len(templates), D))
    np.save(os.path.join(a.out_dir, "valid.bool.npy"), valid)
    json.dump(ids, open(os.path.join(a.out_dir, "image_ids.json"), "w"))
    json.dump({ax: df_tab[ax] for ax in AXES}, open(os.path.join(a.out_dir, "vocab.json"), "w"), indent=1)
    ex = {ax: [{"image_id": ids[r], "caption": caps[ids[r]][ax], "question": questions[r * M + m],
                "entities": entities[r * M + m],
                "answer_t0": templates[0].split("{}")[0] + " and ".join(entities[r * M + m])}
               for r in range(0, R, R // 6)[:6]] for m, ax in enumerate(AXES)}
    json.dump(ex, open(os.path.join(a.out_dir, "examples.json"), "w"), indent=1)
    ent_counts = np.array([len(e) for e in entities]).reshape(R, M)
    meta = {
        "axes": list(AXES), "rows": R, "text_dim": D, "max_len": a.max_len, "n_templates": len(templates),
        "top_k": a.top_k, "min_df": a.min_df, "vocab_rows": "opt_train_rows.npy", "vocab_images": len(vocab_ids),
        "coverage_per_axis": {ax: round(float(valid[:, m].mean()), 4) for m, ax in enumerate(AXES)},
        "mean_entities_per_axis": {ax: round(float(ent_counts[:, m].mean()), 3) for m, ax in enumerate(AXES)},
        "mask_token_id": int(tok("M")["input_ids"][1]),
        "space_check_min_cos_vs_cached_text_part": round(space_cos, 6),
        "sources": {
            "qwen_jsonl": a.qwen_jsonl, "qwen_jsonl_sha256": sha(a.qwen_jsonl),
            "feature_cache_image_ids_sha256": sha(os.path.join(a.feature_cache_dir, "image_ids.json")),
            "opt_train_rows_sha256": sha(os.path.join(a.feature_cache_dir, "opt_train_rows.npy")),
            "clip_snapshot_dir": a.clip_snapshot_dir,
            "ovseg_templates_py": a.ovseg_templates_py, "ovseg_templates_sha256": sha(a.ovseg_templates_py),
            "ovseg_commit": "cfcbc3ab1a299b14a30a0a8b5d1176f7f6d29cda",
            "builder_sha256": sha(os.path.abspath(__file__)),
        },
    }
    json.dump(meta, open(os.path.join(a.out_dir, "meta.json"), "w"), indent=1)
    print(json.dumps({k: meta[k] for k in ("rows", "text_dim", "coverage_per_axis", "mean_entities_per_axis", "mask_token_id")}))


if __name__ == "__main__":
    main()
