"""Stage-2 step 4: per-image attribute captions with Qwen3-VL (PROMPT_CAPTIONS_TEMPLATE + attributes.json).

Row schema (one per image):
    image_id, image_path,
    codebook_texts      {C_primary_object, C_secondary_object, C_activity_or_relation, C_color_texture,
                         C_global, C_scene_type: ""}   <- legacy keys via attributes.json key_map
    codebook_texts_v10  {<attribute key>: ...}         <- the VLM's own keys
    prompt_version "v10", prompt_sha256 (of the FILLED prompt), vlm, attributes_sha256
A row whose JSON does not parse, or where any of the 4 attribute keys or C_global is missing/empty,
carries ``_parse_error`` and is retried on rerun.

``--ids all_opt`` + ``--limit N`` (pilot): N ids sampled with ``--seed`` from opt_train_rows excluding
the survey samples (main + extra) of ``--out_dir``; the sample is fixed in ``captions_pilot_sample.json``.
``--ids all_train`` (full): every train row (``setting1/train.txt``, which the extractor needs).
``--ids <file.json>``: an explicit list of ids (or a sample json with "image_ids").

Usage:
    CUDA_VISIBLE_DEVICES=1 python tools/stage2_captions.py --dataset Flickr25k --tag pilot \
        --attributes cache_eval/stage2/flickr25k/attributes.json --ids all_opt --limit 500 \
        --shard_id 0 --n_shards 4 --out_dir cache_eval/stage2/flickr25k
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage2_common import (LEGACY_KEYS, MODEL_ID, DatasetSpec, done_ids, image_chat, load_cache_ids,  # noqa: E402
                           load_vlm, opt_train_ids, parse_json_object, run_image_job, sample_ids, sha256_file,
                           shard_of, train_id_to_path, write_json_once)
from stage2_prompts import PROMPT_VERSION, build_caption_prompt, sha256_of  # noqa: E402

MAX_NEW_TOKENS = 384


def load_attributes(path: str) -> dict:
    a = json.load(open(path))
    attrs = a["attributes"]
    if len(attrs) != 4 or any(not x.get("key") or not x.get("definition") for x in attrs):
        raise SystemExit(f"{path}: needs exactly 4 attributes with key and definition")
    km = a["key_map"]
    if [km[x["key"]] for x in attrs] != list(LEGACY_KEYS[1:5]):
        raise SystemExit(f"{path}: key_map is not the positional map onto {LEGACY_KEYS[1:5]}")
    a["_sha256"] = sha256_file(path)
    return a


def parse_caption_reply(raw: str, attr_keys, key_map) -> dict:
    """-> {"codebook_texts": six legacy keys, "codebook_texts_v10": original keys}; raises on any gap."""
    d = parse_json_object(raw)
    cb = d.get("codebook_texts")
    if not isinstance(cb, dict):
        cb = d if all(k in d for k in attr_keys) else None
    if not isinstance(cb, dict):
        raise ValueError('missing "codebook_texts" object')
    expected = list(attr_keys) + ["C_global"]
    v10 = {}
    missing = []
    for k in expected:
        v = cb.get(k)
        if isinstance(v, (list, tuple)):
            v = " ".join(str(x) for x in v)
        v = " ".join(str(v).split()) if v is not None else ""
        if not v:
            missing.append(k)
        v10[k] = v
    if missing:
        raise ValueError(f"missing or empty keys: {missing}; got keys {list(cb)}")
    legacy = {key_map[k]: v10[k] for k in attr_keys}
    legacy["C_global"] = v10["C_global"]
    legacy["C_scene_type"] = ""
    legacy = {k: legacy[k] for k in LEGACY_KEYS}
    return {"codebook_texts": legacy, "codebook_texts_v10": v10}


def make_row_factory(spec: DatasetSpec, attributes: dict, prompt_sha: str):
    attr_keys = [x["key"] for x in attributes["attributes"]]
    key_map = attributes["key_map"]

    def make_row(path: str, raw: str) -> dict:
        rec = {"image_id": spec.id_for_path(path), "image_path": path, "codebook_texts": {},
               "codebook_texts_v10": {}, "prompt_version": PROMPT_VERSION, "prompt_sha256": prompt_sha,
               "vlm": MODEL_ID, "attributes_sha256": attributes["_sha256"]}
        try:
            rec.update(parse_caption_reply(raw, attr_keys, key_map))
        except Exception as e:  # noqa: BLE001
            rec["_parse_error"] = str(e)
            rec["_raw"] = raw[:500]
        return rec
    return make_row


def resolve_ids(spec: DatasetSpec, ids_arg: str, limit: int, seed: int, out_dir: str, tag: str):
    """-> (ordered ids, {id: path}); writes captions_<tag>_sample.json once when a sample is drawn."""
    if ids_arg == "all_train":
        cache_ids, _ = load_cache_ids(spec)
        id2path = train_id_to_path(spec, cache_ids)
        ids = [i for i in cache_ids if i in id2path]           # train rows in cache order
        sample = write_json_once(os.path.join(out_dir, f"captions_{tag}_sample.json"), {
            "dataset": spec.loader, "kind": "all_train", "n": len(ids), "image_ids": ids})
        return sample["image_ids"], id2path
    if ids_arg == "all_opt":
        opt, id2path = opt_train_ids(spec)
        exclude = []
        for side in ("survey_sample.json", "survey_extra_sample.json"):
            p = os.path.join(out_dir, side)
            if os.path.exists(p):
                exclude += json.load(open(p))["image_ids"]
        ids = sample_ids(opt, limit, seed, exclude) if limit and limit > 0 else [i for i in opt if i not in set(exclude)]
        sample = write_json_once(os.path.join(out_dir, f"captions_{tag}_sample.json"), {
            "dataset": spec.loader, "kind": "opt_sample", "n": len(ids), "seed": seed, "limit": limit,
            "excluded_survey_ids": len(set(exclude)), "image_ids": ids})
        return sample["image_ids"], id2path
    cache_ids, _ = load_cache_ids(spec)
    id2path = train_id_to_path(spec, cache_ids)
    obj = json.load(open(ids_arg))
    ids = obj["image_ids"] if isinstance(obj, dict) else list(obj)
    if limit and limit > 0:
        ids = ids[:limit]
    bad = [i for i in ids if i not in id2path]
    if bad:
        raise SystemExit(f"{len(bad)} ids in {ids_arg} are not train images (first: {bad[:3]})")
    return ids, id2path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--attributes", required=True)
    ap.add_argument("--ids", default="all_opt", help='"all_opt" | "all_train" | path to a json id list')
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20261009)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tag", choices=["pilot", "full"], required=True)
    ap.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    a = ap.parse_args(argv)
    spec = DatasetSpec(a.dataset)
    attributes = load_attributes(a.attributes)
    prompt = build_caption_prompt(attributes["attributes"])
    prompt_sha = sha256_of(prompt)
    ids, id2path = resolve_ids(spec, a.ids, a.limit, a.seed, a.out_dir, a.tag)
    out_path = os.path.join(a.out_dir, f"captions_{a.tag}.shard{a.shard_id}.jsonl")
    shard = shard_of(ids, a.shard_id, a.n_shards)
    done = done_ids([out_path])
    todo = [id2path[i] for i in shard if i not in done]
    print(f"{spec.loader} captions[{a.tag}]: {len(ids)} ids, shard {a.shard_id}/{a.n_shards}: {len(shard)}, "
          f"done {len(shard) - len(todo)}, remaining {len(todo)}; prompt sha {prompt_sha[:12]} "
          f"attributes sha {attributes['_sha256'][:12]}", flush=True)
    json.dump({"prompt": prompt, "prompt_sha256": prompt_sha, "attributes_sha256": attributes["_sha256"]},
              open(os.path.join(a.out_dir, f"captions_{a.tag}_prompt.json"), "w"), indent=1)
    if not todo:
        print("nothing to do")
        return 0
    proc, mdl = load_vlm()
    chat = image_chat(proc, prompt)
    stats = run_image_job(proc, mdl, chat, todo, spec.id_for_path, make_row_factory(spec, attributes, prompt_sha),
                          out_path, a.max_new_tokens)
    print(f"DONE -> {out_path} {stats}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
