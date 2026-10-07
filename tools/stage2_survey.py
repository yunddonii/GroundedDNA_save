"""Stage-2 step 1: per-image phrase survey with Qwen3-VL (PROMPT_SURVEY).

Samples ``--n`` opt-train images (cache ``opt_train_rows`` -> ``image_ids.json`` -> train image path),
writes ``<out_dir>/survey_sample.json`` once (first shard wins) and generates phrases per image into
``<out_dir>/survey.shard{k}.jsonl``.  Rows with ``_parse_error`` are not counted as done, so a rerun
retries them.  ``--extra_n N`` switches to the saturation sample: N more ids disjoint from the main
sample (``survey_extra_sample.json``, ``survey_extra.shard{k}.jsonl``).

Usage (one shard per process; two processes fit on one 49 GB GPU):
    CUDA_VISIBLE_DEVICES=1 python tools/stage2_survey.py --dataset Flickr25k --shard_id 0 --n_shards 4 \
        --out_dir cache_eval/stage2/flickr25k
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage2_common import (MODEL_ID, DatasetSpec, done_ids, image_chat, load_vlm, opt_train_ids,  # noqa: E402
                           parse_json_object, run_image_job, sample_ids, shard_of, write_json_once)
from stage2_prompts import PROMPT_SURVEY, PROMPT_VERSION, sha256_of  # noqa: E402

MAX_NEW_TOKENS = 256
PROMPT_SHA = sha256_of(PROMPT_SURVEY)


def parse_survey(raw: str) -> list:
    """Strict: object with a "phrases" list of non-empty strings."""
    d = parse_json_object(raw)
    ph = d.get("phrases")
    if not isinstance(ph, list):
        raise ValueError('missing "phrases" list')
    out = [str(p).strip() for p in ph if isinstance(p, (str, int, float)) and str(p).strip()]
    if not out:
        raise ValueError("empty phrase list")
    return out


def make_row_factory(spec: DatasetSpec):
    def make_row(path: str, raw: str) -> dict:
        rec = {"image_id": spec.id_for_path(path), "image_path": path, "phrases": [],
               "prompt_version": PROMPT_VERSION, "prompt_sha256": PROMPT_SHA, "vlm": MODEL_ID}
        try:
            rec["phrases"] = parse_survey(raw)
        except Exception as e:  # noqa: BLE001
            rec["_parse_error"] = str(e)
            rec["_raw"] = raw[:500]
        return rec
    return make_row


def build_sample(spec: DatasetSpec, out_dir: str, n: int, seed: int, extra_n: int = 0, extra_seed: int = 0):
    """Returns (sample dict, jsonl stem) for the main or the extra sample."""
    opt, id2path = opt_train_ids(spec)
    main = write_json_once(os.path.join(out_dir, "survey_sample.json"), {
        "dataset": spec.loader, "kind": "survey_main", "n": n, "seed": seed, "n_opt_train_rows": len(opt),
        "image_ids": sample_ids(opt, n, seed), "prompt_sha256": PROMPT_SHA, "prompt_version": PROMPT_VERSION,
        "vlm": MODEL_ID})
    main["paths"] = [id2path[i] for i in main["image_ids"]]
    if extra_n <= 0:
        return main, "survey"
    extra = write_json_once(os.path.join(out_dir, "survey_extra_sample.json"), {
        "dataset": spec.loader, "kind": "survey_extra", "n": extra_n, "seed": extra_seed,
        "n_opt_train_rows": len(opt), "excluded_main_n": len(main["image_ids"]),
        "image_ids": sample_ids(opt, extra_n, extra_seed, exclude=main["image_ids"]),
        "prompt_sha256": PROMPT_SHA, "prompt_version": PROMPT_VERSION, "vlm": MODEL_ID})
    if set(extra["image_ids"]) & set(main["image_ids"]):
        raise SystemExit("extra sample overlaps the main sample")
    extra["paths"] = [id2path[i] for i in extra["image_ids"]]
    return extra, "survey_extra"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="Flickr25k | NUS-WIDE | MS-COCO")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261007)
    ap.add_argument("--extra_n", type=int, default=0, help="> 0: generate the disjoint saturation sample instead")
    ap.add_argument("--extra_seed", type=int, default=20261008)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    a = ap.parse_args(argv)
    spec = DatasetSpec(a.dataset)
    sample, stem = build_sample(spec, a.out_dir, a.n, a.seed, a.extra_n, a.extra_seed)
    out_path = os.path.join(a.out_dir, f"{stem}.shard{a.shard_id}.jsonl")
    shard = shard_of(sample["paths"], a.shard_id, a.n_shards)
    done = done_ids([out_path])
    todo = [p for p in shard if spec.id_for_path(p) not in done]
    print(f"{spec.loader} {stem}: sample {len(sample['image_ids'])}, shard {a.shard_id}/{a.n_shards}: "
          f"{len(shard)}, done {len(shard) - len(todo)}, remaining {len(todo)}", flush=True)
    if not todo:
        print("nothing to do")
        return 0
    proc, mdl = load_vlm()
    chat = image_chat(proc, PROMPT_SURVEY)
    stats = run_image_job(proc, mdl, chat, todo, spec.id_for_path, make_row_factory(spec), out_path,
                          a.max_new_tokens)
    print(f"DONE -> {out_path} {stats}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
