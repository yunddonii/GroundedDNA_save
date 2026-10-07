"""Shared helpers for the Stage-2 caption pipeline (survey / concepts / captions / validate).

Conventions (verified 2026-10-07 on the three v6prov caches):
  * ``image_id`` = image path relative to ``<DATASET_DIR>/<Loader>/`` (e.g. ``images/im15316.jpg``,
    ``images/saguaro/0427_2609543221.jpg``, ``images/train2014/COCO_train2014_000000440877.jpg``);
    this equals the entries of the cache's ``image_ids.json`` and of the V4/V5b caption jsonl files.
  * the train split (``setting1/train.txt``, loader order) is ``image_ids[:N_train]`` for all three
    datasets, and ``opt_train_rows.npy`` indexes ``image_ids.json`` directly.
  * a jsonl row counts as DONE only when it has no ``_parse_error``; a rerun retries the others.

Only ``image_ids.json`` and ``opt_train_rows.npy`` are read from the feature caches.

CLI (used by the orchestration script):
    python tools/stage2_common.py concat --out OUT [--expect SAMPLE_JSON] [--max_missing_frac F] IN...
    python tools/stage2_common.py check  --expect SAMPLE_JSON IN...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

WORKTREE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKTREE not in sys.path:
    sys.path.insert(0, WORKTREE)

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
BATCH = 4
DATASET_DIR = "/home/yschoi/GroundedDNA/dataset"
CACHE_ROOT = "/data/yschoi/groundeddna_cache_v6prov"
LEGACY_LOCAL_KEYS = ("C_primary_object", "C_secondary_object", "C_activity_or_relation", "C_color_texture")
LEGACY_KEYS = ("C_global",) + LEGACY_LOCAL_KEYS + ("C_scene_type",)

# accepted --dataset spellings -> (loader name, cache/output short name)
_DATASET_ALIASES = {
    "flickr25k": ("Flickr25k", "flickr25k"),
    "flickr": ("Flickr25k", "flickr25k"),
    "nuswide": ("NUSWIDE", "nuswide"),
    "nus-wide": ("NUSWIDE", "nuswide"),
    "nus": ("NUSWIDE", "nuswide"),
    "mscoco": ("MSCOCO", "mscoco"),
    "ms-coco": ("MSCOCO", "mscoco"),
    "coco": ("MSCOCO", "mscoco"),
}


class DatasetSpec:
    def __init__(self, name: str):
        key = name.strip().lower()
        if key not in _DATASET_ALIASES:
            raise SystemExit(f"unknown dataset {name!r}; use one of Flickr25k, NUS-WIDE, MS-COCO")
        self.loader, self.short = _DATASET_ALIASES[key]
        self.root = f"{DATASET_DIR}/{self.loader}/"
        self.cache_dir = f"{CACHE_ROOT}/{self.short}_clip_tokens"

    def id_for_path(self, p: str) -> str:
        p = str(p)
        return p[len(self.root):] if p.startswith(self.root) else os.path.basename(p)

    def path_for_id(self, iid: str) -> str:
        return self.root + iid


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------- ids and paths
def load_cache_ids(spec: DatasetSpec):
    """(image_ids list, opt_train_rows int array) from the feature cache's two small files."""
    ids = json.load(open(os.path.join(spec.cache_dir, "image_ids.json")))
    rows = np.load(os.path.join(spec.cache_dir, "opt_train_rows.npy"))
    rows = np.asarray(rows, dtype=np.int64)
    if rows.min() < 0 or rows.max() >= len(ids):
        raise SystemExit(f"opt_train_rows out of range for image_ids ({rows.min()}..{rows.max()} vs {len(ids)})")
    return ids, rows


def load_train_paths(spec: DatasetSpec) -> List[str]:
    """Train split image paths in loader order (mirrors tools/qwen3_v4_*_trainset.py)."""
    from dataloaders import load_dataset
    from dna_utils import get_transform
    t = get_transform("test")
    tr, _, _ = load_dataset(DATASET_DIR, spec.loader, setting="setting1", train_transform=t, test_transform=t,
                            load_train=True, load_database=False, load_test=False, return_index=True,
                            qwen_text_cache_path=None, siglip2_feature_cache_dir=None)
    return [str(p) for p in tr.img_paths]


def train_id_to_path(spec: DatasetSpec, cache_ids: Optional[Sequence[str]] = None) -> Dict[str, str]:
    """{image_id: image_path} over the train split; every id is checked against image_ids.json."""
    paths = load_train_paths(spec)
    m = {spec.id_for_path(p): p for p in paths}
    if cache_ids is not None:
        cset = set(cache_ids)
        missing = [i for i in m if i not in cset]
        if missing:
            raise SystemExit(f"{len(missing)} train ids are not in image_ids.json (first: {missing[:3]}); "
                             "id convention mismatch")
    return m


def opt_train_ids(spec: DatasetSpec):
    """Ordered list of opt-train image ids (cache row order) plus the id->path map."""
    ids, rows = load_cache_ids(spec)
    id2path = train_id_to_path(spec, ids)
    opt = [ids[int(r)] for r in rows]
    bad = [i for i in opt if i not in id2path]
    if bad:
        raise SystemExit(f"{len(bad)} opt_train ids are not in the train split (first: {bad[:3]})")
    return opt, id2path


def sample_ids(pool: Sequence[str], n: int, seed: int, exclude: Iterable[str] = ()) -> List[str]:
    """Deterministic sample of n ids from pool minus exclude, keeping pool order."""
    ex = set(exclude)
    cand = [i for i in pool if i not in ex]
    if n > len(cand):
        raise SystemExit(f"cannot sample {n} ids from {len(cand)} candidates")
    rng = np.random.default_rng(seed)
    pick = sorted(rng.choice(len(cand), size=n, replace=False).tolist())
    return [cand[i] for i in pick]


def shard_of(items: Sequence, shard_id: int, n_shards: int) -> list:
    return [x for i, x in enumerate(items) if i % n_shards == shard_id]


def write_json_once(path, payload: dict) -> dict:
    """First writer wins (atomic rename); later callers read what is there."""
    path = Path(path)
    if path.exists():
        return json.load(open(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=1)
    if path.exists():                      # lost the race; keep the other writer's file
        tmp.unlink()
        return json.load(open(path))
    os.replace(tmp, path)
    return payload


# --------------------------------------------------------------------------- jsonl
def iter_jsonl(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def read_rows(paths: Iterable) -> Dict[str, dict]:
    """{image_id: row}; the LAST row without _parse_error wins; ids with only failed rows are absent."""
    ok: Dict[str, dict] = {}
    for p in paths:
        if not Path(p).exists():
            continue
        for r in iter_jsonl(p):
            iid = r.get("image_id")
            if iid is None or r.get("_parse_error"):
                continue
            ok[iid] = r
    return ok


def failed_rows(paths: Iterable) -> Dict[str, dict]:
    """{image_id: last failed row} for ids that never produced a clean row."""
    last_fail: Dict[str, dict] = {}
    ok = set()
    for p in paths:
        if not Path(p).exists():
            continue
        for r in iter_jsonl(p):
            iid = r.get("image_id")
            if iid is None:
                continue
            if r.get("_parse_error"):
                last_fail[iid] = r
            else:
                ok.add(iid)
    return {k: v for k, v in last_fail.items() if k not in ok}


def done_ids(paths: Iterable) -> set:
    return set(read_rows(paths))


# --------------------------------------------------------------------------- VLM
def _lenient_json_candidates(text: str):
    """Repairs for the bracket typos Qwen3-VL makes on ~2 % of replies (seen 2026-10-07: the
    object closed with ']]' instead of ']}', or the final '}' missing). Only the ending is touched."""
    t = text.strip()
    yield t
    if t.endswith("]]"):
        yield t[:-2] + "]}"
    if t.endswith("]"):
        yield t + "}"
    if t.endswith('"'):
        yield t + "]}"
    i, j = t.find("{"), t.rfind("}")
    if i >= 0 and j > i:
        yield t[i:j + 1]
        if t[i:j + 1].endswith("]]}"):
            yield t[i:j + 1][:-3] + "]}"


def parse_json_object(raw: str) -> dict:
    from dna_utils.vlm_qwen25_descriptions import _strip_code_fences
    text = _strip_code_fences(raw)
    err = None
    for cand in _lenient_json_candidates(text):
        try:
            d = json.loads(cand)
        except Exception as e:          # noqa: BLE001 - try the next repair
            err = err or e
            continue
        if not isinstance(d, dict):
            raise ValueError(f"top-level JSON is {type(d).__name__}, not an object")
        return d
    raise err if err is not None else ValueError("empty reply")


def load_vlm(model_id: str = MODEL_ID):
    """Qwen3-VL-8B-Instruct, bf16, on the single visible GPU (mirrors tools/qwen3_v4_flickr25k_evaldb.py)."""
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(model_id)
    if hasattr(proc, "tokenizer"):
        proc.tokenizer.padding_side = "left"
    mdl = Qwen3VLForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16).cuda().eval()
    print(f"loaded {model_id} in {time.time() - t0:.0f}s", flush=True)
    return proc, mdl


def image_chat(proc, prompt: str) -> str:
    return proc.apply_chat_template(
        [{"role": "user", "content": [{"type": "image", "image": None}, {"type": "text", "text": prompt}]}],
        tokenize=False, add_generation_prompt=True)


def text_chat(proc, prompt: str) -> str:
    return proc.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        tokenize=False, add_generation_prompt=True)


def generate_images(proc, mdl, chat: str, paths: Sequence[str], max_new_tokens: int) -> List[str]:
    """Greedy generation for a batch of image paths with one shared prompt; returns decoded strings."""
    import torch
    from PIL import Image
    imgs = [Image.open(p).convert("RGB") for p in paths]
    inp = proc(text=[chat] * len(imgs), images=imgs, return_tensors="pt", padding=True).to(mdl.device)
    with torch.no_grad():
        out = mdl.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
    return proc.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)


def make_text_generator(proc, mdl, max_new_tokens: int, repetition_penalty: float = 1.0) -> Callable[..., str]:
    """prompt -> decoded greedy completion (text-only chat, no image).

    ``gen(prompt, repetition_penalty=1.1)`` overrides the default penalty for one call (the concept
    step's single retry of a degenerate reply).  After every call ``gen.last_n_tokens`` and
    ``gen.last_hit_cap`` (generation stopped at max_new_tokens) describe the reply."""
    import torch

    def gen(prompt: str, repetition_penalty: Optional[float] = None) -> str:
        rp = repetition_penalty if repetition_penalty is not None else gen.default_repetition_penalty
        chat = text_chat(proc, prompt)
        inp = proc(text=[chat], return_tensors="pt", padding=True).to(mdl.device)
        with torch.no_grad():
            out = mdl.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=float(rp))
        n_new = int(out.shape[1] - inp.input_ids.shape[1])
        gen.last_n_tokens, gen.last_hit_cap = n_new, n_new >= max_new_tokens
        return proc.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0]

    gen.default_repetition_penalty = float(repetition_penalty)
    gen.max_new_tokens = int(max_new_tokens)
    gen.last_n_tokens, gen.last_hit_cap = 0, False
    return gen


def run_image_job(proc, mdl, chat: str, items: Sequence[str], id_of: Callable[[str], str],
                  make_row: Callable[[str, str], dict], out_path, max_new_tokens: int,
                  batch: int = BATCH, retry_failed: bool = True, log_every: int = 100) -> dict:
    """Generate for every path in ``items`` (append to out_path), then retry this run's failures once
    with batch 1 (different padding; greedy decoding is otherwise deterministic).  ``make_row(path, raw)``
    returns the row and sets ``_parse_error`` on failure.  Returns counts."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fp = open(out_path, "a")
    t0 = time.time()
    fail_paths: List[str] = []
    n_fail = 0

    def one_pass(paths: Sequence[str], bs: int, attempt: int):
        nonlocal n_fail
        for s in range(0, len(paths), bs):
            bp = paths[s:s + bs]
            raws = generate_images(proc, mdl, chat, bp, max_new_tokens)
            for p, raw in zip(bp, raws):
                row = make_row(p, raw)
                row["_attempt"] = attempt
                if row.get("_parse_error"):
                    n_fail += 1
                    if attempt == 1:
                        fail_paths.append(p)
                fp.write(json.dumps(row) + "\n")
            fp.flush()
            n = s + len(bp)
            if n % log_every == 0 or n == len(paths):
                el = time.time() - t0
                print(f"  [attempt {attempt}: {n}/{len(paths)}] {el:.0f}s {n / max(el, 1e-6):.2f} img/s "
                      f"parse_fail={n_fail}", flush=True)

    one_pass(list(items), batch, 1)
    if retry_failed and fail_paths:
        print(f"retrying {len(fail_paths)} failed rows with batch 1", flush=True)
        one_pass(fail_paths, 1, 2)
    fp.close()
    still = failed_rows([out_path])
    still = {k for k in still if k in {id_of(p) for p in items}}
    return {"n_items": len(items), "n_fail_events": n_fail, "n_still_failed": len(still), "seconds": time.time() - t0}


# --------------------------------------------------------------------------- CLI: concat / check
def _concat(args):
    expected = None
    if args.expect:
        expected = json.load(open(args.expect))["image_ids"]
    rows = read_rows(args.inputs)
    fails = failed_rows(args.inputs)
    keep = expected if expected is not None else sorted(rows)
    missing = [i for i in keep if i not in rows]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for iid in keep:
            if iid in rows:
                f.write(json.dumps(rows[iid]) + "\n")
    frac = len(missing) / max(len(keep), 1)
    print(f"concat -> {out}: {len(keep) - len(missing)} rows, {len(missing)} missing "
          f"({frac:.4f}), {len(fails)} ids with only failed rows")
    if missing[:5]:
        print("  first missing:", missing[:5])
    if frac > args.max_missing_frac:
        print(f"FAIL: missing fraction {frac:.4f} > {args.max_missing_frac}")
        return 2
    return 0


def _check(args):
    expected = json.load(open(args.expect))["image_ids"]
    rows = read_rows(args.inputs)
    missing = [i for i in expected if i not in rows]
    print(f"check: {len(expected) - len(missing)}/{len(expected)} done, {len(missing)} missing")
    return 0 if not missing else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("concat")
    c.add_argument("--out", required=True)
    c.add_argument("--expect", default=None, help="sample json with image_ids; keeps that order")
    c.add_argument("--max_missing_frac", type=float, default=0.01)
    c.add_argument("inputs", nargs="+")
    k = sub.add_parser("check")
    k.add_argument("--expect", required=True)
    k.add_argument("inputs", nargs="+")
    a = ap.parse_args(argv)
    return _concat(a) if a.cmd == "concat" else _check(a)


if __name__ == "__main__":
    sys.exit(main())
