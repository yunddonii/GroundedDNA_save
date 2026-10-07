"""Stage-2 steps 2a / 2b / 3: concept list, visual check, attribute grouping (text-only Qwen3-VL).

Input: the survey shards of one dataset (``<in_dir>/survey.shard*.jsonl``; rows with ``_parse_error``
are ignored).  Output (all in ``--out_dir``):
    concepts_shuffle{0,1,2}.json   step 2a per phrase-order shuffle: one concept list PER MINI-BATCH
                                   (PROMPT_CONCEPTS_BATCH, no cumulative input) and their union
                                   de-duplicated by lowercased name (resume unit)
    concepts_consolidated.json     PROMPT_CONCEPTS_CONSOLIDATE over the union of the three shuffle
                                   unions; chunks of CONSOLIDATE_CHUNK when the union is larger, then
                                   the chunk outputs are consolidated once more (re-chunked while they
                                   still exceed the chunk size, at most MAX_CONSOLIDATE_LEVELS levels)
    concepts.json                  per-shuffle sizes, agreement between the three shuffle unions,
                                   consolidation levels, working list = the consolidated list
    concepts_visual.json           working list after PROMPT_VISUAL_CHECK, dropped concepts recorded
    attributes_shuffle{0,1,2}.json PROMPT_ATTRIBUTES on three concept-order shuffles
    attributes.json                final 4 attributes + positional key_map to the legacy slot keys

Why two levels (2026-10-07): the earlier cumulative design (PROMPT_CONCEPTS_UPDATE, kept unused in
stage2_prompts.py) made the model restate the whole list every batch; on NUS-WIDE the reply overran
max_new_tokens and was cut mid-string, on Flickr25k the count oscillated 69 -> 46 -> 81.  Per-batch
extraction + consolidation (X-Cluster) bounds every reply to one batch's or one chunk's concepts.

Strict parsing: a reply whose JSON does not close (after the ending-only repairs in
``stage2_common.parse_json_object``) or lacks the expected list raises ParseFailure -- no silent skip.
Concepts are sent to the model with at most RENDER_MAX_EXAMPLES examples each; files keep them all.
Downstream files (visual check, attribute shuffles) record the sha of the list they were built from
and are recomputed when that list changed, so stale files from an earlier method are never reused.
All model calls go through one ``gen(prompt) -> str`` callable so tests drive the steps with a stub.
Every JSON output is written to a sibling temp file and renamed into place.

Usage:
    CUDA_VISIBLE_DEVICES=1 python tools/stage2_concepts.py --dataset Flickr25k \
        --in_dir cache_eval/stage2/flickr25k --out_dir cache_eval/stage2/flickr25k
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage2_common import LEGACY_LOCAL_KEYS, MODEL_ID, DatasetSpec, parse_json_object, read_rows  # noqa: E402
from stage2_prompts import (PROMPT_ATTRIBUTES, PROMPT_ATTRIBUTES_MERGE, PROMPT_CONCEPTS_BATCH,  # noqa: E402
                            PROMPT_CONCEPTS_CONSOLIDATE, PROMPT_CONCEPTS_UPDATE, PROMPT_SURVEY,
                            PROMPT_VERSION, PROMPT_VISUAL_CHECK, sha256_of)

AGREE_MIN_JACCARD = 0.6
VISUAL_CHECK_CHUNK = 100
CONSOLIDATE_CHUNK = 150
MAX_CONSOLIDATE_LEVELS = 3
RENDER_MAX_EXAMPLES = 2
STEP2A_METHOD = "per_batch_extraction+consolidation"
PROMPT_SHAS = {k: sha256_of(v) for k, v in {
    "survey": PROMPT_SURVEY, "concepts_update": PROMPT_CONCEPTS_UPDATE, "concepts_batch": PROMPT_CONCEPTS_BATCH,
    "concepts_consolidate": PROMPT_CONCEPTS_CONSOLIDATE, "visual_check": PROMPT_VISUAL_CHECK,
    "attributes": PROMPT_ATTRIBUTES, "attributes_merge": PROMPT_ATTRIBUTES_MERGE}.items()}


class ParseFailure(RuntimeError):
    pass


# --------------------------------------------------------------------------- helpers
def _lc(s) -> str:
    return " ".join(str(s).strip().lower().split())


def key_form(name: str) -> str:
    """JSON-key form of an attribute display name: trimmed, spaces -> underscores."""
    return "_".join(str(name).strip().split())


def atomic_json_dump(obj, path) -> None:
    """Write JSON to a sibling temp file and rename it into place (readers never see a partial file)."""
    path = Path(path)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def list_sha(items: Sequence[dict]) -> str:
    """Order-free digest of a concept list by lowercased names (used to detect stale downstream files)."""
    return hashlib.sha256(json.dumps(sorted(_lc(c["name"]) for c in items)).encode("utf-8")).hexdigest()


def norm_concepts(items) -> List[dict]:
    """[{name, examples[]}] with trimmed strings; items without a name are dropped."""
    out = []
    for it in items:
        if isinstance(it, str):
            it = {"name": it, "examples": []}
        if not isinstance(it, dict) or not str(it.get("name", "")).strip():
            continue
        ex = it.get("examples") or []
        if not isinstance(ex, list):
            ex = [ex]
        out.append({"name": str(it["name"]).strip(), "examples": [str(e).strip() for e in ex if str(e).strip()]})
    return out


def dedupe_by_name(items: Sequence[dict]) -> List[dict]:
    seen: Dict[str, dict] = {}
    order = []
    for it in items:
        k = _lc(it["name"])
        if k in seen:
            for e in it["examples"]:
                if _lc(e) not in {_lc(x) for x in seen[k]["examples"]}:
                    seen[k]["examples"].append(e)
        else:
            seen[k] = {"name": it["name"], "examples": list(it["examples"])}
            order.append(k)
    return [seen[k] for k in order]


def strict_list(raw: str, field: str) -> list:
    """Parse a reply and return its ``field`` list; raise ParseFailure otherwise (fail hard)."""
    try:
        d = parse_json_object(raw)
    except Exception as e:  # noqa: BLE001
        raise ParseFailure(f"reply is not a closed JSON object ({e}); head={raw[:200]!r} tail={raw[-120:]!r}")
    if field not in d or not isinstance(d[field], list):
        raise ParseFailure(f'reply lacks a "{field}" list; keys={list(d)}')
    return d[field]


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a | b) else 1.0


def render_concepts(items: Sequence[dict], max_examples: Optional[int] = RENDER_MAX_EXAMPLES) -> str:
    """Model-facing rendering: JSON list with at most ``max_examples`` examples per concept."""
    return json.dumps([{"name": c["name"], "examples": c["examples"][:max_examples] if max_examples is not None
                        else c["examples"]} for c in items], ensure_ascii=False, indent=0)


def _concept_reply(gen: Callable[[str], str], prompt: str) -> List[dict]:
    return dedupe_by_name(norm_concepts(strict_list(gen(prompt), "concepts")))


# --------------------------------------------------------------------------- step 2a (i): per-batch extraction
def concepts_batch_pass(gen: Callable[[str], str], survey: Dict[str, List[str]], seed: int,
                        batch_images: int, log=print) -> dict:
    """One seeded shuffle of the images; every mini-batch is extracted independently (no cumulative
    input).  Returns the per-batch lists and their union de-duplicated by lowercased name."""
    ids = sorted(survey)
    order = np.random.default_rng(seed).permutation(len(ids))
    n_batches = (len(ids) + batch_images - 1) // batch_images
    batch_lists = []
    for b in range(0, len(ids), batch_images):
        batch = [ids[i] for i in order[b:b + batch_images]]
        lines = "\n".join(", ".join(survey[i]) for i in batch)
        lst = _concept_reply(gen, PROMPT_CONCEPTS_BATCH.replace("<PHRASES>", lines))
        batch_lists.append({"batch": b // batch_images, "image_ids": batch, "concepts": lst})
        log(f"  shuffle {seed} batch {b // batch_images + 1}/{n_batches}: {len(lst)} concepts "
            f"(union so far {len(dedupe_by_name([c for bl in batch_lists for c in bl['concepts']]))})")
    union = dedupe_by_name([c for bl in batch_lists for c in bl["concepts"]])
    return {"method": STEP2A_METHOD, "seed": seed, "n_images": len(ids), "batch_images": batch_images,
            "counts_per_batch": [len(bl["concepts"]) for bl in batch_lists], "batch_lists": batch_lists,
            "union": union, "union_size": len(union), "prompt_sha256": PROMPT_SHAS["concepts_batch"]}


# --------------------------------------------------------------------------- step 2a (ii): consolidation
def consolidate(gen: Callable[[str], str], concepts: Sequence[dict], chunk: int = CONSOLIDATE_CHUNK,
                max_levels: int = MAX_CONSOLIDATE_LEVELS, log=print) -> dict:
    """PROMPT_CONCEPTS_CONSOLIDATE with bounded output per call: a list of at most ``chunk`` concepts
    is consolidated in one call; a larger list is consolidated in chunks of ``chunk`` and the merged
    chunk outputs are consolidated once more (re-chunked while they still exceed ``chunk``; stops
    after ``max_levels`` levels or when a level no longer shrinks the list)."""
    cur = dedupe_by_name(concepts)
    levels = []
    for level in range(1, max_levels + 1):
        if len(cur) <= chunk:
            out = _concept_reply(gen, PROMPT_CONCEPTS_CONSOLIDATE.replace("<CONCEPTS>", render_concepts(cur)))
            levels.append({"level": level, "n_in": len(cur), "n_chunks": 1, "chunk_sizes_out": [len(out)], "n_out": len(out),
                           "final": True})
            log(f"  consolidate level {level}: {len(cur)} -> {len(out)} (single call)")
            return {"concepts": out, "levels": levels, "chunk": chunk, "stopped": "single_call"}
        chunks = [cur[i:i + chunk] for i in range(0, len(cur), chunk)]
        outs = [_concept_reply(gen, PROMPT_CONCEPTS_CONSOLIDATE.replace("<CONCEPTS>", render_concepts(ch)))
                for ch in chunks]
        merged = dedupe_by_name([c for o in outs for c in o])
        levels.append({"level": level, "n_in": len(cur), "n_chunks": len(chunks), "chunk_sizes_out": [len(o) for o in outs],
                       "n_out": len(merged), "final": False})
        log(f"  consolidate level {level}: {len(cur)} -> {len(merged)} in {len(chunks)} chunks of <= {chunk}")
        if len(merged) >= len(cur):
            log("  consolidation stopped: the level did not shrink the list")
            return {"concepts": merged, "levels": levels, "chunk": chunk, "stopped": "not_shrinking"}
        cur = merged
    log(f"  consolidation stopped after {max_levels} levels with {len(cur)} concepts (> chunk {chunk})")
    return {"concepts": cur, "levels": levels, "chunk": chunk, "stopped": "max_levels"}


def agreement_report(lists: Sequence[Sequence[dict]]) -> dict:
    names = [set(_lc(c["name"]) for c in L) for L in lists]
    loose = [set(_lc(c["name"]) for c in L) | set(_lc(e) for c in L for e in c["examples"]) for L in lists]
    pair = {}
    for i in range(len(lists)):
        for j in range(i + 1, len(lists)):
            pair[f"{i}-{j}"] = {"jaccard_names": jaccard(names[i], names[j]),
                                "jaccard_names_plus_examples": jaccard(loose[i], loose[j])}
    return {"sizes": [len(L) for L in lists], "pairwise": pair,
            "mean_jaccard_names": float(np.mean([p["jaccard_names"] for p in pair.values()])) if pair else 1.0,
            "mean_jaccard_names_plus_examples": float(np.mean([p["jaccard_names_plus_examples"] for p in pair.values()])) if pair else 1.0}


# --------------------------------------------------------------------------- step 2b
def visual_check(gen: Callable[[str], str], concepts: Sequence[dict], chunk: int = VISUAL_CHECK_CHUNK) -> dict:
    verdict: Dict[str, bool] = {}
    for s in range(0, len(concepts), chunk):
        part = concepts[s:s + chunk]
        prompt = PROMPT_VISUAL_CHECK.replace("<CONCEPTS>", render_concepts(part))
        for it in strict_list(gen(prompt), "concepts"):
            if isinstance(it, dict) and "name" in it:
                v = it.get("visible")
                if isinstance(v, str):
                    v = v.strip().lower() in ("true", "yes", "1")
                verdict[_lc(it["name"])] = bool(v)
    kept, dropped, unanswered = [], [], []
    for c in concepts:
        v = verdict.get(_lc(c["name"]))
        if v is None:
            unanswered.append(c["name"]); kept.append(c)
        elif v:
            kept.append(c)
        else:
            dropped.append(c["name"])
    return {"concepts": kept, "dropped_not_visible": dropped, "unanswered_kept": unanswered}


# --------------------------------------------------------------------------- step 3
def parse_attributes(raw: str) -> dict:
    attrs = strict_list(raw, "attributes")
    out = []
    for a in attrs:
        if not isinstance(a, dict):
            continue
        out.append({"name": str(a.get("name", "")).strip(), "definition": str(a.get("definition", "")).strip(),
                    "concepts": [str(c).strip() for c in (a.get("concepts") or []) if str(c).strip()]})
    try:
        note = parse_json_object(raw).get("note", "")
    except Exception:  # noqa: BLE001
        note = ""
    return {"attributes": out, "note": str(note or "")}


def groupings_agree(groupings: Sequence[dict], min_jaccard: float = AGREE_MIN_JACCARD) -> dict:
    """Same 4 names (case-insensitive) in every grouping AND mean membership Jaccard >= min_jaccard
    after greedy name matching against grouping 0."""
    g0 = groupings[0]["attributes"]
    names0 = sorted(_lc(a["name"]) for a in g0)
    detail = []
    agree = all(len(g["attributes"]) == 4 for g in groupings)
    for gi in groupings[1:]:
        gi_attrs = gi["attributes"]
        same_names = sorted(_lc(a["name"]) for a in gi_attrs) == names0 and len(set(names0)) == 4
        # greedy matching on membership Jaccard
        sets0 = [set(_lc(c) for c in a["concepts"]) for a in g0]
        setsi = [set(_lc(c) for c in a["concepts"]) for a in gi_attrs]
        cand = sorted(((jaccard(sets0[p], setsi[q]), p, q) for p in range(len(sets0)) for q in range(len(setsi))),
                      reverse=True)
        used_p, used_q, matched = set(), set(), []
        for j, p, q in cand:
            if p in used_p or q in used_q:
                continue
            used_p.add(p); used_q.add(q); matched.append((j, p, q))
        mean_j = float(np.mean([m[0] for m in matched])) if matched else 0.0
        ok = same_names and mean_j >= min_jaccard and len(gi_attrs) == 4
        agree = agree and ok
        detail.append({"same_names": same_names, "mean_membership_jaccard": mean_j, "ok": ok,
                       "matches": [{"g0": g0[p]["name"], "gi": gi_attrs[q]["name"], "jaccard": j} for j, p, q in matched]})
    return {"agree": bool(agree), "min_jaccard": min_jaccard, "vs_shuffle0": detail}


def validate_attributes(attrs: Sequence[dict]) -> List[dict]:
    if len(attrs) != 4:
        raise ParseFailure(f"expected exactly 4 attributes, got {len(attrs)}")
    out = []
    for a in attrs:
        name, definition = str(a.get("name", "")).strip(), str(a.get("definition", "")).strip()
        if not name or not definition:
            raise ParseFailure(f"attribute with empty name or definition: {a!r}")
        key = key_form(name)
        if not key or any(ch in key for ch in '"\\\n'):
            raise ParseFailure(f"attribute name not usable as a JSON key: {name!r}")
        out.append({"name": name, "key": key, "definition": definition, "concepts": list(a.get("concepts") or [])})
    names = [_lc(a["name"]) for a in out]
    keys = [a["key"].lower() for a in out]
    if len(set(names)) != 4 or len(set(keys)) != 4 or any(k == "c_global" for k in keys):
        raise ParseFailure(f"attribute names must be distinct and not C_global: {names}")
    return out


def key_map_by_position(attrs: Sequence[dict]) -> Dict[str, str]:
    """Positional map attribute key -> legacy slot key (A1->C_primary_object, ..., A4->C_color_texture)."""
    if len(attrs) != 4:
        raise ParseFailure(f"key_map needs 4 attributes, got {len(attrs)}")
    return {a["key"]: legacy for a, legacy in zip(attrs, LEGACY_LOCAL_KEYS)}


def attributes_step(gen: Callable[[str], str], concepts: Sequence[dict], out_dir, shuffles: int = 3, log=print) -> dict:
    src_sha = list_sha(concepts)
    groupings = []
    for s in range(shuffles):
        p = Path(out_dir) / f"attributes_shuffle{s}.json"
        if p.exists():
            g = json.load(open(p))
            if g.get("concepts_sha256") == src_sha:
                groupings.append(g); continue
            log(f"  attributes shuffle {s}: {p} was built from a different concept list; recomputing")
        order = np.random.default_rng(s).permutation(len(concepts))
        prompt = PROMPT_ATTRIBUTES.replace("<CONCEPTS>", render_concepts([concepts[i] for i in order]))
        g = parse_attributes(gen(prompt))
        g.update(seed=s, prompt_sha256=PROMPT_SHAS["attributes"], concepts_sha256=src_sha)
        atomic_json_dump(g, p)
        groupings.append(g)
        log(f"  attributes shuffle {s}: {[a['name'] for a in g['attributes']]} note={g['note']!r}")
    agreement = groupings_agree(groupings)
    if agreement["agree"]:
        final, source = groupings[0], "shuffle0"
    else:
        prompt = PROMPT_ATTRIBUTES_MERGE
        for i, g in enumerate(groupings[:3], start=1):
            prompt = prompt.replace(f"<G{i}>", json.dumps(
                {"attributes": g["attributes"], "note": g.get("note", "")}, ensure_ascii=False, indent=0))
        final, source = parse_attributes(gen(prompt)), "merge"
        log(f"  merged attributes: {[a['name'] for a in final['attributes']]} note={final['note']!r}")
    attrs = validate_attributes(final["attributes"])
    return {"attributes": attrs, "note": final.get("note", ""), "key_map": key_map_by_position(attrs),
            "legacy_keys_in_order": list(LEGACY_LOCAL_KEYS), "source": source, "agreement": agreement,
            "concepts_sha256": src_sha, "prompt_sha256s": PROMPT_SHAS, "prompt_version": PROMPT_VERSION, "vlm": MODEL_ID}


# --------------------------------------------------------------------------- driver
def run_all(gen: Callable[[str], str], survey: Dict[str, List[str]], out_dir, batch_images: int = 80,
            shuffles: int = 3, dataset: str = "", log=print, consolidate_chunk: int = CONSOLIDATE_CHUNK) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # 2a (i): per-batch extraction, one file per shuffle (resume unit)
    passes = []
    for s in range(shuffles):
        p = out_dir / f"concepts_shuffle{s}.json"
        if p.exists():
            r = json.load(open(p))
            if r.get("method") == STEP2A_METHOD and "union" in r:
                passes.append(r); log(f"  shuffle {s}: reusing {p}"); continue
            log(f"  shuffle {s}: {p} is from an earlier method; recomputing")
        r = concepts_batch_pass(gen, survey, s, batch_images, log)
        atomic_json_dump(r, p)
        passes.append(r)
    unions = [r["union"] for r in passes]
    union_all = dedupe_by_name([c for u in unions for c in u])
    union_sha = list_sha(union_all)
    # 2a (ii): consolidation over the union of the shuffle unions
    cons_p = out_dir / "concepts_consolidated.json"
    cons = None
    if cons_p.exists():
        cons = json.load(open(cons_p))
        if cons.get("union_sha256") != union_sha:
            log(f"  {cons_p} was built from a different union; recomputing"); cons = None
    if cons is None:
        cons = consolidate(gen, union_all, consolidate_chunk, log=log)
        cons.update(union_sha256=union_sha, n_union_in=len(union_all), prompt_sha256=PROMPT_SHAS["concepts_consolidate"])
        atomic_json_dump(cons, cons_p)
    working = cons["concepts"]
    concepts_json = {"dataset": dataset, "method": STEP2A_METHOD, "n_images": len(survey), "batch_images": batch_images,
                     "shuffles": [{"seed": r["seed"], "n_union": r["union_size"], "counts_per_batch": r["counts_per_batch"],
                                   "union": r["union"]} for r in passes],
                     "agreement": agreement_report(unions),
                     "n_union_all": len(union_all), "consolidation": {k: cons[k] for k in ("levels", "chunk", "stopped")},
                     "n_consolidated": len(working),
                     "working_list_rule": "per-batch extraction (PROMPT_CONCEPTS_BATCH, 3 shuffles) -> union of all batch "
                                          "lists de-duplicated by lowercased name (examples merged) -> "
                                          "PROMPT_CONCEPTS_CONSOLIDATE (chunked when > chunk); working = consolidated list",
                     "working": working, "prompt_sha256s": {k: PROMPT_SHAS[k] for k in ("concepts_batch", "concepts_consolidate")},
                     "vlm": MODEL_ID}
    atomic_json_dump(concepts_json, out_dir / "concepts.json")
    log(f"step 2a: per-shuffle unions {[len(u) for u in unions]}, union of unions {len(union_all)}, consolidated "
        f"{len(working)}; agreement {concepts_json['agreement']}")
    # 2b
    working_sha = list_sha(working)
    vis_p = out_dir / "concepts_visual.json"
    vis = None
    if vis_p.exists():
        vis = json.load(open(vis_p))
        if vis.get("working_sha256") != working_sha:
            log(f"  {vis_p} was built from a different working list; recomputing"); vis = None
    if vis is None:
        vis = visual_check(gen, working)
        vis.update(n_before=len(working), n_after=len(vis["concepts"]), working_sha256=working_sha,
                   prompt_sha256=PROMPT_SHAS["visual_check"])
        atomic_json_dump(vis, vis_p)
    log(f"visual check: kept {len(vis['concepts'])}, dropped {len(vis['dropped_not_visible'])}, "
        f"unanswered {len(vis['unanswered_kept'])}")
    # 3
    attrs = attributes_step(gen, vis["concepts"], out_dir, shuffles, log)
    attrs.update(dataset=dataset, n_concepts=len(vis["concepts"]))
    atomic_json_dump(attrs, out_dir / "attributes.json")
    log("ATTRIBUTES: " + json.dumps([{a["key"]: a["definition"]} for a in attrs["attributes"]], ensure_ascii=False))
    log(f"key_map: {attrs['key_map']}  source={attrs['source']}")
    return attrs


def load_survey(in_dir, pattern: str = "survey.shard*.jsonl") -> Dict[str, List[str]]:
    files = sorted(glob.glob(os.path.join(in_dir, pattern)))
    if not files:
        raise SystemExit(f"no survey shards under {in_dir} ({pattern})")
    rows = read_rows(files)
    return {iid: r["phrases"] for iid, r in rows.items() if r.get("phrases")}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_images", type=int, default=80)
    ap.add_argument("--shuffles", type=int, default=3)
    ap.add_argument("--max_new_tokens", type=int, default=4096)
    ap.add_argument("--consolidate_chunk", type=int, default=CONSOLIDATE_CHUNK)
    a = ap.parse_args(argv)
    spec = DatasetSpec(a.dataset)
    survey = load_survey(a.in_dir)
    print(f"{spec.loader}: {len(survey)} survey images", flush=True)
    from stage2_common import load_vlm, make_text_generator
    proc, mdl = load_vlm()
    gen = make_text_generator(proc, mdl, a.max_new_tokens)
    run_all(gen, survey, a.out_dir, a.batch_images, a.shuffles, spec.loader,
            log=lambda m: print(m, flush=True), consolidate_chunk=a.consolidate_chunk)
    return 0


if __name__ == "__main__":
    sys.exit(main())
