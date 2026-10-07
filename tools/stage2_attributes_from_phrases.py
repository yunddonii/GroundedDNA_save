"""Stage-2 alternative for steps 2+3 (2026-10-07): the 8B text-only model does not abstract concepts from
phrase lists (it copies them), so step 2 becomes a NON-VLM exact-match de-duplication of the survey
phrases and step 3 (PROMPT_ATTRIBUTES, wording unchanged) is run directly on a random sample of
--sample phrases per shuffle seed (0..shuffles-1).  Groupings are compared as before; a merge pass runs
if they disagree.  Output: <out_dir>/attributes.json (same schema as stage2_concepts.attributes_step),
attributes_shuffle{s}.json, phrases_dedup.json (counts + top-30 for information only; never shown to the model).
Usage: stage2_attributes_from_phrases.py --dataset Flickr25k --in_dir <survey dir> --out_dir <dir> [--sample 300] [--shuffles 3]
"""
import argparse, collections, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from stage2_common import DatasetSpec, MODEL_ID  # noqa: E402
from stage2_prompts import PROMPT_ATTRIBUTES, PROMPT_ATTRIBUTES_MERGE, sha256_of  # noqa: E402

# 2026-10-08: with 300 input phrases the model echoed every phrase into the "concepts" lists (and looped),
# overflowing 3072 tokens on all three datasets. Only the OUTPUT SIZE clause changes: at most 8 example
# concepts per attribute. Selection criteria are unchanged.
PROMPT_ATTRIBUTES_P = PROMPT_ATTRIBUTES.replace(
    "and the concepts that belong to it.", "and up to 8 of the listed concepts that belong to it.")
PROMPT_ATTRIBUTES_MERGE_P = PROMPT_ATTRIBUTES_MERGE.replace(
    '"concepts": ["..."]}], "note": ""}.', '"concepts": ["..."]}], "note": ""} with at most 8 concepts per attribute.')
assert PROMPT_ATTRIBUTES_P != PROMPT_ATTRIBUTES and PROMPT_ATTRIBUTES_MERGE_P != PROMPT_ATTRIBUTES_MERGE
from stage2_concepts import (load_survey, render_concepts, parse_attributes, groupings_agree,  # noqa: E402
                             validate_attributes, key_map_by_position, atomic_json_dump, LEGACY_LOCAL_KEYS, PROMPT_SHAS, PROMPT_VERSION)


def dedupe_phrases(survey):
    cnt = collections.Counter()
    for phrases in survey.values():
        for p in phrases:
            q = " ".join(str(p).lower().split())
            if q:
                cnt[q] += 1
    return cnt


def run(gen, survey, out_dir, sample=300, shuffles=3, log=print):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cnt = dedupe_phrases(survey)
    phrases = sorted(cnt)
    atomic_json_dump({"n_images": len(survey), "n_phrase_tokens": int(sum(cnt.values())), "n_unique": len(phrases),
                      "top30_for_information_only": cnt.most_common(30), "sample_per_shuffle": sample}, out_dir / "phrases_dedup.json")
    log(f"phrases: {sum(cnt.values())} total, {len(phrases)} unique; sampling {sample} per shuffle")
    groupings = []
    for s in range(shuffles):
        p = out_dir / f"attributes_shuffle{s}.json"
        rng = np.random.default_rng(s)
        idx = rng.choice(len(phrases), size=min(sample, len(phrases)), replace=False)
        sampled = [phrases[i] for i in idx]
        prompt = PROMPT_ATTRIBUTES_P.replace("<CONCEPTS>", render_concepts(sampled))
        g = parse_attributes(gen(prompt))
        assigned = set(c.lower().strip() for a in g["attributes"] for c in a.get("concepts", []))
        g.update(seed=s, prompt_sha256=sha256_of(PROMPT_ATTRIBUTES_P), input="phrases", n_input=len(sampled),
                 n_assigned=len(assigned & set(sampled)), input_sample=sampled)
        atomic_json_dump(g, p); groupings.append(g)
        log(f"  shuffle {s}: {[a['name'] for a in g['attributes']]} note={g.get('note','')!r} assigned {g['n_assigned']}/{len(sampled)}")
    agreement = groupings_agree(groupings)
    if agreement["agree"]:
        final, source = groupings[0], "shuffle0"
    else:
        prompt = PROMPT_ATTRIBUTES_MERGE_P
        for i, g in enumerate(groupings[:3], start=1):
            prompt = prompt.replace(f"<G{i}>", json.dumps({"attributes": g["attributes"], "note": g.get("note", "")}, ensure_ascii=False, indent=0))
        final, source = parse_attributes(gen(prompt)), "merge"
        log(f"  merged: {[a['name'] for a in final['attributes']]} note={final.get('note','')!r}")
    attrs = validate_attributes(final["attributes"])
    res = {"attributes": attrs, "note": final.get("note", ""), "key_map": key_map_by_position(attrs),
           "legacy_keys_in_order": list(LEGACY_LOCAL_KEYS), "source": source, "agreement": agreement,
           "method": "attributes_from_phrases", "n_unique_phrases": len(phrases), "sample_per_shuffle": sample,
           "prompt_sha256s": {"attributes_phrases": sha256_of(PROMPT_ATTRIBUTES_P), "attributes_merge_phrases": sha256_of(PROMPT_ATTRIBUTES_MERGE_P)},
           "prompt_text": {"attributes_phrases": PROMPT_ATTRIBUTES_P, "attributes_merge_phrases": PROMPT_ATTRIBUTES_MERGE_P},
           "prompt_version": PROMPT_VERSION, "vlm": MODEL_ID}
    atomic_json_dump(res, out_dir / "attributes.json")
    log("ATTRIBUTES " + json.dumps([(a["name"], a["definition"]) for a in attrs], ensure_ascii=False))
    log(f"agreement {json.dumps(agreement)}")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True); ap.add_argument("--in_dir", required=True); ap.add_argument("--out_dir", required=True)
    ap.add_argument("--sample", type=int, default=300); ap.add_argument("--shuffles", type=int, default=3)
    ap.add_argument("--max_new_tokens", type=int, default=1536)
    a = ap.parse_args(argv)
    spec = DatasetSpec(a.dataset)
    survey = load_survey(a.in_dir)
    try:
        survey.update(load_survey(a.in_dir, "survey_extra.shard*.jsonl"))
    except Exception:
        pass
    print(f"{spec.loader}: {len(survey)} survey images", flush=True)
    from stage2_common import load_vlm, make_text_generator
    proc, mdl = load_vlm(); gen = make_text_generator(proc, mdl, a.max_new_tokens)
    run(gen, survey, a.out_dir, a.sample, a.shuffles, log=lambda m: print(m, flush=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
