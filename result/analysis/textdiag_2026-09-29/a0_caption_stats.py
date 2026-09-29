"""A0: caption statistics of the four approved caption files (CPU, no model).

Per dataset and per used axis (the first five fields; C_scene_type is not used by the 5-slot model):
- vocabulary: content-word types, type/token ratio, coverage of caption tokens by the top-50 content words
  (how repeatable the axis vocabulary is across images);
- cross-axis leak: shared words among each pair of axes' top-100 content words;
- local<->local text cosine: mean pairwise cosine among the four local axes' CLIP text embeddings
  (text_part cache, rows matched by image_id);
- exact duplicate sentences (per axis) and the 'none' rate;
- second-generation agreement where a second caption file exists for the same images
  (MS-COCO v4 vs v5b, Flickr25K v4 vs v5b): content-word Jaccard per axis.
Reads JSON/NPY only. No training input is modified.
"""
import json, os, re, sys, collections, itertools
import numpy as np

CACHE = "/data/yschoi/groundeddna_cache_v6prov"
Q = "/data/yschoi/dataset/deephashing/cache"
FILES = {
    "cifar10":   (f"{Q}/cifar10_qwen_v4.jsonl",              f"{CACHE}/cifar10_clip_tokens",   None),
    "flickr25k": (f"{Q}/flickr25k_qwen3_v4_trainset.jsonl", f"{CACHE}/flickr25k_clip_tokens", f"{Q}/flickr25k_qwen3_v5b_trainset.jsonl"),
    "nuswide":   (f"{Q}/nuswide_qwen3_v4_trainset.jsonl",   f"{CACHE}/nuswide_clip_tokens",   None),
    "mscoco":    (f"{Q}/mscoco_qwen3_v5b_trainset.jsonl",   f"{CACHE}/mscoco_clip_tokens",    f"{Q}/mscoco_qwen3_v4_trainset.jsonl"),
}
AXES = ["C_global", "C_primary_object", "C_secondary_object", "C_activity_or_relation", "C_color_texture"]
LOCAL = AXES[1:]
STOP = set("""a an the and or of to in on at with by for from as is are was were be been being this that these those it its
their his her they them he she we you i our your into onto over under above below near beside between behind front
against along across through around while during before after up down out off no not none very more most some
any each other another such than then there here where when which who whom what how all both few many several
also just only own same so too can could may might will would shall should do does did done has have had having
one two three four five six seven eight nine ten first second third left right top bottom center centre middle
side sides background foreground image scene photo picture shot view visible shown seen appears appear appearing
positioned located situated set placed standing sitting lying stands sits lies""".split())
WORD = re.compile(r"[a-z][a-z\-']+")

def load(path):
    rows = {}
    for line in open(path):
        r = json.loads(line)
        d = r.get("descriptions") or r.get("codebook_texts") or {}
        rows[r["image_id"]] = {k: str(d.get(k, "")) for k in AXES}
    return rows

def words(s):
    return [w for w in WORD.findall(s.lower()) if w not in STOP and len(w) > 2]

def axis_stats(rows):
    out = {}
    tops = {}
    for ax in AXES:
        toks = [words(rows[i][ax]) for i in rows]
        flat = [w for t in toks for w in t]
        c = collections.Counter(flat)
        top50 = set(w for w, _ in c.most_common(50))
        top100 = [w for w, _ in c.most_common(100)]
        tops[ax] = top100
        sents = [rows[i][ax].strip().lower() for i in rows]
        none = sum(1 for s in sents if s.rstrip(".") in ("none", "not visible", "", "n/a"))
        dup = 1 - len(set(sents)) / len(sents)
        out[ax] = {
            "content_types": len(c), "content_tokens": len(flat),
            "type_token_ratio": round(len(c) / max(len(flat), 1), 4),
            "top50_coverage_of_tokens": round(sum(c[w] for w in top50) / max(len(flat), 1), 4),
            "images_with_a_top50_word": round(np.mean([any(w in top50 for w in t) for t in toks]), 4),
            "exact_duplicate_sentence_rate": round(dup, 4), "none_rate": round(none / len(sents), 4),
            "top20": [w for w, _ in c.most_common(20)],
        }
    leak = {}
    for a, b in itertools.combinations(LOCAL, 2):
        leak[f"{a}|{b}"] = len(set(tops[a]) & set(tops[b]))
    return out, leak

def text_cosine(rows, cache_dir):
    ids = json.load(open(os.path.join(cache_dir, "image_ids.json")))
    row_of = {img: i for i, img in enumerate(ids)}
    T = np.load(os.path.join(cache_dir, "text_part.f16.npy"), mmap_mode="r")
    sel = [row_of[i] for i in rows if i in row_of]
    if not sel:
        return {"matched_rows": 0}
    X = np.asarray(T[np.sort(sel)][:, :5, :], dtype=np.float32)          # [N, 5, 512]
    X /= np.linalg.norm(X, axis=-1, keepdims=True) + 1e-8
    G = np.einsum("nmd,nad->nma", X, X)                                   # per-image axis x axis cosine
    loc = G[:, 1:, 1:]
    M = loc.shape[1]
    off = (loc.sum(axis=(1, 2)) - np.trace(loc, axis1=1, axis2=2)) / (M * M - M)
    glob = G[:, 0, 1:].mean(axis=1)
    pair = {}
    for a in range(1, 5):
        for b in range(a + 1, 5):
            pair[f"{AXES[a]}|{AXES[b]}"] = round(float(G[:, a, b].mean()), 4)
    return {"matched_rows": len(sel), "local_local_mean_cos": round(float(off.mean()), 4),
            "global_local_mean_cos": round(float(glob.mean()), 4), "pairwise": pair}

def agreement(rows_a, rows_b):
    common = [i for i in rows_a if i in rows_b]
    out = {"common_images": len(common)}
    for ax in AXES:
        js = []
        for i in common:
            wa, wb = set(words(rows_a[i][ax])), set(words(rows_b[i][ax]))
            if wa or wb:
                js.append(len(wa & wb) / len(wa | wb))
        out[ax] = {"content_word_jaccard_mean": round(float(np.mean(js)), 4),
                   "share_with_jaccard_ge_0.25": round(float(np.mean([j >= 0.25 for j in js])), 4)}
    return out

def main(out_path):
    report = {}
    for ds, (path, cache_dir, second) in FILES.items():
        rows = load(path)
        stats, leak = axis_stats(rows)
        rep = {"file": path, "rows": len(rows), "per_axis": stats, "top100_shared_words_local_pairs": leak,
               "text_cosine": text_cosine(rows, cache_dir)}
        if second and os.path.exists(second):
            rep["second_generation"] = {"file": second, **agreement(rows, load(second))}
        report[ds] = rep
        print(f"== {ds} rows={len(rows)}  local<->local cos={rep['text_cosine'].get('local_local_mean_cos')}  "
              f"leak(top100 shared)={leak}")
        for ax in AXES:
            s = stats[ax]
            print(f"   {ax:24s} types={s['content_types']:5d} TTR={s['type_token_ratio']:.3f} "
                  f"top50cov={s['top50_coverage_of_tokens']:.3f} dup={s['exact_duplicate_sentence_rate']:.4f} none={s['none_rate']:.3f}")
        if "second_generation" in rep:
            sg = rep["second_generation"]
            print("   v4<->v5b jaccard:", {ax: sg[ax]["content_word_jaccard_mean"] for ax in AXES})
    json.dump(report, open(out_path, "w"), indent=1)

if __name__ == "__main__":
    main(sys.argv[1])
