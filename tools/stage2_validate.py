"""Stage-2 step 5: pilot / full caption validation (CPU; CLIP text encoder on GPU when available).

Inputs: caption shards (step 4), survey shards (step 1), attributes.json (step 3) and the reference
caption jsonl (V4 for Flickr25k / NUS-WIDE, V5b for MS-COCO), compared on the SAME image ids.

Gates (design record 2026-10-07):  parse-or-unmatched-key <= 1 %;  none-like/negation/hedge <= 1 %;
attribute-name prefix <= 1 %;  cross-attribute top-100 content-word overlap <= reference;  within-image
mean cosine of the 4 attribute fields <= reference + .03;  every per-attribute cross-image cosine < .75.
Everything else (lengths, duplicates, type/token, unseen-noun rate, per-field rates) is reported only.

Usage:
    python tools/stage2_validate.py --dataset Flickr25k --captions cache_eval/stage2/flickr25k/captions_pilot.shard*.jsonl \
        --survey cache_eval/stage2/flickr25k/survey*.shard*.jsonl --attributes cache_eval/stage2/flickr25k/attributes.json \
        --out_json .../pilot_validation.json --out_md .../pilot_validation.md [--exit_on_fail]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage2_common import LEGACY_KEYS, LEGACY_LOCAL_KEYS, WORKTREE, DatasetSpec, failed_rows, read_rows  # noqa: E402

REFERENCE = {
    "flickr25k": "/data/yschoi/dataset/deephashing/cache/flickr25k_qwen3_v4_trainset.jsonl",
    "nuswide": "/data/yschoi/dataset/deephashing/cache/nuswide_qwen3_v4_trainset.jsonl",
    "mscoco": "/data/yschoi/dataset/deephashing/cache/mscoco_qwen3_v5b_trainset.jsonl",
}
A3_V2 = os.path.join(WORKTREE, "result", "analysis", "textdiag_2026-10", "a3_v2.py")
NONE_LIKE = re.compile(r"(^\s*none\s*$|\bnone\b|not visible|no visible|\bwithout\b|\bcannot\b|\bunclear\b|"
                       r"\bn/a\b|\bbarely\b|\bfaint\b|hard to see|\babsent\b)", re.IGNORECASE)
WORD = re.compile(r"[a-z][a-z\-']+")
TOP_K = 100
GATES = {"parse_rate": 0.01, "none_like_rate": 0.01, "name_prefix_rate": 0.01, "within_image_cos_margin": 0.03,
         "cross_image_cos_max": 0.75}


def _load_content() -> Callable[[str], List[str]]:
    """``content`` from result/analysis/textdiag_2026-10/a3_v2.py (rule-based lemma + stop list)."""
    spec = importlib.util.spec_from_file_location("a3_v2_stage2", A3_V2)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.content


content = _load_content()


# --------------------------------------------------------------------------- loading
def load_captions(paths: Sequence[str], attributes: dict):
    """-> ok rows {id: row} (schema-checked), parse-failed ids, unmatched-key ids."""
    rows = read_rows(paths)
    failed = set(failed_rows(paths))
    attr_keys = [a["key"] for a in attributes["attributes"]]
    ok, unmatched = {}, []
    for iid, r in rows.items():
        cb, v10 = r.get("codebook_texts") or {}, r.get("codebook_texts_v10") or {}
        good = (set(cb) == set(LEGACY_KEYS) and cb.get("C_scene_type", None) == ""
                and all(str(cb.get(k, "")).strip() for k in LEGACY_KEYS[:5])
                and set(v10) == set(attr_keys) | {"C_global"} and all(str(v10[k]).strip() for k in v10))
        (ok.__setitem__(iid, r) if good else unmatched.append(iid))
    return ok, sorted(failed - set(rows)), sorted(unmatched)


def load_reference(path: str, ids: Sequence[str]) -> Dict[str, dict]:
    want = set(ids)
    out = {}
    for iid, r in read_rows([path]).items():
        if iid in want:
            cb = r.get("codebook_texts") or {}
            if all(str(cb.get(k, "")).strip() for k in LEGACY_KEYS[:5]):
                out[iid] = {k: " ".join(str(cb[k]).split()) for k in LEGACY_KEYS[:5]}
    return out


def load_survey(paths: Sequence[str]) -> Dict[str, List[str]]:
    return {iid: r["phrases"] for iid, r in read_rows(paths).items() if r.get("phrases")}


# --------------------------------------------------------------------------- text metrics
def _prefix_regex(name: str, key: str) -> re.Pattern:
    alts = []
    for n in {name, key}:
        words = [w for w in re.split(r"[\s_]+", n.strip()) if w]
        if words:
            alts.append(r"[\s_]+".join(re.escape(w) for w in words))
    return re.compile(r"^\s*(?:" + "|".join(alts) + r")\b\s*[:\-–]?", re.IGNORECASE)


def _q(x: Sequence[float], p: float) -> float:
    return float(np.percentile(np.asarray(x, dtype=float), p)) if len(x) else float("nan")


def field_stats(fields: Dict[str, List[str]], attributes: dict) -> dict:
    """fields: {field name: [text per row]} in row order; the first 4 names are the attribute keys.

    Gate rates are per IMAGE (share of rows with at least one offending field: ``none_like_rate`` over
    the 5 fields, ``name_prefix_rate`` over the 4 attribute fields); per-field rates are reported too.
    """
    out = {}
    attrs = {a["key"]: a for a in attributes["attributes"]}
    n_rows = len(next(iter(fields.values()))) if fields else 0
    row_none = np.zeros(n_rows, dtype=bool)
    row_prefix = np.zeros(n_rows, dtype=bool)
    for name, texts in fields.items():
        lens = [len(t.split()) for t in texts]
        none_mask = np.array([bool(NONE_LIKE.search(t)) for t in texts], dtype=bool)
        none_hits = int(none_mask.sum())
        row_none |= none_mask
        st = {"n": len(texts), "none_like_rate": none_hits / max(len(texts), 1),
              "words_mean": float(np.mean(lens)) if lens else float("nan"),
              "words_p10": _q(lens, 10), "words_p50": _q(lens, 50), "words_p90": _q(lens, 90),
              "duplicate_rate": 1.0 - len(set(t.lower().strip() for t in texts)) / max(len(texts), 1)}
        toks = [w for t in texts for w in WORD.findall(t.lower())]
        st["type_token_ratio"] = len(set(toks)) / max(len(toks), 1)
        if name in attrs:
            rx = _prefix_regex(attrs[name]["name"], name)
            pre_mask = np.array([bool(rx.search(t)) for t in texts], dtype=bool)
            st["name_prefix_rate"] = float(pre_mask.mean()) if len(texts) else 0.0
            row_prefix |= pre_mask
        out[name] = st
    return {"per_field": out, "none_like_rate": float(row_none.mean()) if n_rows else 0.0,
            "name_prefix_rate": float(row_prefix.mean()) if n_rows else 0.0, "rate_unit": "images"}


def top_words(texts: Sequence[str], k: int = TOP_K) -> List[str]:
    c = Counter(w for t in texts for w in content(t))
    return [w for w, _ in c.most_common(k)]


def cross_attribute_overlap(fields: Dict[str, List[str]], names: Sequence[str], k: int = TOP_K) -> dict:
    tops = {n: set(top_words(fields[n], k)) for n in names}
    pair = {f"{a}|{b}": len(tops[a] & tops[b]) for a, b in combinations(names, 2)}
    return {"top_k": k, "pairwise_shared": pair, "mean_pairwise_shared": float(np.mean(list(pair.values()))),
            "mean_pairwise_shared_frac": float(np.mean(list(pair.values()))) / k}


def unseen_word_rate(rows: Dict[str, dict], survey: Dict[str, List[str]], names: Sequence[str]) -> dict:
    per = {n: [0, 0] for n in names}
    n_img = 0
    for iid, r in rows.items():
        if iid not in survey:
            continue
        n_img += 1
        seen = set(w for ph in survey[iid] for w in content(ph))
        for n in names:
            ws = content(r["codebook_texts_v10"][n])
            per[n][0] += sum(1 for w in ws if w not in seen)
            per[n][1] += len(ws)
    tot_u, tot_n = sum(v[0] for v in per.values()), sum(v[1] for v in per.values())
    return {"n_images_with_survey": n_img, "per_attribute": {n: v[0] / max(v[1], 1) for n, v in per.items()},
            "overall": tot_u / max(tot_n, 1)}


# --------------------------------------------------------------------------- CLIP cosines
def clip_text_encoder(device: Optional[str] = None, max_length: int = 64) -> Callable[[Sequence[str]], np.ndarray]:
    """Raw CLIP EOS-projection text features (L2-normalised), same model / tokenizer settings as
    extract_clip_text_features.py (openai/clip-vit-base-patch16, max_length 64, get_text_features)."""
    import torch
    try:
        from models.pretrained_backbone_clip import DEFAULT_CLIP_BACKBONE as name
    except Exception:  # noqa: BLE001
        name = "openai/clip-vit-base-patch16"
    try:
        from models.pretrained_backbone import coerce_pooled_to_tensor      # same coercion as the extractor
    except Exception:  # noqa: BLE001
        def coerce_pooled_to_tensor(x):
            if torch.is_tensor(x):
                return x
            for attr in ("pooler_output", "text_embeds"):
                if getattr(x, attr, None) is not None:
                    return getattr(x, attr)
            raise TypeError(f"cannot extract pooled text features from {type(x).__name__}")
    from transformers import CLIPModel, CLIPTokenizer
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(name).to(device).eval()
    tok = CLIPTokenizer.from_pretrained(name)

    def enc(texts: Sequence[str]) -> np.ndarray:
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), 256):
                e = tok(list(texts[i:i + 256]), padding="max_length", truncation=True, max_length=max_length,
                        return_tensors="pt").to(device)
                f = coerce_pooled_to_tensor(model.get_text_features(input_ids=e["input_ids"],
                                                                    attention_mask=e["attention_mask"]))
                out.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy())
        return np.concatenate(out, 0) if out else np.zeros((0, 512), dtype=np.float32)

    return enc


def cosine_stats(fields: Dict[str, List[str]], names: Sequence[str], enc, n_pairs: int, seed: int) -> dict:
    """within-image mean pairwise cosine over the 4 attribute fields; per-attribute cross-image cosine
    over n_pairs random pairs of distinct images."""
    n = len(fields[names[0]])
    if n < 2:
        return {"n": n, "within_image_mean": float("nan"), "cross_image": {k: float("nan") for k in names}}
    mats = [np.asarray(enc(fields[k])) for k in names]
    if any(m.ndim != 2 or m.shape[0] != n for m in mats):
        raise ValueError(f"encoder must return [N, D] per field; got {[m.shape for m in mats]}")
    V = np.stack(mats, 1)                                                   # [N, 4, D]
    V = V / (np.linalg.norm(V, axis=-1, keepdims=True) + 1e-8)
    G = np.einsum("nad,nbd->nab", V, V)
    iu = np.triu_indices(len(names), 1)
    within = float(G[:, iu[0], iu[1]].mean())
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, size=n_pairs)
    j = (i + rng.integers(1, n, size=n_pairs)) % n
    cross = {k: float((V[i, a] * V[j, a]).sum(-1).mean()) for a, k in enumerate(names)}
    return {"n": n, "within_image_mean": within, "cross_image": cross, "n_pairs": int(n_pairs)}


# --------------------------------------------------------------------------- driver
def compute_metrics(ok: Dict[str, dict], n_failed: int, n_unmatched: int, ref: Dict[str, dict],
                    survey: Dict[str, List[str]], attributes: dict, enc=None, n_pairs: int = 2000,
                    seed: int = 0) -> dict:
    attr_keys = [a["key"] for a in attributes["attributes"]]
    n_ids = len(ok) + n_failed + n_unmatched
    m = {"n_ids": n_ids, "n_ok": len(ok), "n_parse_failed": n_failed, "n_unmatched_keys": n_unmatched,
         "parse_rate": (n_failed + n_unmatched) / max(n_ids, 1), "attribute_keys": attr_keys,
         "key_map": attributes["key_map"]}
    ids = sorted(ok)
    fields = {k: [ok[i]["codebook_texts_v10"][k] for i in ids] for k in attr_keys + ["C_global"]}
    m["new"] = field_stats(fields, attributes)
    m["new"]["overlap"] = cross_attribute_overlap(fields, attr_keys)
    m["new"]["unseen_word"] = unseen_word_rate(ok, survey, attr_keys)
    m["none_like_rate"], m["name_prefix_rate"] = m["new"]["none_like_rate"], m["new"]["name_prefix_rate"]
    common = [i for i in ids if i in ref]
    m["n_common_with_reference"] = len(common)
    ref_fields = {k: [ref[i][k] for i in common] for k in LEGACY_KEYS[:5]}
    new_common = {k: [ok[i]["codebook_texts_v10"][k] for i in common] for k in attr_keys}
    if common:
        m["reference"] = {"per_field": field_stats(ref_fields, {"attributes": []})["per_field"],
                          "overlap": cross_attribute_overlap(ref_fields, list(LEGACY_LOCAL_KEYS))}
        m["new_on_common"] = {"overlap": cross_attribute_overlap(new_common, attr_keys)}
    if enc is not None and common:
        m["new_on_common"]["cosine"] = cosine_stats(new_common, attr_keys, enc, n_pairs, seed)
        m["reference"]["cosine"] = cosine_stats(ref_fields, list(LEGACY_LOCAL_KEYS), enc, n_pairs, seed)
    m["gates"] = evaluate_gates(m)
    m["all_pass"] = all(g["pass"] for g in m["gates"].values())
    return m


def evaluate_gates(m: dict) -> dict:
    g = {}
    g["parse"] = {"value": m["parse_rate"], "threshold": GATES["parse_rate"], "pass": m["parse_rate"] <= GATES["parse_rate"]}
    g["none_like"] = {"value": m["none_like_rate"], "threshold": GATES["none_like_rate"],
                      "pass": m["none_like_rate"] <= GATES["none_like_rate"]}
    g["name_prefix"] = {"value": m["name_prefix_rate"], "threshold": GATES["name_prefix_rate"],
                        "pass": m["name_prefix_rate"] <= GATES["name_prefix_rate"]}
    ref, newc = m.get("reference"), m.get("new_on_common")
    if ref and newc:
        v, t = newc["overlap"]["mean_pairwise_shared"], ref["overlap"]["mean_pairwise_shared"]
        g["overlap_le_reference"] = {"value": v, "threshold": t, "pass": v <= t}
    else:
        g["overlap_le_reference"] = {"value": None, "threshold": None, "pass": False, "reason": "no reference rows"}
    if ref and newc and "cosine" in newc:
        v, t = newc["cosine"]["within_image_mean"], ref["cosine"]["within_image_mean"] + GATES["within_image_cos_margin"]
        g["within_image_cos"] = {"value": v, "threshold": t, "pass": v <= t}
        worst = max(newc["cosine"]["cross_image"].values())
        g["cross_image_cos"] = {"value": worst, "threshold": GATES["cross_image_cos_max"],
                                "pass": worst < GATES["cross_image_cos_max"], "per_attribute": newc["cosine"]["cross_image"]}
    else:
        for k in ("within_image_cos", "cross_image_cos"):
            g[k] = {"value": None, "threshold": None, "pass": False, "reason": "no cosine computed"}
    return g


def render_md(m: dict, title: str) -> str:
    L = [f"# {title}", "", f"ids {m['n_ids']}, ok {m['n_ok']}, parse-failed {m['n_parse_failed']}, unmatched keys "
         f"{m['n_unmatched_keys']}, common with reference {m.get('n_common_with_reference', 0)}", "",
         "| gate | value | threshold | result |", "|---|---|---|---|"]
    for k, g in m["gates"].items():
        v = "n/a" if g["value"] is None else f"{g['value']:.4f}"
        t = "n/a" if g["threshold"] is None else f"{g['threshold']:.4f}"
        L.append(f"| {k} | {v} | {t} | {'PASS' if g['pass'] else 'FAIL'}{' (' + g['reason'] + ')' if g.get('reason') else ''} |")
    L += ["", f"**overall: {'PASS' if m['all_pass'] else 'FAIL'}**", "",
          "| field | none-like | name-prefix | words mean/p10/p50/p90 | dup | TTR | unseen-word |", "|---|---|---|---|---|---|---|"]
    uw = m["new"]["unseen_word"]["per_attribute"]
    for f, s in m["new"]["per_field"].items():
        L.append(f"| {f} | {s['none_like_rate']:.4f} | {s.get('name_prefix_rate', float('nan')):.4f} | "
                 f"{s['words_mean']:.1f}/{s['words_p10']:.0f}/{s['words_p50']:.0f}/{s['words_p90']:.0f} | "
                 f"{s['duplicate_rate']:.4f} | {s['type_token_ratio']:.3f} | {uw.get(f, float('nan')):.3f} |")
    L.append("")
    L.append(f"cross-attribute top-{TOP_K} shared words, mean pairwise: new {m['new']['overlap']['mean_pairwise_shared']:.1f}"
             + (f" (on common ids {m['new_on_common']['overlap']['mean_pairwise_shared']:.1f}); reference "
                f"{m['reference']['overlap']['mean_pairwise_shared']:.1f}" if m.get("reference") else ""))
    if m.get("reference") and "cosine" in m.get("new_on_common", {}):
        c, r = m["new_on_common"]["cosine"], m["reference"]["cosine"]
        L.append(f"CLIP within-image cosine: new {c['within_image_mean']:.4f}, reference {r['within_image_mean']:.4f}; "
                 f"cross-image per attribute new {json.dumps({k: round(v, 4) for k, v in c['cross_image'].items()})}, "
                 f"reference {json.dumps({k: round(v, 4) for k, v in r['cross_image'].items()})}")
    L.append(f"unseen-word rate overall {m['new']['unseen_word']['overall']:.3f} "
             f"(images with survey: {m['new']['unseen_word']['n_images_with_survey']})")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--captions", nargs="+", required=True)
    ap.add_argument("--survey", nargs="*", default=[])
    ap.add_argument("--attributes", required=True)
    ap.add_argument("--reference", default=None)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_md", required=True)
    ap.add_argument("--n_pairs", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no_clip", action="store_true", help="skip the CLIP cosines (the two cosine gates then FAIL)")
    ap.add_argument("--exit_on_fail", action="store_true", help="exit 3 when any gate fails")
    ap.add_argument("--title", default=None)
    a = ap.parse_args(argv)
    spec = DatasetSpec(a.dataset)
    attributes = json.load(open(a.attributes))
    ok, failed, unmatched = load_captions(a.captions, attributes)
    ref = load_reference(a.reference or REFERENCE[spec.short], list(ok))
    survey = load_survey([p for p in a.survey if Path(p).exists()])
    enc = None if a.no_clip else clip_text_encoder(a.device)
    m = compute_metrics(ok, len(failed), len(unmatched), ref, survey, attributes, enc, a.n_pairs, a.seed)
    m.update(dataset=spec.loader, captions=list(a.captions), reference=m.get("reference"),
             reference_path=a.reference or REFERENCE[spec.short], failed_ids=failed[:50], unmatched_ids=unmatched[:50])
    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(m, open(a.out_json, "w"), indent=1)
    md = render_md(m, a.title or f"Stage-2 validation {spec.loader}")
    open(a.out_md, "w").write(md)
    print(md)
    for k, g in m["gates"].items():
        print(f"GATE {k}: {'PASS' if g['pass'] else 'FAIL'} value={g['value']} threshold={g['threshold']}")
    print(f"OVERALL: {'PASS' if m['all_pass'] else 'FAIL'}")
    return 3 if (a.exit_on_fail and not m["all_pass"]) else 0


if __name__ == "__main__":
    sys.exit(main())
