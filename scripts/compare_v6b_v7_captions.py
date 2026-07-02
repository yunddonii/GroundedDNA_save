"""Compare v6b vs v7 CUB captions on hedging + forbidden vocabulary.
Uses the same 24-image indices for apples-to-apples comparison.
"""
from __future__ import annotations
import json, os, re
from collections import Counter, defaultdict

V6B_FILE = "cache/cub200_qwen_v6b_trainset.jsonl"
V7_FILE = "cache/cub200_qwen_v7_sample24.jsonl"

HEDGING_PATTERNS = [
    r"\bno distinct\b", r"\bno clear(?:ly)?\b", r"\bno visible\b",
    r"\bnot (?:clearly )?visible\b", r"\bnot discern\w+\b",
    r"\bnot determin\w+\b", r"\bnot present\b", r"\bnot shown\b",
    r"\bcannot be (?:seen|determined|identified)\b", r"\bunclear\b",
    r"\bindiscernible\b", r"\bindeterminate\b", r"\bappears? to be\b",
    r"\bnone (?:visible|shown|apparent)\b",
    r"\bno (?:visible|apparent) (?:markings|features|patterns|bands|structures)\b",
    r"\bwith no visible\b",
]

FORBIDDEN_WORDS = [
    "bird", "birds", "avian", "animal", "creature",
    "specimen", "subject", "individual", "entity",
    "feathered", "plumage",
]

V6B_SLOTS = [
    "C_global", "C_head_bill", "C_upperparts_wing",
    "C_underparts", "C_tail_appendages", "C_pattern_markings",
]
V7_SLOTS = [
    "C_global", "C_head_face_eye", "C_bill",
    "C_wing_back", "C_underparts", "C_tail_legs",
]


def load(path, cb_key):
    caps = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            r = json.loads(line)
            if r.get(cb_key):
                caps.append((r["image_id"], r[cb_key]))
    return caps


def count_hedging(text: str) -> int:
    t = text.lower()
    n = 0
    hits = []
    for pat in HEDGING_PATTERNS:
        matches = re.findall(pat, t)
        if matches:
            n += len(matches)
            hits.extend(matches)
    return n, hits


def count_forbidden(text: str) -> list:
    t = re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()
    hits = [w for w in FORBIDDEN_WORDS if w in t]
    return hits


def report(name, caps, slots):
    print(f"\n{'='*70}\n{name}  (N={len(caps)} captions, slots={len(slots)})\n{'='*70}")
    hedge_per_slot = defaultdict(int)
    hedge_examples = defaultdict(list)
    forb_per_slot = defaultdict(int)
    forb_examples = defaultdict(list)
    lengths = defaultdict(list)
    for iid, cb in caps:
        for slot in slots:
            text = cb.get(slot, "") or ""
            n_h, hits = count_hedging(text)
            if n_h > 0:
                hedge_per_slot[slot] += 1
                if len(hedge_examples[slot]) < 3:
                    hedge_examples[slot].append((iid, hits, text))
            forb = count_forbidden(text)
            if forb:
                forb_per_slot[slot] += len(forb)
                if len(forb_examples[slot]) < 3:
                    forb_examples[slot].append((iid, forb, text))
            lengths[slot].append(len(text.split()))
    print("\n  HEDGING per slot:")
    for slot in slots:
        n = hedge_per_slot[slot]
        pct = 100.0 * n / len(caps) if len(caps) else 0.0
        print(f"    {slot:<22s}  {n:>3d}/{len(caps)}  ({pct:>5.1f}%)")
        for iid, hits, text in hedge_examples[slot][:2]:
            print(f"      hits={hits!r}  → \"{text[:80]}...\"")
    print("\n  FORBIDDEN word occurrences (total across images) per slot:")
    for slot in slots:
        n = forb_per_slot[slot]
        print(f"    {slot:<22s}  {n:>3d} occurrences")
        for iid, forb, text in forb_examples[slot][:2]:
            print(f"      {forb} → \"{text[:80]}...\"")
    print("\n  Length (words) per slot:")
    for slot in slots:
        L = lengths[slot]
        m = sum(L)/max(len(L),1)
        print(f"    {slot:<22s}  mean={m:.1f}  min={min(L)}  max={max(L)}")


def main():
    v6b = load(V6B_FILE, "codebook_texts_v6b")
    v7 = load(V7_FILE, "codebook_texts_v7")
    print(f"v6b caps: {len(v6b)}")
    print(f"v7 caps: {len(v7)}")

    # Match by image_id
    v6b_by_id = dict(v6b)
    v7_ids = {iid for iid, _ in v7}
    matched = [(iid, v6b_by_id[iid]) for iid, _ in v7 if iid in v6b_by_id]
    print(f"Matched v6b for v7 images: {len(matched)}")

    report("V6B (matched subset)", matched, V6B_SLOTS)
    report("V7 (sample 24)", v7, V7_SLOTS)


if __name__ == "__main__":
    main()
