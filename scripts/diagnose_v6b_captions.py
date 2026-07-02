"""Diagnose v6b CUB captions: hedging language, forbidden vocabulary,
slot vocabulary overlap (Jaccard), and length distribution.

Motivation: v179a (2026-07-01) qualitative viz shows local slots don't
cleanly separate anatomy. User observed "No distinct ~" style hedging
that suggests caption quality issues.
"""
from __future__ import annotations
import json, os, re
from collections import Counter, defaultdict
from pathlib import Path

CAPTION_FILE = "cache/cub200_qwen_v6b_trainset.jsonl"

# What we look for
HEDGING_PATTERNS = [
    r"\bno distinct\b",
    r"\bno clear(?:ly)?\b",
    r"\bnot (?:clearly )?visible\b",
    r"\bnot visible\b",
    r"\bnot (?:clearly )?discern\w+\b",
    r"\bnot discern\w+\b",
    r"\bnot determin\w+\b",
    r"\bnot present\b",
    r"\bnot shown\b",
    r"\bcannot be (?:seen|determined|identified)\b",
    r"\bunclear\b",
    r"\bindiscernible\b",
    r"\bindeterminate\b",
    r"\bappears? to be\b",
    r"\bnone (?:visible|shown|apparent)\b",
    r"\bno (?:visible|apparent) (?:markings|features|patterns)\b",
]

FORBIDDEN_WORDS = [
    "bird", "birds", "avian", "avifauna", "animal", "animals",
    "creature", "specimen", "subject", "individual", "entity",
    "feathered", "plumage",
]

SLOTS = [
    "C_global",
    "C_head_bill",
    "C_upperparts_wing",
    "C_underparts",
    "C_tail_appendages",
    "C_pattern_markings",
]


def tokenize(text: str) -> list:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    return [t for t in text.split() if t]


def load_captions(path: str) -> list:
    caps = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                r = json.loads(line)
                caps.append(r)
            except json.JSONDecodeError:
                pass
    return caps


def report_hedging(records: list):
    print("\n" + "=" * 70)
    print("HEDGING LANGUAGE FREQUENCY (per slot)")
    print("=" * 70)
    N = len(records)
    per_slot = defaultdict(int)
    pattern_hits = defaultdict(Counter)
    for r in records:
        cb = r.get("codebook_texts_v6b", {})
        for slot in SLOTS:
            text = (cb.get(slot) or "").lower()
            for pat in HEDGING_PATTERNS:
                hits = re.findall(pat, text)
                if hits:
                    per_slot[slot] += 1
                    for h in hits:
                        pattern_hits[slot][h] += 1
                    break
    print(f"\n  N={N} captions total")
    print(f"  {'Slot':<24s} | {'Hedged':>8s} | {'%':>6s}")
    print(f"  {'-'*24} | {'-'*8} | {'-'*6}")
    for slot in SLOTS:
        n = per_slot[slot]
        print(f"  {slot:<24s} | {n:>8d} | {100*n/N:>5.2f}%")
    # Top patterns per slot
    print("\n  Top hedging patterns per slot:")
    for slot in SLOTS:
        if not pattern_hits[slot]: continue
        top = pattern_hits[slot].most_common(3)
        top_str = ", ".join(f'"{h}"({c})' for h, c in top)
        print(f"    {slot}: {top_str}")


def report_forbidden(records: list):
    print("\n" + "=" * 70)
    print("FORBIDDEN WORD OCCURRENCE (per slot)")
    print("=" * 70)
    N = len(records)
    per_slot = defaultdict(Counter)
    for r in records:
        cb = r.get("codebook_texts_v6b", {})
        for slot in SLOTS:
            text = (cb.get(slot) or "").lower()
            tokens = tokenize(text)
            for w in FORBIDDEN_WORDS:
                if w in tokens:
                    per_slot[slot][w] += 1
    print(f"\n  N={N}")
    print(f"  {'Slot':<24s} | " + " | ".join(f"{w:>9s}" for w in FORBIDDEN_WORDS))
    print(f"  {'-'*24} | " + " | ".join(["-"*9]*len(FORBIDDEN_WORDS)))
    for slot in SLOTS:
        row = f"  {slot:<24s} | "
        row += " | ".join(f"{per_slot[slot][w]:>9d}" for w in FORBIDDEN_WORDS)
        print(row)


def report_length(records: list):
    print("\n" + "=" * 70)
    print("CAPTION LENGTH DISTRIBUTION (per slot, in words)")
    print("=" * 70)
    per_slot = defaultdict(list)
    for r in records:
        cb = r.get("codebook_texts_v6b", {})
        for slot in SLOTS:
            text = (cb.get(slot) or "")
            per_slot[slot].append(len(tokenize(text)))
    print(f"\n  {'Slot':<24s} | {'mean':>6s} | {'min':>4s} | {'max':>4s} | {'std':>5s}")
    print(f"  {'-'*24} | {'-'*6} | {'-'*4} | {'-'*4} | {'-'*5}")
    for slot in SLOTS:
        lens = per_slot[slot]
        import statistics
        mean = statistics.mean(lens)
        std = statistics.stdev(lens) if len(lens) > 1 else 0
        print(f"  {slot:<24s} | {mean:>6.2f} | {min(lens):>4d} | {max(lens):>4d} | {std:>5.2f}")


def report_slot_jaccard(records: list):
    print("\n" + "=" * 70)
    print("SLOT VOCABULARY OVERLAP (per-image Jaccard, averaged)")
    print("=" * 70)
    STOP = set("a an the this that is are was were be been being of to in on at by "
               "with for from and or but not no as it its their".split())
    print(f"  Stop words excluded ({len(STOP)} tokens).")
    slot_pairs = [(i, j) for i in range(len(SLOTS)) for j in range(i+1, len(SLOTS))]
    pair_jacc = defaultdict(list)
    for r in records:
        cb = r.get("codebook_texts_v6b", {})
        tokens_by_slot = {}
        for slot in SLOTS:
            text = (cb.get(slot) or "")
            toks = set(tokenize(text)) - STOP
            tokens_by_slot[slot] = toks
        for i, j in slot_pairs:
            a = tokens_by_slot[SLOTS[i]]
            b = tokens_by_slot[SLOTS[j]]
            if not a or not b:
                continue
            j_val = len(a & b) / len(a | b)
            pair_jacc[(SLOTS[i], SLOTS[j])].append(j_val)
    print(f"\n  Per-image Jaccard (mean over N):")
    print(f"  {'slot A':<24s} | {'slot B':<24s} | {'mean J':>8s} | {'max J':>7s}")
    print(f"  {'-'*24} | {'-'*24} | {'-'*8} | {'-'*7}")
    for pair, vals in sorted(pair_jacc.items(), key=lambda x: -sum(x[1])/max(len(x[1]),1)):
        m = sum(vals) / max(len(vals), 1)
        mx = max(vals) if vals else 0
        print(f"  {pair[0]:<24s} | {pair[1]:<24s} | {m:>8.4f} | {mx:>7.4f}")


def main():
    if not os.path.exists(CAPTION_FILE):
        print(f"[error] {CAPTION_FILE} not found")
        return
    print(f"[diag] loading {CAPTION_FILE}")
    records = load_captions(CAPTION_FILE)
    print(f"[diag] {len(records)} captions")
    report_hedging(records)
    report_forbidden(records)
    report_length(records)
    report_slot_jaccard(records)


if __name__ == "__main__":
    main()
