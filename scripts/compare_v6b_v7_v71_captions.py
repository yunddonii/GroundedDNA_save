"""Compare v6b, v7, v7.1 CUB captions on hedging + forbidden vocabulary."""
from __future__ import annotations
import json, os, re
from collections import Counter, defaultdict

FILES = {
    "v6b": ("cache/cub200_qwen_v6b_trainset.jsonl", "codebook_texts_v6b",
            ["C_global", "C_head_bill", "C_upperparts_wing",
             "C_underparts", "C_tail_appendages", "C_pattern_markings"]),
    "v7":  ("cache/cub200_qwen_v7_sample24.jsonl", "codebook_texts_v7",
            ["C_global", "C_head_face_eye", "C_bill",
             "C_wing_back", "C_underparts", "C_tail_legs"]),
    "v7.1":("cache/cub200_qwen_v7_1_sample24.jsonl", "codebook_texts_v7_1",
            ["C_global", "C_head_face_eye", "C_bill",
             "C_wing_back", "C_underparts", "C_tail_legs"]),
}

HEDGING_PATTERNS = [
    r"\bno distinct\b", r"\bno clear(?:ly)?\b", r"\bno visible\b",
    r"\bnot (?:clearly )?visible\b", r"\bnot discern\w+\b",
    r"\bnot determin\w+\b", r"\bcannot be (?:seen|determined|identified)\b",
    r"\bunclear\b", r"\bindiscernible\b", r"\bindeterminate\b",
    r"\bappears? to be\b", r"\bnone (?:visible|shown|apparent)\b",
    r"\bno (?:visible|apparent) (?:markings|features|patterns|bands|structures|streaks|bars|spots)\b",
    r"\bwith(?:out)? (?:no|any) visible\b",
    r"\bwith no\b",
]

FORBIDDEN_WORDS = [
    "bird", "birds", "avian", "animal", "creature",
    "specimen", "subject", "individual", "feathered", "plumage",
]


def count_hedging(text: str) -> int:
    t = text.lower()
    n = 0
    for pat in HEDGING_PATTERNS:
        n += len(re.findall(pat, t))
    return n


def count_forbidden(text: str) -> int:
    t = re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()
    return sum(1 for w in FORBIDDEN_WORDS if w in t)


def load(path, cb_key):
    caps = []
    with open(path) as f:
        for line in f:
            r = json.loads(line.strip())
            if r.get(cb_key): caps.append((r["image_id"], r[cb_key]))
    return caps


def slot_stats(caps, slots):
    hedge_captions = defaultdict(int)
    forb_captions = defaultdict(int)
    for iid, cb in caps:
        for slot in slots:
            text = cb.get(slot, "") or ""
            if count_hedging(text) > 0:
                hedge_captions[slot] += 1
            if count_forbidden(text) > 0:
                forb_captions[slot] += 1
    return hedge_captions, forb_captions


def main():
    # Get common image_ids across all three
    common = None
    for tag, (path, cb_key, _) in FILES.items():
        caps = load(path, cb_key)
        ids = {iid for iid, _ in caps}
        common = ids if common is None else common & ids
    print(f"Common image_ids across all 3 sources: {len(common)}")
    print()

    print(f"{'Metric':<35s} | " + " | ".join(f"{tag:>10s}" for tag in FILES))
    print("-" * (35 + 3 + sum(13 for _ in FILES)))

    results = {}
    for tag, (path, cb_key, slots) in FILES.items():
        caps = load(path, cb_key)
        # Filter to common
        caps = [(iid, cb) for iid, cb in caps if iid in common]
        hedge_by_slot, forb_by_slot = slot_stats(caps, slots)
        total_hedge_imgs = sum(1 for iid, cb in caps
                                if any(count_hedging(cb.get(s, "") or "") > 0 for s in slots))
        total_forb_imgs = sum(1 for iid, cb in caps
                              if any(count_forbidden(cb.get(s, "") or "") > 0 for s in slots))
        results[tag] = {
            "slots": slots, "caps": caps,
            "hedge_by_slot": hedge_by_slot, "forb_by_slot": forb_by_slot,
            "total_hedge_imgs": total_hedge_imgs, "total_forb_imgs": total_forb_imgs,
            "N": len(caps),
        }

    N = len(common)
    row = f"{'Total images with ANY hedging':<35s} | "
    row += " | ".join(f"{results[tag]['total_hedge_imgs']:>4d}/{N} ({100*results[tag]['total_hedge_imgs']/N:>3.0f}%)"
                      for tag in FILES)
    print(row)

    row = f"{'Total images with ANY forbidden':<35s} | "
    row += " | ".join(f"{results[tag]['total_forb_imgs']:>4d}/{N} ({100*results[tag]['total_forb_imgs']/N:>3.0f}%)"
                      for tag in FILES)
    print(row)

    print()
    print("=== HEDGING per slot (images containing hedge) ===")
    # Use v7.1 slot ordering as reference
    slot_names = FILES["v7.1"][2]
    v6b_slots = FILES["v6b"][2]
    print(f"  {'v7.1 slot':<24s} | " + " | ".join(f"{tag:>10s}" for tag in FILES))
    slot_map = {"C_global": "C_global",
                "C_head_face_eye": "C_head_bill (v6b)",
                "C_bill": "C_upperparts_wing (v6b)",
                "C_wing_back": "C_underparts (v6b)",
                "C_underparts": "C_tail_appendages (v6b)",
                "C_tail_legs": "C_pattern_markings (v6b)"}
    for i, s71 in enumerate(slot_names):
        s6b = v6b_slots[i]
        row = f"  {s71:<24s} | "
        row_parts = []
        for tag in FILES:
            slots = FILES[tag][2]
            slot = slots[i]
            n = results[tag]["hedge_by_slot"][slot]
            n_total = results[tag]["N"]
            pct = 100 * n / max(n_total, 1)
            row_parts.append(f"{n:>3d} ({pct:>3.0f}%)")
        row += " | ".join(row_parts)
        print(row)

    print()
    print("=== FORBIDDEN per slot ===")
    for i, s71 in enumerate(slot_names):
        row = f"  {s71:<24s} | "
        row_parts = []
        for tag in FILES:
            slots = FILES[tag][2]
            slot = slots[i]
            n = results[tag]["forb_by_slot"][slot]
            row_parts.append(f"{n:>3d}")
        row += " | ".join(row_parts)
        print(row)

    print()
    print("=== v7.1 hedging examples (if any) ===")
    for iid, cb in results["v7.1"]["caps"]:
        for slot in slot_names:
            text = cb.get(slot, "") or ""
            n = count_hedging(text)
            if n > 0:
                print(f"  {iid[:60]}  {slot}: {text}")


if __name__ == "__main__":
    main()
