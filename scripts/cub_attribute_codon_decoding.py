#!/usr/bin/env python
"""Held-out codon decoding on CUB-200 with ATTRIBUTES as the target.

Why this exists. The reported codon-decoding probe asks whether a
`(slot, codon) -> concept` dictionary built on train predicts a test image's
concepts from the code alone. On 2026-08-10 that probe was shown to be invalid
on single-label data: CIFAR-10 carries exactly 1.00 positives per image, so
label-ranking AP collapses to reciprocal rank, every slot competes to predict
the SAME target, per-slot specialisation cannot help, and the slot that simply
re-encodes the CLIP global embedding through the saturated global gate wins.
Measured consequence -- on CIFAR a slot scores HIGHER on the images it could not
see at all (.9856) than on the ones it could (.8536), while on multi-label
NUS-WIDE the sign flips on 4 of 5 slots.

CUB-200 is also single-label by class (200 species, `MULTI_LABEL[CUB_200] =
False`), so class-target decoding would inherit exactly that flaw. But CUB ships
`attributes/image_attribute_labels.txt` -- 312 per-image binary attributes
(bill shape, wing colour, ...) -- which turns the SAME dataset into a genuine
multi-label target where different slots can carry different attributes and
specialisation is rewarded rather than punished.

Attributes are noisy: each is annotated with a certainty in 1..4. `--min_certainty`
drops low-confidence positives; the default 3 ("probably"/"definitely") follows
the usual CUB practice. Attributes that end up too rare or too common carry no
ranking signal and are dropped by `--min_pos_frac` / `--max_pos_frac`, which is
reported so the target set is auditable.

Everything downstream reuses `heldout_codon_decoding` unchanged -- same
dictionary construction, same smoothing, same min-support fallback, same
label-ranking AP -- so the numbers are directly comparable to the main table's.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts"))

SLOTS = ["global", "primary_object", "secondary_object",
         "activity_relation", "color_texture", "scene_type"]


def _load_attribute_matrix(root: str, min_certainty: int):
    """-> (rel_path -> row index, [n_images, 312] uint8)."""
    idx_to_rel = {}
    with open(os.path.join(root, "images.txt")) as fh:
        for line in fh:
            i, rel = line.strip().split(" ", 1)
            idx_to_rel[int(i)] = rel
    n_img = max(idx_to_rel)
    n_att = 312
    A = np.zeros((n_img + 1, n_att + 1), dtype=np.uint8)
    with open(os.path.join(root, "attributes", "image_attribute_labels.txt")) as fh:
        for line in fh:
            parts = line.split()
            # a few rows in the official release carry extra tokens
            if len(parts) < 4:
                continue
            img, att, present, cert = (int(parts[0]), int(parts[1]),
                                       int(parts[2]), int(float(parts[3])))
            if present and cert >= min_certainty:
                A[img, att] = 1
    return idx_to_rel, A


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", required=True)
    ap.add_argument("--cub_root", default="dataset/CUB_200")
    ap.add_argument("--min_certainty", type=int, default=3)
    ap.add_argument("--min_pos_frac", type=float, default=0.02)
    ap.add_argument("--max_pos_frac", type=float, default=0.98)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import heldout_codon_decoding as H

    ours = a.result_dir
    w = os.path.join(ours, "withids")
    if os.path.exists(os.path.join(w, "extract_query.npz")):
        ours = w
    db = np.load(os.path.join(ours, "extract_db.npz"), allow_pickle=True)
    qy = np.load(os.path.join(ours, "extract_query.npz"), allow_pickle=True)
    H._set_n_slots(int(db["codebook_indices"].shape[1]))
    n_bases = db["base_indices"].shape[1] // H.N_SLOTS

    idx_to_rel, A = _load_attribute_matrix(a.cub_root, a.min_certainty)
    rel_to_idx = {rel: i for i, rel in idx_to_rel.items()}

    def rows_for(split):
        paths = [os.path.basename(os.path.dirname(p)) + "/" + os.path.basename(p)
                 for p in split["image_paths"]]
        ids, keep = [], []
        for k, rel in enumerate(paths):
            i = rel_to_idx.get(rel)
            if i is not None:
                ids.append(i); keep.append(k)
        return np.asarray(keep, dtype=np.int64), np.asarray(ids, dtype=np.int64)

    db_keep, db_ids = rows_for(db)
    qy_keep, qy_ids = rows_for(qy)
    if len(db_keep) == 0 or len(qy_keep) == 0:
        raise SystemExit("image_paths did not match CUB images.txt; cannot build "
                         "the attribute target")

    tr_lab_full = A[db_ids, 1:]
    te_lab_full = A[qy_ids, 1:]
    frac = tr_lab_full.mean(axis=0)
    keep_att = (frac >= a.min_pos_frac) & (frac <= a.max_pos_frac)
    tr_lab = tr_lab_full[:, keep_att].astype(np.float64)
    te_lab = te_lab_full[:, keep_att].astype(np.float64)

    tr_u = H.codon_ids(db["base_indices"][db_keep], n_bases)
    qy_u = H.codon_ids(qy["base_indices"][qy_keep], n_bases)
    n_units = 4 ** n_bases

    print(f"CUB-200 attribute decoding   dir={os.path.basename(a.result_dir)}")
    print(f"  slots={H.N_SLOTS} bases/slot={n_bases}  train={len(tr_u)} test={len(qy_u)}")
    print(f"  attributes kept {int(keep_att.sum())}/312 "
          f"(pos-frac in [{a.min_pos_frac}, {a.max_pos_frac}], certainty >= {a.min_certainty})")
    print(f"  positives/image: train {tr_lab.sum(1).mean():.2f}  test {te_lab.sum(1).mean():.2f}"
          f"   <- MULTI-LABEL, unlike the 200-way class target")
    print()

    per = [H.decode_slot(tr_u[:, m], tr_lab, qy_u[:, m], te_lab,
                         n_units, H.DEFAULT_ALPHA, H.DEFAULT_MIN_SUPPORT)
           for m in range(H.N_SLOTS)]
    ap_stack = np.stack([r["_ap"] for r in per])
    slot_mean = float(np.nanmean(np.nanmean(ap_stack, axis=0)))

    # lower bounds, same construction as the main probe
    prior = (tr_lab.sum(axis=0) + H.DEFAULT_ALPHA) / (len(tr_lab) + 2 * H.DEFAULT_ALPHA)
    maj = float(np.nanmean(H.label_ranking_ap(
        np.repeat(prior[None, :], len(te_lab), axis=0), te_lab)))
    rng = np.random.default_rng(42)
    sh = []
    for _ in range(3):
        perm = rng.permutation(len(tr_u))
        s = [H.decode_slot(tr_u[perm, m], tr_lab, qy_u[:, m], te_lab,
                           n_units, H.DEFAULT_ALPHA, H.DEFAULT_MIN_SUPPORT)["_ap"]
             for m in range(H.N_SLOTS)]
        sh.append(float(np.nanmean(np.nanmean(np.stack(s), axis=0))))

    print(f"  {'slot':20s}{'concept mAP':>13s}{'coverage':>10s}{'active units':>14s}")
    rows = {}
    for m in range(H.N_SLOTS):
        print(f"  {SLOTS[m] if m < len(SLOTS) else m:20s}{per[m]['concept_mAP']:13.4f}"
              f"{per[m]['coverage']:10.3f}{per[m]['n_active_units']:14d}")
        rows[SLOTS[m] if m < len(SLOTS) else str(m)] = {
            "concept_mAP": round(per[m]["concept_mAP"], 5),
            "coverage": round(per[m]["coverage"], 5),
            "n_active_units": int(per[m]["n_active_units"]),
        }
    print(f"\n  slot-mean concept mAP = {slot_mean:.4f}")
    print(f"  majority (code ignored) = {maj:.4f}")
    print(f"  shuffled (structure destroyed) = {np.mean(sh):.4f} +-{np.std(sh):.4f}")
    print(f"  margin over majority = {slot_mean - maj:+.4f}")

    out = {"dataset": "CUB_200", "target": "attributes",
           "dir": ours, "slots": int(H.N_SLOTS), "bases_per_slot": int(n_bases),
           "n_attributes": int(keep_att.sum()),
           "min_certainty": a.min_certainty,
           "positives_per_image_test": round(float(te_lab.sum(1).mean()), 4),
           "slot_mean_concept_mAP": round(slot_mean, 5),
           "majority": round(maj, 5),
           "shuffled_mean": round(float(np.mean(sh)), 5),
           "per_slot": rows}
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=2)
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
