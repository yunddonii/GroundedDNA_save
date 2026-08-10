#!/usr/bin/env python
"""Which slot can we afford to DELETE? Interpretability side of the question.

`codebook_drop_ablation.py` answers the retrieval half: neutralise slot m's
three bases and see what mAP@R loses. It says nothing about the paper's leading
claim, so this is its counterpart on held-out codon decoding.

Two numbers per slot, both from the SAME dictionary machinery the paper reports
(`heldout_codon_decoding.decode_slot`), so they are directly comparable to the
published table:

  own      concept mAP of slot m decoded on its own -- how much concept
           information that slot's codon carries by itself.
  drop     slot-mean concept mAP over the REMAINING five slots when m is
           removed. This is what the reported number would become if the slot
           were deleted, and it is the one that matters: a slot can score well
           on its own and still be redundant, because the others already carry
           what it carries.

`delta = drop - full` is therefore the cost of deletion. A slot with delta >= 0
is free to remove -- the code loses nothing the other slots do not already say.

No retraining: the codes are read from the existing extraction, exactly as the
drop ablation does.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts"))

SLOTS = ["global", "primary_object", "secondary_object",
         "activity_relation", "color_texture", "scene_type"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--train_manifest", default=None)
    ap.add_argument("--bio_project", action="store_true")
    ap.add_argument("--gc_min_frac", type=float, default=0.4444)
    ap.add_argument("--gc_max_frac", type=float, default=0.5556)
    ap.add_argument("--max_run", type=int, default=3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import heldout_codon_decoding as H

    ours = a.result_dir
    w = os.path.join(ours, "withids")
    if os.path.exists(os.path.join(w, "extract_query.npz")):
        ours = w

    d = H.load_split(ours, a.train_manifest)
    tr_lab, te_lab = d["train_labels"], d["test_labels"]
    tr_idx, src, qy = d["train_idx"], d["train_src"], d["qy"]
    n_bases = src["base_indices"].shape[1] // H.N_SLOTS

    def _bioproj(arr):
        if not a.bio_project:
            return arr
        from dna_utils.bio_constraints import project_to_valid, is_valid_batch
        arr = np.ascontiguousarray(arr).astype(np.int8)
        uniq, inv = np.unique(arr, axis=0, return_inverse=True)
        valid = is_valid_batch(uniq, a.gc_min_frac, a.gc_max_frac, a.max_run)
        out = uniq.copy()
        for i in np.where(~valid)[0]:
            out[i], _ = project_to_valid(uniq[i], a.gc_min_frac, a.gc_max_frac, a.max_run)
        return out[inv].astype(np.int64)

    tr_u = H.codon_ids(_bioproj(src["base_indices"][tr_idx]), n_bases)
    qy_u = H.codon_ids(_bioproj(qy["base_indices"]), n_bases)
    n_units = 4 ** n_bases

    per = []
    for m in range(H.N_SLOTS):
        r = H.decode_slot(tr_u[:, m], tr_lab, qy_u[:, m], te_lab,
                          n_units, H.DEFAULT_ALPHA, H.DEFAULT_MIN_SUPPORT)
        per.append(r)
    ap_stack = np.stack([r["_ap"] for r in per])              # [6, N_test]
    full = float(np.nanmean(np.nanmean(ap_stack, axis=0)))

    print(f"{a.dataset}  test={len(te_lab)}  bases/slot={n_bases}  "
          f"bio_project={a.bio_project}")
    print(f"  full 6-slot concept mAP = {full:.4f}\n")
    print(f"  {'slot':20s}{'own':>9s}{'drop->5':>10s}{'delta':>9s}{'coverage':>10s}   verdict")
    rows = {}
    for m in range(H.N_SLOTS):
        keep = [k for k in range(H.N_SLOTS) if k != m]
        dropped = float(np.nanmean(np.nanmean(ap_stack[keep], axis=0)))
        delta = dropped - full
        own = per[m]["concept_mAP"]
        verdict = ("FREE to drop" if delta >= 0 else
                   ("cheap" if delta > -0.005 else "costly"))
        print(f"  {SLOTS[m]:20s}{own:9.4f}{dropped:10.4f}{delta:+9.4f}"
              f"{per[m]['coverage']:10.3f}   {verdict}")
        rows[SLOTS[m]] = {"own_concept_mAP": round(own, 5),
                          "concept_mAP_without_this_slot": round(dropped, 5),
                          "delta_vs_full": round(delta, 5),
                          "coverage": round(per[m]["coverage"], 5),
                          "n_active_units": int(per[m]["n_active_units"])}

    out = {"dataset": a.dataset, "dir": ours, "bases_per_slot": int(n_bases),
           "bio_project": bool(a.bio_project),
           "full_concept_mAP": round(full, 5), "slots": rows}
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=2)
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
