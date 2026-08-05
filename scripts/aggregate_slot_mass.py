#!/usr/bin/env python
"""Aggregate the 4 x 3seed x {train,test} slot-mass matrix.

Two questions:
  (b) is the dead slot the SAME across seeds?  A slot that dies in 3/3 seeds is
      a schema-vs-data mismatch; a slot that dies in 1/3 is training noise.
  (c) does it hold on the official test split, or is it an artifact of the rows
      the router was fitted on?

"Underused" = share < 1/4 of the uniform expectation (1/M).  Reported instead
of the script's much stricter --dead_frac because a slot at 1.5 % of the mass
is already unusable in practice even though it is not exactly zero.
"""
from __future__ import annotations
import glob, json, os, statistics as st

SLOTS = ["global", "primary_object", "secondary_object",
         "activity_relation", "color_texture", "scene_type"]
DS = ["Flickr25k", "MSCOCO", "NUSWIDE", "CIFAR10"]
SEEDS = [42, 43, 44]

rows = {}
for f in glob.glob("docs/newmodel_analysis/slotmass/*.json"):
    d = json.load(open(f))
    b = os.path.basename(f)[:-5]           # e.g. cifar_s42_test
    ds_key, sd, sp = b.rsplit("_", 2)
    rows[(d["dataset"], int(sd[1:]), sp)] = d

M = 6
uni = 1.0 / M
THR = 0.25 * uni

print(f"uniform share = {uni*100:.2f}%   underused threshold = {THR*100:.2f}%\n")
summary = {}
for ds in DS:
    have = [(s, sp) for s in SEEDS for sp in ("train", "test") if (ds, s, sp) in rows]
    if not have:
        print(f"{ds}: no data\n"); continue
    print(f"### {ds}   ({len(have)}/6 cells)")
    hdr = "".join(f"{f's{s}/{sp[:2]}':>11s}" for s, sp in have)
    print(f"  {'slot':20s}{hdr}{'mean':>9s}{'under':>7s}")
    ds_sum = {}
    for m in range(M):
        vals, marks = [], []
        for s, sp in have:
            v = rows[(ds, s, sp)]["share"][m] * 100
            vals.append(v)
            marks.append(f"{v:9.2f}{'!' if v < THR*100 else ' '} ")
        n_under = sum(1 for v in vals if v < THR * 100)
        mean = st.mean(vals)
        tag = "" if m == 0 else (f"{n_under}/{len(vals)}" if n_under else "-")
        print(f"  {SLOTS[m]:20s}{''.join(marks)}{mean:8.2f}%{tag:>7s}")
        ds_sum[SLOTS[m]] = {"mean_share_pct": round(mean, 3),
                            "per_cell_pct": [round(v, 3) for v in vals],
                            "n_underused": n_under, "n_cells": len(vals)}
    # consistency: same slot underused in every cell?
    dead_all = [SLOTS[m] for m in range(1, M)
                if ds_sum[SLOTS[m]]["n_underused"] == len(have)]
    dead_some = [SLOTS[m] for m in range(1, M)
                 if 0 < ds_sum[SLOTS[m]]["n_underused"] < len(have)]
    if dead_all:
        print(f"  => DEAD in ALL {len(have)} cells (seed- and split-independent): "
              f"{', '.join(dead_all)}")
    if dead_some:
        print(f"  => underused in SOME cells (unstable): {', '.join(dead_some)}")
    if not dead_all and not dead_some:
        print("  => all local slots healthy")
    print()
    summary[ds] = {"cells": [f"s{s}/{sp}" for s, sp in have], "slots": ds_sum,
                   "dead_all_cells": dead_all, "dead_some_cells": dead_some}

out = "docs/newmodel_analysis/slot_routing_mass_SUMMARY.json"
json.dump({"uniform_share": uni, "underused_threshold": THR,
           "per_dataset": summary}, open(out, "w"), indent=2)
print(f"wrote {out}")
