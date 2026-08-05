#!/usr/bin/env python
"""ON vs OFF for the adaptive top-p nucleus mask, per slot, same checkpoint."""
import glob, json, os
SLOTS = ["global", "primary_object", "secondary_object",
         "activity_relation", "color_texture", "scene_type"]
rows = {}
for f in glob.glob("docs/newmodel_analysis/toppabl/*.json"):
    b = os.path.basename(f)[:-5]                 # cifar_s42_toppon
    ds, sd, mode = b.rsplit("_", 2)
    rows[(ds, sd, mode.replace("topp", ""))] = json.load(open(f))
keys = sorted({(ds, sd) for ds, sd, _ in rows})
for ds, sd in keys:
    on, off = rows.get((ds, sd, "on")), rows.get((ds, sd, "off"))
    if not on or not off:
        continue
    print(f"### {on['dataset']} {sd}  (E*={on.get('epoch')})")
    print(f"  {'slot':20s}{'ON %':>9s}{'OFF %':>9s}{'OFF/ON':>9s}{'top1 %':>9s}")
    for m, name in enumerate(SLOTS):
        a, b_ = on["share"][m] * 100, off["share"][m] * 100
        r = b_ / a if a > 1e-9 else float("inf")
        mark = "  <- killed by top-p" if (a < 4.17 and b_ >= 4.17) else ""
        print(f"  {name:20s}{a:9.2f}{b_:9.2f}{r:9.2f}"
              f"{on['top1_frac'][m]*100:9.1f}{mark}")
    print()
