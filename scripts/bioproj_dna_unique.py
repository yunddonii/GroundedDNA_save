#!/usr/bin/env python
"""DNA-unique(DB) before/after bio-projection, CPU-only (no GPU / no mAP recompute).

mAP@R is already recorded in docs/bio_projection_comparison.json (unchanged by
the [44.4,55.6]% = GC[8,10] band, which equals the [40,60]% band already used).
This adds the compositional axis the user asked for -- DNA-base uniqueness of the
DB codes -- pre and post projection, for every method and both code lengths.

18-base cells: GC count [8,10]  (user 44.4-55.6%).
24-base cells: GC count [10,14] (user 41.67-58.33%), 4-base codon Gen-0 runs.
"""
from __future__ import annotations
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.apply_bio_projection import get_base, project_batch_memoized


def dna_unique(codes):
    return len(np.unique(codes, axis=0)) / len(codes)


def cell(name, ddir, gc_min, gc_max, max_run=3):
    db = np.load(os.path.join(ddir, "extract_db.npz"), allow_pickle=True)
    dbb = get_base(db)
    L = dbb.shape[1]
    gc_c = [int(np.ceil(gc_min * L)), int(np.floor(gc_max * L))]
    proj = project_batch_memoized(dbb, gc_min, gc_max, max_run)
    return {
        "name": name, "dir": os.path.basename(ddir), "L": int(L),
        "gc_count_range": gc_c,
        "dna_unique_pre": dna_unique(dbb),
        "dna_unique_post": dna_unique(proj["projected_codes"]),
        "pre_compliance": float((proj["edit_distances"] == 0).mean()),
    }


def main():
    tasks = []
    # 18-base: ours P0refit + baselines E*-matched, GC[8,10]
    ours18 = {
        "Flickr25k": "result/260717+flickr25k_setting1_flickr_P0refit_e4+bs+64+e+60+proj_lr+0.001",
        "MSCOCO": "result/260717+mscoco_setting1_mscoco_P0refit_e49+bs+64+e+60+proj_lr+0.001",
        "NUSWIDE": "result/260717+nuswide_setting1_nuswide_P0refit_e4+bs+64+e+60+proj_lr+0.001",
        "CIFAR10": "result/260717+cifar10_setting1_cifar10_P0refit_e14+bs+64+e+60+proj_lr+0.001",
    }
    for ds, d in ours18.items():
        tasks.append((f"Ours|{ds}|18", d, 0.4444, 0.5556))
    for m in ["cibhash", "cimon", "mls3rduh"]:
        for ds in ["Flickr25k", "MSCOCO", "NUSWIDE", "CIFAR10"]:
            g = glob.glob(f"result_baseline/260721/{m}_{ds}_clip_E*_dnaeval")
            if g:
                tasks.append((f"{m}|{ds}|18", g[0], 0.4444, 0.5556))
    # 24-base: ours 4-base codon Gen-0, GC[10,14]
    ours24 = {
        "Flickr25k": "result/260715+flickr25k_setting1_flickr_F2_wholeimg_4base_L4_K128+bs+64+e+60+proj_lr+0.001",
        "MSCOCO": "result/260715+mscoco_setting1_mscoco_F2_wholeimg_4base_L4_K128+bs+64+e+60+proj_lr+0.001",
    }
    for ds, d in ours24.items():
        if os.path.isdir(d):
            tasks.append((f"Ours-4base|{ds}|24", d, 0.416, 0.584))

    out = []
    print(f"{'cell':26s}{'L':>3s}{'GC':>9s}{'dUniq pre':>11s}{'dUniq post':>12s}{'delta':>9s}", flush=True)
    for tag, d, lo, hi in tasks:
        name, ds, L = tag.split("|")
        r = cell(f"{name}|{ds}", d, lo, hi)
        r["dataset"] = ds
        out.append(r)
        print(f"{name+'/'+ds:26s}{r['L']:>3d}{str(r['gc_count_range']):>9s}"
              f"{r['dna_unique_pre']:>11.4f}{r['dna_unique_post']:>12.4f}"
              f"{r['dna_unique_post']-r['dna_unique_pre']:>+9.4f}", flush=True)
        json.dump(out, open("docs/bioproj_dna_unique.json", "w"), indent=2)
    print("wrote docs/bioproj_dna_unique.json", flush=True)


if __name__ == "__main__":
    main()
