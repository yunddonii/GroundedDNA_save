"""Assemble the P0 comparison table: ours (val-selected) vs baselines (val-selected).

Both sides select their epoch on the same held-out val split (seed 42, ratio
0.1) at the same 5-epoch cadence using val mAP@R, and each reports a single
evaluation on the official test split. This is the paper's table of record.

Usage: python scripts/build_p0_table.py [--out docs/p0_comparison.json]
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DS_LABEL = {"flickr25k": "Flickr25k (@5000)", "mscoco": "MS-COCO (@5000)",
            "nuswide": "NUS-WIDE (@5000)", "cifar10": "CIFAR-10 (@1000)"}
ORDER = ["flickr25k", "mscoco", "nuswide", "cifar10"]


def ours() -> dict:
    out = {}
    for d in glob.glob("result/*P0val*/"):
        f = os.path.join(d, "evaluation_siglip2_base.json")
        if not os.path.exists(f):
            continue
        ds = os.path.basename(d.rstrip("/")).split("+")[1].split("_setting1")[0]
        ev = json.load(open(f))
        pk = ev.get("precision_at_k", {})
        row = {"mAP_at_R": ev["mAP_at_R"], "mAP": ev["mAP"],
               "P@1": pk.get("1"), "P@10": pk.get("10"),
               "dna_unique_db": ev.get("unique_code_ratio")}
        p = os.path.join(d, "pairwise_nmi.json")
        if os.path.exists(p):
            row["NMI"] = json.load(open(p))["mean_off_diag_nmi"]
        c = os.path.join(d, "compositional_eval.json")
        if os.path.exists(c):
            mb = json.load(open(c)).get("metric_b") or {}
            for tag, k in (("b0_raw_text", "B0"), ("b1_centered_text", "B1"),
                           ("b2_visual_global", "B2")):
                if tag in mb:
                    row[k] = mb[tag]["mean_compositional_lift"]
        for dd in glob.glob(os.path.join(d, "codebook_drop_ablation_subset*.json")):
            k = json.load(open(dd))
            base = k["baseline"]["mAP"]
            row["drop_mean"] = float(np.mean([v["mAP"] - base for v in k["drops"].values()]))
        # epoch chosen by val
        for lg in glob.glob(f"logs/*{ds.replace('25k','')}*P0val*.log"):
            m = re.search(r"swapping final checkpoint with best \(epoch (\d+), "
                          r"eval_mAP_at_R=([0-9.]+)", open(lg, errors="ignore").read())
            if m:
                row["val_selected_epoch"] = int(m.group(1))
                row["val_mAP_at_R"] = float(m.group(2))
                break
        out[ds] = row
    return out


def baselines() -> dict:
    out = {}
    for f in glob.glob("docs/baseline_val_select/*.json"):
        m, ds = os.path.basename(f)[:-5].split("_", 1)
        j = json.load(open(f))
        out.setdefault(ds, {})[m] = {
            "mAP_at_R": j["test_mAP_at_R_val_selected"],
            "epoch": j["selected_epoch_by_val"],
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/p0_comparison.json")
    args = ap.parse_args()

    o, b = ours(), baselines()

    print("\n표 1. mAP@R (36-bit, frozen CLIP-ViT-B/16, whole-image, P0 val-selected)\n")
    print("| Dataset (cutoff) | GroundedDNA | CIBHash | CIMON | MLS3RDUH |")
    print("|---|---:|---:|---:|---:|")
    for ds in ORDER:
        if ds not in o:
            print(f"| {DS_LABEL[ds]} | (pending) | | | |")
            continue
        cells = [f"{o[ds]['mAP_at_R']:.4f}"]
        vals = {"ours": o[ds]["mAP_at_R"]}
        for m in ("cibhash", "cimon", "mls3rduh"):
            v = b.get(ds, {}).get(m)
            cells.append(f"{v['mAP_at_R']:.4f}" if v else "-")
            if v:
                vals[m] = v["mAP_at_R"]
        win = max(vals, key=vals.get)
        names = ["ours", "cibhash", "cimon", "mls3rduh"]
        cells = [f"**{c}**" if names[i] == win else c for i, c in enumerate(cells)]
        print(f"| {DS_LABEL[ds]} | " + " | ".join(cells) + " |")

    print("\n델타 (ours - best baseline):")
    for ds in ORDER:
        if ds not in o or ds not in b:
            continue
        bb = max(b[ds].items(), key=lambda kv: kv[1]["mAP_at_R"])
        d = o[ds]["mAP_at_R"] - bb[1]["mAP_at_R"]
        print(f"  {ds:10s}: {d:+.4f}  (vs {bb[0]} {bb[1]['mAP_at_R']:.4f}; "
              f"ours epoch {o[ds].get('val_selected_epoch')}, {bb[0]} epoch {bb[1]['epoch']})")

    print("\n4축 compositional (ours, P0):")
    for ds in ORDER:
        if ds not in o:
            continue
        r = o[ds]
        f = lambda k: f"{r[k]:.4f}" if r.get(k) is not None else "n/a"
        print(f"  {ds:10s}: P@1={f('P@1')} NMI={f('NMI')} DNA-uniq={f('dna_unique_db')} "
              f"B0={f('B0')} B1={f('B1')} B2={f('B2')} drop={f('drop_mean')}")

    json.dump({"ours": o, "baselines": b}, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
