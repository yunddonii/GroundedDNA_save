"""Stage 8: A-prime + C on all four datasets against each dataset's stage-1 base (and K64 controls).

Flickr25K rows come from stage 7 (same code path). Probe JSONs live in stage6_p2anc/eval, concept
readings in stage7_bigarms/eval (Flickr) or stage8_extend/eval. Descriptive only: n = 3, no test.
"""
import csv
import glob
import json
import os
import statistics as st

E6 = "result/analysis/stage6_p2anc/eval"
W, W2 = "/data/yschoi/gdna_wt_mscoco/result", "/data/yschoi/gdna_wt_arms/result"
RUNS = {
    ("cifar10", "base"): f"{W}/*cifar10_setting1_cifar_base", ("cifar10", "ac2"): f"{W2}/*cifar10_setting1_cifar_ac2",
    ("flickr25k", "base"): "result/260920+flickr25k_setting1_base", ("flickr25k", "ac2"): f"{W2}/*flickr25k_setting1_ac2",
    ("flickr25k", "k64"): f"{W2}/*flickr25k_setting1_k64",
    ("nuswide", "base"): "result/*nuswide_setting1_nus_text", ("nuswide", "ac2"): f"{W2}/*nuswide_setting1_nus_ac2",
    ("nuswide", "k64"): f"{W2}/*nuswide_setting1_nus_k64",
    ("mscoco", "base"): f"{W}/*mscoco_setting1_msc0695", ("mscoco", "ac2"): f"{W2}/*mscoco_setting1_msc_ac2",
    ("mscoco", "k64"): f"{W2}/*mscoco_setting1_msc_k64",
}
AXES = ("C_primary_object", "C_secondary_object", "C_activity_or_relation", "C_color_texture")


def concept_file(ds, s):
    for p in (f"result/analysis/stage8_extend/eval/concept_{ds}_ac2_s{s}.json",
              f"result/analysis/stage7_bigarms/eval/concept_ac2_s{s}.json" if ds == "flickr25k" else ""):
        if p and os.path.exists(p):
            return p
    return None


def row(ds, arm, s):
    d = glob.glob(f"{RUNS[(ds, arm)]}_s{s}+*")
    t = f"{ds}_{arm}_s{s}"
    need = [f"{E6}/{p}_{t}.json" for p in ("zscr", "m1", "qgap")]
    if not d or not all(os.path.exists(p) for p in need):
        return None
    r = list(csv.DictReader(open(os.path.join(d[0], "log.csv"))))[-1]
    z = json.load(open(need[0]))
    out = {"mAP@R": float(r["eval_mAP_at_R"]), "unique": float(r["eval_unique_code_ratio"]),
           "dead": float(r["eval_dead_code_ratio_mean"]),
           "ceiling": z["mean_over_local_slots"]["supervised_ceiling"],
           "caption path": z["mean_over_local_slots"]["text_path"],
           "M1": json.load(open(need[1]))["M1_role"]["advantage_mean"],
           "code->axis": json.load(open(need[2]))["quantised_codeword"]["code_picks_own_axis"],
           "ceiling_axes": [z["slots"][ax]["supervised_ceiling"] for ax in AXES]}
    cf = concept_file(ds, s) if arm == "ac2" else None
    if cf:
        c = json.load(open(cf))
        out["concept name"] = c["mean"]["concept_name_reading"]
        out["concept hit"] = c["mean"]["deployed_equals_caption_concept"]
    return out


keys = ["mAP@R", "unique", "dead", "ceiling", "caption path", "concept name", "concept hit", "M1", "code->axis"]
res = {}
print("mean ± sample SD over seeds 42/43/44 (n shown); descriptive, no test")
print(f"{'dataset':10s}{'arm':5s}{'n':>2s}" + "".join(f"{k:>15s}" for k in keys))
for ds in ("cifar10", "flickr25k", "nuswide", "mscoco"):
    for arm in ("base", "ac2", "k64"):
        if (ds, arm) not in RUNS:
            continue
        rows = [x for x in (row(ds, arm, s) for s in (42, 43, 44)) if x]
        res[f"{ds}/{arm}"] = rows
        cells = []
        for k in keys:
            v = [x[k] for x in rows if k in x]
            cells.append(f"{st.mean(v):>8.4f}±{st.stdev(v):.4f}" if len(v) > 1 else
                         (f"{v[0]:>15.4f}" if v else f"{'—':>15s}"))
        print(f"{ds:10s}{arm:5s}{len(rows):>2d}" + "".join(cells))
print("\nreplication rule (pre-registered): ceiling >= base + .02 and >= base on 3/4 axes; concept name > base caption path;"
      " M1 > base on >= 2/3 seeds; retrieval loss > .09 flagged")
for ds in ("cifar10", "flickr25k", "nuswide", "mscoco"):
    b, a = res.get(f"{ds}/base", []), res.get(f"{ds}/ac2", [])
    if len(b) < 3 or len(a) < 3:
        print(f"{ds:10s} incomplete ({len(b)} base, {len(a)} ac2)")
        continue
    mb = lambda k, R: st.mean([x[k] for x in R])  # noqa: E731
    ax_b = [st.mean(x["ceiling_axes"][i] for x in b) for i in range(4)]
    ax_a = [st.mean(x["ceiling_axes"][i] for x in a) for i in range(4)]
    c1 = mb("ceiling", a) >= mb("ceiling", b) + .02 and sum(x >= y for x, y in zip(ax_a, ax_b)) >= 3
    c2 = mb("concept name", a) > mb("caption path", b)
    c3 = sum(x["M1"] > y["M1"] for x, y in zip(a, b)) >= 2
    dm = mb("mAP@R", a) - mb("mAP@R", b)
    print(f"{ds:10s} ceiling {mb('ceiling', a) - mb('ceiling', b):+.4f} (axes >= base: {sum(x >= y for x, y in zip(ax_a, ax_b))}/4) -> {c1}; "
          f"concept name {mb('concept name', a):.4f} vs caption path {mb('caption path', b):.4f} -> {c2}; "
          f"M1 up on {sum(x['M1'] > y['M1'] for x, y in zip(a, b))}/3 -> {c3}; mAP@R {dm:+.4f}{' FLAG' if dm < -.09 else ''}"
          f" | replicates: {c1 and c2 and c3}")
json.dump(res, open("result/analysis/stage8_extend/summary.json", "w"), indent=1)
