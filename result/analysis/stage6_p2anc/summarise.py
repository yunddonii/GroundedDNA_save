"""Stage 6: axis-centred anchors (--axis_center anchors) against each dataset's stage-1 base, 3 seeds."""
import csv, json, os, statistics as st, sys
E = "result/analysis/stage6_p2anc/eval"
runs = {}
for f in ("eval_runs.txt", "eval_ready_2.txt", "eval_ready_3.txt"):
    p = os.path.join("result/analysis/stage6_p2anc", f)
    if os.path.exists(p):
        for line in open(p):
            ds, arm, s, d = line.split()
            if "PENDING" not in d:
                runs[(ds, arm, int(s))] = d
keys = ["mAP@R", "unique", "dead", "code->axis", "M1", "ceiling", "text path", "clip only"]
out = {}
print("mean ± sample SD over seeds 42/43/44 (n shown); descriptive, no test")
print(f"{'dataset':10s}{'arm':6s}{'n':>2s}" + "".join(f"{k:>16s}" for k in keys))
for ds in ("cifar10", "flickr25k", "nuswide", "mscoco"):
    for arm in ("base", "p2anc"):
        rows = []
        for s in (42, 43, 44):
            d = runs.get((ds, arm, s)); t = f"{ds}_{arm}_s{s}"
            if not d or not all(os.path.exists(f"{E}/{p}_{t}.json") for p in ("qgap", "m1", "zscr")):
                continue
            r = list(csv.DictReader(open(os.path.join(d, "log.csv"))))[-1]
            z = json.load(open(f"{E}/zscr_{t}.json"))["mean_over_local_slots"]
            rows.append({"mAP@R": float(r["eval_mAP_at_R"]), "unique": float(r["eval_unique_code_ratio"]),
                         "dead": float(r["eval_dead_code_ratio_mean"]),
                         "code->axis": json.load(open(f"{E}/qgap_{t}.json"))["quantised_codeword"]["code_picks_own_axis"],
                         "M1": json.load(open(f"{E}/m1_{t}.json"))["M1_role"]["advantage_mean"],
                         "ceiling": z["supervised_ceiling"], "text path": z["text_path"], "clip only": z["clip_only"]})
        out[f"{ds}/{arm}"] = rows
        if not rows:
            print(f"{ds:10s}{arm:6s} 0"); continue
        cell = lambda k: (f"{st.mean([x[k] for x in rows]):>9.4f}±{st.stdev([x[k] for x in rows]):.4f}"  # noqa: E731
                          if len(rows) > 1 else f"{rows[0][k]:>16.4f}")
        print(f"{ds:10s}{arm:6s}{len(rows):>2d}" + "".join(cell(k) for k in keys))
json.dump(out, open("result/analysis/stage6_p2anc/summary.json", "w"), indent=1)
