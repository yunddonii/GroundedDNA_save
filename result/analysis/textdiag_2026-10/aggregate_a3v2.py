"""Aggregate a3_v2 JSONs: per dataset x arm, 3-seed mean of S (with per-seed values and CIs), for the
codeword and codon level and each pair rule; plus the calibration curve. Usage: aggregate_a3v2.py <dir> <out.md>"""
import glob, json, os, re, sys, collections
import numpy as np

RULES = ("lexical", "lexical_axis_exclusive", "caption_cos_top2pct")


def arm_of(name):
    ds = name.split("_")[0]
    arm = re.sub(r"_s4[234]$", "", name).replace(ds + "_setting1_", "").replace("msc_", "").replace("nus_", "")
    return ds, arm


def main(d, out_md):
    g = collections.defaultdict(list)
    for p in sorted(glob.glob(os.path.join(d, "*.json"))):
        r = json.load(open(p)); r["_name"] = os.path.basename(p)[:-5]
        g[arm_of(r["_name"])].append(r)
    L = ["# A3 v2 readings (S = mean over axes of log(lift_own / lift_other); bootstrap 95 % CI over images; lexical pair rule on the run's reference captions)", "",
         "Exploratory checkpoints (Gumbel ON) unless stated; validation rows only; descriptive, no test.", ""]
    for lvl in ("codeword", "codon"):
        for rule in RULES:
            if lvl == "codon" and rule == "caption_cos_top2pct":
                continue
            L += [f"## {lvl} level, rule = {rule}", "", "| dataset | arm | n | S per seed | S mean | CI95 per seed (lo..hi) | axes R>0 per seed |", "|---|---|---:|---|---:|---|---|"]
            for (ds, arm), rs in sorted(g.items()):
                S = [r[lvl][rule]["S"] for r in rs if r[lvl][rule]["S"]]
                if not S:
                    continue
                per_seed = " / ".join("%+.3f" % s["value"] for s in S)
                cis = " / ".join("[%+.2f,%+.2f]" % (s["ci95"][0], s["ci95"][1]) for s in S)
                pos = " / ".join(str(s["positive_axes"]) for s in S)
                L.append("| %s | %s | %d | %s | %+.3f | %s | %s |" % (ds, arm, len(S), per_seed, np.mean([s["value"] for s in S]), cis, pos))
            L.append("")
    L += ["## Calibration (codeword, lexical): S when a fraction f of rows carries the oracle code (3-seed mean; CI of seed 1)", "",
          "| dataset | arm | " + " | ".join(f"f={f}" for f in ("0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.7", "1.0")) + " |", "|---|---|" + "---|" * 8]
    for (ds, arm), rs in sorted(g.items()):
        cal = [r.get("calibration_lexical_codeword", {}) for r in rs]
        cells = []
        for f in ("0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.7", "1.0"):
            v = [c[f]["value"] for c in cal if c.get(f)]
            ci = next((c[f]["ci95"] for c in cal if c.get(f)), None)
            cells.append(("%+.2f [%+.2f,%+.2f]" % (np.mean(v), ci[0], ci[1])) if v and ci else "—")
        L.append(f"| {ds} | {arm} | " + " | ".join(cells) + " |")
    open(out_md, "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
