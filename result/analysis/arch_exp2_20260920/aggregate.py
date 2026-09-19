#!/usr/bin/env python
"""Aggregate arms vs the same-mode baseline and apply the PRE-REGISTERED operational rule
(PREREGISTRATION.md). Descriptive only: n = 3, no significance test is performed or implied."""
import csv, glob, json, os, statistics as st, sys

ROOT = "/home/yschoi/GroundedDNA"
EXP = f"{ROOT}/result/analysis/arch_exp2_20260920"
ARMS = [a for a in (sys.argv[1:] or ["base", "arm_a1", "arm_b1", "arm_c1"])]


def cell(tag):
    d = glob.glob(f"{ROOT}/result/260920+flickr25k_setting1_{tag}+*")
    if not d:
        return None
    r = list(csv.DictReader(open(d[0] + "/log.csv")))
    if not r or r[-1].get("eval_mAP_at_R", "") == "":
        return None
    last = r[-1]
    out = {"mAP": float(last["eval_mAP_at_R"]), "dead": float(last["eval_dead_code_ratio_mean"]),
           "unique": float(last["eval_unique_code_ratio"]), "epochs": len(r)}
    for k in ("train_loss_mec", "train_mec_acc"):
        if last.get(k, "") != "":
            out[k] = float(last[k])
    pj = f"{EXP}/probe/{tag}.json"
    if os.path.exists(pj):
        p = json.load(open(pj))
        D = p["M1_role"]["D"]
        out.update({"M1_adv": p["M1_role"]["advantage_mean"], "M1_own_argmax": p["M1_role"]["own_slot_is_argmax"],
                    "M1_diag": sum(D[i][i] for i in range(4)) / 4,
                    "M1_shuffled": sum(map(sum, p["M1_role"]["D_shuffled_control"])) / 16,
                    "M1_adv_axes": p["M1_role"]["advantage_per_axis"],
                    "M2": p["M2_label_decoding"]["mean"], "M3_ratio": p["M3_codebook"]["min_over_median"],
                    "M3_pp": p["M3_codebook"]["perplexity"]})
        if p.get("M4_mass"):
            out["M4_maxmin"] = p["M4_mass"]["per_image_max_over_min_median"]
            out["M4_empty"] = p["M4_mass"]["fraction_images_with_empty_slot"]
    return out


def ms(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    return st.mean(xs), (st.stdev(xs) if len(xs) > 1 else float("nan"))


rows = {}
for arm in ARMS:
    rows[arm] = {s: cell(f"{arm}_s{s}") for s in (42, 43, 44)}

keys = ["mAP", "dead", "unique", "M1_adv", "M1_own_argmax", "M1_diag", "M1_shuffled", "M2", "M3_ratio", "M4_maxmin", "M4_empty"]
summary = {}
for arm, cs in rows.items():
    summary[arm] = {k: ms([c.get(k) for c in cs.values() if c]) for k in keys}
    summary[arm]["n"] = sum(1 for c in cs.values() if c)

print(f"{'arm':8s} {'n':>2s} " + " ".join(f"{k:>14s}" for k in keys))
for arm in ARMS:
    s = summary[arm]
    print(f"{arm:8s} {s['n']:>2d} " + " ".join(
        (f"{s[k][0]:7.4f}±{s[k][1]:.4f}" if s[k][0] is not None and s[k][1] == s[k][1] else
         (f"{s[k][0]:14.4f}" if s[k][0] is not None else f"{'-':>14s}")) for k in keys))

b = summary.get("base")
verdicts = {}
if b and b["mAP"][0] is not None and b["M1_adv"][0] is not None:
    print("\nPRE-REGISTERED RULE (operational, not a test): pass iff")
    print(f"  (1) mean mAP >= {b['mAP'][0]:.4f} - {b['mAP'][1]:.4f} = {b['mAP'][0]-b['mAP'][1]:.4f}")
    print(f"  (2) mean M1_adv - {b['M1_adv'][0]:.4f} > {b['M1_adv'][1]:.4f}")
    for arm in ARMS:
        if arm == "base" or summary[arm]["n"] < 3 or summary[arm]["M1_adv"][0] is None:
            continue
        s = summary[arm]
        c1 = s["mAP"][0] >= b["mAP"][0] - b["mAP"][1]
        dA = s["M1_adv"][0] - b["M1_adv"][0]
        c2 = dA > b["M1_adv"][1]
        verdicts[arm] = {"retrieval_noninferior": c1, "role_gain": c2, "pass": c1 and c2,
                         "d_mAP": s["mAP"][0] - b["mAP"][0], "d_M1_adv": dA,
                         "d_M2": (s["M2"][0] - b["M2"][0]) if s["M2"][0] is not None else None}
        print(f"  {arm:8s} dmAP={s['mAP'][0]-b['mAP'][0]:+.4f} (1)={c1}   dM1adv={dA:+.4f} (2)={c2}   dM2={verdicts[arm]['d_M2']:+.4f}   -> {'PASS' if c1 and c2 else 'fail'}")
json.dump({"arms": ARMS, "per_cell": rows, "summary": {a: {k: v for k, v in s.items()} for a, s in summary.items()},
           "rule_verdicts": verdicts}, open(f"{EXP}/aggregate_{'_'.join(ARMS)}.json", "w"), indent=1, default=str)
