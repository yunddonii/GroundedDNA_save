"""Stage 12 reducer, fixed before the cells finish.

Rule, identical to scripts/phase3_select_n.py: argmax of raw base-Hamming mAP@R at the candidate's
OWN terminal epoch; ties go to the smallest N. Values come from the terminal row of each cell's
log.csv on the 10 % held-out train split. No official test is read.
"""
import csv, glob, json, os, sys

R = "/data/yschoi/gdna_wt_mscoco/result"
CAND = (4, 9, 19, 39)
INCUMBENT = {"flickr25k": 4, "cifar10": 19, "nuswide": 4, "mscoco": 39}
SHORT = {"flickr25k": "fl", "cifar10": "ci", "nuswide": "nu", "mscoco": "ms"}
REUSED = {("cifar10", 19): "*cifar10_setting1_cifar_p2anc_s42+*",      # stage 6b, same tree/commit
          ("nuswide", 4): "*nuswide_setting1_nus_p2anc_s42+*",
          ("mscoco", 39): "*mscoco_setting1_msc_p2anc_s42+*"}


def cell_dir(ds, n):
    pat = REUSED.get((ds, n), f"*{ds}_setting1_{SHORT[ds]}anc_N{n}_s42+*")
    hits = glob.glob(os.path.join(R, pat))
    if len(hits) != 1:
        raise SystemExit(f"REFUSED {ds} N={n}: {len(hits)} directories match {pat}")
    return hits[0]


def terminal(d, n):
    rows = list(csv.DictReader(open(os.path.join(d, "log.csv"))))
    r = rows[-1]
    if int(r["epoch"]) != n:
        raise SystemExit(f"REFUSED {d}: terminal epoch {r['epoch']} != candidate N {n}")
    if not r["eval_mAP_at_R"]:
        raise SystemExit(f"REFUSED {d}: no eval at the terminal epoch")
    return float(r["eval_mAP_at_R"])


out = {}
print(f"{'dataset':10s} " + "".join(f"{'N=' + str(n):>12s}" for n in CAND) + f"{'chosen':>9s}{'incumbent':>11s}")
for ds in ("flickr25k", "cifar10", "nuswide", "mscoco"):
    vals = {n: terminal(cell_dir(ds, n), n) for n in CAND}
    best = max(vals.values())
    chosen = min(n for n in CAND if vals[n] == best)          # ties -> smallest N
    out[ds] = {"values": vals, "chosen_n": chosen, "incumbent_n": INCUMBENT[ds],
               "moved": chosen != INCUMBENT[ds]}
    print(f"{ds:10s} " + "".join(f"{vals[n]:12.6f}" for n in CAND) + f"{chosen:9d}{INCUMBENT[ds]:11d}"
          + ("   MOVED" if chosen != INCUMBENT[ds] else ""))
json.dump(out, open(os.path.join(os.path.dirname(__file__), "selected_n.json"), "w"), indent=1)
