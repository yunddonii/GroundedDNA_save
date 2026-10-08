"""Training-length comparison for every Stage 1-3 Flickr variation (user request 2026-10-09).

For each arm: the original N=4 cells (eval at epoch 4) and the N=19 re-runs (eval at epochs 4/9/14/19,
same delta as B1's L19): validation mAP@R, unique-code ratio, dead-codeword ratio (log.csv), and the
deployed A3 v2 S (codon / codeword) on the final checkpoint of each run when scored.
Run dirs are looked up in the main tree's result/ and in the worktree's result/.
Usage: length_table.py [--json out.json]
"""
import argparse, csv, glob, json, os, re, statistics as st

ROOTS = ["/home/yschoi/GroundedDNA/result", "/home/yschoi/gdna_textdiag/result"]
TD = "/home/yschoi/gdna_textdiag/result/analysis/textdiag_2026-10"
SCORE_DIRS = [os.path.join(TD, d) for d in ("stage1/a3v2_2000", "stage3/a3v2_2000", "stageL/a3v2_2000")]
# arm label -> (N=4 tag stem, N=19 tag stem)
ARMS = [("B0", "td1_flickr_B0", "td1_flickr_B0_N19"), ("B1 (=H2)", "td1_flickr_H2", "td3_L19"),
        ("OFF", "td1_flickr_OFF", "td1_flickr_OFF_N19"),
        ("TD", "td3_TD", "td3_TD_N19"), ("XM0", "td3_XM0", "td3_XM0_N19"), ("TDXM", "td3_TDXM", "td3_TDXM_N19"),
        ("TDXM5", "td3_TDXM5", "td3_TDXM5_N19"), ("S", "td3_S", "td3_S_N19"), ("SN2", "td3_SN2", "td3_SN2_N19"),
        ("BN2", "td3_BN2", "td3_BN2_N19"), ("P-mem+SS", "td3_PmemSS", "td3_L19p"), ("P-head", "td3_Phead", "td3_Phead_N19"),
        ("P-head noSS", "td3_PheadNoSS", "td3_PheadNoSS_N19"), ("MIX", "td3_MIX", "td3_MIX_N19")]
N4_OVERRIDE = {("td3_TD", 42): "td3_TDv2_s42"}   # the s42 TD cell is the post-fix rerun


def run_dir(tag):
    for r in ROOTS:
        ds = [d for d in glob.glob(os.path.join(r, f"*_setting1_{tag}+*")) if os.path.exists(os.path.join(d, "log.csv"))]
        if ds:
            return ds[0]
    return None


def evals(d):
    out = {}
    for r in csv.DictReader(open(os.path.join(d, "log.csv"))):
        if r.get("eval_mAP_at_R"):
            out[int(float(r["epoch"]))] = (float(r["eval_mAP_at_R"]), float(r["eval_unique_code_ratio"]), float(r["eval_dead_code_ratio_mean"]))
    return out


def a3(tag):
    for sd in SCORE_DIRS:
        for f in glob.glob(os.path.join(sd, f"*_{tag}.json")):
            d = json.load(open(f))
            return d["codon"]["lexical"]["S"]["value"], d["codeword"]["lexical"]["S"]["value"]
    return None


def fmt(xs, k=3):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "—"
    m = st.mean(xs)
    return f"{m:.{k}f}" + (f"±{st.stdev(xs):.{k}f}" if len(xs) > 1 else "")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--json", default=None); a = ap.parse_args()
    rows = []
    for label, t4, t19 in ARMS:
        rec = {"arm": label, "seeds": {}}
        for s in (42, 43, 44):
            tag4 = N4_OVERRIDE.get((t4, s), f"{t4}_s{s}"); tag19 = f"{t19}_s{s}"
            d4, d19 = run_dir(tag4), run_dir(tag19)
            rec["seeds"][s] = {"N4": {"tag": tag4, "run_dir": d4, "eval": evals(d4) if d4 else {}, "A3": a3(tag4)},
                               "N19": {"tag": tag19, "run_dir": d19, "eval": evals(d19) if d19 else {}, "A3": a3(tag19)}}
        rows.append(rec)
    print("mean ± SD over seeds 42/43/44 (validation, Flickr25K). N4 = original cell (epoch 4); N19 = re-run with --stop_after_epoch 19 --sinkhorn_schedule_horizon 20")
    hdr = f"{'arm':12s} | {'mAP@R N4':12s} {'N19@e4':12s} {'@e9':12s} {'@e14':12s} {'@e19':12s} | {'unique N4':11s} {'N19@e19':11s} | {'dead N4':11s} {'N19@e19':11s} | {'codon S N4':12s} {'N19':12s} | {'codeword S N4':13s} {'N19':12s} | n19"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        S = r["seeds"]
        g = lambda key, e, i: [S[s][key]["eval"].get(e, (None, None, None))[i] for s in S]
        aa = lambda key, i: [S[s][key]["A3"][i] if S[s][key]["A3"] else None for s in S]
        n19 = sum(1 for s in S if S[s]["N19"]["eval"].get(19))
        print(f"{r['arm']:12s} | {fmt(g('N4',4,0)):12s} {fmt(g('N19',4,0)):12s} {fmt(g('N19',9,0)):12s} {fmt(g('N19',14,0)):12s} {fmt(g('N19',19,0)):12s} | "
              f"{fmt(g('N4',4,1)):11s} {fmt(g('N19',19,1)):11s} | {fmt(g('N4',4,2)):11s} {fmt(g('N19',19,2)):11s} | "
              f"{fmt(aa('N4',0)):12s} {fmt(aa('N19',0)):12s} | {fmt(aa('N4',1)):13s} {fmt(aa('N19',1)):12s} | {n19}/3")
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
