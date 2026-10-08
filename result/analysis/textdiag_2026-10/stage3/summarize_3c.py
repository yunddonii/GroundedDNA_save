"""Stage 3-C summary: per arm x seed, training metrics (log.csv last row), deployed codon / codeword S
(a3v2_2000), train/deploy codeword agreement (d4d6), against B1 (td1_flickr_H2). Usage: summarize_3c.py [arms...]"""
import csv, glob, json, os, sys
S3 = os.path.dirname(os.path.abspath(__file__)); S1 = os.path.join(os.path.dirname(S3), "stage1", "a3v2_2000")
RES = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(S3))))  # .../result
ARMS = sys.argv[1:] or ["PmemSS", "Phead", "PheadNoSS", "MIX"]
B1 = {"mAP": {42: .7569, 43: .7474, 44: .7405}, "unique": {42: .557, 43: .554, 44: .563}, "dead": {42: .155, 43: None, 44: None}}

def a3(path):
    d = json.load(open(path)); out = {}
    for lvl in ("codon", "codeword"):
        x = d[lvl]["lexical"]; out[lvl] = (x["S"]["value"], x["S"]["ci95"], {ax[2:9]: round(v["R"], 2) for ax, v in x["per_axis"].items()})
    return out

def last_row(run_dir):
    rows = list(csv.DictReader(open(os.path.join(run_dir, "log.csv")))); r = rows[-1]
    f = lambda k: (round(float(r[k]), 4) if r.get(k) not in (None, "") else None)
    return {k: f(k) for k in ("eval_mAP_at_R", "eval_unique_code_ratio", "eval_dead_code_ratio_mean", "train_anchor_fidelity", "train_anchor_ss_p", "train_loss_anchor_pred", "train_anchor_mix_alpha")}

b1 = {s: a3(os.path.join(S1, f"flickr25k_setting1_td1_flickr_H2_s{s}.json")) for s in (42, 43, 44)}
for s in (42, 43, 44):
    rd = glob.glob(os.path.join(RES, f"*td1_flickr_H2_s{s}+*"))[0]; lr = last_row(rd); B1["dead"][s] = lr["eval_dead_code_ratio_mean"]; B1["mAP"][s] = lr["eval_mAP_at_R"]; B1["unique"][s] = lr["eval_unique_code_ratio"]
print("B1 (H2):", {s: {"mAP": B1["mAP"][s], "unique": B1["unique"][s], "dead": B1["dead"][s], "codon S": round(b1[s]["codon"][0], 3), "codeword S": round(b1[s]["codeword"][0], 3)} for s in (42, 43, 44)})
for arm in ARMS:
    print(f"\n=== {arm}")
    for s in (42, 43, 44):
        rds = glob.glob(os.path.join(RES, f"*td3_{arm}_s{s}+*"))
        if not rds: print(f"s{s}: no run"); continue
        lr = last_row(rds[0]); ap = os.path.join(S3, "a3v2_2000", f"flickr25k_setting1_td3_{arm}_s{s}.json"); dp = os.path.join(S3, "d4d6", f"flickr25k_setting1_td3_{arm}_s{s}.json")
        line = f"s{s}: mAP@R {lr['eval_mAP_at_R']} ({lr['eval_mAP_at_R'] - B1['mAP'][s]:+.4f}) unique {lr['eval_unique_code_ratio']} ({lr['eval_unique_code_ratio'] - B1['unique'][s]:+.3f}) dead {lr['eval_dead_code_ratio_mean']} fid {lr['train_anchor_fidelity']} p {lr['train_anchor_ss_p']} pred {lr['train_loss_anchor_pred']} alpha {lr['train_anchor_mix_alpha']}"
        if os.path.exists(ap):
            x = a3(ap)
            line += f"\n      codon S {x['codon'][0]:+.3f} ci {x['codon'][1]} (B1 {b1[s]['codon'][0]:+.3f}; delta {x['codon'][0] - b1[s]['codon'][0]:+.3f}) R {x['codon'][2]}"
            line += f"\n      codeword S {x['codeword'][0]:+.3f} ci {x['codeword'][1]} (B1 {b1[s]['codeword'][0]:+.3f}; delta {x['codeword'][0] - b1[s]['codeword'][0]:+.3f})"
        if os.path.exists(dp):
            d = json.load(open(dp)); ag = {ax[2:9]: v.get("p_same_codeword_train_vs_deploy") for ax, v in d["D4_pairwise"].items()}
            line += f"\n      d4d6 train/deploy agreement {ag}; train-routed codeword S {d['A3_codeword_lexical_train_routed']['S']['value']:+.3f}, deployed {d['A3_codeword_lexical_deployed']['S']['value']:+.3f}"
        print(line)
