"""Stage 6b: what the axis-centred anchors change in the code itself (one run -> one JSON).

Deployment forward (no text) on the run's own stage-1 split: opt rows (dictionary side) and val rows
(held out).  Reports
  - slot redundancy: pairwise NMI between slot CODEWORD assignments and between slot CODONS (val+opt);
  - label decoding per slot (dictionary codon -> dataset-label posterior on opt, AP on val) and the
    joint naive-Bayes curve over codon subsets of size 1..5 (stage-3 method, on this split);
  - routing: effective slots per patch and the top-1 fraction from the last log row;
  - codebook use: distinct codewords / codons per slot on val.
Descriptive only.
"""
import argparse
import csv
import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "result", "analysis", "stage5_bigidea"))
os.environ.setdefault("GDNA_NUM_SEMANTIC_PARTS", "5")


def nmi(a, b):
    from sklearn.metrics import normalized_mutual_info_score
    return float(normalized_mutual_info_score(a, b))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    from slot_role_probe import parse_args_txt
    from heldout_codon_decoding import build_dictionary, label_ranking_ap
    from dataloaders import load_dataset
    from zscr_like import deployed_codes
    args = parse_args_txt(os.path.join(a.result_dir, "args.txt"))
    cache = args.siglip2_feature_cache_dir
    opt = set(int(r) for r in np.load(os.path.join(cache, "opt_train_rows.npy")))
    tr, _, _ = load_dataset(getattr(args, "dataset_dir", "dataset"), args.dataset, setting="setting1",
                            train_transform=None, test_transform=None, load_train=True, load_database=False,
                            load_test=False, return_index=True, qwen_text_cache_path=args.qwen_text_cache_path,
                            siglip2_feature_cache_dir=cache)
    rows = np.asarray(tr._feat_cache_rows)
    io = [i for i in range(len(tr)) if int(rows[i]) in opt]
    iv = [i for i in range(len(tr)) if int(rows[i]) not in opt]
    lab = lambda idx: np.stack([np.asarray(tr[i]["label"], np.int64) for i in idx])  # noqa: E731
    yo, yv = lab(io), lab(iv)
    co, bo = deployed_codes(a.result_dir, tr, io, a.device)       # codons [n,5], bases [n,15]
    cv, bv = deployed_codes(a.result_dir, tr, iv, a.device)
    # codeword indices are not returned by deployed_codes; codons are the deployed symbols, and the
    # pairwise NMI of codons is what the paper's s4.10.1 measures on the DB. Use codons here.
    call = np.concatenate([co, cv])
    pairs = {f"{i}-{j}": nmi(call[:, i], call[:, j]) for i in range(5) for j in range(i + 1, 5)}
    local_pairs = [v for k, v in pairs.items() if not k.startswith("0-")]
    rep = {"result_dir": a.result_dir, "n_opt": len(io), "n_val": len(iv),
           "codon_nmi_pairs": pairs, "codon_nmi_local_mean": float(np.mean(local_pairs)),
           "codon_nmi_global_local_mean": float(np.mean([pairs[f"0-{j}"] for j in range(1, 5)])),
           "distinct_codons_per_slot_val": [int(len(np.unique(cv[:, m]))) for m in range(5)],
           "unique_full_code_val": float(len(np.unique(bv, axis=0)) / len(bv))}
    # label decoding per slot and joint curve (naive Bayes over subsets), alpha 1, min_support 10
    alpha, ms = 1.0, 10
    prior = (yo.sum(0) + alpha) / (len(yo) + 2 * alpha)
    per_pos, per_slot_ap = [], []
    for m in range(5):
        p, sup = build_dictionary(co[:, m], yo, 64, alpha)
        post = np.where((sup[cv[:, m]] >= ms)[:, None], p[cv[:, m]], prior[None, :])
        per_pos.append(logit(post))
        per_slot_ap.append(float(np.nanmean(label_ranking_ap(post, yv))))
    base = logit(prior)[None, :]
    curve = {}
    for k in range(1, 6):
        aps = [float(np.nanmean(label_ranking_ap(sum(per_pos[m] for m in S) - (k - 1) * base, yv)))
               for S in itertools.combinations(range(5), k)]
        curve[k] = {"mean": float(np.mean(aps)), "max": float(np.max(aps))}
    rep["label_ap_per_slot"] = per_slot_ap
    rep["label_ap_prior"] = float(np.nanmean(label_ranking_ap(np.tile(prior, (len(yv), 1)), yv)))
    rep["joint_curve"] = curve
    rep["joint_gain_1_to_5"] = curve[5]["mean"] - curve[1]["mean"]
    last = list(csv.DictReader(open(os.path.join(a.result_dir, "log.csv"))))[-1]
    for k in ("val_routing_mean_effective_k", "val_routing_fraction_top1", "train_routing_mean_effective_k",
              "train_routing_fraction_top1"):
        rep[k] = float(last[k]) if last.get(k, "") != "" else None
    json.dump(rep, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("result_dir", "codon_nmi_pairs")}))


if __name__ == "__main__":
    main()
