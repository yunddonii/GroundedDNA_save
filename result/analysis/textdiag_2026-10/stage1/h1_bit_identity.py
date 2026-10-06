"""H1 tier-0 check: the cell with lambda_anchor = lambda_cibhash_kl = lambda_codon_text_anchor = lambda_recon = 0
must have parameters bit-identical to B0 (same seed, same launch mode) and identical shared log.csv columns
(train_loss differs by .05 x train_loss_anchor up to float32 rounding). Usage: h1_bit_identity.py <B0 dir> <H1 dir> <out.json>"""
import csv, json, sys
import torch


def main(b0, h1, out):
    sb = torch.load(f"{b0}/model_state_dict.pth", map_location="cpu", weights_only=False)
    sh = torch.load(f"{h1}/model_state_dict.pth", map_location="cpu", weights_only=False)
    keys_b, keys_h = set(sb), set(sh)
    diff = [k for k in keys_b & keys_h if not torch.equal(sb[k], sh[k])]
    maxabs = max((float((sb[k].float() - sh[k].float()).abs().max()) for k in diff), default=0.0)
    rb = list(csv.DictReader(open(f"{b0}/log.csv"))); rh = list(csv.DictReader(open(f"{h1}/log.csv")))
    shared = sorted((set(rb[0]) & set(rh[0])) - {"train_loss"})
    col_diff = {}
    for c in shared:
        try:
            d = max(abs(float(a[c]) - float(b[c])) for a, b in zip(rb, rh))
        except (ValueError, TypeError):
            d = 0.0 if all(a[c] == b[c] for a, b in zip(rb, rh)) else 1.0
        if d > 0:
            col_diff[c] = d
    only_b0 = sorted(set(rb[0]) - set(rh[0])); only_h1 = sorted(set(rh[0]) - set(rb[0]))
    tl = [(float(a["train_loss"]), float(b["train_loss"]), float(a.get("train_loss_anchor", "nan"))) for a, b in zip(rb, rh)]
    res = {"B0": b0, "H1": h1, "tensors_compared": len(keys_b & keys_h), "tensors_only_in_one": sorted(keys_b ^ keys_h),
           "tensors_differing": diff, "max_abs_param_diff": maxabs, "log_rows": [len(rb), len(rh)],
           "shared_log_columns_differing": col_diff, "columns_only_B0": only_b0, "columns_only_H1": only_h1,
           "train_loss_B0_minus_H1_vs_0.05xanchor": [(round(a - b, 6), round(0.05 * c, 6)) for a, b, c in tl],
           "PASS": (not diff) and (not col_diff) and len(rb) == len(rh)}
    json.dump(res, open(out, "w"), indent=1); print(json.dumps({k: res[k] for k in ("tensors_compared", "tensors_differing", "max_abs_param_diff", "shared_log_columns_differing", "columns_only_B0", "columns_only_H1", "PASS")}))
    print("train_loss diff vs .05*anchor:", res["train_loss_B0_minus_H1_vs_0.05xanchor"])


if __name__ == "__main__":
    main(*sys.argv[1:4])
