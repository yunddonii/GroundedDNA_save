"""Re-select a baseline's epoch on a held-out validation split (P0, symmetric).

Our model selects its checkpoint on val mAP@R (P0 protocol). Comparing that
against baselines selected on TEST mAP would hand the baselines the very leak we
removed from ours; comparing against an arbitrary final epoch is not symmetric
either. Since the baselines already checkpoint every `eval_period` epochs AND
already wrote a per-epoch TEST eval, no retraining is needed: score each saved
checkpoint on the SAME val split ours uses (val_split.py, seed 42, ratio 0.1),
pick the best-val epoch, then report that epoch's already-computed test mAP@R.

The val split is carved from the baseline's TRAIN split, so the reported test
number stays untouched by selection.

Usage:
    python scripts/baseline_val_select.py \
        --params_dir  params_baseline/260714/cibhash_nuswide_clip_mapr_unsup60 \
        --result_dir  result_baseline/260714/cibhash_nuswide_clip_mapr_unsup60 \
        --cache_dir   cache/nuswide_clip_tokens \
        --device cuda:4 --out docs/baseline_val_select_nuswide_cibhash.json
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baseline.base_model import (BackboneWithEncoder, _extract_codes,
                                 evaluate_retrieval_model, CachedFeatureDataset,
                                 MAP_AT_R_BY_DATASET)
from val_split import carve_val_indices


def build_loader(ds, idx, bs, workers):
    from torch.utils.data import DataLoader, Subset
    return DataLoader(Subset(ds, [int(i) for i in idx]), batch_size=bs,
                      shuffle=False, num_workers=workers, drop_last=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params_dir", required=True)
    ap.add_argument("--result_dir", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--dataset_dir", default="dataset")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--val_split_ratio", type=float, default=0.1)
    ap.add_argument("--val_split_seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(args.result_dir, "config.json")))
    dataset, setting, bit = cfg["dataset"], cfg["setting"], int(cfg["bit"])
    map_r = MAP_AT_R_BY_DATASET.get(str(dataset))
    print(f"[val-select] {cfg['trial_name']}: dataset={dataset} bit={bit} mAP@R cutoff={map_r}")

    trainset = CachedFeatureDataset(dataset, setting, "train", args.dataset_dir,
                                    args.cache_dir)
    labels = np.asarray(trainset.labels)
    opt_idx, val_idx, strat = carve_val_indices(
        labels, ratio=args.val_split_ratio, seed=args.val_split_seed)
    print(f"[val-select] train {len(labels)} -> opt-train {len(opt_idx)} + val {len(val_idx)} ({strat})")

    val_loader = build_loader(trainset, val_idx, args.batch_size, args.num_workers)
    db_loader  = build_loader(trainset, opt_idx, args.batch_size, args.num_workers)

    d_in = int(np.asarray(trainset.visual_global).shape[1])
    hidden = [] if cfg.get("encoder_layers", "none") in ("version2", "layer=1", "none") else [4096]

    rows = []
    for ck in sorted(glob.glob(os.path.join(args.params_dir, "epoch_*.pth"))):
        ep = int(re.search(r"epoch_(\d+)\.pth", os.path.basename(ck)).group(1))
        model = BackboneWithEncoder(d_in=d_in, bit=bit, hidden_nodes=hidden,
                                    batch_norm=bool(cfg.get("batch_norm", False)))
        state = torch.load(ck, map_location="cpu")
        model.encoder_layers.load_state_dict(state["encoder_layers"])
        model = model.to(args.device)

        _, vq_bin, vq_lbl = _extract_codes(model, val_loader, args.device)
        _, db_bin, db_lbl = _extract_codes(model, db_loader,  args.device)
        res = evaluate_retrieval_model({"B": vq_bin}, {"B": db_bin}, vq_lbl, db_lbl,
                                       precision_at_k_list=(1, 10), map_at_r=map_r)
        # test eval already computed during training -- never recomputed here
        tj = os.path.join(args.result_dir, f"eval_epoch_{ep:03d}.json")
        test = json.load(open(tj)) if os.path.exists(tj) else {}
        rows.append({
            "epoch": ep,
            "val_mAP_at_R": float(res.get("mAP_at_R", res["mAP"])),
            "val_mAP": float(res["mAP"]),
            "test_mAP_at_R": float(test.get("mAP_at_R")) if test.get("mAP_at_R") is not None else None,
            "test_mAP": float(test.get("mAP")) if test.get("mAP") is not None else None,
        })
        print(f"[val-select] epoch {ep:3d}: val mAP@R={rows[-1]['val_mAP_at_R']:.4f} "
              f"| test mAP@R={rows[-1]['test_mAP_at_R']}")
        del model
        torch.cuda.empty_cache()

    if not rows:
        raise SystemExit(f"[val-select] no epoch_*.pth under {args.params_dir}")

    best = max(rows, key=lambda r: r["val_mAP_at_R"])
    final = max(rows, key=lambda r: r["epoch"])
    leaky = max((r for r in rows if r["test_mAP_at_R"] is not None),
                key=lambda r: r["test_mAP_at_R"], default=None)
    out = {
        "trial": cfg["trial_name"], "dataset": dataset, "bit": bit,
        "map_at_r_cutoff": map_r,
        "val_split": {"ratio": args.val_split_ratio, "seed": args.val_split_seed,
                      "strategy": strat, "n_val": int(len(val_idx)),
                      "n_opt_train": int(len(opt_idx))},
        "selected_epoch_by_val": best["epoch"],
        "test_mAP_at_R_val_selected": best["test_mAP_at_R"],
        "test_mAP_at_R_final_epoch": final["test_mAP_at_R"],
        "test_mAP_at_R_test_selected_LEAKY": leaky["test_mAP_at_R"] if leaky else None,
        "selection_bias_leaky_minus_val": (
            (leaky["test_mAP_at_R"] - best["test_mAP_at_R"])
            if leaky and best["test_mAP_at_R"] is not None else None),
        "per_epoch": rows,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\n[val-select] {cfg['trial_name']}: val picks epoch {best['epoch']} "
          f"-> test mAP@R {best['test_mAP_at_R']}  "
          f"(final-epoch {final['test_mAP_at_R']}, test-selected/LEAKY "
          f"{leaky['test_mAP_at_R'] if leaky else None})")
    print(f"[val-select] wrote {args.out}")


if __name__ == "__main__":
    main()
