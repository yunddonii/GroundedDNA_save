"""Re-evaluate an EXISTING checkpoint under a deploy-time flag override (Stage 3-C P-mem,
2026-10-08): validation mAP@R / mAP / dead / unique exactly as the trainer's mid-eval
(`train_siglip2._mid_train_eval` on the P0 val_query-vs-val_db split carved with the run's
own ratio and seed), plus the anchor-fidelity diagnostic on the validation rows.

Usage:
  eval_ckpt_anchor.py --result_dir <run> --out out.json [--set anchor_source=memory ...] [--baseline]
`--baseline` also evaluates the untouched model (no overrides) so the script can be checked
against the run's own log.csv (epoch-4 eval_mAP_at_R) before any override is trusted.
"""
import argparse, ast, json, os, sys
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
for _ in range(4):
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.environ.setdefault("GDNA_NUM_SEMANTIC_PARTS", "5")
# Import every project module from THIS worktree before scripts/slot_role_probe (whose import
# chain prepends the main tree to sys.path and would silently swap in the main tree's model).
import model_siglip2 as MS            # noqa: E402
import dataloaders as DL              # noqa: E402
import train_siglip2 as TS            # noqa: E402
import evaluation_siglip2 as ES       # noqa: E402
import extraction_siglip2 as XS       # noqa: E402
import val_split as VS                # noqa: E402
for _m in (MS, DL, TS, ES, XS, VS):
    assert os.path.abspath(_m.__file__).startswith(ROOT + os.sep), (_m.__name__, _m.__file__, ROOT)


def build(args, ckpt, device):
    from dna_utils.runtime_state import apply_inference_epoch
    model = MS.SigLIP2SemanticOTModel(args)
    missing, unexpected = model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False), strict=False)
    apply_inference_epoch(model, ckpt, args)
    return model.to(device).eval(), {"missing": list(missing), "unexpected": list(unexpected)}


def fidelity(model, trainset, idx, device, bs=100):
    """Mean cos(image-conditioned anchor, true caption anchor) in adapter space on `idx` rows
    (model in train mode with the quantizer frozen, as d4d6 does), plus raw-space centred cos."""
    q = model.quantizer; saved = q.update_mode; q.update_mode = "frozen_for_diag"
    model.train(); fids = []; raw_cos = []; maxw = []
    with torch.no_grad():
        for s0 in range(0, len(idx), bs):
            smp = [trainset[i] for i in idx[s0:s0 + bs]]
            o = model(pixel_values=None, part_input_ids=None, part_attention_mask=None, return_routing=True,
                      cached_visual_tokens_raw=torch.stack([s["cached_visual_tokens_raw"] for s in smp]).to(device),
                      cached_visual_global=torch.stack([s["cached_visual_global"] for s in smp]).to(device),
                      cached_text_part_raw=torch.stack([s["cached_text_part_raw"] for s in smp]).to(device),
                      cached_has_text=torch.stack([torch.as_tensor(s["has_text"]) for s in smp]).bool().to(device))
            if o.get("anchor_fidelity") is not None:
                fids.append(o["anchor_fidelity"].float().cpu())
            if o.get("anchor_memory_max_w") is not None:
                maxw.append(o["anchor_memory_max_w"].float().cpu())
    q.update_mode = saved; model.eval()
    out = {}
    if fids:
        f = torch.cat(fids); out["anchor_fidelity_adapter_mean"] = round(float(f.mean()), 4); out["anchor_fidelity_adapter_sd"] = round(float(f.std()), 4)
    if maxw:
        w = torch.cat(maxw); out["memory_max_w_median"] = round(float(w.median()), 4); out["memory_max_w_p90"] = round(float(w.quantile(0.9)), 4)
    return out


def evaluate(model, args, trainset, opt_idx, val_idx, device):
    from torch.utils.data import DataLoader, Subset
    _mid_train_eval = TS._mid_train_eval
    resolve_map_at_r = ES.resolve_map_at_r
    bs = int(getattr(args, "batch_size", 64)); nw = 0
    vq = DataLoader(Subset(trainset, val_idx.tolist()), batch_size=bs, shuffle=False, num_workers=nw, drop_last=False)
    vd = DataLoader(Subset(trainset, opt_idx.tolist()), batch_size=bs, shuffle=False, num_workers=nw, drop_last=False)
    retrieval, collapse = _mid_train_eval(model, vq, device, distance_mode=str(getattr(args, "dna_distance_mode", "base")),
                                          codebook_size=int(getattr(args, "codebook_size", 32)),
                                          map_at_r=resolve_map_at_r(getattr(args, "dataset", None)), db_loader=vd)
    keep = {k: (round(float(v), 5) if isinstance(v, (int, float, np.floating)) else v)
            for k, v in retrieval.items() if isinstance(v, (int, float, np.floating))}
    col = {k: (round(float(v), 5) if isinstance(v, (int, float, np.floating)) else v)
           for k, v in collapse.items() if isinstance(v, (int, float, np.floating))}
    return {"retrieval": keep, "collapse": col}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--baseline", action="store_true", help="also evaluate the untouched model")
    ap.add_argument("--ckpt", default="model_state_dict.pth")
    a = ap.parse_args()
    from slot_role_probe import parse_args_txt
    load_dataset = DL.load_dataset
    carve_val_indices, get_train_labels = VS.carve_val_indices, VS.get_train_labels
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_args = parse_args_txt(os.path.join(a.result_dir, "args.txt")); base_args.lambda_mec = 0.0; base_args.device = device
    ckpt = os.path.join(a.result_dir, a.ckpt)
    cache = base_args.siglip2_feature_cache_dir
    tr, _, _ = load_dataset(getattr(base_args, "dataset_dir", "dataset"), base_args.dataset, setting=getattr(base_args, "setting", "setting1"),
                            train_transform=None, test_transform=None, load_train=True, load_database=False, load_test=False,
                            return_index=True, qwen_text_cache_path=getattr(base_args, "qwen_text_cache_path", None),
                            siglip2_feature_cache_dir=cache)
    ratio = float(getattr(base_args, "val_split_ratio", 0.0) or 0.0); assert ratio > 0, "P0 stage-1 run expected"
    opt_idx, val_idx, strat = carve_val_indices(get_train_labels(tr), ratio=ratio, seed=int(getattr(base_args, "val_split_seed", 42)))
    import hashlib
    res = {"result_dir": a.result_dir, "ckpt": a.ckpt, "overrides": a.set,
           "module_provenance": {m.__name__: {"file": m.__file__, "sha256": hashlib.sha256(open(m.__file__, "rb").read()).hexdigest()} for m in (MS, TS, ES)}, "val_split": {"ratio": ratio, "n_opt": int(len(opt_idx)), "n_val": int(len(val_idx)), "strat": str(strat)},
           "device": str(device)}
    # log.csv reference (last row)
    try:
        import csv
        rows = list(csv.DictReader(open(os.path.join(a.result_dir, "log.csv"))))
        last = rows[-1]
        res["log_csv_last"] = {k: last.get(k) for k in ("epoch", "eval_mAP_at_R", "eval_mAP", "eval_dead_code_ratio_mean", "eval_unique_code_ratio")}
    except Exception as e:  # noqa: BLE001
        res["log_csv_last"] = {"error": str(e)}
    if a.baseline:
        import copy
        m0, ld = build(copy.deepcopy(base_args), ckpt, device)
        res["baseline"] = evaluate(m0, base_args, tr, opt_idx, val_idx, device); res["baseline"]["load"] = ld
        print("baseline:", json.dumps(res["baseline"]["retrieval"]), flush=True)
        del m0
    args = parse_args_txt(os.path.join(a.result_dir, "args.txt")); args.lambda_mec = 0.0; args.device = device
    for kv in a.set:
        k, v = kv.split("=", 1)
        try:
            v = ast.literal_eval(v)
        except Exception:  # noqa: BLE001
            pass
        setattr(args, k, v)
    model, ld = build(args, ckpt, device)
    res["variant"] = evaluate(model, args, tr, opt_idx, val_idx, device); res["variant"]["load"] = ld
    res["variant"]["anchor_memory_meta"] = getattr(model, "anchor_memory_meta", None)
    res["variant"].update(fidelity(model, tr, val_idx.tolist(), device))
    print("variant:", json.dumps(res["variant"]["retrieval"]), json.dumps({k: v for k, v in res["variant"].items() if k.startswith("anchor") or k.startswith("memory")}), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
