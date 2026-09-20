"""arch-exp-3 diagnostic: is the DEPLOYMENT routing plan informative at all?

M1 is measured through the deployment forward, where the router's anchors are
codebook means, not captions. If those anchors are nearly parallel the plan is
near-uniform and no training-side mechanism can show up in M1. Reports, on the
same held-out images, the anchor geometry and the plan geometry in both modes.
"""
import argparse, json, os, sys
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

def plan_stats(P):                                  # P [B, N, M] non-negative
    p = P / P.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    ent = -(p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).sum(-1)
    M = P.shape[-1]
    cols = P.reshape(-1, M)
    cols = cols - cols.mean(0, keepdim=True)
    cn = torch.nn.functional.normalize(cols, dim=0)
    corr = (cn.t() @ cn)
    off = corr[~torch.eye(M, dtype=torch.bool, device=corr.device)]
    return dict(max_prob=float(p.max(-1).values.mean()),
                effective_k=float(ent.exp().mean()),
                spread_of_patch_max=float(p.max(-1).values.std()),
                plan_column_corr_mean=float(off.mean()))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--n", type=int, default=256)
    a = ap.parse_args()
    os.environ.setdefault("GDNA_NUM_SEMANTIC_PARTS", "5")
    sys.path.insert(0, "scripts")
    from slot_role_probe import parse_args_txt
    from model_siglip2 import SigLIP2SemanticOTModel
    from dataloaders import load_dataset
    from dna_utils.runtime_state import apply_inference_epoch

    args = parse_args_txt(os.path.join(a.result_dir, "args.txt")); args.lambda_mec = 0.0
    ckpt = os.path.join(a.result_dir, "model_state_dict.pth")
    model = SigLIP2SemanticOTModel(args).to(a.device).eval()
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False), strict=False)
    apply_inference_epoch(model, ckpt, args)

    cache = args.siglip2_feature_cache_dir
    opt = set(int(r) for r in np.load(os.path.join(cache, "opt_train_rows.npy")))
    allr = set(int(r) for r in np.load(os.path.join(cache, "train_all_rows.npy")))
    tr, _, _ = load_dataset(getattr(args, "dataset_dir", "dataset"), args.dataset,
        setting=getattr(args, "setting", "setting1"), train_transform=None, test_transform=None,
        load_train=True, load_database=False, load_test=False, return_index=True,
        qwen_text_cache_path=getattr(args, "qwen_text_cache_path", None), siglip2_feature_cache_dir=cache)
    rows = np.asarray(tr._feat_cache_rows)
    idx = [i for i in range(len(tr)) if int(rows[i]) in (allr - opt)][:a.n]

    out = {"result_dir": a.result_dir, "n_images": len(idx)}
    # ---- anchor geometry: the codebook means the deployment router uses -----
    with torch.no_grad():
        anc = model.quantizer.get_codebook_mean_anchors(exclude_global=True).float()   # [M_local, D]
        an = torch.nn.functional.normalize(anc, dim=-1)
        C = an @ an.t(); M = C.shape[0]
        off = C[~torch.eye(M, dtype=torch.bool, device=C.device)]
        out["deploy_anchor_pairwise_cos"] = dict(mean=float(off.mean()), min=float(off.min()), max=float(off.max()))
        out["deploy_anchor_norm_mean"] = float(anc.norm(dim=-1).mean())
        tp = getattr(model, "text_prototype_ema", None)
        ini = getattr(model, "_text_prototype_initialized", None)
        out["text_prototype_initialized"] = bool(ini.item()) if ini is not None else None
        if tp is not None and out["text_prototype_initialized"]:
            t = torch.nn.functional.normalize(tp.float()[1:], dim=-1)
            Ct = t @ t.t(); m2 = Ct.shape[0]
            o2 = Ct[~torch.eye(m2, dtype=torch.bool, device=Ct.device)]
            out["text_prototype_pairwise_cos"] = dict(mean=float(o2.mean()), min=float(o2.min()), max=float(o2.max()))

        # ---- plan geometry in both modes -----------------------------------
        for mode in ("deploy", "train_text"):
            acc = []
            for s0 in range(0, len(idx), 64):
                smp = [tr[i] for i in idx[s0:s0+64]]
                vt = torch.stack([s["cached_visual_tokens_raw"] for s in smp]).to(a.device)
                vg = torch.stack([s["cached_visual_global"] for s in smp]).to(a.device)
                kw = dict(pixel_values=None, part_input_ids=None, part_attention_mask=None,
                          return_routing=True, cached_visual_tokens_raw=vt, cached_visual_global=vg)
                if mode == "deploy":
                    kw.update(cached_text_part_raw=None, cached_has_text=None)
                else:
                    tp_raw = torch.stack([s["cached_text_part_raw"] for s in smp]).to(a.device)
                    ht = torch.ones(len(smp), dtype=torch.bool, device=a.device)
                    kw.update(cached_text_part_raw=tp_raw, cached_has_text=ht)
                    if "cached_text_tokens" in smp[0]:
                        kw["cached_text_tokens"] = torch.stack([s["cached_text_tokens"] for s in smp]).to(a.device)
                        kw["cached_text_token_mask"] = torch.stack([s["cached_text_token_mask"] for s in smp]).to(a.device)
                try:
                    o = model(**kw)
                except Exception as e:
                    out[f"plan_{mode}"] = f"UNAVAILABLE: {type(e).__name__}: {e}"; acc = None; break
                rm = o.get("local_routing_matrix")
                if rm is None: out[f"plan_{mode}"] = "UNAVAILABLE: no local_routing_matrix"; acc = None; break
                acc.append(rm.float().cpu())
            if acc:
                out[f"plan_{mode}"] = plan_stats(torch.cat(acc))
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))

main()
