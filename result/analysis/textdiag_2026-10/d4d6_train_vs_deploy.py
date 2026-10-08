"""D4 + D6 on one trained run (CPU, no gradient, no buffer update).

D4  training routing vs deployment routing on the same validation images:
      forward A = training mode with the captions (caption-anchor routing, exactly as in training;
                  EMA update and dead-code revival switched off for the call),
      forward B = deployment (eval, no captions, codebook-mean anchors).
    Reports per local slot: P(same codeword A vs B), cos(z_A, z_B) of the pre-quantisation slot token,
    the routing-plan overlap (cosine of the [N,4] plan columns), and the A3 v2 lexical reading of the
    text-routed codes next to the deployed codes (same pairs) -- how much slot-specific signal is
    present at training time that the deployment lookup loses.
D6  what the legacy token pruning does to the routing anchor: forward A with pruning (recipe) vs
    forward A' with pruning off (EOS-pooled caption embedding); cos of the adapted per-slot anchors,
    the logged keep ratio, and P(same codeword A vs A').
"""
import argparse, hashlib, json, os, sys
import numpy as np, torch, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = HERE
for _ in range(3):
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from a3_v2 import AXES, KIND, elements, Reader  # noqa: E402


def forward(model, tr, idx, with_text, training, prune):
    q = model.quantizer
    saved = (q.update_mode, model.bidirectional_token_prune)
    q.update_mode = "frozen_for_diag"; model.bidirectional_token_prune = prune
    model.train(training)
    cb, z, tpt, plan, keep = [], [], [], [], []
    with torch.no_grad():
        for s0 in range(0, len(idx), 50):
            smp = [tr[i] for i in idx[s0:s0 + 50]]
            kw = dict(pixel_values=None, part_input_ids=None, part_attention_mask=None, return_routing=True,
                      cached_visual_tokens_raw=torch.stack([s["cached_visual_tokens_raw"] for s in smp]),
                      cached_visual_global=torch.stack([s["cached_visual_global"] for s in smp]))
            if with_text:
                kw.update(cached_text_part_raw=torch.stack([s["cached_text_part_raw"] for s in smp]),
                          cached_has_text=torch.stack([torch.as_tensor(s["has_text"]) for s in smp]).bool(),
                          cached_text_tokens=torch.stack([s["cached_text_tokens"] for s in smp]) if "cached_text_tokens" in smp[0] else None,
                          cached_text_token_mask=torch.stack([s["cached_text_token_mask"] for s in smp]) if "cached_text_token_mask" in smp[0] else None)
            else:
                kw.update(cached_text_part_raw=None, cached_has_text=None)
            o = model(**kw)
            cb.append(o["codebook_indices"][:, 1:].cpu())
            zz = o.get("semantic_visual_tokens") if o.get("semantic_visual_tokens") is not None else o["quantizer_input"]
            z.append(zz[:, 1:].cpu())
            plan.append(o["local_routing_matrix"].cpu())
            tpt.append(o["text_part_tokens"][:, 1:].cpu() if o.get("text_part_tokens") is not None else None)
            k = o.get("bidirectional_text_keep_ratio_per_slot")
            keep.append(k.cpu() if k is not None else None)
            mode = o["routing_mode"]
    q.update_mode, model.bidirectional_token_prune = saved
    model.eval()
    return {"cb": torch.cat(cb).long(), "z": torch.cat(z), "plan": torch.cat(plan), "mode": mode,
            "tpt": torch.cat(tpt) if tpt[0] is not None else None, "keep": torch.cat(keep) if keep[0] is not None else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--caption_file", default=None); ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--boot", type=int, default=500); ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override an args.txt field after parsing (e.g. anchor_source=memory, anchor_memory_tau=0.02); repeatable")
    ap.add_argument("--max_df", type=float, default=0.20); ap.add_argument("--min_shared_colour", type=int, default=2)
    a = ap.parse_args()
    os.environ.setdefault("GDNA_NUM_SEMANTIC_PARTS", "5")
    import model_siglip2 as MS
    from dataloaders import load_dataset
    from dna_utils.runtime_state import apply_inference_epoch
    from slot_role_probe import parse_args_txt
    args = parse_args_txt(os.path.join(a.result_dir, "args.txt"))
    import ast as _ast
    for _kv in a.set:                      # (2026-10-08, Stage 3-C) deploy-time overrides, e.g. anchor_source=memory
        _k, _v = _kv.split("=", 1)
        try:
            _v = _ast.literal_eval(_v)
        except Exception:
            pass
        setattr(args, _k, _v)
    print("overrides:", a.set, flush=True)
    args.lambda_mec = 0.0; args.device = "cpu"
    ckpt = os.path.join(a.result_dir, "model_state_dict.pth")
    model = MS.SigLIP2SemanticOTModel(args).eval()
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False), strict=False)
    apply_inference_epoch(model, ckpt, args)
    assert float(getattr(args, "adapter_dropout", 0.0)) == 0.0, "train-mode forward would be stochastic"
    cache = args.siglip2_feature_cache_dir
    opt = set(int(r) for r in np.load(os.path.join(cache, "opt_train_rows.npy")))
    allr = set(int(r) for r in np.load(os.path.join(cache, "train_all_rows.npy")))
    tr, _, _ = load_dataset(getattr(args, "dataset_dir", "dataset"), args.dataset, setting=getattr(args, "setting", "setting1"),
                            train_transform=None, test_transform=None, load_train=True, load_database=False, load_test=False,
                            return_index=True, qwen_text_cache_path=getattr(args, "qwen_text_cache_path", None), siglip2_feature_cache_dir=cache)
    rows = np.asarray(tr._feat_cache_rows)
    idx = [i for i in range(len(tr)) if int(rows[i]) in (allr - opt)]
    if a.n:
        idx = idx[:a.n]
    image_ids = json.load(open(os.path.join(cache, "image_ids.json")))
    cap_file = a.caption_file or args.qwen_text_cache_path
    captions = {}
    for line in open(cap_file):
        r = json.loads(line); captions[str(r["image_id"])] = r.get("descriptions") or r.get("codebook_texts") or {}
    caps = [captions.get(str(image_ids[int(rows[i])])) for i in idx]
    keep_i = [k for k, c in enumerate(caps) if c is not None]; idx = [idx[k] for k in keep_i]; caps = [caps[k] for k in keep_i]
    N, M = len(idx), 4

    A = forward(model, tr, idx, with_text=True, training=True, prune=bool(model.bidirectional_token_prune))
    B = forward(model, tr, idx, with_text=False, training=False, prune=bool(model.bidirectional_token_prune))
    A2 = forward(model, tr, idx, with_text=True, training=True, prune=False)
    assert A["mode"] == "text" and B["mode"] in ("codebook_mean", "memory", "predictor"), (A["mode"], B["mode"])

    res = {"result_dir": a.result_dir, "dataset": args.dataset, "rows": N, "axis_center": getattr(args, "axis_center", None),
           "use_gumbel_softmax": bool(getattr(args, "use_gumbel_softmax", False)), "prune_in_recipe": bool(model.bidirectional_token_prune),
           "caption_sha256": hashlib.sha256(open(cap_file, "rb").read()).hexdigest()}
    d4 = {}
    for m in range(M):
        ax = AXES[m + 1]
        pa, pb = A["plan"][:, :, m], B["plan"][:, :, m]
        d4[ax] = {"p_same_codeword_train_vs_deploy": round(float((A["cb"][:, m] == B["cb"][:, m]).float().mean()), 4),
                  "cos_z_train_vs_deploy": round(float(F.cosine_similarity(A["z"][:, m], B["z"][:, m], dim=-1).mean()), 4),
                  "plan_column_cos": round(float(F.cosine_similarity(pa, pb, dim=-1).mean()), 4),
                  "codewords_used_train": int(A["cb"][:, m].unique().numel()), "codewords_used_deploy": int(B["cb"][:, m].unique().numel())}
    res["D4_pairwise"] = d4
    # slot-token similarity across slots (is z_m ~ z_m' ?) in both modes
    res["D4_cross_slot_cos_mean"] = {name: round(float(np.mean([F.cosine_similarity(X["z"][:, i], X["z"][:, j], dim=-1).mean().item()
                                                              for i in range(M) for j in range(i + 1, M)])), 4) for name, X in (("train", A), ("deploy", B))}
    # A3 v2 lexical reading of the text-routed codes vs the deployed codes
    E = {AXES[m + 1]: [elements((c or {}).get(AXES[m + 1], ""), KIND[AXES[m + 1]]) for c in caps] for m in range(M)}
    share = {}
    for m in range(M):
        ax = AXES[m + 1]; cnt = {}
        for s in E[ax]:
            for w in s:
                cnt[w] = cnt.get(w, 0) + 1
        ok = set(w for w, c in cnt.items() if c / N <= a.max_df); El = [s & ok for s in E[ax]]
        need = a.min_shared_colour if KIND[ax] == "colour" else 1
        S = torch.zeros(N, N, dtype=torch.bool)
        for i in range(N):
            if len(El[i]) < need:
                continue
            for j in range(i + 1, N):
                if len(El[i] & El[j]) >= need:
                    S[i, j] = True
        share[ax] = S
    iu = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
    rules = {AXES[m + 1]: {"lexical": share[AXES[m + 1]] & iu} for m in range(M)}
    reader = Reader(N, a.boot, a.seed)
    for name, X in (("train_routed", A), ("deployed", B), ("train_routed_no_prune", A2)):
        same = [(X["cb"][:, m][:, None] == X["cb"][:, m][None, :]) for m in range(M)]
        res[f"A3_codeword_lexical_{name}"] = reader.reading(same, rules, "lexical")
    # D6
    d6 = {"keep_ratio_per_slot_mean": [round(float(v), 4) for v in A["keep"].mean(0)] if A["keep"] is not None else None}
    if A["tpt"] is not None and A2["tpt"] is not None:
        d6["cos_adapted_anchor_pruned_vs_eos_per_slot"] = [round(float(F.cosine_similarity(A["tpt"][:, m], A2["tpt"][:, m], dim=-1).mean()), 4) for m in range(M)]
        d6["cos_adapted_anchor_pruned_vs_eos_min"] = round(float(F.cosine_similarity(A["tpt"], A2["tpt"], dim=-1).min()), 4)
    d6["p_same_codeword_pruned_vs_eos_per_slot"] = [round(float((A["cb"][:, m] == A2["cb"][:, m]).float().mean()), 4) for m in range(M)]
    d6["plan_column_cos_pruned_vs_eos"] = round(float(F.cosine_similarity(A["plan"], A2["plan"], dim=1).mean()), 4)
    res["D6"] = d6
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("dataset", "rows", "axis_center", "use_gumbel_softmax", "prune_in_recipe")}))
    print("D4", json.dumps(d4)); print("D4 cross-slot cos", res["D4_cross_slot_cos_mean"])
    for name in ("train_routed", "deployed", "train_routed_no_prune"):
        print("A3 lexical", name, res[f"A3_codeword_lexical_{name}"]["S"])
    print("D6", json.dumps(d6))


if __name__ == "__main__":
    main()
