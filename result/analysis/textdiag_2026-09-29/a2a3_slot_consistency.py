"""A1/A2/A3 diagnostics of one trained run, CPU, deployment forward (no caption reaches the model).

Rows: the run's own held-out validation rows (train rows not in opt_train_rows), first --n in dataset
order -- the same population the anchor probes use. Captions are used only as labels/targets.

A1  text-side assignment audit: for each local codebook m, the text_code_kl target
    p_t = softmax(cos(adapted caption m, C_m) / tau_t); reports the share of samples whose confidence
    1 - H(p_t)/log K is below the recipe threshold (excluded from the loss), and the agreement between
    argmax p_t and the codeword the image actually received.
A2a code->own-axis matching (4-way, batch-centred) at the slot token the quantiser receives and at the
    codeword it returns (the 09-20 diagnostic).
A2b geometry: effective rank of the pre-quantisation slot tokens, quantisation fidelity cos(z, q),
    how far the empirical prototype of each used codeword sits from the codeword, codewords used.
A3  cross-image slot consistency (the property under question): for axis m, image pairs whose axis-m
    captions share content (word Jaccard >= theta; or raw CLIP caption cosine in the top 2 %) --
    P(same codeword in slot m | pair) against P(same | any pair), the lift, and the base-Hamming
    distance of the two 3-base codons; the same pairing scored on the OTHER slots as a control.
"""
import argparse, collections, itertools, json, math, os, re, sys
import numpy as np, torch, torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
for _ in range(3):
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))

AXES = ["C_global", "C_primary_object", "C_secondary_object", "C_activity_or_relation", "C_color_texture"]
STOP = set("""a an the and or of to in on at with by for from as is are was were be been being this that these those it its
their his her they them he she we you i our your into onto over under above below near beside between behind front
against along across through around while during before after up down out off no not none very more most some
any each other another such than then there here where when which who whom what how all both few many several
also just only own same so too can could may might will would shall should do does did done has have had having
one two three four five six seven eight nine ten first second third left right top bottom center centre middle
side sides background foreground image scene photo picture shot view visible shown seen appears appear appearing
positioned located situated set placed standing sitting lying stands sits lies""".split())
WORD = re.compile(r"[a-z][a-z\-']+")


def words(s):
    return set(w for w in WORD.findall(str(s).lower()) if w not in STOP and len(w) > 2)


def match_acc(q, t):
    q = F.normalize(q.float() - q.float().mean(0, keepdim=True), dim=-1)
    t = F.normalize(t.float() - t.float().mean(0, keepdim=True), dim=-1)
    S = torch.einsum("bmd,bad->bma", q, t)
    M = S.shape[1]
    tgt = torch.arange(M).expand(S.shape[0], M)
    return {"code_picks_own_axis": round((S.argmax(2) == tgt).float().mean().item(), 4),
            "axis_picks_own_code": round((S.argmax(1) == tgt).float().mean().item(), 4), "chance": round(1 / M, 4)}


def eff_rank(x):
    x = x.float() - x.float().mean(0, keepdim=True)
    ev = torch.linalg.eigvalsh(x.t() @ x / max(x.shape[0] - 1, 1)).clamp(min=0)
    p = ev / ev.sum().clamp(min=1e-12)
    return float(torch.exp(-(p * (p + 1e-12).log()).sum()))


def pair_stats(same_slot, hamming, mask):
    """same_slot [N,N] bool, hamming [N,N] float, mask [N,N] bool (upper triangle pairs)."""
    n = int(mask.sum())
    if n == 0:
        return None
    return {"pairs": n, "p_same_codeword": round(float(same_slot[mask].float().mean()), 4),
            "mean_base_hamming": round(float(hamming[mask].mean()), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--jaccard", type=float, nargs="+", default=[0.15, 0.25])
    ap.add_argument("--cos_top_frac", type=float, default=0.02)
    a = ap.parse_args()
    os.environ.setdefault("GDNA_NUM_SEMANTIC_PARTS", "5")
    # Bind the model/data modules to THIS worktree before slot_role_probe, which hard-codes the main
    # tree's path into sys.path; the imported files' digests are recorded in the output.
    import model_siglip2 as MS
    import dataloaders as DL
    import dna_utils.runtime_state as RS
    from dataloaders import load_dataset
    from dna_utils.runtime_state import apply_inference_epoch
    from slot_role_probe import parse_args_txt
    import hashlib
    provenance = {mod.__name__: {"file": mod.__file__, "sha256": hashlib.sha256(open(mod.__file__, "rb").read()).hexdigest()}
                  for mod in (MS, DL, RS)}
    args = parse_args_txt(os.path.join(a.result_dir, "args.txt"))
    args.lambda_mec = 0.0
    args.device = "cpu"                                   # the model calls self.to(args.device)
    ckpt = os.path.join(a.result_dir, "model_state_dict.pth")
    model = MS.SigLIP2SemanticOTModel(args).eval()
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False), strict=False)
    apply_inference_epoch(model, ckpt, args)
    cache = args.siglip2_feature_cache_dir
    opt = set(int(r) for r in np.load(os.path.join(cache, "opt_train_rows.npy")))
    allr = set(int(r) for r in np.load(os.path.join(cache, "train_all_rows.npy")))
    tr, _, _ = load_dataset(getattr(args, "dataset_dir", "dataset"), args.dataset,
                            setting=getattr(args, "setting", "setting1"), train_transform=None,
                            test_transform=None, load_train=True, load_database=False, load_test=False,
                            return_index=True, qwen_text_cache_path=getattr(args, "qwen_text_cache_path", None),
                            siglip2_feature_cache_dir=cache)
    rows = np.asarray(tr._feat_cache_rows)
    idx = [i for i in range(len(tr)) if int(rows[i]) in (allr - opt)][:a.n]
    cache_rows = [int(rows[i]) for i in idx]
    image_ids = json.load(open(os.path.join(cache, "image_ids.json")))
    captions = {}
    for line in open(args.qwen_text_cache_path):
        r = json.loads(line)
        captions[str(r["image_id"])] = r.get("descriptions") or r.get("codebook_texts") or {}
    caps = [captions.get(str(image_ids[cr])) for cr in cache_rows]
    matched = sum(c is not None for c in caps)

    pre, quant, txt_adapt, txt_raw, idx_cb, bases = [], [], [], [], [], []
    with torch.no_grad():
        for s0 in range(0, len(idx), 50):
            smp = [tr[i] for i in idx[s0:s0 + 50]]
            vt = torch.stack([s["cached_visual_tokens_raw"] for s in smp])
            vg = torch.stack([s["cached_visual_global"] for s in smp])
            raw = torch.stack([s["cached_text_part_raw"] for s in smp])
            o = model(pixel_values=None, part_input_ids=None, part_attention_mask=None, return_routing=True,
                      cached_visual_tokens_raw=vt, cached_visual_global=vg,
                      cached_text_part_raw=None, cached_has_text=None)          # deployment forward
            z = o.get("semantic_visual_tokens") if o.get("semantic_visual_tokens") is not None else o["quantizer_input"]
            pre.append(z[:, 1:].cpu()); quant.append(o["quantized_tokens"][:, 1:].cpu())
            idx_cb.append(o["codebook_indices"][:, 1:].cpu()); bases.append(o["base_indices_per_codebook"][:, 1:].cpu())
            txt_adapt.append(model._adapt_pooled_text_for_loss(raw)[:, 1:].cpu()); txt_raw.append(raw[:, 1:].cpu())
            codebooks = o["codebooks"].detach().cpu()
    pre, quant, ta, traw = torch.cat(pre), torch.cat(quant), torch.cat(txt_adapt), torch.cat(txt_raw)
    idx_cb, bases = torch.cat(idx_cb).long(), torch.cat(bases).long()          # [N,4], [N,4,3]
    N, M = idx_cb.shape
    K = codebooks.shape[1]

    # ---- A1: the text_code_kl target on these rows
    tau_t = float(getattr(args, "text_code_kl_tau_t", 0.07)); thr = float(getattr(args, "text_code_kl_conf_threshold", 0.2))
    a1 = {}
    for m in range(M):
        C = F.normalize(codebooks[m + 1].float(), dim=-1)
        active = o["codebook_active_mask"][m + 1].cpu().bool() if o.get("codebook_active_mask") is not None else torch.ones(K, dtype=torch.bool)
        logit = F.normalize(ta[:, m].float(), dim=-1) @ C.t() / tau_t
        logit = torch.where(active[None], logit, torch.full_like(logit, -1e9))
        p = logit.softmax(-1)
        H = -(p * (p.clamp_min(1e-8)).log()).sum(-1)
        conf = (1 - H / math.log(K)).clamp(0, 1)
        a1[AXES[m + 1]] = {"mean_confidence": round(float(conf.mean()), 4),
                           "share_below_threshold": round(float((conf <= thr).float().mean()), 4),
                           "text_argmax_equals_visual_codeword": round(float((p.argmax(-1) == idx_cb[:, m]).float().mean()), 4),
                           "text_top1_mass_mean": round(float(p.max(-1).values.mean()), 4)}

    # ---- A2
    a2 = {"pre_quantisation_slot_token": match_acc(pre, ta), "quantised_codeword": match_acc(quant, ta), "per_slot": {}}
    for m in range(M):
        z, q = pre[:, m].float(), quant[:, m].float()
        used = torch.unique(idx_cb[:, m])
        proto_cos = []
        for k in used.tolist():
            zk = z[idx_cb[:, m] == k].mean(0)
            proto_cos.append(float(F.cosine_similarity(zk, codebooks[m + 1, k].float(), dim=0)))
        a2["per_slot"][AXES[m + 1]] = {"eff_rank_z": round(eff_rank(z), 2), "cos_z_q_mean": round(float(F.cosine_similarity(z, q, dim=-1).mean()), 4),
                                       "codewords_used": int(len(used)), "K": int(K),
                                       "prototype_vs_codeword_cos_mean": round(float(np.mean(proto_cos)), 4),
                                       "usage_entropy_norm": round(float(-(torch.bincount(idx_cb[:, m], minlength=K).float() / N).clamp_min(1e-12).log().mul(torch.bincount(idx_cb[:, m], minlength=K).float() / N).sum() / math.log(K)), 4)}

    # ---- A3
    same = [(idx_cb[:, m][:, None] == idx_cb[:, m][None, :]) for m in range(M)]                 # [N,N] per slot
    ham = [(bases[:, m][:, None, :] != bases[:, m][None, :, :]).float().sum(-1) for m in range(M)]  # [N,N]
    iu = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
    a3 = {"rows": N, "captions_matched": matched, "baseline_all_pairs": {}, "rules": {}}
    for m in range(M):
        a3["baseline_all_pairs"][AXES[m + 1]] = pair_stats(same[m], ham[m], iu)
    trn = F.normalize(traw.float(), dim=-1)
    for m in range(M):
        ax = AXES[m + 1]
        W = [words((c or {}).get(ax, "")) for c in caps]
        jac = torch.zeros(N, N)
        for i in range(N):
            for j in range(i + 1, N):
                u = len(W[i] | W[j])
                jac[i, j] = len(W[i] & W[j]) / u if u else 0.0
        cos = trn[:, m] @ trn[:, m].t()
        thr_cos = torch.quantile(cos[iu], 1 - a.cos_top_frac)
        rules = {f"jaccard>={t}": (jac >= t) & iu for t in a.jaccard}
        rules[f"caption_cos_top{int(a.cos_top_frac * 100)}pct(>= {thr_cos:.3f})"] = (cos >= thr_cos) & iu
        a3["rules"][ax] = {}
        for name, mask in rules.items():
            own = pair_stats(same[m], ham[m], mask)
            if own is None:
                a3["rules"][ax][name] = None; continue
            base = a3["baseline_all_pairs"][ax]
            others = [pair_stats(same[k], ham[k], mask) for k in range(M) if k != m]
            a3["rules"][ax][name] = {**own, "lift_same_codeword": round(own["p_same_codeword"] / max(base["p_same_codeword"], 1e-9), 3),
                                     "hamming_delta_vs_all_pairs": round(own["mean_base_hamming"] - base["mean_base_hamming"], 4),
                                     "other_slots_p_same_codeword_mean": round(float(np.mean([x["p_same_codeword"] for x in others])), 4),
                                     "other_slots_lift_mean": round(float(np.mean([x["p_same_codeword"] / max(a3["baseline_all_pairs"][AXES[k + 1]]["p_same_codeword"], 1e-9)
                                                                                    for k, x in zip([k for k in range(M) if k != m], others)])), 3)}
    res = {"result_dir": a.result_dir, "dataset": args.dataset, "axis_center": getattr(args, "axis_center", None),
           "text_supervision_disabled": bool(getattr(args, "disable_text_supervision", False)),
           "model_module": MS.__file__, "module_provenance": provenance, "rows": N, "captions_matched": matched, "caption_file": args.qwen_text_cache_path,
           "A1_text_code_kl_target": {"tau_t": tau_t, "conf_threshold": thr, **a1}, "A2": a2, "A3": a3,
           "note": "deployment forward; captions only as labels/targets; exploratory checkpoints unless stated"}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("dataset", "axis_center", "text_supervision_disabled", "captions_matched", "model_module")}))
    print("A2", json.dumps(a2["pre_quantisation_slot_token"]), json.dumps(a2["quantised_codeword"]))
    for ax in AXES[1:]:
        print("A1", ax, a1[ax]); print("A3", ax, {k: (v["lift_same_codeword"], v["other_slots_lift_mean"], v["pairs"]) if v else None for k, v in a3["rules"][ax].items()})


if __name__ == "__main__":
    main()
