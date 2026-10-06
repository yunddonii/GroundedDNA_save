"""D5 -- audit of the Stage-3 target X on an existing checkpoint, before any training.

X: for image i and local slot m, the target is the codeword distribution held by i's axis-m TEXT
neighbours (top-k by per-axis-centred EOS caption cosine among the training rows), each neighbour's
slot-m token re-assigned against the current codebook. Here the tokens are the training-mode
(caption-routed) pre-quantisation slot tokens of the opt-train rows, i.e. what the memory would hold.

Reports per axis: top-vote share (mean / quartiles), P(vote mode == own codeword) and P(!=), the share
of rows whose vote mode is a codeword the row itself does not use, the entropy of the vote-mode
distribution over codewords vs the entropy of current usage (does the target narrow the codebook?),
P(cosine argmax == Euclidean argmin) on the token, and the k-NN set overlap between axes. Also the
same numbers for a VISUAL neighbour graph (slot-m token k-NN) as the X-vis control.
"""
import argparse, json, math, os, sys
import numpy as np, torch, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = HERE
for _ in range(3):
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from a3_v2 import AXES  # noqa: E402
from d4d6_train_vs_deploy import forward  # noqa: E402


def knn(x, k):
    x = F.normalize(x, dim=-1); sim = x @ x.t(); sim.fill_diagonal_(-2.0)
    return sim.topk(k, dim=1).indices


def entropy(counts):
    p = counts / counts.sum(); p = p[p > 0]
    return float(-(p * p.log()).sum())


def audit(cb, nbr, K, N):
    """cb [N] long codeword per row; nbr [N,k] neighbour rows."""
    votes = torch.zeros(N, K)
    votes.scatter_add_(1, cb[nbr], torch.ones_like(nbr, dtype=torch.float))
    top = votes.max(1); share = top.values / nbr.shape[1]; mode = top.indices
    usage = torch.bincount(cb, minlength=K).float(); tgt = torch.bincount(mode, minlength=K).float()
    return {"top_vote_share_mean": round(float(share.mean()), 4),
            "top_vote_share_q25_q50_q75": [round(float(torch.quantile(share, q)), 3) for q in (.25, .5, .75)],
            "p_mode_equals_own": round(float((mode == cb).float().mean()), 4),
            "distinct_codewords_used": int((usage > 0).sum()), "distinct_codewords_as_mode": int((tgt > 0).sum()),
            "entropy_usage_nats": round(entropy(usage), 3), "entropy_mode_nats": round(entropy(tgt), 3), "log_K": round(math.log(K), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=10); ap.add_argument("--max_rows", type=int, default=0)
    a = ap.parse_args()
    os.environ.setdefault("GDNA_NUM_SEMANTIC_PARTS", "5")
    import model_siglip2 as MS
    from dataloaders import load_dataset
    from dna_utils.runtime_state import apply_inference_epoch
    from slot_role_probe import parse_args_txt
    args = parse_args_txt(os.path.join(a.result_dir, "args.txt")); args.lambda_mec = 0.0; args.device = "cpu"
    ckpt = os.path.join(a.result_dir, "model_state_dict.pth")
    model = MS.SigLIP2SemanticOTModel(args).eval()
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False), strict=False)
    apply_inference_epoch(model, ckpt, args)
    cache = args.siglip2_feature_cache_dir
    opt = set(int(r) for r in np.load(os.path.join(cache, "opt_train_rows.npy")))
    tr, _, _ = load_dataset(getattr(args, "dataset_dir", "dataset"), args.dataset, setting=getattr(args, "setting", "setting1"),
                            train_transform=None, test_transform=None, load_train=True, load_database=False, load_test=False,
                            return_index=True, qwen_text_cache_path=getattr(args, "qwen_text_cache_path", None), siglip2_feature_cache_dir=cache)
    rows = np.asarray(tr._feat_cache_rows)
    idx = [i for i in range(len(tr)) if int(rows[i]) in opt]
    if a.max_rows:
        idx = idx[:a.max_rows]
    N, M = len(idx), 4
    tp = np.load(os.path.join(cache, "text_part.f16.npy"), mmap_mode="r")
    T = torch.tensor(np.stack([np.asarray(tp[int(rows[i])][1:5], dtype=np.float32) for i in idx])); T = T - T.mean(0, keepdim=True)
    A = forward(model, tr, idx, with_text=True, training=True, prune=bool(model.bidirectional_token_prune))
    z = A["z"]                                                             # [N,4,D] caption-routed tokens
    with torch.no_grad():
        codebooks = model.quantizer.codebooks.detach().float()             # [5,K,D]
    K = codebooks.shape[1]
    res = {"result_dir": a.result_dir, "dataset": args.dataset, "rows": N, "k": a.k, "K": int(K), "routing_mode": A["mode"], "axes": {}}
    nbr_txt = {m: knn(T[:, m], a.k) for m in range(M)}
    nbr_vis = {m: knn(z[:, m], a.k) for m in range(M)}
    for m in range(M):
        C = codebooks[m + 1]
        d = torch.cdist(z[:, m], C); eu = d.argmin(1)
        cs = (F.normalize(z[:, m], dim=-1) @ F.normalize(C, dim=-1).t()).argmax(1)
        ax = AXES[m + 1]
        res["axes"][ax] = {"p_cos_argmax_equals_euclid_argmin": round(float((eu == cs).float().mean()), 4),
                           "p_euclid_argmin_equals_model_codeword": round(float((eu == A["cb"][:, m]).float().mean()), 4),
                           "text_graph": audit(eu, nbr_txt[m], K, N), "visual_graph_control": audit(eu, nbr_vis[m], K, N),
                           "knn_overlap_text_vs_other_axes": {AXES[k + 1]: round(float(np.mean([len(set(nbr_txt[m][i].tolist()) & set(nbr_txt[k][i].tolist())) / a.k for i in range(N)])), 4) for k in range(M) if k != m},
                           "knn_overlap_text_vs_visual_same_slot": round(float(np.mean([len(set(nbr_txt[m][i].tolist()) & set(nbr_vis[m][i].tolist())) / a.k for i in range(N)])), 4)}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("dataset", "rows", "k", "K", "routing_mode")}))
    for ax, v in res["axes"].items():
        t, c = v["text_graph"], v["visual_graph_control"]
        print(f"{ax:24s} cos=euclid {v['p_cos_argmax_equals_euclid_argmin']:.3f} | text: top-share {t['top_vote_share_mean']:.3f} q {t['top_vote_share_q25_q50_q75']} mode=own {t['p_mode_equals_own']:.3f} "
              f"modes {t['distinct_codewords_as_mode']}/{t['distinct_codewords_used']} H {t['entropy_mode_nats']:.2f}/{t['entropy_usage_nats']:.2f} | visual ctrl: top-share {c['top_vote_share_mean']:.3f} mode=own {c['p_mode_equals_own']:.3f} | nbr overlap other-axes {np.mean(list(v['knn_overlap_text_vs_other_axes'].values())):.3f} text-vs-visual {v['knn_overlap_text_vs_visual_same_slot']:.3f}")


if __name__ == "__main__":
    main()
