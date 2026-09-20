"""arch-exp-3 diagnostic: do the router's text anchors carry per-IMAGE content,
or has the per-slot adapter turned each axis into a constant direction?

If anchor m is (nearly) the same vector for every image, the transport cost is
image-independent up to the patch features, so the partition it induces cannot
be about *this image's* axis m, and no slot can own an axis. Compares the raw
cached axis embeddings with the adapted anchors the router actually sees.
"""
import argparse, json, os, sys
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "scripts")

def stats(t):                                   # t [B, M, D]
    out = {}
    tn = F.normalize(t.float(), dim=-1)
    mu = t.float().mean(0)                                    # [M, D]
    out["concentration"] = [round(float(mu[m].norm() / t.float()[:, m].norm(dim=-1).mean()), 4)
                            for m in range(t.shape[1])]
    cos_same = []
    for m in range(t.shape[1]):
        G = tn[:, m] @ tn[:, m].t()
        n = G.shape[0]
        cos_same.append(round(float((G.sum() - G.diag().sum()) / (n * (n - 1))), 4))
    out["mean_cos_between_images_same_axis"] = cos_same
    mun = F.normalize(mu, dim=-1); C = mun @ mun.t(); M = C.shape[0]
    off = C[~torch.eye(M, dtype=torch.bool)]
    out["axis_mean_pairwise_cos"] = dict(mean=round(float(off.mean()), 4),
                                         min=round(float(off.min()), 4),
                                         max=round(float(off.max()), 4))
    # per-image deviation kept after removing the axis mean
    dev = t.float() - mu.unsqueeze(0)
    out["residual_energy_fraction"] = [
        round(float((dev[:, m].norm(dim=-1) ** 2).mean() / (t.float()[:, m].norm(dim=-1) ** 2).mean()), 4)
        for m in range(t.shape[1])]
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--n", type=int, default=256)
    a = ap.parse_args()
    os.environ.setdefault("GDNA_NUM_SEMANTIC_PARTS", "5")
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
    raw = torch.stack([tr[i]["cached_text_part_raw"] for i in idx]).to(a.device)     # [B, 5, D_proj]
    with torch.no_grad():
        adapted = model._adapt_pooled_text_for_loss(raw)                              # [B, 5, D]
    res = {"result_dir": a.result_dir, "n_images": len(idx),
           "note": "column 0 is the global slot; axes are columns 1..4",
           "raw_cached_axis_embeddings": stats(raw.cpu()),
           "adapted_router_anchors": stats(adapted.cpu())}
    json.dump(res, open(a.out, "w"), indent=1); print(json.dumps(res, indent=1))
main()
