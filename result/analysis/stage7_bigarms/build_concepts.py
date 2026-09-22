"""Stage 7 (A): build the caption-concept file for one dataset.

Per local axis m, K=64 k-means centres of the per-axis-centred cached caption embeddings of the
TRAINING (opt) rows only -- held-out val rows are never used -- and a bijective, Hamming-aware codon
layout: each centre's top-3 principal-component coordinates are rank-scaled to [0, 4) and the 64
centres are assigned to the 64 codons {0..3}^3 by minimum total squared distance (Hungarian), so
nearby concepts receive codons that share bases.  Output npz: centers [4,64,D], axis_mean [5,D],
layout [4,64,3]; a JSON sidecar records rows, seeds, cluster sizes and the layout's quality."""
import argparse, glob, hashlib, json, os, sys
import numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
os.environ.setdefault("GDNA_NUM_SEMANTIC_PARTS", "5")


def layout_for(centres):
    from scipy.optimize import linear_sum_assignment
    c = centres - centres.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    p = c @ vt[:3].T                                                   # [64, 3]
    r = np.stack([(np.argsort(np.argsort(p[:, j])) + 0.5) / len(p) * 4.0 for j in range(3)], 1)
    codons = np.array([(a, b, d) for a in range(4) for b in range(4) for d in range(4)], float)
    cost = ((r[:, None, :] - (codons[None, :, :] + 0.5)) ** 2).sum(-1)
    rows, cols = linear_sum_assignment(cost)
    lay = np.zeros((len(centres), 3), np.int64); lay[rows] = codons[cols].astype(np.int64)
    return lay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="a finished run of this dataset (for its args.txt)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--K", type=int, default=64)
    a = ap.parse_args()
    from sklearn.cluster import KMeans
    from slot_role_probe import parse_args_txt
    from dataloaders import load_dataset
    args = parse_args_txt(os.path.join(a.run_dir, "args.txt")); cache = args.siglip2_feature_cache_dir
    opt = set(int(r) for r in np.load(os.path.join(cache, "opt_train_rows.npy")))
    tr, _, _ = load_dataset(getattr(args, "dataset_dir", "dataset"), args.dataset, setting="setting1",
                            train_transform=None, test_transform=None, load_train=True, load_database=False,
                            load_test=False, return_index=True, qwen_text_cache_path=args.qwen_text_cache_path,
                            siglip2_feature_cache_dir=cache)
    rows = np.asarray(tr._feat_cache_rows)
    io = [i for i in range(len(tr)) if int(rows[i]) in opt]
    tx = np.stack([np.asarray(tr[i]["cached_text_part_raw"], np.float32) for i in io])      # [n, 5, D]
    has = np.array([bool(tr[i]["has_text"]) for i in io])
    tx = tx[has]
    mu = tx.mean(0)                                                                         # [5, D]
    centres, layouts, sizes, quality = [], [], [], []
    for m in range(1, 5):
        km = KMeans(n_clusters=a.K, n_init=4, random_state=0).fit(tx[:, m] - mu[m])
        lay = layout_for(km.cluster_centers_)
        centres.append(km.cluster_centers_.astype(np.float32)); layouts.append(lay)
        sizes.append(np.bincount(km.labels_, minlength=a.K).tolist())
        # layout quality: correlation between centre cosine and shared bases
        cn = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)
        cos = cn @ cn.T; share = (lay[:, None, :] == lay[None, :, :]).sum(-1)
        iu = np.triu_indices(a.K, 1); quality.append(float(np.corrcoef(cos[iu], share[iu])[0, 1]))
    np.savez(a.out, centers=np.stack(centres), axis_mean=mu.astype(np.float32), layout=np.stack(layouts))
    meta = {"dataset": args.dataset, "rows": "opt_train_rows with captions", "n_rows": int(len(tx)),
            "K": a.K, "kmeans_seed": 0, "n_init": 4, "cluster_sizes": sizes,
            "layout_corr_centre_cos_vs_shared_bases": quality,
            "npz_sha256": hashlib.sha256(open(a.out, "rb").read()).hexdigest()}
    json.dump(meta, open(a.out + ".json", "w"), indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k != "cluster_sizes"}, indent=1))
    print("cluster size min/median/max per axis:", [(min(s), int(np.median(s)), max(s)) for s in sizes])


if __name__ == "__main__":
    main()
