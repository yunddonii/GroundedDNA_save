"""Stage 5b: can a Hamming-aware codon layout recover the concept code's retrieval cost?

Each concept (k-means centroid of axis-m captions) gets 3 base digits from the quartiles of the
centroids' top-3 principal components, so nearby concepts share bases; retrieval then scores base
agreement over all 15 bases, as the trained model's native base-Hamming does.  Same rows, split,
k-means seed and linear probe as feasibility.py.  Also prints the trained model's per-axis readings
next to the concept code's, from the stage-4 ZSCR JSONs of the same base runs."""
import json, os, sys
import numpy as np
import torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import feasibility as fz  # noqa: E402


def pca_digits(centroids):
    c = centroids - centroids.mean(0)
    u, s, vt = np.linalg.svd(c, full_matrices=False)
    proj = c @ vt[:3].T                                           # [K, 3]
    return np.stack([np.searchsorted(np.quantile(proj[:, j], [.25, .5, .75]), proj[:, j])
                     for j in range(3)], 1)                       # [K, 3] in {0..3}


def main():
    import argparse
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LogisticRegression
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    from slot_role_probe import parse_args_txt, AXES
    from dataloaders import load_dataset
    import glob
    d = glob.glob(os.path.join(fz.REPO, "result/260920+flickr25k_setting1_base_s42+*"))[0]
    args = parse_args_txt(os.path.join(d, "args.txt")); cache = args.siglip2_feature_cache_dir
    opt = set(int(r) for r in np.load(os.path.join(cache, "opt_train_rows.npy")))
    tr, _, _ = load_dataset(getattr(args, "dataset_dir", "dataset"), args.dataset, setting="setting1",
                            train_transform=None, test_transform=None, load_train=True, load_database=False,
                            load_test=False, return_index=True, qwen_text_cache_path=args.qwen_text_cache_path,
                            siglip2_feature_cache_dir=cache)
    rows = np.asarray(tr._feat_cache_rows)
    io = [i for i in range(len(tr)) if int(rows[i]) in opt]; iv = [i for i in range(len(tr)) if int(rows[i]) not in opt]
    samp = lambda idx, k: np.stack([np.asarray(tr[i][k], dtype=np.float32) for i in idx])  # noqa: E731
    vg_o, vg_v, tx_o, tx_v = samp(io, "cached_visual_global"), samp(iv, "cached_visual_global"), \
        samp(io, "cached_text_part_raw"), samp(iv, "cached_text_part_raw")
    lab_o, lab_v = samp(io, "label"), samp(iv, "label")
    cen_o = tx_o - tx_o.mean(0, keepdims=True)
    nv_o = F.normalize(torch.tensor(vg_o), dim=-1).numpy(); nv_v = F.normalize(torch.tensor(vg_v), dim=-1).numpy()
    bases_o, bases_v = [], []
    for m in range(5):
        km = KMeans(n_clusters=64, n_init=4, random_state=0).fit(cen_o[:, m])
        dig = pca_digits(km.cluster_centers_)
        pred_v = LogisticRegression(max_iter=2000, C=1.0).fit(nv_o, km.labels_).predict(nv_v)
        bases_o.append(dig[km.labels_]); bases_v.append(dig[pred_v])
    bo, bv = np.concatenate(bases_o, 1), np.concatenate(bases_v, 1)           # [n, 15]
    distinct = [len(np.unique(np.concatenate(bases_o, 1)[:, 3 * m:3 * m + 3], axis=0)) for m in range(5)]
    rep = {"concept_base_agreement_15": fz.retrieval_ap(fz.slot_match(bv, bo), lab_v, lab_o),
           "distinct_codons_per_slot_after_layout": distinct}
    zs = [json.load(open(os.path.join(fz.REPO, f"result/analysis/stage4_arms/zscr/base_s{s}.json")))["slots"]
          for s in (42, 43, 44)]
    rep["current_model_per_axis"] = {ax: {k: float(np.mean([z[ax][k] for z in zs]))
                                          for k in ("supervised_ceiling", "text_path", "prior_only")} for ax in AXES}
    json.dump(rep, open(a.out, "w"), indent=1); print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
