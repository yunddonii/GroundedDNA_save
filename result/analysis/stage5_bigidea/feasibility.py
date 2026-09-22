"""Stage 5: no-training feasibility of two larger redesigns, on the Flickr25K train split.

The dominant loss (cibhash_ntxent, lambda 1.0, per_codebook on pre-VQ slot tokens) asks EVERY slot to
identify the same image on its own; that rewards each slot for carrying the same image-level
identity. Two redesigns change what a local slot is asked for:

  (A) axis-neighbour targets: slot m's positives are images whose axis-m CAPTIONS are similar,
      instead of the two augmentations of one image.  Useful only if axis-m neighbourhoods differ
      from axis-m' neighbourhoods and from image-level neighbourhoods.  -> section 1
  (B) named concept codebook: each local codebook is the K=64 k-means of axis-m caption embeddings,
      so every codeword (and, with K = 4^3, every codon) has a caption-defined name; the image picks
      a concept without text.  -> sections 2 and 3

Rows: opt (4,500) = dictionary / training side, val (500) = held out.  Readability uses the same
AP of the image's own axis-m distinctive caption words and the same posterior scoring as
stage2_zscr.  Retrieval is a PROXY (query = val, DB = opt, relevant = shares a label, full-ranking
AP, random tie-break) computed identically for every code; it is not the paper's mAP@R.
Nothing is trained except a linear probe on frozen CLIP image features.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
os.environ.setdefault("GDNA_NUM_SEMANTIC_PARTS", "5")


def knn_sets(x, k):
    x = F.normalize(torch.as_tensor(x, dtype=torch.float32), dim=-1)
    s = x @ x.T
    s.fill_diagonal_(-2.0)
    return s.topk(k, dim=1).indices.numpy()


def overlap(a, b):
    return float(np.mean([len(set(a[i]) & set(b[i])) / a.shape[1] for i in range(len(a))]))


def retrieval_ap(score, qlab, dlab, seed=0):
    """Mean full-ranking AP; relevant = shares at least one label; ties broken at random."""
    rng = np.random.default_rng(seed)
    rel = (qlab @ dlab.T) > 0
    aps = []
    for i in range(len(score)):
        order = np.lexsort((rng.random(score.shape[1]), -score[i]))
        r = rel[i, order]
        if r.sum() == 0:
            continue
        hits = np.cumsum(r)
        aps.append(float((hits[r] / (np.nonzero(r)[0] + 1)).mean()))
    return float(np.mean(aps))


def slot_match(qc, dc):
    return (qc[:, None, :] == dc[None, :, :]).sum(-1).astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--k_nn", type=int, default=20)
    ap.add_argument("--K", type=int, default=64)
    a = ap.parse_args()
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LogisticRegression
    from slot_role_probe import parse_args_txt, distinctive_vocab, multi_hot, AXES
    from heldout_codon_decoding import label_ranking_ap
    from dataloaders import load_dataset
    from zscr_like import read_scores, deployed_codes

    base_dirs = {s: glob.glob(os.path.join(REPO, f"result/260920+flickr25k_setting1_base_s{s}+*"))[0]
                 for s in (42, 43, 44)}
    args = parse_args_txt(os.path.join(base_dirs[42], "args.txt"))
    cache = args.siglip2_feature_cache_dir
    opt = set(int(r) for r in np.load(os.path.join(cache, "opt_train_rows.npy")))
    tr, _, _ = load_dataset(getattr(args, "dataset_dir", "dataset"), args.dataset, setting="setting1",
                            train_transform=None, test_transform=None, load_train=True,
                            load_database=False, load_test=False, return_index=True,
                            qwen_text_cache_path=args.qwen_text_cache_path,
                            siglip2_feature_cache_dir=cache)
    rows = np.asarray(tr._feat_cache_rows)
    idx_opt = [i for i in range(len(tr)) if int(rows[i]) in opt]
    idx_val = [i for i in range(len(tr)) if int(rows[i]) not in opt]
    samp = lambda idx, key: np.stack([np.asarray(tr[i][key], dtype=np.float32) for i in idx])  # noqa: E731
    vg_o, vg_v = samp(idx_opt, "cached_visual_global"), samp(idx_val, "cached_visual_global")
    tx_o, tx_v = samp(idx_opt, "cached_text_part_raw"), samp(idx_val, "cached_text_part_raw")
    lab_o, lab_v = samp(idx_opt, "label"), samp(idx_val, "label")

    caps = {}
    for line in open(args.qwen_text_cache_path):
        d = json.loads(line)
        caps[str(d["image_id"])] = {k: str(d["codebook_texts"].get(k, "") or "") for k in AXES}
    name = lambda i: "images/" + os.path.basename(str(tr[i]["image_path"]))  # noqa: E731
    id_o, id_v = [name(i) for i in idx_opt], [name(i) for i in idx_val]
    vocab = distinctive_vocab(caps, id_o)
    T_o = {ax: multi_hot(caps, id_o, ax, vocab[ax]) for ax in AXES}
    T_v = {ax: multi_hot(caps, id_v, ax, vocab[ax]) for ax in AXES}
    rep = {"n_opt": len(idx_opt), "n_val": len(idx_val), "K": a.K, "k_nn": a.k_nn}

    # ---- 1. are axis-m caption neighbourhoods different from each other and from the image's? ----
    mu = tx_o.mean(0, keepdims=True)                                   # per-axis mean (opt)
    cen_o, cen_v = tx_o - mu, tx_v - mu
    nn_img = knn_sets(vg_o, a.k_nn)
    nn_ax = [knn_sets(cen_o[:, m], a.k_nn) for m in range(5)]
    names5 = ("global",) + AXES
    rep["knn_overlap"] = {
        "random": a.k_nn / (len(idx_opt) - 1),
        "axis_vs_axis": {f"{names5[p]}|{names5[q]}": overlap(nn_ax[p], nn_ax[q])
                         for p in range(5) for q in range(p + 1, 5)},
        "axis_vs_image": {names5[m]: overlap(nn_ax[m], nn_img) for m in range(5)},
    }

    # ---- 2. named concept codebook: readability of the concept an image picks without text ----
    km, lp, conc_o, conc_v, conc_v_oracle = {}, {}, np.zeros((len(idx_opt), 5), int), \
        np.zeros((len(idx_val), 5), int), np.zeros((len(idx_val), 5), int)
    for m in range(5):
        km[m] = KMeans(n_clusters=a.K, n_init=4, random_state=0).fit(cen_o[:, m])
        conc_o[:, m] = km[m].labels_
        conc_v_oracle[:, m] = km[m].predict(cen_v[:, m])               # uses the caption: upper bound
        lp[m] = LogisticRegression(max_iter=2000, C=1.0).fit(F.normalize(torch.tensor(vg_o), dim=-1).numpy(),
                                                              conc_o[:, m])
        conc_v[:, m] = lp[m].predict(F.normalize(torch.tensor(vg_v), dim=-1).numpy())
    read = {}
    for m in range(1, 5):
        ax = AXES[m - 1]
        To, Tv = T_o[ax], T_v[ax]
        prior, n_prior = To.sum(0).astype(float), float(len(To))
        counts = np.stack([To[conc_o[:, m] == c].sum(0) for c in range(a.K)]).astype(float)
        support = np.bincount(conc_o[:, m], minlength=a.K).astype(float)
        S = read_scores(counts, support, prior, n_prior)               # name of every concept
        read[ax] = {
            "concept_oracle_assignment": float(np.nanmean(label_ranking_ap(S[conc_v_oracle[:, m]], Tv))),
            "concept_text_free": float(np.nanmean(label_ranking_ap(S[conc_v[:, m]], Tv))),
            "probe_accuracy": float((conc_v[:, m] == conc_v_oracle[:, m]).mean()),
            "prior_only": float(np.nanmean(label_ranking_ap(
                np.tile((prior + 1.0) / (n_prior + 2.0), (len(Tv), 1)), Tv))),
        }
    rep["concept_readability"] = read
    rep["concept_readability_mean"] = {k: float(np.mean([read[ax][k] for ax in AXES]))
                                       for k in read[AXES[0]]}

    # ---- 3. retrieval proxy, identical for every code -------------------------------------------
    rel_o, rel_v = lab_o.astype(float), lab_v.astype(float)
    ret = {"clip_image_cosine": retrieval_ap(
        F.normalize(torch.tensor(vg_v), dim=-1).numpy() @ F.normalize(torch.tensor(vg_o), dim=-1).numpy().T,
        rel_v, rel_o),
        "concept_code_5slot_match": retrieval_ap(slot_match(conc_v, conc_o), rel_v, rel_o),
        "concept_code_slot_alone": [retrieval_ap(slot_match(conc_v[:, [m]], conc_o[:, [m]]), rel_v, rel_o)
                                    for m in range(5)]}
    models = {}
    for s, d in base_dirs.items():
        c_o, b_o = deployed_codes(d, tr, idx_opt, a.device)
        c_v, b_v = deployed_codes(d, tr, idx_val, a.device)
        models[s] = {"codon_5slot_match": retrieval_ap(slot_match(c_v, c_o), rel_v, rel_o),
                     "base_hamming_15": retrieval_ap(slot_match(b_v, b_o), rel_v, rel_o),
                     "codon_slot_alone": [retrieval_ap(slot_match(c_v[:, [m]], c_o[:, [m]]), rel_v, rel_o)
                                          for m in range(5)]}
    ret["current_model"] = models
    zs = [json.load(open(os.path.join(REPO, f"result/analysis/stage4_arms/zscr/base_s{s}.json")))
          ["mean_over_local_slots"] for s in (42, 43, 44)]
    rep["current_model_readability_mean"] = {k: float(np.mean([z[k] for z in zs]))
                                             for k in ("supervised_ceiling", "text_path", "clip_only",
                                                       "prior_only")}
    rep["retrieval_proxy"] = ret
    json.dump(rep, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items()}, indent=1))


if __name__ == "__main__":
    main()
