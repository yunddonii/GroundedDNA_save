"""Priority 1: calibrate M1 against codes whose role structure is known.

Every verdict of arch-exp-2 and arch-exp-3 rests on M1, and nothing has ever
established what M1 reads when a role IS present. This builds synthetic codes
with a dialled-in amount of role structure and runs the SAME M1 computation on
them, plus the vector-based endpoint the two programs disagreed on.

Slot m's code for image i is the k-means label of

    e_m(alpha) = normalise( (1 - alpha) * global_caption_emb_i
                          + alpha       * axis_m_caption_emb_i )

so alpha = 0 gives four slots quantising the same shared vector (no role, the
null) and alpha = 1 gives slot m quantising axis m alone (a perfect role). No
model and no training are involved: the cached caption embeddings, the probe's
own vocabulary, split and decoder are used unchanged.
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "scripts")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", required=True, help="only for args.txt: cache paths and K")
    ap.add_argument("--out", required=True)
    ap.add_argument("--alphas", default="0.0,0.1,0.25,0.5,0.75,1.0")
    ap.add_argument("--kmeans_seed", type=int, default=0)
    a = ap.parse_args()
    os.environ.setdefault("GDNA_NUM_SEMANTIC_PARTS", "5")
    from slot_role_probe import parse_args_txt, distinctive_vocab, multi_hot, AXES
    from heldout_codon_decoding import decode_slot, DEFAULT_ALPHA, DEFAULT_MIN_SUPPORT
    from dataloaders import load_dataset
    from sklearn.cluster import KMeans
    import torch

    args = parse_args_txt(os.path.join(a.result_dir, "args.txt"))
    K = int(getattr(args, "codebook_size", 128))
    cache = args.siglip2_feature_cache_dir
    opt = set(int(r) for r in np.load(os.path.join(cache, "opt_train_rows.npy")))
    allr = set(int(r) for r in np.load(os.path.join(cache, "train_all_rows.npy")))
    tr, _, _ = load_dataset(getattr(args, "dataset_dir", "dataset"), args.dataset,
        setting=getattr(args, "setting", "setting1"), train_transform=None, test_transform=None,
        load_train=True, load_database=False, load_test=False, return_index=True,
        qwen_text_cache_path=getattr(args, "qwen_text_cache_path", None), siglip2_feature_cache_dir=cache)
    rows = np.asarray(tr._feat_cache_rows)
    idx_opt = [i for i in range(len(tr)) if int(rows[i]) in opt]
    idx_val = [i for i in range(len(tr)) if int(rows[i]) in (allr - opt)]
    order = idx_opt + idx_val
    n_opt = len(idx_opt)

    raw = torch.stack([tr[i]["cached_text_part_raw"] for i in order]).float().numpy()   # [N, 5, D]
    ids = ["images/" + os.path.basename(str(tr[i]["image_path"])) for i in order]
    caps = {}
    for line in open(getattr(args, "qwen_text_cache_path", None)):
        d = json.loads(line)
        caps[str(d["image_id"])] = {k: str(d["codebook_texts"].get(k, "") or "") for k in AXES}
    # MS-COCO keys images as "images/train2014/<file>"; the ids built here are
    # "images/" + basename. Add that spelling as an alias where it is not
    # already a key, which leaves Flickr25K keys untouched.
    caps.update({("images/" + os.path.basename(k)): v for k, v in caps.items()
                 if ("images/" + os.path.basename(k)) not in caps})
    id_opt, id_val = ids[:n_opt], ids[n_opt:]
    vocab = distinctive_vocab(caps, id_opt)
    T_opt = {ax: multi_hot(caps, id_opt, ax, vocab[ax]) for ax in AXES}
    T_val = {ax: multi_hot(caps, id_val, ax, vocab[ax]) for ax in AXES}

    def unit(x): return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)

    def vec_endpoint(cent, lab_val, txt_val):
        """The post-hoc endpoint: batch-centred 4-way match, codeword vectors."""
        q = np.stack([cent[m][lab_val[:, m]] for m in range(4)], axis=1)          # [B,4,D]
        t = txt_val
        q = unit(q - q.mean(0, keepdims=True)); t = unit(t - t.mean(0, keepdims=True))
        S = np.einsum("bmd,bad->bma", q, t)
        tgt = np.arange(4)[None, :].repeat(S.shape[0], 0)
        return dict(code_picks_own_axis=round(float((S.argmax(2) == tgt).mean()), 4),
                    axis_picks_own_code=round(float((S.argmax(1) == tgt).mean()), 4))

    res = {"n_opt": n_opt, "n_val": len(idx_val), "K": K, "chance_vec": 0.25, "rows": []}
    for alpha in [float(x) for x in a.alphas.split(",")]:
        lab = np.zeros((len(order), 4), dtype=np.int64); cents = []
        for m in range(4):
            e = unit((1.0 - alpha) * raw[:, 0, :] + alpha * raw[:, m + 1, :])
            km = KMeans(n_clusters=K, n_init=3, random_state=a.kmeans_seed + m).fit(e)
            lab[:, m] = km.labels_; cents.append(km.cluster_centers_)
        c_opt, c_val = lab[:n_opt], lab[n_opt:]
        D = np.zeros((4, 4))
        for m in range(4):
            for j, ax in enumerate(AXES):
                D[m, j] = decode_slot(c_opt[:, m], T_opt[ax], c_val[:, m], T_val[ax],
                                      K, DEFAULT_ALPHA, DEFAULT_MIN_SUPPORT)["concept_mAP"]
        adv = [float(D[j, j] - np.mean([D[i, j] for i in range(4) if i != j])) for j in range(4)]
        row = {"alpha": alpha, "M1_advantage_mean": round(float(np.mean(adv)), 5),
               "M1_advantage_per_axis": [round(x, 5) for x in adv],
               "own_slot_is_argmax": int(sum(int(np.argmax(D[:, j]) == j) for j in range(4))),
               "D": D.round(4).tolist()}
        row.update(vec_endpoint(cents, c_val, raw[n_opt:, 1:, :]))
        res["rows"].append(row)
        print(f"alpha={alpha:<5} M1={row['M1_advantage_mean']:+.4f}  argmax={row['own_slot_is_argmax']}/4  "
              f"code->axis={row['code_picks_own_axis']:.3f}", flush=True)
    json.dump(res, open(a.out, "w"), indent=1)

main()
