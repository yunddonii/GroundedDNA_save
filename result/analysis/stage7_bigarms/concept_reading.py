"""Stage 7 (A): dictionary-free reading of a named-concept code.

For a run trained with --concept_codebook_npz, every deployed local codon IS a concept (the layout is
a fixed bijection). A concept's name is the caption-word posterior of the TRAINING (opt) rows whose own
axis-m caption falls in that concept (nearest centre in the per-axis-centred cache space, as in the
loss) -- no image code is paired with any caption. Each held-out val image is read from its deployed
codon alone and scored against its own axis-m caption words with the stage-2 AP and posterior.
Also reports how often the deployed concept equals the image's own caption concept.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "result", "analysis", "stage5_bigidea"))
os.environ.setdefault("GDNA_NUM_SEMANTIC_PARTS", "5")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    from slot_role_probe import parse_args_txt, distinctive_vocab, multi_hot, AXES
    from heldout_codon_decoding import label_ranking_ap
    from dataloaders import load_dataset
    from zscr_like import read_scores, deployed_codes

    args = parse_args_txt(os.path.join(a.result_dir, "args.txt"))
    z = np.load(args.concept_codebook_npz)
    centres, mu, layout = z["centers"], z["axis_mean"], z["layout"]            # [4,64,D] [5,D] [4,64,3]
    inv = np.zeros((4, 64), np.int64)
    for m in range(4):
        inv[m, layout[m, :, 0] * 16 + layout[m, :, 1] * 4 + layout[m, :, 2]] = np.arange(64)
    cache = args.siglip2_feature_cache_dir
    opt = set(int(r) for r in np.load(os.path.join(cache, "opt_train_rows.npy")))
    tr, _, _ = load_dataset(getattr(args, "dataset_dir", "dataset"), args.dataset, setting="setting1",
                            train_transform=None, test_transform=None, load_train=True,
                            load_database=False, load_test=False, return_index=True,
                            qwen_text_cache_path=args.qwen_text_cache_path, siglip2_feature_cache_dir=cache)
    rows = np.asarray(tr._feat_cache_rows)
    io = [i for i in range(len(tr)) if int(rows[i]) in opt]
    iv = [i for i in range(len(tr)) if int(rows[i]) not in opt]
    tx = lambda idx: np.stack([np.asarray(tr[i]["cached_text_part_raw"], np.float32) for i in idx])  # noqa: E731
    tx_o, tx_v = tx(io), tx(iv)

    def concept_of(t):                                    # [n, 5, D] -> [n, 4]
        out = np.zeros((len(t), 4), np.int64)
        for m in range(4):
            d = ((t[:, m + 1, None, :] - mu[m + 1] - centres[m][None]) ** 2).sum(-1)
            out[:, m] = d.argmin(1)
        return out
    y_o, y_v = concept_of(tx_o), concept_of(tx_v)
    codon_v, _ = deployed_codes(a.result_dir, tr, iv, a.device)                 # [n_val, 5]
    k_v = np.stack([inv[m][codon_v[:, m + 1]] for m in range(4)], 1)             # deployed concept

    caps = {}
    for line in open(args.qwen_text_cache_path):
        d = json.loads(line)
        caps[str(d["image_id"])] = {k: str(d["codebook_texts"].get(k, "") or "") for k in AXES}
    caps.update({"images/" + os.path.basename(k): v for k, v in list(caps.items())
                 if "images/" + os.path.basename(k) not in caps})
    fc_ids = json.load(open(os.path.join(cache, "image_ids.json")))
    # CIFAR-10 samples carry no image_path; their captions are keyed by the cache row id.
    name = lambda i: ("images/" + os.path.basename(str(tr[i]["image_path"]))     # noqa: E731
                      if "image_path" in tr[i] else str(fc_ids[int(rows[i])]))
    id_o, id_v = [name(i) for i in io], [name(i) for i in iv]
    vocab = distinctive_vocab(caps, id_o)
    rep = {"result_dir": a.result_dir, "n_opt": len(io), "n_val": len(iv), "axes": {}}
    for m, ax in enumerate(AXES):
        To, Tv = multi_hot(caps, id_o, ax, vocab[ax]), multi_hot(caps, id_v, ax, vocab[ax])
        prior, n_prior = To.sum(0).astype(float), float(len(To))
        counts = np.stack([To[y_o[:, m] == c].sum(0) for c in range(64)]).astype(float)
        support = np.bincount(y_o[:, m], minlength=64).astype(float)
        S = read_scores(counts, support, prior, n_prior)
        rep["axes"][ax] = {
            "concept_name_reading": float(np.nanmean(label_ranking_ap(S[k_v[:, m]], Tv))),
            "caption_concept_upper_bound": float(np.nanmean(label_ranking_ap(S[y_v[:, m]], Tv))),
            "deployed_equals_caption_concept": float((k_v[:, m] == y_v[:, m]).mean()),
            "prior_only": float(np.nanmean(label_ranking_ap(np.tile((prior + 1) / (n_prior + 2), (len(Tv), 1)), Tv))),
            "distinct_concepts_used": int(len(np.unique(k_v[:, m]))),
        }
    rep["mean"] = {k: float(np.mean([rep["axes"][ax][k] for ax in AXES])) for k in rep["axes"][AXES[0]]}
    json.dump(rep, open(a.out, "w"), indent=1)
    print(json.dumps(rep["mean"], indent=1))


if __name__ == "__main__":
    main()
