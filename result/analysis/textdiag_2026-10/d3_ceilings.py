"""D3 -- ceilings for the A3 statistic, no trained model.

On the held-out validation rows of a dataset, three synthetic codes are scored with the A3 v2 reader
(lexical pair rule from the reference captions, bootstrap over images):
  oracle  : slot m = the k-means id (fit on opt-train axis-m caption embeddings) of the row's OWN
            axis-m caption -> S_oracle, the most a code can follow the text.
  probe   : slot m = a linear probe (logistic regression on frozen CLIP visual_global, fit on opt rows)
            predicting that id from the IMAGE -> S_probe, the most a text-free reader of the image can
            follow the text; plus its top-1 and balanced accuracy.
  image   : every slot = the k-means id of visual_global (one visual partition shared by all slots) ->
            a text-free control that should give S ~ 0.
"""
import argparse, json, os, sys
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from a3_v2 import AXES, KIND, elements, Reader  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True); ap.add_argument("--caption_file", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--K", type=int, default=128); ap.add_argument("--boot", type=int, default=1000); ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max_df", type=float, default=0.20); ap.add_argument("--min_shared_colour", type=int, default=2)
    a = ap.parse_args()
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    ids = json.load(open(os.path.join(a.cache_dir, "image_ids.json")))
    opt = sorted(int(r) for r in np.load(os.path.join(a.cache_dir, "opt_train_rows.npy")))
    allr = sorted(int(r) for r in np.load(os.path.join(a.cache_dir, "train_all_rows.npy")))
    caps = {}
    for line in open(a.caption_file):
        r = json.loads(line); caps[str(r["image_id"])] = r.get("descriptions") or r.get("codebook_texts") or {}
    opt = [r for r in opt if str(ids[r]) in caps]; val = [r for r in allr if r not in set(opt) and str(ids[r]) in caps]
    tp = np.load(os.path.join(a.cache_dir, "text_part.f16.npy"), mmap_mode="r")
    vg = np.load(os.path.join(a.cache_dir, "visual_global.f16.npy"), mmap_mode="r")
    Tt = np.stack([np.asarray(tp[r][1:5], dtype=np.float32) for r in opt]); Tv = np.stack([np.asarray(tp[r][1:5], dtype=np.float32) for r in val])
    mu = Tt.mean(0, keepdims=True); Tt -= mu; Tv -= mu
    Vt = np.stack([np.asarray(vg[r], dtype=np.float32) for r in opt]); Vv = np.stack([np.asarray(vg[r], dtype=np.float32) for r in val])
    Vt /= np.linalg.norm(Vt, axis=1, keepdims=True); Vv /= np.linalg.norm(Vv, axis=1, keepdims=True)
    N, M = len(val), 4
    oracle = np.zeros((N, M), int); probe = np.zeros((N, M), int); acc = {}
    for m in range(M):
        km = KMeans(n_clusters=a.K, n_init=4, random_state=0).fit(Tt[:, m])
        yt, yv = km.labels_, km.predict(Tv[:, m]); oracle[:, m] = yv
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(Vt, yt)
        pv = clf.predict(Vv); probe[:, m] = pv
        acc[AXES[m + 1]] = {"top1": round(float((pv == yv).mean()), 4), "balanced": round(float(balanced_accuracy_score(yv, pv)), 4),
                            "chance_top1": round(float(np.bincount(yv, minlength=a.K).max() / N), 4), "K": a.K}
    img = KMeans(n_clusters=a.K, n_init=4, random_state=0).fit_predict(Vv)
    image = np.repeat(img[:, None], M, 1)
    # pair rules on val rows (same construction as a3_v2)
    E = {AXES[m + 1]: [elements(caps[str(ids[r])].get(AXES[m + 1], ""), KIND[AXES[m + 1]]) for r in val] for m in range(M)}
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
    rules = {}
    for m in range(M):
        ax = AXES[m + 1]; excl = share[ax].clone()
        for k in range(M):
            if k != m:
                excl &= ~share[AXES[k + 1]]
        rules[ax] = {"lexical": share[ax] & iu, "lexical_axis_exclusive": excl & iu}
    reader = Reader(N, a.boot, a.seed)
    res = {"cache_dir": a.cache_dir, "caption_file": a.caption_file, "val_rows": N, "opt_rows": len(opt), "K": a.K, "probe_accuracy": acc}
    for name, code in (("oracle", oracle), ("probe", probe), ("image_kmeans", image)):
        c = torch.tensor(code)
        same = [(c[:, m][:, None] == c[:, m][None, :]) for m in range(M)]
        res[name] = {r: reader.reading(same, rules, r) for r in ("lexical", "lexical_axis_exclusive")}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({"val_rows": N, "probe_accuracy": acc}))
    for name in ("oracle", "probe", "image_kmeans"):
        for r in ("lexical", "lexical_axis_exclusive"):
            v = res[name][r]; print(f"{name:13s} {r:22s} S={v['S']['value']:+.3f} ci={v['S']['ci95']} pos_axes={v['S']['positive_axes']} ",
                                    {ax[2:10]: (x['pairs'], x['lift_own'], x['lift_other_mean']) if x else None for ax, x in v['per_axis'].items()})


if __name__ == "__main__":
    main()
