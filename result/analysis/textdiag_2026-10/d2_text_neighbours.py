"""D2 -- does the axis-m text-embedding neighbourhood share the axis-m ELEMENT (user decision 1)?

No model. For each dataset: opt-train rows with captions; per axis m the per-axis-centred EOS caption
embedding (cache text_part); for each row its k nearest rows (k in --k). Reports, per axis and k,
P(a neighbour pair shares >= 1 element word of that axis's kind) against three comparison sets:
  random pairs; the neighbours of the OTHER axes (same rows, neighbourhood taken on axis m'); and the
  CLIP image neighbours (visual_global). Also the share of rows with no element word, the
  neighbour-set overlap between axes (Jaccard of the k-NN sets), and the top shared element words.
Pass rule (plan, D2): share >= .50 at k = 10 on >= 3/4 axes, >= 1.5x the other-axis rate and >= 1.25x
the image-neighbour rate. Element extraction = a3_v2.elements (lexical, rule-based).
"""
import argparse, collections, glob, json, os, sys
import numpy as np, torch, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from a3_v2 import AXES, KIND, elements  # noqa: E402


def knn(x, k):
    x = F.normalize(x, dim=-1)
    sim = x @ x.t()
    sim.fill_diagonal_(-2.0)
    return sim.topk(k, dim=1).indices                                   # [N,k]


def share_rate(nbr, E, rng=None, n_random=200000):
    """nbr [N,k] long or None (random). E: list of element sets. Returns P(share>=1) and the top shared words."""
    N = len(E)
    cnt = collections.Counter(); hits = 0; tot = 0
    if nbr is None:
        I = rng.integers(0, N, n_random); J = rng.integers(0, N, n_random)
        pairs = [(int(i), int(j)) for i, j in zip(I, J) if i != j]
    else:
        pairs = [(i, int(j)) for i in range(N) for j in nbr[i].tolist()]
    for i, j in pairs:
        if not E[i] or not E[j]:
            continue
        tot += 1
        s = E[i] & E[j]
        if s:
            hits += 1
            cnt.update(s)
    return (hits / tot if tot else float("nan")), tot, cnt.most_common(8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True); ap.add_argument("--caption_file", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--k", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--max_rows", type=int, default=0); ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    ids = json.load(open(os.path.join(a.cache_dir, "image_ids.json")))
    opt = sorted(int(r) for r in np.load(os.path.join(a.cache_dir, "opt_train_rows.npy")))
    caps = {}
    for line in open(a.caption_file):
        r = json.loads(line); caps[str(r["image_id"])] = r.get("descriptions") or r.get("codebook_texts") or {}
    rows = [r for r in opt if str(ids[r]) in caps]
    if a.max_rows:
        rows = rows[:a.max_rows]
    tp = np.load(os.path.join(a.cache_dir, "text_part.f16.npy"), mmap_mode="r")
    vg_path = sorted(glob.glob(os.path.join(a.cache_dir, "visual_global*.f16.npy")))
    vg_path = [p for p in vg_path if "aug" not in os.path.basename(p)][0]
    vg = np.load(vg_path, mmap_mode="r")
    T = torch.tensor(np.stack([np.asarray(tp[r][1:5], dtype=np.float32) for r in rows]))      # [N,4,512]
    V = torch.tensor(np.stack([np.asarray(vg[r], dtype=np.float32) for r in rows]))             # [N,512]
    T = T - T.mean(0, keepdim=True)
    N = len(rows)
    E = {ax: [elements(caps[str(ids[r])].get(ax, ""), KIND[ax]) for r in rows] for ax in AXES[1:]}
    res = {"cache_dir": a.cache_dir, "caption_file": a.caption_file, "rows": N, "visual_global_file": os.path.basename(vg_path),
           "rows_without_element": {ax: round(float(np.mean([not s for s in E[ax]])), 3) for ax in AXES[1:]},
           "elements_per_row_mean": {ax: round(float(np.mean([len(s) for s in E[ax]])), 2) for ax in AXES[1:]}, "k": {}}
    nbr_txt = {ax: {k: knn(T[:, m], k) for k in a.k} for m, ax in enumerate(AXES[1:])}
    nbr_img = {k: knn(V, k) for k in a.k}
    rnd = {ax: share_rate(None, E[ax], rng) for ax in AXES[1:]}
    res["random_pairs"] = {ax: {"share": round(v[0], 4), "pairs": v[1], "top_shared": v[2]} for ax, v in rnd.items()}
    for k in a.k:
        res["k"][str(k)] = {}
        for m, ax in enumerate(AXES[1:]):
            own, n_own, top = share_rate(nbr_txt[ax][k], E[ax])
            other = {ax2: round(share_rate(nbr_txt[ax2][k], E[ax])[0], 4) for ax2 in AXES[1:] if ax2 != ax}
            img, _, _ = share_rate(nbr_img[k], E[ax])
            overlap = {ax2: round(float(np.mean([len(set(nbr_txt[ax][k][i].tolist()) & set(nbr_txt[ax2][k][i].tolist())) / k for i in range(N)])), 4)
                       for ax2 in AXES[1:] if ax2 != ax}
            overlap_img = round(float(np.mean([len(set(nbr_txt[ax][k][i].tolist()) & set(nbr_img[k][i].tolist())) / k for i in range(N)])), 4)
            oth_mean = float(np.mean(list(other.values())))
            res["k"][str(k)][ax] = {"own_axis_share": round(own, 4), "pairs_scored": n_own, "random_share": round(rnd[ax][0], 4),
                                    "other_axis_share": other, "other_axis_share_mean": round(oth_mean, 4), "image_nbr_share": round(img, 4),
                                    "ratio_vs_other": round(own / max(oth_mean, 1e-9), 3), "ratio_vs_image": round(own / max(img, 1e-9), 3),
                                    "ratio_vs_random": round(own / max(rnd[ax][0], 1e-9), 3),
                                    "knn_overlap_with_other_axes": overlap, "knn_overlap_with_image": overlap_img, "top_shared": top}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("rows", "rows_without_element", "elements_per_row_mean")}))
    for k in a.k:
        for ax in AXES[1:]:
            r = res["k"][str(k)][ax]
            print(f"k={k:>2} {ax:24s} own {r['own_axis_share']:.3f}  other {r['other_axis_share_mean']:.3f}  image {r['image_nbr_share']:.3f}  random {r['random_share']:.3f}  "
                  f"x_other {r['ratio_vs_other']:.2f} x_image {r['ratio_vs_image']:.2f}  knn-overlap other {np.mean(list(r['knn_overlap_with_other_axes'].values())):.3f} image {r['knn_overlap_with_image']:.3f}  top {[w for w,_ in r['top_shared'][:5]]}")


if __name__ == "__main__":
    main()
