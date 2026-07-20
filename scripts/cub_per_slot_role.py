"""Role validity on CUB, using per-attribute targets instead of image-level labels.

Image-level labels cannot test role assignment: all six slots share one target,
and the 2026-07-20 control showed the resulting column effect is a property of
the captions, identical under arbitrary flat-hash chunks. CUB gives 312 binary
part attributes (bill shape, wing colour, belly pattern, ...), i.e. genuinely
per-region targets, and its v6b captions were written per anatomical region
(tools/qwen3_v6b_cub_trainset.py):

    C_global -> whole bird     C_primary_object     -> head / bill
    C_secondary_object -> wing C_activity_or_relation -> underparts
    C_color_texture -> tail    C_scene_type         -> pattern / markings

The test needs neither the canonical 28 attribute groups nor the attribute names
(neither is present in this copy of CUB, and both were unreliable to reconstruct:
colour groups are multi-select, so mutual-exclusivity recovery fragments them).
Instead, for every attribute a we ask two questions and compare the answers:

    teacher side  which CAPTION slot best predicts a?   -> T(a)
    code side     which CODE slot best decodes a?       -> C(a)

Role validity = agreement(T, C) above chance. If the code inherited the
teacher's division of labour, the slot that best decodes an attribute should be
the slot whose caption talks about it. Flat-hash chunk baselines get the same
treatment, and a label-permutation null fixes the chance level.

Everything is fitted on train (= CUB database split) and evaluated on the
official test split, which is disjoint from it.

Usage:
    python scripts/cub_per_slot_role.py \
        --ours_dir result/260619+cub_200_..._K64_... \
        --cache_dir cache/cub200_clip_v6bplus \
        --out docs/cub_per_slot_role.json
"""
import argparse
import json
import os

import numpy as np

SLOT_NAMES = ["global", "head_bill", "wing_upperparts",
              "underparts", "tail_appendages", "pattern_markings"]


def auc(scores: np.ndarray, y: np.ndarray) -> float:
    """AUROC via rank statistic; nan if a class is missing."""
    pos, neg = int(y.sum()), int((~y.astype(bool)).sum())
    if pos == 0 or neg == 0:
        return np.nan
    r = np.argsort(np.argsort(scores)) + 1.0
    return float((r[y.astype(bool)].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def _basenames(p):
    return np.array([os.path.basename(str(x)) for x in p])


def load_attributes(root: str) -> np.ndarray:
    d = np.loadtxt(os.path.join(root, "attributes/image_attribute_labels.txt"),
                   usecols=(0, 1, 2), dtype=np.int64)
    N, A = int(d[:, 0].max()), int(d[:, 1].max())
    M = np.zeros((N, A), dtype=np.int8)
    M[d[:, 0] - 1, d[:, 1] - 1] = d[:, 2]
    return M


def code_side_auc(tr_codes, te_codes, tr_y, te_y, alpha=1.0):
    """[M, A] AUC of decoding each attribute from each code slot's symbol."""
    M, A = tr_codes.shape[1], tr_y.shape[1]
    out = np.full((M, A), np.nan)
    for m in range(M):
        u, inv = np.unique(tr_codes[:, m], return_inverse=True)
        pos = np.zeros((len(u), A)); cnt = np.zeros(len(u))
        np.add.at(pos, inv, tr_y.astype(np.float64))
        np.add.at(cnt, inv, 1.0)
        p = (pos + alpha) / (cnt[:, None] + 2 * alpha)          # p(a | codeword)
        prior = (tr_y.sum(0) + alpha) / (len(tr_y) + 2 * alpha)
        pos_of = {v: i for i, v in enumerate(u)}
        idx = np.array([pos_of.get(v, -1) for v in te_codes[:, m]])
        S = np.where(idx[:, None] >= 0, p[idx], prior[None, :])  # unseen -> prior
        for a in range(A):
            out[m, a] = auc(S[:, a], te_y[:, a])
    return out


def teacher_side_auc(txt, y, rng):
    """[K, A] AUC of predicting each attribute from each caption slot embedding.

    Evaluated by a split-half of the CAPTIONED rows, not on the test split:
    CUB captions were generated for the train split only (has_text is 5,994 of
    11,788), so scoring the teacher on test would compare zero vectors and
    return chance for every slot -- which is exactly what the first run did.
    The teacher assignment is a property of the captions, so measuring it
    within the captioned rows (fit the centroid on one half, score the other)
    is the right scope. The CODE side is still evaluated out-of-sample on test.
    """
    K, A = txt.shape[1], y.shape[1]
    out = np.full((K, A), np.nan)
    perm = rng.permutation(len(txt))
    h = len(perm) // 2
    fit, ev = perm[:h], perm[h:]
    for k in range(K):
        Xf, Xe = txt[fit, k, :], txt[ev, k, :]
        for a in range(A):
            m = y[fit, a].astype(bool)
            if m.sum() < 5 or (~m).sum() < 5:
                continue
            c = Xf[m].mean(0)
            c /= (np.linalg.norm(c) + 1e-12)
            out[k, a] = auc(Xe @ c, y[ev, a])
    return out


def _double_center(M: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Strip row and column main effects, leaving only (slot, attribute) interaction.

    Necessary here: one code slot (global) has the highest AUC on almost every
    attribute (244 of 269 raw argmax wins), so a raw argmax comparison measures
    that row effect and nothing else. The 2026-07-20 Flickr control established
    the same point for the caption side -- main effects are properties of the
    targets, not evidence of role assignment.
    """
    X = M[:, keep].copy()
    return X - X.mean(1, keepdims=True) - X.mean(0, keepdims=True) + X.mean()


def agreement(T: np.ndarray, C: np.ndarray, keep: np.ndarray,
              residual: bool = False) -> dict:
    """How often does the best-decoding code slot match the best-describing caption slot?"""
    if residual:
        t = _double_center(T, keep).argmax(0)
        c = _double_center(C, keep).argmax(0)
    else:
        t = np.nanargmax(T[:, keep], axis=0)
        c = np.nanargmax(C[:, keep], axis=0)
    acc = float((t == c).mean())
    per = {}
    for s in range(T.shape[0]):
        sel = t == s
        per[SLOT_NAMES[s]] = {
            "n_attributes": int(sel.sum()),
            "code_agrees": float((c[sel] == s).mean()) if sel.sum() else float("nan"),
        }
    return {"agreement": acc, "n_attributes": int(keep.sum()),
            "per_teacher_slot": per,
            "teacher_assignment_counts": np.bincount(t, minlength=6).tolist(),
            "code_assignment_counts": np.bincount(c, minlength=6).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours_dir", required=True)
    ap.add_argument("--baseline_dirs", nargs="*", default=[])
    ap.add_argument("--baseline_names", nargs="*", default=[])
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--dataset_root", default="dataset/CUB_200")
    ap.add_argument("--min_pos", type=int, default=50)
    ap.add_argument("--n_perm", type=int, default=500)
    ap.add_argument("--n_shuffle_partitions", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    Y = load_attributes(args.dataset_root)
    order = [l.split()[1] for l in open(os.path.join(args.dataset_root, "images.txt"))]
    row_of_img = {os.path.basename(p): i for i, p in enumerate(order)}
    print(f"[cub] attributes {Y.shape}")

    tr = np.load(os.path.join(args.ours_dir, "extract_db.npz"), allow_pickle=True)
    te = np.load(os.path.join(args.ours_dir, "extract_query.npz"), allow_pickle=True)
    tr_bn, te_bn = _basenames(tr["image_paths"]), _basenames(te["image_paths"])
    assert len(np.intersect1d(tr_bn, te_bn)) == 0, "train/test overlap"
    tr_y = Y[[row_of_img[b] for b in tr_bn]]
    te_y = Y[[row_of_img[b] for b in te_bn]]
    print(f"[cub] fit on {len(tr_bn)} (db=train), evaluate on {len(te_bn)} held-out test")

    keep = (tr_y.sum(0) >= args.min_pos) & (te_y.sum(0) >= args.min_pos) & \
           ((len(tr_y) - tr_y.sum(0)) >= args.min_pos) & ((len(te_y) - te_y.sum(0)) >= args.min_pos)
    print(f"[cub] usable attributes: {keep.sum()}/{Y.shape[1]} (min_pos={args.min_pos})")

    # caption embeddings, aligned to the same rows
    ids = json.load(open(os.path.join(args.cache_dir, "image_ids.json")))
    crow = {os.path.basename(str(s)): i for i, s in enumerate(ids)}
    T_all = np.load(os.path.join(args.cache_dir, "text_part.f16.npy"), mmap_mode="r")
    has = np.asarray(np.load(os.path.join(args.cache_dir, "has_text.bool.npy"),
                             mmap_mode="r")).astype(bool)
    cap_rows = np.array([crow[b] for b in tr_bn])
    cap_ok = has[cap_rows]
    print(f"[cub] captioned train rows: {cap_ok.sum()}/{len(cap_ok)}  "
          f"(test split has no captions -> teacher scored within train)")
    tr_txt = np.asarray(T_all[cap_rows[cap_ok]], dtype=np.float64)
    tr_txt -= tr_txt.mean(0, keepdims=True)
    tr_txt /= (np.linalg.norm(tr_txt, axis=-1, keepdims=True) + 1e-12)

    print("[cub] teacher side (split-half within captioned train) ...")
    T = teacher_side_auc(tr_txt, tr_y[cap_ok], rng)

    def codon(b):
        return b[:, 0::3] * 16 + b[:, 1::3] * 4 + b[:, 2::3]

    results = {}
    print("[cub] code side: ours ...")
    C = code_side_auc(codon(tr["base_indices"]), codon(te["base_indices"]), tr_y, te_y)
    results["ours"] = agreement(T, C, keep)
    results["ours_residual"] = agreement(T, C, keep, residual=True)

    # Control: keep the same 18 bases but destroy the slot boundaries. Same
    # information content, no slot structure -- so anything the true grouping
    # scores above this is attributable to where the boundaries fall.
    shuf = []
    for i in range(args.n_shuffle_partitions):
        r2 = np.random.default_rng(1000 + i)
        pm = r2.permutation(tr["base_indices"].shape[1])
        Cs = code_side_auc(codon(tr["base_indices"][:, pm]),
                           codon(te["base_indices"][:, pm]), tr_y, te_y)
        shuf.append(agreement(T, Cs, keep, residual=True)["agreement"])
    results["shuffled_slot_boundaries"] = {
        "agreements": shuf, "mean": float(np.mean(shuf)),
        "note": "same 18 bases regrouped at random; destroys slot boundaries only",
    }

    for bdir, bname in zip(args.baseline_dirs, args.baseline_names):
        btr = np.load(os.path.join(bdir, "extract_db.npz"), allow_pickle=True)
        bte = np.load(os.path.join(bdir, "extract_query.npz"), allow_pickle=True)
        p1 = {b: i for i, b in enumerate(_basenames(btr["image_paths"]))}
        p2 = {b: i for i, b in enumerate(_basenames(bte["image_paths"]))}
        Cb = code_side_auc(codon(btr["base_indices"][[p1[b] for b in tr_bn]]),
                           codon(bte["base_indices"][[p2[b] for b in te_bn]]), tr_y, te_y)
        results[bname] = agreement(T, Cb, keep)
        print(f"[cub] code side: {bname} ...")

    # chance level: permute the code-slot identities per attribute
    t = _double_center(T, keep).argmax(0)
    c = _double_center(C, keep).argmax(0)
    null = [float((t == rng.permutation(c)).mean()) for _ in range(args.n_perm)]
    lo, hi = np.percentile(null, [2.5, 97.5])
    results["permutation_null"] = {"mean": float(np.mean(null)),
                                   "ci95_low": float(lo), "ci95_high": float(hi)}

    print(f"\n{'partition':14s} {'agreement':>10s}   (chance {np.mean(null):.3f} "
          f"[{lo:.3f},{hi:.3f}], uniform 1/6={1/6:.3f})")
    for k, v in results.items():
        if "agreement" in v:
            print(f"  {k:22s} {v['agreement']:>10.3f}")
    sb = results.get("shuffled_slot_boundaries")
    if sb:
        print(f"  {'shuffled_boundaries':22s} {sb['mean']:>10.3f}   "
              f"(runs: {', '.join(f'{x:.3f}' for x in sb['agreements'])})")
    print(f"\n  ours per teacher slot:")
    for s, v in results["ours"]["per_teacher_slot"].items():
        print(f"    {s:18s} n={v['n_attributes']:>3d}  code agrees {v['code_agrees']:.3f}")
    print(f"  teacher assignment counts: {results['ours']['teacher_assignment_counts']}")
    print(f"  code    assignment counts: {results['ours']['code_assignment_counts']}")

    json.dump({"results": results, "slot_names": SLOT_NAMES,
               "teacher_auc_mean": float(np.nanmean(T[:, keep])),
               "config": vars(args)}, open(args.out, "w"), indent=2)
    print(f"\n[cub] wrote {args.out}")


if __name__ == "__main__":
    main()
