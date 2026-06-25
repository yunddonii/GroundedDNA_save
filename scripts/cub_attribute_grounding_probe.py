"""CUB-200 Pillar-1 attribute grounding probe (v3 - drop-out unique contribution).

For each attribute group g, train two probes:
  - Full probe: all 6 codebooks → predict attribute
  - Drop-cb_m probe: all 6 except cb_m → predict attribute
Unique contribution UC[m, g] = AUC_full - AUC_drop_m.

A grounded codebook m has high UC[m, g_own] >> UC[m, g_other].
Diagonal dominance = mean(diag(UC)) - mean(off-diag(UC)).
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

ATTR_GROUPS = {
    "global":          ["has_shape", "has_size"],
    "head_bill":       ["has_bill_shape", "has_bill_color", "has_bill_length", "has_crown_color",
                        "has_forehead_color", "has_eye_color", "has_nape_color"],
    "upperparts_wing": ["has_wing_color", "has_wing_shape", "has_wing_pattern",
                        "has_back_color", "has_back_pattern",
                        "has_upperparts_color", "has_primary_color", "has_upper_tail_color"],
    "underparts":      ["has_belly_color", "has_belly_pattern", "has_breast_color",
                        "has_breast_pattern", "has_throat_color", "has_underparts_color"],
    "tail_appendages": ["has_tail_shape", "has_tail_pattern", "has_under_tail_color", "has_leg_color"],
    "pattern_markings":["has_head_pattern", "has_wing_pattern", "has_tail_pattern",
                        "has_breast_pattern", "has_belly_pattern", "has_back_pattern"],
}
GROUP_ORDER = ["global", "head_bill", "upperparts_wing", "underparts", "tail_appendages", "pattern_markings"]


def load_attribute_db(attr_names_path, image_attr_path, certainty_threshold=3):
    attr_names = {}
    with open(attr_names_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                attr_names[int(parts[0])] = parts[1]
    img_attr = {}
    with open(image_attr_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4: continue
            img_id, attr_id, is_present, cert = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            if cert < certainty_threshold: continue
            if img_id not in img_attr: img_attr[img_id] = np.zeros(313, dtype=np.uint8)
            img_attr[img_id][attr_id] = is_present
    return img_attr, attr_names


def cub_path_to_id(image_paths, images_txt_path):
    path_to_id = {}
    with open(images_txt_path) as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                path_to_id[parts[1]] = int(parts[0])
    out = []
    for p in image_paths:
        if isinstance(p, bytes): p = p.decode()
        matched = False
        for suffix in path_to_id:
            if p.endswith(suffix):
                out.append(path_to_id[suffix]); matched = True; break
        if not matched: out.append(-1)
    return out


def build_attr_matrix(img_attr, attr_names, image_ids):
    name_to_id = {v: k for k, v in attr_names.items()}
    N = len(image_ids)
    X_attr = np.zeros((N, 313), dtype=np.uint8)
    for i, iid in enumerate(image_ids):
        if iid in img_attr:
            X_attr[i] = img_attr[iid]
    group_attrs = {}
    for g in GROUP_ORDER:
        members = ATTR_GROUPS[g]
        ids = [aid for name, aid in name_to_id.items() if any(name.startswith(p + "::") for p in members)]
        group_attrs[g] = ids
    return X_attr, group_attrs


def one_hot_concat(codebook_indices, K, mask_m=None):
    """Concatenate one-hot of each codebook. If mask_m, drop that column block."""
    N, M = codebook_indices.shape
    blocks = []
    for m in range(M):
        if mask_m is not None and m == mask_m: continue
        blocks.append(np.eye(K, dtype=np.float32)[codebook_indices[:, m]])
    return np.concatenate(blocks, axis=1)


def probe_dropout(codebook_indices, X_attr, group_attrs, K=64, seed=0,
                  min_pos=50, min_neg=50):
    M = codebook_indices.shape[1]
    G = len(GROUP_ORDER)
    UC = np.zeros((M, G), dtype=np.float32)
    n_attrs = np.zeros((M, G), dtype=int)
    print(f"[probe] M={M} G={G}, training {M+1} probes per attribute...")
    for g_idx, gname in enumerate(GROUP_ORDER):
        aids = group_attrs[gname]
        for a in aids:
            y = X_attr[:, a]
            pos = int(y.sum()); neg = len(y) - pos
            if pos < min_pos or neg < min_neg: continue
            # full probe
            X_full = one_hot_concat(codebook_indices, K)
            try:
                Xtr, Xte, ytr, yte = train_test_split(X_full, y, test_size=0.3, stratify=y, random_state=seed)
                clf = LogisticRegression(max_iter=200, C=1.0, solver='lbfgs')
                clf.fit(Xtr, ytr)
                auc_full = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
            except Exception:
                continue
            for m in range(M):
                X_drop = one_hot_concat(codebook_indices, K, mask_m=m)
                try:
                    Xtr, Xte, ytr, yte = train_test_split(X_drop, y, test_size=0.3, stratify=y, random_state=seed)
                    clf = LogisticRegression(max_iter=200, C=1.0, solver='lbfgs')
                    clf.fit(Xtr, ytr)
                    auc_drop = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
                    UC[m, g_idx] += (auc_full - auc_drop)
                    n_attrs[m, g_idx] += 1
                except Exception:
                    continue
    UC_mean = np.where(n_attrs > 0, UC / np.maximum(n_attrs, 1), 0)
    return UC_mean, n_attrs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--attr_names", default="/data/yschoi/dataset/deephashing/attributes.txt")
    ap.add_argument("--image_attr", default="dataset/CUB_200/attributes/image_attribute_labels.txt")
    ap.add_argument("--images_txt", default="dataset/CUB_200/images.txt")
    ap.add_argument("--out_subdir", default="grounding_probe_dropout")
    args = ap.parse_args()

    arr = np.load(os.path.join(args.run_dir, "extract_db.npz"), allow_pickle=True)
    cb_idx = arr["codebook_indices"]
    paths  = arr["image_paths"]
    print(f"[run] codebook_indices shape={cb_idx.shape}")

    img_attr, attr_names = load_attribute_db(args.attr_names, args.image_attr)
    iids = cub_path_to_id(paths.tolist(), args.images_txt)
    mask = np.array([i > 0 for i in iids])
    cb_idx = cb_idx[mask]
    iids = [i for i in iids if i > 0]
    X_attr, group_attrs = build_attr_matrix(img_attr, attr_names, iids)
    print(f"[run] N(valid)={cb_idx.shape[0]}")

    UC, n_attrs = probe_dropout(cb_idx, X_attr, group_attrs, K=args.K)
    print("\nUnique Contribution matrix UC[m,g] = AUC_full - AUC_drop_m (mean over attrs in g):")
    print(f"{'cb_vs_g':>20s} " + " ".join(f"{g[:10]:>10s}" for g in GROUP_ORDER))
    cb_names = ["C0_global", "C1_head_bill", "C2_upperparts", "C3_underparts",
                "C4_tail_appendages", "C5_pattern_markings"]
    for m in range(UC.shape[0]):
        print(f"{cb_names[m]:>20s} " + " ".join(f"{UC[m,g]:>10.4f}" for g in range(UC.shape[1])))

    print("\nAttribute counts per (cb, group):")
    print(f"{'cb_vs_g':>20s} " + " ".join(f"{g[:10]:>10s}" for g in GROUP_ORDER))
    for m in range(n_attrs.shape[0]):
        print(f"{cb_names[m]:>20s} " + " ".join(f"{n_attrs[m,g]:>10d}" for g in range(n_attrs.shape[1])))

    diag = np.diag(UC); off = (UC.sum() - diag.sum()) / (UC.size - len(diag))
    DD = diag.mean() - off
    print(f"\nDiagonal mean UC : {diag.mean():.5f}")
    print(f"Off-diagonal mean: {off:.5f}")
    print(f"Diagonal Dominance: {DD:.5f}  (>0 = grounded)")
    # per-cb argmax (which group does each cb specialize in?)
    print("\nPer-cb argmax (which group does cb_m contribute most to?):")
    for m in range(UC.shape[0]):
        g_argmax = int(np.argmax(UC[m]))
        match = "★" if g_argmax == m else ""
        print(f"  {cb_names[m]:>20s}: argmax={GROUP_ORDER[g_argmax]} {match}")

    out = {
        "run_dir": args.run_dir, "K": int(args.K),
        "UC_matrix": UC.tolist(),
        "n_attrs_per_cell": n_attrs.tolist(),
        "groups": GROUP_ORDER,
        "codebooks": cb_names,
        "diag_mean": float(diag.mean()),
        "off_diag_mean": float(off),
        "diagonal_dominance": float(DD),
    }
    out_dir = os.path.join(args.run_dir, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "probe.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {out_dir}/probe.json")


if __name__ == "__main__":
    main()
