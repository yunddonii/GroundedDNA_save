"""For each individual attribute, find which single codebook predicts it best.
Argmax-cb should match the attribute's expected anatomy group.

Output: confusion matrix [expected_group, argmax_cb] showing how often
each attribute group's attrs are best-predicted by their expected cb.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--attr_names", default="/data/yschoi/dataset/deephashing/attributes.txt")
    ap.add_argument("--image_attr", default="dataset/CUB_200/attributes/image_attribute_labels.txt")
    ap.add_argument("--images_txt", default="dataset/CUB_200/images.txt")
    args = ap.parse_args()

    arr = np.load(os.path.join(args.run_dir, "extract_db.npz"), allow_pickle=True)
    cb_idx = arr["codebook_indices"]
    paths = arr["image_paths"]

    img_attr, attr_names = load_attribute_db(args.attr_names, args.image_attr)
    iids = cub_path_to_id(paths.tolist(), args.images_txt)
    mask = np.array([i > 0 for i in iids])
    cb_idx = cb_idx[mask]; iids = [i for i in iids if i > 0]
    name_to_id = {v: k for k, v in attr_names.items()}

    # Build per-attribute: (attr_name, attr_id, expected_group_idx, y_vector)
    target_attrs = []
    for g_idx, gname in enumerate(GROUP_ORDER):
        for prefix in ATTR_GROUPS[gname]:
            for name, aid in name_to_id.items():
                if name.startswith(prefix + "::"):
                    y = np.array([img_attr.get(iid, np.zeros(313, dtype=np.uint8))[aid] for iid in iids])
                    if y.sum() < 50 or y.sum() > len(y) - 50: continue
                    target_attrs.append((name, aid, g_idx, y))

    print(f"[probe] {len(target_attrs)} predictable attributes")

    M = cb_idx.shape[1]
    confusion = np.zeros((len(GROUP_ORDER), M), dtype=int)
    matches = 0
    per_attr_results = []
    for name, aid, g_exp, y in target_attrs:
        aucs = np.zeros(M)
        for m in range(M):
            X = np.eye(args.K, dtype=np.float32)[cb_idx[:, m]]
            try:
                Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
                clf = LogisticRegression(max_iter=200, C=1.0, solver='lbfgs')
                clf.fit(Xtr, ytr)
                aucs[m] = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
            except Exception:
                aucs[m] = 0.5
        cb_argmax = int(np.argmax(aucs))
        if cb_argmax == g_exp: matches += 1
        confusion[g_exp, cb_argmax] += 1
        per_attr_results.append({
            "name": name, "expected_cb": g_exp, "argmax_cb": cb_argmax,
            "aucs": aucs.tolist(), "match": cb_argmax == g_exp,
        })

    n_total = len(target_attrs)
    cb_names = ["C0_global", "C1_head_bill", "C2_upperparts", "C3_underparts",
                "C4_tail_appendages", "C5_pattern_markings"]
    print(f"\nConfusion: rows=expected group, cols=argmax cb")
    print(f"{'expected':>20s} " + " ".join(f"{n[:8]:>10s}" for n in cb_names))
    for g_idx, gname in enumerate(GROUP_ORDER):
        row_total = confusion[g_idx].sum()
        if row_total == 0:
            print(f"{gname:>20s} (no attrs)"); continue
        norm = confusion[g_idx] / row_total
        print(f"{gname:>20s} " + " ".join(f"{norm[m]:>10.3f}" for m in range(M)))
    diag = float(np.trace(confusion) / n_total)
    random = 1.0 / M
    print(f"\nDiagonal accuracy (% attributes best-predicted by their EXPECTED cb): {diag:.3f}")
    print(f"Random baseline ({M} cb): {random:.3f}")
    print(f"Lift over random: {diag - random:+.3f}")

    out = {
        "run_dir": args.run_dir, "K": int(args.K),
        "groups": GROUP_ORDER, "codebooks": cb_names,
        "confusion_count": confusion.tolist(),
        "n_total_attrs": n_total,
        "diagonal_accuracy": diag,
        "random_baseline": random,
        "lift": diag - random,
    }
    out_dir = os.path.join(args.run_dir, "grounding_probe_argmax")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "argmax_probe.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {out_dir}/argmax_probe.json")


if __name__ == "__main__":
    main()
