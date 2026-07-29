#!/usr/bin/env python
"""Give CIFAR-10 extractions the `image_paths` field they were written without.

Why the CIFAR held-out codon-decoding cell has always been blank: unlike
Flickr25k / NUS-WIDE / MSCOCO, the CIFAR-10 extractions contain only
`base_indices, hash_2bit, codebook_indices, multi_hot_labels` -- **no
`image_paths`**. `scripts/heldout_codon_decoding.py` aligns the train rows to
the DB rows by image basename, so with no identifier it cannot run at all.
This was never a missing `setting1/train.txt`; the field is absent from the
npz.

CIFAR-10 has no per-image filesystem path, but `dataloaders._cifar10_image_id`
gives a stable content-md5 id (that is exactly how the feature cache is keyed).
This script recomputes those ids for the train / database / query splits with
the same `load_dataset` dispatcher `train_siglip2.py` uses, checks the row
counts against the saved extractions, and writes *new* npz files carrying an
`image_paths` field. Originals are never modified.

Row-order assumption (checked, not assumed silently): the extraction loops the
split loader without shuffling, so extraction row i corresponds to dataset row
i. The script asserts the per-row label vectors agree between the dataset and
the extraction; a mismatch aborts.

Output goes to `<result_dir>/withids/`; the paper artifacts in `result_dir`
stay byte-identical.

Usage:
  python scripts/cifar_inject_image_ids.py --result_dir result/...cifar_A_v4_P0refit_e19...
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)


def split_ids(args, which: str):
    """Return (ids, labels) for 'train' | 'database' | 'test'."""
    from dataloaders import load_dataset, _cifar10_image_id

    kw = dict(load_train=False, load_database=False, load_test=False)
    kw[{"train": "load_train", "database": "load_database",
        "test": "load_test"}[which]] = True
    tr, te, db = load_dataset(
        getattr(args, "dataset_dir", "dataset"), "CIFAR10",
        setting=getattr(args, "setting", "setting1"),
        train_transform=None, test_transform=None, return_index=True,
        qwen_text_cache_path=getattr(args, "qwen_text_cache_path", None),
        siglip2_feature_cache_dir=getattr(args, "siglip2_feature_cache_dir", None),
        **kw)
    ds = {"train": tr, "test": te, "database": db}[which]
    if ds is None:
        raise SystemExit(f"load_dataset returned None for split={which}")
    data = np.asarray(ds.data)
    ids = np.array([_cifar10_image_id(data[i]) for i in range(len(data))])
    return ids, np.asarray(ds.targets)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", required=True)
    ap.add_argument("--train_manifest",
                    default="dataset/CIFAR10/setting1/train_image_ids.txt")
    a = ap.parse_args()

    args = torch.load(os.path.join(a.result_dir, "config.pt"),
                      map_location="cpu", weights_only=False)

    # Write augmented copies into an overlay dir; the paper artifacts in
    # result_dir are left byte-identical.
    out_dir = os.path.join(a.result_dir, "withids")
    os.makedirs(out_dir, exist_ok=True)
    for which, fn in (("database", "extract_db.npz"), ("test", "extract_query.npz")):
        src = os.path.join(a.result_dir, fn)
        dst = os.path.join(out_dir, fn)
        z = dict(np.load(src, allow_pickle=True))
        if "image_paths" in z:
            print(f"  {fn}: already has image_paths, skipped")
            continue
        ids, targets = split_ids(args, which)
        n_ex = len(z["base_indices"])
        if len(ids) != n_ex:
            raise SystemExit(f"{fn}: dataset has {len(ids)} rows, extraction has {n_ex}")
        # order check: extraction labels must match dataset targets row-for-row
        lab = np.asarray(z["multi_hot_labels"])
        lab_cls = lab.argmax(1) if lab.ndim == 2 else lab.ravel()
        n_bad = int((lab_cls != targets).sum())
        if n_bad:
            raise SystemExit(f"{fn}: row order mismatch, {n_bad}/{n_ex} labels disagree")
        z["image_paths"] = ids
        np.savez(dst, **z)
        print(f"  {fn}: wrote {dst} with {len(ids)} image ids (label order verified)")

    ids_tr, _ = split_ids(args, "train")
    os.makedirs(os.path.dirname(a.train_manifest) or ".", exist_ok=True)
    with open(a.train_manifest, "w") as f:
        f.write("\n".join(ids_tr.tolist()) + "\n")
    print(f"wrote {a.train_manifest}  n={len(ids_tr)}  unique={len(set(ids_tr.tolist()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
