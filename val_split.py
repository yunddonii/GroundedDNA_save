"""P0 protocol: the single definition of the train -> opt-train / val carve-out.

Both `train_siglip2.py` (which trains on opt-train and selects the checkpoint on
val) and `scripts/build_opt_train_rows.py` (which restricts the whitening fit to
opt-train) must carve the split IDENTICALLY -- if they ever diverge, the
whitening matrix is fitted on rows the model also validates on, silently
re-introducing the leak the protocol exists to remove. So the logic lives here
once and is imported by both.
"""
from typing import Tuple

import numpy as np


def get_train_labels(trainset) -> np.ndarray:
    """Labels of a train split, across the two dataset layouts in this repo.

    `ImgRtvDataset` (Flickr/MSCOCO/NUS-WIDE) exposes `img_labels` [N, C]
    one-hot/multi-hot; `ImgRtvCIFAR10` subclasses torchvision and exposes
    `targets` [N] as integer class ids instead.
    """
    lab = getattr(trainset, "img_labels", None)
    if lab is None:
        lab = getattr(trainset, "targets", None)
    if lab is None:
        raise AttributeError(
            f"{type(trainset).__name__} exposes neither `img_labels` nor "
            f"`targets`; cannot carve a stratified validation split."
        )
    return np.asarray(lab)


def carve_val_indices(labels: np.ndarray, ratio: float, seed: int
                      ) -> Tuple[np.ndarray, np.ndarray, str]:
    """Split train row ids into (opt_train, val_query).

    Single-label datasets are class-stratified so the small val split keeps the
    class balance; multi-label datasets fall back to a plain seeded shuffle
    (exact multi-label stratification is NP-hard and unnecessary at these sizes).

    Returns (opt_idx, val_idx, description). Deterministic given `seed`.
    """
    labels = np.asarray(labels)
    n = int(len(labels))
    if not (0.0 < float(ratio) < 0.5):
        raise ValueError(f"ratio must be in (0, 0.5), got {ratio}")
    rng = np.random.RandomState(int(seed))

    cls = None
    if labels.ndim == 1:
        cls = labels.astype(np.int64)
    elif labels.ndim == 2 and bool((labels.sum(axis=1) == 1).all()):
        cls = labels.argmax(axis=1).astype(np.int64)

    if cls is not None:
        parts = []
        for c in np.unique(cls):
            rows = np.where(cls == c)[0]
            rng.shuffle(rows)
            parts.append(rows[:max(1, int(round(len(rows) * ratio)))])
        val_idx = np.sort(np.concatenate(parts))
        desc = f"class-stratified over {len(np.unique(cls))} classes"
    else:
        n_val = max(1, int(round(n * ratio)))
        val_idx = np.sort(rng.permutation(n)[:n_val])
        desc = "random (multi-label split)"

    mask = np.ones(n, dtype=bool)
    mask[val_idx] = False
    opt_idx = np.where(mask)[0]
    assert len(np.intersect1d(opt_idx, val_idx)) == 0, "val/opt-train overlap"
    assert len(opt_idx) + len(val_idx) == n, "split does not partition train"
    return opt_idx, val_idx, desc


def cache_rows_for_dataset(ds, root: str) -> np.ndarray:
    """Cache row index per dataset row, across both dataset layouts."""
    rows = getattr(ds, "_feat_cache_rows", None)
    if rows is not None:
        return np.asarray(rows)
    from dataloaders import _build_pathkeyed_cache_row_map, _SigLIP2FeatureCache
    cache_dir = getattr(ds, "_siglip2_feature_cache_dir", None)
    if cache_dir is None:
        raise ValueError("dataset has no feature cache; cannot map rows.")
    fc = _SigLIP2FeatureCache(cache_dir)
    return np.asarray(_build_pathkeyed_cache_row_map(list(ds.img_paths), root, fc))
