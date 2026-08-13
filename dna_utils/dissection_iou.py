"""Dataset-level Network Dissection IoU, accumulated over every probe image (F12).

Bau, Zhou, Khosla, Oliva and Torralba (CVPR 2017) define the score as one ratio
over the whole probe set, not a mean of per-image ratios:

    IoU(k, c) = sum_i |M_k(i) AND L_c(i)|  /  sum_i |M_k(i) OR L_c(i)|

`slot_dissection_coco` iterated `per_cat.items()`, i.e. only the categories
annotated in image i. An image where category c is ABSENT contributed to neither
sum, so its activation area -- pure false positive for c -- never entered the
union. A slot that fires on everything then scored as though it only ever fired
on images containing c. The shuffled control had the identical bug, so it was
inflated in the same direction and could not reveal the problem.

Both properties matter and both are pinned by tests:

  * the category universe is FIXED up front, so a category absent from an image
    still contributes that image's activation area to the union, and a category
    absent from every image scores 0 rather than disappearing from the ranking;
  * the result is a ratio of sums, so a one-pixel image does not carry the same
    weight as a full one.
"""
from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np


def accumulate_iou(
    activations: np.ndarray,
    masks: Sequence[Mapping[int, np.ndarray]],
    categories: Sequence[int],
    *,
    _present_images_only: bool = False,
) -> Dict[int, np.ndarray]:
    """Dataset-level IoU per (category, slot).

    activations : [P, M, H, W] bool -- thresholded activation per probe image
    masks       : length-P; masks[i][c] is the binary mask of category c in
                  image i. A category missing from the dict is ABSENT there.
    categories  : the fixed category universe to score.

    `_present_images_only` reproduces the defective behaviour and exists solely
    so a test can pin how large the inflation was; production must never set it.
    """
    act = np.asarray(activations, dtype=bool)
    if act.ndim != 4:
        raise ValueError(f"activations must be [P, M, H, W], got {act.shape}")
    P, M, H, W = act.shape
    if len(masks) != P:
        raise ValueError(f"{len(masks)} mask dicts for {P} probe images")

    # Per-image activation area, reused for every absent category: with an empty
    # mask the intersection is 0 and the union is exactly this area.
    act_area = act.reshape(P, M, -1).sum(axis=2).astype(np.int64)   # [P, M]

    inter = {int(c): np.zeros(M, dtype=np.int64) for c in categories}
    union = {int(c): np.zeros(M, dtype=np.int64) for c in categories}

    for i in range(P):
        per_cat = masks[i] or {}
        for c in inter:
            cm = per_cat.get(c)
            if cm is None:
                if _present_images_only:
                    continue          # the defect: absent images skipped entirely
                # Absent category: no intersection, and the activation area is a
                # false positive that MUST be charged to the union.
                union[c] += act_area[i]
                continue
            cmb = np.asarray(cm, dtype=bool)
            if cmb.shape != (H, W):
                raise ValueError(
                    f"mask for category {c} in image {i} is {cmb.shape}, "
                    f"expected {(H, W)}")
            inter[c] += (act[i] & cmb[None]).reshape(M, -1).sum(axis=1)
            union[c] += (act[i] | cmb[None]).reshape(M, -1).sum(axis=1)

    return {c: inter[c] / np.maximum(union[c], 1) for c in inter}


def category_universe(masks: Sequence[Mapping[int, np.ndarray]]) -> list:
    """Every category annotated anywhere in the probe set, sorted.

    Fixing this BEFORE accumulating is what makes the union complete; deriving
    it per image is the defect.
    """
    seen = set()
    for per_cat in masks:
        seen.update(int(c) for c in (per_cat or {}))
    return sorted(seen)
