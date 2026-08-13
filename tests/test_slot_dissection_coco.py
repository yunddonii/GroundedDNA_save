"""F12: Network Dissection IoU must accumulate over ALL probe images.

The dataset-level IoU in Bau et al. (CVPR 2017) is a single ratio over the whole
probe set:

    IoU(k, c) = sum_i |M_k(i) AND L_c(i)|  /  sum_i |M_k(i) OR L_c(i)|

`slot_dissection_coco` looped over `per_cat.items()` -- the categories that
appear in image i -- so an image where category c is ABSENT contributed nothing
to either sum. Its activation area, which is pure false positive for c, never
entered the union. A slot that fires everywhere therefore scored as if it only
ever fired on images containing c, and the reported IoU was systematically
inflated. The shuffled control had the same bug, so the control was inflated too
and could not expose it.

The fix is to fix the category universe first and accumulate over every
(image, category) pair, using an empty mask where the category is absent: the
intersection is 0 and the activation area lands in the union, which is exactly
the penalty the metric is supposed to apply.

The test below is the audit's fixture: two images, the category present only in
the first, activation on in both. The old code sees only image 1 and reports
IoU = 1.0; the correct value is 0.5, and the difference is the entire defect.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from dna_utils.dissection_iou import accumulate_iou  # noqa: E402


def _mask(*rows):
    return np.array(rows, dtype=bool)


def test_absent_category_activation_enters_the_union():
    """The audit's fixture. Old behaviour: 1.0. Correct: 0.5."""
    act = np.stack([                       # [2 images, 1 slot, 2, 2]
        _mask([1, 1], [0, 0])[None],
        _mask([1, 1], [0, 0])[None],
    ])
    masks = [
        {7: _mask([1, 1], [0, 0])},        # image 0 has category 7
        {},                                # image 1 does not
    ]
    iou = accumulate_iou(act, masks, categories=[7])
    # inter = 2 (image 0 only); union = 2 + 2 = 4, because image 1's activation
    # is a false positive for category 7 and must be counted.
    assert iou[7][0] == pytest.approx(0.5)


def test_old_behaviour_would_have_scored_one():
    """Pin the magnitude of the inflation so a regression is unmistakable."""
    act = np.stack([_mask([1, 1], [0, 0])[None], _mask([1, 1], [0, 0])[None]])
    masks = [{7: _mask([1, 1], [0, 0])}, {}]
    present_only = accumulate_iou(act, masks, categories=[7],
                                  _present_images_only=True)
    assert present_only[7][0] == pytest.approx(1.0)
    assert accumulate_iou(act, masks, categories=[7])[7][0] == pytest.approx(0.5)


def test_category_universe_is_fixed_not_derived_per_image():
    """A category that appears in no probe image must still be scored, and must
    score 0 rather than vanishing from the ranking."""
    act = np.stack([_mask([1, 0], [0, 0])[None]])
    masks = [{7: _mask([1, 0], [0, 0])}]
    iou = accumulate_iou(act, masks, categories=[7, 99])
    assert 99 in iou
    assert iou[99][0] == pytest.approx(0.0)


def test_perfect_overlap_is_one():
    act = np.stack([_mask([1, 1], [1, 1])[None]])
    masks = [{7: _mask([1, 1], [1, 1])}]
    assert accumulate_iou(act, masks, categories=[7])[7][0] == pytest.approx(1.0)


def test_disjoint_is_zero():
    act = np.stack([_mask([1, 1], [0, 0])[None]])
    masks = [{7: _mask([0, 0], [1, 1])}]
    assert accumulate_iou(act, masks, categories=[7])[7][0] == pytest.approx(0.0)


def test_empty_activation_and_empty_mask_is_zero_not_nan():
    act = np.stack([_mask([0, 0], [0, 0])[None]])
    masks = [{}]
    v = accumulate_iou(act, masks, categories=[7])[7][0]
    assert v == pytest.approx(0.0) and np.isfinite(v)


def test_multiple_slots_are_scored_independently():
    act = np.stack([np.stack([_mask([1, 1], [0, 0]),      # slot 0 hits
                              _mask([0, 0], [1, 1])])])   # slot 1 misses
    masks = [{7: _mask([1, 1], [0, 0])}]
    iou = accumulate_iou(act, masks, categories=[7])
    assert iou[7][0] == pytest.approx(1.0)
    assert iou[7][1] == pytest.approx(0.0)


def test_ratio_is_of_sums_not_a_mean_of_ratios():
    """Bau et al. define a dataset-level ratio. Averaging per-image IoUs gives a
    different number and weights a one-pixel image like a full one."""
    act = np.stack([_mask([1, 1], [1, 1])[None],          # 4 px
                    _mask([1, 0], [0, 0])[None]])         # 1 px
    masks = [{7: _mask([1, 1], [1, 1])}, {7: _mask([0, 0], [0, 1])}]
    got = accumulate_iou(act, masks, categories=[7])[7][0]
    # sums: inter = 4 + 0 = 4 ; union = 4 + 2 = 6
    assert got == pytest.approx(4 / 6)
    per_image_mean = (1.0 + 0.0) / 2
    assert got != pytest.approx(per_image_mean)


def test_shuffled_control_uses_the_same_universe():
    """The control has to be penalised the same way, or it cannot bound the
    real score. Same helper, same categories, so this is structural."""
    act = np.stack([_mask([1, 1], [0, 0])[None], _mask([1, 1], [0, 0])[None]])
    masks = [{7: _mask([1, 1], [0, 0])}, {}]
    real = accumulate_iou(act, masks, categories=[7])
    shuf = accumulate_iou(act[::-1], masks, categories=[7])
    assert set(real) == set(shuf)


def test_accepts_a_category_absent_from_every_image():
    act = np.stack([_mask([1, 1], [1, 1])[None]])
    masks = [{}]
    iou = accumulate_iou(act, masks, categories=[1, 2, 3])
    assert set(iou) == {1, 2, 3}
    assert all(v[0] == pytest.approx(0.0) for v in iou.values())
