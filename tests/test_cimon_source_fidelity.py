"""F05: CIMON's objective must match the official implementation exactly.

Checked against the official repository at the exact commit the audit names:

    https://github.com/luoxiao12/CIMON
    commit 4107234f87dc832819d03f5eba922e82b73d714b, cimon.py

Two divergences were found, both inside `generate_similarity_weight_matrix`:

1. **Histogram search range.** Official: `interval = 1./100` and
   `for i in range(100)`, i.e. 100 bins over [0, 1). The local copy ran
   `range(200)`, i.e. 200 bins over [0, 2), so a mode anywhere above 1.0 could
   be selected as `max_cos`. Every downstream quantity -- the left/right split,
   both mirrored Gaussians, and therefore `weight_1` -- is computed from
   `max_cos`, so the whole confidence weighting shifts.

2. **`weight_2` was off by a factor of two.** Official:

       weight_2 = ((((A - A.T) == 0) - 1/2) * 2 * S + 1) / 2      # {0, 1}

   Local dropped the trailing `/ 2`, giving values in {0, 2}. `W = weight_1 *
   weight_2` feeds all four SEM-CON terms, so those four terms doubled while
   `eta * nce_loss` did not -- the objective's relative weighting changed.
   The loss assembly itself is byte-identical to the official line, which is
   exactly why the error was invisible: nothing about the code around it looks
   wrong.

The remaining ~60 lines of that function (cosine distance, S, the mirrored
left/right reconstruction, `norm.fit`, `weight_norm`, the clipped `weight_1`,
spectral clustering with `random_state=0` and `assign_labels="discretize"`)
match the official source line for line, as do `SEM_CON_Loss.forward` and the
per-batch loss assembly.
"""
from __future__ import annotations

import inspect
import os
import sys

import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from baseline.CIMON import (  # noqa: E402
    CIMON_OFFICIAL_COMMIT,
    CIMON_OFFICIAL_REPO,
    CIMON_HISTOGRAM_BINS,
    CIMON_HISTOGRAM_INTERVAL,
    SEM_CON_Loss,
    _cluster_agreement_weight,
    _max_density_cosine,
    generate_similarity_weight_matrix,
)


# --------------------------------------------------------------- weight_2

def _official_weight_2(A: np.ndarray, S: np.ndarray) -> np.ndarray:
    """The official line, transcribed verbatim from cimon.py:201-202."""
    return ((((A - A.T) == 0) - 1 / 2) * 2 * S + 1) / 2


def test_weight_2_matches_the_official_golden_tensor():
    """Small fixed A and S, compared exactly against the official expression."""
    A = np.array([[0, 0, 1, 1]])
    S = np.array([
        [1.0, 1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0, 1.0],
        [-1.0, -1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0, 1.0],
    ])
    got = _cluster_agreement_weight(A, S)
    np.testing.assert_array_equal(got, _official_weight_2(A, S))
    # Golden values, written out so a regression is readable rather than merely
    # "not equal". The rule is agreement XNOR similarity: same cluster and
    # S=+1 -> 1, same cluster and S=-1 -> 0, different cluster and S=+1 -> 0,
    # different cluster and S=-1 -> 1.
    np.testing.assert_array_equal(got, np.array([
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 1.0],
    ]))


def test_weight_2_is_zero_one_not_zero_two():
    """The defect in one assertion: the local copy produced {0, 2}, which
    doubled all four SEM-CON terms against an unchanged `eta * nce_loss`."""
    A = np.array([[0, 0, 1]])
    S = np.array([[1.0, 1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]])
    got = _cluster_agreement_weight(A, S)
    assert set(np.unique(got)) <= {0.0, 1.0}
    assert got.max() == 1.0
    # The pre-fix expression, which is exactly twice the official one.
    local_before_fix = ((((A - A.T) == 0) - 1 / 2) * 2 * S + 1)
    np.testing.assert_array_equal(local_before_fix, got * 2)
    assert local_before_fix.max() == 2.0


@pytest.mark.parametrize("same_cluster,similar,expected", [
    (True, 1.0, 1.0),     # agree, similar     -> full weight
    (True, -1.0, 0.0),    # agree, dissimilar  -> zero
    (False, 1.0, 0.0),    # disagree, similar  -> zero
    (False, -1.0, 1.0),   # disagree, dissimilar -> full weight
])
def test_weight_2_truth_table(same_cluster, similar, expected):
    A = np.array([[0, 0 if same_cluster else 1]])
    S = np.array([[1.0, similar], [similar, 1.0]])
    assert _cluster_agreement_weight(A, S)[0, 1] == pytest.approx(expected)


# ------------------------------------------------------------- histogram

def test_histogram_is_one_hundred_bins_over_zero_to_one():
    assert CIMON_HISTOGRAM_BINS == 100
    assert CIMON_HISTOGRAM_INTERVAL == pytest.approx(1.0 / 100)
    # The searched range is [0, 1), not [0, 2).
    assert CIMON_HISTOGRAM_BINS * CIMON_HISTOGRAM_INTERVAL == pytest.approx(1.0)


def test_mode_above_one_cannot_be_selected():
    """Cosine distance ranges over [0, 2], but the official search stops at 1.
    With the local 200-bin loop, a mass at 1.30 was eligible to become
    `max_cos`, which then defines the left/right split and both Gaussians."""
    cos_dist = np.concatenate([
        np.full(1000, 1.305),      # dominant mass, outside the official range
        np.full(50, 0.205),        # the official mode
    ])
    assert _max_density_cosine(cos_dist) == pytest.approx(0.20, abs=1e-9)


def test_mode_is_the_densest_bin_within_range():
    cos_dist = np.concatenate([np.full(300, 0.635), np.full(100, 0.115)])
    assert _max_density_cosine(cos_dist) == pytest.approx(0.63, abs=1e-9)


def test_two_hundred_bin_search_would_have_differed():
    """Pin the magnitude of the defect: the old range picks a different mode
    on the same input, so this is not a cosmetic change."""
    cos_dist = np.concatenate([np.full(1000, 1.305), np.full(50, 0.205)])
    official = _max_density_cosine(cos_dist)
    old = _max_density_cosine(cos_dist, bins=200)
    assert official == pytest.approx(0.20, abs=1e-9)
    assert old == pytest.approx(1.30, abs=1e-9)
    assert official != old


# ------------------------------------------------- end-to-end through W

def test_generated_weights_stay_within_zero_one():
    """`W = weight_1 * weight_2` with weight_1 clipped to [0, 1] and weight_2
    in {0, 1}. Under the defect W reached 2."""
    rng = np.random.default_rng(0)
    features = rng.normal(size=(24, 8)).astype(np.float32)
    import torch
    S, W = generate_similarity_weight_matrix(
        torch.from_numpy(features), threshold=0.1, num_clusters=3)
    W = W.numpy()
    assert W.min() >= 0.0
    assert W.max() <= 1.0


# --------------------------------------------------- untouched contracts

def test_sem_con_loss_matches_the_official_forward():
    import torch
    source = inspect.getsource(SEM_CON_Loss.forward)
    assert "(W * S.abs() * (H - S).pow(2)).sum() / (H.shape[0] ** 2)" in source
    H = torch.tensor([[1.0, 0.5], [0.5, 1.0]])
    W = torch.tensor([[1.0, 0.5], [0.5, 1.0]])
    S = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
    expected = (W * S.abs() * (H - S).pow(2)).sum() / 4
    assert SEM_CON_Loss()(H, W, S).item() == pytest.approx(expected.item())


def test_official_provenance_is_recorded_in_the_source():
    """The audit requires the exact upstream commit to be recorded, so a future
    reader can re-diff rather than re-derive."""
    assert CIMON_OFFICIAL_REPO == "https://github.com/luoxiao12/CIMON"
    assert CIMON_OFFICIAL_COMMIT == (
        "4107234f87dc832819d03f5eba922e82b73d714b")
