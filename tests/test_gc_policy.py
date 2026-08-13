"""F07: one GC feasible set, resolved centrally, for every evaluator.

Four different GC conventions were live at once:

    0.40  / 0.60    ours main, baseline projection default, aggregator
    0.4444/ 0.5556  held-out codon decoding, codebook-drop ablations
    0.416 / 0.584   the 20-base launcher and the 24-base baseline evaluator
    L==18 ? .40/.60 : .416/.584   bioproj_effect_newmodel

The reason nobody noticed is worth pinning in a test: **at L=18 all three
fraction pairs collapse to the same integer window [8, 10]**. The legacy budget
hid the disagreement entirely. It only appears at L=15 and L=20 -- exactly the
two budgets the paper now uses -- where 0.40/0.60 gives [6,9] and [8,12] while
0.4444/0.5556 gives [7,8] and [9,11].

D3 fixes the policy as true 40-60% inclusive. What callers must consume is the
INTEGER count range, not the fraction: re-deriving `ceil(frac*L)` in each script
is how the same nominal policy produced different feasible sets.
"""
from __future__ import annotations

import math
import os
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from dna_utils.gc_policy import (  # noqa: E402
    POLICY_VERSION,
    GCPolicy,
    UnsupportedBudget,
    resolve_gc_policy,
)


def test_decided_windows():
    """D3: L=15 -> [6,9], L=20 -> [8,12], homopolymer run <= 3."""
    p15 = resolve_gc_policy(15)
    assert (p15.gc_min_count, p15.gc_max_count) == (6, 9)
    assert p15.max_run == 3
    p20 = resolve_gc_policy(20)
    assert (p20.gc_min_count, p20.gc_max_count) == (8, 12)
    assert p20.max_run == 3


def test_legacy_budgets_still_resolve():
    """18- and 24-base artefacts must remain interpretable; they are kept as
    historical diagnostics, not silently reinterpreted under the new window."""
    assert (resolve_gc_policy(18).gc_min_count,
            resolve_gc_policy(18).gc_max_count) == (8, 10)
    assert (resolve_gc_policy(24).gc_min_count,
            resolve_gc_policy(24).gc_max_count) == (10, 14)


def test_the_disagreement_was_invisible_at_18_bases():
    """Regression guard for the reason this defect survived: at L=18 the three
    historical fraction pairs agree, so no test that only used 18 could catch
    it. At 15 and 20 they do not."""
    def win(L, lo, hi):
        return (math.ceil(lo * L), math.floor(hi * L))
    assert win(18, .40, .60) == win(18, .4444, .5556) == win(18, .416, .584)
    assert win(15, .40, .60) != win(15, .4444, .5556)
    assert win(20, .40, .60) != win(20, .416, .584)


def test_policy_is_derived_from_one_fraction_pair():
    """Every supported budget must come from the same 40-60% rule, so adding a
    budget cannot quietly introduce a fourth convention."""
    for L in (15, 18, 20, 24):
        p = resolve_gc_policy(L)
        assert p.gc_min_frac == pytest.approx(0.40)
        assert p.gc_max_frac == pytest.approx(0.60)
        assert p.gc_min_count == math.ceil(0.40 * L)
        assert p.gc_max_count == math.floor(0.60 * L)


def test_unknown_budget_fails_closed():
    """A budget nobody reviewed must not silently get a window."""
    with pytest.raises(UnsupportedBudget):
        resolve_gc_policy(17)


def test_manifest_record_carries_counts_and_version():
    p = resolve_gc_policy(15)
    rec = p.as_manifest_record()
    assert rec["gc_min_count"] == 6 and rec["gc_max_count"] == 9
    assert rec["max_run"] == 3
    assert rec["total_bases"] == 15
    assert rec["policy_version"] == POLICY_VERSION
    # the fractions are recorded for provenance but the counts are authoritative
    assert "gc_min_frac" in rec and "gc_max_frac" in rec


def test_window_is_non_empty_and_ordered():
    for L in (15, 18, 20, 24):
        p = resolve_gc_policy(L)
        assert 0 <= p.gc_min_count <= p.gc_max_count <= L


def test_policy_matches_the_projection_backend():
    """The counts this policy hands out must be the ones the DP projection
    actually enforces, otherwise the manifest would describe a feasible set the
    codes were never projected onto."""
    from dna_utils.bio_constraints import _resolve_gc_count_range
    for L in (15, 18, 20, 24):
        p = resolve_gc_policy(L)
        assert _resolve_gc_count_range(L, p.gc_min_frac, p.gc_max_frac) == (
            p.gc_min_count, p.gc_max_count)


def test_equality_and_hash_are_by_value():
    assert resolve_gc_policy(15) == resolve_gc_policy(15)
    assert resolve_gc_policy(15) != resolve_gc_policy(20)
    assert isinstance(hash(resolve_gc_policy(15)), int)


def test_policy_is_immutable():
    p = resolve_gc_policy(15)
    with pytest.raises(Exception):
        p.gc_min_count = 7  # type: ignore[misc]


def test_geometry_helper_agrees_with_budget():
    """M slots x L bases per slot must give the same policy as the total."""
    from dna_utils.gc_policy import resolve_gc_policy_for_geometry
    assert resolve_gc_policy_for_geometry(num_slots=5, bases_per_slot=3) == \
        resolve_gc_policy(15)
    assert resolve_gc_policy_for_geometry(num_slots=5, bases_per_slot=4) == \
        resolve_gc_policy(20)
    assert resolve_gc_policy_for_geometry(num_slots=6, bases_per_slot=3) == \
        resolve_gc_policy(18)


def test_GCPolicy_is_the_only_construction_path():
    """Constructing a window by hand must not be mistaken for a reviewed one."""
    hand = GCPolicy(total_bases=15, gc_min_count=7, gc_max_count=8,
                    max_run=3, gc_min_frac=0.4444, gc_max_frac=0.5556,
                    policy_version="hand-rolled")
    assert hand != resolve_gc_policy(15)
    assert hand.policy_version != POLICY_VERSION
