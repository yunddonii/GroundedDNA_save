"""F15: the aggregator must be able to complete on the slice you asked for.

`BITS = (30, 36, 40, 48)` is a module constant and there is no `--bits`, so the
expected matrix always spans four budgets. Finishing the 30-bit panel therefore
can never be "strict complete": the aggregator keeps waiting for 36, 40 and 48
cells that the paper does not need. `--require-paper-eligible` exits 3 against
an expected 432 with 0 eligible, and 432 is the four-budget expectation rather
than anything that was run.

The same constant broke `tests/test_crh_supervised_matrix.py`, whose fixtures
expect 8 and 78 cells and now observe 16 and 156 -- exactly double, because the
budget count doubled. The right fix is for the requested slice to determine the
expected set, not for the fixture numbers to be doubled by hand.

There is also a geometry claim in the module header worth pinning: it says
`bit//6` codebooks is length-generic. It is not. At 40 bits that is 6.67, and
F06 established that 48 bits is 6 slots of 4 bases, not 8 slots of 3. The
aggregator must take geometry from the declared panel.
"""
from __future__ import annotations

import os
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from dna_utils.bit_slice import (  # noqa: E402
    BitSliceInvalid,
    expected_cell_count,
    resolve_bit_slice,
    selection_metric_for_bit,
)


def test_default_slice_is_the_paper_budget():
    """Asking for nothing should mean the 15-base main panel, not all four."""
    assert resolve_bit_slice(None) == (30,)


def test_explicit_slice_is_honoured():
    assert resolve_bit_slice([30]) == (30,)
    assert resolve_bit_slice([30, 40]) == (30, 40)


def test_slice_is_sorted_and_deduped():
    assert resolve_bit_slice([40, 30, 40]) == (30, 40)


def test_unknown_budget_is_rejected():
    with pytest.raises(BitSliceInvalid):
        resolve_bit_slice([32])


def test_expected_count_scales_with_the_slice_not_the_constant():
    """The CRH fixture observed 16 where it expected 8 because the constant grew
    from two budgets to four. Scoping by request removes that coupling."""
    one = expected_cell_count(bits=(30,), variants=("cibhash",),
                              datasets=("Flickr25k", "MSCOCO"), seeds=(42,))
    assert one == 2
    two = expected_cell_count(bits=(30, 40), variants=("cibhash",),
                              datasets=("Flickr25k", "MSCOCO"), seeds=(42,))
    assert two == 4
    assert two == 2 * one


def test_expected_count_covers_seeds():
    n = expected_cell_count(bits=(30,), variants=("cibhash", "oh"),
                            datasets=("Flickr25k",), seeds=(42, 43, 44))
    assert n == 6


def test_a_completed_30bit_panel_can_be_strict_complete():
    """The point of the fix: 9 methods x 4 datasets x 3 seeds at 30 bit is a
    complete slice, and must be reportable as such."""
    n = expected_cell_count(
        bits=(30,),
        variants=("cibhash", "cimon", "mls3rduh", "greedyhash", "bihalf",
                  "sdc-paper", "oh", "hhch", "crovca"),
        datasets=("Flickr25k", "MSCOCO", "NUSWIDE", "CIFAR10"),
        seeds=(42, 43, 44))
    assert n == 108


# ------------------------------------------------------------ selection metric

@pytest.mark.parametrize("bit,expected", [
    (30, "raw_15base_base_hamming_mAP_at_R"),
    (36, "raw_18base_base_hamming_mAP_at_R"),
    (40, "raw_20base_base_hamming_mAP_at_R"),
    (48, "raw_24base_base_hamming_mAP_at_R"),
])
def test_selection_metric_is_named_for_the_base_length(bit, expected):
    """Only 36 and 48 were registered, so a 30-bit run had no declared selection
    metric at all -- the protocol metadata simply omitted it."""
    assert selection_metric_for_bit(bit) == expected


def test_selection_metric_rejects_an_unregistered_budget():
    with pytest.raises(BitSliceInvalid):
        selection_metric_for_bit(32)


# ------------------------------------------------------------------- geometry

@pytest.mark.parametrize("bit,slots,bases", [
    (30, 5, 3), (36, 6, 3), (40, 5, 4), (48, 6, 4),
])
def test_geometry_comes_from_the_declared_panel(bit, slots, bases):
    """`bit//6` gives 6.67 at 40 bits and 8 at 48. F06 settled that 48 is 6x4."""
    from dna_utils.flat_geometry import resolve_flat_geometry
    g = resolve_flat_geometry(bit)
    assert (g.num_codebooks, g.bases_per_codebook) == (slots, bases)
    assert bit // 6 != g.num_codebooks or bit in (30, 36)


def test_forty_bits_would_break_the_old_inference():
    assert 40 // 6 == 6 and 40 / 6 != 6.0      # not integral, so bit//6 is wrong
    from dna_utils.flat_geometry import resolve_flat_geometry
    assert resolve_flat_geometry(40).num_codebooks == 5


# ------------------------------------------------------------ gc consistency

@pytest.mark.parametrize("bit", [30, 36, 40, 48])
def test_each_slice_budget_has_a_reviewed_gc_window(bit):
    """F07 and F15 have to agree: every budget the aggregator will accept must
    already have a central GC policy, or results would be projected onto a
    window nobody reviewed."""
    from dna_utils.flat_geometry import resolve_flat_geometry
    from dna_utils.gc_policy import resolve_gc_policy
    g = resolve_flat_geometry(bit)
    pol = resolve_gc_policy(g.total_bases)
    assert pol.gc_min_count <= pol.gc_max_count


# ------------------------------------------------- aggregator wiring (F15)

def _agg():
    import scripts.aggregate_baseline_p0_matrix as agg
    return agg


def test_expected_keys_default_to_the_paper_slice():
    """Previously the expected matrix always spanned four budgets, so a
    finished 30-bit panel could never be strict-complete."""
    agg = _agg()
    keys = agg._expected_keys((42,))
    assert {k.bit for k in keys} == {30}


def test_expected_keys_honour_the_requested_slice():
    agg = _agg()
    keys = agg._expected_keys((42,), agg.DEFAULT_PANELS, (36, 48))
    assert {k.bit for k in keys} == {36, 48}
    assert len(keys) == 78          # the historical U0+U2 single-seed count


def test_expected_keys_scale_linearly_in_the_slice():
    agg = _agg()
    one = agg._expected_keys((42,), ("supervised",), (36,))
    two = agg._expected_keys((42,), ("supervised",), (36, 48))
    assert len(two) == 2 * len(one) == 8


def test_aggregator_exposes_a_bits_flag():
    """Without a CLI flag the slice cannot be requested at all."""
    import subprocess
    import sys
    out = subprocess.run(
        [sys.executable, os.path.join(_REPO, "scripts",
                                      "aggregate_baseline_p0_matrix.py"),
         "--help"], capture_output=True, text=True, cwd=_REPO, timeout=120)
    assert "--bits" in out.stdout


# ------------------------------------------- bit-scoped source exemption

def test_flat_baseline_transition_is_registered_for_the_current_source():
    """F06 changed `scripts/extract_flat_baseline.py`; an unregistered edit
    would classify every future cell as unknown implementation drift."""
    import hashlib
    import os
    agg = _agg()
    entry = agg.KNOWN_NON_SCIENTIFIC_IMPLEMENTATION_TRANSITIONS[
        "scripts/extract_flat_baseline.py"]
    path = os.path.join(_REPO, "scripts", "extract_flat_baseline.py")
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    assert entry["after_sha256"] == digest
    assert digest in entry["reviewed_sha256"]


def test_forty_eight_bit_cells_are_scoped_out_of_the_geometry_exemption():
    """The old code packed 48 bits as 8 groups of 6; the declared panel is 6
    slots of 4 bases. Those cells must NOT ride in on an exemption written for
    30 and 36, where the change is a genuine no-op."""
    agg = _agg()
    entry = agg.KNOWN_NON_SCIENTIFIC_IMPLEMENTATION_TRANSITIONS[
        "scripts/extract_flat_baseline.py"]
    scopes = entry["non_scientific_bits_by_sha256"]
    assert all(48 not in bits for bits in scopes.values())
    assert 36 in scopes[
        "02c805b0ce914daf8d54469f9a9369cf1cd96cacba55367ad8da1adff8284c4f"]
    assert 30 in scopes[
        "433be227bf28131a3266e298e37b7a7fcab5e11d41005b75629d6904f81d990e"]


def test_every_registered_classification_clears_the_variant_gate():
    """The path audit and the per-variant heterogeneity gate must agree.

    While `non_scientific_bit_budget_extension` was missing from the gate's
    allow-list, a 30-bit matrix spanning the seed-42 and seed-43/44 roots
    reported 72 of 108 cells implementation-blocked even though every audited
    path came back as a reviewed, non-scientific transition.
    """
    agg = _agg()
    registered = {
        entry["classification"]
        for entry in
        agg.KNOWN_NON_SCIENTIFIC_IMPLEMENTATION_TRANSITIONS.values()
    }
    assert registered <= agg.NON_SCIENTIFIC_TRANSITION_CLASSIFICATIONS
