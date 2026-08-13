"""F06: pack a flat baseline's bits using a declared geometry, not a constant.

`extract_flat_baseline` packs `codebook_indices` with `group_size=6` hard-coded
and `base_indices` with `group_size=2`. The 6 is a leftover from the 36-bit /
6-slot era, where `36 // 6` happened to be the slot count. It is wrong the
moment the budget changes: at 40 bits `40 % 6 != 0`, so the assertion fires and
the run dies at flat extraction. That is not a hypothetical -- 108 cells were
launched at 40 bit on 2026-08-13 and every one of them was going to die there.

The fix is not `bits // 6`. Inferring the slot count from the bit width is the
same mistake in a different form: 30 bits would infer 5 slots and 48 bits would
infer 8, when the 48-bit panel is 6 slots of 4 bases. The geometry has to be
declared and then checked, `bits == 2 * M * L`.

    panel            bits   slots   bases/slot   bits/group
    main (15-base)     30       5            3            6
    20-base            40       5            4            8
    legacy 18-base     36       6            3            6
    legacy 24-base     48       6            4            8

One further point the audit makes: a flat hashing baseline has no semantic
codebook at all. Its `codebook_indices` are an artificial grouping we impose for
matched analysis, so the manifest must say so rather than let a reader take them
for learned codewords.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from dna_utils.flat_geometry import (  # noqa: E402
    GeometryMismatch,
    FlatGeometry,
    resolve_flat_geometry,
)
from scripts.extract_flat_baseline import pack_bits_to_indices  # noqa: E402


# ------------------------------------------------------------------ geometry

@pytest.mark.parametrize("bits,slots,bases,group", [
    (30, 5, 3, 6),
    (40, 5, 4, 8),
    (36, 6, 3, 6),
    (48, 6, 4, 8),
])
def test_declared_panels(bits, slots, bases, group):
    g = resolve_flat_geometry(bits)
    assert (g.num_codebooks, g.bases_per_codebook) == (slots, bases)
    assert g.bits_per_group == group
    assert g.total_bits == bits


@pytest.mark.parametrize("bits", [30, 36, 40, 48])
def test_the_identity_holds(bits):
    g = resolve_flat_geometry(bits)
    assert g.total_bits == 2 * g.num_codebooks * g.bases_per_codebook
    assert g.bits_per_group == 2 * g.bases_per_codebook


def test_forty_bits_does_not_die_on_the_old_constant():
    """The concrete 2026-08-13 failure: 40 % 6 != 0."""
    assert 40 % 6 != 0
    g = resolve_flat_geometry(40)
    assert 40 % g.bits_per_group == 0


def test_slot_count_is_not_inferred_from_width():
    """`bits // 6` would give 8 slots at 48 bits and 5 at 30. The declared
    panels are 6x4 and 5x3."""
    assert resolve_flat_geometry(48).num_codebooks == 6 != 48 // 6
    assert resolve_flat_geometry(30).num_codebooks == 5


def test_unknown_budget_fails_closed():
    with pytest.raises(GeometryMismatch):
        resolve_flat_geometry(42)


def test_explicit_geometry_must_be_consistent():
    resolve_flat_geometry(40, num_codebooks=5, bases_per_codebook=4)   # ok
    with pytest.raises(GeometryMismatch):
        resolve_flat_geometry(40, num_codebooks=6, bases_per_codebook=4)  # 48 != 40


def test_explicit_geometry_may_differ_from_the_default_panel():
    """A deliberate non-default grouping is allowed as long as it is declared
    and consistent; what is refused is guessing."""
    g = resolve_flat_geometry(48, num_codebooks=8, bases_per_codebook=3)
    assert (g.num_codebooks, g.bases_per_codebook, g.bits_per_group) == (8, 3, 6)


# ------------------------------------------------------------------- packing

@pytest.mark.parametrize("bits", [30, 36, 40, 48])
def test_pack_round_trips_through_bases(bits):
    """Base indices must reconstruct the bits exactly: 2 bits <-> 1 base."""
    rng = np.random.default_rng(0)
    b = rng.integers(0, 2, size=(7, bits)).astype(np.uint8)
    base = pack_bits_to_indices(b, group_size=2)
    assert base.shape == (7, bits // 2)
    hi, lo = base // 2, base % 2
    back = np.empty_like(b)
    back[:, 0::2], back[:, 1::2] = hi, lo
    assert np.array_equal(back, b)


@pytest.mark.parametrize("bits", [30, 36, 40, 48])
def test_pack_with_resolved_group_size_has_the_right_shape(bits):
    g = resolve_flat_geometry(bits)
    rng = np.random.default_rng(1)
    b = rng.integers(0, 2, size=(5, bits)).astype(np.uint8)
    cb = pack_bits_to_indices(b, group_size=g.bits_per_group)
    assert cb.shape == (5, g.num_codebooks)
    assert cb.max() < 2 ** g.bits_per_group


def test_codebook_grouping_agrees_with_base_grouping(bits=40):
    """A codebook index must be exactly the bases of its own slot, so the two
    views of one code cannot disagree."""
    g = resolve_flat_geometry(bits)
    rng = np.random.default_rng(2)
    b = rng.integers(0, 2, size=(4, bits)).astype(np.uint8)
    cb = pack_bits_to_indices(b, group_size=g.bits_per_group)
    base = pack_bits_to_indices(b, group_size=2)
    for m in range(g.num_codebooks):
        seg = base[:, m * g.bases_per_codebook:(m + 1) * g.bases_per_codebook]
        val = np.zeros(len(b), dtype=np.int64)
        for r in range(g.bases_per_codebook):
            val = val * 4 + seg[:, r]
        assert np.array_equal(val, cb[:, m])


def test_manifest_marks_codebooks_as_artificial():
    g = resolve_flat_geometry(30)
    rec = g.as_manifest_record()
    assert rec["codebook_indices_are_artificial_grouping"] is True
    assert rec["total_bits"] == 30 and rec["total_bases"] == 15
