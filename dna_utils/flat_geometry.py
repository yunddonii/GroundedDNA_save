"""Declared code geometry for flat hashing baselines (F06).

`extract_flat_baseline` packed `codebook_indices` with `group_size=6` written as
a literal. The 6 came from the 36-bit / 6-slot era, where `36 // 6` coincided
with the slot count. It breaks as soon as the budget moves: `40 % 6 != 0`, so a
40-bit run dies inside `pack_bits_to_indices`. 108 cells were launched at 40 bit
on 2026-08-13 and every one was going to fail there.

Replacing the constant with `bits // 6` would be the same mistake wearing a
different hat -- it would read 48 bits as 8 slots when the 48-bit panel is 6
slots of 4 bases, and it silently invents a grouping nobody declared. So the
geometry is declared per budget and then CHECKED against `bits == 2 * M * L`:

    panel            bits   slots   bases/slot   bits/group
    main (15-base)     30       5            3            6
    20-base            40       5            4            8
    legacy 18-base     36       6            3            6
    legacy 24-base     48       6            4            8

A caller may pass an explicit geometry that differs from the default panel; what
is refused is guessing. Unknown budgets raise.

ONE HONEST CAVEAT, which the manifest carries. A flat hashing baseline has no
semantic codebook. Its `codebook_indices` are an artificial grouping we impose so
that slot-wise analyses can run on both sides; `base_indices` is the
authoritative representation. Downstream code that treats these as learned
codewords is reading something into them that is not there.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

#: bits -> (num_codebooks, bases_per_codebook)
_PANELS = {
    30: (5, 3),
    36: (6, 3),
    40: (5, 4),
    48: (6, 4),
}


class GeometryMismatch(ValueError):
    """Raised instead of inventing or guessing a grouping."""


@dataclass(frozen=True)
class FlatGeometry:
    total_bits: int
    num_codebooks: int
    bases_per_codebook: int

    @property
    def bits_per_group(self) -> int:
        return 2 * self.bases_per_codebook

    @property
    def total_bases(self) -> int:
        return self.num_codebooks * self.bases_per_codebook

    def as_manifest_record(self) -> Dict[str, Any]:
        d = asdict(self)
        d.update({
            "bits_per_group": self.bits_per_group,
            "total_bases": self.total_bases,
            # A flat baseline has no learned codebook; this grouping exists so
            # slot-wise analyses can run on both sides of the comparison.
            "codebook_indices_are_artificial_grouping": True,
        })
        return d


def resolve_flat_geometry(
    total_bits: int, *,
    num_codebooks: Optional[int] = None,
    bases_per_codebook: Optional[int] = None,
) -> FlatGeometry:
    """Declared geometry for `total_bits`, verified against 2*M*L."""
    bits = int(total_bits)
    if num_codebooks is None and bases_per_codebook is None:
        if bits not in _PANELS:
            raise GeometryMismatch(
                f"no declared panel for {bits} bits; known: {sorted(_PANELS)}. "
                f"Pass num_codebooks and bases_per_codebook explicitly, or add "
                f"the panel deliberately -- inferring the slot count from the "
                f"bit width is what made 40 bits crash and would read 48 bits "
                f"as 8 slots.")
        m, l = _PANELS[bits]
    else:
        if num_codebooks is None or bases_per_codebook is None:
            raise GeometryMismatch(
                "give both num_codebooks and bases_per_codebook, or neither")
        m, l = int(num_codebooks), int(bases_per_codebook)

    if 2 * m * l != bits:
        raise GeometryMismatch(
            f"geometry {m} codebooks x {l} bases = {2 * m * l} bits, but the "
            f"code is {bits} bits. 2*M*L must equal the bit width.")
    if bits % (2 * l) != 0:
        raise GeometryMismatch(
            f"{bits} bits is not divisible by the group size {2 * l}")
    return FlatGeometry(total_bits=bits, num_codebooks=m, bases_per_codebook=l)
