"""Which bit budgets an aggregation is about (F15).

`aggregate_baseline_p0_matrix` holds `BITS = (30, 36, 40, 48)` as a module
constant and has no `--bits`, so the expected matrix always spans four budgets.
A finished 30-bit panel can therefore never be strict-complete: the aggregator
keeps waiting for 36-, 40- and 48-bit cells the paper does not need, and
`--require-paper-eligible` reports 0 eligible out of an expected 432 -- a number
that describes four budgets, not anything that was run.

The same constant is why `tests/test_crh_supervised_matrix.py` fails: its
fixtures expect 8 and 78 cells and observe 16 and 156, exactly double, because
the budget count doubled. Doubling the fixture numbers would hide the cause; the
requested slice should determine the expected set instead.

Also fixed here: the selection metric was registered for 36 and 48 only, so a
30-bit run had no declared metric at all, and the module header claimed
`bit//6` codebooks is "length-generic". It is not -- that is 6.67 at 40 bits,
and F06 established 48 bits as 6 slots of 4 bases. Geometry comes from the
declared panel in `dna_utils.flat_geometry`.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

#: Budgets with a declared geometry, a reviewed GC window and a selection metric.
_KNOWN_BITS: Tuple[int, ...] = (30, 36, 40, 48)

#: The paper's main panel. Asking for nothing means this, not everything ever run.
_DEFAULT_SLICE: Tuple[int, ...] = (30,)

#: Named for the BASE length, since that is what the metric is computed over.
_SELECTION_METRIC = {
    30: "raw_15base_base_hamming_mAP_at_R",
    36: "raw_18base_base_hamming_mAP_at_R",
    40: "raw_20base_base_hamming_mAP_at_R",
    48: "raw_24base_base_hamming_mAP_at_R",
}


class BitSliceInvalid(ValueError):
    """A requested budget has no declared geometry, GC window or metric."""


def resolve_bit_slice(bits: Optional[Iterable[int]]) -> Tuple[int, ...]:
    """Normalise a requested slice; `None` means the paper's main panel."""
    if bits is None:
        return _DEFAULT_SLICE
    out = sorted({int(b) for b in bits})
    if not out:
        return _DEFAULT_SLICE
    unknown = [b for b in out if b not in _KNOWN_BITS]
    if unknown:
        raise BitSliceInvalid(
            f"no declared panel for bit budget(s) {unknown}; known: "
            f"{list(_KNOWN_BITS)}. Add the geometry, the GC window and the "
            f"selection metric deliberately before aggregating it.")
    return tuple(out)


def selection_metric_for_bit(bit: int) -> str:
    b = int(bit)
    if b not in _SELECTION_METRIC:
        raise BitSliceInvalid(
            f"no selection metric registered for {b} bit. A run with no declared "
            f"metric cannot be aggregated as a main result.")
    return _SELECTION_METRIC[b]


def selection_metric_by_bit(bits: Sequence[int]) -> dict:
    """The `selection_metric_by_bit` block, scoped to the requested slice."""
    return {str(b): selection_metric_for_bit(b) for b in bits}


def expected_cell_count(*, bits: Sequence[int], variants: Sequence[str],
                        datasets: Sequence[str], seeds: Sequence[int]) -> int:
    """Cells the requested slice should contain.

    Scoping this by request rather than by a module constant is what lets a
    finished 30-bit panel be complete, and what decouples the CRH fixtures from
    how many budgets happen to be declared.
    """
    return len(bits) * len(variants) * len(datasets) * len(seeds)


def expected_cells(*, bits: Sequence[int], variants: Sequence[str],
                   datasets: Sequence[str], seeds: Sequence[int]) -> list:
    """The explicit (bit, variant, dataset, seed) tuples, for completion checks
    that need to name what is missing rather than only count it."""
    return [(int(b), str(v), str(d), int(s))
            for b in bits for v in variants for d in datasets for s in seeds]
