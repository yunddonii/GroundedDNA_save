"""The single source of the biological feasible set (F07).

Four GC conventions were live at once -- `0.40/0.60` in the main path and the
aggregator, `0.4444/0.5556` in held-out decoding and the codebook-drop
ablations, `0.416/0.584` in the 20-base launcher and the 24-base baseline
evaluator, and a length-conditional mix in `bioproj_effect_newmodel`. Comparing
ours against a baseline under two different feasible sets is not a comparison,
and a projection cost measured under one window cannot be quoted for the other.

The disagreement went unnoticed because **at L=18 all three fraction pairs give
the same integer window [8, 10]**. It only separates at L=15 and L=20, the two
budgets the paper now uses.

D3 (2026-08-13) fixes the policy as **true 40-60% inclusive**, which is also the
projection default the baselines already used:

    L = 15  ->  GC in [6, 9]      homopolymer run <= 3
    L = 20  ->  GC in [8, 12]     homopolymer run <= 3

WHAT CALLERS MUST CONSUME. The integer counts, not the fractions. Re-deriving
`ceil(frac * L)` inside each script is precisely how one nominal policy produced
several feasible sets; a caller that needs to talk to the projection backend
should pass `gc_min_frac`/`gc_max_frac` from here rather than literals, and
record `as_manifest_record()` so a result can be traced to the window it was
actually projected onto.

Unknown budgets raise rather than receiving an interpolated window: a budget
nobody reviewed should not silently acquire a feasible set.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, Mapping

#: Bump when the decided window changes. Recorded in every manifest so a stored
#: result cannot be silently reinterpreted under a later policy.
POLICY_VERSION = "gc-40-60-inclusive-v1"

#: The one rule. Every supported budget is derived from it, so adding a budget
#: cannot introduce a fourth convention by hand.
_GC_MIN_FRAC = 0.40
_GC_MAX_FRAC = 0.60
_MAX_RUN = 3

#: Budgets that have been reviewed. 15 and 20 are the paper's; 18 and 24 are
#: kept so legacy artefacts stay interpretable as historical diagnostics --
#: NOT so they can be reinterpreted under the new window.
_SUPPORTED_BASES = (15, 18, 20, 24)


class UnsupportedBudget(ValueError):
    """Raised instead of handing out an unreviewed feasible set."""


@dataclass(frozen=True)
class GCPolicy:
    total_bases: int
    gc_min_count: int
    gc_max_count: int
    max_run: int
    gc_min_frac: float
    gc_max_frac: float
    policy_version: str

    def as_manifest_record(self) -> Dict[str, Any]:
        """What every result manifest must carry. The counts are authoritative;
        the fractions are recorded for provenance only."""
        return dict(asdict(self))

    def matches(self, other: Mapping[str, Any]) -> bool:
        """True when a stored manifest record describes this same feasible set.
        Compares counts and version, not fractions, because the counts are what
        the projection enforced."""
        return (
            int(other.get("total_bases", -1)) == self.total_bases
            and int(other.get("gc_min_count", -1)) == self.gc_min_count
            and int(other.get("gc_max_count", -1)) == self.gc_max_count
            and int(other.get("max_run", -1)) == self.max_run
            and str(other.get("policy_version", "")) == self.policy_version
        )


def resolve_gc_policy(total_bases: int) -> GCPolicy:
    """The feasible set for a code of `total_bases` bases."""
    L = int(total_bases)
    if L not in _SUPPORTED_BASES:
        raise UnsupportedBudget(
            f"no reviewed GC policy for {L} bases; supported: "
            f"{list(_SUPPORTED_BASES)}. Add it deliberately -- deriving a window "
            f"on the fly is how four conventions ended up in the codebase.")
    return GCPolicy(
        total_bases=L,
        gc_min_count=math.ceil(_GC_MIN_FRAC * L),
        gc_max_count=math.floor(_GC_MAX_FRAC * L),
        max_run=_MAX_RUN,
        gc_min_frac=_GC_MIN_FRAC,
        gc_max_frac=_GC_MAX_FRAC,
        policy_version=POLICY_VERSION,
    )


def resolve_gc_policy_for_geometry(*, num_slots: int, bases_per_slot: int) -> GCPolicy:
    """Convenience for callers that hold the geometry rather than the total."""
    return resolve_gc_policy(int(num_slots) * int(bases_per_slot))


def assert_manifest_policy(record: Mapping[str, Any], total_bases: int) -> GCPolicy:
    """Fail loudly when a stored result was projected under a different window
    than the one now in force, rather than silently comparing across the two."""
    expected = resolve_gc_policy(total_bases)
    if not expected.matches(record):
        raise UnsupportedBudget(
            f"stored GC policy {dict(record)} does not match the current policy "
            f"{expected.as_manifest_record()}. These results were projected onto "
            f"a different feasible set and must not be compared or aggregated "
            f"with current ones.")
    return expected
