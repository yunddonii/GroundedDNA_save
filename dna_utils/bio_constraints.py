"""DNA bio-constraint validators + minimum-edit projection.

For DNA-storage applications the synthesizer / sequencer typically requires:

    GC content        :  fraction of {C, G} bases in the strand falls in
                          ``[gc_min, gc_max]``.  Default range here is
                          ``[40%, 60%]`` which on an 18-base code maps to
                          ``GC count in [7, 11]``.
    Homopolymer       :  no run of identical bases longer than ``max_run``
                          (default 3 -- i.e. ``AAAA`` / ``GGGG`` are bad).

Post-processing protocol (see project notes):

    1) Detect whether each extracted DNA code violates either constraint.
    2) For violating codes, compute the *Hamming-minimum* valid replacement
       via dynamic programming. Edits stay local; valid codes are untouched.
    3) Re-run retrieval on the projected codes and compare mAP.

This file is the (1)+(2) layer. Step (3) is wired in `evaluation_siglip2.py`
behind the ``--bio_project`` CLI flag.
"""

from __future__ import annotations
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


# ----------------------------------------------------------------- defaults

# Base index convention -- MUST match `dna_code_utils.BASE_TO_BITS`:
#   A=0  C=1  G=2  T=3
BASE_A, BASE_C, BASE_G, BASE_T = 0, 1, 2, 3
NUM_BASES = 4

_IS_GC = np.array([0, 1, 1, 0], dtype=np.int8)   # [4] -- 1 for C / G

DEFAULT_GC_MIN_FRAC: float = 0.40
DEFAULT_GC_MAX_FRAC: float = 0.60
DEFAULT_MAX_HOMOPOLYMER_RUN: int = 3


def _resolve_gc_count_range(L: int, gc_min_frac: float, gc_max_frac: float) -> Tuple[int, int]:
    """Convert fractional GC bounds to integer counts for length L."""
    gc_min = int(np.ceil (gc_min_frac * L))
    gc_max = int(np.floor(gc_max_frac * L))
    return gc_min, gc_max


# ----------------------------------------------------------------- validators

def is_valid(
    code: np.ndarray,
    gc_min_frac: float = DEFAULT_GC_MIN_FRAC,
    gc_max_frac: float = DEFAULT_GC_MAX_FRAC,
    max_run:     int   = DEFAULT_MAX_HOMOPOLYMER_RUN,
) -> bool:
    """Single-code validator.

    code : ``[L]`` int array, values in ``{0, 1, 2, 3}``.
    """
    c = np.asarray(code, dtype=np.int8)
    L = c.shape[-1]
    gc_min, gc_max = _resolve_gc_count_range(L, gc_min_frac, gc_max_frac)
    # GC content
    gc = int(_IS_GC[c].sum())
    if gc < gc_min or gc > gc_max:
        return False
    # Homopolymer
    run = 1
    for i in range(1, L):
        if c[i] == c[i - 1]:
            run += 1
            if run > max_run:
                return False
        else:
            run = 1
    return True


def violation_report(
    code: np.ndarray,
    gc_min_frac: float = DEFAULT_GC_MIN_FRAC,
    gc_max_frac: float = DEFAULT_GC_MAX_FRAC,
    max_run:     int   = DEFAULT_MAX_HOMOPOLYMER_RUN,
) -> Dict[str, int]:
    """Itemize what's wrong with `code` (zeros if all good)."""
    c = np.asarray(code, dtype=np.int8)
    L = c.shape[-1]
    gc_min, gc_max = _resolve_gc_count_range(L, gc_min_frac, gc_max_frac)
    gc = int(_IS_GC[c].sum())
    gc_under = max(0, gc_min - gc)
    gc_over  = max(0, gc - gc_max)
    # count distinct illegal runs (each contiguous run >max_run counts once)
    homo_runs = 0
    run = 1
    in_bad = False
    for i in range(1, L):
        if c[i] == c[i - 1]:
            run += 1
            if run == max_run + 1 and not in_bad:
                homo_runs += 1
                in_bad = True
        else:
            run = 1
            in_bad = False
    return {
        "gc_count":        gc,
        "gc_min_count":    gc_min,
        "gc_max_count":    gc_max,
        "gc_under":        gc_under,
        "gc_over":         gc_over,
        "homopolymer_runs_over_limit": homo_runs,
    }


def is_valid_batch(
    codes: np.ndarray,
    gc_min_frac: float = DEFAULT_GC_MIN_FRAC,
    gc_max_frac: float = DEFAULT_GC_MAX_FRAC,
    max_run:     int   = DEFAULT_MAX_HOMOPOLYMER_RUN,
) -> np.ndarray:
    """``[N, L] -> [N] bool``. Vectorized over the batch dim only."""
    arr = np.asarray(codes, dtype=np.int8)
    assert arr.ndim == 2, f"expected [N, L], got {arr.shape}"
    N, L = arr.shape
    gc_min, gc_max = _resolve_gc_count_range(L, gc_min_frac, gc_max_frac)
    gc = _IS_GC[arr].sum(axis=1)                                  # [N]
    gc_ok = (gc >= gc_min) & (gc <= gc_max)                       # [N]
    # homopolymer: any run > max_run anywhere
    same_as_prev = (arr[:, 1:] == arr[:, :-1]).astype(np.int32)   # [N, L-1]
    # cumulative run length: reset at 1 wherever transition; +1 wherever same
    run = np.ones((N, L), dtype=np.int32)
    for i in range(1, L):
        run[:, i] = np.where(same_as_prev[:, i - 1] == 1, run[:, i - 1] + 1, 1)
    homo_ok = (run.max(axis=1) <= max_run)                        # [N]
    return gc_ok & homo_ok


# ----------------------------------------------------------------- DP projection

def project_to_valid(
    code: np.ndarray,
    gc_min_frac: float = DEFAULT_GC_MIN_FRAC,
    gc_max_frac: float = DEFAULT_GC_MAX_FRAC,
    max_run:     int   = DEFAULT_MAX_HOMOPOLYMER_RUN,
) -> Tuple[np.ndarray, int]:
    """Hamming-minimum valid projection of a SINGLE DNA code via DP.

    Method:
        State at position ``i`` is ``(last_base, run_length, gc_count_so_far)``.
        Transition to base ``b'`` at position ``i+1``:
            run' = run + 1 if b' == last_base else 1   (must be <= max_run)
            gc'  = gc + 1 if b' in {C, G} else gc
            cost = 0 if b' == code[i+1] else 1
        Final accept condition: gc_count_total in [gc_min_count, gc_max_count].
        Return the trajectory minimizing total edit cost.

    Args:
        code : np.ndarray [L]   int values in {0, 1, 2, 3}

    Returns:
        (proj_code, edit_distance)
            proj_code : np.ndarray [L] int8, valid w.r.t. the constraints
            edit_distance : non-negative int (Hamming distance from original).
            edit_distance == 0  -> input was already valid (returned as-is).
            edit_distance == -1 -> no valid trajectory (e.g. impossible
                                   constraint range); original returned unchanged.
    """
    c = np.asarray(code, dtype=np.int8)
    assert c.ndim == 1, f"expected [L], got {c.shape}"
    L = c.shape[0]
    if is_valid(c, gc_min_frac, gc_max_frac, max_run):
        return c.copy(), 0

    gc_min, gc_max = _resolve_gc_count_range(L, gc_min_frac, gc_max_frac)
    INF = 10 ** 9

    # dp[b][r][g] = min edits to land at end-of-position-i in state (b, r, g)
    R = max_run + 1                     # run index 0..max_run; we use 1..max_run
    G = L + 1                           # gc count 0..L
    dp = np.full((NUM_BASES, R, G), INF, dtype=np.int32)
    # parent[i][b][r][g] = (prev_b, prev_r, prev_g) -- backtrack pointers
    # i ranges over 1..L-1 (transitions from position i-1 to i)
    parent = np.full((L, NUM_BASES, R, G, 3), -1, dtype=np.int32)

    # ---- initialize at position 0 ----
    for b in range(NUM_BASES):
        cost = 0 if int(c[0]) == b else 1
        g0 = int(_IS_GC[b])
        if g0 < G:
            dp[b][1][g0] = cost

    # ---- forward DP ----
    for i in range(1, L):
        new_dp = np.full((NUM_BASES, R, G), INF, dtype=np.int32)
        for b in range(NUM_BASES):
            for r in range(1, max_run + 1):
                for g in range(G):
                    base_cost = dp[b][r][g]
                    if base_cost >= INF:
                        continue
                    for bp in range(NUM_BASES):
                        if bp == b:
                            rp = r + 1
                            if rp > max_run:
                                continue
                        else:
                            rp = 1
                        gp = g + int(_IS_GC[bp])
                        if gp >= G:
                            continue
                        step = 0 if int(c[i]) == bp else 1
                        cand = base_cost + step
                        if cand < new_dp[bp][rp][gp]:
                            new_dp[bp][rp][gp] = cand
                            parent[i][bp][rp][gp][0] = b
                            parent[i][bp][rp][gp][1] = r
                            parent[i][bp][rp][gp][2] = g
        dp = new_dp

    # ---- pick the best terminal state with gc in [gc_min, gc_max] ----
    best_cost = INF
    best_state: Optional[Tuple[int, int, int]] = None
    for b in range(NUM_BASES):
        for r in range(1, max_run + 1):
            for g in range(gc_min, gc_max + 1):
                if dp[b][r][g] < best_cost:
                    best_cost = int(dp[b][r][g])
                    best_state = (b, r, g)

    if best_state is None:
        return c.copy(), -1

    # ---- backtrack ----
    out = np.zeros(L, dtype=np.int8)
    b, r, g = best_state
    out[L - 1] = b
    for i in range(L - 1, 0, -1):
        prev = parent[i][b][r][g]
        prev_b, prev_r, prev_g = int(prev[0]), int(prev[1]), int(prev[2])
        out[i - 1] = prev_b
        b, r, g = prev_b, prev_r, prev_g
    return out, best_cost


def batch_project_to_valid(
    codes: np.ndarray,
    gc_min_frac: float = DEFAULT_GC_MIN_FRAC,
    gc_max_frac: float = DEFAULT_GC_MAX_FRAC,
    max_run:     int   = DEFAULT_MAX_HOMOPOLYMER_RUN,
    progress:    bool  = True,
) -> Dict[str, np.ndarray]:
    """Per-row projection over a ``[N, L]`` batch.

    Returns:
        {
          "projected_codes" : np.ndarray [N, L] int8,
          "edit_distances"  : np.ndarray [N]    int32  (-1 if no valid solution),
          "was_valid"       : np.ndarray [N]    bool   (input already valid),
          "compliance_rate" : float    fraction of OUTPUT rows that are valid,
          "mean_edit_distance": float (over all N),
        }
    """
    arr = np.asarray(codes, dtype=np.int8)
    assert arr.ndim == 2, f"expected [N, L], got {arr.shape}"
    N, L = arr.shape
    out_codes = np.empty_like(arr)
    edits     = np.zeros(N, dtype=np.int32)
    was_valid = is_valid_batch(arr, gc_min_frac, gc_max_frac, max_run)        # [N]

    iterator: Iterable[int] = range(N)
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, total=N, desc="bio-project")
        except ImportError:
            pass

    for i in iterator:
        if was_valid[i]:
            out_codes[i] = arr[i]
            edits[i]     = 0
            continue
        proj, cost = project_to_valid(arr[i], gc_min_frac, gc_max_frac, max_run)
        out_codes[i] = proj
        edits[i]     = cost

    new_valid = is_valid_batch(out_codes, gc_min_frac, gc_max_frac, max_run)
    compliance_rate = float(new_valid.mean())
    # only count edit distances on rows whose projection succeeded
    succ_edits = edits[edits >= 0]
    mean_edit  = float(succ_edits.mean()) if succ_edits.size > 0 else 0.0
    return {
        "projected_codes":   out_codes,
        "edit_distances":    edits,
        "was_valid":         was_valid,
        "compliance_rate":   compliance_rate,
        "mean_edit_distance": mean_edit,
    }
