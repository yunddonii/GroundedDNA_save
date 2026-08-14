"""F18 follow-up: editing the driver must not invalidate every native cell.

`scripts/run_native_dna_p0.py` is listed in `IMPLEMENTATION_PATHS`, so its
SHA256 feeds `implementation_artifacts`, which feeds
`_method_protocol_lock_payload`, which is what the method-protocol lock digests.
F18 edited that file to thread a `NativeProtocol` through the validators, which
moved the digest 7b354a21... -> eb394591... and left all four registered
15-base locks stale. A correct new cell was then rejected with

    protocol_identity.method_protocol_lock_sha256: stable source/stage/bio/
    implementation/execution contract mismatch

Replacing the numbers in place would silently invalidate the historical 18-base
cells, which were produced under the older source and legitimately carry the
older digest. The registry therefore holds a REVIEWED SET per (length, method):
the pre-F18 digest and the current-source digest, both accepted, because the
F18 diff is numerically a no-op at every declared length --
`protocol.gc_min_count == ceil(GC_MIN * L)`, `gc_max_count == floor(GC_MAX * L)`
and `max_homopolymer_run == MAX_RUN`, with no change to training, extraction or
projection.

An unreviewed digest must still be rejected; that is the whole point of the lock.
"""
from __future__ import annotations

import os
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import scripts.aggregate_native_dna_p0 as agg  # noqa: E402
import scripts.run_native_dna_p0 as driver  # noqa: E402
from dna_utils.native_protocol import resolve_native_protocol  # noqa: E402


def _reviewed(length, method):
    return agg.reviewed_method_locks(length)[method]


@pytest.mark.parametrize("length", (15, 18, 24))
@pytest.mark.parametrize("method", ("bee2018", "bee2021", "koike2024",
                                    "koike2026"))
def test_every_method_has_a_reviewed_set(length, method):
    assert len(_reviewed(length, method)) >= 1


def test_current_source_lock_is_registered_at_15_bases():
    """The regression that motivated this file: a cell built from the CURRENT
    driver must validate. Recomputed here rather than hard-coded, so the test
    fails the next time someone edits an implementation path without review."""
    from scripts.recompute_native_method_locks import EXECUTION, PROBE
    from pathlib import Path
    for method in driver.METHODS:
        identity, _ = driver._protocol_identity(
            method=method, dataset=PROBE["dataset"], setting=PROBE["setting"],
            seed=PROBE["seed"],
            cache_audit={"cache_dir": "<lock-independent>", "blockers": []},
            dataset_root=Path(_REPO) / "dataset",
            predictor=(Path("/data/yschoi/groundeddna_native_p0/artifacts/"
                            "primo_yield_predictor.npz")
                       if method == "bee2021" else None),
            device="cuda:0", **EXECUTION)
        digest = agg._method_protocol_lock_digest(identity)
        assert digest in _reviewed(15, method), (
            f"{method}: current-source lock {digest[:12]}... is not reviewed; "
            f"run scripts/recompute_native_method_locks.py, read the source "
            f"diff, then register it")


def test_historical_pre_f18_lock_still_validates():
    """The 18-base diagnostic cells carry the pre-F18 digest and must keep
    validating; the re-registration must not orphan them."""
    assert ("533c7fe2fe7c2da593fc2b061ada18ea1e55feccf3f73a024ca6cbcda28935a4"
            in _reviewed(18, "bee2018"))
    assert ("bc3c5eb824adcce1adbc62809e18cc4ecc4971390263cad29c658360f28ac8a2"
            in _reviewed(15, "bee2018"))


def test_unreviewed_digest_is_rejected():
    errors = []
    agg._validate_method_protocol_lock(
        {"method_protocol_lock_sha256": "f" * 64},
        agg.Key("bee2018", "CIFAR10", 42), errors, 15)
    assert errors and "method_protocol_lock_sha256" in errors[0]


def _current_identity(method="bee2018"):
    """The identity the driver builds today, from the real source files."""
    from pathlib import Path
    from scripts.recompute_native_method_locks import EXECUTION, PROBE
    identity, _ = driver._protocol_identity(
        method=method, dataset=PROBE["dataset"], setting=PROBE["setting"],
        seed=PROBE["seed"],
        cache_audit={"cache_dir": "<lock-independent>", "blockers": []},
        dataset_root=Path(_REPO) / "dataset",
        predictor=(Path("/data/yschoi/groundeddna_native_p0/artifacts/"
                        "primo_yield_predictor.npz")
                   if method == "bee2021" else None),
        device="cuda:0", **EXECUTION)
    return identity


def test_current_source_identity_validates_without_errors():
    """The end the lock exists for: a cell built from today's source passes.
    The validator recomputes the digest from the identity, so this exercises the
    real payload rather than a hand-written digest field."""
    for method in driver.METHODS:
        errors = []
        agg._validate_method_protocol_lock(
            _current_identity(method), agg.Key(method, "CIFAR10", 42),
            errors, 15)
        assert errors == [], f"{method}: {errors}"


def test_f18_diff_is_numerically_a_noop_at_every_length():
    """The justification for accepting both digests, asserted rather than
    asserted-in-prose: the protocol object reproduces the module constants."""
    import math
    for length in (15, 18, 20, 24):
        p = resolve_native_protocol(length)
        assert p.gc_min_count == math.ceil(driver.GC_MIN * length)
        assert p.gc_max_count == math.floor(driver.GC_MAX * length)
        assert p.max_homopolymer_run == driver.MAX_RUN
