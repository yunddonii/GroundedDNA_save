"""The 24-base wrapper must actually run at 24 bases (re-audit §11.2).

`bc4868b` threaded a protocol through the native validators but bound it to the
IMPORT-TIME `DEFAULT_PROTOCOL`. The 24-base wrapper patches `MATCHED_LENGTH` and
the GC constants but not `DEFAULT_PROTOCOL`, so inside the wrapper context the
driver reported `MATCHED_LENGTH=24` while `DEFAULT_PROTOCOL.length_bases=15`,
and `main()` passed the 15-base protocol to the verifiers. A fresh 24-base run
would therefore validate 24-base artefacts against the 15-base contract.

The same import-time assumption broke the lock recomputation: the tool computed
the CANONICAL driver's digests, but the 24-base wrapper adds implementation
paths and a pipeline variant, so the digests a real 24-base run produces are
different ones -- none of which were registered.

These tests assert effective VALUES in each context, not the presence of a
keyword in source text.
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
from scripts.run_native_dna_p0_24 import (  # noqa: E402
    configured_canonical_driver,
)


def test_wrapper_context_agrees_on_length():
    """MATCHED_LENGTH and the effective protocol must not disagree."""
    with configured_canonical_driver():
        assert driver.MATCHED_LENGTH == 24
        assert driver.effective_protocol().length_bases == 24


def test_canonical_context_is_unaffected():
    assert driver.MATCHED_LENGTH == driver.effective_protocol().length_bases


def test_effective_protocol_is_resolved_per_call_not_at_import():
    """The regression in one assertion: reading it at call time is what makes
    the wrapper's patch visible."""
    outside = driver.effective_protocol().length_bases
    with configured_canonical_driver():
        inside = driver.effective_protocol().length_bases
    assert (outside, inside) == (15, 24)


# ------------------------------------------------------------------- locks

@pytest.mark.parametrize("length", (15, 18, 20, 24))
def test_every_supported_length_has_reviewed_locks(length):
    """20 was declared supported by NativeProtocol but absent from the registry,
    so `reviewed_method_locks(20)` raised SystemExit -- F18's own completion
    criterion asks for 15/18/20/24."""
    locks = agg.reviewed_method_locks(length)
    assert set(locks) >= set(driver.METHODS)
    assert all(locks[m] for m in driver.METHODS)


def test_24_wrapper_locks_are_the_registered_ones():
    """The digests a real 24-base run produces, computed in the wrapper context
    the runner uses -- not the canonical driver under an env override."""
    from scripts.recompute_native_method_locks import locks_for
    computed = locks_for(24)
    for method, digest in computed.items():
        assert digest in agg.reviewed_method_locks(24)[method], (
            f"{method}: wrapper lock {digest[:12]}... unregistered")


@pytest.mark.parametrize("length", (15, 18, 20))
def test_canonical_locks_are_the_registered_ones(length):
    from scripts.recompute_native_method_locks import locks_for
    for method, digest in locks_for(length).items():
        assert digest in agg.reviewed_method_locks(length)[method], (
            f"{method}@{length}: lock {digest[:12]}... unregistered")


def test_lengths_produce_distinct_locks():
    """If the recomputation ignored the length, every length would emit the same
    digest -- which is exactly how the wrong 24 values got registered."""
    from scripts.recompute_native_method_locks import locks_for
    seen = {L: locks_for(L)["bee2018"] for L in (15, 18, 20, 24)}
    assert len(set(seen.values())) == 4, seen


# -------------------------------------------------------------- aggregate

def test_aggregate_reports_the_accepted_lock_set(tmp_path):
    """The payload advertised the pre-F18 table while the validator accepted the
    reviewed set, so the provenance did not describe the gate."""
    payload, _ = agg.aggregate([tmp_path], seeds=(42,), verify_hashes=False,
                               protocol=15)
    reported = payload["method_protocol_lock_sha256"]
    for method, digests in agg.reviewed_method_locks(15).items():
        assert set(reported[method]) == set(digests)


# ---------------------------------------------- re-audit §12 regressions

@pytest.mark.parametrize("length", (15, 18, 20, 24))
def test_aggregator_imports_in_a_fresh_process_at_every_length(length):
    """§12.3: the 20-base lock went into the post-F18 table only, while the
    import guard still keyed off the pre-F18 one, so
    `GDNA_NATIVE_DNA_BASES=20 python -c "import ..."` exited 1 and the env20
    aggregator could not print `--help`. A same-process function call cannot see
    this -- the module is already imported."""
    import subprocess
    env = {**os.environ, "GDNA_NATIVE_DNA_BASES": str(length)}
    proc = subprocess.run(
        [sys.executable, "-c", "import scripts.aggregate_native_dna_p0"],
        cwd=_REPO, env=env, capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stderr[-400:]


@pytest.mark.parametrize("length", (15, 18, 20, 24))
def test_matrix_imports_in_a_fresh_process_at_every_length(length):
    import subprocess
    env = {**os.environ, "GDNA_NATIVE_DNA_BASES": str(length)}
    proc = subprocess.run(
        [sys.executable, "-c", "import scripts.run_native_dna_p0_matrix"],
        cwd=_REPO, env=env, capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stderr[-400:]


def test_matrix_completion_gate_receives_the_wrapper_protocol():
    """§12.2: both `_completed_cells()` calls omitted the protocol, so
    `coerce_protocol(None)` read the process default. A fresh 24-base child
    could finish training and still be failed by the post-child gate."""
    import inspect
    import re
    import scripts.run_native_dna_p0_matrix as matrix
    source = inspect.getsource(matrix)
    # Actual invocations only -- the module docstring names the function too.
    calls = [m.end() for m in re.finditer(r"=\s*_completed_cells\(", source)]
    assert len(calls) >= 2, "expected a preflight and a post-child gate"
    for pos in calls:
        block = source[pos:pos + 400]
        assert "protocol=" in block, (
            "every completion gate must be told which length it is judging")


def test_launch_options_snapshots_the_protocol():
    import scripts.run_native_dna_p0_matrix as matrix
    from scripts.run_native_dna_p0_matrix_24 import configured_canonical_matrix
    assert "protocol" in matrix.LaunchOptions.__dataclass_fields__
    outside = matrix.LaunchOptions.__dataclass_fields__[
        "protocol"].default_factory().length_bases
    with configured_canonical_matrix():
        inside = matrix.LaunchOptions.__dataclass_fields__[
            "protocol"].default_factory().length_bases
    assert (outside, inside) == (15, 24)


@pytest.mark.parametrize("length", (15, 18, 20, 24))
def test_explicit_length_aggregate_is_internally_consistent(tmp_path, length):
    """§12.4: `aggregate(protocol=18)` in a default-15 process reported
    `length_bases=18` while every record said 15, the display said PRIMO-15 and
    the Markdown header said 15 -- the payload contradicted itself."""
    import re
    payload, _ = agg.aggregate([tmp_path], seeds=(42,), verify_hashes=False,
                               protocol=length)
    assert payload["length_bases"] == length
    assert payload["display_names"]["bee2021"] == (
        f"PRIMO-{length} frozen-predictor length-transfer")
    for record in payload["records"]:
        assert record["length"] == length, record["key"]
        assert record["display_name"] == payload["display_names"][record["method"]]
    markdown = agg._markdown(payload)
    capacity = re.search(r"Capacity: .(\d+)", markdown)
    assert capacity and int(capacity.group(1)) == length
