"""F18: the matched length must be a property of the artefact, not the process.

`run_native_dna_p0`, `run_native_dna_p0_matrix` and `aggregate_native_dna_p0`
each read `GDNA_NATIVE_DNA_BASES` into a module-level constant at import time,
and `run_native_dna_p0_24` hard-codes 24. A length is therefore a property of
the process: once the environment says 15, an 18-base manifest cannot be
validated at all, which is what produces

    length: expected 15, found 18;
    protocol_identity.matched_length_bases: expected 15, found 18;
    evaluation.protocol: expected 'matched_15nt_adaptation',
                         found 'matched_18nt_adaptation'

across 18 native cases, and why `tests/test_native_dna_p0_24.py` imports the
shared driver as `driver18` and asserts `MATCHED_LENGTH == 18`.

These tests pin the contract that lets 15, 18, 20 and 24 be exercised in ONE
process without any of them mutating a global the others read.
"""
from __future__ import annotations

import os
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from dna_utils.native_protocol import (  # noqa: E402
    DEFAULT_LENGTH,
    ENV_VAR,
    SUPPORTED_LENGTHS,
    NativeProtocol,
    NativeProtocolError,
    coerce_protocol,
    resolve_native_protocol,
)


@pytest.mark.parametrize("length", SUPPORTED_LENGTHS)
def test_every_supported_length_resolves(length):
    assert resolve_native_protocol(length).length_bases == length


def test_protocol_label_matches_the_manifest_string(  ):
    """`evaluation.protocol` is compared verbatim, so the label must be derived
    from the length rather than written out per module."""
    assert resolve_native_protocol(15).protocol_label == "matched_15nt_adaptation"
    assert resolve_native_protocol(18).protocol_label == "matched_18nt_adaptation"
    assert resolve_native_protocol(24).protocol_label == "matched_24nt_adaptation"


def test_length_token_matches_the_run_directory_fragment():
    assert resolve_native_protocol(15).length_token == "15nt"
    assert resolve_native_protocol(18).length_token == "18nt"


def test_unsupported_length_is_refused():
    with pytest.raises(NativeProtocolError):
        resolve_native_protocol(16)


def test_two_lengths_coexist_in_one_process():
    """The whole point of F18: an 18-base contract must be constructible while
    a 15-base one is alive, with neither changing the other."""
    fifteen = resolve_native_protocol(15)
    eighteen = resolve_native_protocol(18)
    assert (fifteen.length_bases, eighteen.length_bases) == (15, 18)
    assert fifteen.protocol_label != eighteen.protocol_label
    assert resolve_native_protocol(15) == fifteen


def test_protocol_is_immutable():
    """A shared mutable contract is how one suite contaminated another."""
    protocol = resolve_native_protocol(15)
    with pytest.raises(Exception):
        protocol.length_bases = 18       # type: ignore[misc]


# ------------------------------------------------------------------ defaults

def test_default_is_the_paper_length():
    assert DEFAULT_LENGTH == 15
    assert resolve_native_protocol(None, env={}).length_bases == 15


def test_environment_is_read_only_when_no_length_is_given():
    env = {ENV_VAR: "18"}
    assert resolve_native_protocol(None, env=env).length_bases == 18
    # An explicit length wins; this is what makes a fixture independent of how
    # the process happens to be configured.
    assert resolve_native_protocol(24, env=env).length_bases == 24


def test_non_integer_environment_is_refused():
    with pytest.raises(NativeProtocolError):
        resolve_native_protocol(None, env={ENV_VAR: "fifteen"})


# ------------------------------------------------------------ GC consistency

@pytest.mark.parametrize("length,gc_min,gc_max", [
    (15, 6, 9), (18, 8, 10), (20, 8, 12), (24, 10, 14),
])
def test_gc_window_comes_from_the_central_policy(length, gc_min, gc_max):
    """F07 fixed one 40-60% inclusive window. This file must not become a
    fifth convention, so the counts are asserted against it directly."""
    from dna_utils.gc_policy import resolve_gc_policy
    protocol = resolve_native_protocol(length)
    policy = resolve_gc_policy(length)
    assert (protocol.gc_min_count, protocol.gc_max_count) == (gc_min, gc_max)
    assert (protocol.gc_min_count, protocol.gc_max_count) == (
        policy.gc_min_count, policy.gc_max_count)
    assert protocol.max_homopolymer_run == policy.max_run


# -------------------------------------------------------------- call boundary

def test_coerce_accepts_a_protocol_a_length_or_none():
    protocol = resolve_native_protocol(18)
    assert coerce_protocol(protocol) is protocol
    assert coerce_protocol(18) == protocol
    assert isinstance(coerce_protocol(None), NativeProtocol)
