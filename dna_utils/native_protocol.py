"""One object that carries a native-DNA length contract (F18).

Three modules each derived the matched length independently from the same
environment variable, and a fourth hard-coded 24:

    run_native_dna_p0.py        MATCHED_LENGTH = int(os.environ.get(..., "15"))
    run_native_dna_p0_matrix.py MATCHED_LENGTH = int(os.environ.get(..., "15"))
    aggregate_native_dna_p0.py  LENGTH = int(os.environ.get(..., "15"))
    run_native_dna_p0_24.py     MATCHED_LENGTH = 24

Because each is a module-level constant read at import time, a length is a
property of the PROCESS rather than of the manifest being validated. An 18-base
fixture therefore cannot be checked at all once the environment says 15 -- which
is why `tests/test_native_dna_p0_24.py` imports the shared driver as `driver18`
and asserts `MATCHED_LENGTH == 18`, and why 18 of the native cases fail with
`length: expected 15, found 18`.

The length belongs to the artefact. `NativeProtocol` makes it an argument that
validators accept, so 15, 18, 20 and 24 can each be exercised in one process
without any of them mutating a global the others read.

GC bounds come from `dna_utils.gc_policy`, so this file never introduces a
fifth convention.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Optional, Tuple

from dna_utils.gc_policy import resolve_gc_policy

#: The environment variable the native scripts have always read.
ENV_VAR = "GDNA_NATIVE_DNA_BASES"

#: The paper's length. 5 slots x 3 bases; see the 5-slot architecture decision.
DEFAULT_LENGTH = 15

#: Lengths with a declared GC window and a reviewed protocol lock.
SUPPORTED_LENGTHS: Tuple[int, ...] = (15, 18, 20, 24)


class NativeProtocolError(ValueError):
    """A length with no declared contract was requested."""


@dataclass(frozen=True)
class NativeProtocol:
    """Everything that follows from the matched length, in one value."""

    length_bases: int
    gc_min_count: int
    gc_max_count: int
    max_homopolymer_run: int

    @property
    def protocol_label(self) -> str:
        """The `evaluation.protocol` string, e.g. `matched_15nt_adaptation`."""
        return f"matched_{self.length_bases}nt_adaptation"

    @property
    def length_token(self) -> str:
        """The run-directory fragment, e.g. `15nt`."""
        return f"{self.length_bases}nt"

    def describe(self) -> str:
        return (
            f"{self.length_bases} bases, GC in "
            f"[{self.gc_min_count}, {self.gc_max_count}], "
            f"max run {self.max_homopolymer_run}")


def resolve_native_protocol(
        length: Optional[int] = None,
        *,
        env: Optional[dict] = None) -> NativeProtocol:
    """The contract for `length`; `None` reads the environment, then defaults.

    Passing the length explicitly is the point: it is how a test validates an
    18-base manifest in a process whose environment says 15.
    """
    if length is None:
        raw = (env if env is not None else os.environ).get(
            ENV_VAR, str(DEFAULT_LENGTH))
        try:
            length = int(raw)
        except (TypeError, ValueError) as error:
            raise NativeProtocolError(
                f"{ENV_VAR}={raw!r} is not an integer") from error
    length = int(length)
    if length not in SUPPORTED_LENGTHS:
        raise NativeProtocolError(
            f"no declared native protocol for {length} bases; supported: "
            f"{list(SUPPORTED_LENGTHS)}. Declare the GC window and the method "
            f"protocol lock deliberately before running it.")
    policy = resolve_gc_policy(length)
    return NativeProtocol(
        length_bases=length,
        gc_min_count=policy.gc_min_count,
        gc_max_count=policy.gc_max_count,
        max_homopolymer_run=policy.max_run,
    )


def coerce_protocol(
        protocol: Optional["NativeProtocol | int"]) -> NativeProtocol:
    """Accept a protocol, a bare length, or `None` at a call boundary."""
    if isinstance(protocol, NativeProtocol):
        return protocol
    return resolve_native_protocol(protocol)
