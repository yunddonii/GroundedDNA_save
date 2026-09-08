"""The matrix launcher's own source transition resolves outside itself.

`_matches_canonical_source_profile` compares every recorded implementation
digest against the current file. When the differing path is the launcher
itself, the approved digest cannot live in the launcher: writing the new SHA
into its own alias map changes that SHA again, so no fixed point exists
(audit §219). The aggregator is a separate file and is not in the hashed
implementation set, so its transition registry is the non-circular anchor.

The earlier test for this area only asserted that a digest string appeared
somewhere in the aggregator. That passed while all 105 stored manifests were
being rejected. These tests assert acceptance of the real manifests instead,
and -- equally important -- that tampering is still refused.
"""
from __future__ import annotations

import copy
import json
import pathlib
import sys

import pytest


REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.run_baseline_p0_matrix import (  # noqa: E402
    SELF_SOURCE_RELATIVE,
    _matches_canonical_source_profile,
    _self_transition_is_reviewed,
    _current_source_sha256,
)
from scripts.aggregate_baseline_p0_matrix import (  # noqa: E402
    KNOWN_NON_SCIENTIFIC_IMPLEMENTATION_TRANSITIONS as REGISTRY,
)

RESULT_ROOT = pathlib.Path(
    "/data/yschoi/gdna_p4baseline/result_baseline"
    "/p0_matrix_seeds42-43-44_author_fixed_30b")


def _stored_manifests():
    if not RESULT_ROOT.is_dir():
        return []
    return sorted(RESULT_ROOT.glob("*/attempt_*/*/*_dnaeval/p0_run_manifest.json"))


def _variant(payload):
    return str(payload.get("variant") or payload.get("method"))


def test_the_registry_declares_this_files_current_digest():
    """`after_sha256` must be what the launcher hashes to right now.

    If it drifts, every stored manifest is rejected -- which is exactly the
    failure this test exists to catch, and it is silent otherwise.
    """
    entry = REGISTRY.get(SELF_SOURCE_RELATIVE)
    assert entry is not None, (
        f"{SELF_SOURCE_RELATIVE} has no transition entry in the aggregator; "
        "the launcher cannot approve its own edit from inside itself")
    assert entry["after_sha256"] == _current_source_sha256(SELF_SOURCE_RELATIVE)
    assert entry["before_sha256"] in entry["reviewed_sha256"]
    assert entry["after_sha256"] in entry["reviewed_sha256"]


@pytest.mark.skipif(not _stored_manifests(),
                    reason="stored 30-bit manifests are not on this host")
def test_every_stored_manifest_is_still_accepted():
    rejected = []
    for path in _stored_manifests():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not _matches_canonical_source_profile(payload, _variant(payload)):
            rejected.append(str(path))
    assert not rejected, (
        f"{len(rejected)} of {len(_stored_manifests())} stored manifests are "
        f"rejected by the current source; first: {rejected[:1]}")


@pytest.mark.skipif(not _stored_manifests(),
                    reason="stored 30-bit manifests are not on this host")
def test_an_unreviewed_recorded_digest_is_refused():
    payload = json.loads(_stored_manifests()[0].read_text(encoding="utf-8"))
    tampered = copy.deepcopy(payload)
    tampered["protocol_identity"]["implementation_sha256"][
        SELF_SOURCE_RELATIVE] = "f" * 64
    assert not _matches_canonical_source_profile(tampered, _variant(payload))


def test_a_current_file_that_is_not_after_sha256_is_refused():
    """Editing the launcher again must fail closed until it is reviewed."""
    entry = REGISTRY[SELF_SOURCE_RELATIVE]
    assert not _self_transition_is_reviewed(entry["before_sha256"], "e" * 64)


def test_a_missing_registry_entry_fails_closed():
    import scripts.aggregate_baseline_p0_matrix as aggregator
    saved = aggregator.KNOWN_NON_SCIENTIFIC_IMPLEMENTATION_TRANSITIONS
    aggregator.KNOWN_NON_SCIENTIFIC_IMPLEMENTATION_TRANSITIONS = {}
    try:
        entry = REGISTRY[SELF_SOURCE_RELATIVE]
        assert not _self_transition_is_reviewed(
            entry["before_sha256"], entry["after_sha256"])
    finally:
        aggregator.KNOWN_NON_SCIENTIFIC_IMPLEMENTATION_TRANSITIONS = saved
