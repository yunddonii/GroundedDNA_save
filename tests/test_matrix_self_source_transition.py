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
    _implementation_fingerprint,
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


SYNTHETIC_VARIANT = "bihalf"


def _synthetic_manifest(self_digest=None):
    """A manifest built from the current tree, not from stored experiments.

    The stored-manifest tests only protect the host that happens to hold
    /data. On a fresh clone they skip, and a suite that skips its own contract
    reports green while the contract is unenforced (audit §261). This payload
    reproduces the shape the profile check reads -- the exact required path set
    at their current digests, with the launcher's entry set to whatever the
    case under test needs -- so the negative contract runs everywhere.
    """
    from scripts.run_baseline_p0_matrix import (
        _required_implementation_paths, CANONICAL_VARIANT_SOURCE_PROFILES)
    snapshot = {}
    for path in _required_implementation_paths(SYNTHETIC_VARIANT):
        snapshot[path] = _current_source_sha256(path)
    profile_path, profile_digest = CANONICAL_VARIANT_SOURCE_PROFILES[
        SYNTHETIC_VARIANT]
    snapshot[profile_path] = profile_digest
    if self_digest is not None:
        snapshot[SELF_SOURCE_RELATIVE] = self_digest
    return {"variant": SYNTHETIC_VARIANT,
            "bit_length": 30,
            "protocol_identity": {"implementation_sha256": snapshot}}


def _synthetic_audit_records(self_digest):
    """One `complete_*` record built from the synthetic snapshot.

    Lets the aggregator's own gate be exercised with no experiment artifacts,
    so its behaviour is not left to a host that happens to hold /data.
    """
    payload = _synthetic_manifest(self_digest=self_digest)
    snapshot = payload["protocol_identity"]["implementation_sha256"]
    key = f"u0/{SYNTHETIC_VARIANT}/Flickr25k/30b/seed42"
    return {key: {
        "key": key,
        "variant": SYNTHETIC_VARIANT,
        "bit": 30,
        "status": "complete_main_eligible",
        "implementation_sha256": snapshot,
        "implementation_fingerprint_sha256": _implementation_fingerprint(
            snapshot),
    }}


def test_the_aggregator_gate_runs_without_stored_experiments():
    from scripts.aggregate_baseline_p0_matrix import (
        _audit_implementation_fingerprints)
    entry = REGISTRY[SELF_SOURCE_RELATIVE]
    audit = _audit_implementation_fingerprints(
        _synthetic_audit_records(entry["before_sha256"]))
    assert audit.get("status") != "no_completed_cells_to_audit"
    assert audit.get("comparison_safe") is True, audit.get(
        "historical_snapshot_current_drift_paths")
    assert not (audit.get("blocked_cells") or {})


def test_the_aggregator_gate_refuses_an_unknown_digest_without_stored_data():
    from scripts.aggregate_baseline_p0_matrix import (
        _audit_implementation_fingerprints)
    audit = _audit_implementation_fingerprints(
        _synthetic_audit_records("f" * 64))
    assert audit.get("comparison_safe") is False or (
        audit.get("blocked_cells") or {})


def test_the_synthetic_fixture_is_accepted_when_untampered():
    """Anchors the negatives below: if this fails they prove nothing."""
    entry = REGISTRY[SELF_SOURCE_RELATIVE]
    payload = _synthetic_manifest(self_digest=entry["before_sha256"])
    assert _matches_canonical_source_profile(payload, SYNTHETIC_VARIANT), (
        "the synthetic manifest must pass before it can demonstrate refusal")


@pytest.mark.parametrize("name,mutate", [
    ("classification removed",
     lambda e: {k: v for k, v in e.items() if k != "classification"}),
    ("evidence removed",
     lambda e: {k: v for k, v in e.items() if k != "evidence"}),
    ("classification not in the allowed vocabulary",
     lambda e: {**e, "classification": "made_up_classification"}),
    ("evidence blank", lambda e: {**e, "evidence": "   "}),
    ("variant scope present",
     lambda e: {**e, "non_scientific_variants_by_sha256":
                {e["before_sha256"]: ("not-a-variant",)}}),
    ("bit scope present",
     lambda e: {**e, "non_scientific_bits_by_sha256":
                {e["before_sha256"]: (999,)}}),
    ("empty variant scope map",
     lambda e: {**e, "non_scientific_variants_by_sha256": {}}),
    ("empty bit scope map",
     lambda e: {**e, "non_scientific_bits_by_sha256": {}}),
])
def test_the_contract_holds_without_stored_experiments(name, mutate):
    import scripts.aggregate_baseline_p0_matrix as aggregator
    saved = aggregator.KNOWN_NON_SCIENTIFIC_IMPLEMENTATION_TRANSITIONS
    payload = _synthetic_manifest(self_digest=saved[SELF_SOURCE_RELATIVE][
        "before_sha256"])
    aggregator.KNOWN_NON_SCIENTIFIC_IMPLEMENTATION_TRANSITIONS = {
        **saved, SELF_SOURCE_RELATIVE: mutate(saved[SELF_SOURCE_RELATIVE])}
    try:
        assert not _matches_canonical_source_profile(
            payload, SYNTHETIC_VARIANT), name
    finally:
        aggregator.KNOWN_NON_SCIENTIFIC_IMPLEMENTATION_TRANSITIONS = saved


def test_an_unreviewed_self_digest_is_refused_without_stored_experiments():
    payload = _synthetic_manifest(self_digest="f" * 64)
    assert not _matches_canonical_source_profile(payload, SYNTHETIC_VARIANT)


def test_the_registry_has_no_duplicate_keys_and_full_metadata():
    """A duplicated key is silently shadowed; a bare entry skips review.

    Python keeps the last of two identical dict keys, so a second entry for a
    path is dead code that still reads as registered. Both defects were present
    in the first version of this fix.
    """
    import ast
    import collections
    source = (REPO / "scripts/aggregate_baseline_p0_matrix.py").read_text(
        encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Assign):
            continue
        if not any(getattr(t, "id", None)
                   == "KNOWN_NON_SCIENTIFIC_IMPLEMENTATION_TRANSITIONS"
                   for t in node.targets):
            continue
        keys = [k.value for k in node.value.keys if isinstance(k, ast.Constant)]
        duplicates = {k: n for k, n in collections.Counter(keys).items() if n > 1}
        assert not duplicates, f"shadowed registry keys: {duplicates}"
        for key, value in zip(node.value.keys, node.value.values):
            fields = {f.value for f in value.keys if isinstance(f, ast.Constant)}
            missing = {"classification", "evidence"} - fields
            assert not missing, f"{key.value} is missing {sorted(missing)}"
        return
    raise AssertionError("transition registry not found")


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


def _audit_fixture(historical_only=True):
    """Records shaped the way `_audit_implementation_fingerprints` reads them.

    `status` must start with "complete_": a plain "complete" audits zero cells
    and the function then answers `no_completed_cells_to_audit`, which reads
    like a pass. That is how a first attempt at this fixture missed the very
    blocking it was written to detect.

    `historical_only` selects the cells recorded BEFORE this launcher's
    reviewed transition. Those are the ones whose acceptance the transition has
    to preserve, and their number is fixed at 105. Counting every completed
    cell instead would break the moment the matrix legitimately grows to 108.
    """
    before = REGISTRY[SELF_SOURCE_RELATIVE]["before_sha256"]
    records = {}
    for path in _stored_manifests():
        payload = json.loads(path.read_text(encoding="utf-8"))
        identity = payload.get("protocol_identity") or {}
        if historical_only:
            snapshot = identity.get("implementation_sha256") or {}
            if snapshot.get(SELF_SOURCE_RELATIVE) != before:
                continue
        key = (f"u0/{payload.get('variant')}/{payload.get('dataset')}"
               f"/30b/seed{payload.get('seed')}")
        snapshot = identity.get("implementation_sha256")
        records[key] = {
            "key": key,
            "variant": payload.get("variant"),
            # `bit` is read directly at :1693; without it the audit raises
            # instead of judging. `protocol_digest_sha256` is a different
            # digest -- the fingerprint the audit groups by is
            # `_implementation_fingerprint(snapshot)`.
            "bit": int(payload.get("bit_length") or payload.get("bit") or 30),
            "status": "complete_main_eligible",
            "implementation_sha256": snapshot,
            "implementation_fingerprint_sha256": _implementation_fingerprint(
                snapshot),
        }
    return records


@pytest.mark.skipif(not _stored_manifests(),
                    reason="stored 30-bit manifests are not on this host")
def test_the_aggregator_source_audit_also_clears_every_stored_cell():
    """The launcher and the aggregator are two different gates.

    Registering `reviewed_sha256` and `after_sha256` made the launcher accept
    all 105 while the aggregator still blocked all 105, because its scope map
    demands an explicit variant list for every past mismatch digest. Only this
    second call catches that.
    """
    from scripts.aggregate_baseline_p0_matrix import (
        _audit_implementation_fingerprints)
    records = _audit_fixture()
    assert len(records) == 105, (
        "expected the 105 cells recorded before this launcher's transition, "
        f"built {len(records)}")
    audit = _audit_implementation_fingerprints(records)
    assert audit.get("status") != "no_completed_cells_to_audit", (
        "the fixture audited nothing -- check that status starts with 'complete_'")
    assert audit.get("comparison_safe") is True, audit.get(
        "historical_snapshot_current_drift_paths")
    assert not (audit.get("blocked_cells") or {})


@pytest.mark.skipif(not _stored_manifests(),
                    reason="stored 30-bit manifests are not on this host")
def test_the_aggregator_source_audit_blocks_an_unreviewed_digest():
    from scripts.aggregate_baseline_p0_matrix import (
        _audit_implementation_fingerprints)
    records = _audit_fixture()
    victim = sorted(records)[0]
    tampered = copy.deepcopy(records)
    tampered[victim]["implementation_sha256"] = dict(
        tampered[victim]["implementation_sha256"])
    tampered[victim]["implementation_sha256"][SELF_SOURCE_RELATIVE] = "f" * 64
    audit = _audit_implementation_fingerprints(tampered)
    assert audit.get("comparison_safe") is False or (
        audit.get("blocked_cells") or {}), (
        "an unreviewed digest must not clear the aggregator's source audit")


@pytest.mark.skipif(not _stored_manifests(),
                    reason="stored 30-bit manifests are not on this host")
@pytest.mark.parametrize("name,mutate", [
    ("classification removed",
     lambda e: {k: v for k, v in e.items() if k != "classification"}),
    ("evidence removed",
     lambda e: {k: v for k, v in e.items() if k != "evidence"}),
    ("classification not in the allowed vocabulary",
     lambda e: {**e, "classification": "made_up_classification"}),
    ("evidence blank",
     lambda e: {**e, "evidence": "   "}),
    ("variant scope present",
     lambda e: {**e, "non_scientific_variants_by_sha256":
                {e["before_sha256"]: ("not-a-variant",)}}),
    ("bit scope present",
     lambda e: {**e, "non_scientific_bits_by_sha256":
                {e["before_sha256"]: (999,)}}),
    # An EMPTY scope map is the sharp case: it is falsy, so a truthiness check
    # lets it through, while the aggregator's audit still demands a per-digest
    # entry and blocks all 105 cells. The two gates disagreed here (§255).
    ("empty variant scope map",
     lambda e: {**e, "non_scientific_variants_by_sha256": {}}),
    ("empty bit scope map",
     lambda e: {**e, "non_scientific_bits_by_sha256": {}}),
])
def test_the_self_transition_refuses_an_unreviewed_registry_entry(name, mutate):
    """Digests alone are not review.

    An entry can carry the right before/after SHAs and still never have been
    reviewed: no classification, no written evidence, or a scope claiming the
    change is non-scientific for only some variants or bit budgets. The
    launcher trains and scores nothing, so a scoped entry for it is a
    contradiction the caller cannot even evaluate -- refuse rather than ignore.

    All six passed as accepted before this contract existed (audit §254).
    """
    import scripts.aggregate_baseline_p0_matrix as aggregator
    payload = json.loads(_stored_manifests()[0].read_text(encoding="utf-8"))
    saved = aggregator.KNOWN_NON_SCIENTIFIC_IMPLEMENTATION_TRANSITIONS
    aggregator.KNOWN_NON_SCIENTIFIC_IMPLEMENTATION_TRANSITIONS = {
        **saved, SELF_SOURCE_RELATIVE: mutate(saved[SELF_SOURCE_RELATIVE])}
    try:
        assert not _matches_canonical_source_profile(payload, _variant(payload)), name
    finally:
        aggregator.KNOWN_NON_SCIENTIFIC_IMPLEMENTATION_TRANSITIONS = saved
