"""Whether epsilon was annealed is its own field, not a guess from another one.

The manifest schema branched on `inference_epoch_source == "no_annealing"`. That
field says only where the EPOCH was recovered from. A fresh run configured
without an epsilon schedule still writes a checkpoint sidecar, so its epoch
resolves as `checkpoint_metadata` -- and the validator then read it as an
annealed run with a null epsilon and refused it. A configuration the trainer
supports was therefore producer-to-consumer incompatible end to end (§20.9).

The other half of the same confusion: the no-anneal path recorded
`effective_sinkhorn_epsilon: null`. The router does not run at "no epsilon"; it
runs at `args.sinkhorn_temperature`. Recording null conflated the internal
`epsilon_override=None` sentinel with the operating point of the forward pass,
so the manifest could not say what produced the codes.

These walk the real writer -> resolver -> manifest -> validator path for both
annealing states and both epoch provenances.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dna_utils.extraction_validation import (  # noqa: E402
    ExtractionInvalid,
    validate_extraction_run,
)
from dna_utils.runtime_state import (  # noqa: E402
    DEFAULT_STATIC_SINKHORN_EPSILON,
    resolve_inference_epoch,
    write_checkpoint_metadata,
)
from tests.test_extraction_run_validation import _cell  # noqa: E402


def _args(**overrides) -> SimpleNamespace:
    base = dict(epoch=5, stop_after_epoch=4, sinkhorn_temperature=0.07,
                inference_epoch=None)
    base.update(overrides)
    return SimpleNamespace(**base)


def _checkpoint(tmp_path: Path) -> Path:
    path = tmp_path / "model_state_dict.pth"
    path.write_bytes(b"weights")
    return path


def _sidecar(checkpoint: Path, *, annealed: bool) -> None:
    write_checkpoint_metadata(
        str(checkpoint), checkpoint_epoch_zero_based=4,
        training_epoch_budget=5, stop_after_epoch=4,
        lr_schedule_horizon=5, sinkhorn_schedule_horizon=5,
        sinkhorn_epsilon_init=1.0 if annealed else None,
        sinkhorn_epsilon_final=0.1 if annealed else None)


# ------------------------------------------------------- the resolver

def test_a_fresh_no_anneal_run_with_a_sidecar_is_resolvable(tmp_path):
    """The §20.9 counterexample: source is checkpoint_metadata, not no_annealing."""
    checkpoint = _checkpoint(tmp_path)
    _sidecar(checkpoint, annealed=False)
    resolved = resolve_inference_epoch(str(checkpoint), _args())

    assert resolved.source == "checkpoint_metadata"
    assert resolved.sinkhorn_annealing_enabled is False
    # The operating point, not null.
    assert resolved.effective_sinkhorn_epsilon == pytest.approx(0.07)
    assert resolved.sinkhorn_schedule_horizon is None


def test_an_explicit_flag_on_a_no_anneal_run_is_also_resolvable(tmp_path):
    """The same run with `--inference_epoch`: source explicit_flag, still static."""
    checkpoint = _checkpoint(tmp_path)
    _sidecar(checkpoint, annealed=False)
    resolved = resolve_inference_epoch(
        str(checkpoint), _args(inference_epoch=4))

    assert resolved.source == "explicit_flag"
    assert resolved.sinkhorn_annealing_enabled is False
    assert resolved.effective_sinkhorn_epsilon == pytest.approx(0.07)
    assert resolved.sinkhorn_schedule_horizon is None


def test_an_annealed_run_still_records_its_schedule(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    _sidecar(checkpoint, annealed=True)
    resolved = resolve_inference_epoch(str(checkpoint), _args())

    assert resolved.sinkhorn_annealing_enabled is True
    assert resolved.sinkhorn_schedule_horizon == 5
    assert 0.1 <= resolved.effective_sinkhorn_epsilon <= 1.0


def test_a_legacy_no_sidecar_no_schedule_run_reports_its_static_epsilon(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    resolved = resolve_inference_epoch(str(checkpoint), _args())

    assert resolved.source == "no_annealing"
    assert resolved.sinkhorn_annealing_enabled is False
    assert resolved.effective_sinkhorn_epsilon == pytest.approx(0.07)


def test_the_static_epsilon_falls_back_to_the_router_default(tmp_path):
    """`SemanticSinkhornRouter(epsilon=getattr(args, ..., 0.05))`."""
    checkpoint = _checkpoint(tmp_path)
    args = _args()
    del args.sinkhorn_temperature
    resolved = resolve_inference_epoch(str(checkpoint), args)
    assert resolved.effective_sinkhorn_epsilon == pytest.approx(
        DEFAULT_STATIC_SINKHORN_EPSILON)


# ------------------------------------------------------- the validator

def _rewrite(cell: Path, **changes) -> None:
    for split in ("db", "query"):
        path = cell / f"extraction_manifest_{split}.json"
        payload = json.loads(path.read_text())
        payload.update(changes)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    marker = cell / "extraction_complete.json"
    record = json.loads(marker.read_text())
    from dna_utils.runtime_state import sha256_file
    record["manifest_sha256"] = {
        split: sha256_file(str(cell / f"extraction_manifest_{split}.json"))
        for split in ("db", "query")}
    marker.write_text(json.dumps(record))


def test_a_static_epsilon_run_validates_under_any_epoch_source(tmp_path):
    """Previously refused whenever the source was not literally no_annealing."""
    for source in ("checkpoint_metadata", "explicit_flag", "no_annealing"):
        sub = tmp_path / source
        sub.mkdir(parents=True, exist_ok=True)
        cell = _cell(sub)
        _rewrite(cell, sinkhorn_annealing_enabled=False,
                 sinkhorn_schedule_horizon=None,
                 effective_sinkhorn_epsilon=0.05,
                 inference_epoch_source=source)
        run = validate_extraction_run(str(cell))
        assert run.common["sinkhorn_annealing_enabled"] is False


def test_a_null_epsilon_is_refused_however_it_got_there(tmp_path):
    """The forward pass ran at some epsilon; an absence cannot describe it."""
    cell = _cell(tmp_path)
    _rewrite(cell, sinkhorn_annealing_enabled=False,
             sinkhorn_schedule_horizon=None,
             effective_sinkhorn_epsilon=None)
    with pytest.raises(ExtractionInvalid) as excinfo:
        validate_extraction_run(str(cell))
    assert "effective_sinkhorn_epsilon" in str(excinfo.value)


def test_annealing_enabled_without_a_horizon_is_refused(tmp_path):
    cell = _cell(tmp_path)
    _rewrite(cell, sinkhorn_annealing_enabled=True,
             sinkhorn_schedule_horizon=None)
    with pytest.raises(ExtractionInvalid) as excinfo:
        validate_extraction_run(str(cell))
    assert "horizon" in str(excinfo.value)


def test_annealing_disabled_with_a_horizon_is_refused(tmp_path):
    """A schedule that is not in force must not be recorded as if it were."""
    cell = _cell(tmp_path)
    _rewrite(cell, sinkhorn_annealing_enabled=False,
             sinkhorn_schedule_horizon=5)
    with pytest.raises(ExtractionInvalid):
        validate_extraction_run(str(cell))


def test_a_manifest_without_the_field_is_refused(tmp_path):
    """A null horizon must not be readable as either state."""
    cell = _cell(tmp_path)
    for split in ("db", "query"):
        path = cell / f"extraction_manifest_{split}.json"
        payload = json.loads(path.read_text())
        del payload["sinkhorn_annealing_enabled"]
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    from dna_utils.runtime_state import sha256_file
    marker = cell / "extraction_complete.json"
    record = json.loads(marker.read_text())
    record["manifest_sha256"] = {
        split: sha256_file(str(cell / f"extraction_manifest_{split}.json"))
        for split in ("db", "query")}
    marker.write_text(json.dumps(record))
    with pytest.raises(ExtractionInvalid) as excinfo:
        validate_extraction_run(str(cell))
    assert "sinkhorn_annealing_enabled" in str(excinfo.value)


# ------------------------------- source tokens carry invariants (§23.4)

def test_f01_unrestored_can_only_mean_epoch_zero(tmp_path):
    """The defect IS that nothing restored the epoch, so it cannot be 4."""
    cell = _cell(tmp_path)
    _rewrite(cell, inference_epoch_source="f01_unrestored", inference_epoch=4)
    with pytest.raises(ExtractionInvalid) as excinfo:
        validate_extraction_run(str(cell))
    assert "f01_unrestored" in str(excinfo.value)


def test_f01_unrestored_at_epoch_zero_is_admitted(tmp_path):
    cell = _cell(tmp_path)
    _rewrite(cell, inference_epoch_source="f01_unrestored", inference_epoch=0)
    run = validate_extraction_run(str(cell))
    assert run.common["inference_epoch"] == 0


def test_no_annealing_with_annealing_enabled_is_refused(tmp_path):
    """The epoch was skipped because there was no schedule to place it on."""
    cell = _cell(tmp_path)
    _rewrite(cell, inference_epoch_source="no_annealing",
             sinkhorn_annealing_enabled=True)
    with pytest.raises(ExtractionInvalid) as excinfo:
        validate_extraction_run(str(cell))
    assert "no_annealing" in str(excinfo.value)


def test_the_sidecar_endpoints_win_over_the_loaders(tmp_path):
    """A checkpoint annealed 1.0 -> 0.1, loaded by a process set to 0.5 -> 0.2.

    The horizon was read from the sidecar and the endpoints from `args`, so the
    manifest recorded 0.2 -- the value of a schedule that never ran.
    """
    checkpoint = _checkpoint(tmp_path)
    _sidecar(checkpoint, annealed=True)          # 1.0 -> 0.1 over 5 epochs
    resolved = resolve_inference_epoch(
        str(checkpoint),
        _args(inference_epoch=4, sinkhorn_epsilon_init=0.5,
              sinkhorn_epsilon_final=0.2))
    assert resolved.effective_sinkhorn_epsilon == pytest.approx(0.1)


def test_the_sidecar_endpoints_win_without_an_explicit_flag(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    _sidecar(checkpoint, annealed=True)
    resolved = resolve_inference_epoch(
        str(checkpoint),
        _args(sinkhorn_epsilon_init=0.5, sinkhorn_epsilon_final=0.2))
    assert resolved.effective_sinkhorn_epsilon == pytest.approx(0.1)
