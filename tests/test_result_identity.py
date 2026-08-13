"""F08: a run directory must identify the run, and a lookup must not guess.

`_resolve_save_path` builds the path from date, tag, batch, epoch and LR, then
calls `os.makedirs(..., exist_ok=True)`. Nothing in the path records the seed,
the slot count, the bases per slot or whether this is a selection or a refit, so
two runs that differ only in those land in the SAME directory and the second
overwrites the first.

That is not hypothetical. On 2026-08-12 eight 3-seed cells were launched with a
variable name the launcher does not read; every seed for a given (dataset, N)
wrote to one directory. The surviving artefacts were worse than missing ones --
`args.txt` came from whichever process started last and the evaluation JSON from
whichever finished last, so a directory labelled seed 44 held seed 43's numbers,
and the two agreed to full float precision, which is how it was noticed at all.
The wreckage is preserved in `result_quarantine_collided_20260812/`.

The lookup side is the same defect. `ls -dt ... | head -1` picks the newest path
matching a glob, so an ablation or a lambda variant that shares the prefix can be
selected as the reference run.

These tests pin: identity covers what distinguishes runs; a collision is refused
rather than merged; and resolving a run means reading a manifest, not globbing.
"""
from __future__ import annotations

import json
import os
import sys
import types

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from dna_utils.run_identity import (  # noqa: E402
    RunCollision,
    RunIdentity,
    RunNotFound,
    claim_run_dir,
    load_run_manifest,
    resolve_run_dir,
    write_run_manifest,
)


def _args(**kw):
    a = types.SimpleNamespace(
        dataset="Flickr25k", setting="setting1", random_seed=42,
        num_semantic_parts=5, num_codons_per_codebook=3,
        stop_after_epoch=4, epoch=5,
        lr_schedule_horizon=None, sinkhorn_schedule_horizon=None,
        selection_mode="refit",
    )
    for k, v in kw.items():
        setattr(a, k, v)
    return a


# ------------------------------------------------------------------ identity

def test_identity_covers_what_distinguishes_runs():
    i = RunIdentity.from_args(_args())
    assert i.dataset == "Flickr25k" and i.seed == 42
    assert i.num_slots == 5 and i.bases_per_slot == 3
    assert i.total_bases == 15 and i.total_bits == 30
    assert i.stop_after_epoch == 4
    assert len(i.digest) == 64


@pytest.mark.parametrize("field,value", [
    ("random_seed", 43),
    ("num_semantic_parts", 6),
    ("num_codons_per_codebook", 4),
    ("stop_after_epoch", 9),
    ("selection_mode", "select"),
    ("dataset", "NUSWIDE"),
])
def test_every_distinguishing_field_changes_the_digest(field, value):
    """The 2026-08-12 loss happened because seed did not reach the path. Each of
    these must move the digest, or two different runs can share a directory."""
    base = RunIdentity.from_args(_args()).digest
    other = RunIdentity.from_args(_args(**{field: value})).digest
    assert other != base, f"{field} does not affect run identity"


def test_identical_args_give_identical_digest():
    assert RunIdentity.from_args(_args()).digest == \
        RunIdentity.from_args(_args()).digest


def test_horizons_are_part_of_identity():
    """Search and final configs differ only in the horizons (D2); they must not
    share a directory."""
    search = RunIdentity.from_args(_args(epoch=60, lr_schedule_horizon=60,
                                         sinkhorn_schedule_horizon=5))
    final = RunIdentity.from_args(_args(epoch=5))
    assert search.digest != final.digest


# ----------------------------------------------------------------- claiming

def test_claim_writes_a_manifest(tmp_path):
    d = tmp_path / "run"
    ident = RunIdentity.from_args(_args())
    claim_run_dir(str(d), ident)
    md = load_run_manifest(str(d))
    assert md is not None and md.digest == ident.digest
    assert md.seed == 42 and md.total_bases == 15


def test_reclaiming_the_same_identity_is_allowed(tmp_path):
    """A resume of the same run must work; only a DIFFERENT run is a collision."""
    d = tmp_path / "run"
    ident = RunIdentity.from_args(_args())
    claim_run_dir(str(d), ident)
    claim_run_dir(str(d), ident)          # must not raise


def test_a_different_run_in_the_same_dir_is_refused(tmp_path):
    """The exact 2026-08-12 failure: seed 44 landing on seed 43's directory."""
    d = tmp_path / "run"
    claim_run_dir(str(d), RunIdentity.from_args(_args(random_seed=43)))
    with pytest.raises(RunCollision):
        claim_run_dir(str(d), RunIdentity.from_args(_args(random_seed=44)))


def test_collision_message_names_the_differing_fields(tmp_path):
    d = tmp_path / "run"
    claim_run_dir(str(d), RunIdentity.from_args(_args(random_seed=43)))
    with pytest.raises(RunCollision) as ex:
        claim_run_dir(str(d), RunIdentity.from_args(
            _args(random_seed=44, num_codons_per_codebook=4)))
    msg = str(ex.value)
    assert "seed" in msg and "bases_per_slot" in msg


def test_unmanifested_nonempty_dir_is_refused(tmp_path):
    """A legacy directory has no manifest; writing into it would mix artefacts
    from a run whose identity cannot be established."""
    d = tmp_path / "run"
    d.mkdir()
    (d / "model_state_dict.pth").write_bytes(b"x")
    with pytest.raises(RunCollision):
        claim_run_dir(str(d), RunIdentity.from_args(_args()))


def test_empty_dir_is_claimable(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    claim_run_dir(str(d), RunIdentity.from_args(_args()))
    assert load_run_manifest(str(d)) is not None


# ----------------------------------------------------------------- resolving

def test_resolve_requires_a_matching_manifest(tmp_path):
    a = tmp_path / "a"
    claim_run_dir(str(a), RunIdentity.from_args(_args(random_seed=42)))
    got = resolve_run_dir(str(tmp_path), RunIdentity.from_args(_args(random_seed=42)))
    assert os.path.realpath(got) == os.path.realpath(str(a))


def test_resolve_ignores_a_newer_directory_of_a_different_run(tmp_path):
    """`ls -dt | head -1` would return the ablation. Identity must win over
    mtime, or an A2 run gets evaluated as if it were the reference."""
    ref = tmp_path / "ref"
    claim_run_dir(str(ref), RunIdentity.from_args(_args(selection_mode="refit")))
    abl = tmp_path / "ablation"
    claim_run_dir(str(abl), RunIdentity.from_args(_args(selection_mode="A2")))
    os.utime(abl, (10 ** 10, 10 ** 10))          # make the ablation newest
    got = resolve_run_dir(str(tmp_path), RunIdentity.from_args(_args(selection_mode="refit")))
    assert os.path.realpath(got) == os.path.realpath(str(ref))


def test_resolve_fails_closed_when_absent(tmp_path):
    with pytest.raises(RunNotFound):
        resolve_run_dir(str(tmp_path), RunIdentity.from_args(_args()))


def test_resolve_refuses_when_two_dirs_claim_one_identity(tmp_path):
    """Ambiguity must not be resolved by picking the newer one."""
    for name in ("a", "b"):
        claim_run_dir(str(tmp_path / name), RunIdentity.from_args(_args()))
    with pytest.raises(RunCollision):
        resolve_run_dir(str(tmp_path), RunIdentity.from_args(_args()))


def test_quarantined_collisions_are_never_resolved(tmp_path):
    """result_quarantine_collided_20260812/ is evidence, not a source."""
    q = tmp_path / "result_quarantine_collided_20260812"
    claim_run_dir(str(q / "run"), RunIdentity.from_args(_args()))
    with pytest.raises(RunNotFound):
        resolve_run_dir(str(tmp_path), RunIdentity.from_args(_args()))


def test_manifest_round_trip_is_json(tmp_path):
    d = tmp_path / "run"
    ident = RunIdentity.from_args(_args())
    claim_run_dir(str(d), ident)
    raw = json.load(open(os.path.join(str(d), "run_identity.json")))
    assert raw["digest"] == ident.digest
    assert raw["total_bits"] == 30
