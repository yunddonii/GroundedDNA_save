"""§17: one strict validator, and the counterexamples it must refuse.

Every consumer had its own partial idea of "complete", and each was satisfiable
by artefacts that do not exist. These are the re-audit's own probes, turned into
regressions:

  * two manifests with fake digests, `schema_version=999`, `dataset=WRONG` and
    no NPZ/checkpoint/config at all were accepted by the aggregator;
  * two files containing `not-json` were accepted by the launcher as a finished
    cell;
  * a completion marker naming only `db` was written and looked authoritative;
  * `hash_2bit` was never checked to be the encoding of `base_indices`, so two
    arrays describing different codes passed.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from dna_utils.extraction_validation import (  # noqa: E402
    ExtractionInvalid,
    base_indices_to_2bit,
    describe_failure,
    validate_extraction_run,
)
from dna_utils.runtime_state import sha256_file  # noqa: E402

SLOTS, PER_SLOT = 5, 3
BASES = SLOTS * PER_SLOT


def _cell(tmp_path, rows=4, *, splits=("db", "query"), corrupt=None):
    ck = tmp_path / "model_state_dict.pth"
    ck.write_bytes(b"weights")
    cfg = tmp_path / "config.pt"
    cfg.write_bytes(b"cfg")
    rng = np.random.default_rng(0)
    manifests = {}
    for split in splits:
        base = rng.integers(0, 4, size=(rows, BASES), dtype=np.int64)
        hashed = base_indices_to_2bit(base)
        codebook = rng.integers(0, 64, size=(rows, SLOTS), dtype=np.int64)
        if corrupt == "hash":
            hashed = hashed.copy()
            hashed[0, 0] ^= 1
        if corrupt == "codebook":
            codebook = codebook.copy()
            codebook[0, 0] = 999
        if corrupt == "dtype":
            base = base.astype(np.float32)
        npz = tmp_path / f"extract_{split}.npz"
        np.savez(npz, base_indices=base, hash_2bit=hashed,
                 codebook_indices=codebook)
        manifest = {
            "schema_version": 1, "split": split, "n_rows": rows,
            "checkpoint_path": str(ck), "checkpoint_sha256": sha256_file(str(ck)),
            "config_path": str(cfg), "config_sha256": sha256_file(str(cfg)),
            "inference_epoch": 4, "inference_epoch_source": "explicit_flag",
            "effective_sinkhorn_epsilon": 0.1,
            "sinkhorn_schedule_horizon": 5, "lr_schedule_horizon": 5,
            "training_epoch_budget": 5, "training_stop_epoch": 4,
            "num_slots": SLOTS, "bases_per_slot": PER_SLOT,
            "total_bases": BASES, "total_bits": 2 * BASES,
            "dataset": "CIFAR10", "random_seed": 42, "codebook_size": 64,
            "npz_path": str(npz), "npz_sha256": sha256_file(str(npz)),
        }
        path = tmp_path / f"extraction_manifest_{split}.json"
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        manifests[split] = path
    marker = {
        "schema_version": 1, "splits": sorted(manifests),
        "manifest_sha256": {s: sha256_file(str(p))
                            for s, p in sorted(manifests.items())},
    }
    (tmp_path / "extraction_complete.json").write_text(json.dumps(marker))
    return tmp_path


def test_a_real_run_validates(tmp_path):
    run = validate_extraction_run(str(_cell(tmp_path)))
    assert sorted(run.splits) == ["db", "query"]
    assert run.common["inference_epoch"] == 4
    assert run.common["lr_schedule_horizon"] == 5
    assert run.common["training_stop_epoch"] == 4
    assert run.backfilled is False


# ------------------------------------------------- the re-audit's probes

def test_manifests_naming_nonexistent_artefacts_are_refused(tmp_path):
    """The §17.3 probe verbatim: no NPZ, no checkpoint, no config."""
    fake = {"schema_version": 999, "dataset": "WRONG", "n_rows": 1,
            "checkpoint_sha256": "f" * 64, "inference_epoch": 4,
            "effective_sinkhorn_epsilon": 0.1, "total_bases": 15,
            "npz_path": "/nonexistent.npz", "npz_sha256": "0" * 64}
    paths = {}
    for split in ("db", "query"):
        m = dict(fake, split=split)
        p = tmp_path / f"extraction_manifest_{split}.json"
        p.write_text(json.dumps(m))
        paths[split] = p
    (tmp_path / "extraction_complete.json").write_text(json.dumps({
        "schema_version": 1, "splits": ["db", "query"],
        "manifest_sha256": {s: sha256_file(str(p)) for s, p in paths.items()}}))
    with pytest.raises(ExtractionInvalid):
        validate_extraction_run(str(tmp_path))


def test_malformed_manifests_are_refused(tmp_path):
    """The §17.4 probe: two files containing `not-json`."""
    for split in ("db", "query"):
        (tmp_path / f"extraction_manifest_{split}.json").write_text("not-json")
    (tmp_path / "extraction_complete.json").write_text(json.dumps({
        "schema_version": 1, "splits": ["db", "query"],
        "manifest_sha256": {"db": "0" * 64, "query": "0" * 64}}))
    assert describe_failure(str(tmp_path)) is not None


def test_one_split_marker_is_refused(tmp_path):
    """The §17.6 probe: a marker naming only `db`."""
    cell = _cell(tmp_path, splits=("db",))
    with pytest.raises(ExtractionInvalid) as excinfo:
        validate_extraction_run(str(cell))
    assert "splits" in str(excinfo.value)


def test_marker_writer_refuses_a_partial_split_set(tmp_path):
    import extraction_siglip2 as E
    (tmp_path / "extraction_manifest_db.json").write_text("{}")
    with pytest.raises(E.ManifestBindingError):
        E._write_completion_marker(
            str(tmp_path), {"db": str(tmp_path / "extraction_manifest_db.json")})


# ------------------------------------------------------ artefact tamper

def test_tampered_npz_is_refused(tmp_path):
    cell = _cell(tmp_path)
    npz = cell / "extract_db.npz"
    npz.write_bytes(npz.read_bytes() + b"\x00")
    with pytest.raises(ExtractionInvalid) as excinfo:
        validate_extraction_run(str(cell))
    assert "hashes to" in str(excinfo.value)


def test_hash_that_is_not_the_encoding_is_refused(tmp_path):
    """Two arrays describing different codes previously passed on shape alone."""
    cell = _cell(tmp_path, corrupt="hash")
    with pytest.raises(ExtractionInvalid) as excinfo:
        validate_extraction_run(str(cell))
    assert "2-bit encoding" in str(excinfo.value)


def test_codebook_index_outside_the_codebook_is_refused(tmp_path):
    cell = _cell(tmp_path, corrupt="codebook")
    with pytest.raises(ExtractionInvalid) as excinfo:
        validate_extraction_run(str(cell))
    assert "codebook_indices outside" in str(excinfo.value)


def test_float_codes_are_refused(tmp_path):
    cell = _cell(tmp_path, corrupt="dtype")
    with pytest.raises(ExtractionInvalid) as excinfo:
        validate_extraction_run(str(cell))
    assert "dtype" in str(excinfo.value)


def test_splits_from_different_runtimes_are_refused(tmp_path):
    cell = _cell(tmp_path)
    path = cell / "extraction_manifest_query.json"
    manifest = json.loads(path.read_text())
    manifest["inference_epoch"] = 9
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    marker = cell / "extraction_complete.json"
    payload = json.loads(marker.read_text())
    payload["manifest_sha256"]["query"] = sha256_file(str(path))
    marker.write_text(json.dumps(payload))
    with pytest.raises(ExtractionInvalid) as excinfo:
        validate_extraction_run(str(cell))
    assert "disagree on inference_epoch" in str(excinfo.value)


# ---------------------------------------------------- backfill opt-in

def test_backfilled_requires_an_explicit_opt_in(tmp_path):
    cell = _cell(tmp_path)
    for split in ("db", "query"):
        path = cell / f"extraction_manifest_{split}.json"
        manifest = json.loads(path.read_text())
        manifest["backfilled"] = True
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    marker = cell / "extraction_complete.json"
    payload = json.loads(marker.read_text())
    payload["manifest_sha256"] = {
        s: sha256_file(str(cell / f"extraction_manifest_{s}.json"))
        for s in ("db", "query")}
    marker.write_text(json.dumps(payload))

    with pytest.raises(ExtractionInvalid):
        validate_extraction_run(str(cell))
    run = validate_extraction_run(str(cell), allow_backfilled=True)
    assert run.backfilled is True


# ------------------------------------------- the preserved Phase 2 cells

def test_the_preserved_phase2_cells_pass_the_strict_validator():
    """A golden test over the real diagnostic snapshot."""
    root = os.path.join(_REPO, "result_diagnostic", "phase2_F01_only")
    if not os.path.isdir(root):
        pytest.skip("Phase 2 snapshot not present")
    cells = sorted(d for d in os.listdir(root)
                   if os.path.isdir(os.path.join(root, d)))
    assert len(cells) == 15
    for cell in cells:
        run = validate_extraction_run(os.path.join(root, cell),
                                      allow_backfilled=True)
        assert run.backfilled is True
        assert run.common["total_bases"] == 15
        assert run.common["inference_epoch_source"] == "explicit_flag"
