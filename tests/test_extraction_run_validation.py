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
from pathlib import Path
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
from dna_utils.runtime_state import (  # noqa: E402
    ResolvedEpoch,
    sha256_file,
    write_extraction_manifest,
)

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
        # Built by the PRODUCTION writer, not hand-rolled. A hand-written
        # fixture drifts from the schema silently -- and a fixture that is
        # already invalid proves nothing about the counterexample under test.
        path = write_extraction_manifest(
            str(tmp_path / f"extraction_manifest_{split}.json"),
            checkpoint_path=str(ck),
            resolved=ResolvedEpoch(
                epoch=4, source="explicit_flag",
                effective_sinkhorn_epsilon=0.1,
                sinkhorn_schedule_horizon=5,
                checkpoint_sha256=sha256_file(str(ck)),
                sinkhorn_annealing_enabled=True),
            num_slots=SLOTS, bases_per_slot=PER_SLOT, split=split,
            n_rows=rows, lr_schedule_horizon=5, training_epoch_budget=5,
            training_stop_epoch=4,
            extra={
                "config_path": str(cfg),
                "config_sha256": sha256_file(str(cfg)),
                "dataset": "CIFAR10", "random_seed": 42, "codebook_size": 64,
                # `backfilled` is a required bool: a missing field used to read
                # as false, so deleting it promoted a backfilled cell to native.
                "backfilled": False,
                "npz_path": str(npz), "npz_sha256": sha256_file(str(npz)),
            })
        manifests[split] = Path(path)
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
    # A value that is schema-valid on its own, so this test exercises the
    # cross-split check rather than being caught earlier by the schema.
    manifest["random_seed"] = 43
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    marker = cell / "extraction_complete.json"
    payload = json.loads(marker.read_text())
    payload["manifest_sha256"]["query"] = sha256_file(str(path))
    marker.write_text(json.dumps(payload))
    with pytest.raises(ExtractionInvalid) as excinfo:
        validate_extraction_run(str(cell))
    assert "disagree on random_seed" in str(excinfo.value)


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


# ------------------------------------------- metric binding (§17.3 part two)

def test_metric_without_a_binding_is_refused(tmp_path):
    """A metric file that names no inputs can describe a different extraction
    entirely while sitting beside a valid one."""
    from dna_utils.extraction_validation import check_metric_input_binding
    cell = _cell(tmp_path)
    with pytest.raises(ExtractionInvalid) as excinfo:
        check_metric_input_binding(str(cell), {"mAP_at_R": 0.9},
                                   what="cell_result.json",
                                   allow_backfilled=False)
    assert "input_binding" in str(excinfo.value)


def test_metric_bound_to_other_files_is_refused(tmp_path):
    from dna_utils.extraction_validation import (
        check_metric_input_binding, metric_input_binding)
    cell = _cell(tmp_path)
    binding = metric_input_binding(str(cell))
    binding["npz_sha256"]["db"] = "0" * 64        # a different extraction
    with pytest.raises(ExtractionInvalid) as excinfo:
        check_metric_input_binding(str(cell), {"input_binding": binding},
                                   what="cell_result.json",
                                   allow_backfilled=False)
    assert "npz_sha256" in str(excinfo.value)


def test_matching_binding_is_accepted(tmp_path):
    from dna_utils.extraction_validation import (
        check_metric_input_binding, metric_input_binding)
    cell = _cell(tmp_path)
    check_metric_input_binding(
        str(cell), {"input_binding": metric_input_binding(str(cell))},
        what="cell_result.json", allow_backfilled=False)


def test_binding_follows_the_npz_not_the_filename(tmp_path):
    """Rewriting the NPZ must invalidate a metric computed from the old one."""
    from dna_utils.extraction_validation import (
        check_metric_input_binding, metric_input_binding)
    cell = _cell(tmp_path)
    stale = {"input_binding": metric_input_binding(str(cell))}
    npz = cell / "extract_db.npz"
    npz.write_bytes(npz.read_bytes() + b"\x00")
    with pytest.raises(ExtractionInvalid):
        check_metric_input_binding(str(cell), stale, what="cell_result.json",
                                   allow_backfilled=False)


# ------------------------------------------------- train split (§17.7)

def test_marker_can_cover_three_splits(tmp_path):
    """Adding train re-commits the marker over all three splits, so a DB/query
    marker cannot stay valid while train is stale or from another runtime."""
    cell = _cell(tmp_path, splits=("db", "query", "train"))
    run = validate_extraction_run(
        str(cell), required_splits=("db", "query", "train"))
    assert sorted(run.splits) == ["db", "query", "train"]


def test_a_three_split_run_is_not_admitted_as_two(tmp_path):
    cell = _cell(tmp_path, splits=("db", "query", "train"))
    with pytest.raises(ExtractionInvalid):
        validate_extraction_run(str(cell), required_splits=("db", "query"))


# ------------------------------------------ §18.3 adversarial identities

def _mutate(tmp_path, manifest_fn=None, marker_fn=None, **kw):
    cell = _cell(tmp_path, **kw)
    for split in ("db", "query"):
        path = cell / f"extraction_manifest_{split}.json"
        manifest = json.loads(path.read_text())
        if manifest_fn:
            manifest_fn(manifest)
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    marker_path = cell / "extraction_complete.json"
    marker = json.loads(marker_path.read_text())
    marker["manifest_sha256"] = {
        s: sha256_file(str(cell / f"extraction_manifest_{s}.json"))
        for s in ("db", "query")}
    if marker_fn:
        marker_fn(marker)
    marker_path.write_text(json.dumps(marker))
    return cell


@pytest.mark.parametrize("name,manifest_fn,marker_fn", [
    ("bogus epoch source",
     lambda m: m.update(inference_epoch_source="bogus"), None),
    ("marker schema 999", None, lambda k: k.update(schema_version=999)),
    ("duplicate split in marker", None,
     lambda k: k.update(splits=["db", "query", "db"])),
    ("fields deleted from both manifests",
     lambda m: [m.pop(f, None) for f in ("dataset", "random_seed",
                                         "total_bases")], None),
    ("codebook_size is a string",
     lambda m: m.update(codebook_size="not-an-int"), None),
    ("total_bits inconsistent with geometry",
     lambda m: m.update(total_bits=1), None),
    ("null field", lambda m: m.update(dataset=None), None),
    ("bool where int expected", lambda m: m.update(random_seed=True), None),
    ("stop epoch outside the budget",
     lambda m: m.update(training_stop_epoch=99), None),
])
def test_consistently_wrong_identities_are_refused(tmp_path, name,
                                                   manifest_fn, marker_fn):
    """§18.3: split-vs-split equality cannot catch a cell that is wrong the
    same way on both sides. Deleting a field from both manifests passed as
    `None == None`, and nothing said what the cell was supposed to be."""
    cell = _mutate(tmp_path, manifest_fn, marker_fn)
    with pytest.raises(ExtractionInvalid):
        validate_extraction_run(str(cell))


def test_expected_identity_catches_a_self_consistent_wrong_cell(tmp_path):
    from dna_utils.extraction_validation import ExpectedIdentity
    cell = _cell(tmp_path)
    validate_extraction_run(str(cell))                      # internally fine
    with pytest.raises(ExtractionInvalid) as excinfo:
        validate_extraction_run(
            str(cell), expected=ExpectedIdentity(dataset="MSCOCO"))
    assert "expects 'MSCOCO'" in str(excinfo.value)


def test_mixed_backfill_state_is_refused(tmp_path):
    """One split recorded by its extraction and the other bound afterwards is
    not one run's provenance."""
    def only_db(manifest):
        manifest["backfilled"] = manifest["split"] == "db"
    cell = _mutate(tmp_path, only_db)
    with pytest.raises(ExtractionInvalid) as excinfo:
        validate_extraction_run(str(cell), allow_backfilled=True)
    assert "mixed provenance" in str(excinfo.value)


def test_describe_failure_never_leaks_a_raw_exception(tmp_path):
    """The shell reads this string; a KeyError traceback is not a refusal."""
    cell = _cell(tmp_path)
    path = cell / "extraction_manifest_db.json"
    path.write_text(json.dumps({"split": "db"}))
    marker = cell / "extraction_complete.json"
    payload = json.loads(marker.read_text())
    payload["manifest_sha256"]["db"] = sha256_file(str(path))
    marker.write_text(json.dumps(payload))
    reason = describe_failure(str(cell))
    assert reason and "missing" in reason


# ---------------------------------------------- §18.6 shared array contract

def test_hash_wraparound_cannot_pass_as_valid(tmp_path):
    """`astype(uint8)` folds 256 to 0, so a cast-then-compare check passed
    arrays that are not the encoding at all."""
    from dna_utils.extraction_validation import validate_code_arrays
    base = np.zeros((2, 15), dtype=np.int64)
    hashed = base_indices_to_2bit(base).astype(np.int64)
    hashed[0, 0] = 256
    with pytest.raises(ExtractionInvalid) as excinfo:
        validate_code_arrays(base, hashed, np.zeros((2, 5), dtype=np.int64),
                             rows=2, slots=5, per_slot=3, codebook_size=64,
                             split="db")
    assert "outside 0..1" in str(excinfo.value)


def test_missing_codebook_size_is_refused_not_skipped(tmp_path):
    from dna_utils.extraction_validation import validate_code_arrays
    base = np.zeros((2, 15), dtype=np.int64)
    with pytest.raises(ExtractionInvalid) as excinfo:
        validate_code_arrays(base, base_indices_to_2bit(base),
                             np.full((2, 5), 999, dtype=np.int64),
                             rows=2, slots=5, per_slot=3, codebook_size=None,
                             split="db")
    assert "codebook_size" in str(excinfo.value)
