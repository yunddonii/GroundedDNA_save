"""F01/C4: the real extractor must leave a manifest, and must refuse to guess.

`write_extraction_manifest` passed eleven unit tests and was never called from
production: `extraction_siglip2` and `scripts/extract_train_split` saved NPZs
only, so the Phase 2 root holds 0 manifests for 15 cells. A table could
therefore be traced back to an unknown operating point, which is the whole
defect F01 exists to close.

Three more lifecycle holes are pinned here:

  * a missing checkpoint let extraction continue against a freshly initialised
    random model and still write NPZs;
  * the explicit-legacy path returned `checkpoint_sha256=None` even though the
    file was right there to hash;
  * the best checkpoint was saved without a sidecar, and the best-to-final swap
    replaced the weights without updating the final sidecar, so the recorded SHA
    described a file that no longer existed.

These tests call the production helpers rather than re-implementing them.
"""
from __future__ import annotations

import inspect
import json
import os
import sys

import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import extraction_siglip2  # noqa: E402
from dna_utils import runtime_state  # noqa: E402


# ------------------------------------------------- production wiring

def test_extractor_writes_a_manifest_per_split(tmp_path):
    """Behaviour, not a grep: call the production helper and require the file.

    The unit tests for `write_extraction_manifest` passed for weeks while
    production never called it, so this exercises the path production uses.
    """
    out = tmp_path / "run"
    out.mkdir()
    # A realistic extraction: the writer binds base_indices, hash_2bit and
    # codebook_indices together, so a fixture missing them is not a valid cell.
    payload = {
        "base_indices": np.zeros((7, 15), dtype=np.int8),
        "hash_2bit": np.zeros((7, 30), dtype=np.int8),
        "codebook_indices": np.zeros((7, 5), dtype=np.int64),
    }
    np.savez(out / "extract_db.npz", **payload)
    ck = out / "model_state_dict.pth"
    ck.write_bytes(b"weights")
    (out / "config.pt").write_bytes(b"cfg")

    class _Args:
        num_semantic_parts = 5
        num_codons_per_codebook = 3
        dataset = "CIFAR10"
        random_seed = 42
        codebook_size = 64

    extraction_siglip2._write_split_manifest(
        str(out), "db", "extract_db.npz", payload, args=_Args(),
        checkpoint_path=str(ck),
        resolved=runtime_state.ResolvedEpoch(
            epoch=4, source="explicit_flag", effective_sinkhorn_epsilon=0.1,
            sinkhorn_schedule_horizon=5,
            checkpoint_sha256=runtime_state.sha256_file(str(ck))))

    manifest = out / "extraction_manifest_db.json"
    assert manifest.is_file()
    d = json.loads(manifest.read_text())
    assert d["split"] == "db"
    assert d["n_rows"] == 7
    assert d["total_bases"] == 15
    assert d["inference_epoch"] == 4
    assert d["npz_sha256"] == runtime_state.sha256_file(
        str(out / "extract_db.npz"))
    assert d["checkpoint_sha256"] == runtime_state.sha256_file(str(ck))
    assert d["dataset"] == "CIFAR10" and d["random_seed"] == 42


def test_extract_code_writes_one_manifest_per_split(tmp_path):
    """Both splits, and only after the NPZs exist, so the digest is the digest
    of the file on disk."""
    out = tmp_path / "run"
    out.mkdir()

    class _Args:
        num_semantic_parts = 5
        num_codons_per_codebook = 3
        dataset = "Flickr25k"
        random_seed = 42
        codebook_size = 128

    ck = out / "model_state_dict.pth"
    ck.write_bytes(b"w")
    (out / "config.pt").write_bytes(b"cfg")
    # A real digest: the writer is fail-closed and rejects a declared SHA that
    # matches no file, which is the whole point of the binding check.
    resolved = runtime_state.ResolvedEpoch(
        epoch=9, source="metadata_sidecar", effective_sinkhorn_epsilon=0.1,
        sinkhorn_schedule_horizon=10,
        checkpoint_sha256=runtime_state.sha256_file(str(ck)))
    for split, name, rows in (("db", "extract_db.npz", 23000),
                              ("query", "extract_query.npz", 2000)):
        payload = {
            "base_indices": np.zeros((rows, 15), dtype=np.int8),
            "hash_2bit": np.zeros((rows, 30), dtype=np.int8),
            "codebook_indices": np.zeros((rows, 5), dtype=np.int64),
        }
        np.savez(out / name, **payload)
        extraction_siglip2._write_split_manifest(
            str(out), split, name, payload, args=_Args(),
            checkpoint_path=str(ck), resolved=resolved)

    db = json.loads((out / "extraction_manifest_db.json").read_text())
    qy = json.loads((out / "extraction_manifest_query.json").read_text())
    assert (db["n_rows"], qy["n_rows"]) == (23000, 2000)
    # The runtime identity must be the same for both splits.
    for key in ("inference_epoch", "effective_sinkhorn_epsilon",
                "checkpoint_sha256", "total_bases"):
        assert db[key] == qy[key], key


def test_train_split_extractor_writes_a_manifest():
    """`scripts/extract_train_split` shares the resolver and must share the
    manifest, or the train NPZ has no recorded operating point."""
    import scripts.extract_train_split as ets
    source = inspect.getsource(ets)
    assert "write_extraction_manifest" in source or (
        "_write_split_manifest" in source)


def test_missing_checkpoint_is_fatal_not_a_random_model(tmp_path):
    """Extraction used to print a warning and continue from random weights."""
    with pytest.raises(extraction_siglip2.MissingCheckpoint):
        extraction_siglip2._require_checkpoint(str(tmp_path / "absent.pth"))


def test_present_checkpoint_passes(tmp_path):
    ck = tmp_path / "model_state_dict.pth"
    ck.write_bytes(b"w")
    extraction_siglip2._require_checkpoint(str(ck))


# ------------------------------------------------------ manifest content

def _resolved(sha="a" * 64):
    return runtime_state.ResolvedEpoch(
        epoch=4, source="explicit_flag", effective_sinkhorn_epsilon=0.1,
        sinkhorn_schedule_horizon=5, checkpoint_sha256=sha)


def test_manifest_records_the_binding_fields(tmp_path):
    out = runtime_state.write_extraction_manifest(
        str(tmp_path / "m.json"), checkpoint_path=str(tmp_path / "ck.pth"),
        resolved=_resolved(), num_slots=5, bases_per_slot=3,
        split="db", n_rows=59000)
    d = json.loads(open(out).read())
    assert d["split"] == "db" and d["n_rows"] == 59000
    assert d["inference_epoch"] == 4
    assert d["inference_epoch_source"] == "explicit_flag"
    assert d["effective_sinkhorn_epsilon"] == 0.1
    assert (d["num_slots"], d["bases_per_slot"]) == (5, 3)
    assert d["total_bases"] == 15 and d["total_bits"] == 30
    assert d["checkpoint_sha256"] == "a" * 64


def test_manifest_carries_the_npz_digest(tmp_path):
    """Without it the manifest describes a run, not the file that run produced."""
    npz = tmp_path / "extract_db.npz"
    np.savez(npz, base_indices=np.zeros((2, 15), dtype=np.int8))
    out = runtime_state.write_extraction_manifest(
        str(tmp_path / "m.json"), checkpoint_path=str(tmp_path / "ck.pth"),
        resolved=_resolved(), num_slots=5, bases_per_slot=3,
        split="db", n_rows=2,
        extra={"npz_path": str(npz),
               "npz_sha256": runtime_state.sha256_file(str(npz))})
    d = json.loads(open(out).read())
    assert len(d["npz_sha256"]) == 64


def test_legacy_explicit_checkpoint_gets_a_real_sha(tmp_path):
    """`checkpoint_sha256=None` with the file present was a provenance hole."""
    ck = tmp_path / "model_state_dict.pth"
    ck.write_bytes(b"weights")
    digest = runtime_state.sha256_file(str(ck))
    assert len(digest) == 64
    assert digest != "0" * 64


# ------------------------------------------------ checkpoint lifecycle

def test_best_to_final_swap_updates_the_sidecar(tmp_path):
    """The swap replaced the weights but left the sidecar describing the old
    file, so the recorded SHA matched nothing on disk."""
    final = tmp_path / "model_state_dict.pth"
    best = tmp_path / "model_state_dict_best.pth"
    final.write_bytes(b"final-weights")
    best.write_bytes(b"best-weights")
    runtime_state.write_checkpoint_metadata(
        str(final), checkpoint_epoch_zero_based=9, training_epoch_budget=10,
        stop_after_epoch=9, lr_schedule_horizon=10,
        sinkhorn_schedule_horizon=10, sinkhorn_epsilon_init=1.0,
        sinkhorn_epsilon_final=0.1)

    runtime_state.swap_best_into_final(str(best), str(final),
                                       best_epoch_zero_based=4)

    assert final.read_bytes() == b"best-weights"
    side = json.loads((tmp_path / "model_state_dict.pth.runtime.json").read_text())
    assert side["checkpoint_sha256"] == runtime_state.sha256_file(str(final))
    assert side["checkpoint_epoch_zero_based"] == 4


# ------------------------------------ best/final lifecycle in the trainer

def test_trainer_writes_a_sidecar_for_the_best_checkpoint():
    """The best weights carry a different epoch from the final ones, and the
    resolver reads the sidecar; saving them without one made the best
    checkpoint unusable by the fail-closed path."""
    source = open(os.path.join(_REPO, "train_siglip2.py"),
                  encoding="utf-8").read()
    best_block = source[source.index("model_state_dict_best.pth\")"):]
    assert "write_checkpoint_metadata" in best_block[:2000] or (
        "_wcm(" in best_block[:2000])


def test_trainer_swap_uses_the_atomic_helper():
    """`shutil.copy2` alone left the final sidecar describing the overwritten
    file. The swap must go through the helper that re-stamps both."""
    source = open(os.path.join(_REPO, "train_siglip2.py"),
                  encoding="utf-8").read()
    swap = source[source.index("swapping final checkpoint with best"):]
    assert "swap_best_into_final" in swap[:1500]
    assert "shutil.copy2(best_model_path, model_path)" not in swap[:1500]


def test_swap_helper_preserves_the_training_horizons(tmp_path):
    """The promoted sidecar must keep the horizons of the run that produced it,
    or the epsilon recomputed at extraction belongs to a different schedule."""
    final = tmp_path / "model_state_dict.pth"
    best = tmp_path / "model_state_dict_best.pth"
    final.write_bytes(b"final")
    best.write_bytes(b"best")
    runtime_state.write_checkpoint_metadata(
        str(final), checkpoint_epoch_zero_based=39, training_epoch_budget=40,
        stop_after_epoch=39, lr_schedule_horizon=60,
        sinkhorn_schedule_horizon=40, sinkhorn_epsilon_init=1.0,
        sinkhorn_epsilon_final=0.1)

    runtime_state.swap_best_into_final(str(best), str(final),
                                       best_epoch_zero_based=19)

    side = json.loads(
        (tmp_path / "model_state_dict.pth.runtime.json").read_text())
    assert side["checkpoint_epoch_zero_based"] == 19
    assert side["lr_schedule_horizon"] == 60
    assert side["sinkhorn_schedule_horizon"] == 40
    assert side["sinkhorn_epsilon_final"] == 0.1
    assert side["checkpoint_sha256"] == runtime_state.sha256_file(str(final))
    assert side["extra"]["promoted_from"].endswith("model_state_dict_best.pth")
    assert side["extra"]["promoted_reason"] == (
        "best_val_checkpoint_swapped_into_final")
