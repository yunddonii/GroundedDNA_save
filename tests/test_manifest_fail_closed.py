"""§13.3/§13.4: a manifest must be checked against the files, not trusted.

`_write_split_manifest` recorded whatever it was handed. A probe wrote a
manifest whose declared checkpoint SHA matched no file, whose `n_rows` was 7
against an NPZ holding 2, and whose `total_bases` was 15 against a 12-wide
`base_indices` -- all of it accepted. A manifest that can disagree with the
artefacts it describes provides no provenance at all.

`train_siglip2` also stamped the best sidecar from `args.sinkhorn_eps` and
`args.sinkhorn_eps_final`, which do not exist; the real flags are
`--sinkhorn_epsilon_init` and `--sinkhorn_epsilon_final`. Every best-checkpoint
sidecar therefore recorded `None` for the initial, final and effective epsilon,
so re-inferring from a best checkpoint had no operating point to restore.
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

import extraction_siglip2 as E  # noqa: E402
from dna_utils import runtime_state as rs  # noqa: E402


class _Args:
    num_semantic_parts = 5
    num_codons_per_codebook = 3
    dataset = "CIFAR10"
    random_seed = 42
    codebook_size = 64


def _resolved(sha):
    return rs.ResolvedEpoch(epoch=4, source="explicit_flag",
                            effective_sinkhorn_epsilon=0.1,
                            sinkhorn_schedule_horizon=5,
                            checkpoint_sha256=sha)


def _cell(tmp_path, rows=2, width=15, slots=5, companions=True):
    ck = tmp_path / "ck.pth"
    ck.write_bytes(b"real-weights")
    (tmp_path / "config.pt").write_bytes(b"cfg")
    payload = {"base_indices": np.zeros((rows, width), dtype=np.int8)}
    if companions:
        payload["hash_2bit"] = np.zeros((rows, 2 * width), dtype=np.int8)
        payload["codebook_indices"] = np.zeros((rows, slots), dtype=np.int64)
    np.savez(tmp_path / "extract_db.npz", **payload)
    return ck, payload


def test_declared_checkpoint_sha_must_match_the_file(tmp_path):
    ck, payload = _cell(tmp_path)
    with pytest.raises(Exception) as excinfo:
        E._write_split_manifest(
            str(tmp_path), "db", "extract_db.npz", payload, args=_Args(),
            checkpoint_path=str(ck), resolved=_resolved("b" * 64))
    assert "sha" in str(excinfo.value).lower()
    assert not (tmp_path / "extraction_manifest_db.json").exists(), (
        "a rejected manifest must not be left behind")


def test_row_count_must_match_the_npz(tmp_path):
    ck, _ = _cell(tmp_path, rows=2)
    lying = {"base_indices": np.zeros((7, 15), dtype=np.int8)}
    with pytest.raises(Exception) as excinfo:
        E._write_split_manifest(
            str(tmp_path), "db", "extract_db.npz", lying, args=_Args(),
            checkpoint_path=str(ck),
            resolved=_resolved(rs.sha256_file(str(ck))))
    assert "row" in str(excinfo.value).lower()


def test_geometry_must_match_the_code_width(tmp_path):
    """M x L declared in args must equal the width actually written."""
    ck, payload = _cell(tmp_path, rows=2, width=12)
    with pytest.raises(Exception) as excinfo:
        E._write_split_manifest(
            str(tmp_path), "db", "extract_db.npz", payload, args=_Args(),
            checkpoint_path=str(ck),
            resolved=_resolved(rs.sha256_file(str(ck))))
    assert "base" in str(excinfo.value).lower()


def test_a_consistent_manifest_is_written(tmp_path):
    ck, payload = _cell(tmp_path, rows=2, width=15)
    out = E._write_split_manifest(
        str(tmp_path), "db", "extract_db.npz", payload, args=_Args(),
        checkpoint_path=str(ck), resolved=_resolved(rs.sha256_file(str(ck))))
    d = json.loads(open(out).read())
    assert d["checkpoint_sha256"] == rs.sha256_file(str(ck))
    assert d["n_rows"] == 2
    assert d["total_bases"] == 15
    assert d["config_sha256"] == rs.sha256_file(str(tmp_path / "config.pt"))
    assert d["npz_sha256"] == rs.sha256_file(str(tmp_path / "extract_db.npz"))


def test_missing_checkpoint_sha_is_refused(tmp_path):
    """A manifest that cannot name its weights is worthless."""
    ck, payload = _cell(tmp_path)
    with pytest.raises(Exception):
        E._write_split_manifest(
            str(tmp_path), "db", "extract_db.npz", payload, args=_Args(),
            checkpoint_path=str(ck), resolved=_resolved(None))


def test_no_annealing_resolver_still_names_the_checkpoint(tmp_path):
    """The earlier fix pinned a workaround: the resolver returned None and the
    writer refused, which made static-epsilon extraction impossible rather than
    fixing the resolver. The resolver hashes the file it loaded."""
    ck = tmp_path / "m.pth"
    ck.write_bytes(b"weights")

    class _NoAnneal:
        inference_epoch = None
        epoch = 5
        stop_after_epoch = 4
        sinkhorn_epsilon_init = None
        sinkhorn_epsilon_final = None
        lr_schedule_horizon = None
        sinkhorn_schedule_horizon = None

    resolved = rs.resolve_inference_epoch(str(ck), _NoAnneal())
    assert resolved.source == "no_annealing"
    assert resolved.checkpoint_sha256 == rs.sha256_file(str(ck))


def test_companion_arrays_must_match_the_geometry(tmp_path):
    """`hash_2bit` and `codebook_indices` are what retrieval and the codebook
    analyses read; a manifest that ignores them binds only a third of the NPZ."""
    ck, payload = _cell(tmp_path, rows=2, width=15, companions=False)
    with pytest.raises(Exception) as excinfo:
        E._write_split_manifest(
            str(tmp_path), "db", "extract_db.npz", payload, args=_Args(),
            checkpoint_path=str(ck),
            resolved=_resolved(rs.sha256_file(str(ck))))
    assert "hash_2bit" in str(excinfo.value)


def test_wrong_codebook_width_is_refused(tmp_path):
    ck = tmp_path / "ck.pth"
    ck.write_bytes(b"w")
    (tmp_path / "config.pt").write_bytes(b"cfg")
    payload = {
        "base_indices": np.zeros((2, 15), dtype=np.int8),
        "hash_2bit": np.zeros((2, 30), dtype=np.int8),
        "codebook_indices": np.zeros((2, 6), dtype=np.int64),   # 6 != 5 slots
    }
    np.savez(tmp_path / "extract_db.npz", **payload)
    with pytest.raises(Exception) as excinfo:
        E._write_split_manifest(
            str(tmp_path), "db", "extract_db.npz", payload, args=_Args(),
            checkpoint_path=str(ck),
            resolved=_resolved(rs.sha256_file(str(ck))))
    assert "codebook_indices" in str(excinfo.value)


def test_rank_error_is_a_domain_error_not_an_indexerror(tmp_path):
    ck = tmp_path / "ck.pth"
    ck.write_bytes(b"w")
    (tmp_path / "config.pt").write_bytes(b"cfg")
    np.savez(tmp_path / "extract_db.npz",
             base_indices=np.zeros(15, dtype=np.int8))
    with pytest.raises(E.ManifestBindingError) as excinfo:
        E._write_split_manifest(
            str(tmp_path), "db", "extract_db.npz",
            {"base_indices": np.zeros(15, dtype=np.int8)}, args=_Args(),
            checkpoint_path=str(ck),
            resolved=_resolved(rs.sha256_file(str(ck))))
    assert "rank" in str(excinfo.value)


def test_nested_legacy_config_is_recorded(tmp_path):
    """Resume supports `<out>/model_state/config.pt`; recording only the flat
    path left a nested run with a wrong path and a null SHA."""
    ck, payload = _cell(tmp_path)
    (tmp_path / "config.pt").unlink()
    nested = tmp_path / "model_state"
    nested.mkdir()
    (nested / "config.pt").write_bytes(b"nested-cfg")
    out = E._write_split_manifest(
        str(tmp_path), "db", "extract_db.npz", payload, args=_Args(),
        checkpoint_path=str(ck), resolved=_resolved(rs.sha256_file(str(ck))))
    d = json.loads(open(out).read())
    assert d["config_path"].endswith("model_state/config.pt")
    assert d["config_sha256"] == rs.sha256_file(str(nested / "config.pt"))


# ------------------------------------------------------- best sidecar fields

def test_best_sidecar_uses_the_real_epsilon_flag_names():
    """`args.sinkhorn_eps` does not exist -- the flag is
    `--sinkhorn_epsilon_init` -- so every best sidecar stored None."""
    source = open(os.path.join(_REPO, "train_siglip2.py"),
                  encoding="utf-8").read()
    best = source[source.index("checkpoint_role"):]
    window = source[max(0, source.index("checkpoint_role") - 1200):
                    source.index("checkpoint_role")]
    assert 'getattr(args, "sinkhorn_eps"' not in window
    assert 'getattr(args, "sinkhorn_eps_final"' not in window
    assert "sinkhorn_epsilon_init" in window
    assert "sinkhorn_epsilon_final" in window


def test_config_declares_the_epsilon_flags_this_code_reads():
    from config import Config
    parser = Config.build_parser()
    names = {a.dest for a in parser._actions}
    assert "sinkhorn_epsilon_init" in names
    assert "sinkhorn_epsilon_final" in names
    assert "sinkhorn_eps" not in names


# ------------------------------------------ completion marker and isolation

def test_completion_marker_binds_both_manifests(tmp_path):
    """Written last, so a consumer that sees it knows both splits validated.
    Without it an interrupted run leaves one manifest and one NPZ and looks
    partially complete to everything downstream."""
    manifests = {}
    for split, name in (("db", "extract_db.npz"), ("query", "extract_query.npz")):
        ck, payload = _cell(tmp_path)
        (tmp_path / name).write_bytes((tmp_path / "extract_db.npz").read_bytes())
        manifests[split] = E._write_split_manifest(
            str(tmp_path), split, name, payload, args=_Args(),
            checkpoint_path=str(ck), resolved=_resolved(rs.sha256_file(str(ck))))

    marker = E._write_completion_marker(str(tmp_path), manifests)
    d = json.loads(open(marker).read())
    assert sorted(d["splits"]) == ["db", "query"]
    for split, path in manifests.items():
        assert d["manifest_sha256"][split] == rs.sha256_file(path)


def test_wrapper_dispatch_is_restored_after_the_context():
    """§15.6: `WRAPPER_DISPATCH` is a shared module global. Sequential CLI use
    is fine because the context restores it, but a nested or concurrent library
    caller would otherwise observe the wrapper's value."""
    import scripts.run_native_dna_p0 as driver
    from scripts.run_native_dna_p0_24 import configured_canonical_driver

    before = driver.WRAPPER_DISPATCH
    with configured_canonical_driver():
        assert driver.WRAPPER_DISPATCH is True
    assert driver.WRAPPER_DISPATCH == before


def test_nested_wrapper_contexts_restore_correctly():
    import scripts.run_native_dna_p0 as driver
    from scripts.run_native_dna_p0_24 import configured_canonical_driver

    before = driver.MATCHED_LENGTH
    with configured_canonical_driver():
        with configured_canonical_driver():
            assert driver.MATCHED_LENGTH == 24
        assert driver.MATCHED_LENGTH == 24, (
            "the inner context must not restore the outer one's patch early")
    assert driver.MATCHED_LENGTH == before
