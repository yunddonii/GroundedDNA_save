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


def _cell(tmp_path, rows=2, width=15):
    ck = tmp_path / "ck.pth"
    ck.write_bytes(b"real-weights")
    payload = {"base_indices": np.zeros((rows, width), dtype=np.int8)}
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
    (tmp_path / "config.pt").write_bytes(b"cfg")
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
    """The no-annealing resolver returns `checkpoint_sha256=None` even with the
    file present; a manifest that cannot name its weights is worthless."""
    ck, payload = _cell(tmp_path)
    with pytest.raises(Exception):
        E._write_split_manifest(
            str(tmp_path), "db", "extract_db.npz", payload, args=_Args(),
            checkpoint_path=str(ck), resolved=_resolved(None))


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
