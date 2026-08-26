"""A run must not be able to consume a cache it cannot account for.

Two defects of the same shape, both found after the caches were rebuilt:

* The rebuilt provenance-bearing caches landed under
  `/data/yschoi/groundeddna_cache_v6prov` while every runner still pointed at
  the old `./cache/...` paths. Regenerating a cache and using it are different
  things, and nothing checked which one a training run had loaded.

* All four rebuilt caches were first written with `leakage_free_fit: false`:
  `build_clip_cache_v6prov.sh` fitted the whitening matrix over every cache row
  instead of restricting it to the optimization-train split, so the transform
  observed the validation rows that pick the epoch and the query/DB rows that
  are the reported number.

Both gates are fail-closed and both live where the artefact is LOADED, because
that is the one place all eleven un-updated runners have to pass through.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dna_utils.cache_provenance import (  # noqa: E402
    OPT_OUT_ENV,
    CacheProvenanceMissing,
    require_cache_provenance,
    require_leakage_free_whitening,
)

_REVISION = "57c216476eefef5ab752ec549e440a49ae4ae5f3"

_GOOD_META = {
    "canonical_transform": {"image_size": 224, "interpolation": "bicubic"},
    "hf_provenance": {"model_name": "openai/clip-vit-base-patch16",
                      "model_revision": _REVISION},
}


@pytest.fixture(autouse=True)
def _no_opt_out(monkeypatch):
    monkeypatch.delenv(OPT_OUT_ENV, raising=False)


# ------------------------------------------------------- the cache gate

def test_a_fully_provenanced_cache_is_admitted():
    record = require_cache_provenance("probe", _GOOD_META)
    assert record["model_revision"] == _REVISION
    assert record["canonical_transform"] is True
    assert record["unprovenanced_opt_out"] is False


@pytest.mark.parametrize("meta,expected", [
    ({}, "canonical_transform"),
    ({"canonical_transform": {"image_size": 224}}, "hf_provenance"),
    ({**_GOOD_META, "hf_provenance": {"model_revision": "main"}}, "immutable"),
    ({**_GOOD_META, "hf_provenance": {"model_revision": "57c2164"}},
     "immutable"),
    ({**_GOOD_META, "canonical_transform": "224"}, "canonical_transform"),
])
def test_an_unaccountable_cache_is_refused(meta, expected):
    with pytest.raises(CacheProvenanceMissing) as excinfo:
        require_cache_provenance("probe", meta)
    assert expected in str(excinfo.value)


def test_a_branch_name_is_not_a_revision():
    """`main` moves; a paper cannot cite it."""
    with pytest.raises(CacheProvenanceMissing):
        require_cache_provenance(
            "probe", {**_GOOD_META,
                      "hf_provenance": {"model_revision": "main"}})


def test_the_opt_out_admits_but_announces(monkeypatch, capsys):
    monkeypatch.setenv(OPT_OUT_ENV, "1")
    record = require_cache_provenance("probe", {})
    assert record["unprovenanced_opt_out"] is True
    out = capsys.readouterr().out
    assert "NOT paper-table eligible" in out


# ------------------------------------------------- the whitening gate

def _whiten(tmp_path: Path, *, row_index: bool = True, **meta) -> str:
    npz = tmp_path / "text_whiten.npz"
    np.savez(npz, mu=np.zeros(4, np.float32), U=np.eye(4, dtype=np.float32),
             S=np.ones(4, np.float32))
    if row_index:
        rows = tmp_path / "opt_train_rows.npy"
        np.save(rows, np.arange(3))
        meta.setdefault("row_index_npy", str(rows))
    (tmp_path / "text_whiten.npz.meta.json").write_text(json.dumps(meta))
    return str(npz)


def test_a_restricted_fit_is_admitted(tmp_path):
    path = _whiten(tmp_path, leakage_free_fit=True, rows_used=54000)
    assert require_leakage_free_whitening(path)["rows_used"] == 54000


def test_an_unrestricted_fit_is_refused(tmp_path):
    """Exactly what the rebuilt caches shipped with."""
    path = _whiten(tmp_path, leakage_free_fit=False, rows_used=122218)
    with pytest.raises(CacheProvenanceMissing) as excinfo:
        require_leakage_free_whitening(path)
    assert "leakage_free_fit" in str(excinfo.value)


@pytest.mark.parametrize("flag", ["false", "true", 1, "yes"])
def test_a_non_boolean_flag_is_refused(tmp_path, flag):
    """The JSON string "false" is truthy; only a literal true is a fit."""
    path = _whiten(tmp_path, leakage_free_fit=flag, rows_used=100)
    with pytest.raises(CacheProvenanceMissing):
        require_leakage_free_whitening(path)


def test_a_restriction_naming_a_missing_index_is_refused(tmp_path):
    """`row_index_npy` has to be re-checkable, not merely recorded."""
    path = _whiten(tmp_path, row_index=False, leakage_free_fit=True,
                   rows_used=100, row_index_npy="/nonexistent/rows.npy")
    with pytest.raises(CacheProvenanceMissing) as excinfo:
        require_leakage_free_whitening(path)
    assert "row_index_npy" in str(excinfo.value)


@pytest.mark.parametrize("rows", [0, -5, None, "9000"])
def test_a_nonsensical_row_count_is_refused(tmp_path, rows):
    path = _whiten(tmp_path, leakage_free_fit=True, rows_used=rows)
    with pytest.raises(CacheProvenanceMissing):
        require_leakage_free_whitening(path)


def test_a_whitening_matrix_with_no_metadata_is_refused(tmp_path):
    npz = tmp_path / "text_whiten.npz"
    np.savez(npz, mu=np.zeros(4, np.float32))
    with pytest.raises(CacheProvenanceMissing) as excinfo:
        require_leakage_free_whitening(str(npz))
    assert "unknown" in str(excinfo.value)


def test_an_older_matrix_predating_the_field_is_refused(tmp_path):
    """A missing flag is not a passing flag."""
    path = _whiten(tmp_path, rows_used=122218)
    with pytest.raises(CacheProvenanceMissing):
        require_leakage_free_whitening(path)


# --------------------------------------- the gates are wired to the loader

def _minimal_cache(tmp_path: Path, meta: dict) -> Path:
    """The smallest directory `_SigLIP2FeatureCache` will open."""
    rows, tokens, hv, dproj = 2, 3, 4, 5
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    (tmp_path / "image_ids.json").write_text(json.dumps(["a" * 16, "b" * 16]))
    np.save(tmp_path / "visual_tokens.f16.npy",
            np.zeros((rows, tokens, hv), np.float16))
    np.save(tmp_path / "visual_global.f16.npy",
            np.zeros((rows, dproj), np.float16))
    np.save(tmp_path / "text_part.f16.npy",
            np.zeros((rows, 6, dproj), np.float16))
    np.save(tmp_path / "has_text.bool.npy", np.ones(rows, bool))
    return tmp_path


def test_the_loader_refuses_an_unprovenanced_cache(tmp_path):
    """The gate must fire in production, not only when called directly."""
    from dataloaders import _SigLIP2FeatureCache

    cache = _minimal_cache(tmp_path, {"N": 2})
    with pytest.raises(CacheProvenanceMissing):
        _SigLIP2FeatureCache(str(cache))


def test_the_loader_admits_a_provenanced_cache(tmp_path):
    from dataloaders import _SigLIP2FeatureCache

    cache = _minimal_cache(tmp_path, dict(_GOOD_META, N=2))
    loaded = _SigLIP2FeatureCache(str(cache))
    assert loaded.provenance["model_revision"] == _REVISION


# ------------------------------------------- the caches actually on disk

_V6PROV = Path("/data/yschoi/groundeddna_cache_v6prov")


@pytest.mark.parametrize("slug", ["cifar10", "flickr25k", "nuswide", "mscoco"])
def test_the_rebuilt_caches_pass_both_gates(slug):
    pooled = _V6PROV / f"{slug}_clip"
    tokens = _V6PROV / f"{slug}_clip_tokens"
    if not pooled.is_dir() or not tokens.is_dir():
        pytest.skip(f"{slug} is not built on this machine")
    for directory in (pooled, tokens):
        meta = json.loads((directory / "meta.json").read_text())
        require_cache_provenance(str(directory), meta)
    whiten = tokens / "text_whiten.npz"
    if whiten.is_file():
        record = require_leakage_free_whitening(str(whiten))
        assert record["rows_used"] > 0
        # The restriction must name this cache's own split index.
        assert record["row_index_npy"].endswith("opt_train_rows.npy")


def test_a_configured_but_missing_cache_directory_is_refused(tmp_path):
    """The gate's biggest hole: you could bypass it by pointing at nothing.

    `if dir is not None and os.path.isdir(dir)` read a missing directory as "no
    cache requested", so a typo or a moved cache root silently switched the run
    to live images -- and a run that never loads a cache can never be refused
    for the cache it loaded.
    """
    from dataloaders import _require_cache_dir_present

    _require_cache_dir_present(None)                      # no cache asked for
    _require_cache_dir_present(str(tmp_path))             # present
    with pytest.raises(FileNotFoundError) as excinfo:
        _require_cache_dir_present(str(tmp_path / "gone"))
    assert "does not exist" in str(excinfo.value)


# ------------------------------------------- the overlay launcher (§29.6)

_LAUNCHER = REPO / "scripts" / "build_foil_overlays_v6prov.sh"


def test_the_overlay_launcher_reports_a_failing_child(tmp_path):
    """A bare `wait` returns 0 whatever the children did.

    The first version printed "all datasets finished" after a child had exited
    1, and the overlay it implied was in fact built by a separate hand-run
    retry.
    """
    script = tmp_path / "probe.sh"
    script.write_text(
        "set -uo pipefail\n"
        "( exit 7 ) &\n"
        "A=$!\n"
        "( exit 0 ) &\n"
        "B=$!\n"
        "FAILED=0\n"
        "wait $A || FAILED=1\n"
        "wait $B || FAILED=1\n"
        "exit $FAILED\n")
    proc = subprocess.run(["bash", str(script)], capture_output=True,
                          text=True, timeout=60)
    assert proc.returncode == 1, "per-child wait must surface the failure"


def test_the_overlay_launcher_supplies_the_cifar_foil_jsonl():
    """CIFAR's overlay needs the v4 foils, or a standalone run cannot rebuild it.

    The checked-in `cifar10.foils.jsonl` was generated from
    `cifar10_qwen.jsonl`; the paper's CIFAR runs record `cifar10_qwen_v4.jsonl`
    and the rebuilt cache is built from it, so the builder refuses the pair.
    """
    source = _LAUNCHER.read_text()
    assert "cifar10_v4.foils.jsonl" in source


def test_the_optional_foil_variable_actually_reaches_the_child():
    """`${FOIL:+FOIL_JSONL="$FOIL"}` looked right and exited 127.

    Bash expands the conditional and then looks for a COMMAND by the resulting
    name; it does not re-read it as an assignment word. A source-substring test
    saw `FOIL_JSONL=` in the file and passed while a clean CIFAR launch could
    not start at all -- exactly the grep-shaped test this repo keeps getting
    burned by. This runs the construct.
    """
    script = (
        'FOIL=/tmp/example.jsonl\n'
        'ENVV=(A=1)\n'
        '[[ -n "$FOIL" ]] && ENVV+=(FOIL_JSONL="$FOIL")\n'
        'env "${ENVV[@]}" bash -c \'echo "got=$FOIL_JSONL"\'\n'
    )
    proc = subprocess.run(["bash", "-c", script], capture_output=True,
                          text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert "got=/tmp/example.jsonl" in proc.stdout

    broken = (
        'FOIL=/tmp/example.jsonl\n'
        'A=1 ${FOIL:+FOIL_JSONL="$FOIL"} bash -c \'true\'\n'
    )
    bad = subprocess.run(["bash", "-c", broken], capture_output=True,
                         text=True, timeout=60)
    assert bad.returncode == 127, "the broken form must still be broken"


def test_the_overlay_launcher_exits_nonzero_when_a_dataset_is_missing(tmp_path):
    """A missing input used to be a silent skip inside an all-finished report."""
    source = _LAUNCHER.read_text()
    assert "NOT all datasets built" in source
    for phrase in ("missing base cache", "missing caption cache",
                   "missing foil jsonl"):
        assert phrase in source
