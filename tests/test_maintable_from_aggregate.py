"""F15: the main table may only be generated from an admissible aggregate.

The table was assembled by hand, so a diagnostic-only number could reach the
paper with nothing in the file recording that the matrix behind it was never
admissible. These tests pin the two properties that close that path: the
generator refuses when `paper_table_admission.eligible` is false and writes NO
file, and it never silently substitutes `diagnostic_mean_map_at_R_post` for the
eligible `mean_map_at_R_post` the aggregator deliberately leaves null.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from scripts.build_maintable_from_aggregate import (  # noqa: E402
    NotPaperEligible,
    admission_failure_reasons,
    render_tex,
)

_SCRIPT = os.path.join(_REPO, "scripts", "build_maintable_from_aggregate.py")


def _aggregate(*, eligible: bool, mean=0.7912, std=0.0021, diagnostic=False):
    entry = {
        "panel": "u0", "variant": "cibhash", "dataset": "Flickr25k", "bit": 30,
        "mean_map_at_R_post": None if diagnostic else mean,
        "sample_std_map_at_R_post": None if diagnostic else std,
        "diagnostic_mean_map_at_R_post": mean,
        "diagnostic_sample_std_map_at_R_post": std,
        "paper_table_eligible": eligible,
    }
    return {
        "protocol": {"requested_bit_slice": [30], "train_seeds": [42, 43, 44]},
        "paper_table_admission": {
            "eligible": eligible,
            "reason_codes": [] if eligible else ["not_all_records_paper_table_eligible"],
        },
        "aggregates": [entry],
    }


def _write(tmp_path, payload):
    path = tmp_path / "aggregate.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ------------------------------------------------------------------ refusal

def test_ineligible_aggregate_is_refused(tmp_path):
    payload = _aggregate(eligible=False, diagnostic=True)
    with pytest.raises(NotPaperEligible):
        render_tex(payload, source=_write(tmp_path, payload))


def test_refusal_writes_no_file(tmp_path):
    """A partial or stale TeX left behind would be worse than none."""
    src = _write(tmp_path, _aggregate(eligible=False, diagnostic=True))
    out = tmp_path / "table.tex"
    proc = subprocess.run(
        [sys.executable, _SCRIPT, str(src), "--out", str(out)],
        capture_output=True, text=True, cwd=_REPO, timeout=120)
    assert proc.returncode == 3
    assert not out.exists()
    assert "REFUSED" in proc.stderr


def test_reason_codes_are_reported(tmp_path):
    payload = _aggregate(eligible=False, diagnostic=True)
    assert admission_failure_reasons(payload) == [
        "not_all_records_paper_table_eligible"]


def test_missing_admission_block_is_a_refusal():
    assert admission_failure_reasons({"aggregates": []})


# ------------------------------------------------------- eligible generation

def test_eligible_aggregate_renders_the_value(tmp_path):
    payload = _aggregate(eligible=True)
    tex = render_tex(payload, source=_write(tmp_path, payload))
    assert "0.7912 \\pm 0.0021" in tex
    assert "CIBHash" in tex


def test_provenance_is_stamped(tmp_path):
    payload = _aggregate(eligible=True)
    src = _write(tmp_path, payload)
    tex = render_tex(payload, source=src)
    assert str(src) in tex
    assert "source_sha256:" in tex
    assert "[42, 43, 44]" in tex


def test_missing_cells_render_as_a_dash(tmp_path):
    payload = _aggregate(eligible=True)
    tex = render_tex(payload, source=_write(tmp_path, payload))
    mscoco_row = [line for line in tex.splitlines() if line.startswith("CIBHash")][0]
    assert mscoco_row.count("-") >= 3       # only Flickr25k has a number


def test_geometry_header_comes_from_the_declared_panel(tmp_path):
    payload = _aggregate(eligible=True)
    tex = render_tex(payload, source=_write(tmp_path, payload))
    assert "30 bit / 15 bases (5x3)" in tex


# --------------------------------------------- diagnostic must stay visible

def test_diagnostic_mean_is_never_substituted_silently(tmp_path):
    """`mean_map_at_R_post` is null exactly when the cell is diagnostic-only.
    Falling back to the diagnostic field without being asked would typeset a
    number the aggregator refused to certify."""
    payload = _aggregate(eligible=True, diagnostic=True)
    tex = render_tex(payload, source=_write(tmp_path, payload))
    assert "0.7912" not in tex


def test_allow_diagnostic_stamps_every_row(tmp_path):
    payload = _aggregate(eligible=False, diagnostic=True)
    tex = render_tex(payload, source=_write(tmp_path, payload),
                     allow_diagnostic=True)
    assert "0.7912" in tex
    assert "NOT PAPER-ELIGIBLE" in tex
    assert r"$^{\dagger}$" in tex


def test_allow_diagnostic_exits_zero(tmp_path):
    src = _write(tmp_path, _aggregate(eligible=False, diagnostic=True))
    out = tmp_path / "table.tex"
    proc = subprocess.run(
        [sys.executable, _SCRIPT, str(src), "--out", str(out),
         "--allow-diagnostic"],
        capture_output=True, text=True, cwd=_REPO, timeout=120)
    assert proc.returncode == 0
    assert "NOT PAPER-ELIGIBLE" in out.read_text(encoding="utf-8")
