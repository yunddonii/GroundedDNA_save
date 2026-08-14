"""§14.2: the canonical driver must refuse 24 bases instead of failing later.

24-base runs need `run_native_dna_p0_24`, which injects a pipeline variant and
extra implementation paths into the protocol identity. The canonical driver
happily accepts `GDNA_NATIVE_DNA_BASES=24`, runs the ordinary trainer, and
produces `pipeline_variant=None` with digest `6e2c1e51...` -- which is not in
the reviewed set for 24. The launcher then flips a return code of 0 into 3 at
the post-child gate, AFTER the training and extraction have been paid for, and
every later resume rejects the same cell.

Import and dry-run both succeed, so a fresh-process import test cannot catch
this. The refusal has to happen before a child is launched.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def _run(module: str, length: int, *args: str):
    env = {**os.environ, "GDNA_NATIVE_DNA_BASES": str(length)}
    return subprocess.run(
        [sys.executable, "-c",
         f"import runpy,sys; sys.argv=['{module}']+{list(args)!r}; "
         f"runpy.run_module('{module}', run_name='__main__')"],
        cwd=_REPO, env=env, capture_output=True, text=True, timeout=300)


@pytest.mark.parametrize("module", (
    "scripts.run_native_dna_p0",
    "scripts.run_native_dna_p0_matrix",
))
def test_canonical_entrypoint_refuses_24_bases(module):
    """It must name the wrapper rather than run and be rejected later."""
    # `--help` exits inside argparse, so drive main() directly: the guard must
    # fire before any child is launched, and import must stay total.
    proc = _run(module, 24)
    assert proc.returncode != 0, (
        "the canonical entrypoint accepted 24 bases; a fresh 24-base run would "
        "train, extract, and only then be failed by the lock gate")
    combined = proc.stdout + proc.stderr
    assert "24" in combined
    assert "run_native_dna_p0_24" in combined or "_24" in combined, (
        "the refusal must point at the 24-base wrapper")


@pytest.mark.parametrize("module,length", [
    ("scripts.run_native_dna_p0", 15),
    ("scripts.run_native_dna_p0", 18),
    ("scripts.run_native_dna_p0", 20),
    ("scripts.run_native_dna_p0_matrix", 15),
    ("scripts.run_native_dna_p0_matrix", 18),
    ("scripts.run_native_dna_p0_matrix", 20),
])
def test_canonical_entrypoint_still_serves_its_own_lengths(module, length):
    proc = _run(module, length, "--help")
    assert proc.returncode == 0, proc.stderr[-400:]


def test_canonical_24_identity_is_not_reviewed():
    """Pin WHY the refusal exists: the canonical identity at 24 carries no
    pipeline variant, so its digest can never be in the reviewed 24 set."""
    env = {**os.environ, "GDNA_NATIVE_DNA_BASES": "24"}
    probe = (
        "from pathlib import Path;"
        "import scripts.aggregate_native_dna_p0 as agg;"
        "import scripts.run_native_dna_p0 as d;"
        "from scripts.recompute_native_method_locks import EXECUTION, PROBE;"
        "i,_=d._protocol_identity(method='bee2018',dataset=PROBE['dataset'],"
        "setting='setting1',seed=42,"
        "cache_audit={'cache_dir':'x','blockers':[]},"
        "dataset_root=Path('dataset'),predictor=None,device='cuda:0',**EXECUTION);"
        "print(i.get('pipeline_variant'));"
        "print(agg._method_protocol_lock_digest(i) in "
        "agg.reviewed_method_locks(24)['bee2018'])"
    )
    proc = subprocess.run([sys.executable, "-c", probe], cwd=_REPO, env=env,
                          capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        pytest.skip("canonical import refuses 24, which is the fix itself")
    variant, reviewed = proc.stdout.strip().splitlines()[-2:]
    assert variant == "None"
    assert reviewed == "False"


@pytest.mark.parametrize("module", (
    "scripts.run_native_dna_p0_24",
    "scripts.run_native_dna_p0_matrix_24",
))
def test_the_24_wrappers_still_import_under_env24(module):
    """The wrappers import the canonical modules deliberately, so the guard has
    to check who is asking. A guard that also blocks the wrapper would leave no
    way to run 24 bases at all."""
    env = {**os.environ, "GDNA_NATIVE_DNA_BASES": "24"}
    proc = subprocess.run([sys.executable, "-c", f"import {module}"],
                          cwd=_REPO, env=env, capture_output=True, text=True,
                          timeout=300)
    assert proc.returncode == 0, proc.stderr[-400:]


def test_wrapper_context_still_reports_24():
    from scripts.run_native_dna_p0_24 import configured_canonical_driver
    import scripts.run_native_dna_p0 as driver
    with configured_canonical_driver():
        assert driver.effective_protocol().length_bases == 24
