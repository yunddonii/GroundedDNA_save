"""F09: the A2/A4/A5 chain, whose commands were wrong in three different ways.

  A2 zeroed three text losses and left `--disable_text_supervision` off, so
     caption-derived routing and token pruning stayed active. The cell claimed
     "no text path" while text still shaped the codes.
  A4 passed `--num_codebooks 1` against M=5 inputs, which raises inside the
     quantizer's first forward.
  A5 passed `--router_type mean`, which argparse rejects outright -- and the
     paper's A5 is not a router ablation at all but the 2x2 factorial of
     `L_joint` and `noGumbel`, whose point is that noGumbel alone is harmful on
     some datasets and only helps in combination.

Two of the three could not have run, and would have failed at the first cell,
hours into a chain that had already trained the ones before it.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
CHAIN = REPO / "scripts" / "auto_chain_after_3seed.sh"
PREFLIGHT = REPO / "scripts" / "_ablation_preflight.py"
PY = os.environ.get("PY", sys.executable)

_S5 = ["--num_semantic_parts", "5", "--num_codebooks", "5"]


def _preflight(*flags: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, GDNA_NUM_SEMANTIC_PARTS="5")
    return subprocess.run(
        [PY, str(PREFLIGHT), "--", *_S5, *flags],
        capture_output=True, text=True, cwd=str(REPO), env=env, timeout=300)


# ------------------------------------------- the commands, actually parsed

def test_the_router_ablation_that_never_existed_is_refused():
    """`mean` is not one of sinkhorn/attention/slot/cluster_attn/cross_attn."""
    proc = _preflight("--router_type", "mean")
    assert proc.returncode == 1
    assert "refused" in proc.stderr


def test_a_codebook_count_that_disagrees_with_M_is_refused():
    """`--num_codebooks 1` raises in the quantizer, not in argparse."""
    proc = subprocess.run(
        [PY, str(PREFLIGHT), "--", "--num_semantic_parts", "5",
         "--num_codebooks", "1"],
        capture_output=True, text=True, cwd=str(REPO), timeout=300,
        env=dict(os.environ, GDNA_NUM_SEMANTIC_PARTS="5"))
    assert proc.returncode == 1
    assert "--share_codebook" in proc.stderr


@pytest.mark.parametrize("flags", [
    ["--share_codebook"],
    ["--disable_text_supervision"],
    ["--no_gumbel_softmax"],
    ["--lambda_codon_joint", "0.0"],
])
def test_the_corrected_ablation_flags_parse(flags):
    proc = _preflight(*flags)
    assert proc.returncode == 0, proc.stderr


# ------------------------------------------------- what the chain now runs

def _commands() -> str:
    """The lines bash executes -- comments explain the old bugs by name, so
    asserting over the whole file would match its own explanation."""
    return "\n".join(line for line in CHAIN.read_text().splitlines()
                      if not line.strip().startswith("#"))


def test_a2_actually_disables_the_text_path():
    a2 = [line for line in _commands().splitlines() if '"A2:' in line]
    assert len(a2) == 2, "A2 appears in the preflight and in the run loop"
    for line in a2:
        assert "--disable_text_supervision" in line, (
            "zeroing the text losses leaves caption-derived routing active")


def test_a4_ties_the_codebooks_rather_than_shrinking_them():
    commands = _commands()
    assert "--num_codebooks 1" not in commands
    assert commands.count('"A4:--share_codebook"') == 2


def test_a5_is_the_two_by_two_factorial_the_paper_reports():
    """One cell would hide the interaction the ablation exists to show."""
    commands = _commands()
    assert "--router_type mean" not in commands
    for cell in ("A5none", "A5joint", "A5nogumbel", "A5both"):
        assert f'"{cell}:' in commands, f"{cell} missing from the factorial"
    # noGumbel alone is one of the four, because alone it is sometimes harmful.
    assert '"A5nogumbel:--lambda_codon_joint 0.0 --no_gumbel_softmax"' in commands


def test_the_chain_stops_on_a_failed_step():
    source = _commands()
    assert "set -Eeuo pipefail" in source
    assert "trap " in source
    # Background children are waited on individually; `while busy` watches
    # process names and GPU memory, which is not the same as success.
    assert 'wait "$pid"' in source
    assert "step 2 INCOMPLETE" in source


def test_the_reference_run_is_resolved_uniquely():
    source = CHAIN.read_text()  # comments checked separately below
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        assert not ("head -1" in stripped and "ls " in stripped), \
            f"still selects a run by recency: {stripped}"
    assert 'printf "[auto-chain] %d reference runs match' in source


def test_the_chain_still_parses():
    proc = subprocess.run(["bash", "-n", str(CHAIN)], capture_output=True,
                          text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr


def test_rundir_refuses_ambiguity(tmp_path):
    """Executed: two matching directories must not resolve to the newer one."""
    (tmp_path / "result").mkdir()
    for suffix in ("+proj_lr+0.001", "+proj_lr+0.0003"):
        (tmp_path / "result" /
         f"260828+cifar10_setting1_ref_A_v4_P0refit_e4+bs+64+e+5{suffix}").mkdir()
    body = CHAIN.read_text()
    start = body.index("rundir(){")
    end = body.index("\n}", start) + 2
    script = body[start:end] + '\nrundir "cifar10_setting1_ref_A_v4" 4 5\n'
    proc = subprocess.run(["bash", "-c", script], capture_output=True,
                          text=True, cwd=str(tmp_path), timeout=60)
    assert proc.returncode != 0
    assert "refusing to guess" in proc.stderr
    assert proc.stdout.strip() == ""
