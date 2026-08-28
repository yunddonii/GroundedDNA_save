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
    """The lines bash executes -- the comments name the old bugs, so asserting
    over the whole file would match its own explanation."""
    return "\n".join(line for line in CHAIN.read_text().splitlines()
                      if not line.strip().startswith("#"))


_BASE = ("--num_semantic_parts 5 --num_codebooks 5 --no_gumbel_softmax "
         "--lambda_codeword_codon_sinkhorn 0.0 "
         "--routing_adaptive_topp_min 0.6 --routing_adaptive_topp_max 0.95")
_CELLS = REPO / "scripts" / "_ablation_cells.py"


def _effective(cell: str, dataset: str = "CIFAR10") -> list:
    """What the child actually receives, composed base and all."""
    proc = subprocess.run(
        [PY, str(_CELLS), "--flags", cell, "--dataset", dataset,
         "--base", _BASE],
        capture_output=True, text=True, cwd=str(REPO), timeout=300)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.split()


def _declared_cells(dataset: str = "CIFAR10") -> list:
    proc = subprocess.run(
        [PY, str(_CELLS), "--list", "A2", "A4", "A5", "--dataset", dataset],
        capture_output=True, text=True, cwd=str(REPO), timeout=300)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.split()


def test_the_a5_factorial_is_four_DISTINCT_commands():
    """The test that would have caught the collapse.

    The shared base carries `--no_gumbel_softmax` unconditionally, so A5_none
    composed to the same command as A5_nogumbel and A5_joint to the same as
    A5_both: four tags, two configurations. Checking that four tag STRINGS
    existed said the factorial was present. This compares what the child gets.
    """
    composed = {c: _effective(c) for c in
                ("A5_none", "A5_joint", "A5_nogumbel", "A5_both")}
    # The axis that collapsed.
    gumbel = {c: ("--no_gumbel_softmax" in f) for c, f in composed.items()}
    assert gumbel == {"A5_none": False, "A5_joint": False,
                      "A5_nogumbel": True, "A5_both": True}
    # And no two cells are the same command.
    keys = [tuple(sorted(f)) for f in composed.values()]
    assert len(set(keys)) == 4, "two A5 cells compose to the same command"


def test_the_base_recipe_cannot_decide_a_cell_owned_axis():
    """A base that already says the flag makes "with" and "without" identical."""
    for cell in ("A5_none", "A5_joint"):
        assert "--no_gumbel_softmax" not in _effective(cell)


def test_a2_actually_disables_the_text_path():
    flags = _effective("A2_no_text")
    assert "--disable_text_supervision" in flags, (
        "zeroing the text losses leaves caption-derived routing active")
    for loss in ("--lambda_text_code_kl", "--lambda_text_hash_ntxent",
                 "--lambda_xmodal_commit"):
        assert flags[flags.index(loss) + 1] == "0.0"


@pytest.mark.parametrize("dataset,k", [
    ("CIFAR10", 64), ("Flickr25k", 128), ("NUSWIDE", 128), ("MSCOCO", 128)])
def test_a4_shares_at_matched_total_capacity(dataset, k):
    """Sharing at the per-slot size would confound sharing with size."""
    flags = _effective("A4_shared_codebook", dataset)
    assert "--share_codebook" in flags
    assert "--num_codebooks" in flags and \
        flags[flags.index("--num_codebooks") + 1] == "5"
    assert flags[flags.index("--codebook_size") + 1] == str(5 * k)


def test_every_cell_carries_its_own_selection_mode():
    """Two cells sharing a mode would compete for one result directory."""
    modes = {}
    for cell in _declared_cells():
        flags = _effective(cell)
        assert "--selection_mode" in flags, cell
        modes[cell] = flags[flags.index("--selection_mode") + 1]
    assert len(set(modes.values())) == len(modes), modes


def test_the_chain_reads_the_declared_spec_rather_than_its_own_strings():
    commands = _commands()
    assert "_ablation_cells.py" in commands
    # The hand-written cell strings are gone, along with the flags that could
    # not run.
    assert '"A5nogumbel:' not in commands
    assert "--router_type mean" not in commands
    assert "--num_codebooks 1" not in commands


def test_the_declared_set_is_a2_a4_and_the_a5_factorial():
    assert _declared_cells() == [
        "A2_no_text", "A4_shared_codebook",
        "A5_none", "A5_joint", "A5_nogumbel", "A5_both"]


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


# --------- the confounds the narrow A5 fix introduced or left behind

def test_a2_and_a4_keep_the_champion_recipe():
    """Stripping a fixed set from the base for every cell was its own confound.

    A2 and A4 lost `--no_gumbel_softmax`, which the champion HAS, so each
    differed from the reference in two ways at once and the delta stopped being
    about the thing being ablated.
    """
    for cell in ("A2_no_text", "A4_shared_codebook"):
        assert "--no_gumbel_softmax" in _effective(cell), cell


def test_only_a5_decides_the_gumbel_axis():
    owned = {c: ("--no_gumbel_softmax" in _effective(c))
             for c in _declared_cells()}
    assert owned["A2_no_text"] is True
    assert owned["A4_shared_codebook"] is True
    assert owned["A5_none"] is False and owned["A5_joint"] is False
    assert owned["A5_nogumbel"] is True and owned["A5_both"] is True


@pytest.mark.parametrize("dataset,k", [("CIFAR10", 64), ("MSCOCO", 128)])
def test_a4_evaluates_at_the_width_it_trained(dataset, k):
    """Training one shared codebook at 5*K and evaluating at K reads a
    codebook of a different width than the one trained."""
    flags = _effective("A4_shared_codebook", dataset)
    trained = flags[flags.index("--codebook_size") + 1]
    assert trained == str(5 * k)
    commands = _commands()
    assert 'CELL_K=$(printf' in commands, "the runner is not told the cell's K"
    assert '${CELL_K:+K="$CELL_K"}' in commands


def test_the_reference_resolver_excludes_the_new_cell_names():
    """`_A2_P0refit` matched none of A2_no_text, A5_none, A5_both ..."""
    import re

    body = CHAIN.read_text()
    line = next(l for l in body.splitlines() if "keep+=(" in l and "=~" in l)
    pattern = re.search(r'=~ (\S+) \]\]', line).group(1)
    regex = re.compile(pattern.replace("_(", "_(").strip())
    for cell in _declared_cells() + ["BU05", "s43", "s44"]:
        name = f"260829+cifar10_setting1_promptAblA_cifar_A_v4_{cell}_P0refit_e4"
        assert regex.search(name), f"{cell} would be taken as the reference run"
    reference = "260829+cifar10_setting1_promptAblA_cifar_A_v4_P0refit_e39"
    assert not regex.search(reference), "the reference itself must not be excluded"


def test_a_failing_cell_listing_stops_the_chain():
    """`done < <(cmd)` swallowed cmd's failure: zero iterations, then carry on."""
    proc = subprocess.run(
        ["bash", "-c",
         'set -Eeuo pipefail\n'
         'mapfile -t CELLS < <(false) || { echo caught; exit 3; }\n'
         '[[ "${#CELLS[@]}" -eq 6 ]] || { echo "wrong count"; exit 4; }\n'],
        capture_output=True, text=True, timeout=60)
    assert proc.returncode == 4, "an empty listing must not pass as success"
    commands = _commands()
    assert "mapfile -t CELLS" in commands
    assert 'done < <("$PY" scripts/_ablation_cells.py' not in commands
    assert commands.count('"${#CELLS[@]}" -eq 6') == 2


def test_the_chain_uses_the_n_phase_three_chose(tmp_path):
    """The legacy table says CIFAR N=4; the train-only selection says 39."""
    import json as _json

    legacy = _json.loads(
        (REPO / "docs" / "newmodel_analysis" / "fixed_N.json").read_text())
    chosen = _json.loads(
        (REPO / "artifacts" / "phase3_selection"
         / "selected_n.json").read_text())["selected"]
    assert legacy["cifar_A_v4"]["N"] != chosen["cifar10"]["selected_N"], (
        "the two sources agree, so this test no longer proves anything")

    commands = _commands()
    assert "artifacts/phase3_selection/selected_n.json" in commands
    assert "d['$1']['N']" not in commands or "sel[" in commands
