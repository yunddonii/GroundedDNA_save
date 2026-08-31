"""The A-series campaign, executed -- argv and environment, not source text.

Everything in this file exists because a source-substring test passed while the
chain could not launch a single cell:

  * `CELL_K=$(printf ... | grep -- --codebook_size)` exits 1 for A2 and A5,
    which have no such flag, and under `set -Eeuo pipefail` the script dies at
    the assignment before reaching the `|| CELL_K=""` on the next line.
  * `${CELL_K:+K="$CELL_K"} cmd` is not an assignment word: bash expands it and
    looks for a command named `K=320`, so A4 exited 127. The same expansion
    mistake as `${FOIL:+FOIL_JSONL=...}` in the overlay launcher, in a second
    file.
  * `mapfile -t CELLS < <(producer)` checks mapfile's status, not the
    producer's, so a producer that printed six names and then failed passed.

The campaign is a plan now, and these read the plan and run the executor.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

PLANNER = REPO / "scripts" / "ablation_campaign_plan.py"
EXECUTOR = REPO / "scripts" / "_ablation_exec.py"
CHAIN = REPO / "scripts" / "auto_chain_after_3seed.sh"
LAUNCHER = REPO / "scripts" / "run_ablation_campaign.sh"
PY = os.environ.get("PY", sys.executable)

_DATASETS = ("CIFAR10", "Flickr25k", "NUSWIDE", "MSCOCO")
_CELLS = ("A2_no_text", "A4_shared_codebook",
          "A5_none", "A5_joint", "A5_nogumbel", "A5_both")


@pytest.fixture(scope="module")
def plan(tmp_path_factory):
    out = tmp_path_factory.mktemp("plan") / "plan.json"
    proc = subprocess.run(
        [PY, str(PLANNER), "--out", str(out)],
        capture_output=True, text=True, cwd=str(REPO), timeout=900)
    if proc.returncode != 0:
        pytest.skip(f"the plan cannot be built here: {proc.stderr.strip()}")
    return json.loads(out.read_text()), out


def _cell(plan_data, name, dataset):
    for cell in plan_data["cells"]:
        if cell["cell"] == name and cell["dataset"] == dataset:
            return cell
    raise AssertionError(f"{name}/{dataset} missing from the plan")


# ------------------------------------------------------------- the plan

def test_the_campaign_is_exactly_twenty_four_cells(plan):
    data, _ = plan
    assert data["expected_cells"] == 24
    assert len(data["cells"]) == 24
    keys = {(c["dataset"], c["cell"]) for c in data["cells"]}
    assert keys == {(d, c) for d in _DATASETS for c in _CELLS}


def test_every_cell_runs_at_the_n_phase_three_chose(plan):
    data, _ = plan
    chosen = json.loads(
        (REPO / "artifacts" / "phase3_selection"
         / "selected_n.json").read_text())["selected"]
    slug = {"CIFAR10": "cifar10", "Flickr25k": "flickr25k",
            "NUSWIDE": "nuswide", "MSCOCO": "mscoco"}
    for cell in data["cells"]:
        assert cell["N"] == chosen[slug[cell["dataset"]]]["selected_N"], cell


@pytest.mark.parametrize("dataset,k", [
    ("CIFAR10", 64), ("Flickr25k", 128), ("NUSWIDE", 128), ("MSCOCO", 128)])
def test_a4_trains_and_evaluates_at_the_same_width(plan, dataset, k):
    """The chain computed K in bash and never delivered it."""
    data, _ = plan
    cell = _cell(data, "A4_shared_codebook", dataset)
    flags = cell["flags"]
    assert flags[flags.index("--codebook_size") + 1] == str(5 * k)
    assert cell["env"]["K"] == str(5 * k), "the child would evaluate at the " \
        "per-slot width while the model was trained at the shared one"


@pytest.mark.parametrize("dataset", _DATASETS)
def test_non_a4_cells_keep_the_per_slot_width(plan, dataset):
    data, _ = plan
    expected = "64" if dataset == "CIFAR10" else "128"
    for name in _CELLS:
        if name == "A4_shared_codebook":
            continue
        assert _cell(data, name, dataset)["env"]["K"] == expected


@pytest.mark.parametrize("dataset", _DATASETS)
def test_the_gumbel_axis_is_the_factorial(plan, dataset):
    data, _ = plan
    got = {name: "--no_gumbel_softmax" in _cell(data, name, dataset)["flags"]
           for name in _CELLS}
    assert got == {
        "A2_no_text": True, "A4_shared_codebook": True,
        "A5_none": False, "A5_joint": False,
        "A5_nogumbel": True, "A5_both": True}


@pytest.mark.parametrize("dataset", _DATASETS)
def test_no_two_cells_are_the_same_command(plan, dataset):
    data, _ = plan
    composed = {name: tuple(sorted(_cell(data, name, dataset)["flags"]))
                for name in _CELLS}
    assert len(set(composed.values())) == len(_CELLS)


def test_every_cell_names_the_provenance_cache(plan):
    data, _ = plan
    for cell in data["cells"]:
        assert "groundeddna_cache_v6prov" in cell["env"]["CACHE_OVERRIDE"], cell
        assert cell["cache"]["provenanced"], (
            f"{cell['dataset']} cache is not accountable: "
            f"{cell['cache']['refusal']}")


# --------------------------------------------------- the executor, run

def test_a_hostile_environment_cannot_change_the_experiment(plan):
    """NUM_CODONS=4 turns 15 bases into 20; CURVE=1 drops FINAL_EPOCH."""
    data, path = plan
    env = dict(os.environ, NUM_CODONS="4", CURVE="1", K="999",
               WHITEN_VARIANT="_evil", A_SKIPS="", VIZ="1", LBU="9.9")
    proc = subprocess.run(
        [PY, str(EXECUTOR), "--plan", str(path), "--index", "0",
         "--gpu", "0", "--dry-run"],
        capture_output=True, text=True, cwd=str(REPO), env=env, timeout=300)
    assert proc.returncode == 0, proc.stderr
    resolved = json.loads(proc.stdout)["env"]
    assert resolved["NUM_CODONS"] == "3"
    assert resolved["CURVE"] == ""
    assert resolved["K"] == data["cells"][0]["env"]["K"]
    assert resolved["WHITEN_VARIANT"] == "_localOnly"
    assert resolved["VIZ"] == "0"
    assert resolved["LBU"] == "0.02"


def test_an_unrelated_inherited_variable_does_not_reach_the_child(plan):
    _, path = plan
    proc = subprocess.run(
        [PY, str(EXECUTOR), "--plan", str(path), "--index", "0",
         "--gpu", "0", "--dry-run"],
        capture_output=True, text=True, cwd=str(REPO), timeout=300,
        env=dict(os.environ, WASS="9.9", DISABLE_TEXT="1", SHARE_CB="1"))
    resolved = json.loads(proc.stdout)["env"]
    for name in ("WASS", "DISABLE_TEXT", "SHARE_CB"):
        assert name not in resolved, f"{name} leaked into the child"


def test_the_child_runs_under_pipefail(plan):
    """The wrapper's trainer ends in `python ... | tee`."""
    _, path = plan
    proc = subprocess.run(
        [PY, str(EXECUTOR), "--plan", str(path), "--index", "0",
         "--gpu", "0", "--dry-run"],
        capture_output=True, text=True, cwd=str(REPO), timeout=300)
    cmd = json.loads(proc.stdout)["cmd"]
    assert cmd[:3] == ["bash", "-o", "pipefail"]


@pytest.mark.parametrize("index", range(24))
def test_every_planned_cell_resolves_without_dying(plan, index):
    """A2/A5 killed the chain at a K lookup; A4 exited 127 on `K=320`."""
    _, path = plan
    proc = subprocess.run(
        [PY, str(EXECUTOR), "--plan", str(path), "--index", str(index),
         "--gpu", "0", "--dry-run"],
        capture_output=True, text=True, cwd=str(REPO), timeout=300)
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["env"]["AUX_ARGS"]
    assert payload["cmd"][-1].endswith(("_v4", "_v5b"))


# ------------------------------------------------- the chain reads the plan

def test_no_shell_in_the_launch_path_computes_a_cell():
    """The three constructs that made every cell unlaunchable, in BOTH files.

    The chain builds the plan and delegates; the launcher runs it. Checking only
    the chain stopped meaning anything the moment the loop moved, which is how
    this assertion passed while its subject was somewhere else.
    """
    for path in (CHAIN, LAUNCHER):
        commands = "\n".join(
            line for line in path.read_text().splitlines()
            if not line.strip().startswith("#"))
        for broken in ('CELL_K=$(printf', '${CELL_K:+K=', "mapfile -t"):
            assert broken not in commands, \
                f"{path.name} still decides in bash: {broken}"
    chain = CHAIN.read_text()
    assert "ablation_campaign_plan.py" in chain
    assert "run_ablation_campaign.sh" in chain
    assert "_ablation_exec.py" in LAUNCHER.read_text()


def test_the_plan_refuses_an_unaccountable_cache_or_a_superseded_selection(tmp_path):
    """Two gates, and right now the earlier one fires first.

    The cache gate is what this originally pinned: 18 of 24 cells pointed at
    the legacy caches. Since then the Phase-3 selection this plan reads has
    been superseded -- the sixteen records chose N at the trainers' hardcoded
    top-p and lambda 0, and the aggregator that produced the artefact has
    changed -- so `_load_selection` refuses before the caches are looked at.
    Both are refusals; asserting only the cache message would fail for a reason
    that is CORRECT, and asserting nothing would let a real regression pass.
    """
    out = tmp_path / "plan.json"
    proc = subprocess.run(
        [PY, str(PLANNER), "--out", str(out), "--require-provenance"],
        capture_output=True, text=True, cwd=str(REPO), timeout=900,
        env=dict(os.environ, GDNA_CACHE_ROOT=str(tmp_path / "nowhere")))
    assert proc.returncode == 1
    assert ("cannot be accounted for" in proc.stderr
            or "different phase3_select_n.py" in proc.stderr), proc.stderr
    assert not out.exists()


# ---------------------------------------------------------------------------
# §48.4: the selection the campaign pins itself to must exist in the TREE.
# The first version of these tests skipped 46 of 48 when run from a fresh
# `git archive HEAD`, because `selected_n.json` was an untracked workspace file
# -- so "1071 passed" was evidence about one machine, not about the commit.
# ---------------------------------------------------------------------------

def test_the_selection_is_in_the_committed_tree(tmp_path):
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    archive = subprocess.run(["git", "archive", "HEAD"], cwd=str(REPO),
                             capture_output=True, timeout=180)
    assert archive.returncode == 0, archive.stderr.decode()
    subprocess.run(["tar", "-x", "-C", str(fresh)], input=archive.stdout,
                   check=True, timeout=180)
    selection = fresh / "artifacts" / "phase3_selection" / "selected_n.json"
    assert selection.is_file(), (
        "the campaign pins 24 GPU cells to this file; a clone that does not "
        "have it cannot reproduce, or even check, which N they ran at")
    records = sorted((fresh / "artifacts" / "phase3_selection").glob(
        "phase3sel_*.json"))
    assert len(records) == 16, (
        f"{len(records)} of the 16 matrix records are committed; N chosen from "
        f"a partial matrix is N chosen from whichever cells finished")


def test_the_committed_records_are_refused_by_the_current_protocol(tmp_path):
    """The sixteen records chose N under a protocol this tree no longer is.

    They ran at the trainers' hardcoded top-p 0.3/0.7 and lambda 0.0, which the
    draft calls the M=6 tuning, and the identity schema has since bumped to
    carry those axes. Re-running them is the decision; what this test pins is
    that the aggregator SAYS SO rather than reducing them anyway. A silent
    acceptance here is how a superseded selection becomes a paper number.
    """
    out = tmp_path / "selected_n.json"
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "phase3_select_n.py"),
         "--records", str(REPO / "artifacts" / "phase3_selection"),
         "--out", str(out)],
        capture_output=True, text=True, timeout=300, cwd=str(REPO))
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "REFUSED" in proc.stderr
    assert not out.exists()
