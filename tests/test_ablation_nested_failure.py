"""The campaign must not report a cell that died as a cell that finished.

Three shells sit between the chain and the number: the executor, the fixedN
wrapper, and the dataset trainer, whose last statement is `python ... | tee`.
Every one of those boundaries was open at once.

  * the executor started the wrapper with `bash -o pipefail`, but shell options
    are NOT exported, so the trainer's nested `bash "$SCRIPT"` ran without it
    and the pipeline reported tee's status;
  * the wrapper had only `set -u`, so a trainer that exited non-zero left it
    running on to the evaluator and printing DONE;
  * therefore a Python process that died mid-training surfaced at the chain as
    rc 0, and the cell was recorded complete.

These tests build a repo-shaped tree with a stub trainer and run the REAL
wrapper and the REAL executor through it. Nothing here reads source text: the
previous version of this check asserted that the string `bash,-o,pipefail`
appeared in the executor's `--dry-run` output, which was true the entire time
the failure was being swallowed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
WRAPPER = "scripts/prompt_ablation_A_cell_fixedN.sh"
PY = sys.executable

#: The stub stands in for `scripts/train_cifar10_v185_...`, which the wrapper
#: selects for `cifar_A_v4`. Its shape is the one that mattered: the last
#: statement is a pipeline whose left side fails.
TRAINER = """#!/usr/bin/env bash
echo "[stub-trainer] gpu=$1 cache=$CACHE whiten=$WHITEN_NPZ qwen=$QWEN k=$K"
echo "[stub-trainer] codons=$NUM_CODONS final_epoch=${FINAL_EPOCH:-unset} stop=$STOP_EP"
__EXIT_LINE__
"""

EVALUATOR = """import sys
print("[stub-eval]", " ".join(sys.argv[1:]))
"""

PAIRWISE = """import sys
print("[stub-nmi]", " ".join(sys.argv[1:]))
"""

SEALER = """import json, pathlib, sys
cell = pathlib.Path(sys.argv[sys.argv.index("--dir") + 1])
(cell / "analysis_complete.json").write_text(json.dumps({"stub": True}))
print("[stub-seal]", cell)
"""


def _tree(tmp_path: Path, *, trainer_rc: int, make_result: bool = True) -> Path:
    root = tmp_path / "repo"
    (root / "scripts" / "lib").mkdir(parents=True)
    shutil.copy(REPO / WRAPPER, root / WRAPPER)
    shutil.copy(REPO / "scripts" / "lib" / "result_dir.sh",
                root / "scripts" / "lib" / "result_dir.sh")

    if trainer_rc:
        exit_line = f'python3 -c "import sys; sys.exit({trainer_rc})" | tee /dev/null'
    else:
        exit_line = 'python3 -c "print(0)" | tee /dev/null'
    (root / "scripts" / "train_cifar10_v185_bidirTokenPrune05_ccs01_clip.sh"
     ).write_text(TRAINER.replace("__EXIT_LINE__", exit_line))
    (root / "scripts" / "eval_cell_bioproj.py").write_text(EVALUATOR)
    (root / "scripts" / "pairwise_nmi.py").write_text(PAIRWISE)
    (root / "scripts" / "seal_cell_analysis.py").write_text(SEALER)

    whiten = root / "wdir"
    whiten.mkdir()
    for name in ("text_whiten_optTrain_localOnly.npz",
                 "text_whiten_trainOnly_localOnly.npz"):
        (whiten / name).write_bytes(b"npz")
    (root / "cache").mkdir()
    (root / "cache" / "qwen.jsonl").write_text("{}\n")

    if make_result:
        # What the trainer would have produced, so the success path can reach
        # the evaluator without a GPU.
        out = root / "result" / "260829+cifar10_promptAblA_cifar_A_v4_A5_both_P0refit_e4+x"
        out.mkdir(parents=True)
        (out / "extract_db.npz").write_bytes(b"npz")
    return root


def _plan(root: Path) -> Path:
    """A plan shaped exactly like the real one, pointing at the stub tree."""
    sys.path.insert(0, str(REPO))
    from scripts.ablation_campaign_plan import ENV_PASSTHROUGH, _PINNED

    env = {k: "" for k in _PINNED}
    env.update({
        "GDNA_NUM_SEMANTIC_PARTS": "5", "NUM_CODONS": "3", "K": "64",
        "CIBNT": "1.0", "VIZ": "0", "CURVE": "", "WHITEN_VARIANT": "_localOnly",
        "A_SKIPS": "", "LBU": "0.02", "FIXED_N": "4", "EVERY": "1",
        "TAG_SUFFIX": "_A5_both", "AUX_ARGS": "",
        "CACHE_OVERRIDE": str(root / "cache"),
        "WDIR_OVERRIDE": str(root / "wdir"),
        "QWEN_OVERRIDE": str(root / "cache" / "qwen.jsonl"),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    })
    plan = {
        "schema_version": 2,
        "expected_cells": 1,
        "env_passthrough": sorted(ENV_PASSTHROUGH),
        "env_passthrough_values": {"HOME": str(root),
                                   "PY": PY},
        "env_pinned": list(_PINNED),
        "runners": [WRAPPER],
        "cells": [{
            "exp": "cifar_A_v4", "tag": "promptAblA_cifar_A_v4_A5_both",
            "cell": "A5_both", "runner": WRAPPER, "env": env,
        }],
    }
    plan["plan_digest"] = hashlib.sha256(
        json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path = root / "plan.json"
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    return path


def _exec(root: Path, plan: Path):
    sha = hashlib.sha256(plan.read_bytes()).hexdigest()
    return subprocess.run(
        [PY, str(REPO / "scripts" / "_ablation_exec.py"),
         "--plan", str(plan), "--index", "0", "--gpu", "0",
         "--repo", str(root), "--expect-plan-sha256", sha,
         "--test-only-unsealed"],
        capture_output=True, text=True, timeout=180)


@pytest.mark.parametrize("rc", [1, 23])
def test_a_trainer_that_dies_surfaces_as_a_failed_cell(tmp_path, rc):
    """The whole defect, end to end: inner rc -> what the chain collects."""
    root = _tree(tmp_path, trainer_rc=rc)
    proc = _exec(root, _plan(root))
    assert proc.returncode != 0, (
        f"the trainer exited {rc} and the campaign saw success:\n"
        f"{proc.stdout}\n{proc.stderr}")
    assert "DONE" not in proc.stdout, (
        "the wrapper announced the cell as finished after its trainer died")
    assert "[stub-eval]" not in proc.stdout, (
        "the evaluator ran on a run that never trained")


def test_the_pipeline_status_is_not_tees(tmp_path):
    """`python | tee` is the exact shape that returned 0 for a dead Python."""
    root = _tree(tmp_path, trainer_rc=23)
    trainer = root / "scripts" / "train_cifar10_v185_bidirTokenPrune05_ccs01_clip.sh"
    assert "| tee" in trainer.read_text(), "the stub lost the shape under test"
    proc = _exec(root, _plan(root))
    assert proc.returncode != 0, proc.stdout + proc.stderr


def test_the_success_path_still_reaches_the_evaluator(tmp_path):
    """A refusal that refuses everything is not a fix.

    This is the check the earlier launcher work was missing twice: a renamed
    field and a `${VAR:+NAME=...}` non-assignment both survived because no test
    executed the path where nothing is wrong.
    """
    root = _tree(tmp_path, trainer_rc=0)
    proc = _exec(root, _plan(root))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "[stub-eval]" in proc.stdout
    assert "DONE" in proc.stdout


def test_the_plan_supplies_the_cache_the_whitening_and_the_captions(tmp_path):
    """§48.3: the wrapper's WDIR was hardcoded to the legacy root.

    The Flickr and MS-COCO trainOnly matrices differ in bytes between the two
    roots, so a cell handed a v6prov feature cache still trained on a transform
    the plan never named.
    """
    root = _tree(tmp_path, trainer_rc=0)
    proc = _exec(root, _plan(root))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert f"cache={root / 'cache'}" in proc.stdout
    assert f"whiten={root / 'wdir'}/text_whiten_trainOnly_localOnly.npz" \
        in proc.stdout
    assert f"qwen={root / 'cache' / 'qwen.jsonl'}" in proc.stdout
    assert "./cache/" not in proc.stdout


def test_a_missing_result_directory_is_a_failure_not_a_pass(tmp_path):
    root = _tree(tmp_path, trainer_rc=0, make_result=False)
    proc = _exec(root, _plan(root))
    assert proc.returncode != 0
    assert "DONE" not in proc.stdout


def test_the_child_cannot_inherit_an_experiment_changing_variable(tmp_path):
    """NUM_CODONS=4 is 20 bases, not the paper's 15; observed in the child."""
    root = _tree(tmp_path, trainer_rc=0)
    plan = _plan(root)
    sha = hashlib.sha256(plan.read_bytes()).hexdigest()
    hostile = {"PATH": "/usr/bin:/bin", "NUM_CODONS": "4", "CURVE": "1",
               "K": "999", "HOME": str(root)}
    proc = subprocess.run(
        [PY, str(REPO / "scripts" / "_ablation_exec.py"),
             "--plan", str(plan), "--index", "0", "--gpu", "0",
             "--repo", str(root), "--expect-plan-sha256", sha,
             "--test-only-unsealed"],
        capture_output=True, text=True, timeout=180, env=hostile)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "codons=3" in proc.stdout and "k=64" in proc.stdout
    # CURVE=1 would have dropped FINAL_EPOCH and switched the cell to the
    # test-monitored path -- a different experiment, not a different log line.
    assert "final_epoch=1" in proc.stdout
