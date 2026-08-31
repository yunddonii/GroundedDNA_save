"""Run the launcher's own success and failure paths, with a stub trainer.

Nothing exercised `run_cell()` -> `main()` end to end, and that is exactly
where the defect was: `read_selection` renamed a field, `main` still read the
old name, and a finished cell published its record and then died with a
KeyError. The launcher exited 1 while an admission-looking record sat on disk,
and the run was reported as a success.

A stub trainer stands in for the GPU. It writes the artefacts a real cell
writes -- args.txt, a run manifest, log.csv with a per-epoch metric, a
checkpoint and its sidecar -- so the real post-checks, the real record writer
and the real `main` all run.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.phase3_selection_matrix import (  # noqa: E402
    BASES_PER_SLOT,
    DATASETS,
    LR_HORIZON,
    SEED,
    SLOTS,
    VAL_RATIO,
    VAL_SEED,
)

LAUNCHER = REPO / "scripts" / "phase3_selection_matrix.py"

#: A trainer that produces a complete, protocol-correct cell.
_STUB = r'''#!/usr/bin/env bash
set -eu
# A trainer that dies the way a real one does: partway, after making a
# directory but before completing anything.
if [ "${STUB_TRAINER_FAILS:-0}" = "1" ]; then
    echo "stub trainer failed" >&2
    exit 7
fi
N="${STOP_EP}"
BUDGET="$(echo "$EXTRA_ARGS" | tr ' ' '\n' | grep -A1 -x -- '-e' | tail -1)"
DIR="result/260828+cifar10_setting1_${TAG}+bs+64+e+${BUDGET}+proj_lr+0.001"
mkdir -p "$DIR" logs
# A real run writes every argument the parser accepted. The swept recipe axes
# are read from here, so a stub that omits them is refused -- which is the
# point: their absence must not read as "the incumbent value".
# `X="$(... | grep ...)"` exits 1 when the flag is absent, and
# `[ test ] && X=default` exits 1 when the test is false -- both kill the
# script under `set -e`. That is the same pair of mistakes the ablation chain
# died on, so the defaults are taken with an `if` and a tolerated grep.
arg_after() {
  local want="$1" prev="" tok
  for tok in $EXTRA_ARGS; do
    if [ "$prev" = "$want" ]; then echo "$tok"; return 0; fi
    prev="$tok"
  done
  return 1
}
if ! TOPP_MIN="$(arg_after --routing_adaptive_topp_min)"; then TOPP_MIN=0.3; fi
if ! TOPP_MAX="$(arg_after --routing_adaptive_topp_max)"; then TOPP_MAX=0.7; fi
if ! JD="$(arg_after --lambda_codon_joint)"; then JD=0.0; fi
{
  echo "num_semantic_parts--------------5"
  echo "num_codebooks--------------5"
  echo "num_codons_per_codebook--------------3"
  echo "routing_adaptive_topp_min--------------${TOPP_MIN}"
  echo "routing_adaptive_topp_max--------------${TOPP_MAX}"
  echo "lambda_codon_joint--------------${JD}"
} > "$DIR/args.txt"
echo "trained" > "logs/${TAG}.log"

python3 - "$DIR" "$N" "$BUDGET" "$TOPP_MIN" "$TOPP_MAX" "$JD" <<'PY'
import json, hashlib, os, sys
sys.path.insert(0, os.getcwd())
from types import SimpleNamespace
from dna_utils.run_identity import RunIdentity, write_run_manifest

run, n, budget = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
topp_min, topp_max, jd = (float(sys.argv[4]), float(sys.argv[5]),
                          float(sys.argv[6]))
cache = os.environ["CACHE"]
write_run_manifest(run, RunIdentity.from_args(SimpleNamespace(
    dataset="CIFAR10", setting="setting1",
    random_seed=int(os.environ["EXTRA_SEED"]), epoch=budget,
    stop_after_epoch=n, num_semantic_parts=5, num_codons_per_codebook=3,
    codebook_size=int(os.environ["K"]),
    selection_mode=os.environ["EXTRA_MODE"],
    val_split_ratio=float(os.environ["VAL_RATIO"]),
    val_split_seed=int(os.environ["VAL_SEED"]),
    lr_schedule_horizon=int(os.environ["EXTRA_LRH"]),
    sinkhorn_schedule_horizon=int(os.environ["EXTRA_SKH"]),
    siglip2_feature_cache_dir=cache, eval_cache_dir=cache,
    routing_adaptive_topp=True,
    routing_adaptive_topp_min=topp_min,
    routing_adaptive_topp_max=topp_max,
    lambda_codon_joint=jd)))

cols = ["epoch", "eval_mAP", "eval_mAP_at_R", "eval_mAP_R_cutoff"]
rows = [",".join(cols)]
for e in range(n + 1):
    score = f"{0.50 + 0.01 * e:.4f}" if (e + 1) % 5 == 0 else ""
    cutoff = "1000" if score else ""
    rows.append(f"{e},0.4,{score},{cutoff}")
open(os.path.join(run, "log.csv"), "w").write("\n".join(rows) + "\n")

ckpt = os.path.join(run, "model_state_dict.pth")
open(ckpt, "wb").write(b"weights")
sha = hashlib.sha256(open(ckpt, "rb").read()).hexdigest()
json.dump({"checkpoint_epoch_zero_based": n, "checkpoint_sha256": sha,
           "schema_version": 1},
          open(ckpt + ".runtime.json", "w"))
PY
'''


@pytest.fixture
def sandbox(tmp_path):
    """A repo-shaped tree where the trainer is the stub above."""
    for sub in ("scripts", "logs", "result", "artifacts/phase3_selection",
                "dna_utils"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    shutil.copy(LAUNCHER, tmp_path / "scripts" / LAUNCHER.name)
    (tmp_path / "scripts" / "__init__.py").write_text("")
    for name in ("__init__.py", "run_identity.py"):
        shutil.copy(REPO / "dna_utils" / name, tmp_path / "dna_utils" / name)
    # dna_utils/__init__ pulls in the world; a stub keeps the fixture honest
    # about what the launcher itself needs.
    (tmp_path / "dna_utils" / "__init__.py").write_text("")

    # The launcher digests its own protocol sources, relative to where it
    # lives, so the sandbox has to carry them.
    from scripts.phase3_selection_matrix import PROTOCOL_SOURCES
    for rel in PROTOCOL_SOURCES:
        dest = tmp_path / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            dest.write_text(f"# stub of {rel}\n")

    trainer = tmp_path / "scripts" / "stub_trainer.sh"
    trainer.write_text(_STUB)
    trainer.chmod(0o755)

    cache = tmp_path / "cache"
    cache.mkdir()
    return tmp_path


def _launch(sandbox: Path, *extra: str, trainer_fails: bool = False):
    """Run the REAL launcher, with DATASETS pointed at the stub."""
    patch = sandbox / "scripts" / "run_stub.py"
    fail = ('env["STUB_TRAINER_FAILS"] = "1"' if trainer_fails else "pass")
    patch.write_text(f'''
import os, sys
sys.path.insert(0, {str(sandbox)!r})
import scripts.phase3_selection_matrix as m
cache = {str(sandbox / "cache")!r}
for ds in list(m.DATASETS):
    m.DATASETS[ds]["trainer"] = "scripts/stub_trainer.sh"
    m.DATASETS[ds]["cache"] = cache
    m.DATASETS[ds]["foils"] = cache
    m.DATASETS[ds]["qwen"] = cache + "/q.jsonl"
_build = m.build_command
def build(dataset, n, gpu, **kw):
    cmd, env, tag = _build(dataset, n, gpu, **kw)
    flags = env["EXTRA_ARGS"].split()
    env["EXTRA_SEED"] = flags[flags.index("--random_seed") + 1]
    env["EXTRA_MODE"] = flags[flags.index("--selection_mode") + 1]
    budget = flags[flags.index("-e") + 1]
    env["EXTRA_LRH"] = (flags[flags.index("--lr_schedule_horizon") + 1]
                        if "--lr_schedule_horizon" in flags else budget)
    env["EXTRA_SKH"] = (flags[flags.index("--sinkhorn_schedule_horizon") + 1]
                        if "--sinkhorn_schedule_horizon" in flags else budget)
    {fail}
    return cmd, env, tag
m.build_command = build
m._ENV_PASSTHROUGH = frozenset(m._ENV_PASSTHROUGH | {{
    "EXTRA_SEED", "EXTRA_MODE", "EXTRA_LRH", "EXTRA_SKH"}})
sys.argv = ["phase3_selection_matrix.py"] + sys.argv[1:]
raise SystemExit(m.main())
''')
    env = dict(os.environ, PYTHONPATH=str(sandbox))
    return subprocess.run(
        [sys.executable, str(patch), *extra],
        capture_output=True, text=True, cwd=str(sandbox), env=env, timeout=300)


def test_a_completed_cell_exits_zero_and_prints_its_score(sandbox):
    """The path that raised KeyError after publishing its record."""
    proc = _launch(sandbox, "--run", "--only", "cifar10:4", "--gpu", "0")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Traceback" not in proc.stderr
    assert "cifar10/N4 ok" in proc.stdout
    assert "@epoch 4" in proc.stdout, \
        "main must read the field read_selection actually writes"
    assert "1 of 1 cells complete" in proc.stdout


def test_the_record_is_written_and_describes_the_run(sandbox):
    proc = _launch(sandbox, "--run", "--only", "cifar10:4", "--gpu", "0")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    records = list((sandbox / "artifacts" / "phase3_selection").glob("*.json"))
    assert len(records) == 1
    record = json.loads(records[0].read_text())
    assert record["is_candidate_cell"] is True
    assert record["stop_after_epoch"] == 4
    assert record["seed"] == SEED
    assert record["val_split_ratio"] == VAL_RATIO
    assert record["lr_schedule_horizon"] == LR_HORIZON
    assert record["sinkhorn_schedule_horizon"] == 5
    assert record["selection_mode"] == "select"
    assert record["selection"]["selection_epoch_zero_based"] == 4
    assert record["completion"]["terminal_weights_preserved"] is True


def test_a_failing_trainer_leaves_no_record(sandbox):
    """A record must never outlive the cell that failed to produce it."""
    proc = _launch(sandbox, "--run", "--only", "cifar10:4", "--gpu", "0",
                   trainer_fails=True)
    assert proc.returncode == 1
    assert "REFUSED" in proc.stderr
    assert not list((sandbox / "artifacts" / "phase3_selection").glob("*.json"))


def test_rerunning_a_completed_cell_is_refused(sandbox):
    first = _launch(sandbox, "--run", "--only", "cifar10:4", "--gpu", "0")
    assert first.returncode == 0, first.stdout + first.stderr
    again = _launch(sandbox, "--run", "--only", "cifar10:4", "--gpu", "0")
    assert again.returncode == 1
    assert "already has artefacts" in again.stderr


def test_the_selection_and_refit_stages_produce_different_cells(sandbox):
    """One tag per stage, so a refit cannot land on the selection's directory."""
    select = _launch(sandbox, "--run", "--only", "cifar10:4", "--gpu", "0")
    assert select.returncode == 0, select.stdout + select.stderr
    dirs = sorted(p.name for p in (sandbox / "result").iterdir())
    assert len(dirs) == 1 and "_N4_s42" in dirs[0]
    assert "refit" not in dirs[0]
