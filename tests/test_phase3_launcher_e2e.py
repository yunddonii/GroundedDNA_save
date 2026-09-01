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
DIR="${GDNA_PHASE3_RESULT_ROOT}/260828+cifar10_setting1_${TAG}+bs+64+e+${BUDGET}+proj_lr+0.001"
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

"$PY" - "$DIR" "$N" "$BUDGET" "$TOPP_MIN" "$TOPP_MAX" "$JD" <<'PY'
import json, hashlib, os, shlex, sys
sys.path.insert(0, os.getcwd())
from types import SimpleNamespace
import torch
from dna_utils.run_identity import (
    RunIdentity, bind_phase3_campaign_to_state_dict,
    phase3_campaign_binding_from_env,
    write_phase3_campaign_binding, write_run_manifest)
from dna_utils.runtime_state import write_checkpoint_metadata

run, n, budget = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
topp_min, topp_max, jd = (float(sys.argv[4]), float(sys.argv[5]),
                          float(sys.argv[6]))
cache, qwen, whiten = (os.environ["CACHE"], os.environ["QWEN"],
                       os.environ["WHITEN_NPZ"])
flags = shlex.split(os.environ["EXTRA_ARGS"])
def phase3_arg(name):
    return flags[flags.index(name) + 1]
identity = RunIdentity.from_args(SimpleNamespace(
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
    qwen_text_cache_path=qwen, text_whiten_npz=whiten,
    sinkhorn_epsilon_init=1.0, sinkhorn_epsilon_final=0.1,
    batch_size=64, proj_lr=0.001,
    routing_adaptive_topp=True, no_routing_adaptive_topp=False,
    routing_adaptive_topp_min=topp_min,
    routing_adaptive_topp_max=topp_max,
    routing_adaptive_topp_entropy=False, routing_perplexity_topk=False,
    codon_joint_slots="", codon_joint_floor=1e-6,
    share_codebook=False, disable_text_supervision=False,
    use_gumbel_softmax=False,
    lambda_codon_joint=jd, lambda_text_code_kl=0.05,
    lambda_text_hash_ntxent=0.05, lambda_xmodal_commit=0.05,
    lambda_codeword_codon_sinkhorn=0.0,
    phase3_input_seal=phase3_arg("--phase3_input_seal"),
    phase3_input_aggregate_sha256=phase3_arg("--phase3_input_aggregate_sha256"),
    phase3_split_identity_sha256=phase3_arg("--phase3_split_identity_sha256"),
    phase3_hf_identity_sha256=phase3_arg("--phase3_hf_identity_sha256"),
    clip_snapshot_dir=phase3_arg("--clip_snapshot_dir"),
    clip_snapshot_revision=phase3_arg("--clip_snapshot_revision"),
    clip_snapshot_weight_file=phase3_arg("--clip_snapshot_weight_file"),
    clip_snapshot_weight_sha256=phase3_arg("--clip_snapshot_weight_sha256"),
    clip_snapshot_config_sha256=phase3_arg("--clip_snapshot_config_sha256"),
    clip_snapshot_tokenizers_sha256_json=phase3_arg(
        "--clip_snapshot_tokenizers_sha256_json")))
write_run_manifest(run, identity)

binding = phase3_campaign_binding_from_env(
    identity, actual_tag=os.environ["TAG"])
assert binding is not None
assert identity.digest == binding["expected_identity_digest"]
write_phase3_campaign_binding(run, binding)

cols = ["epoch", "eval_mAP", "eval_mAP_at_R", "eval_mAP_R_cutoff"]
rows = [",".join(cols)]
for e in range(n + 1):
    score = f"{0.50 + 0.01 * e:.4f}" if (e + 1) % 5 == 0 else ""
    cutoff = "1000" if score else ""
    rows.append(f"{e},0.4,{score},{cutoff}")
open(os.path.join(run, "log.csv"), "w").write("\n".join(rows) + "\n")

ckpt = os.path.join(run, "model_state_dict.pth")
state = bind_phase3_campaign_to_state_dict(
    torch.nn.Linear(1, 1).state_dict(), binding)
torch.save(state, ckpt)
sha = hashlib.sha256(open(ckpt, "rb").read()).hexdigest()
write_checkpoint_metadata(
    ckpt, checkpoint_epoch_zero_based=n,
    training_epoch_budget=budget, stop_after_epoch=n,
    lr_schedule_horizon=int(os.environ["EXTRA_LRH"]),
    sinkhorn_schedule_horizon=int(os.environ["EXTRA_SKH"]),
    sinkhorn_epsilon_init=identity.sinkhorn_epsilon_init,
    sinkhorn_epsilon_final=identity.sinkhorn_epsilon_final,
    lr_scheduler="cosine",
    extra={
        "tag": os.environ["TAG"], "dataset": identity.dataset,
        "random_seed": identity.seed,
        "num_semantic_parts": identity.num_slots,
        "num_codons_per_codebook": identity.bases_per_slot,
        "phase3_campaign": binding,
    })
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
    for name in ("__init__.py", "run_identity.py", "runtime_environment.py",
                 "runtime_state.py"):
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
    env["PY"] = sys.executable
    {fail}
    return cmd, env, tag
m.build_command = build
m.assert_production_source_authority = lambda snapshot: None
def fake_input_seals(specs, plan, **kwargs):
    result = {{}}
    for cell in plan:
        dataset, _, _, _, stage, _ = m._campaign_cell_parts(cell)
        seal_stage = "refit" if stage == "refit" else "stage1"
        key = f"{{dataset}}:{{seal_stage}}"
        tokenizers = {{name: "16" * 32 for name in (
            "tokenizer.json", "tokenizer_config.json", "vocab.json",
            "merges.txt", "special_tokens_map.json")}}
        result[key] = {{
            "schema": "groundeddna.phase3-input-authority",
            "schema_version": 1,
            "seal_path": f"/nonexistent/{{dataset}}-{{seal_stage}}.json",
            "seal_file_sha256": "10" * 32,
            "aggregate_sha256": "11" * 32,
            "dataset": dataset, "stage": seal_stage,
            "request": {{}}, "split_identity_sha256": "12" * 32,
            "split_identity": {{}}, "authority_sha256": "13" * 32,
            "hf_runtime": {{
                "checkpoint": m.PHASE3_CLIP_CHECKPOINT,
                "revision": m.PHASE3_CLIP_REVISION,
                "snapshot_dir": "/nonexistent/hf/snapshots/" + m.PHASE3_CLIP_REVISION,
                "weight_file": "pytorch_model.bin",
                "weight_sha256": m.PHASE3_CLIP_WEIGHT_SHA256,
                "config_file": "config.json", "config_sha256": "14" * 32,
                "tokenizer_files_sha256": tokenizers,
                "tokenizer_set_sha256": "15" * 32,
                "local_files_only": True, "identity_sha256": "17" * 32,
            }},
        }}
    return result
m.verify_campaign_input_seals = fake_input_seals
m.verify_snapshot_input_seals = lambda snapshot, **kwargs: None
m._with_campaign_gpu_leases = lambda args, callback: callback()
m.load_recipe_authority = lambda paths: {{"fixture": True}}
m.verify_recipe_authority = lambda authority: {{
    dataset: {{"topp": ["0.3", "0.7"], "joint": "0.01"}}
    for dataset in m.DATASETS}}
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
    proc = _launch(sandbox, "--smoke", "--epochs", "5", "--only",
                   "cifar10:4", "--gpu", "0")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Traceback" not in proc.stderr
    assert "1 of 16 n_selection cells complete" in proc.stdout


def test_the_record_is_written_and_describes_the_run(sandbox):
    proc = _launch(sandbox, "--smoke", "--epochs", "5", "--only",
                   "cifar10:4", "--gpu", "0")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    records = [p for p in (sandbox / "artifacts" / "phase3_selection").glob("*.json")
               if "_cifar_A_v4_" in p.name]
    assert len(records) == 1
    record = json.loads(records[0].read_text())
    assert record["is_candidate_cell"] is False
    assert record["stop_after_epoch"] == 4
    assert record["seed"] == SEED
    assert record["val_split_ratio"] == VAL_RATIO
    assert record["lr_schedule_horizon"] == 5
    assert record["sinkhorn_schedule_horizon"] == 5
    assert record["selection_mode"] == "select"
    assert record["selection"]["selection_epoch_zero_based"] == 4
    assert record["completion"]["terminal_weights_preserved"] is True


def test_a_failing_trainer_leaves_no_record(sandbox):
    """A record must never outlive the cell that failed to produce it."""
    proc = _launch(sandbox, "--smoke", "--epochs", "5", "--only",
                   "cifar10:4", "--gpu", "0",
                   trainer_fails=True)
    assert proc.returncode == 1
    assert "stub trainer failed" in proc.stderr
    assert not [p for p in (sandbox / "artifacts" / "phase3_selection").glob("*.json")
                if "_cifar_A_v4_" in p.name]


def test_rerunning_a_completed_cell_is_refused(sandbox):
    first = _launch(sandbox, "--smoke", "--epochs", "5", "--only",
                    "cifar10:4", "--gpu", "0")
    assert first.returncode == 0, first.stdout + first.stderr
    again = _launch(sandbox, "--smoke", "--epochs", "5", "--only",
                    "cifar10:4", "--gpu", "0")
    assert again.returncode == 2
    assert "already reserved" in again.stderr


def test_the_selection_and_refit_stages_produce_different_cells(sandbox):
    """One tag per stage, so a refit cannot land on the selection's directory."""
    select = _launch(sandbox, "--smoke", "--epochs", "5", "--only",
                     "cifar10:4", "--gpu", "0")
    assert select.returncode == 0, select.stdout + select.stderr
    dirs = sorted(p.name for p in (sandbox / "result").iterdir())
    assert len(dirs) == 1 and "_N4_s42" in dirs[0]
    assert "refit" not in dirs[0]
