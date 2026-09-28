"""Anchor confirmation: the launcher-to-wrapper-to-child environment handoff (audit 722/723).

Anchor stage S (run 20260928T052252Z-35772b95) admitted all four seals, then every trainer refused
at its own admission: "Phase-3 child environment differs from its plan: ['library_environment']".
The plan records the launcher's START-UP values of LD_LIBRARY_PATH, HF_HOME, HF_HUB_OFFLINE and
TRANSFORMERS_OFFLINE (`caller_environment`, read from /proc/self/environ); `build_command` copied
them from os.environ, which the anchor admission's `config` import (-> dataloaders -> cv2) rewrites.

The composed test runs, in a fresh process started with a controlled environment:
  1. the launcher's start-up mapping (`caller_environment`) and the real parent fingerprint;
  2. the real `import config` (the real cv2 rewrite of os.environ -- checked, as a positive control);
  3. the real `build_command` for a real anchor cell of each dataset;
  4. the PINNED WRAPPER under bash, whose `$PY` is a capture child that imports cv2 itself and then
     checks its own start-up environment with the real `verify_child_environment`.
Only library_environment, pythonpath and the interpreter are measured live in the child; the GPU,
torch-device and package fields are copied from the expectation (no GPU in these tests). No trainer,
data, model, GPU or lease is used. `DRIVER`/`SHIM` also serve the diagnostic reproduction against
the unrepaired launcher (artifacts/anchor_confirmation/env_handoff_v7/).
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

import scripts.phase3_selection_matrix as M                 # noqa: E402
from dna_utils import runtime_environment as RE             # noqa: E402

PY = os.path.realpath(sys.executable)
CUDA_VALUE = "/usr/local/cuda-12.4/lib64:/usr/local/cuda/extras/CUPTI/:"   # the failed run's value
CELLS = {"cifar10": (("0.3", "0.7"), "0.02"), "flickr25k": (("0.6", "0.95"), "0.02"),
         "nuswide": (("0.4", "0.8"), "0.05"), "mscoco": (("0.6", "0.95"), "0.03")}

SHIM = r'''
import hashlib, json, os, sys
import cv2                                    # the child's own import-time rewrite (must not matter)
repo = os.environ["HANDOFF_REPO"]
sys.path.insert(0, repo)
from dna_utils.runtime_environment import (RUNTIME_ENV_KEYS, EnvironmentAttestationError,
                                           caller_environment, verify_child_environment)
if len(sys.argv) < 2 or os.path.basename(sys.argv[1]) != "train_siglip2.py":
    sys.exit(0)                               # only the trainer invocation is the boundary
expected = json.loads(os.environ["HANDOFF_EXPECTED_CHILD_ENVIRONMENT_JSON"])
actual = dict(expected)                       # GPU/torch/package fields: copied, not measured
exe = os.path.realpath(sys.executable)
actual.update(library_environment=caller_environment(), pythonpath=os.environ.get("PYTHONPATH"),
              python_executable=exe,
              python_executable_sha256=hashlib.sha256(open(exe, "rb").read()).hexdigest())
try:
    verify_child_environment(expected, actual=actual)
    verdict = "admitted"
except EnvironmentAttestationError as error:
    verdict = str(error)
with open(os.environ["HANDOFF_CHILD_OUT"], "w") as out:
    json.dump({"child_observed": actual["library_environment"],
               "child_os_environ_after_its_imports": {k: os.environ.get(k) for k in RUNTIME_ENV_KEYS},
               "verdict": verdict}, out)
'''

DRIVER = r'''
import json, os, subprocess, sys, tempfile
from pathlib import Path
repo, out, dataset, topp, joint, change = (Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3],
                                           tuple(json.loads(sys.argv[4])), sys.argv[5],
                                           json.loads(sys.argv[6]))
sys.path.insert(0, str(repo))
from dna_utils import runtime_environment as RE
startup = RE.caller_environment()
import scripts.phase3_selection_matrix as M
before = {k: os.environ.get(k) for k in RE.RUNTIME_ENV_KEYS}
from config import Config                     # the anchor admission's import: config -> dataloaders -> cv2
after = {k: os.environ.get(k) for k in RE.RUNTIME_ENV_KEYS}
RE._gpu_inventory = lambda errors: [{"index": 0, "uuid": "GPU-handoff-test", "name": "test"}]
parent = RE.capture_parent_environment([0])   # the real parent fingerprint (inventory stubbed)
expected = RE.expected_child_environment(parent, 0)
scratch = Path(tempfile.mkdtemp(prefix="handoff_", dir=os.environ["HANDOFF_TMP"]))
whiten = scratch / "whiten.npz"
whiten.write_bytes(b"")
M._whitening = lambda spec, stage="select": str(whiten)
cache = scratch / "cache"
cache.mkdir()
for name in ("text_tokens.f16.npy", "text_token_mask.bool.npy"):
    (cache / name).write_bytes(b"")
M.DATASETS[dataset]["cache"] = str(cache)
M.DATASETS[dataset]["qwen"] = str(scratch / "qwen.jsonl")
cmd, env, _ = M.build_command(dataset, 4, 0, topp=topp, joint=joint, seed=42, anchor_arm="anchors")
passed = {k: env.get(k) for k in RE.RUNTIME_ENV_KEYS}
for key, value in change.items():            # a genuinely different variable handed to the child
    if value is None:
        env.pop(key, None)
    else:
        env[key] = value
shim = scratch / "capture_child"
shim.write_text(f"#!{os.path.realpath(sys.executable)}\n" + os.environ["HANDOFF_SHIM"])
shim.chmod(0o700)
child_out = scratch / "child.json"
proc = subprocess.run(["bash", str(repo / cmd[1]), *cmd[2:]], cwd=scratch, capture_output=True,
                      text=True, timeout=180,
                      env=dict(env, PY=str(shim), HANDOFF_REPO=str(repo),
                               HANDOFF_CHILD_OUT=str(child_out),
                               HANDOFF_EXPECTED_CHILD_ENVIRONMENT_JSON=json.dumps(expected)))
report = {"startup": startup, "before_config_import": before, "after_config_import": after,
          "plan_expected": expected["library_environment"], "passed_to_child": passed,
          "change": change, "wrapper_rc": proc.returncode, "wrapper_stderr": proc.stderr[-400:]}
report.update(json.loads(child_out.read_text()) if child_out.is_file() else {"verdict": None})
out.write_text(json.dumps(report, indent=1, sort_keys=True))
'''


def handoff(tmp_path, startup_env: dict, *, dataset="flickr25k", change=None, repo=REPO) -> dict:
    """Run the composed boundary in a fresh process whose start-up block is exactly `startup_env`
    on top of a minimal base (no PYTHONPATH, no runtime variables unless given)."""
    base = {k: os.environ[k] for k in ("PATH", "HOME", "USER", "LANG") if k in os.environ}
    env = {**base, "GDNA_NUM_SEMANTIC_PARTS": "5", "CUDA_VISIBLE_DEVICES": "",
           "HANDOFF_TMP": str(tmp_path), "HANDOFF_SHIM": SHIM,
           **{k: v for k, v in startup_env.items() if v is not None}}
    driver = tmp_path / "driver.py"
    driver.write_text(DRIVER)
    out = tmp_path / f"handoff_{dataset}_{len(list(tmp_path.glob('handoff_*')))}.json"
    topp, joint = CELLS[dataset]
    done = subprocess.run([PY, str(driver), str(repo), str(out), dataset, json.dumps(topp), joint,
                           json.dumps(change or {})], env=env, capture_output=True, text=True,
                          timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    return json.loads(out.read_text())


UNSET = {k: None for k in RE.RUNTIME_ENV_KEYS}


# ---- the composed boundary: start-up values reach the child unchanged ----------------------------
@pytest.mark.parametrize("dataset", sorted(CELLS))
@pytest.mark.parametrize("startup", [
    UNSET,                                                           # all four absent at start-up
    {**UNSET, "LD_LIBRARY_PATH": CUDA_VALUE},                         # the failed run's start-up block
], ids=["unset", "set"])
def test_the_child_starts_under_the_attested_startup_values(tmp_path, dataset, startup):
    got = handoff(tmp_path, startup, dataset=dataset)
    assert got["startup"] == startup
    # positive control: the config import really rewrote the launcher's in-memory copy
    assert got["after_config_import"]["LD_LIBRARY_PATH"] != startup["LD_LIBRARY_PATH"]
    assert "cv2" in got["after_config_import"]["LD_LIBRARY_PATH"]
    assert got["plan_expected"] == startup
    assert got["passed_to_child"] == startup                           # absent keys removed
    assert got["child_observed"] == startup and got["verdict"] == "admitted", got
    assert got["wrapper_rc"] == 0


def test_all_four_runtime_variables_survive_the_handoff(tmp_path):
    startup = {"LD_LIBRARY_PATH": CUDA_VALUE, "HF_HOME": str(tmp_path / "hf"),
               "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    got = handoff(tmp_path, startup)
    assert got["passed_to_child"] == got["child_observed"] == got["plan_expected"] == startup
    assert got["verdict"] == "admitted"


def test_the_childs_own_import_rewrite_does_not_matter(tmp_path):
    got = handoff(tmp_path, {**UNSET, "LD_LIBRARY_PATH": CUDA_VALUE})
    assert "cv2" in got["child_os_environ_after_its_imports"]["LD_LIBRARY_PATH"]
    assert got["child_observed"]["LD_LIBRARY_PATH"] == CUDA_VALUE and got["verdict"] == "admitted"


# ---- a genuinely different child variable still refuses ---------------------------------------------
@pytest.mark.parametrize("startup,change", [
    ({**UNSET, "LD_LIBRARY_PATH": CUDA_VALUE}, {"LD_LIBRARY_PATH": "/genuinely/other"}),
    ({**UNSET, "LD_LIBRARY_PATH": CUDA_VALUE}, {"LD_LIBRARY_PATH": None}),      # removed
    (UNSET, {"LD_LIBRARY_PATH": "/added/by/a/caller"}),                          # added
    (UNSET, {"HF_HUB_OFFLINE": "1"}),
    ({**UNSET, "TRANSFORMERS_OFFLINE": "1"}, {"TRANSFORMERS_OFFLINE": "0"}),
], ids=["ld-changed", "ld-removed", "ld-added", "hf-offline-added", "tf-offline-changed"])
def test_a_genuinely_different_child_variable_refuses(tmp_path, startup, change):
    got = handoff(tmp_path, startup, change=change)
    assert got["passed_to_child"] == startup                           # the launcher handed the plan's
    assert got["verdict"] == "Phase-3 child environment differs from its plan: ['library_environment']"


# ---- build_command itself -----------------------------------------------------------------------------
def test_build_command_takes_the_runtime_variables_from_the_startup_block(monkeypatch, tmp_path):
    """In process: whatever os.environ holds now, the child gets the start-up mapping; a variable
    absent at start-up is removed even when an import created it."""
    monkeypatch.setattr(M, "caller_environment",
                        lambda: {"LD_LIBRARY_PATH": "/startup", "HF_HOME": None,
                                 "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": None})
    monkeypatch.setenv("LD_LIBRARY_PATH", "/import/prepended:/startup")
    monkeypatch.setenv("HF_HOME", "/created/by/an/import")
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.setattr(M, "_whitening", lambda spec, stage="select": str(tmp_path / "w.npz"))
    _, env, _ = M.build_command("flickr25k", 4, 0, topp=("0.6", "0.95"), joint="0.02", seed=42,
                                anchor_arm="anchors")
    assert {k: env.get(k) for k in RE.RUNTIME_ENV_KEYS} == {
        "LD_LIBRARY_PATH": "/startup", "HF_HOME": None, "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": None}
    assert "HF_HOME" not in env and "TRANSFORMERS_OFFLINE" not in env


def test_the_plan_attests_the_same_mapping_build_command_uses():
    """One definition on both sides: the parent fingerprint's library_environment and the keys
    build_command overrides are `caller_environment` and `RUNTIME_ENV_KEYS`."""
    assert M.caller_environment is RE.caller_environment and M.RUNTIME_ENV_KEYS is RE.RUNTIME_ENV_KEYS
    assert set(RE.RUNTIME_ENV_KEYS) <= M._ENV_PASSTHROUGH
