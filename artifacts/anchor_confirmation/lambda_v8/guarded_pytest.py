"""Run pytest under an independent file-open boundary guard (audit 734.2 item 3).

A Python audit hook sees every open() of this process (not of child processes). It REFUSES, with
PermissionError, and records any open of a real-data or real-artifact path, so a test that reaches
real data fails instead of passing quietly:
  * anything under /data/yschoi/ except this worktree's source (its artifacts/ subtree is denied);
  * the main tree /home/yschoi/GroundedDNA/, the anchor result, ops and stage-L ledger roots, and the
    Hugging Face cache;
  * any file with a binary payload suffix outside the temporary directory and the interpreter's own
    installation.
Every guarded open is appended to a JSON-lines log, and every refused one to the violation list.
The process exits nonzero if any open was refused, whatever pytest returned.
Usage: guarded_pytest.py <log-dir> <pytest args...>
"""
import json
import os
import sys

WORKTREE = "/data/yschoi/gdna_anchor_lambda_v8/"
LOG_DIR = sys.argv[1]
os.makedirs(LOG_DIR, exist_ok=False)
DENY = ("/data/yschoi/", "/home/yschoi/GroundedDNA/", "/home/yschoi/gdna_anchor4_result",
        "/home/yschoi/gdna_anchor4_ops", "/home/yschoi/gdna_anchorL_ops", "/home/yschoi/.cache/huggingface")
BINARY = (".npz", ".npy", ".pt", ".pth", ".safetensors", ".bin", ".ckpt", ".pkl")
SAFE_BINARY_ROOTS = ("/tmp/", sys.prefix + "/", sys.base_prefix + "/")
violations, opened = [], open(os.path.join(LOG_DIR, "opened.jsonl"), "a", buffering=1)


def verdict(path: str):
    if path.startswith(WORKTREE):
        return "deny" if path.startswith(WORKTREE + "artifacts/") else "allow"
    if path.startswith(DENY):
        return "deny"
    if path.endswith(BINARY) and not path.startswith(SAFE_BINARY_ROOTS):
        return "deny"
    return "allow"


def hook(event, args):
    if event != "open" or not args or isinstance(args[0], int):
        return
    try:
        path = os.path.abspath(os.fsdecode(args[0]))
    except (TypeError, ValueError):
        return
    if path in (opened.name, os.path.abspath(__file__)):
        return
    decision = verdict(path)
    if path.startswith(("/data/", "/home/")) or decision == "deny":
        opened.write(json.dumps({"path": path, "mode": str(args[1]) if len(args) > 1 else None,
                                 "decision": decision}) + "\n")
    if decision == "deny":
        violations.append(path)
        raise PermissionError(f"boundary guard refused a real-data open: {path}")


sys.addaudithook(hook)
import pytest  # noqa: E402

rc = pytest.main(sys.argv[2:])
with open(os.path.join(LOG_DIR, "violations.json"), "w") as out:
    json.dump({"pytest_rc": int(rc), "violations": violations}, out, indent=1)
print(f"[guard] pytest rc {int(rc)}; refused opens {len(violations)}: {sorted(set(violations))[:8]}")
sys.exit(int(rc) if not violations else (int(rc) or 97))
