"""Run one preparation command (a Python script) under an allow-list open() guard (audit 734.3:
the exact read footprint of a non-synthetic preparation command, enforced, not only stated).

Allowed: the v8 worktree's source (not its artifacts/ except the files named with --allow), the
temporary directory, the interpreter's installation, /proc, /sys, /dev, /etc, and exactly the files
named with --allow. Refused, with PermissionError, and recorded: every other path under /data/ or
/home/, and any binary payload suffix anywhere outside the interpreter and /tmp. Every open under
/data/ or /home/ is logged with its decision. The audit hook covers this process, not its children
(the pinned wrapper is rendered by a bash child whose argv capture shim runs this same interpreter).
A process image replacement (os.exec*) would drop this hook, so it is refused and recorded; child
processes are recorded with their argv. The launcher re-executes itself at its pre-import boundary
(GDNA_PHASE3_PREIMPORT_SOURCE_BUNDLE); with --launcher-bundle the guard imports the launcher once,
puts the same bundle the re-executed child would receive into the environment, and the script then
verifies it in this process instead of re-executing. (A first attempt without this lost the hook at
that exec: the four 2026-10-01 15:14 plan renders were NOT guarded after the exec.)
Usage: guarded_run.py <log-dir> [--allow PATH]... [--launcher-bundle] -- <script.py> <args...>
"""
import json
import os
import runpy
import sys

WORKTREE = "/data/yschoi/gdna_anchor_lambda_v8/"
argv = sys.argv[1:]
LOG_DIR = argv.pop(0)
allowed, launcher_bundle = set(), False
while argv and argv[0] in ("--allow", "--launcher-bundle"):
    if argv[0] == "--launcher-bundle":
        launcher_bundle, argv = True, argv[1:]
        continue
    allowed.add(os.path.abspath(argv[1]))
    argv = argv[2:]
assert argv and argv[0] == "--", "usage: guarded_run.py <log-dir> [--allow PATH]... -- <script> <args>"
script, script_args = argv[1], argv[2:]
os.makedirs(LOG_DIR, exist_ok=False)
BINARY = (".npz", ".npy", ".pt", ".pth", ".safetensors", ".bin", ".ckpt", ".pkl")
SAFE = ("/tmp/", sys.prefix + "/", sys.base_prefix + "/", "/proc/", "/sys/", "/dev/", "/etc/", "/usr/")
log = open(os.path.join(LOG_DIR, "opened.jsonl"), "a", buffering=1)
children = open(os.path.join(LOG_DIR, "children.jsonl"), "a", buffering=1)
refused = []


def verdict(path):
    if path in allowed:
        return "allow-named"
    if path.endswith(BINARY) and not path.startswith(("/tmp/", sys.prefix + "/", sys.base_prefix + "/")):
        return "deny"
    if path.startswith(WORKTREE):
        return "deny" if path.startswith(WORKTREE + "artifacts/") else "allow-source"
    if path.startswith(SAFE):
        return "allow-system"
    if path.startswith(("/data/", "/home/")):
        return "deny"
    return "allow-system"


def hook(event, args):
    if event == "os.exec":
        refused.append(f"exec:{args[0]}")
        raise PermissionError(f"preparation guard refused a process image replacement: {args[0]}")
    if event == "subprocess.Popen":
        children.write(json.dumps({"executable": os.fsdecode(args[0]) if args[0] else None,
                                   "argv": [os.fsdecode(a) for a in (args[1] or [])],
                                   "cwd": os.fsdecode(args[2]) if args[2] else None}) + "\n")
        return
    if event != "open" or not args or isinstance(args[0], int):
        return
    try:
        path = os.path.abspath(os.fsdecode(args[0]))
    except (TypeError, ValueError):
        return
    if path in (log.name, children.name, os.path.abspath(__file__)):
        return
    decision = verdict(path)
    if path.startswith(("/data/", "/home/")) and not path.startswith((sys.prefix + "/", sys.base_prefix + "/")):
        log.write(json.dumps({"path": path, "decision": decision}) + "\n")
    if decision == "deny":
        refused.append(path)
        raise PermissionError(f"preparation guard refused an unlisted real-data open: {path}")


sys.addaudithook(hook)
if launcher_bundle:
    sys.path.insert(0, WORKTREE.rstrip("/"))
    import importlib
    launcher = importlib.import_module("scripts.phase3_selection_matrix")
    os.environ[launcher._BOOTSTRAP_ENV] = json.dumps(launcher._BOOTSTRAP_PREIMPORT_BUNDLE,
                                                     sort_keys=True, separators=(",", ":"))
sys.argv = [script, *script_args]
code = 0
try:
    runpy.run_path(script, run_name="__main__")
except SystemExit as stop:
    code = stop.code if isinstance(stop.code, int) else (0 if stop.code is None else 1)
finally:
    with open(os.path.join(LOG_DIR, "summary.json"), "w") as out:
        json.dump({"script": script, "args": script_args, "exit": code, "allowed_named": sorted(allowed),
                   "refused": refused}, out, indent=1)
print(f"[prep-guard] exit {code}; refused {len(refused)}: {sorted(set(refused))[:6]}", file=sys.stderr)
sys.exit(code if not refused else (code or 96))
