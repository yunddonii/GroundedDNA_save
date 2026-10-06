"""Run pytest inside an operating-system boundary (generation v9 r8; audits 797.3, 802, 804-808).

The preparation tests run in a bubblewrap sandbox: new user, mount, PID, network, IPC and UTS namespaces,
in which only these exist:
  * /usr (and the merged /bin /lib /lib64 /sbin links), /etc, a fresh /proc, a minimal /dev (no GPU device
    node), a private /dev/shm -- all read-only except /dev/shm;
  * the interpreter's conda environment, read-only;
  * the tree under test, read-only, with its data directories (artifacts/, dataset/, result*/, logs/,
    cache*/) replaced by empty private mounts;
  * a source-only git store for that tree: exactly the commits the tests name (HEAD and the historical
    commits in GIT_COMMITS), their commit and tree objects, and every blob EXCEPT those at data or payload
    paths, so commit and tree identities are unchanged but no payload can be read through git;
  * /tmp = this run's private directory (the only writable place; pytest's basetemp and TMPDIR live there).
Nothing else of the host is visible: no /data or /home path (live data, results, caches, ledgers, claims,
leases, other worktrees, the audit ledger), no network interface, no GPU device, CUDA hidden. This bounds
every process the tests start -- any child, isolated interpreter, shell, re-exec or fixture -- without
per-command exceptions. With --fake-nvidia-smi a reviewed synthetic inventory script is mounted over
/usr/bin/nvidia-smi (the legacy launcher tests query it; no driver exists in the sandbox either way).

Before pytest, a probe in a sandbox of the SAME configuration (a separate launch, not one persistent
namespace) records that the host roots are absent, the tree is not writable, the network is down, there is
no GPU device node, and that the launcher, stage, entry, supervisor and manifest modules import from inside
the tree under test (file and digest: audit 807); the run refuses if any check fails.

Usage: sandboxed_pytest_r8.py <evidence-dir> <tree> [--fake-nvidia-smi] [--copy-in FILE]... -- <pytest args...>
(--copy-in places FILE at /tmp/controls/<name> inside the sandbox, for tests that are not in the tree)
Writes into <evidence-dir> (new): bwrap.json (the exact sandbox argv), git_store.json, probe.json,
pytest.log, result.json (pytest rc, probe, source pins). Exit status = pytest's.
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

PY_ENV = "/home/yschoi/.conda/envs/dna_hashing"
PY = f"{PY_ENV}/bin/python"
GIT_COMMITS = ("3dd1c02", "88c3a25b1b309550eafc276c2ce5be7575507173")   # the v8 tip; the historical base
DATA_DIRS = ("artifacts", "dataset", "result", "result_diagnostic", "logs", "cache")
PAYLOAD = (".npz", ".npy", ".pt", ".pth", ".safetensors", ".bin", ".ckpt", ".pkl", ".h5")
HOST_ROOTS_ABSENT = ("/data/yschoi/dataset", "/data/yschoi/groundeddna_cache_v6prov", "/data/yschoi/gdna_anchor_refit_v9r6",
                     "/home/yschoi/GroundedDNA/docs", "/home/yschoi/gdna_anchorRT_result", "/home/yschoi/gdna_anchorRT_ops",
                     "/home/yschoi/gdna_anchorRTrec_ops", "/home/yschoi/gdna_anchorRT_recovery_claims",
                     "/home/yschoi/.cache", "/dev/nvidia0", "/dev/nvidiactl")
FAKE_NVIDIA_SMI = """#!/bin/sh
# synthetic GPU inventory for the sandboxed legacy launcher tests (no driver exists in the sandbox)
for i in 0 1 2 3 4 5; do
  echo "$i, GPU-00000000-0000-0000-0000-00000000000$i, Synthetic GPU, 00000000:0$i:00.0, 000.00"
done
"""
PROBE = r"""
import json, os, socket, sys
roots, tree = json.loads(sys.argv[1]), sys.argv[2]
out = {"absent": {r: not os.path.lexists(r) for r in roots}}
try:
    open(os.path.join(tree, "probe-write-attempt"), "w")
    out["tree_writable"] = True
except OSError as error:
    out["tree_writable"] = False
    out["tree_write_error"] = type(error).__name__
try:
    socket.create_connection(("1.1.1.1", 53), timeout=2)
    out["network"] = True
except OSError as error:
    out["network"] = False
    out["network_error"] = type(error).__name__
out["nvidia_devices"] = sorted(d for d in os.listdir("/dev") if d.startswith("nvidia"))
out["visible_data_entries"] = sorted(os.listdir("/data/yschoi")) if os.path.isdir("/data/yschoi") else []
out["visible_home_entries"] = sorted(os.listdir("/home/yschoi")) if os.path.isdir("/home/yschoi") else []
out["tmp_is_private"] = os.path.realpath("/tmp")
# the modules the tests import resolve inside the tree under test, at these bytes (audit 807)
import hashlib, importlib
sys.path.insert(0, tree)
identities = {}
for name in ("scripts.phase3_selection_matrix", "scripts.anchor_refit_stage", "scripts.anchor_terminal_test",
             "scripts.anchor_confirm_supervisor", "scripts.anchor_confirm_manifest"):
    module = importlib.import_module(name)
    path = os.path.realpath(module.__file__)
    identities[name] = {"file": path, "inside_tree": path.startswith(os.path.realpath(tree) + "/"),
                        "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()}
out["module_identities"] = identities
print("PROBE " + json.dumps(out, sort_keys=True))
"""


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, **kw)


def git(tree, *args, **kw):
    return run(["git", "-C", str(tree), *args], **kw).stdout


def build_git_store(tree: Path, out: Path) -> dict:
    """A source-only object store: the named commits with all their trees and non-payload blobs."""
    gitdir = Path((tree / ".git").read_text().split("gitdir:", 1)[1].strip())
    head = git(tree, "rev-parse", "HEAD", text=True).strip()
    commits = [head] + [git(tree, "rev-parse", f"{c}^{{commit}}", text=True).strip() for c in GIT_COMMITS]
    listing = git(tree, "rev-list", "--objects", "--no-walk", *commits, text=True).splitlines()
    excluded_paths, by_sha = {}, {}
    for line in listing:
        sha, _, path = line.partition(" ")
        by_sha.setdefault(sha, set()).add(path)
    keep, excluded = [], []
    for sha, paths in by_sha.items():
        bad = [p for p in paths if p and (p.split("/")[0] in DATA_DIRS or p.split("/")[0].startswith(("result", "cache"))
                                         or p.endswith(PAYLOAD))]
        if bad and all(p in bad for p in paths if p):
            # a blob or tree only at data/payload paths: blobs are dropped; trees are kept (identities)
            kind = git(tree, "cat-file", "-t", sha, text=True).strip()
            if kind == "blob":
                excluded.append(sha)
                excluded_paths[sha] = sorted(paths)
                continue
        keep.append(sha)
    store = out / "store.git"
    run(["git", "init", "--bare", "-q", str(store)])
    pack = run(["git", "-C", str(tree), "pack-objects", "--stdout"], input="\n".join(keep).encode()).stdout
    run(["git", "-C", str(store), "unpack-objects", "-q"], input=pack)
    branch = (gitdir / "HEAD").read_text().strip()
    if branch.startswith("ref: "):
        ref = store / branch[5:]
        ref.parent.mkdir(parents=True, exist_ok=True)
        ref.write_text(head + "\n")
    replica = out / "worktree_gitdir"
    replica.mkdir()
    for name in ("HEAD", "gitdir", "index"):
        (replica / name).write_bytes((gitdir / name).read_bytes())
    (replica / "commondir").write_text(str(store) + "\n")
    return {"gitdir": str(gitdir), "replica": str(replica), "store": str(store), "head": head,
            "commits": commits, "objects_kept": len(keep), "blobs_excluded": len(excluded),
            "excluded_paths": excluded_paths}


def bwrap_argv(tree: Path, priv_tmp: Path, git_info: dict, fake_smi) -> list:
    argv = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session",
            "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin", "--symlink", "usr/lib", "/lib",
            "--symlink", "usr/lib64", "/lib64", "--symlink", "usr/sbin", "/sbin",
            "--ro-bind", "/etc", "/etc", "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/dev/shm",
            "--ro-bind", PY_ENV, PY_ENV,
            "--bind", str(priv_tmp), "/tmp",
            "--ro-bind", str(tree), str(tree)]
    for name in sorted(os.listdir(tree)):
        if name in DATA_DIRS or name.startswith(("result", "cache")):
            if (tree / name).is_dir():
                argv += ["--tmpfs", str(tree / name)]
    argv += ["--ro-bind", git_info["replica"], git_info["gitdir"],
             "--ro-bind", git_info["store"], git_info["store"]]
    if fake_smi:
        argv += ["--ro-bind", str(fake_smi), "/usr/bin/nvidia-smi"]
    argv += ["--chdir", str(tree), "--clearenv",
             "--setenv", "PATH", f"{PY_ENV}/bin:/usr/bin:/bin", "--setenv", "HOME", "/tmp/home",
             "--setenv", "TMPDIR", "/tmp", "--setenv", "CUDA_VISIBLE_DEVICES", "",
             "--setenv", "PYTHONDONTWRITEBYTECODE", "1", "--setenv", "GDNA_NUM_SEMANTIC_PARTS", "5",
             "--setenv", "LANG", "C.UTF-8", "--setenv", "GIT_CONFIG_NOSYSTEM", "1"]
    return argv


def main() -> int:
    args = sys.argv[1:]
    sep = args.index("--")
    head, pytest_args = args[:sep], args[sep + 1:]
    evidence, tree = Path(head[0]).resolve(), Path(head[1]).resolve()
    fake = "--fake-nvidia-smi" in head[2:]
    copy_in = [Path(head[i + 1]) for i, a in enumerate(head) if a == "--copy-in"]
    evidence.mkdir(parents=True, exist_ok=False)
    priv_tmp = evidence / "tmp"
    (priv_tmp / "home").mkdir(parents=True)
    for source in copy_in:                     # e.g. the boundary's own control tests (seen at /tmp/controls/)
        (priv_tmp / "controls").mkdir(exist_ok=True)
        (priv_tmp / "controls" / source.name).write_bytes(source.read_bytes())
    git_info = build_git_store(tree, evidence)
    (evidence / "git_store.json").write_text(json.dumps(git_info, indent=1, sort_keys=True))
    fake_smi = None
    if fake:
        fake_smi = evidence / "fake-nvidia-smi"
        fake_smi.write_text(FAKE_NVIDIA_SMI)
        fake_smi.chmod(0o555)
    base = bwrap_argv(tree, priv_tmp, git_info, fake_smi)
    (evidence / "bwrap.json").write_text(json.dumps(base, indent=1))
    probe = subprocess.run(base + [PY, "-c", PROBE, json.dumps(HOST_ROOTS_ABSENT), str(tree)],
                           capture_output=True, text=True)
    line = [ln for ln in probe.stdout.splitlines() if ln.startswith("PROBE ")]
    probe_out = json.loads(line[-1][6:]) if probe.returncode == 0 and line else \
        {"error": probe.stderr[-2000:], "rc": probe.returncode}
    (evidence / "probe.json").write_text(json.dumps(probe_out, indent=1, sort_keys=True))
    ok = (isinstance(probe_out.get("absent"), dict) and all(probe_out["absent"].values())
          and probe_out.get("tree_writable") is False and probe_out.get("network") is False
          and probe_out.get("nvidia_devices") == []
          and all(v.get("inside_tree") for v in (probe_out.get("module_identities") or {"x": {}}).values()))
    pins = {"runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "tree_head": git_info["head"],
            "tree_status_clean": git(tree, "status", "--porcelain", text=True) == ""}
    if not ok:
        (evidence / "result.json").write_text(json.dumps({"refused": "sandbox probe failed", "probe": probe_out,
                                                          **pins}, indent=1))
        print("[sandbox] REFUSED: the probe did not confirm the boundary", file=sys.stderr)
        return 3
    with open(evidence / "pytest.log", "w") as log:
        done = subprocess.run(base + [PY, "-m", "pytest", "-p", "no:cacheprovider", "--basetemp=/tmp/basetemp",
                                      *pytest_args], stdout=log, stderr=subprocess.STDOUT)
    tail = (evidence / "pytest.log").read_text(errors="replace").strip().splitlines()[-1:]
    (evidence / "result.json").write_text(json.dumps({"pytest_rc": done.returncode, "summary": tail, "probe": probe_out,
                                                      **pins}, indent=1, sort_keys=True))
    print(f"[sandbox] pytest rc {done.returncode}: {tail}")
    return done.returncode


if __name__ == "__main__":
    raise SystemExit(main())
