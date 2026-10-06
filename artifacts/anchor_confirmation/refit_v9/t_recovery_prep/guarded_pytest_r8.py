"""Run pytest under the r8 allow-list boundary (audits 734.2 item 3, 797.3, 802.2, 804): the policy in
guard_site/gdna_guard_policy.py, enforced by an audit hook in this process and, through
guard_site/sitecustomize.py on PYTHONPATH, in every Python child (a Python child that would start
without it is refused). See that module for the exact read, write and child rules.

Usage (from the tree under test): guarded_pytest_r8.py <log-dir> <private-root> <pytest args...>
Every violation of this process or of any guarded child is counted; the process exits 97 if there is any,
whatever pytest returned. CUDA is hidden and bytecode is never written.
"""
import glob
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
SITE = os.path.join(HERE, "guard_site")
#: the tree under test is the directory this is run from (the r8 worktree, or the battery's sandbox of it)
WORKTREE = os.path.realpath(os.getcwd())
if not os.path.isfile(os.path.join(WORKTREE, "scripts", "anchor_refit_stage.py")):
    raise SystemExit(f"run from the tree under test, not {WORKTREE}")
LOG_DIR, PRIVATE = os.path.realpath(sys.argv[1]), os.path.realpath(sys.argv[2])
os.makedirs(LOG_DIR, exist_ok=False)
os.makedirs(os.path.join(PRIVATE, "tmpdir"), exist_ok=False)
named = sorted({os.path.dirname(os.path.realpath(a.split("::")[0])) for a in sys.argv[3:]
                if a.split("::")[0].endswith(".py")
                and not os.path.realpath(a.split("::")[0]).startswith(WORKTREE + "/")})
os.environ.update(GDNA_GUARD_WORKTREE=WORKTREE, GDNA_GUARD_PRIVATE=PRIVATE, GDNA_GUARD_LOGDIR=LOG_DIR,
                  GDNA_GUARD_SITE=SITE, GDNA_GUARD_RUNNER=os.path.realpath(__file__),
                  GDNA_GUARD_NAMED_DIRS=os.pathsep.join(d + "/" for d in named),
                  TMPDIR=os.path.join(PRIVATE, "tmpdir"), CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
                  PYTHONPATH=SITE)
os.environ.pop("GDNA_ALLOW_REAL_ARTIFACT_TESTS", None)
sys.dont_write_bytecode = True
sys.path.insert(0, SITE)
import gdna_guard_policy as G  # noqa: E402

PINS = {"policy_sha256": hashlib.sha256(open(G.__file__, "rb").read()).hexdigest(),
        "runner_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
        "sitecustomize_sha256": hashlib.sha256(open(os.path.join(SITE, "sitecustomize.py"), "rb").read()).hexdigest()}
G.install()
import pytest  # noqa: E402

rc = pytest.main([f"--basetemp={os.path.join(PRIVATE, 'basetemp')}", *sys.argv[3:]])
rows = []
for path in sorted(glob.glob(os.path.join(LOG_DIR, "violations.*.jsonl"))):
    with open(path) as handle:
        rows += [dict(json.loads(line), pid=path.rsplit(".", 2)[-2]) for line in handle if line.strip()]
summary = {"pytest_rc": int(rc), "violations": rows, "private_root": PRIVATE, **PINS}
with open(os.path.join(LOG_DIR, "summary.json"), "w") as out:
    json.dump(summary, out, indent=1)
print(f"[guard] pytest rc {int(rc)}; violations {len(rows)} (all processes): "
      f"{[r['kind'] + r['what'][:90] for r in rows[:6]]}")
sys.exit(int(rc) if not rows else 97)
