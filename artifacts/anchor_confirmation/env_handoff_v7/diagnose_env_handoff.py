"""Audit 722.2 item 2: bounded CPU-only reproduction of the launcher-to-child environment handoff.

Runs the composed driver of tests/test_anchor_confirm_env_handoff.py (real launcher start-up block,
real `import config`, real `build_command`, the pinned wrapper under bash, a capture child checked by
the real `verify_child_environment`) against two trees:
  - the UNREPAIRED launcher: a detached sandbox worktree at the given commit;
  - the REPAIRED tree: this worktree.
For each of the four datasets and two start-up blocks (runtime variables unset; LD_LIBRARY_PATH set
to the failed run's value) it records the start-up mapping, the in-memory values before and after
the config import, the values passed to the child and the values the child observed, and the
child-side verdict. No trainer, data, model, GPU or lease; fixtures in a private temporary tree.
"""
import json, subprocess, sys, tempfile
from pathlib import Path

WT = Path("/data/yschoi/gdna_anchor_confirm_v1")
OLD_COMMIT, OUT = sys.argv[1], Path(sys.argv[2])
sys.path.insert(0, str(WT / "tests"))
import test_anchor_confirm_env_handoff as H           # noqa: E402

cases = {"unset": H.UNSET, "set": {**H.UNSET, "LD_LIBRARY_PATH": H.CUDA_VALUE}}
with tempfile.TemporaryDirectory(prefix="env_handoff_diag_") as scratch:
    old = Path(scratch) / "old_tree"
    subprocess.run(["git", "-C", str(WT), "worktree", "add", "--detach", str(old), OLD_COMMIT],
                   check=True, capture_output=True)
    report = {"old_commit": OLD_COMMIT, "new_tree": str(WT),
              "new_head": subprocess.run(["git", "-C", str(WT), "rev-parse", "HEAD"], capture_output=True,
                                         text=True).stdout.strip(), "runs": []}
    try:
        for tree, repo in (("unrepaired", old), ("repaired", WT)):
            for case, startup in cases.items():
                for dataset in sorted(H.CELLS):
                    tmp = Path(tempfile.mkdtemp(prefix=f"{tree}_{case}_{dataset}_", dir=scratch))
                    got = H.handoff(tmp, startup, dataset=dataset, repo=repo)
                    report["runs"].append({"tree": tree, "case": case, "dataset": dataset, **got})
                    print(f"{tree:10s} {case:5s} {dataset:9s} verdict={got['verdict']!r}", flush=True)
    finally:
        subprocess.run(["git", "-C", str(WT), "worktree", "remove", "--force", str(old)], check=True,
                       capture_output=True)
OUT.write_text(json.dumps(report, indent=1, sort_keys=True))
