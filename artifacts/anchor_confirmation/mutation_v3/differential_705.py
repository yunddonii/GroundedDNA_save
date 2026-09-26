"""Differential check (audit 705/706): the new lifecycle tests against the PRE-REPAIR launcher.

A detached worktree at 63654dd (launcher 1badc49) receives only the new test file. Every declared
test runs alone with a timeout; it counts only when its outcome matches the declaration made here,
in advance: the positive controls pass, and each escape case fails AT its declared assertion
(a descendant or an owned process still live), never by a crash, collection error or timeout.
The real anchor worktree's tracked bytes and git status are compared before and after.
"""
import hashlib, json, os, re, shutil, subprocess, sys, time
from pathlib import Path

WT = Path("/data/yschoi/gdna_anchor_confirm_v1")
OUT = Path(sys.argv[1])
OLD_COMMIT = "63654dd58d42187152cb67c22924a3351e470702"
OLD_LAUNCHER = "1badc49c86574137b354f698093713ddcb1cb64211b13e6dd654d0d6fef0673f"
PY = "/home/yschoi/.conda/envs/dna_hashing/bin/python"
F = "tests/test_anchor_confirm_lifecycle.py"
LIVE_CHILD = "assert not live(child)"
OWNED_AT_RELEASE = 'assert lease.at_release[0]["owned_live"] == []'
DECLARED = [
    ("test_cooperative_sleeper_tree_is_drained_by_cleanup", "pass", None),
    ("test_term_resistant_descendant_is_killed_although_its_leader_dies_on_term", "fail", LIVE_CHILD),
    ("test_descendant_of_a_leader_that_exited_first_is_found_by_cleanup", "fail", LIVE_CHILD),
    ("test_a_leftover_descendant_is_stopped_by_its_own_stream_and_the_cell_refused", "fail", LIVE_CHILD),
    ("test_cleanup_blocks_later_launches", "pass", None),
    ("test_a_real_sigterm_releases_the_lease_only_after_every_owned_process_died"
     "[leader:child-cooperative:stay]", "pass", None),
    ("test_a_real_sigterm_releases_the_lease_only_after_every_owned_process_died"
     "[leader:child-resist:stay]", "fail", OWNED_AT_RELEASE),
    ("test_a_leader_that_exited_first_does_not_let_the_lease_go_early", "fail", OWNED_AT_RELEASE),
]
NEW_API_ONLY = ["test_a_session_that_outlives_sigkill_is_reported_within_the_bound_and_kept",
                "test_a_reaped_record_is_never_signalled",
                "test_a_survivor_keeps_the_lease_held_and_the_error_propagates"]


def tracked_state():
    files = subprocess.run(["git", "-C", str(WT), "ls-files", "-z"], capture_output=True,
                           check=True).stdout.split(b"\0")
    digest = hashlib.sha256()
    for name in sorted(f for f in files if f):
        digest.update(name + b"\0" + hashlib.sha256((WT / name.decode()).read_bytes()).digest())
    status = subprocess.run(["git", "-C", str(WT), "status", "--porcelain"], capture_output=True,
                            text=True, check=True).stdout
    return digest.hexdigest(), status


def failing_line(text):
    """The source line pytest marks with '>' in the first failure, if any."""
    for line in text.splitlines():
        if line.startswith(">"):
            return line[1:].split("  #")[0].strip()      # without a trailing comment
    return None


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "tmp").mkdir()                    # pytest creates each --basetemp without parents
    before = tracked_state()
    sandbox = OUT / "old"
    subprocess.run(["git", "-C", str(WT), "worktree", "add", "--detach", str(sandbox), OLD_COMMIT],
                   check=True, capture_output=True)
    report = {"old_commit": OLD_COMMIT, "cases": [], "excluded_new_api": NEW_API_ONLY}
    try:
        launcher = hashlib.sha256((sandbox / "scripts/phase3_selection_matrix.py").read_bytes()).hexdigest()
        assert launcher == OLD_LAUNCHER, launcher
        shutil.copyfile(WT / F, sandbox / F)
        report["test_file_sha256"] = hashlib.sha256((sandbox / F).read_bytes()).hexdigest()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", GDNA_NUM_SEMANTIC_PARTS="5")
        for test, want, marker in DECLARED:
            began = time.monotonic()
            try:
                run = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                                      f"--basetemp={OUT / 'tmp' / str(len(report['cases']))}",
                                      f"{F}::{test}"],
                                     cwd=sandbox, env=env, capture_output=True, text=True, timeout=240)
                rc, text = run.returncode, run.stdout + run.stderr
            except subprocess.TimeoutExpired:
                rc, text = None, "TIMEOUT"
            got = "pass" if rc == 0 else "fail" if rc == 1 else f"other rc={rc}"
            line = failing_line(text)
            intended = (got == want) and (want == "pass" or (marker is not None and line == marker))
            (OUT / f"{len(report['cases']):02d}.log").write_text(text)
            report["cases"].append({"test": test, "declared": want, "marker": marker, "outcome": got,
                                    "failing_line": line, "intended": intended,
                                    "seconds": round(time.monotonic() - began, 1)})
            print(("OK  " if intended else "BAD ") + f"{got:5s} {test} :: {line}")
    finally:
        subprocess.run(["git", "-C", str(WT), "worktree", "remove", "--force", str(sandbox)],
                       check=True, capture_output=True)
    # Fixture processes live under the per-case basetemp; this harness's own argv names only OUT.
    stray = subprocess.run(["pgrep", "-af", str(OUT / "tmp")], capture_output=True,
                           text=True).stdout.strip()
    after = tracked_state()
    report.update({"tracked_before": before[0], "tracked_after": after[0],
                   "status_equal": before[1] == after[1], "stray_processes": stray})
    (OUT / "report.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    ok = all(c["intended"] for c in report["cases"]) and before == after and not stray
    print(f"intended {sum(c['intended'] for c in report['cases'])}/{len(report['cases'])}; "
          f"tree unchanged {before == after}; stray {bool(stray)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
