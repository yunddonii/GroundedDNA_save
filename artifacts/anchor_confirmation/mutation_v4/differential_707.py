"""Differential check (audit 707.1): the new supervisor tests against the v3 supervisor.

A detached worktree at 0ea88af (supervisor f634fc55) receives only the new test file. Every declared
test runs alone with a timeout; it counts only when its outcome matches the declaration made here,
in advance: the positive controls pass, and each escape case fails AT its declared assertion
(a descendant or an owned process still live), never by a crash, collection error or timeout.
The real anchor worktree's tracked bytes and git status are compared before and after.
"""
import hashlib, json, os, re, shutil, subprocess, sys, time
from pathlib import Path

WT = Path("/data/yschoi/gdna_anchor_confirm_v1")
OUT = Path(sys.argv[1])
OLD_COMMIT = "0ea88af47a375a9bf58cade194fe41adca65fc7d"
OLD_LAUNCHER = "f634fc55447c5e946d5ec5bd73006c0f445b0230b5d8302c311a417578b7350c"   # the supervisor
PY = "/home/yschoi/.conda/envs/dna_hashing/bin/python"
F = "tests/test_anchor_confirm_supervisor.py"
DECLARED = [
    ("test_space_refusal_before_the_command_starts", "pass", None),
    ("test_an_attempt_that_lives_between_two_observations_is_still_charged", "fail",
     'assert done["charged_seconds"] >= lived >= 1.5'),
    ("test_the_headroom_covers_the_watchdog_and_the_stop_bound", "fail",
     'assert world.run([PY, "-c", "pass"], budget_seconds=4.0 - 0.01, **common) == S.EXIT_REFUSED'),
]
NEW_API_ONLY = ["test_a_normally_observed_attempt_is_charged_from_its_own_lifetime",
                "test_an_observation_stalled_past_the_watchdog_stops_the_command_and_is_unresolved",
                "test_the_watchdog_signals_before_the_stalled_observation_returns",
                "test_more_attempts_than_planned_cells_void_the_allowance",
                "test_a_failed_observation_stops_dispatch (now unresolved)"]


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
        launcher = hashlib.sha256((sandbox / "scripts/anchor_confirm_supervisor.py").read_bytes()).hexdigest()
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
