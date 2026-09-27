"""Mutation battery v6 (audit 707.1: the supervisor's observation windows, watchdog and unresolved state).

Sandbox: a detached worktree at the source commit given on the command line, its tracked bytes
checked equal to the anchor worktree's. Every declared test first passes unmutated. Each mutant
is ONE exact replacement (count must be 1), disables a whole condition or one call, and runs only
its declared tests, each alone with its own basetemp and a timeout. A mutant counts as detected
only when every declared test fails (pytest rc 1) with the marker declared here in advance: the
'>' line pytest marks (trailing comment removed) and, where given, a required text. Errors,
other exit codes and timeouts are not detections. The real worktree's tracked bytes and git
status are compared before and after; fixture processes live under the per-run basetemp and none
may remain.
"""
import hashlib, json, os, subprocess, sys, time
from pathlib import Path

WT = Path("/data/yschoi/gdna_anchor_confirm_v1")
COMMIT, OUT = sys.argv[1], Path(sys.argv[2])
PY = "/home/yschoi/.conda/envs/dna_hashing/bin/python"
L, SUP = "scripts/phase3_selection_matrix.py", "scripts/anchor_confirm_supervisor.py"
LC, SV, LA = ("tests/test_anchor_confirm_lifecycle.py", "tests/test_anchor_confirm_supervisor.py",
              "tests/test_anchor_confirm_launcher.py")
D1A = (SV, "test_an_attempt_that_lives_between_two_observations_is_still_charged")
D1B = (SV, "test_an_observation_stalled_past_the_watchdog_stops_the_command_and_is_unresolved")
D2 = (SV, "test_the_watchdog_signals_before_the_stalled_observation_returns")
D3 = (SV, "test_more_attempts_than_planned_cells_void_the_allowance")
D5 = (SV, "test_the_headroom_covers_the_watchdog_and_the_stop_bound")
LW = (SV, "test_a_long_window_counts_against_the_budget_while_the_command_runs")
FO = (SV, "test_a_failed_observation_stops_dispatch")
CHARGED = ('assert done["charged_seconds"] >= lived >= 1.5', None)
MUTANTS = [
    ("MV1 the final allowance charges the nominal poll", SUP,
     '    if attempts == "sessions":\n        allowance = planned_cells * max_window\n    if watch["fired"]',
     '    if attempts == "sessions":\n        allowance = planned_cells * poll_seconds\n    if watch["fired"]',
     [(D1A, CHARGED)]),
    ("MV2 the last window before the exit is not measured", SUP,
     "    window = (ended if exit_seen is None else exit_seen) - window_start\n"
     "    max_window = max(max_window, window)\n",
     "    window = 0.0\n",
     [(D1A, CHARGED)]),
    ("MV3 the live allowance charges the nominal poll", SUP,
     "                    allowance = planned_cells * max_window\n                    if len(done)",
     "                    allowance = planned_cells * poll_seconds\n                    if len(done)",
     [(LW, ("assert rc == S.EXIT_STOPPED", None))]),
    ("MV4 the watchdog never fires", SUP,
     '            if quiet > watchdog_seconds and watch["fired"] is None:\n',
     "            if False:\n",
     [(D2, ("assert blocked[2] + 0.9 <= signalled < blocked[3]", None))]),
    ("MV5 the watchdog's stop sends no signal", SUP,
     "                        proc.send_signal(signal.SIGTERM)\n"
     '                        watch["stop_sent"] = boot_seconds()\n',
     "                        pass\n"
     '                        watch["stop_sent"] = boot_seconds()\n',
     [(D2, ("signalled = float(stamp.read_text())", None))]),
    ("MV6 more attempts than cells are not noticed", SUP,
     "                    if len(done) + len(live) > planned_cells and not any(\n"
     '                            c.startswith("attempts:") for c in continuity):\n',
     "                    if False:\n",
     [(D3, ('assert rc == S.EXIT_UNRESOLVED and done["status"] == "unresolved"', None))]),
    ("MV7 an unresolved run is accepted as prior charge", SUP,
     "        if unresolved:\n", "        if False:\n",
     [(D1B, (None, "DID NOT RAISE"))]),
    ("MV8 a failed observation keeps the run settled", SUP,
     '                continuity.append(f"observation failed: {error!r}")\n',
     "                pass\n",
     [(FO, ("assert rc == S.EXIT_UNRESOLVED", None))]),
    ("MV9 the headroom leaves out the watchdog", SUP,
     "        headroom = gpus * (watchdog_seconds + stop_bound_seconds)\n",
     "        headroom = gpus * (poll_seconds + stop_bound_seconds)\n",
     [(D5, ('assert world.run([PY, "-c", "pass"], budget_seconds=4.0 - 0.01, **common) == S.EXIT_REFUSED',
            None))]),
]


def tracked_state():
    files = subprocess.run(["git", "-C", str(WT), "ls-files", "-z"], capture_output=True,
                           check=True).stdout.split(b"\0")
    digest = hashlib.sha256()
    for name in sorted(f for f in files if f):
        digest.update(name + b"\0" + hashlib.sha256((WT / name.decode()).read_bytes()).digest())
    status = subprocess.run(["git", "-C", str(WT), "status", "--porcelain"], capture_output=True,
                            text=True, check=True).stdout
    return digest.hexdigest(), status


def sandbox_state(box):
    files = subprocess.run(["git", "-C", str(box), "ls-files", "-z"], capture_output=True,
                           check=True).stdout.split(b"\0")
    digest = hashlib.sha256()
    for name in sorted(f for f in files if f):
        digest.update(name + b"\0" + hashlib.sha256((box / name.decode()).read_bytes()).digest())
    return digest.hexdigest()


def failing_line(text):
    for line in text.splitlines():
        if line.startswith(">"):
            return line[1:].split("  #")[0].strip()
    return None


RUN = [0]


def run_test(box, test, timeout=420):
    RUN[0] += 1
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", GDNA_NUM_SEMANTIC_PARTS="5")
    began = time.monotonic()
    try:
        done = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                               f"--basetemp={OUT / 'tmp' / str(RUN[0])}", f"{test[0]}::{test[1]}"],
                              cwd=box, env=env, capture_output=True, text=True, timeout=timeout)
        return done.returncode, done.stdout + done.stderr, time.monotonic() - began
    except subprocess.TimeoutExpired as error:
        return None, f"TIMEOUT\n{error.stdout or ''}", time.monotonic() - began


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "logs").mkdir()
    (OUT / "tmp").mkdir()                    # pytest creates each --basetemp without parents
    before = tracked_state()
    box = OUT / "sandbox"
    subprocess.run(["git", "-C", str(WT), "worktree", "add", "--detach", str(box), COMMIT],
                   check=True, capture_output=True)
    report = {"commit": COMMIT, "baseline": [], "mutants": []}
    try:
        assert sandbox_state(box) == before[0], "the sandbox is not the worktree's tracked bytes"
        tests = sorted({t for *_x, declared in MUTANTS for t, _m in declared})
        for test in tests:
            rc, text, seconds = run_test(box, test)
            report["baseline"].append({"test": test[1], "rc": rc, "seconds": round(seconds, 1)})
            print(f"baseline {'ok ' if rc == 0 else 'BAD'} {test[1]} ({seconds:.0f}s)", flush=True)
            if rc != 0:
                (OUT / "logs" / f"baseline_{test[1]}.log").write_text(text)
        if any(row["rc"] != 0 for row in report["baseline"]):
            raise SystemExit("an unmutated declared test failed; no mutant is interpretable")
        for number, (name, rel, old, new, declared) in enumerate(MUTANTS):
            path = box / rel
            original = path.read_bytes()
            text = original.decode()
            count = text.count(old)
            row = {"mutant": name, "file": rel, "count": count, "tests": []}
            if count != 1:
                row["result"] = f"NOT APPLIED (pattern count {count})"
            else:
                path.write_text(text.replace(old, new))
                try:
                    for test, (line_marker, text_marker) in declared:
                        rc, output, seconds = run_test(box, test)
                        line = failing_line(output)
                        intended = (rc == 1
                                    and (line_marker is None or line == line_marker)
                                    and (text_marker is None or text_marker in output))
                        (OUT / "logs" / f"{number:02d}_{test[1]}.log").write_text(output)
                        row["tests"].append({"test": test[1], "rc": rc, "failing_line": line,
                                             "declared_line": line_marker, "declared_text": text_marker,
                                             "intended": intended, "seconds": round(seconds, 1)})
                finally:
                    path.write_bytes(original)
                assert hashlib.sha256(path.read_bytes()).digest() == hashlib.sha256(original).digest()
                row["result"] = ("detected" if row["tests"] and all(t["intended"] for t in row["tests"])
                                 else "NOT DETECTED AS DECLARED")
            report["mutants"].append(row)
            print(f"{row['result']:26s} {name}", flush=True)
        report["sandbox_restored"] = sandbox_state(box) == before[0]
    finally:
        subprocess.run(["git", "-C", str(WT), "worktree", "remove", "--force", str(box)],
                       check=True, capture_output=True)
        stray = subprocess.run(["pgrep", "-af", str(OUT / "tmp")], capture_output=True,
                               text=True).stdout.strip()
        after = tracked_state()
        report.update({"tracked_before": before[0], "tracked_after": after[0],
                       "status_equal": before[1] == after[1], "stray_processes": stray})
        (OUT / "report.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    detected = sum(1 for m in report["mutants"] if m["result"] == "detected")
    print(f"detected {detected}/{len(MUTANTS)}; tree unchanged {before == after}; stray {bool(stray)}")
    return 0 if detected == len(MUTANTS) and before == after and not stray else 1


if __name__ == "__main__":
    raise SystemExit(main())
