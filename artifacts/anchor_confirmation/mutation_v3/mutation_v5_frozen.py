"""Mutation battery v5 (audit 705/706 lifecycle, 703/704 supervisor, the pre-dispatch storage rule).

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
T2 = (LC, "test_term_resistant_descendant_is_killed_although_its_leader_dies_on_term")
T3 = (LC, "test_descendant_of_a_leader_that_exited_first_is_found_by_cleanup")
T4 = (LC, "test_a_leftover_descendant_is_stopped_by_its_own_stream_and_the_cell_refused")
T5 = (LC, "test_a_session_that_outlives_sigkill_is_reported_within_the_bound_and_kept")
T6 = (LC, "test_a_reaped_record_is_never_signalled")
T9 = (LC, "test_a_leader_that_exited_first_does_not_let_the_lease_go_early")
T10 = (LC, "test_a_survivor_keeps_the_lease_held_and_the_error_propagates")
L1 = (LA, "test_no_cell_is_dispatched_when_the_result_filesystem_lacks_the_floor")
L4 = (LA, "test_the_first_dispatch_reserves_space_for_every_unfinished_cell")
S2 = (SV, "test_space_refusal_before_the_command_starts")
S5 = (SV, "test_space_breach_while_a_child_lives_stops_through_the_launchers_own_path")
S6 = (SV, "test_budget_breach_while_a_child_lives_stops_it")
S7 = (SV, "test_an_operator_signal_becomes_one_sigterm_to_the_command")
S8 = (SV, "test_a_failed_observation_stops_dispatch")
S9 = (SV, "test_a_command_slower_than_the_stop_bound_is_waited_for_not_declared_dead")
S4 = (SV, "test_failed_attempts_are_charged_and_carried_into_the_next_stage")
S11 = (SV, "test_a_launcher_that_leaves_a_live_attempt_is_unclean_and_blocks_the_ledger")
S12 = (SV, "test_an_unfinished_prior_run_and_a_second_supervisor_are_refused")
S14 = (SV, "test_the_generation_check")

LIVE_CHILD = ("assert not live(child)", None)
OWNED = ('assert lease.at_release[0]["owned_live"] == []', None)
STOPPED = ("assert rc == S.EXIT_STOPPED", None)
MUTANTS = [
    # --- launcher lifecycle (audit 705/706) ---
    ("ML1 forget the record when the leader exits", L,
     "    deadline = time.monotonic() + _DRAIN_SECONDS\n",
     "    with _CHILDREN_LOCK:\n        _ACTIVE_CHILDREN.discard(proc)\n"
     "    deadline = time.monotonic() + _DRAIN_SECONDS\n",
     [(T3, LIVE_CHILD), (T9, OWNED)]),
    ("ML2 drained as soon as the leader exited", L,
     "            if _session_live_members(proc.pid) or _session_live_members(proc.pid):\n",
     "            if False:\n",
     [(T2, LIVE_CHILD), (T3, LIVE_CHILD)]),
    ("ML3 no SIGKILL escalation", L,
     "            _signal_owned_session(proc, signal.SIGKILL)\n", "            pass\n",
     [(T2, ("M._terminate_active_children()", "still have live processes after SIGKILL"))]),
    ("ML4 a reaped record is signalled", L,
     "        if proc.returncode is not None:\n            return\n        try:\n"
     "            # The leader's own group",
     "        if False:\n            return\n        try:\n            # The leader's own group",
     [(T6, ("assert sent == [(proc.pid, signal.SIGKILL)]", None))]),
    ("ML5 survivors are not reported", L,
     "    if survivors:\n        raise RuntimeError(\n            \"campaign child session(s) \"",
     "    if False:\n        raise RuntimeError(\n            \"campaign child session(s) \"",
     [(T5, (None, "DID NOT RAISE"))]),
    ("ML6 the lease is released despite survivors", L,
     "        try:\n            _terminate_active_children()\n            live = _owned_live_sessions()\n",
     "        try:\n            try:\n                _terminate_active_children()\n"
     "            except RuntimeError:\n                pass\n            live = []\n",
     [(T10, (None, "DID NOT RAISE"))]),
    ("ML7 the interpreter-exit release stays registered", L,
     "    atexit.unregister(leases.release)\n", "    pass\n",
     [(T10, ("assert lease.exit_release_pending() == []", None))]),
    ("ML8 a leftover descendant is not stopped by its stream", L,
     "        if time.monotonic() >= deadline:\n            survivors = _stop_owned_sessions((proc,))\n",
     "        if True:\n            break\n        if time.monotonic() >= deadline:\n"
     "            survivors = _stop_owned_sessions((proc,))\n",
     [(T4, LIVE_CHILD)]),
    # --- the pre-dispatch storage rule (audit 703.3) ---
    ("ML9 no refusal before a dispatch", L,
     "                    if refusal:\n                        raise CellRefused(f\"before {key}: {refusal}\")\n",
     "                    if False:\n                        raise CellRefused(f\"before {key}: {refusal}\")\n",
     [(L1, ("assert go() == 1", None))]),
    ("ML10 only this cell is counted", L,
     "                        unfinished = len(plan) - sum(\n"
     "                            1 for status, _ in results.values() if status == \"ok\")\n",
     "                        unfinished = 1\n",
     [(L4, ("assert go() == 1", None))]),
    # --- the supervisor (audit 703/704) ---
    ("MS1 no space refusal before start", SUP,
     "        if free < need:\n            refusal = (f\"space:", "        if False:\n            refusal = (f\"space:",
     [(S2, ("assert rc == S.EXIT_REFUSED and not marker.exists()", None))]),
    ("MS2 no space stop while running", SUP,
     "                elif free < FREE_FLOOR_BYTES + gpus * CELL_OUTPUT_BYTES:\n", "                elif False:\n",
     [(S5, STOPPED)]),
    ("MS3 no budget stop while running", SUP,
     "                elif projected >= budget_seconds:\n", "                elif False:\n",
     [(S6, STOPPED)]),
    ("MS4 earlier stages are not charged", SUP,
     "        prior, _runs = ledger.prior_device_seconds()\n",
     "        prior, _runs = 0.0, ledger.prior_device_seconds()[1]\n",
     [(S4, ('assert start["prior_charged_seconds"] == pytest.approx(charged)', None))]),
    ("MS5 a failed observation does not stop", SUP,
     "                reason = f\"monitor failure: {error!r}\"\n", "                reason = None\n",
     [(S8, STOPPED)]),
    ("MS6 death inferred at the stop bound", SUP,
     "            if stop_sent is not None and overdue_noted is None \\\n",
     "            if stop_sent is not None and now - stop_sent > stop_bound_seconds:\n                break\n"
     "            if stop_sent is not None and overdue_noted is None \\\n",
     [(S9, ("assert rc == S.EXIT_STOPPED and time.monotonic() - began >= 2.0", None))]),
    ("MS7 an unclean prior run is accepted", SUP,
     "        if unclean:\n            raise Refused(", "        if False:\n            raise Refused(",
     [(S11, (None, "DID NOT RAISE"))]),
    ("MS8 orphans are not checked", SUP,
     "    orphans = [a[\"id\"] for a in done if still_live(tuple(a[\"id\"]))]\n", "    orphans = []\n",
     [(S11, ("assert rc == S.EXIT_UNCLEAN", None))]),
    ("MS9 an unfinished prior run is accepted", SUP,
     "        if unfinished:\n            raise Refused(", "        if False:\n            raise Refused(",
     [(S12, (None, "DID NOT RAISE"))]),
    ("MS10 the stop sends no signal", SUP,
     "                proc.send_signal(signal.SIGTERM)\n", "                pass\n",
     [(S7, ('assert final(world.ops)["returncode"] == -signal.SIGTERM', None))]),
    ("MS11 the supervisor's own pin is not checked", SUP,
     "    if files.get(SELF_PATH) != _IMPORTED_SOURCE_SHA256:\n", "    if False:\n",
     [(S14, (None, "DID NOT RAISE"))]),
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
