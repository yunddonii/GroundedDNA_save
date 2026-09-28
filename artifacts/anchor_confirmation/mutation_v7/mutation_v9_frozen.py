"""Mutation battery v9 (audit 722/723: the child environment handoff and the carried admission,
generation v7).

Same method as v8 (artifacts/anchor_confirmation/mutation_v6/mutation_v8_frozen.py): a detached
sandbox at the given commit with the worktree's tracked bytes; every declared test passes unmutated
first; each mutant is ONE exact replacement that disables one check or loop; it runs only its declared
tests, each alone; it counts only on pytest rc 1 with the marker declared here in advance (the '>'
line, trailing comment removed, and where given a required text). EX9 mutates the TRAINER-side
comparator, which this generation does not change: it proves the composed test's refusal cases are
live. The worktree's tracked bytes and status are compared before and after; no fixture process
may remain.
"""
import hashlib, json, os, subprocess, sys, time
from pathlib import Path

WT = Path("/data/yschoi/gdna_anchor_confirm_v1")
COMMIT, OUT = sys.argv[1], Path(sys.argv[2])
PY = "/home/yschoi/.conda/envs/dna_hashing/bin/python"
L, RE = "scripts/phase3_selection_matrix.py", "dna_utils/runtime_environment.py"
TL, TE = "tests/test_anchor_confirm_launcher.py", "tests/test_anchor_confirm_env_handoff.py"
PASSED = ('assert got["passed_to_child"] == startup', None)
UNIT = ('assert {k: env.get(k) for k in RE.RUNTIME_ENV_KEYS} == {', None)

MUTANTS = [
    ("EX1 build_command copies the runtime variables from os.environ again", L,
     "    for key in RUNTIME_ENV_KEYS:\n        if startup.get(key) is None:\n",
     "    for key in ():\n        if startup.get(key) is None:\n",
     [((TE, "test_the_child_starts_under_the_attested_startup_values[unset-flickr25k]"), PASSED),
      ((TE, "test_the_child_starts_under_the_attested_startup_values[set-flickr25k]"), PASSED),
      ((TE, "test_build_command_takes_the_runtime_variables_from_the_startup_block"), UNIT)]),
    ("EX2 a variable absent at start-up is left behind", L,
     "        if startup.get(key) is None:\n            env.pop(key, None)\n",
     "        if startup.get(key) is None:\n            pass\n",
     [((TE, "test_the_child_starts_under_the_attested_startup_values[unset-cifar10]"), PASSED),
      ((TE, "test_build_command_takes_the_runtime_variables_from_the_startup_block"), UNIT)]),
    ("EX3 the carried-admission guard is not called", L,
     "                if refusal is not None:\n                    raise CellRefused(refusal)\n",
     "                if False:\n                    raise CellRefused(refusal)\n",
     [((TL, "test_the_anchor_admission_carries_only_a_full_historical_admission[False]"),
       ('assert rc == 2 and "not exactly those its snapshot admitted" in capsys.readouterr().err', None))]),
    ("EX4 the carried seal set is not checked against the evidence (reason change)", L,
     "    if not isinstance(evidence, dict) or not isinstance(carried, dict) or set(evidence) != set(carried):\n",
     "    if False:\n",
     [((TL, "test_a_snapshot_without_historical_evidence_cannot_be_carried"),
       ('assert refusal is not None and "are not exactly those" in refusal', None))]),
    ("EX5 the per-seal evidence is not compared", L,
     "        if got != want:\n", "        if False:\n",
     [((TL, "test_a_carried_authority_that_is_not_a_full_historical_admission_refuses[edit0-returncode]"),
       ("assert refusal is not None and reason in refusal", None)),
      ((TL, "test_a_carried_authority_that_is_not_a_full_historical_admission_refuses[edit3-seal]"),
       ("assert refusal is not None and reason in refusal", None))]),
    ("EX6 the carried snapshot is not read at its approved bytes", L,
     "    if hashlib.sha256(raw).hexdigest() != expected_sha256:\n", "    if False:\n",
     [((TL, "test_the_carried_snapshot_is_read_at_its_approved_bytes"),
       ('assert refusal is not None and "changed after its approval pin" in refusal', None))]),
    ("EX7 the full-admission provenance of a carried authority is ignored at the call site", L,
     "                refusal = anchor_carried_admission_refusal(args.admission_authority, carried,\n"
     "                                                           expected_sha256=before[\"sha256\"])\n",
     "                refusal = None\n",
     [((TL, "test_the_anchor_admission_carries_only_a_full_historical_admission[False]"),
       ('assert rc == 2 and "not exactly those its snapshot admitted" in capsys.readouterr().err', None))]),
    ("EX8 a caller's start-up LD_LIBRARY_PATH is dropped", L,
     "        else:\n            env[key] = startup[key]\n",
     "        else:\n            env.pop(key, None)\n",
     [((TE, "test_the_child_starts_under_the_attested_startup_values[set-mscoco]"), PASSED),
      ((TE, "test_all_four_runtime_variables_survive_the_handoff"),
       ('assert got["passed_to_child"] == got["child_observed"] == got["plan_expected"] == startup', None))]),
    ("EX9 (control) the trainer-side comparator is disabled", RE,
     "    if observed != expected_copy:\n", "    if False:\n",
     [((TE, "test_a_genuinely_different_child_variable_refuses[ld-changed]"),
       ('assert got["verdict"] == "Phase-3 child environment differs from its plan: [\'library_environment\']"', None)),
      ((TE, "test_a_genuinely_different_child_variable_refuses[hf-offline-added]"),
       ('assert got["verdict"] == "Phase-3 child environment differs from its plan: [\'library_environment\']"', None))]),
]


def tree_state(root):
    files = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], capture_output=True,
                           check=True).stdout.split(b"\0")
    digest = hashlib.sha256()
    for name in sorted(f for f in files if f):
        digest.update(name + b"\0" + hashlib.sha256((root / name.decode()).read_bytes()).digest())
    return digest.hexdigest()


def tracked_state():
    status = subprocess.run(["git", "-C", str(WT), "status", "--porcelain"], capture_output=True,
                            text=True, check=True).stdout
    return tree_state(WT), status


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
        assert tree_state(box) == before[0], "the sandbox is not the worktree's tracked bytes"
        tests = sorted({t for *_x, declared in MUTANTS for t, _m in declared})
        for test in tests:
            rc, text, seconds = run_test(box, test)
            ran = "1 passed" in text
            report["baseline"].append({"test": test[1], "rc": rc, "one_passed": ran,
                                       "seconds": round(seconds, 1)})
            print(f"baseline {'ok ' if rc == 0 and ran else 'BAD'} {test[1]} ({seconds:.0f}s)", flush=True)
            if rc != 0 or not ran:
                (OUT / "logs" / f"baseline_{len(report['baseline']):02d}.log").write_text(text)
        if any(row["rc"] != 0 or not row["one_passed"] for row in report["baseline"]):
            raise SystemExit("an unmutated declared test failed or did not run; no mutant is interpretable")
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
                        (OUT / "logs" / f"{number:02d}_{len(row['tests'])}.log").write_text(output)
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
        report["sandbox_restored"] = tree_state(box) == before[0]
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
