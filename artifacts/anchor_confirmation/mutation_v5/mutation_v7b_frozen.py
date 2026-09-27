"""Mutation battery v7b (audit 709: the fixed four-dataset anchor model, contract v3 / generation v5).

Same method as v6 (artifacts/anchor_confirmation/mutation_v4/mutation_v6_frozen.py):
- Sandbox: a detached worktree at the source commit given on the command line, its tracked bytes
  checked equal to the anchor worktree's.
- Every declared test first passes unmutated.
- Each mutant is ONE exact replacement (count must be 1). It disables one predicate or a whole
  condition, removes one call, or (MX11 only, declared as such) re-introduces the superseded v2
  adoption rule. It runs only its declared tests, each alone with its own basetemp and a timeout.
- A mutant counts as detected only when every declared test fails (pytest rc 1) with the marker
  declared here in advance: the '>' line pytest marks (trailing comment removed) and, where given,
  a required text. Under this Python (3.10) a missing refusal is marked at the
  `with pytest.raises(...)` line (calibrated on a throwaway test before the run). Errors, other exit codes and timeouts are not detections.
- The real worktree's tracked bytes and git status are compared before and after; fixture processes
  live under the per-run basetemp and none may remain.
v7b differs from v7 only in MX13's declared line: the v7 run found that test was not isolating
(another guard refused first); the test was fixed and is re-run here with every other mutant.
"""
import hashlib, json, os, subprocess, sys, time
from pathlib import Path

WT = Path("/data/yschoi/gdna_anchor_confirm_v1")
COMMIT, OUT = sys.argv[1], Path(sys.argv[2])
PY = "/home/yschoi/.conda/envs/dna_hashing/bin/python"
L, DEC, PRB, REC = ("scripts/phase3_selection_matrix.py", "scripts/anchor_confirm_decision.py",
                    "scripts/anchor_confirm_code_axis.py", "dna_utils/scientific_recipe.py")
TL, TR, TC = ("tests/test_anchor_confirm_launcher.py", "tests/test_anchor_confirm_reducer.py",
              "tests/test_anchor_confirm_recipe.py")

MUTANTS = [
    ("MX1 the launcher lets a control arm into the arm plan", L,
     "    if arms != ANCHOR_RUN_ARMS:\n        raise CellRefused(f\"contract v3 runs the arms",
     "    if False:\n        raise CellRefused(f\"contract v3 runs the arms",
     [((TL, "test_invalid_arm_plans_refuse[None-none,anchors-runs the arms]"),
       ("with pytest.raises(CellRefused, match=reason):", "DID NOT RAISE")),
      ((TL, "test_invalid_arm_plans_refuse[None-none-runs the arms]"),
       ("with pytest.raises(CellRefused, match=reason):", "DID NOT RAISE"))]),
    ("MX2 the launcher accepts a plan without CIFAR-10 or with a control", L,
     "    if set(arm_plan) != set(ANCHOR_DATASETS) \\\n"
     "            or any(tuple(arms) != ANCHOR_RUN_ARMS for arms in arm_plan.values()):\n",
     "    if False:\n",
     [((TL, "test_a_plan_that_drops_a_dataset_or_runs_a_control_refuses[plan0]"),
       ('with pytest.raises(CellRefused, match="must run"):', "DID NOT RAISE")),
      ((TL, "test_a_plan_that_drops_a_dataset_or_runs_a_control_refuses[plan1]"),
       ('with pytest.raises(CellRefused, match="must run"):', "DID NOT RAISE"))]),
    ("MX3 the request's GPU count need not equal its dataset streams", L,
     "    if len(gpus) != streams or len(set(gpus)) != len(gpus):\n",
     "    if False:\n",
     [((TL, "test_a_changed_request_is_not_covered_by_the_old_approval[change3-needs exactly that many "
            "distinct GPUs]"),
       ("assert reason in capsys.readouterr().err and boundaries == []", None)),
      ((TL, "test_a_one_cell_smoke_needs_one_gpu_and_a_run_four"),
       ('assert run_main(monkeypatch, *BASE, *MODES[1], "--gpus", "0,1", *manifest_args, "--plan") == 2',
        None))]),
    ("MX4 the CIFAR-10 alternate literal is not admitted", REC,
     "              or spans[0] in REVIEWED_OVERRIDE_ALTERNATES.get(dest, ()),\n",
     "              or False,\n",
     [((TC, "test_the_cifar_wrapper_literal_is_a_reviewed_alternate_and_still_needs_the_override"),
       (None, "not a reviewed wrapper literal"))]),
    ("MX5 the first-occurrence check is disabled (any wrapper literal)", REC,
     "        _need(spans[0] == REVIEWED_OVERRIDES[dest]\n"
     "              or spans[0] in REVIEWED_OVERRIDE_ALTERNATES.get(dest, ()),\n",
     "        _need(True,\n",
     [((TC, "test_the_alternate_literal_is_narrow[argv0-not a reviewed wrapper literal]"),
       ("with pytest.raises(R.RecipeMismatch, match=reason):", "DID NOT RAISE")),
      ((TC, "test_the_alternate_literal_is_narrow[argv2-not a reviewed wrapper literal]"),
       ("with pytest.raises(R.RecipeMismatch, match=reason):", "DID NOT RAISE"))]),
    ("MX6 the reducer takes arms other than the anchor arm", DEC,
     "    need(arms == M.ANCHOR_RUN_ARMS,\n",
     "    need(True,\n",
     [((TR, "test_select_takes_the_anchor_arm_only[none,anchors]"),
       ("assert \"not the contract's ('anchors',)\" in capsys.readouterr().err", None))]),
    ("MX7 the reducer admits a record that is not a campaign receipt", DEC,
     '    if source == "receipt":\n',
     "    if True:\n",
     [((TR, "test_a_record_whose_source_is_not_a_campaign_receipt_refuses[reuse]"),
       ('assert run_select(world, tmp_path / "o.json", sources=world.sources(entries)) == 1', None))]),
    ("MX8 the reducer ignores the probe population", DEC,
     '    need(probe.get("population") == PROBE_POPULATION,\n',
     "    need(True,\n",
     [((TR, "test_a_probe_that_is_not_this_records_contract_measurement_refuses[changes12-measured another "
            "population]"),
       ('assert run_decide(evidence, frozen, frozen_sha, tmp_path / "d.json") == 1', None)),
      ((TR, "test_a_probe_that_is_not_this_records_contract_measurement_refuses[changes14-measured another "
            "population]"),
       ('assert run_decide(evidence, frozen, frozen_sha, tmp_path / "d.json") == 1', None))]),
    ("MX9 the reducer ignores the 500 x 4 counts", DEC,
     "    need(n_images == PROBE_IMAGES and total == PROBE_LOCAL_SLOTS * n_images and 0 <= hits <= total,\n",
     "    need(True,\n",
     [((TR, "test_a_probe_that_is_not_this_records_contract_measurement_refuses[changes9-500 images]"),
       ('assert run_decide(evidence, frozen, frozen_sha, tmp_path / "d.json") == 1', None)),
      ((TR, "test_a_probe_that_is_not_this_records_contract_measurement_refuses[changes10-500 images]"),
       ('assert run_decide(evidence, frozen, frozen_sha, tmp_path / "d.json") == 1', None))]),
    ("MX10 the probe measures a validation split short of 500 rows", PRB,
     "    D.need(len(ordered) >= D.PROBE_IMAGES,",
     "    D.need(True,",
     [((TR, "test_a_validation_split_short_of_500_rows_refuses"),
       ("with pytest.raises(D.NotReducible, match=\"has 499 rows, fewer than the contract's 500\"):",
        "DID NOT RAISE"))]),
    ("MX11 (re-introduction) the v2 adoption rule restores the control on a poor score", DEC,
     '            "axis_center": arm, "frozen_N": n,\n',
     '            "axis_center": arm if statistics.fmean(axis) > 0 else "none", "frozen_N": n,\n',
     [((TR, "test_poor_scores_are_reported_and_never_restore_the_control"),
       ('assert d["axis_center"] == "anchors" and d["frozen_N"] == 4', None))]),
    ("MX12 the probe measures every validation row, not the first 500", PRB,
     "    return ordered[:D.PROBE_IMAGES]\n",
     "    return ordered\n",
     [((TR, "test_the_probe_takes_the_first_500_rows_of_each_datasets_validation_split[nuswide]"),
       ('assert rows == sorted(int(i) for i in val)[:500] and len(rows) == D.PROBE_POPULATION["n_images"]',
        None)),
      ((TR, "test_the_probe_takes_the_first_500_rows_of_each_datasets_validation_split[mscoco]"),
       ('assert rows == sorted(int(i) for i in val)[:500] and len(rows) == D.PROBE_POPULATION["n_images"]',
        None))]),
    ("MX13 the stage-D replay ignores the fixed-architecture declaration", DEC,
     '         and frozen.get("fixed_architecture") == FIXED_ARCHITECTURE,\n',
     "         and True,\n",
     [((TR, "test_an_n_record_that_is_not_of_the_fixed_four_dataset_model_refuses[changes1]"),
       ('assert run_decide(evidence, frozen, frozen_sha, tmp_path / "d.json") == 1', None)),
      ((TR, "test_an_n_record_that_is_not_of_the_fixed_four_dataset_model_refuses[changes2]"),
       ('assert run_decide(evidence, frozen, frozen_sha, tmp_path / "d.json") == 1', None))]),
    ("MX14 an approval line of any version is accepted", L,
     '    want = {"version": ANCHOR_CONFIRM_VERSION, "scope": scope, **pins}\n',
     '    want = {"version": fields.get("version"), "scope": scope, **pins}\n',
     [((TL, "test_an_approval_that_is_not_exactly_this_operation_refuses[sections12-5-stage-S-run-pins12-"
            "approves stage-S-run for]"), ("with pytest.raises(CellRefused, match=reason):", "DID NOT RAISE"))]),
    ("MX15 the probe request omits the population", DEC,
     '            "population": dict(PROBE_POPULATION),\n            "records": sorted(',
     '            "records": sorted(',
     [((TR, "test_the_probe_request_names_the_population"),
       ('assert request.get("population") == D.PROBE_POPULATION == {', None))]),
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
