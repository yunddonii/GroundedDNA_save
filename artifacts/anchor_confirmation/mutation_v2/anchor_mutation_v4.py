"""Anchor confirmation v2 mutation battery, fourth revision, targeted (audit 673.3, 688.1, 698).

Each mutant breaks ONE guard in a sandbox COPY of the anchor worktree's tracked files and runs only
the tests declared to catch it. A mutant counts as detected only when those tests fail (pytest rc 1)
and the output carries the failure mode declared in advance (`expect`); a collection/import error,
another rc or a timeout is NOT detection. An unmutated baseline of every test group must pass first.
The real worktree's tracked bytes and git status are compared before and after.

Compound conditions are disabled as a WHOLE (`True or ...`, `if False:`) or one predicate at a time,
never by a text prefix that leaves a later `or` operand live (688.1). Boundary mutants are caught
by the tests' independent deny guards (no process, no deserialisation, no model).

usage: python anchor_mutation_v4.py <new sandbox dir> <report.json>
"""
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REAL = Path("/data/yschoi/gdna_anchor_confirm_v1")
SB = Path(sys.argv[1])
REPORT = Path(sys.argv[2])
PY = "/home/yschoi/.conda/envs/dna_hashing/bin/python"
ENV = {"PATH": "/usr/bin:/bin", "HOME": "/home/yschoi", "CUDA_VISIBLE_DEVICES": "",
       "GDNA_NUM_SEMANTIC_PARTS": "5"}
REC, LAU, RED, POR = ("tests/test_anchor_confirm_recipe.py", "tests/test_anchor_confirm_launcher.py",
                      "tests/test_anchor_confirm_reducer.py", "tests/test_anchor_confirm_port.py")
HELPER, LAUNCHER, REDUCER, PROBE = ("dna_utils/scientific_recipe.py", "scripts/phase3_selection_matrix.py",
                                    "scripts/anchor_confirm_decision.py", "scripts/anchor_confirm_code_axis.py")
RAISE = "DID NOT RAISE"
RC_PLAN, RC_REDUCE = r"assert 0 == 2", r"assert 0 == 1"
DEPTH = "defense in depth: another guard still refuses; detected by the reason assertion"

# (id, file, old, new, tests, expect regex, note)
RAISE_OR_DEPTH = RAISE + "|Regex pattern did not match"
NEW = [
    # --- audit 694: approved inputs survive admission ---
    ("N1 admitted seals are the approved request's (before the lease)", LAUNCHER,
     '            assert_request_seals(request, input_seals, "at input admission")', '            pass',
     [f"{LAU}::test_a_seal_changed_after_its_approval_refuses_before_the_lease"], RC_PLAN, ""),
    ("N2 carried authority bytes are the approved pin while parsed", LAUNCHER,
     '                if not (before == _file_pin(args.admission_authority) == request["admission_authority"]):',
     '                if False:', [f"{LAU}::test_the_carried_authority_is_the_approved_bytes_while_it_is_parsed"],
     RC_PLAN, ""),
    ("N4 re-admitted seals after the lease are the approved request's", LAUNCHER,
     '            assert_request_seals((authorities or {}).get("anchor_request"), input_seals,\n                                 "after the lease")',
     '            pass', [f"{LAU}::test_seals_readmitted_after_the_lease_must_still_be_the_approved_ones"],
     r"assert (0|1) == 2", "the sweep proceeds past the lease with foreign seals"),
    # --- audit 697: the generation is re-verified at every boundary ---
    ("N3 generation re-verified after input admission", LAUNCHER,
     '            recheck_generation(manifest, "after input admission")', '            pass',
     [f"{LAU}::test_a_generation_drifting_during_input_admission_refuses_before_the_lease"], RC_PLAN, ""),
    ("N5 generation re-verified after the lease", LAUNCHER,
     '            recheck_generation((authorities or {}).get("anchor_manifest"), "after the lease")', '            pass',
     [f"{LAU}::test_a_generation_drifting_inside_the_sweep_stops_before_its_next_boundary"], r"assert 0 == 2", ""),
    ("N6 generation re-verified before every cell", LAUNCHER,
     '                    recheck_generation((authorities or {}).get("anchor_manifest"), f"before {key}")',
     '                    pass', [f"{LAU}::test_a_generation_drifting_inside_the_sweep_stops_before_its_next_boundary"],
     r"assert 0 == 1", ""),
    ("N7 generation re-verified before the receipt", LAUNCHER,
     '            recheck_generation((authorities or {}).get("anchor_manifest"), "before the receipt")',
     '            pass', [f"{LAU}::test_a_generation_drifting_inside_the_sweep_stops_before_its_next_boundary"],
     r"assert 0 == 1", ""),
    ("N9 recheck reads every pinned file", LAUNCHER,
     '        again = load_anchor_manifest(manifest["path"], manifest["sha256"])',
     '        again = {"sha256": manifest["sha256"], "files_sha256": {}}',
     [f"{LAU}::test_a_member_changed_after_admission_refuses"], RAISE, ""),
    ("N10 recheck binds the imported bytes", LAUNCHER,
     '        if rel in pins and digest != pins[rel]:', '        if False:',
     [f"{LAU}::test_a_module_imported_from_other_bytes_refuses"], RAISE, ""),
    # --- audit 696: input consistency in the reducer ---
    ("N11 campaign seals are its approved request's", REDUCER,
     '    need(bool(admitted) and admitted == request.get("input_seals"),', '    need(True,',
     [f"{RED}::test_inconsistent_input_bindings_refuse_before_publication",
      f"{RED}::test_a_request_approved_for_other_seals_refuses"], RC_REDUCE, ""),
    ("N12 record input authority is the admitted seal", REDUCER,
     '        need(isinstance(inputs, dict) and record.get("input_authority") == inputs,', '        need(True,',
     [f"{RED}::test_inconsistent_input_bindings_refuse_before_publication"], RC_REDUCE, ""),
    ("N13 launch and cell bindings carry the admitted seal", REDUCER,
     '            need(is_sha256(value) and campaign.get(field) == value and binding.get(field) == value,',
     '            need(True,', [f"{RED}::test_inconsistent_input_bindings_refuse_before_publication"], RC_REDUCE, ""),
    ("N14 reducer re-verifies its generation before publishing", REDUCER,
     '        M.recheck_generation(manifest, "before the reducer publishes")', '        pass',
     [f"{RED}::test_the_reducer_publishes_nothing_if_its_generation_drifts_before_publication"], r"assert", ""),
    ("N16 probe re-verifies its generation before its first load", PROBE,
     '    M.recheck_generation(manifest, "before the probe\'s first load")', '    pass',
     [f"{RED}::test_the_probe_rechecks_its_generation_before_its_first_load"], r"a binary was deserialised",
     "reaches the (sentinel) first load"),
]
RERUN = [
    ('L5 execution needs the manifest', 'scripts/phase3_selection_matrix.py', '        if execute and manifest is None:', '        if False:', ['tests/test_anchor_confirm_launcher.py::test_execution_without_the_generation_manifest_refuses'], "assert 'need --anchor-manifest' in", 'defense in depth: another guard still refuses; detected by the reason assertion (the approval needs the manifest digest)'),
    ('L14 decide: stage S under this generation', 'scripts/phase3_selection_matrix.py', '            if manifest is not None and selection.get("anchor_manifest_sha256") != manifest["sha256"]:', '            if False:', ['tests/test_anchor_confirm_launcher.py::test_decide_refuses_a_stage_s_from_another_generation'], 'assert 0 == 2', ''),
    ('L19 run_cell needs the sealed recipe', 'scripts/phase3_selection_matrix.py', '    if anchor_arm is not None and (campaign_binding is None\n                                   or not isinstance(scientific_recipe, dict)):', '    if False:', ['tests/test_anchor_confirm_launcher.py::test_run_cell_refuses_an_anchor_cell_without_a_sealed_recipe'], 'a trainer was launched', 'reaches the (sentinel) trainer launch'),
    ('L22e request binds the input seals', 'scripts/phase3_selection_matrix.py', '        "input_seals": {f"{ds}:{stage}": _file_pin(path)', '        "input_seals": {} and {f"{ds}:{stage}": _file_pin(path)', ['tests/test_anchor_confirm_launcher.py::test_a_changed_request_is_not_covered_by_the_old_approval'], 'assert 0 == 2', ''),
    ('L22f request binds the admission authority', 'scripts/phase3_selection_matrix.py', '        "admission_authority": (_file_pin(args.admission_authority)', '        "admission_authority": None and (_file_pin(args.admission_authority)', ['tests/test_anchor_confirm_launcher.py::test_a_changed_request_is_not_covered_by_the_old_approval'], 'assert 0 == 2', ''),
    ('D4 decide: stage S under the given generation', 'scripts/anchor_confirm_decision.py', '    need(selection["anchor_manifest_sha256"] == manifest["sha256"],', '    need(True,', ['tests/test_anchor_confirm_reducer.py::test_decide_refuses_stage_s_from_another_generation'], "assert 'stage S ran under another generation manifest' in", "defense in depth: another guard still refuses; detected by the reason assertion (the stage-S records' approvals are re-verified for the decide generation)"),
    ('D8 selection replay', 'scripts/anchor_confirm_decision.py', '    need(replay["n_selected"] == frozen["n_selected"] and replay["scores"] == frozen["scores"],', '    need(True,', ['tests/test_anchor_confirm_reducer.py::test_a_frozen_n_record_that_its_evidence_does_not_reproduce_refuses'], "assert 'not what its own evidence reduces to' in", 'defense in depth: another guard still refuses; detected by the reason assertion (the evidence membership no longer matches the tampered N)'),
    ('D9 probe counts', 'scripts/anchor_confirm_decision.py', '    need(n_images == PROBE_IMAGES and total == 4 * n_images and 0 <= hits <= total,', '    need(True,', ['tests/test_anchor_confirm_reducer.py::test_a_probe_that_is_not_this_records_contract_measurement_refuses'], 'assert 0 == 1', ''),
    ('D38 probe split identity (whole condition)', 'scripts/anchor_confirm_decision.py', '    need(is_sha256(probe.get("split_identity_sha256"))', '    need(True or is_sha256(probe.get("split_identity_sha256"))', ['tests/test_anchor_confirm_reducer.py::test_a_consistently_other_split_identity_refuses'], 'assert 0 == 1', ''),
]
MUTANTS = NEW + RERUN
UNREACHABLE = {
    "probe recheck before publication": "after the real forward pass; synthetic tests cannot reach it",
    "probe `_assert_phase3_runtime_rows` after the real dataset load": "needs real data",
}
OUTPUTS = REPORT.parent / (REPORT.stem + "_outputs")


def tracked(root: Path) -> dict:
    names = subprocess.run(["git", "ls-files", "-z"], cwd=REAL, capture_output=True,
                           check=True).stdout.split(b"\0")
    out = {}
    for raw in names:
        if raw:
            rel = raw.decode()
            path = root / rel
            if path.is_file():
                out[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def status() -> str:
    return subprocess.run(["git", "status", "--porcelain"], cwd=REAL, capture_output=True, text=True,
                          check=True).stdout


def pytest(tests, timeout=900):
    try:
        r = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--tb=short", "-rfE",
                            *tests], cwd=SB, capture_output=True, text=True, env=ENV, timeout=timeout)
        return r.returncode, r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return "timeout", ""


def main():
    assert not SB.exists(), f"{SB} exists; the sandbox must be fresh"
    before, status_before = tracked(REAL), status()
    # a detached worktree at HEAD (the port tests diff against the historical base commit), then
    # the anchor worktree's current bytes of every tracked file on top
    subprocess.run(["git", "worktree", "add", "--detach", str(SB), "HEAD"], cwd=REAL, check=True,
                   capture_output=True)
    for rel in before:
        (SB / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REAL / rel, SB / rel)
    assert tracked(SB) == before, "the sandbox is not a faithful copy"
    for name, rel, old, new, *_ in MUTANTS:              # every mutant applies exactly once
        assert (SB / rel).read_text().count(old) == 1, (name, (SB / rel).read_text().count(old))
    groups = sorted({t for m in MUTANTS for t in m[4]})
    baseline = {}
    for group in groups:
        rc, out = pytest([group])
        baseline[group] = rc
        print(f"baseline {'PASS' if rc == 0 else 'FAIL'} {group}", flush=True)
        assert rc == 0, f"baseline failed for {group}:\n{out[-3000:]}"
    results = []
    for name, rel, old, new, tests, expect, note in MUTANTS:
        path = SB / rel
        original = path.read_bytes()
        path.write_text(original.decode().replace(old, new))
        started = time.time()
        rc, out = pytest(tests)
        path.write_bytes(original)
        OUTPUTS.mkdir(exist_ok=True)
        output_file = OUTPUTS / (re.sub(r"[^A-Za-z0-9]+", "_", name) + ".txt")
        output_file.write_text(out if isinstance(out, str) else "")
        assert hashlib.sha256(path.read_bytes()).hexdigest() == before[rel], f"{rel} not restored"
        failed = [l for l in out.splitlines() if l.startswith(("FAILED", "ERROR"))]
        last = out.strip().splitlines()[-1] if out.strip() else ""
        errored = rc != 1 or any(l.startswith("ERROR") for l in failed) or " error" in last
        if expect is None:
            outcome = "SURVIVED (declared unreachable)" if rc == 0 else "DETECTED (declared unreachable!)"
        elif rc == 0:
            outcome = "SURVIVED"
        elif errored:
            outcome = "NOT DETECTED (error/timeout)"
        elif re.search(expect, out):
            outcome = "KILLED (intended)"
        else:
            outcome = "KILLED (other reason)"
        results.append({"mutant": name, "file": rel, "tests": tests, "expect": expect, "note": note,
                        "rc": rc, "outcome": outcome, "failed": failed[:6], "output": str(output_file),
                        "seconds": round(time.time() - started, 1)})
        print(f"{outcome:32s} {name:48s} {(failed[0] if failed else '')[:100]}", flush=True)
    after, status_after = tracked(REAL), status()
    summary = {
        "sandbox": str(SB), "baseline": baseline, "mutants": results, "unreachable_by_synthetic_tests": UNREACHABLE,
        "counts": {k: sum(r["outcome"] == k for r in results) for k in sorted({r["outcome"] for r in results})},
        "real_tree_unchanged": after == before and status_after == status_before,
        "real_tree_tracked_files": len(before),
    }
    REPORT.write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["counts"]), "real tree unchanged:", summary["real_tree_unchanged"])


if __name__ == "__main__":
    main()
