"""Mutation battery v8 (audit 715/716: the historical input-verification bridge, generation v6).

Same method as v7b (artifacts/anchor_confirmation/mutation_v5/mutation_v7b_frozen.py):
- Sandbox: a detached worktree at the source commit given on the command line, its tracked bytes
  checked equal to the anchor worktree's.
- Every declared test first passes unmutated.
- Each mutant is ONE exact replacement (count must be 1) that disables one check or removes one
  call. It runs only its declared tests, each alone with its own basetemp and a timeout.
- A mutant counts as detected only when every declared test fails (pytest rc 1) with the marker
  declared here in advance: the '>' line pytest marks (trailing comment removed) and, where given,
  a required text. Under Python 3.10 a missing refusal is marked at the `with pytest.raises(...)`
  line. "the historical verifier was started" is the no-child sentinel: the refusal it guards
  would have come before any rehash. BX9 is declared as a reason change (another guard, the
  authority's own seal digest, still refuses).
- The real worktree's tracked bytes and git status are compared before and after; fixture processes
  live under the per-run basetemp and none may remain.
"""
import hashlib, json, os, subprocess, sys, time
from pathlib import Path

WT = Path("/data/yschoi/gdna_anchor_confirm_v1")
COMMIT, OUT = sys.argv[1], Path(sys.argv[2])
PY = "/home/yschoi/.conda/envs/dna_hashing/bin/python"
L = "scripts/phase3_selection_matrix.py"
TL, TB = "tests/test_anchor_confirm_launcher.py", "tests/test_anchor_confirm_input_bridge.py"
CALL = 'M.verify_seal_historically(world["seal"])'
STARTED = (CALL, "the historical verifier was started")
FORGED = "test_a_forged_or_mutated_handoff_refuses[%s]"

MUTANTS = [
    ("BX1 the verifier's pinned bytes are not checked before the child", L,
     "    if verifier_pin() != HISTORICAL_SEAL_VERIFIER_SHA256:\n"
     "        raise SealError(f\"the historical verifier {verifier} is not the pinned bytes \"",
     "    if False:\n"
     "        raise SealError(f\"the historical verifier {verifier} is not the pinned bytes \"",
     [((TB, "test_an_unpinned_verifier_refuses_before_the_child"), STARTED)]),
    ("BX2 the seal's aggregate is not checked", L,
     "    if aggregate != hashlib.sha256(_canonical_bytes(payload)).hexdigest():\n",
     "    if False:\n",
     [((TB, "test_a_seal_whose_aggregate_does_not_hold_refuses_before_the_child"),
       ("M.verify_seal_historically(path)", "the historical verifier was started"))]),
    ("BX3 the seal's six source records are not checked against the pins", L,
     "    refusal = _historical_sources_refusal(payload)\n    if refusal is not None:\n",
     "    refusal = _historical_sources_refusal(payload)\n    if False:\n",
     [((TB, "test_a_seal_binding_another_producer_digest_refuses_before_the_child"
            "[scripts/build_counterfactual_caption_foils.py]"), STARTED),
      ((TB, "test_a_seal_binding_another_producer_digest_refuses_before_the_child"
            "[scripts/prepare_semantic_detail_cache.py]"), STARTED)]),
    ("BX4 the six historical files are not measured before the child (audit 716)", L,
     "    observed = _historical_source_observation(payload, \"before the historical verifier\")\n",
     "    observed = None\n",
     [((TB, "test_an_altered_historical_source_refuses_before_the_child[val_split.py]"), STARTED),
      ((TB, "test_an_altered_historical_source_refuses_before_the_child"
            "[scripts/build_text_whiten_matrix.py]"), STARTED),
      ((TB, "test_a_restamped_historical_source_refuses_before_the_child[val_split.py]"), STARTED)]),
    ("BX5 the six historical files are not re-measured after the child", L,
     "    if _historical_source_observation(payload, \"after the historical verifier\") != observed:\n",
     "    if False:\n",
     [((TB, "test_a_historical_source_changed_during_the_child_refuses_after_it[val_split.py]"),
       ('with pytest.raises(S.SealError, match=r"^after the historical verifier: the historical source"):',
        "DID NOT RAISE"))]),
    ("BX6 this generation's val_split.py is not checked", L,
     "    if not own.is_file() or _sha(own) != HISTORICAL_PRODUCER_SOURCES[\"val_split.py\"]:\n",
     "    if False:\n",
     [((TB, "test_a_changed_new_generation_val_split_refuses_before_the_child"), STARTED)]),
    ("BX7 a nonzero exit is not a refusal", L,
     "    if child.returncode != 0:\n", "    if False:\n",
     [((TB, FORGED % "print(f'verified {seal} {agg}')\\nsys.exit(2)\\n-refused \\\\(rc 2\\\\)"),
       ("with pytest.raises(S.SealError, match=reason):", "DID NOT RAISE"))]),
    ("BX8 the verifier's bytes are not re-checked after the child", L,
     "    if verifier_pin() != HISTORICAL_SEAL_VERIFIER_SHA256:\n"
     "        raise SealError(f\"the historical verifier {verifier} changed while it ran\")",
     "    if False:\n"
     "        raise SealError(f\"the historical verifier {verifier} changed while it ran\")",
     [((TB, FORGED % "open(__file__, 'a').write('#')\\nprint(f'verified {seal} {agg}')\\n-changed while it ran"),
       ("with pytest.raises(S.SealError, match=reason):", "DID NOT RAISE"))]),
    ("BX9 the seal's bytes are not re-checked after the child (reason change)", L,
     "    if hashlib.sha256(seal_bytes()).hexdigest() != before:\n"
     "        raise SealError(f\"seal {seal_path} changed while the historical verifier ran\")",
     "    if False:\n"
     "        raise SealError(f\"seal {seal_path} changed while the historical verifier ran\")",
     [((TB, FORGED % "os.chmod(seal, 0o644)\\nopen(seal, 'a').write(' ')\\nprint(f'verified {seal} {agg}')"
                     "\\n-changed while the historical verifier ran"),
       ("with pytest.raises(S.SealError, match=reason):", "is not the bytes the historical verifier checked"))]),
    ("BX10 the child's report is not checked", L,
     "    if (out or \"\").splitlines() != [line]:\n", "    if False:\n",
     [((TB, FORGED % "print(f'verified {seal} {\"0\" * 64}')\\n-report is not"),
       ("with pytest.raises(S.SealError, match=reason):", "DID NOT RAISE")),
      ((TB, FORGED % "pass\\n-report is not"),
       ("with pytest.raises(S.SealError, match=reason):", "DID NOT RAISE"))]),
    ("BX11 a stale expected authority is accepted", L,
     "    if expected is not None and dict(authority) != dict(expected):\n"
     "        raise SealError(\"Phase-3 input authority changed across the historical verification\")",
     "    if False:\n"
     "        raise SealError(\"Phase-3 input authority changed across the historical verification\")",
     [((TB, "test_a_stale_expected_authority_refuses"),
       ('with pytest.raises(S.SealError, match="authority changed across the historical verification"):',
        "DID NOT RAISE"))]),
    ("BX12 the child does not die with its launcher", L,
     "                             preexec_fn=_die_with_parent(os.getpid()))",
     "                             preexec_fn=None)",
     [((TB, "test_the_child_dies_with_a_killed_launcher"),
       ('assert not alive(child), "the historical verifier outlived its launcher"', None))]),
    ("BX13 an interrupted wait leaves the child running", L,
     "    finally:\n        if child.poll() is None:\n            child.kill()\n",
     "    finally:\n        if False:\n            child.kill()\n",
     [((TB, "test_an_interrupted_wait_kills_the_child"),
       ('assert not alive(child), "an interrupted wait left the historical verifier running"', None))]),
    ("BX14 the anchor admission does not ask for the historical verifier", L,
     "                historical=True, evidence=historical_admission)",
     "                evidence=historical_admission)",
     [((TL, "test_the_anchor_admission_uses_the_historical_verifier_and_records_it"),
       ('assert seen.get("full") is True and seen.get("historical") is True and seen.get("expected") is None',
        None))]),
    ("BX15 a full anchor check does not take the bridge", L,
     "            if full and historical:\n", "            if False:\n",
     [((TB, "test_the_campaign_check_takes_the_bridge_only_for_a_full_anchor_admission"),
       ("admitted = M.verify_campaign_input_seals(specs, [CELL], full=True, historical=True, evidence=evidence)",
        "drifted"))]),
    ("BX16 the manifest's historical input-verifier pins are not checked", L,
     "    if historical.get(\"historical_input_verifier\") != historical_input_verifier_pins():\n",
     "    if False:\n",
     [((TL, "test_a_manifest_that_is_not_this_generation_refuses[changes7-historical input-verifier pins]"),
       ("with pytest.raises(CellRefused, match=reason):", "DID NOT RAISE")),
      ((TL, "test_a_manifest_that_is_not_this_generation_refuses[changes8-historical input-verifier pins]"),
       ("with pytest.raises(CellRefused, match=reason):", "DID NOT RAISE"))]),
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
