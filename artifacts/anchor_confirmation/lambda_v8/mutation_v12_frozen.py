"""Mutation battery v12 (audits 734-736: generation v8 revision 2, stage L; v11's 26 mutants).

v11's method, with the read paths bounded (audit 736):
  * source state: content hashes ONLY for the reviewed inventory (the generation manifest's closure
    plus the declared test files), via artifacts/anchor_confirmation/lambda_v8/bounded_tree.py; every
    other tracked file is compared by git index object id and stat metadata, never read. (v10 and
    v11 hashed every tracked file, including six unrelated artifacts/umrch_* .npy/.npz binaries.)
  * this harness runs under its own open() guard: binary payload suffixes outside /tmp and the
    interpreter, and real-data roots, are refused; os.exec* is refused; children are recorded.
  * every declared test runs under guarded_pytest.py (the open() guard of the pytest child) with its
    own log directory; a run counts only with zero refused opens.
  * the sandbox is a sparse checkout without artifacts/ (v10/v11 full checkouts materialised the
    unrelated binaries) and, with the outputs, lives under the scratch OUT directory.
The baseline (each declared test passes unmutated first), one exact replacement per mutant, the
declared-text detection rule and the restoration checks are v11's.
"""
import hashlib, json, os, subprocess, sys, time
from pathlib import Path

WT = Path("/data/yschoi/gdna_anchor_lambda_v8")
COMMIT, OUT = sys.argv[1], Path(sys.argv[2])
PY = "/home/yschoi/.conda/envs/dna_hashing/bin/python"
L, D, S, R = ("scripts/phase3_selection_matrix.py", "scripts/anchor_confirm_decision.py",
              "scripts/anchor_confirm_supervisor.py", "dna_utils/scientific_recipe.py")
T, TL = "tests/test_anchor_lambda_stage.py", "tests/test_anchor_confirm_launcher.py"
NO_RAISE = "DID NOT RAISE"

MUTANTS = [
    ("LX1 the stream passes no override in anchor mode", L,
     '                                             or getattr(args, "anchor_confirm", None)\n'
     '                                             == ANCHOR_LAMBDA_STAGE\n',
     '                                             or False\n',
     [((T, "test_every_cell_receives_its_own_override_and_the_campaign_completes"),
       "assert [(label, overrides) for _, label, overrides in ran] == ")]),
    ("LX2 the plan snapshot seals no override", L,
     "                                  anchor_arm=anchor_arm, input_authority=input_authority,\n"
     "                                  overrides=overrides)\n",
     "                                  anchor_arm=anchor_arm, input_authority=input_authority,\n"
     "                                  overrides=())\n",
     [((T, "test_the_snapshot_binding_seals_each_cells_own_override[lambda_bu=0]"),
       "assert {k: fields[k] for k in M.LAMBDA_AXES} == {k: want[k] for k in M.LAMBDA_AXES}")]),
    ("LX3 the stream ignores the control gate", L,
     "            if refusal:\n                print(f\"[phase3] STOPPED",
     "            if False:\n                print(f\"[phase3] STOPPED",
     [((T, "test_a_control_off_its_reference_stops_the_stream_before_any_candidate[0.75]"),
       "assert stream.go() == 1")]),
    ("LX4 the gate tolerates a near score", L,
     "    if type(got) is not float or got != want:\n",
     "    if type(got) is not float or abs(got - want) > 1e-9:\n",
     [((T, "test_a_control_off_its_reference_stops_the_stream_before_any_candidate[0.5000000000000001]"),
       "assert stream.go() == 1"),
      ((T, "test_an_approved_run_verifies_inputs_then_leases_with_the_control_gate"),
       'assert refusal is not None and "continuity control scored" in refusal')]),
    ("LX5 the admission skips the v7 incumbent comparison", L,
     "            if drift:\n                raise CellRefused(f\"{ds}: the rendered incumbent",
     "            if False:\n                raise CellRefused(f\"{ds}: the rendered incumbent",
     [((T, "test_an_incumbent_that_is_not_the_v7_cells_recipe_refuses"), NO_RAISE)]),
    ("LX6 the admission accepts any difference from the incumbent", L,
     "        if differing != expected:\n            raise CellRefused(f\"{key}: differs from the incumbent",
     "        if False:\n            raise CellRefused(f\"{key}: differs from the incumbent",
     [((T, "test_a_render_that_moves_a_second_field_refuses"), NO_RAISE)]),
    ("LX7 the admission does not require the control and every candidate", L,
     "        if not want <= seen:\n", "        if False:\n",
     [((T, "test_an_off_contract_plan_refuses[<lambda>-lacks0]"), NO_RAISE)]),
    ("LX8 an anchor cell tuple admits any override", L,
     "            anchor_lambda_label(cell[0], overrides)      # one declared stage-L candidate, or refuse\n",
     "            pass\n",
     [((TL, "test_eight_tuples_are_anchor_cells_and_nothing_else_changes"), NO_RAISE)]),
    ("LX9 an anchor command admits any override", L,
     "            anchor_lambda_label(dataset, overrides)      # one declared stage-L candidate, or refuse\n",
     "            pass\n",
     [((TL, "test_an_anchor_cell_moves_axis_center_alone_or_one_declared_stage_l_lambda[nuswide-overrides3]"),
       NO_RAISE)]),
    ("LX10 the v7 D seed-42 score is not tied to the N record", L,
     "        if seeds[0] != (((frozen.get(", "        if False and seeds[0] != (((frozen.get(",
     [((T, "test_inconsistent_v7_history_refuses[decision-<lambda>-not the frozen N record's]"), NO_RAISE)]),
    ("LX11 the reducer threshold is not strict", D,
     '            qualifies = delta > h["threshold"]\n', '            qualifies = delta >= h["threshold"]\n',
     [((T, "test_the_threshold_is_strict_on_unrounded_values"),
       'assert by[("lambda_bu", 0.0)]["delta"] == 0.25 and by[("lambda_bu", 0.0)]["qualifies"] is False')]),
    ("LX12 the reducer drops the control equality", D,
     '        need(control["score"] == h["incumbent_seed42"],\n', "        need(True,\n",
     [((T, "test_a_control_off_the_v7_score_refuses_the_decision[0.5000000000000001]"), "assert rc == 1")]),
    ("LX13 the reducer skips the control-to-v7 recipe check", D,
     "        need(not drift, f\"{ds}: the continuity control's sealed recipe",
     "        need(True, f\"{ds}: the continuity control's sealed recipe",
     [((T, "test_a_control_whose_recipe_is_not_the_v7_cells_refuses"),
       "assert rc == 1 and \"continuity control's sealed recipe\" in capsys.readouterr().err")]),
    ("LX14 the reducer skips the one-lambda check", D,
     "            need(moved == [flag] and canonical(", "            need(True or moved == [flag] and canonical(",
     [((T, "test_a_candidate_sealed_with_a_second_change_refuses"), "assert rc == 1 and")]),
    ("LX15 the request's preregistered rule is not checked", D,
     '    need(request.get("lambda") == lambda_rule,\n', "    need(True,\n",
     [((T, "test_a_request_without_the_preregistered_rule_refuses"), "assert rc == 1 and")]),
    ("LX16 the tie order prefers the farther value", D,
     "    return min(qualifying, key=lambda vs: (-vs[1], nearness(vs[0]), vs[0]))[0]\n",
     "    return min(qualifying, key=lambda vs: (-vs[1], -nearness(vs[0]), vs[0]))[0]\n",
     [((T, "test_the_highest_qualifier_wins_and_ties_go_to_the_value_nearer_the_incumbent"),
       'assert result["datasets"][FLICKR]["winners"] == {"lambda_wasserstein": 0.3, "lambda_bu": None,')]),
    ("LX17 the S/D admission admits an override", D,
     '    need(role == "lambda" or not overrides, "an S/D record carries no lambda override")\n',
     '    need(True, "an S/D record carries no lambda override")\n',
     [((T, "test_an_s_d_admission_refuses_an_override"), "Regex pattern did not match")]),
    ("LX18 two lambda repeats can be sealed", R,
     '    _need(len(lambdas) <= 1, f"a stage-L cell moves one lambda, not {lambdas}")\n',
     '    _need(True, f"a stage-L cell moves one lambda, not {lambdas}")\n',
     [((T, "test_two_lambda_repeats_cannot_be_sealed[tail0-moves one lambda]"), NO_RAISE)]),
    ("LX19 a lambda repeat after another literal can be sealed", R,
     "            _need(spans[0] == STAGE_L_REVIEWED_OVERRIDES[dest],\n", "            _need(True,\n",
     [((T, "test_a_lambda_repeat_after_another_literal_cannot_be_sealed"), NO_RAISE)]),
    ("LX20 the supervisor lets stage L use another ledger root", S,
     "    if is_l and root != L_OPS_ROOT:\n", "    if False:\n",
     [((T, "test_a_crossed_ledger_root_refuses[stage-L-run-/home/yschoi/gdna_anchor4_ops]"), NO_RAISE)]),
    ("LX21 the request omits the preregistered rule", L,
     '        **({"lambda": lambda_rule} if lambda_rule is not None else {}),\n', "        **({}),\n",
     [((T, "test_the_plan_previews_the_stage_l_request_and_touches_nothing"),
       'assert request.get("lambda") == M.anchor_lambda_rule(M.anchor_v7_history(INC))')]),
    ("LX22 the decision does not require one campaign", D,
     "    one_lambda_campaign(consumed, coords, keyed, admitted, incumbent)\n", "",
     [((T, "test_two_complete_campaigns_cannot_be_mixed[candidates_from_another_campaign]"),
       'assert rc == 1 and "come from 2 campaign receipts" in capsys.readouterr().err'),
      ((T, "test_two_complete_campaigns_cannot_be_mixed[substituted_control]"),
       'assert rc == 1 and "come from 2 campaign receipts" in capsys.readouterr().err')]),
    ("LX23 the receipt's cell membership is not checked", D,
     "    need(listed == ids, f\"the campaign receipt lists", "    need(True, f\"the campaign receipt lists",
     [((T, "test_a_receipt_listing_a_cell_beyond_the_six_refuses"),
       'assert rc == 1 and "beyond or short of the 6 stage-L cells" in capsys.readouterr().err')]),
    ("LX24 the request's cell membership is not checked", D,
     '    need(request.get("mode") == "run" and request.get("cells") == want\n',
     '    need(True or request.get("mode") == "run" and request.get("cells") == want\n',
     [((T, "test_a_request_declaring_a_cell_beyond_the_six_refuses"),
       'assert rc == 1 and "execute and declare exactly the stage-L cells" in capsys.readouterr().err')]),
    ("LX25 the historical reads bypass the read-once record", D,
     "    read = consumed_reader(consumed)\n", "    read = None\n",
     [((T, "test_the_v7_snapshot_changing_before_publication_refuses_and_writes_nothing"),
       'assert str(snapshot) in payload["_consumed"].digests'),
      ((T, "test_the_reduction_records_every_historical_file_it_read"),
       "assert {str(M.ANCHOR_V7_SELECTION), str(M.ANCHOR_V7_DECISION), str(M.ANCHOR_V7_S_RECEIPT),")]),
    ("LX26 the v7 snapshot alone is read outside the reader", L,
     '        snapshot = json.loads(read(snapshot_path, None, "v7 plan snapshot"))\n',
     '        snapshot = json.loads(snapshot_path.read_bytes())\n',
     [((T, "test_the_reduction_records_every_historical_file_it_read"),
       "assert {str(M.ANCHOR_V7_SELECTION), str(M.ANCHOR_V7_DECISION), str(M.ANCHOR_V7_S_RECEIPT),")]),
]


GUARD = WT / "artifacts/anchor_confirmation/lambda_v8/guarded_pytest.py"
BOUNDED = WT / "artifacts/anchor_confirmation/lambda_v8/bounded_tree.py"
MANIFEST = WT / "artifacts/anchor_confirmation/authority_manifest_v8r2.json"
EXTRAS = sorted({t[0] for *_x, d in MUTANTS for t, _m in d})
BINARY = (".npz", ".npy", ".pt", ".pth", ".safetensors", ".bin", ".ckpt", ".pkl")
REFUSED, CHILDREN = [], []


def _harness_hook(event, args):
    if event == "os.exec":
        REFUSED.append(f"exec:{args[0]}")
        raise PermissionError("battery guard refused a process image replacement")
    if event == "subprocess.Popen":
        CHILDREN.append([os.fsdecode(a) for a in (args[1] or [])][:4])
        return
    if event != "open" or not args or isinstance(args[0], int):
        return
    try:
        path = os.path.abspath(os.fsdecode(args[0]))
    except (TypeError, ValueError):
        return
    safe = path.startswith(("/tmp/", sys.prefix + "/", sys.base_prefix + "/"))
    in_wt = path.startswith(str(WT) + "/")
    allowed_wt = in_wt and (not path.startswith(str(WT / "artifacts") + "/")
                            or path in (str(MANIFEST), str(BOUNDED), str(GUARD), os.path.abspath(__file__)))
    deny = (path.endswith(BINARY) and not safe) or \
        (path.startswith(("/data/", "/home/")) and not safe and not allowed_wt)
    if deny:
        REFUSED.append(path)
        raise PermissionError(f"battery guard refused an open: {path}")


import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("bounded_tree", BOUNDED)
BT = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(BT)


def tree_state(root):
    return BT.state(root, BT.inventory(root, MANIFEST, EXTRAS))


def tracked_state():
    return tree_state(WT), BT.head_clean(WT, BT.inventory(WT, MANIFEST, EXTRAS))


RUN = [0]


def run_test(box, test, timeout=600):
    RUN[0] += 1
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", GDNA_NUM_SEMANTIC_PARTS="5")
    env.pop("PYTHONPATH", None)
    env.pop("GDNA_ALLOW_REAL_ARTIFACT_TESTS", None)
    guard_dir = OUT / "guard" / str(RUN[0])
    began = time.monotonic()
    try:
        done = subprocess.run([PY, str(GUARD), str(guard_dir), "-q", "-p", "no:cacheprovider",
                               f"--basetemp={OUT / 'tmp' / str(RUN[0])}", f"{test[0]}::{test[1]}"],
                              cwd=box, env=env, capture_output=True, text=True, timeout=timeout)
        refused = json.loads((guard_dir / "violations.json").read_text())["violations"]
        return done.returncode, done.stdout + done.stderr, time.monotonic() - began, refused
    except subprocess.TimeoutExpired as error:
        return None, f"TIMEOUT\n{error.stdout or ''}", time.monotonic() - began, ["timeout"]


def main():
    sys.addaudithook(_harness_hook)
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "logs").mkdir()
    (OUT / "tmp").mkdir()
    (OUT / "guard").mkdir()
    before = tracked_state()
    box = OUT / "sandbox"
    # A sparse sandbox without artifacts/: a full checkout would materialise the unrelated tracked
    # binaries (v10 and v11 sandboxes did) although no declared test needs them.
    subprocess.run(["git", "-C", str(WT), "worktree", "add", "--no-checkout", "--detach", str(box), COMMIT],
                   check=True, capture_output=True)
    subprocess.run(["git", "-C", str(box), "sparse-checkout", "set", "--no-cone", "/*", "!/artifacts/"],
                   check=True, capture_output=True)
    subprocess.run(["git", "-C", str(box), "checkout", "--detach", COMMIT], check=True, capture_output=True)
    report_sparse = sorted(p.name for p in box.iterdir())
    report = {"commit": COMMIT, "baseline": [], "mutants": [], "sandbox_top_level": report_sparse,
              "sandbox_has_artifacts": (box / "artifacts").exists()}
    try:
        assert not before[1], f"the worktree is not clean at HEAD for the inventory: {before[1][:6]}"
        assert tree_state(box)["content"] == before[0]["content"], "the sandbox is not the worktree's inventory bytes"
        for name, rel, old, new, declared in MUTANTS:     # every pattern exactly once before any run
            assert (box / rel).read_text().count(old) == 1, f"{name}: pattern count is not 1"
        tests = sorted({t for *_x, declared in MUTANTS for t, _m in declared})
        for test in tests:
            rc, text, seconds, refused = run_test(box, test)
            ran = "1 passed" in text and not refused
            report["baseline"].append({"test": test[1], "rc": rc, "one_passed": ran, "refused_opens": refused,
                                       "seconds": round(seconds, 1)})
            print(f"baseline {'ok ' if rc == 0 and ran else 'BAD'} {test[1]} ({seconds:.0f}s)", flush=True)
            if rc != 0 or not ran:
                (OUT / "logs" / f"baseline_{len(report['baseline']):02d}.log").write_text(text)
        if any(row["rc"] != 0 or not row["one_passed"] for row in report["baseline"]):
            raise SystemExit("an unmutated declared test failed or did not run; no mutant is interpretable")
        for number, (name, rel, old, new, declared) in enumerate(MUTANTS):
            path = box / rel
            original = path.read_bytes()
            row = {"mutant": name, "file": rel, "tests": []}
            path.write_text(original.decode().replace(old, new))
            try:
                for test, marker in declared:
                    rc, output, seconds, refused = run_test(box, test)
                    intended = rc == 1 and marker in output and not refused
                    (OUT / "logs" / f"{number:02d}_{len(row['tests'])}.log").write_text(output)
                    row["tests"].append({"test": test[1], "rc": rc, "declared_text": marker, "refused_opens": refused,
                                         "intended": intended, "seconds": round(seconds, 1)})
            finally:
                path.write_bytes(original)
            assert hashlib.sha256(path.read_bytes()).digest() == hashlib.sha256(original).digest()
            row["result"] = ("detected" if row["tests"] and all(t["intended"] for t in row["tests"])
                             else "NOT DETECTED AS DECLARED")
            report["mutants"].append(row)
            print(f"{row['result']:26s} {name}", flush=True)
        report["sandbox_restored"] = tree_state(box)["content"] == before[0]["content"]
    finally:
        subprocess.run(["git", "-C", str(WT), "worktree", "remove", "--force", str(box)],
                       check=True, capture_output=True)
        stray = subprocess.run(["pgrep", "-af", str(OUT / "tmp")], capture_output=True,
                               text=True).stdout.strip()
        after = tracked_state()
        report.update({"inventory_sha256_equal": before[0]["content"] == after[0]["content"],
                       "other_tracked_identity_and_stat_equal": before[0]["other"] == after[0]["other"],
                       "head_clean_before": not before[1], "head_clean_after": not after[1],
                       "inventory": sorted(before[0]["content"]), "harness_refused_opens": REFUSED,
                       "harness_children": CHILDREN[:400], "stray_processes": stray})
        (OUT / "report.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    detected = sum(1 for m in report["mutants"] if m["result"] == "detected")
    unchanged = before[0] == after[0] and before[1] == after[1] == []
    print(f"detected {detected}/{len(MUTANTS)}; inventory and checkout unchanged {unchanged}; "
          f"harness refused {len(REFUSED)}; stray {bool(stray)}")
    return 0 if detected == len(MUTANTS) and unchanged and not REFUSED and not stray else 1


if __name__ == "__main__":
    raise SystemExit(main())
