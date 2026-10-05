"""Mutation battery v13 (generation v9, stages R and T; audits 743-754), revision d.

Attempt 1 at bdaba58 detected 22/24 as declared: RX13 (a stage-S scope) was an always-refusing
mutant (a stage-S scope carries no F pin, so every line refuses) and is replaced by a run-scope line
approving a smoke, with a new test; RX16 was detected by its tests but for another reason than
declared (the [None] retry re-ran the chain and failed at record publication), so it now declares
the [0] case, where the overwritten reservation lets a retry run silently. Revision c (audits
746-754): RX8 now removes only the exact-acceptance digest, and RX25-RX33 disable the repairs
(the T entry claim, config byte pin, typed fields, single read, effective arguments, supervisor
children per cell, the T boundary rechecks and the R-to-T source equality). Revision d
(audit 756 and battery c): RX25/RX27 declare the assertions that first see their defect, and
RX34-RX38 disable the per-cell stage-R input checks at T boundaries, inside train extraction and
in the T entry, and the pins handed to the producers. Revision d at b68c1f6 detected 37/38: RX35
crash-killed in the train-extraction fixture (its stand-in legacy resume set no arguments). The
mutants are unchanged at revision 5; the fixture now sets the arguments as the real resume does.
Revision e (generation v9 r6, audits 759-760; at 586187d revision d detected 38/38): the paths name
the r6 worktree and manifest; RX35 declares the r6 train fixture's load record (the real weight
loader now runs, so the verified path is two buffer loads); RX39-RX49 disable the consumed-object
binding: weights and epoch from the verified objects in both consumers, the pass-through from the T
entry, the witness-to-checkpoint and terminal-epoch binding, the effective-runtime check and the
terminal-epoch pin. RX43 is caught by the layered effective-runtime check (its declared text is that
refusal) because the resolver mutant alone yields the forged epoch, which that check refuses.

v12's bounded and guarded method, unchanged except for its paths: the v9 worktree, the v9 manifest
(artifacts/anchor_confirmation/authority_manifest_v9r6.json) and the v9 copies of bounded_tree.py and
guarded_pytest.py under artifacts/anchor_confirmation/refit_v9/. Content hashes ONLY for the reviewed
inventory (the manifest closure plus the declared test files); every other tracked file by git index
object id and stat; the harness under its own open() guard (binary payloads, real-data roots and
os.exec refused); every declared test under guarded_pytest.py with zero refused opens; a sparse
sandbox without artifacts/. Each mutant replaces exactly one pattern (each found exactly once before
any run) and disables a whole condition or step; a mutant counts as detected only if every declared
test fails (rc 1) with its declared text in the output and no refused open.
"""
import hashlib, json, os, subprocess, sys, time
from pathlib import Path

WT = Path("/data/yschoi/gdna_anchor_refit_v9r6")
COMMIT, OUT = sys.argv[1], Path(sys.argv[2])
PY = "/home/yschoi/.conda/envs/dna_hashing/bin/python"
P0, TR, L, RT, TE, S, TOT = ("p0_protocol.py", "train_siglip2.py", "scripts/phase3_selection_matrix.py",
                             "scripts/anchor_refit_stage.py", "scripts/anchor_terminal_test.py",
                             "scripts/anchor_confirm_supervisor.py", "terminal_official_test.py")
T = "tests/test_anchor_refit_stage.py"
NO_RAISE = "DID NOT RAISE"
ENTRY = "test_composed_an_entry_without_complete_admission_refuses_before_any_load"
STARTUP = "test_composed_an_anchor_refit_without_its_stage_r_cell_refuses_before_the_run_directory"
OFFICIAL = "test_composed_the_official_extraction_consumes_the_admitted_weights_and_epoch"
REPLACED = "test_composed_a_t_input_replaced_after_verification_never_reaches_the_official_encoding"
TRAIN_CONSUMES = "test_composed_train_extraction_consumes_the_verified_weights_and_epoch"

MUTANTS = [
    ("RX1 a sealed stage-R cell is not withheld", P0,
     '            raise ValueError("an anchor stage-R cell must carry its sealed scientific recipe")\n'
     '        return True\n',
     '            raise ValueError("an anchor stage-R cell must carry its sealed scientific recipe")\n'
     '        return False\n',
     [((T, "test_predicate_a_sealed_anchor_stage_r_cell_withholds_the_official_test"),
       "assert P0.anchor_refit_withholds_official_test("),
      ((T, "test_composed_the_terminal_decision_withholds_a_stage_r_cell"),
       "assert T._official_test_allowed(decision_args(), p0_refit=True) is False")]),
    ("RX2 an anchor P0 refit outside a stage-R cell falls through", P0,
     '    if bool(p0_refit_active):\n        raise ValueError(\n            "an anchor-model P0 refit runs only',
     '    if False:\n        raise ValueError(\n            "an anchor-model P0 refit runs only',
     [((T, "test_predicate_an_anchor_refit_outside_a_sealed_stage_r_cell_refuses[marker-deleted]"), NO_RAISE),
      ((T, f"{STARTUP}[marker-altered]"), NO_RAISE),
      ((T, f"{STARTUP}[no-campaign]"), NO_RAISE)]),
    ("RX3 a stage-R cell need not be a P0 refit", P0,
     '        if not bool(p0_refit_active):\n            raise ValueError(\n'
     '                "an anchor stage-R cell must be a P0 refit',
     '        if False:\n            raise ValueError(\n'
     '                "an anchor stage-R cell must be a P0 refit',
     [((T, "test_predicate_an_anchor_refit_outside_a_sealed_stage_r_cell_refuses[not-refit]"), NO_RAISE)]),
    ("RX4 the trainer start-up does not apply the policy", TR,
     '    anchor_refit_withholds_official_test(\n        axis_center=getattr(args, "axis_center", "none"),\n'
     '        p0_refit_active=_is_p0_refit(',
     '    (lambda **k: None)(\n        axis_center=getattr(args, "axis_center", "none"),\n'
     '        p0_refit_active=_is_p0_refit(',
     [((T, f"{STARTUP}[marker-altered]"), NO_RAISE),
      ((T, f"{STARTUP}[no-campaign]"), NO_RAISE)]),
    ("RX5 the terminal decision ignores the stage-R policy", TR,
     '        allowed = False\n        print("[anchor-refit] stage R ends',
     '        pass\n        print("[anchor-refit] stage R ends',
     [((T, "test_composed_the_terminal_decision_withholds_a_stage_r_cell"),
       "assert T._official_test_allowed(decision_args(), p0_refit=True) is False")]),
    ("RX6 a stage-R cell runs the legacy post-chain", L,
     '    anchor_refit = stage == "refit" and anchor_arm is not None\n',
     '    anchor_refit = False\n',
     [((T, "test_composed_a_stage_r_cell_runs_no_post_chain_and_records_the_test_withheld"),
       'assert "postprocess" not in rcell.calls')]),
    ("RX7 a run directory holding test outputs passes as withheld", L,
     '    if present:\n        raise CellRefused(f"{run_dir}: a stage-R run holds',
     '    if False:\n        raise CellRefused(f"{run_dir}: a stage-R run holds',
     [((T, "test_composed_a_stage_r_run_directory_holding_any_test_output_refuses[evaluation_siglip2_base.json]"),
       NO_RAISE),
      ((T, "test_composed_an_output_already_present_refuses_before_the_attempt"), NO_RAISE)]),
    ("RX8 the F acceptance is substring recognition again", RT,
     '    if digest != ANCHOR_F_ACCEPTANCE_SHA256 or str(ANCHOR_F_RECORD) not in flat \\\n',
     '    if str(ANCHOR_F_RECORD) not in flat \\\n',
     [((T, "test_composed_an_f_record_without_its_acceptance_refuses[token-only]"), NO_RAISE),
      ((T, "test_composed_an_f_record_without_its_acceptance_refuses[changed-language]"), NO_RAISE)]),
    ("RX9 the F summary is not tied to its validated recipe", RT,
     '                or any(fields.get(k) != v for k, v in entry["lambdas"].items()):\n',
     '                or False:\n',
     [((T, "test_composed_a_corrupt_f_record_refuses[copied-lambda]"), NO_RAISE)]),
    ("RX10 a refit recipe may move any field", RT,
     '        if beyond:\n            raise CellRefused(f"{key}: the refit recipe differs',
     '        if False:\n            raise CellRefused(f"{key}: the refit recipe differs',
     [((T, "test_composed_a_refit_render_off_the_mapping_refuses[lambda]"), NO_RAISE)]),
    ("RX11 any seal authority stands in for the refit seal", RT,
     '            if off or authority.get("stage") != REFIT_STAGE:\n',
     '            if False:\n',
     [((T, "test_composed_a_stage_1_seal_cannot_stand_in_for_the_refit_seal"), NO_RAISE)]),
    ("RX12 stage R takes any input seal files", RT,
     '        if request["input_seals"] != want_seals:\n',
     '        if False:\n',
     [((T, "test_composed_stage_r_takes_only_the_approved_refit_seals"),
       "assert LT.run_main(monkeypatch, *base, *RUN, *extra) == 2")]),
    ("RX13 a stage-R smoke is approved by a run-scope line", RT,
     "        scope = f\"stage-R-{'smoke' if args.smoke else 'run'}\"\n",
     "        scope = \"stage-R-run\"\n",
     [((T, "test_composed_a_run_scope_line_does_not_approve_a_stage_r_smoke"),
       "assert LT.run_main(monkeypatch, *rmain.base, *SMOKE, *extra) == 2")]),
    ("RX14 stages R and T run under the v8 manifest", RT,
     '        if manifest["sha256"] in (M.ANCHOR_V7_MANIFEST_SHA256, ANCHOR_V8_MANIFEST_SHA256):\n',
     '        if False:\n',
     [((T, "test_composed_stages_r_and_t_never_run_under_an_older_generation[v8]"),
       'assert LT.run_main(monkeypatch, *rmain.base, *RUN, "--plan") == 2')]),
    ("RX15 the T admission accepts a stage-R cell whose test was not withheld", RT,
     '                or (completion.get("official_test") or {}).get("status") != "withheld" \\\n',
     '                or False \\\n',
     [((T, "test_composed_a_stage_r_cell_that_is_not_withheld_and_terminal_refuses[not-withheld]"), NO_RAISE)]),
    ("RX16 the attempt reservation is not exclusive", RT,
     '        M._publish_json_exclusive(path, payload)\n',
     '        M._atomic_json(path, payload)\n',
     [((T, "test_composed_an_attempted_cell_is_never_retried[0]"), NO_RAISE),
      ((T, "test_composed_concurrent_attempts_reserve_once"),
       'assert sorted(status for status, _ in results) == ["ok", "refused", "refused", "refused"]')]),
    ("RX17 the attempt is not reserved before the T entry runs", RT,
     '    attempt, attempt_sha = reserve_attempt(namespace, cell=cell, request=request, approval=approval,\n'
     '                                           campaign_nonce=campaign_nonce)\n'
     '    env = cell_input_env(terminal_test_env(gpu_uuid), cell)\n',
     '    attempt, attempt_sha = attempt_path(namespace, cell), "0" * 64\n'
     '    env = cell_input_env(terminal_test_env(gpu_uuid), cell)\n',
     [((T, "test_composed_a_t_cell_reserves_then_runs_the_entry_and_the_chain_in_order"),
       'assert tcell.state["attempt_seen"] == [True]')]),
    ("RX18 the T cell runs no post-chain", RT,
     '    M._run_refit_postprocess(run_dir, dataset=cell["dataset"], env=env, snapshot=snapshot,\n'
     '                             boundary_check=lambda: check_cell_inputs(cell))\n',
     '    pass\n',
     [((T, "test_composed_a_t_cell_reserves_then_runs_the_entry_and_the_chain_in_order"),
       "assert tcell.commands == CHAIN")]),
    ("RX19 the T entry does not need the ledger approval", TE,
     '        live = M.audit_approval((attempt.get("approval") or {}).get("section"), scope,\n',
     '        live = {"line": (attempt.get("approval") or {}).get("line")} or M.audit_approval('
     '(attempt.get("approval") or {}).get("section"), scope,\n',
     [((T, f"{ENTRY}[no-approval]"), "assert TE.main(entry.argv) == 2"),
      ((T, "test_composed_the_t_approval_cannot_come_from_the_caller_environment"),
       "assert TE.main(entry.argv) == 2 and entry.calls == []")]),
    ("RX20 the T entry does not hash the checkpoint", TE,
     '    need(_sha(checkpoint) == cell["final_checkpoint_sha256"], f"{checkpoint} is not the stage-R checkpoint")\n',
     '    need(True, f"{checkpoint} is not the stage-R checkpoint")\n',
     [((T, f"{ENTRY}[checkpoint]"), "assert TE.main(entry.argv) == 2")]),
    ("RX21 the T entry does not check the saved configuration", TE,
     '        check_config(args, cell, run_dir)\n',
     '        pass\n',
     [((T, "test_composed_arguments_built_off_the_verified_object_refuse_before_the_test[run-dir]"),
       "assert TE.main(entry.argv) == 2")]),
    ("RX22 the stage-R/T stages write the default ledger", S,
     '              (RT_STAGES, RT_OPS_ROOT, RT_BUDGET_DEVICE_SECONDS))\n',
     '              ((), RT_OPS_ROOT, RT_BUDGET_DEVICE_SECONDS))\n',
     [((T, "test_composed_stages_r_and_t_keep_their_own_ledger[stage-R-run]"),
       "assert S.stage_ledger(stage, None) == (S.RT_OPS_ROOT, S.RT_BUDGET_DEVICE_SECONDS)")]),
    ("RX23 the raw evaluation is replaced by the post-BIO one", TOT,
     '            dataset_name=getattr(args, "dataset", None),\n        )\n',
     '            dataset_name=getattr(args, "dataset", None), bio_project=True,\n        )\n',
     [((T, "test_composed_the_terminal_function_runs_extraction_then_the_raw_evaluator"),
       'assert calls[1][2] == {"distance_mode": "base"'),
      ((T, "test_structural_the_terminal_function_is_the_v8_block_moved_unchanged"), "assert new_block == old_block")]),
    ("RX24 an anchor refit tag loses its arm", L,
     '_refit_N{n}_s{seed}{anchor}"', '_refit_N{n}_s{seed}"',
     [((T, "test_composed_an_anchor_refit_command_is_the_legacy_refit_plus_the_arm"),
       'assert tag.endswith("_refit_N4_s43_AXanchors_P06095_JD002")')]),
    ("RX25 the T entry does not claim its entry", TE,
     '        claim_entry(Path(cli.attempt), cli.attempt_sha256, cell)\n',
     '        pass\n',
     [((T, "test_composed_the_same_attempt_never_enters_twice_after_a_before_output_failure"),
       "assert TE.main(entry.argv) == 2                          # the same attempt, unchanged bytes"),
      ((T, "test_composed_concurrent_entries_with_the_same_attempt_reach_the_test_once"),
       "assert sorted(codes) == [0, 2, 2, 2]")]),
    ("RX26 the saved configuration's typed fields are not compared", TE,
     '    need(not differing, f"the saved configuration differs from the sealed recipe in {differing[:8]}")\n',
     '    need(True, f"the saved configuration differs from the sealed recipe in {differing[:8]}")\n',
     [((T, "test_composed_a_saved_configuration_off_the_sealed_recipe_never_reaches_the_test[non-summary-drift]"),
       "assert TE.main(e.argv) == 2")]),
    ("RX27 config.pt is deserialized without its byte pin", TE,
     '    need(hashlib.sha256(raw).hexdigest() == cell["config_pt_sha256"], "config.pt is not the pinned bytes")\n',
     '    need(True, "config.pt is not the pinned bytes")\n',
     [((T, "test_composed_a_config_changed_after_admission_refuses_before_it_is_loaded"),
       "assert TE.main(entry.argv) == 2")]),
    ("RX28 the arguments come from a second read of config.pt", TE,
     '    _apply_saved_config(args, saved, str(run_dir))\n',
     '    __import__("extraction_siglip2")._resume_args_flat_or_legacy(args)\n',
     [((T, "test_composed_the_entry_reads_config_once_and_uses_that_object"),
       "the stage-T entry must not reopen config.pt through the resume helper")]),
    ("RX29 the effective arguments are not compared", TE,
     '    need(not effective, f"the effective arguments differ from the sealed recipe in {effective[:8]}")\n',
     '    need(True, f"the effective arguments differ from the sealed recipe in {effective[:8]}")\n',
     [((T, "test_composed_arguments_built_off_the_verified_object_refuse_before_the_test[lambda]"),
       "assert TE.main(entry.argv) == 2")]),
    ("RX30 the supervisor counts one managed child per T cell", S,
     '    planned_attempts = planned_cells * children_per_cell\n',
     '    planned_attempts = planned_cells\n',
     [((T, "test_composed_a_t_cell_s_five_producers_are_ordinary_supervised_work[1]"),
       'assert rc == 0 and done["status"] == "exited", done.get("reason")')]),
    ("RX31 no boundary check between the T entry and train extraction", RT,
     '    M.verify_snapshot(snapshot)\n    check_cell_inputs(cell)                              # the entry-to-train-extraction transition\n',
     '',
     [((T, "test_composed_drift_between_producers_stops_the_next_producer[entry-to-train]"),
       "assert tcell.commands == CHAIN[:k]")]),
    ("RX32 the post-chain runs without the T snapshot", RT,
     '    M._run_refit_postprocess(run_dir, dataset=cell["dataset"], env=env, snapshot=snapshot,\n',
     '    M._run_refit_postprocess(run_dir, dataset=cell["dataset"], env=env, snapshot=None,\n',
     [((T, "test_composed_drift_between_producers_stops_the_next_producer[after-train]"), NO_RAISE),
      ((T, "test_composed_real_verifier_drift_between_producers_stops_the_chain"), NO_RAISE)]),
    ("RX33 stage T does not require the stage-R sources unchanged", RT,
     '    if sources != r_snapshot.get("sources") or inputs != r_snapshot.get("inputs") \\\n            or {k: (v.get("head_sha256"), v.get("worktree_sha256")) for k, v in authority["entries"].items()} \\\n            != {k: (v.get("head_sha256"), v.get("worktree_sha256")) for k, v in before.items()}:\n',
     '    if False:\n',
     [((T, "test_composed_sources_changed_since_stage_r_refuse_before_any_t_access"), NO_RAISE)]),
    ("RX34 the consumed stage-R cell's inputs are not rechecked at T boundaries", RT,
     '    run_dir = Path(cell["run_dir"])\n    for name, pin in (("config.pt", cell["config_pt_sha256"]),\n',
     '    run_dir = Path(cell["run_dir"])\n    return\n    for name, pin in (("config.pt", cell["config_pt_sha256"]),\n',
     [((T, "test_composed_a_changed_r_artifact_stops_the_next_t_producer[anchor_terminal_test.py-witness]"),
       NO_RAISE),
      ((T, "test_composed_a_changed_r_artifact_refuses_before_the_t_entry[witness]"), NO_RAISE)]),
    ("RX35 train extraction ignores the stage-T pins", "scripts/extract_train_split.py",
     '    if not any(values.values()):\n',
     '    if True:\n',
     [((T, "test_composed_train_extraction_loads_the_verified_bytes_once"),
       'assert t.calls["resume"] == 0 and t.calls["loads"] == ["buffer", "buffer"]')]),
    ("RX36 train extraction writes without rechecking its consumed inputs", "scripts/extract_train_split.py",
     '        recheck_stage_t_inputs(ckpt, pins)\n',
     '        pass\n',
     [((T, "test_composed_train_extraction_writes_nothing_after_an_input_changed_during_it[config]"), NO_RAISE)]),
    ("RX37 the T entry does not recheck the consumed files before the test", TE,
     '        check_consumed(run_dir, cell)            # immediately before extract_code reads them (audit 756)\n',
     '        pass\n',
     [((T, "test_composed_a_config_left_changed_after_the_read_refuses_before_the_test"),
       "assert TE.main(entry.argv) == 2")]),
    ("RX38 the producers are not given the consumed cell's pins", RT,
     '    return dict(env, GDNA_T_EXPECT_CONFIG_SHA256=cell["config_pt_sha256"],\n',
     '    return dict(env) or dict(env, GDNA_T_EXPECT_CONFIG_SHA256=cell["config_pt_sha256"],\n',
     [((T, "test_composed_the_producers_receive_the_consumed_cell_s_pins"), "assert len(seen) == 5 and all(")]),
    ("RX39 the official extraction loads the weights by path despite a binding", "extraction_siglip2.py",
     '                model, ckpt_path if verified is None else io.BytesIO(verified.checkpoint_bytes),\n',
     '                model, ckpt_path,\n',
     [((T, f"{REPLACED}[changed-read-restored-checkpoint]"), 'assert o.calls["dataset_at"] == [(1.0, epoch)]'),
      ((T, OFFICIAL), 'assert o.calls["loads"] == ["buffer", "buffer"]')]),
    ("RX40 the official extraction resolves the epoch from the files despite a binding", "extraction_siglip2.py",
     '            _resolved = apply_inference_epoch(model, ckpt_path, args, verified=verified)\n',
     '            _resolved = apply_inference_epoch(model, ckpt_path, args)\n',
     [((T, f"{REPLACED}[changed-read-restored-witness]"), 'assert o.calls["dataset_at"] == [(1.0, epoch)]')]),
    ("RX41 the T entry hands no binding to the official test", TE,
     '                      codebook_size=int(getattr(args, "codebook_size", 32)), verified=runtime)\n',
     '                      codebook_size=int(getattr(args, "codebook_size", 32)))\n',
     [((T, OFFICIAL), 'assert o.calls["loads"] == ["buffer", "buffer"]')]),
    ("RX42 the terminal function drops the binding", TOT,
     '        if verified is None:\n            _extract_code(args)\n        else:\n            _extract_code(args, verified=verified)\n',
     '        _extract_code(args)\n',
     [((T, OFFICIAL), 'assert o.calls["loads"] == ["buffer", "buffer"]')]),
    ("RX43 the resolver ignores the binding and reopens the witness", "dna_utils/runtime_state.py",
     '    if verified is None:\n        md = CheckpointMetadata.load(checkpoint_path)\n',
     '    if True:\n        md = CheckpointMetadata.load(checkpoint_path)\n',
     [((T, f"{REPLACED}[changed-read-restored-witness]"), "is not the admitted terminal epoch"),
      ((T, f"{TRAIN_CONSUMES}[changed-read-restored-witness]"), "is not the admitted terminal epoch")]),
    ("RX44 the effective runtime is not checked against the admitted one", "dna_utils/runtime_state.py",
     '    if verified is not None and (resolved.epoch != verified.terminal_epoch\n'
     '                                 or resolved.checkpoint_sha256 != verified.checkpoint_sha256):\n',
     '    if False:\n',
     [((T, "test_predicate_the_effective_runtime_must_be_the_admitted_one[epoch]"), NO_RAISE)]),
    ("RX45 the verified witness is not bound to the verified checkpoint", "dna_utils/runtime_state.py",
     '    if md.checkpoint_sha256 != checkpoint_sha256:\n'
     '        raise RuntimeBindingRefused("the admitted runtime witness describes another checkpoint")\n',
     '    if False:\n'
     '        raise RuntimeBindingRefused("the admitted runtime witness describes another checkpoint")\n',
     [((T, "test_predicate_the_verified_runtime_binds_both_files_and_the_terminal_epoch[other-checkpoint]"),
       NO_RAISE)]),
    ("RX46 the verified witness need not be at the admitted terminal epoch", "dna_utils/runtime_state.py",
     '    if md.checkpoint_epoch_zero_based != int(terminal_epoch):\n',
     '    if False:\n',
     [((T, "test_predicate_the_verified_runtime_binds_both_files_and_the_terminal_epoch[other-epoch]"), NO_RAISE)]),
    ("RX47 train extraction resolves the epoch from the files despite its binding", "scripts/extract_train_split.py",
     '        _resolved = apply_inference_epoch(model, ckpt, args, verified=runtime)\n',
     '        _resolved = apply_inference_epoch(model, ckpt, args)\n',
     [((T, f"{TRAIN_CONSUMES}[changed-read-restored-witness]"), 'assert t.calls["dataset_at"] == [(1.0, epoch)]')]),
    ("RX48 the producers are not given the admitted terminal epoch", RT,
     '                GDNA_T_EXPECT_TERMINAL_EPOCH=str(cell["terminal_epoch"]))\n',
     '                )\n',
     [((T, "test_composed_the_producers_receive_the_consumed_cell_s_pins"), "assert len(seen) == 5 and all(")]),
    ("RX49 the T entry does not bind the checkpoint and witness", TE,
     '        runtime = verified_runtime(run_dir, cell)\n',
     '        runtime = None\n',
     [((T, OFFICIAL), 'assert o.calls["loads"] == ["buffer", "buffer"]')]),
]



GUARD = WT / "artifacts/anchor_confirmation/refit_v9/guarded_pytest.py"
BOUNDED = WT / "artifacts/anchor_confirmation/refit_v9/bounded_tree.py"
MANIFEST = WT / "artifacts/anchor_confirmation/authority_manifest_v9r6.json"
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
