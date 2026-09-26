"""Anchor confirmation v2 mutation battery, third revision (audit 673.3, 688.1).

Each mutant breaks ONE guard in a sandbox COPY of the anchor worktree's tracked files and runs only
the tests declared to catch it. A mutant counts as detected only when those tests fail (pytest rc 1)
and the output carries the failure mode declared in advance (`expect`); a collection/import error,
another rc or a timeout is NOT detection. An unmutated baseline of every test group must pass first.
The real worktree's tracked bytes and git status are compared before and after.

Compound conditions are disabled as a WHOLE (`True or ...`, `if False:`) or one predicate at a time,
never by a text prefix that leaves a later `or` operand live (688.1). Boundary mutants are caught
by the tests' independent deny guards (no process, no deserialisation, no model).

usage: python anchor_mutation_v3.py <new sandbox dir> <report.json>
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
MUTANTS = [
    # --- recipe helper: override policy and trainer checks (669, 670, 677.2) ---
    ("R1 unreviewed repeat refused", HELPER,
     '        _need(dest in REVIEWED_OVERRIDES,\n              f"repeated options outside the reviewed wrapper overrides: {dest} {spans}")',
     '        if dest not in REVIEWED_OVERRIDES:\n            continue',
     [f"{REC}::test_an_unreviewed_or_conflicting_repeat_cannot_be_sealed"], RAISE, ""),
    ("R2 exactly two occurrences", HELPER,
     '        _need(len(spans) == 2, f"{dest} is given', '        _need(True, f"{dest} is given',
     [f"{REC}::test_any_other_override_sequence_refuses",
      f"{LAU}::test_a_third_occurrence_injected_into_the_actual_composition_refuses"], RAISE, ""),
    ("R3 first occurrence is the wrapper literal", HELPER,
     '        _need(spans[0] == REVIEWED_OVERRIDES[dest],', '        _need(True,',
     [f"{REC}::test_any_other_override_sequence_refuses"], RAISE, ""),
    ("R4 axis_center never repeated (named)", HELPER,
     '    _need(len(occurrences.get("axis_center", [])) <= 1, "axis_center is given more than once")',
     '    pass', [f"{REC}::test_an_unreviewed_or_conflicting_repeat_cannot_be_sealed"],
     r"Regex pattern did not match", DEPTH + " (the general repeat guard names axis_center)"),
    ("R5 value token captured in a span", HELPER,
     '            span = (token, argv[i + 1])', '            span = (token,)',
     [f"{REC}::test_the_wrapper_literal_then_one_override_is_admitted_for_every_reviewed_destination"],
     r"not the reviewed wrapper literal", "positive control refused"),
    ("R6 literals are the pinned wrappers'", HELPER,
     '    "routing_adaptive_topp_min": ("--routing_adaptive_topp_min", "0.3"),',
     '    "routing_adaptive_topp_min": ("--routing_adaptive_topp_min", "0.4"),',
     [f"{LAU}::test_the_actual_wrapper_composition_repeats_exactly_the_reviewed_overrides"],
     r"not the reviewed wrapper literal", "actual-composition positive control refused"),
    ("R7 runtime field presence", HELPER,
     '        _need(hasattr(post_processed, dest), f"{dest}: absent from the post-processed arguments")',
     '        if not hasattr(post_processed, dest):\n            continue',
     [f"{REC}::test_a_post_processed_field_that_is_absent_refuses"], RAISE, ""),
    ("R8 runtime typed value", HELPER,
     '        _need(canonical(_typed(dest, _normalise(dest, post))) == canonical(value),', '        _need(True,',
     [f"{REC}::test_a_post_processed_value_of_another_type_refuses"], RAISE, ""),
    ("R9 type-sensitive field comparison", HELPER,
     '                  if (k in a) != (k in b) or canonical(a.get(k)) != canonical(b.get(k)))',
     '                  if (k in a) != (k in b) or a.get(k) != b.get(k))',
     [f"{REC}::test_a_bool_saved_for_a_float_field_is_a_difference"], r"AssertionError|assert \[\]", ""),
    ("R10 anchor cell must carry its recipe", HELPER,
     '        _need(planned_arm is None,\n              "an anchor-confirmation cell must carry its sealed scientific recipe")',
     '        pass', [f"{REC}::test_an_anchor_cell_without_its_sealed_recipe_refuses"], RAISE,
     "the control-arm case is accepted; the anchors case is still refused by the axis guard"),
    ("R11 a sealed recipe only in an anchor cell", HELPER,
     '    _need(campaign_binding is not None and planned_arm is not None,', '    _need(True,',
     [f"{REC}::test_a_sealed_recipe_outside_an_anchor_cell_refuses"], RAISE, ""),
    ("R12 rebuilt digest equals the sealed digest", HELPER,
     '    _need(digest(actual) == sealed_digest, "the rebuilt recipe digest is not the sealed digest")',
     '    pass', [REC], None,
     "expected SURVIVOR, unreachable by design: exact shape + schema + argv equality + type-sensitive "
     "field equality already imply digest equality"),
    ("R13 tokenizer normalisation", HELPER,
     '    if dest == "clip_snapshot_tokenizers_sha256_json" and value not in (None, ""):', '    if False:',
     [f"{REC}::test_sealed_recipe_with_the_real_quoted_tokenizer_path_is_accepted"],
     r"clip_snapshot_tokenizers_sha256_json", "positive control refused"),
    ("T1 trainer calls the recipe check", "train_siglip2.py",
     '    recipe_binding = verify_trainer_recipe(Config.build_parser(), _sys.argv[1:], args,\n                                           campaign_binding)',
     '    recipe_binding = None',
     [f"{REC}::test_resolve_save_path_refuses_before_any_directory_or_claim",
      f"{REC}::test_resolve_save_path_carries_the_recipe_digest_into_the_campaign_binding"], RAISE, ""),
    # --- launcher: admission, protocol, manifest, approval, request (671, 677, 679, 683, 688, 689) ---
    ("L1 protocol values checked", LAUNCHER,
     '            wrong = sorted(k for k, v in want.items()\n                           if k not in payload["fields"] or canonical(payload["fields"][k]) != canonical(v))',
     '            wrong = []',
     [f"{LAU}::test_a_protocol_value_off_the_contract_refuses_in_every_mode",
      f"{LAU}::test_both_arms_off_the_protocol_refuse_although_they_differ_in_the_axis_alone"],
     rf"{RC_PLAN}|{RAISE}", ""),
    ("L2 cell carries the approved recipe", LAUNCHER,
     '        if (tuple(topp), str(joint)) != (tuple(incumbent[ds]["topp"]), str(incumbent[ds]["joint"])):',
     '        if False:', [f"{LAU}::test_a_cell_off_the_approved_recipe_refuses"],
     r"Regex pattern did not match", DEPTH + " (the protocol check names the top-p)"),
    ("L3 unsupervised invariant in the protocol", LAUNCHER,
     '        "hash_target_mode": "siglip_cos", "disable_text_supervision": False,',
     '        "hash_target_mode": "jaccard", "disable_text_supervision": False,',
     [f"{LAU}::test_the_actual_wrapper_composition_repeats_exactly_the_reviewed_overrides",
      f"{LAU}::test_protocol_fields_fix_the_stage_1_horizons_and_the_smoke_shortens_them_alone"],
     r"jaccard|siglip_cos", "positive controls refuse"),
    ("L4 arms differ in axis_center alone", LAUNCHER,
     '        if differing != ["axis_center"]:\n            raise CellRefused(f"{key}: the arms differ',
     '        if False:\n            raise CellRefused(f"{key}: the arms differ',
     [f"{LAU}::test_an_invalid_scientific_request_refuses_in_every_mode_before_any_boundary"], RC_PLAN, ""),
    ("L5 execution needs the manifest", LAUNCHER,
     '        if execute and manifest is None:', '        if False:',
     [f"{LAU}::test_execution_without_the_generation_manifest_refuses"],
     r"assert 'need --anchor-manifest' in", DEPTH + " (the approval needs the manifest digest)"),
    ("L6 input seals verified before any lease", LAUNCHER,
     '            input_seals = verify_campaign_input_seals(\n                args.input_seal_specs, cells, full=admission_is_full(carried), expected=carried)',
     '            input_seals = None',
     [f"{LAU}::test_an_approved_execution_verifies_its_inputs_then_reaches_only_the_lease",
      f"{LAU}::test_an_input_seal_that_does_not_verify_refuses_before_any_lease"],
     rf"{RC_PLAN}|assert \['_with_campaign_gpu_leases'\]", ""),
    ("L7 manifest: tree is the generation", LAUNCHER,
     '    if drifted:\n        raise CellRefused(f"the tree is not', '    if False:\n        raise CellRefused(f"the tree is not',
     [f"{LAU}::test_a_manifest_that_is_not_this_generation_refuses",
      f"{LAU}::test_a_manifest_of_another_tree_refuses_in_every_mode"], rf"{RAISE}|{RC_PLAN}", ""),
    ("L8 manifest: wrapper pins", LAUNCHER,
     '    if generation.get("dataset_scripts_sha256") != DATASET_SCRIPT_SHA256:', '    if False:',
     [f"{LAU}::test_a_manifest_that_is_not_this_generation_refuses"], RAISE, ""),
    ("L9a manifest: approved-aggregate pin (one predicate)", LAUNCHER,
     '    if (historical.get("approved_p3_refit_aggregate") or {}).get("sha256") != APPROVED_P3_REFIT_AGGREGATE_SHA256:',
     '    if False:', [f"{LAU}::test_a_manifest_that_is_not_this_generation_refuses",
                      f"{LAU}::test_both_historical_pins_wrong_refuses"], RAISE_OR_DEPTH, ""),
    ("L9b manifest: selected-N pin (one predicate)", LAUNCHER,
     '    if (historical.get("approved_selected_n") or {}).get("sha256") != APPROVED_SELECTED_N_SHA256:',
     '    if False:', [f"{LAU}::test_a_manifest_that_is_not_this_generation_refuses"], RAISE, ""),
    ("L10 manifest: designated contract", LAUNCHER,
     '    if contract.get("path") != ANCHOR_CONTRACT_PATH or contract.get("sha256") != files[ANCHOR_CONTRACT_PATH]:',
     '    if False:', [f"{LAU}::test_a_manifest_that_is_not_this_generation_refuses",
                      f"{LAU}::test_a_substituted_contract_at_its_correct_hash_refuses"], RAISE, ""),
    ("L11 manifest: exact closure", LAUNCHER,
     '    if missing or extra:', '    if False:',
     [f"{LAU}::test_a_manifest_that_is_not_this_generation_refuses"], RAISE_OR_DEPTH,
     "a missing member is accepted; an extra existing file still fails the drift check"),
    ("L12 manifest: commit, branch, clean (whole condition)", LAUNCHER,
     '    if not (isinstance(generation.get("commit"), str)', '    if False and not (isinstance(generation.get("commit"), str)',
     [f"{LAU}::test_a_manifest_that_is_not_this_generation_refuses"], RAISE, ""),
    ("L13 manifest: environment", LAUNCHER,
     '    if stated != here:', '    if False:',
     [f"{LAU}::test_a_manifest_that_is_not_this_generation_refuses"], RAISE, ""),
    ("L14 decide: stage S under this generation", LAUNCHER,
     '            if manifest is not None and selection.get("anchor_manifest_sha256") != manifest["sha256"]:',
     '            if False:', [f"{LAU}::test_decide_refuses_a_stage_s_from_another_generation"], RC_PLAN, ""),
    ("L15 render only pinned wrapper bytes", LAUNCHER,
     '    if want is None or hashlib.sha256((REPO / rel).read_bytes()).hexdigest() != want:', '    if False:',
     [f"{LAU}::test_render_refuses_a_script_that_is_not_the_pinned_bytes"], r"a process was started",
     "the independent deny guard stops the wrapper process"),
    ("L16 missing whitening refuses before the wrapper runs", LAUNCHER,
     '    if not whiten or not Path(whiten).is_file():', '    if False:',
     [f"{LAU}::test_render_refuses_a_missing_whitening_file_instead_of_letting_the_script_build_it"],
     r"a process was started", "the independent deny guard stops the wrapper: no real builder can run"),
    ("L17 no anchor refit in this generation", LAUNCHER,
     '        if stage == "refit":\n            raise CellRefused("anchor confirmation runs stage-1',
     '        if False:\n            raise CellRefused("anchor confirmation runs stage-1',
     [f"{LAU}::test_the_refit_path_refuses_an_anchor_arm"], RAISE, ""),
    ("L18 completion: saved config is the sealed recipe", LAUNCHER,
     '    if differing:\n        raise CellRefused(f"{run_dir}: saved config.pt differs',
     '    if False:\n        raise CellRefused(f"{run_dir}: saved config.pt differs',
     [f"{LAU}::test_completion_refuses_what_is_not_the_sealed_recipe"], RAISE, ""),
    ("L19 run_cell needs the sealed recipe", LAUNCHER,
     '    if anchor_arm is not None and (campaign_binding is None\n                                   or not isinstance(scientific_recipe, dict)):',
     '    if False:', [f"{LAU}::test_run_cell_refuses_an_anchor_cell_without_a_sealed_recipe"],
     r"a trainer was launched", "reaches the (sentinel) trainer launch"),
    ("L20 the tag carries the arm", LAUNCHER,
     '    anchor = f"_AX{anchor_arm}" if anchor_arm is not None else ""', '    anchor = ""',
     [f"{LAU}::test_arms_differ_in_the_axis_flag_and_tag_alone"], r"assert", ""),
    ("L21 execution needs the approval", LAUNCHER,
     '            approval = audit_approval(args.anchor_approval_section, scope, **pins)', '            approval = None',
     [f"{LAU}::test_execution_without_an_approval_refuses_before_inputs_and_leases",
      f"{LAU}::test_a_changed_request_is_not_covered_by_the_old_approval"], RC_PLAN, ""),
    ("L22a request binds the namespace", LAUNCHER,
     '        "manifest": manifest_sha256, "selection": selection_sha256,\n        "namespace": str(args.namespace),',
     '        "manifest": manifest_sha256, "selection": selection_sha256,\n        "namespace": None,',
     [f"{LAU}::test_a_changed_request_is_not_covered_by_the_old_approval"], r"assert 0 == 2", ""),
    ("L22b request binds the result root", LAUNCHER,
     '"result_root": str(Path(args.result_root).resolve()),', '"result_root": None,',
     [f"{LAU}::test_a_changed_request_is_not_covered_by_the_old_approval"], r"assert 0 == 2", ""),
    ("L22c request binds the cells", LAUNCHER,
     '        "declared_cells": sorted([c[0], c[7], c[1], c[5]] for c in cells),\n        "cells": sorted([c[0], c[7], c[1], c[5]] for c in executed),',
     '        "declared_cells": None,\n        "cells": None,',
     [f"{LAU}::test_a_changed_request_is_not_covered_by_the_old_approval",
      f"{LAU}::test_a_changed_smoke_is_not_covered_by_the_old_approval"], r"assert 0 == 2", ""),
    ("L22d request binds the smoke horizon", LAUNCHER,
     '        "epochs": int(args.epochs) if args.smoke else None,', '        "epochs": None,',
     [f"{LAU}::test_a_changed_smoke_is_not_covered_by_the_old_approval"], r"assert 0 == 2", ""),
    ("L22e request binds the input seals", LAUNCHER,
     '        "input_seals": {f"{ds}:{stage}": _file_pin(path)', '        "input_seals": {} and {f"{ds}:{stage}": _file_pin(path)',
     [f"{LAU}::test_a_changed_request_is_not_covered_by_the_old_approval"], r"assert 0 == 2", ""),
    ("L22f request binds the admission authority", LAUNCHER,
     '        "admission_authority": (_file_pin(args.admission_authority)', '        "admission_authority": None and (_file_pin(args.admission_authority)',
     [f"{LAU}::test_a_changed_request_is_not_covered_by_the_old_approval"], r"assert 0 == 2", ""),
    ("L22g request binds the GPU count", LAUNCHER,
     '        "gpu_count": len(gpus),', '        "gpu_count": None,',
     [f"{LAU}::test_a_changed_request_is_not_covered_by_the_old_approval"], r"assert 0 == 2", ""),
    # --- reducer (672, 679, 680, 689, 690) ---
    ("D1 reuse only through source approval", REDUCER,
     '    need(approver is not None,', '    need(True,',
     [f"{RED}::test_a_self_pinned_reuse_admission_is_not_approval"], RC_REDUCE, ""),
    ("D2 one generation for the chain", REDUCER,
     '    need(named == [manifest_sha256], f"the campaigns', '    need(True, f"the campaigns',
     [f"{RED}::test_a_campaign_under_another_generation_refuses"], RC_REDUCE, ""),
    ("D3 campaign names a generation", REDUCER,
     '        need(is_sha256(generation.get("sha256")),', '        need(True,',
     [f"{RED}::test_a_campaign_that_names_no_generation_refuses"],
     r"assert 'names no generation manifest' in", DEPTH + " (one_generation refuses the missing digest)"),
    ("D4 decide: stage S under the given generation", REDUCER,
     '    need(selection["anchor_manifest_sha256"] == manifest["sha256"],', '    need(True,',
     [f"{RED}::test_decide_refuses_stage_s_from_another_generation"],
     r"assert 'stage S ran under another generation manifest' in",
     DEPTH + " (the stage-S records' approvals are re-verified for the decide generation)"),
    ("D5 type-exact coordinate labels", REDUCER,
     '    need(same(record.get("dataset"), ds) and same(record.get("N"), n) and same(record.get("seed"), seed),',
     '    need(record.get("dataset") == ds and record.get("N") == n and record.get("seed") == seed,',
     [f"{RED}::test_a_record_that_is_not_train_only_terminal_evidence_refuses"], RC_REDUCE, ""),
    ("D6 type-exact split", REDUCER,
     '    need(same(record.get("val_split_ratio"), SPLIT["val_split_ratio"])\n         and same(record.get("val_split_seed"), SPLIT["val_split_seed"]),',
     '    need(record.get("val_split_ratio") == SPLIT["val_split_ratio"]\n         and record.get("val_split_seed") == SPLIT["val_split_seed"],',
     [f"{RED}::test_a_record_that_is_not_train_only_terminal_evidence_refuses"], RC_REDUCE, ""),
    ("D7 old refit only conditional", REDUCER,
     '        elif n_adopted == approved:', '        elif False:',
     [f"{RED}::test_a_kept_control_at_the_approved_n_names_the_old_refit_only_as_conditional"],
     r"AssertionError|assert", ""),
    ("D8 selection replay", REDUCER,
     '    need(replay["n_selected"] == frozen["n_selected"] and replay["scores"] == frozen["scores"],',
     '    need(True,', [f"{RED}::test_a_frozen_n_record_that_its_evidence_does_not_reproduce_refuses"],
     r"assert 'not what its own evidence reduces to' in",
     DEPTH + " (the evidence membership no longer matches the tampered N)"),
    ("D9 probe counts", REDUCER,
     '    need(n_images == PROBE_IMAGES and total == 4 * n_images and 0 <= hits <= total,', '    need(True,',
     [f"{RED}::test_a_probe_that_is_not_this_records_contract_measurement_refuses"], RC_REDUCE, ""),
    ("D10 probe ratio is hits/total", REDUCER,
     '    need(type(ratio) is float and ratio == hits / total,', '    need(True,',
     [f"{RED}::test_a_probe_that_is_not_this_records_contract_measurement_refuses"], RC_REDUCE, ""),
    ("D10b probe ratio is a float (one predicate)", REDUCER,
     '    need(type(ratio) is float and ratio == hits / total,', '    need(ratio == hits / total,',
     [f"{RED}::test_a_probe_that_is_not_this_records_contract_measurement_refuses"], RC_REDUCE, ""),
    ("D11 one row population per dataset", REDUCER,
     '        need(len({probes[c]["rows"] for c in coords if c[0] == ds}) == 1,', '        need(True,',
     [f"{RED}::test_probes_must_share_one_row_population_per_dataset"], RC_REDUCE, ""),
    ("D12 probe producer", REDUCER,
     '    need(probe.get("producer_sha256") == hashlib.sha256(consumed.read(PROBE_SOURCE)).hexdigest(),',
     '    need(True,', [f"{RED}::test_a_probe_that_is_not_this_records_contract_measurement_refuses"], RC_REDUCE, ""),
    ("D13 probe measured this record (whole condition)", REDUCER,
     '    need(probe.get("record_sha256") == admitted["entry"]["record_sha256"]',
     '    need(True or probe.get("record_sha256") == admitted["entry"]["record_sha256"]',
     [f"{RED}::test_a_probe_that_is_not_this_records_contract_measurement_refuses"], RC_REDUCE, ""),
    ("D14 retrieval equality passes", REDUCER,
     '        retrieval_ok = mean["anchors"] >= mean["none"] - sd_control',
     '        retrieval_ok = mean["anchors"] > mean["none"] - sd_control',
     [f"{RED}::test_retrieval_exactly_at_the_margin_passes"], r"assert", ""),
    ("D15 ties to the smallest N", REDUCER,
     '    return min(n for n, value in scores.items() if value == best)',
     '    return max(n for n, value in scores.items() if value == best)',
     [f"{RED}::test_ties_go_to_the_smallest_n"], r"assert", ""),
    ("D16 reverify before the write", REDUCER,
     '    changed = consumed.reverify()', '    changed = []',
     [f"{RED}::test_an_input_changed_after_it_was_read_refuses_the_write"], RC_REDUCE, ""),
    ("D17 write once", REDUCER,
     '    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)',
     '    fd = os.open(path, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o644)',
     [f"{RED}::test_select_reduces_new_evidence_to_the_argmax_n_and_writes_once"], r"assert 0 == 1", ""),
    ("D18 receipt lists the record", REDUCER,
     "        need(isinstance(cell, dict), f\"{entry['receipt']}: no completed cell {cell_id}\")",
     "        need(True, f\"{entry['receipt']}: no completed cell {cell_id}\")",
     [f"{RED}::test_a_record_the_receipt_does_not_list_refuses"],
     r"assert 'no completed cell' in", DEPTH + " (the record/receipt listing check refuses)"),
    ("D19 receipt campaign binding (whole condition)", REDUCER,
     '        need(isinstance(campaign, dict) and campaign.get("cell_id") == cell_id',
     '        need(True or isinstance(campaign, dict) and campaign.get("cell_id") == cell_id',
     [f"{RED}::test_an_empty_campaign_binding_on_both_sides_refuses"], RC_REDUCE, ""),
    ("D20 receipt completion pins", REDUCER,
     '        need(isinstance(pins, dict) and all(is_sha256(pins.get(k)) for k in RECEIPT_COMPLETION_KEYS),',
     '        need(True,', [f"{RED}::test_a_receipt_cell_without_its_campaign_proof_refuses"], RC_REDUCE, ""),
    ("D21 snapshot of this campaign", REDUCER,
     '        need(snapshot["plan"].get("campaign_nonce") == nonce,', '        need(True,',
     [f"{RED}::test_a_snapshot_of_another_campaign_refuses"], RC_REDUCE, ""),
    ("D22 approval present", REDUCER,
     '    need(isinstance(recorded, dict), "the campaign names no audit approval")',
     '    need(True, "the campaign names no audit approval")',
     [f"{RED}::test_a_campaign_without_an_approval_refuses"], r"assert 'names no audit approval' in",
     DEPTH + " (the scope check refuses the missing approval)"),
    ("D23 approval scope", REDUCER,
     '    need(recorded.get("scope") == scope,', '    need(True,',
     [f"{RED}::test_a_smoke_approval_is_not_a_run_approval"], r"assert \"ran under approval scope",
     DEPTH + " (the ledger has no line for the run scope)"),
    ("D24 approval re-verified in the ledger", REDUCER,
     '    live = M.audit_approval(recorded.get("section"), scope, **pins)',
     '    live = {"section": recorded.get("section"), "line": recorded.get("line")}',
     [f"{RED}::test_a_withdrawn_approval_line_refuses", f"{RED}::test_an_approval_changed_after_the_run_refuses",
      f"{RED}::test_a_request_edited_after_its_approval_refuses"], RC_REDUCE, ""),
    ("D25 recorded line is the ledger's", REDUCER,
     '    need(live["line"] == recorded.get("line"), "the approval line the campaign recorded is not the ledger\'s")',
     '    need(True, "")', [f"{RED}::test_a_recorded_approval_line_that_is_not_the_ledgers_refuses"], RC_REDUCE, ""),
    ("D26 record inside its approved request (whole condition)", REDUCER,
     '    need(request.get("schema") == M.REQUEST_SCHEMA and request.get("stage") == stage',
     '    need(True or request.get("schema") == M.REQUEST_SCHEMA and request.get("stage") == stage',
     [f"{RED}::test_a_record_outside_its_approved_request_refuses",
      f"{RED}::test_a_receipt_of_another_namespace_than_its_request_refuses"], RC_REDUCE, ""),
    ("D27 request present", REDUCER,
     '    need(isinstance(request, dict), "the campaign names no execution request")',
     '    need(True, "the campaign names no execution request")',
     [f"{RED}::test_a_campaign_without_its_request_refuses"], r"assert 'names no execution request' in",
     DEPTH + " (the approval does not match the digest of a missing request)"),
    ("D28 sealed recipe shape", REDUCER,
     '            check_payload_shape(payload)', '            pass',
     [f"{RED}::test_a_sealed_recipe_off_the_contract_refuses"], RC_REDUCE, ""),
    ("D29 sealed recipe declared digest", REDUCER,
     '        need(recipe_digest(payload) == binding.get("expected_scientific_recipe_sha256"),', '        need(True,',
     [f"{RED}::test_a_sealed_recipe_off_its_declared_digest_refuses"], r"assert 'not its declared digest' in",
     DEPTH + " (the trainer evidence names the sealed digest)"),
    ("D30 sealed recipe protocol values", REDUCER,
     '        need(not wrong, f"{cell_id}: the sealed recipe is not the contract\'s', '        need(True, f"{cell_id}: the sealed recipe is not the contract\'s',
     [f"{RED}::test_a_sealed_recipe_off_the_contract_refuses"], RC_REDUCE, ""),
    ("D31 completed recipe check in the record (whole condition)", REDUCER,
     '        need(anchor.get("version") == M.ANCHOR_CONFIRM_VERSION and anchor.get("arm") == arm',
     '        need(True or anchor.get("version") == M.ANCHOR_CONFIRM_VERSION and anchor.get("arm") == arm',
     [f"{RED}::test_a_record_without_its_completed_recipe_check_refuses"], RC_REDUCE, ""),
    ("D32 config bytes are their pin", REDUCER,
     '    consumed.read(admitted["run_dir"] / "config.pt", want)', '    consumed.read(admitted["run_dir"] / "config.pt")',
     [f"{RED}::test_config_bytes_off_their_pin_refuse_without_deserialising"], RC_REDUCE, ""),
    ("D33 probe coordinate (whole condition)", REDUCER,
     '    need(isinstance(stated, list) and len(stated) == 4', '    need(True or isinstance(stated, list) and len(stated) == 4',
     [f"{RED}::test_a_probe_coordinate_of_another_type_refuses"], RC_REDUCE, ""),
    ("D34 probe execution authority", REDUCER,
     '    need(probe.get("authority") == admitted["authority"],', '    need(True,',
     [f"{RED}::test_a_probe_that_is_not_this_records_contract_measurement_refuses"], RC_REDUCE, ""),
    ("D35 probe generation", REDUCER,
     '    need(probe.get("manifest_sha256") == manifest_sha256, f"{name}: produced under another generation")',
     '    need(True, "")', [f"{RED}::test_a_probe_that_is_not_this_records_contract_measurement_refuses"], RC_REDUCE, ""),
    ("D36 probe approval re-verified", REDUCER,
     '    live = M.audit_approval(approval.get("section"), "probe", manifest=manifest_sha256,\n                            selection=selection_sha256, request=request_sha256)',
     '    live = {"line": approval.get("line")}',
     [f"{RED}::test_a_probe_that_is_not_this_records_contract_measurement_refuses",
      f"{RED}::test_probes_approved_for_other_records_refuse"], RC_REDUCE, ""),
    ("D37 probe split with exact types (one predicate)", REDUCER,
     '         and all(same(split[k], v) for k, v in SPLIT.items())', '         and all(split[k] == v for k, v in SPLIT.items())',
     [f"{RED}::test_a_probe_that_is_not_this_records_contract_measurement_refuses"], RC_REDUCE, ""),
    ("D38 probe split identity (whole condition)", REDUCER,
     '    need(is_sha256(probe.get("split_identity_sha256"))', '    need(True or is_sha256(probe.get("split_identity_sha256"))',
     [f"{RED}::test_a_consistently_other_split_identity_refuses"], RC_REDUCE, ""),
    ("D39 probe caption target (whole condition)", REDUCER,
     '    need(isinstance(caption, dict)', '    need(True or isinstance(caption, dict)',
     [f"{RED}::test_a_probe_that_is_not_this_records_contract_measurement_refuses"], RC_REDUCE, ""),
    ("D40 probe ties bound", REDUCER,
     '    need(0 <= ties <= total - hits,', '    need(True,',
     [f"{RED}::test_a_probe_that_is_not_this_records_contract_measurement_refuses"], RC_REDUCE, ""),
    # --- probe producer (673, 679, 680, 681) ---
    ("P1 probe: record under this generation", PROBE,
     '        D.need(admitted["authority"]["manifest_sha256"] == manifest["sha256"],', '        D.need(True,',
     [f"{RED}::test_the_probe_refuses_a_record_from_another_generation_before_any_binary_load"],
     r"a binary was deserialised", "reaches the (sentinel) binary load"),
    ("P2 probe: ties are misses", PROBE,
     '    hits = int((own > others).sum().item())', '    hits = int((own >= others).sum().item())',
     [f"{RED}::test_strict_hits_count_aligned_permuted_and_tied_rows"], r"assert", ""),
    ("P3 probe: zero norm refuses", PROBE,
     '    D.need(bool((norms > ZERO_NORM).all()), "a centred feature has zero norm; the measurement refuses")',
     '    pass', [f"{RED}::test_invalid_features_refuse_instead_of_scoring"],
     r"Regex pattern did not match", DEPTH + " (non-finite similarities)"),
    ("P4 probe: finite inputs", PROBE,
     '    D.need(bool(torch.isfinite(q).all()) and bool(torch.isfinite(t).all()), "non-finite input features")',
     '    pass', [f"{RED}::test_invalid_features_refuse_instead_of_scoring"],
     r"Regex pattern did not match", DEPTH + " (NaN norms fail the zero-norm check)"),
    ("P5 probe: coordinate of the approved selection", PROBE,
     '    D.need(coordinate in D.expected_coordinates("decide", arms=M.ANCHOR_ARMS,',
     '    D.need(True or coordinate in D.expected_coordinates("decide", arms=M.ANCHOR_ARMS,',
     [f"{RED}::test_the_probe_refuses_a_coordinate_outside_the_approved_selection"],
     r"a binary was deserialised", "reaches the (sentinel) binary load"),
    ("P6 probe: inputs are the record's admitted inputs", PROBE,
     '    D.need(inputs is not None and all(inputs.get(k) == admitted_inputs.get(k) for k in keys),',
     '    D.need(True,', [f"{RED}::test_the_probe_refuses_inputs_other_than_the_records_admitted_ones"],
     r"a model was constructed", "reaches the (sentinel) model construction"),
    ("P7 probe CLI: the approval check", PROBE,
     '        approval = D.M.audit_approval(args.approval_section, "probe", manifest=manifest["sha256"],',
     '        approval = {} or dict(section=args.approval_section, line="") or D.M.audit_approval(args.approval_section, "probe", manifest=manifest["sha256"],',
     [f"{RED}::test_the_probe_cli_refuses_an_approval_of_another_operation"], r"assert", ""),
    ("M1 model gate", "model_siglip2.py",
     '        if self.axis_center == "anchors" and route_centroids is not None:',
     '        if route_centroids is not None:', [POR], r"assert|Error", ""),
]
#: guards no synthetic test can reach, stated rather than mutated (audit 673.3)
UNREACHABLE = {
    "probe `_assert_phase3_runtime_rows` after the real dataset load":
        "needs a real dataset and cache; covered by the trainer's own row check tests, not here",
}


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
                        "rc": rc, "outcome": outcome, "failed": failed[:6],
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
