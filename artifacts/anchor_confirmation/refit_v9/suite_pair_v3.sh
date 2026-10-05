#!/usr/bin/env bash
# The 27-file regression suite (generation v9 r6: the 19 files plus the 8 that exercise the shared
# runtime resolver and extraction code r6 changes) at a commit, unguarded and then under the open() guard (audit 735.2).
# The readiness verdict is BOTH: it fails if either run fails or the guard refused any open.
# Precondition (audit 736): bounded_tree head-clean over the manifest closure plus the 17 test files;
# no other tracked file's bytes are read.
set -u
WANT=${1:?commit}; OUT=${2:?out dir}
W=/data/yschoi/gdna_anchor_refit_v9r6; A=$W/artifacts/anchor_confirmation
cd "$W" || exit 90
[ "$(git rev-parse --short HEAD)" = "$WANT" ] || exit 91
PY=/home/yschoi/.conda/envs/dna_hashing/bin/python
FILES="tests/test_anchor_confirm_port.py tests/test_anchor_confirm_recipe.py tests/test_anchor_confirm_launcher.py tests/test_anchor_confirm_reducer.py tests/test_anchor_confirm_lifecycle.py tests/test_anchor_confirm_supervisor.py tests/test_anchor_confirm_input_bridge.py tests/test_anchor_confirm_env_handoff.py tests/test_phase3_selection_matrix.py tests/test_phase3_select_n.py tests/test_result_identity.py tests/test_seal_phase3_inputs.py tests/test_phase3_clip_snapshot.py tests/test_extraction_manifest_integration.py tests/test_phase3_launcher_e2e.py tests/test_siglip2_criterion_runtime.py tests/test_anchor_lambda_stage.py tests/test_p0_protocol.py tests/test_anchor_refit_stage.py tests/test_annealing_state_schema.py tests/test_extraction_cli_overrides.py tests/test_extraction_run_validation.py tests/test_manifest_fail_closed.py tests/test_phase2_inventory.py tests/test_phase2_launcher_wiring.py tests/test_seal_cell_analysis.py tests/test_siglip2_extraction_runtime_state.py"
mkdir -p "$OUT"
$PY $A/refit_v9/bounded_tree.py head-clean "$W" $A/authority_manifest_v9r7.json $FILES > "$OUT/head_clean.json" || { cat "$OUT/head_clean.json"; exit 92; }
unset PYTHONPATH GDNA_ALLOW_REAL_ARTIFACT_TESTS
export CUDA_VISIBLE_DEVICES= GDNA_NUM_SEMANTIC_PARTS=5
timeout 3600 $PY -m pytest -q -p no:cacheprovider $FILES > "$OUT/suite_unguarded.log" 2>&1
rc1=$?
timeout 3600 $PY $A/refit_v9/guarded_pytest.py "$OUT/guard" -q -p no:cacheprovider -rf $FILES > "$OUT/suite_guarded.log" 2>&1
rc2=$?
refused=$($PY -c "import json,sys; print(len(json.load(open(sys.argv[1]))['violations']))" "$OUT/guard/violations.json" 2>/dev/null || echo unknown)
$PY $A/refit_v9/bounded_tree.py head-clean "$W" $A/authority_manifest_v9r7.json $FILES > "$OUT/head_clean_after.json"; rc3=$?
echo "{\"commit\": \"$WANT\", \"unguarded_rc\": $rc1, \"guarded_rc\": $rc2, \"guard_refused_opens\": \"$refused\", \"head_clean_after_rc\": $rc3}" > "$OUT/verdict.json"
echo "unguarded: $(tail -1 "$OUT/suite_unguarded.log") rc=$rc1"
echo "guarded:   $(grep -E 'passed|failed' "$OUT/suite_guarded.log" | tail -1) rc=$rc2 refused=$refused"
[ "$rc1" -eq 0 ] && [ "$rc2" -eq 0 ] && [ "$refused" = "0" ] && [ "$rc3" -eq 0 ] && exit 0
exit 1
