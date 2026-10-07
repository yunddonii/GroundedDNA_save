#!/usr/bin/env bash
# The final pinned-source suite (r8, commit 3fd0117): all 28 files inside the OS boundary, nothing deselected,
# individual skip reasons (-ra).
set -u
cd /data/yschoi/gdna_anchor_refit_v9r8 || exit 90
[ "$(git rev-parse --short HEAD)" = "3fd0117" ] || exit 91
[ -z "$(git status --porcelain)" ] || exit 92
timeout 3600 /home/yschoi/.conda/envs/dna_hashing/bin/python artifacts/anchor_confirmation/refit_v9/t_recovery_prep/sandboxed_pytest_r8.py /home/yschoi/anchor_rt_session_state/recovery_prep/final_3fd0117/suite /data/yschoi/gdna_anchor_refit_v9r8 --fake-nvidia-smi -- -q -ra tests/test_anchor_confirm_port.py tests/test_anchor_confirm_recipe.py tests/test_anchor_confirm_launcher.py tests/test_anchor_confirm_reducer.py tests/test_anchor_confirm_lifecycle.py tests/test_anchor_confirm_supervisor.py tests/test_anchor_confirm_input_bridge.py tests/test_anchor_confirm_env_handoff.py tests/test_phase3_selection_matrix.py tests/test_phase3_select_n.py tests/test_result_identity.py tests/test_seal_phase3_inputs.py tests/test_phase3_clip_snapshot.py tests/test_extraction_manifest_integration.py tests/test_phase3_launcher_e2e.py tests/test_siglip2_criterion_runtime.py tests/test_anchor_lambda_stage.py tests/test_p0_protocol.py tests/test_anchor_refit_stage.py tests/test_annealing_state_schema.py tests/test_extraction_cli_overrides.py tests/test_extraction_run_validation.py tests/test_manifest_fail_closed.py tests/test_phase2_inventory.py tests/test_phase2_launcher_wiring.py tests/test_seal_cell_analysis.py tests/test_siglip2_extraction_runtime_state.py tests/test_anchor_t_recovery.py
rc=$?
echo "suite rc=$rc"
exit "$rc"
