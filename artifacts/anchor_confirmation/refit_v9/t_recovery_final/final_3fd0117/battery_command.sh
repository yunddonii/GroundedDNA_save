#!/usr/bin/env bash
set -u
cd /data/yschoi/gdna_anchor_refit_v9r8 || exit 90
[ "$(git rev-parse --short HEAD)" = "3fd0117" ] || exit 91
[ -z "$(git status --porcelain)" ] || exit 92
/home/yschoi/.conda/envs/dna_hashing/bin/python artifacts/anchor_confirmation/refit_v9/mutation_v14.py 3fd0117 /tmp/claude-1003/-home-yschoi-GroundedDNA/dad53253-2628-45a2-849e-bb4f82a11b67/scratchpad/battery14_3fd0117 > /home/yschoi/anchor_rt_session_state/recovery_prep/final_3fd0117/battery_console.log 2>&1
rc=$?
echo "battery rc=$rc"
exit "$rc"
