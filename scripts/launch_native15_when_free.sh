#!/usr/bin/env bash
# Launch the 15-base native-DNA matrix once the Phase 2 chains on GPUs 0 and 1
# have exited, so the two never contend for the same device.
set -uo pipefail
REPO=/home/yschoi/GroundedDNA
PY=/home/yschoi/.conda/envs/dna_hashing/bin/python
cd "$REPO"

while pgrep -f "phase2_f01_reinference.sh 0 " >/dev/null \
   || pgrep -f "phase2_f01_reinference.sh 1 " >/dev/null; do
    sleep 120
done

echo "[native15] GPUs 0,1 free at $(date -Is); launching 48 cells"
GDNA_NATIVE_DNA_BASES=15 "$PY" scripts/run_native_dna_p0_matrix.py \
    --data-root /data/yschoi/groundeddna_native_p0_15base_v2 \
    --gpus 0 1 \
    --primo-predictor-npz /data/yschoi/groundeddna_native_p0/artifacts/primo_yield_predictor.npz
echo "[native15] exit=$? at $(date -Is)"
