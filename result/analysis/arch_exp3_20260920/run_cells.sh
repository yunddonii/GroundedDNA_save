#!/bin/bash
# usage: run_cells.sh <cells_file> <gpu> [<gpu> ...]
# Launches one cell per listed GPU, waits for the wave, repeats. Off-protocol:
# no GDNA_PHASE3_* variables, so every cell lands in this repo's result/.
set -u
CELLS="$1"; shift; GPUS=("$@")
REPO=${GDNA_RUN_REPO:-/home/yschoi/GroundedDNA}
PY=/home/yschoi/.conda/envs/dna_hashing/bin/python
LOGDIR=$(dirname "$CELLS")
export GDNA_NUM_SEMANTIC_PARTS=5
for v in $(env | grep -o '^GDNA_PHASE3[A-Z_]*'); do unset "$v"; done
cd "$REPO" || exit 1
echo "start $(date '+%F %T')  cells=$CELLS  gpus=${GPUS[*]}"
n=0; fails=0; pids=()
while IFS='|' read -r idx tag cmd; do
  [ -z "$tag" ] && continue
  g=${GPUS[$(( n % ${#GPUS[@]} ))]}
  CUDA_VISIBLE_DEVICES=$g $PY ${cmd#python } > "$LOGDIR/${tag}.log" 2>&1 &
  pids+=($!); echo "  launched $tag on gpu $g $(date '+%T')"
  n=$((n+1))
  if [ $(( n % ${#GPUS[@]} )) -eq 0 ]; then
    for p in "${pids[@]}"; do wait "$p" || fails=$((fails+1)); done; pids=(); echo "  --- wave done $(date '+%T') fails=$fails"
  fi
done < "$CELLS"
for p in "${pids[@]}"; do wait "$p" || fails=$((fails+1)); done
echo "end $(date '+%F %T') fails=$fails"
exit "$fails"
