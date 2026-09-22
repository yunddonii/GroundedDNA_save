#!/usr/bin/env bash
# usage: eval_batch.sh <runs_file> <gpu> [<gpu> ...]; one run per GPU slot at a time, rc = number of failures
set -u
RUNS=$1; shift; GPUS=("$@"); n=0; fails=0; pids=()
while read -r ds arm s d; do
  g=${GPUS[$(( n % ${#GPUS[@]} ))]}
  bash /home/yschoi/GroundedDNA/result/analysis/stage6_p2anc/eval_one.sh $ds $arm $s "$d" $g & pids+=($!); n=$((n+1))
  if [ $(( n % ${#GPUS[@]} )) -eq 0 ]; then for p in "${pids[@]}"; do wait $p || fails=$((fails+1)); done; pids=(); fi
done < "$RUNS"
for p in "${pids[@]}"; do wait $p || fails=$((fails+1)); done
echo "eval done fails=$fails"; exit "$fails"
