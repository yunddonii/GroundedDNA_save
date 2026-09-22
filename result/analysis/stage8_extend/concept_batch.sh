#!/usr/bin/env bash
# usage: concept_batch.sh <runs_file> <gpu> ...: concept-name reading for every ac2 run listed
set -u; RUNS=$1; shift; G=("$@"); n=0; fails=0; pids=()
cd /home/yschoi/GroundedDNA; E=result/analysis/stage8_extend/eval
while read -r ds arm s d; do [ "$arm" = ac2 ] || continue
  g=${G[$(( n % ${#G[@]} ))]}
  [ -f $E/concept_${ds}_ac2_s$s.json ] || CUDA_VISIBLE_DEVICES=$g GDNA_NUM_SEMANTIC_PARTS=5 /home/yschoi/.conda/envs/dna_hashing/bin/python result/analysis/stage7_bigarms/concept_reading.py --result_dir "$d" --out $E/concept_${ds}_ac2_s$s.json > $E/concept_${ds}_ac2_s$s.log 2>&1 &
  pids+=($!); n=$((n+1)); done < "$RUNS"
for p in "${pids[@]}"; do wait $p || fails=$((fails+1)); done; echo "concept done fails=$fails"; exit "$fails"
