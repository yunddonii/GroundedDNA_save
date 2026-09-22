#!/usr/bin/env bash
set -u; cd /home/yschoi/GroundedDNA; O=result/analysis/stage9_levers; A=result/analysis/stage6_p2anc; G=(5 4 3 2 1); n=0; fails=0; pids=()
while read -r ds arm s d; do g=${G[$(( n % 5 ))]}; t=${ds}_${arm}_s${s}
  [ -f $A/analysis/$t.json ] || CUDA_VISIBLE_DEVICES=$g GDNA_NUM_SEMANTIC_PARTS=5 /home/yschoi/.conda/envs/dna_hashing/bin/python $A/anchor_analysis.py --result_dir "$d" --out $A/analysis/$t.json > $A/analysis/$t.log 2>&1 &
  pids+=($!); n=$((n+1)); if [ $(( n % 5 )) -eq 0 ]; then for p in "${pids[@]}"; do wait $p || fails=$((fails+1)); done; pids=(); fi
done < $O/eval_runs.txt
for p in "${pids[@]}"; do wait $p || fails=$((fails+1)); done; echo "analysis done fails=$fails"; exit "$fails"
