#!/usr/bin/env bash
# usage: eval_one.sh <dataset> <arm> <seed> <run_dir> <gpu>  -> three probe JSONs under stage6_p2anc/eval/
set -uo pipefail
cd /home/yschoi/GroundedDNA
ds=$1; arm=$2; s=$3; d=$4; g=$5; E=result/analysis/stage6_p2anc/eval; mkdir -p $E; t=${ds}_${arm}_s${s}
export CUDA_VISIBLE_DEVICES=$g GDNA_NUM_SEMANTIC_PARTS=5
PY=/home/yschoi/.conda/envs/dna_hashing/bin/python
rc=0
[ -f $E/qgap_$t.json ] || $PY result/analysis/arch_exp3_20260920/quant_gap_diag.py --result_dir "$d" --out $E/qgap_$t.json > $E/qgap_$t.log 2>&1 || rc=1
[ -f $E/m1_$t.json ]   || $PY scripts/slot_role_probe.py --result_dir "$d" --out $E/m1_$t.json > $E/m1_$t.log 2>&1 || rc=1
[ -f $E/zscr_$t.json ] || $PY result/analysis/stage2_zscr/zscr_pilot.py --result_dir "$d" --out $E/zscr_$t.json > $E/zscr_$t.log 2>&1 || rc=1
exit "$rc"
