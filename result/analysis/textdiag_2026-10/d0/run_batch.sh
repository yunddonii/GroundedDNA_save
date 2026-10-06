#!/usr/bin/env bash
# D0: score checkpoints with a3_v2.py (CPU, low priority). Usage: run_batch.sh <runs.txt> <out_dir> [extra args...]
set -u
cd /home/yschoi/gdna_textdiag || exit 90
RUNS=$1; OUT=$2; shift 2; mkdir -p "$OUT"
rc_all=0
while read -r d; do
  [ -z "$d" ] && continue
  name=$(basename "$d" | sed -E 's/^[0-9]+\+//; s/\+bs.*$//')
  [ -s "$OUT/$name.json" ] && { echo "$name skip"; continue; }
  nice -n 10 env -u PYTHONPATH OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= GDNA_NUM_SEMANTIC_PARTS=5 \
    /home/yschoi/.conda/envs/dna_hashing/bin/python result/analysis/textdiag_2026-10/a3_v2.py \
    --result_dir "$d" --out "$OUT/$name.json" "$@" > "$OUT/$name.log" 2>&1
  rc=$?; echo "$name rc=$rc"; [ "$rc" -eq 0 ] || rc_all=1
done < "$RUNS"
exit "$rc_all"
