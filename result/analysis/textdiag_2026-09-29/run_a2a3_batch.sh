#!/usr/bin/env bash
# CPU only, low priority; the anchor stage-D campaign on GPUs 0-3 must not be disturbed.
set -u
cd /home/yschoi/gdna_textdiag || exit 90
OUT=result/analysis/textdiag_2026-09-29/a2a3; mkdir -p "$OUT"
rc_all=0
while read -r d; do
  [ -z "$d" ] && continue
  name=$(basename "$d" | sed -E 's/^[0-9]+\+//; s/\+bs.*$//')
  nice -n 10 env -u PYTHONPATH OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= GDNA_NUM_SEMANTIC_PARTS=5 \
    /home/yschoi/.conda/envs/dna_hashing/bin/python result/analysis/textdiag_2026-09-29/a2a3_slot_consistency.py \
    --result_dir "$d" --out "$OUT/$name.json" > "$OUT/$name.log" 2>&1
  rc=$?; echo "$name rc=$rc"; [ "$rc" -eq 0 ] || rc_all=1
done < result/analysis/textdiag_2026-09-29/runs.txt
exit "$rc_all"
