#!/bin/bash
# Slot-health measurement across the 8 p3lamA lambda cells (2026-09-15 campaign).
# Read-only: loads each terminal checkpoint and measures slot separation at every
# stage of the routing path. No training, no writes outside $OUT.
set -u
REPO=/home/yschoi/GroundedDNA
OUT=$REPO/result/analysis/lambda_slot_health_20260919
PY=/home/yschoi/.conda/envs/dna_hashing/bin/python
export GDNA_NUM_SEMANTIC_PARTS=5
cd "$REPO" || exit 1
echo "start $(date '+%F %T')"
for t in s42 s43 s44 s42_LW030 s42_LW050 s42_LTH0025 s42_LTH010 s42_LBU0; do
  d=$(ls -d /data/yschoi/gdna_p3exec_result/260915+flickr25k_setting1_p3lamA_flickr_A_v4_N4_${t}_P06095_JD002* 2>/dev/null | head -1)
  if [ -z "$d" ]; then echo "=== $t : DIR MISSING"; continue; fi
  echo "=== $t"
  CUDA_VISIBLE_DEVICES=0 "$PY" scripts/run_slot_collapse_probe.py \
      --result_dir "$d" --n 256 --device cuda:0 --out "$OUT/${t}.json" 2>&1 \
    | grep -vE "Loading weights|UserWarning|^  out\[|it/s\]|siglip2-cache|ImgRtvDataset|model_siglip2\]"
done
rc=$?
echo "end $(date '+%F %T')"
exit "$rc"
