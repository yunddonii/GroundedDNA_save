#!/bin/bash
# (B) loss-budget rebalancing: single-delta on lambda_text_hash_ntxent over the
# 2026-09-15 p3lamA selection-cell recipe, 3 seeds. Train-only validation
# (val_split_ratio 0.1, selection_mode select) -- the official test split is
# never read. Off-protocol: no GDNA_PHASE3_* environment, so results land in
# this repo's result/ and join no approved campaign.
set -u
REPO=/home/yschoi/GroundedDNA
OUT=$REPO/result/analysis/B_lossbudget_20260919
PY=/home/yschoi/.conda/envs/dna_hashing/bin/python
export GDNA_NUM_SEMANTIC_PARTS=5
unset $(env | grep -o '^GDNA_PHASE3[A-Z_]*' | tr '\n' ' ') 2>/dev/null || true
cd "$REPO" || exit 1
echo "start $(date '+%F %T')"
while IFS='|' read -r idx tag cmd; do
  gpu=$idx
  echo "=== cell $idx  tag=$tag  gpu=$gpu  $(date '+%T')"
  CUDA_VISIBLE_DEVICES=$gpu $PY ${cmd#python } > "$OUT/${tag}.log" 2>&1 &
  echo $! > "$OUT/${tag}.pid"
done < "$OUT/cells.txt"
wait
echo "end $(date '+%F %T')"
for f in "$OUT"/bexpB_*.log; do
  echo "--- $(basename $f): $(tail -3 "$f" | tr '\n' ' ' | cut -c1-160)"
done
