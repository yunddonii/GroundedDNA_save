#!/bin/bash
# (A) frozen text-initialised codebook, Flickr25k, 4 arms x 3 seeds.
#   A1 kmeans init, not frozen   -- does text init alone do anything?
#   A2 kmeans init, frozen at 0  -- fixed text-derived codewords from the start
#   A3 kmeans init, frozen at 2  -- late consolidation, as the VQ literature does it
#   A4 random init, frozen at 0  -- CONTROL: separates "text init" from "not moving"
# Same p3lamA stage-1 recipe as (B); official test split never read.
set -u
REPO=/home/yschoi/GroundedDNA
OUT=$REPO/result/analysis/A_frozencodebook_20260919
PY=/home/yschoi/.conda/envs/dna_hashing/bin/python
export GDNA_NUM_SEMANTIC_PARTS=5
cd "$REPO" || exit 1
echo "start $(date '+%F %T')"
n=0
while IFS='|' read -r idx tag cmd; do
  gpu=$(( n % 6 ))
  CUDA_VISIBLE_DEVICES=$gpu $PY ${cmd#python } > "$OUT/${tag}.log" 2>&1 &
  echo "  launched $tag on gpu $gpu  $(date '+%T')"
  n=$((n+1))
  if [ $(( n % 6 )) -eq 0 ]; then echo "  --- wave barrier"; wait; fi
done < "$OUT/cells.txt"
wait
echo "end $(date '+%F %T')"
