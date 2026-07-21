#!/usr/bin/env bash
# Main-table grid launcher: fill the 12 missing (dataset, K, L) cells with the
# P0 2-stage protocol + bio-projected mAP@R. Existing P0 champions are NOT
# re-run: flickr K128 L3, mscoco K128 L3, nuswide K128 L3, cifar10 K64 L3.
#
# 12 cells across 6 GPUs (2 per GPU), heavy MSCOCO cells spread over GPUs 0-2.
# Each GPU runs its cells sequentially via scripts/maintable_cell.sh.
#
#   Usage: bash scripts/run_maintable_grid.sh            (launches all, waits)
# Per-GPU progress -> logs/grid_gpu<N>.log ; per-cell -> logs/<ds>_K<K>_L<L>_P0*.log
set -u
mkdir -p logs
STAMP="$(date '+%F %T')"
echo "[grid] launch $STAMP"

run_gpu() {   # $1=gpu ; $2.. = "DS:K:L" cells (run sequentially)
  local gpu="$1"; shift
  for cell in "$@"; do
    IFS=: read -r ds k l <<< "$cell"
    echo "[grid][gpu$gpu] >>> $ds K=$k L=$l  $(date '+%T')"
    bash scripts/maintable_cell.sh "$gpu" "$ds" "$k" "$l" \
      || echo "[grid][gpu$gpu] !!! cell $ds K=$k L=$l FAILED (continuing)"
  done
  echo "[grid][gpu$gpu] queue done $(date '+%T')"
}

run_gpu 0 mscoco:64:3  flickr:64:3   > logs/grid_gpu0.log 2>&1 &
run_gpu 1 mscoco:128:4 flickr:128:4  > logs/grid_gpu1.log 2>&1 &
run_gpu 2 mscoco:64:4  flickr:64:4   > logs/grid_gpu2.log 2>&1 &
run_gpu 3 nuswide:64:3 cifar10:128:3 > logs/grid_gpu3.log 2>&1 &
run_gpu 4 nuswide:128:4 cifar10:64:4 > logs/grid_gpu4.log 2>&1 &
run_gpu 5 nuswide:64:4 cifar10:128:4 > logs/grid_gpu5.log 2>&1 &

wait
echo "[grid] ALL QUEUES DONE $(date '+%F %T')"
echo "[grid] cell results:"
grep -h "\[CELL-RESULT\]" logs/*_P0refit_*.log logs/grid_gpu*.log 2>/dev/null | sort -u
