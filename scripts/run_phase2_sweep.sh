#!/usr/bin/env bash
# Phase-2 capacity (CIBNT) sweep for the two main-table cells that dropped below
# their dataset's champion-K reference:
#   * Flickr25k K=64 L=3  (18-base, -0.0101 vs K128 champion) -- low-capacity corner
#   * CIFAR10  K=128 L=3  (18-base, -0.0086, DNA-uniq 0.049)   -- pigeonhole collision
# Champion CIBNT = 1.0 for both; sweep {0.5, 1.5}. If a value beats the
# baseline-recipe cell by >0.005 bio-projected mAP@R it is adopted, else the
# current value stands (robustness / structural-limit finding).
#
# Launch only with GPUs 0-3 free.  Usage: bash scripts/run_phase2_sweep.sh
set -u
cd /home/yschoi/GroundedDNA
mkdir -p logs
echo "[phase2] launch $(date '+%F %T')"

CIBNT=0.5 TAG_SUFFIX=_cibnt0p5 bash scripts/maintable_cell.sh 0 flickr  64  3 \
    > logs/sweep_flickr_K64_L3_cibnt0p5.log  2>&1 &
CIBNT=1.5 TAG_SUFFIX=_cibnt1p5 bash scripts/maintable_cell.sh 1 flickr  64  3 \
    > logs/sweep_flickr_K64_L3_cibnt1p5.log  2>&1 &
CIBNT=0.5 TAG_SUFFIX=_cibnt0p5 bash scripts/maintable_cell.sh 2 cifar10 128 3 \
    > logs/sweep_cifar_K128_L3_cibnt0p5.log  2>&1 &
CIBNT=1.5 TAG_SUFFIX=_cibnt1p5 bash scripts/maintable_cell.sh 3 cifar10 128 3 \
    > logs/sweep_cifar_K128_L3_cibnt1p5.log  2>&1 &
wait
echo "[phase2] ALL SWEEP DONE $(date '+%F %T')"
grep -h "\[CELL-RESULT\]" logs/sweep_*.log 2>/dev/null | sort -u
