#!/usr/bin/env bash
# Unsupervised hashing baselines (CIBHash / CIMON / MLS3RDUH) on CUB-200,
# CLIP ViT-B/16 frozen features (same backbone as our cub200 v160b/v162b),
# 36-bit codes = K=64 DNA schema (6 codebooks x 6 bits). Fair direct
# comparison: identical frozen backbone, same setting1 split, same bit budget.
#
# Each method runs on its own GPU (passed as $1 $2 $3; default 2 3 4).
# Usage: bash scripts/run_unsup_baselines_cub.sh [gpu_cib] [gpu_cimon] [gpu_mls]
set -u
PY=/home/yschoi/.conda/envs/dna_hashing/bin/python
cd /home/yschoi/GroundedDNA
COMMON="-d CUB_200 -s setting1 --bit 36 -me 60 -ep 60 --batch_size 64 \
  --model_root ./params_baseline --result_root ./result_baseline \
  --compress_root ./compress_baseline --cache_dir ./cache/cub200_clip"

GPU_CIB="${1:-2}"; GPU_CIMON="${2:-3}"; GPU_MLS="${3:-4}"
mkdir -p logs

run() {
  local method=$1 gpu=$2
  local tag="${method}_cub200_unsup60"
  echo "=========== launching ${method} on cuda:${gpu} (${tag}) ==========="
  CUDA_VISIBLE_DEVICES=${gpu} $PY -m baseline.base_model \
      --method ${method} ${COMMON} --device cuda:0 -tn ${tag} \
      > logs/baseline_${method}_cub200.log 2>&1
  echo "=========== ${method} done (exit $?) ==========="
}

run cibhash   "$GPU_CIB"   &
run cimon     "$GPU_CIMON" &
run mls3rduh  "$GPU_MLS"   &
wait
echo "=== ALL CUB BASELINES DONE ==="
