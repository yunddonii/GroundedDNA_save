#!/usr/bin/env bash
# Unsupervised hashing baselines (CIBHash / CIMON / MLS3RDUH) on CIFAR10
# setting1, CLIP ViT-B/16 frozen features (matches our CIFAR10 champion
# comparison), 36-bit codes. Fair direct comparison.
#
# Usage: bash scripts/run_unsup_baselines_cifar10.sh [gpu_cib] [gpu_cimon] [gpu_mls]
set -u
PY=/home/yschoi/.conda/envs/dna_hashing/bin/python
cd /home/yschoi/GroundedDNA
COMMON="-d CIFAR10 -s setting1 --bit 36 -me 60 -ep 60 --batch_size 64 \
  --model_root ./params_baseline --result_root ./result_baseline \
  --compress_root ./compress_baseline --cache_dir ./cache/cifar10_clip"

GPU_CIB="${1:-3}"; GPU_CIMON="${2:-4}"; GPU_MLS="${3:-5}"
mkdir -p logs

run() {
  local method=$1 gpu=$2
  local tag="${method}_cifar10_clip_unsup60"
  echo "=========== launching ${method} on cuda:${gpu} (${tag}) ==========="
  CUDA_VISIBLE_DEVICES=${gpu} $PY -m baseline.base_model \
      --method ${method} ${COMMON} --device cuda:0 -tn ${tag} \
      > logs/baseline_${method}_cifar10.log 2>&1
  echo "=========== ${method} done (exit $?) ==========="
}

run cibhash   "$GPU_CIB"   &
run cimon     "$GPU_CIMON" &
run mls3rduh  "$GPU_MLS"   &
wait
echo "=== ALL CIFAR10 BASELINES DONE ==="
