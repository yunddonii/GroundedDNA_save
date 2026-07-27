#!/usr/bin/env bash
# Unsupervised hashing baselines (CIBHash / CIMON / MLS3RDUH) on NUS-WIDE
# setting1 (10,500 balanced train / 2,100 test / 193,734 database),
# CLIP ViT-B/16 frozen features (cache/nuswide_clip). 36-bit codes.
#
# Usage: bash scripts/run_unsup_baselines_nuswide.sh [gpu_cib] [gpu_cimon] [gpu_mls]
set -u
PY=/home/yschoi/.conda/envs/dna_hashing/bin/python
cd /home/yschoi/GroundedDNA
# -ep 5: eval every 5 epochs (matches our model's eval_every=5) so each
# baseline can be reported at its BEST epoch — fair vs our best-ckpt protocol.
COMMON="-d NUSWIDE -s setting1 --bit 36 -me 60 -ep 5 --batch_size 64 \
  --model_root ./params_baseline --result_root ./result_baseline \
  --compress_root ./compress_baseline --cache_dir ./cache/nuswide_clip"

GPU_CIB="${1:-0}"; GPU_CIMON="${2:-1}"; GPU_MLS="${3:-2}"
mkdir -p logs

run() {
  local method=$1 gpu=$2
  local tag="${method}_nuswide_clip_unsup60"
  echo "=========== launching ${method} on cuda:${gpu} (${tag}) ==========="
  CUDA_VISIBLE_DEVICES=${gpu} $PY -m baseline.base_model \
      --method ${method} ${COMMON} --device cuda:0 -tn ${tag} \
      > logs/baseline_${method}_nuswide.log 2>&1
  echo "=========== ${method} done (exit $?) ==========="
}

run cibhash   "$GPU_CIB"   &
run cimon     "$GPU_CIMON" &
run mls3rduh  "$GPU_MLS"   &
wait
echo "=== ALL NUSWIDE BASELINES DONE ==="
