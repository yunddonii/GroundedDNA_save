#!/usr/bin/env bash
# P0 stage 1 for the baselines: retrain each on the optimization-train 90% so
# that E* can be picked on val rows the model never saw.
#
# The earlier val-selection (docs/baseline_val_select/) scored checkpoints from
# runs trained on 100% of train, so those "val" rows were part of their training
# data -- an optimistic, non-held-out estimate. These stage-1 runs fix that.
#
# The stage-1 model itself is NOT reported. Only E* is taken from it; the number
# in the table comes from the existing 100%-train run's eval_epoch_{E*}.json,
# mirroring our own stage-2 refit (train on 100%, stop at E*).
#
# Usage: bash scripts/run_baselines_p0_stage1.sh <GPU_A> <GPU_B>
set -u
PY=/home/yschoi/.conda/envs/dna_hashing/bin/python
cd /home/yschoi/GroundedDNA
mkdir -p logs

GPU_A="${1:-2}"; GPU_B="${2:-3}"

cache_for() {
    case "$1" in
        CIFAR10)   echo ./cache/cifar10_clip ;;
        Flickr25k) echo ./cache/flickr25k_clip_v4plus_qwen3_tokens ;;
        MSCOCO)    echo ./cache/mscoco_clip_v5b ;;
        NUSWIDE)   echo ./cache/nuswide_clip_tokens ;;
    esac
}
short() {
    case "$1" in
        CIFAR10) echo cifar10 ;; Flickr25k) echo flickr25k ;;
        MSCOCO) echo mscoco ;; NUSWIDE) echo nuswide ;;
    esac
}

run() {   # $1=method $2=dataset $3=gpu
    local m=$1 d=$2 gpu=$3
    local tag="${m}_$(short "$d")_clip_P0s1_unsup60"
    echo "=== ${tag} on cuda:${gpu}"
    CUDA_VISIBLE_DEVICES="${gpu}" $PY -m baseline.base_model \
        --method "${m}" -d "${d}" -s setting1 --bit 36 -me 60 -ep 5 \
        --batch_size 64 --device cuda:0 -tn "${tag}" \
        --model_root ./params_baseline --result_root ./result_baseline \
        --compress_root ./compress_baseline --cache_dir "$(cache_for "$d")" \
        --val_split_ratio 0.1 --val_split_seed 42 \
        > "logs/baseline_${tag}.log" 2>&1
    echo "=== ${tag} done ($(grep -c . "logs/baseline_${tag}.log") lines)"
}

# Two serial lanes so the two free GPUs stay busy without oversubscribing them.
(
  for d in Flickr25k NUSWIDE; do
      for m in cibhash cimon mls3rduh; do run "$m" "$d" "$GPU_A"; done
  done
) &
(
  for d in MSCOCO CIFAR10; do
      for m in cibhash cimon mls3rduh; do run "$m" "$d" "$GPU_B"; done
  done
) &
wait
echo "[baselines-P0s1] all done at $(date '+%F %T')"
