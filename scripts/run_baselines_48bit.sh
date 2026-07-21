#!/usr/bin/env bash
# 48-bit (24-base DNA) baseline training for the main-table comparison, P0-clean.
#
# Produces, at --bit 48, for CIBHash / CIMON / MLS3RDUH:
#   * stage-1 (90% opt-train, --val_split_ratio 0.1 seed 42) for ALL 4 datasets
#     -> per-epoch checkpoints used to pick E* on held-out val (leak-free).
#   * 100%-train runs for NUSWIDE + CIFAR10 (Flickr25k/MSCOCO already exist at
#     result_baseline/260715/*_48bit_unsup60, same clip_v4plus cache) -> the
#     eval_epoch_{E*}.json of these is the reported number.
#
# Consistent cache per dataset for BOTH stages (avoids the 36-bit stage1/full
# cache mismatch): clip_v4plus / nuswide_clip / cifar10_clip. All have the
# visual_global + visual_global_aug{0,1} views cibhash/cimon require.
#
# Launch ONLY when GPUs are free (i.e. after the model grid finishes).
#   Usage: bash scripts/run_baselines_48bit.sh          (uses GPUs 0-5)
set -u
PY=/home/yschoi/.conda/envs/dna_hashing/bin/python
cd /home/yschoi/GroundedDNA
mkdir -p logs
echo "[bl48] launch $(date '+%F %T')"

cache_for() {
    case "$1" in
        CIFAR10)   echo ./cache/cifar10_clip ;;
        Flickr25k) echo ./cache/flickr25k_clip_v4plus ;;
        MSCOCO)    echo ./cache/mscoco_clip_v4plus ;;
        NUSWIDE)   echo ./cache/nuswide_clip ;;
    esac
}
short() {
    case "$1" in
        CIFAR10) echo cifar10 ;; Flickr25k) echo flickr25k ;;
        MSCOCO) echo mscoco ;; NUSWIDE) echo nuswide ;;
    esac
}

# $1=method $2=dataset $3=gpu $4=stage(s1|full)
run() {
    local m=$1 d=$2 gpu=$3 stage=$4
    local sd; sd="$(short "$d")"
    local tag extra
    if [ "$stage" = "s1" ]; then
        tag="${m}_${sd}_clip_P0s1_48bit"
        extra="--val_split_ratio 0.1 --val_split_seed 42"
    else
        tag="${m}_${sd}_clip_48bit_unsup60"
        extra=""
    fi
    echo "[bl48] >>> ${tag} on cuda:${gpu}  $(date '+%T')"
    CUDA_VISIBLE_DEVICES="${gpu}" $PY -m baseline.base_model \
        --method "${m}" -d "${d}" -s setting1 --bit 48 -me 60 -ep 5 \
        --batch_size 64 --device cuda:0 -tn "${tag}" \
        --model_root ./params_baseline --result_root ./result_baseline \
        --compress_root ./compress_baseline --cache_dir "$(cache_for "$d")" \
        ${extra} > "logs/bl48_${tag}.log" 2>&1 \
        && echo "[bl48] OK  ${tag}  $(date '+%T')" \
        || echo "[bl48] FAIL ${tag}  $(date '+%T')"
}

lane() {  # $1=gpu ; $2.. = "method:dataset:stage"
    local gpu="$1"; shift
    for spec in "$@"; do
        IFS=: read -r m d st <<< "$spec"
        run "$m" "$d" "$gpu" "$st"
    done
    echo "[bl48][gpu$gpu] lane done $(date '+%T')"
}

# 6 lanes x 3 runs = 18. Each lane is one (dataset, stage) triple.
lane 0 cibhash:NUSWIDE:full  cimon:NUSWIDE:full  mls3rduh:NUSWIDE:full  > logs/bl48_gpu0.log 2>&1 &
lane 1 cibhash:NUSWIDE:s1    cimon:NUSWIDE:s1    mls3rduh:NUSWIDE:s1    > logs/bl48_gpu1.log 2>&1 &
lane 2 cibhash:CIFAR10:full  cimon:CIFAR10:full  mls3rduh:CIFAR10:full  > logs/bl48_gpu2.log 2>&1 &
lane 3 cibhash:CIFAR10:s1    cimon:CIFAR10:s1    mls3rduh:CIFAR10:s1    > logs/bl48_gpu3.log 2>&1 &
lane 4 cibhash:Flickr25k:s1  cimon:Flickr25k:s1  mls3rduh:Flickr25k:s1  > logs/bl48_gpu4.log 2>&1 &
lane 5 cibhash:MSCOCO:s1     cimon:MSCOCO:s1     mls3rduh:MSCOCO:s1     > logs/bl48_gpu5.log 2>&1 &
wait
echo "[bl48] ALL TRAINING DONE $(date '+%F %T')"
