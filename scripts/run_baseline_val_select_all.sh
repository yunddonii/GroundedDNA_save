#!/usr/bin/env bash
# Re-select every baseline's epoch on the P0 val split (no retraining, CPU only
# -- the encoder is a single linear layer over cached features, and the GPUs are
# reserved for the P0 training runs and an unrelated job).
set -u
PY=/home/yschoi/.conda/envs/dna_hashing/bin/python
mkdir -p docs/baseline_val_select

cache_for() {
    case "$1" in
        cifar10)   echo cache/cifar10_clip ;;
        flickr25k) echo cache/flickr25k_clip_v4plus_qwen3_tokens ;;
        mscoco)    echo cache/mscoco_clip_v5b ;;
        nuswide)   echo cache/nuswide_clip_tokens ;;
    esac
}

for d in params_baseline/*/*_clip_mapr_unsup60; do
    trial=$(basename "$d"); day=$(basename "$(dirname "$d")")
    method="${trial%%_*}"; rest="${trial#*_}"; ds="${rest%%_*}"
    out="docs/baseline_val_select/${method}_${ds}.json"
    if [ -f "$out" ]; then echo "[skip] $trial (done)"; continue; fi
    echo "=== $trial  (cache $(cache_for "$ds"))"
    $PY scripts/baseline_val_select.py \
        --params_dir "params_baseline/$day/$trial" \
        --result_dir "result_baseline/$day/$trial" \
        --cache_dir  "$(cache_for "$ds")" \
        --device cpu --out "$out" 2>&1 | grep -vE "eval\[binary\]|it/s\]" | tail -4
done
echo "[val-select] all done at $(date '+%F %T')"
