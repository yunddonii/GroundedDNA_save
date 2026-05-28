#!/usr/bin/env bash
# Build MSCOCO CLIP cache mirroring the Flickr25k one. Run AFTER the
# Flickr25k CLIP extraction completes (the GPU + disk get freed up).
#
# Usage:
#   bash scripts/extract_clip_mscoco.sh <GPU_ID>
# Example:
#   bash scripts/extract_clip_mscoco.sh 0
set -eu
GPU="${1:-0}"
LOG="logs/extract_clip_mscoco.log"
mkdir -p logs

CUDA_VISIBLE_DEVICES="$GPU" \
/home/yschoi/.conda/envs/dna_hashing/bin/python extract_clip_features.py \
    --mode pathlist \
    --pathlist_root ./dataset/MSCOCO \
    --pathlist_setting setting1 \
    --qwen_cache_path ./cache/mscoco_qwen_v4.jsonl \
    --cache_dir ./cache/mscoco_clip_v4plus \
    --batch_size 128 \
    --num_workers 8 \
    --save_aug_views 2 2>&1 | tee "$LOG"
