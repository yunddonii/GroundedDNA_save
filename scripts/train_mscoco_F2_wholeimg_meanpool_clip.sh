#!/usr/bin/env bash
# MSCOCO F2 (token-mean pooling) — WHOLE-IMAGE training + inference.
# Structural unification: train cache = mscoco_clip_v5b_tokens (196-patch
# whole-image) instead of FAIRrank L8K3 multi-crop. Base F2 recipe otherwise.
# Usage: bash scripts/train_mscoco_F2_wholeimg_meanpool_clip.sh <GPU>
set -eu
GPU="${1:-5}"
CACHE=./cache/mscoco_clip_v5b_tokens BI_V=1.0 BI_T=1.0 \
  TAG=mscoco_F2_WHOLEIMG_meanpool_v1.0_t1.0_K128 \
  bash "$(dirname "$0")/train_mscoco_v185_bidirTokenPrune05_clip.sh" "$GPU"
