#!/usr/bin/env bash
# Flickr25k F2 (token-mean pooling) — WHOLE-IMAGE training + inference.
# Structural unification: all 4 datasets train+infer on whole-image (196 patches).
# = base F2 recipe with the training cache swapped from FAIRrank L8K3 (588-patch
#   multi-crop) to the whole-image 196-patch tokens cache.
# Usage: bash scripts/train_flickr25k_F2_wholeimg_meanpool_clip.sh <GPU>
set -eu
GPU="${1:-4}"
CACHE=./cache/flickr25k_clip_v4plus_qwen3_tokens BI_V=1.0 BI_T=1.0 \
  TAG=flickr_F2_WHOLEIMG_meanpool_v1.0_t1.0_K128 \
  bash "$(dirname "$0")/train_flickr25k_v185_bidirTokenPrune05_clip.sh" "$GPU"
