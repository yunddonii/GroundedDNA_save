#!/usr/bin/env bash
# v81a (confidence-adaptive top-p routing) on Flickr25k setting1 with the
# CLIP backbone. Mirrors the exact v81a hyperparameter set from
# `result/260526+flickr25k_setting1_v81a_v62b_adaptiveTopP_05_09+bs+64+e+60+proj_lr+0.001/args.txt`,
# only changing --backbone_type to 'clip' and pointing the feature cache at
# the CLIP cache built by `extract_clip_features.py`. d_model is left to
# auto-default (768, see model_siglip2.py rationale for CLIP).
#
# Usage:
#   bash scripts/train_v81a_flickr25k_clip.sh <GPU_ID>
# Example:
#   bash scripts/train_v81a_flickr25k_clip.sh 0
set -eu

GPU="${1:-0}"
CACHE="${CACHE:-./cache/flickr25k_clip_v4plus}"
QWEN="${QWEN:-./cache/flickr25k_qwen_v4.jsonl}"
TAG="${TAG:-v81a_clip_v62b_adaptiveTopP_05_09}"
LOG="logs/train_${TAG}.log"
mkdir -p logs

if [[ ! -d "$CACHE" ]]; then
    echo "[run-v81a-clip] missing cache: $CACHE"
    echo "  Run extract_clip_features.py first."
    exit 2
fi

echo "[run-v81a-clip] GPU=$GPU cache=$CACHE qwen=$QWEN tag=$TAG"

CUDA_VISIBLE_DEVICES="$GPU" \
/home/yschoi/.conda/envs/dna_hashing/bin/python train_siglip2.py \
    --tag "$TAG" \
    --dataset Flickr25k --setting 1 \
    --dataset_dir /home/yschoi/GroundedDNA/dataset \
    --num_devices 0 \
    -bs 64 -e 60 \
    --proj_lr 1e-3 \
    --num_workers 16 \
    --qwen_text_cache_path "$QWEN" \
    --siglip2_feature_cache_dir "$CACHE" \
    --backbone_type clip \
    --codebook_size 64 \
    --c_global_source siglip2_global \
    --disable_global_gate \
    --router_type sinkhorn \
    --sinkhorn_epsilon_init 1.0 --sinkhorn_epsilon_final 0.1 \
    --routing_adaptive_topp \
    --routing_adaptive_topp_min 0.5 \
    --routing_adaptive_topp_max 0.9 \
    --codon_residual_gamma 0.3 \
    --use_paired_aug_ntxent \
    --lambda_ntxent 1.0 \
    --ntxent_temperature 0.3 \
    --ntxent_mode per_codebook \
    --ntxent_dynamic_tau \
    --ntxent_dynamic_tau_alpha 0.3 \
    --lambda_hash 1.0 --lambda_hash_hard 0.5 \
    --lambda_hash_type mse \
    --lambda_wasserstein 0.05 \
    --lambda_vq 0.25 --lambda_quant 0.05 \
    --lambda_anchor 0.05 --lambda_dna 0.05 --lambda_bu 0.02 \
    --eval_every 10 \
    --dna_distance_mode base \
    -ev -s 2>&1 | tee "$LOG"
