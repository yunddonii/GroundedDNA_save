#!/usr/bin/env bash
# v110a = v107a recipe with --lambda_bu 0.0 (codebook-balance regularizer OFF).
#
# User insight (2026-06-03): in the v107a (proto-cosine) setting, loss_bu
# may be redundant because:
#   - proto_cluster_cos's InfoNCE structure (pull z to assigned codeword,
#     push from others) implicitly encourages each codeword to be used by
#     SOME z's (otherwise it sits in an unpopulated region).
#   - Sinkhorn router's marginal balance already distributes patches
#     across parts.
#   - EMA codebook revival (codebook_revive_threshold=0.01) automatically
#     replaces dead codewords with active samples.
#   - loss_dna's base_balance regularizes codon distribution at the
#     downstream level.
#
# Past evidence: v101d (lambda_bu 0.02 -> 0.5, 25x boost) gave +33x
# unique but -0.022 mAP -- loss_bu can be HARMFUL when too strong.
# Likely contributes minimally at lambda=0.02; test by removing.
#
# Recipe vs v107a:
#   --lambda_bu 0.02 -> 0.0   # turn OFF codebook-balance regularizer
# Everything else from v107a preserved.
#
# Expected outcomes:
#   (A) loss_bu redundant -> v110a matches v107a (mAP 0.7624, DNA 0.213).
#       Conclusion: drop permanently to simplify the recipe.
#   (B) loss_bu had real (small) effect -> minor regression (~0.005-0.010).
#       Conclusion: keep for robustness margin.
#   (C) dead codewords emerge -> codebook collapse without explicit
#       balance enforcement. Conclusion: loss_bu is necessary.
#
# Usage: bash scripts/train_v110a_flickr25k_clip.sh <GPU_ID>
set -eu

GPU="${1:-2}"
CACHE="${CACHE:-./cache/flickr25k_clip_v4plus}"
QWEN="${QWEN:-./cache/flickr25k_qwen_v4.jsonl}"
TAG="${TAG:-v110a_v107a_noBU}"
LOG="logs/${TAG}.log"
mkdir -p logs

echo "[run-v110a] GPU=$GPU cache=$CACHE qwen=$QWEN tag=$TAG"

CUDA_VISIBLE_DEVICES="$GPU" \
/home/yschoi/.conda/envs/dna_hashing/bin/python train_siglip2.py \
    --tag "$TAG" \
    --dataset Flickr25k --setting 1 \
    --dataset_dir /home/yschoi/GroundedDNA/dataset \
    --num_devices 0 \
    -bs 64 -e 60 \
    --proj_lr 1e-3 \
    --num_workers 4 \
    --qwen_text_cache_path "$QWEN" \
    --siglip2_feature_cache_dir "$CACHE" \
    --backbone_type clip \
    --codebook_size 64 \
    --c_global_source siglip2_global \
    --per_slot_text_adapter \
    --global_gate_init_logit 4.595 \
    --router_type sinkhorn \
    --sinkhorn_epsilon_init 1.0 --sinkhorn_epsilon_final 0.1 \
    --routing_adaptive_topp \
    --routing_adaptive_topp_min 0.5 \
    --routing_adaptive_topp_max 0.9 \
    --codon_residual_gamma 0.0 \
    --use_paired_aug_ntxent \
    --lambda_ntxent 1.0 \
    --ntxent_temperature 0.3 \
    --ntxent_mode per_codebook \
    --ntxent_dynamic_tau \
    --ntxent_dynamic_tau_alpha 0.3 \
    --lambda_hash 0.0 \
    --lambda_hash_hard 0.0 \
    --lambda_hash_type mse \
    --hash_target_mode siglip_cos \
    --lambda_text_hash 0.05 \
    --lambda_text_hash_ntxent 0.05 \
    --text_hash_ntxent_temperature 0.07 \
    --lambda_wasserstein 0.05 \
    --lambda_vq 0.25 --lambda_quant 0.05 \
    --lambda_anchor 0.05 --lambda_dna 0.05 --lambda_bu 0.0 \
    --eta_base_balance 0.3 \
    --lambda_codeword_codon_sinkhorn 0.0 \
    --lambda_proto_cluster_cos 0.5 \
    --proto_cluster_cos_tau 0.1 \
    --eval_every 5 \
    --post_eval_compositional \
    --dna_distance_mode base \
    -ev -s 2>&1 | tee "$LOG"
