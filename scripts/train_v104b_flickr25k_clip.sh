#!/usr/bin/env bash
# v104b (beta) = v103a recipe + RE-ENABLE BOTH pairwise paths with
# --hash_target_mode siglip_cos_topk --siglip_cos_pos_rate 0.1.
#
# Hypothesis: the v101 batch showed siglip_cos pairwise (smooth real-
# valued S) causes collapse because it is BOTH under-discriminative
# (loses 0/1 hardness) AND over-aligning (high-cos pairs pulled onto
# the same codeword). siglip_cos_topk converts the target back to a
# sharp 0/1 form by keeping only the top 10% positives per row and
# treating the rest as negatives -- this is the unsupervised analogue
# of the v95a jaccard regime, which DID work well at mAP 0.8476
# (although tag-supervised). Test whether sharp unsupervised S
# recovers the v95a recipe's instance-discrimination power without
# tag supervision.
#
# Recipe diff vs v103a:
#   --lambda_hash       0.0 -> 1.0  (re-enable soft pairwise on top-k S)
#   --lambda_hash_hard  0.0 -> 0.5  (re-enable hard pairwise on top-k S)
#   --hash_target_mode  siglip_cos -> siglip_cos_topk
#   --siglip_cos_pos_rate (default 0.2) -> 0.1
# All other flags identical (KL base_balance eta=0.3, etc.).
#
# Usage: bash scripts/train_v104b_flickr25k_clip.sh <GPU_ID>
set -eu

GPU="${1:-4}"
CACHE="${CACHE:-./cache/flickr25k_clip_v4plus}"
QWEN="${QWEN:-./cache/flickr25k_qwen_v4.jsonl}"
TAG="${TAG:-v104b_v103a_siglipCosTopk01}"
LOG="logs/${TAG}.log"
mkdir -p logs

echo "[run-v104b] GPU=$GPU cache=$CACHE qwen=$QWEN tag=$TAG"

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
    --codon_residual_gamma 0.3 \
    --use_paired_aug_ntxent \
    --lambda_ntxent 1.0 \
    --ntxent_temperature 0.3 \
    --ntxent_mode per_codebook \
    --ntxent_dynamic_tau \
    --ntxent_dynamic_tau_alpha 0.3 \
    --lambda_hash 1.0 \
    --lambda_hash_hard 0.5 \
    --lambda_hash_type mse \
    --hash_target_mode siglip_cos_topk \
    --siglip_cos_pos_rate 0.1 \
    --lambda_text_hash 0.05 \
    --lambda_text_hash_ntxent 0.05 \
    --text_hash_ntxent_temperature 0.07 \
    --lambda_wasserstein 0.05 \
    --lambda_vq 0.25 --lambda_quant 0.05 \
    --lambda_anchor 0.05 --lambda_dna 0.05 --lambda_bu 0.02 \
    --eta_base_balance 0.3 \
    --eval_every 5 \
    --post_eval_compositional \
    --dna_distance_mode base \
    -ev -s 2>&1 | tee "$LOG"
