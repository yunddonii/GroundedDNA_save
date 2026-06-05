#!/usr/bin/env bash
# v116b = v106b + Method 1 (per-image slot-mean removal).
#
# Hypothesis
# ----------
# CLIP text embeddings exhibit high cross-slot cosine similarity (~0.6-0.9)
# even after the per_slot text adapter. Centering each image's local slots
# against the per-image mean of those slots strips the common direction
# shared across captions of the SAME image, sharpening per-slot identity
# for the Sinkhorn router cost.
#
# Recipe vs v106b
# ---------------
#   --text_embed_transform per_image_mean
#   --text_transform_routing_only   (NEW for v116b)
# All other v106b knobs preserved (Sinkhorn-OT codeword-codon bij at the
# codon decoder, gamma=0, paired-aug NtXent per_codebook, hash_target_mode
# siglip_cos, etc.).
#
# Reference (v106b baseline, Flickr25k-CLIP):
#   mAP 0.7407, DNA-uniq 0.347, cb-tuple 0.477, P@1 0.917, NMI 0.604,
#   B1 0.115, B2 0.072.
#
# Usage: bash scripts/train_v114a_perImageMean_onV106b_flickr25k_clip.sh <GPU_ID>
set -eu

GPU="${1:-2}"
CACHE="${CACHE:-./cache/flickr25k_clip_v4plus}"
QWEN="${QWEN:-./cache/flickr25k_qwen_v4.jsonl}"
TAG="${TAG:-v116b_v115a_routingOnly_perImageMean}"
LOG="logs/${TAG}.log"
mkdir -p logs

echo "[run-v116b] GPU=$GPU cache=$CACHE qwen=$QWEN tag=$TAG"

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
    --text_embed_transform per_image_mean \
    --text_transform_routing_only \
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
    --lambda_text_hash 0.0 \
    --lambda_text_hash_ntxent 0.05 \
    --text_hash_ntxent_temperature 0.07 \
    --lambda_wasserstein 0.05 \
    --lambda_vq 0.25 --lambda_quant 0.05 \
    --lambda_anchor 0.05 --lambda_dna 0.05 --lambda_bu 0.02 \
    --eta_base_balance 0.3 \
    --lambda_codeword_codon_sinkhorn 0.0 \
    --codeword_codon_sinkhorn_eps 0.1 \
    --codeword_codon_sinkhorn_iters 30 \
    --eval_every 5 \
    --post_eval_compositional \
    --dna_distance_mode base \
    -ev -s 2>&1 | tee "$LOG"
