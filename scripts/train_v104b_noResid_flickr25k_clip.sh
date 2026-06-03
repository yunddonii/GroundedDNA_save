#!/usr/bin/env bash
# v104b_noResid = v104b recipe + --codon_residual_gamma 0.0 (no residual injection).
#
# v104b baseline: mAP 0.7581, DNA-uniq 0.338 (DNA-axis champion).
# Hypothesis: removing residual makes codon output a deterministic function
# of codeword_idx ALONE. DNA unique should converge to cb-tuple unique
# (≈0.540 for v104b). If cb-tuple stays at ≈0.54, DNA unique should JUMP
# from 0.338 to 0.54 (+60 %). On the other hand, encoder training may
# suffer from sparser gradient signal.
#
# Usage: bash scripts/train_v104b_noResid_flickr25k_clip.sh <GPU_ID>
set -eu

GPU="${1:-4}"
CACHE="${CACHE:-./cache/flickr25k_clip_v4plus}"
QWEN="${QWEN:-./cache/flickr25k_qwen_v4.jsonl}"
TAG="${TAG:-v104b_noResid_v103a_siglipCosTopk01}"
LOG="logs/${TAG}.log"
mkdir -p logs

echo "[run-v104b-noResid] GPU=$GPU cache=$CACHE qwen=$QWEN tag=$TAG"

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
