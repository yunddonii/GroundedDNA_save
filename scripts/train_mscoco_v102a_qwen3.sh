#!/usr/bin/env bash
# mscoco_v102a-Qwen3 = mscoco port of v102a (Flickr) recipe + Qwen3-VL captions
#   - lambda_hash = 0 (no siglip_cos pairwise signal — was collapse cause on Flickr)
#   - lambda_text_hash_ntxent 0.05 (additive image-text DNA InfoNCE)
#   - paired-aug NtXent per_codebook + dynamic-tau
#   - hash_target_mode siglip_cos (HARD INVARIANT)
#   - base_balance loss = KL(uniform || p_bar) (code-level change)
#   - K=128 (mscoco_v92a SOTA codebook size)
#   - qwen_text_cache: mscoco_qwen3_v4_trainset.jsonl (Qwen3-VL-8B-Instruct V4 captions)
#
# Usage: bash scripts/train_mscoco_v102a_qwen3.sh <GPU_ID>
set -eu

GPU="${1:-1}"
CACHE="${CACHE:-./cache/mscoco_clip_v4plus}"
QWEN="${QWEN:-./cache/mscoco_qwen3_v4_trainset.jsonl}"
TAG="${TAG:-mscoco_v102a_qwen3_v101cRecipe_baseBalanceKL}"
LOG="logs/${TAG}.log"
mkdir -p logs

echo "[run-mscoco-v102a] GPU=$GPU cache=$CACHE qwen=$QWEN tag=$TAG"

CUDA_VISIBLE_DEVICES="$GPU" \
/home/yschoi/.conda/envs/dna_hashing/bin/python train_siglip2.py \
    --tag "$TAG" \
    --dataset MSCOCO --setting 1 \
    --dataset_dir /home/yschoi/GroundedDNA/dataset \
    --num_devices 0 \
    -bs 64 -e 60 \
    --proj_lr 1e-3 \
    --num_workers 4 \
    --qwen_text_cache_path "$QWEN" \
    --siglip2_feature_cache_dir "$CACHE" \
    --backbone_type clip \
    --codebook_size 128 \
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
    --lambda_hash 0.0 \
    --lambda_hash_hard 0.0 \
    --lambda_hash_type mse \
    --hash_target_mode siglip_cos \
    --lambda_text_hash 0.05 \
    --lambda_text_hash_ntxent 0.05 \
    --text_hash_ntxent_temperature 0.07 \
    --lambda_wasserstein 0.05 \
    --lambda_vq 0.25 --lambda_quant 0.05 \
    --lambda_anchor 0.05 --lambda_dna 0.05 --lambda_bu 0.02 \
    --eta_base_balance 1.0 \
    --eval_every 5 \
    --post_eval_compositional \
    --dna_distance_mode base \
    -ev -s 2>&1 | tee "$LOG"
