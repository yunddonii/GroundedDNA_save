#!/usr/bin/env bash
# v105_head_only = v103a recipe + --codon_full_linear (ONLY architectural change).
#
# Test (i) from the bijection proposal review: does increasing CodonHead
# expressivity ALONE (Linear(d_model=768, 12) view [B, 3, 4], 9× params)
# reduce codeword→DNA codon collisions, without any added bijection loss?
#
# Baseline (v103a): chunk-partition Linear(chunk=256, 4) shared across 3
#   positions. DNA-base unique (DB) = 0.241, mAP 0.7602.
# Hypothesis: increased decoder expressivity allows the codon head to map
#   distinct codewords to distinct codons more easily, reducing collisions.
# Predicted: DNA-base unique 0.27-0.32, mAP within ±0.01 of v103a.
#
# All other flags identical to v103a (genuinely-unsupervised regime):
#   --hash_target_mode siglip_cos, --lambda_hash 0.0, paired-aug NtXent,
#   text-DNA NtXent, KL base-balance with eta=0.3.
#
# Usage: bash scripts/train_v105_head_only_flickr25k_clip.sh <GPU_ID>
set -eu

GPU="${1:-0}"
CACHE="${CACHE:-./cache/flickr25k_clip_v4plus}"
QWEN="${QWEN:-./cache/flickr25k_qwen_v4.jsonl}"
TAG="${TAG:-v105_head_only_v103a_codonFullLinear}"
LOG="logs/${TAG}.log"
mkdir -p logs

echo "[run-v105] GPU=$GPU cache=$CACHE qwen=$QWEN tag=$TAG"

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
    --codon_full_linear \
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
    --eta_base_balance 0.3 \
    --eval_every 5 \
    --post_eval_compositional \
    --dna_distance_mode base \
    -ev -s 2>&1 | tee "$LOG"
