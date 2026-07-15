#!/usr/bin/env bash
# v185 CIFAR10 — BIDIRECTIONAL token pruning applied to CIFAR10 champion base
# (which already includes lambda_codeword_codon_sinkhorn = 0.1).
#
# Structural consistency (v181 + bidirectional): 3-dataset champion architecture
# now unified as:
#   * --per_slot_text_adapter
#   * --codon_residual_gamma 0.0
#   * --text_code_kl_skip_global + --text_hash_ntxent_skip_global
#   * --bidirectional_token_prune (NEW)
#   * --bidirectional_token_prune_visual_ratio 0.5
#   * --bidirectional_token_prune_text_ratio 0.5
# Loss weight `lambda_codeword_codon_sinkhorn` remains 0.1 on CIFAR10 (K=64,
# bijection possible — kept from prior champion) but is 0.0 on Flickr/MSCOCO
# (K=128, pigeonhole makes bijection impossible). Per v181 principle:
# architecture identical, weights dataset-tuned.
#
# Delta vs scripts/train_cifar10_flickrChamp_ccs01_clip.sh:
#   + --bidirectional_token_prune
#   + --bidirectional_token_prune_visual_ratio 0.5
#   + --bidirectional_token_prune_text_ratio   0.5
#
# Precondition: cache/cifar10_clip must have text_tokens.f16.npy +
# text_token_mask.bool.npy — extracted by extract_clip_text_tokens_cifar10.py
# (add-on run 2026-07-12).
#
# Usage: bash scripts/train_cifar10_v185_bidirTokenPrune05_ccs01_clip.sh <GPU_ID>
set -eu

GPU="${1:-5}"
CACHE="${CACHE:-./cache/cifar10_clip}"
QWEN="${QWEN:-./cache/cifar10_qwen.jsonl}"
WHITEN_NPZ="${WHITEN_NPZ:-${CACHE}/text_whiten.npz}"
WHITEN_GAMMA="${WHITEN_GAMMA:-0.25}"
K="${K:-64}"
BI_V="${BI_V:-0.5}"
BI_T="${BI_T:-0.5}"
TAG="${TAG:-cifar10_v185_bidir_v${BI_V}_t${BI_T}_ccs01_K${K}_partialWhiten_g${WHITEN_GAMMA}}"
LOG="logs/${TAG}.log"
mkdir -p logs

if [ ! -f "${CACHE}/text_tokens.f16.npy" ] || [ ! -f "${CACHE}/text_token_mask.bool.npy" ]; then
    echo "[v185-cifar10] MISSING ${CACHE}/text_tokens.f16.npy or text_token_mask.bool.npy"
    echo "Run: python extract_clip_text_tokens_cifar10.py --cache_dir ${CACHE} --qwen_cache_path ${QWEN} --text_max_length 32"
    exit 1
fi

echo "[v185-cifar10] GPU=$GPU cache=$CACHE K=$K bi_v=$BI_V bi_t=$BI_T tag=$TAG"

CUDA_VISIBLE_DEVICES="$GPU" \
/home/yschoi/.conda/envs/dna_hashing/bin/python train_siglip2.py \
    --tag "$TAG" \
    --dataset CIFAR10 --setting 1 \
    --dataset_dir /home/yschoi/GroundedDNA/dataset \
    --num_devices 0 \
    -bs 64 -e 60 \
    --proj_lr 1e-3 \
    --num_workers 4 \
    --qwen_text_cache_path "$QWEN" \
    --siglip2_feature_cache_dir "$CACHE" \
    --backbone_type clip \
    --codebook_size "$K" \
    --c_global_source siglip2_global \
    --per_slot_text_adapter \
    --global_gate_init_logit 4.595 \
    --router_type sinkhorn \
    --sinkhorn_epsilon_init 1.0 --sinkhorn_epsilon_final 0.1 \
    --sinkhorn_lambda_a 1.0 \
    --sinkhorn_lambda_b 1.0 \
    --routing_adaptive_topp \
    --routing_adaptive_topp_min 0.3 \
    --routing_adaptive_topp_max 0.7 \
    --codon_residual_gamma 0.0 \
    ${DISABLE_TEXT:+--disable_text_supervision} \
    --num_codons_per_codebook "${NUM_CODONS:-3}" \
    --text_embed_transform partial_whiten \
    --text_whiten_npz "$WHITEN_NPZ" \
    --text_whiten_gamma "$WHITEN_GAMMA" \
    --use_paired_aug_ntxent \
    --lambda_ntxent 0.0 \
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
    --text_hash_ntxent_mode per_codebook \
    --lambda_wasserstein 0.15 \
    --lambda_vq 0.25 --lambda_quant 0.05 \
    --lambda_xmodal_commit 0.05 \
    --lambda_anchor 0.05 --lambda_dna 0.05 --lambda_bu 0.02 \
    --eta_base_balance 0.3 \
    --lambda_codeword_codon_sinkhorn 0.1 \
    --lambda_cibhash_ntxent 1.0 \
    --lambda_cibhash_kl 0.001 \
    --cibhash_mode per_codebook \
    --cibhash_temperature 0.3 \
    --cibhash_dynamic_tau \
    --cibhash_dynamic_tau_alpha 0.3 \
    --cibhash_ntxent_continuous \
    --cibhash_ntxent_source visual_token \
    --eval_every 5 \
    --post_eval_compositional \
    --dna_distance_mode base \
    --lambda_text_code_kl 0.05 \
    --text_code_kl_tau_v 0.1 \
    --text_code_kl_tau_t 0.07 \
    --text_code_kl_conf_threshold 0.2 \
    --text_code_kl_skip_global \
    --text_hash_ntxent_skip_global \
    --bidirectional_token_prune \
    --bidirectional_token_prune_visual_ratio "$BI_V" \
    --bidirectional_token_prune_text_ratio "$BI_T" \
    --eval_cache_dir "$CACHE" \
    -ev -s 2>&1 | tee "$LOG"
