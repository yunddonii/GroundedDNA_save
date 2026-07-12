#!/usr/bin/env bash
# v185 MSCOCO — BIDIRECTIONAL token pruning applied to MSCOCO champion base.
#
# Structural consistency (v181 + bidirectional): 3-dataset champion architecture
# now unified as:
#   * --per_slot_text_adapter
#   * --codon_residual_gamma 0.0
#   * --text_code_kl_skip_global + --text_hash_ntxent_skip_global (2 skip flags)
#   * --bidirectional_token_prune (NEW)
#   * --bidirectional_token_prune_visual_ratio 0.5
#   * --bidirectional_token_prune_text_ratio 0.5
# Loss weights differ per dataset (allowed per v181 principle: architecture and
# skip-flag structure identical, weights dataset-tuned).
#
# Delta vs scripts/train_mscoco_v180B_textHashOnly.sh:
#   + --bidirectional_token_prune
#   + --bidirectional_token_prune_visual_ratio 0.5
#   + --bidirectional_token_prune_text_ratio   0.5
# (ccs=0.1 on MSCOCO was DISCARDED — kept at 0.0.)
#
# Precondition: cache/mscoco_clip_v5b_FAIRrankL8K3 has text_tokens.f16.npy +
# text_token_mask.bool.npy (verified — donor symlinks from
# cache/mscoco_clip_v5b_tokens/).
#
# Usage: bash scripts/train_mscoco_v185_bidirTokenPrune05_clip.sh <GPU_ID>
set -eu

GPU="${1:-4}"
CACHE="${CACHE:-./cache/mscoco_clip_v5b_FAIRrankL8K3}"
QWEN="${QWEN:-./cache/mscoco_qwen3_v5b_trainset.jsonl}"
WHITEN_NPZ="${WHITEN_NPZ:-${CACHE}/text_whiten.npz}"
WHITEN_GAMMA="${WHITEN_GAMMA:-0.25}"
BI_V="${BI_V:-0.5}"
BI_T="${BI_T:-0.5}"
TAG="${TAG:-mscoco_v185_bidir_v${BI_V}_t${BI_T}_K128_partialWhiten_gamma${WHITEN_GAMMA}}"
LOG="logs/${TAG}.log"
mkdir -p logs

if [ ! -f "$WHITEN_NPZ" ]; then
    echo "[v185-mscoco] whitening .npz missing"
    exit 1
fi

echo "[v185-mscoco] GPU=$GPU cache=$CACHE bi_v=$BI_V bi_t=$BI_T tag=$TAG"

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
    --sinkhorn_lambda_a 1.0 \
    --sinkhorn_lambda_b 1.0 \
    --routing_adaptive_topp \
    --routing_adaptive_topp_min 0.3 \
    --routing_adaptive_topp_max 0.7 \
    --codon_residual_gamma 0.0 \
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
    --lambda_text_hash_ntxent 0.10 \
    --text_hash_ntxent_temperature 0.07 \
    --text_hash_ntxent_mode per_codebook \
    --lambda_wasserstein 0.05 \
    --lambda_vq 0.25 --lambda_quant 0.05 \
    --lambda_xmodal_commit 0.10 \
    --lambda_anchor 0.05 --lambda_dna 0.05 --lambda_bu 0.02 \
    --eta_base_balance 0.3 \
    --lambda_codeword_codon_sinkhorn 0.0 \
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
    --lambda_text_code_kl 0.10 \
    --text_code_kl_tau_v 0.1 \
    --text_code_kl_tau_t 0.07 \
    --text_code_kl_conf_threshold 0.2 \
    --text_code_kl_skip_global \
    --text_hash_ntxent_skip_global \
    --bidirectional_token_prune \
    --bidirectional_token_prune_visual_ratio "$BI_V" \
    --bidirectional_token_prune_text_ratio "$BI_T" \
    --eval_cache_dir ./cache/mscoco_clip_v5b \
    -ev -s 2>&1 | tee "$LOG"
