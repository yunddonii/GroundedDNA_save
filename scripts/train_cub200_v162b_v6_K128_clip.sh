#!/usr/bin/env bash
# CUB-200 port of Flickr25k-CLIP v162b = v160b + grounded_text_routing
# (single delta vs scripts/train_cub200_v160b_v6_K128_clip.sh).
#
# Differences vs the CUB v160b script:
#   - cache dir -> cub200_clip_v6plus_tokens (adds text_tokens.f16.npy +
#     text_token_mask.bool.npy; visual_* symlinked from cub200_clip_v6plus).
#   - flags added:
#       --grounded_text_routing
#       --grounded_text_k_t 5
#       --grounded_text_eps 0.05
#       --grounded_text_stage1_sg
#       --grounded_text_skip_global
#
# Usage: bash scripts/train_cub200_v162b_v6_K128_clip.sh <GPU_ID>
set -eu

GPU="${1:-1}"
CACHE="${CACHE:-./cache/cub200_clip_v6plus_tokens}"
QWEN="${QWEN:-./cache/cub200_qwen_v6_trainset.jsonl}"
WHITEN_NPZ="${WHITEN_NPZ:-${CACHE}/text_whiten.npz}"
WHITEN_GAMMA="${WHITEN_GAMMA:-0.25}"
TAG="${TAG:-cub200_v162b_v6_groundedTextRouting_K128_partialWhiten_gamma${WHITEN_GAMMA}}"
LOG="logs/${TAG}.log"
mkdir -p logs

if [ ! -f "${CACHE}/text_tokens.f16.npy" ]; then
    echo "[cub-v162b] text_tokens.f16.npy missing at ${CACHE}"
    echo "[cub-v162b] run: python extract_clip_text_token_features.py \\"
    echo "         --qwen_cache ${QWEN} \\"
    echo "         --donor_dir  ./cache/cub200_clip_v6plus \\"
    echo "         --out_dir    ${CACHE}"
    exit 1
fi
if [ ! -f "$WHITEN_NPZ" ]; then
    echo "[cub-v162b] whitening .npz missing — building from token cache ..."
    /home/yschoi/.conda/envs/dna_hashing/bin/python \
        scripts/build_text_whiten_matrix.py \
        --cache_dir "$CACHE" \
        --out       "$WHITEN_NPZ"
fi

echo "[run-cub-v162b] GPU=$GPU cache=$CACHE k_t=5 eps=0.05 sg=on skip_global=on tag=$TAG"

CUDA_VISIBLE_DEVICES="$GPU" \
/home/yschoi/.conda/envs/dna_hashing/bin/python train_siglip2.py \
    --tag "$TAG" \
    --dataset CUB_200 --setting 1 \
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
    --lambda_text_hash_ntxent 0.05 \
    --text_hash_ntxent_temperature 0.07 \
    --text_hash_ntxent_mode per_codebook \
    --lambda_wasserstein 0.05 \
    --lambda_vq 0.25 --lambda_quant 0.05 \
    --lambda_xmodal_commit 0.025 \
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
    --lambda_text_code_kl 0.02 \
    --text_code_kl_tau_v 0.1 \
    --text_code_kl_tau_t 0.07 \
    --text_code_kl_conf_threshold 0.2 \
    --text_code_kl_skip_global \
    --grounded_text_routing \
    --grounded_text_k_t 5 \
    --grounded_text_eps 0.05 \
    --grounded_text_stage1_sg \
    --grounded_text_skip_global \
    -ev -s 2>&1 | tee "$LOG"
