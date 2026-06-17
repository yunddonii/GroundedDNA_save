#!/usr/bin/env bash
# mscoco_v160b_localResid = mscoco_v160b + local_residual_quant (γ=1.0) +
#                           local_residual_text + local_residual_detach_global
# (single delta cluster: enables the local-residual path that removes the
#  C_0 (global slot) projection from EVERY local visual+text embedding
#  immediately before VQ).
#
# Motivation. MSCOCO Qwen3 v4 captions show C_0↔local mean cos = 0.637
# (vs Flickr's 0.605) and local↔local mean cos = 0.666 (vs Flickr's
# 0.592). The activity caption in particular re-mentions subject /
# scene / color at ~0.70 cosine. local_residual_text strips the C_0
# component from text_part_tokens[1:] and the per-image global
# projection from semantic_visual_tokens[1:], forcing each local
# codebook to encode only the codebook-specific residual signal.
#
# Reference (mscoco qwen3 v4_trainset):
#   mscoco_v160b (γ=0):     mAP 0.6134, P@1 0.897, DNA 0.119, cbT 0.205,
#                           NMI 0.726, L↔L 0.755, B2 0.166, collision 1.72×
#   mscoco_v160b_whitenG1   (in flight on GPU 4)
#   mscoco_v160b_localResid (THIS RUN, γ=1.0)
#
# Single delta (cluster) vs mscoco_v160b:
#   --local_residual_quant
#   --local_residual_gamma 1.0
#   --local_residual_text
#   (--local_residual_detach_global is default True, included explicitly)
#
# Predicted (analysis-driven):
#   mAP:    0.6134 -> 0.59-0.62 (small risk; v122 era found γ=1.0 mildly
#                                 hurtful on Flickr; MSCOCO different)
#   DNA:    0.119  -> 0.14-0.18 (caption redundancy ↓ -> codon collision ↓)
#   cbT:    0.205  -> 0.25-0.35 (codeword diversity ↑)
#   NMI:    0.726  -> 0.66-0.72 (C_0 axis removed -> NMI ↓ better)
#   L↔L:    0.755  -> 0.68-0.73
#   collision ratio: 1.72× -> 1.40-1.60×
#
# Usage: bash scripts/train_mscoco_v160b_localResid_qwen3.sh <GPU_ID>
set -eu

GPU="${1:-5}"
CACHE="${CACHE:-./cache/mscoco_clip_v4plus}"
QWEN="${QWEN:-./cache/mscoco_qwen3_v4_trainset.jsonl}"
WHITEN_NPZ="${WHITEN_NPZ:-${CACHE}/text_whiten.npz}"
WHITEN_GAMMA="${WHITEN_GAMMA:-0.25}"
RES_GAMMA="${RES_GAMMA:-1.0}"
TAG="${TAG:-mscoco_v160b_localResid_g${RES_GAMMA}_v150b_xmodalCommit_0p025_qwen3_K128_partialWhiten_gamma${WHITEN_GAMMA}}"
LOG="logs/${TAG}.log"
mkdir -p logs

if [ ! -f "$WHITEN_NPZ" ]; then
    echo "[mscoco-v160b-localResid] whitening .npz missing — building ..."
    /home/yschoi/.conda/envs/dna_hashing/bin/python \
        scripts/build_text_whiten_matrix.py \
        --cache_dir "$CACHE" \
        --out       "$WHITEN_NPZ"
fi

echo "[mscoco-v160b-localResid] GPU=$GPU cache=$CACHE res_gamma=$RES_GAMMA tag=$TAG"

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
    --local_residual_quant \
    --local_residual_gamma "$RES_GAMMA" \
    --local_residual_text \
    --local_residual_detach_global \
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
    -ev -s 2>&1 | tee "$LOG"
