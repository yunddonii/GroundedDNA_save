#!/usr/bin/env bash
# mscoco_v162b = mscoco_v160b + grounded_text_routing (single delta).
#
# Cross-dataset port of Flickr v162b_fix (Flickr P@1/P@10/B2 champion:
# P@1 0.9250, P@10 0.9214, B2 0.102, mAP +0.0075 over v160b base).
# Tests whether the "grounded routing sharpens retrieval on the full
# UOT+CIBHash+Eq.(8) stack" effect replicates on MSCOCO.
#
# MSCOCO references (qwen3 v4_trainset caption, ~8 % coverage):
#   mscoco_v160b (multi-axis champion):   mAP 0.6134, P@1 0.897, DNA 0.119, NMI 0.726, B2 0.166, dead 0.000
#   mscoco_v160h (K=256 Pareto champion): mAP 0.6028, P@1 0.9022, DNA 0.155, cbT 0.343, NMI 0.715, B2 0.179
#   mscoco_v162b (THIS RUN):              TBD
#
# Single delta vs mscoco_v160b:
#   cache: mscoco_clip_v4plus -> mscoco_clip_v4plus_tokens
#   +flags: --grounded_text_routing --grounded_text_k_t 5 --grounded_text_eps 0.05
#           --grounded_text_stage1_sg --grounded_text_skip_global
#
# Predicted (Flickr v162b_fix pattern):
#   mAP:    0.6134 -> 0.62-0.63 (mAP gain if v160b stack pattern replicates)
#   P@1:    0.897  -> 0.90-0.92
#   DNA:    0.119  -> 0.09-0.12 (small decrease)
#
# Usage: bash scripts/train_mscoco_v162b_v160b_groundedTextRouting_qwen3.sh <GPU_ID>
set -eu

GPU="${1:-3}"
CACHE="${CACHE:-./cache/mscoco_clip_v4plus_tokens}"
QWEN="${QWEN:-./cache/mscoco_qwen3_v4_trainset.jsonl}"
WHITEN_NPZ="${WHITEN_NPZ:-${CACHE}/text_whiten.npz}"
WHITEN_GAMMA="${WHITEN_GAMMA:-0.25}"
TAG="${TAG:-mscoco_v162b_v160b_groundedTextRouting_qwen3_K128_partialWhiten_gamma${WHITEN_GAMMA}}"
LOG="logs/${TAG}.log"
mkdir -p logs

if [ ! -f "${CACHE}/text_tokens.f16.npy" ]; then
    echo "[mscoco-v162b] text_tokens.f16.npy missing at ${CACHE}"
    exit 1
fi
if [ ! -f "$WHITEN_NPZ" ]; then
    echo "[mscoco-v162b] whitening .npz missing — building ..."
    /home/yschoi/.conda/envs/dna_hashing/bin/python \
        scripts/build_text_whiten_matrix.py \
        --cache_dir "$CACHE" \
        --out       "$WHITEN_NPZ"
fi

echo "[mscoco-v162b] GPU=$GPU cache=$CACHE k_t=5 eps=0.05 sg=on skip_global=on tag=$TAG"

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
