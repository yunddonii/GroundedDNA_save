#!/usr/bin/env bash
# MSCOCO champion + codeword-codon Sinkhorn disjointness regularizer.
#
# Single delta vs scripts/train_mscoco_v180B_textHashOnly.sh:
#   --lambda_codeword_codon_sinkhorn 0.0 -> 0.1
#
# Motivation: current MSCOCO champion (v180B, mAP 0.6214, AUC-PR 0.0734)
# loses AUC-PR to CIBHash 0.0749 by -0.002. Root cause hypothesis =
# codeword-codon collision (DB-unique 0.223 vs CIBHash 0.742). This
# regularizer forces distinct codewords within a slot to decode to distinct
# codons (Sinkhorn bijection potential = 64 per slot for K=128 = 2x pigeonhole).
#
# Expected: DB-unique 0.223 -> 0.35+, P@1 0.9164 -> 0.92+, AUC-PR up.
# Trade-off risk: mAP possibly -0.005 (semantic clustering slightly relaxed).
#
# Verdict criterion: adopt IF AUC-PR beats CIBHash 0.0749 AND mAP within -0.005.
#
# Usage: bash scripts/train_mscoco_v180B_ccs01_clip.sh <GPU_ID>
set -eu

GPU="${1:-4}"
CACHE="${CACHE:-./cache/mscoco_clip_v5b_FAIRrankL8K3}"
QWEN="${QWEN:-./cache/mscoco_qwen3_v5b_trainset.jsonl}"
WHITEN_NPZ="${WHITEN_NPZ:-${CACHE}/text_whiten.npz}"
WHITEN_GAMMA="${WHITEN_GAMMA:-0.25}"
TAG="${TAG:-mscoco_v180B_ccs01_K128_partialWhiten_gamma${WHITEN_GAMMA}}"
LOG="logs/${TAG}.log"
mkdir -p logs

if [ ! -f "$WHITEN_NPZ" ]; then
    echo "[mscoco-v180B-ccs01] whitening .npz missing"
    exit 1
fi

echo "[mscoco-v180B-ccs01] GPU=$GPU cache=$CACHE tag=$TAG"

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
    --lambda_text_code_kl 0.10 \
    --text_code_kl_tau_v 0.1 \
    --text_code_kl_tau_t 0.07 \
    --text_code_kl_conf_threshold 0.2 \
    --text_code_kl_skip_global \
    --text_hash_ntxent_skip_global \
    --eval_cache_dir ./cache/mscoco_clip_v5b \
    -ev -s 2>&1 | tee "$LOG"
