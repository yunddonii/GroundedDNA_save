#!/usr/bin/env bash
# v108b = v108a recipe + --text_init_codebook mean
#         (text-supervised codebook initialization, v41 flag).
#
# Background (corrected 2026-06-03)
# ---------------------------------
# All prior v10x runs (including v108a) use text_init_codebook=none, which
# means the codebook is initialized with Normal(0, 1/sqrt(D)) -- pure
# random Gaussian. The text_part_tokens influence the codebook only through
# training dynamics (Sinkhorn router queries + text-related losses + EMA
# updates). They do NOT initialize the codebook entries directly.
#
# This run tests the OPPOSITE end of the comparison: explicitly initialize
# codebooks from train-set text_part_raw (per v41's "mean" mode: K random
# text vectors per codebook).
#
# Hypothesis directions
# ---------------------
# (A) text-init improves NMI / B1 -> text supervision is amplified by good
#     starting point. Contribution 2 (text-supervision -> interpretability)
#     is strengthened.
# (B) random init + proto-cluster already converges to a text-grounded
#     codebook -> text-init is unnecessary; Contribution 2 emerges from
#     training dynamics alone.
# (C) text-init clusters codebooks too close together -> NMI increases
#     (compositional independence harmed). Contribution 1 weakens.
#
# Recipe vs v108a:
#   --text_init_codebook mean   (was 'none')
#   --text_init_subset 4096     (default; 4096 text vectors per codebook)
#   --text_init_seed 42         (reproducibility)
# Everything else from v108a preserved (gamma=0, proto-cluster lambda 0.5,
# VQ regs all OFF, MSE + InfoNCE visual-textual both at 0.05, paired-aug
# NtXent on, router=sinkhorn, hash_target_mode=siglip_cos).
#
# v108a baseline (random init): mAP 0.7382, DNA 0.219, cb-tuple 0.547,
#                                P@1 0.901, NMI 0.575, B1 0.093, B2 0.055.
#
# Usage: bash scripts/train_v108b_textInit_flickr25k_clip.sh <GPU_ID>
set -eu

GPU="${1:-0}"
CACHE="${CACHE:-./cache/flickr25k_clip_v4plus}"
QWEN="${QWEN:-./cache/flickr25k_qwen_v4.jsonl}"
TAG="${TAG:-v108b_v108a_textInitMean}"
LOG="logs/${TAG}.log"
mkdir -p logs

echo "[run-v108b] GPU=$GPU cache=$CACHE qwen=$QWEN tag=$TAG"

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
    --lambda_hash 0.0 \
    --lambda_hash_hard 0.0 \
    --lambda_hash_type mse \
    --hash_target_mode siglip_cos \
    --lambda_text_hash 0.05 \
    --lambda_text_hash_ntxent 0.05 \
    --text_hash_ntxent_temperature 0.07 \
    --lambda_wasserstein 0.05 \
    --lambda_vq 0.0 \
    --lambda_quant 0.0 \
    --lambda_anchor 0.0 \
    --lambda_dna 0.0 \
    --lambda_bu 0.0 \
    --eta_base_balance 0.3 \
    --lambda_codeword_codon_sinkhorn 0.0 \
    --lambda_proto_cluster_cos 0.5 \
    --proto_cluster_cos_tau 0.1 \
    --text_init_codebook mean \
    --text_init_subset 4096 \
    --text_init_seed 42 \
    --eval_every 5 \
    --post_eval_compositional \
    --dna_distance_mode base \
    -ev -s 2>&1 | tee "$LOG"
