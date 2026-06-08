#!/usr/bin/env bash
# v121a = v119a (CIBHash per-codebook K=128) + SwAV-style swapped
#         balanced codeword-assignment loss (additive, slot 1..5 only,
#         C_global excluded).
#
# Motivation. v120c/g/h variable-separation revealed that:
#   - thNX ALONE (v120h) recovered dead from 0.168 -> 0.000 but
#     destroyed P@1 (0.906 -> 0.852, the worst run yet).
#   - bij ALONE (v120g) made dead WORSE (0.168 -> 0.287) and dropped
#     mAP.
#   - v120c (bij + thNX) struck a partial balance (dead 0.001, P@1
#     0.893) but no run reached v106b's P@1 0.917 at K=128.
# Hypothesis. The bit-level CIBHash NtXent winner-take-all collapses
# 1-2 codewords per codebook (per-cb-unique stuck at ~0.014 = ~2/128).
# A codeword-level supervision that ENCOURAGES BATCH-LEVEL CODEWORD
# USAGE BALANCE — without forcing global codon distribution like the
# Sinkhorn bij does — should:
#   (a) recover dead codewords by balancing usage,
#   (b) preserve mAP by keeping fine-grained discriminativity per
#       codeword,
#   (c) preserve or recover P@1 by maintaining codebook expressiveness
#       (vs. v120c's bij which forces all codewords to participate
#       uniformly in codon mass, diluting top-1 specificity).
#
# Recipe vs v119a:
#   --lambda_swav_assign 0.05         (NEW additive loss)
#   --swav_assign_tau 0.1
#   --swav_sinkhorn_eps 0.05
#   --swav_sinkhorn_iters 3
#   (--swav_assign_include_global defaults to False; slot 0 excluded)
# Everything else identical to v119a.
#
# Reference (v119a):
#   mAP 0.7620  P@1 0.906  P@10 0.895  DNA 0.246  dead 0.168  NMI 0.596
# Reference (v120h thNX ONLY = best alternative dead-recovery):
#   mAP 0.7673  P@1 0.852 ⚠  DNA 0.227  dead 0.000  NMI 0.624
#
# Usage: bash scripts/train_v121a_v119a_swavAssign_K128_flickr25k_clip.sh <GPU_ID>
set -eu

GPU="${1:-0}"
CACHE="${CACHE:-./cache/flickr25k_clip_v4plus}"
QWEN="${QWEN:-./cache/flickr25k_qwen_v4.jsonl}"
WHITEN_NPZ="${WHITEN_NPZ:-${CACHE}/text_whiten.npz}"
WHITEN_GAMMA="${WHITEN_GAMMA:-0.25}"
TAG="${TAG:-v121a_v119a_swavAssign_cibhash_K128_partialWhiten_gamma${WHITEN_GAMMA}}"
LOG="logs/${TAG}.log"
mkdir -p logs

echo "[run-v121a] GPU=$GPU cache=$CACHE whiten=$WHITEN_NPZ gamma=$WHITEN_GAMMA K=128 swav-assign tag=$TAG"

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
    --codebook_size 128 \
    --c_global_source siglip2_global \
    --per_slot_text_adapter \
    --global_gate_init_logit 4.595 \
    --router_type sinkhorn \
    --sinkhorn_epsilon_init 1.0 --sinkhorn_epsilon_final 0.1 \
    --routing_adaptive_topp \
    --routing_adaptive_topp_min 0.5 \
    --routing_adaptive_topp_max 0.9 \
    --codon_residual_gamma 0.0 \
    --text_embed_transform partial_whiten \
    --text_whiten_npz "$WHITEN_NPZ" \
    --text_whiten_gamma "$WHITEN_GAMMA" \
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
    --lambda_text_hash 0.0 \
    --lambda_text_hash_ntxent 0.0 \
    --text_hash_ntxent_temperature 0.07 \
    --lambda_wasserstein 0.05 \
    --lambda_vq 0.25 --lambda_quant 0.05 \
    --lambda_anchor 0.05 --lambda_dna 0.05 --lambda_bu 0.02 \
    --eta_base_balance 0.3 \
    --lambda_codeword_codon_sinkhorn 0.0 \
    --lambda_cibhash_ntxent 1.0 \
    --lambda_cibhash_kl 0.001 \
    --cibhash_temperature 0.3 \
    --lambda_swav_assign 0.05 \
    --swav_assign_tau 0.1 \
    --swav_sinkhorn_eps 0.05 \
    --swav_sinkhorn_iters 3 \
    --eval_every 5 \
    --post_eval_compositional \
    --dna_distance_mode base \
    -ev -s 2>&1 | tee "$LOG"
