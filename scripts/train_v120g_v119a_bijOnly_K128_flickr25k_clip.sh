#!/usr/bin/env bash
# v120g = v119a (CIBHash per-codebook K=128) + Sinkhorn codeword-codon
#         bijection (lambda 0.1) ONLY. text_hash_ntxent stays OFF.
#
# Variable-separation pair with v120c (bij + thNX) and v120h (thNX only).
# Goal: isolate whether the bij ALONE is responsible for the DNA-uniq
# recovery (v120c: 0.246 -> 0.344) and the P@1 drop (v119a 0.906 ->
# v120c 0.893), or if the thNX is contributing to either or both.
#
# Recipe vs v119a:
#   --lambda_codeword_codon_sinkhorn 0.1     (NEW: bij ON)
#   --codeword_codon_sinkhorn_eps 0.1
#   --codeword_codon_sinkhorn_iters 30
# Everything else identical to v119a.
#
# Reference (v119a):     mAP 0.7620, P@1 0.906, DNA 0.246, dead 0.168, NMI 0.596
# Reference (v120c bij+thNX): mAP 0.7393, P@1 0.893, DNA 0.344, dead 0.001, NMI 0.619
#
# Hypothesis: bij ALONE recovers most of v120c's DNA gain (since bij is
# the historical DNA-champion mechanism from v106b) and accounts for
# most of the P@1 regression as well (since bij forces codebook
# usage uniformity that may dilute top-1 specificity).
#
# Usage: bash scripts/train_v120g_v119a_bijOnly_K128_flickr25k_clip.sh <GPU_ID>
set -eu

GPU="${1:-3}"
CACHE="${CACHE:-./cache/flickr25k_clip_v4plus}"
QWEN="${QWEN:-./cache/flickr25k_qwen_v4.jsonl}"
WHITEN_NPZ="${WHITEN_NPZ:-${CACHE}/text_whiten.npz}"
WHITEN_GAMMA="${WHITEN_GAMMA:-0.25}"
TAG="${TAG:-v120g_v119a_bijOnly_cibhash_K128_partialWhiten_gamma${WHITEN_GAMMA}}"
LOG="logs/${TAG}.log"
mkdir -p logs

echo "[run-v120g] GPU=$GPU cache=$CACHE whiten=$WHITEN_NPZ gamma=$WHITEN_GAMMA K=128 cibhash+bij tag=$TAG"

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
    --lambda_codeword_codon_sinkhorn 0.1 \
    --codeword_codon_sinkhorn_eps 0.1 \
    --codeword_codon_sinkhorn_iters 30 \
    --lambda_cibhash_ntxent 1.0 \
    --lambda_cibhash_kl 0.001 \
    --cibhash_temperature 0.3 \
    --eval_every 5 \
    --post_eval_compositional \
    --dna_distance_mode base \
    -ev -s 2>&1 | tee "$LOG"
