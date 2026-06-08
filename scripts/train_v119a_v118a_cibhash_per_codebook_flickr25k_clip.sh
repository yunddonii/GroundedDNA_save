#!/usr/bin/env bash
# v119a = v118a base (K=128, partialWhiten γ=0.25) with all text-DNA and
#         hash-MSE losses OFF; replaced by per-codebook CIBHash losses
#         (NtXent on binary hash + symmetric Bernoulli KL on bit-probs).
#
# Keep (per user spec):
#   --lambda_wasserstein 0.05
#   --use_paired_aug_ntxent + --ntxent_dynamic_tau (the "dynamic tau"
#     mechanism on the existing per-codebook NtXent)
#   --lambda_vq 0.25, --lambda_quant 0.05            (essential VQ)
#   --lambda_anchor 0.05, --lambda_dna 0.05, --lambda_bu 0.02 (regularizers)
#
# OFF (per user spec):
#   --lambda_hash 0.0, --lambda_hash_hard 0.0
#   --lambda_text_hash 0.0, --lambda_text_hash_ntxent 0.0
#   --lambda_codeword_codon_sinkhorn 0.0
#
# NEW (v119):
#   --lambda_cibhash_ntxent 1.0      # CIBHash's contrastive on binary hash
#   --lambda_cibhash_kl     0.001    # CIBHash's symmetric Bernoulli KL
#   --cibhash_temperature   0.3      # CIBHash paper default
#
# Per-codebook decomposition: continuous_code [B, 18, 4] is mapped to
# 6-bit-per-codebook probabilities [B, 6, 6] using the DNA encoding
# A=00, C=01, G=10, T=11:
#     bit_0_prob = P(G) + P(T)        bit_1_prob = P(C) + P(T)
# z = sign(prob - 0.5) via STE. NtXent and KL averaged over 6 codebooks.
#
# References:
#   v118a (K=128 mAP champion): mAP 0.7721, DNA 0.207, P@1 0.900,
#                               NMI 0.628, B1 0.107, B2 0.064.
#   external CIBHash-CLIP Flickr25k mAP 0.6844 (DNA-uniq 0.967).
#
# Usage: bash scripts/train_v119a_v118a_cibhash_per_codebook_flickr25k_clip.sh <GPU_ID>
set -eu

GPU="${1:-0}"
CACHE="${CACHE:-./cache/flickr25k_clip_v4plus}"
QWEN="${QWEN:-./cache/flickr25k_qwen_v4.jsonl}"
WHITEN_NPZ="${WHITEN_NPZ:-${CACHE}/text_whiten.npz}"
WHITEN_GAMMA="${WHITEN_GAMMA:-0.25}"
TAG="${TAG:-v119a_v118a_cibhash_perCb_K128_partialWhiten_gamma${WHITEN_GAMMA}}"
LOG="logs/${TAG}.log"
mkdir -p logs

if [ ! -f "$WHITEN_NPZ" ]; then
    echo "[run-v119a] whitening .npz missing — computing it now ..."
    /home/yschoi/.conda/envs/dna_hashing/bin/python \
        scripts/build_text_whiten_matrix.py \
        --cache_dir "$CACHE" \
        --out       "$WHITEN_NPZ"
fi

echo "[run-v119a] GPU=$GPU cache=$CACHE whiten=$WHITEN_NPZ gamma=$WHITEN_GAMMA K=128 cibhash=ON tag=$TAG"

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
    --eval_every 5 \
    --post_eval_compositional \
    --dna_distance_mode base \
    -ev -s 2>&1 | tee "$LOG"
