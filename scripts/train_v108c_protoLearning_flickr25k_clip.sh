#!/usr/bin/env bash
# v108c = v108a recipe + --codebook_update gradient
#         (EMA codebook buffer -> learnable nn.Parameter, i.e. full
#         prototype learning instead of VQ-VAE-style EMA quantization).
#
# Architectural difference vs v108a
# ---------------------------------
# v108a (EMA, current path):
#   self.codebooks = register_buffer(...)        # no gradient
#   updates via EMA from per-codebook sample assignments
#
# v108c (gradient, prototype learning):
#   self.codebooks = nn.Parameter(...)           # has gradient
#   updates via gradient backprop through:
#     - STE-quantized_tokens used downstream
#     - loss_proto_cluster_cos (cosine InfoNCE between z and codebook)
#
# Why this is "prototype learning":
#   - The codebook becomes a set of learnable parameters that act as
#     prototypes for each codebook-attribute.
#   - loss_proto_cluster_cos (lambda 0.5) creates a MUTUAL contrastive
#     pull: z toward assigned codeword AND codeword toward assigned z's.
#   - This is closer to ProtoCL / SwAV-style prototype clustering than
#     to vanilla VQ-VAE (which would use commitment-MSE + EMA).
#   - All VQ-VAE auxiliary regularizers (vq, quant, dna, bu, anchor)
#     are OFF -- the only learning signals on the codebook are
#     proto_cluster_cos + STE backprop.
#
# Hypothesis directions (vs v108a baseline)
# -----------------------------------------
# (A) Gradient + proto_cluster_cos -> codebook converges DIRECTLY to
#     prototype semantics -> Contribution 2 (text-grounded) is
#     architecturally more explicit. Performance maintained.
# (B) Without EMA's stabilizing effect, prototypes drift -> codebook
#     collapse or assignment instability. Lower mAP / DNA.
# (C) Codebooks become more text-coherent (B1 ↑) but possibly more
#     correlated (NMI ↑) -- ambiguous compositional effect.
#
# Recipe vs v108a:
#   --codebook_update gradient  (was 'ema')
# Everything else from v108a preserved (gamma=0, lambda_proto_cluster_cos
# 0.5, VQ regs all 0, paired-aug NtXent, text_hash + text_hash_ntxent,
# router=sinkhorn, hash_target_mode=siglip_cos, no text_init_codebook).
#
# Reference (v108a baseline, EMA): mAP 0.7382, DNA 0.219, cb-tuple 0.547,
#                                   P@1 0.901, NMI 0.575, B1 0.093, B2 0.055.
#
# Usage: bash scripts/train_v108c_protoLearning_flickr25k_clip.sh <GPU_ID>
set -eu

GPU="${1:-1}"
CACHE="${CACHE:-./cache/flickr25k_clip_v4plus}"
QWEN="${QWEN:-./cache/flickr25k_qwen_v4.jsonl}"
TAG="${TAG:-v108c_v108a_protoLearningGradient}"
LOG="logs/${TAG}.log"
mkdir -p logs

echo "[run-v108c] GPU=$GPU cache=$CACHE qwen=$QWEN tag=$TAG"

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
    --codebook_update gradient \
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
    --eval_every 5 \
    --post_eval_compositional \
    --dna_distance_mode base \
    -ev -s 2>&1 | tee "$LOG"
