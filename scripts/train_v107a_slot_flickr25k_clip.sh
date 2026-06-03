#!/usr/bin/env bash
# v107a_slot = v107a recipe with --router_type slot (Slot Attention, Locatello et al. NeurIPS 2020).
#
# User proposal (2026-06-03):
#   The M textual semantic embeddings act as slots. Like slot attention,
#   patches are distributed across slots via COMPETITIVE softmax (softmax
#   over the M slot axis, NOT over the patch axis). Slot updates follow
#   the original paper: GRU(updates, slots_prev) + residual MLP.
#
# Differences vs other routers
# - Sinkhorn:   balanced marginals both ways (sum_n=1/M and sum_m=1/N).
# - Attention:  per-part softmax over patches (sum_n routing[:, m] = 1).
#               No competition between slots; each part picks its own patches.
# - Slot:       per-patch softmax over slots (sum_m routing[n, :] = 1).
#               Slots COMPETE for explaining each patch. Iterative refinement
#               with GRU + MLP (3 iterations by default).
#
# Note: --routing_adaptive_topp is NOT used (slot attention has its own
# iterative competition; the adaptive-topp mask was a Sinkhorn add-on).
#
# Recipe vs v107a:
#   --router_type slot                # was 'sinkhorn'
#   --slot_attention_iters 3          # NEW (paper default)
#   (REMOVED) --routing_adaptive_topp / topp_min / topp_max
#   (REMOVED) --sinkhorn_epsilon_init / epsilon_final
#
# Usage: bash scripts/train_v107a_slot_flickr25k_clip.sh <GPU_ID>
set -eu

GPU="${1:-1}"
CACHE="${CACHE:-./cache/flickr25k_clip_v4plus}"
QWEN="${QWEN:-./cache/flickr25k_qwen_v4.jsonl}"
TAG="${TAG:-v107a_slot_v107a_slotAttentionRouter}"
LOG="logs/${TAG}.log"
mkdir -p logs

echo "[run-v107a-slot] GPU=$GPU cache=$CACHE qwen=$QWEN tag=$TAG"

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
    --router_type slot \
    --slot_attention_iters 3 \
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
    --lambda_vq 0.25 --lambda_quant 0.05 \
    --lambda_anchor 0.05 --lambda_dna 0.05 --lambda_bu 0.02 \
    --eta_base_balance 0.3 \
    --lambda_codeword_codon_sinkhorn 0.0 \
    --lambda_proto_cluster_cos 0.5 \
    --proto_cluster_cos_tau 0.1 \
    --eval_every 5 \
    --post_eval_compositional \
    --dna_distance_mode base \
    -ev -s 2>&1 | tee "$LOG"
