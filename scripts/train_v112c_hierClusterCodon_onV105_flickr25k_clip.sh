#!/usr/bin/env bash
# v112c = v105 head structure (codon_full_linear) + Hierarchical Codon
#         Decomposition + gamma=0 (path consistency).
#
# Three v112 variants now in parallel
# -----------------------------------
#   v112a = v107a base + hier (encoder proto-cluster + hierarchical)
#   v112b = v106b base + hier (decoder Sinkhorn-OT bij + hierarchical)
#   v112c = v105 head + hier  (full-Linear codon decoder + hierarchical)
# All use gamma=0 for path consistency (the codon supervision at the
# codeword level only manifests at sample-level DNA when the inference
# path matches; this was v106b's key finding).
#
# Why v105 + hier might synergize
# -------------------------------
# v105's codon head: Linear(d_model=768, 12).view(B, 3, 4)
#   - Each (codon position, base) output uses ALL 768 dims of the codeword.
#   - 9x more parameters than v103a's Linear(chunk=256, 4) shared head.
#   - Previously: cb-tuple unique +0.10 vs v103a, but DNA unique unchanged
#     (the chunk-partition was NOT the bottleneck; codon collision is in
#     the fc weights themselves).
# v112 hierarchical: first base CE supervises cluster ID.
# Together:
#   - v105's expressive decoder has more freedom to separate codewords
#     into cluster-conditioned codon distributions
#   - Hierarchical loss directs that freedom toward the cluster axis
#     (first base = cluster ID, bases 2+3 = within-cluster diversity)
#
# Recipe vs v105_head_only (the original v105 launch)
# ---------------------------------------------------
# Original v105_head_only:
#   --codon_residual_gamma 0.3  (v103a default)
#   --codon_full_linear
#   (no hierarchical)
#
# v112c (this run):
#   --codon_residual_gamma 0.0   (CHANGE: path consistency)
#   --codon_full_linear          (KEEP: v105 head)
#   --lambda_hierarchical_cluster_codon 0.3   (NEW: v112)
#   --hierarchical_cluster_n_clusters 4
#   --hierarchical_cluster_refresh_every 5
#   --hierarchical_cluster_warmup_epochs 5
# All v103a flags preserved (vq=0.25, quant=0.05, anchor=0.05,
# dna=0.05, bu=0.02, paired-aug NtXent per_codebook + dynamic text_cos,
# text_hash MSE+InfoNCE 0.05/0.05, hash_target_mode siglip_cos).
#
# Reference (v105_head_only baseline, gamma=0.3):
#   mAP 0.7353, DNA-uniq 0.244, cb-tuple 0.655, P@1 0.878, NMI 0.536,
#   B1 0.095, B2 0.058, dead 0.
#
# Hypotheses
# ----------
# (A) Synergy: expressive full-Linear decoder + hierarchical supervision
#     gives best of both - high cb-tuple diversity (v105 strength) AND
#     interpretable first-base cluster structure (v112 contribution).
# (B) Path consistency (gamma 0.3 -> 0) recovers some mAP loss from
#     v105_head_only (mAP -0.025 vs v103a), AND hierarchical brings
#     additional NMI improvement (sweet spot ~0.5).
# (C) v105's higher cb-tuple unique (0.655) translates to better
#     within-cluster diversity (16 sub-states fully used) when first
#     base is constrained to cluster ID.
#
# Usage: bash scripts/train_v112c_hierClusterCodon_onV105_flickr25k_clip.sh <GPU_ID>
set -eu

GPU="${1:-2}"
CACHE="${CACHE:-./cache/flickr25k_clip_v4plus}"
QWEN="${QWEN:-./cache/flickr25k_qwen_v4.jsonl}"
TAG="${TAG:-v112c_v105_hierClusterCodon_gamma0}"
LOG="logs/${TAG}.log"
mkdir -p logs

echo "[run-v112c] GPU=$GPU cache=$CACHE qwen=$QWEN tag=$TAG"

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
    --codon_full_linear \
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
    --lambda_proto_cluster_cos 0.0 \
    --lambda_hierarchical_cluster_codon 0.3 \
    --hierarchical_cluster_n_clusters 4 \
    --hierarchical_cluster_refresh_every 5 \
    --hierarchical_cluster_warmup_epochs 5 \
    --eval_every 5 \
    --post_eval_compositional \
    --dna_distance_mode base \
    -ev -s 2>&1 | tee "$LOG"
