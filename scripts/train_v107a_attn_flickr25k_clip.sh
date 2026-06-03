#!/usr/bin/env bash
# v107a_attn = v107a recipe with --router_type attention (instead of sinkhorn).
#
# Motivation (user proposal 2026-06-03):
#   Replace Sinkhorn OT router with attention-based router (cross-attention,
#   text part = query, visual = key/value). Keep confidence-adaptive top-p
#   routing -- now supported by AttentionRouter (new in this commit).
#
# v107a baseline (sinkhorn router):
#   mAP 0.7624 (best so far at gamma=0), DNA-uniq 0.213, NMI 0.6229 (v9x max),
#   collision ratio 2.08x.
#
# Difference from Sinkhorn -> Attention:
#   - Sinkhorn: balanced marginals both ways (sum_n=1/M, sum_m=1/N).
#               Forces every part to receive ~equal patch mass even if image
#               has no content for that part.
#   - Attention: only sum_n = 1 (per-part softmax over patches). Parts pull
#                whichever patches they like, no marginal-balance penalty.
#                Per-part heatmap focuses on semantically relevant regions.
#
# Expected: attention router may better localize semantic parts -> stronger
# clustering signal in proto-cosine loss -> higher NMI, possibly higher mAP.
# Risk: without marginal balance on parts, some parts may get no patches
# (codebook collapse / dead parts).
#
# Usage: bash scripts/train_v107a_attn_flickr25k_clip.sh <GPU_ID>
set -eu

GPU="${1:-1}"
CACHE="${CACHE:-./cache/flickr25k_clip_v4plus}"
QWEN="${QWEN:-./cache/flickr25k_qwen_v4.jsonl}"
TAG="${TAG:-v107a_attn_v107a_attentionRouter}"
LOG="logs/${TAG}.log"
mkdir -p logs

echo "[run-v107a-attn] GPU=$GPU cache=$CACHE qwen=$QWEN tag=$TAG"

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
    --router_type attention \
    --attention_router_temperature 0.1 \
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
