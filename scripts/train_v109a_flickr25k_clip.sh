#!/usr/bin/env bash
# v109a = v107a recipe with the MSE-form visual-textual loss REMOVED.
#
# User insight (2026-06-03): in v107a (proto-cosine cluster on), the two
# Visual-Textual losses are unnecessary because L_proto_cluster_cos
# already grounds z in textual semantics. Test by:
# - REMOVE: loss_text_hash (MSE between text_cc and image continuous_code)
# - KEEP:   loss_text_hash_ntxent (InfoNCE between text DNA and image DNA)
#           -- this is the "visual DNA <-> textual DNA contrastive" the
#           user described as loss 2.
# - KEEP:   paired-aug NtXent per_codebook with text_cos dynamic tau
#           -- this is the "per-codon paired-aug + textual dynamic tau"
#           the user described as loss 1.
#
# Recipe vs v107a (Flickr25k-CLIP):
#   --lambda_text_hash 0.05 -> 0.0    # REMOVE MSE-form visual-textual
#
# Required code fix (v109 commit): model_siglip2.py _text_path_active
# gating now activates the text codon path when lambda_text_hash_ntxent
# > 0 (previously required lambda_text_hash > 0). Without this fix, text
# continuous_code would be None and the InfoNCE loss would be 0.
#
# v107a baseline (sinkhorn router + proto-cosine, no Sinkhorn bij):
#   mAP 0.7624, DNA-uniq 0.213, cb-tuple 0.442, P@1 0.880, NMI 0.623.
#
# Expected (3 scenarios):
#   (A) MSE form was redundant -> mAP & DNA similar to v107a.
#       Conclusion: drop MSE form permanently for v107-line recipes.
#   (B) MSE form provided real per-sample distribution alignment ->
#       mAP regression, DNA unique drop.
#       Conclusion: keep both forms; the user's hypothesis is wrong.
#   (C) Removing MSE frees the codon distribution to differentiate
#       -> DNA unique INCREASES (less averaging).
#
# Usage: bash scripts/train_v109a_flickr25k_clip.sh <GPU_ID>
set -eu

GPU="${1:-4}"
CACHE="${CACHE:-./cache/flickr25k_clip_v4plus}"
QWEN="${QWEN:-./cache/flickr25k_qwen_v4.jsonl}"
TAG="${TAG:-v109a_v107a_noMSEvisualTextual}"
LOG="logs/${TAG}.log"
mkdir -p logs

echo "[run-v109a] GPU=$GPU cache=$CACHE qwen=$QWEN tag=$TAG"

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
    --lambda_text_hash 0.0 \
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
