#!/usr/bin/env bash
# v106a = v103a recipe + Sinkhorn-OT codeword<->codon bijection loss.
#
# v103a baseline (Flickr25k-CLIP, 36-bit, genuinely unsupervised):
#   mAP 0.7602, DNA-uniq 0.241, cb-tuple-uniq 0.553, P@1 0.9000
#
# Bijection hypothesis (from the 9-way analysis):
#   The codeword->DNA codon collision (DNA 0.241 << cb-tuple 0.553, 2.3x)
#   originates INSIDE the codon head fc weights, not chunk-partition or
#   residual injection. The Sinkhorn-OT loss directly enforces K=64
#   codewords -> 64 distinct codons via marginal constraints on the
#   transport plan.
#
# Expected: DNA-uniq 0.55 ~ 0.80, mAP within +-0.02 of v103a.
#
# Hyperparams:
#   --lambda_codeword_codon_sinkhorn 0.1   (scale ~ log K=4.16, lambda 0.1 makes term ~0.4 at start)
#   --codeword_codon_sinkhorn_eps 0.1      (sharp enforcement; closer to Hungarian)
#   --codeword_codon_sinkhorn_iters 30     (convergence)
#
# Usage: bash scripts/train_v106a_flickr25k_clip.sh <GPU_ID>
set -eu

GPU="${1:-4}"
CACHE="${CACHE:-./cache/flickr25k_clip_v4plus}"
QWEN="${QWEN:-./cache/flickr25k_qwen_v4.jsonl}"
TAG="${TAG:-v106a_v103a_sinkhornBij}"
LOG="logs/${TAG}.log"
mkdir -p logs

echo "[run-v106a] GPU=$GPU cache=$CACHE qwen=$QWEN tag=$TAG"

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
    --codon_residual_gamma 0.3 \
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
    --lambda_codeword_codon_sinkhorn 0.1 \
    --codeword_codon_sinkhorn_eps 0.1 \
    --codeword_codon_sinkhorn_iters 30 \
    --eval_every 5 \
    --post_eval_compositional \
    --dna_distance_mode base \
    -ev -s 2>&1 | tee "$LOG"
