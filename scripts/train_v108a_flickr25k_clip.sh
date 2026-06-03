#!/usr/bin/env bash
# v108a = v107a recipe + ALL VQ/codebook-style losses turned OFF.
#
# Motivation: the v107a proto-cosine cluster loss (L_proto_cluster_cos)
# already pulls z toward its assigned codeword in cosine InfoNCE space.
# Test whether this single loss can REPLACE the traditional VQ-VAE
# regularization stack (commitment + entropy + base-balance + balance-
# usage + anchor), or whether those losses contribute orthogonal signal.
#
# v107a active VQ/codebook losses (turned OFF in v108a):
#   --lambda_vq         0.25 -> 0.0    (z -> q commitment MSE)
#   --lambda_quant      0.05 -> 0.0    (codon STE commitment)
#   --lambda_dna        0.05 -> 0.0    (per-position entropy + KL base-balance)
#   --lambda_bu         0.02 -> 0.0    (codebook balance + assignment uncorrelated)
#   --lambda_anchor     0.05 -> 0.0    (ema text anchor; already inert in EMA mode)
#
# Kept ON (these are NOT codebook-internal):
#   --lambda_wasserstein 0.05          (router OT cost; routing-level)
#   --lambda_proto_cluster_cos 0.5     (the cosine InfoNCE clustering)
#   --use_paired_aug_ntxent + dynamic_tau  (visual-visual contrastive)
#   --lambda_text_hash 0.05            (visual-textual MSE)
#   --lambda_text_hash_ntxent 0.05     (visual-textual InfoNCE)
#
# Expected outcomes:
#   (A) proto-cosine alone suffices -> mAP and DNA unique similar to v107a.
#       Conclusion: VQ commitment + entropy + balance are redundant when
#       proto-cosine clustering is in place.
#   (B) VQ losses were doing real work -> mAP drop or codebook collapse.
#       Conclusion: proto-cosine is complementary, not replacement.
#   (C) Removing regularizers frees the codebook to specialize ->
#       DNA unique may INCREASE (less averaging pressure).
#
# v107a baseline (sinkhorn router + proto-cosine, no Sinkhorn bij):
#   mAP 0.7624, DNA-uniq 0.213, cb-tuple 0.442, P@1 0.880, NMI 0.623.
#
# Usage: bash scripts/train_v108a_flickr25k_clip.sh <GPU_ID>
set -eu

GPU="${1:-1}"
CACHE="${CACHE:-./cache/flickr25k_clip_v4plus}"
QWEN="${QWEN:-./cache/flickr25k_qwen_v4.jsonl}"
TAG="${TAG:-v108a_v107a_noVQcodebook}"
LOG="logs/${TAG}.log"
mkdir -p logs

echo "[run-v108a] GPU=$GPU cache=$CACHE qwen=$QWEN tag=$TAG"

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
    --eval_every 5 \
    --post_eval_compositional \
    --dna_distance_mode base \
    -ev -s 2>&1 | tee "$LOG"
