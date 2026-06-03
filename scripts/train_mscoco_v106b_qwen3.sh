#!/usr/bin/env bash
# mscoco_v106b = v106b recipe ported to MSCOCO (K=128 + Qwen3 captions).
#
# Flickr v106b result (paper-final candidate):
#   mAP 0.7407, DNA unique (DB) 0.347 (+44 % vs v103a), P@1 0.917,
#   B1/B2/NMI all v9x maxima, collision ratio 2.29x -> 1.37x.
#
# MSCOCO context (DB=107K, K=128 > 4^3=64 codons forces pigeonhole
# collisions, so Sinkhorn marginals (1/128, 1/64) spread 2 codewords
# per codon uniformly -- best possible under pigeonhole constraint):
#   - mscoco_v102a (Sinkhorn router only, no bij loss): mAP 0.5440,
#     DNA uniq 0.116 (= 12,481/107,218), cb-tuple 0.462, ratio 3.98x
#     (worst collision ratio in family).
#   - CIBHash-CLIP MSCOCO baseline: mAP 0.5842 (we lose by -0.040).
#
# Expected:
#   - DNA unique improvement comparable to Flickr (DNA 0.12 -> 0.20+).
#   - mAP impact unclear -- could improve (more meaningful DNA codes)
#     or regress (loss of mAP from removing residual).
#   - If mAP regresses heavily, may need stronger backbone signal.
#
# Recipe vs v106b (Flickr): only changes are dataset, codebook_size, qwen cache.
#
# Usage: bash scripts/train_mscoco_v106b_qwen3.sh <GPU_ID>
set -eu

GPU="${1:-4}"
CACHE="${CACHE:-./cache/mscoco_clip_v4plus}"
QWEN="${QWEN:-./cache/mscoco_qwen3_v4_trainset.jsonl}"
TAG="${TAG:-mscoco_v106b_qwen3_sinkhornBij_noResid}"
LOG="logs/${TAG}.log"
mkdir -p logs

echo "[run-mscoco-v106b] GPU=$GPU cache=$CACHE qwen=$QWEN tag=$TAG"

CUDA_VISIBLE_DEVICES="$GPU" \
/home/yschoi/.conda/envs/dna_hashing/bin/python train_siglip2.py \
    --tag "$TAG" \
    --dataset MSCOCO --setting 1 \
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
