#!/usr/bin/env bash
# mscoco_v106b_K256 = mscoco_v106b recipe (L=3, bij ON) + codebook_size 128 -> 256.
#
# v106b reference (K=128 L=3, bij Sinkhorn 0.1): mAP 0.7407, DNA 0.347 (Flickr).
# Flickr v106b -> v122a (K=256 L=4): DNA 0.347 -> 0.551 (+0.204).
#
# This MSCOCO port pushes K to 256 *without* changing L (stays L=3). That puts
# the codebook into 4x pigeonhole regime (256 codewords vs 4^3=64 codons),
# so the Sinkhorn bij loss cannot achieve true permutation; instead it picks a
# 4-to-1 surjection. Compared to mscoco_v148b (K=256 UOT+aggressive topp + no
# bij), this run keeps L=3 + bij ON + standard topp 0.5/0.9 - tests whether
# the K-expansion alone (without UOT/sharper topp) brings the DNA breakthrough
# seen on Flickr v122a even under pigeonhole pressure.
#
# Predicted: codeword diversity (cb_tuple) up significantly, DNA-uniq up
# modestly (still pigeonholed); mAP risk small.
#
# Usage: bash scripts/train_mscoco_v106b_K256_qwen3.sh <GPU_ID>
set -eu

GPU="${1:-4}"
CACHE="${CACHE:-./cache/mscoco_clip_v4plus}"
QWEN="${QWEN:-./cache/mscoco_qwen3_v4_trainset.jsonl}"
TAG="${TAG:-mscoco_v106b_K256_qwen3_sinkhornBij_noResid}"
LOG="logs/${TAG}.log"
mkdir -p logs

echo "[run-mscoco-v106b-K256] GPU=$GPU cache=$CACHE qwen=$QWEN K=256 (vs v106b K=128) tag=$TAG"

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
    --codebook_size 256 \
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
