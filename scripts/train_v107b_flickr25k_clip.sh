#!/usr/bin/env bash
# v107b = v106b + v107a additive combination.
#
# Hypothesis: the two best-performing mechanisms operate at ORTHOGONAL
# levels of the model:
#   - v106b Sinkhorn-OT bijection loss: DECODER side
#       enforces K=64 codewords -> K distinct codons mapping at the
#       codon-head fc weights.
#   - v107a proto-cosine clustering   : ENCODER side
#       pulls semantic_visual_tokens z[b, m] toward their assigned
#       codeword in cosine InfoNCE space.
# Combining them should keep:
#   - v106b's DNA unique gain (0.241 -> 0.347) from codeword->codon bijection
#   - v107a's NMI gain (0.571 -> 0.623) from prototype clustering tightening
#   - v107a's mAP recovery (0.7407 -> 0.7624) from the encoder-side signal
#
# Recipe vs v106b:
#   --lambda_proto_cluster_cos 0.5         # NEW (from v107a)
#   --proto_cluster_cos_tau    0.1         # NEW (from v107a)
# All v106b flags preserved:
#   --codon_residual_gamma 0.0
#   --lambda_codeword_codon_sinkhorn 0.1
#   --codeword_codon_sinkhorn_eps 0.1
#   --codeword_codon_sinkhorn_iters 30
#   --hash_target_mode siglip_cos
#   --lambda_hash 0.0, lambda_hash_hard 0.0
#   --lambda_text_hash 0.05, lambda_text_hash_ntxent 0.05
#   --use_paired_aug_ntxent + per_codebook + dynamic_tau
#   --eta_base_balance 0.3 (KL form)
#   --router_type sinkhorn (default)
#
# Reference numbers (all Flickr25k-CLIP, DNA-base unique on DB):
#                                       mAP    DNA-uniq  P@1    NMI
#   v103a (no Sinkhorn, no proto)    0.7602    0.241   0.900  0.571
#   v106b (Sinkhorn only)            0.7407    0.347   0.917  0.604
#   v107a (proto only)               0.7624    0.213   0.880  0.623
#   v107b (BOTH; this run, TBD)       ???       ???    ???    ???
#
# Predicted (if mechanisms are truly orthogonal):
#   mAP 0.74-0.76, DNA-uniq 0.30-0.40+, P@1 0.90-0.92, NMI 0.62-0.65.
#
# Usage: bash scripts/train_v107b_flickr25k_clip.sh <GPU_ID>
set -eu

GPU="${1:-0}"
CACHE="${CACHE:-./cache/flickr25k_clip_v4plus}"
QWEN="${QWEN:-./cache/flickr25k_qwen_v4.jsonl}"
TAG="${TAG:-v107b_v106b_plus_v107a_sinkhornBij_protoCluster}"
LOG="logs/${TAG}.log"
mkdir -p logs

echo "[run-v107b] GPU=$GPU cache=$CACHE qwen=$QWEN tag=$TAG"

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
    --lambda_vq 0.25 --lambda_quant 0.05 \
    --lambda_anchor 0.05 --lambda_dna 0.05 --lambda_bu 0.02 \
    --eta_base_balance 0.3 \
    --lambda_codeword_codon_sinkhorn 0.1 \
    --codeword_codon_sinkhorn_eps 0.1 \
    --codeword_codon_sinkhorn_iters 30 \
    --lambda_proto_cluster_cos 0.5 \
    --proto_cluster_cos_tau 0.1 \
    --eval_every 5 \
    --post_eval_compositional \
    --dna_distance_mode base \
    -ev -s 2>&1 | tee "$LOG"
