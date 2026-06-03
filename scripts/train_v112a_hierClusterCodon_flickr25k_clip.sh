#!/usr/bin/env bash
# v112a = v107a recipe + Hierarchical Codon Decomposition loss.
#
# User insight 2026-06-04: within each codebook, codewords have latent
# text-semantic clusters; each cluster should map to a distinct DNA codon.
# Proposal 2 from the design brief: HIERARCHICALLY DECOMPOSE the 3-base
# codon into [cluster_bit, intra_bit_1, intra_bit_2]:
#
#   3-base codon = [first base, second base, third base]
#                  [4 choices,  4 choices,   4 choices]   = 64 codons
#
#   First base   = CLUSTER IDENTIFIER (4 clusters per codebook, matching
#                  the 4 base options A/C/G/T).
#   Bases 2 + 3  = within-cluster variation (16 sub-states each = 64
#                  codons subdivided into 4 cluster x 16 sub-states).
#
# Mechanism
# ---------
# 1) After warmup (default 5 epochs of EMA codebook stabilization), run
#    k-means on each codebook's K=64 codewords (cosine-normalized) with
#    C=4 cluster centers. Cluster labels stored as buffer
#    self.hierarchical_cluster_labels [M=6, K=64] long.
# 2) Forward computes codeword_codon_logits [M, K, 3, 4] via the model's
#    decode_codeword path.
# 3) Loss: CE between the FIRST BASE logits of each codeword and its
#    assigned cluster label. The remaining 2 bases are unsupervised
#    (handled by other losses).
# 4) Re-cluster every 5 epochs (default) as the codebook evolves.
#
# How this addresses user's contributions
# ---------------------------------------
# Contribution 1 (compositional / attribute-aware code):
#   The first codon position now has an EXPLICIT, INTERPRETABLE role --
#   it identifies which text-semantic cluster the codeword belongs to.
# Contribution 2 (text-grounded codon semantics):
#   K-means clusters codewords by their (text-derived) embedding
#   similarity, so the first base's labels carry text-grounded meaning.
#
# Recipe vs v107a:
#   --lambda_hierarchical_cluster_codon 0.3
#   --hierarchical_cluster_n_clusters 4
#   --hierarchical_cluster_refresh_every 5
#   --hierarchical_cluster_warmup_epochs 5
# Everything else from v107a preserved (gamma=0, proto_cluster_cos 0.5,
# paired-aug NtXent, text_hash MSE+InfoNCE 0.05/0.05, sinkhorn router,
# all v107a VQ regs at their original values).
#
# Reference (v107a baseline): mAP 0.7624, DNA-uniq 0.213, cb-tuple 0.442,
#                              P@1 0.880, NMI 0.623, B1 0.107, B2 0.066.
#
# Expected outcomes
# -----------------
# (A) First base correctly classifies cluster after training -> codewords
#     in same cluster share first base. Different clusters get different
#     first bases. DNA codes become hierarchically interpretable.
# (B) cb-tuple unique should remain or improve (clustering doesn't reduce
#     codeword diversity, only adds structure).
# (C) DNA unique may DROP at first base (since all codewords in cluster c
#     output base c) but the (base2, base3) bits can still distinguish
#     16 sub-states. Net effect on DNA unique unclear -- depends on how
#     well bases 2+3 differentiate within-cluster codewords.
# (D) NMI should DROP (cluster supervision tightens compositional axis).
#
# Usage: bash scripts/train_v112a_hierClusterCodon_flickr25k_clip.sh <GPU_ID>
set -eu

GPU="${1:-0}"
CACHE="${CACHE:-./cache/flickr25k_clip_v4plus}"
QWEN="${QWEN:-./cache/flickr25k_qwen_v4.jsonl}"
TAG="${TAG:-v112a_v107a_hierClusterCodon}"
LOG="logs/${TAG}.log"
mkdir -p logs

echo "[run-v112a] GPU=$GPU cache=$CACHE qwen=$QWEN tag=$TAG"

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
    --lambda_codeword_codon_sinkhorn 0.0 \
    --lambda_proto_cluster_cos 0.5 \
    --proto_cluster_cos_tau 0.1 \
    --lambda_hierarchical_cluster_codon 0.3 \
    --hierarchical_cluster_n_clusters 4 \
    --hierarchical_cluster_refresh_every 5 \
    --hierarchical_cluster_warmup_epochs 5 \
    --eval_every 5 \
    --post_eval_compositional \
    --dna_distance_mode base \
    -ev -s 2>&1 | tee "$LOG"
