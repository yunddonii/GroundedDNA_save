#!/usr/bin/env bash
# v112b = v106b recipe + Hierarchical Codon Decomposition loss.
#
# v112a was implemented on v107a (proto-cosine cluster) base. v112b is
# implemented on v106b (Sinkhorn-OT codeword-codon bijection) base.
#
# Why combine v106b's Sinkhorn-OT bij with v112's hierarchical decomp?
# -------------------------------------------------------------------
# v106b's Sinkhorn-OT bijection: forces K=64 codewords to map to K=64
#   DISTINCT codons via marginal constraints on the transport plan.
# v112's hierarchical decomposition: forces first base of each codon to
#   classify the codeword's text-similarity cluster (4 clusters per
#   codebook, mapping to 4 base options A/C/G/T).
# These mechanisms are ORTHOGONAL and can synergize:
#   - 4 clusters x 16 sub-states each = 64 codons (matches K=64)
#   - Sinkhorn ensures all 64 codons used distinctly
#   - Hierarchical ensures the 4 clusters use different first bases
#   - Net: codewords organized as [cluster_bit, intra_bits] AND bijective
#
# Recipe vs v106b:
#   --lambda_hierarchical_cluster_codon 0.3
#   --hierarchical_cluster_n_clusters 4
#   --hierarchical_cluster_refresh_every 5
#   --hierarchical_cluster_warmup_epochs 5
# Everything else from v106b preserved:
#   gamma=0 (path consistency)
#   --lambda_codeword_codon_sinkhorn 0.1 (KEPT: Sinkhorn-OT bij)
#   --codeword_codon_sinkhorn_eps 0.1 --codeword_codon_sinkhorn_iters 30
#   --lambda_proto_cluster_cos 0.0 (v106b does NOT have proto-cluster)
#   All VQ regs at v103a values (vq=0.25, quant=0.05, anchor=0.05,
#                                dna=0.05, bu=0.02)
#   paired-aug NtXent + per_codebook + dynamic text_cos tau
#   text_hash MSE 0.05 + text_hash_ntxent 0.05
#   sinkhorn router, hash_target_mode siglip_cos
#
# Reference (v106b baseline, Flickr25k-CLIP):
#   mAP 0.7407, DNA-uniq 0.347, cb-tuple 0.477, P@1 0.917, NMI 0.604,
#   B1 0.115, B2 0.072, collision ratio 1.37x (best in v9x family).
#
# Expected outcomes
# -----------------
# (A) Synergy: hierarchical adds interpretable structure on top of v106b's
#     DNA-unique champion. mAP ~ 0.74, DNA ~ 0.35+, NMI 0.55-0.58, paper-
#     final candidate for combined Contribution 1 + 2.
# (B) Conflict: hierarchical (4-way first-base) clashes with bijection
#     (each codeword distinct codon) -> only 4 codons used at first
#     position -> DNA unique drops.
# (C) Hierarchical dominates first base; Sinkhorn still works on bases
#     2+3 -> DNA unique maintained, first base interpretable.
#
# Usage: bash scripts/train_v112b_hierClusterCodon_onV106b_flickr25k_clip.sh <GPU_ID>
set -eu

GPU="${1:-4}"
CACHE="${CACHE:-./cache/flickr25k_clip_v4plus}"
QWEN="${QWEN:-./cache/flickr25k_qwen_v4.jsonl}"
TAG="${TAG:-v112b_v106b_hierClusterCodon}"
LOG="logs/${TAG}.log"
mkdir -p logs

echo "[run-v112b] GPU=$GPU cache=$CACHE qwen=$QWEN tag=$TAG"

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
    --lambda_proto_cluster_cos 0.0 \
    --lambda_hierarchical_cluster_codon 0.3 \
    --hierarchical_cluster_n_clusters 4 \
    --hierarchical_cluster_refresh_every 5 \
    --hierarchical_cluster_warmup_epochs 5 \
    --eval_every 5 \
    --post_eval_compositional \
    --dna_distance_mode base \
    -ev -s 2>&1 | tee "$LOG"
