#!/usr/bin/env bash
# v141a = v133a with TWO changes:
#   (1) --router_type sinkhorn -> cluster_attn
#       DiVT-inspired soft Sinkhorn cluster + masked cross-attention,
#       operating on RAW visual patches (before visual_adapter).
#   (2) --visual_adapter_after_router (new flag, default off)
#       Relocates visual_adapter from per-patch to AFTER the M=6 router
#       output, keeping the gradient path connected through the
#       cross-attention back to the encoder (no hard-clustering cut-off).
#
# v133a recipe (Flickr25k-CLIP, commit 1eecc32 + 1eecc32):
#   K=128, L=3, bij OFF, NO local-residual, per_codebook text NtXent
#   (M=6 InfoNCEs on [B, 12]), cibhash per_codebook + dynamic tau,
#   partial whitening gamma=0.25, paired-aug DNA NtXent OFF.
#   Result Flickr (DB=23K): mAP 0.7541, P@1 0.9150, NMI 0.626,
#   NMI local↔local 0.702, B1 0.128, B2 0.079, dead 0.040.
#
# v141 motivation (from DiVT, CVPR 2026; arXiv 2503.16876).
# ViT patches become highly entangled in late layers (cosine sim ~0.38)
# because self-attention homogenizes the embedding. v133a's Sinkhorn
# router routes patches AFTER the visual_adapter, so the codebook input
# is already a tangled weighted-sum of entangled patches. v141 inserts a
# DiVT-style "cluster + cluster-restricted cross-attention" step BEFORE
# the codebook, producing M=6 disentangled visual tokens whose attention
# is restricted to within-cluster patches via a soft log-P mask.
#
# Design summary (see models/cluster_attention_router.py for math).
#   Raw patches [B, N, D]
#       |
#       |  Sinkhorn-balanced soft cluster (M=6 learnable prototypes)
#       v
#   P_cluster [B, N, 6]   (rows sum to 1, cols balanced ~N/6)
#       |
#       |  Attention-pooled centroid (differentiable)
#       v
#   centroids [B, 6, D]
#       |
#       |  Soft-masked cross-attention (DiVT Eq. 2-4 with soft log-P mask)
#       |    Q from centroid, K from patches, V from patches+pos_emb
#       v
#   M tokens [B, 6, D]
#       |
#       |  visual_adapter (NEW position: after router output)
#       v
#   semantic_visual_tokens [B, 6, D_model]
#       |
#       |  VQ codebook + CodonHead (unchanged from v133a)
#       v
#   DNA [B, 6, 3, 4]
#
# Inference path. The router skips the cross-attention branch
# (training=False) and falls back to a cheap cluster-weighted sum,
# replacing v133a's anchor-based Sinkhorn OT routing. Cluster
# prototypes serve as the routing anchors in both train and inference,
# avoiding a train/test anchor mismatch.
#
# Hypothesis (from v139a atlas analysis: NMI local↔local 0.702 means
# the 5 local codebooks heavily redundantly encode similar concepts).
# v141 explicitly disentangles patches BEFORE VQ -> compositional
# axis predicted to improve dramatically:
#   NMI mean        : 0.626 -> 0.50-0.58 (compositional ↑)
#   NMI local↔local : 0.702 -> 0.45-0.55 (cluster purity)
#   B1 lift          : 0.128 -> 0.13-0.16
#   mAP              : 0.7541 ± 0.02 (cross-attention info preserved)
#   dead avg         : 0.040 -> 0.03-0.07 (Sinkhorn cluster balance)
#
# Usage: bash scripts/train_v141a_v133a_clusterAttn_K128_flickr25k_clip.sh <GPU_ID>
set -eu

GPU="${1:-0}"
CACHE="${CACHE:-./cache/flickr25k_clip_v4plus}"
QWEN="${QWEN:-./cache/flickr25k_qwen_v4.jsonl}"
WHITEN_NPZ="${WHITEN_NPZ:-${CACHE}/text_whiten.npz}"
WHITEN_GAMMA="${WHITEN_GAMMA:-0.25}"
TAG="${TAG:-v141a_v133a_clusterAttn_K128_partialWhiten_gamma${WHITEN_GAMMA}}"
LOG="logs/${TAG}.log"
mkdir -p logs

if [ ! -f "$WHITEN_NPZ" ]; then
    echo "[run-v141a] whitening .npz missing — computing it now ..."
    /home/yschoi/.conda/envs/dna_hashing/bin/python \
        scripts/build_text_whiten_matrix.py \
        --cache_dir "$CACHE" \
        --out       "$WHITEN_NPZ"
fi

echo "[run-v141a] GPU=$GPU cache=$CACHE whiten=$WHITEN_NPZ gamma=$WHITEN_GAMMA K=128 (cluster_attn + adapter-after) tag=$TAG"

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
    --codebook_size 128 \
    --c_global_source siglip2_global \
    --per_slot_text_adapter \
    --global_gate_init_logit 4.595 \
    --router_type cluster_attn \
    --cluster_attn_heads 4 \
    --cluster_attn_mlp_ratio 4.0 \
    --cluster_attn_sinkhorn_eps 0.1 \
    --cluster_attn_sinkhorn_iters 3 \
    --cluster_attn_pool_temperature 0.3 \
    --visual_adapter_after_router \
    --codon_residual_gamma 0.0 \
    --text_embed_transform partial_whiten \
    --text_whiten_npz "$WHITEN_NPZ" \
    --text_whiten_gamma "$WHITEN_GAMMA" \
    --use_paired_aug_ntxent \
    --lambda_ntxent 0.0 \
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
    --text_hash_ntxent_mode per_codebook \
    --lambda_wasserstein 0.05 \
    --lambda_vq 0.25 --lambda_quant 0.05 \
    --lambda_anchor 0.05 --lambda_dna 0.05 --lambda_bu 0.02 \
    --eta_base_balance 0.3 \
    --lambda_codeword_codon_sinkhorn 0.0 \
    --lambda_cibhash_ntxent 1.0 \
    --lambda_cibhash_kl 0.001 \
    --cibhash_mode per_codebook \
    --cibhash_temperature 0.3 \
    --cibhash_dynamic_tau \
    --cibhash_dynamic_tau_alpha 0.3 \
    --eval_every 5 \
    --post_eval_compositional \
    --dna_distance_mode base \
    -ev -s 2>&1 | tee "$LOG"
