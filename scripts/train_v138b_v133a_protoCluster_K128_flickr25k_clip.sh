#!/usr/bin/env bash
# v138a (K=128) = v133a with FIVE changes:
#   (1) --codon_input_source quantized -> routed
#       (codon_head sees RAW router-weighted-sum vectors, not VQ codewords)
#   (2) --lambda_vq 0.25 -> 0.0       (VQ commitment OFF)
#   (3) --lambda_quant 0.05 -> 0.0    (codon-onehot commitment OFF)
#   (4) --lambda_anchor 0.05 -> 0.0   (codebook anchor OFF -- already grad=0)
#   (5) --lambda_bu 0.02 -> 0.0       (codebook balance OFF -- no codebook
#                                       in the loss-path sense)
#   PLUS NEW:
#   (6) --lambda_proto_cluster 0.1    (paired-view prototype InfoNCE on
#                                       codebook_distances softmax)
#   (7) --proto_cluster_temperature 0.3
#   (8) --codebook_size 128 -> 64     (this script: K=128)
#
# Recipe vs v133a (Flickr25k-CLIP commit 1eecc32, mAP 0.7541):
#   K=128, L=3, bij OFF, NO local-residual, per_codebook text NtXent
#   (M=6 InfoNCEs on [B, 12]), cibhash per_codebook + dynamic tau,
#   partial whitening gamma=0.25, paired-aug DNA NtXent OFF.
#
# v138a HYPOTHESIS. The user-proposed architecture:
#   "Replace VQ bottleneck with prototype-supervised clustering. Codon
#    head receives raw routed vectors (no quantization). Prototype
#    learning is a separate paired-view InfoNCE that aligns the
#    distance-based assignment of two augmented views, but it does NOT
#    bottleneck codon_head input."
#
# Mechanism. The Quantizer still runs (so codebook_distances [B, M, K]
# is exposed), but its quantized_tokens output is NOT used by
# codon_head. Instead, codon_head sees `quant_input` (= the post-router
# weighted-sum semantic_visual_tokens). The prototypes (= the
# quantizer's codebooks) learn via EMA from assignments AND via
# gradient through the new paired-view InfoNCE loss
# (encoder-side gradient only, since codebook is a buffer in EMA mode).
#
# What this tests. (a) Does the discrete VQ bottleneck actually help
# retrieval/compositional? (b) Does prototype clustering replace it
# adequately? (c) Does removing it free DNA-uniq toward v122a's 0.551
# range without losing NMI?
#
# Reference (Flickr25k-CLIP K=128, partial whitening gamma=0.25):
#   v107a (proto cosine cluster, sinkhorn router): mAP 0.7624, DNA 0.213, NMI 0.623
#     (closest precedent; v107a also passed raw tokens to codon_head)
#   v133a (legacy VQ, perCb cibhash):              mAP 0.7541, DNA 0.340, NMI 0.626
#   v138b-K128  (this run, K=128,  proto cluster lam=0.1, raw codon input): TBD
#   v138a-K128 (sibling run, K=128 on GPU other):  TBD
#
# Usage: bash scripts/train_v138b_v133a_protoCluster_K128_flickr25k_clip.sh <GPU_ID>
set -eu

GPU="${1:-0}"
CACHE="${CACHE:-./cache/flickr25k_clip_v4plus}"
QWEN="${QWEN:-./cache/flickr25k_qwen_v4.jsonl}"
WHITEN_NPZ="${WHITEN_NPZ:-${CACHE}/text_whiten.npz}"
WHITEN_GAMMA="${WHITEN_GAMMA:-0.25}"
TAG="${TAG:-v138b_v133a_protoCluster_K128_partialWhiten_gamma${WHITEN_GAMMA}}"
LOG="logs/${TAG}.log"
mkdir -p logs

if [ ! -f "$WHITEN_NPZ" ]; then
    echo "[run-v138b-K128] whitening .npz missing — computing it now ..."
    /home/yschoi/.conda/envs/dna_hashing/bin/python \
        scripts/build_text_whiten_matrix.py \
        --cache_dir "$CACHE" \
        --out       "$WHITEN_NPZ"
fi

echo "[run-v138b-K128] GPU=$GPU cache=$CACHE whiten=$WHITEN_NPZ gamma=$WHITEN_GAMMA K=128 (proto cluster + raw codon input) tag=$TAG"

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
    --router_type sinkhorn \
    --sinkhorn_epsilon_init 1.0 --sinkhorn_epsilon_final 0.1 \
    --routing_adaptive_topp \
    --routing_adaptive_topp_min 0.5 \
    --routing_adaptive_topp_max 0.9 \
    --codon_residual_gamma 0.0 \
    --codon_input_source routed \
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
    --lambda_vq 0.0 --lambda_quant 0.0 \
    --lambda_anchor 0.0 --lambda_dna 0.05 --lambda_bu 0.0 \
    --eta_base_balance 0.3 \
    --lambda_codeword_codon_sinkhorn 0.0 \
    --lambda_cibhash_ntxent 1.0 \
    --lambda_cibhash_kl 0.001 \
    --cibhash_mode per_codebook \
    --cibhash_temperature 0.3 \
    --cibhash_dynamic_tau \
    --cibhash_dynamic_tau_alpha 0.3 \
    --lambda_proto_cluster 0.1 \
    --proto_cluster_temperature 0.3 \
    --eval_every 5 \
    --post_eval_compositional \
    --dna_distance_mode base \
    -ev -s 2>&1 | tee "$LOG"
