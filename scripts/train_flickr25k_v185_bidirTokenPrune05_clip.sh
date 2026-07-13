#!/usr/bin/env bash
# v185 Flickr25k — BIDIRECTIONAL token pruning.
#
# Motivation: CUB v182 introduced visual-only per-slot token attention pruning
# (union across slots). User request 2026-07-11: extend to Flickr with a
# TEXT-side pruning step, and ensure that all downstream text embeddings
# come from the KEPT text tokens only (mean-pool over kept tokens rewrites
# text_part_raw BEFORE the whiten/adapter path).
#
# Mechanism (v185, model_siglip2.py):
#   Precondition: --backbone_type clip + cached_text_tokens available.
#   1. Project visual patches to shared 512-D via CLIP.visual_projection (frozen).
#   2. Compute per-slot per-patch-per-token cos-sim [B, M_loc, N, T].
#   3. Direction A (visual): softmax over t → sum → per-patch importance per
#      slot; keep top-K% patches per slot; UNION across slots → visual keep mask.
#   4. Direction B (text): softmax over n → sum → per-token importance per
#      slot; keep top-K% tokens per slot → text keep mask.
#   5. text_part_raw REBUILT: mean-pool cached_text_tokens over KEPT tokens
#      per slot. C_global (cb0) unchanged. Then downstream whiten +
#      per_slot_text_adapter + text_token_attention + losses observe ONLY
#      the pruned text embeddings.
#   6. visual_attention_mask combined with the visual keep mask before Sinkhorn
#      routing.
#
# Delta vs scripts/train_flickr25k_v180_wass015_partial_dropXmodalSkip_clip.sh:
#   +--bidirectional_token_prune
#   +--bidirectional_token_prune_visual_ratio 0.5
#   +--bidirectional_token_prune_text_ratio   0.5
#   (fg_ratio flag NOT set; the bidirectional block handles visual masking.)
#
# Usage: bash scripts/train_flickr25k_v185_bidirTokenPrune05_clip.sh <GPU_ID>
set -eu

GPU="${1:-0}"
CACHE="${CACHE:-./cache/flickr25k_clip_v4plus_qwen3_tokens_FAIRrankL8K3}"
QWEN="${QWEN:-./cache/flickr25k_qwen3_v4_trainset.jsonl}"
WHITEN_NPZ="${WHITEN_NPZ:-${CACHE}/text_whiten.npz}"
WHITEN_GAMMA="${WHITEN_GAMMA:-0.25}"
BI_V="${BI_V:-0.5}"
BI_T="${BI_T:-0.5}"
BIDIR_MODE="${BIDIR_MODE:-legacy}"
TAG="${TAG:-flickr25k_v185_bidir_v${BI_V}_t${BI_T}_K128_partialWhiten_gamma${WHITEN_GAMMA}}"
LOG="logs/${TAG}.log"
mkdir -p logs

if [ ! -f "$WHITEN_NPZ" ]; then
    echo "[v185] whitening .npz missing — building ..."
    /home/yschoi/.conda/envs/dna_hashing/bin/python \
        scripts/build_text_whiten_matrix.py \
        --cache_dir "$CACHE" \
        --out       "$WHITEN_NPZ"
fi

echo "[v185] GPU=$GPU cache=$CACHE bi_v=$BI_V bi_t=$BI_T tag=$TAG"

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
    --sinkhorn_lambda_a 1.0 \
    --sinkhorn_lambda_b 1.0 \
    --routing_adaptive_topp \
    --routing_adaptive_topp_min 0.3 \
    --routing_adaptive_topp_max 0.7 \
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
    --lambda_wasserstein 0.15 \
    --lambda_vq 0.25 --lambda_quant 0.05 \
    --lambda_xmodal_commit 0.05 \
    --lambda_anchor 0.05 --lambda_dna 0.05 --lambda_bu 0.02 \
    --eta_base_balance 0.3 \
    --lambda_codeword_codon_sinkhorn 0.0 \
    --lambda_cibhash_ntxent 1.0 \
    --lambda_cibhash_kl 0.001 \
    --cibhash_mode per_codebook \
    --cibhash_temperature 0.3 \
    --cibhash_dynamic_tau \
    --cibhash_dynamic_tau_alpha 0.3 \
    --cibhash_ntxent_continuous \
    --cibhash_ntxent_source visual_token \
    --eval_every 5 \
    --post_eval_compositional \
    --dna_distance_mode base \
    --lambda_text_code_kl 0.05 \
    --text_code_kl_tau_v 0.1 \
    --text_code_kl_tau_t 0.07 \
    --text_code_kl_conf_threshold 0.2 \
    --text_code_kl_skip_global \
    --text_hash_ntxent_skip_global \
    --bidirectional_token_prune \
    --bidirectional_token_prune_visual_ratio "$BI_V" \
    --bidirectional_token_prune_text_ratio "$BI_T" \
    --bidirectional_token_prune_mode "$BIDIR_MODE" \
    --eval_cache_dir ./cache/flickr25k_clip_v4plus_qwen3_tokens \
    -ev -s 2>&1 | tee "$LOG"
