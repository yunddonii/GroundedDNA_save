#!/usr/bin/env bash
# CUB-200 port of the Flickr25k-CLIP family champion v160b
# (= v150b recipe + Eq.(8) xmodal commit). Single-object fine-grained
# bird dataset; uses V6 anatomical Qwen3 captions (head/wing/underparts/
# tail/background disjoint axes), CLIP ViT-B/16 frozen backbone, K=64,
# partial-whiten gamma=0.25.
#
# Recipe is byte-identical to
#   scripts/train_v160b_v150b_xmodalCommit_K64_flickr25k_clip.sh
# except dataset=CUB_200 and the cache / qwen paths.
#
# Usage: bash scripts/train_cub200_v160b_v6b_K64_clip.sh <GPU_ID>
set -eu

GPU="${1:-0}"
CACHE="${CACHE:-./cache/cub200_clip_v6bplus_FAIRrankL8K3}"
QWEN="${QWEN:-./cache/cub200_qwen_v6b_trainset.jsonl}"
WHITEN_NPZ="${WHITEN_NPZ:-${CACHE}/text_whiten.npz}"
WHITEN_GAMMA="${WHITEN_GAMMA:-0.25}"
TAG="${TAG:-cub200_v170a_wass015_textPreqContrastive_FAIRrankL8K3_partialWhiten_gamma${WHITEN_GAMMA}}"
LOG="logs/${TAG}.log"
mkdir -p logs

if [ ! -f "$WHITEN_NPZ" ]; then
    echo "[run-cub-v160b] whitening .npz missing — computing it now ..."
    /home/yschoi/.conda/envs/dna_hashing/bin/python \
        scripts/build_text_whiten_matrix.py \
        --cache_dir "$CACHE" \
        --out       "$WHITEN_NPZ"
fi

echo "[run-cub-v160b] GPU=$GPU cache=$CACHE whiten=$WHITEN_NPZ K=64 tag=$TAG"

CUDA_VISIBLE_DEVICES="$GPU" \
/home/yschoi/.conda/envs/dna_hashing/bin/python train_siglip2.py \
    --tag "$TAG" \
    --dataset CUB_200 --setting 1 \
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
    --global_gate_init_logit -3.0 \
    --router_type sinkhorn \
    --sinkhorn_epsilon_init 1.0 --sinkhorn_epsilon_final 0.1 \
    --sinkhorn_lambda_a 1.0 \
    --sinkhorn_lambda_b 1.0 \
    --routing_adaptive_topp \
    --routing_adaptive_topp_min 0.6 \
    --routing_adaptive_topp_max 1.0 \
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
    --lambda_text_hash_ntxent 0.0 \
    --text_hash_ntxent_temperature 0.07 \
    --text_hash_ntxent_mode per_codebook \
    --lambda_wasserstein 0.15 \
    --lambda_vq 0.25 --lambda_quant 0.05 \
    --lambda_xmodal_commit 0.10 \
    --lambda_anchor 0.05 --lambda_dna 0.05 --lambda_bu 0.02 \
    --eta_base_balance 1.0 \
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
    --lambda_text_code_kl 0.10 \
    --text_code_kl_tau_v 0.1 \
    --text_code_kl_tau_t 0.07 \
    --text_code_kl_conf_threshold 0.2 \
    --text_code_kl_skip_global \
    --lambda_text_preq_contrastive 0.10 \
    --text_preq_contrastive_tau 0.07 \
    --text_preq_contrastive_skip_global \
    --eval_cache_dir ./cache/cub200_clip_v6bplus \
    -ev -s 2>&1 | tee "$LOG"
