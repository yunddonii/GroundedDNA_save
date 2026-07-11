#!/usr/bin/env bash
# CIFAR10 validation of Flickr25k CHAMPION recipe (v180 + wass 0.15 partial).
#
# Purpose:
#   Validate the universal recipe's transferability from Flickr25k (mid-scale,
#   scene-rich, 24-tag) to CIFAR10 (10-class object, 32x32 upscaled to 224).
#   CIFAR10 has no local-crop cache (FAIRrank not built for 32x32 sources) —
#   whole-image training/inference throughout.
#
# Delta vs scripts/train_flickr25k_v180_wass015_partial_dropXmodalSkip_clip.sh:
#   --dataset CIFAR10 (was Flickr25k)
#   --siglip2_feature_cache_dir cache/cifar10_clip (was ..._FAIRrankL8K3)
#   --qwen_text_cache_path cache/cifar10_qwen.jsonl
#   --codebook_size 64 (was 128; matches K=4^3 codon slots and 10-class scale)
#   --eval_cache_dir same as training cache (no crop suffix to strip)
#   Everything else IDENTICAL to Flickr champion.
#
# Skip flags (structural consistency v181):
#   text_code_kl_skip_global + text_hash_ntxent_skip_global (DROP xmodal_commit_skip)
#
# Usage: bash scripts/train_cifar10_flickrChamp_v180wass015_clip.sh <GPU_ID>
set -eu

GPU="${1:-3}"
CACHE="${CACHE:-./cache/cifar10_clip}"
QWEN="${QWEN:-./cache/cifar10_qwen.jsonl}"
WHITEN_NPZ="${WHITEN_NPZ:-${CACHE}/text_whiten.npz}"
WHITEN_GAMMA="${WHITEN_GAMMA:-0.25}"
K="${K:-64}"
TAG="${TAG:-cifar10_flickrChamp_v180wass015_K${K}_partialWhiten_g${WHITEN_GAMMA}}"
LOG="logs/${TAG}.log"
mkdir -p logs

if [ ! -f "$WHITEN_NPZ" ]; then
    echo "[cifar10-flickrChamp] whitening .npz missing — building ..."
    /home/yschoi/.conda/envs/dna_hashing/bin/python \
        scripts/build_text_whiten_matrix.py \
        --cache_dir "$CACHE" \
        --out       "$WHITEN_NPZ"
fi

echo "[cifar10-flickrChamp] GPU=$GPU cache=$CACHE qwen=$QWEN K=$K tag=$TAG"

CUDA_VISIBLE_DEVICES="$GPU" \
/home/yschoi/.conda/envs/dna_hashing/bin/python train_siglip2.py \
    --tag "$TAG" \
    --dataset CIFAR10 --setting 1 \
    --dataset_dir /home/yschoi/GroundedDNA/dataset \
    --num_devices 0 \
    -bs 64 -e 60 \
    --proj_lr 1e-3 \
    --num_workers 4 \
    --qwen_text_cache_path "$QWEN" \
    --siglip2_feature_cache_dir "$CACHE" \
    --backbone_type clip \
    --codebook_size "$K" \
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
    --eval_cache_dir "$CACHE" \
    -ev -s 2>&1 | tee "$LOG"
