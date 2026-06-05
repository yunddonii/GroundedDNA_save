#!/usr/bin/env bash
# v117e = v115c (partialWhiten γ=0.25 noBij+noTextHashMSE) modified:
#   (1) Transform changed: partial_whiten -> global_residual_whiten
#       - C_global (slot 0) kept UNTRANSFORMED (raw)
#       - Local slots (1..5) first residualized against slot 0
#         (T_local <- T_local - T_global) then partial-whitened
#         using stats computed on the *residualized* population
#   (2) --lambda_text_hash_ntxent 0.05 (v115c had 0.05)
#       So NO direct text-DNA <-> visual-DNA cross-modal loss remains;
#       text supervision now flows only through anchor + paired-aug NtXent
#       (dynamic tau) + wasserstein.
#
# Whitening stats:
#   built on FULL Flickr25k (25k images, has_text=True) * 5 LOCAL slots
#   (i.e. 125,000 vectors of T_local_residualized). top1 = 13.0% of
#   variance, top5 = 23.4% (vs raw text 14.9% / 26.1%).
#
# Reference (v115c, raw partialWhiten γ=0.25):
#   mAP 0.7622, DNA 0.212, P@1 0.888, NMI 0.598, B1 0.100
#
# Usage: bash scripts/train_v117e_v117a_globalResidualWhiten_withTextHashNtxent_flickr25k_clip.sh <GPU_ID>
set -eu

GPU="${1:-5}"
CACHE="${CACHE:-./cache/flickr25k_clip_v4plus}"
QWEN="${QWEN:-./cache/flickr25k_qwen_v4.jsonl}"
WHITEN_NPZ="${WHITEN_NPZ:-${CACHE}/text_whiten_localres.npz}"
WHITEN_GAMMA="${WHITEN_GAMMA:-0.25}"
TAG="${TAG:-v117e_v117a_globalResidualWhiten_withTextHashNtxent_gamma${WHITEN_GAMMA}}"
LOG="logs/${TAG}.log"
mkdir -p logs

if [ ! -f "$WHITEN_NPZ" ]; then
    echo "[run-v117e] residualized whitening .npz missing — computing it now ..."
    /home/yschoi/.conda/envs/dna_hashing/bin/python \
        scripts/build_text_whiten_matrix.py \
        --cache_dir "$CACHE" \
        --out       "$WHITEN_NPZ" \
        --residualize_first
fi

echo "[run-v117e] GPU=$GPU cache=$CACHE whiten=$WHITEN_NPZ gamma=$WHITEN_GAMMA tag=$TAG"

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
    --text_embed_transform global_residual_whiten \
    --text_whiten_npz "$WHITEN_NPZ" \
    --text_whiten_gamma "$WHITEN_GAMMA" \
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
    --lambda_text_hash 0.0 \
    --lambda_text_hash_ntxent 0.05 \
    --text_hash_ntxent_temperature 0.07 \
    --lambda_wasserstein 0.05 \
    --lambda_vq 0.25 --lambda_quant 0.05 \
    --lambda_anchor 0.05 --lambda_dna 0.05 --lambda_bu 0.02 \
    --eta_base_balance 0.3 \
    --lambda_codeword_codon_sinkhorn 0.0 \
    --eval_every 5 \
    --post_eval_compositional \
    --dna_distance_mode base \
    -ev -s 2>&1 | tee "$LOG"
