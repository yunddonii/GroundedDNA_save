#!/usr/bin/env bash
# v117b = v115c with --text_whiten_gamma changed from 0.25 -> 0.10.
#
# Rationale (from the v114c diagnostic):
#   On Flickr25k-CLIP, gamma=0.25 reduced cross-slot text cosine from
#   0.686 (raw) -> 0.210. This was very aggressive — v114c gained mAP
#   (+0.010) but lost DNA-uniq (-0.030). The v114-cosine analysis
#   identified a Goldilocks zone: DNA-uniq is preserved best when
#   transformed cosine stays in 0.4-0.6 range. gamma=0.10 should land
#   in that zone (whitening is mild — only the dominant principal
#   direction is partially down-weighted, sub-dominant signal preserved).
#
# Hypothesis:
#   gamma=0.10 -> cross-slot cos ≈ 0.4-0.5 (interpolating raw 0.686 and
#   gamma=0.25's 0.210). mAP gain should be smaller than gamma=0.25
#   (because anisotropy reduction is smaller), but DNA-uniq retention
#   should be better. Net: a better mAP / DNA / NMI trade-off.
#
# Everything else preserved from v115c:
#   --text_embed_transform partial_whiten
#   --lambda_codeword_codon_sinkhorn 0.0     (Sinkhorn bij OFF — v115)
#   --lambda_text_hash 0.0                    (text-DNA MSE OFF — v115)
#   --lambda_text_hash_ntxent 0.05           (text-DNA InfoNCE ON)
#   paired-aug NtXent per_codebook + dynamic tau
#
# References:
#   v115c (gamma=0.25): mAP 0.7622, DNA 0.212, P@1 0.888, NMI 0.598, B1 0.100
#
# Usage: bash scripts/train_v117b_v115c_partialWhiten_gamma0.10_flickr25k_clip.sh <GPU_ID>
set -eu

GPU="${1:-0}"
CACHE="${CACHE:-./cache/flickr25k_clip_v4plus}"
QWEN="${QWEN:-./cache/flickr25k_qwen_v4.jsonl}"
WHITEN_NPZ="${WHITEN_NPZ:-${CACHE}/text_whiten.npz}"
WHITEN_GAMMA="${WHITEN_GAMMA:-0.10}"
TAG="${TAG:-v117b_v115c_partialWhiten_gamma${WHITEN_GAMMA}}"
LOG="logs/${TAG}.log"
mkdir -p logs

if [ ! -f "$WHITEN_NPZ" ]; then
    echo "[run-v117b] whitening .npz missing — computing it now ..."
    /home/yschoi/.conda/envs/dna_hashing/bin/python \
        scripts/build_text_whiten_matrix.py \
        --cache_dir "$CACHE" \
        --out       "$WHITEN_NPZ"
fi

echo "[run-v117b] GPU=$GPU cache=$CACHE whiten=$WHITEN_NPZ gamma=$WHITEN_GAMMA tag=$TAG"

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
    --text_embed_transform partial_whiten \
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
