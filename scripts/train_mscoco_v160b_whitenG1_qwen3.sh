#!/usr/bin/env bash
# mscoco_v160b_whitenG1 = mscoco_v160b + WHITEN_GAMMA 0.25 -> 1.0 (single delta).
#
# Motivation. MSCOCO Qwen3 v4 captions show strong cross-codebook
# redundancy: intra-image local↔local cosine sim = 0.666 vs Flickr's
# 0.592 (+0.075). Codebook C_activity_or_relation overlaps every other
# local codebook at ~0.70 cosine — the activity caption repeats subject /
# scene / color info. Result: NMI 0.726, DNA 0.119, collision ratio
# 1.72× (vs Flickr v160b's 0.625 / 0.423 / 1.32×).
#
# Hypothesis. Stronger partial whitening (γ=1.0 = full ZCA whitening
# along the dominant cross-codebook variance directions) removes the
# common semantic axis from text_part_tokens BEFORE the text adapter
# sees it, forcing each codebook's adapter to focus on the residual
# codebook-specific signal.
#
# γ semantics: text' = (1-γ) * text + γ * (text - mean) * U S^-0.5 U^T
#   γ = 0:    no whitening (raw CLIP text)
#   γ = 0.25: light whitening (current v160b default)
#   γ = 1.0:  full ZCA whitening on the top-(D-1) directions of the
#             cross-codebook covariance matrix (kept zero-mean too)
#
# Reference (qwen3 v4_trainset caption):
#   mscoco_v160b (γ=0.25): mAP 0.6134, P@1 0.897, DNA 0.119, cbT 0.205,
#                          NMI 0.726, L↔L 0.755, B2 0.166, collision 1.72×
#   mscoco_v160b_whitenG1 (γ=1.0, THIS RUN): TBD
#
# Predicted (analysis-driven):
#   mAP:    0.6134 -> 0.59-0.62 (small risk from text channel disruption)
#   DNA:    0.119  -> 0.14-0.18 (caption redundancy ↓ -> codon collision ↓)
#   cbT:    0.205  -> 0.22-0.30 (codeword diversity ↑)
#   NMI:    0.726  -> 0.66-0.72 (semantic axis removed -> NMI ↓ better)
#   L↔L:    0.755  -> 0.68-0.73
#   collision ratio: 1.72× -> 1.45-1.60×
#
# Usage: bash scripts/train_mscoco_v160b_whitenG1_qwen3.sh <GPU_ID>
set -eu

GPU="${1:-4}"
CACHE="${CACHE:-./cache/mscoco_clip_v4plus}"
QWEN="${QWEN:-./cache/mscoco_qwen3_v4_trainset.jsonl}"
WHITEN_NPZ="${WHITEN_NPZ:-${CACHE}/text_whiten.npz}"
WHITEN_GAMMA="${WHITEN_GAMMA:-1.0}"
TAG="${TAG:-mscoco_v160b_whitenG1_v150b_xmodalCommit_0p025_qwen3_K128_partialWhiten_gamma${WHITEN_GAMMA}}"
LOG="logs/${TAG}.log"
mkdir -p logs

if [ ! -f "$WHITEN_NPZ" ]; then
    echo "[mscoco-v160b-whitenG1] whitening .npz missing — building ..."
    /home/yschoi/.conda/envs/dna_hashing/bin/python \
        scripts/build_text_whiten_matrix.py \
        --cache_dir "$CACHE" \
        --out       "$WHITEN_NPZ"
fi

echo "[mscoco-v160b-whitenG1] GPU=$GPU cache=$CACHE qwen=$QWEN whiten_gamma=$WHITEN_GAMMA tag=$TAG"

CUDA_VISIBLE_DEVICES="$GPU" \
/home/yschoi/.conda/envs/dna_hashing/bin/python train_siglip2.py \
    --tag "$TAG" \
    --dataset MSCOCO --setting 1 \
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
    --lambda_wasserstein 0.05 \
    --lambda_vq 0.25 --lambda_quant 0.05 \
    --lambda_xmodal_commit 0.025 \
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
    --lambda_text_code_kl 0.02 \
    --text_code_kl_tau_v 0.1 \
    --text_code_kl_tau_t 0.07 \
    --text_code_kl_conf_threshold 0.2 \
    --text_code_kl_skip_global \
    -ev -s 2>&1 | tee "$LOG"
