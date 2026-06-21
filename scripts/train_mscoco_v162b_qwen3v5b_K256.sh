#!/usr/bin/env bash
# mscoco_v162b_qwen3v5b = mscoco_v162b base + qwen3 caption rev v4 -> v5b
# (single delta: caption prompt regenerated with PROMPT_V5b).
#
# Tests whether grounded text routing (Stage-2 OT top-k_t token pruning)
# replicates on MSCOCO once the v4 caption redundancy is removed. On
# MSCOCO v4 captions, v162b lost mAP -0.021 because top-k pruned only
# the same shared global tokens; v5b has -38.4% cross-slot vocab leak
# and length -1.4 words, so each codebook's text tokens are more
# distinct -- grounded routing should now select genuinely
# axis-specific tokens.
#
# Reference (qwen3 v4 captions):
#   mscoco_v160b (ref):            mAP 0.6134, P@1 0.897, DNA 0.119, NMI 0.726
#   mscoco_v162b (v4 + grounded):  mAP 0.5924 (-0.021) ✗  -- DISCARDED on v4
#   mscoco_v160b_qwen3v5b (HERE base): mid-eval ep 24 mAP 0.6273 (still rising)
#   mscoco_v162b_qwen3v5b (THIS RUN):  TBD
#
# Single delta vs mscoco_v162b (qwen3 v4):
#   --qwen_text_cache_path cache/mscoco_qwen3_v5b_trainset.jsonl
#   --siglip2_feature_cache_dir cache/mscoco_clip_v5b_tokens
#
# Predicted:
#   - If v5b caption disjointness recovers grounded routing's value:
#       mAP >= 0.62, DNA >= 0.12, NMI < 0.72
#   - If v5b doesn't recover it (redundancy was secondary to MSCOCO's
#       large db scale):
#       mAP ~ 0.60-0.61, similar regression as v162b on v4
#
# Usage: bash scripts/train_mscoco_v162b_qwen3v5b.sh <GPU_ID>
set -eu

GPU="${1:-1}"
CACHE="${CACHE:-./cache/mscoco_clip_v5b_tokens}"
QWEN="${QWEN:-./cache/mscoco_qwen3_v5b_trainset.jsonl}"
WHITEN_NPZ="${WHITEN_NPZ:-${CACHE}/text_whiten.npz}"
WHITEN_GAMMA="${WHITEN_GAMMA:-0.25}"
TAG="${TAG:-mscoco_v162b_qwen3v5b_v160b_groundedTextRouting_K256_partialWhiten_gamma${WHITEN_GAMMA}}"
LOG="logs/${TAG}.log"
mkdir -p logs

if [ ! -f "${CACHE}/text_tokens.f16.npy" ]; then
    echo "[mscoco-v162b-qwen3v5b] text_tokens.f16.npy missing at ${CACHE}"
    echo "[mscoco-v162b-qwen3v5b] run: python extract_clip_text_token_features.py \\"
    echo "         --qwen_cache cache/mscoco_qwen3_v5b_trainset.jsonl \\"
    echo "         --donor_dir cache/mscoco_clip_v5b \\"
    echo "         --out_dir ${CACHE}"
    exit 1
fi
if [ ! -f "$WHITEN_NPZ" ]; then
    echo "[mscoco-v162b-qwen3v5b] whitening .npz missing — building ..."
    /home/yschoi/.conda/envs/dna_hashing/bin/python \
        scripts/build_text_whiten_matrix.py \
        --cache_dir "$CACHE" \
        --out       "$WHITEN_NPZ"
fi

echo "[mscoco-v162b-qwen3v5b] GPU=$GPU cache=$CACHE qwen=$QWEN tag=$TAG"

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
    --codebook_size 256 \
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
    --grounded_text_routing \
    --grounded_text_k_t 5 \
    --grounded_text_eps 0.05 \
    --grounded_text_stage1_sg \
    --grounded_text_skip_global \
    -ev -s 2>&1 | tee "$LOG"
