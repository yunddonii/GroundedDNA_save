#!/usr/bin/env bash
# mscoco_v160b_qwen3v5b = mscoco_v160b + qwen3 caption rev v4 -> v5b
# (single delta: caption prompt regenerated with PROMPT_V5b which forces
# strict disjoint vocabulary per axis; everything else identical to
# mscoco_v160b).
#
# Caption stats (full 10K trainset):
#   v4 -> v5b: cross-slot vocab leak 74124 -> 45673 (-38.4%)
#              avg length 13.0 words -> 11.6 words (-1.4)
#
# v5b prompt design rationale: MSCOCO v4 captions have local↔local
# cosine 0.666 (Flickr 0.592). PROMPT_V5b adds explicit FORBIDDEN
# vocabulary lists per axis (object names only in primary/secondary,
# action verbs only in activity_or_relation, colors only in
# color_texture, scene words only in scene_type). C_global is the only
# multi-domain slot.
#
# Reference:
#   mscoco_v160b (v4 captions):     mAP 0.6134, P@1 0.897, DNA 0.119, NMI 0.726
#   mscoco_v160b_whitenG1:          mAP 0.6195, P@1 0.903, DNA 0.097, NMI 0.706 [aggressive]
#   mscoco_v160b_localResid:        mAP 0.6146, P@1 0.9042, DNA 0.124, NMI 0.721 [clean Pareto]
#   mscoco_v160b_qwen3v5b (HERE):   TBD
#
# Single delta vs mscoco_v160b:
#   --qwen_text_cache_path cache/mscoco_qwen3_v5b_trainset.jsonl
#   --siglip2_feature_cache_dir cache/mscoco_clip_v5b_FAIRrankL8K3
#
# (cache/mscoco_clip_v5b_FAIRrankL8K3 has new text_part.f16.npy computed from v5b
#  captions via extract_clip_text_features.py; visual_*/image_ids etc.
#  symlinked from cache/mscoco_clip_v4plus.)
#
# Predicted (analysis-driven):
#   mAP:    0.6134 -> 0.61-0.63 (caption redundancy ↓ -> retrieval should
#                                 hold or improve; less aggressive than
#                                 whitenG1 because input-level fix is
#                                 milder than ZCA whitening)
#   DNA:    0.119  -> 0.13-0.18 (caption disjoint -> codeword diversity ↑
#                                 -> codon collision ↓)
#   NMI:    0.726  -> 0.65-0.70 (cross-codebook redundancy ↓)
#   L↔L:   0.755  -> 0.69-0.73
#
# If v5b achieves mAP >= 0.61 AND DNA >= 0.13 AND NMI <= 0.70, this is
# the cleanest root-cause fix and becomes the preferred MSCOCO recipe.
#
# Usage: bash scripts/train_mscoco_v160b_qwen3v5b.sh <GPU_ID>
set -eu

GPU="${1:-2}"
CACHE="${CACHE:-./cache/mscoco_clip_v5b_FAIRrankL8K3}"
QWEN="${QWEN:-./cache/mscoco_qwen3_v5b_trainset.jsonl}"
WHITEN_NPZ="${WHITEN_NPZ:-${CACHE}/text_whiten.npz}"
WHITEN_GAMMA="${WHITEN_GAMMA:-0.25}"
TAG="${TAG:-mscoco_v170a_topp02_05_FAIRrankL8K3_stackedText_K128_partialWhiten_gamma${WHITEN_GAMMA}}"
LOG="logs/${TAG}.log"
mkdir -p logs

if [ ! -f "${CACHE}/text_part.f16.npy" ]; then
    echo "[mscoco-v160b-qwen3v5b] CLIP text cache missing at ${CACHE}/text_part.f16.npy"
    echo "[mscoco-v160b-qwen3v5b] build: python extract_clip_text_features.py \\"
    echo "         --qwen_cache cache/mscoco_qwen3_v5b_trainset.jsonl \\"
    echo "         --donor_dir cache/mscoco_clip_v4plus \\"
    echo "         --out_dir ${CACHE}"
    exit 1
fi
if [ ! -f "$WHITEN_NPZ" ]; then
    echo "[mscoco-v160b-qwen3v5b] whitening .npz missing — building from v5b cache ..."
    /home/yschoi/.conda/envs/dna_hashing/bin/python \
        scripts/build_text_whiten_matrix.py \
        --cache_dir "$CACHE" \
        --out       "$WHITEN_NPZ"
fi

echo "[mscoco-v160b-qwen3v5b] GPU=$GPU cache=$CACHE qwen=$QWEN whiten_gamma=$WHITEN_GAMMA tag=$TAG"

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
    --routing_adaptive_topp_min 0.2 \
    --routing_adaptive_topp_max 0.5 \
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
    --lambda_text_hash_ntxent 0.10 \
    --text_hash_ntxent_temperature 0.07 \
    --text_hash_ntxent_mode per_codebook \
    --lambda_wasserstein 0.05 \
    --lambda_vq 0.25 --lambda_quant 0.05 \
    --lambda_xmodal_commit 0.10 \
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
    --lambda_text_code_kl 0.10 \
    --text_code_kl_tau_v 0.1 \
    --text_code_kl_tau_t 0.07 \
    --text_code_kl_conf_threshold 0.2 \
    --text_code_kl_skip_global \
    -ev -s 2>&1 | tee "$LOG"
