#!/usr/bin/env bash
# v155a = v144a recipe with ONE single-delta change (no flag flip — REMOVAL of action_store_true):
#   --cibhash_dynamic_tau  (currently ON) -> OMITTED (OFF, static T)
#   --cibhash_dynamic_tau_alpha 0.3 -> 0.0 (defensive zeroing)
#
# Motivation. The cibhash dynamic-tau per-pair temperature
#   tau_ij = T * (1 + alpha * cos(text_i, text_j))
# softens the NtXent push for pairs with HIGH text similarity. The intent
# is semantic preservation (avoid pushing apart "near-positives"), but in
# Flickr25k's dense caption regime (typical text-cos 0.5-0.7) it
# systematically suppresses uniformity for negatives that share keywords
# with the positive. User hypothesis: this directly limits DNA-uniq by
# letting visually-distinct-but-caption-similar images converge to the
# same code.
#
# Single delta from v144a: dynamic tau OFF -> uniform static T=0.3 across
# all pairs. All other v144a flags unchanged (partial whiten, text_code_kl
# 0.02, cibhash per_codebook + lambda 1.0, adaptive_topp, etc.).
#
# Predicted (analysis-driven):
#   DNA-uniq: 0.376 -> 0.42-0.48 (uniformity restored on caption-similar pairs)
#   cb_tuple: 0.664 -> 0.68-0.72 (codeword combinations diversify)
#   mAP:      0.7499 -> 0.73-0.75 (slight risk; semantic cluster preservation lost)
#   NMI:      0.591 -> 0.58-0.61 (slight improvement; codes more dispersed)
#
# Reference (Flickr25k-CLIP K=128, partial-whiten gamma=0.25):
#   v144a (dyn_tau ON,  no bij):  mAP 0.7499, DNA 0.376, cb_tuple 0.664
#   v154a (dyn_tau ON,  bij 0.1): mAP 0.7350, DNA 0.396, cb_tuple 0.659
#   v155a (dyn_tau OFF, no bij):  TBD (this run)
#
# Usage: bash scripts/train_v155a_v144a_cibhashDynTauOff_K128_flickr25k_clip.sh <GPU_ID>
set -eu

GPU="${1:-0}"
CACHE="${CACHE:-./cache/flickr25k_clip_v4plus}"
QWEN="${QWEN:-./cache/flickr25k_qwen_v4.jsonl}"
WHITEN_NPZ="${WHITEN_NPZ:-${CACHE}/text_whiten.npz}"
WHITEN_GAMMA="${WHITEN_GAMMA:-0.25}"
TAG="${TAG:-v155a_v144a_cibhashDynTauOff_K128_partialWhiten_gamma${WHITEN_GAMMA}}"
LOG="logs/${TAG}.log"
mkdir -p logs

if [ ! -f "$WHITEN_NPZ" ]; then
    echo "[run-v155a] whitening .npz missing — computing it now ..."
    /home/yschoi/.conda/envs/dna_hashing/bin/python \
        scripts/build_text_whiten_matrix.py \
        --cache_dir "$CACHE" \
        --out       "$WHITEN_NPZ"
fi

echo "[run-v155a] GPU=$GPU cache=$CACHE whiten=$WHITEN_NPZ K=128 cibhash_dynamic_tau OFF (static T=0.3) tag=$TAG"

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
    --cibhash_dynamic_tau_alpha 0.0 \
    --eval_every 5 \
    --post_eval_compositional \
    --dna_distance_mode base \
    --lambda_text_code_kl 0.02 \
    --text_code_kl_tau_v 0.1 \
    --text_code_kl_tau_t 0.07 \
    --text_code_kl_conf_threshold 0.2 \
    --text_code_kl_skip_global \
    -ev -s 2>&1 | tee "$LOG"
