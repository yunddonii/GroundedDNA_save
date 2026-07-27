#!/usr/bin/env bash
# v152b = v150a recipe with TWO single-delta changes (text-supervision FORM swap):
#   --lambda_text_hash_ntxent 0.05 -> 0.0  (OFF InfoNCE form of post-codon text-image matching)
#   --lambda_text_hash         0.0 -> 0.05 (ON  MSE form: MSE(text_cc, image_cc) on continuous_code)
#
# v150a's visual_token cibhash NtXent (image-image contrastive on pre-VQ
# semantic_visual_tokens) stays ON. Only the text-image matching FORM
# changes: InfoNCE -> MSE on the same continuous_code [B, 18, 4].
#
# Layer + form of text-image supervision:
#   Layer = post-codon_head (continuous_code [B, 18, 4])  -- same as v150a
#   Form  = MSE (per-element)                              -- v152b (this)
#   Form  = InfoNCE (symmetric, batch negatives)           -- v150a (reference)
#
# Motivation. v152a (image NtXent OFF, image-reconstruction ON via post-VQ
# FeatureDecoder) showed poor mid-eval and was discarded. User-driven
# pivot: keep v150a's strong visual_token NtXent + replace post-codon
# text NtXent with the older v91 MSE form to test whether the harder
# per-element supervision shape benefits codeword diversity (cb_tuple)
# without the NtXent's batch-saturation tendency.
#
# Code path (already in tree).
#   - --lambda_text_hash triggers the v91 text-only forward through
#     shared quantizer + codon_heads -> text_continuous_code.
#   - Loss: F.mse_loss(text_cc, u) over [B, 18, 4].
#   - Gradient flows back through codon_heads (shared) and text_adapter.
#
# Reference (Flickr25k-CLIP K=128, partial-whiten gamma=0.25):
#   v144a (no visual NtXent, ntxent form text):   mAP 0.7499, NMI 0.591, DNA 0.376, dead 0.065
#   v149a (continuous bits NtXent, ntxent text):  mAP 0.7354, NMI 0.628, DNA 0.380, dead 0.000
#   v150a (visual_token NtXent, ntxent text):     mAP 0.7330, DNA 0.265, NMI 0.648, dead 0.000, B1 0.159
#   v150b (visual_token NtXent + UOT + topp):     mAP 0.7509, P@1 0.9190, dead 0.000, B1 0.159, B2 0.098
#   v152b (visual_token NtXent + MSE text form):  TBD (this run)
#
# Usage: bash scripts/train_v152b_v150a_textHashMSE_K128_flickr25k_clip.sh <GPU_ID>
set -eu

GPU="${1:-1}"
CACHE="${CACHE:-./cache/flickr25k_clip_v4plus}"
QWEN="${QWEN:-./cache/flickr25k_qwen_v4.jsonl}"
WHITEN_NPZ="${WHITEN_NPZ:-${CACHE}/text_whiten.npz}"
WHITEN_GAMMA="${WHITEN_GAMMA:-0.25}"
TAG="${TAG:-v152b_v150a_textHashMSE_K128_partialWhiten_gamma${WHITEN_GAMMA}}"
LOG="logs/${TAG}.log"
mkdir -p logs

if [ ! -f "$WHITEN_NPZ" ]; then
    echo "[run-v152b] whitening .npz missing — computing it now ..."
    /home/yschoi/.conda/envs/dna_hashing/bin/python \
        scripts/build_text_whiten_matrix.py \
        --cache_dir "$CACHE" \
        --out       "$WHITEN_NPZ"
fi

echo "[run-v152b] GPU=$GPU cache=$CACHE whiten=$WHITEN_NPZ K=128 text-NtXent OFF + text-MSE ON (v150a base) tag=$TAG"

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
    --lambda_text_hash 0.05 \
    --lambda_text_hash_ntxent 0.0 \
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
