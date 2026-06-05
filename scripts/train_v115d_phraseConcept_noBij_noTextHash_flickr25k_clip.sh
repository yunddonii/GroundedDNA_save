#!/usr/bin/env bash
# v115d = v106b + Method 4 (phrase-level concept-centric text cache).
#
# Hypothesis
# ----------
# The default CLIP per-slot embedding pools an entire sentence
# ("red fabric, glossy metal, rough wooden textures") into one vector,
# diluting distinct concepts. Splitting on commas and conjunctions and
# re-encoding each phrase, then mean/attention-pooling the phrase vectors
# per slot, gives a concept-centric per-slot embedding while keeping the
# downstream cache layout identical.
#
# Prerequisites (cache rebuild, runs once)
# ----------------------------------------
#   python extract_clip_text_phrase_features.py \\
#         --qwen_cache ./cache/flickr25k_qwen_v4.jsonl \\
#         --donor_dir  ./cache/flickr25k_clip_v4plus \\
#         --out_dir    ./cache/flickr25k_clip_v4plus_phrase \\
#         --aggregation mean
# All visual_* / image_ids / has_text are symlinked from the donor so the
# resulting cache is drop-in.
#
# Recipe vs v106b
# ---------------
#   --siglip2_feature_cache_dir <phrase cache>  (REDIRECTED)
#   --text_embed_transform phrase_concept       (no-op flag; documents intent
#                                                in args.txt)
#
# Reference (v106b baseline): mAP 0.7407 / DNA 0.347 / NMI 0.604.
#
# Usage: bash scripts/train_v114d_phraseConcept_onV106b_flickr25k_clip.sh <GPU_ID>
set -eu

GPU="${1:-5}"
DONOR="${DONOR:-./cache/flickr25k_clip_v4plus}"
CACHE="${CACHE:-./cache/flickr25k_clip_v4plus_phrase}"
QWEN="${QWEN:-./cache/flickr25k_qwen_v4.jsonl}"
AGG="${AGG:-mean}"
TAG="${TAG:-v115d_v106b_noBij_noTextHash_phraseConcept_${AGG}}"
LOG="logs/${TAG}.log"
mkdir -p logs

if [ ! -f "$CACHE/text_part.f16.npy" ]; then
    echo "[run-v115d] phrase cache missing -- building it now ..."
    /home/yschoi/.conda/envs/dna_hashing/bin/python \
        extract_clip_text_phrase_features.py \
        --qwen_cache  "$QWEN" \
        --donor_dir   "$DONOR" \
        --out_dir     "$CACHE" \
        --aggregation "$AGG"
fi

echo "[run-v115d] GPU=$GPU cache=$CACHE qwen=$QWEN tag=$TAG"

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
    --text_embed_transform phrase_concept \
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
    --codeword_codon_sinkhorn_eps 0.1 \
    --codeword_codon_sinkhorn_iters 30 \
    --eval_every 5 \
    --post_eval_compositional \
    --dna_distance_mode base \
    -ev -s 2>&1 | tee "$LOG"
