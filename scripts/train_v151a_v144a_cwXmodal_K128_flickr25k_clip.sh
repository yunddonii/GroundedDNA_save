#!/usr/bin/env bash
# v151a = v144a recipe with TWO single-delta changes (text-supervision LAYER swap):
#   --lambda_text_hash_ntxent 0.05 -> 0.0    (OFF the post-codon_head NtXent)
#   --lambda_cw_xmodal        0.0  -> 0.05  (ON  the post-VQ pre-codon_head NtXent)
#
# Motivation. User-driven hypothesis: contrastive should attack the
# representation BEFORE codon_head bottlenecks the gradient through softmax.
# Current text_hash_ntxent operates on continuous_code [B, 18, 4] (post-
# codon_head softmax). cw_xmodal operates on quantized_tokens [B, M=6, D]
# (post-VQ, pre-codon_head) -- D-dim continuous embeddings at codeword level.
#
# Layer position of three contrastive losses (visual to text):
#   pre-VQ   semantic_visual_tokens [B, M, D]  -- (no cross-modal version yet)
#   post-VQ  quantized_tokens       [B, M, D]  -- v151a cw_xmodal  *** this layer ***
#   post-codon_head continuous_code [B, 18, 4]  -- v144a text_hash_ntxent
#
# Historical context. cw_xmodal was DISCARDED in v93a (2026-05-30) with
# mAP -0.025 on a much earlier recipe (no text_code_kl, no partial whiten,
# no UOT, no adaptive_topp, no cibhash per_codebook). v93a also reported
# B1 lift +20%, hinting at a compositional gain that retrieval cost
# masked. The current v144a recipe has all the recent regularizers, so
# the cw_xmodal collapse mode (cb0 re-coupling) may not reproduce.
#
# v151a vs v151b. v151a SWAPS the contrastive layer (replace post-codon_head
# with post-VQ). v151b STACKS both (parallel run).
#
# Reference (Flickr25k-CLIP K=128, partial-whiten gamma=0.25):
#   v144a (ref, post-codon_head NtXent only): mAP 0.7499, P@1 0.9170, DNA 0.376, NMI 0.591, dead 0.065, B1 0.125, B2 0.076
#   v150b (Pareto-better, visual_token NtXent + post-codon_head NtXent): mAP 0.7509, dead 0.000, B1 0.159, B2 0.098
#   v151a (text-supervision LAYER SWAPPED to post-VQ): TBD (this run)
#   v151b (text-supervision STACKED: post-VQ + post-codon_head):       TBD (parallel)
#
# Usage: bash scripts/train_v151a_v144a_cwXmodal_K128_flickr25k_clip.sh <GPU_ID>
set -eu

GPU="${1:-0}"
CACHE="${CACHE:-./cache/flickr25k_clip_v4plus}"
QWEN="${QWEN:-./cache/flickr25k_qwen_v4.jsonl}"
WHITEN_NPZ="${WHITEN_NPZ:-${CACHE}/text_whiten.npz}"
WHITEN_GAMMA="${WHITEN_GAMMA:-0.25}"
TAG="${TAG:-v151a_v144a_cwXmodal_swap_K128_partialWhiten_gamma${WHITEN_GAMMA}}"
LOG="logs/${TAG}.log"
mkdir -p logs

if [ ! -f "$WHITEN_NPZ" ]; then
    echo "[run-v151a] whitening .npz missing — computing it now ..."
    /home/yschoi/.conda/envs/dna_hashing/bin/python \
        scripts/build_text_whiten_matrix.py \
        --cache_dir "$CACHE" \
        --out       "$WHITEN_NPZ"
fi

echo "[run-v151a] GPU=$GPU cache=$CACHE whiten=$WHITEN_NPZ K=128 text-NtXent SWAP (post-codon_head OFF, post-VQ ON) tag=$TAG"

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
    --lambda_text_hash_ntxent 0.0 \
    --text_hash_ntxent_temperature 0.07 \
    --text_hash_ntxent_mode per_codebook \
    --lambda_cw_xmodal 0.05 \
    --cw_xmodal_temperature 0.07 \
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
    --eval_every 5 \
    --post_eval_compositional \
    --dna_distance_mode base \
    --lambda_text_code_kl 0.02 \
    --text_code_kl_tau_v 0.1 \
    --text_code_kl_tau_t 0.07 \
    --text_code_kl_conf_threshold 0.2 \
    --text_code_kl_skip_global \
    -ev -s 2>&1 | tee "$LOG"
