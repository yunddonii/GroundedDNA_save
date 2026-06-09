#!/usr/bin/env bash
# v129a = v128a with ONE additional change:
#   --lambda_codeword_codon_sinkhorn 0.1 -> 0.0  (bij OFF)
#
# v128a = v126a EXACT reproduction with text_hash_ntxent_mode per_codon
#   -> per_codebook (M=6 InfoNCEs over [B, L*4]=[B, 12] codon segments).
#
# v129a isolates the per_codebook text supervision effect *without* the
# bij (codeword<->codon Sinkhorn permutation) confounder. The bij is the
# v106b DNA-uniqueness mechanism and its removal at K=128 (L=3) collapses
# DNA-uniq from ~0.43 to ~0.24-0.34 historically. The question this run
# answers:
#
#   When text supervises each codebook DIRECTLY (per_codebook InfoNCE at
#   the right contribution-#2 granularity), does the model recover any of
#   the DNA-uniq lost by removing the bij? Or does per_codebook only
#   improve compositional (B1/B2) metrics regardless of bij?
#
# Recipe vs v128a: ONLY `--lambda_codeword_codon_sinkhorn 0.1 -> 0.0`.
# All other v126a-inherited flags identical (local_residual_quant + text,
# CIBHash per_codebook + dynamic tau, partial whitening gamma=0.25,
# paired-aug DNA NtXent lambda=0, text_hash MSE OFF, reconstruction
# lambda kept at 1.0 with use_decoder default).
#
# References (Flickr25k-CLIP K=128, partial whitening gamma=0.25):
#   v122b (local-res, global text-hash, bij ON):     mAP 0.7607, P@1 0.9285, DNA 0.339
#   v125d (local-res, global text-hash, bij ON):     mAP 0.7385, P@1 0.9165, DNA 0.402
#   v126a (local-res, per_codon text-hash, bij ON):  mAP 0.7633, P@1 0.917,  DNA 0.4291
#   v127a (per_codebook + bij OFF, OLD code):        mAP 0.7365, P@1 0.9225, DNA 0.2384
#                                                    (used reverted per_codebook impl;
#                                                     v129a uses freshly re-implemented
#                                                     code matching EXACT v126a base)
#   v128a (v126a EXACT + per_codebook, bij ON):      TBD (still running on GPU 1)
#   v129a (v128a + bij OFF):                         TBD (this run)
#
# Hypothesis. per_codebook text supervision at the correct contribution-#2
# granularity may PARTIALLY compensate for bij removal (some DNA-uniq
# recovery vs v127a's 0.2384 baseline) by giving each codebook its own
# text-driven discriminative pressure. Expected:
#   - DNA-uniq: between v127a (0.24) and v126a (0.43); probably 0.30-0.35
#   - P@1: 0.92+ (per_codebook + bij OFF preserves top-1, like v127a)
#   - mAP: comparable to or slightly below v128a (bij removal cost)
#   - B1/B2: matched or improved vs v128a (per_codebook is supposed to
#     boost compositional lift)
#
# Usage: bash scripts/train_v129a_v128a_perCodebook_noBij_K128_flickr25k_clip.sh <GPU_ID>
set -eu

GPU="${1:-2}"
CACHE="${CACHE:-./cache/flickr25k_clip_v4plus}"
QWEN="${QWEN:-./cache/flickr25k_qwen_v4.jsonl}"
WHITEN_NPZ="${WHITEN_NPZ:-${CACHE}/text_whiten.npz}"
WHITEN_GAMMA="${WHITEN_GAMMA:-0.25}"
TAG="${TAG:-v129a_v128a_perCodebook_noBij_K128_partialWhiten_gamma${WHITEN_GAMMA}}"
LOG="logs/${TAG}.log"
mkdir -p logs

if [ ! -f "$WHITEN_NPZ" ]; then
    echo "[run-v129a] whitening .npz missing — computing it now ..."
    /home/yschoi/.conda/envs/dna_hashing/bin/python \
        scripts/build_text_whiten_matrix.py \
        --cache_dir "$CACHE" \
        --out       "$WHITEN_NPZ"
fi

echo "[run-v129a] GPU=$GPU cache=$CACHE whiten=$WHITEN_NPZ gamma=$WHITEN_GAMMA K=128 v128a + bij OFF tag=$TAG"

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
    --local_residual_quant \
    --local_residual_gamma 1.0 \
    --local_residual_text \
    --local_residual_detach_global \
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
    --cibhash_dynamic_tau \
    --cibhash_dynamic_tau_alpha 0.3 \
    --eval_every 5 \
    --post_eval_compositional \
    --dna_distance_mode base \
    -ev -s 2>&1 | tee "$LOG"
