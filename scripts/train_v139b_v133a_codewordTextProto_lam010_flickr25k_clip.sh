#!/usr/bin/env bash
# v139b = v133a with ONE change:
#   ADD --lambda_codeword_text_proto 0.10  (+ tau, momentum, min_count defaults
#   from v123c) -- strengthen the per-codebook text-supervision via
#   EMA-tracked text prototype per (codebook m, codeword k).
#
# Recipe vs v133a (commit 1eecc32):
#   K=128, L=3, bij OFF, NO local-residual, per_codebook text NtXent
#   (M=6 InfoNCEs on [B,12]), cibhash per_codebook + dynamic tau,
#   partial whitening gamma=0.25.
#
# v139b hypothesis. v123c (the only prior codeword_text_proto run, with
# v122b base which had local-residual ON) used lambda=0.02 and saw mAP
# 0.7417 (vs v122b's 0.7607) -- partial regression because (a) lambda
# was too low to meaningfully push and (b) local-residual + codeword
# text proto both pulled in the same compositional direction. v139b
# applies it on the v133a base (which has localRes OFF, freeing the
# compositional budget for text-prototype supervision) AND at 2.5x the
# lambda (0.05).
#
# Mechanism. For each (codebook m, codeword k), the text prototype is
# an EMA mean of text_part_tokens[:, m] over samples whose visual slot
# m selected codeword k. The CE loss aligns z_m (encoder output for
# that slot) to its assigned codeword's text prototype. This pushes
# each codeword to be CO-AYNCHRONOUSLY visual + text discriminative,
# giving each codebook a *text-grounded prototype set*.
#
# Reference:
#   v122b (localRes ON, no text proto):       mAP 0.7607, P@1 0.9285, NMI 0.635
#   v123c (localRes ON, text proto lam=0.02): mAP 0.7417, P@1 0.8950 (regressed)
#   v133a (localRes OFF, no text proto):      mAP 0.7541, P@1 0.9150, NMI 0.626
#   v132a (localRes OFF, bij ON):             mAP 0.7418, P@1 0.9070, NMI 0.636
#   v126a (localRes ON, perCodon text):       mAP 0.7633 (K=128 mAP champ)
#   v139b (localRes OFF, text proto lam=0.10): TBD (this run)
#   v139b (localRes OFF, text proto lam=0.10): TBD (sibling)
#
# Usage: bash scripts/train_v139b_v133a_codewordTextProto_lam010_flickr25k_clip.sh <GPU_ID>
set -eu

GPU="${1:-0}"
CACHE="${CACHE:-./cache/flickr25k_clip_v4plus}"
QWEN="${QWEN:-./cache/flickr25k_qwen_v4.jsonl}"
WHITEN_NPZ="${WHITEN_NPZ:-${CACHE}/text_whiten.npz}"
WHITEN_GAMMA="${WHITEN_GAMMA:-0.25}"
TAG="${TAG:-v139b_v133a_codewordTextProto_lam010_K128_partialWhiten_gamma${WHITEN_GAMMA}}"
LOG="logs/${TAG}.log"
mkdir -p logs

if [ ! -f "$WHITEN_NPZ" ]; then
    echo "[run-v139b] whitening .npz missing — computing it now ..."
    /home/yschoi/.conda/envs/dna_hashing/bin/python \
        scripts/build_text_whiten_matrix.py \
        --cache_dir "$CACHE" \
        --out       "$WHITEN_NPZ"
fi

echo "[run-v139b] GPU=$GPU cache=$CACHE whiten=$WHITEN_NPZ gamma=$WHITEN_GAMMA K=128 (v133a + codeword_text_proto lam=0.10) tag=$TAG"

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
    --cibhash_dynamic_tau \
    --cibhash_dynamic_tau_alpha 0.3 \
    --lambda_codeword_text_proto 0.10 \
    --codeword_text_proto_tau 0.1 \
    --codeword_text_proto_momentum 0.95 \
    --codeword_text_proto_min_count 4 \
    --eval_every 5 \
    --post_eval_compositional \
    --dna_distance_mode base \
    -ev -s 2>&1 | tee "$LOG"
