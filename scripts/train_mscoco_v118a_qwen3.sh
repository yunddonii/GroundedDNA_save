#!/usr/bin/env bash
# mscoco_v118a = v118a recipe ported to MSCOCO (Qwen3 captions).
#
# Recipe = Flickr25k v118a (the new mAP champion at K=128):
#   --codebook_size 128
#   --text_embed_transform partial_whiten --text_whiten_gamma 0.25
#     (whitening matrix built on MSCOCO has_text rows only:
#      10k images * 6 slots = 60k vectors, top1 = 14.1 % variance,
#      vs Flickr 14.9 % -- anisotropy is dataset-independent CLIP
#      property)
#   --lambda_codeword_codon_sinkhorn 0.0      (bij OFF; v115 family)
#   --lambda_text_hash 0.0                    (MSE OFF; v115 family)
#   --lambda_text_hash_ntxent 0.05            (text-DNA InfoNCE kept)
#   paired-aug NtXent per_codebook + dynamic tau
#
# Flickr25k v118a reference:
#   mAP 0.7721 (+0.031 vs v106b, +0.010 vs v115c), DNA 0.207,
#   P@1 0.900, NMI 0.628, B1 0.107, B2 0.064, dead 0.004.
#
# MSCOCO mscoco_v106b reference (DB=107K, K=128, with Sinkhorn bij):
#   mAP not yet logged in PROJECT_LOG (mid-eval-only). Used as point
#   of comparison once both runs are done. CIBHash-CLIP MSCOCO
#   external baseline = 0.5842.
#
# Hypothesis on MSCOCO:
#   With K=128 + 4^3=64 codons, pigeonhole forces 2x DNA collisions;
#   v118a's bij-free recipe should let the model distribute the
#   collisions more evenly (vs v106b's marginal-constrained 2x
#   uniform spread), and the partial whitening should reduce the
#   text anisotropy that disproportionately affects MSCOCO's only
#   8.2 % has_text coverage (less text signal per image -> stronger
#   need for clean text geometry).
#
# Usage: bash scripts/train_mscoco_v118a_qwen3.sh <GPU_ID>
set -eu

GPU="${1:-1}"
CACHE="${CACHE:-./cache/mscoco_clip_v4plus}"
QWEN="${QWEN:-./cache/mscoco_qwen3_v4_trainset.jsonl}"
WHITEN_NPZ="${WHITEN_NPZ:-${CACHE}/text_whiten.npz}"
WHITEN_GAMMA="${WHITEN_GAMMA:-0.25}"
TAG="${TAG:-mscoco_v118a_qwen3_partialWhiten_gamma${WHITEN_GAMMA}_K128}"
LOG="logs/${TAG}.log"
mkdir -p logs

if [ ! -f "$WHITEN_NPZ" ]; then
    echo "[run-mscoco-v118a] whitening .npz missing — computing it now ..."
    /home/yschoi/.conda/envs/dna_hashing/bin/python \
        scripts/build_text_whiten_matrix.py \
        --cache_dir "$CACHE" \
        --out       "$WHITEN_NPZ"
fi

echo "[run-mscoco-v118a] GPU=$GPU cache=$CACHE qwen=$QWEN whiten=$WHITEN_NPZ gamma=$WHITEN_GAMMA K=128 tag=$TAG"

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
