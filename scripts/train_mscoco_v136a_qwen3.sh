#!/usr/bin/env bash
# mscoco_v136a = MSCOCO port of Flickr v136a (= v132a - bij + posSpec head).
#
# Two changes vs mscoco_v132a:
#   (1) --lambda_codeword_codon_sinkhorn 0.1 -> 0.0  (bij OFF)
#   (2) ADD --codon_position_specific_head             (v69a flag; per-codon
#       position separate Linear(chunk, 4); 3 fc layers replace the shared
#       one to let codon positions 0/1/2 specialize)
#
# This is the MOST direct test of the user's original hypothesis: position-
# specific heads worked on MSCOCO historically (mscoco_v69a/v81a/v78a era),
# so applying them to the modern v132a-family recipe (per_codebook text
# NtXent + cibhash per_codebook + partial whitening + adaptive top-p +
# noLocalRes) should show whether the historical MSCOCO posSpec gain
# composes with the new compositional + retrieval recipe.
#
# References on MSCOCO (DB=107K, K=128):
#   mscoco_v132a (bij ON,  shared head):    mAP 0.5534, P@1 0.8044, NMI 0.693, B2 0.143, dead 0.014
#   mscoco_v133a (bij OFF, shared head):    mAP 0.5652, P@1 0.7976, NMI 0.687, B2 0.141
#   mscoco_v135a (bij OFF, shared head, K=256): TBD (running)
#   mscoco_v136a (bij OFF, *posSpec head*, K=128): TBD (this run)
#   mscoco_v81a (legacy posSpec champion):  mAP 0.4891 (50-percentile shifted-K=128)
#   CIBHash-CLIP MSCOCO baseline:           mAP 0.5842, P@1 0.9264
#
# Hypothesis on MSCOCO:
#   (a) mAP: should rise vs mscoco_v133a's 0.5652 toward CIBHash's 0.5842
#       (historical posSpec gain +0.0096 on MSCOCO at K=128, but the
#       modern recipe already has cibhash + per_codebook text + partial
#       whitening so the absolute gain may be smaller).
#   (b) P@1: mscoco_v132a has 0.8044 (bij ON, no posSpec) and v133a has
#       0.7976 (bij OFF, no posSpec). posSpec should help most when bij
#       is OFF -- prediction ~0.80-0.81.
#   (c) DNA-uniq: K=128 / |C|=64 gives 2x pigeonhole at L=3. Without bij
#       distributing the collisions uniformly, DNA-uniq might stay below
#       0.13 (mscoco_v132a) but cb-tuple should rise above 0.27.
#   (d) NMI: mscoco_v133a 0.687, mscoco_v132a 0.693. posSpec gives each
#       codon position independent decoding so could trade compositional
#       NMI for per-position discriminability. Expected 0.67-0.69.
#   (e) Dead: mscoco_v132a (with bij) had dead 0.014 (excellent). Without
#       bij + posSpec, dead could rise to 0.04 range like mscoco_v133a.
#
# Usage: bash scripts/train_mscoco_v136a_qwen3.sh <GPU_ID>
set -eu

GPU="${1:-3}"
CACHE="${CACHE:-./cache/mscoco_clip_v4plus}"
QWEN="${QWEN:-./cache/mscoco_qwen3_v4_trainset.jsonl}"
WHITEN_NPZ="${WHITEN_NPZ:-${CACHE}/text_whiten.npz}"
WHITEN_GAMMA="${WHITEN_GAMMA:-0.25}"
TAG="${TAG:-mscoco_v136a_qwen3_noBij_posSpec_K128_partialWhiten_gamma${WHITEN_GAMMA}}"
LOG="logs/${TAG}.log"
mkdir -p logs

if [ ! -f "$WHITEN_NPZ" ]; then
    echo "[run-mscoco-v136a] whitening .npz missing — computing it now ..."
    /home/yschoi/.conda/envs/dna_hashing/bin/python \
        scripts/build_text_whiten_matrix.py \
        --cache_dir "$CACHE" \
        --out       "$WHITEN_NPZ"
fi

echo "[run-mscoco-v136a] GPU=$GPU cache=$CACHE qwen=$QWEN whiten=$WHITEN_NPZ gamma=$WHITEN_GAMMA K=128 (bij OFF + posSpec head) tag=$TAG"

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
    --codon_position_specific_head \
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
