#!/usr/bin/env bash
# mscoco_v135a = mscoco_v133a with ONE change:
#   --codebook_size 128 -> 256
#
# Parallel Flickr counterpart v135a (Flickr-CLIP) tests the SAME single
# delta (K=128 -> 256) at L=3. This is the MSCOCO cross-dataset port.
#
# mscoco_v133a recipe (commit 1eecc32 + cross-dataset done):
#   K=128, L=3, bij OFF, NO local-residual, per_codebook text NtXent
#   M=6 on [B, 12], cibhash per_codebook + dynamic tau, partial whitening
#   gamma=0.25, paired-aug DNA NtXent OFF (cibhash replaces it).
#   Result on MSCOCO (DB=107K): mAP 0.5652, P@1 0.7976, P@10 0.7889,
#   P@1000 0.7393, DNA-base uniq 0.0934, codeword-tuple uniq 0.2666,
#   NMI off-diag 0.6865 (NEW MSCOCO compositional max), B2 visual lift
#   0.1407 (NEW MSCOCO B2 max).
#
# mscoco_v135a hypothesis. L stays at 3 so |C|=4^3=64 codon possibilities,
# K jumps to 256 -> 4x codeword-per-codon redundancy. On MSCOCO's 107K DB
# this is a STRONGER test of the codeword redundancy mechanism than the
# Flickr 23K port: more samples per codon-tuple slot expose whether the
# 4x redundancy actually translates to distinct cross-sample DNA codes.
#
# References:
#   mscoco_v133a (K=128, L=3, bij OFF, noLocalRes, perCb + cibhash):
#     mAP 0.5652, P@1 0.7976, DNA 0.093, NMI 0.687, B2 0.141
#   mscoco_v106b (K=128, L=3, bij ON, perCodon):
#     mAP 0.5581, P@1 0.7914, DNA 0.125, NMI 0.671, B2 0.137
#   CIBHash-CLIP MSCOCO (external baseline):
#     mAP 0.5842, P@1 0.9264, DNA 0.742, NMI 0.235, B2 0.083
#   mscoco_v135a (K=256, L=3, bij OFF, noLocalRes, perCb + cibhash):  TBD
#
# Predictions (analogous to Flickr v135a):
#   - DNA-base uniq:    0.093 -> 0.20-0.40 (4x redundancy at 107K scale)
#   - codeword-tuple:   0.267 -> 0.50+
#   - mAP:              0.5652 ±0.01 (granularity vs dead trade-off)
#   - codebook dead:    MAJOR concern -- mscoco_v133a already had dead
#                       issues at K=128 (107K DB / 128 codewords per cb
#                       is barely viable). At K=256 the per-codeword
#                       cluster size could collapse below the EMA support
#                       threshold for some codebooks.
#   - NMI:              ~0.69 (per_codebook keeps the compositional
#                       supervision regardless of K)
#
# Usage: bash scripts/train_mscoco_v135a_qwen3.sh <GPU_ID>
set -eu

GPU="${1:-2}"
CACHE="${CACHE:-./cache/mscoco_clip_v4plus}"
QWEN="${QWEN:-./cache/mscoco_qwen3_v4_trainset.jsonl}"
WHITEN_NPZ="${WHITEN_NPZ:-${CACHE}/text_whiten.npz}"
WHITEN_GAMMA="${WHITEN_GAMMA:-0.25}"
TAG="${TAG:-mscoco_v135a_qwen3_v133a_K256_partialWhiten_gamma${WHITEN_GAMMA}}"
LOG="logs/${TAG}.log"
mkdir -p logs

if [ ! -f "$WHITEN_NPZ" ]; then
    echo "[run-mscoco-v135a] whitening .npz missing — computing it now ..."
    /home/yschoi/.conda/envs/dna_hashing/bin/python \
        scripts/build_text_whiten_matrix.py \
        --cache_dir "$CACHE" \
        --out       "$WHITEN_NPZ"
fi

echo "[run-mscoco-v135a] GPU=$GPU cache=$CACHE qwen=$QWEN whiten=$WHITEN_NPZ gamma=$WHITEN_GAMMA K=256 (L=3, 4x redundancy) mscoco_v133a + K=256 tag=$TAG"

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
    --eval_every 5 \
    --post_eval_compositional \
    --dna_distance_mode base \
    -ev -s 2>&1 | tee "$LOG"
