#!/usr/bin/env bash
# v120e = v119a (CIBHash per-codebook on K=128) + text-cos dynamic tau
#         on the CIBHash NtXent itself. Mirrors the v42 dynamic-tau
#         mechanism that has been alive on the existing
#         _loss_ntxent_dna_per_codebook for the entire v9x family.
#
# Per-pair temperature on each codebook m:
#   tau_ij^m = T * (1 + alpha * cos(t_i^m, t_j^m))
# where t_i^m is the cached CLIP per-slot text embedding for sample i and
# codebook m. Semantically similar samples get a softer push (larger tau,
# more tolerance); semantically distant ones get a harder push (smaller
# tau, sharper discrimination). Targets the uniformity-tolerance
# dilemma (Wang et al. CVPR 2021).
#
# Recipe vs v119a (CIBHash per-cb K=128):
#   --cibhash_dynamic_tau          (NEW, enables text-cos dyn tau)
#   --cibhash_dynamic_tau_alpha 0.3 (mirrors --ntxent_dynamic_tau_alpha 0.3)
# Everything else identical to v119a (paired-aug NtXent + dynamic tau on
# DNA-codes stays; CIBHash NtXent/KL on binary hash stays; bij + thNX
# still OFF; partial whitening + K=128).
#
# Reference (v119a, NO dyn-tau on CIBHash):
#   mAP 0.7620, P@1 0.906, DNA 0.246, dead 0.168, NMI 0.596, B1 0.103, B2 0.063
#
# Hypothesis. Dynamic tau on CIBHash provides cross-modal awareness in
# the binary-hash supervision: samples whose CAPTIONS are similar are
# allowed to share more codes (softer push), while semantically distant
# pairs (CAPTION dissimilar) must produce very different hash codes.
# This may:
#   - improve mAP (text-grounded discriminativity)
#   - recover some dead codewords (softer push -> less winner-take-all
#     in CIBHash NtXent -> more codewords stay active)
#   - paper-relevant: directly couples CIBHash supervision to text
#     contribution #2 without needing a separate text_hash_ntxent loss
#
# Usage: bash scripts/train_v120e_v119a_cibhash_textCosDynTau_K128_flickr25k_clip.sh <GPU_ID>
set -eu

GPU="${1:-3}"
CACHE="${CACHE:-./cache/flickr25k_clip_v4plus}"
QWEN="${QWEN:-./cache/flickr25k_qwen_v4.jsonl}"
WHITEN_NPZ="${WHITEN_NPZ:-${CACHE}/text_whiten.npz}"
WHITEN_GAMMA="${WHITEN_GAMMA:-0.25}"
TAG="${TAG:-v120e_v119a_cibhash_textCosDynTau_K128_partialWhiten_gamma${WHITEN_GAMMA}}"
LOG="logs/${TAG}.log"
mkdir -p logs

if [ ! -f "$WHITEN_NPZ" ]; then
    echo "[run-v120e] whitening .npz missing — computing it now ..."
    /home/yschoi/.conda/envs/dna_hashing/bin/python \
        scripts/build_text_whiten_matrix.py \
        --cache_dir "$CACHE" \
        --out       "$WHITEN_NPZ"
fi

echo "[run-v120e] GPU=$GPU cache=$CACHE whiten=$WHITEN_NPZ gamma=$WHITEN_GAMMA K=128 cibhash+textCosDynTau tag=$TAG"

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
    --lambda_text_hash_ntxent 0.0 \
    --text_hash_ntxent_temperature 0.07 \
    --lambda_wasserstein 0.05 \
    --lambda_vq 0.25 --lambda_quant 0.05 \
    --lambda_anchor 0.05 --lambda_dna 0.05 --lambda_bu 0.02 \
    --eta_base_balance 0.3 \
    --lambda_codeword_codon_sinkhorn 0.0 \
    --lambda_cibhash_ntxent 1.0 \
    --lambda_cibhash_kl 0.001 \
    --cibhash_temperature 0.3 \
    --cibhash_mode per_codebook \
    --cibhash_dynamic_tau \
    --cibhash_dynamic_tau_alpha 0.3 \
    --eval_every 5 \
    --post_eval_compositional \
    --dna_distance_mode base \
    -ev -s 2>&1 | tee "$LOG"
