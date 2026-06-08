#!/usr/bin/env bash
# v120f = v119a (CIBHash per-codebook K=128) with cibhash_mode=global
#         instead of per_codebook. Computes ONE NtXent + ONE KL on the
#         full 36-bit DNA hash (mirrors the original CIBHash design that
#         supervises a single binary hash per image).
#
# Ablation: isolates the effect of per-codebook decomposition vs the
# original "one hash, one contrastive" formulation.
#
# Recipe vs v119a:
#   --cibhash_mode global    (was per_codebook in v119a)
# Everything else identical.
#
# Reference (v119a per_codebook):
#   mAP 0.7620, P@1 0.906, DNA 0.246, dead 0.168, NMI 0.596, B1 0.103, B2 0.063
# Reference (external CIBHash-CLIP, flat 36-bit, no codebook structure):
#   Flickr25k mAP 0.6844, DNA-uniq 0.967
#
# Hypothesis. Global CIBHash NtXent:
#   - Has 36-bit signal per sample (vs 6-bit per codebook averaged across 6
#     codebooks). The 36-dim cosine sim has more resolution -> stronger
#     contrastive signal per pair. Plausible mAP gain.
#   - Loses per-codebook independence pressure. Cross-codebook redundancy
#     no longer directly penalized -> NMI may rise (worse compositional).
#   - The "winner-take-all" pressure that produced 17 % dead codewords in
#     v119a should be reduced (no per-codebook NtXent; the global
#     gradient is shared across 36 bits and less concentrated on any one
#     codebook's 6 bits). Plausible dead-codeword recovery.
#
# Usage: bash scripts/train_v120f_v119a_cibhash_global_K128_flickr25k_clip.sh <GPU_ID>
set -eu

GPU="${1:-5}"
CACHE="${CACHE:-./cache/flickr25k_clip_v4plus}"
QWEN="${QWEN:-./cache/flickr25k_qwen_v4.jsonl}"
WHITEN_NPZ="${WHITEN_NPZ:-${CACHE}/text_whiten.npz}"
WHITEN_GAMMA="${WHITEN_GAMMA:-0.25}"
TAG="${TAG:-v120f_v119a_cibhash_global_K128_partialWhiten_gamma${WHITEN_GAMMA}}"
LOG="logs/${TAG}.log"
mkdir -p logs

if [ ! -f "$WHITEN_NPZ" ]; then
    echo "[run-v120f] whitening .npz missing — computing it now ..."
    /home/yschoi/.conda/envs/dna_hashing/bin/python \
        scripts/build_text_whiten_matrix.py \
        --cache_dir "$CACHE" \
        --out       "$WHITEN_NPZ"
fi

echo "[run-v120f] GPU=$GPU cache=$CACHE whiten=$WHITEN_NPZ gamma=$WHITEN_GAMMA K=128 cibhash_global tag=$TAG"

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
    --cibhash_mode global \
    --eval_every 5 \
    --post_eval_compositional \
    --dna_distance_mode base \
    -ev -s 2>&1 | tee "$LOG"
