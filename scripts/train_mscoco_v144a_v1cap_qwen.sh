#!/usr/bin/env bash
# mscoco_v133a = v133a recipe (Flickr25k-CLIP) ported to MSCOCO (Qwen3
# captions). Cross-dataset check for the noLocalRes ablation family.
#
# v133a Flickr25k-CLIP recipe (2026-06-09 commit 1eecc32):
#   K=128, codon length L=3, bij OFF (lambda_codeword_codon_sinkhorn=0),
#   NO local-residual quant, per_codebook text NtXent (M=6 InfoNCEs on
#   [B, 12] codon segments), cibhash per_codebook + dynamic tau,
#   partial whitening gamma=0.25, paired-aug DNA NtXent OFF (cibhash
#   replaces it), reconstruction OFF (default; lambda_recon inert when
#   --use_decoder absent).
#
# Flickr25k-CLIP v133a reference (DB=23K):
#   mAP 0.7541, P@1 0.9150, P@10 0.9172, DNA-base uniq 0.340,
#   codeword-tuple uniq 0.618, NMI off-diag 0.626, B0 0.056,
#   B1 0.128 (family-max), B2 0.079 (family-max),
#   codewords used [128,128,128,124,123,106] -> dead avg 0.040
#   (cb5 = 106/128 used, the only sub-90% codebook utilization).
#
# Cross-dataset hypothesis (vs prior MSCOCO references).
#   - mscoco_v106b   (K=128, bij ON,  no local-res, per_codon)
#   - mscoco_v118a   (K=128, bij OFF, no local-res, GLOBAL text NtXent
#                     + paired-aug DNA NtXent + NO cibhash)
#   - mscoco_v133a   (K=128, bij OFF, no local-res, PER_CODEBOOK text
#                     NtXent + cibhash per_codebook + NO paired-aug
#                     DNA NtXent)                                    <-- this run
#
# Two open questions on MSCOCO:
#   (a) Does v133a's bij-OFF noLocalRes recipe close the 2x pigeonhole
#       collision gap (K=128 vs |C|=4^3=64) on MSCOCO too?  We expect
#       the per_codebook text NtXent to help DNA-uniq beyond mscoco_v118a's
#       global-text recipe.
#   (b) Does the cibhash per_codebook + dyn_tau swap-in (replacing
#       paired-aug DNA NtXent) generalize from Flickr (where it gave
#       v133a B1/B2 family-max) to MSCOCO's harder 107K DB?
#
# Notes on MSCOCO infrastructure (carried over from mscoco_v118a):
#   - Partial whitening matrix: built once via build_text_whiten_matrix.py
#     on MSCOCO has_text rows only (~10k images * 6 slots = 60k vectors,
#     top-1 EV ~14.1 % vs Flickr's 14.9 %; anisotropy is a
#     dataset-independent CLIP property).
#   - paired-aug NtXent OFF (lambda_ntxent 0.0); cibhash replaces it.
#     Trainer still loads --use_paired_aug_ntxent for the second view
#     (cibhash NEEDS the v2 view to compute u_v1 vs u_v2 InfoNCE).
#
# Usage: bash scripts/train_mscoco_v133a_qwen3.sh <GPU_ID>
set -eu

GPU="${1:-0}"
CACHE="${CACHE:-./cache/mscoco_clip_v1}"
QWEN="${QWEN:-./cache/mscoco_qwen.jsonl}"
WHITEN_NPZ="${WHITEN_NPZ:-${CACHE}/text_whiten.npz}"
WHITEN_GAMMA="${WHITEN_GAMMA:-0.25}"
TAG="${TAG:-mscoco_v144a_v1cap_textCodeKL_lam002_K128_partialWhiten_gamma${WHITEN_GAMMA}}"
LOG="logs/${TAG}.log"
mkdir -p logs

if [ ! -f "$WHITEN_NPZ" ]; then
    echo "[run-mscoco-v144a] whitening .npz missing — computing it now ..."
    /home/yschoi/.conda/envs/dna_hashing/bin/python \
        scripts/build_text_whiten_matrix.py \
        --cache_dir "$CACHE" \
        --out       "$WHITEN_NPZ"
fi

echo "[run-mscoco-v144a] GPU=$GPU cache=$CACHE qwen=$QWEN whiten=$WHITEN_NPZ gamma=$WHITEN_GAMMA K=128 (bij OFF, noLocalRes, per_codebook + cibhash) tag=$TAG"

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
    --lambda_text_code_kl 0.02 \
    --text_code_kl_tau_v 0.1 \
    --text_code_kl_tau_t 0.07 \
    --text_code_kl_conf_threshold 0.2 \
    --text_code_kl_skip_global \
    -ev -s 2>&1 | tee "$LOG"
