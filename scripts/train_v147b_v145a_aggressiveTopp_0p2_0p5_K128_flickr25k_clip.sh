#!/usr/bin/env bash
# v147b = v145a (UOT lambda=1.0 + adaptive_topp + text_code_kl 0.02) with
# ONE change: AGGRESSIVE adaptive_topp
#   --routing_adaptive_topp_min 0.5 -> 0.2
#   --routing_adaptive_topp_max 0.9 -> 0.5
#
# This is the more aggressive sibling of v147a (0.3/0.7). Tests whether
# v146-style cross-attn-like sharp routing can be safely induced here
# by Sinkhorn-UOT + very-low-topp without triggering the codebook
# collapse that killed v146a (98% dead) and v146b (94% dead).
#
# Risk. v146 failure mechanism: sharp routing + EMA-VQ -> codebook
# starvation. UOT's mass-redistribution should counter this, BUT very
# low topp (0.2-0.5) is in the same sharpness regime as cross-attn.
# Watch dead and DNA-uniq carefully.
#
# Sweet spot prediction (v147b):
#   NMI 0.54-0.57 (best compositional)
#   dead 0.03-0.08 (UOT might not save it at this sharpness)
#   DNA-uniq risk: 0.20-0.30
#   mAP risk: -0.01 to -0.03 vs v145a
#
# Reference (Flickr25k-CLIP K=128, partial-whiten gamma=0.25):
#   v144a (no UOT, topp 0.5/0.9):  mAP 0.7499, NMI 0.591, DNA 0.376, dead 0.065
#   v145a (UOT 1.0, topp 0.5/0.9): mAP 0.7512, NMI 0.630, DNA 0.262, dead 0.012
#   v145b (UOT 0.5, topp 0.5/0.9): mAP 0.7464, NMI 0.568, DNA 0.204, dead 0.026
#   v146a (cross-attn + bij):       mAP 0.5640, NMI 0.016, DNA 0.001, dead 0.979 (DISCARDED)
#   v146b (cross-attn no-bij+topp): mAP 0.6499, NMI 0.179, DNA 0.004, dead 0.943 (DISCARDED)
#   v147a (UOT 1.0, topp 0.3/0.7): TBD (parallel)
#   v147b (UOT 1.0, topp 0.2/0.5): TBD (this run)
#
# Usage: bash scripts/train_v147b_v145a_aggressiveTopp_0p2_0p5_K128_flickr25k_clip.sh <GPU_ID>
set -eu

GPU="${1:-1}"
CACHE="${CACHE:-./cache/flickr25k_clip_v4plus}"
QWEN="${QWEN:-./cache/flickr25k_qwen_v4.jsonl}"
WHITEN_NPZ="${WHITEN_NPZ:-${CACHE}/text_whiten.npz}"
WHITEN_GAMMA="${WHITEN_GAMMA:-0.25}"
TAG="${TAG:-v147b_v145a_aggressiveTopp_0p2_0p5_K128_partialWhiten_gamma${WHITEN_GAMMA}}"
LOG="logs/${TAG}.log"
mkdir -p logs

if [ ! -f "$WHITEN_NPZ" ]; then
    echo "[run-v147b] whitening .npz missing — computing it now ..."
    /home/yschoi/.conda/envs/dna_hashing/bin/python \
        scripts/build_text_whiten_matrix.py \
        --cache_dir "$CACHE" \
        --out       "$WHITEN_NPZ"
fi

echo "[run-v147b] GPU=$GPU cache=$CACHE whiten=$WHITEN_NPZ K=128 UOT 1.0 + topp 0.2/0.5 (aggressive) tag=$TAG"

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
    --sinkhorn_lambda_a 1.0 \
    --sinkhorn_lambda_b 1.0 \
    --routing_adaptive_topp \
    --routing_adaptive_topp_min 0.2 \
    --routing_adaptive_topp_max 0.5 \
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
