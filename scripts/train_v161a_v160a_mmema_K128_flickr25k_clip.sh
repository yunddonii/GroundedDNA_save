#!/usr/bin/env bash
# v161a = v160a recipe + --mm_ema (Uni-Code Section 4.3 MM-EMA, simplified).
#
# Single-delta from v160a: add bi-modal EMA codebook update.
#   Currently: text path runs quantizer in eval mode (EMA-disabled);
#              codebook is updated ONLY from visual signals.
#   v161a:     text path keeps quantizer in train mode;
#              codebook EMA is updated by BOTH visual AND text vectors.
#
# This is the minimum-viable Uni-Code MM-EMA: we omit the cross-attention
# intermediary (r^a, r^b) terms from Eq.(7) -- those are a separate
# v161b/c step. The Eq.(8) cross-modal commitment (carried from v160a)
# remains active.
#
# Hypothesis. Eq.(8) pulls each encoder toward the OPPOSITE modality's
# quantized codeword; MM-EMA goes one level deeper and lets the codebook
# itself learn from text too. Stacked, they should mutually reinforce
# cross-modal alignment AT THE CODEBOOK LEVEL.
#
# Reference (Flickr25k-CLIP K=128, partial-whiten gamma=0.25):
#   v144a (baseline):                    mAP 0.7499, NMI 0.591, DNA 0.376, B1 0.125
#   v150b (interpretability champion):   mAP 0.7509, B1 0.159, B2 0.098, dead 0.000
#   v160a (Eq.8 alone, mAP champion):    mAP 0.7617, DNA 0.318, dead 0.033
#   v161a (Eq.8 + MM-EMA):               TBD (this run)
#
# Predicted:
#   mAP:    0.76-0.77 (Eq.8 retained, possibly slight gain)
#   DNA:    0.30-0.35 (MM-EMA could help recover some DNA-uniq if
#                      text-supervised codebook learning diversifies codewords)
#   NMI:    0.62-0.65 (codebook is now jointly text/visual; mid-range)
#   B1, B2: 0.13-0.16, 0.08-0.10 (text-codebook coupling may raise B1)
#   dead:   0.02-0.05 (bi-modal EMA helps utilization)
#
# Watch first mid-eval (ep 4):
#   mAP >= 0.65 AND unique >= 0.10 -> normal, continue
#   mAP < 0.65 OR unique < 0.10    -> catastrophic, KILL (the MM-EMA
#                                     could destabilize codebook learning).
#
# Usage: bash scripts/train_v161a_v160a_mmema_K128_flickr25k_clip.sh <GPU_ID>
set -eu

GPU="${1:-4}"
CACHE="${CACHE:-./cache/flickr25k_clip_v4plus}"
QWEN="${QWEN:-./cache/flickr25k_qwen_v4.jsonl}"
WHITEN_NPZ="${WHITEN_NPZ:-${CACHE}/text_whiten.npz}"
WHITEN_GAMMA="${WHITEN_GAMMA:-0.25}"
TAG="${TAG:-v161a_v160a_mmema_K128_partialWhiten_gamma${WHITEN_GAMMA}}"
LOG="logs/${TAG}.log"
mkdir -p logs

if [ ! -f "$WHITEN_NPZ" ]; then
    echo "[run-v161a] whitening .npz missing — computing it now ..."
    /home/yschoi/.conda/envs/dna_hashing/bin/python \
        scripts/build_text_whiten_matrix.py \
        --cache_dir "$CACHE" \
        --out       "$WHITEN_NPZ"
fi

echo "[run-v161a] GPU=$GPU cache=$CACHE whiten=$WHITEN_NPZ K=128 v160a + MM-EMA tag=$TAG"

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
    --lambda_xmodal_commit 0.025 \
    --mm_ema \
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
