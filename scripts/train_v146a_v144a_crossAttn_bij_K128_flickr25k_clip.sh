#!/usr/bin/env bash
# v146a = v144a with THREE changes:
#   (1) --router_type sinkhorn -> cross_attn
#       Text-as-query multi-head cross-attention (TextCrossAttentionRouter).
#       Sinkhorn baseline blended via alpha-annealing for stability.
#   (2) --routing_adaptive_topp REMOVED (cross-attn already sharpens; avoid
#       over-concentration like the v145b NMI/P@1 collapse).
#   (3) --lambda_codeword_codon_sinkhorn 0.0 -> 0.1 (bij ON, v106b proven)
#       Forces codeword<->codon bijection, prevents base_balance collapse
#       under cross-attention (lesson from v141 failure).
#
# All other v144a flags retained (lambda_text_code_kl 0.02, cibhash per_codebook
# + dynamic tau, partial whitening gamma=0.25, no local-residual).
#
# Stability safeguards in TextCrossAttentionRouter:
# - Near-identity init on W_Q/W_K/W_V (avoid encoder shock)
# - Zero-init on W_O (initial cross-attn contribution = 0)
# - Residual blend with Sinkhorn baseline: alpha = 0 at ep 0, 1 after warmup
# - Multi-head attention (4 heads) + pre-norm
# - Temperature anneal: 0.2 -> 0.07 over warmup
# - Train: Q = text_part_tokens; Inference: Q = codebook_anchors
#
# References (Flickr25k-CLIP, K=128, partial whitening gamma=0.25):
#   v133a (Sinkhorn, balanced, no UOT, no bij):
#     mAP 0.7541, P@1 0.9150, NMI 0.626, L-L 0.702, DNA 0.340, dead 0.040
#   v144a (Sinkhorn + text_code_kl + adaptive_topp):
#     mAP 0.7499, P@1 0.9170, NMI 0.591, L-L 0.657, DNA 0.376, dead 0.065
#   v145a (UOT lambda=1.0 + adaptive_topp):
#     mAP 0.7512, P@1 0.9220, NMI 0.630, dead 0.012 (P@1/dead champ)
#   v145b (UOT lambda=0.5 + adaptive_topp):
#     mAP 0.7464, NMI 0.568, L-L 0.618, DNA 0.204 (NMI champ but DNA collapse)
#   v146a (cross-attn + bij, NO UOT, NO adaptive_topp): TBD
#
# Hypothesis. Cross-attention with text as query directly attends to the
# patches that match each text part's meaning, providing explicit
# grounding. Bijection loss prevents base_balance collapse (v141 failure
# mode). Without adaptive_topp, no double-sharpening (v145b failure mode).
# Predicted: mAP ~0.75, NMI ~0.55-0.58 (similar to v145b but without DNA
# collapse thanks to bij), dead ~0.04, with interpretable heatmap.
#
# Usage: bash scripts/train_v146a_v144a_crossAttn_bij_K128_flickr25k_clip.sh <GPU_ID>
set -eu

GPU="${1:-0}"
CACHE="${CACHE:-./cache/flickr25k_clip_v4plus}"
QWEN="${QWEN:-./cache/flickr25k_qwen_v4.jsonl}"
WHITEN_NPZ="${WHITEN_NPZ:-${CACHE}/text_whiten.npz}"
WHITEN_GAMMA="${WHITEN_GAMMA:-0.25}"
TAG="${TAG:-v146a_v144a_crossAttn_bij_K128_partialWhiten_gamma${WHITEN_GAMMA}}"
LOG="logs/${TAG}.log"
mkdir -p logs

if [ ! -f "$WHITEN_NPZ" ]; then
    echo "[run-v146a] whitening .npz missing — computing it now ..."
    /home/yschoi/.conda/envs/dna_hashing/bin/python \
        scripts/build_text_whiten_matrix.py \
        --cache_dir "$CACHE" \
        --out       "$WHITEN_NPZ"
fi

echo "[run-v146a] GPU=$GPU cache=$CACHE whiten=$WHITEN_NPZ K=128 (cross_attn + bij + warmup 20ep) tag=$TAG"

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
    --router_type cross_attn \
    --cross_attn_heads 4 \
    --cross_attn_dropout 0.1 \
    --cross_attn_temp_init 0.2 \
    --cross_attn_temp_final 0.07 \
    --cross_attn_warmup_epochs 20 \
    --cross_attn_near_identity_scale 0.1 \
    --cross_attn_alpha_final 1.0 \
    --sinkhorn_epsilon_init 1.0 --sinkhorn_epsilon_final 0.1 \
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
    --lambda_codeword_codon_sinkhorn 0.1 \
    --codeword_codon_sinkhorn_eps 0.1 \
    --codeword_codon_sinkhorn_iters 30 \
    --lambda_cibhash_ntxent 1.0 \
    --lambda_cibhash_kl 0.001 \
    --cibhash_mode per_codebook \
    --cibhash_temperature 0.3 \
    --cibhash_dynamic_tau \
    --cibhash_dynamic_tau_alpha 0.3 \
    --lambda_text_code_kl 0.02 \
    --text_code_kl_tau_v 0.1 \
    --text_code_kl_tau_t 0.07 \
    --text_code_kl_conf_threshold 0.2 \
    --text_code_kl_skip_global \
    --eval_every 5 \
    --post_eval_compositional \
    --dna_distance_mode base \
    -ev -s 2>&1 | tee "$LOG"
