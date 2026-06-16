#!/usr/bin/env bash
# v162a = v160a + grounded_text_routing (single delta).
#
# v162 introduces Stage-2 OT-based top-k_t local-text-token pruning:
# the Stage-1 routed semantic_visual_tokens[B, M, D] query the per-codebook
# local text tokens (cached_text_tokens [B, M, T, D_proj]) and the top
# k_t=5 tokens are softmax-weighted and pooled into a refined per-codebook
# text embedding that REPLACES the static pooled embed flowing into the
# loss layer (xmodal_commit / text_code_kl / text_hash_ntxent).
#
# Hypothesis:
#   - Per-codebook Qwen captions contain redundancy (only 1-3 of ~32 tokens
#     are actually meaningful for that semantic part).
#   - v160 family uses the dilute pooled embed for loss supervision.
#   - Token-pruning sharpens the cross-modal alignment signal -> stronger
#     text_code_kl gradient -> better DNA/cb_tuple diversity.
#
# Single delta vs v160a:
#   --grounded_text_routing
#   --grounded_text_k_t 5
#   --grounded_text_eps 0.05
#   --grounded_text_stage1_sg            (default True, explicit)
#   --grounded_text_skip_global          (default True, explicit; preserve C_0)
#   + cache dir -> flickr25k_clip_v4plus_qwen3_tokens (with text_tokens.f16.npy)
#
# Requires: extract_clip_text_token_features.py output at
#   cache/flickr25k_clip_v4plus_qwen3_tokens/
#
# v160a reference (Flickr mAP champion):
#   mAP 0.7617, P@1 0.918, DNA 0.318, NMI 0.635
#
# Predicted (analysis-driven):
#   mAP: 0.760 -> 0.755-0.775 (small risk, possible gain from sharper signal)
#   DNA: 0.318 -> 0.34-0.40 (sharper text signal -> better codeword diversity)
#   NMI: 0.635 -> 0.60-0.65 (token-pruning may slightly hurt clustering)
#
# Usage: bash scripts/train_v162a_v160a_groundedTextRouting_K128_flickr25k_clip.sh <GPU_ID>
set -eu

GPU="${1:-0}"
CACHE="${CACHE:-./cache/flickr25k_clip_v4plus_qwen3_tokens}"
QWEN="${QWEN:-./cache/flickr25k_qwen3_v4_trainset.jsonl}"
TAG="${TAG:-v162a_v160a_groundedTextRouting_K128_flickr25k_clip}"
LOG="logs/${TAG}.log"
mkdir -p logs

if [ ! -f "${CACHE}/text_tokens.f16.npy" ]; then
    echo "[v162a] text_tokens.f16.npy missing at ${CACHE}"
    echo "[v162a] run: python extract_clip_text_token_features.py \\"
    echo "         --qwen_cache ${QWEN} \\"
    echo "         --donor_dir  cache/flickr25k_clip_v4plus_qwen3 \\"
    echo "         --out_dir    ${CACHE}"
    exit 1
fi

echo "[run-v162a] GPU=$GPU cache=$CACHE k_t=5 eps=0.05 sg=on skip_global=on tag=$TAG"

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
    --routing_adaptive_topp_min 0.3 \
    --routing_adaptive_topp_max 0.7 \
    --codon_residual_gamma 0.0 \
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
    --text_hash_ntxent_mode per_codebook \
    --lambda_wasserstein 0.05 \
    --lambda_vq 0.25 --lambda_quant 0.05 \
    --lambda_xmodal_commit 0.025 \
    --lambda_anchor 0.05 --lambda_dna 0.05 --lambda_bu 0.02 \
    --eta_base_balance 0.3 \
    --lambda_codeword_codon_sinkhorn 0.0 \
    --eval_every 5 \
    --post_eval_compositional \
    --dna_distance_mode base \
    --lambda_text_code_kl 0.02 \
    --text_code_kl_tau_v 0.1 \
    --text_code_kl_tau_t 0.07 \
    --text_code_kl_conf_threshold 0.2 \
    --text_code_kl_skip_global \
    --grounded_text_routing \
    --grounded_text_k_t 5 \
    --grounded_text_eps 0.05 \
    --grounded_text_stage1_sg \
    --grounded_text_skip_global \
    -ev -s 2>&1 | tee "$LOG"
