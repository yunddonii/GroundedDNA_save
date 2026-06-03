#!/usr/bin/env bash
# v106b = v106a (Sinkhorn bijection loss) + --codon_residual_gamma 0.0.
#
# v106a result: DNA unique 0.210 (-0.031 vs v103a 0.241), DESPITE Sinkhorn
# enforcing bijection at the codeword level. Hypothesis (H1): residual
# injection (gamma=0.3) at INFERENCE breaks the codeword-level bijection
# because:
#   - Sinkhorn loss operates on decode_codeword(z) = head.fc(z.view(3, chunk))
#     (NO residual injection)
#   - Inference computes head.fc(input_proj(cat(quantized, 0.3*residual)))
#     (residual injection on)
#   - The two paths produce different codon mappings; bijection at the
#     codebook level doesn't translate to per-sample DNA uniqueness.
#
# v106b makes the paths CONSISTENT by setting gamma=0. Then:
#   sample DNA = head.fc(quantized.view(3, chunk)) for any sample assigned
#   to codeword k, which is exactly what Sinkhorn shaped to bijection.
#
# Expected outcomes:
#   - If H1 correct: DNA unique should jump to ~0.55+ (= cb-tuple unique,
#     since codon = f(codeword_idx) only when bijection holds + no residual).
#   - If H4 (bijection doesn't help at all): DNA unique stays ~0.15-0.25.
#
# Compare against v103a_noResid baseline (Sinkhorn loss OFF + gamma=0):
#   DNA unique 0.149, cb-tuple 0.495, ratio 3.3x.
# v106b should bring ratio toward 1.0x if Sinkhorn works.
#
# Usage: bash scripts/train_v106b_flickr25k_clip.sh <GPU_ID>
set -eu

GPU="${1:-4}"
CACHE="${CACHE:-./cache/flickr25k_clip_v4plus}"
QWEN="${QWEN:-./cache/flickr25k_qwen_v4.jsonl}"
TAG="${TAG:-v106b_v106a_noResid}"
LOG="logs/${TAG}.log"
mkdir -p logs

echo "[run-v106b] GPU=$GPU cache=$CACHE qwen=$QWEN tag=$TAG"

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
    --codebook_size 64 \
    --c_global_source siglip2_global \
    --per_slot_text_adapter \
    --global_gate_init_logit 4.595 \
    --router_type sinkhorn \
    --sinkhorn_epsilon_init 1.0 --sinkhorn_epsilon_final 0.1 \
    --routing_adaptive_topp \
    --routing_adaptive_topp_min 0.5 \
    --routing_adaptive_topp_max 0.9 \
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
    --lambda_text_hash 0.05 \
    --lambda_text_hash_ntxent 0.05 \
    --text_hash_ntxent_temperature 0.07 \
    --lambda_wasserstein 0.05 \
    --lambda_vq 0.25 --lambda_quant 0.05 \
    --lambda_anchor 0.05 --lambda_dna 0.05 --lambda_bu 0.02 \
    --eta_base_balance 0.3 \
    --lambda_codeword_codon_sinkhorn 0.1 \
    --codeword_codon_sinkhorn_eps 0.1 \
    --codeword_codon_sinkhorn_iters 30 \
    --eval_every 5 \
    --post_eval_compositional \
    --dna_distance_mode base \
    -ev -s 2>&1 | tee "$LOG"
