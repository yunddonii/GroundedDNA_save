#!/usr/bin/env bash
# v122a = v106b recipe + codon length L = 4 + K = 256.
#
# Goal. v106b's Sinkhorn codeword-codon bijection (the DNA-champion
# mechanism, paper-final candidate) requires K == 4^L. With L=3 the
# hard ceiling is K=64; on Flickr25k DB (23k images) v106b's K=64 codes
# achieve DNA-uniq 0.347 / P@1 0.917 but cannot scale. K=128 attempts
# (v118a, v119a, v120c) all suffer pigeonhole-forced collisions
# (>=2x at the codon level) that either suppress DNA-uniq, fragment
# P@1, or require expensive secondary regularizers. v122a removes the
# ceiling by extending the codon to L=4 positions per codebook:
#   - codon space: 4^L = 256 codons per codebook
#   - DNA hash: M * L * 2 = 6 * 4 * 2 = 48 bits (was 36)
#   - bij can be reactivated cleanly at K = 256 (perfect bijection,
#     no pigeonhole)
#
# Recipe = v106b verbatim, with three changes:
#   --num_codons_per_codebook 4      (NEW; L=4)
#   --codebook_size 256              (K matches |C| = 4^L = 256)
#   (all other v106b flags unchanged; bij ON at lambda 0.1)
#
# Reference (v106b, K=64, L=3): mAP 0.7407, P@1 0.917, DNA 0.347,
#                                dead 0.003, NMI 0.604, B1 0.115, B2 0.072
# Reference (v118a, K=128, L=3, bij OFF): mAP 0.7721, P@1 0.900,
#                                        DNA 0.207 (no bij)
# v122a hypothesis (additive interpretation): v106b's compositional/DNA
# envelope at the larger code space (more codewords -> richer
# representation), with mAP potentially rising toward v118a's 0.77.
#
# Note on extraction. With L=4 the saved base_indices is [N, 24]
# instead of [N, 18]. evaluation_siglip2.py already takes
# num_dna_positions as a parameter (default 18), so this run also
# passes --dna_distance_mode base which uses the actual stored shape;
# no eval-side change needed.
#
# Usage: bash scripts/train_v122a_v106b_L4_K256_flickr25k_clip.sh <GPU_ID>
set -eu

GPU="${1:-0}"
CACHE="${CACHE:-./cache/flickr25k_clip_v4plus}"
QWEN="${QWEN:-./cache/flickr25k_qwen_v4.jsonl}"
TAG="${TAG:-v122a_v106b_L4_K256}"
LOG="logs/${TAG}.log"
mkdir -p logs

echo "[run-v122a] GPU=$GPU cache=$CACHE qwen=$QWEN L=4 K=256 bij=ON tag=$TAG"

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
    --codebook_size 256 \
    --num_codons_per_codebook 4 \
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
