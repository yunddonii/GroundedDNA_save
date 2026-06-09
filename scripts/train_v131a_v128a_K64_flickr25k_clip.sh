#!/usr/bin/env bash
# v131a = v128a (per_codebook text NtXent + bij ON + local_residual + CIBHash
# per_codebook + dyn-tau + partial_whiten) with ONE change:
#   --codebook_size 128 -> 64
#
# Motivation. v128a uses K=128 codewords mapped to |C| = 4^L = 4^3 = 64 codons
# via Sinkhorn-OT bijection. At K=128 the pigeonhole principle forces each
# codon to receive EXACTLY 2 codewords (the bij distributes the 2x collision
# uniformly via the (1/K, 1/|C|) marginal constraints). v131a moves to K=64:
#   K = |C| = 4^L = 64
# This is the v106b "perfect 1:1 permutation" regime — every codeword maps
# to its OWN distinct codon, no collisions. Tests whether the perfect-bij
# regime composes with v128a's per_codebook text-NtXent compositional
# supervision.
#
# Expected (interaction matrix):
#   - DNA-uniq: should rise toward v106b's 0.347 (cap at L=3 capacity)
#   - dead: should stay <0.005 (bij at K=|C| is the strongest dead-codeword
#     prevention)
#   - mAP: between v128a (0.7365) and v106b (0.7407); per_codebook text
#     adds compositional pressure that may slightly cost mAP
#   - P@1: between v128a (0.897) and v106b (0.917); same trade-off
#   - NMI: near v128a's 0.636 (per_codebook is the NMI driver, K is not)
#   - B1/B2: best of v128a / v106b on each (per_codebook supervision +
#     bij-protected codebook utilization)
#
# Recipe vs v128a: ONLY `--codebook_size 128 -> 64`. All other v128a flags
# (which themselves are v126a EXACT + per_codebook) preserved.
#
# Reference (Flickr25k-CLIP, partial whitening gamma=0.25, local-residual):
#   v106b (K=64, L=3, bij ON, per_codon, NO local-residual):
#     mAP 0.7407, P@1 0.917, DNA 0.347, dead 0.003, NMI 0.604
#   v128a (K=128, L=3, bij ON, per_codebook, local-residual):
#     mAP 0.7365, P@1 0.897, DNA 0.377, dead 0.003, NMI 0.636
#   v131a (K=64, L=3, bij ON, per_codebook, local-residual) — this run: TBD
#
# Usage: bash scripts/train_v131a_v128a_K64_flickr25k_clip.sh <GPU_ID>
set -eu

GPU="${1:-0}"
CACHE="${CACHE:-./cache/flickr25k_clip_v4plus}"
QWEN="${QWEN:-./cache/flickr25k_qwen_v4.jsonl}"
WHITEN_NPZ="${WHITEN_NPZ:-${CACHE}/text_whiten.npz}"
WHITEN_GAMMA="${WHITEN_GAMMA:-0.25}"
TAG="${TAG:-v131a_v128a_K64_partialWhiten_gamma${WHITEN_GAMMA}}"
LOG="logs/${TAG}.log"
mkdir -p logs

if [ ! -f "$WHITEN_NPZ" ]; then
    echo "[run-v131a] whitening .npz missing — computing it now ..."
    /home/yschoi/.conda/envs/dna_hashing/bin/python \
        scripts/build_text_whiten_matrix.py \
        --cache_dir "$CACHE" \
        --out       "$WHITEN_NPZ"
fi

echo "[run-v131a] GPU=$GPU cache=$CACHE whiten=$WHITEN_NPZ gamma=$WHITEN_GAMMA K=64 (perfect bij) v128a + K64 tag=$TAG"

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
    --local_residual_quant \
    --local_residual_gamma 1.0 \
    --local_residual_text \
    --local_residual_detach_global \
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
    --eval_every 5 \
    --post_eval_compositional \
    --dna_distance_mode base \
    -ev -s 2>&1 | tee "$LOG"
