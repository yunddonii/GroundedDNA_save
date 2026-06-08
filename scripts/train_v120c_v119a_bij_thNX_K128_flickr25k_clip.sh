#!/usr/bin/env bash
# v120c = v119a (CIBHash per-codebook on K=128) + restore two key v118a/v106b losses:
#   --lambda_codeword_codon_sinkhorn 0.1  (Sinkhorn codeword<->codon bijection)
#   --lambda_text_hash_ntxent 0.05        (text-DNA <-> visual-DNA InfoNCE)
#
# Rationale (per the 2026-06-08 priority-2 analysis):
# - bij directly counters v119a's 17 % dead-codeword collapse by forcing
#   codon-distribution uniformity across codewords. K=128 vs 4^3=64 codons
#   means pigeonhole forces a 2x codon collision, but Sinkhorn marginals
#   spread that uniformly (2 codewords per codon).
# - thNX restores cross-modal text supervision (paper contribution #2 was
#   silently absent in v119a). Isolated v117a -> v117e ablation showed
#   thNX 0 -> 0.05 gives mAP +0.017, DNA +0.040, NMI -0.030.
# The two pressures are ORTHOGONAL: CIBHash per-codebook discriminativity
# + bij cross-codebook codon distinctness + thNX cross-modal alignment.
# Expected (additive): mAP 0.768-0.774, DNA 0.31-0.35, dead 0.01-0.03,
#   NMI 0.59-0.61, B1/B2 recovered to v118a level.
#
# Reference (v119a):
#   mAP 0.7620, P@1 0.906, DNA 0.246, dead 0.168, NMI 0.596, B1 0.103, B2 0.063
# Reference (v118a, K=128 mAP champion):
#   mAP 0.7721, P@1 0.900, DNA 0.207, dead 0.004, NMI 0.628, B1 0.107, B2 0.064
# Reference (v106b, paper-final DNA champion):
#   mAP 0.7407, P@1 0.917, DNA 0.347, dead 0.003, NMI 0.604, B1 0.115, B2 0.072
#
# Usage: bash scripts/train_v120c_v119a_bij_thNX_K128_flickr25k_clip.sh <GPU_ID>
set -eu

GPU="${1:-0}"
CACHE="${CACHE:-./cache/flickr25k_clip_v4plus}"
QWEN="${QWEN:-./cache/flickr25k_qwen_v4.jsonl}"
WHITEN_NPZ="${WHITEN_NPZ:-${CACHE}/text_whiten.npz}"
WHITEN_GAMMA="${WHITEN_GAMMA:-0.25}"
TAG="${TAG:-v120c_v119a_bij_thNX_cibhash_K128_partialWhiten_gamma${WHITEN_GAMMA}}"
LOG="logs/${TAG}.log"
mkdir -p logs

if [ ! -f "$WHITEN_NPZ" ]; then
    echo "[run-v120c] whitening .npz missing — computing it now ..."
    /home/yschoi/.conda/envs/dna_hashing/bin/python \
        scripts/build_text_whiten_matrix.py \
        --cache_dir "$CACHE" \
        --out       "$WHITEN_NPZ"
fi

echo "[run-v120c] GPU=$GPU cache=$CACHE whiten=$WHITEN_NPZ gamma=$WHITEN_GAMMA K=128 cibhash+bij+thNX tag=$TAG"

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
    --lambda_text_hash_ntxent 0.05 \
    --text_hash_ntxent_temperature 0.07 \
    --lambda_wasserstein 0.05 \
    --lambda_vq 0.25 --lambda_quant 0.05 \
    --lambda_anchor 0.05 --lambda_dna 0.05 --lambda_bu 0.02 \
    --eta_base_balance 0.3 \
    --lambda_codeword_codon_sinkhorn 0.1 \
    --codeword_codon_sinkhorn_eps 0.1 \
    --codeword_codon_sinkhorn_iters 30 \
    --lambda_cibhash_ntxent 1.0 \
    --lambda_cibhash_kl 0.001 \
    --cibhash_temperature 0.3 \
    --eval_every 5 \
    --post_eval_compositional \
    --dna_distance_mode base \
    -ev -s 2>&1 | tee "$LOG"
