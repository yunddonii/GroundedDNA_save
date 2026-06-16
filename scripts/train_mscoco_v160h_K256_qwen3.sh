#!/usr/bin/env bash
# mscoco_v160h = mscoco_v160b recipe + codebook_size 128 -> 256 (single delta).
#
# Background:
# - mscoco_v160b (K=128) is the MSCOCO MULTI-AXIS CHAMPION (mAP 0.6134,
#   P@1 0.897, DNA 0.119, cbT 0.205, NMI 0.726, B2 0.166, dead 0.000).
# - The earlier mscoco_v106b_K256 mis-specified the base: v106b is an older
#   recipe (NO Eq.(8), NO partial-whiten, NO CIBHash visual_token NtXent, NO
#   text_code_kl) and the +K alone produced mAP -0.059 vs mscoco_v160b.
# - This run replaces that mistake: we re-run the K=256 ablation on the
#   CURRENT champion recipe (v160b) so the K-axis can be evaluated in
#   isolation against the strongest baseline.
#
# Hypothesis:
# - K=256 puts the codebook into 4x pigeonhole vs 4^3=64 codon space.
# - v160b's Uni-Code Eq.(8) + text_code_kl + CIBHash visual_token NtXent
#   provide stronger code-diversity signal than v106b had.
# - Predicted: cbT explodes (0.205 -> ~0.5), DNA rises modestly (0.119 ->
#   ~0.15-0.18), mAP cost SMALLER than the v106b_K256 -0.059 collapse because
#   v160b has 3 extra loss terms (xmodal_commit, text_code_kl, CIBHash)
#   compensating for K-expansion sparsity.
#
# Predicted (analysis-driven):
#   mAP:    0.6134 -> 0.58-0.60 (smaller drop than v106b_K256's -0.059)
#   P@1:    0.897  -> 0.85-0.88
#   DNA:    0.119  -> 0.14-0.18
#   cbT:    0.205  -> 0.4-0.55 (4x diversity ceiling)
#   NMI:    0.726  -> 0.66-0.72
#   dead:   0.000  -> 0.00-0.03 (v160b's 100% utilization is K=128 specific)
#
# Usage: bash scripts/train_mscoco_v160h_K256_qwen3.sh <GPU_ID>
set -eu

GPU="${1:-1}"
CACHE="${CACHE:-./cache/mscoco_clip_v4plus}"
QWEN="${QWEN:-./cache/mscoco_qwen3_v4_trainset.jsonl}"
WHITEN_NPZ="${WHITEN_NPZ:-${CACHE}/text_whiten.npz}"
WHITEN_GAMMA="${WHITEN_GAMMA:-0.25}"
TAG="${TAG:-mscoco_v160h_v160b_K256_qwen3_partialWhiten_gamma${WHITEN_GAMMA}}"
LOG="logs/${TAG}.log"
mkdir -p logs

if [ ! -f "$WHITEN_NPZ" ]; then
    echo "[run-mscoco-v160h] whitening .npz missing — computing it now ..."
    /home/yschoi/.conda/envs/dna_hashing/bin/python \
        scripts/build_text_whiten_matrix.py \
        --cache_dir "$CACHE" \
        --out       "$WHITEN_NPZ"
fi

echo "[run-mscoco-v160h] GPU=$GPU codebook_size=256 (single delta from mscoco_v160b K=128) tag=$TAG"

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
    --codebook_size 256 \
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
    --lambda_anchor 0.05 --lambda_dna 0.05 --lambda_bu 0.02 \
    --eta_base_balance 0.3 \
    --lambda_codeword_codon_sinkhorn 0.0 \
    --lambda_cibhash_ntxent 1.0 \
    --lambda_cibhash_kl 0.001 \
    --cibhash_mode per_codebook \
    --cibhash_temperature 0.3 \
    --cibhash_dynamic_tau \
    --cibhash_dynamic_tau_alpha 0.3 \
    --cibhash_ntxent_continuous \
    --cibhash_ntxent_source visual_token \
    --eval_every 5 \
    --post_eval_compositional \
    --dna_distance_mode base \
    --lambda_text_code_kl 0.02 \
    --text_code_kl_tau_v 0.1 \
    --text_code_kl_tau_t 0.07 \
    --text_code_kl_conf_threshold 0.2 \
    --text_code_kl_skip_global \
    -ev -s 2>&1 | tee "$LOG"
