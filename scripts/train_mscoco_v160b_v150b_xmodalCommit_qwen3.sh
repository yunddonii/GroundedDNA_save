#!/usr/bin/env bash
# mscoco_v160b = v160b recipe (Flickr25k-CLIP) ported to MSCOCO (Qwen3 v4).
#
# v160b = v150b + Uni-Code Eq.(8) cross-modal commitment loss
#   Single-delta from v150b (Flickr): --lambda_xmodal_commit 0.025 (= beta/2).
#
# Flickr v160b reference (NEW 4-axis family champion, B1 0.162, B2 0.100,
# DNA 0.400, collision ratio 1.34x):
#   mAP 0.7390, P@1 0.917, DNA 0.400, cb_tuple 0.536, NMI 0.630, L-L 0.671
#   B1 0.162, B2 0.100, dead 0.000, codewords used per cb [128]x6
#
# MSCOCO references (for trade-off comparison):
#   mscoco_v144a (base, no UOT/topp/Eq.8):  mAP 0.5693, P@1 0.8120, DNA 0.126, NMI 0.660, B2 0.138
#   mscoco_v147a (UOT 1.0 + topp 0.3/0.7):  mAP 0.5637, P@1 0.8096, DNA 0.127, NMI 0.634, B2 0.139
#   mscoco_v148b (K=256, UOT, topp 0.2/0.5): mAP 0.5353, P@1 0.8072, DNA 0.154, NMI 0.580, B2 0.147
#   mscoco_v160b (v150b + Eq.8):            TBD (this run)
#
# Cross-dataset hypothesis (carrying over v160b's Flickr findings):
# - On Flickr (100% caption coverage), v160b's Eq.(8) fires every batch.
# - On MSCOCO (8.2% caption coverage), Eq.(8) fires only on 8% of batches.
# - Therefore the cross-modal commitment effect is RATE-LIMITED on MSCOCO.
# - Predicted mAP cost may be SMALLER than Flickr's -0.011, but B1/B2/DNA
#   gains may also be smaller. Net: less dramatic trade-off, possibly Pareto
#   improvement over mscoco_v147a on the compositional axis.
#
# Predicted (analysis-driven):
#   mAP:    0.5637 -> 0.55-0.57 (small risk vs mscoco_v147a)
#   DNA:    0.127  -> 0.13-0.16 (Eq.8 helps when paired with v150b base)
#   NMI:    0.634  -> 0.61-0.64 (slight improvement)
#   B1, B2: 0.139 (no B1 measured on MSCOCO) -> may rise via cross-modal commitment
#   dead:   0.014  -> 0.00-0.03 (codebook utilization should stay high)
#
# Usage: bash scripts/train_mscoco_v160b_v150b_xmodalCommit_qwen3.sh <GPU_ID>
set -eu

GPU="${1:-0}"
CACHE="${CACHE:-./cache/mscoco_clip_v4plus}"
QWEN="${QWEN:-./cache/mscoco_qwen3_v4_trainset.jsonl}"
WHITEN_NPZ="${WHITEN_NPZ:-${CACHE}/text_whiten.npz}"
WHITEN_GAMMA="${WHITEN_GAMMA:-0.25}"
TAG="${TAG:-mscoco_v160b_v150b_xmodalCommit_0p025_qwen3_K128_partialWhiten_gamma${WHITEN_GAMMA}}"
LOG="logs/${TAG}.log"
mkdir -p logs

if [ ! -f "$WHITEN_NPZ" ]; then
    echo "[run-mscoco-v160b] whitening .npz missing — computing it now ..."
    /home/yschoi/.conda/envs/dna_hashing/bin/python \
        scripts/build_text_whiten_matrix.py \
        --cache_dir "$CACHE" \
        --out       "$WHITEN_NPZ"
fi

echo "[run-mscoco-v160b] GPU=$GPU cache=$CACHE qwen=$QWEN whiten=$WHITEN_NPZ K=128 v150b base + Eq.8 (Uni-Code) tag=$TAG"

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
