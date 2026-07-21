#!/usr/bin/env bash
# v185 NUS-WIDE weight-tuning sweep — parameterized single/multi-delta cells.
#
# Base (Flickr weights, mAP 0.6012): WASS=0.15 XMODAL=0.05 THASH=0.05
#   TCKL=0.05 CIBNT=1.0 CCS=0.0
# Override any knob via env var; TAG suffix names the cell.
#
# Usage:
#   WASS=0.05 CELL=A bash scripts/train_nuswide_v185_sweep_clip.sh <GPU>
set -eu

GPU="${1:-0}"
CACHE="${CACHE:-./cache/nuswide_clip_tokens}"
QWEN="${QWEN:-./cache/nuswide_qwen3_v4_trainset.jsonl}"
WHITEN_NPZ="${WHITEN_NPZ:-${CACHE}/text_whiten.npz}"
WHITEN_GAMMA="${WHITEN_GAMMA:-0.25}"
BI_V="${BI_V:-0.5}"; BI_T="${BI_T:-0.5}"

# Tunable weights (default = Flickr base)
WASS="${WASS:-0.15}"
XMODAL="${XMODAL:-0.05}"
THASH="${THASH:-0.05}"
TCKL="${TCKL:-0.05}"
CIBNT="${CIBNT:-1.0}"
CCS="${CCS:-0.0}"
GATE="${GATE:-4.595}"     # global_gate_init_logit
CELL="${CELL:-base}"

TAG="${TAG:-nuswide_v185_sweep_${CELL}_w${WASS}_x${XMODAL}_th${THASH}_tk${TCKL}_cb${CIBNT}_ccs${CCS}_g${GATE}}"
LOG="logs/${TAG}.log"
mkdir -p logs

echo "[sweep-$CELL] GPU=$GPU WASS=$WASS XMODAL=$XMODAL THASH=$THASH TCKL=$TCKL CIBNT=$CIBNT CCS=$CCS GATE=$GATE"

CUDA_VISIBLE_DEVICES="$GPU" \
/home/yschoi/.conda/envs/dna_hashing/bin/python train_siglip2.py \
    --tag "$TAG" \
    --dataset NUSWIDE --setting 1 \
    --dataset_dir /home/yschoi/GroundedDNA/dataset \
    --num_devices 0 \
    -bs 64 -e 60 \
    --proj_lr 1e-3 \
    --num_workers 4 \
    --qwen_text_cache_path "$QWEN" \
    --siglip2_feature_cache_dir "$CACHE" \
    --backbone_type clip \
    --codebook_size "${K:-128}" \
    --c_global_source siglip2_global \
    --per_slot_text_adapter \
    --global_gate_init_logit "$GATE" \
    --router_type sinkhorn \
    --sinkhorn_epsilon_init 1.0 --sinkhorn_epsilon_final 0.1 \
    --sinkhorn_lambda_a 1.0 \
    --sinkhorn_lambda_b 1.0 \
    --routing_adaptive_topp \
    --routing_adaptive_topp_min 0.3 \
    --routing_adaptive_topp_max 0.7 \
    --codon_residual_gamma 0.0 \
    ${FINAL_EPOCH:+--final_epoch_eval} \
    ${VAL_RATIO:+--val_split_ratio "$VAL_RATIO"} \
    ${VAL_SEED:+--val_split_seed "$VAL_SEED"} \
    ${STOP_EP:+--stop_after_epoch "$STOP_EP"} \
    ${DISABLE_TEXT:+--disable_text_supervision} \
    --num_codons_per_codebook "${NUM_CODONS:-3}" \
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
    --lambda_text_hash_ntxent "$THASH" \
    --text_hash_ntxent_temperature 0.07 \
    --text_hash_ntxent_mode per_codebook \
    --lambda_wasserstein "$WASS" \
    --lambda_vq 0.25 --lambda_quant 0.05 \
    --lambda_xmodal_commit "$XMODAL" \
    --lambda_anchor 0.05 --lambda_dna 0.05 --lambda_bu 0.02 \
    --eta_base_balance 0.3 \
    --lambda_codeword_codon_sinkhorn "$CCS" \
    --lambda_cibhash_ntxent "$CIBNT" \
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
    --lambda_text_code_kl "$TCKL" \
    --text_code_kl_tau_v 0.1 \
    --text_code_kl_tau_t 0.07 \
    --text_code_kl_conf_threshold 0.2 \
    --text_code_kl_skip_global \
    --text_hash_ntxent_skip_global \
    --bidirectional_token_prune \
    --bidirectional_token_prune_visual_ratio "$BI_V" \
    --bidirectional_token_prune_text_ratio "$BI_T" \
    --eval_cache_dir "${EVAL_CACHE:-$CACHE}" \
    ${EXTRA_ARGS:-} -ev -s 2>&1 | tee "$LOG"
