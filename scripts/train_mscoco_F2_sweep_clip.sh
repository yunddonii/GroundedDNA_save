#!/usr/bin/env bash
# MSCOCO F2 whole-image weight-tuning sweep — parameterized single-delta cells.
# Base (whole-image F2, mAP@R 0.8102): WASS=0.05 XMODAL=0.10 THASH=0.10 TCKL=0.10 CIBNT=1.0 CCS=0.0
# Usage: WASS=0.15 CELL=A bash scripts/train_mscoco_F2_sweep_clip.sh <GPU>
set -eu
GPU="${1:-4}"
CACHE="${CACHE:-./cache/mscoco_clip_v5b_tokens}"
QWEN="${QWEN:-./cache/mscoco_qwen3_v5b_trainset.jsonl}"
WHITEN_NPZ="${WHITEN_NPZ:-${CACHE}/text_whiten.npz}"
WASS="${WASS:-0.05}"; XMODAL="${XMODAL:-0.10}"; THASH="${THASH:-0.10}"; TCKL="${TCKL:-0.10}"
CIBNT="${CIBNT:-1.0}"; CCS="${CCS:-0.0}"; CELL="${CELL:-base}"; NUM_CODONS="${NUM_CODONS:-3}"
TAG="${TAG:-mscoco_F2sweep_${CELL}_w${WASS}_x${XMODAL}_th${THASH}_tk${TCKL}_cb${CIBNT}_ccs${CCS}_L${NUM_CODONS}}"
LOG="logs/${TAG}.log"; mkdir -p logs
echo "[mscoco-sweep-$CELL] GPU=$GPU WASS=$WASS XMODAL=$XMODAL THASH=$THASH TCKL=$TCKL CIBNT=$CIBNT CCS=$CCS L=$NUM_CODONS"
CUDA_VISIBLE_DEVICES="$GPU" /home/yschoi/.conda/envs/dna_hashing/bin/python train_siglip2.py \
    --tag "$TAG" --dataset MSCOCO --setting 1 --dataset_dir /home/yschoi/GroundedDNA/dataset \
    --num_devices 0 -bs 64 -e 60 --proj_lr 1e-3 --num_workers 4 \
    --qwen_text_cache_path "$QWEN" --siglip2_feature_cache_dir "$CACHE" \
    --backbone_type clip --codebook_size 128 --c_global_source siglip2_global --per_slot_text_adapter \
    --global_gate_init_logit 4.595 --router_type sinkhorn \
    --sinkhorn_epsilon_init 1.0 --sinkhorn_epsilon_final 0.1 --sinkhorn_lambda_a 1.0 --sinkhorn_lambda_b 1.0 \
    --routing_adaptive_topp --routing_adaptive_topp_min 0.3 --routing_adaptive_topp_max 0.7 \
    --codon_residual_gamma 0.0 --num_codons_per_codebook "$NUM_CODONS" ${FINAL_EPOCH:+--final_epoch_eval} \
    ${VAL_RATIO:+--val_split_ratio "$VAL_RATIO"} ${VAL_SEED:+--val_split_seed "$VAL_SEED"} \
    ${STOP_EP:+--stop_after_epoch "$STOP_EP"} \
    --text_embed_transform partial_whiten --text_whiten_npz "$WHITEN_NPZ" --text_whiten_gamma 0.25 \
    --use_paired_aug_ntxent --lambda_ntxent 0.0 --ntxent_temperature 0.3 --ntxent_mode per_codebook \
    --ntxent_dynamic_tau --ntxent_dynamic_tau_alpha 0.3 \
    --lambda_hash 0.0 --lambda_hash_hard 0.0 --lambda_hash_type mse --hash_target_mode siglip_cos \
    --lambda_text_hash 0.0 --lambda_text_hash_ntxent "$THASH" --text_hash_ntxent_temperature 0.07 --text_hash_ntxent_mode per_codebook \
    --lambda_wasserstein "$WASS" --lambda_vq 0.25 --lambda_quant 0.05 --lambda_xmodal_commit "$XMODAL" \
    --lambda_anchor 0.05 --lambda_dna 0.05 --lambda_bu 0.02 --eta_base_balance 0.3 \
    --lambda_codeword_codon_sinkhorn "$CCS" --lambda_cibhash_ntxent "$CIBNT" --lambda_cibhash_kl 0.001 \
    --cibhash_mode per_codebook --cibhash_temperature 0.3 --cibhash_dynamic_tau --cibhash_dynamic_tau_alpha 0.3 \
    --cibhash_ntxent_continuous --cibhash_ntxent_source visual_token \
    --eval_every 5 --post_eval_compositional --dna_distance_mode base \
    --lambda_text_code_kl "$TCKL" --text_code_kl_tau_v 0.1 --text_code_kl_tau_t 0.07 --text_code_kl_conf_threshold 0.2 \
    --text_code_kl_skip_global --text_hash_ntxent_skip_global \
    --bidirectional_token_prune --bidirectional_token_prune_visual_ratio 1.0 --bidirectional_token_prune_text_ratio 1.0 \
    --eval_cache_dir ./cache/mscoco_clip_v5b -ev -s 2>&1 | tee "$LOG"
