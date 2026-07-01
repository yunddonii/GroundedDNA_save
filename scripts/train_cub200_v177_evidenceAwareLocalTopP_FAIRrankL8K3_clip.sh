#!/usr/bin/env bash
# v177: CUB-200 evidence-aware local token pressure.
# Base recipe is v176a (soft text-evidence Sinkhorn bias) on top of v170a
# FAIRrankL8K3 stackedText + wasserstein 0.15. This keeps all tokens available
# but applies an extra soft cost penalty outside each local slot's top evidence
# token set.
#
# Usage:
#   VARIANT=v177a KEEP_RATIO=0.30 PENALTY=0.20 bash scripts/train_cub200_v177_evidenceAwareLocalTopP_FAIRrankL8K3_clip.sh <GPU_ID>
set -eu

GPU="${1:-0}"
VARIANT="${VARIANT:-v177a}"
KEEP_RATIO="${KEEP_RATIO:-0.30}"
PENALTY="${PENALTY:-0.20}"
CACHE="${CACHE:-./cache/cub200_clip_v6bplus_FAIRrankL8K3}"
QWEN="${QWEN:-./cache/cub200_qwen_v6b_trainset.jsonl}"
WHITEN_NPZ="${WHITEN_NPZ:-${CACHE}/text_whiten.npz}"
WHITEN_GAMMA="${WHITEN_GAMMA:-0.25}"
KEEP_TAG="${KEEP_RATIO/./}"
PEN_TAG="${PENALTY/./}"
TAG="${TAG:-cub200_${VARIANT}_v176a_evidenceAwareLocalTopP_keep${KEEP_TAG}_pen${PEN_TAG}_wass015_FAIRrankL8K3_stackedText_partialWhiten_gamma${WHITEN_GAMMA}}"
LOG="logs/${TAG}.log"
mkdir -p logs

if [ ! -f "$WHITEN_NPZ" ]; then
    echo "[run-cub-v177] whitening .npz missing - computing it now ..."
    /home/yschoi/.conda/envs/dna_hashing/bin/python \
        scripts/build_text_whiten_matrix.py \
        --cache_dir "$CACHE" \
        --out       "$WHITEN_NPZ"
fi

echo "[run-cub-v177] GPU=$GPU variant=$VARIANT keep=$KEEP_RATIO penalty=$PENALTY cache=$CACHE tag=$TAG"

CUDA_VISIBLE_DEVICES="$GPU" \
/home/yschoi/.conda/envs/dna_hashing/bin/python train_siglip2.py \
    --tag "$TAG" \
    --dataset CUB_200 --setting 1 \
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
    --global_gate_init_logit -3.0 \
    --router_type sinkhorn \
    --sinkhorn_epsilon_init 1.0 --sinkhorn_epsilon_final 0.1 \
    --sinkhorn_lambda_a 1.0 \
    --sinkhorn_lambda_b 1.0 \
    --routing_adaptive_topp \
    --routing_adaptive_topp_min 0.6 \
    --routing_adaptive_topp_max 1.0 \
    --routing_text_evidence_beta 0.10 \
    --routing_text_evidence_warmup_epochs 20 \
    --routing_text_evidence_keep_ratio "$KEEP_RATIO" \
    --routing_text_evidence_penalty "$PENALTY" \
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
    --lambda_text_hash_ntxent 0.10 \
    --text_hash_ntxent_temperature 0.07 \
    --text_hash_ntxent_mode per_codebook \
    --lambda_wasserstein 0.15 \
    --lambda_vq 0.25 --lambda_quant 0.05 \
    --lambda_xmodal_commit 0.10 \
    --lambda_anchor 0.05 --lambda_dna 0.05 --lambda_bu 0.02 \
    --eta_base_balance 1.0 \
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
    --lambda_text_code_kl 0.10 \
    --text_code_kl_tau_v 0.1 \
    --text_code_kl_tau_t 0.07 \
    --text_code_kl_conf_threshold 0.2 \
    --text_code_kl_skip_global \
    --eval_cache_dir ./cache/cub200_clip_v6bplus \
    -ev -s 2>&1 | tee "$LOG"
