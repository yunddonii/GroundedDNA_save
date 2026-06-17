#!/usr/bin/env bash
# mscoco_v162a = mscoco_v160a + grounded_text_routing (single delta).
#
# Cross-dataset port of Flickr v162a_fix (Flickr NMI/L<->L/cb_tuple
# champion: NMI 0.566, L<->L 0.636, cb_tuple 0.779). Tests whether the
# "grounded routing sharpens semantic clustering on Eq.(8)-only base"
# effect replicates on MSCOCO 80-class fine-grained retrieval.
#
# MSCOCO references (qwen3 v4_trainset caption, ~8 % coverage):
#   mscoco_v144a (baseline):              mAP 0.5693, P@1 0.812, DNA 0.126, NMI 0.660, B2 0.138
#   mscoco_v160a (v144a + Eq.8):          mAP 0.5743, P@1 0.826, DNA 0.090, NMI 0.726, B2 0.147
#   mscoco_v162a (THIS RUN):              TBD
#
# Single delta vs mscoco_v160a:
#   cache: mscoco_clip_v4plus -> mscoco_clip_v4plus_tokens
#   +flags: --grounded_text_routing --grounded_text_k_t 5 --grounded_text_eps 0.05
#           --grounded_text_stage1_sg --grounded_text_skip_global
#
# Predicted (Flickr v162a_fix pattern):
#   mAP:    0.5743 -> 0.56-0.58 (small risk)
#   DNA:    0.090  -> 0.05-0.10 (DNA decreases)
#   cbT:    0.218  -> 0.30-0.45 (codeword diversity ↑)
#   NMI:    0.726  -> 0.65-0.70 (semantic clustering improves)
#
# Usage: bash scripts/train_mscoco_v162a_v160a_groundedTextRouting_qwen3.sh <GPU_ID>
set -eu

GPU="${1:-1}"
CACHE="${CACHE:-./cache/mscoco_clip_v4plus_tokens}"
QWEN="${QWEN:-./cache/mscoco_qwen3_v4_trainset.jsonl}"
WHITEN_NPZ="${WHITEN_NPZ:-${CACHE}/text_whiten.npz}"
WHITEN_GAMMA="${WHITEN_GAMMA:-0.25}"
TAG="${TAG:-mscoco_v162a_v160a_groundedTextRouting_qwen3_K128_partialWhiten_gamma${WHITEN_GAMMA}}"
LOG="logs/${TAG}.log"
mkdir -p logs

if [ ! -f "${CACHE}/text_tokens.f16.npy" ]; then
    echo "[mscoco-v162a] text_tokens.f16.npy missing at ${CACHE}"
    echo "[mscoco-v162a] run: python extract_clip_text_token_features.py \\"
    echo "         --qwen_cache cache/mscoco_qwen3_v4_trainset.jsonl \\"
    echo "         --donor_dir  cache/mscoco_clip_v4plus \\"
    echo "         --out_dir    ${CACHE}"
    exit 1
fi
if [ ! -f "$WHITEN_NPZ" ]; then
    echo "[mscoco-v162a] whitening .npz missing — building ..."
    /home/yschoi/.conda/envs/dna_hashing/bin/python \
        scripts/build_text_whiten_matrix.py \
        --cache_dir "$CACHE" \
        --out       "$WHITEN_NPZ"
fi

echo "[mscoco-v162a] GPU=$GPU cache=$CACHE k_t=5 eps=0.05 sg=on skip_global=on tag=$TAG"

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
    --routing_adaptive_topp \
    --routing_adaptive_topp_min 0.5 \
    --routing_adaptive_topp_max 0.9 \
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
