#!/usr/bin/env bash
# v160b = v150b recipe with ONE single-delta change:
#   --lambda_xmodal_commit 0.025   (= beta/2 in Uni-Code Eq.8, NeurIPS 2023)
#
# Tests whether Eq.(8) cross-modal commitment loss STACKS positively on
# v150b's visual_token NtXent recipe (which holds the family B1/B2
# interpretability champion: B1 0.159, B2 0.098). v160a established the
# +0.012 mAP gain on v144a base; v160b checks if v150b's compositional
# advantages survive the cross-modal commit pressure.
#
# Reference (Flickr25k-CLIP K=128, partial-whiten gamma=0.25):
#   v144a (baseline):                      mAP 0.7499, NMI 0.591, DNA 0.376, B1 0.125, B2 0.076
#   v150b (Pareto-better B1/B2 champion):  mAP 0.7509, DNA 0.329, B1 0.159, B2 0.098, dead 0.000
#   v160a (v144a + Eq.8):                  mAP 0.7617, DNA 0.318, B1 0.128, B2 0.078, dead 0.033
#   v160b (v150b + Eq.8):                  TBD (this run)
#
# Predicted:
#   mAP:    0.7509 -> 0.76-0.77 (Eq.8 carries +0.012 retrieval gain to v150b base)
#   DNA:    0.329  -> 0.28-0.31 (Eq.8 typically lowers DNA slightly)
#   B1, B2: 0.159, 0.098 -> 0.13-0.16, 0.08-0.10 (visual_token NtXent advantage
#                                                 may be partly absorbed by Eq.8)
#   dead:   0.000  -> 0.000-0.02 (both fixes drive utilization up)
#
# If v160b achieves mAP > 0.76 AND B1 >= 0.14 AND B2 >= 0.08, it becomes
# the NEW Pareto-better candidate combining retrieval + interpretability.
#
# Usage: bash scripts/train_v160b_v150b_xmodalCommit_K128_flickr25k_clip.sh <GPU_ID>
set -eu

GPU="${1:-1}"
CACHE="${CACHE:-./cache/flickr25k_clip_v4plus}"
QWEN="${QWEN:-./cache/flickr25k_qwen_v4.jsonl}"
WHITEN_NPZ="${WHITEN_NPZ:-${CACHE}/text_whiten.npz}"
WHITEN_GAMMA="${WHITEN_GAMMA:-0.25}"
TAG="${TAG:-v160b_v150b_xmodalCommit_0p025_K128_partialWhiten_gamma${WHITEN_GAMMA}}"
LOG="logs/${TAG}.log"
mkdir -p logs

if [ ! -f "$WHITEN_NPZ" ]; then
    echo "[run-v160b] whitening .npz missing — computing it now ..."
    /home/yschoi/.conda/envs/dna_hashing/bin/python \
        scripts/build_text_whiten_matrix.py \
        --cache_dir "$CACHE" \
        --out       "$WHITEN_NPZ"
fi

echo "[run-v160b] GPU=$GPU cache=$CACHE whiten=$WHITEN_NPZ K=128 v150b base + Eq.8 xmodal commit tag=$TAG"

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
