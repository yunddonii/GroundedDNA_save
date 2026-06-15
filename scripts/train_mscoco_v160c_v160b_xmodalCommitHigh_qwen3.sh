#!/usr/bin/env bash
# mscoco_v160c = mscoco_v160b recipe with ONE single-delta change:
#   --lambda_xmodal_commit 0.025 -> 0.05  (= 2x stronger cross-modal commit)
#
# Motivation. Uni-Code Eq.(8) cross-modal commitment loss only fires when
# the text path is available. On MSCOCO, caption coverage is 8.2% (vs Flickr
# 100%), so the effective gradient contribution of the Eq.(8) term is roughly
# 1/12 of what it is on Flickr. Doubling the weight from beta/2=0.025 to 0.05
# compensates partially for the sparse-caption regime (full compensation
# would require ~0.3, but that risks destabilizing the standard commitment
# loss at lambda_quant=0.05).
#
# Hypothesis. mscoco_v160b achieved a major retrieval gain (mAP +0.044,
# P@1 +0.085) over mscoco_v144a by combining v150b's visual_token NtXent
# with Eq.(8) cross-modal commit at lambda 0.025. If the cross-modal effect
# is rate-limited by caption sparsity, doubling lambda should push the
# retrieval + B2 axes further while still leaving room for DNA-uniq.
#
# Predicted vs mscoco_v160b:
#   mAP:    0.6134 -> 0.61-0.63 (small additional gain or plateau)
#   P@1:    0.8970 -> 0.89-0.91 (could approach CIBHash 0.9264)
#   B2:     0.166  -> 0.16-0.18
#   DNA:    0.119  -> 0.10-0.14 (similar; could drop slightly)
#   dead:   0.000  -> 0.000-0.02
#
# Risk: lambda_xmodal_commit too high relative to lambda_quant=0.05 could
# cause cross-modal alignment to dominate self-modality structure ->
# codebook collapse. Watch first mid-eval.
#
# Reference (MSCOCO-CLIP K=128, partial-whiten gamma=0.25):
#   mscoco_v144a (base, no Eq.8):                      mAP 0.5693, P@1 0.812, DNA 0.126
#   mscoco_v147a (UOT 1.0 + topp 0.3/0.7):             mAP 0.5637, P@1 0.810, DNA 0.127
#   mscoco_v148b (K=256, UOT, topp 0.2/0.5):           mAP 0.5353, P@1 0.807, DNA 0.154
#   mscoco_v160a (v144a + Eq.8 lambda 0.025):          mAP 0.5743, P@1 0.826, DNA 0.090
#   mscoco_v160b (v150b + Eq.8 lambda 0.025):          mAP 0.6134, P@1 0.897, DNA 0.119, B2 0.166
#   mscoco_v160c (v150b + Eq.8 lambda 0.05):           TBD (this run)
#
# Usage: bash scripts/train_mscoco_v160c_v160b_xmodalCommitHigh_qwen3.sh <GPU_ID>
set -eu

GPU="${1:-2}"
CACHE="${CACHE:-./cache/mscoco_clip_v4plus}"
QWEN="${QWEN:-./cache/mscoco_qwen3_v4_trainset.jsonl}"
WHITEN_NPZ="${WHITEN_NPZ:-${CACHE}/text_whiten.npz}"
WHITEN_GAMMA="${WHITEN_GAMMA:-0.25}"
TAG="${TAG:-mscoco_v160c_v160b_xmodalCommitHigh_0p05_qwen3_K128_partialWhiten_gamma${WHITEN_GAMMA}}"
LOG="logs/${TAG}.log"
mkdir -p logs

if [ ! -f "$WHITEN_NPZ" ]; then
    echo "[run-mscoco-v160c] whitening .npz missing — computing it now ..."
    /home/yschoi/.conda/envs/dna_hashing/bin/python \
        scripts/build_text_whiten_matrix.py \
        --cache_dir "$CACHE" \
        --out       "$WHITEN_NPZ"
fi

echo "[run-mscoco-v160c] GPU=$GPU cache=$CACHE qwen=$QWEN whiten=$WHITEN_NPZ K=128 lambda_xmodal_commit=0.05 (2x v160b) tag=$TAG"

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
    --lambda_xmodal_commit 0.05 \
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
