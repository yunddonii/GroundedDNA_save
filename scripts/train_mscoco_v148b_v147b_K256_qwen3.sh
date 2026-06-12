#!/usr/bin/env bash
# mscoco_v148b = mscoco_v147a recipe (UOT lambda=1.0 + topp 0.3/0.7 + text_code_kl 0.02)
# with TWO deltas vs mscoco_v147a:
#   (1) --routing_adaptive_topp_min/max 0.3/0.7 -> 0.2/0.5  (v147b's aggressive topp)
#   (2) --codebook_size 128 -> 256
#
# Combined hypothesis. K=256 expands codeword capacity (more representational
# slots per codebook); aggressive topp 0.2/0.5 sharpens routing to exploit
# that capacity for lower per-codebook redundancy (NMI). MSCOCO's K=128
# Flickr port (mscoco_v147a) already showed a MORE favorable trade-off than
# Flickr (mAP cost -0.006 vs -0.022, DNA preserved). The K=256 + sharper topp
# combination tests whether MSCOCO's structural sparsity (8.2% text coverage)
# extends that favorability to even larger codebooks.
#
# Risk vectors.
#   (a) Pigeonhole pressure 4x: K=256 vs codon space |C|=4^3=64. Even with
#       UOT + sharp topp, DNA-uniq may collapse below 0.10.
#   (b) Sharp topp (0.2/0.5) + K=256 + sparse text -> dead spike risk
#       (the v146 collapse mechanism in a softer form).
#
# Reference (MSCOCO-CLIP K=128, partial-whiten gamma=0.25):
#   mscoco_v144a (no UOT, topp 0.5/0.9):           mAP 0.5693, P@1 0.8120, NMI 0.660, DNA 0.126, dead 0.085
#   mscoco_v147a (UOT 1.0, topp 0.3/0.7, K=128):   mAP 0.5637, P@1 0.8096, NMI 0.634, DNA 0.127, dead 0.014
#   mscoco_v148b (UOT 1.0, topp 0.2/0.5, K=256):   TBD (this run)
#
# Reference (Flickr25k-CLIP K=128):
#   v147a (UOT 1.0, topp 0.3/0.7): mAP 0.7280, NMI 0.566, DNA 0.307, dead 0.008
#   v147b (UOT 1.0, topp 0.2/0.5): mAP 0.7388, NMI 0.508, DNA 0.230, dead 0.020
#
# Usage: bash scripts/train_mscoco_v148b_v147b_K256_qwen3.sh <GPU_ID>
set -eu

GPU="${1:-0}"
CACHE="${CACHE:-./cache/mscoco_clip_v4plus}"
QWEN="${QWEN:-./cache/mscoco_qwen3_v4_trainset.jsonl}"
WHITEN_NPZ="${WHITEN_NPZ:-${CACHE}/text_whiten.npz}"
WHITEN_GAMMA="${WHITEN_GAMMA:-0.25}"
TAG="${TAG:-mscoco_v148b_v147b_K256_qwen3_uot_topp_0p2_0p5_partialWhiten_gamma${WHITEN_GAMMA}}"
LOG="logs/${TAG}.log"
mkdir -p logs

if [ ! -f "$WHITEN_NPZ" ]; then
    echo "[run-mscoco-v148b] whitening .npz missing — computing it now ..."
    /home/yschoi/.conda/envs/dna_hashing/bin/python \
        scripts/build_text_whiten_matrix.py \
        --cache_dir "$CACHE" \
        --out       "$WHITEN_NPZ"
fi

echo "[run-mscoco-v148b] GPU=$GPU cache=$CACHE qwen=$QWEN whiten=$WHITEN_NPZ K=256 UOT 1.0 + topp 0.2/0.5 tag=$TAG"

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
    --routing_adaptive_topp_min 0.2 \
    --routing_adaptive_topp_max 0.5 \
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
    -ev -s 2>&1 | tee "$LOG"
