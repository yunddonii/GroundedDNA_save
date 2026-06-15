#!/usr/bin/env bash
# mscoco_v160a = v160a recipe (Flickr25k-CLIP) ported to MSCOCO (Qwen3 v4).
#
# v160a = v144a + Uni-Code Eq.(8) cross-modal commitment loss (single delta).
#   --lambda_xmodal_commit 0.025  (= beta/2 in the paper)
#
# Flickr v160a reference (NEW mAP family champion):
#   mAP 0.7617, P@1 0.913, P@10 0.916, DNA 0.318, cb_tuple 0.603, NMI 0.635, L-L 0.719
#   B1 0.128, B2 0.078, dead 0.033, codewords used per cb [128, 126, 122, 125, 126, 116] avg 123.8
#
# MSCOCO references (for trade-off comparison):
#   mscoco_v144a (base, no Eq.8):           mAP 0.5693, P@1 0.8120, DNA 0.126, NMI 0.660, B2 0.138
#   mscoco_v147a (UOT 1.0 + topp 0.3/0.7):  mAP 0.5637, P@1 0.8096, DNA 0.127, NMI 0.634, B2 0.139
#   mscoco_v148b (K=256, UOT, topp 0.2/0.5): mAP 0.5353, P@1 0.8072, DNA 0.154, NMI 0.580, B2 0.147
#   mscoco_v160a (v144a + Eq.8):            TBD (this run)
#   mscoco_v160b (v150b + Eq.8):            TBD (parallel)
#
# Predicted (analysis-driven). On Flickr v160a gained mAP +0.012 over v144a
# while losing DNA -0.058. On MSCOCO the trade-off may be smaller because
# Eq.(8) fires only on 8.2% of batches (captioned subset). Single-delta
# isolates the Eq.(8) effect on the mscoco_v144a base. Expected:
#   mAP:    0.5693 -> 0.57-0.58 (smaller +mAP gain than Flickr)
#   DNA:    0.126  -> 0.10-0.13 (slight drop, smaller than Flickr's -0.058)
#   NMI:    0.660  -> 0.63-0.66 (slight improvement possible)
#
# Usage: bash scripts/train_mscoco_v160a_v144a_xmodalCommit_qwen3.sh <GPU_ID>
set -eu

GPU="${1:-5}"
CACHE="${CACHE:-./cache/mscoco_clip_v4plus}"
QWEN="${QWEN:-./cache/mscoco_qwen3_v4_trainset.jsonl}"
WHITEN_NPZ="${WHITEN_NPZ:-${CACHE}/text_whiten.npz}"
WHITEN_GAMMA="${WHITEN_GAMMA:-0.25}"
TAG="${TAG:-mscoco_v160a_v144a_xmodalCommit_0p025_qwen3_K128_partialWhiten_gamma${WHITEN_GAMMA}}"
LOG="logs/${TAG}.log"
mkdir -p logs

if [ ! -f "$WHITEN_NPZ" ]; then
    echo "[run-mscoco-v160a] whitening .npz missing — computing it now ..."
    /home/yschoi/.conda/envs/dna_hashing/bin/python \
        scripts/build_text_whiten_matrix.py \
        --cache_dir "$CACHE" \
        --out       "$WHITEN_NPZ"
fi

echo "[run-mscoco-v160a] GPU=$GPU cache=$CACHE qwen=$QWEN whiten=$WHITEN_NPZ K=128 v144a base + Eq.8 (Uni-Code) tag=$TAG"

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
    -ev -s 2>&1 | tee "$LOG"
