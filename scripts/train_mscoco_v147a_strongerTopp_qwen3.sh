#!/usr/bin/env bash
# mscoco_v147a = v147a recipe (Flickr25k-CLIP) ported to MSCOCO (Qwen3 v4).
#
# v147a = v145a (UOT lambda=1.0 + topp 0.5/0.9 + text_code_kl 0.02) with
# ONE change: --routing_adaptive_topp_min/max 0.5/0.9 -> 0.3/0.7.
#
# Single delta from mscoco_v144a (which has UOT OFF, topp 0.5/0.9):
#   --sinkhorn_lambda_a/b 1.0 (UOT ON, both marginals)
#   --routing_adaptive_topp_min/max 0.5/0.9 -> 0.3/0.7
#
# Flickr25k-CLIP v147a reference (NEW Pareto-better compositional candidate):
#   mAP 0.7280, P@1 0.9145, NMI 0.566, L-L 0.607, DNA 0.307, dead 0.008
#   Codewords used per cb: [128, 126, 128, 124, 128, 128] (utilization > 96%)
#   Pareto-dominates v145a on 5 axes (NMI / L-L / DNA / dead / B2).
#
# MSCOCO references (DB=107K, 8.2% text coverage):
#   mscoco_v118a (K=128, bij OFF, GLOBAL text NtXent):    mAP 0.5532
#   mscoco_v133a (K=128, bij OFF, per_codebook):          mAP 0.5652, NMI 0.687
#   mscoco_v144a (K=128, + text_code_kl):                 mAP 0.5693, P@1 0.8120, NMI 0.660
#   mscoco_v147a (K=128, + UOT 1.0 + topp 0.3/0.7):       TBD (this run)
#
# Cross-dataset hypothesis. v147a's NMI gain on Flickr was driven by sharper
# adaptive_topp recovering the NMI lost to UOT's mass redistribution. On MSCOCO
# the relative gains may differ:
#   (a) MSCOCO's sparse text coverage already makes routing softer than
#       Flickr -> sharper topp might help more relative to baseline.
#   (b) Or, sharper topp + sparse text might over-concentrate routing,
#       making mAP regression worse than Flickr's -0.023.
# Either way, this is the first MSCOCO test of the UOT + adaptive_topp combo
# under text_code_kl.
#
# Usage: bash scripts/train_mscoco_v147a_strongerTopp_qwen3.sh <GPU_ID>
set -eu

GPU="${1:-4}"
CACHE="${CACHE:-./cache/mscoco_clip_v4plus}"
QWEN="${QWEN:-./cache/mscoco_qwen3_v4_trainset.jsonl}"
WHITEN_NPZ="${WHITEN_NPZ:-${CACHE}/text_whiten.npz}"
WHITEN_GAMMA="${WHITEN_GAMMA:-0.25}"
TAG="${TAG:-mscoco_v147a_qwen3_uot_strongerTopp_0p3_0p7_K128_partialWhiten_gamma${WHITEN_GAMMA}}"
LOG="logs/${TAG}.log"
mkdir -p logs

if [ ! -f "$WHITEN_NPZ" ]; then
    echo "[run-mscoco-v147a] whitening .npz missing — computing it now ..."
    /home/yschoi/.conda/envs/dna_hashing/bin/python \
        scripts/build_text_whiten_matrix.py \
        --cache_dir "$CACHE" \
        --out       "$WHITEN_NPZ"
fi

echo "[run-mscoco-v147a] GPU=$GPU cache=$CACHE qwen=$QWEN whiten=$WHITEN_NPZ K=128 UOT 1.0 + topp 0.3/0.7 tag=$TAG"

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
