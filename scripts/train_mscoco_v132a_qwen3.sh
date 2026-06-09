#!/usr/bin/env bash
# mscoco_v132a = v132a recipe (Flickr25k-CLIP) ported to MSCOCO (Qwen3
# captions). Cross-dataset check for the Pareto-dominant v128a replacement.
#
# v132a Flickr25k-CLIP recipe (2026-06-09 commit 1eecc32):
#   K=128, codon length L=3, bij ON (lambda_codeword_codon_sinkhorn=0.1
#   + eps=0.1 + iters=30), NO local-residual quant, per_codebook text
#   NtXent (M=6 InfoNCEs on [B, 12] codon segments), cibhash per_codebook
#   + dynamic tau (alpha=0.3), partial whitening gamma=0.25, paired-aug
#   DNA NtXent OFF (cibhash replaces it).
#
# Flickr25k-CLIP v132a reference (DB=23K):
#   mAP 0.7418, P@1 0.9070, P@10 0.9115, DNA-base uniq 0.367,
#   codeword-tuple uniq 0.613, NMI off-diag 0.636 (family-best, tied
#   with v128a), B0 0.054, B1 0.123, B2 0.078, dead 0.005.
#   STRICTLY Pareto-dominates v128a: retrieval +0.005/+0.010/+0.006,
#   compositional axis tied.
#
# Cross-dataset hypothesis vs the v133a port (also running):
#   v132a differs from v133a by ONE flag:
#     v132a:  --lambda_codeword_codon_sinkhorn 0.1 (bij ON)
#     v133a:  --lambda_codeword_codon_sinkhorn 0.0 (bij OFF)
#   On Flickr the bij ON case (v132a) loses mAP -0.012 vs v133a (0.7418
#   vs 0.7541) but eliminates the cb5 dead codeword (0.005 vs 0.040).
#   MSCOCO test: does bij ON still trade mAP for codebook health on
#   the 5x larger DB?
#
# References on MSCOCO (DB=107K, K=128):
#   - mscoco_v106b   (K=128, bij ON,  no local-res, per_codon, no cibhash)
#   - mscoco_v118a   (K=128, bij OFF, no local-res, GLOBAL text NtXent,
#                     paired-aug DNA NtXent, no cibhash)
#   - mscoco_v133a   (K=128, bij OFF, no local-res, PER_CODEBOOK text,
#                     cibhash per_cb + dyn tau)         <-- running on GPU 0
#   - mscoco_v132a   (K=128, bij ON,  no local-res, PER_CODEBOOK text,
#                     cibhash per_cb + dyn tau)         <-- this run
#
# Usage: bash scripts/train_mscoco_v132a_qwen3.sh <GPU_ID>
set -eu

GPU="${1:-3}"
CACHE="${CACHE:-./cache/mscoco_clip_v4plus}"
QWEN="${QWEN:-./cache/mscoco_qwen3_v4_trainset.jsonl}"
WHITEN_NPZ="${WHITEN_NPZ:-${CACHE}/text_whiten.npz}"
WHITEN_GAMMA="${WHITEN_GAMMA:-0.25}"
TAG="${TAG:-mscoco_v132a_qwen3_noLocalRes_perCb_cibhash_bij_K128_partialWhiten_gamma${WHITEN_GAMMA}}"
LOG="logs/${TAG}.log"
mkdir -p logs

if [ ! -f "$WHITEN_NPZ" ]; then
    echo "[run-mscoco-v132a] whitening .npz missing — computing it now ..."
    /home/yschoi/.conda/envs/dna_hashing/bin/python \
        scripts/build_text_whiten_matrix.py \
        --cache_dir "$CACHE" \
        --out       "$WHITEN_NPZ"
fi

echo "[run-mscoco-v132a] GPU=$GPU cache=$CACHE qwen=$QWEN whiten=$WHITEN_NPZ gamma=$WHITEN_GAMMA K=128 (bij ON, noLocalRes, per_codebook + cibhash) tag=$TAG"

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
    --lambda_anchor 0.05 --lambda_dna 0.05 --lambda_bu 0.02 \
    --eta_base_balance 0.3 \
    --lambda_codeword_codon_sinkhorn 0.1 \
    --codeword_codon_sinkhorn_eps 0.1 \
    --codeword_codon_sinkhorn_iters 30 \
    --lambda_cibhash_ntxent 1.0 \
    --lambda_cibhash_kl 0.001 \
    --cibhash_mode per_codebook \
    --cibhash_temperature 0.3 \
    --cibhash_dynamic_tau \
    --cibhash_dynamic_tau_alpha 0.3 \
    --eval_every 5 \
    --post_eval_compositional \
    --dna_distance_mode base \
    -ev -s 2>&1 | tee "$LOG"
