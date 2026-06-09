#!/usr/bin/env bash
# v134a = v131a with ONE change: REMOVE all 4 local-residual flags.
#
# v131a = v128a + K=128->64 (perfect 1:1 codeword<->codon bij regime
# where K=|C|=4^L=64). DISCARDED: DNA-base unique collapsed 0.377 ->
# 0.266 due to loss of K=128's 2x codeword redundancy combined with
# local-residual removing C0/global axis from local codebooks.
#
# v134a tests whether REMOVING local-residual rescues the K=64 regime
# — i.e. whether the bij-K=|C| combo can achieve v106b-like DNA-uniq
# (0.347) when local codebooks are allowed to share the global axis
# again (like v106b did).
#
# Combined ablation reads:
#   - if v134a recovers DNA-uniq toward v106b's 0.347 -> confirms
#     local-residual was the K=64 collapse cause (NOT K itself)
#   - if v134a stays at v131a's 0.266 -> K=64 itself is wrong recipe
#     for the per_codebook + bij combination, regardless of residual
#
# Reference (Flickr25k-CLIP):
#   v106b (K=64,  no local-res, per_codon, bij ON):       mAP 0.7407, P@1 0.917,  DNA 0.347
#   v128a (K=128, local-res g=1.0, per_codebook, bij ON): mAP 0.7365, P@1 0.897,  DNA 0.377
#   v131a (K=64,  local-res g=1.0, per_codebook, bij ON): mAP 0.7381, P@1 0.8975, DNA 0.266 (DISCARDED)
#   v132a (K=128, NO local-res,   per_codebook, bij ON):  TBD (parallel)
#   v133a (K=128, NO local-res,   per_codebook, bij OFF): TBD (parallel)
#   v134a (K=64,  NO local-res,   per_codebook, bij ON):  TBD (this run)
#
# Usage: bash scripts/train_v134a_v131a_noLocalRes_K64_flickr25k_clip.sh <GPU_ID>
set -eu

GPU="${1:-5}"
CACHE="${CACHE:-./cache/flickr25k_clip_v4plus}"
QWEN="${QWEN:-./cache/flickr25k_qwen_v4.jsonl}"
WHITEN_NPZ="${WHITEN_NPZ:-${CACHE}/text_whiten.npz}"
WHITEN_GAMMA="${WHITEN_GAMMA:-0.25}"
TAG="${TAG:-v134a_v131a_noLocalRes_K64_partialWhiten_gamma${WHITEN_GAMMA}}"
LOG="logs/${TAG}.log"
mkdir -p logs

if [ ! -f "$WHITEN_NPZ" ]; then
    echo "[run-v134a] whitening .npz missing — computing it now ..."
    /home/yschoi/.conda/envs/dna_hashing/bin/python \
        scripts/build_text_whiten_matrix.py \
        --cache_dir "$CACHE" \
        --out       "$WHITEN_NPZ"
fi

echo "[run-v134a] GPU=$GPU cache=$CACHE whiten=$WHITEN_NPZ gamma=$WHITEN_GAMMA K=64 v131a + NO local-res tag=$TAG"

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
    --codebook_size 64 \
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
