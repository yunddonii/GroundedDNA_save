#!/usr/bin/env bash
# v154a = v144a recipe with ONE single-delta change:
#   --lambda_codeword_codon_sinkhorn 0.0 -> 0.1
#
# Re-introduces the Sinkhorn-OT codeword <-> codon bijection loss (v106a)
# on top of the current v144a recipe. v106b proved this loss can shift
# the codeword-to-codon mapping toward bijection (DNA-uniq 0.347 at
# K=128, L=3 in v106b's earlier setup). Current v144a recipe adds:
# partial whitening (gamma=0.25), text_code_kl (lambda=0.02), cibhash
# per_codebook + dynamic tau, adaptive_topp -- none of which v106b had.
# The bijection loss has been OFF in v144a-v153b because earlier
# experiments showed mAP cost; this experiment re-tests it in a much
# improved baseline.
#
# Framing correction (2026-06-15, user-driven): the L=3 codon space
# (64^6 ≈ 6.9e10 distinct DNAs) is NOT structurally bottlenecked for
# 23K Flickr DB images. The supervised (jaccard) regime achieves
# cb_tuple ≈ DNA (bijection by pairwise tag signal). Unsupervised
# recipes WITHOUT this loss have ~2x codeword-codon collision because
# no term forces distinct codeword decodes. Bij loss is the natural
# unsupervised analogue of the jaccard pairwise mechanism.
#
# Hypothesis. Current v144a's accumulated indirect uniqueness signals
# (text_code_kl, cibhash per_codebook, partial whiten) ALREADY push the
# codon distribution toward diversity. Re-adding the explicit bij loss
# should compose: the indirect signals shape the routing/codebook, and
# the bij loss directly forces the codeword->codon decoder to spread.
#
# Predicted (v106b reference + v144a improvements):
#   DNA-uniq: 0.376 -> 0.45-0.55 (v106b achieved 0.347 with weaker base;
#                                  v144a's better base should overshoot)
#   cb_tuple: 0.664 -> 0.65-0.70 (codebook usage approximately unchanged)
#   mAP:      0.7499 -> 0.73-0.75 (small risk; bij loss has historical mAP cost)
#   NMI:      0.591 -> 0.60-0.65 (codebook decoupling under bij)
#
# Reference (Flickr25k-CLIP K=128, partial-whiten gamma=0.25):
#   v106b (K=128, L=3, bij ON, no current improvements): mAP 0.7407, DNA 0.347
#   v122a (K=256, L=4, bij ON): mAP 0.74?, DNA 0.551 (unsupervised SOTA territory)
#   v144a (K=128, L=3, bij OFF, current recipe): mAP 0.7499, DNA 0.376
#   v154a (K=128, L=3, bij ON, current recipe): TBD (this run)
#
# Usage: bash scripts/train_v154a_v144a_bijLoss_K128_flickr25k_clip.sh <GPU_ID>
set -eu

GPU="${1:-0}"
CACHE="${CACHE:-./cache/flickr25k_clip_v4plus}"
QWEN="${QWEN:-./cache/flickr25k_qwen_v4.jsonl}"
WHITEN_NPZ="${WHITEN_NPZ:-${CACHE}/text_whiten.npz}"
WHITEN_GAMMA="${WHITEN_GAMMA:-0.25}"
TAG="${TAG:-v154a_v144a_bijLoss_lam0p1_K128_partialWhiten_gamma${WHITEN_GAMMA}}"
LOG="logs/${TAG}.log"
mkdir -p logs

if [ ! -f "$WHITEN_NPZ" ]; then
    echo "[run-v154a] whitening .npz missing — computing it now ..."
    /home/yschoi/.conda/envs/dna_hashing/bin/python \
        scripts/build_text_whiten_matrix.py \
        --cache_dir "$CACHE" \
        --out       "$WHITEN_NPZ"
fi

echo "[run-v154a] GPU=$GPU cache=$CACHE whiten=$WHITEN_NPZ K=128 bij_loss=0.1 tag=$TAG"

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
    --lambda_text_code_kl 0.02 \
    --text_code_kl_tau_v 0.1 \
    --text_code_kl_tau_t 0.07 \
    --text_code_kl_conf_threshold 0.2 \
    --text_code_kl_skip_global \
    -ev -s 2>&1 | tee "$LOG"
