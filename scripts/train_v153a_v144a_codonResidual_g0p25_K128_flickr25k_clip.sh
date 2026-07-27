#!/usr/bin/env bash
# v153a = v144a recipe with ONE single-delta change:
#   --codon_residual_gamma 0.0 -> 0.25
#
# Motivation. v149a + mscoco_v148b established that DNA-uniq's ceiling is
# the codon-space pigeonhole (K=128 codewords -> 4^3=64 codons -> ~1.77x
# forced collision on Flickr K=128). codon_residual_gamma injects
# gamma * (z - q) into the codon head input, allowing two images with
# the same codebook_indices to decode to DIFFERENT codons -- structurally
# bypassing the codon pigeonhole without changing codebook structure.
#
# Config docstring (config.py:620-629):
#   "head receives both the quantized codeword AND a residual signal
#    (z - q) scaled by gamma. Lets two images sharing the same codeword
#    index produce different codons -> higher unique-code ratio without
#    breaking compositional structure (codebook indices unchanged)."
#
# Expected (analysis-driven):
#   cb_tuple: 0.664 -> 0.664-0.68 (codebook_indices unchanged so cb_tuple
#                                   moves only via codon-side coupling)
#   DNA-uniq: 0.376 -> 0.45-0.55 (codon pigeonhole partially broken)
#   mAP:      0.7499 -> 0.74-0.76 (small risk from residual injecting noise)
#   NMI:      0.591 -> 0.60-0.62 (slight increase from per-image codon variation)
#
# Pairs with v153b (= v153a + lambda_bu 0.02 -> 0.05) to test whether
# stronger codeword-balance regularization stacks with codon-residual.
#
# Reference (Flickr25k-CLIP K=128, partial-whiten gamma=0.25):
#   v144a (baseline, gamma=0): mAP 0.7499, NMI 0.591, DNA 0.376, cb_tuple 0.664, dead 0.065
#   v149a (continuous bits):   mAP 0.7354, NMI 0.628, DNA 0.380, cb_tuple 0.665, dead 0.000
#   v150b (Pareto-better):     mAP 0.7509, NMI 0.612, DNA 0.329, cb_tuple 0.566, B1 0.159
#   v153a (codon residual):    TBD (this run)
#   v153b (residual + lambda_bu): TBD (parallel)
#
# Usage: bash scripts/train_v153a_v144a_codonResidual_g0p25_K128_flickr25k_clip.sh <GPU_ID>
set -eu

GPU="${1:-0}"
CACHE="${CACHE:-./cache/flickr25k_clip_v4plus}"
QWEN="${QWEN:-./cache/flickr25k_qwen_v4.jsonl}"
WHITEN_NPZ="${WHITEN_NPZ:-${CACHE}/text_whiten.npz}"
WHITEN_GAMMA="${WHITEN_GAMMA:-0.25}"
TAG="${TAG:-v153a_v144a_codonResidual_g0p25_K128_partialWhiten_gamma${WHITEN_GAMMA}}"
LOG="logs/${TAG}.log"
mkdir -p logs

if [ ! -f "$WHITEN_NPZ" ]; then
    echo "[run-v153a] whitening .npz missing — computing it now ..."
    /home/yschoi/.conda/envs/dna_hashing/bin/python \
        scripts/build_text_whiten_matrix.py \
        --cache_dir "$CACHE" \
        --out       "$WHITEN_NPZ"
fi

echo "[run-v153a] GPU=$GPU cache=$CACHE whiten=$WHITEN_NPZ K=128 codon_residual_gamma=0.25 tag=$TAG"

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
    --codon_residual_gamma 0.25 \
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
