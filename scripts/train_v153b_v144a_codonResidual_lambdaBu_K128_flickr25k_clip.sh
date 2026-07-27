#!/usr/bin/env bash
# v153b = v144a recipe with TWO single-delta changes:
#   --codon_residual_gamma 0.0  -> 0.25  (post-VQ residual into codon head)
#   --lambda_bu             0.02 -> 0.05  (stronger codeword-balance + uncorrelated)
#
# Motivation. v153b stacks two complementary uniqueness regularizers:
#   - codon_residual_gamma 0.25: structural bypass of codon pigeonhole
#       (DNA can differ even when codebook_indices are identical)
#   - lambda_bu 0.05: stronger off-diagonal Gram penalty per codebook
#       (different samples should use different codewords)
# Mechanism stacking: lambda_bu drives cb_tuple ↑, codon_residual drives
# DNA-uniq ↑ even when cb_tuple is unchanged. Together they attack the
# DNA-uniq ceiling from two distinct angles.
#
# Reference (Flickr25k-CLIP K=128, partial-whiten gamma=0.25):
#   v144a (baseline):                   mAP 0.7499, NMI 0.591, DNA 0.376, cb_tuple 0.664, dead 0.065
#   v153a (codon_residual only):        TBD (parallel)
#   v153b (codon_residual + lambda_bu): TBD (this run)
#
# Predicted v153b vs v153a (incremental from lambda_bu):
#   cb_tuple +0.02-0.05 (better codeword spread)
#   DNA-uniq +0.01-0.04 (proportional to cb_tuple gain)
#   mAP risk: -0.005 to -0.02 (forced uniform usage breaks semantic clusters)
#
# Usage: bash scripts/train_v153b_v144a_codonResidual_lambdaBu_K128_flickr25k_clip.sh <GPU_ID>
set -eu

GPU="${1:-4}"
CACHE="${CACHE:-./cache/flickr25k_clip_v4plus}"
QWEN="${QWEN:-./cache/flickr25k_qwen_v4.jsonl}"
WHITEN_NPZ="${WHITEN_NPZ:-${CACHE}/text_whiten.npz}"
WHITEN_GAMMA="${WHITEN_GAMMA:-0.25}"
TAG="${TAG:-v153b_v144a_codonResidual_lambdaBu_K128_partialWhiten_gamma${WHITEN_GAMMA}}"
LOG="logs/${TAG}.log"
mkdir -p logs

if [ ! -f "$WHITEN_NPZ" ]; then
    echo "[run-v153b] whitening .npz missing — computing it now ..."
    /home/yschoi/.conda/envs/dna_hashing/bin/python \
        scripts/build_text_whiten_matrix.py \
        --cache_dir "$CACHE" \
        --out       "$WHITEN_NPZ"
fi

echo "[run-v153b] GPU=$GPU cache=$CACHE whiten=$WHITEN_NPZ K=128 codon_residual_gamma=0.25 + lambda_bu=0.05 tag=$TAG"

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
    --lambda_anchor 0.05 --lambda_dna 0.05 --lambda_bu 0.05 \
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
