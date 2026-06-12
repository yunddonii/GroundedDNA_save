#!/usr/bin/env bash
# v149a = v144a recipe + ONE delta: --cibhash_ntxent_continuous
#
# Motivation. CIBHash NtXent is currently computed on STE-sign(bit_probs)
# in {-1, +1}^6 per codebook. The 7-level cosine granularity ceiling
# (from 6-bit signed slices) neuters the Wang-Isola uniformity gradient.
# v149a replaces STE-sign with continuous shifted bits (2*p - 1) in (-1, +1),
# restoring full continuous cosine granularity. DNA code path unchanged
# (computed at inference via argmax) -- this is exactly the gradient cut
# between continuous representation and binary DNA code.
#
# Expected outcome (analysis-driven, not yet confirmed):
#   DNA-uniq: 0.376 -> 0.45-0.55  (per_codebook ceiling partially broken)
#   NMI:      0.591 -> 0.58-0.62  (compositional structure preserved)
#   mAP:      0.7499 -> 0.74-0.76 (uniformity gradient stronger or equal)
#   dead:     0.065 -> 0.04-0.07  (small change)
#
# Same loss decomposition as v144a otherwise:
#   - UOT OFF (sinkhorn balanced default, no --sinkhorn_lambda_a/b)
#   - adaptive_topp 0.5/0.9 (v144a setting)
#   - text_code_kl 0.02 (v144a's pre-VQ KL distillation)
#   - cibhash per_codebook + dyn tau alpha=0.3 (v144a setting)
#   - partial-whiten gamma=0.25
#
# Reference (Flickr25k-CLIP K=128, partial-whiten gamma=0.25):
#   v144a (STE-sign cibhash):              mAP 0.7499, NMI 0.591, DNA 0.376, dead 0.065
#   v149a (continuous cibhash NtXent):     TBD (this run)
#
# Usage: bash scripts/train_v149a_v144a_cibhashContinuous_K128_flickr25k_clip.sh <GPU_ID>
set -eu

GPU="${1:-1}"
CACHE="${CACHE:-./cache/flickr25k_clip_v4plus}"
QWEN="${QWEN:-./cache/flickr25k_qwen_v4.jsonl}"
WHITEN_NPZ="${WHITEN_NPZ:-${CACHE}/text_whiten.npz}"
WHITEN_GAMMA="${WHITEN_GAMMA:-0.25}"
TAG="${TAG:-v149a_v144a_cibhashContinuous_K128_partialWhiten_gamma${WHITEN_GAMMA}}"
LOG="logs/${TAG}.log"
mkdir -p logs

if [ ! -f "$WHITEN_NPZ" ]; then
    echo "[run-v149a] whitening .npz missing — computing it now ..."
    /home/yschoi/.conda/envs/dna_hashing/bin/python \
        scripts/build_text_whiten_matrix.py \
        --cache_dir "$CACHE" \
        --out       "$WHITEN_NPZ"
fi

echo "[run-v149a] GPU=$GPU cache=$CACHE whiten=$WHITEN_NPZ K=128 v144a + cibhashContinuous tag=$TAG"

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
    --lambda_codeword_codon_sinkhorn 0.0 \
    --lambda_cibhash_ntxent 1.0 \
    --lambda_cibhash_kl 0.001 \
    --cibhash_mode per_codebook \
    --cibhash_temperature 0.3 \
    --cibhash_dynamic_tau \
    --cibhash_dynamic_tau_alpha 0.3 \
    --cibhash_ntxent_continuous \
    --eval_every 5 \
    --post_eval_compositional \
    --dna_distance_mode base \
    --lambda_text_code_kl 0.02 \
    --text_code_kl_tau_v 0.1 \
    --text_code_kl_tau_t 0.07 \
    --text_code_kl_conf_threshold 0.2 \
    --text_code_kl_skip_global \
    -ev -s 2>&1 | tee "$LOG"
