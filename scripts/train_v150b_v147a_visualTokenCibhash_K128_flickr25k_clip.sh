#!/usr/bin/env bash
# v150b = v147a recipe (UOT lambda=1.0 + topp 0.3/0.7 + text_code_kl 0.02) with
# TWO deltas: --cibhash_ntxent_continuous (v149 fix) + --cibhash_ntxent_source visual_token (v150 fix).
#
# v150a was v144a (baseline) + both flags. v150b is v147a (the Pareto-better
# compositional cell on K=128) + both flags. Tests whether the v150 visual-token
# contrastive composes with the UOT + sharper-topp Pareto-front.
#
# Recipe summary:
#   - codebook_size 128, codon L=3
#   - UOT lambda_a=lambda_b=1.0 (carried from v145a/v147a)
#   - routing_adaptive_topp 0.3/0.7 (v147a's sharper topp)
#   - text_code_kl 0.02 (v144a's pre-VQ KL distillation)
#   - cibhash per_codebook + dyn tau alpha=0.3
#   - cibhash_ntxent_continuous (v149 fix: no STE-sign)
#   - cibhash_ntxent_source visual_token (v150 fix: pre-VQ D-dim embedding NtXent)
#   - partial-whiten gamma=0.25, hash_target_mode siglip_cos
#   - bij OFF, no local-residual
#
# Hypothesis. v147a's UOT keeps dead low (0.008) while sharper topp pushes
# NMI down (0.566). Adding v149 + v150 NtXent on visual_token should:
#   - Raise codeword utilization further (v149 effect)
#   - Raise cb_tuple diversity by shaping pre-VQ routing (v150 effect)
#   - Without compromising the UOT + sharp topp compositional axis (no NMI regression)
# Together: a candidate that simultaneously holds compositional (NMI low),
# utilization (dead near 0), and uniqueness (DNA-uniq higher).
#
# Reference (Flickr25k-CLIP K=128, partial-whiten gamma=0.25):
#   v144a (baseline):           mAP 0.7499, NMI 0.591, DNA 0.376, cb_tuple 0.664, dead 0.065
#   v147a (UOT + sharper topp): mAP 0.7280, NMI 0.566, DNA 0.307, cb_tuple 0.771, dead 0.008
#   v149a (v144a + continuous): mAP 0.7354, NMI 0.628, DNA 0.380, cb_tuple 0.665, dead 0.000
#   v150a (v144a + visual tok): TBD (running)
#   v150b (v147a + visual tok): TBD (this run)
#
# Usage: bash scripts/train_v150b_v147a_visualTokenCibhash_K128_flickr25k_clip.sh <GPU_ID>
set -eu

GPU="${1:-1}"
CACHE="${CACHE:-./cache/flickr25k_clip_v4plus}"
QWEN="${QWEN:-./cache/flickr25k_qwen_v4.jsonl}"
WHITEN_NPZ="${WHITEN_NPZ:-${CACHE}/text_whiten.npz}"
WHITEN_GAMMA="${WHITEN_GAMMA:-0.25}"
TAG="${TAG:-v150b_v147a_visualTokenCibhash_K128_partialWhiten_gamma${WHITEN_GAMMA}}"
LOG="logs/${TAG}.log"
mkdir -p logs

if [ ! -f "$WHITEN_NPZ" ]; then
    echo "[run-v150b] whitening .npz missing — computing it now ..."
    /home/yschoi/.conda/envs/dna_hashing/bin/python \
        scripts/build_text_whiten_matrix.py \
        --cache_dir "$CACHE" \
        --out       "$WHITEN_NPZ"
fi

echo "[run-v150b] GPU=$GPU cache=$CACHE whiten=$WHITEN_NPZ K=128 v147a + cibhash visual_token + continuous tag=$TAG"

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
