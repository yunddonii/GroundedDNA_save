#!/usr/bin/env bash
# v150a = v149a recipe + ONE delta: --cibhash_ntxent_source visual_token
#
# Carries v149a's --cibhash_ntxent_continuous flag (no STE-sign) on top.
# v150a's single conceptual change: instead of running NtXent on bit_probs
# (post-VQ + post-codon, 6-bit slice per codebook), run it on the pre-VQ
# semantic_visual_tokens [B, M=6, D] from the router output. D-dim
# continuous cosine uniformity gradient (vs the 7-level granularity ceiling
# that bit-mode imposes).
#
# Implementation. New flag --cibhash_ntxent_source {continuous_code,
# visual_token} (config.py). When "visual_token", the loss caller pulls
# outputs["semantic_visual_tokens"] from both views and invokes a new
# method _loss_cibhash_visual_per_codebook (loss_siglip2.py). KL term is
# auto-zeroed (Bernoulli KL undefined on continuous vectors). All other
# v149a flags retained, including cibhash dynamic tau alpha=0.3 (per-pair
# tau scaling from text-cos, exactly the same mechanism as v149a).
#
# Hypothesis. v149a demonstrated that STE-sign was the codebook utilization
# bottleneck (dead 0.065 -> 0.000) but NOT the DNA-uniq bottleneck (DNA
# unchanged at 0.376 -> 0.380, cb_tuple unchanged at 0.664 -> 0.665).
# Reason: cb_tuple is bounded by the codebook diversity that VQ assignment
# captures, which is gated by HOW VARIED the pre-VQ visual tokens are.
# v149a's NtXent gradient flows backward through codon -> VQ -> routing
# (long chain). v150a places the NtXent DIRECTLY on the routing output,
# so the gradient signal arrives much earlier and shapes the visual
# representation that drives VQ assignment.
#
# Expected (analysis-driven):
#   cb_tuple:   0.665 -> 0.75-0.85 (richer codeword combinations)
#   DNA-uniq:   0.380 -> 0.42-0.48 (via cb_tuple gain; pigeonhole still ~2x)
#   NMI:        0.628 -> 0.55-0.60 (per-codebook embedding becomes more
#                                    orthogonal in D-dim space)
#   dead:       0.000 -> 0.00-0.03 (similar to v149a, slight risk if router
#                                    over-concentrates)
#   mAP:        0.7354 -> 0.74-0.76 (visual SSL on continuous embedding
#                                     should not hurt retrieval)
#
# Reference (Flickr25k-CLIP K=128, partial-whiten gamma=0.25):
#   v144a (legacy STE-sign cibhash):           mAP 0.7499, NMI 0.591, DNA 0.376, cb_tuple 0.664, dead 0.065
#   v149a (continuous on bit_probs):           mAP 0.7354, NMI 0.628, DNA 0.380, cb_tuple 0.665, dead 0.000
#   v150a (continuous on visual_token):        TBD (this run)
#
# Usage: bash scripts/train_v150a_v149a_visualTokenCibhash_K128_flickr25k_clip.sh <GPU_ID>
set -eu

GPU="${1:-4}"
CACHE="${CACHE:-./cache/flickr25k_clip_v4plus}"
QWEN="${QWEN:-./cache/flickr25k_qwen_v4.jsonl}"
WHITEN_NPZ="${WHITEN_NPZ:-${CACHE}/text_whiten.npz}"
WHITEN_GAMMA="${WHITEN_GAMMA:-0.25}"
TAG="${TAG:-v150a_v149a_visualTokenCibhash_K128_partialWhiten_gamma${WHITEN_GAMMA}}"
LOG="logs/${TAG}.log"
mkdir -p logs

if [ ! -f "$WHITEN_NPZ" ]; then
    echo "[run-v150a] whitening .npz missing — computing it now ..."
    /home/yschoi/.conda/envs/dna_hashing/bin/python \
        scripts/build_text_whiten_matrix.py \
        --cache_dir "$CACHE" \
        --out       "$WHITEN_NPZ"
fi

echo "[run-v150a] GPU=$GPU cache=$CACHE whiten=$WHITEN_NPZ K=128 v149a + cibhash_ntxent_source=visual_token tag=$TAG"

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
