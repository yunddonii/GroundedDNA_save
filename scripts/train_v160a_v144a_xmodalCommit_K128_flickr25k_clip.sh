#!/usr/bin/env bash
# v160a = v144a recipe with ONE single-delta change:
#   --lambda_xmodal_commit 0.025   (= beta/2 in Uni-Code Eq.8, NeurIPS 2023)
#
# Motivation. Uni-Code (Xia et al., NeurIPS 2023) introduces a cross-modal
# extension to the standard VQ-VAE commitment loss:
#   L_commit^a = beta * ||phi^a(x^a) - sg[e^a]||^2 + (beta/2) * ||phi^a(x^a) - sg[e^b]||^2
# The first term is the STANDARD self-modality commitment (already covered
# by --lambda_quant = 0.05 in v144a; this stays ON). The second term pulls
# modality A's encoder output toward modality B's QUANTIZED codeword,
# enforcing cross-modal alignment AT THE CODEBOOK level.
#
# Our implementation applies this SYMMETRICALLY across the two modalities:
#   L_xmodal = 0.5 * (||z_v - sg[q_t]||^2 + ||z_t - sg[q_v]||^2)
# where z_v = semantic_visual_tokens (visual encoder output),
#       z_t = text_part_tokens (text adapter output),
#       q_v = quantized_tokens_raw (visual quantized codeword),
#       q_t = text_quantized_tokens (text quantized codeword, EMA-disabled pass).
# Weight: lambda_xmodal_commit = 0.025 (= beta/2 when lambda_quant = beta = 0.05).
#
# Comparison to discarded cw_xmodal (v93a/v151a/v151b):
#   cw_xmodal: per-codebook InfoNCE between visual and text quantized
#   xmodal_commit: per-codebook L2 alignment between encoder output and
#                  OPPOSITE modality's quantized codeword
# Different layers and different geometries; we expect xmodal_commit to be
# more stable than the historically failed cw_xmodal because (a) it is MSE
# (deterministic) not contrastive (batch-dependent saturation), and
# (b) it operates at the encoder-output to quantized-codeword level, not
# at the quantized-vs-quantized level.
#
# Standard commitment (lambda_quant 0.05) STAYS ON. Eq.(8) is "self + cross",
# not "cross only".
#
# Reference (Flickr25k-CLIP K=128, partial-whiten gamma=0.25):
#   v144a (baseline):                                       mAP 0.7499, NMI 0.591, DNA 0.376, B1 0.125, B2 0.076
#   v149a (continuous bits NtXent):                         mAP 0.7354, DNA 0.380
#   v150b (Pareto-better, visual_token NtXent + UOT):       mAP 0.7509, B1 0.159, B2 0.098
#   v154a (bij loss 0.1):                                   mAP 0.7350, DNA 0.396
#   v160a (Eq.8 cross-modal commitment):                    TBD (this run)
#
# Predicted (analysis-driven):
#   mAP:        0.7499 -> 0.74-0.76 (small risk)
#   DNA-uniq:   0.376 -> 0.38-0.42 (codebook alignment is not the codon-uniq bottleneck)
#   NMI:       0.591 -> 0.59-0.62 (compositional structure preserved)
#   B1 (interp):0.125 -> 0.13-0.16 (text-anchored compositional lift expected)
#   B2:         0.076 -> 0.08-0.10
#
# Watch first mid-eval (ep 4):
#   mAP >= 0.65 AND unique >= 0.10 -> normal trajectory, continue
#   mAP < 0.65 OR unique < 0.10    -> catastrophic, KILL
#
# Usage: bash scripts/train_v160a_v144a_xmodalCommit_K128_flickr25k_clip.sh <GPU_ID>
set -eu

GPU="${1:-0}"
CACHE="${CACHE:-./cache/flickr25k_clip_v4plus}"
QWEN="${QWEN:-./cache/flickr25k_qwen_v4.jsonl}"
WHITEN_NPZ="${WHITEN_NPZ:-${CACHE}/text_whiten.npz}"
WHITEN_GAMMA="${WHITEN_GAMMA:-0.25}"
TAG="${TAG:-v160a_v144a_xmodalCommit_0p025_K128_partialWhiten_gamma${WHITEN_GAMMA}}"
LOG="logs/${TAG}.log"
mkdir -p logs

if [ ! -f "$WHITEN_NPZ" ]; then
    echo "[run-v160a] whitening .npz missing — computing it now ..."
    /home/yschoi/.conda/envs/dna_hashing/bin/python \
        scripts/build_text_whiten_matrix.py \
        --cache_dir "$CACHE" \
        --out       "$WHITEN_NPZ"
fi

echo "[run-v160a] GPU=$GPU cache=$CACHE whiten=$WHITEN_NPZ K=128 xmodal commit (Uni-Code Eq.8) tag=$TAG"

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
