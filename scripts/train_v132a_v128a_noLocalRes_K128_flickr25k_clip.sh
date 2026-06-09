#!/usr/bin/env bash
# v132a = v128a with ONE change:
#   REMOVE all local-residual flags (--local_residual_quant +
#   --local_residual_gamma 1.0 + --local_residual_text +
#   --local_residual_detach_global)
#
# Isolates the *local-residual quantization* (C0/global orthogonal
# residual on C_1..C_5 before VQ) contribution to v128a's composition
# axis. v128a's recipe is v126a EXACT + per_codebook text NtXent + bij
# at K=128 + local-residual gamma=1.0; v132a removes the last component.
#
# Hypothesis. Local-residual was introduced in v122 to force codebook
# specialization (each local codebook learns C0-orthogonal info). Per
# v131a's failure mode analysis, local-residual *also* destroys
# v106b-style codon diversity that depended on local slots sharing
# global axis with C0. v132a tests whether removing local-residual
# (a) recovers v106b-style DNA-uniq from v128a's 0.377 (predicted
# ~0.42-0.50 toward v122a's 0.551), (b) costs the v128a-family-best
# NMI 0.636 (predicted -0.02 to -0.03 toward v106b's 0.604), and
# (c) changes mAP/P@1 within +/-0.01.
#
# Reference table (Flickr25k-CLIP K=128, partial whitening gamma=0.25):
#   v106b (K=64,  no local-res, per_codon text):  mAP 0.7407, P@1 0.917,  DNA 0.347, NMI 0.604
#   v126a (K=128, local-res g=1.0, per_codon):    mAP 0.7633, P@1 0.917,  DNA 0.429, NMI 0.605
#   v128a (K=128, local-res g=1.0, per_codebook): mAP 0.7365, P@1 0.897,  DNA 0.377, NMI 0.636
#   v131a (K=64,  local-res g=1.0, per_codebook): mAP 0.7381, P@1 0.8975, DNA 0.266, NMI 0.633 (DISCARDED)
#   v132a (K=128, NO   local-res, per_codebook): TBD  (this run)
#
# Usage: bash scripts/train_v132a_v128a_noLocalRes_K128_flickr25k_clip.sh <GPU_ID>
set -eu

GPU="${1:-3}"
CACHE="${CACHE:-./cache/flickr25k_clip_v4plus}"
QWEN="${QWEN:-./cache/flickr25k_qwen_v4.jsonl}"
WHITEN_NPZ="${WHITEN_NPZ:-${CACHE}/text_whiten.npz}"
WHITEN_GAMMA="${WHITEN_GAMMA:-0.25}"
TAG="${TAG:-v132a_v128a_noLocalRes_K128_partialWhiten_gamma${WHITEN_GAMMA}}"
LOG="logs/${TAG}.log"
mkdir -p logs

if [ ! -f "$WHITEN_NPZ" ]; then
    echo "[run-v132a] whitening .npz missing — computing it now ..."
    /home/yschoi/.conda/envs/dna_hashing/bin/python \
        scripts/build_text_whiten_matrix.py \
        --cache_dir "$CACHE" \
        --out       "$WHITEN_NPZ"
fi

echo "[run-v132a] GPU=$GPU cache=$CACHE whiten=$WHITEN_NPZ gamma=$WHITEN_GAMMA K=128 v128a + NO local-res tag=$TAG"

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
    -ev -s 2>&1 | tee "$LOG"
