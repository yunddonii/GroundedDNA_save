#!/usr/bin/env bash
# v160b_qwen3 = v160b recipe with ONE SINGLE-DELTA:
#   QWEN caption rev:  qwen_v4 -> qwen3_v4_trainset (Qwen3-VL-8B-Instruct)
#   donor cache:       flickr25k_clip_v4plus -> flickr25k_clip_v4plus_qwen3_tokens_FAIRrankL8K3
#
# Re-runs the v160b multi-axis champion recipe under the same caption rev
# as v162b (qwen3 v4 trainset, 5K coverage, Qwen3-VL-8B-Instruct VLM).
# This is the proper baseline for the grounded-text-routing comparison
# on the Flickr family champion.
#
# Reference (v160b Flickr25k-CLIP K=128, qwen_v4 = previous Qwen VLM):
#   mAP 0.7390, P@1 0.917, DNA 0.400, cb_tuple 0.536, NMI 0.630, L<->L 0.671,
#   B1 0.162, B2 0.100, dead 0.000, codewords used [128]x6, collision 1.34x
#   v160b_qwen3 (this run): TBD
#
# Single delta vs v160b:
#   --qwen_text_cache_path cache/flickr25k_qwen3_v4_trainset.jsonl  (rev v4_trainset)
#   --siglip2_feature_cache_dir cache/flickr25k_clip_v4plus_qwen3_tokens_FAIRrankL8K3   (qwen3 pooled embeds)
#   text_whiten.npz built on qwen3 donor if missing (auto via build_text_whiten_matrix.py)
#
# Note: qwen3 v4_trainset has 5K image coverage (20%), so the text path
# trains on a smaller effective sample than qwen_v4 (25K, 100%). Compare
# against v162b (= this base + grounded text routing) to isolate the
# grounded-routing effect under identical caption rev.
#
# Usage: bash scripts/train_v160b_qwen3_v150b_xmodalCommit_K128_flickr25k_clip.sh <GPU_ID>
set -eu

GPU="${1:-2}"
CACHE="${CACHE:-./cache/flickr25k_clip_v4plus_qwen3_tokens_FAIRrankL8K3}"
QWEN="${QWEN:-./cache/flickr25k_qwen3_v4_trainset.jsonl}"
WHITEN_NPZ="${WHITEN_NPZ:-${CACHE}/text_whiten.npz}"
WHITEN_GAMMA="${WHITEN_GAMMA:-0.25}"
TAG="${TAG:-flickr25k_v180_H3_skipGlobalXmodal_K128_partialWhiten_gamma${WHITEN_GAMMA}}"
LOG="logs/${TAG}.log"
mkdir -p logs

if [ ! -f "$WHITEN_NPZ" ]; then
    echo "[run-v160b-qwen3] whitening .npz missing — building from qwen3 donor ..."
    /home/yschoi/.conda/envs/dna_hashing/bin/python \
        scripts/build_text_whiten_matrix.py \
        --cache_dir "$CACHE" \
        --out       "$WHITEN_NPZ"
fi

echo "[run-v160b-qwen3] GPU=$GPU cache=$CACHE qwen=$QWEN whiten=$WHITEN_NPZ K=128 tag=$TAG"

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
    --lambda_xmodal_commit 0.05 \
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
    --lambda_text_code_kl 0.05 \
    --text_code_kl_tau_v 0.1 \
    --text_code_kl_tau_t 0.07 \
    --text_code_kl_conf_threshold 0.2 \
    --text_code_kl_skip_global \
    --xmodal_commit_skip_global \
    --text_hash_ntxent_skip_global \
    -ev -s 2>&1 | tee "$LOG"
