#!/usr/bin/env bash
# v108c-v3 = v108a + --codebook_update gradient + --text_init_codebook mean (no commitment).
# Tests "pure prototype learning + text-supervised init":
#   - gradient mode (codebook = nn.Parameter, learnable)
#   - text init (codebook starts in semantic space, prevents collapse)
#   - proto_cluster_cos maintains training (cosine InfoNCE z <-> codeword)
#   - NO VQ commitment (lambda_vq = 0)
# Most aligned with user's contribution if it actually trains.
set -eu
GPU="${1:-1}"
CACHE="${CACHE:-./cache/flickr25k_clip_v4plus}"
QWEN="${QWEN:-./cache/flickr25k_qwen_v4.jsonl}"
TAG="${TAG:-v108c_v3_v108a_gradPlusTextInit}"
LOG="logs/${TAG}.log"
mkdir -p logs
echo "[run-v108c-v3] GPU=$GPU tag=$TAG"
CUDA_VISIBLE_DEVICES="$GPU" \
/home/yschoi/.conda/envs/dna_hashing/bin/python train_siglip2.py \
    --tag "$TAG" --dataset Flickr25k --setting 1 \
    --dataset_dir /home/yschoi/GroundedDNA/dataset \
    --num_devices 0 -bs 64 -e 60 --proj_lr 1e-3 --num_workers 4 \
    --qwen_text_cache_path "$QWEN" --siglip2_feature_cache_dir "$CACHE" \
    --backbone_type clip --codebook_size 64 \
    --c_global_source siglip2_global --per_slot_text_adapter \
    --global_gate_init_logit 4.595 \
    --router_type sinkhorn \
    --sinkhorn_epsilon_init 1.0 --sinkhorn_epsilon_final 0.1 \
    --routing_adaptive_topp --routing_adaptive_topp_min 0.5 --routing_adaptive_topp_max 0.9 \
    --codon_residual_gamma 0.0 \
    --codebook_update gradient \
    --use_paired_aug_ntxent --lambda_ntxent 1.0 --ntxent_temperature 0.3 \
    --ntxent_mode per_codebook --ntxent_dynamic_tau --ntxent_dynamic_tau_alpha 0.3 \
    --lambda_hash 0.0 --lambda_hash_hard 0.0 --lambda_hash_type mse \
    --hash_target_mode siglip_cos \
    --lambda_text_hash 0.05 --lambda_text_hash_ntxent 0.05 --text_hash_ntxent_temperature 0.07 \
    --lambda_wasserstein 0.05 \
    --lambda_vq 0.0 --lambda_quant 0.0 \
    --lambda_anchor 0.0 --lambda_dna 0.0 --lambda_bu 0.0 \
    --eta_base_balance 0.3 \
    --lambda_codeword_codon_sinkhorn 0.0 \
    --lambda_proto_cluster_cos 0.5 --proto_cluster_cos_tau 0.1 \
    --text_init_codebook mean --text_init_subset 4096 --text_init_seed 42 \
    --eval_every 5 --post_eval_compositional --dna_distance_mode base \
    -ev -s 2>&1 | tee "$LOG"
