#!/usr/bin/env bash
# v135a = v133a with ONE change:
#   --codebook_size 128 -> 256
#
# v133a recipe (Flickr25k-CLIP commit 1eecc32):
#   K=128, L=3, bij OFF, NO local-residual, per_codebook text NtXent
#   M=6 on [B, 12], cibhash per_codebook + dynamic tau, partial whitening
#   gamma=0.25, paired-aug DNA NtXent OFF (cibhash replaces it).
#   Result: mAP 0.7541, P@1 0.9150, DNA 0.340, NMI 0.626, B1 0.128, B2 0.079,
#   dead avg 0.040 (cb5 = 106/128 used, 17 % dead).
#
# v135a hypothesis. L stays at 3 (|C|=4^3=64 codon possibilities) but K jumps
# to 256 -> 4x codeword-per-codon redundancy. Per v131a's causal mechanism
# (commit b00ac60): "K=128's 2x codeword redundancy multiplies the effective
# DNA space by 2^M=64 codeword-tuples per codon-tuple. At K=64 this
# redundancy disappears -> fewer distinct routes to the same codon-tuple ->
# more cross-sample collisions."
#
# At K=256 (4x redundancy), the same logic predicts 4^M=4096 codeword-tuples
# per codon-tuple. This is the bij-FREE regime (lambda_codeword_codon_sinkhorn
# = 0) so the marginal-constraint Sinkhorn does NOT force uniform distribution
# -- the model is FREE to populate the 4x redundancy space arbitrarily.
#
# Two-axis predictions:
#   (a) DNA-base unique should INCREASE vs v133a's 0.340 (more codeword
#       routes -> more distinct DNA codes possible). Toward v122a's 0.551.
#   (b) Codebook dead might INCREASE (256 slots per codebook to populate, vs
#       128); risk that some codebooks under-utilize their K=256 capacity.
#       v133a already had cb5 17% dead at K=128 -> v135a might see > 20%
#       dead at K=256.
#   (c) Retrieval (mAP/P@1) is uncertain: more granularity helps, but if
#       dead spikes, the active codeword pool shrinks proportionally.
#
# References (Flickr25k-CLIP, partial whitening gamma=0.25):
#   v133a (K=128, bij OFF, noLocalRes, perCb):  mAP 0.7541, P@1 0.9150, DNA 0.340, NMI 0.626
#   v122a (K=256, L=4,    bij ON,  localRes,   perCodon): mAP 0.7479, P@1 0.9215, DNA 0.551, NMI 0.627
#   v135a (K=256, L=3,    bij OFF, noLocalRes, perCb):    TBD (this run)
#
# Usage: bash scripts/train_v135a_v133a_K256_flickr25k_clip.sh <GPU_ID>
set -eu

GPU="${1:-0}"
CACHE="${CACHE:-./cache/flickr25k_clip_v4plus}"
QWEN="${QWEN:-./cache/flickr25k_qwen_v4.jsonl}"
WHITEN_NPZ="${WHITEN_NPZ:-${CACHE}/text_whiten.npz}"
WHITEN_GAMMA="${WHITEN_GAMMA:-0.25}"
TAG="${TAG:-v135a_v133a_K256_partialWhiten_gamma${WHITEN_GAMMA}}"
LOG="logs/${TAG}.log"
mkdir -p logs

if [ ! -f "$WHITEN_NPZ" ]; then
    echo "[run-v135a] whitening .npz missing — computing it now ..."
    /home/yschoi/.conda/envs/dna_hashing/bin/python \
        scripts/build_text_whiten_matrix.py \
        --cache_dir "$CACHE" \
        --out       "$WHITEN_NPZ"
fi

echo "[run-v135a] GPU=$GPU cache=$CACHE whiten=$WHITEN_NPZ gamma=$WHITEN_GAMMA K=256 (L=3, 4x codeword redundancy) v133a + K256 tag=$TAG"

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
    --codebook_size 256 \
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
    --eval_every 5 \
    --post_eval_compositional \
    --dna_distance_mode base \
    -ev -s 2>&1 | tee "$LOG"
