#!/usr/bin/env bash
# v136a = v132a with TWO changes:
#   (1) --lambda_codeword_codon_sinkhorn 0.1 -> 0.0  (bij OFF)
#   (2) ADD --codon_position_specific_head             (v69a flag; per-codon
#       position separate Linear(chunk, 4); 3 fc layers replace the shared
#       one to let codon positions 0/1/2 specialize)
#
# v132a recipe (Flickr25k-CLIP commit 1eecc32):
#   K=128, L=3, bij ON, NO local-residual, per_codebook text NtXent
#   M=6 InfoNCEs on [B, 12], cibhash per_codebook + dynamic tau,
#   partial whitening gamma=0.25, paired-aug DNA NtXent OFF (cibhash
#   replaces it).
#   Result on Flickr (DB=23K): mAP 0.7418, P@1 0.9070, NMI 0.636 (tied
#   family-max), B1 0.123, B2 0.078, DNA-uniq 0.367, dead 0.005.
#
# Position-specific CodonHead motivation. mscoco_v69a / mscoco_v81a era
# showed that letting codon positions 0,1,2 each have their own
# Linear(chunk, 4) (instead of a shared classifier with 3-way logits split)
# improved MSCOCO mAP +0.0096 vs the shared head at K=128, and was the
# foundation of mscoco_v78a/v81a final-checkpoint SOTA. The intuition: each
# codon position can specialize its 4-base alphabet to a different visual
# concept axis. v132a (and v133a) uses the *shared* classifier path. v136a
# tests whether position-specific heads + per_codebook text NtXent + cibhash
# + bij-OFF (recipe v132a without bij) combine constructively on Flickr.
#
# Hypothesis on Flickr.
#   (a) DNA-uniq:    UP toward v122a's 0.551 (each position has more
#                    capacity to differentiate; per-position Linear has
#                    3x more parameters than the shared one).
#   (b) NMI:         tied or slight gain vs v132a's 0.636 (per_codebook
#                    text NtXent + per-position heads should compound).
#   (c) mAP / P@1:   uncertain. mscoco showed +0.0096 mAP. On Flickr, the
#                    smaller DB (23K) gives less specialization budget per
#                    position cluster -> +0.005 mAP would be at the high
#                    end of the expected range.
#   (d) dead:        could rise -- 3 separate heads = more chance for one
#                    head to under-converge per codebook. Watch [cb5
#                    dead] which already cost v133a 17% utilization.
#
# Reference (Flickr25k-CLIP, K=128, partial whitening gamma=0.25):
#   v132a (bij ON,  shared head):  mAP 0.7418, P@1 0.9070, DNA 0.367, NMI 0.636
#   v133a (bij OFF, shared head):  mAP 0.7541, P@1 0.9150, DNA 0.340, NMI 0.626
#   v126a (bij ON,  shared head, perCodon text, localRes): mAP 0.7633 mAP champ
#   v136a (bij OFF, *position-specific* head, perCb text):   TBD (this run)
#
# Usage: bash scripts/train_v136a_v132a_noBij_posSpec_K128_flickr25k_clip.sh <GPU_ID>
set -eu

GPU="${1:-0}"
CACHE="${CACHE:-./cache/flickr25k_clip_v4plus}"
QWEN="${QWEN:-./cache/flickr25k_qwen_v4.jsonl}"
WHITEN_NPZ="${WHITEN_NPZ:-${CACHE}/text_whiten.npz}"
WHITEN_GAMMA="${WHITEN_GAMMA:-0.25}"
TAG="${TAG:-v136a_v132a_noBij_posSpec_K128_partialWhiten_gamma${WHITEN_GAMMA}}"
LOG="logs/${TAG}.log"
mkdir -p logs

if [ ! -f "$WHITEN_NPZ" ]; then
    echo "[run-v136a] whitening .npz missing — computing it now ..."
    /home/yschoi/.conda/envs/dna_hashing/bin/python \
        scripts/build_text_whiten_matrix.py \
        --cache_dir "$CACHE" \
        --out       "$WHITEN_NPZ"
fi

echo "[run-v136a] GPU=$GPU cache=$CACHE whiten=$WHITEN_NPZ gamma=$WHITEN_GAMMA K=128 (bij OFF + posSpec head) v132a + posSpec + noBij tag=$TAG"

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
    --codon_position_specific_head \
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
