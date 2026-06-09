#!/usr/bin/env bash
# v137a = v133a with TWO changes:
#   (1) --cibhash_mode per_codebook -> global  (full 36-bit DNA code, one
#       NtXent + one KL on the flattened [B, 36] hash)
#   (2) REMOVE --cibhash_dynamic_tau (and --cibhash_dynamic_tau_alpha 0.3)
#       so cibhash uses STATIC temperature=0.3 for all batch pairs
#
# v133a recipe (Flickr25k-CLIP commit 1eecc32):
#   K=128, L=3, bij OFF, NO local-residual, per_codebook text NtXent
#   (M=6 InfoNCEs on [B, 12]), **cibhash per_codebook + dynamic tau**,
#   partial whitening gamma=0.25, paired-aug DNA NtXent OFF.
#   Result Flickr (DB=23K): mAP 0.7541, P@1 0.9150, DNA-base 0.340,
#   NMI 0.626, B1 0.128, B2 0.079, dead 0.040 (cb5 17% dead).
#
# Hypothesis. v133a (the Flickr family mAP-best among noLocalRes runs)
# applies CIBHash NtXent + KL per_codebook -- 6 INDEPENDENT 6-bit InfoNCE
# losses, averaged. v137a tests the ORIGINAL CIBHash design: one
# discriminative loss on the FULL 36-bit DNA hash. This is the closest
# *direct apples-to-apples* comparison with the external CIBHash-CLIP
# baseline (mAP 0.6844 on Flickr, mAP 0.5842 on MSCOCO).
#
# Concretely, in our setup the global cibhash branch (loss_siglip2.py:987):
#   z1_flat = z_v1.reshape(B, M*K_bits)        # [B, 36] binary hash
#   z2_flat = z_v2.reshape(B, M*K_bits)
#   sim = (cat(z1, z2) @ cat(z1, z2).T) / T    # [2B, 2B]
#   ntxent = CE(sim, positives at diag offsets)
#   kl = sym_KL(bits_v1_flat, bits_v2_flat)    # one KL on [B, 36]
#
# Two-axis predictions vs v133a.
#   (a) NtXent at the GLOBAL level removes the per-codebook 6-way
#       discriminative pressure -> codebook specialization weakens ->
#       NMI predicted to DROP from 0.626 toward ~0.55 (closer to CIBHash
#       baseline 0.157 but still high because per_codebook text NtXent
#       still applies).
#   (b) Retrieval (mAP/P@1) is uncertain:
#       - Global CIBHash matches the baseline's design exactly -> mAP
#         could rise toward CIBHash 0.6844 IF the per-codebook structure
#         was hurting. But it also matches our v118a/v119a era which had
#         mAP 0.7721/0.7620, and v133a's 0.7541 was the family-noLocalRes
#         peak with per_codebook cibhash. So global mode may give mAP
#         around 0.75-0.77 with weaker compositional anchoring.
#   (c) Dead: per_codebook cibhash explicitly forces each codebook to be
#       discriminative -> with global mode the model has no per-codebook
#       discriminative push (only the per_codebook TEXT NtXent remains).
#       Expected: dead slightly increases vs v133a's 0.040.
#   (d) DNA-uniq: global mode encourages the full 36-bit code to be
#       diverse across samples -> DNA-uniq could rise from 0.340 toward
#       0.45-0.55 range.
#
# Dynamic tau OFF rationale. Dynamic tau (alpha=0.3) reweights cibhash
# per-pair temperature by text similarity -> only makes sense when
# cibhash operates per_codebook (each codebook has its own text segment).
# In global mode there's no per-codebook text geometry to condition on,
# so dynamic tau loses its justification. We turn it off to keep the
# global comparison clean (matches the original CIBHash paper's static
# T=0.3).
#
# Reference (Flickr25k-CLIP K=128, partial whitening gamma=0.25):
#   CIBHash-CLIP baseline (flat hash):       mAP 0.6844, P@1 0.9365, uniq 0.968, NMI 0.157
#   v133a (perCb cibhash + dyn tau):          mAP 0.7541, P@1 0.9150, DNA 0.340, NMI 0.626
#   v137a (global cibhash + static tau):      TBD (this run)
#
# Usage: bash scripts/train_v137a_v133a_globalCib_noDynTau_K128_flickr25k_clip.sh <GPU_ID>
set -eu

GPU="${1:-0}"
CACHE="${CACHE:-./cache/flickr25k_clip_v4plus}"
QWEN="${QWEN:-./cache/flickr25k_qwen_v4.jsonl}"
WHITEN_NPZ="${WHITEN_NPZ:-${CACHE}/text_whiten.npz}"
WHITEN_GAMMA="${WHITEN_GAMMA:-0.25}"
TAG="${TAG:-v137a_v133a_globalCib_noDynTau_K128_partialWhiten_gamma${WHITEN_GAMMA}}"
LOG="logs/${TAG}.log"
mkdir -p logs

if [ ! -f "$WHITEN_NPZ" ]; then
    echo "[run-v137a] whitening .npz missing — computing it now ..."
    /home/yschoi/.conda/envs/dna_hashing/bin/python \
        scripts/build_text_whiten_matrix.py \
        --cache_dir "$CACHE" \
        --out       "$WHITEN_NPZ"
fi

echo "[run-v137a] GPU=$GPU cache=$CACHE whiten=$WHITEN_NPZ gamma=$WHITEN_GAMMA K=128 (cibhash global + STATIC tau) tag=$TAG"

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
    --cibhash_mode global \
    --cibhash_temperature 0.3 \
    --eval_every 5 \
    --post_eval_compositional \
    --dna_distance_mode base \
    -ev -s 2>&1 | tee "$LOG"
