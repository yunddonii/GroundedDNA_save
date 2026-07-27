#!/usr/bin/env bash
# Build NUS-WIDE CLIP cache for our training pipeline.
#
# Prerequisites:
#   cache/nuswide_qwen3_v4_trainset.jsonl exists (Qwen captions for 10,500 subset).
#   dataset/NUSWIDE/setting1/ contains train.txt (10,500 balanced) plus the
#   full test.txt / database.txt manifests.
#
# Outputs (single GPU, sequential; ETA ~2-3 hours total for 193K images):
#   1. cache/nuswide_clip/                  visual (visual_tokens, visual_global,
#                                            aug0/1) + text_part (pooled) + has_text +
#                                            image_ids/meta. Uses setting1 so
#                                            train.txt is 10,500 but test/database
#                                            still full 193K (unioned).
#   2. cache/nuswide_clip_tokens/           text_tokens.f16.npy + text_token_mask
#                                            for bidirectional pruning support.
#                                            Donor-symlinks visual files from #1.
#   3. cache/nuswide_clip_tokens/text_whiten.npz
#                                           partial-whitening matrix from
#                                            build_text_whiten_matrix.py.
#
# Training scripts will point --siglip2_feature_cache_dir at #2.
# Usage: bash scripts/build_nuswide_clip_cache.sh [GPU_ID]
set -eu

GPU="${1:-3}"
PY="/home/yschoi/.conda/envs/dna_hashing/bin/python"

QWEN_JSONL="/home/yschoi/GroundedDNA/cache/nuswide_qwen3_v4_trainset.jsonl"
POOLED_DIR="/home/yschoi/GroundedDNA/cache/nuswide_clip"
TOKENS_DIR="/home/yschoi/GroundedDNA/cache/nuswide_clip_tokens"
DATASET_ROOT="/home/yschoi/GroundedDNA/dataset/NUSWIDE"

if [ ! -f "$QWEN_JSONL" ]; then
    echo "[nuswide-cache] MISSING $QWEN_JSONL — run tools/qwen3_v4_nuswide_trainset.py first."
    exit 1
fi
QWEN_ROWS=$(grep -c '"codebook_texts"' "$QWEN_JSONL" 2>/dev/null || echo 0)
echo "[nuswide-cache] Qwen JSONL has $QWEN_ROWS rows."
if [ "$QWEN_ROWS" -lt 10000 ]; then
    echo "[nuswide-cache] WARNING: Qwen JSONL only has $QWEN_ROWS rows (<10000)."
    echo "               Waiting is recommended, but continuing on request."
fi

# ---- Step 1: pooled visual + text extraction (train+test+database over 193K) ----
if [ ! -f "$POOLED_DIR/text_part.f16.npy" ]; then
    echo "[nuswide-cache] STEP 1: pooled CLIP visual + text extraction"
    CUDA_VISIBLE_DEVICES="$GPU" $PY extract_clip_features.py \
        --mode pathlist \
        --pathlist_root "$DATASET_ROOT" \
        --pathlist_setting setting1 \
        --qwen_cache_path "$QWEN_JSONL" \
        --cache_dir "$POOLED_DIR" \
        --save_aug_views 2 \
        --batch_size 128 \
        --image_size 224
else
    echo "[nuswide-cache] Step 1 already done ($POOLED_DIR/text_part.f16.npy present)."
fi

# ---- Step 2: token-level text extraction + symlink visual donor ----
if [ ! -f "$TOKENS_DIR/text_tokens.f16.npy" ]; then
    echo "[nuswide-cache] STEP 2: token-level CLIP text extraction"
    mkdir -p "$TOKENS_DIR"
    CUDA_VISIBLE_DEVICES="$GPU" $PY extract_clip_text_token_features.py \
        --qwen_cache "$QWEN_JSONL" \
        --donor_dir  "$POOLED_DIR" \
        --out_dir    "$TOKENS_DIR"
else
    echo "[nuswide-cache] Step 2 already done ($TOKENS_DIR/text_tokens.f16.npy present)."
fi

# ---- Step 3: text whitening matrix -------------------------------------
WHITEN_NPZ="$TOKENS_DIR/text_whiten.npz"
if [ ! -f "$WHITEN_NPZ" ]; then
    echo "[nuswide-cache] STEP 3: text_whiten matrix"
    $PY scripts/build_text_whiten_matrix.py \
        --cache_dir "$TOKENS_DIR" \
        --out       "$WHITEN_NPZ"
else
    echo "[nuswide-cache] Step 3 already done ($WHITEN_NPZ present)."
fi

echo ""
echo "[nuswide-cache] DONE."
echo "  pooled cache : $POOLED_DIR"
echo "  tokens cache : $TOKENS_DIR (use as --siglip2_feature_cache_dir)"
echo "  whiten npz   : $WHITEN_NPZ"
