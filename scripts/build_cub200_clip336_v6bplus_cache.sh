#!/usr/bin/env bash
# CUB-200 v6b cache at 336x336 input resolution (CLIP-ViT-B/16 position
# embeddings bicubic-interpolated from 14x14 -> 21x21 = 441 patches).
# Single delta from cub200_clip_v6bplus: --image_size 224 -> 336.
# Text path (text_part / text_tokens / text_whiten) reuses the 224 cache via
# symlinks since the text encoder + caption are unchanged.
set -e
GPU="${1:-0}"
BACKBONE="openai/clip-vit-base-patch16"
QWEN="cache/cub200_qwen_v6b_trainset.jsonl"
ROOT="dataset/CUB_200"
SETTING="setting1"
CACHE="cache/cub200_clip336_v6bplus"
CACHE_TOK="cache/cub200_clip336_v6bplus_tokens"
SRC_224="$PWD/cache/cub200_clip_v6bplus"
SRC_224_TOK="$PWD/cache/cub200_clip_v6bplus_tokens"

mkdir -p "$CACHE" "$CACHE_TOK"

echo "[cub-clip336-cache] step 1/3: visual @ 336x336 + text_part on $CACHE"
CUDA_VISIBLE_DEVICES=$GPU /home/yschoi/.conda/envs/dna_hashing/bin/python extract_clip_features.py \
    --mode pathlist \
    --pathlist_root "$ROOT" \
    --pathlist_setting "$SETTING" \
    --qwen_cache_path "$QWEN" \
    --cache_dir "$CACHE" \
    --clip_backbone "$BACKBONE" \
    --tokenizer_name "$BACKBONE" \
    --image_size 336 \
    --save_aug_views 2 \
    --batch_size 64

echo "[cub-clip336-cache] step 2/3: text token cache (reuse 224's text_tokens since text encoder unchanged)"
# Symlink visual + has_text from main cache into tokens dir
for f in visual_global.f16.npy visual_global_aug0.f16.npy visual_global_aug1.f16.npy \
         visual_tokens.f16.npy visual_tokens_aug0.f16.npy visual_tokens_aug1.f16.npy \
         has_text.bool.npy image_ids.json text_part.f16.npy; do
    [ -e "$CACHE_TOK/$f" ] || ln -s "$(realpath $CACHE/$f)" "$CACHE_TOK/$f"
done
# text_tokens / text_token_mask from 224 cache (text path is identical)
for f in text_tokens.f16.npy text_token_mask.bool.npy; do
    [ -e "$CACHE_TOK/$f" ] || ln -s "$SRC_224_TOK/$f" "$CACHE_TOK/$f"
done

echo "[cub-clip336-cache] step 3/3: whitening matrix"
CUDA_VISIBLE_DEVICES=$GPU /home/yschoi/.conda/envs/dna_hashing/bin/python scripts/build_text_whiten_matrix.py \
    --cache_dir "$CACHE" \
    --out       "$CACHE/text_whiten.npz"
[ -e "$CACHE_TOK/text_whiten.npz" ] || ln -s "$(realpath $CACHE/text_whiten.npz)" "$CACHE_TOK/text_whiten.npz"

echo "[cub-clip336-cache] DONE"
ls -lh "$CACHE/" | head -15
echo "---"
ls -lh "$CACHE_TOK/" | head -15
