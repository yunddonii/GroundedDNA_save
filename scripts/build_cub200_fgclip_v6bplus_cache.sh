#!/usr/bin/env bash
# Build FG-CLIP-base CUB-200 v6bplus cache.
# Mirrors cache/cub200_clip_v6bplus structure but with qihoo360/fg-clip-base
# weights (visual + text + tokens) so we can A/B against the CLIP record
# (cub200_v160b_v6b_K64 mAP 0.0739).
set -e
GPU="${1:-0}"
BACKBONE="qihoo360/fg-clip-base"
QWEN="cache/cub200_qwen_v6b_trainset.jsonl"
ROOT="dataset/CUB_200"
SETTING="setting1"
CACHE="cache/cub200_fgclip_v6bplus"
CACHE_TOK="cache/cub200_fgclip_v6bplus_tokens"

mkdir -p "$CACHE" "$CACHE_TOK"

echo "[fgclip-cub-cache] step 1/3: visual + text_part on $CACHE"
CUDA_VISIBLE_DEVICES=$GPU /home/yschoi/.conda/envs/dna_hashing/bin/python extract_clip_features.py \
    --mode pathlist \
    --pathlist_root "$ROOT" \
    --pathlist_setting "$SETTING" \
    --qwen_cache_path "$QWEN" \
    --cache_dir "$CACHE" \
    --clip_backbone "$BACKBONE" \
    --tokenizer_name "$BACKBONE" \
    --save_aug_views 2 \
    --batch_size 128

echo "[fgclip-cub-cache] step 2/3: token-level text on $CACHE_TOK"
# Symlink visual + has_text from main cache
for f in visual_global.f16.npy visual_global_aug0.f16.npy visual_global_aug1.f16.npy \
         visual_tokens.f16.npy visual_tokens_aug0.f16.npy visual_tokens_aug1.f16.npy \
         has_text.bool.npy image_ids.json; do
    [ -e "$CACHE_TOK/$f" ] || ln -s "$(realpath $CACHE/$f)" "$CACHE_TOK/$f"
done
CUDA_VISIBLE_DEVICES=$GPU /home/yschoi/.conda/envs/dna_hashing/bin/python extract_clip_text_token_features.py \
    --qwen_cache "$QWEN" \
    --donor_dir "$CACHE" \
    --out_dir   "$CACHE_TOK" \
    --clip_backbone "$BACKBONE" \
    --tokenizer "$BACKBONE"

echo "[fgclip-cub-cache] step 3/3: whitening matrix"
CUDA_VISIBLE_DEVICES=$GPU /home/yschoi/.conda/envs/dna_hashing/bin/python scripts/build_text_whiten_matrix.py \
    --cache_dir "$CACHE" \
    --out       "$CACHE/text_whiten.npz"
[ -e "$CACHE_TOK/text_whiten.npz" ] || ln -s "$(realpath $CACHE/text_whiten.npz)" "$CACHE_TOK/text_whiten.npz"

echo "[fgclip-cub-cache] DONE"
ls -lh "$CACHE/" | head -15
echo "---"
ls -lh "$CACHE_TOK/" | head -15
