#!/usr/bin/env bash
# Continue the CIFAR-v4 prompt-ablation pipeline after feature extraction:
# wait for extraction -> text tokens -> whitening -> P0 2-stage train + eval.
set -u
cd /home/yschoi/GroundedDNA
PY=/home/yschoi/.conda/envs/dna_hashing/bin/python
GPU="${1:-3}"
C=cache/cifar10_clip_v4_tokens
echo "[cifar-v4] waiting for feature extraction ..."
until [ -f "$C/text_part.f16.npy" ] && ! pgrep -f "extract_clip_features_cifar10.py.*cifar10_clip_v4_tokens" >/dev/null 2>&1; do sleep 30; done
echo "[cifar-v4] features done $(date '+%F %T')"
CUDA_VISIBLE_DEVICES=$GPU $PY extract_clip_text_tokens_cifar10.py --cache_dir "$C" \
    --qwen_cache_path cache/cifar10_qwen_v4.jsonl --text_max_length 32 || { echo "[cifar-v4] token extract FAILED"; exit 1; }
echo "[cifar-v4] tokens done $(date '+%F %T')"
$PY scripts/build_opt_train_rows.py --dataset CIFAR10 --cache_dir "$C" --val_split_ratio 0.1 --val_split_seed 42 --out "$C/opt_train_rows.npy"
$PY scripts/build_opt_train_rows.py --dataset CIFAR10 --cache_dir "$C" --all_train --out "$C/train_all_rows.npy"
$PY scripts/build_text_whiten_matrix.py --cache_dir "$C" --row_index_npy "$C/opt_train_rows.npy"  --out "$C/text_whiten_optTrain.npz"
$PY scripts/build_text_whiten_matrix.py --cache_dir "$C" --row_index_npy "$C/train_all_rows.npy" --out "$C/text_whiten_trainOnly.npz"
echo "[cifar-v4] whitening done $(date '+%F %T')"
bash scripts/prompt_ablation_cell.sh "$GPU" cifar_v4
echo "[cifar-v4] PIPELINE DONE $(date '+%F %T')"
