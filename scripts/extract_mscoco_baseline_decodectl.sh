#!/usr/bin/env bash
# Extract train+query splits for the MSCOCO modern-U0 baselines so they can act
# as raw-base-E* chunk controls in scripts/heldout_codon_decoding.py.
#
# Why: the P0-matrix baseline dirs ship only extract_{db,query}.npz. For
# Flickr/NUS/CIFAR the decoder slices train rows out of the DB by basename, but
# MSCOCO's train split is DISJOINT from its DB, so that fallback raises
# KeyError. These methods therefore need a purpose-built train extraction, the
# same way docs' 260719 `*_mscoco_clip_decodectl` dirs were built for the
# classic three.
#
# Runs sequentially on one GPU (extraction reads a frozen feature cache, so it
# is light) to avoid the file-descriptor / checkpoint-write contention that
# killed two concurrent runs on 2026-07-28.
#
#   Usage: bash scripts/extract_mscoco_baseline_decodectl.sh <GPU>
set -u
GPU="${1:-4}"
PY=/home/yschoi/.conda/envs/dna_hashing/bin/python
CACHE=/data/yschoi/dataset/deephashing/cache/mscoco_clip_v5b
OUTROOT=result_baseline/260729_mscoco_decodectl_rawbaseEstar
mkdir -p "$OUTROOT"

# name : glob for the P0refit checkpoint dir (E* already raw-base-selected)
declare -A W=(
  [cibhash]="params_baseline/p0_matrix_seeds42_legacy_cache/u0_cibhash_mscoco_36b_seed42/attempt_*/*/cibhash_mscoco_36b_P0refit_seed42_*_e*"
  [cimon]="params_baseline/p0_matrix_seeds42_legacy_cache/u0_cimon_mscoco_36b_seed42/attempt_*/*/cimon_mscoco_36b_P0refit_seed42_*_e*"
  [sdc]="params_baseline/p0_matrix_seeds42_legacy_cache/u0_sdc-paper_mscoco_36b_seed42/attempt_*/*/sdc_paper_mscoco_36b_P0refit_seed42_*_e*"
  [oh]="params_baseline/p0_matrix_seeds42_legacy_cache/u0_oh_mscoco_36b_seed42/attempt_*/*/oh_mscoco_36b_P0refit_seed42_*_e*"
  [crovca]="params_baseline/260722/crovca_mscoco_36b_P0refit_seed42_*_e*"
)

for name in cibhash cimon sdc oh crovca; do
  d=$(ls -d ${W[$name]} 2>/dev/null | grep -v _dnaeval | head -1)
  if [ -z "${d:-}" ]; then echo "[$name] SKIP: no checkpoint dir"; continue; fi
  ck=$(ls "$d"/epoch_*.pth 2>/dev/null | sort -V | tail -1)
  if [ -z "${ck:-}" ]; then echo "[$name] SKIP: no epoch_*.pth in $d"; continue; fi
  out="$OUTROOT/${name}_mscoco_decodectl"
  if [ -f "$out/extract_train.npz" ] && [ -f "$out/extract_query.npz" ]; then
    echo "[$name] already done -> $out"; continue
  fi
  echo "[$name] $(basename "$ck") -> $out  @ $(date '+%T')"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/baseline_extract_splits.py \
    --weights "$ck" --dataset MSCOCO --cache_dir "$CACHE" \
    --out "$out" --splits train query --batch_size 256 --num_workers 2 \
    --device cuda:0 2>&1 | tail -3
done
echo "[decodectl] DONE @ $(date '+%T')  root=$OUTROOT"
