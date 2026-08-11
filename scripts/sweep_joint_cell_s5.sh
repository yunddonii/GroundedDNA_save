#!/usr/bin/env bash
# 5-slot / 15-base variant of sweep_joint_cell.sh (2026-08-11).
#
# The only change is the baseline control set: 36-bit (18-base) dirs ->
# 30-bit (15-base). heldout_codon_decoding chunks a baseline's DNA code
# into N_SLOTS codons, and with N_SLOTS inferred as 5 from our extraction
# an 18-base baseline raises `expected 15 bases, got 18` and the whole
# decoding JSON is never written.
# End-to-end sweep cell: train (P0 2-stage) -> post-process -> held-out codon
# decoding -> one comparison row. Every cell in the 2026-07-31 MSCOCO/CIFAR
# sweep must produce BOTH a retrieval number and an interpretability number
# under identical conditions, so the two are collected here rather than by hand.
#
#   Usage: bash scripts/sweep_joint_cell.sh <GPU> <mscoco_A_v5b|cifar_A_v4> <TAG> "<AUX_ARGS>"
#
# Post-processing differs per dataset and both cases have bitten us before:
#   MSCOCO  train is DISJOINT from the DB, so the decoder needs a purpose-built
#           extract_train.npz (scripts/extract_train_split.py).
#   CIFAR   extractions carry no image_paths at all, so ids must be injected
#           into a `withids/` overlay (scripts/cifar_inject_image_ids.py).
set -u
GPU="$1"; EXP="$2"; TAG="$3"; AUX="${4:-}"
PY=/home/yschoi/.conda/envs/dna_hashing/bin/python
cd /home/yschoi/GroundedDNA
mkdir -p logs docs/sweep_rows

echo "[sweep $TAG] GPU=$GPU EXP=$EXP AUX='$AUX' @ $(date '+%F %T')"

# ---- 1) train -------------------------------------------------------------
env AUX_ARGS="$AUX" TAG_SUFFIX="_$TAG" bash scripts/prompt_ablation_A_cell.sh "$GPU" "$EXP"

RD=$(ls -dt result/*promptAblA_${EXP}_${TAG}_P0refit_* 2>/dev/null | head -1)
if [ -z "${RD:-}" ] || [ ! -f "$RD/cell_result.json" ]; then
  echo "[sweep $TAG] ERROR: no finished refit dir (RD='${RD:-}')"; exit 4
fi
echo "[sweep $TAG] refit dir = $RD"

# ---- 2) dataset-specific decoding prerequisites ---------------------------
B=result_baseline/p0_matrix_seeds42_legacy_cache
case "$EXP" in
  mscoco_A_v5b)
    DS=MSCOCO; OURS="$RD"; MANIFEST=""
    [ -f "$RD/extract_train.npz" ] || "$PY" scripts/extract_train_split.py --config_path "$RD"
    R=result_baseline/260729_mscoco_decodectl_rawbaseEstar
    CTRL_DIRS=("$R/cibhash_mscoco_decodectl" "$R/cimon_mscoco_decodectl" \
               "$R/sdc_mscoco_decodectl"     "$R/oh_mscoco_decodectl" \
               "$R/crovca_mscoco_decodectl")
    ;;
  cifar_A_v4)
    DS=CIFAR10; MANIFEST=dataset/CIFAR10/setting1/train_image_ids.txt
    [ -f "$RD/withids/extract_query.npz" ] || "$PY" scripts/cifar_inject_image_ids.py --result_dir "$RD"
    OURS="$RD/withids"
    CTRL_DIRS=($(ls -d $B/u0_cibhash_cifar10_30b_seed42/attempt_*/*/*_dnaeval    | head -1) \
               $(ls -d $B/u0_cimon_cifar10_30b_seed42/attempt_*/*/*_dnaeval      | head -1) \
               $(ls -d $B/u0_sdc-paper_cifar10_30b_seed42/attempt_*/*/*_dnaeval  | head -1) \
               $(ls -d $B/u0_oh_cifar10_30b_seed42/attempt_*/*/*_dnaeval         | head -1) \
               $(ls -d result_baseline/260722/crovca_cifar10_30b_P0refit_seed42_*_dnaeval | head -1))
    ;;
  nuswide_A_v4)
    DS=NUSWIDE; OURS="$RD"; MANIFEST=dataset/NUSWIDE/setting1/train_10500.txt
    CTRL_DIRS=($(ls -d $B/u0_cibhash_nuswide_30b_seed42/attempt_*/*/*_dnaeval   | head -1) \
               $(ls -d $B/u0_cimon_nuswide_30b_seed42/attempt_*/*/*_dnaeval     | head -1) \
               $(ls -d $B/u0_sdc-paper_nuswide_30b_seed42/attempt_*/*/*_dnaeval | head -1) \
               $(ls -d $B/u0_oh_nuswide_30b_seed42/attempt_*/*/*_dnaeval        | head -1) \
               $(ls -d result_baseline/260722/crovca_nuswide_30b_P0refit_seed42_*_dnaeval | head -1))
    ;;
  flickr_A_v4)
    DS=Flickr25k; OURS="$RD"; MANIFEST=dataset/Flickr25k/setting1/train.txt
    CTRL_DIRS=($(ls -d $B/u0_cibhash_flickr25k_30b_seed42/attempt_*/*/*_dnaeval   | head -1) \
               $(ls -d $B/u0_cimon_flickr25k_30b_seed42/attempt_*/*/*_dnaeval     | head -1) \
               $(ls -d $B/u0_sdc-paper_flickr25k_30b_seed42/attempt_*/*/*_dnaeval | head -1) \
               $(ls -d $B/u0_oh_flickr25k_30b_seed42/attempt_*/*/*_dnaeval        | head -1) \
               $(ls -d result_baseline/260722/crovca_flickr25k_30b_P0refit_seed42_*_dnaeval | head -1))
    ;;
  *) echo "[sweep $TAG] unknown EXP=$EXP"; exit 2 ;;
esac

# ---- 3) held-out codon decoding vs raw-base-E* controls -------------------
DEC=docs/heldout_decoding_${DS,,}_${TAG}.json
"$PY" scripts/heldout_codon_decoding.py \
  --ours_dir "$OURS" --dataset "$DS" --bio_project \
  ${MANIFEST:+--train_manifest "$MANIFEST"} \
  --baseline_dirs "${CTRL_DIRS[@]}" \
  --baseline_names CIBHash CIMON SDC OH CroVCA \
  --out "$DEC" 2>&1 | grep -E "mAP=|rror" || true

# ---- 4) one comparison row ------------------------------------------------
"$PY" scripts/sweep_summarize_cell.py --dir "$RD" --tag "$TAG" \
  --decoding_json "$DEC" --out "docs/sweep_rows/${DS,,}_${TAG}.json"

echo "[sweep $TAG] DONE @ $(date '+%F %T')"
