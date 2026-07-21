#!/usr/bin/env bash
# Run ONE main-table cell end-to-end under the P0 2-stage protocol, then score
# the mandatory bio-projected mAP@R.
#
#   Usage: bash scripts/maintable_cell.sh <GPU> <DS> <K> <L>
#     DS in {flickr,mscoco,nuswide,cifar10};  K in {64,128};  L in {3,4}
#
# Optional env:
#   CIBNT=<w>       override lambda_cibhash_ntxent (phase-2 capacity sweep)
#   TAG_SUFFIX=<s>  appended to the cell base tag (e.g. _cibnt2.0 for a sweep cell)
#
# Stage 1 (val):   --val_split_ratio 0.1, whiten optTrain, select E* on val mAP@R.
# Stage 2 (refit): --final_epoch_eval --stop_after_epoch E*, whiten trainOnly,
#                  train on 100% train, extract at E*.
# Post: eval_cell_bioproj.py -> bio-projected mAP@R (GC window by code length).
#
# Compositional/viz are skipped (EXTRA_ARGS) to keep the grid fast; the reported
# retrieval number + DNA-unique come from the bio-projection scorer.
set -u

GPU="$1"; DS="$2"; K="$3"; L="$4"
SUF="${TAG_SUFFIX:-}"

case "$DS" in
  flickr)
    SCRIPT=scripts/train_flickr25k_v185_bidirTokenPrune05_clip.sh
    CACHE=./cache/flickr25k_clip_v4plus_qwen3_tokens
    CANON=Flickr25k;  DEF_CIBNT=1.0;  EXTRA_ENV=(BIDIR_MODE=legacy) ;;
  mscoco)
    SCRIPT=scripts/train_mscoco_F2_sweep_clip.sh
    CACHE=./cache/mscoco_clip_v5b_tokens
    CANON=MSCOCO;     DEF_CIBNT=1.5;  EXTRA_ENV=(CELL=grid) ;;
  nuswide)
    SCRIPT=scripts/train_nuswide_v185_sweep_clip.sh
    CACHE=./cache/nuswide_clip_tokens
    CANON=NUSWIDE;    DEF_CIBNT=1.5;  EXTRA_ENV=(CELL=grid) ;;
  cifar10)
    SCRIPT=scripts/train_cifar10_v185_bidirTokenPrune05_ccs01_clip.sh
    CACHE=./cache/cifar10_clip
    CANON=CIFAR10;    DEF_CIBNT=1.0;  EXTRA_ENV=() ;;
  *) echo "[cell] unknown DS=$DS"; exit 2 ;;
esac

CIBNT="${CIBNT:-$DEF_CIBNT}"

# CIFAR only: codeword<->codon bijection loss is meaningful iff 4^L >= K.
CCS_ENV=()
if [ "$DS" = "cifar10" ]; then
  cap=$(( 4 ** L ))
  if [ "$cap" -ge "$K" ]; then CCS_ENV=(CCS=0.1); else CCS_ENV=(CCS=0.0); fi
fi

# GC window by code length (18-base -> [8,10]; 24-base -> [10,14]).
if [ "$L" = "4" ]; then GC_MIN=0.416; GC_MAX=0.584; else GC_MIN=0.40; GC_MAX=0.60; fi

WHITEN_OPT="$CACHE/text_whiten_optTrain.npz"
WHITEN_TR="$CACHE/text_whiten_trainOnly.npz"
BASE="${DS}_K${K}_L${L}${SUF}"
PY=/home/yschoi/.conda/envs/dna_hashing/bin/python
SKIP_SLOW="--no-post_eval_compositional --no_visualize"

echo "[cell $BASE] GPU=$GPU CANON=$CANON CIBNT=$CIBNT ${CCS_ENV[*]:-} GC=[$GC_MIN,$GC_MAX] @ $(date '+%F %T')"

# ---------------- Stage 1: P0 validation (find E*) ----------------
S1TAG="${BASE}_P0val"
env CACHE="$CACHE" WHITEN_NPZ="$WHITEN_OPT" K="$K" NUM_CODONS="$L" CIBNT="$CIBNT" \
    VAL_RATIO=0.1 VAL_SEED=42 TAG="$S1TAG" EXTRA_ARGS="$SKIP_SLOW" \
    "${EXTRA_ENV[@]}" "${CCS_ENV[@]}" \
    bash "$SCRIPT" "$GPU"

S1LOG="logs/${S1TAG}.log"
ESTAR=$(grep -oE "new best mid-eval mAP=[0-9.]+ at epoch [0-9]+" "$S1LOG" 2>/dev/null \
        | grep -oE "epoch [0-9]+$" | grep -oE "[0-9]+" | tail -1)
if [ -z "${ESTAR:-}" ]; then
  echo "[cell $BASE] WARNING: could not parse E* from $S1LOG; defaulting to 59"
  ESTAR=59
fi
echo "[cell $BASE] E*=$ESTAR"

# ---------------- Stage 2: refit at E* on 100% train ----------------
S2TAG="${BASE}_P0refit_e${ESTAR}"
env CACHE="$CACHE" WHITEN_NPZ="$WHITEN_TR" K="$K" NUM_CODONS="$L" CIBNT="$CIBNT" \
    FINAL_EPOCH=1 STOP_EP="$ESTAR" TAG="$S2TAG" EXTRA_ARGS="$SKIP_SLOW" \
    "${EXTRA_ENV[@]}" "${CCS_ENV[@]}" \
    bash "$SCRIPT" "$GPU"

# ---------------- Post: bio-projected mAP@R ----------------
RD=$(ls -d result/*"${S2TAG}"* 2>/dev/null | head -1)
if [ -z "${RD:-}" ] || [ ! -f "$RD/extract_db.npz" ]; then
  echo "[cell $BASE] ERROR: refit result dir / extract_db.npz missing (RD=$RD)"; exit 3
fi
"$PY" scripts/eval_cell_bioproj.py --dir "$RD" --dataset "$CANON" --K "$K" \
    --gc_min "$GC_MIN" --gc_max "$GC_MAX"
echo "[cell $BASE] DONE @ $(date '+%F %T')"
