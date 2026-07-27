#!/usr/bin/env bash
# Prompt-schema ablation: run one dataset's CHAMPION recipe with ONLY the text
# caption prompt swapped (via an alternate text-feature cache), under the P0
# 2-stage protocol + bio-projected mAP@R. Single delta = the VLM prompt schema.
#
#   Usage: bash scripts/prompt_ablation_cell.sh <GPU> <flickr_v5b|mscoco_v4|cifar_v4>
#
# Champions: Flickr V4 / MSCOCO V5b / CIFAR V1. This swaps to the OTHER prompt to
# test "dataset-appropriate prompting". Eval cache (visual) stays the champion's,
# so the ONLY difference from the champion is the training-time text schema.
set -u
GPU="$1"; EXP="$2"
PY=/home/yschoi/.conda/envs/dna_hashing/bin/python
SKIP="--no-post_eval_compositional${VIZ:+}"
[ "${VIZ:-1}" = "0" ] && SKIP="$SKIP --no_visualize"

case "$EXP" in
  flickr_v5b)   # champion is V4; test V5b (strict disjoint-vocab)
    CANON=Flickr25k; SCRIPT=scripts/train_flickr25k_v185_bidirTokenPrune05_clip.sh
    CACHE=./cache/flickr25k_clip_v5b_tokens; QWEN=./cache/flickr25k_qwen3_v5b_trainset.jsonl
    K=128; L=3; CIBNT=1.0; EXTRA=(BIDIR_MODE=legacy) ;;
  mscoco_v4)    # champion is V5b; test V4 (loose evidence-axis)
    CANON=MSCOCO; SCRIPT=scripts/train_mscoco_F2_sweep_clip.sh
    CACHE=./cache/mscoco_clip_v4plus_tokens; QWEN=./cache/mscoco_qwen3_v4_trainset.jsonl
    K=128; L=3; CIBNT=1.5; EXTRA=(CELL=promptAbl) ;;
  cifar_v4)     # champion is V1; test V4 (evidence-axis) -- needs cifar v4 cache
    CANON=CIFAR10; SCRIPT=scripts/train_cifar10_v185_bidirTokenPrune05_ccs01_clip.sh
    CACHE=./cache/cifar10_clip_v4_tokens; QWEN=./cache/cifar10_qwen_v4.jsonl
    K=64; L=3; CIBNT=1.0; EXTRA=(CCS=0.1) ;;
  *) echo "[promptAbl] unknown EXP=$EXP"; exit 2 ;;
esac

WOPT="$CACHE/text_whiten_optTrain.npz"; WTR="$CACHE/text_whiten_trainOnly.npz"
if [ ! -f "$WOPT" ] || [ ! -f "$WTR" ]; then
  echo "[promptAbl $EXP] ERROR: whitening npz missing in $CACHE (build first)"; exit 3
fi
BASE="promptAbl_${EXP}"
echo "[promptAbl $EXP] GPU=$GPU CANON=$CANON K=$K CIBNT=$CIBNT cache=$CACHE @ $(date '+%F %T')"

# ---- stage 1: P0 val ----
env CACHE="$CACHE" QWEN="$QWEN" WHITEN_NPZ="$WOPT" K="$K" NUM_CODONS="$L" CIBNT="$CIBNT" \
    VAL_RATIO=0.1 VAL_SEED=42 TAG="${BASE}_P0val" EXTRA_ARGS="$SKIP" "${EXTRA[@]}" \
    bash "$SCRIPT" "$GPU"
S1LOG="logs/${BASE}_P0val.log"
ESTAR=$(grep -oE "new best mid-eval mAP=[0-9.]+ at epoch [0-9]+" "$S1LOG" 2>/dev/null \
        | grep -oE "epoch [0-9]+$" | grep -oE "[0-9]+" | tail -1)
[ -z "${ESTAR:-}" ] && { echo "[promptAbl $EXP] WARN: E* parse failed -> 59"; ESTAR=59; }
echo "[promptAbl $EXP] E*=$ESTAR"

# ---- stage 2: refit ----
env CACHE="$CACHE" QWEN="$QWEN" WHITEN_NPZ="$WTR" K="$K" NUM_CODONS="$L" CIBNT="$CIBNT" \
    FINAL_EPOCH=1 STOP_EP="$ESTAR" TAG="${BASE}_P0refit_e${ESTAR}" EXTRA_ARGS="$SKIP" "${EXTRA[@]}" \
    bash "$SCRIPT" "$GPU"

RD=$(ls -d result/*"${BASE}_P0refit_e${ESTAR}"* 2>/dev/null | head -1)
if [ -z "${RD:-}" ] || [ ! -f "$RD/extract_db.npz" ]; then
  echo "[promptAbl $EXP] ERROR: refit dir/extract missing (RD=$RD)"; exit 4
fi
"$PY" scripts/eval_cell_bioproj.py --dir "$RD" --dataset "$CANON" --K "$K" --gc_min 0.40 --gc_max 0.60
echo "[promptAbl $EXP] DONE @ $(date '+%F %T')"
