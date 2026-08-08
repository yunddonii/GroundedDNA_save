#!/usr/bin/env bash
# Fixed-E* variant of prompt_ablation_A_cell.sh.
#
# The stock cell runs stage-1, parses E* out of the mid-eval log, and refits to
# it. E* is chosen on `eval_mAP_at_R` ALONE (train_siglip2.py:1185), i.e. purely
# on retrieval -- yet the paper leads with the interpretable compositional code.
# On Flickr25k the two peaks sit in different places:
#
#   metric                      peak epoch   value at e3   value at peak
#   eval_mAP                        3          .7805          .7805
#   unique code ratio               6          .5904          .6529
#   base normalized entropy        37          .9588          .9952
#   per-codebook unique ratio      47         .01115         .01419
#
# so the current protocol buys retrieval with the headline contribution. This
# variant takes E* from $ESTAR_FIXED and SKIPS stage-1 (stage-2 only needs E*;
# it uses the train-only whitening matrix either way), making an E*-sweep cheap:
# one refit per candidate stopping point, each scored on the 4 axes the paper
# reports.
#
# A NEW FILE rather than a flag on the original: bash reads a script
# incrementally, so editing prompt_ablation_A_cell.sh while cells are running it
# corrupts those runs (observed earlier in this project).
#
# Usage: ESTAR_FIXED=6 bash scripts/prompt_ablation_A_cell_fixedE.sh <GPU> <EXP>
set -u
GPU="$1"; EXP="$2"
PY=/home/yschoi/.conda/envs/dna_hashing/bin/python
# A_SKIPS: the two A-recipe global-slot skips added here (the other two,
# --text_code_kl_skip_global / --text_hash_ntxent_skip_global, live in the
# champion launchers and are toggled there via GLOBAL_SKIPS). Set A_SKIPS=""
# AND GLOBAL_SKIPS="" to train slot0 exactly like the local slots.
A_FLAGS="${A_SKIPS-"--xmodal_commit_skip_global --cibhash_dynamic_tau_skip_global"} ${AUX_ARGS:-}"
SKIP="--no-post_eval_compositional${VIZ:+}"
[ "${VIZ:-1}" = "0" ] && SKIP="$SKIP --no_visualize"

case "$EXP" in
  mscoco_A_v5b)   # A + champion prompt (V5b) at L=3
    CANON=MSCOCO; SCRIPT=scripts/train_mscoco_F2_sweep_clip.sh
    CACHE="${CACHE_OVERRIDE:-./cache/mscoco_clip_v5b_tokens}"; QWEN=./cache/mscoco_qwen3_v5b_trainset.jsonl
    WDIR=./cache/mscoco_clip_v5b_tokens_foils; K="${K:-128}"; CIBNT="${CIBNT:-1.5}"; EXTRA=(CELL=Aprompt) ;;
  mscoco_A_v4)    # A + V4 prompt at L=3
    CANON=MSCOCO; SCRIPT=scripts/train_mscoco_F2_sweep_clip.sh
    CACHE=./cache/mscoco_clip_v4plus_tokens; QWEN=./cache/mscoco_qwen3_v4_trainset.jsonl
    WDIR=./cache/mscoco_clip_v4plus_tokens; K="${K:-128}"; CIBNT="${CIBNT:-1.5}"; EXTRA=(CELL=Aprompt) ;;
  cifar_A_v1)     # A + champion prompt (V1) at L=3
    CANON=CIFAR10; SCRIPT=scripts/train_cifar10_v185_bidirTokenPrune05_ccs01_clip.sh
    CACHE=./cache/cifar10_clip; QWEN=./cache/cifar10_qwen.jsonl
    WDIR=./cache/cifar10_clip_foils; K="${K:-64}"; CIBNT="${CIBNT:-1.0}"; EXTRA=(CCS=0.1) ;;
  cifar_A_v4)     # A + V4 prompt at L=3
    CANON=CIFAR10; SCRIPT=scripts/train_cifar10_v185_bidirTokenPrune05_ccs01_clip.sh
    CACHE=./cache/cifar10_clip_v4_tokens; QWEN=./cache/cifar10_qwen_v4.jsonl
    WDIR=./cache/cifar10_clip_v4_tokens; K="${K:-64}"; CIBNT="${CIBNT:-1.0}"; EXTRA=(CCS=0.1) ;;
  flickr_A_v4)    # A + V4 (Flickr champion prompt) at L=3
    CANON=Flickr25k; SCRIPT=scripts/train_flickr25k_v185_bidirTokenPrune05_clip.sh
    CACHE=./cache/flickr25k_clip_v4plus_qwen3_tokens; QWEN=./cache/flickr25k_qwen3_v4_trainset.jsonl
    WDIR=./cache/flickr25k_clip_v4plus_qwen3_tokens_foils; K="${K:-128}"; CIBNT="${CIBNT:-1.0}"; EXTRA=(BIDIR_MODE=legacy) ;;
  nuswide_A_v4)   # A + V4 (NUS champion prompt) at L=3
    CANON=NUSWIDE; SCRIPT=scripts/train_nuswide_v185_sweep_clip.sh
    CACHE=./cache/nuswide_clip_tokens; QWEN=./cache/nuswide_qwen3_v4_trainset.jsonl
    WDIR=./cache/nuswide_clip_tokens_foils; K="${K:-128}"; CIBNT="${CIBNT:-1.5}"; EXTRA=(CELL=Aprompt) ;;
  *) echo "[promptAblA] unknown EXP=$EXP"; exit 2 ;;
esac

# WHITEN_VARIANT: "_localOnly" (A recipe default, fit on local slots 1-5 only)
# or "" for full-slot whitening, which is what a symmetric slot0 needs.
WV="${WHITEN_VARIANT-_localOnly}"
WOPT="$WDIR/text_whiten_optTrain${WV}.npz"
WTR="$WDIR/text_whiten_trainOnly${WV}.npz"
for f in "$WOPT" "$WTR"; do [ -f "$f" ] || { echo "[promptAblA $EXP] MISSING $f"; exit 3; }; done
BASE="promptAblA_${EXP}${TAG_SUFFIX:-}"
echo "[promptAblA $EXP] GPU=$GPU CANON=$CANON K=$K CIBNT=$CIBNT L=3 cache=$CACHE @ $(date '+%F %T')"

# ---- stage 1: SKIPPED, E* supplied by the caller ----
ESTAR="${ESTAR_FIXED:?ESTAR_FIXED must be set for the fixed-E* variant}"
case "$ESTAR" in ''|*[!0-9]*) echo "[fixE] ESTAR_FIXED must be an integer, got '$ESTAR'"; exit 2 ;; esac
echo "[fixE $EXP] E*=$ESTAR (fixed; stage-1 not run)"

# ---- stage 2: refit ----
env LBU="${LBU:-0.02}" CACHE="$CACHE" QWEN="$QWEN" WHITEN_NPZ="$WTR" K="$K" NUM_CODONS="${NUM_CODONS:-3}" CIBNT="$CIBNT" \
    FINAL_EPOCH=1 STOP_EP="$ESTAR" TAG="${BASE}_P0refit_e${ESTAR}" EXTRA_ARGS="$A_FLAGS $SKIP" "${EXTRA[@]}" \
    bash "$SCRIPT" "$GPU"

RD=$(ls -d result/*"${BASE}_P0refit_e${ESTAR}"* 2>/dev/null | head -1)
[ -n "${RD:-}" ] && [ -f "$RD/extract_db.npz" ] || { echo "[promptAblA $EXP] ERROR refit dir/extract missing (RD=$RD)"; exit 4; }
if [ "${NUM_CODONS:-3}" = "4" ]; then GCMIN=0.416; GCMAX=0.584; else GCMIN=0.40; GCMAX=0.60; fi
"$PY" scripts/eval_cell_bioproj.py --dir "$RD" --dataset "$CANON" --K "$K" --gc_min "$GCMIN" --gc_max "$GCMAX"
echo "[promptAblA $EXP] DONE @ $(date '+%F %T')  RD=$RD"
