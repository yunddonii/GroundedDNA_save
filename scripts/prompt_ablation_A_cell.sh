#!/usr/bin/env bash
# Faithful "A" (strict global-caption-free) recipe × prompt swap, at 18-base (L=3),
# under P0 2-stage + bio-projected mAP@R. Self-contained: uses the champion
# launchers with env overrides + the A flags via EXTRA_ARGS + local-only
# whitening. Does NOT touch the concurrent session's semantic_detail runner.
#
# "A" = champion recipe + the 4 global-slot skips + local-only whitening:
#   already in champion launchers: --text_code_kl_skip_global --text_hash_ntxent_skip_global
#   added here (EXTRA_ARGS):       --xmodal_commit_skip_global --cibhash_dynamic_tau_skip_global
#   local-only whitening:          text_whiten_*_localOnly.npz
#
#   Usage: bash scripts/prompt_ablation_A_cell.sh <GPU> <mscoco_A_v5b|mscoco_A_v4|cifar_A_v1|cifar_A_v4>
set -u

# Exactly one result per tag, or refuse -- `ls | head -1` silently returned a
# concurrently running cell's directory (F08).
source "$(dirname "${BASH_SOURCE[0]}")/lib/result_dir.sh"

GPU="$1"; EXP="$2"
# The provenance-bearing caches (D-A). Only the four EXPs the paper's 15-base
# panel uses are switched; the older prompt variants keep their historical
# caches and are diagnostic-only -- the loader refuses those outright now, which
# is the intended signal rather than a silent downgrade.
CACHE_ROOT="${GDNA_CACHE_ROOT:-/data/yschoi/groundeddna_cache_v6prov}"
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
    CACHE="${CACHE_OVERRIDE:-$CACHE_ROOT/mscoco_clip_tokens}"; QWEN=./cache/mscoco_qwen3_v5b_trainset.jsonl
    WDIR=$CACHE_ROOT/mscoco_clip_tokens_foils; K="${K:-128}"; CIBNT="${CIBNT:-1.5}"; EXTRA=(CELL=Aprompt) ;;
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
    CACHE="${CACHE_OVERRIDE:-$CACHE_ROOT/cifar10_clip_tokens}"; QWEN=./cache/cifar10_qwen_v4.jsonl
    WDIR=$CACHE_ROOT/cifar10_clip_tokens_foils; K="${K:-64}"; CIBNT="${CIBNT:-1.0}"; EXTRA=(CCS=0.1) ;;
  flickr_A_v4)    # A + V4 (Flickr champion prompt) at L=3
    CANON=Flickr25k; SCRIPT=scripts/train_flickr25k_v185_bidirTokenPrune05_clip.sh
    CACHE="${CACHE_OVERRIDE:-$CACHE_ROOT/flickr25k_clip_tokens}"; QWEN=./cache/flickr25k_qwen3_v4_trainset.jsonl
    WDIR=$CACHE_ROOT/flickr25k_clip_tokens_foils; K="${K:-128}"; CIBNT="${CIBNT:-1.0}"; EXTRA=(BIDIR_MODE=legacy) ;;
  nuswide_A_v4)   # A + V4 (NUS champion prompt) at L=3
    CANON=NUSWIDE; SCRIPT=scripts/train_nuswide_v185_sweep_clip.sh
    CACHE="${CACHE_OVERRIDE:-$CACHE_ROOT/nuswide_clip_tokens}"; QWEN=./cache/nuswide_qwen3_v4_trainset.jsonl
    WDIR=$CACHE_ROOT/nuswide_clip_tokens_foils; K="${K:-128}"; CIBNT="${CIBNT:-1.5}"; EXTRA=(CELL=Aprompt) ;;
  *) echo "[promptAblA] unknown EXP=$EXP"; exit 2 ;;
esac

# WHITEN_VARIANT: "_localOnly" (A recipe default, fit on local slots 1-5 only)
# or "" for full-slot whitening, which is what a symmetric slot0 needs.
WV="${WHITEN_VARIANT-_localOnly}"
WOPT="$WDIR/text_whiten_optTrain${WV}.npz"
WTR="$WDIR/text_whiten_trainOnly${WV}.npz"
for f in "$WOPT" "$WTR"; do [ -f "$f" ] || { echo "[promptAblA $EXP] MISSING $f"; exit 3; }; done
BASE="promptAblA_${EXP}${TAG_SUFFIX:-}"

# No-overwrite preflight, the same rule the semantic-detail P0 runner has always
# had. August's runs occupy exactly these deterministic names -- their logs
# still hold E*=4/39/4/19 and their result dirs are the Phase 2 legacy sources,
# so they cannot be moved. Starting a new run in that namespace is what lets a
# stale log or a stale directory be read as this run's. Give the run its own
# namespace with TAG_SUFFIX instead.
declare -a COLLISIONS=()
[ -f "logs/${BASE}_P0val.log" ] && COLLISIONS+=("logs/${BASE}_P0val.log")
shopt -s nullglob
for c in "result/"*"${BASE}_P0val"* "result/"*"${BASE}_P0refit"* \
         "logs/${BASE}_P0refit"*.log; do
    COLLISIONS+=("$c")
done
shopt -u nullglob
if [ "${#COLLISIONS[@]}" -gt 0 ] && [ "${ALLOW_TAG_REUSE:-0}" != "1" ]; then
    echo "[promptAblA $EXP] REFUSING: this tag already has artifacts:" >&2
    printf '    %s\n' "${COLLISIONS[@]}" >&2
    echo "  Set TAG_SUFFIX to give this run its own namespace (e.g." >&2
    echo "  TAG_SUFFIX=_v6), or ALLOW_TAG_REUSE=1 if you really mean to" >&2
    echo "  read and overwrite those." >&2
    exit 8
fi

echo "[promptAblA $EXP] GPU=$GPU CANON=$CANON K=$K CIBNT=$CIBNT L=3 cache=$CACHE @ $(date '+%F %T')"

# ---- stage 1: P0 val ----
# The log path is deterministic, and August's logs are still on disk with
# E*=4/39/4/19 in them. If this child dies before `tee` truncates the file, the
# OLD log is what the parse below reads -- so the child's exit status is checked
# BEFORE the log is trusted, and the log has to be newer than this launch.
S1LOG="logs/${BASE}_P0val.log"
S1_STARTED_AT="$(date +%s)"
if ! env LBU="${LBU:-0.02}" CACHE="$CACHE" EVAL_CACHE="${EVAL_CACHE:-$CACHE}" \
    QWEN="$QWEN" WHITEN_NPZ="$WOPT" K="$K" NUM_CODONS="${NUM_CODONS:-3}" CIBNT="$CIBNT" \
    VAL_RATIO=0.1 VAL_SEED=42 TAG="${BASE}_P0val" EXTRA_ARGS="$A_FLAGS $SKIP" "${EXTRA[@]}" \
    bash "$SCRIPT" "$GPU"; then
    echo "[promptAblA $EXP] ERROR: stage 1 exited nonzero; refusing to read" >&2
    echo "                 $S1LOG, which may predate this run." >&2
    exit 6
fi
if [ ! -f "$S1LOG" ]; then
    echo "[promptAblA $EXP] ERROR: stage 1 wrote no $S1LOG" >&2; exit 6
fi
if [ "$(stat -c %Y "$S1LOG")" -lt "$S1_STARTED_AT" ]; then
    echo "[promptAblA $EXP] ERROR: $S1LOG is older than this launch, so it" >&2
    echo "                 belongs to a previous run. Move it aside." >&2
    exit 6
fi
ESTAR=$(grep -oE "new best mid-eval mAP=[0-9.]+ at epoch [0-9]+" "$S1LOG" 2>/dev/null \
        | grep -oE "epoch [0-9]+$" | grep -oE "[0-9]+" | tail -1)
# Falling back to 59 turned "the selection could not be read" into "the
# selection is the last epoch" -- a different experiment, reported as this one.
if [ -z "${ESTAR:-}" ]; then
    echo "[promptAblA $EXP] ERROR: no 'new best mid-eval mAP ... at epoch N'" >&2
    echo "                 line in $S1LOG; stage 1 did not select an epoch." >&2
    exit 5
fi
echo "[promptAblA $EXP] E*=$ESTAR"

# ---- stage 2: refit ----
if ! env LBU="${LBU:-0.02}" CACHE="$CACHE" EVAL_CACHE="${EVAL_CACHE:-$CACHE}" \
    QWEN="$QWEN" WHITEN_NPZ="$WTR" K="$K" NUM_CODONS="${NUM_CODONS:-3}" CIBNT="$CIBNT" \
    FINAL_EPOCH=1 STOP_EP="$ESTAR" TAG="${BASE}_P0refit_e${ESTAR}" EXTRA_ARGS="$A_FLAGS $SKIP" "${EXTRA[@]}" \
    bash "$SCRIPT" "$GPU"; then
    echo "[promptAblA $EXP] ERROR: stage 2 exited nonzero; the directory that" >&2
    echo "                 matches this tag may be August\'s, not this run\'s." >&2
    exit 7
fi

RD=$(resolve_one_claimed_result_dir "${BASE}_P0refit_e${ESTAR}" extract_db.npz "$CANON") \
    || { echo "[promptAblA $EXP] ERROR refit dir/extract/manifest missing"; exit 4; }
if [ "${NUM_CODONS:-3}" = "4" ]; then GCMIN=0.416; GCMAX=0.584; else GCMIN=0.40; GCMAX=0.60; fi
"$PY" scripts/eval_cell_bioproj.py --dir "$RD" --dataset "$CANON" --K "$K" --gc_min "$GCMIN" --gc_max "$GCMAX"
echo "[promptAblA $EXP] DONE @ $(date '+%F %T')  RD=$RD"
