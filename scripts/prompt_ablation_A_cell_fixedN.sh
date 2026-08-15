#!/usr/bin/env bash
# Single-stage, fixed-epoch variant of prompt_ablation_A_cell.sh.
#
# Protocol change requested 2026-08-09: drop the P0 two-stage split and train
# ONCE to a fixed epoch, monitoring the FULL official test split every epoch --
# the conventional deep-hashing setup, and what this repo did before P0. There
# is no stage-2 refit; the checkpoint at the fixed epoch IS the reported model.
#
# Justification for the short horizon: with a frozen CLIP backbone this is now
# standard. CroVCA (CVPRW'26), one of our own baselines, trains "for only 5
# epochs on a single GPU" and sells that as a contribution; CLIP Multi-modal
# Hashing reports test mAP flat after 10 epochs while the loss keeps falling to
# 45. Our E* has been landing at 4-6 for exactly the same reason.
#
# LEAKAGE, stated plainly: picking the fixed epoch by looking at test curves
# leaks one scalar per dataset. To keep it to that, choose N from the seed-42
# curve ONLY and apply the same N unchanged to seeds 43/44.
#
# Usage:
#   FIXED_N=<n> bash scripts/prompt_ablation_A_cell_fixedN.sh <GPU> <EXP>
#   FIXED_N=59 ... EVERY=1   -> the curve run used to CHOOSE N
#
# A separate file, not a flag: bash reads scripts incrementally, so editing the
# original while cells are running it corrupts those runs.
set -u

# Exactly one result per tag, or refuse -- `ls | head -1` silently returned a
# concurrently running cell's directory (F08).
source "$(dirname "${BASH_SOURCE[0]}")/lib/result_dir.sh"

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

# ---- single stage: no val split, mid-eval on the official test split ------
NEPOCH="${FIXED_N:?FIXED_N must be set}"
case "$NEPOCH" in ''|*[!0-9]*) echo "[fixN] FIXED_N must be an integer, got '$NEPOCH'"; exit 2 ;; esac
ESTAR="$NEPOCH"
echo "[fixN $EXP] fixed epoch = $NEPOCH, val_split_ratio=0 (test-monitored), no refit"

# is_p0_refit() fires on (stop_after_epoch AND final_epoch_eval) and then
# ISOLATES the official test -- no per-epoch test mid-eval at all. The two
# modes therefore need different flags:
#   CURVE=1 : stop_after_epoch only -> legacy path with test mid-eval every
#             epoch. Used to CHOOSE the fixed N; the end-of-run best-ckpt swap
#             is irrelevant because only the per-epoch curve is consumed.
#   default : + final_epoch_eval -> keeps the epoch-N checkpoint (no best-ckpt
#             swap) and runs the single final extraction. This is the reported
#             model; mid-eval being off is fine once N is fixed.
if [ -n "${CURVE:-}" ]; then FE=""; else FE="1"; fi
env LBU="${LBU:-0.02}" CACHE="$CACHE" QWEN="$QWEN" WHITEN_NPZ="$WTR" K="$K" NUM_CODONS="${NUM_CODONS:-3}" CIBNT="$CIBNT" \
    ${FE:+FINAL_EPOCH=1} STOP_EP="$NEPOCH" TAG="${BASE}_P0refit_e${NEPOCH}" EXTRA_ARGS="$A_FLAGS $SKIP --eval_every ${EVERY:-1}" "${EXTRA[@]}" \
    bash "$SCRIPT" "$GPU"

RD=$(resolve_one_result_dir ""${BASE}_P0refit_e${ESTAR}"")
[ -n "${RD:-}" ] && [ -f "$RD/extract_db.npz" ] || { echo "[promptAblA $EXP] ERROR refit dir/extract missing (RD=$RD)"; exit 4; }
if [ "${NUM_CODONS:-3}" = "4" ]; then GCMIN=0.416; GCMAX=0.584; else GCMIN=0.40; GCMAX=0.60; fi
"$PY" scripts/eval_cell_bioproj.py --dir "$RD" --dataset "$CANON" --K "$K" --gc_min "$GCMIN" --gc_max "$GCMAX"
echo "[promptAblA $EXP] DONE @ $(date '+%F %T')  RD=$RD"
