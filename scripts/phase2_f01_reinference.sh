#!/usr/bin/env bash
# Phase 2: F01-only diagnostic re-inference of the seed-42 N candidates.
#
# The audit's Phase 2 asks what the epoch-0 extraction bug (F01) actually cost.
# `_current_epoch` is a plain int, absent from the state dict, so every
# standalone/final extraction ran the router at the INITIAL Sinkhorn epsilon
# rather than the annealed value the weights were trained with.
#
# This re-infers each candidate checkpoint with the epoch restored, into a
# SEPARATE diagnostic root. The legacy artefacts are never touched: the
# checkpoint and config are copied out, and nothing is written back.
#
# This is a diagnostic of the defect's impact. It does NOT replace the
# paper-valid N selection, which is F02/F03's train-only protocol in Phase 3.
#
# Usage:
#   scripts/phase2_f01_reinference.sh <gpu> <cell> [<cell> ...]
#   scripts/phase2_f01_reinference.sh 0 cifar10/4
set -euo pipefail

PY="${PY:-/home/yschoi/.conda/envs/dna_hashing/bin/python}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${PHASE2_ROOT:-$REPO/result_diagnostic/phase2_F01_only}"

GPU="${1:?usage: phase2_f01_reinference.sh <gpu> <dataset/N> ...}"
shift

# dataset/N -> legacy run directory. Verified against args.txt before use.
legacy_dir() {
    case "$1" in
    cifar10/4)    echo "result/260811+cifar10_setting1_promptAblA_cifar_A_v4_P0refit_e4+bs+64+e+5+proj_lr+0.001" ;;
    cifar10/9)    echo "result/260811+cifar10_setting1_promptAblA_cifar_A_v4_P0refit_e9+bs+64+e+10+proj_lr+0.001" ;;
    cifar10/19)   echo "result/260811+cifar10_setting1_promptAblA_cifar_A_v4_P0refit_e19+bs+64+e+20+proj_lr+0.001" ;;
    cifar10/39)   echo "result/260812+cifar10_setting1_promptAblA_cifar_A_v4_P0refit_e39+bs+64+e+40+proj_lr+0.001" ;;
    flickr25k/4)  echo "result/260811+flickr25k_setting1_promptAblA_flickr_A_v4_P0refit_e4+bs+64+e+5+proj_lr+0.001" ;;
    flickr25k/9)  echo "result/260811+flickr25k_setting1_promptAblA_flickr_A_v4_P0refit_e9+bs+64+e+10+proj_lr+0.001" ;;
    flickr25k/19) echo "result/260812+flickr25k_setting1_promptAblA_flickr_A_v4_P0refit_e19+bs+64+e+20+proj_lr+0.001" ;;
    nuswide/4)    echo "result/260811+nuswide_setting1_promptAblA_nuswide_A_v4_P0refit_e4+bs+64+e+5+proj_lr+0.001" ;;
    nuswide/9)    echo "result/260811+nuswide_setting1_promptAblA_nuswide_A_v4_P0refit_e9+bs+64+e+10+proj_lr+0.001" ;;
    nuswide/19)   echo "result/260811+nuswide_setting1_promptAblA_nuswide_A_v4_P0refit_e19+bs+64+e+20+proj_lr+0.001" ;;
    nuswide/39)   echo "result/260812+nuswide_setting1_promptAblA_nuswide_A_v4_P0refit_e39+bs+64+e+40+proj_lr+0.001" ;;
    mscoco/4)     echo "result/260811+mscoco_setting1_promptAblA_mscoco_A_v5b_P0refit_e4+bs+64+e+5+proj_lr+0.001" ;;
    mscoco/9)     echo "result/260811+mscoco_setting1_promptAblA_mscoco_A_v5b_P0refit_e9+bs+64+e+10+proj_lr+0.001" ;;
    mscoco/19)    echo "result/260811+mscoco_setting1_promptAblA_mscoco_A_v5b_P0refit_e19+bs+64+e+20+proj_lr+0.001" ;;
    mscoco/39)    echo "result/260812+mscoco_setting1_promptAblA_mscoco_A_v5b_s42_P0refit_e39+bs+64+e+40+proj_lr+0.001" ;;
    *) return 1 ;;
    esac
}

canon() {
    case "$1" in
    cifar10) echo CIFAR10 ;; flickr25k) echo Flickr25k ;;
    nuswide) echo NUSWIDE ;; mscoco) echo MSCOCO ;;
    *) return 1 ;;
    esac
}

for CELL in "$@"; do
    DS="${CELL%%/*}"; N="${CELL##*/}"
    SRC="$REPO/$(legacy_dir "$CELL")" || { echo "unknown cell $CELL" >&2; exit 2; }
    CANON="$(canon "$DS")"
    OUT="$ROOT/${DS}_N${N}"

    [[ -d "$SRC" ]] || { echo "missing legacy dir: $SRC" >&2; exit 2; }
    # A distinct output dir per cell. The 3-seed directory collision of
    # 2026-08-12 came from every cell resolving to one path, so this is
    # asserted rather than assumed.
    # A DB NPZ alone is not completion: the 15 original cells had exactly that
    # and no manifests at all, so re-running this script backfilled nothing.
    # Both manifests must exist for a cell to count as done.
    if [[ -e "$OUT/extraction_manifest_db.json" \
       && -e "$OUT/extraction_manifest_query.json" ]]; then
        echo "[phase2] $CELL already complete at $OUT; skipping" >&2
        continue
    fi
    if [[ -e "$OUT/extract_db.npz" ]]; then
        echo "[phase2] $CELL has extractions but no manifests at $OUT." >&2
        echo "         Run scripts/backfill_phase2_manifests.py, or delete the" >&2
        echo "         directory to re-infer from scratch." >&2
        exit 4
    fi
    mkdir -p "$OUT"
    cp "$SRC/config.pt" "$SRC/model_state_dict.pth" "$OUT/"

    STOP=$("$PY" "$REPO/scripts/_phase2_read_arg.py" "$SRC/args.txt" stop_after_epoch)
    [[ "$STOP" == "$N" ]] || {
        echo "[phase2] $CELL: args.txt stop_after_epoch=$STOP != N=$N" >&2; exit 3; }

    # The slot count is baked in at IMPORT time, so it must be exported before
    # python starts; passing --num_semantic_parts alone aborts with "the module
    # was imported with GDNA_NUM_SEMANTIC_PARTS -> 6". Read it from the cell's
    # own args.txt rather than hard-coding, so a mismatch is impossible.
    M=$("$PY" "$REPO/scripts/_phase2_read_arg.py" "$SRC/args.txt" num_semantic_parts)
    [[ "$M" == "5" ]] || {
        echo "[phase2] $CELL: num_semantic_parts=$M, expected the paper's 5" >&2
        exit 3; }

    echo "[phase2] $CELL -> $OUT (inference_epoch=$N, slots=$M, gpu=$GPU)"
    CUDA_VISIBLE_DEVICES="$GPU" GDNA_NUM_SEMANTIC_PARTS="$M" \
        "$PY" "$REPO/extraction_siglip2.py" \
        --config_path "$OUT" --inference_epoch "$N" 2>&1 | tee "$OUT/extract.log"

    K=$("$PY" "$REPO/scripts/_phase2_read_arg.py" "$SRC/args.txt" codebook_size)
    CUDA_VISIBLE_DEVICES="$GPU" GDNA_NUM_SEMANTIC_PARTS="$M" \
        "$PY" "$REPO/scripts/eval_cell_bioproj.py" \
        --dir "$OUT" --dataset "$CANON" --K "$K" 2>&1 | tee -a "$OUT/extract.log"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$REPO/scripts/pairwise_nmi.py" \
        --results "$OUT" --out "$OUT/pairwise_nmi.json" \
        2>&1 | tee -a "$OUT/extract.log"
done
