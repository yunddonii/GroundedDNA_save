#!/usr/bin/env bash
# Re-score the two inventory-bound Phase-2 F01 roots with one committed CPU
# evaluator, then issue the exact-15/15 diagnostic report receipt.
set -euo pipefail

SCRIPT=$(realpath "${BASH_SOURCE[0]}")
REPO=$(dirname "$(dirname "$SCRIPT")")
DEFAULT_PY=/home/yschoi/.conda/envs/dna_hashing/bin/python

if [[ "${1:-}" == "--run-cell" ]]; then
    [[ "$#" -eq 8 ]] || exit 64
    cd "$2"
    PY=$3; side=$4; cell=$5; dataset=$6; codebook_size=$7; log=$8
    [[ ! -e "$log" && ! -L "$log" ]] || {
        echo "refusing existing log $log" >&2; exit 65; }
    umask 077
    set -o noclobber
    exec >"$log" 2>&1
    set +o noclobber
    echo "[phase2] side=$side cell=$cell dataset=$dataset K=$codebook_size"
    env -u PYTHONPATH CUDA_VISIBLE_DEVICES='' GDNA_NUM_SEMANTIC_PARTS=5 \
        OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
        "$PY" evaluation_siglip2.py --extraction_path "$cell" \
        --distance_mode base --codebook_size "$codebook_size" \
        --dataset "$dataset" --no-bio_project --query_chunk_size 64
    env -u PYTHONPATH CUDA_VISIBLE_DEVICES='' GDNA_NUM_SEMANTIC_PARTS=5 \
        OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
        "$PY" scripts/eval_cell_bioproj.py --dir "$cell" \
        --dataset "$dataset" --K "$codebook_size" --allow-backfilled
    env -u PYTHONPATH CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 \
        MKL_NUM_THREADS=4 "$PY" scripts/pairwise_nmi.py \
        --results "$cell" --allow-backfilled
    env -u PYTHONPATH CUDA_VISIBLE_DEVICES='' "$PY" \
        scripts/seal_cell_analysis.py --dir "$cell" --allow-backfilled
    state=$(env -u PYTHONPATH CUDA_VISIBLE_DEVICES='' "$PY" \
        scripts/_phase2_cell_state.py "$cell" \
        --allow-backfilled --require-valid)
    [[ "$state" == complete ]] || {
        echo "cell did not finish complete: $state" >&2; exit 66; }
    echo "[phase2] COMPLETE $side/$(basename "$cell")"
    exit 0
fi

usage() {
    echo "usage: $0 --fixed-root PATH --legacy-root PATH --log-root PATH --report-root PATH [--python PATH] [--jobs N]" >&2
    exit 64
}

FIXED_ROOT= LEGACY_ROOT= LOG_ROOT= REPORT_ROOT=
PY=$DEFAULT_PY
JOBS=4
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --fixed-root) FIXED_ROOT=${2:-}; shift 2 ;;
        --legacy-root) LEGACY_ROOT=${2:-}; shift 2 ;;
        --log-root) LOG_ROOT=${2:-}; shift 2 ;;
        --report-root) REPORT_ROOT=${2:-}; shift 2 ;;
        --python) PY=${2:-}; shift 2 ;;
        --jobs) JOBS=${2:-}; shift 2 ;;
        *) usage ;;
    esac
done
[[ -n "$FIXED_ROOT" && -n "$LEGACY_ROOT" && -n "$LOG_ROOT" \
   && -n "$REPORT_ROOT" ]] || usage
[[ "$JOBS" =~ ^[1-9][0-9]*$ && "$JOBS" -le 8 ]] || usage
[[ -x "$PY" ]] || { echo "refusing non-executable Python: $PY" >&2; exit 2; }

cd "$REPO"
[[ ! -L "$FIXED_ROOT" && ! -L "$LEGACY_ROOT" ]] || {
    echo "input roots must not be symlinks" >&2; exit 2; }
FIXED_ROOT=$(realpath "$FIXED_ROOT")
LEGACY_ROOT=$(realpath "$LEGACY_ROOT")
LOG_ROOT=$(realpath -m "$LOG_ROOT")
REPORT_ROOT=$(realpath -m "$REPORT_ROOT")
[[ -d "$FIXED_ROOT" && -d "$LEGACY_ROOT" ]] || {
    echo "both input roots must exist" >&2; exit 2; }
[[ ! -e "$LOG_ROOT" && ! -L "$LOG_ROOT" \
   && ! -e "$REPORT_ROOT" && ! -L "$REPORT_ROOT" ]] || {
    echo "log/report roots must be fresh and absent" >&2; exit 2; }
PATHS=("$FIXED_ROOT" "$LEGACY_ROOT" "$LOG_ROOT" "$REPORT_ROOT")
for ((i=0; i<${#PATHS[@]}; i++)); do
    for ((j=i+1; j<${#PATHS[@]}; j++)); do
        if [[ "${PATHS[$i]}" == "${PATHS[$j]}" \
           || "${PATHS[$i]}" == "${PATHS[$j]}/"* \
           || "${PATHS[$j]}" == "${PATHS[$i]}/"* ]]; then
            echo "input/log/report roots must be disjoint" >&2; exit 2
        fi
    done
done
[[ "$LOG_ROOT" != "$REPO" && "$LOG_ROOT" != "$REPO/"* \
   && "$REPORT_ROOT" != "$REPO" && "$REPORT_ROOT" != "$REPO/"* ]] || {
    echo "log/report roots must be outside the clean source tree" >&2; exit 2; }

STATUS_START=$(git status --porcelain) || {
    echo "cannot inspect Phase-2 source status" >&2; exit 2; }
if [[ -n "$STATUS_START" ]]; then
    echo "refusing dirty Phase-2 source" >&2
    git status --short >&2
    exit 2
fi
HEAD_START=$(git rev-parse HEAD)

"$PY" scripts/bind_legacy_phase2.py --verify-root "$FIXED_ROOT" --side fixed
"$PY" scripts/bind_legacy_phase2.py --verify-root "$LEGACY_ROOT" --side legacy
for root in "$FIXED_ROOT" "$LEGACY_ROOT"; do
    if find "$root" -mindepth 2 -maxdepth 2 \
        \( -name 'evaluation_siglip2*.json' -o -name 'cell_result.json' \
           -o -name 'pairwise_nmi.json' -o -name 'analysis_complete.json' \) \
        -print -quit | grep -q .; then
        echo "refusing input root that already contains metric output: $root" >&2
        exit 2
    fi
done

mkdir -m 0755 -- "$LOG_ROOT" "$REPORT_ROOT"
PAIR_RECEIPT="$REPORT_ROOT/phase2_input_pair_receipt.json"
"$PY" scripts/aggregate_phase2_f01.py \
    --phase2-root "$FIXED_ROOT" --legacy-root "$LEGACY_ROOT" \
    --pair-receipt "$PAIR_RECEIPT" --seal-input-pair

EXPECTED_CELLS=(cifar10_N4 cifar10_N9 cifar10_N19 cifar10_N39
                flickr25k_N4 flickr25k_N9 flickr25k_N19
                nuswide_N4 nuswide_N9 nuswide_N19 nuswide_N39
                mscoco_N4 mscoco_N9 mscoco_N19 mscoco_N39)
canonical() {
    case "$1" in
        cifar10) echo "CIFAR10 64" ;;
        flickr25k) echo "Flickr25k 128" ;;
        nuswide) echo "NUSWIDE 128" ;;
        mscoco) echo "MSCOCO 128" ;;
        *) return 1 ;;
    esac
}

{
    for side in fixed legacy; do
        root=$FIXED_ROOT; [[ "$side" == legacy ]] && root=$LEGACY_ROOT
        for name in "${EXPECTED_CELLS[@]}"; do
            slug=${name%_N*}; read -r dataset k <<<"$(canonical "$slug")"
            printf '%s\0%s\0%s\0%s\0%s\0' "$side" "$root/$name" \
                "$dataset" "$k" "$LOG_ROOT/${side}_${name}.log"
        done
    done
} | xargs -0 -r -n 5 -P "$JOBS" "$SCRIPT" --run-cell "$REPO" "$PY"

"$PY" scripts/bind_legacy_phase2.py --verify-root "$FIXED_ROOT" --side fixed
"$PY" scripts/bind_legacy_phase2.py --verify-root "$LEGACY_ROOT" --side legacy
for root in "$FIXED_ROOT" "$LEGACY_ROOT"; do
    [[ "$(find "$root" -mindepth 2 -maxdepth 2 -type f \
        -name analysis_complete.json | wc -l)" -eq 15 ]] || {
        echo "not all 15 cells are sealed under $root" >&2; exit 1; }
done
[[ "$(git rev-parse HEAD)" == "$HEAD_START" ]] || {
    echo "source HEAD changed during Phase 2" >&2; exit 2; }
STATUS_END=$(git status --porcelain) || {
    echo "cannot re-inspect Phase-2 source status" >&2; exit 2; }
[[ -z "$STATUS_END" ]] || {
    echo "Phase-2 source changed during execution" >&2; exit 2; }

"$PY" scripts/aggregate_phase2_f01.py \
    --phase2-root "$FIXED_ROOT" --legacy-root "$LEGACY_ROOT" \
    --pair-receipt "$PAIR_RECEIPT" \
    --out-json "$REPORT_ROOT/phase2_f01_impact.json" \
    --out-md "$REPORT_ROOT/phase2_f01_impact.md" \
    --out-receipt "$REPORT_ROOT/phase2_f01_receipt.json"
[[ "$(git rev-parse HEAD)" == "$HEAD_START" ]] || {
    echo "source HEAD changed during final aggregation" >&2; exit 2; }
STATUS_FINAL=$(git status --porcelain) || {
    echo "cannot inspect final Phase-2 source status" >&2; exit 2; }
[[ -z "$STATUS_FINAL" ]] || {
    echo "Phase-2 source changed during final aggregation" >&2; exit 2; }
"$PY" scripts/aggregate_phase2_f01.py --verify-report-receipt \
    "$REPORT_ROOT/phase2_f01_receipt.json" \
    --expected-fixed-root "$FIXED_ROOT" \
    --expected-legacy-root "$LEGACY_ROOT" \
    --expected-pair-receipt "$PAIR_RECEIPT" \
    --expected-report-json "$REPORT_ROOT/phase2_f01_impact.json" \
    --expected-report-md "$REPORT_ROOT/phase2_f01_impact.md"
[[ -f "$REPORT_ROOT/phase2_f01_receipt.json" \
   && ! -L "$REPORT_ROOT/phase2_f01_receipt.json" ]] || {
    echo "aggregator returned without its completion receipt" >&2; exit 1; }
echo "PHASE2 COMPLETE head=$HEAD_START report=$REPORT_ROOT/phase2_f01_receipt.json"
