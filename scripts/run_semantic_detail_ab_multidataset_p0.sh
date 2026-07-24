#!/usr/bin/env bash
# Launch the four paper-protocol AB semantic-detail cells in parallel.
#
# AB = strict global-caption-free A + own local minimal-pair foils B.
# It deliberately excludes C (`--cibhash_visual_token_bit_kl`).
#
# Fixed matrix:
#   {flickr, mscoco, nuswide, cifar10} x {AB}, K=128/L=4, seed=42.
#
# Usage:
#   bash scripts/run_semantic_detail_ab_multidataset_p0.sh
#
# Optional env:
#   GPU_LIST=0,1,2,3
#       Exactly four distinct GPU IDs, one per dataset in the order above.
#   PREFLIGHT_ONLY=1
#       Validate all four cells without creating files or launching training.
#   ARTIFACT_DIR=...
#   PY=...
#
# All four preflights complete before the first GPU process is started.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/home/yschoi/.conda/envs/dna_hashing/bin/python}"
ARTIFACT_DIR="${ARTIFACT_DIR:-artifacts/semantic_detail_multidataset/p0}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
IFS=',' read -r -a GPUS <<< "${GPU_LIST:-0,1,2,3}"

if [[ "$PREFLIGHT_ONLY" != "0" && "$PREFLIGHT_ONLY" != "1" ]]; then
    printf '[semantic-p0-ab] PREFLIGHT_ONLY must be 0 or 1\n' >&2
    exit 2
fi
if [[ "${#GPUS[@]}" -ne 4 ]]; then
    printf '[semantic-p0-ab] GPU_LIST must contain exactly four IDs\n' >&2
    exit 2
fi
declare -A SEEN_GPUS=()
for gpu in "${GPUS[@]}"; do
    if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
        printf '[semantic-p0-ab] invalid GPU ID: %q\n' "$gpu" >&2
        exit 2
    fi
    if [[ -n "${SEEN_GPUS[$gpu]:-}" ]]; then
        printf '[semantic-p0-ab] GPU IDs must be distinct: %s\n' "$gpu" >&2
        exit 2
    fi
    SEEN_GPUS["$gpu"]=1
done

DATASETS=(flickr mscoco nuswide cifar10)

printf '[semantic-p0-ab] validating all four AB cells before launch\n'
for index in "${!DATASETS[@]}"; do
    env \
        -u BASE_CACHE -u FOIL_CACHE -u EVAL_CACHE -u QWEN \
        PREFLIGHT_ONLY=1 \
        ARTIFACT_DIR="$ARTIFACT_DIR" \
        PY="$PY" \
        bash scripts/run_semantic_detail_ab_p0_cell.sh \
        "${GPUS[$index]}" "${DATASETS[$index]}" AB
done

if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
    printf '[semantic-p0-ab] all-cell preflight passed\n'
    exit 0
fi

DRIVER_DIR="${ARTIFACT_DIR}/drivers"
for ds in "${DATASETS[@]}"; do
    driver_log="${DRIVER_DIR}/${ds}_AB.log"
    if [[ -e "$driver_log" ]]; then
        printf '[semantic-p0-ab] refusing to overwrite %s\n' \
            "$driver_log" >&2
        exit 3
    fi
done
mkdir -p "$DRIVER_DIR"

run_cell() {
    local gpu="$1"
    local ds="$2"
    local driver_log="${DRIVER_DIR}/${ds}_AB.log"
    printf '[semantic-p0-ab] START dataset=%s arm=AB gpu=%s\n' "$ds" "$gpu"
    if env \
        -u BASE_CACHE -u FOIL_CACHE -u EVAL_CACHE -u QWEN \
        ARTIFACT_DIR="$ARTIFACT_DIR" \
        PY="$PY" \
        bash scripts/run_semantic_detail_ab_p0_cell.sh \
        "$gpu" "$ds" AB >"$driver_log" 2>&1; then
        printf '[semantic-p0-ab] DONE  dataset=%s arm=AB gpu=%s\n' "$ds" "$gpu"
        return 0
    fi
    printf '[semantic-p0-ab] FAIL  dataset=%s arm=AB gpu=%s log=%s\n' \
        "$ds" "$gpu" "$driver_log" >&2
    return 1
}

declare -a PIDS=()
for index in "${!DATASETS[@]}"; do
    run_cell "${GPUS[$index]}" "${DATASETS[$index]}" &
    PIDS+=("$!")
done

status=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done

if [[ "$status" == "0" ]]; then
    printf '[semantic-p0-ab] ALL FOUR CELLS COMPLETE\n'
    grep -h '\[CELL-RESULT\]' "${DRIVER_DIR}"/*_AB.log | sort
else
    printf '[semantic-p0-ab] one or more cells failed; inspect %s\n' \
        "$DRIVER_DIR" >&2
fi
exit "$status"
