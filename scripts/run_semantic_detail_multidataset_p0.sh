#!/usr/bin/env bash
# Launch the eight paper-protocol semantic-detail cells across six GPUs.
#
# The full matrix is:
#   {flickr, mscoco, nuswide, cifar10} x {A, ABC}, fixed K=128/L=4.
#
# Usage:
#   bash scripts/run_semantic_detail_multidataset_p0.sh
#
# Optional env:
#   GPU_LIST=0,1,2,3,4,5
#       Exactly six distinct GPU IDs. MSCOCO occupies queues 0/1, NUS-WIDE
#       queues 2/3, and CIFAR10 queues 4/5; Flickr follows NUS-WIDE on 2/3.
#   PREFLIGHT_ONLY=1
#       Validate all eight cells without creating files or launching training.
#   ARTIFACT_DIR=...
#   PY=...
#
# All eight preflights complete before the first GPU process is started, so a
# missing/stale cache or an existing tag cannot leave a partially launched
# matrix.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/home/yschoi/.conda/envs/dna_hashing/bin/python}"
ARTIFACT_DIR="${ARTIFACT_DIR:-artifacts/semantic_detail_multidataset/p0}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
IFS=',' read -r -a GPUS <<< "${GPU_LIST:-0,1,2,3,4,5}"

if [[ "$PREFLIGHT_ONLY" != "0" && "$PREFLIGHT_ONLY" != "1" ]]; then
    printf '[semantic-p0-matrix] PREFLIGHT_ONLY must be 0 or 1\n' >&2
    exit 2
fi
if [[ "${#GPUS[@]}" -ne 6 ]]; then
    printf '[semantic-p0-matrix] GPU_LIST must contain exactly six IDs\n' >&2
    exit 2
fi
declare -A SEEN_GPUS=()
for gpu in "${GPUS[@]}"; do
    if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
        printf '[semantic-p0-matrix] invalid GPU ID: %q\n' "$gpu" >&2
        exit 2
    fi
    if [[ -n "${SEEN_GPUS[$gpu]:-}" ]]; then
        printf '[semantic-p0-matrix] GPU IDs must be distinct: %s\n' "$gpu" >&2
        exit 2
    fi
    SEEN_GPUS["$gpu"]=1
done

CELLS=(
    mscoco:A
    mscoco:ABC
    nuswide:A
    flickr:A
    nuswide:ABC
    flickr:ABC
    cifar10:A
    cifar10:ABC
)
CELL_GPUS=(
    "${GPUS[0]}"
    "${GPUS[1]}"
    "${GPUS[2]}"
    "${GPUS[2]}"
    "${GPUS[3]}"
    "${GPUS[3]}"
    "${GPUS[4]}"
    "${GPUS[5]}"
)

printf '[semantic-p0-matrix] validating all eight cells before launch\n'
for index in "${!CELLS[@]}"; do
    IFS=: read -r ds arm <<< "${CELLS[$index]}"
    env \
        -u BASE_CACHE -u FOIL_CACHE -u EVAL_CACHE -u QWEN \
        PREFLIGHT_ONLY=1 \
        ARTIFACT_DIR="$ARTIFACT_DIR" \
        PY="$PY" \
        bash scripts/run_semantic_detail_p0_cell.sh \
        "${CELL_GPUS[$index]}" "$ds" "$arm"
done

if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
    printf '[semantic-p0-matrix] all-cell preflight passed\n'
    exit 0
fi

DRIVER_DIR="${ARTIFACT_DIR}/drivers"
for cell in "${CELLS[@]}"; do
    IFS=: read -r ds arm <<< "$cell"
    driver_log="${DRIVER_DIR}/${ds}_${arm}.log"
    if [[ -e "$driver_log" ]]; then
        printf '[semantic-p0-matrix] refusing to overwrite %s\n' \
            "$driver_log" >&2
        exit 3
    fi
done
mkdir -p "$DRIVER_DIR"

run_cell() {
    local gpu="$1"
    local ds="$2"
    local arm="$3"
    local driver_log="${DRIVER_DIR}/${ds}_${arm}.log"
    printf '[semantic-p0-matrix] START dataset=%s arm=%s gpu=%s\n' \
        "$ds" "$arm" "$gpu"
    if env \
        -u BASE_CACHE -u FOIL_CACHE -u EVAL_CACHE -u QWEN \
        ARTIFACT_DIR="$ARTIFACT_DIR" \
        PY="$PY" \
        bash scripts/run_semantic_detail_p0_cell.sh \
        "$gpu" "$ds" "$arm" >"$driver_log" 2>&1; then
        printf '[semantic-p0-matrix] DONE  dataset=%s arm=%s gpu=%s\n' \
            "$ds" "$arm" "$gpu"
        return 0
    fi
    printf '[semantic-p0-matrix] FAIL  dataset=%s arm=%s gpu=%s log=%s\n' \
        "$ds" "$arm" "$gpu" "$driver_log" >&2
    return 1
}

run_queue() {
    local gpu="$1"
    shift
    local status=0
    local cell ds arm
    for cell in "$@"; do
        IFS=: read -r ds arm <<< "$cell"
        if ! run_cell "$gpu" "$ds" "$arm"; then
            status=1
        fi
    done
    return "$status"
}

declare -a PIDS=()
run_queue "${GPUS[0]}" mscoco:A &
PIDS+=("$!")
run_queue "${GPUS[1]}" mscoco:ABC &
PIDS+=("$!")
run_queue "${GPUS[2]}" nuswide:A flickr:A &
PIDS+=("$!")
run_queue "${GPUS[3]}" nuswide:ABC flickr:ABC &
PIDS+=("$!")
run_queue "${GPUS[4]}" cifar10:A &
PIDS+=("$!")
run_queue "${GPUS[5]}" cifar10:ABC &
PIDS+=("$!")

status=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done

if [[ "$status" == "0" ]]; then
    printf '[semantic-p0-matrix] ALL EIGHT CELLS COMPLETE\n'
    grep -h '\[CELL-RESULT\]' "${DRIVER_DIR}"/*.log | sort
else
    printf '[semantic-p0-matrix] one or more queues failed; inspect %s\n' \
        "$DRIVER_DIR" >&2
fi
exit "$status"
