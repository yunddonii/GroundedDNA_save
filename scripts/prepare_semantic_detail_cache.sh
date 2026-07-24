#!/usr/bin/env bash
# Build/reuse the semantic-detail cache overlay for one MAIN-TABLE dataset.
#
# Usage:
#   bash scripts/prepare_semantic_detail_cache.sh \
#       <flickr|mscoco|nuswide|cifar10> [GPU]
#
# Examples:
#   PREFLIGHT_ONLY=1 bash scripts/prepare_semantic_detail_cache.sh mscoco 2
#   bash scripts/prepare_semantic_detail_cache.sh mscoco 2
#
# Optional environment overrides:
#   PY, BASE_CACHE, QWEN, OVERLAY, FOIL_JSONL, FOIL_SEED,
#   POOLED_BATCH_SIZE, TOKEN_BATCH_SIZE
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ "$#" -lt 1 || "$#" -gt 2 ]]; then
    printf 'usage: %s <flickr|mscoco|nuswide|cifar10> [GPU]\n' "$0" >&2
    exit 2
fi

DATASET="$1"
GPU_INDEX="${2:-${GPU:-0}}"
PY="${PY:-/home/yschoi/.conda/envs/dna_hashing/bin/python}"
if [[ ! -x "$PY" ]]; then
    printf '[semantic-cache] Python is not executable: %s\n' "$PY" >&2
    exit 2
fi

declare -a args=(
    --dataset "$DATASET"
    --gpu "$GPU_INDEX"
    --seed "${FOIL_SEED:-0}"
    --pooled-batch-size "${POOLED_BATCH_SIZE:-256}"
    --token-batch-size "${TOKEN_BATCH_SIZE:-256}"
)

if [[ -n "${BASE_CACHE:-}" ]]; then
    args+=(--base-cache "$BASE_CACHE")
fi
if [[ -n "${QWEN:-}" ]]; then
    args+=(--qwen "$QWEN")
fi
if [[ -n "${OVERLAY:-}" ]]; then
    args+=(--overlay "$OVERLAY")
fi
if [[ -n "${FOIL_JSONL:-}" ]]; then
    args+=(--foil-jsonl "$FOIL_JSONL")
fi
if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
    args+=(--preflight-only)
fi

exec "$PY" scripts/prepare_semantic_detail_cache.py "${args[@]}"
