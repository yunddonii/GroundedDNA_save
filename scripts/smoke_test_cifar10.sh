#!/usr/bin/env bash
# End-to-end smoke test on CIFAR10:
#   1) Generate Qwen3-VL JSONL cache for a small subset of CIFAR10 images
#   2) Train SigLIP2 + DNA hashing for a few epochs (text-guided routing active)
#   3) Run final extraction + evaluation (auto-triggered by `-ev`)
#
# This is a SMOKE TEST — small batch, few epochs, capped images. Use it to
# verify all pipeline stages connect; expect mAP to be modest. Increase
# epochs / drop --limit for real runs.

set -euo pipefail

# ---- env ----------------------------------------------------------------
# Adjust if your conda env / python is elsewhere
PY="${PY:-/home/yschoi/.conda/envs/dna_hashing/bin/python}"
CUDA="${CUDA:-0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

# ---- knobs --------------------------------------------------------------
TAG="${TAG:-cifar10_smoke}"
LIMIT="${LIMIT:-100}"          # how many CIFAR10 images to caption with Qwen
EPOCHS="${EPOCHS:-2}"
BATCH="${BATCH:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_EVERY="${EVAL_EVERY:-1}"

DATASET_DIR="${DATASET_DIR:-./dataset}"          # CIFAR10 will be downloaded here
CIFAR10_ROOT="${CIFAR10_ROOT:-${DATASET_DIR}/CIFAR10}"
CACHE_DIR="${CACHE_DIR:-./cache}"
CACHE_PATH="${CACHE_PATH:-${CACHE_DIR}/cifar10_qwen.jsonl}"

mkdir -p "${CACHE_DIR}"

# ---- step 1: Qwen JSONL cache -------------------------------------------
# Skip if (a) explicit SKIP_STEP1=1, or (b) cache already has >= LIMIT good rows.
EXISTING=0
if [ -f "${CACHE_PATH}" ]; then
    EXISTING=$(grep -c '"codebook_texts"' "${CACHE_PATH}" || true)
fi

if [ "${SKIP_STEP1:-}" = "1" ] || [ "${EXISTING}" -ge "${LIMIT}" ]; then
    echo "============================================================"
    echo "[1/3] SKIPPED — cache already has ${EXISTING} good rows (>= LIMIT=${LIMIT})"
    echo "      cache    : ${CACHE_PATH}"
    echo "      override : SKIP_STEP1=0 to force re-run"
    echo "============================================================"
else
    echo "============================================================"
    echo "[1/3] Qwen3-VL preprocessing — limit=${LIMIT} images"
    echo "      cache -> ${CACHE_PATH}"
    echo "      (existing cache rows: ${EXISTING}; will append-skip duplicates)"
    echo "============================================================"
    CUDA_VISIBLE_DEVICES="${CUDA}" "${PY}" preprocess_qwen_codebook_texts.py \
        --cifar10 \
        --cifar10_root "${CIFAR10_ROOT}" \
        --cache_path  "${CACHE_PATH}" \
        --limit       "${LIMIT}"
fi

# Sanity: how many usable rows ended up in the cache?
GOOD=$(grep -c '"codebook_texts"' "${CACHE_PATH}" || true)
echo "[1/3] cache rows with codebook_texts: ${GOOD}"
if [ "${GOOD}" -eq 0 ]; then
    echo "[1/3] ERROR: cache is empty — Qwen run failed?"
    exit 1
fi

# ---- step 2 + 3: training + auto extract+eval ---------------------------
echo "============================================================"
echo "[2/3] training SigLIP2 + DNA hashing  (epochs=${EPOCHS}, bs=${BATCH})"
echo "[3/3] extract + evaluate auto-trigger via -ev"
echo "============================================================"
CUDA_VISIBLE_DEVICES="${CUDA}" "${PY}" train_siglip2.py \
    --tag        "${TAG}" \
    --dataset    CIFAR10 \
    --setting    1 \
    --dataset_dir "${DATASET_DIR}" \
    -bs "${BATCH}" -e "${EPOCHS}" \
    -nd 0 \
    --num_workers "${NUM_WORKERS}" \
    --qwen_text_cache_path "${CACHE_PATH}" \
    --eval_every "${EVAL_EVERY}" \
    --dna_distance_mode base \
    --save \
    -ev

echo "============================================================"
echo "[done] artifacts under: ${REPO_ROOT}/result/$(date +%y%m%d)+${TAG}+bs+${BATCH}+e+${EPOCHS}+proj_lr+0.001/"
echo "       ├── log.csv                    (per-epoch metrics, mid-eval mAP)"
echo "       ├── model_state_dict.pth       (final checkpoint)"
echo "       ├── criterion_state_dict.pth   (EMA buffer)"
echo "       ├── extract_db.npz / extract_query.npz"
echo "       └── evaluation_siglip2_base.json   (final mAP / collapse metrics)"
echo "============================================================"
