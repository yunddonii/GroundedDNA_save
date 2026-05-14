#!/usr/bin/env bash
# Full CIFAR10 setting1 pipeline (plan A 축소판):
#   1) wait for / verify Qwen JSONL cache for the 5K + 1K subset
#   2) extract SigLIP2 visual + text features once, save to ./cache/cifar10_siglip2/
#   3) train SigLIP2 + DNA hashing for 60 epochs (text-guided routing on, bs=64)
#   4) auto extract + evaluate at the end (-ev flag)
#
# Re-running this script is safe: each step skips if its output already exists.
# Override knobs via env vars (TAG, EPOCHS, BATCH, ...).

set -euo pipefail

PY="${PY:-/home/yschoi/.conda/envs/dna_hashing/bin/python}"
CUDA="${CUDA:-0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

TAG="${TAG:-cifar10_setting1}"
EPOCHS="${EPOCHS:-60}"
BATCH="${BATCH:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_EVERY="${EVAL_EVERY:-5}"
PROJ_LR="${PROJ_LR:-0.001}"

DATASET_DIR="${DATASET_DIR:-./dataset}"
CIFAR10_ROOT="${CIFAR10_ROOT:-${DATASET_DIR}/CIFAR10}"
CACHE_DIR="${CACHE_DIR:-./cache}"
QWEN_CACHE="${QWEN_CACHE:-${CACHE_DIR}/cifar10_qwen.jsonl}"
FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR:-${CACHE_DIR}/cifar10_siglip2}"

# ---- step 1: confirm Qwen cache ----------------------------------------
GOOD=$(grep -c '"codebook_texts"' "${QWEN_CACHE}" 2>/dev/null || echo 0)
echo "============================================================"
echo "[1/4] Qwen cache check"
echo "      cache : ${QWEN_CACHE}"
echo "      rows  : ${GOOD}"
echo "============================================================"
if [ "${GOOD}" -lt 5000 ]; then
    echo "[1/4] WARNING: Qwen cache has only ${GOOD} rows (<5000)."
    echo "      You probably want to wait for the Qwen captioner to finish."
    echo "      Override: SKIP_QWEN_CHECK=1 to proceed anyway."
    if [ "${SKIP_QWEN_CHECK:-0}" != "1" ]; then
        exit 1
    fi
fi

# ---- step 2: extract SigLIP2 features (skip if cache complete) ---------
META_PATH="${FEATURE_CACHE_DIR}/meta.json"
echo "============================================================"
echo "[2/4] SigLIP2 feature extraction"
echo "      out -> ${FEATURE_CACHE_DIR}"
echo "============================================================"
if [ -f "${META_PATH}" ] && [ "${SKIP_FEATURE_EXTRACT:-0}" != "0" ]; then
    echo "[2/4] SKIPPED -- ${META_PATH} exists."
elif [ -f "${META_PATH}" ]; then
    EXISTING_N=$(${PY} -c "import json; print(json.load(open('${META_PATH}'))['N'])")
    echo "[2/4] cache exists with N=${EXISTING_N}; skipping. Set SKIP_FEATURE_EXTRACT=0 to force re-run."
else
    CUDA_VISIBLE_DEVICES="${CUDA}" "${PY}" extract_siglip2_features.py \
        --cifar10_root "${CIFAR10_ROOT}" \
        --qwen_cache_path "${QWEN_CACHE}" \
        --cache_dir "${FEATURE_CACHE_DIR}" \
        --batch_size 128
fi

# ---- step 3 + 4: train + auto extract+eval ----------------------------
echo "============================================================"
echo "[3/4] training SigLIP2 + DNA hashing  (epochs=${EPOCHS}, bs=${BATCH}, proj_lr=${PROJ_LR})"
echo "[4/4] extract + evaluate auto-trigger via -ev"
echo "============================================================"
CUDA_VISIBLE_DEVICES="${CUDA}" "${PY}" train_siglip2.py \
    --tag        "${TAG}" \
    --dataset    CIFAR10 \
    --setting    1 \
    --dataset_dir "${DATASET_DIR}" \
    -bs "${BATCH}" -e "${EPOCHS}" \
    --proj_lr "${PROJ_LR}" \
    -nd 0 \
    --num_workers "${NUM_WORKERS}" \
    --qwen_text_cache_path "${QWEN_CACHE}" \
    --siglip2_feature_cache_dir "${FEATURE_CACHE_DIR}" \
    --eval_every "${EVAL_EVERY}" \
    --dna_distance_mode base \
    --save \
    -ev

echo "============================================================"
echo "[done] artifacts under: ${REPO_ROOT}/result/$(date +%y%m%d)+${TAG}+bs+${BATCH}+e+${EPOCHS}+proj_lr+${PROJ_LR}/"
echo "============================================================"
