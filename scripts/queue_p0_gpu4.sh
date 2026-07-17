#!/usr/bin/env bash
# P0 queue on GPU4: wait for the running NUS-WIDE final-epoch job to release the
# GPU, then run CIFAR10 and NUS-WIDE under the P0 validation protocol.
# GPUs 0-3 are occupied by an unrelated job (train.py) and must not be touched.
set -u

wait_for_tag() {   # $1 = substring of the --tag to wait on
    while pgrep -f "train_siglip2.py" | while read -r p; do
              ps -p "$p" -o args= 2>/dev/null | grep -q -- "$1" && echo hit
          done | grep -q hit; do
        sleep 60
    done
}

echo "[queue-gpu4] waiting for NUS-WIDE final-epoch to release GPU4 ..."
wait_for_tag "nuswide_v185_sweep_FINALEPOCH"
echo "[queue-gpu4] GPU4 free at $(date '+%F %T')"

# ---- CIFAR10 P0 (champion recipe: ccs=0.1, K=64) --------------------------
C=./cache/cifar10_clip
CACHE="$C" WHITEN_NPZ="$C/text_whiten_optTrain.npz" \
VAL_RATIO=0.1 VAL_SEED=42 TAG=cifar10_P0val_ccs01 \
    bash scripts/train_cifar10_v185_bidirTokenPrune05_ccs01_clip.sh 4
echo "[queue-gpu4] CIFAR10 P0 done at $(date '+%F %T')"

# ---- NUS-WIDE P0 (champion recipe: cibhash_ntxent=1.5) --------------------
N=./cache/nuswide_clip_tokens
CACHE="$N" WHITEN_NPZ="$N/text_whiten_optTrain.npz" \
CIBNT=1.5 VAL_RATIO=0.1 VAL_SEED=42 CELL=P0val TAG=nuswide_P0val_cb15 \
    bash scripts/train_nuswide_v185_sweep_clip.sh 4
echo "[queue-gpu4] NUS-WIDE P0 done at $(date '+%F %T')"
