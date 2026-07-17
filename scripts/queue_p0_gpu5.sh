#!/usr/bin/env bash
# P0 queue on GPU5: wait for the running Flickr P0 job, then run MSCOCO P0.
set -u

wait_for_tag() {   # $1 = substring of the --tag to wait on
    while pgrep -f "train_siglip2.py" | while read -r p; do
              ps -p "$p" -o args= 2>/dev/null | grep -q -- "$1" && echo hit
          done | grep -q hit; do
        sleep 60
    done
}

echo "[queue-gpu5] waiting for Flickr P0 to release GPU5 ..."
wait_for_tag "flickr_P0val_wholeimg"
echo "[queue-gpu5] GPU5 free at $(date '+%F %T')"

# ---- MSCOCO P0 (champion recipe: cibhash_ntxent=1.5, whole-image F2) -------
M=./cache/mscoco_clip_v5b_tokens
CACHE="$M" WHITEN_NPZ="$M/text_whiten_optTrain.npz" \
CIBNT=1.5 VAL_RATIO=0.1 VAL_SEED=42 CELL=P0val TAG=mscoco_P0val_cb15 \
    bash scripts/train_mscoco_F2_sweep_clip.sh 5
echo "[queue-gpu5] MSCOCO P0 done at $(date '+%F %T')"
