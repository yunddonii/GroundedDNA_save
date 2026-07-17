#!/usr/bin/env bash
# P0 stage 2 -- refit on 100% of train, stopping at the epoch E* that stage 1
# selected on the held-out val split.
#
# Why not just re-run with `-e E*`: the LR scheduler is CosineAnnealingLR with
# T_max=args.epoch, so `-e 5` completes a full cosine cycle in 5 epochs and
# yields a different model than epoch 4 of a 60-epoch schedule. --stop_after_epoch
# keeps `-e 60` (schedule intact) and breaks the loop at E*, which is also how
# the baselines' epoch_XXX.pth checkpoints were produced -- so both sides compare
# like for like.
#
# Whitening switches to text_whiten_trainOnly.npz: stage 2 trains on all of
# train, so the statistics may see all train rows -- but still never test/db.
# --final_epoch_eval prevents any best-checkpoint swap: the evaluated weights are
# exactly epoch E*, and test is touched once.
set -u

E_FLICKR="${E_FLICKR:-4}"
E_CIFAR="${E_CIFAR:-14}"
E_MSCOCO="${E_MSCOCO:-49}"
E_NUSWIDE="${E_NUSWIDE:-}"      # filled from the stage-1 log below if empty

wait_free() {   # $1 = tag substring to wait on
    while pgrep -f "train_siglip2.py" | while read -r p; do
              ps -p "$p" -o args= 2>/dev/null | grep -q -- "$1" && echo hit
          done | grep -q hit; do
        sleep 60
    done
}

if [ -z "$E_NUSWIDE" ]; then
    echo "[refit] waiting for NUS-WIDE stage-1 to finish so E* is known ..."
    wait_free "nuswide_P0val_cb15"
    E_NUSWIDE=$(grep -oE "swapping final checkpoint with best \(epoch [0-9]+" \
                logs/nuswide_P0val_cb15.log | grep -oE "[0-9]+$" | tail -1)
    [ -z "$E_NUSWIDE" ] && { echo "[refit] FATAL: could not read NUS-WIDE E*"; exit 1; }
fi
echo "[refit] E* -> flickr=$E_FLICKR cifar10=$E_CIFAR mscoco=$E_MSCOCO nuswide=$E_NUSWIDE"

# ---- GPU 5: Flickr then MSCOCO ------------------------------------------
(
  F=./cache/flickr25k_clip_v4plus_qwen3_tokens
  CACHE="$F" WHITEN_NPZ="$F/text_whiten_trainOnly.npz" \
  FINAL_EPOCH=1 STOP_EP="$E_FLICKR" TAG="flickr_P0refit_e${E_FLICKR}" BIDIR_MODE=legacy \
      bash scripts/train_flickr25k_v185_bidirTokenPrune05_clip.sh 5
  M=./cache/mscoco_clip_v5b_tokens
  CACHE="$M" WHITEN_NPZ="$M/text_whiten_trainOnly.npz" \
  CIBNT=1.5 FINAL_EPOCH=1 STOP_EP="$E_MSCOCO" CELL=P0refit TAG="mscoco_P0refit_e${E_MSCOCO}" \
      bash scripts/train_mscoco_F2_sweep_clip.sh 5
) > logs/queue_refit_gpu5.log 2>&1 &

# ---- GPU 4: CIFAR10 then NUS-WIDE ---------------------------------------
(
  C=./cache/cifar10_clip
  CACHE="$C" WHITEN_NPZ="$C/text_whiten_trainOnly.npz" \
  FINAL_EPOCH=1 STOP_EP="$E_CIFAR" TAG="cifar10_P0refit_e${E_CIFAR}" \
      bash scripts/train_cifar10_v185_bidirTokenPrune05_ccs01_clip.sh 4
  N=./cache/nuswide_clip_tokens
  CACHE="$N" WHITEN_NPZ="$N/text_whiten_trainOnly.npz" \
  CIBNT=1.5 FINAL_EPOCH=1 STOP_EP="$E_NUSWIDE" CELL=P0refit TAG="nuswide_P0refit_e${E_NUSWIDE}" \
      bash scripts/train_nuswide_v185_sweep_clip.sh 4
) > logs/queue_refit_gpu4.log 2>&1 &

wait
echo "[refit] all done at $(date '+%F %T')"
