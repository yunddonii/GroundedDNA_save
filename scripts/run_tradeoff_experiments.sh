#!/usr/bin/env bash
# Two parallel attempts to MOVE the capacity <-> gradedness trade-off rather
# than traverse it (2026-07-20).
#
# Established: instance discrimination (cibhash_ntxent) is the amplifier -- it
# sustains retrieval by keeping the codebook from collapsing, but it trades away
# semantic gradedness. Flickr cb0.5 buys rho_codebook +0.104 for only -0.016
# mAP@R, so the champion (cb1.0) overshoots. The question is whether capacity
# can be supplied from somewhere else, letting cb stay low while mAP recovers.
#
# EXP 1 -- F3: explicit codebook usage-balance regulariser (`loss_bu`).
#   `_loss_bu` already implements exactly this (usage -> uniform MSE + an
#   off-diagonal Gram decorrelation term), weight `--lambda_bu`, default 0.02.
#   Cells raise it while holding cb at 0.5. Proposed 2026-07-13 as "F3", never run.
#
# EXP 2 -- per-slot independent OT (the reviewable part of the routing proposal).
#   NO new code needed: the unbalanced-OT visual marginal already does this.
#   In `_log_sinkhorn`, tau_a = lambda_a / (lambda_a + eps). Driving lambda_a -> 0
#   sends tau_a -> 0, so log_u stays 0 and only the column (per-slot) scaling
#   survives: P[:, :, m] becomes an independent softmax over patches for slot m.
#   Each slot then selects its own tokens with NO cross-slot competition for
#   patch mass -- which is what the proposal asks for.
#
# Both run on Flickr (E*=4, ~10 min/cell) as single deltas off the cb0.5 cell,
# so every number is comparable to the cb1.0 champion and to cb0.5.
#
# Read-outs: mAP@R + DNA-uniq + codewords-used (capacity) vs rho_z / rho_codebook
# (gradedness), plus slot routing overlap for EXP 2 -- if removing the coupling
# makes every slot read the same patches, overlap goes to ~1.0 and it is dead.
set -u
cd /home/yschoi/GroundedDNA
mkdir -p logs

F=./cache/flickr25k_clip_v4plus_qwen3_tokens
COMMON=(env CACHE="$F" WHITEN_NPZ="$F/text_whiten_trainOnly.npz"
        FINAL_EPOCH=1 STOP_EP=4 BIDIR_MODE=legacy)

launch() {  # $1=gpu $2=tag $3..=extra env assignments
    local gpu="$1" tag="$2"; shift 2
    ( "${COMMON[@]}" "$@" TAG="$tag" \
        bash scripts/train_flickr25k_v185_bidirTokenPrune05_clip.sh "$gpu"
    ) > "logs/${tag}.log" 2>&1 &
}

# ---- EXP 1: F3 usage regulariser, cb held at 0.5 -------------------------
launch 0 "flickr_F3_cb0.5_bu0.10" CIBNT=0.5 LBU=0.10
launch 1 "flickr_F3_cb0.5_bu0.30" CIBNT=0.5 LBU=0.30

# ---- EXP 2: per-slot independent OT (free visual marginal) ---------------
launch 2 "flickr_PSOT_la0.05"     SLA=0.05
launch 3 "flickr_PSOT_la0.20"     SLA=0.20

wait
echo "[tradeoff] all 4 cells done at $(date '+%F %T')"
