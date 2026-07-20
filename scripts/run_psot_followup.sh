#!/usr/bin/env bash
# PSOT follow-up (2026-07-20): grid refinement + MSCOCO transfer.
#
# Context. Per-slot independent OT -- reached with NO new code by freeing the
# Sinkhorn visual marginal (`--sinkhorn_lambda_a`; tau_a = la/(la+eps) -> 0 kills
# the row scaling, so P[:, :, m] becomes an independent per-slot softmax over
# patches) -- was the only candidate that MOVED the capacity <-> gradedness
# trade-off rather than traversing it:
#
#   cell             mAP@R    DNA-uniq   rho_cbk   slot overlap
#   cb1.0 champion   0.8810   0.4014     0.586     0.442
#   cb0.5            0.8654   0.2036     0.690     --
#   PSOT la=0.20     0.8708   0.3837     0.655     0.414
#   PSOT la=0.05     0.8628   0.1818     0.710     0.409
#
# la=0.20 keeps 96% of the champion's code diversity while gaining +0.069 rho.
# The predicted failure mode (slots freed from competition all read the same
# patches) did NOT occur -- overlap went DOWN, 0.442 -> 0.414.
#
# Two open questions this addresses:
#   (1) is la=0.20 actually the optimum, or an artefact of a 3-point grid?
#   (2) does it transfer to MSCOCO -- the dataset under the most capacity
#       pressure (107K DB, 80 labels, champion runs the highest cibhash of all)?
#       This is the decisive generality test.
#
# Single delta throughout: --sinkhorn_lambda_a only. Everything else matches the
# P0refit champions (cache, trainOnly whitening, schedule, stop epoch, final-
# epoch eval), so each cell is directly comparable to its own champion.
set -u
cd /home/yschoi/GroundedDNA
mkdir -p logs

F=./cache/flickr25k_clip_v4plus_qwen3_tokens
M=./cache/mscoco_clip_v5b_tokens

# ---- (1) Flickr grid refinement: 0.10 / 0.35 / 0.50 ----------------------
#      (0.05 / 0.20 / 1.0 already measured)
gpu=0
for la in 0.10 0.35 0.50; do
  (
    env CACHE="$F" WHITEN_NPZ="$F/text_whiten_trainOnly.npz" \
        FINAL_EPOCH=1 STOP_EP=4 BIDIR_MODE=legacy SLA="$la" \
        TAG="flickr_PSOT_la${la}" \
        bash scripts/train_flickr25k_v185_bidirTokenPrune05_clip.sh "$gpu"
  ) > "logs/flickr_PSOT_la${la}.log" 2>&1 &
  gpu=$((gpu + 1))
done

# ---- (2) MSCOCO transfer: champion la=1.0 vs 0.20 and 0.35 ---------------
#      MSCOCO champion = cibhash 1.5, E*=49. The sweep script has no SLA env,
#      so lambda_a is appended via EXTRA_ARGS-style override below.
for cell in "3:0.20" "4:0.35"; do
  IFS=':' read -r g la <<< "$cell"
  (
    env CACHE="$M" WHITEN_NPZ="$M/text_whiten_trainOnly.npz" \
        CIBNT=1.5 FINAL_EPOCH=1 STOP_EP=49 CELL=PSOT SLA="$la" \
        TAG="mscoco_PSOT_la${la}_e49" \
        bash scripts/train_mscoco_F2_sweep_clip.sh "$g"
  ) > "logs/mscoco_PSOT_la${la}.log" 2>&1 &
done

wait
echo "[psot-followup] all cells done at $(date '+%F %T')"
