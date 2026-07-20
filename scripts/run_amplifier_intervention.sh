#!/usr/bin/env bash
# Amplifier intervention (2026-07-20).
#
# Finding to test: the framework AMPLIFIES semantic gradedness that neither input
# has (patch-mean rho 0.021 -> z 0.338 on Flickr, 0.063 -> 0.332 on NUS-WIDE),
# but on MSCOCO it INVERTS (CLS 0.131 -> z 0.080). Five design directions were
# already refuted by measurement; the remaining hypothesis is that the amplifier
# is instance discrimination (`lambda_cibhash_ntxent`) interacting with VQ, and
# that on MSCOCO -- where mean-pooled patches are ANTI-correlated with label
# similarity (-0.115) -- the same pressure is counterproductive.
#
# Single delta: lambda_cibhash_ntxent only. Everything else (cache, whitening,
# schedule, stop epoch, final-epoch eval) is identical to the P0refit champions,
# so each cell is directly comparable to its champion.
#
#   Flickr25k  champion CIBNT=1.0  E*=4   -> cells 0.0, 0.5
#   MSCOCO     champion CIBNT=1.5  E*=49  -> cells 0.0, 0.5
#   (MSCOCO also has an existing clean CIBNT=1.0 single-delta run: the
#    260714 F2_WHOLEIMG base, which differs from sweepC only in this weight.)
#
# Predictions:
#   amplifier = cibhash        -> Flickr rho_z COLLAPSES toward patch-mean
#   counterproductive on COCO  -> MSCOCO rho_z RISES above 0.080
#   both together = double dissociation (strong support)
#   collapse on both           -> cibhash is the amplifier but MSCOCO's defect
#                                 lies elsewhere
set -u
cd /home/yschoi/GroundedDNA
mkdir -p logs

F=./cache/flickr25k_clip_v4plus_qwen3_tokens
M=./cache/mscoco_clip_v5b_tokens

# ---- GPU 0/1: Flickr (stops at epoch 4, ~10 min each) --------------------
for cell in "0:0.0" "1:0.5"; do
  IFS=':' read -r gpu cb <<< "$cell"
  (
    CACHE="$F" WHITEN_NPZ="$F/text_whiten_trainOnly.npz" \
    FINAL_EPOCH=1 STOP_EP=4 BIDIR_MODE=legacy CIBNT="$cb" \
    TAG="flickr_AMPL_cb${cb}_e4" \
        bash scripts/train_flickr25k_v185_bidirTokenPrune05_clip.sh "$gpu"
  ) > "logs/ampl_flickr_cb${cb}.log" 2>&1 &
done

# ---- GPU 2/3: MSCOCO (stops at epoch 49, ~90 min each) -------------------
for cell in "2:0.0" "3:0.5"; do
  IFS=':' read -r gpu cb <<< "$cell"
  (
    CACHE="$M" WHITEN_NPZ="$M/text_whiten_trainOnly.npz" \
    CIBNT="$cb" FINAL_EPOCH=1 STOP_EP=49 CELL=AMPL \
    TAG="mscoco_AMPL_cb${cb}_e49" \
        bash scripts/train_mscoco_F2_sweep_clip.sh "$gpu"
  ) > "logs/ampl_mscoco_cb${cb}.log" 2>&1 &
done

wait
echo "[ampl] all 4 cells done at $(date '+%F %T')"
