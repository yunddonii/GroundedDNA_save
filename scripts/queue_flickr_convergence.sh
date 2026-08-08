#!/usr/bin/env bash
# Make Flickr25k training CONVERGE instead of peaking at the first eval point.
#
# Diagnosis (2026-08-07): Flickr's val eval_mAP falls monotonically from the
# first measured epoch, .7761 @e4 -> .7275 @e59, so E* is always 4 (the first
# eval, since --eval_every 5). This is NOT overfitting: val_loss keeps
# IMPROVING over the same span, 1.375 -> 0.585. The composite objective
# generalises; it is simply misaligned with retrieval past ~4 epochs.
#
# Correlating each val-loss component with eval_mAP across epochs
# (positive = the term goes down while mAP goes down, i.e. misaligned):
#
#   term            Flickr   MS-COCO(healthy)
#   dna             +0.982     -0.638
#   entropy         +0.981     -0.636
#   quant           +0.963     -0.600
#   cb_uncorr       +0.946     -0.328
#   wasserstein     +0.857     -0.404
#   TOTAL           +0.798     -0.209
#   codon_joint     -0.939     +0.502     <- aligned on Flickr
#   text_hash_ntxent-0.963     +0.366     <- aligned on Flickr
#
# The structural-regularisation family is what drags retrieval down once the
# semantic-alignment terms have said what they can. `lambda_wasserstein` is
# 0.15, the largest of that family.
#
# Two independent levers:
#   ep20   -e 60 -> 20. E*=4 out of 60 means the eps/gumbel cosine schedules
#          never leave their initial value; a 20-epoch budget lets them finish
#          inside the span that is actually used.
#   wass005  --lambda_wasserstein 0.15 -> 0.05.
#
# --eval_every 1 everywhere: epochs 0-3 have never been measured, so the true
# peak may lie before the first eval point we have ever seen.
set -u
cd /home/yschoi/GroundedDNA
mkdir -p logs
NB="--no_gumbel_softmax"
JD="--lambda_codon_joint 0.02"
QUEUE=(
  "0|cvEp20|$JD $NB -e 20 --eval_every 1"
  "1|cvWass005|$JD $NB --lambda_wasserstein 0.05 --eval_every 1"
  "2|cvEp20Wass005|$JD $NB -e 20 --lambda_wasserstein 0.05 --eval_every 1"
  "3|cvBaseEv1|$JD $NB --eval_every 1"
)
for spec in "${QUEUE[@]}"; do
  IFS='|' read -r g tag aux <<<"$spec"
  echo "[cv] GPU$g <- flickr_A_v4 $tag @ $(date '+%F %T')"
  setsid nohup env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    bash scripts/sweep_joint_cell.sh "$g" flickr_A_v4 "$tag" "$aux" \
    > "logs/SWEEP_flickr_A_v4_${tag}.out" 2>&1 < /dev/null &
  sleep 40
done
echo "[cv] dispatched @ $(date '+%F %T')"
