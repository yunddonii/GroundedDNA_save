#!/usr/bin/env bash
# lambda_bio sweep on GPUs 0-3, in parallel with the audit-fix pipeline on 4/5.
#
# Why: every `+ L_bio` number so far uses a single lambda_bio = 10, picked on
# 2026-07-29 from a loss-magnitude estimate on Flickr ALONE and under the
# A-champion recipe -- never swept on the unified recipe or on the other three
# datasets. The measured spread says that value is wrong per dataset:
#
#   dataset    validity gain   DNA-uniq delta   mAP delta
#   CIFAR10        +46.8          -0.0411        -0.0053
#   Flickr25k      +43.5          -0.0725        -0.0011
#   MSCOCO         +40.6          +0.0067        -0.0120
#   NUSWIDE        +21.1          -0.0210        +0.0014
#
# NUS is clearly under-pressured (lowest validity 77.5%, retrieval even improves)
# and Flickr/CIFAR are over-pressured (validity already 84-86% but paying the
# largest diversity cost). Same pattern L_joint showed before per-dataset lambda
# fixed it.
#
# Objective: keep pre-projection validity >= 80% while minimising the DNA-unique
# loss.
#
# GPUs 4/5 belong to the audit-fix pipeline (CIBHash matrix, then A2/A4). This
# script is hard-pinned to 0-3 and never polls for "free" GPUs, so it cannot
# steal a slot from that pipeline between its cells.
set -u
cd /home/yschoi/GroundedDNA
mkdir -p logs
ALLOWED=(0 1 2 3)

NB="--no_gumbel_softmax"
CB="--no_gumbel_softmax --lambda_codeword_codon_sinkhorn 0.0"

# exp | tag | aux
QUEUE=(
  "nuswide_A_v4|bioL20|--lambda_codon_joint 0.05 $NB --lambda_bio_constraint 20.0"
  "nuswide_A_v4|bioL30|--lambda_codon_joint 0.05 $NB --lambda_bio_constraint 30.0"
  "flickr_A_v4|bioL03|--lambda_codon_joint 0.02 $NB --lambda_bio_constraint 3.0"
  "flickr_A_v4|bioL05|--lambda_codon_joint 0.02 $NB --lambda_bio_constraint 5.0"
  "cifar_A_v4|bioL03|--lambda_codon_joint 0.03 $CB --lambda_bio_constraint 3.0"
  "cifar_A_v4|bioL05|--lambda_codon_joint 0.03 $CB --lambda_bio_constraint 5.0"
  "mscoco_A_v5b|bioL05|--lambda_codon_joint 0.03 $NB --lambda_bio_constraint 5.0"
  "mscoco_A_v5b|bioL20|--lambda_codon_joint 0.03 $NB --lambda_bio_constraint 20.0"
)

echo "[bioL] ${#QUEUE[@]} cells on GPUs ${ALLOWED[*]} @ $(date '+%F %T')"
declare -A PID
i=0
while [ "$i" -lt "${#QUEUE[@]}" ]; do
  for g in "${ALLOWED[@]}"; do
    if [ -n "${PID[$g]:-}" ] && kill -0 "${PID[$g]}" 2>/dev/null; then continue; fi
    [ "$i" -ge "${#QUEUE[@]}" ] && break
    IFS='|' read -r exp tag aux <<<"${QUEUE[$i]}"
    echo "[bioL] GPU$g <- $exp $tag @ $(date '+%F %T')"
    setsid nohup bash scripts/sweep_joint_cell.sh "$g" "$exp" "$tag" "$aux" \
      > "logs/SWEEP_${exp}_${tag}.out" 2>&1 < /dev/null &
    PID[$g]=$!
    i=$((i+1)); sleep 45
  done
  sleep 60
done
echo "[bioL] all ${#QUEUE[@]} cells dispatched @ $(date '+%F %T')"
