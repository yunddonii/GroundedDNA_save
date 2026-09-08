#!/usr/bin/env bash
# Sharpen the router instead of tuning the mask that sits on top of it.
#
# Measured: the Sinkhorn plan is nearly uniform on three of four datasets.
# p_max median vs the 1/6 = 0.1667 uniform value, each at that run's E*:
#
#   dataset    E*   eps(E*)   p_max    x uniform
#   NUS-WIDE    4    0.990   0.2024      1.21
#   Flickr25k   4    0.990   0.2011      1.21
#   CIFAR-10   19    0.789   0.2065      1.24
#   MS-COCO    39    0.332   0.3264      1.96
#
# eps and p_max are perfectly monotone, and MS-COCO -- the only dataset with no
# dead slots, no empty-slot images and the best slot balance -- is the only one
# whose router actually discriminates. The three weak ones all stop at E*=4-19,
# where the cosine anneal has barely moved eps off eps_init = 1.0.
#
# Two consequences of a near-uniform plan:
#   * the adaptive top-p threshold is not adaptive -- p_max spans 0.192-0.216 on
#     NUS, so tau spans 0.874-0.883, i.e. 2.6 % of its nominal 0.60-0.95 range.
#   * the mask then selects among near-equal probabilities, i.e. on noise.
#
# So lower eps_init rather than widening the nucleus. eps_final and the cosine
# schedule are untouched; only the starting point moves, which is what the
# early-stopping runs actually experience.
#
# NOT the same as the discarded `otNOANN` cell: that set eps_init = eps_final =
# 0.1 and removed the anneal entirely, collapsing slot 0 from 50 to 31 codons.
# These keep the anneal and only start it lower.
set -u
cd /home/yschoi/GroundedDNA
mkdir -p logs
NB="--no_gumbel_softmax"
CB="--no_gumbel_softmax --lambda_codeword_codon_sinkhorn 0.0"
QUEUE=(
  "nuswide_A_v4|epsI03|--lambda_codon_joint 0.05 $NB --sinkhorn_epsilon_init 0.3"
  "nuswide_A_v4|epsI05|--lambda_codon_joint 0.05 $NB --sinkhorn_epsilon_init 0.5"
  "cifar_A_v4|epsI03|--lambda_codon_joint 0.03 $CB --sinkhorn_epsilon_init 0.3"
  "flickr_A_v4|epsI03|--lambda_codon_joint 0.02 $NB --sinkhorn_epsilon_init 0.3"
)
ALLOWED=(2 3 5 0 4)
declare -A PID
i=0
while [ "$i" -lt "${#QUEUE[@]}" ]; do
  for g in "${ALLOWED[@]}"; do
    if [ -n "${PID[$g]:-}" ] && kill -0 "${PID[$g]}" 2>/dev/null; then continue; fi
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
    [ "$u" -gt 500 ] && continue
    [ "$i" -ge "${#QUEUE[@]}" ] && break
    IFS='|' read -r exp tag aux <<<"${QUEUE[$i]}"
    echo "[eps] GPU$g <- $exp $tag @ $(date '+%F %T')"
    setsid nohup env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      bash scripts/sweep_joint_cell.sh "$g" "$exp" "$tag" "$aux" \
      > "logs/SWEEP_${exp}_${tag}.out" 2>&1 < /dev/null &
    PID[$g]=$!
    i=$((i+1)); sleep 45
  done
  sleep 60
done
echo "[eps] all dispatched @ $(date '+%F %T')"
