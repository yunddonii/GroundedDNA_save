#!/usr/bin/env bash
# 3-seed verification of the 5-slot / 15-base (30-bit) variant.
#
# Seed 42 gave a codon-decoding gain on 4/4 datasets while cutting the code
# budget by 17 %: MS-COCO +.0279, Flickr +.0174, CIFAR +.0169, NUS +.0107.
# Retrieval moved -.0064 (Flickr) and +.0049 (CIFAR). The slot removed is
# `scene_type`, the only slot whose deletion IMPROVED decoding on 4/4 in the
# drop ablation, and it is index 5 in the cache so truncation drops exactly it.
#
# GPUs 2 and 3 belong to the 30-bit baseline matrix; this is hard-pinned to
# 0/1/4/5 and never polls for free GPUs, so it cannot steal a slot from that
# matrix between its cells.
set -u
cd /home/yschoi/GroundedDNA
mkdir -p logs
ALLOWED=(0 1 4 5)
QUEUE=(
  "flickr_A_v4|0.02|43|"
  "flickr_A_v4|0.02|44|"
  "cifar_A_v4|0.03|43|--lambda_codeword_codon_sinkhorn 0.0"
  "cifar_A_v4|0.03|44|--lambda_codeword_codon_sinkhorn 0.0"
  "mscoco_A_v5b|0.03|43|"
  "mscoco_A_v5b|0.03|44|"
  "nuswide_A_v4|0.05|43|"
  "nuswide_A_v4|0.05|44|"
)
declare -A PID
i=0
while [ "$i" -lt "${#QUEUE[@]}" ]; do
  for g in "${ALLOWED[@]}"; do
    if [ -n "${PID[$g]:-}" ] && kill -0 "${PID[$g]}" 2>/dev/null; then continue; fi
    [ "$i" -ge "${#QUEUE[@]}" ] && break
    IFS='|' read -r exp jd s extra <<<"${QUEUE[$i]}"
    echo "[slot5] GPU$g <- $exp s$s @ $(date '+%F %T')"
    setsid nohup env GDNA_NUM_SEMANTIC_PARTS=5 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      bash scripts/sweep_joint_cell.sh "$g" "$exp" "slot5_s${s}" \
      "--lambda_codon_joint $jd --no_gumbel_softmax --num_semantic_parts 5 --num_codebooks 5 --random_seed $s $extra" \
      > "logs/SWEEP_${exp}_slot5_s${s}.out" 2>&1 < /dev/null &
    PID[$g]=$!
    i=$((i+1)); sleep 40
  done
  sleep 60
done
echo "[slot5] all ${#QUEUE[@]} cells dispatched @ $(date '+%F %T')"
