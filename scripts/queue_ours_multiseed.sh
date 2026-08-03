#!/usr/bin/env bash
# Queue the GroundedDNA multi-seed cells (seeds 43, 44) behind whatever is
# currently occupying the GPUs, one cell per free GPU as they free up.
#
# Protocol notes:
#   * Only `--random_seed` varies. `VAL_SEED` stays 42 inside
#     prompt_ablation_A_cell.sh, because the held-out validation split must be
#     IDENTICAL across seeds -- otherwise the three runs would not be selecting
#     E* against the same set and mean±std would be meaningless.
#   * Each seed re-runs the full P0 2-stage (stage-1 selects its own E*), which
#     is what invariant #3 requires; we are not reusing seed 42's E*.
#   * Configs are the per-dataset optima from the 2026-08-03 unified-recipe
#     sweep. CIFAR-10 is excluded: its bijection-OFF lambda sweep is still
#     running, so its optimum is not yet fixed.
#
# A GPU counts as free when memory.used < 2000 MiB and utilization < 15 % on
# two consecutive polls 30 s apart (transient dips are common mid-extraction).
set -u
cd /home/yschoi/GroundedDNA
mkdir -p logs

# exp : tag : aux-args-without-seed
CELLS=(
  "mscoco_A_v5b:uni003:--lambda_codon_joint 0.03 --no_gumbel_softmax"
  "nuswide_A_v4:uni005:--lambda_codon_joint 0.05 --no_gumbel_softmax"
  "flickr_A_v4:uni002:--lambda_codon_joint 0.02 --no_gumbel_softmax"
)
SEEDS=(43 44)

QUEUE=()
for s in "${SEEDS[@]}"; do
  for c in "${CELLS[@]}"; do QUEUE+=("$c:$s"); done
done
echo "[queue] ${#QUEUE[@]} cells to run @ $(date '+%F %T')"

# GPUs this queue is ALLOWED to use. The baseline multi-seed matrix owns its own
# GPUs and reports them free between cells, so without an allowlist this queue
# steals them and the two jobs fight (observed 2026-08-03).
ALLOWED="${ALLOWED_GPUS:-0 1 2 3}"

free_gpus() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
    | awk -F', ' '($2+0) < 2000 && ($3+0) < 15 {print $1}' \
    | grep -xE "$(echo $ALLOWED | tr ' ' '|')" | sort -n
}

BUSY=()   # gpu -> pid of the cell we launched
declare -A RUNNING

i=0
while [ "$i" -lt "${#QUEUE[@]}" ]; do
  # reap finished
  for g in "${!RUNNING[@]}"; do
    if ! kill -0 "${RUNNING[$g]}" 2>/dev/null; then unset "RUNNING[$g]"; fi
  done
  g1=$(free_gpus); sleep 30; g2=$(free_gpus)
  COMMON=$(comm -12 <(echo "$g1") <(echo "$g2"))
  for g in $COMMON; do
    [ -n "${RUNNING[$g]:-}" ] && continue
    [ "$i" -ge "${#QUEUE[@]}" ] && break
    IFS=: read -r exp tag aux seed <<<"${QUEUE[$i]}"
    T="${tag}_s${seed}"
    echo "[queue] GPU$g <- $exp $T @ $(date '+%F %T')"
    setsid nohup bash scripts/sweep_joint_cell.sh "$g" "$exp" "$T" \
      "$aux --random_seed $seed" > "logs/SWEEP_${exp}_${T}.out" 2>&1 < /dev/null &
    RUNNING[$g]=$!
    i=$((i+1))
    sleep 60          # stagger launches; concurrent cold starts have crashed before
  done
  sleep 30
done
echo "[queue] all ${#QUEUE[@]} cells dispatched @ $(date '+%F %T')"
