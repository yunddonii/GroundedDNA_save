#!/usr/bin/env bash
# Wait for genuinely-free GPUs, then launch the 4 "A + prompt" cells (18-base).
# A GPU counts as free = memory.used < 3000 MiB AND utilization < 15 %, confirmed
# on two consecutive polls 30 s apart (avoids transient dips). Max wait 12 h.
set -u
cd /home/yschoi/GroundedDNA
mkdir -p logs
MAXWAIT=$((12*3600)); START=$(date +%s)

free_gpus() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
    | awk -F', ' '($2+0) < 3000 && ($3+0) < 15 {print $1}' | sort -n
}

echo "[watcher] polling for free GPUs @ $(date '+%F %T')"
COMMON=""
while :; do
  now=$(date +%s); [ $((now-START)) -ge $MAXWAIT ] && { echo "[watcher] TIMEOUT 12h, giving up @ $(date '+%F %T')"; exit 2; }
  g1=$(free_gpus); sleep 30; g2=$(free_gpus)
  COMMON=$(comm -12 <(echo "$g1") <(echo "$g2"))
  n=$(echo "$COMMON" | grep -c '[0-9]')
  if [ "$n" -ge 2 ]; then echo "[watcher] $n free GPUs stable: $(echo $COMMON) @ $(date '+%F %T')"; break; fi
  sleep 30
done

mapfile -t G < <(echo "$COMMON" | grep '[0-9]')
N=${#G[@]}
echo "[watcher] launching 4 cells on GPUs: ${G[*]}"

if [ "$N" -ge 3 ]; then
  bash scripts/prompt_ablation_A_cell.sh "${G[0]}" mscoco_A_v5b > logs/promptAblA_mscoco_A_v5b_master.log 2>&1 &
  bash scripts/prompt_ablation_A_cell.sh "${G[1]}" mscoco_A_v4  > logs/promptAblA_mscoco_A_v4_master.log  2>&1 &
  ( bash scripts/prompt_ablation_A_cell.sh "${G[2]}" cifar_A_v1 && \
    bash scripts/prompt_ablation_A_cell.sh "${G[2]}" cifar_A_v4 ) > logs/promptAblA_cifar_master.log 2>&1 &
else   # exactly 2 free: two lanes, each MSCOCO then CIFAR
  ( bash scripts/prompt_ablation_A_cell.sh "${G[0]}" mscoco_A_v5b && \
    bash scripts/prompt_ablation_A_cell.sh "${G[0]}" cifar_A_v1 ) > logs/promptAblA_lane0.log 2>&1 &
  ( bash scripts/prompt_ablation_A_cell.sh "${G[1]}" mscoco_A_v4  && \
    bash scripts/prompt_ablation_A_cell.sh "${G[1]}" cifar_A_v4 ) > logs/promptAblA_lane1.log 2>&1 &
fi
wait

echo "[watcher] ALL CELLS DONE @ $(date '+%F %T')"
echo "=== [CELL-RESULT] ==="
grep -h "\[CELL-RESULT\]" logs/promptAblA_*.log 2>/dev/null | sort -u
echo "=== failures ==="
grep -hE "ERROR|MISSING|Traceback" logs/promptAblA_*.log 2>/dev/null | head || echo none
