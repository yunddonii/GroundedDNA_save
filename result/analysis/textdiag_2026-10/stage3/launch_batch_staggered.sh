#!/usr/bin/env bash
# Launch one cell per GPU with a stagger, each in its own tmux session. Usage: launch_batch_staggered.sh <stagger_s> <gpu:cmdfile> [<gpu:cmdfile> ...]
set -u; cd /home/yschoi/gdna_textdiag || exit 90
ST=$1; shift; T=/data/yschoi/gdna_p3exec_authority/bin/tmux_run.sh; R=result/analysis/textdiag_2026-10/stage1/run_cell.sh
for spec in "$@"; do
  g=${spec%%:*}; f=${spec#*:}; tag=$(basename "$f" .cmd)
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
  [ "$used" -gt 1000 ] && { echo "GPU $g busy ($used MiB) -> skip $tag"; continue; }
  $T "$tag" bash "$PWD/$R" "$g" "$PWD/$f" | head -1
  sleep "$ST"
done
