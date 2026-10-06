#!/usr/bin/env bash
# Run a list of cell cmd files sequentially on ONE gpu. Usage: run_queue.sh <gpu> <cmdfile> [<cmdfile> ...]
# A cell whose result dir already has log.csv with the final epoch is skipped (resume after a drop).
set -u
GPU=$1; shift
cd /home/yschoi/gdna_textdiag || exit 90
rc_all=0
for f in "$@"; do
  tag=$(basename "$f" .cmd)
  if ls -d result/*"+$tag+"* >/dev/null 2>&1 && [ -s "$(ls -d result/*"+$tag+"*/log.csv 2>/dev/null | head -1)" ] && [ -f "$(ls -d result/*"+$tag+"* | head -1)/model_state_dict.pth" ]; then
    echo "$tag skip (exists)"; continue
  fi
  echo "=== $tag start $(date)"
  bash result/analysis/textdiag_2026-10/stage1/run_cell.sh "$GPU" "$f" > "result/analysis/textdiag_2026-10/stage1/logs/$tag.log" 2>&1
  rc=$?; echo "=== $tag rc=$rc $(date)"; [ "$rc" -eq 0 ] || rc_all=1
done
exit "$rc_all"
