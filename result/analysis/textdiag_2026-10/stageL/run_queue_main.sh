#!/usr/bin/env bash
# Run cell command files sequentially on ONE GPU and record them the way the main tree's
# result/ folder does (user instruction 2026-10-09):
#   * the finished run directory is MOVED to /home/yschoi/GroundedDNA/result/<run dir name>
#     (same filesystem, no copy; no symlink),
#   * <stage analysis dir>/<tag>.log  = the cell's stdout/stderr,
#   * <stage analysis dir>/cells.txt  = "<rc>|<tag>|<command>" (one line per finished cell),
#   * <stage analysis dir>/eval_runs.txt = "<dataset> <arm> <seed> <run dir>" (rc 0 only).
# Every cell runs under a hard timeout (a hung DataLoader cost 2 h on 2026-10-08).
# Usage: run_queue_main.sh <gpu> <stage_analysis_dir> <timeout_s> [--wait-status <file> <max_wait_s>] <cmdfile>...
set -u
GPU=$1; STAGE=$2; TMO=$3; shift 3
WT=/home/yschoi/gdna_textdiag
MAIN_RESULT=/home/yschoi/GroundedDNA/result
PY=/home/yschoi/.conda/envs/dna_hashing/bin/python
mkdir -p "$STAGE"
if [ "${1:-}" = "--wait-status" ]; then
  WS=$2; MAXW=$3; shift 3; t0=$(date +%s)
  until [ -s "$WS" ] && grep -q '^rc=' "$WS"; do
    if [ $(( $(date +%s) - t0 )) -gt "$MAXW" ]; then echo "WARN: $WS not finished after ${MAXW}s; starting anyway"; break; fi
    sleep 30
  done
fi
cd "$WT" || exit 90
rc_all=0
for f in "$@"; do
  tag=$(basename "$f" .cmd)
  if ls -d "$MAIN_RESULT"/*"_${tag}+"* >/dev/null 2>&1 || ls -d "$WT"/result/*"_${tag}+"*/model_state_dict.pth >/dev/null 2>&1; then
    echo "$tag skip (exists)"; continue
  fi
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
  echo "=== $tag start $(date '+%F %T') gpu=$GPU used=${used}MiB timeout=${TMO}s"
  CMD=$(cat "$f")
  (
    echo "GPU=$GPU"; echo "CMD=$CMD"; echo "HEAD=$(git rev-parse --short HEAD)"; date
    # `env` execs python, so timeout signals the trainer itself (not a wrapper shell).
    timeout --kill-after=60 "$TMO" env -u PYTHONPATH CUDA_VISIBLE_DEVICES="$GPU" GDNA_NUM_SEMANTIC_PARTS=5 \
      /home/yschoi/.conda/envs/dna_hashing/bin/${CMD}
    rc=$?
    echo "rc=$rc"; date
    exit "$rc"
  ) > "$STAGE/$tag.log" 2>&1
  rc=$?
  ( flock 9; echo "$rc|$tag|$CMD" >> "$STAGE/cells.txt" ) 9>"$STAGE/.lock"
  echo "=== $tag rc=$rc $(date '+%F %T')"
  if [ "$rc" -eq 0 ]; then
    src=$(ls -d "$WT"/result/*"_${tag}+"* 2>/dev/null | head -1)
    if [ -n "$src" ] && [ -f "$src/model_state_dict.pth" ] && [ ! -e "$MAIN_RESULT/$(basename "$src")" ]; then
      mv "$src" "$MAIN_RESULT/" && dst="$MAIN_RESULT/$(basename "$src")"
      ds=$(echo "$tag" | sed -E 's/^td1_(flickr|nus|coco)_/\1 /; s/^td3_/flickr /' | awk '{print $1}')
      ds=$( [ "$ds" = flickr ] && echo flickr25k || echo "$ds")
      arm=$(echo "$tag" | sed -E 's/^td1_[a-z]+_//; s/^td3_//; s/_s[0-9]+$//')
      seed=$(echo "$tag" | sed -E 's/.*_s([0-9]+)$/\1/')
      ( flock 9; echo "$ds $arm $seed $dst" >> "$STAGE/eval_runs.txt" ) 9>"$STAGE/.lock"
      echo "    moved -> $dst"
    else
      echo "    WARN: run dir not moved (src='$src')"; rc_all=1
    fi
  else
    rc_all=1
  fi
done
exit "$rc_all"
