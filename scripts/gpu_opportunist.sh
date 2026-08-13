#!/usr/bin/env bash
# Keep a queue of cells and start each one the moment a GPU frees up.
#
# The chain and the reseed runner both sleep 90-150 s between checks and only
# ever hold one cell in flight, so a GPU that frees at the wrong moment sits
# idle for minutes. This drains a queue file instead: it polls every 20 s and
# launches as many cells as there are free devices.
#
#   QUEUE format, one cell per line, '#' comments and blanks ignored:
#     <label>|<command to run, with __GPU__ where the device index goes>
#
#   bash scripts/gpu_opportunist.sh <queue-file>
#
# Idle is MEASURED (< IDLE_MIB), never assumed: other projects share this
# machine and stacking onto a busy device can OOM a run this script did not
# start. A device is also held for HOLD_S after a launch so a slow-starting
# job is not double-booked before its memory shows up in nvidia-smi.
set -u
cd /home/yschoi/GroundedDNA
Q="${1:?usage: gpu_opportunist.sh <queue-file>}"
IDLE_MIB="${IDLE_MIB:-200}"
POLL_S="${POLL_S:-20}"
HOLD_S="${HOLD_S:-90}"
RESERVE="${RESERVE:-1}"
LOG="${LOG:-logs/GPU_OPPORTUNIST.log}"
mkdir -p logs
say(){ echo "[opp $(date '+%F %T')] $*" | tee -a "$LOG"; }

declare -A HELD=()
free_gpus(){
    local now; now=$(date +%s)
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
      | awk -F', ' -v t="$IDLE_MIB" '$2+0 < t {print $1}' \
      | while read -r g; do
            local until=${HELD[$g]:-0}
            [ "$now" -ge "$until" ] && echo "$g"
        done
}

mapfile -t CELLS < <(grep -vE '^\s*(#|$)' "$Q")
say "queue: ${#CELLS[@]} cells from $Q"
i=0
while [ "$i" -lt "${#CELLS[@]}" ]; do
    # Leave RESERVE devices for whatever else is running. The chain polls every
    # 120 s while this polls every 20 s, so without a reserve this would win
    # every race and starve it.
    mapfile -t FREE < <(free_gpus)
    TAKE=$(( ${#FREE[@]} - RESERVE ))
    [ "$TAKE" -lt 0 ] && TAKE=0
    for g in "${FREE[@]:0:$TAKE}"; do
        [ "$i" -lt "${#CELLS[@]}" ] || break
        LINE="${CELLS[$i]}"
        LABEL="${LINE%%|*}"; CMD="${LINE#*|}"
        CMD="${CMD//__GPU__/$g}"
        say "[$((i+1))/${#CELLS[@]}] $LABEL -> GPU $g"
        setsid nohup bash -c "$CMD" > "logs/OPP_${LABEL}.out" 2>&1 < /dev/null &
        HELD[$g]=$(( $(date +%s) + HOLD_S ))
        i=$((i+1))
    done
    [ "$i" -lt "${#CELLS[@]}" ] && sleep "$POLL_S"
done
say "all ${#CELLS[@]} cells issued; waiting for the last ones"
while pgrep -f "train_siglip2.py|train_native_dna_baseline" >/dev/null; do sleep 60; done
say "queue drained"
