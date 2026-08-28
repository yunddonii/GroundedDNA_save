#!/usr/bin/env bash
# Run the A-series campaign: plan -> reserve -> 24 cells -> seal.
#
# This lived inside auto_chain_after_3seed.sh, which exits 91 at line 8 under
# the protocol audit's stop order. That made the whole loop unreachable, so
# nothing about it could be executed -- and the re-audit was right to call the
# tests around it evidence about a file rather than about a launcher. It is a
# separate program now: the chain calls it, and a test can too.
#
# It launches nothing on its own initiative. Every cell is a line in a plan that
# was written by scripts/ablation_campaign_plan.py and refused to be written at
# all if any input could not be accounted for.
#
# Usage:
#   scripts/run_ablation_campaign.sh --plan P --ledger L [--dry-run]
#   GPUS="0 1" scripts/run_ablation_campaign.sh ...   # else nvidia-smi decides
set -Eeuo pipefail
trap 'rc=$?; printf "[campaign] FAILED at line %s (rc %s): %s\n" \
        "$LINENO" "$rc" "$BASH_COMMAND" >&2; exit "$rc"' ERR

PY="${PY:-/home/yschoi/.conda/envs/dna_hashing/bin/python}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLAN=""; LEDGER=""; DRY=""; STAGGER="${STAGGER:-150}"
# Where the plan's runner paths are rooted. The cells name their runner
# relative to the repository, and the executor is what changes directory, so
# this has to be passed through rather than inherited from $PWD.
CHILD_REPO=""
IDLE_MIB="${IDLE_MIB:-200}"
LOGDIR="${LOGDIR:-logs}"

while [ $# -gt 0 ]; do
  case "$1" in
    --plan)    PLAN="$2"; shift 2 ;;
    --ledger)  LEDGER="$2"; shift 2 ;;
    --repo)    CHILD_REPO="$2"; shift 2 ;;
    --dry-run) DRY="1"; shift ;;
    *) echo "[campaign] unknown argument $1" >&2; exit 2 ;;
  esac
done
[ -n "$PLAN" ] && [ -n "$LEDGER" ] || {
    echo "[campaign] --plan and --ledger are required" >&2; exit 2; }

CHILD_REPO="${CHILD_REPO:-$REPO}"
say(){ echo "[campaign $(date '+%F %T')] $*"; }
mkdir -p "$LOGDIR"

# ---- the plan is read ONCE, and hashed once --------------------------------
# Re-reading a mutable file per cell means the twenty-fourth cell can run a plan
# the first cell never saw. The executor is handed this digest and refuses a
# mismatch, so an edit between here and the last cell stops the campaign.
PLAN_SHA=$("$PY" -c \
    "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" \
    "$PLAN")

# NOT `mapfile -t < <(producer)`: that checks mapfile's status and never the
# producer's, so a producer that prints rows and then dies is accepted. A temp
# file makes the producer's status the only status.
ROWS_FILE=$(mktemp)
trap 'rm -f "$ROWS_FILE"' EXIT
"$PY" -c "
import json, sys
plan = json.load(open(sys.argv[1]))
for cell in plan['cells']:
    print(cell['cell'], cell['exp'], cell['tag'], sep='\t')
" "$PLAN" > "$ROWS_FILE" || { say "could not read the plan's cells"; exit 1; }

ROWS=(); while IFS= read -r row; do ROWS+=("$row"); done < "$ROWS_FILE"
NCELLS=${#ROWS[@]}
EXPECTED=$("$PY" -c \
    "import json,sys;print(json.load(open(sys.argv[1]))['expected_cells'])" "$PLAN")
[ "$NCELLS" = "$EXPECTED" ] || {
    say "plan has $NCELLS cells, expected $EXPECTED"; exit 1; }
say "$NCELLS cells planned, plan ${PLAN_SHA:0:12}"

# ---- reserve the whole set before starting any of it -----------------------
"$PY" scripts/campaign_ledger.py open --ledger "$LEDGER" --plan "$PLAN" || {
    say "campaign not reserved; starting nothing"; exit 1; }

# ---- GPUs ------------------------------------------------------------------
free_gpus(){
    if [ -n "${GPUS:-}" ]; then printf '%s ' $GPUS; return; fi
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | awk -F', ' -v t="$IDLE_MIB" '$2+0 < t {printf "%s ", $1}'
}
next_gpu(){
    local g
    while :; do
        g=$(free_gpus); g=${g%% *}
        [ -n "$g" ] && { echo "$g"; return; }
        sleep 60
    done
}

# ---- launch ----------------------------------------------------------------
declare -a PIDS=() TAGS=() NAMES=()
for i in $(seq 0 $((NCELLS - 1))); do
    IFS=$'\t' read -r CELL EXP TAG <<<"${ROWS[$i]}"
    TAGS[$i]="$TAG"; NAMES[$i]="${CELL}_${EXP}"
    g=$(next_gpu)
    say "  $CELL $EXP -> GPU $g"
    "$PY" scripts/_ablation_exec.py --plan "$PLAN" --index "$i" --gpu "$g" \
        --expect-plan-sha256 "$PLAN_SHA" --repo "$CHILD_REPO" \
        ${DRY:+--dry-run} \
        > "$LOGDIR/ABL_${CELL}_${EXP}.out" 2>&1 &
    PIDS[$i]=$!
    [ "$STAGGER" = "0" ] || sleep "$STAGGER"
done

# ---- collect: `wait` is the only place a child's status exists -------------
FAILED=0
for i in $(seq 0 $((NCELLS - 1))); do
    rc=0; wait "${PIDS[$i]}" || rc=$?
    if [ "$rc" = "0" ]; then
        say "  ${NAMES[$i]} ok"
        "$PY" scripts/campaign_ledger.py record --ledger "$LEDGER" \
            --tag "${TAGS[$i]}" --status ok
    else
        say "  ${NAMES[$i]} FAILED rc=$rc (see $LOGDIR/ABL_${NAMES[$i]}.out)"
        "$PY" scripts/campaign_ledger.py record --ledger "$LEDGER" \
            --tag "${TAGS[$i]}" --status failed --detail "rc=$rc"
        FAILED=1
    fi
done

# The seal is the gate, not the flag. A campaign that never started a cell has
# FAILED=0 and must still not become a table.
"$PY" scripts/campaign_ledger.py seal --ledger "$LEDGER" || {
    say "INCOMPLETE -- no completion receipt; nothing downstream may read these cells"
    exit 1; }
[ "$FAILED" = "0" ] || { say "INCOMPLETE"; exit 1; }
say "done: $NCELLS cells sealed"
