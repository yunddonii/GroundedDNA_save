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
#
# `--dry-run` prints each cell's command and DOES NOT touch the ledger. It used
# to pass the flag to the executor and then record every rc 0 as `status=ok`
# with a null run directory, so a run that started no trainer and claimed no GPU
# sealed a `campaign_complete.json` naming 24 cells. I did that with the real
# plan and then wrote "planned, reserved, executed and sealed" in the project
# log; nothing had been executed. A receipt now means trainers ran.
set -Eeuo pipefail
trap 'rc=$?; printf "[campaign] FAILED at line %s (rc %s): %s\n" \
        "$LINENO" "$rc" "$BASH_COMMAND" >&2; exit "$rc"' ERR

PY="${PY:-/home/yschoi/.conda/envs/dna_hashing/bin/python}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Exactly one result per tag, or refuse. Same helper the wrapper uses.
source "$REPO/scripts/lib/result_dir.sh"
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

# Everything below runs inside `main`, and `main` is called on the last line.
# Bash reads a script INCREMENTALLY at the top level: editing this file while a
# campaign is running corrupts the part that has not been read yet. That is not
# hypothetical -- during the top-p smoke I edited the GPU allocator while the
# three cells were running, and the seal step died with "syntax error near
# unexpected token `('" after all three cells had succeeded. A function is
# parsed in full before any of it executes, so an edit mid-run is ignored.
main(){
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
    print(cell['cell'], cell['exp'], cell['tag'], cell['N'], sep='\t')
" "$PLAN" > "$ROWS_FILE" || { say "could not read the plan's cells"; exit 1; }

ROWS=(); while IFS= read -r row; do ROWS+=("$row"); done < "$ROWS_FILE"
NCELLS=${#ROWS[@]}
EXPECTED=$("$PY" -c \
    "import json,sys;print(json.load(open(sys.argv[1]))['expected_cells'])" "$PLAN")
[ "$NCELLS" = "$EXPECTED" ] || {
    say "plan has $NCELLS cells, expected $EXPECTED"; exit 1; }
say "$NCELLS cells planned, plan ${PLAN_SHA:0:12}"

# ---- reserve the whole set before starting any of it -----------------------
if [ -n "$DRY" ]; then
    say "DRY RUN: printing commands only. No reservation, no receipt."
    for i in $(seq 0 $((NCELLS - 1))); do
        IFS=$'\t' read -r CELL EXP TAG <<<"${ROWS[$i]}"
        "$PY" scripts/_ablation_exec.py --plan "$PLAN" --index "$i" --gpu 0 \
            --expect-plan-sha256 "$PLAN_SHA" --repo "$CHILD_REPO" --dry-run \
            > "$LOGDIR/DRY_${CELL}_${EXP}.json"
    done
    say "DRY RUN: $NCELLS commands written to $LOGDIR/DRY_*.json"
    exit 0
fi

"$PY" scripts/campaign_ledger.py open --ledger "$LEDGER" --plan "$PLAN" || {
    say "campaign not reserved; starting nothing"; exit 1; }

# ---- GPUs ------------------------------------------------------------------
candidate_gpus(){
    if [ -n "${GPUS:-}" ]; then printf '%s ' $GPUS; return; fi
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | awk -F', ' -v t="$IDLE_MIB" '$2+0 < t {printf "%s ", $1}'
}

# A GPU this launcher has handed to a child that is still alive. Without this,
# `next_gpu` took `${g%% *}` -- the FIRST token of the candidate list -- every
# time, so `GPUS="0 1 2"` put all three cells on GPU 0. Observed, not theorised:
# the three-cell smoke landed entirely on GPU 0 while 1-5 sat idle. The
# nvidia-smi path had the same hole for as long as it takes a child's memory to
# show up, which is longer than the stagger.
declare -A LEASED=()
# `kill -0` is NOT the liveness test here. Children are waited for only in the
# collect loop below, so a cell that has finished during the launch loop is a
# ZOMBIE -- and `kill -0` succeeds on a zombie. The lease was therefore never
# released while launching, and a campaign with more cells than GPUs blocked
# forever on the first full pass. `/proc/<pid>/stat` field 3 is the state, and
# a zombie reports Z.
still_running(){
    local state
    state=$(awk '{print $3}' "/proc/$1/stat" 2>/dev/null) || return 1
    [ "$state" != "Z" ]
}
reap_leases(){
    local g
    for g in "${!LEASED[@]}"; do
        still_running "${LEASED[$g]}" || unset 'LEASED['"$g"']'
    done
}
next_gpu(){
    local g
    while :; do
        reap_leases
        for g in $(candidate_gpus); do
            if [ -z "${LEASED[$g]:-}" ]; then echo "$g"; return; fi
        done
        sleep "${GPU_WAIT:-30}"
    done
}

# ---- launch ----------------------------------------------------------------
declare -a PIDS=() TAGS=() NAMES=() NS=()
for i in $(seq 0 $((NCELLS - 1))); do
    IFS=$'\t' read -r CELL EXP TAG N <<<"${ROWS[$i]}"
    TAGS[$i]="$TAG"; NAMES[$i]="${CELL}_${EXP}"; NS[$i]="$N"
    g=$(next_gpu)
    say "  $CELL $EXP -> GPU $g"
    "$PY" scripts/_ablation_exec.py --plan "$PLAN" --index "$i" --gpu "$g" \
        --expect-plan-sha256 "$PLAN_SHA" --repo "$CHILD_REPO" \
        > "$LOGDIR/ABL_${CELL}_${EXP}.out" 2>&1 &
    PIDS[$i]=$!
    LEASED[$g]=$!
    [ "$STAGGER" = "0" ] || sleep "$STAGGER"
done

# ---- collect: `wait` is the only place a child's status exists -------------
FAILED=0
for i in $(seq 0 $((NCELLS - 1))); do
    rc=0; wait "${PIDS[$i]}" || rc=$?
    if [ "$rc" = "0" ]; then
        # The ledger used to accept `ok` with no run directory, and this loop
        # never passed one -- so a real campaign sealed a receipt whose every
        # output was null. The wrapper names the directory it evaluated; this
        # resolves the same tag and refuses if it is not exactly one.
        RD=""
        RD=$(resolve_one_result_dir "${TAGS[$i]}_P0refit_e${NS[$i]}" \
             "$CHILD_REPO/result") || RD=""
        if [ -z "$RD" ]; then
            say "  ${NAMES[$i]} exited 0 but its result directory could not be resolved"
            "$PY" scripts/campaign_ledger.py record --ledger "$LEDGER" \
                --tag "${TAGS[$i]}" --status failed --detail "unresolvable result dir"
            FAILED=1
            continue
        fi
        say "  ${NAMES[$i]} ok  ->  $RD"
        "$PY" scripts/campaign_ledger.py record --ledger "$LEDGER" \
            --tag "${TAGS[$i]}" --status ok --run-dir "$RD"
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
}

main "$@"
