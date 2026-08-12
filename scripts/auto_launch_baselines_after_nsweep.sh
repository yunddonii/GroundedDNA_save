#!/usr/bin/env bash
# Wait for the running N-sweep cells, then start the two baseline backlogs on
# whatever GPUs are genuinely idle.
#
# Standing instruction from the user (2026-08-12): once the N sweep finishes,
# start the native-DNA 15-base re-adaptation and the three missing U0 30-bit
# methods WITHOUT waiting for a further go-ahead.
#
# GPU choice is measured, not assumed. This machine runs other projects, so the
# script takes only devices reporting under IDLE_MIB and re-checks between the
# two launches; stacking onto a busy GPU can OOM a run this script did not start
# and cannot see.
set -u
cd /home/yschoi/GroundedDNA
PY=/home/yschoi/.conda/envs/dna_hashing/bin/python
IDLE_MIB="${IDLE_MIB:-200}"
LOG=logs/AUTO_BASELINES.log
mkdir -p logs
say() { echo "[auto $(date '+%F %T')] $*" | tee -a "$LOG"; }

free_gpus() {   # -> space-separated indices currently under IDLE_MIB
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
      | awk -F', ' -v t="$IDLE_MIB" '$2+0 < t {printf "%s ", $1}'
}

say "waiting for N-sweep cells to finish"
while pgrep -f "prompt_ablation_A_cell_fixedN.sh" > /dev/null; do sleep 120; done
say "N sweep done"

# ---- 1) native-DNA baselines, 15 bases ------------------------------------
# 4 methods x 4 datasets x 3 seeds = 48 cells. The matrix launcher schedules
# them across the GPUs it is given, so it gets the idle set in one go.
#
# --allow-main-ineligible-diagnostic is REQUIRED and is not a shortcut: the
# shared visual cache lacks strict provenance metadata, so every cell is gated
# as `legacy_cache_missing_strict_provenance`. The 18-base sealed matrix hit the
# same gate and is recorded in PROJECT_LOG as 48 complete_diagnostic_only /
# 0 strict-main eligible. The 15-base results inherit that status and must be
# reported as a diagnostic table, never as main-comparison rows.
G=$(free_gpus)
say "native-DNA 15-base: idle GPUs = [${G:-none}]"
if [ -n "${G// /}" ]; then
    # --data-root is an OUTPUT root and the launcher requires an absolute path
    # under /data. A NEW directory, not the 18-base sealed one: run dirs are
    # named ..._{n}nt_... so they would not collide, but keeping the budgets in
    # separate roots means the aggregator cannot silently mix them.
    # A FRESH root per attempt. The launcher refuses to overwrite existing
    # launcher logs/status, and mixing attempts is worse than the disk cost:
    # cells written before a protocol fix carry the old label and the
    # aggregator rejects the whole matrix rather than the stale cells.
    ND_ROOT=/data/yschoi/groundeddna_native_p0_15base
    if [ -d "$ND_ROOT/logs/status" ] && \
       [ "$(ls -1 "$ND_ROOT/logs/status" 2>/dev/null | wc -l)" -gt 0 ]; then
        STAMP=$(date +%Y%m%dT%H%M%S)
        say "archiving previous attempt -> ${ND_ROOT}_attempt_${STAMP}"
        mv "$ND_ROOT" "${ND_ROOT}_attempt_${STAMP}"
    fi
    mkdir -p "$ND_ROOT/artifacts"
    # bee2021 (PRIMO) needs the yield predictor; reuse the one already fitted.
    PRIMO=/data/yschoi/groundeddna_native_p0/artifacts/primo_yield_predictor.npz
    GDNA_NATIVE_DNA_BASES=15 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    setsid nohup "$PY" scripts/run_native_dna_p0_matrix.py \
        --data-root "$ND_ROOT" --gpus $G \
        ${PRIMO:+--primo-predictor-npz "$PRIMO"} \
        --methods bee2018 bee2021 koike2024 koike2026 \
        --datasets Flickr25k MSCOCO NUSWIDE CIFAR10 \
        --seeds 42 43 44 \
        --allow-main-ineligible-diagnostic \
        > logs/AUTO_nativedna_15base.out 2>&1 < /dev/null &
    say "launched native-DNA matrix (48 cells) -> logs/AUTO_nativedna_15base.out"
    sleep 90
else
    say "SKIPPED native-DNA: no idle GPU"
fi

# ---- 2) the three U0 methods still missing at 30 bit ----------------------
# Panel A has 9 U0 variants; 6 were re-run at 30 bit, these 3 were not.
G=$(free_gpus)
say "U0 30-bit (mls3rduh, greedyhash, hhch): idle GPUs = [${G:-none}]"
if [ -n "${G// /}" ]; then
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    setsid nohup "$PY" scripts/run_baseline_p0_matrix.py \
        --variants mls3rduh greedyhash hhch \
        --datasets Flickr25k MSCOCO NUSWIDE CIFAR10 \
        --bits 30 --seeds 42 --gpus $G \
        > logs/AUTO_u0_30b.out 2>&1 < /dev/null &
    say "launched U0 30-bit matrix -> logs/AUTO_u0_30b.out"
else
    say "SKIPPED U0: no idle GPU"
fi
say "all launches issued"
