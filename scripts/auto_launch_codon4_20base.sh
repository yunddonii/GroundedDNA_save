#!/usr/bin/env bash
# 4-base codon (5 slots x 4 bases = 20 bases = 40 bit), ours + baselines.
#
# Standing instruction (2026-08-12): once the N sweep and the two baseline
# backlogs are done, run this without waiting for a further go-ahead. The paper
# keeps the 3-base codon as its main axis and reports 4-base alongside it, so
# these numbers fill a panel that already exists in the draft with `-` cells.
#
# N per dataset is read from docs/newmodel_analysis/fixed_N.json, written when
# the N sweep is settled. `-e` is set to N+1 so the Sinkhorn epsilon schedule
# completes inside the run -- at 4 bases the codon-joint prior spans 4^4 = 256
# cells instead of 64, which needs the plan to be sharp to be reachable at all.
#
# GC window: the DP projection takes FRACTIONS, so [0.416, 0.584] at 20 bases
# gives the integer window [9, 11]; the cell script already branches on
# NUM_CODONS=4. Do not hand it the 3-base [0.40, 0.60].
set -u
cd /home/yschoi/GroundedDNA
PY=/home/yschoi/.conda/envs/dna_hashing/bin/python
IDLE_MIB="${IDLE_MIB:-200}"
LOG=logs/AUTO_CODON4.log
mkdir -p logs
say() { echo "[codon4 $(date '+%F %T')] $*" | tee -a "$LOG"; }
free_gpus() {
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
      | awk -F', ' -v t="$IDLE_MIB" '$2+0 < t {printf "%s ", $1}'
}

NJSON=docs/newmodel_analysis/fixed_N.json
[ -f "$NJSON" ] || { say "ABORT: $NJSON missing -- settle N first"; exit 2; }

say "waiting for the N sweep and the baseline backlogs"
while pgrep -f "prompt_ablation_A_cell_fixedN.sh|run_native_dna_p0_matrix.py|run_baseline_p0_matrix.py" \
      > /dev/null; do sleep 180; done
say "prerequisites clear"

S5="--num_semantic_parts 5 --num_codebooks 5 --no_gumbel_softmax --lambda_codeword_codon_sinkhorn 0.0"
T69="--routing_adaptive_topp_min 0.6 --routing_adaptive_topp_max 0.95"

# ---- 1) ours at 4 bases/codon --------------------------------------------
for EXP in cifar_A_v4 flickr_A_v4 nuswide_A_v4 mscoco_A_v5b; do
    N=$("$PY" -c "import json;print(json.load(open('$NJSON'))['$EXP']['N'])" 2>/dev/null)
    JD=$("$PY" -c "import json;print(json.load(open('$NJSON'))['$EXP']['lambda_codon_joint'])" 2>/dev/null)
    [ -n "${N:-}" ] || { say "SKIP $EXP: no N in $NJSON"; continue; }
    while :; do G=$(free_gpus); [ -n "${G// /}" ] && break; sleep 180; done
    g=${G%% *}
    say "ours 4-base $EXP  N=$N -e $((N+1))  -> GPU $g"
    GDNA_NUM_SEMANTIC_PARTS=5 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    FIXED_N="$N" EVERY=1 NUM_CODONS=4 \
    AUX_ARGS="$S5 --num_codons_per_codebook 4 --lambda_codon_joint $JD $T69 -e $((N+1))" \
    setsid nohup bash scripts/prompt_ablation_A_cell_fixedN.sh "$g" "$EXP" \
        > "logs/CODON4_${EXP}.out" 2>&1 < /dev/null &
    sleep 120
done
while pgrep -f "prompt_ablation_A_cell_fixedN.sh" > /dev/null; do sleep 180; done
say "ours 4-base done"

# ---- 2) the same U0 control set at 40 bit ---------------------------------
G=$(free_gpus)
if [ -n "${G// /}" ]; then
    say "U0 40-bit: idle GPUs = [$G]"
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    setsid nohup "$PY" scripts/run_baseline_p0_matrix.py \
        --variants cibhash cimon sdc-paper oh crovca bihalf \
        --datasets Flickr25k MSCOCO NUSWIDE CIFAR10 \
        --bits 40 --seeds 42 --gpus $G \
        > logs/CODON4_u0_40b.out 2>&1 < /dev/null &
else
    say "SKIPPED U0 40-bit: no idle GPU"
fi
say "all launches issued"
