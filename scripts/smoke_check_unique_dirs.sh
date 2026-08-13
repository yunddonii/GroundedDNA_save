#!/usr/bin/env bash
# Smoke test: does each cell of a batch land in its OWN result directory?
#
# Standing rule (2026-08-12): run this before committing GPU hours to a batch.
#
# The failure it exists to catch is silent. `prompt_ablation_A_cell_fixedN.sh`
# builds the result directory as
#     promptAblA_{EXP}{TAG_SUFFIX}_P0refit_e{N}
# with no seed and no ablation marker, so two cells that differ only in
# --random_seed or in an ablation flag write to the SAME path and the second
# overwrites the first. Bash ignores an unknown env var without a word, so
# passing the wrong variable name (BASE_SUFFIX instead of TAG_SUFFIX) looks like
# it worked right up until the results are gone. That cost seed 43 on four
# datasets before it was noticed.
#
# Usage:
#   bash scripts/smoke_check_unique_dirs.sh <GPU> <EXP> "<SUFFIX1>|<ARGS1>" "<SUFFIX2>|<ARGS2>" ...
#
# Each cell trains for ONE epoch on the given EXP, then the script reports the
# directory each landed in, whether they are distinct, and whether args.txt
# actually carries the flags that were passed. Minutes, not hours.
set -u
GPU="$1"; EXP="$2"; shift 2
cd /home/yschoi/GroundedDNA
mkdir -p logs
S5="--num_semantic_parts 5 --num_codebooks 5 --no_gumbel_softmax"
STAMP=$(date +%H%M%S)

declare -a DIRS=() SUFFIXES=()
i=0
for CELL in "$@"; do
    SUF="${CELL%%|*}"; ARGS="${CELL#*|}"
    TAG="smoke${STAMP}_${SUF}"
    echo "[smoke] cell $i: TAG_SUFFIX=_$TAG  args: $ARGS"
    GDNA_NUM_SEMANTIC_PARTS=5 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    FIXED_N=0 EVERY=1 TAG_SUFFIX="_$TAG" \
    AUX_ARGS="$S5 -e 1 $ARGS" \
    bash scripts/prompt_ablation_A_cell_fixedN.sh "$GPU" "$EXP" \
        > "logs/SMOKE_${TAG}.out" 2>&1
    D=$(ls -dt result/*"${TAG}"* 2>/dev/null | head -1)
    DIRS+=("${D:-MISSING}"); SUFFIXES+=("$SUF")
    i=$((i+1))
done

echo
echo "[smoke] ---- result directories ----"
for k in "${!DIRS[@]}"; do
    printf "  %-16s %s\n" "${SUFFIXES[$k]}" "$(basename "${DIRS[$k]}" 2>/dev/null)"
done

UNIQ=$(printf '%s\n' "${DIRS[@]}" | sort -u | wc -l)
if [ "$UNIQ" -eq "${#DIRS[@]}" ] && ! printf '%s\n' "${DIRS[@]}" | grep -q MISSING; then
    echo "[smoke] PASS: ${#DIRS[@]} cells -> $UNIQ distinct directories"
else
    echo "[smoke] FAIL: ${#DIRS[@]} cells -> $UNIQ distinct directories."
    echo "[smoke] Cells would overwrite each other. Do NOT launch the batch."
    exit 1
fi

echo "[smoke] ---- args.txt carries the passed flags? ----"
for k in "${!DIRS[@]}"; do
    A="${DIRS[$k]}/args.txt"
    [ -f "$A" ] || { echo "  ${SUFFIXES[$k]}: no args.txt"; continue; }
    printf "  %-16s seed=%s lambda_bu=%s codebooks=%s\n" "${SUFFIXES[$k]}" \
        "$(grep -oE '^random_seed-+[0-9]+$' "$A" | grep -oE '[0-9]+$')" \
        "$(grep -oE '^lambda_bu-+[0-9.]+$'  "$A" | grep -oE '[0-9.]+$')" \
        "$(grep -oE '^num_codebooks-+[0-9]+$' "$A" | grep -oE '[0-9]+$')"
done
echo "[smoke] check the values above match what you passed, then launch."
