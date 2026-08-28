#!/usr/bin/env bash
# Run the 16 selection cells, four datasets in parallel, one GPU each.
#
# The launcher itself is per-cell and sequential by design -- it refuses a cell
# whose tag already has artefacts, and each cell must finish before its record
# is written. This spreads the four datasets across four GPUs and keeps each
# dataset's four N candidates in sequence, so at most one process ever writes a
# given dataset's namespace.
#
# Each dataset's exit status is collected. A bare `wait` returns 0 whatever the
# children did, which is how an earlier launcher reported "all datasets
# finished" after one had died.
#
# Usage: scripts/phase3_launch_matrix.sh [gpu0 gpu1 gpu2 gpu3]
set -uo pipefail
cd /home/yschoi/GroundedDNA
PY=/home/yschoi/.conda/envs/dna_hashing/bin/python
mkdir -p logs/phase3_matrix

DEPS=(scripts/phase3_selection_matrix.py scripts/phase3_select_n.py
      train_siglip2.py model_siglip2.py dna_utils/run_identity.py
      dna_utils/extraction_validation.py dataloaders.py)
if [[ -n "$(git status --porcelain "${DEPS[@]}")" ]]; then
    echo "refusing: the matrix source is dirty; commit it first" >&2
    git status --short "${DEPS[@]}" >&2
    exit 2
fi
HEAD_SHA=$(git rev-parse HEAD)
echo "[matrix] source commit $HEAD_SHA"

DATASETS=(cifar10 flickr25k nuswide mscoco)
GPUS=(${1:-0} ${2:-1} ${3:-2} ${4:-3})
declare -A PIDS=()

for i in "${!DATASETS[@]}"; do
    DS="${DATASETS[$i]}"; GPU="${GPUS[$i]}"
    LOG="logs/phase3_matrix/${DS}.log"
    echo "[matrix] $DS on gpu $GPU -> $LOG"
    (
        for N in 4 9 19 39; do
            "$PY" scripts/phase3_selection_matrix.py --run \
                --only "${DS}:${N}" --gpu "$GPU" || exit 1
        done
    ) > "$LOG" 2>&1 &
    PIDS[$DS]=$!
done

FAILED=0
for DS in "${!PIDS[@]}"; do
    if wait "${PIDS[$DS]}"; then
        echo "  $DS ok"
    else
        echo "  $DS FAILED (see logs/phase3_matrix/${DS}.log)" >&2
        FAILED=1
    fi
done

SEALED=$(ls artifacts/phase3_selection/*.json 2>/dev/null | grep -vc selected_n || true)
echo "[matrix] $SEALED of 16 cell records present"
[[ "$FAILED" == "0" && "$SEALED" == "16" ]] || {
    echo "MATRIX INCOMPLETE $HEAD_SHA" >&2; exit 1; }
echo "MATRIX DONE $HEAD_SHA"
