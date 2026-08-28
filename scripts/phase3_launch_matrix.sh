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

# This script is one of its own dependencies: a change to which cells run, or
# in what order, changes the artefacts as surely as a change to the trainer.
DEPS=(scripts/phase3_launch_matrix.sh scripts/phase3_selection_matrix.py
      scripts/phase3_select_n.py
      scripts/train_cifar10_v185_bidirTokenPrune05_ccs01_clip.sh
      scripts/train_flickr25k_v185_bidirTokenPrune05_clip.sh
      scripts/train_nuswide_v185_sweep_clip.sh
      scripts/train_mscoco_F2_sweep_clip.sh
      train_siglip2.py model_siglip2.py config.py evaluation_siglip2.py
      extraction_siglip2.py dna_utils/run_identity.py
      dna_utils/extraction_validation.py dna_utils/cache_provenance.py
      dataloaders.py)
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

# Count the EXACT sixteen keys, not every JSON in the directory. Counting
# files meant a leftover diagnostic record made 16 unreachable, and a missing
# cell could be offset by a smoke that happened to be sitting there.
MISSING=()
for DS in "${DATASETS[@]}"; do
    case "$DS" in
        cifar10)   EXP=cifar_A_v4 ;;
        flickr25k) EXP=flickr_A_v4 ;;
        nuswide)   EXP=nuswide_A_v4 ;;
        mscoco)    EXP=mscoco_A_v5b ;;
    esac
    for N in 4 9 19 39; do
        F="artifacts/phase3_selection/phase3sel_${EXP}_N${N}_s42.json"
        [[ -f "$F" ]] || MISSING+=("${DS}/N${N}")
    done
done
if [[ "${#MISSING[@]}" -gt 0 ]]; then
    echo "[matrix] missing ${#MISSING[@]} of 16: ${MISSING[*]}" >&2
    FAILED=1
else
    echo "[matrix] all 16 cell records present"
fi
[[ "$FAILED" == "0" ]] || { echo "MATRIX INCOMPLETE $HEAD_SHA" >&2; exit 1; }
echo "MATRIX DONE $HEAD_SHA"
echo "[matrix] next: python scripts/phase3_select_n.py"
