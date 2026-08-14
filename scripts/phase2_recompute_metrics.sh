#!/usr/bin/env bash
# Recompute the Phase 2 metrics from a COMMITTED source and seal each result.
#
# The first attempt ran while the validator was still being edited, so the first
# seven cells and the rest were computed by different code and the artefacts
# cannot tell them apart. The second ran from a clean commit but called the
# bio-evaluation only: no NMI, so every marker it wrote was missing a metric,
# and the fixed side was recomputed while the legacy side was left as whatever
# ran in August (§20.7). A delta whose two halves come from different evaluators
# is not a delta anyone can quote.
#
# This runs all three stages -- evaluation, NMI, seal -- over both sides, and
# records the commit it ran from. It refuses to start if the tree is dirty for
# any file the numbers depend on.
#
# Usage:
#   scripts/phase2_recompute_metrics.sh [gpu]
set -euo pipefail
cd /home/yschoi/GroundedDNA
PY=/home/yschoi/.conda/envs/dna_hashing/bin/python
GPU="${1:-0}"

FIXED_ROOT=result_diagnostic/phase2_F01_only
LEGACY_ROOT=result_diagnostic/phase2_legacy_bound

DEPS=(scripts/eval_cell_bioproj.py scripts/pairwise_nmi.py
      scripts/seal_cell_analysis.py
      dna_utils/extraction_validation.py dna_utils/runtime_state.py
      dna_utils/bio_constraints.py dna_utils/gc_policy.py
      evaluation_siglip2.py)
if [[ -n "$(git status --porcelain "${DEPS[@]}")" ]]; then
    echo "refusing: the evaluation source is dirty; commit it first" >&2
    git status --short "${DEPS[@]}" >&2
    exit 2
fi
HEAD_SHA=$(git rev-parse HEAD)
echo "[recompute] source commit $HEAD_SHA on gpu $GPU"

canon() {
    case "$1" in
    cifar10)   echo "CIFAR10 64"   ;;
    flickr25k) echo "Flickr25k 128" ;;
    nuswide)   echo "NUSWIDE 128"  ;;
    mscoco)    echo "MSCOCO 128"   ;;
    *) return 1 ;;
    esac
}

FAILED=0

recompute_side() {
    local d="$1" side="$2"
    local n ds C K
    n=$(basename "$d"); ds=${n%_N*}
    read -r C K <<<"$(canon "$ds")" || { echo "  [$side] $n: unknown dataset" >&2; FAILED=1; return; }

    # Phase 2 is the retrospectively bound diagnostic on both sides, so the
    # opt-in is explicit rather than a default anything else can inherit.
    if ! CUDA_VISIBLE_DEVICES="$GPU" GDNA_NUM_SEMANTIC_PARTS=5 \
         "$PY" scripts/eval_cell_bioproj.py --dir "$d" --dataset "$C" --K "$K" \
         --allow-backfilled >> logs/phase2_recompute.log 2>&1; then
        echo "  [$side] $n bioproj FAIL"; FAILED=1; return
    fi
    if ! CUDA_VISIBLE_DEVICES="$GPU" \
         "$PY" scripts/pairwise_nmi.py --results "$d" --allow-backfilled \
         >> logs/phase2_recompute.log 2>&1; then
        echo "  [$side] $n nmi FAIL"; FAILED=1; return
    fi
    if ! "$PY" scripts/seal_cell_analysis.py --dir "$d" --allow-backfilled \
         >> logs/phase2_recompute.log 2>&1; then
        echo "  [$side] $n seal FAIL"; FAILED=1; return
    fi
    echo "  [$side] $n ok"
}

for d in "$FIXED_ROOT"/*/; do
    recompute_side "$d" fixed
done

# The legacy faces must come from the SAME evaluator, or the delta measures the
# evaluator change as much as F01 (§19.4). They are bound by
# scripts/bind_legacy_phase2.py, which never writes into the legacy run itself.
if [[ -d "$LEGACY_ROOT" ]]; then
    for d in "$LEGACY_ROOT"/*/; do
        recompute_side "$d" legacy
    done
else
    echo "[recompute] $LEGACY_ROOT is absent; run scripts/bind_legacy_phase2.py" >&2
    FAILED=1
fi

if [[ "$FAILED" != "0" ]]; then
    echo "RECOMPUTE INCOMPLETE $HEAD_SHA" >&2
    exit 1
fi
echo "RECOMPUTE DONE $HEAD_SHA"
