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

# The driver itself is a dependency: a change to which stages run, or in which
# order, changes the artefacts as surely as a change to the evaluator.
DEPS=(scripts/eval_cell_bioproj.py scripts/pairwise_nmi.py
      scripts/seal_cell_analysis.py scripts/phase2_recompute_metrics.sh
      scripts/_phase2_cell_state.py scripts/bind_legacy_phase2.py
      dna_utils/extraction_validation.py dna_utils/runtime_state.py
      dna_utils/bio_constraints.py dna_utils/gc_policy.py
      dna_utils/dna_code_utils.py evaluation_siglip2.py)
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

# PREFLIGHT, before anything is written. The first version recomputed all 15
# fixed cells and only then noticed that the legacy root did not exist, so it
# overwrote production and exited INCOMPLETE with half a delta on disk (§22.2).
# The exact expected cell set is checked on both roots, not an arbitrary subset:
# one fixed plus one legacy would otherwise finish with FAILED=0 and print DONE.
EXPECTED_CELLS=(cifar10_N4 cifar10_N9 cifar10_N19 cifar10_N39
                flickr25k_N4 flickr25k_N9 flickr25k_N19
                nuswide_N4 nuswide_N9 nuswide_N19 nuswide_N39
                mscoco_N4 mscoco_N9 mscoco_N19 mscoco_N39)

preflight_root() {
    local root="$1" side="$2" missing=0
    if [[ ! -d "$root" ]]; then
        echo "[preflight] $side root $root is absent" >&2
        if [[ "$side" == legacy ]]; then
            echo "            run scripts/bind_legacy_phase2.py first" >&2
        fi
        return 1
    fi
    for cell in "${EXPECTED_CELLS[@]}"; do
        if [[ ! -d "$root/$cell" ]]; then
            echo "[preflight] $side: missing cell $cell" >&2; missing=1; continue
        fi
        # An unreadable extraction here means an aborted run later, after the
        # earlier cells have already been overwritten.
        if ! "$PY" scripts/_phase2_cell_state.py "$root/$cell" \
             --allow-backfilled >/dev/null; then
            echo "[preflight] $side: $cell does not validate" >&2; missing=1
        fi
    done
    local found
    found=$(find "$root" -mindepth 1 -maxdepth 1 -type d | wc -l)
    if [[ "$found" != "${#EXPECTED_CELLS[@]}" ]]; then
        echo "[preflight] $side: $found cell dirs, expected ${#EXPECTED_CELLS[@]}" >&2
        missing=1
    fi
    return "$missing"
}

PREFLIGHT_OK=0
preflight_root "$FIXED_ROOT" fixed || PREFLIGHT_OK=1
preflight_root "$LEGACY_ROOT" legacy || PREFLIGHT_OK=1
mkdir -p logs
: >> logs/phase2_recompute.log || { echo "[preflight] logs are not writable" >&2; PREFLIGHT_OK=1; }
if [[ "$PREFLIGHT_OK" != "0" ]]; then
    echo "RECOMPUTE REFUSED $HEAD_SHA (nothing was written)" >&2
    exit 2
fi
echo "[recompute] preflight ok: ${#EXPECTED_CELLS[@]} cells on each of 2 roots"

for cell in "${EXPECTED_CELLS[@]}"; do
    recompute_side "$FIXED_ROOT/$cell" fixed
done

# The legacy faces must come from the SAME evaluator, or the delta measures the
# evaluator change as much as F01 (§19.4). They are bound by
# scripts/bind_legacy_phase2.py, which never writes into the legacy run itself.
for cell in "${EXPECTED_CELLS[@]}"; do
    recompute_side "$LEGACY_ROOT/$cell" legacy
done

# Count what is actually on disk rather than trusting the loop's own bookkeeping.
for root in "$FIXED_ROOT" "$LEGACY_ROOT"; do
    sealed=$(find "$root" -name analysis_complete.json -type f | wc -l)
    if [[ "$sealed" != "${#EXPECTED_CELLS[@]}" ]]; then
        echo "[recompute] $root: $sealed seals, expected ${#EXPECTED_CELLS[@]}" >&2
        FAILED=1
    fi
done

if [[ "$FAILED" != "0" ]]; then
    echo "RECOMPUTE INCOMPLETE $HEAD_SHA" >&2
    exit 1
fi
echo "RECOMPUTE DONE $HEAD_SHA"
