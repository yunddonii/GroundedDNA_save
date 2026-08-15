#!/usr/bin/env bash
# Rebuild the counterfactual-foil overlays against the v6prov CLIP caches.
#
# The overlays are derived from a factual cache, so an overlay built on the old
# unprovenanced cache cannot be paired with a rebuilt one: the foil features
# would come from a different backbone snapshot than the factual features they
# are contrasted against. The four existing overlays live beside the OLD caches
# only, which is why cutting the runners over to v6prov without this step would
# have broken the foil path outright.
#
# Each dataset runs on its own GPU and writes its own log. One dataset failing
# does not stop the others -- this is meant to be launched and left.
#
# Usage: scripts/build_foil_overlays_v6prov.sh [gpu0 gpu1 gpu2 gpu3]
set -uo pipefail            # NOT -e: a failed dataset must not kill the rest

REPO=/home/yschoi/GroundedDNA
ROOT=/data/yschoi/groundeddna_cache_v6prov
cd "$REPO"
mkdir -p logs/foil_v6prov

GPUS=(${1:-0} ${2:-1} ${3:-2} ${4:-3})

# dataset : base-cache dir : qwen jsonl : foil jsonl
#
# CIFAR-10 needs its own foil JSONL: the checked-in `cifar10.foils.jsonl` was
# generated from `cifar10_qwen.jsonl`, while the paper's CIFAR runs record
# `cifar10_qwen_v4.jsonl` in their args.txt and the rebuilt cache is built from
# it. The overlay builder refuses the mismatch, correctly -- but the launcher
# has to supply the right file rather than rely on someone passing it by hand,
# or a standalone four-dataset run cannot reproduce what was built.
#
# Always the `_tokens` directory. The old CIFAR-10 cache was a single merged
# directory holding both the pooled visual features and the text tokens, so the
# P0 runner pointed at `./cache/cifar10_clip`; the rebuild splits those in two,
# and it is the token directory that carries `text_tokens.f16.npy` while
# symlinking the visual arrays back to the pooled one. Pointing at the pooled
# directory fails with "factual cache is incomplete".
FOILS=artifacts/semantic_detail_multidataset
SPECS=(
  "flickr:flickr25k_clip_tokens:cache/flickr25k_qwen3_v4_trainset.jsonl:"
  "mscoco:mscoco_clip_tokens:cache/mscoco_qwen3_v5b_trainset.jsonl:"
  "nuswide:nuswide_clip_tokens:cache/nuswide_qwen3_v4_trainset.jsonl:"
  "cifar10:cifar10_clip_tokens:cache/cifar10_qwen_v4.jsonl:$FOILS/cifar10_v4.foils.jsonl"
)

declare -A PIDS=()
declare -A STATUS=()
i=0
for spec in "${SPECS[@]}"; do
    IFS=: read -r DS SUB QWEN FOIL <<<"$spec"
    GPU="${GPUS[$i]}"; i=$((i + 1))
    BASE="$ROOT/$SUB"
    OVERLAY="$ROOT/${SUB}_foils"
    LOG="logs/foil_v6prov/${DS}.log"

    # A missing input is a failure, not a skip: the caller asked for four
    # overlays and would otherwise be told all four finished.
    if [[ ! -d "$BASE" ]]; then
        echo "[foil-v6prov] $DS: no base cache at $BASE" >&2
        STATUS[$DS]="missing base cache"; continue
    fi
    if [[ ! -f "$QWEN" ]]; then
        echo "[foil-v6prov] $DS: no caption cache at $QWEN" >&2
        STATUS[$DS]="missing caption cache"; continue
    fi
    if [[ -n "$FOIL" && ! -f "$FOIL" ]]; then
        echo "[foil-v6prov] $DS: no foil JSONL at $FOIL" >&2
        STATUS[$DS]="missing foil jsonl"; continue
    fi
    if [[ -f "$OVERLAY/semantic_detail_cache_manifest.json" ]]; then
        echo "[foil-v6prov] $DS: overlay already built; skipping"
        STATUS[$DS]=ok; continue
    fi

    echo "[foil-v6prov] $DS -> $OVERLAY on gpu $GPU (log: $LOG)"
    BASE_CACHE="$BASE" QWEN="$QWEN" OVERLAY="$OVERLAY" \
        ${FOIL:+FOIL_JSONL="$FOIL"} \
        nohup bash scripts/prepare_semantic_detail_cache.sh "$DS" "$GPU" \
        > "$LOG" 2>&1 &
    PIDS[$DS]=$!
done

# A bare `wait` returns 0 whatever the children did, so the first version
# printed "all datasets finished" after a child had exited 1 -- and the CIFAR
# overlay it claimed was in fact built by a separate hand-run retry.
for DS in "${!PIDS[@]}"; do
    if wait "${PIDS[$DS]}"; then
        STATUS[$DS]=ok
    else
        STATUS[$DS]="failed (rc=$?)"
    fi
done

FAILED=0
echo "[foil-v6prov] ---- per-dataset status ----"
for spec in "${SPECS[@]}"; do
    IFS=: read -r DS _ _ _ <<<"$spec"
    state="${STATUS[$DS]:-not started}"
    printf '  %-9s %s\n' "$DS" "$state"
    [[ "$state" == ok ]] || FAILED=1
done
if [[ "$FAILED" != "0" ]]; then
    echo "[foil-v6prov] NOT all datasets built; see logs/foil_v6prov/*.log" >&2
    exit 1
fi
echo "[foil-v6prov] all 4 datasets built"
