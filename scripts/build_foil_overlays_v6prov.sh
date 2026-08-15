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

# dataset : base-cache dir : qwen jsonl
#
# Always the `_tokens` directory. The old CIFAR-10 cache was a single merged
# directory holding both the pooled visual features and the text tokens, so the
# P0 runner pointed at `./cache/cifar10_clip`; the rebuild splits those in two,
# and it is the token directory that carries `text_tokens.f16.npy` while
# symlinking the visual arrays back to the pooled one. Pointing at the pooled
# directory fails with "factual cache is incomplete".
SPECS=(
  "flickr:flickr25k_clip_tokens:cache/flickr25k_qwen3_v4_trainset.jsonl"
  "mscoco:mscoco_clip_tokens:cache/mscoco_qwen3_v5b_trainset.jsonl"
  "nuswide:nuswide_clip_tokens:cache/nuswide_qwen3_v4_trainset.jsonl"
  "cifar10:cifar10_clip_tokens:cache/cifar10_qwen_v4.jsonl"
)

i=0
for spec in "${SPECS[@]}"; do
    IFS=: read -r DS SUB QWEN <<<"$spec"
    GPU="${GPUS[$i]}"; i=$((i + 1))
    BASE="$ROOT/$SUB"
    OVERLAY="$ROOT/${SUB}_foils"
    LOG="logs/foil_v6prov/${DS}.log"

    if [[ ! -d "$BASE" ]]; then
        echo "[foil-v6prov] $DS: no base cache at $BASE; skipping" >&2
        continue
    fi
    if [[ ! -f "$QWEN" ]]; then
        echo "[foil-v6prov] $DS: no caption cache at $QWEN; skipping" >&2
        continue
    fi
    if [[ -f "$OVERLAY/semantic_detail_cache_manifest.json" ]]; then
        echo "[foil-v6prov] $DS: overlay already built; skipping"
        continue
    fi

    echo "[foil-v6prov] $DS -> $OVERLAY on gpu $GPU (log: $LOG)"
    BASE_CACHE="$BASE" QWEN="$QWEN" OVERLAY="$OVERLAY" \
        nohup bash scripts/prepare_semantic_detail_cache.sh "$DS" "$GPU" \
        > "$LOG" 2>&1 &
done

wait
echo "[foil-v6prov] all datasets finished; check logs/foil_v6prov/*.log"
