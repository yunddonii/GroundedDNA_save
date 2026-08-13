#!/usr/bin/env bash
# Build the remaining provenance-complete caches as GPUs free up.
# Phase 2 owns GPUs 0,2,3,4,5 until its chains exit; Flickr already has GPU 1.
set -uo pipefail
REPO=/home/yschoi/GroundedDNA
cd "$REPO"

gpu_free() {  # a GPU is free when no phase2 chain and no cache build owns it
    ! pgrep -f "phase2_f01_reinference.sh $1 " >/dev/null \
    && ! pgrep -f "build_clip_cache_v6prov.sh $1 " >/dev/null
}

for spec in "MSCOCO" "NUSWIDE" "CIFAR10"; do
    while true; do
        for g in 1 0 2 3 4 5; do
            if gpu_free "$g"; then
                echo "[queue] $spec -> gpu $g at $(date -Is)"
                nohup bash scripts/build_clip_cache_v6prov.sh "$g" "$spec" \
                    > "logs/cache_v6prov/$(echo "$spec" | tr 'A-Z' 'a-z').log" 2>&1 &
                sleep 45          # let it claim the device before scanning again
                break 2
            fi
        done
        sleep 120
    done
done
echo "[queue] all three dispatched at $(date -Is)"
