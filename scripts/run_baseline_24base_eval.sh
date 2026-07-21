#!/usr/bin/env bash
# Dataset-parallel driver for the 48-bit (24-base) baseline DNA-space eval.
# One dataset per GPU (each does its 3 methods). Merges into one JSON at the end.
#   Usage: bash scripts/run_baseline_24base_eval.sh
set -u
cd /home/yschoi/GroundedDNA
mkdir -p logs docs
PY=/home/yschoi/.conda/envs/dna_hashing/bin/python
echo "[bl24eval] launch $(date '+%F %T')"

run() {  # $1=gpu $2=dataset
    CUDA_VISIBLE_DEVICES="$1" $PY scripts/baseline_48bit_dnaeval.py \
        --datasets "$2" --device cuda:0 \
        --out "docs/baseline_24base_${2}.json" \
        > "logs/baseline_24base_${2}.log" 2>&1 \
        && echo "[bl24eval] OK $2 $(date '+%T')" \
        || echo "[bl24eval] FAIL $2 $(date '+%T')"
}

run 0 Flickr25k &
run 1 MSCOCO &
run 2 NUSWIDE &
run 3 CIFAR10 &
wait

# merge the 4 per-dataset JSONs into one
$PY - <<'PYEOF'
import json, glob
cells, cfg = [], None
for f in sorted(glob.glob("docs/baseline_24base_*.json")):
    if "_all" in f: continue
    d = json.load(open(f)); cfg = cfg or d.get("config")
    cells += d.get("cells", [])
json.dump({"config": cfg, "cells": cells},
          open("docs/baseline_24base_dnaeval_all.json", "w"), indent=2)
print(f"merged {len(cells)} cells -> docs/baseline_24base_dnaeval_all.json")
PYEOF
echo "[bl24eval] ALL DONE $(date '+%F %T')"
