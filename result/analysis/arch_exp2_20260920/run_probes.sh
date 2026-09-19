#!/bin/bash
# usage: run_probes.sh <gpu> <tag> [<tag> ...]  -- slot_role_probe on result/260920+*_<tag>+*
set -u
REPO=/home/yschoi/GroundedDNA; cd "$REPO" || exit 1
export GDNA_NUM_SEMANTIC_PARTS=5
G=$1; shift
for t in "$@"; do
  d=$(ls -d result/260920+flickr25k_setting1_${t}+* | head -1)
  CUDA_VISIBLE_DEVICES=$G /home/yschoi/.conda/envs/dna_hashing/bin/python scripts/slot_role_probe.py \
    --result_dir "$d" --out result/analysis/arch_exp2_20260920/probe/${t}.json --device cuda:0 \
    > result/analysis/arch_exp2_20260920/probe/${t}.log 2>&1
  echo "$t rc=$?"
done
