#!/usr/bin/env bash
set -u; cd /home/yschoi/gdna_textdiag || exit 90
P=/home/yschoi/.conda/envs/dna_hashing/bin/python; rc_all=0
GPU=${GPU:-1}
for spec in "Flickr25k flickr25k" "NUS-WIDE nuswide" "MS-COCO mscoco"; do set -- $spec
  if [ -s cache_eval/stage2_alt/$2/attributes.json ]; then echo "$2 skip (attributes.json exists)"; continue; fi
  env -u PYTHONPATH CUDA_VISIBLE_DEVICES=$GPU GDNA_NUM_SEMANTIC_PARTS=5 $P tools/stage2_attributes_from_phrases.py --dataset $1 --in_dir cache_eval/stage2/$2 --out_dir cache_eval/stage2_alt/$2 > cache_eval/stage2_alt/$2.log 2>&1; rc=$?; echo "$2 rc=$rc"; [ $rc -eq 0 ] || rc_all=1
done; exit $rc_all
