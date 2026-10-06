#!/usr/bin/env bash
set -u; cd /home/yschoi/gdna_textdiag || exit 90
C=/data/yschoi/dataset/deephashing/cache; P=/home/yschoi/.conda/envs/dna_hashing/bin/python; rc_all=0
for spec in "nuswide_clip_tokens nuswide_qwen3_v4_trainset nuswide_v4" "mscoco_clip_tokens mscoco_qwen3_v5b_trainset mscoco_v5b"; do
  set -- $spec
  nice -n 10 env -u PYTHONPATH OMP_NUM_THREADS=16 $P result/analysis/textdiag_2026-10/d3_ceilings.py --cache_dir /data/yschoi/groundeddna_cache_v6prov/$1 --caption_file $C/$2.jsonl --out result/analysis/textdiag_2026-10/d3/$3.json > result/analysis/textdiag_2026-10/d3/$3.log 2>&1
  rc=$?; echo "$3 rc=$rc"; [ "$rc" -eq 0 ] || rc_all=1
done
exit "$rc_all"
