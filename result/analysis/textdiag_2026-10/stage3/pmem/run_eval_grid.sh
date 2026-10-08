#!/usr/bin/env bash
# P-mem deploy-only re-evaluation grid on the B1 checkpoints: seeds x tau (GPU 0).
set -u; cd /home/yschoi/gdna_textdiag || exit 90; S3=result/analysis/textdiag_2026-10/stage3; rc_all=0
while read -r d; do s=$(basename "$d" | sed -E 's/.*_s([0-9]+)\+.*/\1/')
  for tau in 0.01 0.02 0.05; do out=$S3/pmem/eval_s${s}_tau${tau}.json; [ -s "$out" ] && { echo "skip $out"; continue; }
    env -u PYTHONPATH CUDA_VISIBLE_DEVICES=0 GDNA_NUM_SEMANTIC_PARTS=5 OMP_NUM_THREADS=8 /home/yschoi/.conda/envs/dna_hashing/bin/python $S3/eval_ckpt_anchor.py --result_dir "$d" --out "$out" --baseline --set anchor_source=memory --set anchor_memory_tau=$tau > $S3/pmem/eval_s${s}_tau${tau}.log 2>&1; rc=$?; echo "s$s tau$tau rc=$rc"; [ $rc -eq 0 ] || rc_all=1
  done
done < $S3/runs_b1.txt; exit $rc_all
