#!/usr/bin/env bash
# After the 36 N=19 cells: score them (A3 v2 2,000 rows + d4d6) and score P-mem deploy-only on the
# B1 N=19 checkpoints. Every wait and step is bounded.
set -u; cd /home/yschoi/gdna_textdiag || exit 90
RUNS=/data/yschoi/gdna_p3exec_authority/runs; MA=/home/yschoi/GroundedDNA/result/analysis/textdiag_stageL_train_length_20261008
S3=result/analysis/textdiag_2026-10/stage3; L=result/analysis/textdiag_2026-10/stageL
t0=$(date +%s)
for g in 0 1 2 3 4 5; do
  until [ -s $RUNS/qL19_g$g.status ] && grep -q '^rc=' $RUNS/qL19_g$g.status; do
    [ $(( $(date +%s) - t0 )) -gt 10800 ] && { echo "TIMEOUT waiting for qL19_g$g"; break; }; sleep 60; done
  echo "qL19_g$g: $(tr '\n' ' ' < $RUNS/qL19_g$g.status 2>/dev/null)"
done
awk '{print $4}' $MA/eval_runs.txt | grep "_N19_" > $L/runs_n19.txt; echo "N19 runs to score: $(wc -l < $L/runs_n19.txt)"
timeout 7200 bash $S3/score_list.sh $PWD/$L/runs_n19.txt; echo "score rc=$?"
ls -d /home/yschoi/GroundedDNA/result/*td3_L19_s4* > $L/runs_l19.txt
timeout 3600 bash result/analysis/textdiag_2026-10/d0/run_batch.sh $L/runs_l19.txt $L/a3v2_2000_pmem --caption_file $PWD/cache_eval/flickr25k_v4_train_plus_evaldb.jsonl --reference_text_cache $PWD/cache_eval/ref_text_flickr_v4_plus_evaldb --extra_rows_from_caption_file --oracle_mix 0.0 0.3 1.0 --set anchor_source=memory; echo "pmem a3 rc=$?"
timeout 3600 bash result/analysis/textdiag_2026-10/d4d6/run_batch.sh $L/runs_l19.txt $L/d4d6_pmem --set anchor_source=memory; echo "pmem d4d6 rc=$?"
/home/yschoi/.conda/envs/dna_hashing/bin/python $L/length_table.py --json $L/length_table.json > $L/length_table.txt 2>&1; echo "table rc=$?"
/home/yschoi/.conda/envs/dna_hashing/bin/python result/analysis/textdiag_2026-10/make_main_bundles.py --only stageL; echo "bundle rc=$?"
cp $L/length_table.txt $L/length_table.json $MA/ 2>/dev/null; echo DONE
