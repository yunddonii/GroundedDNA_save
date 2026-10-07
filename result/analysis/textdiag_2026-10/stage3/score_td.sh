#!/usr/bin/env bash
# Score the Stage-3 TD cells: a3_v2 on 2,000 Flickr rows (val 500 + eval-DB 1,500) and d4d6 (caption-routed vs deployed). CPU, nice.
set -u; cd /home/yschoi/gdna_textdiag || exit 90
S3=result/analysis/textdiag_2026-10/stage3; mkdir -p $S3/a3v2_2000 $S3/d4d6
ls -d $PWD/result/*_td3_TDv2_s42* $PWD/result/*_td3_TD_s43* $PWD/result/*_td3_TD_s44* > $S3/runs_td.txt
bash result/analysis/textdiag_2026-10/d0/run_batch.sh $S3/runs_td.txt $S3/a3v2_2000 --caption_file $PWD/cache_eval/flickr25k_v4_train_plus_evaldb.jsonl --reference_text_cache $PWD/cache_eval/ref_text_flickr_v4_plus_evaldb --extra_rows_from_caption_file --oracle_mix 0.0 0.3 1.0; r1=$?
bash result/analysis/textdiag_2026-10/d4d6/run_batch.sh $S3/runs_td.txt $S3/d4d6; r2=$?
echo "rc a3v2=$r1 d4d6=$r2"; [ $r1 -eq 0 ] && [ $r2 -eq 0 ]; exit $?
