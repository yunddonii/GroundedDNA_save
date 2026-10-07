#!/usr/bin/env bash
# Score a list of Stage-3 cells (a3_v2 on 2,000 Flickr rows + d4d6). Usage: score_list.sh <runs.txt>
set -u; cd /home/yschoi/gdna_textdiag || exit 90
S3=result/analysis/textdiag_2026-10/stage3; RUNS=$1
bash result/analysis/textdiag_2026-10/d0/run_batch.sh "$RUNS" $S3/a3v2_2000 --caption_file $PWD/cache_eval/flickr25k_v4_train_plus_evaldb.jsonl --reference_text_cache $PWD/cache_eval/ref_text_flickr_v4_plus_evaldb --extra_rows_from_caption_file --oracle_mix 0.0 0.3 1.0; r1=$?
bash result/analysis/textdiag_2026-10/d4d6/run_batch.sh "$RUNS" $S3/d4d6; r2=$?
echo "rc a3v2=$r1 d4d6=$r2"; [ $r1 -eq 0 ] && [ $r2 -eq 0 ]; exit $?
