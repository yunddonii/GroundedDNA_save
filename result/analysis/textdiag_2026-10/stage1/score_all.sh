#!/usr/bin/env bash
# Score every finished Stage-1 cell: a3_v2 (all validation rows) for all cells, d4d6 for B0/H2 cells. Skips cells already scored. CPU only.
set -u; cd /home/yschoi/gdna_textdiag || exit 90
S=result/analysis/textdiag_2026-10/stage1; mkdir -p $S/a3v2 $S/d4d6
ls -d result/*_td1_* 2>/dev/null | while read -r d; do [ -f "$d/model_state_dict.pth" ] && echo "$d"; done > $S/runs_done.txt
bash result/analysis/textdiag_2026-10/d0/run_batch.sh $S/runs_done.txt $S/a3v2 --oracle_mix 0.0 0.2 0.3 0.4 0.5 1.0; rc1=$?
grep -E "_B0_|_H2_" $S/runs_done.txt > $S/runs_b0.txt
bash result/analysis/textdiag_2026-10/d4d6/run_batch.sh $S/runs_b0.txt $S/d4d6; rc2=$?
[ "$rc1" -eq 0 ] && [ "$rc2" -eq 0 ]; exit $?
