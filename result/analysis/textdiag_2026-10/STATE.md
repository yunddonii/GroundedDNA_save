# Text-path line — resume point (update at every step boundary)

Read this first after any session restart. Branch `text-diag-2026-09`, worktree `/home/yschoi/gdna_textdiag`.
Plan: `/home/yschoi/.claude/plans/hidden-tinkering-pine.md` (approved 2026-10-05). Log: `docs/MODI_PROJECT_LOG.md`.

## How to check running work
- Long jobs run in tmux via `/data/yschoi/gdna_p3exec_authority/bin/tmux_run.sh <name> <cmd> <args...>` (separate argv).
- Status files: `/data/yschoi/gdna_p3exec_authority/runs/<name>.status` (written at exit: rc, seconds, end). `tmux ls | grep textdiag` shows live sessions.
- Each diagnostic has its own directory under `result/analysis/textdiag_2026-10/<id>/`; a job is complete when its result file exists and the .status has rc=0.

## Done
- 2026-10-05 D1: 45 checkpoints scored (`d1/a2a3/*.json`, tmux `textdiag_d1b` rc 0). Logged in MODI (2026-10-05 D1 entry).

## In progress
- D2 DONE 2026-10-06 (d2/*.json for flickr v4/v5b, nus v4, coco v4/v5b; script d2_text_neighbours.py) -- not yet logged in MODI; log together with D0.
- D0 DONE: a3_v2.py smoke-tested (ac2, p2anc; vectorised bootstrap 14 s/run). Batch over 63 checkpoints running: tmux `textdiag_d0`, out `result/analysis/textdiag_2026-10/d0/a3v2/`, status `/data/yschoi/gdna_p3exec_authority/runs/textdiag_d0.status`. When done: aggregate (S, CI, calibration threshold per dataset), log D0 in MODI, update Done.

## Next (in order)
1. D0: write `a3_v2.py` (reference caption file + reference text cache, all validation rows, lexical pair rule, bootstrap CIs, codeword + codon level, oracle-mixing calibration); positive control before any use.
3. Stage 1 prep: build_cmd.py (off-protocol command from approved args.txt: strip seal/authority flags, add --no_gumbel_softmax), one-cell smoke on an idle GPU (nvidia-smi first), then B0/OFF/H2 cells (Flickr first). Write PREREGISTRATION.md for Stage 1 before the first cell. Score with a3_v2 on ALL validation rows.
NOTE: run_queue.sh skip pattern (*+tag+*) never matches -> fix to *_tag+* AFTER the queues finish (bash reads a running script incrementally)
