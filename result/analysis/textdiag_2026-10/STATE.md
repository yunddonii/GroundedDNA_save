# Text-path line — resume point (update at every step boundary)

Read this first after any session restart. Branch `text-diag-2026-09`, worktree `/home/yschoi/gdna_textdiag`.
Plan: `/home/yschoi/.claude/plans/hidden-tinkering-pine.md` (approved 2026-10-05). Log: `docs/MODI_PROJECT_LOG.md`.

## GPU constraint (binding; recorded 2026-10-06 from anchor session groundeddna-29, user decision after audit §786)
- Until the anchor full-T campaign (namespace ancT9, tmux ancT9_v9r7) settles: the text-path line uses **GPUs 4 and 5 ONLY**
  (GPU-09e3ce04-0bce-569f-5cc8-e58957758b82, GPU-bf4ed000-d77c-f059-2bb2-31ee39703e99). Never GPUs 0-3, even if idle.
- Pin by UUID (CUDA_VISIBLE_DEVICES=<uuid>) and re-check `nvidia-smi --query-gpu=index,uuid` before every launch.
- Record: /home/yschoi/anchor_rt_session_state/coordination/gpu_partition_full_T.json. Lift only when that campaign has settled.

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
