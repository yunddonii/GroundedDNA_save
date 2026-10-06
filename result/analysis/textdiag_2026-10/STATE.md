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
- (none)

## Next (in order)
1. D0: write `a3_v2.py` (reference caption file + reference text cache, all validation rows, lexical pair rule, bootstrap CIs, codeword + codon level, oracle-mixing calibration); positive control before any use.
2. D2: decision-1 assumption test on V4/V5b captions (per-axis element extractors; text neighbours vs random / other-axis / image neighbours).
3. D3: ceilings S_oracle, S_probe (reuse `stage5_bigidea/feasibility.py`).
4. D4, D6, D5(final-checkpoint part). Then Stage 1 (needs idle GPUs; nvidia-smi first).
