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
- 2026-10-07 11:50Z Stage 2 INCIDENT: survey steps done for all 3 datasets (NUS needed the lenient JSON repair, commit after 665d83b). Step 2a (cumulative concept list) failed: NUS reply cut at 4096 tokens; Flickr list oscillated 69/46/81. All three chains stopped at concepts (killed by PID 11:50Z). Fix in progress: two-level extraction (per-batch concepts -> union -> consolidation, X-Cluster style). After the fix: rerun `tools/stage2_run_dataset.sh <ds> <gpuA> <gpuB>` per dataset (survey steps are skipped: rc=0 on record); the concepts step reruns.
- Stage 2 (2026-10-07, approved: no relation wording, 5-10 words, no A/B): tools committed 665d83b. RUNNING since 2026-10-07 11:24Z: tmux stage2_flickr (GPUs 0,1), stage2_nuswide (2,3), stage2_mscoco (4,5); status files cache_eval/stage2/<ds>/pipeline.status; chain stops rc=3 at validate_pilot if a gate fails (then user decision); rerun the same tmux command to resume. Plan was with 2 GPUs each: Flickr GPUs 0,1; NUS 2,3; COCO 4,5 (tmux stage2_<ds>); outputs cache_eval/stage2/<dataset>/; validation runs automatically after the pilot and after the full file. Report attributes + pilot gates per dataset as each finishes.
- 2026-10-07 07:1xZ: Stage 3 on V4/V5b CLOSED (MODI entries): no arm beats B1; B1=H2 confirmed on NUS/COCO for retrieval; codon text effect replicates only on Flickr. Stage 2 prompts (4 steps) proposed to the user; waiting for confirmation of wording, then run step 1 (survey, 1,000 images x 3 datasets) on GPUs.
- Stage 3 status 2026-10-07 06:40Z: N1 family closed (TD/TDXM/TDXM5 negative). S arm done: retrieval +.010, codon S +.05-.10 (3/3 CI>0) but below B1, unique -.11. Next: N2 = existing --lambda_text_preq_contrastive (text_m vs pre-VQ z_m InfoNCE) replacing text_code_kl, on S and on B1 (3 seeds each). Stage 2 still waits for the user's caption-length answer.
- Stage 3 N1 DONE + logged (MODI 2026-10-07 Stage 3 entry): TD fails P-DELTA (codon); xmodal_commit must stay; TDXM keeps retrieval, halves dead codes, no transfer. 'td3 /data reads done' SENT 06:03Z -> no /data reads for 4 h without a heads-up. Next: implement --train_routing_mode codebook_mean (S arm) on CPU; then cells TDXM-λ.5 and S (Flickr 3 seeds) after a heads-up to the anchor session.
- Stage 3 N1 (TD): code committed 498cc4a; gate PASS (td3_gate_H2_s42 bit-identical to td1_flickr_H2_s42, stage3/gate_bit_identity.json). Smoke td3_TD_s42 FAILED: path_consistency term 589 -> 29 (raw squared distances / tau 0.1), codebook collapsed (dead .76, mAP .680). Fix committed f0dd43e (z-scored distances, tau .5). Re-smoke td3_TDv2_s42 PASSED (term .13-.38, mAP .7416 vs B1 .7569, dead .091, uniq .508) = TD seed 42. B1 cells are NOT re-run (gate proved default-flag bit-identity; td1_flickr_H2_s4x serve as B1). TD 3 seeds done + scored (stage3/a3v2_2000, d4d6): deployed codeword S +.065/-.021/+.139 vs B1 +.083/-.005/-.021; deployed CODON S +.131/-.091/-.046 vs B1 +.161/+.196/+.109 -> TD fails P-DELTA at the codon level; mAP -.015/-.012/+.000. TD changed two things (xmodal_commit removed + path_consistency added), so diagnostics running 05:50Z: td3_XM0_s42-44 (B1 minus xmodal_commit only) and td3_TDXM_s42-44 (TD with xmodal_commit kept). After: score with stage3/score_list.sh <runs.txt>, then MODI entry for Stage 3 N1, then SEND 'td3 /data reads done <UTC>' to the anchor session via stage3/launch_batch_staggered.sh (60 s stagger, GPUs 1-5), score 2,000 rows + d4d6, then SEND 'td3 /data reads done <UTC>' to the anchor session.
- NOTE: tmux .status files use lowercase rc=; waiters must grep '^rc='.
- (none; Stage 1 complete and logged 2026-10-07)

## Done (Stage 1)
- 22 cells + scoring + controls + H1 check; MODI 2026-10-07 entry; B1 = H2 (10-term recipe). Thresholds: T Flickr ≈ .21 (provisional), NUS .15, COCO .18.

## In progress
- Flickr 2,000-row rescoring (500 val + 1,500 eval-DB rows): tmux td1_score2000 -> stage1/a3v2_2000/, summary A3V2_2000_SUMMARY.md. Then: re-fix Flickr T, add a paragraph to the MODI 2026-10-07 Stage 1 entry.

## Done 2026-10-07
- Evaluation-only V4 captions for 1,500 Flickr DB images: cache_eval/flickr25k_qwen3_v4_evaldb1500.jsonl (+ evaldb_sample.json; 0 train overlap, 0 parse failures). Reference text cache cache_eval/ref_text_flickr_v4_plus_evaldb (6,500 has_text rows; text_part not committed, rebuild with extract_clip_text_features.py from cache_eval/flickr25k_v4_train_plus_evaldb.jsonl).
- Peer notice (anchor session, audit §795): GPU partition lifted; receipt sent 2026-10-07. Always nvidia-smi before launch.

## GPU constraint
- 2026-10-07 06:12Z: GPU 0 RELEASED by the anchor session (its run failed at preflight after 12 s; no /data or GPU plans until the audit decides). Quiet window ended early; heads-up sent before resuming /data-reading cells. All 6 GPUs usable again (nvidia-smi before each launch).
- 2026-10-07 06:11Z: anchor recovery run STARTED on GPU 0 (tmux ancT9r_v9r8, reads NUS caches from /data for hours, 8 h wall). Until its release message: GPUs 1-5 only; any /data-reading cell needs a heads-up message to groundeddna-29 first (quiet window promised until ~10:00Z).
- 2026-10-07 05:3xZ: launch-window agreement (audit §831): the anchor recovery launch happens only AFTER I send 'td3 /data reads done <UTC time>' (planned ~06:20Z) and before 17:00Z; any later /data reads need a heads-up message first. MUST send that message after the td3 batch + CPU scoring finish.
- 2026-10-07 ~05:30Z: agreed with anchor session groundeddna-29 to leave GPU 0 (GPU-4ac2ea6b...) for one recovery run, <= 8 h after its launch, lapses if not launched within 12 h of receipt; release message will follow. Use GPUs 1-5 only until the release (or lapse). Plan: no heavy /data reads of NUS caches in that window (Flickr-only cells, VLM captioning reads images from /home).

## Next
1. USER DECISIONS pending (asked 2026-10-07): Stage 2 (a) use dataset label vocabulary as concept material? (b) caption length ~10-15 words? Then rewrite plan Stage 2 per the user's 4-step flow (dataset survey -> VLM concepts -> VLM groups into 4 attributes -> short captions).
2. Stage 3 N1 (TD) implementation: text dropout + two-path consistency replacing xmodal_commit; flag design, entry gate (defaults reproduce B1 s42 log.csv), unit tests, one-cell smoke; PREREGISTRATION for Stage 3.
3. Fix run_queue.sh skip pattern (queues finished).
- D2 DONE 2026-10-06 (d2/*.json for flickr v4/v5b, nus v4, coco v4/v5b; script d2_text_neighbours.py) -- not yet logged in MODI; log together with D0.
- D0 DONE: a3_v2.py smoke-tested (ac2, p2anc; vectorised bootstrap 14 s/run). Batch over 63 checkpoints running: tmux `textdiag_d0`, out `result/analysis/textdiag_2026-10/d0/a3v2/`, status `/data/yschoi/gdna_p3exec_authority/runs/textdiag_d0.status`. When done: aggregate (S, CI, calibration threshold per dataset), log D0 in MODI, update Done.

## Next (in order)
1. D0: write `a3_v2.py` (reference caption file + reference text cache, all validation rows, lexical pair rule, bootstrap CIs, codeword + codon level, oracle-mixing calibration); positive control before any use.
3. Stage 1 prep: build_cmd.py (off-protocol command from approved args.txt: strip seal/authority flags, add --no_gumbel_softmax), one-cell smoke on an idle GPU (nvidia-smi first), then B0/OFF/H2 cells (Flickr first). Write PREREGISTRATION.md for Stage 1 before the first cell. Score with a3_v2 on ALL validation rows.
NOTE: run_queue.sh skip pattern (*+tag+*) never matches -> fix to *_tag+* AFTER the queues finish (bash reads a running script incrementally)
