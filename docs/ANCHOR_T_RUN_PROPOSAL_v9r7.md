# Anchor model — r7 full stage T: exact request, command and resource plan (audit §785)

**PREPARATION ONLY. Nothing has been executed for full T.**
- No weight or config load, image forward pass, test-payload access or producer has run.
- The request was rendered from JSON/text metadata under the named-open guard.
- Full T runs only after its own approval line names this exact request.

## 1. Receipt of §785 and archive

§785 accepted the full R campaign `ancR9`:
- run `20261006T115225Z-216b629c`;
- receipt `ancR9_sweep_complete.json` `2897aa880dddd07cbb0aeb49ef090b7e5ccc06e6e9b540922a9ca77a30d81d4a`;
- 12 cells, R charge 6,766.29 s, cumulative 7,280.51 s of 80,000.

§780 is spent. Full T needs its own approval, and the downstream TODO and paper gates stay open.

**Archive** (`refit_v9/full_r_ancR9/`, commit `cd08c64` plus this commit). The live originals are
preserved. The archive holds:
- copies of the settled operations ledger, the command log, and the tmux `cmd`/`log`/`status`;
- `approved_argv_as_used.json`;
- `cells_summary.json`, with the per-cell checkpoint, runtime and config digests re-hashed, the
  terminal epochs and official-test absence;
- **`records_manifest.sha256`**: digests of the 15 R records (reservation, snapshot, receipt, 12 cell
  records);
- `SHA256SUMS` over the archive.

The proposal's resource wording was corrected after the run, in `cd08c64`, as a documentation-only
change. The approved headroom is 520 GPU-s (4 × 130), and the actual dataset-to-GPU mapping is now
recorded. No execution evidence or source pin was changed.

## 2. The request

| Item | Value |
|---|---|
| Semantic request SHA256 | **`a74b68e1db10c0d71d911ce491a25a2fa97dda6258c88d9f2c063223e1b40612`** |
| Mode, namespace | `run`, `ancT9`; 12 cells; `gpu_count` 4; `refit_epochs` none (each cell's N) |
| Manifest | r7 `2f24fc80bb83e2c73b17ec473c2731ade6d34a356e02414e6719fe9e9bec947c` (tree `/data/yschoi/gdna_anchor_refit_v9r6`) |
| F | `ancF_candidate_v1.json` `5165f5dc…`, §744 acceptance `13ef776b…` |
| R receipt / approval | `2897aa88…` / the §780 `stage-R-run` line |
| Chain (5 producers) | T entry (official query/DB extraction + raw base evaluation), `extract_train_split`, `eval_cell_bioproj --require-train`, `pairwise_nmi --require-train`, `seal_cell_analysis --require-train` |
| Outputs field | the unchanged 13-name set (§776: the allowed/exclusion set) |

**Cells and pins.** These are the per-cell pins the T entry and each producer verify before any
deserialization, as reviewed in r6/r7. All equal the R records.

| Cell | Terminal epoch | Checkpoint | Runtime witness | config.pt |
|---|---|---|---|---|
| cifar10 N4 s42 | 4 | `304bcfc9f604…` | `b12ec041210a…` | `069230337fbd…` |
| cifar10 N4 s43 | 4 | `4c442a36d2bf…` | `521b28dd3bee…` | `5e1d3768d9a5…` |
| cifar10 N4 s44 | 4 | `24ffa3ce5847…` | `4c049f25f8c4…` | `8847960e32c1…` |
| flickr25k N4 s42 | 4 | `697c677655cb…` | `e85a78573238…` | `eaa2eff39874…` |
| flickr25k N4 s43 | 4 | `133c21c263f8…` | `f0a7fd892c30…` | `3ad16d5ce17e…` |
| flickr25k N4 s44 | 4 | `0de91ec2688a…` | `f983ec28aa47…` | `d15643e5ec90…` |
| mscoco N39 s42 | 39 | `19d548095127…` | `a47cbf42b978…` | `0d4f0f3f211a…` |
| mscoco N39 s43 | 39 | `9979a34ed75b…` | `2c476ffd06e6…` | `08a2dda601d6…` |
| mscoco N39 s44 | 39 | `6f7655a64a45…` | `2998ec7c2d1e…` | `858b01307a25…` |
| nuswide N4 s42 | 4 | `d1318bfdadc7…` | `11092b13d203…` | `eaee9134e177…` |
| nuswide N4 s43 | 4 | `c303cbb24a95…` | `08dcc8b8790c…` | `22d49a589f0e…` |
| nuswide N4 s44 | 4 | `7c852680611a…` | `1166f9ace992…` | `7e63262c0f19…` |

Each request cell also binds:
- the cell record;
- the trainer campaign evidence `phase3_campaign_binding.json`;
- the scientific-recipe digest.

The full values are in `t_run_request.json`.

## 3. Admission and guarded render

**Admission** (`scripts/anchor_refit_stage.py` `admit_refit_receipt`, `--plan` path), from metadata
only. It re-verifies the following and requires exactly the twelve F cells:
- the R receipt at its digest;
- its snapshot `93c4c856…` and the r7 manifest;
- F and the §744 acceptance;
- the §780 approval line in the ledger;
- per cell: the record, the campaign binding, the six completion pins, the trainer evidence and
  runtime witness at the terminal epoch, and the sealed recipe;
- that no official-test output exists in any run directory.

**At execution, before each cell's T entry**, the reviewed r7 runtime checks run, as in the T smoke:
- the exclusive attempt reservation;
- the entry claim;
- config.pt read once;
- the checkpoint and witness read once and bound at the terminal epoch (`verified_runtime`);
- boundary re-checks of the snapshot and the cell's three files between producers.

**Guarded render** (`refit_v9/t_run_request_r7/`, `SHA256SUMS`):
- exit 0, **0 refused opens**;
- 45 named opens of the 44 allow-listed JSON/ledger files:
  - the r7 manifest, R receipt and R snapshot;
  - the 12 R cell records;
  - each run directory's `phase3_campaign_binding.json` and `model_state_dict.pth.runtime.json` (24 files);
  - F, the v7 `ancS7` snapshot, p3rfB aggregate, selected_n, and the audit ledger;
- 195 source opens;
- 206 child invocations, all git source queries.

The argument list is in `launcher_argv_plan_base.txt`.

## 4. The command proposed for execution (after its approval line)

tmux **`ancT9_v9r7`** through `/data/yschoi/gdna_p3exec_authority/bin/tmux_run.sh`:

```
env -C /data/yschoi/gdna_anchor_refit_v9r6 -u PYTHONPATH -u CUDA_VISIBLE_DEVICES GDNA_NUM_SEMANTIC_PARTS=5 \
  /home/yschoi/.conda/envs/dna_hashing/bin/python scripts/anchor_confirm_supervisor.py \
  --manifest /data/yschoi/gdna_anchor_refit_v9r6/artifacts/anchor_confirmation/authority_manifest_v9r7.json \
  --manifest-sha256 2f24fc80bb83e2c73b17ec473c2731ade6d34a356e02414e6719fe9e9bec947c \
  --stage stage-T-run --planned-cells 12 --ops-root /home/yschoi/gdna_anchorRT_ops --watch-path /home/yschoi -- \
  /home/yschoi/.conda/envs/dna_hashing/bin/python scripts/phase3_selection_matrix.py --anchor-confirm test \
  --anchor-manifest /data/yschoi/gdna_anchor_refit_v9r6/artifacts/anchor_confirmation/authority_manifest_v9r7.json \
  --anchor-manifest-sha256 2f24fc80bb83e2c73b17ec473c2731ade6d34a356e02414e6719fe9e9bec947c \
  --anchor-refit-receipt /data/yschoi/gdna_anchor_refit_v9r6/artifacts/anchor_confirmation/ancR9_sweep_complete.json \
  --anchor-refit-receipt-sha256 2897aa880dddd07cbb0aeb49ef090b7e5ccc06e6e9b540922a9ca77a30d81d4a \
  --namespace ancT9 --run --gpus 2,3,4,5 --anchor-approval-section <SECTION>
```

- This is the render's argument list without `--plan`, with the approval section added.
- There is no `--only`, extra cell or override.
- The GPU indices do not enter the request; only `gpu_count` 4 does.

**Devices: four distinct physical GPUs, coordinated with the separate session.**

| GPU | UUID |
|---|---|
| 2 | `GPU-65da79f2-d9fe-007d-c5dd-7f3599156777` |
| 3 | `GPU-c1b4c38b-bf2f-40c3-a1b7-2107d7d0cfed` |
| 4 | `GPU-09e3ce04-0bce-569f-5cc8-e58957758b82` |
| 5 | `GPU-bf4ed000-d77c-f059-2bb2-31ee39703e99` |

- **Availability now:** at preparation all six GPUs are idle (11 MiB, no compute process). The
  separate text-path session's `td1_*` jobs have ended (no tmux session).
- **The split:**
  - That session's resume file (read only) says its next cells take an idle GPU at launch.
  - The proposed partition for the duration of full T is: **GPUs 2–5 for T, GPUs 0–1 for the
    text-path session.**
  - The user decides the partition (§783) and may change it.
- **At launch:**
  - The four named UUIDs must be present with no compute process.
  - The launcher takes its exclusive host leases.
  - If any named device is busy, I wait or return for review. I never substitute another device.
- **Dataset → GPU:** one dataset stream per GPU, in the launcher's sorted order. Following the R
  mapping, that would be CIFAR-10→2, Flickr25K→3, MS-COCO→4, NUS-WIDE→5; the T snapshot records the
  actual mapping.

**Gate in the same command as the launch** (fail closed on any unavailable observation):
- the tree is clean at its submitted HEAD;
- the manifest, F, R receipt and submitted-request digests match;
- no `ancT9` record, reservation or attempt exists;
- tmux `ancT9_v9r7` is confirmed absent;
- the four named GPU UUIDs are idle;
- the R/T ledger digest is unchanged, with four settled starts;
- no T output exists in any of the 12 R run directories;
- the space is above the floor.

## 5. Resource plan

| Item | Plan |
|---|---|
| Ledger | the same append-only R/T ledger (now `31d3d386…`, 4 starts / 4 finals); cumulative **7,280.51 s** of the 80,000-s envelope (72,719.49 s remain); no reset |
| Supervision | `stage-T-run`: **12 logical cells, 60 managed producer attempts** (5 per cell, `CHILDREN_PER_CELL`; a 6th child of a cell is excess) |
| Limits | 4 GPUs, wall 28,800 s (8 h) including admission; poll 1 s, watchdog 10 s, stop bound 120 s; stop headroom **520 GPU-s** = 4 × (10 + 120) |
| Storage | floor 10 GiB + 12 × 0.75 GiB = 19 GiB; free now 339,141,206,016 B. T writes NPZs/JSONs into the existing R run directories (a Flickr25K cell wrote about 19 MB in the T smoke; larger databases scale with their row counts) |
| Expected device time | planning estimate from the T smoke (Flickr25K: 429 device-s over its five producers), scaled by split sizes: about 8k–17k device-s in total. Wall is bounded by the largest-database stream (NUS-WIDE or MS-COCO, three cells in sequence): a few hours, inside 8 h |

The estimates are planning figures, not measurements. Sharing host I/O with the separate session may
change timing.

## 6. Output contract (§776)

Per cell, T **must** create, once, in the cell's R run directory, the 12 required outputs:
- `extract_query.npz`, `extract_db.npz`, `extract_train.npz`;
- `extraction_manifest_query.json`, `extraction_manifest_db.json`, `extraction_manifest_train.json`;
- `extraction_complete.json`;
- `evaluation_siglip2_base.json` (raw base), `evaluation_siglip2_base_bioproj.json` (base/BIO);
- `pairwise_nmi.json`, `cell_result.json`, `analysis_complete.json`.

**`evaluation_siglip2_bit2.json` must be verified absent.** No producer makes it for the
base-distance recipe, and none is added.

The request's 13-name `outputs` field is unchanged and remains the allowed/exclusion set.
`assert_refit_outputs` validates:
- the three bound splits;
- the raw/base and base/BIO metrics;
- the NMI and analysis bindings;
- that the R files are unchanged.

The T records are single-use:
- `ancT9_attempt_<tag>.json` and `ancT9_entry_<tag>.json` per cell;
- the T record;
- the snapshot;
- the receipt `ancT9_test_complete.json`.

## 7. Failure rule and limits

**Failure rule.** One attempt. On any of the following, preserve all reservations, entry claims,
records and partial outputs, stop and settle through the supervisor, and return to the audit:
- a refusal;
- a producer failure;
- boundary drift;
- a resource stop;
- unresolved supervision.

There is no retry, partial continuation, replacement namespace, cleanup or accounting reset.

**Limits.**
- The full-T metrics will be the first official-test values of the new model. Reporting them
  (means and SDs over the three seeds, next to the incumbent) is descriptive. No test was
  pre-specified for them in this stage.
- They do not reopen F.
- The downstream TODO items and the paper remain separate gates.
