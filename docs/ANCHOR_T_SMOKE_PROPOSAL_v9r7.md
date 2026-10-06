# Anchor model — r7 stage-T smoke: exact request, command and resource plan (audits §768–§772)

**PREPARATION ONLY. Nothing has been executed for T.**
- Nothing was loaded: no checkpoint or config load, no test extraction, no forward pass, no
  evaluation, no payload hashing, no producer.
- The request was rendered from JSON/text metadata under the named-open guard.
- T runs only after its own approval line names this exact request.

## 1. Receipt of §768–§772

All five sections were read in full.

| Section | Content |
|---|---|
| §768 | verified the completed r7 R smoke (receipt `796e70d1…`, snapshot `aa7476e7…`, settlement, cumulative charge 77.873 s) and asked for this request |
| §769 | verified the §767 hook repair and the archived R evidence (`db39989`) |
| §770 | rendered an independent T-smoke metadata preview, request `8ad639cc…` |
| §771 | records the user's reaffirmed decision: `axis_center=anchors` for all four datasets, with F, N, lambdas, seeds and pins unchanged |
| §772 | asks for this submission |

- §758 and §763 are spent.
- No T, full R, retry or downstream run is authorized until its own line.
- The order stays R smoke → T smoke → full R → full T.

## 2. The request

| Item | Value |
|---|---|
| Semantic request SHA256 | **`8ad639cc7c1eaa28fb83925416b2d35a4bd21ee339b3803fc4e3e18de9d8f748`** |
| Against the §770 preview | identical digest; all fields equal (compared field by field with the audit's `t_metadata_request.json`); no discrepancy |
| Manifest | r7 `2f24fc80bb83e2c73b17ec473c2731ade6d34a356e02414e6719fe9e9bec947c`, tree `/data/yschoi/gdna_anchor_refit_v9r6` at `db39989` (clean) |
| F | `ancF_candidate_v1.json` `5165f5dc…`, §744 acceptance `13ef776b…` |
| R receipt | `artifacts/anchor_confirmation/ancRsmk9r6_sweep_complete.json` `796e70d1…`, §763 approval |
| Mode, epochs | `smoke`, `refit_epochs` 1 |
| Namespace, GPUs | `ancTsmk9`, `gpu_count` 1 |
| Cell | one: Flickr25K / anchors / N4 / seed 42, terminal epoch 0 |
| Cell pins | checkpoint `748824…`, runtime witness `d22bd0f2…`, config `c1a3bc2a…`, plus the record, trainer evidence and sealed-recipe pins |
| Chain | five producers: the T entry (query/DB extraction + raw evaluation), train extraction, BIO evaluation, NMI, analysis seal |
| Outputs | the contract's 13 names, each created once |

**Render evidence** (`artifacts/anchor_confirmation/refit_v9/t_smoke_request_r7/`, with `SHA256SUMS`):
- The render ran under `refit_v9/guarded_run.py --launcher-bundle`. The guard computes the pre-import
  bundle under a throwaway module name, so the launcher's binding check passes as in a real exec.
- Exit 0 and **0 refused opens**.
- It made 12 named opens of the 11 allow-listed JSON/ledger files (below) and 195 source opens.
- It made 206 child invocations, all git source queries.
- These are the same counts as §770.

The 11 allow-listed files:
- the r7 manifest;
- the R receipt, snapshot and cell record;
- the R run's `phase3_campaign_binding.json` and `model_state_dict.pth.runtime.json`;
- the F record;
- the v7 `ancS7` snapshot;
- `p3rfB_refit_aggregate.json` and `selected_n.json`;
- the audit ledger.

The launcher argument list used for the render (`launcher_argv_plan_base.txt`):

```
scripts/phase3_selection_matrix.py --anchor-confirm test
  --anchor-manifest /data/yschoi/gdna_anchor_refit_v9r6/artifacts/anchor_confirmation/authority_manifest_v9r7.json
  --anchor-manifest-sha256 2f24fc80bb83e2c73b17ec473c2731ade6d34a356e02414e6719fe9e9bec947c
  --anchor-refit-receipt /data/yschoi/gdna_anchor_refit_v9r6/artifacts/anchor_confirmation/ancRsmk9r6_sweep_complete.json
  --anchor-refit-receipt-sha256 796e70d1b54a4f02e58e0a222fb1ee1261e34504857c5ff27e09296c86e4b0a3
  --namespace ancTsmk9 --smoke --gpus 0 --plan
```

The GPU index does not enter the request; only `gpu_count` does.

## 3. The command proposed for execution (after its approval line)

Run in tmux **`ancTsmk9_v9r7`** through `/data/yschoi/gdna_p3exec_authority/bin/tmux_run.sh`, on one GPU
chosen at launch by inventory (idle, no compute process):

```
env -C /data/yschoi/gdna_anchor_refit_v9r6 -u PYTHONPATH -u CUDA_VISIBLE_DEVICES GDNA_NUM_SEMANTIC_PARTS=5 \
  /home/yschoi/.conda/envs/dna_hashing/bin/python scripts/anchor_confirm_supervisor.py \
  --manifest /data/yschoi/gdna_anchor_refit_v9r6/artifacts/anchor_confirmation/authority_manifest_v9r7.json \
  --manifest-sha256 2f24fc80bb83e2c73b17ec473c2731ade6d34a356e02414e6719fe9e9bec947c \
  --stage stage-T-smoke --planned-cells 1 --ops-root /home/yschoi/gdna_anchorRT_ops --watch-path /home/yschoi -- \
  /home/yschoi/.conda/envs/dna_hashing/bin/python scripts/phase3_selection_matrix.py --anchor-confirm test \
  --anchor-manifest /data/yschoi/gdna_anchor_refit_v9r6/artifacts/anchor_confirmation/authority_manifest_v9r7.json \
  --anchor-manifest-sha256 2f24fc80bb83e2c73b17ec473c2731ade6d34a356e02414e6719fe9e9bec947c \
  --anchor-refit-receipt /data/yschoi/gdna_anchor_refit_v9r6/artifacts/anchor_confirmation/ancRsmk9r6_sweep_complete.json \
  --anchor-refit-receipt-sha256 796e70d1b54a4f02e58e0a222fb1ee1261e34504857c5ff27e09296c86e4b0a3 \
  --namespace ancTsmk9 --smoke --gpus <GPU> --anchor-approval-section <SECTION>
```

This is the render's argument list without `--plan`, with the approval section added. There is no
carried admission, recipe override, `--run` or other cell.

**Gate in the same command as the launch:**
- the tree is clean at its submitted HEAD;
- the manifest digest matches;
- tmux `ancTsmk9_v9r7` is absent;
- the chosen GPU has no compute process;
- no `ancTsmk9` record, reservation or attempt exists;
- the R/T ledger has exactly the two settled starts (r5, r7 R smoke);
- the R run directory holds none of the 13 outputs.

## 4. Resource plan

| Item | Plan |
|---|---|
| Ledger | the same append-only R/T ledger `/home/yschoi/gdna_anchorRT_ops` (now `bb359bb4…`, 2 starts / 2 finals); cumulative 77.873 s of the 80,000-s envelope; no reset, no cross-charge |
| Supervision | stage `stage-T-smoke`: **one logical cell, five managed producer attempts** (`CHILDREN_PER_CELL` 5 → `planned_attempts` 5; a sixth is excess) |
| Limits | 1 GPU, wall 10,800 s including admission; poll 1 s, watchdog 10 s, stop bound 120 s, headroom 130 GPU-s |
| Storage rule | 10 GiB + 0.75 GiB per unfinished cell (11.54 GB) |
| Outputs | written once into the R smoke's run directory `/home/yschoi/gdna_anchorRT_result/261006+flickr25k_setting1_ancRsmk9r6_…`. The R record and its pins are not rewritten |
| T records | in the r7 tree's record directory: `ancTsmk9_attempt_…`, `ancTsmk9_entry_…`, T record, snapshot, receipt `ancTsmk9_test_complete.json` |

**Current checks (2026-10-06, before submission):**
- All six GPUs idle (11 MiB, 0 %, no compute process).
- `/home/yschoi` free: 375,715,524,608 B.
- tmux `ancTsmk9_v9r7` absent (tmux answered "can't find session").
- No `ancTsmk9` file in the record directory.
- The R run directory holds only the R outputs and none of the 13 T outputs.

**Expected duration.** Each piece is short: Flickr25K query/DB extraction from cached features, train
extraction, BIO and NMI on one cell, and the seal. The whole run should take minutes. The wall limit
stays the pinned 10,800 s.

## 5. Failure rule

One attempt. On any of the following, preserve all evidence, stop, settle (or report an unresolved
ledger) and return to the audit:
- a refusal, admission error or producer failure;
- drift at any T boundary;
- a resource stop;
- unresolved supervision.

There is no retry, replacement namespace, overwritten reservation, partial continuation or deletion.

## 6. Limits stated plainly

- **First real use of two r6/r7 mechanisms on a real checkpoint (0.65 GB):**
  - the verified-runtime binding;
  - the T entry and boundary rechecks.

  The preparation evidence for both is synthetic or metadata-only.
- **The T smoke evaluates a one-epoch diagnostic checkpoint on the official split.** Its numbers are
  diagnostics, not paper evidence.
- **Memory.** Each T consumer holds the verified checkpoint bytes while it loads.
- **Full R** (carried from the r7 R smoke snapshot) is a separate proposal, not implied here.
