# Anchor model — r7 full stage R: exact request, command and resource plan (audits §776–§779)

**PREPARATION ONLY. Nothing has been executed for full R.**
- Rendering and previews read source, JSON and text only, under the named-open guard.
- Nothing was loaded: no payload verifier, no checkpoint or config load, no training, no test access.
- The full R run happens only after its own approval line names this exact request.

## 1. Receipt of §776–§777

- **§776** corrected the output count. The 13 `OFFICIAL_TEST_OUTPUTS` names are an exclusion/allowed
  set; a T cell's required outputs are the 12 names other than `evaluation_siglip2_bit2.json`, plus a
  verified absence of that name.
- **§777** accepted the T smoke in diagnostic scope (receipt `613fb6fe…`, cumulative charge 514.225 s)
  and asked for this proposal:
  - namespace `ancR9`;
  - twelve scratch refits, CIFAR-10/Flickr25K/NUS-WIDE/MS-COCO × seeds 42/43/44, all anchors;
  - the F values, with N 4/4/4/39;
  - carried historical admission from the r7 R-smoke snapshot.
- The one-epoch smokes are not reused as training results.
- §758, §763 and §773 are spent.
- Full T and all downstream work are later, separate gates.
- **§778** verified the T archive (`e137e80`).
- **§779** rendered an independent reference for this request: semantic SHA256 `2aa9bf99…`, with the
  snapshot bound by its byte digest `e134c9ba…`, not by its semantic digest. It also exercised the
  carry-record predicate.

## 2. The request

| Item | Value |
|---|---|
| Semantic request SHA256 | **`2aa9bf99b5cb4b5ee73a4a4ac78767d62b297e0950f590bbbcac7b705afd179d`** |
| Mode, namespace | `run`, `ancR9`; 12 cells; `gpu_count` 4; `epochs` none (each cell's N+1) |
| Manifest | r7 `2f24fc80bb83e2c73b17ec473c2731ade6d34a356e02414e6719fe9e9bec947c` (tree `/data/yschoi/gdna_anchor_refit_v9r6`, clean) |
| F | `ancF_candidate_v1.json` `5165f5dc…`, §744 acceptance `13ef776b…`, per-dataset validated-recipe digests in `freeze.recipes_sha256` |
| Input seals | the four approved refit seals: CIFAR-10 `943bb953…`, Flickr25K `71506fd1…`, NUS-WIDE `05b7d24b…`, MS-COCO `a9d49e5d…` |
| Carried admission | `admission_authority` = `ancRsmk9r6_snapshot_aa7476e79f15a159.json`. File bytes SHA256 `e134c9ba56fe6cda9e0051b2250e2a977e34980a18cee52a92188e164d725d8b`; semantic SHA256 `aa7476e7…` (the §777 identity) |
| Result root / records | `/home/yschoi/gdna_anchorRT_result` / the r7 tree's `artifacts/anchor_confirmation` |

**Against the audit's §779 reference** (`full_r_metadata/request.json`): identical digest, and every
field equal. Against the earlier full-admission preview (`26665ca4…`, `r7_evidence/renders/`), the
request differs only in `admission_authority`.

**F values per dataset:**

| Dataset | N | lambda_wasserstein / bu / text_hash_ntxent | top-p | lambda_codon_joint |
|---|---|---|---|---|
| CIFAR-10 | 4 | 0.15 / 0.02 / 0.05 | 0.3–0.7 | 0.02 |
| Flickr25K | 4 | 0.15 / 0.02 / 0.05 | 0.6–0.95 | 0.02 |
| NUS-WIDE | 4 | 0.15 / 0.02 / 0.05 | 0.4–0.8 | 0.05 |
| MS-COCO | 39 | 0.05 / 0.02 / 0.10 | 0.6–0.95 | 0.03 |

**Full-train mapping** (`refit_protocol_fields`; `protocol_values.json`):
- `epoch` = N+1 (5, or 40 for MS-COCO);
- LR and Sinkhorn horizons unset, so they resolve to N+1;
- `stop_after_epoch` N;
- `val_split_ratio` 0.0 and `selection_mode` refit;
- `keep_final_checkpoint` False and `final_epoch_eval` True;
- train-only local whitening (`text_whiten_trainOnly_localOnly.npz`);
- `random_seed` = the cell seed;
- anchors, base distance, Gumbel off.

The official test stays withheld, enforced by the trainer gate and the launcher.

**Per-cell typed-recipe mapping, bound mode** (`bound_recipe_mapping.json`). This is the check
execution repeats after input admission, rendered here with the admitted refit seals. In all 12 cells
the anchor recipe differs from the F record's validated recipe only in the contracted refit fields:
- **12 fields at seed 42:** `epoch`, `final_epoch_eval`, `keep_final_checkpoint`,
  `lr_schedule_horizon`, `selection_mode`, `sinkhorn_schedule_horizon`, `text_whiten_npz`,
  `val_split_ratio`, plus the four seal fields `phase3_input_seal`, `phase3_input_seal_sha256`,
  `phase3_input_aggregate_sha256` and `phase3_split_identity_sha256`.
- **13 fields at seeds 43/44:** the same plus `random_seed`.
- **0 fields outside the contract.**
- The CLIP snapshot identity equals F's.
- Each rendered control differs from the anchors arm in `axis_center` alone.

The plan-mode listing (`plan_mode_recipe_mapping.json`, 19/20 fields) also shows the CLIP and HF
identity fields. Those fields are unrendered without admitted seals and are exempt by design in
that mode. They are not differences.

| Cell | Recipe digest | Run tag |
|---|---|---|
| cifar10 N4 s42 | `4535ba2f477075e4…` | `ancR9_cifar_A_v4_refit_N4_s42_AXanchors_JD002` |
| cifar10 N4 s43 | `b3f13cd84dff24ed…` | `ancR9_cifar_A_v4_refit_N4_s43_AXanchors_JD002` |
| cifar10 N4 s44 | `7b91164ddc75fa11…` | `ancR9_cifar_A_v4_refit_N4_s44_AXanchors_JD002` |
| flickr25k N4 s42 | `9bbeea8069c82216…` | `ancR9_flickr_A_v4_refit_N4_s42_AXanchors_P06095_JD002` |
| flickr25k N4 s43 | `d3d5b216730592f1…` | `ancR9_flickr_A_v4_refit_N4_s43_AXanchors_P06095_JD002` |
| flickr25k N4 s44 | `3425d7a382de6a1c…` | `ancR9_flickr_A_v4_refit_N4_s44_AXanchors_P06095_JD002` |
| nuswide N4 s42 | `e8df2579ba2b75bb…` | `ancR9_nuswide_A_v4_refit_N4_s42_AXanchors_P0408_JD005` |
| nuswide N4 s43 | `f614eafb7ccfcc27…` | `ancR9_nuswide_A_v4_refit_N4_s43_AXanchors_P0408_JD005` |
| nuswide N4 s44 | `8e9d0a77003f25ae…` | `ancR9_nuswide_A_v4_refit_N4_s44_AXanchors_P0408_JD005` |
| mscoco N39 s42 | `e78dc8521685d966…` | `ancR9_mscoco_A_v5b_refit_N39_s42_AXanchors_P06095_JD003` |
| mscoco N39 s43 | `a3bb7a509510be2a…` | `ancR9_mscoco_A_v5b_refit_N39_s43_AXanchors_P06095_JD003` |
| mscoco N39 s44 | `a08dc774ae414e62…` | `ancR9_mscoco_A_v5b_refit_N39_s44_AXanchors_P06095_JD003` |

## 3. Guarded evidence (`artifacts/anchor_confirmation/refit_v9/r_run_request_r7/`, `SHA256SUMS`)

| Check | Result |
|---|---|
| **Request render** (`guarded_run.py --launcher-bundle`, `--plan`) | exit 0, **0 refused**; 11 named opens of 11 allow-listed files (the r7 manifest, R-smoke snapshot, F record, v7 `ancS7` snapshot, p3rfB aggregate, selected_n, audit ledger, four refit seal JSONs); 245 source opens; 230 child invocations: 206 git source queries and 24 pinned wrapper renders (12 coordinates × 2 arms, through the argv-capture shim in temporary directories; the `none` arm is a recipe comparison only). Matches §779 |
| **Carried-admission preview** (`carried_admission_preview.py`, the launcher's own functions) | exit 0, 0 refused |
| — `anchor_carried_admission_refusal` | `None`: the snapshot records the pinned historical verifier's rc-0 report for exactly the four seals on these bytes |
| — `admission_is_full` | False |
| — stats-only seal check (`verify_seal_stats`) | passed for all four seals, 0.02 s: aggregate digest; lstat/link/target stat of every sealed file record (66/62/62/62); directory identity; cache filename inventory. No content read |
| — bound-mode recipe mapping | as in §2 |
| **Values preview** (`protocol_values.py`) | exit 0, 0 refused |

The previews are preparation evidence only. The real command repeats the carried-admission and
stats-only checks itself, after its approval line.

## 4. The command proposed for execution (after its approval line)

tmux **`ancR9_v9r7`** through `/data/yschoi/gdna_p3exec_authority/bin/tmux_run.sh`; the same form as
the §763 R smoke:

```
env -C /data/yschoi/gdna_anchor_refit_v9r6 -u PYTHONPATH -u CUDA_VISIBLE_DEVICES GDNA_NUM_SEMANTIC_PARTS=5 \
  /home/yschoi/.conda/envs/dna_hashing/bin/python scripts/anchor_confirm_supervisor.py \
  --manifest artifacts/anchor_confirmation/authority_manifest_v9r7.json \
  --manifest-sha256 2f24fc80bb83e2c73b17ec473c2731ade6d34a356e02414e6719fe9e9bec947c \
  --stage stage-R-run --planned-cells 12 --ops-root /home/yschoi/gdna_anchorRT_ops --watch-path /home/yschoi -- \
  /home/yschoi/.conda/envs/dna_hashing/bin/python scripts/phase3_selection_matrix.py \
  --anchor-confirm refit --anchor-manifest artifacts/anchor_confirmation/authority_manifest_v9r7.json \
  --anchor-manifest-sha256 2f24fc80bb83e2c73b17ec473c2731ade6d34a356e02414e6719fe9e9bec947c \
  --result-root /home/yschoi/gdna_anchorRT_result \
  --input-seal cifar10:refit=/data/yschoi/gdna_p3exec_seals/cifar10.refit.input-seal.json \
  --input-seal flickr25k:refit=/data/yschoi/gdna_p3exec_seals/flickr25k.refit.input-seal.json \
  --input-seal nuswide:refit=/data/yschoi/gdna_p3exec_seals/nuswide.refit.input-seal.json \
  --input-seal mscoco:refit=/data/yschoi/gdna_p3exec_seals/mscoco.refit.input-seal.json \
  --admission-authority /data/yschoi/gdna_anchor_refit_v9r6/artifacts/anchor_confirmation/ancRsmk9r6_snapshot_aa7476e79f15a159.json \
  --namespace ancR9 --run --gpus 0,1,2,3 --anchor-approval-section <SECTION>
```

- This is the render's argument list (`launcher_args_plan_base.txt`) without `--plan`.
- There is no `--only`, `--epochs`, recipe override or extra cell.
- The GPU indices do not enter the request; only the count (4) does.

**Proposed devices:**

| GPU | UUID |
|---|---|
| 0 | `GPU-4ac2ea6b-1925-e1ba-5463-02827507d897` |
| 1 | `GPU-47b162a4-5db7-e986-03ea-181752d39fd5` |
| 2 | `GPU-65da79f2-d9fe-007d-c5dd-7f3599156777` |
| 3 | `GPU-c1b4c38b-bf2f-40c3-a1b7-2107d7d0cfed` |

- All six are idle now.
- The launcher runs one dataset stream per GPU, with the three seeds in sequence; the snapshot
  records the exact mapping.
- If any named GPU is busy at launch, I wait or return for review. I do not substitute another
  device unless the approval names one.

**Gate in the same command as the launch** (fail closed on any unavailable observation):
- the tree is clean at its submitted HEAD;
- the manifest digest matches, and the snapshot file still hashes to `e134c9ba…`;
- the four seal JSONs match their pins;
- tmux `ancR9_v9r7` is confirmed absent ("can't find session");
- no `ancR9` record, reservation or run directory exists;
- the four named GPU UUIDs are present with no compute process;
- the R/T ledger matches its pre-launch digest, with three settled starts (r5 smoke, r7 R smoke, r7
  T smoke);
- the space is above the floor.

## 5. Resource plan

| Item | Plan |
|---|---|
| Ledger | the same append-only R/T ledger (now `56e371f7…`, 3 starts/3 finals); cumulative **514.225 s** of the 80,000-s envelope; no reset, no cross-charge |
| Supervision | `stage-R-run`: **12 logical cells, 12 managed trainer attempts** (one per cell), 4 GPUs (one per dataset stream) |
| Limits | wall 28,800 s (8 h) including admission; poll 1 s, watchdog 10 s, stop bound 120 s, headroom 130 GPU-s |
| Storage | floor 10 GiB + 12 × 0.75 GiB = 19 GiB; free now 375,959,592,960 B |
| Expected size | about 8 GB of run directories (an R-only cell holds about 0.65 GB, mostly the 655 MB checkpoint) |
| Admission | stats-only on the carried authority (seconds) plus the bound-mode recipe recheck; no full re-verification |

**Expected device time.** The estimate comes from the v7 D/S cells of the same recipes (N+1 epochs,
with a 0.1 validation split), scaled by 1/0.9 for the full train split.

| Dataset | Per cell |
|---|---|
| CIFAR-10 | 180–1,320 s |
| Flickr25K | 137–305 s |
| NUS-WIDE | 250–1,660 s |
| MS-COCO (N39) | about 1,860–1,920 s |

- **Sum over 12 cells:** 7,300–15,600 device-seconds, so cumulative ≈ 8.3k–16.7k of 80,000
  including allowances and headroom.
- **Wall:** about 1.6–1.7 h. The MS-COCO stream, three sequential N39 cells, bounds it, well inside
  8 h.
- **These are planning figures, not a measurement.**

## 6. Required versus exclusion outputs (§776)

**Stage R (this request).** Each cell must produce, in its own run directory:
- the terminal checkpoint `model_state_dict.pth` at epoch N;
- its runtime witness `model_state_dict.pth.runtime.json` (epoch N);
- `config.pt`, `args.txt`, `log.csv` (N+1 epoch rows), `run_identity.json`,
  `phase3_campaign_binding.json` and `criterion_state_dict.pth`.

The cell record carries the completion pins and `official_test: withheld`. The campaign publishes a
reservation, the snapshot (carrying the admission evidence) and the receipt `ancR9_sweep_complete.json`
over 12 cells.

The 13 `OFFICIAL_TEST_OUTPUTS` names are the R **exclusion set**: none may exist in any R run
directory, and `assert_official_test_withheld` checks this before each record is published.

**Later full T (not requested here).** The request's `outputs` field keeps the unchanged 13-name
allowed/exclusion set, with no repin. The **required** membership is the 12 names other than
`evaluation_siglip2_bit2.json`, plus a verified absence of that name. No producer makes a bit2
evaluation for the base-distance recipe, and none will be added.

## 7. Failure rule and limits

**Failure rule.** One attempt. On any of the following, preserve every reservation, record and
partial run directory, stop and settle through the supervisor, and return to the audit:
- a refusal or admission error;
- a trainer failure;
- drift;
- a resource stop;
- unresolved supervision.

There is no retry, partial continuation, replacement namespace or deletion.

**Limits.**
- The carried admission skips content re-hashing by design (stats-only after the r7 R smoke's full
  verification). A byte rewrite that preserves every stat is the documented boundary of that mode.
- The expected-time figures are estimates.
- The r7 R smoke and T smoke diagnostics carry no scientific weight here.
