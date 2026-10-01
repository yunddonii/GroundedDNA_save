# Anchor Confirmation — the probe package (response to audit §729)

**Status: preparation only. Nothing here is executable until the audit ledger holds a `probe`
approval line naming the request below.**
- Since §729 no probe ran. Nothing was deserialised, no dataset was built, and no GPU was used.
- The builder read 55 files, all JSON or CSV; `metadata_check.json` lists them with their digests.
  It read no `config.pt` or checkpoint byte. For that reason it read the frozen N record without
  replaying it; each probe replays it at admission.
- The D evidence is unchanged:
  - the eight records, receipt `4272eeac…` and snapshot `ancD7_snapshot_64f311500273a3ed.json`;
  - the operations ledger is still `1275c714…`, the settlement §729.1 verified.

  S/D are not retried, no seed is added and the frozen N record is untouched.
- Branch `arch-exp-2026-09-anchor-confirm`. The generation source is unchanged since `59477d8`
  (manifest v7, 62 files, all at their pins: the builder rechecked the generation before and after it
  ran).

## 1. The package

| Item | Path (under `artifacts/anchor_confirmation/`) | SHA256 |
|---|---|---|
| Probe sources (12 coordinates) | `ancP7_probe_sources.json` | `39b304e92a94f879a71bdbfdc8b2cf9bce3209e81e09bdf460d047a4ca2b8702` |
| **Canonical probe request** | printed in `request_preview_ancP7_v7.txt` (file `baceedef…`) | **`748851142a8734703c076bbccd52af8c383dba14ee105d29235c55c92c476c44`** |
| Metadata check report | `ancP7_prep/metadata_check.json` | `f26b69dcd48d3bf5d3dc0180e2dc74be6f1c1d2858bc06328a0484c4514acfce` |
| Builder (metadata only, not a manifest member) | `ancP7_prep/build_probe_package.py` | `e02172a9c7ba32f4b4bfd2e643d9c4bac35ece12b3d5ca72d9eb38b5455441d5` |
| Chain wrapper (§6; not a manifest member) | `probe_chain_ancP7_v7.sh` | `5874fc6dc7b63be016889cac2ef95eb556d42d9dafc5e8d5c937e4c837cf1de8` |
| Chain self-test and its output | `ancP7_prep/chain_selftest.sh`, `ancP7_prep/chain_selftest.log` | `3603d768…`, `80c65fb4…` |

Inputs, each read at its pin:

| Input | SHA256 |
|---|---|
| Generation manifest v7 | `c061296309fe41203205dff65de3a1afb843c950e93c0614dfd27a8216a90128` |
| Frozen N record `ancS7_selected_n.json` (N 4/4/4/39) | `5cda7adb055ed126efb0a8ec198e06d1176ccdc57e7d38fe35ff74b74a4a92ff` |
| The audit's S source list `ancS7_select_sources.json` | `0cfab95106bfadb03e5e9750d8e5a17d7ff16523ff14b66bba1cab5519067656` |
| S receipt `ancS7_sweep_complete.json` | `5915768767e3fd12b28c3f09e1a071c26fd5cbddbbac15662c495b80c4876f6b` |
| D receipt `ancD7_sweep_complete.json` | `4272eeac4759354f94a58af13d84ab62cf78ca3457fa0d07cd379bd0ad0aa6e8` |

Code the probe runs (manifest members, unchanged): probe `scripts/anchor_confirm_code_axis.py`
`31537dde1cc8dd338221bb7fd49241cc0c618c0de444a2adf3500e86f708b19e` (the producer identity its
envelope records), reducer `79fbe131…`, launcher `2cb8343d…`, supervisor `b5eecf58…`.

## 2. The twelve coordinates (§729.2 item 2)

Each dataset's selected stage-S seed-42 cell plus its two D cells, anchors only.
- The four seed-42 entries are the audit's own entries from `ancS7_select_sources.json`, unchanged
  (an independent check found each one in that file).
- The eight D entries are built from the ancD7 receipt: the cell id is recomputed and every one of
  its eight cells is used.
- Excluded: the smoke (`ancSmk7`) and the twelve unselected S cells (N ≠ N_S).

| Dataset | N | Seed | Record | Record SHA256 | Campaign approval | Split identity |
|---|---:|---:|---|---|---|---|
| CIFAR-10 | 4 | 42 | `ancS7_cifar_A_v4_N4_s42_AXanchors_JD002.json` | `6e8fdfba192f…` | §725 stage-S-run | `f5603262…` |
| CIFAR-10 | 4 | 43 | `ancD7_cifar_A_v4_N4_s43_AXanchors_JD002.json` | `13a6fc6f53ac…` | §727 stage-D-run | `f5603262…` |
| CIFAR-10 | 4 | 44 | `ancD7_cifar_A_v4_N4_s44_AXanchors_JD002.json` | `db4fe4792ca6…` | §727 stage-D-run | `f5603262…` |
| Flickr25K | 4 | 42 | `ancS7_flickr_A_v4_N4_s42_AXanchors_P06095_JD002.json` | `a8da86d3df6e…` | §725 stage-S-run | `9f5d955d…` |
| Flickr25K | 4 | 43 | `ancD7_flickr_A_v4_N4_s43_AXanchors_P06095_JD002.json` | `70214f8311d0…` | §727 stage-D-run | `9f5d955d…` |
| Flickr25K | 4 | 44 | `ancD7_flickr_A_v4_N4_s44_AXanchors_P06095_JD002.json` | `d7f7494e2c8d…` | §727 stage-D-run | `9f5d955d…` |
| NUS-WIDE | 4 | 42 | `ancS7_nuswide_A_v4_N4_s42_AXanchors_P0408_JD005.json` | `ee824f7108b5…` | §725 stage-S-run | `815baab9…` |
| NUS-WIDE | 4 | 43 | `ancD7_nuswide_A_v4_N4_s43_AXanchors_P0408_JD005.json` | `98b51c19bc25…` | §727 stage-D-run | `815baab9…` |
| NUS-WIDE | 4 | 44 | `ancD7_nuswide_A_v4_N4_s44_AXanchors_P0408_JD005.json` | `f6daecbcc763…` | §727 stage-D-run | `815baab9…` |
| MS-COCO | 39 | 42 | `ancS7_mscoco_A_v5b_N39_s42_AXanchors_P06095_JD003.json` | `f20ebc375c37…` | §725 stage-S-run | `28447149…` |
| MS-COCO | 39 | 43 | `ancD7_mscoco_A_v5b_N39_s43_AXanchors_P06095_JD003.json` | `1512c4e9c373…` | §727 stage-D-run | `28447149…` |
| MS-COCO | 39 | 44 | `ancD7_mscoco_A_v5b_N39_s44_AXanchors_P06095_JD003.json` | `c6553b5921d1…` | §727 stage-D-run | `28447149…` |

- The full digests are in `ancP7_probe_sources.json` and `metadata_check.json`. The report also
  carries each record's run directory, `config.pt` pin, checkpoint pin and input seal.
- **The JSON/CSV-level admission passed for all twelve.** This is the reducer's `admit_metadata`, the
  same function each probe runs before its first load:
  - receipt, campaign binding and completion pins;
  - plan snapshot, with the campaign approval re-verified in the ledger now;
  - sealed recipe, trainer evidence and runtime sidecar;
  - input authority and the terminal `log.csv` row.

  One generation holds all twelve records, and each dataset's three records name one split
  identity.
- **Independent check.** Plain JSON, without the project's functions:
  - membership equals {4 datasets} × {42, 43, 44} at the frozen N;
  - every record's bytes match both its listed digest and the receipt that lists it;
  - the request digest recomputes to the same value;
  - control: one altered record digest changes the request digest.

## 3. The canonical probe request and the approval it needs (§729.2 item 2)

- **How it is formed.** The request is what the probe's own `main()` forms from these sources:
  - `probe_records` requires exactly the twelve stage-D coordinates;
  - `anchor_confirm_decision.probe_request` builds the request;
  - `_json_digest` computes the canonical-JSON SHA256.
- **Its fields:**
  - schema `anchor-confirm-request/2`, version `anchor-confirm/2`, operation `probe`;
  - manifest `c0612963…`, selection `5cda7adb…`;
  - the §8.2 population `{rows: "first 500 of the train-only validation split, ascending dataset
    index", n_images: 500, local_slots: 4, decisions: 2000}`;
  - the twelve `[dataset, "anchors", N, seed, record digest]`.
- **Digest:** `748851142a8734703c076bbccd52af8c383dba14ee105d29235c55c92c476c44`. The stage-D summary
  reducer recomputes the same request from the same record digests and requires each probe envelope
  to carry it.

The line the probes need (one `probe` line covers all twelve):

```
ANCHOR-CONFIRM-APPROVAL version=anchor-confirm/2 scope=probe manifest=c061296309fe41203205dff65de3a1afb843c950e93c0614dfd27a8216a90128 selection=5cda7adb055ed126efb0a8ec198e06d1176ccdc57e7d38fe35ff74b74a4a92ff request=748851142a8734703c076bbccd52af8c383dba14ee105d29235c55c92c476c44
```

## 4. Population and measurement (contract v3 §8.2, unchanged; §729.2 item 3)

- **Rows.** The run's own train-only validation split (`carve_val_indices`, ratio 0.1, seed 42), in
  ascending dataset index, the first 500:
  - the records' split identities give 500 validation rows for CIFAR-10 and Flickr25K, 1050 for
    NUS-WIDE and 1000 for MS-COCO;
  - fewer than 500 refuses;
  - no optimisation-train or official-test row is used.
- **Row identity.** The probe hashes the selected rows' cache identities (`row_ids_sha256`), and the
  reducer requires one row digest and one split identity across the three seeds of a dataset.
  - The digest cannot be computed in advance without reading the binary row mapping.
  - The split identity each probe must reproduce is in §2.
- **Forward.** Deployment: eval mode, `cached_text_part_raw=None` and `cached_has_text=None`, so no
  caption reaches the model. The output is the four local slots' quantised codewords
  `quantized_tokens[:, 1:, :]`.
- **Target.** The same rows' cached caption features through the same checkpoint's text adapter
  (`_adapt_pooled_text_for_loss`). Captions are only the target.
- **Scoring.** Centring in float64 per slot over all 500 rows, then L2 normalisation; a centred norm
  ≤ 1e-12 refuses. A hit needs the own-axis cosine **strictly** above the other three, and a tie
  is a miss. Each probe makes 2000 decisions.
- **Envelope.** Integer hits, ties and total, and the unrounded ratio, plus:
  - the coordinate, record, checkpoint and `config.pt` digests;
  - the execution authority, the manifest, the probe approval and the request;
  - the split and split identity, the row digest and the caption target with its seal;
  - the producer digest.
- **No exploration.** Exactly the terminal checkpoint each record pins (the runtime sidecar must be
  epoch N). There is no checkpoint search and no population change.

## 5. Admission and dependency chain (what runs before each probe's first load)

1. **Supervisor** (`--stage probe`):
   - the 62 manifest files at their pins;
   - the ledger has no unfinished, unclean or unresolved run;
   - the budget and space rules of §7–§8.
2. **Probe entry:**
   - the generation recheck;
   - replay of the frozen N record (the 16 S records at JSON level, their logs, and their `config.pt`
     bytes hashed, never deserialised);
   - the sources must be exactly the twelve stage-D coordinates;
   - the request is recomputed;
   - `audit_approval(section, "probe", manifest, selection, request)` requires exactly one matching
     line.
3. **Per coordinate:**
   - it is a stage-D coordinate;
   - `admit_metadata`, with the campaign approvals §725/§727 re-verified;
   - the `config.pt` bytes and the checkpoint bytes match their pins;
   - the sidecar is at the terminal epoch;
   - `GDNA_NUM_SEMANTIC_PARTS=5`;
   - the generation is rechecked.
4. **Only then the first deserialisation.**
   - It reads `config.pt`. The trainer's own input admission (`_phase3_input_authority_from_args`:
     seal stats and runtime paths) must then reproduce the record's seal, aggregate and split
     identity.
   - Next it loads the model, and the checkpoint must load with no missing or unexpected key.
   - It builds the dataset, and `_assert_phase3_runtime_rows` must hold.
   - Then it measures.
5. **Before writing**, a generation recheck. The output is written once (`O_EXCL`).

## 6. Supervised commands and output paths (§729.2 item 4)

**Before launch** (after the approval line exists):
- `sha256sum` must show the chain `5874fc6d…` and the sources `39b304e9…`;
- `nvidia-smi` gives one GPU with no compute process;
- the ledger's last record is a `final`.

```
/data/yschoi/gdna_p3exec_authority/bin/tmux_run.sh ancP7_v7 \
  env -C /data/yschoi/gdna_anchor_confirm_v1 -u PYTHONPATH -u CUDA_VISIBLE_DEVICES GDNA_NUM_SEMANTIC_PARTS=5 \
  bash artifacts/anchor_confirmation/probe_chain_ancP7_v7.sh <1 free GPU index> <approving ledger section>
```

**What the chain runs.** For each coordinate, in this order:
- Flickr25K s42/43/44, the cheapest, so a defect costs one short probe;
- CIFAR-10 s42/43/44;
- NUS-WIDE s42/43/44;
- MS-COCO s42/43/44.

Each one is exactly:

```
CUDA_VISIBLE_DEVICES=<GPU> /home/yschoi/.conda/envs/dna_hashing/bin/python scripts/anchor_confirm_supervisor.py \
  --manifest $REC/authority_manifest_v7.json \
  --manifest-sha256 c061296309fe41203205dff65de3a1afb843c950e93c0614dfd27a8216a90128 \
  --stage probe --planned-cells 0 --label ancP7:<coordinate> --watch-path $REC -- \
  /home/yschoi/.conda/envs/dna_hashing/bin/python scripts/anchor_confirm_code_axis.py \
  --sources $REC/ancP7_probe_sources.json \
  --sources-sha256 39b304e92a94f879a71bdbfdc8b2cf9bce3209e81e09bdf460d047a4ca2b8702 \
  --manifest $REC/authority_manifest_v7.json \
  --manifest-sha256 c061296309fe41203205dff65de3a1afb843c950e93c0614dfd27a8216a90128 \
  --selection $REC/ancS7_selected_n.json \
  --selection-sha256 5cda7adb055ed126efb0a8ec198e06d1176ccdc57e7d38fe35ff74b74a4a92ff \
  --approval-section <section> --coordinate <dataset>:anchors:<N>:<seed> \
  --out $REC/ancP7_probes/<dataset>_anchors_N<N>_s<seed>.json --device cuda:0
```

where `$REC` = `/data/yschoi/gdna_anchor_confirm_v1/artifacts/anchor_confirmation`.
- **Outputs.** Twelve JSON envelopes in the new directory `$REC/ancP7_probes/`. Supervisor records go
  to the same operations ledger; command logs go to `/home/yschoi/gdna_anchor4_ops/<run_id>.command.log`.
- **The wrapper adds no measurement logic.** It:
  - checks its two integer arguments and the environment (`GDNA_NUM_SEMANTIC_PARTS=5`, `PYTHONPATH`
    and `CUDA_VISIBLE_DEVICES` unset);
  - makes the output directory with plain `mkdir`, so an existing directory refuses and nothing is
    resumed or overwritten;
  - runs the twelve supervised commands one at a time (the supervisor's ledger lock admits one);
  - **stops at the first nonzero exit and exits with that status.**
- **Self-test** (`chain_selftest.log`). It ran on a sandbox copy that differs in three lines (working
  directory, interpreter, records root), with a stub interpreter; nothing real ran.
  - Eleven scenarios pass: the full order and call shape, a failure at probe 1, 3 or 12 (stops,
    exit 1/5/4), an existing directory, bad arguments, and each environment violation.
  - Three mutants are each caught: `exit 0` in place of the child status, no stop on failure, and
    `mkdir -p`.
  - No `ancP7_probes/` directory exists in the real records root.
- **Alternative.** If the audit prefers no non-manifest wrapper, the same twelve supervised commands
  can be run one at a time in this order, each after the previous one's `final` record.

## 7. Budget (settled cumulative charge, §729.1)

- **Prior charge:** 19,033.569830481894 s (5.287 GPU-h), leaving 25,966.430169518106 s under the
  45,000-s S/D/probe ceiling.
- **Per probe (supervisor stage `probe`, unchanged code):**
  - 1 GPU and a 15-minute wall limit; it stops at 770 s (limit − 120-s stop bound − 10-s watchdog);
  - headroom 130 s;
  - the attempt is the probe process itself (self mode), so the allowance is 0, and the charge is its
    process lifetime.
- **Planning:**
  - The contract §12 estimate is ≈ 0.4 GPU-h (≤ 2 min each).
  - The data are small: 5,000–10,500 training rows load from feature caches. The D trainers' start-up
    was short. The six N = 4 D cells (CIFAR-10, Flickr25K and NUS-WIDE) lived 119.9–241.9 s each,
    including start-up, five epochs and validation (the ledger's attempt lifetimes for run
    `20260929T140924Z-9b031227`).
  - So 1–3 min per probe is expected, about 12–36 min in all. That is an estimate, not a bound.
- **Bound:** 12 × 900 s = 10,800 s. The cumulative charge would then be ≤ 29,833.57 s (8.29 GPU-h),
  under the ceiling even if every probe ran to its limit. The wall time is ≤ 3 h in that case.
- This is process-lifetime accounting, not measured GPU utilisation.

## 8. Storage policy

- **Where.** The outputs are twelve small JSON files (tens of KB in all) on `/data`, in the records
  root, as contract v3 §14 places records. No run directory or checkpoint is written. The ledger and
  command logs are on `/` (373.6 GB free).
- **Space.** `/data` had 38,179,926,016 bytes (35.56 GiB) free on 2026-10-01 (`df -B1`, 99 % used,
  shared with other work).
- **Supervisor rules on the watched records root:**
  - ≥ 10 GiB at start (planned cells 0);
  - ≥ 10.75 GiB while running (10 GiB + 1 GPU × 0.75 GiB).
- A refusal or stop on space is the designed outcome. Nothing is deleted or moved.

## 9. Payload-reading preparation commands

None is needed and none is requested. Every byte check of a binary (`config.pt`, checkpoint) and every
deserialisation happens inside the probe, under its approval (§5). A stand-alone byte-only pin check of
the 24 binaries (about 7.9 GB) would only repeat the probe's own step 3, so it is not proposed.

## 10. Static pre-verification and limits

- **The probe's success path has never run on a real checkpoint.** The tests drive its admission
  and refusal paths with fixtures.
  - **Static comparison.** I compared its calls with the current manifest sources: the
    `load_dataset` keywords, the model `forward` keywords and its `quantized_tokens` key, the
    dataset item keys (`cached_visual_tokens_raw`, `cached_visual_global`,
    `cached_text_part_raw`), `_adapt_pooled_text_for_loss` on `[B, 5, D]`,
    `_assert_phase3_runtime_rows` and `apply_inference_epoch`. I found no mismatch.
  - **The row check should run, not return early.** The trainer saves `config.pt`
    (`Config.save_arg`) with the same attribute filter that writes `args.txt` (`print_info`), and the
    D record's `args.txt` carries `_phase3_input_authority`. Without that attribute
    `_assert_phase3_runtime_rows` returns early. The pickle itself was not opened to confirm this.
  - **If a defect remains**, it shows as a refusal (rc 1) at the first, cheapest coordinate, and the
    chain stops. A fix is a source change and needs its own generation and audit.
- **GPU leases.** A probe takes no GPU lease (leases are the launcher's mechanism). The GPU is
  chosen from `nvidia-smi` at launch.
- **The wrapper is not a manifest member.** Its digest is pinned here. The approval line's fields are
  fixed (manifest, selection, request), so the line cannot name it.

## 11. Not requested here

- Probe execution before its approval.
- The stage-D summary (`anchor_confirm_decision.py decide`). It needs these twelve probe digests
  added to the sources, and is proposed after the probes.
- The required TODO 13–15 lambda proposal (L).
- The freeze F, R/T, official-test evaluation and everything downstream.

## 12. Requested decisions

1. A `probe` line naming manifest `c0612963…`, selection `5cda7adb…` and request `74885114…`
   (exact text in §3).
2. Whether the chain wrapper `5874fc6d…` is acceptable, or the twelve commands should run one at
   a time (§6).

Ledger at submission: SHA256 `2b6bb32802ba4f74e6096b877d8097cd4335078adc50f4033fc223fdb64e2a6f`, 56593 lines, last section §729.
