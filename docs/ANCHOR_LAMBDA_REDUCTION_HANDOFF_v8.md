# Anchor model — generation v8: the stage-L CPU reduction, submitted (audit §740.2)

**Status: submission only. The reducer has NOT run.** No model or checkpoint weight, scientific
array, `config.pt` byte or GPU was touched to prepare this. Branch `arch-exp-2026-09-anchor-lambda`.

## 1. What is submitted

| Item | Value |
|---|---|
| Sources file | `artifacts/anchor_confirmation/ancL8_lambda_sources.json` |
| Sources file bytes, SHA256 | `e0c32688fdb86ac13f3bf6ede3326f633df051b391b4f3592fd20f55e785b003` |
| Generation manifest | `artifacts/anchor_confirmation/authority_manifest_v8r2.json` `58e69ae1a3bed5366da61a4d61a6cdd90c338d9ac7b0474fcf9b6e2e2f7f5bbc` |
| Reducer | `scripts/anchor_confirm_decision.py` `ae247d4a0fe7398198e0975cc81ddc0457b104b14e46faadd20d7d0f70bc7dd9` (closure member) |
| Launcher module it imports | `scripts/phase3_selection_matrix.py` `61dafdc8bf480e10386bb762e0cddf4bd828948377dbf054ca0c0b7c9e7ef4dc` (closure member) |
| Output (fresh, write-once) | `artifacts/anchor_confirmation/ancL8_lambda_decision.json`; absent now; written with `O_CREAT\|O_EXCL`, mode 0444 |
| Interpreter | `/home/yschoi/.conda/envs/dna_hashing/bin/python` |

**The sources file** binds exactly this campaign: six coordinates (Flickr25K, anchors, N4, seed 42),
each `source: receipt`, with the label the reducer keys on (`lambda`):

| `lambda` label | Record | Record SHA256 |
|---|---|---|
| `incumbent` | `ancL8_flickr_A_v4_N4_s42_AXanchors_P06095_JD002.json` | `36e2be8e790ab4daa9faccfe96007a8e839da40b551e02746bcd1dbe8ccdd753` |
| `lambda_wasserstein=0.30` | `…_LW030_P06095_JD002.json` | `8537da4f4ddc9195873e7f6b29923410531fbe60db73b32247e02f6361d43504` |
| `lambda_wasserstein=0.50` | `…_LW050_P06095_JD002.json` | `2df1d039defd82ae3a54a881e6a492192d0a294ea76709c7c54730362979857d` |
| `lambda_bu=0` | `…_LBU0_P06095_JD002.json` | `019d9c0dc48c72fb39526c21315c71bcac60a14582e49ec05c2b1ac959d069c8` |
| `lambda_text_hash_ntxent=0.025` | `…_LTH0025_P06095_JD002.json` | `9fb3dd789c23860eeb96b3129746523f502070a820388a947d96c1814ef4ec10` |
| `lambda_text_hash_ntxent=0.10` | `…_LTH010_P06095_JD002.json` | `3aaeacbfc1c542f547094002a5b1f0ee71f903be3aeb65cd93b308da98785a29` |

All six name the one receipt `ancL8_sweep_complete.json`
`0991f0a031eea624de9ff42cb96c034589e596307a0bf61c2d9710f58c24b838` by absolute path. Paths are absolute
under `/data/yschoi/gdna_anchor_lambda_v8/artifacts/anchor_confirmation/`. Serialisation:
`json.dumps(obj, indent=2, sort_keys=True) + "\n"`, coordinates in run order.

**How it was built.** A scratch script derived each coordinate from the receipt's cell list (record
name and digest), re-hashed each record file, and wrote the file once with `O_EXCL`. It read
JSON/CSV only and does not import the reducer. Before writing, it checked the JSON-level conditions
the reducer's admission tests: stage `select`, not smoke, candidate cell, 90/10 split seed 42,
terminal epoch 4, raw base-Hamming mAP@R, receipt completion pins, campaign binding, request digest
`daf99128…`, input authority, and the digests of each binding, sidecar and `log.csv`. All hold for
6/6. A positive control (a mutated request digest and metric name) made it fail on every cell. This
pre-check is advisory; the reducer's own checks are the official ones.

## 2. The exact command (not run)

```bash
env -C /data/yschoi/gdna_anchor_lambda_v8 -u PYTHONPATH \
  CUDA_VISIBLE_DEVICES= GDNA_NUM_SEMANTIC_PARTS=5 PYTHONDONTWRITEBYTECODE=1 \
  /home/yschoi/.conda/envs/dna_hashing/bin/python scripts/anchor_confirm_decision.py lambda \
  --sources artifacts/anchor_confirmation/ancL8_lambda_sources.json \
  --sources-sha256 e0c32688fdb86ac13f3bf6ede3326f633df051b391b4f3592fd20f55e785b003 \
  --manifest artifacts/anchor_confirmation/authority_manifest_v8r2.json \
  --manifest-sha256 58e69ae1a3bed5366da61a4d61a6cdd90c338d9ac7b0474fcf9b6e2e2f7f5bbc \
  --out artifacts/anchor_confirmation/ancL8_lambda_decision.json
```

- **No `--selection`.** Stage L binds the v7 frozen N record `5cda7adb…` by its pinned constant; the
  CLI refuses `--selection` in stage `lambda`.
- **No new approval line is needed by the code.** For each cell the reducer re-reads the ledger and
  requires §739's `stage-L-run` line, unchanged, naming manifest `58e69ae1…`, selection `5cda7adb…` and
  request `daf99128…`. Permission to run this command is requested as prose, as for §726.3 and §731.
- **Environment.** CPU only (`CUDA_VISIBLE_DEVICES` empty), `PYTHONPATH` removed,
  `GDNA_NUM_SEMANTIC_PARTS=5`, no bytecode written. Run once in the foreground; the v7 stage-D
  reduction took 6.9 s.
- **Preserved record.** A new directory `artifacts/anchor_confirmation/ancL8_reduction/` (absent now)
  receives `command.txt`, `start.txt`, `end.txt` (UTC), `stdout.txt`, `stderr.txt` and `rc.txt`, as in
  v7's `ancP7_reduction/`.

## 3. Bounded read footprint (static, from the pinned source)

| Group | What is read | How often |
|---|---|---|
| Import | `scripts/anchor_confirm_decision.py` and its imports. `phase3_selection_matrix` hashes its 48 bootstrap source paths at import and runs `git rev-parse` / `git show HEAD:<path>` for each (object store only). `dna_utils/__init__` eagerly imports library code, including numpy and torch. With `CUDA_VISIBLE_DEVICES` empty there is no device and no CUDA context. | once |
| Generation | manifest v8r2 and its 64 closure files, hashed against the pins | at entry and again before publishing |
| Historical pins (JSON, read once, parsed from the hashed bytes) | `p3rfB_refit_aggregate.json` `b4f3b0df…`, `selected_n.json` `2bf6133d…` (both `/data/yschoi/gdna_p3exec/artifacts/phase3_selection/`); v7 `ancS7_selected_n.json` `5cda7adb…`, `ancP7_decision.json` `28b10a4c…`, `ancS7_sweep_complete.json` `59157687…`, `ancS7_snapshot_23e0763089474895.json` (file `6f2c0352…`; bound semantically by the receipt) | once, plus pre-publication re-verification |
| Audit ledger | `docs/PHASE1_PHASE2_REAUDIT_2026-08-14.md` in the main tree, to re-verify §739's line | once per cell (6) |
| Sources | `ancL8_lambda_sources.json` | once, plus re-verification |
| Stage-L campaign (JSON/CSV) | receipt, `ancL8_snapshot_1eab97a081863d9f.json`, 6 records; per run dir `phase3_campaign_binding.json`, `<final_checkpoint>.runtime.json` (the JSON sidecar, not the checkpoint) and `log.csv` | once each, plus re-verification |
| `config.pt` (6 files, 26,652–26,716 bytes each) | **byte hash only**, compared with each record's anchor-block pin (`verify_config_pin`) | 6 reads, plus 6 more in pre-publication re-verification = **12 byte reads, no deserialisation** |
| Write | the one output JSON | once, exclusive create |

- **Total read set the output will record** (`consumed_sha256`): 39 entries. That is 6 historical
  files, the sources file, the receipt, the snapshot, and 5 per cell (record, binding, sidecar,
  `log.csv`, `config.pt`). The manifest, the closure files and the ledger are checked outside that
  record.
- **Never read:** checkpoint or weight files (`final_checkpoint` and others), criterion payloads,
  feature caches, foil caches, captions, whitening files, split arrays, seal payloads. Nothing is
  unpickled or `torch.load`ed. No GPU lease, no training, no extraction, no probe, no official test.
- **Not observed, only derived from the code.** This footprint is read from the pinned source. The
  reducer has not run, so no runtime capture exists yet. Its stage-L path has run only on synthetic
  fixtures through `main()` in `tests/test_anchor_lambda_stage.py`. This would be its first run on
  real records.

## 4. What the run would do, and its stop rules

- **Refusal paths** (rc 1, `[anchor-reducer] REFUSED`, nothing written): any changed digest, a
  §739 line that no longer stands, a control whose sealed recipe or score differs from the v7
  seed-42 cell, a candidate whose sealed recipe moves more than its one lambda, mixed campaigns, or
  a changed closure file. A refusal is reported as is: no retry, no edit, no input change.
- **On rc 0:** report the output digest, stdout/stderr, the 39 consumed digests, and per candidate
  the unrounded delta and `qualifies`. Then stop for audit. Applying the result (the Flickr-first stop
  rule, or a new generation if the recipe changed) and the F record are separate steps.
- **The rule it applies is unchanged:** candidate − 0.7641936888306327 > T = 0.02819158958924184,
  strict and unrounded; per axis the highest qualifier wins; no qualifier keeps the incumbent.

## 5. State at submission

- Tree `622ade1` + this commit. The 64 closure files equal manifest `58e69ae1…`, and the bounded
  head-clean check passes (`bounded_tree.py head-clean`, rc 0, no mismatches).
- The S/D/probe ledger is `986bdcd1…` (19,264.613316638395 s). The L ledger is `477e447c…`
  (776.3320168741047 s of 3,600). A CPU reduction charges neither.
- No trainer, supervisor or tmux session of this chain is alive; `nvidia-smi` shows no compute
  process.
