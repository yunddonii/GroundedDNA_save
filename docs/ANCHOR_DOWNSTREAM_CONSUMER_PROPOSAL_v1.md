# Anchor model — downstream consumer/aggregate contract for stage-T results (proposal v1; audits 798, 824)

**Status: documentation-only proposal.**
- No source, test, record, manuscript or r8 file changed.
- No reader, formatter, producer or test ran.
- The r8 tree stays at the audited submission HEAD S `04e7fe871cf9fc19a3e9ed6b713038686ff31830`.
- The recovered NUS-WIDE seed-44 cell has not run. Every value that only it can create is written here
  as a named placeholder `⟨…⟩`, never as a digest.
- Implementation and synthetic execution are a later, separately scoped review (§824).

## 1. What the consumer must admit

One complete stage-T result set for the changed model (`axis_center=anchors`):
- four datasets: CIFAR-10, Flickr25K, NUS-WIDE, MS-COCO;
- three seeds each: 42, 43, 44.

It rests on this chain. Each link is read once, by its pinned digest.

| Link | Identity | Source of the pin |
|---|---|---|
| F (recipe/N freeze) | `/data/yschoi/gdna_anchor_lambda_v8/artifacts/anchor_confirmation/ancF_candidate_v1.json` `5165f5dc9fcfb8334270bc16aa9816d09db67b03a04abae7ff846d5235bdca1d`, acceptance §744 (`13ef776b…`) | recovery request `freeze` |
| R receipt (12 refits) | `/data/yschoi/gdna_anchor_refit_v9r6/artifacts/anchor_confirmation/ancR9_sweep_complete.json` `2897aa880dddd07cbb0aeb49ef090b7e5ccc06e6e9b540922a9ca77a30d81d4a`, approval §780 | request `refit_receipt`, `refit_approval` |
| Stopped T campaign `ancT9` | request `a74b68e1…`, §788 line; snapshot `ancT9_snapshot_da40a183d4ca9c85.json` (file `b6eb8a5e…`); reservation `83462894…`; campaign nonce `f6234d0e…` | request `stopped` |
| Parent settlement | `/home/yschoi/gdna_anchorRT_ops/device_budget_ledger.jsonl` `316915141fdfa215940014dbeb5efb63ba2b9e98bb84b49146cab69a96fd25bf`, final line `ab122e44…`, run `20261006T145139Z-b08e4ae2`, 79,482.14623009507 s | request `stopped.settlement`; supervisor `REC_PARENT_LEDGER_SHA256` |
| Recovery request | semantic `d34f505b90db1ef02dce74bd0326350053a6708315f505c0c881fe89a91ad666`; lineage `b7676019…`; r8 manifest `3537e297…`; r7 manifest `2f24fc80…` | §813, §823, §827 |
| Recovery approval | the `stage-T-recovery` line in audit section ⟨N⟩ | future |
| Recovery receipt | `/data/yschoi/gdna_anchor_refit_v9r8/artifacts/anchor_confirmation/ancT9r_recovery_complete.json` ⟨sha⟩ | future |
| Recovery settlement | `/home/yschoi/gdna_anchorRTrec_ops/device_budget_ledger.jsonl` ⟨sha⟩, run ⟨run_id⟩ | future |
| Recovery claim | `/home/yschoi/gdna_anchorRT_recovery_claims/dc4a8bc061c5d286d74e21b158091710c165737403e1c85e957f810740f92c6d.json` ⟨sha⟩ | future, bound by the receipt |

**Values come in at the consumer run, never in source:**
- The consumer pins the existing digests above as constants.
- It takes the four future identities (⟨N⟩, the receipt, settlement and claim digests) as explicit inputs.
- Those inputs must agree with each other and with the files.

The consumer never fills a future value from a guess, a default or a different run.

## 2. Requirement 1 (§824.1): bind the recovery's own settlement, not the receipt's settlement field

`final_recovery_closure` returns the **stopped** lineage's settlement (`anchor_refit_stage.py:1028–1083`).
`_run_recovery` writes that value into `receipt.final_closure.settlement` and publishes the receipt
(`:1318–1333`). The supervisor's own `final` event is written only later, after the child exits
(`anchor_confirm_supervisor.py:572`). So the receipt cannot describe the recovery's own accounting. The
consumer admits the recovery supervisor run separately.

### 2.1 Rules on the recovery ledger

- **Exactly one run.** The recovery ledger holds exactly one `start` and exactly one `final`, with the
  same `run_id` and `stage == "stage-T-recovery"`, and nothing after that `final`. Its whole bytes are
  hashed once, and that digest is the ⟨settlement sha⟩ input.
- **`start` must show:**
  - `refused` is null;
  - `manifest_sha256 == 3537e297…`;
  - `parent == {ledger: <parent path>, sha256: 316915141f…, charged_seconds: 79482.14623009507, last_run_id: 20261006T145139Z-b08e4ae2}`;
  - `rules.budget_seconds == 15000`, `rules.gpus == 1`, `rules.planned_cells == 1`;
  - `command_sha256` equals the SHA256 of the audited argv list with `--anchor-approval-section ⟨N⟩`, as
    the audit's launch section names it;
  - the argv contains `--namespace ancT9r` and `--gpus 0`.
- **`final` must show:**
  - `status == "exited"` and `returncode == 0`;
  - `continuity_lost == []`, `monitor_failures == []`, `orphaned_live_attempts == []` and
    `leases_held_after_exit == []`;
  - `cumulative_charged_seconds <= 15000`;
  - `parent_charged_seconds == 79482.14623009507`;
  - `cumulative_including_parent_seconds <= 94482.14623009507`.
- **Refused final statuses:** `stopped`, `unresolved`, `unclean`, `refused-before-start` and
  `failed-to-start` all refuse.

### 2.2 Joining the receipt to that run

- The receipt has `namespace == "ancT9r"` and `request_sha256 == d34f505b…`, and the start argv names the
  same namespace.
- The executed attempt (read through the receipt; §3) has `approval.section == ⟨N⟩`, and its
  `approval.line` is the exact `stage-T-recovery` line in section ⟨N⟩.
- The claim file is at the receipt's `recovery_claim.sha256`.

### 2.3 Order

- A receipt with no `final` in the recovery ledger is "receipt before settlement" and refuses.
- The **stopped** campaign's `final` (in the parent ledger) never counts as the recovery's `final`.

## 3. Requirement 2 (§824.2): read each cell through its own lineage

### 3.1 Receipt rows

`receipt.cells` must hold exactly 12 entries:
- **11 carried:** `origin "carried"`, `namespace "ancT9"`, `record_dir` = the r7 record root
  `/data/yschoi/gdna_anchor_refit_v9r6/artifacts/anchor_confirmation`.
- **1 executed:** `origin "executed"`, `namespace "ancT9r"`, `record_dir` = the r8 record root.
- The executed `cell_id` must be `nuswide|N=4|P=0.4,0.8|JD=0.05|stage=refit|seed=44|axis_center=anchors`.
- The carried rows must be exactly the 11 that the lineage file `b7676019…` lists, with their pinned
  record/attempt/entry digests.

### 3.2 For every row

The consumer reads the T record at `record_dir/record` against `record_sha256`, then requires:
- `schema == "anchor-terminal-test-record/1"`, `stage == "test"`, and `namespace` equal to the row's;
- `request_sha256`:
  - carried: `a74b68e1…`;
  - executed: `d34f505b…`;
- `campaign_nonce`:
  - carried: the stopped snapshot's `f6234d0e…`;
  - executed: `receipt.campaign_nonce`;
- `record.attempt.sha256 == attempt_sha256`; the attempt file is read at that digest. Its `cell` and
  `request_sha256` match the record.
- The entry file `…_entry_…` is read at `entry_sha256`. Its `attempt_sha256`, `cell_id`,
  `final_checkpoint_sha256` and `config_pt_sha256` match.
- The record's `cell.record` + `cell.record_sha256` (its stage-R record) is the one the R receipt names
  for that coordinate.
- The R-receipt and F digests come only from the pinned files of §1.

### 3.3 Coordinates

- They are taken from the record's `cell`, never from the receipt's optional summaries.
- They must be exactly {cifar10, flickr25k, nuswide, mscoco} × {42, 43, 44}: 12 unique and none missing.
- `cell.N` must equal F's N: 4/4/4/39 for CIFAR-10/Flickr25K/NUS-WIDE/MS-COCO.
- The receipt's convenience `map_at_R` / `bio_map_at_R` (executed row only) are never read for any table.

### 3.4 Field mapping

The records are preserved as they are. They are never flattened into a legacy `p3rfB` record.

| Value | Legacy P3 record | Anchor T record (measured on all 11 carried records) |
|---|---|---|
| dataset | top-level `dataset` | `cell.dataset` (`cifar10`, `flickr25k`, `nuswide`, `mscoco`); `completion.analysis_protocol.dataset` (`CIFAR10`, `Flickr25k`, `NUSWIDE`, `MSCOCO`) must agree by a fixed table |
| seed | top-level `seed` | `cell.seed` |
| run directory | top-level `run_dir` | `cell.run_dir` |
| final checkpoint digest | `completion.final_checkpoint_sha256` | `cell.final_checkpoint_sha256` (absent from `completion` in all 11) |
| terminal epoch | `completion` | `cell.terminal_epoch` |
| raw / BIO evaluation digests | `completion.evaluation_sha256` / `bio_evaluation_sha256` | same keys under `completion` |
| pairwise NMI digest | `completion.pairwise_nmi_sha256` | same key under `completion` |
| R cutoff | — | `completion.map_R_cutoff` (1000 for CIFAR-10, 5000 for the others, as measured) |

## 4. Requirement 3 (§824.3): the reader's true scope

Three tiers are kept separate. The consumer declares which tier each statement rests on.

### Tier A — JSON bytes this consumer reads and verifies (the only reads)

**Lineage and authority files:**
- F;
- the R receipt;
- the stopped snapshot, reservation and parent ledger;
- the lineage file;
- the recovery receipt, recovery ledger and claim file;
- the audit ledger section ⟨N⟩;
- 12 T records, 12 attempts and 12 entries;
- the 12 stage-R records named by the cells.

**Per cell, only the table's inputs:**
- `evaluation_siglip2_base.json` and `evaluation_siglip2_base_bioproj.json` (TODO2);
- `pairwise_nmi.json` (TODO8).

Each one is checked against the digest its T record binds, with one read per file.

All reads go through one function that records `{path: sha256}`. The bundle receipt lists exactly that
set, and a test asserts the set equals the declared list.

### Tier B — inherited reviewed authority (not re-verified here)

The checkpoints, NPZ extractions, configs and other outputs that the T records bind by digest were
checked by:
- the reviewed T entry when each cell ran;
- for the carried 11, the recovery's final closure as well (`verify_carried_payloads`, `:1018`).

The consumer cites those digests as inherited authority. It does **not** open those files and does not
claim to have verified them.

### Tier C — a binary closure scan

A scan that hashes checkpoints and NPZ files is **not** in this proposal. It needs its own approved
input, read and resource scope.

### What stays unchanged

- The legacy `build_maintable_combined.ours_closure` (`:600`) and its `_phase3_record_closure` (`:274`)
  hash checkpoints and extraction NPZ files. The new consumer does **not** call them.
- The legacy consumers keep their old pins (`b4f3b0df…`) and stay working, byte for byte unchanged:
  `make_bioproj_table.py`, `make_nmi_table.py`, `heldout_codon_decoding.py`,
  `collect_final_results_for_paper.py`, `f10_seed_aggregate.py` and `build_maintable_combined.py`.
- No legacy check is disabled.
- No helper runs on live artifacts during preparation.

## 5. Requirement 4 (§824.4): statistics and outputs

### 5.1 TODO2 (bio-projection, §4.6)

**Values, per cell, taken from the two pinned evaluation JSONs only** (the same fields the legacy
generator uses):
- pre and post mAP@R;
- paired Δ;
- mean DB edits;
- DB valid pre and post;
- DB DNA-unique pre and post.

**Checks kept:**
- raw `bio_project == false` and BIO `bio_project == true`;
- the same `mAP_R_cutoff` in both, equal to `completion.map_R_cutoff`;
- `bio_stats.mAP_at_R_pre_projection == raw mAP_at_R` within 1e-12;
- zero projection failures;
- post compliance 1.0;
- `db_num_total` is the full DB, with one value per dataset across its three seeds.

### 5.2 TODO8 (NMI, §4.10.1)

**Values, per cell, taken from the pinned `pairwise_nmi.json` only:**
- a 5×5 matrix with diagonal 1, symmetric within 1e-12, every entry finite in [0, 1];
- the ten unordered off-diagonal pairs;
- the mean, min and max agree with the file's own fields;
- `nmi_average_method == "arithmetic"`;
- the DB row count is recorded.

### 5.3 Reduction

- Per dataset: the explicit seed vector {42: x, 43: y, 44: z}, the mean, and the sample SD (n − 1).
- Any non-finite or missing value refuses; nothing is imputed.
- No extraction, retrieval, projection or NMI producer is re-run.

### 5.4 Outputs

- **Destination:** a new exclusive directory per bundle under `/home/yschoi/gdna_anchor_downstream/<todo>/`.
  - It is created with `mkdir` and refuses an existing path.
  - It is outside `docs/paper_draft`, every record root and every result root.
  - Copying into the paper is a separate gate and is not part of this work.
- **Files:** the TeX and JSON tables.
- **Receipt,** written last, holding:
  - the consumed-bytes map;
  - the consumer source digests;
  - the chain identities of §1, with the four future values filled from the inputs;
  - the output digests.
- **No-drift publication check:** before the receipt is written, every consumed path is re-read and must
  have the same digest. On drift, the bundle is abandoned (marked, not silently replaced).
- **Labels:** every output says `scope: "formatting of stored values"` and
  `independent_numerical_verification: false`.
  - It does not turn any `paper_result_eligible: false` marker into approval.
  - It does not carry old-model completion over to this model.

## 6. Proposed consumer source generation

**When:** only after (a) the recovery has settled, whatever its outcome, and (b) the audit has reviewed
this proposal. Nothing is written into the r8 tree while it must stay at S.

**Where:** a new branch from S, `arch-exp-2026-09-anchor-consumer`, in its own worktree. It adds files
only:

| File | Content |
|---|---|
| `scripts/anchor_t_consumer.py` | admission of §1–§4: the chain, both settlements, the 12-row lineage, coordinates, the read recorder |
| `scripts/make_anchor_bioproj_table.py` | TODO2 on top of the admission |
| `scripts/make_anchor_nmi_table.py` | TODO8 on top of the admission |
| `tests/test_anchor_t_consumer.py` | the synthetic tests of §7 |
| `artifacts/anchor_consumer/consumer_manifest_v1.json` | digests of the four new files |

**Source authority:**
- The manifest records the r8 manifest `3537e297…` and the r7 manifest `2f24fc80…` as **historical
  producer authority**, separately from the consumer's own source authority.
- The consumer does not import any r8 producer module. It reads only the JSON listed in §4 Tier A.

**If the recovery does not complete:**
- There is no recovery receipt, so the consumer refuses (§2).
- An 11-cell report is not part of this contract. It would need an explicit user request, labelled as
  partial, with the missing NUS-WIDE seed 44 and the resource stop stated, kept apart from protocol
  acceptance (§825).

## 7. Synthetic test plan (tests in `tests/`, driving `main()`, inside the OS sandbox)

**Fixture world:** built in `tmp_path` only, never pointing at a real artifact. It contains:
- F, an R receipt and 12 stage-R records;
- a stopped snapshot and reservation, and a parent ledger ending settled;
- 11 carried and 1 executed T records with attempts and entries;
- per-cell evaluation, BIO and NMI JSONs;
- a recovery receipt, a recovery ledger (one start and one final) and a claim file;
- an audit-ledger fragment with the line.

The real constants are replaced only through the fixture's injection points.

**Positive control:** the genuine fixture produces both bundles, and the consumed-bytes set equals the
declared Tier-A set.

**Refusal cases (each one test, each checking its own diagnostic):**

| Area | Cases |
|---|---|
| Recovery settlement | ledger absent; `start` only (receipt before settlement); `final` status `stopped`, `unresolved`, `unclean`, `refused-before-start` or `failed-to-start`; `returncode` 1; leases held after exit; orphaned attempts; continuity lost; cumulative over 15,000 or over 94,482.14623009507; two runs in the ledger; a line after `final` |
| Substitution | the parent ledger's `final` offered as the recovery's; a recovery ledger whose `start.parent` digest is not `316915141f…`; the `start` argv with another namespace, GPU count or approval section |
| Approval | the attempt's approval line or section is not ⟨N⟩'s exact line |
| Origin | the executed row labelled carried; a carried row with namespace `ancT9r`; a carried row whose record root is the r8 root; the executed row not NUS-WIDE s44 |
| Identity | wrong request digest or campaign nonce; record ↔ attempt ↔ entry digest mismatch; a stage-R record not in the R receipt; N ≠ F's N |
| Coverage | 11 or 13 rows; a duplicate coordinate; a missing coordinate; a seed outside {42, 43, 44} |
| Old/new mixing | a legacy `p3rfB` record (top-level dataset/seed) in place of a T record; a T record from another campaign; the receipt's convenience `map_at_R` disagreeing with the evaluation JSON (the table uses the JSON, and a test pins that) |
| Metric JSON | the raw, BIO or NMI file missing; a digest mismatch; raw/BIO pairing broken; R cutoff mismatch; a projection failure; post compliance < 1; NMI not 5×5, asymmetric or out of [0, 1]; a non-finite value |
| Publication | an existing destination; a destination inside `docs/paper_draft` or a record root; a consumed file changed between read and final check (drift), so no receipt is written |

**Mutation battery:** declared at implementation time from observed failure lines. It runs in a sandbox
copy, with one whole condition disabled per mutant.

## 8. Not covered here

- TODO3 held-out decoding: it needs new anchor-model analyses for all 12 cells and 4 aggregates, not the
  old F10 digests.
- F09/F12, interventions, D4/D5/D6, randomization repair, real human responses and paper regeneration.
- The old collector `collect_final_results_for_paper.py`, which stays untouched and is not re-run.
- Any baseline retraining or two-seed substitution.

All of these keep their full requirements from the migration matrix
(`docs/ANCHOR_MODEL_TODO_MIGRATION_v2.md` `32e7b88f…`, rows 2, 3 and 8).

## 9. Questions for the audit

1. Is `/home/yschoi/gdna_anchor_downstream/` an acceptable bundle root? It sits outside every record,
   result and paper root.
2. Should the Tier-A read set also include the stage-R records' own run-directory sidecars? The legacy
   L2 read them; this proposal reads only the stage-R record JSON named by each cell.
3. Should ⟨N⟩ and the three future digests be passed as command-line inputs, as proposed, or pinned in a
   follow-up manifest after the recovery settles?
