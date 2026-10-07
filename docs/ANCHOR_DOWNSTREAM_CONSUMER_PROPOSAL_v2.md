# Anchor model — downstream consumer/aggregate contract for stage-T results (proposal v2; audits 798, 824, 831)

**Status: documentation-only proposal.**
- No source, test, record, manuscript or r8 file changed.
- No reader, formatter, producer or test ran.
- The r8 tree stays at the audited submission HEAD S `04e7fe871cf9fc19a3e9ed6b713038686ff31830`.
- The recovered NUS-WIDE seed-44 cell has not run. Every value that only it can create is written here
  as a named placeholder `⟨…⟩`, never as a digest.
- Implementation and synthetic execution are a later, separately scoped review (§824).

**Revision v2 (audit §831.2).** v1 (`75b3fca0…`) was reviewed in §831.2 and is kept unchanged as history.
v2 adds:
1. metadata joins from every raw/BIO/NMI `input_binding` to the cell and completion, with stored-value
   cross-checks (§3.5, §5);
2. the DB extraction manifest as the full-DB denominator (§3.6, §4);
3. a separately audited post-settlement **consumer input manifest** in place of CLI digests (§1, §6);
4. the exact supervisor start rules and settlement arithmetic (§2.1);
5. single-capture reads, accurate file-type declarations, and incomplete-bundle preservation (§4, §5.4).

It also records the audit's answers to the three v1 questions (§9).

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
| Recovery approval (historical launch authority) | the `stage-T-recovery` line in audit section 830. A first invocation was refused at preflight before any start (§831); any re-invocation needs the audit's word | §830 |
| Recovery receipt | `/data/yschoi/gdna_anchor_refit_v9r8/artifacts/anchor_confirmation/ancT9r_recovery_complete.json` ⟨sha⟩ | future |
| Recovery settlement | `/home/yschoi/gdna_anchorRTrec_ops/device_budget_ledger.jsonl` ⟨sha⟩, run ⟨run_id⟩ | future |
| Recovery claim | `/home/yschoi/gdna_anchorRT_recovery_claims/dc4a8bc061c5d286d74e21b158091710c165737403e1c85e957f810740f92c6d.json` ⟨sha⟩ | future, bound by the receipt |

**Where the future identities come from (§831.2 item 3).** The consumer pins the existing digests above as
constants. The values that exist only after a **successful** recovery settlement are **not** consumer
inputs.

They are pinned in a separately audited **consumer input manifest**
(`anchor-consumer-inputs/1`), written after settlement and reviewed by the audit. It holds:
- the launch section (830) and its exact line;
- the recovery run ID;
- the complete recovery receipt;
- the parent and recovery ledgers (digests of their whole bytes);
- the claim file;
- the recovery snapshot and reservation;
- the origin-specific record/attempt/entry triples (11 carried r7, 1 executed r8).

How it is used:
- The later exact formatting request binds this manifest's digest and the consumer source generation.
- A CLI argument may only **locate** the manifest. It never supplies or redefines an identity, and mutual
  agreement among caller-supplied digests is not an approval anchor.

Nothing in this document fills a future value. An incomplete or unsettled recovery has no input manifest,
and the consumer refuses.

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
  - `prior_charged_seconds == 0` (the recovery ledger is new);
  - exact `rules`, from the supervisor constants (`anchor_confirm_supervisor.py:99–134`):

    | Rule | Value |
    |---|---|
    | `budget_seconds` | 15000 |
    | `gpus` | 1 |
    | `planned_cells` | 1 |
    | `children_per_cell` | 5 |
    | `planned_attempts` | 5 |
    | `poll_seconds` | 1.0 |
    | `watchdog_seconds` | 10.0 |
    | `stop_bound_seconds` | 120.0 |
    | `headroom_seconds` | 130.0 |
    | `wall_limit_seconds` | 28800.0 |
    | `free_floor_bytes` | 10 GiB |
    | `cell_output_bytes` | 0.75 GiB |

  - `command` equals, element by element, the audited argv list after `--`. Its `command_sha256` equals
    SHA256 of `json.dumps(list(command)).encode()`, the supervisor's own serialisation (`:360`), so the
    list and its digest are both compared.
  - The argv contains `--namespace ancT9r`, `--gpus 0` and `--anchor-approval-section 830`.
- **`final` must show:**
  - `status == "exited"` and `returncode == 0`;
  - `continuity_lost == []`, `monitor_failures == []`, `orphaned_live_attempts == []` and
    `leases_held_after_exit == []`;
  - `cumulative_charged_seconds <= 15000`;
  - `parent_charged_seconds == 79482.14623009507`;
  - `cumulative_including_parent_seconds <= 94482.14623009507`.
- **Settlement arithmetic must hold exactly** (floating sums within 1e-6). Every number must be finite and
  non-negative:
  - `device_seconds == Σ attempts[i].seconds`;
  - `charged_seconds == device_seconds + unobserved_allowance_seconds`;
  - `cumulative_charged_seconds == 0 + charged_seconds`;
  - `cumulative_including_parent_seconds == 79482.14623009507 + cumulative_charged_seconds`;
  - the number of attempts ≤ 5, and every attempt has an end.

  The observation allowance is never discarded, and the charge is never reduced to the longest producer.
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

### 3.5 Metric input bindings (§831.2 item 1; metadata comparison only)

Each of the three pinned metric files has an `input_binding` block (schema in
`dna_utils/extraction_validation.py:677–712`):
- the raw evaluation;
- the BIO evaluation;
- `pairwise_nmi.json`.

Each block must join to the T record:

| Binding field | Must equal |
|---|---|
| `schema_version` | 1 |
| `run_dir` (absolute) | `cell.run_dir` |
| `dataset` | the canonical name of `cell.dataset` (`CIFAR10`/`Flickr25k`/`NUSWIDE`/`MSCOCO`) |
| `random_seed` | `cell.seed` |
| `inference_epoch` | `cell.terminal_epoch` |
| `checkpoint_sha256` | `cell.final_checkpoint_sha256` (**not** the legacy `completion` location) |
| `config_sha256` | `cell.config_pt_sha256`, the same convention the T entry enforces (`anchor_refit_stage.py:563`) |
| `npz_sha256` (map) | `completion.npz_sha256` |
| `manifest_sha256` (map) | `completion.extraction_manifest_sha256` |
| `codebook_size` | F's codebook size for the dataset; it also equals `completion.analysis_protocol.codebook_size` |
| `backfilled_inputs` | `false` |

**The three blocks must also be equal to each other.** The reviewed seal already requires that between
the evaluation and NMI blocks (`seal_cell_analysis.py:259`).

**What the consumer does not do:**
- It compares these values as metadata. It does **not** call `metric_input_binding` or
  `check_metric_input_binding`, because they re-validate the extraction and open the NPZ files.
- It never interprets a value as an approval.

### 3.6 Full-DB denominator (§831.2 item 2)

For each cell, the consumer also reads `run_dir/extraction_manifest_db.json`, pinned by
`completion.extraction_manifest_sha256.db`. That is twelve bounded JSON metadata reads, not an
extraction or NPZ scan. It requires:
- `split == "db"`;
- `n_rows` a positive integer;
- `dataset`, `random_seed`, `checkpoint_sha256` and `inference_epoch` equal to the cell's;
- `backfilled == false`.

Then:
- BIO `bio_stats.db_num_total == n_rows`, and NMI `N == n_rows`;
- `n_rows` equals the dataset's reviewed full-DB row count. That count is pinned from the accepted
  protocol authority, named with its source in the implementation review, and never derived from the
  files being checked;
- NMI `K_per_cb` and the 5-slot shape (5×5 matrix, ten pairs) equal F's codebook size and slot count.

An equal count across seeds alone is not accepted as the full denominator.

## 4. Requirement 3 (§824.3): the reader's true scope

Three tiers are kept separate. The consumer declares which tier each statement rests on.

### Tier A — bytes this consumer reads and verifies (the only reads)

**File types, declared exactly:**
- JSON authority files;
- the consumer input manifest (JSON);
- the audit ledger's named section (Markdown);
- the parent and recovery ledgers (JSONL);
- the per-cell JSON listed below.

**One read per file (§831.2 item 5):** each file is read **once** into a byte capture. It is hashed
against its pin, then parsed and used only from that capture. A second read happens **only** in the final
no-drift check before publication.

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
- `pairwise_nmi.json` (TODO8);
- `extraction_manifest_db.json`, the denominator binding of §3.6.

No other run-directory sidecar is read (§831.2 answer 2).

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
- `db_num_total` equals the DB manifest's `n_rows` (§3.6);
- the input bindings of §3.5;
- stored-value cross-checks: `completion.map_at_R` == raw `mAP_at_R`; `completion.bio_map_at_R` == BIO
  `mAP_at_R`; `completion.map_R_cutoff` == both files' `mAP_R_cutoff`. A disagreement refuses.

### 5.2 TODO8 (NMI, §4.10.1)

**Values, per cell, taken from the pinned `pairwise_nmi.json` only:**
- a 5×5 matrix with diagonal 1, symmetric within 1e-12, every entry finite in [0, 1];
- the ten unordered off-diagonal pairs;
- the mean, min and max agree with the file's own fields;
- `nmi_average_method == "arithmetic"`;
- the pair mean equals both the file's `mean_off_diag_nmi` **and** `completion.mean_off_diag_nmi` (1e-12),
  as the existing reader requires (`make_nmi_table.py:101`);
- the input binding of §3.5;
- `N` equals the DB manifest `n_rows` (§3.6).

### 5.3 Reduction

- Per dataset: the explicit seed vector {42: x, 43: y, 44: z}, the mean, and the sample SD (n − 1).
- Any non-finite or missing value refuses; nothing is imputed.
- No extraction, retrieval, projection or NMI producer is re-run.

### 5.4 Outputs

- **Destination:** a new exclusive directory per bundle under `/home/yschoi/gdna_anchor_downstream/<todo>/`
  (accepted in §831.2).
  - The resolved destination must lie inside that root.
  - It must not alias any producer, record, source or paper root, including through symlinks.
  - This does not authorise creating a live bundle now.
  - It is created with `mkdir` and refuses an existing path.
  - It is outside `docs/paper_draft`, every record root and every result root.
  - Copying into the paper is a separate gate and is not part of this work.
- **Files:** the TeX and JSON tables.
- **Receipt,** written last, holding:
  - the consumed-bytes map;
  - the consumer source digests;
  - the chain identities of §1, with the four future values filled from the inputs;
  - the output digests.
- **No-drift publication check:** before the receipt is written, every consumed path is re-read once and
  must have the same digest.
- **Failure handling:** on drift or any publication failure, the bundle is **kept as incomplete**. It gets
  an `INCOMPLETE` marker naming the failure and **no** success receipt, and is never silently replaced or
  deleted.
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

**Consumer input manifest:** after a successful recovery settlement, `artifacts/anchor_consumer/
consumer_inputs_v1.json` (§1) is proposed for audit review **before** any formatting request. The
formatting request then binds its digest and `consumer_manifest_v1.json`.

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
| Input binding (§3.5) | wrong checkpoint (including one taken from the legacy `completion` location); `backfilled_inputs` true; wrong inference epoch; a wrong `npz_sha256` or `manifest_sha256` map entry; wrong config, dataset, seed or run directory; raw/BIO/NMI bindings unequal; the NMI pair mean disagreeing with `completion.mean_off_diag_nmi`; stored `map_at_R` / `bio_map_at_R` disagreeing with the files |
| DB denominator (§3.6) | DB manifest missing or off its pin; `n_rows` ≤ 0 or not an integer; `n_rows` ≠ the reviewed full-DB count; BIO `db_num_total` or NMI `N` ≠ `n_rows`; the DB manifest naming another checkpoint, seed or epoch |
| Accounting (§2.1) | any start rule off its value; a non-finite or negative number; `charged_seconds` ≠ device + allowance; a cumulative not matching parent + charge; the argv list differing from the audited one while its digest is recomputed to match; the digest serialised differently from `json.dumps(list(command))` |
| Input manifest (§1) | absent; off its reviewed digest; CLI digests that agree with each other but not with the manifest |
| Publication | an existing destination; a destination inside `docs/paper_draft` or a record root, or reached through a symlink alias; a consumed file changed between read and final check (drift), so the bundle is kept as incomplete with no success receipt; a file parsed from a second read instead of the verified capture |

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

## 9. The audit's answers to v1's questions (§831.2)

1. **Bundle root:** `/home/yschoi/gdna_anchor_downstream/` is accepted, with resolved containment and
   rejection of any alias, including symlinks (§5.4).
2. **Stage-R run sidecars:** no general scan. Their binary/runtime closure stays inherited authority.
   Only the named DB extraction manifest is added (§3.6). Any further metadata must be named and justified
   first.
3. **Future identities:** they come from the separately reviewed post-settlement input manifest (§1),
   not from CLI digest overrides. §830 is recorded only as the historical launch authority.
