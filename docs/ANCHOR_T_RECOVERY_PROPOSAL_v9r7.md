# Anchor model — recovery proposal for the one interrupted full-T cell (NUS-WIDE seed 44), audit §795

**NON-EXECUTABLE.**
- This is a design for review, not a request. No source change, render, model load, evaluation or
  launch has been made for it.
- Nothing runs before the audit reviews the design and then approves an exact request.
- §795 grants no retry exception and no added budget, and this document does not assume either.

## 1. Receipt of §795–§796 and the starting state

§795 was read in full (ack 795). It accepts the full-T budget stop and settlement; it does not accept
full-T completion. It asks for this proposal, metadata-only, for NUS-WIDE seed 44 only.

§796 was read in full (ack 796).
- The `td1_*` runs belong to the separate text-path session, confirmed by the user. They are kept
  apart from anchor evidence and accounting.
- Any future approved run first agrees GPU indices and UUIDs, ownership, duration and release with
  that peer, then rechecks live use and leases.
- Peer processes and locks are never disturbed.

**Archived evidence.**
- Commit `76e05fe`, folder `artifacts/anchor_confirmation/refit_v9/full_t_ancT9/`: command, log and
  status, settlement, cell evidence, stopped-cell inventory and record digests.
- The peer's settlement receipt (`peer_settlement_notice.json`) goes in with this document.

| Item | State |
|---|---|
| Completed T cells | 11: CIFAR-10, Flickr25K and MS-COCO seeds 42–44; NUS-WIDE seeds 42–43. Each has a T record, attempt, entry and the 12 required outputs |
| Interrupted cell | NUS-WIDE seed 44 (`ancR9_nuswide_A_v4_refit_N4_s44_AXanchors_P0408_JD005`) |
| ancT9 receipt | absent; it is not to be created |
| R/T ledger | settled at 79,482.15 of 80,000 device-s, 517.85 unspent |

## 2. Producer boundaries of the interrupted cell

Facts come from the command log (`grep -n` lines) and the run directory.

| Producer | State |
|---|---|
| 1. T entry: query/DB extraction and raw evaluation | **Interrupted** (details below) |
| 2. Train extraction | not started |
| 3. BIO evaluation | not started |
| 4. NMI | not started |
| 5. Analysis seal | not started |

**What the T entry did before the stop:**
- reserved the attempt `ancT9_attempt_…s44…` and claimed the entry `ancT9_entry_…s44…`
  (PID 2006624, 20:41:40Z);
- was admitted (line 5662) and loaded the pinned checkpoint (line 5669);
- began DB extraction (line 5678) and reached batch 140 of 757, about 35,800 of 193,734 DB rows at
  256 rows per batch;
- then received SIGTERM at the budget stop.

**What it did not do or leave behind:**
- No query encoding appears for this cell, and no raw evaluation ran.
- The codes it computed stayed in process memory and were never written.
- None of the 13 contract outputs exists. No metric was produced or seen.
- The checkpoint, runtime JSON and config are unchanged (re-hashed after the stop).

So the official-test path was opened (DB rows passed through the pinned model), but no test
information was published.

## 3. Why the current source cannot recover it

The current r7 source refuses every way to rerun this one cell. That is by design.

| Guard | Location | Effect |
|---|---|---|
| Fixed cell set | `admit_refit_receipt` | stage T in `run` mode admits exactly the 12 F cells |
| Test withheld | `admit_refit_receipt`, `assert_official_test_withheld` | each R run directory must hold no official-test output; the 11 completed cells now hold theirs |
| No cell subset | `_test_main` | `--only` is refused for stage T |
| Attempt never removed | `reserve_attempt` (`anchor_refit_stage.py:567-584`) | the docstring says the attempt "is never removed … a second attempt — concurrent, duplicate or retry — refuses … Recovering an attempted cell needs a new authorization" |
| Namespace-scoped attempts | `attempt_path` | the attempt check is per namespace; a new namespace would not see the spent ancT9 attempt unless new code binds it explicitly |
| Fixed budget | `RT_BUDGET_DEVICE_SECONDS = 80000.0` (`anchor_confirm_supervisor.py:92`) | a source constant; any increment is a source change |

A recovery therefore needs **new source authority**: a new generation manifest (r8). Without it the
only path is option B below.

## 4. Options

### Option A — recover NUS-WIDE seed 44 under an explicit, audited exception (recommended)

This keeps three seeds per dataset as the protocol planned.

**Test-once.** Every fact that bears on test-once is in §2. The selection (F) is frozen and was accepted
in §744. The cell's checkpoint and config are pinned. The interrupted attempt yielded no number, so
no information exists that could steer a choice. The re-run would evaluate the same pinned checkpoint
on the same split.

That makes the exception narrow: **one** further supervised T chain for exactly this cell.
- The spent ancT9 attempt and entry are carried into its lineage as "interrupted, no output". They
  are not deleted or reused.
- It is still an exception to the never-retry rule, and only the audit can grant it.

**Source changes**, to be written only after the audit accepts this design:

1. **`anchor_refit_stage.py`: a recovery admission**, e.g., `--anchor-confirm test-recovery`, which
   takes the ancT9 snapshot and the ancR9 receipt. It admits **only** when all of these hold:
   - **R campaign:** the ancR9 receipt verifies exactly as now (F, approvals, records, trainer
     evidence, runtime sidecars).
   - **The 11 carried cells:** each run directory's 12 outputs equal the bytes bound by its ancT9 T
     record. Each record, attempt and entry is at the digest archived in `records_manifest.sha256`.
     None is re-evaluated.
   - **The recovery cell:**
     - exactly one prior attempt and one prior entry exist, both at the archived digests;
     - there is no T record and none of the 13 outputs;
     - its R pins are unchanged.
   - **Settlement:** the ancT9 snapshot (`da40a183…`) and reservation are at their digests. The ops
     ledger holds the ancT9 `final` event with `status=stopped` for run `20261006T145139Z-b08e4ae2`.
   - **Request:** it names one cell, `gpu_count` 1, the carried-lineage digests above, and
     `recovery_of` (the spent attempt and entry digests).
2. **Attempts in the new namespace** (e.g., `ancT9r`):
   - `reserve_attempt` runs as now.
   - A cross-namespace check refuses unless the ancT9 attempt is the cell's only prior attempt.
   - A second recovery attempt refuses like any other.
3. **Receipt lineage:** a new receipt `ancT9r_test_complete.json` with its own schema.
   - It lists 12 cells: 11 carried by ancT9 record name and digest, and 1 new record.
   - It binds the ancT9 snapshot, the settlement event and the spent attempt.
   - No `ancT9_test_complete.json` is ever written.
   - Consumers that read a T receipt must accept this kind explicitly.
4. **Budget:** a recovery stage with its own budget, chosen by the audit. Two shapes:
   - **(i)** A separate ledger group, e.g., `stage-T-recovery` with ledger root
     `/home/yschoi/gdna_anchorRTrec_ops` and budget 15,000 device-s. The settled R/T ledger stays
     closed. This mirrors how the L and R/T ledgers are already separate.
   - **(ii)** Raise `RT_BUDGET_DEVICE_SECONDS` to 95,000. The run would continue the R/T ledger from
     79,482.15.

   Both give the same cumulative bound (table below). I recommend (i), so the settled ledger is never
   appended under a changed constant.
5. **Generation r8:** a new manifest pinning the changed sources. F, the input seals, the R records
   and checkpoints, and the 11 T records are unchanged and bound by digest.
6. **Verification before any request:**
   - unit tests through `main()`, including refusals for:
     - a carried output whose bytes differ;
     - a second prior attempt;
     - any output already present in the recovery cell;
     - a missing or non-`stopped` settlement;
     - a recovery receipt that tries to carry a cell twice;
     - a request that names more than one cell;
   - the full suite;
   - a mutation battery on a sandboxed copy;
   - a guarded metadata-only render of the recovery request, whose digest is the one the approval
     must name.

**Resource plan** (estimates from this run's own cold-cache timings; one GPU, NUS-WIDE alone):

| Item | Plan |
|---|---|
| Observed NUS-WIDE cell time | 14,627 s (seed 42, four parallel streams) and 6,353 s (seed 43, beside the Flickr25K and MS-COCO seed-44 streams for all but its last ~15 min). The feature cache (text tokens alone `(195834, 6, 32, 512)`) is read from the HDD at about 44 MB/s and exceeds page cache |
| Expected | about 5,000–7,000 device-s for the five producers, if no other heavy `/data` reader runs |
| Budget increment | 15,000 device-s: about 2× the seed-43 time, plus the 130 s stop headroom and 5 × max-window allowance |
| Cumulative | ≤ 79,482.15 + 15,000 = 94,482.15 device-s over R, T and the recovery |
| Wall | the stage-T-run limit, 8 h; the budget stop binds first |
| Launch gates | **Before the request:** agree with the text-path peer (§796) the physical GPU index and UUID, ownership, duration and release condition, with a written receipt. **In the launch command:** that GPU idle by UUID; leases checked with `lslocks` only; `/data` read load measured with `/proc/diskstats` and per-process I/O. If another session is reading `/data` heavily, report the conflict for coordination and wait. Never stop or slow a peer process |
| Failure rule | one recovery attempt. Any refusal, drift or resource stop: preserve, settle, return. No further exception is implied |

**Cost before the run:**
- the source change, tests and battery: about 1–2 h of CPU and my time;
- audit reviews of the design, the source and the request;
- one guarded render.

### Option B — no recovery

- Report NUS-WIDE with seeds 42 and 43 only. Seed 44 is recorded as "attempted, interrupted by the
  resource stop, not evaluated".
- No source change, budget or exception.
- Cost: the planned three-seed design is incomplete for one dataset. Every NUS-WIDE mean/SD has n = 2,
  and the paper must say so.

## 5. What stays fixed in either option

- F, N, the λ values, seeds, pins, the input seals and the R campaign.
- The 11 completed T cells: never re-evaluated, and their outputs never rewritten.
- The spent ancT9 attempt and entry for seed 44: never deleted or reused.
- No `ancT9` receipt is created. No downstream analysis of the T outputs runs before its own approval.

## 6. Decision needed

**From the audit:**
- whether option A's exception is acceptable in principle;
- if it is, which budget shape (i or ii) and what increment.

After that I write and verify the r8 source (§4 item 6) and submit an exact request with its rendered
digest.

**From the user:** whether the paper should carry NUS-WIDE with two seeds (option B) if the audit
declines option A.

## 7. Errata after §797 (2026-10-07)

The text above stays as reviewed (SHA256 `a01450e2…`). These corrections supersede it where they
differ.

1. **The budget shapes are not equivalent.** "Both give the same cumulative bound" in §4 item 4 was
   wrong.

   | Shape | Cumulative maximum | Additional seconds allowed |
   |---|---|---|
   | (i) Separate fixed recovery ledger, 15,000 device-s | 79,482.14623009507 + 15,000 = **94,482.14623009507** (not 95,000) | 15,000 |
   | (ii) Raise the R/T constant to 95,000 | 95,000 | 15,517.85376990493 |

   r8 implements shape (i), pending the user's decision. The new ledger binds the settled R/T ledger's
   digest and its 79,482.14623009507 s charge, and never rewrites it.
2. **The exception wording.** The recovery is the recovery of a predetermined missing cell after a
   resource failure. It is **not** proof that no test information exists:
   - the interrupted entry opened the official-test path (DB batch 140/757);
   - the eleven other results are already visible.

   F, N, the λ values, the checkpoint, inputs, seed, inference epoch and ranking policy never change,
   and none of the eleven evaluations is repeated.
3. **The I/O explanation and the time forecast are estimates.** The 5,000–7,000 s forecast is not a
   completion guarantee. The four process and disk samples (16:32–16:36 UTC, in
   `anchor_rt_session_state/recovery_prep/full_t_io_observations.json`) are bounded historical
   observations:
   - the process counters span all disks;
   - the samples are not simultaneous;
   - the first process sample also shows 61.9 MB/s of git reads, which came from `/home` (`sda`).

   They do not establish exclusive causation, whole-run rates or the recovery's duration.
4. **The design corrections of §797.2 items 1–6** are implemented in the separate r8 preparation tree
   (`/data/yschoi/gdna_anchor_refit_v9r8`, `docs/ANCHOR_T_RECOVERY_CONTRACT_v1.md`):
   - historical r7 authority versus executing r8 sources, with an exact allowlist;
   - exact membership of the carried eleven and the recovered one;
   - one shared, race-safe recovery claim that the T entry verifies itself;
   - the separate ledger;
   - the wording above.

   §800's final carry and lineage closure before the combined receipt is implemented there as well.
