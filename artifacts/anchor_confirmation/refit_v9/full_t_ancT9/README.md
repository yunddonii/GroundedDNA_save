# Full stage T (audit §788, r7): terminal evidence after a budget stop

This folder holds the evidence for the separate recovery review required by §794–§795. Nothing has run
since the stop.

§795 was read in full and acknowledged (ack 795). It accepts the stop and settlement, not full-T
completion. The NUS-WIDE seed 44 recovery proposal is a separate, non-executable submission.

## Outcome

| Item | Value |
|---|---|
| Run | `20261006T145139Z-b08e4ae2`, tmux `ancT9_v9r7`, argv equal to the audit's `full_t_coordination/approved_argv.json` (42 tokens) |
| Stop | supervisor `STOP: budget: projected 80001 s of 80000 s; SIGTERM sent to 1935565` at 2026-10-06T20:50:07Z (ledger `stop` event) |
| Exit | tmux rc=3, 21,514 s; launcher rc=143 (SIGTERM) |
| Cells | 11 of 12 complete. NUS-WIDE seed 44 was stopped before it wrote any T output |
| Receipt | `ancT9_test_complete.json` absent: the campaign is not complete |

## Settlement

From the `final` ledger event; checks are in `settlement.json`.

| Item | Value |
|---|---|
| Device seconds | 72,109.77 |
| Allowance | 91.87 = 60 planned attempts × max window 1.5311 s |
| Charged | 72,201.64 |
| Cumulative R/T | 79,482.15 of 80,000 |
| Attempts | 56 starts and 56 ends (11 cells × 5 producers, plus the NUS s44 entry) |

- No leases were held after exit.
- There were no orphaned attempts, no continuity loss and no monitor failure.
- No ledger line follows the `final` event.
- After the stop, GPUs 0–5 showed no compute process, and `lslocks` showed no lease lock.

The remaining envelope is 517.85 s.
- **Start rule** (`anchor_confirm_supervisor.py:306-316`): one cell on one GPU would still pass it,
  because 79,482.15 + 5 + 130 < 80,000.
- **Budget stop:** that run would be stopped after roughly 380 device-seconds. A NUS-WIDE cell took
  6,353–14,627 s here, so the current envelope cannot complete one.

## The stopped cell (`nus_s44_stopped_cell.json`)

NUS-WIDE seed 44 is not an untouched cell (§794).
- **Attempt:** the attempt record exists, so the reservation is spent.
- **Entry:** the entry record binds the attempt bytes; entry PID 2006624, created 20:41:40Z.
- **Outputs:** none of the 13 contract names exists in its run directory. There is no T record.
- **R pins:** the checkpoint, runtime JSON and config are unchanged.
- **Producer boundaries** (from the command log, `grep -n` line numbers):
  - **Producer 1, the T entry, was interrupted.** It was admitted at line 5662 and loaded the checkpoint
    at line 5669. It started DB extraction at line 5678 and reached batch 140/757 before the stop.
  - The entry encodes the DB first; no query encoding appears for this cell, and no raw evaluation ran.
  - **Producers 2–5 never started:** train extraction, BIO evaluation, NMI and the analysis seal.
- **Test-path access:** the official-test path was opened, since the checkpoint ran forward on DB rows,
  even though nothing was published.

Nothing was deleted or rewritten.

## Cells (`t_cells_evidence.json`, produced by `collect_full_t_evidence.py`, read-only)

For each of the 11 complete cells:
- **Records:** the T record, attempt and entry bind each other, the approved request `a74b68e1…` and the
  campaign nonce.
- **Outputs:** all 12 required outputs exist, their bytes match the T record's hashes, and
  `evaluation_siglip2_bit2.json` is absent.
- **R pins:** recomputed now, they equal both the request and the full-R summary.
- **Metrics:** every metric is read from the evaluation, NMI and cell_result files themselves, and equals
  the T record and the analysis seal.

The collector reports 0 problems. Before use it was checked on scratch copies; it caught each of six
injected faults (extra bit2, byte edit, corrupt JSON, missing output, edited NMI, relocated run dir).

## Metrics (descriptive; `t_metrics_descriptive.md`)

Values are mean ± SD over the complete seeds. No statistical test was run, and nothing here reopens F.

| Dataset | Seeds | raw mAP@R | BIO mAP@R | raw P@1 | NMI (mean off-diag) | BIO unique (DB) |
|---|---|---|---|---|---|---|
| CIFAR-10 (R=1000) | 42, 43, 44 | 0.8706 ± 0.0061 | 0.8669 ± 0.0096 | 0.8823 ± 0.0170 | 0.6050 ± 0.0200 | 0.0679 ± 0.0111 |
| Flickr25K (R=5000) | 42, 43, 44 | 0.8486 ± 0.0039 | 0.8434 ± 0.0066 | 0.9255 ± 0.0054 | 0.4859 ± 0.0250 | 0.3284 ± 0.0252 |
| MS-COCO (R=5000) | 42, 43, 44 | 0.8262 ± 0.0047 | 0.8205 ± 0.0066 | 0.9174 ± 0.0036 | 0.6195 ± 0.0013 | 0.1839 ± 0.0121 |
| NUS-WIDE (R=5000) | 42, 43 | 0.8281 ± 0.0053 | 0.8220 ± 0.0040 | 0.8581 ± 0.0067 | 0.5207 ± 0.0048 | 0.1974 ± 0.0138 |

## Why the run took so long (disclosure)

The proposal's estimate of 8k–17k device-s was wrong; the run used 72.1k. I scaled the T smoke linearly,
but the smoke ran alone, on one dataset, with its feature cache already in page cache.

**Cause.** The feature caches are on `/data` (`sdb1`), a rotational HDD.
- During the run the disk was 100 % busy at about 44 MB/s of random reads. Per-process I/O accounting
  attributed all of that load to the run's own extraction processes.
- Four dataset streams read caches that together exceed the about 220 GB page cache.
- `/home` (`sda4`, SSD), where the run directories and other sessions' git work live, was not the
  bottleneck.

**Per-cell wall time (s):**

| Dataset | Seed 42 | Seed 43 | Seed 44 |
|---|---|---|---|
| CIFAR-10 | 5,626 | 5,115 | 3,850 |
| Flickr25K | 7,781 | 6,852 | 5,456 |
| MS-COCO | 7,677 | 6,710 | 5,529 |
| NUS-WIDE | 14,627 | 6,353 | — |

Cells became faster as the parallel streams finished.

**Recovery.** Any recovery of NUS-WIDE seed 44 needs the audit's recovery review and a budget decision.
No source, budget, rule or namespace has been changed.
