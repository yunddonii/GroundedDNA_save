# Anchor confirmation — operational addendum `anchor-confirm-ops/2`

Written for the audit (sections 703–708). It states how the stage-S request is supervised; the
same rules carry to stage D and the probes.
- It supersedes `anchor-confirm-ops/1` (`b36c84ba…`) in the accounting (§3), the stop mechanism
  (§5), the pins, the commands and the evidence. The rest is unchanged.
- It does not change the S/D scientific design, the contract
  (`docs/ANCHOR_CONFIRMATION_CONTRACT_v2.md`, unchanged), the launcher or the approval format.
- Nothing runs until an audit-ledger approval line names the manifest and the exact request below.

| Item | Pin |
|---|---|
| Source generation | `a153c1577073edc0f6a97a41617b9bb6c69e69e5` on `arch-exp-2026-09-anchor-confirm` |
| Generation manifest v4 (59 files) | `artifacts/anchor_confirmation/authority_manifest_v4.json`, SHA256 `5a4481f4898fda3df27250e0214e204495022a91546d7d78e2e1c2749e17647b` |
| Launcher (manifest member, unchanged since v3; audit §707.2) | `scripts/phase3_selection_matrix.py`, SHA256 `253deb4f7fab03268f24986cb4493efee7389b3ce02c016c6a8ad5454b014cea` |
| Supervisor (manifest member) | `scripts/anchor_confirm_supervisor.py`, SHA256 `9f5184788a60d3a5e424acca251ac5ef5b2a2871f24efd87a562a188e0495c6d` |
| Stage-S request | SHA256 `033b697945067d587473c81551f05e145d124ea9f69914549fd7199d333eec54` (`artifacts/anchor_confirmation/request_preview_ancS2_v4.txt`) |

Which component supplies each guarantee (audit 704.3):

| Guarantee | Supplied by |
|---|---|
| Frozen interpreter and environment | the manifest's environment block, checked by the launcher (`load_anchor_manifest`) |
| Full input verification before any lease | the launcher, because the request carries no admission authority |
| Free space before EVERY dispatch | the launcher (`anchor_dispatch_space_refusal`, before each cell) |
| Free space, budget and wall time while cells run | the supervisor (1-s polling) |
| A stalled observation cannot keep work running | the supervisor's watchdog thread (10 s) |
| Accounting across S, D and probes, including unobserved work | the supervisor's windows, allowance and append-only ledger |
| Stop on a breach | the supervisor sends one SIGTERM to the launcher; the launcher's own handler cleans up |
| Owned processes gone before the lease release | the launcher (repaired lifecycle, audit 705/706; accepted in scope, §707.2) |
| Post-exit check for orphaned attempts and held leases | the supervisor |

## 1. Environment and full-verification scope

- Interpreter `/home/yschoi/.conda/envs/dna_hashing/bin/python`: Python 3.10.20, torch
  2.6.0+cu124, torchvision 0.21.0+cu124, numpy 2.2.6. The manifest pins python, torch and the
  interpreter path, and the launcher refuses any other. `GDNA_NUM_SEMANTIC_PARTS=5`; `PYTHONPATH`
  unset (the launcher refuses otherwise). Each child's `CUDA_VISIBLE_DEVICES` is set by the launcher
  to one sealed physical UUID.
- The request carries no admission authority. Before any lease, the launcher therefore runs full
  verification: `verify_campaign_input_seals(full=True)` → `verify_seal` → `build_seal` over the
  three stage-1 seals. This is more than hashing:
  - every sealed file is re-hashed;
  - semantic row checks;
  - NPY/NPZ arrays are loaded safely, memory-mapped;
  - the whitening mean and covariance are re-derived from the sealed train-only caption rows.

  It is CPU work with no model. The sealed targets total about 452.14 GB (Flickr25K 33.43,
  MS-COCO 161.06, NUS-WIDE 257.65; audit 703.2, deduplicated per seal). That figure comes from the
  inventory, not from measured I/O. The 82-minute figure was an estimate for a different request;
  it is not a bound. Stats-level rechecks after the lease and before the receipt are unchanged.
- Offline model and input identities (the seals' `hf_runtime` and the split identities) are
  unchanged.

## 2. Storage

- **Estimate.** One historical stage-1 N39 cell directory holds 655,495,602–655,495,734 bytes
  (0.61 GiB; ten files; `du -sb` on the p3gE Flickr25K and MS-COCO seed-42 directories,
  2026-09-27, stat only). Stage S writes 24 cells, about 14.65 GiB. Stage D writes 12, about
  7.3 GiB. Probes write JSON only. The trainer's terminal-only checkpoint policy (audit 704.2)
  means there is no second full state dict.
- **Before every dispatch (launcher, after full verification).** The free space on the result
  filesystem must be at least 10 GiB plus 0.75 GiB for every cell not yet finished (in flight or
  still to run). Otherwise the cell is refused, its stream stops, and no receipt is written. At
  the first S dispatch this rule needs 10 + 24 × 0.75 = 28 GiB; at the first D dispatch it needs
  10 + 12 × 0.75 = 19 GiB.
- **Before start (supervisor).** The same 10 GiB plus 0.75 GiB per planned cell. Otherwise the
  command never starts.
- **While running (supervisor, every second).** At least 10 GiB plus 0.75 GiB per GPU (the outputs
  that may be in flight). Otherwise it stops (section 5).
- **Observed.** `/data` had 39,000,846,336 bytes (36.32 GiB) free on 2026-09-27; the filesystem is
  99 % used by other work. If nothing else writes to `/data` after S, about 21.7 GiB would remain
  against D's 19 GiB. A refusal is the designed outcome. Nothing is deleted, and the approved
  result root is never moved. These observations do not guarantee future space.

## 3. GPU budget accounting (12.5 GPU-hours, S + D + probes)

- **Budgeted quantity: `charged_seconds` = `device_seconds` + `unobserved_allowance_seconds`.**
- **`device_seconds`** is the sum of attempt process lifetimes.
  - An attempt is one managed campaign child: one trainer on one GPU, a child session leader of the
    launcher. For a probe, it is the probe process itself; it cannot vanish before the supervisor
    reaps it.
  - Each lifetime starts at the kernel's start time of that process (from `/proc`) and ends at the
    first scan that no longer sees it, so the end is an upper bound.
  - The repaired launcher keeps a trainer's leader unreaped until its whole session is drained, so
    the drain time is included.
  - Failed, partial and in-flight attempts all count.
  - This is an upper bound on per-attempt device occupancy, **not measured GPU utilisation**.
- **Unobserved work (audit §707.1).**
  - A launcher attempt can start and end between two scans. Such an attempt lies inside one
    observation window: from the START of one scan to the END of the next, on the boot clock. A
    window therefore includes every `statvfs`, `/proc` scan, ledger write or scheduling delay.
  - The window before the command's exit is measured too.
  - The pinned launcher runs one managed child per stage-1 cell and never retries a cell
    (`_run_sweep` → `run_cell` → one `_run_managed_process`). So `unobserved_allowance_seconds` =
    planned cells × the longest window actually observed. It is recomputed at every poll and used
    in the live projection as well as in the final charge.
  - If more attempts are observed than cells were planned, that premise fails and the run becomes
    unresolved.
  - In self mode (probes) the allowance is 0.
- **Continuity.** The run is stopped, and its final record is `unresolved`, if any of these
  happens:
  - a window is longer than the 10-s watchdog bound;
  - the watchdog fires;
  - an observation or a ledger write fails;
  - more attempts are observed than cells were planned.

  An unresolved record keeps its observed charge and adds `pessimistic_bound_seconds` (GPUs × wall
  time) for reconciliation. No later stage starts until the audit reconciles it. The same holds for
  a run without a `final` record and for an `unclean` one (live attempts or held leases after exit).
- **Ledger.** One ledger, `/data/yschoi/gdna_anchor_confirm_v1_ops/device_budget_ledger.jsonl`,
  carries the charge across S, D and every probe and is never reset.
  - It is append-only, and every record is fsynced.
  - It records `start`, `attempt-start`, `attempt-end`, `poll` (every 30 s, now with the longest
    window), `stop`, `stop-overdue` and `final`.
  - Only one supervisor may hold it at a time (flock).
- **Rule.**
  - Stop when prior + device + allowance + headroom ≥ 45,000 s.
  - The headroom is GPUs × (10-s watchdog + 120-s stop bound) = 390 s on three GPUs. After the last
    completed observation, work can continue for at most the watchdog bound before the stop is
    sent, and for at most the stop bound while the launcher cleans up.
  - A start is refused when prior + planned cells × poll + headroom already reaches the budget.
- **Reported separately, not budgeted.**
  - `lease_gpu_seconds`: GPUs leased by the launcher's own PID (from `/proc/locks`) multiplied by
    time. It includes idle leases.
  - `pre_lease_seconds`: CPU verification before the lease.
  - `wall_seconds`, `max_observation_window_seconds` and `continuity_lost`.
  - The contract's arithmetic budgets cell work (S = 5.575 GPU-h). Lease occupancy for S can reach
    about 8.1 GPU-h. It is recorded, not charged.

## 4. Wall-time limits

These include CPU verification: stage-S-run 6 h, stage-D-run 6 h, each smoke 2 h, each probe
15 min. The supervisor stops at limit − 130 s (the stop bound plus the watchdog bound). The S and D limits allow about 1.4× the planned
verification plus the slowest stream (2.7 h for S; up to 2.9 h for D).

## 5. Stop mechanism

- **Triggers.** A space, budget or wall breach; an operator SIGINT/SIGTERM/SIGHUP to the
  supervisor; any loss of continuity (§3); and the watchdog.
- **Watchdog.** A separate thread checks every 0.25 s. If no observation has completed for 10 s,
  it sends the stop itself. It does not wait for a `statvfs`, scan or ledger write that has not
  returned.
- **Signalling.** Reaping and signalling the launcher share one lock, so a signal never reaches a
  reused PID. Only ONE SIGTERM is ever sent, whichever path sends it.
- **The launcher's handler (repaired, audit 705/706).**
  1. It blocks new launches.
  2. It sends TERM to every owned session.
  3. After 5 s it re-sends SIGKILL for up to 30 s.
  4. Cleanup is complete only when no live process remains in any owned session.
  5. Only then are the GPU leases released; a survivor keeps them held.
- **Waiting.** The supervisor keeps polling, and keeps accounting, until the launcher has actually
  exited. After 120 s it records `stop-overdue` and goes on waiting. It never infers death from a
  timeout.
- **After exit.** The supervisor checks every observed attempt, by PID and start time, for
  liveness, and checks `/proc/locks` for leases still held under the launcher's PID. Either finding
  makes the run `unclean` (exit 4). Loss of continuity makes it `unresolved` (exit 5). Either one
  blocks the ledger.
- **What it never does.** It signals nothing but the launcher, never acquires or test-locks a
  lease, and never deletes or moves an artifact. Logs are kept at
  `<ops-root>/<run_id>.command.log`.

## 6. Exact commands

Run from `/data/yschoi/gdna_anchor_confirm_v1` with `GDNA_NUM_SEMANTIC_PARTS=5`. `<3 free GPUs>`
is chosen from `nvidia-smi` at admission; the request fixes the count, not the indices.

```
/data/yschoi/gdna_p3exec_authority/bin/tmux_run.sh ancS2_v4 \
  /home/yschoi/.conda/envs/dna_hashing/bin/python scripts/anchor_confirm_supervisor.py \
  --manifest artifacts/anchor_confirmation/authority_manifest_v4.json \
  --manifest-sha256 5a4481f4898fda3df27250e0214e204495022a91546d7d78e2e1c2749e17647b \
  --stage stage-S-run --planned-cells 24 \
  --watch-path /data/yschoi/gdna_anchor_confirm_v1_result -- \
  /home/yschoi/.conda/envs/dna_hashing/bin/python scripts/phase3_selection_matrix.py \
  --anchor-confirm select --namespace ancS2 --anchor-arms none,anchors --run --gpus <3 free GPUs> \
  --result-root /data/yschoi/gdna_anchor_confirm_v1_result \
  --input-seal flickr25k:stage1=/data/yschoi/gdna_p3exec_seals/flickr25k.stage1.input-seal.json \
  --input-seal nuswide:stage1=/data/yschoi/gdna_p3exec_seals/nuswide.stage1.input-seal.json \
  --input-seal mscoco:stage1=/data/yschoi/gdna_p3exec_seals/mscoco.stage1.input-seal.json \
  --anchor-manifest artifacts/anchor_confirmation/authority_manifest_v4.json \
  --anchor-manifest-sha256 5a4481f4898fda3df27250e0214e204495022a91546d7d78e2e1c2749e17647b \
  --anchor-approval-section <the approving ledger section>
```

Stage D and the probes use the same supervisor:
- stage D: `--stage stage-D-run --planned-cells <12, or fewer if reuse is admitted>`;
- each probe: `--stage probe --planned-cells 0` in front of its `anchor_confirm_code_axis.py`
  command.

## 7. Test evidence

All at source commit `a153c15`, CPU only (`CUDA_VISIBLE_DEVICES=` empty,
`GDNA_NUM_SEMANTIC_PARTS=5`). No real GPU lease, trainer, dataset or seal payload was used.

- **Supervisor, `tests/test_anchor_confirm_supervisor.py` (19 tests).** The v3 cases are kept:
  - the space refusal before start;
  - the space and budget breaches while a child lives, stopped through the REAL launcher lifecycle
    in a synthetic launcher with a private lock root;
  - a failed attempt carried into the next stage;
  - the operator signal;
  - `unclean` after a leftover attempt;
  - the ledger refusals, the self pin and the CLI.

  New for audit §707.1:
  - an entire attempt between two observations, charged at least the child's own boot-clock
    lifetime, with a normal-observation control;
  - the same stall under a 1-s watchdog: stopped, `unresolved`, the next stage refused;
  - the watchdog's SIGTERM stamped by the command during the stall;
  - more attempts than cells: `unresolved`;
  - a long window crossing the live budget: stopped;
  - the headroom boundary;
  - a failed observation: `unresolved`.
- **Full run.** 14 files: **807 passed, 1 skipped** (the opt-in real-artifact test), 344 s, rc 0,
  on a clean tree.
- **Differential against the v3 supervisor `f634fc55…`.** 3/3 as declared. The between-observations
  test fails at `charged_seconds >= lived`: v3 charged 0.1 s against the child's 1.502 s.
- **Mutation battery v6, in a detached sandbox:** 9/9 detected as declared:
  - the final allowance charging the nominal poll;
  - the last window left unmeasured;
  - the live allowance charging the poll;
  - the watchdog never firing;
  - the watchdog's stop sending no signal;
  - excess attempts going unnoticed;
  - an unresolved run accepted as a prior charge;
  - a failed observation keeping the run settled;
  - the headroom leaving out the watchdog.

  Reports and outputs are in `artifacts/anchor_confirmation/mutation_v4/`; the v3-era evidence is
  in `mutation_v3/`.

## 8. Limits, stated

- A descendant that calls `setsid()` leaves its session and is not tracked. None of the four
  trainer shells does so, and DataLoader workers fork within the session.
- If the launcher itself is SIGKILLed from outside, its trainers keep the inherited lease
  descriptors. The supervisor's lease check then reports the run `unclean`, as audit §707.2
  confirmed, and it never signals them.
- If a process survives SIGKILL for more than 30 s (uninterruptible), the launcher reports it,
  keeps the leases and exits nonzero.
- The allowance's premise (one managed child per cell, no retries) is a property of the pinned
  launcher. Observed excess voids the settlement; an unobserved excess cannot be seen.
- If the supervisor process itself is frozen (all threads stopped), nothing can be stopped while
  it is frozen. The long window is recorded when it resumes and voids the settlement.
- If the supervisor is killed, the ledger lacks a `final` record, which blocks every later stage
  until the audit reconciles it.
