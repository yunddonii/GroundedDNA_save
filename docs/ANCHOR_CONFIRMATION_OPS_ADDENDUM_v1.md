# Anchor confirmation — operational addendum `anchor-confirm-ops/1`

Written for the audit (sections 703–706). It states how the stage-S request is supervised; the
same rules carry to stage D and the probes. It does not change the S/D scientific design, the
contract (`docs/ANCHOR_CONFIRMATION_CONTRACT_v2.md`, unchanged) or the approval format. Nothing
runs until an audit-ledger approval line names the manifest and the exact request below.

| Item | Pin |
|---|---|
| Source generation | `0ea88af47a375a9bf58cade194fe41adca65fc7d` on `arch-exp-2026-09-anchor-confirm` |
| Generation manifest v3 (59 files) | `artifacts/anchor_confirmation/authority_manifest_v3.json`, SHA256 `9e54bda303d6c34f080e8026492ff7eec7bfe432a6e53bd34a5a6056881a9b83` |
| Launcher (manifest member) | `scripts/phase3_selection_matrix.py`, SHA256 `253deb4f7fab03268f24986cb4493efee7389b3ce02c016c6a8ad5454b014cea` |
| Supervisor (manifest member) | `scripts/anchor_confirm_supervisor.py`, SHA256 `f634fc55447c5e946d5ec5bd73006c0f445b0230b5d8302c311a417578b7350c` |
| Stage-S request | SHA256 `0c940824a6f598d733439486cd2eab9bf6966f156edf48589cab8801a17d781f` (`artifacts/anchor_confirmation/request_preview_ancS2_v3.txt`) |

Which component supplies each guarantee (audit 704.3):

| Guarantee | Supplied by |
|---|---|
| Frozen interpreter and environment | the manifest's environment block, checked by the launcher (`load_anchor_manifest`) |
| Full input verification before any lease | the launcher, because the request carries no admission authority |
| Free space before EVERY dispatch | the launcher (`anchor_dispatch_space_refusal`, before each cell) |
| Free space, budget and wall time while cells run | the supervisor (1-s polling) |
| Accounting across S, D and probes | the supervisor's append-only ledger |
| Stop on a breach | the supervisor sends one SIGTERM to the launcher; the launcher's own handler cleans up |
| Owned processes gone before the lease release | the launcher (repaired lifecycle, audit 705/706) |
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
  - `device_seconds` is the sum of attempt process lifetimes. An attempt is one managed campaign
    child, that is, one trainer on one GPU (a child session leader of the launcher). For a probe,
    it is the probe process itself.
  - Each lifetime starts at the kernel's start time of that process (from `/proc`) and ends at the
    first 1-s poll that no longer sees it. The end is therefore an upper bound.
  - A repaired launcher keeps a trainer's leader unreaped until its whole session is drained, so
    the drain time is included.
  - Failed, partial and in-flight attempts all count.
  - `unobserved_allowance_seconds` charges one poll per planned cell, for attempts shorter than a
    poll.
  - This is an upper bound on per-attempt device occupancy. It is **not measured GPU utilisation**.
- **Ledger.** One ledger, `/data/yschoi/gdna_anchor_confirm_v1_ops/device_budget_ledger.jsonl`,
  carries the charge across S, D and every probe and is never reset. It is append-only and every
  record is fsynced. It records `start`, `attempt-start`, `attempt-end`, `poll` (every 30 s),
  `stop`, `stop-overdue` and `final`.
  - A run without a `final` record blocks every later start until the audit reconciles it. So does
    a run that ended with live attempts or held leases.
  - Only one supervisor may hold the ledger at a time (flock).
- **Rule.** Stop when prior + device + allowance + headroom ≥ 45,000 s. The headroom is
  GPUs × (1 s poll + 120 s stop bound) = 363 s on three GPUs. A start is refused when prior +
  allowance + headroom already reaches the budget.
- **Reported separately, not budgeted.**
  - `lease_gpu_seconds`: GPUs leased by the launcher's own PID (read from `/proc/locks` on the
    host-global lease files) multiplied by time. It includes idle leases; the edges are
    conservative.
  - `pre_lease_seconds`: the time from start to the first lease seen, which is the CPU
    verification.
  - `wall_seconds`.
  - The contract's arithmetic budgets cell work (S = 5.575 GPU-h). Lease occupancy for S can reach
    about 3 × 2.7 h = 8.1 GPU-h. It is recorded, not charged.

## 4. Wall-time limits

These include CPU verification: stage-S-run 6 h, stage-D-run 6 h, each smoke 2 h, each probe
15 min. The supervisor stops at limit − 121 s. The S and D limits allow about 1.4× the planned
verification plus the slowest stream (2.7 h for S; up to 2.9 h for D).

## 5. Stop mechanism

- **Triggers.** A space, budget or wall breach, an operator SIGINT/SIGTERM/SIGHUP to the
  supervisor, or any failed observation (a `/proc` read, `statvfs` or a ledger write). A failed
  observation never lets dispatch continue.
- **Stop.** The supervisor sends ONE SIGTERM to the launcher. The launcher is the supervisor's own
  unreaped child, started in its own session, so the signal reaches exactly that process. It then
  records `stop`.
- **The launcher's handler (repaired, audit 705/706).**
  1. It blocks new launches.
  2. It sends TERM to every owned session.
  3. After 5 s it re-sends SIGKILL for up to 30 s.
  4. Cleanup is complete only when no live process remains in any owned session.
  5. Only then are the GPU leases released.
  6. If a process survives, the leases stay held and the error propagates.
- **Waiting.** The supervisor keeps polling, and keeps accounting for in-flight attempts, until
  the launcher has actually exited. After 120 s it records `stop-overdue` and goes on waiting. It
  never infers death from a timeout.
- **After exit.** The supervisor checks every observed attempt, by PID and start time, for
  liveness, and checks `/proc/locks` for leases still held under the launcher's PID. Either
  finding makes the run `unclean` (exit 4) and blocks the ledger.
- **What it never does.** It signals nothing but the launcher. It never acquires or test-locks a
  lease, and never deletes or moves an artifact. Logs are kept at
  `<ops-root>/<run_id>.command.log`, append-only, alongside the ledger.

## 6. Exact commands

Run from `/data/yschoi/gdna_anchor_confirm_v1` with `GDNA_NUM_SEMANTIC_PARTS=5`. `<3 free GPUs>`
is chosen from `nvidia-smi` at admission; the request fixes the count, not the indices.

```
/data/yschoi/gdna_p3exec_authority/bin/tmux_run.sh ancS2_v3 \
  /home/yschoi/.conda/envs/dna_hashing/bin/python scripts/anchor_confirm_supervisor.py \
  --manifest artifacts/anchor_confirmation/authority_manifest_v3.json \
  --manifest-sha256 9e54bda303d6c34f080e8026492ff7eec7bfe432a6e53bd34a5a6056881a9b83 \
  --stage stage-S-run --planned-cells 24 \
  --watch-path /data/yschoi/gdna_anchor_confirm_v1_result -- \
  /home/yschoi/.conda/envs/dna_hashing/bin/python scripts/phase3_selection_matrix.py \
  --anchor-confirm select --namespace ancS2 --anchor-arms none,anchors --run --gpus <3 free GPUs> \
  --result-root /data/yschoi/gdna_anchor_confirm_v1_result \
  --input-seal flickr25k:stage1=/data/yschoi/gdna_p3exec_seals/flickr25k.stage1.input-seal.json \
  --input-seal nuswide:stage1=/data/yschoi/gdna_p3exec_seals/nuswide.stage1.input-seal.json \
  --input-seal mscoco:stage1=/data/yschoi/gdna_p3exec_seals/mscoco.stage1.input-seal.json \
  --anchor-manifest artifacts/anchor_confirmation/authority_manifest_v3.json \
  --anchor-manifest-sha256 9e54bda303d6c34f080e8026492ff7eec7bfe432a6e53bd34a5a6056881a9b83 \
  --anchor-approval-section <the approving ledger section>
```

Stage D and the probes use the same supervisor:
- stage D: `--stage stage-D-run --planned-cells <12, or fewer if reuse is admitted>`;
- each probe: `--stage probe --planned-cells 0` in front of its `anchor_confirm_code_axis.py`
  command.

## 7. Test evidence

All at source commit `0ea88af`, CPU only (`CUDA_VISIBLE_DEVICES=` empty,
`GDNA_NUM_SEMANTIC_PARTS=5`). No real GPU lease, trainer, dataset or seal payload was used.

- **Supervisor, `tests/test_anchor_confirm_supervisor.py` (12 tests).** The composed cases run a
  synthetic launcher process that executes the REAL `_with_campaign_gpu_leases`,
  `_run_managed_process` and signal handler. It uses a stand-in GPU inventory and the real
  `GpuLeaseSet` on a private lock root. The cases:
  - space refusal before start: the command never starts;
  - a space breach while a TERM-resistant child lives: the launcher's own handler exits 143, every
    owned process is gone, and no lease is held afterwards;
  - a budget breach while a child lives;
  - a failed and partial attempt charged into the next stage: the next stage is refused once the
    charge meets the budget;
  - a failed observation stops the run;
  - a command slower than the stop bound: `stop-overdue` is recorded and the final record is
    written only after the real exit;
  - an operator SIGTERM becomes one SIGTERM to the command, and the handlers are restored;
  - a launcher that leaves a live attempt: the run is `unclean`, the ledger is blocked, and the
    orphan is not signalled;
  - an unfinished prior run and a second supervisor are both refused;
  - the generation self-check;
  - the GPU count is taken from the launcher's `--gpus`;
  - the launcher and the supervisor share one set of rule values.

  An unrelated control process survives every case.
- **Launcher storage rule, four tests in `tests/test_anchor_confirm_launcher.py`:**
  - no dispatch below the floor;
  - a breach in mid-sweep stops every later dispatch;
  - the first dispatch reserves space for all twelve unfinished cells, not one stream's four;
  - the arithmetic of the rule.
- **Full run.** 14 files: **800 passed, 1 skipped** (the opt-in real-artifact test), 333 s, rc 0,
  on a clean tree.
- **Mutation battery v5, in a detached sandbox.** Every mutant must fail with the marker declared
  in advance.
  - The supervisor mutants MS1–MS11 (disabling the pre-start space refusal, the live space, budget
    and monitor-failure stops, prior-stage charging, the stop signal, the orphan check, the
    unclean/unfinished ledger refusals and the self-pin check) were all detected as declared.
  - So were the storage-rule mutants ML9 and ML10.
  - Reports and every mutant's output are in `artifacts/anchor_confirmation/mutation_v3/`.

## 8. Limits, stated

- A descendant that calls `setsid()` leaves its session and is not tracked. None of the four
  trainer shells does so, and DataLoader workers fork within the session.
- If the launcher itself is SIGKILLed from outside, its trainers keep the inherited lease
  descriptors. The supervisor then reports them as orphaned live attempts, keeps them out of any
  completed-stage claim and does not signal them.
- If a process survives SIGKILL for more than 30 s (uninterruptible), the launcher reports it,
  keeps the leases and exits nonzero. After that, the kernel lock is held only through the
  descriptors that the survivor inherited.
- If the supervisor itself is killed, the launcher runs unsupervised. The ledger then lacks a
  `final` record, which blocks every later stage until the audit reconciles it.
