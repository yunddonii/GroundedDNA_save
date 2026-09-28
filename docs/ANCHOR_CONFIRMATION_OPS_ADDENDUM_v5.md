# Anchor confirmation — operational addendum `anchor-confirm-ops/5`

Written for the audit (§722–§723). It states how generation v7 (contract v3 as revised for v7, the
fixed four-dataset anchor model) is supervised, in two separately approved steps: a one-cell stage-S
smoke, then stage S. The same rules carry to stage D and the probes.
- It supersedes `anchor-confirm-ops/4` (`docs/ANCHOR_CONFIRMATION_OPS_ADDENDUM_v4.md`, `b5184154…`,
  kept as history) in: the child environment (§1a), the two admission modes (§1: full or carried),
  the prior charge (§3), the smoke (§4, §6), the namespaces and tmux records (§6), the pins and the
  evidence (§7).
- The supervisor is unchanged (`b5eecf58…`). The accounting rules, watchdog, stop mechanism, storage
  rule, wall limits and unresolved/unclean settlement are unchanged.
- The failed attempts stay recorded as they are: generation v5 (`ancS5`, tmux `ancS5_v5`, run
  `20260927T143532Z-4ada8a0d`, refused at the full input check) and generation v6 (`ancS6`, tmux
  `ancS6_v6`, run `20260928T052252Z-35772b95`, all four seals admitted, then every trainer refused
  on `library_environment`). Neither trained a cell.
- Nothing runs until an audit-ledger approval line names the manifest and the exact request below.
  A passing smoke does not approve S.

| Item | Pin |
|---|---|
| Source generation | `59477d8be2c1750139ae436af778fd324c24d35a` on `arch-exp-2026-09-anchor-confirm` |
| Generation manifest v7 (62 files) | `artifacts/anchor_confirmation/authority_manifest_v7.json`, SHA256 `c061296309fe41203205dff65de3a1afb843c950e93c0614dfd27a8216a90128` |
| Contract v3, revised for v7 (manifest member) | `docs/ANCHOR_CONFIRMATION_CONTRACT_v3.md`, SHA256 `e6c4978e6d30559e01c2ee34ae2b4d39c01717557ee16c9e96f85980aa8d3efc` |
| Launcher (manifest member) | `scripts/phase3_selection_matrix.py`, SHA256 `2cb8343dba923514d9c3fbc0ffffd79c66058d582c35486fab51a518ba541f53` |
| Supervisor (manifest member, unchanged) | `scripts/anchor_confirm_supervisor.py`, SHA256 `b5eecf58cbbc9d7d03463e4ea7a35ac107842aa6657a3b66e3770528cbdd1dd0` |
| Historical verifier (pinned; unchanged) | `/data/yschoi/gdna_p3exec/scripts/seal_phase3_inputs.py`, SHA256 `12233f8e4967c90afcab131a1061fe76abfbaed86648e1388826538a6d42f214` |
| Carried admission (proposed) | `artifacts/anchor_confirmation/ancS6_snapshot_acaa0374fa0780e2.json`, SHA256 `ef5a3e8d857a7a6cc22d1e978854c3f721470ed19f426a020873718199a2b8ff` |
| **Smoke request, carried admission (proposed first)** | SHA256 `27928f010fd4c2a09f64386eb94c96d4999c51f37af01b44a0a9961d97b780c8` (`request_preview_ancSmk7_v7_carried.txt`) |
| Smoke request, full admission (alternative) | SHA256 `21834192e69515f2c7caa168edd3a7a1f4d3198d7d5e21ab93f6a9848a4ea1e7` (`request_preview_ancSmk7_v7_full.txt`) |
| **Stage-S request, carried admission (after the smoke)** | SHA256 `bf2d3f56e0292020b361794d1b1238613ccc9b5998b4a84795d7dd756a7741e3` (`request_preview_ancS7_v7_carried.txt`) |
| Stage-S request, full admission (alternative) | SHA256 `495e24d0c199345af802098fc3b08d86398a6bbdeffa4abe1aa548a5709349b3` (`request_preview_ancS7_v7_full.txt`) |

Which component supplies each guarantee (unchanged from ops/2 except the numbers):

| Guarantee | Supplied by |
|---|---|
| Frozen interpreter and environment | the manifest's environment block, checked by the launcher (`load_anchor_manifest`) |
| Input admission before any lease | the launcher: either a full check through the historical verifier bridge, or the stats-only check against a carried full historical admission that the request pins (§1) |
| The child starts under the attested start-up environment | the launcher's `build_command` (§1a); the trainer's own check is unchanged |
| One dataset stream per GPU, the four datasets present | the launcher: the request's GPU count must equal its dataset streams (4), with distinct GPUs |
| Free space before EVERY dispatch | the launcher (`anchor_dispatch_space_refusal`, before each cell) |
| Free space, budget and wall time while cells run | the supervisor (1-s polling) |
| A stalled observation cannot keep work running | the supervisor's watchdog thread (10 s) |
| Accounting across S, D and probes, including unobserved work | the supervisor's windows, allowance and append-only ledger |
| Stop on a breach | the supervisor sends one SIGTERM to the launcher; the launcher's own handler cleans up |
| Owned processes gone before the lease release | the launcher (lifecycle unchanged; accepted in scope, §707.2) |
| Post-exit check for orphaned attempts and held leases | the supervisor |

## 1. Environment and full-verification scope

- Interpreter, packages and environment: as in ops/2 §1 (Python 3.10.20, torch 2.6.0+cu124,
  `GDNA_NUM_SEMANTIC_PARTS=5`, `PYTHONPATH` unset, one sealed GPU UUID per child).
- **Two admission modes, fixed by the approved request.**
  - **Carried** (proposed; the smoke and S requests marked "carried"). The request pins the ancS6
    plan snapshot (`ef5a3e8d…`). Before any lease the launcher reads it at those bytes, requires
    that it records the historical verifier's rc-0 admission of exactly the four carried seals on
    the same seal bytes and aggregates under the pinned verifier and root
    (`anchor_carried_admission_refusal`, contract v3 §5), and runs the existing protocol's stats-only
    recheck (`load_admission_authority` → `verify_seal_stats`). That recheck covers every sealed
    file record, the six historical sources included, and the directory inventories, but no
    content. It takes seconds: 0.0 s measured on 2026-09-28 against the real seals, metadata only,
    which also found the guard satisfied and the four seal files at their pinned digests.
    The protocol's stated limit applies: a byte rewrite that preserves every stat is not seen.
  - **Full** (the alternatives marked "full"). The request carries no admission authority, so before
    any lease the launcher runs a full verification of each of the **four** stage-1 seals through
    `verify_seal_historically` (contract v3 §5; audit §715–§716). The v6 attempt measured 4721 s for
    the four (CIFAR-10 529, Flickr25K 252, MS-COCO 1515, NUS-WIDE 2425):
  1. the historical verifier's bytes are the pinned `12233f8e…`;
  2. the seal JSON's aggregate holds, and its six historical source records (five caption-foil
     producers and `val_split.py`) name the pinned root `/data/yschoi/gdna_p3exec` and pinned
     content digests;
  3. the six historical files, measured now with the seal's own record builder, are exactly the
     sealed records (content and stat identity). A changed file refuses with no child started, so
     `val_split.py` is authenticated before the verifier executes it;
  4. this tree's `val_split.py` has the sealed content;
  5. the child `python -I -B /data/yschoi/gdna_p3exec/scripts/seal_phase3_inputs.py verify --seal
     PATH` runs in the historical tree. It is the unchanged legacy full check: every sealed file
     re-hashed, the semantic row checks, the safe memory-mapped NPY/NPZ loads, the re-derived
     whitening statistics and the exact comparison with the sealed payload;
  6. afterwards the verifier bytes, the seal bytes and the six source observations must be
     unchanged, the exit code 0 and the report exactly `verified PATH AGGREGATE`;
  7. the launcher derives the input authority from the seal bytes it hashed in step 2.

  CPU work with no model. The plan snapshot records the bridge evidence per seal
  (`authorities.historical_input_admission`).
- **The verifier child and the supervisor.** The child is a direct child of the launcher in the
  launcher's own session, not a session leader, so the supervisor does not count it as a training
  attempt (`managed_sessions` counts only child session leaders). It dies with the launcher
  (`PR_SET_PDEATHSIG`, SIGKILL): before any lease the launcher has no signal handler, so a
  supervisor stop kills the launcher outright and the child with it. An interrupted wait kills and
  reaps it. Its lifetime is launcher wall time, not device time.
- Sealed target sizes, from each seal's inventory (sum of the sealed files' recorded sizes,
  deduplicated per seal; JSON only, no payload read, 2026-09-27):

  | Seal (`/data/yschoi/gdna_p3exec_seals/`) | Files | GB |
  |---|---:|---:|
  | `cifar10.stage1.input-seal.json` | 47 | 79.55 |
  | `flickr25k.stage1.input-seal.json` | 43 | 33.43 |
  | `nuswide.stage1.input-seal.json` | 43 | 257.65 |
  | `mscoco.stage1.input-seal.json` | 43 | 161.06 |
  | **Total** | 176 | **531.69** |

  The three multi-label figures equal audit §703.2's, and audit §712.2 recorded the same total.
  The v6 attempt's full check took 4721 s in total (about 79 min; part of the CIFAR-10 payload may
  have been in the page cache). A later full check may differ.
- The seals record these split populations (`split_identity.counts`), which the probe population
  relies on (contract v3 §8.2): held-out validation 500 / 500 / 1050 / 1000 rows for CIFAR-10 /
  Flickr25K / NUS-WIDE / MS-COCO.

## 1a. The child environment (generation v7; audit §722–§723)

- The plan attests the launcher's START-UP values of `LD_LIBRARY_PATH`, `HF_HOME`,
  `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE` (`caller_environment`, the exec block), and each
  trainer checks its own start-up values against them (`verify_child_environment`, unchanged).
- `build_command` now takes those four from the same start-up block, and removes any that were
  absent at start-up. Before, it copied them from `os.environ`, which the anchor admission's
  `config` import (-> dataloaders -> cv2) rewrites in memory.
- A reproduction on the unrepaired launcher shows the exact failure for all four datasets, with
  the variable unset and set. The repaired launcher admits all eight (§7).
- The supervisor command sets `LD_LIBRARY_PATH` nowhere: the start-up value is whatever the tmux
  runner's `env -C … -u PYTHONPATH -u CUDA_VISIBLE_DEVICES GDNA_NUM_SEMANTIC_PARTS=5` hands the
  supervisor and launcher (the tmux server currently holds
  `/usr/local/cuda-12.4/lib64:/usr/local/cuda/extras/CUPTI/:`). The plan records it and the
  children receive it.

## 2. Storage

- **Where.** Run directories: `/home/yschoi/gdna_anchor4_result`. Operations ledger and logs:
  `/home/yschoi/gdna_anchor4_ops`. Both are on the root filesystem (`/dev/sda4`), by the user's
  decision (contract v3 §2). Neither exists yet. The trainer creates the result root with its
  first run directory (`os.makedirs` in `train_siglip2.py`), and the supervisor creates the
  operations root. Until then both space checks read the nearest existing parent (`/home/yschoi`,
  the same filesystem). The records stay in the worktree (`artifacts/anchor_confirmation/`).
- **Estimate.** Historical stage-1 cell directories (`du -sb`, stat only, 2026-09-27):
  - CIFAR-10 p3gE seed 42, N 4/9/19/39: 652,409,732 – 652,542,512 bytes;
  - Flickr25K and MS-COCO p3gE N39 seed 42: 655,495,602 – 655,496,190 bytes.

  All are 0.61 GiB, under the rule's 0.75 GiB per cell. Stage S writes 16 cells, about 9.8 GiB;
  stage D writes 8, about 4.9 GiB. Probes write JSON only.
- **Before every dispatch (launcher, after full verification).** Free space on the result
  filesystem ≥ 10 GiB + 0.75 GiB × every cell not yet finished. At the first S dispatch that is
  10 + 16 × 0.75 = **22 GiB**; at the first D dispatch, 10 + 8 × 0.75 = **16 GiB**.
- **Before start (supervisor).** The same rule over the planned cells. Otherwise the command never
  starts.
- **While running (supervisor, every second).** At least 10 GiB + 0.75 GiB × 4 GPUs = 13 GiB.
  Otherwise it stops (§5).
- **Observed.** `/` had 391,827,513,344 bytes (364.9 GiB) free on 2026-09-27 (`df -B1`, 56 % used).
  This does not guarantee future space. A refusal is the designed outcome; nothing is deleted or
  moved.

## 3. GPU budget accounting (12.5 GPU-hours, S + D + probes)

The rules are ops/2 §3, unchanged in code. Only the numbers change:

- **Charged quantity.** `charged_seconds` = `device_seconds` + `unobserved_allowance_seconds`.
- **Allowance.** Planned cells × the longest observation window actually observed: 16 cells for S,
  8 for D, 0 for a probe (self mode).
- **Stop rule.** Stop when prior + device + allowance + headroom ≥ 45,000 s. The headroom is
  GPUs × (10-s watchdog + 120-s stop bound) = **520 s on four GPUs** (390 s on three in ops/2).
- **Refusal at start.** A start is refused when prior + planned cells × poll + headroom already
  reaches the budget.
- **Per-stage GPU maximum** (`STAGES` in the supervisor). 4 for the S and D runs and smokes; 1 for a
  probe. The supervisor takes the count from the launcher's own `--gpus` and refuses a command that
  names more.
- **One ledger for the whole chain.** `/home/yschoi/gdna_anchor4_ops/device_budget_ledger.jsonl`
  carries S, D and every probe. It is append-only and fsynced, one supervisor holds it at a time
  (flock), and it is never reset.
- **Prior charge: 73.88978339359164 s.** It is not subtracted or reset, and the next start record
  must show it as `prior_charged_seconds`. It is the sum of:
  - the failed generation-v5 attempt: 19.092536613345146 s (run `20260927T143532Z-4ada8a0d`, no
    attempts);
  - the failed generation-v6 attempt: 54.797246780246496 s (run `20260928T052252Z-35772b95`: 4
    attempts of about 8 s each = 33.35 device seconds, plus the 21.45-s observation allowance).
- **The smoke** is charged in the same ledger: one cell, one epoch of Flickr25K N=4 on one GPU. The
  planning charge is well under 0.1 GPU-h (its device lifetime plus 1 planned cell × the longest
  window).
- **Unresolved, unclean, no final record.** Each blocks every later stage until the audit
  reconciles it (ops/2 §3).
- **Planning arithmetic (contract v3 §12), not a runtime promise.** S 3.50 GPU-h; D 0.47–3.73;
  probes ≈ 0.4; total ≈ 4.4–7.6 of the 12.5-GPU-h ceiling. Lease occupancy (reported, not charged)
  can reach about 4 GPUs × the slowest S stream (NUS-WIDE, about 81 min) ≈ 5.4 GPU-h.

## 4. Wall-time limits

These include CPU verification: stage-S-run 4 h, stage-D-run 4 h, each smoke 2 h, each probe 15 min.
The supervisor stops at limit − 130 s (the stop bound plus the watchdog bound).
- **Smoke:** carried admission (seconds), rendering, then one trainer: data and model loading, one
  epoch (about 0.4 min for Flickr25K), validation and the record. Minutes, against a 2-h limit; with
  a full admission about 79 min more.
- **S, carried admission:** the slowest stream (NUS-WIDE, 75 epochs × 1.08 min ≈ 81 min), about 1.4 h.
  **S, full admission:** 79 min (measured in v6) + 81 min ≈ 2.7 h. The limit is about 1.5× that.
- **D, worst case** (NUS-WIDE at N = 39, full admission): 79 min + 2 × 40 epochs × 1.08 min ≈ 2.8 h. The limit is
  about 1.4× that.

## 5. Stop mechanism

Unchanged from ops/2 §5: the triggers, the 10-s watchdog thread, one SIGTERM under the reaping
lock, the launcher's TERM → 5 s → SIGKILL for up to 30 s cleanup with leases released only after
every owned session is empty, `stop-overdue` after 120 s with no death inferred, and the post-exit
`unclean` (exit 4) / `unresolved` (exit 5) checks.

## 6. Exact commands

Run from `/data/yschoi/gdna_anchor_confirm_v1`. A new tmux session takes its environment from the
tmux server, so each command sets the working directory, `PYTHONPATH`/`CUDA_VISIBLE_DEVICES` unset
and `GDNA_NUM_SEMANTIC_PARTS=5` itself through its `env -C …` prefix. GPUs are chosen from
`nvidia-smi` at admission; each request fixes the count, not the indices.

**Step 1 — the one-cell smoke** (request `27928f01…`; one free GPU; never evidence):

```
/data/yschoi/gdna_p3exec_authority/bin/tmux_run.sh ancSmk7_v7 \
  env -C /data/yschoi/gdna_anchor_confirm_v1 -u PYTHONPATH -u CUDA_VISIBLE_DEVICES GDNA_NUM_SEMANTIC_PARTS=5 \
  /home/yschoi/.conda/envs/dna_hashing/bin/python scripts/anchor_confirm_supervisor.py \
  --manifest artifacts/anchor_confirmation/authority_manifest_v7.json \
  --manifest-sha256 c061296309fe41203205dff65de3a1afb843c950e93c0614dfd27a8216a90128 \
  --stage stage-S-smoke --planned-cells 1 \
  --watch-path /home/yschoi/gdna_anchor4_result -- \
  /home/yschoi/.conda/envs/dna_hashing/bin/python scripts/phase3_selection_matrix.py \
  --anchor-confirm select --namespace ancSmk7 --anchor-arms anchors \
  --smoke --only flickr25k:4:anchors:42 --epochs 1 --gpus <1 free GPU> \
  --result-root /home/yschoi/gdna_anchor4_result \
  --input-seal cifar10:stage1=/data/yschoi/gdna_p3exec_seals/cifar10.stage1.input-seal.json \
  --input-seal flickr25k:stage1=/data/yschoi/gdna_p3exec_seals/flickr25k.stage1.input-seal.json \
  --input-seal nuswide:stage1=/data/yschoi/gdna_p3exec_seals/nuswide.stage1.input-seal.json \
  --input-seal mscoco:stage1=/data/yschoi/gdna_p3exec_seals/mscoco.stage1.input-seal.json \
  --anchor-manifest artifacts/anchor_confirmation/authority_manifest_v7.json \
  --anchor-manifest-sha256 c061296309fe41203205dff65de3a1afb843c950e93c0614dfd27a8216a90128 \
  --admission-authority /data/yschoi/gdna_anchor_confirm_v1/artifacts/anchor_confirmation/ancS6_snapshot_acaa0374fa0780e2.json \
  --anchor-approval-section <the approving ledger section>
```

**Step 2 — stage S, after the smoke and its own approval** (request `bf2d3f56…`; four free GPUs):

```
/data/yschoi/gdna_p3exec_authority/bin/tmux_run.sh ancS7_v7 \
  env -C /data/yschoi/gdna_anchor_confirm_v1 -u PYTHONPATH -u CUDA_VISIBLE_DEVICES GDNA_NUM_SEMANTIC_PARTS=5 \
  /home/yschoi/.conda/envs/dna_hashing/bin/python scripts/anchor_confirm_supervisor.py \
  --manifest artifacts/anchor_confirmation/authority_manifest_v7.json \
  --manifest-sha256 c061296309fe41203205dff65de3a1afb843c950e93c0614dfd27a8216a90128 \
  --stage stage-S-run --planned-cells 16 \
  --watch-path /home/yschoi/gdna_anchor4_result -- \
  /home/yschoi/.conda/envs/dna_hashing/bin/python scripts/phase3_selection_matrix.py \
  --anchor-confirm select --namespace ancS7 --anchor-arms anchors --run --gpus <4 free GPUs> \
  --result-root /home/yschoi/gdna_anchor4_result \
  --input-seal cifar10:stage1=/data/yschoi/gdna_p3exec_seals/cifar10.stage1.input-seal.json \
  --input-seal flickr25k:stage1=/data/yschoi/gdna_p3exec_seals/flickr25k.stage1.input-seal.json \
  --input-seal nuswide:stage1=/data/yschoi/gdna_p3exec_seals/nuswide.stage1.input-seal.json \
  --input-seal mscoco:stage1=/data/yschoi/gdna_p3exec_seals/mscoco.stage1.input-seal.json \
  --anchor-manifest artifacts/anchor_confirmation/authority_manifest_v7.json \
  --anchor-manifest-sha256 c061296309fe41203205dff65de3a1afb843c950e93c0614dfd27a8216a90128 \
  --admission-authority /data/yschoi/gdna_anchor_confirm_v1/artifacts/anchor_confirmation/ancS6_snapshot_acaa0374fa0780e2.json \
  --anchor-approval-section <the approving ledger section>
```

The "full" alternatives (`21834192…`, `495e24d0…`) are the same commands without the
`--admission-authority` line. The operations root defaults to `/home/yschoi/gdna_anchor4_ops`. Stage
D (`--stage stage-D-run --planned-cells 8`, namespace `ancD7`) and the probes (`--stage probe
--planned-cells 0`) use the same supervisor.

## 7. Test evidence

All CPU only (`CUDA_VISIBLE_DEVICES=` empty, `GDNA_NUM_SEMANTIC_PARTS=5`). No real trainer, dataset,
model, GPU, lease or full input verification was used.

- **Full run at `59477d8`** (the manifest's source commit), 16 files (v6's 15 plus the new handoff
  file), tracked tree clean: **921 passed, 1 skipped** (the opt-in real-artifact test), 519 s, rc 0
  (`artifacts/anchor_confirmation/mutation_v7/suite_59477d8.log`). New or changed counts: launcher
  tests 193 (1 opt-in skip); `tests/test_anchor_confirm_env_handoff.py` 17 (new); the other anchor
  files are as in v6.
- **The composed boundary** (`tests/test_anchor_confirm_env_handoff.py`). A fresh process with a
  controlled start-up block does the following:
  - records the real parent fingerprint;
  - really imports `config` (the cv2 rewrite is asserted, as a positive control);
  - runs the real `build_command` for a real anchor cell;
  - runs the pinned wrapper under bash, with a capture child that imports cv2 itself and checks its
    own start-up environment with the real `verify_child_environment`.

  The cases:
  - all four datasets × runtime variables unset or set to the failed run's value: admitted, and the
    child observes exactly the start-up values;
  - all four variables set: all four survive;
  - the child's own cv2 rewrite does not matter;
  - five genuine changes (another, removed or added `LD_LIBRARY_PATH`, an added `HF_HUB_OFFLINE`, a
    changed `TRANSFORMERS_OFFLINE`) refuse with the real message;
  - an in-process `build_command` test covers the start-up source and the removal of absent keys.

  Only the library, `pythonpath` and interpreter fields are measured live in the child; the GPU,
  torch-device and package fields are copied from the expectation (no GPU).
- **Diagnostic reproduction (§722.2 item 2)** (`env_handoff_v7/diagnose_env_handoff.py`,
  `6a52490c…`; report `env_handoff_v7/diagnosis_59477d8.json`, `5f6ad256…`).
  - Same driver, on the unrepaired launcher (a detached sandbox at `6b8d83d`) and on the repaired
    tree (`59477d8`), 4 datasets × unset/set.
  - Unrepaired: **8/8 refused** with exactly the real run's message. The only differing key is
    `LD_LIBRARY_PATH`: the plan expects the start-up value (absent, or the CUDA/CUPTI path), and the
    child received cv2's `…/site-packages/cv2/../../lib64:` prepended.
  - Repaired: **8/8 admitted**.
  - An earlier run of the same script on the uncommitted fix (`diagnosis.json`, `a859a6e8…`) gave
    the same result.
- **Carried-admission guard:**
  - the positive case, and one refusal each for another returncode, root, verifier, seal digest,
    aggregate, seal path, report or a missing seal;
  - a snapshot without historical evidence;
  - a snapshot not at its approved bytes;
  - end to end in the launcher: with evidence it takes the stats-only check (`full=False`, the
    carried authorities), and without evidence it refuses before the lease.
- **The legacy caller-environment test** (`tests/test_phase3_selection_matrix.py`, not a closure
  member) now supplies the caller's value as the start-up block. It checks that an in-memory rewrite
  does not reach the child; its old premise (an in-memory value reaches the child) was the defect.
- **Mutation battery v9, in a detached sandbox** (harness `mutation_v7/mutation_v9_frozen.py`,
  `c7dacd93…`; report `mutation_v7/mutation_v9/report.json`, `f2275808…`). **9/9 detected as
  declared** at `59477d8`, after all 13 declared tests passed unmutated. Tree and status unchanged,
  no stray process.
  - EX1: `os.environ` copied again;
  - EX2: absent keys left behind;
  - EX3, EX7: the guard not called or its result dropped;
  - EX4: the evidence set not checked (declared as a reason change);
  - EX5: rows not compared;
  - EX6: the approved-bytes check off;
  - EX8: a start-up value dropped;
  - EX9 (control): the trainer-side comparator disabled, which proves the refusal cases are live.

## 8. Limits, stated

- New in ops/5: the composed test measures only the library, `pythonpath` and interpreter fields in
  the child. The GPU, torch-device and package fields are checked only by a real trainer, which is
  why the smoke comes first.
- The carried admission gives up what the protocol's stats-only path always gives up: a byte
  rewrite between campaigns that preserves every file's lstat, link text, target stat and directory
  inventory.

- New in ops/4: the verifier child runs `-I` (no user site, no `PYTHON*` variables, no script
  directory on `sys.path`) with the pinned interpreter. Its confinement rests on the pinned verifier
  bytes, the six authenticated sources and the recheck after it. It is not OS-level isolation.
- Between the pre-child measurement and the child's own read of a source, a change followed by a
  restore cannot keep its stat identity (ctime changes on any write), so the post-child
  measurement refuses it. A change the child itself notices refuses earlier.

Unchanged from ops/2 §8: a `setsid()` descendant is not tracked; a launcher SIGKILLed from outside
leaves its trainers holding the inherited leases (reported `unclean`, never signalled); a process
surviving SIGKILL for more than 30 s keeps the leases; the allowance's one-child-per-cell premise
belongs to the pinned launcher; a frozen supervisor stops nothing while frozen; a killed supervisor
leaves no `final` record and blocks every later stage. New here: the result root and the
operations root share the root filesystem, so the ledger's own writes count against the same free
space the storage rule watches (a few kB per poll).
