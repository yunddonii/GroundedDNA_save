# Anchor confirmation — operational addendum `anchor-confirm-ops/4`

Written for the audit (§715–§718). It states how the stage-S request of generation v6 (contract
v3 as revised for v6, the fixed four-dataset anchor model) is supervised; the same rules carry to
stage D and the probes.
- It supersedes `anchor-confirm-ops/3` (`docs/ANCHOR_CONFIRMATION_OPS_ADDENDUM_v3.md`, `f270229e…`,
  kept as history) in: the full input verification (§1, the historical verifier bridge), the prior
  charge (§3), the namespace and tmux record (§6), the pins and the evidence (§7).
- The supervisor is unchanged (`b5eecf58…`). The accounting rules, watchdog, stop mechanism,
  storage rule, wall limits and unresolved/unclean settlement of ops/3 are unchanged.
- The failed generation-v5 attempt (`ancS5`, tmux `ancS5_v5`, run `20260927T143532Z-4ada8a0d`) stays
  recorded as it is. Nothing of it is reused; it trained no cell.
- Nothing runs until an audit-ledger approval line names the manifest and the exact request below.

| Item | Pin |
|---|---|
| Source generation | `1f93db92ade8fad60d6b963f13ff8016f8adf1b4` on `arch-exp-2026-09-anchor-confirm` |
| Generation manifest v6 (61 files) | `artifacts/anchor_confirmation/authority_manifest_v6.json`, SHA256 `a5ff2a0e93acd3a0bef7ecc47a69d550b28fc7ee57c1a25aa48e7e1169f739d7` |
| Contract v3, revised for v6 (manifest member) | `docs/ANCHOR_CONFIRMATION_CONTRACT_v3.md`, SHA256 `a97ed217dca96fc44f91944951b868a1699ca95acb1cea128965349cdd81387b` |
| Launcher (manifest member) | `scripts/phase3_selection_matrix.py`, SHA256 `0f7081b9770735ebc22c8a7e8b02ce9fbd3d5792a367ce16d762dd22d8d142da` |
| Supervisor (manifest member, unchanged) | `scripts/anchor_confirm_supervisor.py`, SHA256 `b5eecf58cbbc9d7d03463e4ea7a35ac107842aa6657a3b66e3770528cbdd1dd0` |
| Historical verifier (pinned in the launcher and the manifest) | `/data/yschoi/gdna_p3exec/scripts/seal_phase3_inputs.py`, SHA256 `12233f8e4967c90afcab131a1061fe76abfbaed86648e1388826538a6d42f214` |
| Stage-S request | SHA256 `1625b50faf74b311040a8f56f1d7103b7352e2733e8c8fe9b8860ed18a44793c` (`artifacts/anchor_confirmation/request_preview_ancS6_v6.txt`) |

Which component supplies each guarantee (unchanged from ops/2 except the numbers):

| Guarantee | Supplied by |
|---|---|
| Frozen interpreter and environment | the manifest's environment block, checked by the launcher (`load_anchor_manifest`) |
| Full input verification before any lease | the launcher, through the historical verifier bridge (§1), because the request carries no admission authority |
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
- The request carries no admission authority, so before any lease the launcher runs a full
  verification of each of the **four** stage-1 seals through `verify_seal_historically` (contract v3
  §5; audit §715–§716):
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
  The v5 attempt read about 94 MB/s over four minutes, which puts the full check near 95 min. This
  is an estimate, not a bound.
- The seals record these split populations (`split_identity.counts`), which the probe population
  relies on (contract v3 §8.2): held-out validation 500 / 500 / 1050 / 1000 rows for CIFAR-10 /
  Flickr25K / NUS-WIDE / MS-COCO.

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
- **Prior charge: 19.092536613345146 s.** This is the failed generation-v5 attempt (run
  `20260927T143532Z-4ada8a0d`, final record `exited`, rc 2): 16 planned cells × its longest
  1.1932835383340716-s observation window, with 0 device seconds, no attempts and no leases. The
  next start record must show it as `prior_charged_seconds`. It is not subtracted or reset.
- **Unresolved, unclean, no final record.** Each blocks every later stage until the audit
  reconciles it (ops/2 §3).
- **Planning arithmetic (contract v3 §12), not a runtime promise.** S 3.50 GPU-h; D 0.47–3.73;
  probes ≈ 0.4; total ≈ 4.4–7.6 of the 12.5-GPU-h ceiling. Lease occupancy (reported, not charged)
  can reach about 4 GPUs × the slowest S stream (NUS-WIDE, about 81 min) ≈ 5.4 GPU-h.

## 4. Wall-time limits

These include CPU verification: stage-S-run 4 h, stage-D-run 4 h, each smoke 2 h, each probe 15 min.
The supervisor stops at limit − 130 s (the stop bound plus the watchdog bound).
- **S:** about 95 min of verification (estimate, §1) + the slowest stream (NUS-WIDE, 75 epochs ×
  1.08 min ≈ 81 min) ≈ 2.9 h. The limit is about 1.4× that.
- **D, worst case** (NUS-WIDE at N = 39): 95 min + 2 × 40 epochs × 1.08 min ≈ 3.0 h. The limit is
  about 1.3× that.

## 5. Stop mechanism

Unchanged from ops/2 §5: the triggers, the 10-s watchdog thread, one SIGTERM under the reaping
lock, the launcher's TERM → 5 s → SIGKILL for up to 30 s cleanup with leases released only after
every owned session is empty, `stop-overdue` after 120 s with no death inferred, and the post-exit
`unclean` (exit 4) / `unresolved` (exit 5) checks.

## 6. Exact commands

Run from `/data/yschoi/gdna_anchor_confirm_v1` with `GDNA_NUM_SEMANTIC_PARTS=5` and `PYTHONPATH`
unset. A new tmux session takes its environment from the tmux server, so the command sets both,
and the working directory, itself through its `env -C …` prefix (as the v5 attempt did).
`<4 free GPUs>` is chosen from `nvidia-smi` at admission; the request fixes the count (4), not the
indices.

```
/data/yschoi/gdna_p3exec_authority/bin/tmux_run.sh ancS6_v6 \
  env -C /data/yschoi/gdna_anchor_confirm_v1 -u PYTHONPATH -u CUDA_VISIBLE_DEVICES GDNA_NUM_SEMANTIC_PARTS=5 \
  /home/yschoi/.conda/envs/dna_hashing/bin/python scripts/anchor_confirm_supervisor.py \
  --manifest artifacts/anchor_confirmation/authority_manifest_v6.json \
  --manifest-sha256 a5ff2a0e93acd3a0bef7ecc47a69d550b28fc7ee57c1a25aa48e7e1169f739d7 \
  --stage stage-S-run --planned-cells 16 \
  --watch-path /home/yschoi/gdna_anchor4_result -- \
  /home/yschoi/.conda/envs/dna_hashing/bin/python scripts/phase3_selection_matrix.py \
  --anchor-confirm select --namespace ancS6 --anchor-arms anchors --run --gpus <4 free GPUs> \
  --result-root /home/yschoi/gdna_anchor4_result \
  --input-seal cifar10:stage1=/data/yschoi/gdna_p3exec_seals/cifar10.stage1.input-seal.json \
  --input-seal flickr25k:stage1=/data/yschoi/gdna_p3exec_seals/flickr25k.stage1.input-seal.json \
  --input-seal nuswide:stage1=/data/yschoi/gdna_p3exec_seals/nuswide.stage1.input-seal.json \
  --input-seal mscoco:stage1=/data/yschoi/gdna_p3exec_seals/mscoco.stage1.input-seal.json \
  --anchor-manifest artifacts/anchor_confirmation/authority_manifest_v6.json \
  --anchor-manifest-sha256 a5ff2a0e93acd3a0bef7ecc47a69d550b28fc7ee57c1a25aa48e7e1169f739d7 \
  --anchor-approval-section <the approving ledger section>
```

The operations root defaults to `/home/yschoi/gdna_anchor4_ops`. Stage D and the probes use the same
supervisor:
- stage D: `--stage stage-D-run --planned-cells 8`, namespace `ancD6`, with the frozen N record;
- each probe: `--stage probe --planned-cells 0` in front of its `anchor_confirm_code_axis.py`
  command.

## 7. Test evidence

All CPU only (`CUDA_VISIBLE_DEVICES=` empty, `GDNA_NUM_SEMANTIC_PARTS=5`). No real seal, cache,
GPU lease or trainer was used. The bridge tests build real small seals from private fixtures.

- **Full run at `1f93db9`** (the manifest's source commit), 15 files (the 14 of v5 plus the new
  bridge file), tracked tree clean: **891 passed, 1 skipped** (the opt-in real-artifact test),
  364 s, rc 0 (`artifacts/anchor_confirmation/mutation_v6/suite_1f93db9.log`).

  | Suite | Tests |
  |---|---:|
  | `tests/test_anchor_confirm_port.py` | 16 |
  | `tests/test_anchor_confirm_recipe.py` | 69 |
  | `tests/test_anchor_confirm_launcher.py` | 180 (1 opt-in skip) |
  | `tests/test_anchor_confirm_reducer.py` | 183 |
  | `tests/test_anchor_confirm_lifecycle.py` | 11 |
  | `tests/test_anchor_confirm_supervisor.py` | 19 |
  | `tests/test_anchor_confirm_input_bridge.py` (new) | 50 |
  | legacy and related (8 files, unchanged) | 364 |

- **The bridge file** builds a real small seal with a byte copy of the verifier inside a temporary
  "historical" tree and checks it from this tree:
  - the in-process full check refuses (the §715 failure, reproduced), and the same-tree legacy
    check passes;
  - the bridge admits the relocated seal, and its authority equals the in-process derivation;
  - refused before any child starts: an unpinned verifier; each of the six sources with another
    pinned digest; each source altered in content; each source re-stamped (same bytes, new mtime);
    a substituted root; a seal built in this tree; extra, missing or re-pointed source records; a
    broken aggregate; a changed new-generation `val_split.py`;
  - refused after the child: each source changed while the child ran; changed payload bytes (the
    legacy rebuild refuses); a forged report (another aggregate, another seal, silence, extra
    output); a nonzero exit; the seal or the verifier changed during the child; a stale expected
    authority. An honest stand-in is the positive control;
  - lifecycle: the child dies when its launcher is SIGKILLed, and an interrupted wait kills it;
  - routing: only a full anchor admission takes the bridge; stats-only and non-anchor checks keep
    the in-process verifier.
- **Launcher tests:** the anchor admission passes `full=True`, `historical=True`, and records the
  evidence in the authorities handed to the sweep; a bridge refusal stops before the lease; the
  manifest must carry the historical input-verifier pins (a changed verifier digest, another root
  or missing pins refuse).
- **Mutation battery v8, in a detached sandbox** (harness `mutation_v6/mutation_v8_frozen.py`,
  `8c2f0ca2…`; report `mutation_v6/mutation_v8/report.json`, `662475bf…`). **16/16 detected as
  declared** at `1f93db9`, after all 21 declared tests passed unmutated. The worktree's tracked
  bytes and `git status` were the same before and after, and no fixture process remained.
  - BX1–BX6: the verifier pin, the aggregate, the source records, the pre-child source
    measurement, the post-child source measurement and this tree's `val_split.py`, each disabled;
    the pre-child ones are detected by the no-child sentinel.
  - BX7–BX11: the exit code, the post-child verifier and seal checks, the report line and the
    expected-authority comparison. BX9 is declared as a reason change: with the seal re-check
    disabled, the authority's own seal digest still refuses.
  - BX12–BX13: no parent-death signal; no kill on an interrupted wait.
  - BX14–BX16: the call site not asking for the bridge, the dispatch disabled, and the manifest pin
    check disabled.
- **Audit checks of the same bytes** (§717: 14 private pre-child cases; §718: 50 bridge tests and 20
  launcher/manifest tests). The launcher `0f7081b9…` and the manifest script `18725bc1…` are those
  bytes. The two test files differ from the audit's capture only by two robustness edits made for
  the battery; reversing them reproduces the captured digests exactly:
  - `tests/test_anchor_confirm_input_bridge.py` (captured `1853b620…`, now `80c78c36…`): the
    interrupted-wait test kills the child in a `finally` if its assertion fails, so a surviving
    mutant leaves no stray process;
  - `tests/test_anchor_confirm_launcher.py` (captured `7c646a7f…`, now `704b5413…`): `seen[...]`
    became `seen.get(...)`, so a missing keyword fails the assertion instead of crashing.

## 8. Limits, stated

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
