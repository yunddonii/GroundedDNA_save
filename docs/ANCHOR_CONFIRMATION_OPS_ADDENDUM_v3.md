# Anchor confirmation — operational addendum `anchor-confirm-ops/3`

Written for the audit (§709–§711). It states how the stage-S request of generation v5 (contract
v3, the fixed four-dataset anchor model) is supervised; the same rules carry to stage D and the
probes.
- It supersedes `anchor-confirm-ops/2` (`docs/ANCHOR_CONFIRMATION_OPS_ADDENDUM_v2.md`, `6bdd2460…`,
  kept as history) in: the GPU count (4), the planned cells (S 16, D 8, 12 probes), the roots (on
  `/home/yschoi`), the wall-time limits, the verification scope (four seals), the storage numbers,
  the pins, the commands and the evidence.
- The accounting rules, the watchdog, the stop mechanism and the unresolved/unclean settlement of
  ops/2 (audit §707–§709.4) are unchanged. The supervisor's code changed only in two constants: the
  default operations root and the per-stage GPU maximum (§3).
- Nothing runs until an audit-ledger approval line names the manifest and the exact request below.

| Item | Pin |
|---|---|
| Source generation | `2f79cb05354e3c5a290d4e26616dd7242d8c10dd` on `arch-exp-2026-09-anchor-confirm` |
| Generation manifest v5 (59 files) | `artifacts/anchor_confirmation/authority_manifest_v5.json`, SHA256 `84e94e2f1897e9910df2aa0352e4737f87dd485f018a644b5be23c1b4e036284` |
| Contract v3 (manifest member) | `docs/ANCHOR_CONFIRMATION_CONTRACT_v3.md`, SHA256 `bd99ab0d8810d1247cc1716246e736056592ce913f701393981b7408db626799` |
| Launcher (manifest member) | `scripts/phase3_selection_matrix.py`, SHA256 `abd902c650286d585b755196117becb9a345950674fa4e3b597bd58baf63e9e5` |
| Supervisor (manifest member) | `scripts/anchor_confirm_supervisor.py`, SHA256 `b5eecf58cbbc9d7d03463e4ea7a35ac107842aa6657a3b66e3770528cbdd1dd0` |
| Stage-S request | SHA256 `bd3117a9108ee558dfb27f9b429806a62092d23736febd7c0c0f8c858e4882c7` (`artifacts/anchor_confirmation/request_preview_ancS5_v5.txt`) |

Which component supplies each guarantee (unchanged from ops/2 except the numbers):

| Guarantee | Supplied by |
|---|---|
| Frozen interpreter and environment | the manifest's environment block, checked by the launcher (`load_anchor_manifest`) |
| Full input verification before any lease | the launcher, because the request carries no admission authority |
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
- The request carries no admission authority, so before any lease the launcher runs full
  verification of **four** stage-1 seals (`verify_campaign_input_seals(full=True)`): every sealed file
  re-hashed, the semantic row checks, the safe memory-mapped NPY/NPZ loads and the re-derived
  whitening statistics. It is CPU work with no model.
- Sealed target sizes, from each seal's inventory (sum of the sealed files' recorded sizes,
  deduplicated per seal; JSON only, no payload read, 2026-09-27):

  | Seal (`/data/yschoi/gdna_p3exec_seals/`) | Files | GB |
  |---|---:|---:|
  | `cifar10.stage1.input-seal.json` | 47 | 79.55 |
  | `flickr25k.stage1.input-seal.json` | 43 | 33.43 |
  | `nuswide.stage1.input-seal.json` | 43 | 257.65 |
  | `mscoco.stage1.input-seal.json` | 43 | 161.06 |
  | **Total** | 176 | **531.69** |

  The three multi-label figures equal audit §703.2's. Elapsed time is not measured; the historical
  four-dataset figure (about 82 min) is an estimate, not a bound.
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
  - The ops/2 ledger (`/data/yschoi/gdna_anchor_confirm_v1_ops/`) and result root
    (`/data/yschoi/gdna_anchor_confirm_v1_result`) do not exist (checked 2026-09-27). Nothing of
    generation v4 or earlier was ever executed or charged. The new ledger therefore starts at zero,
    and no earlier charge is dropped.
- **Unresolved, unclean, no final record.** Each blocks every later stage until the audit
  reconciles it (ops/2 §3).
- **Planning arithmetic (contract v3 §12), not a runtime promise.** S 3.50 GPU-h; D 0.47–3.73;
  probes ≈ 0.4; total ≈ 4.4–7.6 of the 12.5-GPU-h ceiling. Lease occupancy (reported, not charged)
  can reach about 4 GPUs × the slowest S stream (NUS-WIDE, about 81 min) ≈ 5.4 GPU-h.

## 4. Wall-time limits

These include CPU verification: stage-S-run 4 h, stage-D-run 4 h, each smoke 2 h, each probe 15 min.
The supervisor stops at limit − 130 s (the stop bound plus the watchdog bound).
- **S:** about 82 min of verification (estimate) + the slowest stream (NUS-WIDE, 75 epochs × 1.08
  min ≈ 81 min) ≈ 2.7 h. The limit is about 1.5× that.
- **D, worst case** (NUS-WIDE at N = 39): 82 min + 2 × 40 epochs × 1.08 min ≈ 2.8 h. The limit is
  about 1.4× that.

## 5. Stop mechanism

Unchanged from ops/2 §5: the triggers, the 10-s watchdog thread, one SIGTERM under the reaping
lock, the launcher's TERM → 5 s → SIGKILL for up to 30 s cleanup with leases released only after
every owned session is empty, `stop-overdue` after 120 s with no death inferred, and the post-exit
`unclean` (exit 4) / `unresolved` (exit 5) checks.

## 6. Exact commands

Run from `/data/yschoi/gdna_anchor_confirm_v1` with `GDNA_NUM_SEMANTIC_PARTS=5`. `<4 free GPUs>` is
chosen from `nvidia-smi` at admission; the request fixes the count (4), not the indices.

```
/data/yschoi/gdna_p3exec_authority/bin/tmux_run.sh ancS5_v5 \
  /home/yschoi/.conda/envs/dna_hashing/bin/python scripts/anchor_confirm_supervisor.py \
  --manifest artifacts/anchor_confirmation/authority_manifest_v5.json \
  --manifest-sha256 84e94e2f1897e9910df2aa0352e4737f87dd485f018a644b5be23c1b4e036284 \
  --stage stage-S-run --planned-cells 16 \
  --watch-path /home/yschoi/gdna_anchor4_result -- \
  /home/yschoi/.conda/envs/dna_hashing/bin/python scripts/phase3_selection_matrix.py \
  --anchor-confirm select --namespace ancS5 --anchor-arms anchors --run --gpus <4 free GPUs> \
  --result-root /home/yschoi/gdna_anchor4_result \
  --input-seal cifar10:stage1=/data/yschoi/gdna_p3exec_seals/cifar10.stage1.input-seal.json \
  --input-seal flickr25k:stage1=/data/yschoi/gdna_p3exec_seals/flickr25k.stage1.input-seal.json \
  --input-seal nuswide:stage1=/data/yschoi/gdna_p3exec_seals/nuswide.stage1.input-seal.json \
  --input-seal mscoco:stage1=/data/yschoi/gdna_p3exec_seals/mscoco.stage1.input-seal.json \
  --anchor-manifest artifacts/anchor_confirmation/authority_manifest_v5.json \
  --anchor-manifest-sha256 84e94e2f1897e9910df2aa0352e4737f87dd485f018a644b5be23c1b4e036284 \
  --anchor-approval-section <the approving ledger section>
```

The operations root defaults to `/home/yschoi/gdna_anchor4_ops`. Stage D and the probes use the same
supervisor:
- stage D: `--stage stage-D-run --planned-cells 8`, namespace `ancD5`, with the frozen N record;
- each probe: `--stage probe --planned-cells 0` in front of its `anchor_confirm_code_axis.py`
  command.

## 7. Test evidence

All CPU only (`CUDA_VISIBLE_DEVICES=` empty, `GDNA_NUM_SEMANTIC_PARTS=5`). No real GPU lease,
trainer, dataset or seal payload was used.

- **Full run at `2f79cb0`** (the manifest's source commit), the same 14 files as v4, tracked tree
  clean: **836 passed, 1 skipped** (the opt-in real-artifact test), 344 s, rc 0
  (`artifacts/anchor_confirmation/mutation_v5/suite_2f79cb0.log`). Every earlier commit of this
  generation also passed in full, each with 1 skipped and rc 0: `9891acb` 831, `4caa76e` 835,
  `b0b9d6e` 836 (logs kept beside it).

  | Suite | Tests |
  |---|---:|
  | `tests/test_anchor_confirm_port.py` | 16 |
  | `tests/test_anchor_confirm_recipe.py` | 69 |
  | `tests/test_anchor_confirm_launcher.py` | 175 (1 opt-in skip) |
  | `tests/test_anchor_confirm_reducer.py` | 183 |
  | `tests/test_anchor_confirm_lifecycle.py` | 11 |
  | `tests/test_anchor_confirm_supervisor.py` | 19 |
  | legacy and related (8 files, unchanged) | 364 |

- **Supervisor.** The 19 tests of ops/2 pass unchanged except the CLI test, which now takes four
  GPUs and refuses five ("1 to 4 GPUs").
- **Mutation battery v7b, in a detached sandbox** (harness `mutation_v5/mutation_v7b_frozen.py`,
  `8bc1bb6f…`). **15/15 detected as declared** at `b0b9d6e` (report
  `mutation_v5/mutation_v7b/report.json`, `54bf888f…`) and again at `2f79cb0` (report
  `mutation_v5/mutation_v7b_2f79cb0/report.json`, `3aa53ec4…`). Each time the worktree's tracked
  bytes and `git status` were the same before and after, and no fixture process remained. Before
  the first run, the marker line pytest uses for a missing refusal under Python 3.10 was
  calibrated on a throwaway test. The mutants:
  - MX1: a control arm let into the arm plan;
  - MX2: a plan without CIFAR-10, or with a control, accepted;
  - MX3: the GPU count need not equal the dataset streams;
  - MX4: the CIFAR-10 alternate literal not admitted;
  - MX5: the first-occurrence check disabled;
  - MX6: the reducer taking other arms;
  - MX7: a non-receipt record admitted;
  - MX8: the probe population ignored;
  - MX9: the 500 × 4 counts ignored;
  - MX10: a validation split short of 500 rows measured;
  - MX11: the v2 adoption rule re-introduced (a re-introduction, declared as such, not a
    disabled condition);
  - MX12: every validation row measured instead of the first 500;
  - MX13: the fixed-architecture declaration ignored in the replay;
  - MX14: an approval line of any version accepted;
  - MX15: the population left out of the probe request.
- **The first attempt (v7, at `4caa76e`; report `mutation_v5/mutation_v7/report.json`,
  `5f00c4b2…`) is kept.** It found 14/15 as declared. MX13 was killed for another reason: the
  test edited a copy of the frozen N record, so the stage-D approval of the original digest refused
  first. That test did not isolate the check. It now approves stage D and the probes for the edited
  record itself, and a positive control shows that an edit to an unchecked field still reduces
  (commit `b0b9d6e`). v7b differs from v7 only in MX13's declared line.
- **Between `b0b9d6e` and `2f79cb0`** only the contract (§711.3 correction), migration v2, one
  note string of the reducer and one assertion of its test changed.

## 8. Limits, stated

Unchanged from ops/2 §8: a `setsid()` descendant is not tracked; a launcher SIGKILLed from outside
leaves its trainers holding the inherited leases (reported `unclean`, never signalled); a process
surviving SIGKILL for more than 30 s keeps the leases; the allowance's one-child-per-cell premise
belongs to the pinned launcher; a frozen supervisor stops nothing while frozen; a killed supervisor
leaves no `final` record and blocks every later stage. New here: the result root and the
operations root share the root filesystem, so the ledger's own writes count against the same free
space the storage rule watches (a few kB per poll).
