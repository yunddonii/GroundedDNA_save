# Anchor model — r8 stage-T recovery: source, tests, mutation evidence and the exact request (audits 797–822)

**NON-EXECUTABLE.**
- Nothing below has run against real data.
- No claim, budget or GPU was used.
- Execution needs two things: the user's decision on the added 15,000 device-s budget and the one-attempt
  exception, then an audit line of scope `stage-T-recovery` naming the request digest in §6.

**Revision (audit 822).** First submitted at `4a5f5fe`. This revision changes only records and wording:
- §5: the 113 per-execution boundary records of the battery are now archived (`battery_guard_3fd0117/`).
- §6: the read footprint is given by format, and the guard's parent-only limit is stated.
- §7 and §9: the launch is bound to the exact audited submission HEAD, with `3fd0117` kept as the tested identity.

No source, test, declaration, harness, runner, helper or manifest changed; nothing was rerun.

## 1. Receipt

- **§797–§821 were read in full** (ack 821).
- **§797** accepted option A for bounded preparation, with corrections in items 1–6.
- **§800–§801** required the final carry, lineage and new-chain closure before publication.
- **§802–§808** found successive gaps in the Python-hook test guard (v1–v3). That guard is retired
  (kept unchanged as failed history in `t_recovery_prep/`), replaced by the OS boundary that §810
  verified independently.
- **§811–§812** required a test-fixture repair of the four launcher e2e tests instead of deselecting them.
  The repair is done (§4).
- **§813** verified the request metadata (43 named files and the whole stopped-T ledger).
- **§814–§818** reviewed the repair commit and the declaration corrections.
- **§819** accepted the final suite within its scope and recorded the skip limits.
- **§820** queued the D5/D6 diagnostic item, which this submission does not touch.
- **§821** accepted the fresh battery within its scope, keeping RY26's refusal-ordering limitation.
- **§822** verified the source bridge, the 84 archive members and the request. It required three
  archive and wording corrections, which this revision makes (§5, §6, §7, §9).

## 2. Source identity

| Item | Value |
|---|---|
| Branch / tree | `arch-exp-2026-09-anchor-refit-r8`, `/data/yschoi/gdna_anchor_refit_v9r8` |
| Implementation commit | `0b3eed7` |
| Generation manifest | `authority_manifest_v9r8.json` SHA256 `3537e297c0429c8ab5b02428651025903765a48c3c126eea172f30c37955c9f8`, 71 members, built at `0b3eed7`, committed in `efd6c31` |
| Transition from r7 (`2f24fc80…`) | changed exactly 7 (`scripts/anchor_refit_stage.py`, `scripts/anchor_terminal_test.py`, `scripts/phase3_selection_matrix.py`, `scripts/anchor_confirm_manifest.py`, `scripts/anchor_confirm_supervisor.py`, `tests/test_anchor_confirm_launcher.py`, `tests/test_anchor_refit_stage.py`), added exactly 2 (`docs/ANCHOR_T_RECOVERY_CONTRACT_v1.md`, `tests/test_anchor_t_recovery.py`), removed 0 |
| Boundary allowlist (stage-R snapshot sources) | exactly 3 control-plane files: `anchor_refit_stage.py` `3ddaa422…`, `anchor_terminal_test.py` `99068bab…`, `phase3_selection_matrix.py` `d8092436…` |
| Scientific sources | unchanged: extraction, evaluation, terminal test, train extraction, BIO, NMI, seal, trainer, model, loss, data, recipe, runtime state |
| Test-only change outside the closure (§811) | `tests/test_phase3_launcher_e2e.py` at `3fd0117`, SHA256 `a74594dd…` (§4); `mutants_v14.py` `ae06e17c…` (declarations, §5) |
| Pinned lineage | `anchor_t_recovery_lineage_v1.json` `b7676019…` (§799 verified its 37 records and the settled-ledger pins) |

## 3. How each requirement is met

| Requirement | Implementation | Pinned by tests (`tests/test_anchor_t_recovery.py`) |
|---|---|---|
| 797.2(1): historical r7 versus executing r8 | `historical_generation`, `generation_transition` (exact changed/added/removed sets); `recovery_boundary` (exactly the 3-source allowlist; r7 bytes at stage R and committed r8 bytes now; all other source authority unchanged) | transition admitted / extra-change / missing-change / removed / same-generation; boundary allowlist cases; plan refuses an undeclared transition |
| 797.2(2): one exact cell plus 11 carried; full lineage pins | `verify_stopped_lineage` (snapshot, request, reservation, §788 line re-verified, receipt absent, settlement, exact membership, every carried record/attempt/entry, the recovered cell's attempt/entry, no foreign-namespace attempt); `admit_stopped_campaign` adds the pre-execution target absence | membership (missing / duplicate / foreign seed / crossed tag); stage-T record present; foreign attempt; output present; receipt present; carried run directory off contract; record / nonce / attempt / entry bindings; settlement (unclean, orphan, charge, held lease, wall stop, line after final, line pin); reservation; recorded line |
| 797.2(3): durable, race-safe claim; the entry verifies the authority itself | `consume_recovery_claim` (O_EXCL, lineage key only, root outside every worktree, before the attempt); `verify_recovery_authority` in the T entry (re-admits the lineage, checks the request's lineage part, checks the claim fields) before any configuration, model or test access | concurrent namespaces claim once; same and other namespace after success or failure; alternate record root; claim-root mismatch; entry without claim / other namespace / other root / run-scope line / mixed form / counterfeit lineage / second entry |
| 797.2(4): budget shape (i) | supervisor `stage-T-recovery`, own ledger `/home/yschoi/gdna_anchorRTrec_ops`, budget 15,000; `parent_settlement` binds the settled R/T ledger digest `31691514…` and 79,482.14623009507 s; the final event carries the cumulative total including the parent | own ledger and budget; 1 GPU and 5 producers; parent binding and carry-forward; changed / reformatted / unsettled / other-charge parent refuses; sixth producer is excess; budget binds |
| 797.2(5): wording | the contract describes recovery of a predetermined missing cell after a resource failure, not "no test information" | — |
| 797.2(6): I/O estimates | `full_t_io_observations.json`: four bounded samples (16:32–16:36 UTC) with the §799 limits | — |
| 800: final carry/lineage closure | `final_recovery_closure` re-runs the lineage with the target executed, re-hashes the carried payloads, re-checks the claim bytes | nine drift-during-the-cell cases; receipt binds the verified identities |
| 801: the new cell's own chain | the closure re-reads the new record, then the attempt at the digest the record names (name, root, schema, namespace, nonce, request, approval, cell), the entry claim (attempt digest, cell, pins) and the 12 outputs | new attempt changed / missing; new entry missing / mismatched; positive with the genuine synthetic entry claim |

## 4. Tests, inside the OS boundary

**Boundary** (`t_recovery_prep/sandboxed_pytest_r8.py`; §810 verified it independently):
- bubblewrap with new user, mount, PID, network, IPC and UTS namespaces;
- only `/usr`, `/etc`, the conda environment and the tree under test exist, all read-only; the tree's data
  directories are empty mounts;
- a source-only git store holds HEAD, `3dd1c02` and `88c3a25`, with 620 data/payload blobs left out;
- `/tmp` is private;
- no live data, result, cache, ledger, claim or lease path; no network; no GPU device.

A probe in a separate launch of the same configuration confirms all of this. It also records that the
launcher, stage, entry, supervisor and manifest modules import from inside the tree, with their digests.

**Controls** (`guard_controls/test_sandbox_v4_controls.py`, run inside): 32/32.
- Live paths are absent; tree data paths are empty; the tree is read-only.
- A bare relative payload name resolves to nothing; an alias in `/tmp` dangles.
- There is no network and no GPU.
- Children of every form see the same boundary: plain, `-I`, minimal environment, `-S -E`, and a copied
  shell.
- git serves source but not the payload blob.
- A live-looking write lands only in the sandbox; the host claim root remains absent.

**§811 fixture repair** (`tests/test_phase3_launcher_e2e.py`, test-only, outside the closure):
- The stub trainer derives its one-device torch observation from the GPU inventory row that
  `CUDA_VISIBLE_DEVICES` names. That is the same synthetic inventory the parent fingerprints, never copied
  from the plan.
- The real child attestation still compares every field.
- New test `test_a_child_device_observation_off_the_plan_refuses_before_publication`: a disagreeing
  observation refuses before any record is published.
- A focused pre-commit run passed 6/6; all 6 pass again inside the final suite at `3fd0117`.

**Final pinned suite at `3fd0117`** (tree clean; runner `a684299c…`):
- **Result:** 1556 passed, 11 skipped, 2 subtests passed, rc 0. All 28 files, nothing deselected.
- **Module identities:** all five modules were imported from inside the tree at their manifest digests.
- **Skip reasons** (`-ra`; coverage limits, not passes; §819):

| Count | Where | Reason |
|---|---|---|
| 1 | `test_anchor_confirm_launcher.py:1336` | opt-in real-artifact plan rendering |
| 2 | `test_anchor_lambda_stage.py:739`, `:1234` | opt-in real wrapper; opt-in real v7 history |
| 1 | `test_anchor_refit_stage.py:1611` | opt-in real pinned wrapper and whitening path |
| 1 | `test_extraction_run_validation.py:284` | Phase-2 snapshot not present inside the data-masked boundary |
| 5 | `test_phase2_inventory.py` | no validating Phase-2 cell to build a fixture from |
| 1 | `test_phase2_launcher_wiring.py:195` | Phase-2 diagnostic root not present |

- **The superseded partial run at `efd6c31`** (1551 passed, 4 deselected) is kept as history, not as final
  coverage.
- **Evidence:** `refit_v9/t_recovery_final/final_3fd0117/{suite,controls}`.

## 5. Mutation battery v14 at `3fd0117`

- **Setup:** 47 mutants (RY1–RY47), each disabling one whole condition or step in the recovery code.
- **Baseline:** every declared test first passed unmutated inside the sandbox over the sparse copy.
- **Each run:** confirmed the sandbox boundary, the tree HEAD and that the imported modules came from the
  mutated copy (§807).
- **Result at `3fd0117`** (fresh run, rc 0, 2,353 s):
  - baseline: 54/54 passed;
  - mutants: **47/47 detected as declared**, over 59 declared case executions;
  - sandbox restored; inventory and other tracked identities unchanged; clean before and after;
  - harness refused 0; no stray process.

  The harness is `mutation_v14.py` `32d6040d…`; the declarations are `mutants_v14.py` `ae06e17c…`.
  Evidence is in `refit_v9/t_recovery_final/final_3fd0117/battery`, including all logs, with SHA256SUMS.
- **Per-execution boundary records (§822).** The battery kept one record directory for each of its 113
  executions (54 baseline + 59 mutant cases) in a temporary root.
  - **What is archived:** for each execution, its five named files: `result.json`, `probe.json`,
    `bwrap.json`, `git_store.json` and `pytest.log`, copied byte for byte into
    `t_recovery_final/battery_guard_3fd0117/<counter>/`.
  - **What is left out, by name:** the sandbox git object stores, the private `/tmp` and the worktree git
    directories.
  - **Archiver:** `t_recovery_final/archive_battery_guard_records.py` (`5e73dde3…`).
  - **Hashes:** `INDEX.json` `56d4f335…` and `SHA256SUMS` `da3b3411…`, listing the 566 other files.
  - **Ignored files:** the `pytest.log` files there are tracked despite the `*.log` ignore rule.
  - **Counter order** follows `mutation_v14.main()`: counters 1–54 are the baseline tests in report order,
    and counters 55–113 are the mutant cases in order. The archiver checks every record against that order:
    - the run's own report is byte-identical to the archived one;
    - each record names HEAD `3fd0117` and runner `a684299c…`;
    - each record's rc equals the report's rc;
    - the boundary probe shows no network, no GPU device, a read-only tree and the live roots absent;
    - five module identities are inside the sandbox, at the `3fd0117` bytes, except exactly the one mutated
      file of a mutant case;
    - a baseline tree is clean and a mutant tree is not;
    - each mutant case's archived log ends with that execution's `pytest.log`;
    - record times are in counter order.

    Result: 113 records, 0 problems.
  - **Positive control** (`battery_guard_3fd0117_control/control_report.json`): a scratch copy with five
    tamperings (a module digest, HEAD, an extra directory, a log tail, an rc). The archiver reported every
    one and exited 1.
  - **Limit:** a baseline record's content cannot tell which of the 54 tests it ran (each says `1 passed`).
    Its test name rests on the counter order alone, supported by the record times.
  - **Not archived:** the superseded `efd6c31` battery's per-execution records. Its aggregate and logs
    stay in `battery_efd6c31/`.
- **History:** the first run at `efd6c31` ended rc 1 (§817).
  - It detected 41/47 as declared; 51/59 cases matched their declared diagnostic.
  - The other six were wrong declared markers, not missed defects (§815–§816).
  - That run is archived unchanged with all 63 files hashed, in `t_recovery_final/battery_efd6c31`.
  - The corrections:
    - RY4 → `DID NOT RAISE`;
    - RY23–25 and RY27 → pytest's parenthesised and-chain comparisons;
    - RY26 → the later approval refusal `no standing stage-T approval for this request`, i.e. detection
      through refusal ordering only. Removing that guard alone does not admit the mixed request in this
      fixture.
- **Two conditions are not mutated:** both mutants would be equivalent.
  - The explicit pre-execution absence check is redundant with the stage-R admission's own withheld check
    for the non-carried cell.
  - The T entry's scope expression yields the same string for mode `recovery`.

## 6. The exact request (metadata-only render)

- **Render:** under the named-open guard, the launcher's own `--plan` with
  `--anchor-confirm recover --namespace ancT9r --run --gpus 0`, r8 manifest `3537e297…`.
- **Reads:** exactly the 84 allow-listed files, opened by name, plus separately allowed source reads.
  - **82 JSON files:** lineage, both manifests, stage-R receipt/snapshot/records, the 12 run directories'
    campaign binding and runtime sidecar, stopped snapshot/reservation/attempts/entries/records, F and its
    v7 snapshot, and the approved aggregates.
  - **1 Markdown file:** the audit ledger.
  - **1 JSONL file:** the settled R/T device-budget ledger.
- **Other evidence:** output presence by `stat` only; 206 git children (104 `rev-parse`, 102 `show`, all
  on named source); 0 refused.
- **Limit of this guard:**
  - The named-open guard is a Python audit hook in the rendering (parent) process only. It refuses and logs
    that process's opens. For children it records only their argv; it does not enforce their opens.
  - So the 84 named reads are the observed footprint of the parent. The children are known only by their
    command lines: git `rev-parse` and `show` of named source paths.
  - This is not a measured child-I/O footprint and not a general subprocess isolation guarantee.
- **Request SHA256: `d34f505b90db1ef02dce74bd0326350053a6708315f505c0c881fe89a91ad666`** (re-rendered at `3fd0117`:
  identical request, 0 refused; evidence in `final_3fd0117/render`).
- **What it names:** schema `anchor-terminal-test-recovery-request/1`, mode `recovery`; one cell (NUS-WIDE
  seed 44, N 4, terminal epoch 4); `gpu_count` 1; namespace `ancT9r`; record root
  `/data/yschoi/gdna_anchor_refit_v9r8/artifacts/anchor_confirmation`; claim key `dc4a8bc0…` under
  `/home/yschoi/gdna_anchorRT_recovery_claims`; the 11 carried pins; `recovery_of` the spent attempt
  `d626f950…` and entry `7689d689…`; the stopped settlement pins; the r7→r8 transition.

The approval this request would need:

```
ANCHOR-CONFIRM-APPROVAL version=anchor-confirm/2 scope=stage-T-recovery manifest=3537e297c0429c8ab5b02428651025903765a48c3c126eea172f30c37955c9f8 freeze=5165f5dc9fcfb8334270bc16aa9816d09db67b03a04abae7ff846d5235bdca1d request=d34f505b90db1ef02dce74bd0326350053a6708315f505c0c881fe89a91ad666
```

## 7. Command, resources and GPU coordination (after the decision and approval only)

```
env -C /data/yschoi/gdna_anchor_refit_v9r8 -u PYTHONPATH -u CUDA_VISIBLE_DEVICES GDNA_NUM_SEMANTIC_PARTS=5 \
  /home/yschoi/.conda/envs/dna_hashing/bin/python scripts/anchor_confirm_supervisor.py \
  --manifest /data/yschoi/gdna_anchor_refit_v9r8/artifacts/anchor_confirmation/authority_manifest_v9r8.json \
  --manifest-sha256 3537e297c0429c8ab5b02428651025903765a48c3c126eea172f30c37955c9f8 \
  --stage stage-T-recovery --planned-cells 1 --ops-root /home/yschoi/gdna_anchorRTrec_ops --watch-path /home/yschoi -- \
  /home/yschoi/.conda/envs/dna_hashing/bin/python scripts/phase3_selection_matrix.py --anchor-confirm recover \
  --namespace ancT9r --anchor-manifest /data/yschoi/gdna_anchor_refit_v9r8/artifacts/anchor_confirmation/authority_manifest_v9r8.json \
  --anchor-manifest-sha256 3537e297c0429c8ab5b02428651025903765a48c3c126eea172f30c37955c9f8 \
  --run --gpus <G> --anchor-approval-section <SECTION>
```

Run in tmux `ancT9r_v9r8`, one attempt.

| Item | Plan |
|---|---|
| Budget | shape (i): 15,000 device-s in the separate recovery ledger; cumulative maximum over R, T and the recovery 94,482.14623009507 s |
| Limits | 1 logical cell, 1 GPU, 5 producers, 8 h wall, 130 s headroom, 5 × max-window allowance |
| Duration (estimate only) | NUS-WIDE cells took 14,627 s (beside three other streams) and 6,353 s (beside two). No completion guarantee (§799) |
| GPU | one physical GPU chosen at launch. Before the launch, agree with the text-path session (§796) the index and UUID, ownership, duration (≤ 8 h) and release at settlement, with a written receipt. All six GPUs were idle when this was written; nothing is reserved |
| `/data` | at launch, measure `/data` read load (`/proc/diskstats`, per-process I/O). If another session reads heavily, report the conflict for coordination and wait; never stop or slow a peer |
| Same-command gates | tree clean at the exact audited submission HEAD `S` (the commit at which the audit verifies this revision; `3fd0117` stays the tested identity, and no reset to it); `git diff --name-status 3fd0117 S` lists only additions (`A`), all within `docs/ANCHOR_T_RECOVERY_SUBMISSION_v9r8.md` and `artifacts/anchor_confirmation/refit_v9/t_recovery_final/`; all 71 manifest members re-hashed equal to their pins; manifest digest; tmux absent; the chosen GPU idle by UUID; leases via `lslocks` only; no `ancT9r_*` record and no recovery claim yet; the settled R/T ledger at `31691514…`; the recovered run directory still without outputs |
| Failure rule | one attempt. On any refusal, drift, producer failure or resource stop: preserve every claim, record and partial output, settle, and return. No retry |

## 8. Limits

- The tests prove the admission and publication logic on synthetic worlds inside the OS boundary. The
  real payload checks happen only inside the approved, supervised run: re-hashing the 11 carried cells'
  outputs, and the T entry's checkpoint and config binding.
- The probe and pytest are separate launches of the same sandbox configuration, not one persistent
  namespace.
- The superseded Python-hook guard's runs (`suite_dev1`, g-series, `suite_v3a`) are failed development
  history and are not counted.

## 9. Bridge from this submission commit to the tested commit `3fd0117`

All tests, controls, the battery and the render above ran at `3fd0117` (clean).

**What the submission commits add:** both `4a5f5fe` and the §822 revision add only two kinds of path:
- this document, `docs/ANCHOR_T_RECOVERY_SUBMISSION_v9r8.md`;
- evidence files under `artifacts/anchor_confirmation/refit_v9/t_recovery_final/`.

The revision adds the archiver, `battery_guard_3fd0117/` and `battery_guard_3fd0117_control/`.
Relative to `3fd0117`, every path is an addition.

**What it leaves unchanged** (check with `git diff --stat 3fd0117 <submission commit>`):
- production sources;
- tests and the e2e fixture;
- mutation declarations and harness;
- the OS runner;
- the bounded-tree helper;
- the manifest and the environment.

`git diff --stat` lists only the two paths above. The results are therefore bound to `3fd0117`, and
none is relabelled as a run at the submission commit.

**What is tracked and what is not:**
- **Tracked:** reports, results, probes, commands, statuses and the SHA256SUMS lists. Also the 113
  `pytest.log` files under `battery_guard_3fd0117/`, force-added for the §822 archive.
- **Ignored by `.gitignore` but kept on disk:** the `*.log` files (pytest logs, battery case logs,
  consoles). Their identities are in the tracked SHA256SUMS of each archive folder.
