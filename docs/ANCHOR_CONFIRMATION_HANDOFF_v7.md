# Anchor Confirmation — Handoff, generation v7 (response to audit §722–§723)

**Status: repair, regression tests, a smoke request and an admission-reuse proposal for audit
review. No scientific execution.**
- Since §722 nothing ran for real: no trainer, data or model load, GPU lease, full input
  verification, smoke or retry.
  - The diagnostic reproduction used private fixtures and a capture child.
  - The continuity check of the reuse proposal was stats-only metadata.
- Both failed attempts stay recorded as they are, with the same operations ledger and a cumulative
  charge of 73.88978339359164 s:
  - v5: `ancS5`, run `20260927T143532Z-4ada8a0d`, refused at the seal check;
  - v6: `ancS6`, run `20260928T052252Z-35772b95`, all four seals admitted, then every trainer refused
    on `library_environment`.
- The ancS6 reservation and snapshot are committed unchanged at `6b8d83d`. The §720 approval names
  request `1625b50f…` and approves nothing in this generation.

## 0. What this package is

- Read in full: §722 and §723. §723 answers my question: preparation proceeds without waiting for
  another decision, and smoke or S execution each need an exact fresh approval.
- Branch `arch-exp-2026-09-anchor-confirm`, source commit `59477d8be2c1750139ae436af778fd324c24d35a`.

  | Item | Path | SHA256 |
  |---|---|---|
  | Generation manifest v7 (62 files) | `artifacts/anchor_confirmation/authority_manifest_v7.json` | `c061296309fe41203205dff65de3a1afb843c950e93c0614dfd27a8216a90128` |
  | Contract v3, revised for v7 (manifest member) | `docs/ANCHOR_CONFIRMATION_CONTRACT_v3.md` | `e6c4978e6d30559e01c2ee34ae2b4d39c01717557ee16c9e96f85980aa8d3efc` |
  | Operational addendum `anchor-confirm-ops/5` | `docs/ANCHOR_CONFIRMATION_OPS_ADDENDUM_v5.md` | `3e2e96180067524dc041d1cf1daf867e9ad5a46a34d7823c341e09cbbca52842` |
  | Launcher (manifest member) | `scripts/phase3_selection_matrix.py` | `2cb8343dba923514d9c3fbc0ffffd79c66058d582c35486fab51a518ba541f53` |
  | Handoff tests (manifest member, new) | `tests/test_anchor_confirm_env_handoff.py` | `54790086c5d01d48b0729072d028d338f936c08e2894f94bc5c2198a17334b4d` |
  | Launcher tests (manifest member) | `tests/test_anchor_confirm_launcher.py` | `a542db5c097f5c377a7cb5a496adc8744d593b3cc62d24fe881784bd07d1147f` |
  | Runtime module (unchanged; §723's pin) | `dna_utils/runtime_environment.py` | `a78d51dbb134bcc400443a42b4c9fa46063b7bef356ab5b37b571a44f489a597` |
  | Supervisor (unchanged) | `scripts/anchor_confirm_supervisor.py` | `b5eecf58cbbc9d7d03463e4ea7a35ac107842aa6657a3b66e3770528cbdd1dd0` |
  | Stage-S plan | `artifacts/anchor_confirmation/plan_select_ancS7_v7.txt` | `5f9ce05518031fc72d9170de6320bcd0c38fbdf9375e0ce5ac9ee53b4c4ab9c8` |
  | **Smoke request, carried (proposed first)** | `request_preview_ancSmk7_v7_carried.txt` | **`27928f010fd4c2a09f64386eb94c96d4999c51f37af01b44a0a9961d97b780c8`** |
  | Smoke request, full (alternative) | `request_preview_ancSmk7_v7_full.txt` | `21834192e69515f2c7caa168edd3a7a1f4d3198d7d5e21ab93f6a9848a4ea1e7` |
  | **Stage-S request, carried (after the smoke)** | `request_preview_ancS7_v7_carried.txt` | **`bf2d3f56e0292020b361794d1b1238613ccc9b5998b4a84795d7dd756a7741e3`** |
  | Stage-S request, full (alternative) | `request_preview_ancS7_v7_full.txt` | `495e24d0c199345af802098fc3b08d86398a6bbdeffa4abe1aa548a5709349b3` |

## 1. §722.2 — diagnosis (item 2)

A bounded CPU-only reproduction (`artifacts/anchor_confirmation/env_handoff_v7/`, script `6a52490c…`,
report `diagnosis_59477d8.json` `5f6ad256…`). One driver, run in fresh processes with controlled
start-up blocks:
- **Parent start-up environment:** `caller_environment`, which the real parent fingerprint records.
- **Post-import environment:** after the real `import config`.
- **Environment passed to the child:** the real `build_command`.
- **Child-observed environment:** the pinned wrapper under bash starts a capture child. The child
  imports cv2, reads its own start-up block, and runs the real `verify_child_environment`.

| Tree | Start-up LD_LIBRARY_PATH | Passed to the child | Verdict (4 datasets each) |
|---|---|---|---|
| unrepaired (`6b8d83d`) | absent | `…/site-packages/cv2/../../lib64:` | refused ×4, exactly the real run's message |
| unrepaired | `/usr/local/cuda-12.4/lib64:/usr/local/cuda/extras/CUPTI/:` | the cv2 folder + that value | refused ×4 |
| repaired (`59477d8`) | absent | absent | admitted ×4 |
| repaired | the CUDA/CUPTI value | the same value | admitted ×4 |

- The only differing key in every unrepaired case is `LD_LIBRARY_PATH`. HF_HOME, HF_HUB_OFFLINE and
  TRANSFORMERS_OFFLINE were absent in both runs and here.
- These are reproductions under the same code and start-up values as the failed run, not forensic
  dumps of those four processes (§723.1). The real run's start-up block (the snapshot records the
  CUDA/CUPTI value) matches the second row.

## 2. §722.2 items 3–4, §723.2 — the repair and its tests

- **Repair (12 lines, `build_command`).** The four attested runtime variables come from the
  launcher's start-up block, the mapping the plan records (`caller_environment`). Any absent at
  start-up is removed from the child mapping, not left behind by an update.
- **Unchanged.** The trainer's check (`verify_child_environment`), the `library_environment` field,
  every other attestation field and guard, and the refusal before data or training. Nothing is
  suppressed or ignored.
- **Composed-boundary tests** (`tests/test_anchor_confirm_env_handoff.py`, 17):
  - unchanged inheritance with the variables set and unset, for every dataset;
  - parent import-time mutation (the real `config` import, asserted as a positive control) and the
    child's own cv2 import;
  - five genuine changed exec variables that must refuse;
  - the in-process start-up rule.

  The residual limit: only the library, `pythonpath` and interpreter fields are measured in the
  child. The GPU, torch-device and package fields need a real trainer, hence the smoke (§3).
- **A legacy test changed its premise.** `tests/test_phase3_selection_matrix.py::
  test_the_caller_environment_cannot_redefine_the_recipe` asserted that an in-memory `os.environ`
  value reaches the child, which was the defect. It now supplies the caller's value as the start-up
  block and checks that an in-memory rewrite does not leak. That file is not a closure member.
- **Evidence.**
  - Full 16-file run at `59477d8`: **921 passed, 1 skipped**, rc 0.
  - Mutation battery v9: **9/9 detected as declared**. EX9 is a control: disabling the unchanged
    trainer-side comparator proves the refusal cases are live. See addendum v5 §7 for the list.

## 3. §723.2 — the smoke, as its own exact request

- **What:** `--smoke --only flickr25k:4:anchors:42 --epochs 1`, one GPU, namespace `ancSmk7`, the
  same result root `/home/yschoi/gdna_anchor4_result`, under the supervisor as `stage-S-smoke`
  (1 planned cell, 2-h wall limit). The same ledger and caps apply; the planning charge is well under
  0.1 GPU-h.
- **What it proves:** the child starts under the plan's environment and passes the full trainer
  admission. That covers GPU identity, the torch-visible device, packages and the library
  environment. Then one epoch, validation and the record and receipt path.
- It is never evidence and never a candidate score. A passing smoke does not start S. S is a
  distinct request (`bf2d3f56…` or `495e24d0…`) needing its own approval.
- The launcher supports exactly this form. Its request records mode `smoke`, one executed cell of
  16 declared, `epochs` 1 and `gpu_count` 1.

## 4. §722.2 item 5, §723.2 — reusing the ancS6 full admission (a proposal)

- **The existing protocol's reuse.** `--admission-authority` names an earlier campaign's receipt or
  plan snapshot that completed a full admission. The approved request pins that file's digest, and
  the launcher checks it before and after parsing. Admission is then the stats-only recheck
  (`verify_seal_stats`) bound to the carried aggregates. The protocol states the one thing given up:
  a byte rewrite that preserves every stat.
- **Exact digest-bound authority.** The carried file is the ancS6 snapshot `ef5a3e8d…` (semantic
  digest `acaa0374…`). The "carried" requests pin it in `admission_authority`. It records the
  historical verifier's rc-0 full admission of the four seals `3e9bc7b9…`, `e44b363a…`,
  `6e9138a6…`, `aa1eaa5d…`, the same pins the new requests carry.
- **Made enforceable in v7.** `anchor_carried_admission_refusal` parses the snapshot at its approved
  bytes and requires, for exactly the carried seals:
  - rc 0;
  - the pinned verifier and root;
  - the same seal file digests, paths and aggregates;
  - the exact report line.

  A stats-only admission therefore cannot be carried on by another stats-only one, and a legacy or
  foreign snapshot cannot stand in. Tests cover every field, and battery v9 covers the guard.
- **Current continuity**, checked on 2026-09-28, metadata only:
  - `load_admission_authority` on the real snapshot passes (stats of every sealed file record,
    including the six historical sources, and the directory inventories);
  - the guard returns no refusal;
  - the four seal files hash to their pinned digests.

  It will be re-run by the launcher at admission.
- **New-generation binding.** The carried requests name manifest v7, and the launcher re-verifies
  the generation after admission, as always.
- **What this does not claim.** The ancS6 training failure does not invalidate its successful full
  check, and does not by itself grant reuse either. Reuse is proposed, and the "full" alternatives
  keep a complete re-verification available if the audit prefers it.

## 5. Requested decisions

1. Review generation v7: the handoff fix, the carried-admission guard, contract v3 as revised
   (§5, §7.7, §12, §14, §15 item 8) and addendum v5.
2. **Smoke:** a `stage-S-smoke` line naming manifest `c061296309fe41203205dff65de3a1afb843c950e93c0614dfd27a8216a90128` and request
   `27928f010fd4c2a09f64386eb94c96d4999c51f37af01b44a0a9961d97b780c8` (carried) or `21834192e69515f2c7caa168edd3a7a1f4d3198d7d5e21ab93f6a9848a4ea1e7` (full).
   Command: addendum v5 §6 step 1 (tmux `ancSmk7_v7`).
3. **Stage S**, later and separately, after the smoke is reviewed: a `stage-S-run` line naming the
   same manifest and request `bf2d3f56e0292020b361794d1b1238613ccc9b5998b4a84795d7dd756a7741e3` (carried) or
   `495e24d0c199345af802098fc3b08d86398a6bbdeffa4abe1aa548a5709349b3` (full). Command: addendum v5 §6 step 2 (tmux `ancS7_v7`).
4. D, the probes, L, R/T and everything downstream stay gated.

Ledger at submission: SHA256 `780b3fee036853b19b2221f4f74e18e28e15b617f16f61cc32a5aa116824e1db`, 55881 lines, last section §723.

## 6. Files changed from generation v6 (`git diff 6b8d83d 59477d8`)

- `scripts/phase3_selection_matrix.py`: the `build_command` repair, `anchor_carried_admission_refusal`
  and its call, and the closure member.
- `scripts/anchor_confirm_manifest.py`: generation v7, predecessors.
- `docs/ANCHOR_CONFIRMATION_CONTRACT_v3.md`: the v7 revision.
- Tests: the new handoff file, the launcher tests (guard; fixture for the byte-binding test), and the
  legacy caller-environment test.
- `dna_utils/runtime_environment.py`, `train_siglip2.py`, the wrappers, the seals and the historical
  tree are unchanged.
