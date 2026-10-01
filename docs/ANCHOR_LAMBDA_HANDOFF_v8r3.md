# Anchor model — generation v8: final pins after audits §735–§737 (stage L, Flickr25K first)

**Status: preparation only. No smoke, training, reduction, input rehash or GPU work ran.**
- The submitted package is still revision 2 (`docs/ANCHOR_LAMBDA_HANDOFF_v8r2.md`): manifest
  `58e69ae1…` and its four requests. §5 shows they are current at the final commit.
- This document adds the §735 fixture repair, the §736 bounded and guarded mutation harness, the
  final regression evidence, the parent/child read coverage and the final pins.
- Branch `arch-exp-2026-09-anchor-lambda`, final commit `2420d47` (pushed). The generation source
  (the 64 manifest members) is unchanged since `69c70fd`.

## 1. §735: synthetic legacy fixtures (`f8963b8`)

- **The repair.** `tests/test_phase3_selection_matrix.py` and `tests/test_phase3_select_n.py` (not
  closure members) gain an autouse fixture.
  - Every dataset spec points at a private synthetic feature cache (`meta.json`), a foil cache
    (`meta.json`), both whitening files and a caption file under a temporary directory.
  - `RunIdentity._artifact` and the plan-snapshot input digests are unchanged and hash those fixture
    files. No identity check is bypassed and no production path is allow-listed.
- **One exempt test.** `test_the_child_reads_the_provenance_caches` compares the production path
  *strings* of a built command and keeps the real specs. The guard shows it opens nothing.
- **Positive control (new, 3 cases).** `test_the_synthetic_inputs_are_hashed_by_the_production_identity`:
  appending bytes to the fixture whitening file, cache `meta.json` or caption file changes
  `expected_run_identity`'s digest.
- **Pins.**
  - `test_phase3_select_n.py` is `dad3a3a6…`, equal to the audit's §737 capture.
  - `test_phase3_selection_matrix.py` is now `9d06df17…`. The audit captured `7e404185…` before
    the positive-control cases were added (the only difference).
- **Retained failed evidence.** The first fixture attempt failed 16 path-string cases. That led to
  the named exemption above; no production path or identity check was changed.

## 2. §736: the mutation harness's read paths

- **The v10 and v11 defects** (harnesses `a304ec12`, `ea538639`). They are disclosed, not repaired
  retroactively.
  - `tree_state` hashed every git-tracked file, including the six unrelated `artifacts/umrch_*`
    `embeddings.npy` and `vision_adapter.npz` binaries, before and after each battery.
  - Their full sandbox checkouts also materialised those binaries.
  - Their `git status` checks may re-read racily-clean files.
- **v11's terminal state.** It completed at 15:12:58 KST: 26/26 detected, harness rc 0.
  - `report.json` `172f3625…`; logs retained in the worktree.
  - Its detections stand as control-flow evidence. They do not show that the harness made no
    unrelated reads (it did).
- **v12, the bounded and guarded harness** (`mutation_v12_frozen.py` `a7365986…`, with
  `bounded_tree.py` `2e719678…`):
  - **Content hashes** only for the reviewed inventory: the 64 manifest-closure files plus the
    declared test files, with source suffixes enforced.
  - **Every other tracked file** is compared by its git index object id and stat metadata, never
    read. That shows unchanged checkout membership, not working-tree byte equality.
  - **The clean check** is index-vs-HEAD plus inventory blob ids, in place of `git status`.
  - **The harness process** runs under an `open()` guard that refuses binary payloads, real-data
    roots and `os.exec`.
  - **Every declared test** runs through `guarded_pytest.py`, and a run counts only with zero
    refused opens.
  - **The sandbox** is a sparse checkout without `artifacts/`, under the scratch output directory.
- **v12 result at `f8963b8`** (`r3_evidence/battery12/report.json` `0a22871d…`):
  - **26/26 detected as declared**; 28 declared tests passed unmutated first;
  - 0 refused opens in the harness and in every pytest child;
  - inventory bytes and the other files' identity and stat unchanged; head-clean before and after;
  - no `artifacts/` in the sandbox; no stray process.

## 3. Regression evidence (`suite_pair_v2.sh` `170bff8b…` at `f8963b8`; it fails unless both runs pass with 0 refused opens)

| Run | Result |
|---|---|
| precondition: bounded head-clean (closure + 17 test files) | clean before and after |
| 17-file suite, unguarded | **1087 passed, 3 skipped**, rc 0 |
| 17-file suite under the `open()` guard (`guarded_pytest.py` `d987ed5c…`) | **1087 passed, 3 skipped**, rc 0, **0 refused opens** |
| verdict | `{unguarded_rc 0, guarded_rc 0, refused 0, head_clean_after 0}` |

- The 3 skips are the opt-in real-artifact tests.
- 1087 = the earlier 1084 + the 3 positive-control cases.
- The earlier paired run at `1469669` (guarded: 82 failures, 85 refused opens) stays recorded. Its
  wrapper returned the unguarded status; v2 fixes that.

## 4. Read coverage, parent and child

- **Parent processes (hooked).**
  - The pytest process of all 17 files: 0 refused opens.
  - The v12 harness: 0 refused opens.
  - The manifest inventory and the plan renders: the allow-list guard, which refuses `os.exec`.
    Their footprint is in r2 §5.
- **Child processes (not hooked; recorded and analysed).**
  - **The legacy files' Python children.** Five launcher / `phase3_select_n.py` commands: an invalid
    `--only`, `--plan --only cifar10:4`, `--refit`, `--run --refit --only`, and
    `--emit-stage-plan`.
    - Each was replayed under the blocking guard with the launcher bundle pre-set. Each gave the
      test's expected rc (2, 0, 2, 2, 0) and the expected refusal text.
    - 0 refused opens; their grandchildren were only `git rev-parse` and `git show`.
    - Evidence: `r3_evidence/child_replays/`.
  - **`bash` wrapper renders.** The launcher tests' actual-composition cases point every input at
    `tmp_path`. The plan renders read the pinned wrapper and test that the whitening file exists
    (r2 §5).
  - **`git` source queries.** Read-only.
  - **Python `-c` probes** in the environment-attestation tests. Synthetic.
  - **The v12 pytest children.** Each runs under its own guard.
- **What remains unhooked:** reads made inside child processes beyond these analysed commands.

## 5. Final pins and the current package

| Item | SHA256 |
|---|---|
| Manifest v8 r2 (closure unchanged since `69c70fd`; loads, and `verify_generation` passes at `2420d47`) | `58e69ae1a3bed5366da61a4d61a6cdd90c338d9ac7b0474fcf9b6e2e2f7f5bbc` |
| Smoke request, carried / full | `6e7827936e751fdd60f6350e77f5b635a6fab6e283aa81358f2fd4ec543a4c78` / `21e2e48a666fad31990ec6cf712a34a079dc159993a61d587868410f3f83030f` |
| Run request, carried / full | `daf99128fe014dea65958ee620697bc44d9bba27bfbc1863901b2ff8114cd70a` / `3505ec928cf2a70063d80b5a3d31bf9b936abf1196103d1935a81b041eba81b0` |
| New test file | `2700082f…` |
| Legacy test files | `9d06df17…`, `dad3a3a6…` |
| Guards | `guarded_pytest.py` `d987ed5c…`, `prep_guard/guarded_run.py` `c2062e65…` |
| Harness tools | `bounded_tree.py` `2e719678…`, `mutation_v12_frozen.py` `a7365986…`, `suite_pair_v2.sh` `170bff8b…` |

**Currency check at `2420d47`.** The four request previews were re-rendered under the allow-list guard
(launcher bundle pre-set, exec refused). All four came out **byte-identical** to the committed
revision-2 preview files, with 0 refused opens and only the named JSON files read. No new manifest
or request is needed, because no closure byte changed.

## 6. Retained disclosures

- **r1.** Its `--plan` renders were unguarded. The first guarded r2 attempt lost its hook at the
  launcher's self-exec; a later successful render does not guard that attempt.
- **Real-file reads by tests.**
  - The r1 binding test hashed real Flickr25K caption, cache-meta and whitening files.
  - The legacy suite files hashed real cache metadata, captions and whitening files in every
    unguarded run since v7, until `f8963b8`.
- **The v10 and v11 harnesses** hashed six unrelated tracked binaries.

None of these involved training, GPU work, model or checkpoint loading, or deserialisation. All
are preparation-path reads, now bounded or guarded.

## 7. Decisions requested

1. The §735 fixture repair and the §736 harness correction, with the evidence of §2–§4.
2. The current package: manifest `58e69ae1…`, and a `stage-L-smoke` line for `6e782793…` (carried)
   or `21e2e48a…` (full). This follows the final source, dependency, environment and carried-input
   review that §734.3 names.
3. Stage L (`daf99128…` / `3505ec92…`) and its reduction, later and separately.
