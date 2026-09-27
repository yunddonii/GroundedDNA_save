# Anchor Confirmation — Handoff, generation v6 (response to audit §715–§719)

**Status: stage-S readiness package for audit review. No scientific execution.**
- Since §715 nothing ran for real: no full seal verification, lease, smoke, training, probe or
  refit, and no retry of stage S. The bridge tests used private fixtures only.
- The failed generation-v5 attempt is preserved as it is:
  - namespace `ancS5`, tmux record `ancS5_v5`, run `20260927T143532Z-4ada8a0d`;
  - operations ledger `/home/yschoi/gdna_anchor4_ops/device_budget_ledger.jsonl`, final record
    `exited`, rc 2, 0 attempts, 0 device seconds, 19.092536613345146 s charged.
- The v5 package stays as history: manifest `84e94e2f…`, request `bd3117a9…`, handoff v5
  `6dfcabd9…`, addendum v3 `f270229e…`, contract v3 at `bd99ab0d…`. The §713 approval names that
  request and approves nothing in this generation.

## 0. Acknowledgment and what this package is

- Read in full before this revision: §715, §716, §717, §718 and §719.
  - §715: the stop, the preserved charge, and the repair instructions answered in §1–§3.
  - §716: the missing pre-child source check, answered in §2.
  - §717, §718, §719: they verify the corrected paths and the committed evidence at these
    bytes. They add no new requirement.
- The scientific design is unchanged from generation v5: all four datasets, anchors fixed,
  S 16 / D 8 / 12 probes, the required pre-freeze lambda checks, the 12.5-GPU-h ceiling. Only the
  full input check and the namespaces change.
- **This revision.** Branch `arch-exp-2026-09-anchor-confirm`, source commit
  `1f93db92ade8fad60d6b963f13ff8016f8adf1b4`.

  | Item | Path | SHA256 |
  |---|---|---|
  | Generation manifest v6 (61 files) | `artifacts/anchor_confirmation/authority_manifest_v6.json` | `a5ff2a0e93acd3a0bef7ecc47a69d550b28fc7ee57c1a25aa48e7e1169f739d7` |
  | Contract v3, revised for v6 (manifest member) | `docs/ANCHOR_CONFIRMATION_CONTRACT_v3.md` | `a97ed217dca96fc44f91944951b868a1699ca95acb1cea128965349cdd81387b` |
  | Operational addendum `anchor-confirm-ops/4` | `docs/ANCHOR_CONFIRMATION_OPS_ADDENDUM_v4.md` | `b5184154c52212ea5deacfea2acc65ae4c7858c97ab84eeb8907039712895812` |
  | Launcher (manifest member; = §718's pin) | `scripts/phase3_selection_matrix.py` | `0f7081b9770735ebc22c8a7e8b02ce9fbd3d5792a367ce16d762dd22d8d142da` |
  | Manifest builder (manifest member; = §718's pin) | `scripts/anchor_confirm_manifest.py` | `18725bc1d207eae4ac78fe6efd89d01246e6e8711c8d56f7bd746c2e6f5cbbb1` |
  | Bridge tests (manifest member; = §719's pin) | `tests/test_anchor_confirm_input_bridge.py` | `80c78c36fbd016974e9bf143541c28074760f8de3ea0ad80375901b55674de17` |
  | Launcher tests (manifest member; = §719's pin) | `tests/test_anchor_confirm_launcher.py` | `704b5413860ee9d6f6893508741c34e6284829284589f9c176312acef5a37a5e` |
  | Supervisor (unchanged) | `scripts/anchor_confirm_supervisor.py` | `b5eecf58cbbc9d7d03463e4ea7a35ac107842aa6657a3b66e3770528cbdd1dd0` |
  | Historical verifier (pinned; unchanged) | `/data/yschoi/gdna_p3exec/scripts/seal_phase3_inputs.py` | `12233f8e4967c90afcab131a1061fe76abfbaed86648e1388826538a6d42f214` |
  | Stage-S plan | `artifacts/anchor_confirmation/plan_select_ancS6_v6.txt` | `7a445460a8a5d228c70913fd7b1c75c566c40207027e805dad516dd4e5128178` |
  | **Stage-S request** | `artifacts/anchor_confirmation/request_preview_ancS6_v6.txt` | **`1625b50faf74b311040a8f56f1d7103b7352e2733e8c8fe9b8860ed18a44793c`** |

## 1. §715.3 — the repair

- **Design.** A digest-bound isolated historical verifier (the first design §715.3 allows).
  `verify_seal_historically` in the launcher, used only for a FULL anchor-confirmation admission
  (`verify_campaign_input_seals(..., historical=True)` at the anchor call site). Contract v3 §5
  lists its seven steps; addendum v4 §1 describes its operation.
- **Separation of provenance and execution.** The historical verifier runs in the historical tree
  as `python -I -B /data/yschoi/gdna_p3exec/scripts/seal_phase3_inputs.py verify --seal PATH`. Its
  own REPO is the tree that sealed the inputs, so the legacy full check (payload re-hash, row checks,
  whitening re-derivation, exact comparison) runs unchanged. The new generation's own checks stay
  in the launcher: the manifest recheck after admission, this tree's `val_split.py` against the
  sealed content, and the request's seal pins (`assert_request_seals`).
- **Authentication.**
  - The root, the verifier digest and the six source digests are constants of the pinned launcher,
    carried in the manifest (`historical.historical_input_verifier`, required by the loader). They
    are never caller input.
  - The child is a yes/no oracle. Its exit code and exact report line are checked, and the input
    authority is derived in the launcher from the seal bytes it hashed before the child ran.
  - The verifier bytes, the seal bytes and the six source observations are re-checked after the
    child.
- **Preserved.** The four seals and their digests (request pins `3e9bc7b9…`, `e44b363a…`,
  `aa1eaa5d…`, `6e9138a6…`, unchanged from v5), the historical tree and its verifier, the legacy
  comparison, stats-only rechecks after admission, the trainer's input admission, every non-anchor
  check and every pre-lease refusal boundary.
  - No cache is rebuilt, no link repointed, no seal path or stat rewritten, no refusal caught and
    ignored, and admission is not stats-only.
  - The failed run is not relabelled as input authority.
- **All six sources**, not only the first reported difference: five caption-foil producers and
  `val_split.py`, in both `producer_sources`/`protocol_sources` and the split identity's
  `protocol_source`.

## 2. §716 — the six historical files are measured before any child

- Before the child, `_historical_source_observation` measures each of the six files with the seal's
  own record builder (`_seal_file`: content digest, lstat identity, file type). It requires each to
  equal the seal's record exactly. A mismatch refuses with no child started. So `val_split.py` is
  authenticated before the historical verifier executes it, and no payload re-hash is spent on
  known-bad provenance.
- The same measurement runs again after the child. A change across the handoff refuses, and the
  evidence records the observed records.
- The distinct new-generation `val_split.py` check and the legacy full comparison are kept.
- Tests assert that no child was created (a `Popen` sentinel): for each of the six files a content
  change and a same-content timestamp change, plus the other pre-child refusals. §717 verified 14
  such cases independently.

## 3. §715.3 — the requested tests (all in `tests/test_anchor_confirm_input_bridge.py`)

They use real small seals, built with a byte copy of the verifier in a temporary "historical" tree
and checked from this tree. The groups below are listed in full in addendum v4 §7:
- the relocated-worktree case reproduced, and the same-root legacy check still passing;
- altered historical producers;
- a changed new-generation `val_split.py`;
- substituted roots and seals;
- an unpinned verifier;
- forged and stale reports, and a stale expected authority;
- mutation across the handoff (seal, verifier, sources);
- child cleanup: SIGKILL of the launcher, and an interrupted wait;
- routing (stats-only and non-anchor unchanged).

Launcher tests cover the caller: `full=True`, `historical=True`, evidence in the authorities; a
refusal before the lease; the manifest pins.

## 4. Evidence (CPU only)

- **Full run at `1f93db9`**, 15 files, tracked tree clean: **891 passed, 1 skipped**, rc 0, 364 s.
- **Mutation battery v8:** **16/16 detected as declared** (21 baseline tests passed unmutated;
  tree and status unchanged; no stray process). BX9 is a refusal-reason change, as §719 notes:
  the authority's own seal digest still refuses.
- **Independent audit checks:** §717 (14 private pre-child cases), §718 (50 bridge tests, 20
  launcher and manifest tests), §719 (the 21 mutant logs compared with their declarations, the
  final test delta replayed, and the 24 real small-source records matched).
- **Test files versus the audit's §718 capture.** The two test files carry two robustness edits made
  for the battery. Reversing them reproduces §718's captured digests exactly, and §719 pins the
  final bytes. The launcher and the manifest builder are §718's bytes.
- **Request.** The printed canonical request recomputes to `1625b50f…`. It differs from the v5
  request only in `manifest` and `namespace` (`ancS6`): 16 cells (4 datasets × N 4/9/19/39,
  seed 42, anchors only), 4 GPUs, the four seal pins, the result root
  `/home/yschoi/gdna_anchor4_result`.
- **Plan.** It admits all 16 coordinates, each anchor cell differing from its rendered control in
  `axis_center` alone.
- **Manifest.** 61 files: the v5 closure plus the bridge test file and the seal test file it
  borrows its fixture from. Every recorded digest equals the bytes at `1f93db9`, and the historical
  verifier pins were checked on disk when it was built.

## 5. For the eventual S report (§715.3)

- The N selection reducer admits the JSON receipts, records and `log.csv` rows first. It then
  reads and hashes each record's `config.pt` bytes against their pin, without deserialising them.
  It is not a JSON/log-only step, and contract v3 §11 and §7.7 now say so.
- D planning, which replays the N record across that real-binary read, stays gated.

## 6. Requested decision and scope

- Review generation v6: contract v3 as revised (§5, §11, §12, §14, §15 item 7), addendum v4, and
  the stage-S request.
- A stage-S approval, if given, is a ledger line
  `ANCHOR-CONFIRM-APPROVAL version=anchor-confirm/2 scope=stage-S-run manifest=a5ff2a0e93acd3a0bef7ecc47a69d550b28fc7ee57c1a25aa48e7e1169f739d7 request=1625b50faf74b311040a8f56f1d7103b7352e2733e8c8fe9b8860ed18a44793c`.
- The command is addendum v4 §6 (tmux `ancS6_v6`), with the same operations ledger and the
  19.092536613345146-s prior charge.
- D, the probes, L, R/T and everything downstream stay gated.
- Ledger at submission: SHA256 `f27b20911be75cf246380f67c0eaa7471e8f8f747b27ce4862677178fc5818d0`, 55505 lines, last section §719.

## 7. Real files touched in this revision (nothing deserialised)

- The manifest inventory: the two pinned JSON authorities, the `p3lamA` receipt bytes, the closure
  files, and the historical verifier and six historical sources (small files, digest-checked).
- The stage-S `--plan` and the request preview (the four seal JSON files by digest; no payload).
- The bridge tests read only private fixtures in temporary directories.
