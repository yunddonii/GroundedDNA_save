# Anchor Confirmation v1 — Handoff, generation v2 (revision response to audit §665–§701)

**Status: preparation package for audit review. No scientific execution.** No training, smoke,
probe, refit or terminal evaluation, no GPU lease, namespace reservation or run-directory claim,
and no deserialisation of any real binary happened for this revision. The operations that did touch
real files are listed in §4. The v1 package (handoff `a46db3d5…`, contract `418091ee…`, manifest
`c1eed986…`) is unchanged and stays as history.

## 0. Acknowledgment (§678, §684, §686, §693)

- Read in full before this submission: §665–§701 (§679–§701 were appended while this revision was
  being prepared; ledger mtime 2026-09-26 18:48:41 KST at the last reading). §693's instruction is
  acknowledged: this is the coherent readiness submission it asks for, and it stops here.
- **This revision:** branch `arch-exp-2026-09-anchor-confirm`, source commit
  `491c3ce84ec0677f42c7bdb7ed9c136f6bece4c7`; generation manifest
  `artifacts/anchor_confirmation/authority_manifest_v2.json`, SHA256
  `a5f7b8436e3e81b189deedb4d1f3e99eb993bce59d8792bada12f7c1a6a1bf0a`; contract
  `docs/ANCHOR_CONFIRMATION_CONTRACT_v2.md`, SHA256
  `26ebe2c104e89702bbc9daef0a3470d123ef6f7c5c30eefc49222cb26e29b001`.
- **Outstanding, not closed by this revision:** (1) the audit's decisions of contract §15 — fresh
  controls, the override policy, the approval-line and request format, the §8.2 endpoint, and the
  separately scoped input/environment admission for stage S; (2) an approval line for the exact
  stage-S request of §5 (none exists; every execution path refuses without it); (3) stage D, the
  probes, R and T, each later; (4) stage-R commands and the historical `config.pt` inspection are
  not part of this generation; (5) the probe's post-load runtime-row check is reachable only with
  real data (stated in §3, not mutated).

## 1. What changed since v1, in one paragraph

Both arms train fresh in stages S and D (24 + 12 = 36 cells); the v1 reuse proposal is withdrawn
and the reducer accepts reuse only through a digest listed in reviewed source (none). The repeat
rule is an exact wrapper-literal-then-one-override policy, checked on the real pinned wrappers. One
admission enforces the contract's typed protocol values (including `hash_target_mode siglip_cos`
and the text path) for both arms at every coordinate, in `--plan`, `--smoke` and `--run`. Every
execution and every probe needs a line in the audit ledger naming the generation manifest and the
SHA256 of the exact canonical request (cells, namespace, roots, smoke horizon, seals, authority,
GPU count; §689.2); `--plan` with the execution arguments prints that digest; the snapshot and
receipt carry the request and approval; the reducer re-verifies both for every record. The reducer
never deserialises: `config.pt` is checked by byte pin against the completed in-campaign recipe
check. The probe verifies the approval, the frozen N record, the record chain and byte pins before
its first load, and the trainer's own input admission before any model. The manifest must list the
exact generation closure (56 files, wrappers and tests included) and the designated contract. The
approved input pins survive admission (the seals actually admitted must be the request's, before and
after the lease), the reducer ties request, snapshot, record and bindings together on their inputs, and
the whole generation — with the anchor modules' import digests — is re-verified at every boundary up to
publication.

## 2. Per-finding response

| § | Finding | Response | Where | Tests (intended reason) |
|---|---|---|---|---|
| 665.1 | 24 cells are not a three-seed confirmation; declare other costs | S 24 + D 12 = 36 fresh cells; probes, R, T costed separately | contract §7.1, §7.2, §12 | membership tests; `test_decide_cells_take_each_planned_arm_at_its_frozen_n` |
| 665.1 | retrieval-rule details | raw base-Hamming only in S/D; per-arm N; ddof 1; equality passes; zero SD allowed; unpaired means over shared seeds; per dataset; missing/non-finite refuses | contract §7.3, §10 | `test_retrieval_exactly_at_the_margin_passes`; non-proportion score tests |
| 665.2, 668.4, 680.2 | one exact endpoint | strict within-image top-1; first 512 rows of the trainer's split, fewer refuses; float64; whole-set centring; norm ≤ 1e-12 refuses; ties are misses; integer hits/ties/2048; unrounded float ratio; contract, producer, schema and reducer say the same | contract §8.2; probe | numeric cases; 7 invalid-feature cases |
| 667 | conflicting axis flags | refused first, by name, in the helper and at `_resolve_save_path` | helper | repeat and trainer-boundary tests |
| 668.1 | old refit only on exact match | stage R per branch: anchors → scratch; control at approved N → old refit only if a separately authorized inspection admits it; otherwise scratch | `reduce_decide`; contract §7.4 | three decision-branch tests |
| 668.2 | circular gate; no CIFAR | stage-specific gates; S/D never wait for R/T; no CIFAR cell | contract §3, §7.6 | — |
| 668.3 | every branch in the budget | S 5.575; D 0.7433–5.9467; probes ≈ 0.6; R 0.5575–4.46 (+≈11 %); ceiling 12.5 GPU-h for S+D+probes | contract §12 | — (matches the audit's arithmetic) |
| 669, 670, 676, 677.1 | typed recipe, presence, binding, tokenizer | accepted in §676/§677.1; the control arm of an anchor campaign is bound like the candidate | helper, trainer | incl. `test_an_anchor_cell_without_its_sealed_recipe_refuses[none]` |
| 669.3, 677.2, 685, 687 | override policy vs the real wrappers | exact policy, 7 destinations; actual pinned wrappers rendered for 3 datasets × 2 arms and all 12 S coordinates | helper | positive composed/alias/actual; 8 negative sequences |
| 671.1 | one admission; inputs before resources | shared admission incl. protocol values; run/smoke need manifest and approval and verify seals, all before the lease | launcher | paired plan/smoke/run refusals before every sentinel; order `seals → lease` |
| 671.2, 672.3, 681 | stage-S replay before D; metadata-only plan | `verify_selection` replays from receipts, pinned bytes only; D plan and every refusal reach no deserialiser | launcher, reducer | `test_the_stage_d_launcher_replays_stage_s_without_deserialising` (plan, implicit plan, run, smoke) with a `torch.load` sentinel |
| 671.3, 673.3, 688.1 | real plan is not synthetic; boundary guards | the plan deserialises nothing; render refusals are tested under an independent no-process guard | launcher | `no_process` fixture on both render refusals |
| 672.1, 679 | approval from an independent authority | ledger approval lines (fail-closed); campaign approval re-verified per record; reuse only via source registry; historical inspection removed | launcher `audit_approval`; reducer `campaign_approval`, `load_reuse` | parser: 19 cases; campaign: missing, withdrawn, changed, scope, recorded-line; reuse: self-pinned (3), kind, listing, pins |
| 679.2 | empty proofs; recipe shape/arm; pin before load | complete receipt campaign binding and completion pins; snapshot nonce; sealed recipe shape, digest and protocol values; `config.pt` by byte pin only | reducer | empty campaign (both sides, receipt only), empty/partial pins, nonce mismatches, 5 sealed-recipe cases, digest case, config pin with sentinel |
| 680.3, 690 | measurement envelope | coordinate (exact types), record/checkpoint/config, authority, generation, approval and request, split (exact types), split identity, caption target and seal, integer ties within misses, float ratio, hex rows | reducer `verify_probe` | 27 envelope cases incl. `val_split_seed=42.0` |
| 683.2 | exact closure, designated contract | manifest must list exactly the 56-file closure; contract must be the designated path; commit/branch/clean/environment required and equal to the process | launcher `load_anchor_manifest` | 17 manifest cases incl. missing wrapper, extra file, substituted contract at its correct hash, each historical pin alone and both |
| 689.2 | approval must bind the actual request | canonical request digest in the approval; plan preview; carried in snapshot/receipt; reducer checks coordinate, namespace, result root, stage, mode | launcher, reducer, probe | per-dimension launcher negatives (namespace, root, membership, GPUs, authority, seal, smoke horizon, smoke cell); reducer request negatives (6 + namespace + edited-after-approval + missing); probe request negatives (3) and a positive reaching the first load |
| 694 | approved inputs must survive admission | admitted seals equal the request's pins before the lease and when re-admitted after it; the carried authority's bytes equal its pin before and after parsing | launcher `assert_request_seals`, `_anchor_confirmation_main`, `_run_sweep` | seal changed after approval (refused before the lease); carried authority changed while parsed; readmitted foreign seal after the lease (no snapshot, no reservation); matching seals reach the lease; 4 unit cases |
| 696 | cross-object input consistency | request seal pins = snapshot seals; record input authority = its dataset's admitted seal; launch and cell bindings carry that seal's 5 identities | reducer `campaign_approval`, `admit_metadata` | a consistent positive; 7 inconsistent cases (the audit's reproduced set) and 2 request-seal cases, each re-issued consistently downstream |
| 697 | the generation after admission | the 56-file generation and the anchor modules' import digests re-verified after input admission, after the lease, before every cell, before the receipt; at the reducer's and probe's entry, before the probe's first load and before either publishes | launcher `recheck_generation`; reducer; probe | unchanged positive; drift of each anchor-only member, the contract, a test and a legacy file in a private copy; a module imported from other bytes; sweep boundaries with recorders (after the lease: no snapshot/reservation; before a cell: no dispatch; before the receipt: no publication); reducer publication and probe first load |
| 698, 699, 701 | mutation evidence | L22e corrected; the 8 v3 exceptions re-examined from full outputs; D9 fixtures fixed; per-mutant outputs retained | harness | §3 |
| 675.2 | evidence scope | the plan renders the pinned wrappers under bash with a capture interpreter in temporary directories; the D-plan replay reads pinned bytes (JSON, logs, `config.pt` bytes) but deserialises nothing — neither is "no artifact access"; the trainer check is pre-claim in `_resolve_save_path`, not pre-directory | contract §13; this handoff | — |
| 678, 686 | version boundary; one coherent revision | separate versioned mode; legacy replay unchanged (legacy suites pass); this package | all | §3 |

## 3. Evidence produced for this revision (CPU only)

All at source commit `491c3ce` (`CUDA_VISIBLE_DEVICES=` and `GDNA_NUM_SEMANTIC_PARTS=5`,
`python -m pytest -q -p no:cacheprovider <files>`), one run of all twelve files: **773 passed, 1 skipped,
293 s, rc 0**.

| Suite | Tests |
|---|---:|
| `tests/test_anchor_confirm_port.py` | 16 |
| `tests/test_anchor_confirm_recipe.py` | 65 |
| `tests/test_anchor_confirm_launcher.py` | 160 (one real-artifact test is opt-in and skipped) |
| `tests/test_anchor_confirm_reducer.py` | 169 |
| legacy: `test_phase3_selection_matrix`, `test_phase3_select_n`, `test_result_identity`, `test_seal_phase3_inputs`, `test_phase3_clip_snapshot` | 341 |
| related: `test_extraction_manifest_integration`, `test_phase3_launcher_e2e`, `test_siglip2_criterion_runtime` | 23 |

The audit's independent guarded runs: 372 anchor tests at `c81aca8` (§695), 409 at `dcf6777` (§700) and
31 probe-envelope cases with in-memory mutants at `491c3ce` (§701).

**Mutation batteries** (audit §673.3, §688.1, §698): a detached worktree at the source commit, the anchor
worktree's tracked bytes copied on top and checked equal; every test group passed unmutated first;
each mutant runs only its declared tests and counts only when they fail with the failure mode
declared in advance; errors, other exit codes and timeouts do not count; compound conditions are
disabled whole or one predicate at a time; boundary mutants are caught by independent deny guards
(no process, no deserialisation, no model). The real worktree's tracked bytes and `git status` were
identical before and after.

| Battery | Source commit | Mutants | Result |
|---|---|---:|---|
| v3, full | `c81aca8` | 92 | 82 intended; 8 other reason (re-examined below); 1 survivor L22e = an equivalent mutant (`{} or X`), confirmed by audit §698.1 and corrected; 1 declared unreachable (R12: the rebuilt-digest equality is implied by the preceding checks) |
| v4, targeted | `dcf6777` | 23 | the 14 new guards (§694, §696, §697) plus corrected L22e and the 8 v3 exceptions: 19 intended; 4 other reason, read from the retained outputs: N2 intended (the marker missed pytest's parentheses), L5 and L22f defense in depth, D9 a fixture bug (fixed in `491c3ce`) — the audit's §701.2 reaches the same classification |
| v4b, rerun | `491c3ce` | 4 | N2, L5, L22f, D9: 4 intended (L5 and L22f declared as defense in depth) |

Harnesses, reports, the v3 log and every v4/v4b mutant's full pytest output are in
`artifacts/anchor_confirmation/mutation_v2/`. Stated coverage limits, not mutated: the probe's
re-verification before it publishes and its post-load runtime-row check (both after a real forward
or dataset load); and the import-digest check covers the modules that record their digest (the anchor
modules), with the legacy pre-import handshake covering the trainer closure (audit §700.1).

## 4. Real-artifact operations disclosed

**Before this revision (v1, 2026-09-26, command issue times from the session transcript, UTC):**

| Time | Command | Historical binaries deserialised |
|---|---|---|
| 06:37:12, 06:38:22 | v1 `--anchor-confirm select --plan` (twice) | the three approved refit `config.pt` per run (`anchor_incumbent_recipe_check`, `torch.load(weights_only=False)`), plus 24 wrapper renders per run |
| 06:51:19 | v1 `verify_record` over the 12 approved `p3gE` selection records | 12 `config.pt` |
| 07:03:15 | v1 `verify_record` over the `p3lamA` Flickr25K seed 43/44 records | 2 `config.pt` |

Up to 20 `config.pt` deserialisations; no checkpoint, array, cache, model or forward. The word
"admitted" those scripts printed was the v1 reducer's compatibility label, not reuse admission (§674).

**This revision:** text reads of one `p3gE` selection record and the `p3gE` plan snapshot (JSON) to
design the protocol-value check; SHA256 of `p3lamA_sweep_complete.json` against ledger §536.1; a
listing of `/data/yschoi/gdna_p3exec_seals/` and an idle-GPU query (no job); and the package commands
below (all at `491c3ce`):
`anchor_confirm_manifest.py inventory` (reads the two pinned JSON authorities, the `p3lamA` receipt
bytes and the 56 closure files; writes the manifest once), and the stage-S `--plan` and request
preview at 09:50:42–09:50:59 UTC (issue times). Each plan run re-executes the launcher entrypoint once
(its existing pre-import self-check, which queries git), reads the manifest and the 56 closure files,
reads the two pinned JSON authorities, checks that the three real opt-train whitening files exist
(a stat), renders the three pinned wrappers 24 times under bash with a capture interpreter in
temporary directories (their only writes, removed on exit); the preview also reads and hashes the three
stage-1 seal JSON files it declares. The same three commands had run once at 08:56–08:57 UTC on
`c81aca8`, writing into the scratchpad (superseded). Nothing was deserialised; no result root, record, reservation
or lease was created.

## 5. The stage-S plan and the request proposed for approval

Plan output (`artifacts/anchor_confirmation/plan_select_ancS2_v2.txt`, SHA256
`324d961fb3a75dc551aa701a60d4b5671c39d89f4f5b71b14cdba74fb39a0699`): stage `select`, **24 cells**
(3 datasets × 2 arms × N ∈ {4, 9, 19, 39}, seed 42), namespace `ancS2`, tags
`ancS2_<exp>_N<N>_s42_AX<arm>_<P>_<JD>`. At all 12 coordinates both arms render, repeat exactly the seven
reviewed overrides, carry every protocol value, and differ in `axis_center` alone; their sealed recipe
digests (first 16 hex):

| Coordinate | control (`none`) | candidate (`anchors`) |
|---|---|---|
| flickr25k, N=4, seed 42 | `f452c4f32166acbc` | `f6211f71053316ff` |
| flickr25k, N=9, seed 42 | `37ec43128594d57b` | `0278c91b7bfefd4d` |
| flickr25k, N=19, seed 42 | `d04b60acf7bf9f70` | `b48eecba9383b364` |
| flickr25k, N=39, seed 42 | `a7ea00fcac01fd8a` | `2f5842114a5b703b` |
| nuswide, N=4, seed 42 | `3ce2600a5e899308` | `d762d9d9c96648ef` |
| nuswide, N=9, seed 42 | `59aedf6f529cf827` | `3a0c94bb6e9fed80` |
| nuswide, N=19, seed 42 | `cb79db75321ab510` | `f16a633c3a5de56e` |
| nuswide, N=39, seed 42 | `54ed9b7096ac9dd0` | `2e5d9a1b3b491077` |
| mscoco, N=4, seed 42 | `02736cfd7130e262` | `d060af12dbb13dea` |
| mscoco, N=9, seed 42 | `8f5d422089d8da5e` | `da7f4e5e40426d4e` |
| mscoco, N=19, seed 42 | `4b9d85590b078f1c` | `d02c0a4d2f21252d` |
| mscoco, N=39, seed 42 | `2c9d58d57c5dbe58` | `a5520376fbc97f1f` |

**Proposed stage-S execution** (the command the approval would cover, `GDNA_NUM_SEMANTIC_PARTS=5`):

```
python scripts/phase3_selection_matrix.py --anchor-confirm select --namespace ancS2 \
  --anchor-arms none,anchors --run --gpus <3 free GPUs> \
  --result-root /data/yschoi/gdna_anchor_confirm_v1_result \
  --input-seal flickr25k:stage1=/data/yschoi/gdna_p3exec_seals/flickr25k.stage1.input-seal.json \
  --input-seal nuswide:stage1=/data/yschoi/gdna_p3exec_seals/nuswide.stage1.input-seal.json \
  --input-seal mscoco:stage1=/data/yschoi/gdna_p3exec_seals/mscoco.stage1.input-seal.json \
  --anchor-manifest artifacts/anchor_confirmation/authority_manifest_v2.json \
  --anchor-manifest-sha256 a5f7b8436e3e81b189deedb4d1f3e99eb993bce59d8792bada12f7c1a6a1bf0a \
  --anchor-approval-section <the approving ledger section>
```

Its canonical request (`artifacts/anchor_confirmation/request_preview_ancS2_v2.txt`, SHA256
`19c5e6b5a7decb9be400dfc5247c46f075dd1dd48c492e32fdc6edbfc72ed770`): stage `select`, mode `run`, the 24
cells as executed and declared cells, namespace `ancS2`, record directory
`/data/yschoi/gdna_anchor_confirm_v1/artifacts/anchor_confirmation`, result root
`/data/yschoi/gdna_anchor_confirm_v1_result`, no smoke horizon, the three stage-1 seals at their
file digests (`e44b363a…` Flickr25K, `aa1eaa5d…` NUS-WIDE, `6e9138a6…` MS-COCO — the seals the
approved `p3gE` campaign used), no carried admission authority (a full seal rehash before the lease,
about 82 minutes), GPU count 3, manifest `a5f7b843…`. **Request SHA256
`ee367b84cc29d6d9d61be86e083b641070150b3e160ea01992af1bd47bf00d75`** (recomputed independently from the
printed JSON). The approval line this stage would need, if the audit approves it as submitted:

```
ANCHOR-CONFIRM-APPROVAL version=anchor-confirm/1 scope=stage-S-run manifest=a5f7b8436e3e81b189deedb4d1f3e99eb993bce59d8792bada12f7c1a6a1bf0a request=ee367b84cc29d6d9d61be86e083b641070150b3e160ea01992af1bd47bf00d75
```

Choosing a carried admission authority (to skip the full rehash), another GPU count, a smoke first,
or any other change is a different request with a different digest, previewed the same way.

## 6. Known limitations

- GPU leases precede the plan snapshot and production source-authority check (as in the approved
  lambda path: the snapshot records the leased GPU identities); they are released on exit. The
  irreversible steps (namespace reservation, dispatch) come after every check; input seals and the
  approval are verified before the lease.
- Without a carried admission authority, the pre-lease seal verification rehashes every sealed byte
  (about 82 minutes on this host, ledger-era figure); with one it is stats-only.
- The rebuilt-digest equality in the trainer check is implied by the preceding checks (mutant R12
  survives by design). Mutants marked "defense in depth" are still refused by another guard.
- The probe's runtime-row check (`_assert_phase3_runtime_rows`, after the real dataset load) and
  its forward pass are not exercised by synthetic tests; the gates before them are.
- The control arm equals the approved incumbent in the enforced protocol values, the pinned
  wrapper bytes and the AST-additive `config.py`/`model_siglip2.py` (§667); full-field equality
  with the historical saved configurations is not established (not needed for fresh S/D; it is what
  a stage-R inspection would establish for contract §7.4 row 2).
- The request records the GPU count, not GPU indices, so a stage can start on whichever GPUs are
  free; say if the audit wants indices bound as well.

## 7. Requested decisions and execution scope

As contract §15: (1) fresh controls; (2) the override policy and the approval-line/request format;
(3) the §8.2 endpoint with these probe and reducer bytes; (4) input and environment admission for
stage S (the seals and admission choice in §5); (5) stage S of this exact generation, by an approval
line naming the manifest digest above and the request digest in §5, within 12.5 GPU-hours on 3
GPUs — and whether a one-cell smoke (its own request) comes first; (6) D, probes, R and T later.
