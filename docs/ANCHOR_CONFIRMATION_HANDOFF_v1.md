# Anchor Confirmation v1 — Preparation Handoff (audit §659.4 return list)

**Status: preparation complete for review; NOTHING has been executed.** No GPU smoke, training,
selection, probe forward, refit or official-test evaluation has run under this generation. The
contract is NON-EXECUTABLE until the audit approves its source, plan, rules, inputs and runtime
scope, and the decisions in contract §15 are resolved. Nothing in the historical execution tree
`/data/yschoi/gdna_p3exec`, its seals, manifests or verifier was edited.

## 1. Source diff and immutable pins

- Worktree `/data/yschoi/gdna_anchor_confirm_v1`, branch `arch-exp-2026-09-anchor-confirm`, from base
  `88c3a25b1b309550eafc276c2ce5be7575507173` (approved P3 model/trainer bytes of `5304005`, which are
  unchanged to `88c3a25`, plus the lambda launcher). At creation all 47 closure files equalled the
  historical tree. `git diff 88c3a25 HEAD`: 13 files, +2727/−38. The 38 deletions are the
  input-authority flag list moved into `_input_authority_flags` (same values, same order; the 341
  legacy launcher/selection/identity/seal/snapshot tests pass). The port itself
  (`config.py`, `model_siglip2.py`) deletes nothing.

| Commit | Content |
|---|---|
| `f1f1000` | `--axis_center {none, anchors}` port, additive, the only call gated on `anchors` |
| `201353e` | typed scientific recipe (`dna_utils/scientific_recipe.py`), trainer-boundary check |
| `f6b2bef` | launcher `--anchor-confirm {select,decide}` mode, sealed recipe per cell |
| `174738d` | reducer (`scripts/anchor_confirm_decision.py`) and probe (`scripts/anchor_confirm_code_axis.py`) |
| `11fecb1`, `b8cb11c` | tests closing the mutation-battery gaps (section 6) |
| `9020a3c` | contract with the verified reuse evidence; manifest generator |

Every file digest of the closure plus the anchor files (56) is in the manifest below.

## 2. Historical-to-new authority manifest

`artifacts/anchor_confirmation/authority_manifest_v1.json`, SHA-256
`c1eed986312ba9a017cc559813ce1d0d2b2dc2fc5aba1df3f032424cbda8b94e`, mode 0444, written once by
`scripts/anchor_confirm_manifest.py` from a clean tree at `9020a3c`. It records: the aggregate
`b4f3b0df…` (§285), selected-N `2bf6133d…` (namespace `p3gE`, its 16 record digests and protocol
sources), recipe authority `e61748d7…`, all nine in-scope refit records with the `config.pt` each
reaches through its query extraction manifest, the `p3lamA` receipt `5a8901b7…` (§536; the full
digest was taken from the ledger text, not computed from the file), the incumbent coordinates, the
new commit and file digests, the environment (Python, torch, interpreter), the contract digest
`418091ee…`, and the two read-only checks of section 3. The worktree's committed
`artifacts/phase3_selection/selected_n.json` (`f2218aa7…`) is an older copy and is marked as not
authority.

## 3. Evidence produced without execution

- **Control recipe = approved recipe.** For each in-scope dataset, this generation's rendered
  control refit command (seed 42, approved N/top-p/JD) parses to typed fields equal to the approved
  refit's saved `config.pt` (Flickr25K `d4502d42…`, NUS-WIDE `27145906…`, MS-COCO `abeade4a…`) in
  every field outside 11 sealed-input fields that a campaign supplies from its own seals at launch
  (`phase3_input_seal`, `phase3_input_seal_sha256`, `phase3_input_aggregate_sha256`,
  `phase3_split_identity_sha256`, `phase3_hf_identity_sha256`, six `clip_snapshot_*`).
- **Arms differ in one field.** At all 12 stage-S coordinates the rendered `none` and `anchors`
  recipes differ in `axis_center` alone.
- **Reuse candidates pass the evidence rule.** All 12 `p3gE` seed-42 records and the two `p3lamA`
  incumbent seed-43/44 records pass the reducer's rule for (dataset, `none`, N, seed); the rule
  reproduces N = 4 / 4 / 39.
- **Plan is side-effect free.** `--plan` wrote no file in the worktree, the historical artifacts or
  the lease directory (checked by modification time), reserved nothing and started no process.

## 4. Typed recipe / schema design

`groundeddna-scientific-recipe/1`: `{schema, argv, fields}` as canonical JSON (sorted keys, no NaN).
`fields` = every destination of the trainer's own parser except `tag` and `log_dir`, typed. `argv`
is the exact token list the per-dataset script hands the trainer, rendered by running the script with
a capture shim as `PY` in a scratch directory (refused if the whitening file is missing, since the
script would build it). The approved commands repeat some options on purpose (script default then
launcher override, e.g. CIFAR `--lambda_codeword_codon_sinkhorn 0.1` then `0.0`); a repeat is allowed
only as rendered, because the trainer compares the exact argv. Checks: launcher at plan time (arms,
incumbent); trainer in `_resolve_save_path` before any directory or claim (argv, typed fields, no
abbreviated/unknown option, post-processed args equal parsed ones); `run_cell` after exit (trainer
evidence digest, saved `config.pt` fields); reducer (saved `config.pt` against the rendered recipe).
`args.txt` is never read for a value: its layout cannot distinguish `-3.0` from `3.0`.

## 5. Contract, reducer and plan

- Contract `docs/ANCHOR_CONFIRMATION_CONTRACT_v1.md` (`418091ee…`): question, recorded decisions
  (user choices of 2026-09-23/25/26 incl. the §664.1 train-only path), dataset policy (CIFAR out of
  scope with both negative figures), arms, source generation, split/schedule/score, stages S/D/R/T,
  the frozen rule (retrieval within one control validation SD, equality passes; code-to-axis
  strictly higher), endpoint definitions, aggregation (n = 3, no test), missing/failed-cell policy,
  disclosures, budget, namespaces/commands, unresolved decisions.
- Rendered plan `artifacts/anchor_confirmation/plan_select_ancS1_v1.txt` (stage S, 12 anchor cells,
  approved recipes, tags, per-coordinate recipe digests, incumbent checks).
- Budget (contract §12): stage S candidate 12 cells ≈ 2.8 GPU-h; stage D 6 candidate + 4 control
  cells ≈ 1.5–4 GPU-h depending on the chosen N; probes ≈ 0.3 GPU-h; refit (separately authorized)
  up to 9 cells.

## 6. Tests and negative controls

Command (CPU, CUDA hidden): `GDNA_NUM_SEMANTIC_PARTS=5 python -m pytest -q -p no:cacheprovider
tests/test_phase3_selection_matrix.py tests/test_phase3_select_n.py tests/test_result_identity.py
tests/test_seal_phase3_inputs.py tests/test_phase3_clip_snapshot.py tests/test_anchor_confirm_*.py`
→ **476 passed** (341 legacy, 135 new). Coverage against §659.4: none/anchors commands for all four
dataset profiles; refit and exploratory-arm refusal paths; declared anchors with runtime none,
missing axis, unsupported value, runtime-injected repeat, dropped declared repeat, changed gate
sign, undeclared feature, changed default, abbreviated option, forged display-only args, args
mutated before the boundary; wrong/absent/replaced/disagreeing incumbent pins; missing, extra,
duplicate and reused evidence; replaced log bytes; smoke, non-candidate, refit, no-validation,
wrong-epoch, wrong-metric/distance, wrong-coordinate records; NaN/inf/out-of-range/empty scores;
input changed after reading; output overwrite; probe bound to another checkpoint/config; the real
`_resolve_save_path` refusing before `makedirs`/claim; one integration test of the real `--plan`
against the real approved artifacts (read-only). Refusal tests assert the refusal reason.

**Mutation battery** (a throwaway detached worktree; the real tree verified clean afterwards):
23 guards broken one at a time. First run: 20 killed, 3 survived — the trainer argv-equality check
(tests passed an empty post-processed namespace, so another check refused), reducer membership (the
"extra" case reused a record, the "missing" case refused with a KeyError) and the probe's config pin
(no test). Tests were fixed to pin reasons and the missing test added; second run: **23/23 killed**.
Toy passes are not model equivalence, successful input admission or a completed campaign.

## 7. Proposed namespaces and roots

`ancS1` (stage S), `ancD1` (stage D), `ancR1` (refit), `ancT1` (test); a smoke `ancSmk1` never counts
as evidence. Records in `artifacts/anchor_confirmation/` of this worktree (the anchor mode rebinds
the record directory); run directories under `--result-root /data/yschoi/gdna_anchor_confirm_v1_result`.
The launcher refuses a namespace not matching `anc[A-Za-z0-9]+`.

## 8. Unresolved decisions (contract §15)

1. Admit the `p3gE` (stage S) and `p3lamA` Flickr25K seed 43/44 (stage D) control reuse, or require
   new control cells.
2. CIFAR-10: is the exploratory negative sufficient for the out-of-scope statement?
3. The code-to-axis probe definition and implementation as a decision endpoint.
4. With reuse, whether seed 42 from `p3gE` and seeds 43/44 from `p3lamA`/new cells form one
   three-seed control sample for the SD.
5. Stage R/T authorization after the frozen decision.
6. Per-dataset control arms in stage D (the launcher's arm list applies to all datasets).
7. Acceptance of the reducer and probe bytes as decision authorities.

New-campaign **input admission** (seals for the stage-S/D cells, environment attestation) has not
been performed; the launcher runs the existing seal checks at `--run`, which were not exercised.
