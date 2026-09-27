# Changed-Model TODO Migration Matrix v2 — the fixed four-dataset anchor model

Written for the audit (§709.5, corrected per §711.3). It maps every TODO item and panel from the
approved incumbent model to the changed model: `axis_center=anchors` on all four datasets (contract
v3). **Nothing here is approved or run.**

**What changed from v1** (`docs/ANCHOR_MODEL_TODO_MIGRATION_v1.md`, `056f79ad…`, kept unchanged):
the TODO 13–15 lambda checks are required, not optional; they follow S/D and precede the recipe
freeze F, so the v1 statement that every item depends on F is corrected (§2); the consequences of a
lambda change are stated (§2); the refits run at the final N. Each downstream item needs its own scoped proposal, budget and approval
after the model is frozen (contract v3 §7.6). Old-model completion never becomes changed-model
completion by relabelling. The old statuses stay recorded in `docs/TODO_reexperiments.md`
(`a56f3e9d…`), unchanged.

## 1. Proposed specification amendment (`docs/MODEL_AND_PROTOCOL_SPEC.md`)

The spec does not define `axis_center`. This text is proposed, not yet applied. It goes into the
spec when the changed model is consolidated, so that the spec never describes a model that is
neither approved nor frozen.

> **Routing anchors, axis-centred (model revision, 2026-09-27; audit §709).** Before the routing
> transport cost is computed, each local routing anchor has the per-image mean of the active local
> anchors subtracted (`--axis_center anchors`; `SigLIP2SemanticOTModel._axis_center_local`). The
> global prefix is unchanged. The revision adds no parameter and no loss term. It is distinct from:
> - the codebook anchor-EMA loss `loss_anchor` (`lambda_anchor`, unchanged at 0.05);
> - the per-codon text-anchored prototype CE `codon_text_anchor` (off).
>
> The revision applies to all four datasets. `--axis_center none` is the pre-revision model, whose
> approved results (`p3rfB`, ledger §285) stay the historical comparator.

## 2. Freeze order and shared dependencies

The order is S → D (with the probes) → L → F → R → T → downstream (contract v3 §7.6):
- **S/D** (contract v3): train-only N selection and seed behaviour, with the 12 probes.
- **L (required; TODO 13–15):** preregistered, bounded, train-only lambda checks of the anchor
  model at N_S(d). L follows S/D and precedes F. It is not waived by the anchor-only design, and
  the old-model `p3lamA` evidence does not transfer.
- **If L changes a lambda of dataset d:** N is reselected for d's new recipe on seed 42 in a new
  generation whose cells are the affected datasets only. d's stage D (seeds 43/44) and its three
  probes are regenerated at that N from records of the same generation. d's old-lambda records
  stay historical, not final-model validation. Unchanged datasets keep the v5 records. No reduction
  merges records of two generations; F binds, per dataset, one generation's frozen N record and
  stage-D decision record.
- **F:** the final recipe freeze (anchors, the final N, the final lambdas, and the validating
  generation per dataset), recorded before any R/T or official-test access.
- **R:** 12 scratch full-train refits at the final N.
- **T:** the official-test evaluation and bound extraction of every R checkpoint.

S does not wait for L, R/T or downstream implementations or approvals. Items 13–15 depend on S/D
and precede F. Every other item depends on F, and most also on R/T. Each has its own output root
under `/home/yschoi/gdna_anchor4_result/<item>/` and its own ledger charge. None uses the S/D/probe
allowance.

GPU-hours below are planning arithmetic from contract v3 §12. "Inference" means forward passes on
existing checkpoints, costed per proposal.

## 3. Matrix

| Item | Old-model evidence retained (scope) | New work for the anchor model | Depends on | Cells / seeds | Budget (planning) | Acceptance checks |
|---|---|---|---|---|---|---|
| 1 final-N refits | the 12 `p3rfB` incumbent refits: historical comparator only, not the new model; the comparison is descriptive, not a paired or causal anchor effect (§711.1) | scratch full-train anchor refits at the final N (contract v3 §7.4); bound extraction; T once per checkpoint | F | 12 (4 × 42/43/44) | ≈ 0.7–5.6 GPU-h + 11 % (full train) | approved request; trainer admission of the sealed anchor recipe; terminal checkpoint = N; test accessed once, after F; raw and post-BIO reported against `b4f3b0df…` |
| 2 bio-projection | the uniform reference 62.44 % (mathematical, model-independent) with its derivation | recompute raw → projected effect from the new T outputs (15-base GC [6, 9], max run 3) | T | 12 | inference | projector policy version; per-cell pre/post mAP@R from the same frozen extraction; no re-opened test |
| 3 held-out decoding | none for the new model; old §4.7 numbers stay old-model | remeasure 12 model cells and 4 aggregates; training/probe separation kept; privileged-caption caveat stated | T | 12 | inference | held-out split identities; concept AP not called retrieval mAP |
| 4 native-DNA baselines | the accepted 48-cell baseline scope (inputs, protocol and geometry unchanged) | rebuild the comparison tables against the new ours; no automatic baseline retraining | T | 0 new baseline cells | table regeneration | baseline admission unchanged; the table binds the new ours generation only |
| 5 causal ablations | the old 24 cells test the old parent only | new anchor-parent A2/A4/A5 evidence with exact factor controls; F09 semantics and seed/cell coverage kept | F (parent frozen) | 24 (as the old design; own proposal) | ≈ 24 × (N+1) epochs; own proposal | each factor differs from the anchor parent in one declared field; seeds 42/43/44 |
| 6 | deleted | — | — | — | — | not resurrected |
| 7 slot interventions | old 12 cells = old model | recompute on the new deployment checkpoints/codes with the fixed intervention and control definitions | T | 12 | inference | 12-cell coverage; definitions byte-identical to the accepted ones |
| 8 NMI | old values = old model | new full-database codes for all 12 cells; regenerate statistics and tables | T | 12 | inference | DB split; `pairwise_nmi` of the approved post-processing |
| 9 / 9b figures | old bundles = old model | regenerate with text-free deployment, the declared row IDs, captions only as labels | T | per bundle | inference | row IDs preregistered; qualitative only, no performance claim |
| 10 empty/tiny slots | old empty = 0 not inherited | remeasure routing, slots, codewords, raw/projected codons | T | 12 | inference | local/global separated; the denominator of empty-slot codons stated |
| 11 / D5 20-base | four old nonconforming ours records kept as history; U0 components keep their accepted scope | a length-only intervention of the frozen anchor recipe (40-bit identity, GC [8, 12]) | F | 4 ours cells (seed 42, as the panel) | ≈ 4 × (N+1) epochs | recipe identical to F except the length fields; separate 40-bit identity |
| 12 U0 main baselines | the admitted author-fixed 30-bit evidence, unchanged | update comparisons only | T | 0 | table regeneration | no 108-cell retraining |
| 13–15 lambda checks (**required**) | the old 8-cell Flickr campaign `p3lamA` (ledger §536.1) is not anchor-model evidence | preregistered, bounded, train-only checks on the anchor model at N_S(d), one factor at a time: `lambda_wasserstein` 0.30/0.50 vs 0.15, `lambda_bu` 0 vs 0.02, `lambda_text_hash_ntxent` 0.025/0.10 vs 0.05; a change triggers the §2 lineage rule | S/D; precedes F | own proposal: at most 5 alternates per dataset, seed 42 (not a Cartesian grid) | own budget, not the S/D allowance; planning bound ≤ 20 cells ≈ 1.2–9.3 GPU-h | train-only; no test feedback; dataset scope, seeds, decision rule and budget fixed before any L score |
| 16 Network Dissection | old COCO evidence = old model | new COCO anchor-model routing and mask evidence; fixed universe; absent-category false-positive accounting (F12) | T | COCO × 3 seeds | inference | fixed category universe; FP accounting |
| 17 concept specificity | old 12 cells = old model | new 12-cell interventions, fixed label mappings and controls, 3-seed aggregation | T | 12 | inference | label mapping verified as before |
| 18 randomization sanity | old v1 is nonconforming and cannot stand in | anchor-specific reset/mask/routing definitions; producer repair scope and execution gates kept | T | per proposal | inference | a diagnostic probe is not an admission bypass |
| 19 human evaluation | none | new-model stimuli under the agreed procedure; actual human responses | 9, T | per protocol | human time | no synthetic answers; no completion claim without responses |
| D4 sensitivity | admitted baseline components reusable if unchanged | new MS-COCO ours codes, 3 seeds; exclusion policy kept | T | 3 | inference | old generations and open provenance limits kept |
| D6 appendix | existing baseline metric/provenance holds remain | update comparisons against the new ours; validation-selected appendix separate from the author-fixed main | T | 0 | table regeneration | no mixing of the appendix and main protocols |
| Paper | nothing transfers | regenerate bound tables, figures, uncertainty and claims from the correct generation | all above | — | — | no old/new ours mixing; no unsupported superiority claim; no completion by transcription |

## 4. What is open and not closed by the architecture decision

- The five D5/D6 metric discrepancies.
- The D4 provenance limits.
- The TODO 18 execution requirements.
- The missing human responses.
- The exploratory Gumbel-ON label (PROJECT_LOG 2026-09-27).

The architecture decision does not close any of these (audit §709.5).
