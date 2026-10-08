# Stage 3 pre-registration — teaching the deployment path inside the shared embedding space (text-path line)

Written 2026-10-07 before the first Stage-3 cell (after the gate and smoke cells). Branch `text-diag-2026-09`.
Off-protocol exploration; idle GPUs only (GPU 0 reserved for the anchor line's recovery run until released).

## Base
B1 = the Stage-1 H2 recipe: approved ancS7 arguments, `--no_gumbel_softmax`, token pruning removed
(EOS-pooled caption anchors), the four inert weights at 0 (`lambda_anchor`, `lambda_cibhash_kl`,
`lambda_codon_text_anchor`, `lambda_recon`). B1 cells are re-run under the Stage-3 code (`td3_B1_s*`)
and must be bit-identical to `td1_flickr_H2_s*` (gate: `h1_bit_identity.py`, seed 42 first).

## Arms (base + one delta; seeds 42/43/44; Flickr25K first)
| arm | delta vs B1 | what it tests |
|---|---|---|
| N1 **TD** | `--lambda_xmodal_commit 0 --lambda_path_consistency 0.05 --text_dropout_p 1.0` (term swap; every step also runs a no-text, codebook-mean-routed forward whose codeword distribution must follow the caption-routed one; EMA sees deployment-routed tokens) | does teaching the deployment path move the deployed codes' S? |
| N1′ **S** | `--train_routing_mode codebook_mean` (to be written) | the extreme form: no caption routing at all |
| N2 | TD + cross-modal InfoNCE on the no-text slot token vs the batch's axis captions, replacing `text_code_kl` | cross-image pressure without any offline structure |
| controls | OFF (`--disable_text_supervision`), TD-vis (teacher from a visual-only routed forward: to be defined only if TD passes), axis permutation | text-caused, axis-specific |

Evaluation-only captions: 1,500 Flickr DB rows (`cache_eval/`), never a training input.

## Measurements (every cell)
- Validation mAP@R (last epoch), dead-codeword share, unique-code ratio.
- A3 v2 on 2,000 Flickr rows (500 val + 1,500 eval DB; lexical rule; boot 1000): S at codeword and codon level
  for the DEPLOYED codes; plus `d4d6_train_vs_deploy.py` for the caption-routed codes and the
  transfer ratio S_deployed / S_caption_routed.
- Replication item (formal this time): deployed **codon** S of B1 vs OFF on Flickr (expected ≥ .145 on
  3/3 seeds from the Stage-1 addendum) and, when the NUS/COCO H2 cells exist, on NUS-WIDE and MS-COCO.

## Pre-declared rules (Flickr, T = .10; NUS .15; COCO .18; paired by seed)
- **P-A3:** deployed S > 0 with CI excluding 0 on 3/3 seeds, R(m) > 0 on ≥ 3/4 axes (3-seed mean).
- **P-DELTA:** S(arm) − S(B1) ≥ T on 3/3 seeds, at the codon level (claim endpoint) and reported at the codeword level (mechanism endpoint).
- **P-TXT:** S(arm) − S(OFF) ≥ T on 3/3 seeds.
- **P-RET:** mean val mAP@R ≥ B1 − .008 and no seed below its B1 seed by > .02; −.008…−.03 = user decision; < −.03 terminal.
- **P-HEALTH:** dead ≤ max(.30, B1 + .05); unique ≥ B1 − .08.
- **F6 (falsifier):** caption-routed S rises but deployed S does not → the arm does not teach the deployment path.
- Pass → carry to NUS-WIDE and MS-COCO (own B1/OFF cells); must pass on ≥ 2 of 3 datasets.
- n = 3 seeds, descriptive; no significance test is claimed.

## Entry gates
- Gate cell `td3_gate_H2_s42` (new code, flags at default) bit-identical to `td1_flickr_H2_s42`.
- Smoke `td3_TD_s42`: `args.txt` shows the three TD flags, `train_path_consistency` column present and non-zero, run completes.

## Stage 3-C (written 2026-10-08 before any Stage 3-C cell) — image-conditioned routing anchors

Base stays B1 (= Stage-1 H2, 10 terms). Every arm is B1 + one change; Flickr25K seeds 42/43/44;
claim endpoint = deployed CODON S on the 2,000-row set (same caption file, reference cache and
`score_list.sh` as above). Prior-work basis: DeCap (ICLR 2023) projection-based decoding for the
memory anchor; modality hallucination (Hoffman et al. 2016) for the predictor; scheduled sampling
(Bengio et al. 2015) for the swap schedule. Offline feasibility (CPU, raw CLIP space, 1,000 held-out
rows, memory = 4,500 opt-train rows): centred cos(memory anchor, true caption anchor) .48/.32/.49/.44
per local axis (axis-mean baseline .01–.02; oracle 1-NN .62/.59/.61/.65).

| arm | delta vs B1 | trains? | what it tests |
|---|---|---|---|
| **P-mem** | `--anchor_source memory` (τ .01) applied to the EXISTING B1 checkpoints at scoring time only (`a3_v2.py --set anchor_source=memory`, same for `d4d6`) | no | does a per-image caption-space query at deployment let the B1 weights emit the slot signal? (answers the v184 objection before any training) |
| P-mem τ | τ .02 / .05 on the same checkpoints | no | τ sensitivity of the free arm (reported, not a selection) |
| **P-mem+SS** | `--anchor_source memory --anchor_ss_p_end 1.0 --anchor_ss_horizon 5` (linear 0 → 1 over epochs 0–4; variant `_p05` with p_end .5 only if p_end 1 fails P-HEALTH) | yes | training the routing on the deployment-like query |
| **P-head** | `--anchor_source predictor --lambda_anchor_pred 1.0 --anchor_ss_p_end 1.0 --anchor_ss_horizon 5`; control `PheadNoSS` (p 0: head trained, routing never swapped) | yes | memory-free deployment |
| **MIX** (control) | `--anchor_mix --anchor_mix_horizon 5` (deployment = codebook mean) | yes | is the memory needed at all? expected to end as arm S (unique −.11) |

Rules (unchanged from above): **P-DELTA** deployed codon S(arm) − S(B1) ≥ T = .10 on 3/3 seeds
(B1 = +.161/+.196/+.109); **P-RET** mean val mAP@R ≥ B1 − .008 (P-mem: the retrieval of the rescored
checkpoint is measured by `d4d6`/eval re-extraction, reported beside B1's); **P-HEALTH** dead ≤
max(.30, B1 + .05), unique ≥ B1 − .08; **P-TXT** vs OFF as above.
New diagnostics (every cell): anchor fidelity = cos(image-conditioned anchor, true caption anchor)
in adapter space on held-out rows (threshold ≥ .30 raw-space centred cos is the feasibility level;
own-vs-others rank accuracy ≥ .90), `d4d6` train/deploy codeword agreement (B1 .52–.69; expected to
rise under SS), memory max-weight distribution (τ .01 with 4,500 rows is close to 1-NN).
**F7 (new falsifier):** fidelity high (≥ .30) but deployed codon S unchanged → the caption signal that
reaches the codes is not image-predictable; the image-conditioned family is then closed.
**Ceiling note:** D3's S_probe (Flickr .13) bounds any image-predicted anchor; a pass is read against it.
MIX is not bit-gated against B1 (α = 1 renormalises the caption query); it is compared at the
codon/mAP level only. n = 3 seeds, descriptive, no significance test.
Entry gates: defaults bit-identical to B1 (unit test + seed-42 `log.csv` gate cell before the first
training arm); mutant battery on the new tests (5/5 caught, 2026-10-08); one-cell smoke
(`td3_PmemSS_smoke_s42`, `--stop_after_epoch 1`) checked for `anchor_source memory`, SS flags,
`anchor_memory.json`, `train_anchor_fidelity` and `train_anchor_ss_p` columns.

## Stage 4-G (written 2026-10-08 before any cell) — codon-stage global mixing, on two bases

Motivation: Stage 3-C raised the deployed CODEWORD S (P-mem+SS CI > 0 on 3/3) without moving the
CODON S. The local codon head reads q_m + sigmoid(alpha_m)·sg(q_global) with the gate at ≈ .99
(init 4.595), so a codeword-level slot signal can be washed out by the shared global codeword
before the codon. Both flags exist (v23b / Stage 9); no code change.

| arm | base | delta |
|---|---|---|
| G0 | B1 | `--disable_global_gate` (gate fixed at 0) |
| G1 | B1 | `--global_gate_init_logit -3.0` (gate init .047, learnable; = plan 4c) |
| G0p | P-mem+SS | `--disable_global_gate` |
| G1p | P-mem+SS | `--global_gate_init_logit -3.0` |

Flickr25K seeds 42/43/44; scored on 2,000 rows (`score_list.sh`). Rules: P-DELTA deployed codon S
(arm) − S(own base) ≥ T = .10 on 3/3 seeds (bases: B1 +.161/+.196/+.109; P-mem+SS +.051/+.061/+.121);
P-RET mean val mAP@R ≥ base − .008 (bases B1 .7569/.7474/.7405; P-mem+SS .7633/.7528/.7522);
P-HEALTH dead ≤ max(.30, base + .05), unique ≥ base − .08. Prior records for the same flags (no A3
then): disable → mAP −.013 (2026-07-20) or ±0 (P3gate 09-20), unique +.02–.05, dead down; weak gate →
Flickr mAP ±0, unique +.04, but CIFAR −.07 / NUS −.007 (Stage 11, discarded outside Flickr).
Expected if the "washing" reading is right: codon S follows the codeword S on the P-mem+SS base.
Next after this (user order 2026-10-08): 4a (local instance-contrastive removal), then 4d (text-term
reduction). n = 3, descriptive, no test.

## Stage 4-A (written 2026-10-08 before any cell) — remove the local slots' instance-contrastive term (plan 4a), on two bases

Stage 4-G refuted the codon-stage reading. Next in the user's order: 4a. B1's dominant term is the
per-slot visual-token NT-Xent (λ 1.0, each of the 5 slots asked to identify the image alone); the
plan's cause 1 says this makes every slot carry the whole image. `--cibhash_local_target none`
(stage 7, existing) drops the four local terms but the legacy reduction then gives the global term
weight 1 (5× its share). New option `--cibhash_ntxent_slot_norm all` (sum / M) keeps the global term
at 1/5, so the delta is exactly "the local half removed". Code change: config + loss reduction only;
tests `tests/test_cibhash_slot_norm.py` (defaults bit-identical: sum/M == mean when every term exists).

| arm | base | delta |
|---|---|---|
| A0 | B1 | `--cibhash_local_target none --cibhash_ntxent_slot_norm all` |
| A0p | P-mem+SS | same |
| A1 (control) | B1 | `--cibhash_local_target none` (legacy norm: global term ×5) |

Flickr25K seeds 42/43/44; 2,000-row scoring. Rules as Stage 4-G (P-DELTA codon T .10 vs own base;
P-RET −.008; P-HEALTH dead ≤ max(.30, base+.05), unique ≥ base −.08). Prior record: with the term at
0 for ALL slots Flickr fell .881 → .577 and used 10/128 codewords (plan §5-B); the local-only removal
has not been run. Expected failure modes: dead codewords in the local codebooks (no instance pressure
on local tokens); if P-HEALTH fails, Stage 5 (patch-feature decoder) is the pre-registered remedy, a
user decision. n = 3, descriptive, no test.

## Stage 4-D (written 2026-10-08 before any cell) — text-term reduction (plan 4d), on two bases

| arm | base | delta | note |
|---|---|---|---|
| TH0 | B1 | `--lambda_text_hash_ntxent 0` | the codon heads' only text signal removed (9 terms) |
| TH0p | P-mem+SS | same | |
| XM0p | P-mem+SS | `--lambda_xmodal_commit 0` | XM0 on B1 already measured (Stage 3: codon S −.08/−.08/−.19) |

Flickr25K seeds 42/43/44; 2,000-row scoring; rules as Stage 4-G. Question: does the codon-level
text effect of each base need `text_hash_ntxent` (and, on P-mem+SS, `xmodal_commit`)? A term is
"removable" only if codon S stays within T of the base on 3/3 seeds AND P-RET/P-HEALTH hold.

## Stage L (written 2026-10-08 before any cell) — training-length control

User question (2026-10-08): every A3-scored checkpoint so far trained for N = 4 epochs (70 steps/epoch
on the 4,500 opt-train rows → 350 steps); N = 4 was frozen for validation mAP@R (N 4/9/19/39 = .764/
.736/.719/.706), while old 60-epoch records put codebook maturity later (unique peak e6, base entropy
e37). Slot specificity has never been measured on a longer-trained model under the approved recipe.

| cell | base | delta |
|---|---|---|
| L9 | B1 | `--stop_after_epoch 9 --sinkhorn_schedule_horizon 10` |
| L19 | B1 | `--stop_after_epoch 19 --sinkhorn_schedule_horizon 20` |
| L19p | P-mem+SS | same + `--anchor_ss_horizon 20` (deployment-regime exposure ≈ 15 epochs) |
| PmemSSp1 | P-mem+SS | `--anchor_ss_p_start 1.0` at N = 4 (full 350-step exposure; separates "length" from "exposure") |

Flickr25K seeds 42/43/44; 2,000-row scoring; `lr_schedule_horizon 60` unchanged (approved value).
Readings, not pass/fail: (a) deployed codon and codeword S vs N (B1@4 +.161/+.196/+.109 codon,
+.083/−.005/−.021 codeword); (b) L19p vs L19 at equal length (SS effect) and vs P-mem+SS@4 (length
effect); (c) mAP@R / unique / dead trajectories (eval every 5 epochs → epochs 4/9/14/19 in one log).
If S rises with N on 3/3 seeds, the four 2026-10-08 campaigns are re-read as "within 350 steps" and
the plan is rewritten around training length (user instruction: find an effective remedy, then
rewrite the text-path plan). Descriptive, n = 3, no test.
