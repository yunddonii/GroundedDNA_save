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
