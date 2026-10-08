# GroundedDNA — Modification Log (text path)

Living record of the **text-path modification line** (branch `text-diag-2026-09`, worktree
`/home/yschoi/gdna_textdiag`). It follows the same rules and layout as `docs/PROJECT_LOG.md`:
dated entries, newest first; a status marker at the top of each entry (🟢 active / 🟡 superseded or
analysis / 🔴 reverted or negative); tables over prose for numbers; each model change stated as
"base + one delta"; completed results only (no in-progress entries); never delete, mark
`[reverted YYYY-MM-DD]`.

**Scope and merge policy (agreed 2026-10-01).**
- This file is updated **only on this branch**. `docs/PROJECT_LOG.md` on `arch-exp-2026-09`/`main`
  is not edited from here.
- If a model from this line is adopted as the final model, its entries are **merged** into
  `main`'s `PROJECT_LOG.md` then; until then nothing here is a paper result.
- The anchor-confirmation chain (audit §709–§729: stage S/D/probes, N record, L/R/T) belongs to
  another line and is not driven from this branch. Its results are cited here only as context.
- Nothing in this line modifies the sealed inputs, the audited worktree, or a running campaign.
  Measurements use CPU unless a GPU job is explicitly requested; audited checkpoints are not loaded
  without their own admission.

---

## Current state (as of 2026-10-05)

- **Question of this line.** Does the text path make images that share an element in slot *m*
  receive the same slot-*m* codeword/codon, and if not, what change would make it so without
  growing the objective?
- **Answer so far (exploratory checkpoints, Gumbel ON):** caption-similar images share codewords
  4–14× more often than random pairs, but **equally in every slot** and **equally without any text
  supervision**. The claim "텍스트 경로가 유의미하게 관여" cannot be made for the current model.
- **Plan approved 2026-10-05** (entry below): weakest intervention first. Stage 0 CPU diagnostics →
  Stage 1 baseline + text-OFF at the approved recipe → Stage 2 dataset-specific axes and compact
  phrases → Stage 3 cross-image text target X replacing the `text_code_kl` target → conditional
  structural deltas → confirmation. The "text assigns membership" form (A′ family) is NOT approved
  for training; existing A′ checkpoints are only scored on CPU as a ceiling.
- **User decisions in force (2026-10-04):** element = axis nouns / relation words; slot specificity
  from text supervision, separation from the multiple codebooks; one compact phrase per axis (no
  label/detail split); text-OFF = codebook-mean routing in training; CIFAR-10 dropped.
- **Metric of record:** A3 cross-image slot consistency (own-slot lift vs other-slot lift vs
  text-OFF), `result/analysis/textdiag_2026-09-29/a2a3_slot_consistency.py`; A3 v2 to be written
  (Stage 0, D0).

---

## 2026-10-09 [design record, no results] Text-path modification plan v2: the input side (patch-to-slot grounding, axis-caption distinctness) is the only part no campaign has touched; primary dataset moves to MS-COCO (masks + headroom)

Full text: `docs/TEXT_PATH_PLAN_v2_2026-10-09.md` (user instruction 2026-10-08 #2: find an effective remedy, then rewrite the plan). Basis: Stages 3, 3-C, 4-G, 4-A, 4-D, L (loss side, routing-query side and training length all tested; none produced slot-specific deployed codes; B1's codon signal is largely global information copied through the gate and fades with training). Untested causes kept from the records: (A) slots see nearly the same patches (router cost on CLIP last-layer patch tokens, known to be poorly localized — ClearCLIP ECCV'24, MaskCLIP ECCV'22, SCLIP'24; strongest patch 2.3–4.1× the uniform share; each patch to ~3 of 5 slots); (B) the four axis captions of one image barely differ (local-local cosine .58–.62). Ceiling: S_probe .13/.29/.20, so Flickr has little headroom.
Stages: S0 diagnostics (D7 MS-COCO pointing game with instance masks, current vs dense CLIP features; D8 axis-caption distinctness of V4/V5b vs Stage 2 option-C pilot; D9 headroom), S1 dense-grounded routing keys (training-free ClearCLIP-style last layer, routing cost only), S2 distinct axis captions (option C), S3 image-conditioned anchors on the passing base (code from Stage 3-C), S4 conditional (decoder as the instance-term substitute; gate-off readout as a diagnostic). New endpoint: axis-exclusive S; new stability rule: pass at N and N+5. Kept: N, the gate, text_hash_ntxent / xmodal_commit, H2 recipe; excluded: offline graph X and constructed T. User decisions pending: COCO as primary, dense cache re-extraction on /home, resuming Stage 2 option C (axis names), the stability rule.

## 2026-10-09 [OFF-PROTOCOL exploration, branch text-diag-2026-09 — not a paper result] Stage L: training length. Every Stage 1–3 Flickr variation re-run to N = 19 (and B1 at N = 9): retrieval falls on every arm, the codebooks mature (dead → 0, unique +.05–.14), but slot specificity does not appear — no arm's deployed codon S at N = 19 reaches B1's at N = 4, and B1's own codon text effect shrinks from +.15–.19 to −.01/+.17/−.00

**Status:** 🔴 for the "too short to learn" hypothesis (user question 2026-10-08); 🟢 as a control (it closes a confound that applied to every earlier campaign). Pre-registration `stage3/PREREGISTRATION.md` (Stage L and Stage L extension). Cells 2026-10-08 14:30Z – 2026-10-09 17:00Z: B1 at N = 9 (L9) and 19 (L19), P-mem+SS at 19 (L19p), P-mem+SS with p_start 1 at 4 (PmemSSp1), and 12 arms × 3 seeds at N = 19 (tags `<original>_N19_s<seed>`), all with `--sinkhorn_schedule_horizon N+1` (length and Sinkhorn schedule change together, for every arm alike). 2,000-row A3 v2 on the final checkpoint; validation metrics at epochs 4/9/14/19 from `log.csv`. Records: `/home/yschoi/GroundedDNA/result/analysis/textdiag_stageL_train_length_20261008/` (`length_table.txt`, `cells.txt`, `eval_runs.txt`, per-cell logs). d4d6 does not apply to OFF/S/SN2/MIX (no caption-routed training forward) and returned rc 1 for those 12 runs, as at N = 4.

Mean ± SD over seeds 42/43/44 (validation, Flickr25K):

| arm | mAP@R N4 | N19 @e9 | N19 @e19 | unique N4 → N19 | dead N4 → N19 | deployed codon S N4 → N19 | deployed codeword S N4 → N19 |
|---|---|---|---|---|---|---|---|
| B0 | .748 | .727 | .719 | .540 → .649 | .182 → .005 | +.011 → −.021 | +.012 → −.048 |
| **B1 (H2)** | .748 | .727 | .709 | .558 → .646 | .157 → .013 | **+.155 → +.072** | +.019 → +.040 |
| OFF | .735 | .700 | .687 | .459 → .539 | .151 → .000 | −.011 → +.021 | +.002 → +.009 |
| TD | .739 | .731 | .728 | .515 → .591 | .083 → .019 | −.002 → +.037 | +.061 → +.027 |
| XM0 | .744 | .727 | .718 | .507 → .597 | .192 → .013 | +.041 → −.004 | −.025 → +.015 |
| TDXM | .754 | .725 | .714 | .542 → .589 | .084 → .009 | +.108 → +.033 | +.051 → +.042 |
| TDXM5 | .753 | .722 | .714 | .457 → .555 | .177 → .001 | −.068 → −.003 | −.018 → +.045 |
| S | .758 | .735 | .721 | .449 → .523 | .166 → .000 | +.071 → +.060 | +.029 → +.019 |
| SN2 | .756 | .736 | .722 | .463 → .523 | .154 → .000 | +.022 → +.071 | +.012 → +.030 |
| BN2 | .751 | .731 | .718 | .581 → .630 | .140 → .011 | +.055 → +.019 | +.021 → +.028 |
| P-mem+SS | .756 | .735 | .721 | .449 → .569 | .191 → .000 | +.078 → +.059 | +.070 → +.032 |
| P-head | .756 | .737 | .726 | .465 → .589 | .226 → .000 | +.100 → +.064 | +.079 → +.037 |
| P-head noSS | .753 | .736 | .716 | .528 → .670 | .208 → .001 | +.023 → +.036 | +.040 → +.076 |
| MIX | .755 | .736 | .720 | .455 → .545 | .179 → .000 | +.060 → +.035 | +.029 → +.033 |

B1 at N = 9 (L9): mAP@R .730/.739/.734, unique .63–.65, dead .02–.03, codon S +.004/+.124/+.181. P-mem+SS with full deployment-regime exposure at N = 4 (PmemSSp1): codon S +.015/+.061/+.098, codeword S +.043/+.025/+.026 (below P-mem+SS). P-mem deploy-only on the B1 N = 19 checkpoints: codon S +.007/+.148/+.061 → +.026/+.064/+.066. **B1 − OFF codon S per seed: N4 +.145/+.187/+.166 → N19 −.012/+.169/−.004.**

**Reading.** (1) Training length is not why the designs fail: with 4.75× the steps every codebook is healthy (dead ≈ 0) and unique codes rise on every arm, yet the slot-specific sharing that A3 measures does not grow on any arm; the largest N19 codeword S is +.076 (P-head noSS) and the largest codon S +.072 (B1), both below T = .10. (2) Longer training ERODES B1's codon effect (2 of 3 seeds lose it entirely) and costs .02–.05 mAP@R on every arm; only TD keeps retrieval (−.011). (3) Every arm's N = 19 codon S is close to OFF's (+.021), so whatever text adds at N = 4 is a transient of early training, consistent with the Stage 4-G finding that B1's codon S is largely global information copied into the local codons. (4) The two levers left untested by any campaign are on the input side: how distinct the four axis captions of one image are (local-local text cosine .58–.62; D2/D3 ceilings S_probe .13/.29/.20) and how separately the router hands patches to the slots. These go into the plan rewrite (user instruction 2026-10-08 #2). Descriptive, n = 3, no test.

## 2026-10-08 [OFF-PROTOCOL exploration, branch text-diag-2026-09 — not a paper result] Stage 4-D (plan 4d): `text_hash_ntxent` is not removable on either base (codon S and retrieval fall); `xmodal_commit` is removable on the P-mem+SS base by the pre-registered rule (codon S within T, retrieval −.005)

**Status:** 🟡 one removable term (on the P-mem+SS base only). Pre-registration `stage3/PREREGISTRATION.md` (Stage 4-D); cells 2026-10-08 13:27Z onward (Flickr25K, seeds 42/43/44); `td3_TH0_s42` hung in its DataLoader for 2.2 h with no error and was killed by PID at 15:35Z and re-run in a per-GPU queue (record: `textdiag_stage4d_text_term_reduction_20261008/td3_TH0_s42.failed1.log`); 2,000-row scoring; `summarize_3c.py TH0 TH0p XM0p`. Removable = codon S within T = .10 of the base on 3/3 seeds AND P-RET (mean ≥ base − .008) AND P-HEALTH.

| arm | base | delta | val mAP@R Δ (42/43/44) | unique Δ | dead | deployed codon S (base) | deployed codeword S (base) |
|---|---|---|---|---|---|---|---|
| TH0 | B1 | `--lambda_text_hash_ntxent 0` | −.010 / −.023 / −.021 | −.03 / −.04 / −.00 | .209/.219/.145 | −.068 / +.035 / +.159 (+.161/+.196/+.109) | −.018 / −.004 / +.070 |
| TH0p | P-mem+SS | same | −.013 / −.018 / −.029 | +.07 / +.04 / +.07 | .148/.167/.105 | −.043 / −.091 / +.044 (+.051/+.061/+.121) | −.022 / −.032 / +.022 |
| XM0p | P-mem+SS | `--lambda_xmodal_commit 0` | −.004 / −.007 / −.003 | +.01 / −.03 / −.03 | .152/.117/.200 | +.064 / +.011 / +.044 | +.058 / −.003 / +.050 |

**Verdicts.** TH0: not removable (codon S −.229/−.161/+.051; P-RET fails, mean −.018). TH0p: not removable (codon S −.095/−.152/−.077; P-RET fails, mean −.020). XM0p: removable by the rule (codon S +.013/−.050/−.077, all within T; mAP mean −.005; unique −.014; dead within bound), though its codeword S falls (−.042/−.049/−.014). On B1, `xmodal_commit` removal was already measured as XM0 (codon S −.08/−.08/−.19, Stage 3) and stays. So `text_hash_ntxent` (the codon heads' only text signal) is load-bearing for retrieval and codon S on both bases; `xmodal_commit` matters when the training router sees captions (B1) and not when it mostly sees memory anchors (P-mem+SS), consistent with its role of pulling caption-routed slot tokens toward the caption's codeword. Descriptive, n = 3, no test.

**Recording change (user order 2026-10-09).** From this stage on, every run directory of this line lives in `/home/yschoi/GroundedDNA/result/` and every stage has a bundle in `/home/yschoi/GroundedDNA/result/analysis/textdiag_stage*/` (cells.txt, eval_runs.txt, per-cell logs, PREREGISTRATION.md, summary, RESULT.md), built by `result/analysis/textdiag_2026-10/make_main_bundles.py`; 107 earlier run directories were moved there on 2026-10-09 (old → new paths in `result/analysis/textdiag_2026-10/RUN_DIR_MOVES.tsv`). New cells run through `stageL/run_queue_main.sh` (per-cell timeout, records written as the cell finishes).

## 2026-10-08 [OFF-PROTOCOL exploration, branch text-diag-2026-09 — not a paper result] Stage 4-A (plan 4a): removing the local slots' instance-contrastive term collapses the local codebooks (dead .47–.61, unique .13–.35) on both bases; the codon S becomes seed-erratic; 4a fails P-HEALTH on 9/9 cells

**Status:** 🔴 negative. Pre-registration `stage3/PREREGISTRATION.md` (Stage 4-A); code `92bed6e` (`--cibhash_ntxent_slot_norm {computed,all}`: sum / M so the global NT-Xent term keeps its 1/M share when the four local terms are dropped; test `tests/test_cibhash_slot_norm.py`, mutant caught; default bit-identical); cells 2026-10-08 13:05–13:13Z (Flickr25K, seeds 42/43/44); 2,000-row scoring; `summarize_3c.py A0 A0p A1`.

| arm | base | delta | val mAP@R Δ (42/43/44) | unique (base) | dead (base) | deployed codon S (base) | deployed codeword S (base) |
|---|---|---|---|---|---|---|---|
| A0 | B1 | `--cibhash_local_target none --cibhash_ntxent_slot_norm all` | **−.031 / −.011 / −.011** | .209/.207/.202 (.557/.554/.563) | **.508/.544/.606** (.155/.117/.200) | +.009 / +.179 / −.122 (+.161/+.196/+.109) | −.046 / +.021 / −.070 (+.083/−.005/−.021) |
| A0p | P-mem+SS | same | +.004 / +.014 / +.007 | .300/.181/.349 (.429/.454/.463) | .520/.472/.575 (.186/.159/.228) | +.171 / −.045 / −.105 (+.051/+.061/+.121) | +.090 / +.007 / −.048 (+.100/+.046/+.064) |
| A1 (control, legacy norm = global term ×5) | B1 | `--cibhash_local_target none` | −.002 / +.003 / +.020 | .212/.127/.269 | .477/.494/.588 | −.021 / +.140 / −.090 | −.005 / +.018 / −.066 |

**Verdicts.** P-HEALTH fails on every cell (dead 2.4–5× the base, unique −.11 to −.43). P-DELTA fails (no arm ≥ T on 3/3; the per-seed codon S swings from −.12 to +.18 because the lexical statistic becomes unstable when a codebook keeps 50–70 dead codewords). P-RET: A0 fails (s42 −.031 = terminal bound), A0p and A1 pass. The train-routed codeword S of A0 s42 (+.275) shows the slot tokens still separate axes before quantisation; the collapse is in the codebooks, which the local instance pressure was keeping alive.

**Reading.** The plan's cause 1 ("each slot asked to identify the image alone makes every slot carry the whole image") cannot be fixed by deleting the term: without instance pressure the local tokens of different images fall onto a few codewords. Stage 5 (a patch-feature reconstruction decoder as the replacement information-preserving term) is the pre-registered remedy and a user decision. Per the user's order the remaining structural lever is 4d (text-term reduction); `xmodal_commit` removal on B1 was already measured as XM0 (codon S −.08/−.08/−.19, Stage 3 entry), so 4d runs `--lambda_text_hash_ntxent 0` on both bases and `--lambda_xmodal_commit 0` on P-mem+SS. Descriptive, n = 3, no test.

## 2026-10-08 [OFF-PROTOCOL exploration, branch text-diag-2026-09 — not a paper result] Stage 4-G: removing (G0) or weakening (G1) the codon-stage global gate on B1 and on P-mem+SS. The deployed codon S FALLS on both bases: the global codeword mixed into every local codon is load-bearing for B1's codon S, not what washes a codeword-level signal out. Unique codes +.02–.09, dead codes down, retrieval within −.009

**Status:** 🔴 both arms fail P-DELTA on both bases; the "washing" reading of Stage 3-C is refuted. Pre-registration `stage3/PREREGISTRATION.md` (Stage 4-G); cells 2026-10-08 12:31–12:55Z (Flickr25K, seeds 42/43/44, GPUs 0–5; the first wave-2 launch failed before python with an empty command — my launcher-path error, relaunched as `td3_G1r_*`/`td3_G1pr_*`, result dirs `td3_G1_*`/`td3_G1p_*`); scored on 2,000 rows (`stage3/a3v2_2000`, `d4d6`); `summarize_3c.py G0 G0p G1 G1p`. Flags are the existing `--disable_global_gate` (v23b) and `--global_gate_init_logit -3.0` (Stage 9); no code change; B1 trains with the gate at sigmoid(4.595) ≈ .99.

| arm | base | val mAP@R Δ (42/43/44) | unique Δ | dead (base) | **deployed codon S** (base) | deployed codeword S (base) | train/deploy agreement |
|---|---|---|---|---|---|---|---|
| G0 `--disable_global_gate` | B1 | −.005 / −.008 / +.007 | +.03 / +.03 / +.04 | .148/.067/.177 (.155/.117/.200) | **−.015 / +.047 / −.022** (+.161/+.196/+.109) | −.054 / +.058 / +.021 (+.083/−.005/−.021) | .54–.61 |
| G0p | P-mem+SS | −.007 / −.004 / −.009 | +.09 / +.05 / +.05 | .106/.147/.192 (.186/.159/.228) | +.049 / +.004 / +.092 (+.051/+.061/+.121) | −.007 / +.084 / +.049 (+.100/+.046/+.064) | .81–.86 |
| G1 gate init −3.0 | B1 | −.006 / −.005 / +.014 | +.03 / +.04 / +.02 | .089/.177/.203 | +.065 / +.020 / +.075 | +.034 / −.006 / −.010 | .49–.62 |
| G1p | P-mem+SS | −.007 / −.001 / −.007 | +.08 / +.04 / +.05 | .150/.148/.181 | +.077 / −.027 / +.089 | +.056 / +.040 / +.022 | .81–.85 |

**Verdicts.** P-DELTA (codon, T .10, 3/3): all four fail; codon S − base = G0 −.176/−.149/−.131, G0p −.003/−.056/−.030, G1 −.096/−.176/−.033, G1p +.026/−.088/−.033. P-RET: passes (means −.002 / −.007 / +.001 / −.005; G0 s43 −.0083 is at the bound). P-HEALTH: passes everywhere (unique up on 12/12 cells, dead down on 10/12).

**Reading.** The global codeword added to each local codon (gate ≈ .99) is where a large part of B1's codon-level S comes from: pairs that share an axis-m word also share scene-level content, and a shared global codeword inside every local codon makes their slot-m codons coincide more often. Removing it does not release a hidden slot-specific signal; the codeword-level gain of P-mem+SS (+.10/+.05/+.06) stays at the codeword level with or without the gate. So the codon-level S that the paper's claim rests on is, in part, global information replicated into the local codons, and the codeword→codon stage is not the bottleneck that Stage 3-C suggested. Side effect confirmed from earlier records: the gate trades code diversity for retrieval (unique +.02–.09, dead −.01–.08, mAP −.002…−.007 mean). Next per the user's order (2026-10-08): 4a (remove the local slots' instance-contrastive term, with a new normalisation option so the global term keeps weight 1/M), then 4d. Descriptive, n = 3, no test.

## 2026-10-08 [OFF-PROTOCOL exploration, branch text-diag-2026-09 — not a paper result] Stage 3-C: image-conditioned routing anchors (caption-memory projection / predictor head, scheduled sampling, MIX control). Train/deploy codeword agreement rises to .83–.89 and the deployed CODEWORD-level S becomes consistently positive, but the CODON-level S does not exceed B1 on any arm and unique codes fall by ~.10; the image-conditioned family does not pass P-DELTA

**Status:** 🔴 negative on the claim endpoint (codon S), 🟡 informative on the mechanism endpoint (codeword S, agreement). Plan §Stage 3-C (approved 2026-10-08); pre-registration `stage3/PREREGISTRATION.md` (Stage 3-C section); code `dd8b5f9` (tests 37, mutants 5/5 caught, defaults bit-identical); cells 2026-10-08 10:54–11:03Z (Flickr25K, seeds 42/43/44, GPUs 0–5); scored on 2,000 rows (`stage3/a3v2_2000`, `d4d6`); `stage3/summarize_3c.py` prints the table. Base B1 = Stage-1 H2.

**Design (prior work → arm).** DeCap (ICLR 2023) projection-based decoding → **memory anchor**: Σ_j softmax(cos(v_img, t_j,m)/τ)·t_j,m over the 4,500 opt-train rows' raw per-axis caption features (τ .01; vector-match self-exclusion in training), passed through the SAME whitening + per-slot text adapter as real captions. Modality hallucination (Hoffman et al. 2016) / CoCoOp meta-net → **predictor head** (LN–Linear–GELU–Linear, residual on the image feature, zero-init; loss 1 − cos to the stop-grad cached caption feature, λ 1.0; detached before the router). Scheduled sampling (Bengio et al. 2015) → per-image swap of the caption routing query for the image-conditioned one with p 0 → 1 over epochs 0–4 (dedicated RNG); text losses always see the caption. MIX control: query = norm(α·norm(caption) + (1−α)·codebook mean), α 1 → 0; deployment = codebook mean. Offline feasibility (raw CLIP space, held-out 1,000 rows): centred cos(memory anchor, true anchor) .48/.32/.49/.44 per axis vs axis-mean .01.

| arm | training | val mAP@R Δ vs B1 (42/43/44) | unique (B1 .557/.554/.563) | dead (B1 .155/.117/.200) | **deployed codon S** (B1 +.161/+.196/+.109) | deployed codeword S (B1 +.083/−.005/−.021) | train/deploy codeword agreement (B1 .52–.69) | anchor fidelity (adapter cos) |
|---|---|---|---|---|---|---|---|---|
| **P-mem** (deploy-only, B1 weights, τ .01) | none | −.0007 / +.0011 / +.0033 | .555 / .545 / .582 | — | +.044 / +.123 / +.099 | +.015 / +.024 / +.048 | .68–.75 | .67 (τ .02 .65, τ .05 .43) |
| **P-mem+SS** | yes | **+.0064 / +.0054 / +.0117** | .429 / .454 / .463 (−.13/−.10/−.10) | .186 / .159 / .228 | +.051 / +.061 / +.121 | **+.100 / +.046 / +.064** (CI > 0 on 3/3) | **.83–.89** | .70 |
| **P-head** (+SS) | yes | +.0004 / +.0044 / +.0172 | .475 / .466 / .454 (−.08/−.09/−.11) | .209 / .241 / .227 | +.133 / +.009 / +.157 | +.111 / +.035 / +.090 | .82–.86 | .79 (pred loss .19) |
| P-head, no SS (control) | yes | −.0033 / +.0033 / +.0135 | .519 / .533 / .533 (−.04/−.02/−.03) | .184 / .223 / .217 | −.001 / −.044 / +.115 | +.040 / −.006 / +.087 | .67–.76 | .78 |
| MIX (control) | yes | +.0039 / +.0036 / +.0118 | .461 / .436 / .468 (−.10/−.12/−.10) | .170 / .209 / .156 | +.030 / +.074 / +.076 | +.038 / +.011 / +.038 | n/a (`d4d6` refuses routing_mode "mix") | — |

P-mem τ grid on the B1 weights (val mAP@R Δ): τ .02 −.0008/+.0015/+.0057, τ .05 −.0007/−.0037/+.0095; unique falls with τ (s42 .555/.551/.543). Memory softmax max-weight median .23 at τ .01 (not 1-NN), .03 at τ .02.

**Verdicts (pre-registered rules).**
- **P-DELTA (codon, T = .10, 3/3 seeds): fails on every arm.** Codon S − B1: P-mem −.117/−.073/−.009; P-mem+SS −.110/−.135/+.013; P-head −.028/−.187/+.048; P-head-noSS −.162/−.240/+.006; MIX −.131/−.122/−.032.
- **P-RET: passes on every trained arm** (means +.008 P-mem+SS, +.007 P-head, +.005 P-head-noSS, +.006 MIX); P-mem deploy-only is neutral.
- **P-HEALTH: fails on P-mem+SS, P-head and MIX** (unique −.10 to −.13 vs the −.08 bound; dead > B1 + .05 on 5/9 of their cells); P-head-noSS passes.
- Mechanism endpoint: the deployed **codeword** S is positive with CI > 0 on 3/3 seeds for P-mem+SS (first arm to do so) and on 2/3 for P-head; the caption-routed and deployed codes now agree on 83–89 % of codewords (B1 52–69 %, TDXM5 73–75 %).
- F7 (fidelity ≥ .30 but S unchanged): adapter-space fidelity .67–.79 on every arm while the codon S does not rise → **the image-conditioned family is closed for the codon-level claim.**

**Reading.** (1) Giving the deployment path a per-image query in the caption space does what it was built for — the deployed codewords now follow the caption-routed ones — and it costs nothing in retrieval (P-mem+SS even gains +.008). (2) But the slot-specific sharing that the lexical rule measures survives only at the codeword level (+.02 to +.11) and not at the codon level, which is the paper's endpoint. The codon of a local slot is formed from its own codeword plus the global codeword (plan cause 5), so a codeword-level gain can be washed out in the codon stage; this is now the most direct lever left (Stage 4 structural changes: codon-stage global mixing, local instance-contrastive 4a, global gate 4c). (3) Every arm that trains on image-conditioned queries loses ~.10 unique codes, as arm S did: the per-image query variation that the caption supplied was holding codes apart. (4) The B1 codon S of .11–.20 (Flickr only) remains the best deployed codon signal, and it depends on caption-anchor routing during training. Descriptive, n = 3, no test. Nothing under the anchor trees was touched; GPUs 0–5 idle again.

## 2026-10-08 [design record + pilot of the attribute step, no training] Stage 2 step 2 cannot be done by the 8B VLM as "concept abstraction": it copies phrases; replaced by a non-VLM exact dedupe + direct attribute grouping on phrase samples. Attribute sets per dataset below (NOT yet confirmed by the user; chains not relaunched)

**Branch** `text-diag-2026-09`, worktree `/home/yschoi/gdna_textdiag`. Tools: `tools/stage2_attributes_from_phrases.py` (46a9c5d, 5e01a76), runner `cache_eval/stage2_alt_run.sh`; outputs `cache_eval/stage2_alt/<ds>/` (`phrases_dedup.json`, `attributes_shuffle{0,1,2}.json`, `attributes.json`); tmux `stage2_alt2` (Flickr, NUS; 2026-10-08 00:54–01:00Z, GPU 1) and `stage2_alt3` (COCO; 01:02–01:05Z, GPU 1). Qwen3-VL-8B-Instruct, greedy decoding, text-only input.

### 1. Why step 2 (VLM concept list) was dropped
Step 1 (survey: free phrases per image, 1,000 + 500 images per dataset) is complete and unchanged. Step 2 as designed ("read the phrases and write the collection's concepts") was tried in four forms (cumulative list; two-level per-batch extraction + consolidation, batches of 80 then 25 images; v3 names-only lists with regex fallback, commit abb290e). In every form the text-only model **copied the input phrases** instead of abstracting them: 70–100 % of returned "concepts" were verbatim survey phrases, consolidation passed 444 → 403 strings through, long replies overflowed 4,096–6,144 tokens (NUS produced one 26 k-character line, COCO malformed JSON). The step added nothing a sort-and-dedupe does not, so step 2 is now **exact-match dedupe of the survey phrases (lower-cased, whitespace-normalised), no VLM**. This changes the approved 4-step flow at step 2 only; steps 1, 3 (attributes) and 4 (captions) and their prompt wordings are unchanged.

### 2. Step 3 as run: PROMPT_ATTRIBUTES on random phrase samples
Input = 300 phrases sampled uniformly from the deduplicated pool (seeds 0, 1, 2); the three groupings are then merged by PROMPT_ATTRIBUTES_MERGE when they differ (they always did). Two execution fixes, both to the OUTPUT-SIZE clause only, selection criteria unchanged: (i) "the concepts that belong to it" → "up to 8 of the listed concepts that belong to it" (with 300 phrases the model enumerated every phrase and looped past 3,072 tokens on all three datasets); (ii) a shuffle whose reply still overflows is retried with the first 150, then 100 phrases of the same permutation (needed once: COCO shuffle 1). Prompt sha256s and full texts are stored in `attributes.json`.

| dataset | survey images | phrase tokens | unique phrases | shuffle 0 | shuffle 1 | shuffle 2 | merged (final candidate) |
|---|---|---|---|---|---|---|---|
| flickr25k | 1477 | 17557 | 11992 | Lighting and Atmosphere / Composition and Framing / Subject and Focus / Color and Pattern | Lighting and Atmosphere / Composition and Framing / Subject and Action / Texture and Detail | Color Palette and Tone / Surface Texture and Material / Lighting and Shadow Direction / Composition and Framing | **Lighting and Atmosphere / Composition and Framing / Subject and Action / Texture and Detail** |
| nuswide | 1499 | 17789 | 9461 | Lighting Conditions / Color Palette / Surface Texture / Composition and Framing | Lighting Condition / Composition and Framing / Surface and Texture / Subject and Focus | Lighting and Atmosphere / Composition and Framing / Color and Texture / Subject and Activity | **Lighting and Atmosphere / Color Palette / Surface and Texture / Subject and Activity** |
| mscoco | 1479 | 17631 | 11188 | Lighting and Atmosphere / Color Palette and Dominant Hues / Surface Texture and Material / Object and Scene Composition | Color Dominance / Surface Texture and Material / Subject Type and Action / Environmental Context and Setting (150 phrases, retry) | Subject Focus / Color Palette / Surface Texture / Environmental Context | **Lighting and Mood / Color Dominance / Surface Texture and Material / Subject and Action** |

**Merged definitions**

- flickr25k:
  - *Lighting and Atmosphere* — Describes the quality, source, and mood of illumination, including natural or artificial light, ambient glow, and atmospheric effects.
  - *Composition and Framing* — Describes the spatial arrangement, perspective, and structural elements that define how the image is framed or oriented.
  - *Subject and Action* — Describes the primary subject or activity depicted, including human or animal figures and their behaviors.
  - *Texture and Detail* — Describes surface qualities, patterns, and fine visual elements that add tactile or intricate visual interest.
- nuswide:
  - *Lighting and Atmosphere* — Describes the overall illumination, time of day, weather, or atmospheric conditions present in the image.
  - *Color Palette* — Describes the dominant or notable color tones present in the image, including saturated hues or monochromatic schemes.
  - *Surface and Texture* — Describes the physical properties and visual feel of surfaces, such as wetness, roughness, or material type.
  - *Subject and Activity* — Describes the primary living or inanimate subjects and their actions or states within the image.
- mscoco:
  - *Lighting and Mood* — Describes the quality, source, and emotional tone of illumination in the scene.
  - *Color Dominance* — The primary color scheme or hue that defines the visual tone and contrast of the image.
  - *Surface Texture and Material* — Describes the tactile or visual quality of surfaces, including material composition and physical state.
  - *Subject and Action* — The primary living subject or object and its state of motion, behavior, or compositional role.

### 3. Observations (facts, not a verdict)
1. **The model groups by descriptive modality, not by content.** On all three datasets the attributes are lighting, colour, surface/texture and (in some shuffles) subject or composition. This happens although the sampled inputs are mostly object phrases (e.g. Flickr shuffle 0 saw "squirrel", "white dress", "biplane flying in sky", "yellow umbrella", …). The prompt's requirement "a kind of visual information that can be described for most images and whose description differs from image to image" is satisfied trivially by lighting/colour/texture (every photo has them), which plausibly drives this choice. Relation words appear nowhere; "Composition and Framing" (Flickr merged, NUS/COCO shuffles) is a camera-side attribute.
2. **The attribute set is sample-dependent.** A subject/object attribute appears in 2/3 Flickr shuffles, 2/3 NUS shuffles, 3/3 COCO shuffles; colour appears in 1/3 Flickr, 3/3 NUS, 3/3 COCO; composition in 3/3 Flickr, 3/3 NUS, 1/3 COCO. The merge call then decides which four survive. The merged sets of NUS and COCO agree (lighting / colour / surface / subject+action); Flickr differs (composition instead of colour).
3. The `agreement` field (membership Jaccard) in `attributes.json` is uninformative under the 8-example clause (different samples, 8 examples each) and is labelled so; `n_assigned` likewise.
4. Survey pools are dominated by background/sky/lighting phrases (top phrases: "dark background", "black and white", "blue sky", "cloudy sky"; NUS "blue sky" 174, "horizon line" 150; COCO "trees in background" 64). Reported for information only; no frequency was used to choose anything (user rule 2026-10-07).

### 4. Status
No caption was generated under these attributes; the dataset chains (`tools/stage2_run_dataset.sh`) remain stopped at the concepts step and were not rewired. Pending user decisions: (a) accept the structural change at step 2 (non-VLM dedupe), (b) accept the merged attribute sets as the VLM's decision, or spend one of the two allowed wording revisions of PROMPT_ATTRIBUTES, or something else. Rule ③ (attributes are never edited after generation) is respected: nothing above was hand-edited.

## 2026-10-07 [design record, no results] Stage 2 caption redesign: four VLM prompts fixed after a history review (PROJECT_LOG prompt issues V1–V9) and a literature review; user decisions (a) no relation-avoidance wording, (b) field length "about 5 to 10 words", no length A/B

**Status:** 🟢 approved design (user, 2026-10-07). Principle: the VLM decides concepts and axes; the
designer supplies only images, output format, the count (4 local attributes = slot count) and a rough
length. No labels, no tags, no caption statistics, no designer taxonomy. Image-based survey only
(label vocabulary rejected: it would narrow the unsupervised claim).

**History review (what the plan must not repeat; PROJECT_LOG lines in the 2026-10-07 review):** V5
NO-GO — universal/abstract axes collapse to image-independent phrases (L20068) and `scene_type` was the
redundant slot (L6246); V8 critique — relations/colours are arguments of objects (vlm file L1181);
frequency-selected concepts are the ones every image shares (A0 overlap 34–51); V7 hedging migrates
when "none" is banned (34–83 %) and the extractor catches only literal ""/"none"; the .83 cosine stop
rule was a SigLIP2 number (CLIP never reaches it); slot-name prefixes survive axis centring; V6b-style
duplicate captions are false negatives for `text_hash_ntxent`; tools import the prompt from the main
repo; whitening uses slots 1–5 (empty 5th slot = constant vector); positional key mapping fails
silently; A3's lexical rule assumes noun/noun/verb/colour kinds by slot position.

**Literature review (web, 2026-10-07; judge over three threads):** the four-step structure matches
ALBM (summarise freely generated concepts rather than asking for attributes), X-Cluster / IC|TC /
TnT-LLM (caption → propose → consolidate). Contradictions fixed: mini-batch concept updates instead
of one 12k-line pass (VisDiff, GoalEx, TnT-LLM, Order Effect); C_global 10–15 words (Long-CLIP
effective length ≈ 20 tokens); separate visual-check pass (ALBM q_vis; VLG-CBM/DN-CBM); phrase length
1–4 words and "fewer is fine" (LeHaCE: hallucination ∝ output length; POPE). Supported: image input,
"do not guess", generic/rare concept removal, 5–10-word fields (Long-CLIP; CLIP single words are OOD),
"never state absence" (NegBench), no attribute-name prefix, greedy without few-shot. Open (to be
measured): stability of a fixed K = 4 grouping, effect of "do not repeat", guessing induced by "nearest
visible thing", survey saturation at 1,000 images, whether frozen CLIP separates the found axes
(S_probe per attribute, after full generation).

**Prompts (final; the exact text lives in `tools/stage2_prompts.py` with sha256 recorded in every
output row):** ① SURVEY (image; 1,000 `opt_train_rows` per dataset; 1–4-word phrases, ≤ 12, fewer is
fine; JSON; do not guess). ②-a CONCEPTS (text; 50–100-image mini-batches updating a cumulative list;
drop non-visual / same-meaning / nearly-every-or-almost-none concepts). ②-b VISUAL CHECK (text; keep
only concepts visible without outside knowledge; CLIP-cosine > .9 pairs merged and reported).
③ ATTRIBUTES (text; exactly 4; "most images … differs from image to image"; minimal overlap;
"describable without naming what another attribute describes"; `note` if fewer than 4; three
concept-order shuffles, agreement reported, merge pass if they differ; no relation-avoidance sentence —
user decision (a)). ④ CAPTIONS (image; attribute fields first, C_global last; each field "about 5 to
10 words … keep it to what is actually in the image"; C_global 10–15 words; visible things only, never
state absence; no repetition; no attribute-name prefix; greedy) — user decision (b): no length A/B.

**Pilot (500 opt-train images per dataset, disjoint from evaluation rows) and gates:** parse or
unmatched-key ≤ 1 %; none-like/negation/hedge patterns ≤ 1 %; field starts with attribute name ≤ 1 %;
cross-attribute top-100 word overlap ≤ the V4 (COCO: V5b) value on the same rows; stop for a user
decision if the within-image mean cosine of the four fields (raw CLIP EOS space, slots 1–4) exceeds the
matched V4 value + .03 or any attribute's cross-image same-slot cosine ≥ .75. Reported only: length,
per-field-position none/hedge/duplicate rates, per-attribute duplicate rate (user decision if > 5 %),
type/token, unseen-noun rate (④ nouns not in ① phrases), survey saturation (new-phrase rate on 500
extra images), shuffle agreement (Jaccard). At most two wording revisions of ④; ③ is never edited.
S_probe per attribute after full generation, beside V4. Full generation starts automatically when the
gates pass (user instruction 2026-10-07); the four attributes are reported at that point and the user
may halt.

**Pipeline:** new tools in this worktree (prompt imported locally, outputs to `cache_eval/stage2/`
on /home, explicit attribute→legacy-key map with unmatched keys counted as failures and retried,
original fields kept under `codebook_texts_v10`, resume-safe, 2 shards per GPU) → pooled text
features → token features (donor = new pooled dir) → whitening fitted on slots 1–4 only (new option)
with `GDNA_NUM_SEMANTIC_PARTS=5` → cell C = B1 + new captions, Flickr first. A3 v2's per-slot lexical
kind is set from each attribute's definition before any caption is scored and validated with the
oracle-mix control; V4/V5b re-scored under the same rule.

---

## 2026-10-07 [OFF-PROTOCOL exploration, branch text-diag-2026-09 — not a paper result] Stage 3 arm N1 (TD, text-dropout path consistency): fails P-DELTA at the codon level; the codon-level text effect of B1 depends on `xmodal_commit`; the consistency term at λ .05 halves dead codewords but does not transfer the caption-routed signal to the deployed codes

**Status:** 🔴 negative for N1 as configured; 🟡 diagnostics informative. Pre-registered in
`result/analysis/textdiag_2026-10/stage3/PREREGISTRATION.md`. Code `498cc4a` + `f0dd43e` (flags
`--lambda_path_consistency`, `--path_consistency_tau` (.5, on per-row standardised codebook
distances), `--text_dropout_p`; 28 tests; gate `td3_gate_H2_s42` bit-identical to `td1_flickr_H2_s42`).
Cells (Flickr25K, seeds 42/43/44, GPUs 1-5, 2026-10-07 05:28–05:58Z), scored on 2,000 rows (500 val +
1,500 eval-DB) with A3 v2 and `d4d6`; records `stage3/{a3v2_2000,d4d6,A3V2_2000_SUMMARY.md}`.
First smoke with τ .1 on raw squared distances collapsed the codebook (term 589, dead .76); fixed before any cell.

| arm | delta vs B1 (= Stage-1 H2, 10 terms) | mAP@R Δ (42/43/44) | dead | **codon S** (B1: +.161/+.196/+.109) | codeword S (B1: +.083/−.005/−.021) | caption-routed S |
|---|---|---|---|---|---|---|
| **TD** (N1) | `xmodal_commit` → `path_consistency` .05, p = 1 | −.015 / −.012 / +.000 | .07–.09 | **+.131 / −.091 / −.046** | +.065 / −.021 / +.139 | +.08 / +.09 / +.11 |
| XM0 | `xmodal_commit` 0 only | −.010 / −.010 / +.006 | .17–.21 | +.086 / +.115 / −.079 | −.054 / +.032 / −.051 | +.02 / +.04 / +.06 |
| TDXM | + `path_consistency` .05, `xmodal_commit` kept (11 terms, diagnostic) | −.009 / +.006 / +.021 | **.07–.09** | +.070 / +.141 / +.111 | +.128 / +.078 / −.053 | +.17 / +.16 / +.17 |

- **N1 verdict:** fails P-DELTA at the codon level (Δ vs B1 −.03 / −.29 / −.16) and is in the user-decision
  zone on P-RET (mean −.009). Codeword level mean +.06 vs B1 +.02, not ≥ T = .10 on 3/3 seeds.
- **Why:** removing `xmodal_commit` alone (XM0) lowers the codon S on 3/3 seeds (−.08 / −.08 / −.19) and
  the caption-routed S (.02–.06 vs B1 .12–.20): the codon-level text effect of B1 is carried by
  `xmodal_commit` (image slot token → caption's codeword). That term stays.
- **The consistency term itself** (TDXM): keeps retrieval (mean +.006), halves dead codewords (EMA sees
  deployment-routed tokens), raises the caption-routed S to .16–.17, but the deployed codon S is still
  ≤ B1 on 2/3 seeds and train/deploy codeword agreement is unchanged (.52–.69). At λ .05 the term is
  ≈ .5 % of the objective; it does not move the deployed path toward the teacher. F6 in its weak form.
- Descriptive, n = 3, no test. Nothing under the anchor trees or GPU 0 was touched (reservation agreement).

**λ escalation (TDXM form, λ .5; the single allowed escalation) — negative.** Cells `td3_TDXM5_s42-44`
(2026-10-07 06:13Z, code `4a2b240` at default flags for everything but the TD flags):

| arm | mAP@R Δ | dead | unique | train/deploy codeword agreement | deployed codon S | deployed codeword S | caption-routed S (500 rows) |
|---|---|---|---|---|---|---|---|
| TDXM5 | +.007 / +.005 / +.003 | .14–.21 | .42–.49 | **.73–.75** (B1 .56–.65; s44 primary slot .09) | **−.053 / −.028 / −.125** | +.027 / −.000 / −.081 | +.19 / +.08 / +.14 |

At λ .5 the consistency term does what it was built for — the deployed codewords now agree with the
caption-routed ones on ~74 % of images — yet the deployed codon S turns negative on 3/3 seeds and the
codeword S is ~0, while unique codes fall (.56 → .45). Matching the caption-routed assignment
distribution does not carry the slot-specific sharing across; it narrows the codebook instead. This is
F6 in its strong form. **The N1 family (TD / TDXM / TDXM5) is closed.**

**Arm N1′ S (`--train_routing_mode codebook_mean`; code `4a2b240`, gate `td3_gate2_H2_s42` bit-identical
to Stage-1 H2; cells `td3_S_s42-44`, 2026-10-07 06:19Z):** routing uses the codebook-mean anchors in
training too; the text path and every text loss stay on.

| arm | mAP@R Δ vs B1 | dead | unique | deployed codon S (B1: +.161/+.196/+.109) | deployed codeword S (B1: +.083/−.005/−.021) |
|---|---|---|---|---|---|
| S | **+.007 / +.003 / +.020** | .11–.22 | **.45** (B1 .56) | **+.050 / +.060 / +.103** (CI > 0 on 3/3) | +.063 / −.015 / +.040 |

- Retrieval improves on 3/3 seeds (mean +.010): caption-anchor routing during training was costing
  retrieval, not helping it. Unique codes fall by .11 (fails P-HEALTH's −.08 bound).
- The text losses alone, applied to deployment-routed tokens, leave a consistent codon-level S
  (+.05–.10, every CI > 0, S − OFF ≈ +.05–.10) — smaller than B1's (+.11–.20). So part of B1's
  codon effect needs the caption-anchor routing (the caption-routed tokens that `xmodal_commit` and
  the codon heads see), and part survives without it. Neither arm reaches T = .10 over the other.
- `d4d6` is not applicable to S (its "caption-routed" forward is codebook-mean by construction).
- **Verdict:** S fails P-DELTA (codon S below B1) and P-HEALTH (unique); it passes P-RET and P-TXT.
  Next: N2 (cross-modal contrastive on the deployment-routed slot token) on top of S and of B1.

**Arm N2 (`--lambda_text_preq_contrastive 0.05` replacing `text_code_kl`; the v174-α term, text_m ↔
pre-VQ slot token InfoNCE, in-batch; cells `td3_SN2_s42-44` on S and `td3_BN2_s42-44` on B1,
2026-10-07 06:27Z):**

| arm | mAP@R Δ vs B1 | dead | unique | deployed codon S | deployed codeword S |
|---|---|---|---|---|---|
| BN2 (B1 + N2) | +.000 / +.007 / −.000 | .12–.15 | **.58–.59** | +.041 / +.011 / +.112 (B1 +.161/+.196/+.109) | +.001 / −.028 / +.091 |
| SN2 (S + N2) | +.003 / +.009 / +.010 | .11–.19 | .44–.48 | +.020 / +.054 / −.007 (S +.050/+.060/+.103) | +.007 / +.030 / −.000 |

- No collapse this time (the v174-α CUB failure does not recur at λ .05 on Flickr); retrieval and
  codebook health are fine (BN2 even raises unique codes to .58).
- But the codon-level S **falls** relative to each base on 2/3 seeds (BN2 − B1: −.12 / −.19 / +.00;
  SN2 − S: −.03 / −.01 / −.11). Replacing `text_code_kl` with an in-batch cross-modal contrastive term
  removes part of the codon effect — `text_code_kl` (judged "removable" for mAP on 2026-09-20) is not
  removable for A3. Both arms fail P-DELTA.

**Stage-3 standing (2026-10-07 07:00Z).** Five arms (TD, TDXM, TDXM5, S, N2×2) all leave the deployed
codon S at or below B1's +.11–.20; the two that teach the deployment path most directly (S, TDXM5)
improve retrieval or agreement but not the slot signal. B1's codon effect depends on the caption-anchor
routing during training, on `xmodal_commit` and on `text_code_kl` together. The pre-registered
replication of "B1 vs OFF codon S" on NUS-WIDE and MS-COCO (own H2 cells) is the next step; Stage 2
(captions) waits for the user's caption-length decision.

**Pre-registered replication on NUS-WIDE and MS-COCO (cells `td3_{nus,coco}_H2_s42-44`, 2026-10-07
06:40Z; scored on all validation rows, 1,050 / 1,000; OFF and B0 from Stage 1; T = .15 / .18):**

| dataset | seed | mAP@R H2 / B0 / OFF | unique H2 (B0) | codon S H2 [CI] / B0 / OFF | H2 − OFF |
|---|---|---|---|---|---|
| NUS | 42 | .7256 / .7225 / .6950 | .548 (.533) | +.086 [+.02,+.15] / +.098 / −.013 | +.10 |
| NUS | 43 | .7217 / .7292 / .6985 | .548 (.512) | +.071 [−.00,+.14] / +.046 / −.018 | +.09 |
| NUS | 44 | .7302 / .7247 / .6874 | .554 (.521) | −.032 [−.10,+.03] / +.078 / +.018 | −.05 |
| COCO | 42 | .6375 / .6341 / .5825 | .385 (.403) | +.055 [+.00,+.11] / +.029 / −.008 | +.06 |
| COCO | 43 | .6439 / .6426 / .5765 | .383 (.392) | +.081 [+.03,+.13] / +.040 / −.025 | +.11 |
| COCO | 44 | .6331 / .6302 / .5855 | .392 (.385) | +.011 [−.05,+.06] / +.078 / +.030 | −.02 |

- **H2 as a recipe holds on all three datasets:** retrieval equal to B0 (NUS mean +.000, COCO +.003),
  dead codewords lower, unique codes equal or higher. **B1 = H2 is confirmed for NUS-WIDE and MS-COCO.**
- **The codon-level text effect does NOT replicate at T:** H2 − OFF is positive on 4/6 seed–dataset
  cells (+.06 to +.11) but negative on seed 44 of both datasets, and no cell reaches T (.15 / .18).
  The Flickr finding (+.145 to +.187 on 3/3 seeds at 2,000 rows) stays a Flickr-only observation; on
  NUS/COCO the deployed codes carry at most a small, seed-dependent text signal.
- Descriptive, n = 3 per dataset, no test. This closes Stage 3 on the V4/V5b captions: no arm beats
  B1, and B1's own effect is confirmed only on Flickr25K. Stage 2 (captions) is the next lever.


---

## 2026-10-07 [OFF-PROTOCOL exploration, branch text-diag-2026-09 — not a paper result] Stage 1: baseline, text-OFF and two clean-ups at the approved recipe. Text keeps retrieval (OFF −.013 / −.032 / −.054) and gives a small codon-level slot signal (B0 − OFF > 0 on 9/9 seed–dataset cells) but nothing at the codeword level; the caption-input dependence of D4 is confirmed at the approved recipe; H1 bit-identical; H2 passes

**Status:** 🟡 exploratory, pre-registered (`result/analysis/textdiag_2026-10/stage1/PREREGISTRATION.md`).
Commands rebuilt from the approved ancS7 seed-42 `args.txt` by `build_cmd.py` (seal/authority keys
dropped, `--no_gumbel_softmax` explicit), launched directly on idle GPUs 0/1/3/4/5 (tmux `td1_q*`);
22 cells, all rc 0, 2026-10-06 21:20–22:16. Results `result/261006+*_td1_*`; scoring
`stage1/{a3v2,d4d6,d4_control_*.json,h1_bit_identity.json,A3V2_SUMMARY.md}`. Gate: the seed-42 Flickr
B0 cell reproduces the campaign run's last-epoch val mAP@R (.76419 vs .764194). Seeds 42/43/44,
`hash_target_mode siglip_cos`, 5 slots, N = 4/4/39. Descriptive, no test.

**Retrieval and codebook health** (last-epoch validation mAP@R; paired Δ = arm − B0, same seed):

| dataset | arm | mAP@R per seed | mean ± SD | paired Δ | dead | unique |
|---|---|---|---:|---|---:|---:|
| Flickr25K | B0 | .7642 / .7436 / .7360 | .7479 ± .0146 | — | .182 | .540 |
| Flickr25K | OFF | .7395 / .7323 / .7336 | .7351 ± .0038 | −.025 / −.011 / −.002 | .151 | .459 |
| Flickr25K | H1 (4 inert λ → 0, s42) | .7642 | — | +.0000 (bit-identical) | .158 | .529 |
| Flickr25K | H2 (no token pruning) | .7569 / .7474 / .7405 | .7483 ± .0083 | −.007 / +.004 / +.005 | .157 | .558 |
| NUS-WIDE | B0 | .7225 / .7292 / .7247 | .7255 ± .0034 | — | .020 | .522 |
| NUS-WIDE | OFF | .6950 / .6985 / .6874 | .6936 ± .0057 | −.028 / −.031 / −.037 | .000 | .486 |
| MS-COCO | B0 | .6341 / .6426 / .6302 | .6356 ± .0063 | — | .020 | .393 |
| MS-COCO | OFF | .5825 / .5765 / .5855 | .5815 ± .0046 | −.052 / −.066 / −.045 | .000 | .300 |

- **H1 passes** (`h1_bit_identity.json`): 1,250 parameter tensors `torch.equal` to B0 s42; every shared
  `log.csv` column identical; `train_loss` lower by exactly .05 × `train_loss_anchor` (.04918 = .04918).
  The 10-term recipe (`--lambda_anchor 0 --lambda_cibhash_kl 0 --lambda_codon_text_anchor 0 --lambda_recon 0`)
  is the base of every later cell.
- **H2 passes P-RET and P-HEALTH** (mean +.0003, no seed below −.008; dead .182 → .157; unique
  .540 → .558). **B1 = H2.**
- Text-OFF costs retrieval on every seed of every dataset (−.013 / −.032 / −.054) and lowers unique
  codes (.540 → .459, .522 → .486, .393 → .300): the text path is a retrieval contribution even before
  any interpretability claim.

**A3 v2 on the deployed codes** (lexical rule, all validation rows, S with per-seed CIs in
`A3V2_SUMMARY.md`; B0 − OFF paired by seed):

| dataset | level | S(B0) per seed | S(OFF) per seed | B0 − OFF |
|---|---|---|---|---|
| Flickr25K | codeword | +.06 / +.01 / −.02 | −.00 / −.02 / −.05 | +.06 / +.03 / +.03 |
| Flickr25K | codeword, H2 | +.09 / +.01 / −.06 | — | — |
| Flickr25K | **codon** | **+.13 / +.08 / +.01** | +.09 / +.07 / −.11 | +.04 / +.01 / +.12 |
| Flickr25K | codon, H2 | **+.14 / +.13 / +.12** (CI > 0 on 2/3) | — | vs OFF +.05 / +.06 / +.23 |
| NUS-WIDE | codeword | +.04 / −.01 / +.04 | +.03 / −.01 / −.01 | +.01 / −.00 / +.06 |
| NUS-WIDE | **codon** | **+.10 / +.05 / +.08** (CI > 0 on 2/3) | −.01 / −.02 / +.02 | +.11 / +.06 / +.06 |
| MS-COCO | codeword | −.06 / +.01 / +.02 | −.02 / −.01 / −.06 | −.03 / +.02 / +.08 |
| MS-COCO | **codon** | +.03 / +.04 / +.08 | −.01 / −.03 / +.03 | +.04 / +.07 / +.05 |

- Codeword level: every cell within its CI of 0; no slot-specific sharing in the deployed codewords,
  with or without text (as on the exploratory checkpoints).
- **Codon level: B0 − OFF is positive on 9 of 9 seed–dataset cells** (+.01 to +.12; Flickr H2 − OFF
  +.05 to +.23), the only consistent text-vs-no-text difference on A3 so far. It is small (≤ S_probe)
  and single cells are inside their CIs; n = 3 seeds per dataset, no test. Working hypothesis (not
  tested): the text supervision that reaches the deployed code sits in the **codon heads**
  (`text_hash_ntxent`, 28 % of the gradient, is the only text term on the codon path), not in the
  codeword assignment.
- Calibration at these row counts: the first mixing fraction whose CI excludes 0 is f = .5 on Flickr
  (S ≈ .47; f = .4 gives +.21 [−.03, +.39]), f = .3 on NUS (+.15 [+.02, +.25]), f = .4 on MS-COCO
  (+.18 [+.07, +.26]). **With 500 Flickr rows the smallest detectable S (≈ .2–.5) exceeds S_probe
  (.13)**: no realistic Flickr effect can pass P-DELTA until the evaluation sample is enlarged
  (plan: caption a fixed 1,500-image DB sample as evaluation labels). Thresholds T are therefore
  fixed per dataset as max(2 × SD_seed[S(OFF)], detectable S) = Flickr ≈ .21 (provisional until the
  larger sample), NUS .15, COCO .18.

**D4 at the approved recipe** (`stage1/d4d6`, `stage1/d4_control_*.json`; Flickr/NUS/COCO B0, Flickr
H2 and OFF, 3 seeds):

| cell | S caption-routed (own captions) | S caption-routed (ANOTHER image's captions) | S deployed | P(same codeword, train vs deploy) |
|---|---|---|---|---|
| Flickr B0 | +.15 / +.17 / +.08 (CI > 0 on 2/3) | −.01 / −.06 / −.04 | +.06 / +.01 / −.02 | .57–.70 |
| Flickr H2 | +.19 / +.20 / +.12 | — | +.09 / +.01 / −.06 | .56–.65 |
| NUS B0 | +.13 / +.16 / +.13 (3/3) | — | +.04 / −.01 / +.04 | .57–.61 |
| COCO B0 | +.18 / +.21 / +.16 (3/3) | — | −.06 / +.01 / +.02 | .54–.56 |
| Flickr OFF, routed with captions through its untrained adapter | +.02 / −.06 / −.00 | +.01 / −.03 / −.02 | −.00 / −.02 / −.05 | .91–.94 |

- Confirmed at the approved recipe: the caption-routed codes carry S ≈ S_probe; the deployed codes do
  not; **swapping in another image's captions removes the signal, and a model that never learned
  from text shows none when given captions.** The signal is carried by *this image's caption as an
  input* through the learned text adapter. Nothing in training ever applies a loss to the
  deployment-routed token (codebook-mean anchors), and 30–46 % of codewords differ between the two
  routings. Corrects the 2026-10-06 reading "the deployment lookup discards it" (see the correction
  note on that entry).
- D6 at the approved recipe: legacy pruning keeps .62–.66 of caption tokens, the adapted anchor has
  cosine .49–.70 to the EOS-pooled anchor, 15–25 % of codewords change; without pruning the
  caption-routed S rises on Flickr (+.22/+.19/+.11 vs +.15/+.17/+.08) and falls on NUS/COCO. H2 is
  adopted on P-RET/P-HEALTH, not on this.

**Addendum 2026-10-07 — Flickr re-scored on 2,000 rows** (500 validation + 1,500 evaluation-only
database images captioned with V4 on 2026-10-07, `cache_eval/`; 0 training overlap; reference text
cache `cache_eval/ref_text_flickr_v4_plus_evaldb`; `stage1/a3v2_2000/`, `A3V2_2000_SUMMARY.md`).
CI half-width falls from ±.13 to ±.04; the calibration now separates f = .3 from f = 0 (+.09), so
**T(Flickr) = .10** (max of 2 × SD_seed[S(OFF)] = .02 and the detectable step ≈ .09).

| arm | codeword S per seed | codon S per seed | codon, axis-exclusive | codon S − OFF (paired) |
|---|---|---|---|---|
| OFF | +.005 / +.010 / −.010 | +.016 / +.009 / −.057 | −.005 / +.028 / −.090 | — |
| B0 | **+.109** / −.022 / −.051 | +.096 / +.005 / −.067 | +.122 / −.004 / −.113 | +.08 / −.00 / −.01 |
| **H2 (= B1)** | +.083 / −.005 / −.021 | **+.161 / +.196 / +.109** (CI > 0 on 3/3) | **+.220 / +.250 / +.161** | **+.145 / +.187 / +.166** |

- OFF is a clean zero at 2,000 rows (every CI within ±.05 of 0).
- B0's codeword reading is seed-unstable and lives in the primary-object axis: R(primary) = +.41 /
  −.17 / −.82 by seed. The legacy token pruning (noise-keyed anchors) is the suspected cause; not tested.
- **H2 (EOS-pooled anchors) shows a consistent codon-level effect:** S .11–.20 on 3/3 seeds with CIs
  excluding 0, H2 − OFF ≥ .145 on every seed, driven by the primary-object axis (codon R +.40 / +.41
  / +.53) with secondary/activity positive and colour mixed. This is the first text-vs-no-text
  difference on A3 that clears T. It was **not pre-registered as a test** (Stage 1 listed B0 − OFF as
  "reported, not judged" and did not list H2 on A3), so it is recorded as a descriptive finding; B1
  = H2 is the base that every Stage 3 arm must now beat (P-DELTA), and the Stage 3 pre-registration
  will include "deployed codon S of B1 vs OFF" as a formal replication on NUS/COCO with their own
  H2 cells.
- Codeword-level S of H2 is small (+.02 mean): the slot-specific signal sits in the codon mapping,
  consistent with the codon-head hypothesis above.

**Consequences.** (1) B1 = H2 with the 10-term recipe. (2) Stage 3 as revised on 2026-10-07: teach
the deployment path inside the shared embedding space (TD / S, then cross-modal contrastive N2),
no offline structure; judge on the deployed codes' S and report the caption-routed S beside it.
(3) Evaluation-sample expansion is a precondition for Flickr (user decision: caption 1,500 DB images
as evaluation-only labels, ≈ 40 GPU-min). (4) The codon-level B0 > OFF pattern is the first candidate
for a text-caused effect and is re-measured on every later arm.

---

## 2026-10-06 [analysis, no training, CPU only] Stage 0 complete (D0/D2/D3/D4/D5/D6): the deployed codes carry no detectable slot-specific signal at 500–1,050 rows; the training-time text-routed codes do (S .14–.24 on 12/12 seed–dataset cells), and the deployment lookup discards it

**Status:** 🟡 diagnostic, exploratory checkpoints (Gumbel ON, own N; 2026-09-20/22 arms); V4/V5b
captions. Records: `result/analysis/textdiag_2026-10/` — `a3_v2.py` (instrument), `d0/a3v2/*.json` +
`d0/A3V2_SUMMARY.md` (60 runs), `d2/*.json` (5 caption sets), `d3/*.json` (3 datasets), `d4d6/*.json`
(12 runs), `d5/*.json` (9 runs); tmux `textdiag_d0/d3/d4d6/d5`, all rc 0. CPU only; nothing under the
anchor worktrees or any campaign touched. Descriptive, n = 3 seeds, no test run.

**D0 — the instrument (A3 v2).** Same reference captions and text cache for every arm; all validation
rows (Flickr 500, NUS 1,050, COCO 1,000); primary pair rule = lexical "shared element" (object axes:
noun-like word; colour axis: ≥ 2 colour/material words; relation axis: verb-like word; a word used by
> 20 % of rows does not define a pair); statistic R(m) = log(lift_own / lift_other), S = mean over the
four axes; image bootstrap (1,000) for CIs; codeword and codon level. **Calibration:** the slot-m code
of a fraction f of rows replaced by an oracle (k-means id of the row's axis-m caption):

| rows | f = .1 | .2 | .3 | .4 | .5 | 1.0 |
|---|---:|---:|---:|---:|---:|---:|
| Flickr 500 | +.05 [−.11,+.17] | +.09 [−.06,+.25] | +.17 [−.05,+.31] | **+.33 [+.13,+.57]** | +.41 | +.94 |
| COCO 1,000 | +.05 | +.10 | +.10 [−.02,+.15] | **+.21 [+.10,+.28]** | +.38 | +.81 |
| NUS 1,050 | +.02 | +.04 | +.12 [+.00,+.22] | **+.23 [+.12,+.35]** | +.47 | +.89 |

A role carried by fewer than ≈ 40 % of images (Flickr) / ≈ 30 % (NUS, COCO) is invisible at these row
counts; f = 0 reproduces the plain reading; f = 1 is the positive control. The 09-29 Jaccard rule
(66–136 pairs on object axes) is retired: its D1 "+.5 lift" readings do not survive the new rule.

**D0 — readings, S (3-seed mean; seeds with CI excluding 0 / 3):**

| dataset | arm | codeword, lexical | codeword, caption-cosine (training relation) | **codon, lexical** |
|---|---|---:|---:|---:|
| Flickr | notext | +.00 (0) | −.02 (0) | **−.06** (2 negative) |
| Flickr | base | +.02 (1) | −.06 (0) | +.08 (1) |
| Flickr | anchors (p2anc) | +.05 (0) | +.02 (0) | **+.12 (2)** |
| Flickr | ac2 (A′) | +.06 (0) | +.13 (2) | +.06 (0) |
| Flickr | bc / bq / codsoft / k64 / p1b / p5prequ | +.06 / +.04 / +.02 / −.02 / +.01 / +.00 | +.02 / +.05 / −.03 / −.07 / +.03 / +.09 | +.12 / +.11 / +.09 / +.07 / +.12 / +.08 |
| NUS | notext | −.02 (0) | −.03 (0) | −.01 (0) |
| NUS | anchors | +.02 (0) | −.01 (0) | **+.11 (2)** |
| NUS | ac2 | **+.10 (2)** | +.14 (2) | +.10 (2) |
| COCO | notext | +.01 (0) | +.02 (0) | −.02 (0) |
| COCO | anchors | +.04 (1) | +.02 (0) | +.04 (1) |
| COCO | ac2 | −.00 (0) | −.01 (0) | −.00 (0) |

Reading: at the codeword level no arm is separable from zero on 3/3 seeds; the constructed form (ac2)
reaches +.10 only on NUS. At the **codon** level the anchors arm is +.11/+.12 on Flickr and NUS (2/3
seeds each) while the text-OFF arm is ≤ 0 — the first text-vs-no-text difference seen on A3, small and
to be re-measured at the approved recipe (Stage 1).

**D2 — decision-1 assumption on the approved captions** (opt rows; P(text k-NN pair shares an
axis element), k = 10; comparison sets random / other-axis neighbours / CLIP image neighbours):

| dataset (captions) | primary | secondary | relation | colour | verdict |
|---|---|---|---|---|---|
| Flickr (V4) | .66 (×2.5 other, ×1.6 image) | .62 (×3.2, ×2.6) | .25 (×3.0, ×2.1) | .92 (random .48) | object axes pass; relation low; colour uninformative |
| NUS (V4) | .83 (×2.2, ×1.5) | .79 (×2.6, ×2.1) | .32 | .96 (random .55) | same |
| COCO (V5b) | .93 (×1.9, ×1.4) | .80 (×2.6, ×1.9) | .43 | .99 (random .83) | same; V5b adjectives dominate ("upright", "matte") |
| COCO (V4) | .69 (×1.6, **×1.1**) | .40 (×1.6, ×1.1) | .17 | .86 | fails the image-neighbour margin |

Text neighbours do share object nouns; 35–38 % of rows have no verb-like word (extractor limit + captions);
colour words are shared by half of all random pairs. Text and image neighbourhoods overlap only
.05–.15, and axis neighbourhoods overlap .03–.08.

**D3 — ceilings (no model):** S_oracle (code = own text cluster) .97 / .94 / .83 (Flickr / NUS / COCO);
**S_probe (linear probe CLIP global → text cluster) .13 [−.03,.28] / .29 [.21,.37] / .20 [.14,.26]**;
probe top-1 to 128 clusters .14–.45. One visual k-means for all slots gives S = 0 by construction.
Any text-free deployed code is bounded by S_probe; the plan's T = max(2·SD, detectable S) will sit near it.

**D4 — training routing vs deployment routing** (same validation images; training-mode forward with the
captions, EMA and revival off, vs the deployment forward):

| dataset | arm | P(same codeword) 4 axes | plan column cos | S train-routed (3 seeds) | S deployed |
|---|---|---|---|---|---|
| Flickr | base | .60/.52/.61/.64 | .60–.68 | **+.23 +.22 +.19** (3/3 CI > 0) | −.08 +.03 +.12 |
| Flickr | anchors | .60/.65/.66/.67 | .52–.57 | +.14 +.20 +.14 (2/3) | +.06 +.06 +.03 (0/3) |
| NUS | anchors | .56–.59 | .41–.44 | +.18 +.16 +.15 (3/3) | +.03 +.08 −.06 |
| COCO | anchors | .53–.56 | .47–.49 | +.14 +.17 +.24 (3/3) | −.01 +.04 +.11 |

The caption-routed codes DO carry slot-specific sharing (11/12 cells with CI > 0, S ≈ S_probe); the
deployed codes do not. Only 52–67 % of codewords survive the switch to codebook-mean anchors; pre-quant
slot tokens of different slots have cosine .70–.84 in both modes.

**D5 — the X target on existing anchors checkpoints** (opt rows, caption-routed tokens, k = 10):
top-vote share .30–.53 (visual-graph control .66–.79); vote mode ≠ own codeword 47–77 % of rows; the
modes cover 84–128 of the used codewords (no narrowing: H_mode ≥ .94·H_usage); axis neighbourhoods
overlap .04–.05, text vs visual neighbours .08–.09; cosine argmax = Euclidean argmin on .84–.89.

**D6 — legacy pruning:** keep ratio .64 (COCO 1.0, i.e. mean over all tokens); adapted anchor cosine
pruned vs EOS-pooled .64–.72 (COCO .45; per-slot minima negative); codewords change on 21–32 % of rows;
S of the caption-routed codes without pruning ≥ with pruning on Flickr (+.25/+.20/+.19 vs +.23/+.22/+.19
base; anchors +.18/+.21/+.12 vs +.14/+.20/+.14), ≈ equal on NUS, lower on COCO (ratio 1.0 there).

> **Correction (2026-10-07).** The D4 reading "the deployment lookup discards it" is replaced by the
> caption-swap control (fork session 2026-10-06 on the exploratory checkpoints; recorded re-run on the
> Stage 1 cells, `stage1/d4_control_*.json`): the caption-routed S vanishes when another image's
> captions are routed (−.01/−.06/−.04) and is absent for a text-OFF model given captions
> (+.02/−.06/−.00). The caption-routed S comes from *this image's caption being present as an input*;
> the deployment-routed token is never trained. Stage 3 was revised accordingly (plan 2026-10-07).

**Consequences for the plan.**
1. Stage 1 must add **evaluation rows**: at 500 rows only a ≥ 40 % role is visible. Score on all
   validation rows and, for Flickr, caption a fixed 1,500-image database sample as evaluation labels
   (Stage 2 generation budget); otherwise P-A3 cannot distinguish S_probe-sized effects from 0.
2. The S arm (codebook-mean routing in training, text as supervision only) moves from "attribution
   control" to **co-primary with X**: D4 shows the loss of signal happens at the train/deploy routing
   switch, not in the losses.
3. H2 (EOS anchor instead of noise-keyed token mean) stays as a cheap Stage-1 delta (D6: it changes a
   quarter of the codewords for no stated reason).
4. The 2026-10-05 D1 reading (own − other +.5 on ac2) is superseded: under the fixed lexical rule it is
   +.06 [CI incl. 0] on Flickr, +.10 (2/3) on NUS, 0 on COCO.
5. Decision 1 holds for object axes on V4/V5b; the relation axis is weak (user decision point in
   Stage 2); the colour axis needs a different pair definition or a different axis.

---

## 2026-10-05 [analysis, no training, CPU only] D1: 45 never-scored exploratory checkpoints on the A1/A2/A3 instrument — the constructed-membership form (ac2) moves own−other lift by only ≈ +.5 (Flickr) / +1 to +1.7 (NUS) and not at all on MS-COCO

**Status:** 🟡 diagnostic, exploratory (Gumbel ON checkpoints of 2026-09-22; own N; not the approved
recipe). Records: `result/analysis/textdiag_2026-10/d1/` (`runs.txt`, `run_batch.sh`, `a2a3/*.json`
and logs for 45 runs), tmux `textdiag_d1b` (rc 0, 1,629 s, end 2026-10-05 20:35). Instrument =
the unchanged 09-29 script (500 validation rows, own cache captions); CIFAR-10 excluded.

**own − other lift** (own-slot lift minus the mean lift of the other three slots on the same pairs,
averaged over the four axes; Jaccard ≥ .25 rule / caption-cosine top-2 % rule; 3-seed mean, per-seed
signs in brackets). The 09-29 arms are repeated for comparison.

| dataset | arm | A2 codeword→own-axis | own − other (Jaccard) | own − other (cosine) |
|---|---|---:|---:|---:|
| Flickr25K | base (09-29) | .307 | −.42 (−/−/−) | −.10 (−/+/−) |
| Flickr25K | anchors (09-29) | .370 | +.34 (−/+/+) | +.15 (+/+/+) |
| Flickr25K | notext (09-29) | .252 | +.68 (+/+/−) | −.03 (−/+/+) |
| Flickr25K | **ac2** (A′) | .32 | +.62 (+/+/+) | **+.56 (+/+/+)** |
| Flickr25K | ancsoft | .38 | +.65 | +.13 |
| Flickr25K | bc / bq / codsoft | .33 / .28 / .27 | +.35 / +.54 / −.04 | +.16 / +.12 / −.01 |
| Flickr25K | k64 / p1b / p5prequ | .29 / .30 / .45 | −.04 / −.03 / +.63 | +.01 / +.19 / +.17 |
| NUS-WIDE | anchors (09-29) | .444 | +.00 (+/−/−) | +.04 (−/+/+) |
| NUS-WIDE | notext | .24 | +.10 (+/−/+) | −.21 (−/−/−) |
| NUS-WIDE | **ac2** | .36 | **+1.74 (+/+/+)** | **+.77 (+/+/+)** |
| NUS-WIDE | k64 | .37 | +.33 | +.16 |
| MS-COCO | anchors (09-29) | .418 | +.85 (−/+/+) | +.19 (−/+/+) |
| MS-COCO | notext | .24 | −.06 | +.02 |
| MS-COCO | **ac2** | .26 | −.32 (+/−/−) | **−.26 (−/−/−)** |
| MS-COCO | k64 | .38 | +.29 | −.03 |

**Reading.**
1. The constructed form (text clusters assign EMA membership) is the only arm with own − other
   positive on 3/3 seeds under both rules on Flickr25K and NUS-WIDE. The size is small: ≈ +.5 on
   lifts of 7–10 (Flickr), +.8 to +1.7 on NUS. On MS-COCO it is negative on 3/3 seeds, matching the
   stage-8 verdict that A′ fails there.
2. `bq` lowers every lift (own 3.3–4.8): the queue target crowded the codebook (dead .49) rather
   than organising it. `codsoft` and `k64` are at zero. `p5prequ` raises A2 (.45) but halves the
   lifts.
3. This is the ceiling the plan's Stage 0 asked for: with text assigning membership directly, the
   architecture can hold a slot-specific signal, but a weak one. Whether +.5 is above the noise of
   the instrument is exactly what A3 v2 (D0: bootstrap intervals, lexical pair rule, all validation
   rows) must decide; the plan's stop condition ("ac2 does not move A3 AND S_probe is at noise") is
   not triggered on Flickr/NUS and is open on MS-COCO pending D3.

Descriptive, n = 3 seeds, no test run. Nothing under `arch-exp-2026-09`, the anchor worktrees or any
running campaign was touched; no GPU.

---

## 2026-10-05 [design record, no results] Approved modification plan: weakest intervention first; codebooks read as prototype banks; loss-term review (14 nominal weights, 10 real terms)

**Status:** 🟢 active plan, approved by the user on 2026-10-05. No measurement in this entry. Sources
read: `model_siglip2.py`, `loss_siglip2.py`, `config.py`, the three approved ancS7 seed-42
`args.txt`/`log.csv` (read only; no checkpoint loaded), `docs/PROJECT_LOG.md`, CTRL-O (arXiv
2503.21747) and ALBM (arXiv 2503.20301) as fetched. Ten read-only agents (explore → design → refute);
deciding lines re-read directly.

**Hypothesis under test.** Axis-wise text supervision makes (a) slot *m*'s code carry axis *m*'s
element, (b) images sharing an axis-*m* element share the slot-*m* codeword/codon **in slot *m***,
and (c) this is caused by the text.

**Causes confirmed in code.**

| # | fact | where |
|---|---|---|
| 1 | per-slot instance NT-Xent (λ 1.0–1.5, all 5 slots) asks every slot to identify the image alone | `loss_siglip2.py:1555-1596` |
| 2 | the three text terms compare slot *m* only with the SAME image's caption *m*; caption similarity enters only as a per-pair temperature | `loss_siglip2.py:1578-1584`, `:3254-3267` |
| 3 | codebooks are EMA buffers updated by nearest-codeword assignment of visual tokens; text never updates them | `model_siglip2.py:1180-1192` |
| 4 | routing uses per-image caption anchors in training and one constant codebook-mean anchor at deployment | `model_siglip2.py:3807-3813` |
| 5 | a local codon reads `q_local + σ(4.595)·sg(q_global)` (gate ≈ .99) | `model_siglip2.py:5262-5266` |
| 6 | legacy token pruning has a constant importance (code comment), so the training anchor is the mean of an arbitrary ≈ 77 % subset of caption tokens, not the EOS-pooled vector | `model_siglip2.py:3906-3926`; approved `log.csv` keep ratio .769 |

**Codebook view (user question).** Prototype bank, not VQ-VAE latent and not Slot-Attention slot:
the contrastive loss acts on the pre-quantisation token, the code receives only commitment and EMA,
each codebook is an online k-means of its slot's tokens, and `--router_type slot` collapsed
(commit `ced8546`: dead .688). Consequence: meaning = membership, so text must influence membership.

**Verdicts on the user's ideas.**

| idea | verdict | reason |
|---|---|---|
| ALBM dataset-specific axes + Description → Summary → Supplement | adopt, adapted to unlabeled data | fixes the missing repeatable per-axis phrase; one field per axis |
| CTRL-O decoder conditioning | conditional (Stage 5) | CTRL-O's own ablation: binding hits 8.1 (init) → 10.1 (+decoder) → 56.3 (+contrastive) → 61.3; literal `[slot ; caption]` input lets the decoder read the element from a caption absent at deployment |
| ALBM visual attribute prompt learning | rejected as a first phase | published loss needs class labels; needs online CLIP or a 12-layer cache; learned text-free attention queries failed four times here (v7, v79d, v107a_attn, v146) |

**Stage 3 mechanism (X).** Replace the target of `text_code_kl` (term count unchanged): the
codeword distribution currently held by the image's axis-*m* text neighbours (top-10 by per-axis-
centred EOS caption cosine, opt-train rows), neighbours' stored slot tokens re-assigned against the
current codebook, usage-corrected, Euclidean-distance logits, view 1, local slots, λ .05. Controls:
X-perm (axes permuted), X-vis (per-slot visual neighbours), S (codebook-mean routing in training).
Earlier soft text targets failed at other levels (`bc` inert, `bq` dead .492, `codsoft` −.022 mAP).

**Pre-registered rules (to be frozen per stage in `PREREGISTRATION.md`).** R(m) = log(own-slot lift /
other-slot lift), S = mean over axes; P-A3 (S > 0 on 3/3 seeds, ≥ 3/4 axes), P-DELTA (vs own base),
P-TXT (vs `--disable_text_supervision`), P-RET (mean val mAP@R ≥ base − .008; −.008 to −.03 is the
user's call), P-HEALTH. Primary pair rule is lexical (held-out rows), not the training relation.

**Loss-term review.** `train_loss` equals Σ λ·term within 4e-8 on all three runs.

| class | terms | handling |
|---|---|---|
| unnecessary (zero gradient) | `anchor` .05, `cibhash_kl` .001, `codon_text_anchor` .1, `recon` 1.0; the codebook half of `vq` (description only) | set to 0 without training; confirm by `scripts/audit_loss_gradients.py` and a bit-identity pair in Stage 1 |
| overlapping, removal candidates | `quant` ↔ entropy half of `dna`; base-balance half of `dna` ↔ `codon_joint`; visual half of `xmodal_commit` ↔ `text_code_kl` | one single-delta cell each after Stage 1 (`--lambda_quant 0`, then `--eta_base_balance 0`); text pair in Stage 3/4 |
| keep | `cibhash_ntxent`, `text_hash_ntxent`, `xmodal_commit`, `codon_joint`, `wasserstein`, `vq` commitment (.0625), `dna` entropy, `bu`, `text_code_kl` (target replaced) | `bu` removal was already rejected at Gumbel OFF (09-15, 1 seed: .7636 → .7540, dead +.069) |

**Corrections to earlier records.**
- The 09-29 entry's "two generations of the same image agree on 12–24 % of content words" compares
  V4 with V5b (two different prompts, greedy decoding), not two generations of one prompt.
- "Per-slot NT-Xent is 82 % of the objective" (PL 09-22) is 54.4 % contrastive + 28.1 %
  `text_hash_ntxent` from the 2026-08-12 gradient audit (`docs/loss_function_summary.md`).
- The 10-01 next step "two-level captions (label + detail)" is `[reverted 2026-10-04]` (user: no
  label/detail split).
- Tool traps found: every caption tool imports the prompt from `/home/yschoi/GroundedDNA`;
  `tools/qwen3_v5b_flickr25k_trainset.py` now emits `_PROMPT_V8`, which has a 500-image sample and no
  training record.

**Next:** Stage 0 (CPU): D1 scoring of 45 never-scored exploratory checkpoints started 2026-10-05.

---

## 2026-10-01 [design record, no results] Literature survey for the two levers: repeatable per-axis caption tokens and structural blocks on slot redundancy; encoder replacement declined

**Status:** 🟡 design + literature record. No measurement in this entry.

**Encoder replacement — declined.** Binding failure (attribute↔object) is common to all dual
encoders: *The Limits of Binding in Dual Encoders* (arXiv 2608.15971) finds 18 text encoders at
25–35 % of their theoretical ceiling and locates the cause in the contrastive incentive and code
structure; *Auto-Comp* (arXiv 2602.02043) reports the same failure class across 25+ CLIP/SigLIP/
hard-negative/generative models. Our own records agree: SigLIP2 text ≈ CLIP text after whitening
(47.1 vs 47.3 %, 2026-07-20), SigLIP2 vision −0.114 mAP on the same recipe, FG-CLIP text worse
(40.3 %), FG-CLIP vision collapsed (2026-06-21), a text-only swap is impossible because routing
needs one shared image–text space, and a full swap re-runs the experiments section (4 caches,
4 seals, ours ×4, baselines ×12). A CPU pre-check (SigLIP2 vs CLIP on our own axis captions: same-
element pair cosine gap and axis separability) is the only step kept open.

**Lever 1 — repeatable per-axis tokens in the captions.** Related work: label-free concept
bottlenecks that build a *fixed concept vocabulary* and score images against it (LaBo, CVPR 2023;
Label-free CBM; *Explain via Any Concept*, ECCV 2024; attribute-formed concept spaces, CVPR 2025);
structured JSON captions with predefined semantic fields that improve consistency over free prose
(OS-W2S; VLM-Run caption&tag); caption filtering/soft targets for noisy text (BLIP CapFilt; ALBEF);
multi-sample self-consistency as a hallucination filter (arXiv 2509.23236; MRFD). Design implied:
per axis `label` (1–3 words from a controlled, lemmatised vocabulary; repeats across images) +
`detail` (10–15-word grounding sentence, V4 style); confidence from label agreement across two
generations, replacing the entropy-only confidence in `text_code_kl`.

**Lever 2 — structural block on slot redundancy.** Related work: Slot Attention's softmax over
slots makes slots *compete* for each input element (NeurIPS 2020); grounded slot dictionaries bind
object types to canonical slots (ICLR 2024); CTRL-O conditions slots on language queries (CVPR
2025); expert-choice routing removes the MoE balancing loss by letting each expert pick its tokens
(arXiv 2202.09368); Semantic VQ / factor-quantised VAEs give each factor its own codebook block and
report that disentangled factors need c + s codes instead of c × s (PMLR v243; NeurIPS 2023
"Disentanglement via Latent Quantization"; arXiv 2409.14851); Concept Whitening aligns latent axes
with concepts through a whitening + rotation module (2020); additive/block-diagonal decoders give
identifiability of latent blocks (arXiv 2307.02598). Design implied, in the order fixed on
2026-09-20 (fewest added losses first): slot-choice routing (each slot selects its patches),
block-diagonal caption decoder (slot *m* reconstructs only axis *m*; replaces three text terms),
factor-wise codebooks with an independence penalty only if the structural options fail.

**Next step proposed (not started):** (1) CPU pre-check of the encoder question; (2) V9 two-level
caption pilot on 500 Flickr25K images with the A0 acceptance table (label repeat ≥ 60 %, two-
generation label agreement ≥ 80 %, leak ≤ 25 shared words, local↔local cosine .50–.60); (3) the A3
metric with a text-OFF control at the final anchor recipe (needs admission); (4) one pre-registered
Flickr25K 3-seed arm per lever, judged on A3 own−other lift and text-OFF gap before mAP@R.

---

## 2026-09-29 [analysis, no training, CPU only] Text-path diagnostics A0/A1/A2/A3: caption-similar images share codes 4–14× more often than random pairs, but equally in every slot and equally without any text supervision

**Status:** 🟡 diagnostic, exploratory. Records: `result/analysis/textdiag_2026-09-29/`
(`a0_caption_stats.{py,json}`, `a2a3_slot_consistency.py`, `a2a3/*.json` for 18 runs,
`A2A3_SUMMARY.md`), commit `56ba19b`. All forwards on CPU (`nice`), no GPU; nothing under
`arch-exp-2026-09`, the anchor worktree, the seals or the then-running stage-D campaign was touched.

> **Recipe label:** A1–A3 use the September exploratory anchor checkpoints (`use_gumbel_softmax=True`,
> own N: CIFAR-10/Flickr25K/NUS-WIDE N=4, MS-COCO N=39; `/data/yschoi/gdna_wt_mscoco/result`,
> `/data/yschoi/gdna_archexp_result`) and the Flickr25K `base` / `notext` runs. The audited ancS7
> checkpoints were not loaded. Same-recipe numbers for the approved model need a separate admission.

**A0 — the four approved caption files** (V4: CIFAR-10, Flickr25K, NUS-WIDE; V5b: MS-COCO).

| dataset | rows | local↔local CLIP text cosine | top-100 words shared, primary/secondary | color axis top-50 word coverage | object axes top-50 coverage | v4↔v5b word Jaccard (same images) |
|---|---:|---:|---:|---:|---:|---|
| CIFAR-10 | 6000 | .689 | 34 | .66 | .43–.45 | — |
| Flickr25K | 5000 | .600 | 51 | .56 | .26–.27 | primary .19, secondary .17, activity .12, color .23 |
| NUS-WIDE | 10500 | .617 | 50 | .61 | .27–.33 | — |
| MS-COCO | 10000 | .583 | 42 | .59 | .29–.39 | primary .23, secondary .20, activity .14, color .24 |

- 'none' 0 % everywhere; exact duplicate sentences ≤ 1.5 % (CIFAR) and ≤ 0.1 % elsewhere. The
  'none' problem is historical (V3, May 2026).
- Object and activity captions are image-specific (top-50 content words cover 24–39 % of tokens);
  two generations of the same image agree on 12–24 % of content words. No repeatable concept token
  per axis exists on the text side.

**A1 — the `text_code_kl` target on the validation rows** (τ_t .07, threshold .2; mean over the 4
local slots, 3 seeds). `text_code_kl` is the confidence-weighted KL of the visual codeword
distribution onto the same image's caption distribution designed on 2026-06-11 (v144a) and present
in the approved recipe at λ = 0.05.

| dataset / arm | mean confidence | share excluded (≤ .2) | text argmax = visual codeword | text top-1 mass |
|---|---:|---:|---:|---:|
| CIFAR-10 anchors | .526 | .006 | .093 | .351 |
| Flickr25K anchors | .314 | .139 | .098 | .181 |
| Flickr25K base | .414 | .014 | .152 | .260 |
| Flickr25K notext | .026 | 1.000 | .011 | .024 |
| MS-COCO anchors | .359 | .079 | .116 | .233 |
| NUS-WIDE anchors | .377 | .026 | .106 | .211 |

The target is active (6–14 % excluded, except CIFAR) but nearly flat (top-1 mass .18–.35 over
K = 64/128) and picks the image's actual codeword in 9–15 % of samples. `notext` is the sanity
control (100 % excluded).

**A2 — code→own-axis (chance .25) and geometry**, 500 rows, 3 seeds:

| dataset / arm | pre-quant slot token | codeword | eff-rank of z | cos(z, q) | codewords used / K |
|---|---:|---:|---:|---:|---|
| CIFAR-10 anchors | .273 ± .030 | .332 ± .051 | 16.1 | .694 | 42 / 64 |
| Flickr25K anchors | .273 ± .017 | .370 ± .014 | 46.7 | .661 | 81 / 128 |
| Flickr25K base | .267 ± .003 | .307 ± .007 | 32.8 | .708 | 77 / 128 |
| Flickr25K notext | .249 ± .004 | .252 ± .004 | 45.5 | .684 | 85 / 128 |
| MS-COCO anchors | .260 ± .010 | .418 ± .023 | 71.1 | .657 | 109 / 128 |
| NUS-WIDE anchors | .293 ± .011 | .444 ± .018 | 43.4 | .708 | 110 / 128 |

The pre-quantisation slot token is at chance on every arm; the within-image axis alignment the
text path produces lives in the codeword (.31–.44 with text, .25 without).

**A3 — cross-image slot consistency.** Pairs of the 500 rows whose axis-*m* captions share content
(word Jaccard ≥ .25; or top 2 % caption CLIP cosine); lift = P(same codeword | pair) / P(same
codeword | any pair), in the OWN slot and in the OTHER three slots on the same pairs (3-seed
mean; axes primary / secondary / activity / color; full tables with SDs in `A2A3_SUMMARY.md`).

| dataset / arm | rule | own-slot lift | other-slots lift (same pairs) | own − other |
|---|---|---|---|---|
| Flickr25K anchors | Jaccard ≥ .25 | 9.0 / 6.5 / 9.7 / 4.9 | 8.9 / 7.6 / 7.5 / 4.8 | +.1 / −1.1 / +2.2 / +.1 |
| Flickr25K base | Jaccard ≥ .25 | 7.8 / 5.7 / 8.2 / 4.7 | 7.3 / 7.4 / 8.8 / 4.5 | +.5 / −1.7 / −.6 / +.2 |
| Flickr25K **notext** | Jaccard ≥ .25 | 10.5 / 9.0 / 9.7 / 4.9 | 9.6 / 7.8 / 9.1 / 4.9 | +.9 / +1.2 / +.6 / −.0 |
| CIFAR-10 anchors | Jaccard ≥ .25 | 8.1 / 3.7 / 4.4 / 4.2 | 8.3 / 4.2 / 3.9 / 4.1 | −.2 / −.5 / +.5 / +.0 |
| NUS-WIDE anchors | Jaccard ≥ .25 | 14.6 / 5.6 / 9.9 / 4.8 | 11.6 / 6.4 / 12.4 / 4.6 | +3.0 / −.7 / −2.5 / +.2 |
| MS-COCO anchors | Jaccard ≥ .25 | 8.6 / 13.7 / 6.5 / 3.8 | 7.9 / 11.4 / 6.3 / 3.5 | +.7 / +2.3 / +.1 / +.3 |

The caption-cosine rule gives the same picture at lifts 2.5–7.9. Base-Hamming of the own-slot
codon drops by .4–1.2 of 3 bases for paired images on every arm, including `notext`.

**Reading.**
1. 🟢 "Similar caption → same code" holds: 4–14× on every dataset.
2. 🔴 It is **not slot-specific**: the other three slots move by the same factor (own − other
   within the seed SD, signs mixed). This is the image-level redundancy measured on 2026-09-20
   (2.97 of 4 slots per patch), seen now on the codes themselves.
3. 🔴 It is **not caption-dependent**: the model trained with no text has the same lifts. The
   consistency comes from the frozen CLIP features.
4. A1 shows why the designed lever is weak (flat target, ~10 % agreement); A0 shows the captions
   carry no repeatable per-axis token such a target could lock onto.

**Consequence.** With the current model, "the text path makes images with the same element share
the slot's code" cannot be stated. Any repair must first produce slot-specific, text-caused sharing
on this metric with a text-OFF control at the final recipe. Descriptive, n = 3 seeds, no test run.

---

## 2026-09-29 [design record, no results] What the text path can and cannot be credited with, from the existing evidence

**Status:** 🟡 record of the evidence review that opened this line (sources: PROJECT_LOG entries of
2026-09-20/21/22, the approved recipe `args.txt`, `loss_siglip2.py`).

| property asked for | evidence | credited to the text path? |
|---|---|---|
| a slot's codon reads out the axis's caption words (same codon ≈ similar element) | .354 vs chance .227 | no — identical with text OFF (.357); inherited from frozen CLIP (2026-09-21 necessity ablation) |
| §4.7 codon decoding beats the flat hash | +.10 to +.15 | mostly no — the text-OFF model still beats it by ≈ +.11; text adds decodability only on MS-COCO (F09 A2) |
| slots carry their own element (meaning accumulates across positions) | five codons ≈ flat hash; cross-slot decoding ≈ .36 everywhere | no — every slot carries image-level meaning (2026-09-21) |
| codewords are nameable by nearest captions | axis specificity .25 = chance | no (stage 10, negative) |
| within-image codeword → own-axis alignment | .3075 with text vs .2515 without; .370 with anchors | **yes**, modest; the one caption-dependent effect |
| reading the code through the model's own caption path | +.022 over the best caption-free reader (Flickr, 3 seeds) | **yes**, one dataset |

Structural reason (2026-09-20 design record): none of the twelve loss terms compares axis *m* with
axis *m′* of the same image or across images; a solution in which every slot carries the whole image
satisfies all alignment terms, and the model finds it. Ten mechanisms measured against the role
metric (routing windows, gates, anchors, readout centring, in-image cross-axis loss on pre-quant and
quantised tokens, concept codebook, codon-level axis target) moved their proximal target and not the
role; the only loss-free gain was the anchors. `text_code_kl` is removable (P4drop: −.004 mAP within
the seed SD, role unchanged).

---
