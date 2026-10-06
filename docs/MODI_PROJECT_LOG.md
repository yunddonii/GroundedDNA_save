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
