# GroundedDNA (A-champion) vs baselines — 18-base main + 24-base diagnostic, P0 + mandatory bio-projection

**Metric.** base-Hamming **mAP@R** after mandatory bio-constraint projection (GC count [8,10], homopolymer ≤ 3)
in the **18-base / 36-bit** DNA code space. Cutoffs: CIFAR10@1000, others@5000. Higher is better.

**Ours = "A" champion** (strict global-caption-free recipe): champion recipe + 4 global-slot skips
(`text_code_kl`, `text_hash_ntxent`, `xmodal_commit`, `cibhash_dynamic_tau`) + local-only whitening, at
K=128/L=3 (CIFAR K=64/L=3), seed 42. Per-dataset prompt = the A-recipe optimum
(**MSCOCO V5b; Flickr/NUS/CIFAR V4** — see the 2026-07-27 PROJECT_LOG entry).

**Information-tier legend.** These are repository-local audit tags, not names
claimed verbatim by every source paper.

- `VLM-T`: target ground-truth labels and benchmark taxonomy do not enter the
  representation objective, but VLM-generated instance captions and a frozen
  text encoder provide text supervision. GroundedDNA belongs here, not in
  visual-only `U0`.
- `U0`: visual-only unsupervised encoder objective; target labels, class names,
  captions, and external text banks are absent.
- `U0-FD`: native-DNA subtag of `U0`; pseudo-pair targets come from frozen
  optimization-train visual-feature distances.
- `U2`: benchmark-taxonomy-assisted without instance labels. `S`: target
  ground-truth train labels enter the objective.

For `VLM-T`, `U0`, and `U0-FD`, held-out labels are used only to score
validation retrieval and select \(E^*\); they do not enter the encoder
objective. Thus “target-label-free” describes representation learning, not a
label-blind end-to-end evaluation protocol.

> ⚠️ **All numbers on both sides are single-seed diagnostics** unless marked 3-seed. Under the current
> sealed-strict definition the strict 3-seed MAIN table is still unfilled for every method, ours included.

---

## Table 1 — Target-label-free encoder-objective comparison: `VLM-T` vs `U0`/`U0-FD`

| Method | tier | seeds | Flickr25k | MSCOCO | NUS-WIDE | CIFAR10 |
|---|:---:|:---:|---:|---:|---:|---:|
| **Ours (A-champion)** | **VLM-T** | 1 | **0.8675** | 0.8170 | **0.8262** | **0.9058** |
| *Ours (non-A champion, for reference)* | **VLM-T** | 1 | *0.8723* | *0.8063* | *0.8274* | *0.9009* |
| **— classic 3 (same eval pipeline) —** | | | | | | |
| CIBHash | U0 | 1 | 0.7914 | 0.7764 | 0.7901 | 0.8933 |
| CIMON | U0 | 1 | 0.8168 | 0.6583 | 0.7774 | 0.8221 |
| MLS3RDUH | U0 | 1 | 0.7666 | 0.6289 | 0.7647 | 0.5788 |
| **— modern U0 (legacy-cache diagnostic) —** | | | | | | |
| OH (Hashing One With All) | U0 | 1 | 0.8362 | 0.7587 | 0.8023 | 0.8737 |
| CroVCA | U0 | 1 | 0.7682 | **0.8257** | 0.7944 | 0.8819 |
| SDC | U0 | 1 | 0.7230 | 0.8185 | 0.7520 | 0.8442 |
| Bi-half | U0 | 1 | 0.8161 | 0.7062 | 0.7489 | 0.7581 |
| GreedyHash | U0 | 1 | 0.6077 | 0.5639 | 0.6511 | 0.1851 |
| HHCH | U0 | 1 | 0.5867 | 0.4709 | 0.4329 | 0.2992 |
| **— native-DNA `U0-FD` direct predecessors (sealed, 3-seed) —** | | | | | | |
| DNA24-18 analytic-transfer (Stewart 2018) | **U0-FD** | 3 | 0.7808±.0071 | 0.6330±.0127 | 0.7427±.0055 | 0.7786±.0101 |
| PRIMO-18 frozen-predictor length-transfer (Bee 2021)§ | **U0-FD** | 3 | 0.7882±.0209 | 0.6251±.0129 | 0.7320±.0109 | 0.7344±.0123 |

§ PRIMO additionally uses the official 80-nt frozen predictor at the reported
shorter length; this is a length-transfer adaptation, not a length-calibrated
PRIMO reproduction. This symbol is independent of `†`, which the authoritative
native-DNA aggregate documents reserve for diagnostic-only cells.

**Margins for Ours (A-champion):**

| vs | Flickr25k | MSCOCO | NUS-WIDE | CIFAR10 |
|---|---:|---:|---:|---:|
| best **classic** baseline | **+0.0507** (CIMON) | **+0.0406** (CIBHash) | **+0.0361** (CIBHash) | **+0.0125** (CIBHash) |
| best **U0-FD native-DNA** predecessor | **+0.0793** (PRIMO) | **+0.1840** (DNA24) | **+0.0835** (DNA24) | **+0.1272** (DNA24) |
| best **any-U0** (incl. modern) | **+0.0313** (OH) | **−0.0087** (CroVCA) | **+0.0239** (OH) | **+0.0125** (CIBHash) |

🟢 GroundedDNA (`VLM-T`) **outperforms every `U0`/`U0-FD` baseline on 3/4
datasets**, and wins **4/4 against the classic three and both `U0-FD`
native-DNA direct predecessors**. This is a cross-tier target-label-free
comparison, not a claim that GroundedDNA itself is `U0`.
🔴 **MSCOCO is not a clean win**: CroVCA (0.8257) and SDC (0.8185) exceed our 0.8170. Adopting A narrows this
gap (non-A was 0.8063, i.e. −0.019 → −0.009) but does not close it. State this honestly.

---

## Table 2 — Taxonomy-assisted and supervised tiers (excluded from the U0 margin)

| Method | tier | seeds | Flickr25k | MSCOCO | NUS-WIDE | CIFAR10 |
|---|:---:|:---:|---:|---:|---:|---:|
| UMRCH | U2 (taxonomy) | 1 | 0.7994 | 0.8009 | 0.8224 | n/a |
| Koike DATE/DAC 2024 | **S** | 3 | 0.8353±.0047 | 0.5795±.0151 | 0.7453±.0419 | 0.8850±.0090 |
| Koike TCBB 2026 | **S** | 3 | 0.8916±.0037 | 0.6189±.0045 | 0.8156±.0009 | 0.9304±.0054 |
| CRH (Codebook-Centric, AAAI 2026) | **S** | 3 | 0.8628±.0068 | 0.8435±.0013 | 0.8532±.0025 | 0.9346±.0019 |

Supervised methods use ground-truth labels and must never be placed inside the
U0 baseline margin. CRH and Koike-TCBB exceed GroundedDNA on several datasets;
those rows answer a different information condition and are reported without
being folded into the target-label-free comparison claim.

---

## Table 3 — 24-base (48-bit), bio-projected. Compare only within this budget.

| Method | tier | seeds | Flickr25k | MSCOCO | NUS-WIDE | CIFAR10 |
|---|:---:|:---:|---:|---:|---:|---:|
| Ours K=128 (non-A)‡ | **VLM-T** | 1 | 0.8742 | **0.8257** | **0.8328** | **0.9033** |
| Ours K=64 (non-A)‡ | **VLM-T** | 1 | **0.8762** | 0.8251 | 0.8313 | 0.9013 |
| CIBHash | U0 | 1 | 0.8057 | 0.8018 | 0.8074 | 0.8994 |
| CIMON | U0 | 1 | 0.8277 | 0.6723 | 0.7858 | 0.8231 |
| MLS3RDUH | U0 | 1 | 0.7670 | 0.6294 | 0.7765 | 0.5694 |
| DNA24-24 analytic-transfer | **U0-FD** | 3 | 0.7927±.0068 | 0.6412±.0110 | 0.7542±.0046 | 0.7814±.0090 |
| PRIMO-24 frozen-predictor length-transfer§ | **U0-FD** | 3 | 0.7962±.0125 | 0.6326±.0049 | 0.7522±.0033 | 0.7569±.0085 |
| CRH-24 | **S** | 3 | 0.8816±.0068 | 0.8668±.0011 | 0.8634±.0037 | 0.9383±.0019 |
| Koike DATE/DAC-24 | **S** | 3 | 0.8167±.0071 | 0.5788±.0203 | 0.7598±.0113 | 0.8812±.0049 |
| Koike TCBB-24 | **S** | 3 | 0.8939±.0043 | 0.6359±.0128 | 0.8285±.0016 | 0.9295±.0014 |

‡ Ours-24-base is the **non-A** grid; its main-protocol eligibility was revoked 2026-07-23 (test not sealed
exact-once; Phase-2 contained test-informed CIBNT selection). **No A-recipe 24-base run exists.** Use for
direction only.

---

## Protocol notes (must accompany the table)

1. **Ours-A is better on 2/4, worse on 2/4 vs non-A**: MSCOCO +0.0107, CIFAR +0.0049, Flickr −0.0048,
   NUS −0.0012 (mean +0.0024). A was adopted for recipe consistency with the sealed strict-P0 direction,
   not because it wins everywhere.
2. **Classic-3 vs modern-U0 numbers come from different run profiles** and must not be mixed within a column
   (`bio_projection_comparison.json` vs `baseline_p0_matrix_seeds42_legacy_cache.json`).
3. Every 18-base cell here is bio-projected (post-projection compliance 1.0) and P0 val-selected.
4. Single seed except the native-DNA and supervised rows (3 seeds, mean±std).
5. A strict paper MAIN table requires 3-seed sealed reruns on **both** sides.

## Sources

`docs/bio_projection_comparison.json` (Ours-non-A + classic-3, 18-base) ·
`docs/baseline_24base_dnaeval_all.json` (classic-3, 24-base) ·
`docs/baseline_p0_matrix_seeds42_legacy_cache.{json,md}` (modern U0 + UMRCH) ·
`docs/native_dna_p0_aggregate.md`, `docs/native_dna_p0_24base_aggregate.md` (DNA24/PRIMO/Koike) ·
`docs/baseline_p0_matrix_seeds42-43-44_supervised_legacy_cache.md` (CRH) ·
A-champion cells: `result/2607{24,27}+*promptAblA_{flickr_A_v4,mscoco_A_v5b,nuswide_A_v4,cifar_A_v4}_P0refit_*`
(`cell_result.json`) · PROJECT_LOG 2026-07-27 (prompt × recipe interaction), 2026-07-21 (bio-projection).
