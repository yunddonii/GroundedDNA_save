# GroundedDNA — Project Log

Living record of *important* design decisions, experiment results, and
infrastructure issues for the GroundedDNA deep-hashing project. Append a
new section whenever a non-trivial change lands (new architectural variant,
new ablation, dataset/cache change, baseline result, etc.). Keep entries
short and dated; if a decision was later reverted, mark it `[reverted YYYY-MM-DD]`
rather than deleting — the reasoning is the value.

Format conventions:
- **Date** = YYYY-MM-DD (absolute; never "yesterday").
- **Status** at top of each section: 🟢 active / 🟡 superseded / 🔴 reverted.
- Tables are preferred over prose when reporting numbers.
- "v6" etc. without dataset prefix = the Flickr25k run unless noted.

---

## 2026-05-14 — v23a (mean-pool text) reverted, v23b (no global gate) → Pareto step

🔴 v23a reverted · 🟢 v23b kept as available ablation.

**v23a** — replace SigLIP2's `pooler_output` with the mean of valid token
features as the per-slot pooled text embedding. Hypothesis: pooler is
last-token pool and lands on near-identical vectors per image (cos 0.88).
Token mean is empirically more diverse (cos 0.78), so this should propagate
to more discriminative routing centroids.

Result: stopped mid-training around epoch 30 — mid-eval mAP tracked
v20-K64 within 0.005 but unique stayed in the same 0.10 band. Effect was
not large enough to be worth a full 60-epoch run, given v23b was already
queued. The (small) `--text_pool_mode mean` code path was removed from
the source tree on revert.

**v23b** — disable the gated C_0 → C_1..5 codeword addition before the
codon heads. Each local codon head now sees its own quantised codeword in
isolation (no global blending). New CLI flag `--disable_global_gate`.

Setup: v20-K64 base (K=64, hashnet binary, siglip2_global) + only the
new flag.

Result (Flickr25k, single seed):

| run                                    |   mAP   | unique  | dup    | dead  | pos→neg gap |
|----------------------------------------|--------:|--------:|-------:|------:|------------:|
| v20-K64 (with C_0 gate)                | 0.7892  | 0.0982  | 0.9018 | 0.005 | 5.20 |
| **v23b (no C_0 gate)**                 | **0.7877** | **0.1107** | 0.8893 | 0.005 | 5.05 |
| Δ                                      | −0.0015 | **+0.013** | −0.013 |  0    | −0.15 |

- mAP regression is within noise (0.0015 / 0.78 ≈ 0.2%).
- Unique codes: **2,256 → 2,556** for the same 23K DB → +300 distinct
  hashes (+13.3%) at essentially zero retrieval cost.
- Per-codebook perplexity unchanged (mean 52.2 → 51.8), so the gain isn't
  from reviving dead codewords — it's purely from undoing the redundancy
  the C_0 addition was injecting into every local codon head.

This is the first ablation that genuinely moves the Pareto frontier on
the (mAP, unique) plane near our best operating point. The codebook
geometry diagnostic earlier in the log warned about exactly this: adding
the C_0 codeword vector (whose codewords are nearly orthogonal, low
effective rank 2.5) to each local codeword (clustered around its own
centroid, low effective rank 3.4) was contaminating the local hashes
with a shared additive offset and reducing the count of distinct hash
patterns the model could produce.

Decision: keep `--disable_global_gate` in the tree as a recommended
toggle for "compositional-code-prioritising" runs. v20-K64 (with gate)
keeps the absolute mAP record by 0.0015; v23b is the better choice when
unique-code count matters for the paper claim.

Pushed to `github.com:yunddonii/GroundedDNA_save` (`main`).

---

## 2026-05-14 — v24a (Wasserstein restored + no gate), v24b (Jaccard + no gate)

🟢 v24b is a **new Pareto point** (mAP > v6, unique 3x v23b). v24a is
inactive (no net benefit over v23b).

Goal: revisit two previously-killed loss/target choices now that v23b has
opened the (mAP ≈ 0.788, unique ≈ 0.11) corner of the frontier without a
C_0 gate.

### v24a — Wasserstein restored, K=64, no gate

R4's `lambda_wasserstein` (v11 sweet spot 0.05) was removed during the
2026-05-13 cleanup. The Sinkhorn router still computes `ot_cost`; we
re-added the loss term plus its config / train-types entry. Smoke test
confirmed gradient flow.

| run                            |   mAP  | unique | dup    | dead  | pos→neg gap |
|--------------------------------|-------:|-------:|-------:|------:|------------:|
| v23b (no gate, binary S)       | 0.7877 | 0.1107 | 0.8893 | 0.005 | 5.05 |
| **v24a (v23b + Wasserstein 0.05)** | 0.7881 | 0.0734 | 0.9266 | 0.000 | **5.38** |

mAP is statistically tied with v23b (+0.0004). The Wasserstein term
widened the pos→neg Hamming gap from 5.05 to 5.38 (best of any variant
so far), but it collapsed the per-image unique count: 0.111 → 0.073, the
LOWEST in the v18-derived line. Net result: the extra OT alignment
trades unique codes for margin without buying any mAP. Not Pareto-
improving on v23b. `--lambda_wasserstein` stays in the tree but
defaults to 0.

### v24b — Jaccard target + no gate, K=64

The other never-combined ablation: replace binary `S = any-shared` with
fractional `S = Jaccard`, on top of v23b (no gate). Hypothesis: with the
gate gone, the model has the structural room to honour Jaccard's
graduated target probabilities instead of collapsing all positive pairs
onto a single cluster.

| run                            |   mAP  | unique | dup    | dead  | pos→neg gap |
|--------------------------------|-------:|-------:|-------:|------:|------------:|
| v19a (Jaccard + gate, K=32)    | 0.7728 | 0.2240 | 0.7760 | 0.000 | 3.90 |
| v23b (binary + no gate, K=64)  | 0.7877 | 0.1107 | 0.8893 | 0.005 | 5.05 |
| **v24b (Jaccard + no gate, K=64)** | 0.7742 | **0.3237** | 0.6763 | 0.000 | 3.81 |
| v6 (MSE-Jaccard, K=32)         | 0.7556 | 0.3958 | 0.6042 | 0.000 | 2.86 |

v24b is now the strongest "balanced" variant on the (mAP, unique) plane:
- vs v6 baseline (the prior balanced anchor): mAP +0.0186, unique only
  drops 0.07 (0.396 → 0.324).
- vs v19a (previous Pareto balanced point): mAP +0.0014, unique
  **+0.0997** (0.224 → 0.324) — strict Pareto-dominate.
- vs v23b (current default operating point): mAP −0.014, unique **+0.213**
  (0.111 → 0.324) — different trade-off, not a strict dominance.

Mid-eval trajectory was characteristic: starts low (ep10 mAP 0.7343 with
unique 0.435) because Jaccard granular targets need more time to settle,
then climbs to 0.76 range while unique stays 0.6+. The final hash code
fans out into 7,448 distinct hashes for 23K DB images vs v18's 2,243.

Updated Pareto frontier:
```
mAP
0.79  v20-K64 ⭐ (max)   v18   v24a   v23b
0.78
0.77                                          ⭐ v24b ⭐ (max-balance)
0.76                            v19a
0.75  v6
      +-----------+-----------+-----------+-----------> unique
      0.10        0.20        0.30        0.40
```

### Recommendation

Paper-wise, **report both v20-K64 and v24b** — same dataset, same backbone,
same training budget, just different `lambda_hash_type` and target form.
The pair gives a clean (high-mAP) ↔ (high-unique-compositional) trade-off
story instead of forcing one number to do everything.

Pushed to `github.com:yunddonii/GroundedDNA_save` (`main`).

---

## 2026-05-14 — v25a / v25b: `loss_hash_hard` form ablation

🔴 both reverted — confirms that the **mixed loss form** in v23b is intentional.

User question: should the STE hard-code retrieval loss (`loss_hash_hard`,
λ=0.5) follow the same form as `loss_hash`? It is currently always MSE-
Jaccard even when `lambda_hash_type=hashnet`. Two ablations launched in
parallel from the v23b setup (K=64, no global gate):

- **v25a**: route the STE hard code through `_loss_hash_hashnet` whenever
  `lambda_hash_type=hashnet` so soft and hard paths share the logistic form.
- **v25b**: `--lambda_hash_hard 0.0` — disable the hard-path loss entirely.

Result (Flickr25k):

| run                                |   mAP  | Δ vs v23b | unique | dup    | gap |
|------------------------------------|-------:|----------:|-------:|-------:|----:|
| v23b (mixed form)                  | 0.7877 |    —      | 0.1107 | 0.8893 | 5.05 |
| **v25a (hashnet form on hard)**    | 0.7608 | **−0.027** | 0.0444 | 0.9556 | 4.63 |
| **v25b (lambda_hash_hard=0)**      | 0.7631 | **−0.025** | 0.0512 | 0.9488 | 5.00 |

Both regress on **both axes** — mAP AND unique simultaneously. The
unique-code count drops to 0.044 / 0.051 (half of v23b) which contradicts
the prior hypothesis that "hard MSE-Jaccard was the collision source".

Reinterpretation: the soft-and-hard combination is doing complementary
work:

- `loss_hash` (hashnet binary) gives a **margin signal** — push positives
  above 0, negatives below 0 — but lets all positives pile onto one
  cluster.
- `loss_hash_hard` (MSE-Jaccard on STE-quantised codes) supplies a
  **graduated target** per pair (Jaccard ∈ [0,1] instead of {0,1}). For
  partial-overlap positives it asks for `sim_dna ≈ 0.33` not `sim_dna ≈ 1`,
  which gives the model room to spread same-cluster samples across nearby
  codes instead of merging them.

Remove or homogenise that signal and the model collapses onto fewer
hash codes (unique 0.11 → 0.04) AND the hard path loses its train↔test
bridge (mAP −0.025).

The forward branch was reverted to "soft → hashnet if selected; hard →
always MSE-Jaccard". The comment in the code documents the rationale so
future passes don't try the same thing again. `--lambda_hash_hard` stays
in the CLI as a tunable knob (with the v25b result as warning).

Pushed to `github.com:yunddonii/GroundedDNA_save` (`main`).

---

## 2026-05-14 — v27a: unsupervised hash target (frozen SigLIP2 cosine) [running]

🟢 new ablation — replaces the label-derived Jaccard pairwise similarity
in `loss_hash` / `loss_hash_hard` with the frozen SigLIP2 visual_global
cosine similarity, rescaled `(cos+1)/2 ∈ [0,1]`. Goal: put GroundedDNA in
the same supervision regime as the four unsupervised baselines (CIBHash,
CIMON, SPQ, MLS3RDUH) so the comparison is apples-to-apples.

Motivation:
- The current best v18-family loss treats Jaccard of `multi_hot_labels`
  as the pairwise target — that is a label-supervised signal, structurally
  identical to HashNet / DPSH / CSQ / OrthoHash. Calling our method
  "compositional unsupervised" while the hash supervision itself is fully
  supervised is misleading.
- CIBHash / CIMON / SPQ / MLS3RDUH all sidestep labels by deriving the
  pairwise target from feature-space proximity (NtXent positives, spectral
  pseudo-labels, kNN graphs). Using SigLIP2's frozen visual_global cosine
  is the cleanest version of that pattern given our existing cache.

Implementation (`loss_siglip2.py` + `config.py`, ~25 LoC additive):
- New CLI flag `--hash_target_mode {jaccard, siglip_cos}`. Default
  `jaccard` preserves the legacy supervised behaviour bit-perfectly.
- New helper `build_siglip_cos_similarity(visual_global_feat)` returns
  `[B, B]` in `[0, 1]` from the frozen SigLIP2 image embedding (detached).
- `DNACodonHashLoss.forward` branches on `hash_target_mode`: when set to
  `siglip_cos` it pulls `outputs["visual_global_feat"]` (always populated
  by `model_siglip2.forward`, both for the live and the cached path).
  When `jaccard`, the label-similarity matrix path is untouched.
- The criterion still accepts `multi_hot_labels` in unsupervised mode;
  they are silently unused. No data-pipeline change needed.

Pareto target for v27a: regress mAP gracefully (expect 0.70-0.76) while
unique stays comparable to v24b's 0.324 — the SigLIP2 cosine target is
much smoother than binary Jaccard, so same-class pairs no longer get
forced onto an identical score.

Pushed to `github.com:yunddonii/GroundedDNA_save` (`main`).

---

## 2026-05-14 — v26a / v26b: V3 prompt × current SOTA setups → both regress

🟡 superseded — V3 prompt helps v6 (MSE form) but not v18+ (hashnet form).

Observation that triggered the experiment: the standalone v6+V3-prompt
run from 2026-05-13 (`260513+flickr25k_setting1_v6_promptv3`) showed an
unusually healthy hash-utilisation profile — unique 0.4553 (vs V1's
0.3958) with mAP 0.7543 (vs V1's 0.7556, basically tied). User noted
that signal and asked to retest V3 prompt under the current best setups.

Two parallel runs (Flickr25k, 60 epochs):

- **v26a** = v20-K64 (K=64, hashnet binary, with global gate) + V3 prompt.
- **v26b** = v23b (K=64, hashnet binary, no global gate) + V3 prompt.

Result table comparing each V3 variant to its V1 sibling:

| run                            |   mAP  | Δ mAP   | unique | Δ unique | dup    | gap |
|--------------------------------|-------:|--------:|-------:|---------:|-------:|----:|
| v6  V1 prompt (MSE)            | 0.7556 |   —     | 0.3958 |    —     | 0.6042 | 2.86 |
| v6  V3 prompt (MSE)            | 0.7543 | −0.001  | 0.4553 |  +0.06   | 0.5447 | 2.77 |
| v20-K64 V1 (hashnet, gate)     | 0.7892 |   —     | 0.0982 |    —     | 0.9018 | 5.20 |
| **v26a V3 (hashnet, gate)**    | 0.7765 | **−0.013** | 0.1402 |  +0.042  | 0.8598 | 4.41 |
| v23b V1 (hashnet, no gate)     | 0.7877 |   —     | 0.1107 |    —     | 0.8893 | 5.05 |
| **v26b V3 (hashnet, no gate)** | 0.7664 | **−0.021** | 0.1102 |   ≈ 0    | 0.8898 | 4.37 |

Findings:

1. **V3-prompt effect is loss-form dependent.** Under MSE-Jaccard (v6) it
   gives a strict diversity win (unique +0.06) at essentially zero mAP
   cost. Under the hashnet binary form (v20-K64 / v23b) the cost flips:
   mAP regresses 0.013–0.021 and the diversity gain is at best modest.

2. **The "no gate" + V3 combo is the worst.** v26b loses both axes: mAP
   −0.021 vs v23b AND unique barely changes (0.110 vs 0.111). The two
   diversity mechanisms (gate-removal and richer prompt) are not
   complementary — they collapse to the same operating point in the
   logistic-loss regime.

3. **Why MSE responds to V3 but hashnet does not.** MSE-Jaccard makes the
   per-pair target a sharp function of label overlap, so any extra
   per-sample text variation flows through to per-sample code variation.
   The hashnet binary logistic only cares about sign-of-score, so all
   positives are pushed into the same `score > 0` half-plane regardless
   of how nuanced the underlying text was — the V3 prompt's extra
   information has nowhere to land.

Decision: keep V3 prompt available (cache stays at
`./cache/flickr25k_qwen_v3.jsonl` and `./cache/flickr25k_siglip2_v3/`)
for any future MSE-form ablations or for cross-dataset narratives, but
do not adopt it as the default on the current hashnet-based runs. The
Pareto frontier is unchanged: v20-K64 (max mAP), v23b (mAP-tied at
slightly higher unique), v24b (balanced — high unique).

Pushed to `github.com:yunddonii/GroundedDNA_save` (`main`).

---

## 2026-05-14 — MSCOCO baseline comparison in 4-base DNA space (fair-axis ablation)

🟢 reference

Motivation: the existing 2026-05-12 cross-dataset table compares baselines
in their native **36-bit binary** space against our model in **18-base
DNA** space. The two spaces use different distance metrics (signed binary
Hamming vs base-level Hamming) and different "unique code" semantics
(36-bit string vs 18-base sequence), which makes the mAP / unique trade-off
not directly comparable. This ablation forces every method onto the same
DNA axis.

Procedure:
1. Re-train OrthoHash / HashNet / DPSH on MSCOCO setting1 (60 epochs,
   36-bit head on cached SigLIP2, identical optimiser to the 2026-05-12
   run). CSQ skipped — its 260512 run had failed to produce a result.
   Trial names: `<method>_mscoco_dnacompare` under
   `result_baseline/260514/`, checkpoints at `params/260514/`.
2. Re-extract sign-binary codes `{-1,+1}^36` on query (5,000) and DB
   (107,218).
3. Convert to 18-base DNA: reshape `[N, 36] → [N, 18, 2]`, map
   `{00=A, 01=C, 10=G, 11=T}` (same `BASE_TO_BITS` convention as
   `dna_utils/dna_code_utils.py`).
4. Run `evaluation_siglip2.evaluate_retrieval(..., distance_mode='base')`
   — base-level Hamming, multi-hot Jaccard relevance. `unique_code_ratio`
   computed over the full 18-base sequences.
5. Compare to Ours v6_promptv3, which already produces base codes
   natively (`result/260514+mscoco_setting1_v6_promptv3.../`).

Conversion + eval script: `/tmp/convert_baseline_to_dna_eval.py`. Summary
dump: `cache/baseline_mscoco_dna/summary.json`.

| Method               | mAP (base) | Δ vs native binary mAP | Unique DNA codes / 107,218 | Unique % |
|----------------------|-----------:|-----------------------:|---------------------------:|---------:|
| **OrthoHash**        | **0.6025** | −0.022 (was 0.6243)    | 88,774                     | 82.80%   |
| **HashNet**          |   0.6015   | −0.006 (was 0.6076)    | 31,489                     | 29.37%   |
| **Ours v6_promptv3** |   0.5339   | (DNA-native, no Δ)     | 21,688                     | **20.23%** |
| **DPSH**             |   0.4637   | +0.007 (was 0.4567)    |    138                     |  0.13%   |

Findings:

1. **mAP gap survives the axis change.** OrthoHash / HashNet stay ahead of
   v6_promptv3 by +0.07 even after collapsing their 36-bit space into 18
   base pairs. The base-mode handicap (1-bit-different and 2-bits-different
   pairs both count as one mismatch) costs them at most −0.022 mAP, not
   enough to close the gap.

2. **Compression axis is where our model wins.** v6_promptv3 produces 4×
   fewer unique DB codes than OrthoHash (21.7k vs 88.8k) and ~1.5× fewer
   than HashNet. That's by design — the codebook-VQ + DNA construction
   actively shares codes across semantically similar images, whereas
   OrthoHash's 36 bits are effectively unconstrained random per image.

3. **DPSH is degenerate.** 138 unique codes across 107K images = 99.87%
   collapse. Its mAP 0.4637 is therefore not a meaningful retrieval signal
   — it ranks within a tiny number of equivalence classes. Should be
   excluded from any "compactness vs retrieval" discussion.

4. **No apples-to-apples Pareto neighbour.** None of the binary baselines
   live in our compression regime (20% unique). Reporting this as
   "Pareto-incomparable" is more honest than reporting only mAP.

Practical implication for the paper: when claiming retrieval mAP we should
report the **native distance** for each method (OrthoHash 0.6243 vs Ours
0.5339), not the base-mode rewrite — base mode unfairly handicaps the
binary baselines. The DNA-axis numbers above are the right context for
the **compression** claim (unique-code ratio), not the retrieval claim.

CSQ MSCOCO still missing — its 260512 trial folder contains only
`config.json`. Re-add to a future re-run if a 4-method baseline panel is
needed.

---

## 2026-05-14 — Repository placed under git version control

🟢 active — pushed to `github.com:yunddonii/GroundedDNA_save` (`main`).

---

## Current state (as of 2026-05-14)

- **Best Flickr25k**: **v18** = v6 baseline + `loss_hash` replaced with
  HashNet-style class-weighted logistic likelihood on the DNA continuous
  code. Final mAP **0.7883** — exceeds every binary baseline incl.
  HashNet's own 0.7800 and our prior best (v11 R4 Wasserstein 0.7624).
- **Standard baseline**: **v6** = `sinkhorn router + c_global_source=siglip2_global + K=32, 6 codebooks × 3 codons × 2 bits = 36-bit DNA hash`. SigLIP2 backbone frozen; only adapter + codebooks + codon head are trained. 7 active loss terms after the cleanup. mAP 0.7556 with the MSE-Jaccard form of `loss_hash`.
- **R-series archive**: R1/R2/R3/R4 + v15 codebook-orthogonality result dirs are in `backup/results_R_series/` with per-experiment summary in the README there. The corresponding loss / config / output-dict code paths were removed from the main tree on 2026-05-13.
- **MSCOCO**: v6 = mAP **0.5243**. Underperforms OrthoHash / HashNet baselines.
  V2 prompt redesign brought MSCOCO to mAP **0.5385** (+0.014). v18 form not yet tested on MSCOCO.
- **CIFAR10**: v6 = mAP **0.5335**, top-1 vs all four binary baselines.
- **NUS-WIDE**: cache build still running (SigLIP2 done; Qwen 10% complete, ~50h ETA).

---

## 2026-05-08 — Baseline establishment (v1 → v6)

🟢 active baseline

Goal: pick a stable SigLIP2-based DNA hashing baseline before exploring
alignment-style improvements.

Important changes that mattered (only the inflection points; intermediate
hyper-parameter sweeps elided):

| Tag | Key change                                | CIFAR10 mAP | Flickr25k mAP |
|-----|-------------------------------------------|------------:|--------------:|
| v1  | initial SigLIP2 skeleton                  |      0.3719 |        0.7482 |
| v3  | CIFAR setting1 stabilised                 |      **0.6217** |       — |
| v4  | K=48 codebook size                        |      0.5789 |       — |
| v5  | K=48 + global adapter (`globadp`)         |      0.5929 |       — |
| **v6** | **K=32 + `c_global = SigLIP visual_global`** | 0.5335 | **0.7556** |

Key takeaway: pulling SigLIP2's pooled visual feature directly into
C_global (bypassing the routing path for the global slot) was the single
biggest gain on Flickr25k. CIFAR10 peaked at v3 — we accept the small
CIFAR regression at v6 because Flickr is the multi-label benchmark we're
prioritising.

v6 architecture details:
- 6 codebooks: C_0 = global (fed from `visual_global`), C_1..C_5 = routed local parts.
- Qwen2.5-VL produces 6 per-codebook text descriptions per image (V1 prompt: head/body/limb decomposition).
- Sinkhorn OT router with marginal balance: each part receives ~equal patch mass even when the image has no obvious content for that part.
- Codebook update: EMA with revive-on-dead.
- Final hash: 6 codebook indices → 18 DNA codons (3/codebook) → 36 bits.

---

## 2026-05-10 to 2026-05-12 — R-series alignment ablations (v7 → v14)

🟡 superseded — all archived to `backup/results_R_series/` with a per-version README.

Hypothesis tested: more semantically aligned routing → better retrieval.

| Tag | Modification (one line)                            | Flickr25k mAP | Δ vs v6 |
|-----|----------------------------------------------------|--------------:|--------:|
| v7  | **R1**: Sinkhorn → Attention router               |        0.6975 |  −0.058 ❌ |
| v8  | **R2**: per-image part-contrastive CE loss (λ=0.1) |       0.7189 |  −0.037 ❌ |
| v9  | **R3**: post-adapter SigLIP-style InfoNCE (λ=0.1)  |       0.7073 |  −0.048 ❌ |
| v10 | **R4**: Wasserstein OT cost (λ=0.10)               |       0.7494 |  −0.006 ❌ |
| v11 | R4 λ=0.05                                          |    **0.7624** | **+0.007** ✅ |
| v12 | R4 λ=0.03                                          |       0.7583 |  +0.003 ⚪ |
| v13 | R4 λ=0.07                                          |       0.7586 |  +0.003 ⚪ |
| v14 | R4 λ=0.05 on MSCOCO                                |       0.3528 |  see note |

Per-modification notes:
- **R1 (v7)**: attention router removed Sinkhorn's marginal-balance constraint. 5 local codebooks all collapsed (47–56% dead codes). Confirmed Sinkhorn balance is what was protecting codebook diversity.
- **R2 (v8)**: per-image part-vs-part contrastive CE between routed visual tokens and text part embeddings. Forced part specialisation conflicts with codebook utilisation; net regression.
- **R3 (v9)**: image-level InfoNCE between pooled visual and text features. Initial draft used pre-adapter cached features → **zero gradient** → bit-identical to v6. Fixed to post-adapter; still net regression.
- **R4 (v10–v13)**: added the entropic-OT cost `<π, cost>` from the router (already computed) as a loss term. λ=0.05 is the unique net-positive setting; inverted-U around it. Improves pos→neg distance gap (2.86 → 3.43) but increases full-hash duplicate rate.
- **v14 MSCOCO**: looked like catastrophic collapse (mAP 0.35) but the root cause was a corrupted MSCOCO feature cache, *not* the Wasserstein loss (see next entry). Comparison vs R4 baseline is therefore inconclusive.

Code paths for R1–R4 remain in the main tree but default to `lambda_*=0` / `router_type=sinkhorn`, so they are inert unless explicitly re-enabled. See `backup/results_R_series/README.md` for the full per-experiment context.

Aggregate lesson: **"more interpretable routing" did NOT translate to "better retrieval mAP"** for this architecture on these datasets. The four interventions were all retrieval-neutral-to-negative despite producing visibly cleaner routing / higher unique-code-ratio.

---

## 2026-05-12 — MSCOCO cache corruption discovered & fixed

🟢 fixed

Symptom: MSCOCO v6 baseline (and v14) both showed identical mAP 0.36, unique_code_ratio 0.0002, dead_code 0.97. Initially attributed to R4 Wasserstein loss, but reproduced WITHOUT the loss.

Root cause: `dataset/MSCOCO/images` symlink was never created (`download.sh`'s `ln -s` step had been skipped). The SigLIP2 feature extractor processed 122,218 catalog entries but **could not decode 114,141 of them** → those rows in `visual_global.f16.npy` were left as zeros. The dataloader read zero features → no learning signal → permanent codebook collapse.

Fix:
1. Created the missing symlink: `dataset/MSCOCO/images -> /data/.../MSCOCO/images`
2. Moved corrupted cache aside as `cache/mscoco_siglip2.broken.<timestamp>/`
3. Re-extracted SigLIP2 features (~40 min, 122K images)
4. Re-trained v6 baseline → mAP **0.5243** (was 0.3528 with broken cache)

Side finding: `val2014` partition shows transient I/O errors on a subset of files. Doesn't fully break extraction but means a small slice of val images is unreliable. Not fixed; flagged for later disk health check.

**Practical implication**: any R-series MSCOCO conclusion before 2026-05-12 is unreliable because the underlying baseline was broken. Fresh MSCOCO results are still being collected.

---

## 2026-05-12 — Cross-dataset baseline comparison

🟢 reference

All baselines: 36-bit hash, SigLIP2-frozen backbone, identical 60-epoch training, batch 64, Adam lr=1e-4.

**Flickr25k setting1:**

| Method      |     mAP | Rank |
|-------------|--------:|:----:|
| HashNet     |  0.7800 | 🥇 |
| CSQ         |  0.7596 | 🥈 |
| OrthoHash   |  0.7565 |  3 |
| **Ours v6** |  0.7556 |  4 |
| DPSH        |  0.7092 |  5 |

**MSCOCO setting1** (final-epoch eval, freshly extracted cache):

| Method      |     mAP | P@1   | P@10  | P@100 |
|-------------|--------:|------:|------:|------:|
| OrthoHash   |  0.6243 | 0.853 | 0.848 | 0.840 |
| HashNet     |  0.6076 | 0.453 | 0.691 | 0.704 |
| **Ours v6** |  0.5243 | 0.703 | 0.690 | 0.672 |
| DPSH        |  0.4567 | 0.445 | 0.455 | 0.520 |

**CIFAR10 setting1** (single-label):

| Method      |     mAP | Rank |
|-------------|--------:|:----:|
| **Ours v6** |  0.5335 | 🥇 |
| OrthoHash   |  0.4901 | 🥈 |
| CSQ         |  0.4589 |  3 |
| HashNet     |  0.4358 |  4 |
| DPSH        |  0.1111 |  5 |

Observations:
- **Single-label CIFAR10**: ours is the top method, beating the next baseline (OrthoHash) by +4.3 mAP.
- **Multi-label Flickr25k**: competitive but behind HashNet by 2.4 mAP. v11 (Wasserstein) closes most of that gap.
- **Multi-label MSCOCO**: we are clearly behind. Long-tail ranking is the weak spot — our P@1 (0.703) is the second-best, but P@1000 drops to 0.632 while OrthoHash sustains 0.797. duplicate_rate 0.888 is the direct diagnostic.

Code path to add MSCOCO/NUSWIDE to baseline driver: `baseline/base_model.py::NUM_CLASS / MULTI_LABEL / DEFAULT_CACHE_DIR` patched 2026-05-12.

---

## 2026-05-12 — Per-codeword interpretability tooling

🟢 active

Script: `interpret_compositional_code.py`. Produces, per codebook m in {0..5}:
- **V1**: image grid (`codeword_grid_C{m}_<name>.png`) — rows = codewords, cols = top-N example DB images per codeword. Visual evidence for "codeword X of C_head is always faces"-style claims.
- **V3**: per-codeword Qwen-text word histogram (`codeword_qwen_words_C{m}_<name>.{json,txt}`) — auto-generated semantic label per codeword.

Output goes under `<run_dir>/interpret/`. Already generated for:
- Flickr v6: `result/260510+flickr25k_setting1_v6_globadp.../interpret/`
- Flickr v11: `result/260511+flickr25k_setting1_v11_wass005.../interpret/`

Comparison v6 vs v11 (codebook interpretability): v6's per-codeword Qwen word lists are more semantically clean (e.g., C_4 codeword 1 → "white, texture, smooth, fur" — clear furry-animal cluster). v11 has higher per-image P@1 but its codewords have more mixed semantics. Confirms the alignment-vs-retrieval trade-off.

---

## 2026-05-13 — v15 codebook orthogonality (per-image selected-codeword)

🔴 reverted — net regression on Flickr25k.

Motivation: post-hoc diagnostic on Flickr v6 (see "deep-dive analysis" entry
below) revealed mean off-diagonal codebook MI = 1.36 and 60% hash duplicate
rate -- 5 of 6 codebooks carry ~40% redundant information. Standard
multi-codebook VQ fix: enforce that the M=6 codewords selected for each
image span mutually orthogonal directions in R^D.

Implementation:
- New loss term in `loss_siglip2.py::_loss_codebook_ortho`:
  `mean over (m != m') of <q_m, q_m'>^2` where `q_m` is the L2-normalised,
  STE-wrapped, selected codeword for codebook m.
- Applied to `outputs["quantized_tokens"]` (STE) so the loss value reflects
  the actual selected codewords while the gradient straight-throughs into
  the encoder / router (EMA codebook itself is a non-grad buffer).
- New config arg `--lambda_codebook_ortho`. v15 used λ=0.01.
- Note: an earlier prototype applied the penalty to `semantic_visual_tokens`
  (pre-quantization routed tokens) instead. User clarified the intent was
  the **selected codewords**, and the prototype was replaced.

Result (Flickr25k):

| Metric         |   v6   |   v15  |    Δ     |
|----------------|-------:|-------:|---------:|
| mAP            | 0.7556 | 0.7219 | −0.034 ❌ |
| unique_codes   | 0.3958 | 0.3549 | −0.041   |
| duplicate_rate | 0.6042 | 0.6451 | +0.041   |
| dead_codes     | 0.0000 | 0.1771 | +0.177 ⚠️|
| pos→neg gap    | 2.8618 | 2.5043 | −0.36    |

Per-codebook perplexity collapsed on the 5 local slots:
```
           C_0   C_1   C_2   C_3   C_4   C_5
v6:        31.3  26.8  29.0  28.7  26.4  24.3   (mean 27.8)
v15:       30.1  10.0  13.3   7.3  13.6   6.3   (mean 13.4)
```

Mechanism — orthogonality penalty pulled all images toward a common set of
~13 "mutually-orthogonal anchor codewords" per local codebook. Within-image
orthogonality was achieved at the cost of between-image diversity. Same
"representation specialisation hurts retrieval" pattern as R1-R3.

Code path remains in the tree (`lambda_codebook_ortho` config + loss method)
but defaults to 0; superseded for the current line of work.

---

## 2026-05-13 — v6 Flickr25k deep-dive diagnostic

🟢 reference (still-useful audit of where v6 wins / loses)

Standalone diagnostic on `result/.../v6_globadp.../extract_db.npz` +
`extract_query.npz`, no model needed. Output of `/tmp/analyze_v6_flickr.py`.

1. **Per-codebook drop-ablation** (set one codebook's 6 hash bits to constant,
   re-measure mAP):

| Codebook |   mAP  |  Δ mAP  |
|----------|-------:|--------:|
| C_0      | 0.7488 | +0.0000 |
| C_1      | 0.7343 | −0.0144 |
| C_2      | 0.7455 | −0.0033 |
| C_3      | 0.7389 | −0.0099 |
| C_4      | 0.7351 | −0.0137 |
| C_5      | 0.7460 | −0.0027 |

C_0 (global) contributes **zero** to retrieval distance ranking. The win
from `c_global_source=siglip2_global` was therefore a TRAIN-time effect
(stable anchor for the other codebooks) not a retrieval-time contribution.

2. **Pairwise codebook mutual information** (db, 23000 samples): mean
   off-diagonal = 1.36, mean diagonal = 3.32 → ratio 0.41. C_0 is roughly
   independent of locals (off-diag 0.5-0.6) but C_1–C_5 are heavily
   correlated with each other (1.4–2.1).

3. **Pos / neg Hamming distance**: pos mean 15.56 ± 4.94, neg mean 18.84 ±
   3.80. 22.75% of positive pairs land beyond the median negative — direct
   long-tail mAP loss.

4. **Hash collision**: 9,104 unique hashes for 23,000 samples (duplicate
   rate 60.4%). Worst single-hash cluster: 180 samples. Per-codebook
   codeword usage is full (32/32 used in every codebook); the collisions
   come from the high codebook-pair MI, not under-trained codebooks.

5. **Per-class P@10**: best classes (c_0, c_22, c_5) hit ~0.78; worst
   (c_7, c_20, c_2) drop to 0.04–0.17 — rare-class retrieval is essentially
   random. mean per-class P@10 = 0.4175, vs raw sample-mean P@10 ≈ 0.69 —
   the mAP we report is dominated by common classes.

Headline lesson: the diagnosed bottleneck for v6 on Flickr is
**codebook-pair MI 1.36 → high hash collision → rare-class P@10 collapse**,
not codebook underutilisation.

---

## 2026-05-13 — Prompt V3 (Option D, scene-aware **caption-style sentences**) [running]

🟡 in-progress

Motivation: inspecting V2 outputs confirmed that most slots produce bare
noun phrases or short labels (`C_primary_object: "boat"`,
`C_activity_or_relation: "fishing"`, `C_scene_type: "lake at dusk"`).
SigLIP2's text encoder is contrastively trained on natural captions, not
bare nouns, so sparse-label slots feed it the kind of input it under-utilises
at inference -- reducing how discriminative each slot's embedding is.

V3 (Option D) **keeps V2's six axes unchanged** (so PART_ORDER and the
auto-detect schema in `extract_codebook_texts` are unaffected) but rewrites
the per-axis instructions to require a short caption-style sentence
(~15-25 words) per slot, with an explicit "do not paraphrase C_global across
the other slots" rule. The expectation is **denser, captionish embeddings →
higher per-slot discriminability**, while still preserving V2's
multi-object friendliness.

Implementation:
- New constant `_PROMPT_V3` in `dna_utils/vlm_qwen25_descriptions.py`
  (V3 reuses `CODEBOOK_KEYS_V2` -- same keys, different instructions).
- `preprocess_qwen_codebook_texts.py --prompt_version v3` added; auto-detect
  schema in the extractor already handles V3 because the key tuple matches V2.
- Cache files separated: `*_qwen_v3.jsonl` + `*_siglip2_v3/`.
- Generic launcher at `/tmp/run_prompt_pipeline.sh` parameterises the prompt
  version so future variants (v4, …) reuse the same harness.

Background pipelines launched 2026-05-13 (GPU 0: Flickr 5K, GPU 2: MSCOCO 10K).
Results table to be filled in once both runs complete.

| Dataset    | V1 prompt | V2 prompt | V3 prompt |
|------------|----------:|----------:|----------:|
| Flickr25k  |    0.7556 |    0.7457 |       *(running)* |
| MSCOCO     |    0.5243 |    0.5385 |       *(running)* |

---

## 2026-05-13 — Prompt V2 (Option C, scene-aware) ablation

🟢 active

Hypothesis: V1's head/body/limb decomposition is biologically biased and
breaks down on multi-object scenes (MSCOCO has dining/sports/urban scenes
with no clear primary object). Forcing the VLM to fill all 6 fields under
V1 produces near-paraphrases across C_global, C_head, C_body, C_limb
(confirmed by inspecting sample MSCOCO Qwen outputs).

V2 (Option C) keys, same positional order so PART_ORDER stays compatible:

| Slot | V1 (legacy)              | V2 (scene-aware)        |
|------|--------------------------|-------------------------|
|  0   | C_global                 | C_global                |
|  1   | C_head_or_main_part      | C_primary_object        |
|  2   | C_body_or_secondary_part | C_secondary_object      |
|  3   | C_limb_or_detail_part    | C_activity_or_relation  |
|  4   | C_color_texture          | C_color_texture         |
|  5   | C_background_null        | C_scene_type            |

Implementation:
- New prompt constant `_PROMPT_V2` + key tuple `CODEBOOK_KEYS_V2` in `dna_utils/vlm_qwen25_descriptions.py`.
- `preprocess_qwen_codebook_texts.py --prompt_version {v1,v2}` flag.
- Cache files separated: `*_qwen_v2.jsonl` + `*_siglip2_v2/`.
- `text_description_processor.extract_codebook_texts` auto-detects schema from which keys are present.

Results (mAP, v6 architecture, 60 epochs):

| Dataset    | V1 prompt | V2 prompt |    Δ     |
|------------|----------:|----------:|---------:|
| Flickr25k  |    0.7556 |    0.7457 | −0.0099 ❌ |
| MSCOCO     |    0.5243 |    0.5385 | +0.0142 ✅ |

Interpretation: prompt design is **dataset-dependent**.
- MSCOCO (multi-object scenes) benefits from scene-aware decomposition.
- Flickr25k (single-object photos: flowers, animals, landscapes) suffers
  because head/body/limb is the more natural decomposition for that
  content distribution. unique_code_ratio also marginally drops there.

No single prompt is universally optimal. For a paper claim, this is a
"prompt-as-hyperparameter" story rather than "V2 strictly dominates".

---

## 2026-05-12 (background, ongoing) — NUS-WIDE cache build

🟡 in-progress

Background pipeline kicked off to build a permanent NUS-WIDE cache:
- SigLIP2 features for full union (195,834 images): **done 2026-05-12 16:07** (~41 min)
- Qwen V1 text for full train.txt (193,734 images): ~50 hours, **~10% complete** as of 2026-05-13 morning. GPU 1.

When the Qwen cache completes, NUS-WIDE will be ready for direct training
without further preprocessing.

---

## 2026-05-13 — v18: HashNet-style logistic `loss_hash` (Option A) — new best

🟢 active — adopted as the recommended hash loss form.

Trigger: diagnostic showed that the MSE-on-Jaccard form of `loss_hash`
structurally pushes full-label-overlap pairs to *identical* continuous
codes, producing the 60% duplicate-rate that has been the headline
problem since v6. The user asked to follow the HashNet (Cao et al.,
ICCV 2017) line of prior work and replace the MSE form with a sigmoid /
logistic-likelihood form.

Implementation: see `_loss_hash_hashnet` in `loss_siglip2.py`. Adapts
HashNet to our continuous-code shape `[B, 18, 4]`:

```text
sim_dna  = mean_r <u[i,r,:], u[j,r,:]>            # ∈ [0, 1] -- existing DNA sim
score    = α · (2 · sim_dna − 1)                  # ∈ [-α, α] -- centred + scaled
S_bin    = (label_sim > 0).float()                # binary: any-shared = positive
nll      = softplus(-score) + (1 − S_bin) · score # HashNet logistic likelihood
loss_hash = weighted-mean-of-nll over off-diagonal pairs
            with class-balance weight (HashNet's S/S₀ and S/S₁).
```

CLI: `--lambda_hash_type {mse, hashnet}` (default `mse` preserves v6),
plus `--hashnet_alpha` (default 1.0).

Result (Flickr25k, 60 epochs):

| run                                |   mAP  | unique | dup    | dead | pos→neg gap |
|------------------------------------|-------:|-------:|-------:|-----:|------------:|
| v6  (MSE-Jaccard `loss_hash`)      | 0.7556 | 0.3958 | 0.6042 | 0.00 | 2.86 |
| **v18** (HashNet-style, α=1.0)     | **0.7883** | 0.0975 | 0.9025 | 0.00 | **5.10** |
| Δ                                  | **+0.033** | −0.30 | +0.30 | 0 | **+2.24** |

Standing vs the four binary baselines on Flickr25k (36-bit, frozen SigLIP2):

| Method                | mAP    | P@1   | P@10  | P@100 | P@1000 |
|-----------------------|-------:|------:|------:|------:|-------:|
| **v18 (Ours)**        | **0.7883** | 0.880 | 0.862 | 0.860 | 0.849 |
| HashNet               | 0.7800 | 0.872 | 0.871 | 0.864 | 0.853 |
| CSQ                   | 0.7596 | 0.889 | 0.879 | 0.875 | 0.856 |
| OrthoHash             | 0.7565 | 0.895 | 0.888 | 0.882 | 0.860 |
| v6 (Ours, MSE form)   | 0.7556 | —     | —     | —     | —      |

→ v18 leads on mAP by +0.008 over HashNet itself, even though baselines
still edge us out marginally on the top-K precision metrics. Codebook
utilisation stays healthy (per-codebook perplexity 23.0–30.0 on average
26.7, vs v6's 27.8 — small drop).

Diagnostic surprises:
1. **Collision goes *up*, mAP goes *up* together.** Duplicate-rate
   60% → 90%, yet mAP improves by +0.033. The collisions land on the
   "right" same-label clusters; the binary-S logistic form actively
   encourages this in multi-label data, while pos→neg distance gap
   widens from 2.86 → 5.10 (the real driver of mAP).
2. Earlier hypothesis that "high collision necessarily harms mAP" is
   wrong for multi-label benchmarks like Flickr25k. Collision quality
   matters, not collision quantity.
3. The MSE-on-Jaccard form was over-constraining same-label pairs to
   identical codes WITHOUT widening their margin from negatives;
   HashNet's logistic form makes that margin a first-class objective
   ("score > 0 for positives, < 0 for negatives") and our model
   exploits the freed capacity to push it further (gap 5.10 ≫ 2.86).

Code state: `--lambda_hash_type` defaults to `mse` so existing scripts
reproduce v6 exactly; the new default for fresh runs going forward is
`hashnet`.

---

## 2026-05-13 — v19a / v19b: trade-off ablations off v18

🟢 reference — v18 remains the highest-mAP variant; v19a is the
diversity-recovery balance point.

After v18 won mAP-wise (+0.033 over v6) but pushed collision to 90% and
per-position base normalised entropy to 0.745 (vs v6's 0.952), two
single-variable ablations were launched in parallel from the v18 starting
point:

- **v19a**: same logistic likelihood but plug FRACTIONAL Jaccard S into the
  target instead of binary "any-shared". Hypothesis: per-label-combo
  granularity in the target probability restores per-image distinctness.
  New CLI flag: `--hashnet_use_jaccard`.
- **v19b**: v18's binary S form unchanged, but raise the base-balance
  coefficient inside `loss_dna`: `eta_base_balance 1.0 → 5.0`. Hypothesis:
  per-position 25% A/C/G/T uniformity is what the model is failing.
  New CLI flag: `--eta_base_balance`.

Results (Flickr25k, 36-bit, otherwise v6 architecture, single seed):

| run                                              |   mAP  | Δ vs v6 | unique | dup    | pos→neg gap | base normH |
|--------------------------------------------------|-------:|--------:|-------:|-------:|------------:|-----------:|
| v6   baseline (MSE-Jaccard)                      | 0.7556 |    —    | 0.3958 | 0.6042 | 2.86 | 0.9516 |
| **v18  logistic + binary S, η=1**                | **0.7883** | **+0.033** | 0.0975 | 0.9025 | **5.10** | 0.7454 |
| **v19a logistic + Jaccard S, η=1**               | 0.7728 | +0.017 | **0.2240** | 0.7760 | 3.90 | **0.9662** |
| **v19b logistic + binary S, η=5**                | 0.7827 | +0.027 | 0.0858 | 0.9142 | 4.85 | 0.7373 |

Findings:

1. **Both single-variable changes leave v18 the mAP winner.** v19a drops by
   −0.0155, v19b by −0.0056. v18's binary-S logistic remains the
   strongest mAP configuration.

2. **v19a successfully restores diversity** with a non-trivial mAP cost.
   `unique_code_ratio` 0.098 → 0.224 (×2.3), `mean_base_normalised_entropy`
   0.745 → 0.966 (essentially uniform A/C/G/T). The Jaccard target form
   gives each label-powerset combination its own target probability, so the
   model doesn't collapse all positive pairs onto a single cluster.

3. **v19b does NOT recover base balance.** `mean_base_normalised_entropy`
   actually drops 0.7454 → 0.7373. Raising `eta_base_balance` 5× had no
   visible effect, because (a) the term is on the batch-mean base
   distribution rather than per-position, and (b) the entropy term in
   `loss_dna` (one-hot pressure) competes with it. A proper fix would
   require a per-position uniformity term, not just a bigger η.

4. **Per-codebook perplexity** is highest in v19a (mean 28.3) — v19a is
   the most "compositional" code in terms of codebook diversity, even
   though v18 still gives the best retrieval. Useful for the paper's
   compositionality claim.

Pareto frontier (diversity vs mAP) for the four runs:
```
v18  (unique 0.098, mAP 0.7883)   ← max mAP
v19b (unique 0.086, mAP 0.7827)
v19a (unique 0.224, mAP 0.7728)   ← balance point, Pareto-dominates v6
v6   (unique 0.396, mAP 0.7556)   ← max diversity (legacy MSE)
```
v19a strictly Pareto-dominates v6 on both axes (better mAP AND lower
duplicate rate). v18 trades almost all the diversity to win the last
0.0155 of mAP.

Recommendation:
- Keep `--lambda_hash_type hashnet` (v18 form) as the **primary** running
  configuration.
- Keep `--hashnet_use_jaccard` available as the **balanced** variant
  documented in the paper's compositionality argument.
- Drop `--eta_base_balance` strengthening for now; address base imbalance
  via a per-position term in a future iteration (not committed).

---

## 2026-05-13 — v20a / v20-K64: S-cap and capacity ablations (new mAP SOTA, unique unchanged)

🟢 reference — v20-K64 establishes a new Flickr25k mAP best (0.7892); unique
code utilisation remains at the v18 floor (~0.10) across all hashnet
variants, regardless of S-cap or K.

User goal restatement: maximise unique_code_ratio simultaneously with mAP,
since "compositional code" is a paper contribution. v18 collision (90%)
undermines that claim. Two single-variable ablations off v18:

- **v20a**: cap `S_target` at 0.9 inside the hashnet logistic. Same-powerset
  positives no longer aim for `sigmoid(score)=1` exactly → residual degree
  of freedom inside each cluster should yield distinct codes.
  New CLI flag: `--hashnet_S_cap`.
- **v20-K64**: same v18 form but `--codebook_size 64` (vs default 32).
  Doubles per-codebook codeword count → larger candidate hash space.

Results (Flickr25k, 36-bit, otherwise v6 architecture, single seed):

| run                                |   mAP  | Δ vs v6 | unique | dup    | dead  | pos→neg gap | base normH |
|------------------------------------|-------:|--------:|-------:|-------:|------:|------------:|-----------:|
| v6   baseline (MSE-Jaccard)        | 0.7556 |    —    | 0.3958 | 0.6042 | 0.000 | 2.86 | 0.9516 |
| v18  logistic + binary S, K=32     | 0.7883 | +0.033  | 0.0975 | 0.9025 | 0.000 | 5.10 | 0.7454 |
| v19a logistic + Jaccard S, K=32    | 0.7728 | +0.017  | 0.2240 | 0.7760 | 0.000 | 3.90 | 0.9662 |
| v19b logistic + binary, η=5, K=32  | 0.7827 | +0.027  | 0.0858 | 0.9142 | 0.000 | 4.85 | 0.7373 |
| **v20a logistic + binary, S_cap=0.9, K=32** | 0.7840 | +0.028 | 0.0965 | 0.9035 | 0.000 | 4.60 | 0.7876 |
| **v20-K64 logistic + binary, K=64** | **0.7892** | **+0.034** | 0.0982 | 0.9018 | 0.005 | **5.20** | 0.7628 |

Findings:

1. **`K=64` is the new mAP best on Flickr25k (0.7892)**, marginally above
   v18's 0.7883. The bigger codebook produces slightly better
   discriminability (pos→neg gap 5.20 vs 5.10) at the cost of 5 of 64
   codewords going dead per codebook. mAP ceiling for the v18 form
   appears to be ~0.79; doubling K does not break through it.

2. **S-cap=0.9 (v20a) does NOT recover unique code count.** Diversity stays
   at 0.097 — *identical* to v18. The cap successfully raised base
   normalised entropy 0.745 → 0.788 (per-position usage slightly more
   uniform), but the cluster-level collapse (same-powerset → one hash)
   was unaffected. The capped target still has all positives aiming at
   the same `sigmoid(score)=0.9`, so they still pile up on the same code.

3. **K does not increase unique either.** v20-K64 hash-unique-ratio is
   0.098 — almost the same as K=32 (v18: 0.0975). Capacity is not the
   bottleneck for collision; the *form* of the loss is.

4. **The unique-code-ratio floor ≈0.10 is structural** across every
   hashnet-form variant tried (v18, v19b, v20a, v20-K64). Flickr25k has
   ~3.7 labels per image → ~60–70% of random pairs share at least one
   label → all those positives converge onto a small number of cluster
   centres under the binary-S logistic. Single-knob fixes (cap, η, K)
   shift base-entropy and mAP but leave the cluster count unchanged.

Pareto frontier of the variants explored so far (Flickr25k):
```
v20-K64 (unique 0.098, mAP 0.7892)   ← new mAP SOTA
v18     (unique 0.098, mAP 0.7883)
v20a    (unique 0.097, mAP 0.7840)
v19b    (unique 0.086, mAP 0.7827)
v19a    (unique 0.224, mAP 0.7728)   ← Pareto-dominates v6
v6      (unique 0.396, mAP 0.7556)
```
v19a stays Pareto-optimal for the "balanced" axis (best mAP/unique trade-off);
v20-K64 wins pure-mAP.

Implication: to get both high mAP AND high unique simultaneously, the
hashnet form alone is insufficient. Candidates for the next iteration:
- Explicit anti-collision loss (e.g. soft cap on max cluster size)
- Hadamard / class-centre target codes (CSQ-style) so distinct
  label-powersets target deterministically distinct hashes
- Combine v19a's Jaccard target with v20-K64's larger K

---

## 2026-05-13 — v21a / v21b: `c_global_source=mean_pool` ablation under hashnet form

🟡 superseded — both regress vs their `siglip2_global` counterparts.

User question: under the new hashnet logistic form (v18/v20-K64), does the
legacy `c_global_source=mean_pool` work better than `siglip2_global`?
The original v5→v6 jump was switching mean_pool → siglip2_global, but
that win predates the loss-form switch.

Setup: identical to v18 / v20-K64 except `--c_global_source mean_pool`.

Results (Flickr25k, 36-bit, single seed):

| run                                |   mAP  | Δ vs sib | unique | dup    | dead  | pos→neg gap | base normH |
|------------------------------------|-------:|---------:|-------:|-------:|------:|------------:|-----------:|
| v18  K=32, siglip2_global          | 0.7883 |    —     | 0.0975 | 0.9025 | 0.000 | 5.10 | 0.7454 |
| **v21a K=32, mean_pool**           | 0.7821 |  −0.0062 | **0.0423** | 0.9577 | 0.000 | 4.47 | 0.7151 |
| v20-K64 K=64, siglip2_global       | 0.7892 |    —     | 0.0982 | 0.9018 | 0.005 | 5.20 | 0.7628 |
| **v21b K=64, mean_pool**           | 0.7834 |  −0.0058 | 0.0683 | 0.9317 | 0.021 | 4.84 | 0.7443 |

Findings:

1. **Both mean_pool variants regress.** mAP drops by 0.006 at both
   K=32 and K=64; magnitude is similar so the effect is purely from the
   global slot, not from K.

2. **Unique code count gets WORSE under mean_pool.** v21a hits 0.042 —
   a new minimum across all v6-derivative runs. Of 23,000 DB images, only
   ~970 distinct hashes are produced.

3. **Mechanism**: `mean_pool` averages the *post-adapter* visual_tokens —
   that's the same pool the 5-part router feeds from. C_0 ends up sampling
   the same feature distribution as C_1..C_5, so the 6 codebooks become
   more correlated. v18/v20-K64's `siglip2_global` feeds the MAP-pooled
   pre-projection class-aware embedding into C_0 through `global_adapter`,
   giving C_0 an information channel the local codebooks don't share.

4. The original v5→v6 win (+0.007 on Flickr) was therefore the SigLIP2 MAP
   feature being a strong, codebook-decorrelated input — a property the
   hashnet form preserves and amplifies. `siglip2_global` is firmly the
   default going forward.

Pareto frontier (unchanged top):
```
v20-K64 (unique 0.098, mAP 0.7892)   ← mAP SOTA
v18     (unique 0.098, mAP 0.7883)
v20a    (unique 0.097, mAP 0.7840)
v21b    (unique 0.068, mAP 0.7834)
v19b    (unique 0.086, mAP 0.7827)
v21a    (unique 0.042, mAP 0.7821)   ← worst unique
v19a    (unique 0.224, mAP 0.7728)
v6      (unique 0.396, mAP 0.7556)
```

---

## 2026-05-13 — v22a: per-slot `text_adapter` (Option A) — regression

🔴 reverted — per-slot text adapter hurts mAP by −0.046 on Flickr25k.

Trigger: diagnostic on SigLIP2 cached text embeddings revealed that the
six per-slot pooled vectors are **near-identical** across an image
(pairwise cos ~0.88, per-slot effective rank 2.7/D=768). The shared
`TextAdapter` MLP applied to all 6 slots preserves that collapse, so the
Sinkhorn router has essentially identical centroids per image -> codebook
supervision is uninformative -> high cross-codebook correlation
(observed v18/v20-K64 codeword pairwise cos 0.40, codebook effective
rank 3 / K).

User proposal — **Option A (cheap)**: replace the shared MLP with an
`nn.ModuleList` of six independent MLPs (one per codebook slot). The
slot-specific weights have the structural freedom to push the six
near-identical inputs apart.

Architectural change in `model_siglip2.py`:
```python
if self.per_slot_text_adapter:
    self.text_adapter = nn.ModuleList([TextAdapter(...) for _ in range(6)])
# forward
per_slot = [self.text_adapter[m](raw[:, m, :]) for m in range(6)]
text_part_tokens = torch.stack(per_slot, dim=1)         # [B, 6, D]
```
New CLI: `--per_slot_text_adapter`.

Init smoke test confirmed the architectural goal works *at init*: six
independent random MLPs produce cross-slot cos ~0.01 (essentially
orthogonal) from cos 0.88 inputs.

Result (Flickr25k, K=64, hashnet binary, single seed):

| run                                  |   mAP  | Δ vs sib | unique | dup    | dead  | pos→neg gap | base normH |
|--------------------------------------|-------:|---------:|-------:|-------:|------:|------------:|-----------:|
| v20-K64 (shared text_adapter)        | 0.7892 |    —     | 0.0982 | 0.9018 | 0.005 | 5.20 | 0.7628 |
| **v22a (per-slot text_adapter)**     | 0.7433 | −0.0459 | 0.0640 | 0.9360 | 0.060 | 3.82 | 0.6809 |

Mid-eval trajectory was the diagnostic warning:
```
ep 10  0.7475
ep 20  0.7525   ← peak
ep 30  0.7408
ep 40  0.7295   ← bouncing down
ep 50  0.7353
ep 60  0.7293 / final 0.7433
```
Per-codebook perplexity (K=64) shows where the damage lands:
```
                 C_0   C_1   C_2   C_3   C_4   C_5
v20-K64:        60.4  54.0  54.6  49.8  46.3  47.9   (mean 52.2)
v22a:           60.8  23.5  22.5  33.4  22.5  31.0   (mean 32.3)
                       ↓↓    ↓↓          ↓↓
```
C_0 is intact (it's fed by `visual_global`, not by text routing) but the
five local codebooks lose ~half their effective codeword count.

Mechanism (post-hoc): the 0.88 cross-slot cos sim under the shared
`TextAdapter` was NOT a bug — it carried "all six descriptions refer to
the *same image*" as a consistency anchor. The Sinkhorn router used that
anchor to route patches of one image to a coherent set of part-specific
centroids that all lie in the same image-conditional neighbourhood.
Six random orthogonal centroids (the per-slot init state) destroy that
anchor: patches now fall onto whichever centroid they happen to lie
closest to, with no image-level consistency. Routing becomes
content-blind → codebook EMA centres see inconsistent inputs → local
codebooks (C_1..C_5) collapse to a smaller effective codeword count.

The "cos 0.88 is bad" hypothesis was wrong; the right description is
"semantic orthogonality across slots is missing", which is a different
problem and the per-slot MLP can't solve it (it has no signal to
discover semantic axes from near-identical inputs).

Decision: revert `--per_slot_text_adapter` to default off. Code path
remains but inert. If we still want to spread the slot embeddings, the
fix has to inject extra information — Option B (visual-attention-pooled
TEXT TOKENS, currently un-cached) or Option C (use Qwen hidden states
directly) — not just structural slot-specific weights with no extra
signal.

---

## 2026-05-13 — v22b: visual-attention pooled text tokens (Option B) — regression

🔴 reverted — net regression of −0.022 vs v20-K64. With v22a (−0.046) this
closes the "fix the text encoder collapse" line of work: pushing the six
slot embeddings apart, by ANY mechanism explored so far, hurts retrieval.

Setup: v20-K64 base + cross-attention block that takes the (re-extracted)
token-level SigLIP2 text features and lets visual patches cross-attend
over them per slot. New infrastructure:

- `extract_siglip2_features.py --save_text_tokens`: additionally saves
  `text_tokens.f16.npy` ([N, 6, T, D_proj]) and `text_token_mask.bool.npy`
  ([N, 6, T]). Falls back to `last_hidden_state` directly if the model
  has no separate `text_projection` layer (SigLIP2 does not).
- `dataloaders.py::_SigLIP2FeatureCache.__init__/get`: loads the token
  cache if present and exposes `cached_text_tokens` / `cached_text_token_mask`
  in the batch.
- `model_siglip2.py`: new `nn.MultiheadAttention` block + LN. Per slot m,
  cross-attend `visual_tokens` (Q) over `text_tokens[:, m, :, :]` (K, V)
  with the per-slot attention mask, then mean-pool over the visual axis to
  get a [B, D] visually-conditioned slot embedding. Stacks to [B, 6, D]
  for the Sinkhorn router.
- `config.py`: new CLI `--use_text_token_attention --text_attn_num_heads`.

Diversity check at init (untrained random attention block): cross-slot
mean cos sim ≈ 0.07 (vs 0.88 of the pooled cache). Architecturally the
slots ARE distinguishable.

Token-level cache verification on Flickr25k:
- Token mean-pooled cross-slot cos: ~0.78 (vs 0.88 for pooler_output)
  → token-level is inherently more diverse than pooled, confirming
  the cache contains additional signal.

Result (Flickr25k, K=64, hashnet binary, single seed):

| run                                |   mAP  | Δ vs v20-K64 | unique | dup    | dead  | pos→neg gap |
|------------------------------------|-------:|-------------:|-------:|-------:|------:|------------:|
| v20-K64 (pooled text, shared MLP)  | 0.7892 |       —      | 0.0982 | 0.9018 | 0.005 | 5.20 |
| **v22b (token-level + visual-attn pool)** | **0.7673** | **−0.022** | 0.0660 | 0.9340 | 0.003 | 5.09 |

Mid-eval trajectory was volatile:
```
ep 10  0.7408
ep 20  0.7271   ← deepest dip
ep 30  0.7396
ep 40  0.7570
ep 50  0.7549
ep 60  0.7543 / final 0.7673
```

Compared to v22a (per-slot MLP) which final-ed at 0.7433, v22b is
markedly better (visual conditioning is informative), but still
substantially worse than v20-K64. The bouncing trajectory matches
v22a's pattern: introducing a mechanism that decorrelates the slot
embeddings destabilises the codebook learning step.

Mechanism (post-hoc, same as v22a's lesson): the 0.88 cos similarity is
NOT a defect of the SigLIP2 text encoder — it is **an image-level
consistency anchor**. With it, the six routing centroids per image lie
in a small neighbourhood, and the Sinkhorn coupling is driven primarily
by visual content rather than by which-centroid-is-closer. Both v22a
(per-slot MLP, random orthogonality) and v22b (visual-attention, content-
aware orthogonality) break that anchor. v22b breaks it less aggressively
(content-aware), hence smaller regression (−0.022 vs v22a's −0.046).

Decision: code path remains in the tree (cache extraction, dataloader,
model branch) but `--use_text_token_attention` defaults to off. The
cache at `cache/flickr25k_siglip2_tokens/` is retained for future
"text + something else" experiments.

The conclusion across v22a/v22b is paper-relevant: trying to "fix" the
text-encoder collapse from the OUTPUT side (after the encoder) does not
help retrieval. Any future revisiting should attack at the source —
e.g., richer per-slot prompts, fine-tuned text encoder, or replacement
with Qwen2.5-VL hidden states.

---

## 2026-05-13 — Codebase cleanup: removed deprecated / no-op loss terms

🟢 active — main tree now reflects the verified-active loss set only.

After v17 confirmed that activating `loss_global` regresses Flickr25k mAP
by 0.012, the user requested option (b): drop the deprecated term entirely
and also strip the four R1–R4 / v15 inactive loss terms that were still
emitting empty / no-op columns into `log.csv`. All trial-and-error history
for the removed terms is preserved in (a) this document and (b)
`backup/results_R_series/README.md`; nothing was removed before checking
documentation coverage.

Files touched (single 2026-05-13 cleanup commit conceptually):

- `loss_siglip2.py`:
  - Removed methods `_loss_global`, `_loss_align`, `_loss_siglip_align`,
    `_loss_codebook_ortho`.
  - Removed `lambda_global`, `lambda_align`, `align_temperature`,
    `lambda_siglip_align`, `siglip_align_temperature`,
    `lambda_wasserstein`, `lambda_codebook_ortho` from `__init__`.
  - Removed corresponding entries from the forward `total` sum and the
    returned dict.
  - Top-line docstring updated to enumerate only the active components.
- `config.py`:
  - Removed CLI args `--lambda_global`, `--lambda_align`,
    `--align_temperature`, `--lambda_siglip_align`,
    `--siglip_align_temperature`, `--lambda_wasserstein`,
    `--lambda_codebook_ortho`.
- `train_siglip2.py`:
  - Trimmed `loss_types` list so `log.csv` no longer carries
    `train_loss_global / val_loss_global / train_loss_align / ... /
     train_loss_codebook_ortho / ...` columns.

Active loss set after cleanup (CSV-tracked):

| Term            | λ     | Role |
|-----------------|------:|------|
| `loss_hash`     | 1.00  | continuous code pairwise sim ↔ label sim |
| `loss_hash_hard`| 0.50  | STE-quantised hard code, same supervision |
| `loss_vq`       | 0.25  | codebook commitment (z ↔ q) |
| `loss_quant`    | 0.05  | codon-side commitment |
| `loss_anchor`   | 0.05  | local codebooks track EMA text anchor |
| `loss_dna`      | 0.05  | codon entropy + base balance (aggregate) |
| `loss_bu`       | 0.02  | codebook usage balance + uncorrelated (aggregate) |

→ **7 top-level active terms**, 11 individually-tracked CSV columns
(`loss_dna` and `loss_bu` each expand into 2 inner components).

Smoke verified: `python -c "from loss_siglip2 import DNACodonHashLoss; ..."`
returns exactly the 12 expected keys and `loss.backward()` succeeds.

The R-series code (attention router class, `ot_cost` propagation) is left
untouched in `models/semantic_router.py` and `model_siglip2.py` since
those don't pollute `log.csv` and removing them is more invasive (router
class + config branch). Future cleanup possible but low-priority.

---

## 2026-05-13 — v17: re-wired `loss_global` to use STE (option (a)) → net regression

🔴 reverted — properly activating `loss_global` HURTS Flickr25k mAP.

Follow-up to the "no-op under EMA" entry below. Two ways to make
`loss_global` actually train something were proposed: (a) feed it the
STE-wrapped `quantized_tokens` so the gradient reaches `global_adapter`,
(b) drop the term entirely so the inventory matches reality. The user chose
to try (a) first.

Implementation (one-line patch in `loss_siglip2.py::forward`):
```diff
- loss_global = self._loss_global(q[:, 0, :], S, mask)        # q = quantized_tokens_raw (no grad)
+ q_ste = outputs.get("quantized_tokens")
+ loss_global = self._loss_global(q_ste[:, 0, :], S, mask)    # STE-wrapped (grad path to global_adapter)
```
Smoke test confirmed `loss_global.grad_fn` now exists and a 4421-magnitude
gradient flows into `global_adapter` (with `visual_adapter` / `text_adapter`
correctly receiving zero, since C_0 routes only through `global_adapter`).

Result (Flickr25k, 60 epochs, otherwise v6 config):

| Run                                   |   mAP  | unique | dup    | dead  | pos→neg gap |
|---------------------------------------|-------:|-------:|-------:|------:|------------:|
| v6  baseline (`loss_global` no-op)    | 0.7556 | 0.3958 | 0.6042 | 0.000 | 2.86 |
| **v17 `loss_global` rewired (STE)**   | **0.7439** | 0.3247 | 0.6753 | 0.010 | 2.78 |
| Δ                                     | **−0.012** | −0.071 | +0.071 | +0.010 | −0.08 |

Per-codebook perplexity collapsed asymmetrically on the local slots:
```
            C_0   C_1   C_2   C_3   C_4   C_5
v6:        31.3  26.8  29.0  28.7  26.4  24.3   (mean 27.8)
v17:       31.0  15.3  27.1  26.7  14.6  25.0   (mean 23.3)
                  ↓↓                ↓↓
```
C_0 itself looks normal (perp 31). C_1 and C_4 dropped to ~half their v6
utilisation -- their codeword count effectively halved.

Proposed mechanism: when `loss_global` actually trains `global_adapter`,
the C_0 codeword pairwise similarity gets pulled directly toward the
label-similarity matrix S. `loss_hash` (which already supervises the full
36-bit code against S) then receives a much weaker gradient on C_0 because
C_0 is already in agreement -- so `loss_hash`'s gradient concentrates on the
five local slots. C_1 and C_4 absorb the brunt and partially collapse.

Same "alignment-style intervention → codebook utilisation degradation →
mAP regression" pattern as R1–R4 and v15.

Conclusion for the immediate decision: **option (a) does not help**. The
two remaining real options are:
- keep `loss_global` as no-op cosmetic (current v6, since `lambda_global > 0`
  contributes only to the CSV log; no harm done), or
- option (b) — remove `lambda_global` / `_loss_global` from the codebase to
  clean up the inventory. Mathematically identical to current v6.

Result directory of v17 retained under `result/` for the record; it is NOT
moved to backup because the "STE rewire" itself is documented here.

---

## 2026-05-13 — `loss_global` is a no-op under EMA codebook mode (cosmetic only)

🟡 inventory correction — `loss_global` was listed as λ=0.1 active in the
inventory below but a code trace + empirical bit-perfect parity confirms it
contributes ZERO gradient to any trainable parameter in EMA mode.

Investigation trigger: the user asked whether `loss_global` is necessary.
Launched **v16** = v6 with `--lambda_global 0.0` on Flickr25k to ablate. At
the first mid-eval checkpoint v16 produced **bit-perfect identical** metrics
to v6 (mAP=0.7383 / unique=0.5625 / dead=0.3698, all four decimal places
matching). That should not happen if `loss_global` contributed any gradient.

Root cause:
- `loss_siglip2.py` calls `self._loss_global(q[:, 0, :], S, mask)` where
  `q = outputs["quantized_tokens_raw"]`.
- `quantized_tokens_raw = self.codebooks[m_idx, indices]` (`model_siglip2.py`).
- In EMA mode `self.codebooks` is a `register_buffer`, so the indexed result
  has `requires_grad=False` and `grad_fn=None`.
- The loss value is correctly computed for logging, but `.backward()` on
  `loss_global` alone fails with "element 0 of tensors does not require grad",
  i.e. **no gradient flows to any trainable parameter**.

Verified by isolated smoke test: `loss_global.requires_grad == False`,
`loss_global.grad_fn is None`.

Implication for the loss inventory:
- Declared active terms (λ > 0):   8
- **Effectively active (carry gradient):  7**
- The 8th declared term (`loss_global`) is cosmetic only under EMA mode.
- If we ever switch the codebook to gradient mode (`--codebook_update gradient`)
  OR route `loss_global` through `quantized_tokens` (STE-wrapped) instead of
  `quantized_tokens_raw`, it would re-acquire a real gradient path. Both are
  one-line changes; not done here because the immediate question was "is
  `loss_global` necessary in its current form" — answer: no.

v16 result directory was deleted (training would have been mathematically
identical to v6; no new data to keep).

Recommended follow-ups (not done yet, recorded for future reference):
- Either re-wire `loss_global` to `quantized_tokens` so it actually trains C_0;
  or drop the `lambda_global` argument entirely so the codebase isn't
  misleading.

---

## 2026-05-13 — Active loss-term inventory (post-cleanup)

🟢 reference — supersedes the earlier "as of v15" inventory.

Final active set in `loss_siglip2.DNACodonHashLoss` after the codebase
cleanup. All previously-listed inactive / no-op terms (R1–R4 alignment
family, v15 `loss_codebook_ortho`, deprecated `loss_global`) have been
**removed from the source tree**, not just λ=0-gated. The archived
history of each removed term is in `docs/PROJECT_LOG.md` and
`backup/results_R_series/README.md`.

| # | Term            | λ    | Role |
|---|-----------------|-----:|------|
| 1 | `loss_hash`     | 1.00 | continuous code pairwise sim ↔ label sim (MSE) |
| 2 | `loss_hash_hard`| 0.50 | STE-quantised hard code, same supervision |
| 3 | `loss_vq`       | 0.25 | semantic_visual_tokens ↔ quantized_tokens commitment |
| 4 | `loss_quant`    | 0.05 | continuous code ↔ DNA hard code commitment |
| 5 | `loss_anchor`   | 0.05 | local codebook centroids track EMA text anchor |
| 6 | `loss_dna`      | 0.05 | aggregate: `loss_entropy + η · loss_base_balance` |
| 7 | `loss_bu`       | 0.02 | aggregate: `loss_cb_balance + ρ · loss_cb_uncorr` |

→ **7 top-level active terms.**

Internal expansion (CSV-tracked individual components):
```
loss = 1.0  · loss_hash
     + 0.5  · loss_hash_hard
     + 0.25 · loss_vq
     + 0.05 · loss_quant
     + 0.05 · loss_anchor
     + 0.05 · (loss_entropy + 1.0 · loss_base_balance)
     + 0.02 · (loss_cb_balance + 0.1 · loss_cb_uncorr)
```
→ 11 CSV columns (7 top-level − 2 aggregates + 4 inner).

Removed from the source tree (and from `log.csv`):
`loss_global`, `loss_align`, `loss_siglip_align`, `loss_wasserstein`,
`loss_codebook_ortho`.

---

## Infrastructure & repo hygiene

🟢 active

- All R-series result directories moved to `backup/results_R_series/` with
  per-experiment summary in `backup/results_R_series/README.md`.
- Empty / smoke result dirs left under `result/` (low priority cleanup).
- Slide-ready visualisation snapshots are mirrored under `slides_assets/`
  with `/`-replaced-by-`_` filenames so they can be uploaded into a
  presentation generator without context loss.

---

<!--
HOW TO UPDATE THIS DOCUMENT
- For every non-trivial change (new variant, new ablation, new dataset,
  new cache, new finding, reverted decision), append a section dated with
  the **absolute** YYYY-MM-DD it lands. Do not edit older sections except
  to mark them 🟡 superseded / 🔴 reverted.
- Keep the "Current state" snapshot at the top in sync.
- Numbers belong in tables; rationale belongs in 1–2 sentences below.
- Code paths or commit refs (when applicable) belong in fenced spans.
-->
