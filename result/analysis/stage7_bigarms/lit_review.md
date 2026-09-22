<!-- Copied from the literature/theory review of 2026-09-22 (subagent report). Independently re-opened by the
     main session on 2026-09-22: [B6] SCE (arXiv 2111.14585, WACV 2023), [B7] X-CLR (arXiv 2407.18134; venue not
     stated on the arXiv page, ICLR 2025 as reported is NOT independently verified), [D1] Chen, Luo, Li,
     NeurIPS 2021 (arXiv 2011.02803, quote confirmed), [B5] LooC (arXiv 2008.05659, ICLR 2021). Other entries
     are as the report states, including its own UNVERIFIED marks. -->
# Literature and theory check: proposals A, B, C for slot specialisation

*2026-09-22. I opened every URL listed here in this session. OpenReview, CVF, IEEE Xplore, Springer and ACM pages blocked the fetcher, so I used arXiv, ar5iv, proceedings mirrors, ML Anthology or the Semantic Scholar API instead. The derivations are my own. No experiment was run and no repository file was read or changed.*

Notation: M = 4 local slots (+1 global); K = 64 concepts per axis; 4³ = 64 codons; B = batch size; N = training images; N_db = database size. JS is the Jensen–Shannon divergence: 0 for equal distributions, at most ln M for M distributions. Citation format: **title**, authors, venue, year, URL, then the mechanism (→ how it maps to the proposal).

## Summary
| Item | Verdict | Main reason |
|---|---|---|
| Diagnosis | **Supported** | Every slot has the same one-hot target and no term links two slots, so identical slots are an exact optimum (with a shared temperature; per-pair temperatures only reweight repulsion). Feature suppression and multi-view theory predict what they share. |
| B, soft positives | **Supported, with conditions** | Once axis targets differ, identical slots provably pay at least M·JS extra loss. Direct precedents: SCE (same target form), X-CLR (caption-similarity targets), LooC (a different positive set per head). |
| C, division of labour | **Supported as the enabling step; not enough alone** | Removes the dominant term all slots share. Capacity cost is small next to today's effective ~1 codon. Local slots still need A or B. |
| A, named codebook | **Plausible** | Every part has precedent. But concept bottlenecks usually cost accuracy, the relation axis is likely not identifiable from frozen CLIP, and the uniform-codon regulariser conflicts with uneven k-means clusters. |

---

## A. Named concept codebook

### Supporting literature
- **[A1] Concept Bottleneck Models.** Koh, Nguyen, Tang, Mussmann, Pierson, Kim, Liang. ICML 2020. https://proceedings.mlr.press/v119/koh20a.html. Predicts concepts first, then the label from concepts alone, competitive with end-to-end models. → A is an independently trained CBM with one categorical concept per axis; Hamming retrieval replaces the label head.
- **[A2] Label-Free Concept Bottleneck Models.** Oikarinen, Das, Nguyen, Weng. ICLR 2023. https://ar5iv.labs.arxiv.org/html/2304.06129. Uses GPT-3 concepts and fits the concept layer to CLIP image–text similarities. Drops concepts that do not activate on the data or whose projection similarity is below 0.45. ImageNet: 71.95 % vs 76.13 % for the standard model. → Labels made by CLIP; precedent for pruning concepts the image cannot predict.
- **[A3] Language in a Bottle: Language Model Guided Concept Bottlenecks for Interpretable Image Classification.** Yang, Panagopoulou, Zhou, Jin, Callison-Burch, Yatskar. CVPR 2023. https://arxiv.org/abs/2211.11158. GPT-3 sentence concepts with a submodular discriminative-plus-diverse selection and CLIP alignment; matches linear probes when data is plentiful. → A language-defined concept vocabulary.
- **[A4] Post-hoc Concept Bottleneck Models.** Yuksekgonul, Wang, Zou. ICLR 2023. https://ar5iv.labs.arxiv.org/html/2205.15480. CLIP-RN50, CIFAR-100: original 0.701, concepts only 0.520, concepts plus residual (PCBM-h) 0.680. COCO-Stuff: 0.770 / 0.741 / 0.768. → A combined with C, with the global slot as the residual. The gap depends on the dataset.
- **[A5] Interpreting CLIP with Sparse Linear Concept Embeddings (SpLiCE).** Bhalla, Oesterling, Srinivas, Calmon, Lakkaraju. NeurIPS 2024. https://proceedings.neurips.cc/paper_files/paper/2024/file/996bef37d8a638f37bdfcac2789e835d-Paper-Conference.pdf. CLIP image embeddings are close to sparse non-negative sums of 1–2-word concepts once each modality is mean-centred. Before centring, within-modality cosine is ≈0.3 and cross-modal ≈0. → Object and attribute words are recoverable from images; this is the standard modality-gap fix.
- **[A6] Interpreting CLIP's Image Representation via Text-Based Decomposition.** Gandelsman, Efros, Steinhardt. ICLR 2024. https://iclr.cc/virtual/2024/oral/19791. Many heads have "property-specific roles (e.g. location or shape)". → Some axis information exists in frozen CLIP.
- **[A7] SPAE: Semantic Pyramid AutoEncoder for Multimodal Generation with Frozen LLMs.** L. Yu, Y. Cheng, Z. Wang, … L. Jiang. NeurIPS 2023. https://ar5iv.labs.arxiv.org/html/2306.17842. A frozen LLM vocabulary is the codebook, and a CLIP-relevance loss makes the chosen words match the image. → A named codebook works when there is an explicit semantic assignment loss (A's cross-entropy).
- **[A8] LG-VQ: Language-Guided Codebook Learning.** Liang, Zhang, Wang, Li, Ye, et al. NeurIPS 2024. https://arxiv.org/abs/2405.14206. Aligns codes with text semantics and code relations with text relations. → Same intent as the Hamming-aware layout.
- **[A9] ConceptHash: Interpretable Fine-Grained Hashing via Concept Discovery.** Ng, Zhu, Song, Xiang. CVPR Workshops 2024. https://arxiv.org/abs/2406.08457. One sub-code per concept token, with language guidance; sub-codes align with object parts. → Closest hashing precedent (a workshop paper; concepts are discovered, not caption-named).
- **[A10] Efficient Object Category Recognition Using Classemes.** Torresani, Szummer, Fitzgibbon. ECCV 2010. https://www.microsoft.com/en-us/research/publication/efficient-object-category-recognition-using-classemes/. The descriptor is the outputs of many named category classifiers; a 200-byte version supports category queries. → The earliest named-concept retrieval code.
- **[A11] Deep Hashing with Semantic Hash Centers for Image Retrieval.** Chen, Liu, Zhou, Ma, Chen, Zhang. ACM TOIS 2025 (venue confirmed via Semantic Scholar DOI 10.1145/3749983). https://arxiv.org/abs/2507.08404. Fixed per-class codes, with related classes placed closer in Hamming distance under a minimum-distance constraint; beats data-independent centres such as [A12]. → Direct support for a Hamming-aware layout of fixed concept codes.
- **[A12] Central Similarity Quantization for Efficient Image and Video Retrieval.** Yuan, Wang, Zhang, Tay, Jie, Liu, Feng. CVPR 2020. https://arxiv.org/abs/1908.00347. Fixed, mutually distant Hadamard centres. → The uniform layout that [A11] improves on.
- **[A13] Pseudo-Gray coding.** Zeger, Gersho. IEEE Trans. Commun. 38:2147–2158, 1990. https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/26.64657. **[A14] A study of vector quantization for noisy channels.** Farvardin. IEEE Trans. Inf. Theory 36:799–809, 1990. https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/18.53739. Index assignment so that Hamming distance tracks codevector distance (binary switching), and channel-optimised VQ. → The layout is this problem, with image-side concept errors as the channel. Metadata only; see the notes.
- **[A15] Polysemous codes.** Douze, Jégou, Perronnin. ECCV 2016. https://ar5iv.labs.arxiv.org/html/1609.01882. Permutes PQ indices by simulated annealing to minimise Σ w(f(d_ij))·[h(π(i),π(j)) − f(d_ij)]², adapting the pseudo-Gray ideas to ranking. → A ready objective for A's assignment step.
- **[A16] Transformer Memory as a Differentiable Search Index.** Tay, Tran, Dehghani, et al. NeurIPS 2022. https://ar5iv.labs.arxiv.org/html/2202.06991. IDs from hierarchical k-means beat atomic IDs (NQ320K, XXL model: 40.4 vs 24.0 Hits@1). **[A17] Recommender Systems with Generative Retrieval.** Rajput, Mehta, Singh, et al. NeurIPS 2023. https://ar5iv.labs.arxiv.org/html/2305.05065. Semantic IDs beat LSH IDs, which beat random IDs. → IDs structured by similarity beat random ones.
- **[A18] Deep Clustering for Unsupervised Learning of Visual Features.** Caron, Bojanowski, Joulin, Douze. ECCV 2018. https://ar5iv.labs.arxiv.org/html/1807.05520. k-means pseudo-labels with cross-entropy; uneven clusters handled by sampling uniformly over pseudo-labels. **[A19] Self-labelling via simultaneous clustering and representation learning.** Asano, Rupprecht, Vedaldi. ICLR 2020. https://arxiv.org/abs/1911.05371. Equal-size labels via Sinkhorn–Knopp. **[A20] Long-tail learning via logit adjustment.** Menon, Jayasumana, Rawat, Jain, Veit, Kumar. ICLR 2021. https://arxiv.org/abs/2007.07314. Prior-based logit shifts trade calibration for balance. → Tools and theory for A.3.

### Counter-evidence / risks
- **[A21] Language Quantized AutoEncoders: Towards Unsupervised Text-Image Alignment.** Liu, Yan, Abbeel. NeurIPS 2023. https://ar5iv.labs.arxiv.org/html/2302.00902. Uses a frozen RoBERTa codebook; the authors say its outputs are "not interpretable to human" ("an image of dog can have text representation describing … rivers"). → Word-named codewords carry no meaning without a semantic assignment loss.
- **[A22] PiCoDes: Learning a Compact Code for Novel-Category Recognition.** Bergamo, Torresani, Fitzgibbon. NIPS 2011. https://papers.nips.cc/paper_files/paper/2011/file/1896a3bf730516dd643ba67b4c447d36-Paper.pdf. Learned bits beat binarised classemes, LSH and spectral hashing at equal bits. → Naming usually costs accuracy at a fixed budget.
- **[A23] Learning Concise and Descriptive Attributes for Visual Recognition.** Yan, Wang, Zhong, et al. ICCV 2023. https://arxiv.org/abs/2308.03685. "LLM-generated attributes in a large quantity perform almost the same as random words." → Accuracy does not validate the names.
- **[A24] When and why vision-language models behave like bags-of-words, and what to do about it?** Yuksekgonul, Bianchi, Kalluri, Jurafsky, Zou. ICLR 2023. https://arxiv.org/abs/2210.01936. CLIP-style models are weak at relations, attribute binding and word order. → The relation axis is at risk in images and in CLIP-text clusters, which may form by nouns.
- **[A25] Mind the Gap: Understanding the Modality Gap in Multi-modal Contrastive Representation Learning.** Liang, Zhang, Kwon, Yeung, Zou. NeurIPS 2022. https://arxiv.org/abs/2203.02053. Images and texts are embedded "at arm's length", and the gap changes zero-shot accuracy. → Text centroids are not image prototypes without correction.
- **[A26] Manhattan Hashing for Large-Scale Image Retrieval.** Kong, Li, Guo. SIGIR 2012. https://cs.nju.edu.cn/lwj/paper/SIGIR12_MH.pdf. For 4 ordered levels, no 2-bit assignment preserves neighbourhoods under Hamming distance. → PCA-quartile digits lose their order under base-mismatch counting.
- CLIP trains only its pooled embedding into the joint space, not its patch tokens. So text centroids can name and define concepts, but need a learned map before they can serve as slot-token prototypes.

### Mathematical check
**A.1 Cross-entropy to caption clusters vs a concept bottleneck**
1. The label y_m(x) is the k-means cluster of x's axis-m caption. Log loss is proper, so the optimum is p_θ(k|x) = P(y_m = k | x). The code is ĉ_m = argmax_k p_θ(k|x), with concept accuracy E_x[max_k P(k|x)]. This is the *independent* CBM of [A1], with one categorical concept per axis.
2. A concept is identifiable only if all three hold:
   - (i) The caption fixes the cluster. Cluster agreement between two independent captions of one image caps the accuracy.
   - (ii) The features carry it. Because I(y_m; slot features) ≤ I(y_m; image), content CLIP lacks (relations, word order [A24]) sends the argmax to frequent clusters.
   - (iii) Clusters split by visual differences, not wording ([A2] prunes concepts that fail).
   Objects and attributes are likely identifiable [A5, A6]; relations are doubtful [A24]. This fits the project's 2026-09-15 relation-foil note, which I did not re-check.
3. Modality gap. Assigning an image vector to its nearest text centroid is biased: cross-modal cosine is ≈0, versus ≈0.3 within a modality [A5]. Either centre each modality and renormalise [A5], or learn the classifier by cross-entropy so the constant offset is absorbed. The content gap remains either way, since captions state things that are not visible and omit things that are. It shows up in (i) and (ii).

**A.2 Fixed bijection and Hamming-aware layout**
1. Score = Σ_m agree(b(ĉ_m(x)), b(ĉ_m(y))) with b fixed. The same concept means full agreement, so the layout orders only pairs of different concepts.
2. Under a random b, a different codon differs in 1, 2 or 3 bases with probability 9/63, 27/63 and 27/63 (mean 2.29), whatever the concept similarity, so near misses earn nothing.
3. R = 5,000 is ≈22 % of a 23k database, so the ranked list is mostly partial matches and near-miss order drives mAP@R. A no-training gain of .067 (.635 → .702) is plausible in size; I ran no test.
4. This is pseudo-Gray / channel-optimised VQ index assignment [A13, A14], and [A15] optimises it directly.
5. Limits and refinement:
   - Codons under mismatch counting have diameter 3 and 9 neighbours at distance 1, so at most 9 concepts are "near" any concept.
   - Mismatch counting ignores order: quartile 1 vs 2 costs the same as quartile 1 vs 4 [A26].
   - Refinement: optimise b with [A15], weighting pairs by the image-side confusion matrix on held-out images. That matrix is the "channel" of [A14].
6. Caveats:
   - If the .702 used caption-side concepts, it is an oracle (perfect-predictor) figure.
   - If its relevance was also defined from captions, the layout was scored against its own geometry. Rescore with dataset labels and image-predicted concepts.
   - K = 128 cannot map one-to-one onto 64 codons; merge nearest concept pairs 2-to-1.

**A.3 Uneven clusters vs KL(u‖p̄)** (u uniform, p̄ the batch-mean codon distribution)
1. Under a fixed bijection, the codon marginal is the predicted-concept marginal. Cross-entropy is calibrated at its optimum, so p̄_k = π_k (the cluster share). There the regulariser equals KL(u‖π) = −ln K − (1/K)Σ ln π_k, which is > 0 unless clusters are balanced.
2. CE + λ·KL(u‖p̄) is minimised by p(k|x) = P(k|x)/(ν_x − λ/(K p̄_k)). The boost grows as p̄_k shrinks, which is a logit adjustment towards balance [A20]. Boundary images flip to rare concepts, and the codon stops naming the caption's concept.
3. Size: KL(u‖π) ≈ H(u) − H(π) ≈ CV²/2 nats, where CV is the coefficient of variation of cluster sizes. CV = 0.5 gives ≈0.13 nats (0.19 bit of 6). CV = 1 gives KL 0.60 nats and an entropy loss of 0.42 nats (≈0.6 bit), computed numerically. The capacity cost is small; the cost to faithfulness is real.
4. On an axis the image cannot predict, the regulariser forces a spread driven by whatever varies, usually the global features. The axis becomes another hash of image identity (today's failure), and DNA-unique rises without meaning.
5. Options:
   - balanced clustering via Sinkhorn [A19], so that π = u;
   - use KL(π‖p̄) on the local axes;
   - drop the regulariser on the local axes and report any collapse.
   Inverse-size weighting [A18] is a milder version of the same tilt.

### Verdict: **Plausible**
- The interpretability side is well supported [A1–A4, A10–A17]. The retrieval side is doubtful [A4, A22, A23].
- Needs: per-axis identifiability checks (A.1), a regulariser consistent with the cluster sizes (A.3), a layout optimised on image-side confusions (A.2), and a plan for the relation axis.
- Cheaper variant: name B-trained codebooks afterwards.

---

## B. Axis-neighbour soft positives

### Supporting literature
- **[B1] Conditional Similarity Networks.** Veit, Belongie, Karaletsos. CVPR 2017. https://arxiv.org/abs/1603.07810. Learned masks select a subspace per similarity notion, trained on notion-conditioned triplets; this beats separate per-notion networks. → Slot m corresponds to notion m, with axis captions replacing the triplet labels.
- **[B2] Learning Type-Aware Embeddings for Fashion Compatibility.** Vasileva, Plummer, Dusad, Rajpal, Kumar, Forsyth. ECCV 2018. https://arxiv.org/abs/1803.09196. Type-specific projections, because one space cannot hold both similarity and compatibility. → One similarity cannot serve every axis.
- **[B3] Learning Weakly-Supervised Contrastive Representations.** Tsai, Li, Liu, Liao, Salakhutdinov, Morency. ICLR 2022. https://arxiv.org/abs/2202.06670. Same-cluster samples under auxiliary information (e.g. hashtags) are positives. **[B4] Conditional Contrastive Learning with Kernel.** Tsai, Li, Ma, Zhao, Zhang, Morency, Salakhutdinov. ICLR 2022. https://iclr.cc/virtual/2022/poster/6076. Kernel weights on the similarity of the conditioning variable replace hard conditional sampling. → B's exp(cos/τ_t) weights are this kernel, with the axis caption as the condition.
- **[B5] What Should Not Be Contrastive in Contrastive Learning.** Xiao, Wang, Efros, Darrell. ICLR 2021. https://arxiv.org/abs/2008.05659. Shared backbone; each head is invariant to all augmentations but one, so the heads capture different factors, and their concatenation is best. → Closest structural precedent: a different positive set per head.
- **[B6] Similarity Contrastive Estimation for Self-Supervised Soft Contrastive Learning.** Denize, Rabarisoa, Orcesi, Hérault, Canu. WACV 2023. https://ar5iv.labs.arxiv.org/html/2111.14585. Target w = λ·one-hot + (1−λ)·(sharpened similarity softmax with self masked), trained with cross-entropy. ImageNet-100: λ = 0.5 → 82.94, λ = 1 → 81.11, λ = 0 → 81.53. → Exactly B's target form.
- **[B7] 𝕏-Sample Contrastive Loss: Improving Contrastive Learning with Sample Similarity Graphs.** Sobal, Ibrahim, Balestriero, Cabannes, Bouchacourt, Astolfi, Cho, LeCun. ICLR 2025. https://arxiv.org/html/2407.18134. The target is a batch softmax of caption-embedding cosines, trained with cross-entropy; attribute and background results beat CLIP. → Caption-derived soft targets do train image encoders.
- **[B8] With a Little Help from My Friends: Nearest-Neighbor Contrastive Learning of Visual Representations.** Dwibedi, Aytar, Tompson, Sermanet, Zisserman. ICCV 2021. https://arxiv.org/abs/2104.14548. Nearest neighbours from the dataset serve as positives. **[B9] Supervised Contrastive Learning.** Khosla, Teterwak, Wang, Sarna, Tian, Isola, et al. NeurIPS 2020. https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html. All same-class samples are positives. → B's hard-neighbour and hard-cluster limits.
- **[B10] Unsupervised Hashing with Semantic Concept Mining.** Tu, Mao, Lin, Cai, Qin, Wang, Wei, Huang. Proc. ACM Manag. Data (Semantic Scholar lists 2022). https://arxiv.org/abs/2209.11475. Concepts mined with a VLM define an image–image similarity that weights a contrastive hashing loss. → Text-guided positives in unsupervised hashing, but one similarity for the whole code, not one per axis.

### Counter-evidence / risks
- **[B11] Understanding the Behaviour of Contrastive Loss.** Wang, Liu. CVPR 2021. https://arxiv.org/abs/2012.09740. Temperature sets the penalty on hard negatives (the "uniformity–tolerance dilemma"). → The current per-pair temperature changes how hard neighbours are pushed, never whether they are pulled (D.2).
- Feature suppression [D1] acts on the α share of the target. Large α lets local slots learn easy instance features first.
- CLIP text sits in a narrow cone [A5], so τ_t must match the small spread of cosines, or the embeddings must be centred per axis. Bag-of-words text [A24] can make the relation target overlap the object target.

### Mathematical check
Setup: for slot m and anchor i, p_ij = exp(s_ij/τ)/Σ_k exp(s_ik/τ). The target is q_ij = α·[j = i] + (1−α)·r_ij, with r_ij ∝ [j ≠ i]·exp(cos(t_i, t_j)/τ_t), where t is the axis-m caption embedding.
1. −Σ_j q_ij log p_ij = H(q_i) + KL(q_i‖p_i). The captions fix H(q_i), so training minimises the KL term.
2. KL = 0 iff p_i = q_i.
   - If the model softmax runs over the target's candidates (for α = 0, drop j = i from both), s_ij/τ = cos(t_i, t_j)/τ_t + c_i achieves this in every batch. Slot m's similarity then becomes a rescaled copy of the axis-m caption similarity.
   - For α > 0 the match holds on average only, because the self weight depends on the batch.
   - Either way, the optimum depends on q^(m) and so differs by slot.
   - Caution: with α = 0, masking self in r but keeping it in the model softmax gives q_ii = 0, which pushes an image's two views apart.
3. **Identical slots are not optimal once the targets differ.** For a shared p_i, exactly
   Σ_m KL(q^(m)_i‖p_i) = Σ_m KL(q^(m)_i‖q̄_i) + M·KL(q̄_i‖p_i),
   where q̄_i is the mean target. The best shared p_i is q̄_i. It still pays **M·JS(q^(1)_i, …, q^(M)_i)** more than separate slots can reach, and JS = 0 only if all targets are equal.
4. **One-hot case.** If q^(m)_i = e_i for all m, then JS = 0 and identical slots are optimal (see the Diagnosis).
5. **How different is enough?** Any JS > 0 excludes identical slots in principle. In practice the pressure must beat the shared router and backbone and the α term.
   - α: JS is jointly convex (it is the mutual information between slot index and candidate), so JS(q) ≤ (1−α)·JS(r).
   - τ_t: as τ_t → ∞, r becomes uniform and JS → 0. Without masking, as τ_t → 0, r → e_i (an image's own caption has cosine 1), so again JS → 0.
   - Your overlaps: with each target spread evenly over its full-data 20-NN set, pairwise JS = (1−ω)·ln 2 = 0.60–0.67 nats for ω = .04–.14 (maximum 0.69). For four axes, JS = ln 4 − 3ω·ln 2 = 1.10–1.30 nats (maximum 1.39). That is near maximal.
   - But on average only 20(B−1)/(N−1) of an anchor's 20 neighbours are in the batch: ≈1.0 at B = 256 if N ≈ 5,000. N is inferred from the .004 random baseline (= 20/N); please check it. In-batch targets are therefore set mostly by ordinary pairs, which the 20-NN overlap does not measure.
   - Remedy: build q^(m) over a caption-embedding queue ([B8] support set, [B6] memory buffer), and first measure the in-batch JS offline. That check needs no training.
6. **Reachability.** A slot can match only the part of q^(m) its image features predict. For axes CLIP does not encode [A24], that part is likely the component shared with other axes, so partial redundancy returns.
7. **Gradient.** ∂L/∂s_ij = (p_ij − q_ij)/τ < 0 whenever q_ij > p_ij, so axis neighbours are pulled together. The current design never pulls (D.2).

### Verdict: **Supported, with conditions**
- The strongest theory of the three (step 3 is an exact identity), with direct precedent [B1, B3–B7].
- Mask self in r and keep α small but positive on local slots. SCE's single-head optimum was λ = 0.5; a smaller α gives more pressure. With α = 0 (C + B), drop the positive from the model softmax.
- Use a target queue, or verify the in-batch JS first. Centre the axis embeddings and tune τ_t.
- Expect a partly redundant relation axis.
- Distinct slot geometry should yield distinct codons, since k-means cells follow the geometry [D10], but this must be measured.

---

## C. Division of labour (instance discrimination only on the global slot)

### Supporting literature
- **[C1] VICRegL: Self-Supervised Learning of Local Visual Features.** Bardes, Ponce, LeCun. NeurIPS 2022. https://arxiv.org/abs/2210.01571. Global and local criteria applied together, trading segmentation against classification. → A global/local split is established, and it is a trade-off.
- **[C2] Unsupervised Hashing with Contrastive Information Bottleneck.** Qiu, Su, Ou, Yu, Chen. IJCAI 2021. https://ar5iv.labs.arxiv.org/html/2105.06138. Contrastive loss on the full binary code. mAP at 16/32/64 bits: CIFAR-10 .590/.622/.641; MSCOCO .737/.760/.775. → Label-level mAP barely depends on code length.
- **[C3] On Variational Bounds of Mutual Information.** Poole, Ozair, van den Oord, Alemi, Tucker. ICML 2019. https://ar5iv.labs.arxiv.org/html/1905.06922. "I_NCE is upper bounded by log K", and I_NCE is a lower bound on mutual information. → Used in step 2.
- Structural precedents: PCBM-h's concepts plus residual [A4], LooC's per-head objectives [B5], CSN [B1].

### Counter-evidence / risks
- SCE [B6] with no instance term (λ = 0) lands 1.4 points below the best mix, though not below instance-only. A small α on the local slots may help quality, at some cost to differentiation (B, step 5).
- VICRegL [C1] and the project's own observation show that splitting objectives can cost global retrieval. Concept-only bottlenecks need a residual [A4]; under C that residual is only 3 of 15 bases.

### Mathematical check
1. **Capacity.** The code is c = (c_0, …, c_4), one of 64 codons each. H(c_0) ≤ 6 bits and H(c) ≤ log₂ 64⁵ = 30 bits (≈1.07×10⁹ codes). The global codon alone leaves ≈359 images per bucket (23k) or ≈1,672 (107k) on average.
2. **What the contrastive loss can buy.** On a 64-valued code, I ≤ ln 64 = 4.16 nats. With I_NCE = ln B − L ≤ I [C3], a contrastive loss on a deployed codon is ≥ ln B − ln 64 (1.39 nats at B = 256), so one codon separates at most 64 images. This holds for all five slots today: instance discrimination on continuous tokens mostly buys information the codon then discards.
3. **Uniqueness.** Unique database codes need H(c) ≥ log₂ N_db (14.5 / 16.7 bits). With only c_0 instance-trained, each local axis must add ≥ 2.1 (Flickr) or 2.7 (MSCOCO) bits of H(c_m | c_0). The expected number of other items sharing a full code is (N_db−1)·2^(−H₂(c)), with H₂ the collision entropy: at 12 bits ≈5.6 / 26, at 16 bits ≈0.4 / 1.6.
4. **What mAP@R actually needs.** It needs I(code; labels), not unique codes: CIBHash gains only .05 / .04 from 16 to 64 bits [C2]. C's real risks are:
   - (a) coarser local codes → more ties (15-base agreement has only 16 score levels);
   - (b) label-irrelevant axes (colour/texture vs Flickr tags) add unweighted noise;
   - (c) the only instance-trained slot carries 3/15 of the score.
5. **Against the status quo.** Since one codon is today about as informative as all five, the effective code is ≈1 codon. C loses deployed information only if the local codes end up with lower entropy than now. Its upside is exactly H(c_1..c_4 | c_0).

### Verdict: **Supported as the enabling step; not enough alone**
Any slot-specific term breaks the symmetry. But the shared instance term is 82 % of the objective, and feature suppression [D1] favours its easy features. C removes that competition at a small capacity cost relative to today (step 5). Without A or B, the local slots have no objective at all.

---

## Diagnosis: why every slot carries the same image-level information

### Supporting literature
- **[D1] Intriguing Properties of Contrastive Losses.** Chen, Luo, Li. NeurIPS 2021. https://arxiv.org/abs/2011.02803. "A few bits of easy-to-learn shared features can suppress, and even fully prevent, the learning of other sets of competing features."
- **[D2] What Makes for Good Views for Contrastive Learning?** Tian, Sun, Poole, Krishnan, Schmid, Isola. NeurIPS 2020. https://arxiv.org/abs/2005.10243. **[D3] Self-supervised Learning from a Multi-view Perspective.** Tsai, Wu, Salakhutdinov, Morency. ICLR 2021. https://mlanthology.org/iclr/2021/tsai2021iclr-selfsupervised-a/. A representation keeps what the views share, and an input and its augmentation are "two redundant views". → All slots see the same views, so they have the same information to learn.
- **[D4] Self-supervised Product Quantization for Deep Unsupervised Image Retrieval.** Jang, Cho. ICCV 2021. https://ar5iv.labs.arxiv.org/html/2109.02244. Disjoint sub-vectors, each with its own codebook; the contrastive loss compares full descriptors, not one codebook at a time. **[D5] Product Quantization for Nearest Neighbor Search.** Jégou, Douze, Schmid. IEEE TPAMI 33:117–128, 2011. https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/TPAMI.2010.57. A Cartesian product of disjoint subspaces. → In PQ hashing, sub-codes cannot copy each other.
- **[D6] Object-Centric Learning with Slot Attention.** Locatello, Weissenborn, Unterthiner, et al. NeurIPS 2020. https://ar5iv.labs.arxiv.org/html/2006.15055. Attention is "normalized over the slots", and each slot is decoded to reconstruct the image. **[D7] Bridging the Gap to Real-World Object-Centric Learning.** Seitzer, Horn, Zadaianchuk, et al. ICLR 2023. https://arxiv.org/abs/2209.14860. Decomposition of real images works when slots reconstruct self-supervised features. → Competition splits information only when the loss rewards splitting.
- **[D8] Vision Transformers Need Registers.** Darcet, Oquab, Mairal, Bojanowski. ICLR 2024. https://proceedings.iclr.cc/paper_files/paper/2024/hash/0b408293619f725fd30162af057e531a-Abstract-Conference.html. High-norm background tokens in DeiT-III, OpenCLIP and DINOv2 hold global image information. → Any slot receiving them can identify the image. This fits the project memory note on extreme-norm tokens in Flickr's primary_object slot, which I did not re-check.
- **[D9] Adaptive Multi-head Contrastive Learning.** Wang, Koniusz, Gedeon, Zheng. ECCV 2024. https://arxiv.org/html/2310.05615. Heads get identical positives with per-pair adaptive temperatures and differ only by initialisation and temperature; redundancy is not measured. → The published design closest to the current one, with no evidence that heads specialise.
- **[D10] K-means Clustering via Principal Component Analysis.** Ding, He. ICML 2004. https://icml.cc/Conferences/2004/proceedings/papers/262.pdf. Principal components are the continuous solution of the k-means cluster indicators.

### Counter-evidence / risks
- [D9] and [B6] report gains in accuracy, not specialisation, so they do not contradict the diagnosis.
- The diagnosis explains the redundancy, not why differentiating slots costs retrieval. For that, see [C1] and [A4].

### Mathematical check
1. **Symmetry.** L = Σ_m L_m(S^m), where S^m is slot m's similarity matrix and all slots share a one-hot target. No term couples two slots. With a shared temperature, S^m = S* for every m minimises every term, so identical slots are an exact global minimiser and duplicating the best slot costs nothing.
2. **Temperatures only push.** L_i = −s_ii/τ_ii + log Σ_j exp(s_ij/τ_ij) gives ∂L_i/∂s_ij = p_ij/τ_ij > 0 for all j ≠ i. Every other image is repelled; axis similarity changes only how strongly, and there is never a pull (compare B, step 7). In a high-dimensional slot space, weaker repulsion leaves neighbours near-orthogonal rather than close (reasoning, consistent with [B11]).
3. **Which shared solution wins.** The easiest instance features, in every slot [D1], because all slots see the same views [D2, D3].
4. **The router divides tokens, not information.** Late CLIP tokens carry global information [D8], and competition divides information only under losses that reward it [D6, D7].
5. **One codon ≈ five.** Each codon holds at most 6 bits, and VQ cells follow each slot's dominant variance directions [D10]. If those directions are the same instance-level ones in every slot, the five 64-way partitions nearly coincide (an inference, not tested).
6. **The PQ contrast.** Disjoint slices plus a loss on the joint code [D4, D5, C2] forbid copying and reward complementary sub-codes: one 6-bit sub-code cannot separate a batch of 256. Per-slot InfoNCE has neither property.
7. **The 14 failed mechanisms.** Anything that keeps the per-slot target one-hot and identical (routing, temperatures, codebook or codon regularisers) keeps the symmetric optimum. Only three changes break it:
   - changing the target (B);
   - removing it from the local slots (C);
   - penalising redundancy between whole similarity matrices.
   A linear decorrelation penalty can be met by rotated copies of the same content, since cosines are rotation-invariant (reasoning).

### Verdict: **Supported**
The failure is what the objective predicts (steps 1–2), and the literature explains which shared features win. Designs that avoid it change the positives per head [B5], use disjoint subspaces with a joint loss [D4, D5], or reward splitting [D6, D7].

---

## Verification notes
- [A13] and [A14]: only the metadata was opened (Semantic Scholar API); the IEEE pages did not render. The mechanism wording comes from the IEEE abstract as returned by search and from [A15] — **UNVERIFIED at source**.
- Not used and **UNVERIFIED**: "Image Retrieval with Well-Separated Semantic Hash Centers" (ACCV 2022, page returned 403), and the fairness variant of Tsai et al.'s conditional contrastive learning (not checked).
- [B10]'s "SIGMOD 2023" (from a search snippet) is **UNVERIFIED**; Semantic Scholar says Proc. ACM Manag. Data, 2022.
- [A6]: only the abstract was checked; heads for colour or texture are **UNVERIFIED**.
- [A9] is a workshop paper. Venues that arXiv does not state were confirmed via proceedings pages, ML Anthology or the arXiv comment field.
- N ≈ 5,000 is inferred, not read from the repository.
- No statistical test underlies any verdict; .635 vs .702 is a reported difference.
