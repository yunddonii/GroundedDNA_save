# Plan: 높은 unique-code ratio를 가진 compositional hash 설계

작성일 2026-05-20. 사용자 요청: "CIBHash 처럼 미세하게 이미지마다 다른 hash code를
할당하면서도 compositional code를 생성하는 방법을 고안. 구조적 수정 가능."

현재 v57 (absolute SOTA, mAP 0.6683 test) vs CIBHash (mAP 0.6543) 비교 시 **mAP는
우리가 +0.014 우세**이지만 unique-code ratio는 **v57 0.409 vs CIBHash 0.997**로
극단적 차이. 본 문서는 (1) 이 차이의 메커니즘적 원인을 진단하고, (2) compositional
구조를 보존하면서 unique를 높이는 architectural 처방안 3가지를 제시한다.

---

## 1. 진단 — 왜 v57은 unique가 0.409인가

### 1.1 정량적 사실 (v57 extract_db.npz 분석)

| 지표 | v57 | CIBHash |
|---|---:|---:|
| Unique full-code | 9419 / 23000 (40.9%) | ~22931 / 23000 (99.7%) |
| Singleton (size=1) clusters | 5235 (55.6% of codes, 22.8% of data) | ~22000 |
| Top-1 cluster size | 49 samples | 1-2 |
| Top-10 cluster sum | 400 samples (1.7%) | ~10-20 |
| 50% data를 덮는 # of codes | 1550 | ~11500 |
| Per-cb codebook 사용 | 64/64 (100%) | n/a (no codebook) |
| Per-cb top-1 codeword 점유율 | 2.21% ~ 3.25% | n/a |

### 1.2 v57이 collision을 만드는 구조적 이유 (단계별)

코드 생성 파이프라인:
```
visual_token [B, N, D]
    ↓ Sinkhorn router (with text centroids)
semantic_visual_token [B, 6, D]           ← (a) bottleneck #1: 6개 part로 routing
    ↓ codebook quantizer (per-cb K=64, EMA)
quantized_token [B, 6, D]                  ← (b) bottleneck #2: 각 part는 K=64 codeword 중 하나
    ↓ codon head[m=0..5]
codon_logits [B, 6, 3, 4]                  ← (c) Gumbel softmax over 4 bases × 3 codons
    ↓ argmax / Gumbel sampling
DNA code [B, 6, 3] ∈ {0,1,2,3}^18 = 36-bit
```

각 단계의 collision 메커니즘:

**(a) Sinkhorn routing의 collision pressure**: balanced OT는 모든 patch를 6 part에
*거의 균등*하게 분배. 비슷한 visual content를 가진 두 이미지는 *동일한 routing
matrix*를 거의 얻고 → 동일한 `semantic_visual_token`을 얻음.

**(b) Codebook quantization의 collision pressure** (★ 가장 큰 원인):
- K=64로 768-d → 64 discrete 매핑 → 정보 손실 양이 큼 (`log2(64) = 6 bit per slot`)
- 6 slot × 6 bit = **36 bit 정보 한계**. 이론상 최대 unique = 2^36 = 6.9×10^10
  이지만 실제 codebook 활용은 균등하지 않음 (Gini 0.10~0.29)
- EMA codebook은 *visual feature 평균*에 자석처럼 끌림 → 비슷한 visual은 같은
  codeword로 quantize됨 → identical `quantized_token`
- VQ commitment loss (`lambda_vq=0.25`)는 이걸 *더 강하게* 압박

**(c) Codon head의 collision pressure**:
- Codon head는 **`quantized_codeword`만 보고** codon distribution을 출력 (residual
  무시) — 즉 `codon = head(codeword_index의 embedding)`
- 동일 codeword → 동일 codon → 동일 36-bit code
- 이게 가장 결정적: 같은 codeword에 매핑된 K개 이미지는 *원리적으로 동일한
  hash code*를 받음

### 1.3 v57이 mAP는 높으면서 unique는 낮은 이유

비교: CIBHash는 `sign(Linear(visual_feat, 36))`로 *연속* 36-d projection을 hard
sign으로 binarize. 어떤 미세한 visual 차이도 다른 36-bit code를 만들 수 있음
(36 차원의 continuous 공간 → 2^36 discrete 공간 직접 mapping).

v57은 routing + quantization으로 *의미적 cluster*를 만들고 *cluster 안에서는 같은
code* 부여. 따라서:
- 의미적으로 유사한 두 이미지 → 같은 code → 함께 retrieve (mAP ↑)
- 의미적으로 유사한 두 이미지 → 다른 code → CIBHash 식. retrieval에서 miss
  가능 (mAP CIBHash 패배 원인)

**결론**: v57의 collision은 *retrieval에 유익한 collision* (의미적 응집). 그러나
*compositional structure의 풍부함*은 손해. 두 가지를 동시에 만족시키는 방법이
필요.

---

## 2. 설계 목표

다음 셋을 동시에 만족하는 architecture:

| 목표 | 측정 metric |
|---|---|
| 1. 높은 mAP (v57 0.6683 유지 또는 갱신) | Flickr25k mAP |
| 2. 높은 unique-code ratio (≥ 0.7, CIBHash 수준에 근접) | unique = #distinct / N |
| 3. Compositional structure 유지 (paper claim #1) | B0/B1/B2 lift, per-cb specialization |

기존 v57은 (1) 우세 / (2) 약함 / (3) 우세. **(2)만 끌어올리면 됨.**

---

## 3. 3가지 architectural 처방안

3가지 모두 *codon head 단계*에 개입하는 옵션. routing/codebook은 그대로 유지
(compositional 보존). codon head를 *residual에 conditioned*하게 만들어서 같은
codeword라도 image별로 다른 codon을 출력하게 함.

### Option A — Residual-Conditioned Codon Head ★ 가장 적은 변경

**아이디어**: codon head 입력에 residual 정보 추가.

현재:
```
codon_logits[cb] = codon_head[cb](quantized_codeword[cb])
                                   ↑
                          [D] embedding, image별 동일 (같은 codeword일 때)
```

수정:
```
residual[cb] = visual_token[cb] - quantized_codeword[cb]    # [D], image별 다름
codon_input[cb] = concat([quantized_codeword[cb], γ · residual[cb]])  # [2D]
codon_logits[cb] = codon_head[cb](codon_input[cb])
```

`γ` ∈ [0, 1] hyperparameter — residual의 영향력 조절. γ=0이면 v57과 동일.
γ→1이면 codon이 *완전 residual-driven* (compositional 의미 약화).

**장점**:
- 코드 변경 최소 (codon_head 입력 차원 D → 2D, head MLP 첫 layer만 수정)
- compositional structure 명시적 (cb별 codeword index 그대로)
- residual 가중치 γ로 unique vs mAP trade-off 조절 가능

**위험**:
- codon head 파라미터 약간 증가 (D → 2D 입력)
- γ가 너무 크면 codon이 codebook 의미와 무관해짐

**Hyperparameter**: γ ∈ {0.1, 0.3, 0.5, 1.0}.

**예상 효과**:
- γ=0.1: unique 0.5~0.6 / mAP +0.000~−0.002
- γ=0.3: unique 0.7~0.8 / mAP ±0.005
- γ=1.0: unique 0.9+ / mAP −0.01~−0.02 (compositional 약화 위험)

### Option B — Residual Vector Quantization (Multi-stage VQ)

**아이디어**: codebook 자체를 2단계로 — primary VQ (compositional) + residual VQ
(fine).

```
visual_token [B, 6, D]
    ↓ primary quantize (K_p = 64, compositional)
quantized_primary [B, 6, D] + primary_index [B, 6]
    ↓ residual = token - quantized_primary
    ↓ secondary quantize (K_s = 16, fine)
quantized_secondary [B, 6, D] + secondary_index [B, 6]

codon_input = concat([quantized_primary, quantized_secondary])
codon_logits = codon_head(codon_input)
```

또는 더 간단하게: codon head가 (primary_index, secondary_index)를 직접 보고 코드 출력.

**장점**:
- 표상 공간 확장: K_p × K_s = 64 × 16 = 1024 effective codewords per cb
- residual_codebook은 *fine-grained variation*에 특화
- RVQ는 audio/video tokenization (Encodec, RVQGAN)에서 검증된 기법

**단점**:
- 코드 변경 큼 (quantizer 2단계, codon_head 입력 dim 2D)
- 두 codebook의 학습 dynamics 더 복잡 (둘 다 EMA 또는 한쪽만)
- "secondary codebook"이 compositional 해석에 잘 들어맞지 않음 — paper narrative 약화

**Hyperparameter**: K_s ∈ {4, 8, 16}.

**예상 효과**: unique 0.8+ / mAP ±0.005, B1/B2 약간 손실.

### Option C — Image-specific Stochastic Codon Augmentation

**아이디어**: codon Gumbel sampling 단계에 *image-specific 작은 perturbation*을
더해서 미세하게 다른 code 출력.

```
codon_logits[cb] = codon_head[cb](quantized_codeword[cb])  # [3, 4]
img_perturb[cb] = small_mlp(visual_global)                  # [3, 4], image별 다름
adjusted_logits = codon_logits + λ_p · img_perturb
hard_code = argmax(gumbel_softmax(adjusted_logits))
```

`λ_p` 작으면 codon은 거의 codeword에 의존하지만, 가끔 perturbation이 boundary
flip을 만들어서 image마다 *조금씩 다른* code.

**장점**:
- 가장 minimal — codon head 자체는 그대로
- λ_p로 collision↔unique trade-off 직관적 조절

**단점**:
- compositional 의미 유지에 도움 안 됨 — 단지 boundary noise 주입
- inference 시 deterministic이지만 image_global에 의존하는 perturbation이 추가됨
- "왜 그 코드?"의 해석 가능성 약화

### 비교 요약

| Option | 코드 변경 | Compositional 보존 | 예상 unique | 예상 mAP | 권장 우선순위 |
|---|---|:-:|---:|---:|:-:|
| **A. Residual-Conditioned Codon Head** | 작음 (codon_head 입력 D→2D) | ✓✓ | 0.5-0.9 | ±0.005 | **★ 1st** |
| B. RVQ (Multi-stage codebook) | 큼 (quantizer 2단계) | ✓ | 0.7-0.9 | ±0.005 ~ −0.01 | 2nd |
| C. Stochastic Perturbation | 가장 작음 | △ | 0.5-0.8 | ±0.005 | 3rd |

---

## 4. 실행 계획 (Option A 기준)

### Phase 1 — Minimal-change 검증 (1-2 runs)
1. `models/codon_head.py` 또는 `model_siglip2.py`의 codon head 구성 위치에서:
   - Codon head 입력 D → 2D로 확장
   - `residual[cb] = visual_token[cb] - quantized_codeword[cb]` 계산 (no_grad 또는 grad)
   - `codon_input = concat([quantized, γ · residual])`
2. CLI flag 추가: `--codon_residual_gamma <float>` (default 0.0 = v57과 동일)
3. v57 setup 위에 γ=0.3 1 run (v62a)
4. ep9/19/29 mid-eval로 unique/mAP/B1/B2 trajectory 확인

### Phase 2 — γ sweep (2-3 runs)
γ ∈ {0.1, 0.3, 0.5} 3-run 병렬. mAP-unique 산점도 확보.

### Phase 3 — Compositional metric 확인
가장 좋은 γ로 학습한 v62a 변형의 B0/B1/B2 + 의미 grounding 검증.

### Phase 4 — Failure path (만약 Option A가 mAP 손해 크면)
→ Option B (RVQ)로 전환. K_s=8 시도.

---

## 5. 측정 지표 (success criteria)

### Primary
- **mAP** ≥ 0.66 (v57의 0.668 대비 −1% 이내 손해 허용)
- **unique ratio** ≥ 0.7 (v57의 0.41 대비 +0.3 이상 개선)

### Secondary
- **B1** ≥ 0.05 (v57과 동등 수준 유지)
- **B2** ≥ 0.03 (v57과 동등)
- **per-cb specialization**: 모든 6 cb의 B1 distributed (cb별 0.04-0.10 범위)

### Failure criteria (즉시 폐기)
- mAP < 0.65 또는 unique < 0.5 → 시도 자체가 의미 없음

---

## 6. 잠재적 후속 — Hybrid 강화안 (옵션)

Option A + Option B 결합도 가능:
```
visual_token → primary VQ (K_p=64) → residual
codon_head(concat([primary_codeword, γ · residual, secondary_codeword]))
```
세 항 모두 input. 더 강력하지만 학습 어려움.

Phase 1/2 결과 본 다음 결정.

---

## 7. 시급도 vs paper narrative 관점

현재 v57 mAP 0.6683은 CIBHash 대비 +0.014로 *충분히 우세*. paper에서는:
- "**mAP에서 CIBHash 추월** + **compositional structure 정량 입증**"
- 만약 v62 (Option A)가 mAP 유지하면서 unique를 0.7+로 끌어올리면:
  - "**unique-code ratio까지 CIBHash 수준 + compositional structure**"
  - paper의 contribution claim이 강화됨 (모든 axis에서 우위)
- mAP 손해가 −0.005~−0.01이라도 OK — unique 0.7+ 자체가 새로운 ablation 결과로
  paper에 추가 가능

→ **결론: Option A는 무위험 시도이며 paper 영향력 확장에 효과적**. 작업 우선순위 높음.
