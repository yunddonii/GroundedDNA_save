# Summary — post-v62b 실험 종합 정리 + 향후 방향

작성일: 2026-05-21

대상 실험 (v62b SOTA 갱신 이후 직전 24시간 사이클):
- **v62 (Option A)**: Residual-conditioned codon head. Flickr25k에서 새 SOTA 갱신.
- **v62b → MSCOCO 이식 (mscoco_v62b)**: v63b 위에 γ=0.3 residual.
- **v64a/b (Option α)**: EMA codeword repulsion (Gaussian, auto-sigma) on Flickr25k + MSCOCO.
- **v65a/b**: codon head fc capacity 확장 (256→64→4 MLP, 256→128→4 MLP) on Flickr25k.
- **v66 (Option #2, 진행 중)**: Per-codon text-anchored prototype classifier.

근거 baseline:
- Flickr25k: **v62b** (0.6778) = v57 + residual γ=0.3
- MSCOCO: **v63b** (0.4563) = v57 setup + K=128

---

## 1. Flickr25k 종합

### 1.1 최종 mAP 표 (final test, 2K query × 23K db)

| Run | Setting (delta) | mAP | Δ vs v62b | unique | per-cb-unique | base-entropy |
|---|---|---:|---:|---:|---:|---:|
| **v62b ★** | v57 + residual γ=0.3 | **0.6778** | — | 0.0745 | 0.00077 | 0.900 |
| v65b | v62b + codon MLP h=128 | 0.6702 | −0.0076 | 0.080 | 0.00060 | 0.804 |
| v57 (baseline) | + lambda_wass 0.05 | 0.6683 | −0.0095 | 0.066 | 0.00049 | 0.897 |
| v64a | v57 + EMA repulsion α (str=0.01) | 0.6602 | −0.0176 | 0.083 | 0.00060 | **0.822** ↓ |
| v65a | v62b + codon MLP h=64 | 0.6301 | −0.0477 | 0.101 | 0.00080 | 0.908 |
| v66 (진행 중) | v62b + text-anchored prototype | TBD | TBD | TBD | TBD | TBD |

### 1.2 다방면 결론 (Flickr25k)

#### A. **Residual injection (v62b)이 여전히 압도적**
- v62b의 강점은 *codeword 자체는 보존하면서* 같은 codeword에 묶이는 이미지들이 codon decoding 시 서로 다른 결과를 내도록 residual `z − q`를 입력에 추가한 점.
- 후속 시도들이 모두 이 효과를 능가 못함 — *codon decoding 단계의 정보 압축은 이미 잘 튜닝되어 있다*는 강한 신호.

#### B. **Codon decoding bottleneck은 실재하나 "용량 확장"으로는 풀리지 않음**
- per-codon collision 분석 결과: codon 4가 19.3% (4443/23000) 동일 "GAA" 패턴. effective K ≈ 9-12 (max 64 중).
- 직관: `Linear(256, 4)`의 표현력이 부족 → MLP로 확장하면 풀릴 것
- 실제 결과: **v65a (h=64) −0.048**, **v65b (h=128) −0.008** — 모두 회귀.
- 가설: 단일 Linear의 *bottleneck 자체가 의미 있는 압축* 역할 → MLP가 codeword 신호를 우회해 residual continuous 정보를 직접 통과시킴.
- 함의: codon level의 collision은 capacity 문제가 아니라 **목적 함수가 codon-level 구별을 직접 supervise하지 않기 때문**.

#### C. **EMA repulsion은 sigma_factor 튜닝 실패 (보수적 시도 폐기)**
- strength=0.01, sigma_factor=0.5 → base entropy 0.900 → 0.822 (감소).
- 의도(codeword 분산 ↑)와 반대로 일부 codeword가 sparse region으로 밀려 *사실상 dead*.
- sigma_factor가 너무 큼 — 멀리 있는 pair까지 반발. 재시도 시 sigma_factor 0.1 / strength 0.001 같은 더 보수적인 값 필요.

#### D. **Per-codon-unique vs mAP 상관관계의 한계**
- v62b: per-cb-unique 0.00077, mAP 0.6778
- v64a: per-cb-unique 0.00060, mAP 0.6602 (per-cb는 비슷한데 mAP 큰 차이)
- v65a: per-cb-unique 0.00080 (v62b 동급), mAP 0.6301 ← **higher unique ≠ higher mAP**
- 함의: unique-code 개선이 mAP의 *충분조건*은 아님. **codeword가 의미적으로 일관성을 유지하면서 unique 증가**가 핵심.

---

## 2. MSCOCO 종합

### 2.1 최종 mAP 표 (final test, 5K query × 107K db, K=128)

| Run | Setting (delta) | mAP | Δ vs v63b | unique | per-cb-unique |
|---|---|---:|---:|---:|---:|
| **v63b ★** | v57 setup + K=128 | **0.4563** | — | 0.315 | 0.00077 |
| v64b | v63b + EMA repulsion α | 0.4470 | −0.0093 | 0.015 ⚠ | 0.0001 |
| mscoco_v62b | v63b + residual γ=0.3 (port v62b) | 0.4378 | **−0.0185** | 0.035 | 0.0002 |
| (외부) CIBHash | flat Linear(768, 36) | 0.5051 | +0.0488 | ~0.99 | — |
| (외부) CIMON | spectral PL + Linear | 0.4777 | +0.0214 | — | — |
| (외부) MLS3RDUH | kNN + LogCosh | 0.4434 | −0.0129 | — | — |

### 2.2 다방면 결론 (MSCOCO)

#### A. **v62b의 핵심 contribution(residual head)이 MSCOCO에선 역효과**
- Flickr25k: v57 → v62b 효과 **+0.0095** (vibrant boost)
- MSCOCO: v63b → mscoco_v62b 효과 **−0.0185** (regression)
- 원인 가설:
  - Flickr25k 5K train × K=64 → codeword 당 평균 78 images 공유 → residual로 분리하는 효과 큼
  - MSCOCO 10K train × K=128 → codeword 당 평균 78 images로 비슷하지만 **이미지 다양성이 훨씬 큼**(80 vs 24 class). residual `z − q`가 *noise*에 더 가까워서 분류 신호 해침
- 함의: "residual = useful per-image signal" 가정은 dataset 다양성에 의존.

#### B. **MSCOCO compositional structure의 강함과 약함 (v63b 특성)**
- 강함: per-cb K=128 모두 활성, dead=0, base-entropy 0.90, unique-code 0.315 (CIBHash 0.99 대비 1/3)
- 약함: P@1 0.5606 — CIBHash의 0.7802에 −0.22 격차. top-rank sharpness 부족.
- 핵심 trade-off: **compositional grouping (의미 cluster) vs instance-level discrimination**.
  - Flickr25k 24 class multi-label에선 grouping이 *enough* → 우리 모델 우세
  - MSCOCO 80 class multi-label에선 instance-level이 *필요* → CIBHash 우세

#### C. **EMA repulsion은 MSCOCO에서도 동일하게 실패**
- v64b: unique 0.315 → 0.015 (95% 감소), mAP −0.009
- 같은 sigma_factor 문제. MSCOCO에서는 K=128이라 pairwise interactions 더 많아 효과 증폭됨.

---

## 3. 양 dataset 공통 결론 + paper narrative 영향

### 3.1 SOTA 모델 paradigm의 두 갈래

| Dataset | SOTA | Mechanism |
|---|---|---|
| Flickr25k | v62b (residual γ=0.3) | codeword 보존 + codon-level fine variation |
| MSCOCO | v63b (no residual) | K=128 grouping, residual 추가 시 오히려 손해 |

**Paper narrative 시사점**: "single best model"이라는 SOTA prose가 약해짐.
대신 "**dataset-dependent trade-off**" narrative로 전환 권장:
- "Flickr25k (24 class)에선 codon-level residual injection이 효과적"
- "MSCOCO (80 class)에선 codebook capacity (K=128)만으로 충분"

### 3.2 Codon collision은 dataset 무관한 구조적 문제

| Run | Dataset | codon 4 top-1 % (worst codon collision) |
|---|---|---:|
| v62b | Flickr25k | 19.3% (4443/23000) |
| v63b | MSCOCO | TBD (구체 측정 미실시, P@1 0.56 → 비슷할 추정) |

→ codon-level decoding이 hash 표현력의 *근본 bottleneck*. Capacity (v65) / EMA repulsion (v64)으로는 풀 수 없음. 진행 중인 **v66 (text-anchored prototype)**이 유일하게 *목적 함수에 codon-level supervision을 직접 박아넣는* 시도.

### 3.3 외부 baseline 대비 위치

| | Flickr25k 우리 | Flickr25k best ext | MSCOCO 우리 | MSCOCO best ext |
|---|---:|---:|---:|---:|
| mAP | **0.6778** | 0.6543 (CIBHash) | 0.4563 | 0.5051 (CIBHash) |
| Δ ours vs best ext | **+0.0235** ✓ | — | **−0.0488** ⚠ | — |

→ Flickr25k에선 CIBHash를 명확히 능가. MSCOCO에선 −0.05 격차. **MSCOCO 격차 좁히는 것이 최우선 paper-readiness 작업**.

---

## 4. 향후 해결 방향 (우선순위 순)

### A. **v66 결과 대기 + text supervision 강화 연속 sweep** [현재 진행]
- v66 (text-anchored prototype on Flickr25k) ep59 결과 보고 판단
- 효과 검증 시 MSCOCO에 동일 변종 + temperature/λ sweep
- 효과 미검증 시 → **text-reconstruction head (Option #1)**로 전환:
  - Decoder: dna_hash → MLP → predicted text embedding
  - Loss: MSE / cosine on text reconstruction
  - Inference 안전, 가장 강한 final-code supervision

### B. **MSCOCO 특화 hyperparameter sweep** (paper-readiness)
- 외부 baseline (CIBHash) 격차 −0.05 좁히기
- 후보:
  1. **K sweep**: K ∈ {128, 192, 256} on MSCOCO — K↑로 unique 증가 + CIBHash 격차 보완
  2. **lambda_wasserstein sweep**: 0.05 → {0.1, 0.2, 0.3} — 더 강한 visual-text alignment
  3. **CodonHead 다중 fc**: 3 position에 *별도* fc (현재 공유). position-specific 학습.
  4. **Hash-text contrastive aux loss**: `NtXent(hash_code, text_global)` — final hash 직접 supervise

### C. **EMA repulsion 재튜닝** (이전 실패 보완)
- sigma_factor 0.5 → **0.1** (sharper, 진짜 가까운 pair만 분리)
- strength 0.01 → **0.001** (덜 공격적)
- repel_every 10 → **1** (매 step)
- 다시 시도해서 base-entropy 유지하면서 unique 증가하는지 확인

### D. **Per-codon position-specific weight** (구조적)
- 현재 CodonHead.fc는 3개 position에 *동일* Linear(256, 4).
- 변경: position마다 *별도* fc — 미세하지만 codon 4(worst collision)가 position 1을 공유하는 codon 1과 같은 weight 쓰는 게 collision 강화 가능성.
- 1줄 변경 (`Linear(chunk, 4)` → `Linear(3*chunk, 12)` or 3개 분리)
- 매우 가벼움 — sanity check로 한 번 돌려볼 가치.

### E. **MSCOCO 격차 → "hybrid hash" 시도**
- 우리 36-bit DNA hash (compositional) **AND** CIBHash-style 36-bit flat hash 동시 학습.
- 두 hash를 concat → 72-bit (또는 36-bit 평균/min). retrieval에선 hybrid.
- 우리 모델의 의미 grouping + CIBHash의 instance sharpness 동시 활용.

### F. **Paper narrative 재구성** (실험 외 작업)
- "single SOTA" → "dataset-paradigm contribution" narrative
- 별도 ablation: "compositional code는 어디서 이기고 어디서 지는가" 도표화
  - Flickr25k = grouping-friendly, multi-label class ≤ 24
  - MSCOCO = instance-friendly, multi-label class = 80
- → Reviewer가 "왜 single SOTA가 아니냐"에 대비 가능

---

## 5. 현재 학습 진행 중 작업

| Tag | Dataset | GPU | 예상 종료 | 의미 |
|---|---|---|---|---|
| v66 | Flickr25k | 0 | ~19:30 | text-anchored prototype codon head (Option #2 진행) |

(다른 GPU는 모두 idle. 위 우선순위 A의 v66 결과 보고 B/C 중 1-2개 launch 가능.)

---

## 6. 데이터 출처

- v62b: `result/260521+flickr25k_setting1_v62b_v57_residual_g03+bs+64+e+60+proj_lr+0.001/evaluation_siglip2_base.json`
- v64a: `result/260521+flickr25k_setting1_v64a_v57_repelStr01_sig05_every10+bs+64+e+60+proj_lr+0.001/evaluation_siglip2_base.json`
- v64b: `result/260521+mscoco_setting1_mscoco_v64b_v63b_repelStr01_sig05_every10+bs+64+e+60+proj_lr+0.001/evaluation_siglip2_base.json`
- v65a: `result/260521+flickr25k_setting1_v65a_v62b_codonMLP_h64+bs+64+e+60+proj_lr+0.001/evaluation_siglip2_base.json`
- v65b: `result/260521+flickr25k_setting1_v65b_v62b_codonMLP_h128+bs+64+e+60+proj_lr+0.001/evaluation_siglip2_base.json`
- mscoco_v62b: `result/260521+mscoco_setting1_mscoco_v62b_v63b_residual_g03+bs+64+e+60+proj_lr+0.001/evaluation_siglip2_base.json`
- 외부 baseline (MSCOCO): `result_baseline/260521/*_mscoco_unsup60/eval_epoch_059.json`
- v62b codon collision 분석: 본 conversation Bash 출력 (재현 가능 — `extract_db.npz`의 `base_indices` 사용)
