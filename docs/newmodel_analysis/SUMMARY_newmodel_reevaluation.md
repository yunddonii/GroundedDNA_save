# 신규 모델 재평가 요약 (2026-08-04)

> 대상 모델 = **통일 레시피**: `jd_loss`(전 슬롯) + `--no_gumbel_softmax` + bijection OFF,
> 데이터셋별 λ. `DRAFT_GROUNDEDDNA_PAPER_KO.md`의 실험 항목을 이 모델로 재수행한 결과.
> 이전 수치는 대부분 pre-P0 / pre-bio / 구 레시피 기반이라 draft 자체가 "main causal table에
> 사용 금지"로 표시해 두었던 것들이다.

## 0. 확정된 최종 구성

| dataset | λ | 기타 | prompt |
|---|---:|---|---|
| MS-COCO | 0.03 | noGumbel | v5b |
| NUS-WIDE | 0.05 | noGumbel | v4 |
| Flickr25k | 0.02 | noGumbel | v4 |
| CIFAR-10 | 0.03 | noGumbel + **bijection OFF** | v4 |

bijection(`--lambda_codeword_codon_sinkhorn`)은 원래 **CIFAR에만** 0.1로 켜져 있었고, 끄는 것이
전 축에서 유리했다(val +0.0227). 이로써 4개 데이터셋 아키텍처가 통일된다.

## 1. 🔴 3-seed 검증 — 단일 seed 결론이 뒤집힘

seeds {42, 43, 44}, mean ± sample std.

| dataset | mAP@R | **vs best `U0`** | codon decode | DNA-uniq |
|---|---|---:|---|---|
| **Flickr25k** | 0.8668 ± 0.0017 | **+0.0306** (OH) | 0.7694 ± 0.0017 | 0.4681 ± 0.0156 |
| **NUS-WIDE** | 0.8283 ± 0.0008 | **+0.0260** (OH) | 0.7368 ± 0.0026 | 0.2116 ± 0.0079 |
| **MS-COCO** | 0.8232 ± 0.0098 | **−0.0025** (CroVCA) | 0.6466 ± 0.0048 | 0.1975 ± 0.0106 |
| **CIFAR-10** | 0.8940 ± 0.0033 | **−0.0028** (CIBHash) | 0.8903 ± 0.0104 | 0.1300 ± 0.0059 |

**단일 seed(42)에서 보였던 "4/4 baseline 전승"은 성립하지 않는다. 3-seed에서는 2승 2패다.**

- **MS-COCO**: seed 42의 0.8287(+0.0030 우세)이 3-seed 평균 0.8232(−0.0025 열세)로 회귀.
  seed 43이 0.8120으로 낮고 **std 0.0098 > CroVCA 격차 0.0087**. 열세는 해소되지 않았고,
  A-champion(−0.0087) 대비 격차를 **약 3분의 1로 좁힌** 것이 정확한 서술이다.
- **CIFAR-10**: A-champion은 CIBHash 대비 +0.0090 **우세**였으나 신규 모델은 −0.0028 **열세**.
  즉 CIFAR는 검색에서 **승 → 패로 후퇴**했다. 이것이 신규 모델의 가장 큰 비용이다.
- NUS-WIDE·Flickr25k는 std가 0.0008 / 0.0017로 매우 안정적이며 baseline 우위가 확고하다.

**A-champion 대비 검색 변화**: MSCOCO +0.0062, NUS +0.0021, Flickr −0.0007, CIFAR −0.0118.

**해석성은 4/4 개선이며 std 대비 이득이 크다**: +0.0322 / +0.0239 / +0.0227 / +0.0101
(decode std 0.0017–0.0104). 다양성도 4/4 개선.

## 2. §4.6 Bio-projection 효과 — 재측정

draft의 legacy 표(provenance 없음)를 각 run의 자체 extraction에서 다시 계산.

| dataset | pre mAP@R | post mAP@R | Δ | mean DB edits | valid pre | valid post | DNA-uniq pre→post |
|---|---:|---:|---:|---:|---:|---:|---|
| MS-COCO | 0.8317 | 0.8287 | **−0.0031** | 0.793 | 46.4 % | 100 % | 0.2033 → 0.1954 |
| NUS-WIDE | 0.8297 | 0.8274 | **−0.0024** | 0.631 | 56.5 % | 100 % | 0.2272 → 0.2158 |
| Flickr25k | 0.8756 | 0.8673 | **−0.0084** | 1.108 | 40.5 % | 100 % | 0.4907 → 0.4703 |

- projection 비용은 −0.0024 … −0.0084로 legacy 표(−0.0037 … −0.0088)와 같은 범위이며,
  **feasibility 100 %는 여전히 값싸게 얻어진다**.
- ⚠️ 사전 유효율이 40–57 %로, 균등 4^L 기대치(45.2 %)와 비슷하거나 그 이하다. 즉
  **신규 모델도 제약을 학습하지 않는다** — `--lambda_bio_constraint`는 이번 레시피에 미포함.

## 3. §4.10 Compositional 분석 — 재측정

### 3.1 Inter-codebook NMI (낮을수록 중복이 적음)

| dataset | mean off-diag NMI | max | min | A-era 참조 |
|---|---:|---:|---:|---|
| Flickr25k | **0.5606** | 0.6379 | 0.4712 | ~0.604 |
| NUS-WIDE | 0.6079 | 0.6814 | 0.5028 | — |
| MS-COCO | 0.6719 | 0.7341 | 0.6143 | ~0.658 |

Flickr는 개선(0.604 → 0.561), MS-COCO는 소폭 악화(0.658 → 0.672).

### 3.2 Codebook-drop (informative budget)

| dataset | baseline full mAP | Σ drop | anti-codebook | 슬롯별 Δ |
|---|---:|---:|---:|---|
| MS-COCO | 0.6086 | **−0.0558** | 0 | −.014 / −.011 / −.009 / −.009 / −.011 / −.003 |
| NUS-WIDE | 0.5952 | −0.0476 | 1 | −.005 / −.012 / −.011 / −.011 / −.010 / **+.000** |
| Flickr25k | 0.7626 | −0.0418 | 0 | −.011 / −.006 / −.005 / −.006 / −.002 / −.012 |

**모든 슬롯이 기여한다** (anti-codebook 0–1개). 어떤 슬롯도 제거가 이득이 아니다.

### 3.3 Slot intervention — 🔴 **결론 불변, 개선 없음**

| dataset | ours target gain | random | 비율 | selectivity |
|---|---:|---:|---:|---:|
| Flickr25k | 0.0259 | 0.0168 | **1.54×** | 0.0010 |
| NUS-WIDE | 0.0160 | 0.0094 | **1.69×** | −0.0059 |

A-era 기록(intended gain ~1.6× random, selectivity ≈ 0)과 **사실상 동일**하다.
codon 붕괴를 고치고 해석성 디코딩을 올렸음에도 **슬롯을 독립적으로 제어할 수 있다는 주장은
여전히 지지되지 않는다.** draft §4.10 / §5.2의 해당 한계 서술은 그대로 유지해야 한다.

## 4. 진행 중 (미완)

| 항목 | 상태 |
|---|---|
| **A2 no-text-supervision** × 4 dataset | GPU 실행 중 |
| **A4 shared-codebook (K=768)** × 4 dataset | 대기 |
| baseline multi-seed {43,44} 60셀 | GPU 4·5, 18/60 |
| MS-COCO λ=0.05 3-seed (λ 재검토) | 실행 중 |

A2/A4가 끝나면 draft §4.8의 causal ablation 표가 처음으로 **동일 프로토콜(P0 + bio-projected)**
수치로 채워진다. 현재 draft의 A2/A4 행은 전부 pre-P0/pre-bio 진단값이다.

## 5. 종합 판단

- **논문 주 contribution 축(조합성·다양성·해석성)에서는 신규 모델이 4/4 우세**하며 3-seed에서
  안정적이다.
- **검색 축에서는 2승 2패**로, A-champion의 3승 1패보다 나쁘다. 특히 CIFAR가 승→패로 후퇴한다.
- 따라서 **단일 모델로 두 축을 모두 최적화하지는 못했다.** 논문 서술은 둘 중 하나다.
  1. A-champion을 main으로, 신규 모델을 "compositional variant"로 병기
  2. 신규 모델을 main으로 하고 CIFAR·MS-COCO 검색 열세를 명시
- **slot intervention selectivity ≈ 0은 신규 모델에서도 해결되지 않았다.** 이는
  "독립 제어 가능한 semantic factor"를 주장할 수 없다는 기존 한계를 재확인한다.
