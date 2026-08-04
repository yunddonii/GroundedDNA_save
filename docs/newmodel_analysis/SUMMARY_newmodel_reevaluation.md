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

## 1. 3-seed 대 3-seed — MS-COCO가 동률로 회복

seeds {42,43,44}. **baseline도 60셀(9 variant × 3 dataset × 2 seed)을 실패 0건으로 완료**하여
Flickr/MS-COCO/NUS-WIDE는 3-seed 대 3-seed 비교다.

| dataset | ours 3-seed | best `U0` 3-seed | **Δ** | A-champion Δ |
|---|---|---|---:|---:|
| **Flickr25k** | 0.8668 ± 0.0017 | OH 0.8327 | **+0.0341** | +0.0313 |
| **NUS-WIDE** | 0.8283 ± 0.0008 | OH 0.8028 | **+0.0255** | +0.0239 |
| **MS-COCO** | 0.8232 ± 0.0098 | CroVCA 0.8216 | **+0.0016** | −0.0087 |
| **CIFAR-10** | 0.8940 ± 0.0033 | CIBHash 0.8968 *(1 seed)* | −0.0028 | +0.0090 |

🟢 **MS-COCO 열세가 사라졌다 — 단, 우리가 올라서가 아니다.** CroVCA의 3-seed 평균이
0.8257(seed 42) → **0.8216**으로 내려갔다. 즉 그동안 문서화해 온 MS-COCO 열세의 상당 부분은
**baseline 쪽 seed 운**이었다.

⚠️ 그래도 +0.0016은 우리 std 0.0098의 6분의 1이므로 **통계적으로는 동률**이다. "대등하다"가
정확한 서술이고 "이겼다"는 쓰지 않는다.

**남은 열세는 CIFAR-10 하나**이며, 그 baseline은 아직 single seed다(60셀 배치에 CIFAR 미포함).
CIFAR `{43,44}` 실행이 유일한 빈칸이다.

**해석성은 4/4 개선이며 안정적**: A-champion 대비 +0.0322 / +0.0239 / +0.0227 / +0.0101,
decode std 0.0017–0.0104. 다양성도 4/4 개선.

⚠️ **방법론적 교훈**: MS-COCO의 seed 폭은 우리 0.017, CroVCA도 유사하게 컸다. 우리가 비교해 온
λ 간 폭(0.007)보다 크므로, **단일 seed로 0.003 수준의 우열을 판정해서는 안 된다.**

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

## 4. §4.8 Causal ablation — 4/4 완료

동일 프로토콜(P0 2-stage + bio-projected). full = 통일 레시피 3-seed 평균.

### A2. Text supervision 제거

| dataset | Δ mAP@R | **Δ codon decode** |
|---|---:|---:|
| MS-COCO | −.0605 | **−.0860** |
| CIFAR-10 | −.0197 | −.0124 |
| NUS-WIDE | −.0184 | **−.0427** |
| Flickr25k | −.0126 | −.0203 |

**텍스트 감독은 4/4에서 필수**이며, MS-COCO·NUS-WIDE에서는 **검색보다 해석성에 더 크게 기여**한다
(NUS는 2.3배). "text가 codon의 의미 조직을 만든다"는 인과 주장의 직접 근거다.

### A4. Shared codebook (K=768)

| dataset | Δ mAP@R | Δ codon decode |
|---|---:|---:|
| Flickr25k | −.0147 | −.0188 |
| CIFAR-10 | −.0146 | −.0053 |
| MS-COCO | −.0092 | **−.0312** |
| NUS-WIDE | −.0031 | −.0128 |

**슬롯별 독립 codebook이 4/4에서 필요**하다. ⚠️ A-champion 시절 NUS-WIDE에서는 shared가 검색을
앞섰으나(.8262 → .8301) 신규 레시피에서는 그 역전이 사라졌다(−.0031).

### MS-COCO λ 재검토 (3-seed)

| λ | mAP@R | decode |
|---:|---|---|
| **.03** | **.8232 ± .0098** | **.6466 ± .0048** |
| .05 | .8194 ± .0056 | .6366 ± .0027 |

평균은 `.03`이 높으나 분산이 겹쳐 **통계적으로 구분되지 않는다.** `.03`을 유지한다.

## 4b. 진행 중 / 미완

| 항목 | 상태 |
|---|---|
| baseline multi-seed {43,44} 60셀 | **60/60 dispatch 완료**, 집계 미수행 |
| K × L grid (16셀 중 4셀) | 미착수 |
| `--lambda_bio_constraint` 통합 | 미착수 (사전 유효율 40–57 %가 chance 수준) |

## 5. 종합 판단

- **논문 주 contribution 축(조합성·다양성·해석성)에서는 신규 모델이 4/4 우세**하며 3-seed에서
  안정적이다.
- **검색 축에서는 2승 2패**로, A-champion의 3승 1패보다 나쁘다. 특히 CIFAR가 승→패로 후퇴한다.
- 따라서 **단일 모델로 두 축을 모두 최적화하지는 못했다.** 논문 서술은 둘 중 하나다.
  1. A-champion을 main으로, 신규 모델을 "compositional variant"로 병기
  2. 신규 모델을 main으로 하고 CIFAR·MS-COCO 검색 열세를 명시
- **slot intervention selectivity ≈ 0은 신규 모델에서도 해결되지 않았다.** 이는
  "독립 제어 가능한 semantic factor"를 주장할 수 없다는 기존 한계를 재확인한다.
