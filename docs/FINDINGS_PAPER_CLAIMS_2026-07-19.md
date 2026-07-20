# GroundedDNA — 논문 주장의 증거 상태 (2026-07-19)

> **이 문서의 목적.** 다른 세션이 이 파일 하나만 읽고 "논문에서 무엇을 주장할 수 있고 무엇을 주장할 수 없는가"를 재구성할 수 있게 한다.
> **범위.** 2026-07-19에 완료한 세 실험(대칭 P0 retrieval, held-out codon decoding, slot intervention)과, 그 결과가 기존 초안·로그의 어떤 서술을 무효화하는지.
> **상위 문서와의 관계.** `PROJECT_LOG.md`가 여전히 전체 이력의 source of truth다. 이 문서는 그 중 **논문 claim에 직접 영향을 주는 부분만** 추린 판정문이다.
> **관련 문서.** 실험 설계 원본 `REQUIRED_EXPERIMENTS_GROUNDEDDNA_PAPER.md`, 초안 `DRAFT_GROUNDEDDNA_PAPER_KO.md`(**다수 항목이 stale — §6 참조**).

---

## 0. 한 문단 요약

GroundedDNA의 논문 중심 주장은 **held-out codon decoding** 하나로 좁혀야 한다. 학습 데이터로만 만든 `(slot, codon) → concept` 사전이 unseen test 이미지의 concept를 코드만 보고 예측하며, 동일 비트 예산의 flat hash 임의 partition보다 3개 데이터셋 모두에서 유의하게 잘 예측한다(+0.057 ~ +0.099, 모든 95% CI가 0 배제). 그것도 slot당 활성 symbol이 baseline의 1/3 수준인 상태에서다. 반면 **slot intervention은 selectivity를 입증하지 못했고**(CIBHash의 임의 6-bit chunk가 사실상 동일한 target gain을 냄), **retrieval 우위는 4개 중 2개 데이터셋에서 noise 수준**이다. 따라서 논문은 `compositional / readable`은 주장하되 `disentangled / independently controllable`은 주장하면 안 된다.

---

## 1. 실험 1 — 대칭 P0 프로토콜 retrieval

### 프로토콜

우리 모델과 baseline이 **완전히 같은 절차**를 밟는다.

| 단계 | 내용 |
|---|---|
| val 분리 | train에서 10% carve (`val_split.carve_val_indices`, seed 42, 단일 label은 class-stratified) |
| epoch 선택 E\* | **val_query vs opt-train DB** 검색의 val mAP@R 최대 (test 미사용) |
| stage 2 | train 100%로 재학습하되 E\*에서 정지 (LR 스케줄은 60-epoch 코사인 유지) |
| test | **단 1회** 평가 |
| whitening | opt-train / train-only 통계만 사용 (test caption 누수 제거 완료) |

baseline은 재학습된 stage-1 체크포인트(opt-train 90%)로 E\*를 고르고, 그 E\*에서의 **100%-train 런** 값을 보고 — 우리 stage 2와 대칭.

### 결과 (mAP@R, 36-bit, frozen CLIP-ViT-B/16, whole-image)

| Dataset (cutoff) | **Ours** | CIBHash | CIMON | MLS3RDUH | Δ vs best |
|---|---:|---:|---:|---:|---:|
| Flickr25k (@5000) | **0.8810** | 0.8233 | 0.8288 | 0.7811 | **+0.052** |
| NUS-WIDE (@5000) | **0.8334** | 0.8152 | 0.7860 | 0.7746 | **+0.018** |
| CIFAR-10 (@1000) | **0.9046** | 0.9004 | 0.8367 | 0.5793 | +0.004 |
| MS-COCO (@5000) | **0.8134** | 0.8112 | 0.6716 | 0.6423 | +0.002 |

baseline E\*: CIBHash 4/19/4/4, CIMON 49/59/54/59, MLS3RDUH 59 (전부).

### 판정

- **Flickr25k와 NUS-WIDE는 견고한 우위.**
- **MS-COCO(+0.002)와 CIFAR-10(+0.004)은 noise 수준.** "4개 데이터셋 SOTA"라고 쓰면 안 된다. 정직한 표현은 **"2개에서 명확한 우위, 2개에서 동등"**.
- retrieval은 "해석 가능성을 위해 검색 성능을 희생하지 않았다"는 필요조건으로만 쓰고, main contribution으로 세우지 않는다.

### 반드시 함께 적어야 할 caveat

1. **Flickr의 선택 지표가 퇴화한다.** val DB(4,500) < R(5,000)이므로 Flickr에서는 val mAP@R = val mAP. 선택 통계와 보고 통계가 동일하지 않다. 우리와 baseline이 같은 val DB를 쓰므로 대칭성은 유지되지만 명시해야 한다. (CIFAR10 R=1000, MS-COCO·NUS-WIDE는 정상 truncate.)
2. **MLS3RDUH는 4개 데이터셋 전부 E\*=59** (val 곡선이 단조 증가) — 60 epoch에서 미수렴. 그 수치는 하한이다.
3. selection regret은 모든 셀에서 ≤ 0.005, 6개 셀에서 정확히 0.

### 산출물

- `docs/baseline_p0_stage2.json`, `docs/baseline_p0_stage2.md`, `docs/baseline_p0_stage2_partial/*.json`
- `scripts/baseline_val_select_p0.py`
- 우리 쪽 수치: `result/260717+*_P0refit_e*/evaluation_siglip2_base.json`
- commit `e4cae82`

---

## 2. 실험 2 — Held-out codon decoding ✅ **성공 (논문의 중심 근거)**

### 프로토콜 (`REQUIRED_EXPERIMENTS` §2)

| 항목 | 값 |
|---|---|
| 사전 생성 split | train만 (Flickr/NUS는 DB 추출에서 basename으로 slice, MS-COCO는 `extract_train.npz` 신규 생성) |
| 평가 split | official test (`extract_query.npz`) — db·train과 disjoint, 로드 시 검증 |
| 정답 | dataset multi-hot label — **Qwen caption과 독립** → 순환 평가 아님 |
| decoder 입력 | slot의 정수 code id **뿐**. test caption·test image feature 미사용 |
| 사전 등록 hyperparameter | `alpha=1.0`, `min_support=10`, seed 42 — test 보고 조정하지 않음 |

`p(c | m, u) = (count + α) / (support + 2α)` per label, 지표는 per-sample label-ranking AP(= concept mAP).

### 결과 (concept mAP)

| unit (slot당 용량) | Flickr25k | MS-COCO | NUS-WIDE |
|---|---:|---:|---:|
| ours **codeword** (K=128) | 0.8143 | 0.7147 | 0.7806 |
| ours **codon** (64) | **0.7794** | **0.6323** | **0.7339** |
| CIBHash chunk (64) | 0.6670 | 0.5338 | 0.6516 |
| CIMON chunk (64) | 0.7200 | 0.5074 | 0.6765 |
| MLS3RDUH chunk (64) | 0.6749 | 0.4837 | 0.6403 |
| majority control | 0.4730 | 0.3160 | 0.4822 |
| shuffled control | 0.4810 | 0.3129 | 0.4795 |

**paired bootstrap (ours codon − best chunk control), 95% CI:**

| Dataset | Δ | CI | 0 배제 |
|---|---:|---|:---:|
| Flickr25k (vs CIMON) | **+0.0594** | [+0.0537, +0.0648] | ✅ |
| MS-COCO (vs CIBHash) | **+0.0985** | [+0.0944, +0.1028] | ✅ |
| NUS-WIDE (vs CIMON) | **+0.0575** | [+0.0511, +0.0638] | ✅ |

majority·shuffled 대비 CI도 전부 0 배제 (예: Flickr vs majority +0.3064 [+0.2955, +0.3163]).

### 가장 강한 논거 — 용량이 아니라 조직화

slot당 **활성** symbol 수:

| Dataset | ours codon (of 64) | best chunk (of 64) |
|---|---|---|
| Flickr25k | 21, 41, 39, 34, 39, 35 | 53, 57, 53, 58, 53, 56 |
| MS-COCO | 21, 57, 62, 59, 56, 56 | 64 × 6 (포화) |
| NUS-WIDE | 23, 45, 38, 43, 51, 43 | 57, 57, 61, 58, 62, 59 |

**우리 코드는 baseline의 1/3~2/3 symbol만 쓰면서 더 잘 디코딩한다.** 우위가 code capacity에서 오지 않는다는 직접 증거.

### Robustness

`min_support ∈ {1,5,10,25,50} × alpha ∈ {0.5,1.0,5.0}` 전 구간에서 ours 0.768~0.781, CIMON chunk 0.693~0.722 — **gap이 +0.059~+0.074로 일정.** knife-edge 아님. alpha는 ranking 지표라 영향 없음(monotone), min_support는 coverage만 조절.

### §2.8 collision 회계 (codeword → codon 손실)

| Dataset | codeword | codon | decoding loss | 병합 codeword pair 평균 JSD |
|---|---:|---:|---:|---:|
| Flickr25k | 0.8143 | 0.7794 | **−0.0349** | 0.026~0.029 (max 0.104) |
| NUS-WIDE | 0.7806 | 0.7339 | **−0.0467** | — |
| MS-COCO | 0.7147 | 0.6323 | **−0.0823** | — |

K=128 → 64 codon 병합에서 의미가 일부 손실되지만, codon 점수는 여전히 모든 control을 크게 상회. **손실을 투명하게 보고하되 codon-level 주장은 유지 가능.**

### §2.11 성공 조건 판정

| # | 조건 | 판정 |
|---|---|:---:|
| 1 | majority·shuffled보다 유의하게 높음 | ✅ |
| 2 | flat hash chunk보다 높음 | ✅ 3개 데이터셋, **6개 slot 전부** |
| 3 | decoding loss 투명 보고 + codon 점수 유의미 유지 | ✅ |
| 4 | 한 slot에 집중되지 않음 | ✅ |

### ⚠️ 이 실험이 증명하지 **않는** 것 (반드시 논문에 명시)

decoding 타깃이 **image-level label 하나**이고 6개 slot이 이를 공유한다. 따라서:

- 증명됨: "각 slot의 codon이 의미를 담고 있고, 임의 bit partition보다 잘 디코딩된다"
- **증명 안 됨: "global slot이 정말 global을, color slot이 정말 color를 담당한다"** (역할 배정의 타당성)

Flickr per-slot codon 점수는 0.768 / 0.791 / 0.787 / 0.779 / 0.775 / 0.776 — **spread 0.023**. 여섯 slot이 거의 균일하다. 이는 slot 특화보다 **중복(redundancy)**을 시사하며, `ANALYSIS_compositional_contribution.md`의 "local slot pairwise NMI 0.74–0.82"와 일치한다.

역할 검증에는 per-slot 독립 타깃(CUB attribute)이 필요하다. → **2026-07-20 실행 완료, §4c.** 주효과 제거 후 유의한 역할 계승 검출(0.264 vs 경계파괴 0.161).

### 산출물

- `scripts/heldout_codon_decoding.py`, `scripts/extract_train_split.py`, `scripts/baseline_extract_splits.py`
- `docs/heldout_decoding_{flickr25k,mscoco,nuswide}.json`
- baseline 코드: `result_baseline/260719/{cibhash,cimon,mls3rduh}_{mscoco,nuswide}_clip_decodectl/`

### 🔧 미해결 정합성 이슈 (camera-ready 전 처리)

**Flickr의 chunk control만 epoch 불일치.** `result_baseline/260527/*` (epoch 59)를 쓰는데 P0 E\*는 CIBHash 4 / CIMON 49 / MLS3RDUH 59다. MS-COCO·NUS-WIDE는 E\*로 맞춰져 있다. `scripts/baseline_extract_splits.py`로 재추출하면 ~2분. 결론을 뒤집을 크기는 아니나 표의 일관성을 위해 처리할 것.

---

## 3. 실험 3 — Slot intervention ⚠️ **부분 성공 / 핵심 control 미통과**

### 프로토콜 (`REQUIRED_EXPERIMENTS` §3)

query code에서 **codon 하나만** donor의 것으로 교체 후 재검색. **forward pass 재계산 없음** — 방출된 코드만 조작.

- **target concept 정의**: image-level label에는 per-slot 정답이 없으므로, donor의 slot-m codon이 **train-only 사전**에서 가장 강하게 예측하는 label 중 query가 갖지 않은 것.
- **donor 선택**: target concept를 가지면서 나머지 label이 query와 최대한 유사한 DB 샘플.
- 설정: query 400, DB subsample 12,000, K=100, bootstrap 1,000.

### 🔴 2026-07-19 재분석 — 판정 통계를 새로 계산했고, 결론은 **더 부정적**이 되었다

초판은 `selectivity`를 **점추정으로만** 보고했고(`float(tgt_gain.mean() - off_drift.mean())`), CI는 `target_gain`에만 있었다. 사전등록 조건 2는 selectivity에 대한 조건이므로 **판정 근거가 검정되지 않은 상태**였다. per-query drift를 보존하고 arm 간 **paired bootstrap**(모든 arm이 동일 query 사용)을 추가해 재계산했다.

**ours selectivity > control 이 유의한 셀 수 (95% CI가 0 배제 & ours 우세):**

| dataset | vs cibhash chunk | vs cimon chunk | vs random_slot | vs random_donor |
|---|---:|---:|---:|---:|
| Flickr25k | **3/6** | 4/6 | **3/6** | 6/6 |
| MS-COCO | — | — | **2/6** | 6/6 |
| NUS-WIDE | — | — | **3/6** | 6/6 |

핵심 비교(CIBHash chunk, random_slot)에서 **절반 이하만 유의**하다. Flickr `color_texture`는 점추정부터 음수(−0.0005). "ours만 selectivity 양수"라는 겉보기 분리는 **slot 평균에서만** 성립하고 slot별로는 절반이 flat chunk와 구분되지 않는다.

### 🔴 예산 동일성 주장은 사실이 아니었다

초판의 "모든 arm이 정확히 6 bit를 바꾸므로 예산 동일"은 **틀렸다.** codon/chunk 교체는 donor와 이미 다른 심볼만 바꾸므로 arm마다 섭동 크기가 다르다. 실측(각 arm 자신의 Hamming 공간 기준 slot 내 변경 비율):

| arm | 변경 비율 | TargetGain | **gain / 변경비율** |
|---|---:|---:|---:|
| ours | 2.02/3 base = **0.673** | +0.0342 | 0.0508 |
| cibhash chunk | 2.69/6 bit = **0.448** | +0.0365 | **0.0816** |
| cimon chunk | 2.39/6 bit = 0.399 | +0.0141 | 0.0353 |

**CIBHash는 slot을 45%만 바꾸고 우리(67%)보다 큰 gain을 낸다 — 단위 섭동당 60% 효율적.** 정규화는 우리에게 유리하지 않다.

### 왜 이런 결과인가 — 구조적 원인 (2026-07-19 측정)

**slot 간 중복도** (Flickr DB 23,000, 6 unit × 64 symbol):

| code | H(unit) | unique = H(m\|나머지) | 중복률 | 6 unit 독립정보 합 |
|---|---:|---:|---:|---:|
| **ours codon** | 4.59b | **0.635b** | **86.2%** | **3.81b** |
| cibhash chunk | 5.85b | 0.093b | 98.4% | 0.56b |
| cimon chunk | 5.28b | 0.414b | 92.2% | 2.48b |
| mls3rduh chunk | 4.16b | 0.620b | 85.1% | 3.72b |

**CIBHash는 6 unit 전체 독립정보가 0.56 bit인데도 우리(3.81b, 6.8배)와 같은 target gain을 낸다.** → TargetGain은 slot 고유 정보가 아니라 **코드 공간 이동 거리**를 재는 지표다. 구조 유무와 무관하게 같은 값이 나온다.

중복의 출처:
1. **공유 global gate.** `model_siglip2.py:4360` `q_conditioned_local = q_local + sigmoid(gate)*q_global`, 학습된 gate = **[0.993]×5** (init 4.595 → 4.89~5.05로 **상승**). `H(cb_m|cb_0)`이 각 local slot 엔트로피의 **48%** 제거. (norm 기준 주입량은 20%라 단독 설명은 못 함.)
2. **손실 예산 ~80:1.** `cibhash_ntxent`(slot 무관) 실효 기여 ≈2.92 vs `text_code_kl`(유일한 slot 차별화) ≈0.037. `cb_uncorr` ≈1e-4로 사실상 비활성.
3. **시각 입력이 slot 간 동일.** bidirectional prune이 visual mask를 slot 간 **UNION**으로 취함(+ legacy 상수-중요도 버그, union keep 99.86%) → 6 slot이 같은 패치를 봄.
4. **텍스트 타깃은 범인이 아님.** 6개 감독 신호는 충분히 구별됨(slot 간 평균 코사인 **0.21**; whiten gamma 1.0이면 0.059). 구별되는 신호가 중복 코드로 붕괴하는 것이므로 원인은 하류에 있다.

각 slot의 고유 정보가 0.635 bit뿐이면 **편집할 독립 factor 자체가 없다.** 실험 설계를 개선해도 잡아낼 selectivity가 존재하지 않는다.

### §3.6 성공 조건 판정 (재분석 반영)

| # | 조건 | 판정 |
|---|---|:---:|
| 1 | 의도한 slot 교체 시 TargetGain > 0 | ✅ 18/18 셀 CI가 0 배제 |
| 2 | Selectivity가 random-slot **및 CIBHash chunk**보다 높음 | ❌ **미통과** (핵심 control 대비 8/18, 3/6) |
| 3 | 한 dataset·slot에 국한되지 않음 | ✅ |
| 4 | off-target drift 함께 보고 | ✅ |

### 사전 등록된 판정 적용

`REQUIRED_EXPERIMENTS` §3.6: "intervention selectivity가 없다면 **'독립적으로 조작 가능한 compositional factor'라는 표현은 피해야 한다**." → **그대로 적용.** 결과를 본 뒤 기준을 완화하지 않는다.

### 그래도 남는 양성 결과

중복도 비교는 intervention과 **독립적으로 성립**한다: 우리 코드는 flat baseline보다 **덜 중복**이다(86.2% vs CIBHash 98.4%, 독립정보 **6.8배**). 이는 §2 held-out decoding의 "용량이 아니라 조직화" 논거의 정보이론적 판본이며, `disentangled`를 주장하지 않으면서 쓸 수 있다.

### 다음 단계 (인과 검증)

`--disable_global_gate`(v23b, 이미 구현됨) 런 1개로 (1) 중복률, (2) decoding per-slot spread(현재 0.023), (3) selectivity를 재측정. 개선되면 **"공유 global conditioning이 독립 조작 가능성을 막는다"는 인과적 주장**이 성립한다. 미개선이면 원인은 손실 예산(2)·공유 입력(3) 쪽이다.

### 산출물

- `scripts/slot_intervention_eval.py`
- `docs/slot_intervention_{flickr25k,mscoco,nuswide}.json`

---

## 4. 🔴 NMI — 논문에서 제거할 것 (부호가 반대)

`scripts/pairwise_nmi.py`는 **codebook 쌍 사이의 상호정보량**을 잰다(레이블·텍스트 무관). **낮을수록 codebook이 독립적.**

근거:
- `PROJECT_LOG.md:831` — "**CRITICAL CORRECTION**: NMI between codebook pairs measures MUTUAL INFORMATION — **lower NMI = more orthogonal**."
- `PROJECT_LOG.md:906` — 2026-06-30 사용자가 직접 지적, 이후 로그 컬럼 헤더는 전부 `NMI off-diag (low better)`.

**초안이 부호를 뒤집어 쓰고 있다.** `DRAFT_GROUNDEDDNA_PAPER_KO.md:255`(A4), `:228`(4-base), `:285`(multi-crop) 전부 "NMI 하락 = 구조 붕괴"로 읽는다. 저장소 규약대로면 정반대 진술이다.

추가로:
- 이 프로젝트는 이미 `PROJECT_LOG.md:1295`에서 "predictive orthogonality는 non-goal"이라 결론내고 서사를 atlas/B-metric으로 옮기기로 했다.
- baseline과의 NMI 비교(0.55 vs 0.19)는 `REPORT_2026-06-25_final.md:41`에서 "flat hash를 가상 6-codebook으로 쪼갠 것이라 apples-to-apples가 아니다"라고 이미 무효 처리됐다.
- A2(no-text)에서 NMI가 **오르므로**, 텍스트 감독의 기여를 NMI로 주장할 수도 없다.

**조치: NMI를 해석 가능성 지표에서 완전히 제거하고, held-out decoding으로 대체한다.**

⚠️ **파급효과**: NMI를 빼면 **A4(codebook 분리) ablation에 mAP −0.006/−0.025만 남아 근거가 약해진다.** A4를 decoding 지표로 다시 측정하면 근거가 복구될 가능성이 높다 — **권장 후속 실험 1순위.**

---

## 4b. 실험 4 — Slot 역할 특화 (A) + 슬롯 내 등급적 일관성 (B)

> **동기.** 직교성은 contribution의 전제가 아니다(이 요구는 §3 intervention 프로토콜에서 딸려온 것). 실제로 필요한 주장은 ① 각 슬롯이 **자기 몫**의 의미를 설명하고, ② 슬롯 **안에서** 비슷한 의미 → 비슷한 codeword, ③ 그것이 codon까지 이어진다는 것이다. §2 decoding은 타깃이 슬롯 공유 label 하나여서 ①②③ 어느 것도 직접 검증하지 못했다.

### (A) Cross-slot decoding matrix — ① 검증 → 🔴 **반박됨**

`D[m,m']` = 슬롯 m의 코드로 슬롯 m'의 캡션 어휘를 디코딩. 역할 특화 = **열 방향 대각 우세**. 비대각이 높은 것은 중복이지 실패가 아니다.

🔴 **순환성 명시**: 슬롯별 타깃이 학습에 쓰인 Qwen 캡션이라 §2.2 위반. **상대 진단(대각 vs 비대각)으로만 유효**하며 절대값은 grounding 근거가 아니다.

**Flickr25k, codon, 슬롯 고유 어휘(≥2배):**

| target slot | diag | off-diag mean | advantage | column argmax |
|---|---:|---:|---:|:---:|
| global | 0.3599 | 0.3712 | **−0.0112** | OTHER |
| primary_object | 0.3106 | 0.2845 | +0.0261 | OWN |
| secondary_object | 0.2236 | 0.2245 | −0.0009 | OTHER |
| activity_relation | 0.3074 | 0.3067 | +0.0007 | OTHER |
| color_texture | 0.3071 | 0.3016 | +0.0055 | OWN |
| scene_type | 0.4498 | 0.4318 | +0.0180 | OWN |

**6개 중 3개만 자기 코드가 argmax.** 최대 대각 우세(+0.026)가 §2의 flat-hash 대비 격차(+0.059~+0.099)보다 한 자릿수 작다. `global`·`secondary_object`·`activity_relation`은 **자기 코드보다 남의 코드로 더 잘 디코딩된다.**

🔬 **어휘 교란 배제됨.** 슬롯 캡션은 일반 어휘를 공유하므로('white'가 4개 슬롯에 등장) 공유 어휘가 null을 *만들어낼* 수 있다. 슬롯 고유 어휘만으로 재실행해도 결과 동일(3/6 argmax OWN, 우세 −0.011…+0.026 vs 평범 어휘 −0.009…+0.025). **진짜 null이다.**

⚠️ **Flickr 전용, 확장 불가.** MS-COCO·NUS-WIDE는 Qwen 캡션이 **train에만** 존재(NUS-WIDE는 test 2100장 중 0장 커버). Flickr만 `cache/flickr25k_qwen_v4.jsonl`이 25,000행으로 train+test를 덮는다 — 2026-07-17 whitening leak의 원인과 같은 커버리지다.

### (B) 슬롯 내 등급적 일관성 — ②③ 검증 → 🟢 **확인, 단 flat 대비 마진은 데이터셋 의존**

test 이미지 쌍 ~20만 개에서 **슬롯 내 코드 거리 ↔ 의미 거리** Spearman ρ. 의미 거리 = `1 − label Jaccard` (**Qwen과 독립 → 비순환**).

| Dataset | ours codeword | **ours codon** | CIBHash chunk | CIMON chunk | shuffled | Δ(codon − best flat) |
|---|---:|---:|---:|---:|---:|---:|
| Flickr25k | 0.3688 | **0.3497** | 0.1382 | 0.2708 | 0.0002 | **+0.079** |
| NUS-WIDE | 0.2391 | **0.2697** | 0.1537 | 0.2590 | 0.0003 | +0.011 |
| MS-COCO | 0.0848 | **0.1654** | 0.0928 | 0.1531 | 0.0007 | +0.012 |

1. **②③은 3/3 데이터셋에서 chance 대비 확인** (shuffled ≈ 0.000).
2. **flat 대비 마진은 Flickr에서만 결정적** (+0.079). NUS-WIDE·MS-COCO는 CIMON과 사실상 동률(+0.011/+0.012).
3. 🔬 **codon ρ > codeword ρ (MS-COCO +0.081, NUS-WIDE +0.031)** — 3-base 양자화가 상관을 *높이는* 건 codeword 거리가 충실한 의미 proxy라면 불가능하다. **발견이 아니라 방법론적 caveat**: codebook 임베딩 코사인은 VQ 목적함수가 빚은 기하라 label 유사도의 대리가 못 된다. MS-COCO에서 ours-codeword(0.085)는 flat baseline 둘 다보다 낮다. **codon 행만 인용할 것.**

### 종합

| 주장 | 상태 |
|---|---|
| ① 각 슬롯이 **자기** 의미를 설명 | 🔴 **반박** (Flickr, 3/6) — 다른 데이터셋은 검증 불가 |
| ② 슬롯 내 비슷한 의미 → 비슷한 codeword | 🟢 3/3 chance 대비 확인, flat 대비는 Flickr만 결정적 |
| ③ ②가 codon까지 유지 | 🟢 확인 (codon ρ ≥ codeword ρ, 2/3) |

🧭 **결론.** 직교성 요구를 버려도 **역할 배정 주장은 살아나지 않는다.** `slot-specialized`는 주장 불가. 남는 것은 "코드가 슬롯 내에서 의미적으로 조직되어 있고 그 조직이 codon까지 유지된다" + §2의 held-out decoding 우위다. 여섯 슬롯은 **역할이 배정된 여섯 부분이 아니라 같은 의미 내용의 부분적으로 중복된 여섯 view**로 서술해야 한다.

**산출물**: `scripts/slot_role_analysis.py`, `docs/slot_role_flickr25k_distinctive.json`(보고본), `docs/slot_role_flickr25k.json`(평범 어휘), `docs/slot_role_{mscoco,nuswide}.json`(B만).

---

## 4c. 실험 5 — CUB per-attribute 역할 타당성 ✅ **유의 (경계 특이적), 단 약함**

### 왜 CUB인가

image-level label로는 역할 배정을 검증할 수 없다. 6개 slot이 **하나의 타깃을 공유**하고, 2026-07-20 열 효과 대조에서 그 결과가 **caption의 성질**임이 확정됐다(semantic slot 개념이 없는 flat chunk와 열 프로파일 Spearman **+1.000**). CUB는 312개 부위별 이진 속성을 주고, v6b 캡션이 해부학적 부위별로 작성됐다(`tools/qwen3_v6b_cub_trainset.py`): C_global→전체, C_primary_object→**head/bill**, C_secondary_object→**wing/upperparts**, C_activity_or_relation→**underparts**, C_color_texture→**tail/appendages**, C_scene_type→**pattern/markings**. (CUB에서 slot 이름은 잔재이며 내용은 부위다.)

### 설계 (`scripts/cub_per_slot_role.py`)

속성 이름·28개 표준 그룹 **둘 다 불필요**하다(이 사본에 `attributes.txt` 없음; 경험적 그룹 복원은 색상 그룹이 다중선택이라 61개로 파편화되어 폐기). 속성 a마다:

| | 정의 | 평가 범위 |
|---|---|---|
| 교사 측 T(a) | 어느 **caption slot**이 a를 가장 잘 예측 (centroid AUC) | 캡션 보유 행 내부 split-half |
| 코드 측 C(a) | 어느 **code slot**이 a를 가장 잘 디코딩 (train 사전 → test AUC) | **held-out test** |

역할 타당성 = agreement(T, C). fit=train(=CUB database, 5,994), 평가=official test(5,794, disjoint), 사용 속성 269/312(min_pos=50).

### ⚠️ 1차 실행은 버그였다 (기록)

교사 측 AUC가 6개 slot 전부 **0.4895로 동일**하게 나왔다 → CUB 캡션은 **train에만 존재**(has_text 5,994/11,788)하는데 test에서 채점해 영벡터를 비교한 것. 의미 있는 수치처럼 보였을 값이다. 교사 측을 캡션 보유 행 내부 split-half로 변경하여 수정.

### 결과 — raw argmax는 주효과에 완전히 가려진다

| | slot별 배정 (269개 속성) |
|---|---|
| 교사(caption) | global 20, head/bill 16, **wing 124**, underparts 55, tail 14, markings 40 |
| 코드 | **global 244**, head/bill 6, wing 4, underparts 3, tail 2, markings 10 |

raw 일치율 **0.074** vs 우연 0.084 — 우연 이하. code slot 0(global)이 269개 중 244개에서 최고 AUC(행 평균 **0.703** vs 나머지 0.629~0.659)라 argmax가 그 행 효과만 잰다. **교사 측은 6개 slot에 고루 분화되므로 측정 도구 자체는 작동한다.**

### 주효과 제거(이중중심화) 후 — 유의한 역할 계승

| 조건 | 일치율 |
|---|---:|
| **진짜 슬롯 경계** (교사 seed 42 / 7 / 123) | **0.264 / 0.253 / 0.249** |
| permutation null (우연) | 0.168 [0.126, **0.216**] |
| **슬롯 경계 파괴 대조** (같은 18 base 무작위 재분할 ×5) | 0.164, 0.141, 0.201, 0.164, 0.138 → **평균 0.161** |
| CIMON chunk (flat, 2026-07-20 추가) | 0.141 |
| CIBHash chunk (flat) | 0.112 |
| MLS3RDUH chunk (flat) | 0.063 |

**flat baseline 3개 모두 우연 이하이고 ours만 상한(0.216)을 넘는다.** 최고 flat 대비 **1.87×**.

🟢 **3중 대조 통과.** (1) 우연 대비 유의(0.264 > 상한 0.216), (2) 교사 split seed 3개에서 안정, (3) **경계 특이적** — 같은 비트를 유지한 채 슬롯 경계만 무작위로 재분할하면 우연 수준(0.161)으로 붕괴. **역할 정보는 코드의 정보량이 아니라 경계 위치에 있다.** 효과 크기 **1.64×**.

### 이것이 답하는 질문

2026-07-20 gate ablation 이후 남은 물음 — "역할이 없는 것인가, image-level caption으로 볼 수 없는 것인가" — 에 대해 **후자**임을 보인다. per-part 타깃을 주면 역할 계승이 검출된다. §4b-A의 Flickr 반박과 모순되지 않는다: Flickr는 image-level label, CUB는 per-part attribute다.

### 반드시 함께 적을 한계

1. **CUB 한정.** Flickr/MS-COCO/NUS-WIDE는 per-part 타깃이 없어 미검증.
2. **절대값 26.4%** — 속성 다수는 여전히 교사 배정을 따르지 않는다.
3. **주효과 제거 후에만 가시.** raw로는 global slot이 244/269 독식 → "6개 역할이 분리되어 있다", "slot m = 역할 m" 서술은 **여전히 불가**.
4. 교사 측은 train 내부, 코드 측은 held-out test — 평가 범위가 다르다.
5. ~~flat baseline 대조 미실행~~ → **2026-07-20 완료.** `scripts/baseline_extract_splits.py`로 CUB baseline 코드 추출(epoch 059), 3개 전부 우연 이하(0.063~0.141). §4c의 모든 대조가 채워졌다.

### 산출물

- `scripts/cub_per_slot_role.py`, `docs/cub_per_slot_role.json`, `logs/cub_per_slot_role.log`
- commit `85b1e48`

---

## 5. 최종 판정 — 무엇을 주장할 수 있는가

### 🆕 2026-07-20 추가 — 열 효과 대조 + gate ablation

**(1) 열 효과는 caption의 성질이다 (H_data 확정).** 역할 타당성 행렬의 열 효과(scene·global이 높고 secondary가 낮음)를, semantic slot 개념이 **전혀 없는** flat baseline의 임의 6-bit chunk로 재현했다. 열 프로파일 순위가 **완전히 동일**하다(ours vs CIBHash Spearman **+1.000**, vs CIMON **+1.000**, vs MLS3RDUH +0.886). scene_type은 eff_rank 58.5로 압도적 저차원(나머지 111~219)이며 모든 분할에서 1위.
→ **이중중심화가 검증됐다**(주효과는 반드시 제거). 상호작용 +0.0052는 이미 제거한 값이므로 **역할 타당성 미성립 판정은 불변**.
→ 단 2026-07-19의 "모든 local slot이 global caption과 정렬 = 모델이 global을 퍼뜨림"이라는 해석은 **열 방향에 한해 철회**한다.

**(2) gate는 원인이 아니다.** `--disable_global_gate`로 검정: 상호작용 +0.0052→+0.0064, 열 rank-1 1/6→**1/6 불변**, global/local 행 비 1.50→1.43, retrieval **−0.0126**. global-행 지배는 gate가 만드는 것이 아니다. 남은 후보는 손실 예산(~80:1), UNION visual mask, 또는 측정 도구의 한계.

**(3) 새 양성 결과 — 조직화 우위 1.53×.** 열 순위는 데이터가 정하지만 **크기는 모델이 정한다**: ours가 best flat 대비 6개 caption 차원 **전부**에서 1.41~1.64× (평균 **1.53×**) 높은 lift. 평가 이미지 18,000장의 caption은 감독에 쓰인 적 없다. §2 held-out decoding(label 기준)의 **텍스트 측 대응물**이며, orthogonality·역할 배정을 주장하지 않고 성립한다.

**(4) slot 내 일관성·codon 전이는 성립.** 전 셀 lift 0.04~0.15(chance 상회), codeword→codon 0.095→0.070(**74% 보존**).

---

### ✅ 주장 가능 (측정으로 뒷받침)

**Main contribution (단일):**
> 학습된 slot 구조를 가진 DNA code는 **held-out 이미지에서 코드만으로 의미를 디코딩할 수 있으며, 동일 비트 예산의 임의 bit partition보다 유의하게 잘 디코딩된다** (+0.059 / +0.099 / +0.057, 모든 95% CI가 0 배제). slot당 활성 symbol이 baseline의 1/3~2/3인 상태에서 달성되므로, 우위는 code capacity가 아니라 의미 조직화에서 온다.

**부차 주장:**
- Text supervision이 retrieval과 code diversity를 좌우 (A2, 4개 데이터셋 −0.012 ~ −0.052; DNA-uniq Flickr 0.380→0.262, MS-COCO 0.207→0.128)
- 검색 성능이 해석 가능성을 위해 희생되지 않음 (Flickr +0.052, NUS-WIDE +0.018; MS-COCO·CIFAR-10 동등)
- slot 위치는 무의미하지 않음 — 단 **약한 근거**. 의도한 slot이 random slot보다 target gain은 크지만, selectivity 차이가 유의한 셀은 **18개 중 8개**뿐 (§3 재분석). "slot 선택이 무작위보다 낫다" 정도로만 쓰고 조작 가능성으로 확장하지 말 것.
- **코드가 flat baseline보다 덜 중복** — slot 중복률 86.2% vs CIBHash 98.4%, 6 unit 독립정보 **3.81b vs 0.56b (6.8배)** (§3). `disentangled`를 주장하지 않고 쓸 수 있는 구조 지표.
- **조직화 우위 1.53×** — 6개 caption 차원 전부에서 동일 예산 flat 분할보다 1.41~1.64× 높은 lift (2026-07-20). held-out 이미지, caption은 감독 미사용.
- **slot 경계가 교사의 역할 분담을 부분 계승 (CUB, §4c)** — 주효과 제거 후 교사-코드 배정 일치율 **0.264** vs 우연 0.168[상한 0.216], **경계 파괴 대조 0.161** (같은 18 base 무작위 재분할). 교사 seed 3개에서 0.249~0.264로 안정. **역할 정보는 정보량이 아니라 경계 위치에 있다** (1.64×). ⚠️ CUB 한정·절대값 26%·주효과 제거 후에만 가시.
- **slot 내 의미 일관성 + codon 전이** — 전 셀 lift가 chance 상회, codeword→codon 74% 보존 (2026-07-19 role alignment).
- 텍스트 집계 방식(EOS vs mean-pool vs pruning)은 성능 원인이 아님 — negative result지만 강건성의 증거 (A1, ±0.003)
- codebook drop ablation: 5~6개 slot이 non-trivial retrieval 기여 (Σ|drop| 0.038~0.053)
- **슬롯 내 의미 조직화가 codon까지 유지됨** (§4b-B): 코드 거리↔의미 거리 Spearman ρ가 shuffled(≈0.000) 대비 3/3 데이터셋에서 확인. flat hash 대비 우세는 Flickr25k(+0.079)에서만 결정적, NUS-WIDE·MS-COCO는 동률(+0.011/+0.012) — **데이터셋 의존임을 함께 적을 것**

### ❌ 주장 불가

| 주장 | 이유 |
|---|---|
| `disentangled`, `independently controllable`, `causal semantic factor` | intervention selectivity **미통과** — 핵심 control 대비 8/18 셀만 유의, 단위 섭동당 gain은 CIBHash가 **60% 우세**, slot 고유 정보 0.635b (§3) |
| "여섯 개의 독립 semantic factor" | decoding per-slot spread 0.023 + **slot 중복률 86.2%** (H(m\|나머지)=0.635b / H=4.59b) → 중복 확정 |
| **`slot-specialized`** 강한 형태 ("global slot = global", slot별 역할 명명) | Flickr §4b-A에서 반박(6개 중 3개만 argmax, 대각 우세 ≤ +0.026). **CUB에서도 raw argmax는 global slot이 269개 중 244개 독식** → 개별 slot에 역할 이름을 붙이는 서술은 불가 (§4c) |
| NMI 기반 compositional 주장 | 부호 반대 + baseline 비교 무효 + 텍스트 감독에 무반응 |
| "4개 데이터셋 SOTA" | MS-COCO +0.002, CIFAR-10 +0.004는 noise |
| atlas grounding이 baseline의 3.2배 | CUB 모델 vs Flickr baseline — cross-dataset 아티팩트 |
| 사람이 느끼는 해석 가능성 우위 | human evaluation 전무 |
| B1(text-grounding lift)이 텍스트 감독의 효과를 보여줌 | 텍스트 제거해도 0.140 → 0.139로 불변 (frozen CLIP이 이미 text-aligned) |

### 포지셔닝 한 줄

**`compositional` · `readable`은 지키고, `disentangled`와 `slot-specialized`를 모두 버린다.** 제목의 "Grounded"는 §2 decoding 결과가 지탱한다. 여섯 슬롯은 **역할이 배정된 여섯 부분이 아니라 같은 의미 내용의 부분적으로 중복된 여섯 view**로 서술한다 — §4b-A가 역할 배정을 능동적으로 반박했으므로, 이는 신중한 표현이 아니라 **측정에 따른 서술**이다.

---

## 6. `DRAFT_GROUNDEDDNA_PAPER_KO.md`에서 고쳐야 할 것

| 위치 | 문제 | 조치 |
|---|---|---|
| §4.2 표 1 (L190-195) | Ours·baseline 전부 **pre-P0 leaky 수치**. baseline 열은 val-selected가 아니라 **test-selected** 값 (CIBHash MS-COCO 0.8161 ← 실제 0.8112) | §1 표로 전면 교체 |
| §4.1 L180 vs §4.2~4.4 | L180은 P0 프로토콜을 정확히 서술하는데 이하 모든 수치는 leaky 산물. **가장 위험한 불일치** | 수치 교체로 해소 |
| §4.1 L181 | "우리 +0.005~+0.025 vs baseline +0.000~+0.004" — **두 범위의 기준이 다름**. 같은 기준(leaky−val-selected)이면 +0.007 vs +0.004이고 NUS-WIDE는 **부호가 뒤집힘** | 한 기준으로 재작성하거나 삭제 |
| §4.2 L186/197 | "3/4 SOTA, MS-COCO 2위" | "2개 명확 우위, 2개 동등"으로 |
| §4.4 표 3·4 (48-bit) | **P0 재실행 안 됨**, queue에도 없음. Flickr 48-bit는 whitening leak 상속. "MS-COCO +0.0011로 뒤집힘" 주장의 마진이 측정된 selection bias(+0.009)에 완전히 묻힘 | 미검증으로 표시하거나 절 삭제 |
| §4.5 A4 (L255, L267 **중복 붙여넣기**) | NMI 부호 반대 + 동일 단락 2회 반복 | 중복 제거 + NMI 제거, decoding으로 재측정 |
| §4.4 L228, §4.6 L285 | NMI를 "높을수록 좋음"으로 사용 | 제거 |
| §4.7 "(진행 중)" | 두 실험 모두 **완료됨** | §2·§3 결과로 교체 |
| §4.3 표 2 | A1 수치는 pre-P0지만 **단일 delta 비교이므로 결론(무영향)은 유효** | 수치 갱신 권장, 결론 유지 |
| 초록·서론 "mean pooling" | A1이 반박함 (초안 머리말이 이미 인지) | EOS pooling으로 서술 통일 |

---

## 7. 권장 후속 실험 (우선순위)

| 순위 | 실험 | 이유 | 비용 |
|---|---|---|---|
| P0 | **A4(codebook 분리)를 decoding 지표로 재측정** | NMI 제거로 A4 근거가 비었음. compositional 구조 주장의 유일한 구조적 ablation | 기존 A4 런에 `heldout_codon_decoding.py` 적용 — GPU 불필요 |
| P0 | Flickr chunk control을 E\*로 재추출 | 표 정합성 (§2 미해결 이슈) | ~2분 |
| P1 | **A2(no-text)를 decoding 지표로 재측정** | 텍스트 감독이 *해석 가능성*에 기여하는지 — B1·NMI로는 못 보였음. 성공하면 핵심 주장이 대폭 강화 | GPU 불필요 |
| ~~P0~~ **완료** | ~~CUB-200 attribute 기반 per-slot decoding~~ → **§4c 참조** | **실패 아님.** 주효과 제거 후 일치율 0.264 vs 경계파괴 대조 0.161 (1.64×, 유의, seed 안정). 역할 배정 프레이밍을 **완전히 제거할 필요는 없으나**, 개별 slot 명명은 여전히 불가 | 완료 |
| ~~P1~~ **완료** | ~~CUB flat baseline chunk control~~ | 3개 전부 우연 이하(CIMON 0.141, CIBHash 0.112, MLS3RDUH 0.063) vs ours 0.264. §4c 대조 완결 | 완료 |
| P2 | 48-bit / 4-base codon을 P0로 재실행 | §4.4 전체가 미검증 | 런 4개 |
| P2 | 3 seeds + 신뢰구간 | `REQUIRED_EXPERIMENTS` §4.5 요구 | 런 다수 |
| P3 | human evaluation | relation·color·scene slot은 label로 검증 불가 | 높음 |

---

## 8. 재현 명령

```bash
PY=/home/yschoi/.conda/envs/dna_hashing/bin/python

# --- held-out codon decoding ---
$PY scripts/heldout_codon_decoding.py --dataset Flickr25k \
  --ours_dir "result/260717+flickr25k_setting1_flickr_P0refit_e4+bs+64+e+60+proj_lr+0.001" \
  --baseline_dirs result_baseline/260527/{cibhash,cimon,mls3rduh}_flickr25k_clip_unsup60 \
  --baseline_names cibhash cimon mls3rduh \
  --train_manifest dataset/Flickr25k/setting1/train.txt \
  --out docs/heldout_decoding_flickr25k.json

# MS-COCO는 train이 DB와 disjoint -> 먼저 train 추출 (1회)
CUDA_VISIBLE_DEVICES=4 $PY scripts/extract_train_split.py \
  --config_path "result/260717+mscoco_setting1_mscoco_P0refit_e49+bs+64+e+60+proj_lr+0.001"

# --- slot intervention ---
$PY scripts/slot_intervention_eval.py --dataset Flickr25k \
  --ours_dir "result/260717+flickr25k_setting1_flickr_P0refit_e4+bs+64+e+60+proj_lr+0.001" \
  --baseline_dirs result_baseline/260527/{cibhash,cimon}_flickr25k_clip_unsup60 \
  --baseline_names cibhash cimon \
  --train_manifest dataset/Flickr25k/setting1/train.txt \
  --n_query 400 --db_subsample 12000 --k 100 --out docs/slot_intervention_flickr25k.json

# --- baseline P0 stage 2 (E* on held-out val) ---
$PY scripts/baseline_val_select_p0.py     # -> docs/baseline_p0_stage2.{json,md}
```

**환경 주의**: `.conda/bin/python`(프로젝트 루트)에는 numpy가 없다. 반드시 `/home/yschoi/.conda/envs/dna_hashing/bin/python`을 쓸 것.

---

## 9. 이 문서가 대체·무효화하는 서술

| 출처 | 무효화된 내용 |
|---|---|
| `PROJECT_LOG.md:18989` | "SOTA 주장이 살아남지 못한다" — final-epoch 잠정 프로토콜의 아티팩트. P0 stage 2에서는 4/4 우위(단 2개는 noise) |
| `PROJECT_LOG.md:19028` | "baseline은 재실행 불필요" — 틀림. commit `b9fbd42`가 이미 철회 (baseline의 val 행이 자기 학습 데이터였음) |
| `docs/p0_comparison.json` | P0 **stage 1**. 우리 모델만 90% train으로 핸디캡. stage 2(`result/260717+*_P0refit_e*`)가 정본 |
| `PROJECT_LOG.md:1144` | atlas "baseline 대비 3.2배" — CUB vs Flickr cross-dataset 아티팩트 |
| 초안 전반의 NMI 해석 | §4 참조 — 부호 반대 |
| `DRAFT...KO.md:295-296` "(진행 중)" | 두 실험 모두 완료 |
