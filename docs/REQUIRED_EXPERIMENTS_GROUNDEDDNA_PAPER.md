# GroundedDNA 논문 핵심 주장 검증을 위한 필수 실험 계획

> 작성일: 2026-07-14
> 상태: 다른 작업 세션으로 넘길 수 있는 실행 계획
> 기준 모델: `PROJECT_LOG.md`에서 채택한 **F2 per-slot text-token mean-pooling 모델**
> 핵심 목적: 검색 정확도와 별개로, GroundedDNA의 DNA code가 실제로 읽을 수 있는 compositional semantic code인지 검증한다.

## 0. 이 문서가 해결하려는 문제

GroundedDNA 논문의 가장 중요한 주장은 다음과 같다.

> 이미지의 36-bit DNA code를 여섯 semantic slot의 codon으로 나누면, 각 codon이 가리키는 semantic concept를 학습 데이터로 만든 dictionary를 통해 held-out 이미지에서도 부분적으로 유추할 수 있다.

현재 저장소에는 다음과 같은 유용한 분석이 이미 존재한다.

- `compositional_eval.py`: 동일 codeword에 배정된 이미지들의 text-feature 응집도와 shuffled baseline을 계산한다.
- `tools/codeword_concept_atlas.py`: `(slot, codeword)`별 대표 이미지와 caption 단어를 시각화한다.
- `scripts/codebook_drop_ablation_fast.py`: slot 하나를 제거했을 때 retrieval metric이 얼마나 변하는지 계산한다.
- `evaluation_siglip2.py`: retrieval, code usage, dead-code, base entropy 및 unique-code 지표를 계산한다.

그러나 이 분석만으로는 핵심 주장이 충분히 검증되지 않는다.

1. 현재 atlas와 text concentration은 평가 대상 이미지의 caption을 다시 사용하므로, unseen sample의 concept를 code만으로 예측할 수 있는지 보여 주지 못한다.
2. Codeword 수준의 응집도가 높더라도 K=128 codeword가 64개 codon으로 합쳐질 때 의미가 손실될 수 있다.
3. Slot을 제거했을 때 mAP가 변한다는 사실은 해당 slot이 의도한 semantic role만 선택적으로 담당한다는 인과적 증거가 아니다.
4. 현재 training loop는 official test split의 mAP를 이용해 best checkpoint를 선택하므로, 기존 결과를 그대로 최종 논문 수치로 사용하기 어렵다.
5. Frozen backbone과 image-only inference의 효율성은 아직 일관된 측정 프로토콜로 보고되지 않았다.

따라서 아래 다섯 작업은 선택적 부가 실험이 아니라, 논문의 중심 주장을 지탱하기 위한 필수 검증으로 취급한다.

| 우선순위 | 작업 | 검증할 주장 |
|---|---|---|
| P0 | Train/validation/test 프로토콜 수정 | 성능 수치가 test leakage 없이 재현되는가 |
| P1 | Held-out codeword/codon decoding | DNA codon만으로 unseen concept를 유추할 수 있는가 |
| P1 | Slot intervention | 각 slot이 의도한 의미를 선택적으로 바꾸는가 |
| P2 | 최종 구조 ablation | 해석 가능성이 structured text와 compositional architecture에서 오는가 |
| P2 | 효율성 측정 | Text supervision이 배포 시 retrieval 효율을 훼손하지 않는가 |

## 1. 고정해야 할 기준 모델과 공정성 조건

### 1.1 기준 모델

Reported model은 `PROJECT_LOG.md`의 2026-07-14 결론에 따라 F2로 고정한다.

- 핵심 메커니즘: 각 semantic slot caption의 모든 유효 token을 mean-pooling한다.
- 폐기된 설명: bidirectional token pruning이 성능 향상의 원인이라는 주장은 사용하지 않는다.
- Backbone: 현재 F2 실험과 동일한 frozen CLIP 계열 backbone을 사용한다.
- Code: 6 slots, slot당 3 bases, 총 18 bases 또는 36 bits를 유지한다.
- Flickr25k, MSCOCO, NUS-WIDE: 현재 reported configuration은 K=128이다.
- CIFAR10: 현재 reported configuration은 K=64이다.

현재 F2 reference result는 다음 디렉터리에서 확인할 수 있다.

```text
result/260713+flickr25k_setting1_flickr_F2_meanpool_legacy_v1.0_t1.0+bs+64+e+60+proj_lr+0.001
result/260714+mscoco_setting1_mscoco_F2_meanpool_v1.0_t1.0_K128+bs+64+e+60+proj_lr+0.001
result/260713+nuswide_setting1_nuswide_v185_sweep_F2meanpool_w0.15_x0.05_th0.05_tk0.05_cb1.5_ccs0.0_g4.595+bs+64+e+60+proj_lr+0.001
result/260714+cifar10_setting1_cifar10_F2_meanpool_ccs01_v1.0_t1.0_K64+bs+64+e+60+proj_lr+0.001
```

이 결과들은 개발 과정의 reference로는 사용할 수 있지만, test-based checkpoint selection을 거쳤으므로 수정된 protocol의 최종 논문 수치를 대신할 수는 없다.

### 1.2 모든 비교에서 고정할 조건

아래 조건은 ablation과 baseline에 동일하게 적용한다.

- 동일한 vision backbone과 pretrained checkpoint
- 동일한 image resolution과 augmentation
- 동일한 36-bit code length
- 동일한 train/validation/test manifest
- 동일한 optimizer, epoch budget, batch size 및 evaluation interval
- 동일한 random seed 집합
- 동일한 retrieval database와 relevance 정의
- 동일한 checkpoint-selection metric
- 동일한 dataset별 mAP@R cutoff
- 모델별 test set 확인 횟수 1회

하나의 ablation이 구조상 parameter 수를 정확히 맞출 수 없다면 억지로 숨기지 말고 trainable parameter 수를 함께 보고한다.

## 2. 필수 실험 A: Held-out codeword/codon decoding

### 2.1 연구 질문

이 실험은 다음 질문에 직접 답해야 한다.

1. Train split으로 만든 `(slot, codeword) -> concept` 사전이 unseen test 이미지의 concept를 예측하는가?
2. Codeword를 실제 3-base codon으로 변환한 뒤에도 concept 정보가 유지되는가?
3. GroundedDNA의 semantic slot이 flat CIBHash의 임의 6-bit chunk보다 더 잘 해독되는가?
4. K=128에서 128 codeword가 64 codon으로 합쳐질 때 얼마만큼의 의미 손실이 발생하는가?

현재 atlas와 가장 큰 차이는 **dictionary 생성 split과 평가 split을 완전히 분리한다는 점**이다.

### 2.2 데이터 누수 금지 원칙

Train dictionary를 만든 뒤에는 test caption이나 test attribute 통계를 이용해 dictionary를 수정하면 안 된다.

허용되는 정보는 다음과 같다.

- Dictionary 생성: train image의 code, train slot caption, train independent annotation
- Threshold 및 top-k 선택: validation split
- 최종 평가: test image의 code와 독립된 ground-truth concept

다음 정보는 최종 test dictionary 생성에 사용하면 안 된다.

- Test split의 Qwen/VLM caption
- Test concept frequency
- Test 결과를 본 뒤 선택한 stemming, stopword 또는 concept merge 규칙
- Test 성능을 기준으로 선택한 dictionary smoothing 값

특히 학습 teacher로 사용한 Qwen caption을 test 정답으로 다시 사용하면 순환 평가가 된다. Qwen caption은 train dictionary의 concept 이름을 만드는 데 사용할 수 있지만, test 정답은 CUB attribute, dataset label 또는 blind human annotation처럼 독립된 source여야 한다.

### 2.3 평가 데이터 우선순위

#### CUB-200: 가장 강한 정량 평가

CUB-200에는 322개 image-level attribute, species label, 15개 part annotation이 있으므로 local semantic interpretation을 검증하기 가장 적합하다.

권장 mapping은 test 결과를 보기 전에 고정한다.

| GroundedDNA slot | 독립 평가 target 예시 |
|---|---|
| `C_global` | bird species 또는 coarse taxonomy |
| `C_primary_object` | head, bill, crown 등 주요 part attribute |
| `C_secondary_object` | breast, belly, wing, tail 등 보조 part attribute |
| `C_activity_relation` | CUB에 신뢰할 target이 없으면 N/A로 보고하고 강제로 점수를 만들지 않음 |
| `C_color_texture` | color, pattern, shape attribute |
| `C_scene_type` | 공식 scene label이 없으면 human annotation 대상으로 분리 |

Repository에는 다음 annotation이 이미 존재한다.

```text
dataset/CUB_200/attributes/image_attribute_labels.txt
dataset/CUB_200/parts/part_locs.txt
dataset/CUB_200/parts/parts.txt
dataset/CUB_200/image_class_labels.txt
```

Attribute 이름 metadata가 현재 checkout에 없다면 official CUB-200-2011 metadata를 추가해 attribute group을 명시적으로 정의해야 한다. Slot mapping 파일은 test evaluation 전에 version-controlled JSON으로 저장한다.

#### Flickr25k와 MSCOCO: 일반 장면에서의 외적 타당성

Dataset multi-hot label은 객체와 전역 category 평가에는 사용할 수 있지만 color, relation, scene slot의 완전한 정답은 아니다. 따라서 다음 원칙을 따른다.

- Dataset label과 직접 대응하는 slot만 quantitative decoding에 사용한다.
- 대응하지 않는 local slot을 0점으로 처리하거나 전체 평균에 억지로 포함하지 않는다.
- Relation, color, scene은 별도의 human evaluation으로 보완한다.
- Independent annotation coverage와 평가 가능한 slot 수를 표에 함께 보고한다.

### 2.4 필요한 extraction artifact

현재 `extraction_siglip2.py`는 다음 배열을 저장한다.

```text
base_indices       [N, 18]
hash_2bit          [N, 36]
codebook_indices   [N, 6]
labels 또는 multi_hot_labels
image_paths
```

Held-out decoding을 위해 train, validation, test 각각에 대해 같은 artifact가 필요하다.

```text
extract_train.npz
extract_val.npz
extract_query.npz
extract_db.npz
```

`extract_train.npz`와 `extract_val.npz`를 additive하게 생성하도록 extraction flow를 확장한다. 기존 `extract_query.npz`와 `extract_db.npz`의 schema는 변경하지 않는다.

### 2.5 Codeword 및 codon dictionary 정의

Sample `i`, slot `m`에 대해 다음 값을 사용한다.

```text
k_i^m = codebook_indices[i, m]
b_i^m = base_indices[i, 3m : 3m+3]
y_i^m = independent concept target vector
```

Base index가 `A=0, C=1, G=2, T=3`이라면 3-base codon은 다음과 같이 0부터 63까지의 ID로 변환할 수 있다.

```text
codon_id = 16 * b_0 + 4 * b_1 + b_2
```

Train split에서 codeword dictionary와 codon dictionary를 각각 만든다.

```text
D_codeword(m, k, c) = count of train samples with codeword k and concept c
D_codon(m, b, c)    = count of train samples with codon b and concept c
```

각 count는 train support로 정규화해 conditional distribution으로 저장한다.

```text
p(c | m, k) = normalized D_codeword(m, k, c)
p(c | m, b) = normalized D_codon(m, b, c)
```

Zero-count와 작은 cluster를 위한 smoothing 값은 validation에서 한 번 선택하고 test에는 고정한다. Support가 너무 작은 code는 `unknown`으로 처리하며, 이때 coverage를 반드시 함께 보고한다. 낮은 support code를 임의의 concept로 강제 매핑하면 decoding accuracy가 부풀려질 수 있다.

### 2.6 Test-time decoding

Test image의 codeword 또는 codon만 사용해 concept ranking을 출력한다.

```text
pred_codeword(i, m) = rank concepts by p(c | m, k_i^m)
pred_codon(i, m)    = rank concepts by p(c | m, b_i^m)
```

Test caption과 test image feature는 decoder의 입력으로 사용하지 않는다. Decoder가 보는 것은 오직 `(slot, codeword)` 또는 `(slot, codon)` ID이다.

### 2.7 반드시 보고할 지표

| 지표 | 의미 |
|---|---|
| Top-1 / Top-k concept accuracy | 가장 가능성 높은 concept가 정답인지 측정 |
| Macro-F1 또는 concept mAP | class imbalance와 multi-label concept를 반영 |
| Weighted purity | 각 code가 하나의 concept에 얼마나 집중되는지 측정 |
| Conditional entropy `H(concept | code)` | code를 알았을 때 남는 의미 불확실성 측정 |
| Coverage | 충분한 train support를 가진 code로 해독 가능한 test 비율 |
| Per-slot score | 여섯 slot이 실제로 다른 강점과 약점을 갖는지 확인 |

전체 평균만 보고하지 말고 slot별 결과를 main table 또는 appendix에 남긴다.

### 2.8 K=128 collision 손실 정량화

Flickr25k, MSCOCO, NUS-WIDE의 K=128 설정에서는 하나의 slot에 128 codeword가 있지만 가능한 3-base codon은 64개뿐이다. 따라서 최소 두 codeword가 같은 codon으로 합쳐지는 collision이 구조적으로 발생한다.

다음 차이를 직접 보고한다.

```text
DecodingLoss = Score_codeword - Score_codon
EntropyIncrease = H(concept | codon) - H(concept | codeword)
```

추가로 같은 codon에 합쳐진 codeword pair의 concept distribution 사이 Jensen-Shannon divergence를 계산한다. 이 값이 크면 서로 다른 의미 codeword가 같은 codon으로 병합되었음을 뜻한다.

최종 표에는 최소한 다음 열이 필요하다.

| Dataset | Slot | Codeword score | Codon score | Decoding loss | Entropy increase | Coverage |
|---|---|---:|---:|---:|---:|---:|

이 비교 결과가 좋지 않다면 “codeword는 해석 가능하지만 DNA codon은 해석 가능하다”는 주장은 분리해서 써야 한다. Codon score가 control과 구분되지 않으면 논문의 핵심 문구를 codon-level readability가 아니라 codeword-level organization으로 낮춰야 한다.

### 2.9 Control과 baseline

Held-out decoding에는 다음 control이 반드시 필요하다.

| Control | 목적 |
|---|---|
| Majority concept | Dataset 빈도만으로 얻는 성능 확인 |
| Shuffled code assignment | Code와 concept의 관계가 우연인지 확인 |
| GroundedDNA codeword | Quantization 직후 의미 보존량 측정 |
| GroundedDNA codon | 실제 논문에서 주장하는 DNA-level 의미 측정 |
| CIBHash 36-bit의 6개 contiguous 6-bit chunk | Flat hash를 같은 slot 수와 같은 chunk capacity로 해독 |

CIBHash는 36 bits를 `[0:6], [6:12], ..., [30:36]`의 여섯 chunk로 고정하여 각 6-bit value를 0부터 63까지의 ID로 변환한다. GroundedDNA codon과 동일한 dictionary-building 및 test decoding 절차를 적용한다.

Contiguous partition 하나가 우연히 유리하거나 불리할 수 있으므로, appendix에는 bit permutation으로 만든 여러 6-bit partition의 평균과 분산도 함께 보고한다. Primary comparison은 test 전에 고정한 contiguous partition으로 한다.

### 2.10 Human evaluation이 필요한 경우

Flickr25k와 MSCOCO의 relation, color, scene slot처럼 independent label이 불완전한 경우에만 human evaluation을 사용한다.

권장 protocol은 다음과 같다.

- Test image 200개 이상을 seed로 고정해 표본 추출한다.
- Train dictionary가 출력한 slot별 top concept를 image와 함께 제시한다.
- Method 이름을 숨긴 상태에서 GroundedDNA codon과 CIBHash chunk prediction을 평가한다.
- Rater는 `relevant`, `specific`, `wrong/unsupported`를 판단하거나 1-5 relevance score를 부여한다.
- 이미지당 최소 3명의 독립 rater를 배정한다.
- Krippendorff's alpha 또는 Fleiss' kappa로 agreement를 보고한다.
- Method 차이는 image-level paired bootstrap confidence interval로 검정한다.

Human evaluator에게 Qwen 원문 caption을 정답처럼 보여 주지 않는다. 평가 대상은 train dictionary에서 나온 concept가 test image의 해당 slot과 맞는지 여부이다.

### 2.11 성공 판정

Codon readability 주장을 유지하려면 다음 조건을 만족해야 한다.

1. Held-out codon decoding이 majority와 shuffled control보다 유의하게 높다.
2. 최소한 독립 annotation이 충분한 slot에서 GroundedDNA codon이 CIBHash chunk보다 높다.
3. Codeword-to-codon decoding loss가 투명하게 보고되며, codon score가 여전히 유의미한 수준을 유지한다.
4. 결과가 특정 한 slot에만 집중되지 않고, 논문에서 semantic role을 주장하는 복수 slot에서 재현된다.

숫자 threshold를 test 결과를 본 뒤 정하지 않는다. Primary criterion은 paired bootstrap 95% confidence interval에서 GroundedDNA와 control의 차이가 0을 넘는지 여부로 둔다.

## 3. 필수 실험 B: Slot intervention

### 3.1 연구 질문

Held-out decoding은 code와 concept가 연관되어 있음을 보여 주지만, 각 slot이 검색 결과에 선택적으로 영향을 미치는지는 증명하지 못한다. Slot intervention은 다음 반사실적 질문을 검증한다.

> Query code에서 하나의 codon만 다른 image의 codon으로 바꾸면, 나머지 의미는 비교적 유지된 채 해당 slot의 target concept를 가진 retrieval result가 증가하는가?

### 3.2 Intervention 정의

Query `i`의 6-slot code와 donor `j`의 code를 다음과 같이 둔다.

```text
B_i = [b_i^0, b_i^1, b_i^2, b_i^3, b_i^4, b_i^5]
B_j = [b_j^0, b_j^1, b_j^2, b_j^3, b_j^4, b_j^5]
```

Slot `m`만 donor의 codon으로 교체한다.

```text
B_i^(m <- j) = [b_i^0, ..., b_j^m, ..., b_i^5]
```

나머지 다섯 codon은 bit 단위까지 완전히 동일해야 한다. Intervention code를 query로 사용하여 기존 database DNA code에 대해 다시 retrieval한다. 모델 forward pass나 image/text feature를 다시 계산하지 않는다.

### 3.3 Donor 선택

Random donor만 사용하면 target slot뿐 아니라 모든 의미가 달라져 해석이 어렵다. 가능한 경우 donor는 다음 기준으로 선택한다.

- Target slot concept는 query와 다르다.
- Non-target concept vector는 query와 최대한 유사하다.
- Query와 donor는 동일 sample이 아니다.
- Donor selection은 independent annotation만 사용한다.

CUB-200 attribute를 사용할 경우 target attribute group은 다르고 나머지 attribute group은 가장 가까운 donor를 선택할 수 있다. Flickr25k와 MSCOCO에서는 공통 multi-label을 최대한 유지하면서 target object label이 다른 donor를 선택한다.

유효 donor가 없는 query는 제외하되, slot별 intervention coverage를 함께 보고한다.

### 3.4 반드시 보고할 지표

Top-K retrieval 결과의 independent concept 분포를 intervention 전후로 비교한다.

```text
TargetGain@K
  = target concept prevalence after intervention
  - target concept prevalence before intervention

OffTargetDrift@K
  = mean absolute change of non-target concept prevalence

Selectivity@K
  = TargetGain@K - OffTargetDrift@K
```

추가로 다음을 보고한다.

- Retrieval-set Jaccard@K: intervention 전후 이웃 집합의 변화량
- Donor-concept rank change: donor target concept를 가진 database sample의 평균 rank 변화
- Per-slot intervention coverage
- Per-slot paired bootstrap 95% confidence interval

Global slot은 본래 여러 의미를 함께 담으므로 local slot과 같은 selectivity를 기대하지 않는다. Primary intervention 결과는 local slots `C_1`부터 `C_5`까지 보고하고, `C_global`은 non-selective reference로 분리한다.

### 3.5 Control과 비교 방법

| Method | Intervention 단위 |
|---|---|
| GroundedDNA | 의미가 지정된 3-base, 6-bit slot codon |
| CIBHash | 동일 위치의 contiguous 6-bit chunk |
| Random GroundedDNA slot | 의도한 slot이 아닌 다른 codon 교체 |
| Random donor | Non-target matching 없이 donor 선택 |
| Codebook drop | 기존 ablation과 연결하기 위한 제거 control |

CIBHash에서도 donor의 동일 chunk를 query에 복사하고 같은 지표를 계산한다. 두 방법 모두 36-bit code 중 정확히 6 bits만 바뀌므로 intervention budget이 같다.

### 3.6 성공 판정

Compositional intervention 주장을 유지하려면 다음 결과가 필요하다.

1. 의도한 local slot 교체 후 `TargetGain@K`가 양수이다.
2. `Selectivity@K`가 random-slot 및 CIBHash chunk control보다 높다.
3. 효과가 한 dataset 또는 한 slot에만 국한되지 않는다.
4. Target gain과 함께 off-target drift를 보고하여, 단순히 retrieval 전체가 무작위로 바뀐 결과가 아님을 보여 준다.

Codon decoding은 성공하지만 intervention selectivity가 없다면 “semantic association”은 주장할 수 있어도 “독립적으로 조작 가능한 compositional factor”라는 표현은 피해야 한다.

## 4. 필수 실험 C: 최종 구조의 인과적 ablation

### 4.1 목적

이 ablation은 최종 성능이 단순히 강력한 CLIP backbone이나 parameter 수에서 온 것이 아니라, structured text supervision과 semantic slot decomposition에서 왔는지 확인한다.

### 4.2 최소 실험 행렬

| ID | 구성 | Full F2 대비 단일 변경 | 검증 질문 |
|---|---|---|---|
| A0 | Structured slot text + token mean-pooling | 없음 | 최종 모델 |
| A1 | EOS pooling | 각 slot의 valid-token mean을 CLIP pooled/EOS representation으로 교체 | Mean-pooling이 실제 기여하는가 |
| A2 | No text supervision | 모든 text-derived routing/loss를 제거하고 visual-only learnable anchor 사용 | Text supervision 자체가 필요한가 |
| A3 | Random slot permutation | 매 sample/epoch마다 local text slots `1..5`를 무작위 permutation | Slot 이름과 내용의 일관된 대응이 필요한가 |
| A4 | Single global codebook | 여섯 slot/codebook을 하나의 global representation과 codebook으로 교체 | Compositional decomposition이 필요한가 |

### 4.3 각 control의 구현 원칙

#### A1: EOS pooling

Caption, tokenizer, adapter, routing, loss는 그대로 두고 text aggregation만 바꾼다. F2 cache와 EOS cache가 다른 경우 caption source와 backbone checkpoint는 동일해야 한다.

#### A2: No text supervision

Text-related loss weight만 0으로 만들고 forward에서 몰래 text feature를 계속 사용하는 형태가 되어서는 안 된다.

- `cached_text_part_raw=None`
- Text-guided OT centroid 제거
- Text-code KL, cross-modal commitment, text-hash contrastive 등 text-derived objective 제거
- Local slot 구조와 36-bit output은 유지
- Visual-only learnable anchor 또는 동일한 image-only routing을 사용

이 ablation은 text가 없을 때의 자연스러운 visual compositional baseline이어야 한다.

#### A3: Random slot permutation

Dataset 전체에 하나의 고정 permutation만 적용하면 모델이 단순히 slot 이름을 다시 학습할 수 있으므로 의미 있는 control이 아니다. 각 sample 또는 mini-batch에서 local slot `1..5`를 무작위로 섞되, global slot `0`은 유지한다.

Permutation은 caption token, attention mask, slot target을 함께 이동시켜 tensor shape를 보존한다. Seed를 기록해 재현 가능하게 만든다. 이 control은 text content의 양은 유지하면서 role consistency만 제거한다.

#### A4: Single global codebook

단순히 K=128 codeword 하나만 선택하게 하면 표현 용량까지 크게 줄어 불공정하다. 다음과 같이 total prototype parameter budget을 가능한 한 맞춘다.

```text
Full:   6 codebooks x K prototypes x D
Global: 1 codebook x (6K) prototypes x D
```

Global image representation 하나를 quantize한 뒤 18-base 또는 36-bit code로 decode한다. Semantic slot routing과 slot-specific dictionary는 제거한다. Code length, backbone, training budget은 Full과 같게 유지한다.

Parameter matching이 optimization을 지나치게 불안정하게 만들면 K=128 single-codebook 결과도 함께 보고하되, 두 설정의 parameter 수를 명시한다.

### 4.4 Ablation에서 함께 볼 지표

Retrieval metric만 비교하면 interpretability contribution을 검증할 수 없다. 각 ablation에 대해 다음을 함께 계산한다.

- Dataset 표준 mAP@R, full mAP, P@1
- Held-out codeword decoding
- Held-out codon decoding
- Slot intervention selectivity
- Codebook usage, dead-code ratio, unique-code ratio
- Inter-codebook dependence 또는 pairwise NMI
- Trainable parameter 수

### 4.5 Seed와 비교 방식

- 최소 seed: 3개
- 권장 seed: 5개
- 모든 method가 같은 seed 집합을 사용한다.
- Main table은 mean, standard deviation, 95% confidence interval을 보고한다.
- Full과 ablation의 차이는 seed와 query가 대응되므로 paired difference도 함께 계산한다.
- 하나의 seed에서 우연히 가장 좋은 run만 선택하지 않는다.

## 5. 필수 실험 D: Train/validation/test protocol 수정

### 5.1 현재 코드의 문제

`train_siglip2.py`의 현재 흐름은 다음과 같다.

1. `load_dataset(..., load_test=True)`로 official test set을 `test_loader`로 만든다.
2. 매 epoch의 validation loss를 `test_loader`에서 계산한다.
3. `eval_every`마다 `_mid_train_eval(model, test_loader, ...)`로 retrieval mAP를 계산한다.
4. 가장 높은 test mAP epoch를 `model_state_dict_best.pth`로 저장한다.
5. Final checkpoint를 이 best-test checkpoint로 교체한다.

따라서 official test query가 checkpoint selection에 반복적으로 사용된다. Baseline에도 같은 best-test protocol을 적용하면 방법 간 비교의 대칭성은 생기지만, 최종 test가 validation으로 사용되었다는 문제는 해결되지 않는다.

### 5.2 새 validation split 생성

Official test와 database 파일은 그대로 보존한다. Validation은 기존 train split에서만 분리한다.

| Dataset | 현재 train 수 | 권장 optimization train | 권장 validation query |
|---|---:|---:|---:|
| Flickr25k | 5,000 | 4,500 | 500 |
| MSCOCO | 10,000 | 9,000 | 1,000 |
| NUS-WIDE | 10,500 | 9,450 | 1,050 |
| CUB-200 | 5,994 | 5,394 | 600 |

Multi-label dataset은 random split 대신 iterative multilabel stratification을 사용한다. CUB-200과 같은 single-label dataset은 class-stratified split을 사용한다. Rare class가 validation에서 사라지지 않도록 class별 최소 sample을 보장한다.

Split manifest는 seed와 source file hash를 포함해 저장한다.

```text
dataset/<DATASET>/setting1/protocol_v1/train_opt.txt
dataset/<DATASET>/setting1/protocol_v1/val_query.txt
dataset/<DATASET>/setting1/protocol_v1/val_database.txt
dataset/<DATASET>/setting1/protocol_v1/protocol.json
```

`val_database.txt`는 optimization train subset으로 구성한다. Validation query는 train에서 분리된 held-out subset이므로 self-match가 없다. Final evaluation은 기존 official `test.txt`를 query로, `database.txt`를 database로 사용한다.

CIFAR10은 파일 기반 split 방식이 다르므로 기존 deterministic index split 위에 같은 원칙의 class-stratified validation index를 별도 저장한다.

### 5.3 수정된 training 흐름

```text
optimization train -> gradient update
validation query/database -> checkpoint 및 hyperparameter 선택
official test/database -> 모든 설정을 고정한 뒤 최종 1회 평가
```

Checkpoint selection metric은 dataset 표준 validation mAP@R로 사전에 고정한다. 동일 점수일 때는 더 이른 epoch를 선택한다. Interpretability metric을 보고 checkpoint를 다시 고르면 안 된다.

권장 저장 이름은 다음과 같다.

```text
model_state_dict_best_val.pth
validation_metrics.json
test_metrics_final.json
protocol.json
```

`model_state_dict.pth`를 자동으로 덮어쓰는 대신, 어떤 checkpoint가 final test에 사용되었는지 `protocol.json`에 명시한다.

### 5.4 Hyperparameter lock

Final test 직전에 다음 내용을 하나의 lock file로 저장한다.

- Model arguments 전체
- Dataset manifest hash
- Selected epoch
- Random seed
- Primary metric과 cutoff
- Code commit hash
- Evaluation script version

Test 결과가 좋지 않다는 이유로 hyperparameter나 epoch를 변경하면 해당 test는 더 이상 final one-shot evaluation이 아니다. 변경이 필요하면 새로운 validation study로 돌아가고, paper final test는 별도 untouched split이 없는 한 반복해서 주장하지 않는다.

### 5.5 Baseline fairness

CIBHash, CIMON, MLS3RDUH에도 같은 train/validation/test manifest와 checkpoint selection을 적용한다. Baseline의 original hyperparameter를 유지하되, epoch 선택에는 official test가 아니라 validation mAP@R만 사용한다.

모든 baseline은 다음 두 조건을 만족해야 한다.

- GroundedDNA와 같은 frozen visual backbone feature 사용
- 동일한 36-bit code length 및 retrieval evaluation code 사용

### 5.6 불확실성 보고

최소 3개 seed에 대해 다음을 보고한다.

```text
mean ± standard deviation across seeds
95% confidence interval across seeds
paired bootstrap 95% CI across test queries
```

Seed가 3개뿐이면 정규근사 CI보다 Student-t interval을 사용한다. Method 차이는 같은 query와 seed를 대응시킨 paired bootstrap으로 계산한다. 단일 run의 소수점 네 자리 차이를 유의미한 개선처럼 서술하지 않는다.

## 6. 필수 실험 E: 효율성 측정

### 6.1 분리해서 보고할 비용

GroundedDNA는 text-supervised training과 image-only inference를 구분하는 것이 contribution이므로 비용 역시 분리해야 한다.

#### Offline teacher cost

- VLM caption 생성 GPU-hours
- Dataset당 caption 생성 wall-clock time
- Caption cache 크기
- Text feature extraction time과 cache 크기
- 사용한 VLM 이름, precision 및 GPU

#### GroundedDNA training cost

- Trainable parameter 수
- 전체 parameter 수와 frozen parameter 수
- Epoch당 시간과 총 GPU-hours
- Peak GPU memory
- Training FLOPs 또는 image당 forward/backward cost

#### Image-only inference cost

- Visual backbone 포함 end-to-end latency
- Hash head만의 latency
- Batch size 1의 latency
- Batch size 64의 throughput
- Peak inference memory
- Image당 code storage: 36 bits
- Database 전체 code storage
- Hamming/base-distance search time

Offline VLM 비용을 inference latency에 포함할 필요는 없지만, 전체 pipeline 비용에서 숨기면 안 된다. “추론 시 text-free”와 “학습 전체가 text-free”를 혼동하지 않도록 표를 분리한다.

### 6.2 측정 프로토콜

동일한 GPU, CUDA, PyTorch, precision 및 input resolution에서 측정한다.

```text
warm-up iterations: 100
timed iterations: at least 1000
repetitions: 5
CUDA synchronization: before and after timed region
latency report: mean, p50, p95
throughput report: images/second
memory report: torch.cuda.max_memory_allocated()
```

Cached-feature throughput과 raw-image end-to-end throughput을 섞지 않는다. Baseline이 cached CLIP feature를 사용한다면 다음 두 표를 제공한다.

1. 동일 cached visual feature에서 hashing head만 비교
2. Raw image 입력부터 36-bit code 생성까지 end-to-end 비교

FLOPs는 backbone 포함 값과 trainable head 값으로 분리한다. FLOP counter가 Sinkhorn 또는 custom operation을 누락하면 누락 항목을 명시하고 wall-clock latency를 함께 제공한다.

### 6.3 최소 비교 대상

| Model | 이유 |
|---|---|
| GroundedDNA F2 | 최종 모델 |
| CIBHash 36-bit | 가장 강한 retrieval baseline 중 하나 |
| EOS-pooling ablation | Mean-pooling의 추가 비용 확인 |
| Single-global-codebook ablation | Compositional architecture의 비용 확인 |

## 7. 권장 구현 단위와 산출물

대규모 refactor 없이 새 script 중심으로 구현한다.

| 파일 | 역할 |
|---|---|
| `scripts/build_validation_protocol.py` | Stratified train/val manifest와 `protocol.json` 생성 |
| `scripts/heldout_codon_decoding.py` | Train dictionary 구축 및 held-out decoding 평가 |
| `scripts/slot_intervention_eval.py` | Codon/chunk swap과 selectivity 계산 |
| `scripts/profile_efficiency.py` | Parameter, latency, throughput, memory 측정 |
| `extraction_siglip2.py` | `extract_train.npz`, `extract_val.npz` 지원만 additive하게 추가 |
| `train_siglip2.py` | `val_loader`와 official `test_loader` 분리 |

각 final run 디렉터리는 최소한 다음 artifact를 가져야 한다.

```text
args.txt
protocol.json
validation_metrics.json
test_metrics_final.json
extract_train.npz
extract_val.npz
extract_query.npz
extract_db.npz
heldout_codon_decoding.json
slot_intervention.json
efficiency.json
```

Paper용 aggregate 결과는 별도 디렉터리에 모은다.

```text
result_paper/
  protocol_v1/
  retrieval_summary.csv
  heldout_decoding_summary.csv
  intervention_summary.csv
  ablation_summary.csv
  efficiency_summary.csv
```

기존 `result/2607...` 디렉터리는 수정하거나 덮어쓰지 않는다.

## 8. 실행 순서

### Phase 0: Protocol을 먼저 수정

1. Validation manifest generator를 구현한다.
2. `train_siglip2.py`가 official test 대신 validation으로 checkpoint를 선택하도록 수정한다.
3. Flickr25k의 짧은 smoke run으로 train, validation, final test가 분리되는지 확인한다.
4. Log에 official test metric이 training 종료 전 한 번도 나타나지 않는지 확인한다.

이 단계가 끝나기 전에는 새로운 full-scale result를 paper final로 간주하지 않는다.

### Phase 1: Held-out decoding pilot

1. CUB-200 attribute mapping을 test를 보기 전에 고정한다.
2. F2 구조를 CUB-200에서 protocol v1로 한 seed 학습한다.
3. `extract_train.npz`와 test extraction을 생성한다.
4. Codeword, codon, shuffled, majority, CIBHash chunk decoding을 실행한다.
5. Collision loss와 per-slot coverage를 확인한다.

Pilot에서 codon decoding이 control보다 높지 않으면 즉시 원인을 분석한다. 이 경우 대규모 3-seed ablation을 먼저 돌리지 않는다.

### Phase 2: Slot intervention pilot

1. CUB attribute 기반 matched donor를 만든다.
2. Local slot별 codon swap을 수행한다.
3. Target gain과 off-target drift를 계산한다.
4. CIBHash 6-bit chunk와 동일 budget으로 비교한다.

Decoding은 되지만 intervention이 실패하면 논문 표현을 “semantically associated parts”로 제한하고 “independently controllable factors”는 주장하지 않는다.

### Phase 3: Full ablation과 3 seeds

1. A0-A4를 같은 protocol과 seed로 실행한다.
2. Retrieval, decoding, intervention을 모두 계산한다.
3. CUB-200에서 mechanism을 먼저 검증한다.
4. Flickr25k와 MSCOCO에서 일반 장면으로 확장한다.
5. NUS-WIDE와 CIFAR10은 retrieval generalization과 효율성 표를 완성하는 데 사용한다.

### Phase 4: Human evaluation과 효율성

1. Label coverage가 부족한 slot만 blind human evaluation한다.
2. 최종 frozen checkpoint로 효율성을 측정한다.
3. Offline caption cost와 image-only inference cost를 분리해 정리한다.

## 9. 결과에 따른 논문 주장 수준

### 강한 주장 가능

다음 결과가 모두 있으면 “interpretable compositional DNA code”를 중심 주장으로 사용할 수 있다.

- Held-out codon decoding이 control과 CIBHash chunk보다 높음
- 복수 local slot에서 intervention selectivity가 확인됨
- Structured text와 slot decomposition ablation에서 성능 원인이 확인됨
- Retrieval mAP@R이 corrected protocol에서도 경쟁력 있음
- Image-only inference 효율성이 baseline과 비교해 수용 가능함

### 제한된 주장만 가능

Held-out decoding은 성공하지만 intervention이 실패하면 다음처럼 표현한다.

> GroundedDNA는 slot-conditioned semantic association을 가진 부분 코드를 학습한다.

이 경우 `disentangled`, `independently controllable`, `causal semantic factor`라는 표현은 사용하지 않는다.

### Codon-level 주장 철회 필요

Codeword decoding은 성공하지만 codon decoding이 majority 또는 CIBHash와 구분되지 않으면 다음처럼 범위를 낮춘다.

> Text supervision은 codeword organization을 개선하지만, 현재 3-base bottleneck에서는 그 의미가 완전히 보존되지 않는다.

이 경우 논문의 interpretability unit은 DNA codon이 아니라 `(slot, codeword)`가 되므로, 현재 제목과 contribution을 다시 검토해야 한다.

## 10. 완료 조건 체크리스트

- [ ] Official test를 사용하지 않는 validation protocol이 구현됨
- [ ] Dataset별 protocol manifest와 hash가 저장됨
- [ ] 최소 3 seeds의 F2 final run이 존재함
- [ ] Train-only codeword/codon dictionary가 생성됨
- [ ] Independent test target으로 held-out decoding이 완료됨
- [ ] K=128 codeword-to-codon collision loss가 정량화됨
- [ ] CIBHash 6-bit chunk decoding control이 완료됨
- [ ] Local slot intervention과 matched-donor control이 완료됨
- [ ] A0-A4 structural ablation이 동일 조건으로 완료됨
- [ ] Mean, standard deviation 및 confidence interval이 보고됨
- [ ] Trainable parameter, FLOPs, latency, throughput, memory가 측정됨
- [ ] Offline VLM 비용과 image-only inference 비용이 분리됨
- [ ] 결과에 맞춰 논문 claim의 강도가 조정됨

## 11. 다음 세션이 가장 먼저 할 일

다음 작업 세션은 새 모델 아이디어를 추가하기 전에 아래 순서로 시작한다.

1. `train_siglip2.py`의 test-based best-checkpoint selection을 val-based selection으로 분리한다.
2. Validation manifest generator를 구현하고 Flickr25k smoke test를 수행한다.
3. `extraction_siglip2.py`에 train/val extraction을 additive하게 추가한다.
4. CUB-200용 held-out decoding evaluator를 구현한다.
5. Pilot 결과가 control을 넘는지 확인한 뒤에만 full ablation을 실행한다.

이 순서는 계산 비용을 줄이기 위한 것뿐 아니라, 핵심 interpretability 주장이 실제로 성립하는지 가장 먼저 확인하기 위한 것이다.
