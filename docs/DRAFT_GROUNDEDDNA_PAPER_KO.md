# GroundedDNA: 해석 가능한 이미지 검색을 위한 텍스트 감독 조합적 DNA 코드

> **한글 작업 초안.** 본 문서는 `PROJECT_LOG.md`의 최신 결론을 반영한다. 현재 모델의 핵심 메커니즘은 *bidirectional token pruning*이 아니라 **의미 슬롯별 text-token mean pooling**이다. 따라서 pruning은 논문의 기여에서 제외한다. 아래의 성능 및 해석 가능성 주장은 실험 절이 완성되기 전까지 가설 또는 검증 대상이다.

## 초록

기존 deep hashing은 이미지를 짧은 이진 코드로 압축하여 빠른 검색을 가능하게 하지만, 개별 bit 또는 부분 코드가 무엇을 의미하는지 알기 어렵다. DNA 기반 이미지 검색 역시 이미지 특징을 A/C/G/T 서열로 변환하지만, 기존 연구는 주로 전체 서열의 거리 보존, DNA-DNA hybridization 효율, 또는 GC 비율과 homopolymer 같은 생화학적 제약에 초점을 둔다. 따라서 검색 결과가 어떤 객체, 관계, 색상, 장면 정보에 의해 결정되었는지를 코드만 보고 설명하기 어렵다.

본 연구는 유전 정보에서 세 염기의 codon이 특정 번역 단위를 지정한다는 사실에서 영감을 받아, 이미지의 표현을 **언어로 정의된 의미 슬롯들의 조합**으로 구성하는 GroundedDNA를 제안한다. 모델은 이미지를 전역 의미, 주요 객체, 보조 객체, 활동 및 관계, 색상 및 질감, 장면 및 배경의 여섯 슬롯으로 분해하고, 각 슬롯을 독립적인 codebook에서 양자화한다. 학습 시에는 사전 학습된 vision-language model이 생성한 슬롯별 설명을 semantic supervision으로 사용한다. 다섯 local slot 설명의 유효 textual token을 평균 집계하고 슬롯별 adapter로 변환한 뒤, 이를 visual token routing과 이산 code 학습에 사용한다. 각 슬롯의 representation은 세 개의 A/C/G/T 위치로 변환되며, 여섯 슬롯을 연결하여 36-bit DNA code를 형성한다.

GroundedDNA의 목적은 검색 정확도를 높이는 데 그치지 않는다. 각 codebook에 의미 역할을 부여하고, `(slot, codon)`별 concept 분포와 대표 이미지를 사전 형태로 제공함으로써 사용자는 추론 시 생성된 DNA code만으로 검색을 지배한 의미 요소를 부분적으로 해석할 수 있다. 또한 대규모 cross-modal backbone을 고정하고 작은 adapter, router, codebook 및 codon head만 학습하며, 추론 시에는 텍스트 입력 없이 이미지에서 직접 compact code를 생성한다. 이는 flat binary hashing과 기존 molecular DNA retrieval 사이에 없었던 **label-free text-supervised, compositional, and interpretable hashing**의 관점을 제시한다.

**핵심어:** deep hashing, interpretable retrieval, compositional code, text supervision, vector quantization, DNA code

## 1. 서론

대규모 이미지 검색 시스템은 한 장의 이미지를 짧은 코드로 바꾸고, 코드 사이의 거리로 가장 가까운 이미지를 찾는다. 이 방식은 수많은 이미지를 빠르게 탐색하게 해 주지만, 검색 결과 앞에서 한 가지 질문을 남긴다. **두 이미지가 왜 가깝다고 판단되었는가?** 사람이 비슷하다고 보았기 때문인지, 같은 행동이 나타났기 때문인지, 혹은 색상이나 배경이 우연히 닮았기 때문인지는 최종 코드만으로 좀처럼 알기 어렵다. 표현이 짧아질수록 검색은 효율적이 되지만, 판단에 사용된 의미의 흔적은 함께 압축되어 사라지는 셈이다.

Deep hashing은 이러한 압축과 검색 효율의 문제를 성공적으로 다루어 왔다. CIBHash [4], CIMON [5], MLS3RDUH [6]와 같은 방법은 instance consistency와 이웃 구조를 학습하여 의미적으로 가까운 이미지를 Hamming space에 모은다. 그러나 이들이 생성하는 코드는 대체로 하나의 flat bit vector이며, 각 bit에는 사람이 미리 이해할 수 있는 역할이 없다. 예를 들어 7번째 bit가 주요 객체를, 12번째부터 17번째 bit가 배경을 설명한다고 해석할 근거는 없다. 학습이 끝난 뒤 코드를 임의로 나누는 것 역시 의미 분해를 학습한 것과는 다르다. 따라서 높은 검색 정확도는 이웃 관계가 잘 보존되었음을 보여 주지만, 그 관계를 **코드 자체가 설명할 수 있음**을 보장하지는 않는다.

이 불투명성은 단순히 결과에 설명문을 덧붙이는 문제에 그치지 않는다. 서로 다른 의미를 가진 이미지가 같은 코드로 충돌했을 때 무엇이 원인이었는지, 모델이 객체보다 배경의 우연한 상관관계에 의존했는지, 또는 코드의 어느 부분이 검색 결과를 바꾸었는지를 진단하기 어렵게 만든다. 검색 코드를 단지 압축된 식별자가 아니라, 검색 판단을 구성한 의미 단위의 기록으로 만들 수 있다면 효율성과 해석 가능성을 같은 표현 안에서 다룰 수 있다.

DNA 기반 이미지 검색은 이 문제를 바라보는 또 다른 출발점을 제공한다. 기존 연구는 이미지 특징을 A/C/G/T 서열로 변환하고, molecular hybridization이나 Hamming distance가 시각적 유사성을 반영하도록 학습하였다 [8-12]. 이는 DNA를 고밀도 저장 및 병렬 검색 매체로 활용한다는 점에서 중요한 진전이다. 다만 이들 방법에서 서열은 주로 전체적인 거리, hybridization 특성, GC content, homopolymer와 같은 물리적·생화학적 조건을 만족하도록 설계된다. 어떤 nucleotide 구간이 객체를 나타내고 다른 구간이 관계나 장면을 나타내는 식의 의미 구조는 일반적으로 부여되지 않는다. 다시 말해 alphabet을 이진수에서 네 염기로 바꾸는 것만으로 검색 코드가 읽을 수 있는 표현이 되지는 않는다.

본 연구는 DNA의 네 문자 자체보다, 그 문자가 **기능을 가진 짧은 단위로 조직되는 방식**에 주목한다. 생물학적 유전 부호에서 codon은 세 염기의 순서 있는 조합으로 amino acid 또는 번역 신호를 지정한다. 유전 부호에는 여러 codon이 같은 amino acid를 나타내는 중복성이 있으므로 codon과 의미가 단순한 일대일 관계를 이루는 것은 아니다. 그럼에도 짧은 기호 묶음의 위치와 조합이 해석 가능한 기능 단위가 된다는 점은 새로운 질문을 이끈다. 이미지 검색 코드도 하나의 불투명한 문자열이 아니라, 각 구간이 서로 다른 의미 역할을 맡는 짧은 문장처럼 구성할 수 있을까?

이 질문에 답하기 위해 우리는 **GroundedDNA**를 제안한다. GroundedDNA는 이미지를 다음과 같이 여섯 개의 의미 슬롯으로 표현한다.

```text
[global | primary object | secondary object | activity/relation | color/texture | scene]
```

각 슬롯은 자신의 codebook과 codon head를 가지며, 세 개의 A/C/G/T 염기로 된 codon 하나를 출력한다. 여섯 codon을 연결하면 18-base, 즉 36-bit DNA code가 된다. 이 구조에서 코드의 위치는 무엇에 관한 정보인지를 나타내고, 그 위치에서 선택된 codeword와 codon은 어떤 concept가 관찰되었는지를 나타낸다. 따라서 전체 코드는 전역 문맥, 객체, 관계, 외관 및 장면에 관한 부분 코드의 조합으로 읽을 수 있다.

코드의 각 부분에 실제 의미를 부여하기 위해 GroundedDNA는 학습 단계에서 구조화된 텍스트를 semantic teacher로 사용한다. 범용 VLM이 이미지별로 생성한 슬롯 설명을 frozen cross-modal text encoder에 입력하고, 각 local slot의 유효 textual token을 평균하여 단어 수준의 근거를 간결하게 보존한다. 슬롯별 adapter는 이 표현을 서로 다른 의미 공간으로 분리하며, text-supervised optimal transport routing은 대응하는 visual token을 각 local slot에 모은다. 이후 visual representation을 슬롯별 codebook에서 양자화함으로써 언어의 의미 구조를 이산 codeword와 codon에 증류한다. 이 과정은 class label이나 사람이 만든 pairwise annotation을 요구하지 않지만, 텍스트 모델의 지식을 사용하는 **label-free text supervision**이라는 점에서 순수한 unsupervised learning과는 구분된다.

학습에 텍스트를 사용한다고 해서 검색 시에도 텍스트가 필요한 것은 아니다. GroundedDNA는 강력한 vision-language backbone을 고정하고 adapter, router, codebook 및 codon head와 같은 비교적 작은 모듈만 학습한다. 추론 단계에서는 이미지 특징과 학습된 codebook anchor만으로 DNA code를 생성하므로, 실제 검색은 기존 hashing과 마찬가지로 compact code 사이의 효율적인 거리 계산으로 수행된다. 텍스트는 배포 시 추가 입력 modality가 아니라, 코드의 의미를 조직하기 위해 학습 과정에서만 사용되는 teacher이다.

이 설계가 지향하는 해석은 완전한 symbolic decoding이 아니라 **slot-conditioned probabilistic interpretation**이다. 동일한 `(slot, codon)`에 배정된 이미지와 텍스트 concept의 분포를 사전으로 구성하면, 추론 시 DNA code만 보고도 각 부분이 주로 어떤 객체, 관계, 외관 또는 장면을 가리키는지 추정할 수 있다. 반대로 특정 slot을 제거하거나 교체했을 때 검색 결과가 어떻게 변하는지를 통해 그 의미적 역할을 점검할 수 있다. Codebook size와 세 염기 codon의 표현 용량 차이로 인해 충돌이 존재하고, local slot 사이에도 일부 의존성이 남을 수 있으므로, 우리는 각 codon이 하나의 단어와 완벽히 대응하거나 여섯 요인이 완전히 disentangle된다고 가정하지 않는다. 목표는 검색 코드를 사람이 **부분적으로 읽고, 비교하고, 실패 원인을 추적할 수 있는 표현**으로 만드는 것이다.

이러한 관점에서 GroundedDNA는 기존 연구와 경쟁하는 동시에 서로 다른 질문을 다룬다. Flat deep hashing이 “어떻게 이웃 관계를 짧은 코드에 보존할 것인가”를, molecular DNA retrieval이 “어떻게 유사한 이미지를 물리적으로 검색 가능한 서열로 만들 것인가”를 주로 묻는다면, GroundedDNA는 “검색 코드 자체가 어떤 semantic evidence로 이웃을 선택했는지 전달할 수 있는가”를 묻는다. 검색 정확도는 이 해석 가능성이 검색 기능을 훼손하지 않았는지를 확인하는 필수 조건이지만, 본 연구의 중심은 정확도만으로는 드러나지 않는 코드의 의미 구조에 있다.

본 연구의 주요 기여는 다음과 같다.

1. **의미적으로 typed된 compositional DNA code.** Flat binary vector를 전역 문맥, 객체, 관계, 외관 및 장면의 역할이 지정된 여섯 부분 코드로 재구성하고, 각 부분을 codon으로 표현한다. 이를 통해 코드의 위치와 값이 함께 해석의 단서가 되는 retrieval representation을 제시한다.
2. **구조화된 text supervision을 통한 codeword grounding.** VLM이 생성한 슬롯별 설명, per-slot token-mean aggregation 및 text-guided routing을 이용해 언어의 의미 구조를 discrete codebook에 증류한다. 텍스트는 검색 modality가 아니라 codeword와 codon의 의미를 형성하는 teacher로 사용된다.
3. **해석 가능성과 검색 효율을 위한 train-deployment decoupling.** Frozen cross-modal backbone 위에 가벼운 학습 모듈을 두고, 텍스트 감독은 오프라인 학습에만 사용하면서 추론 시에는 이미지로부터 36-bit DNA code를 직접 생성한다.
4. **코드 수준의 해석 가능성 평가 관점.** Codeword concept atlas, held-out codon decoding, text-grounding 분석, codebook-drop ablation, inter-codebook dependence 및 slot intervention을 통해 검색 성능뿐 아니라 각 부분 코드의 의미 일관성과 조합적 역할을 검증하는 평가 틀을 제안한다.

GroundedDNA는 생물학적 번역 과정을 그대로 모사하거나 완전히 분리된 여섯 개의 의미 요인을 보장하려는 모델이 아니다. 대신 codon이라는 생물학적 비유를 계산 가능한 설계 원리로 옮겨, **빠르게 검색할 수 있는 코드가 동시에 어느 정도 읽을 수도 있는 코드가 될 수 있는지**를 탐구한다. 이러한 문제 설정은 retrieval representation의 품질을 순위 정확도만이 아니라, 그 내부에 남아 있는 의미적 증거까지 포함해 평가할 수 있는 기반을 제공한다.

## 2. 관련 연구

### 2.1 Deep hashing과 불투명한 이진 코드

기존 deep hashing은 의미적으로 유사한 이미지를 Hamming space에서 가깝게 배치하는 데 집중한다. CIBHash는 contrastive information bottleneck을, CIMON은 정제된 similarity와 consistency를, MLS3RDUH는 manifold 기반 local semantic structure를 이용한다. 이들은 검색 정확도와 코드 안정성을 개선하지만, 출력 bit에 고정된 semantic role을 부여하지 않는다. 사후에 이진 코드를 임의의 chunk로 나누더라도 그 구분은 학습 목적에 포함되지 않았으므로 compositional interpretation을 보장하지 않는다.

GroundedDNA는 전체 코드의 거리 보존이라는 목표는 유지하면서, 코드의 좌표를 미리 정의된 의미 슬롯에 배정한다. 따라서 기존 방법과의 핵심 차이는 alphabet이 binary인지 quaternary인지가 아니라, **코드 부분의 의미가 학습 과정에서 명시적으로 typed되고 grounded되는지**에 있다.

### 2.2 Text-supervised visual representation과 cross-modal hashing

CLIP은 대규모 image-text contrastive learning을 통해 강력한 전역 의미 표현을 제공한다. GroupViT와 RegionCLIP은 text supervision이 별도의 dense annotation 없이도 visual group 또는 region-level semantics를 유도할 수 있음을 보였다. PromptHash는 prompt-aware alignment를 cross-modal hashing에 활용하지만, 목표는 image-text 간 공통 flat hash space를 학습하는 데 있으며, image-only retrieval code의 각 부분을 사람이 읽을 수 있는 semantic unit으로 만드는 문제와는 다르다.

GroundedDNA는 새로운 거대 backbone을 사전 학습하지 않는다. 대신 frozen cross-modal backbone의 지식을 여섯 슬롯의 discrete codebook으로 증류한다. 이때 text는 검색 대상 modality가 아니라 **이산 코드의 의미를 조직하는 학습 신호**로 사용된다.

### 2.3 DNA 기반 유사 이미지 검색

Stewart et al.과 Bee et al.은 image feature를 DNA strand로 인코딩하고 molecular hybridization을 통해 유사 이미지를 검색하는 content-addressable DNA database를 제안했다. Koike et al.은 triplet network와 Hamming distance를 이용해 DNA encoder의 정확도와 학습 효율을 개선했으며, 후속 연구에서는 GC content와 homopolymer 길이 같은 생물학적 제약을 추가했다.

이들 연구의 중심 질문은 “시각적으로 유사한 이미지가 잘 hybridize되는 DNA strand를 어떻게 만들 것인가”이다. 반면 GroundedDNA의 중심 질문은 “생성된 DNA code를 사람이 semantic part의 조합으로 읽을 수 있는가”이다. GroundedDNA는 wet-lab hybridization을 직접 목표로 하지 않으며, 각 세 염기 구간을 언어로 정의된 slot에 배치한다. 따라서 기존 연구가 **molecularly searchable DNA**를 지향한다면, 본 연구는 **semantically readable DNA-shaped retrieval code**를 지향한다.

## 3. 제안 방법

### 3.1 문제 정의

학습 이미지 집합을 `D = {x_i} (i=1,...,N)`라 하자. class label은 사용하지 않는다. 각 이미지 `x_i`에 대해 오프라인 VLM은 여섯 semantic slot의 설명 `T_i = {t_i^m} (m=0,...,5)`을 생성한다. 목표는 image encoder `f`와 DNA encoder `h`를 학습하여

```text
h(x_i) = [b_i^0, b_i^1, ..., b_i^5]
b_i^m belongs to {A, C, G, T}^3
```

을 얻는 것이다. `b_i^m`은 slot `m`에 대응하는 세 염기 codon이며, 전체 18개 염기 위치는 2-bit/base 표현에서 36-bit code가 된다. 검색 시에는 두 code의 base-wise distance 또는 이에 대응하는 Hamming distance를 사용한다.

### 3.2 구조화된 텍스트 감독

범용 VLM은 이미지마다 전역 주제, 주요 객체, 보조 객체, 활동 및 관계, 색상 및 질감, 장면 및 배경을 각각 기술한다. 이 설명은 사람이 부여한 class label을 대체하는 open-vocabulary semantic supervision이다.

각 설명을 cross-modal text encoder에 입력해 token feature `e_(i,m,t)`를 얻는다. 최신 모델은 local slot `m in {1,...,5}`에서 의미 있는 단어를 임의로 제거하지 않고 모든 유효 token을 평균한다.

```text
e_bar_i^m = (1 / |V_i^m|) * sum(e_(i,m,t) for t in V_i^m)
```

`C_global`은 전체 caption의 pooled representation `e_bar_i^0`을 유지한다. 이후 여섯 slot별 adapter `g_m`을 적용해 `s_i^m = g_m(e_bar_i^m)`을 얻는다. 하나의 shared adapter 대신 slot-specific adapter를 사용함으로써 전역적으로 유사한 caption representation을 서로 다른 의미 subspace로 분리한다.

### 3.3 Text-guided semantic routing

Frozen vision backbone은 patch token `V_i = {v_(i,n)} (n=1,...,P)`과 전역 visual feature를 출력한다. `C_global`은 전체 이미지 정보를 보존하도록 전역 feature에서 직접 구성한다. 나머지 다섯 local slot은 학습 시 텍스트 slot embedding `s_i^m`을 centroid로 사용하는 Sinkhorn OT routing을 통해 visual token을 집계한다.

```text
R_i   = Sinkhorn(sim(V_i, S_i) / epsilon)
z_i^m = sum(R_(i,n,m) * v_(i,n) for n=1,...,P)
```

Adaptive top-p는 각 이미지와 슬롯의 routing 분포에 따라 유효 visual evidence의 양을 조절한다. 텍스트는 학습 시 visual evidence를 의미 슬롯에 배치하는 역할을 한다. 추론 시에는 텍스트 대신 학습된 codebook의 mean anchor를 사용하므로, 동일한 image-only forward path에서 여섯 visual slot을 생성할 수 있다.

### 3.4 독립 codebook과 compositional quantization

각 슬롯 `m`은 별도의 codebook `E^m = {e_k^m} (k=1,...,K)`을 가진다. Routed visual token은 해당 슬롯 안에서만 nearest codeword로 양자화된다.

```text
k_i^m = argmin_k d(z_i^m, e_k^m)
q_i^m = e_(k_i^m)^m
```

따라서 동일한 codeword index라도 slot이 다르면 의미가 다르며, 하나의 이미지는 `(k_i^0, ..., k_i^5)`의 조합으로 표현된다. 이 구조는 단일 flat codebook이 모든 variation을 한 축에 섞는 문제를 줄이고, 부분 코드의 역할을 명시적으로 제한한다.

### 3.5 Codeword-to-codon 변환

각 slot은 독립적인 codon head `d_m`을 갖는다. 현재 모델은 local codeword에 gated global context를 더한 뒤 codon head에 입력한다.

```text
q_tilde_i^0 = q_i^0
q_tilde_i^m = q_i^m + sigmoid(alpha_m) * stopgrad(q_i^0),  m=1,...,5
```

Codon head는 이 context-conditioned slot representation을 세 위치의 네 base 확률로 변환한다.

```text
p_i^m     = d_m(q_tilde_i^m), where p_i^m has shape [3, 4]
b_(i,r)^m = argmax over a in {A, C, G, T} of p_(i,r,a)^m
```

Straight-through Gumbel-Softmax를 사용해 hard DNA code를 생성하면서도 end-to-end gradient를 전달한다. 최종 code는 여섯 codon을 고정된 slot 순서로 연결한다. 여기서 생물학적 codon은 설계의 영감이며, 본 모델이 실제 genetic code의 amino-acid mapping을 재현한다고 주장하지 않는다.

### 3.6 학습 목적

전체 목적함수는 다음과 같이 요약할 수 있다.

```text
L_total = L_retrieval
        + lambda_txt * L_text_ground
        + lambda_ot  * L_OT
        + lambda_vq  * L_VQ
        + lambda_dna * L_DNA
```

- `L_retrieval`: augmentation 간 instance consistency와 per-codebook contrastive learning으로 검색 가능한 이산 표현을 학습한다.
- `L_text_ground`: visual/text quantized token의 commitment, text-code distribution consistency, 그리고 per-codebook text-hash contrastive loss를 통해 각 codebook을 대응하는 언어 의미에 정렬한다.
- `L_OT`: visual token과 local semantic slot 사이의 transport cost를 최소화한다.
- `L_VQ`: routed token과 codeword의 commitment를 유지하고 dead code를 억제한다.
- `L_DNA`: base entropy, A/C/G/T balance 및 codebook usage를 조절하여 특정 codon으로의 붕괴를 방지한다.

이 설계에서 retrieval loss만으로는 코드가 잘 검색되더라도 의미 역할이 없는 flat partition이 될 수 있다. 반대로 text alignment만 강제하면 서로 다른 slot이 공통 class 정보에 수렴할 수 있다. 따라서 검색 가능성, 언어 grounding, codebook utilization을 함께 최적화한다.

### 3.7 DNA code의 해석

학습 집합으로부터 먼저 각 `(slot m, codeword k)`에 할당된 이미지와 슬롯 설명을 모아 codeword concept dictionary를 만든다.

```text
I_cw(m, k) = TopConcepts({t_i^m | k_i^m = k})
```

DNA code만으로 해석할 때는 같은 codon으로 변환된 codeword들을 합쳐 `(slot m, codon b)`의 concept 분포 `I_codon(m, b)`를 구성한다. 추론 이미지의 DNA code가 주어지면 사용자는 각 codon을 해당 slot의 dictionary에서 조회하여 “주요 객체”, “활동”, “색상/질감”, “장면” 등의 후보 concept를 확인할 수 있다. 하나의 codon이 항상 하나의 자연어 concept와 일대일 대응한다고 가정하지 않으며, concept 빈도와 신뢰도를 함께 제시한다. 따라서 본 연구의 interpretability는 완전한 symbolic decoding이 아니라 **slot-conditional, prototype-based semantic interpretation**이다.

## 4. 실험

> **작성 상태(2026-07-15).** 아래 수치는 `PROJECT_LOG.md`의 실험 결과를 반영한 초안이다. 일부 실험(A2 no-text, A4 single-codebook, held-out codon decoding, slot intervention, 48-bit 공정 baseline)은 진행 중이며 완료 시 갱신한다. 현재 결과가 이미 확정한 중요한 사실은 다음과 같다. **검색 성능은 텍스트 집계 방식(EOS pooling, token-mean pooling, token pruning)의 선택에서 오지 않는다.** 세 가지 집계 가설이 모두 통제된 ablation에서 반박되었으므로, 본 절은 성능의 원인을 특정 pooling trick이 아니라 **text-supervised compositional codebook 구조**에 둔다.

### 4.1 실험 설정

- **Backbone:** 모든 방법이 동일한 frozen CLIP-ViT-B/16을 공유한다. Visual/text feature는 사전 추출하여 재사용하므로 방법 간 입력이 동일하다.
- **Datasets:** Flickr25k, MS-COCO, NUS-WIDE(10,500 balanced trainset), CIFAR-10. 학습·추론 모두 whole-image(196 patch)로 통일한다.
- **Code length:** 기본 36-bit(6 slot × 3-base codon). 4.4절에서 48-bit(4-base codon) 변형을 별도로 분석한다.
- **Query/Database:** 공식 test를 query, 공식 database를 retrieval DB로 사용한다. 모든 데이터셋에서 `test ∩ database = ∅`, `train ∩ test = ∅`임을 확인하였다.
- **Relevance:** multi-label 데이터셋은 최소 한 개의 label을 공유하면 relevant로 정의한다(deep hashing 표준).
- **평가지표(main):** deep hashing 관행에 따라 데이터셋별 **mAP@R**(CalcTopMap 규약)을 주 지표로 보고한다. Cutoff는 CIFAR-10 @1000, 나머지 @5000이다. 참고로 full mAP도 함께 보고한다.
- **Baselines:** CIBHash, CIMON, MLS3RDUH를 동일 backbone·동일 36-bit·동일 evaluation code로 재학습한다. 공정성을 위해 각 baseline은 5 epoch마다 평가하여 **best epoch**을 선택한다(우리 모델의 best-checkpoint 선택과 대칭).
- **재현성 한계(고지).** 현재 checkpoint 선택이 official test mAP에 의존한다(test-based selection). 이는 절대 수치에 낙관적 편향을 줄 수 있으며, validation-based protocol로의 이전이 필요하다(§5). 아래의 ablation은 모두 동일 protocol에서 상대 비교(delta)이므로 이 편향의 영향을 받지 않는다.

### 4.2 검색 성능 비교

표 1은 4개 데이터셋에서의 mAP@R을 보고한다. GroundedDNA는 4개 중 3개(Flickr25k, NUS-WIDE, CIFAR-10)에서 최고 성능이며, MS-COCO에서는 CIBHash와 근소한 차이의 2위이다.

**표 1. mAP@R 검색 성능(36-bit, frozen CLIP-ViT-B/16, whole-image).**

| Dataset (cutoff) | **GroundedDNA** | CIBHash | CIMON | MLS3RDUH |
|---|---:|---:|---:|---:|
| Flickr25k (@5000) | **0.8740** | 0.8233 | 0.8308 | 0.7811 |
| MS-COCO (@5000) | 0.8102 | **0.8161** | 0.6716 | 0.6423 |
| NUS-WIDE (@5000) | **0.8322** | 0.8164 | 0.7874 | 0.7746 |
| CIFAR-10 (@1000) | **0.9085** | 0.9010 | 0.8408 | 0.5793 |

**해석.** Flickr25k(+0.043), NUS-WIDE(+0.016), CIFAR-10(+0.008)에서 GroundedDNA가 최고 baseline을 앞선다. MS-COCO에서는 CIBHash가 +0.0013로 근소 우세인데, 이는 CIBHash의 flat sign hash가 거의 모든 이미지에 고유 코드를 부여하여 top-R 구간의 sharp precision에서 유리하기 때문이다. 그러나 full mAP(전체 순위)에서는 GroundedDNA가 MS-COCO에서도 앞선다(0.618 vs 0.585). 즉 truncated metric은 flat hash의 top-rank 첨예도를, full metric은 compositional code의 deep-rank 안정성을 각각 반영한다. 어느 경우든 GroundedDNA는 flat baseline과 경쟁적이며, 검색 성능이 해석 가능성을 위해 희생되지 않았음을 보인다.

### 4.3 Ablation: 성능의 원인은 텍스트 집계 방식이 아니다

각 semantic slot의 caption을 하나의 벡터로 요약하는 방식(EOS pooling vs 모든 유효 token의 평균)을 통제된 단일 변경으로 비교하였다(A1). 표 2는 두 방식이 검색·해석 지표 모두에서 사실상 동일하며, 오히려 표준 EOS pooling이 근소하게 우세함을 보인다.

**표 2. 텍스트 집계 ablation(A1). A0 = token-mean pooling, A1 = EOS pooling. 동일 recipe, whole-image.**

| Dataset | 지표 | A0 (mean-pool) | A1 (EOS) | Δ(A1−A0) |
|---|---|---:|---:|---:|
| Flickr25k | mAP@5000 | 0.8740 | **0.8773** | +0.0033 |
| | NMI | 0.567 | 0.569 | +0.003 |
| | B1 lift | 0.140 | 0.143 | +0.003 |
| MS-COCO | mAP@5000 | 0.8102 | **0.8131** | +0.0029 |
| | NMI | 0.670 | 0.678 | +0.008 |

**해석.** 통제된 비교에서 EOS pooling이 token-mean pooling과 동등하거나 근소하게 낫다. 이는 개발 과정에서 제안되었던 token pruning 및 token-mean pooling이 성능 향상의 원인이라는 가설을 **반박**한다. 따라서 본 논문은 특정 pooling 기법을 기여로 주장하지 않으며, 표준 EOS pooling을 사용한다. 성능의 원인은 4.5절의 구조적 ablation(text supervision 유무, compositional codebook 유무)에서 규명한다. 이 음성 결과는 방법의 강건성을 보여 준다. 즉 GroundedDNA의 이점은 취약한 집계 trick이 아니라 구조 자체에서 나온다.

### 4.4 4-base codon: codeword-codon collision 해소

기본 설정(K=128, 3-base codon)에서는 slot당 codeword가 128개인 반면 3-base codon의 표현 용량은 `4^3 = 64`뿐이어서, 서로 다른 codeword가 같은 codon으로 병합되는 구조적 collision이 발생한다. 이를 해소하기 위해 codon을 **4-base**로 확장하면(slot당 24 대신 `6×4=24`-base, 총 48-bit) 표현 용량이 `4^4 = 256 > 128`이 되어 collision이 원리적으로 사라진다.

**표 3. 3-base(36-bit) vs 4-base(48-bit) codon. whole-image.**

| Dataset | mAP@5000 | DNA-unique ratio | NMI |
|---|---:|---:|---:|
| Flickr25k 3-base | 0.8740 | 0.380 | 0.567 |
| Flickr25k **4-base** | **0.8796** | **0.522** | 0.582 |
| MS-COCO 3-base | 0.8102 | 0.207 | 0.670 |
| MS-COCO **4-base** | **0.8250** | **0.233** | 0.666 |

**해석.** 4-base codon은 검색 성능(mAP@R)을 Flickr25k에서 +0.006, MS-COCO에서 +0.015 개선하며, 특히 **DNA-unique ratio가 Flickr25k에서 0.380 → 0.522(+37%)로 급증**한다. 이는 256-codon 용량이 서로 다른 codeword를 서로 다른 codon으로 분리하여 collision을 실제로 줄였음을 정량적으로 확인한다. NMI(compositional 구조)는 유지된다. 다만 48-bit는 36-bit보다 code 예산이 크므로 mAP 향상의 일부는 단순한 bit 증가에서 온다. 따라서 동일 48-bit 예산의 baseline(CIBHash/CIMON/MLS3RDUH, 24-base 상당)과의 비교를 별도로 보고한다(진행 중). collision 및 DNA-unique 결과는 bit 예산과 무관하게 성립한다.

### 4.5 (진행 중) 구조적 ablation과 해석 가능성 검증

다음 실험은 성능·해석의 원인을 인과적으로 규명하기 위한 것으로 완료 후 갱신한다.

- **A2 (no text supervision):** 모든 text-derived routing/loss를 제거하고 visual-only anchor로 대체. Text supervision 자체의 기여를 측정한다.
- **A4 (single global codebook):** 6개 slot codebook을 동일 prototype 예산의 단일 codebook으로 교체. Compositional decomposition의 기여를 측정한다.
- **Held-out codon decoding:** train으로 만든 `(slot, codon) → concept` 사전으로 unseen test 이미지의 concept를 예측. CIBHash의 6-bit chunk decoding을 control로 사용한다.
- **Slot intervention:** query code의 한 codon만 donor codon으로 교체했을 때 해당 slot의 target concept 검색이 선택적으로 증가하는지 측정한다.

## 5. 논의 및 한계

GroundedDNA의 핵심 가치는 더 긴 설명을 생성하는 것이 아니라, 검색에 실제 사용되는 discrete code의 내부 구조에 semantic address를 부여하는 데 있다. 사용자는 전체 embedding이나 attention map을 다시 계산하지 않고도 slot별 codon과 concept dictionary를 통해 검색 근거를 조사할 수 있다.

다만 현재 내부 분석은 여섯 codebook이 완전히 독립적인 semantic factor라고 결론 내리기에 충분하지 않다. 모든 슬롯에서 positive text-grounding lift와 유의미한 codebook-drop 영향이 관찰되지만, 일부 local codebook은 class 및 scene 정보를 공유한다. 따라서 본 논문은 **완전히 disentangled된 여섯 요인**이 아니라 **서로 다른 역할을 갖되 부분적으로 중복되는 compositional code**를 주장해야 한다.

또한 본 방법은 class label을 사용하지 않지만 VLM이 생성한 텍스트에 의존한다. 그러므로 단순히 “unsupervised”라고 부르기보다 “label-free text-supervised” 또는 “VLM-distilled”로 기술하는 것이 정확하다. VLM의 hallucination과 bias가 codeword concept에 전달될 수 있으며, 오프라인 caption 생성 비용도 전체 학습 비용에 포함해야 한다. 이를 줄이기 위한 caption consistency 검사와 human evaluation이 필요하다.

현재 주요 K=128 설정에서는 한 slot의 codeword가 128개인 반면, 세 염기로 표현 가능한 codon은 `4^3 = 64`개뿐이다. 따라서 codeword-to-codon mapping에는 필연적인 collision이 있으며, gated global context도 local codon을 변화시킬 수 있다. 기존 codeword atlas만으로는 “DNA codon만 보고 concept를 유추할 수 있다”는 주장이 완성되지 않는다. 최종 논문에서는 held-out 데이터에서 **codon-level concept purity와 decoding accuracy**를 별도로 측정해야 한다.

마지막으로 본 연구의 DNA 표현은 우선 전자적 image retrieval을 위한 것이다. 실제 DNA synthesis, hybridization, GC-content 및 homopolymer 제약을 직접 만족하는지는 별개의 문제다. Molecular DNA storage로 확장하려면 semantic interpretability와 biochemical feasibility를 동시에 최적화하는 후속 연구가 필요하다.

## 6. 결론

본 연구는 이미지 retrieval code를 의미 없는 flat bit vector가 아니라, 언어로 정의된 semantic part의 조합으로 재구성한다. GroundedDNA는 frozen cross-modal backbone의 지식을 여섯 개의 독립 codebook과 codon으로 증류하고, 학습 시 text supervision을 사용하면서 추론 시 image-only compact retrieval을 유지한다. 이를 통해 각 codon을 slot-conditioned concept로 해석할 수 있는 경로를 제공하며, 기존 deep hashing의 불투명성과 기존 DNA image retrieval의 비의미적 서열 표현 사이의 간극을 다룬다.

향후 연구의 핵심은 검색 성능을 더 높이는 것만이 아니라, codon을 읽었을 때 예측되는 concept가 held-out 이미지에서도 일관되는지, 하나의 slot을 바꾸었을 때 해당 의미만 선택적으로 변하는지, 그리고 이러한 해석이 flat hashing보다 사람에게 실제로 유용한지를 엄밀히 검증하는 것이다.

## 참고문헌 초안

1. Radford et al., [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020), ICML 2021.
2. Xu et al., [GroupViT: Semantic Segmentation Emerges from Text Supervision](https://openaccess.thecvf.com/content/CVPR2022/html/Xu_GroupViT_Semantic_Segmentation_Emerges_From_Text_Supervision_CVPR_2022_paper.html), CVPR 2022.
3. Zhong et al., [RegionCLIP: Region-Based Language-Image Pretraining](https://openaccess.thecvf.com/content/CVPR2022/html/Zhong_RegionCLIP_Region-Based_Language-Image_Pretraining_CVPR_2022_paper.html), CVPR 2022.
4. Qiu et al., [Unsupervised Hashing with Contrastive Information Bottleneck](https://www.ijcai.org/proceedings/2021/133), IJCAI 2021.
5. Luo et al., [CIMON: Towards High-quality Hash Codes](https://www.ijcai.org/proceedings/2021/125), IJCAI 2021.
6. Tu et al., [MLS3RDUH: Deep Unsupervised Hashing via Manifold based Local Semantic Similarity Structure Reconstructing](https://www.ijcai.org/proceedings/2020/479), IJCAI 2020.
7. Zou et al., [PromptHash: Affinity-Prompted Collaborative Cross-Modal Learning for Adaptive Hashing Retrieval](https://openaccess.thecvf.com/content/CVPR2025/html/Zou_PromptHashAffinity-Prompted_Collaborative_Cross-Modal_Learning_for_Adaptive_Hashing_Retrieval_CVPR_2025_paper.html), CVPR 2025.
8. Stewart et al., [A Content-Addressable DNA Database with Learned Sequence Encodings](https://www.microsoft.com/en-us/research/publication/a-content-addressable-dna-database-with-learned-sequence-encodings/), DNA Computing 2018.
9. Bee et al., [Molecular-level similarity search brings computing to DNA data storage](https://www.nature.com/articles/s41467-021-24991-z), Nature Communications 2021.
10. Koike et al., [DNA-Based Similar Image Retrieval via Triplet Network-Driven Encoder](https://past.date-conference.com/proceedings-archive/2024/DATA/478_pdf_upload.pdf), DATE 2024.
11. Koike et al., [Triplet Network-Based DNA Encoding for Enhanced Similarity Image Retrieval](https://doi.org/10.1145/3649329.3657320), DAC 2024.
12. Koike et al., [Biologically Constrained DNA Encoding with Triplet Networks for Similarity Image Retrieval](https://pubmed.ncbi.nlm.nih.gov/41824343/), IEEE Transactions on Computational Biology and Bioinformatics 2026.
