# GroundedDNA Related Works 및 MAINTABLE 문헌 선정

> 문헌·repository 감사 기준일: **2026-07-27**
> 목적: 논문의 `Related Works`에서 다룰 연구와 `Experiments`의
> MAINTABLE에 실제로 배치할 비교 방법을 구분하고, 각 연구의 핵심
> contribution·method·비교 경계를 한 문서에 고정한다.

## 1. 핵심 결정

- 문헌은 크게 **non-bio image hashing/discrete representation**과
  **bio/native-DNA retrieval and coding**으로 나눈다.
- `Related Works`의 범위는 넓게 잡되, MAINTABLE에는 다음 조건을
  만족하는 방법만 넣는다.
  - image-to-image retrieval이라는 동일 과제를 다룰 것
  - 36-bit↔18-base 또는 48-bit↔24-base의 동일 저장 용량으로 다시
    학습·평가할 수 있을 것
  - 원 방법의 핵심 objective는 유지하면서 공통 split, frozen feature
    cache, validation/refit, DNA projection을 적용할 수 있을 것
  - supervision과 외부 정보 조건을 숨기지 않을 것
- 논문의 **headline MAINTABLE은 18-base**로 한다.
  - non-bio visual-only `U0`
  - native-DNA feature-distance-pair `U0-FD`
  - GroundedDNA `VLM-T`
  - 세 그룹을 같은 표에 놓되 `Training information`과 `Code formation`
    열을 반드시 둔다.
- benchmark taxonomy 또는 정답 label을 쓰는 `U2`/`S` 방법은 headline
  순위와 합치지 않고 **별도 contextual panel**에 둔다.
- 24-base는 동일 예산의 **capacity appendix**로 보고한다. 현재
  A-recipe GroundedDNA의 24-base 결과가 없으므로 24-base 수치로
  headline 우열을 주장하지 않는다.
- 현재 완료된 modern non-bio, native-DNA, CRH 결과는 legacy cache
  provenance를 사용한 diagnostic이다. 따라서 이 문서의 strict
  MAINTABLE 수치는 모두 `-`이며, 기존 수치는 `† diagnostic appendix`
  에서만 사용할 수 있다.

## 2. 정보 조건과 문서 표기

아래 tag는 원 논문의 self-description이 아니라, 이 repository에서
비교 조건을 감사하기 위한 용어다.

| Tag | 학습 시 사용하는 정보 | 대표 방법 |
|---|---|---|
| `U0` | target label, benchmark class name, caption, 외부 text bank를 모두 사용하지 않는 visual-only objective | CIBHash, CIMON, MLS³RDUH, UGH, Bi-half, SDC, OH, HHCH, CroVCA |
| `U0-FD` | `U0`의 native-DNA 하위 유형. frozen optimization-train visual-feature distance로 pseudo-pair target을 구성 | DNA24, PRIMO |
| `U1` | target label·taxonomy는 없지만 범용 외부 지식을 사용 | authors' external bank가 검증된 DUH-EG |
| `U2` | instance label은 없지만 평가 benchmark의 class-name/concept taxonomy를 사용 | UMRCH |
| `U?` | 외부 term bank 또는 선택 provenance가 검증되지 않음 | 현재 DUH-EG adapter |
| `VLM-T` | target label·benchmark taxonomy는 objective에 없지만 VLM-generated instance caption과 frozen text encoder를 사용 | GroundedDNA |
| `S` | ground-truth train label이 objective에 직접 들어감 | CRH, Koike DATE/DAC, Koike TCBB |

`U0`, `U0-FD`, `U1`, `U2`, `U?`, `VLM-T` 방법도 공통 P0 model
selection에서는 held-out train label로 validation retrieval을 계산한다.
따라서 위 tag는 **representation objective의 정보 조건**이지 전체
실험 절차가 label-blind라는 뜻은 아니다.

문헌의 배치 표시는 다음과 같다.

| 표기 | 의미 |
|---|---|
| `RW+M-A` | Related Works 핵심 인용 + 18-base headline MAINTABLE 후보 |
| `RW+M-B` | Related Works 핵심 인용 + 정보 조건을 분리한 contextual panel 후보 |
| `RW` | Related Works에는 인용하되 현재 MAINTABLE 수치 행에서는 제외 |
| `RW/B` | 관련성은 높지만 재현 artifact 또는 protocol이 막혀 수치 행은 `-` |

## 3. Non-bio 관련 연구

### 3.1 Visual-only deep hashing: headline MAINTABLE 후보

| 연구 | 정보·배치 | 주요 contribution | 핵심 method와 본 repository의 비교 경계 |
|---|---|---|---|
| [CIBHash — *Unsupervised Hashing with Contrastive Information Bottleneck*, IJCAI 2021](https://www.ijcai.org/proceedings/2021/133) | `U0`, `RW+M-A` | 입력 복원형 비지도 해싱이 검색에 불필요한 배경까지 보존하는 문제를 information bottleneck 관점에서 다룬다. | 두 증강 view의 stochastic Bernoulli binary representation을 contrastive learning으로 정렬하고 symmetric Bernoulli IB regularizer를 적용한다. 공통 비교에서는 online image augmentation·원 backbone을 두 frozen cached view로 바꾼 matched adapter를 사용한다. |
| [CIMON — *CIMON: Towards High-quality Hash Codes*, IJCAI 2021](https://www.ijcai.org/proceedings/2021/125) | `U0`, `RW+M-A` | pretrained feature에서 만든 local pseudo-similarity의 false positive/negative와 augmentation 불안정성을 함께 줄인다. | global refinement와 statistical confidence weighting으로 similarity를 정제하고, parallel/cross semantic consistency 및 contrastive consistency를 학습한다. 원 VGG-F/online path 대신 동일 frozen cache를 입력한다. |
| [MLS³RDUH — *Deep Unsupervised Hashing via Manifold based Local Semantic Similarity Structure Reconstructing*, IJCAI 2020](https://www.ijcai.org/proceedings/2020/479) | `U0`, `RW+M-A` | 단순 kNN graph에 포함되는 의미적으로 다른 noisy neighbor를 줄이는 local-structure hashing을 제안한다. | mutual-neighbor manifold structure와 cosine similarity로 local semantic matrix를 재구성하고 log-cosh hashing loss로 학습한다. 논문과 release의 neighborhood·초기화·momentum 차이를 숨기지 않고 `paper-cache` variant로 보고한다. |
| [GreedyHash — *Greedy Hash: Towards Fast Optimization for Accurate Hash Coding in CNN*, NeurIPS 2018](https://papers.neurips.cc/paper_files/paper/2018/hash/13f3cf8c531952d72e5847c4183e6910-Abstract.html) | **UGH 설정만** `U0`, `RW+M-A` | continuous relaxation의 quantization gap 없이 discrete hash를 직접 최적화하는 간단한 학습 규칙을 제시한다. | forward에서는 hard `sign`, backward에서는 identity straight-through gradient를 사용한다. 본 비교는 feature cosine-similarity reconstruction과 cubic quantization을 쓰는 논문의 **unsupervised UGH configuration**이며, 논문 전체가 unsupervised인 것으로 서술하지 않는다. |
| [Bi-Half Net — *Deep Unsupervised Image Hashing by Maximizing Bit Entropy*, AAAI 2021](https://ojs.aaai.org/index.php/AAAI/article/view/16296) | `U0`, `RW+M-A` | 각 bit가 한 값으로 치우쳐 정보량이 줄어드는 문제를 명시적으로 bit-entropy maximization으로 다룬다. | mini-batch의 각 bit에서 절반을 `+1`, 절반을 `-1`로 rank-assignment하는 parameter-free layer를 사용하며, continuous feature와 half-half target 사이의 penalized Wasserstein optimization으로 해석한다. NUS-WIDE는 공식 trainer가 없어 adaptation 표지가 필요하다. |
| [SDC — *Unsupervised Hashing with Similarity Distribution Calibration*, BMVC 2023](https://proceedings.bmvc2023.org/53/) | `U0`, `RW+M-A` | 연속 공간의 좁고 편향된 similarity가 유한한 Hamming 값으로 압축되며 positive/negative가 겹치는 similarity collapse를 완화한다. | 개별 similarity를 그대로 복원하는 대신 hash-similarity **분포 전체**를 충분히 퍼진 symmetric Beta target에 Wasserstein distance로 정렬하고 contrastive·quantization loss를 결합한다. MAINTABLE에는 release 변형을 중복 행으로 세지 않고 `SDC-paper` 한 행만 둔다. |
| [Hashing One With All — OH, ACM MM 2023](https://doi.org/10.1145/3581783.3611977) | `U0`, `RW+M-A` | mini-batch 관계만으로 놓치는 한 이미지와 전체 dataset 사이의 global structure를 memory 기반으로 사용한다. | EMA key encoder와 binary/continuous/overview FIFO queue를 유지하고, binary Hamming affinity로 continuous memory를 집계한 overview representation에 queue-level·in-batch contrastive objective를 적용한다. 이 repository의 `OH`는 **OrthoHash가 아니다**. |
| [HHCH — *Exploring Hierarchical Information in Hyperbolic Space for Self-Supervised Image Hashing*, IEEE TIP 2024](https://pubmed.ncbi.nlm.nih.gov/38442063/) | `U0`, `RW+M-A` | 평면적인 instance relation을 넘어 이미지 collection에 내재한 hierarchy를 hash code에 보존한다. | continuous code를 Poincaré ball에 투영하고 hyperbolic K-Means hierarchy를 구성한 뒤 instance-wise·prototype-wise contrastive learning과 log-cosh quantization을 사용한다. 현 비교는 paper-equation `paper-cache3v` adapter다. |
| [CroVCA — *Image Hashing via Cross-View Code Alignment in the Age of Foundation Models*, CVPRW 2026](https://openaccess.thecvf.com/content/CVPR2026W/ECV/html/Moummad_Image_Hashing_via_Cross-View_Code_Alignment_in_the_Age_of_CVPRW_2026_paper.html) | `U0`, `RW+M-A` | foundation embedding을 짧은 binary code로 압축하면서 복잡한 다중 목적과 긴 학습을 줄이는 경량 hashing 방법을 제안한다. | 두 aligned view의 code를 symmetric stop-gradient BCE로 정렬하고 coding-rate maximization으로 collapse를 억제하는 HashCoder를 사용한다. `CroVCA-cache2v-probe`는 논문명이 아니라 frozen two-view probing이라는 **local adapter tag**이며, 논문의 LoRA/asymmetric-Hamming 결과와 구분한다. |

위 아홉 방법은 36/48-bit single-seed matched diagnostic이 존재하지만,
strict provenance 및 seeds `{42,43,44}` 조건을 아직 만족하지 않는다.
따라서 최종 논문에는 “published table reproduction”이 아니라
**source-core를 보존한 common frozen-cache comparison**으로 표현하고,
strict 재실행 전 수치는 `-`로 둔다.

### 3.2 외부 의미 정보와 supervised reference: 별도 panel

| 연구 | 정보·배치 | 주요 contribution | 핵심 method와 표 배치 판단 |
|---|---|---|---|
| [UMRCH — *Unsupervised Multi-semantic Similarity Reconstruction Contrastive Hashing for Multi-label Image Retrieval*, ESWA 2026](https://www.sciencedirect.com/science/article/pii/S0957417425040291) | `U2`, `RW+M-B` | multi-label 이미지의 복수 의미를 single global cluster가 충분히 표현하지 못하는 문제를 다룬다. | CLIP global/local patch feature와 **benchmark class-name text embeddings**로 multi-semantic pseudo-target을 만든 뒤 similarity reconstruction과 contrastive learning을 결합한다. instance label은 없지만 target taxonomy를 사용하므로 `U0` 순위에서 제외하며, 공식 범위 밖인 CIFAR-10은 `n/a`로 쓴다. |
| [DUH-EG — *Deep Unsupervised Hashing via External Guidance*, ICML 2025](https://proceedings.mlr.press/v267/song25h.html) | intended `U1`, 현재 `U?`, `RW/B` | visual-only unsupervised hashing의 의미 정보 상한을 외부 textual knowledge로 보완한다. | 외부 text database에서 중복이 적은 semantic noun을 선택하고 image–noun matching 및 internal/external hash-space 양방향 contrastive agreement를 학습한다. authors' ordered WordNet bank provenance가 확보되기 전에는 exact MAINTABLE 행을 만들지 않는다. |
| [CRH — *Codebook-Centric Deep Hashing: End-to-End Joint Learning of Semantic Hash Centers and Neural Hash Function*, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/38190) | `S`, `RW+M-B` | fixed/random class center와 two-stage semantic-center optimization의 mismatch를 label-guided dynamic center assignment로 줄인다. | 각 head의 `M=2C` binary candidate subcode에서 class별 center를 동적으로 조합하고 margin classification과 neural hash function을 공동 학습한다. multi-head codebook은 본 연구와 개념적으로 가깝지만 label이 center reassignment와 loss에 직접 들어가므로 supervised upper-bound panel에만 둔다. |
| [OrthoHash — *One Loss for All: Deep Hashing with a Single Cosine Similarity based Learning Objective*, NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/cbcb58ac2e496207586df2854b17995f-Abstract.html) | `S`, `RW` | class별 orthogonal binary proxy와 하나의 cosine objective로 discriminativeness와 quantization을 함께 학습한다. | label-derived class target을 사용하는 supervised 방법이다. 현재 strict matrix의 `OH`와 전혀 다른 논문이며, 별도 재구현 없이 `OH` 성능으로 대체해서는 안 된다. |
| [FSCH — *Unsupervised Deep Hashing With Fine-Grained Similarity-Preserving Contrastive Learning for Image Retrieval*, IEEE TCSVT 2024](https://dblp.org/rec/journals/tcsv/CaoHNW24) | `U0` claim, `RW/B` | background·비관심 객체 때문에 global similarity mining이 놓치는 fine-grained local relation을 활용한다. | patch matching의 local pair structure와 global hash similarity의 consistency, 증강 간 common regional feature contrast를 학습한다. 공개 repository에 trainer·loss·optimizer·entry point가 없어 exact row는 blocked다. |

### 3.3 MAINTABLE에는 넣지 않지만 positioning에 필요한 non-bio 문헌

#### 3.3.1 Binary hashing과 center-based learning

| 연구 | 주요 contribution과 method | 본 논문에서의 용도 |
|---|---|---|
| [DSH — *Deep Supervised Hashing for Fast Image Retrieval*, CVPR 2016](https://openaccess.thecvf.com/content_cvpr_2016/html/Liu_Deep_Supervised_Hashing_CVPR_2016_paper.html) | pairwise label similarity를 보존하는 contrastive hashing loss와 quantization regularization으로 CNN과 binary code를 공동 학습한다. | deep supervised hashing의 초기 계보를 설명한다. label 조건과 backbone이 달라 current MAINTABLE에서는 제외한다. |
| [HashNet — *Deep Learning to Hash by Continuation*, ICCV 2017](https://openaccess.thecvf.com/content_iccv_2017/html/Cao_HashNet_Deep_Learning_ICCV_2017_paper.html) | imbalanced pairwise data를 weighted likelihood로 다루고, smooth `tanh`에서 `sign`으로 수렴하는 continuation으로 discrete code를 학습한다. | binary discreteness·continuous relaxation 문제의 고전적 근거다. supervised 비교 행으로 혼합하지 않는다. |
| [CSQ — *Central Similarity Quantization for Efficient Image and Video Retrieval*, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Yuan_Central_Similarity_Quantization_for_Efficient_Image_and_Video_Retrieval_CVPR_2020_paper.html) | 각 class에 global binary hash center를 할당하고 center similarity 및 quantization으로 code를 학습한다. | flat class-center 방식과 GroundedDNA의 instance-caption-grounded slot codebook을 대비한다. |

#### 3.3.2 Product/compositional quantization

| 연구 | 주요 contribution과 method | 본 논문에서의 용도 |
|---|---|---|
| [Product Quantization, IEEE TPAMI 2011](https://pubmed.ncbi.nlm.nih.gov/21088323/) | vector를 여러 subspace로 나누고 각 subspace codebook index의 Cartesian product로 큰 조합 용량을 만든다. | multi-codebook 조합 표현의 고전적 출발점이다. GroundedDNA는 dimension partition이 아니라 semantic role별 codebook이라는 점을 구분한다. |
| [VQ-VAE — *Neural Discrete Representation Learning*, NeurIPS 2017](https://papers.neurips.cc/paper_files/paper/2017/hash/7a98af17e63a0ac09ce2e96d03992fbc-Abstract.html) | encoder output을 nearest codeword로 양자화하고 straight-through estimator와 codebook/commitment update로 discrete latent를 학습한다. | GroundedDNA의 VQ 및 EMA update 계보를 명시한다. VQ 자체를 novelty로 주장하지 않는다. |
| [HiHPQ — *Hierarchical Hyperbolic Product Quantization for Unsupervised Image Retrieval*, AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/28261) | hyperbolic product codebook attention, quantized contrastive learning, hierarchical semantic supervision으로 multi-level relation을 PQ code에 보존한다. | 최신 non-binary/compositional retrieval 선행이다. asymmetric real-valued query–codeword distance가 base-Hamming과 달라 headline 수치 행에서는 제외한다. |

#### 3.3.3 Language supervision과 interpretable hashing

| 연구 | 주요 contribution과 method | 본 논문에서의 용도 |
|---|---|---|
| [CLIP — *Learning Transferable Visual Models From Natural Language Supervision*, ICML 2021](https://proceedings.mlr.press/v139/radford21a.html) | 대규모 image–text pair를 contrastive pretraining하여 공유 embedding과 zero-shot transfer를 가능하게 한다. | frozen image/text teacher의 근거다. CLIP embedding 자체는 retrieval baseline이 아니라 GroundedDNA의 입력 조건이다. |
| [WDHT — *Weakly Supervised Deep Image Hashing Through Tag Embeddings*, CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Gattupalli_Weakly_Supervised_Deep_Image_Hashing_Through_Tag_Embeddings_CVPR_2019_paper.html) | noisy user tag를 semantic embedding으로 정제해 image-only binary hash를 weak supervision으로 학습한다. | text를 privileged training signal로 쓰고 inference는 image-only인 직접 선행 계보다. |
| [A²-Net — *Learning Attribute-Aware Hash Codes for Large-Scale Fine-Grained Image Retrieval*, NeurIPS 2021](https://proceedings.neurips.cc/paper_files/paper/2021/hash/2d3acd3e240c61820625fff66a19938f-Abstract.html) | 별도 attribute annotation 없이 visual attribute와 hash representation의 대응을 학습해 retrieval과 해석 가능성을 함께 개선한다. | attribute-level interpretable hashing이 이미 존재함을 인정하고, fixed semantic slot/UOT grounding과 구분한다. |
| [ConceptHash — *Interpretable Fine-Grained Hashing via Concept Discovery*, CVPRW 2024](https://arxiv.org/abs/2406.08457) | concept token을 발견하고 concept별 binary subcode와 language-guided class center를 학습하여 fine-grained hash를 해석 가능하게 만든다. | GroundedDNA와 가장 가까운 interpretable sub-code 선행이다. class/class-name guidance와 fine-grained benchmark를 사용하므로 현재 공통 MAINTABLE보다 Related Works의 직접 비교가 우선이다. |
| [Attribute-Aware Hashing — *Learning Attribute-Aware Hash Codes for Fine-Grained Image Retrieval via Query Optimization*, ICML 2025](https://proceedings.mlr.press/v267/wang25bj.html) | learnable query가 세밀한 visual attribute를 포착해 bit-level code와 대응하고, auxiliary high-order interaction branch로 low-bit optimization을 보강한다. | semantic component와 code의 대응을 주장할 때 반드시 인용할 최신 interpretability 선행이다. supervised fine-grained 조건이라 headline 행에서는 제외한다. |
| [LPAH — *Language-guided Patch Aggregation Hashing*, PRCV 2025/LNCS 2026](https://github.com/zhenglab/LPAH) | synthetic text로 relevant patch를 모으고 patch–word alignment를 통해 local semantic hash를 학습한다. | synthetic-caption-guided local hashing과 GroundedDNA의 caption-axis/UOT routing을 비교한다. 현재 공통 runner가 없어 수치 행은 두지 않는다. |
| [HTH — *Hierarchical Text-Guided Hashing for Open-World Image Retrieval*, IEEE TCSVT 2026](https://doi.org/10.1109/TCSVT.2026.3651707) | 자동 생성한 coarse-to-fine text와 local attention pooling, global hierarchical alignment, local fine-grained alignment를 결합해 unseen-category retrieval의 일반화를 높인다. | 최신 text-guided hashing 계보를 다루되, open-world protocol과 정보 조건이 달라 검증된 common runner가 생기기 전에는 Related Works로 한정한다. |

#### 3.3.4 OT와 local concept grounding

| 연구 | 주요 contribution과 method | 본 논문에서의 용도 |
|---|---|---|
| [Cuturi — *Sinkhorn Distances*, NeurIPS 2013](https://proceedings.neurips.cc/paper_files/paper/2013/hash/af21d0c97db2e27e13572cbf59eb343d-Abstract.html) | entropic regularization으로 OT를 Sinkhorn matrix scaling으로 효율적으로 푼다. | balanced entropic OT의 수학적 출발점이다. |
| [Chizat et al. — *Scaling Algorithms for Unbalanced Optimal Transport Problems*, Mathematics of Computation 2018](https://arxiv.org/abs/1607.05816) | marginal 불일치를 KL로 허용하는 unbalanced OT를 generalized scaling으로 푼다. | GroundedDNA의 KL-relaxed entropic UOT objective와 Sinkhorn update의 직접 이론 근거다. |
| [DOT-CBM — *Discovering Fine-Grained Visual-Concept Relations by Disentangled Optimal Transport Concept Bottleneck Models*, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Xie_Discovering_Fine-Grained_Visual-Concept_Relations_by_Disentangled_Optimal_Transport_Concept_Bottleneck_CVPR_2025_paper.html) | patch와 textual concept 사이의 disentangled OT를 통해 concept localization과 concept bottleneck prediction을 함께 학습한다. | patch↔concept OT grounding의 가장 가까운 선행 중 하나다. classification CBM과 DNA retrieval code라는 task 차이를 분명히 한다. |
| [*Unsupervised Deep Hashing Based on Multi-Scale Aggregation and Optimal Transport Matching for Image Retrieval*, Neurocomputing 2026](https://www.sciencedirect.com/science/article/pii/S092523122503262X) | multi-scale cross-image region aggregation과 augmented views 사이의 pixel-level dense OT correspondence로 background·scale 변화에 강한 unsupervised hash를 학습한다. | “OT를 hashing에 처음 사용했다”는 주장을 피하기 위한 직접 경계 문헌이다. GroundedDNA의 OT는 view correspondence가 아니라 text-defined slot으로 patch mass를 선택·배분하는 router다. |

## 4. Bio 관련 연구

### 4.1 Learned native-DNA retrieval: 직접 MAINTABLE 후보

| 연구 | 정보·배치 | 주요 contribution | 핵심 method와 재현 경계 |
|---|---|---|---|
| [Stewart et al. — *A Content-Addressable DNA Database with Learned Sequence Encodings*, DNA24 2018](https://www.microsoft.com/en-us/research/wp-content/uploads/2018/08/dna24.pdf) | `U0-FD`, `RW+M-A` | key를 미리 알아야 하는 random access를 넘어 image content similarity가 DNA hybridization에 반영되는 learned sequence database를 제안한다. | frozen VGG16-FC2를 PCA-10으로 줄이고 shared pointwise layers와 FC로 `30×4` soft DNA를 생성한다. feature-distance pseudo-pair, soft cosine-Hamming, NUPACK 30-mer yield의 analytic sigmoid fit을 BCE로 학습한다. 공식 code가 없어 현 구현은 18/24-base **analytic-transfer** clean-room adapter이며 strict 수치는 `-`다. |
| [Bee et al. — *Molecular-level Similarity Search Brings Computing to DNA Data Storage* (PRIMO), Nature Communications 2021](https://www.nature.com/articles/s41467-021-24991-z) | `U0-FD`, `RW+M-A` | learned molecular similarity search를 1.6M OpenImages 규모로 확장하고 비선형 hybridization 반응을 neural surrogate로 근사한다. | `4096→2048 ReLU→80×4` encoder와 local misalignment를 허용하는 3-mer interaction/CNN yield predictor를 두고 encoder와 NUPACK-fitted predictor를 교대로 학습한다. 현 비교는 official predictor를 freeze해 18/24-mer에 옮긴 **frozen-predictor length-transfer**이지, 길이별 thermodynamic recalibration이 아니다. |
| [Koike et al. — *DNA-based Similar Image Retrieval via Triplet Network-driven Encoder*, DATE 2024](https://past.date-conference.com/proceedings-archive/2024/DATA/478_pdf_upload.pdf) 및 [*Triplet Network-Based DNA Encoding for Enhanced Similarity Image Retrieval*, DAC 2024](https://doi.org/10.1145/3649329.3657320) | `S`, `RW+M-B` | DNA encoder를 image classification/search accuracy로 정량 평가하고 PRIMO류 simulator training을 supervised metric learning으로 대체한다. | `80×4` softmax encoder, winning-probability mask 기반 normalized DNA distance, semi-hard triplet loss, class별 positionwise consensus DNA query와 DDH classification을 사용한다. DATE 2-page와 DAC 6-page는 동일 core·저자·headline 결과를 공유하는 한 계보이므로 **`Koike-TN-DNA (DATE/DAC 2024)` 한 행**으로 집계한다. |
| [Koike et al. — *Biologically Constrained DNA Encoding with Triplet Networks for Similarity Image Retrieval*, IEEE TCBB 2026](https://pubmed.ncbi.nlm.nih.gov/41824343/) | `S`, `RW+M-B` | DATE/DAC triplet encoder가 생성하는 DNA에 GC와 homopolymer feasibility를 학습·추론 단계에서 직접 반영한다. | supervised triplet objective에 probability sharpness, differentiable homopolymer, GC-balance loss를 더하고, inference에서 긴 run의 low-confidence base를 차선 base로 바꾸는 heuristic을 적용한다. 현 18/24-base 공통-DP 결과는 original 80-base protocol의 exact reproduction이 아닌 matched adaptation이다. |

주의 사항:

- 2018 논문의 제1저자는 **Kendall Stewart**다. `Bee et al. 2018`로
  표기하지 않는다.
- `DNA24`의 `24`는 24-base 길이가 아니라 **24th DNA Computing
  conference**를 뜻한다. `DNA24-18`/`DNA24-24`에서 뒤 숫자만
  matched code length다.
- DNA24와 PRIMO는 paper가 스스로 “unsupervised”라고 명명했다는 뜻이
  아니라, 본 repository의 목표 dataset 기준으로 label-free인
  `U0-FD`다.
- Koike DATE/DAC와 TCBB는 train class label로 triplet을 만들기 때문에
  명시적인 `S`다.

### 4.2 Molecular access와 codon-inspired CBIR: Related Works 전용

| 연구 | 주요 contribution과 method | MAINTABLE에서 제외하는 이유 |
|---|---|---|
| [Tsaftaris et al. — *DNA-based Matching of Digital Signals*, ICASSP 2004](https://doi.org/10.1109/ICASSP.2004.1327177) 및 [*DNA Hybridization as a Similarity Criterion for Querying Digital Signals Stored in DNA Databases*, ICASSP 2006](https://doi.org/10.1109/ICASSP.2006.1660535) | digital signal similarity를 DNA sequence와 hybridization thermodynamics에 대응시켜 분자 반응 자체를 similarity criterion으로 사용하는 초기 방향을 제시한다. | 현대 learned image hash, 공통 dataset, mAP@R protocol이 아니다. Stewart/PRIMO 이전의 역사적 계보로 인용한다. |
| [Cas9 semantic search — *Random Access and Semantic Search in DNA Data Storage Enabled by Cas9 and Machine-guided Design*, Nature Communications 2025](https://www.nature.com/articles/s41467-025-61264-5) | Cas9 cleavage predictor와 sequence encoder를 결합해 off-target cleavage를 semantic signal로 사용하고, 1.74M images를 457개 Cas9 target address/semantic cluster로 접근한다. | per-image compact hash와 base-Hamming retrieval가 아니라 cluster-address cleavage 및 wet-lab access 과제다. |
| [Pradhan et al. — *Content-Based Image Retrieval Using DNA Transcription and Translation*, IEEE TNB 2023](https://pubmed.ncbi.nlm.nih.gov/35486561/) | image를 DNA로 전사·번역해 amino-acid feature를 구성하고 classifier/ensemble로 instance·class retrieval을 수행한다. | codon/amino-acid CBIR의 직접 개념 선행이지만 learned fixed-length similarity code가 아니다. |
| [Pradhan et al. — *DNA Encoding-Based Nucleotide Pattern and Deep Features for Instance and Class-Based Image Retrieval*, IEEE TNB 2024](https://doi.org/10.1109/TNB.2023.3303512) | image MSB DNA plane의 handcrafted nucleotide-pattern feature와, translated/amplified plane을 입력으로 쓰는 CNN class branch를 결합한다. | compact DNA code를 학습하는 문제가 아니며 handcrafted와 supervised branch가 섞여 있다. |
| [DNA-CBIR — *DNA Translation Inspired Codon Pattern-Based Deep Image Feature Extraction for Content-Based Image Retrieval*, IEEE TNB 2025](https://pubmed.ncbi.nlm.nih.gov/40031697/) | pixel의 3 MSB를 3-nt로 매핑하고 codon group histogram 및 amplified DNA plane의 CNN feature로 검색한다. | 3 bases per pixel/channel인 feature extraction 방법이라 18/24-base learned hash와 저장 용량·거리 정의가 다르다. 다만 “codon을 image retrieval에 처음 도입했다”는 주장을 막는 필수 인용이다. |
| [El-Shaikh and Seeger — *Content-based Filter Queries on DNA Data Storage Systems*, Scientific Reports 2023](https://www.nature.com/articles/s41598-023-34160-5) | structured table attribute를 content barcode로 encode하여 predicate filter와 random access를 지원한다. | semantic image similarity retrieval이 아닌 structured database filtering이다. |
| [DNA-ELMR — *An End-to-End DNA Storage Coding Method Based on a Low-Complexity Multiple Biological Constraints Loss and RL-Inspired Differentiable Solver*, ESWA 2026](https://www.sciencedirect.com/science/article/pii/S0957417426006391) | Transformer image encoder에 GC, homopolymer, hairpin, error-prone motif를 다루는 differentiable multi-constraint loss와 RL-inspired solver를 결합해 DNA storage/recovery code를 생성한다. | 목적이 retrieval ranking이 아니라 storage/reconstruction이므로 bio-loss context에만 둔다. |

### 4.3 DNA storage와 constrained coding: 동기·제약 근거

| 연구 | 주요 contribution과 method | 본 논문에서의 용도 |
|---|---|---|
| [Church et al. — *Next-Generation Digital Information Storage in DNA*, Science 2012](https://doi.org/10.1126/science.1226355) | digital data를 합성 DNA에 저장하고 sequencing으로 복구하는 현대 DNA storage proof-of-concept를 제시했다. | Introduction의 DNA storage 출발점이다. retrieval baseline은 아니다. |
| [Goldman et al. — *Towards Practical, High-Capacity, Low-Maintenance Information Storage in Synthesized DNA*, Nature 2013](https://www.nature.com/articles/nature11875) | ternary/differential mapping과 overlapping redundant oligo로 확장 가능한 archival DNA encoding을 보였다. | capacity·durability·redundancy 동기를 설명한다. |
| [Erlich and Zielinski — *DNA Fountain Enables a Robust and Efficient Storage Architecture*, Science 2017](https://doi.org/10.1126/science.aaj2038) | fountain-code droplet을 biochemical screen과 결합해 높은 밀도와 robust recovery를 달성했다. | 저장 효율과 biochemical constraint가 함께 필요한 이유를 설명한다. |
| [Organick et al. — *Random Access in Large-Scale DNA Data Storage*, Nature Biotechnology 2018](https://doi.org/10.1038/nbt.4079) | primer library로 대규모 oligo pool의 특정 file을 선택하고 zero-error recovery를 수행했다. | key/address-based random access와 content/semantic access의 차이를 만든다. |
| [HEDGES — *HEDGES Error-Correcting Code for DNA Storage Corrects Indels and Allows Sequence Constraints*, PNAS 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7414044/) | sequential hash와 tree/greedy decoding으로 insertion·deletion·substitution을 교정하면서 repeat와 windowed-GC 제약을 허용한다. | GroundedDNA의 GC/run projection이 더 넓은 physical-channel coding 문제의 일부임을 설명한다. |
| [Nguyen et al. — *Capacity-Approaching Constrained Codes With Error Correction for DNA-Based Data Storage*, IEEE TIT 2021](https://doi.org/10.1109/TIT.2021.3066430) | max homopolymer run, GC band, single insertion/deletion/substitution correction을 동시에 만족하는 low-complexity constrained code를 구성한다. | post-processing의 coding-theory 맥락과 향후 ECC 확장을 설명한다. |
| [DNA-Aeon — *DNA-Aeon Provides Flexible Arithmetic Coding for Constraint Adherence and Error Correction in DNA Storage*, Nature Communications 2023](https://www.nature.com/articles/s41467-023-36297-3) | outer fountain code와 codebook-driven arithmetic inner code로 user-defined GC, homopolymer, motif 제약 및 sequencing error를 다룬다. | 40–60% 수준의 GC와 run constraint를 쓰는 설계 근거이자, 현 DP projection이 motif/hairpin/ECC까지 해결하지 않는다는 limitation의 근거다. |

## 5. 권장 MAINTABLE 설계

### 5.1 Panel A — 18-base headline, target-label-free encoder objectives

모든 strict cell은 provenance-complete cache와 seeds `{42,43,44}`의
sealed rerun이 완료될 때까지 `-`다.

| Domain | Method | Training information | Code formation | Flickr25K @5K | MS-COCO @5K | NUS-WIDE @5K | CIFAR-10 @1K |
|---|---|---|---|---:|---:|---:|---:|
| non-bio | CIBHash | `U0` | 36-bit → 18-base | - | - | - | - |
| non-bio | CIMON | `U0` | 36-bit → 18-base | - | - | - | - |
| non-bio | MLS³RDUH | `U0` | 36-bit → 18-base | - | - | - | - |
| non-bio | GreedyHash-UGH | `U0` | 36-bit → 18-base | - | - | - | - |
| non-bio | Bi-half | `U0` | 36-bit → 18-base | - | - | - | - |
| non-bio | SDC-paper | `U0` | 36-bit → 18-base | - | - | - | - |
| non-bio | Hashing One With All (OH) | `U0` | 36-bit → 18-base | - | - | - | - |
| non-bio | HHCH | `U0` | 36-bit → 18-base | - | - | - | - |
| non-bio | CroVCA | `U0` | 36-bit → 18-base | - | - | - | - |
| bio-native | DNA24-18 analytic-transfer | `U0-FD` | learned 18×4 DNA head | - | - | - | - |
| bio-native | PRIMO-18 frozen-predictor length-transfer | `U0-FD` | learned 18×4 DNA head | - | - | - | - |
| **ours** | **GroundedDNA** | **`VLM-T`** | **six grounded codons → 18-base** | **-** | **-** | **-** | **-** |

같은 표에 놓는 목적은 동일 target dataset의 label을 representation
objective에 넣지 않은 방법들을 공통 pipeline에서 비교하기 위함이다.
이는 GroundedDNA를 visual-only `U0`라고 부르거나, 서로 다른 정보
조건을 무시한 “unsupervised SOTA”를 주장하기 위함이 아니다.

### 5.2 Panel B — additional information 또는 supervised contextual comparison

이 panel은 Panel A와 통합 순위를 만들지 않는다.

| Domain | Method | Training information | Code formation | Flickr25K @5K | MS-COCO @5K | NUS-WIDE @5K | CIFAR-10 @1K |
|---|---|---|---|---:|---:|---:|---:|
| non-bio | UMRCH | `U2`, benchmark taxonomy | 36-bit → 18-base | - | - | - | n/a |
| non-bio | CRH | `S`, train labels | 36-bit → 18-base | - | - | - | - |
| bio-native | Koike-TN-DNA (DATE/DAC 2024) | `S`, train labels | learned 18×4 DNA head | - | - | - | - |
| bio-native | Koike-BC-TN-DNA (TCBB 2026) | `S`, labels + bio-aware losses | learned 18×4 DNA head | - | - | - | - |

DUH-EG는 authors' ordered semantic bank provenance가 확인되면 `U1`
행으로 추가한다. 그 전에는 `U?` related-work/blocked 상태이며 빈
성능 행을 현재 MAINTABLE에 둘 필요가 없다.

### 5.3 24-base appendix

- 동일한 row grouping을 48-bit↔24-base로 반복한다.
- 24-base와 18-base를 같은 순위로 비교하지 않는다.
- current non-A GroundedDNA-24 결과는 A-recipe headline comparator가
  아니므로 진단값으로만 보존한다.
- PRIMO-24는 official 80-mer predictor의 length-transfer이며,
  DNA24-24는 original 30-mer analytic mapping의 transfer다.
- UMRCH/CIFAR-10은 실행 미지원이므로 `-`가 아니라 `n/a`다.

## 6. 공정 비교 protocol

1. 동일 dataset setting1 train/validation/query/database split과 동일
   frozen CLIP cache를 사용한다.
2. 각 원 논문의 핵심 loss, discrete layer, pseudo-similarity construction,
   memory/codebook mechanism은 유지한다. backbone과 online pixel
   augmentation을 cache adapter로 바꾼 경우 방법명 옆에 경계를 쓴다.
3. 36-bit non-bio code는 고정 mapping
   `00→A, 01→C, 10→G, 11→T`로 18-base로 변환한다. 48-bit는 같은
   방식으로 24-base로 변환한다.
4. native-DNA baseline은 binary를 거치지 않고 position별 4-way head로
   염기를 직접 학습한다. 이 차이를 `Code formation` 열에 공개한다.
5. Stage 1에서 test를 열지 않고 validation query 대 optimization-train
   database의 **raw base-Hamming mAP@R**로 `E*`를 고른다.
6. full designated train split에서 scratch로 `E*+1` epochs refit하고,
   official query/database는 terminal evaluation에서 정확히 한 번
   추출한다.
7. query와 database 양쪽에 동일 minimum-Hamming exact DP를 적용한다.
   - 18-base: GC count `[8,10]`, homopolymer run `≤3`
   - 24-base: GC count `[10,14]`, homopolymer run `≤3`
8. 최종 metric은 projected base-Hamming의 mAP@R이다.
   - Flickr25K, MS-COCO, NUS-WIDE: `@5000`
   - CIFAR-10: `@1000`
9. seeds `{42,43,44}`의 mean±sample-std만 strict 표에 넣는다.
10. published 16/32/64-bit 수치를 36/48-bit cell에 복사·보간하지 않는다.
11. single-seed GroundedDNA/non-bio와 three-seed bio/CRH diagnostic을
    통계적으로 직접 순위화하지 않는다.
12. strict admission을 통과하지 못한 값은 `† diagnostic`으로만 보존하고
    MAINTABLE에는 `-`를 유지한다.

## 7. Related Works 권장 서술 순서

1. **Visual-only binary hashing**
   - local similarity reconstruction에서 contrastive, memory, entropy,
     distribution calibration, hyperbolic hierarchy, foundation-feature
     probing으로 발전한 흐름을 설명한다.
   - 대표 인용: MLS³RDUH, CIBHash, CIMON, GreedyHash, Bi-half, SDC, OH,
     HHCH, CroVCA.
2. **Compositional·language-grounded·interpretable code**
   - PQ/VQ의 조합적 discrete representation과 class-center hashing을
     소개한 뒤, tag/caption/concept/attribute-guided hashing으로
     연결한다.
   - 기존 sub-code가 존재하므로 “최초의 compositional/interpretable
     hash”를 주장하지 않는다.
   - GroundedDNA의 차이는 instance VLM caption으로 고정된 여섯 semantic
     axis, patch-to-slot UOT, 독립 EMA codebook, codon composition,
     image-only inference를 한 체계로 결합한 데 둔다.
3. **Learned native-DNA similarity retrieval**
   - Tsaftaris의 molecular similarity에서 Stewart DNA24와 PRIMO의
     feature-distance/yield learning, Koike의 supervised triplet 및
     bio-aware encoder로 이어지는 직접 계보를 설명한다.
   - DNA24/PRIMO의 target-label-free `U0-FD`와 Koike의 `S`를 분리한다.
4. **Codon-inspired CBIR 및 DNA constrained coding**
   - Pradhan/DNA-CBIR가 codon image feature의 선행임을 인정하되,
     per-pixel handcrafted feature와 compact learned codon code의
     차이를 밝힌다.
   - HEDGES, constrained codes, DNA-Aeon을 통해 GC·homopolymer 제약의
     근거를 제시하고, 현재 projection이 full storage codec, ECC,
     hairpin/motif/thermodynamic guarantee는 아니라는 범위를 명시한다.

논문의 gap은 다음처럼 정리한다.

> 기존 non-bio hashing은 retrieval-efficient code를 잘 학습하지만
> biochemical validity를 목적으로 하지 않으며, native-DNA retrieval은
> hybridization 또는 supervised class separation을 최적화하지만
> compact sequence 내부의 위치별 의미 역할과 compositional
> interpretability를 제공하지 않는다. GroundedDNA는 language-defined
> visual concepts를 독립 codebook과 codon unit에 grounding하면서,
> 동일 DNA-space retrieval protocol과 sequence feasibility projection을
> 결합한다.

## 8. Repository source of truth

- modern non-bio 구현·원문 감사:
  [`baseline/MODERN_UNSUPERVISED_BASELINES.md`](../baseline/MODERN_UNSUPERVISED_BASELINES.md)
- native-DNA 구현·원문 감사:
  [`baseline/NATIVE_DNA_BASELINES.md`](../baseline/NATIVE_DNA_BASELINES.md)
- modern diagnostic aggregate:
  [`baseline_p0_matrix_seeds42_legacy_cache.md`](./baseline_p0_matrix_seeds42_legacy_cache.md)
- native-DNA 18-base diagnostic aggregate:
  [`native_dna_p0_aggregate.md`](./native_dna_p0_aggregate.md)
- native-DNA 24-base diagnostic aggregate:
  [`native_dna_p0_24base_aggregate.md`](./native_dna_p0_24base_aggregate.md)
- current A-champion diagnostic comparison:
  [`comparison_Achampion_vs_baselines_2026-07-27.md`](./comparison_Achampion_vs_baselines_2026-07-27.md)

이 문서는 **문헌·행 선정의 source of truth**이고, 위 aggregate 문서는
diagnostic 수치와 provenance의 source of truth다. 숫자 승격 여부는
[`PROJECT_LOG.md`](./PROJECT_LOG.md)의 paper invariants와 strict admission
판정을 따른다.
