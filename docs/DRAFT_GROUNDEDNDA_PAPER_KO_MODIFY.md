# GroundedDNA: 언어로 정초된 조합적 DNA 코드를 이용한 이미지 검색

> 논문 뼈대 초안 · **2026-08-04 재평가판** · 본문 문장화 전 개조식 문서
>
> 이 문서는 `DRAFT_GROUNDEDDNA_PAPER_KO.md`(2026-07-25 스냅샷)의 실험 절을 **신규 통일
> 레시피 모델의 재실험 결과로 교체**한 버전이다. 교체된 절: §0 결론, §3.8.4(신규 손실),
> §3.9(레시피), §4.5–§4.10, §5. 구 수치는 대부분 pre-P0/pre-bio 또는 구 레시피 기반이라
> 원본 draft가 스스로 "main table 사용 금지"로 표시해 둔 것들이다.
> 재평가 요약: [`newmodel_analysis/SUMMARY_newmodel_reevaluation.md`](./newmodel_analysis/SUMMARY_newmodel_reevaluation.md)

## 0. 작성 원칙과 현재 결론

- 한 문장 문제 정의
  - ground-truth label을 gradient objective에 넣지 않고 VLM이 만든 구조화 caption을 학습 시 semantic teacher로 사용
  - 이미지 한 장을 여섯 의미 슬롯의 조합으로 분해
  - 각 슬롯을 독립 EMA codebook에서 양자화하고 3-base codon으로 변환
  - 추론 시 text 없이 18-base DNA-valued code를 생성하고 생물학적 제약으로 투영하여 검색
- `PROJECT_LOG.md`에서 확인한 설계 의도
  - flat retrieval code를 global/object/relation/appearance/scene 단위의 typed composition으로 바꾸어 “가까운 이유”를 분석 가능하게 만들기
  - 강한 VLM text를 privileged training signal로만 쓰고 deployment에서는 image-only 검색을 유지하기
  - local patch를 caption-defined slot에 정초하되 UOT marginal relaxation으로 저친화 patch의 상대 질량을 낮추기
  - slot별 독립 EMA codebook으로 조합 용량과 online prototype learning을 얻고, 각 slot을 고정 길이 codon으로 직렬화하기
  - retrieval 성능뿐 아니라 code usage, collision, semantic decoding, 생물 제약 만족을 동시에 평가하기
  - 의도와 현재 증거를 구분: slot은 완전 독립 factor가 아니고, K=128 codon collision이 불가피하며, fixed DP는 학습 모듈이 아님
- 안전한 novelty 문구
  - **언어로 정의된 semantic slots, patch-to-slot UOT routing, 독립 multi-codebook quantization, codon 구조의 DNA-valued code, 고정된 생물 제약 투영을 하나의 image hashing framework로 통합**
- 사용하지 않을 novelty 문구
  - “최초의 DNA 이미지 검색” — Bee/Koike/Cas9 계열 존재
  - “최초의 codon 기반 CBIR” — Pradhan 계열 존재
  - “최초의 4진 이미지 해시” — quaternary/multi-valued hashing 존재
  - “최초의 text-guided/interpretable sub-code hashing” — ConceptHash, LPAH, HTH 등 존재
  - “최초의 patch–text OT/UOT” — DOT-CBM, OMIT, ConceptOT 등 존재
- 학습 체제 명칭
  - 권장: **label-free representation learning with VLM text supervision and label-aware validation selection**
  - 금지: “purely unsupervised”
  - 이유: ground-truth class label은 optimization objective에는 쓰지 않지만 Qwen caption과 frozen CLIP text tower의 외부 지식을 사용하고, held-out validation label은 \(E^*\) 선택에 사용
  - 재현 flag: `hash_target_mode=siglip_cos`; `jaccard` 사용 금지
  - 단, current P0는 `lambda_hash=lambda_hash_hard=0`이어서 해당 pairwise hash target 자체의 gradient 기여는 없음
- DNA 관련 주장 범위
  - 권장: “DNA-valued”, “bio-constrained”, “two representative biochemical constraints satisfied”
  - 금지: “synthesis-ready”, “wet-lab validated”, “molecular retrieval system”
  - 이유: primer/address/ECC, forbidden motif, secondary structure, synthesis·sequencing·hybridization 실험 없음
- 해석 가능성 주장 범위
  - 권장: “partially interpretable”, “semantically organized, partially redundant compositional views”
  - 금지: “fully disentangled”, “independently controllable semantic factors”, “one codon = one concept”
- 성능 주장 범위 (**2026-08-04 갱신**)
  - GroundedDNA는 seeds `{42,43,44}` 3-seed mean±std를 보고한다. baseline의 `{43,44}` 재실행은
    진행 중이므로 **상대 순위는 baseline 3-seed 완료 후에만 주장한다**
  - 권장: “조합성·다양성·해석성 세 축에서 4/4 개선을 3-seed로 보이고, 검색은 2승 2패임을 명시”
  - 금지: “statistically significant SOTA”, “전 데이터셋 전역 최적 K/L”, **“4/4 baseline 전승”**
    (single seed에서만 성립했고 3-seed에서 무너졌다)
  - 방법론적 교훈: MS-COCO의 seed 간 폭은 `.017`로 우리가 비교해 온 λ 간 폭(`.007`)보다 크다.
    **단일 seed로 `.003` 수준의 우열을 판정하지 않는다**
  - 이유: GroundedDNA와 최신 binary baseline의 strict 3-seed 반복 및 K/L grid 대부분이 미완료이고, native-DNA 3-seed 결과도 legacy cache provenance 때문에 diagnostic-only; 기존 CIBHash/CIMON/MLS3RDUH post-bio 수치는 binary-Hamming으로 고른 historical E*를 사용하여 새 raw-base-Hamming 선택 규약의 main row로 소급 사용할 수 없음
- 구현과 논문 서술의 핵심 경고
  - P0 champion backbone: 파일명과 달리 **frozen OpenAI CLIP ViT-B/16**, SigLIP2 아님
  - P0 router: classical partial OT가 아니라 **양 marginal KL-relaxed entropic UOT**
  - P0 caption pooling: legacy bidirectional-pruning flag가 켜져 있으나 score가 상수로 퇴화; semantic pruning으로 주장 불가
  - PROJECT_LOG의 clean EOS 권고안과 headline P0 결과를 분리하여 기술
  - P0의 `lambda_anchor=.05`, `lambda_cibhash_kl=.001` 등은 양수여도 실효 gradient가 없음
- 정본 우선순위
  - 논문 불변 조건: [`PROJECT_LOG.md`](./PROJECT_LOG.md)의 `PAPER INVARIANTS`
  - 수치: [`bio_projection_comparison.json`](./bio_projection_comparison.json), [`bioproj_dna_unique.json`](./bioproj_dna_unique.json), P0 result artifact
  - 구조/수식: `model_siglip2.py`, `models/semantic_router.py`, `loss_siglip2.py`, `dna_utils/bio_constraints.py`
  - 재현 설정: 각 P0 directory의 `args.txt`

## 제목 후보

- 1안: **GroundedDNA: Text-Grounded Compositional DNA Codes for Image Retrieval**
- 2안: **From Images to Semantic Codons: Bio-Constrained Compositional Hashing with Language Supervision**
- 3안: **Semantically Organized DNA-Valued Hashes: Language-Grounded Compositional Codes for Image Retrieval**
- 제목에서 피할 표현
  - “Molecular Retrieval” — wet-lab 검증 없음
  - “Biological Semantics” — 실제 genetic translation/amino-acid mapping 없음
  - “Unsupervised” 단독 표기 — VLM text supervision 은폐 위험

## 초록 뼈대

- 배경
  - deep hashing의 짧고 빠른 code와 DNA storage의 고밀도·장기 보존 잠재력
  - 기존 flat binary/DNA code의 낮은 sub-code 해석 가능성
- 공백
  - 기존 molecular image retrieval: hybridization·triplet distance·물리 제약 중심
  - 기존 interpretable hashing: binary concept sub-code 중심, DNA/codon·bio-feasibility 미통합
- 제안
  - Qwen이 생성한 여섯 semantic-axis caption을 privileged supervision으로 사용
  - frozen CLIP patch와 local caption anchor 사이 KL-relaxed UOT routing
  - 여섯 독립 EMA codebook과 slot-conditioned 3-base codon head
  - image-only inference와 최소 base-Hamming bio projection
- 결과
  - 동일 18-base 공간과 동일 bio projection 아래 mAP@R
  - Flickr25K `.8723`, MS-COCO `.8063`, NUS-WIDE `.8274`, CIFAR-10 `.9009`
  - native-DNA matched adaptation은 동일 5-epoch candidate grid에서 raw 18-base validation mAP@R로 E*를 선택한 sealed 3-seed scratch refit 결과를 별도 diagnostic 표에 보고
  - 최신/legacy binary hashing baseline의 strict 3-seed 결과는 provenance-complete 재실행 전까지 `-`
- 해석성 결과
  - held-out projected-codon decoding의 majority·post-hoc chunk control 대비 결과를 보고하되, 새 공통 E*로 baseline을 재생성하기 전 기존 우위 수치는 historical diagnostic으로만 유지
  - local slot은 완전 분리 요인이 아니라 부분 중복된 semantic views
- 결론
  - semantic organization을 학습하고 고정된 inference-time bio projection과 함께 평가하는 관점 제안

## 1. Introduction

### 1.1 도입부 스토리텔링

- 문단 1 — “저장”에서 “검색”으로
  - DNA가 초고밀도·장기 archive 매체라는 점으로 시작
  - 파일을 DNA에 쓰는 것만으로는 대규모 archive 안에서 원하는 내용을 찾을 수 없다는 전환
  - exact address/PCR access와 content-based similarity access의 차이 제시
  - Church et al. 2012, Goldman et al. 2013, Organick et al. 2018 인용
- 문단 2 — DNA 위의 유사도 검색
  - Stewart et al. DNA24 2018: **`U0-FD` unsupervised; target-label-free encoder objective**; VGG+PCA feature를 30-nt sequence로 학습하고 Hamming 기반 yield 근사와 소규모 wet-lab 검증
  - Bee et al. PRIMO: **`U0-FD` unsupervised; target-label-free encoder objective**; learned DNA sequence와 hybridization을 통한 1.6M image similarity search
  - Koike et al. DATE/DAC 2024: **`S` supervised**; ground-truth label 기반 triplet-network DNA encoder를 통한 image retrieval
  - Koike et al. TCBB 2026: **`S` supervised**; GC·homopolymer-aware loss를 추가한 bio-aware 확장
  - Cas9 semantic search: molecular random access와 semantic cluster retrieval
  - 성과 인정 후 공백 제시: sequence가 “왜 두 이미지가 가까운가”를 semantic parts로 설명하지는 않음
- 문단 3 — conventional hashing의 장점과 한계
  - compact binary code와 Hamming search의 효율성
  - 전통적 방법 대부분은 global/flat code 중심
  - 단, category-aware·concept-subcode 방법도 존재하므로 “모두 불투명”이라는 절대화 금지
  - 최신 엄격한 visual-only 계열은 false-negative 완화(DDCH), similarity-distribution calibration(SDC), 전체 memory를 hash-distance attention으로 요약하는 Overview Hashing(OH), hidden-group/similarity-knowledge modeling(CGHash), hyperbolic hierarchy(HHCH), masked-patch 복원(CTMIH), foundation-feature cross-view code alignment(CroVCA)로 label-free visual structure를 강화
  - HiHPQ는 `U0` visual-only이지만 asymmetric product-quantization distance를 쓰므로 binary/Hamming main comparison과 분리하고, 외부 Faster R-CNN pseudo-label을 쓰는 OMUH는 `U1` 계열로 분리
  - FSCH는 regional common feature와 global–local similarity를 제안하는 중요한 선행이지만, 공개 repo에 trainer·loss·optimizer가 누락되어 현재 exact executable baseline으로는 취급하지 않음
  - DUH-EG의 의도된 조건은 WordNet+CLIP `U1`이지만, 저자 ordered bank provenance가 없는 현재 user-list adapter는 `U?`; target taxonomy semantic set과 일치하면 `U2`로 보수적으로 재분류. UHSCM·UMRCH도 benchmark taxonomy를 쓰므로 visual-only와 분리
  - TGUH는 generated/descriptive text를 소비하는 별도 multimodal-guidance 계열; 공개 code completeness 감사 완료 전에는 실행 baseline으로 주장하지 않음
  - 따라서 “language가 처음 hashing을 감독한다”가 아니라, “language-defined semantic roles를 bio-valid compositional DNA code의 고정 위치에 조직한다”를 차별점으로 설정
  - 남는 질문: semantic guidance를 flat binary code가 아니라 bio-valid, role-typed compositional DNA code로 어떻게 조직할 것인가
- 문단 4 — naïve binary-to-DNA adaptation의 한계
  - 실험 baseline으로서 `00→A, 01→C, 10→G, 11→T` 변환 정의
  - bit geometry를 base geometry로 재표기할 뿐 semantic role, codon boundary, biochemical feasibility를 새로 학습하지 않음
  - 실제 Bee/Koike/Pradhan 계열 전체를 이 단순 변환으로 일반화하지 않음
- 문단 5 — compositional/interpretable code의 필요성
  - 검색 code를 하나의 opaque identifier가 아닌 근거 단위들의 조합으로 설계할 필요
  - global/object/relation/appearance/background를 고정 위치에 기록하면 failure diagnosis와 부분 분석 가능
  - ConceptHash, attribute-aware hashing과의 연결 및 차이 예고
- 문단 6 — codon motivation
  - 자연 codon: 세 nucleotide로 구성된 번역 단위, 64 triplets, genetic-code degeneracy
  - 본 연구: amino acid 번역을 모사하지 않고 “짧은 기호 묶음이 위치별 기능 단위가 된다”는 구조적 비유만 사용
  - 선택된 slot codeword를 slot-conditioned head로 3-base 단위에 decode하고 여섯 단위를 연결; codeword–codon 일대일 대응은 보장하지 않음
  - Pradhan 계열은 pixel/MSB-derived DNA plane에서 codon·amino-acid feature를 구성하지만, 본 연구는 genetic translation table 없이 language-defined slot을 학습된 codon에 기록
- 문단 7 — GroundedDNA 개요
  - Qwen caption → CLIP text anchor → patch-to-slot UOT → EMA VQ → codon head → bio projection
  - 학습 시 text teacher, 추론 시 image only
  - representation-learning gradient에는 relevance label을 사용하지 않으며, held-out validation/test label은 \(E^*\) 선택과 공식 평가에만 사용
- 문단 8 — 핵심 실험 메시지
  - 네 표준 image hashing benchmark
  - 모든 방법을 동일 18-base·base Hamming·동일 bio projection으로 평가
  - 검색 성능, bio feasibility, held-out decoding, code semantics를 함께 보고

### 1.2 Main Contributions 초안

- **Text-grounded compositional DNA hashing**
  - 여섯 language-defined slots와 독립 codebook/codon head로 구성된 18-base retrieval representation
- **Train–deployment decoupled UOT routing**
  - 학습 시 instance caption anchor, 추론 시 codebook-mean anchor를 사용하는 text-free image hashing
  - adaptive top-p로 UOT plan의 patch-to-slot 상대 가중치를 희소화
- **EMA multi-codebook와 bio-aware decoding의 통합**
  - 학습된 slot별 online-k-means형 codebook·soft/argmax codon decoder와 고정된 최소 base-Hamming DP projection
- **검색·생물 제약·코드 의미의 통합 평가**
  - 동일 post-projection code space에서 baseline 비교
  - held-out codon decoding, semantic-distance correlation, intervention 및 failure-analysis protocol
- 기여 문장 끝의 보수적 한정
  - “codon 자체가 해석 가능성을 보장하지 않으며, 관측된 의미 구조를 정량 분석으로 검증”

## 2. Related Works

### 2.1 문헌 조사 범위와 포함·제외 기준

- 조사 cutoff: 2026-07-21
- high-recall 검색 대상
  - Google Scholar, Crossref, DBLP, arXiv, PubMed, IEEE Xplore
  - CVF Open Access, PMLR, IJCAI, NeurIPS, Nature 계열
  - 핵심 논문의 backward/forward citation chaining과 공개 code 확인
- 권장 검색식 묶음
  - `deep image hashing`, `unimodal image retrieval hashing`, `interpretable sub-code hashing`
  - `quaternary hash image retrieval`, `multiple code hashing`, `compositional quantization`
  - `DNA image retrieval`, `DNA similarity search`, `codon content-based image retrieval`
  - `text supervised image hashing`, `caption supervised hashing`, `language guided patch hashing`
  - `patch text optimal transport`, `unbalanced optimal transport visual grounding`, `selective Sinkhorn routing`
- “전부”의 논문상 표현
  - 문자 그대로의 완전성 주장 금지
  - “2026-07-21까지 정의한 DB·검색식·citation chaining으로 수행한 재현 가능한 high-recall review”로 기술
- main scope 포함
  - unimodal image-to-image hashing
  - non-binary/compositional quantization
  - text-supervised/interpretable image hashing
  - DNA storage의 random/similarity retrieval와 codon-inspired CBIR
  - OT 기반 fine-grained vision–language alignment
- main baseline에서 제외
  - image↔text cross-modal hashing — query/database modality가 다름
  - genomic sequence/k-mer hashing — DNA가 출력 code가 아니라 입력 생물 서열
  - DNA image encryption — 보안 변환 목적
  - image-to-DNA reconstruction/storage — similarity retrieval 목적 아님
  - 단, 경계 설정과 bio-constraint 설계 근거에는 관련 문헌으로 인용

### 2.2 Deep binary hashing과 non-binary/compositional code

| 연구 | 핵심 내용 | 본 논문에서 사용할 요점 |
|---|---|---|
| [DSH, CVPR 2016](https://openaccess.thecvf.com/content_cvpr_2016/html/Liu_Deep_Supervised_Hashing_CVPR_2016_paper.html) | pairwise supervised similarity와 quantization | CNN deep hashing의 출발점; label-dependent baseline 계보 |
| [DHN, AAAI 2016](https://ojs.aaai.org/index.php/AAAI/article/view/10235) | pairwise likelihood와 quantization | continuous relaxation→binary code 학습 계보 |
| [HashNet, ICCV 2017](https://openaccess.thecvf.com/content_iccv_2017/html/Cao_HashNet_Deep_Learning_ICCV_2017_paper.html) | continuation으로 discrete sign 최적화, imbalance 처리 | binary discreteness 문제 설명 |
| [GreedyHash, NeurIPS 2018](https://papers.nips.cc/paper_files/paper/2018/hash/13f3cf8c531952d72e5847c4183e6910-Abstract.html) | sign forward와 straight-through gradient | 본 연구 Gumbel-ST와 구별되는 binary 학습 계보 |
| [MLS3RDUH, IJCAI 2020](https://www.ijcai.org/proceedings/2020/479) | manifold local similarity reconstruction | 현재 unsupervised baseline; noisy neighbor 구조 처리 |
| [CIBHash, IJCAI 2021](https://www.ijcai.org/proceedings/2021/133) | contrastive information bottleneck | 현재 가장 강한 baseline 중 하나; paired-view contrastive loss의 출발점 |
| [CIMON, IJCAI 2021](https://www.ijcai.org/proceedings/2021/125) | refined similarity와 semantic/contrastive consistency | 현재 baseline; disturbance robustness |
| [Bi-half, AAAI 2021](https://ojs.aaai.org/index.php/AAAI/article/view/16296) | balanced binary activation | bit-balance 계열 보조 인용 |
| [CSQ, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Yuan_Central_Similarity_Quantization_for_Efficient_Image_and_Video_Retrieval_CVPR_2020_paper.html) | global hash centers | flat center-based code와 slot codebook 대비 |
| [DHD, ECCV 2022](https://arxiv.org/abs/2112.08816) | self-distillation과 proxy/hash loss | 대표적인 strong self-distillation/proxy baseline 계보 |
| [AMVH, CVPR 2017](https://openaccess.thecvf.com/content_cvpr_2017/papers/Da_AMVH_Asymmetric_Multi-Valued_CVPR_2017_paper.pdf) | asymmetric multi-valued hashing | non-binary hashing 선행; 4진 alphabet 최초 주장 방지 |
| [Yu et al., SKG 2019](https://doi.org/10.1109/SKG49510.2019.00018) | four-valued/quaternary image hash | DNA의 4-symbol 형식 자체는 novelty가 아님 |
| [Instance-Aware Hashing](https://arxiv.org/abs/1603.03234) | category별 code pieces로 multi-label image 표현 | compositional sub-code 선행 인정 |
| [Dual Purpose Hashing, CVPR 2017](https://openaccess.thecvf.com/content_cvpr_2017/html/Liu_Learning_Multifunctional_Binary_CVPR_2017_paper.html) | category·attribute retrieval과 code-to-attribute reconstruction | interpretable attribute sub-code의 직접 선행 |
| [A²-Net, NeurIPS 2021](https://proceedings.neurips.cc/paper_files/paper/2021/hash/2d3acd3e240c61820625fff66a19938f-Abstract.html) | attribute annotation 없이 hash–visual attribute 대응 학습 | ConceptHash 이전의 attribute-level interpretability 계보 |
| [Multiple Code Hashing](https://arxiv.org/abs/2008.01503) | region별 multiple hash codes | multiple/local code 최초 주장 방지 |
| [DSCH](https://arxiv.org/abs/2203.09420) | latent semantic components와 hierarchy | latent compositional hashing 대비 |

- 작성 방향
  - “기존 hashing은 모두 flat” 대신 “global flat code가 지배적이지만 category/region/concept sub-code 계열이 확장 중”으로 서술
  - GroundedDNA의 차이: fixed semantic axes + spatial UOT grounding + independent VQ + codon/DNA feasibility

#### 2.2.1 2023–2026 최신 uni-modal deep binary hashing audit

- repository-local supervision/information-condition 판정(“unsupervised” 자체 표기보다 우선하며, 원 논문이 사용하는 공식 명칭으로 주장하지 않음)
  - `VLM-T text-supervised, target-label-free objective`: target ground-truth label·benchmark taxonomy는 representation objective에 미사용하지만 VLM 생성 instance caption과 frozen text encoder를 supervision으로 사용; GroundedDNA의 tier
  - `U0 visual-only`: 학습 objective에 instance label, class name, caption, external text bank 모두 미사용
  - `U0-FD feature-distance pseudo-pair`: `U0`의 native-DNA subtag; frozen optimization-train visual-feature distance로 pair target을 구성
  - `U1 external-knowledge`: 학습 objective에 instance label·benchmark taxonomy는 미사용, WordNet/CLIP과 같은 범용 외부 지식 사용
  - `U2 taxonomy-assisted`: 학습 objective에 instance label은 미사용이지만 평가 benchmark의 class-name/concept taxonomy 사용
  - `U? source-unverified`: 외부 term source 또는 selection provenance를 검증할 수 없어 `U1`을 주장할 수 없는 상태; target taxonomy semantic-set match는 즉시 `U2`로 승격
  - `S / SEMI / WEAK`: 각각 ground-truth label / 일부 label / tag·weak annotation을 사용; `S`는 Koike·CRH 표기와 동일하며 main target-label-free 표와 분리
  - 모든 tier의 P0 모델 선택에는 공통으로 held-out train label을 **평가에만** 사용하므로, tier는 representation objective의 정보 조건이고 전체 protocol이 label-blind라는 뜻은 아님
- 구현 상태 표기
  - `I — IMPLEMENTED`: 논문 수식과 공개 코드 차이를 감사한 clean-room runner 구현; full 3-seed 수치는 아직 `-`
  - `B — BLOCKED`: 공개 자료로 exact implementation을 검증할 수 없음; 숫자를 만들지 않고 `-`
  - `P — PLANNED`: 논문상 수동 이식 가능성이 있으나 아직 검증된 runner가 없음
  - `C — CITE`: task·supervision·평가가 달라 related work/별도 oracle에서만 다룸
  - protocol-boundary suffix `M`: method core를 공통 frozen-cache에 이식한 matched adapter; published-table reproduction 아님
  - external-boundary suffix `E`: method core는 구현했으나 공개되지 않거나 모순된 외부 artifact 때문에 exact paper row는 blocked
- 공통 적용 조건
  - 논문의 32/64-bit 수치를 36-bit와 직접 비교하거나 보간하지 않음
  - 모든 후보를 정확히 36-bit로 재학습한 뒤 `2 bits↔1 base` 변환과 동일 DP projection을 적용
  - 동일 split·frozen CLIP backbone·36-bit capacity·18-base evaluator를 사용하되, 방법 정의에 필요한 global/local/two-view/text 입력은 숨기지 않고 열로 표시
  - released-backbone official reproduction과 frozen-CLIP matched adaptation을 혼합하지 않음
  - checkpoint·cache metadata·split·implementation·semantic asset를 SHA-256 run fingerprint로 고정하고, 기존 trial directory 재사용을 거부

| 연구 | 연도·상태 | 정보 조건과 핵심 | 공개 구현·공통 dataset | 비교 판단 |
|---|---|---|---|---|
| [Similarity Distribution Calibration (SDC)](https://papers.bmvc2023.org/0053.pdf) | BMVC 2023 Oral | `U0`; pairwise hash-similarity 분포를 beta target quantile에 calibration | [code](https://github.com/kamwoh/sdc); paper/release의 head·SDC 적용 view·WD·target clipping 차이 존재 | `I/M`; `release_no_cl`, `release_simclr`, `paper_cache2v`를 서로 섞지 않고 별도 구현·보고 |
| [Deep Debiased Contrastive Hashing (DDCH)](https://doi.org/10.1016/j.patcog.2023.109483) | Pattern Recognition 2023, 139:109483 | `U0`; contrastive hashing에서 false-negative/semantic bias를 완화 | 검증 가능한 공식 code 미확인; CIFAR/NUS/COCO | `B`; 최신 visual-only 계보의 필수 인용, 저자 구현 또는 독립 equation audit 전 수치는 `-` |
| [UHSCM](https://arxiv.org/abs/2209.11475) | Proc. ACM Manag. Data 2023 | `U2`; VLP prompt concept를 mining·denoising하여 semantic similarity 구성 | [code](https://github.com/rongchengtu1/UHSCM); benchmark taxonomy file 사용, 공개 train path의 미정의 변수/스케줄 오류 | `B`; 엄격한 visual-only가 아니며 public train path repair 전 exact runner 주장 보류 |
| [Hashing One With All / Overview Hashing (OH)](https://doi.org/10.1145/3581783.3611977) | ACM MM 2023 | `U0`; binary-head Hamming affinity로 continuous memory를 집계하고 EMA key encoder·세 queue·두 contrastive loss로 학습 | [code](https://github.com/RosieYuu/OH); TensorFlow/CIFAR-10/online ResNet-50 release | `I/M`; 공개 실행 core를 fixed two-view CLIP cache와 36-bit에 이식한 clean-room adapter, 원 table 재현과 분리 |
| [CGHash](https://doi.org/10.1145/3581783.3612596) | ACM MM 2023 | `U0`; 비지도 contrastive similarity knowledge와 hidden structure | [code](https://github.com/KARLSZP/CGHash); 실질적 CIFAR path, 외부 SimCLR checkpoint 및 실행 defect | `B`; COCO/NUS loader·config와 검증된 external checkpoint가 없어 exact 공통 runner 수치 생성 금지 |
| [MDSHC](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_Deep_Hashing_With_Minimal-Distance-Separated_Hash_Centers_CVPR_2023_paper.html) | CVPR 2023 | 감독; 최소거리 보장이 있는 class hash center | 논문 기반 이식 가능 | `C — CITE`; supervised oracle 후보, label-free main 표와 분리 |
| [HHCH](https://pubmed.ncbi.nlm.nih.gov/38442063/) | IEEE TIP 2024 | `U0`; hyperbolic hierarchy와 instance/prototype contrast | [code](https://github.com/HUST-IDSM-AI/HHCH); paper/release의 K-means·quantization·temperature·scheduler 차이 존재 | `I/M`; 논문 수식 `paper_cache3v`를 명시적 구현 |
| [CTMIH](https://www.ijcai.org/proceedings/2024/0135) | IJCAI 2024 | `U0`; degraded-image retrieval을 위한 transformed/masked ViT, cross-view debiased contrast와 patch-level semantic mask reconstruction | 검증 가능한 공식 code 미확인; raw patch reconstruction이 method-defining | `B`; frozen global cache head로 바꾸면 CTMIH가 아니므로 실행 상태는 blocked, Related Works에만 인용하며 저자 구현 확보 전 exact 수치 `-` |
| [HiHPQ](https://ojs.aaai.org/index.php/AAAI/article/view/28261) | AAAI 2024 | `U0`; hyperbolic product codebook와 hierarchical contrastive quantization | PQ의 asymmetric query–codeword 실수 distance를 사용 | `C`; binary/Hamming main table이 아닌 non-binary/compositional quantization 선행 및 별도 PQ protocol 후보 |
| [FSCH](https://dblp.org/rec/journals/tcsv/CaoHNW24) | IEEE TCSVT 2024 | `U0` 주장; regional common feature와 global–local fine-grained similarity | [code](https://github.com/huanglab-research/FSCH); trainer·loss·optimizer·entry point 누락, 공개 model의 shape/constructor defect | `B — BLOCKED`; 저자 artifact 없이 exact 수치 생성 금지 |
| [A²-SSL](https://openaccess.thecvf.com/content/CVPR2024/html/Hu_An_Asymmetric_Augmented_Self-Supervised_Learning_Method_for_Unsupervised_Fine-Grained_Image_CVPR_2024_paper.html) | CVPR 2024 | 비지도 fine-grained; asymmetric augmentation, part-oriented dense contrast, self-consistent hash | 공식 code 미확인; CUB/Flowers/Dogs/Cars/Food101 | `C — CITE`; CUB/local-part 계보의 필수 직접 선행 |
| [HQT](https://bmvc2024.org/proceedings/482/) | BMVC 2024 | hierarchical binary clustering/quantization tree를 기존 hasher에 부착 | [code](https://github.com/Lab-LVM/HQT); 실질적으로 CIFAR 중심 | `P`; 독립 method 행이 아니라 `HQT+base` 보조 실험 |
| [BRCD](https://arxiv.org/abs/2403.06071) | WWW 2024 | bit-mask robust contrastive distillation; base hasher용 teacher/student add-on | [code](https://github.com/hly1998/BRCD); CIFAR/COCO/ImageNet100 | `P`; standalone baseline이 아니라 효율·강건성 보조 |
| [LGH](https://icmr2024.org/virtual-posters/168_outline.html) | ICMR 2024 | language model에서 high-level concept와 similarity를 채굴하는 image-only retrieval hash | 공식 code 미확인; CIFAR/NUS/COCO | `P`; 필수 인용, 실행 우선순위는 DUH-EG보다 낮음 |
| [HARR](https://doi.org/10.1145/3627162) | ACM TOMM 2024, 20(5) | 비지도 Winner-Take-All similarity, bit-correlation reduction, cosine quantization | 공식 code 미확인 | `C — CITE`; manual 재구현 우선순위 낮음 |
| [UDHPM](https://doi.org/10.1109/LSP.2024.3379085) | IEEE SPL 2024 | 비지도 dynamic soft-clustering pseudo-multilabel과 KL 학습 | 공식 code 미확인 | `C — CITE`; manual 재구현 우선순위 낮음 |
| [DUH-EG](https://proceedings.mlr.press/v267/song25h.html) | ICML 2025 | intended `U1`; WordNet+CLIP external guidance와 multi-positive contrastive hashing. 현재 arbitrary bank는 `U?`, target-taxonomy semantic match는 `U2` | [code](https://github.com/XLearning-SCU/2025-ICML-DUHEG); 네 dataset, selected-noun bank이 외부 artifact | `I/E`; released objective 구현, official ordered bank/provenance 부재로 exact row blocked; U0 표와 분리 |
| [RCSH](https://doi.org/10.1109/TNNLS.2023.3333294) | IEEE TNNLS 2025, online 2023 | self-supervised; data–prototype/data–data relational consistency | [claimed code](https://github.com/IMAG-LuJin/RCSH)는 README-only | `B`; 현재 공식 repo만으로 exact 실행 불가 |
| [MambaHash](https://doi.org/10.1145/3731715.3733380) | ICMR 2025 | 감독; visual state-space hashing | [code](https://github.com/shuaichaochao/MambaHash); CIFAR/NUS/ImageNet | `C`; supervised oracle 후보, label-free main 표와 분리 |
| [GDSH](https://ojs.aaai.org/index.php/AAAI/article/view/32600) | AAAI 2025 | semi-supervised; label completion과 debiasing | 공식 code 미확인; CIFAR/COCO/NUS/Flickr 등 | `C — CITE`; label-free main protocol과 정보 조건 불일치 |
| [VPDS](https://proceedings.neurips.cc/paper_files/paper/2025/hash/c23ccf9eedf87e4380e92b75b24955bb-Abstract-Conference.html) | NeurIPS 2025 | VLM pseudo-label을 쓰는 unsupervised domain-adaptive hashing | 안정적 공식 repo 미확인; Office 계열/MNIST↔USPS | `C — CITE`; VLM novelty 방어, task는 domain adaptation |
| [Few-shot Prompt Learning for Image Deep Hashing](https://doi.org/10.1109/ICME59968.2025.11209312) | ICME 2025 | few-shot supervised VLM prompt adaptation | 공식 code 미확인 | `C — CITE`; KALAHash와 함께 prompt/VLM-guided 계보 |
| [Factorized Transformer Hashing with Adaptive Routing](https://doi.org/10.1145/3746027.3755201) | ACM MM 2025 | 감독; factorized transformer와 learnable selector/router | [code](https://github.com/QinLab-WFU/FTH); Flickr/NUS/COCO, 현재 Flickr 중심 script·36-bit 미지원 | `C`; supervised 별도 표 후보, GroundedDNA의 OT router와 목적·정보 조건을 구분 |
| [KALAHash](https://ojs.aaai.org/index.php/AAAI/article/view/33136) | AAAI 2025 | knowledge-anchored low-resource supervised adaptation | 공식 논문 기준 재현 가능 | `C — CITE`; label budget이 다른 별도 oracle |
| [DGOH](https://ojs.aaai.org/index.php/AAAI/article/view/32191) | AAAI 2025 | supervised online/streaming multi-label hashing | 공식 논문 | `C — CITE`; offline fixed-database main protocol에서 제외 |
| [DBHE](https://doi.org/10.1016/j.neucom.2025.131411) | Neurocomputing 2025, 655:131411 | 감독; ViT와 Poincaré pairwise objective | [code](https://github.com/QinLab-WFU/DBHE); Flickr/NUS/COCO | `C`; supervised upper-bound 후보 |
| [UMRCH](https://www.sciencedirect.com/science/article/pii/S0957417425040291) | ESWA 2026, 301:130414 | `U2`; benchmark class-name CLIP probability와 ViT global/local multi-semantic reconstruction | [code](https://github.com/Lab201A/UMRCH); Flickr/NUS/COCO | `I/M`; exact objective + official taxonomy + matched CLIP local-token adapter, taxonomy-assisted 보조 표에 보고 |
| [TGUH](https://doi.org/10.1109/TMM.2026.3673565) | IEEE TMM 2026 | 생성/서술 text-guided multimodal fusion, dynamic community, hash center | [code](https://github.com/caoyuan618/TGUH) 공개 표기 | `B`; 공개 trainer의 hard-coded/incomplete 경로와 test-selection을 해소하기 전 exact 실행 금지 |
| [Multi-scale aggregation + OT hashing](https://www.sciencedirect.com/science/article/pii/S092523122503262X) | Neurocomputing 2026, 671:132590 | 비지도; multi-scale region과 augmented-view dense correspondence를 OT로 정렬 | 공식 code 미확인; CIFAR/COCO/ImageNet | `P`; “OT hashing 최초” 주장 방지, 목적이 router UOT와 다름 |
| [OMUH](https://www.sciencedirect.com/science/article/pii/S0893608026003084) | Neural Networks 2026, 200:108846 | `U1`; 외부 Faster R-CNN object pseudo-label과 multi-granularity affinity | [code](https://github.com/caoyuan618/OMUH); VOC/Flickr 중심, pseudo-label row alignment·test-best selection·hard-coded path 문제 | `B`; visual-only `U0`로 분류하지 않으며 공개 protocol defect 해소 전 exact 수치 금지 |
| [CroVCA/HashCoder](https://openaccess.thecvf.com/content/CVPR2026W/ECV/html/Moummad_Image_Hashing_via_Cross-View_Code_Alignment_in_the_Age_of_CVPRW_2026_paper.html) | CVPRW 2026 | `U0`; foundation embedding의 cross-view stop-gradient BCE와 coding-rate maximization | [code](https://github.com/ilyassmoummad/cross-view-code-alignment); 공개 SimDINO/DFN loader defect와 paper/code coding-rate 표기 차이 존재 | `I/M`; official objective의 frozen matched-cache probing 구현, paper 주 결과의 LoRA/asymmetric-Hamming reproduction과 분리 |
| [FAPI](https://doi.org/10.1145/3786797) | ACM TOMM 2026, 22(3):89 | 비지도 fine-grained; feature augmentation, cross-contrastive learning, progressive granularity fusion | 공식 code 미확인; CUB/Flowers/Dogs/Cars/Food101 | `C — CITE`; 공통 4-dataset과 불일치하지만 CUB 관련 필수 |
| [CS3H](https://arxiv.org/abs/2605.18288) | ICIP 2026 accepted preprint | 비지도 fine-grained; normalized Hamming 직접 최적화와 collision-sensitive attention | 공식 code 미확인; fine-grained 5종+NUS | `P`; collision 분석과 가까우나 proceedings/코드 확인 필요 |
| [MKDH](https://doi.org/10.1109/TMM.2026.3651086) | IEEE TMM 2026 | weakly supervised tag+CLIP multi-knowledge distillation | [code](https://github.com/IMAG-LZY/MKDH); Flickr/NUS/COCO | `B`; 공개 script의 미정의 인자·절대경로 수정 필요, label-free 표와 분리 |
| [Deep Semantic Channel Hashing (DSCH-2026)](https://doi.org/10.1016/j.image.2026.117559) | SPIC 2026, 146:117559 | 감독; label similarity별 differentiated hash constraint | [code](https://github.com/QinLab-WFU/DSCH); Flickr/NUS/COCO | `C`; supervised upper-bound 후보 |
| [DGrH](https://doi.org/10.1016/j.eswa.2026.131557) | ESWA 2026 | 감독; 최신 generic deep hashing | [code](https://github.com/QinLab-WFU/DGrH); Flickr/NUS/COCO | `C`; supervised upper-bound 후보 |
| [AHIR](https://doi.org/10.1016/j.neucom.2026.132639) | Neurocomputing 2026, 671:132639 | 감독; ResNet50+autoencoder hashing | 공식 code 미확인; Flickr/NUS/COCO | `C — CITE`; 실행보다 최신 서지 보강용 |
| [DNCPH](https://doi.org/10.1016/j.eswa.2026.131924) | ESWA 2026, 317:131924 | 감독; neighborhood-component proxy와 triplet upper bound | [code](https://github.com/QinLab-WFU/DNCPH); Flickr/NUS/COCO | `C`; supervised upper-bound 후보 |
| [CRH](https://ojs.aaai.org/index.php/AAAI/article/view/38190) | AAAI 2026 | 감독; dynamic semantic-center reassignment | [code](https://github.com/iFamilyi/CRH); 공통은 COCO 중심 | `C — CITE`; supervised oracle/analysis 전용 |

- 추가 2025–2026 bibliography screen (`C — CITE`, main 실행 전 protocol/code 재감사)
  - [Graph Hashing Network](https://doi.org/10.1016/j.imavis.2025.105677), [DCPH](https://doi.org/10.1016/j.neucom.2025.130014), [LECH](https://doi.org/10.1016/j.ipm.2025.104277), [CTSAH](https://doi.org/10.1016/j.asoc.2025.112752), [CAPMH](https://doi.org/10.1016/j.eswa.2025.128228), [Deep Global Distance Estimation Hashing](https://doi.org/10.1109/TBDATA.2025.3566532)
  - [DRKDH](https://doi.org/10.1145/3820061): sparse/noisy-label supervised, [code](https://github.com/QinLab-WFU/DRKDH)
  - [DQAH](https://doi.org/10.1016/j.dsp.2026.106092): supervised, Flickr/NUS/COCO, 공식 code 미확인
  - FAPI의 개별 논문 DOI는 `10.1145/3786797`; `10.1145/3762853`은 TOMM 22(3) issue DOI이며, 기존 DFMH는 2022 JVCIR의 별도 방법임
- 실행 shortlist
  - `U0` visual-only main: `UGH(GreedyHash)-cache`, `Bi-half-cache`, `SDC-release-noCL`, `SDC-release-SimCLR`, `SDC-paper-cache2v`, `OH-cache2v`, `HHCH-paper-cache3v`, `CroVCA-cache2v-probe`
  - external-knowledge 별도: `DUH-EG`의 현재 released-objective adapter는 `U?`; 공식 ordered selected-noun bank/provenance 확보 시에만 `U1`, target-taxonomy semantic match면 `U2`. OMUH는 공개 protocol defect로 실행 `B`
  - `U2` taxonomy-assisted 별도: `UMRCH`; `UHSCM`은 공개-code defect 해소 전 수치 보류
  - `U0` non-binary 경계: HiHPQ는 `C`; asymmetric PQ distance 전용 protocol 없이 binary/Hamming main table에 포함하지 않음
  - exact 실행 blocked: `DDCH`, `CGHash`, `CTMIH`, `FSCH`; 저자 trainer/loss/checkpoint 또는 method-defining raw-image pipeline 확보 전까지 `-`
  - text-generated multimodal: `TGUH`(code completeness 감사 후)
  - CUB/fine-grained 별도 track: `A²-SSL`, `FAPI`, `CS3H`
  - 여력이 있을 때: multi-scale OT hashing, `RCSH` 수동 이식, `HQT+base`, `BRCD+base`
  - 별도 supervised upper-bound: `DNCPH`, `DBHE`, `DSCH-2026`, `DGrH`, `MDSHC/CRH`, `MambaHash`
- novelty 문장 수정
  - “최초의 VLM/text-guided hashing”, “최초의 local semantic hashing”, “최초의 OT hashing”은 사용하지 않음
  - 안전한 구분점: direct quaternary DNA learning + role-typed selective UOT + EMA multi-codebook/codon composition + exact biological projection의 결합

### 2.3 Product/compositional quantization과 discrete representation

| 연구 | 핵심 내용 | 본 논문과의 연결 |
|---|---|---|
| [Product Quantization](https://pubmed.ncbi.nlm.nih.gov/21088323/) | vector subspace별 codebook의 Cartesian product | multi-codebook 조합 용량의 고전적 근거 |
| [Cartesian k-means, CVPR 2013](https://openaccess.thecvf.com/content_cvpr_2013/papers/Norouzi_Cartesian_K-Means_2013_CVPR_paper.pdf) | Cartesian product codebook 학습 | 독립 subspace quantization 계보 |
| [Additive Quantization, CVPR 2014](https://openaccess.thecvf.com/content_cvpr_2014/html/Babenko_Additive_Quantization_for_2014_CVPR_paper.html) | 여러 codeword의 합으로 근사 | additive composition과 slot-typed composition 구별 |
| [Composite Quantization, ICML 2014](https://proceedings.mlr.press/v32/zhangd14.html) | near-orthogonal composite codebooks | codebook interaction 관련 근거 |
| [SUBIC, ICCV 2017](https://openaccess.thecvf.com/content_ICCV_2017/papers/Jain_SUBIC_A_Supervised_ICCV_2017_paper.pdf) | concatenated one-hot blocks와 entropy | block-structured discrete code의 직접 선행 |
| [VQ-VAE, NeurIPS 2017](https://arxiv.org/pdf/1711.00937) | vector quantization, STE, Appendix의 EMA update | EMA 수식의 직접 출처; 원 논문 본 실험에서는 EMA 미사용임을 정확히 표기 |
| [Product Quantization Network, ECCV 2018](https://openaccess.thecvf.com/content_ECCV_2018/html/Tan_Yu_Product_Quantization_Network_ECCV_2018_paper.html) | end-to-end product quantization | learned retrieval quantizer 계보 |
| [DPQ, CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/papers/Klein_End-To-End_Supervised_Product_Quantization_for_Image_Search_and_Retrieval_CVPR_2019_paper.pdf) | soft/hard product quantization 공동학습 | hard deployment와 soft training 대비 |
| [Uni-Code, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/c89f09849eb5af489abb122394ff0f0b-Abstract-Conference.html) | paired modality를 shared discrete latent space에 맞추는 cross-modal commitment와 MM-EMA | champion XM loss의 직접 출처; 본 구현은 Eq. (8)형 commitment만 차용하고 full DCID/MM-EMA는 사용하지 않음 |

- 본 논문의 구분점
  - 전통 PQ: dimension partition이 중심
  - GroundedDNA: 각 codebook이 사전에 정의된 semantic role을 담당하도록 text/UOT로 정초
  - 조합 용량 (K^6)과 최종 codon 용량 (64^6)을 혼동하지 않도록 별도 보고

### 2.4 Text-supervised visual representation과 interpretable hashing

| 연구 | 핵심 내용 | 본 논문에서의 위치 |
|---|---|---|
| [CLIP, ICML 2021](https://arxiv.org/abs/2103.00020) | 대규모 image–text contrastive pretraining | frozen image/text feature teacher |
| [ALIGN, ICML 2021](https://arxiv.org/abs/2102.05918) | noisy web-scale image–text supervision | language-supervised visual representation 배경 |
| [SigLIP, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhai_Sigmoid_Loss_for_Language_Image_Pre-Training_ICCV_2023_paper.pdf) | pairwise sigmoid image–text objective | repository의 대안 backbone 계보; current P0가 CLIP임을 구분 |
| [SigLIP2, 2025](https://arxiv.org/abs/2502.14786) | captioning·self-distillation·localization을 결합한 multilingual VLM | future backbone ablation; 파일명과 champion backbone 혼동 방지 |
| [VirTex, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Desai_VirTex_Learning_Visual_Representations_From_Textual_Annotations_CVPR_2021_paper.html) | caption generation을 dense visual supervision으로 활용 | caption이 label보다 풍부한 신호라는 근거 |
| [FILIP, ICLR 2022](https://openreview.net/pdf?id=cpDhcsEDC2) | token-wise late interaction | patch/token fine-grained alignment 계보 |
| [RegionCLIP, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Zhong_RegionCLIP_Region-Based_Language-Image_Pretraining_CVPR_2022_paper) | region–text alignment | local semantic grounding 근거 |
| [GroupViT, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Xu_GroupViT_Semantic_Segmentation_Emerges_From_Text_Supervision_CVPR_2022_paper.pdf) | image-level text에서 semantic group 발견 | dense annotation 없는 visual grouping 근거 |
| [WDHT, CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Gattupalli_Weakly_Supervised_Deep_Image_Hashing_Through_Tag_Embeddings_CVPR_2019_paper.html) | tag embedding으로 image-only hash 감독 | text-supervised unimodal hashing의 직접 선행 |
| [SCADH, IEEE TIP 29:1271–1284, 2020 (online 2019)](https://doi.org/10.1109/TIP.2019.2940693) | refined social-tag semantics로 image hash 학습 | accompanying text를 privileged supervision으로 쓰는 직접 선행 |
| [Tag-based Weakly-supervised Hashing, IJCAI 2018](https://www.ijcai.org/proceedings/2018/525) | noisy tag와 latent semantic vector 공동학습 | label-free가 text-free는 아님을 설명 |
| [DSRPH, Information Sciences 2020](https://doi.org/10.1016/j.ins.2020.05.114) | caption embedding으로 semantic similarity/ranking 감독 | caption-supervised hashing의 직접 선행 |
| [ConceptHash, CVPRW 2024](https://arxiv.org/abs/2406.08457) | concept token별 interpretable binary sub-code, language class center | 가장 가까운 해석 가능 hashing 선행; 반드시 직접 비교 |
| [Attribute-Aware Hashing, ICML 2025](https://proceedings.mlr.press/v267/wang25bj.html) | query를 visual attribute와 hash code에 대응 | attribute-level interpretability 선행 |
| [LPAH, PRCV 2025; LNCS 16283, 2026, pp. 63–77](https://github.com/zhenglab/LPAH) | synthetic text로 patch aggregation·patch-word alignment | synthetic-caption-guided patch hashing 선행 |
| [HTH, IEEE TCSVT 2026](https://doi.org/10.1109/TCSVT.2026.3651707) | hierarchical global/local text-guided open-world hashing | 최신 text-guided binary hashing 비교 |
| [Interpretable Binary Codes, Pattern Recognition 172:112380, 2026](https://doi.org/10.1016/j.patcog.2025.112380) | semantic alignment로 bit-level customized retrieval | interpretable bit selection 선행 |

- ConceptHash와의 핵심 차이 문장
  - ConceptHash: fine-grained class label/class-name guidance, learned concept token, binary sub-code
  - GroundedDNA: instance-level VLM captions, fixed six axes, patch-to-slot UOT, EMA VQ, codon/DNA projection, image-only inference
- cross-modal hashing의 처리
  - 관련 문단에서만 경계 설명
  - image↔text retrieval 수치를 image→image main table에 혼합하지 않음

### 2.5 DNA storage, molecular similarity search, codon-inspired CBIR

| 연구 | 핵심 내용 | 본 논문에서 사용할 요점 |
|---|---|---|
| [Church et al., Science 2012](https://doi.org/10.1126/science.1226355) | digital information의 DNA encoding | DNA storage 서사의 출발점 |
| [Goldman et al., Nature 2013](https://www.nature.com/articles/nature11875) | scalable archival encoding과 error correction | density·durability·archive 맥락 |
| [DNA Fountain, Science 2017](https://doi.org/10.1126/science.aaj2038) | near-capacity robust DNA storage | storage codec 계보 |
| [Organick et al., Nature Biotechnology 2018](https://doi.org/10.1038/nbt.4079) | large-scale random access | address-based access와 semantic access 대비 |
| [Stewart et al., DNA24 2018](https://www.microsoft.com/en-us/research/wp-content/uploads/2018/08/dna24.pdf) | learned content-addressable DNA database의 전신; 공식 PDF의 제1저자는 Kendall Stewart | 직접 선행의 초기 형태; “Bee et al. 2018”로 잘못 표기하지 않음 |
| [Bee et al., Nature Communications 2021](https://www.nature.com/articles/s41467-021-24991-z) | VGG feature→80-base sequence, hybridization-aware 1.6M image search | molecular similarity retrieval의 핵심 직접 선행; [code](https://github.com/uwmisl/primo-similarity-search) |
| [Koike et al., DATE 2024](https://past.date-conference.com/proceedings-archive/2024/DATA/478_pdf_upload.pdf) | triplet network DNA encoder, CIFAR-100 | learned image-to-DNA metric encoding 직접 비교; [official code](https://github.com/tkoike-kuee/dna-triplet-network) |
| [Koike et al., DAC 2024](https://doi.org/10.1145/3649329.3657320) | DATE의 6-page 확장판; 동일 triplet-DNA core | DATE와 독립 baseline 두 개로 세지 않음 |
| [Koike et al., IEEE TCBB 2026](https://pubmed.ncbi.nlm.nih.gov/41824343/) | triplet encoder에 probability·GC·homopolymer loss와 inference heuristic 추가 | 가장 최신 직접 경쟁 연구; [official code](https://github.com/tkoike-kuee/DNA-Encoder-under-Bioconstraints) |
| [Cas9 semantic search, Nature Communications 2025](https://www.nature.com/articles/s41467-025-61264-5) | 20-nt target(+3-nt PAM); 1.74M images를 457 semantic cluster address에 매핑 | molecular random/semantic access의 최신 사례; [code](https://github.com/uwmisl/cas9-similarity-search) |
| [Pradhan et al., IEEE TNB 2023](https://pubmed.ncbi.nlm.nih.gov/35486561/) | DNA transcription/translation 기반 CBIR | codon·amino-acid CBIR 직접 선행 |
| [Pradhan et al., IEEE TNB 2024](https://doi.org/10.1109/TNB.2023.3303512) | pixel bit-plane DNA pattern + deep feature retrieval | handcrafted nucleotide-pattern 계열 |
| [DNA-CBIR, IEEE TNB 2025](https://pubmed.ncbi.nlm.nih.gov/40031697/) | 3-MSB→DNA plane→codon-pattern deep feature | “codon 최초” 주장 방지; learned semantic codon과 대비 |
| [HEDGES, PNAS 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7414044/) | indel/substitution 및 sequence constraint 고려 | physical-channel 제약의 더 넓은 범위 |
| [DNA-Aeon, Nature Communications 2023](https://www.nature.com/articles/s41467-023-36297-3) | user-defined GC, homopolymer, motif 제약 | 본 연구 40–60% GC/run≤3의 설계 근거 |
| [Capacity-Approaching Constrained Codes](https://arxiv.org/abs/2001.02839) | GC/RLL/error-correcting constrained code | DP projection의 coding-theory 맥락 |
| [DNA-ELMR, ESWA 2026](https://www.sciencedirect.com/science/article/pii/S0957417426006391) | image storage/reconstruction와 다중 생물 제약 | bio-loss 참고; retrieval baseline에서는 제외 |

- 직접 비교 프레임
  - Bee/Koike: 거리·hybridization과 biochemical validity 중심, flat DNA sequence
  - Pradhan: pixel/MSB에서 nucleotide/codon/amino-acid feature를 구성하는 handcrafted CBIR
  - GroundedDNA: language-defined role, learned local codebook, compact hashing benchmark, text-free deployment
- codon 표현의 한계
  - 자연 genetic code의 translation table을 사용하지 않음
  - “biological semantics”가 아니라 “3-base typed unit이라는 structural analogy”로 한정

#### 2.5.1 직접 DNA-image retrieval 연구의 code/재현 가능성 판단

| 연구 | 공개 구현 | 이 repository에서의 재현 판단 | main comparison 계획 |
|---|---|---|---|
| Stewart et al. DNA24 2018 | 공식 code 미공개 | 논문 수식 기반 독립 PyTorch 구현 완료: train-only PCA-10, shared pointwise encoder, soft cosine-Hamming, analytic yield BCE | **`U0-FD` unsupervised; target-label-free encoder objective**; `DNA24-30-original`과 `DNA24-18-matched` 분리; strict `-`, sealed 3-seed diagnostic 완료 |
| Bee et al. 2021 | [PRIMO code/data](https://github.com/uwmisl/primo-similarity-search), [Zenodo](https://doi.org/10.5281/zenodo.5090717) | clean-room encoder/local-match CNN과 공식 `pub` commit의 Keras predictor를 exact 변환·검증; full alternating refit은 외부 NUPACK oracle가 필요 | **`U0-FD` unsupervised; target-label-free encoder objective**; `PRIMO-80-original`, 완료된 diagnostic `PRIMO-18-frozen-predictor-length-transfer`, 향후 미완료 `PRIMO-18-calibrated`를 분리; strict `-`, sealed 3-seed diagnostic 완료 |
| Koike DATE/DAC 2024 | [official code](https://github.com/tkoike-kuee/dna-triplet-network) | masked-softmax normalized-Hamming과 semi-hard triplet의 독립 PyTorch 구현 완료 | **`S` supervised**; `Koike-TN-DNA`, 18-base adaptation; strict `-`, sealed 3-seed diagnostic 완료 |
| Koike TCBB 2026 | [official code](https://github.com/tkoike-kuee/DNA-Encoder-under-Bioconstraints) | entropy/probability/HP/GC loss와 HP inference heuristic의 독립 PyTorch 구현 완료 | **`S` supervised**; `Koike-BC-TN-DNA`, bio-aware 18-base adaptation; strict `-`, sealed 3-seed diagnostic 완료 |
| Cas9 semantic search 2025 | [similarity-search code](https://github.com/uwmisl/cas9-similarity-search), [random-access code](https://github.com/uwmisl/cas9-random-access) | computational address encoder는 재현 가능; Cas9 off-target cleavage와 wet-lab protocol은 현재 범위 밖 | Related Work/qualitative context, 직접 mAP baseline은 `-` |
| Pradhan 2023–2025 | 공식 code 미확인 | pixel MSB→DNA plane→codon/amino-acid feature algorithm은 논문 기반 구현 가능하나 learned compact hashing과 task가 크게 다름 | 소규모 원 dataset 재현 또는 method-boundary 설명, main table은 `-` |
| Binary→2-bit/base | repository 구현 존재 | 완전 재현 완료; native DNA learner가 아닌 naïve transcoding control | 현재 main controlled baseline |

- 완료된 matrix와 남은 재현 과제
  - 구현 entry point: `baseline/native_dna.py`, `scripts/train_native_dna_baseline.py`
  - PRIMO weight adapter: `scripts/convert_primo_predictor.py`
  - 완료: 네 방법 × 네 데이터셋 × 세 seed의 18-base sealed diagnostic matrix 48/48; launcher failure 0, strict-main eligible 0/48
  - 완료: official PRIMO predictor exact conversion
  - 남음: provenance-complete cache의 strict matrix 재실행
  - 남음: DNA24 original 30-nt 및 Koike original 80-base computational sanity reproduction
  - 남음: 새 18-mer thermodynamic yield로 PRIMO predictor를 재보정한 `PRIMO-18-calibrated`
  - wet-lab 요소가 필요한 Cas9/PRIMO physical assay는 본 논문 범위 밖으로 명시
  - 외부 repository는 모두 명시적 code license가 없으므로 source 복사가 아닌 clean-room 구현을 유지

### 2.6 OT, sparse/selective routing, concept grounding

| 연구 | 핵심 내용 | 사용 방식 |
|---|---|---|
| [Cuturi, NeurIPS 2013](https://proceedings.neurips.cc/paper_files/paper/2013/hash/af21d0c97db2e27e13572cbf59eb343d-Abstract.html) | entropy-regularized OT와 Sinkhorn | balanced entropic OT의 기초 |
| [Liero, Mielke, Savaré, Inventiones Mathematicae 2018](https://doi.org/10.1007/s00222-017-0759-8) | marginal deviation을 entropy/KL로 완화하는 optimal entropy-transport | KL-relaxed UOT objective의 이론적 기초 |
| [Chizat et al., Math. Comp. 2018](https://arxiv.org/abs/1607.05816) | KL-relaxed UOT generalized scaling | P0 UOT 수식의 직접 근거 |
| [Pham et al., ICML 2020](https://proceedings.mlr.press/v119/pham20a.html) | UOT Sinkhorn 분석 | 수렴·복잡도 보조 근거 |
| [Smooth and Sparse OT, AISTATS 2018](https://proceedings.mlr.press/v84/blondel18a.html) | entropy plan의 dense 문제와 sparse regularization | top-p 동기; 동일 알고리즘이라고 주장하지 않음 |
| [Sparsity-Constrained OT](https://arxiv.org/abs/2209.15466) | explicit cardinality constraint | sparse plan 이론 비교; 본 heuristic과 구분 |
| [UNITER, ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/7093_ECCV_2020_paper.php) | word–region OT alignment | image–text local alignment의 기반 선행 |
| [Graph Optimal Transport, ICML 2020](https://proceedings.mlr.press/v119/chen20e.html) | image object–sentence word의 구조적 OT | cross-modal graph alignment 계보 |
| [VoLTA, TMLR 2023](https://shramanpramanick.github.io/VoLTA/) | caption-only patch–token graph-OT alignment | caption-supervised local routing과 가까운 선행 |
| [Semantic Correspondence as OT, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Liu_Semantic_Correspondence_as_an_Optimal_Transport_Problem_CVPR_2020_paper.html) | visual correspondence를 OT로 정식화 | vision alignment 선행 |
| [Unbalanced Feature Transport, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Zhan_Unbalanced_Feature_Transport_for_Exemplar-Based_Image_Translation_CVPR_2021_paper.html) | feature alignment에서 UOT | visual UOT 응용 근거 |
| [UOT for Object Detection, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/De_Plaen_Unbalanced_Optimal_Transport_A_Unified_Framework_for_Object_Detection_CVPR_2023_paper.html) | matching과 discard를 UOT로 통합 | relaxed mass의 selection 해석 보조 |
| [FedOTP, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Li_Global_and_Local_Prompts_Cooperation_via_Optimal_Transport_for_Federated_CVPR_2024_paper.html) | prompt와 core patch의 relaxed OT alignment | patch selection 동기 |
| [DOT-CBM, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Xie_Discovering_Fine-Grained_Visual-Concept_Relations_by_Disentangled_Optimal_Transport_Concept_Bottleneck_CVPR_2025_paper.html) | patch↔text concept OT와 localization | 가장 가까운 concept-grounding 선행; classification CBM과 retrieval codon 차이 |
| [Selective Sinkhorn Routing, ICML 2026 AdaptFM Workshop; arXiv first posted 2025](https://arxiv.org/abs/2511.08972) | top-k Sinkhorn MoE routing, KL projection 정리 | top-k renormalization 증명 참고; task/solver 동일성 주장 금지 |
| [OMIT, 2026 preprint](https://arxiv.org/abs/2603.14349) | dustbin을 둔 optimal partial image–text matching | partial/dustbin OT와 본 UOT 구분 |
| [ConceptOT, CVPRW 2026](https://openreview.net/pdf?id=EU0tuTbrKn) | CLIP patch–concept low-rank UOT | concurrent closest work; DNA hashing·fixed six slots·bio projection 차이 |

- 용어 선택
  - 본문: **KL-relaxed entropic unbalanced optimal transport**
  - `PSOT`를 쓸 경우 프로젝트 variant 이름으로만 정의
  - fixed-mass partial OT의 `$P\mathbf 1\le a$`, `$P^\top\mathbf 1\le b$`, 총질량 제약을 만족한다고 주장하지 않음
- adaptive top-p의 위치
  - UOT 해 이후의 row-wise sparsification heuristic
  - 전체 sparsity-constrained UOT의 exact optimizer라고 주장하지 않음

## 3. Methodology

### 3.1 문제 정의

- label-free parameter optimization set
  - \(\mathcal D_{\rm tr}=\{x_i\}_{i=1}^{N}\)
  - ground-truth class/multi-label은 parameter optimization, caption 생성, codebook 초기화에 사용하지 않음
  - 단, 표준 hashing protocol의 held-out validation relevance로 \(E^*\)를 선택하고 test label로 공식 metric/probe를 계산
  - frozen VLM으로 train image별 six-axis description \(T_i=\{t_i^m\}_{m=0}^{5}\) 사전 생성
- 학습 목표
  - \(h_\theta:\mathcal X\rightarrow\mathcal A^{18}\), \(\mathcal A=\{A,C,G,T\}\)
  - \(h_\theta(x)=[c^0;c^1;\ldots;c^5]\), \(c^m\in\mathcal A^3\)
- 검색 거리
  - base-wise Hamming distance
  - \(d_{\rm DNA}(y,y')=\sum_{\ell=1}^{18}\mathbf1[y_\ell\ne y'_\ell]\)
  - raw alphabet capacity: \(4^{18}=2^{36}\)
- 평가 relevance
  - single-label: 같은 class
  - multi-label: 하나 이상의 label 공유
  - label은 held-out validation의 \(E^*\) 선택, official retrieval metric, interpretability probe에 사용
- 용량 구분
  - pre-codon tuple capacity: \(K^6\)
  - 3-base codon capacity/slot: \(4^3=64\)
  - \(K=128\)에서는 서로 다른 codeword의 codon collision이 구조적으로 불가피

### 3.2 전체 학습 프레임워크

```text
offline caches
train image ──Qwen3-VL──> 6 axis captions ──CLIP text──> pooled/token text features
split-union images ──frozen CLIP vision──> 196 patch tokens + global image feature

training
global feature ───────────────> global slot
patches + 5 local text anchors ─UOT + adaptive top-p─> 5 local slots
6 slots ─6 independent EMA VQ codebooks─> 6 codewords
6 codewords ─global-conditioned codon heads─> 6×3 bases

inference
image ─frozen CLIP─> global + patches
patches + 5 codebook-mean anchors ─UOT/top-p─> local slots
fixed VQ + argmax codon ─> raw 18-base code ─DP bio projection─> valid code
query/database ─base Hamming─> ranked retrieval
```

### 3.3 Offline caption 및 feature cache

#### 3.3.1 Six-axis caption 생성

- offline annotator
  - [`Qwen/Qwen3-VL-8B-Instruct`](https://github.com/QwenLM/Qwen3-VL)
  - bfloat16, `do_sample=False`, `max_new_tokens=384`, batch size 4
  - 학습 가능한 모듈 아님; retrieval inference에 포함되지 않음
- six axes
  - \(m=0\): global summary
  - \(m=1\): primary object
  - \(m=2\): secondary object/cue
  - \(m=3\): activity/relation
  - \(m=4\): color/texture
  - \(m=5\): scene/background
- 데이터셋별 cache provenance

| Dataset | Caption rows | Prompt/cache | 논문에 적을 상태 |
|---|---:|---|---|
| Flickr25K | 5,000 | Qwen3-VL-8B, V4 | model/prompt/generation config 확인됨 |
| MS-COCO | 10,000 | Qwen3-VL-8B, vocabulary-constrained V5b | 16 decode failure와 fallback 처리 공개 필요 |
| NUS-WIDE | 10,500 | Qwen3-VL-8B, V4 | model/prompt/generation config 확인됨 |
| CIFAR-10 | 6,097 | legacy V1, old head/body/limb schema remap | 생성 script/model revision metadata 미확인; 재생성 또는 artifact 보완 필요 |

- leakage 규칙
  - visual feature cache: train/query/database split union을 path 기준 deduplicate하여 구축; 이는 frozen feature의 계산 cache이며 supervision leakage가 아님
  - structured caption supervision 대상: designated train rows
  - stage-1 whitening: optimization-train rows만 사용
  - stage-2 refit whitening: full designated train rows만 사용
  - query/database caption을 inference input으로 사용하지 않음
- missing-caption 구현 위험
  - `cached_has_text`가 sample-wise criterion mask까지 전달되지 않음
  - mixed batch의 fallback row가 text routing/loss에 들어갈 수 있음
  - 최종 제출 전 mask 수정 + 영향 ablation 필요

#### 3.3.2 Frozen CLIP encoder의 입력과 출력

- 실제 champion checkpoint: `openai/clip-vit-base-patch16`; result/script 이름의 `siglip2`와 구분

| 경로 | 입력 | frozen encoder 출력 | 학습 모듈 입력/출력 |
|---|---|---|---|
| Image global | 224×224 normalized RGB | projected image \(g_i\in\mathbb R^{512}\) | Linear \(512\to768\), global slot \(z_i^0\) |
| Image local | 같은 image/augmentation | final hidden \([197,768]\), CLS 제외 \(V_i\in\mathbb R^{196\times768}\) | visual MLP \(768\to1536\to768\) |
| Text pooled | six captions, tokenizer max length 64 | cached \([6,512]\) | P0 global slot의 pooled CLIP feature |
| Text token | six captions | cached \([6,32,512]\) + token mask | P0 local 5 slots의 legacy selected-token mean |

- 실제 P0 text assembly
  - global slot: caption 0의 pooled CLIP feature \([B,1,512]\)
  - local slots: caption 1–5의 token feature에서 legacy mask로 선택된 token mean \([B,5,512]\)
  - 두 경로를 \([B,6,512]\)로 결합한 뒤 train-only partial whitening과 slot별 독립 MLP \(512\to1536\to768\) 적용

- paired-view training cache
  - random resized crop, horizontal flip, color jitter, grayscale augmentation의 두 visual views
  - 두 view feature는 offline 한 번 생성·저장되어 epoch마다 새로 sampling되지 않음
  - visual patch/global과 text cache는 fp16 저장 후 row loading 시 fp32 계산
- backbone 동결
  - vision tower와 text tower 모두 gradient 없음
  - cache training에서는 backbone forward 자체를 생략
- partial whitening
  - train-only text covariance \(\Sigma=U\operatorname{diag}(s)U^\top\)
  - \(W_\gamma=U\operatorname{diag}((s+10^{-5})^{-\gamma})U^\top\), \(\gamma=.25\)
  - \(\tilde t=(t-\mu)W_\gamma\)
- P0 legacy pooling 사실
  - Flickr/NUS/CIFAR visual/text keep ratio `.5/.5`; MS-COCO `1/1`
  - legacy score가 `softmax(...).sum(same axis)`로 상수화
  - tied score에 `>= kth`를 적용하므로 `.5` 요청이 정확히 50% 유지됨을 보장하지 않으며 모든 token이 남을 수도 있음
  - 따라서 selected subset은 semantic importance가 아니라 tie/index behavior일 가능성
  - 논문 canonical clean EOS/full-token 경로의 P0 재실험 전까지 “semantic mutual pruning”을 방법 기여에서 제외

### 3.4 Text-guided UOT Router

#### 3.4.1 입력과 cost

- global slot \(m=0\)
  - CLIP projected global image feature에서 직접 생성
  - local OT 경쟁에 참여하지 않음
- local slots \(m=1,\ldots,5\)
  - visual token \(v_n\in\mathbb R^{768}\), \(n=1,\ldots,196\)
  - text centroid \(c_m\in\mathbb R^{768}\)
  - cosine cost

\[
C_{nm}=1-\frac{v_n^\top c_m}{\lVert v_n\rVert_2\lVert c_m\rVert_2}.
\]

- 주의
  - UOT에서는 총질량이 고정되지 않으므로 \(1-\cos\)와 \(-\cos\)의 상수차가 일반적으로 동치가 아님
  - 논문과 실험은 실제 P0의 `one_minus_cos`만 정의

#### 3.4.2 KL-relaxed entropic UOT

- prior mass
  - patch/slot validity mask를 각각 \(\mu_{in},\nu_{im}\in\{0,1\}\)라 두면

\[
a_{in}=\frac{\mu_{in}}{\sum_j\mu_{ij}},\qquad
b_{im}=\frac{\nu_{im}}{\sum_j\nu_{ij}}.
\]

  - champion의 일반적인 no-mask case에서만 \(a_n=1/196\), \(b_m=1/5\)로 환원
- objective

\[
\min_{P\ge0}
\langle P,C\rangle
+\varepsilon\sum_{n,m}P_{nm}(\log P_{nm}-1)
+\lambda_a\,\mathrm{KL}(P\mathbf1\Vert a)
+\lambda_b\,\mathrm{KL}(P^\top\mathbf1\Vert b).
\]

\[
\mathrm{KL}(p\Vert q)=\sum_j\left[p_j\log\frac{p_j}{q_j}-p_j+q_j\right]
\quad\text{(generalized KL for non-normalized masses)}.
\]

- champion 설정
  - \(\lambda_a=\lambda_b=1\)
  - log-domain generalized Sinkhorn 20 iterations
  - 60-epoch nominal cosine schedule \(\varepsilon:1.0\rightarrow0.1\)
  - objective의 exact minimizer \(P^*\)와 실제 20-step 근사 plan \(\widehat P^{(20)}\)을 논문에서 구분
- scaling

\[
K=\exp(-C/\varepsilon),\qquad
\tau_a=\frac{\lambda_a}{\lambda_a+\varepsilon},\qquad
\tau_b=\frac{\lambda_b}{\lambda_b+\varepsilon},
\]

\[
u\leftarrow\left(\frac{a}{Kv}\right)^{\tau_a},\qquad
v\leftarrow\left(\frac{b}{K^\top u}\right)^{\tau_b},\qquad
P=\operatorname{diag}(u)K\operatorname{diag}(v).
\]

- reviewer용 proof sketch
  - entropy 항이 양의 영역에서 strict convex이므로 양의 prior와 \(\varepsilon>0\) 아래 minimizer의 유일성 설명
  - first-order optimality로 \(P_{nm}=u_nK_{nm}v_m\) factorization 도출
  - KL marginal penalty의 convex conjugate/coordinate minimization으로 exponent \(\tau_a,\tau_b\) 도출
  - Chizat et al.의 generalized Sinkhorn scaling을 직접 인용
- “selective”의 정확한 해석
  - generalized Sinkhorn이 수렴한 fixed point에서 \(q_n=(Kv)_n\), row mass \(r_n=(P^*\mathbf1)_n=a_n^{\tau_a}q_n^{1-\tau_a}\)
  - 따라서 fixed point에서 \(r_n/a_n=(q_n/a_n)^{1-\tau_a}\)
  - 실제 20-step \(\widehat P^{(20)}\)에서는 이 항등식이 근사적으로만 성립할 수 있음
  - prior 대비 semantic affinity가 낮은 patch는 mass가 줄고 높은 patch는 커질 수 있음
  - 모든 \(r_n\le a_n\)이라고 주장하지 않음
  - 양 marginal이 relaxed되어 total transported mass도 1로 강제되지 않음

#### 3.4.3 Adaptive top-p sparsification

- 20-step UOT plan \(\widehat P\)의 row를 조건부 slot distribution으로 변환

\[
r_n=\sum_m\widehat P_{nm},\qquad
p_{nm}=\frac{\widehat P_{nm}}{r_n},\qquad q_n=\max_m p_{nm}.
\]

- patch별 threshold

\[
\rho_n=\rho_{\min}+(1-q_n)(\rho_{\max}-\rho_{\min}),
\qquad(\rho_{\min},\rho_{\max})=(.3,.7).
\]

- selection
  - \(p_{n,:}\)를 내림차순 정렬
  - cumulative mass가 \(\rho_n\)에 도달하는 최소 prefix 유지
  - top-1은 항상 유지
  - 선택 후 원래 UOT row mass \(r_n\)으로 재정규화
  - exact column marginal은 이 단계에서 깨질 수 있음

\[
\widetilde P_{nm}=
r_n\frac{p_{nm}\mathbf1[m\in S_n]}
{\sum_{j\in S_n}p_{nj}}.
\]
- KL projection lemma
  - 고정 support \(S\)에서 \(\min_{q\in\Delta,\operatorname{supp}(q)\subseteq S}\mathrm{KL}(q\Vert p)\)의 해는 \(q_j=p_j/\sum_{s\in S}p_s\)
  - proof: Lagrange multiplier로 conditional normalization 도출
  - \(|S|=k\)일 때 objective는 \(-\log\sum_{s\in S}p_s\); retained mass가 가장 큰 top-k support가 최적
  - 본 adaptive rule은 \(k\)를 cumulative threshold로 선택한 뒤 그 support 위 KL projection을 수행
  - 전체 heuristic이 cardinality-constrained UOT의 전역해라는 주장은 하지 않음
- barycentric pooling

\[
z_m=\frac{\sum_n\widetilde P_{nm}v_n}
{\max(\sum_n\widetilde P_{nm},10^{-12})}.
\]

- 해석상 주의
  - column-normalized pooling이 absolute column mass를 상쇄
  - UOT 효과는 주로 slot 내부 patch 상대 가중치와 \(\langle \widetilde P,C\rangle\)에 남음
  - “background를 이론적으로 버린다”보다 “low-affinity patch를 transport plan에서 downweight”로 표현

#### 3.4.4 Training–inference 차이

| 요소 | Training | Inference |
|---|---|---|
| Local centroid | image별 Qwen caption→CLIP text→whitening→slot adapter | local codebook의 normalized active-codeword mean |
| Text/Qwen | offline cache로 사용 | 완전 제거 |
| UOT/top-p | 현재 image의 196 patches와 5 instance anchors | 196 patches와 5 dataset-level learned anchors |
| Codebook | EMA update + dead-code revival | 고정 |
| Codon output | soft distribution·deterministic argmax target와 별도로 Gumbel-hard ST sample 생성 | deterministic argmax |
| Bio projection | training objective에는 미포함 | query/database code 모두 retrieval 직전 적용 |

- inference anchor

\[
\bar e_m=\operatorname{norm}\!\left(\frac1{|\mathcal K_m|}
\sum_{k\in\mathcal K_m}\operatorname{norm}(e_{mk})\right),\qquad m=1,\ldots,5.
\]

- 논문에 명시할 deployment gap
  - training은 instance-conditioned text anchors
  - inference는 dataset-level codebook anchors
  - `text_prototype` inference가 실패하고 `codebook_mean`을 채택한 실험 근거를 appendix에 기록
  - anchor swap에 대한 localization/accuracy 민감도 분석 추가
  - 현재 champion의 gradient-effective loss는 생성된 Gumbel-ST sample을 소비하지 않으므로 temperature schedule의 실효 기여를 주장하지 않음

### 3.5 Independent Multi-Codebook Quantization

- 여섯 독립 codebooks

\[
\mathcal E_m=\{e_{mk}\in\mathbb R^{768}\}_{k=1}^{K_m},\qquad m=0,\ldots,5.
\]

- nearest Euclidean assignment

\[
k^*_{im}=\arg\min_k\lVert z_{im}-e_{mk}\rVert_2^2,qquad
q_{im}=e_{m,k^*_{im}}.
\]

- straight-through token

\[
q^{\rm ST}_{im}=z_{im}+\operatorname{sg}(q_{im}-z_{im}).
\]

- EMA statistics

\[
n_{mk}^{(t)}=\sum_i\mathbf1[k^*_{im}=k],\qquad
s_{mk}^{(t)}=\sum_i\mathbf1[k^*_{im}=k]z_{im},
\]

\[
N_{mk}^{(t)}=\gamma N_{mk}^{(t-1)}+(1-\gamma)n_{mk}^{(t)},
\]

\[
A_{mk}^{(t)}=\gamma A_{mk}^{(t-1)}+(1-\gamma)s_{mk}^{(t)},
\]

\[
\widetilde N_{mk}=\frac{N_{mk}+\epsilon_{\rm ema}}
{\sum_jN_{mj}+K\epsilon_{\rm ema}}\sum_jN_{mj},\qquad
e_{mk}\leftarrow\frac{A_{mk}}{\widetilde N_{mk}}.
\]

- hyperparameters
  - \(\gamma=.99\), \(\epsilon_{\rm ema}=10^{-5}\)
  - 50 forward마다 count가 slot 최대 count의 1% 미만인 entry를 현재 batch vector로 revive
  - codebook은 trainable parameter가 아니라 EMA buffer
- 의미
  - 여섯 slot index tuple \((k_i^0,\ldots,k_i^5)\)이 Cartesian compositional representation
  - 동일 index \(k\)라도 slot이 다르면 별도 vector와 의미
  - independent codebook이 independent semantic factor를 보장하지는 않음

### 3.6 Global-conditioned codon composition

- global conditioning

\[
\widetilde q_{i0}=q^{\rm ST}_{i0},\qquad
\widetilde q_{im}=q^{\rm ST}_{im}+\sigma(\alpha_m)\operatorname{sg}(q^{\rm ST}_{i0}),\quad m=1,\ldots,5.
\]

- gate
  - \(\alpha_m\) 초기값 `4.595`, \(\sigma(\alpha_m)\approx.99\)
  - local code에 image-global context 주입
  - local head에 복사해 넣는 global token만 stop-gradient
  - global codon branch 자체는 \(q^{\rm ST}_{i0}\)를 직접 받아 upstream gradient가 흐름
- slot별 독립 CodonHead
  - \(\widetilde q_{im}\in\mathbb R^{768}\)을 세 256-D chunk로 분할
  - 같은 slot 안의 세 위치는 동일 `Linear(256,4)` 공유
  - slot 간 head parameter는 공유하지 않음

\[
\widetilde q_{im}=[\widetilde q_{im}^{(1)};\widetilde q_{im}^{(2)};\widetilde q_{im}^{(3)}],
\qquad
u_{imr}=\operatorname{softmax}(W_m\widetilde q_{im}^{(r)}+b_m).
\]

- training/inference
  - training forward: soft \(u\), deterministic argmax one-hot \(h\), hard Gumbel-softmax ST sample을 모두 생성
  - nominal Gumbel temperature schedule \(2.0\rightarrow.3\)
  - 그러나 champion의 활성 loss는 soft \(u\)와 deterministic \(h\) target을 사용하고, `lambda_hash_hard=lambda_ntxent=0`이므로 Gumbel-ST sample에는 실효 gradient가 없음
  - inference: \(b_{imr}=\arg\max_c u_{imr,c}\), \(c\in\{A,C,G,T\}\)
- 전체 DNA code
  - \(6\) slots \(\times3\) bases = 18 bases = raw 36 bits
- codon collision
  - Flickr/MS-COCO/NUS: \(K=128>64\), one-to-one codeword→codon 불가능
  - CIFAR: \(K=64\), balanced codeword–codon OT를 사용할 수 있는 용량 조건
  - codon collision rate와 codeword/codon mutual information을 별도 보고
- text-side auxiliary path의 비대칭
  - visual과 같은 codebook/CodonHead를 사용하지만 text quantization 동안 quantizer를 eval mode로 두어 EMA를 갱신하지 않음
  - text codon에는 global residual gate를 적용하지 않음
  - XM/TDNA 해석에서 image-conditioned local codon과 ungated text codon의 비대칭을 명시

### 3.7 Bio-constrained post-processing

#### 3.7.1 제약 정의

- alphabet: \(\Sigma=\{A,C,G,T\}\)
- GC count

\[
g_{\min}=\lceil.4L\rceil,\qquad g_{\max}=\lfloor.6L\rfloor.
\]

- homopolymer
  - 동일 base의 최대 연속 길이 \(R_{\max}=3\)
- 실제 window
  - \(L=18\): GC count `[8,10]`, 즉 44.4–55.6%
  - \(L=24\): GC count `[10,14]`, 즉 41.67–58.33%
- 해석
  - 40–60%/run≤3은 보편 법칙이 아니라 흔한 platform-dependent design rule
  - 두 제약만 만족하며 synthesis-ready를 의미하지 않음

#### 3.7.2 Algorithm: minimum-Hamming valid projection

```text
Algorithm 1  Bio-constrained projection Πbio(x)
Input: raw strand x1:L, alphabet Σ, GC interval [gmin,gmax], Rmax=3
Output: projected strand y1:L and edit count d minimizing dH(x,y)

1: if x already satisfies both constraints: return (x, 0)
2: define DP state D[i,g,b,r]
      i = processed length, g = cumulative GC count,
      b = last base, r = current homopolymer run length
3: initialize for every b in Σ:
      D[1, 1GC(b), b, 1] = 1[b ≠ x1]
4: for i = 1,...,L-1:
5:   for every reachable (g,b,r) and next base b' in Σ:
6:      r' = r+1 if b'=b else 1
7:      if r' > Rmax: continue
8:      g' = g + 1GC(b')
9:      relax
          D[i+1,g',b',r'] = min(
             D[i+1,g',b',r'],
             D[i,g,b,r] + 1[b' ≠ xi+1])
10: choose the minimum terminal state with g in [gmin,gmax]
11: if no feasible terminal exists: return (x, -1)
12: otherwise recover y by backpointers and return (y, dH(x,y))
```

- optimality 설명
  - state가 prefix의 미래 feasibility에 필요한 정보 `(GC count, last base, run length)`를 모두 보존
  - optimal substructure에 의해 Bellman recurrence가 최소 base-Hamming edit 보장
- 복잡도
  - 시간 \(O(L^2|\Sigma|^2R_{\max})\)
  - backpointer 포함 메모리 \(O(L^2|\Sigma|R_{\max})\)
- 평가 적용
  - query와 database 양쪽의 모든 row에 동일 projection
  - 동일 code의 projection은 순수함수이므로 unique code마다 정확히 한 번 계산하고 inverse index로 복원하는 exact memoization 사용
  - equal-cost DP 해는 versioned traversal tie policy(이전 base A→C→G→T, run/GC 오름차순; terminal도 동일 순서)로 결정하며 manifest에 기록
  - projected base Hamming으로 ranking
  - baseline에도 완전히 동일한 projection
- 한계
  - 18-base 전체에서 편집하여 3-base codon boundary/slot semantics를 보존하지 않음
  - pre/post mAP, edit count, codon decoding 변화를 함께 보고

### 3.8 Training Objectives

#### 3.8.1 실제 gradient가 있는 champion loss

- notation
  - \(z\): pre-VQ semantic token
  - \(q\): selected codeword
  - \(u\in[0,1]^{18\times4}\): soft base distribution
  - \(h\): argmax one-hot base code
  - \(\operatorname{MSE}(A,B)=|A|^{-1}\sum_j(A_j-B_j)^2\): PyTorch default mean reduction
- VQ commitment

\[
\mathcal L_{\rm VQ}=
\operatorname{MSE}(q,\operatorname{sg}(z))
+\beta_{\rm VQ}\operatorname{MSE}(z,\operatorname{sg}(q)),
\qquad\beta_{\rm VQ}=.25.
\]

  - EMA buffer이므로 첫 codebook-side 항은 실제 parameter gradient 없음
  - encoder commitment 항이 실효 학습 신호
- hard codon commitment

\[
\mathcal L_{\rm quant}=\operatorname{MSE}(u,\operatorname{sg}(h)).
\]

  - \(h\)는 Gumbel sample이 아니라 soft \(u\)의 deterministic argmax one-hot

- DNA discreteness와 base balance

\[
\mathcal L_{\rm DNA}
=-\frac1{BL}\sum_{i,\ell,c}u_{i\ell c}\log u_{i\ell c}
+\eta\frac1L\sum_\ell
\mathrm{KL}\!\left(U_4\middle\Vert\bar u_\ell\right),
\quad \bar u_\ell=\frac1B\sum_i u_{i\ell},\quad\eta=.3.
\]

- codebook balance/uncorrelation

\[
p_{imk}=\operatorname{softmax}_k(-d_{imk}/.1),\qquad
\bar p_{mk}=B^{-1}\sum_ip_{imk},
\]

\[
\mathcal L_{\rm bal}=\frac1{MK}\sum_{m,k}(\bar p_{mk}-1/K)^2,
\quad
G_m=\frac KBP_m^\top P_m,
\quad
\mathcal L_{\rm uncorr}=\frac1{MK^2}\sum_{m,k,j}
[\operatorname{offdiag}(G_m)_{kj}]^2,
\]

\[
\mathcal L_{\rm BU}=\mathcal L_{\rm bal}+.1\mathcal L_{\rm uncorr}.
\]

- routed transport cost

\[
\mathcal L_{\rm OT}=B^{-1}\sum_i\langle \widetilde P_i,C_i\rangle.
\]

  - 이름은 `loss_wasserstein`
  - 전체 UOT objective가 아니라 sparsification 이후 transport term만 사용
- paired-view per-codebook contrastive loss
  - 두 augmentation의 같은 image/slot을 positive, 다른 image를 negative로 하는 symmetric InfoNCE
  - visual pre-VQ routed token 기준
  - pairwise text-conditioned temperature

\[
\tau_{ij}^{m}=.3\left[1+.3\cos(t_i^{m,{\rm raw}},t_j^{m,{\rm raw}})\right].
\]

  - \(t^{m,{\rm raw}}\)는 whitening/adapter 전 512-D feature; local P0에서는 legacy token-mean feature
  - \(\zeta_r^m\)를 두 view의 normalized pre-VQ token을 이어 붙인 \(2B\)개 vector, \(\pi(r)\)를 같은 image의 반대 view index, \(\iota(r)\)를 원 image index라 두면

\[
s_{rj}^m=
\frac{(\zeta_r^m)^\top\zeta_j^m}
{\tau_{\iota(r)\iota(j)}^m},\qquad
\mathcal L_{\rm CIB}=
-\frac1{6(2B)}\sum_{m=0}^{5}\sum_{r=1}^{2B}
\log\frac{\exp s_{r,\pi(r)}^m}
{\sum_{j\ne r}\exp s_{rj}^m}.
\]

  - 현재 `source=visual_token`에서는 별도 Bernoulli CIB-KL이 exact zero이고 위 InfoNCE만 실효
- cross-modal commitment

\[
\mathcal L_{\rm XM}=\frac12\left[
\operatorname{MSE}(z^v,\operatorname{sg}(q^t))
+\operatorname{MSE}(t,\operatorname{sg}(q^v))
\right].
\]

  - champion은 global 포함
- confidence-weighted text-code KL

\[
p^v_{imk}\propto\exp(\cos(z^v_{im},e_{mk})/.1),\qquad
p^t_{imk}\propto\exp(\cos(t_{im},e_{mk})/.07),
\]

\[
w_{im}=1-\frac{H(p^t_{im})}{\log K},\qquad
\mathcal L_{\rm TCKL}=\mathbb E_{m=1:5,w>.2}
\left[w_{im}\mathrm{KL}(\operatorname{sg}(p^t_{im})\Vert p^v_{im})\right].
\]

- local text–DNA InfoNCE
  - visual/text soft DNA의 slot별 12-D block
  - local slots 1–5에 symmetric InfoNCE, temperature `.07`
  - global slot skip

\[
d_{im}^{v}=\operatorname{vec}(u_{im}^{v})\in\mathbb R^{12},\qquad
d_{im}^{t}=\operatorname{vec}(u_{im}^{t})\in\mathbb R^{12},
\]

\[
S_{ij}^{m}=\frac{\cos(d_{im}^{v},d_{jm}^{t})}{.07},\qquad
\mathcal L_{\rm TDNA}=\frac1{10}\sum_{m=1}^{5}
\left[\operatorname{CE}(S^m,I)+\operatorname{CE}((S^m)^\top,I)\right].
\]
- CIFAR 전용 codeword–codon Sinkhorn

\[
p_{mk}(c_1c_2c_3)=\prod_{r=1}^{3}p_{mk}^{(r)}(c_r),\qquad
C_{mk,c}=-\log p_{mk}(c),
\]

\[
T_m^*=\operatorname{Sinkhorn}(C_m;U_K,U_{64},\epsilon=.1,30\text{ iters}),
\qquad
\mathcal L_{\rm CCS}=\frac1M\sum_m\langle T_m^*,C_m\rangle.
\]

  - \(K=64\)일 때 soft balanced assignment가 bijection을 장려
  - 여기서 \(p_{mk}\)는 raw EMA codeword를 gate/residual 없이 \(H_m(e_{mk})\)로 decode하여 구성
  - 실제 local sample은 \(H_m(q^{\rm ST}_{im}+\sigma(\alpha_m)\operatorname{sg}(q^{\rm ST}_{i0}))\)에서 방출되므로 CCS가 emitted local codon의 bijection을 직접 강제하지는 않음
  - codebook은 EMA buffer이므로 CCS gradient는 주로 CodonHead parameter로 전달
  - deterministic argmax codon의 완전한 일대일성을 수학적으로 보장한다고 쓰지 않음

#### 3.8.2 Gradient-effective champion objective

- 아래 식은 parameter update에 실효 gradient를 주는 항만 모은 objective
- 실제 `loss_total` scalar에는 no-gradient anchor와 exact-zero CIB-KL 등도 config weight와 함께 더해짐

\[
\begin{aligned}
\mathcal L={}&
\lambda_{\rm VQ}\mathcal L_{\rm VQ}
+\lambda_{\rm quant}\mathcal L_{\rm quant}
+\lambda_{\rm DNA}\mathcal L_{\rm DNA}
+\lambda_{\rm BU}\mathcal L_{\rm BU}\\
&+\lambda_{\rm OT}\mathcal L_{\rm OT}
+\lambda_{\rm CIB}\mathcal L_{\rm CIB}
+\lambda_{\rm XM}\mathcal L_{\rm XM}
+\lambda_{\rm TCKL}\mathcal L_{\rm TCKL}\\
&+\lambda_{\rm TDNA}\mathcal L_{\rm TDNA}
+\lambda_{\rm CCS}\mathcal L_{\rm CCS}.
\end{aligned}
\]

#### 3.8.3 양수 config지만 실질적으로 비활성인 항

| 항 | Config | 실제 상태 | 논문 처리 |
|---|---:|---|---|
| Anchor alignment | `.05` | EMA codebook mean과 detached text EMA만 사용하여 trainable gradient 없음 | gradient-effective objective에서 제외; implementation audit로 명시 |
| CIBHash KL | `.001` | `source=visual_token` branch가 exact zero 반환 | gradient-effective objective에서 제외 |
| Reconstruction | `1.0` | `use_decoder=False` | 비활성 |
| Codon text anchor | `.1` | `codon_text_anchor=False` | 비활성 |
| Pairwise hash / hard hash | `0/0` | similarity target은 구성되더라도 loss weight 0 | label-free gradient objective 확인 |
| Generic DNA NtXent | `0` | 비활성 | per-codebook CIB와 혼동 금지 |

- 그 밖의 weight 0 champion 경로
  - codebook orthogonality
  - codeword-codon aggregated entropy/pairwise
  - text-cluster codon OT, text-codon relation, hierarchical codon
  - hash reconstruction, dual semantic/instance heads
  - text-hash MSE, codeword xmodal, routing-text
  - text-codeword/pre-quant/visual-hash contrastive
  - codeword-text prototype, global DNA NtXent, text orthogonality
  - prototype clustering, SwAV assignment

#### 3.8.4 Joint codon-diversity 규제 `L_joint` (본 논문 신규)

기존 `L_base-balance`는 위치별 주변분포에만 걸리므로, **위치별로는 균등한데 세 위치의 결합만
붕괴한** 슬롯을 구조적으로 볼 수 없다. 실제로 A-champion의 global 슬롯은 A/C/G/T를 모두 쓰고
정규화 엔트로피 `.932`(정상 슬롯 `.951–.985`)인데도 64칸 중 **21칸**만 사용했다. 자기 marginal이
허용하는 기대 커버리지는 48.6칸이므로 **27.6칸이 3-way 종속으로 사라진 것**이다.

슬롯 `m`, 위치 `l`의 4-way 사후분포를 `p[b][m][l]`이라 하면, 한 샘플 안에서 L개 위치는 각자
독립 softmax이므로 그 샘플의 codon 분포는 외적이다.

```
J[b][m][c] = Π_l  p[b][m][l][ c_l ]                    c = (c_0 … c_{L-1})
Q[m]       = (1/B) Σ_b J[b][m]                          ∈ Δ^(4^L − 1)
L_joint    = (1/|S|) Σ_{m∈S} KL( U_{4^L} ‖ max(Q[m], ε) )
```

- `L_base-balance`와 **동일한 forward-KL 형태**를 쓴다. mode-covering이므로 `Q[m][c] → 0`인,
  즉 한 번도 방출되지 않는 codon에 페널티가 집중된다 — 정확히 관측된 병리다.
- `ε = 1e-6` floor가 codon당 페널티를 `log((1/4^L)/ε) ≈ 9.7`로 유계화한다.
- 배치 평균 후에 KL을 취하므로 샘플별 분포가 뾰족해도 무방하다. `L_entropy`·`L_quant`와 충돌하지 않는다.
- 단위 검증: 64칸 균등 `0.008`, 21칸 붕괴 `6.12`, 1칸 `9.44`.
- 이 항이 줄이는 양은 다중정보 `TC(m) = Σ_l H(Q_marg[m][l]) − H(Q[m])` 이다. MS-COCO global 슬롯
  기준 `1.722 → .159`로, 정상 슬롯 범위(`.25–.41`) 안으로 들어간다.

⚠️ **원인이 아니라 증상을 규제한다.** 왜 global 슬롯에서만 세 위치가 종속되는지는 규명하지
못했다. head 측 자유도(위치 간 가중치 공유·chunk 스케일 정규화·chunk 분할 방식)는 모두 반증되었고,
종속은 codebook 형성 단계에 존재한다. 논문에는 "원인 미규명 상태의 효과적 규제"로 정직하게 적는다.

### 3.9 데이터셋별 P0 recipe — 신규 통일 레시피 (2026-08-04)

네 데이터셋이 **동일 아키텍처**를 쓴다. 데이터셋별로 다른 것은 λ, K, 프롬프트, 그리고 기존
champion에서 물려받은 loss weight뿐이다.

| Dataset | prompt | K | **λ_joint** | noGumbel | bijection | OT | XM | TDNA | TCKL | CIB | CCS |
|---|---|---:|---:|:---:|:---:|---:|---:|---:|---:|---:|---:|
| Flickr25K | v4 | 128 | **.02** | ✔ | off | `.15` | `.05` | `.05` | `.05` | `1.0` | `0` |
| MS-COCO | **v5b** | 128 | **.03** | ✔ | off | `.05` | `.10` | `.10` | `.10` | `1.5` | `0` |
| NUS-WIDE | v4 | 128 | **.05** | ✔ | off | `.15` | `.05` | `.05` | `.05` | `1.5` | `0` |
| CIFAR-10 | v4 | 64 | **.03** | ✔ | **off (변경)** | `.15` | `.05` | `.05` | `.05` | `1.0` | `.10` |

- **A-champion과의 차이는 정확히 세 가지**다: `λ_joint > 0`, `--no_gumbel_softmax`,
  그리고 CIFAR의 bijection을 `.1 → 0`으로 끈 것.
- bijection은 원래 **CIFAR에만** 켜져 있었다. `L_joint`와 codeword→codon 사상을 두고 상반된
  압력을 걸어 충돌하며, 끄면 검색·해석성 모두 개선된다(val `+.0227`). 이로써 아키텍처가 통일된다.
- 프롬프트는 4개 중 3개가 v4이고 MS-COCO만 v5b다. **“데이터셋마다 다른 프롬프트”라는 일반 주장은
  하지 않는다**; v5b의 우위는 global-caption-free 레시피에서만 나타나는 조건부 결과다.
- 공통 설정(batch 64, Adam, LR `.001`, frozen backbone, λ_VQ `.25`, λ_quant `.05`, λ_DNA `.05`,
  λ_BU `.02`, UOT `λ_a=λ_b=1`, EMA `.99`, partial whitening `γ=.25`, nominal 60-epoch cosine)은
  A-champion과 동일하다.
- E\*는 seed마다 stage-1에서 독립적으로 선택한다. seed 42의 E\*를 재사용하지 않는다.

## 4. Experiments

### 4.1 Research Questions

- RQ1: 동일 bio-valid 18-base space에서 GroundedDNA가 binary hashing baseline보다 높은 retrieval accuracy를 보이는가?
- RQ2: text/UOT/multi-codebook 중 무엇이 성능과 codon semantics에 기여하는가?
- RQ3: DP projection이 feasibility를 얻는 대가로 retrieval·uniqueness·semantic decoding을 얼마나 바꾸는가?
- RQ4: 각 slot/codon은 held-out data에서 어떤 의미 구조를 보존하는가?
- RQ5: train-time text anchor와 inference codebook anchor의 gap은 얼마나 큰가?
- RQ6: code length, K, codebook 수, latency/memory 사이 trade-off는 무엇인가?

### 4.2 Dataset과 관행적 protocol

| Dataset | 사용 subset | Train | Query | Retrieval DB | Relevance | Metric |
|---|---:|---:|---:|---:|---|---|
| Flickr25K | 25,000 / 24 concepts | 5,000 | 2,000 | 23,000 | ≥1 shared label | mAP@5,000 |
| MS-COCO | 122,218 / 80 classes | 10,000 | 5,000 | 107,218 | ≥1 shared label | mAP@5,000 |
| NUS-WIDE | 195,834 / 21 frequent concepts | 10,500 | 2,100 | 193,734 | ≥1 shared label | mAP@5,000 |
| CIFAR-10 | 60,000 / 10 classes | 5,000 | 1,000 | 59,000 | same class | mAP@1,000 |

- 현재 설정의 문헌 계보
  - CIFAR/NUS/COCO: [CIBHash protocol](https://www.ijcai.org/proceedings/2021/0133.pdf) 계열
  - Flickr 5K train: [SPQ, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Jang_Self-Supervised_Product_Quantization_for_Deep_Unsupervised_Image_Retrieval_ICCV_2021_paper.pdf) 계열
- repository split의 정확한 포함 관계
  - Flickr/NUS/CIFAR: train set이 retrieval DB의 subset인 관행적 형태
  - MS-COCO: train 10K와 DB 107,218이 disjoint
  - CIBHash 원 protocol의 “query 5K, remaining 117,218 DB, 그중 train 10K sample”과 다를 수 있으므로 CIBHash와 완전히 동일한 split이라고 쓰지 않음
  - split file 자체와 path-level overlap count를 supplementary에 공개
- protocol이 하나가 아니라는 점 명시
  - CIMON 원 논문 Flickr: train 10K
  - CIMON 원 논문 CIFAR: mAP@50K
  - CSQ/DHD NUS subset 및 split count가 서로 다름
  - published number를 다른 protocol끼리 한 표에 직접 혼합하지 않음
- main table의 baseline 의미
  - 논문 원문 수치가 아니라 repository에서 동일 split/artifact와 동일 post-processing으로 재평가한 결과
  - split manifest, seed, file hash를 supplementary에 공개

### 4.3 Validation-based P0 selection/refit 및 test-use audit

- Stage 1
  - designated train의 90% optimization / 10% validation query
  - single-label: stratified split
  - multi-label actual code: seed 42 random shuffle
  - 문서의 iterative stratification 권고와 실제 구현 불일치; 수정 또는 명시 필요
  - validation query는 optimization-train database에 검색
  - 5 epoch마다 validation mAP로 E* 선택
  - Flickr validation DB 4,500 < R=5,000이어서 사실상 full-DB validation mAP
- Stage 2 P0 refit
  - GroundedDNA: 전체 designated train에서 scratch refit, stage-1과 동일한 nominal 60-epoch schedule을 두고 E*에서 `stop_after_epoch`
  - GroundedDNA whitening 통계는 full train에만 재적합
  - 최신 baseline runner: full designated train에서 scratch refit하되 `max_epoch=E*+1`과 nominal `schedule_horizon=H`를 분리; DUH-EG/CroVCA cosine schedule을 짧은 refit 길이로 압축하지 않음
  - legacy CIBHash/CIMON/MLS3RDUH historical artifact: 90%-train validation checkpoint로 E*를 고른 뒤 별도 100%-train run의 epoch-E* checkpoint를 사용; strict one-shot clean refit과 구분
- Test-use audit
  - 모든 method의 보고 E* 선택은 held-out validation에만 의존하여 test-independent
  - GroundedDNA final refit은 E*에서 멈추고 final test를 한 번 평가
  - 최신 baseline runner도 test/database를 stage 1에서 생성하지 않고 test와 무관하게 하나의 final checkpoint를 고정; raw와 post-projection metric은 같은 checkpoint에서 계산하므로 extraction이 test를 다시 읽더라도 test-dependent choice는 없음
  - historical baseline artifact에는 여러 epoch의 test JSON/curve가 존재하므로 모든 method의 literal “test touched once”를 주장하지 않음
  - strict one-shot protocol 주장을 위해서는 baseline clean rerun 필요
- 현재 반복
  - validation carve seed `42`
  - method당 single training run: GroundedDNA seed `42`, baseline 기본 training seed `1`
  - 최종본: 3 seeds mean±std + query bootstrap 95% CI 필요

### 4.4 Baselines와 공정한 DNA 변환

- 현재 main baselines
  - CIBHash
  - CIMON
  - MLS3RDUH
- 추가 구현·실행 queue
  - direct native DNA: DNA24-18, PRIMO-18, Koike-TN-DNA, Koike-BC-TN-DNA의 sealed 3-seed diagnostic 48/48 완료; strict provenance 재실행 대기
  - `U0` visual-only: UGH/GreedyHash(`official unsupervised core`), Bi-half(`official image-release core`), SDC(`release_no_cl`, `release_simclr`, `paper_cache2v`), OH(`OH-cache2v` matched adapter), HHCH(`paper_cache3v`), CroVCA(`matched-cache probing`)
  - `U0` 실행 `B` / Related Works 인용: DDCH(공식 code 미확인), CGHash(외부 checkpoint·dataset path 불완전), CTMIH(raw masked-ViT/patch 복원 core와 공식 code 부재), FSCH(trainer/loss/optimizer 누락)
  - `U0` 대표 공간 경계 `C`: HiHPQ는 asymmetric product-quantization distance이므로 binary/Hamming main table에서 제외
  - external-semantic: DUH-EG objective adapter는 현재 `U?` (`I/E`), 저자 ordered bank provenance 확보 시에만 `U1`; OMUH는 detector pseudo-label·공개 protocol defect로 `B`; TGUH는 caption pipeline·36-bit 파라미터·test-selection 문제 감사 후
  - `U2` target-taxonomy-guided: UMRCH; UHSCM은 공개 train code repair 전에는 cite-only
  - CUB/fine-grained 별도 track: A²-SSL, FAPI, CS3H
  - supervised upper-bound 별도 표: FTH, MambaHash, DSCH-2026, DGrH
- baseline conversion
  - 36-bit sign code \([N,36]\)을 \([N,18,2]\)로 reshape
  - `00→A`, `01→C`, `10→G`, `11→T`
  - query/database 양쪽 DP projection
  - projected base-Hamming으로 재평가
- 공정성 범위
  - 동일 raw capacity 36 bits ↔ 18 bases
  - 동일 frozen CLIP backbone, split, relevance, cutoff, bio constraints, distance
  - 단, method-defining 정보는 허용: global-only(GreedyHash/Bi-half/SDC/OH/DUH-EG/HHCH/CroVCA), global+local(UMRCH), fixed cached two-view/three-view, external noun/taxonomy bank 여부를 information-condition 열에 명시
  - `stage 1`: test split을 로드하지 않고 val-query vs opt-train DB의 **raw 2-bit→18-base base-Hamming mAP@R**로 (E^*) 선택
  - E* candidate grid: GroundedDNA/legacy P0와 동일한 5-epoch cadence, 즉 0-indexed epoch `4,9,...`; 방법별 candidate 수를 임의로 늘리지 않음
  - `stage 2`: full designated train에서 scratch refit 후 test와 무관하게 final checkpoint 1개 고정; binary-Hamming val 선택과 혼용 금지; nominal schedule horizon은 stage 1과 동일하게 유지
  - 공통 cache의 fixed augmentation은 각 논문의 online augmentation을 그대로 재현하지 않으므로 결과를 published-table reproduction이 아닌 matched-cache adapter로 명시
  - 현재 repository의 default legacy cache metadata에는 exact transform/augmentation seed/immutable HF revision·weight provenance 중 일부가 없으므로 modern binary runner는 기본적으로 학습 전 중단하고 `--allow-main-ineligible-smoke`를 명시한 smoke만 허용
  - native-DNA driver는 `--allow-main-ineligible-diagnostic`을 명시한 full sealed 실행을 허용하되 main 승격은 금지; 현재 extractor로 cache를 재생성한 뒤 consumed NPY SHA까지 protocol/checkpoint에 고정해야 strict main row 가능
  - 특히 UMRCH release는 CLIP ViT-B/32의 49 patch를 사용하지만 공통 cache adapter는 CLIP ViT-B/16의 196 patch를 사용
- 한계
  - binary hashing baseline은 DNA head를 재학습한 비교가 아니라 Level-1 transcoding
  - DNA24/PRIMO/Koike는 learned 4-way native-DNA head를 사용하므로 위 transcoding 한계에 해당하지 않지만, matched-cache adaptation이며 strict provenance 결과는 아직 `-`
  - Pradhan 계열의 표준 benchmark 재구현 비교는 아직 `-`
  - GreedyHash·Bi-half·SDC·OH·HHCH·CroVCA·DUH-EG·UMRCH runner는 구현했으나 full P0 3-seed 결과는 아직 `-`
  - DUH-EG는 공개 자료가 representative/center noun 선택에서 서로 모순되고 ordered selected-noun artifact가 없어 **exact paper baseline은 blocked**; 현재 코드는 manifest·명시적 opt-in이 필요한 released-objective adapter만 제공
  - UMRCH는 target taxonomy를 쓰므로 main U0 표와 분리
  - FSCH는 구현한 것으로 표기하지 않으며, 저자의 누락 artifact를 받기 전까지 `-`
  - ConceptHash/LPAH/HTH의 동일 protocol 재학습 비교도 아직 `-`

#### 4.4.0 최신 hashing baseline 구현 변형과 재현 경계

| Runner | 학습 정보 | 구현·보고 단위 | 현재 상태 |
|---|---|---|---|
| UGH(GreedyHash)-cache | `U0`, canonical global | hard-sign forward/identity-backward STE, half-pair cosine MSE, cubic quantization; official SGD recipe | clean-room runner·release-parity test; frozen VGG16-fc7/train-mode dropout→deterministic common cache·36-bit adaptation, full run `-` |
| Bi-half-cache | `U0`, canonical global | per-bit half-rank assignment, released proxy `g+6(U-B)/(NK)`, half-pair cosine MSE, inference sign | clean-room runner·release-parity test; paper coefficient 3 vs release 6 차이 명시, NUS trainer 미공개, full run `-` |
| SDC-release-noCL | `U0`, canonical global | 공개 기본 D→D→36+BN, SDC+cosine quantization, Adam (10^{-4}), WD (10^{-5}) | runner·unit test 구현; full run `-` |
| SDC-release-SimCLR | `U0`, fixed two-view global | 공개 실행 config D→D→36+BN, 양 view에 4-block SDC+quantization, ReLU-clipped target, NT-Xent, WD (10^{-5}) | runner·pair-layout golden test; full run `-` |
| SDC-paper-cache2v | `U0`, fixed two-view global | paper Algorithm 1의 4096-hidden, 첫 view에만 SDC+quantization, 두 view NT-Xent, unclipped [-1,1] target, WD (5×10^{-4}) | runner·unit test 구현; release config과 혼합하지 않음; full run `-` |
| OH-cache2v | `U0`, fixed two-view global | 2048 shared trunk, sigmoid hard-`0/1` STE와 1024-D continuous head, EMA key encoder, P/B/C FIFO queue, old→updated queue 순서의 두 CE loss | 공개 TensorFlow core의 clean-room runner·queue/checkpoint/signed-code golden test; online ResNet-50 원 table 재현 아님; full run `-` |
| HHCH-paper-cache3v | `U0`, canonical+fixed two-view global | paper HIC/HPC, Poincaré hierarchy, K-means++/Einstein midpoint, log-cosh quantization; paper에 없는 LR schedule은 official release의 epoch-30 piecewise decay로 고정 | clean-room runner·math/schedule-boundary test; full run `-` |
| CroVCA-cache2v-probe | `U0`, fixed two-view global | small HashCoder, symmetric stop-gradient BCE, Algorithm-1 coding-rate loss; frozen probing adaptation | runner·independent equation/gradient test; LoRA official reproduction 아님; full run `-` |
| DUH-EG-cache2v | 현재 `U?`, source/selection-unverified external terms; taxonomy semantic match는 `U2` | released three-way multi-positive objective; prompt·ordered noun·cache/output SHA manifest 검증 | objective adapter 구현·legacy-cache smoke 통과; 저자 ordered bank provenance 전 main/exact row **blocked**, full result `-` |
| UMRCH-cache2v-local | `U2`, target class names+CLIP | released distribution/contrastive objective; cached pre-LN patch에 exact CLIP LN/projection 적용 | official ordered taxonomy SHA·asset verification·real-cache smoke 통과; ViT-B/16 adaptation, full result `-` |
| FSCH | `U0` 주장 | 공개 repo만으로 loss/training 재구성 불가 | **blocked**, `-` |

- 구현 provenance
  - GreedyHash·Bi-half·SDC·OH·HHCH·FSCH·CroVCA·DUH-EG·UMRCH 공개 repo에서 명시적 software license를 확인하지 못했으므로 source copy 대신 논문 수식 기반 clean-room 재구현
  - 공개 repo는 equation/config audit에만 사용; paper/release discrepancy를 variant 이름·checkpoint config에 저장
  - 공식 논문의 16/32/64-bit 숫자는 가져오지 않고 36-bit matched run만 main comparison에 기입
  - external CLIP asset는 immutable Hugging Face commit, model/tokenizer SHA, ordered term SHA, cache-meta SHA까지 manifest로 검증

#### 4.4.1 직접 선행 native-DNA baseline의 구현 및 보고 단위

- 결과를 두 층으로 분리
  - `original reproduction`: 원 feature, 30/80 nt, 원 query와 NUPACK/DDH 또는 wet-lab 지표
  - `matched adaptation`: 동일 cached CLIP feature, 동일 split, 18 bases, per-image query, 공통 DP, base-Hamming mAP@R
  - 원 논문의 classification accuracy·molecular recall을 matched mAP 표에 복사하지 않음
- `DNA24-18-matched`
  - train feature에만 PCA-10 fitting
  - original Caltech/PCA space의 Euclidean threshold `.2`는 original reproduction에만 사용; matched CLIP cache에서는 optimization-train pair-distance quantile로 threshold를 calibration하고 값을 기록
  - shared $1\to128$ sine, $128\to128$ ReLU pointwise layers와 $18\times4$ softmax output
  - $d_{ij}=L^{-1}\sum_t[1-\cos(p_{it},p_{jt})]$
  - $\widehat Y_{ij}=[1+\exp\{12.1(d_{ij}-.5)\}]^{-1}$, feature-distance pair label과 BCE
  - `12.1/.5`는 30-mer calibration이므로 18-mer에서 그대로 쓴 행에는 `analytic-transfer` 표기
- `PRIMO-18`
  - $D\to D/2\to18\times4$, position softmax, entropy coefficient $10^{-2}$
  - source-equivalent entropy: regularizer 자체가 batch mean을 반환한 뒤 Keras `activity_regularizer`가 각 encoder call을 다시 $B$로 나누므로; pair의 두 call을 합하여 $10^{-2}(H_1+H_2)/B$
  - original VGG-FC2 space의 Euclidean threshold `75`는 original reproduction에만 고정; source sampler는 유사/비유사 candidate pool을 균형 있게 구성한 뒤 합쳐 shuffle하고 batch 크기로 truncate하므로 실제 minibatch를 강제로 50:50으로 만들지 않음
  - matched CLIP cache threshold는 optimization-train에서만 calibration
  - local 3-mer interaction $4\times3\times3=36$ channels → avg-pool → Conv1D-36 → global pool → yield logit
  - 공식 `pub` commit의 NUPACK-trained Keras predictor를 PyTorch layout으로 exact 변환·검증했으며, artifact가 없거나 SHA가 다르면 실행을 중단; random predictor나 DNA24 sigmoid를 PRIMO로 부르지 않음
  - official 80-mer predictor를 freeze한 18-mer 사용은 `frozen-predictor-length-transfer`; 매 encoder epoch의 NUPACK label 생성·10-epoch predictor refit까지 수행한 경우만 full PRIMO reproduction, 새 18-mer thermo yield로 refit한 경우만 `PRIMO-18-calibrated`
- `Koike-TN-DNA`
  - DATE/DAC는 동일 core이므로 한 행
  - $z_{itb}=p_{itb}\mathbf1[b=\arg\max_c p_{itc}]$
  - $d(i,j)=(2L)^{-1}\sum_t\|z_{it}-z_{jt}\|_1$
  - semi-hard triplet margin `.8` + source-equivalent entropy $.01H/B$; Keras Adagrad LR `.01`, initial accumulator `.1`, epsilon $10^{-7}$
- `Koike-BC-TN-DNA`
  - 위 목적 + $\mathcal L_{prob}=-\operatorname{mean}\log(\max p-\min p+\epsilon)$
  - expected GC를 `.5`로 당기는 $\mathcal L_{GC}=\operatorname{mean}(g_i-.5)^2$
  - adjacent soft-cosine window 기반 $\mathcal L_{HP}$와 원 저자 HP heuristic
  - heuristic은 minimum edit와 GC validity를 보장하지 않으므로 이후 공통 exact DP 적용
  - `neural raw`(argmax) → `paper HP post-process` → `common exact DP`의 세 단계 성능·validity를 분리; soft probability와 argmax가 일치하는 neural-raw code도 artifact에 별도 저장
- supervision 경계
  - GroundedDNA는 **`VLM-T`**: target-dataset ground-truth label·taxonomy는 representation objective에 넣지 않지만 VLM-generated caption과 frozen text encoder를 supervision으로 사용하므로 visual-only `U0`로 부르지 않음
  - DNA24/PRIMO는 **`U0-FD` unsupervised; target-label-free encoder objective**: target-dataset ground-truth label·taxonomy·caption을 목적함수에 넣지 않고 frozen optimization-train feature distance에서 pair target을 구성
  - PRIMO의 frozen hybridization-yield predictor는 non-semantic external artifact이므로 정보 조건에는 명시하지만 `S`로 재분류하지 않음
  - Koike DATE/DAC/TCBB는 ground-truth training label을 사용하는 **`S` supervised direct-prior baseline**
  - `VLM-T`/`U0`/`U0-FD`도 held-out label을 validation retrieval 평가와 \(E^*\) 선택에만 사용하므로, 이 tag는 encoder objective의 정보 조건이지 전체 protocol의 label-blind성을 뜻하지 않음
  - multi-label에서는 $y_i^\top y_j>0$을 positive, `=0`을 negative로 정의한 명시적 adaptation
  - 결과표의 `†`는 supervision이 아니라 strict-main 승격이 금지된 diagnostic-only cell만 뜻함
- P0 실행
  - stage 1: `val_split.carve_val_indices(...,.1,42)`, **neural-raw 18-base Hamming** val mAP@R로 E* 선택, test 미로딩; projected-E* 선택은 별도 protocol ablation으로만 허용
  - stage 2: 전체 designated train에서 E*+1 epochs refit 후 test와 무관하게 checkpoint 1개를 고정; raw·post-projection terminal metric은 같은 checkpoint에서 계산하며 test-dependent 선택은 없음
  - artifact: `extract_{query,db}_neural_raw.npz`, standard pre-DP deployment `extract_{query,db}.npz`, projected files; train file은 `neural_raw_base_indices`를 함께 저장
  - metric key도 `neural_raw`, `paper_hp_postprocessed`(TCBB), `projected`로 분리; DP edit/compliance/unique ratio의 기준 stage를 필드명에 표시
- 구현 검증
  - hard one-hot에서 Koike soft distance = normalized base-Hamming 확인
  - `A,T,C,G→A,C,G,T` remap과 2-bit serialization 확인
  - query/DB 양쪽 exact DP 후 18-base GC count `[8,10]`·run≤3 확인
  - 실제 Flickr25K cache로 TCBB variant의 1-step P0 stage-1 smoke test 완료

### 4.5 Main Results — 신규 통일 레시피, 3-seed

#### 4.5.0 보고 대상 모델

본 절의 모든 수치는 다음 **통일 레시피**로 재학습한 모델이다.

- 전 슬롯 **joint codon-diversity 규제** `L_joint` (§3.8.4)
- **deterministic straight-through** codon 이산화 (`--no_gumbel_softmax`)
- **codeword–codon Sinkhorn bijection 비활성** (4개 데이터셋 공통)
- 데이터셋별 λ: MS-COCO `.03`, NUS-WIDE `.05`, Flickr25K `.02`, CIFAR-10 `.03`
  - MS-COCO의 λ는 3-seed로 재확인했다: `.03` → `.8232 ± .0098`, `.05` → `.8194 ± .0056`.
    평균은 `.03`이 높지만 두 분산이 겹치므로 **통계적으로 구분되지 않는다**
- 그 외 구조·프롬프트·P0 프로토콜은 §3과 동일

이전 draft의 champion(이하 `A-champion`)과의 유일한 차이는 위 세 항목이다.

#### 4.5.1 Panel A — 18-base headline, seeds `{42,43,44}` mean ± sample std

**모든 행이 3-seed다.** baseline은 9개 `U0` variant × 3 dataset × seeds `{43,44}` 60셀을
실패 0건으로 완료하여 seed 42와 합산했다.

| Method | Info | Code formation | Flickr25K @5K | MS-COCO @5K | NUS-WIDE @5K | CIFAR-10 @1K |
|---|---|---|---:|---:|---:|---:|
| CIBHash | `U0` | 36-bit → 18-base | .7826 | .7700 | .7871 | - |
| CIMON | `U0` | 36-bit → 18-base | .8140 | .6708 | .7946 | - |
| MLS³RDUH | `U0` | 36-bit → 18-base | .7561 | .6332 | .7561 | - |
| GreedyHash-UGH | `U0` | 36-bit → 18-base | .6493 | .5563 | .6447 | - |
| Bi-half | `U0` | 36-bit → 18-base | .8180 | .7062 | .7547 | - |
| SDC-paper | `U0` | 36-bit → 18-base | .7263 | **.8092** | .7529 | - |
| OH | `U0` | 36-bit → 18-base | **.8327** | .7656 | **.8028** | - |
| HHCH | `U0` | 36-bit → 18-base | .6119 | .4724 | .4026 | - |
| CroVCA | `U0` | 36-bit → 18-base | .7698 | **.8216** | .7984 | - |
| DNA24-18 analytic-transfer | `U0-FD` | learned 18×4 DNA head | .7808 ± .0071 | .6330 ± .0127 | .7427 ± .0055 | .7786 ± .0101 |
| PRIMO-18 length-transfer | `U0-FD` | learned 18×4 DNA head | .7882 ± .0209 | .6251 ± .0129 | .7320 ± .0109 | .7344 ± .0123 |
| **GroundedDNA (unified)** | **`VLM-T`** | **six grounded codons → 18-base** | **.8668 ± .0017** | **.8232 ± .0098** | **.8283 ± .0008** | **.8940 ± .0033** |

- CIFAR-10의 `U0` 열은 seeds `{43,44}`를 아직 돌리지 않아 `-`다(해당 60셀 배치는
  Flickr/MS-COCO/NUS-WIDE만 포함했다). CIFAR 비교는 §4.5.2의 single-seed diagnostic만 유효하다.
- baseline은 legacy cache provenance 때문에 invariant #6상 여전히 **strict-main ineligible**이다.
  seed 수를 채운 것은 invariant #3·#9를 만족시킬 뿐, #6은 별개 조건이다.

#### 4.5.2 최강 `U0` baseline 대비 (3-seed 대 3-seed)

| Dataset | ours 3-seed | best `U0` 3-seed | **Δ** | A-champion Δ (seed 42) |
|---|---:|---|---:|---:|
| Flickr25K | .8668 ± .0017 | OH .8327 | **+.0341** | +.0313 |
| NUS-WIDE | .8283 ± .0008 | OH .8028 | **+.0255** | +.0239 |
| MS-COCO | .8232 ± .0098 | CroVCA .8216 | **+.0016** | −.0087 |
| CIFAR-10 | .8940 ± .0033 | CIBHash .8968 *(1 seed)* | −.0028 | +.0090 |

🟢 **3-seed 대 3-seed 비교에서 MS-COCO가 `+.0016`으로 뒤집힌다.** 이는 우리 값이 올라서가 아니라
**CroVCA의 3-seed 평균이 `.8257`(seed 42) → `.8216`으로 내려갔기 때문**이다. 즉 이전에 관측된
MS-COCO 열세의 상당 부분은 **baseline 쪽의 seed 운**이었다.

⚠️ 그러나 `+.0016`은 우리 std `.0098`의 6분의 1에 불과하므로 **통계적으로는 동률**로 서술해야 한다.
"MS-COCO에서 CroVCA와 대등하다"가 정확하고, "이겼다"는 쓰지 않는다.

- Flickr25K·NUS-WIDE는 3-seed 대 3-seed에서도 `+.034` / `+.026`으로 확고하다.
- CIFAR-10만 열세이며 baseline이 아직 single seed다. **CIFAR `{43,44}` 실행이 남은 유일한 빈칸이다.**
- full mAP는 mAP@R 개선 대비 하락한다(상위 절단 이득, 꼬리 손실). §4.6에 수치를 둔다.

### 4.6 Bio projection 효과 — 신규 모델에서 재측정

legacy `bio_projection_18base.json`은 protocol/checkpoint provenance가 없어 폐기하고, 각 run의
자체 extraction에서 다시 계산했다(`scripts/bioproj_effect_newmodel.py`).

| Dataset | pre mAP@R | post mAP@R | Δ | mean DB edits | valid pre | valid post | DNA-unique pre→post |
|---|---:|---:|---:|---:|---:|---:|---|
| MS-COCO | .8317 | .8287 | −.0031 | .793 | 46.4 % | 100 % | .2033 → .1954 |
| NUS-WIDE | .8297 | .8274 | −.0024 | .631 | 56.5 % | 100 % | .2272 → .2158 |
| Flickr25K | .8756 | .8673 | −.0084 | 1.108 | 40.5 % | 100 % | .4907 → .4703 |

- **feasibility 100 %를 mAP `−.0024 … −.0084`에 얻는다.** legacy 범위(`−.0037 … −.0088`)와 동일한
  수준이며, 신규 레시피가 projection 비용을 늘리지 않았다.
- unique ratio는 전 데이터셋에서 감소한다. projection이 다양성을 개선한다고 주장하지 않는다.
- ⚠️ **사전 유효율이 40–57 %로, 균등 4^L 기대치 45.2 %와 같은 수준이다.** 즉 학습이 생물 제약을
  전혀 학습하지 않으며, 제약은 순수한 사후 투영으로 남는다. constraint-aware 학습은 §6 향후 연구다.
- projection은 `base_indices`만 바꾸고 `codebook_indices`는 바꾸지 않는다. NMI/drop/codeword decoding은
  구조적으로 불변이다.

### 4.7 Held-out codon decoding — raw-base-E\* control 대비 4/4

control은 `U0` baseline을 **동일 raw-base-Hamming E\* 규약**으로 선택한 P0 matrix artifact이며,
3-base chunk로 나눈 post-hoc 통제다. 우리 값은 seeds `{42,43,44}` mean ± std다.

| Dataset | ours codon (3-seed) | 최강 control | margin | majority | shuffled |
|---|---|---|---:|---:|---|
| Flickr25K | **.7694 ± .0017** | OH .7261 | **+.0433** | .4730 | .4783 ± .0118 |
| MS-COCO | **.6466 ± .0048** | CroVCA .5727 | **+.0739** | .3160 | .3117 ± .0035 |
| NUS-WIDE | **.7368 ± .0026** | OH .6772 | **+.0596** | .4822 | .4784 ± .0027 |
| CIFAR-10 | **.8903 ± .0104** | CroVCA .8263 | **+.0640** | .2929 | .2847 ± .0236 |

- A-champion 대비 이득: `+.0101 / +.0322 / +.0239 / +.0227`. **4/4 개선이며 seed 분산의 2–19배**다.
- `shuffled ≈ majority`가 전 데이터셋에서 성립하므로 probe 자체는 정상 동작한다.
- 이 표가 논문의 해석성 주 근거다. 단, decoder accuracy가 human understanding이나 인과적 제어와
  같지 않다는 §5.3의 범위 제한을 유지한다.

### 4.8 Causal ablation — P0 + bio-projected 동일 프로토콜

이전 draft의 A1/A2/A4 행은 모두 pre-P0 / pre-bio 진단값이었고 draft 스스로 "main causal table에
사용하지 않는다"고 적어 두었다. 아래는 신규 모델 위에서 동일 프로토콜로 재실행한 것이다.

#### A2. Text supervision 제거 (`--disable_text_supervision`)

| Dataset | full mAP@R | no text | **Δ mAP@R** | full decode | no text | **Δ decode** |
|---|---:|---:|---:|---:|---:|---:|
| MS-COCO | .8232 | .7627 | **−.0605** | .6466 | .5606 | **−.0860** |
| CIFAR-10 | .8940 | .8743 | −.0197 | .8903 | .8779 | −.0124 |
| NUS-WIDE | .8283 | .8099 | −.0184 | .7368 | .6941 | **−.0427** |
| Flickr25K | .8668 | .8542 | −.0126 | .7694 | .7491 | −.0203 |

🟢 **텍스트 감독은 4/4에서 필수적이며, 검색보다 해석성에 더 크게 기여한다.**
MS-COCO `−.086` / NUS-WIDE `−.043`처럼 decode 손실이 mAP 손실을 크게 웃도는 데이터셋이 있다
(NUS는 2.3배). 이는 **"text가 codon의 의미 조직을 만든다"는 인과 주장의 직접 근거**다.
구 draft의 A2 행은 pre-P0/pre-bio 진단값이었고, 여기서 처음으로 동일 프로토콜 수치가 된다.

#### A4. Shared codebook (6 slots × K=128 → 단일 K=768)

| Dataset | separate | shared | **Δ mAP@R** | separate decode | shared | **Δ decode** |
|---|---:|---:|---:|---:|---:|---:|
| Flickr25K | .8668 | .8521 | −.0147 | .7694 | .7506 | −.0188 |
| CIFAR-10 | .8940 | .8794 | −.0146 | .8903 | .8850 | −.0053 |
| MS-COCO | .8232 | .8140 | −.0092 | .6466 | .6154 | **−.0312** |
| NUS-WIDE | .8283 | .8252 | −.0031 | .7368 | .7240 | −.0128 |

🟢 **슬롯별 독립 codebook이 4/4에서 필요하다.** 검색 `−.003 … −.015`, 해석성 `−.005 … −.031`.
⚠️ 구 draft 시점에는 **NUS-WIDE에서 shared가 검색을 앞섰으나**(A-champion `.8262 → .8301`),
신규 레시피에서는 역전이 사라지고 separate가 우세하다(`−.0031`). `L_joint`가 슬롯별 codon 공간을
정리하면서 공유 bank의 이점이 없어진 것으로 해석된다.

#### A5. 신규 손실 항의 기여 (본 논문에서 새로 추가한 ablation)

| 조합 | MS-COCO mAP@R | codon decode | slot0 codons |
|---|---:|---:|---:|
| 없음 (A-champion) | .8170 | .6144 | 21 / 64 |
| `+ L_joint` only | .8227 | .6455 | 63 / 64 |
| `+ noGumbel` only | .8201 | .6283 | 21 / 64 |
| **`+ 둘 다` (unified)** | **.8287** | .6413 | **64 / 64** |

- 두 항의 효과는 **가산적**이다(`+.0057` 와 `+.0031` → `+.0117`).
- `--no_gumbel_softmax`는 **단독으로는 데이터셋에 따라 해롭다**(Flickr `−.0042`, CIFAR `−.0101`).
  `L_joint`와 결합할 때만 3/3에서 유익하다. 단일 인자 결과로 조합을 예단할 수 없다는 점을 명시한다.
- **`--codon_input_source routed`는 배제한다**: 검색 `−.0141`, 해석성 `−.0153`. 다만 DNA-unique는
  `.4696`(CroVCA `.457` 초과)으로 최고치이므로 diversity 전용 variant로만 부기한다.

### 4.9 K × bases-per-slot grid

| Dataset | K64, L3 | K64, L4 | K128, L3 | K128, L4 |
|---|---:|---:|---:|---:|
| Flickr25K | - | - | **.8668** | - |
| MS-COCO | - | - | **.8232** | - |
| NUS-WIDE | - | - | **.8283** | - |
| CIFAR-10 | **.8940** | - | - | - |

- 16 cell 중 4 cell만 신규 모델로 완료. global optimum·length scaling은 grid 완료 후 서술한다.
- A-champion 시절의 24-base 진단값은 레시피가 다르므로 신규 모델 행과 같은 표에 두지 않는다.

### 4.10 Analysis — 신규 모델에서 재측정

#### 4.10.1 Inter-codebook 중복 (pairwise NMI, 낮을수록 좋음)

| Dataset | mean off-diagonal NMI | max | min | A-champion 참조 |
|---|---:|---:|---:|---|
| Flickr25K | **.5606** | .6379 | .4712 | ~.604 |
| NUS-WIDE | .6079 | .6814 | .5028 | — |
| MS-COCO | .6719 | .7341 | .6143 | ~.658 |

Flickr는 개선(`.604 → .561`), MS-COCO는 소폭 악화(`.658 → .672`). 일관된 방향이 아니므로
"중복을 줄인다"는 일반 주장은 하지 않는다.

#### 4.10.2 Codebook-drop (informative budget)

| Dataset | baseline full mAP | Σ drop | anti-codebook | 슬롯별 Δ |
|---|---:|---:|---:|---|
| MS-COCO | .6086 | −.0558 | 0 | −.014 / −.011 / −.009 / −.009 / −.011 / −.003 |
| NUS-WIDE | .5952 | −.0476 | 1 | −.005 / −.012 / −.011 / −.011 / −.010 / +.000 |
| Flickr25K | .7626 | −.0418 | 0 | −.011 / −.006 / −.005 / −.006 / −.002 / −.012 |

**모든 슬롯이 기여한다.** 제거가 이득인 슬롯(anti-codebook)은 0–1개이며, 그 1개도 `+.0002`로
사실상 0이다. 여섯 슬롯 구성이 낭비가 아님을 보이는 직접 근거다.

#### 4.10.3 Slot intervention — 🔴 결론 불변

| Dataset | ours target gain | random slot | 비율 | selectivity |
|---|---:|---:|---:|---:|
| Flickr25K | .0259 | .0168 | **1.54×** | .0010 |
| NUS-WIDE | .0160 | .0094 | **1.69×** | −.0059 |

A-champion 기록(`~1.6×`, selectivity `≈0`)과 **사실상 동일하다.** codon 붕괴를 제거하고 held-out
decoding을 4/4 개선했음에도 **슬롯을 독립적으로 제어할 수 있다는 주장은 여전히 지지되지 않는다.**
이 한계는 §5.2에 그대로 유지한다.

#### 4.10.4 codon 붕괴 지표 (본 논문에서 새로 도입)

codeword→codon 사상은 `codon_input_source=quantized`·`codon_residual_gamma=0`에서 codeword의
결정론적 함수이므로 K개 codeword를 전수 열거해 정확히 재구성할 수 있다.

| Dataset | A-champion slot0 codons | **unified slot0** | A-champion gap | **unified gap** |
|---|---:|---:|---:|---:|
| MS-COCO | 21 / 64 | **64** | −27.6 | **+8.9** |
| NUS-WIDE | 21 / 64 | **63** | −27.1 | **+6.7** |
| Flickr25K | 26 / 64 | **43** | −16.1 | **+5.9** |
| CIFAR-10 | 26 / 64 | **50** | −11.7 | **+6.2** |

`gap` = 관측 codon 수 − 해당 슬롯의 위치별 marginal이 독립일 때 기대되는 codon 수. A-champion에서는
global 슬롯만 자기 marginal 예산을 크게 밑돌았고(3-way 종속), 신규 손실이 이를 4/4에서 제거한다.

#### 4.10.5 아직 측정하지 않은 항목

- UOT transported mass와 foreground/part mask overlap
- adaptive top-p effective support-size histogram
- train text anchor vs inference codebook anchor 일치도
- codon concept purity, seed 간 codon dictionary 안정성
- query별 retrieval gain과 caption 품질의 상관
- DP edit 위치의 슬롯별 분포
- latency / memory

### 4.11 필수 추가 ablation 우선순위

- P0 + bio projection + 3 seeds
  - clean EOS/full-token model
  - no text
  - shared codebook
  - NUS A4와 CIFAR decoding
- Router
  - mean pooling vs balanced OT vs full UOT
  - \(\lambda_a,\lambda_b\), \(\varepsilon\), Sinkhorn iterations
  - adaptive top-p 제거/fixed top-k/fixed top-p
  - `one_minus_cos` vs `neg_cos` — UOT에서는 상수 shift 비동치
- Text supervision
  - global caption only / local captions / six axes
  - Qwen prompt V4 vs V5b
  - Qwen/CLIP swap 또는 smaller VLM
  - missing-caption mask fix 전/후
  - train anchor vs codebook-mean anchor
- Codebook/codon
  - 1 vs 6 codebooks
  - K=64/128, L=3/4
  - EMA decay, revival off, codeword-codon Sinkhorn on/off
  - global gate on/off 및 stop-gradient
- Bio projection
  - no projection / GC only / homopolymer only / joint DP
  - codon-boundary-preserving DP
  - post-hoc baseline과 native constraint-aware training 비교
- Backbone
  - CLIP ViT-B/16 vs SigLIP2 under identical pipeline
  - frozen vs light adapter tuning

### 4.12 Efficiency 보고 계획

- training
  - offline Qwen caption GPU-hours
  - CLIP cache 시간·disk size
  - trainable parameter 수와 refit wall-clock
- inference
  - frozen CLIP image encode time
  - UOT 20 iterations latency
  - VQ/codon/DP projection latency
  - text tower/Qwen 호출 0회 확인
- storage/search
  - 18 bases/item = raw 36-bit symbol storage
  - codebook memory \(6K\times768\)
  - base-Hamming exhaustive search time와 packed implementation
  - binary baseline과 동일 database size에서 latency 비교
- 아직 측정되지 않은 값은 모두 `-`

## 5. Discussion

### 5.1 핵심 해석 (2026-08-04 재평가 반영)

- **무엇이 개선되었나**
  - 여섯 슬롯 전부에서 codon 결합 붕괴가 제거되었다(global 슬롯 21/26 → 43–64칸, gap 음수 → 양수).
  - held-out codon decoding이 4/4에서 개선되었고(`+.010 … +.032`), seed 분산의 2–19배로 안정적이다.
  - DNA-unique가 4/4에서 개선되었다. Flickr는 `.398 → .468`로 본 모델 계열 최고치다.
- **무엇이 개선되지 않았나**
  - 검색은 3-seed에서 2승 2패다. CIFAR-10은 A-champion의 우위를 잃었다.
  - **slot intervention selectivity는 여전히 ≈0**이다(1.54–1.69× random, selectivity `.001/−.006`).
    조합적 코드가 곧 독립 제어 가능한 factor를 뜻하지는 않는다.
  - 사전 bio-유효율은 40–57 %로 균등 기대치 수준이다. 제약은 학습되지 않고 사후 투영으로만 만족된다.
- **codon의 의미**
  - nucleotide triplet 자체가 의미를 만들지 않는다. slot role, text-aligned codebook, held-out
    dictionary가 함께 확률적 해석을 제공한다.
- **deployment**
  - text는 privileged training signal이고 inference는 image-only다. train/inference anchor
    mismatch는 동시에 핵심 한계다.

### 5.2 한계

- **검색과 조합성이 동시에 최적화되지 않는다.** 신규 레시피는 조합성·해석성 3축을 4/4 개선하는
  대신 CIFAR-10 검색에서 baseline 우위를 잃는다. 단일 모델로 두 축을 모두 최대화하지 못했다.
- **slot intervention selectivity ≈ 0** — 독립 제어 가능한 semantic factor를 주장할 수 없다.
- **codon 결합 붕괴의 원인 미규명** — `L_joint`는 증상 규제다.
- **생물 제약이 학습되지 않는다** — 사전 유효율이 chance 수준이며 constraint-aware 학습은 향후 과제.
- baseline의 seeds `{43,44}` 재실행 미완료 → 현재 상대 순위는 baseline single-seed 대비 diagnostic.
- legacy CLIP cache provenance 미완 → invariant #6상 baseline 행은 strict-main ineligible.
- GroundedDNA 3-seed는 확보했으나 query bootstrap CI는 미산출.
- CIFAR caption generator provenance 미확인; sample-wise missing-caption mask bug 가능성.
- K=128에서 codeword–codon collision은 pigeonhole상 불가피.
- DP projection이 codon 경계와 semantic assignment를 변경한다.
- 슬롯 간 redundancy가 크며(A-champion 기준 65 %), 이는 제거 대상이 아니라 load-bearing으로 확인되었다.
- VLM hallucination·bias·closed semantic-axis 설계.
- 18-base code에 primer/address/ECC/motif/hairpin 제약 없음; wet-lab 검증 없음.
- multi-label validation이 iterative stratification이 아닌 random split.
- **V5b 프롬프트의 재생성 경로가 코드에 없다** (`preprocess_qwen_codebook_texts.py`는 v1–v4만 지원).
  재현성 관점의 미해결 항목.

### 5.3 Threats to validity

- protocol validity
  - 동일 dataset 이름이라도 subset/train/query/R가 다른 published number와 직접 비교 위험
- capacity validity
  - 18 bases의 raw 36 bits와 \(K^6\) latent tuple capacity 차이
  - codebook memory를 code length에서 숨기지 않음
- interpretability validity
  - decoder accuracy가 human understanding이나 causal control과 동일하지 않음
  - qualitative atlas만으로 claim하지 않음
- biochemical validity
  - GC/homopolymer 만족이 synthesis 성공을 보장하지 않음
- selection validity
  - historical test-selected diagnostic를 leakage-free P0 main result와 분리

## 6. Conclusion 뼈대

- GroundedDNA의 문제 재진술
  - 빠른 image retrieval code를 semantic parts의 조합이자 DNA-valued sequence로 설계
- 기술 요약
  - VLM caption supervision, CLIP patch UOT, EMA multi-codebook, codon decoder, DP projection
- 결과 요약
  - 네 benchmark의 controlled post-projection mAP와 held-out decoding
- 보수적 결론
  - 완전한 disentanglement나 molecular readiness가 아니라 semantic organization과 feasibility의 공동 가능성 제시
- 향후 연구
  - codon-preserving constrained decoding
  - constraint-aware training
  - native molecular retrieval/wet-lab validation
  - open-vocabulary/dynamic semantic slots

## Appendix 구성 계획

- A. 전체 notation과 tensor-shape table
- B. Qwen prompts, decoding config, cache schema/hash, failure rows
- C. CLIP preprocessing, augmentation, whitening derivation
- D. UOT first-order condition과 generalized Sinkhorn proof
- E. fixed-support top-k KL projection proof
- F. EMA update와 dead-code revival pseudocode
- G. bio projection optimality proof 및 complexity
- H. 모든 loss의 exact implementation equation과 active/inactive audit
- I. split manifests, P0 selection curves, E*와 schedule의 실제 값
- J. pre/post projection full metrics: mAP, P@1/5/10/100, unique, edits, validity
- K. per-seed table와 confidence intervals
- L. code atlas, routing maps, collision/failure cases
- M. source-code/result artifact mapping

## 제출 전 체크리스트

- [ ] Clean EOS/full-token P0 네 데이터셋 재실행
- [ ] 3 seeds와 bootstrap CI
- [ ] baseline clean P0 one-shot rerun 또는 historical test-curve 사용 명시
- [ ] K/L 12개 미완료 셀
- [ ] P0 projected A2/A4 및 NUS A4
- [ ] sample-wise caption mask 수정/ablation
- [ ] CIFAR Qwen provenance 복원 또는 cache 재생성
- [ ] multi-label iterative stratification 여부 결정 후 manifest 고정
- [x] 최신 baseline supervision audit: `U0/U1/U2/S` 분리
- [x] GreedyHash·Bi-half·SDC·OH·HHCH·CroVCA·DUH-EG·UMRCH clean-room matched runner 구현·unit/checkpoint protocol 검증; 신규 2개도 Flickr25K 1-epoch CPU에서 E* selection→scratch refit→query/database extraction→bio-projection→SHA manifest 전 경로 통과(`--allow-main-ineligible-smoke`, 비표준 horizon/cadence·legacy cache로 main-ineligible, full-horizon 수치는 미실행)
- [ ] `U0` 36-bit 3-seed full 실행: GreedyHash, Bi-half, SDC 세 variant, OH, HHCH paper variant, CroVCA frozen-probe variant
- [ ] DUH-EG: 저자 ordered selected-WordNet bank provenance 확정→`U1` 승격 후 36-bit 3-seed 실행; 그 전 `U?`/main-ineligible
- [ ] default legacy CLIP cache를 현재 extractor로 재생성(transform·augmentation seed·immutable HF provenance·consumed-array SHA contract 충족)
- [ ] legacy bio-projection JSON을 immutable protocol/checkpoint/extraction provenance와 versioned tie policy로 재생성하고 optimal-tie/order sensitivity 추가
- [ ] `U2` UMRCH: official ordered taxonomy SHA 검증은 완료; 별도 표 3-seed full 실행
- [ ] FSCH 저자에게 누락 trainer/loss/config 문의; 확보 전 exact row는 `-`
- [ ] UHSCM/TGUH는 public-code defect·caption pipeline·test-selection 해소 후만 실행
- [ ] CUB/fine-grained 별도 track: A²-SSL, FAPI, CS3H 수동 이식/실행
- [ ] supervised upper-bound 별도 표: FTH, MambaHash, DSCH-2026, DGrH
- [ ] ConceptHash/LPAH/HTH 동일 protocol baseline
- [x] DNA24/PRIMO/Koike DATE·DAC/TCBB clean-room computational core 구현
- [x] DNA24-18/PRIMO-18/Koike-TN-DNA/Koike-BC-TN-DNA sealed P0 3-seed diagnostic 48/48 완료; strict 0/48
- [x] PRIMO official `pub` predictor artifact exact conversion·weight 검증
- [ ] provenance-complete cache에서 native-DNA strict 48-cell 재실행
- [ ] PRIMO 18-mer thermodynamic yield 재생성·predictor calibration
- [ ] DNA24 original 30-nt computational sanity reproduction
- [ ] Koike original 80-base CIFAR classification sanity reproduction
- [ ] Koike/PRIMO 저자에게 공식 code license와 누락 artifact 문의
- [ ] 24-base P0와 48-bit baseline 공정 비교
- [ ] UOT localization·mass·support analysis
- [ ] efficiency/parameter/cache-cost 측정
- [ ] codon dictionary seed stability와 human-readable concept study
- [ ] codon-boundary-preserving bio projection 비교
- [ ] wet-lab 미수행 범위와 용어 최종 점검

## 참고문헌 정리 방침

- 본문 핵심 인용
  - deep hashing: DSH, CIBHash, CIMON, MLS3RDUH, DDCH, SDC, OH, CGHash, HHCH, CTMIH, FSCH, CroVCA, DUH-EG, UMRCH, OMUH, TGUH
  - non-binary/PQ 경계: HiHPQ는 중요한 U0 retrieval 선행이지만 asymmetric product-quantization distance이므로 binary/Hamming main table과 분리
  - compositional/non-binary: PQ, SUBIC, VQ-VAE, Uni-Code, quaternary hashing, ConceptHash
  - text supervision/interpretability: CLIP, VirTex, RegionCLIP/GroupViT, WDHT, Dual Purpose Hashing, A²-Net, LPAH
  - DNA retrieval: Stewart 2018, Bee 2021, Koike DATE/DAC 2024와 TCBB 2026, Cas9 2025, Pradhan 2023/2025
  - OT: Cuturi 2013, Liero/Chizat 2018, UNITER/VoLTA, sparse OT, DOT-CBM, concurrent ConceptOT/OMIT
  - bio constraints: HEDGES, DNA-Aeon, constrained DNA codes
- supplementary literature table
  - 검색식별 검색일, DB, hit 수, screening 사유, code/data URL 기록
  - peer-reviewed / workshop / preprint 상태를 분리
  - 직접 인용 문장 없이 원 논문 내용을 paraphrase
- 최종 bibliography 검증
  - DOI, venue, year, author order를 Crossref/공식 publisher에서 재확인
  - 2026 논문의 final volume/page와 preprint version 갱신
