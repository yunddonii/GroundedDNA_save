# GroundedDNA: 언어로 정초된 조합적 DNA 코드를 이용한 이미지 검색

> 논문 뼈대 초안 · **2026-08-04 재평가판** · 본문 문장화 전 개조식 문서
>
> 이 문서는 `DRAFT_GROUNDEDDNA_PAPER_KO.md`(2026-07-25 스냅샷)의 실험 절을 **신규 통일
> 레시피 모델의 재실험 결과로 교체**한 버전이다. 교체된 절: §0 결론, **§3 전체(아키텍처 서술로
> 재작성; 데이터셋별 레시피는 §4.2b로 이동)**, §4.5–§4.10, §5. 구 수치는 대부분 pre-P0/pre-bio 또는 구 레시피 기반이라
> 원본 draft가 스스로 "main table 사용 금지"로 표시해 둔 것들이다.
> 재평가 요약: [`newmodel_analysis/SUMMARY_newmodel_reevaluation.md`](./newmodel_analysis/SUMMARY_newmodel_reevaluation.md)

## 0. 작성 원칙과 현재 결론

- 한 문장 문제 정의
  - ground-truth label을 gradient objective에 넣지 않고 VLM이 만든 구조화 caption을 학습 시 semantic teacher로 사용
  - 이미지 한 장을 다섯 의미 슬롯의 조합으로 분해
  - 각 슬롯을 독립 EMA codebook에서 양자화하고 3-base codon으로 변환
  - 추론 시 text 없이 15-base DNA-valued code를 생성하고 생물학적 제약으로 투영하여 검색
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
  - Qwen이 생성한 여섯 semantic-axis caption 중 다섯(`scene_type` 제외)을 privileged supervision으로 사용
  - frozen CLIP patch와 local caption anchor 사이 KL-relaxed UOT routing
  - 다섯 독립 EMA codebook과 slot-conditioned 3-base codon head
  - image-only inference와 최소 base-Hamming bio projection
- 결과
  - 동일 15-base 공간과 동일 bio projection 아래 mAP@R
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
  - 선택된 slot codeword를 slot-conditioned head로 3-base 단위에 decode하고 다섯 단위를 연결; codeword–codon 일대일 대응은 보장하지 않음
  - Pradhan 계열은 pixel/MSB-derived DNA plane에서 codon·amino-acid feature를 구성하지만, 본 연구는 genetic translation table 없이 language-defined slot을 학습된 codon에 기록
- 문단 7 — GroundedDNA 개요
  - Qwen caption → CLIP text anchor → patch-to-slot UOT → EMA VQ → codon head → bio projection
  - 학습 시 text teacher, 추론 시 image only
  - representation-learning gradient에는 relevance label을 사용하지 않으며, held-out validation/test label은 \(E^*\) 선택과 공식 평가에만 사용
- 문단 8 — 핵심 실험 메시지
  - 네 표준 image hashing benchmark
  - 모든 방법을 동일 15-base·base Hamming·동일 bio projection으로 평가
  - 검색 성능, bio feasibility, held-out decoding, code semantics를 함께 보고

### 1.2 Main Contributions 초안

- **Text-grounded compositional DNA hashing**
  - 다섯 language-defined slots와 독립 codebook/codon head로 구성된 15-base retrieval representation
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
  - 논문의 32/64-bit 수치를 30-bit와 직접 비교하거나 보간하지 않음
  - 모든 후보를 정확히 30-bit로 재학습한 뒤 `2 bits↔1 base` 변환과 동일 DP projection을 적용
  - 동일 split·frozen CLIP backbone·30-bit capacity·15-base evaluator를 사용하되, 방법 정의에 필요한 global/local/two-view/text 입력은 숨기지 않고 열로 표시
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
| [ConceptOT, CVPRW 2026](https://openreview.net/pdf?id=EU0tuTbrKn) | CLIP patch–concept low-rank UOT | concurrent closest work; DNA hashing·fixed five slots·bio projection 차이 |

- 용어 선택
  - 본문: **KL-relaxed entropic unbalanced optimal transport**
  - `PSOT`를 쓸 경우 프로젝트 variant 이름으로만 정의
  - fixed-mass partial OT의 `$P\mathbf 1\le a$`, `$P^\top\mathbf 1\le b$`, 총질량 제약을 만족한다고 주장하지 않음
- adaptive top-p의 위치
  - UOT 해 이후의 row-wise sparsification heuristic
  - 전체 sparsity-constrained UOT의 exact optimizer라고 주장하지 않음

## 3. Methodology

### 3.1 문제 정의와 설계 원리

**과제.** 라벨 없이 이미지 집합 \(\mathcal D=\{x_i\}\)만 주어졌을 때, 각 이미지를
길이 \(L\)의 DNA 서열로 사상하는 인코더 \(h_\theta:\mathcal X\rightarrow\mathcal A^{L}\),
\(\mathcal A=\{A,C,G,T\}\)를 학습한다. 검색은 염기 단위 Hamming 거리
\(d(y,y')=\sum_{\ell}\mathbf 1[y_\ell\ne y'_\ell]\)로 수행한다.

**왜 이진 해시의 확장이 아닌가.** 4-문자 알파벳은 비트당 용량이 2배라는 점에서만 다른 것이
아니다. DNA 저장·합성이 요구하는 제약 — GC 함량 구간, homopolymer 길이 상한, 서열 간 최소
Hamming 거리 — 은 **염기 위치 사이의 결합 조건**이다. 비트별로 독립인 목적함수로는 표현할
수 없으며, 사후에 강제하면 코드가 손상된다. 따라서 제약은 표현 설계 단계에서 함께 다뤄야 한다.

**설계 원리.** 코드가 해석 가능하려면 *"코드 전체가 무엇을 뜻하는가"*가 아니라
**각 심볼이 고정된 의미 축에 묶여 있어야** 한다. 이를 위해 코드를 \(M\)개의 **슬롯**으로
분해하고, 각 슬롯에 **언어로 정의된 의미 축**을 부여한다:

\[
h_\theta(x)=[\,c^{1};c^{2};\ldots;c^{M}\,],\qquad c^{m}\in\mathcal A^{\ell},\quad L=M\ell .
\]

슬롯 \(m\)의 코돈 \(c^m\)은 그 축에 관한 진술이고, 전체 코드는 그 진술들의 **조합**이다.
이것이 본 논문이 주장하는 조합적 해석가능성이며, 이후 모든 설계 결정은 이 한 문장에서 나온다.

세 가지가 따라온다.

1. 의미 축은 **모델 밖에서 언어로 고정**되어야 한다 — 학습으로 발견된 클러스터는 사후
   해석이 필요하지만, 미리 문장으로 정의된 축은 그렇지 않다.
2. 각 슬롯은 이미지의 **자기 축에 해당하는 영역**만 봐야 한다.
3. 슬롯들의 코드북은 **서로 독립**이어야 한다 — 공유하면 심볼의 의미가 슬롯에 따라 달라진다.

---

### 3.2 프레임워크 개요

```text
 [offline]  image ──frozen VLM──> M axis captions ──frozen text encoder──> axis anchors
            image ──frozen vision encoder──> patch tokens + global feature

 [encode]   global feature ─────────────────────────────> slot 1 (global)
            patches × axis anchors ──text-guided UOT──> slots 2..M (local)
            each slot ──its own VQ codebook──> codeword
            codewords ──codon heads──> M × ℓ bases

 [deploy]   raw code ──DP projection onto the biologically valid set──> emitted code
            query vs database ──base-wise Hamming──> ranking
```

시각·언어 인코더는 **동결**한다. 학습되는 것은 어댑터, 라우팅, 코드북, 코돈 헤드뿐이다.
이는 계산 비용 때문이 아니라, **의미 축의 정의를 학습에서 분리**하기 위해서다. 축이 학습과
함께 움직이면 "슬롯 \(m\)이 축 \(m\)을 뜻한다"는 주장이 순환한다.

추론 시에는 캡션이 필요 없다. 텍스트는 **학습 시 슬롯의 의미를 고정하는 교사**로만 쓰이고,
배포된 인코더는 이미지만 받는다.

---

### 3.3 언어로 정의된 의미 축

동결된 VLM에 이미지를 주고, 미리 정한 \(M\)개 질문에 답하게 하여 축별 서술
\(T=\{t^{m}\}_{m=1}^{M}\)를 얻는다. 축은 **전역 요약** 하나와 **국소 축** \(M-1\)개로
구성한다. 본 논문은 \(M=5\)를 쓰며, 국소 축은 다음 넷이다:
**주 객체**(`primary_object`), **부차 객체**(`secondary_object`),
**행위·관계**(`activity_relation`), **색·질감**(`color_texture`).

두 가지가 본질적이다.

- **축은 데이터가 아니라 질문으로 정의된다.** 라벨을 쓰지 않으므로 비지도 설정이 유지되고,
  축의 의미는 데이터셋이 바뀌어도 동일하다.
- **캡션은 오프라인에서 한 번만 생성한다.** VLM은 학습 그래프에 들어가지 않으며 추론에도
  관여하지 않는다.

축별 서술을 동결 텍스트 인코더로 임베딩해 **축 앵커** \(a^{m}\)를 얻는다. 앵커는 다음 절의
수송 비용을 정의하는 유일한 언어 신호다.

> **축 설계가 방법의 유효 범위를 결정한다.** 이미지가 어떤 축을 지지하지 못하면 — 예컨대
> 저해상도 단일 객체 이미지에서 "부차 객체" — 그 축의 앵커는 다른 축과 구별되지 않는다.
> 축 간 앵커 유사도는 본 프레임워크가 언제 작동하는지를 예측하는 진단량이며 §4에서 정량화한다.

---

### 3.4 텍스트 유도 수송: 패치를 슬롯에 배분

전역 슬롯은 전역 이미지 특징에서 직접 만들고, 국소 슬롯은 **패치를 축 앵커로 수송**하여 만든다.

패치 \(v_n\)과 앵커 \(a^m\) 사이의 비용을 코사인 거리로 두고,

\[
C_{nm}=1-\frac{v_n^\top a^{m}}{\lVert v_n\rVert\,\lVert a^{m}\rVert},
\]

엔트로피 정규화된 **불균형 최적수송**으로 수송 계획 \(P\in\mathbb R_{\ge0}^{N\times(M-1)}\)를 푼다:

\[
\min_{P}\;\langle P,C\rangle+\varepsilon H(P)
+\lambda_a\,\mathrm{KL}(P\mathbf 1\,\Vert\,\mathbf a)
+\lambda_b\,\mathrm{KL}(P^{\!\top}\mathbf 1\,\Vert\,\mathbf b).
\]

**균형 OT가 아니라 불균형 OT를 쓰는 이유**는 모든 패치가 어떤 축엔가 속해야 할 이유가 없기
때문이다. 배경·흐림처럼 어느 축과도 잘 맞지 않는 패치는 행 주변부 제약을 완화하여 **부분적으로
기각**될 수 있어야 한다.

수송 계획에 이어 패치별 **confidence-adaptive top-p mask**를 적용한다.[^topp] 각 패치에
대해 정렬된 배분 확률의 누적합이 임계 \(\tau\)에 이를 때까지의 슬롯만 남기고 나머지를 0으로
만든 뒤 행을 재정규화한다. 임계는 그 패치의 배분 확신도 \(p_{\max}\)에 따라

\[
\tau=\tau_{\min}+(1-p_{\max})(\tau_{\max}-\tau_{\min})
\]

로 조절되어, 확신 있는 패치는 소수 슬롯에, 모호한 패치는 여러 슬롯에 기여한다. 각 패치는
자신의 1순위 슬롯을 항상 유지한다. 이는 각 슬롯이 **자기 축과 관련된 영역만 모으도록**
강제하는 장치다.

[^topp]: 이름은 언어 모델 생성의 nucleus sampling(Holtzman et al., ICLR 2020)에서
    차용했으며 **생물학적 의미는 없다**. 정확한 구현은 공개 코드
    `models/semantic_router.py`를 참조하라.

슬롯 표현은 남은 계획으로 가중 평균한 값이다:

\[
s^{m}=\frac{\sum_n P_{nm}\,v_n}{\sum_n P_{nm}} .
\]

라우팅이 텍스트에 의존하는 것은 학습 시뿐이다. 추론에서는 캡션 대신 학습된 코드북의 축별
평균을 앵커로 사용하므로, 배포 인코더는 이미지만으로 동작한다.

---

### 3.5 슬롯별 독립 양자화

각 슬롯 표현을 **그 슬롯 전용 코드북** \(\mathcal C^{m}=\{e^{m}_k\}_{k=1}^{K}\)로 양자화한다:

\[
k^{m}=\arg\min_k\lVert s^{m}-e^{m}_k\rVert_2,\qquad q^{m}=e^{m}_{k^{m}} .
\]

코드북은 슬롯마다 **분리**되어 있다. 공유하면 같은 인덱스가 슬롯에 따라 다른 것을 뜻하게 되어
§3.1의 원리가 무너진다. 코드북은 EMA로 갱신하고, 사용률 균등화 항으로 붕괴를 방지한다.

---

### 3.6 코돈 합성

각 슬롯의 코드워드를 \(\ell\)개 위치의 4-way 분포로 사상하고 argmax로 염기를 얻는다:

\[
u^{m}=\mathrm{CodonHead}_m(q^{m})\in[0,1]^{\ell\times4},\qquad
c^{m}_j=\arg\max_{\alpha\in\mathcal A}u^{m}_{j\alpha}.
\]

코돈 헤드는 슬롯마다 독립이며, 전역 슬롯의 코드워드를 게이트를 통해 국소 헤드에 함께 넣어
전역 문맥을 조건으로 준다. 이는 같은 국소 코드워드라도 장면 전체가 다르면 다른 코돈을 낼 수
있게 한다.

> 이 게이트는 국소 슬롯이 시각 증거를 충분히 받지 못할 때 **전역 정보가 그 슬롯의 코돈을
> 대신 결정할 수 있는 경로**이기도 하다. §4에서 이 경로가 실제로 열리는 조건을 측정한다.

---

### 3.7 생물학적 유효성

배포되는 코드는 생물학적 제약 집합
\(\mathcal V=\{y:\text{GC}(y)\in[g_{\lo},g_{\hi}],\ \text{run}(y)\le r\}\)
안에 있어야 한다. 원시 코드 \(y\)를 동적 계획법으로 \(\mathcal V\) 안의 최근접 서열로 투영한다:

\[
\hat y=\arg\min_{y'\in\mathcal V} d(y,y').
\]

투영은 **모든 보고 지표에 적용**된다. 유효성을 만족한 상태에서의 성능만이 배포 가능한 성능이기
때문이다. 학습 중에는 GC 힌지와 homopolymer 항으로 원시 코드를 미리 유효 영역 쪽으로 밀어
투영이 코드를 크게 바꾸지 않도록 할 수 있으며, 이 항의 유무는 §4에서 별도 행으로 보고한다.

---

### 3.8 학습 목적

목적함수는 세 갈래이며, 각각 §3.1의 원리 중 하나에 대응한다.

**(a) 축 정렬 — 슬롯이 자기 축을 뜻하게 한다.**
네 항이 서로 다른 표현 수준에서 작동한다.

- **교차 모달 커밋**은 슬롯 표현 \(s^m\)과 축 앵커 \(a^m\)을 **양방향**으로 맞춘다.
  각 방향의 표적은 상대 모달리티의 **양자화된** 토큰이며 stop-gradient가 걸린다:
  \(\lVert s^m-\mathrm{sg}[q_t^m]\rVert^2\) 와 \(\lVert a^m-\mathrm{sg}[q_v^m]\rVert^2\)
  의 평균. 두 번째 항은 앵커를 시각 쪽으로 이동시키므로, 앵커는 학습 중 고정된 상수가 아니다.
  축의 **정의**(캡션 문장과 그 동결 임베딩)는 고정된 채, 그 임베딩을 시각 공간으로 옮기는
  어댑터만 움직인다는 점에서 §3.2의 분리는 유지되지만, 이 경로의 존재는 명시해 둔다.
- **텍스트→코드 KL 증류**는 슬롯별로 코드북 \(\mathcal C^m\) 위의 \(K\)-way 분포를
  시각·텍스트 양쪽에서 만들고(각각 온도 \(\tau_v,\tau_t\)), \(\mathrm{KL}(p_t\Vert p_v)\)를
  최소화한다. \(p_t\)는 detach하며, 표본별 텍스트 확신도
  \(1-H(p_t)/\log K\)로 가중하고 임계 미만은 버린다 — 캡션이 모호한 표본이 코드를
  끌고 가지 못하게 하기 위해서다.
- **코돈 텍스트 앵커**는 텍스트 토큰과 코돈 프로토타입의 유사도 argmax를 의사 라벨로 삼아
  코돈 로짓에 교차 엔트로피를 건다. 즉 정렬을 **염기 수준**까지 내린다.
- **슬롯별 대조 학습**(이미지 간 InfoNCE)이 같은 축 안에서 이미지를 구별하게 한다.

수송 비용 \(\langle P,C\rangle\) 자체도 목적에 포함되어, 라우팅이 앵커에 가까운 패치를
모으도록 유도한다.

**(b) 양자화 — 코드가 이산 심볼이 되게 한다.**
VQ 커밋먼트, 코드북 사용률 균등화, 그리고 염기 위치별 사용 균형 항.

**(c) 코드 다양성 — 심볼이 실제로 구별되게 한다.**
슬롯별 코돈의 **결합 분포**를 균등 쪽으로 미는 규제를 둔다. 위치별 주변 분포만 균등해도
결합 분포는 소수 조합에 몰릴 수 있으므로, 규제는 \(4^{\ell}\)개 조합 위에서 정의한다.

\[
\mathcal L_{\rm joint}=\frac1{|S|}\sum_{m\in S}\mathrm{KL}\!\left(\mathcal U_{4^\ell}\,\Vert\,\bar J^{m}\right),
\qquad
\bar J^{m}=\mathbb E_{x}\Big[\textstyle\bigotimes_{j=1}^{\ell}u^{m}_{j}\Big].
\]

(a)와 (b)는 데이터 적합 항이고 (c)는 사전(prior)이다. 둘은 본질적으로 상충하며 — 코드가
데이터 구조를 담으면 결합 분포는 균등에서 멀어진다 — 이 상충의 균형점이 검색 성능과
해석가능성의 교환을 결정한다. §4에서 그 교환을 정량화한다.

---

### 3.9 추론

이미지를 동결 인코더에 넣어 전역 특징과 패치를 얻고, 학습된 코드북의 축별 평균을 앵커로
수송·마스킹을 수행하여 슬롯 표현을 만든다. 각 슬롯을 자기 코드북으로 양자화하고 코돈 헤드의
argmax로 원시 코드를 얻은 뒤, DP 투영으로 유효 코드를 방출한다. 질의와 데이터베이스 코드
사이의 염기 Hamming 거리로 순위를 매긴다. **텍스트도, VLM도 추론 경로에 없다.**


## 4. Experiments

### 4.1 Research Questions

- RQ1: 동일 bio-valid 15-base space에서 GroundedDNA가 binary hashing baseline보다 높은 retrieval accuracy를 보이는가?
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

### 4.2b 구현 세부 — 데이터셋별 레시피

> Methodology(§3)는 아키텍처만 기술한다. 아래 하이퍼파라미터는 실험 설정이므로
> 여기에 둔다.

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

### 4.2c Sinkhorn 엔트로피 계수 \(\varepsilon\)의 스케줄 — **학습 지평과 정렬**

\(\varepsilon\)은 학습 중 \(\varepsilon_{\rm init}\to\varepsilon_{\rm final}\)로 코사인
스케줄에 따라 어닐링된다. **스케줄의 지평은 실제로 학습하는 epoch 수 \(N\)과 일치시킨다**
(`-e` \(=N+1\)). 이는 재현에 필수적인 설정이며, 하이퍼파라미터 나열이 아니라 결과를 지배하는
요인이다.

이전 실험들은 명목상 `-e 60`으로 스케줄을 정의한 뒤 \(N=4\sim19\)에서 학습을 멈췄고, 그 결과
스케줄의 7–32 %만 소화되어 \(\varepsilon\)이 초기값 근처(0.49–0.99)에 머물렀다. 이 영역에서는
엔트로피 항이 수송 계획을 주변분포의 곱 쪽으로 평탄화하므로 **슬롯 열이 서로 달라질 수 없다.**
지평을 정렬하면 동일 계산량에서 다음이 얻어진다(5 슬롯, `final_epoch_eval` on):

| dataset | 슬롯 간 코사인 (마스크 후) | 텍스트 앵커 코사인 |
|---|---|---|
| | `-e 60`, 조기 정지 → **정렬** | `-e 60`, 조기 정지 → **정렬** |
| CIFAR-10 | .9880 → **.5095** | .4877 → **.4398** |
| Flickr25k | .9984 → **.5411** | .5285 → **.2528** |
| NUS-WIDE | .9204 → **.4711** | .2892 → **.2261** |
| MS-COCO | .9741 → **.4686** | .3652 → **.1878** |

mAP@R 변화는 −.022 이내이고 NUS-WIDE는 오히려 개선된다. \(\varepsilon_{\rm init}\)은
NUS-WIDE만 0.5, 나머지는 1.0이며 \(\varepsilon_{\rm final}=0.1\)로 공통이다.

**§3.4의 confidence-adaptive top-p mask도 이 조건에서만 작동한다.** 마스크 직전·직후의 슬롯
코사인 차이로 그 순수 기여를 재면, \(\varepsilon\)이 초기값 근처일 때는 −.0000 ~ −.0397에
그치고, 정렬 후에는 네 데이터셋 모두 −.1835 ~ −.1909로 거의 동일하다. 즉 마스크는 데이터셋이
아니라 \(\varepsilon\)에만 좌우되는 안정적 연산자이며, 평탄한 계획에는 잘라낼 구조가 없다.

**보고 시 주의.** \(N\)이나 \(E^*\)를 `-e` 없이 인용하면 해석할 수 없다. 두 값과 그때 도달한
\(\varepsilon\)을 함께 기재한다. 시각화·사후 분석에서도 체크포인트가 **학습된** epoch에
모델을 두어야 한다 — `_current_epoch`는 0으로 초기화되고 state dict에 저장되지 않으므로,
새로 로드한 체크포인트는 초기 \(\varepsilon\)에서 동작한다.

---

### 4.3 학습 프로토콜 — 고정 epoch 단일 단계 (P0 2단계에서 전환)

> **전환 고지 (2026-08-09 결정, 2026-08-11 갱신).** 아래 4.3.x는 이전 프로토콜인
> validation 기반 P0 2단계 선택/재학습의 기록이며, 감사 이력 보존을 위해 남긴다.
> **현행 프로토콜은 여기에 기술하는 단일 단계 고정 epoch 방식이다.**

**절차.**

- `--val_split_ratio 0.0` — validation split을 만들지 않는다. 중간 평가는 공식 test split
  **전량**에 대해 매 epoch 수행한다(P0 이전 경로).
- 2단계 재학습이 없다. **고정 epoch \(N\)의 체크포인트가 곧 보고 모델**이다.
- \(N\)은 데이터셋마다 경험적으로 정하고, `-e` \(=N+1\)로 두어 §4.2c의
  \(\varepsilon\) 스케줄이 그 안에서 완주하게 한다.

**전환 이유.** 세 가지가 P0를 지탱하지 못하게 했다.

1. \(E^*\)가 항상 **첫 중간 평가 지점**에 걸렸다. `--eval_every 5`는 epoch 4, 9, 14…에서
   발화하는데, 관측된 모든 \(E^*\)(4, 9, 14, 19, 24, 29, 39)가 \(\equiv 4 \pmod 5\)였다.
   `--eval_every 1`로 재측정하니 Flickr의 실제 val 정점은 e4가 아니라 **e3**이었다.
2. **검색과 해석가능성의 정점이 서로 다른 곳에 있는데 \(E^*\)는 검색만 보고 선택된다.**
   Flickr val에서 eval_mAP는 e3, unique-code-ratio는 e6, base entropy는 e37,
   per-codebook unique는 e47에서 정점이다.
3. **val→test 전이가 두 번 실패했다.** val 정점 e3은 test \(E^*\) 스윕에서 가장 나쁜 셀이었고
   (DNA-unique −.0542), `--proj_lr 3e-4`는 val 세 축을 개선하면서 test에서 slot0 코돈을
   43→27로 무너뜨렸다.

**누수, 명시적으로.** test 곡선을 보고 \(N\)을 고르는 것은 **데이터셋당 스칼라 하나**를
누수시킨다. 이를 그 하나로 제한하기 위해, \(N\)은 **seed 42 곡선에서만** 고르고 seed 43/44에는
변경 없이 그대로 적용한다.

**선택 축.** \(N\)은 네 축을 함께 보고 정한다: (i) bio-projected mAP@R, (ii) DB split 기준
DNA-unique 및 코드북 사용률, (iii) 죽은 코드워드 비율, (iv) **슬롯 간 라우팅 코사인**.
(iv)는 본 논문의 핵심 주장에 직접 대응하는 축이며, 초기 후보 선정에서 누락되어 있었다 —
그때 선택된 \(N\)들은 모두 \(\varepsilon=0.49\sim0.94\) 지점에 놓여 있었다.

**짧은 학습에 대하여.** 동결 CLIP 백본에서 4–10 epoch은 현재 관행이다. 본 논문의 baseline 중
하나인 CroVCA(CVPRW'26)는 "단일 GPU에서 5 epoch"을 기여로 제시하고, CLIP Multi-modal
Hashing은 loss가 45까지 계속 내려가는 동안 test mAP는 10 epoch 이후 평탄하다고 보고한다.
짧은 지평 자체는 결함이 아니며, 고쳐야 할 것은 그 지평과 \(\varepsilon\) 스케줄의 불일치였다.

---

#### 4.3.0 (이전 프로토콜 기록) Validation-based P0 selection/refit 및 test-use audit

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
  - 30-bit sign code \([N,30]\)을 \([N,15,2]\)로 reshape
  - `00→A`, `01→C`, `10→G`, `11→T`
  - query/database 양쪽 DP projection
  - projected base-Hamming으로 재평가
- 공정성 범위
  - 동일 raw capacity 30 bits ↔ 15 bases
  - 동일 frozen CLIP backbone, split, relevance, cutoff, bio constraints, distance
  - 단, method-defining 정보는 허용: global-only(GreedyHash/Bi-half/SDC/OH/DUH-EG/HHCH/CroVCA), global+local(UMRCH), fixed cached two-view/three-view, external noun/taxonomy bank 여부를 information-condition 열에 명시
  - `stage 1`: test split을 로드하지 않고 val-query vs opt-train DB의 **raw 2-bit→15-base base-Hamming mAP@R**로 (E^*) 선택
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
  - 공식 논문의 16/32/64-bit 숫자는 가져오지 않고 30-bit matched run만 main comparison에 기입
  - external CLIP asset는 immutable Hugging Face commit, model/tokenizer SHA, ordered term SHA, cache-meta SHA까지 manifest로 검증

#### 4.4.1 직접 선행 native-DNA baseline의 구현 및 보고 단위

- 결과를 두 층으로 분리
  - `original reproduction`: 원 feature, 30/80 nt, 원 query와 NUPACK/DDH 또는 wet-lab 지표
  - `matched adaptation`: 동일 cached CLIP feature, 동일 split, 15 bases, per-image query, 공통 DP, base-Hamming mAP@R
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
  - stage 1: `val_split.carve_val_indices(...,.1,42)`, **neural-raw 15-base Hamming** val mAP@R로 E* 선택, test 미로딩; projected-E* 선택은 별도 protocol ablation으로만 허용
  - stage 2: 전체 designated train에서 E*+1 epochs refit 후 test와 무관하게 checkpoint 1개를 고정; raw·post-projection terminal metric은 같은 checkpoint에서 계산하며 test-dependent 선택은 없음
  - artifact: `extract_{query,db}_neural_raw.npz`, standard pre-DP deployment `extract_{query,db}.npz`, projected files; train file은 `neural_raw_base_indices`를 함께 저장
  - metric key도 `neural_raw`, `paper_hp_postprocessed`(TCBB), `projected`로 분리; DP edit/compliance/unique ratio의 기준 stage를 필드명에 표시
- 구현 검증
  - hard one-hot에서 Koike soft distance = normalized base-Hamming 확인
  - `A,T,C,G→A,C,G,T` remap과 2-bit serialization 확인
  - query/DB 양쪽 exact DP 후 15-base GC count `[7,8]`·run≤3 확인
    (GC 분율 `[0.444, 0.556]`은 불변; 15-base에서 정수 창이 `[8,10]`에서 바뀐다)
  - 실제 Flickr25K cache로 TCBB variant의 1-step P0 stage-1 smoke test 완료

#### 4.4z 15-base 미이관 자산 (제출 전 필수)

슬롯 축소로 headline 예산이 30-bit / 15-base가 되었으나, 아래는 아직 36-bit /
18-base 적응 상태다. 본문·표에서 해당 수치를 인용할 때는 예산을 명시해야 하며,
제출 전 재적응이 필요하다.

| 자산 | 현재 상태 | 필요 작업 |
|---|---|---|
| native-DNA baselines (DNA24, PRIMO, Koike-TN, Koike-BC) | 18-base sealed diagnostic 48/48 완료 | **15-base 재적응 48셀** |
| `U0` binary baselines (9종) Panel A | 36-bit → 18-base | 30-bit 6종은 §4.5.1b에 완료, 나머지 3종(MLS³RDUH·GreedyHash·HHCH) 미실행 |
| OH / GreedyHash clean-room adapter 서술 | "36-bit에 이식" | 30-bit 이식 여부 명시 |
| §4.6 bio-projection 유효성 | 18-base GC `[8,10]` 기준 | **15-base GC `[7,8]`로 재측정** |
| §4.8 A2/A4/A5 ablation | 6-slot 기준 | 5-slot 재실행 |
| §4.9 K × bases-per-slot grid | 6-slot 기준 | 5-slot 재확인 |
| §4.10 slot intervention | 6 슬롯 | 5 슬롯 재측정 |

### 4.5 Main Results — 신규 통일 레시피, 3-seed

#### 4.5.0 보고 대상 모델

본 절의 모든 수치는 다음 **통일 레시피**로 재학습한 모델이다.

- 전 슬롯 **joint codon-diversity 규제** `L_joint` (§3.8c)
- **deterministic straight-through** codon 이산화 (`--no_gumbel_softmax`)
- **codeword–codon Sinkhorn bijection 비활성** (4개 데이터셋 공통)
- 데이터셋별 λ: MS-COCO `.03`, NUS-WIDE `.05`, Flickr25K `.02`, CIFAR-10 `.03`
  - MS-COCO의 λ는 3-seed로 재확인했다: `.03` → `.8232 ± .0098`, `.05` → `.8194 ± .0056`.
    평균은 `.03`이 높지만 두 분산이 겹치므로 **통계적으로 구분되지 않는다**
- 그 외 구조·프롬프트·P0 프로토콜은 §3과 동일

이전 draft의 champion(이하 `A-champion`)과의 유일한 차이는 위 세 항목이다.

> **2026-08-10 갱신 — 슬롯 수 6 → 5, 코드 18-base → 15-base (36 → 30 bit).**
>
> 슬롯 제거 대상은 추측이 아니라 측정으로 정했다. 기존 체크포인트에서 슬롯 하나를
> 무력화하고 held-out codon decoding을 재계산하면(`scripts/slot_drop_decoding_ablation.py`),
> **`scene_type`만이 4/4에서 제거가 이득**이다(Flickr `+.0059`, MS-COCO `+.0144`,
> NUS `+.0117`, CIFAR `+.0111`). 단독 decoding도 전 데이터셋 최하위이며
> (`.7378 / .5693 / .6655 / .8433`), 검색 기여도 역시 CIFAR 최소(`−.0089`)다.
> 텍스트 앵커 중복도는 `activity_relation`을 지목했으나, 중복도는 *누가 겹치는가*를
> 말할 뿐 *누가 없어도 되는가*를 답하지 않는다.
>
> `scene_type`은 캐시상 인덱스 5이므로 앞 5슬롯만 취하면 정확히 그것만 빠지고
> 캡션 재생성이 필요 없다. 3-seed 결과(§4.5.1)는 **비트를 17 % 줄이면서
> 해석성 4/4 개선, 검색 2/4 개선**이다.
>
> 미해결 사항: top-p mask 파라미터는 `M=6` 기준으로 튜닝된 채 남아 있어,
> `M=5`에서는 컷이 `k=3/5`에 고정되어 CIFAR `color_texture`가 이미지의 23.44 %에서
> 굶는다. `--routing_adaptive_topp_min 0.6 --routing_adaptive_topp_max 0.95`가
> 이를 0.00 %로 해소한다(§4.10). 나머지 3개 데이터셋의 재튜닝은 미완이다.

#### 4.5.1 Panel A — 18-base / 36-bit **legacy** 패널, seeds `{42,43,44}` mean ± sample std

> 이 패널은 슬롯 축소 이전의 36-bit 예산에서 측정한 값이다. **현행 headline은
> 15-base / 30-bit인 §4.5.1b**이며, 이 표는 (a) 이전 예산에서의 비교와
> (b) 예산 축소가 상대 우위에 미친 영향을 보이기 위해 유지한다.

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
| **GroundedDNA (unified)** | **`VLM-T`** | **six grounded codons → 18-base** | **.8668 ± .0017** | **.8232 ± .0098** | **.8246 ± .0033**† | **.8940 ± .0033** |

- CIFAR-10의 `U0` 열은 seeds `{43,44}`를 아직 돌리지 않아 `-`다(해당 60셀 배치는
  Flickr/MS-COCO/NUS-WIDE만 포함했다). CIFAR 비교는 §4.5.2의 single-seed diagnostic만 유효하다.
- baseline은 legacy cache provenance 때문에 invariant #6상 여전히 **strict-main ineligible**이다.
  seed 수를 채운 것은 invariant #3·#9를 만족시킬 뿐, #6은 별개 조건이다.

> † **NUS-WIDE는 `--sinkhorn_epsilon_init 0.5`로 학습한다** (2026-08-06 채택).
> 기존 `1.0`에서는 NUS가 `E*=4`에서 멈추므로 라우터가 ε≈0.99로 동작해 전송 계획이
> 거의 균등해지고(p_max 중앙값 0.2024 대 균등 0.1667), adaptive top-p의 컷이 고정
> 순위에 걸려 `primary_object`가 **이미지의 33.20 %에서 시각 토큰을 하나도 받지
> 못했다**. 그 경우 `denom.clamp_min(1e-12)` 때문에 pooled feature가 영벡터가 되고,
> 해당 슬롯의 codon은 global gate를 통해 들어온 CLIP 전역 임베딩의 재인코딩이 된다.
> ε=0.5는 이를 전 슬롯 **0.00 %**로 없앤다.
>
> | | eps 1.0 | eps 0.5 (채택) | Δ |
> |---|---:|---:|---:|
> | mAP@R | .8283 ± .0008 | .8246 ± .0033 | −.0037 |
> | codon decoding | .7368 ± .0026 | .7318 ± .0071 | −.0050 |
> | DNA-unique | .2116 ± .0079 | **.2353 ± .0039** | **+.0237** |
> | 빈 이미지 비율 | primary 33.20 % | **0.00 %** | — |
>
> 검색·해석성 지표가 소폭 내려가지만, 내려간 부분은 **슬롯이 아무것도 보지 않은
> 이미지에서 얻던 점수**다. ε은 원인이 아니라 처방이다 — MS-COCO를 ε=0.99에서
> 평가해도 빈 슬롯은 0 %이며, 근본 원인은 텍스트 앵커의 축 분리도(MS-COCO .5742,
> NUS .6020, CIFAR .6693)다. Flickr25k·MS-COCO는 빈 슬롯이 없어 적용하지 않는다
> (Flickr에 적용 시 mAP −.0138).
>
> §4.8의 A2/A4 ablation 표는 아직 ε=1.0 기준이며, full/ablated 양쪽이 같은 조건이라
> Δ는 유효하다. 제출 전 재실행 필요.

#### 4.5.1b Panel B — 15-base / 30-bit 동일 예산 비교

5-slot 모델은 30 bit를 쓰므로 36-bit 표와 직접 비교할 수 없다. baseline 6종을
**동일한 30 bit**로 재학습해(같은 러너·같은 캐시 경로, seed 42) 맞춘 결과다.
우리 값은 seeds `{42,43,44}` 평균이다.

| Dataset | CIBHash | CIMON | Bi-half | SDC | OH | CroVCA | 최강 BL | **ours** | **Δ** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Flickr25K | .7939 | .8121 | .8116 | .7252 | **.8328** | .7716 | .8328 | **.8529** | **+.0201** |
| MS-COCO | .7922 | .6681 | .7137 | .8076 | .7662 | **.8304** | .8304 | **.8313** | **+.0009** |
| NUS-WIDE | .7992 | .7890 | .7464 | .7689 | **.8025** | .8015 | .8025 | **.8221** | **+.0196** |
| CIFAR-10 | .8701 | .8524 | .7594 | .8398 | .8607 | **.8860** | .8860 | **.8998** | **+.0138** |

**동일 비트 예산에서 4/4 우위**다. 36-bit에서의 우위(`+.0341 / +.0016 / +.0218 / +.0156`)와
비교하면 Flickr에서 줄고 나머지는 유지된다.

> ⚠️ **집계 게이트 미통과.** 이 값들은 `map_at_R_post`에서 직접 읽었다. 집계기는
> baseline 구현 파일의 SHA-256을 고정해 baseline 수치가 코드 변경으로 조용히 바뀌는
> 것을 막는데, 30-bit 지원을 위해 수정한 3개 파일 중 2개는 검토된 비과학적 전환으로
> 등록했으나(diff가 비트 예산 튜플과 주석뿐이고 36/48 경로는 바이트 동일) 나머지
> 하나는 등록할 수 없었다 — **기존 36-bit 셀이 기록한 `baseline_val_select_p0.py`
> 해시가 git 이력에 없다. 즉 기존 baseline은 커밋되지 않은 워킹트리에서 실행됐고
> 그 내용은 복원 불가능하다.** 따라서 이 표는 초안에 이미 실린 36-bit 표와
> **동일한 증거 수준**이며(양쪽 모두 `legacy_cache_diagnostic_only_not_main_table_eligible`),
> 제출 전 전 비트를 커밋된 상태에서 재실행해야 한다.

#### 4.5.2 최강 `U0` baseline 대비 (3-seed 대 3-seed)

| Dataset | ours 3-seed | best `U0` 3-seed | **Δ** | A-champion Δ (seed 42) |
|---|---:|---|---:|---:|
| Flickr25K | .8668 ± .0017 | OH .8327 | **+.0341** | +.0313 |
| NUS-WIDE | .8246 ± .0033† | OH .8028 | **+.0218** | +.0239 |
| MS-COCO | .8232 ± .0098 | CroVCA .8216 | **+.0016** | −.0087 |
| CIFAR-10 | .8940 ± .0033 | *(CIBHash 재실행 중)* | *withheld* | +.0090 |

- **MS-COCO는 3-seed 대 3-seed에서 `+.0016`으로 뒤집힌다.** 우리 값이 올라서가 아니라 CroVCA의
  3-seed 평균이 `.8257`(seed 42) → `.8216`으로 내려갔기 때문이다. ⚠️ 그러나 `+.0016`은 우리 std
  `.0098`의 6분의 1이므로 **통계적으로는 동률**이다. "대등하다"가 정확하고 "이겼다"는 쓰지 않는다.
- ⚠️ **CIFAR-10 열은 보류**한다. 그 데이터셋의 최강 `U0`였던 CIBHash가 2026-08-04 source-fidelity
  감사(F3)로 무효화되어 재실행 중이다. 참고로 차순위 CroVCA는 `.8819`이므로, corrected CIBHash가
  우리 `.8940` 아래로 내려오면 이 열도 우세로 바뀐다.

#### 4.5.3 변형 행 — `+ L_bio` (생물 제약을 학습하는 조건)

main row는 제약을 **사후 투영으로만** 만족시킨다. 아래 변형은 학습 중 GC/homopolymer 항(§3.7)을
추가해 제약을 **학습**한다. 두 행 모두 배포 코드는 동일한 DP 투영을 거쳐 validity 100 %다.

| Dataset | mAP@R | Δ vs main | codon decode | Δ | DNA-unique | Δ | **사전 유효율** | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MS-COCO | .8167 | −.0120 | .6429 | +.0016 | .2021 | +.0067 | 46.4 % → **87.0 %** | **+40.6** |
| NUS-WIDE | .8288 | **+.0014** | .7339 | −.0008 | .1948 | −.0210 | 56.5 % → **77.5 %** | **+21.1** |
| Flickr25K | .8662 | −.0011 | .7565 | −.0110 | .3978 | −.0725 | 40.5 % → **84.0 %** | **+43.5** |
| CIFAR-10 | .8917 | −.0053 | .8942 | −.0047 | .0896 | −.0411 | 39.0 % → **85.8 %** | **+46.8** |

🟢 **이 행이 있어야 “모델이 생물 제약을 학습한다”를 쓸 수 있다.** 참조점 두 개와 함께 읽는다.

| 기준 | 사전 유효율 |
|---|---:|
| 균등 4^L 무작위 (DP 정확계산) | **45.2 %** |
| binary `U0` baseline 실측 (CIBHash/SDC/OH/CroVCA) | 43–46 % |
| ours, main row | 39–57 % |
| **ours, `+ L_bio`** | **77–87 %** |

binary baseline이 chance에 붙어 있는 것은 우연이 아니다. `base ∈ {C,G}`는 두 bit의 XOR이 1인 경우와
동치이므로 **GC 함량은 bit-pair parity 술어**이고, 독립적인 per-bit 목적함수로는 표현되지 않는다.
실측 GC 분포가 Binomial(18, ½)(평균 9.00, 표준편차 2.12)에 정확히 앉는 것이 그 서명이다
(CIBHash 9.04±2.14, SDC 8.84±2.15, OH 9.56±2.19, CroVCA 8.74±2.22). 반면 위치별 4-way 사후분포에
직접 hinge를 거는 우리 모델은 77–87 %에 도달한다. **알파벳 크기가 아니라 제약을 표현할 수 있는
좌표계가 다르다는 것이 요점**이며(36 bit ↔ 18 base는 전단사이므로 용량 이점은 없다), 이 행이 그
주장의 유일한 실증이다.

⚠️ **대가는 다양성이다.** 4개 중 3개에서 DNA-unique가 떨어지며 Flickr `−.0725`가 최대다
(MS-COCO만 `+.0067`로 예외). 검색은 `−.0120 … +.0014`로 대체로 미미하고, 해석성은
`−.0110 … +.0016`으로 **사실상 중립**이다 — 제약 학습이 codon의 의미 보존을 훼손하지는 않는다.

⚠️ **λ_bio는 아직 튜닝되지 않았다.** 위 수치는 전부 `λ_bio = 10` 단일값이며, 이 값은 2026-07-29에
**Flickr 한 데이터셋에서, 그것도 A-champion 레시피 위에서** 손실 크기를 보고 고른 것이다. 통일
레시피에서도, 다른 세 데이터셋에서도 스윕한 적이 없다. 유효율 상승폭이 `+21.1`(NUS)에서 `+46.8`
(CIFAR)까지 벌어지고 다양성 비용도 `+.0067`에서 `−.0725`까지 벌어지는 것은 **데이터셋별 최적 λ가
서로 다르다는 신호**다. 따라서 이 표는 *“제약 학습이 가능하다”*의 증거로만 쓰고, *“이것이 최적
절충이다”* 로는 쓰지 않는다.

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
- ⚠️ **main row의 사전 유효율은 39–57 %로 균등 4^L 기대치 45.2 %와 같은 수준이다.** 즉 main
  recipe는 생물 제약을 학습하지 않으며 제약은 순수한 사후 투영으로 남는다. `+ L_bio` 변형 행(§4.5.3)이
  이를 77–87 %로 끌어올리며, 그것이 "제약을 학습한다"고 쓸 수 있는 유일한 근거다.
- projection은 `base_indices`만 바꾸고 `codebook_indices`는 바꾸지 않는다. NMI/drop/codeword decoding은
  구조적으로 불변이다.

### 4.7 Held-out codon decoding — raw-base-E\* control 대비 4/4

control은 `U0` baseline을 **동일 raw-base-Hamming E\* 규약**으로 선택한 P0 matrix artifact이며,
3-base chunk로 나눈 post-hoc 통제다. 우리 값은 seeds `{42,43,44}` mean ± std다.

| Dataset | ours codon (3-seed) | 최강 control | margin | majority | shuffled |
|---|---|---|---:|---:|---|
| Flickr25K | **.7694 ± .0017** | OH .7261 | **+.0433** | .4730 | .4783 ± .0118 |
| MS-COCO | **.6466 ± .0048** | CroVCA .5727 | **+.0739** | .3160 | .3117 ± .0035 |
| NUS-WIDE | **.7318 ± .0071**† | OH .6772 | **+.0546** | .4822 | .4784 ± .0027 |
| CIFAR-10 | **.8903 ± .0104** | CroVCA .8263 | **+.0640** | .2929 | .2847 ± .0236 |

- A-champion 대비 이득: `+.0101 / +.0322 / +.0239 / +.0227`. **4/4 개선이며 seed 분산의 2–19배**다.
- `shuffled ≈ majority`가 전 데이터셋에서 성립하므로 probe 자체는 정상 동작한다.
- 이 표가 논문의 해석성 주 근거다. 단, decoder accuracy가 human understanding이나 인과적 제어와
  같지 않다는 §5.3의 범위 제한을 유지한다.

#### 4.7b 5-slot / 15-base (30 bit) 재측정 — 3-seed

| Dataset | 6-slot / 36 bit | **5-slot / 30 bit** | Δ |
|---|---:|---:|---:|
| MS-COCO | .6466 | **.6737** | **+.0270** |
| CIFAR-10 | .8903 | **.9144** | **+.0240** |
| Flickr25K | .7694 | **.7821** | **+.0127** |
| NUS-WIDE | .7318 | **.7350** | +.0031 |

**코드 길이를 17 % 줄이면서 4/4에서 해석성이 오른다.** 실현 이득이 drop-ablation
예측치를 상회하므로, 슬롯을 제거하면 남은 슬롯이 재분할하는 효과가 추가로 있다.
NUS는 seed 분산(±.0026) 안이라 개선을 주장하지 않는다.

#### 4.7c **이 probe는 single-label 데이터셋에서 무효다**

라벨 다중도가 결과의 부호를 뒤집는다.

| Dataset | 라벨 수 | 이미지당 positive | multi-label |
|---|---:|---:|---|
| CIFAR-10 | 10 | **1.00** (min = max = 1) | **아니오** |
| Flickr25K | 24 | 3.74 | 예 |
| NUS-WIDE | 21 | 3.33 | 예 |
| MS-COCO | 80 | 2.92 | 예 |

positive가 하나뿐이면 label-ranking AP는 역수 순위로 축약되고, **모든 슬롯이 동일한
하나의 타깃을 두고 경쟁**한다. 슬롯별 특화가 기여할 여지가 구조적으로 없으므로,
포화된 global gate를 통해 codon head에 도달하는 CLIP 전역 임베딩이 이긴다.

테스트 집합을 *슬롯이 실제로 본 이미지 / 보지 못한 이미지*로 나누면 확인된다.

| Dataset | slot | 본 것만 | 못 본 것만 |
|---|---|---:|---:|
| CIFAR-10 | `secondary_object` | .8536 | **.9856** |
| CIFAR-10 | `scene_type` | .8161 | **.9000** |
| NUS-WIDE | `primary_object` | .7376 | **.7147** |
| NUS-WIDE | `secondary_object` | .7642 | **.6086** |
| NUS-WIDE | `color_texture` | .7227 | **.6237** |

**CIFAR에서는 눈을 감은 슬롯이 더 높고, multi-label NUS에서는 5개 중 4개에서 더 낮다.**
따라서 이 표의 CIFAR 열은 해석성 근거로 인용하지 않으며, 해석성 주장은
Flickr25K · NUS-WIDE · MS-COCO에 둔다. ImageNet100은 대안이 되지 못한다
(`MULTI_LABEL['ImageNet100'] = False`, 이미지당 positive 1개로 동일한 붕괴).

#### 4.7d CUB-200 속성 기반 재검증 (single-label 우회)

CUB-200은 클래스 기준으로는 single-label(200종)이지만 이미지별 **312개 이진 속성**
주석을 제공하므로, 같은 데이터셋을 진짜 multi-label 타깃으로 쓸 수 있다
(`scripts/cub_attribute_codon_decoding.py`). certainty ≥ 3, positive 비율
`[.02, .98]` 필터 후 **201개 속성**이 남고 **이미지당 positive 28.07개**다.

| | concept mAP |
|---|---:|
| majority (코드 무시) | .4170 |
| shuffled (구조 파괴) | .4125 ± .0012 |
| **GroundedDNA (6-slot)** | **.4805** |
| margin | **+.0635** |

`shuffled ≈ majority`가 성립하므로 probe는 정상 동작하며, 이 조건에서는 슬롯별 특화가
보상된다. 5-slot 및 landmark localization 비교는 진행 중이다.

### 4.8 Causal ablation — P0 + bio-projected 동일 프로토콜

이전 draft의 A1/A2/A4 행은 모두 pre-P0 / pre-bio 진단값이었고 draft 스스로 "main causal table에
사용하지 않는다"고 적어 두었다. 아래는 신규 모델 위에서 동일 프로토콜로 재실행한 것이다.

#### A2. Text supervision 제거 (`--disable_text_supervision`)

| Dataset | full mAP@R | no text | **Δ mAP@R** | full decode | no text | **Δ decode** |
|---|---:|---:|---:|---:|---:|---:|
| MS-COCO | .8232 | .7627 | **−.0605** | .6466 | .5606 | **−.0860** |
| CIFAR-10 | .8940 | .8743 | −.0197 | .8903 | .8779 | −.0124 |
| NUS-WIDE | .8209 | .8001 | −.0209 | .7240 | .6951 | **−.0288** |
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
| NUS-WIDE | .8209 | .8184 | −.0025 | .7240 | .7176 | −.0064 |

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
| NUS-WIDE | - | - | **.8246**† | - |
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

#### 4.10b 슬롯 점유 진단 — 이미지 단위 빈 슬롯

배치 평균 질량은 "모든 이미지에서 얇게 받음"과 "대부분 이미지에서 정확히 0"을
구분하지 못한다. 후자는 `denom.clamp_min(1e-12)` 때문에 pooled feature가 영벡터가
되어 그 슬롯의 3염기가 이미지와 무관해지는 결함이다. 이미지 단위 측정 결과다.

| Dataset | slot | 6-slot | **5-slot** |
|---|---|---:|---:|
| CIFAR-10 | `secondary_object` | **51.76 %** | 2.15 % |
| CIFAR-10 | `scene_type` | 45.12 % | *(제거)* |
| CIFAR-10 | **`color_texture`** | 0.00 % | **23.44 %** |
| MS-COCO | 전 슬롯 | 0 % | **0.00 %** |
| Flickr25K | 전 슬롯 | 0 % | ≤ 1.56 % |

**결함이 사라지지 않고 이동한다.** top-p 임계값이 `M=6` 기준이라, `p_max` 중앙값
`.2406`(균등 `.2000`)에서 `cum ≈ k/5`이고 `tau ≈ .60`이 `k=3/5`를 99.7 %로 고정한다.
패치당 슬롯의 40 %를 버리므로(`M=6`에서는 33 %) **누군가는 반드시 최하위가 된다.**

CIFAR 교정 3셀:

| cell | 조작 | mAP@R | decoding | `color_texture` 빈 이미지 |
|---|---|---:|---:|---:|
| `slot5` | top-p .3–.7 | **.9019** | .9158 | 23.44 % |
| **`s5topp69`** | top-p **.6–.95** | .8875 | .9033 | **0.00 %** |
| `s5topp45` | top-p .4–.8 | .8974 | .9053 | 9.77 % |
| `s5csd005` | `L_csd` 0.05 | .8991 | **.9175** | **43.36 %** |

`L_csd`(ConceptHash Eq. 8 이식)는 **역효과**다 — 슬롯을 밀어내는 힘이 이미 최하위인
슬롯을 더 굶긴다. 그리고 **decoding이 빈 이미지 비율에 완전히 단조**다
(`.9033 < .9053 < .9158 < .9175`). §4.7c에서 보인 대로 CIFAR은 single-label이라
지표가 굶주림을 보상하므로, **CIFAR 구성은 decoding이 아니라 빈 이미지 비율로
선택해야 하며 그 기준으로는 `s5topp69`가 유일하게 정합한다.**

#### 4.10c 정성적 평가 figure — 5 슬롯, 4개 데이터셋 (2026-08-11)

`scripts/regen_viz_routing.py`를 `s5topp69` 셀에 적용해 재생성했다. 각 heatmap은
train 이미지 8장 x (원본 + 로컬 슬롯 4개)이며 패널 제목에 해당 축의 Qwen caption을
싣는다. t-SNE는 codebook별로 test 3000장에 대해 그린다. 소스 run은
`docs/figures/qualitative_s5/README.md`에 기록했다.

| 데이터셋 | routing heatmap | codebook t-SNE |
|---|---|---|
| CIFAR-10 | `docs/figures/qualitative_s5/cifar10_routing_s5.png` | `docs/figures/qualitative_s5/cifar10_codebook_tsne_s5.png` |
| Flickr25k | `docs/figures/qualitative_s5/flickr25k_routing_s5.png` | `docs/figures/qualitative_s5/flickr25k_codebook_tsne_s5.png` |
| NUS-WIDE | `docs/figures/qualitative_s5/nuswide_routing_s5.png` | `docs/figures/qualitative_s5/nuswide_codebook_tsne_s5.png` |
| MS-COCO | `docs/figures/qualitative_s5/mscoco_routing_s5.png` | `docs/figures/qualitative_s5/mscoco_codebook_tsne_s5.png` |

> **주의 — 이 figure는 현재 상태로는 슬롯 특화 주장을 뒷받침하지 못한다.**
> Flickr25k에서 로컬 슬롯 4개의 heatmap은 caption이 크게 다름에도 대부분의
> 이미지에서 거의 동일하다. 이는 마스크 전 슬롯 간 라우팅 코사인 0.987과
> §4.10d의 landmark 결과가 시각적으로 드러난 것이다. figure가 지탱할 수 있는
> 주장은 "각 슬롯이 서로 다른 영역을 소유한다"가 아니라 "caption이 공유된
> attention을 조종한다"이다.

#### 4.10d Landmark localisation error (CUB-200) — 🔴 음성 결과

codon decoding은 코드가 개념 정보를 담는지, caption 교체 반사실은 슬롯의 attention이
자기 caption에 인과적으로 의존하는지를 말한다. **어느 쪽도 그 attention이 올바른
위치에 있는지는 말하지 않는다.** CUB-200은 사람이 표기한 부위 좌표를 가진 유일한
데이터셋이고, ConceptHash(CVPRW'24)가 보고하는 지표이기도 하다.

프로토콜(TASN / ConceptHash): 슬롯별 라우팅 열을 14x14 격자 위 분포로 보고 질량
무게중심을 정규화 좌표에서 구한 뒤, 발견된 슬롯과 사람 부위 사이에 사전 대응이
없으므로 **train에서만** 2M개 슬롯 좌표 → 각 landmark 좌표의 선형 회귀를 적합하고,
held-out L2 거리를 이미지 크기 대비 %로 보고한다. test transform이 crop 없는
`Resize((224,224))`라 원본 픽셀을 W·H로 나누면 격자와 동일한 좌표계가 된다.
`scripts/cub_landmark_localization.py`, 전수 split (train 5994 / test 5794).

| 모델 | 슬롯 | beak | left wing | tail |
|---|---:|---:|---:|---:|
| 상수 위치 baseline | - | 19.94 | 12.05 | 28.58 |
| `cub200_bidirAB_legacy` | 6 | **19.93** | **11.66** | **28.23** |
| `cub200_bidirAB_s5` | 5 | 20.00 | 11.86 | 28.71 |
| `cub200_bidirAB_s5_topp69` | 5 | 20.31 | 12.50 | 28.99 |

상수 baseline은 모든 이미지에 **train 평균 위치**를 찍는 예측, 즉 사진을 보지 않는다.
최선의 모델조차 이를 0.01 / 0.39 / 0.35 %p 앞서는 데 그치고, 5슬롯 모델은 beak과
tail에서 **진다**. 슬롯 무게중심은 새의 부위 위치 정보를 사실상 담지 않으며, 이는
슬롯 축소의 결과가 아니다 — 6슬롯도 거의 나아지지 않는다.

지표 자체의 한계도 함께 적어야 한다: 선형 적합은 슬롯 **앙상블**을 평가하지
"슬롯 m = 부리"를 검증하지 않는다. 선행 연구가 지닌 것과 동일한 한계다.

#### 4.10e CUB-200 5슬롯 top-p — CIFAR와 반대 방향

| cell | topp | mAP | unique code | codebook별 unique |
|---|---|---:|---:|---|
| `cub200_bidirAB_s5` | (0.3, 0.7) | **.0781** | **.3842** | 34, 23, 32, 21, 32 |
| `cub200_bidirAB_s5_topp69` | (0.6, 0.95) | .0755 | .2636 | 39, 19, 16, 23, 25 |

CIFAR-10에서는 top-p를 (0.6, 0.95)로 넓히는 것이 굶주린 슬롯을 제거했으나, CUB에서는
mAP와 코드 다양성을 **모두** 떨어뜨린다. top-p는 전역 상수가 아니라 데이터셋별
하이퍼파라미터로 보고해야 한다.

#### 4.10f MS-COCO baseline decoding 부재 — 사전 존재 한계

MS-COCO baseline 비교는 모든 train basename에서 KeyError로 실패한다. 우리 최적화용
train pool(10 000장)이 baseline DB 추출(107 218장)과 **100 % 서로소**이며, 30-bit와
36-bit 양쪽 모두 그렇다. 과거의 모든 `docs/heldout_decoding_mscoco*.json`도 baseline
키가 비어 있어, 슬롯 축소로 생긴 회귀가 아니다. 해결하려면 baseline runner가
`extract_train.npz`를 내보내거나 우리 train pool을 DB에서 뽑아야 한다. 표에는
`-`로 두고 원인을 명시한다.

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
