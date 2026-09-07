# GroundedDNA — 확정된 모델 아키텍처 · 실험 프로토콜 · 재현 명세

**작성 2026-09-07.** 이 문서는 논문 재현에 필요한 **확정 사항만** 담는다.

- **근거 실행**: `result/260811+…promptAblA_{cifar,flickr,nuswide}_A_v4_P0refit_e4`,
  `result/260812+mscoco_setting1_promptAblA_mscoco_A_v5b_s42_P0refit_e39`
  (프로토콜 감사 `docs/EXPERIMENT_PROTOCOL_AUDIT_2026-08-13.md` **이전**의 마지막 본 모델 3-seed 실행)
- 모든 값은 그 실행들의 `args.txt` 343줄에서 직접 읽었다. 추정값 없음.
- **§7에 "이 문서에서 제외한 것"**을 명시했다. 현재 재조정 중인 항목은 여기 없다.

> ### 문서 작성 중 확인된 정정 사항
> 이전 세션에서 나는 adaptive top-p의 incumbent를 `0.3/0.7`이라고 보고했다. **틀렸다.**
> 네 데이터셋 champion `args.txt` 전부가 `routing_adaptive_topp_min=0.6 / max=0.95`다.
> `0.3/0.7`은 트레이너 셸의 하드코딩 기본값이며, Phase-3 selection이 override에 실패해
> 그 값으로 돈 것이 **회귀**였다. 논문 recipe는 `0.6/0.95`다.

---

## 1. 모델 아키텍처

### 1.1 백본 — 전부 동결

| 항목 | 값 |
|---|---|
| `backbone_type` | **`clip`** |
| 실제 가중치 | **`openai/clip-vit-base-patch16`** (ViT-B/16, `d_model` 768) |
| `freeze_backbone` | **`True`** — 시각·텍스트 인코더 모두 동결 |
| `backbone_lr` | `1e-06` (동결이므로 무효, 기록용) |
| 패치 수 | 196 (14×14) |
| 텍스트 임베딩 차원 | 512 |

`siglip2_backbone`은 `args.txt`에 남아 있으나 **사용하지 않는다**(`backbone_type=clip`).
학습되는 것은 **어댑터, 라우터, 코드북, 코돈 헤드**뿐이다.

### 1.2 기하 — 5 슬롯 / 15 염기 / 30 비트 (확정)

| 항목 | 값 |
|---|---|
| `num_semantic_parts` (M) | **5** |
| `num_codebooks` | **5** |
| `num_codons_per_codebook` (ℓ) | **3** |
| 총 염기 | **15** |
| 총 비트 | **30** |
| 환경변수 | **`GDNA_NUM_SEMANTIC_PARTS=5` 를 python 시작 전에 반드시 설정** |

`--num_semantic_parts 5`만 주고 환경변수를 빠뜨리면 모델이 `ValueError`로 거부한다
(import 시점에 기하가 고정되므로, `args.txt`가 실제 모델을 기술하지 못하는 상태를 막는다).

**슬롯 축 순서** (`global`이 슬롯 0):

| 슬롯 | 축 | 종류 |
|---|---|---|
| 0 | `global` | 전역 요약 |
| 1 | `primary_object` | 국소 |
| 2 | `secondary_object` | 국소 |
| 3 | `activity_relation` | 국소 |
| 4 | `color_texture` | 국소 |

6슬롯 시절의 `scene_type`은 캐시 인덱스 5였고, 앞 5슬롯만 취해 제거했다(캡션 재생성 불필요).

### 1.3 라우팅 — 텍스트 유도 불균형 OT + confidence-adaptive top-p

| 옵션 | 값 |
|---|---|
| `router_type` | **`sinkhorn`** |
| `sinkhorn_lambda_a` / `lambda_b` | **`1.0` / `1.0`** (UOT 주변부 KL 계수) |
| `sinkhorn_epsilon_init` | **`1.0`** (NUS-WIDE만 **`0.5`**) |
| `sinkhorn_epsilon_final` | **`0.1`** |
| ε 스케줄 | cosine, **지평 = `--epoch`** → `-e = N+1`로 정렬 필수 |
| `routing_adaptive_topp` | **`True`** |
| `no_routing_adaptive_topp` | **`False`** |
| `routing_adaptive_topp_min` | **`0.6`** |
| `routing_adaptive_topp_max` | **`0.95`** |
| `routing_adaptive_topp_entropy` | **`False`** (max-probability confidence 규칙) |
| `routing_perplexity_topk` | **`False`** (상호배타) |
| `eval_routing_mode` | **`codebook_mean`** — 추론 시 앵커 = 학습된 코드북의 축별 평균 |
| `use_null_centroid` | `False` |
| `routing_hard` | `False` |
| `routing_topk` / `routing_topp` | `None` / `None` (고정 임계 미사용) |

τ 규칙: `τ = τ_min + (1 − p_max)(τ_max − τ_min)`. 각 패치는 자기 1순위 슬롯을 항상 유지한다.

**ε 정렬은 재현 필수**: `-e 60` + `stop@4`로 두면 ε이 0.49~0.99 구간에 머물러 슬롯이 분화하지
못한다. `-e = N+1`로 맞추면 슬롯 코사인이 .97 → .50으로 떨어진다.

### 1.4 라우팅 — OFF로 확정된 옵션 (빠짐없이)

아래는 모두 **`False` 또는 `0.0`**이며, 논문 모델에서 **사용하지 않는다**.

| 옵션 | 값 |
|---|---|
| `routing_ambiguity_topk` | `False` |
| `routing_specificity_marginal` | `False` |
| `routing_centered_consensus_mask` | `False` |
| `routing_cls_verified_consensus_mask` | `False` |
| `routing_codebook_choice` | `False` |
| `routing_text_evidence_beta` | `0.0` |
| `routing_token_ot_evidence` | `False` |
| `grounded_text_routing` | `False` |
| `soft_visual_grounded_text_pool` | `False` |
| `cosine_visual_grounded_text_pool` | `False` |
| `residualize_visual_for_routing` | `False` |
| `text_transform_routing_only` | `False` |
| `route_global_text` | `False` |
| `visual_adapter_after_router` | `False` |
| `foreground_text_mask_topk_ratio` | `None` |
| `lambda_slot_diversity` | `0.0` |
| `lambda_routing_text` | `0.0` |

`cluster_attn_*`, `cross_attn_*`, `attention_router_temperature`, `slot_attention_iters`는
`router_type != sinkhorn`일 때만 쓰이는 **dead parameter**다.

### 1.5 양자화

| 옵션 | 값 |
|---|---|
| `codebook_size` (K) | **128** (CIFAR-10만 **64**) |
| `share_codebook` | **`False`** — 슬롯마다 분리된 코드북 |
| `codebook_update` | **`ema`** |
| `codebook_ema_decay` | **`0.99`** |
| `codebook_ema_eps` | `1e-05` |
| `codebook_revive` | **`True`** |
| `codebook_revive_threshold` / `_every` | `0.01` / `50` |
| `vq_distance_mode` | **`euclidean`** |
| `vq_loss_cosine` | `False` |
| `codebook_repel_strength` | `0.0` (OFF) |
| `lambda_codebook_ortho` | `0.0` (OFF) |
| `use_codebook_text_prompts` | `False` |
| `text_init_codebook` | `none` |
| `codebook_K_max` / `split_epochs` | `0` / 없음 (코드북 분할 OFF) |
| `pq_slot_subspace` | `False` |
| `local_residual_quant` | **`False`** — v132a에서 무기여·유해로 판정, 제거됨 |
| `slot_sequential_residual` | `False` |

### 1.6 코돈 헤드

| 옵션 | 값 |
|---|---|
| `codon_input_source` | **`quantized`** |
| `codon_residual_gamma` | **`0.0`** |
| `codon_head_hidden_dim` | `0` (선형) |
| `codon_chunk_layernorm` | `False` |
| `codon_chunk_interleave` | `False` (stride 분할 — 반증됨) |
| `codon_position_specific_head` | `False` |
| `codon_position_residual_adapter` | `False` |
| `codon_residual_split` / `_gate` | `False` / `False` |
| `codon_full_linear` | `False` |
| `codon_text_anchor` | **`False`** (`lambda_codon_text_anchor=0.1`은 게이트가 꺼져 무효) |
| `use_gumbel_softmax` | **`False`** — **noGumbel (결정론적 straight-through)** |
| `lambda_codeword_codon_sinkhorn` | **`0.0`** — bijection **OFF (CIFAR 포함 4/4 통일)** |

`gumbel_tau_*` 값들은 `use_gumbel_softmax=False`이므로 **무효**다.

**전역 게이트**: `disable_global_gate=False`, `global_gate_init_logit=4.595`,
`use_stop_grad_global=True`. 전역 코드워드가 게이트를 통해 국소 코돈 헤드에 주입된다.

### 1.7 텍스트 경로 (핵심 기여 — 절대 끄지 않는다)

| 옵션 | 값 |
|---|---|
| `disable_text_supervision` | **`False`** |
| `qwen_text_cache_path` | 데이터셋별 Qwen3 캡션 JSONL (오프라인 1회 생성) |
| `text_embed_transform` | **`partial_whiten`** |
| `text_whiten_gamma` | **`0.25`** |
| `text_whiten_eps` | `1e-05` |
| `text_whiten_npz` | `…_foils/text_whiten_trainOnly_localOnly.npz` |
| `per_slot_text_adapter` | **`True`** |
| `adapter_type` | **`mlp`**, `adapter_dropout=0.0` |
| `use_text_token_attention` | `False` |
| `text_inject_train_only` | `none` |
| `bidirectional_token_prune` | **`True`**, mode **`legacy`** |
| `bidirectional_token_prune_visual_ratio` | **`0.5`** (MS-COCO만 **`1.0`** = 미프루닝) |
| `bidirectional_token_prune_text_ratio` | **`0.5`** (MS-COCO만 **`1.0`**) |

추론에는 텍스트도 VLM도 **없다**.

### 1.8 대조 학습 헤드

| 옵션 | 값 |
|---|---|
| `use_paired_aug_ntxent` | `True` (단 `lambda_ntxent=0.0` → **기여 없음**) |
| `ntxent_mode` | `per_codebook` |
| `ntxent_dynamic_tau` | `True`, `alpha=0.3`, variant **`text_cos`** |
| `cibhash_mode` | **`per_codebook`** |
| `cibhash_temperature` | **`0.3`** |
| `cibhash_dynamic_tau` | **`True`**, `alpha=0.3` |
| `cibhash_dynamic_tau_skip_global` | **`True`** (A-recipe 전역 슬롯 skip) |
| `cibhash_ntxent_continuous` | `True` |
| `cibhash_ntxent_source` | **`visual_token`** |
| `use_decoder` / `use_hash_recon` / `use_dual_hash_proj` | 전부 **`False`** |

---

## 2. 손실 함수 — 실제로 활성인 항만

### 2.1 활성 (terminal epoch에서 값 ≠ 0으로 확인)

| 논문 표기 | 코드 λ | 값 | 로그 컬럼 |
|---|---|---:|---|
| `L_contrastive` | `lambda_cibhash_ntxent` | **1.0** (NUS·COCO **1.5**) | `train_loss_cibhash_ntxent` |
| — (CIB KL) | `lambda_cibhash_kl` | **0.001** | (총합 내) |
| `L_text-code-contrastive` | `lambda_text_hash_ntxent` | **0.05** (COCO **0.10**) | `train_loss_text_hash_ntxent_add` |
| `L_xmodal` | `lambda_xmodal_commit` | **0.05** (COCO **0.10**) | `train_loss_xmodal_commit` |
| `L_text-code-KL` | `lambda_text_code_kl` | **0.05** (COCO **0.10**) | `train_loss_text_code_kl` |
| `L_transport` | `lambda_wasserstein` | **0.15** (COCO **0.05**) | `train_loss_wasserstein` |
| `L_VQ` | `lambda_vq` | **0.25** | `train_loss_vq` |
| `L_commit` | `lambda_quant` | **0.05** | `train_loss_quant` |
| `L_codebook-balance` | `lambda_bu` | **0.02** | `train_loss_bu`, `_cb_balance`, `_cb_uncorr` |
| `L_base-prior` | `lambda_dna` / `eta_base_balance` | **0.05** / **0.3** | `train_loss_dna`, `_base_balance`, `_entropy` |
| `L_codon-joint` | `lambda_codon_joint` | **데이터셋별** (§3) | `train_loss_codon_joint` |
| — (anchor EMA) | `lambda_anchor` | **0.05** | `train_loss_anchor` |

부속: `anchor_ema_momentum=0.99`, `bu_warmup_epochs=0`, `codon_joint_slots=""`(전 슬롯),
`codon_joint_floor=1e-06`, `text_code_kl_tau_v=0.1`, `tau_t=0.07`, `conf_threshold=0.2`,
`text_code_kl_skip_global=True`, `xmodal_commit_skip_global=True`,
`text_hash_ntxent_skip_global=True`, `text_hash_ntxent_temperature=0.07`.

### 2.2 λ > 0 이지만 기여 0 (게이트가 꺼짐 — 반드시 이 상태로 둘 것)

| 항 | λ | 왜 0인가 |
|---|---:|---|
| `lambda_recon` | `1.0` | `use_decoder=False` → `train_loss_recon = 0.000000` |
| `lambda_codon_text_anchor` | `0.1` | `codon_text_anchor=False` |

### 2.3 명시적으로 OFF (λ = 0)

`lambda_hash`, `lambda_hash_hard`, `lambda_ntxent`, `lambda_text_hash`, `lambda_cw_xmodal`,
`lambda_codeword_codon_sinkhorn`, `lambda_codeword_codon_agg_ent`, `lambda_codeword_codon_pairwise`,
`lambda_text_cluster_codon_ot`, `lambda_text_codon_rel`, `lambda_proto_cluster_cos`,
`lambda_proto_cluster`, `lambda_hierarchical_cluster_codon`, `lambda_hash_recon`,
`lambda_dual_semantic`, `lambda_dual_instance`, `lambda_routing_text`,
`lambda_text_codeword_contrastive`, `lambda_text_preq_contrastive`,
`lambda_text_visual_hash_contrastive`, `lambda_codeword_text_proto`, `lambda_global_dna_ntxent`,
`lambda_ortho_text`, `lambda_slot_diversity`, `lambda_codebook_ortho`, `lambda_sim_spread`,
`lambda_swav_assign`, **`lambda_bio_constraint`**.

> `lambda_bio_constraint = 0.0`이 기본이다. 생물 제약을 **학습**하는 조건은 §4.5.3의
> `+ L_bio` **별도 행**이며 본 표 모델이 아니다.

---

## 3. 데이터셋별로 다른 값 — 이것이 전부다

네 데이터셋은 **동일 아키텍처**다. `args.txt` diff로 확인된 차이는 아래가 전부다.

| | CIFAR-10 | Flickr25K | NUS-WIDE | MS-COCO |
|---|---:|---:|---:|---:|
| prompt | v4 | v4 | v4 | **v5b** |
| `codebook_size` K | **64** | 128 | 128 | 128 |
| `lambda_codon_joint` | **.03** | **.02** | **.05** | **.03** |
| `sinkhorn_epsilon_init` | 1.0 | 1.0 | **0.5** | 1.0 |
| `lambda_cibhash_ntxent` | 1.0 | 1.0 | **1.5** | **1.5** |
| `lambda_wasserstein` | .15 | .15 | .15 | **.05** |
| `lambda_xmodal_commit` | .05 | .05 | .05 | **.10** |
| `lambda_text_code_kl` | .05 | .05 | .05 | **.10** |
| `lambda_text_hash_ntxent` | .05 | .05 | .05 | **.10** |
| `bidir_token_prune_{visual,text}_ratio` | .5/.5 | .5/.5 | .5/.5 | **1.0/1.0** |
| `num_classes` | 10 | 24 | 21 | 80 |

**A-champion 대비 차이는 정확히 셋**: `λ_joint > 0`, `--no_gumbel_softmax`,
그리고 CIFAR bijection `.1 → 0` (이로써 4/4 아키텍처 통일).

---

## 4. 공통 하이퍼파라미터 (전 데이터셋 동일)

### 4.1 최적화

| 항목 | 값 |
|---|---|
| `batch_size` | **64** |
| `proj_lr` | **0.001** |
| `backbone_lr` | `1e-06` (동결) |
| `text_adapter_lr` | `None` (= `proj_lr`) |
| `weight_decay` | **0.06** |
| `lr_scheduler` | **`cosine`** |
| `lr_eta_min` | `1e-05` |
| `scheduler` | `reduce` (레거시, `lr_scheduler`가 실효) |
| `random_seed` | **42 / 43 / 44** (3-seed) |
| `beta` | 0.3 |
| `embed_dim` | 64 |

### 4.2 감독 체계 — **HARD INVARIANT**

| 항목 | 값 |
|---|---|
| `hash_target_mode` | **`siglip_cos`** |
| `siglip_cos_pos_rate` | `0.2` |
| `hashnet_use_jaccard` | `False` |

> **`jaccard`는 Flickr25k 태그 라벨을 읽으므로 비지도 주장을 깨뜨린다.**
> 실행 직후 `args.txt`에서 `hash_target_mode=siglip_cos`를 반드시 확인할 것.

### 4.3 평가 · 추출

| 항목 | 값 |
|---|---|
| `dna_distance_mode` | **`base`** (raw base-Hamming) |
| `eval_every` | **1** |
| `extract_batch_size` | 256 |
| `val_select_metric` | `mAP_at_R` |
| `evaluation` | `True` |
| `post_eval_compositional` | `False` (본 실행에서 비활성; 별도 분석으로 수행) |

---

## 5. 생물학적 유효성 — 중앙 정책 (F07 해결)

**단일 출처**: `dna_utils/gc_policy.py`, `POLICY_VERSION = "gc-40-60-inclusive-v1"`

| 예산 | GC 개수 창 | homopolymer |
|---|---|---|
| **L = 15 (본 논문)** | **[6, 9]** | run ≤ 3 |
| L = 20 | [8, 12] | run ≤ 3 |

- 규칙은 **40–60 % inclusive** 하나이며, 각 스크립트가 `ceil(frac·L)`을 다시 계산하면 안 된다.
- **소비해야 하는 것은 정수 개수**이지 분수가 아니다.
- L=18에서는 세 관행이 모두 [8,10]으로 같아 불일치가 드러나지 않았다. **L=15와 L=20에서만 갈린다.**
- 모든 result manifest에 `as_manifest_record()`를 기록한다.
- DP 투영은 **보고되는 모든 지표에 적용**한다.

---

## 6. 실험 프로토콜 — 고정해야 하는 것

> 아래는 **프로토콜 감사가 재조정 중인 항목을 제외한** 확정 사항이다.

### 6.1 데이터셋 split (관행 고정)

| Dataset | subset | Train | Query | DB | Relevance | Metric |
|---|---:|---:|---:|---:|---|---|
| Flickr25K | 25,000 / 24 concepts | 5,000 | 2,000 | 23,000 | ≥1 공유 라벨 | **mAP@5,000** |
| MS-COCO | 122,218 / 80 classes | 10,000 | 5,000 | 107,218 | ≥1 공유 라벨 | **mAP@5,000** |
| NUS-WIDE | 195,834 / 21 concepts | 10,500 | 2,100 | 193,734 | ≥1 공유 라벨 | **mAP@5,000** |
| CIFAR-10 | 60,000 / 10 classes | 5,000 | 1,000 | 59,000 | 동일 클래스 | **mAP@1,000** |

- `setting1` 고정. Flickr/NUS/CIFAR는 train ⊂ DB, **MS-COCO만 train 10K와 DB가 disjoint**.
- CIBHash 원 protocol과 **완전히 동일하다고 쓰지 않는다**; split manifest와 overlap count를 공개한다.
- 다른 protocol의 published number를 한 표에 **혼합하지 않는다**.

### 6.2 headline 지표 규약

- **mAP@R**, R은 데이터셋별 (CIFAR 1000 / 나머지 5000), CalcTopMap 관행.
- **base-Hamming** 거리 (`dna_distance_mode=base`).
- baseline `unique_code_ratio`는 **DB split** 기준 (23K Flickr / 107K MSCOCO).
  한 표 안에서 split 분모를 섞지 않는다.

### 6.3 baseline 공정 변환

- baseline의 이진 해시를 **2 bit → 1 base**로 변환하고, **공통 DP 투영**을 거쳐
  같은 base-Hamming mAP@R로 평가한다.
- 원 논문의 16/32/64-bit 수치는 **가져오지 않는다**. 동일 split·동일 post-processing으로 재평가한다.

### 6.4 텍스트 캡션

- 오프라인 **1회** 생성, 학습 그래프 밖. Qwen3 기반 축별 서술.
- 축은 **질문으로 정의**되며 라벨을 쓰지 않는다.
- 추론 경로에 캡션 없음.

### 6.5 재현에 필수인 실행 규약

1. **`GDNA_NUM_SEMANTIC_PARTS=5`를 python 시작 전에 export.**
2. **`-e = N + 1`** — ε 스케줄이 학습 지평 안에서 완주해야 한다.
3. `--hash_target_mode siglip_cos` 확인.
4. run 디렉토리는 immutable identity로 claim (F08).
5. 장시간 작업은 tmux 세션에서 실행.

---

## 7. 이 문서에서 **의도적으로 제외한 것** (현재 재조정 중)

`docs/EXPERIMENT_PROTOCOL_AUDIT_2026-08-13.md`가 재실행/재설계를 요구한 항목이므로,
**확정 사항으로 인용하지 말 것.**

| 제외 항목 | 관련 결함 | 현재 상태 |
|---|---|---|
| **고정 epoch N의 값** | F02, F03 | D1 train-only 선택으로 재도출 중 |
| **N 선택 규칙** (test 곡선 → train-only validation) | F02 | 프로토콜 변경됨 |
| **`val_split_ratio`** (0.0 → 선택 단계 0.1) | F02 | 단계별로 다름 |
| **LR schedule horizon** | F03 | 탐색/최종 분리 결정됨 |
| **adaptive top-p 창의 재튜닝** | — | 3개 데이터셋 재튜닝 진행 중 |
| **`lambda_codon_joint` 재선택** | — | M=6 시절 sweep, 재선택 예정 |
| **baseline 30-bit 수치** | F04, F05 | provenance 재생성 + CIMON 수정 필요 |
| **held-out codon decoding 수치** | F10 | 1-seed, MSCOCO train extract 누락 |
| **native-DNA baseline 15-base 결과** | F18 | 전환 미완 |
| **COCO dissection IoU** | F12 | absent-category FP 누락 |
| **20-base / 40-bit 패널** | F06, F07 | extraction assertion + GC 정합 필요 |
| **feature cache provenance** | F04 | v6prov 재생성본으로 전환 중 |
| **통계적 동률 문구** | F14 | paired/bootstrap 재분석 필요 |

또한 다음은 **모델 정의가 아니다**:
- `lambda_bu = 0.05` (BU05) — 확인 실험이며 champion은 **0.02**.
- `+ L_bio` 행 — `lambda_bio_constraint > 0`인 **별도 변형**.

---

## 8. 재현 체크리스트

```
□ GDNA_NUM_SEMANTIC_PARTS=5  (python 시작 전)
□ --num_semantic_parts 5 --num_codebooks 5 --num_codons_per_codebook 3
□ --backbone_type clip, openai/clip-vit-base-patch16, --freeze_backbone
□ --hash_target_mode siglip_cos            ← 비지도 invariant
□ --no_gumbel_softmax                      ← use_gumbel_softmax=False
□ --lambda_codeword_codon_sinkhorn 0.0     ← bijection OFF (CIFAR 포함)
□ --routing_adaptive_topp --routing_adaptive_topp_min 0.6 --max 0.95
□ --sinkhorn_epsilon_init {1.0 | NUS 0.5} --sinkhorn_epsilon_final 0.1
□ -e = N+1                                 ← ε 스케줄 정렬
□ --codebook_size {128 | CIFAR 64}
□ --lambda_codon_joint {C .03 | F .02 | N .05 | M .03}
□ --text_embed_transform partial_whiten --text_whiten_gamma 0.25
□ --codon_input_source quantized --codon_residual_gamma 0.0
□ --local_residual_quant OFF
□ --dna_distance_mode base
□ GC policy = gc-40-60-inclusive-v1, L=15 → [6,9], run ≤ 3
□ seeds 42 / 43 / 44
```

---

## 9. 출처

| 내용 | 출처 |
|---|---|
| 전 파라미터 343개 | champion `args.txt` (§ 상단) |
| 활성 손실항 | 같은 실행의 `log.csv` terminal-epoch 값 |
| 아키텍처 서술 | `docs/DRAFT_GROUNDEDNDA_PAPER_KO_MODIFY.md` §3.2–3.9 |
| 데이터셋 recipe | 같은 문서 §4.2, §4.2b |
| 학습 프로토콜 | 같은 문서 §4.3 |
| GC 정책 | `dna_utils/gc_policy.py` |
| 제외 항목 | `docs/EXPERIMENT_PROTOCOL_AUDIT_2026-08-13.md` §3 |
