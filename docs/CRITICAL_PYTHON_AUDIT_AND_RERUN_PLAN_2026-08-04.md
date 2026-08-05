# GroundedDNA 치명 결함 수정 및 재실행 계획

> 감사 기준일: 2026-08-04
>
> 대상: 저장소의 전체 Python 실행 경로와 현재 실험 산출물
>
> 상태: 코드 수정 및 회귀 테스트 완료, 고비용 실험 재실행은 아직 완료되지 않음
> 원칙: 기존 산출물은 삭제하거나 덮어쓰지 않고, 새 태그와 새 결과 루트로 재생성한다.

이 문서는 2026-08-04 Python 코드 감사에서 확인된 결과 영향급 결함, 실제 수정 내용, 무효화되는 기존 실험, 재실행 명령, 완료 판정 기준을 한곳에 고정한다. 이후 수치나 표를 갱신할 때는 이 문서를 체크리스트로 사용한다.

## 1. 결론과 의사결정 표

| ID | 결함 | 기존 산출물의 상태 | 필요한 조치 | 우선순위 |
|---|---|---|---|---:|
| F1 | A2 no-text에서 텍스트 누출 | A2 checkpoint와 그 파생 지표 무효 | 4개 데이터셋의 P0 selection과 refit을 모두 처음부터 재학습 | P0 |
| F2 | A4 shared-codebook이 실제로는 bank 0만 조회하고 slot별 bank를 따로 갱신 | A4 checkpoint와 그 파생 지표 무효 | 4개 데이터셋의 P0 selection과 refit을 모두 처음부터 재학습 | P0 |
| F3 | CIBHash head, scheduler, horizon이 공식 구현과 불일치 | 과거 CIBHash 결과를 canonical baseline으로 사용할 수 없음 | 36/48-bit × 4 datasets × seeds 42/43/44, 총 24셀 재학습 | P0 |
| F4 | OrthoHash가 첫 periodic evaluation 후 eval mode에 고착 | 두 MSCOCO legacy run 무효 | 해당 legacy 설정을 처음부터 재학습 | P2 |
| F5 | Slot intervention이 invalid DNA, 불안정한 tie, 불일치 dose를 사용 | 기존 slot-intervention JSON 전부 무효 | checkpoint는 유지하고 수정된 evaluator로 재평가 | P0 |
| F6 | Codebook-drop을 raw DNA에서 수행하고 24-base geometry를 잘못 처리 | 기존 raw JSON은 deployed-DNA 기여도 근거로 사용할 수 없음 | 최종 checkpoint를 bio-projected evaluator로 재평가 | P0 |
| F7 | FAIR/local-crop cache가 local token뿐 아니라 global embedding도 교체 | 수치는 다른 조건의 결과이며 token-only 주장에는 무효 | cache 재생성 후 그 cache를 사용한 실험 재학습 | P1 |
| F8 | Source-SHA 예외가 variant 범위를 벗어나 old CIBHash 또는 pre-dispatch CRH를 승인 가능 | aggregate/resume admission 신뢰 불가 | 수정된 validator로 재집계; 차단된 셀만 재실행 | P0 |

상태 용어는 다음과 같이 사용한다.

- **무효 / retrain**: checkpoint 자체가 잘못 학습되어 처음부터 재학습해야 한다.
- **무효 / reevaluate**: checkpoint는 유지할 수 있지만 기존 평가 JSON과 수치를 다시 계산해야 한다.
- **재해석**: 수치 계산은 실제 입력에 대해 맞지만 문서가 주장한 실험 조건과 다르다.
- **영향 없음**: 수정된 분기나 파일을 사용하지 않아 재실행할 필요가 없다.

## 2. 감사 범위와 코드 검증 상태

- 현재 저장소의 Python 파일 199개를 인벤토리화했다.
- 미등록된 기존 미완성 stub인 baseline/SPQ.py를 제외한 runnable Python 파일 198개가 compile을 통과했다.
- 전체 회귀 테스트 결과는 **358 passed, 29 subtests passed**이다.
- git diff --check를 통과했다.
- 정상 non-ablation 경로와 non-shared codebook 경로는 변경 전 기준과 bit-exact하게 동일함을 회귀 테스트로 확인했다.
- 현재 핵심 source digest는 다음과 같다.

| 파일 | SHA-256 |
|---|---|
| baseline/CIBHash.py | ce1a1e3fde2c87eb9fe34644e11f63ca4c84fb757ac0eb17c126ccf0cabc3c8e |
| scripts/run_modern_baseline_p0.py | 1dec886eaed08b4f01cc04c8ebaec81b01952d1bc6913696cdcc7a75830461e0 |

### 2.1 감사에서 직접 확인한 결과 영향

| 결함 | 관측 증거 |
|---|---|
| A2 text leak | 최신 refit의 text NT-Xent / cross-modal loss가 CIFAR10 3.8344/0.6031, Flickr25k 3.9619/0.6027, MSCOCO 3.7185/0.6652, NUS-WIDE 3.7129/0.8693으로 모두 0이 아니었음 |
| A4 split update | Flickr A4 checkpoint의 bank 1…5 대 bank 0 cosine이 약 0.045–0.048로, shared state라고 볼 수 없었음 |
| CIBHash | 512→36 기준 old head 18,468 parameters 대 corrected head 562,212 parameters |
| Slot intervention | 최신 Flickr 결과의 exact-splice DNA validity가 slot별 약 79%, 74%, 75%, 71%, 65%, 61%에 불과했음 |
| Ranking tie | 100-query audit에서 old 대 stable top-100 평균 Jaccard 0.6574, 최솟값 0.2121; boundary tie 평균 135개, 최대 381개 |
| FAIR global contamination | donor global과 cache global의 평균 cosine이 CUB FAIR 0.959241, Flickr FAIR 0.950521, MSCOCO FAIR 0.950530, CUB text-local 0.936836, NUS train-only 0.936763이었고 exact row match는 모두 0건 |

이 수치는 결함의 존재를 확인하기 위한 audit evidence이며 새 실험 결과로 인용하지 않는다.

## 3. 수정 내역

### 3.1 F1 — A2 no-text 텍스트 누출

#### 결함

disable_text_supervision이 routing 일부만 끄고 cached factual/foil text tensor를 남겼다. 활성화된 text loss가 있으면 후반 fallback이 cached text를 adapter, quantizer, codon head에 다시 통과시켰다. text-derived codebook initialization도 별도로 실행할 수 있었다.

실제 최신 A2 로그에서 no-text인데도 text NT-Xent와 cross-modal loss가 0이 아니었다. 따라서 이 문제는 명목상 flag 문제가 아니라 학습 gradient와 선택 epoch를 바꾼 결과급 결함이다.

#### 수정

- [model_siglip2.py](../model_siglip2.py#L3504)
  - forward 진입점에서 live/cached factual·foil pooled/token text를 모두 제거한다.
  - compute_text_foil을 강제로 끈다.
- [train_siglip2.py](../train_siglip2.py#L600)
  - no-text와 text-derived codebook initialization의 결합을 즉시 거부한다.

#### 검증

- 서로 극단적으로 다른 whitening bundle을 주입해도 전체 출력과 loss가 bit-exact하게 같다.
- text 관련 loss가 정확히 0이다.
- live text ID도 backbone text extraction 전에 제거된다.
- 회귀 테스트: [test_ablation_integrity.py](../tests/test_ablation_integrity.py#L97)

### 3.2 F2 — A4 shared-codebook의 비공유 학습

#### 결함

lookup은 모든 slot에서 bank 0을 확장해 사용했지만 EMA, dead-code revival, repulsion, text initialization은 물리적인 M개 bank를 각각 갱신했다. 결과적으로 lookup에 실제로 사용되는 bank 0은 global slot만 학습했고 local slot의 관측은 사용되지 않는 bank 1…M−1에 쌓였다.

#### 수정

- [model_siglip2.py](../model_siglip2.py#L503)
  - bank 0을 유일한 canonical shared state로 정의했다.
  - lookup, anchor, loss, EMA, revival, repulsion, text initialization, serialization이 모두 effective shared view를 사용한다.
- [model_siglip2.py](../model_siglip2.py#L823)
  - B×M assignment를 하나의 bank 0 EMA에 집계한다.
- [train_siglip2.py](../train_siglip2.py#L1049)
  - hierarchical refresh도 effective shared bank와 mask를 사용한다.
- shared mode와 adaptive-K, split, 기존 warm-start의 모호한 조합은 fail-closed 처리한다.

#### 검증 및 해석 제한

- local slot만 바꿔도 실제 shared lookup bank가 갱신된다.
- gradient는 bank 0에만 누적되고 저장 시 compatibility row가 bank 0과 동일하다.
- 회귀 테스트: [test_ablation_integrity.py](../tests/test_ablation_integrity.py#L286)

수정된 A4는 **여섯 slot이 하나의 strict shared bank를 사용하는 조건**이다. codebook-mean inference anchor도 slot 간 같아지므로, 이를 단순히 “quantizer만 공유한 조건”이라고 기술하면 안 된다.

### 3.3 F3 — CIBHash source fidelity

#### 결함

과거 adapter는 단일 D→bit linear head, exponential LR scheduler, 100 epochs를 사용했다. 공식 구현은 D→1024→ReLU→bit head, fixed Adam, 기본 60 epochs이다.

512차원 입력과 36비트 출력에서 과거 head는 18,468 parameters, 수정 head는 562,212 parameters이다. 모델 용량과 optimization trajectory가 모두 달라 기존 수치를 공식 CIBHash reproduction으로 볼 수 없다.

공식 근거:

- [CIBHash official model and Adam optimizer](https://raw.githubusercontent.com/zexuanqiu/CIBHash/4c896ee1bab8e8a6060b7343aad06d483658ff78/model/CIBHash.py)
- [CIBHash official 60-epoch default](https://raw.githubusercontent.com/zexuanqiu/CIBHash/4c896ee1bab8e8a6060b7343aad06d483658ff78/model/base_model.py)

#### 수정

- [baseline/CIBHash.py](../baseline/CIBHash.py#L16)
  - 전용 Linear(D,1024) → ReLU → Linear(1024,bit) cached-feature head를 추가했다.
  - canonical run은 fixed Adam, scheduler none, 60-epoch horizon을 강제한다.
  - P0 stage 1은 정확히 60 epochs, stage 2는 validation-selected E* 이하만 허용한다.
  - 구형 single-linear checkpoint는 historical diagnostic으로만 명시적으로 load할 수 있다.
- [run_modern_baseline_p0.py](../scripts/run_modern_baseline_p0.py#L87)
  - CIBHash horizon을 100에서 60으로 수정했다.
- [run_baseline_p0_matrix.py](../scripts/run_baseline_p0_matrix.py#L217)
  - resume admission을 정확한 source digest와 variant 조합에 묶었다.
- [aggregate_baseline_p0_matrix.py](../scripts/aggregate_baseline_p0_matrix.py#L139)
  - aggregate admission에도 동일한 variant-scoped transition 규칙을 적용했다.

#### 검증

- 정확한 layer 순서, parameter 수, fixed LR, 60-epoch guard를 검증했다.
- old runner SHA는 CIBHash에는 과학적 차이로 차단되지만 unaffected Cimon에는 불필요하게 전파되지 않는다.
- pre-CRH-dispatch base_model SHA는 CRH에는 허용되지 않는다.
- 회귀 테스트: [test_cibhash_source_fidelity.py](../tests/test_cibhash_source_fidelity.py#L18), [test_crh_supervised_matrix.py](../tests/test_crh_supervised_matrix.py#L172)

### 3.4 F4 — OrthoHash train/eval mode

#### 결함

periodic evaluation이 model.eval()을 호출한 뒤 다음 epoch에서 model.train()을 복구하지 않았다. BatchNorm이 활성화된 설정에서는 첫 평가 이후 running statistics와 train-time 동작이 동결됐다.

#### 수정

- [baseline/OrthoHash.py](../baseline/OrthoHash.py#L300)
  - 매 epoch 시작 시 model.train()을 명시한다.
- 회귀 테스트: [test_orthohash_training_mode.py](../tests/test_orthohash_training_mode.py#L12)

### 3.5 F5 — Slot intervention protocol

#### 결함

- 큰 Hamming boundary tie에서 argpartition이 임의의 neighbor를 선택했다.
- bio projection 후 exact slot splice가 최종적으로 valid DNA인지 확인하지 않았다.
- ours와 control이 서로 다른 query subset 또는 서로 다른 변경량으로 비교될 수 있었다.
- no-op donor/control이 포함될 수 있었다.
- baseline이 semantic donor 선택에 영향을 줄 수 있었다.

#### 수정

- [slot_intervention_eval.py](../scripts/slot_intervention_eval.py#L53)
  - Hamming distance 뒤 database row index로 정렬하는 stable tie policy를 사용한다.
- [slot_intervention_eval.py](../scripts/slot_intervention_eval.py#L116)
  - target segment만 exact splice하고 전체 DNA validity를 검사한다.
- [slot_intervention_eval.py](../scripts/slot_intervention_eval.py#L393)
  - 모든 arm에서 non-noop, validity, query별 exact dose 일치를 강제한다.
- [slot_intervention_eval.py](../scripts/slot_intervention_eval.py#L507)
  - 모든 arm이 가능한 동일 paired subset만 평가한다.
- [slot_intervention_eval.py](../scripts/slot_intervention_eval.py#L641)
  - protocol version, tie, validity, dose, coverage provenance를 출력한다.
- 회귀 테스트: [test_slot_intervention_eval.py](../tests/test_slot_intervention_eval.py#L23)

현재 evaluator는 baseline run manifest와 checkpoint digest를 아직 결합하지 않는다. 따라서 수정된 출력도 paper_result_eligible=false이다. 이는 수치가 다시 틀렸다는 뜻이 아니라, **corrected diagnostic과 paper-admitted evidence를 구분하는 fail-closed 표시**이다.

### 3.6 F6 — Codebook-drop protocol

#### 결함

기존 결과는 raw/unprojected base coordinate에서 slot을 제거했지만 문서에서는 deployed bio-valid DNA의 slot contribution처럼 해석했다. 느린 구현은 slot당 항상 3 bases라고 가정하여 24-base, 즉 6×4 geometry를 잘못 처리했다.

#### 수정

- [codebook_drop_ablation_fast.py](../scripts/codebook_drop_ablation_fast.py#L35)
  - R % M 기반의 일반 slot slice를 사용한다.
  - stable Hamming ranking을 사용한다.
  - bio projection 후 100% validity를 확인한다.
- [codebook_drop_ablation_fast.py](../scripts/codebook_drop_ablation_fast.py#L105)
  - bio_project, GC bounds, max-run 옵션과 provenance를 추가했다.
- [codebook_drop_ablation.py](../scripts/codebook_drop_ablation.py#L67)
  - 느린 구현도 같은 geometry와 projection helper를 사용한다.
- 회귀 테스트: [test_codebook_drop_protocol.py](../tests/test_codebook_drop_protocol.py#L16)

drop 연산은 target coordinate를 query와 DB 양쪽에서 같은 상수로 만들어 Hamming distance 축을 제거하는 연산이다. 변환된 row 자체는 방출 가능한 DNA라고 주장하지 않는다. 또한 projection manifest binder가 아직 없으므로 현재 출력은 paper_result_eligible=false이다.

### 3.7 F7 — FAIR/local-crop cache contract

#### 결함

local-token-only intervention이어야 했지만 generator가 선택 crop들의 pooled global 평균으로 visual_global과 augmented globals를 덮어썼다. 출력 디렉터리를 재사용하면 이전 donor의 text, row, whitening 또는 augmentation sidecar가 남을 수도 있었다.

#### 수정

- [extract_clip_local_crops.py](../extract_clip_local_crops.py#L74)
  - donor full-image main/aug global의 존재, shape, dtype을 먼저 검증한다.
- [extract_clip_local_crops.py](../extract_clip_local_crops.py#L105)
  - crop-derived global을 만들지 않고 donor full-image global을 원자적으로 연결한다.
- [extract_clip_local_crops.py](../extract_clip_local_crops.py#L122)
  - row/text sidecar를 현재 donor에 다시 묶고 stale optional sidecar를 제거한다.
- [extract_clip_local_crops.py](../extract_clip_local_crops.py#L505)
  - metadata에 donor_full_image source와 정확한 global file 집합을 기록한다.
- [dataloaders.py](../dataloaders.py#L49)
  - geometry, row count, aug 집합, dtype과 shape를 검증한다.
- [dataloaders.py](../dataloaders.py#L85)
  - legacy crop-global cache를 거부한다.
- [dataloaders.py](../dataloaders.py#L145)
  - text, IDs, main/aug globals가 실제 donor 파일과 samefile인지 검사한다.
- 회귀 테스트: [test_local_crop_cache_contract.py](../tests/test_local_crop_cache_contract.py#L87)

과거 checkpoint의 수치는 “crop token + selected-crop-mean global” 조건에는 실제 측정값이다. 그러나 “local token만 바꾼 single-delta” 또는 “global slot 불변”의 근거로는 사용할 수 없다.

### 3.8 F8 — Variant-scoped source admission

#### 결함

baseline matrix의 resume/aggregate 검증은 공용 runner나 dispatcher의 과거 SHA를 경로 단위의 “비과학적 변경”으로 허용했다. 그러나 같은 파일 변경도 variant마다 의미가 달랐다. CIBHash horizon을 100에서 60으로 바꾼 runner transition은 CIBHash에는 과학적 변경이지만 다른 기존 baseline에는 비과학적 변경이고, CRH dispatch 추가 전의 base_model.py는 기존 baseline에는 허용할 수 있지만 CRH 결과를 생성할 수는 없다. 기존 path-wide 예외는 old-horizon CIBHash나 pre-dispatch CRH manifest를 잘못 승인할 수 있었다.

#### 수정

- [run_baseline_p0_matrix.py](../scripts/run_baseline_p0_matrix.py#L217)
  - source alias를 path뿐 아니라 recorded digest와 variant의 조합으로 제한한다.
  - old-horizon CIBHash와 pre-dispatch CRH는 resume 대상에서 차단하되, 동일 transition의 영향을 받지 않은 기존 baseline은 유지한다.
- [aggregate_baseline_p0_matrix.py](../scripts/aggregate_baseline_p0_matrix.py#L139)
  - reviewed digest 집합과 digest별 허용 variant를 aggregate audit에도 동일하게 적용한다.
  - mismatch를 cell variant와 결합해 comparison-safe 여부를 판정한다.
- CIBHash canonical source profile과 horizon을 corrected digest 및 60 epochs로 갱신했다.
- 회귀 테스트: [test_crh_supervised_matrix.py](../tests/test_crh_supervised_matrix.py#L172), [test_baseline_matrix_aggregation.py](../tests/test_baseline_matrix_aggregation.py#L100), [test_modern_driver_protocol.py](../tests/test_modern_driver_protocol.py#L51)

이 수정 자체는 정상 checkpoint의 학습값을 바꾸지 않는다. 수정된 validator로 기존 manifest를 다시 검사한 뒤 실제로 차단되는 cell만 재실행한다. 단, old CIBHash는 F3 때문에 전부 재학습 대상이고, pre-dispatch source를 주장하는 CRH는 유효한 CRH run으로 승인하지 않는다.

## 4. 무효 또는 격리해야 하는 기존 산출물

기존 파일을 삭제하지 않는다. 문서, 표, plotting pipeline에서 제외하고 새 산출물과 명확히 구분한다.

### 4.1 A2/A4 checkpoint

디렉터리 이름보다 저장된 args가 authoritative하다. `result/260715+...` 여섯 개는 archive를 가리키는 심볼릭 링크이므로 `find -L`로 링크 대상을 따라가야 한다. 다음 명령은 현재 영향받는 디렉터리 24개(일반 디렉터리 18개 + archive symlink 6개)를 재현 가능하게 찾는다.

    find -L result -name args.txt -print0 |
      xargs -0 rg -l \
        '^disable_text_supervision-+True$|^share_codebook-+True$' |
      sed 's#/args.txt$##' |
      sort

현재 확인된 legacy final run은 다음과 같다.

- result/260715+cifar10_setting1_cifar10_A2_noText_wholeimg+bs+64+e+60+proj_lr+0.001
- result/260715+flickr25k_setting1_flickr_A2_noText_wholeimg+bs+64+e+60+proj_lr+0.001
- result/260715+mscoco_setting1_mscoco_A2_noText_wholeimg+bs+64+e+60+proj_lr+0.001
- result/260715+nuswide_setting1_nuswide_v185_sweep_A2noText_w0.15_x0.05_th0.05_tk0.05_cb1.5_ccs0.0_g4.595+bs+64+e+60+proj_lr+0.001
- result/260715+flickr25k_setting1_flickr_A4_sharedCB_K768_wholeimg+bs+64+e+60+proj_lr+0.001
- result/260715+mscoco_setting1_mscoco_A4_sharedCB_K768_wholeimg+bs+64+e+60+proj_lr+0.001

추가로 다음 P0 계열의 selection과 refit이 모두 무효다.

- result/260729+nuswide_setting1_promptAblA_nuswide_A_v4_A4shared_P0val...
- result/260729+nuswide_setting1_promptAblA_nuswide_A_v4_A4shared_P0refit...
- result/260804+...promptAblA...abA2_P0val...
- result/260804+...promptAblA...abA2_P0refit...
- result/260804+...promptAblA...abA4_P0val...
- result/260804+...promptAblA...abA4_P0refit...

selection E* 자체가 오염되었으므로 old E*를 고정해 refit만 다시 돌리면 안 된다.

다음 파생 파일도 모두 stale이다.

- docs/heldout_decoding_*A2*.json
- docs/heldout_decoding_*A4*.json
- docs/heldout_decoding_*abA2*.json
- docs/heldout_decoding_*abA4*.json
- docs/sweep_rows/*_abA2.json
- docs/sweep_rows/*_abA4.json
- A2/A4 checkpoint에서 계산한 codebook alignment, decoding, NMI, drop 및 intervention artifact

### 4.2 CIBHash

경로명이나 날짜가 아니라 manifest의 implementation SHA와 encoder spec이 authoritative하다. 수정 전 baseline/CIBHash.py 또는 100-epoch runner로 생성된 모든 CIBHash checkpoint와 extraction은 canonical comparison에서 제외한다.

대표 stale 범위:

- result_baseline/**/cibhash*
- result_baseline/p0_matrix*/u0_cibhash_*
- docs/baseline_val_select/cibhash_*.json
- docs/baseline_p0_stage2_partial/cibhash_*.json
- docs/baseline_p0_matrix*.json 및 대응 Markdown의 CIBHash row
- docs/baselines_final_epoch_mapr.json의 CIBHash 수치
- docs/p0_comparison.json의 CIBHash 수치
- docs/comparison_Achampion_vs_baselines_2026-07-27.md의 18-base와 24-base CIBHash row 및 CIBHash 기반 margin

특히 해당 비교 문서의 24-base CIBHash 네 수치 0.8057, 0.8018, 0.8074, 0.8994도 동일 결함의 영향을 받는다.

### 4.3 OrthoHash

다음 두 디렉터리는 batch_norm=true, eval_period=20, max_epoch=60이므로 무효다.

- result_baseline/260512/orthohash_mscoco
- result_baseline/260514/orthohash_mscoco_dnacompare

Flickr25k와 CIFAR10의 legacy OrthoHash는 평가가 final-only여서 이 mode 고착 결함의 영향을 받지 않는다. 현재 U0 matrix의 OH는 baseline/OH.py를 사용하며 baseline/OrthoHash.py와 별개이므로 재실행 대상이 아니다.

### 4.4 Slot intervention

현재 존재하는 다음 여덟 JSON은 모두 stale이다.

- docs/slot_intervention_flickr25k.json
- docs/slot_intervention_flickr25k_bioproj.json
- docs/slot_intervention_mscoco.json
- docs/slot_intervention_mscoco_bioproj.json
- docs/slot_intervention_nuswide.json
- docs/slot_intervention_nuswide_bioproj.json
- docs/newmodel_analysis/slot_intervention_flickr.json
- docs/newmodel_analysis/slot_intervention_nuswide.json

이를 인용하는 paper draft, FINDINGS 문서, PROJECT_LOG의 exact 수치와 figure도 새 결과가 나오기 전까지 보류한다.

### 4.5 Codebook-drop

다음 규칙에 해당하는 기존 raw 결과는 deployed-DNA contribution 근거로 사용하지 않는다.

- result/**/codebook_drop_ablation*.json 중 _bioproj가 없는 파일
- result_baseline/**/codebook_drop_ablation*.json 중 _bioproj가 없는 파일

감사 시점에 result 아래에는 이 규칙에 해당하는 JSON 19개가 있었다. 역사적 raw diagnostic으로 보존할 수 있지만 paper-facing slot contribution 표에는 넣지 않는다.

### 4.6 FAIR/local/grid cache

다음 cache는 직접 또는 파생 형태로 legacy crop-global을 포함한다.

- cache/cub200_clip_v6bplus_localL8K3
- cache/cub200_clip_v6bplus_FAIRrankL8K1
- cache/cub200_clip_v6bplus_FAIRrankL8K3
- cache/cub200_clip_v6bplus_tokens_FAIRrankL8K3
- cache/cub200_clip_v6bplus_grid3K3
- cache/cub200_clip_v6bplus_grid4K3
- cache/cub200_clip_v7_1_FAIRrankL8K3
- cache/cub200_clip_v7_1_tokens_FAIRrankL8K3
- cache/flickr25k_clip_v4plus_qwen3_tokens_localL8K3
- cache/flickr25k_clip_v4plus_qwen3_tokens_FAIRrankL8K3
- cache/mscoco_clip_v5b_FAIRrankL8K3
- cache/nuswide_clip_FAIRrankL8K3_trainonly
- cache/nuswide_clip_FAIRrankL8K3_testonly
- cache/nuswide_clip_FAIRrankL8K3_tokens

영향 여부의 authoritative rule은 result의 args 또는 config에서 siglip2 feature cache가 위 디렉터리 중 하나로 resolve되는지 확인하는 것이다. 감사 당시 smoke와 OLD archive를 포함해 78개 result 디렉터리가 이 범주에 해당했다.

이 78개를 모두 기계적으로 재학습할 필요는 없다. 유지할 paper-facing FAIR/local claim의 대표 cell과 필요한 control만 재실행하고, 폐기된 exploratory run은 “legacy crop-global condition”으로 archive한다.

## 5. 재실행 순서

권장 dependency는 다음과 같다.

1. 수정 코드와 source digest를 고정한다.
2. CIBHash corrected matrix를 새 root에서 실행한다.
3. A2/A4의 P0 selection과 refit을 새 tag로 실행한다.
4. corrected A2/A4 checkpoint의 held-out decoding과 파생 지표를 다시 계산한다.
5. 현재 최종 A checkpoint의 slot intervention과 codebook-drop을 다시 계산한다.
6. FAIR/local claim을 유지할 경우 cache를 새 versioned path로 재생성하고 관련 모델을 재학습한다.
7. legacy OrthoHash MSCOCO 수치를 계속 인용할 경우 해당 설정을 재학습한다.
8. 모든 표와 draft를 새 artifact만 사용해 다시 생성한다.

CIBHash와 A2/A4 학습 자체는 서로 병렬 실행할 수 있다. 다만 flat-control comparison에 CIBHash를 포함하는 decoding/intervention은 corrected CIBHash extraction이 준비된 뒤 확정한다.

## 6. 재실행 명령

아래 GPU 번호는 예시다. 실행 직전에 실제 예약 상태를 확인해 변경한다.

### 6.1 CIBHash full corrected matrix

현재 18-base main comparison만 최소 복구하려면 36-bit 12셀이 필요하다. 현재 문서의 18/24-base capacity 표를 모두 유지하려면 36/48-bit × 4 datasets × 3 seeds, 총 24셀을 실행해야 한다. 결함은 bit 길이와 무관하므로 이 문서는 24셀 full rerun을 기준으로 한다.

먼저 dry run으로 정확히 24개가 계획되는지 확인한다.

    PY=/home/yschoi/.conda/envs/dna_hashing/bin/python
    $PY scripts/run_baseline_p0_matrix.py \
      --gpus 4 5 \
      --panel u0 \
      --variants cibhash \
      --datasets Flickr25k MSCOCO NUSWIDE CIFAR10 \
      --bits 36 48 \
      --seeds 42 43 44 \
      --model-root params_baseline/p0_cibhash_sourcefix_20260804 \
      --result-root result_baseline/p0_cibhash_sourcefix_20260804 \
      --compress-root compress_baseline/p0_cibhash_sourcefix_20260804 \
      --log-root logs/p0_cibhash_sourcefix_20260804 \
      --dry-run

계획을 확인한 뒤 같은 명령에서 dry-run만 제거한다.

    $PY scripts/run_baseline_p0_matrix.py \
      --gpus 4 5 \
      --panel u0 \
      --variants cibhash \
      --datasets Flickr25k MSCOCO NUSWIDE CIFAR10 \
      --bits 36 48 \
      --seeds 42 43 44 \
      --model-root params_baseline/p0_cibhash_sourcefix_20260804 \
      --result-root result_baseline/p0_cibhash_sourcefix_20260804 \
      --compress-root compress_baseline/p0_cibhash_sourcefix_20260804 \
      --log-root logs/p0_cibhash_sourcefix_20260804

집계:

    $PY scripts/aggregate_baseline_p0_matrix.py \
      --result-root result_baseline/p0_cibhash_sourcefix_20260804 \
      --seeds 42 43 44 \
      --panels u0 \
      --out-json docs/baseline_cibhash_sourcefix_20260804.json \
      --out-markdown docs/baseline_cibhash_sourcefix_20260804.md

aggregate의 u0 completeness target은 CIBHash만이 아니라 전체 U0 panel이므로 CIBHash-only root에 require-complete를 붙이지 않는다. 대신 출력 JSON에서 CIBHash의 4 datasets × 2 budgets, 총 8 aggregate가 모두 all_seeds_complete=true인지 확인한다.

현재 scripts/queue_cibhash_cifar_rerun.sh는 CIFAR10 36-bit만 실행하므로 full 복구 명령을 대신하지 못한다.

현재 four-dataset main/capacity matrix 밖의 CUB-200 및 과거 standalone CIBHash 수치도 source-fidelity 결함 자체에는 동일하게 노출되어 있다. 그 수치를 역사적 비교에서 계속 인용하려면 해당 dataset/config를 corrected head로 별도 재실행한다. 현 main matrix에 포함되지 않는다는 이유만으로 old CIBHash 수치가 다시 유효해지는 것은 아니다.

#### CIBHash 완료 기준

- stage 1 config의 max_epoch와 schedule_horizon이 모두 60이다.
- encoder_layers가 cibhash_d_to_1024_relu_to_bit이다.
- lr_scheduler가 none이다.
- 각 셀에 selection JSON, refit checkpoint, extract_query.npz, extract_db.npz, DNA projection artifact와 p0_run_manifest.json이 있다.
- 36-bit 결과는 18 bases, 48-bit 결과는 24 bases다.
- 24-base projection은 GC count 10–14와 homopolymer ≤3을 만족하고 validity가 100%다.
- manifest가 위 2절의 corrected source digest를 기록한다.
- legacy cache를 계속 사용하면 corrected 수치여도 diagnostic-only이다. paper main table에 넣으려면 immutable transform/model provenance가 있는 strict cache를 먼저 재생성해야 한다.

### 6.2 A2/A4 eight-cell rerun

반드시 새 tag를 사용한다. 기존 queue_ablations_newmodel.sh를 그대로 다시 실행하면 같은 tag의 result 또는 docs JSON을 덮을 수 있다.

다음은 end-to-end launcher를 사용하는 정확한 8셀 예시다.

#### A2

    bash scripts/sweep_joint_cell.sh 0 mscoco_A_v5b abA2_auditfix_20260804 \
      "--lambda_codon_joint 0.03 --no_gumbel_softmax --disable_text_supervision"

    bash scripts/sweep_joint_cell.sh 1 nuswide_A_v4 abA2_auditfix_20260804 \
      "--lambda_codon_joint 0.05 --no_gumbel_softmax --disable_text_supervision"

    bash scripts/sweep_joint_cell.sh 2 flickr_A_v4 abA2_auditfix_20260804 \
      "--lambda_codon_joint 0.02 --no_gumbel_softmax --disable_text_supervision"

    bash scripts/sweep_joint_cell.sh 3 cifar_A_v4 abA2_auditfix_20260804 \
      "--lambda_codon_joint 0.03 --no_gumbel_softmax --lambda_codeword_codon_sinkhorn 0.0 --disable_text_supervision"

#### A4

    K=768 bash scripts/sweep_joint_cell.sh 0 mscoco_A_v5b abA4_auditfix_20260804 \
      "--lambda_codon_joint 0.03 --no_gumbel_softmax --share_codebook"

    K=768 bash scripts/sweep_joint_cell.sh 1 nuswide_A_v4 abA4_auditfix_20260804 \
      "--lambda_codon_joint 0.05 --no_gumbel_softmax --share_codebook"

    K=768 bash scripts/sweep_joint_cell.sh 2 flickr_A_v4 abA4_auditfix_20260804 \
      "--lambda_codon_joint 0.02 --no_gumbel_softmax --share_codebook"

    K=768 bash scripts/sweep_joint_cell.sh 3 cifar_A_v4 abA4_auditfix_20260804 \
      "--lambda_codon_joint 0.03 --no_gumbel_softmax --lambda_codeword_codon_sinkhorn 0.0 --share_codebook"

sweep_joint_cell.sh의 training과 own-model extraction은 사용할 수 있다. 그러나 현재 스크립트가 held-out flat control용 CIBHash 경로를 오래된 matrix root에 고정하고 있으므로, 자동 생성된 heldout decoding과 sweep row의 baseline comparison은 corrected CIBHash 경로로 다시 계산하기 전까지 격리한다.

flat baseline 없이 A0−A2와 A0−A4 자체만 비교할 때는 heldout_codon_decoding.py를 새 checkpoint에 직접 실행할 수 있다. MSCOCO는 extract_train.npz가 필요하고, CIFAR10은 scripts/cifar_inject_image_ids.py로 withids overlay를 만든 뒤 사용한다.

    $PY scripts/heldout_codon_decoding.py \
      --ours_dir NEW_A2_OR_A4_RESULT_DIR \
      --dataset DATASET \
      --train_manifest TRAIN_MANIFEST_IF_NEEDED \
      --bio_project \
      --out docs/heldout_decoding_DATASET_A2_OR_A4_auditfix_20260804.json

corrected flat comparison을 만들 때만 다음을 추가한다.

    --baseline_dirs CORRECTED_CIB_DNAEVAL_DIR OTHER_CONTROL_DIRS \
    --baseline_names CIBHash OTHER_CONTROL_NAMES

#### A2/A4 완료 기준

- 각 cell의 P0val log에서 새 E*를 선택한다.
- old E*를 재사용하지 않는다.
- 새 P0refit result에 cell_result.json, extract_query.npz, extract_db.npz가 있다.
- A2 args에서 disable_text_supervision=True, text_init_codebook=none이다.
- A2 training/eval에서 text-derived losses가 정확히 0이다.
- A4 args에서 share_codebook=True, K=768, adaptive-K와 warm-start가 비활성이다.
- A4 checkpoint의 effective bank가 모든 slot에서 동일하다.
- 새 held-out decoding, sweep row, codebook alignment, NMI, drop 결과만 새 표에 사용한다.
- A4 설명은 strict shared-bank 조건으로 쓴다.

### 6.3 Current champion codebook-drop 재평가

현재 locked single-seed 대표 checkpoint에 대해 먼저 다음 네 개를 실행한다.

    $PY scripts/codebook_drop_ablation_fast.py \
      --result_dir "result/260803+flickr25k_setting1_promptAblA_flickr_A_v4_uni002_P0refit_e4+bs+64+e+60+proj_lr+0.001" \
      --subset_queries 2000 --seed 0 --num_codebooks 6 --bio_project

    $PY scripts/codebook_drop_ablation_fast.py \
      --result_dir "result/260803+mscoco_setting1_promptAblA_mscoco_A_v5b_uni003_P0refit_e39+bs+64+e+60+proj_lr+0.001" \
      --subset_queries 2000 --seed 0 --num_codebooks 6 --bio_project

    $PY scripts/codebook_drop_ablation_fast.py \
      --result_dir "result/260803+nuswide_setting1_promptAblA_nuswide_A_v4_uni005_P0refit_e4+bs+64+e+60+proj_lr+0.001" \
      --subset_queries 2000 --seed 0 --num_codebooks 6 --bio_project

    $PY scripts/codebook_drop_ablation_fast.py \
      --result_dir "result/260803+cifar10_setting1_promptAblA_cifar_A_v4_uniB003_P0refit_e19+bs+64+e+60+proj_lr+0.001" \
      --subset_queries 2000 --seed 0 --num_codebooks 6 --bio_project

출력 파일명은 codebook_drop_ablation_subset2000_bioproj.json이다. 데이터셋의 query 수가 2,000보다 작으면 실제 전체 query를 사용한다.

완료 기준:

- input_code_space=bio_projected
- bio_projection_applied=true
- num_codebooks=6
- codons_per_codebook=3
- drop_operator=hamming_distance_axis_mask
- baseline과 여섯 drop 결과가 모두 존재
- paper_result_eligible=false와 projection_protocol_manifest_not_bound blocker를 임의로 제거하지 않음

24-base/L=4 checkpoint에 수행할 때는 6×4 slicing과 GC count 10–14가 되도록 explicit GC bounds를 사용한다.

    $PY scripts/codebook_drop_ablation_fast.py \
      --result_dir RESULT_24BASE \
      --num_codebooks 6 --bio_project \
      --gc_min_frac 0.416 --gc_max_frac 0.584 --max_run 3

### 6.4 Current champion slot intervention 재평가

기존 run_slot_intervention_bioproj.sh는 260717 구모델만 가리키므로 current unified model 재평가에는 사용하지 않는다.

Flickr25k:

    $PY scripts/slot_intervention_eval.py \
      --dataset Flickr25k \
      --ours_dir "result/260803+flickr25k_setting1_promptAblA_flickr_A_v4_uni002_P0refit_e4+bs+64+e+60+proj_lr+0.001" \
      --train_manifest dataset/Flickr25k/setting1/train.txt \
      --n_query 500 --db_subsample 0 --k 100 \
      --n_boot 1000 --seed 42 --bio_project \
      --out docs/newmodel_analysis/slot_intervention_flickr_v2_auditfix_20260804.json

MSCOCO:

    $PY scripts/slot_intervention_eval.py \
      --dataset MSCOCO \
      --ours_dir "result/260803+mscoco_setting1_promptAblA_mscoco_A_v5b_uni003_P0refit_e39+bs+64+e+60+proj_lr+0.001" \
      --n_query 500 --db_subsample 0 --k 100 \
      --n_boot 1000 --seed 42 --bio_project \
      --out docs/newmodel_analysis/slot_intervention_mscoco_v2_auditfix_20260804.json

NUS-WIDE:

    $PY scripts/slot_intervention_eval.py \
      --dataset NUSWIDE \
      --ours_dir "result/260803+nuswide_setting1_promptAblA_nuswide_A_v4_uni005_P0refit_e4+bs+64+e+60+proj_lr+0.001" \
      --train_manifest dataset/NUSWIDE/setting1/train_10500.txt \
      --n_query 500 --db_subsample 0 --k 100 \
      --n_boot 1000 --seed 42 --bio_project \
      --out docs/newmodel_analysis/slot_intervention_nuswide_v2_auditfix_20260804.json

corrected CIBHash control extraction이 준비되면 각 명령에 다음을 추가해 flat control arm을 다시 계산한다.

    --baseline_dirs CORRECTED_CIB_DNAEVAL_DIR \
    --baseline_names CIBHash

완료 기준:

- intervention_protocol_version=2
- ranking_tie_policy=hamming_then_database_index_stable
- validity_policy=exact_slot_common_valid_only
- intervention_dose_policy=exact_per_query_changed_slot_fraction_matched
- deployment_valid_intervention=true
- 모든 slot에서 paired_common_subset_n>0
- 모든 arm이 동일 query별 nonzero dose를 사용
- target segment 밖의 base는 불변
- paper_result_eligible=false를 유지하고, final paper 승격 전에 baseline manifest/checkpoint binder를 추가

### 6.5 FAIR/local cache 재생성

기존 cache를 in-place overwrite하지 말고 새 versioned out_dir을 사용한다. 일반 template은 다음과 같다.

    CUDA_VISIBLE_DEVICES=GPU_ID $PY extract_clip_local_crops.py \
      --donor_dir DONOR_FULL_IMAGE_CACHE \
      --out_dir NEW_TOKENONLY_CACHE_V2 \
      --pathlist_root dataset/DATASET \
      --pathlist_setting setting1 \
      --num_local_crops 8 \
      --local_crops_top_k 3 \
      --crop_scale_min 0.25 \
      --crop_scale_max 0.6 \
      --num_aug_views 2 \
      --seed SEED \
      --anchor_mode text_OR_image_global \
      --device cuda:0

기존 조건을 복구할 때의 seed와 anchor는 다음과 같다.

| 계열 | donor | seed | anchor |
|---|---|---:|---|
| CUB localL8K3 | cache/cub200_clip_v6bplus | 42 | text |
| CUB FAIRrankL8K3 | cache/cub200_clip_v6bplus | 44 | image_global |
| Flickr localL8K3 | cache/flickr25k_clip_v4plus_qwen3_tokens | 42 | text |
| Flickr FAIRrankL8K3 | cache/flickr25k_clip_v4plus_qwen3_tokens | 44 | image_global |
| MSCOCO FAIRrankL8K3 | cache/mscoco_clip_v5b | 45 | image_global |
| NUS-WIDE FAIRrank split cache | 해당 full-image split donor | 42 | image_global |

grid 조건은 crop_mode=grid, crop_grid_n=3 또는 4, crop_grid_frac=0.5를 추가한다. NUS-WIDE merged cache는 corrected train/test direct cache를 row identity 검증 후 다시 합쳐야 한다. 기존 merged cache를 그대로 복사하면 안 된다.

token-level text 파일이 필요한 계열은 fresh visual cache에 대해 해당 Qwen cache로 text-token overlay를 다시 만든다. 이전 out_dir에 남아 있던 text_tokens 파일을 묵시적으로 재사용하지 않는다.

cache 완료 기준:

- meta.json의 visual_global_source=donor_full_image
- donor_cache가 실제 distinct donor directory로 resolve
- visual_global_files가 main과 선언된 모든 aug view를 정확히 열거
- image_ids, text_part, has_text가 donor와 samefile
- visual_global main/aug가 donor와 samefile이며 array도 exact equality
- visual_tokens main/aug의 shape와 dtype이 metadata와 일치
- stale extra aug 또는 whitening sidecar가 없음
- 새 cache를 dataloader로 열 때 fail-closed 검증을 통과

그 뒤 새 cache path를 relevant official train launcher의 CACHE override로 넣어 representative FAIR/local cell과 whole-image control을 모두 다시 학습한다. 역사적 78개 exploratory result 전체를 재학습하지 않고, 논문에 유지할 비교만 predeclare한다.

### 6.6 Legacy OrthoHash MSCOCO

현재 main U0 OH와 무관한 legacy 결과다. 과거 MSCOCO OrthoHash 수치를 계속 인용할 때만 실행한다.

    CUDA_VISIBLE_DEVICES=GPU_ID $PY -m baseline.base_model \
      --method orthohash \
      --trial_name orthohash_mscoco_trainmodefix_20260804 \
      --dataset MSCOCO \
      --setting setting1 \
      --cache_dir ./cache/mscoco_siglip2 \
      --bit 36 \
      --batch_norm \
      --batch_size 64 \
      --max_epoch 60 \
      --eval_period 20 \
      --optimizer_name adam \
      --lr_scheduler none \
      --learning_rate 0.0001 \
      --seed 1 \
      --device cuda:0

하나의 corrected epoch-59 checkpoint로 native retrieval과 DNA-space extraction을 모두 다시 계산할 수 있다. 동일 config를 trial name만 바꿔 두 번 학습할 필요는 없다.

완료 기준:

- epoch 20 이후에도 BatchNorm running statistics가 계속 변한다.
- epoch 19, 39, 59 evaluation이 모두 새 checkpoint에서 생성된다.
- final extraction과 DNA-space 평가가 같은 corrected checkpoint digest를 기록한다.

## 7. 재실행하지 않아도 되는 범위

- A0/current champion의 정상 text-supervised, non-shared codebook 학습 경로는 A2/A4 수정의 영향을 받지 않는다.
- 2026-08-04 bioON queue의 정상 model path는 no-text/shared 분기를 쓰지 않으므로 해당 두 결함 때문에 재실행할 필요가 없다.
- Cimon, MLS3RDUH, GreedyHash, BiHalf, SDC, HHCH, CroVCA의 모델 수치는 CIBHash head 수정의 영향을 받지 않는다.
- source-SHA variant scoping 변경만으로 unaffected baseline을 다시 학습하지 않는다. 수정된 aggregator가 차단하는 실제 cell만 재실행한다.
- baseline/OH.py 기반 common-P0 OH matrix는 baseline/OrthoHash.py mode 결함의 영향을 받지 않는다.
- Slot intervention과 codebook-drop은 평가기 결함이므로 source checkpoint가 별도 학습 결함에 걸리지 않는 한 retraining이 아니라 reevaluation만 필요하다.
- FAIR/local result를 “crop token + crop-mean global”이라는 역사적 조건으로만 보존한다면 숫자를 다시 계산할 필요는 없다. 다만 token-only 또는 global-invariant claim에는 사용할 수 없다.

## 8. 문서와 논문 갱신 규칙

새 실험이 완료되기 전에는 다음 주장을 보류한다.

- “A2가 네 데이터셋 모두에서 일관되게 성능을 낮춘다”의 기존 exact 수치
- “A4가 separate per-slot codebook의 효과를 격리한다”의 기존 exact 수치와 quantizer-only 표현
- 과거 CIBHash 대비 margin과 CIBHash가 winner 또는 runner-up인지에 대한 결론
- 기존 slot-intervention gain, selectivity, Jaccard와 confidence interval
- raw codebook-drop으로부터 도출한 “배포 DNA에서 모든 slot이 기여한다”는 결론
- FAIR/local이 global slot을 고정한 single-delta라는 설명

문서별 처리:

- docs/PROJECT_LOG.md: 역사 기록은 삭제하지 않고 가장 위쪽 current state에 이 감사 문서 링크와 superseded 경고를 추가한다.
- docs/DRAFT_GROUNDEDDNA_PAPER_KO.md 및 수정본: stale 표와 figure를 새 결과가 나올 때까지 TODO/withheld로 바꾼다.
- docs/FINDINGS_PAPER_CLAIMS_2026-07-19.md: A2/A4, intervention, drop 관련 exact 수치에 superseded 표시를 붙인다.
- docs/comparison_Achampion_vs_baselines_2026-07-27.md: corrected CIBHash 36/48-bit three-seed aggregate가 나오기 전 CIB row와 CIB 기반 margin을 비운다.
- historical report: 원래 측정 기록은 유지하되 canonical 또는 current evidence로 재사용하지 않는다.

## 9. 최종 완료 체크리스트

### 코드

- [x] A2 forward-boundary text 제거
- [x] A2 text-init guard
- [x] A4 canonical shared bank
- [x] CIBHash official cached head와 60-epoch fixed-Adam protocol
- [x] OrthoHash epoch별 train mode
- [x] stable, valid, paired slot intervention
- [x] bio-aware general-geometry codebook-drop
- [x] token-only local-crop cache contract
- [x] variant-scoped source admission
- [x] 전체 회귀 테스트 및 compile

### 실험

- [ ] CIBHash 36-bit 4 datasets × 3 seeds
- [ ] CIBHash 48-bit 4 datasets × 3 seeds
- [ ] A2 4 datasets의 새 P0 selection
- [ ] A2 4 datasets의 새 P0 refit와 decoding
- [ ] A4 4 datasets의 새 P0 selection
- [ ] A4 4 datasets의 새 P0 refit와 decoding
- [ ] current champion 4 datasets codebook-drop bioproj
- [ ] current champion Flickr/MSCOCO/NUS slot intervention v2
- [ ] corrected CIBHash control을 포함한 intervention/decoding 재평가
- [ ] retained FAIR/local cache 재생성과 representative retraining
- [ ] 필요한 경우 legacy OrthoHash MSCOCO 재학습
- [ ] aggregate, paper table, draft와 figure 갱신

### Admission

- [ ] 새 artifact가 old result를 overwrite하지 않음
- [ ] source digest와 config가 manifest에 기록됨
- [ ] selection과 terminal test가 protocol에 맞게 분리됨
- [ ] bio projection validity 100%
- [ ] 모든 reported mean±std가 seeds 42/43/44의 실제 aggregate
- [ ] diagnostic-only와 paper-table-eligible 수치를 혼합하지 않음
- [ ] paper_result_eligible=false blocker를 수동 편집으로 제거하지 않음

## 10. 변경 파일 인벤토리

### 구현 및 protocol

- [baseline/CIBHash.py](../baseline/CIBHash.py)
- [baseline/OrthoHash.py](../baseline/OrthoHash.py)
- [dataloaders.py](../dataloaders.py)
- [extract_clip_local_crops.py](../extract_clip_local_crops.py)
- [model_siglip2.py](../model_siglip2.py)
- [train_siglip2.py](../train_siglip2.py)
- [aggregate_baseline_p0_matrix.py](../scripts/aggregate_baseline_p0_matrix.py)
- [codebook_drop_ablation.py](../scripts/codebook_drop_ablation.py)
- [codebook_drop_ablation_fast.py](../scripts/codebook_drop_ablation_fast.py)
- [run_baseline_p0_matrix.py](../scripts/run_baseline_p0_matrix.py)
- [run_modern_baseline_p0.py](../scripts/run_modern_baseline_p0.py)
- [slot_intervention_eval.py](../scripts/slot_intervention_eval.py)

### 수정된 기존 테스트

- [test_baseline_matrix_aggregation.py](../tests/test_baseline_matrix_aggregation.py)
- [test_crh_supervised_matrix.py](../tests/test_crh_supervised_matrix.py)
- [test_modern_driver_protocol.py](../tests/test_modern_driver_protocol.py)

### 신규 회귀 테스트

- [test_ablation_integrity.py](../tests/test_ablation_integrity.py)
- [test_cibhash_source_fidelity.py](../tests/test_cibhash_source_fidelity.py)
- [test_codebook_drop_protocol.py](../tests/test_codebook_drop_protocol.py)
- [test_local_crop_cache_contract.py](../tests/test_local_crop_cache_contract.py)
- [test_orthohash_training_mode.py](../tests/test_orthohash_training_mode.py)
- [test_slot_intervention_eval.py](../tests/test_slot_intervention_eval.py)

---

이 문서의 체크박스는 실험이 실제로 완료되고 artifact와 manifest를 검증한 뒤에만 갱신한다. 프로세스가 시작되었거나 checkpoint만 생성된 상태는 완료로 보지 않는다.
