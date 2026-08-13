# 프로토콜 결정 — 2026-08-13

`docs/EXPERIMENT_PROTOCOL_AUDIT_2026-08-13.md`가 제기한 다섯 개 미결 사항에 대한
사용자 결정이다. Phase 1 이후의 모든 구현과 재실행은 이 문서를 따른다.

---

## D1 (F02) — selection은 test-independent로. **전량 재학습**

> **[개정 2026-08-14, D6 참조]** baseline 쪽 규약이 바뀌었다. baseline은 더 이상
> validation selection + scratch refit을 하지 않고 **원저자 고정 epoch의 마지막
> checkpoint**를 쓴다. 아래 표에서 **ours에만 해당**하는 것으로 읽는다.


| 항목 | 값 |
|---|---|
| validation | designated train의 **10 %**, `val_seed=42` |
| selection 중 공식 test | **로드하지 않음** |
| N 후보 grid | **모든 데이터셋 공통 `{4, 9, 19, 39}`** |
| selection metric | baseline과 동일한 **raw base-Hamming mAP@R 하나** |
| 동률 처리 | **가장 작은 N** |
| N 재사용 | seed 42 validation에서 dataset별로 한 번 정하고 **seeds 42/43/44에 그대로** |
| 최종 학습 | 선택된 N에서 **full designated train scratch refit** |

기존 full-train checkpoint는 validation subset까지 학습에 썼으므로 train-only
selection을 소급 적용할 수 없다. **ours main은 재학습해야 한다.**

이것만으로 해결되지 않는 것: baseline 108셀의 eligibility(F04)와 CIMON 구현
결함(F05)은 **별도 blocker로 남는다.**

## D2 (F03) — horizon을 **탐색 단계에서만** 분리. 최종은 셋을 함께 움직인다

압축 cosine 자체가 부당한 것은 아니다. 문제는 N 하나가 학습 길이 · LR horizon ·
ε horizon을 동시에 바꾸고 논문 서술이 실제 코드와 달랐다는 점이다. 다만 세 값을
독립 축으로 두면 탐색 그리드가 감당할 수 없이 커지므로 **두 단계로 나눈다.**

### 용어 — off-by-one

`N`은 **0-based 정지 epoch**이다. `--stop_after_epoch N`은 epoch `0..N`을 돌므로
학습 길이는 **N+1 epoch**이다. horizon은 길이 단위이므로 `N+1`을 쓴다.

### 탐색 단계 — LR을 고정해 그리드를 단순화

```
-e 60  --stop_after_epoch N  --lr_schedule_horizon 60  --sinkhorn_schedule_horizon N+1
```

| 항목 | 값 |
|---|---|
| training stop | **N** |
| `lr_schedule_horizon` | **60 고정** |
| `sinkhorn_schedule_horizon` | **N + 1** |
| final extraction의 inference epoch | **N** |

모든 N 후보가 **동일한 LR schedule prefix**를 공유하므로, N 비교가 학습 길이
비교가 된다. 세 값은 manifest에 각각 기록한다.

### 최종 단계 — 확정된 N에 세 값을 모두 맞춘다

```
-e N+1  --stop_after_epoch N
```

`--lr_schedule_horizon`과 `--sinkhorn_schedule_horizon`을 지정하지 않으면 둘 다
`--epoch`로 fallback하므로, 위 한 줄이 **세 값을 전부 N+1로 묶는다.** 이것이
사용자가 의도한 최종 구성이며, 별도 플래그가 필요 없다.

### ⚠️ 명시해야 할 한계

**탐색 config와 최종 config는 서로 다른 모델을 만든다.** 탐색에서 N이 최적이었다는
것은 "60-epoch cosine의 앞 N+1 epoch"에서의 최적이고, 최종 모델은 "N+1 epoch에
압축된 cosine"이다. LR 궤적이 다르므로 **탐색 최적 N이 최종 config에서도 최적이라는
보장은 없다.**

논문에는 이 절차를 그대로 쓴다: *"N은 고정 LR prefix 하에서 선택했고, 최종 모델은
그 N에 맞춘 schedule로 재학습했다."* 선택 절차와 최종 모델이 다르다는 사실을
숨기지 않는다.

## D3 (F07) — GC는 **실제 40–60 % inclusive**를 중앙 policy로

| L | GC 정수 범위 | homopolymer |
|---:|---|---|
| 15 | **[6, 9]** | run ≤ 3 |
| 20 | **[8, 12]** | run ≤ 3 |

fraction을 evaluator마다 다시 반올림하지 않는다. 중앙 policy가
`(gc_min_count, gc_max_count, max_run, policy_version)`을 반환하고, **manifest에
정수 범위와 policy version을 기록**한다. ours · baseline · native-DNA ·
held-out · ablation · aggregator · 논문이 모두 같은 값을 쓴다.

## D4 (F16) — MSCOCO decode 실패 16건: **유지 + 민감도 분석**

16건은 **전부 MSCOCO DB에만** 있고 train/query에는 없다. 따라서 제외 실험은
재학습이 아니라 **DB 필터링 후 재평가**만으로 된다.

- **주 protocol**: 16건 유지, zero-feature 처리와 **16개 ID 공개**
- **보조 분석**: 16건 제외 후 주요 수치 전량 재계산
  - 4자리 표와 결론이 유지되면 → 유지 방침을 정당화
  - 유의미하게 달라지면 → **제외한 split을 canonical로 전환**
- 원본이 단순 decoder 문제로 복구 가능하면 복구가 더 낫다. 이 경우에도 DB-only라
  **재학습은 불필요**하다.

## D5 (20-base) — **28셀, seed 42 단일, diagnostic**

| 항목 | 값 |
|---|---|
| 규모 | **ours 4셀 + 사전 지정 `U0` 6 methods × 4 datasets = 28셀** |
| seed | **42 단일** |
| 지위 | **diagnostic**. mean±std나 통계적 우열 주장 없음 |
| 최강 표현 | "best baseline"이 아니라 **"best among evaluated six"** |
| TeX | 실행하지 않은 3개 baseline 행은 **제거하거나 명시적 미측정 표시** |

**셀 수 정정.** 2026-08-13에 시작된 117셀은 28셀의 3-seed 판본이 **아니다**:
`U0 9×4×3 = 108` + `U2 3×3 = 9` = 117이며 **ours가 빠져 있다.** 완전한 U0+ours
3-seed는 `108 + 12 = 120`, U2까지 포함하면 **129셀**이다.

---

## 결정이 만드는 재실행 규모

| 대상 | 재학습 | 이유 |
|---|---|---|
| ours main (4 dataset × N grid 4 + 3 seed) | **필요** | D1 selection, D2 horizon |
| baseline `U0` 30-bit 108셀 | **필요** | F04 provenance, F05 CIMON |
| native-DNA 15-base 48셀 | 필요 | F18 |
| 20-base 28셀 | 신규 | D5 |
| MSCOCO 16건 민감도 | 재평가만 | D4 |


---

## D6 (F02 개정) — baseline은 **원저자 고정 epoch**, ours만 train-only selection

### 결정

baseline 9종은 각 원논문/공식 release가 지정한 epoch 수까지 full designated
train으로 학습하고 **마지막 epoch checkpoint만** 사용한다. validation selection도
scratch refit도 하지 않는다. test는 최종 평가에 한 번만 쓴다.

이렇게 하면 baseline 쪽 checkpoint-selection leakage가 원천적으로 사라지고,
2단계(stage1 선택 → stage2 재학습) 대비 계산량도 준다.

ours는 D1대로 train-only validation에서 N을 고른다. 이 **비대칭은 숨기지 않고
본문에 명시**한다.

- baseline: 원래 backbone·데이터 조건에서 정해진 epoch를 그대로 적용
- ours: 현재 frozen-CLIP adaptation에 맞춰 train-only validation으로 N 선택

주장 문구는 다음으로 제한한다.

> All baselines were trained to their pre-specified author-recommended horizons
> under the shared frozen-cache adaptation.

리뷰어 방어를 위해 **baseline validation-selected 소규모 민감도 분석**을 부표로
같이 싣는다.

### 고정 epoch만으로는 부족하다 — 함께 보존해야 하는 것

optimizer / learning rate / **LR scheduler와 그 horizon** / batch size / epoch당
update 수 / augmentation / `drop_last` / backbone frozen 여부 / bit별 설정 /
dataset별 설정.

### 원전 대조 결과 (2026-08-14)

`scripts/run_modern_baseline_p0.py:90-166`의 horizon을 원전과 대조했다.

| method | horizon | 원전 | 대조 결과 |
|---|---|---|---|
| GreedyHash-UGH | 60 | `ssppp/GreedyHash` `unsupervised_vgg.py:15-23` | **정확**. `num_epochs=60`, batch 32, lr 1e-4, SGD(0.9, 5e-4), `adjust_learning_rate` 호출이 주석 처리되어 **LR decay 없음** — 로컬 `lr_scheduler='none'`과 일치. `if encode_length == 16: num_epochs = 300`이므로 30-bit는 16-bit가 아니어서 60 분기가 그대로 적용된다 |
| Bi-half CIFAR-10 | 300 | `liyunqianggyn/…` `ImageHashing/Cifar10_I.py:14-16` | horizon **정확**하나 **scheduler 불일치**(아래) |
| Bi-half Flickr25k | 100 | `ImageHashing/Flickr25k.py:11-13` | **정확** (`epoch_lr_decrease=60`) |
| Bi-half MS-COCO | 150 | `ImageHashing/Mscoco.py:16-18` | **정확** (`epoch_lr_decrease=60`) |
| Bi-half NUS-WIDE | 100 | 공식 repo에 NUS-WIDE trainer 없음 | **adaptation**. Flickr profile 적용, manifest에 그렇게 표기됨 |
| CIMON | 150 | `luoxiao12/CIMON` | F05에서 objective를 원전과 일치시킴 |
| CIBHash | 60 | 공식 설정 | `tests/test_cibhash_source_fidelity.py`가 이미 고정 |
| CroVCA | 5 | 공개 probing 설정 | cosine이 `schedule_horizon * len(loader)`에서 유도되어 horizon과 자기정합 |
| HHCH | 80 | release 기본값(논문 미명시) | `1.0 if epoch < 30 else 0.9 ** (epoch // 20)` — 절대 epoch 경계이므로 horizon 80에서 release 동작과 일치 |
| MLS3RDUH | 150 | 논문/공개 구현 | scheduler 없음 |
| SDC | 100 | release/paper 공통 | `step_size=80` 단일 decay |
| OH | 200 | CIFAR-10 공개 구현 | epoch 정의 차이가 이미 선언되어 있음(아래) |

### 발견된 결함 1 — Bi-half CIFAR-10의 LR decay 주기 (수정 필요)

`baseline/BiHalf.py:178-180`의 `_get_config_dict_for_dataset`는 `del dataset`으로
데이터셋을 버리고 기본 config를 그대로 돌려준다. 따라서 `step_size`는 항상 60이다.
`baseline/base_model.py:1013-1017`의 `StepLR`은 `schedule_horizon`을 보지 않는다.

| | 공식 `Cifar10_I.py` | 현재 로컬 |
|---|---|---|
| epochs | 300 | 300 |
| `epoch_lr_decrease` / `step_size` | **120** | **60** |
| decay 시점 | 120, 240 | 60, 120, 180, 240 |
| 최종 LR | 1e-4 × 0.1² = **1e-6** | 1e-4 × 0.1⁴ = **1e-8** |

**최종 LR이 100배 차이 난다.** 이전 프로토콜에서는 E\*가 대체로 앞쪽이라 뒤쪽 decay가
결과에 닿지 않는 경우가 많았지만, **고정 epoch + 마지막 checkpoint 규약에서는 이
차이가 곧 보고 수치**가 된다. Bi-half는 `step_size`를 dataset별로 주어야 한다
(CIFAR-10 120, Flickr25k·MS-COCO·NUS-WIDE 60).

### 발견된 결함 아님 2 — OH의 epoch 정의 (선언되어 있음)

`baseline/OH.py:407-410`이 이미 기록하고 있다.

> release consumes 50000 repeated CIFAR rows per epoch; matched P0 consumes one
> finite pass over its designated train split

같은 "200 epoch"라도 optimizer update 수가 다르다. 이건 은폐가 아니라 선언된
adaptation이므로, 표와 본문에 **update 수 기준으로도 명시**한다.

### 남은 확인 항목

- SDC `step_size=80`을 release와 대조
- MLS3RDUH·CIMON의 LR decay 유무를 원전에서 재확인
- 각 method의 `drop_last`, augmentation, frozen 여부를 manifest 필드로 승격
- bit별 설정: 30-bit는 어느 원전 설정에도 없으므로 method별 adaptation 규칙을
  **실행 전에** 선언한다 (GreedyHash는 16-bit가 아니므로 60, 나머지는 bit 무관)

### 함께 바뀌어야 하는 것

- `scripts/run_baseline_p0_matrix.py` / `run_modern_baseline_p0.py`의 2-stage
  규약 → 1-stage 고정 epoch
- `scripts/baseline_val_select_p0.py`는 **민감도 분석 전용**으로 격하
- aggregator의 selection metric 필드는 ours에만 적용됨을 metadata에 명시
