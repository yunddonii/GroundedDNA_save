# 프로토콜 결정 — 2026-08-13

`docs/EXPERIMENT_PROTOCOL_AUDIT_2026-08-13.md`가 제기한 다섯 개 미결 사항에 대한
사용자 결정이다. Phase 1 이후의 모든 구현과 재실행은 이 문서를 따른다.

---

## D1 (F02) — selection은 test-independent로. **전량 재학습**

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
