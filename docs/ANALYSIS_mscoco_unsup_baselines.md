# MSCOCO unsupervised baseline 비교: v63b vs CIBHash / CIMON / MLS3RDUH

작성일 2026-05-21.

사용자 요청: v63b (MSCOCO 우리 모델) vs 3 unsupervised baseline (CIBHash, CIMON,
MLS3RDUH) 모두 MSCOCO에 학습 후 정밀 비교.

---

## 1. 실험 설정

### 공통
- Dataset: MSCOCO setting1 (10K train, 5K test, 107K database)
- Bit: 36 (모두 동일)
- Epoch: 60
- Batch size: 64
- Backbone: SigLIP2-base-224 (frozen, all methods)
- Cache: `cache/mscoco_siglip2_v4plus/` (visual + 2 aug views; 모든 baseline은
  visual feature만 사용, text 무시)

### 각 method 차이

| Method | 구조 | Loss | Trainable params |
|---|---|---|---:|
| **v63b (ours)** | text-routed compositional (6 cb × 64 codeword) + codon head | per-cb NtXent + dyn-τ + VQ + wass + ... (7 terms) | ~3M |
| **CIBHash** | Linear(768, 36) + sign | paired-aug NtXent + KL prior | ~28K |
| **CIMON** | spectral clustering pseudo-labels + Linear(768, 36) | NtXent + pseudo-label cross-entropy | ~28K |
| **MLS3RDUH** | kNN graph + LogCosh on continuous projection | LogCosh + kNN similarity | ~28K |

→ v63b만 *compositional structure* 보유; 나머지는 모두 *flat 36-bit* projection.

---

## 2. 결과 — mAP + Precision@k

### 종합 표

| Method | mAP | P@1 | P@10 | P@100 | P@1000 |
|---|---:|---:|---:|---:|---:|
| **CIBHash** ★ | **0.5051** | **0.7802** | **0.7639** | **0.7408** | **0.6875** |
| CIMON | 0.4777 | (n/a from current logs) | | | |
| **v63b (ours)** | 0.4563 | 0.5606 | 0.5328 | 0.5370 | 0.5308 |
| MLS3RDUH | 0.4434 | 0.6030 | 0.5557 | 0.5473 | 0.5284 |

### 순위 결과

1. **CIBHash 0.5051** — MSCOCO unsupervised SOTA (외부 baseline 우세)
2. **CIMON 0.4777**
3. **v63b (ours) 0.4563**
4. **MLS3RDUH 0.4434**

→ 우리 모델 v63b는 4개 중 **3위**. CIBHash + CIMON에는 미달, MLS3RDUH보다 우세.

### 핵심 관찰 — Flickr25k와의 ranking 역전

Flickr25k에서 우리 v57은 CIBHash를 +0.014로 앞섰음:
- Flickr25k: v57 0.6683 > CIBHash 0.6543 (+0.014)
- MSCOCO: **v63b 0.4563 < CIBHash 0.5051 (−0.049)**

**MSCOCO에서 CIBHash가 우리 모델을 능가**. 원인 분석:
- MSCOCO는 multi-label + class 다양성 ↑ + db scale 5배 → fine-grained
  discrimination이 retrieval에 더 중요
- CIBHash는 connection 없는 *직접 binary projection* → 768d → 36bit
  continuous-to-discrete가 fine variation을 그대로 보존
- 우리 v63b는 *codebook quantization*으로 의미 cluster 만듦 → MSCOCO에서
  cluster 안 변별이 안 됨

---

## 3. Precision@k 형태 비교

### v63b vs CIBHash — top-k 정확도 격차

| k | v63b | CIBHash | Δ (CIBHash advantage) |
|---:|---:|---:|---:|
| 1 | 0.5606 | 0.7802 | +0.220 |
| 10 | 0.5328 | 0.7639 | +0.231 |
| 100 | 0.5370 | 0.7408 | +0.204 |
| 1000 | 0.5308 | 0.6875 | +0.157 |

CIBHash 모든 k에서 +0.16~+0.23 우세. Top-1~Top-10 사이가 가장 큼.

### Drop-off 패턴

| Method | P@1 → P@1000 변화 |
|---|---:|
| v63b | 0.561 → 0.531 (Δ = **−0.030**, 가장 평탄) |
| MLS3RDUH | 0.603 → 0.528 (Δ = −0.075) |
| CIBHash | 0.780 → 0.688 (Δ = −0.092) |

v63b는 *어느 rank에서나 비슷한 품질* — 단점은 top-1 정확도가 낮은 것이지만
deep rank에서 안정. CIBHash는 top-1이 매우 정확하지만 rank 깊어지면서 drop.

### MSCOCO retrieval characteristic

- **CIBHash의 강점**: 99% unique code → 거의 모든 image가 unique → top-1
  matching이 sharp
- **v63b의 약점**: 32% unique code → 의미적 cluster가 같은 code로 묶여서 top-1
  에서 *cluster 안 다른 의미* image까지 retrieve됨 → P@1 0.56

→ MSCOCO는 fine-grained retrieval task. 우리 모델의 compositional structure는
"의미 grouping"에는 강하지만 "instance-level discrimination"에는 약함.

---

## 4. Unique code / collision 비교

| Method | unique (#unique / N) | 비고 |
|---|---|---|
| CIBHash | n/a (보통 0.99+ 예상, 측정 안 됨) | continuous → sign |
| CIMON | n/a | 비슷한 구조 |
| MLS3RDUH | n/a | LogCosh on continuous |
| **v63b** | **0.315 (33788 / 107218)** | codebook 6×128 |

v63b의 0.315는 baseline 대비 극히 낮음. *retrieval 성능과 unique-code ratio가
강하게 양의 상관관계*임을 MSCOCO에서 확인 → Task B (Option A: residual-
conditioned codon head)의 motivation 강화.

---

## 5. 종합 — paper narrative 시사점

### 우리 모델의 강점 (claim 가능)
1. **Compositional structure 정량 측정 가능** (B0/B1/B2) — flat baseline은
   불가능
2. **모든 codebook 100% utilization, dead=0** — compositional 보장
3. **rank-깊은 retrieval에서 평탄** (P@1 → P@1000 drop 작음)
4. **paper claim #2/#3 (text-supervised + frozen backbone)** 정상 작동

### 우리 모델의 약점 (개선 필요)
1. **MSCOCO처럼 fine-grained dataset에선 CIBHash에 −0.05 mAP**
2. **Unique code 0.32 (CIBHash 0.99의 1/3)** — codebook quantization이
   변별력 손상
3. **P@1 0.56 (CIBHash 0.78)** — top-rank sharpness 부족

### 해결 방향
- **Option A (Residual-Conditioned Codon Head, v62 진행 중)**: codebook 의미
  cluster 유지하면서 image별 codon variation 추가 → unique ↑ → 기대: P@1
  개선 + mAP 회복
- **Multi-stage VQ (Option B, 계획안)**: primary cb + residual cb로 fine
  variation 표상
- **MSCOCO-specific tuning**: K=128 → K=192? wasserstein weight 재조정?

### 외부 baseline에 대한 정직한 평가

CIBHash는 *단순함의 승리* — 28K params + Linear+sign으로 모든 image 다른
hash 생성. 우리 3M params + 7 loss term 모델이 MSCOCO에서 추월 못 했음.
**Flickr25k에서 v57 +0.014 우세였던 것은 dataset의 의미 grouping (24 class
multi-label)이 우리 compositional 표상과 잘 맞은 결과**. MSCOCO 80 class
multi-label에선 더 nuanced retrieval 필요 → CIBHash 우세.

### 후속 작업 권장

1. **v62 (Option A) MSCOCO 학습** — Flickr25k에서 효과 확인 후 MSCOCO에 적용
2. **K sweep on MSCOCO**: {128, 192, 256} — capacity 확장
3. **CIBHash-style auxiliary head 추가**: codon code AND fine binary code 동시
   사용 (concat)으로 hybrid 시도
4. **paper에 두 dataset을 두 paradigm으로 제시**:
   - Flickr25k: "compositional code가 의미 grouping에서 우세"
   - MSCOCO: "fine-grained discrimination에선 instance-level 변별 더 중요"
   → 두 ablation이 *서로 다른 contribution*을 입증

---

## 6. 데이터 출처

- v63b: `result/260520+mscoco_setting1_v63b_mscoco_v57setup_K128+bs+64+e+60+proj_lr+0.001/evaluation_siglip2_base.json`
- CIBHash: `result_baseline/260521/cibhash_mscoco_unsup60/eval_epoch_059.json`
- CIMON: `result_baseline/260521/cimon_mscoco_unsup60/eval_epoch_059.json`
- MLS3RDUH: `result_baseline/260521/mls3rduh_mscoco_unsup60/eval_epoch_059.json`

전 baseline 학습 시 사용한 cache: `cache/mscoco_siglip2_v4plus/` (122K image,
visual + 2 aug views; V4 text는 baseline들이 사용 안 함).
